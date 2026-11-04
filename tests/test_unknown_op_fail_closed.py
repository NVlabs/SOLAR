# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issue #50: an op with no handler and no recognised family fails closed.

``EinsumAnalyzer.get_einsum_op`` used to lower ANY unrecognised op to the
generic ``ElementwiseSelectEinsumOp`` copy — zero MACs, ``is_einsum_supportable:
true`` — so real work (matmul, conv, reduction) vanished silently; the name
normaliser also missed ``mm.out`` / ``aten.mm.default`` / ``bmm.out`` /
``addmm.out`` (``mm`` is tested by equality, ``matmul`` by substring).

Contract pinned here:
* ``aten.<op>.<overload>`` and ``<op>.out`` / ``<op>.default`` spellings are
  normalised before routing and reach the real handlers (billed MACs);
* an op with no handler whose family is not recognised converts with
  ``is_einsum_supportable: false`` and ``unrecognised_op: <recorded name>``;
  the analyzer reports it ``status: incomplete`` / ``incomplete_reason:
  op_unrecognised`` with 0 MACs and 0 other_ops, the total carries it
  (INTEG-9) and the perf prediction is qualified as zeroed work;
* genuinely elementwise / select / view / cast ops keep the copy lowering
  only when their family is recognised (``where``, casts, scatters, ...).
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.einsum.op_families import (
    RECOGNISED_COPY_FAMILIES,
    UnrecognisedOperationError,
    identity_interpolate_evidence,
    interpolate_evidence_record,
    interpolate_record_refusal,
    normalise_overload_name,
)
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.perf.perf_model import EinsumGraphPerfModel

M, K, N = 64, 128, 32
MM_MACS = M * K * N


# ---------------------------------------------------------------------------
# name normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling, expected", [
    ("mm.out", "mm"), ("aten.mm.default", "mm"), ("torch.ops.aten.mm.default", "mm"),
    ("bmm.out", "bmm"), ("addmm.out", "addmm"), ("aten.addmm.default", "addmm"),
    ("aten.add.Tensor", "add"), ("aten.sum.dim_IntList", "sum"), ("MM.OUT", "mm"),
    ("Model.max", "model.max"), ("clamp_4", "clamp_4"), ("where", "where"),   # lower-cased, otherwise untouched
])
def test_overload_spellings_are_stripped(spelling, expected):
    assert normalise_overload_name(spelling) == expected


@pytest.mark.parametrize("spelling, family", [
    ("mm.out", "matmul"), ("aten.mm.default", "matmul"), ("bmm.out", "bmm"),
    ("addmm.out", "addmm"), ("aten.bmm.out", "bmm"), ("matmul.out", "matmul"),
])
def test_out_and_aten_spellings_route_to_the_real_handler(spelling, family):
    analyzer = EinsumAnalyzer()
    assert analyzer._get_operation_from_name(spelling) == family
    assert analyzer._registry.has_handler(family)


def test_mm_out_is_billed_as_a_matmul():
    op = EinsumAnalyzer().get_einsum_op("mm.out", TensorShapes(inputs=[[M, K], [K, N]], outputs=[[M, N]]))
    assert op.is_real_einsum and op.is_einsum_supportable
    assert op.get_compute_cost(TensorShapes(inputs=[[M, K], [K, N]], outputs=[[M, N]])) == MM_MACS


# ---------------------------------------------------------------------------
# fail-closed binder
# ---------------------------------------------------------------------------

def test_opaque_name_raises_instead_of_lowering_to_a_copy():
    with pytest.raises(UnrecognisedOperationError) as info:
        EinsumAnalyzer().get_einsum_op("frobnicate_rows", TensorShapes(inputs=[[M, K]], outputs=[[M, K]]))
    assert info.value.recorded_name == "frobnicate_rows"
    assert "frobnicate_rows" in str(info.value)


@pytest.mark.parametrize("name", ["where", "masked_fill", "__invert__", "__and__", "logical_or", "one_hot", "long",
                                  "cpu", "repeat_interleave", "index_add_", "unbind", "zeros_like", "unresolved_op"])
def test_recognised_copy_families_keep_the_select_lowering(name):
    shapes = TensorShapes(inputs=[[M, K]], outputs=[[M, K]])
    op = EinsumAnalyzer().get_einsum_op(name, shapes)
    assert op.is_real_einsum is False and op.elementwise_op == "copy" and op.reduction_op == "none"
    assert op.is_einsum_supportable
    assert name in RECOGNISED_COPY_FAMILIES or name == "unresolved_op"


@pytest.mark.parametrize("name", ["sort", "bincount", "interpolate", "argsort", "topk", "nonzero", "unique", "frobnicate"])
def test_work_bearing_names_without_a_handler_are_not_recognised(name):
    assert name not in RECOGNISED_COPY_FAMILIES
    with pytest.raises(UnrecognisedOperationError):
        EinsumAnalyzer().get_einsum_op(name, TensorShapes(inputs=[[M, K]], outputs=[[M, K]]))


def test_handled_elementwise_ops_are_unchanged():
    shapes = TensorShapes(inputs=[[M, K], [M, K]], outputs=[[M, K]])
    op = EinsumAnalyzer().get_einsum_op("add", shapes)
    assert op.elementwise_op == "add" and op.reduction_op == "none" and op.is_einsum_supportable


# ---------------------------------------------------------------------------
# convert -> analyze -> perf controls on a recorded graph
# ---------------------------------------------------------------------------

def _tensor_node(name, shape, dtype, producers, consumers):
    return {"type": name.split(".")[-1].rsplit("_", 1)[0] if "-tensor" in name else name, "node_class": "TensorNode",
            "input_shapes": [list(shape)] if producers else [], "output_shapes": [list(shape)],
            "input_dtypes": [dtype] if producers else [], "output_dtypes": [dtype],
            "input_types": ["input"] if producers else [], "output_types": ["output"] * max(1, len(consumers)),
            "module_args": {"hierarchical_name": f"Model.{name}"},
            "connections": {"inputs": list(producers), "outputs": list(consumers)}}


def _recorded_graph(op_type, in_shapes, out_shape, function_name=None):
    """A capture of one two-input op fed by two model inputs, like torchview records it."""
    op_id = f"Model.{op_type}"
    layers = {}
    for i, shape in enumerate(in_shapes):
        aux = "Model.auxiliary-tensor" + ("" if i == 0 else f"_{i}")
        layers[aux] = {**_tensor_node("auxiliary-tensor", shape, "torch.float32", [], [op_id]), "type": "auxiliary-tensor"}
    layers[op_id] = {
        "type": op_type, "node_class": "FunctionNode",
        "input_shapes": [list(s) for s in in_shapes], "output_shapes": [list(out_shape)],
        "input_dtypes": ["torch.float32"] * len(in_shapes), "output_dtypes": ["torch.float32"], "output_dtypes_recorded": True,
        "input_types": ["input"] * len(in_shapes), "output_types": ["output"],
        "module_args": {"function_name": function_name or op_type,
                        "raw_attributes": "[[" + ", ".join(f"Tensor(shape={tuple(s)}, dtype=torch.float32)" for s in in_shapes) + "], {}]",
                        "hierarchical_name": op_id},
        "connections": {"inputs": list(layers), "outputs": ["Model.output-tensor"]},
    }
    layers["Model.output-tensor"] = {**_tensor_node("output-tensor", out_shape, "torch.float32", [op_id], []), "type": "output-tensor"}
    return {"model_name": "issue50_control", "layers": layers}


def _run(tmp_path, graph, arch="H100_PCIe"):
    (tmp_path / "graph").mkdir()
    gpath = tmp_path / "graph" / "pytorch_graph.yaml"
    gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    einsum = PyTorchToEinsum().convert(str(gpath), str(tmp_path / "einsum"), copy_graph=False, enable_rename=False)
    assert einsum is not None
    analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum" / "einsum_graph.yaml"), str(tmp_path / "analysis"),
                                                   precision="fp32", copy_graph=False)
    assert analysis is not None
    perf = EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"),
                                          arch_config=arch, precision="fp32")
    assert perf is not None
    return einsum, analysis, perf


@pytest.mark.parametrize("op_type, in_shapes, out_shape, macs", [
    ("mm.out", [[M, K], [K, N]], [M, N], MM_MACS),
    ("aten.mm.default", [[M, K], [K, N]], [M, N], MM_MACS),
    ("bmm.out", [[4, M, K], [4, K, N]], [4, M, N], 4 * MM_MACS),
], ids=["mm.out", "aten.mm.default", "bmm.out"])
def test_overload_spellings_are_billed_through_the_pipeline(tmp_path, op_type, in_shapes, out_shape, macs):
    einsum, analysis, perf = _run(tmp_path, _recorded_graph(op_type, in_shapes, out_shape))
    layer = einsum["layers"][f"Model.{op_type}"]
    assert layer["is_real_einsum"] and layer["is_einsum_supportable"] and "unrecognised_op" not in layer
    a = analysis["layers"][f"Model.{op_type}"]
    assert a["macs"] == macs and a.get("status") != "incomplete"
    assert analysis["total"]["macs"] == macs and analysis["total"].get("status") != "incomplete"
    assert perf["workload"].get("status") != "incomplete"


def test_addmm_out_is_billed_through_the_pipeline(tmp_path):
    graph = _recorded_graph("addmm.out", [[N], [M, K], [K, N]], [M, N])
    einsum, analysis, perf = _run(tmp_path, graph)
    a = analysis["layers"]["Model.addmm.out"]
    assert a["macs"] == MM_MACS and a.get("status") != "incomplete"
    assert analysis["total"]["macs"] == MM_MACS


def test_opaque_op_is_incomplete_with_no_macs_through_the_pipeline(tmp_path):
    einsum, analysis, perf = _run(tmp_path, _recorded_graph("frobnicate_rows", [[M, K], [K, N]], [M, N]))
    layer = einsum["layers"]["Model.frobnicate_rows"]
    assert layer["is_einsum_supportable"] is False and layer["unrecognised_op"] == "frobnicate_rows"
    assert layer["is_real_einsum"] is False and layer["elementwise_op"] == "copy" and layer["reduction_op"] == "none"
    assert layer["einsum_equation"]                      # a rank-preserving placeholder keeps the output produced
    a = analysis["layers"]["Model.frobnicate_rows"]
    assert a["status"] == "incomplete" and a["incomplete_reason"] == "op_unrecognised"
    assert a["unrecognised_op"] == "frobnicate_rows"
    assert a["macs"] == 0 and a["other_ops"] == 0 and a["flops"] == 0
    # INTEG-9: the total and the perf qualification carry it
    assert analysis["total"]["status"] == "incomplete"
    assert analysis["total"]["incomplete_layers"] == {"Model.frobnicate_rows": "op_unrecognised"}
    assert analysis["total"]["macs"] == 0
    w = perf["workload"]
    assert w["status"] == "incomplete" and w["incomplete_layers"] == {"Model.frobnicate_rows": "op_unrecognised"}
    assert w["prediction_qualification"].startswith("partial") and "op_unrecognised" in w["prediction_qualification"]
    assert "excluded" in w["prediction_qualification"]


def test_elementwise_control_is_unchanged_through_the_pipeline(tmp_path):
    einsum, analysis, perf = _run(tmp_path, _recorded_graph("add", [[M, K], [M, K]], [M, K]))
    layer = einsum["layers"]["Model.add"]
    assert layer["is_einsum_supportable"] and "unrecognised_op" not in layer and layer["elementwise_op"] == "add"
    a = analysis["layers"]["Model.add"]
    assert a["macs"] == 0 and a["other_ops"] == M * K and a.get("status") != "incomplete"
    assert analysis["total"].get("status") != "incomplete" and perf["workload"].get("status") != "incomplete"


def test_where_control_keeps_the_select_lowering_through_the_pipeline(tmp_path):
    einsum, analysis, perf = _run(tmp_path, _recorded_graph("where", [[M, K], [M, K], [M, K]], [M, K]))
    layer = einsum["layers"]["Model.where"]
    assert layer["is_einsum_supportable"] and "unrecognised_op" not in layer
    assert analysis["layers"]["Model.where"].get("status") != "incomplete"


# ---------------------------------------------------------------------------
# Round-2 controls (an earlier review, checks-4aac214b/test_independent.py, verbatim bodies)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('spelling,canonical', [
    ('aten.add.Tensor', 'add'), ('torch.ops.aten.add.Tensor', 'add'),
    ('aten.mul.Tensor', 'mul'), ('torch.ops.aten.mul.Tensor', 'mul'),
    ('aten.eq.Tensor', 'eq'), ('torch.ops.aten.eq.Tensor', 'eq'),
    ('aten.div.Tensor', 'div'), ('torch.ops.aten.div.Tensor', 'div'),
])
def test_full_normalization_reaches_elementwise_family(spelling, canonical):
    """namespace-1: the full torch.ops.aten.<op>.Tensor identity reaches the family."""
    a = EinsumAnalyzer()
    assert a._get_operation_from_name(spelling) == canonical
    assert a._classify_operation_name(spelling)[1] is True


@pytest.mark.parametrize('spelling,shape,output,macs', [
    ('torch.ops.aten.mm.default', [[4, 8], [8, 3]], [4, 3], 96),
    ('torch.ops.aten.bmm.default', [[2, 4, 8], [2, 8, 3]], [2, 4, 3], 192),
    ('torch.ops.aten.addmm.default', [[3], [4, 8], [8, 3]], [4, 3], 96),
    ('mm.out', [[4, 8], [8, 3]], [4, 3], 96),
    ('aten.mm.default', [[4, 8], [8, 3]], [4, 3], 96),
    ('bmm.out', [[2, 4, 8], [2, 8, 3]], [2, 4, 3], 192),
    ('addmm.out', [[3], [4, 8], [8, 3]], [4, 3], 96),
])
def test_matmul_spelling_pipeline(tmp_path, spelling, shape, output, macs):
    e, a, p = _run(tmp_path, _recorded_graph(spelling, shape, output))
    l = a['layers']['Model.' + spelling]
    assert l['macs'] == macs and a['total']['macs'] == macs
    assert a['total'].get('status', 'ok') == 'ok' and p['workload'].get('status', 'ok') == 'ok'


@pytest.mark.parametrize('spelling', ['aten.add.Tensor', 'torch.ops.aten.add.Tensor', 'aten.mul.Tensor', 'torch.ops.aten.mul.Tensor'])
def test_elementwise_spelling_pipeline(tmp_path, spelling):
    """tensor-node-2: a FunctionNode whose name contains 'tensor' is not erased."""
    e, a, p = _run(tmp_path, _recorded_graph(spelling, [[4, 3], [4, 3]], [4, 3]))
    lid = 'Model.' + spelling
    l = a['layers'].get(lid)
    assert l is not None, 'Recorded FunctionNode was erased as a tensor-side node'
    assert l.get('status', 'ok') == 'ok' and a['total'].get('status', 'ok') == 'ok'
    assert l['macs'] == 0 and l['other_ops'] == 12


@pytest.mark.parametrize('spelling', ['sort', 'bincount', 'topk', 'interpolate', 'opaque_kernel'])
def test_unhandled_is_partial_compute_with_recorded_traffic(tmp_path, spelling):
    e, a, p = _run(tmp_path, _recorded_graph(spelling, [[4, 8]], [4, 3]))
    l = a['layers']['Model.' + spelling]
    assert l['macs'] == l['other_ops'] == l['flops'] == 0
    assert l['status'] == a['total']['status'] == p['workload']['status'] == 'incomplete'
    assert l['incomplete_reason'] == 'op_unrecognised'
    assert 'op_unrecognised' in p['workload']['prediction_qualification']
    # the recorded-shape traffic remains: a partial estimate, not a vanished layer
    assert l['unfused_elements'] == 4 * 8 + 4 * 3 and l['unfused_bytes'] == 4 * (4 * 8 + 4 * 3)


def test_unknown_function_name_with_tensor_is_not_erased(tmp_path):
    spelling = 'opaque_tensor_kernel'
    e, a, p = _run(tmp_path, _recorded_graph(spelling, [[4, 3]], [4, 3]))
    l = a['layers'].get('Model.' + spelling)
    assert l is not None, 'Opaque FunctionNode disappeared before fail-closed routing'
    assert l.get('status') == 'incomplete' and a['total'].get('status') == 'incomplete'
    assert e['layers']['Model.' + spelling]['unrecognised_op'] == spelling


def test_out_destination_split_without_executing_mm():
    import torch
    from solar.graph._torchview_param_shim import _split_destination, _semantic_operand_leaves
    a = torch.empty(0, device='meta'); b = torch.empty(0, device='meta'); dest = torch.empty(0, device='meta')
    kwargs = {'out': dest}
    ops, out = _split_destination(kwargs)
    assert ops == {} and out is dest and kwargs['out'] is dest
    leaves = _semantic_operand_leaves(torch.mm, (a, b), kwargs)
    assert len(leaves) == 2 and leaves[0] is a and leaves[1] is b


# -- node partition: the recorded class is authoritative -------------------

def test_partition_respects_the_recorded_node_class():
    conv = PyTorchToEinsum()
    layers = {
        "Model.hidden-tensor": {"node_class": "TensorNode", "type": "hidden-tensor"},
        "Model.aten.add.Tensor": {"node_class": "FunctionNode", "type": "aten.add.Tensor"},
        "Model.opaque_tensor_kernel": {"node_class": "FunctionNode", "type": "opaque_tensor_kernel"},
        "Model.auxiliary-tensor": {"node_class": "TensorNode", "type": "auxiliary-tensor"},
        "Model.parameter-tensor": {"node_class": "TensorNode", "type": "parameter-tensor"},
        "Model.legacy-hidden": {"type": "hidden-tensor"},            # no class recorded: name decides
        "Model.legacy_mul": {"type": "mul.Tensor"},                  # no class recorded: an op
    }
    tensors, ops, aux, params = conv._partition_nodes(layers)
    assert ops == ["Model.aten.add.Tensor", "Model.opaque_tensor_kernel", "Model.legacy_mul"]
    assert tensors == ["Model.hidden-tensor", "Model.legacy-hidden"]
    assert aux == ["Model.auxiliary-tensor"] and params == ["Model.parameter-tensor"]


# -- estimate: a start-only graph is not a translation ---------------------

def test_estimate_refuses_a_graph_with_no_operation_layer():
    from solar.estimate import mechanical_graph_unsupported
    start = {"type": "start", "einsum_equation": "->AB", "is_einsum_supportable": False}
    assert "no operation layers" in mechanical_graph_unsupported({"layers": {"s": start}})
    assert mechanical_graph_unsupported({"layers": {}}) == "v0 produced no operation graph"
    ok = {"type": "mm", "einsum_equation": "MK,KN->MN", "is_einsum_supportable": True}
    assert mechanical_graph_unsupported({"layers": {"s": start, "m": ok}}) is None
    bad = {"type": "frob", "einsum_equation": "AB->AB", "is_einsum_supportable": False, "unrecognised_op": "frob"}
    msg = mechanical_graph_unsupported({"layers": {"s": start, "f": bad}})
    assert msg.startswith("v0 cannot lower") and "f: frob (unrecognised_op)" in msg


# -- F.interpolate: the narrowly bound identity-copy rule ------------------
# Round 4 (parser-5 / SIZE-6): the rule consumes ONLY the structured capture-time
# record module_args["interpolate_evidence"]; display strings (raw_attributes) never admit.

INTERP_RAW = ("[[Tensor(shape=(1, 64, 95), dtype=torch.float32)], {size: 95, scale_factor: None, mode: 'linear', "
              "align_corners: None, recompute_scale_factor: None, antialias: False}]")


def _evidence(mode="linear", size=95, scale_factor=None, **extra):
    from solar.einsum.op_families import interpolate_evidence_record
    import torch.nn.functional as F
    kwargs = {"size": size, "scale_factor": scale_factor, "mode": mode}
    kwargs.update(extra)
    return interpolate_evidence_record(F.interpolate, (object(),), kwargs)


def _interpolate_graph(raw=INTERP_RAW, evidence="default", shape=(1, 64, 95)):
    g = _recorded_graph("interpolate", [list(shape)], list(shape))
    ma = g["layers"]["Model.interpolate"]["module_args"]
    if raw is None:
        ma.pop("raw_attributes")
    else:
        ma["raw_attributes"] = raw
    if evidence == "default":
        evidence = _evidence()
    if evidence is not None:
        ma["interpolate_evidence"] = evidence
    return g


def test_interpolate_evidence_record_binds_the_real_signature():
    import torch.nn.functional as F
    from solar.einsum.op_families import interpolate_evidence_record as rec
    r = rec(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    assert r["schema"] == "solar-interpolate-evidence-v1" and r["mode"] == "linear"
    assert r["size"] == {"type": "int", "value": 95} and r["scale_factor"] is None    # signature default, recorded as None
    assert r["align_corners"] is None and r["antialias"] is False
    r = rec(F.interpolate, (object(), (8, 8)), {"mode": "nearest"})                  # positional size keeps its tuple type
    assert r["size"] == {"type": "tuple", "value": [8, 8]}
    r = rec(F.interpolate, (object(),), {"scale_factor": 2.0, "mode": "bilinear"})
    assert r["scale_factor"] == 2.0 and r["size"] == {"type": "NoneType", "value": None}
    assert rec(F.interpolate, (), {"bogus": 1}) is None                               # unbindable call -> no record


def test_identity_interpolate_evidence_rule():
    from solar.einsum.op_families import identity_interpolate_evidence as ev
    SH3 = [1, 64, 95]
    assert ev(_evidence(), SH3, SH3)["rule"] == "identity_size_resize_is_a_copy"
    assert ev(None, SH3, SH3) is None                                                  # no structured record
    assert ev(INTERP_RAW, SH3, SH3) is None                                            # a display string is not evidence
    assert ev(_evidence(mode="bilinear"), SH3, SH3) is None
    assert ev(_evidence(scale_factor=2.0), SH3, SH3) is None
    assert ev(_evidence(scale_factor=1.0), SH3, SH3) is None                          # must be None, not merely identity-valued
    assert ev(_evidence(size=190), SH3, [1, 64, 190]) is None
    assert ev(_evidence(size=94), SH3, SH3) is None
    SH4 = [1, 3, 8, 8]
    assert ev(_evidence(mode="nearest", size=8), SH4, SH4)["size"] == [8, 8]            # scalar broadcasts to every spatial dim
    assert ev(_evidence(mode="nearest", size=(8, 8)), SH4, SH4)["size_type"] == "tuple"
    assert ev(_evidence(mode="nearest", size=[8, 8]), SH4, SH4)["size_type"] == "list"
    assert ev(_evidence(mode="nearest", size=(8,)), SH4, SH4) is None                  # SIZE-6: singleton sequence is not a scalar
    assert ev(_evidence(mode="nearest", size=[8]), SH4, SH4) is None
    assert ev(_evidence(mode="nearest", size=(8, 8, 8)), SH4, SH4) is None             # wrong rank
    assert ev(_evidence(mode="nearest", size=(8, 7)), SH4, SH4) is None
    assert ev(_evidence(mode="nearest-exact", size=8), [1, 3, 8, 8, 8], [1, 3, 8, 8, 8])["size"] == [8, 8, 8]


def test_evidenced_identity_interpolate_is_a_zero_arithmetic_copy_with_traffic(tmp_path):
    e, a, p = _run(tmp_path, _interpolate_graph())
    layer = e["layers"]["Model.interpolate"]
    assert layer["is_einsum_supportable"] and "unrecognised_op" not in layer
    assert layer["interpolate_identity_copy"]["mode"] == "linear" and layer["interpolate_identity_copy"]["size"] == [95]
    assert layer["interpolate_identity_copy"]["binding"].startswith("structured capture-time record")
    al = a["layers"]["Model.interpolate"]
    assert al.get("status") != "incomplete" and al["macs"] == 0 and al["other_ops"] == 0
    assert al["unfused_bytes"] == 48_640                       # 24,320 B read + 24,320 B written (fp32 [1, 64, 95])
    assert a["total"].get("status") != "incomplete" and p["workload"].get("status") != "incomplete"


@pytest.mark.parametrize("evidence", [None, "bilinear", "scale", "legacy-string-only"],
                         ids=["no-record", "bilinear", "scale-factor", "legacy-string-only"])
def test_interpolate_without_structured_identity_evidence_stays_unrecognised(tmp_path, evidence):
    ev = {None: None, "bilinear": _evidence(mode="bilinear"), "scale": _evidence(scale_factor=1.0), "legacy-string-only": None}[evidence]
    e, a, p = _run(tmp_path, _interpolate_graph(raw=INTERP_RAW, evidence=ev))
    layer = e["layers"]["Model.interpolate"]
    assert layer["is_einsum_supportable"] is False and layer["unrecognised_op"] == "interpolate"
    assert "interpolate_identity_copy" not in layer
    assert a["layers"]["Model.interpolate"]["incomplete_reason"] == "op_unrecognised"


# ---------------------------------------------------------------------------
# Round-3 controls (an earlier review, checks-dad5170c/test_interpolate_delta.py
# and test_round2_boundaries.py, verbatim bodies): exact top-level key binding
# ---------------------------------------------------------------------------

from solar.einsum.op_families import parse_recorded_kwargs

DELTA_CASES = {
    'no_explicit_scale': "[[], {size: 95, mode: 'linear', recompute_scale_factor: None}]",
    'shadowed_scale_key': "[[], {size: 95, recompute_scale_factor: None, scale_factor: 1.001, mode: 'linear'}]",
    'suffix_mode_key': "[[], {size: 95, scale_factor: None, legacy_mode: 'linear'}]",
    'nested_mode_key': "[[], {size: 95, scale_factor: None, metadata: {mode: 'linear'}}]",
}


@pytest.mark.parametrize('case', list(DELTA_CASES))
def test_unbound_interpolate_metadata_stays_incomplete(tmp_path, case):
    g = _recorded_graph('interpolate', [[1, 64, 95]], [1, 64, 95])
    g['layers']['Model.interpolate']['module_args']['raw_attributes'] = DELTA_CASES[case]
    e, a, p = _run(tmp_path, g)
    assert 'interpolate_identity_copy' not in e['layers']['Model.interpolate']
    assert a['layers']['Model.interpolate']['status'] == a['total']['status'] == p['workload']['status'] == 'incomplete'


SH = [1, 64, 95]
RAW = ("[[Tensor(shape=(1, 64, 95), dtype=torch.float32)], {size: 95, scale_factor: None, mode: 'linear', "
       "align_corners: None, recompute_scale_factor: None, antialias: False}]")


def _bgraph(raw, shape=SH, evidence=None):
    """Graph with a display string only (never evidence) and, optionally, the structured record."""
    g = _recorded_graph('interpolate', [shape], shape)
    if raw is None:
        g['layers']['Model.interpolate']['module_args'].pop('raw_attributes')
    else:
        g['layers']['Model.interpolate']['module_args']['raw_attributes'] = raw
    if evidence is not None:
        g['layers']['Model.interpolate']['module_args']['interpolate_evidence'] = evidence
    return g


@pytest.mark.parametrize('mode', ['linear', 'nearest', 'nearest-exact'])
def test_identity_known_modes_exact_traffic(tmp_path, mode):
    e, a, p = _run(tmp_path, _bgraph(RAW.replace("'linear'", repr(mode)), evidence=_evidence(mode=mode)))
    l = a['layers']['Model.interpolate']
    assert l['macs'] == l['other_ops'] == 0 and l['unfused_bytes'] == 48640
    assert a['total'].get('status', 'ok') == p['workload'].get('status', 'ok') == 'ok'
    assert e['layers']['Model.interpolate']['interpolate_identity_copy']['mode'] == mode


@pytest.mark.parametrize('raw', [
    None, RAW.replace("mode: 'linear', ", ''), RAW.replace('scale_factor: None, ', '', 1),
    RAW.replace('size: 95', 'size: None').replace('scale_factor: None', 'scale_factor: 1.001', 1),
    RAW.replace('scale_factor: None', 'scale_factor: 1.001', 1),
    RAW.replace('size: 95', 'size: 94'), RAW.replace("'linear'", "'bicubic'"),
], ids=['no-attrs', 'no-mode', 'no-explicit-scale', 'real-scale-factor-form', 'contradictory-size-and-scale',
        'different-size', 'unsupported-mode'])
def test_insufficient_interpolate_evidence_fails_closed(tmp_path, raw):
    e, a, p = _run(tmp_path, _bgraph(raw))
    l = a['layers']['Model.interpolate']; el = e['layers']['Model.interpolate']
    assert 'interpolate_identity_copy' not in el
    assert el['unrecognised_op'] == 'interpolate'
    assert l['status'] == a['total']['status'] == p['workload']['status'] == 'incomplete'
    assert l['incomplete_reason'] == 'op_unrecognised'


@pytest.mark.parametrize('mode,shape,size', [('nearest', [1, 3, 8, 8], [8, 8]), ('nearest-exact', [1, 3, 8, 8, 8], 8)])
def test_multidimensional_identity_shape_bound(tmp_path, mode, shape, size):
    raw = "[[], {size: " + str(size) + ", scale_factor: None, mode: '" + mode + "'}]"
    e, a, p = _run(tmp_path, _bgraph(raw, shape, evidence=_evidence(mode=mode, size=size)))
    n = 1
    for x in shape:
        n *= x
    assert a['layers']['Model.interpolate']['unfused_bytes'] == n * 8
    assert a['total'].get('status', 'ok') == 'ok'


@pytest.mark.parametrize('name', ['opaque_tensor_kernel', 'opaque-tensor', 'input-tensor', 'parameter-tensor'])
@pytest.mark.parametrize('cls', ['FunctionNode', 'ModuleNode'])
def test_explicit_operation_class_beats_tensor_name_end_to_end(tmp_path, name, cls):
    g = _recorded_graph(name, [[4, 3]], [4, 3]); g['layers']['Model.' + name]['node_class'] = cls
    e, a, p = _run(tmp_path, g)
    l = a['layers'].get('Model.' + name)
    assert l is not None
    assert l['status'] == a['total']['status'] == p['workload']['status'] == 'incomplete'
    assert l['incomplete_reason'] == 'op_unrecognised'


@pytest.mark.parametrize('name', ['aten.eq.Tensor', 'torch.ops.aten.eq.Tensor', 'aten.div.Tensor', 'torch.ops.aten.div.Tensor'])
def test_remaining_normalized_elementwise_routes(tmp_path, name):
    e, a, p = _run(tmp_path, _recorded_graph(name, [[4, 3], [4, 3]], [4, 3]))
    l = a['layers']['Model.' + name]
    assert l['other_ops'] == 12 and l['macs'] == 0
    assert a['total'].get('status', 'ok') == p['workload'].get('status', 'ok') == 'ok'


@pytest.mark.parametrize('type', ['start', 'end', 'input', 'output'])
def test_boundary_only_graph_is_not_translation(type):
    from solar.estimate import mechanical_graph_unsupported
    g = {'layers': {'boundary': {'type': type, 'einsum_equation': 'AB->AB', 'is_einsum_supportable': True}}}
    assert 'no operation layers' in mechanical_graph_unsupported(g)


def test_estimate_mixed_graph_retains_unknown_refusal():
    from solar.estimate import mechanical_graph_unsupported
    graph = {'layers': {'start': {'type': 'start'}, 'copy': {'type': 'clone', 'einsum_equation': 'AB->AB', 'is_einsum_supportable': True},
                        'x': {'type': 'opaque_tensor_kernel', 'einsum_equation': 'AB->AB', 'is_einsum_supportable': False, 'unrecognised_op': 'opaque_tensor_kernel'}}}
    assert 'x: opaque_tensor_kernel (unrecognised_op)' in mechanical_graph_unsupported(graph)


@pytest.mark.parametrize('raw', [
    "[[], {size: 95, recompute_scale_factor: None, scale_factor: 1.001, mode: 'linear'}]",
    "[[], {size: 95, scale_factor: None, legacy_mode: 'linear'}]",
    "[[], {size: 95, scale_factor: None, metadata: {mode: 'linear'}}]",
], ids=['shadowed-scale-key', 'suffix-mode-key', 'nested-mode-key'])
def test_ambiguous_or_contradictory_metadata_is_not_positive_evidence(tmp_path, raw):
    # Synthetic metadata-contract controls, not claims these are valid native F.interpolate calls.
    e, a, p = _run(tmp_path, _bgraph(raw))
    assert 'interpolate_identity_copy' not in e['layers']['Model.interpolate']
    assert a['total'].get('status') == p['workload'].get('status') == 'incomplete'


# -- the scoped parser itself ----------------------------------------------

def test_recorded_kwargs_parser_binds_exact_top_level_keys_only():
    kw = parse_recorded_kwargs(RAW)
    assert kw == {'size': 95, 'scale_factor': None, 'mode': 'linear', 'align_corners': None,
                  'recompute_scale_factor': None, 'antialias': False}
    nested = parse_recorded_kwargs("[[], {size: 95, scale_factor: None, metadata: {mode: 'linear'}}]")
    assert 'mode' not in nested and 'metadata' in nested                      # nested keys never surface
    assert parse_recorded_kwargs("[[], {size: 95, mode: 'linear', mode: 'bilinear'}]") is None   # duplicate: ambiguous
    assert parse_recorded_kwargs("[[], {size: (8, 8), scale_factor: None, mode: 'nearest'}]")['size'] == [8, 8]
    assert parse_recorded_kwargs("not a recorded call") is None
    assert parse_recorded_kwargs("[[Tensor(shape=(1, 2), dtype=torch.float32)], {}]") == {}


def test_duplicate_or_malformed_records_refuse_the_identity_copy():
    from solar.einsum.op_families import identity_interpolate_evidence as ev, parse_recorded_kwargs
    assert parse_recorded_kwargs("[[], {size: 95, scale_factor: None, mode: 'linear', mode: 'linear'}]") is None   # diagnostic: duplicate key
    assert ev("[[], {size: 95, scale_factor: None, mode: 'linear'}]", SH, SH) is None   # any display string: never evidence
    bad = _evidence(); bad["schema"] = "something-else"
    assert ev(bad, SH, SH) is None


# ---------------------------------------------------------------------------
# Round-4 controls (an earlier review, checks-d5de1628/test_parser_refusals.py and
# test_round3_parser.py, verbatim bodies): display strings never admit; size rule.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', ['quoted_mode', 'quoted_scale', 'extra_dict', 'no_args', 'bad_brackets', 'tuple_size', 'list_size'])
def test_parser_refusal_pipeline(tmp_path, case):
    from torchview.utils import stringify_attributes
    shape = [1, 64, 95]
    if case == 'quoted_mode':
        kwargs = {'size': 95, 'scale_factor': None, 'metadata': "', mode: \"linear\", note: '"}
        raw = stringify_attributes(([], kwargs))
    elif case == 'quoted_scale':
        kwargs = {'size': 95, 'mode': 'linear', 'metadata': "', scale_factor: None, note: '\""}
        raw = stringify_attributes(([], kwargs))
    elif case == 'extra_dict':
        raw = "[[], {size: 94, scale_factor: 1.01, mode: 'bicubic'}, {size: 95, scale_factor: None, mode: 'linear'}]"
    elif case == 'no_args':
        raw = "[{size: 95, scale_factor: None, mode: 'linear'}]"
    elif case == 'bad_brackets':
        raw = "[[], {metadata: (], size: 95, scale_factor: None, mode: 'linear'}]"
    else:
        shape = [1, 3, 8, 8]; raw = "[[], {size: " + ('(8,)' if case == 'tuple_size' else '[8]') + ", scale_factor: None, mode: 'nearest'}]"
    g = _recorded_graph('interpolate', [shape], shape); g['layers']['Model.interpolate']['module_args']['raw_attributes'] = raw
    e, a, p = _run(tmp_path, g)
    rec = {'case': case, 'converted': e['layers']['Model.interpolate'], 'analysis': a['layers']['Model.interpolate'], 'total': a['total'], 'perf_workload': p['workload']}
    assert 'interpolate_identity_copy' not in rec['converted']
    assert rec['analysis']['status'] == rec['total']['status'] == rec['perf_workload']['status'] == 'incomplete'


@pytest.mark.parametrize('missing', ['mode', 'scale_factor'])
def test_real_stringifier_escaped_string_cannot_supply_a_keyword(tmp_path, missing):
    # Actual capture serializer on inert Python objects, not a model or tensor operation.
    from torchview.utils import stringify_attributes
    if missing == 'mode':
        kwargs = {'size': 95, 'scale_factor': None, 'metadata': "', mode: \"linear\", note: '"}
    else:
        kwargs = {'size': 95, 'mode': 'linear', 'metadata': "', scale_factor: None, note: '\""}
    assert missing not in kwargs
    raw = stringify_attributes(([], kwargs))
    e, a, p = _run(tmp_path, _bgraph(raw))
    assert 'interpolate_identity_copy' not in e['layers']['Model.interpolate']
    assert a['total']['status'] == p['workload']['status'] == 'incomplete'


@pytest.mark.parametrize('size', [(8,), [8]], ids=['tuple-singleton', 'list-singleton'])
def test_singleton_sequence_is_not_scalar_broadcast(tmp_path, size):
    """SIZE-6 on the structured record: F.interpolate requires a sequence of the spatial rank."""
    e, a, p = _run(tmp_path, _bgraph(None, [1, 3, 8, 8], evidence=_evidence(mode='nearest', size=size)))
    assert 'interpolate_identity_copy' not in e['layers']['Model.interpolate']
    assert a['total']['status'] == p['workload']['status'] == 'incomplete'


@pytest.mark.parametrize('size', [8, (8, 8), [8, 8]], ids=['int', 'tuple', 'list'])
def test_multidimensional_int_or_exact_length_sequence(tmp_path, size):
    e, a, p = _run(tmp_path, _bgraph(None, [1, 3, 8, 8], evidence=_evidence(mode='nearest', size=size)))
    assert e['layers']['Model.interpolate']['interpolate_identity_copy']['size'] == [8, 8]
    assert a['total'].get('status', 'ok') == 'ok'


def test_diagnostic_parser_does_not_end_a_quote_at_an_escaped_quote():
    from torchview.utils import stringify_attributes
    from solar.einsum.op_families import parse_recorded_kwargs
    raw = stringify_attributes(([], {'size': 95, 'scale_factor': None, 'metadata': "', mode: \"linear\", note: '"}))
    kw = parse_recorded_kwargs(raw)
    assert kw is not None and 'mode' not in kw and set(kw) == {'size', 'scale_factor', 'metadata'}


def test_structured_record_from_a_string_is_not_evidence():
    """The display-string parser is diagnostic only and never feeds the admission."""
    from solar.einsum.op_families import identity_interpolate_evidence as ev, parse_recorded_kwargs
    kw = parse_recorded_kwargs("[[], {size: 95, scale_factor: None, mode: 'linear'}]")
    assert kw == {'size': 95, 'scale_factor': None, 'mode': 'linear'}
    assert ev(kw, [1, 64, 95], [1, 64, 95]) is None                      # a plain dict without the schema never admits


# -- capture-level: the shim writes the structured record from the real call ----

def _capture_interpolate(tmp_path, call):
    import contextlib, io
    from solar.common.types import ProcessingConfig
    from solar.graph.pytorch_processor import PyTorchProcessor
    src = ("import torch, torch.nn.functional as F\nfrom torch import nn\n"
           "class Model(nn.Module):\n    def forward(self, x):\n        return " + call + "\n"
           "def get_inputs():\n    return [torch.randn(1, 64, 95)]\ndef get_init_inputs():\n    return []\n")
    (tmp_path / "model.py").write_text(src)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        ok = PyTorchProcessor(ProcessingConfig(force_rerun=True, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert ok, stream.getvalue()
    g = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text())
    node = next(v for v in g["layers"].values() if v.get("type") == "interpolate")
    return g, node


def test_capture_records_structured_interpolate_evidence_and_admits_identity(tmp_path):
    g, node = _capture_interpolate(tmp_path, "F.interpolate(x, size=95, mode='linear')")
    ev = node["module_args"]["interpolate_evidence"]
    assert ev["schema"] == "solar-interpolate-evidence-v1" and ev["mode"] == "linear"
    assert ev["size"] == {"type": "int", "value": 95} and ev["scale_factor"] is None
    (tmp_path / "pipe").mkdir()
    e, a, p = _run(tmp_path / "pipe", g)
    layer = next(v for v in e["layers"].values() if v.get("type") == "interpolate")
    assert layer["interpolate_identity_copy"]["size"] == [95] and a["total"].get("status") != "incomplete"


def test_capture_with_a_scale_factor_is_refused(tmp_path):
    g, node = _capture_interpolate(tmp_path, "F.interpolate(x, scale_factor=2.0, mode='linear')")
    assert node["module_args"]["interpolate_evidence"]["scale_factor"] == 2.0
    (tmp_path / "pipe").mkdir()
    e, a, p = _run(tmp_path / "pipe", g)
    layer = next(v for v in e["layers"].values() if v.get("type") == "interpolate")
    assert layer["unrecognised_op"] == "interpolate" and a["total"]["status"] == "incomplete"


def test_capture_without_the_record_is_refused_even_with_raw_attributes(tmp_path):
    g, node = _capture_interpolate(tmp_path, "F.interpolate(x, size=95, mode='linear')")
    node["module_args"].pop("interpolate_evidence")          # a legacy graph: raw_attributes only
    assert "raw_attributes" in node["module_args"]
    (tmp_path / "pipe").mkdir()
    e, a, p = _run(tmp_path / "pipe", g)
    layer = next(v for v in e["layers"].values() if v.get("type") == "interpolate")
    assert layer["unrecognised_op"] == "interpolate" and a["total"]["status"] == "incomplete"


# -- round 5 (an earlier review): callee identity + record schema -------------

CUSTOM_INTERPOLATE = (
    "def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, "
    "recompute_scale_factor=None, antialias=False):\n"
    "    if torch.overrides.has_torch_function_unary(input):\n"
    "        return torch.overrides.handle_torch_function(interpolate, (input,), input, size=size, "
    "scale_factor=scale_factor, mode=mode, align_corners=align_corners, "
    "recompute_scale_factor=recompute_scale_factor, antialias=antialias)\n"
    "    return input * 2\n")


def _capture_source(tmp_path, body, extra=""):
    import contextlib, io
    from solar.common.types import ProcessingConfig
    from solar.graph.pytorch_processor import PyTorchProcessor
    src = ("import torch, functools\nimport torch.nn.functional as F\nfrom torch import nn\n" + extra +
           "\nclass Model(nn.Module):\n    def forward(self, x):\n" +
           "".join("        " + line + "\n" for line in body.splitlines()) +
           "\ndef get_inputs():\n    return [torch.randn(1, 64, 95)]\ndef get_init_inputs():\n    return []\n")
    (tmp_path / "model.py").write_text(src)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        ok = PyTorchProcessor(ProcessingConfig(force_rerun=True, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert ok, stream.getvalue()
    g = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text())
    (tmp_path / "pipe").mkdir()
    with contextlib.redirect_stdout(stream):
        e, a, p = _run(tmp_path / "pipe", g)
    ops = {k: v for k, v in e["layers"].items() if v.get("node_class") != "TensorNode"
           and ("interpolate" in str(v.get("type", "")).lower() or "upsample" in str(v.get("type", "")).lower())}
    return g, e, a, p, ops


def test_callee_identity_is_object_identity_not_a_name():
    from solar.einsum.op_families import callee_identity
    real = callee_identity(F.interpolate)
    assert real["is_torch_library"] is True and real["bound_as"] == "torch.nn.functional.interpolate"
    ns = {"torch": torch}
    exec(CUSTOM_INTERPOLATE, ns)
    fake = callee_identity(ns["interpolate"])
    assert fake["qualname"] == "interpolate" and fake["is_torch_library"] is False and fake["bound_as"] is None
    # a same-named attribute planted on a module of the right name is still not the object
    fake2 = dict(fake, module="torch.nn.functional")
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"] = fake2
    assert identity_interpolate_evidence(ev, [1, 64, 95], [1, 64, 95]) is None
    assert interpolate_record_refusal(ev).startswith("interpolate_callee_untrusted")


def test_custom_same_named_callable_is_not_evidence_of_library_identity_copy(tmp_path):
    # reviewer control (checks-a88c1d83/test_structured_capture.py): a torch-function-dispatch callable named
    # `interpolate` with F.interpolate's signature and body `input * 2` must NOT receive the copy rule
    g, e, a, p, ops = _capture_source(tmp_path, "return interpolate(x, size=95, mode='linear')", CUSTOM_INTERPOLATE)
    assert ops
    assert not any(v.get("interpolate_identity_copy") for v in ops.values())
    assert a["total"]["status"] == "incomplete"
    node = next(v for v in g["layers"].values() if v.get("type") == "interpolate")
    ci = node["module_args"]["interpolate_evidence"]["callee_identity"]
    assert ci["is_torch_library"] is False and ci["qualname"] == "interpolate"
    layer = next(v for v in ops.values() if v.get("unrecognised_op") == "interpolate")
    assert layer["interpolate_refusal"].startswith("interpolate_callee_untrusted")


@pytest.mark.parametrize("kind,body", [
    ("direct", "return F.interpolate(x, size=95, mode='linear')"),
    ("partial", "fn = functools.partial(F.interpolate, size=95, mode='linear')\nreturn fn(x)"),
    ("lambda", "fn = lambda y: F.interpolate(y, size=95, mode='linear')\nreturn fn(x)"),
])
def test_real_library_callable_through_wrappers_keeps_the_copy_rule(tmp_path, kind, body):
    g, e, a, p, ops = _capture_source(tmp_path, body)
    identity = [v for v in ops.values() if v.get("interpolate_identity_copy")]
    assert len(identity) == 1 and not any(v.get("unrecognised_op") for v in ops.values())
    assert identity[0]["interpolate_identity_copy"]["callee"] == "torch.nn.functional.interpolate"
    assert a["total"]["macs"] == 0 and a["total"]["other_ops"] == 0 and a["total"].get("status") != "incomplete"


def test_nn_upsample_module_inner_call_supplies_the_evidence_or_fails_closed(tmp_path):
    g, e, a, p, ops = _capture_source(tmp_path, "return nn.Upsample(size=95, mode='linear')(x)")
    # the module boundary may refuse; either a real admitted inner F.interpolate call or fail closed
    identity = [v for v in ops.values() if v.get("interpolate_identity_copy")]
    unknown = [v for v in ops.values() if v.get("unrecognised_op")]
    assert identity or (unknown and a["total"]["status"] == "incomplete")


@pytest.mark.parametrize("field", ["scale_factor", "mode", "size", "align_corners", "recompute_scale_factor",
                                   "antialias", "callee_identity"])
def test_incomplete_structured_record_does_not_invent_default_none(tmp_path, field):
    # reviewer control (RECORD-8): removing one REQUIRED field from a valid record must refuse, never synthesise
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev.pop(field)
    assert interpolate_record_refusal(ev) == f"interpolate_record_incomplete: absent field(s) {field}"
    assert identity_interpolate_evidence(ev, SH, SH) is None
    g = _recorded_graph("interpolate", [SH], SH)
    g["layers"]["Model.interpolate"]["module_args"]["interpolate_evidence"] = ev
    e, a, p = _run(tmp_path, g)
    layer = e["layers"]["Model.interpolate"]
    assert layer.get("unrecognised_op") == "interpolate" and "interpolate_identity_copy" not in layer
    assert layer["interpolate_refusal"].startswith("interpolate_record_incomplete")
    assert a["total"]["status"] == "incomplete" and p["workload"]["status"] == "incomplete"


def test_present_signature_default_none_is_a_bound_value_and_is_accepted():
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    assert "scale_factor" in ev and ev["scale_factor"] is None          # apply_defaults: present, bound None
    assert interpolate_record_refusal(ev) is None
    proof = identity_interpolate_evidence(ev, SH, SH)
    assert proof is not None and proof["scale_factor"] is None and proof["callee"] == "torch.nn.functional.interpolate"


@pytest.mark.parametrize("field,bad", [
    ("mode", 17), ("size", [95]), ("size", {"type": "int"}), ("scale_factor", "1.0"),
    ("scale_factor", {"unrecorded_type": "Tensor"}), ("callee_identity", "torch.nn.functional.interpolate"),
    ("callee_identity", {"is_torch_library": True}), ("schema", "solar-interpolate-evidence-v0"),
])
def test_wrongly_typed_record_fields_refuse(field, bad):
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev[field] = bad
    assert interpolate_record_refusal(ev) is not None and identity_interpolate_evidence(ev, SH, SH) is None


# -- round 7 (an earlier review, KERNEL-ORIGIN-10): DSL provenance, not the type name, admits the generic cost --

def test_dsl_builtin_table_matches_einsum_lib():
    import solar.einsum_lib as L
    from solar.einsum_dsl.builtin_table import DSL_TENSOR_BUILTINS
    from solar.einsum_lib.numba_accel import execution_namespace
    assert DSL_TENSOR_BUILTINS == {"apply_mask", "top_k_mask", "slice_at", "stack", "concat", "gather"}
    assert DSL_TENSOR_BUILTINS <= set(L.__all__) and DSL_TENSOR_BUILTINS <= set(execution_namespace())


def test_generic_cost_is_admitted_by_resolved_builtin_not_by_the_type_name():
    # the reviewer's gather shapes: 64 (the table's elements) ONLY with the exporter's resolved builtin
    ts = TensorShapes(inputs=[[16, 4], [2]], outputs=[[2, 4]])
    assert int(EinsumAnalyzer().get_compute_cost("kernel", ts, raw_attributes=None, dsl_builtin="gather")) == 64
    with pytest.raises(UnrecognisedOperationError):
        EinsumAnalyzer().get_compute_cost("kernel", ts, raw_attributes=None)                      # name alone: refused
    with pytest.raises(UnrecognisedOperationError):
        EinsumAnalyzer().get_compute_cost("kernel", ts, raw_attributes=None, dsl_builtin="opaque_unknown")
    with pytest.raises(UnrecognisedOperationError):
        EinsumAnalyzer().get_einsum_op("kernel", ts)
    # a PARSED DSL kernel body keeps the target's generic cost path (test_nonreduction_loop_evidence: an untrusted
    # loop count falls back to the shape-based cost with the loop_metadata_incomplete marker)
    assert int(EinsumAnalyzer().get_compute_cost("kernel", ts, raw_attributes=None, dsl_kernel_body=True)) == 64
    with pytest.raises(UnrecognisedOperationError):
        EinsumAnalyzer().get_compute_cost("kernel", ts, raw_attributes=None, dsl_kernel_body="yes")
    assert not hasattr(__import__("solar.einsum.op_families", fromlist=["x"]), "DSL_KERNEL_TYPES")


def test_torch_record_named_kernel_remains_unknown(tmp_path):
    # reviewer control (checks-19a3ed4b/test_round6_independent.py): a captured torch-route record named `kernel`
    e, a, p = _run(tmp_path, _recorded_graph("kernel", [[2, 4]], [2, 4]))
    assert e["layers"]["Model.kernel"].get("unrecognised_op") == "kernel"
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"


def test_actual_torch_callable_named_kernel_is_not_dsl(tmp_path):
    # reviewer control: a real torch-function-dispatch callable named `kernel` (body input*2 + input*3)
    code = CUSTOM_INTERPOLATE.replace("interpolate", "kernel").replace("return input * 2", "return input * 2 + input * 3")
    g, e, a, p, ops = _capture_source(tmp_path, "return kernel(x)", code)
    nodes = {k: v for k, v in g["layers"].items() if v.get("type") == "kernel"}
    assert nodes
    assert all(e["layers"][k].get("unrecognised_op") == "kernel" for k in nodes)
    assert all("dsl_route" not in e["layers"][k] for k in nodes)
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"


def _dsl_pipeline(tmp_path, md, py_text="# Saved metadata names an unresolved operation; no kernel body or builtin definition.\n"):
    from solar.einsum_dsl.cli_einsum import translate_with_einsum
    d = tmp_path / "dsl"
    d.mkdir()
    (d / "unknown_einsum_lib.yaml").write_text(yaml.safe_dump(md, sort_keys=False))
    (d / "unknown_einsum_lib.py").write_text(py_text)
    graph = translate_with_einsum(d, tmp_path / "einsum")
    raw = yaml.safe_load(graph.read_text())
    a = EinsumGraphAnalyzer().analyze_graph(str(graph), str(tmp_path / "analysis"))
    p = EinsumGraphPerfModel().predict(str(tmp_path / "analysis/analysis.yaml"), str(tmp_path / "perf"), arch_config="H100_PCIe")
    return raw, a, p


@pytest.mark.parametrize("frm", [None, "input"])
def test_unknown_bodiless_dsl_operation_stays_incomplete(tmp_path, frm):
    # reviewer controls (test_round6_independent.py / test_dsl_connected.py): a DSL record with no body and no
    # resolved builtin is exported unrecognised and reported incomplete, never priced from its shapes
    md = {"tensors": {"x": {"dims": {"N": 2, "H": 4}, "dtype": "fp32", "role": "input"}, "out": {"dims": {"N": 2, "H": 4}, "dtype": "fp32"}},
          "graph": {"opaque_unknown#0": {"function": "opaque_unknown", "inputs": [{"tensor": "x", "from": frm}], "output": "out"}}}
    raw, a, p = _dsl_pipeline(tmp_path, md)
    layer = raw["layers"]["opaque_unknown#0"]
    assert layer["type"] == "kernel" and layer["dsl_route"] is True and layer["builtin"] is None and layer["kernel_body"] is False
    assert layer["unrecognised_op"] == "opaque_unknown"
    if frm == "input":                      # the connected variant (test_dsl_connected.py) is not orphaned
        assert not a["layers"]["opaque_unknown#0"].get("is_orphaned")
    assert a["layers"]["opaque_unknown#0"]["macs"] == 0 and a["layers"]["opaque_unknown#0"]["other_ops"] == 0
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"


def test_bodiless_dsl_builtin_keeps_the_target_generic_cost(tmp_path):
    # the control the kernel-type exemption was patching (test_loop_basis_completeness gather-64): now admitted by
    # the exporter's resolved builtin, with the provenance visible on the node
    md = {"tensors": {"x": {"dims": {"E": 16, "H": 4}, "dtype": "fp32", "role": "input"},
                      "i": {"dims": {"N": 2}, "dtype": "int32", "role": "input"},
                      "out": {"dims": {"N": 2, "H": 4}, "dtype": "fp32"}},
          "graph": {"gather#0": {"function": "gather", "inputs": [{"tensor": "x", "from": "input"}, {"tensor": "i", "from": "input"}], "output": "out"}}}
    raw, a, p = _dsl_pipeline(tmp_path, md, "# bodiless einsum_lib builtin call (gather)\n")
    layer = raw["layers"]["gather#0"]
    assert layer["dsl_route"] is True and layer["builtin"] == "gather" and layer["kernel_body"] is False and "unrecognised_op" not in layer
    # gather rule (issue #59): one address op per gathered element, not the former dense 64
    assert a["layers"]["gather#0"]["macs"] == 0 and a["layers"]["gather#0"]["other_ops"] == 8
    assert a["layers"]["gather#0"]["cost_basis"] == "gather_elements"
    assert "status" not in a["layers"]["gather#0"] and a["total"].get("status") != "incomplete"


# -- round 8 (an earlier review, LEGACY-ORIGIN-12): pre-marker DSL records need POSITIVE body evidence ------

_HISTORICAL_UNRESOLVED = Path(__file__).resolve().parent / "fixtures" / "mr112" / "historical-unresolved-dsl.json"


def _analyze_raw(tmp_path, graph):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(graph, sort_keys=False))
    a = EinsumGraphAnalyzer().analyze_graph(str(path), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
    assert a is not None
    p = EinsumGraphPerfModel().predict(str(tmp_path / "analysis/analysis.yaml"), str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
    return a, p


def test_saved_pre_marker_unresolved_dsl_is_incomplete(tmp_path):
    # reviewer control (checks-65093d2b/test_round7_provenance.py): the exact raw graph the round-6 converter wrote for
    # the connected unknown-body probe -- `loop_expr: null`, no dsl_route, no loop_iterations, no unrecognised_op
    import json
    raw = json.loads(_HISTORICAL_UNRESOLVED.read_text())["raw"]
    l = raw["layers"]["opaque_unknown#0"]
    assert l["loop_expr"] is None and "dsl_route" not in l and "loop_iterations" not in l and "unrecognised_op" not in l
    a, p = _analyze_raw(tmp_path, raw)
    layer = a["layers"]["opaque_unknown#0"]
    assert not layer.get("is_orphaned")
    assert layer["macs"] == 0 and layer["other_ops"] == 0
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "dsl_kernel_unresolved_legacy"
    assert layer["cost_basis"] == "incomplete_evidence"
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"


def _legacy_kernel_graph(extra):
    # a pre-marker DSL einsum graph (route-identifying fields only) with the given extra fields on the kernel
    # ops None/None: exactly what the old exporter wrote for a BODILESS kernel (`_infer_ops(None)`)
    layer = {"type": "kernel", "einsum_equation": "NH->NH", "is_real_einsum": False, "elementwise_op": None, "reduction_op": None,
             "equation_faithful": True, "loop_expr": None,
             "tensor_names": {"inputs": ["x"], "outputs": ["y"]}, "tensor_types": {"inputs": ["input"], "outputs": ["input"]},
             "tensor_shapes": {"inputs": [[2, 4]], "outputs": [[2, 4]]}, "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
             "connections": {"inputs": ["start"], "outputs": []}}
    layer.update(extra)
    return {"model_name": "legacy", "layers": {
        "start": {"type": "start", "einsum_equation": "NH->NH", "is_real_einsum": False, "elementwise_op": "copy", "reduction_op": "none",
                  "tensor_names": {"inputs": [], "outputs": ["x"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                  "tensor_shapes": {"inputs": [], "outputs": [[2, 4]]}, "tensor_dtypes": {"inputs": [], "outputs": ["fp32"]},
                  "connections": {"inputs": [], "outputs": ["k#0"]}},
        "k#0": layer}}


def test_legacy_record_with_a_real_loop_expr_stays_priced(tmp_path):
    # positive control: a saved pre-marker kernel body (non-null loop_expr) keeps the target's generic cost
    a, p = _analyze_raw(tmp_path, _legacy_kernel_graph({"loop_expr": "Y[n, h] = X[n, h] * 2.0", "elementwise_op": "mul", "reduction_op": "none"}))
    layer = a["layers"]["k#0"]
    assert layer["other_ops"] == 8 and layer["cost_basis"] == "shape_generic" and "status" not in layer
    assert a["total"].get("status") != "incomplete"


def test_legacy_record_with_body_inferred_ops_but_no_loop_expr_stays_priced(tmp_path):
    # the landed an earlier change nonreduction controls' shape: ops inferred from a body at export time, no loop_expr key,
    # unproven or absent count -> the target's generic cost (+ its own loop_metadata marker when a count is unproven)
    g = _legacy_kernel_graph({"elementwise_op": "mul", "reduction_op": "none"})
    del g["layers"]["k#0"]["loop_expr"]
    a, p = _analyze_raw(tmp_path, g)
    layer = a["layers"]["k#0"]
    assert layer["other_ops"] == 8 and layer["cost_basis"] == "shape_generic" and "status" not in layer


def test_legacy_record_with_a_complete_numeric_loop_count_is_priced_on_the_loop_basis(tmp_path):
    a, p = _analyze_raw(tmp_path, _legacy_kernel_graph({"loop_iterations": 8, "loop_iterations_complete": True}))
    layer = a["layers"]["k#0"]
    assert layer["other_ops"] == 8 and layer["cost_basis"] == "loop_iterations" and "status" not in layer


def test_legacy_record_with_a_table_builtin_is_priced(tmp_path):
    a, p = _analyze_raw(tmp_path, _legacy_kernel_graph({"builtin": "gather"}))
    layer = a["layers"]["k#0"]
    # gather rule (issue #59): one address op per gathered (output) element; the [2,4] output has 8
    assert layer["other_ops"] == 8 and layer["cost_basis"] == "gather_elements" and "status" not in layer


@pytest.mark.parametrize("extra", [
    {},                                                        # loop_expr: null + equation_faithful only
    {"loop_iterations": 0, "loop_iterations_complete": True},  # non-positive count
    {"loop_iterations": True, "loop_iterations_complete": True},  # bool is not a count
    {"loop_iterations": 8},                                    # unproven count (no completeness)
    {"loop_iterations_omitted": ["while loop"]},               # omissions only
    {"builtin": "opaque_unknown"},                             # name not in the builtin table
], ids=["fields_only", "zero_count", "bool_count", "unproven_count", "omissions_only", "unknown_builtin"])
def test_legacy_record_without_positive_body_evidence_is_incomplete(tmp_path, extra):
    a, p = _analyze_raw(tmp_path, _legacy_kernel_graph(extra))
    layer = a["layers"]["k#0"]
    assert layer["macs"] == 0 and layer["other_ops"] == 0
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] in ("dsl_kernel_unresolved_legacy", "loop_metadata_incomplete")
    assert a["total"]["status"] == "incomplete" and p["workload"]["status"] == "incomplete"


# -- round 9 (an earlier review, LEGACY-SCHEMA-14): typed, parse-validated legacy body evidence ----------

def test_legacy_predicates_come_from_the_exporter():
    from solar.einsum_dsl import einsum_converter as ec
    assert ec.BODY_ACCUMULATION_TOKENS == {"add", "sub"} and ec.BODY_REDUCTION_TOKENS == {"add", "sub", "none"}
    assert ec.BODY_ELEMENTWISE_TOKENS == {"mul", "add", "exp", "max", "copy"}
    # _infer_ops' codomain for bodies is exactly the token sets; a bodiless kernel is (None, None)
    for body in ("Y[n,h] = X[n,h] * 2.0", "Y[n] += X[n,h]", "Y[n] -= X[n]", "Y[n,h] = exp(X[n,h])", "Y[n,h] = X[n,h]"):
        e, r = ec._infer_ops(body)
        assert r in ec.BODY_REDUCTION_TOKENS and (e is None or e in ec.BODY_ELEMENTWISE_TOKENS) and ec.ops_prove_body(e, r)
    assert ec._infer_ops(None) == (None, None) and not ec.ops_prove_body(None, None)
    for bad in ("", "null", "opaque_unknown", None, 7):
        assert not ec.ops_prove_body(None, bad) and not ec.ops_prove_body("mul", bad)
    assert ec.loop_expr_is_body("Y[n,h] = X[n,h] * 2.0") and ec.loop_expr_is_body("Y[b, o] += X[b, k] * W[o, k]")
    for bad in (None, "", "   ", True, 7, ["Y[n,h] = X[n,h]"], {"body": "Y[n,h] = X[n,h]"}, "not an expression", "Y = X", "Y[0] = 1"):
        assert not ec.loop_expr_is_body(bad), bad


def _historical_graph():
    import copy, json
    return copy.deepcopy(json.loads(_HISTORICAL_UNRESOLVED.read_text())["raw"])


@pytest.mark.parametrize("extra,basis", [
    ({"loop_expr": "Y[n,h] = X[n,h] * 2.0"}, "shape_generic"),
    ({"reduction_op": "none", "elementwise_op": "mul"}, "shape_generic"),
    ({"reduction_op": "add", "elementwise_op": "exp"}, "shape_generic"),
    ({"builtin": "gather"}, "gather_elements"),
    ({"loop_iterations": 8, "loop_iterations_complete": True, "loop_iterations_omitted": []}, "loop_iterations"),
], ids=["real_expression", "body_none", "body_add", "table_builtin", "trusted_loop"])
def test_positive_legacy_evidence_stays_priced(tmp_path, extra, basis):
    # the review round-8 control (checks-59528a80/test_round8_legacy.py), verbatim
    g = _historical_graph(); l = g["layers"]["opaque_unknown#0"]; l.update(extra)
    if extra.get("reduction_op"):
        l.pop("loop_expr", None)
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    assert out["other_ops"] == 8 and out["cost_basis"] == basis
    assert a["total"].get("status") != "incomplete" and p["workload"].get("status") != "incomplete"


@pytest.mark.parametrize("extra", [
    {}, {"loop_expr": ""}, {"loop_expr": False}, {"builtin": "opaque_unknown"}, {"builtin": {"name": "gather"}},
    {"loop_iterations": 0, "loop_iterations_complete": True},
    {"loop_iterations": True, "loop_iterations_complete": True},
    {"loop_iterations": 8}, {"loop_iterations_omitted": ["while loop"]},
], ids=["fields_only", "empty_expression", "false_expression", "unknown_builtin", "dict_builtin", "zero_count", "bool_count", "unproven_count", "omissions_only"])
def test_absent_positive_legacy_evidence_is_incomplete(tmp_path, extra):
    # the review round-8 control, verbatim
    g = _historical_graph(); g["layers"]["opaque_unknown#0"].update(extra); a, p = _analyze_raw(tmp_path, g)
    assert a["layers"]["opaque_unknown#0"]["other_ops"] == 0
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"


@pytest.mark.parametrize("extra", [
    {"loop_expr": True}, {"loop_expr": 7}, {"loop_expr": ["Y[n,h] = X[n,h]"]},
    {"loop_expr": {"body": "Y[n,h] = X[n,h]"}},
    {"reduction_op": ""}, {"reduction_op": "opaque_unknown"}, {"reduction_op": "null"},
], ids=["bool_expression", "int_expression", "list_expression", "dict_expression", "empty_reduction", "unknown_reduction", "string_null_reduction"])
def test_malformed_body_evidence_does_not_price_unknown_kernel(tmp_path, extra):
    # the review round-8 control (the 7 refusals that failed at an earlier revision), verbatim: typed admission evidence must
    # identify a body, not just be truthy or any string
    g = _historical_graph(); g["layers"]["opaque_unknown#0"].update(extra); a, p = _analyze_raw(tmp_path, g)
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"
    assert a["layers"]["opaque_unknown#0"]["other_ops"] == 0
    assert a["layers"]["opaque_unknown#0"]["incomplete_reason"] == "dsl_kernel_unresolved_legacy"


# -- round 10 (an earlier review, LEGACY-TOKEN-16 / LEGACY-PARSE-15): total token predicates, ast-parsed bodies,
# fail-closed admission errors -------------------------------------------------------------------------------------

# the reviewer's independent boundary file (checks-da80de37/test_round9_independent.py), CASES verbatim
_ROUND9_INDEPENDENT_CASES = [
 ('syntax_missing_rhs',{'loop_expr':'Y[n,h] ='},False),
 ('syntax_garbage',{'loop_expr':'not valid Python Y[n,h] = X[n,h] !!!'},False),
 ('comparison_only',{'loop_expr':'Y[n,h] == X[n,h]'},False),
 ('keyword_call_only',{'loop_expr':'consume(value=X[n,h])'},False),
 ('quoted_text_only',{'loop_expr':'"Y[n,h] = X[n,h]"'},False),
 ('comment_only',{'loop_expr':'# Y[n,h] = X[n,h]'},False),
 ('no_loop_index',{'loop_expr':'Y[0] = X[0]'},False),
 ('no_index',{'loop_expr':'Y = X'},False),
 ('syntax_mixed',{'loop_expr':'Y[n,h] = X[n,h]; Z[n,h] ='},False),
 ('valid_assignment',{'loop_expr':'Y[n,h] = X[n,h] * 2.0'},True),
 ('reduction_upper',{'reduction_op':'ADD'},False),
 ('reduction_space',{'reduction_op':' add '},False),
 ('reduction_list',{'reduction_op':['add']},False),
 ('reduction_dict',{'reduction_op':{'op':'add'}},False),
 ('elementwise_list',{'reduction_op':'none','elementwise_op':['mul']},False),
 ('elementwise_dict',{'reduction_op':'none','elementwise_op':{'op':'mul'}},False),
 ('elementwise_upper',{'reduction_op':'none','elementwise_op':'MUL'},False),
 ('reduction_valid_add',{'reduction_op':'add'},True),
 ('reduction_valid_none_mul',{'reduction_op':'none','elementwise_op':'mul'},True),
 ('count_bool',{'loop_iterations':True,'loop_iterations_complete':True},False),
 ('count_float',{'loop_iterations':1.0,'loop_iterations_complete':True},False),
 ('count_string',{'loop_iterations':'8','loop_iterations_complete':True},False),
 ('count_int64_boundary',{'loop_iterations':2**63,'loop_iterations_complete':True},True),
]
# The three malformed-COUNT cases are refused by the loop-count gate (an earlier change round 4) with its own reason,
# ``loop_metadata_incomplete``; the review recorded that as "safely refuses" and kept its stricter assertion as a
# known failure.  The body/token cases must carry ``dsl_kernel_unresolved_legacy``.
_COUNT_GATE_CASES = {"count_bool", "count_float", "count_string"}


@pytest.mark.parametrize("name,extra,admit", _ROUND9_INDEPENDENT_CASES, ids=[c[0] for c in _ROUND9_INDEPENDENT_CASES])
def test_round9_independent_boundaries(tmp_path, name, extra, admit):
    g = _historical_graph(); g["layers"]["opaque_unknown#0"].update(extra)
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    if admit:
        assert a["total"].get("status") != "incomplete" and p["workload"].get("status") != "incomplete", out
        assert out["other_ops"] == extra.get("loop_iterations", 8), out
    else:
        assert a["total"].get("status") == p["workload"].get("status") == "incomplete", out
        assert out["other_ops"] == 0, out
        expected_reason = "loop_metadata_incomplete" if name in _COUNT_GATE_CASES else "dsl_kernel_unresolved_legacy"
        assert out["incomplete_reason"] == expected_reason, out


@pytest.mark.parametrize("expr", [
    "Y[0] += X[h] * W[h]",                                  # reduction into a fixed slot over a loop variable
    "S[p, h] = acc",                                        # local accumulator written to an indexed target
    "E[t, h] += math.exp(S[p, h] - max(M[t, h], _NEG))",    # attribute call + builtin call in the value
    "Y[b, kv, t, d] = X[b, kv, t, d] * COS[t, d // 2] + X[b, kv, t, p] * (2 * par - 1) * SIN[t, d // 2]",
    "M[t, h] = max(M[t, h], S[p, h]); PR[p, h] = math.exp(S[p, h] - M[t, h]) / E[t, h]",
], ids=["fixed_slot_reduction", "accumulator_value", "attribute_call", "rotary", "two_statements"])
def test_real_kernel_body_statements_are_bodies(expr):
    # statement forms taken from shipped verified DSL kernels: the ast validation must not refuse them
    from solar.einsum_dsl.einsum_converter import loop_expr_is_body
    assert loop_expr_is_body(expr), expr


@pytest.mark.parametrize("expr", [
    "Y[n,h] =", "not valid Python Y[n,h] = X[n,h] !!!", "Y[n,h] == X[n,h]", "consume(value=X[n,h])",
    '"Y[n,h] = X[n,h]"', "# Y[n,h] = X[n,h]", "Y[n,h] = X[n,h]; Z[n,h] =",            # the reviewer's seven
    "Y[0] = X[0]", "Y = X",                                                           # no loop variable / no index
    "Y[n,h] = 'text'", "Y[n,h] = X[n,h]; print(X)", "Y[n,h] = (lambda v: v)(X[n,h])",  # string value / call statement / lambda
    "Y[n,h] = f(X[n,h], k=1)", "Y[n,h] = [X[n,h]]", "Y[n,h] = X[n,h]\nimport os",   # keyword arg / container / import
    "Y[n,h] = X[n,h] if True else 0; Y[n,h].attr = 1", "Y.z[n] = X[n]", "del Y[n]",   # attribute target / del
    "Y[n,h] = X[n,h] @ W", "Y[n,h] = X[n,h] \x00", "   ", "",
], ids=["missing_rhs", "garbage", "compare", "kw_call", "quoted", "comment", "mixed", "no_loop_index", "no_index",
        "string_value", "call_statement", "lambda", "keyword_arg", "list_value", "import", "attribute_target",
        "attribute_name_target", "del", "matmul", "null_byte", "blank", "empty"])
def test_parse_validated_body_refuses_non_body_strings(expr):
    from solar.einsum_dsl.einsum_converter import loop_expr_is_body
    assert not loop_expr_is_body(expr), expr


def test_body_token_predicates_are_total_over_artifact_value_types():
    # legacy-token-16: no saved-artifact value may raise out of the predicates (containers used to TypeError
    # on set membership and the exception became an unmarked zero)
    from solar.einsum_dsl.einsum_converter import loop_expr_is_body, ops_prove_body
    values = [None, True, False, 0, 1, 2**63, 1.0, float("nan"), "", "add", "ADD", " add ", "none", "mul", b"add",
              ["add"], ("add",), {"op": "add"}, {"add"}, frozenset({"add"}), object(), type, range(3)]
    for v in values:
        assert loop_expr_is_body(v) is (v == "add" and False)  # only parsed indexed assignments are bodies
        for w in values:
            r = ops_prove_body(v, w)
            assert r is (isinstance(w, str) and (w in {"add", "sub"} or (w == "none" and isinstance(v, str) and v in {"mul", "add", "exp", "max", "copy"}))), (v, w)


def test_unexpected_admission_error_fails_closed_with_recorded_type(tmp_path, monkeypatch):
    # legacy-token-16 (b): any exception raised while judging legacy admission evidence is NOT a complete-looking
    # zero: the layer is incomplete (dsl_kernel_unresolved_legacy / incomplete_evidence), records the exception type,
    # and the total and perf workload are incomplete
    import solar.einsum_dsl.einsum_converter as ec

    def explode(*a, **k):
        raise RuntimeError("injected admission failure (control)")
    monkeypatch.setattr(ec, "ops_prove_body", explode)
    g = _historical_graph(); g["layers"]["opaque_unknown#0"].update({"reduction_op": "add"})
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    assert out["other_ops"] == 0 and out["macs"] == 0 and out["cost_basis"] == "incomplete_evidence"
    assert out["status"] == "incomplete" and out["incomplete_reason"] == "dsl_kernel_unresolved_legacy"
    assert out["admission_error"] == "RuntimeError"
    assert a["total"]["status"] == "incomplete" and p["workload"]["status"] == "incomplete"


# -- round 11 (an earlier review, LEGACY-MEMORY-17 / LEGACY-COMMENT-18): resource exhaustion propagates;
# loop-variable evidence comes from the validated AST only -----------------------------------------------------

# the reviewer's parser/pipeline cases (checks-4687a719/test_round10_independent.py), CASES verbatim
_ROUND10_PARSER_CASES = [
 ('rhs_call','Y[n] = f(X[n])',True),
 ('indexed_gather','Y[n] = X[Idx[n]]',True),
 ('chained_subscript','Y[n] = X[n][m]',False),
 ('attribute_target','Y.data[n] = X[n]',False),
 ('tuple_targets','Y[n], Z[n] = X[n]',False),
 ('starred_target','*Y[n], = X[n]',False),
 ('conditional_rhs','Y[n] = X[n] if c else 0',True),
 ('trailing_comment','Y[n] = X[n] # ordinary comment',True),
 ('windows_newlines','Y[n] = X[n]\r\nZ[n] = Y[n]\r\n',True),
 ('unicode_tensor','结果[n] = 输入[n]',True),
 ('unicode_index','Y[λ] = X[λ]',False),
 ('await_rhs','Y[n] = await f(X[n])',False),
 ('yield_rhs','Y[n] = (yield X[n])',False),
 ('yield_from_rhs','Y[n] = (yield from X[n])',False),
 ('no_tensor','y = x + 1',False),
 ('shipped_reduction','Y[0] += X[h] * W[h]',True),
 ('shipped_local_scalar','S[p,h] = acc',True),
 ('constant_index','Y[0] = X[0]',False),
 ('comment_phantom_index','Y[0] = X[0] # Y[n] = X[n]',False),
 ('comment_phantom_index_next_line','Y[0] = X[0]\n# Y[n] = X[n]',False),
]


@pytest.mark.parametrize("name,expr,admit", _ROUND10_PARSER_CASES, ids=[c[0] for c in _ROUND10_PARSER_CASES])
def test_round10_parser_pipeline(tmp_path, name, expr, admit):
    # loop-variable evidence is read from the validated AST, never from the artifact text: a comment cannot supply
    # a phantom index (legacy-comment-18); the exporter's loop-variable rule (ASCII identifier) still applies
    from solar.einsum_dsl.einsum_converter import loop_expr_is_body
    assert loop_expr_is_body(expr) is admit, expr
    g = _historical_graph(); g["layers"]["opaque_unknown#0"]["loop_expr"] = expr
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    assert (a["total"].get("status") != "incomplete") is admit, out
    assert (p["workload"].get("status") != "incomplete") is admit, out
    assert out["other_ops"] == (8 if admit else 0), out


def _fail(kind):
    if kind == "KeyError":
        return {}["required_body_field"]
    if kind == "UnicodeError":
        raise UnicodeError("malformed body decode")
    if kind == "MemoryError":
        raise MemoryError("synthetic resource exception")
    if kind == "RecursionError":
        def deep():
            return deep()
        return deep()
    raise AssertionError(kind)


@pytest.mark.parametrize("route", ["admission", "cost_legacy", "cost_marked"])
@pytest.mark.parametrize("kind", ["KeyError", "RecursionError", "UnicodeError", "MemoryError"])
def test_exception_contract_at_both_dsl_handlers(tmp_path, monkeypatch, route, kind):
    # legacy-memory-17: resource exhaustion PROPAGATES from the admission handler and from the cost handler
    # (the earlier change contract); malformed-input exceptions fail closed with the type recorded (the review cases verbatim)
    import solar.einsum_dsl.einsum_converter as converter
    from solar.einsum.analyzer import EinsumAnalyzer
    g = _historical_graph(); l = g["layers"]["opaque_unknown#0"]
    if route == "admission":
        monkeypatch.setattr(converter, "loop_expr_is_body", lambda *a, **k: _fail(kind))
    else:
        l["loop_expr"] = "Y[n] = X[n]"
        if route == "cost_marked":
            l.update(dsl_route=True, kernel_body=True)
        monkeypatch.setattr(EinsumAnalyzer, "get_compute_cost", lambda self, *a, **k: _fail(kind))
    if kind == "MemoryError":
        with pytest.raises(MemoryError):
            _analyze_raw(tmp_path, g)
        return
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    assert out["other_ops"] == 0 and out["macs"] == 0
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"
    assert out["cost_basis"] == "incomplete_evidence" and out["admission_error"] == kind
    assert out["incomplete_reason"] == ("dsl_kernel_unresolved" if route == "cost_marked" else "dsl_kernel_unresolved_legacy")


def test_memory_error_in_a_mechanical_layer_cost_propagates(tmp_path, monkeypatch):
    # the outer cost handler re-raises resource exhaustion for every layer; the mechanical zero-without-marker on
    # OTHER exceptions is target-owned (TARGET-MECHANICAL-EXCEPTION-19) and unchanged here
    from solar.einsum.analyzer import EinsumAnalyzer
    g = _historical_graph(); g["layers"]["opaque_unknown#0"]["type"] = "relu"
    monkeypatch.setattr(EinsumAnalyzer, "get_compute_cost", lambda self, *a, **k: _fail("MemoryError"))
    with pytest.raises(MemoryError):
        _analyze_raw(tmp_path, g)


def test_deep_expression_string_fails_closed_with_recursion_error(tmp_path):
    # the reviewer's deep-string probe: 1100 chained terms exhaust the validator's recursion -> refused, type recorded
    from solar.einsum_dsl.einsum_converter import loop_expr_is_body
    expr = "Y[n] = " + "+".join(["X[n]"] * 1100)
    try:
        direct = loop_expr_is_body(expr)
    except RecursionError:
        direct = "RecursionError"
    g = _historical_graph(); g["layers"]["opaque_unknown#0"]["loop_expr"] = expr
    a, p = _analyze_raw(tmp_path, g); out = a["layers"]["opaque_unknown#0"]
    assert a["total"].get("status") == p["workload"].get("status") == "incomplete"
    assert out["other_ops"] == 0 and (out.get("admission_error") == "RecursionError" or direct is False), (direct, out)


# -- round 7 (CAPTURE-EVIDENCE-11): the evidence builder is guarded at the capture-shim boundary --------------

def test_bare_kernel_record_without_dsl_provenance_is_incomplete_not_priced(tmp_path):
    # a hand-built einsum-graph layer of type "kernel" with no DSL fields at all (no marker, no loop_expr /
    # loop_iterations / equation_faithful) and no unrecognised_op: the analyzer must not price it from its shapes
    g = {"model_name": "bare_kernel", "layers": {
        "start": {"type": "start", "einsum_equation": "AB->AB", "is_real_einsum": False, "elementwise_op": "copy", "reduction_op": "none",
                  "tensor_names": {"inputs": [], "outputs": ["x"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                  "tensor_shapes": {"inputs": [], "outputs": [[2, 4]]}, "tensor_dtypes": {"inputs": [], "outputs": ["fp32"]},
                  "connections": {"inputs": [], "outputs": ["kernel#0"]}},
        "kernel#0": {"type": "kernel", "einsum_equation": "AB->AB", "is_real_einsum": False, "elementwise_op": None, "reduction_op": None,
                     "tensor_names": {"inputs": ["x"], "outputs": ["y"]}, "tensor_types": {"inputs": ["input"], "outputs": ["input"]},
                     "tensor_shapes": {"inputs": [[2, 4]], "outputs": [[2, 4]]}, "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
                     "connections": {"inputs": ["start"], "outputs": []}}}}
    (tmp_path / "einsum").mkdir()
    gpath = tmp_path / "einsum" / "einsum_graph.yaml"
    gpath.write_text(yaml.safe_dump(g, sort_keys=False))
    a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
    layer = a["layers"]["kernel#0"]
    assert layer["macs"] == 0 and layer["other_ops"] == 0
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "dsl_kernel_unresolved"
    assert a["total"]["status"] == "incomplete"


def test_capture_evidence_builder_exception_keeps_capture_incomplete(tmp_path, monkeypatch):
    # reviewer control: an exception inside interpolate_evidence_record at CAPTURE time must not abort the capture;
    # the node carries a structured error record and the consumer refuses it
    import solar.einsum.op_families as families

    def explode(*a, **k):
        raise TypeError("injected capture evidence failure (control)")
    monkeypatch.setattr(families, "interpolate_evidence_record", explode)
    g, e, a, p, ops = _capture_source(tmp_path, "return F.interpolate(x, size=95, mode='linear')")
    node = next(v for v in g["layers"].values() if v.get("type") == "interpolate")
    err = node["module_args"]["interpolate_evidence_error"]
    assert err["type"] == "TypeError" and "injected capture evidence failure" in err["message"] and "interpolate_evidence" not in node["module_args"]
    layer = next(v for v in ops.values() if v.get("unrecognised_op") == "interpolate")
    assert layer["interpolate_refusal"].startswith("interpolate_evidence_error: TypeError: capture-time evidence builder failed")
    assert not any(v.get("interpolate_identity_copy") for v in ops.values())
    assert a["total"]["status"] == p["workload"]["status"] == "incomplete"


# -- round 6 (an earlier review, SCHEMA-9): nested types before membership; no path to the fallback ---

def _pipeline_record(tmp_path, ev):
    g = _recorded_graph("interpolate", [SH], SH)
    g["layers"]["Model.interpolate"]["module_args"]["interpolate_evidence"] = ev
    e, a, p = _run(tmp_path, g)
    return e["layers"]["Model.interpolate"], a, p


def test_nested_bound_as_wrong_type_refuses_without_crashing(tmp_path):
    # reviewer control (checks-fcef99e8/test_round5.py): a list bound_as with is_torch_library True used to raise
    # TypeError (unhashable) out of the membership test and fall into the permissive supportability fallback
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"]["bound_as"] = ["torch.nn.functional.interpolate"]
    why = interpolate_record_refusal(ev)
    assert why == "interpolate_record_malformed: callee_identity.bound_as has type list"
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert a["total"]["status"] == "incomplete"


@pytest.mark.parametrize("bad", [["torch.nn.functional.interpolate"], {"name": "torch.nn.functional.interpolate"}])
def test_wrong_typed_nested_binding_keeps_pipeline_incomplete(tmp_path, bad):
    # reviewer control (checks-fcef99e8/test_schema_pipeline.py), both cases
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"]["bound_as"] = bad
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert layer.get("unrecognised_op") == "interpolate"
    assert a["total"]["status"] == "incomplete" and p["workload"]["status"] == "incomplete"
    assert "interpolate_identity_copy" not in layer and layer["interpolate_refusal"].startswith("interpolate_record_malformed")
    assert layer["is_einsum_supportable"] is False and not layer.get("is_real_einsum")


@pytest.mark.parametrize("field,bad", [
    ("module", 5), ("module", None), ("qualname", ["interpolate"]), ("is_torch_library", "yes"),
    ("is_torch_library", 1), ("bound_as", {"name": "x"}), ("bound_as", None),
])
def test_malformed_nested_admission_fields_refuse_by_name(tmp_path, field, bad):
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"][field] = bad
    why = interpolate_record_refusal(ev)
    assert why is not None and why.startswith("interpolate_record_malformed: callee_identity." + field), why
    assert identity_interpolate_evidence(ev, SH, SH) is None
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert layer.get("unrecognised_op") == "interpolate" and a["total"]["status"] == "incomplete"


def test_diagnostic_type_field_is_an_observation_not_an_admission_field(tmp_path):
    # reviewer control (test_schema_pipeline.py): `type` is diagnostic-only; its omission does not refuse
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"].pop("type")
    assert interpolate_record_refusal(ev) is None
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert layer.get("interpolate_identity_copy") and a["total"].get("status") != "incomplete"
    ev["callee_identity"]["type"] = 17                     # wrong-typed diagnostic: still not an admission failure
    assert interpolate_record_refusal(ev) is None


def test_non_record_callee_identity_is_malformed(tmp_path):
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    ev["callee_identity"] = "torch.nn.functional.interpolate"
    assert interpolate_record_refusal(ev) == "interpolate_record_malformed: callee_identity has type str (expected a record)"
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert layer.get("unrecognised_op") == "interpolate" and a["total"]["status"] == "incomplete"


def test_any_exception_on_the_evidence_path_lands_on_the_incomplete_path(tmp_path, monkeypatch):
    # regression control: inject a TypeError exactly where the evidence is weighed; the layer must be
    # unrecognised/incomplete with the exception recorded, never the permissive supportability fallback
    import solar.einsum.pytorch_to_einsum as p2e

    def boom(evidence, input_shape, output_shape):
        raise TypeError("unhashable type: 'list'")
    monkeypatch.setattr(p2e, "identity_interpolate_evidence", boom)
    ev = interpolate_evidence_record(F.interpolate, (object(),), {"size": 95, "mode": "linear"})
    layer, a, p = _pipeline_record(tmp_path, ev)
    assert layer.get("unrecognised_op") == "interpolate"
    assert layer["interpolate_refusal"] == "interpolate_evidence_error: TypeError: unhashable type: 'list'"
    assert "interpolate_identity_copy" not in layer
    assert layer["is_einsum_supportable"] is False and not layer.get("is_real_einsum")
    assert a["total"]["status"] == "incomplete" and p["workload"]["status"] == "incomplete"
    assert a["total"]["macs"] == 0


def test_ci_unit_lane_selects_this_module():
    ci = next(p / ".gitlab-ci.yml" for p in Path(__file__).resolve().parents if (p / ".gitlab-ci.yml").is_file())
    assert "tests/test_unknown_op_fail_closed.py" in ci.read_text()
