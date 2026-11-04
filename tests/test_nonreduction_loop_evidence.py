# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issue #52: DSL kernels are billed from the converter's VALIDATED
loop-iteration evidence through ONE shared gate, not the generic shape-based
cost (119-SOL-SCALE-1; round 2 of an earlier change: loop-evidence-1,
density-composition-1, cost-basis-composition-1).

* A DSL kernel whose ``loop_iterations`` is trusted by an earlier fix's shared gate
  ``dsl_loop_count_evidence`` (positive int, ``loop_iterations_complete:
  true``, no omissions; imported, never re-derived) bills its count — other_ops for a non-reduction kernel, MACs for a contraction — and
  records ``cost_basis: loop_iterations`` (119 ``scale_by_routing``:
  68,157,440, not 580,679,578,419,200).
* A count that is present but not validated (complete=false, omissions, no
  proof, malformed) keeps the generic/equation cost AND the
  ``loop_metadata_incomplete`` marker; it is never silently used.
* The validated count is selected once, density-discounted once, then split
  into MACs / other_ops; raw ``loop_iterations`` is never re-read after the
  discount (X*M at density 0.25 bills 2 of 8).
* Mechanical-route layers carry no ``cost_basis`` field at all, so dense
  analyses stay byte-identical.
* the independent reviewer's round-1 probes are included verbatim (DSL sources unchanged).

No comparator, tolerance or protected assertion is touched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SOLAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR))

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer, dsl_loop_count_evidence  # noqa: E402


def loop_evidence_validated(meta) -> bool:
    """The ONE gate is an earlier fix's ``dsl_loop_count_evidence`` (imported, not
    re-derived); this test-local predicate is its ``trusted`` flag."""
    return dsl_loop_count_evidence(meta if isinstance(meta, dict) else {}).trusted

# 119 shape 0: T2080, H4096, K8 (E256, I2048 do not enter this kernel)
T119, H119, K119 = 2080, 4096, 8
SCALE_BY_ROUTING_ITERS = T119 * K119 * H119          # 68,157,440
GENERIC_SHAPE_COST_119 = 580_679_578_419_200           # what the analyzer billed


# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------

def _kernel_layer(equation, elementwise_op, reduction_op, is_real, in_shapes,
                  out_shape, in_names, out_name, loop_iterations=None,
                  loop_expr=None, dtype="fp32", complete=True, omitted=None):
    layer = {
        "type": "kernel",
        "einsum_equation": equation,
        "elementwise_op": elementwise_op,
        "reduction_op": reduction_op,
        "is_real_einsum": is_real,
        "equation_faithful": True,
        "tensor_names": {"inputs": list(in_names), "outputs": [out_name]},
        "tensor_types": {"inputs": ["input"] * len(in_names), "outputs": ["input"]},
        "tensor_shapes": {"inputs": [list(s) for s in in_shapes],
                          "outputs": [list(out_shape)]},
        "tensor_dtypes": {"inputs": [dtype] * len(in_names), "outputs": [dtype]},
        "connections": {"inputs": ["start"], "outputs": []},
    }
    if loop_expr:
        layer["loop_expr"] = loop_expr
    if loop_iterations is not None:
        layer["loop_iterations"] = loop_iterations
        # the converter's completeness proof; the shared gate needs it
        if complete is not None:
            layer["loop_iterations_complete"] = complete
    if omitted:
        layer["loop_iterations_omitted"] = list(omitted)
    return layer


def _start_layer(names, shapes, consumers, dtype="fp32"):
    labels = []
    for shape in shapes:
        labels.append([f"{c}{i}" for i, c in enumerate("ABCDEFG"[:len(shape)])])
    return {
        "type": "start",
        "einsum_equation": "->" + ",".join("".join(lb) for lb in labels),
        "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False,
        "tensor_names": {"inputs": [], "outputs": list(names)},
        "tensor_types": {"inputs": [], "outputs": ["input"] * len(names)},
        "tensor_shapes": {"inputs": [], "outputs": [list(s) for s in shapes]},
        "tensor_dtypes": {"inputs": [], "outputs": [dtype] * len(names)},
        "connections": {"inputs": [], "outputs": list(consumers)},
    }


def _write_graph(tmp_path, layers, name="toy"):
    graph = {"model_name": name, "layers": layers}
    path = tmp_path / f"{name}_einsum_graph.yaml"
    path.write_text(yaml.safe_dump(graph, sort_keys=False))
    return path


def _analyze_graph(tmp_path, graph_path, precision="fp32"):
    out = EinsumGraphAnalyzer().analyze_graph(
        str(graph_path), str(tmp_path / "analysis"), precision=precision, copy_graph=False)
    assert out is not None and "layers" in out
    return out


# ---------------------------------------------------------------------------
# 1. the 119 scale_by_routing case at shape 0 (synthetic graph, exact shapes)
# ---------------------------------------------------------------------------

def test_119_scale_by_routing_bills_loop_iterations(tmp_path):
    layers = {
        "start": _start_layer(["grad_output", "topk_weights"],
                              [(T119, H119), (T119, K119)], ["scale_by_routing#0"]),
        "scale_by_routing#0": _kernel_layer(
            "TH,TK->TKH", "mul", "none", False,
            [(T119, H119), (T119, K119)], (T119, K119, H119),
            ["grad_output", "topk_weights"], "scale_by_routing#0",
            loop_iterations=SCALE_BY_ROUTING_ITERS,
            loop_expr="Y[t, k, h] = GO[t, h] * TW[t, k]"),
    }
    out = _analyze_graph(tmp_path, _write_graph(tmp_path, layers, "p119"), precision="fp16")
    layer = out["layers"]["scale_by_routing#0"]
    assert layer["other_ops"] == SCALE_BY_ROUTING_ITERS == 68_157_440
    assert layer["other_ops"] != GENERIC_SHAPE_COST_119
    assert layer["macs"] == 0 and layer["flops"] == 0
    assert layer["cost_basis"] == "loop_iterations"
    assert "status" not in layer
    assert out["total"]["other_ops"] == SCALE_BY_ROUTING_ITERS
    assert out["total"]["dsl_cost_basis_layers"] == {"loop_iterations": ["scale_by_routing#0"]}


def test_119_without_evidence_reproduces_the_generic_cost_and_marks_it(tmp_path):
    """The same kernel saved by an older converter (no loop_iterations) keeps
    the generic shape-based cost — exactly the 580,679,578,419,200 the issue
    reports — and the receipt says so."""
    layers = {
        "start": _start_layer(["grad_output", "topk_weights"],
                              [(T119, H119), (T119, K119)], ["scale_by_routing#0"]),
        "scale_by_routing#0": _kernel_layer(
            "TH,TK->TKH", "mul", "none", False,
            [(T119, H119), (T119, K119)], (T119, K119, H119),
            ["grad_output", "topk_weights"], "scale_by_routing#0"),
    }
    out = _analyze_graph(tmp_path, _write_graph(tmp_path, layers, "p119_noev"), precision="fp16")
    layer = out["layers"]["scale_by_routing#0"]
    assert layer["other_ops"] == GENERIC_SHAPE_COST_119
    assert layer["cost_basis"] == "shape_generic"
    assert out["total"]["dsl_cost_basis_layers"] == {"shape_generic": ["scale_by_routing#0"]}


@pytest.mark.parametrize("bad", [0, -5, True, "not-a-count"])
def test_non_positive_or_non_int_evidence_is_not_evidence(tmp_path, bad):
    # (the exact type validation of loop_iterations is an earlier fix's gate contract,
    # tested there; here only clearly malformed values are exercised)
    layers = {
        "start": _start_layer(["a", "b"], [(4, 8), (4, 2)], ["k#0"]),
        "k#0": _kernel_layer("TH,TK->TKH", "mul", "none", False,
                             [(4, 8), (4, 2)], (4, 2, 8), ["a", "b"], "k#0",
                             loop_iterations=bad),
    }
    out = _analyze_graph(tmp_path, _write_graph(tmp_path, layers, "bad"))
    assert out["layers"]["k#0"]["cost_basis"] == "shape_generic"
    # a count that is present but not validated is marked, never silently used
    assert out["layers"]["k#0"]["status"] == "incomplete"
    assert out["layers"]["k#0"]["incomplete_reason"] == "loop_metadata_incomplete"
    # the generic shape-based rule: the (4,8) operand does not broadcast-align
    # with the (4,2,8) output, so its dims get fresh labels and the rank
    # product is numel(in0) * numel(out) — the same rule that billed 119's
    # kernel 2080*4096 * 2080*8*4096 = 580,679,578,419,200
    assert out["layers"]["k#0"]["other_ops"] == (4 * 8) * (4 * 2 * 8)


def test_unproven_count_keeps_generic_cost_with_marker(tmp_path):
    """complete=false (or omissions) with a positive count: the count is
    rejected on BOTH sides, the generic cost and the marker stay."""
    layers = {
        "start": _start_layer(["a", "b"], [(4, 8), (4, 2)], ["k#0"]),
        "k#0": _kernel_layer("TH,TK->TKH", "mul", "none", False,
                             [(4, 8), (4, 2)], (4, 2, 8), ["a", "b"], "k#0",
                             loop_iterations=1, complete=False, omitted=["range(7) unrepresented"]),
    }
    out = _analyze_graph(tmp_path, _write_graph(tmp_path, layers, "unproven"))
    layer = out["layers"]["k#0"]
    assert layer["other_ops"] == (4 * 8) * (4 * 2 * 8) and layer["other_ops"] != 1
    assert layer["cost_basis"] == "shape_generic"
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
    assert layer["loop_iterations_omitted"] == ["range(7) unrepresented"]


# ---------------------------------------------------------------------------
# 2. end to end through the real converter: trace -> einsum graph -> analysis
# ---------------------------------------------------------------------------

T, K, H = 16, 4, 32

SCALE_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale_by_routing(GO: Tensor("T,H", dtype="fp32"),
                     TW: Tensor("T,K", dtype="fp32")) -> Tensor("T,K,H"):
    Y = Tensor(T=GO.shape[0], K=TW.shape[1], H=GO.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for k in Dim("K"):
            for h in Dim("H"):
                Y[t, k, h] = GO[t, h] * TW[t, k]
    return Y

def run_model(inputs):
    clear_dims()
    go = Tensor.from_numpy(inputs["go"], "T,H", dtype="fp32")
    tw = Tensor.from_numpy(inputs["tw"], "T,K", dtype="fp32")
    return {"output": scale_by_routing(go, tw).data}
'''

# two statements per (t,h): a multi-statement (two-chain) non-reduction body
TWO_CHAIN_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale_then_shift(X: Tensor("T,H", dtype="fp32"),
                     S: Tensor("H", dtype="fp32"),
                     B: Tensor("H", dtype="fp32")) -> Tensor("T,H"):
    Y = Tensor(T=X.shape[0], H=X.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for h in Dim("H"):
            Y[t, h] = X[t, h] * S[h]
            Y[t, h] = Y[t, h] + B[h]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "T,H", dtype="fp32")
    s = Tensor.from_numpy(inputs["s"], "H", dtype="fp32")
    b = Tensor.from_numpy(inputs["b"], "H", dtype="fp32")
    return {"output": scale_then_shift(x, s, b).data}
'''

# a real (MAC) kernel with two accumulation chains
TWO_ACC_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def two_proj(X: Tensor("T,H", dtype="fp32"),
             W1: Tensor("H,I", role="weight", dtype="fp32"),
             W2: Tensor("H,I", role="weight", dtype="fp32")) -> Tensor("T,I"):
    Y = Tensor(T=X.shape[0], I=W1.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for i in Dim("I"):
            for h in Dim("H"):
                Y[t, i] += X[t, h] * W1[h, i]
                Y[t, i] += X[t, h] * W2[h, i]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "T,H", dtype="fp32")
    w1 = Tensor.from_numpy(inputs["w1"], "H,I", dtype="fp32")
    w2 = Tensor.from_numpy(inputs["w2"], "H,I", dtype="fp32")
    return {"output": two_proj(x, w1, w2).data}
'''


def _traced_analysis(code, inputs, tmp_path, stem="toy_einsum_lib"):
    """dsl (py+yaml) -> einsum graph -> analysis dict, the real pipeline
    (same steps as tests/test_sparse_cost.py)."""
    from solar.einsum_lib import build_metadata, get_trace
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_lib.verify import execute_translated
    from solar.einsum_dsl.cli_einsum import translate_with_einsum

    execute_translated(code, {k: v.shape for k, v in inputs.items()},
                       all_inputs=inputs, trace=True, shape_only=False, fast=False)
    md = build_metadata(get_trace())
    dsl_dir = tmp_path / "dsl"
    dsl_dir.mkdir(parents=True, exist_ok=True)
    (dsl_dir / f"{stem}.py").write_text(code)
    write_metadata_yaml(md, str(dsl_dir / f"{stem}.yaml"))
    graph_path = translate_with_einsum(str(dsl_dir), str(tmp_path / "einsum"))
    graph = yaml.safe_load(Path(graph_path).read_text())
    out = EinsumGraphAnalyzer().analyze_graph(str(graph_path), str(tmp_path / "analysis"),
                                              precision="fp32", copy_graph=False)
    assert out is not None and "layers" in out
    return graph, out


def test_traced_scale_by_routing_is_one_multiply_per_tkh(tmp_path):
    rng = np.random.default_rng(0)
    inputs = {"go": rng.standard_normal((T, H)), "tw": rng.standard_normal((T, K))}
    graph, out = _traced_analysis(SCALE_DSL, inputs, tmp_path)
    g = graph["layers"]["scale_by_routing#0"]
    assert g["is_real_einsum"] is False and g["reduction_op"] == "none"
    assert g["loop_iterations"] == T * K * H and g["loop_iterations_complete"] is True
    assert loop_evidence_validated(g)
    layer = out["layers"]["scale_by_routing#0"]
    assert layer["other_ops"] == T * K * H == g["loop_iterations"]
    assert layer["cost_basis"] == "loop_iterations"
    assert layer["macs"] == 0 and "status" not in layer
    # the generic rank product would have been T*H*T*K... — strictly larger
    assert layer["other_ops"] < T * H * K * T


def test_traced_two_statement_kernel_bills_one_op_per_statement_execution(tmp_path):
    rng = np.random.default_rng(1)
    inputs = {"x": rng.standard_normal((T, H)), "s": rng.standard_normal(H),
              "b": rng.standard_normal(H)}
    graph, out = _traced_analysis(TWO_CHAIN_DSL, inputs, tmp_path)
    g = graph["layers"]["scale_then_shift#0"]
    assert g["is_real_einsum"] is False
    assert g["loop_iterations"] == 2 * T * H       # two statements per (t,h)
    layer = out["layers"]["scale_then_shift#0"]
    assert layer["other_ops"] == 2 * T * H
    assert layer["cost_basis"] == "loop_iterations"


def test_traced_two_accumulation_chain_real_kernel_uses_the_same_gate(tmp_path):
    """A real contraction with two accumulation chains: the ONE shared gate
    selects the validated loop count for the MAC side too (2 chains x T*I*H),
    other_ops stays 0 and the basis records loop_iterations
    (cost-basis-composition-1)."""
    rng = np.random.default_rng(2)
    I = 8
    inputs = {"x": rng.standard_normal((T, H)), "w1": rng.standard_normal((H, I)),
              "w2": rng.standard_normal((H, I))}
    graph, out = _traced_analysis(TWO_ACC_DSL, inputs, tmp_path)
    g = graph["layers"]["two_proj#0"]
    assert g["is_real_einsum"] is True
    assert g["loop_iterations"] == 2 * T * I * H    # weighted: two chains
    assert g["loop_iterations_complete"] is True
    layer = out["layers"]["two_proj#0"]
    assert layer["other_ops"] == 0
    assert layer["macs"] == layer["macs_dense"] == 2 * T * I * H
    assert layer["cost_basis"] == "loop_iterations" and "status" not in layer


# ---------------------------------------------------------------------------
# 3. mechanical-route layers are untouched (no cost_basis field at all)
# ---------------------------------------------------------------------------

def test_mechanical_layers_carry_no_cost_basis(tmp_path):
    layers = {
        "start": _start_layer(["x"], [(2, 4)], ["Model.relu"]),
        "Model.relu": {
            "type": "relu", "einsum_equation": "AB->AB", "elementwise_op": "max",
            "reduction_op": "none", "is_real_einsum": False,
            "tensor_names": {"inputs": ["x"], "outputs": ["Model.relu"]},
            "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
            "tensor_shapes": {"inputs": [[2, 4]], "outputs": [[2, 4]]},
            "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
            "connections": {"inputs": ["start"], "outputs": []},
            # a stray field on a NON-kernel layer must not switch the basis
            "loop_iterations": 1, "loop_iterations_complete": True,
        },
    }
    out = _analyze_graph(tmp_path, _write_graph(tmp_path, layers, "mech"))
    layer = out["layers"]["Model.relu"]
    assert "cost_basis" not in layer
    assert layer["other_ops"] == 8          # generic: one op per element, as before
    assert "status" not in layer
    assert "dsl_cost_basis_layers" not in out["total"]


# ---------------------------------------------------------------------------
# 4. the shared gate (loop-evidence-1) as a unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("meta, ok", [
    ({"loop_iterations": 8, "loop_iterations_complete": True}, True),
    ({"loop_iterations": 8, "loop_iterations_complete": True, "equation_faithful": False}, True),
    ({"loop_iterations": 8, "loop_iterations_complete": False}, False),
    ({"loop_iterations": 8}, False),                                   # proof not recorded
    ({"loop_iterations": 8, "loop_iterations_complete": True,
      "loop_iterations_omitted": ["for j in range(7): not a Dim() loop"]}, False),
    ({"loop_iterations": 0, "loop_iterations_complete": True}, False),
    ({"loop_iterations": True, "loop_iterations_complete": True}, False),
    ({"loop_iterations": "eight", "loop_iterations_complete": True}, False),
    # (type validation of loop_iterations — numeric strings, fractional and
    # non-finite values — is an earlier fix's gate contract and is tested there, not here)
    ({"loop_iterations_complete": False, "loop_iterations_omitted": ["while"]}, False),
    ({}, False), (None, False),
])
def test_loop_evidence_validated_gate(meta, ok):
    """an earlier fix's gate as consumed here: ``trusted`` is the activation predicate;
    ``equation_faithful`` is reported, never gated on."""
    assert loop_evidence_validated(meta) is ok
    if isinstance(meta, dict) and meta.get("loop_iterations") is not None:
        ev = dsl_loop_count_evidence(meta)
        assert ev.present and (ev.count > 0) is ok and (not ev.reasons) is ok


# ---------------------------------------------------------------------------
# 5. the independent reviewer's round-1 probes, verbatim DSL sources
# ---------------------------------------------------------------------------

_PROBE_HEADER = 'from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
_PROBE_PREFIX = '''@kernel
def probe(X:Tensor("T,H"))->Tensor("T,H"):
 Y=Tensor(T=X.shape[0],H=X.shape[1])
 for t in Dim("T",parallel=True):
  for h in Dim("H"):
'''
_PROBE_SUFFIX = ''' return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"T,H")
 return {"out":probe(x).data}
'''
PROBE_RANGE7 = _PROBE_HEADER + _PROBE_PREFIX + '   for j in range(7):\n    Y[t,h] += X[t,h]\n' + _PROBE_SUFFIX
PROBE_OUTER_STATEMENTS = _PROBE_HEADER + _PROBE_PREFIX + '   s=X[t,h]+1\n   for j in Dim("J",size=3):\n    s=s+2\n   Y[t,h]=s\n' + _PROBE_SUFFIX
PROBE_TWO_STATEMENTS = _PROBE_HEADER + _PROBE_PREFIX + '   Y[t,h]=X[t,h]*2\n   Y[t,h]=Y[t,h]+1\n' + _PROBE_SUFFIX
PROBE_MASK = _PROBE_HEADER + '''@kernel
def probe(X:Tensor("T,H"),M:Tensor("T,H"))->Tensor("T,H"):
 Y=Tensor(T=X.shape[0],H=X.shape[1])
 for t in Dim("T",parallel=True):
  for h in Dim("H"):
   Y[t,h]=X[t,h]*M[t,h]
 return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"T,H")
 m=Tensor.from_numpy(inputs["m"],"T,H",sparsity=.25)
 return {"out":probe(x,m).data}
'''


def _probe(code, inputs, tmp_path, expected_value):
    from solar.einsum_lib.verify import execute_translated
    outputs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()},
                                    all_inputs=inputs, trace=False, shape_only=False, fast=False)
    assert np.all(outputs["out"] == expected_value)      # the probe's numerical check
    graph, out = _traced_analysis(code, inputs, tmp_path, stem="probe_einsum_lib")
    return graph["layers"]["probe#0"], out["layers"]["probe#0"]


def test_probe_range7_inner_additions_are_incomplete_or_correct(tmp_path):
    """56 executed additions; the Dim() paths cannot represent range(7).
    Required: the exact count OR the generic cost WITH the incomplete marker —
    never a silent 8 labelled loop_iterations."""
    g, a = _probe(PROBE_RANGE7, {"x": np.ones((2, 4))}, tmp_path, 7)
    assert g["loop_iterations_complete"] is False and g["loop_iterations_omitted"]
    assert not loop_evidence_validated(g)
    assert a["other_ops"] == 56 or (a["status"] == "incomplete"
                                    and a["incomplete_reason"] == "loop_metadata_incomplete")
    assert a["cost_basis"] != "loop_iterations"
    assert a["loop_iterations_omitted"] == g["loop_iterations_omitted"]


def test_probe_two_statements_bills_one_op_per_statement(tmp_path):
    g, a = _probe(PROBE_TWO_STATEMENTS, {"x": np.ones((2, 4))}, tmp_path, 3)
    assert g["loop_iterations"] == 16 and g["loop_iterations_complete"] is True
    assert a["other_ops"] == 16 and a["cost_basis"] == "loop_iterations" and "status" not in a


def test_probe_explicit_incomplete_count_is_rejected(tmp_path):
    """A saved graph marked complete=false / omitted / faithful=false with
    count=1 keeps the generic cost (8) AND the incomplete marker; the
    rejected count is never reintroduced on the other_ops side."""
    graph, _ = _traced_analysis(PROBE_TWO_STATEMENTS, {"x": np.ones((2, 4))}, tmp_path,
                                stem="probe_einsum_lib")
    bad = yaml.safe_load(yaml.safe_dump(graph))
    bad["layers"]["probe#0"].update(loop_iterations=1, loop_iterations_complete=False,
                                    loop_iterations_omitted=["range(7) unrepresented"],
                                    equation_faithful=False)
    path = tmp_path / "explicit_incomplete.yaml"
    path.write_text(yaml.safe_dump(bad, sort_keys=False))
    out = _analyze_graph(tmp_path / "ei", path)
    a = out["layers"]["probe#0"]
    assert a["other_ops"] == 8 and a["other_ops"] != 1
    assert a["cost_basis"] == "shape_generic"
    assert a["status"] == "incomplete" and a["incomplete_reason"] == "loop_metadata_incomplete"
    assert out["total"]["status"] == "incomplete"


def test_probe_mask_density_is_applied_once_to_the_selected_count(tmp_path):
    """X*M with M at density 0.25: the validated count (8) is discounted once
    (2) and that single result is what other_ops carries, with
    compute_density 0.25 and basis loop_iterations (density-composition-1)."""
    mask = np.zeros((2, 4))
    mask[:, 0] = 1
    g, a = _probe(PROBE_MASK, {"x": np.ones((2, 4)), "m": mask}, tmp_path, mask)
    assert g["loop_iterations"] == 8 and g["loop_iterations_complete"] is True
    assert a["other_ops"] == 2 and a["compute_density"] == 0.25
    assert a["cost_basis"] == "loop_iterations" and a["macs"] == 0 and "status" not in a


def test_probe_outer_statements_are_incomplete_or_correct(tmp_path):
    """statement-coverage-1: arithmetic outside the innermost Dim() nest
    (``s=X[t,h]+1`` and ``Y[t,h]=s`` once per (t,h), ``s=s+2`` three times) is
    executed in-loop work: 40 assignment executions.  an earlier review round's parser
    records the outer-level statements and the converter sizes them, so the
    kernel is counted exactly; the contract is incomplete-or-correct, never a
    silent 24."""
    g, a = _probe(PROBE_OUTER_STATEMENTS, {"x": np.ones((2, 4))}, tmp_path, 8)
    assert a["other_ops"] != 24 or a.get("status") == "incomplete"
    assert a["other_ops"] == 40 or a.get("status") == "incomplete"
    if a["other_ops"] == 40:
        assert a["cost_basis"] == "loop_iterations" and "status" not in a
