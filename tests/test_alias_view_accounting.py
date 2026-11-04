# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alias views are free in unfused mode; real copies stay charged (issue #35, an earlier decision).

``tensor.detach()`` (and the other pure aliases listed in
``solar.analysis.graph_analyzer.ALIAS_OPS``) returns a view of the same
storage: no ALU work and no DRAM traffic.  The analyzer used to bill a
``detach`` layer one op per element plus a full read and write in unfused mode
(KernelBench level2/20 and level2/40: ``x.clone().detach()``).  These tests run
the real mechanical route on tiny synthetic modules (CPU capture, no download)
and pin the convention documented in ``docs/EINSUM_GUIDE.md``:

* alias layers carry zero ``other_ops`` and zero ``unfused_elements``;
* ``clone`` (a real copy) keeps its read + write traffic;
* MACs and the fused numbers are unchanged by the convention, including a
  model output that is returned through an alias (the DRAM write survives);
* graph connectivity through the alias is preserved (producers and consumers
  still resolve across it).
"""
import re
import textwrap
from pathlib import Path

import pytest
import torch

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.pytorch_processor import PyTorchProcessor

M, K, N = 64, 48, 80

ALIAS_CHAIN_MODEL = textwrap.dedent(f'''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """real op -> real copy -> alias -> alias -> real op"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear({K}, {N}, bias=False)

        def forward(self, x):
            y = self.fc(x)            # M*K*N MACs
            c = y.clone()             # real copy: read + write
            d = c.detach()            # alias: free
            r = d.reshape({N}, {M})   # alias: free
            return r * 2.0            # M*N elementwise ops

    def get_inputs():
        return [torch.randn({M}, {K})]

    def get_init_inputs():
        return []
''')

ALIAS_AT_OUTPUT_MODEL = textwrap.dedent(f'''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """the model output is returned through an alias"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear({K}, {N}, bias=False)

        def forward(self, x):
            return self.fc(x).detach()

    def get_inputs():
        return [torch.randn({M}, {K})]

    def get_init_inputs():
        return []
''')


def _analyze(tmp_path, name, source):
    model_file = tmp_path / f"{name}.py"
    model_file.write_text(source)
    graph_dir = tmp_path / f"{name}_graph"
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(graph_dir)) is True
    einsum_dir = tmp_path / f"{name}_einsum"
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir),
                                     copy_graph=False, enable_rename=True) is not None
    analysis = EinsumGraphAnalyzer(debug=False).analyze_graph(
        einsum_dir / "einsum_graph_renamed.yaml", tmp_path / f"{name}_analysis",
        precision="fp32", copy_graph=False)
    assert analysis is not None
    return analysis


def _layers_of(analysis, kind):
    return [v for v in analysis["layers"].values() if v["type"] == kind]


def test_alias_layers_are_free_and_clone_stays_charged(tmp_path):
    analysis = _analyze(tmp_path, "alias_chain", ALIAS_CHAIN_MODEL)
    (detach,) = _layers_of(analysis, "detach")
    (reshape,) = _layers_of(analysis, "reshape")
    (clone,) = _layers_of(analysis, "clone")
    for alias in (detach, reshape):
        assert alias["other_ops"] == 0
        assert alias["macs"] == 0
        assert alias["unfused_elements"] == 0
        assert alias["intermediate_elements"] == 0
    # a real copy reads and writes every element, with no ALU work
    assert clone["other_ops"] == 0
    assert clone["unfused_elements"] == 2 * M * N


def test_alias_chain_totals(tmp_path):
    total = _analyze(tmp_path, "alias_chain", ALIAS_CHAIN_MODEL)["total"]
    assert total["macs"] == M * K * N                      # unchanged by the convention
    assert total["other_ops"] == M * N                     # only the multiply; nothing for detach/reshape
    # linear (x + W + y) + clone (read + write) + mul (read + write); aliases add nothing
    assert total["unfused_elements"] == (M * K + K * N + M * N) + 2 * M * N + 2 * M * N
    # fused: x, W and the model output once; unchanged
    assert total["fused_elements"] == M * K + K * N + M * N


def test_alias_preserves_connectivity(tmp_path):
    analysis = _analyze(tmp_path, "alias_chain", ALIAS_CHAIN_MODEL)
    layers = analysis["layers"]
    (detach,) = [k for k, v in layers.items() if v["type"] == "detach"]
    (reshape,) = [k for k, v in layers.items() if v["type"] == "reshape"]
    (clone,) = [k for k, v in layers.items() if v["type"] == "clone"]
    (mul,) = [k for k, v in layers.items() if v["type"] == "mul"]
    assert layers[detach]["connections"]["inputs"] == [clone]
    assert layers[detach]["connections"]["outputs"] == [reshape]
    assert layers[reshape]["connections"]["outputs"] == [mul]
    assert layers[mul]["connections"]["inputs"] == [reshape]


def test_model_output_through_alias_keeps_its_dram_write(tmp_path):
    analysis = _analyze(tmp_path, "alias_out", ALIAS_AT_OUTPUT_MODEL)
    (detach,) = _layers_of(analysis, "detach")
    assert detach["other_ops"] == 0
    assert detach["unfused_elements"] == 0
    # fused mode: the output write still reaches DRAM through the alias
    assert detach["fused_elements"] == M * N
    total = analysis["total"]
    assert total["macs"] == M * K * N
    assert total["other_ops"] == 0
    assert total["fused_elements"] == M * K + K * N + M * N
    assert total["unfused_elements"] == M * K + K * N + M * N


def test_alias_set_is_explicit_and_excludes_copies():
    from solar.analysis.graph_analyzer import ALIAS_OPS
    assert {"detach", "detach_", "alias", "view_as", "unbind"} <= set(ALIAS_OPS)
    # real copies and the unprovable contiguous() are not aliases; reshape_as is layout dependent
    assert not {"clone", "copy_", "contiguous", "to", "reshape_as"} & set(ALIAS_OPS)


def test_docs_state_the_alias_convention():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "EINSUM_GUIDE.md").read_text()
    assert re.search(r"##+ .*[Aa]lias", guide), "EINSUM_GUIDE.md needs an alias/view/copy cost-convention section"
    for word in ("detach", "clone", "contiguous", "unfused"):
        assert word in guide
    assert "layout unknown" in guide                       # unknown layouts are charged, never 'established'
    assert "recorded at capture" in guide


def test_ci_unit_lane_selects_this_module():
    ci = (Path(__file__).resolve().parents[1] / ".gitlab-ci.yml").read_text()
    unit = ci[ci.index("unit-tests:"):ci.index("regression-tests:")]
    assert "tests/test_alias_view_accounting.py" in unit


# --- an earlier decision review P2 reshape-1: reshape_as is layout dependent (view OR copy) --------------

def _two_input_model(body):
    return textwrap.dedent(f'''
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(6, 6))

            def forward(self, x, other):
                {body}

        def get_inputs():
            return [torch.arange(24, dtype=torch.float32).reshape(4, 6), torch.empty(2, 12)]

        def get_init_inputs():
            return []
    ''')


def _layer(analysis, kind):
    rows = [v for v in analysis["layers"].values() if v["type"] == kind]
    assert len(rows) == 1, (kind, [v["type"] for v in analysis["layers"].values()])
    return rows[0]


def test_noncontiguous_reshape_as_is_charged_as_a_copy(tmp_path):
    # x.t() is a strided view; reshape_as must materialize 24 elements (PyTorch: view OR new tensor)
    analysis = _analyze(tmp_path, "reshape_as_copy", _two_input_model("return x.t().reshape_as(other)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["macs"] == 0 and layer["other_ops"] == 0            # a copy has no ALU work
    assert layer["memory_elements"]["inputs"] == [24, 0]              # source read; the shape operand is not a data read
    assert layer["memory_elements"]["outputs"] == [24]
    assert layer["unfused_elements"] == 48
    assert layer["alias_qualification"].startswith("copy_charged")


def test_contiguous_reshape_as_of_a_model_input_is_an_alias(tmp_path):
    analysis = _analyze(tmp_path, "reshape_as_alias_input", _two_input_model("return x.reshape_as(other)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] == 0 and layer["other_ops"] == 0
    assert layer["alias_qualification"].startswith("alias")


def test_contiguous_reshape_as_after_a_matmul_is_an_alias(tmp_path):
    analysis = _analyze(tmp_path, "reshape_as_alias_matmul", _two_input_model("return (x @ self.w).reshape_as(other)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] == 0
    assert layer["alias_qualification"].startswith("alias")
    assert analysis["total"]["macs"] == 4 * 6 * 6


def test_reshape_as_after_a_transposed_elementwise_chain_is_charged(tmp_path):
    # (x * 2).t(): the elementwise op inherits x's contiguity, the transpose breaks it
    analysis = _analyze(tmp_path, "reshape_as_copy_chain", _two_input_model("return (x * 2.0).t().reshape_as(other)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] == 48
    assert layer["alias_qualification"].startswith("copy_charged")


def test_reshape_as_is_not_an_unconditional_alias():
    from solar.analysis.graph_analyzer import ALIAS_OPS, LAYOUT_DEPENDENT_ALIAS_OPS
    assert "reshape_as" not in ALIAS_OPS
    assert "reshape_as" in LAYOUT_DEPENDENT_ALIAS_OPS


# --- an earlier decision review P2 layout-2 / input-2: layout proofs come from recorded evidence only ---

PROVENANCE_CASES = [
    # name, source expression (x: model input, other: shape reference), transposed input, reshape_as copies
    ("transpose", "x.t()", False, True),
    ("as_strided", "x.as_strided((6, 4), (1, 6))", False, True),
    ("expand", "x[:1, :].expand(4, 6)", False, True),
    ("clone_transpose", "x.t().clone()", False, True),                      # clone preserves a dense transposed layout
    ("clone_as_strided", "x.as_strided((6, 4), (1, 6)).clone()", False, True),
    ("noncontiguous_input", "x", True, True),                               # input stride [1, 6]: recorded at capture
    ("noncontiguous_parameter", "self.w", False, True),                     # parameter stride [1, 6]: recorded at capture
    ("einsum_transpose", "torch.einsum('ij->ji', x)", False, True),         # einsum returns the transposed view
    ("channels_last_contiguous", "x.contiguous(memory_format=torch.channels_last)", False, True),
    ("channels_last_cast", "x.to(memory_format=torch.channels_last)", False, True),
    ("transpose_cast", "x.t().to(torch.float64)", False, True),
    ("contiguous_input", "x", False, False),
    ("contiguous_matmul", "x @ self.m", False, False),
    ("expand_clone", "x[:1, :].expand(4, 6).clone()", False, False),        # clone of an overlapping source is contiguous
]


def _provenance_model(expr, inputs, init, tail="return v.reshape_as(other)"):
    return textwrap.dedent(f'''
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                {init}

            def forward(self, x, other):
                v = {expr}
                {tail}

        def get_inputs():
            return [{inputs}, torch.empty(2, 12)]

        def get_init_inputs():
            return []
    ''')


def _provenance_case(name, input_transpose):
    shape = "2, 3, 2, 2" if name.startswith("channels_last") else "4, 6"
    inp = f"torch.arange(24, dtype=torch.float32).reshape({shape})" + (".t()" if input_transpose else "")
    init = {"noncontiguous_parameter": "self.w = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6).t())",
            "contiguous_matmul": "self.m = nn.Parameter(torch.eye(6))"}.get(name, "pass")
    return inp, init


@pytest.mark.parametrize("name,expr,input_transpose,copies", PROVENANCE_CASES, ids=[c[0] for c in PROVENANCE_CASES])
def test_layout_proof_does_not_erase_materializing_reshape(tmp_path, name, expr, input_transpose, copies):
    import types
    import torch
    inp, init = _provenance_case(name, input_transpose)
    # storage control: does PyTorch really materialise a copy here?
    x = eval(inp, {"torch": torch})
    other = torch.empty(2, 12)
    self_ns = types.SimpleNamespace(w=torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6).t()),
                                    m=torch.eye(6))
    v = eval(expr, {"torch": torch}, {"x": x, "self": self_ns})
    out = v.reshape_as(other)
    assert (v.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()) == copies, "storage control expectation"
    analysis = _analyze(tmp_path, f"prov_{name}", _provenance_model(expr, inp, init))
    layer = _layer(analysis, "reshape_as")
    assert layer["macs"] == 0 and layer["other_ops"] == 0
    if copies:
        assert layer["unfused_elements"] >= 48, "24 source reads + 24 writes: the source is not a proven contiguous alias"
        assert layer["memory_elements"]["inputs"][1:] == [0], "the shape reference is not a data read"
        assert layer["alias_qualification"].startswith("copy_charged")
        assert "established" not in layer["alias_qualification"]
    else:
        assert layer["unfused_elements"] == 0
        assert layer["alias_qualification"].startswith("alias (layout established:")


@pytest.mark.parametrize("use_other_as_data", [False, True])
def test_shape_operand_exclusion_preserves_independent_data_use(tmp_path, use_other_as_data):
    tail = "return v.reshape_as(other)" + (" + other" if use_other_as_data else "")
    analysis = _analyze(tmp_path, f"shape_operand_{int(use_other_as_data)}",
                        _provenance_model("x", "torch.arange(24, dtype=torch.float32).reshape(4, 6)", "pass", tail))
    assert _layer(analysis, "reshape_as")["unfused_elements"] == 0
    assert analysis["total"]["fused_elements"] == (72 if use_other_as_data else 48)
    if use_other_as_data:
        assert _layer(analysis, "add")["unfused_elements"] == 72


def test_capture_records_input_and_parameter_contiguity(tmp_path):
    import yaml
    inp, init = _provenance_case("noncontiguous_parameter", False)
    model_file = tmp_path / "rec.py"
    model_file.write_text(_provenance_model("self.w", inp, init))
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(tmp_path / "graph")) is True
    g = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text())["layers"]
    (param,) = [v for v in g.values() if v["type"] == "parameter-tensor"]
    assert param["module_args"]["layout"] == {"contiguous": False, "stride": [1, 6],
                                              "evidence": "recorded at capture (bound to the parameter object)"}
    aux = [v for v in g.values() if v["type"] in ("auxiliary-tensor", "input-tensor")]
    assert aux and all(v["module_args"]["layout"]["contiguous"] is True for v in aux)
    assert PyTorchToEinsum().convert(str(tmp_path / "graph" / "pytorch_graph.yaml"), str(tmp_path / "einsum"),
                                     copy_graph=False, enable_rename=True) is not None
    e = yaml.safe_load((tmp_path / "einsum" / "einsum_graph.yaml").read_text())
    assert e["tensor_layouts"]["Model.reshape_as.Weight"]["contiguous"] is False
    assert e["tensor_layouts"]["start.Output"]["contiguous"] is True


def test_layout_rules_are_evidence_based():
    from solar.analysis import graph_analyzer as ga
    # operation names that do not prove a row-major result are not unconditional proofs
    assert not {"clone", "einsum", "contiguous", "cat", "stack", "conv2d", "to"} & set(ga._FRESH_ROW_MAJOR_OPS)
    assert not {"to", "type", "type_as", "float", "half"} & set(ga._LAYOUT_INHERITING_OPS)
    assert ga._memory_format_evidence(
        {"raw_attributes": "[[Tensor(shape=(2, 3, 2, 2), dtype=torch.float32)], {memory_format: torch.channels_last}]"}
    ) == ("channels_last", True)
    assert ga._memory_format_evidence({"raw_attributes": "[[Tensor(shape=(4, 6), dtype=torch.float32)], {}]"}) == ("default", True)
    assert ga._memory_format_evidence({}) == (None, False)
    # a missing producer is not a proof: only a recorded contiguous layout is
    assert ga.EinsumGraphAnalyzer._recorded_layout_reason("start.Output", {}) is None
    assert ga.EinsumGraphAnalyzer._recorded_layout_reason("start.Output", {"start.Output": {"contiguous": False}}) is None
    assert ga.EinsumGraphAnalyzer._recorded_layout_reason("start.Output", {"start.Output": {"contiguous": True}})


# --- an earlier decision review P2 input-3 / layout-3: a layout proof must belong to the right tensor ---

def _model_source(init, args, body, inputs):
    return textwrap.dedent(f'''
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                {init}

            def forward(self, {args}):
                {body}

        def get_inputs():
            return [{inputs}]

        def get_init_inputs():
            return []
    ''')


NONCONTIG = "torch.arange(24, dtype=torch.float32).reshape(4, 6).t()"


@pytest.mark.parametrize("kind", ["nested_input", "buffer", "plain_tensor"])
def test_layout_record_must_belong_to_its_tensor(tmp_path, kind):
    """Reviewer's round-3 control: a contiguous unused model input x [6,4] must not donate its layout to a
    same-shape noncontiguous receiver that is a nested input, a buffer or a plain tensor attribute."""
    import torch
    noncontig = eval(NONCONTIG, {"torch": torch}); out = noncontig.reshape(2, 12)
    assert noncontig.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
    init = {"buffer": f"self.register_buffer('b', {NONCONTIG})", "plain_tensor": f"self.b = {NONCONTIG}"}.get(kind, "pass")
    args = "x, pair, other" if kind == "nested_input" else "x, other"
    source = "pair[0]" if kind == "nested_input" else "self.b"
    inputs = "torch.arange(24, dtype=torch.float32).reshape(6, 4), " + (f"[{NONCONTIG}], " if kind == "nested_input" else "") + "torch.empty(2, 12)"
    analysis = _analyze(tmp_path, f"belong_{kind}", _model_source(init, args, f"return {source}.reshape_as(other)", inputs))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] >= 48, "a different same-shape contiguous model input cannot prove this noncontiguous tensor's layout"
    assert layer["alias_qualification"].startswith("copy_charged") and "established" not in layer["alias_qualification"]
    if kind != "nested_input":
        # the receiver never became a graph operand: the result is marked incomplete, the shape operand is not read
        assert layer["status"] == "incomplete" and "receiver" in layer["capture_incomplete"]
        assert layer["memory_elements"]["inputs"] == [0]


def test_nested_input_layout_is_bound_by_identity(tmp_path):
    """pytorch_graph.yaml carries each traced input's own record (nested list member included)."""
    import yaml
    model_file = tmp_path / "nested.py"
    model_file.write_text(_model_source("pass", "x, pair, other", "return pair[0].reshape_as(other)",
                                        f"torch.arange(24, dtype=torch.float32).reshape(6, 4), [{NONCONTIG}], torch.empty(2, 12)"))
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(tmp_path / "graph")) is True
    g = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text())["layers"]
    recs = {tuple(v["output_shapes"][0]): v["module_args"]["layout"] for v in g.values()
            if v["type"] in ("auxiliary-tensor", "input-tensor") and v["module_args"].get("layout")}
    # the [6,4] record present in the graph is the traced nested receiver (stride [1,6]), not the unused x
    assert recs[(6, 4)]["contiguous"] is False and recs[(6, 4)]["stride"] == [1, 6], recs
    assert recs[(2, 12)]["contiguous"] is True
    assert all(r["evidence"] == "recorded at capture (bound to the traced input tensor)" for r in recs.values())


def test_mm_out_layout_is_not_fresh_row_major(tmp_path):
    """Reviewer's round-3 control: mm(x, m, out=dst) keeps dst's transposed layout; reshape_as copies."""
    import torch
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6); m = torch.eye(6); dst = torch.empty(6, 4).t()
    v = torch.mm(x, m, out=dst); out = v.reshape(2, 12)
    assert not v.is_contiguous() and v.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
    analysis = _analyze(tmp_path, "mm_out", _model_source("pass", "x, m, dst, other", "return torch.mm(x, m, out=dst).reshape_as(other)",
                        "torch.arange(24, dtype=torch.float32).reshape(4, 6), torch.eye(6), torch.empty(6, 4).t(), torch.empty(2, 12)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] >= 48, "mm(out=...) uses the provided destination layout, not a fresh row-major allocation"
    assert layer["alias_qualification"].startswith("copy_charged")


def test_mm_without_destination_is_still_a_fresh_row_major_proof(tmp_path):
    analysis = _analyze(tmp_path, "mm_fresh", _two_input_model("return torch.mm(x, self.w).reshape_as(other)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] == 0
    assert "no destination argument recorded" in layer["alias_qualification"]


def test_ambiguous_flat_input_layouts_are_not_positive_proofs(tmp_path):
    """Reviewer's round-3 control: two same-shape top-level inputs with different contiguity."""
    analysis = _analyze(tmp_path, "ambiguous_flat", _model_source("pass", "x, y, other", "return y.reshape_as(other)",
                        f"torch.empty(6, 4), {NONCONTIG}, torch.empty(2, 12)"))
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] >= 48
    assert "non-contiguous" in layer["alias_qualification"]     # y's own record, bound by identity


def test_missing_layout_metadata_is_charged(tmp_path):
    """Reviewer's round-3 control: a graph without tensor_layouts never yields a positive proof."""
    import yaml
    model_file = tmp_path / "plain.py"
    model_file.write_text(_two_input_model("return x.reshape_as(other)"))
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(tmp_path / "graph")) is True
    assert PyTorchToEinsum().convert(str(tmp_path / "graph" / "pytorch_graph.yaml"), str(tmp_path / "einsum"),
                                     copy_graph=False, enable_rename=True) is not None
    g = yaml.safe_load((tmp_path / "einsum" / "einsum_graph_renamed.yaml").read_text()); g.pop("tensor_layouts", None)
    q = tmp_path / "no-layout.yaml"; q.write_text(yaml.safe_dump(g, sort_keys=False))
    analysis = EinsumGraphAnalyzer().analyze_graph(q, tmp_path / "no-layout-analysis", copy_graph=False)
    layer = _layer(analysis, "reshape_as")
    assert layer["unfused_elements"] >= 48 and "layout unknown" in layer["alias_qualification"]


def test_receiver_validation_helpers():
    from solar.analysis import graph_analyzer as ga
    rec = {"raw_attributes": "[[Tensor(shape=(6, 4), dtype=torch.float32), Tensor(shape=(2, 12), dtype=torch.float32)], {}]",
           "tensor_shapes": {"inputs": [[6, 4], [2, 12]], "outputs": [[2, 12]]}}
    assert ga._recorded_positional_tensor_shapes(rec) == [[6, 4], [2, 12]]
    # round 5 (input-4): counts and shapes are NOT evidence; only the identity record decides
    assert ga._reshape_as_receiver_present(rec) is None                        # no identity record: unverified
    rec["call_operands"] = {"tensor_args": 2, "positional_tensor_args": 2, "graph_operands": 1, "receiver_bound": False}
    assert ga._reshape_as_receiver_present(rec) is False                       # same shapes, receiver unbound
    rec["call_operands"]["receiver_bound"] = True
    assert ga._reshape_as_receiver_present(rec) is False                       # identity alone never authorizes reuse (STORAGE-5)
    assert ga._reshape_as_receiver_state(rec) == "unverified"
    rec["call_operands"]["receiver_layout"] = {"layout_valid": True, "storage_identity": "unchanged", "geometry_matches_record": True, "at_call": {"contiguous": True}}
    assert ga._reshape_as_receiver_present(rec) is True and ga._reshape_as_receiver_state(rec) == "bound"
    rec["call_operands"]["receiver_layout"]["layout_valid"] = False
    assert ga._reshape_as_receiver_state(rec) == "mutated"
    assert ga._reshape_as_receiver_present({"tensor_shapes": {"inputs": [[6, 4], [2, 12]]}}) is None
    # the identity record is authoritative for the destination too
    assert ga._has_destination_argument({"call_operands": {"receiver_bound": True, "destination": {"keyword": "out", "shape": [4, 6], "graph_operand": False}}}) is True
    assert ga._has_destination_argument({"call_operands": {"receiver_bound": True}, "raw_attributes": "[[Tensor(shape=(4, 6), dtype=torch.float32)], {out: Tensor(shape=(4, 6), dtype=torch.float32)}]"}) is False
    assert ga._has_destination_argument({"raw_attributes": "[[Tensor(shape=(4, 6), dtype=torch.float32), Tensor(shape=(6, 6), dtype=torch.float32)], {out: Tensor(shape=(4, 6), dtype=torch.float32)}]"}) is True
    assert ga._has_destination_argument({"raw_attributes": "[[Tensor(shape=(4, 6), dtype=torch.float32), Tensor(shape=(6, 6), dtype=torch.float32)], {}]"}) is False
    assert ga._has_destination_argument({}) is None


# ---- round 5 (an earlier decision review): identity-bound receiver, no recursion through it, Step 4 + status, out= -------

INT64_OTHER = "torch.empty(6, 4, dtype=torch.int64)"


def _reshape_as_layers(analysis):
    return [v for v in analysis["layers"].values() if v["type"] == "reshape_as"]


@pytest.mark.parametrize("spelling", ["positional", "keyword"])
def test_same_shaped_shape_reference_never_lends_its_layout_to_an_unbound_receiver(tmp_path, spelling):
    """input-4a: buffer receiver, same-shaped contiguous int64 `other`; positional and `other=` spellings."""
    call = "self.b.reshape_as(other=other)" if spelling == "keyword" else "self.b.reshape_as(other)"
    a = _analyze(tmp_path, f"unbound_{spelling}", _model_source(
        f"self.register_buffer('b', {NONCONTIG})", "other", f"return {call}", INT64_OTHER))
    layer = _layer(a, "reshape_as")
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "reshape_as_receiver_unbound"
    assert layer["unfused_elements"] == 48 and "receiver not a connected graph operand" in layer["alias_qualification"]
    assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"] == {k: "reshape_as_receiver_unbound" for k, v in a["layers"].items() if v["type"] == "reshape_as"}


@pytest.mark.parametrize("spelling", ["positional", "keyword"])
def test_unresolved_receiver_layout_does_not_reach_a_downstream_reshape(tmp_path, spelling):
    """input-4b: the chained second reshape_as copies 24 noncontiguous elements (48 charged), never 0."""
    call = ("self.b.reshape_as(other=other)" if spelling == "keyword" else "self.b.reshape_as(other)") + ".reshape_as(final_shape)"
    a = _analyze(tmp_path, f"chain_{spelling}", _model_source(
        f"self.register_buffer('b', {NONCONTIG})", "other, final_shape", f"return {call}", f"{INT64_OTHER}, torch.empty(2, 12)"))
    layers = _reshape_as_layers(a)
    assert len(layers) == 2
    second = [l for l in layers if l["tensor_shapes"]["outputs"] == [[2, 12]]][0]
    assert second["unfused_elements"] >= 48 and "copy_charged" in second["alias_qualification"]


@pytest.mark.parametrize("kind", ["buffer", "plain_tensor"])
def test_unbound_receiver_read_is_external_and_qualifies_total_and_perf(tmp_path, kind):
    """input-4c: the missing receiver's 24-element read enters Step 4; total and perf carry status incomplete."""
    from solar.perf.perf_model import EinsumGraphPerfModel
    init = f"self.register_buffer('b', {NONCONTIG})" if kind == "buffer" else f"self.b = {NONCONTIG}"
    a = _analyze(tmp_path, f"external_{kind}", _model_source(init, "other", "return self.b.reshape_as(other)", "torch.empty(2, 12)"))
    layer = _layer(a, "reshape_as")
    assert layer["status"] == "incomplete" and layer["unfused_elements"] == 48 and layer["unfused_bytes"] == 192
    assert layer["model_io_elements"] == 48                              # 24 receiver read (external) + 24 output
    # INTEG-15: under the earlier decision split the receiver read is a model-level external read
    assert layer["fused_elements"] == layer["weight_read_elements"] + layer["model_io_elements"] == 48
    assert a["total"]["external_input_bytes"] == 96                      # the receiver read, fp32
    assert a["total"]["fused_elements"] == 48                            # receiver read + returned output
    assert a["total"]["status"] == "incomplete"
    perf = EinsumGraphPerfModel().predict(str(tmp_path / f"external_{kind}_analysis" / "analysis.yaml"), str(tmp_path / "perf"),
                                          arch_config="H100_PCIe", precision="fp32")
    assert perf["workload"]["status"] == "incomplete" and perf["workload"]["incomplete_layers"] == a["total"]["incomplete_layers"]


def test_bound_receiver_still_proves_the_alias(tmp_path):
    """Control: a model-input receiver bound by identity keeps the free alias (identity record present and True)."""
    a = _analyze(tmp_path, "bound", _two_input_model("return x.reshape_as(other)"))
    layer = _layer(a, "reshape_as")
    assert layer["unfused_elements"] == 0 and "status" not in layer and "status" not in a["total"]
    assert layer["call_operands"]["receiver_bound"] is True and layer["call_operands"]["graph_operands"] == 2


def test_inheriting_op_with_out_destination_is_not_a_layout_proof(tmp_path):
    """layout-4: add(x, x, out=dst) keeps the transposed buffer's strides; the reshape copies (48 charged)."""
    a = _analyze(tmp_path, "add_out", _model_source(
        "self.register_buffer('dst', torch.empty(4, 6).t())", "x, other",
        "return torch.add(x, x, out=self.dst).reshape_as(other)",
        "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    layer = _layer(a, "reshape_as")
    assert layer["unfused_elements"] >= 48 and "copy_charged" in layer["alias_qualification"]
    add = _layer(a, "add")
    assert add["call_operands"]["destination"] == {"keyword": "out", "shape": [6, 4], "graph_operand": False}


def test_inheriting_op_without_destination_still_inherits(tmp_path):
    """Control: add(x, x) of a contiguous input followed by reshape_as stays an alias (recorded call, no out=)."""
    a = _analyze(tmp_path, "add_plain", _model_source(
        "pass", "x, other", "return torch.add(x, x).reshape_as(other)",
        "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    layer = _layer(a, "reshape_as")
    assert layer["unfused_elements"] == 0 and "no destination argument recorded" in layer["alias_qualification"]


def test_mm_out_destination_is_captured_not_refused(tmp_path):
    """INTEG-10 (option a): out= is torch's aliasing destination, split off before the semantic binding, so
    torch.mm(x, m, out=dst) is captured (not refused) with the destination recorded; the reshape is charged."""
    a = _analyze(tmp_path, "mm_out", _model_source(
        "self.dst = nn.Parameter(torch.empty(6, 4).t(), requires_grad=False)", "x, m, other",
        "return torch.mm(x, m, out=self.dst).reshape_as(other)",
        "torch.arange(24, dtype=torch.float32).reshape(4, 6), torch.eye(6), torch.empty(2, 12)"))
    mm = [v for v in a["layers"].values() if v["type"] in ("mm", "matmul")][0]
    assert mm["call_operands"]["destination"]["keyword"] == "out" and mm["call_operands"]["graph_operands"] == 2
    assert _layer(a, "reshape_as")["unfused_elements"] == 48


def test_split_destination_keeps_the_fail_closed_rule_for_other_tensor_keywords():
    from solar.graph import _torchview_param_shim as shim
    x = torch.ones(2, 2)
    rest, out = shim._split_destination({"out": x, "alpha": 2})
    assert rest == {"alpha": 2} and out is x
    assert shim._split_destination({"other": x}) == ({"other": x}, None)        # not a destination
    assert shim._split_destination({"out": None}) == ({"out": None}, None)
    # mm's testing-stub signature has no out: binding now succeeds through the split; a genuinely unbindable
    # tensor keyword still returns None (fail closed)
    assert shim._semantic_operand_leaves(torch.mm, (x, x), {"out": torch.empty(2, 2)}) == [x, x]
    assert shim._semantic_operand_leaves(torch.mm, (x,), {"mat3": x}) is None


def test_bind_call_operands_identity_rule():
    """The record is identity-based: a same-shaped plain tensor in the receiver slot is unbound."""
    from solar.graph import tensor_layouts as tl
    import torchview.recorder_tensor as rt

    class Node:  # stand-in FunctionNode / TensorNode
        def __init__(self, tensor_id=None): self.tensor_id = tensor_id; self.parents = []
    x = torch.ones(6, 4).as_subclass(rt.RecorderTensor); xn = Node(); x.tensor_nodes = [xn]
    other = torch.ones(6, 4).as_subclass(rt.RecorderTensor); on = Node(); other.tensor_nodes = [on]
    fn = Node(); fn.parents = [xn, on]
    assert tl.bind_call_operands(fn, (x, other), {})["receiver_bound"] is True
    plain = torch.ones(6, 4)                                                   # buffer / attribute receiver
    fn2 = Node(); fn2.parents = [on]
    rec = tl.bind_call_operands(fn2, (plain,), {"other": other})
    assert rec == {"tensor_args": 2, "positional_tensor_args": 1, "graph_operands": 1, "receiver_bound": False}
    p = torch.nn.Parameter(torch.ones(6, 4), requires_grad=False); pn = Node(tensor_id=id(p))
    fn3 = Node(); fn3.parents = [pn, on]
    rec = tl.bind_call_operands(fn3, (p, other), {"out": plain})
    assert rec["receiver_bound"] is True and rec["destination"] == {"keyword": "out", "shape": [6, 4], "graph_operand": False}


@pytest.mark.parametrize("noncontig", [False, True])
def test_cpu_fallback_parameter_layout_stays_unknown(tmp_path, noncontig):
    """The reviewer's follow-up control without its draw-call-count instrumentation (the earlier capability gate
    probes once per process and is cached, so the per-constructor meta probe the count assumed no longer
    exists): the module-backed CPU fallback re-instantiates the model, its parameter nodes stay unbound, and
    the reshape_as is charged with 'layout unknown' whatever the parameter's real contiguity."""
    init = "self.w = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(" + ("4, 6).t()" if noncontig else "6, 4)") + ", requires_grad=False)"
    body = "if x.device.type == 'meta':\n                    raise RuntimeError('review CPU fallback control')\n                return self.w.reshape_as(other)"
    a = _analyze(tmp_path, f"cpu_fallback_{int(noncontig)}", _model_source(init, "x, other", body,
                 "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    import yaml
    g = yaml.safe_load((tmp_path / f"cpu_fallback_{int(noncontig)}_graph" / "pytorch_graph.yaml").read_text())["layers"]
    params = [n for n in g.values() if n["type"] == "parameter-tensor"]
    assert params and all("layout" not in p["module_args"] for p in params)
    layer = _layer(a, "reshape_as")
    assert layer["unfused_elements"] == 48 and "layout unknown" in layer["alias_qualification"]


# ---- round 6 (an earlier decision review P2 storage-5): identity is not layout validity --------------------

@pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
@pytest.mark.parametrize("rebind", ["data", "set_"])
def test_rebound_receiver_storage_is_not_a_layout_proof(tmp_path, rebind, force_cpu):
    """Reviewer control: contiguous input x; in forward x.data = b (or x.set_(b)) with a transposed buffer b
    (stride [1, 6]); id(x) is unchanged, x now aliases b's storage, and reshape_as allocates: 48 charged."""
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    b = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    xid, before = id(x), x.untyped_storage().data_ptr()
    if rebind == "data":
        x.data = b
    else:
        x.set_(b)
    assert id(x) == xid and x.untyped_storage().data_ptr() != before and not x.is_contiguous()
    prefix = "if x.device.type == 'meta':\n                    raise RuntimeError('review CPU fixture')\n                " if force_cpu else ""
    change = "x.data = self.b" if rebind == "data" else "x.set_(self.b)"
    a = _analyze(tmp_path, f"rebound_{rebind}_{int(force_cpu)}", _model_source(
        f"self.register_buffer('b', {NONCONTIG})", "x, other", prefix + change + "\n                return x.reshape_as(other)",
        "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    layer = _layer(a, "reshape_as")
    assert layer["call_operands"]["receiver_bound"] is True              # identity unchanged ...
    rl = layer["call_operands"]["receiver_layout"]
    assert rl["layout_valid"] is False and rl["at_call"]["contiguous"] is False and rl["at_call"]["stride"] == [1, 6]
    assert layer["unfused_elements"] >= 48 and "layout established" not in layer["alias_qualification"]
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "reshape_as_receiver_storage_mutated"
    assert a["total"]["status"] == "incomplete"


def test_contiguous_storage_swap_on_cpu_invalidates_the_record(tmp_path):
    """x.data = another CONTIGUOUS tensor: strides unchanged but the storage differs; on a data device the
    storage identity is checked and the proof is refused (qualified copy)."""
    a = _analyze(tmp_path, "swap_contig", _model_source(
        "self.register_buffer('c', torch.arange(24, dtype=torch.float32).reshape(6, 4))", "x, other",
        "if x.device.type == 'meta':\n                    raise RuntimeError('force the CPU path')\n                x.data = self.c\n                return x.reshape_as(other)",
        "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    layer = _layer(a, "reshape_as")
    rl = layer["call_operands"]["receiver_layout"]
    assert rl["storage_identity"] == "changed" and rl["geometry_matches_record"] is True and rl["layout_valid"] is False
    assert layer["unfused_elements"] == 48 and layer["incomplete_reason"] == "reshape_as_receiver_storage_mutated"


def test_untouched_receiver_keeps_a_valid_record(tmp_path):
    a = _analyze(tmp_path, "untouched", _two_input_model("return x.reshape_as(other)"))
    layer = _layer(a, "reshape_as")
    rl = layer["call_operands"]["receiver_layout"]
    assert rl["layout_valid"] is True and rl["geometry_matches_record"] is True and rl["at_call"]["contiguous"] is True
    assert layer["unfused_elements"] == 0 and "status" not in layer


def test_receiver_layout_validity_helper():
    from solar.graph import tensor_layouts as tl
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    tl._BOUND_STORAGE[id(x)] = tl.storage_record(x)
    assert tl.receiver_layout_validity(x)["layout_valid"] is True
    x.set_(torch.arange(24, dtype=torch.float32).reshape(4, 6).t())
    v = tl.receiver_layout_validity(x)
    assert v["layout_valid"] is False and v["storage_identity"] == "changed" and v["geometry_matches_record"] is False
    y = torch.empty(6, 4, device="meta"); tl._BOUND_STORAGE[id(y)] = tl.storage_record(y)
    v = tl.receiver_layout_validity(y)
    assert v["layout_valid"] is True and v["storage_identity"].startswith("unavailable")
    y.set_(torch.empty(4, 6, device="meta").t())
    assert tl.receiver_layout_validity(y)["layout_valid"] is False                 # geometry differs on meta
    h = torch.ones(2, 2)
    assert tl.receiver_layout_validity(h)["layout_valid"] is None                   # hidden tensor: no record
    tl._BOUND_STORAGE.pop(id(x), None); tl._BOUND_STORAGE.pop(id(y), None)


# ---- TARGET-ALIAS-FALLBACK-1: binding-time records are identity-safe and reset per capture attempt ----------
#
# _BOUND_STORAGE is keyed by id(tensor).  The meta attempt's RecorderTensors are freed when the attempt fails
# and the CPU fallback's fresh nn.Parameter may be allocated at the same address, so the fallback parameter
# met a STALE record (meta geometry) and was classified reshape_as_receiver_storage_mutated instead of
# "layout unknown" — nondeterministically (allocator luck).  Records now keep a strong reference to the
# tensor they describe (the id cannot be recycled while the record lives) and are trusted only when that
# tensor IS the receiver; every capture attempt starts from empty maps.

def test_stale_record_under_a_reused_id_is_no_record():
    from solar.graph import tensor_layouts as tl
    tl.reset_attempt_records()
    p = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6).t(), requires_grad=False)
    # a record written for ANOTHER tensor (the dead meta attempt's traced input), planted under id(p)
    ghost = torch.empty(6, 4, device="meta")
    stale = tl.storage_record(ghost)
    assert stale["tensor"] is ghost and stale["device"] == "meta"
    tl._BOUND_STORAGE[id(p)] = stale
    out = tl.receiver_layout_validity(p)
    assert out["layout_valid"] is None and "stale record" in out["storage_identity"]
    assert "geometry_matches_record" not in out
    # the receiver's own record IS trusted (geometry compared as before)
    tl._BOUND_STORAGE[id(p)] = tl.storage_record(p)
    out = tl.receiver_layout_validity(p)
    assert out["layout_valid"] is True and out["storage_identity"] == "unchanged"
    # layout_for_node: an input binding whose storage record belongs to another tensor is not returned
    class Node:
        tensor_id = id(p)
    tl._INPUT_BINDINGS[id(p)] = tl.layout_record(p, "test")
    tl._BOUND_STORAGE[id(p)] = stale
    assert tl.layout_for_node(Node(), "input-tensor", None) is None
    tl._BOUND_STORAGE[id(p)] = tl.storage_record(p)
    assert tl.layout_for_node(Node(), "input-tensor", None) == tl.layout_record(p, "test")
    tl.reset_attempt_records()
    assert not tl._BOUND_STORAGE and not tl._INPUT_BINDINGS and not tl._SOURCE_LAYOUTS


def test_records_keep_their_tensor_alive_so_ids_are_not_recycled():
    import gc
    from solar.graph import tensor_layouts as tl
    tl.reset_attempt_records()
    t = torch.empty(6, 4, device="meta"); tid = id(t)
    tl._BOUND_STORAGE[tid] = tl.storage_record(t)
    del t; gc.collect()
    assert tl._BOUND_STORAGE[tid]["tensor"] is not None and id(tl._BOUND_STORAGE[tid]["tensor"]) == tid
    tl.reset_attempt_records()


def test_each_capture_attempt_resets_the_records(tmp_path, monkeypatch):
    """A failing meta attempt followed by the CPU fallback must reset the maps at the start of EACH attempt."""
    from solar.graph import tensor_layouts as tl
    calls = []
    real = tl.reset_attempt_records
    def counting():
        calls.append(dict(bound=len(tl._BOUND_STORAGE), inputs=len(tl._INPUT_BINDINGS)))
        real()
    monkeypatch.setattr(tl, "reset_attempt_records", counting)
    init = "self.w = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6).t(), requires_grad=False)"
    body = "if x.device.type == 'meta':\n                    raise RuntimeError('review CPU fallback control')\n                return self.w.reshape_as(other)"
    a = _analyze(tmp_path, "reset_per_attempt", _model_source(init, "x, other", body,
                 "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
    # bind scope entry + meta attempt + the fallback attempt(s): at least one reset AFTER the meta attempt wrote records
    assert len(calls) >= 3 and any(c["bound"] > 0 for c in calls[1:]), calls
    layer = _layer(a, "reshape_as")
    assert layer["unfused_elements"] == 48 and "layout unknown" in layer["alias_qualification"]
    assert layer.get("incomplete_reason") != "reshape_as_receiver_storage_mutated"


@pytest.mark.parametrize("noncontig", [False, True])
def test_cpu_fallback_parameter_layout_is_deterministic_over_twelve_captures(tmp_path, noncontig):
    """The flaky control run 12x in one process: every capture must report 'layout unknown' (24 weight
    elements charged as a copy, 48 total), never the storage-mutated classification."""
    import gc
    init = "self.w = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(" + ("4, 6).t()" if noncontig else "6, 4)") + ", requires_grad=False)"
    body = "if x.device.type == 'meta':\n                    raise RuntimeError('review CPU fallback control')\n                return self.w.reshape_as(other)"
    outcomes = []
    for i in range(12):
        a = _analyze(tmp_path, f"cpu_fallback_{int(noncontig)}_{i}", _model_source(init, "x, other", body,
                     "torch.arange(24, dtype=torch.float32).reshape(6, 4), torch.empty(2, 12)"))
        layer = _layer(a, "reshape_as")
        outcomes.append((layer["unfused_elements"], "layout unknown" in layer["alias_qualification"], layer.get("incomplete_reason"), layer.get("weight_elements")))
        gc.collect()
    assert all(o[0] == 48 and o[1] for o in outcomes), outcomes
    assert not any(o[2] == "reshape_as_receiver_storage_mutated" for o in outcomes), outcomes
    assert len({o[3] for o in outcomes}) == 1, outcomes

# ---- capture-time regression (Reformer > 2400 s at an earlier revision): metadata reads must not reach the recorder --------

def test_storage_and_layout_records_never_dispatch_through_torch_function():
    """storage_record / layout_record on a RecorderTensor must not be torch-function events: dispatched, torchview's
    recorder walks the returned storage element by element (O(numel) per bound receiver)."""
    import time
    import torchview.recorder_tensor as rt
    from solar.graph import tensor_layouts as tl

    calls = []

    class Counting(torch.Tensor):                       # a tensor SUBCLASS: what the recorder is
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            calls.append(getattr(func, "__name__", str(func)))
            return super().__torch_function__(func, types, args, kwargs or {})

    counted = torch.ones(6, 4).as_subclass(Counting)
    tl.storage_record(counted); tl.layout_record(counted, "test"); tl.receiver_layout_validity(counted)
    assert calls == [], calls                           # metadata reads reach no subclass dispatch
    small = torch.ones(6, 4).as_subclass(rt.RecorderTensor)
    # size independence: a 64M-element meta tensor records in the same time as a tiny one
    big = torch.empty(8192, 8192, device="meta").as_subclass(rt.RecorderTensor)
    t0 = time.perf_counter(); tl.storage_record(big); dt_big = time.perf_counter() - t0
    t0 = time.perf_counter(); tl.storage_record(small); dt_small = time.perf_counter() - t0
    assert dt_big < 0.05 and dt_big < 50 * max(dt_small, 1e-5), (dt_big, dt_small)
    rec = tl.storage_record(big)
    assert rec["shape"] == [8192, 8192] and rec["contiguous"] is True and rec["device"] == "meta"
