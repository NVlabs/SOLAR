# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOL accounting defect an earlier decision (issue #33): recurrent layers must be priced over the time loop, the
layer stack and both directions.

Before the fix the LSTM/GRU/RNN handlers produced one summary einsum whose operands were matched
positionally against the recorded operand list (x, h0[, c0], flat weights...), so the count was a
single mislabelled contraction: KernelBench level3/35_LSTM reported 39,321,600 MACs against a hand
count of 15,435,038,720 (390x under) and level3/39_GRU 503,316,480 against 11,576,279,040 (23x).

Hand formula (per the handler's documented convention):

    MACs = sum over (layer, direction) of  S * B * (G*H) * (in_l + H)

with G = 4 (LSTM), 3 (GRU), 1 (RNN), in_0 = input size, in_l = D*H for l >= 1.  Two routes reach
that count (an earlier fix P3-1): the DEFAULT route expands torch.lstm/gru/rnn_tanh/rnn_relu into per-layer,
per-direction ih/hh linear layers plus a gates add (the MACs are the linear layers', gate arithmetic
is other_ops of the gates layer, intermediates carry their own traffic); the RAW-HANDLER route prices
an unexpanded whole node with the convention that biases, gate nonlinearities and state updates are
not charged separately and traffic keeps the generic per-tensor accounting of the single node.  Equal
MACs on both routes do not imply equal traffic/other_ops.

The unit tests drive the handlers through EinsumAnalyzer with recorded-style operand lists; the
end-to-end tests trace tiny nn.LSTM / nn.GRU modules on CPU through the full pipeline.  Nothing here
touches saved KB results, thresholds or precision.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from textwrap import dedent

import pytest

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig, TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops import misc_ops as _misc_ops   # helpers resolved at call time so that on a tree
                                                     # without the fix the tests FAIL (red), not error at import
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor

GATES = {"lstm": 4, "gru": 3, "rnn": 1}


def hand_macs(kind: str, seq: int, batch: int, inp: int, hidden: int, layers: int, dirs: int) -> int:
    g = GATES[kind] * hidden
    total = 0
    for layer in range(layers):
        in_l = inp if layer == 0 else dirs * hidden
        total += dirs * seq * batch * g * (in_l + hidden)
    return total


def recorded_operands(kind, seq, batch, inp, hidden, layers, dirs, batch_first, with_bias=True):
    """Operand lists as the graph processor records torch.lstm/gru/rnn_* calls."""
    g = GATES[kind] * hidden
    x = [batch, seq, inp] if batch_first else [seq, batch, inp]
    states = [[layers * dirs, batch, hidden]] * (2 if kind == "lstm" else 1)
    weights = []
    for layer in range(layers):
        in_l = inp if layer == 0 else dirs * hidden
        for _ in range(dirs):
            weights += [[g, in_l], [g, hidden]] + ([[g], [g]] if with_bias else [])
    out = [[batch, seq, dirs * hidden] if batch_first else [seq, batch, dirs * hidden]] + states
    raw = ("[[Tensor(shape=(%s), dtype=torch.float32), [Tensor(...)], [Tensor(...)], %s, %d, 0.0, False, %s, %s], {}]"
           % (", ".join(map(str, x)), with_bias, layers, dirs == 2, batch_first))
    return TensorShapes(inputs=[x] + states + weights, outputs=out), raw


@pytest.mark.parametrize("kind", ["lstm", "gru", "rnn"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("dirs", [1, 2])
@pytest.mark.parametrize("batch_first", [False, True])
def test_recurrent_macs_follow_the_hand_formula_with_call_attributes(kind, layers, dirs, batch_first):
    seq, batch, inp, hidden = 7, 3, 5, 6
    ts, raw = recorded_operands(kind, seq, batch, inp, hidden, layers, dirs, batch_first)
    assert _misc_ops.rnn_call_attributes(raw) == {"has_biases": True, "num_layers": layers, "bidirectional": dirs == 2, "batch_first": batch_first}
    got = EinsumAnalyzer().get_compute_cost(kind, ts, raw_attributes=raw)
    assert got == hand_macs(kind, seq, batch, inp, hidden, layers, dirs)


@pytest.mark.parametrize("kind", ["lstm", "gru"])
@pytest.mark.parametrize("layers,dirs", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("batch_first", [False, True])
def test_recurrent_macs_are_inferred_from_shapes_without_call_attributes(kind, layers, dirs, batch_first):
    # seq != batch so the layout is recoverable from h0's batch dim (PyTorch default seq-first otherwise)
    seq, batch, inp, hidden = 9, 4, 5, 6
    ts, _ = recorded_operands(kind, seq, batch, inp, hidden, layers, dirs, batch_first)
    got = EinsumAnalyzer().get_compute_cost(kind, ts)
    assert got == hand_macs(kind, seq, batch, inp, hidden, layers, dirs)
    macs, info = _misc_ops.recurrent_mac_cost(kind, ts)
    assert macs == got and info["layers"] == layers and info["directions"] == dirs and info["batch_first"] == batch_first


def test_ambiguous_layout_without_attributes_defaults_to_seq_first():
    # seq == batch: the only information is PyTorch's default; the count must still be the seq-first one
    ts, _ = recorded_operands("gru", 4, 4, 5, 6, 1, 1, batch_first=False)
    macs, info = _misc_ops.recurrent_mac_cost("gru", ts)
    assert info["batch_first"] is False and macs == hand_macs("gru", 4, 4, 5, 6, 1, 1)


def test_unparseable_operand_list_keeps_the_generic_cost_and_says_so(caplog):
    # a weight list that is not (w_ih, w_hh) pairs cannot be priced by the recurrence formula
    ts = TensorShapes(inputs=[[7, 3, 5], [1, 3, 6], [24, 5]], outputs=[[7, 3, 6]])
    macs, info = _misc_ops.recurrent_mac_cost("lstm", ts)
    assert macs is None and "pairs" in info["reason"]
    with caplog.at_level("WARNING"):
        generic = EinsumAnalyzer().get_compute_cost("lstm", ts)
    assert generic > 0 and any("D-04" in r.getMessage() for r in caplog.records)


def test_kernelbench_level3_35_lstm_operands_match_the_hand_count():
    # the operands recorded for KernelBench level3/35_LSTM (batch 10, seq 512, input 128, hidden 256, 6 layers, batch_first)
    h0 = [6, 10, 256]
    weights = [[1024, 128], [1024, 256], [1024], [1024]] + [[1024, 256], [1024, 256], [1024], [1024]] * 5
    ts = TensorShapes(inputs=[[10, 512, 128], h0, h0] + weights, outputs=[[10, 512, 256], h0, h0])
    raw = "[[Tensor(shape=(10, 512, 128), dtype=torch.float32), [Tensor(...), Tensor(...)], [Tensor(...)], True, 6, 0.0, False, False, True], {}]"
    assert EinsumAnalyzer().get_compute_cost("lstm", ts, raw_attributes=raw) == 15_435_038_720 == hand_macs("lstm", 512, 10, 128, 256, 6, 1)
    assert EinsumAnalyzer().get_compute_cost("lstm", ts) == 15_435_038_720          # inferred: h0 batch 10 selects batch_first


def test_kernelbench_level3_39_gru_operands_match_the_hand_count():
    h0 = [6, 10, 256]
    weights = [[768, 128], [768, 256], [768], [768]] + [[768, 256], [768, 256], [768], [768]] * 5
    ts = TensorShapes(inputs=[[512, 10, 128], h0] + weights, outputs=[[512, 10, 256], h0])
    assert EinsumAnalyzer().get_compute_cost("gru", ts) == 11_576_279_040 == hand_macs("gru", 512, 10, 128, 256, 6, 1)


# ── end to end: trace tiny modules on CPU through the full pipeline ──

def _run_pipeline(tmp_path: Path, source: str) -> dict:
    model_file = tmp_path / "model.py"; model_file.write_text(dedent(source))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp32", copy_graph=False)
    assert analysis is not None
    return analysis


def _rnn_source(cls: str, layers: int, bidirectional: bool, batch_first: bool, seq=7, batch=3, inp=5, hidden=6) -> str:
    x_shape = (batch, seq, inp) if batch_first else (seq, batch, inp)
    return f"""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.{cls}({inp}, {hidden}, num_layers={layers}, bidirectional={bidirectional}, batch_first={batch_first})

        def forward(self, x):
            out, _ = self.rnn(x)
            return out

    def get_inputs():
        return [torch.randn{x_shape}]

    def get_init_inputs():
        return []
    """


def _recurrent_module_macs(analysis: dict, kind: str) -> int:
    """MACs attributed to the recurrent module: the single recurrent layer on a tree that prices
    nn.LSTM/nn.GRU as one node (this target), or the sum of the per-layer/per-step sub-layers on a
    tree that expands the recurrence (the earlier lineage: ``Model.lstm.l0.gates`` ...).  Either way the
    module's total must equal the hand formula.  Layer ids are named after the recorded op
    (``Model.lstm`` / ``Model.gru``), not after the module attribute."""
    prefix = f"Model.{kind}"
    layers = analysis.get("layers") or {}
    found = {k: v for k, v in layers.items() if k == prefix or k.startswith(prefix + ".")}
    assert found, f"no layers under {prefix} in {sorted(layers)}"
    return sum(int(v.get("macs", 0)) for v in found.values())


@pytest.mark.parametrize("cls,kind", [("LSTM", "lstm"), ("GRU", "gru")])
@pytest.mark.parametrize("layers,bidirectional", [(1, False), (2, False), (1, True), (2, True)])
@pytest.mark.parametrize("batch_first", [False, True])
def test_end_to_end_recurrent_module_macs_match_the_hand_formula(tmp_path, cls, kind, layers, bidirectional, batch_first):
    seq, batch, inp, hidden = 7, 3, 5, 6
    analysis = _run_pipeline(tmp_path, _rnn_source(cls, layers, bidirectional, batch_first, seq, batch, inp, hidden))
    expected = hand_macs(kind, seq, batch, inp, hidden, layers, 2 if bidirectional else 1)
    assert _recurrent_module_macs(analysis, kind) == expected
    assert int(analysis["total"]["macs"]) >= expected


# ── simple RNN (torch.rnn_tanh / torch.rnn_relu): an earlier fix P2-1 ──
#
# nn.RNN records ONE node named rnn_tanh / rnn_relu with two outputs (output, h_n).  Before the
# repair the recorded name normalised to the tanh/relu activation (the RNNHandler override was never
# reached) and the unexpanded node failed conversion with a pre-AF tensor-shape inconsistency
# (one output name carrying both the hidden-state and the output shape).  Now the recorded name
# normalises to ``rnn`` and the converter expands the node exactly like LSTM/GRU.

def test_recorded_simple_rnn_names_reach_the_recurrent_handler():
    an = EinsumAnalyzer()
    assert an._get_operation_from_name("rnn_tanh") == "rnn" == an._get_operation_from_name("rnn_relu")
    assert an._get_operation_from_name("Model.rnn_tanh_3") == "rnn"
    assert an._get_operation_from_name("tanh") == "tanh" and an._get_operation_from_name("relu") == "relu"
    seq, batch, inp, hidden = 7, 3, 5, 6
    for name in ("rnn_tanh", "rnn_relu", "torch.rnn_tanh"):
        ts, raw = recorded_operands("rnn", seq, batch, inp, hidden, 2, 2, batch_first=True)
        assert an.get_compute_cost(name, ts, raw_attributes=raw) == hand_macs("rnn", seq, batch, inp, hidden, 2, 2)


def _simple_rnn_source(nonlinearity: str, layers: int, bidirectional: bool, batch_first: bool,
                       seq=7, batch=3, inp=5, hidden=6, bias=True) -> str:
    x_shape = (batch, seq, inp) if batch_first else (seq, batch, inp)
    return f"""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN({inp}, {hidden}, num_layers={layers}, bidirectional={bidirectional},
                              batch_first={batch_first}, bias={bias}, nonlinearity="{nonlinearity}")

        def forward(self, x):
            return self.rnn(x)[0]

    def get_inputs():
        return [torch.randn{x_shape}]

    def get_init_inputs():
        return []
    """


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
@pytest.mark.parametrize("layers,bidirectional", [(1, False), (2, False), (1, True), (2, True)])
@pytest.mark.parametrize("batch_first", [False, True])
def test_end_to_end_simple_rnn_is_expanded_and_matches_the_hand_formula(tmp_path, nonlinearity, layers, bidirectional, batch_first):
    seq, batch, inp, hidden = 7, 3, 5, 6
    analysis = _run_pipeline(tmp_path, _simple_rnn_source(nonlinearity, layers, bidirectional, batch_first, seq, batch, inp, hidden))
    node = f"Model.rnn_{nonlinearity}"
    dirs = 2 if bidirectional else 1
    assert node not in analysis["layers"]                       # expanded, no opaque whole node
    assert f"{node}.l0{'d0' if dirs > 1 else ''}.ih_linear" in analysis["layers"]
    gates = {k: v for k, v in analysis["layers"].items() if k.startswith(node + ".") and k.endswith(".gates")}
    assert len(gates) == layers * dirs
    assert _recurrent_module_macs(analysis, f"rnn_{nonlinearity}") == hand_macs("rnn", seq, batch, inp, hidden, layers, dirs)
    assert int(analysis["total"]["macs"]) == hand_macs("rnn", seq, batch, inp, hidden, layers, dirs)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_reviewer_control_two_layer_bidirectional_bias_free_rnn_is_1520_macs(tmp_path, nonlinearity):
    """The independent review's capture: nn.RNN(3, 4, num_layers=2, bidirectional=True, batch_first=True,
    bias=False), input [2, 5, 3]; expected 5*2*[2(4*3+4*4) + 2(4*8+4*4)] = 1,520 recurrent MACs."""
    from solar.perf.perf_model import EinsumGraphPerfModel
    analysis = _run_pipeline(tmp_path, _simple_rnn_source(nonlinearity, 2, True, True, seq=5, batch=2, inp=3, hidden=4, bias=False))
    node = f"Model.rnn_{nonlinearity}"
    assert _recurrent_module_macs(analysis, f"rnn_{nonlinearity}") == 1_520 == int(analysis["total"]["macs"])
    final = analysis["layers"][f"{node}.l1.cat"]
    assert final["tensor_shapes"]["outputs"] == [[2, 5, 8]]        # (output, h_n) no longer share one name
    assert all(l.get("status") != "incomplete" for l in analysis["layers"].values())
    perf = EinsumGraphPerfModel(debug=False).predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"),
                                                   arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert perf is not None and int(perf["workload"]["total_macs"]) == 1_520


# ── consumed state outputs h_n / c_n (an earlier fix P2-2) ──
#
# The expansion's final layer is the sequence output only; a consumer of the hidden-state output
# (slot 1; LSTM's c_n is slot 2) used to be rewired onto `<final>.Output_1`, which nothing emits
# (AF invariant failure: orphan read).  A consumed state now has its own producer: per cell a
# `select` of the last step of that cell's `.gates` tensor, stacked over the cells in PyTorch's
# layer*directions+direction order -> [num_layers*num_directions, B, H].  Both are data-movement
# layers: zero MACs, zero other_ops; recurrent MACs are unchanged.  Sequence-only modules emit no
# state layers, so their accounting (and the KB LSTM/GRU rows) is untouched.

def _state_consumer_source(cls: str, forward: str, extra: str = "") -> str:
    return f"""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.{cls}(3, 4, num_layers=2, bidirectional=True, batch_first=True, bias=False{extra})

        def forward(self, x):
            {forward}

    def get_inputs():
        return [torch.zeros(2, 5, 3)]

    def get_init_inputs():
        return []
    """


def _state_layers(analysis: dict, node: str):
    return {k: v for k, v in analysis["layers"].items() if k.startswith(node + ".") and (".h_n" in k or ".c_n" in k)}


@pytest.mark.parametrize("cls,node,extra,expected", [
    ("RNN", "Model.rnn_tanh", ", nonlinearity='tanh'", 1_520),
    ("RNN", "Model.rnn_relu", ", nonlinearity='relu'", 1_520),
    ("GRU", "Model.gru", "", 4_560),
    ("LSTM", "Model.lstm", "", 6_080),
])
def test_consumed_hidden_state_has_its_own_producer(tmp_path, cls, node, extra, expected):
    """h_n-only consumer: `self.rnn(x)[1]` -> relu.  Converts, the state is produced by
    `<node>.h_n.stack` with shape [4, 2, 4] (2 layers x 2 directions, batch 2, hidden 4), the consumer
    reads it (not the [2, 5, 8] sequence tensor), recurrent MACs are preserved through analysis and perf."""
    from solar.perf.perf_model import EinsumGraphPerfModel
    fwd = "return self.rnn(x)[1][0].relu()" if cls == "LSTM" else "return self.rnn(x)[1].relu()"
    analysis = _run_pipeline(tmp_path, _state_consumer_source(cls, fwd, extra))
    stack = analysis["layers"][f"{node}.h_n.stack"]
    assert stack["tensor_shapes"]["outputs"] == [[4, 2, 4]] and stack["macs"] == 0 and stack["other_ops"] == 0
    assert stack["connections"]["outputs"] == ["Model.relu"]
    relu = analysis["layers"]["Model.relu"]
    assert relu["connections"]["inputs"] == [f"{node}.h_n.stack"]
    assert relu["tensor_shapes"]["inputs"] == [[4, 2, 4]]
    selects = {k: v for k, v in _state_layers(analysis, node).items() if k.endswith(".h_n")}
    assert len(selects) == 4 and all(v["macs"] == 0 and v["other_ops"] == 0 and v["tensor_shapes"]["outputs"] == [[2, 4]] for v in selects.values())
    assert _recurrent_module_macs(analysis, node.split(".", 1)[1]) == expected == int(analysis["total"]["macs"])
    perf = EinsumGraphPerfModel(debug=False).predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"),
                                                   arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert int(perf["workload"]["total_macs"]) == expected
    # the sequence output is NOT consumed here: the final cat has no consumer, the state is never aliased to it
    assert analysis["layers"][f"{node}.l1.cat"]["connections"]["outputs"] == []


@pytest.mark.parametrize("cls,node,extra,expected", [
    ("RNN", "Model.rnn_tanh", ", nonlinearity='tanh'", 1_520),
    ("RNN", "Model.rnn_relu", ", nonlinearity='relu'", 1_520),
    ("GRU", "Model.gru", "", 4_560),
])
def test_dual_output_consumer_reads_sequence_and_hidden_state(tmp_path, cls, node, extra, expected):
    analysis = _run_pipeline(tmp_path, _state_consumer_source(cls, "out, h = self.rnn(x); return out.sum() + h.sum()", extra))
    assert analysis["layers"][f"{node}.l1.cat"]["connections"]["outputs"] == ["Model.sum"]
    assert analysis["layers"][f"{node}.h_n.stack"]["connections"]["outputs"] == ["Model.sum_1"]
    assert analysis["layers"]["Model.sum"]["tensor_shapes"]["inputs"] == [[2, 5, 8]]
    assert analysis["layers"]["Model.sum_1"]["tensor_shapes"]["inputs"] == [[4, 2, 4]]
    assert _recurrent_module_macs(analysis, node.split(".", 1)[1]) == expected


def test_lstm_cell_state_and_dual_outputs(tmp_path):
    """LSTM: c_n (slot 2) consumed alone, and (output, (h_n, c_n)) all consumed.  The expansion has no
    separate cell-state tensor, so c_n is represented from the per-cell state tensor (shape/identity)."""
    (tmp_path / "c").mkdir(); (tmp_path / "d").mkdir()
    a = _run_pipeline(tmp_path / "c", _state_consumer_source("LSTM", "return self.rnn(x)[1][1].relu()"))
    assert a["layers"]["Model.lstm.c_n.stack"]["tensor_shapes"]["outputs"] == [[4, 2, 4]]
    assert a["layers"]["Model.relu"]["connections"]["inputs"] == ["Model.lstm.c_n.stack"]
    assert "Model.lstm.h_n.stack" not in a["layers"]                     # only the consumed slot is produced
    assert _recurrent_module_macs(a, "lstm") == 6_080
    b = _run_pipeline(tmp_path / "d", _state_consumer_source("LSTM", "out, (h, c) = self.rnn(x); return out.sum() + h.sum() + c.sum()"))
    assert b["layers"]["Model.lstm.l1.cat"]["connections"]["outputs"] == ["Model.sum"]
    assert b["layers"]["Model.lstm.h_n.stack"]["connections"]["outputs"] == ["Model.sum_1"]
    assert b["layers"]["Model.lstm.c_n.stack"]["connections"]["outputs"] == ["Model.sum_2"]
    assert _recurrent_module_macs(b, "lstm") == 6_080 == int(b["total"]["macs"])


@pytest.mark.parametrize("cls,node,extra", [("RNN", "Model.rnn_tanh", ", nonlinearity='tanh'"), ("LSTM", "Model.lstm", ""), ("GRU", "Model.gru", "")])
def test_sequence_only_modules_emit_no_state_layers(tmp_path, cls, node, extra):
    analysis = _run_pipeline(tmp_path, _state_consumer_source(cls, "return self.rnn(x)[0]", extra))
    assert not _state_layers(analysis, node)
    assert analysis["layers"][f"{node}.l1.cat"]["connections"]["outputs"] == []


# ── a consumer reading BOTH outputs (state-consumer-p2-3) ──
#
# The captured node appears once per consumed output slot in a consumer's predecessor list.  The
# state producer must be SUBSTITUTED into the slot where the node sat for that output, in operand
# order (sequence slot -> final sequence node, h_n slot -> h_n producer, c_n slot -> c_n producer),
# never appended: the AF builder pairs the consumer's roles with its predecessors in order, so an
# appended producer left a third predecessor and an uncovered state read.

def _dual_read_source(cls: str, forward: str, extra: str = "") -> str:
    return f"""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.{cls}(3, 2, 1, batch_first=True, bias=False{extra})

        def forward(self, x):
            {forward}

    def get_inputs():
        return [torch.zeros(1, 2, 3)]

    def get_init_inputs():
        return []
    """


@pytest.mark.parametrize("cls,node,extra,state,forward,recurrent", [
    ("RNN", "Model.rnn_tanh", ", nonlinearity='tanh'", "h_n", "sequence, h = self.rnn(x); return torch.bmm(h, sequence)", 20),
    ("RNN", "Model.rnn_relu", ", nonlinearity='relu'", "h_n", "sequence, h = self.rnn(x); return torch.bmm(h, sequence)", 20),
    ("GRU", "Model.gru", "", "h_n", "sequence, h = self.rnn(x); return torch.bmm(h, sequence)", 60),
    ("LSTM", "Model.lstm", "", "c_n", "sequence, (h, c) = self.rnn(x); return torch.bmm(c, sequence)", 80),
])
def test_consumer_reading_state_and_sequence_gets_producers_in_operand_order(tmp_path, cls, node, extra, state, forward, recurrent):
    """The review's controls: bmm(state [1,1,2], sequence [1,2,2]) = 4 MACs on top of the recurrence
    (S*B*H*(I+H) = 2*1*2*5 = 20 per gate)."""
    from solar.perf.perf_model import EinsumGraphPerfModel
    a = _run_pipeline(tmp_path, _dual_read_source(cls, forward, extra))
    bmm = a["layers"]["Model.bmm"]
    assert bmm["connections"]["inputs"] == [f"{node}.{state}.stack", f"{node}.l0.gates"]      # operand order, two predecessors
    assert bmm["tensor_shapes"]["inputs"] == [[1, 1, 2], [1, 2, 2]] and bmm["macs"] == 4
    assert int(a["total"]["macs"]) == recurrent + 4
    perf = EinsumGraphPerfModel(debug=False).predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"),
                                                   arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert int(perf["workload"]["total_macs"]) == recurrent + 4
    assert a["layers"][f"{node}.{state}.stack"]["connections"]["outputs"] == ["Model.bmm"]
    assert a["layers"][f"{node}.l0.gates"]["connections"]["outputs"] == ["Model.bmm"]


def test_reversed_order_control_and_lstm_dual_state_readers(tmp_path):
    (tmp_path / "r").mkdir(); (tmp_path / "l").mkdir()
    a = _run_pipeline(tmp_path / "r", _dual_read_source("RNN", "sequence, h = self.rnn(x); return torch.bmm(sequence, h.transpose(1, 2))", ", nonlinearity='tanh'"))
    bmm = a["layers"]["Model.bmm"]
    assert bmm["connections"]["inputs"] == ["Model.rnn_tanh.l0.gates", "Model.transpose"] and int(a["total"]["macs"]) == 24
    assert a["layers"]["Model.transpose"]["connections"]["inputs"] == ["Model.rnn_tanh.h_n.stack"]
    b = _run_pipeline(tmp_path / "l", _dual_read_source("LSTM", "sequence, (h, c) = self.rnn(x); return torch.bmm(h, sequence) + torch.bmm(c, sequence)"))
    assert b["layers"]["Model.bmm"]["connections"]["inputs"] == ["Model.lstm.h_n.stack", "Model.lstm.l0.gates"]
    assert b["layers"]["Model.bmm_1"]["connections"]["inputs"] == ["Model.lstm.c_n.stack", "Model.lstm.l0.gates"]
    assert int(b["total"]["macs"]) == 80 + 4 + 4


# ── repeated state operands in one consumer (state-consumer-p2-4) ──
#
# The AF builder pairs a consumer's non-weight roles with its predecessors POSITIONALLY, so the
# predecessor list must carry one entry per predecessor-consuming slot, repeated producers included
# (the generic converter already does this: cat(y, y) lists the producer twice).  Deduplicating made
# cat(h, h, c) pair the second h slot with c and invent a fresh weight for the last slot — a silently
# wrong graph with no marker.  A slot that cannot be aligned now marks the layer incomplete.

def _af_einsum(tmp_path, layer):
    af = yaml.safe_load((tmp_path / "einsum" / "af_einsum_graph.yaml").read_text())
    return next(e for e in af["workload"]["einsums"] if e["name"] == layer.replace(".", "_"))


@pytest.mark.parametrize("cls,node,extra,forward,layer,expected_conn,expected_reads", [
    ("LSTM", "Model.lstm", "", "sequence, (h, c) = self.rnn(x); return torch.cat((h, h, c), dim=0)", "Model.cat",
     ["Model.lstm.h_n.stack", "Model.lstm.h_n.stack", "Model.lstm.c_n.stack"], ["Model_lstm_h_n_stack", "Model_lstm_h_n_stack", "Model_lstm_c_n_stack"]),
    ("LSTM", "Model.lstm", "", "sequence, (h, c) = self.rnn(x); return torch.einsum('abc,abc,abc->abc', h, h, c)", "Model.einsum",
     ["Model.lstm.h_n.stack", "Model.lstm.h_n.stack", "Model.lstm.c_n.stack"], ["Model_lstm_h_n_stack", "Model_lstm_h_n_stack", "Model_lstm_c_n_stack"]),
    ("GRU", "Model.gru", "", "sequence, h = self.rnn(x); return torch.cat((h, h), dim=0)", "Model.cat",
     ["Model.gru.h_n.stack", "Model.gru.h_n.stack"], ["Model_gru_h_n_stack", "Model_gru_h_n_stack"]),
    ("RNN", "Model.rnn_tanh", ", nonlinearity='tanh'", "sequence, h = self.rnn(x); return torch.cat((h, sequence[:, :1], h), dim=0)", "Model.cat",
     ["Model.rnn_tanh.h_n.stack", "Model.__getitem__", "Model.rnn_tanh.h_n.stack"], ["Model_rnn_tanh_h_n_stack", "Model___getitem__", "Model_rnn_tanh_h_n_stack"]),
])
def test_repeated_state_operands_keep_one_predecessor_per_slot(tmp_path, cls, node, extra, forward, layer, expected_conn, expected_reads):
    analysis = _run_pipeline(tmp_path, _dual_read_source(cls, forward, extra))
    l = analysis["layers"][layer]
    assert l["connections"]["inputs"] == expected_conn                        # one entry per slot, duplicates kept
    assert len(l["connections"]["inputs"]) == len(l["tensor_types"]["inputs"])
    e = _af_einsum(tmp_path, layer)
    reads = [t["name"] for t in e["tensor_accesses"] if not t.get("output")]
    # The AF representation merges identical accesses AFTER positional role matching (the generic
    # cat(y, y) control lists its producer once and renames both roles to it); so the accesses are the
    # distinct producers in slot order, the repeated slot's role is renamed to the SAME producer, and
    # no synthetic weight W<n> exists.
    distinct = list(dict.fromkeys(expected_reads))
    assert reads == distinct, reads
    assert not any(r.startswith("W") and r[1:].isdigit() for r in reads)
    repeated = expected_reads[0]
    assert e["renames"]["input"] == repeated and e["renames"]["weight"] == expected_reads[1]
    assert "status" not in l and analysis["total"].get("status") is None
    assert all(v.get("status") != "incomplete" for v in analysis["layers"].values())


def test_unalignable_consumer_slot_is_refused_as_incomplete(tmp_path):
    """Invariant: a predecessor-consuming slot whose producer is not a graph layer cannot be aligned
    positionally; the fixer marks the consumer incomplete (predecessor_slots_unaligned) instead of
    leaving the AF builder to invent a weight, and the analyzer carries the marker."""
    conv = PyTorchToEinsum()
    final, hn = "Model.rnn_tanh.l0.gates", "Model.rnn_tanh.h_n.stack"
    result = {"model_name": "synthetic", "layers": {
        final: {"type": "add", "einsum_equation": "BSH->BSH", "is_real_einsum": False, "tensor_names": {"inputs": [], "outputs": [final + ".Output"]},
                "tensor_types": {"inputs": [], "outputs": ["output"]}, "tensor_shapes": {"inputs": [], "outputs": [[1, 2, 2]]}, "connections": {"inputs": [], "outputs": ["Model.cat"]}},
        hn: {"type": "stack", "einsum_equation": "BH->ZBH", "is_real_einsum": False, "tensor_names": {"inputs": [], "outputs": [hn + ".Output"]},
             "tensor_types": {"inputs": [], "outputs": ["output"]}, "tensor_shapes": {"inputs": [], "outputs": [[1, 1, 2]]}, "connections": {"inputs": [], "outputs": ["Model.cat"]}},
        "Model.cat": {"type": "cat", "einsum_equation": "ZBH,ZBH,ZBH->YBH", "is_real_einsum": False,
                      "tensor_names": {"inputs": ["Model.rnn_tanh.Output_1", "ghost.Output", "Model.rnn_tanh.Output_1"], "outputs": ["Model.cat.Output"]},
                      "tensor_types": {"inputs": ["input", "input", "input"], "outputs": ["output"]},
                      "tensor_shapes": {"inputs": [[1, 1, 2], [1, 1, 2], [1, 1, 2]], "outputs": [[3, 1, 2]]},
                      "connections": {"inputs": ["Model.rnn_tanh", "ghost", "Model.rnn_tanh"], "outputs": []}},
    }}
    conv._fix_split_connections(result, {"Model.rnn_tanh": final}, {"Model.rnn_tanh": {}}, {"Model.rnn_tanh": {1: hn}})
    cat = result["layers"]["Model.cat"]
    assert cat["tensor_names"]["inputs"] == [hn + ".Output", "ghost.Output", hn + ".Output"]
    assert cat["connections"]["inputs"] == [hn, hn]                                   # the two alignable slots, repeated
    assert cat["status"] == "incomplete" and cat["incomplete_reason"] == "predecessor_slots_unaligned"
    assert cat["predecessor_slots_unaligned"] == [{"slot": 1, "name": "ghost.Output"}]
    gpath = tmp_path / "einsum_graph.yaml"; gpath.write_text(yaml.safe_dump(result, sort_keys=False))
    a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis"), copy_graph=False)
    assert a["layers"]["Model.cat"]["status"] == "incomplete" and a["layers"]["Model.cat"]["incomplete_reason"] == "predecessor_slots_unaligned"
    assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"]["Model.cat"] == "predecessor_slots_unaligned"


# ── the mismatch marker refuses AF export (state-consumer-p2-5) ──
#
# The AF builder pairs non-weight roles with predecessors positionally and synthesizes a weight when
# it runs out; it never inspected the predecessor_slots_unaligned marker, so an unaligned consumer was
# still exported as an ordinary AF workload with an invented W<n>.  The builder now refuses BEFORE
# role matching (typed AFExportRefused carrying the layers and their incomplete reason) and the
# converter serializes no AF file for such a graph, recording the refusal in the einsum graph.

def _mismatch_graph(slot):
    """The review's control: an LSTM three-operand einsum consumer (h, h, c) rewired through the fixer,
    with the producer of one slot replaced by a non-existent layer (slot None = valid control)."""
    final, hn, cn = "Model.lstm.l0.gates", "Model.lstm.h_n.stack", "Model.lstm.c_n.stack"
    def leaf(lid, shape):
        return {"type": "stack", "einsum_equation": "BH->ZBH", "is_real_einsum": False, "tensor_names": {"inputs": [], "outputs": [lid + ".Output"]},
                "tensor_types": {"inputs": [], "outputs": ["output"]}, "tensor_shapes": {"inputs": [], "outputs": [shape]},
                "operands": {"Input": ["B", "H"], "Output": ["Z", "B", "H"]}, "connections": {"inputs": [], "outputs": ["Model.einsum"]}}
    g = {"model_name": "mismatch", "layers": {
        final: leaf(final, [1, 2, 2]), hn: leaf(hn, [1, 1, 2]), cn: leaf(cn, [1, 1, 2]),
        "Model.einsum": {"type": "einsum", "einsum_equation": "ABC,ABC,ABC->ABC", "is_real_einsum": True,
                         "tensor_names": {"inputs": ["Model.lstm.Output_1", "Model.lstm.Output_1", "Model.lstm.Output_2"], "outputs": ["Model.einsum.Output"]},
                         "tensor_types": {"inputs": ["input", "input", "input"], "outputs": ["output"]},
                         "tensor_shapes": {"inputs": [[1, 1, 2]] * 3, "outputs": [[1, 1, 2]]},
                         "operands": {"Input": ["A", "B", "C"], "Input_1": ["A", "B", "C"], "Input_2": ["A", "B", "C"], "Output": ["A", "B", "C"]},
                         "connections": {"inputs": ["Model.lstm", "Model.lstm", "Model.lstm"], "outputs": []}},
    }}
    if slot is not None:
        g["layers"]["Model.einsum"]["tensor_names"]["inputs"][slot] = "ghost.Output"
        g["layers"]["Model.einsum"]["connections"]["inputs"][slot] = "ghost"
    PyTorchToEinsum()._fix_split_connections(g, {"Model.lstm": final}, {"Model.lstm": {}}, {"Model.lstm": {1: hn, 2: cn}})
    return g


def test_valid_repeated_state_consumer_still_exports_af(tmp_path):
    from solar.einsum.af_graph_builder import build_af_graph_from_dict
    g = _mismatch_graph(None)
    assert "status" not in g["layers"]["Model.einsum"]
    af = build_af_graph_from_dict(g)
    e = next(e for e in af["workload"]["einsums"] if e["name"] == "Model_einsum")
    reads = [t["name"] for t in e["tensor_accesses"] if not t.get("output")]
    assert reads == ["Model_lstm_h_n_stack", "Model_lstm_c_n_stack"] and not any(r.startswith("W") and r[1:].isdigit() for r in reads)


@pytest.mark.parametrize("slot", [0, 1, 2], ids=["missing-first", "missing-middle", "missing-last"])
def test_unaligned_slot_refuses_af_export_with_the_incomplete_reason(tmp_path, slot):
    from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_dict
    g = _mismatch_graph(slot)
    cons = g["layers"]["Model.einsum"]
    assert cons["status"] == "incomplete" and cons["predecessor_slots_unaligned"] == [{"slot": slot, "name": "ghost.Output"}]
    with pytest.raises(AFExportRefused) as info:
        build_af_graph_from_dict(g)
    assert info.value.incomplete_reasons == {"Model.einsum": "predecessor_slots_unaligned"}
    assert "Model.einsum" in str(info.value) and "predecessor_slots_unaligned" in str(info.value) and "ghost.Output" in str(info.value)
    gpath = tmp_path / "einsum_graph.yaml"; gpath.write_text(yaml.safe_dump(g, sort_keys=False))
    a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis"), copy_graph=False)
    assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"]["Model.einsum"] == "predecessor_slots_unaligned"


def test_converter_serializes_no_af_file_for_a_refused_graph(tmp_path, monkeypatch):
    """Through the converter's export boundary: convert() on a capture whose expansion rewiring leaves an
    unaligned slot writes the einsum graphs with `af_export_refused` and NO af_einsum_graph.yaml."""
    import solar.einsum.pytorch_to_einsum as P
    real = P.PyTorchToEinsum._fix_split_connections

    def sabotage(self, result, node_id_remap, expanded_input_map=None, expanded_output_slots=None):
        if "Model.cat" in result["layers"]:
            # emulate a producer the trace lost for the consumer's middle slot BEFORE the alignment pass
            result["layers"]["Model.cat"]["tensor_names"]["inputs"][1] = "ghost.Output"
            result["layers"]["Model.cat"]["connections"]["inputs"][1] = "ghost"
        real(self, result, node_id_remap, expanded_input_map, expanded_output_slots)
    monkeypatch.setattr(P.PyTorchToEinsum, "_fix_split_connections", sabotage)
    (tmp_path / "model.py").write_text(dedent(_dual_read_source("LSTM", "sequence, (h, c) = self.rnn(x); return torch.cat((h, h, c), dim=0)")))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(graph_dir))
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    (einsum_dir / "af_einsum_graph.yaml").write_text("stale: true\n")       # a stale file from an earlier run must not survive
    g = PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert not (einsum_dir / "af_einsum_graph.yaml").exists()
    assert (einsum_dir / "af_einsum_graph.REFUSED.yaml.stale").exists()
    assert g["af_export_refused"]["incomplete_reasons"] == {"Model.cat": "predecessor_slots_unaligned"}
    written = yaml.safe_load((einsum_dir / "einsum_graph_renamed.yaml").read_text())
    assert written["af_export_refused"]["layers"]["Model.cat"]["predecessor_slots_unaligned"] == [{"slot": 1, "name": "ghost.Output"}]
    assert written["layers"]["Model.cat"]["status"] == "incomplete"


# ── correspondence check for UNMARKED graphs (state-consumer-p2-6) ──
#
# The P2-5 refusal was gated on the converter's marker.  A saved stage-2 graph from the round-4 head
# (dedup bug) carries three in-graph state slots with two predecessor entries and no marker; through the
# public build_af_graph_from_yaml path it still exported with an invented W3.  The builder now checks
# correspondence: every slot that names an in-graph non-start producer must own a predecessor entry
# (repeated slots need repeated entries); boundary (start-fed), weight and unresolved slots keep their
# predecessor-less representation.

STATE_FIX = Path(__file__).resolve().parent / "fixtures" / "rnn_state_consumers"


def _fresh_valid_graph():
    g = _mismatch_graph(None)                      # three evidenced slots, three predecessor entries, no marker
    assert g["layers"]["Model.einsum"]["connections"]["inputs"] == ["Model.lstm.h_n.stack", "Model.lstm.h_n.stack", "Model.lstm.c_n.stack"]
    return g


@pytest.mark.parametrize("fixture", ["round4_lstm_einsum_hhc_unmarked.yaml", "round4_lstm_einsum_hhc_saved.yaml"])
def test_historical_unmarked_round4_graph_is_refused_through_the_yaml_path(tmp_path, fixture):
    from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_yaml
    g = yaml.safe_load((STATE_FIX / fixture).read_text())
    cons = g["layers"]["Model.einsum"]
    assert "predecessor_slots_unaligned" not in cons and "status" not in cons                 # unmarked
    assert cons["tensor_names"]["inputs"] == ["Model.lstm.h_n.stack.Output"] * 2 + ["Model.lstm.c_n.stack.Output"]
    assert cons["connections"]["inputs"] == ["Model.lstm.h_n.stack", "Model.lstm.c_n.stack"]   # the round-4 dedup
    src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False)); out = tmp_path / "export.yaml"
    with pytest.raises(AFExportRefused) as info:
        build_af_graph_from_yaml(src, out)
    assert not out.exists()
    # positional (state-consumer-p2-7): slot 0 consumes h_n correctly, slot 1 names h_n but the emitter would consume c_n there
    assert info.value.layers["Model.einsum"]["predecessor_slots_unaligned"] == [{"slot": 1, "name": "Model.lstm.h_n.stack.Output", "producer": "Model.lstm.h_n.stack", "consumed": "Model.lstm.c_n.stack"}]
    assert info.value.incomplete_reasons == {"Model.einsum": "predecessor_slots_unaligned"}


@pytest.mark.parametrize("path", ["dict", "yaml"])
@pytest.mark.parametrize("drop", [0, 1, 2], ids=["drop-first", "drop-middle", "drop-last"])
def test_unmarked_predecessor_removed_from_fresh_valid_metadata_is_refused(tmp_path, path, drop):
    from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_dict, build_af_graph_from_yaml
    g = _fresh_valid_graph()
    del g["layers"]["Model.einsum"]["connections"]["inputs"][drop]          # names intact, predecessor removed, no marker
    with pytest.raises(AFExportRefused) as info:
        if path == "dict":
            build_af_graph_from_dict(g)
        else:
            src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False))
            build_af_graph_from_yaml(src, tmp_path / "export.yaml")
    assert not (tmp_path / "export.yaml").exists()
    bad = info.value.layers["Model.einsum"]["predecessor_slots_unaligned"]
    assert len(bad) == 1 and bad[0]["producer"] in ("Model.lstm.h_n.stack", "Model.lstm.c_n.stack")
    assert "W3" not in str(info.value)


def test_valid_three_entry_control_still_exports_and_boundary_slots_stay_exempt(tmp_path):
    from solar.einsum.af_graph_builder import build_af_graph_from_dict, build_af_graph_from_yaml, _evidenced_slot_correspondence
    g = _fresh_valid_graph()
    af = build_af_graph_from_dict(g)
    reads = [t["name"] for t in next(e for e in af["workload"]["einsums"] if e["name"] == "Model_einsum")["tensor_accesses"] if not t.get("output")]
    assert reads == ["Model_lstm_h_n_stack", "Model_lstm_c_n_stack"] and not any(r.startswith("W") and r[1:].isdigit() for r in reads)
    src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False)); build_af_graph_from_yaml(src, tmp_path / "export.yaml")
    assert (tmp_path / "export.yaml").exists()
    # weight and unresolved slots consume no predecessor entry; a start-fed slot keeps its predecessor-less
    # representation only when the emitter has no entry left to hand it (state-consumer-p2-7: with an entry present at
    # its position the emitter WOULD consume it, so [Model.p] alone is a misalignment, not an exemption)
    layers = {"start": {"type": "start"}, "Model.p": {"type": "add"},
              "Model.c": {"tensor_names": {"inputs": ["start.Output", "Model.c.Weight", "Model.c.Unresolved", "Model.p.Output"]},
                          "tensor_types": {"inputs": ["input", "weight", "unresolved", "input"]}, "connections": {"inputs": ["start", "Model.p"]}}}
    assert _evidenced_slot_correspondence(layers, layers["Model.c"]) == []
    layers["Model.c"]["connections"]["inputs"] = ["Model.p"]
    assert _evidenced_slot_correspondence(layers, layers["Model.c"]) == [{"slot": 0, "name": "start.Output", "producer": "start", "consumed": "Model.p"}]
    layers["Model.c"]["connections"]["inputs"] = []
    assert _evidenced_slot_correspondence(layers, layers["Model.c"]) == [{"slot": 3, "name": "Model.p.Output", "producer": "Model.p"}]
    # boundary-only and trailing-boundary representations stay predecessor-less
    layers["Model.c"]["tensor_names"]["inputs"] = ["Model.p.Output", "start.Output"]; layers["Model.c"]["tensor_types"]["inputs"] = ["input", "input"]
    layers["Model.c"]["connections"]["inputs"] = ["Model.p"]
    assert _evidenced_slot_correspondence(layers, layers["Model.c"]) == []


# ── positional correspondence (state-consumer-p2-7) and builder-discovered refusals carry incomplete status (state-consumer-p2-8) ──
#
# Round 8: the correspondence check was a producer MULTISET test while the emitter consumes predecessors
# POSITIONALLY, so (i) a boundary slot without an entry swallowed the next slot's producer (W3 invented for
# the last state slot — the original failure mechanism) and (ii) a permuted predecessor list exported Q @ P
# for P @ Q.  The check now walks slots and entries in the emitter's lockstep.  A refusal the builder
# discovers on an unmarked graph is copied by the converter into the layer's own markers before the stage-2
# files are written, so analysis layer/total and the perf workload carry status "incomplete".

def _reads(af, einsum_name):
    e = next(e for e in af["workload"]["einsums"] if e["name"] == einsum_name)
    return [(t["name"], t["projection"]) for t in e["tensor_accesses"] if not t.get("output")], \
           next(t["projection"] for t in e["tensor_accesses"] if t.get("output"))


@pytest.mark.parametrize("fixture,shape_consistent", [("round8_mixed_boundary_without_edge.yaml", True),    # a Boundary start layer of the slot's shape
                                                      ("round8_mixed_boundary_start_slot.yaml", False)])   # the graph's own model input (shape [1, 2, 3])
def test_boundary_slot_without_its_entry_is_refused_not_handed_the_next_producer(tmp_path, fixture, shape_consistent):
    from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_yaml
    consumed = "Model.lstm.h_n.stack"
    g = yaml.safe_load((STATE_FIX / fixture).read_text())
    cons = g["layers"]["Model.einsum"]
    boundary = cons["tensor_names"]["inputs"][0].rsplit(".Output", 1)[0]
    assert g["layers"][boundary]["type"] == "start"
    assert (g["layers"][boundary]["tensor_shapes"]["outputs"][0] == cons["tensor_shapes"]["inputs"][0]) is shape_consistent
    assert cons["connections"]["inputs"] == ["Model.lstm.h_n.stack", "Model.lstm.c_n.stack"] and "status" not in cons
    src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False)); out = tmp_path / "export.yaml"
    with pytest.raises(AFExportRefused) as info:
        build_af_graph_from_yaml(src, out)
    assert not out.exists()
    assert info.value.layers["Model.einsum"]["predecessor_slots_unaligned"] == [{"slot": 0, "name": boundary + ".Output", "producer": boundary, "consumed": consumed}]
    assert "W3" not in str(info.value) and "consume" in info.value.layers["Model.einsum"]["evidence"]


def test_boundary_slot_with_its_explicit_entry_exports_boundary_h_c_in_order(tmp_path):
    from solar.einsum.af_graph_builder import build_af_graph_from_dict, build_af_graph_from_yaml
    g = yaml.safe_load((STATE_FIX / "round8_mixed_boundary_explicit_edge.yaml").read_text())
    assert g["layers"]["Model.einsum"]["connections"]["inputs"] == ["Boundary", "Model.lstm.h_n.stack", "Model.lstm.c_n.stack"]
    reads, _ = _reads(build_af_graph_from_dict(g), "Model_einsum")
    assert [n for n, _ in reads] == ["Boundary", "Model_lstm_h_n_stack", "Model_lstm_c_n_stack"]
    src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False)); build_af_graph_from_yaml(src, tmp_path / "export.yaml")
    assert (tmp_path / "export.yaml").exists()


def test_two_matrix_consumer_exports_p_times_q_in_slot_order():
    from solar.einsum.af_graph_builder import build_af_graph_from_dict
    g = yaml.safe_load((STATE_FIX / "round8_matrix_valid.yaml").read_text())
    assert g["layers"]["C"]["einsum_equation"] == "IK,KJ->IJ" and g["layers"]["C"]["connections"]["inputs"] == ["P", "Q"]
    reads, out = _reads(build_af_graph_from_dict(g), "C")
    assert [n for n, _ in reads] == ["P", "Q"]
    (p_name, p_proj), (q_name, q_proj) = reads
    # P @ Q: P is read over (I, K) and Q over (K, J) — P shares its first axis with the output, Q its second,
    # and the contracted axis is P's second == Q's first
    assert p_proj[0] == out[0] and q_proj[1] == out[1] and p_proj[1] == q_proj[0] and p_proj[0] != q_proj[1]


def test_two_matrix_consumer_with_permuted_predecessors_is_refused(tmp_path):
    from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_dict, build_af_graph_from_yaml
    g = yaml.safe_load((STATE_FIX / "round8_matrix_permuted.yaml").read_text())
    assert g["layers"]["C"]["tensor_names"]["inputs"] == ["P.Output", "Q.Output"] and g["layers"]["C"]["connections"]["inputs"] == ["Q", "P"]
    with pytest.raises(AFExportRefused) as info:
        build_af_graph_from_dict(g)
    assert info.value.layers["C"]["predecessor_slots_unaligned"] == [{"slot": 0, "name": "P.Output", "producer": "P", "consumed": "Q"}]
    src = tmp_path / "graph.yaml"; src.write_text(yaml.safe_dump(g, sort_keys=False))
    with pytest.raises(AFExportRefused):
        build_af_graph_from_yaml(src, tmp_path / "export.yaml")
    assert not (tmp_path / "export.yaml").exists()


@pytest.mark.parametrize("slot", [0, 1, 2], ids=["lost-first", "lost-middle", "lost-last"])
def test_builder_discovered_refusal_carries_incomplete_status_to_layer_total_and_perf(tmp_path, monkeypatch, slot):
    """The reviewer's three saved-capture fault cases: a predecessor entry lost AFTER the converter's alignment
    pass (so no converter marker); the builder refuses and the converter must propagate the refusal into the
    layer's markers so stage-2 files, analysis layer, analysis total and perf workload all read incomplete."""
    import solar.einsum.pytorch_to_einsum as P
    from solar.perf.perf_model import EinsumGraphPerfModel
    real = P.PyTorchToEinsum._fix_split_connections

    def shortened(self, result, node_id_remap, expanded_input_map=None, expanded_output_slots=None):
        ret = real(self, result, node_id_remap, expanded_input_map, expanded_output_slots)
        result["layers"]["Model.einsum"]["connections"]["inputs"].pop(slot)
        return ret
    monkeypatch.setattr(P.PyTorchToEinsum, "_fix_split_connections", shortened)
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    g = PyTorchToEinsum().convert(str(STATE_FIX / "round4_lstm_capture_pytorch_graph.yaml"), str(einsum_dir), copy_graph=False)
    refused = g["af_export_refused"]
    assert refused["incomplete_reasons"] == {"Model.einsum": "predecessor_slots_unaligned"} and not (einsum_dir / "af_einsum_graph.yaml").exists()
    for f in ("einsum_graph.yaml", "einsum_graph_renamed.yaml"):
        L = yaml.safe_load((einsum_dir / f).read_text())["layers"]["Model.einsum"]
        assert L["status"] == "incomplete" and L["incomplete_reason"] == "predecessor_slots_unaligned"
        assert L["predecessor_slots_unaligned"] == refused["layers"]["Model.einsum"]["predecessor_slots_unaligned"] and len(L["predecessor_slots_unaligned"]) == 1
        assert L["predecessor_slots_unaligned_source"] == "af_builder_positional_correspondence"
    a = EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph.yaml"), str(tmp_path / "analysis"), copy_graph=False)
    assert a["layers"]["Model.einsum"]["status"] == "incomplete" and a["layers"]["Model.einsum"]["incomplete_reason"] == "predecessor_slots_unaligned"
    assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"]["Model.einsum"] == "predecessor_slots_unaligned"
    v = EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert v["workload"]["status"] == "incomplete" and "Model.einsum" in v["workload"]["incomplete_layers"]
