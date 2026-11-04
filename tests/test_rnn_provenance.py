# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic recurrent-placeholder provenance vs the compute sparsity discount.

The RNN lowering (`_expand_rnn`) approximates recurrence wiring with an
``h0_expand`` placeholder that broadcasts the initial hidden state across all
timesteps.  The repeated values along the sequence dim are a lowering
artifact — the real workload uses a distinct hidden state per step — so the
replication-based MAC discount must not apply to them.  The as-written SOL
contract (docs/SOL_GUIDE.md) charges all reference work.

The exclusion is keyed on explicit lowering provenance
(``additional_info.synthetic_recurrence_placeholder``), never on layer names.
Genuine replication (GQA K/V expand, Mamba-2 B/C group expand) keeps its
discount — the control below pins that.
"""

import tempfile
from pathlib import Path

import networkx as nx
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


S, B, I, H = 32, 4, 64, 128


def _lstm_subgraph():
    from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

    op_graph = nx.DiGraph()
    op_graph.add_node("start_0", type="start",
                      input_shapes=[], output_shapes=[[S, B, I]])
    op_graph.add_node("lstm_0", type="lstm",
                      input_shapes=[
                          [S, B, I],
                          [1, B, H],
                          [1, B, H],
                          [4 * H, I],
                          [4 * H, H],
                      ],
                      output_shapes=[[S, B, H], [1, B, H], [1, B, H]],
                      input_types=["input", "input", "input",
                                   "weight", "weight"])
    op_graph.add_edge("start_0", "lstm_0")
    start_nodes_info = [{
        "original_id": "start_0",
        "index": 0,
        "output_shapes": [[S, B, I]],
        "output_dtypes": [],
        "consumers": ["lstm_0"],
    }]
    node_data = dict(op_graph.nodes["lstm_0"])
    converter = PyTorchToEinsum()
    subgraph, _, _ = converter._expand_lstm(
        "lstm_0", node_data, op_graph, start_nodes_info, {"start_0": "start_0"}
    )
    return subgraph


def _analyze(layers):
    graph = {"layers": layers}
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = Path(tmpdir) / "einsum_graph.yaml"
        graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
        return EinsumGraphAnalyzer().analyze_graph(
            graph_path, tmpdir, precision="fp16", copy_graph=False
        )


def _expand_chain(flagged: bool):
    """expand [B,H]->[S,B,H] feeding a linear SBH,GH->SBG (hand-built)."""
    G = 4 * H
    expand_layer = {
        "type": "expand",
        "einsum_equation": "BH->SBH",
        "elementwise_op": "mul",
        "reduction_op": "none",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["h_state"], "outputs": ["h_rep"]},
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "tensor_shapes": {"inputs": [[B, H]], "outputs": [[S, B, H]]},
        "tensor_dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "connections": {"inputs": [], "outputs": ["lin"]},
    }
    if flagged:
        expand_layer["additional_info"] = {
            "synthetic_recurrence_placeholder": True}
    linear_layer = {
        "type": "linear",
        "einsum_equation": "SBH,GH->SBG",
        "elementwise_op": "mul",
        "reduction_op": "add",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["h_rep", "w_hh"], "outputs": ["lin_out"]},
        "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
        "tensor_shapes": {"inputs": [[S, B, H], [G, H]], "outputs": [[S, B, G]]},
        "tensor_dtypes": {"inputs": ["float32", "float32"],
                          "outputs": ["float32"]},
        "connections": {"inputs": ["exp"], "outputs": []},
    }
    return {"exp": expand_layer, "lin": linear_layer}


DENSE_LIN_MACS = S * B * H * 4 * H  # 8,388,608


class TestSyntheticRecurrenceProvenance:
    def test_h0_expand_carries_lowering_provenance(self):
        """The RNN lowering must mark its synthesized placeholder explicitly."""
        subgraph = _lstm_subgraph()
        h0_expands = [l for lid, l in subgraph.items()
                      if l.get("type") == "expand"]
        assert h0_expands, "LSTM expansion emits an expand placeholder"
        for layer in h0_expands:
            info = layer.get("additional_info") or {}
            assert info.get("synthetic_recurrence_placeholder") is True, (
                "the synthesized recurrent placeholder must carry explicit "
                "lowering provenance; the discount exclusion is keyed on it"
            )

    def test_flagged_placeholder_reuse_is_not_discounted(self):
        """A provenance-flagged expand creates no replication MAC discount."""
        result = _analyze(_expand_chain(flagged=True))
        lin = result["layers"]["lin"]
        assert lin["macs_dense"] == DENSE_LIN_MACS
        assert lin["macs"] == DENSE_LIN_MACS, (
            f"synthetic-placeholder reuse must bill dense work "
            f"({DENSE_LIN_MACS}), got {lin['macs']}"
        )
        assert lin["mac_sparsity_fraction"] == 1.0

    def test_genuine_expand_replication_still_discounted(self):
        """Control: an ordinary expand keeps the replication discount (S x)."""
        result = _analyze(_expand_chain(flagged=False))
        lin = result["layers"]["lin"]
        assert lin["macs_dense"] == DENSE_LIN_MACS
        assert lin["macs"] == DENSE_LIN_MACS // S, (
            f"genuine replicated reuse keeps its discount "
            f"({DENSE_LIN_MACS // S}), got {lin['macs']}"
        )

    def test_lstm_expanded_analysis_bills_exact_dense_recurrent_work(self):
        """Exact as-written counts: ih + hh, no placeholder discount."""
        subgraph = _lstm_subgraph()
        layers = {
            "start_0": {
                "type": "start",
                "tensor_shapes": {"inputs": [], "outputs": [[S, B, I]]},
                "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                "tensor_types": {"inputs": [], "outputs": ["input"]},
                "connections": {"inputs": [],
                                "outputs": [list(subgraph.keys())[0]]},
            },
        }
        layers.update(subgraph)
        result = _analyze(layers)
        ih = S * B * I * 4 * H
        hh = S * B * H * 4 * H
        assert result["total"]["macs"] == ih + hh, (
            f"as-written LSTM work is ih {ih} + hh {hh} = {ih + hh}, "
            f"got {result['total']['macs']}"
        )
        hh_layers = {lid: l for lid, l in result["layers"].items()
                     if lid.endswith("hh_linear")}
        assert hh_layers
        for lid, l in hh_layers.items():
            assert l["macs"] == l["macs_dense"] == hh, (lid, l["macs"])
            assert l["mac_sparsity_fraction"] == 1.0
