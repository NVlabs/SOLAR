# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for einsum cost computation bugs.

Issue 5: Raw torch.einsum ops use identity fallback instead of actual equation.
Issue 1: MultiHeadAttention reports 0 MACs (missing subgraph expansion).
"""

import tempfile
from pathlib import Path

import pytest
import yaml
import networkx as nx

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


# ============================================================
# Issue 5: Raw torch.einsum should use actual equation for cost
# ============================================================

class TestRawEinsumCost:
    """Raw torch.einsum nodes have their equation in einsum_equation field.

    Currently, get_compute_cost("einsum", ts) ignores the equation and falls
    back to identity from the first input shape. This produces wrong MACs.
    """

    def test_einsum_cost_uses_provided_equation(self):
        """get_compute_cost should use einsum_equation kwarg for 'einsum' ops."""
        analyzer = EinsumAnalyzer()
        ts = TensorShapes(
            inputs=[[8, 256, 512, 256], [256, 768]],
            outputs=[[8, 256, 512, 768]],
        )
        cost = analyzer.get_compute_cost(
            "einsum", ts, equation="BIJL,LK->BIJK"
        )
        expected = 8 * 256 * 512 * 256 * 768
        assert cost == expected, (
            f"Expected cost {expected} from equation BIJL,LK->BIJK, got {cost}. "
            f"The einsum equation should drive the cost, not an identity fallback."
        )

    def test_einsum_cost_mamba2_style(self):
        """Mamba2-style multi-input einsum should compute correct cost."""
        analyzer = EinsumAnalyzer()
        ts = TensorShapes(
            inputs=[[2048, 8, 2, 64, 64], [2048, 2, 64, 8, 64]],
            outputs=[[2048, 8, 64, 2, 64]],
        )
        cost = analyzer.get_compute_cost(
            "einsum", ts, equation="BCLHN,BCSHN->BHCLS"
        )
        expected = 2048 * 8 * 2 * 64 * 64 * 64
        assert cost == expected, (
            f"Expected cost {expected}, got {cost}"
        )

    def test_graph_analyzer_passes_equation_for_einsum_ops(self):
        """EinsumGraphAnalyzer should pass einsum_equation to cost computation."""
        graph = {
            "layers": {
                "start_0": {
                    "type": "start",
                    "tensor_shapes": {"inputs": [], "outputs": [[8, 256, 512, 256]]},
                    "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                    "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "connections": {"inputs": [], "outputs": ["einsum_0"]},
                },
                "start_1": {
                    "type": "start",
                    "tensor_shapes": {"inputs": [], "outputs": [[256, 768]]},
                    "tensor_names": {"inputs": [], "outputs": ["start_1.Output"]},
                    "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "connections": {"inputs": [], "outputs": ["einsum_0"]},
                },
                "einsum_0": {
                    "type": "einsum",
                    "einsum_equation": "BIJL,LK->BIJK",
                    "is_real_einsum": True,
                    "tensor_shapes": {
                        "inputs": [[8, 256, 512, 256], [256, 768]],
                        "outputs": [[8, 256, 512, 768]],
                    },
                    "tensor_names": {
                        "inputs": ["start_0.Output", "start_1.Output"],
                        "outputs": ["einsum_0.Output"],
                    },
                    "tensor_types": {
                        "inputs": ["input", "input"],
                        "outputs": ["output"],
                    },
                    "connections": {
                        "inputs": ["start_0", "start_1"],
                        "outputs": [],
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "einsum_graph.yaml"
            with open(graph_path, "w") as f:
                yaml.dump(graph, f)

            analyzer = EinsumGraphAnalyzer()
            result = analyzer.analyze_graph(
                graph_path, tmpdir, precision="fp16", copy_graph=False
            )

        assert result is not None
        expected_macs = 8 * 256 * 512 * 256 * 768
        actual_macs = result["total"]["macs"]
        assert actual_macs == expected_macs, (
            f"Graph analyzer should compute MACs={expected_macs} for einsum "
            f"BIJL,LK->BIJK, got {actual_macs}"
        )


# ============================================================
# Issue 1: MHA should expand into sub-nodes with real MACs
# ============================================================

class TestMHAExpansion:
    """nn.MultiheadAttention nodes should produce nonzero MACs.

    MHA should be expanded like SDPA into matmul sub-nodes during
    einsum graph generation.
    """

    S, B, D = 197, 2, 128

    def _build_op_graph_and_args(self):
        """Build a minimal op graph and expansion arguments for MHA."""
        S, B, D = self.S, self.B, self.D
        op_graph = nx.DiGraph()
        op_graph.add_node("start_0", type="start",
                         input_shapes=[], output_shapes=[[S, B, D]])
        op_graph.add_node("mha", type="multi_head_attention_forward",
                         input_shapes=[
                             [S, B, D], [S, B, D], [S, B, D],
                             [3 * D, D], [3 * D], [D, D], [D],
                         ],
                         output_shapes=[[S, B, D], [B, S, S]],
                         input_types=["input", "input", "input",
                                     "weight", "weight", "weight", "weight"])
        op_graph.add_edge("start_0", "mha")

        start_nodes_info = [{
            "original_id": "start_0",
            "index": 0,
            "output_shapes": [[S, B, D]],
            "output_dtypes": [],
            "consumers": ["mha"],
        }]
        start_node_id_map = {"start_0": "start_0"}

        return op_graph, start_nodes_info, start_node_id_map

    def test_mha_expansion_produces_matmul_subnodes(self):
        """_expand_mha should produce sub-nodes with is_real_einsum=True matmuls."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["mha"])

        subgraph, final_node_id, input_mapping = converter._expand_mha(
            "mha", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        # Should have in_proj, qk_matmul, scale, softmax, av_matmul, out_proj
        assert len(subgraph) >= 5, (
            f"MHA expansion should produce at least 5 sub-nodes, got {len(subgraph)}: "
            f"{list(subgraph.keys())}"
        )

        real_einsum_nodes = {
            nid: n for nid, n in subgraph.items()
            if n.get("is_real_einsum") is True
        }
        assert len(real_einsum_nodes) >= 3, (
            f"Should have at least 3 real einsum nodes "
            f"(in_proj, qk_matmul, av_matmul), got {len(real_einsum_nodes)}: "
            f"{list(real_einsum_nodes.keys())}"
        )

    def test_mha_expanded_analysis_produces_correct_macs(self):
        """Expanded MHA sub-nodes should produce correct MACs in graph analysis."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        S, B, D = self.S, self.B, self.D
        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["mha"])

        subgraph, final_node_id, _ = converter._expand_mha(
            "mha", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        # Build a complete einsum graph YAML from the expanded sub-nodes
        layers = {
            "start_0": {
                "type": "start",
                "tensor_shapes": {"inputs": [], "outputs": [[S, B, D]]},
                "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                "tensor_types": {"inputs": [], "outputs": ["input"]},
                "connections": {"inputs": [], "outputs": [list(subgraph.keys())[0]]},
            },
        }
        layers.update(subgraph)

        graph = {"layers": layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "einsum_graph.yaml"
            with open(graph_path, "w") as f:
                yaml.dump(graph, f)

            ga = EinsumGraphAnalyzer()
            result = ga.analyze_graph(
                graph_path, tmpdir, precision="fp16", copy_graph=False
            )

        assert result is not None
        total_macs = result["total"]["macs"]

        # Expected MACs (head count cancels):
        # in_proj:  S*B * D * 3D = 394 * 128 * 384 = 19,365,888
        # QK:       B*H*S*D*S = 2*1*197*128*197 = 9,935,104
        # AV:       B*H*S*S*D = 2*1*197*197*128 = 9,935,104
        # out_proj: S*B * D * D = 394 * 128 * 128 = 6,455,296
        expected = 19_365_888 + 9_935_104 + 9_935_104 + 6_455_296  # 45,691,392
        min_expected = int(expected * 0.8)
        assert total_macs >= min_expected, (
            f"Expanded MHA should produce ~{expected} MACs, got {total_macs}"
        )


# ============================================================
# LSTM should expand into sub-nodes with real MACs
# ============================================================

class TestLSTMExpansion:
    """nn.LSTM nodes should produce nonzero MACs.

    LSTM should be expanded into ih_linear and hh_linear sub-nodes
    (per layer) during einsum graph generation, similar to MHA expansion.
    """

    S, B, I, H = 32, 4, 64, 128  # seq_len, batch, input_size, hidden_size

    def _build_op_graph_and_args(self):
        """Build a minimal op graph for a single-layer LSTM."""
        S, B, I, H = self.S, self.B, self.I, self.H
        op_graph = nx.DiGraph()
        op_graph.add_node("start_0", type="start",
                         input_shapes=[], output_shapes=[[S, B, I]])
        op_graph.add_node("lstm_0", type="lstm",
                         input_shapes=[
                             [S, B, I],        # input
                             [1, B, H],         # h_0
                             [1, B, H],         # c_0
                             [4 * H, I],        # weight_ih
                             [4 * H, H],        # weight_hh
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
        start_node_id_map = {"start_0": "start_0"}

        return op_graph, start_nodes_info, start_node_id_map

    def test_lstm_expansion_produces_linear_subnodes(self):
        """_expand_lstm should produce sub-nodes with is_real_einsum=True linears."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["lstm_0"])

        subgraph, final_node_id, input_mapping = converter._expand_lstm(
            "lstm_0", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        # Should have ih_linear, hh_linear, and gate ops
        assert len(subgraph) >= 2, (
            f"LSTM expansion should produce at least 2 sub-nodes, got {len(subgraph)}: "
            f"{list(subgraph.keys())}"
        )

        real_einsum_nodes = {
            nid: n for nid, n in subgraph.items()
            if n.get("is_real_einsum") is True
        }
        assert len(real_einsum_nodes) >= 2, (
            f"Should have at least 2 real einsum nodes "
            f"(ih_linear, hh_linear), got {len(real_einsum_nodes)}: "
            f"{list(real_einsum_nodes.keys())}"
        )

    def test_lstm_expanded_analysis_produces_correct_macs(self):
        """Expanded LSTM sub-nodes should produce correct MACs in graph analysis."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        S, B, I, H = self.S, self.B, self.I, self.H
        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["lstm_0"])

        subgraph, final_node_id, _ = converter._expand_lstm(
            "lstm_0", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        layers = {
            "start_0": {
                "type": "start",
                "tensor_shapes": {"inputs": [], "outputs": [[S, B, I]]},
                "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                "tensor_types": {"inputs": [], "outputs": ["input"]},
                "connections": {"inputs": [], "outputs": [list(subgraph.keys())[0]]},
            },
        }
        layers.update(subgraph)

        graph = {"layers": layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "einsum_graph.yaml"
            with open(graph_path, "w") as f:
                yaml.dump(graph, f)

            ga = EinsumGraphAnalyzer()
            result = ga.analyze_graph(
                graph_path, tmpdir, precision="fp16", copy_graph=False
            )

        assert result is not None
        total_macs = result["total"]["macs"]

        # Expected MACs per timestep (summed over S steps):
        # ih_linear: S * B * I * 4H = 32 * 4 * 64 * 512 = 4,194,304
        # hh_linear: S * B * H * 4H = 32 * 4 * 128 * 512 = 8,388,608
        expected = S * B * I * 4 * H + S * B * H * 4 * H  # 12,582,912
        min_expected = int(expected * 0.8)
        assert total_macs >= min_expected, (
            f"Expanded LSTM should produce ~{expected} MACs, got {total_macs}"
        )


# ============================================================
# GRU should expand into sub-nodes with real MACs
# ============================================================

class TestGRUExpansion:
    """nn.GRU nodes should produce nonzero MACs.

    GRU should be expanded into ih_linear and hh_linear sub-nodes
    during einsum graph generation. GRU has 3 gates (vs LSTM's 4).
    """

    S, B, I, H = 32, 4, 64, 128

    def _build_op_graph_and_args(self):
        """Build a minimal op graph for a single-layer GRU."""
        S, B, I, H = self.S, self.B, self.I, self.H
        op_graph = nx.DiGraph()
        op_graph.add_node("start_0", type="start",
                         input_shapes=[], output_shapes=[[S, B, I]])
        op_graph.add_node("gru_0", type="gru",
                         input_shapes=[
                             [S, B, I],        # input
                             [1, B, H],         # h_0
                             [3 * H, I],        # weight_ih
                             [3 * H, H],        # weight_hh
                         ],
                         output_shapes=[[S, B, H], [1, B, H]],
                         input_types=["input", "input",
                                     "weight", "weight"])
        op_graph.add_edge("start_0", "gru_0")

        start_nodes_info = [{
            "original_id": "start_0",
            "index": 0,
            "output_shapes": [[S, B, I]],
            "output_dtypes": [],
            "consumers": ["gru_0"],
        }]
        start_node_id_map = {"start_0": "start_0"}

        return op_graph, start_nodes_info, start_node_id_map

    def test_gru_expansion_produces_linear_subnodes(self):
        """_expand_gru should produce sub-nodes with is_real_einsum=True linears."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["gru_0"])

        subgraph, final_node_id, input_mapping = converter._expand_gru(
            "gru_0", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        assert len(subgraph) >= 2, (
            f"GRU expansion should produce at least 2 sub-nodes, got {len(subgraph)}: "
            f"{list(subgraph.keys())}"
        )

        real_einsum_nodes = {
            nid: n for nid, n in subgraph.items()
            if n.get("is_real_einsum") is True
        }
        assert len(real_einsum_nodes) >= 2, (
            f"Should have at least 2 real einsum nodes "
            f"(ih_linear, hh_linear), got {len(real_einsum_nodes)}: "
            f"{list(real_einsum_nodes.keys())}"
        )

    def test_gru_expanded_analysis_produces_correct_macs(self):
        """Expanded GRU sub-nodes should produce correct MACs in graph analysis."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        S, B, I, H = self.S, self.B, self.I, self.H
        converter = PyTorchToEinsum()
        op_graph, start_nodes_info, start_node_id_map = self._build_op_graph_and_args()
        node_data = dict(op_graph.nodes["gru_0"])

        subgraph, final_node_id, _ = converter._expand_gru(
            "gru_0", node_data, op_graph, start_nodes_info, start_node_id_map
        )

        layers = {
            "start_0": {
                "type": "start",
                "tensor_shapes": {"inputs": [], "outputs": [[S, B, I]]},
                "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                "tensor_types": {"inputs": [], "outputs": ["input"]},
                "connections": {"inputs": [], "outputs": [list(subgraph.keys())[0]]},
            },
        }
        layers.update(subgraph)

        graph = {"layers": layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "einsum_graph.yaml"
            with open(graph_path, "w") as f:
                yaml.dump(graph, f)

            ga = EinsumGraphAnalyzer()
            result = ga.analyze_graph(
                graph_path, tmpdir, precision="fp16", copy_graph=False
            )

        assert result is not None
        total_macs = result["total"]["macs"]

        # Expected MACs (3 gates for GRU):
        # ih_linear: S * B * I * 3H = 32 * 4 * 64 * 384 = 3,145,728
        # hh_linear: S * B * H * 3H = 32 * 4 * 128 * 384 = 6,291,456
        expected = S * B * I * 3 * H + S * B * H * 3 * H  # 9,437,184
        min_expected = int(expected * 0.8)
        assert total_macs >= min_expected, (
            f"Expanded GRU should produce ~{expected} MACs, got {total_macs}"
        )


# ============================================================
# Multi-dim reduction: sum/mean with dim=(2,3) must drop all dims
# ============================================================

class TestMultiDimReduction:
    """Regression tests for multi-dimensional reduction ops.

    torch.sum(x, dim=(2, 3)) must produce ABCD->AB, not ABCD->ABD.
    torch.mean(x, dim=[2, 3], keepdim=True) must produce ABCD->ABCD.
    """

    def test_parse_multi_dim_reduction(self):
        """_parse_reduction_args_from_raw_attributes returns all dims."""
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

        converter = PyTorchToEinsum()

        # dim: [2, 3] — must return both dims
        dims, keepdim = converter._parse_reduction_args_from_raw_attributes(
            {"raw_attributes": "[[Tensor(shape=(16, 1, 1, 1))], {dim: [2, 3]}]"}
        )
        assert dims == [2, 3], f"Expected [2, 3], got {dims}"
        assert keepdim is False

        # dim: [2, 3], keepdim: True
        dims, keepdim = converter._parse_reduction_args_from_raw_attributes(
            {"raw_attributes": "[[Tensor(shape=(16, 128, 514, 514))], {dim: [2, 3], keepdim: True}]"}
        )
        assert dims == [2, 3], f"Expected [2, 3], got {dims}"
        assert keepdim is True

        # Single dim: dim: 1
        dims, keepdim = converter._parse_reduction_args_from_raw_attributes(
            {"raw_attributes": "[[Tensor(shape=(16, 128, 1, 1))], {dim: 1, keepdim: True}]"}
        )
        assert dims == [1], f"Expected [1], got {dims}"
        assert keepdim is True

    def test_sum_multi_dim_equation(self):
        """sum(dim=(2,3)) on 4D tensor produces ABCD->AB."""
        analyzer = EinsumAnalyzer()
        ts = TensorShapes(
            inputs=[[16, 1, 1, 1]],
            outputs=[[16, 1]],
        )
        op = analyzer.get_einsum_op("sum", ts, dims=[2, 3], keepdim=False)
        assert op.equation == "ABCD->AB", (
            f"sum(dim=[2,3]) should give ABCD->AB, got {op.equation}"
        )

    def test_mean_multi_dim_keepdim_equation(self):
        """mean(dim=[2,3], keepdim=True) on 4D tensor produces ABCD->ABCD."""
        analyzer = EinsumAnalyzer()
        ts = TensorShapes(
            inputs=[[16, 128, 514, 514]],
            outputs=[[16, 128, 1, 1]],
        )
        op = analyzer.get_einsum_op("mean", ts, dims=[2, 3], keepdim=True)
        assert op.equation == "ABCD->ABCD", (
            f"mean(dim=[2,3], keepdim=True) should give ABCD->ABCD, got {op.equation}"
        )

    def test_sum_multi_dim_graph_analysis(self):
        """End-to-end: sum(dim=[2,3]) in graph produces correct output shape."""
        graph = {
            "layers": {
                "start_0": {
                    "type": "start",
                    "tensor_shapes": {"inputs": [], "outputs": [[16, 1, 1, 1]]},
                    "tensor_names": {"inputs": [], "outputs": ["start_0.Output"]},
                    "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "connections": {"inputs": [], "outputs": ["sum_0"]},
                },
                "sum_0": {
                    "type": "sum",
                    "einsum_equation": "ABCD->AB",
                    "elementwise_op": "copy",
                    "reduction_op": "add",
                    "is_real_einsum": False,
                    "is_einsum_supportable": True,
                    "tensor_names": {
                        "inputs": ["start_0.Output"],
                        "outputs": ["sum_0.Output"],
                    },
                    "tensor_types": {
                        "inputs": ["input"],
                        "outputs": ["output"],
                    },
                    "tensor_shapes": {
                        "inputs": [[16, 1, 1, 1]],
                        "outputs": [[16, 1]],
                    },
                    "connections": {
                        "inputs": ["start_0"],
                        "outputs": [],
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "einsum_graph.yaml"
            with open(graph_path, "w") as f:
                yaml.dump(graph, f)

            ga = EinsumGraphAnalyzer()
            result = ga.analyze_graph(
                graph_path, tmpdir, precision="fp16", copy_graph=False
            )

        assert result is not None
        # Output rank should be 2 (AB), not 3 (ABD)
        sum_node = result["layers"]["sum_0"]
        out_shape = sum_node["tensor_shapes"]["outputs"][0]
        assert len(out_shape) == 2, (
            f"sum(dim=[2,3]) output should be 2D, got {len(out_shape)}D: {out_shape}"
        )
