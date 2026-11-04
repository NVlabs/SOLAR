# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for orphaned subgraph pruning in graph_analyzer.

Orphaned subgraphs occur when torchview fails to track tensor creation
(e.g., torch.zeros() with no RecorderTensor args). This creates nodes
with connections to non-existent IDs. Their entire downstream chain
(view → getitem → setitem → dead-end) should contribute zero memory
to the fused cost model.

Real-world example: BigBird's visualization-only attention_probs tensor
created via torch.zeros(), producing 19K+ dead-end nodes that inflate
fused_elements by 527GB.
"""

import pytest
import yaml
from pathlib import Path

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


def _make_einsum_graph(layers: dict) -> dict:
    """Wrap layers dict in the standard einsum graph format."""
    return {"model_name": "test", "layers": layers}


def _analyze_graph(tmp_path: Path, graph: dict, precision: str = "fp32") -> dict:
    """Write graph YAML and run analysis."""
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir(exist_ok=True)
    graph_path = einsum_dir / "einsum_graph_renamed.yaml"
    with open(graph_path, "w") as f:
        yaml.dump(graph, f, sort_keys=False)

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(
        str(graph_path), str(analysis_dir),
        precision=precision, copy_graph=False,
    )
    assert analysis is not None
    return analysis


# Shared layer builders
def _start_layer(output_shape):
    return {
        "type": "start",
        "einsum_equation": "->{}".format(
            "".join(chr(65 + i) for i in range(len(output_shape)))
        ),
        "is_real_einsum": False,
        "tensor_shapes": {"inputs": [], "outputs": [output_shape]},
        "tensor_types": {"inputs": [], "outputs": ["output"]},
        "tensor_names": {"inputs": [], "outputs": ["start.Output"]},
        "connections": {"inputs": [], "outputs": ["Model.linear"]},
    }


def _linear_layer(in_shape, out_shape, weight_shape):
    ndim = len(in_shape)
    in_labels = "".join(chr(65 + i) for i in range(ndim))
    out_labels = in_labels[:-1] + chr(65 + ndim)
    w_labels = in_labels[-1] + chr(65 + ndim)
    return {
        "type": "linear",
        "einsum_equation": "{},{}->{}" .format(in_labels, w_labels, out_labels),
        "is_real_einsum": True,
        "tensor_shapes": {"inputs": [in_shape, weight_shape], "outputs": [out_shape]},
        "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
        "tensor_names": {
            "inputs": ["start.Output", "Model.linear.Weight"],
            "outputs": ["Model.linear.Output"],
        },
        "connections": {"inputs": ["start", "Model.linear.Weight"], "outputs": ["Model.relu"]},
    }


def _relu_layer(shape):
    ndim = len(shape)
    labels = "".join(chr(65 + i) for i in range(ndim))
    return {
        "type": "relu",
        "einsum_equation": "{}->{}".format(labels, labels),
        "is_real_einsum": False,
        "tensor_shapes": {"inputs": [shape], "outputs": [shape]},
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "tensor_names": {
            "inputs": ["Model.linear.Output"],
            "outputs": ["Model.relu.Output"],
        },
        "connections": {"inputs": ["Model.linear"], "outputs": []},
    }


# ──────────────────────────────────────────────────────────────────
# Test: Orphaned view → getitem chain has zero fused contribution
# ──────────────────────────────────────────────────────────────────
class TestOrphanedViewGetitemChain:
    """Orphaned view (connection to non-existent ID) and its downstream
    getitem should produce zero fused_elements."""

    def _build_graph(self):
        """Graph: start → linear → relu (real), plus orphan → view → getitem (dead)."""
        layers = {
            "start": _start_layer([2, 16, 64]),
            "Model.linear": _linear_layer([2, 16, 64], [2, 16, 32], [64, 32]),
            "Model.relu": _relu_layer([2, 16, 32]),
            # Orphaned subgraph: view reads from non-existent hidden-tensor
            "Model.orphan_view": {
                "type": "view",
                "einsum_equation": "ABCD->ABCD",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 4096, 4096]],
                    "outputs": [[1, 12, 4096, 4096]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["ORPHAN_hidden_tensor.Output"],
                    "outputs": ["Model.orphan_view.Output"],
                },
                "connections": {
                    "inputs": ["ORPHAN_hidden_tensor"],
                    "outputs": ["Model.orphan_getitem"],
                },
            },
            "Model.orphan_getitem": {
                "type": "__getitem__",
                "einsum_equation": "ABCD->ABR0D",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 4096, 4096]],
                    "outputs": [[1, 12, 60, 4096]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["Model.orphan_view.Output"],
                    "outputs": ["Model.orphan_getitem.Output"],
                },
                "connections": {
                    "inputs": ["Model.orphan_view"],
                    "outputs": [],
                },
            },
        }
        return _make_einsum_graph(layers)

    @pytest.fixture
    def analysis(self, tmp_path):
        return _analyze_graph(tmp_path, self._build_graph())

    def test_orphan_view_fused_zero(self, analysis):
        """Orphaned view should have zero fused_elements."""
        layer = analysis["layers"]["Model.orphan_view"]
        assert layer["fused_elements"] == 0, (
            f"Orphaned view fused_elements should be 0, got {layer['fused_elements']}"
        )

    def test_orphan_getitem_fused_zero(self, analysis):
        """Dead-end getitem from orphaned source should have zero fused_elements."""
        layer = analysis["layers"]["Model.orphan_getitem"]
        assert layer["fused_elements"] == 0, (
            f"Orphaned getitem fused_elements should be 0, got {layer['fused_elements']}"
        )

    def test_total_fused_excludes_orphans(self, analysis):
        """Total fused_elements should only count the real computation path."""
        real_layers = {
            lid: l for lid, l in analysis["layers"].items()
            if "orphan" not in lid
        }
        real_fused = sum(l["fused_elements"] for l in real_layers.values())
        assert analysis["total"]["fused_elements"] == real_fused, (
            f"Total fused {analysis['total']['fused_elements']} != "
            f"real computation fused {real_fused}"
        )

    def test_orphan_dead_end_flagged(self, analysis):
        """Dead-end orphaned layers should be marked is_orphaned.
        The orphan_view is NOT flagged because it has a consumer (getitem)
        and its memory is already zero (zero-copy view)."""
        getitem = analysis["layers"]["Model.orphan_getitem"]
        assert getitem.get("is_orphaned", False), (
            "Model.orphan_getitem should be flagged as orphaned"
        )


# ──────────────────────────────────────────────────────────────────
# Test: Orphaned setitem has zero fused contribution
# ──────────────────────────────────────────────────────────────────
class TestOrphanedSetitem:
    """__setitem__ writing into an orphaned tensor (dead end) should
    not contribute model_output_elems to fused."""

    def _build_graph(self):
        layers = {
            "start": _start_layer([2, 16, 64]),
            "Model.linear": _linear_layer([2, 16, 64], [2, 16, 32], [64, 32]),
            "Model.relu": _relu_layer([2, 16, 32]),
            # Orphaned view (reads from non-existent ID)
            "Model.orphan_view": {
                "type": "view",
                "einsum_equation": "ABCD->ABCD",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 64]],
                    "outputs": [[1, 12, 64, 64]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["ORPHAN_hidden_tensor.Output"],
                    "outputs": ["Model.orphan_view.Output"],
                },
                "connections": {
                    "inputs": ["ORPHAN_hidden_tensor"],
                    "outputs": ["Model.orphan_setitem"],
                },
            },
            # Setitem writes connected data into orphaned tensor
            "Model.orphan_setitem": {
                "type": "__setitem__",
                "einsum_equation": "ABCD,ABEF->ABCD",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 64], [1, 12, 32, 64]],
                    "outputs": [[1, 12, 64, 64]],
                },
                "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": [
                        "Model.orphan_view.Output",
                        "Model.relu.Output",
                    ],
                    "outputs": ["Model.orphan_setitem.Output"],
                },
                "connections": {
                    "inputs": ["Model.orphan_view", "Model.relu"],
                    "outputs": [],
                },
            },
        }
        return _make_einsum_graph(layers)

    @pytest.fixture
    def analysis(self, tmp_path):
        return _analyze_graph(tmp_path, self._build_graph())

    def test_orphan_setitem_fused_zero(self, analysis):
        """Setitem into orphaned tensor should have zero fused_elements.
        Even though one input (relu.Output) is real, the setitem writes
        into an orphaned target and has no consumers — the write is phantom."""
        layer = analysis["layers"]["Model.orphan_setitem"]
        assert layer["fused_elements"] == 0, (
            f"Orphaned setitem fused_elements should be 0, got {layer['fused_elements']}"
        )

    def test_orphan_setitem_model_output_zero(self, analysis):
        """Dead-end setitem whose target is orphaned should not produce model output."""
        layer = analysis["layers"]["Model.orphan_setitem"]
        assert layer["model_io_elements"] == 0, (
            f"Orphaned setitem model_io should be 0, got {layer['model_io_elements']}"
        )


# ──────────────────────────────────────────────────────────────────
# Test: Deep orphaned chain (orphan → view → view → getitem → dead)
# ──────────────────────────────────────────────────────────────────
class TestDeepOrphanedChain:
    """Multiple layers chained from an orphan source should all be pruned."""

    def _build_graph(self):
        layers = {
            "start": _start_layer([4, 32]),
            "Model.linear": {
                "type": "linear",
                "einsum_equation": "AB,BC->AC",
                "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[4, 32], [32, 16]], "outputs": [[4, 16]]},
                "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["start.Output", "Model.linear.Weight"],
                    "outputs": ["Model.linear.Output"],
                },
                "connections": {"inputs": ["start"], "outputs": []},
            },
            # Deep orphan chain: view1 → view2 → getitem (all dead)
            "Model.orphan_view1": {
                "type": "view",
                "einsum_equation": "ABCD->ABCD",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 64]],
                    "outputs": [[1, 12, 64, 64]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["NONEXISTENT.Output"],
                    "outputs": ["Model.orphan_view1.Output"],
                },
                "connections": {
                    "inputs": ["NONEXISTENT"],
                    "outputs": ["Model.orphan_view2"],
                },
            },
            "Model.orphan_view2": {
                "type": "reshape",
                "einsum_equation": "ABCD->ABCDE",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 64]],
                    "outputs": [[1, 12, 64, 8, 8]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["Model.orphan_view1.Output"],
                    "outputs": ["Model.orphan_view2.Output"],
                },
                "connections": {
                    "inputs": ["Model.orphan_view1"],
                    "outputs": ["Model.orphan_getitem"],
                },
            },
            "Model.orphan_getitem": {
                "type": "__getitem__",
                "einsum_equation": "ABCDE->ABR0DE",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 8, 8]],
                    "outputs": [[1, 12, 60, 8, 8]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["Model.orphan_view2.Output"],
                    "outputs": ["Model.orphan_getitem.Output"],
                },
                "connections": {
                    "inputs": ["Model.orphan_view2"],
                    "outputs": [],
                },
            },
        }
        return _make_einsum_graph(layers)

    @pytest.fixture
    def analysis(self, tmp_path):
        return _analyze_graph(tmp_path, self._build_graph())

    def test_all_orphan_layers_zero_fused(self, analysis):
        """Every layer in the orphaned chain should have zero fused."""
        for lid in ["Model.orphan_view1", "Model.orphan_view2", "Model.orphan_getitem"]:
            layer = analysis["layers"][lid]
            assert layer["fused_elements"] == 0, (
                f"{lid}: orphaned layer fused_elements should be 0, got {layer['fused_elements']}"
            )

    def test_total_fused_only_real(self, analysis):
        """Total fused should only include the real linear layer."""
        orphan_ids = {"Model.orphan_view1", "Model.orphan_view2", "Model.orphan_getitem"}
        real_fused = sum(
            l["fused_elements"]
            for lid, l in analysis["layers"].items()
            if lid not in orphan_ids
        )
        assert analysis["total"]["fused_elements"] == real_fused


# ──────────────────────────────────────────────────────────────────
# Test: Legitimate disconnected layer is NOT pruned
# ──────────────────────────────────────────────────────────────────
class TestLegitimateDisconnectedNotPruned:
    """A layer whose inputs all point to start nodes (filtered out) should
    NOT be treated as orphaned — its inputs are real model inputs."""

    def _build_graph(self):
        layers = {
            "start": _start_layer([4, 64]),
            "Model.linear": {
                "type": "linear",
                "einsum_equation": "AB,BC->AC",
                "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[4, 64], [64, 32]], "outputs": [[4, 32]]},
                "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["start.Output", "Model.linear.Weight"],
                    "outputs": ["Model.linear.Output"],
                },
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        return _make_einsum_graph(layers)

    @pytest.fixture
    def analysis(self, tmp_path):
        return _analyze_graph(tmp_path, self._build_graph())

    def test_linear_not_orphaned(self, analysis):
        """Linear layer reading from start node should NOT be orphaned."""
        layer = analysis["layers"]["Model.linear"]
        assert not layer.get("is_orphaned", False)

    def test_linear_has_nonzero_fused(self, analysis):
        """Linear with real inputs should have non-zero fused_elements."""
        layer = analysis["layers"]["Model.linear"]
        assert layer["fused_elements"] > 0


# ──────────────────────────────────────────────────────────────────
# Test: Orphan root EXISTS in graph but has no inputs (BigBird pattern)
# ──────────────────────────────────────────────────────────────────
class TestOrphanRootInGraph:
    """BigBird's torch.zeros() creates a hidden-tensor node that EXISTS as
    a layer in the einsum graph but has no inputs (no producer).  The
    downstream view → getitem chain should still be detected as orphaned."""

    def _build_graph(self):
        layers = {
            "start": _start_layer([2, 16, 64]),
            "Model.linear": _linear_layer([2, 16, 64], [2, 16, 32], [64, 32]),
            "Model.relu": _relu_layer([2, 16, 32]),
            # Orphan root: exists in graph, has NO inputs (like torch.zeros)
            "Model.hidden-tensor_231": {
                "type": "hidden-tensor",
                "einsum_equation": "ABCD->ABCD",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 4096, 4096]],
                    "outputs": [[1, 12, 4096, 4096]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": [],
                    "outputs": ["Model.hidden-tensor_231.Output"],
                },
                "connections": {
                    "inputs": [],
                    "outputs": ["Model.orphan_view"],
                },
            },
            "Model.orphan_view": {
                "type": "view",
                "einsum_equation": "ABCD->ABCDEF",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 4096, 4096]],
                    "outputs": [[1, 12, 64, 64, 64, 64]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["Model.hidden-tensor_231.Output"],
                    "outputs": ["Model.orphan_view.Output"],
                },
                "connections": {
                    "inputs": ["Model.hidden-tensor_231"],
                    "outputs": ["Model.orphan_getitem"],
                },
            },
            "Model.orphan_getitem": {
                "type": "__getitem__",
                "einsum_equation": "ABCDEF->ABR0DR1F",
                "is_real_einsum": False,
                "tensor_shapes": {
                    "inputs": [[1, 12, 64, 64, 64, 64]],
                    "outputs": [[1, 12, 60, 64, 62, 64]],
                },
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {
                    "inputs": ["Model.orphan_view.Output"],
                    "outputs": ["Model.orphan_getitem.Output"],
                },
                "connections": {
                    "inputs": ["Model.orphan_view"],
                    "outputs": [],
                },
            },
        }
        return _make_einsum_graph(layers)

    @pytest.fixture
    def analysis(self, tmp_path):
        return _analyze_graph(tmp_path, self._build_graph())

    def test_orphan_getitem_fused_zero(self, analysis):
        """Getitem fed by in-graph orphan root should have zero fused."""
        layer = analysis["layers"]["Model.orphan_getitem"]
        assert layer["fused_elements"] == 0, (
            f"fused_elements should be 0, got {layer['fused_elements']}"
        )

    def test_orphan_root_output_is_intermediate(self, analysis):
        """The orphan root's output should be classified as intermediate
        (consumed by graph-internal views/getitems), not external DRAM I/O."""
        layer = analysis["layers"]["Model.hidden-tensor_231"]
        assert layer["output_is_intermediate"], (
            "orphan root output should be intermediate (consumed by graph-internal chain)"
        )

    def test_total_excludes_orphan_chain(self, analysis):
        """Total fused should only count real layers."""
        orphan_ids = {"Model.hidden-tensor_231", "Model.orphan_view", "Model.orphan_getitem"}
        real_fused = sum(
            l["fused_elements"]
            for lid, l in analysis["layers"].items()
            if lid not in orphan_ids
        )
        assert analysis["total"]["fused_elements"] == real_fused

    def test_real_layers_unaffected(self, analysis):
        """Linear layer should still have normal non-zero fused."""
        layer = analysis["layers"]["Model.linear"]
        assert layer["fused_elements"] > 0
