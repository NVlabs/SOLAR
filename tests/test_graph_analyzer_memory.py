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

"""Tests for fused/unfused memory element counting in graph_analyzer.

Verifies that:
- Weight tensors are always counted as external (model_io), never intermediate
- Activation tensors between graph-internal ops are intermediate (fusable)
- __getitem__ ops count memory as input only (output is a view)
- tensor_shapes, tensor_sizes, tensor_types are present in analysis output
- For interior nodes: fused_elements == sum of weight tensor sizes + boundary I/O
"""

import pytest
from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


def _run_full_pipeline(tmp_path: Path, model_source: str, precision: str = "fp32") -> dict:
    """Run full Solar pipeline from source code to analysis."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(model_source))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    config = ProcessingConfig(
        save_graph=False,
        force_rerun=True,
        debug=False,
        safe_mode=False,
    )
    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(graph_dir))
    assert ok, "PyTorchProcessor.process_model_file failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    converter = PyTorchToEinsum()
    einsum_graph = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert einsum_graph is not None, "PyTorchToEinsum.convert failed"

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"),
        str(analysis_dir),
        precision=precision,
        copy_graph=False,
    )
    assert analysis is not None, "EinsumGraphAnalyzer.analyze_graph failed"
    return analysis


def _weight_elems_for_layer(layer: dict) -> int:
    """Sum tensor_sizes[i] where tensor_types.inputs[i] == 'weight'."""
    types = (layer.get("tensor_types") or {}).get("inputs", [])
    sizes = (layer.get("tensor_sizes") or {}).get("inputs", [])
    total = 0
    for i, t in enumerate(types):
        if t == "weight" and i < len(sizes):
            total += sizes[i]
    return total


# ---------------------------------------------------------------------------
# Test: tensor_shapes, tensor_sizes, tensor_types present in output
# ---------------------------------------------------------------------------
class TestAnalysisOutputFields:
    """Verify that tensor_shapes, tensor_sizes, tensor_types are in each layer."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.fc(x)

    def get_inputs():
        return [torch.randn(2, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_tensor_shapes_present(self, analysis):
        for lid, layer in analysis["layers"].items():
            assert "tensor_shapes" in layer, f"{lid} missing tensor_shapes"
            assert "inputs" in layer["tensor_shapes"]
            assert "outputs" in layer["tensor_shapes"]

    def test_tensor_sizes_present(self, analysis):
        for lid, layer in analysis["layers"].items():
            assert "tensor_sizes" in layer, f"{lid} missing tensor_sizes"
            assert "inputs" in layer["tensor_sizes"]
            assert "outputs" in layer["tensor_sizes"]

    def test_tensor_types_present(self, analysis):
        for lid, layer in analysis["layers"].items():
            assert "tensor_types" in layer, f"{lid} missing tensor_types"
            assert "inputs" in layer["tensor_types"]
            assert "outputs" in layer["tensor_types"]

    def test_tensor_sizes_match_shapes(self, analysis):
        """tensor_sizes should be the product of corresponding tensor_shapes."""
        for lid, layer in analysis["layers"].items():
            shapes = layer["tensor_shapes"]
            sizes = layer["tensor_sizes"]
            for key in ("inputs", "outputs"):
                for i, shp in enumerate(shapes[key]):
                    expected = 1
                    for d in shp:
                        expected *= d
                    assert sizes[key][i] == expected, (
                        f"{lid} {key}[{i}]: shape {shp} -> expected {expected}, got {sizes[key][i]}"
                    )


# ---------------------------------------------------------------------------
# Test: Weight tensors never classified as intermediate
# ---------------------------------------------------------------------------
class TestWeightsNotIntermediate:
    """For interior nodes (not start/end), weight tensors must be in model_io."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            h = self.fc1(x)
            h = F.relu(h)
            return self.fc2(h)

    def get_inputs():
        return [torch.randn(2, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_interior_node_fused_includes_weights(self, analysis):
        """For an interior matmul (fc2), fused_elements must include weight size."""
        for lid, layer in analysis["layers"].items():
            if not layer.get("input_is_intermediate"):
                continue
            w_elems = _weight_elems_for_layer(layer)
            if w_elems > 0:
                assert layer["fused_elements"] >= w_elems, (
                    f"{lid}: fused_elements ({layer['fused_elements']}) < "
                    f"weight_elements ({w_elems})"
                )

    def test_weight_not_in_intermediate(self, analysis):
        """Weight elements should not be counted in intermediate_elements."""
        for lid, layer in analysis["layers"].items():
            w_elems = _weight_elems_for_layer(layer)
            if w_elems == 0:
                continue
            if layer.get("input_is_intermediate") and layer.get("output_is_intermediate"):
                assert layer["intermediate_elements"] < layer["unfused_elements"], (
                    f"{lid}: intermediate should be less than unfused (weights excluded)"
                )

    def test_fused_elements_for_interior_matmul(self, analysis):
        """For an interior node with both intermediate input and output,
        fused_elements should equal sum of weight tensor sizes (only weights
        need DRAM; activations are fused)."""
        for lid, layer in analysis["layers"].items():
            if not (layer.get("input_is_intermediate") and layer.get("output_is_intermediate")):
                continue
            w_elems = _weight_elems_for_layer(layer)
            if w_elems > 0:
                assert layer["fused_elements"] == w_elems, (
                    f"{lid}: fused_elements ({layer['fused_elements']}) != "
                    f"weight_elements ({w_elems}) for fully interior node"
                )


# ---------------------------------------------------------------------------
# Test: __getitem__ memory accounting
# ---------------------------------------------------------------------------
class TestGetitemMemory:
    """__getitem__ is a view/slice — input_elems = output_elems, output_elems = 0."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            # x shape: [2, 3, 16, 64], select frame 0 -> [2, 16, 64]
            frame = x[:, 0, :, :]
            return self.fc(frame)

    def get_inputs():
        return [torch.randn(2, 3, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_getitem_output_elems_zero(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "__getitem__":
                assert layer["output_elements"] == 0, (
                    f"{lid}: __getitem__ output_elements should be 0 (view)"
                )

    def test_getitem_input_elems_equals_selected_size(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "__getitem__":
                assert layer["input_elements"] > 0


# ---------------------------------------------------------------------------
# Test: Single-layer graph (all external, no intermediate)
# ---------------------------------------------------------------------------
class TestSingleLayerAllExternal:
    """A single matmul: all inputs are external, fused == unfused."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 128))

        def forward(self, x):
            return torch.matmul(x, self.weight)

    def get_inputs():
        return [torch.randn(4, 32, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_fused_equals_unfused(self, analysis):
        total = analysis["total"]
        assert total["fused_elements"] == total["unfused_elements"]

    def test_zero_intermediate(self, analysis):
        assert analysis["total"]["intermediate_elements"] == 0


# ---------------------------------------------------------------------------
# Test: Multi-layer chain — fused vs unfused accounting
# ---------------------------------------------------------------------------
class TestMultiLayerFusedAccounting:
    """Linear -> relu -> Linear: verify fused/unfused/intermediate totals."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            h = self.fc1(x)
            h = F.relu(h)
            return self.fc2(h)

    def get_inputs():
        return [torch.randn(2, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_unfused_greater_than_fused(self, analysis):
        total = analysis["total"]
        assert total["unfused_elements"] > total["fused_elements"]

    def test_fused_plus_intermediate_equals_unfused(self, analysis):
        total = analysis["total"]
        assert total["fused_elements"] + total["intermediate_elements"] == total["unfused_elements"]

    def test_intermediate_positive(self, analysis):
        assert analysis["total"]["intermediate_elements"] > 0

    def test_total_model_io_equals_fused(self, analysis):
        total = analysis["total"]
        assert total["model_io_elements"] == total["fused_elements"]

    def test_fused_prefetched_equals_fused(self, analysis):
        total = analysis["total"]
        assert total["fused_prefetched_elements"] == total["fused_elements"]


# ---------------------------------------------------------------------------
# Test: Linear with bias — weight+bias always in fused
# ---------------------------------------------------------------------------
class TestLinearBiasFusedElements:
    """nn.Linear(bias=True) split into matmul + bias_add.
    Both weight and bias should be in fused_elements, never intermediate."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128, bias=True)
            self.fc2 = nn.Linear(128, 32, bias=True)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            return self.fc2(h)

    def get_inputs():
        return [torch.randn(1, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_total_weight_in_fused(self, analysis):
        """Sum of all weight tensor sizes across all layers should be <= fused_elements."""
        total_weights = 0
        for lid, layer in analysis["layers"].items():
            total_weights += _weight_elems_for_layer(layer)
        assert total_weights > 0, "No weight tensors found"
        assert analysis["total"]["fused_elements"] >= total_weights

    def test_bias_add_has_weight_type(self, analysis):
        """bias_add layers should have a 'weight' type input for the bias tensor."""
        for lid, layer in analysis["layers"].items():
            if "bias_add" not in lid:
                continue
            types = (layer.get("tensor_types") or {}).get("inputs", [])
            assert "weight" in types, f"{lid}: bias_add should have 'weight' input type"


# ---------------------------------------------------------------------------
# Test: Conv2d — implicit weight and bias not in connections
# ---------------------------------------------------------------------------
class TestConv2dWeightClassification:
    """Conv2d has 1 connection (activation) but 2-3 input shapes (+ weight, bias).
    All implicit weights must be external."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)

        def forward(self, x):
            h = F.relu(self.conv1(x))
            return self.conv2(h)

    def get_inputs():
        return [torch.randn(1, 3, 32, 32)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_conv_weights_in_fused(self, analysis):
        """Conv weight tensors should be in model_io (fused), not intermediate."""
        for lid, layer in analysis["layers"].items():
            if "conv2d" not in layer.get("type", ""):
                continue
            w_elems = _weight_elems_for_layer(layer)
            assert w_elems > 0, f"{lid}: conv2d should have weight tensors"
            assert layer["model_io_elements"] >= w_elems

    def test_interior_conv_fused_includes_weights(self, analysis):
        """For conv2 (interior node), fused should include its weight."""
        for lid, layer in analysis["layers"].items():
            if "conv2d" not in layer.get("type", ""):
                continue
            if not layer.get("input_is_intermediate"):
                continue
            w_elems = _weight_elems_for_layer(layer)
            assert layer["fused_elements"] >= w_elems, (
                f"{lid}: interior conv2d fused ({layer['fused_elements']}) < weight ({w_elems})"
            )


# ---------------------------------------------------------------------------
# Test: Deep chain — per-layer invariant
# ---------------------------------------------------------------------------
class TestPerLayerInvariant:
    """For every layer: unfused = input_elements + output_elements,
    and fused_elements + intermediate_elements == unfused_elements."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 64, bias=False)
            self.fc2 = nn.Linear(64, 64, bias=False)
            self.fc3 = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc3(h)

    def get_inputs():
        return [torch.randn(1, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_unfused_equals_input_plus_output(self, analysis):
        for lid, layer in analysis["layers"].items():
            assert layer["unfused_elements"] == layer["input_elements"] + layer["output_elements"], (
                f"{lid}: unfused != input + output"
            )

    def test_fused_plus_intermediate_equals_unfused(self, analysis):
        for lid, layer in analysis["layers"].items():
            assert layer["fused_elements"] + layer["intermediate_elements"] == layer["unfused_elements"], (
                f"{lid}: fused + intermediate != unfused"
            )

    def test_total_fused_plus_intermediate_equals_unfused(self, analysis):
        total = analysis["total"]
        assert total["fused_elements"] + total["intermediate_elements"] == total["unfused_elements"]


# ---------------------------------------------------------------------------
# Test: Zero-copy view ops (expand, view, reshape, transpose, etc.)
# ---------------------------------------------------------------------------
class TestZeroCopyViewOps:
    """expand, view, reshape, transpose, permute, unsqueeze, squeeze, flatten
    produce zero-copy aliases and should contribute 0 memory elements."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            # x: (2, 8, 64)
            h = self.fc(x)         # (2, 8, 32)
            h = h.transpose(1, 2)  # (2, 32, 8) — zero-copy
            h = h.reshape(2, 256)  # (2, 256) — zero-copy
            h = h.unsqueeze(1)     # (2, 1, 256) — zero-copy
            return h

    def get_inputs():
        return [torch.randn(2, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_view_ops_zero_memory(self, analysis):
        """View ops should have input_elements == 0 and output_elements == 0."""
        zero_copy_types = {
            "transpose", "permute", "t",
            "view", "reshape", "contiguous",
            "expand", "expand_as",
            "unsqueeze", "squeeze", "flatten",
        }
        for lid, layer in analysis["layers"].items():
            if layer["type"] in zero_copy_types:
                assert layer["input_elements"] == 0, (
                    f"{lid} ({layer['type']}): input_elements should be 0, got {layer['input_elements']}"
                )
                assert layer["output_elements"] == 0, (
                    f"{lid} ({layer['type']}): output_elements should be 0, got {layer['output_elements']}"
                )
                assert layer["unfused_elements"] == 0, (
                    f"{lid} ({layer['type']}): unfused_elements should be 0, got {layer['unfused_elements']}"
                )

    def test_view_ops_zero_compute(self, analysis):
        """View ops should have macs == 0 and other_ops == 0."""
        zero_copy_types = {
            "transpose", "permute", "t",
            "view", "reshape", "contiguous",
            "expand", "expand_as",
            "unsqueeze", "squeeze", "flatten",
        }
        for lid, layer in analysis["layers"].items():
            if layer["type"] in zero_copy_types:
                assert layer["macs"] == 0, (
                    f"{lid} ({layer['type']}): macs should be 0, got {layer['macs']}"
                )
                assert layer["other_ops"] == 0, (
                    f"{lid} ({layer['type']}): other_ops should be 0, got {layer['other_ops']}"
                )

    def test_total_memory_excludes_views(self, analysis):
        """Total unfused should not include view ops."""
        non_view_unfused = 0
        for lid, layer in analysis["layers"].items():
            if layer["type"] not in {
                "transpose", "view", "reshape", "unsqueeze",
                "squeeze", "flatten", "permute", "t",
                "expand", "expand_as", "contiguous",
            }:
                non_view_unfused += layer["unfused_elements"]
        assert analysis["total"]["unfused_elements"] == non_view_unfused


# ---------------------------------------------------------------------------
# Test: Expand + contiguous (the hybrid_attention_mask bug)
# ---------------------------------------------------------------------------
class TestExpandContiguousNoDoubleCount:
    """expand() + contiguous() should not double-count memory.
    expand() is zero-copy, contiguous() is zero-copy in analysis."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, mask):
            # mask: (1, 1, 64, 64)
            expanded = mask.expand(4, 8, 64, 64)    # zero-copy view
            materialized = expanded.contiguous()      # zero-copy in analysis
            return materialized

    def get_inputs():
        return [torch.randn(1, 1, 64, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_expand_zero_memory(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "expand":
                assert layer["unfused_elements"] == 0, (
                    f"{lid}: expand should have 0 unfused_elements"
                )

    def test_contiguous_zero_memory(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "contiguous":
                assert layer["unfused_elements"] == 0, (
                    f"{lid}: contiguous should have 0 unfused_elements"
                )

    def test_no_double_count(self, analysis):
        """Total memory should be roughly input + output, not 2x."""
        total = analysis["total"]
        input_size = 1 * 1 * 64 * 64  # mask
        output_size = 4 * 8 * 64 * 64  # expanded
        assert total["unfused_elements"] <= input_size + output_size + 1000, (
            f"Total {total['unfused_elements']} exceeds expected {input_size + output_size}"
        )


# ---------------------------------------------------------------------------
# Test: Slice/select ops (getitem, narrow, select)
# ---------------------------------------------------------------------------
class TestSliceViewOps:
    """__getitem__ / select / narrow return views into the source tensor.
    Memory = output slice size (input), output_elements = 0."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            # x: (2, 3, 8, 64) — e.g. multi-frame input
            frame = x[:, 0, :, :]     # __getitem__: (2, 8, 64) — view
            return self.fc(frame)

    def get_inputs():
        return [torch.randn(2, 3, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_getitem_output_zero(self, analysis):
        """__getitem__ output_elements should be 0 (returns a view)."""
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "__getitem__":
                assert layer["output_elements"] == 0, (
                    f"{lid}: __getitem__ output_elements should be 0"
                )

    def test_getitem_zero_compute(self, analysis):
        """__getitem__ should have macs == 0 and other_ops == 0."""
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "__getitem__":
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"

    def test_getitem_input_equals_slice_size(self, analysis):
        """__getitem__ input_elements should equal the slice size (output shape)."""
        for lid, layer in analysis["layers"].items():
            if layer["type"] == "__getitem__":
                assert layer["input_elements"] > 0, (
                    f"{lid}: __getitem__ input_elements should be > 0"
                )
                shapes = layer.get("tensor_shapes", {})
                out_shapes = shapes.get("outputs", [])
                if out_shapes:
                    expected = 1
                    for d in out_shapes[0]:
                        expected *= d
                    assert layer["input_elements"] == expected, (
                        f"{lid}: input_elements {layer['input_elements']} != "
                        f"slice size {expected}"
                    )


# ---------------------------------------------------------------------------
# Test: View ops in a real chain (linear -> reshape -> linear)
# ---------------------------------------------------------------------------
class TestViewInChainNoMemory:
    """Reshape between two linears should not add memory."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            # x: (1, 8, 64)
            h = self.fc1(x)           # (1, 8, 128)
            h = h.view(1, 1024)       # (1, 1024) — zero-copy
            h = h.view(1, 8, 128)     # (1, 8, 128) — zero-copy
            return self.fc2(h)         # (1, 8, 32)

    def get_inputs():
        return [torch.randn(1, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_view_layers_zero(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("view", "reshape"):
                assert layer["unfused_elements"] == 0, (
                    f"{lid}: {layer['type']} should have 0 unfused"
                )


# ---------------------------------------------------------------------------
# Test: Scatter/__setitem__ ops (KV cache write)
# ---------------------------------------------------------------------------
class TestScatterSetitemMemory:
    """__setitem__ (tensor[:, :, idx, :] = values) should count the slice
    size, not the full target tensor."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 128, bias=False)

        def forward(self, hidden, cache, position):
            # hidden: (1, 8, 128), cache: (1, 8, 256, 128), position: (1,)
            projected = self.fc(hidden)              # (1, 8, 128)
            cache[:, :, 0:8, :] = projected          # scatter write
            return cache

    def get_inputs():
        cache = torch.randn(1, 8, 256, 128)
        hidden = torch.randn(1, 8, 128)
        position = torch.tensor([0])
        return [hidden, cache, position]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_setitem_not_full_tensor(self, analysis):
        """__setitem__ should not count the full cache tensor."""
        full_cache_elems = 1 * 8 * 256 * 128  # 262144
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("__setitem__", "index_put", "index_put_",
                                  "scatter", "scatter_", "index_copy", "index_copy_"):
                assert layer["unfused_elements"] < full_cache_elems, (
                    f"{lid}: scatter op unfused ({layer['unfused_elements']}) "
                    f"should be < full cache ({full_cache_elems})"
                )

    def test_setitem_zero_compute(self, analysis):
        """__setitem__ / scatter ops should have macs == 0 and other_ops == 0."""
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("__setitem__", "index_put", "index_put_",
                                  "scatter", "scatter_", "index_copy", "index_copy_"):
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"

    def test_setitem_input_zero(self, analysis):
        """__setitem__ input_elements should be 0 (values accounted for by upstream producer)."""
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("__setitem__", "index_put", "index_put_",
                                  "scatter", "scatter_", "index_copy", "index_copy_"):
                assert layer["input_elements"] == 0, (
                    f"{lid}: input_elements ({layer['input_elements']}) should be 0"
                )

    def test_setitem_output_less_than_full_cache(self, analysis):
        """__setitem__ output_elements should be the source/values size, not the full target."""
        full_cache = 1 * 8 * 256 * 128  # 262144
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("__setitem__", "index_put", "index_put_",
                                  "scatter", "scatter_", "index_copy", "index_copy_"):
                assert layer["output_elements"] < full_cache, (
                    f"{lid}: output_elements ({layer['output_elements']}) "
                    f"should be < full cache ({full_cache})"
                )


# ---------------------------------------------------------------------------
# Test: index_copy_ (another scatter variant)
# ---------------------------------------------------------------------------
class TestIndexCopyMemory:
    """index_copy_ writes into a target at specific indices.
    Memory should be the values size, not the full target."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, target, indices, source):
            # target: (64, 128), indices: (4,), source: (4, 128)
            target.index_copy_(0, indices, source)
            return target

    def get_inputs():
        target = torch.randn(64, 128)
        indices = torch.tensor([0, 10, 20, 30])
        source = torch.randn(4, 128)
        return [target, indices, source]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_not_full_target(self, analysis):
        full_target = 64 * 128  # 8192
        source_size = 4 * 128   # 512
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("index_copy", "index_copy_",
                                  "__setitem__", "scatter", "scatter_"):
                assert layer["unfused_elements"] <= source_size, (
                    f"{lid}: unfused ({layer['unfused_elements']}) should be <= source size ({source_size})"
                )

    def test_input_zero(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("index_copy", "index_copy_",
                                  "__setitem__", "scatter", "scatter_"):
                assert layer["input_elements"] == 0, (
                    f"{lid}: input_elements should be 0"
                )

    def test_fused_leq_unfused(self, analysis):
        """fused_elements must never exceed unfused_elements."""
        for lid, layer in analysis["layers"].items():
            assert layer["fused_elements"] <= layer["unfused_elements"], (
                f"{lid}: fused ({layer['fused_elements']}) > unfused ({layer['unfused_elements']})"
            )
        total = analysis["total"]
        assert total["fused_elements"] <= total["unfused_elements"], (
            f"total: fused ({total['fused_elements']}) > unfused ({total['unfused_elements']})"
        )


# ---------------------------------------------------------------------------
# Test: Scatter fused <= unfused invariant across multiple patterns
# ---------------------------------------------------------------------------
class TestScatterFusedLeqUnfused:
    """Verify fused <= unfused for various scatter/index-write patterns.
    This was broken when the scatter memory override set input_elems=0
    but the classification loop still used original shapes."""

    MODEL_SOURCE_INDEX_COPY = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, target, indices, source):
            target.index_copy_(0, indices, source)
            return target

    def get_inputs():
        return [torch.randn(5, 3), torch.tensor([0, 4, 2]), torch.randn(3, 3)]

    def get_init_inputs():
        return []
    """

    MODEL_SOURCE_SETITEM_CHAIN = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 32, bias=False)

        def forward(self, cache, values):
            projected = self.fc(values)
            cache[:, 0:4, :] = projected
            return cache

    def get_inputs():
        return [torch.randn(1, 64, 32), torch.randn(1, 4, 32)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis_index_copy(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE_INDEX_COPY)

    @pytest.fixture
    def analysis_setitem_chain(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE_SETITEM_CHAIN)

    def test_index_copy_fused_leq_unfused_per_layer(self, analysis_index_copy):
        for lid, layer in analysis_index_copy["layers"].items():
            assert layer["fused_elements"] <= layer["unfused_elements"], (
                f"{lid}: fused ({layer['fused_elements']}) > unfused ({layer['unfused_elements']})"
            )

    def test_index_copy_fused_leq_unfused_total(self, analysis_index_copy):
        total = analysis_index_copy["total"]
        assert total["fused_elements"] <= total["unfused_elements"]

    def test_index_copy_output_equals_source(self, analysis_index_copy):
        """index_copy_ output should be source size (9), not target (15)."""
        for lid, layer in analysis_index_copy["layers"].items():
            if "index_copy" in layer["type"]:
                assert layer["output_elements"] == 9, (
                    f"{lid}: output should be 9 (source), got {layer['output_elements']}"
                )
                assert layer["unfused_elements"] == 9

    def test_setitem_chain_fused_leq_unfused_per_layer(self, analysis_setitem_chain):
        for lid, layer in analysis_setitem_chain["layers"].items():
            assert layer["fused_elements"] <= layer["unfused_elements"], (
                f"{lid}: fused ({layer['fused_elements']}) > unfused ({layer['unfused_elements']})"
            )

    def test_setitem_chain_fused_leq_unfused_total(self, analysis_setitem_chain):
        total = analysis_setitem_chain["total"]
        assert total["fused_elements"] <= total["unfused_elements"]
