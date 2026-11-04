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

"""Tests for hidden-tensor node removal during einsum graph conversion.

Verifies that:
- No hidden-tensor IDs appear in einsum connections or tensor_names
- Connections are properly rewired through the producer op
- tensor_names reference the producer op's output, not the hidden-tensor
- Multi-consumer hidden-tensors (fan-out) are handled correctly
"""

import pytest
import yaml
from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def _run_to_einsum(tmp_path: Path, model_source: str) -> dict:
    """Run pipeline from source to einsum graph, return the graph dict."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(model_source))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    config = ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)
    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(graph_dir))
    assert ok, "PyTorchProcessor failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    converter = PyTorchToEinsum()
    einsum_graph = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert einsum_graph is not None, "PyTorchToEinsum failed"

    renamed_path = einsum_dir / "einsum_graph_renamed.yaml"
    with open(renamed_path) as f:
        return yaml.safe_load(f)


def _assert_no_hidden_tensor_refs(layers: dict):
    """Assert that no layer references a hidden-tensor node."""
    for lid, layer in layers.items():
        conn = layer.get("connections", {})
        for inp in conn.get("inputs", []):
            assert "hidden-tensor" not in inp, (
                f"{lid}: connections.inputs contains hidden-tensor ref '{inp}'"
            )
        for out in conn.get("outputs", []):
            assert "hidden-tensor" not in out, (
                f"{lid}: connections.outputs contains hidden-tensor ref '{out}'"
            )
        tnames = layer.get("tensor_names", {})
        for name in tnames.get("inputs", []):
            assert "hidden-tensor" not in name, (
                f"{lid}: tensor_names.inputs contains hidden-tensor ref '{name}'"
            )
        for name in tnames.get("outputs", []):
            assert "hidden-tensor" not in name, (
                f"{lid}: tensor_names.outputs contains hidden-tensor ref '{name}'"
            )


# ---------------------------------------------------------------------------
# Test: Simple linear chain (layer_norm -> linear)
# ---------------------------------------------------------------------------
class TestSimpleLinearChain:
    """layer_norm output feeds linear via hidden-tensor. After conversion,
    linear should reference layer_norm directly."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)
            self.fc = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            h = self.norm(x)
            return self.fc(h)

    def get_inputs():
        return [torch.randn(2, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def graph(self, tmp_path):
        return _run_to_einsum(tmp_path, self.MODEL_SOURCE)

    def test_no_hidden_tensor_refs(self, graph):
        _assert_no_hidden_tensor_refs(graph["layers"])

    def test_linear_references_norm(self, graph):
        """The matmul layer's activation input should reference the norm layer."""
        layers = graph["layers"]
        for lid, layer in layers.items():
            if layer.get("type") != "linear":
                continue
            conn_inputs = layer["connections"]["inputs"]
            tname_inputs = layer["tensor_names"]["inputs"]
            activation_inputs = [
                (c, n) for c, n in zip(conn_inputs, tname_inputs)
                if "parameter" not in c and "Weight" not in n
            ]
            for conn, tname in activation_inputs:
                assert "layer_norm" in conn or "norm" in conn, (
                    f"{lid}: expected norm in connection, got '{conn}'"
                )
                assert ".Output" in tname, (
                    f"{lid}: expected .Output in tensor_name, got '{tname}'"
                )


# ---------------------------------------------------------------------------
# Test: Fan-out (one hidden-tensor feeds multiple consumers)
# ---------------------------------------------------------------------------
class TestFanOutHiddenTensor:
    """layer_norm -> hidden-tensor -> [q_proj, k_proj, v_proj].
    All three should reference layer_norm directly."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            h = self.norm(x)
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
            return q + k + v

    def get_inputs():
        return [torch.randn(1, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def graph(self, tmp_path):
        return _run_to_einsum(tmp_path, self.MODEL_SOURCE)

    def test_no_hidden_tensor_refs(self, graph):
        _assert_no_hidden_tensor_refs(graph["layers"])

    def test_all_projections_reference_norm(self, graph):
        """All q/k/v projections should have norm as activation input."""
        layers = graph["layers"]
        linear_layers = [
            (lid, l) for lid, l in layers.items()
            if l.get("type") == "linear" and l.get("is_real_einsum")
        ]
        assert len(linear_layers) >= 3, f"Expected >= 3 linear layers, got {len(linear_layers)}"
        for lid, layer in linear_layers:
            conn_inputs = layer["connections"]["inputs"]
            has_norm_input = any("norm" in c or "layer_norm" in c for c in conn_inputs)
            assert has_norm_input, (
                f"{lid}: expected norm in connections.inputs, got {conn_inputs}"
            )


# ---------------------------------------------------------------------------
# Test: Linear with bias (split into matmul + bias_add)
# ---------------------------------------------------------------------------
class TestLinearWithBiasHiddenTensor:
    """nn.Linear(bias=True) after layer_norm. The split should not leak
    hidden-tensor refs into either the matmul or bias_add layer."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)
            self.fc = nn.Linear(64, 32, bias=True)

        def forward(self, x):
            h = self.norm(x)
            return self.fc(h)

    def get_inputs():
        return [torch.randn(2, 16, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def graph(self, tmp_path):
        return _run_to_einsum(tmp_path, self.MODEL_SOURCE)

    def test_no_hidden_tensor_refs(self, graph):
        _assert_no_hidden_tensor_refs(graph["layers"])


# ---------------------------------------------------------------------------
# Test: Deep chain (multiple hidden-tensors in sequence)
# ---------------------------------------------------------------------------
class TestDeepChain:
    """Multiple layers chained: norm -> linear -> relu -> linear.
    Each intermediate hidden-tensor should be resolved."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            h = self.norm(x)
            h = F.relu(self.fc1(h))
            return self.fc2(h)

    def get_inputs():
        return [torch.randn(1, 8, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def graph(self, tmp_path):
        return _run_to_einsum(tmp_path, self.MODEL_SOURCE)

    def test_no_hidden_tensor_refs(self, graph):
        _assert_no_hidden_tensor_refs(graph["layers"])

    def test_tensor_names_use_output_suffix(self, graph):
        """All activation tensor_names should use <op>.Output format."""
        for lid, layer in graph["layers"].items():
            if layer.get("type") == "start":
                continue
            tnames = layer.get("tensor_names", {})
            for name in tnames.get("inputs", []):
                if "Weight" not in name:
                    assert ".Output" in name, (
                        f"{lid}: activation input name '{name}' missing .Output suffix"
                    )


# ---------------------------------------------------------------------------
# Test: Conv2d with hidden-tensor input
# ---------------------------------------------------------------------------
class TestConv2dHiddenTensor:
    """Conv2d after an indexing op (getitem). The hidden-tensor between
    getitem and conv2d should be resolved."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        def forward(self, x):
            frame = x[:, 0, :, :, :]
            return self.conv(frame)

    def get_inputs():
        return [torch.randn(1, 2, 3, 32, 32)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def graph(self, tmp_path):
        return _run_to_einsum(tmp_path, self.MODEL_SOURCE)

    def test_no_hidden_tensor_refs(self, graph):
        _assert_no_hidden_tensor_refs(graph["layers"])
