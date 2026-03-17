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

"""End-to-end regression tests for the Solar analysis pipeline.

These tests start from a PyTorch model definition (as a .py source file),
run the full Solar pipeline (graph extraction -> einsum conversion -> analysis),
and verify the final MACs, unfused_elements, and fused_elements.

If any of these values change, the test will fail — indicating a regression.
"""

import pytest
import yaml
from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


def _run_full_pipeline(tmp_path: Path, model_source: str) -> dict:
    """Run full Solar pipeline from source code to analysis.

    Steps:
        1. Write model source to .py file
        2. PyTorchProcessor: source -> pytorch_graph.yaml
        3. PyTorchToEinsum: pytorch_graph.yaml -> einsum_graph.yaml
        4. EinsumGraphAnalyzer: einsum_graph.yaml -> analysis.yaml

    Returns:
        The analysis dict.
    """
    # Write model source file
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(model_source))

    # Step 1: Extract PyTorch graph
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

    pytorch_graph_path = graph_dir / "pytorch_graph.yaml"
    assert pytorch_graph_path.exists(), "pytorch_graph.yaml not produced"

    # Step 2: Convert to einsum graph
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    converter = PyTorchToEinsum()
    einsum_graph = converter.convert(str(pytorch_graph_path), str(einsum_dir))
    assert einsum_graph is not None, "PyTorchToEinsum.convert failed"

    renamed_path = einsum_dir / "einsum_graph_renamed.yaml"
    assert renamed_path.exists(), "einsum_graph_renamed.yaml not produced"

    # Step 3: Analyze
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(
        str(renamed_path), str(analysis_dir), precision="fp32", copy_graph=False
    )
    assert analysis is not None, "EinsumGraphAnalyzer.analyze_graph failed"

    return analysis


# ---------------------------------------------------------------------------
# Test: Single matmul
# ---------------------------------------------------------------------------
class TestMatmulEndToEnd:
    """End-to-end: torch.matmul([4,32,64], [64,128]) -> [4,32,128]."""

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

    def test_total_macs(self, analysis):
        # matmul: B0*B1*K*N = 4*32*64*128 = 1048576
        expected_macs = 4 * 32 * 64 * 128
        assert analysis["total"]["macs"] == expected_macs

    def test_total_flops(self, analysis):
        expected_flops = 2 * 4 * 32 * 64 * 128
        assert analysis["total"]["flops"] == expected_flops

    def test_unfused_elements(self, analysis):
        # input: 4*32*64 = 8192, weight: 64*128 = 8192, output: 4*32*128 = 16384
        expected = (4 * 32 * 64) + (64 * 128) + (4 * 32 * 128)
        assert analysis["total"]["unfused_elements"] == expected

    def test_macs_positive(self, analysis):
        assert analysis["total"]["macs"] > 0

    def test_unfused_greater_equal_fused(self, analysis):
        assert analysis["total"]["unfused_elements"] >= analysis["total"]["fused_elements"]


# ---------------------------------------------------------------------------
# Test: Single nn.Linear (no bias)
# ---------------------------------------------------------------------------
class TestLinearNoBiasEndToEnd:
    """End-to-end: nn.Linear(256, 512, bias=False) with input [2,128,256]."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 512, bias=False)

        def forward(self, x):
            return self.fc(x)

    def get_inputs():
        return [torch.randn(2, 128, 256)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_total_macs(self, analysis):
        # linear MACs = B0*B1*K*N = 2*128*256*512 = 33554432
        expected_macs = 2 * 128 * 256 * 512
        print(analysis)
        assert analysis["total"]["macs"] == expected_macs

    def test_unfused_elements(self, analysis):
        # input: 2*128*256 = 65536, weight: 512*256 = 131072, output: 2*128*512 = 131072
        expected = (2 * 128 * 256) + (512 * 256) + (2 * 128 * 512)
        assert analysis["total"]["unfused_elements"] == expected


# ---------------------------------------------------------------------------
# Test: nn.Linear with bias (split into matmul + add)
# ---------------------------------------------------------------------------
class TestLinearWithBiasEndToEnd:
    """End-to-end: nn.Linear(64, 32, bias=True) with input [2,16,64].

    Solar splits linear+bias into matmul + bias_add.
    Only the matmul should contribute MACs; bias_add should have 0 MACs.
    """

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=True)

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

    def test_total_macs(self, analysis):
        # Only the matmul part contributes MACs: B0*B1*K*N = 2*16*64*32 = 65536
        # Bias add has 0 MACs (not real einsum)
        expected_macs = 2 * 16 * 64 * 32
        assert analysis["total"]["macs"] == expected_macs

    def test_has_bias_add_layer(self, analysis):
        # Should have a bias_add layer with is_real_einsum=False
        bias_layers = [
            lid for lid, l in analysis["layers"].items()
            if "bias_add" in lid
        ]
        assert len(bias_layers) > 0, "No bias_add layer found"

    def test_bias_add_zero_macs(self, analysis):
        for lid, layer in analysis["layers"].items():
            if "bias_add" in lid:
                assert layer["macs"] == 0
                assert layer["other_ops"] == 2 * 16 * 32


# ---------------------------------------------------------------------------
# Test: nn.Conv2d
# ---------------------------------------------------------------------------
class TestConv2dEndToEnd:
    """End-to-end: nn.Conv2d(3, 16, kernel_size=3, padding=1) with input [1,3,32,32]."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    def get_inputs():
        return [torch.randn(1, 3, 32, 32)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_total_macs(self, analysis):
        # conv2d MACs = output_elems * in_channels * kernel_elems
        # output: [1,16,32,32] -> 1*16*32*32 = 16384
        # in_channels = 3, kernel = 3*3 = 9
        # MACs = 16384 * 3 * 9 = 442368
        expected = (1 * 16 * 32 * 32) * 3 * (3 * 3)
        assert analysis["total"]["macs"] == expected

    def test_unfused_elements(self, analysis):
        # input: 1*3*32*32 = 3072, weight: 16*3*3*3 = 432, output: 1*16*32*32 = 16384
        expected = (1 * 3 * 32 * 32) + (16 * 3 * 3 * 3) + (1 * 16 * 32 * 32)
        assert analysis["total"]["unfused_elements"] == expected


# ---------------------------------------------------------------------------
# Test: Two-layer graph (matmul -> relu) — fused vs unfused
# ---------------------------------------------------------------------------
class TestMatmulReluFusionEndToEnd:
    """End-to-end: matmul + relu. Tests that fused < unfused."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 128))

        def forward(self, x):
            y = torch.matmul(x, self.weight)
            return F.relu(y)

    def get_inputs():
        return [torch.randn(4, 32, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_unfused_greater_than_fused(self, analysis):
        total = analysis["total"]
        assert total["unfused_elements"] > total["fused_elements"]

    def test_intermediate_elements_positive(self, analysis):
        assert analysis["total"]["intermediate_elements"] > 0

    def test_fused_equals_unfused_minus_intermediate(self, analysis):
        total = analysis["total"]
        assert total["fused_elements"] == total["unfused_elements"] - total["intermediate_elements"]

    def test_macs_only_from_matmul(self, analysis):
        # relu should contribute 0 MACs, only matmul contributes
        expected = 4 * 32 * 64 * 128
        assert analysis["total"]["macs"] == expected

    def test_relu_has_zero_macs(self, analysis):
        for lid, layer in analysis["layers"].items():
            if layer["type"] in ("relu", "relu_"):
                assert layer["macs"] == 0
                assert layer["is_real_einsum"] is False


# ---------------------------------------------------------------------------
# Test: Elementwise only (no real einsum)
# ---------------------------------------------------------------------------
class TestElementwiseOnlyEndToEnd:
    """End-to-end: x + y, should have 0 total MACs."""

    MODEL_SOURCE = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    def get_inputs():
        return [torch.randn(4, 32, 64), torch.randn(4, 32, 64)]

    def get_init_inputs():
        return []
    """

    @pytest.fixture
    def analysis(self, tmp_path):
        return _run_full_pipeline(tmp_path, self.MODEL_SOURCE)

    def test_total_macs_zero(self, analysis):
        assert analysis["total"]["macs"] == 0

    def test_total_other_ops_nonzero(self, analysis):
        assert analysis["total"]["other_ops"] > 0

    def test_unfused_elements_positive(self, analysis):
        assert analysis["total"]["unfused_elements"] > 0
