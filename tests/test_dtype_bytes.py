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

"""Per-tensor dtype byte accounting.

A single graph-wide ``bytes_per_element`` misprices mixed graphs: a bool mask
counts at 2 B, the fp32 output of an fp8 problem at 1 B. The analyzer now also
emits byte totals at each tensor's own width and the perf model uses them when
constructed with ``dtype_bytes=True`` (the SOL-ExecBench runner's default);
the legacy "precision is a modelling knob" behaviour is unchanged otherwise.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor
from solar.perf import EinsumGraphPerfModel


MIXED_DTYPES = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, mask):
            # x: bf16 [64, 256]; mask: bool [64, 256]; output: fp32 [64, 256]
            return (x * mask).to(torch.float32)

    def get_inputs():
        return [torch.randn(64, 256, dtype=torch.bfloat16), torch.rand(64, 256) > 0.5]

    def get_init_inputs():
        return []
"""

N = 64 * 256


def _analyze(tmp_path: Path, src: str):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp16")
    assert analysis is not None
    return analysis_dir / "analysis.yaml"


def test_analysis_emits_dtype_aware_byte_totals(tmp_path):
    analysis_path = _analyze(tmp_path, MIXED_DTYPES)
    a = yaml.safe_load(analysis_path.read_text())
    assert a["metadata"]["bytes_accounting"] == "per-tensor-dtype"
    total = a["total"]
    # External I/O: bf16 x (2 B) + bool mask (1 B) + fp32 output (4 B).
    assert total["external_input_bytes"] == N * 2 + N * 1, total
    assert total["external_output_bytes"] == N * 4, total
    assert total["fused_bytes"] == N * 7
    # Element counts are unchanged by the new accounting (mask now counted, not zeroed).
    assert total["fused_elements"] == 3 * N


def test_perf_model_uses_dtype_bytes_only_when_asked(tmp_path):
    analysis_path = _analyze(tmp_path, MIXED_DTYPES)
    legacy = EinsumGraphPerfModel().predict(analysis_path, tmp_path / "perf_legacy",
                                            arch_config="B200", precision="fp16")
    dtype = EinsumGraphPerfModel(dtype_bytes=True).predict(analysis_path, tmp_path / "perf_dtype",
                                                           arch_config="B200", precision="fp16")
    assert legacy is not None and dtype is not None
    # Legacy: every element at 2 B (fp16 knob).
    assert legacy["fused"]["memory_bytes"] == 3 * N * 2
    assert legacy["workload"]["bytes_per_element"] == 2
    # dtype-aware: real widths.
    assert dtype["fused"]["memory_bytes"] == N * 7
    assert dtype["workload"]["bytes_per_element"] == "per-tensor-dtype"
    assert dtype["workload"]["mac_rate_bytes_per_element"] == 2
    # Memory-bound elementwise op: runtime scales with bytes.
    assert abs(dtype["fused"]["runtime_ms"] / legacy["fused"]["runtime_ms"] - 7 / 6) < 1e-6
