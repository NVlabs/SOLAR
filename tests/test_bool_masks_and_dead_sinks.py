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

"""Masks are 1 B and unreturned results are not DRAM writes.

Found on the SOL-ExecBench nvfp4 problems: the reference quantizer builds
fourteen ``(x > a) & (x < b)`` masks and applies them with in-place
``result[mask] = v``. torchview does not trace the in-place consumer, so each
mask was (a) priced at 4 B because the dtype repair propagated the fp32 input
width through the comparison, and (b) charged as an external output because it
had no consumer. Together that inflated the fused SOL of a 2 GB-traffic
problem by ~7x.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.common.utils import yaml_safe_load
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor
from solar.perf import EinsumGraphPerfModel


MASKED = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            # x: fp32 [64, 256]
            keep = x > 0                      # bool, consumed below
            dead = (x < 0) & (x > -1)         # bool, never consumed nor returned
            return x * keep

    def get_inputs():
        return [torch.randn(64, 256)]

    def get_init_inputs():
        return []
"""

N = 64 * 256


def _run(tmp_path: Path, src: str):
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
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp32")
    assert analysis is not None
    return einsum_dir / "einsum_graph_renamed.yaml", analysis_dir / "analysis.yaml"


def test_comparison_and_logical_ops_are_bool(tmp_path):
    graph_path, _ = _run(tmp_path, MASKED)
    g = yaml_safe_load(graph_path.read_text())
    by_type = {}
    for lid, layer in g["layers"].items():
        by_type.setdefault(layer["type"], []).append(layer)
    for t in ("gt", "lt", "__and__"):
        assert t in by_type, f"missing {t}: {sorted(by_type)}"
        for layer in by_type[t]:
            assert layer["tensor_dtypes"]["outputs"] == ["torch.bool"], (t, layer["tensor_dtypes"])
    # The multiply mixing fp32 values with the mask stays fp32.
    assert by_type["mul"][0]["tensor_dtypes"]["outputs"] == ["torch.float32"]
    # The converter records which layer feeds the declared output.
    assert g["model_output_ops"] == [[lid for lid, l in g["layers"].items() if l["type"] == "mul"][0]]


def test_dead_mask_is_not_an_external_write(tmp_path):
    _, analysis_path = _run(tmp_path, MASKED)
    a = yaml.safe_load(analysis_path.read_text())
    total = a["total"]
    # External traffic: read x once (fp32), write the fp32 product. The
    # consumed mask is on-chip; the dead mask is not written anywhere.
    assert total["external_output_bytes"] == N * 4, total
    assert total["fused_bytes"] == N * 8, total
    assert total["fused_elements"] == 2 * N, total
    layers = a["layers"]
    dead = [l for l in layers.values() if l["type"] == "__and__"][0]
    assert dead["fused_elements"] == 0
    assert dead["bytes_per_element"]["outputs"] == [1.0]
    perf = EinsumGraphPerfModel(dtype_bytes=True).predict(
        analysis_path, tmp_path / "perf", arch_config="B200", precision="fp32")
    assert perf["fused"]["memory_bytes"] == N * 8
