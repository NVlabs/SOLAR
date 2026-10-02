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

"""Tensor dtypes are rebuilt from torchview's recorded call arguments.

For models without parameters (every SOL-ExecBench problem) torchview pads
missing dtypes with ``torch.float32``: the output of ``.to(torch.bfloat16)``,
an fp16 model input or an fp8 weight all read as 4 B tensors. The call
arguments in ``raw_attributes`` are reliable, so the converter rebuilds
every dtype from them. This test traces a mixed-precision model, then
overwrites every recorded dtype with float32 (what the padded traces look
like) and checks that the converter still recovers the true widths.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.common.utils import yaml_safe_load
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


MIXED = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, ids, table):
            # x: bf16 [64, 256]; ids: int64 [64]; table: fp16 [1000, 256]
            h = x.to(torch.float32) * 2.0           # fp32 compute
            e = torch.nn.functional.embedding(ids, table)   # fp16 rows
            acc = torch.zeros_like(x, dtype=torch.uint8)    # explicit creation dtype
            idx = torch.argmax(h, dim=-1)                    # int64 indices
            y = h + e.float() + acc.float() + idx.unsqueeze(-1).float()
            return y.to(torch.bfloat16)                      # bf16 output

    def get_inputs():
        return [torch.randn(64, 256, dtype=torch.bfloat16),
                torch.randint(0, 1000, (64,)),
                torch.randn(1000, 256, dtype=torch.float16)]

    def get_init_inputs():
        return []
"""


def _trace(tmp_path: Path, src: str) -> Path:
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    return graph_dir / "pytorch_graph.yaml"


def _pad_all_dtypes_to_float32(graph_path: Path) -> None:
    g = yaml_safe_load(graph_path.read_text())
    for layer in g["layers"].values():
        for key in ("input_dtypes", "output_dtypes"):
            if layer.get(key):
                layer[key] = ["torch.float32"] * len(layer[key])
    graph_path.write_text(yaml.safe_dump(g, sort_keys=False))


def _convert(tmp_path: Path, graph_path: Path):
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_path), str(einsum_dir)) is not None
    return yaml_safe_load((einsum_dir / "einsum_graph_renamed.yaml").read_text())


def _by_type(graph):
    out = {}
    for lid, layer in graph["layers"].items():
        out.setdefault(layer["type"], []).append((lid, layer))
    return out


def test_dtypes_recovered_from_raw_attributes_after_float32_padding(tmp_path):
    graph_path = _trace(tmp_path, MIXED)
    _pad_all_dtypes_to_float32(graph_path)
    g = _convert(tmp_path, graph_path)
    bt = _by_type(g)

    # Model inputs get their real width back from the ops that consume them.
    starts = {lid: l["tensor_dtypes"]["outputs"][0] for lid, l in bt["start"]}
    assert sorted(starts.values()) == ["torch.bfloat16", "torch.float16", "torch.int64"], starts

    casts = {l["tensor_dtypes"]["outputs"][0] for _, l in bt["to"]}
    assert {"torch.float32", "torch.bfloat16"} <= casts, casts
    # embedding(ids, table) -> table dtype, not the int64 index width.
    assert bt["embedding"][0][1]["tensor_dtypes"]["outputs"] == ["torch.float16"]
    # zeros_like(x, dtype=uint8) keeps its explicit dtype.
    assert bt["zeros_like"][0][1]["tensor_dtypes"]["outputs"] == ["torch.uint8"]
    # argmax produces indices.
    assert bt["argmax"][0][1]["tensor_dtypes"]["outputs"] == ["torch.int64"]
    # .float() on the fp16 rows is fp32.
    assert all(l["tensor_dtypes"]["outputs"] == ["torch.float32"] for _, l in bt["float"])
    # The declared output is the final bf16 cast.
    out_op = g["model_output_ops"][0]
    assert g["layers"][out_op]["tensor_dtypes"]["outputs"] == ["torch.bfloat16"]
    assert g["model_outputs"][0]["dtype"] == "torch.bfloat16"


def test_padded_trace_prices_external_io_at_true_widths(tmp_path):
    graph_path = _trace(tmp_path, MIXED)
    _pad_all_dtypes_to_float32(graph_path)
    _convert(tmp_path, graph_path)
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(
        str(tmp_path / "einsum" / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp32")
    assert analysis is not None
    total = yaml.safe_load((analysis_dir / "analysis.yaml").read_text())["total"]
    n = 64 * 256
    # x (bf16) + 64 gathered fp16 rows (the embedding rule does not charge the
    # int64 ids); output bf16.
    assert total["external_input_bytes"] == n * 2 + 64 * 256 * 2, total
    assert total["external_output_bytes"] == n * 2, total
