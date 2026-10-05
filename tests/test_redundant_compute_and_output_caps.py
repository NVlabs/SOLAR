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

"""Work the reference code does that a minimal kernel does not.

Found by checking SOLAR's SOL against the measured implementations in the
SOL-ExecBench leaderboard table (a SOL must never exceed a real kernel):

* Mamba-2 expands one B/C group to every head before ``C B^T``: the einsum
  is 16x redundant; a kernel computes the product once per group.
* An audio projector is applied to every window token but only the rows
  at ``audio_token_positions`` are gathered afterwards: only those rows
  need computing.
* ``mask.expand(B, H, T, S).contiguous()`` materialises a broadcast; the
  fastest implementation returns the view and writes the base tensor.
* ``grad = zeros(V, H); grad.index_add_(rows); grad.to(bf16)`` writes only
  the touched rows (the harness provides the zeroed buffer).
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


def _analyze(tmp_path: Path, src: str, precision: str = "fp32"):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    assert EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision=precision) is not None
    return yaml.safe_load((analysis_dir / "analysis.yaml").read_text())


GROUP_SHARED = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, c, b, x):
            # c, b: [2, 128, 1, 32] (one group); x: [2, 128, 16, 64] (16 heads)
            c16 = c.expand(2, 128, 16, 32)
            b16 = b.expand(2, 128, 16, 32)
            g = torch.einsum('bihs,bjhs->bijh', c16, b16)      # identical for every head
            y = torch.einsum('bihs,bihd->bhds', b16, x)         # x varies per head: no redundancy
            return g, y

    def get_inputs():
        return [torch.randn(2, 128, 1, 32), torch.randn(2, 128, 1, 32), torch.randn(2, 128, 16, 64)]

    def get_init_inputs():
        return []
"""


def test_einsum_over_expanded_operands_counted_once(tmp_path):
    a = _analyze(tmp_path, GROUP_SHARED)
    es = {l["einsum_equation"]: l for l in a["layers"].values() if l["type"] == "einsum"}
    g = es["BIHS,BJHS->BIJH"]
    y = es["BIHS,BIHD->BHDS"]
    assert g["macs_dense"] == 2 * 128 * 128 * 16 * 32
    assert abs(g["mac_sparsity_fraction"] - 1 / 16) < 1e-9, g
    assert g["macs"] == g["macs_dense"] // 16
    assert y["mac_sparsity_fraction"] == 1.0 and y["macs"] == y["macs_dense"]


GATHERED_ROWS = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, w, bias, pos):
            # x: [4, 256, 64]; w: [128, 64]; pos: int64 [4, 8] -> only 8 of 256 rows per batch are used
            y = torch.nn.functional.linear(x, w, bias)
            out = torch.zeros(4, 8, 128)
            for i in range(4):
                out[i] = y[i][pos[i]]
            return out

    def get_inputs():
        return [torch.randn(4, 256, 64), torch.randn(128, 64), torch.randn(128),
                torch.randint(0, 256, (4, 8))]

    def get_init_inputs():
        return []
"""


def test_projection_only_needed_for_gathered_rows(tmp_path):
    a = _analyze(tmp_path, GATHERED_ROWS)
    lin = [l for l in a["layers"].values() if l["type"] == "linear"][0]
    assert lin["macs_dense"] == 4 * 256 * 64 * 128
    assert abs(lin["mac_sparsity_fraction"] - 8 / 256) < 1e-6, lin


BROADCAST_OUTPUT = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, t):
            # t: fp32 [64, 64]
            mask = torch.tril(torch.ones(64, 64, dtype=torch.bool))
            return mask[None, None].expand(8, 16, 64, 64).contiguous(), t * 2.0

    def get_inputs():
        return [torch.randn(64, 64)]

    def get_init_inputs():
        return []
"""


def test_materialised_broadcast_output_writes_base_tensor_only(tmp_path):
    a = _analyze(tmp_path, BROADCAST_OUTPUT)
    total = a["total"]
    # The bool mask output is a broadcast of a [64, 64] base: 4 KB, not 8*16*64*64 = 512 KB.
    # The second output is 64*64 fp32 = 16 KB.
    assert total["external_output_bytes"] == 64 * 64 * 1 + 64 * 64 * 4, total


SPARSE_GRAD = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, grad_out, ids):
            # grad_out: fp32 [512, 64]; ids: int64 [512] in [0, 4096)
            table = torch.zeros(4096, 64)
            table.index_add_(0, ids, grad_out)
            return table.to(torch.bfloat16)

    def get_inputs():
        return [torch.randn(512, 64), torch.randint(0, 4096, (512,))]

    def get_init_inputs():
        return []
"""


def test_sparse_update_of_zero_buffer_writes_touched_rows_only(tmp_path):
    a = _analyze(tmp_path, SPARSE_GRAD)
    total = a["total"]
    # 512 updated rows of bf16, not the 4096-row table.
    assert total["external_output_bytes"] == 512 * 64 * 2, total
