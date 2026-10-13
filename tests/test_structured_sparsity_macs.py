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

"""Triangular / masked operands skip MACs.

Causal attention and the Mamba-2 chunk scan multiply a dense product by a
lower-triangular mask. Every MAC of an einsum uses exactly one element of
each operand, so a kernel skips the masked half: the ``P V`` product reads a
half-dense ``P`` and the ``Q K^T`` product only needs the half of its output
that survives the mask. The analyzer tracks the live density through
``tril``/``triu``, ``masked_fill``, ``where``, ``~mask``, ``x * mask``,
``exp``/``softmax`` and views, and scales einsum MACs by it.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.analysis.graph_analyzer import _triangular_density
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


CAUSAL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, q, k, v):
            # q, k, v: [2, 4, 128, 64]
            scores = torch.matmul(q, k.transpose(-1, -2))                 # dense product...
            mask = torch.tril(torch.ones(128, 128, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))            # ...only the lower half survives
            p = torch.softmax(scores, dim=-1)                            # exp(-inf) = 0
            return torch.matmul(p, v)                                    # half the MACs

    def get_inputs():
        return [torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)]

    def get_init_inputs():
        return []
"""

SSD = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, c, b, x, a):
            # c, b: [2, 128, 32]; x: [2, 128, 16]; a: [2, 128]
            g = torch.einsum('bis,bjs->bij', c, b)                        # dense C B^T
            mask = torch.tril(torch.ones(128, 128, dtype=torch.bool, device=a.device))
            seg = torch.cumsum(a, dim=-1)
            l = torch.exp((seg[:, :, None] - seg[:, None, :]).masked_fill(~mask, float('-inf')))
            m = g * l                                                    # lower-triangular
            return torch.einsum('bij,bjd->bid', m, x)

    def get_inputs():
        return [torch.randn(2, 128, 32), torch.randn(2, 128, 32), torch.randn(2, 128, 16), torch.randn(2, 128)]

    def get_init_inputs():
        return []
"""


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
    assert EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp32") is not None
    return yaml.safe_load((analysis_dir / "analysis.yaml").read_text())


def test_triangular_density():
    assert _triangular_density([128, 128], "tril", 0) == (128 * 129 / 2) / (128 * 128)
    assert _triangular_density([128, 128], "tril", -1) == (128 * 127 / 2) / (128 * 128)
    assert _triangular_density([128, 128], "triu", 1) == (128 * 127 / 2) / (128 * 128)
    assert _triangular_density([4, 8], "tril", 0) == (1 + 2 + 3 + 4) / 32


def test_causal_attention_counts_half_the_macs(tmp_path):
    a = _analyze(tmp_path, CAUSAL)
    mm = sorted((l for l in a["layers"].values() if l["type"] == "matmul"), key=lambda l: l["macs_dense"])
    assert len(mm) == 2
    tri = (128 * 129 / 2) / (128 * 128)
    for l in mm:
        assert l["macs_dense"] == 2 * 4 * 128 * 128 * 64
        assert abs(l["mac_sparsity_fraction"] - tri) < 1e-9, l
        assert l["macs"] == round(l["macs_dense"] * tri)
    assert a["total"]["macs"] == 2 * round(2 * 4 * 128 * 128 * 64 * tri)


def test_ssd_chunk_scan_counts_half_the_macs(tmp_path):
    a = _analyze(tmp_path, SSD)
    es = {l["einsum_equation"]: l for l in a["layers"].values() if l["type"] == "einsum"}
    tri = (128 * 129 / 2) / (128 * 128)
    # C B^T: only the lower half of the output is ever used (multiplied by L).
    # M X: M is lower-triangular.
    assert len(es) == 2
    for l in es.values():
        assert abs(l["mac_sparsity_fraction"] - tri) < 1e-9, l
        assert l["macs"] == round(l["macs_dense"] * tri)


def test_dense_graph_unchanged(tmp_path):
    src = """
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, q, k):
                return torch.matmul(q, k.transpose(-1, -2))

        def get_inputs():
            return [torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)]

        def get_init_inputs():
            return []
    """
    a = _analyze(tmp_path, src)
    l = [l for l in a["layers"].values() if l["type"] == "matmul"][0]
    assert l["mac_sparsity_fraction"] == 1.0 and l["macs"] == l["macs_dense"]


ADDITIVE_CAUSAL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, q, k, v):
            # q, k, v: [2, 4, 128, 64]; the HF-style additive causal mask
            scores = torch.matmul(q, k.transpose(-1, -2))
            causal = torch.triu(torch.full((128, 128), float('-inf')), diagonal=1)
            scores = scores + causal
            p = torch.softmax(scores, dim=-1)
            return torch.matmul(p, v)

    def get_inputs():
        return [torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)]

    def get_init_inputs():
        return []
"""


def test_additive_triu_inf_mask_counts_half_the_macs(tmp_path):
    a = _analyze(tmp_path, ADDITIVE_CAUSAL)
    mm = [l for l in a["layers"].values() if l["type"] == "matmul"]
    assert len(mm) == 2
    live = 1 - (128 * 127 / 2) / (128 * 128)   # zeroed lower triangle incl. diagonal
    for l in mm:
        assert abs(l["mac_sparsity_fraction"] - live) < 1e-9, l
