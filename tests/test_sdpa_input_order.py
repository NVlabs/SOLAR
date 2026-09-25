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

"""SDPA expansion must pick Q/K/V by argument order, not by sorted name.

``_expand_sdpa`` used to take ``sorted(op_graph.predecessors(node_id))`` as
(Q, K, V).  With an ``attn_mask`` whose producer sorts before the Q/K/V
producers (``Model.to`` < ``Model.transpose``), the mask became Q and the
pre-AF shape check failed with "a tensor name is referenced with multiple
distinct shapes" (L2/046 conformer relative-position attention).
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


SDPA_WITH_MASK = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def forward(self, q, k, v, bias):
            # producers: Model.transpose, Model.transpose_1, Model.transpose_2, Model.to
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            mask = bias.to(q.dtype)          # sorts before "transpose"
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def get_inputs():
        B, S, H, D = 2, 16, 4, 8
        return [torch.randn(B, S, H, D), torch.randn(B, S, H, D), torch.randn(B, S, H, D),
                torch.randn(B, H, S, S)]

    def get_init_inputs():
        return []
"""


SDPA_5D = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def forward(self, q, k, v):
            # blocked attention: [B, M(blocks), Nh, C(block len), D]
            return F.scaled_dot_product_attention(q, k, v)

    def get_inputs():
        B, M, Nh, C, D = 2, 3, 4, 16, 8
        return [torch.randn(B, M, Nh, C, D), torch.randn(B, M, Nh, C, D), torch.randn(B, M, Nh, C, D)]

    def get_init_inputs():
        return []
"""


def _run_to_einsum(tmp_path: Path, src: str) -> dict:
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    with open(einsum_dir / "einsum_graph.yaml") as f:
        return yaml.safe_load(f)["layers"]


def test_sdpa_qkv_follow_argument_order_with_mask(tmp_path):
    layers = _run_to_einsum(tmp_path, SDPA_WITH_MASK)
    qk = next(v for k, v in layers.items() if k.endswith(".qk_matmul"))
    av = next(v for k, v in layers.items() if k.endswith(".av_matmul"))
    # Q and K come from the transposes, never from the mask cast.
    assert all("Model.to" not in n for n in qk["connections"]["inputs"]), qk["connections"]
    assert qk["connections"]["inputs"] == ["Model.transpose", "Model.transpose_1"]
    assert av["connections"]["inputs"][-1] == "Model.transpose_2"
    # Recorded shapes agree with the producers (B,H,S,D for Q/K, not B,H,S,S).
    assert qk["tensor_shapes"]["inputs"][0] == [2, 4, 16, 8]


def test_sdpa_with_five_dim_inputs_keeps_all_leading_dims(tmp_path):
    layers = _run_to_einsum(tmp_path, SDPA_5D)
    qk = next(v for k, v in layers.items() if k.endswith(".qk_matmul"))
    av = next(v for k, v in layers.items() if k.endswith(".av_matmul"))
    # scores are [B, M, Nh, C, C]; output is [B, M, Nh, C, D]
    assert qk["tensor_shapes"]["outputs"] == [[2, 3, 4, 16, 16]]
    assert av["tensor_shapes"]["outputs"] == [[2, 3, 4, 16, 8]]
    assert len(qk["operands"]["Input"]) == 5
    # equation labels stay single-letter so the compute-cost parser works
    lhs = qk["einsum_equation"].split("->")[0].split(",")[0]
    assert len(lhs) == 5 and lhs.endswith("QD")
