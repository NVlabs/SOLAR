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

"""MACs of addmm, baddbmm, addbmm, addmv, mv and dot.

These ops had no einsum handler and fell through to the generic
elementwise fallback, so they reported zero tensor-core MACs. GPT-2's
``Conv1D`` (``torch.addmm(bias, x, weight)``) and baddbmm attention scores
then vanished from the compute roofline.
"""

from textwrap import dedent

import pytest

from test_graph_analyzer_memory import _run_full_pipeline


def _model(expr: str, shapes) -> str:
    inputs = ", ".join(f"torch.randn{tuple(s)}" for s in shapes)
    return dedent(f"""\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, *a):
            return {expr}

    def get_inputs():
        return [{inputs}]

    def get_init_inputs():
        return []
    """)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


CASES = {
    # name: (expression, input shapes, output shape, MACs)
    "addmm": ("torch.addmm(a[0], a[1], a[2])", [(6,), (4, 5), (5, 6)], (4, 6), 4 * 5 * 6),
    "addmm_full_bias": ("a[0].addmm(a[1], a[2], beta=0.5, alpha=2.0)",
                        [(4, 6), (4, 5), (5, 6)], (4, 6), 4 * 5 * 6),
    "baddbmm": ("torch.baddbmm(a[0], a[1], a[2])", [(3, 4, 6), (3, 4, 5), (3, 5, 6)], (3, 4, 6),
                3 * 4 * 5 * 6),
    "addbmm": ("torch.addbmm(a[0], a[1], a[2])", [(4, 6), (3, 4, 5), (3, 5, 6)], (4, 6),
               3 * 4 * 5 * 6),
    "addmv": ("torch.addmv(a[0], a[1], a[2])", [(4,), (4, 5), (5,)], (4,), 4 * 5),
    "mv": ("torch.mv(a[0], a[1])", [(4, 5), (5,)], (4,), 4 * 5),
    "dot": ("torch.dot(a[0], a[1])", [(7,), (7,)], (), 7),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_macs_and_traffic(tmp_path, name):
    expr, shapes, out_shape, macs = CASES[name]
    analysis = _run_full_pipeline(tmp_path, _model(expr, shapes))
    total = analysis["total"]
    assert total["macs"] == macs
    assert total["flops"] == 2 * macs
    # Every operand is read once and the output written once.
    assert total["fused_elements"] == sum(_numel(s) for s in shapes) + _numel(out_shape)


GPT2_CONV1D = """\
import torch
import torch.nn as nn

class Conv1D(nn.Module):
    # Same forward as transformers.pytorch_utils.Conv1D (GPT-2 c_attn/c_proj/c_fc).
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = Conv1D(96, 32)

    def forward(self, x):
        return self.c_attn(x)

def get_inputs():
    return [torch.randn(2, 8, 32)]

def get_init_inputs():
    return []
"""


def test_gpt2_conv1d(tmp_path):
    analysis = _run_full_pipeline(tmp_path, GPT2_CONV1D)
    x, weight, bias, out = 2 * 8 * 32, 32 * 96, 96, 2 * 8 * 96
    assert analysis["total"]["macs"] == 2 * 8 * 32 * 96
    assert analysis["total"]["fused_elements"] == x + weight + bias + out
