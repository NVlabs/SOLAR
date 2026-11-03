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

"""Rank extents under broadcasting.

A rank shared by several operands can be 1 in one of them and N in another
(``[2, 1, M, K] @ [3, K, N]``). The extent of that rank is N; taking it from
whichever operand comes first undercounts the work by N whenever the
broadcast operand is listed first.
"""

import pytest

from solar.common.types import TensorShapes
from solar.einsum.ops.base import compute_cost_from_equation

from test_graph_analyzer_memory import _run_full_pipeline


@pytest.mark.parametrize(
    "equation, inputs, outputs, expected",
    [
        # Broadcast batch rank in the first operand.
        ("B0B1MK,B1KN->B0B1MN", [[2, 1, 4, 5], [3, 5, 6]], [[2, 3, 4, 6]], 2 * 3 * 4 * 5 * 6),
        # Same shapes, broadcast operand second: already correct.
        ("B0B1MK,B1KN->B0B1MN", [[2, 3, 4, 5], [1, 5, 6]], [[2, 3, 4, 6]], 2 * 3 * 4 * 5 * 6),
        # Elementwise outer broadcast.
        ("AB,AB->AB", [[4, 1], [1, 5]], [[4, 5]], 4 * 5),
        # keepdim reduction: a size-1 output rank must not shrink the input extent.
        ("ABC->ABC", [[2, 8, 16]], [[2, 1, 16]], 2 * 8 * 16),
    ],
)
def test_compute_cost_uses_broadcast_extent(equation, inputs, outputs, expected):
    ts = TensorShapes(inputs=inputs, outputs=outputs)
    assert compute_cost_from_equation(equation, ts) == expected


MATMUL_BROADCAST = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, w):
        return torch.matmul(x, w)

def get_inputs():
    return [torch.randn(2, 1, 4, 5), torch.randn(3, 5, 6)]

def get_init_inputs():
    return []
"""

MUL_BROADCAST = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, scale, x):
        return scale * x

def get_inputs():
    return [torch.randn(1, 8, 1, 1), torch.randn(2, 8, 4, 4)]

def get_init_inputs():
    return []
"""


def test_matmul_broadcast_batch_macs(tmp_path):
    analysis = _run_full_pipeline(tmp_path, MATMUL_BROADCAST)
    # out [2, 3, 4, 6], contraction K = 5
    assert analysis["total"]["macs"] == 2 * 3 * 4 * 6 * 5


def test_elementwise_broadcast_ops(tmp_path):
    analysis = _run_full_pipeline(tmp_path, MUL_BROADCAST)
    # One multiply per output element.
    assert analysis["total"]["other_ops"] == 2 * 8 * 4 * 4
