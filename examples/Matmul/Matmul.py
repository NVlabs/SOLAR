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

"""Single matrix multiplication example for Solar analysis.

This is a bare torch.matmul with a learnable weight parameter.
Useful for verifying the Solar pipeline produces correct MACs and memory
element counts for the most fundamental operation.

Architecture:
    Weight [4, 32, 64] @ Input [64, 128] -> Output [4, 32, 128]

Expected results:
    MACs          = 4 * 32 * 64 * 128 = 1,048,576
    Weight elems   = 4 * 32 * 64       = 8,192
    Input elems  = 64 * 128           = 8,192
    Output elems  = 4 * 32 * 128       = 16,384
    Unfused elems = 8192 + 8192 + 16384 = 32,768
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Single matmul with a learnable weight."""

    def __init__(self):
        super().__init__()
        self.weight =  nn.Parameter(torch.randn(4, 32, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.weight, x)


def get_inputs():
    """Return input tensor [batch=4, seq=32, hidden=64]."""
    torch.manual_seed(0)
    return [torch.randn(64, 128)]


def get_init_inputs():
    """No extra init args needed."""
    return []
