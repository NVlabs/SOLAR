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

"""TwoInputMatmul: a single torch.matmul of two model inputs (no parameters).

Architecture:
    A [32, 64] @ B [64, 128] -> Output [32, 128]

Expected results (mechanical route):
    MACs            = 32 * 64 * 128 = 262,144
    Input elems     = 32*64 + 64*128 = 2,048 + 8,192 = 10,240
    Output elems    = 32 * 128       = 4,096
    Unfused elems   = 10,240 + 4,096 = 14,336
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Matrix product of two inputs."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def get_inputs():
    """Return the two operands [32, 64] and [64, 128]."""
    torch.manual_seed(0)
    return [torch.randn(32, 64), torch.randn(64, 128)]


def get_init_inputs():
    """No init args."""
    return []
