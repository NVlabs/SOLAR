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

"""Single Conv2d layer example for Solar analysis.

Architecture:
    Input [1, 3, 32, 32] -> Conv2d(3, 16, kernel_size=3, padding=1) -> Output [1, 16, 32, 32]

Expected results:
    MACs          = output_elems * in_channels * kernel_elems
                  = (1*16*32*32) * 3 * (3*3) = 442,368
    Input elems   = 1 * 3 * 32 * 32  = 3,072
    Weight elems  = 16 * 3 * 3 * 3   = 432
    Output elems  = 1 * 16 * 32 * 32 = 16,384
    Unfused elems = 3072 + 432 + 16384 = 19,888
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Single Conv2d layer without bias."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def get_inputs():
    """Return input tensor [batch=1, channels=3, height=32, width=32]."""
    torch.manual_seed(0)
    return [torch.randn(1, 3, 32, 32)]


def get_init_inputs():
    """No extra init args needed."""
    return []
