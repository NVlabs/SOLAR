# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PyTorch fixture (release_agentic): y = x @ w with a learned w.

Deterministic, tiny, non-degenerate: no random values, no zeros in the
reference output.  `Model` + `get_inputs()` is the documented pytorch golden
contract (the model file itself is the reference; no adapter).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(
            (torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1.0) / 10.0)

    def forward(self, x):
        return x @ self.w


def get_inputs():
    return [(torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1.0) / 7.0]
