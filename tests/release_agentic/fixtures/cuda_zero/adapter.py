# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter whose reference output is ALL ZEROS (vacuous-pass fixture).

y = x * 0.  A DSL that returns zeros for any input passes numerically.
The checker only prints a WARNING; this fixture pins what the routes record.
"""
import numpy as np

_REFERENCE_SRC = (
    "import torch\n"
    "\n"
    "@torch.no_grad()\n"
    "def run(x):\n"
    "    return x * 0.0\n"
)


def get_source() -> str:
    return _REFERENCE_SRC


def get_inputs() -> dict:
    return {"x": (np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0)}


def kernel_fn(inputs: dict) -> dict:
    import torch
    ns = {}
    exec(get_source(), ns)
    return {"output": ns["run"](torch.from_numpy(np.ascontiguousarray(inputs["x"]))).numpy()}
