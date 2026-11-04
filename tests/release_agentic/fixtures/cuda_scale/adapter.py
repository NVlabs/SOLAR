# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic CUDA-contract adapter (release_agentic): y[b,h] = x[b,h] * w[h] + 1.

Self-contained, deterministic and non-degenerate.  The reference source is a
verbatim torch program executed by kernel_fn (torch-golden, CPU only); the
`cuda` source type selects the adapter CONTRACT, not GPU execution.
Validation shapes: B=4 (var), H=16 (const).  The definition's workload line
declares the ORIGINAL B=32 so capped-vs-original provenance is testable.
"""
import numpy as np

_REFERENCE_SRC = (
    "import torch\n"
    "\n"
    "@torch.no_grad()\n"
    "def run(x, w):\n"
    "    assert x.shape[1] == 16\n"
    "    return x * w + 1.0\n"
)
B, H = 4, 16


def get_source() -> str:
    return _REFERENCE_SRC


def get_inputs() -> dict:
    x = (np.arange(B * H, dtype=np.float32).reshape(B, H) + 1.0) / 8.0
    w = (np.arange(H, dtype=np.float32) + 2.0) / 4.0
    return {"x": x, "w": w}


def kernel_fn(inputs: dict) -> dict:
    import torch
    ns = {}
    exec(get_source(), ns)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in inputs.items()}
    return {"output": ns["run"](t["x"], t["w"]).numpy()}
