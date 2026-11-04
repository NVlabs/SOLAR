# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-contract adapter for FlashInfer-Bench/021_rmsnorm_h128 (manual, no-LLM).

Plain RMSNorm over hidden dim: y = x * rsqrt(mean(x^2) + 1e-6) * weight.
Capped: batch_size=8 (var; workloads use 24..192). Const kept: hidden=128.
Golden execs the reference verbatim (embedded below from definition.json
"reference" so this example is self-contained and portable); it computes
in fp32 (`.to(torch.float32)`) and casts back to the input dtype at the end,
so with fp64 verify inputs the golden is the fp32-computed value in fp64 —
dtype-agnostic. `weight` randomized ~1±0.1 (ones-init would hide a missing
multiply).
"""
import numpy as np

# Verbatim copy of definition.json["reference"] for FlashInfer-Bench/021_rmsnorm_h128.
# In real adapters, read it from the benchmark checkout instead:
#   Path(os.environ["SOLEXECBENCH_DIR"]) / "benchmark/<cat>/<p>/definition.json"
_REFERENCE_SRC = 'import torch\n\n@torch.no_grad()\ndef run(hidden_states, weight):\n    batch_size, hidden_size = hidden_states.shape\n    # Check constants\n    assert hidden_size == 128\n\n    EPS = 1e-6\n\n    x = hidden_states.to(torch.float32)\n    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)\n    y = (x * inv_rms) * weight.to(torch.float32)\n    return y.to(hidden_states.dtype)'
B, H = 8, 128


def get_source() -> str:
    return _REFERENCE_SRC


def get_inputs() -> dict:
    rng = np.random.RandomState(0)
    return {
        "hidden_states": rng.standard_normal((B, H)).astype(np.float32),
        "weight": (1.0 + 0.1 * rng.standard_normal(H)).astype(np.float32),
    }


def kernel_fn(inputs: dict) -> dict:
    import torch
    ns = {}
    exec(get_source(), ns)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in inputs.items()}
    out = ns["run"](t["hidden_states"], t["weight"])
    return {"output": out.numpy()}
