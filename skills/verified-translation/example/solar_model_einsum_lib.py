# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""einsum_lib translation of FlashInfer-Bench/021_rmsnorm_h128 (manual, no-LLM).

RMSNorm over H: y[b,h] = x[b,h] / sqrt(mean_h(x^2) + eps) * w[h].
Dims: B (batch tokens), H=128. eps hardcoded 1e-6 (matches reference).
"""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_EPS = 1e-6


@kernel
def row_sumsq(X: Tensor("B,H", dtype="fp32")) -> Tensor("B"):
    Y = Tensor(B=X.shape[0], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            Y[b] += X[b, h] * X[b, h]
    return Y


@kernel
def rmsnorm_apply(X: Tensor("B,H", dtype="fp32"),
                  SS: Tensor("B", dtype="fp32"),
                  W: Tensor("H", role="weight", dtype="fp32"),
                  eps: float) -> Tensor("B,H"):
    Y = Tensor(B=X.shape[0], H=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            Y[b, h] = W[h] * X[b, h] / sqrt(SS[b] / X.shape[1] + eps)
    return Y


def model(hidden_states, weight):
    ss = row_sumsq(hidden_states)
    return rmsnorm_apply(hidden_states, ss, weight, _EPS)


def run_model(inputs: dict) -> dict:
    clear_dims()
    hs = Tensor.from_numpy(inputs["hidden_states"], "B,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["weight"], "H", role="weight", dtype="fp32")
    out = model(hs, w)
    return {"output": out.data}
