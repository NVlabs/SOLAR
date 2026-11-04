# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correct einsum_lib translation of fixtures/cuda_scale/adapter.py."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def scale_rows(X: Tensor("B,H", dtype="fp32"),
               W: Tensor("H", role="weight", dtype="fp32")) -> Tensor("B,H"):
    Y = Tensor(B=X.shape[0], H=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            Y[b, h] = X[b, h] * W[h] + 1.0
    return Y


def model(x, w):
    return scale_rows(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "H", role="weight", dtype="fp32")
    return {"output": model(x, w).data}
