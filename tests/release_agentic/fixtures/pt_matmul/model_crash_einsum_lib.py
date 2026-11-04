# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translation that fails at execution (unregistered loop dim)."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def matmul(X: Tensor("M,K", dtype="fp32"),
           W: Tensor("K,N", role="weight", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("NOT_A_DIM"):
                Y[m, n] += X[m, k] * W[k, n]
    return Y


def model(x, w):
    return matmul(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight", dtype="fp32")
    return {"output": model(x, w).data}
