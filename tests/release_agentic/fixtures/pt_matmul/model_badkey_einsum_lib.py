# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structurally fine translation that returns the WRONG output key name.

Pins the documented contract that outputs are compared BY KEY: a correct
computation under the key "y" instead of "output" must not verify.
"""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def matmul(X: Tensor("M,K", dtype="fp32"),
           W: Tensor("K,N", role="weight", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m, n] += X[m, k] * W[k, n]
    return Y


def model(x, w):
    return matmul(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight", dtype="fp32")
    return {"y": model(x, w).data}
