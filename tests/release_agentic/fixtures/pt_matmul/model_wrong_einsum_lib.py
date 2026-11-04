# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliberately WRONG translation of fixtures/pt_matmul/model.py.

Same shapes and key as the correct DSL; the math drops the last k term, so
it is structurally valid (executes, right output shape) but numerically
wrong.  Used to pin MISMATCH behavior on every route.
"""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def matmul_truncated(X: Tensor("M,K", dtype="fp32"),
                     W: Tensor("K,N", role="weight", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m, n] += X[m, k] * W[k, n] * 0.5
    return Y


def model(x, w):
    return matmul_truncated(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight", dtype="fp32")
    return {"output": model(x, w).data}
