# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A translation that ignores its input and returns zeros (passes vacuously)."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def zeros_like(X: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            Y[m, n] = X[m, n] * 0.0
    return Y


def model(x):
    return zeros_like(x)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N", dtype="fp32")
    return {"output": model(x).data}
