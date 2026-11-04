# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""solar.einsum_lib — executable loop-nest representation with named dimensions.

Write kernels as pure Python nested loops with named Tensor dimensions.
The resulting code is both executable (for verification) and mechanically
compilable to einsum equations.

    from solar.einsum_lib import Tensor, Dim, kernel

    @kernel
    def matmul(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")) -> Tensor("B,M,N"):
        C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
        for b in Dim("B"):
            for m in Dim("M"):
                for n in Dim("N"):
                    for k in Dim("K"):
                        C[b,m,n] += A[b,m,k] * W[k,n]
        return C
"""

from solar.einsum_lib.core import (
    Tensor, Dim, clear_dims, kernel, apply_mask, top_k_mask, VALID_DTYPES,
    start_trace, stop_trace, get_trace,
    slice_at, stack, concat, gather,
    set_density_profiling, density_profiling_enabled,
    fp32,
)
from solar.einsum_lib.metadata import build_metadata, write_metadata_yaml
from solar.einsum_lib.agent import TranslationAgent, TranslationResult, TimingInfo
from solar.einsum_lib.verify import VerificationResult, CapturedRun, capture_model_run
from solar.einsum_lib.numba_accel import numba_lower, execute_fast, HAS_NUMBA

__all__ = [
    "Tensor",
    "Dim",
    "clear_dims",
    "kernel",
    "apply_mask",
    "top_k_mask",
    "slice_at",
    "stack",
    "concat",
    "gather",
    "fp32",
    "VALID_DTYPES",
    "start_trace",
    "stop_trace",
    "get_trace",
    "set_density_profiling",
    "density_profiling_enabled",
    "build_metadata",
    "write_metadata_yaml",
    "TranslationAgent",
    "TranslationResult",
    "TimingInfo",
    "VerificationResult",
    "CapturedRun",
    "capture_model_run",
    "numba_lower",
    "execute_fast",
    "HAS_NUMBA",
]
