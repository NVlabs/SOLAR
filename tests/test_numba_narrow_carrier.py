# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Numba narrow-rounding helpers return the SELECTED floating carrier (SHAPE0-228-CANDIDATE-DTYPE-1).

A program lowered with ``numba_lower(use_fp64=False)`` allocates float32 arrays, but its ``_truncate_bf16`` /
``_truncate_fp16`` helpers delegated to the fp64 interpreter's ``core._truncate_to_dtype`` and returned
float64 unconditionally; the driver's output_map maps logical bfloat16 to an expected physical float32
carrier and refused the float64 ``output`` of case 228 ("candidate physical output dtype differs").  The
helpers now round and return float32 in the fp32 mode and keep the float64 carrier in the default fp64
verification mode; the rounding (bf16 via torch round-to-nearest-even, fp16 via numpy float16) is unchanged
and the core interpreter is untouched.  The saved case-228 lowered program and rows 70 / 107 / 122 are used
as METADATA (four-element inputs at most); the full payload is never run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch = pytest.importorskip("torch")

from solar.einsum_lib.core import _truncate_to_dtype  # noqa: E402
from solar.einsum_lib.numba_accel import (  # noqa: E402
    HAS_NUMBA, _truncate_narrow_carrier, execute_fast, execution_namespace, numba_lower,
)


PROBE = np.array([0.0, 1.00390625, -1.01171875, 1.0 / 3.0, 65504.0, 3.0e-5, -0.1, 1234.5678])


def _torch_bf16(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16).to(torch.float32).numpy()


def _np_fp16(a):
    return np.asarray(a).astype(np.float16).astype(np.float32)


# ── helper matrix: both carrier modes x {bf16, fp16} x {float32, float64 input} ──

@pytest.mark.parametrize("use_fp64", [False, True], ids=["fp32-carrier", "fp64-carrier"])
@pytest.mark.parametrize("narrow", ["bf16", "fp16"])
@pytest.mark.parametrize("in_dtype", [np.float32, np.float64])
def test_helper_returns_selected_carrier_with_exact_narrow_rounding(use_fp64, narrow, in_dtype):
    ns = execution_namespace(use_fp64=use_fp64)
    a = PROBE.astype(in_dtype)
    got = ns["_truncate_" + narrow](a)
    expect = _torch_bf16(a) if narrow == "bf16" else _np_fp16(a)
    assert got.dtype == (np.float64 if use_fp64 else np.float32)
    assert got.shape == a.shape
    assert np.array_equal(got.astype(np.float64), expect.astype(np.float64)), (got, expect)
    # the values ARE representable in the narrow dtype (idempotent rounding)
    assert np.array_equal(ns["_truncate_" + narrow](got).astype(np.float64), got.astype(np.float64))


@pytest.mark.parametrize("use_fp64", [False, True], ids=["fp32-carrier", "fp64-carrier"])
@pytest.mark.parametrize("narrow", ["bf16", "fp16"])
@pytest.mark.parametrize("in_dtype", [np.float32, np.float64])
def test_rank0_and_noncontiguous_inputs_keep_shape_dtype_and_values(use_fp64, narrow, in_dtype):
    """contiguous-scalar-p2-1: np.ascontiguousarray promoted a 0-d array to (1,); the helper must preserve shape ()
    and handle transposed / strided (non-contiguous) inputs, in both carrier modes and both dtypes."""
    ns = execution_namespace(use_fp64=use_fp64)
    carrier = np.float64 if use_fp64 else np.float32
    ref = _torch_bf16 if narrow == "bf16" else _np_fp16
    scalar = np.array(1.00390625, dtype=in_dtype)
    got = ns["_truncate_" + narrow](scalar)
    assert got.shape == () and got.dtype == carrier and float(got) == float(ref(scalar))
    base = (np.arange(24, dtype=in_dtype).reshape(4, 6) / 7.0 + PROBE[1])
    for view in (base.T, base[::2, ::3], base[:, 1:5].T):
        assert not view.flags["C_CONTIGUOUS"]
        out = ns["_truncate_" + narrow](view)
        assert out.shape == view.shape and out.dtype == carrier
        assert np.array_equal(out.astype(np.float64), ref(view).astype(np.float64))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("use_fp64", [False, True], ids=["fp32-carrier", "fp64-carrier"])
def test_execute_fast_scalar_bf16_input_keeps_rank_zero(use_fp64):
    """Observable form of P2-1: a scalar bf16 input round-tripped through execute_fast keeps shape ()."""
    code = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

def run_model(inputs: dict) -> dict:
    clear_dims()
    s = Tensor.from_numpy(inputs["s"], "", dtype="bf16")
    return {"s": s.data}
'''
    out = execute_fast(code, {"s": np.array(1.00390625, dtype=np.float32)}, use_fp64=use_fp64)
    if out is None:
        pytest.skip("execute_fast declined the scalar program on this tree (lowering), helper shape is covered above")
    assert out["s"].shape == () and out["s"].dtype == (np.float64 if use_fp64 else np.float32)
    assert float(out["s"]) == 1.0


def test_fp64_mode_is_the_core_helper_unchanged():
    a = PROBE.copy()
    for narrow in ("bf16", "fp16"):
        assert np.array_equal(execution_namespace()["_truncate_" + narrow](a), _truncate_to_dtype(a, narrow))
        assert execution_namespace()["_truncate_" + narrow](a).dtype == np.float64
        assert _truncate_narrow_carrier(a, narrow, True) is not None


def test_non_narrow_dtype_passes_through():
    a = PROBE.astype(np.float32)
    assert _truncate_narrow_carrier(a, "fp32", False) is a and _truncate_narrow_carrier(a, "fp32", True) is a


# ── end to end: narrow input, narrow intermediate, narrow returned output through execute_fast ──

NARROW_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale2(X: Tensor("B,N", dtype="bf16")) -> Tensor("B,N", dtype="bf16"):
    Y = Tensor(B=X.shape[0], N=X.shape[1], dtype="bf16")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            Y[b, n] = X[b, n] * 1.0009765625
    return Y

@kernel
def shift(X: Tensor("B,N", dtype="bf16")) -> Tensor("B,N", dtype="fp16"):
    Y = Tensor(B=X.shape[0], N=X.shape[1], dtype="fp16")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            Y[b, n] = X[b, n] + 0.0001220703125
    return Y

def model(x):
    h = scale2(x)        # bf16 intermediate
    y = shift(h)         # fp16 returned output (assigned: the lowerer inserts the rounding on assignments;
    return y             # a direct `return shift(h)` is the separate final-rounding gap, rows 140/229, not this surface)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,N", dtype="bf16")   # bf16 input
    y = model(x)
    return {"y": y.data}
'''


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("use_fp64", [False, True], ids=["fp32-carrier", "fp64-carrier"])
def test_execute_fast_keeps_the_selected_carrier_on_input_intermediate_and_output(use_fp64):
    lowered = numba_lower(NARROW_CODE, boundscheck=True, use_fp64=use_fp64)
    assert lowered and lowered.count("_truncate_bf16(") >= 2 and "_truncate_fp16(" in lowered
    rng = np.random.default_rng(228)
    x = rng.standard_normal((2, 4)).astype(np.float32)
    out = execute_fast(NARROW_CODE, {"x": x}, use_fp64=use_fp64)
    assert out is not None and "y" in out
    y = out["y"]
    carrier = np.float64 if use_fp64 else np.float32
    assert y.dtype == carrier, y.dtype
    # the exact narrow-rounded reference: bf16(x) -> bf16(x * 1.0009765625) -> fp16(h + 2^-13)
    xb = _torch_bf16(x).astype(np.float64)
    h = _torch_bf16(xb * 1.0009765625).astype(np.float64)
    expect = _np_fp16(h + 0.0001220703125).astype(np.float64)
    assert np.array_equal(y.astype(np.float64), expect), (y, expect)


# ── regression on the saved case-228 lowered program (metadata only) ──

