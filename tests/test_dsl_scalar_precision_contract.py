# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSL-SIGMOID-FP32-BOUNDARY-1 — the DSL's scalar precision contract, pinned (two modes, no behaviour change).

**fp64 emulation** — the interpreter and the ``use_fp64=True`` Numba lowering: floating tensors are float64
(``core._numpy_dtype``), scalar math runs in double, and the framework rounds ONLY ``bf16`` / ``fp16`` results
(``core._truncate_to_dtype``: "fp32/fp64 are left untouched"); an ``fp32`` tag is a storage-width label with no
compute effect.  This is the ACCEPTED validation precision (Jenny, 2026-10-26: emulation may be in fp64; an NVFP4
candidate is validated in fp64 after dequantization and then carries its declared nvfp4 types into the SOL/perf
projection).  **Native fp32-carrier lowering** — ``use_fp64=False`` (the ``execute_translated`` default fast
path): float32 buffers and compiled scalar ops that follow their operand types (operand-driven promotion); a
separate mode, described and guarded as such (native-precision-1).

fp32 fidelity to a torch fp32 op chain is the AUTHOR's job, through the grid-snap idiom applied after EVERY
inexact fp32 op of the reference (skills/verified-translation/reference.md).  That idiom is rounding-boundary
EMULATION, not a bit-exactness guarantee (snap-fidelity-2): it agrees exactly with torch's fp32 sigmoid on the
[10, 20] boundary sweep used here, but torch's transcendental implementations are not correctly rounded everywhere,
so other domains must be compared against the actual reference/runtime.  A single snap at the end of a multi-op
chain is not the idiom: for sigmoid(17), ``1 + exp(-17)`` rounds to exactly 1.0 in fp32 (torch returns 1.0) while
``fp32(1 / (1 + exp(-17)))`` computed in double lands on the grid point below 1.0 (0.9999999403953552); the
single-snap form first saturates at logit 17.329, the per-op form and torch at 16.636 — a one-ulp difference on
~40% of the sweep that changes top-k tie-plateau membership (row 203).

Nothing here is a qualification gate: the "up to ten differing torch points" allowance is a host-dependence guard
for vectorised ``expf`` and waives nothing in an earlier repair and accepts no row.
"""
import math

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import execute_fast, HAS_NUMBA
from solar.einsum_lib.verify import execute_translated

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    stop_trace(); clear_dims()


HEADER = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nfrom math import exp, floor, log2\n"
RUN = "\ndef run_model(inputs):\n    clear_dims()\n    l = Tensor.from_numpy(inputs['l'], 'T,E', dtype='fp32')\n    return {'output': k(l).data}\n"

# the row-203 form: double expression, ONE snap to the fp32 grid at the end
SINGLE_SNAP = HEADER + '''
@kernel
def k(L: Tensor("T,E", dtype="fp32")) -> Tensor("T,E"):
    Y = Tensor(T=L.shape[0], E=L.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            s = 1.0 / (1.0 + exp(-L[t, e]))
            e1 = max(floor(log2(max(abs(s), 1e-300))), -126.0)
            Y[t, e] = round(s / 2.0 ** (e1 - 23.0)) * 2.0 ** (e1 - 23.0)
    return Y
''' + RUN
# the documented idiom (row 229's form): snap after EVERY inexact fp32 op of the reference chain
PER_OP_SNAP = HEADER + '''
@kernel
def k(L: Tensor("T,E", dtype="fp32")) -> Tensor("T,E"):
    Y = Tensor(T=L.shape[0], E=L.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            x = exp(-L[t, e])
            ex = max(floor(log2(max(abs(x), 1e-300))), -126.0)
            x = round(x / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)
            x = 1.0 + x
            ex = max(floor(log2(max(abs(x), 1e-300))), -126.0)
            x = round(x / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)
            x = 1.0 / x
            ex = max(floor(log2(max(abs(x), 1e-300))), -126.0)
            Y[t, e] = round(x / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)
    return Y
''' + RUN
# no snap at all: what the framework itself does with an fp32-tagged result
NO_SNAP = HEADER + '''
@kernel
def k(L: Tensor("T,E", dtype="fp32")) -> Tensor("T,E", dtype="fp32"):
    Y = Tensor(T=L.shape[0], E=L.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            Y[t, e] = 1.0 / (1.0 + exp(-L[t, e]))
    return Y
''' + RUN

SWEEP = np.round(np.arange(10.0, 20.0 + 1e-9, 1e-3), 3)          # 10,001 logits across the fp32 saturation boundary
BOUNDARY = np.array([17.0, 16.6356, 16.8])                        # the reviewer's points


def _ulps(u, v):
    a = np.asarray(u, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(v, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def _run(code, logits, fast, use_fp64=True):
    x = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    if fast:
        out = execute_fast(code, {"l": x}, use_fp64=use_fp64)
        assert out is not None
        return np.asarray(out["output"], dtype=np.float64).reshape(-1)
    out, _ = execute_translated(code, {"l": x.shape}, all_inputs={"l": x}, fast=False)
    return np.asarray(out["output"], dtype=np.float64).reshape(-1)


def _torch_fp32_sigmoid(logits):
    import torch
    return torch.sigmoid(torch.from_numpy(np.asarray(logits, dtype=np.float32))).to(torch.float64).numpy()


class TestFrameworkRoundsOnlyNarrowDtypes:
    """fp64 emulation: an ``fp32`` tag rounds nothing — the kernel's double value comes back unrounded from the
    interpreter and the ``use_fp64=True`` lowering."""

    def test_interpreter_keeps_the_double(self):
        v = _run(NO_SNAP, [17.0], fast=False)[0]
        assert v == 1.0 / (1.0 + math.exp(-17.0)) and v != np.float64(np.float32(v))

    @needs_numba
    def test_lowering_fp64_keeps_the_double_and_both_paths_agree_on_the_sweep(self):
        slow = _run(NO_SNAP, SWEEP, fast=False); fast = _run(NO_SNAP, SWEEP, fast=True, use_fp64=True)
        np.testing.assert_array_equal(slow, fast)
        assert fast[SWEEP == 17.0][0] == 1.0 / (1.0 + math.exp(-17.0))


NO_SNAP_SUM = HEADER + '''
@kernel
def k(L: Tensor("T,E", dtype="fp32")) -> Tensor("T,E", dtype="fp32"):
    Y = Tensor(T=L.shape[0], E=L.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            Y[t, e] = (L[t, 0] + L[t, 1]) - L[t, 0]
    return Y
''' + RUN


class TestTwoExecutionModes:
    """native-precision-1: the mode distinction, recorded.  ``(X[0] + X[1]) - X[0]`` with ``X = [2**24, 1]``
    is 1 wherever the intermediate sum is held in double and 0 where it is held in float32 (2**24 + 1 is not
    representable in float32; a final float32 store of an exact 1 would still be 1, so this distinguishes
    intermediate arithmetic from the output cast)."""

    X = np.array([[2.0 ** 24, 1.0]], dtype=np.float64)          # one row, two columns: L[t, 0] + L[t, 1] - L[t, 0]

    @staticmethod
    def _value(fast, use_fp64=True):
        inputs = {"l": TestTwoExecutionModes.X}
        if fast:
            out = execute_fast(NO_SNAP_SUM, inputs, use_fp64=use_fp64); assert out is not None
        else:
            out, _ = execute_translated(NO_SNAP_SUM, {"l": (1, 2)}, all_inputs=inputs, fast=False)
        return float(np.asarray(out["output"], dtype=np.float64).reshape(-1)[0])

    def test_interpreter_holds_intermediates_in_double(self):
        assert self._value(fast=False) == 1.0

    @needs_numba
    def test_fp64_lowering_holds_intermediates_in_double(self):
        assert self._value(fast=True, use_fp64=True) == 1.0

    @needs_numba
    def test_native_fp32_carrier_lowering_promotes_by_operand_type(self):
        """The ``execute_translated`` default fast path: float32 buffers, so the sum rounds to 2**24 and the result
        is 0 — a separate mode from the fp64 emulation, described as such, behaviour unchanged."""
        assert self._value(fast=True, use_fp64=False) == 0.0
        import inspect
        assert inspect.signature(execute_translated).parameters["use_fp64"].default is False


class TestPerOpSnapIsFp32Faithful:
    """The documented idiom EMULATES torch's fp32 rounding boundaries: on the [10, 20] sweep the saturation
    boundary is at the same logit (16.636) and the sweep agrees exactly on this host, through the interpreter and
    both lowering carriers.  This is observed agreement on the stated domain, not a universal bit-exactness
    guarantee (snap-fidelity-2: up to 3 ulps off a literal torch fp32 step chain on parts of [-20, 10])."""

    @pytest.mark.parametrize("fast", [False, pytest.param(True, marks=needs_numba)], ids=["interpreter", "lowering"])
    def test_boundary_points_saturate_like_torch(self, fast):
        got = _run(PER_OP_SNAP, BOUNDARY, fast=fast)
        np.testing.assert_array_equal(got, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(got, _torch_fp32_sigmoid(BOUNDARY))

    @pytest.mark.parametrize("fast", [False, pytest.param(True, marks=needs_numba)], ids=["interpreter", "lowering"])
    def test_sweep_matches_torch_fp32(self, fast):
        got = _run(PER_OP_SNAP, SWEEP, fast=fast)
        ref = _torch_fp32_sigmoid(SWEEP)
        differing = int((_ulps(got, ref) > 0).sum())
        # exact on this host (torch 2.7.1 CPU); the <= 10 allowance is a HOST-DEPENDENCE guard for a vectorised expf
        # elsewhere -- it is not an exact-output qualification gate and waives nothing in an earlier repair
        assert differing <= 10, differing
        np.testing.assert_array_equal(got, np.float32(1.0) / (np.float32(1.0) + np.exp(-SWEEP.astype(np.float32))))

    @needs_numba
    def test_fp32_carrier_lowering_agrees(self):
        np.testing.assert_array_equal(_run(PER_OP_SNAP, SWEEP, fast=True, use_fp64=False), _run(PER_OP_SNAP, SWEEP, fast=False))


class TestSingleSnapIsNotFp32:
    """The row-203 form: one snap of the double result differs from the per-op idiom and torch fp32 by one ulp on
    a large share of the sweep and sits one grid point below 1.0 where torch already saturates (it saturates later,
    first at logit 17.329 against 16.636) — in both execution paths identically."""

    @pytest.mark.parametrize("fast", [False, pytest.param(True, marks=needs_numba)], ids=["interpreter", "lowering"])
    def test_boundary_points_do_not_saturate(self, fast):
        got = _run(SINGLE_SNAP, BOUNDARY, fast=fast)
        np.testing.assert_array_equal(got, [0.9999999403953552] * 3)
        assert (_ulps(got, _torch_fp32_sigmoid(BOUNDARY)) == 1).all()

    @pytest.mark.parametrize("fast", [False, pytest.param(True, marks=needs_numba)], ids=["interpreter", "lowering"])
    def test_sweep_differs_from_the_per_op_idiom_by_one_ulp_often(self, fast):
        single = _run(SINGLE_SNAP, SWEEP, fast=fast); per_op = _run(PER_OP_SNAP, SWEEP, fast=fast)
        d = _ulps(single, per_op)
        assert d.max() == 1 and int((d > 0).sum()) >= 3500          # 4,050 of 10,001 on this host

    @needs_numba
    def test_both_paths_agree_on_the_single_snap_form(self):
        np.testing.assert_array_equal(_run(SINGLE_SNAP, SWEEP, fast=False), _run(SINGLE_SNAP, SWEEP, fast=True))
