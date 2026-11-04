# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSL's ``fp32(x)`` rounding-boundary cast (snap-mac-1 / DSL-SIGMOID-FP32-BOUNDARY-1).

``fp32(x)`` rounds a scalar to the nearest float32 value (round-to-nearest-even, subnormals included, overflow to
inf) and hands it back as a double — the same value in the interpreter, the ``use_fp64=True`` lowering and the
native fp32-carrier lowering, where it lowers to ``np.float64(np.float32(x))``.  It is a CAST for the DSL
classifier: a reduction loop that only snaps its terms is an exp/add reduction (no MACs), unlike the inline grid
idiom ``round(x / 2**k) * 2**k`` whose ``*``/``**`` make the same loop look like a multiply-accumulate.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_sparse_cost import _traced_metadata, _analyze  # noqa: E402

from solar.einsum_lib import clear_dims, stop_trace, fp32
from solar.einsum_lib.numba_accel import HAS_NUMBA, execute_fast, numba_lower
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.einsum_converter import _infer_ops

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    stop_trace(); clear_dims()


HEADER = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nfrom math import exp\n"
CAST_KERNEL = HEADER + '''
@kernel
def k(L: Tensor("T,E", dtype="fp32")) -> Tensor("T,E", dtype="fp32"):
    Y = Tensor(T=L.shape[0], E=L.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            Y[t, e] = fp32(L[t, e])
    return Y

def run_model(inputs):
    clear_dims()
    l = Tensor.from_numpy(inputs['l'], 'T,E', dtype='fp32')
    return {'output': k(l).data}
'''
# values across the float32 landscape: exact, inexact, subnormal, overflow, negative zero, huge negative
VALUES = np.array([[1.003, 1.0 + 2.0 ** -24, 2.0 ** 24 + 1.0, 2.9e-39, 1.4e-45, 7e-46, 1e40, -1e40, -0.0, 16.6356, -88.7, 3.4028235e38]], dtype=np.float64)


def _run(code, x, fast, use_fp64=True):
    if fast:
        out = execute_fast(code, {"l": x}, use_fp64=use_fp64); assert out is not None
        return np.asarray(out["output"], dtype=np.float64)
    out, _ = execute_translated(code, {"l": x.shape}, all_inputs={"l": x}, fast=False)
    return np.asarray(out["output"], dtype=np.float64)


class TestCastSemantics:
    def test_helper_is_the_float32_round_trip(self):
        ref = VALUES.astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(np.array([[fp32(v) for v in VALUES[0]]]), ref)
        np.testing.assert_array_equal(fp32(VALUES), ref)
        assert fp32(1e40) == np.inf and fp32(-1e40) == -np.inf and fp32(7e-46) == 0.0 and fp32(2.9e-39) == float(np.float32(2.9e-39))

    def test_interpreter_matches_the_round_trip(self):
        np.testing.assert_array_equal(_run(CAST_KERNEL, VALUES, fast=False), VALUES.astype(np.float32).astype(np.float64))

    @needs_numba
    @pytest.mark.parametrize("use_fp64", [True, False], ids=["fp64-lowering", "native-fp32-carrier"])
    def test_lowering_matches_in_both_modes(self, use_fp64):
        np.testing.assert_array_equal(_run(CAST_KERNEL, VALUES, fast=True, use_fp64=use_fp64), VALUES.astype(np.float32).astype(np.float64))

    @needs_numba
    def test_lowers_to_a_float32_round_trip_not_arithmetic(self):
        low = numba_lower(CAST_KERNEL, boundscheck=True, use_fp64=False)
        assert "np.float64(np.float32(L[t, e]))" in low and "fp32(" not in low


# a two-input exp reduction whose terms are snapped: the row-209 softmax_expsum form
SNAP_ONLY_REDUCTION = HEADER + '''
@kernel
def rowmax(L: Tensor("T,E", dtype="fp32")) -> Tensor("T"):
    Y = Tensor(T=L.shape[0], dtype="fp32", fill=-1e30)
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            Y[t] = max(Y[t], L[t, e])
    return Y

@kernel
def expsum(L: Tensor("T,E", dtype="fp32"), RM: Tensor("T", dtype="fp32")) -> Tensor("T"):
    Y = Tensor(T=L.shape[0], dtype="fp32", fill=0.0)
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            Y[t] += fp32(exp(fp32(L[t, e] - RM[t])))
    return Y

def run_model(inputs):
    clear_dims()
    l = Tensor.from_numpy(inputs['l'], 'T,E', dtype='fp32')
    return {'output': expsum(l, rowmax(l)).data}
'''
INLINE_GRID_REDUCTION = SNAP_ONLY_REDUCTION.replace(
    "            Y[t] += fp32(exp(fp32(L[t, e] - RM[t])))\n",
    "            d = L[t, e] - RM[t]\n            ex = max(floor(log2(max(abs(d), 1e-300))), -126.0)\n            d = round(d / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)\n            x = exp(d)\n            ex = max(floor(log2(max(abs(x), 1e-300))), -126.0)\n            Y[t] += round(x / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)\n").replace("from math import exp\n", "from math import exp, floor, log2\n")


class TestClassifierRegression:
    def test_infer_ops_treats_the_cast_as_invisible(self):
        assert _infer_ops("Y[t] += fp32(exp(fp32(L[t, e] - RM[t])))") == ("exp", "add")
        assert _infer_ops("Y[t] += exp(L[t, e] - RM[t])") == ("exp", "add")
        assert _infer_ops("Y[t, e] = fp32(fp32(exp(fp32(L[t, e] - RM[t]))) / fp32(SU[t]))") == ("exp", "none")  # no multiply, no reduction
        assert _infer_ops("Y[t] += round(x / 2.0 ** (ex - 23.0)) * 2.0 ** (ex - 23.0)") == ("mul", "add")        # the inline idiom reads as mul+add

    def test_snap_only_reduction_is_not_billed_as_an_einsum(self, tmp_path):
        x = np.random.RandomState(0).randn(4, 8).astype(np.float64)
        md = _traced_metadata(SNAP_ONLY_REDUCTION, {"l": x}, profile=False)
        a = _analyze(SNAP_ONLY_REDUCTION, md, tmp_path)
        import yaml
        g = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())["layers"]["expsum#0"]
        assert g["is_real_einsum"] is False and g["elementwise_op"] == "exp" and g["reduction_op"] == "add"
        assert a["layers"]["expsum#0"]["macs"] == 0

    def test_inline_grid_arithmetic_is_what_the_classifier_misreads(self, tmp_path):
        """The round-1 form (snap-mac-1): the same reduction with the inline ``round(x/2**k)*2**k`` snap is
        classified mul+add over two inputs and billed as a real einsum — the hazard fp32() removes."""
        x = np.random.RandomState(0).randn(4, 8).astype(np.float64)
        md = _traced_metadata(INLINE_GRID_REDUCTION, {"l": x}, profile=False)
        a = _analyze(INLINE_GRID_REDUCTION, md, tmp_path)
        import yaml
        g = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())["layers"]["expsum#0"]
        assert g["is_real_einsum"] is True and g["elementwise_op"] == "mul"
        assert a["layers"]["expsum#0"]["macs"] > 0
