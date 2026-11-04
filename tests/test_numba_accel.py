# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Numba JIT acceleration and dummy trace mode."""

import numpy as np
import pytest

from solar.einsum_lib import (
    Tensor, Dim, clear_dims, kernel, apply_mask,
    start_trace, stop_trace, get_trace,
)
from solar.einsum_lib.metadata import build_metadata
from solar.einsum_lib.numba_accel import numba_lower, execute_fast, HAS_NUMBA
from solar.einsum_lib.verify import execute_translated


@pytest.fixture(autouse=True)
def _clean_dims():
    clear_dims()
    stop_trace()
    yield
    stop_trace()
    clear_dims()


# ---------------------------------------------------------------------------
# Test code snippets (matching LLM-generated patterns)
# ---------------------------------------------------------------------------

MATMUL_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("B,M,K", dtype="fp32"), W: Tensor("K,N", role="weight", dtype="fp32")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("K"):
                    C[b, m, n] += A[b, m, k] * W[k, n]
    return C

def model(x, w):
    return matmul(x, w)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["weight"], "K,N", role="weight", dtype="fp32")
    output = model(x, w)
    return {"output": output.data}
'''

SOFTMAX_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def my_softmax(X: Tensor("B,N", dtype="fp32")) -> Tensor("B,N"):
    MaxX = Tensor(B=X.shape[0], dtype="fp32", fill=float("-inf"))
    E = Tensor(B=X.shape[0], N=X.shape[1], dtype="fp32")
    SumE = Tensor(B=X.shape[0], dtype="fp32")
    P = Tensor(B=X.shape[0], N=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            MaxX[b] = max(MaxX[b], X[b, n])
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            E[b, n] = exp(X[b, n] - MaxX[b])
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            SumE[b] += E[b, n]
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            P[b, n] = E[b, n] / SumE[b]
    return P

def model(x):
    return my_softmax(x)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,N", dtype="fp32")
    output = model(x)
    return {"output": output.data}
'''

CONV2D_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def conv2d(X: Tensor("B,C,H,W", dtype="fp32"), W: Tensor("O,C,R,S", role="weight", dtype="fp32")) -> Tensor("B,O,P,Q"):
    Y = Tensor(B=X.shape[0], O=W.shape[0], P=X.shape[2]-W.shape[2]+1, Q=X.shape[3]-W.shape[3]+1, dtype="fp32")
    for b in Dim("B", parallel=True):
        for o in Dim("O"):
            for p in Dim("P"):
                for q in Dim("Q"):
                    for c in Dim("C"):
                        for r in Dim("R"):
                            for s in Dim("S"):
                                Y[b,o,p,q] += X[b,c,p+r,q+s] * W[o,c,r,s]
    return Y

def model(x, w):
    return conv2d(x, w)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,C,H,W", dtype="fp32")
    w = Tensor.from_numpy(inputs["conv_weight"], "O,C,R,S", role="weight", dtype="fp32")
    output = model(x, w)
    return {"output": output.data}
'''

APPLY_MASK_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask

@kernel
def scale(X: Tensor("B,M,N", dtype="fp32")) -> Tensor("B,M,N"):
    Y = Tensor(B=X.shape[0], M=X.shape[1], N=X.shape[2], dtype="fp32")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                Y[b, m, n] = X[b, m, n] * 0.5
    return Y

def model(x, mask):
    s = scale(x)
    return apply_mask(s, mask, masked_value=-1e9)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,N", dtype="fp32")
    mask = Tensor.from_numpy(inputs["mask"], "M,N", dtype="fp32", sparsity=0.5)
    output = model(x, mask)
    return {"output": output.data}
'''

RELU_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def relu(A: Tensor("B,M,N", dtype="fp32")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=A.shape[2], dtype="fp32")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                C[b, m, n] = max(0, A[b, m, n])
    return C

def model(x):
    return relu(x)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,N", dtype="fp32")
    output = model(x)
    return {"output": output.data}
'''

SCALE_SQRT_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale_scores(X: Tensor("B,M,N", dtype="fp32")) -> Tensor("B,M,N"):
    scale = 1.0 / sqrt(X.shape[2])
    Y = Tensor(B=X.shape[0], M=X.shape[1], N=X.shape[2], dtype="fp32")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                Y[b, m, n] = X[b, m, n] * scale
    return Y

def model(x):
    return scale_scores(x)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,N", dtype="fp32")
    output = model(x)
    return {"output": output.data}
'''

GROUPED_MATMUL_CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def grouped_matmul(A: Tensor("B,C,M,K", dtype="fp32"),
                   W: Tensor("C,K,N", role="weight", dtype="fp32")) -> Tensor("B,C,M,N"):
    groups = 2
    cpg = A.shape[1] // groups
    Y = Tensor(B=A.shape[0], C=A.shape[1], M=A.shape[2], N=W.shape[2], dtype="fp32")
    for b in Dim("B", parallel=True):
        for g in Dim("G", size=groups):
            for cg in Dim("CG", size=cpg):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            Y[b, g * cpg + cg, m, n] += A[b, g * cpg + cg, m, k] * W[g * cpg + cg, k, n]
    return Y

def model(x, w):
    return grouped_matmul(x, w)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,C,M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["weight"], "C,K,N", role="weight", dtype="fp32")
    output = model(x, w)
    return {"output": output.data}
'''


# ---------------------------------------------------------------------------
# numba_lower — AST transform produces valid code
# ---------------------------------------------------------------------------

class TestNumbaLower:
    """Test that numba_lower produces parseable, correct code."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_matmul_lowered_is_string(self):
        result = numba_lower(MATMUL_CODE)
        assert result is not None
        assert isinstance(result, str)
        assert "numba.njit" in result
        assert "numba.prange" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_has_no_einsum_lib_imports(self):
        result = numba_lower(MATMUL_CODE)
        assert "from solar.einsum_lib" not in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_replaces_tensor_with_np_zeros(self):
        result = numba_lower(MATMUL_CODE)
        assert "np.zeros" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_fill_uses_np_full(self):
        result = numba_lower(SOFTMAX_CODE)
        assert "np.full" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_exp_becomes_math_exp(self):
        result = numba_lower(SOFTMAX_CODE)
        assert "math.exp" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_sqrt_becomes_math_sqrt(self):
        result = numba_lower(SCALE_SQRT_CODE)
        assert "math.sqrt" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_removes_clear_dims(self):
        result = numba_lower(MATMUL_CODE)
        assert "clear_dims" not in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_removes_tensor_from_numpy(self):
        result = numba_lower(MATMUL_CODE)
        assert "Tensor.from_numpy" not in result
        assert "np.asarray" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_lowered_apply_mask_becomes_np_where(self):
        result = numba_lower(APPLY_MASK_CODE)
        assert "np.where" in result
        assert "apply_mask" not in result

    def test_returns_none_without_numba(self):
        import solar.einsum_lib.numba_accel as mod
        orig = mod.HAS_NUMBA
        try:
            mod.HAS_NUMBA = False
            assert numba_lower(MATMUL_CODE) is None
        finally:
            mod.HAS_NUMBA = orig

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_prange_only_for_parallel_dims(self):
        result = numba_lower(MATMUL_CODE)
        assert result.count("numba.prange") == 1, "only B is marked parallel"
        assert "range(" in result, "inner dims should use range()"

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_no_prange_without_parallel_flag(self):
        code_no_parallel = MATMUL_CODE.replace(
            'Dim("B", parallel=True)', 'Dim("B")'
        )
        result = numba_lower(code_no_parallel)
        assert "numba.prange" not in result, "no parallel flag means no prange"

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_softmax_prange_count(self):
        result = numba_lower(SOFTMAX_CODE)
        assert result.count("numba.prange") == 4, "4 loop nests each with parallel B"

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_dim_size_lowered_to_range(self):
        result = numba_lower(GROUPED_MATMUL_CODE)
        assert result is not None
        assert "range(groups)" in result or "range(2)" in result

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_dim_size_no_prange_for_non_parallel(self):
        result = numba_lower(GROUPED_MATMUL_CODE)
        assert result.count("numba.prange") == 1, "only B is parallel"

    def test_returns_none_on_syntax_error(self):
        assert numba_lower("def foo(:\n  pass") is None


# ---------------------------------------------------------------------------
# Dim(size=) — explicit loop bound registration
# ---------------------------------------------------------------------------

class TestDimSize:
    """Dim(size=) registers the dimension and iterates correctly."""

    def test_dim_size_registers_and_iterates(self):
        vals = list(Dim("TEST_G", size=4))
        assert vals == [0, 1, 2, 3]

    def test_dim_size_with_parallel(self):
        vals = list(Dim("TEST_P", size=3, parallel=True))
        assert vals == [0, 1, 2]

    def test_dim_size_same_as_existing_ok(self):
        Tensor(TEST_X=3)
        vals = list(Dim("TEST_X", size=3))
        assert vals == [0, 1, 2]

    def test_dim_size_rebinds(self):
        Tensor(TEST_Y=10)
        assert list(Dim("TEST_Y", size=3)) == [0, 1, 2]

    def test_dim_size_zero(self):
        vals = list(Dim("TEST_Z", size=0))
        assert vals == []


# ---------------------------------------------------------------------------
# execute_fast — JIT output matches slow path
# ---------------------------------------------------------------------------

class TestExecuteFastMatmul:
    """Numba-accelerated matmul matches pure-Python einsum_lib."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_matmul_matches_slow(self):
        rng = np.random.RandomState(42)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (2, 4, 6)).astype(np.float64),
            "weight": rng.uniform(-0.1, 0.1, (6, 8)).astype(np.float64),
        }

        slow_out, _ = execute_translated(
            MATMUL_CODE, {"x": (2, 4, 6), "weight": (6, 8)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(MATMUL_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastSoftmax:
    """Numba-accelerated softmax matches pure-Python path."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_softmax_matches_slow(self):
        rng = np.random.RandomState(42)
        inputs = {"x": rng.uniform(-1, 1, (3, 8)).astype(np.float64)}

        slow_out, _ = execute_translated(
            SOFTMAX_CODE, {"x": (3, 8)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(SOFTMAX_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastConv2d:
    """Numba-accelerated conv2d matches pure-Python path."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_conv2d_matches_slow(self):
        rng = np.random.RandomState(42)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (1, 3, 8, 8)).astype(np.float64),
            "conv_weight": rng.uniform(-0.1, 0.1, (4, 3, 3, 3)).astype(np.float64),
        }

        slow_out, _ = execute_translated(
            CONV2D_CODE,
            {"x": (1, 3, 8, 8), "conv_weight": (4, 3, 3, 3)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(CONV2D_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastApplyMask:
    """Numba-accelerated apply_mask matches pure-Python path."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_apply_mask_matches_slow(self):
        rng = np.random.RandomState(42)
        mask = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.float64)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (2, 4, 4)).astype(np.float64),
            "mask": mask,
        }

        slow_out, _ = execute_translated(
            APPLY_MASK_CODE,
            {"x": (2, 4, 4), "mask": (4, 4)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(APPLY_MASK_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastRelu:
    """Numba-accelerated relu matches pure-Python path."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_relu_matches_slow(self):
        rng = np.random.RandomState(42)
        inputs = {"x": rng.uniform(-1, 1, (2, 4, 8)).astype(np.float64)}

        slow_out, _ = execute_translated(
            RELU_CODE, {"x": (2, 4, 8)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(RELU_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastGrouped:
    """Numba-accelerated grouped matmul (using Dim size=) matches pure-Python path."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_grouped_matmul_matches_slow(self):
        rng = np.random.RandomState(42)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (2, 4, 3, 6)).astype(np.float64),
            "weight": rng.uniform(-0.1, 0.1, (4, 6, 5)).astype(np.float64),
        }

        slow_out, _ = execute_translated(
            GROUPED_MATMUL_CODE,
            {"x": (2, 4, 3, 6), "weight": (4, 6, 5)},
            all_inputs=inputs, fast=False,
        )
        fast_out = execute_fast(GROUPED_MATMUL_CODE, inputs)

        assert fast_out is not None
        np.testing.assert_allclose(
            fast_out["output"], slow_out["output"], rtol=1e-10,
        )


class TestExecuteFastFallback:
    """Fallback when Numba is not available."""

    def test_execute_fast_returns_none_without_numba(self):
        import solar.einsum_lib.numba_accel as mod
        orig = mod.HAS_NUMBA
        try:
            mod.HAS_NUMBA = False
            result = execute_fast(MATMUL_CODE, {"x": np.ones((2, 3, 4))})
            assert result is None
        finally:
            mod.HAS_NUMBA = orig


# ---------------------------------------------------------------------------
# execute_translated with fast=True falls back gracefully
# ---------------------------------------------------------------------------

class TestExecuteTranslatedFast:
    """execute_translated fast=True uses Numba or falls back."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    @pytest.mark.parametrize("use_fp64", [False, True])
    def test_fast_produces_correct_output(self, use_fp64):
        rng = np.random.RandomState(42)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (2, 4, 6)).astype(np.float64),
            "weight": rng.uniform(-0.1, 0.1, (6, 8)).astype(np.float64),
        }

        outputs_fast, _ = execute_translated(
            MATMUL_CODE, {"x": (2, 4, 6), "weight": (6, 8)},
            all_inputs=inputs, fast=True, use_fp64=use_fp64,
        )
        outputs_slow, _ = execute_translated(
            MATMUL_CODE, {"x": (2, 4, 6), "weight": (6, 8)},
            all_inputs=inputs, fast=False,
        )

        np.testing.assert_allclose(
            outputs_fast["output"], outputs_slow["output"],
            rtol=1e-10 if use_fp64 else 2e-6,
            atol=0 if use_fp64 else 2e-9,
        )
        np.testing.assert_allclose(
            outputs_slow["output"], inputs["x"] @ inputs["weight"], rtol=1e-10,
        )

    def test_fast_false_still_works(self):
        rng = np.random.RandomState(42)
        inputs = {
            "x": rng.uniform(-0.1, 0.1, (2, 4, 6)).astype(np.float64),
            "weight": rng.uniform(-0.1, 0.1, (6, 8)).astype(np.float64),
        }
        outputs, _ = execute_translated(
            MATMUL_CODE, {"x": (2, 4, 6), "weight": (6, 8)},
            all_inputs=inputs, fast=False,
        )
        expected = np.einsum("bmk,kn->bmn", inputs["x"], inputs["weight"])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Dummy trace — trace_only mode captures metadata without running loops
# ---------------------------------------------------------------------------

class TestDummyTrace:
    """Verify trace_only mode captures full metadata without executing loops."""

    def test_dim_returns_empty_in_trace_only(self):
        start_trace(trace_only=True)
        Tensor(X=5)
        vals = list(Dim("X"))
        stop_trace()
        assert vals == []

    def test_dim_returns_full_in_normal_trace(self):
        start_trace(trace_only=False)
        Tensor(X=5)
        vals = list(Dim("X"))
        stop_trace()
        assert vals == [0, 1, 2, 3, 4]

    def test_dim_returns_full_without_trace(self):
        Tensor(X=5)
        vals = list(Dim("X"))
        assert vals == [0, 1, 2, 3, 4]

    def test_trace_only_captures_tensor_metadata(self):
        start_trace(trace_only=True)
        t = Tensor(B=2, M=4, N=8, dtype="fp32")
        stop_trace()
        entries = get_trace()
        tensor_entries = [e for e in entries if e["type"] == "tensor"]
        assert len(tensor_entries) == 1
        assert tensor_entries[0]["dim_names"] == ["B", "M", "N"]
        assert tensor_entries[0]["shape"] == (2, 4, 8)
        assert tensor_entries[0]["dtype"] == "fp32"

    def test_trace_only_captures_kernel_call(self):
        start_trace(trace_only=True)

        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
            C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        a = Tensor.from_numpy(np.ones((2, 3, 4)), "B,M,K")
        w = Tensor.from_numpy(np.ones((4, 5)), "K,N")
        result = matmul(a, w)

        stop_trace()
        entries = get_trace()

        call_entries = [e for e in entries if e["type"] == "call"]
        assert len(call_entries) == 1
        assert call_entries[0]["name"] == "matmul"
        assert call_entries[0]["call_type"] == "kernel"

    def test_trace_only_output_shape_correct(self):
        """Output Tensor shape is correct despite loops not running."""
        start_trace(trace_only=True)

        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
            C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        a = Tensor.from_numpy(np.ones((2, 3, 4)), "B,M,K")
        w = Tensor.from_numpy(np.ones((4, 5)), "K,N")
        result = matmul(a, w)

        stop_trace()
        assert result.shape == (2, 3, 5)
        assert result.dim_names == ["B", "M", "N"]

    def test_trace_only_apply_mask_records_call(self):
        """apply_mask still fires its tracer hook in trace_only mode."""
        start_trace(trace_only=True)

        scores = Tensor.from_numpy(np.ones((1, 1, 4, 4)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((4, 4)), "M,N", sparsity=0.5)
        result = apply_mask(scores, mask, masked_value=-1e9)

        stop_trace()
        entries = get_trace()

        call_entries = [e for e in entries if e["type"] == "call"]
        assert len(call_entries) == 1
        assert call_entries[0]["name"] == "apply_mask"
        assert call_entries[0]["call_type"] == "builtin"

    def test_trace_only_apply_mask_skips_computation(self):
        """apply_mask returns zeros instead of np.where result in trace_only."""
        start_trace(trace_only=True)

        scores_data = np.full((1, 1, 4, 4), 3.14)
        mask_data = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [1, 1, 0, 0],
                              [0, 0, 1, 1]], dtype=np.float64)

        scores = Tensor.from_numpy(scores_data, "B,H,M,N")
        mask = Tensor.from_numpy(mask_data, "M,N", sparsity=0.5)
        result = apply_mask(scores, mask, masked_value=-1e9)

        stop_trace()
        np.testing.assert_array_equal(result.data, np.zeros((1, 1, 4, 4)))

    def test_execute_translated_trace_uses_dummy(self):
        """execute_translated with trace=True uses dummy execution."""
        outputs, inputs_used = execute_translated(
            MATMUL_CODE,
            input_shapes={"x": (2, 3, 4), "weight": (4, 5)},
            trace=True,
        )
        entries = get_trace()

        tensor_entries = [e for e in entries if e["type"] == "tensor"]
        call_entries = [e for e in entries if e["type"] == "call"]

        assert any(e.get("input_name") == "x" for e in tensor_entries)
        assert any(e.get("input_name") == "weight" for e in tensor_entries)
        assert any(e["name"] == "matmul" for e in call_entries)

    def test_trace_only_metadata_complete(self):
        """Full pipeline: dummy trace produces complete metadata."""
        from solar.einsum_lib.core import _tracer

        inputs = {
            "x": np.ones((2, 3, 4)),
            "weight": np.ones((4, 5)),
        }

        outputs, _ = execute_translated(
            MATMUL_CODE,
            input_shapes={"x": (2, 3, 4), "weight": (4, 5)},
            all_inputs=inputs,
            trace=True,
        )
        entries = get_trace()
        md = build_metadata(entries)

        assert "x" in md["tensors"]
        assert md["tensors"]["x"]["dims"] == {"B": 2, "M": 3, "K": 4}
        assert "weight" in md["tensors"]
        assert md["tensors"]["weight"]["role"] == "weight"
        assert "matmul#0" in md["graph"]
        assert md["graph"]["matmul#0"]["function"] == "matmul"
        assert md["graph"]["matmul#0"]["type"] == "kernel"

    def test_trace_only_multi_kernel_graph(self):
        """Dummy trace captures graph from multi-kernel code."""
        multi_code = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale(A: Tensor("M,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=A.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            C[m, n] = A[m, n] * 2
    return C

@kernel
def add_one(A: Tensor("M,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=A.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            C[m, n] = A[m, n] + 1
    return C

def model(x):
    s = scale(x)
    return add_one(s)

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N")
    return {"output": model(x).data}
'''
        outputs, _ = execute_translated(
            multi_code,
            input_shapes={"x": (3, 4)},
            trace=True,
        )
        entries = get_trace()
        md = build_metadata(entries)

        assert "scale#0" in md["graph"]
        assert "add_one#0" in md["graph"]
        assert md["graph"]["scale#0"]["function"] == "scale"
        assert md["graph"]["add_one#0"]["function"] == "add_one"
