# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for solar.einsum_lib — Tensor, Dim, kernel, and full kernel examples."""

import numpy as np
import pytest
from pathlib import Path

from solar.einsum_lib import (
    Tensor, Dim, clear_dims, kernel, apply_mask,
    start_trace, stop_trace, get_trace,
)
from solar.einsum_lib.metadata import build_metadata, write_metadata_yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_dims():
    """Reset the dim registry and stop any active trace before and after every test."""
    clear_dims()
    stop_trace()
    yield
    stop_trace()
    clear_dims()


# ---------------------------------------------------------------------------
# Tensor — spec mode
# ---------------------------------------------------------------------------

class TestTensorSpec:
    def test_basic_spec(self):
        t = Tensor("B,M,K")
        assert t.dim_names == ["B", "M", "K"]
        assert t.data is None
        assert t.shape is None

    def test_spec_with_role(self):
        t = Tensor("K,N", role="weight")
        assert t.dim_names == ["K", "N"]
        assert t.role == "weight"

    def test_spec_strips_whitespace(self):
        t = Tensor("B , M , K")
        assert t.dim_names == ["B", "M", "K"]

    def test_spec_not_indexable(self):
        t = Tensor("B,M")
        with pytest.raises(RuntimeError, match="no data"):
            _ = t[0, 0]

    def test_spec_repr(self):
        t = Tensor("B,M,K")
        assert repr(t) == "Tensor(\"B,M,K\", dtype='fp32')"

    def test_spec_repr_with_role(self):
        t = Tensor("K,N", role="weight")
        assert repr(t) == "Tensor(\"K,N\", role='weight', dtype='fp32')"


# ---------------------------------------------------------------------------
# Tensor — data mode
# ---------------------------------------------------------------------------

class TestTensorData:
    def test_zero_initialized(self):
        t = Tensor(B=2, M=4, N=8)
        assert t.shape == (2, 4, 8)
        assert t.dim_names == ["B", "M", "N"]
        np.testing.assert_array_equal(t.data, np.zeros((2, 4, 8)))

    def test_registers_dims(self):
        Tensor(X=3, Y=5)
        assert list(Dim("X")) == [0, 1, 2]
        assert list(Dim("Y")) == [0, 1, 2, 3, 4]

    def test_data_repr(self):
        t = Tensor(B=2, M=4)
        assert repr(t) == "Tensor(B=2, M=4, dtype='fp32')"

    def test_no_args_creates_scalar(self):
        t = Tensor()
        assert t.dim_names == []
        assert t.shape == ()
        assert t.data.shape == ()
        assert float(t.data) == 0.0

    def test_empty_dims_string_creates_scalar(self):
        t = Tensor("", dtype="fp32")
        assert t.dim_names == []
        assert t.shape is None
        assert t.data is None

    def test_both_modes_raises(self):
        with pytest.raises(TypeError, match="not both"):
            Tensor("B,M", B=2, M=4)

    def test_bad_dims_type_raises(self):
        with pytest.raises(TypeError, match="string"):
            Tensor(123)

    def test_fill_default_zero(self):
        t = Tensor(A=3, B=2)
        np.testing.assert_array_equal(t.data, np.zeros((3, 2)))

    def test_fill_neg_inf(self):
        t = Tensor(A=3, B=2, fill=float("-inf"))
        assert t.shape == (3, 2)
        assert np.all(np.isneginf(t.data))

    def test_fill_pos_inf(self):
        t = Tensor(A=3, B=2, fill=float("inf"))
        assert np.all(np.isposinf(t.data))

    def test_fill_custom_value(self):
        t = Tensor(A=2, B=4, fill=42.0)
        np.testing.assert_array_equal(t.data, np.full((2, 4), 42.0))


# ---------------------------------------------------------------------------
# Tensor.from_numpy / from_random
# ---------------------------------------------------------------------------

class TestTensorFactory:
    def test_from_numpy(self):
        arr = np.arange(6).reshape(2, 3)
        t = Tensor.from_numpy(arr, "B,M")
        assert t.dim_names == ["B", "M"]
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.data, arr)

    def test_from_numpy_with_role(self):
        arr = np.ones((4, 8))
        t = Tensor.from_numpy(arr, "K,N", role="weight")
        assert t.role == "weight"

    def test_from_numpy_dim_mismatch(self):
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="doesn't match"):
            Tensor.from_numpy(arr, "A,B,C")

    def test_from_random_shape(self):
        t = Tensor.from_random(B=2, M=4, K=8)
        assert t.shape == (2, 4, 8)
        assert t.dim_names == ["B", "M", "K"]
        assert t.data is not None

    def test_from_random_with_role(self):
        t = Tensor.from_random(K=4, N=8, role="weight")
        assert t.role == "weight"


# ---------------------------------------------------------------------------
# Tensor — indexing
# ---------------------------------------------------------------------------

class TestTensorIndexing:
    def test_getitem(self):
        t = Tensor(A=2, B=3)
        t.data[1, 2] = 42.0
        assert t[1, 2] == 42.0

    def test_setitem(self):
        t = Tensor(A=2, B=3)
        t[0, 1] = 7.0
        assert t.data[0, 1] == 7.0

    def test_augmented_assignment(self):
        t = Tensor(A=2, B=3)
        t[0, 0] = 5.0
        t[0, 0] += 3.0
        assert t[0, 0] == 8.0


# ---------------------------------------------------------------------------
# Dim
# ---------------------------------------------------------------------------

class TestDim:
    def test_iteration(self):
        Tensor(B=4)
        assert list(Dim("B")) == [0, 1, 2, 3]

    def test_unregistered_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            list(Dim("MISSING"))

    def test_repr(self):
        assert repr(Dim("B")) == 'Dim("B")'

    def test_reusable(self):
        Tensor(X=3)
        d = Dim("X")
        assert list(d) == [0, 1, 2]
        assert list(d) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Dimension rebinding (kernel scopes restore their caller's dimensions)
# ---------------------------------------------------------------------------

class TestDimConflict:
    def test_same_size_ok(self):
        Tensor(B=4)
        Tensor(B=4, M=8)  # B=4 again — no error

    def test_different_size_rebinds(self):
        Tensor(B=4)
        Tensor(B=8)
        assert list(Dim("B")) == list(range(8))


# ---------------------------------------------------------------------------
# clear_dims
# ---------------------------------------------------------------------------

class TestClearDims:
    def test_clear_allows_reregister(self):
        Tensor(B=4)
        clear_dims()
        Tensor(B=8)  # no conflict after clear
        assert list(Dim("B")) == list(range(8))


# ---------------------------------------------------------------------------
# @kernel decorator — scope isolation
# ---------------------------------------------------------------------------

class TestKernelDecorator:
    def test_scope_isolation(self):
        @kernel
        def inner(A: Tensor("X,Y")) -> Tensor("X,Y"):
            C = Tensor(X=2, Y=3)
            for x in Dim("X"):
                for y in Dim("Y"):
                    C[x, y] = A[x, y] + 1
            return C

        Tensor(P=10)
        a = Tensor.from_numpy(np.ones((2, 3)), "X,Y")
        result = inner(a)
        assert result.shape == (2, 3)
        # P should still be registered (outer scope preserved)
        assert list(Dim("P")) == list(range(10))

    def test_kernel_registers_input_dims(self):
        @kernel
        def identity(A: Tensor("M,N")) -> Tensor("M,N"):
            C = Tensor(M=3, N=4)
            for m in Dim("M"):
                for n in Dim("N"):
                    C[m, n] = A[m, n]
            return C

        a = Tensor.from_numpy(np.ones((3, 4)), "M,N")
        result = identity(a)
        np.testing.assert_array_equal(result.data, np.ones((3, 4)))

    def test_sequential_kernels_different_sizes(self):
        @kernel
        def scale(A: Tensor("M,N")) -> Tensor("M,N"):
            C = Tensor(M=2, N=3)
            for m in Dim("M"):
                for n in Dim("N"):
                    C[m, n] = A[m, n] * 2
            return C

        a1 = Tensor.from_random(M=2, N=3)
        r1 = scale(a1)
        clear_dims()

        @kernel
        def scale2(A: Tensor("M,N")) -> Tensor("M,N"):
            C = Tensor(M=5, N=7)
            for m in Dim("M"):
                for n in Dim("N"):
                    C[m, n] = A[m, n] * 2
            return C

        a2 = Tensor.from_random(M=5, N=7)
        r2 = scale2(a2)
        assert r2.shape == (5, 7)


# ---------------------------------------------------------------------------
# Full kernel examples — verified against numpy
# ---------------------------------------------------------------------------

class TestMatmulKernel:
    def test_matmul_correctness(self):
        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")) -> Tensor("B,M,N"):
            C = Tensor(B=2, M=4, N=8)
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        np.random.seed(42)
        a = Tensor.from_random(B=2, M=4, K=6)
        w = Tensor.from_random(K=6, N=8, role="weight")
        result = matmul(a, w)

        expected = np.einsum("bmk,kn->bmn", a.data, w.data)
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)


class TestReluKernel:
    def test_relu_correctness(self):
        @kernel
        def relu(A: Tensor("B,M,N")) -> Tensor("B,M,N"):
            C = Tensor(B=2, M=4, N=8)
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        C[b, m, n] = max(0, A[b, m, n])
            return C

        np.random.seed(42)
        a = Tensor.from_random(B=2, M=4, N=8)
        result = relu(a)

        expected = np.maximum(0, a.data)
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)


class TestTransposeKernel:
    def test_transpose_correctness(self):
        @kernel
        def transpose(A: Tensor("B,M,N")) -> Tensor("B,N,M"):
            C = Tensor(B=2, N=8, M=4)
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        C[b, n, m] = A[b, m, n]
            return C

        np.random.seed(42)
        a = Tensor.from_random(B=2, M=4, N=8)
        result = transpose(a)

        expected = np.transpose(a.data, (0, 2, 1))
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)


class TestConv2dKernel:
    def test_conv2d_correctness(self):
        @kernel
        def conv2d(
            X: Tensor("B,C,H,W"),
            W: Tensor("O,C,R,S", role="weight"),
        ) -> Tensor("B,O,P,Q"):
            P_size = 6  # H - R + 1 = 8 - 3 + 1
            Q_size = 6  # W - S + 1 = 8 - 3 + 1
            Y = Tensor(B=1, O=4, P=P_size, Q=Q_size)
            for b in Dim("B"):
                for o in Dim("O"):
                    for p in Dim("P"):
                        for q in Dim("Q"):
                            for c in Dim("C"):
                                for r in Dim("R"):
                                    for s in Dim("S"):
                                        Y[b, o, p, q] += X[b, c, p + r, q + s] * W[o, c, r, s]
            return Y

        np.random.seed(42)
        x = Tensor.from_random(B=1, C=3, H=8, W=8)
        w = Tensor.from_random(O=4, C=3, R=3, S=3, role="weight")
        result = conv2d(x, w)

        assert result.shape == (1, 4, 6, 6)
        # Verify against scipy or manual numpy conv
        expected = np.zeros((1, 4, 6, 6))
        for b in range(1):
            for o in range(4):
                for p in range(6):
                    for q in range(6):
                        for c in range(3):
                            for r in range(3):
                                for s in range(3):
                                    expected[b, o, p, q] += (
                                        x.data[b, c, p + r, q + s] * w.data[o, c, r, s]
                                    )
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Model composition — chain matmul -> relu -> matmul
# ---------------------------------------------------------------------------

class TestModelComposition:
    def test_mlp_two_layer(self):
        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")) -> Tensor("B,M,N"):
            C = Tensor(B=2, M=4, N=8)
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        @kernel
        def relu(A: Tensor("B,M,N")) -> Tensor("B,M,N"):
            C = Tensor(B=2, M=4, N=8)
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        C[b, m, n] = max(0, A[b, m, n])
            return C

        np.random.seed(42)
        x = Tensor.from_random(B=2, M=4, K=6)
        w1 = Tensor.from_random(K=6, N=8, role="weight")
        clear_dims()

        # layer 1: matmul + relu
        x = Tensor.from_numpy(x.data, "B,M,K")
        w1 = Tensor.from_numpy(w1.data, "K,N", role="weight")
        h = matmul(x, w1)
        clear_dims()

        h = Tensor.from_numpy(h.data, "B,M,N")
        h = relu(h)

        expected = np.maximum(0, np.einsum("bmk,kn->bmn", x.data, w1.data))
        np.testing.assert_allclose(h.data, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tensor — dtype and precision
# ---------------------------------------------------------------------------

class TestTensorDtype:
    """Tests for dtype metadata on Tensor."""

    def test_default_dtype_spec(self):
        t = Tensor("B,M,K")
        assert t.dtype == "fp32"

    def test_default_dtype_data(self):
        t = Tensor(B=2, M=4)
        assert t.dtype == "fp32"

    def test_explicit_dtype_spec(self):
        t = Tensor("B,M,K", dtype="fp16")
        assert t.dtype == "fp16"

    def test_explicit_dtype_data(self):
        t = Tensor(B=2, M=4, dtype="bf16")
        assert t.dtype == "bf16"

    def test_explicit_dtype_from_numpy(self):
        arr = np.zeros((2, 3))
        t = Tensor.from_numpy(arr, "B,M", dtype="int8")
        assert t.dtype == "int8"

    def test_default_dtype_from_numpy(self):
        arr = np.zeros((2, 3))
        t = Tensor.from_numpy(arr, "B,M")
        assert t.dtype == "fp32"

    def test_explicit_dtype_from_random(self):
        t = Tensor.from_random(B=2, M=4, dtype="fp8")
        assert t.dtype == "fp8"

    def test_default_dtype_from_random(self):
        t = Tensor.from_random(B=2, M=4)
        assert t.dtype == "fp32"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            Tensor("B,M", dtype="float32")

    def test_invalid_dtype_data_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            Tensor(B=2, M=4, dtype="complex64")

    def test_invalid_dtype_from_numpy_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            Tensor.from_numpy(np.zeros((2,)), "X", dtype="bfloat16")

    def test_invalid_dtype_from_random_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            Tensor.from_random(X=2, dtype="uint8")

    def test_dtype_with_role(self):
        t = Tensor("K,N", role="weight", dtype="int8")
        assert t.dtype == "int8"
        assert t.role == "weight"

    def test_all_valid_dtypes(self):
        from solar.einsum_lib.core import VALID_DTYPES
        for dt in sorted(VALID_DTYPES):
            t = Tensor("X", dtype=dt)
            assert t.dtype == dt


class TestBytesPerElement:
    """Tests for Tensor.bytes_per_element property."""

    def test_fp32(self):
        assert Tensor("X", dtype="fp32").bytes_per_element == 4

    def test_fp16(self):
        assert Tensor("X", dtype="fp16").bytes_per_element == 2

    def test_bf16(self):
        assert Tensor("X", dtype="bf16").bytes_per_element == 2

    def test_fp64(self):
        assert Tensor("X", dtype="fp64").bytes_per_element == 8

    def test_int8(self):
        assert Tensor("X", dtype="int8").bytes_per_element == 1

    def test_fp8(self):
        assert Tensor("X", dtype="fp8").bytes_per_element == 1

    def test_nvfp4(self):
        assert Tensor("X", dtype="nvfp4").bytes_per_element == 0.5

    def test_tf32(self):
        assert Tensor("X", dtype="tf32").bytes_per_element == 4

    def test_bytes_per_element_on_data_tensor(self):
        t = Tensor(B=2, M=4, dtype="fp16")
        assert t.bytes_per_element == 2
        assert t.data.dtype == np.float64  # internal computation stays float64

    def test_bytes_per_element_from_numpy(self):
        t = Tensor.from_numpy(np.zeros((2, 3)), "B,M", dtype="int8")
        assert t.bytes_per_element == 1

    def test_bytes_per_element_from_random(self):
        t = Tensor.from_random(B=2, M=4, dtype="bf16")
        assert t.bytes_per_element == 2


class TestSparsityMetadata:
    """Tests for sparsity metadata on Tensor."""

    def test_default_sparsity_is_none(self):
        t = Tensor("B,M,K")
        assert t.sparsity is None

    def test_default_sparsity_data_mode(self):
        t = Tensor(B=2, M=4)
        assert t.sparsity is None

    def test_explicit_sparsity_spec(self):
        t = Tensor("M,N", dtype="fp32", sparsity=0.35)
        assert t.sparsity == 0.35

    def test_explicit_sparsity_data_mode(self):
        t = Tensor(M=4, N=4, sparsity=0.5)
        assert t.sparsity == 0.5

    def test_sparsity_from_numpy(self):
        arr = np.zeros((4, 4))
        t = Tensor.from_numpy(arr, "M,N", sparsity=0.25)
        assert t.sparsity == 0.25

    def test_sparsity_from_random(self):
        t = Tensor.from_random(M=4, N=4, sparsity=0.6)
        assert t.sparsity == 0.6

    def test_sparsity_in_repr_spec(self):
        t = Tensor("M,N", dtype="fp32", sparsity=0.3)
        assert "sparsity=0.3" in repr(t)

    def test_sparsity_in_repr_data(self):
        t = Tensor(M=4, N=4, sparsity=0.5)
        assert "sparsity=0.5" in repr(t)

    def test_no_sparsity_in_repr_when_none(self):
        t = Tensor("M,N")
        assert "sparsity" not in repr(t)

    def test_sparsity_with_role(self):
        t = Tensor.from_numpy(np.zeros((3, 3)), "M,K", role="weight", sparsity=0.4)
        assert t.role == "weight"
        assert t.sparsity == 0.4


class TestComputeSparsity:
    """Tests for compute_sparsity utility."""

    def test_all_nonzero(self):
        from solar.einsum_lib.verify import compute_sparsity
        arr = np.ones((3, 3))
        assert compute_sparsity(arr) == 1.0

    def test_all_zero(self):
        from solar.einsum_lib.verify import compute_sparsity
        arr = np.zeros((3, 3))
        assert compute_sparsity(arr) == 0.0

    def test_half_nonzero(self):
        from solar.einsum_lib.verify import compute_sparsity
        arr = np.array([1, 0, 1, 0])
        assert compute_sparsity(arr) == 0.5

    def test_boolean_mask(self):
        from solar.einsum_lib.verify import compute_sparsity
        mask = np.array([True, False, True, True, False])
        assert compute_sparsity(mask) == 0.6

    def test_empty_array(self):
        from solar.einsum_lib.verify import compute_sparsity
        arr = np.array([])
        assert compute_sparsity(arr) == 0.0


class TestDtypePreservedThroughKernel:
    """Verify dtype metadata survives kernel execution."""

    def test_kernel_preserves_input_dtype(self):
        @kernel
        def identity(A: Tensor("B,M", dtype="fp16")) -> Tensor("B,M"):
            C = Tensor(B=2, M=4, dtype="fp16")
            for b in Dim("B"):
                for m in Dim("M"):
                    C[b, m] = A[b, m]
            return C

        x = Tensor.from_random(B=2, M=4, dtype="fp16")
        result = identity(x)
        assert result.dtype == "fp16"
        assert x.dtype == "fp16"

    def test_mixed_precision_kernel(self):
        @kernel
        def matmul(A: Tensor("B,M,K", dtype="fp16"),
                   W: Tensor("K,N", role="weight", dtype="int8")) -> Tensor("B,M,N"):
            C = Tensor(B=1, M=2, N=3, dtype="fp16")
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        a = Tensor.from_random(B=1, M=2, K=4, dtype="fp16")
        w = Tensor.from_random(K=4, N=3, role="weight", dtype="int8")
        result = matmul(a, w)
        assert a.dtype == "fp16"
        assert w.dtype == "int8"
        assert result.dtype == "fp16"
        assert result.bytes_per_element == 2
        assert w.bytes_per_element == 1


# ---------------------------------------------------------------------------
# Explicit init — sizes from input .shape
# ---------------------------------------------------------------------------

class TestExplicitInit:
    """Tests for the explicit init pattern: Tensor(B=A.shape[0], ...)."""

    def test_sizes_from_input_shape(self):
        a = Tensor.from_random(B=2, M=3, K=4)
        c = Tensor(B=a.shape[0], M=a.shape[1], K=a.shape[2])
        assert c.shape == (2, 3, 4)
        assert c.dim_names == ["B", "M", "K"]

    def test_dims_registered_by_constructor(self):
        c = Tensor(B=5, N=7)
        assert list(Dim("B")) == list(range(5))
        assert list(Dim("N")) == list(range(7))

    def test_explicit_init_in_kernel(self):
        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")) -> Tensor("B,M,N"):
            C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        a = Tensor.from_random(B=2, M=3, K=4)
        w = Tensor.from_random(K=4, N=5, role="weight")
        result = matmul(a, w)
        assert result.shape == (2, 3, 5)
        expected = np.einsum("bmk,kn->bmn", a.data, w.data)
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)

    def test_shape_generic_across_sizes(self):
        """Same kernel code works at different input shapes."""
        @kernel
        def add_one(A: Tensor("B,M")) -> Tensor("B,M"):
            C = Tensor(B=A.shape[0], M=A.shape[1])
            for b in Dim("B"):
                for m in Dim("M"):
                    C[b, m] = A[b, m] + 1
            return C

        x1 = Tensor.from_random(B=2, M=3)
        r1 = add_one(x1)
        assert r1.shape == (2, 3)

        clear_dims()
        x2 = Tensor.from_random(B=10, M=20)
        r2 = add_one(x2)
        assert r2.shape == (10, 20)

    def test_derived_dims(self):
        """Derived dimensions (e.g. num_heads) work in explicit init."""
        @kernel
        def reshape_heads(X: Tensor("B,S,D")) -> Tensor("B,H,S,A"):
            num_heads = 2
            head_dim = X.shape[2] // num_heads
            Y = Tensor(B=X.shape[0], H=num_heads, S=X.shape[1], A=head_dim)
            for b in Dim("B"):
                for s in Dim("S"):
                    for h in Dim("H"):
                        for a in Dim("A"):
                            Y[b, h, s, a] = X[b, s, h * head_dim + a]
            return Y

        x = Tensor.from_random(B=2, S=3, D=4)
        result = reshape_heads(x)
        assert result.shape == (2, 2, 3, 2)
        np.testing.assert_allclose(
            result.data[0, 0, :, :], x.data[0, :, 0:2], rtol=1e-12
        )
        np.testing.assert_allclose(
            result.data[0, 1, :, :], x.data[0, :, 2:4], rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Sparse matrix operations — SpMV / SpMM
# ---------------------------------------------------------------------------

class TestSparseMV:
    """Dense matmul with sparse-origin matrix matches torch.sparse result."""

    def test_spmv_dense_equals_sparse(self):
        import torch

        indices = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 2]])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        A_sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
        A_dense = A_sparse.to_dense().numpy().astype(np.float64)

        x_np = np.array([[1.0], [2.0], [3.0]])

        y_sparse = torch.sparse.mm(
            A_sparse, torch.tensor(x_np, dtype=torch.float32)
        ).numpy().astype(np.float64)
        y_dense = A_dense @ x_np

        np.testing.assert_allclose(y_dense, y_sparse, rtol=1e-6)

    def test_spmv_einsum_lib_kernel(self):
        import torch

        indices = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        values = torch.tensor([2.0, 3.0, 1.0, 4.0])
        A_sparse = torch.sparse_coo_tensor(indices, values, (4, 4))
        A_dense = A_sparse.to_dense().numpy().astype(np.float64)

        x_np = np.array([1.0, 2.0, 3.0, 4.0])

        @kernel
        def matvec(A: Tensor("M,K"), X: Tensor("K,")) -> Tensor("M,"):
            Y = Tensor(M=A.shape[0])
            for m in Dim("M"):
                for k in Dim("K"):
                    Y[m] += A[m, k] * X[k]
            return Y

        a = Tensor.from_numpy(A_dense, "M,K")
        x = Tensor.from_numpy(x_np, "K")
        result = matvec(a, x)

        y_ref = torch.sparse.mm(
            A_sparse, torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
        ).squeeze(1).numpy().astype(np.float64)
        np.testing.assert_allclose(result.data, y_ref, rtol=1e-6)


class TestSparseMM:
    """Dense matmul with sparse-origin matrix matches torch.sparse.mm for matrices."""

    def test_spmm_dense_equals_sparse(self):
        import torch

        indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 2]])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        A_sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
        A_dense = A_sparse.to_dense().numpy().astype(np.float64)

        B_np = np.random.RandomState(42).uniform(-1, 1, (3, 5))

        C_sparse = torch.sparse.mm(
            A_sparse, torch.tensor(B_np, dtype=torch.float32)
        ).numpy().astype(np.float64)
        C_dense = A_dense @ B_np

        np.testing.assert_allclose(C_dense, C_sparse, rtol=1e-6)

    def test_spmm_einsum_lib_kernel(self):
        import torch

        indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
        values = torch.tensor([5.0, 6.0, 7.0])
        A_sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
        A_dense = A_sparse.to_dense().numpy().astype(np.float64)

        B_np = np.random.RandomState(7).uniform(-1, 1, (3, 4))

        @kernel
        def matmul(A: Tensor("M,K"), B: Tensor("K,N")) -> Tensor("M,N"):
            C = Tensor(M=A.shape[0], N=B.shape[1])
            for m in Dim("M"):
                for n in Dim("N"):
                    for k in Dim("K"):
                        C[m, n] += A[m, k] * B[k, n]
            return C

        a = Tensor.from_numpy(A_dense, "M,K")
        b = Tensor.from_numpy(B_np, "K,N")
        result = matmul(a, b)

        C_ref = torch.sparse.mm(
            A_sparse, torch.tensor(B_np, dtype=torch.float32)
        ).numpy().astype(np.float64)
        np.testing.assert_allclose(result.data, C_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# Masked attention — dense + mask_bias == sparse attention
# ---------------------------------------------------------------------------

def _numpy_softmax(x, axis=-1):
    """Numerically stable softmax along an axis."""
    mx = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - mx)
    return ex / np.sum(ex, axis=axis, keepdims=True)


class TestMaskedAttention:
    """Prove that dense attention + conditional mask reproduces sparse (RandomAttention-style) output."""

    def test_dense_with_mask_equals_sparse(self):
        """Pure NumPy: np.where(mask, scores, -1e9) + softmax == gather-based sparse attention."""
        B, H, S, D = 1, 1, 4, 4
        rng = np.random.RandomState(42)
        Q = rng.uniform(-0.1, 0.1, (B, H, S, D))
        K = rng.uniform(-0.1, 0.1, (B, H, S, D))
        V = rng.uniform(-0.1, 0.1, (B, H, S, D))
        scale = 1.0 / np.sqrt(D)

        mask = np.array([
            [True,  True,  False, False],
            [True,  True,  True,  False],
            [False, True,  True,  True],
            [False, False, True,  True],
        ])

        # --- sparse: per-query gather ---
        sparse_out = np.zeros_like(Q)
        for i in range(S):
            sel = np.where(mask[i])[0]
            q_i = Q[:, :, i:i+1, :]
            k_sel = K[:, :, sel, :]
            v_sel = V[:, :, sel, :]
            scores = np.matmul(q_i, k_sel.transpose(0, 1, 3, 2)) * scale
            attn = _numpy_softmax(scores, axis=-1)
            sparse_out[:, :, i:i+1, :] = np.matmul(attn, v_sel)

        # --- dense with conditional replacement ---
        scores_full = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
        scores_masked = np.where(mask, scores_full, -1e9)
        attn_full = _numpy_softmax(scores_masked, axis=-1)
        dense_out = np.matmul(attn_full, V)

        np.testing.assert_allclose(dense_out, sparse_out, rtol=1e-5, atol=1e-6)

    def test_einsum_lib_masked_attention(self):
        """Same logic expressed as einsum_lib kernels + builtin apply_mask."""
        B, H, S, D = 1, 1, 4, 4
        rng = np.random.RandomState(99)
        Q_np = rng.uniform(-0.1, 0.1, (B, H, S, D))
        K_np = rng.uniform(-0.1, 0.1, (B, H, S, D))
        V_np = rng.uniform(-0.1, 0.1, (B, H, S, D))
        scale = 1.0 / np.sqrt(D)

        mask_bool = np.array([
            [True,  False, True,  False],
            [False, True,  False, True],
            [True,  True,  True,  False],
            [False, False, True,  True],
        ])
        mask_01 = mask_bool.astype(np.float64)

        @kernel
        def attn_scores(QQ: Tensor("B,H,M,K"), KK: Tensor("B,H,N,K")) -> Tensor("B,H,M,N"):
            S = Tensor(B=QQ.shape[0], H=QQ.shape[1], M=QQ.shape[2], N=KK.shape[2])
            for b in Dim("B"):
                for h in Dim("H"):
                    for m in Dim("M"):
                        for n in Dim("N"):
                            for k in Dim("K"):
                                S[b, h, m, n] += QQ[b, h, m, k] * KK[b, h, n, k]
            return S

        @kernel
        def attn_softmax(S: Tensor("B,H,M,N")) -> Tensor("B,H,M,N"):
            from solar.einsum_lib.verify import _softmax
            P = Tensor(B=S.shape[0], H=S.shape[1], M=S.shape[2], N=S.shape[3])
            for b in Dim("B"):
                for h in Dim("H"):
                    for m in Dim("M"):
                        row = [S[b, h, m, n] for n in Dim("N")]
                        sm = _softmax(row)
                        for i, n in enumerate(Dim("N")):
                            P[b, h, m, n] = sm[i]
            return P

        @kernel
        def attn_values(P: Tensor("B,H,M,N"), VV: Tensor("B,H,N,K")) -> Tensor("B,H,M,K"):
            O = Tensor(B=P.shape[0], H=P.shape[1], M=P.shape[2], K=VV.shape[3])
            for b in Dim("B"):
                for h in Dim("H"):
                    for m in Dim("M"):
                        for k in Dim("K"):
                            for n in Dim("N"):
                                O[b, h, m, k] += P[b, h, m, n] * VV[b, h, n, k]
            return O

        # --- run pipeline: scores -> scale -> apply_mask (builtin) -> softmax -> output ---
        q = Tensor.from_numpy(Q_np, "B,H,M,K")
        k = Tensor.from_numpy(K_np, "B,H,N,K")
        scores = attn_scores(q, k)

        clear_dims()
        scores_t = Tensor.from_numpy(scores.data * scale, "B,H,M,N")
        mask_t = Tensor.from_numpy(mask_01, "M,N", sparsity=0.5)
        masked = apply_mask(scores_t, mask_t, masked_value=-1e9)

        clear_dims()
        masked_t = Tensor.from_numpy(masked.data, "B,H,M,N")
        probs = attn_softmax(masked_t)

        clear_dims()
        probs_t = Tensor.from_numpy(probs.data, "B,H,M,N")
        v = Tensor.from_numpy(V_np, "B,H,N,K")
        out = attn_values(probs_t, v)

        # --- reference: NumPy sparse gather ---
        sparse_out = np.zeros_like(Q_np)
        for i in range(S):
            sel = np.where(mask_bool[i])[0]
            q_i = Q_np[:, :, i:i+1, :]
            k_sel = K_np[:, :, sel, :]
            v_sel = V_np[:, :, sel, :]
            sc = np.matmul(q_i, k_sel.transpose(0, 1, 3, 2)) * scale
            at = _numpy_softmax(sc, axis=-1)
            sparse_out[:, :, i:i+1, :] = np.matmul(at, v_sel)

        np.testing.assert_allclose(out.data, sparse_out, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_mask builtin
# ---------------------------------------------------------------------------

class TestApplyMaskBuiltin:
    """Tests for the builtin apply_mask function."""

    def test_keeps_where_mask_is_one(self):
        scores_np = np.full((2, 4, 8, 8), 3.14)
        mask_np = np.ones((8, 8))
        mask_np[0, 1] = 0.0
        mask_np[3, 5] = 0.0

        scores = Tensor.from_numpy(scores_np, "B,H,M,N")
        mask = Tensor.from_numpy(mask_np, "M,N")
        result = apply_mask(scores, mask, masked_value=-1e9)

        expected = np.where(mask_np > 0.5, scores_np, -1e9)
        np.testing.assert_array_equal(result.data, expected)
        assert result.data[0, 0, 0, 1] == -1e9
        assert result.data[0, 0, 0, 0] == 3.14

    def test_multiplicative_masking(self):
        scores_np = np.full((1, 1, 4, 4), 5.0)
        mask_np = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 1, 0, 0],
                            [0, 0, 1, 1]], dtype=np.float64)

        scores = Tensor.from_numpy(scores_np, "B,H,M,N")
        mask = Tensor.from_numpy(mask_np, "M,N")
        result = apply_mask(scores, mask, masked_value=0.0)

        expected = np.where(mask_np > 0.5, scores_np, 0.0)
        np.testing.assert_array_equal(result.data, expected)
        assert result.data[0, 0, 0, 1] == 0.0
        assert result.data[0, 0, 0, 0] == 5.0

    def test_custom_masked_value(self):
        scores_np = np.ones((1, 1, 2, 2))
        mask_np = np.array([[1, 0], [0, 1]], dtype=np.float64)

        scores = Tensor.from_numpy(scores_np, "B,H,M,N")
        mask = Tensor.from_numpy(mask_np, "M,N")
        result = apply_mask(scores, mask, masked_value=-42.0)

        assert result.data[0, 0, 0, 0] == 1.0
        assert result.data[0, 0, 0, 1] == -42.0
        assert result.data[0, 0, 1, 0] == -42.0
        assert result.data[0, 0, 1, 1] == 1.0

    def test_default_masked_value(self):
        scores_np = np.ones((1, 1, 2, 2))
        mask_np = np.array([[1, 0], [1, 0]], dtype=np.float64)

        scores = Tensor.from_numpy(scores_np, "B,H,M,N")
        mask = Tensor.from_numpy(mask_np, "M,N")
        result = apply_mask(scores, mask)

        assert result.data[0, 0, 0, 1] == -1e9

    def test_sparsity_propagation(self):
        scores = Tensor.from_numpy(np.ones((2, 4, 8, 8)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((8, 8)), "M,N", sparsity=0.5)
        result = apply_mask(scores, mask)
        assert result.sparsity == 0.5

    def test_sparsity_none_when_mask_has_none(self):
        scores = Tensor.from_numpy(np.ones((2, 4, 8, 8)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((8, 8)), "M,N")
        result = apply_mask(scores, mask)
        assert result.sparsity is None

    def test_preserves_dtype(self):
        scores = Tensor.from_numpy(np.ones((1, 1, 4, 4)), "B,H,M,N", dtype="fp16")
        mask = Tensor.from_numpy(np.ones((4, 4)), "M,N", dtype="fp32")
        result = apply_mask(scores, mask)
        assert result.dtype == "fp16"

    def test_preserves_dim_names(self):
        scores = Tensor.from_numpy(np.ones((1, 1, 4, 4)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((4, 4)), "M,N")
        result = apply_mask(scores, mask)
        assert result.dim_names == ["B", "H", "M", "N"]

    def test_preserves_shape(self):
        scores = Tensor.from_numpy(np.ones((2, 4, 8, 16)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((8, 16)), "M,N")
        result = apply_mask(scores, mask)
        assert result.shape == (2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Runtime tracing
# ---------------------------------------------------------------------------

class TestTraceRecording:
    """Verify that the tracer captures tensor creation and kernel calls."""

    def test_matmul_trace_entries(self):
        start_trace()

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

        tensor_entries = [e for e in entries if e["type"] == "tensor"]
        call_entries = [e for e in entries if e["type"] == "call"]

        assert len(tensor_entries) >= 3  # a, w, C (output)
        assert len(call_entries) == 1
        assert call_entries[0]["name"] == "matmul"
        assert call_entries[0]["call_type"] == "kernel"

    def test_trace_inactive_by_default(self):
        a = Tensor.from_numpy(np.ones((2, 3)), "X,Y")
        entries = get_trace()
        tensor_entries = [e for e in entries if e.get("dim_names") == ["X", "Y"]]
        assert len(tensor_entries) == 0

    def test_apply_mask_traced_as_builtin(self):
        start_trace()

        scores = Tensor.from_numpy(np.ones((1, 1, 4, 4)), "B,H,M,N")
        mask = Tensor.from_numpy(np.ones((4, 4)), "M,N", sparsity=0.5)
        result = apply_mask(scores, mask, masked_value=-1e9)

        stop_trace()
        entries = get_trace()

        call_entries = [e for e in entries if e["type"] == "call"]
        assert len(call_entries) == 1
        assert call_entries[0]["name"] == "apply_mask"
        assert call_entries[0]["call_type"] == "builtin"
        assert call_entries[0]["call_site_line"] is not None

    def test_kernel_call_site_line_captured(self):
        start_trace()

        @kernel
        def add_one(A: Tensor("M,N")) -> Tensor("M,N"):
            C = Tensor(M=A.shape[0], N=A.shape[1])
            for m in Dim("M"):
                for n in Dim("N"):
                    C[m, n] = A[m, n] + 1
            return C

        a = Tensor.from_numpy(np.ones((2, 3)), "M,N")
        result = add_one(a)

        stop_trace()
        entries = get_trace()

        call_entries = [e for e in entries if e["type"] == "call"]
        assert len(call_entries) == 1
        assert isinstance(call_entries[0]["call_site_line"], int)
        assert call_entries[0]["call_site_line"] > 0


class TestInputNameMatching:
    """Verify from_numpy with registered inputs correctly maps tensor names."""

    def test_registered_inputs_matched(self):
        from solar.einsum_lib.core import _tracer

        arr_a = np.ones((2, 3))
        arr_b = np.ones((3, 4))
        inputs = {"x_0": arr_a, "weight": arr_b}

        start_trace()
        _tracer.register_inputs(inputs)

        t_a = Tensor.from_numpy(arr_a, "M,K")
        t_b = Tensor.from_numpy(arr_b, "K,N", role="weight")

        stop_trace()
        entries = get_trace()

        tensor_entries = [e for e in entries if e["type"] == "tensor"]
        names = {e["input_name"] for e in tensor_entries if e["input_name"]}
        assert "x_0" in names
        assert "weight" in names

    def test_unregistered_array_has_no_name(self):
        from solar.einsum_lib.core import _tracer

        arr_a = np.ones((2, 3))
        inputs = {"x_0": arr_a}

        start_trace()
        _tracer.register_inputs(inputs)

        other = np.ones((4, 5))
        t = Tensor.from_numpy(other, "P,Q")

        stop_trace()
        entries = get_trace()

        pq_entries = [e for e in entries if e["type"] == "tensor" and e["dim_names"] == ["P", "Q"]]
        assert len(pq_entries) == 1
        assert pq_entries[0]["input_name"] is None


# ---------------------------------------------------------------------------
# Metadata building
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    """Verify build_metadata() produces correct tensors and graph dicts."""

    def test_matmul_metadata(self):
        from solar.einsum_lib.core import _tracer

        arr_a = np.ones((2, 3, 4))
        arr_w = np.ones((4, 5))
        inputs = {"x_0": arr_a, "weight": arr_w}

        start_trace()
        _tracer.register_inputs(inputs)

        @kernel
        def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
            C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
            for b in Dim("B"):
                for m in Dim("M"):
                    for n in Dim("N"):
                        for k in Dim("K"):
                            C[b, m, n] += A[b, m, k] * W[k, n]
            return C

        a = Tensor.from_numpy(arr_a, "B,M,K")
        w = Tensor.from_numpy(arr_w, "K,N", role="weight")
        result = matmul(a, w)

        stop_trace()
        md = build_metadata(get_trace())

        assert "x_0" in md["tensors"]
        assert md["tensors"]["x_0"]["dims"] == {"B": 2, "M": 3, "K": 4}
        assert md["tensors"]["x_0"]["role"] == "input"

        assert "weight" in md["tensors"]
        assert md["tensors"]["weight"]["role"] == "weight"

        assert "matmul#0" in md["graph"]
        node = md["graph"]["matmul#0"]
        assert node["type"] == "kernel"
        assert node["function"] == "matmul"
        assert node["output"] == "matmul#0"
        assert "call_site_line" in node
        assert len(node["inputs"]) == 2
        assert node["inputs"][0]["tensor"] == "x_0"
        assert node["inputs"][0]["from"] == "input"
        assert node["inputs"][1]["tensor"] == "weight"
        assert node["inputs"][1]["from"] == "input"

    def test_masked_attention_metadata_sparsity(self):
        from solar.einsum_lib.core import _tracer

        scores_np = np.ones((1, 1, 4, 4))
        mask_np = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 1, 0, 0],
                            [0, 0, 1, 1]], dtype=np.float64)

        inputs = {"scores": scores_np, "attention_mask": mask_np}
        start_trace()
        _tracer.register_inputs(inputs)

        scores = Tensor.from_numpy(scores_np, "B,H,M,N")
        mask = Tensor.from_numpy(mask_np, "M,N", role="mask", sparsity=0.5)
        result = apply_mask(scores, mask, masked_value=-1e9)

        stop_trace()
        md = build_metadata(get_trace())

        assert md["tensors"]["attention_mask"]["sparsity"] == 0.5
        assert md["tensors"]["attention_mask"]["role"] == "mask"

        assert "apply_mask#0" in md["graph"]
        node = md["graph"]["apply_mask#0"]
        assert node["type"] == "builtin"
        assert node["function"] == "apply_mask"
        assert "call_site_line" in node
        assert "apply_mask#0" in md["tensors"]
        assert md["tensors"]["apply_mask#0"]["sparsity"] == 0.5

    def test_repeated_kernel_naming(self):
        from solar.einsum_lib.core import _tracer

        arr = np.ones((2, 3))
        inputs = {"x": arr}
        start_trace()
        _tracer.register_inputs(inputs)

        @kernel
        def scale(A: Tensor("M,N")) -> Tensor("M,N"):
            C = Tensor(M=A.shape[0], N=A.shape[1])
            for m in Dim("M"):
                for n in Dim("N"):
                    C[m, n] = A[m, n] * 2
            return C

        a = Tensor.from_numpy(arr, "M,N")
        r1 = scale(a)
        clear_dims()
        r1_t = Tensor.from_numpy(r1.data, "M,N")
        r2 = scale(r1_t)

        stop_trace()
        md = build_metadata(get_trace())

        assert "scale#0" in md["graph"]
        assert "scale#1" in md["graph"]
        assert md["graph"]["scale#0"]["function"] == "scale"
        assert md["graph"]["scale#1"]["function"] == "scale"


class TestWriteMetadataYaml:
    """Verify YAML output is written and parseable."""

    def test_yaml_roundtrip(self, tmp_path):
        import yaml

        metadata = {
            "tensors": {
                "x_0": {"dims": {"B": 2, "M": 4}, "dtype": "fp32", "role": "input"},
            },
            "graph": {
                "matmul": {
                    "type": "kernel",
                    "inputs": [{"tensor": "x_0", "from": "input"}],
                    "output": "matmul",
                },
            },
        }

        yaml_path = str(tmp_path / "test.yaml")
        write_metadata_yaml(metadata, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)

        assert loaded["tensors"]["x_0"]["dims"] == {"B": 2, "M": 4}
        assert loaded["graph"]["matmul"]["type"] == "kernel"


class TestCaptureModelRun:
    """Tests for hook-based model capture."""

    def test_dense_model_captures_output(self):
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class DenseModel(nn.Module):
            def forward(self, x):
                return x * 2.0

        model = DenseModel().double().eval()
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]
        numpy_inputs = {"x": x_np}

        result = capture_model_run(model, torch_inputs, numpy_inputs)

        assert "output" in result.reference_outputs
        np.testing.assert_allclose(result.reference_outputs["output"], x_np * 2.0)
        assert len(result.captured_masks) == 0
        assert len(result.mask_fill_values) == 0
        assert result.all_inputs is numpy_inputs

    def test_masked_fill_captures_mask(self):
        """Captured mask uses 1=keep convention (inverted from masked_fill)."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class MaskedModel(nn.Module):
            def forward(self, x):
                mask = torch.tril(torch.ones(x.shape[-2], x.shape[-1], dtype=torch.bool))
                return x.masked_fill(~mask, float("-inf"))

        model = MaskedModel().double().eval()
        x_np = np.ones((2, 4, 4))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]
        numpy_inputs = {"x": x_np}

        result = capture_model_run(model, torch_inputs, numpy_inputs)

        assert len(result.captured_masks) == 1
        mask_name = list(result.captured_masks.keys())[0]
        assert mask_name == "captured_mask_0"
        mask_arr = result.captured_masks[mask_name]
        assert mask_arr.shape == (4, 4)
        assert mask_arr[0, 0] == 1.0  # lower-left kept (tril)
        assert mask_arr[0, 3] == 0.0  # upper-right masked out
        assert result.mask_fill_values["captured_mask_0"] == float("-inf")

    def test_fill_value_captured(self):
        """Fill value from masked_fill is recorded alongside the mask."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class MaskedModel(nn.Module):
            def forward(self, x):
                mask = torch.zeros(3, 3, dtype=torch.bool)
                mask[0, 0] = True  # only (0,0) replaced
                return x.masked_fill(mask, -999.0)

        model = MaskedModel().double().eval()
        x_np = np.ones((3, 3))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})

        assert len(result.captured_masks) == 1
        assert result.mask_fill_values["captured_mask_0"] == -999.0
        mask = result.captured_masks["captured_mask_0"]
        assert mask[0, 0] == 0.0  # was True in masked_fill → inverted to 0 (masked out)
        assert mask[1, 1] == 1.0  # was False → inverted to 1 (kept)

    def test_multiple_masks_deduplicated(self):
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class DoubleMask(nn.Module):
            def forward(self, x):
                mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
                x = x.masked_fill(~mask, -1e9)
                x = x.masked_fill(~mask, -1e9)  # same mask again
                return x

        model = DoubleMask().double().eval()
        x_np = np.ones((3, 3))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]
        numpy_inputs = {"x": x_np}

        result = capture_model_run(model, torch_inputs, numpy_inputs)
        assert len(result.captured_masks) == 1

    def test_trivial_mask_filtered(self):
        """Masks that are all-keep after inversion (no-ops) are filtered out."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class TrivialMask(nn.Module):
            def forward(self, x):
                mask = torch.zeros(3, 3, dtype=torch.bool)  # all False
                return x.masked_fill(mask, 0.0)  # no-op: nothing replaced

        model = TrivialMask().double().eval()
        x_np = np.ones((3, 3))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})
        assert len(result.captured_masks) == 0
        assert len(result.mask_fill_values) == 0

    def test_tuple_output(self):
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class MultiOut(nn.Module):
            def forward(self, x):
                return x, x * 2

        model = MultiOut().double().eval()
        x_np = np.array([1.0, 2.0])
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]
        numpy_inputs = {"x": x_np}

        result = capture_model_run(model, torch_inputs, numpy_inputs)
        assert "output_0" in result.reference_outputs
        assert "output_1" in result.reference_outputs

    def test_getitem_captures_bool_mask(self):
        """Indexing a 2D+ bool tensor captures the full tensor (gather pattern)."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class GatherModel(nn.Module):
            def forward(self, x):
                seq_len = x.shape[-1]
                mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
                out = torch.zeros_like(x)
                for i in range(seq_len):
                    positions = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
                    out[:, positions] += x[:, i : i + 1]
                return out

        model = GatherModel().double().eval()
        x_np = np.ones((2, 4))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})

        assert len(result.captured_masks) == 1
        mask = result.captured_masks["captured_mask_0"]
        assert mask.shape == (4, 4)
        assert mask[0, 0] == 1.0  # lower-left kept (tril, 1=keep)
        assert mask[0, 3] == 0.0  # upper-right masked out
        assert result.mask_fill_values["captured_mask_0"] == float("-inf")

    def test_where_captures_condition(self):
        """torch.where with a 2D+ bool condition captures the condition."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class WhereModel(nn.Module):
            def forward(self, x):
                mask = torch.zeros(x.shape[-2], x.shape[-1], dtype=torch.bool)
                mask[0, 0] = True
                mask[1, 1] = True
                return torch.where(mask, x, torch.tensor(float("-inf")))

        model = WhereModel().double().eval()
        x_np = np.ones((3, 3))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})

        assert len(result.captured_masks) == 1
        mask = result.captured_masks["captured_mask_0"]
        assert mask.shape == (3, 3)
        assert mask[0, 0] == 1.0  # True = keep
        assert mask[0, 1] == 0.0  # False = masked
        assert result.mask_fill_values["captured_mask_0"] == float("-inf")

    def test_no_false_positive_float_getitem(self):
        """Indexing a float tensor does NOT trigger capture."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class FloatIndexModel(nn.Module):
            def forward(self, x):
                return x[0] + x[1]

        model = FloatIndexModel().double().eval()
        x_np = np.ones((3, 4))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})
        assert len(result.captured_masks) == 0

    def test_combined_masked_fill_and_getitem(self):
        """Both hooks fire for a model with masked_fill AND bool indexing."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class CombinedModel(nn.Module):
            def forward(self, x):
                n = x.shape[-1]
                mask_a = torch.tril(torch.ones(n, n, dtype=torch.bool))
                mask_b = torch.zeros(n, n, dtype=torch.bool)
                mask_b[0, :2] = True
                mask_b[1, :3] = True
                mask_b[2, :] = True
                mask_b[3, :] = True

                filled = x.masked_fill(~mask_a, float("-inf"))
                out = torch.zeros_like(x)
                for i in range(n):
                    pos = torch.nonzero(mask_b[i], as_tuple=False).squeeze(-1)
                    out[:, pos] += filled[:, i : i + 1]
                return out

        model = CombinedModel().double().eval()
        x_np = np.ones((4, 4))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})

        assert len(result.captured_masks) == 2
        shapes = {m.shape for m in result.captured_masks.values()}
        assert (4, 4) in shapes

    def test_getitem_deduplicates_across_loop(self):
        """Same bool tensor indexed in a loop is captured only once."""
        import torch
        import torch.nn as nn
        from solar.einsum_lib.verify import capture_model_run

        class LoopIndex(nn.Module):
            def forward(self, x):
                mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
                total = torch.zeros(4)
                for i in range(4):
                    pos = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
                    total[pos] += 1.0
                return x + total.unsqueeze(0)

        model = LoopIndex().double().eval()
        x_np = np.ones((1, 4))
        torch_inputs = [torch.tensor(x_np, dtype=torch.float64)]

        result = capture_model_run(model, torch_inputs, {"x": x_np})
        assert len(result.captured_masks) == 1


class TestExecuteTranslatedAllInputs:
    """Test that all_inputs bypasses random generation."""

    def test_all_inputs_used_directly(self):
        from solar.einsum_lib.verify import execute_translated

        code = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale(A: Tensor("M,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=A.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            C[m, n] = A[m, n] * 2.0
    return C

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N")
    return {"output": scale(x).data}
'''
        custom = {"x": np.array([[10.0, 20.0], [30.0, 40.0]])}
        outputs, inputs_used = execute_translated(
            code, {"x": (2, 2)}, all_inputs=custom,
        )
        np.testing.assert_allclose(outputs["output"], custom["x"] * 2.0)
        assert inputs_used["x"] is custom["x"]


class TestExecuteTranslatedWithTrace:
    """Verify execute_translated with trace=True populates trace entries."""

    def test_trace_from_execute_translated(self):
        from solar.einsum_lib.verify import execute_translated
        from solar.einsum_lib.core import get_trace

        code = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("M,K"), W: Tensor("K,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=W.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            for k in Dim("K"):
                C[m, n] += A[m, k] * W[k, n]
    return C

def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["x"], "M,K")
    w = Tensor.from_numpy(inputs["weight"], "K,N", role="weight")
    out = matmul(a, w)
    return {"output": out.data}
'''

        outputs, inputs_used = execute_translated(
            code,
            input_shapes={"x": (3, 4), "weight": (4, 5)},
            trace=True,
        )
        entries = get_trace()

        assert len(entries) > 0
        tensor_entries = [e for e in entries if e["type"] == "tensor"]
        call_entries = [e for e in entries if e["type"] == "call"]
        assert any(e.get("input_name") == "x" for e in tensor_entries)
        assert any(e.get("input_name") == "weight" for e in tensor_entries)
        assert any(e["name"] == "matmul" for e in call_entries)

        md = build_metadata(entries)
        assert "x" in md["tensors"]
        assert "weight" in md["tensors"]
        assert "matmul#0" in md["graph"]
        assert md["graph"]["matmul#0"]["function"] == "matmul"
        assert "call_site_line" in md["graph"]["matmul#0"]


class TestIntegerKernelParams:
    """Verify that kernels with int parameters work via all_inputs."""

    _CODE = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def reshape_to_heads(X: Tensor("B,S,H", dtype="fp32"), num_heads: int) -> Tensor("B,Nh,S,D"):
    head_dim = X.shape[2] // num_heads
    Y = Tensor(B=X.shape[0], Nh=num_heads, S=X.shape[1], D=head_dim, dtype="fp32")
    for b in Dim("B", parallel=True):
        for nh in Dim("Nh"):
            for s in Dim("S"):
                for d in Dim("D"):
                    Y[b, nh, s, d] = X[b, s, nh * head_dim + d]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,S,H", dtype="fp32")
    num_heads = inputs["num_heads"]
    out = reshape_to_heads(x, num_heads)
    return {"output": out.data}
'''

    def test_int_param_in_all_inputs(self):
        from solar.einsum_lib.verify import execute_translated

        x = np.arange(24, dtype=np.float64).reshape(1, 2, 12)
        all_inputs = {"x": x, "num_heads": 3}
        outputs, _ = execute_translated(
            self._CODE, {"x": (1, 2, 12)}, all_inputs=all_inputs,
        )
        assert outputs["output"].shape == (1, 3, 2, 4)

    def test_int_param_with_trace(self):
        from solar.einsum_lib.verify import execute_translated
        from solar.einsum_lib.core import get_trace

        full_x = np.zeros((2, 8, 64), dtype=np.float64)
        all_inputs = {"x": full_x, "num_heads": 8}
        execute_translated(
            self._CODE, {"x": (2, 8, 64)}, all_inputs=all_inputs,
            trace=True, shape_only=True,
        )
        entries = get_trace()
        md = build_metadata(entries)
        out_meta = md["tensors"]["reshape_to_heads#0"]
        assert out_meta["dims"]["Nh"] == 8
        assert out_meta["dims"]["D"] == 8

    def test_verify_small_trace_full(self):
        """Verify at small free dims, trace at full — arch dims unchanged."""
        from solar.einsum_lib.verify import execute_translated
        from solar.einsum_lib.core import get_trace

        small_x = np.arange(24, dtype=np.float64).reshape(1, 2, 12)
        small_inputs = {"x": small_x, "num_heads": 3}
        outputs, _ = execute_translated(
            self._CODE, {"x": (1, 2, 12)}, all_inputs=small_inputs,
        )
        assert outputs["output"].shape == (1, 3, 2, 4)

        clear_dims()

        full_x = np.zeros((2, 8, 12), dtype=np.float64)
        full_inputs = {"x": full_x, "num_heads": 3}
        execute_translated(
            self._CODE, {"x": (2, 8, 12)}, all_inputs=full_inputs,
            trace=True, shape_only=True,
        )
        entries = get_trace()
        md = build_metadata(entries)
        out_meta = md["tensors"]["reshape_to_heads#0"]
        assert out_meta["dims"]["B"] == 2
        assert out_meta["dims"]["Nh"] == 3
        assert out_meta["dims"]["S"] == 8
        assert out_meta["dims"]["D"] == 4


# ---------------------------------------------------------------------------
# Memory-reduction tests
# ---------------------------------------------------------------------------

class TestFromNumpySharing:
    """torch.from_numpy shares memory with the source array."""

    def test_from_numpy_shares_memory(self):
        import torch
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = torch.from_numpy(arr)
        assert t.data_ptr() == arr.__array_interface__["data"][0]

    def test_from_numpy_dtype_cast_copies(self):
        """Casting dtype via .to() creates a copy (safe for model params)."""
        import torch
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = torch.from_numpy(arr).to(torch.float64)
        assert t.data_ptr() != arr.__array_interface__["data"][0]


class TestFp32DefaultTolerances:
    """compare_outputs uses KernelBench-aligned tolerances for fp32."""

    def test_fp32_noise_passes(self):
        from solar.einsum_lib.verify import compare_outputs
        ref = {"output": np.array([1.0, 2.0, 3.0])}
        trans = {"output": np.array([1.0 + 5e-5, 2.0 - 5e-5, 3.0 + 5e-5])}
        vr = compare_outputs(trans, ref, atol=1e-4, rtol=1e-4)
        assert vr.success

    def test_fp32_large_error_fails(self):
        from solar.einsum_lib.verify import compare_outputs
        ref = {"output": np.array([1.0, 2.0, 3.0])}
        trans = {"output": np.array([1.01, 2.0, 3.0])}
        vr = compare_outputs(trans, ref, atol=1e-4, rtol=1e-4)
        assert not vr.success

    def test_fp64_tighter_defaults(self):
        """Default compare_outputs tolerances (fp64 path) are tighter."""
        from solar.einsum_lib.verify import compare_outputs
        ref = {"output": np.array([1.0, 2.0, 3.0])}
        trans = {"output": np.array([1.0 + 5e-5, 2.0, 3.0])}
        vr_fp64 = compare_outputs(trans, ref)
        assert vr_fp64.success
        ref2 = {"output": np.array([1.0, 2.0, 3.0])}
        trans2 = {"output": np.array([1.0 + 5e-4, 2.0, 3.0])}
        vr_fp64_fail = compare_outputs(trans2, ref2)
        assert not vr_fp64_fail.success

    def test_fp64_opt_in_execute(self):
        """execute_translated with use_fp64=True still works."""
        from solar.einsum_lib.verify import execute_translated

        code = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def scale(A: Tensor("M,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=A.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            C[m, n] = A[m, n] * 2.0
    return C

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N")
    return {"output": scale(x).data}
'''
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        outputs, _ = execute_translated(
            code, {"x": (2, 2)}, all_inputs={"x": x}, use_fp64=True,
        )
        np.testing.assert_allclose(outputs["output"], x * 2.0, rtol=1e-12)


class TestShapeStubTracing:
    """broadcast_to stubs produce correct metadata during tracing."""

    def test_stub_shape_and_dtype(self):
        shape = (128, 64, 32)
        stub = np.broadcast_to(np.zeros(1, dtype=np.float32), shape)
        stub.flags.writeable = False
        assert stub.shape == shape
        assert stub.dtype == np.float32

    def test_stub_trace_produces_correct_yaml(self):
        from solar.einsum_lib.verify import execute_translated
        from solar.einsum_lib.core import get_trace

        code = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("M,K"), W: Tensor("K,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=W.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            for k in Dim("K"):
                C[m, n] += A[m, k] * W[k, n]
    return C

def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["x"], "M,K")
    w = Tensor.from_numpy(inputs["weight"], "K,N", role="weight")
    out = matmul(a, w)
    return {"output": out.data}
'''
        x_stub = np.broadcast_to(np.zeros(1, dtype=np.float32), (16, 8))
        w_stub = np.broadcast_to(np.zeros(1, dtype=np.float32), (8, 4))
        x_stub.flags.writeable = False
        w_stub.flags.writeable = False

        outputs, _ = execute_translated(
            code,
            input_shapes={"x": (16, 8), "weight": (8, 4)},
            all_inputs={"x": x_stub, "weight": w_stub},
            trace=True, shape_only=True,
        )
        entries = get_trace()
        md = build_metadata(entries)
        assert md["tensors"]["x"]["dims"] == {"M": 16, "K": 8}
        assert md["tensors"]["weight"]["dims"] == {"K": 8, "N": 4}
        assert md["tensors"]["matmul#0"]["dims"] == {"M": 16, "N": 4}


class TestMemoryGuard:
    """Upfront memory guard raises MemoryError when RAM is insufficient."""

    def test_oom_raises(self):
        from unittest.mock import patch, MagicMock
        import psutil
        from solar.einsum_lib.pipeline import _check_memory

        mock_vm = MagicMock()
        mock_vm.available = 1_000_000  # ~1 MB

        shapes = {"x": (1000, 1000, 1000)}  # ~4 GB at float32
        dtypes = {"x": "fp32"}

        with patch.object(psutil, "virtual_memory", return_value=mock_vm):
            with pytest.raises(MemoryError, match="GB needed"):
                _check_memory(shapes, dtypes, use_fp64=False)

    def test_sufficient_memory_passes(self):
        from unittest.mock import patch, MagicMock
        import psutil
        from solar.einsum_lib.pipeline import _check_memory

        mock_vm = MagicMock()
        mock_vm.available = 100_000_000_000  # ~100 GB

        shapes = {"x": (10, 10)}  # tiny
        dtypes = {"x": "fp32"}

        with patch.object(psutil, "virtual_memory", return_value=mock_vm):
            _check_memory(shapes, dtypes, use_fp64=False)

    def test_estimate_bytes_correct(self):
        from solar.einsum_lib.pipeline import _estimate_allocation_bytes

        shapes = {"a": (2, 3), "b": (4,)}
        dtypes = {"a": "fp32", "b": "int64"}
        total = _estimate_allocation_bytes(shapes, dtypes, use_fp64=False)
        assert total == 2 * 3 * 4 + 4 * 8  # 24 + 32 = 56


# ---------------------------------------------------------------------------
# Capping feedback loop tests
# ---------------------------------------------------------------------------

class TestCapCrashFeedback:
    """Loop 1: capping LLM retries when proposed shapes crash the model."""

    def test_bad_shapes_retried_with_error_context(self):
        """scale_shapes_with_llm receives error_context on retry."""
        from unittest.mock import patch, MagicMock, call
        from solar.einsum_lib.pipeline import prepare_pytorch

        call_count = {"n": 0}
        good_shapes = {"x_0": (2, 4), "x_1": (4,)}
        bad_shapes = {"x_0": (2, 4), "x_1": (2,)}

        def fake_scale(source_code, input_shapes, debug=False, error_context=None, **kwargs):
            call_count["n"] += 1
            if error_context is None:
                return {}, bad_shapes, bad_shapes
            return {}, good_shapes, good_shapes

        def fake_run_ref(model, verify_shapes, fixed_inputs, input_dtypes,
                         cap_input_size, use_fp64, debug, mem_budget_bytes=None):
            if verify_shapes == bad_shapes:
                raise RuntimeError("size mismatch")
            mock_captured = MagicMock()
            mock_captured.captured_masks = {}
            mock_captured.reference_outputs = {"output": np.array([1.0])}
            mock_captured.mask_fill_values = {}
            mock_captured.captured_randn = {}
            return {"x_0": np.zeros((2, 4)), "x_1": np.zeros((4,))}, mock_captured

        with patch("solar.einsum_lib.pipeline.load_pytorch_module", return_value=MagicMock(config=None)):
            with patch("solar.einsum_lib.pipeline.get_pytorch_model_and_inputs",
                       return_value=(MagicMock(), [np.zeros((4, 4)), np.zeros((4,))])):
                with patch("solar.einsum_lib.pipeline.build_all_shapes_and_dtypes",
                           return_value=({"x_0": (4, 4), "x_1": (4,)}, {"x_0": "fp32", "x_1": "fp32"}, {}, set())):
                    with patch("solar.einsum_lib.pipeline._cap_with_llm", side_effect=fake_scale):
                        with patch("solar.einsum_lib.pipeline._run_reference", side_effect=fake_run_ref):
                            with patch("solar.einsum_lib.pipeline.Path") as MockPath:
                                MockPath.return_value.read_text.return_value = "# source"
                                with patch("solar.einsum_lib.verify.compute_sparsity", return_value=1.0):
                                    result = prepare_pytorch(
                                        "fake.py", verify=True,
                                        cap_input_size=True, debug=False,
                                    )

        assert call_count["n"] == 2
        assert result[3] is not None  # reference_outputs present

    def test_crash_gives_up_after_max_retries(self):
        """After _MAX_CAP_RETRIES crash failures, RuntimeError is raised."""
        from unittest.mock import patch, MagicMock
        from solar.einsum_lib.pipeline import prepare_pytorch

        def always_bad(source_code, input_shapes, debug=False, error_context=None, **kwargs):
            return {}, {"x": (999,)}, {"x": (999,)}

        def always_crash(model, verify_shapes, fixed_inputs, input_dtypes,
                         cap_input_size, use_fp64, debug, mem_budget_bytes=None):
            raise RuntimeError("always crashes")

        with patch("solar.einsum_lib.pipeline.load_pytorch_module", return_value=MagicMock(config=None)):
            with patch("solar.einsum_lib.pipeline.get_pytorch_model_and_inputs",
                       return_value=(MagicMock(), [np.zeros((4,))])):
                with patch("solar.einsum_lib.pipeline.build_all_shapes_and_dtypes",
                           return_value=({"x": (4,)}, {"x": "fp32"}, {}, set())):
                    with patch("solar.einsum_lib.pipeline._cap_with_llm", side_effect=always_bad):
                        with patch("solar.einsum_lib.pipeline._run_reference", side_effect=always_crash):
                            with patch("solar.einsum_lib.pipeline.Path") as MockPath:
                                MockPath.return_value.read_text.return_value = "# source"
                                with patch("solar.einsum_lib.verify.compute_sparsity", return_value=1.0):
                                    with pytest.raises(RuntimeError, match="after 2 attempts"):
                                        prepare_pytorch(
                                            "fake.py", verify=True,
                                            cap_input_size=True, debug=False,
                                        )


class TestCapEscalation:
    """Loop 2: translation failure escalates to re-capping.

    Fully offline: TranslationAgent.translate is mocked, so no provider
    call can happen — but the agent CONSTRUCTOR still resolves
    credentials.  An explicit stub key file keeps these fixtures
    independent of host credential discovery (OPENAI_API_KEY),
    which made them fail on isolated hosts."""

    @staticmethod
    def _stub_key_file(tmp_path):
        p = tmp_path / "api_key_stub.txt"
        p.write_text("offline-test-stub-key\n")
        return str(p)

    def _find_example_file(self):
        """Locate any .py file for translate_file's exists() check."""
        candidates = [
            Path(__file__).parent.parent / "examples" / "Matmul" / "Matmul.py",
            Path(__file__),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        pytest.skip("No example file found")

    def test_recap_on_translation_failure(self, tmp_path):
        """translate_file re-caps when translation exhausts retries."""
        from unittest.mock import patch, MagicMock
        from solar.einsum_lib.pipeline import translate_file, _DEFAULT_MAX_CAP_ROUNDS
        from solar.einsum_lib.agent import TranslationResult, TimingInfo

        example = self._find_example_file()
        prepare_call_count = {"n": 0}

        def fake_prepare(file_path, verify, cap_input_size=False,
                         use_fp64=False, debug=False, cap_error_context=None, mem_budget_bytes=None):
            prepare_call_count["n"] += 1
            return (
                "# source", {"x": (2, 2)}, {"x": "fp32"},
                {"output": np.array([1.0])}, {"x": np.zeros((2, 2))},
                {}, {}, set(), {"x": (2, 2)}, None,
            )

        translate_call_count = {"n": 0}

        def fake_translate(self, **kwargs):
            translate_call_count["n"] += 1
            if translate_call_count["n"] < 2:
                return TranslationResult(
                    code="# bad", verified=False, attempts=5,
                    verification_error="Output mismatch",
                    timing=TimingInfo(),
                )
            return TranslationResult(
                code="# good", verified=True, attempts=1,
                timing=TimingInfo(),
            )

        with patch("solar.einsum_lib.pipeline.prepare_pytorch", side_effect=fake_prepare):
            with patch.object(
                __import__("solar.einsum_lib.agent", fromlist=["TranslationAgent"]).TranslationAgent,
                "translate", fake_translate,
            ):
                result = translate_file(
                    file_path=example,
                    verify=True, cap_input_size=True,
                    max_cap_rounds=2,
                    api_key_file=self._stub_key_file(tmp_path),
                )

        assert result.verified
        assert prepare_call_count["n"] == 2
        assert translate_call_count["n"] == 2

    def test_gives_up_after_max_cap_rounds(self, tmp_path):
        """translate_file returns FAILED after exhausting cap rounds."""
        from unittest.mock import patch, MagicMock
        from solar.einsum_lib.pipeline import translate_file
        from solar.einsum_lib.agent import TranslationResult, TimingInfo

        example = self._find_example_file()

        def fake_prepare(file_path, verify, cap_input_size=False,
                         use_fp64=False, debug=False, cap_error_context=None, mem_budget_bytes=None):
            return (
                "# source", {"x": (2, 2)}, {"x": "fp32"},
                {"output": np.array([1.0])}, {"x": np.zeros((2, 2))},
                {}, {}, set(), {"x": (2, 2)}, None,
            )

        def fake_translate(self, **kwargs):
            return TranslationResult(
                code="# always bad", verified=False, attempts=5,
                verification_error="Output mismatch",
                timing=TimingInfo(),
            )

        with patch("solar.einsum_lib.pipeline.prepare_pytorch", side_effect=fake_prepare):
            with patch.object(
                __import__("solar.einsum_lib.agent", fromlist=["TranslationAgent"]).TranslationAgent,
                "translate", fake_translate,
            ):
                result = translate_file(
                    file_path=example,
                    verify=True, cap_input_size=True,
                    max_cap_rounds=2,
                    api_key_file=self._stub_key_file(tmp_path),
                )

        assert not result.verified
        assert result.code == "# always bad"
