# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""numba lowering must decline a Dim loop whose bound it cannot establish.

Lowering such a loop to ``range(0)`` silently skips the body and returns
all-zero outputs, while the DSL contract (``Dim`` / ``_registry.get_size``)
raises for an unregistered dim.  ``numba_lower`` returns None instead so the
interpreter path enforces the contract.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("numba")
from solar.einsum_lib.numba_accel import numba_lower  # noqa: E402

UNRESOLVED = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    for b in Dim("B"):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("WRONG_DIM"):
                    C[b, m, n] += A[b, m, k] * W[k, n]
    return C
'''

RESOLVED = UNRESOLVED.replace("WRONG_DIM", "K")


def test_unresolved_dim_declines_lowering():
    assert numba_lower(UNRESOLVED) is None


def test_resolved_dim_still_lowers():
    lowered = numba_lower(RESOLVED)
    assert lowered is not None and "range" in lowered


# ---------------------------------------------------------------------------
# Review follow-ups: explicit size= validation, for/else, nested kernels
# ---------------------------------------------------------------------------

def _with_k_loop(iter_src: str, parallel_outer: bool = False) -> str:
    outer = 'Dim("B", parallel=True)' if parallel_outer else 'Dim("B")'
    return f'''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    for b in {outer}:
        for m in Dim("M"):
            for n in Dim("N"):
                for k in {iter_src}:
                    C[b, m, n] += A[b, m, k] * W[k, n]
    return C
'''


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bad", ["None", '"unknown"', "1.5", "True", "-3"])
def test_unresolvable_explicit_size_declines(bad, parallel):
    # WRONG_DIM is registered nowhere, so size=None cannot fall back either;
    # the other literals can never be a valid extent.
    flag = ", parallel=True" if parallel else ""
    code = _with_k_loop(f'Dim("WRONG_DIM", size={bad}{flag})')
    assert numba_lower(code) is None


@pytest.mark.parametrize("parallel", [False, True])
def test_size_none_uses_the_registered_dim_size(parallel):
    flag = ", parallel=True" if parallel else ""
    lowered = numba_lower(_with_k_loop(f'Dim("K", size=None{flag})'))
    assert lowered is not None
    assert "None" not in lowered.split("def matmul", 1)[1]
    assert ("prange" if parallel else "range") in lowered


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("good", ["4", "A.shape[2]"])
def test_valid_explicit_size_still_lowers(good, parallel):
    flag = ", parallel=True" if parallel else ""
    assert numba_lower(_with_k_loop(f'Dim("WRONG_DIM", size={good}{flag})')) is not None


def test_unresolved_dim_in_for_else_declines():
    code = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def f(A: Tensor("B,M"), W: Tensor("M,N")) -> Tensor("B,N"):
    C = Tensor(B=A.shape[0], N=W.shape[1])
    for b in Dim("B"):
        for n in Dim("N"):
            for m in Dim("M"):
                C[b, n] += A[b, m] * W[m, n]
        else:
            for x in Dim("WRONG_DIM"):
                C[b, 0] += 1.0
    return C
'''
    assert numba_lower(code) is None


def test_kernel_defined_inside_another_function_declines():
    code = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M")

    @kernel
    def inner(A: Tensor("B,M")) -> Tensor("B,M"):
        C = Tensor(B=A.shape[0], M=A.shape[1])
        for b in Dim("B"):
            for m in Dim("WRONG_DIM"):
                C[b, m] = A[b, m]
        return C

    return {"output": inner(x).data}
'''
    assert numba_lower(code) is None


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bad", ["-0.0", "+0.0", "+1.5", "-2", "(1, 2)"])
def test_signed_or_composite_literal_bounds_decline(bad, parallel):
    flag = ", parallel=True" if parallel else ""
    assert numba_lower(_with_k_loop(f'Dim("MISSING", size={bad}{flag})')) is None


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("good", ["0", "-0", "+3"])
def test_signed_integer_literal_bounds_still_lower(good, parallel):
    flag = ", parallel=True" if parallel else ""
    assert numba_lower(_with_k_loop(f'Dim("MISSING", size={good}{flag})')) is not None


# ---------------------------------------------------------------------------
# Second independent review: exact bounds, numerics, caller-level fallback,
# wider nesting placements, documented zero-trip policy
# ---------------------------------------------------------------------------

def _k_loop_line(lowered: str) -> str:
    # a parallel Dim iterates the fresh name ``_k_prange`` and rebinds
    # ``k = np.intp(_k_prange)`` (SHAPE0-205-NUMBA-FLOAT-INDEX-1)
    return next(ln.strip() for ln in lowered.splitlines()
                if ln.strip().startswith(("for k in", "for _k_prange in")))


@pytest.mark.parametrize("src,expected", [
    ('Dim("K")', "for k in range(A.shape[2]):"),
    ('Dim("K", size=None)', "for k in range(A.shape[2]):"),
    ('Dim("K", size=None, parallel=True)', "for _k_prange in numba.prange(A.shape[2]):"),
    ('Dim("WRONG_DIM", size=4)', "for k in range(4):"),
    ('Dim("WRONG_DIM", size=A.shape[2], parallel=True)',
     "for _k_prange in numba.prange(A.shape[2]):"),
])
def test_lowered_bound_is_exactly_the_intended_extent(src, expected):
    assert _k_loop_line(numba_lower(_with_k_loop(src))) == expected


MATMUL_SHAPES = {"x": (2, 3, 5), "w": (5, 4)}
RUN = '''
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight")
    return {"output": matmul(x, w).data}
'''


def _runnable(iter_src: str, parallel_outer: bool = False) -> str:
    return _with_k_loop(iter_src, parallel_outer).replace(
        "import Tensor, Dim, kernel", "import Tensor, Dim, kernel, clear_dims") + RUN


# parallel=True goes on the OUTER b loop: parallelising the k reduction would
# race on C[b, m, n] += ... for any DSL, so numerics are checked there.
@pytest.mark.parametrize("src,parallel_outer", [
    ('Dim("K")', False), ('Dim("K", size=None)', False),
    ('Dim("K", size=5)', False), ('Dim("K", size=None)', True),
    ('Dim("K", size=5)', True),
])
def test_lowered_programs_match_numpy_and_are_not_silent_zeros(src, parallel_outer):
    from solar.einsum_lib.agent import execute_translated
    out, inp = execute_translated(_runnable(src, parallel_outer),
                                  MATMUL_SHAPES, seed=3)
    ref = np.einsum("bmk,kn->bmn", inp["x"], inp["w"])
    assert np.abs(ref).sum() > 0
    np.testing.assert_allclose(out["output"], ref, rtol=1e-6, atol=1e-9)


def test_unregistered_dim_raises_through_the_real_caller():
    # numba declines, the interpreter enforces the DSL contract: an error, never
    # default (all-zero) outputs.
    from solar.einsum_lib.agent import execute_translated
    with pytest.raises(RuntimeError, match="not registered"):
        execute_translated(_runnable('Dim("WRONG_DIM")'), MATMUL_SHAPES)


def test_kernel_inside_async_wrapper_declines():
    code = '''
from solar.einsum_lib import Tensor, Dim, kernel

async def factory():
    @kernel
    def inner(A: Tensor("B,M")) -> Tensor("B,M"):
        C = Tensor(B=A.shape[0], M=A.shape[1])
        for b in Dim("B"):
            for m in Dim("M"):
                C[b, m] = A[b, m]
        return C
    return inner
'''
    assert numba_lower(code) is None


@pytest.mark.parametrize("wrapper", ["if True:", "class Holder:"])
def test_kernel_under_module_level_block_declines(wrapper):
    code = f'''
from solar.einsum_lib import Tensor, Dim, kernel

{wrapper}
    @kernel
    def inner(A: Tensor("B,M")) -> Tensor("B,M"):
        C = Tensor(B=A.shape[0], M=A.shape[1])
        for b in Dim("B"):
            for m in Dim("M"):
                C[b, m] = A[b, m]
        return C
'''
    assert numba_lower(code) is None


def test_zero_trip_outer_loop_does_not_excuse_an_unresolved_inner_dim():
    # Documented conservative policy: the lowerer does not reason about
    # reachability, so an unresolved Dim inside a size=0 body still declines
    # (interpreter fallback, correct output, only slower).
    code = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def f(A: Tensor("B,M")) -> Tensor("B,M"):
    C = Tensor(B=A.shape[0], M=A.shape[1])
    for i in Dim("Z", size=0):
        for j in Dim("MISSING"):
            C[0, 0] += 1.0
    return C
'''
    assert numba_lower(code) is None
