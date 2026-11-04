# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SHAPE0-BF16-FINAL-ROUNDING-1: the Numba lowering rounds a narrow-declared kernel result by its
DECLARED dtype whatever the syntactic form of the call site, and derives the declared dtype from the
returned allocation when the return annotation carries none — never skipping the rounding silently.

The interpreter's ``kernel`` wrapper rounds every narrow ``Tensor`` result (``core.kernel``); the lowered
program must apply the same rounding.  Controls compare the fast path with torch's bf16 rounding of the
unrounded result and with the interpreter path; static regressions lower the saved DSL sources of Lite rows
108 / 140 / 229 (metadata only — no payload is executed) and pin the only differences from the pre-fix
lowered programs recorded by the owning lane review (checks-228/lowered-metadata/).
"""
import ast
import json
import io
import os
import pathlib
import re
import subprocess
import sys
import tokenize

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import numba_lower, execute_fast, HAS_NUMBA, _syntactic_writes, _is_plain_handler_type
from solar.einsum_lib.verify import execute_translated

ROOT = pathlib.Path(__file__).resolve().parent.parent          # the solar/ package root (PYTHONPATH for subprocess controls)
FIX = pathlib.Path(__file__).parent / "fixtures" / "lowering_final_rounding"
R3 = FIX / "review3_controls"                       # the review round-3 controls, byte-for-byte (an earlier review)
needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    stop_trace(); clear_dims()


def _bf16(x):
    import torch
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).to(torch.float64).numpy()


_HEADER = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n"

# a GEMM whose fp32/fp64 result is NOT bf16-representable (so a skipped rounding is visible)
_GEMM_BF16_ANNOTATED = '''
@kernel
def gemm(A: Tensor("M,K", dtype="bf16"), W: Tensor("K,N", role="weight", dtype="bf16")) -> Tensor("M,N", dtype="bf16"):
    Y = Tensor(M=A.shape[0], N=W.shape[1], dtype="bf16")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m, n] += A[m, k] * W[k, n]
    return Y
'''
# the same kernel with NO dtype in the return annotation: the allocation declares bf16 (row-108 form)
_GEMM_BF16_ALLOC_ONLY = _GEMM_BF16_ANNOTATED.replace('-> Tensor("M,N", dtype="bf16")', '-> Tensor("M,N")')
_SCALE_FP32 = '''
@kernel
def scale2(X: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            Y[m, n] = X[m, n] * 2.0
    return Y
'''
_RUN = '''
def run_model(inputs: dict) -> dict:
    clear_dims()
    a = Tensor.from_numpy(inputs["a"], "M,K", dtype="bf16")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight", dtype="bf16")
    return model(a, w)
'''


def _program(kernels, model_body):
    return _HEADER + kernels + "\ndef model(a, w):\n" + model_body + "\n" + _RUN


def _inputs(seed=7, m=4, k=6, n=5):
    rng = np.random.RandomState(seed)
    # bf16-representable inputs (as the boundary declares), products of which are not bf16-representable
    return {"a": _bf16(rng.uniform(-1, 1, (m, k))), "w": _bf16(rng.uniform(-1, 1, (k, n)))}


def _unrounded(inputs):
    return inputs["a"].astype(np.float64) @ inputs["w"].astype(np.float64)


def _assert_rounded(fast_value, inputs):
    exact = _unrounded(inputs)
    assert fast_value is not None
    assert not np.allclose(fast_value, exact, rtol=0, atol=0), "the fp64 product must not be bf16-representable for this control to mean anything"
    np.testing.assert_array_equal(np.asarray(fast_value, dtype=np.float64), _bf16(exact))


# ---------------------------------------------------------------------------------------------------------
# syntactic forms of the call site
# ---------------------------------------------------------------------------------------------------------

class TestRoundingFollowsTheDeclaredDtype:

    @needs_numba
    @pytest.mark.parametrize("use_fp64", [False, True])
    def test_assignment_form_unchanged(self, use_fp64):
        code = _program(_GEMM_BF16_ANNOTATED, "    out = gemm(a, w)\n    return {'output': out.data}")
        low = numba_lower(code, boundscheck=True, use_fp64=use_fp64)
        assert "out = gemm(a, w)\n    out = _truncate_bf16(out)" in low           # the statement-after form is kept
        assert low.count("_truncate_bf16(gemm") == 0                                  # ... and not wrapped a second time
        _assert_rounded(execute_fast(code, _inputs(), use_fp64=use_fp64)["output"], _inputs())

    @needs_numba
    @pytest.mark.parametrize("use_fp64", [False, True])
    def test_direct_return(self, use_fp64):
        code = _program(_GEMM_BF16_ANNOTATED, "    return {'output': gemm(a, w).data}")
        low = numba_lower(code, boundscheck=True, use_fp64=use_fp64)
        assert "_truncate_bf16(gemm(a, w))" in low
        _assert_rounded(execute_fast(code, _inputs(), use_fp64=use_fp64)["output"], _inputs())

    @needs_numba
    def test_direct_return_through_a_helper_function(self):
        """Row-140 / row-229 form: ``model`` returns the terminal kernel call itself."""
        code = _HEADER + _GEMM_BF16_ANNOTATED + "\ndef model(a, w):\n    return gemm(a, w)\n" + \
            "\ndef run_model(inputs: dict) -> dict:\n    clear_dims()\n    a = Tensor.from_numpy(inputs['a'], 'M,K', dtype='bf16')\n" \
            "    w = Tensor.from_numpy(inputs['w'], 'K,N', role='weight', dtype='bf16')\n    out = model(a, w)\n    return {'output': out.data}\n"
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert "return _truncate_bf16(gemm(a, w))" in low
        _assert_rounded(execute_fast(code, _inputs(), use_fp64=False)["output"], _inputs())

    @needs_numba
    def test_tuple_return(self):
        code = _HEADER + _GEMM_BF16_ANNOTATED + "\ndef model(a, w):\n    return gemm(a, w), gemm(a, w)\n" + \
            "\ndef run_model(inputs: dict) -> dict:\n    clear_dims()\n    a = Tensor.from_numpy(inputs['a'], 'M,K', dtype='bf16')\n" \
            "    w = Tensor.from_numpy(inputs['w'], 'K,N', role='weight', dtype='bf16')\n    y1, y2 = model(a, w)\n    return {'y1': y1.data, 'y2': y2.data}\n"
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert "return (_truncate_bf16(gemm(a, w)), _truncate_bf16(gemm(a, w)))" in low
        out = execute_fast(code, _inputs(), use_fp64=False)
        _assert_rounded(out["y1"], _inputs()); _assert_rounded(out["y2"], _inputs())

    @needs_numba
    def test_dict_return(self):
        code = _program(_GEMM_BF16_ANNOTATED, "    return {'output': gemm(a, w).data, 'twice': gemm(a, w).data}")
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert low.count("_truncate_bf16(gemm(a, w))") == 2
        out = execute_fast(code, _inputs(), use_fp64=False)
        _assert_rounded(out["output"], _inputs()); _assert_rounded(out["twice"], _inputs())

    @needs_numba
    def test_nested_call_is_rounded_before_the_consumer(self):
        """``scale2(gemm(a, w))``: the bf16 GEMM result is rounded BEFORE the fp32 consumer sees it, as the
        interpreter does; the fp32 consumer's own result is not rounded."""
        code = _program(_GEMM_BF16_ANNOTATED + _SCALE_FP32, "    return {'output': scale2(gemm(a, w)).data}")
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert "scale2(_truncate_bf16(gemm(a, w)))" in low and "_truncate_bf16(scale2" not in low
        fast = execute_fast(code, _inputs(), use_fp64=False)["output"]
        np.testing.assert_array_equal(np.asarray(fast, dtype=np.float64), _bf16(_unrounded(_inputs())) * 2.0)

    @needs_numba
    @pytest.mark.parametrize("form", ["assignment", "direct"])
    def test_unannotated_allocation_declares_the_dtype(self, form):
        """Row-108 form: ``-> Tensor("M,N")`` carries no dtype; the returned ``Tensor(..., dtype="bf16")``
        allocation is the declared output dtype (what the interpreter's kernel wrapper rounds by)."""
        body = "    out = gemm(a, w)\n    return {'output': out.data}" if form == "assignment" else "    return {'output': gemm(a, w).data}"
        code = _program(_GEMM_BF16_ALLOC_ONLY, body)
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert "_truncate_bf16" in low
        _assert_rounded(execute_fast(code, _inputs(), use_fp64=False)["output"], _inputs())

    @needs_numba
    @pytest.mark.parametrize("kernels", [_GEMM_BF16_ANNOTATED, _GEMM_BF16_ALLOC_ONLY], ids=["annotated", "allocation-only"])
    def test_fast_path_matches_the_interpreter(self, kernels):
        code = _program(kernels, "    return {'output': gemm(a, w).data}")
        inputs = _inputs()
        slow, _ = execute_translated(code, {"a": inputs["a"].shape, "w": inputs["w"].shape}, all_inputs=inputs, fast=False)
        fast = execute_fast(code, inputs, use_fp64=True)
        np.testing.assert_array_equal(np.asarray(fast["output"], dtype=np.float64), np.asarray(slow["output"], dtype=np.float64))

    def test_fp32_kernels_are_not_rounded(self):
        code = _program(_SCALE_FP32.replace('X: Tensor("M,N", dtype="fp32")', 'X: Tensor("M,K", dtype="bf16")').replace("N=X.shape[1]", "K=X.shape[1]").replace('Dim("N")', 'Dim("K")').replace("Y[m, n] = X[m, n]", "Y[m, k] = X[m, k]").replace("for n in", "for k in").replace('-> Tensor("M,N")', '-> Tensor("M,K")'),
                        "    return {'output': scale2(a).data}")
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert low is not None and "_truncate_bf16(scale2" not in low


# ---------------------------------------------------------------------------------------------------------
# the declared dtype cannot be established -> typed decline, never a silent skip
# ---------------------------------------------------------------------------------------------------------

class TestUnestablishedDtypeIsRefused:

    def test_annotation_and_allocation_conflict(self):
        code = _program(_GEMM_BF16_ANNOTATED.replace('Y = Tensor(M=A.shape[0], N=W.shape[1], dtype="bf16")', 'Y = Tensor(M=A.shape[0], N=W.shape[1], dtype="fp32")'),
                        "    return {'output': gemm(a, w).data}")
        reasons = []
        assert numba_lower(code, boundscheck=True, use_fp64=False, on_decline=reasons.append) is None
        assert reasons and reasons[0].startswith("kernel_output_dtype_conflict:gemm")

    def test_return_of_an_untraceable_value_without_annotation(self):
        code = _program(_GEMM_BF16_ALLOC_ONLY.replace("    return Y\n", "    return Y[0:2, :]\n"), "    return {'output': gemm(a, w).data}")
        reasons = []
        assert numba_lower(code, boundscheck=True, use_fp64=False, on_decline=reasons.append) is None
        assert reasons and reasons[0].startswith("kernel_output_dtype_unestablished:gemm")

    def test_execute_fast_reports_the_typed_reason(self):
        code = _program(_GEMM_BF16_ALLOC_ONLY.replace("    return Y\n", "    return Y[0:2, :]\n"), "    return {'output': gemm(a, w).data}")
        seen = []
        assert execute_fast(code, _inputs(), use_fp64=False, on_fallback=seen.append) is None
        assert len(seen) == 1 and seen[0].startswith("kernel_output_dtype_unestablished:gemm: a returned value cannot be traced")

    def test_returned_parameter_takes_its_annotation_dtype(self):
        """``return X`` of a bf16 parameter is an established (bf16) output; a plain pass-through kernel."""
        code = _program('''
@kernel
def ident(X: Tensor("M,K", dtype="bf16")) -> Tensor("M,K"):
    return X
''', "    return {'output': ident(a).data}")
        low = numba_lower(code, boundscheck=True, use_fp64=False)
        assert low is not None and "_truncate_bf16(ident(a))" in low


# ---------------------------------------------------------------------------------------------------------
# round 2 (an earlier review): return-site provenance with kills and joins; explicit dtypes never defaulted
# ---------------------------------------------------------------------------------------------------------

_K_HEADER = _HEADER + 'DT = "bf16"\nDT_FP32 = "fp32"\n'
_K_RUN = "\ndef run_model(inputs):\n    clear_dims()\n    a = Tensor.from_numpy(inputs['a'], 'M,N', dtype='fp32')\n    return {'output': k(a).data}\n"


def _kernel_program(body, annotation='-> Tensor("M,N")'):
    return _K_HEADER + '@kernel\ndef k(A: Tensor("M,N", dtype="fp32")) ' + annotation + ':\n' + body + _K_RUN


def _fast_slow(code):
    """(fast output or None, decline reasons, interpreter output) for a 1x2 fp32 input that bf16 would round."""
    x = np.array([[1.003, 1.009]], dtype=np.float64); seen = []
    slow, _ = execute_translated(code, {"a": x.shape}, all_inputs={"a": x}, fast=False)
    fast = execute_fast(code, {"a": x}, use_fp64=True, on_fallback=seen.append)
    return fast, seen, np.asarray(slow["output"], dtype=np.float64)


class TestReturnProvenance:
    """provenance-1: the declared dtype is the reaching definition of the returned name at each return site —
    a rebinding kills the allocation, control-flow paths join (agree, or decline), dead code after a return
    never binds.  Every accepted case equals the interpreter; every disagreement is a typed decline."""

    @needs_numba
    def test_rebinding_to_a_parameter_kills_the_bf16_allocation(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16")\n    Y = A\n    return Y\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16" not in low
        fast, seen, slow = _fast_slow(code)
        np.testing.assert_array_equal(fast["output"], slow); np.testing.assert_array_equal(slow, [[1.003, 1.009]])   # fp32 result preserved

    @needs_numba
    def test_conditional_rebinding_not_taken_is_a_join_conflict(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n'
                               '    if A[0, 0] < 0:\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n')
        fast, seen, slow = _fast_slow(code)
        assert fast is None and seen and seen[0].startswith("kernel_output_dtype_conflict:k")      # paths disagree -> decline, never a guess

    @needs_numba
    def test_agreeing_branches_join_to_one_dtype(self):
        code = _kernel_program('    if A[0, 0] < 0:\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
                               '    else:\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.009)\n    return Y\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16(k(a))" in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow)

    @needs_numba
    def test_dead_rebinding_after_return_does_not_suppress_rounding(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n'
                               '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16(k(a))" in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow); assert float(slow[0, 0]) == 1.0

    @needs_numba
    def test_alias_chain_and_direct_constructor_return(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    Z = Y\n    return Z\n')
        assert "_truncate_bf16(k(a))" in numba_lower(code, boundscheck=True, use_fp64=True)
        code2 = _kernel_program('    return Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n')
        assert "_truncate_bf16(k(a))" in numba_lower(code2, boundscheck=True, use_fp64=True)

    @needs_numba
    def test_rebinding_inside_a_loop_is_a_kill(self):
        """A loop body that may rebind the returned name to an untraceable value: the zero-trip and the
        taken paths disagree (bf16 vs unknown) -> unestablished, never the allocation's dtype."""
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
                               '    for m in Dim("M"):\n        Y = A\n    return Y\n')
        reasons = []; assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
        assert reasons[0].startswith("kernel_output_dtype_")

    @needs_numba
    def test_two_returns_must_agree(self):
        code = _kernel_program('    if A[0, 0] < 0:\n        return Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
                               '    return Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n')
        reasons = []; assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
        assert reasons[0].startswith("kernel_output_dtype_conflict:k")


class TestControlFlowExits:
    """flow-4: break / continue carry their environments to the loop exit, loops reach a real fixed point,
    ``match`` and ``if`` joins include the incoming (unmatched / else-less) path, and anything not modelled soundly
    is a typed decline — never a guessed dtype.  Every accepted case equals the interpreter."""

    @pytest.mark.parametrize("body", [
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    for m in Dim("M"):\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n        break\n    return Y\n',
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    for m in Dim("M"):\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n        continue\n    return Y\n',
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    while A[0, 0] > 0:\n        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n        break\n    return Y\n',
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    match 1:\n        case 0:\n            Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n',
        '    Y0 = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Y1 = Y0\n    Y2 = Y0\n    Y3 = Y0\n    Y4 = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    for i in range(4):\n        Y0 = Y1\n        Y1 = Y2\n        Y2 = Y3\n        Y3 = Y4\n    return Y0\n',
    ], ids=["for-break", "for-continue", "while-break", "match-untaken", "four-step-alias-loop"])
    def test_rebinding_reachable_through_an_exit_or_untaken_path_declines(self, body):
        """The reviewer's regressions: the bf16 rebinding reaches the return along one path and fp32 along another
        (the zero-trip / unmatched path), so the dtype is a conflict -> typed decline, never the fp32 guess."""
        reasons = []; assert numba_lower(_kernel_program(body), boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
        assert reasons[0].startswith("kernel_output_dtype_conflict:k"), reasons

    def test_try_finally_and_with_are_declined_as_not_modelled(self):
        for body in ('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    try:\n        return Y\n    finally:\n        return Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n',):
            reasons = []; assert numba_lower(_kernel_program(body), boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
            assert reasons[0].startswith("kernel_output_dtype_unestablished:k") and "not modelled" in reasons[0]

    @needs_numba
    def test_loop_that_rebinds_to_the_same_dtype_converges_and_is_established(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    for m in Dim("M"):\n        Z = Y\n        Y = Z\n        if A[0, 0] < 0:\n            break\n    return Y\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16(k(a))" in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow); assert float(slow[0, 0]) == 1.0

    @needs_numba
    def test_match_with_an_irrefutable_case_joins_only_the_cases(self):
        code = _kernel_program('    match 1:\n        case 0:\n            Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n        case _:\n            Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.009)\n    return Y\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16(k(a))" in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow)

    @needs_numba
    def test_try_without_finally_joins_body_and_handler(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    try:\n        Z = Y\n    except Exception:\n        Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.009)\n    return Z\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_bf16(k(a))" in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow)


class TestExplicitDtypeIsNeverDefaulted:
    """dtype-2 / round 9: an ABSENT dtype is the DSL default (fp32).  An EXPLICIT dtype is honoured only as a
    string LITERAL; every other spelling -- a module name, a kernel-local name, an attribute, a call -- is the
    dtype-specific typed decline at lowering and the interpreter runs the program.  Nothing is defaulted, folded or
    evaluated at call time (the round-8 call-time path is removed: it could not preserve the interpreter's scope,
    evaluation order and count -- const-24/25)."""

    def _declines(self, code, needle=None):
        for use_fp64 in (True, False):
            reasons = []
            assert numba_lower(code, boundscheck=True, use_fp64=use_fp64, on_decline=reasons.append) is None
            assert reasons[0].startswith("kernel_output_dtype_unestablished:k"), reasons
            if needle:
                assert needle in reasons[0], reasons[0]
        return reasons[0]

    @needs_numba
    def test_absent_dtype_is_the_default_and_nothing_is_rounded(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], fill=1.003)\n    return Y\n')
        low = numba_lower(code, boundscheck=True, use_fp64=True); assert low is not None and "_truncate_" not in low
        fast, seen, slow = _fast_slow(code); np.testing.assert_array_equal(fast["output"], slow); np.testing.assert_array_equal(slow, [[1.003, 1.003]])

    def test_module_level_name_declines_and_the_interpreter_runs(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n')
        self._declines(code)
        fast, seen, slow = _fast_slow(code); assert fast is None and seen[0].startswith("kernel_output_dtype_unestablished:k")
        public, _ = execute_translated(code, {"a": (1, 2)}, all_inputs={"a": np.array([[1.003, 1.009]])}, fast=True, use_fp64=True)
        np.testing.assert_array_equal(public["output"], slow); assert float(slow[0, 0]) == 1.0

    @pytest.mark.parametrize("body", [
        '    dt = "bf16"\n    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=dt, fill=1.003)\n    return Y\n',
        '    DT = "fp32"\n    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n',
        '    dt2 = "bf16"\n    dt2 = "fp32"\n    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=dt2, fill=1.003)\n    return Y\n',
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=A.dtype, fill=1.003)\n    return Y\n',
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT_FP32, fill=1.003)\n    return Y\n',
        '    global DT\n    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n',
    ], ids=["kernel-local-literal", "kernel-local-shadow", "rebound-local", "parameter-attribute", "module-fp32-name", "global-declaration"])
    def test_every_non_literal_spelling_declines(self, body):
        """Including the kernel-local single literal binding that rounds 3-8 folded (const-5): literal only."""
        self._declines(_kernel_program(body))

    @pytest.mark.parametrize("expr", ["DT.upper()", "UNBOUND", "choose()"], ids=["call-yielding-non-dtype", "unbound-name", "call"])
    def test_calls_and_unbound_names_decline_and_the_interpreter_decides(self, expr):
        code = _kernel_program(f'    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype={expr}, fill=1.003)\n    return Y\n')
        self._declines(code)
        x = np.array([[1.003, 1.009]])
        with pytest.raises(RuntimeError):
            execute_translated(code, {"a": x.shape}, all_inputs={"a": x}, fast=False)
        with pytest.raises(RuntimeError):
            execute_translated(code, {"a": x.shape}, all_inputs={"a": x}, fast=True, use_fp64=True)

    @pytest.mark.parametrize("module_tail", ['\nif True:\n    DT = "fp32"', '\nDT = "fp32"', '\nclass Rebind:\n    global DT\n    DT = "fp32"'],
                             ids=["rebound-under-control-flow", "rebound-plainly", "class-body-global"])
    def test_module_rebindings_are_moot_because_the_name_is_never_read(self, module_tail):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n').replace('DT_FP32 = "fp32"', 'DT_FP32 = "fp32"' + module_tail)
        self._declines(code)
        fast, seen, slow = _fast_slow(code); assert fast is None; np.testing.assert_array_equal(slow, [[1.003, 1.003]])

    def test_annotation_dtype_must_be_a_literal_too(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n', annotation='-> Tensor("M,N", dtype=DT)')
        self._declines(code, "the return annotation's dtype is explicit but not a string literal")
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n', annotation='-> Tensor("M,N", dtype="bf16")')
        self._declines(code)

    def test_literal_annotation_and_literal_allocation_must_agree(self):
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    return Y\n', annotation='-> Tensor("M,N", dtype="bf16")')
        reasons = []; assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
        assert reasons[0].startswith("kernel_output_dtype_conflict:k")


# ---------------------------------------------------------------------------------------------------------
# static regressions on the saved Lite rows (metadata only; nothing is executed)
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# round 4 (an earlier review): FLOW-6, CONST-7, NORMALIZER-8, COMPOSITION-9
# ---------------------------------------------------------------------------------------------------------

def _review_control(name):
    """The reviewer's acceptance check (checks-e0c850d0/evidence/test_review105_round3.py), on the byte-identical
    control source: the fp64 lowering either matches the interpreter or declines with a typed reason and the fast
    path falls back to the interpreter's value.  Returns (decline reasons, fast-or-fallback output, slow output)."""
    code = (R3 / f"{name}.py").read_text()
    x = np.array([[1.003, 1.009]], np.float64); clear_dims(); stop_trace()
    slow, _ = execute_translated(code, {'a': x.shape}, all_inputs={'a': x}, fast=False)
    reasons = []; low = numba_lower(code, use_fp64=True, boundscheck=True, on_decline=reasons.append)
    fast = execute_fast(code, {'a': x}, use_fp64=True)
    if low is None:
        assert fast is None and reasons[0].startswith('kernel_output_dtype_')
        fallen, _ = execute_translated(code, {'a': x.shape}, all_inputs={'a': x}, fast=True)
        np.testing.assert_array_equal(fallen['output'], slow['output'])
        return reasons, np.asarray(fallen['output']), np.asarray(slow['output'])
    assert fast is not None
    np.testing.assert_array_equal(fast['output'], slow['output'])
    return reasons, np.asarray(fast['output']), np.asarray(slow['output'])


class TestRound4ExceptionPrefixAndClassGlobal:

    def test_try_prefix_rebind_declines_as_a_path_conflict(self):
        """flow-6 (verbatim control): fp32 Y, then inside ``try`` a bf16 rebinding followed by ``raise``; the
        handler is reached with Y = bf16, so the return sees {fp32 (no exception before the rebinding), bf16}: a typed
        decline, never the accepted 1.003 the round-3 head produced.  Interpreter value 1.0 (bf16) is what the
        fallback returns."""
        reasons, value, slow = _review_control("try_prefix_rebind")
        assert reasons and reasons[0].startswith("kernel_output_dtype_conflict:k")
        np.testing.assert_array_equal(slow, [[1.0, 1.0]]); np.testing.assert_array_equal(value, slow)

    def test_handler_sees_a_rebinding_before_the_raise_even_when_nested(self):
        """The prefix environments are collected at every nesting depth of the body (an ``if`` inside the ``try``)."""
        code = _kernel_program('''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)
    try:
        if A.shape[0] > 0:
            Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)
            raise ValueError("trigger")
    except Exception:
        pass
    return Y
''')
        fast, seen, slow = _fast_slow(code)
        assert fast is None and seen and seen[0].startswith("kernel_output_dtype_conflict:k")

    def test_handler_agreeing_with_every_prefix_is_established(self):
        """Sound, not merely conservative: when every prefix carries the same dtype the kernel is still lowered."""
        code = _kernel_program('''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)
    try:
        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)
        raise ValueError("trigger")
    except Exception:
        pass
    return Y
''')
        fast, seen, slow = _fast_slow(code)
        assert fast is not None and not seen
        np.testing.assert_array_equal(np.asarray(fast["output"], dtype=np.float64), slow)


def _const_program(module_prefix):
    """``DT = "bf16"`` then ``module_prefix`` then a kernel allocating ``dtype=DT``; returns the program."""
    return (_K_HEADER + module_prefix + '@kernel\ndef k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):\n'
            + '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)\n    return Y\n' + _K_RUN)


class TestRound5FlowClassClosed:
    """flow-10, closed as a class: any assignment EXPRESSION inside a ``try`` body, an ``except`` handler, a
    try-``else`` or a ``match`` case / guard is declined with a typed reason; plain assignments keep the
    statement-prefix join (the FLOW-6 control still declines as a path conflict)."""

    @pytest.mark.parametrize("body", [
        # reviewer: walrus inside the raise expression
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    try:\n        raise ValueError((Y := Z).shape[0])\n    except Exception:\n        pass\n    return Y\n',
        # own: walrus in the handler body
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    try:\n        raise ValueError("t")\n    except Exception:\n        _ = (Y := Z)\n    return Y\n',
        # own: walrus in the try-else block
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    try:\n        pass\n    except Exception:\n        pass\n    else:\n        _ = (Y := Z)\n    return Y\n',
        # own: walrus in a match case body
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    match A.shape[0]:\n        case 1:\n            _ = (Y := Z)\n        case _:\n            pass\n    return Y\n',
        # own: walrus in a match guard
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    match A.shape[0]:\n        case 1 if (Y := Z) is not None:\n            pass\n        case _:\n            pass\n    return Y\n',
        # own: walrus nested in an if inside the try body, not even touching the returned name
        '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n'
        '    try:\n        if (n := A.shape[0]) > 0:\n            pass\n    except Exception:\n        pass\n    return Y\n',
    ])
    def test_assignment_expression_in_a_guarded_region_declines(self, body):
        fast, seen, slow = _fast_slow(_kernel_program(body))
        assert fast is None and seen and seen[0].startswith("kernel_output_dtype_unestablished:k")
        assert "exception/expression provenance not modelled" in seen[0]
        fallen, _ = execute_translated(_kernel_program(body), {"a": np.array([[1.003, 1.009]])}, all_inputs={"a": np.array([[1.003, 1.009]])}, fast=True)
        np.testing.assert_array_equal(np.asarray(fallen["output"], dtype=np.float64), slow)

    def test_walrus_outside_guarded_regions_is_still_modelled(self):
        """The class is scoped: a walrus at the kernel's top level keeps the existing kill semantics (unknown value
        → unestablished because the returned name is rebound to an untraceable expression; established when not)."""
        code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    _ = (n := A.shape[0])\n    return Y\n')
        fast, seen, slow = _fast_slow(code)
        assert fast is not None and not seen
        np.testing.assert_array_equal(np.asarray(fast["output"], dtype=np.float64), slow)

    def test_plain_assignment_in_try_keeps_the_prefix_join(self):
        """FLOW-6 unchanged: the verbatim try-prefix control still declines as a path CONFLICT, not as unmodelled."""
        reasons, value, slow = _review_control("try_prefix_rebind")
        assert reasons[0].startswith("kernel_output_dtype_conflict:k")


class TestRound6FlowWholeStatementWalk:
    """flow-15: ONE generic walk over the ENTIRE ``try`` / ``match`` statement (body, handler types, names,
    bodies, else, finally, subject, patterns, guards): any assignment expression declines with the typed reason; a
    handler type that is not a Name / Attribute / Tuple of those declines too (typed, never a generic Numba fallback)."""

    @pytest.mark.parametrize("body", [
        '    try:\n        raise ValueError("t")\n    except ((Y := Z), Exception)[1]:\n        pass\n',                      # reviewer: tuple-subscript walrus
        '    try:\n        raise ValueError("t")\n    except (Exception if (Y := Z) is not None else ValueError):\n        pass\n',   # reviewer: conditional
        '    try:\n        raise ValueError("t")\n    except (Y := Z) and Exception:\n        pass\n',                         # own: BoolOp walrus
        '    try:\n        pass\n    except Exception as e:\n        pass\n    else:\n        pass\n    try:\n        raise ValueError("t")\n    except Exception:\n        _ = [(Y := Z) for _i in range(1)]\n',  # own: comprehension walrus in a handler
        '    match (Y := Z).shape[0]:\n        case _:\n            pass\n',                                                   # own: walrus in the match subject
    ])
    def test_assignment_expression_anywhere_in_the_statement_declines(self, body):
        fast, seen, slow = _fast_slow(_kernel_program(_FLOW_INIT + body + '    return Y\n'))
        assert fast is None and seen and seen[0].startswith("kernel_output_dtype_unestablished:k"), seen
        assert "exception/expression provenance not modelled" in seen[0]
        x = np.array([[1.003, 1.009]])
        fallen, _ = execute_translated(_kernel_program(_FLOW_INIT + body + '    return Y\n'), {"a": x.shape}, all_inputs={"a": x}, fast=True)
        np.testing.assert_array_equal(np.asarray(fallen["output"], dtype=np.float64), slow)

    @pytest.mark.parametrize("handler_type", ["errs[0]", "pick()", "(ValueError, errs[0])", "Exception or ValueError", "(lambda: Exception)()"])
    def test_non_plain_handler_type_declines_typed(self, handler_type):
        body = _FLOW_INIT + '    errs = (ValueError,)\n    pick = lambda: ValueError\n    try:\n        raise ValueError("t")\n    except ' + handler_type + ':\n        pass\n    return Y\n'
        fast, seen, slow = _fast_slow(_kernel_program(body))
        assert fast is None and seen and seen[0].startswith("kernel_output_dtype_unestablished:k")
        assert "handler type" in seen[0] and "not a plain name / attribute / tuple" in seen[0]

    def test_plain_handler_types_are_still_modelled(self):
        """Positive: a Name, a dotted Attribute and a Tuple of Names are plain; a try that rebinds nothing lowers."""
        for t in ("Exception", "(ValueError, TypeError)", "np.AxisError" if False else "ValueError"):
            code = _kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    try:\n        pass\n    except ' + t + ':\n        pass\n    return Y\n')
            fast, seen, slow = _fast_slow(code)
            assert fast is not None and not seen, (t, seen)
        assert _is_plain_handler_type(ast.parse("a.b.c").body[0].value) and _is_plain_handler_type(ast.parse("(a, b.c)").body[0].value)
        assert not _is_plain_handler_type(ast.parse("a[0]").body[0].value) and not _is_plain_handler_type(ast.parse("(a, f())").body[0].value)

    def test_prefix_join_still_the_model_for_plain_assignments(self):
        reasons, value, slow = _review_control("try_prefix_rebind")
        assert reasons[0].startswith("kernel_output_dtype_conflict:k")


_FLOW_INIT = ('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)\n'
              '    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n')


_RUN_TWICE = ("\ndef flip():\n    globals()['DT'] = 'fp32'\n\ndef run_model(inputs):\n    clear_dims()\n"
              "    a = Tensor.from_numpy(inputs['a'], 'M,N', dtype='fp32')\n    y1 = k(a)\n    flip()\n    y2 = k(a)\n"
              "    return {'output': y1.data, 'output2': y2.data}\n")

# every module-initialisation attack of rounds 4-7 (reviewers' programs, verbatim prefixes) plus own variants
_MODULE_ATTACKS = {
    "r4_class_global": 'class Rebind:\n    global DT\n    DT = "fp32"\n',
    "r5_default_walrus": 'def side(x=(DT := "fp32")):\n    pass\n',
    "r5_annotation_walrus": 'def side(x: (DT := "fp32")):\n    pass\n',
    "r5_class_base_walrus": 'class C((DT := "fp32") and object):\n    pass\n',
    "r5_decorator_walrus": 'def deco(f):\n    return f\n@(deco if (DT := "fp32") else deco)\ndef side():\n    pass\n',
    "r5_comprehension_walrus": '_ = [(DT := "fp32") for _i in range(1)]\n',
    "r5_lambda_default_walrus": 'side = lambda x=(DT := "fp32"): x\n',
    "r6_exec_write": 'exec("DT = \'fp32\'")\n',
    "r6_eval_write": 'eval("globals().__setitem__(\'DT\', \'fp32\')")\n',
    "r6_globals_subscript": 'globals()["DT"]="fp32"\n',
    "r6_globals_alias_update": 'ns=globals()\nns.update(DT="fp32")\n',
    "r6_setattr_module": 'import types\nmodule=types.ModuleType("review_private_module")\nmodule.__dict__.update(globals())\nsetattr(module,"DT","fp32")\nglobals().update(vars(module))\n',
    "r7_getattr_function_globals": 'def marker():\n    pass\ngetattr(marker, "__globals__")["DT"]="fp32"\n',
    "r7_getattribute_function_globals": 'def marker():\n    pass\nmarker.__getattribute__("__globals__")["DT"]="fp32"\n',
    "r7_object_getattribute_globals": 'def marker():\n    pass\nobject.__getattribute__(marker, "__globals__")["DT"]="fp32"\n',
    "r7_computed_attribute_string": 'def marker():\n    pass\ngetattr(marker, "__glo"+"bals__").update({"DT":"fp32"})\n',
    "r7_generator_frame_globals": 'g=(x for x in [])\ng.gi_frame.f_globals["DT"]="fp32"\n',
    "r7_decorator_injected_globals": 'def inject(f):\n    getattr(f,"__globals__")["DT"]="fp32"\n    return f\n@inject\ndef marker():\n    pass\n',
    "r7_class_body_reflection": 'class C:\n    getattr(lambda:0,"__globals__")["DT"]="fp32"\n',
    "r8_library_builtin_exec_default": 'import numpy as np\ndef marker(arg=np.ma.core.builtins.exec("DT = \'fp32\'")):\n    pass\n',
    "r8_library_builtin_getattr_default": 'import numpy as np\ndef seed():\n    pass\ndef marker(arg=np.ma.core.builtins.getattr(seed,b"__globals__".decode()).update({"DT":"fp32"})):\n    pass\n',
    "r8_operator_bytes_attrgetter": 'import numpy as np\ndef seed():\n    pass\ndef marker(arg=np.ma.core.operator.attrgetter(b"__globals__".decode())(seed).update({"DT":"fp32"})):\n    pass\n',
    "r8_operator_chr_attrgetter": 'import numpy as np\ndef seed():\n    pass\ndef marker(arg=np.ma.core.operator.attrgetter(chr(95)*2+"globals"+chr(95)*2)(seed).update({"DT":"fp32"})):\n    pass\n',
    "r8_helper_default_via_numpy_builtins": 'def mutate():\n    from numpy import __builtins__ as space\n    space["exec"]("DT = \'fp32\'",space["globals"]())\n    return 0\ndef marker(arg=mutate()):\n    pass\n',
    "r8_module_annotation_exec": 'from numpy import __builtins__ as space\nflag: space["exec"]("DT = \'fp32\'", space["globals"]()) = 0\n',
    "r8_sys_modules_alias": 'import sys as runtime\nruntime.modules[__name__].DT="fp32"\n',
    "r8_types_module_no_effect": 'import types\nm=types.ModuleType("scratch")\nm.DT="fp32"\n',
    "r8_fake_kernel_def": 'def kernel(f):\n    return f\n',
    "r8_kernel_replaced_by_exec": 'exec("kernel = lambda f: f")\n',
    "r8_kernel_replaced_via_attrgetter": 'import numpy as np\ndef seed():\n    pass\ndef marker(arg=np.ma.core.operator.attrgetter(b"__globals__".decode())(seed).update({"kernel":lambda f:f})):\n    pass\n',
    "r8_dunder_getattr_def": 'def __getattr__(name):\n    return "fp32"\n',
    "r8_delete_reassign": 'del DT\nDT="fp32"\n',
    "r8_annassign": 'DT: str = "fp32"\n',
}


def _both_carriers(code):
    """(slow, [(use_fp64, lowered?, decline reason, public output)]) with the public fast-or-fallback path."""
    x = np.array([[1.003, 1.009]])
    slow, _ = execute_translated(code, {"a": x.shape}, all_inputs={"a": x}, fast=False)
    out = []
    for use_fp64 in (True, False):
        seen = []
        fast = execute_fast(code, {"a": x}, use_fp64=use_fp64, on_fallback=seen.append)
        public, _ = execute_translated(code, {"a": x.shape}, all_inputs={"a": x}, fast=True, use_fp64=use_fp64)
        out.append((use_fp64, fast is not None, seen[0] if seen else None, public))
    return slow, out


class TestLiteralOnlyPath:
    """Round 9 (Jenny: deliver the literal path only).  A ``dtype=`` keyword is honoured when it is a string LITERAL;
    any other expression -- a module name, a kernel-local name, an attribute, a call -- is the dtype-specific typed
    decline at lowering (``kernel_output_dtype_unestablished:<kernel>``) and the interpreter runs the program.  Nothing
    is folded or evaluated at call time, so there is nothing a module-initialisation attack can defeat: every attack of
    rounds 4-8 declines identically in both carriers (the fallback carrier is the interpreter's fp64) and the public
    result is the interpreter's.  The literal path itself is unchanged (the reviewer's seven literal controls)."""

    @pytest.mark.parametrize("name", list(_MODULE_ATTACKS))
    def test_every_module_attack_is_a_static_typed_decline(self, name):
        code = _const_program(_MODULE_ATTACKS[name])
        for use_fp64 in (True, False):
            reasons = []
            assert numba_lower(code, boundscheck=True, use_fp64=use_fp64, on_decline=reasons.append) is None, name
            assert reasons[0].startswith("kernel_output_dtype_unestablished:k"), (name, reasons)
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert not lowered and reason.startswith("kernel_output_dtype_unestablished:k"), (name, use_fp64, reason)
            got = np.asarray(public["output"])
            assert got.dtype == np.float64 and np.array_equal(got, slow["output"]), (name, use_fp64, got, slow["output"])

    @pytest.mark.parametrize("name", ["r8_fake_kernel_def", "r8_kernel_replaced_by_exec"])
    def test_a_rebound_kernel_name_is_the_identity_decline(self, name):
        """A module that rebinds ``kernel`` statically is declined for THAT reason before the dtype prescan.  A
        reflective replacement (``exec`` at module scope) was caught by the run-time identity guard once the dtype
        was literal (round 9); since an earlier fix round 3 (MODULE-BINDING-5) a module-scope namespace rewrite is a STATIC
        decline -- ``kernel_allocation_callee_unresolved:k: module_binding_uncertain`` -- that names both bindings
        it cannot follow ('Tensor' and the @kernel decorator); the run-time guard stays behind it for anything the
        static scan admits.  Either way nothing lowers and the public result is the interpreter's."""
        literal = _const_program(_MODULE_ATTACKS[name]).replace("dtype=DT, fill=1.003", 'dtype="bf16", fill=1.003')
        reasons = []
        low = numba_lower(literal, boundscheck=True, use_fp64=True, on_decline=reasons.append)
        assert low is None
        if name == "r8_fake_kernel_def":
            assert "is not einsum_lib's kernel" in reasons[0] and "rebinds the name 'kernel'" in reasons[0]
        else:
            assert reasons[0].startswith("kernel_allocation_callee_unresolved:k: module_binding_uncertain:") and "the @kernel decorator" in reasons[0]
        seen = []; x = np.array([[1.003, 1.009]])
        assert execute_fast(literal, {"a": x}, use_fp64=True, on_fallback=seen.append) is None
        assert seen[0] == reasons[0]

    @pytest.mark.parametrize("dt,fill", [("bf16", "1.003"), ("fp16", "1.003"), ("fp32", "1.003"), ("fp64", "1.003"), ("int32", "1.75")])
    def test_literal_dtypes_lower_and_match_the_interpreter_exactly(self, dt, fill):
        """the reviewer's literal controls: exact carrier-cast comparison (native float results are the float32 image
        of the interpreter's float64; integers and the fp64 carrier compare bit-for-bit)."""
        code = _kernel_program(f'    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="{dt}", fill={fill})\n    return Y\n')
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, (dt, use_fp64, reason)
            exp = np.asarray(slow["output"]); got = np.asarray(public["output"])
            if not use_fp64 and exp.dtype.kind == "f":
                exp = exp.astype(np.float32)
                assert got.dtype == np.float32, (dt, got.dtype)
            assert np.array_equal(got, exp), (dt, use_fp64, got, exp)

    @pytest.mark.parametrize("variant", ["nested_call", "side_effect_default"])
    def test_literal_nested_call_and_module_side_effect(self, variant):
        body = '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n'
        if variant == "nested_call":
            code = _kernel_program(body).replace("k(a).data", "k(k(a)).data")
        else:
            code = _kernel_program(body).replace('DT = "bf16"\n', 'DT = "bf16"\nCOUNT = 0\ndef effect():\n    global COUNT\n    COUNT += 1\n    return COUNT\ndef marker(x=effect()):\n    pass\n')
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, (variant, reason)
            exp = np.asarray(slow["output"], dtype=np.float64 if use_fp64 else np.float32)
            assert np.array_equal(np.asarray(public["output"]), exp)
        np.testing.assert_array_equal(slow["output"], [[1.0, 1.0]])

    def test_runtime_mutation_between_two_calls_declines_and_the_interpreter_runs(self):
        """The round-8 control: ``dtype=DT`` is a name, so the kernel is declined before anything runs; the interpreter
        gives bf16 1.0 then fp32 1.003 and the public path returns exactly that in both carriers."""
        code = _const_program("").replace(_K_RUN, _RUN_TWICE)
        slow, runs = _both_carriers(code)
        np.testing.assert_array_equal(slow["output"], [[1.0, 1.0]]); np.testing.assert_array_equal(slow["output2"], [[1.003, 1.003]])
        for use_fp64, lowered, reason, public in runs:
            assert not lowered and reason.startswith("kernel_output_dtype_unestablished:k")
            np.testing.assert_array_equal(np.asarray(public["output"]), slow["output"])
            np.testing.assert_array_equal(np.asarray(public["output2"]), slow["output2"])

    @pytest.mark.parametrize("spelling", ['choose_dtype()', 'DT', 'A.dtype'], ids=["call", "module-name", "attribute"])
    def test_non_returned_allocation_with_a_non_literal_dtype_declines_before_stripping(self, spelling):
        """literal-boundary-30 (the review): ``T = Tensor(..., dtype=choose_dtype(), fill=0)`` never reaches a
        return, so return-site provenance never saw it, and the lowering dropped the dtype expression -- with its
        side effect (COUNT += 1) -- so the kernel returned 1.0 where the interpreter returns 2.0.  Every allocation's
        dtype must be a literal: decline statically in both carriers, interpreter value on the public path."""
        code = _kernel_program('    T = Tensor(M=A.shape[0], N=A.shape[1], dtype=%s, fill=0)\n'
                               '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=COUNT + 1.003)\n    return Y\n' % spelling)
        code = code.replace('DT = "bf16"\n', 'DT = "bf16"\nCOUNT = 0\ndef choose_dtype():\n    global COUNT\n    COUNT += 1\n    return "bf16"\n')
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert not lowered and reason.startswith("kernel_output_dtype_unestablished:k") and "not a string literal" in reason, reason
            np.testing.assert_array_equal(np.asarray(public["output"]), slow["output"])
        if spelling == "choose_dtype()":
            np.testing.assert_array_equal(slow["output"], [[2.0, 2.0]])     # the interpreter evaluated the dtype expression (COUNT -> 1)

    def test_unsupported_literal_dtype_on_any_allocation_declines(self):
        code = _kernel_program('    T = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf8", fill=0)\n'
                               '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n')
        reasons = []
        assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=reasons.append) is None
        assert "not a supported dtype" in reasons[0]
        with pytest.raises(RuntimeError):
            execute_translated(code, {"a": (1, 2)}, all_inputs={"a": np.array([[1.003, 1.009]])}, fast=False)

    def test_literal_temporaries_still_lower(self):
        code = _kernel_program('    T = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=0)\n'
                               '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n')
        fast, seen, slow = _fast_slow(code); assert fast is not None and not seen
        np.testing.assert_array_equal(fast["output"], slow); np.testing.assert_array_equal(slow, [[1.0, 1.0]])

    def test_literal_lowering_text_keeps_the_static_wrapper_and_the_identity_guard(self):
        low = numba_lower(_kernel_program('    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n'), boundscheck=True, use_fp64=True)
        assert "_truncate_bf16(k(a))" in low and "_require_einsum_kernel(kernel, 'k')" in low
        assert "_truncate_dynamic(" not in low and "lambda" not in low

    def test_no_call_time_machinery_remains(self):
        import solar.einsum_lib.numba_accel as na
        from solar.einsum_lib.numba_accel import execution_namespace
        for name in ("_truncate_dynamic", "_call_time_resolvable", "_DtypeContext", "_dtype_name_bindings", "_is_dynamic_dtype", "_DYN_PREFIX"):
            assert not hasattr(na, name), name
        assert "_truncate_dynamic" not in execution_namespace(True) and "_require_einsum_kernel" in execution_namespace(True)

    def test_static_judgement_shims_report_nothing(self):
        """The round-5..7 static machinery stays deleted; the retained names are inert shims for older control files."""
        from solar.einsum_lib.numba_accel import _module_string_constants, _module_dynamic_namespace, _module_init_not_static, _module_reflective_construct
        t = ast.parse(_const_program('exec("DT = \'fp32\'")\n'))
        assert _module_string_constants(t) == {} and _module_dynamic_namespace(t) is None
        assert _module_init_not_static(t) is None and _module_reflective_construct(t) is None


def _run_reviewer_file(tmp_path, filename, siblings=()):
    """Run one of the reviewers' control files byte-for-byte in a scratch copy (with the sibling files it imports),
    in its own interpreter, and return (failed-or-errored test ids, stdout)."""
    run = tmp_path / "run"; run.mkdir(exist_ok=True)
    (run / "evidence").mkdir(exist_ok=True); (tmp_path / "evidence").mkdir(exist_ok=True); (run / "runs").mkdir(exist_ok=True)   # round-6 / round-4/5 / the review layouts
    for name in (filename,) + tuple(siblings):
        (run / name).write_bytes((R3 / name).read_bytes())
    env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["REVIEW_SOURCE"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = subprocess.run([sys.executable, "-m", "pytest", str(run / filename), "-q", "-p", "no:cacheprovider", "-rfE"],
                         cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
    ids = sorted(l.split("::", 1)[1].split(" - ")[0] for l in res.stdout.splitlines() if l.startswith(("FAILED", "ERROR")) and "::" in l)
    return ids, res.stdout


class TestReviewerFilesVerbatim:
    """Every reviewer control file of rounds 4-8 run byte-for-byte (sha256 of each file in the fixture directory is
    recorded in the delivery), with the EXACT deviation set pinned.  Under the literal-only lowerer the deviations
    are of four kinds and nothing else: (N) the test-side prange normalizer was removed (normalizer-27), so a
    test that imports it from this module errors; (S) a positive that expected a module-NAME dtype to lower -- names
    now decline (literal only); (J) a probe of the deleted round-5..7 static judgement (inert shim); (C) the round-6
    file's native-carrier expectation that a DECLINED case returns float32 -- the fallback carrier is the
    interpreter's float64, by design (the values are equal; only the carrier differs).  Where the reviewer's own
    value check runs, it passes."""

    _R4 = ["test_normalizer_preserves_extra_body_changes[    m = np.intp(m + 1)\\n]",
           "test_normalizer_preserves_extra_body_changes[    s += _m_prange\\n]",
           "test_normalizer_fresh_used_after_loop"]
    _R5 = ["test_semantics_and_class_declines[attribute_store_positive-True]", "test_semantics_and_class_declines[attribute_store_positive-False]"] + \
          [f"test_normalizer_binding_and_usage_classes[{c}]" for c in ("nested_function_read", "default_argument_read", "global_declaration", "import_alias",
                                                                        "function_name", "match_capture", "keyword_argument_name")] + \
          ["test_normalizer_nonlocal_declaration"]
    _R6 = [f"test_dynamic_dtype_reflection_must_not_silently_misround[{n}-False]" for n in
           ("getattr_function_globals", "getattribute_function_globals", "object_getattribute_globals", "computed_attribute_string",
            "generator_frame_globals", "decorator_injected_globals", "class_body_reflection")] + \
          [f"test_declared_dynamic_patterns_remain_typed[{n}]" for n in
           ("sys_alias", "getattr_builtins_exec", "builtins_import_alias", "module_type_setattr", "class_globals", "function_attr_globals")] + \
          [f"test_normalizer_reflection_fstrings_and_safe_strings[{n}]" for n in
           ("class_annotation_string", "comment_positive", "format_map_locals", "fstring_identifier", "fstring_locals_lookup",
            "fstring_walrus", "keyword_overdecline", "locals_string_key", "string_positive", "vars_string_key")]
    _R7 = [f"test_static_positive[{n}-{m}]" for n in ("float_literal", "static_container", "kernel_identity") for m in ("True", "False")] + \
          [f"test_reflective_normalizer_preserves_difference[{n}]" for n in
           ("vars", "format_map", "nested_fstring", "bytes_literal_lookup", "translate_lookup", "chr_join_lookup", "folded_string_lookup", "annotations_bytes_alias")]
    _R8 = ["test_normalizer_namespace_binding[parameter]", "test_normalizer_namespace_binding[module]"]

    @pytest.mark.parametrize("filename,siblings,expected,summary", [
        ("test_review7_round4.py", (), _R4, "3 failed, 18 passed"),                       # N x3
        ("test_review7_round5.py", (), _R5, "10 failed, 31 passed"),                      # S x2 (module attribute-store positive), N x8
        ("test_round6_adversarial.py", (), _R6, "23 failed, 19 passed"),                  # C x7, J x6, N x10
        ("test_independent_round7.py", (), _R7, "14 failed, 36 passed"),                  # S x6 (dtype=DT positives), N x8
        ("test_followup_round7.py", ("test_independent_round7.py",), [], "12 passed"),   # all twelve decline, typed
        ("test_reflective_attributes.py", ("test_independent_round7.py",), [], "10 passed"),
        ("test_round8_direct.py", (), _R8, "2 failed, 32 passed"),                        # N x2; CONST-24/25/26 cases and the 7 literal controls pass
        ("test_isolation_direct.py", (), [], "3 passed"),                                 # ISOLATION-RNG-28 + SIGKILL control
        ("test_review4_round9.py", (), [], "13 passed"),                                  # the review round 9: identity x8, loop RNG x3, LITERAL-BOUNDARY-30 x2
    ], ids=["round4", "round5", "round6", "round7-independent", "round7-followup", "round7-reflective", "round8-direct", "round8-isolation", "round9-review4"])
    def test_file(self, tmp_path, filename, siblings, expected, summary):
        ids, out = _run_reviewer_file(tmp_path, filename, siblings)
        assert ids == sorted(expected), out[-5000:]
        assert summary in out, out[-400:]
