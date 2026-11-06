# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The static index-typing diagnostic for lowered Numba kernels (``lowered_index_check``).

Finding SHAPE0-205-NUMBA-FLOAT-INDEX-1: the lowered row-205 program failed in
``native parfor lowering`` with ``getitem(array(float64, 3d, C),
Tuple(float64, float64, int64))``.  Numba's parfor gufunc iterates a
``uintp`` schedule, so inside the gufunc a ``numba.prange`` index is
``uint64`` while the outer typing sees ``int64``; ``min``/``max`` of that
index with a signed int unify to ``float64`` (plain binops stay ``int64``).
DSL ``Dim`` indices are Python ints, so the lowerer rebinds every prange
index to ``np.intp`` (the emitter fix of an earlier change).  This module tests the
companion DIAGNOSTIC: it types subscript indices with Numba's gufunc rules and
reports the ones that are provably float64.  Its verdicts are compared with
Numba's own typing on every control (checker-vs-Numba); zero findings is not
an integrality certificate (unknown-call and parameter-element limits are
recorded in the module docstring).
"""

import os
import textwrap

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.lowered_index_check import (
    check_lowered_source,
    iter_lowered_programs,
)
from solar.einsum_lib.numba_accel import HAS_NUMBA, execute_fast, numba_lower
from solar.einsum_lib.verify import execute_translated


@pytest.fixture(autouse=True)
def _clean_dims():
    clear_dims()
    stop_trace()
    yield
    stop_trace()
    clear_dims()


# The pad kernel of the row-205 candidate (solar_model_einsum_lib.py sha256
# e016eaed..., lines 63-75), verbatim: a clamped-token gather with a 0/1 guard.
PAD_KERNEL_DSL = '''\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_ALIGN = 128

@kernel
def pad_upcast_fp32(X: Tensor("B,S,H", dtype="bf16")) -> Tensor("MP,H"):
    nt = X.shape[0] * X.shape[1]
    ns = X.shape[1]
    mp = ((nt + _ALIGN - 1) // _ALIGN) * _ALIGN
    Y = Tensor(MP=mp, H=X.shape[2], dtype="fp32")
    for m in Dim("MP", parallel=True):
        t = min(m, nt - 1)
        g = max(0, min(1, nt - m))
        for h in Dim("H"):
            Y[m, h] = X[t // ns, t % ns, h] * g
    return Y

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,S,H", dtype="bf16")
    return {"output": pad_upcast_fp32(x).data}
'''

# What the emitter produced for that kernel before the fix: lines 17-27 of
# results/cpu/205-numerical-rep1/executed-lowered.py (sha256 339e9b43...,
# lowering-intent.json use_fp64=false).  The TypingError points at the
# ``X[t // ns, t % ns, h]`` read.
SAVED_205_LOWERED = '''\
import numpy as np
import math
import numba
from math import exp, log, sqrt, cos, sin
"(module docstring of the saved program: one line, kept so line numbers match)"
_AXIS_BINDINGS = {}
_FREE_CONSTANTS = ['_BLK', '_ALIGN']
_BLK = 16
_ALIGN = 128

@numba.njit(parallel=True, boundscheck=True)
def pad_upcast_fp32(X):
    nt = X.shape[0] * X.shape[1]
    ns = X.shape[1]
    mp = (nt + _ALIGN - 1) // _ALIGN * _ALIGN
    Y = np.zeros((mp, X.shape[2]), dtype=np.float32)
    for m in numba.prange(mp):
        t = min(m, nt - 1)
        g = max(0, min(1, nt - m))
        for h in range(X.shape[2]):
            Y[m, h] = X[t // ns, t % ns, h] * g
    return Y
'''

# Row 206 (nvfp4_fused_decoder_layer_pre_norm_attention, candidate sha256
# 609e5137...) fails on the same shape: lines 60-72 of its executed-lowered.py
# (sha256 5218f6ef...), TypingError at line 71.
SAVED_206_LOWERED = '''\
import numpy as np
import math
import numba
from math import exp, log, sqrt, cos, sin
"(module docstring of the saved program: one line, kept so line numbers match)"
_AXIS_BINDINGS = {'_HD': 'head_dim', '_NH': 'num_attention_heads', '_KV': 'num_key_value_heads'}
_FREE_CONSTANTS = ['_BLK', '_ROW']
_HD = 128
_NH = 40
_KV = 8
_G = _NH // _KV
_BLK = 16
_ROW = 128
_F32MAX = 3.4028234663852886e+38

@numba.njit(parallel=True, boundscheck=True)
def pad_rows_upcast_fp32(X):
    nt = X.shape[0] * X.shape[1]
    ns = X.shape[1]
    mp = (nt + _ROW - 1) // _ROW * _ROW
    tmax = nt - 1
    Y = np.zeros((mp, X.shape[2]), dtype=np.float32)
    for m in numba.prange(mp):
        t = min(m, tmax)
        g = max(0, min(1, nt - m))
        for k in range(X.shape[2]):
            Y[m, k] = X[t // ns, t % ns, k] * g
    return Y
'''


def _pad_reference(x: np.ndarray) -> np.ndarray:
    t, h = x.shape[0] * x.shape[1], x.shape[2]
    mp = -(-t // 128) * 128
    out = np.zeros((mp, h), dtype=np.float32)
    out[:t] = x.reshape(t, h)
    return out


# ---------------------------------------------------------------------------
# Static regression on the saved row-205 program
# ---------------------------------------------------------------------------

class TestSaved205Program:
    def test_saved_program_has_float_typed_getitem_indices(self):
        findings = check_lowered_source(SAVED_205_LOWERED, "executed-lowered.py")
        assert [f.kernel for f in findings] == ["pad_upcast_fp32"] * 2
        assert [f.index_text for f in findings] == ["t // ns", "t % ns"]
        # the flagged read is the getitem the TypingError named
        assert all(f.lineno == 21 and f.access == "load" for f in findings)
        # and the cause chain names the unsigned prange index in min()
        assert all("prange" in f.reason and "min(" in f.reason for f in findings)

    def test_saved_206_program_is_the_same_finding(self):
        findings = check_lowered_source(SAVED_206_LOWERED, "executed-lowered.py")
        assert [(f.kernel, f.index_text, f.lineno) for f in findings] == [
            ("pad_rows_upcast_fp32", "t // ns", 27),
            ("pad_rows_upcast_fp32", "t % ns", 27),
        ]
        assert all("prange" in f.reason and "min(" in f.reason for f in findings)

    def test_saved_program_integral_indices_are_not_flagged(self):
        findings = check_lowered_source(SAVED_205_LOWERED, "executed-lowered.py")
        assert "h" not in {f.index_text for f in findings}
        assert "m" not in {f.index_text for f in findings}



class TestReLoweredKernel:
    def test_lowered_pad_kernel_indices_are_integral(self):
        lowered = numba_lower(PAD_KERNEL_DSL, boundscheck=True, use_fp64=False)
        assert lowered is not None
        assert check_lowered_source(lowered, "<lowered>") == []


# ---------------------------------------------------------------------------
# The checker itself: Numba's gufunc typing rules it models
# ---------------------------------------------------------------------------

def _kernel(body: str) -> str:
    return "import numpy as np\nimport numba\n\n@numba.njit(parallel=True)\ndef k(X, I):\n" + textwrap.indent(textwrap.dedent(body), "    ")


class TestCheckerRules:
    @pytest.mark.parametrize("index", [
        "m", "m // 2", "m % 3", "m + 1", "nt - m", "m * 2 + 1", "np.intp(m)",
        "int(m)", "max(0, min(1, nt - m))", "I[m]", "min(nt - 1, 3)",
    ])
    def test_integral_index_expressions_are_not_flagged(self, index):
        src = _kernel(f"""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for m in numba.prange(nt):
                Y[m] = X[{index}, 0]
            return Y
        """)
        assert check_lowered_source(src) == []

    @pytest.mark.parametrize("index, why", [
        ("min(m, nt - 1)", "prange"),       # the row-205 shape
        ("max(m, 0)", "prange"),
        ("m if m < nt else 0", "prange"),   # ifexp unifies like min/max
        ("m / 2", "true division"),
        ("nt / 2", "true division"),
        ("int(m) + 0.5", "float literal"),
        ("math.floor(m / 2)", "math.floor") if False else ("m // 2.0", "float literal"),
    ])
    def test_float_typed_index_expressions_are_flagged(self, index, why):
        src = _kernel(f"""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for m in numba.prange(nt):
                Y[m] = X[{index}, 0]
            return Y
        """)
        findings = check_lowered_source(src)
        assert [f.index_text for f in findings] == [index]
        assert why in findings[0].reason

    def test_float_flows_through_local_assignments(self):
        src = _kernel("""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for m in numba.prange(nt):
                t = min(m, nt - 1)
                u = t // 2
                Y[m] = X[u, 0]
            return Y
        """)
        findings = check_lowered_source(src)
        assert [f.index_text for f in findings] == ["u"]
        assert "t = min(m, nt - 1)" in findings[0].reason

    def test_serial_range_index_in_min_is_integral(self):
        src = _kernel("""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for m in range(nt):
                Y[m] = X[min(m, nt - 1), 0]
            return Y
        """)
        assert check_lowered_source(src) == []

    def test_inner_prange_inside_a_prange_is_a_serial_range(self):
        # Numba parallelises only the outermost prange of a nest.
        src = _kernel("""
            nt = X.shape[0]
            Y = np.zeros((nt, nt), dtype=np.float32)
            for m in numba.prange(nt):
                for n in numba.prange(nt):
                    Y[m, n] = X[min(n, nt - 1), 0]
            return Y
        """)
        assert check_lowered_source(src) == []

    def test_rebound_prange_index_is_integral(self):
        src = _kernel("""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for _m in numba.prange(nt):
                m = np.intp(_m)
                Y[m] = X[min(m, nt - 1), 0]
            return Y
        """)
        assert check_lowered_source(src) == []

    def test_store_indices_are_checked_too(self):
        src = _kernel("""
            nt = X.shape[0]
            Y = np.zeros((nt,), dtype=np.float32)
            for m in numba.prange(nt):
                Y[min(m, nt - 1)] = X[m, 0]
            return Y
        """)
        findings = check_lowered_source(src)
        assert [(f.index_text, f.access) for f in findings] == [("min(m, nt - 1)", "store")]

    def test_only_njit_kernels_are_checked(self):
        src = "import numpy as np\ndef run_model(inputs):\n    return {'o': inputs['x'][inputs['n'] / 2]}\n"
        assert check_lowered_source(src) == []

    def test_free_module_constants_take_their_literal_type(self):
        src = "import numpy as np\nimport numba\n_ALIGN = 128\n_SCALE = 0.5\n" + (
            "@numba.njit(parallel=True)\ndef k(X):\n"
            "    Y = np.zeros((X.shape[0],), dtype=np.float32)\n"
            "    for m in numba.prange(X.shape[0]):\n"
            "        Y[m] = X[m // _ALIGN, 0] + X[int(m * _SCALE), 0]\n"
            "    return Y\n")
        assert check_lowered_source(src) == []
        bad = src.replace("int(m * _SCALE)", "m * _SCALE")
        assert [f.index_text for f in check_lowered_source(bad)] == ["m * _SCALE"]


# ---------------------------------------------------------------------------
# Independent reviewer controls (first review of the lowered-index checker,
# verbatim CASES): the checker's verdict must agree with Numba's own typing.
# checker-fn-1 (and/or, branch join, NumPy default dtype, array alias,
# literal iterable) and checker-fp-2 (uint32 is not the uint64 rule).
# ---------------------------------------------------------------------------

def _probe_kernel(body, params='X, flag', parallel=True):
    return ('import numpy as np\nimport numba\n@numba.njit(parallel=' + str(parallel)
            + ', boundscheck=True)\ndef probe(' + params + '):\n'
            + textwrap.indent(textwrap.dedent(body).strip() + '\n', '    '))


REVIEWER_CASES = [
    ('division', 'j = 1 / 2\nreturn X[j]', True),
    ('float_axis', 'for j in (0.5,):\n    return X[j]\nreturn X[0]', True),
    ('float_variable', 'j = float(1)\nreturn X[j]', True),
    ('gathered_float', 'A = np.ones(2, dtype=np.float64)\nj = A[0]\nreturn X[j]', True),
    ('boolean_operand', 'j = 0.5 or 1\nreturn X[j]', True),
    ('branch_join', 'if flag:\n    j = 0.5\nelse:\n    j = 0\nreturn X[j]', True),
    ('default_float_gather', 'A = np.ones(2)\nj = A[0]\nreturn X[j]', True),
    ('array_alias_gather', 'A = np.ones(2, dtype=np.float64)\nB = A\nj = B[0]\nreturn X[j]', True),
    ('uint32_min', 'j = min(np.uint32(1), 3)\nreturn X[j]', False),
    ('uint8_min', 'j = min(np.uint8(1), 3)\nreturn X[j]', False),
    ('signed_cast', 'j = np.intp(1)\nreturn X[j]', False),
]


@pytest.mark.parametrize('name,body,is_float', REVIEWER_CASES, ids=[x[0] for x in REVIEWER_CASES])
def test_checker_verdict_is_static(name, body, is_float):
    findings = check_lowered_source(_probe_kernel(body))
    assert bool(findings) == is_float, (name, [f.reason for f in findings])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,is_float', REVIEWER_CASES, ids=[x[0] for x in REVIEWER_CASES])
def test_checker_matches_numba(name, body, is_float):
    import numba
    src = _probe_kernel(body)
    namespace = {}
    exec(compile(src, '<independent-' + name + '>', 'exec'), namespace)
    fn = namespace['probe']
    findings = check_lowered_source(src)
    error = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'), numba.boolean))
    except numba.core.errors.TypingError as e:
        error = str(e)
    if is_float:
        assert error is not None and 'float' in error and ('getitem' in error or 'index' in error), error
        assert findings, 'Numba rejects a statically known float index, but checker reports zero'
    else:
        assert error is None, error
        assert fn(np.arange(4, dtype=np.float64), True) == 1
        assert not findings, 'Checker calls an integer index float64 although Numba compiles it'


@pytest.mark.parametrize("body, expect_float", [
    ("j = 1\nif flag:\n    j = 0.5\nreturn X[j]", True),            # reassigned on one path only
    ("j = 1\nwhile flag:\n    j = j / 2\nreturn X[j]", True),       # loop-carried float on one trip
    ("j = 1\nfor k in range(3):\n    j = k\nreturn X[j]", False),   # loop keeps integer kinds
    ("A = np.full(2, 1)\nreturn X[A[0]]", False),                    # full infers int from the fill value
    ("A = np.full(2, 0.5)\nreturn X[A[0]]", True),
    ("A = np.arange(4)\nreturn X[A[1]]", False),
    ("A = np.zeros(2, np.int64)\nreturn X[A[0]]", False),            # positional dtype
    ("A = np.zeros(2, dtype=np.uint32)\nreturn X[min(A[0], 3)]", False),
    ("A = np.zeros(2, dtype=np.uint64)\nreturn X[min(A[0], 3)]", True),
    ("j = 0.5 and 1\nreturn X[j]", False),                          # constant short circuit folds to 1 (Numba agrees)
    ("j = flag or 1\nreturn X[j]", False),                           # bool/int union stays integral
    ("j = flag and 0.5\nreturn X[j]", True),                         # runtime mixed case: the value may be 0.5
    ("j, k = (0.5, 0)\nreturn X[j]", True),                           # literal unpacking binds each element
    ("j, k = (0.5, 0)\nreturn X[k]", False),
    ("for j in np.arange(0.0, 3.0):\n    return X[j]\nreturn X[0]", True),
    ("A = np.arange(0.0, 3.0)\nfor j in A:\n    return X[j]\nreturn X[0]", True),
    ("A = np.arange(3)\nfor j in A:\n    return X[j]\nreturn X[0]", False),
    ("j = min(np.uint64(1), np.uint32(3))\nreturn X[j]", False),      # (uint64, uint32) -> uint64
    ("j = min(np.uint64(1), np.int64(3))\nreturn X[j]", True),        # the real uint64/int64 rule
], ids=["one-path-reassign", "while-carried", "for-int", "full-int", "full-float", "arange", "pos-dtype",
        "uint32-array", "uint64-array", "constant-and", "bool-or-int", "runtime-and", "unpack-float", "unpack-int",
        "arange-float-loop", "arange-alias-loop", "arange-int-loop", "uint64-uint32-min", "uint64-int64-min"])
def test_checker_joins_and_array_provenance(body, expect_float):
    findings = check_lowered_source(_probe_kernel(body))
    assert bool(findings) == expect_float, [f.reason for f in findings]


def test_empty_result_is_diagnostic_not_a_certificate():
    # the module says so, and an unknown-typed parameter element is (correctly) not reported
    import solar.einsum_lib.lowered_index_check as m
    assert "not an integrality certificate" in m.__doc__
    assert check_lowered_source(_probe_kernel("return X[I[0]]", params="X, I")) == []


# ---------------------------------------------------------------------------
# Sweep over saved Lite programs (host evidence; the directory is not tracked)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("SOLAR_LOWERED_PROGRAM_DIR"),
                    reason="SOLAR_LOWERED_PROGRAM_DIR not set")
def test_saved_lowered_programs_have_no_float_index_beyond_the_prange_rule():
    root = os.environ["SOLAR_LOWERED_PROGRAM_DIR"]
    programs = list(iter_lowered_programs(root))
    assert programs, root
    for path, findings in programs:
        for f in findings:
            # every float-typed index of a saved program must be explained by
            # the prange rule this fix removes; anything else is a new defect
            assert "prange" in f.reason, (path, f)


# ---------------------------------------------------------------------------
# Independent reviewer round-2 controls (second review of the lowered-index checker,
# verbatim CASES): checker verdict vs Numba's typing on a serial kernel.
# ---------------------------------------------------------------------------

FRESH_CASES = [
    ('tuple_unpack', 'j, k = (0.5, 0)\nreturn X[j]', True),
    ('math_floor', 'j = math.floor(x)\nreturn X[j]', False),
    ('int_cast', 'j = int(x)\nreturn X[j]', False),
    ('float_of_math_floor', 'j = float(math.floor(x))\nreturn X[j]', True),
    ('numpy_floor', 'j = np.floor(x)\nreturn X[j]', True),
    ('nested_return', 'def idx():\n    return 0.5\nj = idx()\nreturn X[j]', True),
    ('arange_direct_loop', 'for j in np.arange(0.0, 3.0):\n    return X[j]\nreturn X[0]', True),
    ('arange_alias_loop', 'A = np.arange(0.0, 3.0)\nfor j in A:\n    return X[j]\nreturn X[0]', True),
    ('uint8_signed_min', 'j = min(np.uint8(1), 3)\nreturn X[j]', False),
    ('uint16_signed_min', 'j = min(np.uint16(1), 3)\nreturn X[j]', False),
    ('uint32_signed_min', 'j = min(np.uint32(1), 3)\nreturn X[j]', False),
    ('uint64_signed_min', 'j = min(np.uint64(1), np.int64(3))\nreturn X[j]', True),
    ('uint64_uint8_min', 'j = min(np.uint64(1), np.uint8(3))\nreturn X[j]', False),
    ('uint64_uint16_min', 'j = min(np.uint64(1), np.uint16(3))\nreturn X[j]', False),
    ('uint64_uint32_min', 'j = min(np.uint64(1), np.uint32(3))\nreturn X[j]', False),
    ('uint64_signed_add', 'j = np.uint64(1) + np.int64(0)\nreturn X[j]', False),
    ('constant_and', 'j = 0.5 and 1\nreturn X[j]', False),
    ('constant_or_zero', 'j = 0.0 or 1\nreturn X[j]', False),
    ('constant_or_int', 'j = 1 or 0.5\nreturn X[j]', False),
    ('runtime_and_float', 'j = flag and 0.5\nreturn X[j]', True),
]
# Numba rejects this one, but the value comes from a call the checker does not
# model (a nested function's return) -- the documented unknown-call limit.
DOCUMENTED_UNKNOWN_CALL_LIMITS = {'nested_return'}


def _fresh_kernel(body):
    return ('import numpy as np\nimport numba\nimport math\n@numba.njit(boundscheck=True)\ndef probe(X, x, flag):\n'
            + textwrap.indent(body + '\n', '    '))


# ---------------------------------------------------------------------------
# Independent reviewer round-1 controls of the float-index repair (sources
# verbatim from the reviewer's evidence files and
# runs/list-followup/evidence/*.json): the 13 confirmed checker-vs-Numba
# disagreements (P2-1..P2-5) plus the agreeing controls the reviewer ran next
# to them. `list_swap_integral` (reviewer expectation: integral) was corrected by
# the list-followup oracle: `[j, k] = [k, j]` builds a float64 LIST, so the
# index IS float -- the corrected record is the one kept here.  Columns:
# name, body, expected_float (Numba rejects a float64 index), and the value
# Numba returned for X = arange(4.0) when it compiled (reviewer's record).
# `uint64_bool_min` is kept apart below: Numba < 0.68 rejects that `min()` itself, Numba >= 0.68 types it uint64.
# ---------------------------------------------------------------------------

REVIEW6_CASES = [
    # P2-1: RHS typed before binding; homogeneous list element type
    ('tuple_swap_integral', 'j = 1\nk = 0.5\nj, k = k, j\nreturn X[k]', False, 1),
    ('tuple_swap_float', 'j = 0.5\nk = 1\nj, k = k, j\nreturn X[k]', True, None),
    ('nested_unpack_integral', 'a = 1\nb = 0.5\na, (b, c) = (b, (a, 0))\nreturn X[b]', False, 1),
    ('list_literal_float_int', 'j, k = [0.5, 1]\nreturn X[k]', True, None),
    ('list_literal_int_float', 'j, k = [1, 0.5]\nreturn X[j]', True, None),
    ('list_swap_corrected_oracle', 'j = 1\nk = 0.5\n[j, k] = [k, j]\nreturn X[k]', True, None),
    # P2-2: narrow unsigned arithmetic promotes to uint64 before min/max unification
    ('uint8_sum_signed_min', 'j = min(np.uint8(1) + np.uint8(1), np.int64(3))\nreturn X[j]', True, None),
    ('uint16_sum_signed_min', 'j = min(np.uint16(1) + np.uint16(1), np.int64(3))\nreturn X[j]', True, None),
    ('uint32_sum_signed_min', 'j = min(np.uint32(1) + np.uint32(1), np.int64(3))\nreturn X[j]', True, None),
    ('narrow_sum_integral_control', 'j = np.uint8(1) + np.uint8(1)\nreturn X[j]', False, 2),
    # P2-3: constant folding through negative literals and constant aliases
    ('negative_literal_and', 'j = -0.5 and 1\nreturn X[j]', False, 1),
    ('local_constant_and', 'a = 0.5\nj = a and 1\nreturn X[j]', False, 1),
    ('runtime_and_control', 'j = flag and 0.5\nreturn X[j]', True, None),
    # P2-4: np.round / np.around keep the input dtype
    ('numpy_round_integer', 'j = np.round(np.int64(1))\nreturn X[j]', False, 1),
    ('numpy_around_integer', 'j = np.around(np.int64(1))\nreturn X[j]', False, 1),
    ('numpy_round_float_control', 'j = np.round(x)\nreturn X[j]', True, None),
    ('numpy_round_static_float_control', 'j = np.round(np.float64(x))\nreturn X[j]', True, None),
    # P2-5: positional arange dtype
    ('arange_positional_integral_dtype',
     'A = np.arange(0.0, 3.0, 1.0, np.int64)\nfor j in A:\n    return X[j]\nreturn X[0]', False, 0),
    ('arange_keyword_integral_dtype_control',
     'A = np.arange(0.0, 3.0, 1.0, dtype=np.int64)\nfor j in A:\n    return X[j]\nreturn X[0]', False, 0),
]
REVIEW6_UINT64_BOOL_MIN = 'j = min(np.uint64(1), True)\nreturn X[j]'
# Ruling P3-R4 (independent review): an UNTYPED rounding argument is unknown, so the reviewer's
# `np.round(x)` control (x is an untyped kernel parameter) is a documented expected disagreement: Numba,
# told x is float64, rejects the index; the checker proves nothing about x and stays silent.
DOCUMENTED_UNTYPED_ARGUMENT = {'numpy_round_float_control'}
REVIEW6_DISAGREEMENTS = {
    'tuple_swap_integral', 'tuple_swap_float', 'nested_unpack_integral', 'list_literal_float_int',
    'list_literal_int_float', 'uint8_sum_signed_min', 'uint16_sum_signed_min', 'uint32_sum_signed_min',
    'negative_literal_and', 'local_constant_and', 'numpy_round_integer', 'numpy_around_integer',
    'arange_positional_integral_dtype',
}


def test_review6_lists_all_thirteen_disagreements():
    names = {c[0] for c in REVIEW6_CASES}
    assert len(REVIEW6_DISAGREEMENTS) == 13 and REVIEW6_DISAGREEMENTS <= names


@pytest.mark.parametrize('name,body,expected_float', FRESH_CASES, ids=[x[0] for x in FRESH_CASES])
def test_fresh_controls_static_verdict(name, body, expected_float):
    findings = check_lowered_source(_fresh_kernel(body))
    if name in DOCUMENTED_UNKNOWN_CALL_LIMITS:
        assert findings == [], "an unknown call must stay unknown (documented limit), not be guessed"
    else:
        assert bool(findings) == expected_float, (name, [f.reason for f in findings])


@pytest.mark.parametrize('name,body,expected_float,expected_value', REVIEW6_CASES,
                         ids=[x[0] for x in REVIEW6_CASES])
def test_review6_controls_static_verdict(name, body, expected_float, expected_value):
    findings = check_lowered_source(_fresh_kernel(body))
    if name in DOCUMENTED_UNTYPED_ARGUMENT:
        assert findings == [], "an untyped rounding argument is unknown, never assumed float64 (ruling P3-R4)"
    else:
        assert bool(findings) == expected_float, (name, [f.reason for f in findings])


REVIEW6_UINT64_BOOL_IFEXP = 'j = np.uint64(1) if flag else True\nreturn X[j]'
REVIEW6_UINT64_INT64_MIN = 'j = min(np.uint64(1), np.int64(3))\nreturn X[j]'


def test_review6_uint64_bool_unifies_to_uint64_in_the_unification_model():
    # Numba's unify(uint64, boolean) is uint64 (checked on 0.61.2 and 0.68.1): a bool operand never
    # triggers the uint64/int64 -> float64 rule, only a SIGNED int64 operand does.  The checker is a
    # proof-of-float-ness diagnostic, so it must stay silent on the bool pairings (min, max and the
    # conditional expression alike) and keep reporting the signed pairing.
    for body in (REVIEW6_UINT64_BOOL_MIN, REVIEW6_UINT64_BOOL_MIN.replace('min', 'max'), REVIEW6_UINT64_BOOL_IFEXP):
        assert check_lowered_source(_fresh_kernel(body)) == [], body
    findings = check_lowered_source(_fresh_kernel(REVIEW6_UINT64_INT64_MIN))
    assert len(findings) == 1 and 'uint64' in findings[0].reason and 'float64' in findings[0].reason


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_review6_uint64_bool_min_numba_never_types_a_float_index():
    # Policy witness independent of Numba's overload set: numba < 0.68 has no min()/max() overload for a
    # Boolean operand and rejects the CALL (message names `min`/`max`, never a float getitem); numba >= 0.68
    # (`ol_min` accepts types.Boolean) compiles it with a uint64 index, so X[min(np.uint64(1), True)] is
    # X[1].  In neither release does a float64 index exist, and the checker reports none.
    import numba
    sig = (numba.types.Array(numba.float64, 1, 'C'), numba.float64, numba.boolean)
    X = np.arange(4, dtype=np.float64)
    for body in (REVIEW6_UINT64_BOOL_MIN, REVIEW6_UINT64_BOOL_MIN.replace('min', 'max')):
        env = {}
        exec(compile(_fresh_kernel(body), '<reviewer-uint64_bool_minmax>', 'exec'), env)
        fn = env['probe']
        try:
            fn.compile(sig)
        except numba.core.errors.TypingError as exc:
            msg = str(exc)
            assert ('min' in msg or 'max' in msg) and 'getitem' not in msg and 'float' not in msg, msg
        else:
            assert float(fn(X, 1.75, True)) == 1.0
        assert check_lowered_source(_fresh_kernel(body)) == [], body
    # the conditional-expression form compiles on every release (unify(uint64, boolean) == uint64)
    env = {}
    exec(compile(_fresh_kernel(REVIEW6_UINT64_BOOL_IFEXP), '<reviewer-uint64_bool_ifexp>', 'exec'), env)
    env['probe'].compile(sig)
    assert float(env['probe'](X, 1.75, True)) == 1.0 and float(env['probe'](X, 1.75, False)) == 1.0
    assert check_lowered_source(_fresh_kernel(REVIEW6_UINT64_BOOL_IFEXP)) == []
    # the SIGNED pairing is the real rule on every release: Numba rejects the float index, the checker reports it
    env = {}
    exec(compile(_fresh_kernel(REVIEW6_UINT64_INT64_MIN), '<reviewer-uint64_int64_min>', 'exec'), env)
    with pytest.raises(numba.core.errors.TypingError) as info:
        env['probe'].compile(sig)
    assert 'getitem' in str(info.value) and 'float' in str(info.value)
    assert len(check_lowered_source(_fresh_kernel(REVIEW6_UINT64_INT64_MIN))) == 1


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,expected_float,expected_value', REVIEW6_CASES,
                         ids=[x[0] for x in REVIEW6_CASES])
def test_review6_controls_against_numba(name, body, expected_float, expected_value):
    import numba
    src = _fresh_kernel(body)
    env = {}
    exec(compile(src, '<reviewer-' + name + '>', 'exec'), env)
    fn = env['probe']
    findings = check_lowered_source(src)
    err = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'), numba.float64, numba.boolean))
    except numba.core.errors.TypingError as exc:
        err = str(exc)
    if expected_float:
        assert err is not None and 'float' in err and ('getitem' in err or 'index' in err), err
    else:
        assert err is None, err
        assert float(fn(np.arange(4, dtype=np.float64), 1.75, True)) == float(expected_value)
    if name in DOCUMENTED_UNTYPED_ARGUMENT:
        assert findings == [] and err is not None     # Numba (x: float64) rejects; the checker cannot type x
    else:
        assert bool(findings) == expected_float, "checker disagrees with Numba index typing: " + name


# ---------------------------------------------------------------------------
# Independent reviewer round-2 repair-edge controls of the float-index repair
# (the reviewer's test_repair_edges.py CASES verbatim).  Columns: name, body,
# the Numba type of the kernel parameter `x`, whether Numba rejects a float64
# index, and the value returned for X = arange(4.0) when it compiles.
#   P2-R1 constant_join_*: equal values from different branches are an SSA
#          phi, not a constant;  P2-R2 *_carried_constant: a constant the loop
#          body reassigns;  P2-R3 *_integral_decimals: decimals keeps an
#          integral dtype;  P3-R4 round_unknown_*: untyped argument -> unknown.
# ---------------------------------------------------------------------------

REPAIR_EDGE_CASES = [
    ('for_carried_constant', 'a = 1\nz = 0.0\nfor i in range(2):\n    j = a or 0.5\n    z = X[j]\n    a = 0\nreturn z', 'float64', True, None),
    ('while_carried_constant', 'a = 1\ni = 0\nz = 0.0\nwhile i < 2:\n    j = a or 0.5\n    z = X[j]\n    a = 0\n    i += 1\nreturn z', 'float64', True, None),
    ('constant_join_float', 'if flag:\n    a = 0.5\nelse:\n    a = 0.5\nj = a and 1\nreturn X[j]', 'float64', True, None),
    ('constant_join_int', 'if flag:\n    a = 1\nelse:\n    a = 1\nj = a or 0.5\nreturn X[j]', 'float64', True, None),
    ('different_constant_join', 'if flag:\n    a = 1\nelse:\n    a = 0\nj = a or 0.5\nreturn X[j]', 'float64', True, None),
    ('unchanged_loop_constant', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    j = a and 1\n    z = X[j]\nreturn z', 'float64', False, 1),
    ('overwritten_constant', 'a = 0.5\na = x\nj = a and 1\nreturn X[j]', 'float64', True, None),
    ('augmented_constant', 'a = 0.5\na += x\nj = a and 1\nreturn X[j]', 'float64', True, None),
    ('round_unknown_int', 'j = np.round(x)\nreturn X[j]', 'int64', False, 1),
    ('around_unknown_int', 'j = np.around(x)\nreturn X[j]', 'int64', False, 1),
    ('round_unknown_float', 'j = np.round(x)\nreturn X[j]', 'float64', True, None),
    ('round_integral_decimals', 'j = np.round(np.int64(1), 0)\nreturn X[j]', 'float64', False, 1),
    ('around_integral_decimals', 'j = np.around(np.int64(1), decimals=0)\nreturn X[j]', 'float64', False, 1),
]
# Numba rejects these, but the value depends on the UNTYPED parameter `x`, which the checker cannot type:
# `overwritten_constant` (a = x; the pre-existing unknown-parameter limitation, documented, not fixed here)
# and `round_unknown_float` (ruling P3-R4: rounding an untyped value is unknown, never assumed float64).
REPAIR_EDGE_UNTYPED_PARAMETER = {'overwritten_constant', 'round_unknown_float'}


def _edge_kernel(body):
    return ('import numpy as np\nimport numba\n@numba.njit(boundscheck=True)\ndef probe(X, x, flag):\n'
            + textwrap.indent(body + '\n', '    '))


@pytest.mark.parametrize('name,body,xtype,float_index,expected_value', REPAIR_EDGE_CASES,
                         ids=[c[0] for c in REPAIR_EDGE_CASES])
def test_repair_edge_controls_static_verdict(name, body, xtype, float_index, expected_value):
    findings = check_lowered_source(_edge_kernel(body))
    if name in REPAIR_EDGE_UNTYPED_PARAMETER:
        assert findings == [], (name, [f.reason for f in findings])
    else:
        assert bool(findings) == float_index, (name, [f.reason for f in findings])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,xtype,float_index,expected_value', REPAIR_EDGE_CASES,
                         ids=[c[0] for c in REPAIR_EDGE_CASES])
def test_repair_edge_controls_against_numba(name, body, xtype, float_index, expected_value):
    import numba
    src = _edge_kernel(body)
    env = {}
    exec(compile(src, '<edge-' + name + '>', 'exec'), env)
    fn = env['probe']
    err = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'), getattr(numba, xtype), numba.boolean))
    except numba.core.errors.TypingError as exc:
        err = str(exc)
    if float_index:
        assert err is not None and 'Unsupported array index type float' in err, err
    else:
        assert err is None, err
        assert float(fn(np.arange(4, dtype=np.float64), 1 if xtype == 'int64' else 1.75, True)) == float(expected_value)
    findings = check_lowered_source(src)
    if name in REPAIR_EDGE_UNTYPED_PARAMETER:
        assert findings == [] and err is not None
    else:
        assert bool(findings) == float_index, 'checker disagrees with Numba index typing: ' + name


# ---------------------------------------------------------------------------
# Independent reviewer round-3 probes of the float-index repair (the
# reviewer's test_requested_probes.py and test_scope_and_dtypes.py CASES
# verbatim).  Columns: name, body, Numba rejects a float64 index, returned
# value for X = arange(32.0) when it compiles (reviewer's records).
#   P2-R3-U  round/around result kind: unify(kind(a), int64) -> uint64 is
#            float64, narrow unsigned is int64, signed is int64 (with or
#            without `decimals`; numba-round-return-types.json);
#   P2-R3-SCOPE  loop-constant invalidation sees the enclosing lexical scope
#            only: nested def / lambda bodies and comprehension targets bind
#            their own locals; nonlocal/global rebinds count.
# ---------------------------------------------------------------------------

ROUND3_PROBE_CASES = [
    ('same_definition_branch', 'a = 0.5\nif flag:\n    b = 1\nelse:\n    b = 2\nj = a and 1\nreturn X[j]', False, 1),
    ('same_definition_loop_reentry', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    if flag:\n        b = 1\n    else:\n        b = 2\n    j = a and 1\n    z = X[j]\nreturn z', False, 1),
    ('nested_loop_carried', 'a = 1\nz = 0.0\nfor i in range(2):\n    for k in range(2):\n        j = a or 0.5\n        z = X[j]\n        a = 0\nreturn z', True, None),
    ('nested_else_carried', 'a = 1\nz = 0.0\nfor i in range(2):\n    j = a or 0.5\n    z = X[j]\n    for k in range(1):\n        pass\n    else:\n        a = 0\nreturn z', True, None),
    ('own_else_preserves_body_constant', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    j = a and 1\n    z = X[j]\nelse:\n    a = 0.0\nreturn z', False, 1),
    ('while_else_preserves_body_constant', 'a = 0.5\nz = 0.0\ni = 0\nwhile i < 2:\n    j = a and 1\n    z = X[j]\n    i += 1\nelse:\n    a = 0.0\nreturn z', False, 1),
    ('prange_carried_constant', 'a = 1\nz = 0.0\nfor i in numba.prange(2):\n    j = a or 0.5\n    z = X[j]\n    a = 0\nreturn z', True, None),
    ('nested_unchanged_constant', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    for k in range(2):\n        j = a and 1\n        z = X[j]\nreturn z', False, 1),
    ('round_uint64_decimals', 'j = np.round(np.uint64(1), 0)\nreturn X[j]', True, None),
    ('around_uint64_decimals', 'j = np.around(np.uint64(1), decimals=0)\nreturn X[j]', True, None),
    ('round_uint8_signed_result', 'j = min(np.round(np.uint8(1), 0), np.uint64(2))\nreturn X[j]', True, None),
    ('around_uint32_signed_result', 'j = min(np.around(np.uint32(1), decimals=0), np.uint64(2))\nreturn X[j]', True, None),
    ('round_integral_negative_decimals', 'j = np.round(np.int64(15), -1)\nreturn X[j]', False, 20),
    ('around_integral_positive_decimals', 'j = np.around(np.int64(1), decimals=1)\nreturn X[j]', False, 1),
    ('round_static_float', 'j = np.round(np.float64(x))\nreturn X[j]', True, None),
    ('nested_function_local', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    def inner():\n        a = 0.0\n        return a\n    j = a and 1\n    z = X[j]\nreturn z', False, 1),
    ('comprehension_local', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    unused = [a for a in range(1)]\n    j = a and 1\n    z = X[j]\nreturn z', False, 1),
    # uint64 rounding is float64 even without `decimals` (probed; same overload branches)
    ('round_uint64_no_decimals', 'j = np.round(np.uint64(1))\nreturn X[j]', True, None),
    ('round_uint8_no_decimals_signed_min', 'j = min(np.round(np.uint8(1)), np.uint64(2))\nreturn X[j]', True, None),
]


def _probe32_kernel(body):
    return ('import numpy as np\nimport numba\n@numba.njit(boundscheck=True)\ndef probe(X, x, flag):\n'
            + textwrap.indent(body + '\n', '    '))


@pytest.mark.parametrize('name,body,float_index,expected_value', ROUND3_PROBE_CASES, ids=[c[0] for c in ROUND3_PROBE_CASES])
def test_round3_probes_static_verdict(name, body, float_index, expected_value):
    findings = check_lowered_source(_probe32_kernel(body))
    assert bool(findings) == float_index, (name, [f.reason for f in findings])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,float_index,expected_value', ROUND3_PROBE_CASES, ids=[c[0] for c in ROUND3_PROBE_CASES])
def test_round3_probes_against_numba(name, body, float_index, expected_value):
    import numba
    src = _probe32_kernel(body)
    env = {}
    exec(compile(src, '<round3-' + name + '>', 'exec'), env)
    fn = env['probe']
    err = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'), numba.float64, numba.boolean))
    except numba.core.errors.TypingError as exc:
        err = str(exc)
    if float_index:
        assert err is not None and 'Unsupported array index type float' in err, err
    else:
        assert err is None, err
        assert float(fn(np.arange(32, dtype=np.float64), 1.75, True)) == float(expected_value)
    assert bool(check_lowered_source(src)) == float_index, 'checker disagrees with Numba index typing: ' + name


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,expr,expected_type', [
    ('round_uint64', 'np.round(np.uint64(1), 0)', 'float64'), ('around_uint64', 'np.around(np.uint64(1), decimals=0)', 'float64'),
    ('round_uint8', 'np.round(np.uint8(1), 0)', 'int64'), ('around_uint32', 'np.around(np.uint32(1), decimals=0)', 'int64'),
    ('round_uint64_nodec', 'np.round(np.uint64(1))', 'float64'), ('round_int32_dec0', 'np.round(np.int32(1), 0)', 'int64'),
])
def test_rounding_result_type(name, expr, expected_type):
    # the Numba return type the checker models (reviewer's dtype probes + the no-decimals uint64 case)
    import numba
    source = 'import numpy as np\nimport numba\n@numba.njit\ndef scalar():\n    return ' + expr + '\n'
    env = {}
    exec(compile(source, '<dtype-' + name + '>', 'exec'), env)
    fn = env['scalar']
    fn.compile(())
    assert str(fn.overloads[()].signature.return_type) == expected_type


@pytest.mark.parametrize('decl,flagged', [('nonlocal', True), ('global', False)])
def test_nested_function_declaration_and_the_kernel_local(decl, flagged):
    # a nested def that declares the name `nonlocal` and assigns it IS an outer rebind of the kernel local
    # (constant dropped -> `a or 0.5` float-typed -> flagged); a nested `global a` targets the MODULE namespace
    # and never the kernel-local `a` (constant kept -> folds to 1 -> unflagged; P2-R5-GLOBAL corrects the
    # round-4 expectation).  Checker-only: Numba compiles neither closure shape.
    body = ('a = 1\nz = 0.0\nfor i in range(2):\n    def inner():\n        ' + decl + ' a\n        a = 0\n'
            '    j = a or 0.5\n    z = X[j]\n    inner()\nreturn z')
    findings = check_lowered_source(_probe32_kernel(body))
    assert bool(findings) == flagged, [f.reason for f in findings]


# ---------------------------------------------------------------------------
# Independent reviewer round-5 probes of the float-index repair (the
# reviewer's test_round5_probes.py CASES verbatim): kernel-local `a`
# collisions with nested nonlocal / global / lambda-default / class / import.
# ---------------------------------------------------------------------------

ROUND5_SCOPE_CASES = [
    ('two_intermediates_nonlocal', 'def first():\n    def second():\n        def inner():\n            nonlocal a\n            a = 0.0\n        inner()\n    second()\nfirst()', True),
    ('intermediate_declares_nonlocal', 'def middle():\n    nonlocal a\n    def inner():\n        nonlocal a\n        a = 0.0\n    inner()\nmiddle()', True),
    ('global_does_not_bind_kernel_local', 'def inner():\n    global a\n    a = 0.0', False),
    ('lambda_default_local', 'unused = lambda a=a: a', False),
    ('class_local', 'class C:\n    a = 0.0', False),
    ('import_alias', 'import math as a', True),
    ('from_import_alias', 'from math import floor as a', True),
]


@pytest.mark.parametrize('name,body,expected', ROUND5_SCOPE_CASES, ids=[c[0] for c in ROUND5_SCOPE_CASES])
def test_round5_enclosing_kernel_binding(name, body, expected):
    import ast
    from solar.einsum_lib.lowered_index_check import _assigned_names
    assert ('a' in _assigned_names(ast.parse(body).body)) == expected


def _r5_full_kernel(body):
    kernel_body = 'a = 0.5\nz = 0.0\nfor i in range(2):\n' + textwrap.indent(body + '\nj = a and 1\nz = X[j]\n', '    ') + 'return z\n'
    return 'import numba\nimport numpy as np\na = 7.0\n@numba.njit(boundscheck=True)\ndef probe(X):\n' + textwrap.indent(kernel_body, '    ')


@pytest.mark.parametrize('name,body,expected', ROUND5_SCOPE_CASES, ids=[c[0] for c in ROUND5_SCOPE_CASES])
def test_round5_full_kernel_static_verdict(name, body, expected):
    # the kernel-local `a = 0.5` (module `a = 7.0` shadowed): a rebind drops the constant and `a and 1` is float-typed
    findings = check_lowered_source(_r5_full_kernel(body))
    assert bool(findings) == expected, (name, [f.reason for f in findings])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,expected', ROUND5_SCOPE_CASES, ids=[c[0] for c in ROUND5_SCOPE_CASES])
def test_round5_full_kernel_against_numba(name, body, expected):
    import numba
    src = _r5_full_kernel(body)
    env = {}
    exec(compile(src, '<r5-' + name + '>', 'exec'), env)
    fn = env['probe']
    err = None
    value = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'),))
        value = float(fn(np.arange(4, dtype=np.float64)))
    except Exception as exc:
        err = type(exc).__name__ + ': ' + str(exc)
    float_index = err is not None and 'Unsupported array index type float' in err
    if err is not None and not float_index:
        pytest.skip('compiler rejection before an index oracle: ' + err.splitlines()[0][:120])
    if name in ('two_intermediates_nonlocal', 'intermediate_declares_nonlocal'):
        # reviewer-recorded Numba-vs-CPython divergence on nested nonlocal closures (observation, not a finding):
        # the checker's lexical expectation stands; only the compile outcome is recorded here
        return
    assert bool(check_lowered_source(src)) == float_index
    if not expected:
        assert value == 1.0


def test_module_constant_rebound_through_nested_global_is_invalidated():
    # module-level `a = 0.5` read by the kernel (no kernel-local `a`): without a rebind the float module constant
    # is a flagged index operand; a nested `global a; a = 1` inside the loop rebinds the MODULE name, so inside the
    # loop `a` is unknown and no finding is asserted (checker-only; Numba freezes globals and rejects the store)
    base = 'import numpy as np\nimport numba\na = 0.5\n@numba.njit\ndef probe(X):\n    z = 0.0\n    for i in range(2):\n'
    plain = base + '        j = a and 1\n        z = X[j]\n    return z\n'
    rebound = base + '        def inner():\n            global a\n            a = 1\n        inner()\n        j = a and 1\n        z = X[j]\n    return z\n'
    assert len(check_lowered_source(plain)) == 1 and 'module constant' in check_lowered_source(plain)[0].reason
    assert check_lowered_source(rebound) == []


# ---------------------------------------------------------------------------
# Independent reviewer round-4 probes of the float-index repair (the
# reviewer's test_round4_probes.py SCOPE_CASES / EXEC_CASES verbatim).
#   P2-R4-SCOPE  a `nonlocal` in a deeper function resolves to the nearest
#            enclosing function that binds the name: `middle` binding `a`
#            locally stops `inner`'s nonlocal there (the kernel's `a` keeps
#            its constant); when `middle` does NOT bind it, the nonlocal
#            passes through and rebinds the kernel's `a`.
#   P3-R4    `except ... as a` and `del a` are collected conservatively
#            (Numba rejects both kernels before index typing).
# ---------------------------------------------------------------------------

ROUND4_SCOPE_CASES = [
    ('nonlocal_rebind', 'def inner():\n    nonlocal a\n    a = 0.5', True),
    # a nested `global a` targets the MODULE namespace: it never rebinds the kernel-local `a` (P2-R5-GLOBAL; the
    # round-4 expectation `True` encoded the wrong semantics and is corrected here)
    ('global_rebind', 'def inner():\n    global a\n    a = 0.5', False),
    ('walrus_element', 'unused = [(a := 0.5) for _ in range(1)]', True),
    ('class_local', 'class C:\n    a = 0.0', False),
    ('with_alias', 'with contextlib.nullcontext(0.5) as a:\n    pass', True),
    ('except_alias', 'try:\n    pass\nexcept Exception as a:\n    pass', True),
    ('tuple_loop', 'for a, b in [(0.5, 1)]:\n    pass', True),
    ('delete_alias', 'del a', True),
    ('nested_else', 'for k in range(1):\n    pass\nelse:\n    a = 0.0', True),
    ('lambda_local', 'unused = lambda a: a', False),
    ('comprehension_target', 'unused = [a for a in range(1)]', False),
    ('nested_function_local', 'def inner():\n    a = 0.0\n    return a', False),
    ('deeper_nonlocal', 'def middle():\n    a = 0.0\n    def inner():\n        nonlocal a\n        a = 1.0\n    return a', False),
    # author controls for the scope rule
    ('deeper_nonlocal_passes_through', 'def middle():\n    def inner():\n        nonlocal a\n        a = 1.0\n    return inner', True),
    ('three_level_middle_nonlocal', 'def middle():\n    nonlocal a\n    def inner():\n        nonlocal a\n        a = 1.0\n    a = 2.0\n    return inner', True),
    ('three_level_inner_stops_at_middle_param', 'def middle(a):\n    def inner():\n        nonlocal a\n        a = 1.0\n    return a', False),
    ('nested_def_name_itself', 'def a():\n    return 0', True),
]


@pytest.mark.parametrize('name,body,binds_a', ROUND4_SCOPE_CASES, ids=[c[0] for c in ROUND4_SCOPE_CASES])
def test_enclosing_scope_binding(name, body, binds_a):
    import ast
    from solar.einsum_lib.lowered_index_check import _assigned_names
    assert ('a' in _assigned_names(ast.parse(body).body)) == binds_a


# (name, body, Numba outcome): 'float' = rejects a float64 index, 'ok' = compiles and returns the value,
# 'unsupported' = Numba rejects the construct before index typing (checker-only expectation given)
ROUND4_EXEC_CASES = [
    ('deeper_nonlocal', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    def middle():\n        a = 0.0\n        def inner():\n            nonlocal a\n            a = 1.0\n        return a\n    j = a and 1\n    z = X[j]\nreturn z', 'ok', 1, False),
    ('walrus_element', 'a = 1\nz = 0.0\nfor i in range(2):\n    j = a or 0.5\n    z = X[j]\n    unused = [(a := 0) for _ in range(1)]\nreturn z', 'float', None, True),
    ('nonlocal_rebind', 'a = 1\nz = 0.0\nfor i in range(2):\n    def inner():\n        nonlocal a\n        a = 0.5\n    j = a or 0.5\n    z = X[j]\n    inner()\nreturn z', 'float', None, True),
    ('class_local', 'a = 0.5\nz = 0.0\nfor i in range(2):\n    class C:\n        a = 0.0\n    j = a and 1\n    z = X[j]\nreturn z', 'unsupported', None, False),
    ('with_alias', 'a = 1\nz = 0.0\nfor i in range(2):\n    j = a or 0.5\n    z = X[j]\n    with contextlib.nullcontext(0) as a:\n        pass\nreturn z', 'unsupported', None, True),
    ('except_alias', 'a = 1\nz = 0.0\nfor i in range(1):\n    j = a or 0.5\n    z = X[j]\n    try:\n        raise ValueError()\n    except Exception as a:\n        pass\nreturn z', 'unsupported', None, True),
    ('tuple_loop', 'a = 1\nz = 0.0\nfor i in range(2):\n    j = a or 0.5\n    z = X[j]\n    for a, b in [(0, 1)]:\n        pass\nreturn z', 'float', None, True),
    # `del a` is collected conservatively: the constant is dropped and the float-typed `a and 1` IS reported
    # (checker-only expectation; Numba rejects the kernel before index typing)
    ('delete_alias', 'a = 0.5\nz = 0.0\nfor i in range(1):\n    j = a and 1\n    z = X[j]\n    del a\nreturn z', 'unsupported', None, True),
    # author control: middle does not bind `a`, so inner's nonlocal rebinds the kernel's `a`
    ('deeper_nonlocal_passes_through', 'a = 1\nz = 0.0\nfor i in range(2):\n    def middle():\n        def inner():\n            nonlocal a\n            a = 0\n        inner()\n    j = a or 0.5\n    z = X[j]\n    middle()\nreturn z', 'checker-only', None, True),
]


def _r4_kernel(body):
    return ('import numpy as np\nimport numba\nimport contextlib\n@numba.njit(boundscheck=True)\ndef probe(X):\n'
            + textwrap.indent(body + '\n', '    '))


@pytest.mark.parametrize('name,body,outcome,value,expect_finding', ROUND4_EXEC_CASES, ids=[c[0] for c in ROUND4_EXEC_CASES])
def test_round4_scope_static_verdict(name, body, outcome, value, expect_finding):
    findings = check_lowered_source(_r4_kernel(body))
    assert bool(findings) == expect_finding, (name, [f.reason for f in findings])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,outcome,value,expect_finding', ROUND4_EXEC_CASES, ids=[c[0] for c in ROUND4_EXEC_CASES])
def test_round4_scope_against_numba(name, body, outcome, value, expect_finding):
    import numba
    if outcome == 'checker-only':
        pytest.skip("Numba does not compile this closure shape; the static verdict is the control")
    src = _r4_kernel(body)
    env = {}
    exec(compile(src, '<r4-scope-' + name + '>', 'exec'), env)
    fn = env['probe']
    err = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'),))
    except Exception as exc:          # UnsupportedBytecodeError / CompilerError are not all NumbaError subclasses here
        err = type(exc).__name__ + ': ' + str(exc)
    if outcome == 'ok':
        assert err is None, err
        assert float(fn(np.arange(32, dtype=np.float64))) == float(value)
    elif outcome == 'float':
        assert err is not None and 'Unsupported array index type float' in err, err
    else:  # unsupported construct: rejected before index typing (bytecode / compiler error, not an index verdict)
        assert err is not None and 'Unsupported array index type float' not in err, err
    assert bool(check_lowered_source(src)) == expect_finding


def test_constant_bound_before_a_branch_and_untouched_still_folds():
    # the same definition on every path IS a constant (one statement, bound before the paths diverged)
    body = 'a = 0.5\nif flag:\n    b = 1\nelse:\n    b = 2\nj = a and 1\nreturn X[j]'
    assert check_lowered_source(_edge_kernel(body)) == []


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize('name,body,expected_float', FRESH_CASES, ids=[x[0] for x in FRESH_CASES])
def test_fresh_controls_against_numba(name, body, expected_float):
    import numba
    src = _fresh_kernel(body)
    env = {}
    exec(compile(src, '<fresh-' + name + '>', 'exec'), env)
    fn = env['probe']
    findings = check_lowered_source(src)
    err = None
    try:
        fn.compile((numba.types.Array(numba.float64, 1, 'C'), numba.float64, numba.boolean))
    except numba.core.errors.TypingError as exc:
        err = str(exc)
    if expected_float:
        assert err is not None and 'float' in err and ('getitem' in err or 'index' in err), err
    else:
        assert err is None, err
        assert float(fn(np.arange(4, dtype=np.float64), 1.75, True)) == 1.0
    if name in DOCUMENTED_UNKNOWN_CALL_LIMITS:
        assert findings == [] and err is not None      # Numba rejects it; the checker records a limit, not a verdict
    else:
        assert bool(findings) == expected_float, 'checker disagrees with Numba index typing: ' + name


def test_ci_unit_lane_selects_this_module():
    import pathlib
    ci = pathlib.Path(__file__).resolve().parents[1] / ".gitlab-ci.yml"
    assert "tests/test_lowered_index_check.py" in ci.read_text()
