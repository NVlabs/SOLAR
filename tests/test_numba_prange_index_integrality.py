# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lowered-kernel index expressions must type as integers inside Numba's parfor.

Finding SHAPE0-205-NUMBA-FLOAT-INDEX-1: the lowered row-205 program failed in
``native parfor lowering`` with ``getitem(array(float64, 3d, C),
Tuple(float64, float64, int64))``.  Numba's parfor gufunc iterates a
``uintp`` schedule, so inside the gufunc a ``numba.prange`` index is
``uint64`` while the outer typing sees ``int64``; ``min``/``max`` of that
index with a signed int unify to ``float64`` (plain binops stay ``int64``).
DSL ``Dim`` indices are Python ints, so the lowerer must rebind every prange
index to ``np.intp`` instead of handing the raw gufunc index to the body.

This module pins the EMITTER: the saved 205/206 kernel texts fail Numba's
typing exactly as recorded, the re-lowered kernels compile and match the
interpreter and a NumPy pad.  The static index-typing diagnostic
(``lowered_index_check``) lives in its own module and MR.
"""

import ast
import re

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
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
# The saved row-205 / row-206 kernels: the recorded failure, reproduced
# ---------------------------------------------------------------------------

def _compile_saved_pad_kernel(source: str, name: str):
    """Compile the saved kernel text for a float64 [1, 100, 8] input; return the TypingError text or None."""
    import numba
    ns = {}
    exec(compile(source, "executed-lowered.py", "exec"), ns)
    try:
        ns[name](np.zeros((1, 100, 8)))
    except numba.core.errors.TypingError as exc:
        return str(exc)
    return None


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestSavedKernelsReproduceTheFailure:
    @pytest.mark.parametrize("source, name, line", [(SAVED_205_LOWERED, "pad_upcast_fp32", 21),
                                                    (SAVED_206_LOWERED, "pad_rows_upcast_fp32", 27)],
                             ids=["205", "206"])
    def test_saved_kernel_text_fails_native_parfor_lowering_with_float_indices(self, source, name, line):
        err = _compile_saved_pad_kernel(source, name)
        assert err is not None, "the saved kernel compiled: the recorded failure did not reproduce"
        assert "native parfor lowering" in err and "Tuple(float64, float64, int64)" in err
        assert f"executed-lowered.py ({line})" in err

    def test_the_rebind_makes_the_saved_205_kernel_compile(self):
        fixed = SAVED_205_LOWERED.replace("    for m in numba.prange(mp):\n",
                                          "    for _m_prange in numba.prange(mp):\n        m = np.intp(_m_prange)\n")
        assert fixed != SAVED_205_LOWERED
        assert _compile_saved_pad_kernel(fixed, "pad_upcast_fp32") is None


# ---------------------------------------------------------------------------
# Emitter control: the lowered kernel has only integral index expressions
# ---------------------------------------------------------------------------

class TestEmitterRebindsPrangeIndex:
    def test_every_prange_loop_in_the_lowered_pad_kernel_is_rebound(self):
        lowered = numba_lower(PAD_KERNEL_DSL, boundscheck=True, use_fp64=False)
        assert lowered is not None
        tree = ast.parse(lowered)
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                 and ast.unparse(n.iter.func) == "numba.prange"]
        assert loops
        for loop in loops:
            first = loop.body[0]
            assert isinstance(first, ast.Assign) and ast.unparse(first.value) == f"np.intp({loop.target.id})"
            # the body never names the raw (uint64) gufunc index
            raw_uses = [n for n in ast.walk(ast.Module(body=loop.body[1:], type_ignores=[]))
                        if isinstance(n, ast.Name) and n.id == loop.target.id]
            assert raw_uses == []

    def test_lowered_pad_kernel_rebinds_the_dsl_name_to_intp(self):
        lowered = numba_lower(PAD_KERNEL_DSL, boundscheck=True, use_fp64=False)
        body = lowered.split("def pad_upcast_fp32", 1)[1]
        # the prange iterates a fresh name; the DSL name is an np.intp rebind
        m = re.search(r"for (\w+) in numba\.prange\(mp\):\n\s+m = np\.intp\((\w+)\)", body)
        assert m is not None, body
        assert m.group(1) == m.group(2) and m.group(1) != "m"
        # the DSL body itself is unchanged
        assert "t = min(m, nt - 1)" in body
        assert "Y[m, h] = X[t // ns, t % ns, h] * g" in body

    def test_fresh_prange_name_avoids_names_used_in_the_kernel(self):
        code = PAD_KERNEL_DSL.replace("ns = X.shape[1]", "_m_prange = 0\n    ns = X.shape[1]")
        lowered = numba_lower(code, boundscheck=True, use_fp64=False)
        assert lowered is not None
        tree = ast.parse(lowered)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "pad_upcast_fp32")
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Call)
                    and ast.unparse(n.iter.func) == "numba.prange")
        assert loop.target.id != "_m_prange"
        assert ast.unparse(loop.body[0]) == f"m = np.intp({loop.target.id})"

    def test_serial_dim_loops_are_not_rebound(self):
        lowered = numba_lower(PAD_KERNEL_DSL, boundscheck=True, use_fp64=False)
        body = lowered.split("def pad_upcast_fp32", 1)[1]
        assert "for h in range(X.shape[2]):" in body
        assert "h = np.intp" not in body

    def test_for_else_of_a_prange_loop_is_preserved(self):
        code = PAD_KERNEL_DSL.replace(
            "    return Y\n",
            "    else:\n        nt = nt\n    return Y\n", 1)
        lowered = numba_lower(code, boundscheck=True, use_fp64=False)
        assert lowered is not None
        assert "else:\n        nt = nt" in lowered

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_numba_compiles_and_matches_the_interpreter(self):
        # B=1, S=100 -> nt=100, mp=128: the pad/crop path of the row-205
        # adapter's capped shape, exactly the case that failed to compile.
        rng = np.random.RandomState(205)
        x = rng.uniform(-1.0, 1.0, (1, 100, 8)).astype(np.float32)
        # the kernel declares a bf16 input: put the samples on the bf16 grid
        # so the fast path's width truncation changes nothing
        x = (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)
        reasons = []
        fast = execute_fast(PAD_KERNEL_DSL, {"x": x}, use_fp64=False,
                            on_fallback=reasons.append)
        assert fast is not None, reasons
        np.testing.assert_array_equal(fast["output"], _pad_reference(x))
        slow, _ = execute_translated(PAD_KERNEL_DSL, {"x": (1, 100, 8)},
                                     all_inputs={"x": x}, fast=False)
        np.testing.assert_allclose(fast["output"], slow["output"], rtol=1e-6)
