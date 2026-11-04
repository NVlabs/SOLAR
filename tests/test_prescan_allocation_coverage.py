# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Issue #57 items 1 + 2 — prescan-coverage-32 and TARGET-TENSOR-POSITIONAL-1.

The literal-only lowering path accepts an allocation only in the ONE spelling it understands: the bare DSL
constructor ``Tensor(...)`` with explicit keywords (or the specification form ``Tensor("A,B", ...)``).  the reviewer's
round-10 matrix showed that keyword expansion (``**{"dtype": ...}``), local / imported aliases (``Factory = Tensor``)
reached a generic Numba TypingError / NameError fallback instead of a typed decline, and that invalid positional
arguments (``Tensor(None, "bf16", M=..., ...)``) were silently dropped by the lowering while the interpreter raises.
Now every such call is a typed decline BEFORE lowering: ``kernel_allocation_callee_unresolved:<kernel>`` (the callee is
not statically the DSL constructor) or ``kernel_allocation_signature_unresolved:<kernel>`` (expansion, or positional
arguments outside the constructor contract), with the interpreter's own outcome on the public path.  The literal
explicit-keyword programs lower exactly as before.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import numba_lower, execute_fast
from solar.einsum_lib.verify import execute_translated

ROOT = pathlib.Path(__file__).resolve().parent.parent
R3 = ROOT / "tests" / "fixtures" / "lowering_final_rounding" / "review3_controls"
FIX = ROOT / "tests" / "fixtures" / "lowering_final_rounding"
needs_numba = pytest.mark.skipif(numba_lower("x = 1\n") is None and execute_fast("def run_model(i):\n    return {}\n", {}) is None, reason="numba unavailable")

H = ('from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nimport numpy as np\nCOUNT = 0\nDT = "bf16"\n'
     'def choose_dtype():\n    global COUNT\n    COUNT += 1\n    return "bf16"\n')
K = '@kernel\ndef k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):\n'
Y = '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=COUNT + 1.003)\n    return Y\n'
RUN = '\ndef run_model(inputs):\n    clear_dims()\n    a = Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")\n    return {"output": k(a).data}\n'


def _program(body, prefix=""):
    return H + prefix + K + body + Y + RUN


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def _both_carriers(code):
    """(interpreter outcome, [(use_fp64, lowered?, decline reason, public outcome)])."""
    x = np.ones((1, 2))
    def run(fn):
        try:
            return {"value": np.asarray(fn()["output"])}
        except Exception as exc:
            return {"error": type(exc).__name__}
    slow = run(lambda: execute_translated(code, {}, all_inputs={"a": x}, fast=False)[0])
    out = []
    for use_fp64 in (True, False):
        seen = []
        low = numba_lower(code, use_fp64=use_fp64, on_decline=seen.append)
        public = run(lambda: execute_translated(code, {}, all_inputs={"a": x}, fast=True, use_fp64=use_fp64)[0])
        out.append((use_fp64, low is not None, seen[0] if seen else None, public))
    return slow, out


def _assert_typed_decline(code, prefix, needle=None, expect_value=None):
    slow, runs = _both_carriers(code)
    for use_fp64, lowered, reason, public in runs:
        assert not lowered and reason is not None and reason.startswith(prefix + ":k"), (use_fp64, reason)
        if needle:
            assert needle in reason, reason
        if "error" in slow:
            assert public.get("error") == slow["error"], (slow, public)          # the interpreter's own refusal reaches the caller
        else:
            np.testing.assert_array_equal(public["value"], slow["value"])          # fallback carrier is the interpreter's
    if expect_value is not None:
        np.testing.assert_array_equal(slow["value"], expect_value)
    return slow, runs


ALLOC = 'Tensor(M=A.shape[0], N=A.shape[1], dtype=choose_dtype(), fill=0)'


class TestCalleeUnresolved:
    @pytest.mark.parametrize("name,body,prefix", [
        ("alias_local", "    Factory = Tensor\n    T = Factory(M=A.shape[0], N=A.shape[1], dtype=choose_dtype(), fill=0)\n", ""),
        ("alias_import", "    T = Factory(M=A.shape[0], N=A.shape[1], dtype=choose_dtype(), fill=0)\n", "from solar.einsum_lib import Tensor as Factory\n"),
        ("attribute_callee", "    T = lib.Tensor(M=A.shape[0], N=A.shape[1], dtype=choose_dtype(), fill=0)\n", "import solar.einsum_lib as lib\n"),
        ("parameter_callee", "    T = make(M=A.shape[0], N=A.shape[1], dtype=choose_dtype(), fill=0)\n", "def make(**kw):\n    return Tensor(**kw)\n"),
        ("literal_dtype_alias", "    Factory = Tensor\n    T = Factory(M=A.shape[0], N=A.shape[1], dtype=\"fp32\", fill=0)\n", ""),
    ])
    def test_non_constructor_callee_with_keywords_declines(self, name, body, prefix):
        """the reviewer's alias cases (and an attribute / parameter callee): the callee is not statically the DSL
        constructor, so the lowering cannot tell whether it allocates or what dtype it would use -- typed decline,
        interpreter value on the public path (2.0 where choose_dtype() ran, 1.0 for the literal alias)."""
        slow, runs = _assert_typed_decline(_program(body, prefix), "kernel_allocation_callee_unresolved", "cannot compile or identify")
        np.testing.assert_array_equal(slow["value"], [[1.0, 1.0]] if name == "literal_dtype_alias" else [[2.0, 2.0]])

    def test_module_rebinding_of_tensor_declines(self):
        code = _program("    T = Tensor(M=A.shape[0], N=A.shape[1], dtype=\"fp32\", fill=0)\n", "real_tensor = Tensor\ndef Tensor(**kw):\n    return real_tensor(**kw)\n")
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "defines the name 'Tensor'")

    def test_kernel_local_rebinding_of_tensor_declines(self):
        code = _program("    Tensor = np.zeros\n    T = Tensor((1, 2))\n")
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "kernel_binding_rebound")


class TestSignatureUnresolved:
    @pytest.mark.parametrize("name,body", [
        ("kwargs_inline", '    T = Tensor(M=A.shape[0], N=A.shape[1], fill=0, **{"dtype": choose_dtype()})\n'),
        ("kwargs_name", '    options = {"dtype": choose_dtype()}\n    T = Tensor(M=A.shape[0], N=A.shape[1], fill=0, **options)\n'),
        ("kwargs_literal", '    options = {"dtype": "fp32"}\n    T = Tensor(M=A.shape[0], N=A.shape[1], fill=0, **options)\n'),
        ("starargs", '    dims = ("M,N",)\n    T = Tensor(*dims, dtype="fp32")\n'),
    ])
    def test_expansion_declines(self, name, body):
        """the reviewer's keyword-expansion cases: the keywords (dtype included) are not visible statically.  In the
        ``kwargs_name`` spelling the helper call in the preceding statement (``choose_dtype()``) is reached first and
        is itself a typed callee decline -- either way no generic fallback."""
        if name == "kwargs_name":
            _assert_typed_decline(_program(body), "kernel_allocation_callee_unresolved", "cannot compile or identify")
        else:
            _assert_typed_decline(_program(body), "kernel_allocation_signature_unresolved", "expands *args / **kwargs")

    @pytest.mark.parametrize("name,body,expect_error", [
        ("positional_dtype", '    T = Tensor(None, choose_dtype(), M=A.shape[0], N=A.shape[1], fill=0)\n', True),
        ("positional_dtype_literal", '    T = Tensor(None, "bf16", M=A.shape[0], N=A.shape[1], fill=0)\n', True),
        ("positional_shape_tuple", '    T = Tensor((1, 2), dtype="fp32")\n', True),
        ("dims_string_plus_keyword_dims", '    T = Tensor("M,N", M=A.shape[0], N=A.shape[1], dtype="fp32")\n', True),
    ])
    def test_positional_arguments_outside_the_contract_decline(self, name, body, expect_error):
        """TARGET-TENSOR-POSITIONAL-1: the interpreter raises TypeError on these; the lowering used to drop the
        extra argument and allocate anyway.  Now a typed decline, and the public path reports the interpreter's own
        error."""
        slow, runs = _assert_typed_decline(_program(body), "kernel_allocation_signature_unresolved")
        assert ("error" in slow) is expect_error, slow


class TestLiteralSpellingsStillLower:
    @needs_numba
    @pytest.mark.parametrize("body", [
        '    T = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=0)\n',
        '    spec = Tensor("M,N", dtype="fp32")\n',
        '    for m in Dim("M", parallel=True):\n        pass\n',
        '    n = Dim("N", size=A.shape[1])\n',
    ], ids=["keyword-allocation", "spec-form", "Dim-parallel", "Dim-size"])
    def test_literal_programs_lower_without_decline(self, body):
        slow, runs = _both_carriers(_program(body))
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, (use_fp64, reason)
            exp = slow["value"] if use_fp64 else slow["value"].astype(np.float32)
            np.testing.assert_array_equal(public["value"], exp)
        np.testing.assert_array_equal(slow["value"], [[1.0, 1.0]])

    @needs_numba
    @pytest.mark.parametrize("dt,fill", [("bf16", "1.003"), ("fp16", "1.003"), ("fp32", "1.003"), ("fp64", "1.003"), ("int32", "1.75")])
    def test_seven_literal_controls_unchanged(self, dt, fill):
        """the reviewer's literal controls (five dtypes here; nested call and side-effect default below) lower in both
        carriers with exact carrier-cast equality."""
        code = H + K + f'    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="{dt}", fill={fill})\n    return Y\n' + RUN
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, (dt, reason)
            exp = slow["value"]
            if not use_fp64 and exp.dtype.kind == "f":
                exp = exp.astype(np.float32)
            assert np.array_equal(public["value"], exp)

    @needs_numba
    @pytest.mark.parametrize("variant", ["nested_call", "side_effect_default"])
    def test_nested_call_and_side_effect_default(self, variant):
        body = '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n'
        code = H + K + body + RUN
        if variant == "nested_call":
            code = code.replace("k(a).data", "k(k(a)).data")
        else:
            code = code.replace('DT = "bf16"\n', 'DT = "bf16"\ndef effect():\n    global COUNT\n    COUNT += 1\n    return COUNT\ndef marker(x=effect()):\n    pass\n')
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, (variant, reason)


class TestCalleeIdentityByBinding:
    """an earlier fix round 2 (the review): the constructor is identified by BINDING and identity, never by the spelling
    ``Tensor``.  A parameter or nested def named Tensor, a module import that binds the name to another symbol, a
    star import or no import at all cannot establish the DSL class -- typed decline with the interpreter's own
    outcome (its value, or its TypeError); the genuine imports, module-level or kernel-local, still lower."""

    def test_parameter_named_tensor_declines(self):
        code = (H + 'def factory(M, N, dtype, fill):\n    return Tensor(M=M, N=N, dtype=dtype, fill=7)\n'
                + '@kernel\ndef k(A: Tensor("M,N", dtype="fp32"), Tensor) -> Tensor("M,N"):\n'
                + '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=2.003)\n    return Y\n' + RUN.replace("k(a)", "k(a, factory)"))
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "parameter of the kernel is named 'Tensor'")
        np.testing.assert_array_equal(slow["value"], [[7.0, 7.0]])

    def test_nested_def_named_tensor_declines(self):
        code = _program('    def Tensor(M, N, dtype, fill):\n        return A\n')
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "defines the name 'Tensor'")

    def test_nested_def_parameter_named_tensor_is_the_nested_scopes_own(self):
        """lexical-scope-6 (the review, round 2 regression): a parameter of a nested def is bound in the NESTED
        scope only; the kernel's ``Tensor`` is untouched and the program lowers (round 2 declined it).  A nested
        scope that declares the name ``nonlocal`` / ``global`` and assigns it may rebind the enclosing name, so
        that form still declines."""
        code = _program('    def inner(Tensor):\n        return 2\n    unused = inner(3)\n')
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, reason
            np.testing.assert_array_equal(public["value"], [[1.0, 1.0]])
        code = _program('    def inner():\n        global Tensor\n        Tensor = 1\n    inner()\n')
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "kernel_binding_uncertain")
        # Round 5: ``nonlocal Tensor`` with no binding in an enclosing def is NOT a compilable program (Python: "no binding
        # for nonlocal 'Tensor' found"); rounds 2-4 pinned it as kernel_binding_uncertain before the source was ever
        # compiled.  It is now the typed kernel_source_invalid decline, the interpreter's error preserved as before; a
        # VALID nonlocal closure (the binding belongs to an enclosing def inside the kernel) keeps the conservative
        # kernel_binding_uncertain of the kernel scan.
        code = _program('    def inner():\n        nonlocal Tensor\n        Tensor = 1\n    inner()\n')
        _assert_typed_decline(code, "kernel_source_invalid", "SyntaxError: no binding for nonlocal 'Tensor' found")
        code = _program('    def outer():\n        Tensor = 1\n        def inner():\n            nonlocal Tensor\n            Tensor = 2\n        inner()\n    outer()\n')
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "kernel_binding_uncertain")

    def test_global_declaration_declines(self):
        code = _program('    global Tensor\n    T = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=0)\n')
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "declares 'Tensor' global")

    def test_import_of_the_wrong_symbol_as_tensor_declines(self):
        """the reviewer's IMPORT-IDENTITY-2: ``from solar.einsum_lib import Dim as Tensor`` -- the module is right, the
        symbol is not; the interpreter raises TypeError, the fast path used to return [[7, 7]]."""
        code = ('from solar.einsum_lib import Dim as Tensor, kernel\n@kernel\ndef k(A):\n'
                '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=7)\n    return Y\n'
                'def run_model(inputs):\n    return {"output": k(inputs["a"]).data}\n')
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "not the DSL Tensor class")
        assert slow["error"] == "RuntimeError"

    @pytest.mark.parametrize("prefix", ["from solar.einsum_lib import *\n", "import solar.einsum_lib\nTensor = solar.einsum_lib.Tensor\n", "from solar.einsum_lib import Tensor as T0\nTensor = T0\n"],
                             ids=["star-import", "assignment-from-module", "assignment-from-alias"])
    def test_other_module_bindings_decline(self, prefix):
        code = prefix + "from solar.einsum_lib import kernel, clear_dims\nimport numpy as np\nCOUNT = 0\n" + K + '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n' + RUN
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved")

    @needs_numba
    @pytest.mark.parametrize("prefix,body", [
        ("from solar.einsum_lib import Tensor, kernel, clear_dims\nimport numpy as np\nCOUNT = 0\n", ""),
        ("from solar.einsum_lib.core import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\nimport numpy as np\nCOUNT = 0\n", ""),
        ("from solar.einsum_lib import Tensor, kernel, clear_dims\nimport numpy as np\nCOUNT = 0\n", "    from solar.einsum_lib import Tensor\n"),
    ], ids=["canonical-import", "core-import", "kernel-local-import-of-the-real-class"])
    def test_identity_verified_imports_lower(self, prefix, body):
        """The genuine class, however imported (module or kernel scope), is established by identity and lowers;
        the review corrected its own over-demanding expectation on the kernel-local import."""
        code = prefix + K + body + '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n' + RUN
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, reason
        np.testing.assert_array_equal(slow["value"], [[1.0, 1.0]])


class TestHelperBoundary:
    @pytest.mark.parametrize("call", ["helper(A)", "helper(A=A)"], ids=["positional", "keyword"])
    def test_plain_module_helper_declines_either_way(self, call):
        """the reviewer's HELPER-BOUNDARY-3: a plain module helper is kept as a Python function by the lowering, which
        compiled code cannot call; positional and keyword spellings now get the same typed decline (the interpreter
        value 2.0 on the public path) instead of a generic numba TypingError fallback for one of them."""
        code = _program(f'    Tmp = {call}\n', 'def helper(A):\n    Tmp = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=3)\n    return Tmp\n')
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "cannot compile or identify")
        np.testing.assert_array_equal(slow["value"], [[1.0, 1.0]])          # fill = COUNT + 1.003 with COUNT = 0, rounded to bf16

    @needs_numba
    def test_other_kernel_and_lowering_builtins_still_lower(self):
        code = (H + '@kernel\ndef inner(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):\n'
                '    Z = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=0)\n    for m in Dim("M"):\n        for n in Dim("N"):\n'
                '            Z[m, n] = max(abs(A[m, n]), 0.0) + exp(0.0) + float(int(1))\n    return Z\n'
                + K + '    T = inner(A)\n    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)\n    return Y\n' + RUN)
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert lowered and reason is None, reason

    def test_computed_and_attribute_callees_decline(self):
        for body in ['    Tmp = getattr(mod, "Tensor")(M=1, N=2, dtype="fp32")\n', '    Tmp = mod.Tensor(M=1, N=2, dtype="fp32")\n']:
            _assert_typed_decline(_program(body, "import solar.einsum_lib as mod\n"), "kernel_allocation_callee_unresolved")


class TestReview4EdgesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_prescan_edges.py`` byte-for-byte (sha256 b828f21e…), 22 cases: 18 pass.  The four pinned
        deviations are the file's own expectations that the reviewed ruling overrides: ``helper_positional`` x 2
        (the file expects the old generic-fallback boundary; it now gets the typed decline, consistently with the
        keyword spelling) and ``kernel_import_tensor`` x 2 (the file expects a decline for a kernel-local import of the
        real class; the review itself noted that expectation as over-demanding, and the program lowers)."""
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        (run / "test_prescan_edges.py").write_bytes((R3 / "test_prescan_edges.py").read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_prescan_edges.py"), "-q", "-p", "no:cacheprovider", "-rf"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        ids = sorted(l.split("::", 1)[1].split(" - ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED") and "::" in l)
        assert ids == sorted(f"test_boundary[{m}-{n}]" for n in ("helper_positional", "kernel_import_tensor") for m in ("False", "True")), res.stdout[-4000:]
        assert "4 failed, 18 passed" in res.stdout, res.stdout[-400:]
        import json
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "edges-head").glob("*.json")}
        for case in ("parameter_tensor", "nested_def_tensor", "import_wrong_symbol_as_tensor", "import_T2_inside", "getattr_tensor", "dim_kwargs", "dim_starargs", "helper_positional", "helper_keywords"):
            for fp in ("True", "False"):
                r = recs[f"{case}-{fp}"]
                assert not r["lowered"] and r["declines"] and r["declines"][0].startswith("kernel_allocation_"), (case, r["declines"])
        for case in ("kernel_import_tensor", "dims_string_fill"):
            for fp in ("True", "False"):
                assert recs[f"{case}-{fp}"]["lowered"] and not recs[f"{case}-{fp}"]["declines"], case


class TestReview4Round10Verbatim:
    def test_file(self, tmp_path):
        """the reviewer's twelve-case matrix byte-for-byte (sha256 f3289e5e…): 24 passed -- the eight expansion / alias
        cases and the four positional cases now reach their outcomes through the typed declines (recorded in the
        file's own runs/), the three dynamic-dtype cases keep ``kernel_output_dtype_unestablished``, the three literal
        cases lower."""
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        (run / "test_review4_round10.py").write_bytes((R3 / "test_review4_round10.py").read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_review4_round10.py"), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "24 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-800:]
        import json
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round10").glob("*.json")}
        for case in ("kwargs_inline", "kwargs_name", "alias_local", "alias_import"):
            for fp in ("True", "False"):
                r = recs[f"{case}-head-{fp}"]
                assert not r["lowered"] and r["reasons"] and r["reasons"][0].startswith("kernel_allocation_" ), (case, r["reasons"])
        for case in ("positional_dtype", "positional_dtype_literal"):
            for fp in ("True", "False"):
                r = recs[f"{case}-head-{fp}"]
                assert not r["lowered"] and r["reasons"][0].startswith("kernel_allocation_signature_unresolved:k"), (case, r["reasons"])


# ---------------------------------------------------------------------------------------------------------------------
# an earlier fix round 3 (the review on an earlier revision): MODULE-BINDING-5, LEXICAL-SCOPE-6, REEXPORT-POLICY-7, BASELINE-UNANNOTATED-NARROW-8
# ---------------------------------------------------------------------------------------------------------------------
import importlib.util
import json

UH = 'from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nimport numpy as np\n'
UK = '@kernel\ndef k(A):\n'
UY = '    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=2.003)\n    return Y\n'
URUN = '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n'


def _unannotated(body="", prefix="", header=UH, run=URUN):
    """the reviewer's unannotated probe shape: ``@kernel def k(A)`` with a literal bf16 allocation, NumPy input."""
    return header + prefix + UK + body + UY + run


def _lowers_to(code, expected):
    """Both carriers lower with no decline and no fallback, and the fast path itself (not the interpreter) returns
    ``expected``."""
    slow, runs = _both_carriers(code)
    for use_fp64, lowered, reason, public in runs:
        assert lowered and reason is None, (use_fp64, reason)
        fell = []
        raw = execute_fast(code, {"a": np.ones((1, 2))}, use_fp64=use_fp64, on_fallback=fell.append)
        assert not fell, fell
        np.testing.assert_array_equal(np.asarray(raw["output"]), expected)
        np.testing.assert_array_equal(public["value"], expected)
    np.testing.assert_array_equal(slow["value"], expected)
    return slow, runs


def _refuses(code, code_name, kind, needle, side=None):
    """Round 8: on BOTH carriers the lowering is the typed refusal ``<code_name>:k: <kind>: ...`` containing ``needle``,
    raw fast execution reports the same reason as its fallback, and the public path answers exactly as the interpreter
    does (value or error; ``side`` is the module side value when the program has one)."""
    x = np.ones((1, 2))

    def run(fn):
        try:
            return {"value": fn()}
        except Exception as exc:
            return {"error": type(exc).__name__}
    slow = run(lambda: execute_translated(code, {}, all_inputs={"a": x}, fast=False)[0])
    reasons = []
    for use_fp64 in (True, False):
        seen = []
        assert numba_lower(code, boundscheck=True, use_fp64=use_fp64, on_decline=seen.append) is None, (use_fp64, "lowered")
        assert len(seen) == 1 and seen[0].startswith(f"{code_name}:k: {kind}:") and needle in seen[0], (use_fp64, kind, needle, seen)
        fell = []
        assert execute_fast(code, {"a": x}, use_fp64=use_fp64, on_fallback=fell.append) is None and fell == seen, (use_fp64, fell)
        public = run(lambda: execute_translated(code, {}, all_inputs={"a": x}, fast=True, use_fp64=use_fp64)[0])
        if "error" in slow:
            assert public.get("error") == slow["error"], (use_fp64, public, slow)
        else:
            assert "value" in public, (use_fp64, public)
            np.testing.assert_array_equal(np.asarray(public["value"]["output"]), np.asarray(slow["value"]["output"]))
            if side is not None:
                assert float(np.asarray(public["value"]["side"])) == side == float(np.asarray(slow["value"]["side"])), (use_fp64, public, slow)
        reasons.append(seen[0])
    return reasons


class TestRound3ModuleBinding:
    """module-binding-5: the module scan follows EVERY module-scope statement through all control flow and
    treats every binding construct as a binding; namespace rewrites it cannot follow are uncertain.  Each form below
    reproduced the reviewer's silent [[2.003, 2.003]] on an earlier revision while the interpreter raises; now a typed decline whose
    detail starts with the sub-reason, and the interpreter's error reaches the caller."""

    @pytest.mark.parametrize("prefix,sub", [
        ("if True:\n    Tensor = int\n", "module_binding_rebound"),
        ("for Tensor in [int]:\n    pass\n", "module_binding_rebound"),
        ("match int:\n    case Tensor:\n        pass\n", "module_binding_rebound"),
        ("while False:\n    pass\nelse:\n    Tensor, _ = int, 0\n", "module_binding_rebound"),
        ("try:\n    pass\nexcept Exception as Tensor:\n    pass\n", "module_binding_rebound"),
        ("with open(__file__) as Tensor:\n    pass\n", "module_binding_rebound"),
        ("_ = (Tensor := int)\n", "module_binding_rebound"),
        ("exec(\"Tensor=int\")\n", "module_binding_uncertain"),
        ("globals()[\"Tensor\"] = int\n", "module_binding_uncertain"),
        ("import types\nM = types.ModuleType(\"M\")\nM.Tensor = int\n", "module_binding_uncertain"),        # a throwaway module: the probe must not mutate the real one
        ("import types\nE = types.ModuleType(\"E\")\nE.__dict__[\"Tensor\"] = int\n", "module_binding_uncertain"),
        ("del Tensor\n", "module_binding_deleted"),
    ], ids=["if", "for", "match", "while-else-unpack", "except-as", "with-as", "walrus", "exec", "globals", "attribute-store", "__dict__", "del"])
    def test_module_scope_forms_decline(self, prefix, sub):
        code = _unannotated(prefix=prefix)
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", sub)
        for use_fp64, lowered, reason, public in runs:
            assert reason.startswith(f"kernel_allocation_callee_unresolved:k: {sub}:"), reason

    def test_class_body_binding_is_the_class_scopes_own(self):
        """``class Holder: Tensor = int`` binds a class attribute, not the module name: the program lowers and the
        unannotated literal bf16 result is rounded (UNANNOTATED-NARROW-8)."""
        _lowers_to(_unannotated(prefix="class Holder:\n    Tensor = int\n"), [[2.0, 2.0]])

    def test_subcode_is_carried_by_the_exception(self):
        from solar.einsum_lib import numba_accel as na
        import ast as _ast
        tree = _ast.parse(_unannotated(prefix="if True:\n    Tensor = int\n"))
        node = next(n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "k")
        with pytest.raises(na.KernelOutputDtypeUnresolved) as info:
            na._reject_unresolved_tensor_binding(node, tree)
        assert info.value.subcode == "module_binding_rebound" and info.value.reason == "kernel_allocation_callee_unresolved:k"


class TestRound3LexicalScope:
    """lexical-scope-6: names confined to nested scopes do not rebind the kernel's ``Tensor``."""

    @pytest.mark.parametrize("body", [
        '    def helper(Tensor):\n        return 2\n    unused = helper(3)\n',
        '    unused = lambda Tensor: 2\n',
        '    def helper():\n        Tensor = 1\n        return 2\n    unused = helper()\n',
        '    Tmp = [Tensor for Tensor in [int]]\n',       # round 4 (COMPREHENSION-SCOPE-11): a comprehension target is comprehension-local; round 3 pinned this as a decline, wrongly
        '    unused = lambda: (Tensor := int)\n',        # round 4 (LAMBDA-SCOPE-12): a walrus in a lambda BODY binds the lambda's scope
    ], ids=["nested-parameter", "lambda-parameter", "nested-local-assignment", "comprehension-target", "lambda-body-walrus"])
    def test_nested_scope_names_lower(self, body):
        _lowers_to(_unannotated(body), [[2.0, 2.0]])

    @pytest.mark.parametrize("body,sub", [
        ('    def outer():\n        Tensor = 1\n        def helper():\n            nonlocal Tensor\n            Tensor = 2\n        helper()\n    outer()\n', "kernel_binding_uncertain"),   # round 5: a VALID nonlocal closure (the round-3 form had no enclosing binding and never compiled: now kernel_source_invalid)
        ('    def Tensor(M, N, dtype, fill):\n        return A\n', "kernel_binding_rebound"),
        ('    Tensor = lambda **kw: A\n', "kernel_binding_rebound"),
        ('    match int:\n        case Tensor:\n            pass\n', "kernel_binding_rebound"),
        ('    from solar.einsum_lib import Tensor\n    del Tensor\n', "kernel_binding_deleted"),
        ('    exec("global Tensor; Tensor=int")\n', "kernel_binding_uncertain"),
        ('    globals()["Tensor"] = int\n', "kernel_binding_uncertain"),
        ('    for Tensor in [int]:\n        pass\n', "kernel_binding_rebound"),
        ('    Tmp = [(Tensor := int) for _ in [1]]\n', "kernel_binding_rebound"),      # round 4: a walrus INSIDE a comprehension binds the enclosing scope (PEP 572)
        ('    unused = lambda x=(Tensor := int): x\n', "kernel_binding_rebound"),     # round 4: a lambda DEFAULT is evaluated in the enclosing scope
    ], ids=["nonlocal-assign", "nested-def-named-Tensor", "lambda-bound", "match-capture", "del", "exec", "globals", "for-target", "comprehension-walrus", "lambda-default-walrus"])
    def test_kernel_scope_bindings_still_decline(self, body, sub):
        slow, runs = _assert_typed_decline(_unannotated(body), "kernel_allocation_callee_unresolved", sub)


class TestRound3ReexportPolicy:
    """reexport-policy-7: the prescan import allowlist (``solar.einsum_lib`` and submodules) is kept, and the
    reason is truthful: a re-export from outside the allowlist is ``import_disallowed_prefix:<module>`` (identity not
    checked), a wrong object under the allowlist is ``import_identity_mismatch:<module>.<symbol>``, a missing module
    ``import_failed:<module>``, a relative import ``import_unresolvable_relative``; the identity-preserving re-export
    INSIDE the allowlist lowers."""

    @pytest.fixture
    def probe_modules(self, tmp_path):
        mods = {"mr124r3_external_reexport": "from solar.einsum_lib import Tensor\n",
                "solar.einsum_lib._mr124r3_reexport": "from solar.einsum_lib import Tensor\n",
                "solar.einsum_lib._mr124r3_subclass": ("from solar.einsum_lib import Tensor as BaseTensor\nclass Tensor(BaseTensor):\n"
                                                      "    def __init__(self, *a, **kw):\n        kw['fill'] = 7\n        super().__init__(*a, **kw)\n")}
        import sys as _sys
        for name, src in mods.items():
            p = tmp_path / (name + ".py"); p.write_text(src)
            spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec)
            _sys.modules[name] = m; spec.loader.exec_module(m)
        from solar.einsum_lib import Tensor as _T
        assert _sys.modules["mr124r3_external_reexport"].Tensor is _T and _sys.modules["solar.einsum_lib._mr124r3_reexport"].Tensor is _T
        yield mods
        for name in mods:
            _sys.modules.pop(name, None)

    def test_external_reexport_declines_by_prefix_policy(self, probe_modules):
        code = _unannotated(header="from mr124r3_external_reexport import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\n")
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "import_disallowed_prefix:mr124r3_external_reexport")
        for use_fp64, lowered, reason, public in runs:
            assert "identity was not checked" in reason and "not the DSL Tensor class" not in reason, reason
        np.testing.assert_array_equal(slow["value"], [[2.0, 2.0]])          # the object IS the class; the interpreter is right

    def test_reexport_inside_the_allowlist_lowers(self, probe_modules):
        code = _unannotated(header="from solar.einsum_lib._mr124r3_reexport import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\n")
        _lowers_to(code, [[2.0, 2.0]])

    def test_subclass_inside_the_allowlist_is_an_identity_mismatch(self, probe_modules):
        code = _unannotated(header="from solar.einsum_lib._mr124r3_subclass import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\n")
        slow, runs = _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "import_identity_mismatch:solar.einsum_lib._mr124r3_subclass.Tensor")
        np.testing.assert_array_equal(slow["value"], [[7.0, 7.0]])

    def test_missing_module_and_relative_import_reasons(self):
        code = _unannotated(header="from solar.einsum_lib._mr124r3_missing import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\n")
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "import_failed:solar.einsum_lib._mr124r3_missing")
        code = _unannotated(header='__package__ = "solar.einsum_lib"\nfrom .core import Tensor\nfrom solar.einsum_lib import kernel, clear_dims\n')
        _assert_typed_decline(code, "kernel_allocation_callee_unresolved", "import_unresolvable_relative")

    def test_allowlist_is_documented(self):
        from solar.einsum_lib import numba_accel as na
        assert na._DSL_IMPORT_MODULES == ("solar.einsum_lib",)
        assert "import_disallowed_prefix" in (na.__doc__ or "") and "allowlist" in (na.__doc__ or "")


class TestRound3UnannotatedNarrow:
    """BASELINE-UNANNOTATED-NARROW-8: the call-site rounding follows the ESTABLISHED output dtype (return-site
    provenance), not the presence of a return-dimension annotation.  On an earlier revision an unannotated literal
    ``dtype="bf16", fill=2.003`` kernel returned 2.003 from the fast path; the interpreter returns 2.0."""

    @pytest.mark.parametrize("dt,fill,expected", [("bf16", 2.003, 2.0), ("fp16", 2.0007, 2.0), ("bf16", 1.003, 1.0)])
    def test_unannotated_literal_narrow_result_is_rounded(self, dt, fill, expected):
        code = _unannotated().replace('dtype="bf16", fill=2.003', f'dtype="{dt}", fill={fill}')
        _lowers_to(code, [[expected, expected]])

    @pytest.mark.parametrize("run", [
        '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": np.asarray(k(inputs["a"]).data)}\n',
        '\ndef run_model(inputs):\n    clear_dims()\n    y = k(inputs["a"])\n    return {"output": y.data}\n',
        '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": np.asarray(k(inputs["a"]).data) + 0.0}\n',
    ], ids=["call-as-argument", "assigned-then-returned", "nested-in-expression"])
    def test_every_call_site_form_is_rounded(self, run):
        _lowers_to(_unannotated(run=run), [[2.0, 2.0]])

    def test_lowered_text_rounds_the_unannotated_call_and_not_a_wide_one(self):
        low = numba_lower(_unannotated(), use_fp64=True)
        assert low is not None and "_truncate_bf16(k(" in low.replace(" ", ""), low
        wide = numba_lower(_unannotated().replace('dtype="bf16"', 'dtype="fp32"'), use_fp64=True)
        assert wide is not None and "_truncate" not in wide, wide

    def test_unestablished_provenance_still_declines(self):
        """When the return cannot be traced to a literal allocation the kernel is declined, never returned unrounded."""
        code = _unannotated(body='    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=2.003)\n    Z = Y.data\n    return Z\n').replace(UY, "")
        slow, runs = _both_carriers(code)
        for use_fp64, lowered, reason, public in runs:
            assert not lowered and reason is not None and reason.startswith("kernel_output_dtype_unestablished:k"), reason


class TestReview4BindingMatrixVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's ``test_binding_edges.py`` (sha256 c5c47226…, 23 cases x 2 carriers) and
        ``test_binding_positive_v2.py`` (sha256 9113cf4b…, 6 annotated positives x 2) byte-for-byte: 52 pass, 6 fail,
        and the six are exactly the two expectations the reviewed rulings override -- ``reexport_external_module``
        x 2 carriers x 2 files (the file expects an identity-preserving re-export from OUTSIDE the allowlist to lower;
        REEXPORT-POLICY-7 keeps the prefix boundary, with the truthful ``import_disallowed_prefix`` reason and the
        interpreter's 2.0 on the public path) and ``module_attribute_constructor`` x 2 (the file expects the
        callee-specific reason; ``E.Tensor(...)`` is refused earlier by return provenance with
        ``kernel_output_dtype_unestablished``, which the review itself recorded as its own over-demanding expectation).
        The six module-binding forms, the nested-scope positives and the unannotated roundings are read back from the
        files' own records."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_binding_positive_v2.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_binding_edges.py", "test_binding_positive_v2.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        ids = sorted(l.split("::", 1)[1].split(" - ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED") and "::" in l)
        expected = sorted([f"test_binding_contract[{fp}-{c}]" for fp in ("False", "True") for c in ("module_attribute_constructor", "reexport_external_module")]
                          + [f"test_annotated_positive_binding[{fp}-reexport_external_module]" for fp in ("False", "True")])
        assert ids == expected, res.stdout[-5000:]
        assert "6 failed, 52 passed" in res.stdout, res.stdout[-400:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "binding-head").glob("*.json")}
        for case in ("module_if_binding", "module_for_binding", "module_match_binding", "module_exec_binding", "module_globals_binding", "module_delete_binding"):
            for fp in ("True", "False"):
                r = recs[f"{case}-{fp}"]
                assert not r["lowered"] and r["declines"][0].startswith("kernel_allocation_callee_unresolved:k: module_binding_"), (case, r["declines"])
                assert r["public"].get("error") == r["slow"].get("error") == "RuntimeError", (case, r["public"], r["slow"])
        for case in ("nested_parameter_only", "lambda_parameter_only", "nested_assignment_only", "class_member_only", "reexport_in_allowed_namespace",
                     "annotated_nested_parameter_only", "annotated_lambda_parameter_only", "annotated_nested_assignment_only", "annotated_class_member_only"):
            for fp in ("True", "False"):
                r = recs[f"{case}-{fp}"]
                assert r["lowered"] and not r["declines"] and not r["fallback"], (case, r)
                assert r["raw"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["public"]["value"]["out"]["data"], (case, r["raw"])
        for fp in ("True", "False"):
            r = recs[f"reexport_external_module-{fp}"]
            assert r["declines"][0].startswith("kernel_allocation_callee_unresolved:k: import_disallowed_prefix:review4_tensor_reexport"), r["declines"]
            assert r["public"]["value"]["out"]["data"] == [[2.0, 2.0]]
            r = recs[f"subclass_in_allowed_namespace-{fp}"]
            assert "import_identity_mismatch:solar.einsum_lib._review4_tensor_subclass.Tensor" in r["declines"][0] and r["public"]["value"]["out"]["data"] == [[7.0, 7.0]], r


# ---------------------------------------------------------------------------------------------------------------------
# an earlier fix round 4 (the review on an earlier revision): ANNOTATION-BINDING-9, INDIRECT-MODULE-WRITE-10, COMPREHENSION-SCOPE-11, LAMBDA-SCOPE-12
# ---------------------------------------------------------------------------------------------------------------------
_WRITER = 'def change():\n    global Tensor\n    Tensor = int\n'


class TestRound4AnnotationBinding:
    """annotation-binding-9: every parameter annotation of a nested def is evaluated in the ENCLOSING scope at
    definition time (Python 3.10), like its defaults, decorators and return annotation; a walrus there rebinds the
    enclosing name.  Plain annotations (loads) and a ``from __future__ import annotations`` module lower."""

    @pytest.mark.parametrize("prefix", [
        'def unrelated(x: (Tensor := int)):\n    pass\n',
        'def unrelated(x: (Tensor := int), /):\n    pass\n',
        'def unrelated(*, x: (Tensor := int)):\n    pass\n',
        'def unrelated(*x: (Tensor := int)):\n    pass\n',
        'def unrelated(**x: (Tensor := int)):\n    pass\n',
        'async def unrelated(x: (Tensor := int)):\n    pass\n',
        'def unrelated() -> (Tensor := int):\n    pass\n',
        'def unrelated(x=(Tensor := int)):\n    pass\n',
    ], ids=["positional", "positional-only", "keyword-only", "vararg", "kwarg", "async-def", "return-annotation", "default"])
    def test_module_annotation_walrus_declines(self, prefix):
        slow, runs = _assert_typed_decline(_unannotated(prefix=prefix), "kernel_allocation_callee_unresolved", "module_binding_rebound: the module binds 'Tensor' by a walrus at line 3")
        assert slow == {"error": "RuntimeError"}, slow          # the interpreter runs the rebound module: Tensor is int

    @pytest.mark.parametrize("body", [
        '    def helper(x: (Tensor := int)):\n        return 2\n    unused = helper(3)\n',
        '    def helper(*, x: (Tensor := int)):\n        return 2\n    unused = helper(x=3)\n',
        '    def helper() -> (Tensor := int):\n        return 2\n    unused = helper()\n',
    ], ids=["nested-parameter-annotation", "nested-kwonly-annotation", "nested-return-annotation"])
    def test_kernel_nested_def_annotation_walrus_declines(self, body):
        _assert_typed_decline(_unannotated(body), "kernel_allocation_callee_unresolved", "kernel_binding_rebound: the kernel binds 'Tensor' by a walrus at line 5")

    @pytest.mark.parametrize("header,prefix", [
        (UH, 'def unrelated(x: Tensor, *a: Tensor, k: Tensor = None, **kw: Tensor) -> Tensor:\n    pass\n'),
        ('from __future__ import annotations\n' + UH, 'def unrelated(x: Tensor) -> Tensor:\n    pass\n'),
    ], ids=["plain-annotations-load", "future-annotations"])
    def test_annotations_that_only_read_the_name_lower(self, header, prefix):
        _lowers_to(_unannotated(prefix=prefix, header=header), [[2.0, 2.0]])

    def test_own_expressions_yield_every_parameter_annotation(self):
        import ast as _ast
        from solar.einsum_lib import numba_accel as na
        fn = _ast.parse("@d\ndef f(a: A, /, b: B = db, *c: C, k: K = dk, **e: E) -> R: pass").body[0]
        got = sorted(_ast.unparse(e) for e in na._own_expressions(fn))
        assert got == sorted(["d", "db", "dk", "A", "B", "C", "K", "E", "R"]), got


class TestRound4ExecutedGlobalWrites:
    """indirect-module-write-10: a nested scope's body is not the module scope -- unless the module EXECUTES it.
    A class body runs at definition (``global Tensor; Tensor = int`` there rebinds the module name -> rebound); a
    module-scope call / decorator application whose callee resolves to a module-defined writer, or whose callee cannot
    be resolved while the module defines a writer -> uncertain.  Uncalled writers, called non-writers, calls to
    imported / builtin names (and their attributes: round 5) and unresolvable calls in writer-free modules keep
    lowering.  Round 5 replaced the per-form scan by the fixpoint effect analysis (:class:`_ModuleEffects`): the
    decline codes and sub-codes are unchanged, the detail wording changed (needles updated here, nothing weakened)."""

    @pytest.mark.parametrize("prefix,sub,needle", [
        ('class Change:\n    global Tensor\n    Tensor = int\n', "module_binding_rebound", "the class body at line 3 executes at definition and declares 'Tensor' global and binds it: the module assigns the name 'Tensor' at line 5"),
        ('class Change:\n    global Tensor\n    del Tensor\n', "module_binding_deleted", "the class body at line 3 executes at definition and declares 'Tensor' global and deletes it: the module deletes the name 'Tensor' at line 5"),
        (_WRITER + 'change()\n', "module_binding_uncertain", "the module calls 'change' at line 6, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        (_WRITER + 'alias = change\nalias()\n', "module_binding_uncertain", "the module calls 'alias' at line 7, which resolves to the function 'change' defined at line 3"),
        (_WRITER + 'print(change())\n', "module_binding_uncertain", "the module calls 'change' at line 6"),
        (_WRITER + 'g = (change() for _ in range(1))\n', "module_binding_uncertain", "the module calls 'change' at line 6"),
        ('def deco(f):\n    global Tensor\n    Tensor = int\n    return f\n@deco\ndef x():\n    pass\n', "module_binding_uncertain", "the module applies the decorator 'deco' at line 8, which resolves to the function 'deco' defined at line 3"),
        ('class Change:\n    def __init__(self):\n        global Tensor\n        Tensor = int\nChange()\n', "module_binding_uncertain", "the module calls 'Change' at line 7, which resolves to the class 'Change' defined at line 3, whose function '__init__' defined at line 4 declares 'Tensor' global and binds it"),
        ('def change():\n    globals()["Tensor"] = int\nchange()\n', "module_binding_uncertain", "the module calls 'change' at line 5, which resolves to the function 'change' defined at line 3, which calls globals(...) at line 4, which may rewrite the module namespace"),
        (_WRITER + 'getattr(np, "random").seed(0)\n', "module_binding_uncertain", "the module calls \"getattr(np, 'random').seed\" at line 6, which cannot be resolved statically (\"getattr(np, 'random').seed\" is reached through an expression the prescan cannot follow), while the function 'change' defined at line 3 declares 'Tensor' global and binds it"),
        (_WRITER + 'f = [change][0]\nf()\n', "module_binding_uncertain", "the module calls 'f' at line 7, which cannot be resolved statically ('f' is bound at line 6 by a statement the prescan cannot follow), while the function 'change'"),
    ], ids=["class-body-global-assign", "class-body-global-del", "called-writer", "aliased-call", "call-nested-in-expression", "generator-calls-writer",
            "decorator-writer", "class-constructor-writer", "writer-through-globals-call", "computed-callee-with-writer", "unknown-name-call-with-writer"])
    def test_executed_writes_decline(self, prefix, sub, needle):
        slow, runs = _assert_typed_decline(_unannotated(prefix=prefix), "kernel_allocation_callee_unresolved", sub + ": " + needle)

    @pytest.mark.parametrize("prefix", [
        _WRITER,                                                                   # the reviewer's module_uncalled_global_writer
        'class Change:\n    def go(self):\n        global Tensor\n        Tensor = int\n',
        'def harmless():\n    Tensor = int\n    return 1\nharmless()\n',               # the reviewer's module_function_local, called
        'COUNT = 0\ndef bump():\n    global COUNT\n    COUNT += 1\nbump()\n',
        'np.random.seed(0)\n',                                                        # attribute call on an import, no writer in the module
        'class Holder:\n    Tensor = int\n',                                           # class attribute, no global declaration
        _WRITER + 'clear_dims()\n',                                                   # imported callee next to an uncalled writer
        _WRITER + 'np.random.seed(0)\n',                                              # round 5: an attribute of an IMPORT is resolved safe even beside an uncalled writer (round 4 declined it as unresolvable)
    ], ids=["uncalled-writer", "uncalled-method-writer", "called-nonwriter-local", "called-nonwriter-other-global", "attribute-call-no-writer",
            "class-member", "imported-call-beside-uncalled-writer", "import-attribute-call-beside-uncalled-writer"])
    def test_unexecuted_or_harmless_calls_lower(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])


class TestRound4ScopeAwareExpressions:
    """COMPREHENSION-SCOPE-11 / LAMBDA-SCOPE-12: :func:`_scope_expression_nodes` visits what the enclosing scope
    evaluates -- lambda defaults, comprehension iterables / conditions / elements -- and never lambda bodies or
    comprehension targets."""

    @pytest.mark.parametrize("body", [
        '    unused = [Tensor for Tensor in range(3)]\n',
        '    unused = {Tensor: 1 for Tensor in range(3)}\n',
        '    unused = [lambda: (Tensor := int) for _ in range(1)]\n',
    ], ids=["list-comprehension-target", "dict-comprehension-target", "lambda-body-inside-comprehension"])
    def test_comprehension_locals_and_lambda_bodies_lower(self, body):
        _lowers_to(_unannotated(body), [[2.0, 2.0]])

    @pytest.mark.parametrize("body", [
        '    unused = {Tensor for Tensor in range(3)}\n',
        '    unused = (Tensor for Tensor in range(3))\n',
    ], ids=["set-comprehension-target", "generator-target"])
    def test_other_comprehension_targets_are_not_binding_declines(self, body):
        """Set comprehensions and generator expressions are not compiled by numba (SET_ADD / closure inlining: a
        compile-time fallback, outside this prescan); the prescan itself does not mistake their targets for a binding."""
        for use_fp64 in (True, False):
            seen = []
            assert numba_lower(_unannotated(body), use_fp64=use_fp64, on_decline=seen.append) is not None and not seen, seen

    def test_scope_expression_nodes_unit(self):
        import ast as _ast
        from solar.einsum_lib import numba_accel as na
        def walrus_targets(src):
            expr = _ast.parse(src).body[0].value
            return sorted(n.target.id for n in na._scope_expression_nodes(expr) if isinstance(n, _ast.NamedExpr))
        assert walrus_targets("lambda: (A := 1)") == []                          # lambda body: its own scope
        assert walrus_targets("lambda x=(B := 1): (A := 2)") == ["B"]            # lambda default: enclosing scope
        assert walrus_targets("[(C := i) for i in range(3) if (D := i)]") == ["C", "D"]      # walrus in a comprehension: enclosing scope
        assert walrus_targets("[i for i in (E := range(3))]") == ["E"]           # the iterable is evaluated in the enclosing scope
        expr = _ast.parse("[Tensor for Tensor in range(3)]").body[0].value
        assert not any(isinstance(n, _ast.Name) and n.id == "Tensor" and isinstance(n.ctx, _ast.Store) for n in na._scope_expression_nodes(expr))   # targets never yielded (the element's LOAD is)


class TestReview4Round3ScopeVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's ``test_round3_scope.py`` (sha256 5f3907b6…, 24 cases x 2 carriers; it imports its round-2
        ``test_binding_edges.py``) and ``test_round3_scope_positive_v2.py`` (sha256 b96f1d48…, 2 annotated positives x 2)
        byte-for-byte: 52 pass, 0 fail on this head (an earlier revision: 34 pass / 14 fail and 0 / 4).  The records are read back:
        the three annotation walruses, the called global writer and the class global writer decline with the
        interpreter's RuntimeError on the public path; the uncalled writer, the function-local write, the comprehension
        target and the lambda-body walrus lower to exactly 2.0 on both carriers without fallback."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_round3_scope.py", "test_round3_scope_positive_v2.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round3_scope.py", "test_round3_scope_positive_v2.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        assert "52 passed" in res.stdout and "failed" not in res.stdout.splitlines()[-1], res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "scope-head").glob("*.json")}
        for case, sub in (("module_argument_annotation_walrus", "module_binding_rebound"), ("module_kwonly_annotation_walrus", "module_binding_rebound"),
                          ("module_vararg_annotation_walrus", "module_binding_rebound"), ("module_called_global_writer", "module_binding_uncertain"),
                          ("module_class_global_writer", "module_binding_rebound")):
            for fp in ("True", "False"):
                r = recs[f"{case}-{fp}"]
                assert not r["lowered"] and r["declines"][0].startswith(f"kernel_allocation_callee_unresolved:k: {sub}"), (case, r["declines"])
                assert r["public"].get("error") == r["slow"].get("error") == "RuntimeError", (case, r["public"], r["slow"])
        for case in ("module_uncalled_global_writer", "module_function_local", "comprehension_local", "lambda_body_local_walrus",
                     "annotated_comprehension_local", "annotated_lambda_body_local_walrus"):
            for fp in ("True", "False"):
                r = recs[f"{case}-{fp}"]
                assert r["lowered"] and not r["declines"] and not r["fallback"], (case, r["declines"], r["fallback"])
                assert r["raw"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["public"]["value"]["out"]["data"], (case, r["raw"])


class TestRound5TransitiveEffects:
    """transitive-writer-13: writer-ness propagates through call edges to a fixpoint (cycles included); a callee
    that cannot be resolved to a non-writer is an unresolved effect, declined whenever the module holds any capability
    to reach its namespace (a writer, a namespace builtin spelled as a string, ``__builtins__``, ``builtins`` /
    ``ctypes`` / ``gc`` / ``inspect`` imported); an unknown callee always declines; the harness's ``run_model`` call
    is executed code too.  The uncalled wrapper, the called nonlocal-only closure and called functions with no
    transitive write lower."""

    @pytest.mark.parametrize("prefix,run,needle", [
        (_WRITER + 'def invoke():\n    change()\ninvoke()\n', URUN, "the module calls 'invoke' at line 8, which resolves to the function 'invoke' defined at line 6, which calls 'change' at line 7, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        ('def first(n):\n    if n:\n        second(n - 1)\ndef second(n):\n    global Tensor\n    Tensor = int\n    if n:\n        first(n - 1)\nfirst(1)\n', URUN, "the module calls 'first' at line 11, which resolves to the function 'first' defined at line 3, which calls 'second' at line 5, which resolves to the function 'second' defined at line 6, which declares 'Tensor' global and binds it"),
        (_WRITER + 'def apply(fn):\n    fn()\napply(change)\n', URUN, "which calls 'fn' at line 7, which cannot be resolved statically ('fn' is a parameter of the function 'apply' defined at line 6), while the function 'change' defined at line 3"),
        (_WRITER + 'list(map(change, []))\n', URUN, "the module passes 'change' to 'list' at line 6, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        (_WRITER + 'def setup():\n    global f\n    f = change\nf = print\nf()\n', URUN, "the module calls 'f' at line 10, which cannot be resolved statically ('f' is bound at line 6 by a statement the prescan cannot follow), while the function 'change'"),
        ('import sys\nf = setattr\nf(sys.modules[__name__], "Tensor", int)\n', URUN, "module_binding_uncertain: the module references 'setattr' at line 4, which may rewrite the namespace"),
        ('import builtins\nbuiltins.setattr(builtins, "x", 1)\n', URUN, "module_binding_uncertain: the module references 'builtins.setattr' at line 4, which may rewrite the namespace"),
        ('import sys\nname = "setattr"\nh = [sys.modules][0]\nh.get(name)\n', URUN, "the module calls 'h.get' at line 6, which cannot be resolved statically ('h' is bound at line 5 by a statement the prescan cannot follow), while the module spells the string 'setattr' at line 4"),
        ('mystery()\n', URUN, "the module calls 'mystery' at line 3, which names 'mystery', which is bound nowhere in the module and is neither a builtin nor a harness-provided name"),
        (_WRITER, '\ndef run_model(inputs):\n    change()\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n', "the harness calls 'run_model' (defined at line 11), which calls 'change' at line 12, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
    ], ids=["transitive-call", "mutual-recursion", "writer-passed-to-module-callee", "writer-passed-to-builtin", "callee-rebound-under-global",
            "alias-of-setattr", "builtins-attribute", "string-capability-with-computed-callee", "unknown-callee", "run_model-calls-writer"])
    def test_transitive_and_unresolved_effects_decline(self, prefix, run, needle):
        _assert_typed_decline(_unannotated(prefix=prefix, run=run), "kernel_allocation_callee_unresolved", needle)

    @pytest.mark.parametrize("prefix", [
        _WRITER + 'def unused():\n    change()\n',                                                   # the reviewer's uncalled wrapper
        'def outer():\n    Tensor = int\n    def inner():\n        nonlocal Tensor\n        Tensor = float\n    inner()\nouter()\n',   # called nonlocal-only closure
        _WRITER + 'def a():\n    return b()\ndef b():\n    return 1\na()\n',                        # called chain with no transitive write, beside an uncalled writer
        'def apply(fn):\n    return fn()\napply(print)\n',                                            # unresolved parameter call in a module WITHOUT any capability
        'h = [np][0]\nh.zeros(1)\n',                                                                  # computed callee (name bound through a container), no capability
        _WRITER + 'np.random.seed(0)\nclear_dims()\nprint(len([]))\n',                                 # imports, harness names and builtins beside an uncalled writer
    ], ids=["uncalled-wrapper", "called-nonlocal-closure", "called-nonwriter-chain", "parameter-call-no-capability", "computed-callee-no-capability", "safe-callees-beside-writer"])
    def test_unreached_or_safe_effects_lower(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])

    def test_harness_names_match_the_interpreter_namespace(self):
        from solar.einsum_lib import numba_accel as na, verify as v
        assert na._HARNESS_NAMES == set(v._MATH_BUILTINS) | {"math", "np"}, sorted(na._HARNESS_NAMES ^ (set(v._MATH_BUILTINS) | {"math", "np"}))


class TestRound5ClassExecution:
    """class-execution-14: a class body is EXECUTED module-scope code at definition (recursively for nested
    classes): namespace calls in it are uncertain, its calls go through the effect analysis, a ``global`` + store in
    it or in a nested class body rebinds; a class's methods run only when the class is instantiated / called / used
    as a base.  Ordinary class attributes, class-local annotations and class-body comprehensions stay class-local."""

    @pytest.mark.parametrize("prefix,sub,needle", [
        ('class Holder:\n    globals()["Tensor"] = int\n', "module_binding_uncertain", "the class body at line 3 executes at definition and calls globals(...) at line 4, which may rewrite the module namespace"),
        ('class Outer:\n    class Inner:\n        global Tensor\n        Tensor = int\n', "module_binding_rebound", "the class body at line 3 executes at definition and contains class 'Inner' defined at line 4, whose body executes at definition and declares 'Tensor' global and binds it: the module assigns the name 'Tensor' at line 6"),
        ('class Holder:\n    def m():\n        global Tensor\n        Tensor = int\n    m()\n', "module_binding_uncertain", "the class body at line 3 executes at definition and calls 'm' at line 7, which resolves to the function 'm' defined at line 4, which declares 'Tensor' global and binds it"),
        ('class Holder:\n    @staticmethod\n    def m():\n        global Tensor\n        Tensor = int\nHolder.m()\n', "module_binding_uncertain", "the module calls 'Holder.m' at line 8, which resolves to the class 'Holder' defined at line 3, whose function 'm' defined at line 5 declares 'Tensor' global and binds it"),
        ('class Base:\n    def __init_subclass__(cls):\n        global Tensor\n        Tensor = int\nclass A(Base):\n    pass\n', "module_binding_uncertain", "the module derives the class 'A' from 'Base' at line 7, which resolves to the class 'Base' defined at line 3, whose function '__init_subclass__' defined at line 4 declares 'Tensor' global and binds it"),
        ('class Holder:\n    def __init__(self):\n        global Tensor\n        Tensor = int\nh = Holder()\n', "module_binding_uncertain", "the module calls 'Holder' at line 7, which resolves to the class 'Holder' defined at line 3, whose function '__init__' defined at line 4 declares 'Tensor' global and binds it"),
        ('class Holder:\n    unused = (lambda: globals().__setitem__("Tensor", int))()\n', "module_binding_uncertain", "the class body at line 3 executes at definition and calls \"lambda : globals().__setitem__('Tensor', int)\" at line 4, which resolves to the lambda at line 4, which calls globals(...) at line 4"),
    ], ids=["class-body-namespace-call", "nested-class-global-store", "class-body-calls-writer-method", "static-writer-method-called", "base-with-writer-init-subclass",
            "constructor-with-writer-init", "class-body-invoked-lambda"])
    def test_executed_class_bodies_decline(self, prefix, sub, needle):
        _assert_typed_decline(_unannotated(prefix=prefix), "kernel_allocation_callee_unresolved", sub + ": " + needle)

    @pytest.mark.parametrize("prefix", [
        'class Holder:\n    Tensor = int\n',                                                 # class attribute
        'class Holder:\n    Tensor: int = 1\n',                                              # class-local annotation + attribute
        'class Holder:\n    unused = [Tensor for Tensor in range(3)]\n',                     # class-body comprehension target
        'class Outer:\n    class Inner:\n        Tensor = int\n',                            # nested class attribute, no global
        'class Holder:\n    def m(self):\n        global Tensor\n        Tensor = int\n',   # writer method, class never instantiated / called
        'class M(type):\n    pass\nclass A(metaclass=M):\n    pass\n',                      # class keywords in a module without any capability
        'class Holder:\n    def m(self):\n        return 1\nHolder().m()\n',                 # instantiated class whose methods do not write
    ], ids=["class-attribute", "class-local-annotation", "class-body-comprehension", "nested-class-attribute", "uncalled-writer-method", "metaclass-no-capability",
            "instantiated-nonwriter"])
    def test_class_locals_and_unexecuted_methods_lower(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])


class TestRound5InvokedLambda:
    """invoked-lambda-15 (round-4 regression): an INVOKED lambda body -- immediately invoked, bound to a name
    that is later called, called through another call, or passed to a call that may invoke it -- is executed code and
    goes through the effect analysis; uncalled lambda bodies stay skipped (LAMBDA-SCOPE-12 positives)."""

    @pytest.mark.parametrize("prefix,needle", [
        ('(lambda: globals().__setitem__("Tensor", int))()\n', "the module calls \"lambda : globals().__setitem__('Tensor', int)\" at line 3, which resolves to the lambda at line 3, which calls globals(...) at line 3, which may rewrite the module namespace"),
        ('f = lambda: globals().__setitem__("Tensor", int)\nf()\n', "the module calls 'f' at line 4, which resolves to the lambda at line 3, which calls globals(...) at line 3"),
        ('f = lambda: (lambda: globals().__setitem__("Tensor", int))\nf()()\n', "the module calls 'f()' at line 4, which cannot be resolved statically ('f()' is the result of an expression the prescan cannot follow), while the lambda at line 3 calls globals(...) at line 3"),
        ('list(map(lambda x: globals().__setitem__("Tensor", int), [1]))\n', "the module passes \"lambda x: globals().__setitem__('Tensor', int)\" to 'list' at line 3, which resolves to the lambda at line 3, which calls globals(...) at line 3"),
        (_WRITER + '(lambda: change())()\n', "the module calls 'lambda : change()' at line 6, which resolves to the lambda at line 6, which calls 'change' at line 6, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        ('def unused(x=(lambda: globals().__setitem__("Tensor", int))()):\n    pass\n', "the module calls \"lambda : globals().__setitem__('Tensor', int)\" at line 3, which resolves to the lambda at line 3, which calls globals(...) at line 3"),
    ], ids=["immediately-invoked", "named-then-called", "called-through-call", "passed-to-builtin", "invoked-wrapper-of-writer", "invoked-in-default"])
    def test_invoked_lambda_bodies_decline(self, prefix, needle):
        _assert_typed_decline(_unannotated(prefix=prefix), "kernel_allocation_callee_unresolved", "module_binding_uncertain: " + needle)

    @pytest.mark.parametrize("prefix", [
        'unused = lambda: globals().__setitem__("Tensor", int)\n',                 # bound, never called (the reviewer's lambda_namespace_uninvoked)
        _WRITER + 'unused = lambda: change()\n',                                    # the reviewer's lambda_calls_writer_uninvoked
        'def unused(x: (lambda: globals().__setitem__("Tensor", int))):\n    pass\n',   # an uncalled lambda in an eager annotation
        'unused = sorted([3, 1], key=lambda t: -t)\n',                               # a lambda passed to a call, with no effect of its own
        'unused = lambda: (lambda: (Tensor := int))\n',                              # nested uncalled; the walrus is lambda-local anyway
    ], ids=["bound-uncalled", "bound-uncalled-wrapper", "annotation-uncalled", "passed-harmless", "nested-uncalled"])
    def test_uninvoked_lambda_bodies_lower(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])


FUTURE = 'from __future__ import annotations\n'
SIDE_RUN = '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": k(inputs["a"]).data, "side": np.array(COUNT)}\n'
MARK = 'COUNT = 0\ndef mark():\n    global COUNT\n    COUNT += 1\n    return int\ndef unused(x: mark()):\n    pass\n'


class TestRound5FutureAnnotations:
    """future-annotations-16: under ``from __future__ import annotations`` no annotation is evaluated -- the
    prescan does not visit them and the lowered module keeps the future import FIRST, so a side effect or a namespace
    call in an annotation is neither counted nor executed (the reviewer's control); without it annotations stay eager
    (round-4 behaviour).  The ORIGINAL source is compile-validated before any rewrite: a source ``ast.parse`` admits
    but Python's compiler rejects (a misplaced future import, a walrus in a postponed annotation or in a comprehension
    iterable) declines ``kernel_source_invalid:<kernel>`` with the interpreter's own error on the public path."""

    def _side(self, code):
        x = np.ones((1, 2))
        slow = execute_translated(code, {}, all_inputs={"a": x}, fast=False)[0]
        out = []
        for use_fp64 in (True, False):
            seen, fell = [], []
            low = numba_lower(code, use_fp64=use_fp64, on_decline=seen.append)
            raw = execute_fast(code, {"a": x}, use_fp64=use_fp64, on_fallback=fell.append)
            public = execute_translated(code, {}, all_inputs={"a": x}, fast=True, use_fp64=use_fp64)[0]
            out.append((use_fp64, low, seen, fell, int(raw["side"]), int(public["side"]), np.asarray(public["output"])))
        return int(slow["side"]), out

    def test_postponed_annotation_side_effect_is_not_executed(self):
        slow_side, runs = self._side(_unannotated(prefix=MARK, header=FUTURE + UH, run=SIDE_RUN))
        assert slow_side == 0
        for use_fp64, low, seen, fell, raw_side, public_side, public in runs:
            assert low is not None and not seen and not fell, (use_fp64, seen, fell)
            assert low.splitlines()[0] == "from __future__ import annotations", low.splitlines()[:3]
            assert raw_side == public_side == 0, (use_fp64, raw_side, public_side)
            np.testing.assert_array_equal(public, [[2.0, 2.0]])

    def test_eager_annotation_side_effect_is_executed_on_both_paths(self):
        slow_side, runs = self._side(_unannotated(prefix=MARK, run=SIDE_RUN))
        assert slow_side == 1
        for use_fp64, low, seen, fell, raw_side, public_side, public in runs:
            assert low is not None and not seen and not fell and not low.startswith("from __future__"), (use_fp64, seen, fell)
            assert raw_side == public_side == 1, (use_fp64, raw_side, public_side)

    @pytest.mark.parametrize("prefix", [
        'def unused(x: globals().__setitem__("Tensor", int)):\n    pass\n',          # the reviewer's control: legal, postponed, never executed
        'def unused(x: Tensor) -> Tensor:\n    pass\n',
        'x: globals().__setitem__("Tensor", int) = 1\n',                               # a postponed module-level annotated assignment
        'def unused(x: (lambda: (Tensor := int))):\n    pass\n',
    ], ids=["postponed-namespace-call", "postponed-plain", "postponed-annassign", "postponed-lambda"])
    def test_postponed_annotations_lower(self, prefix):
        _lowers_to(_unannotated(prefix=prefix, header=FUTURE + UH), [[2.0, 2.0]])

    def test_eager_namespace_call_annotation_still_declines(self):
        _assert_typed_decline(_unannotated(prefix='def unused(x: globals().__setitem__("Tensor", int)):\n    pass\n'), "kernel_allocation_callee_unresolved",
                              "module_binding_uncertain: the module calls globals(...) at line 3")

    @pytest.mark.parametrize("header,prefix,body,needle", [
        (FUTURE + UH, 'def unused(x: (Tensor := int)):\n    pass\n', "", "SyntaxError: 'named expression' can not be used within an annotation (line 4)"),
        ('x = 1\n' + FUTURE + UH, "", "", "SyntaxError: from __future__ imports must occur at the beginning of the file (line 2)"),
        (UH, "", '    unused = [x for x in (Tensor := [1])]\n', "SyntaxError: assignment expression cannot be used in a comprehension iterable expression (line 5)"),
    ], ids=["walrus-in-postponed-annotation", "misplaced-future-import", "walrus-in-comprehension-iterable"])
    def test_sources_the_compiler_rejects_decline_typed(self, header, prefix, body, needle):
        code = _unannotated(body, prefix=prefix, header=header)
        import ast as _ast
        _ast.parse(code)                                                              # parses ...
        with pytest.raises(SyntaxError):
            compile(code, "<probe>", "exec", dont_inherit=True)                       # ... but does not compile
        slow, runs = _assert_typed_decline(code, "kernel_source_invalid", needle)
        assert slow == {"error": "RuntimeError"}, slow

    def test_postponed_detection_and_own_expressions(self):
        import ast as _ast
        from solar.einsum_lib import numba_accel as na
        assert na._postponed_annotations(_ast.parse('"""doc"""\nfrom __future__ import division, annotations\nx = 1\n'))
        assert na._postponed_annotations(_ast.parse('from __future__ import division\nfrom __future__ import annotations\n'))
        assert not na._postponed_annotations(_ast.parse('from __future__ import division\nx = 1\n'))
        assert not na._postponed_annotations(_ast.parse('x = 1\nfrom __future__ import annotations\n'))      # misplaced: not leading (and the compiler rejects it)
        fn = _ast.parse("@d\ndef f(a: A, /, b: B = db, *c: C, k: K = dk, **e: E) -> R: pass").body[0]
        assert sorted(_ast.unparse(e) for e in na._own_expressions(fn, postponed=True)) == ["d", "db", "dk"]
        ann = _ast.parse("x: A = v").body[0]
        assert sorted(_ast.unparse(e) for e in na._own_expressions(ann, postponed=True)) == ["v", "x"]
        assert sorted(_ast.unparse(e) for e in na._own_expressions(ann)) == ["A", "v", "x"]


class TestReview4Round4ScopeVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's ``test_round4_scope.py`` (sha256 7879194d…, 43 cases x 2 carriers; it imports its round-3 and round-2
        files) and ``test_round4_generator_control.py`` (sha256 fd704bf5…, 1 case x 2) byte-for-byte: 86 pass and exactly
        the two ruled ``generator_local_consumed`` deviations fail (``list(generator)`` is outside the callee allowlist:
        a typed ``list`` callee decline with the interpreter value preserved -- kept by ruling).  an earlier revision: 64 pass / 22
        fail (+2 pass).  The records are read back: every ``decline`` case declines with a ``kernel_`` reason and the
        interpreter's error / value on the public path, every ``lower`` case lowers to exactly 2.0 on both carriers
        without fallback, the two ``invalid`` cases and the ``observe`` future control behave as the interpreter does."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_round3_scope.py", "test_round4_scope.py", "test_round4_generator_control.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round4_scope.py", "test_round4_generator_control.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        assert "2 failed, 86 passed" in res.stdout.splitlines()[-1], res.stdout[-3000:]
        failed = sorted(l.split("::")[-1].split(" ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED "))
        assert failed == ["test_round4_scope[False-generator_local_consumed]", "test_round4_scope[True-generator_local_consumed]"], failed
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round4-head").glob("*.json")}
        assert len(recs) == 88, len(recs)
        for name, r in recs.items():
            exp = r["expected"]
            if exp == "decline":
                assert not r["lowered"] and r["declines"] and r["declines"][0].startswith("kernel_"), (name, r["declines"])
            elif exp == "lower":
                assert r["lowered"] and not r["declines"] and not r["fallback"], (name, r["declines"], r["fallback"])
                assert r["raw"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["public"]["value"]["out"]["data"], (name, r["raw"])
            elif exp == "invalid":
                assert r["compile_error"]["type"] == "SyntaxError" and not r["lowered"] and r["declines"][0].startswith("kernel_source_invalid:k: SyntaxError"), (name, r)
            if "error" in r["slow"]:
                assert r["public"].get("error") == r["slow"]["error"], (name, r["public"], r["slow"])
            else:
                assert r["public"].get("value") == r["slow"]["value"] or name.startswith("generator_local_consumed") or not r["fp64"], name
        for fp in ("True", "False"):
            r = recs[f"future_annotation_side_effect-{fp}"]
            assert r["lowered"] and not r["fallback"] and r["slow"]["value"]["side"]["data"] == 0 == r["public"]["value"]["side"]["data"] == r["raw"]["value"]["side"]["data"], r["public"]
            r = recs[f"eager_annotation_side_effect-{fp}"]
            assert r["slow"]["value"]["side"]["data"] == 1 == r["public"]["value"]["side"]["data"], r["public"]
            r = recs[f"generator_local_consumed-{fp}"]
            assert not r["lowered"] and r["declines"][0].startswith("kernel_allocation_callee_unresolved:k: the call 'list' at line 5") and r["public"]["value"] == r["slow"]["value"], r


class TestRound5ModuleAnalysisOnce:
    """Round-5 census follow-up (performance, answers unchanged): the module-wide effect analysis ``_ModuleEffects``
    depends on the tree alone, so one pre-scan builds it ONCE for all kernels of a module -- not once per kernel --
    and not at all when the kernel's own scope already declines.  At an earlier revision a 38-kernel corpus module lowered in
    1.37 s against 0.36 s on an earlier revision (a 4-5x slowdown that pushed the 69,253-file census past its hour)."""

    @staticmethod
    def _lower_counting(monkeypatch, code):
        from solar.einsum_lib import numba_accel as na
        real, built = na._ModuleEffects, []

        class Counting(real):
            def __init__(self, tree, postponed):
                built.append(postponed)
                super().__init__(tree, postponed)

        monkeypatch.setattr(na, "_ModuleEffects", Counting)
        seen = []
        low = na.numba_lower(code, boundscheck=True, use_fp64=True, on_decline=seen.append)
        return len(built), low, seen

    @staticmethod
    def _many_kernels(n, prefix="", first_body=""):
        ks = "".join(f"@kernel\ndef k{i}(A):\n" + (first_body if i == 0 else "") + UY for i in range(n))
        return UH + prefix + ks + '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": k0(inputs["a"]).data}\n'

    def test_twelve_clean_kernels_build_the_analysis_once_and_lower(self, monkeypatch):
        built, low, seen = self._lower_counting(monkeypatch, self._many_kernels(12))
        assert built == 1 and low is not None and not seen, (built, seen)

    def test_module_writer_is_analysed_once_and_still_declines_the_first_kernel(self, monkeypatch):
        built, low, seen = self._lower_counting(monkeypatch, self._many_kernels(3, prefix=_WRITER + "change()\n"))
        assert built == 1 and low is None and len(seen) == 1, (built, seen)
        assert seen[0].startswith("kernel_allocation_callee_unresolved:k0: module_binding_uncertain: the module calls 'change' at line 6, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"), seen[0]

    def test_kernel_scope_decline_comes_first_and_skips_the_module_analysis(self, monkeypatch):
        built, low, seen = self._lower_counting(monkeypatch, self._many_kernels(3, first_body="    Tensor = int\n"))
        assert built == 0 and low is None and len(seen) == 1, (built, seen)
        assert seen[0].startswith("kernel_allocation_callee_unresolved:k0: kernel_binding_"), seen[0]

    def test_direct_two_argument_call_still_works_without_a_memo(self):
        from solar.einsum_lib import numba_accel as na
        import ast as _ast
        tree = _ast.parse(self._many_kernels(2))
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef) and node.name.startswith("k"):
                assert na._reject_unresolved_tensor_binding(node, tree) is None


class TestRound5ComprehensionVariables:
    """Round-5 census follow-up 2 (precision, soundness kept): a comprehension / generator-expression variable is a BOUND
    name whose value the prescan cannot follow -- an unresolved callee root, declined only with a capability -- not a name
    "bound nowhere in the module" (which always declines).  At an earlier revision the census newly declined the BigBird KB source
    (29 copies) on ``max(int(k[1:k.index("_")]) for k in inputs if k.startswith("L") ...)`` inside ``run_model``."""

    def test_generator_variable_as_attribute_callee_root_lowers(self):
        run = '\ndef run_model(inputs):\n    clear_dims()\n    n = sum(1 for key in inputs if key.startswith("a"))\n    return {"output": k(inputs["a"]).data}\n'
        _lowers_to(_unannotated(run=run), [[2.0, 2.0]])

    def test_comprehension_variable_called_without_capability_lowers(self):
        run = '\ndef run_model(inputs):\n    clear_dims()\n    [fn() for fn in (print,)]\n    return {"output": k(inputs["a"]).data}\n'
        _lowers_to(_unannotated(run=run), [[2.0, 2.0]])

    def test_comprehension_variable_called_beside_a_writer_declines_as_unresolved(self):
        run = '\ndef run_model(inputs):\n    clear_dims()\n    [fn() for fn in (print,)]\n    return {"output": k(inputs["a"]).data}\n'
        _assert_typed_decline(_unannotated(prefix=_WRITER, run=run), "kernel_allocation_callee_unresolved",
                              "module_binding_uncertain: the harness calls 'run_model' (defined at line 11), which calls 'fn' at line 13, which cannot be "
                              "resolved statically ('fn' is bound at line 13 by a statement the prescan cannot follow), while the function 'change' "
                              "defined at line 3 declares 'Tensor' global and binds it", expect_value=[[2.0, 2.0]])

    def test_comprehension_variable_shadowing_the_kernel_name_is_a_typed_refusal_since_round_10(self):
        # round 5: no Tensor-rebinding capability, so the allocation path lowered this; round 10 (kernel-call-binding-27): a
        # comprehension target spelled like the kernel is a NEARER BINDING of the callee in run_model, so the '.data' read is the
        # typed refusal kernel_name_shadowed (an over-refusal by the rule's letter -- the generator variable does not leak in
        # Python 3 -- reported, not pre-empted); the public value stays the interpreter's
        run = '\ndef run_model(inputs):\n    clear_dims()\n    n = sum(1 for k in inputs if k.startswith("a"))\n    return {"output": k(inputs["a"]).data}\n'
        _assert_typed_decline(_unannotated(run=run), "kernel_function_statement_unresolved",
                              "kernel_name_shadowed: a '.data' attribute at line", expect_value=[[2.0, 2.0]])
        reasons = []
        assert numba_lower(_unannotated(run=run), use_fp64=True, boundscheck=True, on_decline=reasons.append) is None
        assert "the @kernel name 'k' is bound in the function 'run_model' (a comprehension target at line" in reasons[0], reasons[0]

    def test_module_level_comprehension_variable_lowers(self):
        _lowers_to(_unannotated(prefix="squares = [v * v for v in range(3)]\n"), [[2.0, 2.0]])

    def test_module_level_comprehension_calling_its_variable_beside_a_writer_declines(self):
        _assert_typed_decline(_unannotated(prefix=_WRITER + "r = [fn() for fn in (print,)]\n"), "kernel_allocation_callee_unresolved",
                              "module_binding_uncertain: the module calls 'fn' at line 6, which cannot be resolved statically ('fn' is bound at line 6 "
                              "by a statement the prescan cannot follow), while the function 'change' defined at line 3 declares 'Tensor' global and binds it",
                              expect_value=[[2.0, 2.0]])


_BADRUN = '\ndef runner(inputs):\n    global Tensor\n    Tensor = int\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n'
_SAFERUN = '\ndef runner(inputs):\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n'


class TestRound6HarnessBinding:
    """harness-binding-17: the harness calls whatever ``run_model`` is FINALLY bound to.  Every binding of the name is
    a possible entry (def, ``run_model = lambda``, alias chains, a rebinding after an innocent def); a binding the prescan
    cannot follow (an attribute such as ``Runner.run``, a call result, a tuple target, an import, a ``global`` store in
    another scope, a deletion) is the typed refusal ``harness_binding_unresolved`` whether or not the module holds a
    capability.  The safe alias and the safe lambda entry keep lowering (the lambda is rounded: see
    :class:`TestRound6EntryCallRounding`)."""

    @pytest.mark.parametrize("run,needle", [
        (_BADRUN + "run_model = runner\n", "module_binding_uncertain: the harness calls 'run_model' (bound at line 13 to 'runner'), which resolves to the function 'runner' defined at line 8, which declares 'Tensor' global and binds it"),
        (_BADRUN + "second = runner\nrun_model = second\n", "module_binding_uncertain: the harness calls 'run_model' (bound at line 14 to 'second'), which resolves to the function 'runner' defined at line 8, which declares 'Tensor' global and binds it"),
        (URUN + _BADRUN + "run_model = runner\n", "module_binding_uncertain: the harness calls 'run_model' (bound at line 17 to 'runner'), which resolves to the function 'runner' defined at line 12, which declares 'Tensor' global and binds it"),
        ('\nrun_model = lambda inputs: (change(), clear_dims(), {"output": k(inputs["a"]).data})[-1]\n', "module_binding_uncertain: the harness calls 'run_model' (bound at line 11 to a lambda), which calls 'change' at line 11, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
    ], ids=["alias", "alias_chain", "rebound_after_innocent_def", "lambda_calling_writer"])
    def test_writer_behind_the_final_binding_declines(self, run, needle):
        prefix = _WRITER if "change()" in run else ""
        _assert_typed_decline(_unannotated(prefix=prefix, run=run), "kernel_allocation_callee_unresolved", needle)

    @pytest.mark.parametrize("prefix,run,needle", [
        ("", '\nclass Runner:\n    @staticmethod\n    def run(inputs):\n        global Tensor\n        Tensor = int\n        clear_dims()\n        return {"output": k(inputs["a"]).data}\nrun_model = Runner.run\n',
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 15), which cannot be resolved statically ('run_model' is bound at line 15 by a statement the prescan cannot follow)"),
        ("", _SAFERUN + "def make():\n    return runner\nrun_model = make()\n",
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 13), which cannot be resolved statically ('run_model' is bound at line 13 by a statement the prescan cannot follow)"),
        ("", _SAFERUN + "run_model, other = runner, 1\n",
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 11), which cannot be resolved statically ('run_model' is bound at line 11 by a statement the prescan cannot follow)"),
        ("", URUN + "def rebind():\n    global run_model\n    run_model = print\nrebind()\n",
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 11), which cannot be resolved statically ('run_model' is bound at line 11 by a 'global' store in the function 'rebind' defined at line 11)"),
        ("", _SAFERUN + "run_model = print\n",
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 11 to 'print'), which resolves to an imported, builtin or harness-provided name, not to module-defined code"),
        ("", _SAFERUN + "run_model = missing_name\n",
         "harness_binding_unresolved: the harness calls 'run_model' (bound at line 11 to 'missing_name'), which names 'missing_name', which is bound nowhere in the module"),
    ], ids=["static_method_attribute", "call_result", "tuple_target", "global_store_elsewhere", "builtin", "unbound_name"])
    def test_unestablished_final_binding_is_a_typed_refusal_without_any_capability(self, prefix, run, needle):
        code = _unannotated(prefix=prefix, run=run)
        seen = []
        assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=seen.append) is None
        assert seen and seen[0].startswith("kernel_allocation_callee_unresolved:k: " + needle), seen

    def test_safe_alias_and_plain_def_lower(self):
        _lowers_to(_unannotated(run=_SAFERUN + "run_model = runner\n"), [[2.0, 2.0]])
        _lowers_to(_unannotated(run=_SAFERUN + "second = runner\nrun_model = second\n"), [[2.0, 2.0]])

    def test_subcode_is_carried(self):
        from solar.einsum_lib import numba_accel as na
        import ast as _ast
        tree = _ast.parse(_unannotated(run=_SAFERUN + "def make():\n    return runner\nrun_model = make()\n"))
        node = next(n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "k")
        with pytest.raises(na.KernelOutputDtypeUnresolved) as info:
            na._reject_unresolved_tensor_binding(node, tree)
        assert info.value.subcode == "harness_binding_unresolved"


class TestRound6ImportAttributeWriter:
    """import-attribute-writer-18 (round-5 regression): an attribute the program itself stores anywhere -- on an
    imported root, a nested attribute of it or an alias of it -- stands for what was stored: module-defined code resolves
    to its effects, a store the prescan cannot follow is unresolved (declined with a capability).  Untouched imported
    attributes keep the ordinary-import rule (``np.random.seed(0)`` beside an uncalled writer lowers), and an
    uncalled store is not an effect."""

    @pytest.mark.parametrize("prefix,needle", [
        (_WRITER + "np._review4_change = change\nnp._review4_change()\n", "module_binding_uncertain: the module calls 'np._review4_change' at line 7, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        (_WRITER + "np.random._review4_change = change\nnp.random._review4_change()\n", "module_binding_uncertain: the module calls 'np.random._review4_change' at line 7, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        (_WRITER + "np._review4_change = change\nother = np\nother._review4_change()\n", "module_binding_uncertain: the module calls 'other._review4_change' at line 8, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
        ('np._review4_change = lambda: globals().__setitem__("Tensor", int)\nnp._review4_change()\n', "module_binding_uncertain: the module calls 'np._review4_change' at line 4, which resolves to the lambda at line 3, which calls globals(...) at line 3, which may rewrite the module namespace"),
        (_WRITER + "np._a, np._b = change, 1\nnp._a()\n", "module_binding_uncertain: the module calls 'np._a' at line 7, which cannot be resolved statically (the attribute '_a' is stored at line 6 by a statement the prescan cannot follow), while the function 'change' defined at line 3 declares 'Tensor' global and binds it"),
        (_WRITER + "def install():\n    np._review4_change = change\ninstall()\nnp._review4_change()\n", "module_binding_uncertain: the module calls 'np._review4_change' at line 9, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it"),
    ], ids=["stored_writer", "nested_root", "via_alias", "stored_writing_lambda", "tuple_store_unresolved", "stored_inside_a_called_function"])
    def test_stored_attribute_callee_declines(self, prefix, needle):
        _assert_typed_decline(_unannotated(prefix=prefix), "kernel_allocation_callee_unresolved", needle)

    def test_stored_attribute_invoked_from_run_model_declines(self):
        run = '\ndef run_model(inputs):\n    np._review4_change()\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n'
        _assert_typed_decline(_unannotated(prefix=_WRITER + "np._review4_change = change\n", run=run), "kernel_allocation_callee_unresolved",
                              "module_binding_uncertain: the harness calls 'run_model' (defined at line 12), which calls 'np._review4_change' at line 13, which resolves to the function 'change' defined at line 3, which declares 'Tensor' global and binds it")

    @pytest.mark.parametrize("prefix", [
        _WRITER + "np.random.seed(0)\n",
        _WRITER + "unused = np.zeros((1,))\n",
        _WRITER + "np._review4_change = change\n",
        _WRITER + "np._review4_change = change\nnp.random.seed(0)\n",
        _WRITER + "np._stored_zeros = np.zeros\nunused = np._stored_zeros((1,))\n",
    ], ids=["np_random_seed_beside_writer", "np_zeros_beside_writer", "uncalled_store", "untouched_attribute_beside_a_store", "stored_safe_attribute_called"])
    def test_ordinary_import_rule_kept(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])


class TestRound6EntryCallRounding:
    """entry-call-rounding-19: module-level code is lowered by the same transform as a plain def, so a kernel call in a
    callable entry (``run_model = lambda ...``), in a class body's method or in a statement executed at import is rounded
    at its call site exactly as inside ``run_model``: interpreter 2.0 == fast 2.0 on both carriers, no decline, no
    fallback.  an earlier revision appended such statements unchanged (public 2.003 against interpreter 2.0)."""

    def test_lambda_entry_is_rounded(self):
        code = _unannotated(run='\nrun_model = lambda inputs: (clear_dims(), {"output": k(inputs["a"]).data})[-1]\n')
        _lowers_to(code, [[2.0, 2.0]])
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert "run_model = lambda inputs: (clear_dims(), {'output': _truncate_bf16(k(inputs['a']))})[-1]" in low, low

    def test_class_method_calling_the_kernel_from_run_model_is_rounded(self):
        run = ('\nclass Runner:\n    def go(self, inputs):\n        return {"output": k(inputs["a"]).data}\n'
               '\ndef run_model(inputs):\n    clear_dims()\n    return Runner().go(inputs)\n')
        code = _unannotated(run=run)
        _lowers_to(code, [[2.0, 2.0]])
        assert "_truncate_bf16(k(inputs['a']))" in numba_lower(code, boundscheck=True, use_fp64=True)

    def test_module_level_kernel_call_is_rounded_in_place_and_read_in_its_own_scope(self):
        """The import-time ``FIRST = k(...)`` is rounded at its call site (round 7 wraps in place).  Round 8: ``FIRST.data``
        is rewritten only where ``FIRST`` is established -- in the MODULE scope that binds it once; the same read inside
        ``run_model`` is a free variable there and is the typed refusal ``name_unbound_in_scope`` (the interpreter answers
        2.0 on the public path).  Round 7 lowered the second form by name set; the ruling forbids that."""
        code = _unannotated(run='\nFIRST = k(np.ones((1, 2)))\nRAW = FIRST.data\n' + URUN.replace('k(inputs["a"]).data', 'RAW'))
        _lowers_to(code, [[2.0, 2.0]])
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert "FIRST = _truncate_bf16(k(np.ones((1, 2))))" in low and "RAW = FIRST" in low.splitlines(), low
        code = _unannotated(run='\nFIRST = k(np.ones((1, 2)))\n' + URUN.replace('k(inputs["a"]).data', 'FIRST.data'))
        _refuses(code, "kernel_function_statement_unresolved", "name_unbound_in_scope", "read from 'FIRST' in the function 'run_model'")


class TestReview4Round5EffectsVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round5_effects.py`` (sha256 recorded in the delivery; 24 cases x 2 carriers; it imports the
        round-3 and round-4 files) byte-for-byte: an earlier revision 22 failed / 26 passed (the 17/18/19 reproductions), head
        48 passed, no ruled deviation.  Records read back: every ``decline`` declines with a ``kernel_`` reason and the
        interpreter's error / value on the public path; every ``lower`` lowers to exactly 2.0 on both carriers without
        fallback."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_round3_scope.py", "test_round4_scope.py", "test_round4_generator_control.py", "test_round5_effects.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round5_effects.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        assert res.stdout.splitlines()[-1].startswith("48 passed"), res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round5-head").glob("*.json")}
        assert len(recs) == 48, len(recs)
        for name, r in recs.items():
            if r["expected"] == "decline":
                assert not r["lowered"] and r["declines"] and r["declines"][0].startswith("kernel_"), (name, r["declines"])
            else:
                assert r["lowered"] and not r["declines"] and not r["fallback"], (name, r["declines"], r["fallback"])
                assert r["raw"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["public"]["value"]["out"]["data"] == r["slow"]["value"]["out"]["data"], (name, r["raw"])
            if "error" in r["slow"]:
                assert r["public"].get("error") == r["slow"]["error"], (name, r["public"], r["slow"])
        for name in ("harness_class_static-True", "harness_class_static-False"):
            assert recs[name]["declines"][0].startswith("kernel_allocation_callee_unresolved:k: harness_binding_unresolved"), recs[name]["declines"]


_GOOD = URUN.replace("def run_model", "def good")
_HOLDER = "class Holder:\n    @staticmethod\n    def m():\n        global Tensor\n        Tensor = int\n"
_BOX = "class Box:\n    data = 5\n    def __int__(self):\n        return 9\n"
_SIDE = '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": k(inputs["a"]).data, "side": np.array(SIDE)}\n'


class TestRound7SameLineEntry:
    """same-line-entry-21: bindings of ``run_model`` are identified by their own AST node, never by line; a recognised
    assignment no longer hides a different final assignment sharing its line.  Unknown bindings are retained and make the
    entry the typed refusal ``harness_binding_unresolved``; a same-line annotated alias resolves like a plain alias."""

    @pytest.mark.parametrize("tail,needle", [
        ("run_model = good; run_model = (runner,)[0]\n", "harness_binding_unresolved: the harness calls 'run_model' (bound at line 17), which cannot be resolved statically ('run_model' is bound at line 17 by a statement the prescan cannot follow)"),
        ("run_model = good; run_model, other = runner, 0\n", "harness_binding_unresolved: the harness calls 'run_model' (bound at line 17), which cannot be resolved statically ('run_model' is bound at line 17 by a statement the prescan cannot follow)"),
        ("run_model = good; (run_model := runner)\n", "harness_binding_unresolved: the harness calls 'run_model' (bound at line 17), which cannot be resolved statically ('run_model' is bound at line 17 by an assignment expression the prescan cannot follow)"),
        ("run_model = good\nrun_model = (runner,)[0]\n", "harness_binding_unresolved: the harness calls 'run_model' (bound at line 18), which cannot be resolved statically ('run_model' is bound at line 18 by a statement the prescan cannot follow)"),
        ("run_model = good; run_model: object = runner\n", "module_binding_uncertain: the harness calls 'run_model' (bound at line 17 to 'runner'), which resolves to the function 'runner' defined at line 12, which declares 'Tensor' global and binds it"),
        ("run_model = good\nrun_model += 0\n", "harness_binding_unresolved: the harness calls 'run_model' (bound at line 18), which cannot be resolved statically ('run_model' is bound at line 18 by a statement the prescan cannot follow)"),
    ], ids=["same_line_subscript", "same_line_tuple", "same_line_walrus", "separate_line_subscript", "same_line_annotated_alias_resolves", "augmented"])
    def test_every_binding_statement_counts(self, tail, needle):
        code = _unannotated(run=_GOOD + _BADRUN + tail)
        seen = []
        assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=seen.append) is None
        assert seen and seen[0].startswith("kernel_allocation_callee_unresolved:k: " + needle), seen

    def test_safe_alias_chain_still_lowers(self):
        _lowers_to(_unannotated(run=_GOOD + "one = good\ntwo = one\nrun_model = two\n"), [[2.0, 2.0]])

    def test_same_line_safe_alias_pair_lowers(self):
        _lowers_to(_unannotated(run=_GOOD + "run_model = good; also = good\n"), [[2.0, 2.0]])


class TestRound7IntermediateAttribute:
    """intermediate-attribute-22: the COMPLETE accessed attribute path is resolved -- an intermediate object the program
    stored on an imported root (or on an alias of it, or from an installer called by ``run_model``) stands for what was
    stored (a class: its methods), so ``np._holder = Holder; np._holder.m()`` reaches the writer.  Untouched paths keep the
    ordinary-import rule and an uncalled intermediate store is not an effect."""

    @pytest.mark.parametrize("prefix,run,needle", [
        (_HOLDER + "np._review4_holder = Holder\nnp._review4_holder.m()\n", URUN, "module_binding_uncertain: the module calls 'np._review4_holder.m' at line 9, which resolves to the class 'Holder' defined at line 3, whose function 'm' defined at line 5 declares 'Tensor' global and binds it"),
        (_HOLDER + "other = np\nother._review4_holder = Holder\nnp._review4_holder.m()\n", URUN, "module_binding_uncertain: the module calls 'np._review4_holder.m' at line 10, which resolves to the class 'Holder' defined at line 3, whose function 'm' defined at line 5 declares 'Tensor' global and binds it"),
        (_HOLDER + "def install():\n    np._review4_holder = Holder\n", '\ndef run_model(inputs):\n    install()\n    np._review4_holder.m()\n    clear_dims()\n    return {"output": k(inputs["a"]).data}\n',
         "module_binding_uncertain: the harness calls 'run_model' (defined at line 15), which calls 'np._review4_holder.m' at line 17, which resolves to the class 'Holder' defined at line 3, whose function 'm' defined at line 5 declares 'Tensor' global and binds it"),
        (_WRITER + "def pick():\n    return change\nnp._review4_holder = pick\nnp._review4_holder.m()\n", URUN, "module_binding_uncertain: the module calls 'np._review4_holder.m' at line 9, which cannot be resolved statically ('np._review4_holder.m' is an attribute of the function 'pick' defined at line 6, stored under '_review4_holder'), while the function 'change' defined at line 3 declares 'Tensor' global and binds it"),
    ], ids=["stored_class", "via_alias_of_root", "installer_called_from_run_model", "stored_function_unresolved"])
    def test_stored_intermediate_object_declines(self, prefix, run, needle):
        _assert_typed_decline(_unannotated(prefix=prefix, run=run), "kernel_allocation_callee_unresolved", needle)

    @pytest.mark.parametrize("prefix", [_HOLDER + "np._review4_holder = Holder\n", _WRITER + "np.random.seed(0)\n", _HOLDER + "np._review4_holder = Holder\nunused = np.zeros((1,))\n"],
                             ids=["uncalled_intermediate_store", "np_random_seed_beside_writer", "untouched_path_beside_a_store"])
    def test_untouched_paths_keep_lowering(self, prefix):
        _lowers_to(_unannotated(prefix=prefix), [[2.0, 2.0]])


class TestRound7ModuleStatements:
    """module-data-transform-23 (round 7) under the round-8 ruling: module-level statements are lowered by the narrow
    ``_ModuleStatementLowerer``, which rewrites only ESTABLISHED DSL values -- kernel calls (rounded at the call site) and
    their ``.data`` reads, a module name bound once, unconditionally, in the module scope to one -- and leaves every other
    expression byte-identical.  Round 8 removed the round-7 "ordinary value" category: a ``.data`` read of ANY other base
    (an instance of a module class, an alias of one, a NumPy array, a lambda parameter, a call result, a subscript) is the
    typed refusal ``kernel_module_statement_unresolved:<kernel>: <class>: ...`` and the interpreter answers (side values
    5 / 5 / 5 / 1 preserved on the public path)."""

    @pytest.mark.parametrize("prefix,kind,src,side", [
        (_BOX + "SIDE = int(Box().data)\n", "call_result", "Box()", 5),
        (_BOX + "box = Box()\nSIDE = int(box.data)\n", "name_bound_to_non_kernel_value", "box", 5),
        (_BOX + "class Snapshot:\n    SIDE = int(Box().data)\nSIDE = Snapshot.SIDE\n", "call_result", "Box()", 5),
        ("SIDE = int(isinstance(np.arange(3).data, memoryview))\n", "call_result", "np.arange(3)", 1),
    ], ids=["object_data", "alias_data", "class_body_data", "numpy_memoryview_data"])
    def test_module_data_reads_of_unestablished_bases_refuse_and_the_interpreter_answers(self, prefix, kind, src, side):
        code = _unannotated(prefix=prefix, run=_SIDE)
        _refuses(code, "kernel_module_statement_unresolved", kind, f"read from {src!r} in ", side=side)

    def test_ordinary_numeric_module_statement_is_byte_identical(self):
        code = _unannotated(prefix="SIDE = float(np.sum(np.array([1.25, 2.5])))\n", run=_SIDE)
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert low is not None and "SIDE = float(np.sum(np.array([1.25, 2.5])))" in low.splitlines(), low
        for use_fp64 in (True, False):
            fell = []
            raw = execute_fast(code, {"a": np.ones((1, 2))}, use_fp64=use_fp64, on_fallback=fell.append)
            assert not fell and float(np.asarray(raw["side"])) == 3.75 and np.asarray(raw["output"]).tolist() == [[2.0, 2.0]], (use_fp64, fell, raw)

    def test_unestablished_data_base_is_a_typed_refusal(self):
        code = _unannotated(prefix=_BOX + "read = lambda x: int(x.data)\nSIDE = read(Box())\n", run=_SIDE)
        seen = []
        assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=seen.append) is None
        assert seen[0].startswith("kernel_module_statement_unresolved:k: name_bound_by_parameter: a '.data' attribute at line 7 is read from 'x' in a lambda at line 7; 'x' is a parameter at line 7"), seen
        x = np.ones((1, 2))
        slow = execute_translated(code, {}, all_inputs={"a": x}, fast=False)[0]
        public = execute_translated(code, {}, all_inputs={"a": x}, fast=True, use_fp64=True)[0]
        assert float(np.asarray(public["side"])) == 5.0 == float(np.asarray(slow["side"]))       # the interpreter answers

    @pytest.mark.parametrize("prefix,kind", [("def mk():\n    return Box()\nSIDE = int(mk().data)\n", "call_result"),
                                             ("pair = (Box(), k(np.ones((1, 2))))\nSIDE = int(pair[0].data)\n", "subscript"),
                                             ("def mk():\n    return Box()\nSIDE = int((mk() or Box()).data)\n", "expression"),
                                             ("boxes = [Box()]\nSIDE = int(boxes[0].data)\n", "subscript")],
                             ids=["module_function_result", "subscript_of_a_display_holding_a_kernel_result", "boolop", "subscript_of_an_ordinary_display"])
    def test_other_unestablished_bases_refuse_too(self, prefix, kind):
        code = _unannotated(run="\n" + _BOX + prefix + _SIDE)
        _refuses(code, "kernel_module_statement_unresolved", kind, "", side=5)

    def test_established_dsl_values_are_still_lowered(self):
        code = _unannotated(run='\nFIRST = k(np.ones((1, 2)))\nRAW = FIRST.data\n' + URUN.replace('k(inputs["a"]).data', 'RAW'))
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert "FIRST = _truncate_bf16(k(np.ones((1, 2))))" in low and "RAW = FIRST" in low.splitlines(), low
        _lowers_to(code, [[2.0, 2.0]])

    def test_class_method_kernel_call_and_lambda_entry_still_rounded(self):
        _lowers_to(_unannotated(run='\nclass Runner:\n    def go(self, inputs):\n        return {"output": k(inputs["a"]).data}\n\ndef run_model(inputs):\n    clear_dims()\n    return Runner().go(inputs)\n'), [[2.0, 2.0]])
        _lowers_to(_unannotated(run='\nrun_model = lambda inputs: (clear_dims(), {"output": k(inputs["a"]).data})[-1]\n'), [[2.0, 2.0]])


class TestReview4Round6BoundariesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round6_boundaries.py`` byte-for-byte (30 cases x 2 carriers; it imports the round-3/4/5 files):
        pristine an earlier revision 26 failed / 34 passed; round-7 head 58 passed / 2 ruled.  Round 8: 56 passed and exactly four records
        fail -- the two ``entry_unbound`` records (the reviewer's own over-broad expectation for a module with NO run_model binding
        at all; the public path preserves the interpreter's missing-entry error; ruled not required) and the
        two ``module_importtime_kernel`` records, whose ``run_model`` reads ``FIRST.data`` of a MODULE name: under the round-8
        same-scope ruling that is the typed refusal ``name_unbound_in_scope`` with the interpreter's 2.0 on the public path
        (a reported over-refusal, not a wrong value).  The four ``safe`` module-``.data`` records (``Box().data``, an alias,
        a class body, a NumPy array) are typed refusals too, their side values answered by the interpreter.  Records read back: every ``decline`` declines with a ``kernel_`` reason and
        the interpreter's error on the public path; every ``lower`` lowers to exactly 2.0 without fallback; every ``safe``
        record's public value equals the interpreter's."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_round3_scope.py", "test_round4_scope.py", "test_round4_generator_control.py", "test_round5_effects.py", "test_round6_boundaries.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round6_boundaries.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        assert "4 failed, 56 passed" in res.stdout.splitlines()[-1], res.stdout[-3000:]
        failed = sorted(l.split("::")[-1].split(" ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED "))
        assert failed == ["test_round6_boundaries[False-entry_unbound]", "test_round6_boundaries[False-module_importtime_kernel]", "test_round6_boundaries[True-entry_unbound]", "test_round6_boundaries[True-module_importtime_kernel]"], failed
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round6-head").glob("*.json")}
        assert len(recs) == 60, len(recs)
        for name, r in recs.items():
            if name.startswith("entry_unbound"):
                assert r["lowered"] and r["public"].get("error") == r["slow"].get("error") and "error" in r["slow"], (name, r["slow"], r["public"])
                continue
            if name.startswith("module_importtime_kernel"):
                assert not r["lowered"] and r["declines"][0].startswith("kernel_function_statement_unresolved:k: name_unbound_in_scope: a '.data' attribute at line 12 is read from 'FIRST' in the function 'run_model'"), r["declines"]
                assert r["public"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["slow"]["value"]["out"]["data"], r["public"]
                continue
            if r["expected"] == "decline":
                assert not r["lowered"] and r["declines"] and r["declines"][0].startswith("kernel_"), (name, r["declines"])
            elif r["expected"] == "lower":
                assert r["lowered"] and not r["declines"] and not r["fallback"], (name, r["declines"], r["fallback"])
                assert r["raw"]["value"]["out"]["data"] == [[2.0, 2.0]] == r["public"]["value"]["out"]["data"], (name, r["raw"])
            if "error" in r["slow"]:
                assert r["public"].get("error") == r["slow"]["error"], (name, r["public"], r["slow"])
            else:
                assert r["public"]["value"] == r["slow"]["value"] or not r["fp64"], name
        for fp in ("True", "False"):
            assert recs[f"module_lambda_data-{fp}"]["declines"][0].startswith("kernel_module_statement_unresolved:k: name_bound_by_parameter:"), recs[f"module_lambda_data-{fp}"]["declines"]
            for nm, kind, side in (("module_object_data", "call_result", 5), ("module_alias_data", "name_bound_to_non_kernel_value", 5), ("module_class_body_data", "call_result", 5), ("module_numpy_data", "call_result", 1)):
                r = recs[f"{nm}-{fp}"]
                assert not r["lowered"] and r["declines"][0].startswith(f"kernel_module_statement_unresolved:k: {kind}:"), (nm, r["declines"])
                assert r["public"]["value"]["side"]["data"] == side == r["slow"]["value"]["side"]["data"], (nm, r["public"])
            for nm in ("module_ordinary_numeric", "module_class_kernel", "entry_alias_safe", "attr_ordinary_import"):
                r = recs[f"{nm}-{fp}"]
                assert r["lowered"] and not r["declines"] and not r["fallback"] and r["public"]["value"]["out"]["data"] == [[2.0, 2.0]], (nm, r["declines"])


_MODEL = '\ndef model(a):\n    return k(a)\n'
_R8 = {  # (run_model body after clear_dims(), refusal class, needle) -- 26 and the function-body forms of 24 / 25 (round 9: the
      # plain-function call form moved to TestRound9Narrowing; the mixed tuple display keeps refusing, element 2 is a constant)
    "unpacking": ('    out, other = k(inputs["a"]), 1\n    return {"output": out.data}\n', "name_bound_by_unpacking", "as target 1 of 2 of the unpacking assignment '(out, other) = (k(inputs['a']), 1)' at line"),
    "conditional": ('    if True:\n        out = k(inputs["a"])\n    return {"output": out.data}\n', "name_bound_under_control_flow", "inside the 'if' statement at line"),
    "for_target": ('    for out in [k(inputs["a"])]:\n        pass\n    return {"output": out.data}\n', "name_bound_under_control_flow", "inside the 'for' statement at line"),
    "rebound": ('    out = k(inputs["a"])\n    out = k(inputs["a"])\n    return {"output": out.data}\n', "name_multiply_bound", "'out' has 2 binding sites in the function 'run_model'"),
    "deleted_later": ('    out = k(inputs["a"])\n    r = out.data\n    del out\n    return {"output": r}\n', "name_multiply_bound", "deleted at line"),
    "nonlocal_store": ('    out = k(inputs["a"])\n    def h():\n        nonlocal out\n        out = 1\n    return {"output": out.data}\n', "name_multiply_bound", "declared nonlocal at line"),
    "walrus": ('    (out := k(inputs["a"]))\n    return {"output": out.data}\n', "name_bound_by_namedexpr", "by the namedexpr statement"),
    "augmented": ('    out = 0\n    out += k(inputs["a"])\n    return {"output": out.data}\n', "name_multiply_bound", "'out' has 2 binding sites"),
    "parameter": ('    return {"output": read(k(inputs["a"]))}\n', "name_bound_by_parameter", "'x' is a parameter at line"),
    "subscript": ('    box = {"x": k(inputs["a"])}\n    return {"output": box["x"].data}\n', "subscript", "\"box['x']\" is a subscript, whose value the lowering does not follow"),
    "attribute": ('    box = Box()\n    box.t = k(inputs["a"])\n    return {"output": box.t.data}\n', "attribute", "'box.t' is an attribute of another value, whose value the lowering does not follow"),
    "call_result": ('    return {"output": next(iter([k(inputs["a"])])).data}\n', "call_result", "is the result of a call that is not a direct @kernel call"),
    "stored_attribute": ('    out = k(inputs["a"])\n    out.data = 1\n    return {"output": out.data}\n', "stored_or_deleted_attribute", "store / delete target"),
    "comprehension_free_variable": ('    out = k(inputs["a"])\n    return {"output": [out.data for _ in [0]][0]}\n', "name_unbound_in_scope", "'out' has no binding in a comprehension at line"),
    "lambda_free_variable": ('    out = k(inputs["a"])\n    return {"output": (lambda: out.data)()}\n', "name_unbound_in_scope", "'out' has no binding in a lambda at line"),
}


class TestRound8EstablishedOrRefuse:
    """an earlier fix round 8 -- MODULE-PROVENANCE-24, ORDINARY-VALUE-PROOF-25, FUNCTION-DATA-26 under the reviewed binding ruling: a
    ``.data`` read is rewritten ONLY when its base is established by the syntax of its own statement -- a direct @kernel call,
    or a name bound once, unconditionally, in the SAME scope by an assignment statement to the whole value of one (plain,
    chained or annotated) -- with no other binding site of that name in that scope (a second assignment of any form, a
    binding under control flow, a parameter, a lambda parameter or comprehension target, a del / global / nonlocal).
    Everything else is the typed refusal ``kernel_module_statement_unresolved`` (module / class scope) or
    ``kernel_function_statement_unresolved`` (function bodies, plain functions and class methods alike), never a guess; the
    interpreter answers on the public path.  No value provenance is proved by name sets; nothing is classified "ordinary"."""

    @pytest.mark.parametrize("body,pin", [
        ('    out = k(inputs["a"])\n    return {"output": out.data}\n', "out = _truncate_bf16(out)"),
        ('    x = out = k(inputs["a"])\n    return {"output": out.data}\n', "x = out = _truncate_bf16(k(inputs['a']))"),
        ('    out: object = k(inputs["a"])\n    return {"output": out.data}\n', "out: object = _truncate_bf16(k(inputs['a']))"),
        ('    return {"output": k(inputs["a"]).data}\n', "return {'output': _truncate_bf16(k(inputs['a']))}"),
        ('    return {"output": [k(inputs["a"]).data for _ in [0]][0]}\n', "[_truncate_bf16(k(inputs['a'])) for _ in [0]][0]"),
    ], ids=["plain_assignment", "chained_assignment", "annotated_assignment", "direct_call", "direct_call_in_a_comprehension"])
    def test_established_bases_lower_and_round_on_both_carriers(self, body, pin):
        code = _unannotated(run="\ndef run_model(inputs):\n    clear_dims()\n" + body)
        _lowers_to(code, [[2.0, 2.0]])
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert pin in low and ".data" not in low, (pin, low)

    @pytest.mark.parametrize("case", sorted(_R8), ids=sorted(_R8))
    def test_function_body_reads_of_unestablished_bases_are_typed_refusals(self, case):
        body, kind, needle = _R8[case]
        prefix = _BOX + "def read(x):\n    return x.data\n" if case == "parameter" else _BOX if case == "attribute" else ""
        code = _unannotated(prefix=prefix, run=_MODEL + "\ndef run_model(inputs):\n    clear_dims()\n" + body)
        _refuses(code, "kernel_function_statement_unresolved", kind, needle)

    @pytest.mark.parametrize("body,kind,needle", [
        ("X = k(np.ones((1, 2)))\nX = Box()\nSIDE = int(X.data)\n", "name_multiply_bound", "'X' has 2 binding sites in the module body"),
        ("X = k(np.ones((1, 2)))\nX, unused = Box(), 0\nSIDE = int(X.data)\n", "name_multiply_bound", "'X' has 2 binding sites"),
        ("X = k(np.ones((1, 2)))\nX: object = Box()\nSIDE = int(X.data)\n", "name_multiply_bound", "'X' has 2 binding sites"),
        ("X = k(np.ones((1, 2)))\n(X := Box())\nSIDE = int(X.data)\n", "name_multiply_bound", "'X' has 2 binding sites"),
        ("if False:\n    X = k(np.ones((1, 2)))\nelse:\n    X = Box()\nSIDE = int(X.data)\n", "name_multiply_bound", "'X' has 2 binding sites"),
        ("X = Box()\nclass Snapshot:\n    X = k(np.ones((1, 2)))\nSIDE = int(X.data)\n", "name_bound_to_non_kernel_value", "'X' is bound once, by 'X = Box()' at line"),
        ("X = k(np.ones((1, 2)))\nread = lambda X: int(X.data)\nSIDE = read(Box())\n", "name_bound_by_parameter", "'X' is a parameter at line"),
        ("X = k(np.ones((1, 2)))\nSIDE = [int(X.data) for X in (Box(),)][0]\n", "name_bound_by_a_comprehension_target", "'X' is a comprehension target at line"),
        ("X = k(np.ones((1, 2)))\ndef other():\n    global X\n    X = Box()\nother()\nSIDE = int(X.data)\n", "name_multiply_bound", "declared global at line"),
    ], ids=["reassign_plain", "reassign_tuple", "reassign_annotated", "reassign_walrus", "conditional_binding", "class_scope_leak", "lambda_shadow", "comprehension_shadow", "global_store_elsewhere"])
    def test_module_rebinding_forms_refuse_and_the_side_value_is_the_interpreters(self, body, kind, needle):
        """module-provenance-24: an earlier revision / an earlier revision returned 9 (``int(X)`` of a Box) for the first five; every form
        is now the typed refusal and the public side value is the interpreter's 5."""
        code = _unannotated(run="\n" + _BOX + body + _SIDE)
        _refuses(code, "kernel_module_statement_unresolved", kind, needle, side=5)

    @pytest.mark.parametrize("body,kind,src", [
        ("SIDE = int(isinstance(next(iter([k(np.ones((1, 2)))])).data, np.ndarray))\n", "call_result", "next(iter([k(np.ones((1, 2)))]))"),
        ("class Store:\n    X = k(np.ones((1, 2)))\nSIDE = int(isinstance(getattr(Store, 'X').data, np.ndarray))\n", "call_result", "getattr(Store, 'X')"),
        ("class Factory:\n    def __new__(cls):\n        return k(np.ones((1, 2)))\nSIDE = int(isinstance(Factory().data, np.ndarray))\n", "call_result", "Factory()"),
        ("X = []\nX.append(k(np.ones((1, 2))))\nSIDE = int(isinstance(X[0].data, np.ndarray))\n", "subscript", "X[0]"),
        ("box = Box()\nbox.tensor = k(np.ones((1, 2)))\nSIDE = int(isinstance(box.tensor.data, np.ndarray))\n", "attribute", "box.tensor"),
    ], ids=["builtin_next", "builtin_getattr", "class_new_returns_a_kernel_result", "container_mutation", "object_field"])
    def test_ordinary_roots_are_no_proof_about_reachable_values(self, body, kind, src):
        """ordinary-value-proof-25: an earlier revision kept ``.data`` after replacing the nested kernel result (memoryview, side 0
        against the interpreter's 1); every such base is now the typed refusal and the public side value is 1."""
        code = _unannotated(run="\n" + _BOX + body + _SIDE)
        _refuses(code, "kernel_module_statement_unresolved", kind, f"read from {src!r} in the module body", side=1)

    @pytest.mark.parametrize("prefix,needle", [
        (_BOX + "def read(x):\n    return int(x.data)\n", "read from 'x' in the function 'read'; 'x' is a parameter at line 7"),
        (_BOX + "class Reader:\n    def read(self, x):\n        return int(x.data)\n", "read from 'x' in the function 'read'; 'x' is a parameter at line 8"),
    ], ids=["plain_function", "class_method"])
    def test_function_body_data_reads_of_parameters_refuse(self, prefix, needle):
        """function-data-26: the baseline ``_RunModelLowerer`` stripped ``.data`` unconditionally (``int(x)`` of a Box is
        9); plain functions and class methods now follow the same established-or-refuse rule (public side value 5)."""
        run = "\nSIDE = " + ("Reader().read(Box())" if "Reader" in prefix else "read(Box())") + "\n" + _SIDE
        code = _unannotated(prefix=prefix, run=run)
        _refuses(code, "kernel_function_statement_unresolved", "name_bound_by_parameter", needle, side=5)

    @pytest.mark.parametrize("run", [
        '\ndef run_model(inputs):\n    clear_dims()\n    out = k(inputs["a"])\n    def helper():\n        out = 1\n        return out\n    return {"output": out.data, "h": np.array(helper())}\n',
        '\nout = k(np.ones((1, 2)))\nclass C:\n    out = 1\nRAW = out.data\n\ndef run_model(inputs):\n    clear_dims()\n    return {"output": RAW}\n',
    ], ids=["nested_def_local", "class_body_attribute"])
    def test_bindings_of_other_scopes_do_not_rebind_the_established_name(self, run):
        """Precision of the per-scope analysis: a local of a nested ``def`` and a class-body attribute bind THEIR scopes, not the
        enclosing name (only ``global`` / ``nonlocal`` declarations reach out), so the established name still lowers."""
        _lowers_to(_unannotated(run=run), [[2.0, 2.0]])

    def test_multiple_unestablished_reads_report_the_first_and_the_count(self):
        code = _unannotated(prefix=_BOX + "SIDE = int(Box().data) + int(Box().data)\n", run=_SIDE)
        seen = []
        assert numba_lower(code, boundscheck=True, use_fp64=True, on_decline=seen.append) is None
        assert seen[0].startswith("kernel_module_statement_unresolved:k: call_result: a '.data' attribute at line 7") and seen[0].endswith("(2 unestablished '.data' reads in this program)"), seen


_RM = "\ndef model(a):\n    return k(a)\n"
_RD = '    return {"output": out.data}\n'
_R9_LOWER = {  # name -> module text after the kernel (helpers + run_model); every form lowers to the kernel's 2.0 on both carriers
    "iii_direct_kernel_return": _RM + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD,
    "iii_call_read_directly": _RM + '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": model(inputs["a"]).data}\n',
    "iii_local_then_return": "\ndef model(a):\n    h = k(a)\n    return h\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD,
    "iii_two_levels": "\ndef inner(a):\n    return k(a)\ndef model(a):\n    return inner(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD,
    "iii_every_return_established": "\ndef model(a, flag=True):\n    if flag:\n        return k(a)\n    return k(k(a))\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD,
    "iii_keyword_argument_and_default": "\ndef model(a, scale=1):\n    return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(a=inputs["a"])\n' + _RD,
    "iii_parameter_established_by_the_argument": "\ndef ident(t):\n    return t\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = ident(k(inputs["a"]))\n' + _RD,
    "ii_alias": '\ndef run_model(inputs):\n    clear_dims()\n    x = k(inputs["a"])\n    out = x\n' + _RD,
    "ii_alias_of_a_chained_binding": '\ndef run_model(inputs):\n    clear_dims()\n    x = y = k(inputs["a"])\n    out = y\n' + _RD,
    "iv_tuple_display": '\ndef run_model(inputs):\n    clear_dims()\n    out, other = k(inputs["a"]), k(inputs["a"])\n' + _RD,
    "iv_function_returning_a_display_of_locals": "\ndef model(a):\n    p = k(a)\n    q = k(a)\n    return p, q\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out, other = model(inputs["a"])\n' + _RD,
    "iv_function_returning_a_display_of_calls": "\ndef model(a):\n    return k(a), k(k(a))\n" + '\ndef run_model(inputs):\n    clear_dims()\n    other, out = model(inputs["a"])\n' + _RD,
    "module_level_iii": _RM + "OUT = model(np.ones((1, 2)))\nRAW = OUT.data\n" + '\ndef run_model(inputs):\n    clear_dims()\n    return {"output": RAW}\n',
}
_R9_REFUSE = {  # name -> (module text after the kernel, refusal class, needle); the interpreter answers on the public path
    "iii_recursion": ("\ndef model(a):\n    return model(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' is called recursively (model -> model)"),
    "iii_mutual_recursion": ("\ndef f(a):\n    return g(a)\ndef g(a):\n    return f(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = f(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'f' is called recursively (f -> g -> f)"),
    "iii_bare_return": ("\ndef model(a):\n    if a is None:\n        return\n    return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' has a bare return at line"),
    "iii_fall_through": ("\ndef model(a):\n    if a is not None:\n        return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' can fall through its last statement"),
    "iii_generator": ("\ndef model(a):\n    yield k(a)\n    return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' is a generator / coroutine function"),
    "iii_decorated": ("\nimport functools\n@functools.lru_cache(None)\ndef model(a):\n    return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' is not a module-defined plain function"),
    "iii_class_call": ("\n" + _BOX + '\ndef run_model(inputs):\n    clear_dims()\n    out = Box()\n' + _RD, "name_bound_to_non_kernel_value", "'Box' is not a module-defined plain function"),
    "iii_callee_rebound_in_the_module": (_RM + "model = model\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' has 2 binding sites in the module"),
    "iii_callee_shadowed_in_the_scope": (_RM + '\ndef run_model(inputs):\n    clear_dims()\n    model = lambda a: 1\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' is bound in the function 'run_model'"),
    "iii_star_call": (_RM + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(*[inputs["a"]])\n' + _RD, "name_bound_to_non_kernel_value", "passes *args / **kwargs"),
    "iii_varargs_definition": ("\ndef model(*a):\n    return k(a[0])\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'model' declares *args / **kwargs"),
    "iii_unknown_keyword": (_RM + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"], zz=1)\n' + _RD, "name_bound_to_non_kernel_value", "the keyword 'zz' is not a (keyword-bindable) parameter of 'model'"),
    "iii_missing_argument": ("\ndef model(a, b):\n    return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "does not bind the parameter(s) ['b'] of 'model'"),
    "iii_parameter_without_an_established_argument": ("\ndef ident(t):\n    return t\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = ident(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'t' is a parameter at line 8 in the function 'ident' (no established argument reaches it from this call site)"),
    "iii_return_of_an_attribute_view": ("\ndef model(a):\n    h = k(a)\n    return h.int32\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'h.int32' is an attribute of another value"),
    "iii_return_of_an_expression": ("\ndef model(a):\n    return k(a) + 1\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "'k(a) + 1' is a BinOp expression"),
    "iii_method_call": ("\nclass M:\n    def model(self, a):\n        return k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = M().model(inputs["a"])\n' + _RD, "name_bound_to_non_kernel_value", "is the result of a call that is not a direct @kernel call, whose value the lowering does not follow"),
    "iv_mixed_display": ('\ndef run_model(inputs):\n    clear_dims()\n    out, other = k(inputs["a"]), 1\n' + _RD, "name_bound_by_unpacking", "element 2 of the tuple display ('1') is not established"),
    "iv_arity_mismatch": ("\ndef model(a):\n    return k(a), k(a), k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out, other = model(inputs["a"])\n' + _RD, "name_bound_by_unpacking", "is not a tuple display of 2 values"),
    "iv_starred_target": ("\ndef model(a):\n    return k(a), k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out, *rest = model(inputs["a"])\n' + _RD, "name_bound_by_unpacking", "whose targets are not a flat tuple of names"),
    "iv_nested_target": ("\ndef model(a):\n    return (k(a), k(a)), k(a)\n" + '\ndef run_model(inputs):\n    clear_dims()\n    (out, o2), o3 = model(inputs["a"])\n' + _RD, "name_bound_by_unpacking", "whose targets are not a flat tuple of names"),
    "iv_tuple_through_a_name": ("\ndef model(a):\n    t = (k(a), k(a))\n    return t\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out, other = model(inputs["a"])\n' + _RD, "name_bound_by_unpacking", "is not a tuple display of 2 values"),
    "ii_self_reference": ('\ndef run_model(inputs):\n    clear_dims()\n    out = out\n' + _RD, "name_bound_to_non_kernel_value", "refers back to 'out' itself"),
    "parameter_read_inside_a_helper": ("\ndef model(a):\n    return k(a)\ndef peek(t):\n    return t.data\n" + '\ndef run_model(inputs):\n    clear_dims()\n    out = model(inputs["a"])\n    return {"output": peek(out)}\n', "name_bound_by_parameter", "'t' is a parameter at line"),
}


class TestRound9Narrowing:
    """Established-or-refuse coverage for the supported value forms.
    ESTABLISHED is: (i) a direct ``@kernel`` call; (ii) a name with exactly
    one binding site, an unconditional assignment statement whose right-hand side is established; (iii) a direct call, by
    name, of a MODULE-DEFINED PLAIN FUNCTION -- one binding site in the module (its ``def``), undecorated, not async, no
    ``*args`` / ``**kwargs`` on either side, declared parameters only, no recursion, no generator, at least one ``return``,
    no bare ``return`` and no fall-through -- whose EVERY return is established in its own scope, a parameter counting only
    when this call site passes an established argument; (iv) a target of a flat tuple unpacking whose right-hand side is a
    tuple DISPLAY of established values of matching arity or a (iii) call whose every return is such a display.  Everything
    else keeps its round-8 typed refusal; nothing is proved by name sets."""

    @needs_numba
    @pytest.mark.parametrize("case", sorted(_R9_LOWER), ids=sorted(_R9_LOWER))
    def test_established_through_plain_functions_and_tuple_displays_lowers(self, case):
        code = _unannotated(run=_R9_LOWER[case])
        _lowers_to(code, [[2.0, 2.0]])
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert ".data" not in low, low

    @pytest.mark.parametrize("case", sorted(_R9_REFUSE), ids=sorted(_R9_REFUSE))
    def test_every_other_form_stays_a_typed_refusal(self, case):
        run, kind, needle = _R9_REFUSE[case]
        _refuses(_unannotated(run=run), "kernel_function_statement_unresolved", kind, needle)

    def test_the_analysis_reads_the_pristine_program_not_the_rewritten_one(self):
        """Helpers are lowered before ``run_model`` (``k(a)`` becomes ``_truncate_bf16(k(a))`` inside ``model``); rule (iii)
        must judge the ORIGINAL return, so the lowering snapshots every scope before rewriting it."""
        code = _unannotated(run=_R9_LOWER["iii_direct_kernel_return"])
        low = numba_lower(code, boundscheck=True, use_fp64=True)
        assert low is not None and "return _truncate_bf16(k(a))" in low and "out = model(inputs['a'])" in low and "'output': out}" in low, low

    def test_rule_iii_is_judged_per_call_site(self):
        """``ident`` returns its parameter: established for the call that passes a kernel result, not for the other."""
        run = ("\ndef ident(t):\n    return t\n" + '\ndef run_model(inputs):\n    clear_dims()\n    good = ident(k(inputs["a"]))\n'
               '    bad = ident(inputs["a"])\n    return {"output": good.data, "echo": np.asarray(bad)}\n')
        low = numba_lower(_unannotated(run=run), boundscheck=True, use_fp64=True)
        assert low is not None and "'output': good," in low, low
        run_bad = run.replace('"echo": np.asarray(bad)', '"echo": bad.data')
        _refuses(_unannotated(run=run_bad), "kernel_function_statement_unresolved", "name_bound_to_non_kernel_value", "'t' is a parameter at line 8 in the function 'ident' (no established argument reaches it from this call site)")

    def test_review4_round8_counterexamples_still_refuse_or_compute_correctly(self):
        """The round-7 review's 24 / 25 / 26 shapes are untouched by the narrowing: a module name rebound through a class call,
        an ordinary root reaching a kernel result, a parameter read in a function body."""
        _refuses(_unannotated(run="\n" + _BOX + "X = k(np.ones((1, 2)))\nX = Box()\nSIDE = int(X.data)\n" + _SIDE), "kernel_module_statement_unresolved", "name_multiply_bound", "'X' has 2 binding sites", side=5)
        _refuses(_unannotated(run="\n" + _BOX + "SIDE = int(isinstance(next(iter([k(np.ones((1, 2)))])).data, np.ndarray))\n" + _SIDE), "kernel_module_statement_unresolved", "call_result", "next(iter([k(np.ones((1, 2)))]))", side=1)
        _refuses(_unannotated(prefix=_BOX + "def read(x):\n    return int(x.data)\n", run="\nSIDE = read(Box())\n" + _SIDE), "kernel_function_statement_unresolved", "name_bound_by_parameter", "'x' is a parameter at line 7", side=5)


class TestReview4Round7BoundariesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round7_boundaries.py`` byte-for-byte (sha256 7fec874b…, 30 cases x 2 carriers; it imports the
        round-3/4/5 files): pristine an earlier revision 30 failed / 30 passed (the review's matrix); round-8 head 56 passed / 4 failed;
        round-9 head (the narrowing) 58 passed and exactly two records fail, both the typed refusal of
        ``data_plain_numeric`` (``Box().data`` -- a CLASS call is not a module-defined plain function, rule (iii)), which the review
        expected to LOWER: a reported over-refusal.  ``data_known_dsl_alias`` (``Y = X`` of an established ``X``) is
        established by rule (ii) and lowers.  Every one of the 60 records is either CORRECT on the public path or a typed
        refusal with the interpreter's value: no record changes a public result."""
        run = tmp_path / "run"; run.mkdir()
        for sub in ("runs", "evidence", "probe_modules"):
            (run / sub).mkdir()
        for name in ("test_binding_edges.py", "test_round3_scope.py", "test_round4_scope.py", "test_round4_generator_control.py", "test_round5_effects.py", "test_round6_boundaries.py", "test_round7_boundaries.py"):
            (run / name).write_bytes((R3 / name).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round7_boundaries.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1200)
        assert "2 failed, 58 passed" in res.stdout.splitlines()[-1], res.stdout[-3000:]
        failed = sorted(l.split("::")[-1].split(" ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED "))
        assert failed == ["test_round7_boundaries[False-data_plain_numeric]", "test_round7_boundaries[True-data_plain_numeric]"], failed
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round7-head").glob("*.json")}
        assert len(recs) == 60, len(recs)
        kinds = {"data_reassign_plain": "name_multiply_bound", "data_reassign_tuple": "name_multiply_bound", "data_reassign_ann": "name_multiply_bound",
                 "data_reassign_walrus": "name_multiply_bound", "data_conditional_binding": "name_multiply_bound", "data_class_scope_leak": "name_bound_to_non_kernel_value",
                 "data_lambda_shadow": "name_bound_by_parameter", "data_comprehension_shadow": "name_bound_by_a_comprehension_target",
                 "data_builtin_next": "call_result", "data_builtin_getattr": "call_result", "data_class_new_returns_dsl": "call_result",
                 "data_plain_container_mutation": "subscript", "data_plain_object_field": "attribute",
                 "data_plain_numeric": "call_result"}
        for name, r in recs.items():
            stem = name.rsplit("-", 1)[0]
            if "error" in r["slow"]:
                assert r["public"].get("error") == r["slow"]["error"], (name, r["public"], r["slow"])
            else:
                assert r["public"]["value"] == r["slow"]["value"] or not r["fp64"], name
            refusal = not r["lowered"] and bool(r["declines"]) and r["declines"][0].startswith("kernel_")
            if r["expected"] == "decline":
                assert refusal, (name, r["declines"])
            elif r["expected"] == "lower" and stem != "data_plain_numeric":
                assert r["lowered"] and not r["declines"] and not r["fallback"] and r["public"]["value"]["out"]["data"] == [[2.0, 2.0]], (name, r["declines"])
            if stem in kinds:
                code = "kernel_function_statement_unresolved" if stem in ("data_function_body", "data_method_body") else "kernel_module_statement_unresolved"
                assert refusal and r["declines"][0].startswith(f"{code}:k: {kinds[stem]}:"), (name, r["declines"])
            if stem in ("data_function_body", "data_method_body"):
                assert refusal and r["declines"][0].startswith("kernel_function_statement_unresolved:k: name_bound_by_parameter:") and r["public"]["value"]["side"]["data"] == 5, (name, r)
