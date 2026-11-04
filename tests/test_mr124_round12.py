# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier fix round 12 -- repairs of the reviewer's NO on an earlier revision (an earlier review.md).

* mutation-scope-36: the escape collector proves "this load is NOT the module def" by ACTUAL lexical binding resolution
  (``_lexical_scope`` / ``_load_resolves_to_module``), never by the conservative-refusal inventory ``_BindingScope.sites`` -- which
  charges a nested lambda's parameters, a comprehension's targets and stores inside a nested lambda body to the containing scope so
  that a CALL of that spelling is refused.  Round 11 commit 5 used that inventory as the binding set and hid
  ``unused = lambda model: model; alias = model; alias.__code__ = replacement.__code__`` (fast 9 vs interpreter 5).
* baseline-invalid-dtype-37 (baseline-owned): a LITERAL ``dtype=`` of a ``Tensor`` call that the
  interpreter rejects (``VALID_DTYPES``, case-sensitive) is a typed refusal ``kernel_invalid_dtype_literal``, no longer a value.
* the reviewer's frozen ``test_round11_escape_resolution.py`` (+ its harness ``test_round11_compatible.py``) vendored byte-for-byte and
  run as a subprocess; the harness file's own cases are pinned with their ruled deviation set (the module-lambda parameter ruling).
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.core import VALID_DTYPES
from solar.einsum_lib.numba_accel import (_BindingScope, _Program, _collect_escapes, _lexical_scope, _load_resolves_to_module,
                                          is_typed_decline)
from test_mr124_round11 import (BF16, F32, H, INPUTS, RM, R3, _END, _MODEL, _SIDE, _lowers, _refuses, _run, _run_reviewer_file)

ROOT = Path(__file__).resolve().parents[1]
MODULE_REFUSAL = "kernel_module_statement_unresolved:k: function_implementation_unproved:"


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


# ---------------------------------------------------------------------------------------------------------------------
# mutation-scope-36
# ---------------------------------------------------------------------------------------------------------------------

_NESTED = {                              # the reviewer's six shapes: a nested scope binds the spelling, the containing function does not
    "lambda_parameter": "    unused = lambda model: model\n",
    "lambda_walrus": "    unused = lambda: (model := 0)\n",
    "list_target": "    unused = [model for model in ()]\n",
    "generator_target": "    unused = (model for model in ())\n",
    "dict_target": "    unused = {model: model for model in ()}\n",
    "set_target": "    unused = {model for model in ()}\n",
}
_MUTATE = "    alias = model\n    alias.__code__ = replacement.__code__\nmutate()\n"
_SAFE = {                                # the reviewer's valid shadows: the containing function really binds the spelling
    "local_shadow": "def unused():\n    model = lambda a: a\n    alias = model\n",
    "parameter_shadow": "def unused(model):\n    alias = model\n",
    "loop_target_shadow": "def unused():\n    for model in range(2):\n        alias = model\n",
    "comprehension_shadow": "def unused():\n    return [model for model in range(2)]\n",
}
_READ = RM + '    return {"output": model(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")).data}\n'


class TestMutationScope36:
    @pytest.mark.parametrize("name", sorted(_NESTED))
    def test_a_nested_scope_binding_does_not_hide_the_module_escape(self, name):
        r = _refuses(H + _MODEL + "def mutate():\n" + _NESTED[name] + _MUTATE + _SIDE + _END, MODULE_REFUSAL,
                     "'model' escapes as a value in 'alias = model'")
        assert r["slow"] == {"value": {"output": 5}}, r["slow"]           # the mutation ran: Box.data; an earlier revision returned int(Box()) == 9

    @pytest.mark.parametrize("name", sorted(_SAFE))
    def test_a_real_nearer_binding_is_still_not_an_escape(self, name):
        _lowers(H + "def model(a):\n    return k(a)\n" + _SAFE[name] + _READ, [[2.0, 2.0]])

    def test_the_module_lambda_parameter_ruling_is_kept(self):
        # a module-level ``lambda model: model`` is a second binding site of 'model' in the REFUSAL inventory (the reviewer's explicit
        # conservative ruling, not relaxed): the call is refused although the lambda binds nothing in the module
        r = _refuses(H + "def model(a):\n    return k(a)\n" + "unused = lambda model: model\n" + _READ, "kernel_function_statement_unresolved:k: ",
                     "a lambda parameter")
        assert r["slow"] == {"value": {"output": [[2.0, 2.0]]}}, r["slow"]

    def test_the_two_inventories_differ_on_purpose(self):
        src = "def mutate():\n    unused = lambda model: model\n    other = [model for model in ()]\n    third = lambda: (model := 0)\n    alias = model\n"
        tree = ast.parse(src); fn = tree.body[0]
        prog = _Program(tree, set())
        kinds = sorted(k for k, _ in _BindingScope(fn, prog, "f").sites.get("model", ()))
        assert kinds == ["a comprehension target", "a lambda parameter", "stored"], kinds        # the refusal inventory charges them to mutate()
        locs, glob, nonl = _lexical_scope(fn)
        assert "model" not in locs and locs == {"unused", "other", "third", "alias"}, locs      # the lexical scope does not

    def test_lexical_scope_follows_python(self):
        src = ("import os as o\n"
               "def f(p, *a, q=1, **kw):\n"
               "    x = 1; y: int; (z := 2)\n"
               "    for i in (): pass\n"
               "    with open('f') as fh: pass\n"
               "    try: pass\n    except E as err: pass\n"
               "    import m.n\n"
               "    def g(): return model\n"
               "    class C: cls_local = 1\n"
               "    comp = [t for t in () if (w := t)]\n"
               "    lam = lambda u: (v := u)\n"
               "    global G\n    nonlocal N\n")
        fn = ast.parse(src).body[1]
        locs, glob, nonl = _lexical_scope(fn)
        assert locs == {"p", "a", "q", "kw", "x", "y", "z", "i", "fh", "err", "m", "g", "C", "comp", "w", "lam"}, locs
        assert glob == {"G"} and nonl == {"N"}
        assert "t" not in locs and "u" not in locs and "v" not in locs and "cls_local" not in locs and "model" not in locs
        comp = next(n for n in ast.walk(fn) if isinstance(n, ast.ListComp))
        assert _lexical_scope(comp)[0] == {"t"}
        cls = next(n for n in ast.walk(fn) if isinstance(n, ast.ClassDef))
        assert _lexical_scope(cls) == (frozenset(), frozenset(), frozenset())        # a class body never shadows here

    def test_load_resolution_cases(self):
        L = lambda *names: (frozenset(names), frozenset(), frozenset())
        G = lambda *names: (frozenset(), frozenset(names), frozenset())
        NL = lambda *names: (frozenset(), frozenset(), frozenset(names))
        assert _load_resolves_to_module([], "model") is True
        assert _load_resolves_to_module([L("model")], "model") is False               # a real local
        assert _load_resolves_to_module([L("other")], "model") is True
        assert _load_resolves_to_module([L("model"), L()], "model") is False          # free variable of the enclosing function
        assert _load_resolves_to_module([L("model"), G("model")], "model") is True    # declared global innermost
        assert _load_resolves_to_module([G("model"), L("model")], "model") is False   # the inner local wins over an outer global
        assert _load_resolves_to_module([L("model"), NL("model")], "model") is False  # nonlocal defers outward to the local
        assert _load_resolves_to_module([NL("model")], "model") is True               # nothing binds it: kept as an escape
        assert _load_resolves_to_module([object()], "model") is True                  # unreadable chain: kept as an escape

    def test_the_collector_records_the_review4_shapes_and_the_class_global(self):
        code = H + _MODEL + "def mutate():\n" + "".join(_NESTED.values()) + _MUTATE + \
               "def cg():\n    model = 0\n    class C:\n        global model\n        alias2 = model\n" + \
               "def real():\n    model = 0\n    alias3 = model\n" + _SIDE + _END
        tree = ast.parse(code)
        esc = _collect_escapes(tree, _Program(tree, {"k"}))
        assert [t for t, _ in esc["model"]] == ["alias = model", "alias2 = model"], esc["model"]
        assert "k" not in esc, esc

    def test_the_new_refusal_is_a_typed_decline(self):
        r = _run(H + _MODEL + "def mutate():\n" + _NESTED["lambda_parameter"] + _MUTATE + _SIDE + _END)
        assert r["reasons"] and is_typed_decline(r["reasons"][0]) and r["public"] == r["slow"] == {"value": {"output": 5}}, r


# ---------------------------------------------------------------------------------------------------------------------
# baseline-invalid-dtype-37
# ---------------------------------------------------------------------------------------------------------------------

DTYPE_REFUSAL = "kernel_invalid_dtype_literal:k: "


class TestBaselineInvalidDtype37:
    @pytest.mark.parametrize("dtype", ["__invalid_dtype__", "BF16", "FP32", "float32", " bf16"])
    def test_from_numpy_with_an_unknown_literal_dtype_is_refused(self, dtype):
        r = _refuses(H + RM + f'    return {{"output": Tensor.from_numpy(inputs["a"], "M,N", dtype={dtype!r}).data}}\n', DTYPE_REFUSAL,
                     f"passes dtype={dtype!r}, which is not one of the interpreter's dtypes")
        assert "error" in r["slow"], r["slow"]                 # the interpreter's Unknown dtype (ValueError, re-raised by execute_translated)

    def test_a_tensor_constructor_outside_a_kernel_is_judged_too(self):
        r = _refuses(H + RM + '    t = Tensor(M=1, N=2, dtype="Fp16", fill=1.0)\n    return {"output": t.data}\n', DTYPE_REFUSAL, "'Tensor' at line")
        assert "error" in r["slow"], r["slow"]

    def test_from_random_is_judged_too(self):
        r = _refuses(H + RM + '    t = Tensor.from_random(M=1, N=2, dtype="int8 ")\n    return {"output": t.data}\n', DTYPE_REFUSAL, "'Tensor.from_random' at line")
        assert "error" in r["slow"], r["slow"]

    def test_a_non_string_literal_dtype_is_judged_too(self):
        r = _refuses(H + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype=5).data}\n', DTYPE_REFUSAL, "passes dtype=5,")
        assert "error" in r["slow"], r["slow"]

    def test_a_kernel_body_allocation_keeps_its_literal_boundary_class(self):
        code = ('from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nimport numpy as np\n@kernel\n'
                'def k(A: Tensor("M,N", dtype="bf16")) -> Tensor("M,N", dtype="bf16"):\n    Y = Tensor(M=1, N=2, dtype="BF16", fill=2.0)\n    return Y\n'
                + RM + '    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n')
        _refuses(code, "kernel_output_dtype_unestablished:k:", "names dtype 'BF16', which is not a supported dtype")

    @pytest.mark.parametrize("dtype", sorted(VALID_DTYPES))
    def test_every_interpreter_dtype_is_still_accepted(self, dtype):
        r = _run(H + RM + f'    return {{"output": Tensor.from_numpy(inputs["a"], "M,N", dtype={dtype!r}).data}}\n')
        assert not any(x.startswith("kernel_invalid_dtype_literal") for x in r["reasons"]), r["reasons"]
        assert r["public"] == r["slow"] and "value" in r["slow"], (dtype, r["public"], r["slow"])

    def test_the_dtype_refusal_is_a_typed_decline(self):
        r = _run(H + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype="BF16").data}\n')
        assert r["reasons"] and is_typed_decline(r["reasons"][0]) and r["public"] == r["slow"], r


# ---------------------------------------------------------------------------------------------------------------------
# the reviewer's frozen round-11 files, byte-for-byte (vendored under review3_controls/)
# ---------------------------------------------------------------------------------------------------------------------

class TestReview4Round11FilesVerbatim:
    def test_escape_resolution_file(self, tmp_path):
        """``test_round11_escape_resolution.py`` (sha256 7e043132…, 6 shapes x 2 carriers): 12 failed on an earlier revision (fast 9 vs
        interpreter 5), 12 passed on an earlier revision; after finding 36 every case refuses typed with the interpreter's public value."""
        run, res = _run_reviewer_file(tmp_path, "test_round11_escape_resolution.py", extra=("test_round11_compatible.py", "test_round9_provenance.py"))
        assert res.stdout.splitlines()[-1].startswith("12 passed"), res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round11-compatible-head").glob("escape-*.json")}
        assert len(recs) == 12, len(recs)
        for name, r in recs.items():
            assert r["public"] == r["slow"] == {"value": {"output": 5}}, (name, r["public"], r["slow"])
            assert not r["lowered"] and "function_implementation_unproved" in r["reasons"][0] and "'model' escapes as a value in 'alias = model'" in r["reasons"][0], (name, r["reasons"])
            assert "model" in r["escapes"], (name, r["escapes"])

    def test_compatible_harness_file(self, tmp_path):
        """the reviewer's harness ``test_round11_compatible.py`` (sha256 6659508f…) run on its own: six class-global / function-global /
        nested-nonlocal mutation controls refuse safely (12), four function-level shadows lower (8), the four malformed-dtype cases
        are refusals after finding 37 (4); the two ``lambda_shadow`` cases (a MODULE-level ``lambda model: model``) stay failing
        under the reviewer's explicit conservative lambda-parameter ruling, which is not relaxed -- the exact deviation set is pinned."""
        run, res = _run_reviewer_file(tmp_path, "test_round11_compatible.py", extra=("test_round9_provenance.py",))
        failed = sorted(l.split("::", 1)[1].split(" ")[0] for l in res.stdout.splitlines() if l.startswith("FAILED "))
        assert failed == ["test_nearer_local_load_is_not_module_escape[lambda_shadow-False]", "test_nearer_local_load_is_not_module_escape[lambda_shadow-True]"], res.stdout[-3000:]
        assert res.stdout.splitlines()[-1].startswith("2 failed, 24 passed"), res.stdout[-500:]
        for p in (run / "runs" / "round11-compatible-head").glob("invalid-*.json"):
            r = json.loads(p.read_text()); assert "error" in r["slow"] and r["public"] == r["slow"] and not r["lowered"], (p.name, r)
