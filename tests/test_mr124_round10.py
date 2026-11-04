# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier fix round 10 -- the author's coverage of the reviewer's findings on an earlier revision (an earlier review.md), equivalent to
the reviewer's frozen ``test_round9_provenance.py`` and ``test_round9_agent.py`` (both also run verbatim below).

Every probe compares the FAST public path (``execute_translated(fast=True)``) with the interpreter and requires either a
correct lowering or a TYPED refusal; no program may return a value the interpreter does not.  ``Box`` has ``.data == 5``
and ``int(Box()) == 9``: a program that strips a ``.data`` read on a guess returns 9 where the interpreter returns 5.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import numba_lower, execute_fast
from solar.einsum_lib.verify import execute_translated

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "tests" / "fixtures" / "lowering_final_rounding" / "review3_controls"

H = '''from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
import numpy as np
class Box:
    data = 5
    def __int__(self):
        return 9
@kernel
def k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N", dtype="fp32"):
    Y = Tensor(M=1, N=2, dtype="fp32", fill=2.0)
    return Y
'''
END = '''def run_model(inputs):
    clear_dims()
    return {"output": np.array(SIDE)}
'''
INPUTS = {"a": np.ones((1, 2))}


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def _outcome(fn):
    try:
        return {"value": {k: np.asarray(v).tolist() for k, v in fn().items()}}
    except Exception as e:                      # the interpreter's own error is a legitimate public outcome
        return {"error": type(e).__name__}
    finally:
        clear_dims(); stop_trace()


def _run(code, fp64=True):
    compile(code, "<probe>", "exec")
    reasons, fallback = [], []
    low = numba_lower(code, use_fp64=fp64, boundscheck=True, on_decline=reasons.append)
    slow = _outcome(lambda: execute_translated(code, {}, all_inputs=INPUTS, fast=False)[0])
    raw = _outcome(lambda: execute_fast(code, INPUTS, use_fp64=fp64, on_fallback=fallback.append))
    public = _outcome(lambda: execute_translated(code, {}, all_inputs=INPUTS, fast=True, use_fp64=fp64)[0])
    return dict(lowered=low, reasons=reasons, fallback=fallback, slow=slow, raw=raw, public=public)


def _refuses(code, cls, needle="", scope="function"):
    """The program is a TYPED refusal of class ``cls`` (reason text contains ``needle``) and its public value is the interpreter's."""
    for fp64 in (False, True):
        r = _run(code, fp64)
        assert r["lowered"] is None, (fp64, r["lowered"])
        assert r["reasons"], (fp64, r)
        head = f"kernel_{scope}_statement_unresolved:k: {cls}:"
        assert r["reasons"][0].startswith(head), (fp64, head, r["reasons"][0][:300])
        assert needle in r["reasons"][0], (fp64, needle, r["reasons"][0][:400])
        assert r["public"] == r["slow"], (fp64, r["public"], r["slow"])
    return r


def _lowers(code, value, fp64=True):
    """The program lowers, declines nothing, and the fast path returns ``value`` == the interpreter (``value=None``: the three
    paths must agree on some value)."""
    r = _run(code, fp64)
    assert r["lowered"] is not None and not r["reasons"] and not r["fallback"], r
    assert r["raw"] == r["slow"] == r["public"] and "value" in r["slow"], r
    if value is not None:
        assert r["slow"] == {"value": {"output": value}}, r
    return r


# ---------------------------------------------------------------------------------------------------------------------
# kernel-call-binding-27: the callee of a kernel call is RESOLVED, never accepted by spelling
# ---------------------------------------------------------------------------------------------------------------------

_K27 = {
    "local_lambda_in_run_model": '''def run_model(inputs):
    clear_dims()
    k = lambda a: Box()
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
    "parameter_of_plain_function": '''def model(a, k):
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"], lambda a: Box())
    return {"output": np.array(int(out.data))}
''',
    "default_parameter_of_plain_function": '''def model(a, k=lambda a: Box()):
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": np.array(int(out.data))}
''',
    "local_of_plain_function": '''def model(a):
    k = lambda a: Box()
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": np.array(int(out.data))}
''',
    "class_body_binding": '''class C:
    k = lambda a: Box()
    SIDE = int(k(np.ones((1,2))).data)
SIDE = C.SIDE
''' + END,
    "comprehension_target_in_run_model": '''def run_model(inputs):
    clear_dims()
    ks = [k for k in (lambda a: Box(),)]
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
    "lambda_parameter_in_run_model": '''def run_model(inputs):
    clear_dims()
    f = lambda k: k
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
    "global_declaration_in_run_model": '''def run_model(inputs):
    global k
    clear_dims()
    return {"output": k(inputs["a"]).data}
''',
    "nested_def_in_run_model": '''def run_model(inputs):
    clear_dims()
    def k(a):
        return Box()
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
    "import_in_run_model": '''def run_model(inputs):
    clear_dims()
    from math import exp as k
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
}


class TestKernelCallBinding27:
    """A kernel SPELLING with a nearer binding names an ordinary callable: typed refusal ``kernel_name_shadowed`` (a direct
    read) or the shadow surfaces inside rule (iii) (``out = model(...)`` where ``model`` returns ``k(a)`` with ``k`` a parameter /
    default / local); the public value is the interpreter's (5), never 9."""

    @pytest.mark.parametrize("case", ["local_lambda_in_run_model", "comprehension_target_in_run_model", "lambda_parameter_in_run_model",
                                      "global_declaration_in_run_model", "nested_def_in_run_model", "import_in_run_model"])
    def test_a_nearer_binding_in_run_model_refuses_the_direct_read(self, case):
        _refuses(H + _K27[case], "kernel_name_shadowed", "the @kernel name 'k' is bound in the function 'run_model'")

    def test_a_class_body_binding_refuses_the_class_body_read(self):
        r = _refuses(H + _K27["class_body_binding"], "kernel_name_shadowed", "the @kernel name 'k' is bound in the body of the class 'C' (stored at line", scope="module")
        assert r["slow"] == {"value": {"output": 5}}

    @pytest.mark.parametrize("case", ["parameter_of_plain_function", "default_parameter_of_plain_function", "local_of_plain_function"])
    def test_the_same_resolution_applies_inside_rule_iii(self, case):
        r = _refuses(H + _K27[case], "name_bound_to_non_kernel_value", "the @kernel name 'k' is bound in the function 'model' (")
        assert r["slow"] == {"value": {"output": 5}}

    def test_the_detail_names_the_kind_of_the_nearer_binding(self):
        kinds = {"parameter_of_plain_function": "a parameter at line", "default_parameter_of_plain_function": "a parameter at line",
                 "local_of_plain_function": "stored at line", "local_lambda_in_run_model": "stored at line",
                 "comprehension_target_in_run_model": "a comprehension target at line", "lambda_parameter_in_run_model": "a lambda parameter at line",
                 "global_declaration_in_run_model": "declared global at line", "nested_def_in_run_model": "defined at line", "import_in_run_model": "imported at line"}
        for case, kind in kinds.items():
            reasons = []
            assert numba_lower(H + _K27[case], use_fp64=True, boundscheck=True, on_decline=reasons.append) is None
            assert kind in reasons[0], (case, kind, reasons[0][:300])

    def test_a_shadow_in_another_function_does_not_reach_run_model(self):
        code = H + '''def other(a):
    k = lambda a: Box()
    return k(a)
def run_model(inputs):
    clear_dims()
    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")).data}
'''
        _lowers(code, [[2.0, 2.0]])

    def test_the_resolved_kernel_call_still_lowers(self):
        code = H + '''def model(a):
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32"))
    return {"output": out.data}
'''
        _lowers(code, [[2.0, 2.0]]); _lowers(code, [[2.0, 2.0]], fp64=False)

    def test_a_module_level_rebinding_keeps_its_own_class(self):
        code = H + '''k = k
def run_model(inputs):
    clear_dims()
    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")).data}
'''
        _refuses(code, "kernel_name_rebound", "has more than its own definition as a binding site in the module")


# ---------------------------------------------------------------------------------------------------------------------
# function-code-mutation-28: a visibly mutated callable has no proved implementation
# ---------------------------------------------------------------------------------------------------------------------

_M28_MODEL = '''def model(a):
    return k(a)
def replacement(a):
    return Box()
'''
_M28 = {
    "code_replaced": _M28_MODEL + "model.__code__ = replacement.__code__\nSIDE = int(model(np.ones((1,2))).data)\n",
    "defaults_replaced": _M28_MODEL + "model.__defaults__ = (1,)\nSIDE = int(model(np.ones((1,2))).data)\n",
    "wrapped_set": _M28_MODEL + "model.__wrapped__ = replacement\nSIDE = int(model(np.ones((1,2))).data)\n",
    "dict_written": _M28_MODEL + "model.__dict__['note'] = 1\nSIDE = int(model(np.ones((1,2))).data)\n",
    "globals_of_function_written": _M28_MODEL + "model.__globals__['k'] = lambda a: Box()\nSIDE = int(model(np.ones((1,2))).data)\n",
    "setattr_code": _M28_MODEL + "setattr(model, '__code__', replacement.__code__)\nSIDE = int(model(np.ones((1,2))).data)\n",
    "delattr_doc": _M28_MODEL + "delattr(model, '__doc__')\nSIDE = int(model(np.ones((1,2))).data)\n",
    "globals_dict_literal_key": _M28_MODEL + "globals()['model'] = replacement\nSIDE = int(model(np.ones((1,2))).data)\n",
    "globals_update_wildcard": _M28_MODEL + "globals().update(model=replacement)\nSIDE = int(model(np.ones((1,2))).data)\n",
    "vars_dict_dynamic_key": _M28_MODEL + "NAME = 'model'\nvars()[NAME] = replacement\nSIDE = int(model(np.ones((1,2))).data)\n",
    "setattr_on_non_name": _M28_MODEL + "import sys\nsetattr(sys.modules[__name__], 'model', replacement)\nSIDE = int(model(np.ones((1,2))).data)\n",
    "mutation_inside_a_function": _M28_MODEL + "def swap():\n    model.__code__ = replacement.__code__\nswap()\nSIDE = int(model(np.ones((1,2))).data)\n",
}


_M28_ATTRIBUTE_STORES = ("code_replaced", "defaults_replaced", "wrapped_set", "mutation_inside_a_function")


class TestFunctionCodeMutation28:
    """``model.__code__ = replacement.__code__`` keeps ``model`` a sole binding while changing what runs: rule (iii) must not use
    the original return AST as proof.  An attribute store on the callee -- anywhere, a nested function included -- is the typed
    refusal ``function_implementation_unproved``; the other visible mutations (``setattr`` / ``delattr``, ``globals()`` /
    ``vars()`` writes, ``__dict__`` / ``__globals__`` references) are already typed refusals of the round-5 module-effects
    prescan (``kernel_allocation_callee_unresolved: module_binding_uncertain``) and never reach rule (iii); the collector
    records them all the same (unit-tested below).  The public value is always the interpreter's, never 9."""

    @pytest.mark.parametrize("case", sorted(_M28))
    def test_every_visible_mutation_is_a_typed_refusal_with_the_interpreter_value(self, case):
        for fp64 in (False, True):
            r = _run(H + _M28[case] + END, fp64)
            assert r["lowered"] is None and r["reasons"] and r["reasons"][0].startswith("kernel_"), (case, fp64, r["reasons"][:1], r["lowered"])
            assert r["public"] == r["slow"], (case, fp64, r["public"], r["slow"])
            assert "error" in r["slow"] or r["slow"]["value"]["output"] != 9 or r["public"] == r["slow"]

    @pytest.mark.parametrize("case", _M28_ATTRIBUTE_STORES)
    def test_an_attribute_store_on_the_callee_is_function_implementation_unproved(self, case):
        _refuses(H + _M28[case] + END, "function_implementation_unproved", "'model' is visibly mutated by '", scope="module")

    def test_the_reviewer_reproducer_returns_5_on_both_paths(self):
        r = _refuses(H + _M28["code_replaced"] + END, "function_implementation_unproved",
                     "'model' is visibly mutated by 'model.__code__ = replacement.__code__' at line 15, so the analysed definition of 'model' is not proved to be its implementation", scope="module")
        assert r["slow"] == {"value": {"output": 5}} == r["public"]

    def test_a_mutated_kernel_callee_refuses_too(self):
        code = H + "k.__code__ = (lambda A: Box()).__code__\n" + '''def run_model(inputs):
    clear_dims()
    return {"output": np.array(int(k(inputs["a"]).data))}
'''
        _refuses(code, "function_implementation_unproved", "'k' is visibly mutated by 'k.__code__ = (lambda A: Box()).__code__' at line 11")

    def test_the_collector_records_every_listed_form(self):
        import ast
        from solar.einsum_lib.numba_accel import _collect_mutations
        src = ("def f(a):\n    return a\n" "f.__code__ = f.__code__\n" "f.__defaults__ = (1,)\n" "del f.__wrapped__\n" "f.__dict__['x'] = 1\n"
               "f.__globals__['k'] = 1\n" "setattr(f, 'a', 1)\n" "delattr(f, 'a')\n" "globals()['f'] = 1\n" "del vars()['g']\n" "NAME = 'f'\n"
               "vars()[NAME] = 1\n" "globals().update(f=1)\n" "import sys\n" "setattr(sys.modules[__name__], 'f', 1)\n" "def inner():\n    f.__code__ = f.__code__\n"
               "box = {}\n" "box['x'] = 1\n")
        m = _collect_mutations(ast.parse(src))
        assert len(m["f"]) == 9 and [l for _, l in m["f"]] == [3, 4, 5, 6, 7, 8, 9, 10, 18], m["f"]
        assert m["g"] == [("del vars()['g']", 11)]
        assert "*" not in m, m          # namespace-dictionary CALLS and setattr on a non-name target belong to the round-5 effects analysis
        assert m["f"][-1] == ("f.__code__ = f.__code__", 18) and m["box"] == [("box['x'] = 1", 20)]
        assert "sys" not in m and "NAME" not in m

    def test_an_ordinary_container_store_is_not_a_callee_mutation(self):
        code = H + '''def model(a):
    return k(a)
def run_model(inputs):
    clear_dims()
    box = {}
    box["n"] = 1
    out = model(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32"))
    return {"output": out.data}
'''
        _lowers(code, [[2.0, 2.0]])

    def test_a_mutation_of_another_function_does_not_touch_the_callee(self):
        code = H + '''def model(a):
    return k(a)
def other(a):
    return a
other.__doc__ = "x"
def run_model(inputs):
    clear_dims()
    out = model(Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32"))
    return {"output": out.data}
'''
        _lowers(code, [[2.0, 2.0]])


# ---------------------------------------------------------------------------------------------------------------------
# builtin-result-29: ruling (i') -- a lowered DSL builtin's result is established, binding and call proved
# ---------------------------------------------------------------------------------------------------------------------

_HB = '''from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask, top_k_mask, gather, concat, stack, slice_at
import numpy as np
class Box:
    data = 5
    def __int__(self):
        return 9
@kernel
def k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N", dtype="fp32"):
    Y = Tensor(M=1, N=2, dtype="fp32", fill=2.0)
    return Y
'''
_RUN_A = '''def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")
'''


def _b(body, prefix=""):
    return _HB + prefix + _RUN_A + body


_B29_LOWER = {
    "apply_mask_direct": (_b('    return {"output": apply_mask(k(a), a, masked_value=0.0).data}\n'), [[2.0, 2.0]]),
    "apply_mask_named": (_b('    m = apply_mask(k(a), a, masked_value=0.0)\n    return {"output": m.data}\n'), [[2.0, 2.0]]),
    "apply_mask_positional_value": (_b('    return {"output": apply_mask(k(a), a, 0.0).data}\n'), [[2.0, 2.0]]),
    "apply_mask_through_plain_function": (_b('    out = model(a)\n    return {"output": out.data}\n', prefix='def model(x):\n    s = k(x)\n    return apply_mask(s, x, masked_value=-1e9)\n'), [[2.0, 2.0]]),
    "top_k_mask_default_axis": (_b('    return {"output": top_k_mask(k(a), 1).data}\n'), None),         # a tie: both paths must agree, whichever index wins
    "top_k_mask_int_axis": (_b('    return {"output": top_k_mask(k(a), 2, axis=1).data}\n'), [[1.0, 1.0]]),
    "gather_resolved_axis": (_b('    t = k(a)\n    g = gather(t, a, "N")\n    return {"output": g.data}\n'), [[[2.0, 2.0]]]),
    "concat_resolved_axis": (_b('    t = k(a)\n    c = concat([t, t], "N")\n    return {"output": c.data}\n'), [[2.0, 2.0, 2.0, 2.0]]),
    "stack_always": (_b('    t = k(a)\n    s = stack([t, t], "P")\n    return {"output": s.data}\n'), [[[2.0, 2.0]], [[2.0, 2.0]]]),
    "slice_at_resolved_axis": (_b('    t = k(a)\n    s = slice_at(t, "N", 0)\n    return {"output": s.data}\n'), [2.0]),
    "from_numpy_result": (_b('    return {"output": a.data}\n'), [[1.0, 1.0]]),
    "from_numpy_direct": (_b('    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32").data}\n'), [[1.0, 1.0]]),
    "unpacked_builtin_display": (_b('    p, q = apply_mask(k(a), a, masked_value=0.0), k(a)\n    return {"output": p.data}\n'), [[2.0, 2.0]]),
}

_B29_REFUSE = {
    "local_shadow_of_the_builtin": (_b('    apply_mask = lambda t, m, masked_value=0: Box()\n    return {"output": np.array(int(apply_mask(k(a), a).data))}\n'),
                                    "call_result", "is spelled like the DSL builtin apply_mask, but 'apply_mask' is bound in the function 'run_model' (stored at line"),
    "parameter_shadow_in_plain_function": (_b('    out = model(a, lambda t, m: Box())\n    return {"output": np.array(int(out.data))}\n', prefix='def model(x, apply_mask):\n    return apply_mask(k(x), x)\n'),
                                           "name_bound_to_non_kernel_value", "'apply_mask' is bound in the function 'model' (a parameter at line"),
    "module_rebinding_of_the_builtin": (_b('    return {"output": np.array(int(apply_mask(k(a), a).data))}\n', prefix='apply_mask = lambda t, m, masked_value=0: Box()\n'),
                                        "call_result", "the module binds 'apply_mask' imported at line 1, stored at line 11, not only by imports of solar.einsum_lib"),
    "import_identity_mismatch": (_HB.replace(", apply_mask,", ",") + "from solar.einsum_lib import Dim as apply_mask\n" + _RUN_A + '    return {"output": apply_mask(k(a), a).data}\n',
                                 "call_result", "but its binding is not the DSL object: the import at line 11"),
    "plain_import_of_the_package": (_HB.replace(", apply_mask,", ",") + "import apply_mask\n" + _RUN_A + '    return {"output": apply_mask(k(a), a).data}\n',
                                    "call_result", "is bound by a plain import at line 11"),
    "mutated_builtin": (_b('    return {"output": apply_mask(k(a), a).data}\n', prefix='apply_mask.note = 1\n'),
                        "function_implementation_unproved", "'apply_mask' is visibly mutated by 'apply_mask.note = 1' at line 11"),
    "apply_mask_wrong_arity": (_b('    return {"output": apply_mask(k(a)).data}\n'), "call_result", "whose form the lowering does not rewrite (apply_mask takes (tensor, mask[, masked_value]))"),
    "apply_mask_star_args": (_b('    args = (k(a), a)\n    return {"output": apply_mask(*args).data}\n'), "call_result", "whose form the lowering does not rewrite (the call passes *args / **kwargs)"),
    "top_k_mask_dim_name_axis": (_b('    return {"output": top_k_mask(k(a), 1, "N").data}\n'), "call_result", "top_k_mask is dispatched on raw arrays with an INTEGER axis only"),
    "gather_unresolved_axis": (_b('    g = gather(inputs["a"], a, "N")\n    return {"output": g.data}\n'), "name_bound_to_non_kernel_value", "whose axis the lowering could not resolve"),
    "concat_unresolved_axis": (_b('    c = concat([inputs["a"], inputs["a"]], "N")\n    return {"output": c.data}\n'), "name_bound_to_non_kernel_value", "whose axis the lowering could not resolve"),
    "slice_at_unresolved_axis": (_b('    s = slice_at(inputs["a"], "N", 0)\n    return {"output": s.data}\n'), "name_bound_to_non_kernel_value", "whose axis the lowering could not resolve"),
    "gather_unresolved_axis_direct": (_b('    return {"output": gather(inputs["a"], a, "N").data}\n'), "call_result", "whose axis the lowering could not resolve"),
    "module_level_builtin_is_not_rewritten": (_HB + 'S = apply_mask(k(Tensor.from_numpy(np.ones((1, 2)), "M,N", dtype="fp32")), Tensor.from_numpy(np.ones((1, 2)), "M,N", dtype="fp32")).data\n' + _RUN_A + '    return {"output": S}\n',
                                              "call_result", "that the lowering had not rewritten before this read (a module-level statement is never rewritten"),
    "helper_defined_after_its_reader": (_HB + _RUN_A + '    out = model(a)\n    return {"output": out.data}\n' + 'def model(x):\n    return apply_mask(k(x), x, masked_value=0.0)\n',
                                        "name_bound_to_non_kernel_value", "that the lowering had not rewritten before this read"),
}


class TestBuiltinResult29:
    """(i'): the immediate result of apply_mask / top_k_mask / gather / concat / stack / slice_at / Tensor.from_numpy is
    established when the callee binding is proved (no nearer binding, identity-verified import, not mutated), the call matches
    the supported signature and the lowering actually rewrote / dispatched it; every other form keeps a typed refusal whose
    detail names the unmet condition."""

    @pytest.mark.parametrize("case", sorted(_B29_LOWER))
    def test_a_proved_builtin_result_lowers_with_the_interpreter_value(self, case):
        code, value = _B29_LOWER[case]
        _lowers(code, value); _lowers(code, value, fp64=False)

    @pytest.mark.parametrize("case", sorted(_B29_REFUSE))
    def test_an_unproved_builtin_result_stays_a_typed_refusal(self, case):
        code, cls, needle = _B29_REFUSE[case]
        scope = "module" if case == "module_level_builtin_is_not_rewritten" else "function"
        _refuses(code, cls, needle, scope=scope)

    def test_the_apply_mask_regression_module_shape_lowers_to_np_where(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_numba_accel import APPLY_MASK_CODE
        low = numba_lower(APPLY_MASK_CODE, use_fp64=True, boundscheck=True)
        assert low is not None and "np.where" in low and "apply_mask" not in low and ".data" not in low

    def test_rewrites_are_recorded_by_position_in_the_pristine_program(self):
        import ast
        from solar.einsum_lib.numba_accel import _Program
        code = _B29_LOWER["apply_mask_named"][0]
        reasons = []
        assert numba_lower(code, use_fp64=True, boundscheck=True, on_decline=reasons.append) is not None and not reasons
        tree = ast.parse(code)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "apply_mask"]
        assert len(calls) == 1 and _Program._key(calls[0])[1:] == (calls[0].lineno, calls[0].col_offset, calls[0].end_lineno, calls[0].end_col_offset)


# ---------------------------------------------------------------------------------------------------------------------
# the reviewer's frozen round-9 files, byte-for-byte (vendored under review3_controls/)
# ---------------------------------------------------------------------------------------------------------------------

def _run_reviewer_file(tmp_path, name, extra=()):
    run = tmp_path / "run"; run.mkdir()
    for sub in ("runs", "evidence", "probe_modules"):
        (run / sub).mkdir()
    for f in (name,) + tuple(extra):
        (run / f).write_bytes((R3 / f).read_bytes())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "tests") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
    res = subprocess.run([sys.executable, "-m", "pytest", name, "-q", "-p", "no:cacheprovider", "-rf"],
                         cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
    return run, res


class TestReview4Round9ProvenanceVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round9_provenance.py`` (sha256 394b2979…, 13 cases x 2 carriers): 14 passed / 12 failed on an earlier revision,
        10 / 16 on an earlier revision, 0 / 26 on the baseline revision (the review's matrix); after findings 27 and 28 every record is a typed refusal
        (``kernel_*``) with the interpreter's public value -- 26 passed."""
        run, res = _run_reviewer_file(tmp_path, "test_round9_provenance.py")
        assert res.stdout.splitlines()[-1].startswith("26 passed"), res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round9-head").glob("*.json")}
        assert len(recs) == 26, len(recs)
        classes = {}
        for name, r in recs.items():
            assert not r["lowered"] and r["reasons"] and r["reasons"][0].startswith("kernel_"), (name, r["reasons"])
            assert r["public"] == r["slow"], (name, r["public"], r["slow"])
            classes[name.rsplit("-", 1)[0]] = r["reasons"][0].split(": ")[1]
        assert classes["kernel_local_shadow"] == classes["kernel_class_shadow"] == "kernel_name_shadowed"
        assert classes["plain_code_replaced"] == "function_implementation_unproved"
        assert {classes[c] for c in ("kernel_parameter_return", "kernel_default_return", "kernel_local_return")} == {"name_bound_to_non_kernel_value"}


class TestReview4Round9AgentVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round9_agent.py`` (sha256 9feac14c…; it imports tests/test_translation_agent.py): 4 passed / 4 failed on an earlier revision;
        after finding 30 (keyword-only flag) and finding 31 (typed-decline-only routing) all 8 pass -- the three untyped failures keep
        ``execution_failed`` under BOTH flag values, the old positional contract binds, the typed decline keeps the existing isolation."""
        run, res = _run_reviewer_file(tmp_path, "test_round9_agent.py")
        assert res.stdout.splitlines()[-1].startswith("8 passed"), res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "agent-boundary-head").glob("*.json")}
        assert {k: v["status"] for k, v in recs.items() if k.startswith("untyped-")} and all(v["status"] == "execution_failed" for k, v in recs.items() if k.startswith("untyped-"))
        assert recs["positional-contract"]["source_type"] == "pytorch" and recs["positional-contract"]["mem_budget_bytes"] == 123456789
        assert recs["isolation"]["verified"] and [x["function"] for x in recs["isolation"]["calls"]] == ["_isolated_execute_fast", "_isolated_run_model"]
