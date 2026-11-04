# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier fix round 11 -- the author's coverage of the reviewer's findings on an earlier revision (an earlier review.md), equivalent to
the reviewer's frozen ``test_round10_edges.py`` (also run verbatim below, 40 cases):

* builtin-narrow-32: a DSL builtin's result is established only when its rewrite reproduces the interpreter's
  declared-dtype rounding (expression-position ``Tensor.from_numpy`` rounded inline; ``apply_mask`` rounded to a proved
  operand dtype; ``concat`` / ``stack`` need one proved dtype; a dynamic dtype is not established);
* mutation-aliases-33: a callee that escapes as a value (alias, alias chain, call argument, ``type(f)``) has no proved
  implementation -- ``function_implementation_unproved``;
* readless-callee-34: the call-time rounding wrap of a kernel result is placed only when the callee RESOLVES to the
  module-level ``@kernel``; a module rebinding / mutation / escape of a kernel name refuses the whole program.

Every probe compares the FAST public path with the interpreter and requires a correct lowering or a TYPED refusal.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import (numba_lower, execute_fast, is_typed_decline, _Program, _collect_escapes,
                                          _dual_representable)
from solar.einsum_lib.verify import execute_translated

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "tests" / "fixtures" / "lowering_final_rounding" / "review3_controls"

H = '''from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask, top_k_mask, gather, concat, stack, slice_at
import numpy as np
class Box:
    data = 5
    def __int__(self):
        return 9
@kernel
def k(A: Tensor("M,N", dtype="bf16")) -> Tensor("M,N", dtype="bf16"):
    Y = Tensor(M=1, N=2, dtype="bf16", fill=2.0)
    return Y
@kernel
def kw(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N", dtype="fp32"):
    Y = Tensor(M=1, N=2, dtype="fp32", fill=2.0)
    return Y
'''
RM = 'def run_model(inputs):\n    clear_dims()\n'
MASK = '    m = Tensor.from_numpy(np.array([[1.0, 0.0]]), "M,N", dtype="fp32")\n'
INPUTS = {"a": np.array([[1.003, 1.003]], dtype=np.float32)}
F32 = 1.003000020980835           # float32(1.003) as a float64
BF16, FP16 = 1.0, 1.0029296875    # bf16(1.003), fp16(1.003)
BF16_NEG1E9 = -998244352.0        # bf16(-1e9), what the interpreter stores for apply_mask's default masked value


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


def _refuses(code, head, needle="", carriers=(False, True)):
    """A TYPED refusal whose first reason starts with ``head`` and contains ``needle``; the public value is the interpreter's."""
    for fp64 in carriers:
        r = _run(code, fp64)
        assert r["lowered"] is None, (fp64, r["lowered"])
        assert r["reasons"] and r["reasons"][0].startswith(head), (fp64, head, r["reasons"][:1])
        assert needle in r["reasons"][0], (fp64, needle, r["reasons"][0][:400])
        assert is_typed_decline(r["reasons"][0]), r["reasons"][0][:80]
        assert r["public"] == r["slow"], (fp64, r["public"], r["slow"])
    return r


def _lowers(code, value, carriers=(False, True), contains=(), absent=()):
    """The program lowers, declines nothing, raw == public == interpreter (== ``value`` unless None); the lowered text contains /
    lacks the given fragments."""
    for fp64 in carriers:
        r = _run(code, fp64)
        assert r["lowered"] is not None and not r["reasons"] and not r["fallback"], (fp64, r["reasons"], r["fallback"])
        assert r["raw"] == r["slow"] == r["public"] and "value" in r["slow"], (fp64, r)
        if value is not None:
            assert r["slow"] == {"value": {"output": value}}, (fp64, r["slow"])
        for frag in contains:
            assert frag in r["lowered"], (fp64, frag, r["lowered"])
        for frag in absent:
            assert frag not in r["lowered"], (fp64, frag, r["lowered"])
    return r


FUNC = "kernel_function_statement_unresolved:k: "

# ---------------------------------------------------------------------------------------------------------------------
# builtin-narrow-32
# ---------------------------------------------------------------------------------------------------------------------

class TestBuiltinNarrow32:
    def test_an_expression_position_from_numpy_is_rounded_inline_bf16(self):
        _lowers(H + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16").data}\n', [[BF16, BF16]],
                contains=("_truncate_bf16(np.asarray(",))

    def test_an_expression_position_from_numpy_is_rounded_inline_fp16(self):
        _lowers(H + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype="fp16").data}\n', [[FP16, FP16]],
                contains=("_truncate_fp16(np.asarray(",))

    def test_a_wide_from_numpy_in_expression_position_is_not_rounded(self):
        _lowers(H + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N").data}\n', [[F32, F32]], absent=("_truncate",))

    def test_nested_inside_apply_mask_the_from_numpy_is_rounded(self):
        code = H + RM + '    a = Tensor.from_numpy(np.ones((1, 2)), "M,N", dtype="fp32")\n' \
                        '    return {"output": apply_mask(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16"), a, 0).data}\n'
        _lowers(code, [[BF16, BF16]], contains=("_truncate_bf16(np.asarray(",))

    def test_passed_to_a_kernel_the_from_numpy_is_rounded_inline(self):
        _lowers(H + RM + '    return {"output": kw(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n', [[2.0, 2.0]],
                contains=("kw(_truncate_bf16(np.asarray(",))

    def test_the_assignment_form_keeps_its_appended_rounding_statement(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n    return {"output": x.data}\n'
        _lowers(code, [[BF16, BF16]], contains=("x = _truncate_bf16(x)",), absent=("_truncate_bf16(np.asarray(",))

    def test_apply_mask_on_a_narrow_operand_rounds_its_default_masked_value(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n' + MASK + '    return {"output": apply_mask(x, m).data}\n'
        _lowers(code, [[BF16, BF16_NEG1E9]], contains=("_truncate_bf16(np.where(",))

    def test_apply_mask_on_an_fp16_operand_rounds_a_literal_masked_value(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="fp16")\n' + MASK + '    return {"output": apply_mask(x, m, 1.003).data}\n'
        _lowers(code, [[FP16, FP16]], contains=("_truncate_fp16(np.where(",))

    def test_apply_mask_on_a_resolved_kernel_result_rounds_to_the_kernel_dtype(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n' + MASK + '    return {"output": apply_mask(k(x), m).data}\n'
        _lowers(code, [[2.0, BF16_NEG1E9]], contains=("_truncate_bf16(np.where(m > 0.5, _truncate_bf16(k(x)),",))

    def test_apply_mask_on_a_proved_wide_operand_is_not_rounded(self):
        # fp64 carrier only: the fp32 carrier cannot hold the float64 literal 1.003 (carrier precision, not narrow rounding)
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N")\n' + MASK + '    return {"output": apply_mask(x, m, 1.003).data}\n'
        _lowers(code, [[F32, 1.003]], carriers=(True,), absent=("_truncate",))
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N")\n' + MASK + '    return {"output": apply_mask(kw(x), m, 1.003).data}\n'
        _lowers(code, [[2.0, 1.003]], carriers=(True,), absent=("_truncate_bf16(np.where", "_truncate_fp16(np.where"))

    def test_apply_mask_with_an_unproved_operand_dtype_refuses_the_default_masked_value(self):
        code = H + 'def helper(a):\n    return Tensor.from_numpy(a, "M,N", dtype="bf16")\n' + RM + '    x = helper(inputs["a"])\n' + MASK + \
               '    return {"output": apply_mask(x, m).data}\n'
        r = _refuses(code, FUNC + "call_result:", "whose first argument's declared dtype is not established")
        assert "masked value -1e9 (the default)" in r["reasons"][0]
        assert r["slow"] == {"value": {"output": [[BF16, BF16_NEG1E9]]}}

    def test_apply_mask_with_an_unproved_operand_dtype_lowers_a_dual_representable_masked_value(self):
        code = H + 'def helper(a):\n    return Tensor.from_numpy(a, "M,N", dtype="bf16")\n' + RM + '    x = helper(inputs["a"])\n' + MASK + \
               '    return {"output": apply_mask(x, m, 0.0).data}\n'
        _lowers(code, [[BF16, 0.0]], absent=("_truncate_bf16(np.where",))

    def test_a_dynamic_from_numpy_dtype_is_not_established(self):
        code = H + 'DT = "bf16"\n' + RM + '    return {"output": Tensor.from_numpy(inputs["a"], "M,N", dtype=DT).data}\n'
        _refuses(code, FUNC + "call_result:", "whose dtype is not a literal string")

    def test_concat_of_one_proved_dtype_lowers(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n    y = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n' \
                        '    return {"output": concat([x, y], "N").data}\n'
        _lowers(code, [[BF16] * 4])

    def test_concat_of_mixed_dtypes_is_not_established(self):
        code = H + RM + '    x = Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")\n    y = Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")\n' \
                        '    return {"output": concat([x, y], "N").data}\n'
        _refuses(code, FUNC + "call_result:", "not all established as one dtype")

    def test_stack_of_unproved_dtypes_is_not_established(self):
        code = H + 'def helper(a):\n    return Tensor.from_numpy(a, "M,N", dtype="bf16")\n' + RM + '    x = helper(inputs["a"])\n' \
                   '    return {"output": stack([x, x], "B").data}\n'
        _refuses(code, FUNC + "call_result:", "not all established as one dtype")

    def test_top_k_mask_on_an_unproved_operand_has_nothing_to_round(self):
        code = H + 'def helper(a):\n    return Tensor.from_numpy(a, "M,N", dtype="bf16")\n' + RM + '    x = helper(inputs["a"])\n' \
                   '    return {"output": top_k_mask(x, 1).data}\n'
        _lowers(code, None)

    def test_dual_representability(self):
        assert all(_dual_representable(v) for v in (0.0, 1.0, -1.0, 0.5, 2048.0, 4096.0, 255.0))
        # 65504.0 is the largest fp16 but not a bf16 value (bf16 rounds it to 65536.0); -1e9 is neither; 1.003 is neither
        assert not any(_dual_representable(v) for v in (-1e9, 1.003, 65504.0, 65520.0, float("inf"), float("nan"), None))


# ---------------------------------------------------------------------------------------------------------------------
# mutation-aliases-33
# ---------------------------------------------------------------------------------------------------------------------

_MODEL = 'def model(a):\n    return k(a)\ndef replacement(a):\n    return Box()\n'
_SIDE = 'SIDE = int(model(np.ones((1, 2))).data)\n'
_END = RM + '    return {"output": np.array(SIDE)}\n'
_M33 = {
    "alias_code": "alias = model\nalias.__code__ = replacement.__code__\n",
    "alias_chain_code": "alias = model\nother = alias\nother.__code__ = replacement.__code__\n",
    "parameter_code": "def change(f):\n    f.__code__ = replacement.__code__\nchange(model)\n",
    "type_setattr_code": 'type(model).__setattr__(model, "__code__", replacement.__code__)\n',
    "container_element": "fns = [model]\nfns[0].__code__ = replacement.__code__\n",
    "escape_in_a_body_that_never_runs": "def never():\n    return model\n",
    "escape_as_a_default": "def other(f=model):\n    return f\n",
}


class TestMutationAliases33:
    @pytest.mark.parametrize("name", sorted(_M33))
    def test_an_escaping_callee_has_no_proved_implementation(self, name):
        r = _refuses(H + _MODEL + _M33[name] + _SIDE + _END, "kernel_module_statement_unresolved:k: function_implementation_unproved:",
                     "escapes as a value")
        if name in ("alias_code", "alias_chain_code", "parameter_code", "type_setattr_code", "container_element"):
            assert r["slow"] == {"value": {"output": 5}}, r["slow"]          # the mutation ran: Box.data; a guess returns Box.__int__ == 9
        else:
            assert "error" in r["slow"], r["slow"]                             # not mutated: the kernel sees a raw array -- the interpreter's own error

    def test_a_callee_that_is_only_called_keeps_lowering(self):
        code = H + _MODEL + "def other(a):\n    return model(a)\n" + RM + \
               '    return {"output": model(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n'
        _lowers(code, [[2.0, 2.0]])

    def test_a_direct_mutation_keeps_its_round_10_wording(self):
        _refuses(H + _MODEL + "model.__code__ = replacement.__code__\n" + _SIDE + _END,
                 "kernel_module_statement_unresolved:k: function_implementation_unproved:", "is visibly mutated by")

    def test_an_escaping_kernel_name_is_not_a_resolved_callee_for_a_data_read(self):
        code = H + "g = k\n" + RM + '    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n'
        _refuses(code, FUNC + "function_implementation_unproved:", "'k' escapes as a value in 'g = k'")

    def test_the_collector_records_every_non_call_load_of_a_def(self):
        import ast
        tree = ast.parse(H + _MODEL + _M33["alias_chain_code"] + _M33["parameter_code"] + _M33["type_setattr_code"] + _SIDE + _END)
        esc = _collect_escapes(tree, _Program(tree, {"k"}))
        assert [t for t, _ in esc["model"]] == ["alias = model", "change(model)", "type(model).__setattr__(model, '__code__', replacement.__code__)",
                                               "type(model).__setattr__(model, '__code__', replacement.__code__)"], esc["model"]
        assert [t for t, _ in esc["replacement"]][:1] == ["other.__code__ = replacement.__code__"], esc["replacement"]   # its __code__ is read: an escape too
        assert "change" not in esc and "run_model" not in esc and "k" not in esc, esc      # only ever called

    # commit 5 (follow-up): a load counts only when it RESOLVES to the module def -- finding 27's rule applied to the loads
    def test_the_collector_resolves_a_load_by_scope_not_by_spelling(self):
        import ast
        src = (H + _MODEL
               + "def helper(model):\n    return model\n"                                   # a parameter: Python's local
               + "def loop():\n    for model in []:\n        print(model)\n"                  # a loop target: local
               + "def comp(fns):\n    return [model for model in fns]\n"                     # a comprehension target: its own
               + "def grab():\n    global model\n    alias = model\n"                        # declared global: the module name
               + "def outer():\n    model = 1\n    def inner():\n        return model\n    return inner\n"   # free variable of outer's local
               + "class C:\n    model = 1\n    alias = model\n"                              # a class body never shadows: recorded
               + "def first(fns):\n    return [f for f in [model]]\n"                       # the first iterable is read outside
               + "def deco(f):\n    return f\n@deco\ndef model2(model=model):\n    return model\n"   # a default is read outside
               + _SIDE + _END)
        tree = ast.parse(src)
        esc = _collect_escapes(tree, _Program(tree, {"k"}))
        assert [t for t, _ in esc["model"]] == ["alias = model", "alias = model", "return [f for f in [model]]",
                                               "@deco\ndef model2(model=model):\n    return model"], esc["model"]
        assert "deco" in esc and "helper" not in esc and "loop" not in esc and "comp" not in esc and "k" not in esc, esc

    def test_a_loop_target_spelled_like_the_kernel_is_not_an_escape_of_it(self):
        # the shape of all 298 programs the first round-11 census (an earlier revision vs an earlier revision) newly declined: the reviewer's extent probes
        code = ('from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nimport numpy as np\n'
                '@kernel\ndef k(X: Tensor("M,N", dtype="bf16")) -> Tensor("M,N", dtype="bf16"):\n'
                '    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="bf16")\n'
                '    for m in Dim("M"):\n        for k in Dim("N"):\n            Y[m, k] = X[m, k] * 2.0\n    return Y\n'
                + RM + '    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n')
        _lowers(code, [[2.0, 2.0]])

    def test_a_parameter_spelled_like_the_callee_in_another_function_is_not_an_escape(self):
        code = H + _MODEL + "def helper(model):\n    return model\ndef comp(fns):\n    return [model for model in fns]\n" + RM + \
               '    return {"output": model(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n'
        _lowers(code, [[2.0, 2.0]])

    def test_a_global_declaration_is_a_second_module_binding_site_first(self):
        # the collector records the load (test above); end to end the module-binding check of round 8 speaks first
        _refuses(H + _MODEL + "def grab():\n    global model\n    alias = model\n    return alias\n" + _SIDE + _END,
                 "kernel_module_statement_unresolved:k: call_result:", "'model' has 2 binding sites in the module (defined at line 15, declared global at line 20)")


# ---------------------------------------------------------------------------------------------------------------------
# readless-callee-34
# ---------------------------------------------------------------------------------------------------------------------

_R34 = {
    "local_shadow": RM + '    k = lambda a: a\n    return {"output": k(inputs["a"])}\n',
    "parameter_shadow": 'def model(a, k):\n    return k(a)\n' + RM + '    return {"output": model(inputs["a"], lambda a: a)}\n',
    "default_shadow": 'def model(a, k=lambda a: a):\n    return k(a)\n' + RM + '    return {"output": model(inputs["a"])}\n',
    "assigned_local_shadow": RM + '    k = lambda a: a\n    y = k(inputs["a"])\n    return {"output": y}\n',
}


class TestReadlessCallee34:
    @pytest.mark.parametrize("name", sorted(_R34))
    def test_a_shadowed_call_is_not_wrapped(self, name):
        _lowers(H + _R34[name], [[F32, F32]], absent=("_truncate_bf16(k(", "_truncate_bf16(y)"))

    def test_a_resolved_call_is_wrapped_in_expression_position(self):
        _lowers(H + RM + '    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n', [[2.0, 2.0]],
                contains=("_truncate_bf16(k(",))

    def test_a_resolved_call_is_wrapped_in_assignment_position(self):
        _lowers(H + RM + '    y = k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16"))\n    return {"output": y.data}\n', [[2.0, 2.0]],
                contains=("y = _truncate_bf16(y)",))

    def test_a_module_rebinding_refuses_a_read_less_program(self):
        r = _refuses(H + "k = lambda a: a\n" + RM + '    return {"output": k(inputs["a"])}\n', "kernel_name_rebound:k:", "2 binding sites in the module")
        assert r["slow"] == {"value": {"output": [[F32, F32]]}}

    def test_a_module_rebinding_with_a_data_read_keeps_its_round_7_class(self):
        _refuses(H + "k = lambda a: a\n" + RM + '    return {"output": k(Tensor.from_numpy(inputs["a"], "M,N", dtype="bf16")).data}\n',
                 FUNC + "kernel_name_rebound:", "more than its own definition")

    def test_a_mutated_kernel_refuses_the_whole_program(self):
        code = H + "def r(a):\n    return a\nk.__code__ = r.__code__\n" + RM + '    return {"output": k(inputs["a"])}\n'
        r = _refuses(code, "kernel_implementation_unproved:k:", "is visibly mutated by 'k.__code__ = r.__code__'")
        assert "error" in r["slow"]

    def test_a_type_setattr_on_a_kernel_refuses_the_whole_program(self):
        code = H + 'def r(a):\n    return a\ntype(k).__setattr__(k, "__code__", r.__code__)\n' + RM + '    return {"output": k(inputs["a"])}\n'
        _refuses(code, "kernel_implementation_unproved:k:", "escapes as a value in 'type(k).__setattr__(")

    def test_the_new_refusals_are_typed_declines(self):
        assert is_typed_decline("kernel_implementation_unproved:k: 'k' escapes as a value") and is_typed_decline("kernel_name_rebound:k: x")


# ---------------------------------------------------------------------------------------------------------------------
# the reviewer's frozen round-10 edges file, byte-for-byte (vendored under review3_controls/)
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


class TestReview4Round10EdgesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's ``test_round10_edges.py`` (sha256 66056546…, 20 cases x 2 carriers; it imports its round-9 ``test_round9_provenance``):
        20 passed / 20 failed on an earlier revision (all 20 wrong public values), 20 / 20 on a later revision, 8 / 32 on the baseline revision; after findings
        32, 33 and 34 every case has the interpreter's public value -- 40 passed."""
        run, res = _run_reviewer_file(tmp_path, "test_round10_edges.py", extra=("test_round9_provenance.py",))
        assert res.stdout.splitlines()[-1].startswith("40 passed"), res.stdout[-3000:]
        recs = {p.stem: json.loads(p.read_text()) for p in (run / "runs" / "round10-head").glob("*.json")}
        assert len(recs) == 40, len(recs)
        for name, r in recs.items():
            assert r["public"] == r["slow"], (name, r["public"], r["slow"])
            case = name.rsplit("-", 1)[0]
            if case in ("local_shadow", "parameter_shadow", "default_shadow"):
                assert r["lowered"] and not r["reasons"] and r["slow"]["value"]["output"] == [[F32, F32]], (name, r)
            elif case in ("alias_code", "alias_chain_code", "parameter_code", "type_setattr_code"):
                assert not r["lowered"] and "function_implementation_unproved" in r["reasons"][0] and "escapes as a value" in r["reasons"][0], (name, r["reasons"])
            elif case in ("direct_from_numpy_bf16", "direct_from_numpy_fp16", "nested_from_numpy_bf16", "named_from_numpy_bf16"):
                assert r["lowered"] and not r["reasons"], (name, r["reasons"])
                assert r["slow"]["value"]["output"] == [[FP16, FP16] if case.endswith("fp16") else [BF16, BF16]], (name, r["slow"])
