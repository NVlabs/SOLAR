# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier change round 8 (the review on an earlier revision): nested-dim-eval-25 -- nested ``Dim`` registrations in runtime order, or a typed refusal.

A ``Dim(...)`` call nested inside another Dim call's ARGUMENTS registers its size BEFORE the enclosing call runs (Python
evaluates arguments first, left to right).  an earlier revision -- like an earlier revision and the target an earlier revision -- excluded the ENTIRE
for-header iterator AST from the registrations (a nested call in a header was skipped: ``for t in Dim("T",
parallel=bool(Dim("T", size=3)))`` certified 2 where the runtime executed 3) and replayed standalone registrations in
SOURCE order (the outer constructor before its nested argument: ``Dim("U", parallel=bool(Dim("U", size=4)), size=3)``
certified 4 where the runtime binds 4 then 3 and executes 3) -- all marked ``loop_iterations_complete``.
Round 8: only the header's OWN call is excluded; every registration / allocation of a statement carries its evaluation-
order index (``eseq``; :func:`solar.einsum_dsl.ast_parser._eval_order`: arguments before the enclosing call, left to
right, a subscript's value before its slice, an assignment's value before its targets), a header-nested registration
carries ``header_of`` and is replayed by the converter as a statement of the ENCLOSING body right before the loop (once
per start of that ``for``, never per iteration), and a registration nested in an expression form the parser does not
model (a comprehension, a generator, ``await``, a loop target, an annotation ...) carries ``order_unknown``: the
converter refuses every later loop over that name with ``loop_dim_extent_unresolved`` naming the nested call -- never a
silent skip (no arbitrary-expression evaluator).  A registration in a short-circuit / conditional operand position is
``conditional`` (the established round-4 semantics: refused unless it re-binds the same value).
the reviewer's ``test_round7_nested_complete_v2.py`` (sha256 09f6cc3b...; it imports their ``test_round7_edges.py`` ba1f0702...
and round-5 ``test_round5_edges.py`` cd2acf1b...) is vendored verbatim and run byte-for-byte: 7 + 12 cases.
"""
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from solar.einsum_dsl.ast_parser import extract_kernel_info, _eval_order
from solar.einsum_lib import clear_dims, stop_trace

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX8 = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "round8" / "checks-2f97bdbf"
PINS = {FIX8 / "test_round5_edges.py": "cd2acf1b48f2eaa0ba49ac0d255ab55c1e1a68928ebf11180c9f26abc7bd10f1",
        FIX8 / "test_round7_edges.py": "ba1f0702eb7c7eece8ac9abacbefa54e644b5c5464fdba44ec6b63c3cc1577d1",
        FIX8 / "test_round7_nested_complete_v2.py": "09f6cc3b7c052c51b2b004f60b7f9ee19e7e122baa1cf58ed026196d8285fb73"}
_spec = importlib.util.spec_from_file_location("_r5", ROOT / "tests" / "test_loop_dim_extent_round5.py")
_r5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r5)
observe, counted, refused, prog = _r5.observe, _r5.counted, _r5.refused, _r5.prog


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def header(call, body='        Y[0]+=Z[0]\n'):
    return prog('    Y=Tensor(O=1)\n    for t in ' + call + ':\n' + body + '    return Y\n')


def standalone(reg):
    return prog('    Y=Tensor(O=1)\n    D=' + reg + '\n    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n')


NESTED = ("r7_complete_header_parallel_nested_same_v2", "r7_complete_header_parallel_nested_same_bool_v2",
          "r7_complete_header_parallel_nested_other_v2", "r7_complete_header_parallel_nested_other_bool_v2",
          "r7_complete_standalone_nested_overwrite_v2", "r7_complete_standalone_nested_overwrite_bool_v2")


class TestReview4Round7FilesVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's nested-registration matrix (7) and round-7 edge matrix (12) byte-for-byte, and read back from their own
        records: the six nested cases count exactly the runtime executions (3 / 5 / 3), complete, and the analysis bills them;
        the literal-parallel control still counts 2."""
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        for src, digest in PINS.items():
            assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, src.name
            (run / src.name).write_bytes(src.read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round7_nested_complete_v2.py", "test_round7_edges.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0 and "19 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "round5-head").iterdir()}
        want = {"same": 3, "other": 5, "overwrite": 3}
        for name in NESTED:
            r = recs[name]; L = r["graph"]["layers"]["k#0"]; A = r["analysis"]["layers"]["k#0"]
            n = next(v for k, v in want.items() if k in name)
            assert r["runtime_count"] == n, (name, r["runtime_count"])
            assert L["loop_iterations_complete"] is True and L["loop_iterations"] == n and A["other_ops"] == n and A.get("status") != "incomplete", (name, L, A)
        c = recs["r7_complete_literal_parallel_control_v2"]
        assert c["runtime_count"] == 2 == c["graph"]["layers"]["k#0"]["loop_iterations"]
        # the round-7 edge matrix: ``r7_size_nested_call`` (``size=choose(Dim("U", size=3))``) still refuses on the OPAQUE size
        L = recs["r7_size_nested_call"]["graph"]["layers"]["k#0"]
        assert L["loop_iterations_complete"] is False and "choose(Dim('U', size=3))" in L["loop_dim_extent_unresolved"], L


class TestNestedRegistrationRuntimeOrder:
    """Author controls (the reviewer's list): nested in ``parallel=``, nested in ``size=``, nested two levels, nested inside a
    tensor-shape argument, a nested call whose name differs from the header's, and the literal-parallel control."""

    def test_nested_in_parallel_same_name_counts_three(self, tmp_path):
        r = observe(tmp_path, header('Dim("T",parallel=bool(Dim("T",size=3)))'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    def test_nested_other_name_then_sequential_loop_counts_five(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T",parallel=bool(Dim("U",size=3))):\n        Y[0]+=Z[0]\n'
                                   '    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        assert r["out"] == {"out": [5.0]}
        counted(r, 5)

    def test_nested_other_name_sizes_the_inner_loop(self, tmp_path):
        """The header registration runs once when the ``for`` starts, before any iteration: T=2 trips x U=3 = 6."""
        r = observe(tmp_path, header('Dim("T",parallel=bool(Dim("U",size=3)))', body='        for u in Dim("U"):\n            Y[0]+=Z[0]\n'))
        assert r["out"] == {"out": [6.0]}
        counted(r, 6)

    def test_nested_in_size_is_an_opaque_size_refusal_not_a_skip(self, tmp_path):
        """``size=(Dim("U", size=3), n)[1]`` is an explicit size the parser does not reduce: the T loop refuses on the opaque
        size (count-or-refuse); the nested U registration is still replayed (U=3, not 1) -- nothing is skipped."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=(Dim("U",size=3),n)[1]):\n        Y[0]+=Z[0]\n'
                                   '    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        assert r["out"] == {"out": [6.0]}
        refused(r, "loop_dim_extent_unresolved", r"explicit size expression not supported: \(Dim\('U', size=3\), n\)\[1\]")

    def test_nested_two_levels_counts_the_last_binding(self, tmp_path):
        """4 is registered, then 3, then the header iterates the registry: 3."""
        r = observe(tmp_path, header('Dim("T",parallel=bool(Dim("T",parallel=bool(Dim("T",size=4)),size=3)))'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    def test_nested_inside_a_tensor_shape_argument(self, tmp_path):
        """``Tensor(U=4, V=(Dim("U", size=3), 1)[1])``: the nested registration (U=3) runs while the arguments are evaluated,
        the allocation's own registration (U=4) after them -- the loop over U executes 4."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    Tmp=Tensor(U=4,V=(Dim("U",size=3),1)[1])\n    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        assert r["out"] == {"out": [4.0]}
        counted(r, 4)

    @pytest.mark.parametrize("reg", ['Dim("U",parallel=bool(Dim("U",size=4)),size=3)', 'Dim("U",size=3,parallel=bool(Dim("U",size=4)))',
                                     'Dim("U",parallel=Dim("U",size=4),size=3)'], ids=["parallel-first", "size-first", "no-bool"])
    def test_standalone_nested_overwrite_counts_the_outer_size(self, tmp_path, reg):
        """The nested argument constructor binds 4, the enclosing call binds 3 afterwards -- whatever the keyword order."""
        r = observe(tmp_path, standalone(reg))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    @pytest.mark.parametrize("call", ['Dim("T",parallel=True)', 'Dim("T",parallel=bool(Dim("T")))', 'Dim("T",parallel=bool(Dim("T",size=None)))'],
                             ids=["literal", "nested-no-size", "nested-size-None"])
    def test_literal_parallel_and_size_less_nested_calls_still_count_two(self, tmp_path, call):
        r = observe(tmp_path, header(call))
        assert r["out"] == {"out": [2.0]}
        counted(r, 2)

    def test_nested_in_an_unmodelled_expression_is_a_typed_refusal_naming_the_call(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    L=[Dim("U",size=3) for _ in range(1)]\n    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved",
                r"re-registered by the explicit Dim registration Dim\('U', size=3\) at line 5, nested in a ListComp expression whose evaluation order the parser does not model")

    @pytest.mark.parametrize("reg,out", [('Dim("U",parallel=(n>5 and bool(Dim("U",size=4))))', 1.0), ('Dim("U",parallel=bool(Dim("U",size=4)) if n>2 else False)', 4.0)],
                             ids=["short-circuit-not-taken", "conditional-taken"])
    def test_nested_in_a_conditional_operand_is_a_conditional_registration(self, tmp_path, reg, out):
        """Whether the nested call runs depends on a value: the registration is conditional and re-binds a different value
        than the established U=1 -> refused (round-4 semantics), never counted from either branch."""
        r = observe(tmp_path, standalone(reg))
        assert r["out"] == {"out": [out]}
        refused(r, "loop_dim_extent_unresolved", r"may be re-registered by a conditional explicit Dim registration at line 5 \(Dim\(U=4\)\)")


class TestParserRecords:
    def test_header_nested_registration_is_recorded_with_header_of_and_order(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(header('Dim("T",parallel=bool(Dim("T",parallel=bool(Dim("T",size=4)),size=3)))'))
        info = extract_kernel_info(p)["k"]
        t, = info["loop_nest"]
        assert t["dim"] == "T" and t["size"] is None                      # the header's own call passes no size
        regs = info["dim_registrations"]
        assert [(r["dim"], r["size"], r["header_of"]) for r in regs] == [("T", 4, [t["line"], t["col"]]), ("T", 3, [t["line"], t["col"]])]
        assert regs[0]["eseq"] < regs[1]["eseq"] and regs[0]["col"] > regs[1]["col"], regs   # inner first in evaluation order, later in the source
        assert all("order_unknown" not in r and r["conditional"] is False for r in regs)

    def test_standalone_nested_registrations_in_evaluation_order(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(standalone('Dim("U",parallel=bool(Dim("U",size=4)),size=3)'))
        regs = extract_kernel_info(p)["k"]["dim_registrations"]
        assert [(r["dim"], r["size"]) for r in regs] == [("U", 4), ("U", 3)] and regs[0]["eseq"] < regs[1]["eseq"], regs
        assert all("header_of" not in r for r in regs) and {r["stmt_line"] for r in regs} == {5}

    def test_unmodelled_nesting_is_recorded_as_order_unknown(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(prog('    Y=Tensor(O=1)\n    L=[Dim("U",size=3) for _ in range(1)]\n    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        reg, = extract_kernel_info(p)["k"]["dim_registrations"]
        assert reg["dim"] == "U" and reg["size"] == 3 and reg["order_unknown"] == "a ListComp expression" and reg["text"] == "Dim('U', size=3)", reg

    def test_single_allocation_statements_carry_no_order_keys(self, tmp_path):
        """A statement with one registering node needs no evaluation-order keys: graphs of such kernels are unchanged."""
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y[0]+=Z[0]\n    return Y\n'))
        info = extract_kernel_info(p)["k"]
        alloc, = info["local_tensor_dims"]
        assert not {"eseq", "stmt_line", "stmt_col", "header_of"} & set(alloc) and info["dim_registrations"] is None, info

    def test_tensor_shape_argument_nesting_orders_the_allocation_after_its_arguments(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(prog('    Y=Tensor(O=1)\n    Tmp=Tensor(U=4,V=(Dim("U",size=3),1)[1])\n    for u in Dim("U"):\n        Y[0]+=Z[0]\n    return Y\n'))
        info = extract_kernel_info(p)["k"]
        tmp = next(a for a in info["local_tensor_dims"] if "U" in a["dims"]); reg, = info["dim_registrations"]
        assert reg["eseq"] < tmp["eseq"] and tmp["stmt_line"] == reg["stmt_line"] == 5, (reg, tmp)

    def test_eval_order_helper(self):
        """Arguments before the enclosing call, left to right; a subscript's value before its slice; an assignment's value
        before its targets; dict keys and values pairwise; short-circuit and conditional operands flagged; a comprehension
        is an unmodelled form."""
        stmt = ast.parse('Y[g(a)] = f(h(b), c, k=d, **{e: m(n)}) if p else q').body[0]
        order = _eval_order(stmt)
        names = {}
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                names[node.id] = order[id(node)]
        seq = {k: v[0] for k, v in names.items()}
        assert seq["p"] < seq["h"] < seq["b"] < seq["c"] < seq["d"] < seq["e"] < seq["m"] < seq["n"] < seq["g"] < seq["a"] or seq["p"] < seq["h"]  # value before targets
        assert seq["b"] < seq["c"] < seq["d"] < seq["e"] < seq["n"] < seq["g"] < seq["a"], seq
        assert names["p"][1] is False and all(names[k][1] is True for k in ("h", "b", "c", "d", "e", "m", "n", "q")), names
        assert all(v[2] is None for v in names.values())
        comp = ast.parse('L = [Dim("U", size=3) for _ in range(1)]').body[0]
        o2 = _eval_order(comp)
        call = next(n for n in ast.walk(comp) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Dim")
        assert o2[id(call)][2] == "a ListComp expression"
        aug = ast.parse('A[i] += f(x)').body[0]
        o3 = _eval_order(aug); ids = {n.id: o3[id(n)][0] for n in ast.walk(aug) if isinstance(n, ast.Name)}
        assert ids["A"] < ids["i"] < ids["f"] < ids["x"], ids
