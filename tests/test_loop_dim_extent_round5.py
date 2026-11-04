# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier change round 5 (the review on an earlier revision): return-instance-15, same-line-16, dim-registration-17,
scalar-extent-18.

* RETURN-INSTANCE-15: a returned allocation SITE inside a loop has one dynamic instance per iteration; the recorded
  output size witnesses only the last, so it is borrowed only when the site executes once (outside every loop).  An
  in-loop returned allocation is counted only from a statically evaluable size expression, else refused.
* SAME-LINE-16: statement identity in the replay is the AST position (line, col) against each loop's span, never a
  line range: ``for t in Dim("T"): Tmp = Tensor(U=3)`` on one line is the loop's own statement.
* DIM-REGISTRATION-17: an explicit ``Dim("U", size=...)`` -- in a for-header or standalone -- registers U in the
  runtime registry; the replay mirrors it as loop-carried state (``dim_registrations`` from the parser, the loop
  node's own ``size`` for headers) for every later loop over the name; unresolvable registrations refuse.
* SCALAR-EXTENT-18: an explicit ``size=`` always takes precedence over a registry binding of the same name; the
  tracer records omitted int defaults (``n: int = 3``) and zeros; a scalar the trace did not record refuses with
  ``loop_dim_scalar_unresolved`` even when the registry binds the name; a zero / negative explicit size refuses.
the reviewer's ``test_round4_edges.py`` (sha256 f82206188631d6df66ec3790dc6615b64bdddb824477a95195ed9f7dbb908459) is vendored verbatim and run byte-for-byte.
"""
import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml

from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.einsum_dsl.einsum_converter import LOOP_DIM_SCALAR_UNRESOLVED
from solar.einsum_lib import clear_dims, stop_trace, get_trace, build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "round5" / "checks-9b85d77c"
PINS = {"test_round4_edges.py": "f82206188631d6df66ec3790dc6615b64bdddb824477a95195ed9f7dbb908459"}
H = 'from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
K = '@kernel\ndef k(X:Tensor("T"),Z:Tensor("U"),n:int=3)->Tensor("O"):\n'
RUN = '\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"U")\n    return {"out":k(x,z).data}\n'
INPUTS = {"x": np.ones(2), "z": np.ones(1)}          # registry at entry: T=2, U=1


def prog(body, call="k(x,z)", sig=K, helper=""):
    return H + helper + sig + body + RUN.replace("k(x,z)", call)


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def observe(tmp_path, code, inputs=None, strip_scalar_args=False, strip=()):
    """Run the program through the interpreter with tracing, convert, analyze.  Returns a record with the output,
    the recorded metadata node, the kernel layer (or the conversion error) and the analysis.  ``strip`` removes
    metadata fields from the kernel node before conversion (a legacy trace is simulated by dropping them)."""
    dsl = tmp_path / "dsl"; dsl.mkdir(exist_ok=True)
    (dsl / "probe_einsum_lib.py").write_text(code)
    clear_dims(); stop_trace()
    out, _ = execute_translated(code, {}, all_inputs=inputs or INPUTS, fast=False, trace=True, shape_only=False)
    md = build_metadata(get_trace())
    if strip_scalar_args:
        md["graph"]["k#0"].pop("scalar_args", None)
    for field in strip:
        md["graph"]["k#0"].pop(field, None)
    write_metadata_yaml(md, str(dsl / "probe_einsum_lib.yaml"))
    rec = {"out": {k: np.asarray(v).tolist() for k, v in out.items()}, "node": md["graph"]["k#0"], "md": md}
    try:
        p = translate_with_einsum(dsl, tmp_path / "einsum")
        rec["layer"] = yaml.safe_load(p.read_text())["layers"]["k#0"]
        rec["analysis"] = EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
    except Exception as exc:
        rec["error"] = exc
    return rec


LOOP_REFUSAL_CODES = ("loop_dim_extent_mismatch", "loop_dim_extent_unresolved", "loop_dim_order_unknown", "loop_entry_unresolved",
                      "loop_dim_scalar_unresolved", "loop_iterations_omitted")


def counted(rec, n):
    assert "error" not in rec, rec.get("error")
    L = rec["layer"]
    assert L.get("loop_iterations_complete") is True and L.get("loop_iterations") == n, L
    assert not any(c in L for c in LOOP_REFUSAL_CODES), L
    A = rec["analysis"]["layers"]["k#0"]
    # On the target merged in round 7 (an earlier change, issue #59) a kernel whose statements are outside the classifier's proven
    # grammar is ``incomplete`` for the CLASS-EVIDENCE reason (these tiny probes read ``X[0]`` with a constant index:
    # "assigned value not proven").  That verdict is an earlier change's and orthogonal to the loop extent under test here, so it is
    # the one incompleteness accepted; any other reason (a loop refusal) still fails.
    if A.get("status") == "incomplete":
        assert A.get("incomplete_reason") == "class_evidence_unproven" and L.get("class_evidence_uncertain") == "class_evidence_unproven", A
    return L


def refused(rec, code, needle=None):
    assert "error" not in rec, rec.get("error")
    L = rec["layer"]
    assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
    assert code in L, (code, L)
    if needle:
        assert re.search(needle, L[code]), L[code]
    assert rec["analysis"]["layers"]["k#0"]["status"] == "incomplete"
    return L


class TestReview4Round4EdgesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's nine round-4 probes byte-for-byte: 9 passed, and read back from the file's own records the
        exact outcomes -- counts 3 / 6 / 5 / 5 / 3 / 3 and the three typed refusals."""
        src = FIX / "test_round4_edges.py"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == PINS["test_round4_edges.py"]
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        (run / "test_round4_edges.py").write_bytes(src.read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round4_edges.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "9 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "round4-head").iterdir()}
        expect = {"explicit_dim_before_loop": 3, "explicit_dim_registers": 6, "multiline_allocation_control": 5, "same_line_allocation": 5,
                  "scalar_default_missing": 3, "scalar_explicit_control": 3}
        for name, n in expect.items():
            L = recs[name]["graph"]["layers"]["k#0"]
            assert L["loop_iterations_complete"] is True and L["loop_iterations"] == n == recs[name]["runtime_count"], (name, L)
        for name, code, needle in (("loop_alloc_instances", "loop_dim_extent_unresolved", "returned Tensor allocation .* once per iteration"),
                                   ("scalar_legacy_missing", "loop_dim_scalar_unresolved", "sized by the scalar argument\\(s\\) 'n'"),
                                   ("scalar_zero", "loop_dim_extent_unresolved", "evaluates to 0")):
            r = recs[name]; L = r["graph"]["layers"]["k#0"]
            assert "conversion_error" not in r and L["loop_iterations_complete"] is False and "loop_iterations" not in L, (name, r.get("conversion_error"), L)
            assert re.search(needle, L[code]) and r["analysis"]["layers"]["k#0"]["status"] == "incomplete", (name, L)
        assert recs["scalar_default_missing"]["metadata"]["graph"]["k#0"]["scalar_args"] == {"n": 3}
        assert recs["scalar_zero"]["metadata"]["graph"]["k#0"]["scalar_args"] == {"n": 0}


class TestSameLine:
    ONE_LINE = '    Y=Tensor(O=1)\n    for t in Dim("T"): Tmp=Tensor(U=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'
    MULTI = '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Tmp=Tensor(U=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'

    def test_one_line_body_allocation_is_the_loops_own_statement(self, tmp_path):
        """2 allocations (one per T iteration, each re-registering U=3) + 3 increments = 5; the output witnesses the 3."""
        r = observe(tmp_path, prog(self.ONE_LINE))
        assert r["out"] == {"out": [3.0]}
        counted(r, 5)

    def test_multiline_control(self, tmp_path):
        r = observe(tmp_path, prog(self.MULTI))
        assert r["out"] == {"out": [3.0]}
        counted(r, 5)

    def test_one_line_inner_loop(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"): Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [2.0]}
        counted(r, 2)

    def test_parser_positions(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"; p.write_text(prog(self.ONE_LINE))
        info = extract_kernel_info(p)["k"]
        loop_t, loop_u = info["loop_nest"]
        alloc = next(a for a in info["local_tensor_dims"] if "U" in a["dims"])
        assert loop_t["line"] == loop_t["end"] == alloc["line"] and loop_t["col"] < alloc["col"] <= loop_t["end_col"]
        assert all(isinstance(n.get("col"), int) and isinstance(n.get("end_col"), int) for n in (loop_t, loop_u))
        assert all(isinstance(a.get("col"), int) for a in info["local_tensor_dims"])


class TestReturnInstance:
    IN_LOOP_DYNAMIC = '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y=Tensor(R=1+2*t)\n        for r in Dim("R"):\n            Y[0]+=X[0]\n    return Y\n'

    def test_returned_allocation_inside_a_loop_is_not_borrowed(self, tmp_path):
        """the reviewer's loop_alloc_instances: runtime 1 + 3 increments + 2 allocations = 6; the output (R=3) witnesses
        only the LAST instance -> refused, never 8."""
        r = observe(tmp_path, prog(self.IN_LOOP_DYNAMIC))
        assert r["out"] == {"out": [3.0, 0.0, 0.0]}
        refused(r, "loop_dim_extent_unresolved", "returned Tensor allocation at line 6, which executes once per iteration")

    def test_returned_allocation_inside_a_loop_with_a_static_size_counts(self, tmp_path):
        """The explicit evaluation path: ``Y=Tensor(R=3)`` inside the loop sizes every instance from the literal:
        2 allocations + 2 x 3 increments = 8."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y=Tensor(R=3)\n        for r in Dim("R"):\n            Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0, 0.0, 0.0]}
        counted(r, 8)

    def test_returned_allocation_outside_loops_still_borrows_the_output_size(self, tmp_path):
        """round-3 control ``returned_dynamic``: the site executes once; the recorded output size (3) is its size."""
        r = observe(tmp_path, prog('    Y=Tensor(R=choose())\n    for r in Dim("R"):\n        Y[r]=X[0]\n    return Y\n', helper="def choose():\n    return 3\n"))
        assert r["out"] == {"out": [1.0, 1.0, 1.0]}
        counted(r, 3)


class TestDimRegistration:
    def test_for_header_explicit_size_registers_for_the_next_loop(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for u in Dim("U",size=3):\n        Y[0]+=X[0]\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [6.0]}
        counted(r, 6)

    @pytest.mark.parametrize("stmt", ['    D=Dim("U",size=3)\n', '    Dim("U",size=3)\n'], ids=["assigned", "bare-expression"])
    def test_standalone_registration(self, tmp_path, stmt):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n' + stmt + '    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    def test_registration_inside_the_outer_body_is_loop_carried(self, tmp_path):
        """First outer iteration U=1, the registration at the end of the body makes the second U=3: 1 + 3 increments
        (the output witnesses 4) plus the registration statement itself executed twice at the outer level = 6, the
        in-loop statement count the reviewer's line tracer reports."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0]+=X[0]\n        D=Dim("U",size=3)\n    return Y\n'))
        assert r["out"] == {"out": [4.0]}
        L = counted(r, 6)
        assert L.get("loop_statements") == 6

    @pytest.mark.parametrize("legacy", [False, True], ids=["signature-order", "legacy-no-input_order"])
    def test_unevaluable_header_size_is_refused_evaluable_is_counted(self, tmp_path, legacy):
        """Lite 203 ``block_absmax``: ``nb = X.shape[1] // 16`` sizes the allocation ``Y`` and the inner header
        ``for b in Dim("B", size=nb)`` under ``for t``.  With the signature input-order marker the arithmetic evaluates
        and the nest counts 3*2*(16+2) = 108 statement iterations.  A legacy trace (no marker) cannot evaluate ``nb`` at
        the call: round 6 (header-fallback-19, manager ruling) REFUSES the loop with ``loop_dim_extent_unresolved``
        naming the expression and the missing marker -- an earlier revision sized it from the registry binding and re-registered
        that assumption, which certified 3 where the reviewer's counterexample executed 9.  The row's 7,741,440 is no longer
        certified from its legacy trace (a Lite retrace item, recorded in the handoff)."""
        code = (H + '@kernel\ndef k(X:Tensor("T,H"))->Tensor("T,B"):\n'
                '    nb=X.shape[1]//16\n    Y=Tensor(T=X.shape[0],B=nb)\n    for t in Dim("T"):\n        for b in Dim("B",size=nb):\n'
                '            m=0.0\n            for j in Dim("J",size=16):\n                m=max(m,abs(X[t,b*16+j]))\n            Y[t,b]=m\n    return Y\n'
                '\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T,H")\n    return {"out":k(x).data}\n')
        x = np.arange(96, dtype=float).reshape(3, 32)
        r = observe(tmp_path, code, inputs={"x": x}, strip=("input_order",) if legacy else ())
        assert np.asarray(r["out"]["out"]).tolist() == x.reshape(3, 2, 16).max(axis=2).tolist()
        assert ("input_order" in r["node"]) is not legacy
        if legacy:
            refused(r, "loop_dim_extent_unresolved", r"explicit size \(X\.shape\[1\] // 16\) of the loop over Dim\('B'\).*cannot be evaluated at this call.*signature input-order marker")
        else:
            L = counted(r, 108)
            assert L.get("loop_statements") == 108

    def test_unresolvable_registration_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    D=Dim("U",size=choose())\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n', helper="def choose():\n    return 3\n"))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", "explicit Dim registration|Dim at line")

    def test_conditional_registration_with_another_size_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    if X[0]>0:\n        D=Dim("U",size=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", "conditional explicit Dim registration")

    def test_non_literal_dim_name_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    name="U"\n    D=Dim(name,size=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", "not a string literal")

    def test_parser_records_registrations_outside_headers_only(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(prog('    Y=Tensor(O=1)\n    D=Dim("U",size=3)\n    for u in Dim("V",size=2):\n        Dim("W",size=X.shape[0])\n        Y[0]+=X[0]\n    return Y\n'))
        info = extract_kernel_info(p)["k"]
        regs = info["dim_registrations"]
        assert [(r["dim"], r["size"], r["line"], r["conditional"]) for r in regs] == [("U", 3, 5, False), ("W", ("shape", "X", 0), 7, False)]
        assert all(isinstance(r["col"], int) for r in regs)
        assert info["loop_nest"][0]["dim"] == "V" and info["loop_nest"][0]["size"] == 2


class TestScalarExtent:
    BODY = '    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n'

    def test_omitted_default_is_recorded_and_counted(self, tmp_path):
        r = observe(tmp_path, prog(self.BODY))
        assert r["node"]["scalar_args"] == {"n": 3} and r["out"] == {"out": [3.0]}
        counted(r, 3)

    def test_explicit_size_takes_precedence_over_the_registry(self, tmp_path):
        r = observe(tmp_path, prog(self.BODY, call="k(x,z,n=5)"))
        assert r["node"]["scalar_args"] == {"n": 5} and r["out"] == {"out": [5.0]}
        counted(r, 5)

    def test_missing_scalar_evidence_refuses_even_with_a_registry_binding(self, tmp_path):
        r = observe(tmp_path, prog(self.BODY, call="k(x,z,n=3)"), strip_scalar_args=True)
        assert "scalar_args" not in r["node"]
        refused(r, LOOP_DIM_SCALAR_UNRESOLVED, "sized by the scalar argument\\(s\\) 'n'.*did not record")

    @pytest.mark.parametrize("value,needle", [(0, "evaluates to 0"), (-1, "evaluates to -1")])
    def test_zero_and_negative_explicit_sizes_refuse(self, tmp_path, value, needle):
        r = observe(tmp_path, prog(self.BODY, call=f"k(x,z,n={value})"))
        assert r["node"]["scalar_args"] == {"n": value}
        refused(r, "loop_dim_extent_unresolved", needle)

    def test_bool_defaults_are_not_scalar_args(self, tmp_path):
        sig = '@kernel\ndef k(X:Tensor("T"),Z:Tensor("U"),n:int=3,flag:bool=True)->Tensor("O"):\n'
        r = observe(tmp_path, prog(self.BODY, sig=sig))
        assert r["node"]["scalar_args"] == {"n": 3}
        counted(r, 3)

    def test_standalone_registration_from_a_missing_scalar_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    D=Dim("U",size=n)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'), strip_scalar_args=True)
        refused(r, LOOP_DIM_SCALAR_UNRESOLVED, "explicit Dim\\('U', size=...\\) .* sized by the scalar argument\\(s\\) 'n'")
