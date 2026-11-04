# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier change round 6 (the review on an earlier revision; manager ruling): header-fallback-19 and opaque-size-20.

* HEADER-FALLBACK-19: an explicit ``size=`` the call cannot EVALUATE (shape arithmetic on a legacy trace without the
  signature input-order marker) is a typed refusal ``loop_dim_extent_unresolved`` naming the expression and why -- never
  the registry binding, the positional fallback or "the count the loop was counted with" (an earlier revision certified 3 where the
  runtime executed 9).  An evaluable header size (literal, signature-order trace, recorded scalar, plain ``.shape[i]``)
  counts and registers the evaluated value as before.
* OPAQUE-SIZE-20: the parser preserves an explicit ``size=`` it cannot reduce to a supported form as
  ``("opaque", text, line, col)`` instead of dropping it; the converter refuses it ("explicit size expression not
  supported: <text>") -- never treats it as absent, never reads the registry binding.  Supported forms are unchanged.
the reviewer's ``test_round5_edges.py`` (sha256 cd2acf1b48f2eaa0ba49ac0d255ab55c1e1a68928ebf11180c9f26abc7bd10f1) and ``test_round5_binop.py``
(sha256 e995a78e458cc5378615bbb42b50a2f97f60726638b62f1353efdca417132e6a) are vendored verbatim and run byte-for-byte.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_lib import clear_dims, stop_trace

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "round6" / "checks-ea874283"
PINS = {"test_round5_binop.py": "e995a78e458cc5378615bbb42b50a2f97f60726638b62f1353efdca417132e6a", "test_round5_edges.py": "cd2acf1b48f2eaa0ba49ac0d255ab55c1e1a68928ebf11180c9f26abc7bd10f1"}
_spec = importlib.util.spec_from_file_location("_r5", ROOT / "tests" / "test_loop_dim_extent_round5.py")
_r5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r5)
observe, counted, refused, prog, H, K = _r5.observe, _r5.counted, _r5.refused, _r5.prog, _r5.H, _r5.K

# the reviewer's round-5 counterexample programme: X (T=2,H=3), Z binds B=1; the runtime registers B=3 at each explicit
# header and executes 2*3 + 3 = 9 increments
SHAPE = (H + '@kernel\ndef k(X:Tensor("T,H"),Z:Tensor("B"))->Tensor("O"):\n    nb=X.shape[1]\n    Y=Tensor(O=1)\n    for t in Dim("T"):\n'
         '        for b in Dim("B",size=nb):\n            Y[0]+=X[0,0]\n    for b in Dim("B"):\n        Y[0]+=X[0,0]\n    return Y\n'
         '\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T,H")\n    z=Tensor.from_numpy(inputs["z"],"B")\n    return {"out":k(x,z).data}\n')
SHAPE_INPUTS = {"x": np.ones((2, 3)), "z": np.ones(1)}
COMMON = '    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n'


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


class TestReview4Round5FilesVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's 15-case round-5 matrix and the two binop probes byte-for-byte: 17 passed, and read back from the
        files' own records: the counts the runtime executed, and the typed refusals the ruling requires."""
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        for name, digest in PINS.items():
            src = FIX / name
            assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, name
            (run / name).write_bytes(src.read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round5_edges.py", "test_round5_binop.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0 and "17 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "round5-head").iterdir()}
        assert len(recs) == 17
        expect = {"header_shape_fresh": 9, "header_shape_legacy": 9, "header_binop_fresh": 9, "semicolon_allocations": 9,
                  "semicolon_allocations_legacy_columns": 9, "nested_header_allocation": 8, "nested_header_allocation_legacy_columns": 8,
                  "comment_shifted": 8, "tab_indentation": 8, "keyword_only_default": 3, "keyword_only_supplied": 4}
        for name, n in expect.items():
            L = recs[name]["graph"]["layers"]["k#0"]
            assert L["loop_iterations_complete"] is True and L["loop_iterations"] == n == recs[name]["runtime_count"], (name, L)
        for name, code, needle in (("header_binop_legacy", "loop_dim_extent_unresolved", r"explicit size \(X\.shape\[1\] // 1\).*cannot be evaluated at this call.*signature input-order marker"),
                                   ("header_helper", "loop_dim_extent_unresolved", r"explicit size expression not supported: choose\(\)"),
                                   ("varargs_scalar", "loop_dim_extent_unresolved", r"explicit size expression not supported: sizes\[0\]"),
                                   ("varkw_scalar", "loop_dim_extent_unresolved", r"explicit size expression not supported: sizes\['n'\]"),
                                   ("bool_default_used", "loop_dim_scalar_unresolved", r"sized by the scalar argument\(s\) 'n'"),
                                   ("returned_once", "loop_dim_extent_unresolved", r"returned Tensor allocation .* once per iteration .*single instance")):
            r = recs[name]; L = r["graph"]["layers"]["k#0"]
            assert "conversion_error" not in r and L["loop_iterations_complete"] is False and "loop_iterations" not in L, (name, r.get("conversion_error"), L)
            assert re.search(needle, L[code]) and r["analysis"]["layers"]["k#0"]["status"] == "incomplete", (name, L)
        assert recs["header_binop_legacy"]["runtime_count"] == 9 and "input_order" not in recs["header_binop_legacy"]["metadata"]["graph"]["k#0"]
        assert recs["header_helper"]["runtime_count"] == 9 and recs["varargs_scalar"]["runtime_count"] == 3 and recs["varkw_scalar"]["runtime_count"] == 3


class TestHeaderFallback:
    BINOP = SHAPE.replace("nb=X.shape[1]", "nb=X.shape[1]//1")

    def test_evaluable_arithmetic_counts_the_runtime_nine(self, tmp_path):
        r = observe(tmp_path, self.BINOP, inputs=SHAPE_INPUTS)
        assert r["out"] == {"out": [9.0]}
        counted(r, 9)

    def test_unevaluable_arithmetic_on_a_legacy_trace_refuses(self, tmp_path):
        """the reviewer's counterexample: an earlier revision certified 3 (registry B=1 reused and re-registered); an earlier revision refused."""
        r = observe(tmp_path, self.BINOP, inputs=SHAPE_INPUTS, strip=("input_order",))
        assert r["out"] == {"out": [9.0]} and "input_order" not in r["node"]
        refused(r, "loop_dim_extent_unresolved", r"explicit size \(X\.shape\[1\] // 1\) of the loop over Dim\('B'\).*cannot be evaluated at this call: integer arithmetic.*signature input-order marker.*not sized from the registry binding")

    @pytest.mark.parametrize("legacy", [False, True], ids=["signature-order", "legacy-no-input_order"])
    def test_plain_shape_leaf_is_evaluable_on_both_traces(self, tmp_path, legacy):
        r = observe(tmp_path, SHAPE, inputs=SHAPE_INPUTS, strip=("input_order",) if legacy else ())
        assert r["out"] == {"out": [9.0]}
        counted(r, 9)

    def test_header_registration_carries_the_evaluated_size(self, tmp_path):
        """``for u in Dim("U", size=X.shape[0]*2)`` registers U=4 for the later loop: 4 + 4 = 8."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for u in Dim("U",size=X.shape[0]*2):\n        Y[0]+=X[0]\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [8.0]}
        counted(r, 8)


class TestOpaqueSize:
    HELPER = "def choose():\n    return 3\n"

    def test_helper_call_header_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U",size=choose()):\n            Y[0]+=X[0]\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n', helper=self.HELPER))
        assert r["out"] == {"out": [9.0]}
        refused(r, "loop_dim_extent_unresolved", r"loop over Dim\('U'\).*explicit size expression not supported: choose\(\) \(line 8\).*never read from the registry binding")

    def test_varargs_subscript_refuses(self, tmp_path):
        r = observe(tmp_path, prog(COMMON.replace("size=n", "size=sizes[0]"), sig=K.replace("n:int=3", "*sizes"), call="k(x,z,3)"))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", r"explicit size expression not supported: sizes\[0\]")

    def test_varkw_subscript_refuses(self, tmp_path):
        r = observe(tmp_path, prog(COMMON.replace("size=n", 'size=sizes["n"]'), sig=K.replace("n:int=3", "**sizes"), call="k(x,z,n=3)"))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", r"explicit size expression not supported: sizes\['n'\]")

    def test_attribute_size_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=len(X.shape)):\n        Y[0]+=X[0]\n    return Y\n'))
        refused(r, "loop_dim_extent_unresolved", r"explicit size expression not supported: len\(X\.shape\)")

    def test_standalone_opaque_registration_makes_the_name_unresolved(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    D=Dim("U",size=choose())\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n', helper=self.HELPER))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", r"explicit Dim registration at line 7 \(explicit size expression not supported: choose\(\)\)")

    @pytest.mark.parametrize("body,call,n", [
        ('    Y=Tensor(O=1)\n    for t in Dim("T",size=3):\n        Y[0]+=X[0]\n    return Y\n', "k(x,z)", 3),
        (COMMON, "k(x,z)", 3),
        (COMMON, "k(x,z,n=5)", 5),
        ('    Y=Tensor(O=1)\n    for t in Dim("T",size=X.shape[0]):\n        Y[0]+=X[0]\n    return Y\n', "k(x,z)", 2),
        ('    Y=Tensor(O=1)\n    for t in Dim("T",size=X.shape[0]*2+1):\n        Y[0]+=X[0]\n    return Y\n', "k(x,z)", 5),
        ('    Y=Tensor(O=1)\n    m=X.shape[0]+1\n    for t in Dim("T",size=m):\n        Y[0]+=X[0]\n    return Y\n', "k(x,z)", 3),
    ], ids=["literal", "scalar-default", "scalar-supplied", "shape-leaf", "shape-arithmetic", "local-name"])
    def test_supported_forms_keep_their_counts(self, tmp_path, body, call, n):
        r = observe(tmp_path, prog(body, call=call))
        assert r["out"] == {"out": [float(n)]}
        counted(r, n)

    def test_parser_preserves_opaque_specs(self, tmp_path):
        p = tmp_path / "probe_einsum_lib.py"
        p.write_text(prog('    Y=Tensor(O=1)\n    D=Dim("U",size=choose())\n    for t in Dim("T",size=n):\n        for v in Dim("V",size=sizes[0]):\n            Y[0]+=X[0]\n    return Y\n',
                          sig=K.replace("n:int=3", "n:int=3,*sizes"), helper=self.HELPER))
        info = extract_kernel_info(p)["k"]
        reg, = info["dim_registrations"]
        assert reg["dim"] == "U" and tuple(reg["size"][:2]) == ("opaque", "choose()") and isinstance(reg["size"][2], int) and isinstance(reg["size"][3], int)
        t, = info["loop_nest"]
        assert t["size"] == ("scalar", "n")
        v, = t["children"]
        assert tuple(v["size"][:2]) == ("opaque", "sizes[0]") and v["size"][2] == 9
