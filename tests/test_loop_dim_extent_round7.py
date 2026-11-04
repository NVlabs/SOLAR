# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier change round 7 (the review on an earlier revision; reconciliation onto an earlier revision): dim-call-binding-23.

The parser binds every ``Dim(...)`` call -- a for-header and a standalone registration alike -- against the RUNTIME
signature ``Dim(name, parallel=False, size=None)``: positional index 2 IS the size (``Dim("T", False, 3)`` counts 3),
``size=`` IS the size, ``name=`` is the name, ``parallel`` passed positionally is never taken for the size, an explicit
``size=None`` is no size.  Any argument form the parser cannot bind -- ``**{"size": 3}``, ``*args``, a fourth positional,
an unknown or repeated keyword -- is an OPAQUE explicit size: ``loop_dim_extent_unresolved`` "Dim call arguments not
statically bindable: <call>" for a header, an unresolved registration for a standalone call -- never "absent" (an earlier revision
certified 2 / 2 / 1 / 1 where the runtime executed 3).  A header whose dimension NAME cannot be read (``Dim(*args)``) is
an omission naming the same reason: the kernel fails closed.
KB-REFORMER-CONCAT-EXTENT-1 (the review, acceptance review of an earlier revision; DIM-CALL-BINDING-23-adjacent): the declared Dim extent
wins over the inferred positional size.  The baseline converter (an earlier revision / an earlier revision) takes a MAX over every axis a loop
variable indexes bare and prefers it to the registry binding, so Reformer ``concat_last`` (D=1024 iterated twice into
D2=2048, first store ``C[b, s, d]``) is sized d=2048 from C: 33,554,432 other statements for 16,777,216 executed.  On this
branch (TARGET-DSL-DIM-EXTENT-13, round 4) the binding at the call -- D=1024 from X and Y -- sizes the loop; a bare access on
a WIDER axis is a slice, never an extent; a bare access on a NARROWER axis than the bound extent is a disagreement and refuses
(``loop_dim_extent_mismatch``).  Control: the reviewer's isolated kernel/metadata pair, byte-for-byte.
LITE58-EXTENT-399 / LITE229-EXTENT-399 (the review, Lite acceptance review of an earlier revision): the same baseline inference sizes Lite 58
``stable_scatter``'s cursor loop ``for e in Dim("E")`` from the OUTPUT shape (the first statement's first tensor, the local
``CUR``, is taken for the output: e=16,384 instead of the allocation's E=256; 49,152 for 256 + 2*16,384 = 33,024) and Lite 229
``swiglu_slots``'s ``for i in Dim("I")`` from the bare ``G1[n, i]`` read on the O=4096 axis (458,752 for 2*56*2048 = 229,376).
On this branch E is bound by ``CUR = Tensor(E=_E)`` and I by ``Y = Tensor(N=..., I=half)``.  Controls: both kernels byte-for-byte
on their saved node metadata, and both executed at small shapes against the runtime count.
the reviewer's ``test_round6_edges.py`` (sha256 6bf4492fb7a8074a0e2eefe0a206d696ecaf4487df472abb2547131feadec175; it imports
the reviewer's round-5 ``test_round5_edges.py``, cd2acf1b...) is vendored verbatim and run byte-for-byte: 17 cases, count-or-refuse.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

from solar.einsum_dsl.ast_parser import extract_kernel_info, _dim_call_binding
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.einsum_lib import clear_dims, stop_trace
import ast

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX6 = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "round6" / "checks-ea874283"
FIX7 = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "round7" / "checks-ed9be3d0"

PINS = {FIX6 / "test_round5_edges.py": "cd2acf1b48f2eaa0ba49ac0d255ab55c1e1a68928ebf11180c9f26abc7bd10f1",
        FIX7 / "test_round6_edges.py": "6bf4492fb7a8074a0e2eefe0a206d696ecaf4487df472abb2547131feadec175"}
_spec = importlib.util.spec_from_file_location("_r5", ROOT / "tests" / "test_loop_dim_extent_round5.py")
_r5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r5)
observe, counted, refused, prog, H, K = _r5.observe, _r5.counted, _r5.refused, _r5.prog, _r5.H, _r5.K
UNBINDABLE = "Dim call arguments not statically bindable"


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def header(call):
    return prog('    Y=Tensor(O=1)\n    for t in ' + call + ':\n        Y[0]+=X[0]\n    return Y\n')


def standalone(reg):
    return prog('    Y=Tensor(O=1)\n    D=' + reg + '\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n')


class TestReview4Round6FileVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's 17-case round-6 matrix byte-for-byte (count-or-refuse), and read back from its own records: the four
        binding forms count 3 (positional literal, keyword name) or refuse (``**{"size": 3}``) -- never 2 / 1."""
        run = tmp_path / "run"; run.mkdir(); (run / "runs").mkdir()
        for src, digest in PINS.items():
            assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, src.name
            (run / src.name).write_bytes(src.read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_round6_edges.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0 and "17 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "round5-head").iterdir()}
        for name in ("r6_positional_header_size", "r6_positional_standalone_size", "r6_keyword_name"):
            L = recs[name]["graph"]["layers"]["k#0"]
            assert L["loop_iterations_complete"] is True and L["loop_iterations"] == 3 == recs[name]["runtime_count"], (name, L)
        for name, needle in (("r6_keyword_dict_header_size", UNBINDABLE + r": Dim\('T', \*\*\{'size': 3\}\)"),
                             ("r6_keyword_dict_standalone_size", r"re-registered by the explicit Dim registration at line \d+ \(" + UNBINDABLE)):
            r = recs[name]; L = r["graph"]["layers"]["k#0"]
            assert r["runtime_count"] == 3 and L["loop_iterations_complete"] is False and "loop_iterations" not in L, (name, L)
            assert re.search(needle, L["loop_dim_extent_unresolved"]) and r["analysis"]["layers"]["k#0"]["status"] == "incomplete", (name, L)
        counts = {n: recs[n]["graph"]["layers"]["k#0"].get("loop_iterations") for n in ("r6_two_tensor_shapes", "r6_two_local_chain", "r6_posonly_tensor", "r6_keyword_reordered_tensor")}
        assert counts == {"r6_two_tensor_shapes": 3, "r6_two_local_chain": 3, "r6_posonly_tensor": 9, "r6_keyword_reordered_tensor": 9}, counts


class TestDimCallBinding:
    def test_positional_size_counts(self, tmp_path):
        r = observe(tmp_path, header('Dim("T",False,3)'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    @pytest.mark.parametrize("call,n", [("k(x,z)", 3), ("k(x,z,n=5)", 5)], ids=["default", "supplied"])
    def test_positional_scalar_size_with_positional_parallel(self, tmp_path, call, n):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    for t in Dim("T",True,n):\n        Y[0]+=X[0]\n    return Y\n', call=call))
        assert r["out"] == {"out": [float(n)]}
        counted(r, n)

    def test_keyword_name_counts(self, tmp_path):
        r = observe(tmp_path, header('Dim(name="T",size=3)'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    @pytest.mark.parametrize("call", ['Dim("T",3)', 'Dim("T",size=None)', 'Dim("T",parallel=True)'], ids=["positional-parallel", "size-None", "keyword-parallel"])
    def test_a_positional_parallel_or_an_explicit_none_is_no_size(self, tmp_path, call):
        """``Dim("T", 3)`` binds 3 to ``parallel``, not to ``size``: the runtime iterates the registry binding T=2."""
        r = observe(tmp_path, header(call))
        assert r["out"] == {"out": [2.0]}
        counted(r, 2)

    def test_keyword_dict_header_refuses(self, tmp_path):
        r = observe(tmp_path, header('Dim("T",**{"size":3})'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", UNBINDABLE + r": Dim\('T', \*\*\{'size': 3\}\) \(line 5\).*not sized from the registry binding")

    def test_starred_header_arguments_refuse(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    args=(False,3)\n    for t in Dim("T",*args):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", UNBINDABLE + r": Dim\('T', \*args\)")

    def test_unreadable_header_name_is_an_omission(self, tmp_path):
        """``Dim(*args)``: neither the name nor the size can be read; the loop is an omission naming the reason and the
        kernel fails closed (no count, incomplete)."""
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    args=("T",False,3)\n    for t in Dim(*args):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]} and "error" not in r, r.get("error")
        L = r["layer"]
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert any(UNBINDABLE in str(o) and "Dim(*args)" in str(o) for o in (L.get("loop_iterations_omitted") or [])), L
        assert r["analysis"]["layers"]["k#0"]["status"] == "incomplete"

    def test_positional_standalone_registration_counts(self, tmp_path):
        r = observe(tmp_path, standalone('Dim("U",False,3)'))
        assert r["out"] == {"out": [3.0]}
        counted(r, 3)

    def test_keyword_dict_standalone_registration_refuses(self, tmp_path):
        r = observe(tmp_path, standalone('Dim("U",**{"size":3})'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", r"re-registered by the explicit Dim registration at line 5 \(" + UNBINDABLE + r": Dim\('U', \*\*\{'size': 3\}\)\)")

    def test_starred_standalone_registration_refuses(self, tmp_path):
        r = observe(tmp_path, prog('    Y=Tensor(O=1)\n    args=("U",False,3)\n    D=Dim(*args)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'))
        assert r["out"] == {"out": [3.0]}
        refused(r, "loop_dim_extent_unresolved", r"explicit Dim\(\.\.\., size=\.\.\.\) at line 6 .* name is not a string literal")

    def test_sizeless_standalone_call_registers_nothing(self, tmp_path):
        r = observe(tmp_path, standalone('Dim("U",True)'))
        assert r["out"] == {"out": [1.0]}
        counted(r, 1)

    @pytest.mark.parametrize("call,want", [
        ('Dim("T", False, 3)', ("T", 3)), ('Dim("T", size=3)', ("T", 3)), ('Dim(name="T", size=3)', ("T", 3)), ('Dim("T", True)', ("T", None)),
        ('Dim("T", 3)', ("T", None)), ('Dim("T", size=None)', ("T", None)), ('Dim("T")', ("T", None)),
        ('Dim("T", **{"size": 3})', ("T", "opaque")), ('Dim("T", *args)', ("T", "opaque")), ('Dim(*args)', (None, "opaque")),
        ('Dim("T", False, 3, 4)', ("T", "opaque")), ('Dim("T", size=3, size2=1)', ("T", "opaque")), ('Dim("T", 3, parallel=True)', ("T", "opaque")),
        ('Dim(nm, size=3)', (None, 3)),
    ])
    def test_parser_binding(self, call, want):
        node = ast.parse(call, mode="eval").body
        name, spec = _dim_call_binding(node, {})
        got = (name, "opaque" if isinstance(spec, tuple) and spec[0] == "opaque" else spec)
        assert got == want, (call, name, spec)
        if got[1] == "opaque":
            assert spec[1] == UNBINDABLE + ": " + ast.unparse(node) and spec[2] == 1, spec

    def test_parser_records_header_and_registration(self, tmp_path):
        code = header('Dim("T",False,3)').replace('    Y=Tensor(O=1)\n', '    Y=Tensor(O=1)\n    D=Dim("U",**{"size":3})\n    E=Dim("V",False,4)\n    F=Dim("W",True)\n')
        p = tmp_path / "probe_einsum_lib.py"; p.write_text(code)
        info = extract_kernel_info(p)["k"]
        assert info["loop_nest"][0]["dim"] == "T" and info["loop_nest"][0]["size"] == 3
        regs = {r["dim"]: r["size"] for r in info["dim_registrations"]}
        assert set(regs) == {"U", "V"} and regs["V"] == 4 and regs["U"][0] == "opaque" and regs["U"][1].startswith(UNBINDABLE), regs


# ---- KB-REFORMER-CONCAT-EXTENT-1 control (the review, acceptance review of an earlier revision) ---------------------------------------------------

B_KB, S_KB, D_KB = 32, 256, 1024


def _isolated_layer(tmp_path, source_text, metadata, node="concat_last#0", name="probe"):
    """The reviewer's isolated replay through the real conversion entry point: ONE kernel module (parsed, never executed) and
    the minimal saved metadata (three tensors, one node) in a dsl directory, translated and analyzed.  Returns the converter
    layer and the analysis."""
    dsl = tmp_path / name / "dsl"; dsl.mkdir(parents=True)
    (dsl / "probe_einsum_lib.py").write_text(source_text)
    (dsl / "probe_einsum_lib.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    gp = translate_with_einsum(dsl, tmp_path / name / "einsum")
    layer = yaml.safe_load(pathlib.Path(gp).read_text())["layers"][node]
    analysis = EinsumGraphAnalyzer().analyze_graph(str(gp), str(tmp_path / name / "analysis"), precision="fp32", copy_graph=False)
    return layer, analysis


# ---- LITE58-EXTENT-399 / LITE229-EXTENT-399 controls (the review, Lite acceptance review of an earlier revision) -----------------------------------

# kernel_source.sha256 the 39969dc9 graphs record for the two functions (the saved-row replays of the acceptance reviews)

def _runtime_program(text, fn, run_model, consts=""):
    """The fixture's kernel renamed ``k`` in a runnable program: header, module constants, the verbatim body, a run_model."""
    body = text[text.index("@kernel"):]
    return H + consts + body.replace(f"def {fn}(", "def k(") + run_model


