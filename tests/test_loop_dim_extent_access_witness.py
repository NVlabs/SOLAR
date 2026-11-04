# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P2-203-SCRATCH-EXTENT-1 (the review, acceptance review of the Lite 203 revised candidate on an earlier revision; issue #58 note 78936488):
the converter's narrow-access WITNESS must be bound to the tensor the loop writes as its output -- the kernel's RETURNED
allocation -- never to whichever tensor the extracted loop body happens to access first.

``einsum_converter._kernel_loop_iterations`` sizes every ``Dim("X")`` loop from the runtime registry binding and keeps the
positional accesses only as a witness: a bare access of the loop variable on an axis narrower than the bound extent is
refused as ``loop_dim_extent_mismatch`` (an earlier change round 3, TARGET-DSL-DIM-EXTENT-13).  The witness knows the shapes of the
recorded inputs by parameter name and ONE local name: the kernel's output shape.  Up to an earlier revision that local name was the
FIRST tensor accessed in the loop body.  In the Lite 203 candidate's ``topk_select`` the first access is the E-wide
scratch ``QV`` [2560, 128] while the returned ``Y`` is [2560, 6]: the synthetic six-element shape on ``QV`` made the
E loop look like it indexed a six-element axis and the kernel was refused -- a false diagnostic, not a candidate defect.

The repair binds the witness to the returned allocation (``_returned_allocation_index`` + the returned name, the same
identity rules as the registry mirror: return-identity-5 / RETURN-REBIND-9) and binds nothing when that identity is
not certain.  Every other rule -- the explicit size at the call, nested registrations, typed refusals, the witness on a
genuinely narrow axis -- is unchanged.

Rounds 2-8 (the review, an earlier fix findings 1-21) settle WHICH access the witness speaks for.  Object identity (rounds 2-3): the
binding live at the access, by ``name_bindings`` sequence, through plain aliases, over the parser-normalized statements;
unproved borrows nothing.  Loop identity (round 8, WITNESS_MODEL v2 -- the scope decision after rounds 4-7 had
grown an evaluation-order / scope emulator): a bare index component witnesses its loop ONLY when the spelling is bound
anywhere in the kernel solely as a Dim-loop target, no Dim loop over it nests in another over it, the statement holds only
names / constants / subscripts / arithmetic / single comparisons / attribute loads / tuples / calls to names bound nowhere in
the kernel, every enclosing loop is a Dim loop, the kernel has no class statement, no nested def or lambda mentioning the
spelling, no dynamic-namespace spelling and no call to a kernel-bound name, and some Dim loop over the spelling lexically
encloses the statement (its innermost one is the governor).  Every candidate outside that predicate gets NO witness and a
typed ``witness_unmodelled:<first failing condition>`` note.  Round 9 (WITNESS_MODEL v3, the B′ ruling): the witness is a
geometric consistency cross-check, not the source of completeness -- a declined witness is REPORTED in the informational node /
layer fields ``witness_unmodelled`` and ``witness_coverage`` (partial / full / none) and changes no count, no
``loop_iterations_complete``, no ``incomplete_reason`` and no billing; a ``for ... else`` orelse loop is a sibling segment, not a
nested iteration.  The accepted, visible loss: a genuine narrow access outside the predicate is counted, complete and flagged, not
refused.  Legacy records (no return information) keep the first-access SHAPE
convention for the pooled-maximum fallback only and receive no governor (their genuinely narrow returned-first program is a
documented boundary, not a refusal).  Coverage: ``TestWitnessModelV5`` follows the model's §5 table; the reviewer's eleven probe
files run verbatim in ``TestReview4FilesVerbatim`` against an explicit by-design deviation table.

Vendored verbatim (byte-for-byte, sha256-pinned): the reviewer's two reduced converter-only controls (``scratch_first``:
T/E initialization + T/K copy, no while / if, 670,720 assignment executions; ``returned_first``: the same plus an outer
``Y[t, 0] = 0`` output initialization, 673,280) with the candidate's saved metadata (``0bea4f4c...``), and the revised
candidate's ``topk_select`` kernel segment (candidate ``19c3a3e1...``, parser source_sha256 ``51ac6863...``) with its
minimal saved metadata.  the reviewer's generator ``control_extent.py`` (sha256 65fb0f00...) and record ``EXTENT_CONTROL.json``
(9271d772...) are not vendored (they carry host-specific paths); the delivery evidence keeps them.
"""
import hashlib
import importlib.util
import pathlib
import re

import numpy as np
import pytest
import yaml

from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.einsum_lib import clear_dims, stop_trace
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "loop_dim_extent" / "followup-203"


CONTROL_LOOP_EXPR = "QV[t, e] = S[t, e]; QI[t, e] = e; Y[t, r] = QI[t, r]"

_spec = importlib.util.spec_from_file_location("_r5", ROOT / "tests" / "test_loop_dim_extent_round5.py")
_r5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r5)
observe, counted, refused, LOOP_REFUSAL_CODES = _r5.observe, _r5.counted, _r5.refused, _r5.LOOP_REFUSAL_CODES


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def _isolated_layer(tmp_path, source_text, metadata, node, name="probe"):
    """One kernel module (parsed, never executed) + minimal saved metadata, translated and analyzed (the round-7 recipe)."""
    dsl = tmp_path / name / "dsl"; dsl.mkdir(parents=True)
    (dsl / "probe_einsum_lib.py").write_text(source_text)
    (dsl / "probe_einsum_lib.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    gp = translate_with_einsum(dsl, tmp_path / name / "einsum")
    layer = yaml.safe_load(pathlib.Path(gp).read_text())["layers"][node]
    analysis = EinsumGraphAnalyzer().analyze_graph(str(gp), str(tmp_path / name / "analysis"), precision="fp32", copy_graph=False)
    return layer, analysis


# ---- the reviewer's reduced controls, verbatim --------------------------------------------------------------------------------------

# ---- author controls (a)-(c): tiny executed programs, runtime assignments vs the count ----------------------------------------
H = 'from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
SIG = '@kernel\ndef k(X:Tensor("T,E"),Z:Tensor("T,K"))->Tensor("T,K"):\n'
RUN = ('\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T,E")\n'
       '    z=Tensor.from_numpy(inputs["z"],"T,K")\n    return {"out":k(x,z).data}\n')
T, E, K = 4, 8, 3
X = np.arange(T * E, dtype=np.float32).reshape(T, E) + 1.0
Z = np.arange(T * K, dtype=np.float32).reshape(T, K) * 10.0
INPUTS = {"x": X, "z": Z}


def prog(body):
    return H + SIG + body + RUN


class TestAuthorControls:
    def test_a_k_wide_output_reading_an_e_wide_scratch_is_counted(self, tmp_path):
        """(a) The loop fills an E-wide scratch Q (first access of the body) then writes the K-wide returned Y from it: counted at
        the runtime's T*E + T*K = 44 assignments (the output copy term is K*T = 12), no refusal, output = X[:, :K] + Z.
        an earlier revision bound Q to the output shape [T, K] and refused the E loop (``only 3 elements``)."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
            '        for r in Dim("K"):\n            Y[t,r]=Q[t,r]+Z[t,r]\n    return Y\n'), inputs=INPUTS)
        assert "error" not in rec, rec.get("error")
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] + Z)
        L = counted(rec, T * E + T * K)
        assert L["loop_other_statements"] == 44 and L["loop_mac_statements"] == 0, L
        assert rec["analysis"]["total"]["other_ops"] == 44 and rec["analysis"]["total"]["macs"] == 0, rec["analysis"]["total"]

    def test_a_alias_of_the_returned_allocation(self, tmp_path):
        """(a') The same loop returning a plain alias of Y (``Out = Y; return Out``): the witness follows the alias to the
        returned allocation (RETURN-REBIND-9 identity) -- counted 44, no refusal."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
            '        for r in Dim("K"):\n            Y[t,r]=Q[t,r]+Z[t,r]\n    Out=Y\n    return Out\n'), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] + Z)
        counted(rec, 44)

    def test_b_first_accessed_tensor_is_the_output(self, tmp_path):
        """(b) The body's first access IS the returned Y: unchanged behaviour -- counted T*K = 12, no refusal."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    for t in Dim("T"):\n        for r in Dim("K"):\n'
            '            Y[t,r]=X[t,r]*Z[t,r]\n    return Y\n'), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] * Z)
        L = counted(rec, T * K)
        assert L["loop_other_statements"] == 12, L

    def test_b_scratch_first_but_indexed_consistently(self, tmp_path):
        """(b') A scratch accessed first but only on axes as wide as the loop's binding (the K loop writes Q[t, r] then Y[t, r]):
        counted 2*T*K = 24 on both trees -- the first-access heuristic happened to be harmless here and the repair keeps it so."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1],fill=1.0)\n'
            '    for t in Dim("T"):\n        for r in Dim("K"):\n            Q[t,r]=X[t,r]\n            Y[t,r]=Q[t,r]*Z[t,r]\n    return Y\n'),
            inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] * Z)
        counted(rec, 2 * T * K)


def _probe_metadata():
    return {"tensors": {"x": {"dims": {"T": T, "E": E}, "dtype": "fp32", "role": "input"},
                        "z": {"dims": {"T": T, "K": K}, "dtype": "fp32", "role": "input"},
                        "k#0": {"dims": {"T": T, "K": K}, "dtype": "fp32", "role": "input", "origin": "internal-constant"}},
            "graph": {"k#0": {"type": "kernel", "function": "k", "inputs": [{"tensor": "x", "from": "input"}, {"tensor": "z", "from": "input"}],
                              "output": "k#0", "call_site_line": 20, "input_order": "signature", "registration_order": [0, 1]}}}


class TestGenuineNarrowAxisStillRefuses:
    """(c) Genuine narrow-axis violations keep their typed refusal (parsed, never executed: the programs would index out of range)."""

    def test_c_e_loop_writes_the_k_wide_output(self, tmp_path):
        """The E loop indexes the returned Y [T, K] bare: ``loop_dim_extent_mismatch`` (bound 8, axis of 3) -- on both trees."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    for t in Dim("T"):\n'
                               '        for e in Dim("E"):\n            Y[t,e]=X[t,e]\n    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L
        assert A["layers"]["k#0"]["status"] == "incomplete", A["layers"]["k#0"]

    def test_c_e_loop_reads_a_k_wide_parameter(self, tmp_path):
        """The E loop indexes the recorded input Z [T, K] bare while writing an E-wide scratch: refused on both trees."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=Z[t,e]\n'
                               '        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L
        assert A["layers"]["k#0"]["status"] == "incomplete", A["layers"]["k#0"]

    def test_c_returned_tensor_narrow_even_when_not_accessed_first(self, tmp_path):
        """The scratch is accessed first (``Q[t, e] = X[t, e]``) and the returned Y [T, K] is then indexed bare by ``e``: the witness
        on the RETURNED allocation refuses (an earlier revision refused this one too, through the mis-bound scratch)."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n            Y[t,e]=Q[t,e]\n'
                               '    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L
        assert A["layers"]["k#0"]["status"] == "incomplete", A["layers"]["k#0"]


# ---- (d) the Lite 203 revised candidate's topk_select ---------------------------------------------------------------------------

# ---- round 2 (the review NO on an earlier revision): binding epochs ---------------------------------------------------------------------------
# witness-lifetime-1: a return-time identity is not a global fact about a name; access-alias-2: every plain alias of
# the returned allocation live at the access is a witness.  the reviewer's ``checks-c5fe21a6/test_witness_edges.py`` (sha256 901d3207...)
# is vendored verbatim and run byte-for-byte: 20 cases (14 passed / 6 failed on an earlier revision; 20 on this head).
R4 = FX / "checks-c5fe21a6"
R4_PIN = "5c1c4225d8e1efb9ff2e89c3fa47b81b92ade016a16be57a71503234f7e1570d"


class TestReview4WitnessEdgesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's lifetime / alias-access / identity-boundary matrix byte-for-byte, in the layout it expects (it imports this module
        from ``source/head/solar/tests/``): 20 passed, and read back from its own records: the four lifetime cases count the runtime's 64
        assignments, the two alias-access cases refuse ``loop_dim_extent_mismatch``."""
        import json, os, subprocess, sys
        src = R4 / "test_witness_edges.py"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == R4_PIN
        run = tmp_path / "run"; (run / "runs").mkdir(parents=True); tests = run / "source" / "head" / "solar" / "tests"; tests.mkdir(parents=True)
        (run / "test_witness_edges.py").write_bytes(src.read_bytes())
        for name in ("test_loop_dim_extent_access_witness.py", "test_loop_dim_extent_round5.py"):
            (tests / name).write_bytes((ROOT / "tests" / name).read_bytes())
        (tests / "fixtures").mkdir(); os.symlink(ROOT / "tests" / "fixtures" / "loop_dim_extent", tests / "fixtures" / "loop_dim_extent")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_witness_edges.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "20 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "witness-head").iterdir()}
        for name in ("later_allocation_same_name", "earlier_return_name_rebound", "original_alloc_name_rebound", "returned_name_different_epoch"):
            r = recs[name]; L = r["observation"]["layer"]
            assert r["runtime_assignment_count"] == 64 and L["loop_iterations"] == 64 and L["loop_iterations_complete"] is True, (name, L)
        for name in ("access_alias_return_original", "access_middle_alias_return_chain"):
            L = recs[name]["layer"]
            assert "loop_dim_extent_mismatch" in L and L["loop_iterations_complete"] is False, (name, L)


class TestBindingEpochs:
    """Author controls for the epoch rule.  Valid programs are executed (runtime assignments = the count); the two inconsistent
    programs are parsed only."""

    def test_rebinding_before_the_loop_borrows_the_live_allocation(self, tmp_path):
        """``Q = Tensor(T, K); Q = Tensor(T, E); <loop writes Q[t, e]>; return Q``: the access denotes the second allocation, which
        IS the returned one -- borrowed [T, E], consistent with the E binding, counted 32 (the stale first allocation is never used)."""
        rec = observe(tmp_path, prog(
            '    Q=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    return Q\n'), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X)
        counted(rec, T * E)

    def test_conditional_rebinding_before_the_loop_borrows_nothing(self, tmp_path):
        """``Q = Tensor(T, E); if X[0,0] < 0: Q = Y`` (not taken at run time) then the E loop writes Q: the live binding of Q is not
        proved, so the access borrows nothing -- counted at the runtime's T*E + T*K = 44, no refusal, even though Y [T, K] is the
        returned allocation and Q MIGHT have been it."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n    if X[0,0]<0:\n        Q=Y\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]+Z[t,r]\n    return Y\n'),
            inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] + Z)
        counted(rec, T * E + T * K)

    def test_alias_bound_inside_the_loop_borrows_nothing(self, tmp_path):
        """``for t: A = Y; for k: A[t, k] = ...``: the alias is bound inside a loop enclosing the access (loop-carried) -- not proved,
        nothing borrowed; the K loop is still sized by its binding: counted at the runtime's T (the alias statement, once per t) +
        T*K = 16 assignment executions, no refusal."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    for t in Dim("T"):\n        A=Y\n        for k in Dim("K"):\n            A[t,k]=X[t,k]*Z[t,k]\n    return Y\n'),
            inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] * Z)
        counted(rec, T + T * K)

    def test_alias_captured_before_rebinding_is_not_the_returned_allocation(self, tmp_path):
        """``Y = Tensor(T, K); A = Y; Y = Tensor(T, E); <loop writes A[t, e]>; return Y``: A denotes the FIRST allocation, the return
        the second.  The witness has no shape for A (it is not the returned allocation) and refuses nothing -- ruled conservative:
        the output shape is borrowed only for the returned allocation's epoch, never applied to another object.  Parsed only (the
        program itself indexes A out of range)."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    A=Y\n    Y=Tensor(T=X.shape[0],E=X.shape[1])\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Y\n', _probe_metadata(), "k#0")
        assert "loop_dim_extent_mismatch" not in L and L.get("loop_iterations_complete") is True and L["loop_iterations"] == T * E, L

    def test_alias_of_the_returned_allocation_captured_before_an_unrelated_rebinding_refuses(self, tmp_path):
        """``Y = Tensor(T, K); A = Y; Q = Tensor(T, E); Q = Tensor(T, E); <loop writes A[t, e]>; return Y``: A is a plain alias of the
        returned allocation in the same epoch (the rebinding of Q is unrelated) -- the narrow access through A refuses."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    A=Y\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                               '    Q=Tensor(T=X.shape[0],E=X.shape[1])\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=Q[t,e]\n    return Y\n',
                               _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L
        assert A["layers"]["k#0"]["status"] == "incomplete", A["layers"]["k#0"]

    def test_epochs_helper_boundaries(self, tmp_path):
        """Direct probes of ``_access_witness_epochs``: the borrow set names only the returned allocation's epoch and its live plain
        aliases; a locally rebound parameter spelling is shadowed; a record without the source segment borrows nothing."""
        import types
        from solar.einsum_dsl.einsum_converter import _access_witness_epochs
        code = H + SIG + ('    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    A=Y\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n    X=Q\n'
                          '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n        for r in Dim("K"):\n            A[t,r]=Q[t,r]\n'
                          '    Q=Tensor(T=X.shape[0],K=Z.shape[1])\n    return Y\n')
        p = tmp_path / "probe_einsum_lib.py"; p.write_text(code)
        info = extract_kernel_info(p)["k"]; k = types.SimpleNamespace(**info)
        stmts = [s.strip() for s in info["loop_expr"].split(";")]
        borrow, shadowed = _access_witness_epochs(k, stmts)
        assert borrow == [set(), {"A"}] and shadowed == [{"Q", "X"}, {"A", "Q"}], (borrow, shadowed)
        k2 = types.SimpleNamespace(**{**info, "source_text": None})
        borrow2, shadowed2 = _access_witness_epochs(k2, stmts)       # conservative: nothing borrowed; each statement's accessed locally-bound names shadowed
        assert borrow2 == [set(), set()] and shadowed2 == [{"Q", "X"}, {"A", "Q"}], (borrow2, shadowed2)   # round 3: per statement (round 2 shadowed the union everywhere)


# ---- round 3 (the review NO on an earlier revision): binding SEQUENCE, not line; the parser's normalized statements --------------------------------
# same-line-epoch-4: same-line earlier bindings are live for later captures (name_bindings.seq orders them); an earlier fix-SOURCE-
# NORMALIZATION-5: access sites are matched through the parser's normalized body (AnnAssign is Assign), statement by statement, and an
# unmatched statement is unproved on its own.  the reviewer's ``checks-a1189d62/test_epoch_edges.py`` (sha256 bb2ea83a...) is vendored
# verbatim: 22 cases (16 passed / 6 failed on an earlier revision; 22 on this head).
R4E = FX / "checks-a1189d62"
R4E_PIN = "a337538f55d3314d6596659ecc3a7559bba60c505c4cd582020d4960407bd358"


class TestReview4EpochEdgesVerbatim:
    def test_file(self, tmp_path):
        """the reviewer's round-2 epoch matrix byte-for-byte (same layout as the witness file): 22 passed; read back from its records, the
        three same-line valid programs count the runtime's 64 and the four same-line / annotated narrow accesses refuse."""
        import json, os, subprocess, sys
        src = R4E / "test_epoch_edges.py"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == R4E_PIN
        run = tmp_path / "run"; (run / "runs").mkdir(parents=True); tests = run / "source" / "head" / "solar" / "tests"; tests.mkdir(parents=True)
        (run / "test_epoch_edges.py").write_bytes(src.read_bytes())
        for name in ("test_loop_dim_extent_access_witness.py", "test_loop_dim_extent_round5.py"):
            (tests / name).write_bytes((ROOT / "tests" / name).read_bytes())
        (tests / "fixtures").mkdir(); os.symlink(ROOT / "tests" / "fixtures" / "loop_dim_extent", tests / "fixtures" / "loop_dim_extent")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_epoch_edges.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "22 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "epochs-head").iterdir()}
        for name in ("same_line_rebind_then_capture", "same_line_alias_rebind_then_capture", "same_line_alias_chain"):
            r = recs[name]; L = r["observation"]["layer"]
            assert r["runtime_assignment_count"] == 64 and L["loop_iterations"] == 64 and L["loop_iterations_complete"] is True, (name, L)
        for name in ("same_line_alloc_and_alias", "same_line_switch_to_returned_alias", "annotated_direct_access", "annotated_scalar_in_same_body"):
            L = recs[name]["layer"]
            assert "loop_dim_extent_mismatch" in L and L["loop_iterations_complete"] is False, (name, L)


class TestEpochSequenceAndNormalization:
    def test_same_line_alias_chain_of_a_scratch_counts(self, tmp_path):
        """``Q = Tensor(T, E); A = Q; B = A`` on ONE line, the loop writes ``B[t, e]``, ``Y`` [T, K] is returned: B resolves through the
        same-line captures to the scratch (not the returned allocation) -- nothing borrowed, counted at the runtime's 32, no refusal.
        (an earlier revision resolved the chain by line and lost the same-line sources.)"""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1]); A=Q; B=A\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            B[t,e]=X[t,e]\n    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n    return Y\n'),
            inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K])
        counted(rec, T * E + T * K)

    def test_same_line_alias_chain_of_the_returned_allocation_refuses(self, tmp_path):
        """``Y = Tensor(T, K); A = Y; B = A`` on ONE line, the E loop writes ``B[t, e]``, ``Y`` returned: every capture on that line is
        ordered by sequence, B is the returned allocation -- refused."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1]); A=Y; B=A\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            B[t,e]=X[t,e]\n    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L

    def test_annotated_scalar_keeps_the_other_sites_witnesses(self, tmp_path):
        """A valid program whose loop body has an annotated scalar (``v: float = X[t, e]``) before the scratch write and the K-wide
        output copy: the parser's normalized body is matched statement by statement, the witness of the returned ``Y`` is kept,
        nothing is refused and the count is the runtime's 2*T*E + T*K = 76 (an earlier revision dropped every witness of such a body)."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            v: float = X[t,e]\n            Q[t,e]=v\n'
            '        for r in Dim("K"):\n            Y[t,r]=Q[t,r]+Z[t,r]\n    return Y\n'), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] + Z)
        counted(rec, 2 * T * E + T * K)

    def test_annotated_narrow_write_refuses(self, tmp_path):
        """``Y[t, e]: float = X[t, e]`` with Y [T, K] returned: the annotated statement IS the plain assignment to the parser and to the
        witness -- refused like the plain form."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]: float = X[t,e]\n    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L and "loop_dim_extent_mismatch" in L, L

    def test_unmatched_statement_is_unproved_alone(self, tmp_path):
        """Direct probe: when ONE loop_expr statement cannot be matched to the segment (its text altered here), only that statement's
        accesses lose their witness; the other statements keep theirs; a binding on the access's own line is unproved."""
        import types
        from solar.einsum_dsl.einsum_converter import _access_witness_epochs
        code = H + SIG + ('    Y=Tensor(T=X.shape[0],K=Z.shape[1]); A=Y\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                          '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n        for r in Dim("K"):\n            A[t,r]=Q[t,r]\n    return Y\n')
        p = tmp_path / "probe_einsum_lib.py"; p.write_text(code)
        info = extract_kernel_info(p)["k"]; stmts = [s.strip() for s in info["loop_expr"].split(";")]
        assert _access_witness_epochs(types.SimpleNamespace(**info), stmts)[0] == [set(), {"A"}]
        altered = [stmts[0], stmts[1].replace("A[t, r]", "A[t, r + 0]")]              # the second statement no longer matches the segment
        borrow, shadowed = _access_witness_epochs(types.SimpleNamespace(**info), altered)
        assert borrow == [set(), set()] and shadowed == [{"Q"}, {"A", "Q"}], (borrow, shadowed)
        altered = [stmts[0].replace("Q[t, e]", "Q[t, e + 0]"), stmts[1]]              # the first does not match; A's witness survives
        borrow, shadowed = _access_witness_epochs(types.SimpleNamespace(**info), altered)
        assert borrow == [set(), {"A"}], (borrow, shadowed)
        one_line = code.replace('        for r in Dim("K"):\n            A[t,r]=Q[t,r]\n', '        for r in Dim("K"): A=Y; A[t,r]=Q[t,r]\n')
        p.write_text(one_line); info = extract_kernel_info(p)["k"]; stmts = [s.strip() for s in info["loop_expr"].split(";")]
        borrow, shadowed = _access_witness_epochs(types.SimpleNamespace(**info), stmts)
        assert len(stmts) == 3 and all(not b for b in borrow), (stmts, borrow, shadowed)   # ``A = Y`` on the access's own line (and inside the loop): unproved


# ---- round 4 (the review NO on an earlier revision): the witness belongs to its producing loop ------------------------------------------------------
# witness-site-6 (baseline-owned): the narrow-access minima were pooled by index SPELLING across the whole kernel, so a three-wide
# access in a sibling ``for e in Dim("K")`` refused the eight-wide ``for e in Dim("E")``.  The witness is now keyed by the governing Dim
# loop's site (entry line, dim name) and compared only with that loop's own binding.  the reviewer's ``checks-5d13883d/test_sequence_sites.py``
# (d1c836da...) and ``test_site_controls.py`` (91ba8915...) are vendored verbatim: 24 cases (22 passed / 2 failed on an earlier revision).
R4S = FX / "checks-5d13883d"
R4S_PINS = {"test_sequence_sites.py": "91059277e7b8ebbdf6a8a8cc75bf23d5958ce2c7a0cb1c83eb015a525dadfefb",
            "test_site_controls.py": "0248c7e8d7c86f301862d64b8f220ace63538051d21b3e8186e633b30acb2c5b"}

class TestReview4SequenceSitesVerbatim:
    def test_files(self, tmp_path):
        """the reviewer's round-3 probes + metamorphic controls byte-for-byte (they import their round-2 epoch file, vendored beside them):
        24 passed; read back from their records: the two sibling-loop programs count the runtime's 44 (plain and first-site-annotated),
        their renamed controls 44, and the narrow duplicate refuses in both forms."""
        import json, os, subprocess, sys
        run = tmp_path / "run"; (run / "runs").mkdir(parents=True); tests = run / "source" / "head" / "solar" / "tests"; tests.mkdir(parents=True)
        for name, digest in R4S_PINS.items():
            src = R4S / name; assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, name
            (run / name).write_bytes(src.read_bytes())
        (run / "test_epoch_edges.py").write_bytes((R4E / "test_epoch_edges.py").read_bytes())   # imported by test_sequence_sites.py
        for name in ("test_loop_dim_extent_access_witness.py", "test_loop_dim_extent_round5.py"):
            (tests / name).write_bytes((ROOT / "tests" / name).read_bytes())
        (tests / "fixtures").mkdir(); os.symlink(ROOT / "tests" / "fixtures" / "loop_dim_extent", tests / "fixtures" / "loop_dim_extent")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_sequence_sites.py", "test_site_controls.py", "-q", "-p", "no:cacheprovider", "-rf"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "24 passed" in res.stdout, res.stdout[-4000:] + res.stderr[-800:]
        recs = {p.name: json.loads((p / "record.json").read_text()) for p in (run / "runs" / "sequence-head").iterdir()}
        for name in ("duplicate_sites_different_loop_extents", "normalized_duplicate_different_epochs", "renamed-control-False", "renamed-control-True"):
            L = recs[name]["observation"]["layer"]
            assert L["loop_iterations"] == 44 and L["loop_iterations_complete"] is True, (name, L)
        for name in ("narrow-duplicate-False", "narrow-duplicate-True"):
            assert "loop_dim_extent_mismatch" in recs[name]["layer"], name


class TestWitnessSites:
    SIBLINGS = ('    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n    A=Q\n'
                '    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]{ann}=X[t,e]\n'
                '    A=Y\n    for t in Dim("T"):\n        for e in Dim("K"):\n            A[t,e]=X[t,e]\n    return Y\n')

    @pytest.mark.parametrize("annotated", [False, True])
    def test_sibling_loops_same_spelling_different_extents_count(self, tmp_path, annotated):
        """Two sibling nests spell their inner variable ``e``: the first over Dim("E") writes the eight-wide scratch, the second over
        Dim("K") writes the three-wide returned Y.  Each site witnesses only its own loop: counted at the runtime's 32 + 12 = 44,
        output X[:, :K], no refusal (an earlier revision, an earlier revision and the target refused the E loop with the K site's three-wide access);
        the same with the first site annotated."""
        rec = observe(tmp_path, prog(self.SIBLINGS.format(ann=": float" if annotated else "")), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K])
        counted(rec, T * E + T * K)

    def test_nested_loops_reuse_a_spelling_at_different_depths(self, tmp_path):
        """``for t: for e in Dim("E"): Q[t, e] = ...`` then a sibling ``for e in Dim("K"): for t: Y[t, e] = Q[t, e]`` -- the spelling ``e``
        governs Dim("E") at depth 1 in the first nest and Dim("K") at depth 0 in the second: counted 32 + 12 = 44, no refusal."""
        rec = observe(tmp_path, prog(
            '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
            '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
            '    for e in Dim("K"):\n        for t in Dim("T"):\n            Y[t,e]=Q[t,e]*Z[t,e]\n    return Y\n'), inputs=INPUTS)
        assert np.array_equal(np.asarray(rec["out"]["out"]), X[:, :K] * Z)
        counted(rec, T * E + T * K)

    def test_genuine_narrow_access_in_a_sibling_loop_still_refuses(self, tmp_path):
        """The first nest is valid (``Q[t, e]`` over Dim("E")); the SECOND nest, also over Dim("E"), writes the three-wide returned Y
        bare: its own site refuses ``loop_dim_extent_mismatch`` (8 vs 3) -- the witness is scoped per loop, not suppressed."""
        L, A = _isolated_layer(tmp_path, H + SIG + '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
                               '    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=Q[t,e]\n    return Y\n', _probe_metadata(), "k#0")
        assert L.get("loop_iterations_complete") is False and "loop_iterations" not in L, L
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L


# ---- rounds 5-8: the reviewer's remaining probe files (verbatim) and the round-8 minimal witness model -------------------------------------
R4F = FX / "checks-5accce21"
R4L = FX / "checks-90ca6d10"
R4V = FX / "checks-481962ed"
R4M = FX / "checks-c62a072f"
R4C = FX / "checks-09318804"
R4A = FX / "checks-527b034d"
R4_FILES = {                                         # every the review probe file, byte-for-byte; the files import each other in this chain
    R4 / "test_witness_edges.py": "5c1c4225d8e1efb9ff2e89c3fa47b81b92ade016a16be57a71503234f7e1570d",
    R4E / "test_epoch_edges.py": "a337538f55d3314d6596659ecc3a7559bba60c505c4cd582020d4960407bd358",
    R4S / "test_sequence_sites.py": "91059277e7b8ebbdf6a8a8cc75bf23d5958ce2c7a0cb1c83eb015a525dadfefb",
    R4S / "test_site_controls.py": "0248c7e8d7c86f301862d64b8f220ace63538051d21b3e8186e633b30acb2c5b",
    R4F / "test_site_flow.py": "8599b744b2ddc615bff35971deea7c6a461a0369360b4197f1b2a9f2783eba92",
    R4L / "test_live_binding.py": "7d4d310c5aff0984fa14650d8b416a03251794a9d9c54afda01e9e70004322de",
    R4L / "test_binding_controls.py": "9b3eff3fed2840d62675239bcf77e49e8f11868a165b442c0bb542a628187a9d",
    R4V / "test_eval_order.py": "73d59da4c2236212c3ede6ccb769af93db32ab10ea7b62f22a066e1b79146cd6",
    R4V / "test_legacy_identity.py": "ab68c2ff479e570184785b32193a77994e5d6ce23a393a8601363ef98eeb25d4",
    R4M / "test_model_boundary.py": "7e37262f71c855aabd6d08aa08dcd2c110c2b790b198c36e2920b31645d53696",
    R4M / "test_class_declaration_scope.py": "9de98a471bca1cbaa6b8d7c798bd8868c1afacf15015a287ee37fb1b6c9c9071",
    R4C / "test_minimal_control_flow.py": "e20e3bd599ee898af9c1643d32b039a650c4dc641a4018fc99e8f9918aa78f7f",
    R4A / "test_round10_transfer.py": "b28b3ceb8088517b0166f8d56bbb442c88051763445ebf6f68560d590e4defd8",     # round 11: 5 ancestor + 10 return-policy + 4 coverage controls, all pass
    R4A / "test_ancestor_controls.py": "b61d1456bf51aa4ec6feac6a815fbf05251223f71c710191b28cdbc0cf9377bf",    # round 11: 2 non-vacuous + the separate-loop control (still refuses), all pass
}
# the review cases that fail BY DESIGN under WITNESS_MODEL v4: each is a program outside the modelled subset (or a frozen expectation of an
# earlier round's consequence).  Per case: the excluding condition and the MEASURED tuple read from the record the reviewer's own file saved
# for that case on this head -- (counted, refused, loop-complete, witness_coverage) -- which TestReview4FilesVerbatim re-reads and asserts
# (BOUNDARY-CLAIMS-24: no slogan; 25 of the 49 are loop-complete, 24 carry the counter's own class / def / comprehension omissions).
BY_DESIGN = {
    # (b) the index spelling is bound otherwise (walrus / chained or tuple target / assignment / comprehension target / lambda parameter): typed boundary, not the refusal the frozen assertion expects
    "test_binding_controls.py::test_access_before_rebinding_retains_refusal[Y[t,j]+=(j:=0)]": ("b", True, False, True, "partial"),
    "test_binding_controls.py::test_access_before_rebinding_retains_refusal[Y[t,j]=j=0]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[aug_index_before_later_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[dict_key_read_before_value_walrus_control]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[dict_read_before_later_value_walrus_control]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[dict_value_before_later_key_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[index_annotated_write_before_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[index_read_before_later_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[index_write_before_later_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[repeated_spelling_before_walrus]": ("b", True, False, True, "partial"),
    "test_eval_order.py::test_access_uses_earlier_value[rhs_read_before_walrus_control]": ("b", True, False, True, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[dict_comp_local]": ("b", True, False, False, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[generator_local]": ("b", True, False, False, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[lambda_parameter]": ("b", True, False, True, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[list_comp_local]": ("b", True, False, False, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[set_comp_local]": ("b", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_modelled[comp_first_iter_later_target]": ("b", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_modelled[comp_first_iter_local_target]": ("b", True, False, False, "partial"),
    # B′ (round 9): frozen assertions that expect the round-7/8 consequence of a typed note (loop_iterations_complete False, note inside loop_iterations_omitted)
    "test_model_boundary.py::test_narrow_unmodelled[eval_alias]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[exec_attribute]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[generator_next_read]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[generator_send_read]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[lambda_called_read]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[locals_alias]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[module_setattr]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[await_read]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[eval_direct]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[exec_direct]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[generator_walrus]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_typed_controls[globals_store]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[locals_direct]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_typed_controls[yield_read]": ("B\u2032", True, False, True, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[called_nonlocal]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[class_cell_method_nonlocal]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[decorator_call_nonlocal]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[generator_send_nonlocal]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[lambda_calls_nonlocal]": ("B\u2032", True, False, False, "partial"),
    "test_model_boundary.py::test_valid_unmodelled[star_call_nonlocal]": ("B\u2032", True, False, False, "partial"),
    # (e) class statement / nested def or lambda mentioning the spelling / opaque call -- or, for nested_def_called_read, no candidate in loop_expr at all
    "test_class_declaration_scope.py::test_nested_class_decl_stays_local[child_decl_parent_default]": ("e", True, False, False, "partial"),
    "test_class_declaration_scope.py::test_nested_class_decl_stays_local[child_decl_parent_local]": ("e", True, False, False, "partial"),
    "test_class_declaration_scope.py::test_nested_class_decl_stays_local[sibling_decl_sibling_local]": ("e", True, False, False, "partial"),
    "test_eval_order.py::test_class_local_preserves_outer[class_default_outer]": ("e", True, False, False, "partial"),
    "test_eval_order.py::test_class_local_preserves_outer[class_local]": ("e", True, False, False, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[lambda_local_walrus]": ("e", True, False, True, "partial"),
    "test_live_binding.py::test_genuine_narrow_after_local_binding[nested_def_local]": ("e", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_modelled[lambda_never_called]": ("e", True, False, True, "partial"),
    "test_model_boundary.py::test_narrow_modelled[nonlocal_never_called]": ("e", True, False, False, "partial"),
    "test_model_boundary.py::test_narrow_unmodelled[nested_def_called_read]": ("e (no candidate: the access is not a loop-body statement)", True, False, False, "partial"),
    # v4 coverage (SCOPE-CONTRACT-25): an access the witness did not inspect makes coverage partial, never full
    "test_minimal_control_flow.py::test_domain_controls[nested_def_narrow_no_candidate]": ("coverage (v4): the access inside the nested def was not inspected -> partial, not full", True, False, False, "partial"),
}


_RECORD_ALIASES = {"test_binding_controls.py::test_access_before_rebinding_retains_refusal[Y[t,j]=j=0]": "access-first-10",      # the review names these by len(tail)
                   "test_binding_controls.py::test_access_before_rebinding_retains_refusal[Y[t,j]+=(j:=0)]": "access-first-14"}


def _review4_record_layer(runs, tid):
    """The graph layer inside the record the reviewer's own file saved for case *tid* (their files name records by the case id, sometimes
    with a kind prefix such as ``narrow-`` / ``class-``; two binding-control records are named by the tail's length)."""
    import json as _json
    fname, rest = tid.split("::"); m = re.search(r"\[(.*)\]$", rest); param = _RECORD_ALIASES.get(tid) or (m.group(1) if m else rest)
    def layer_of(obj):
        if isinstance(obj, dict):
            if "loop_iterations_complete" in obj or "witness_coverage" in obj or "loop_dim_extent_mismatch" in obj: return obj
            for v in obj.values():
                r = layer_of(v)
                if r is not None: return r
        elif isinstance(obj, list):
            for v in obj:
                r = layer_of(v)
                if r is not None: return r
        return None
    word = fname[len("test_"):-3].split("_")[0]
    cands = [p for p in runs.rglob("*.json") if p.stem == param or p.stem.endswith("-" + param) or p.parent.name == param or p.parent.name.endswith("-" + param)]
    pref = [p for p in cands if word in p.relative_to(runs).parts[0]]
    for p in pref + [c for c in cands if c not in pref]:
        try: L = layer_of(_json.loads(p.read_text()))
        except Exception: L = None
        if L is not None: return L
    return None


class TestReview4FilesVerbatim:
    def test_files(self, tmp_path):
        """All fourteen the review probe files byte-for-byte in one pytest run (their import chain intact, the author module under
        ``source/head/solar/tests``): every failure is one of the BY_DESIGN cases above -- a program outside the §1 predicate answered
        with the B′ typed boundary (counted, complete, flagged) -- and nothing else fails; the 40 helper probes of the removed
        emulation skip (``hasattr`` guards)."""
        import json, os, subprocess, sys
        run = tmp_path / "run"; (run / "runs").mkdir(parents=True); tests = run / "source" / "head" / "solar" / "tests"; tests.mkdir(parents=True)
        for src, digest in R4_FILES.items():
            assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, src.name
            (run / src.name).write_bytes(src.read_bytes())
        for name in ("test_loop_dim_extent_access_witness.py", "test_loop_dim_extent_round5.py"):
            (tests / name).write_bytes((ROOT / "tests" / name).read_bytes())
        (tests / "fixtures").mkdir(); os.symlink(ROOT / "tests" / "fixtures" / "loop_dim_extent", tests / "fixtures" / "loop_dim_extent")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "head"
        res = subprocess.run([sys.executable, "-m", "pytest", *[s.name for s in R4_FILES], "-q", "-p", "no:cacheprovider", "-rfs"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=1500)
        failed = {l.split()[1] for l in res.stdout.splitlines() if l.startswith("FAILED")}
        assert failed == set(BY_DESIGN), (sorted(failed ^ set(BY_DESIGN)), res.stdout[-3000:])
        # per case: the measured tuple (counted, refused, loop-complete, coverage) re-read from the record the reviewer's file saved for it
        for tid, (cond, counted, refused, complete, coverage) in BY_DESIGN.items():
            L = _review4_record_layer(run / "runs", tid)
            assert L is not None, ("record not located", tid)
            got = ("loop_iterations" in L, "loop_dim_extent_mismatch" in L, L.get("loop_iterations_complete"), L.get("witness_coverage"))
            assert got == (counted, refused, complete, coverage), (tid, got, (counted, refused, complete, coverage))
        summary = [l for l in res.stdout.splitlines() if " passed" in l and " in " in l][-1]
        assert f"{len(BY_DESIGN)} failed" in summary and "135 passed" in summary and "40 skipped" in summary, summary
        # B′ invariant over EVERY flagged record the twelve files produced (not only the by-design cases): the typed notes live in
        # witness_unmodelled only (nothing witness-related in loop_iterations_omitted), witness_coverage is partial, and a False
        # completeness flag is the loop counter's own named omission.  (A flagged record MAY still refuse: the reviewer's
        # narrow_body_with_else refuses on its inspected body access while its orelse access is an uninspected-access note.)
        flagged = 0
        for d in (run / "runs").iterdir():
            for f in d.rglob("*.json"):
                rec = json.loads(f.read_text()); L = rec.get("layer") if isinstance(rec, dict) else None
                if isinstance(L, dict) and L.get("witness_unmodelled"):
                    flagged += 1
                    omitted = [str(o) for o in (L.get("loop_iterations_omitted") or [])]
                    assert L.get("witness_coverage") == "partial" and not any(o.startswith("witness_unmodelled:") for o in omitted), (f, L)
                    assert L.get("loop_iterations_complete") is True or omitted, (f, L)     # incomplete only by the counter's own, named omission
        assert flagged >= len(BY_DESIGN) // 2, flagged


class TestWitnessModelV5:
    """WITNESS_MODEL.v5.md §5: eligible programs count complete with ``witness_coverage: full`` / genuine narrow accesses refuse (§1),
    every §2 reason is a visible B′ typed boundary (no refusal, counted, complete unless the loop counter itself named an omission,
    ``witness_coverage: partial`` + the note, nothing witness-related in ``loop_iterations_omitted``), condition (f) control transfer,
    v4 coverage (``full`` only when every access of the kernel function was inspected and eligible), the for-else sibling rule, the
    analyzer pass-through, placement and legacy records.  ``j`` runs over Dim("E") = 8, ``Y`` [T, K] = [4, 3]
    is returned, ``Q`` [T, E] is a scratch, ``X`` [T, E] and ``Z`` [T, K] are the inputs."""
    PRE = '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
    E = '    for t in Dim("T"):\n        for j in Dim("E"):\n'
    K = '    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n'

    def _src(self, lines, top="", second=True):
        return self.PRE + top + self.E + "".join("            " + l + "\n" for l in lines) + (self.K if second else "") + "    return Y\n"

    def _layer(self, tmp_path, src):
        return _isolated_layer(tmp_path, H + SIG + src, _probe_metadata(), "k#0")

    @pytest.mark.parametrize("lines, n", [
        (["Q[t,j]=X[t,j]"], 44), (["Q[t,j]=X[t,j+0]"], 44), (["Q[t,j]=max(X[t,j],0.0)"], 44), (["Q[t,j]+=X[t,j]"], 44),
        (["v:float=X[t,j]", "Q[t,j]=v"], 76), (["Q[t,j]=X[t,j]*X.shape[1]"], 44),
    ], ids=["plain-copy", "binop-index", "builtin-call", "augmented", "annotated", "attribute-shape"])
    def test_eligible_counts(self, tmp_path, lines, n):
        rec = observe(tmp_path, prog(self._src(lines)), inputs=INPUTS)
        L = counted(rec, n)
        assert "witness_unmodelled" not in L and L.get("witness_coverage") == "full", L

    def test_eligible_counts_sibling_loops(self, tmp_path):
        """Two sibling nests spell their inner variable ``e`` (Dim E over the scratch, Dim K over the returned Y): each access is governed
        by its own lexically innermost ``e`` loop -- counted 44, complete, no note."""
        rec = observe(tmp_path, prog(self.PRE + '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    for t in Dim("T"):\n        for e in Dim("K"):\n            Y[t,e]=Q[t,e]\n    return Y\n'), inputs=INPUTS)
        L = counted(rec, 44); assert "witness_unmodelled" not in L and L.get("witness_coverage") == "full", L

    @pytest.mark.parametrize("lines", [["Y[t,j]=X[t,j]"], ["Q[t,j]=Z[t,j]"], ["Y[t,j]+=X[t,j]"], ["Y[t,j]:float=X[t,j]"], ["Y[t,j]=max(X[t,j],0.0)"], ["Y[t+0,j]=X[t,j]"]],
                             ids=["returned-narrow", "parameter-narrow", "augmented-narrow", "annotated-narrow", "builtin-call-narrow", "mixed-components"])
    def test_eligible_refuses(self, tmp_path, lines):
        L, A = self._layer(tmp_path, self._src(lines, second=False))
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L
        assert L.get("loop_iterations_complete") is False and A["layers"]["k#0"]["status"] == "incomplete", L

    def test_eligible_refuses_sibling_narrow_site(self, tmp_path):
        L, A = self._layer(tmp_path, self.PRE + '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=Q[t,e]\n    return Y\n')
        assert "loop_dim_extent_mismatch" in L, L

    def _boundary(self, tmp_path, src, reason):
        """B′: no refusal; counted and COMPLETE; the typed note in ``witness_unmodelled`` with ``witness_coverage: partial``; nothing
        witness-related in ``loop_iterations_omitted``; the analysis layer carries both fields informationally and is never
        ``loop_metadata_incomplete`` for the witness (other analyzers -- the statement-unit classifier -- may still mark a probe program)."""
        L, A = self._layer(tmp_path, src)
        omitted = [str(o) for o in (L.get("loop_iterations_omitted") or [])]
        assert "loop_dim_extent_mismatch" not in L and "loop_iterations" in L and not any(o.startswith("witness_unmodelled:") for o in omitted), L
        # B′: the witness never clears completeness -- a False flag here is the loop counter's own, pre-existing omission (a comprehension,
        # with / try / class / def body or plain for the counter did not visit), named in loop_iterations_omitted without any witness entry
        assert L.get("loop_iterations_complete") is True or omitted, L
        notes = L.get("witness_unmodelled", [])
        assert notes and all(n.startswith("witness_unmodelled:") for n in notes) and any(n.startswith(f"witness_unmodelled:{reason}") for n in notes), (reason, L)
        assert L.get("witness_coverage") == "partial", L
        lay = A["layers"]["k#0"]                                        # the witness never CAUSES incompleteness: loop_metadata_incomplete only when the
        assert not any(str(o).startswith("witness_unmodelled:") for o in (lay.get("loop_iterations_omitted") or [])), lay   # counter itself named an omission
        assert lay.get("incomplete_reason") != "loop_metadata_incomplete" or omitted, lay
        assert lay.get("witness_coverage") == "partial" and lay.get("witness_unmodelled") == notes, lay
        return L

    @pytest.mark.parametrize("top, lines, reason", [
        ("", ["j=0", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"), ("", ["j-=j", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"),
        ("", ["j,v=0,1", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"), ("", ["del j", "j=0", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"),
        ("", ["from os import F_OK as j", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"), ("", ["Y[t,j]=X[t,(j:=0)]"], "spelling_bound_otherwise"),
        ("", ["v=[j for j in range(2)]", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"), ("", ["with open(__file__) as j:\n                pass", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"),
        ("", ["try:\n                pass\n            except Exception as j:\n                pass", "Y[t,j]=X[t,j]"], "spelling_bound_otherwise"),
        ("    global j\n", ["Y[t,j]=X[t,j]"], "spelling_bound_otherwise"), ("    for j in range(3):\n        pass\n", ["Y[t,j]=X[t,j]"], "spelling_bound_otherwise"),
        ("", ["Y[t,j]=X[t,j] if 1<0 else 0.0"], "statement_ifexp"), ("", ["v=(1<0) and Y[t,j]", "Y[t,0]=X[t,0]"], "statement_boolop"),
        ("", ["v=1<0<Y[t,j]", "Y[t,0]=X[t,0]"], "statement_compare_chain"), ("", ["v=(Y[t,j],lambda: 0.0)", "Y[t,0]=X[t,0]"], "statement_lambda"),
        ("", ["v=[Y[t,j] for q in range(1)]", "Y[t,0]=X[t,0]"], "statement_listcomp"), ("", ["v=(Y[t,j] for q in range(1))", "Y[t,0]=X[t,0]"], "statement_generatorexp"),
        ("", ["v={0:Y[t,j]}", "Y[t,0]=X[t,0]"], "statement_dict"), ("", ["Y[t,j]=(yield)"], "statement_yield"),
        ("    class C:\n        pass\n", ["Y[t,j]=X[t,j]"], "class_statement"), ("    def f():\n        return j\n", ["Y[t,j]=X[t,j]"], "nested_scope_mentions_spelling"),
        ("", ["v=locals()", "Y[t,j]=X[t,j]"], "dynamic_namespace_spelling"), ("    g=max\n", ["Y[t,j]=g(X[t,j],0.0)"], "opaque_call"), ("", ["Y[t,j]=(lambda: 0.0)()"], "opaque_call"),   # a non-Name callee is opaque: the kernel-wide (e) scan precedes (c), so statement_call is unreachable
    ], ids=["assigned-index", "augmented-index", "tuple-index", "del-index", "import-index", "walrus-index", "comprehension-target", "with-as", "except-as", "global-decl",
            "plain-for-target", "ifexp", "boolop", "chained-compare", "lambda", "comprehension", "generator", "dict", "yield", "class-statement", "nested-def-mentions",
            "dynamic-spelling", "opaque-alias-call", "call-non-name"])
    def test_boundary(self, tmp_path, top, lines, reason):
        """§2 (B′): a genuinely narrow ``Y[t, j]`` in a program outside the predicate is NOT refused; the layer is counted, complete unless
        the loop counter's own omission says otherwise, and carries the typed note with the first failing condition in
        ``witness_unmodelled``.  Parsed only."""
        self._boundary(tmp_path, self._src(lines, top=top, second=False), reason)

    def test_boundary_nested_shadow(self, tmp_path):
        """(b') ``for j in Dim(E): for j in Dim(K): ...`` then a sibling nest reading ``j`` inside the outer loop: the lexical governor (E)
        would be wrong (j is the inner survivor) -- typed boundary."""
        self._boundary(tmp_path, self.PRE + '    for j in Dim("E"):\n        for j in Dim("K"):\n            for t in Dim("T"):\n                Q[t,j]=X[t,j]\n        for t in Dim("T"):\n            Y[t,j]=X[t,j]\n    return Y\n',
                       "spelling_shadowed_by_nested_dim_loop")

    def test_boundary_plain_for_body(self, tmp_path):
        """(d) the access sits in a plain ``for`` body under the Dim loops: typed boundary."""
        self._boundary(tmp_path, self.PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            for q in range(1):\n                Y[t,j]=X[t,j]\n    return Y\n', "statement_under_plain_loop")

    def test_boundary_survivor_after_sibling(self, tmp_path):
        """(f) no Dim loop over ``j`` encloses the statement (the survivor of a finished sibling loop): typed boundary."""
        self._boundary(tmp_path, self.PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n    for t in Dim("T"):\n        Y[t,j]=X[t,j]\n    return Y\n', "spelling_not_enclosed")

    def test_boundary_valid_program_is_counted_and_flagged(self, tmp_path):
        """B′ on a VALID program: ``j = 0`` then ``Y[t, j] = X[t, j]`` executes 64 assignments; v3 counts 64, complete, refuses nothing
        and flags the declined witness (``witness_coverage: partial``) -- round 8 had made this layer incomplete."""
        rec = observe(tmp_path, prog(self._src(["j=0", "Y[t,j]=X[t,j]"], second=False)), inputs=INPUTS)
        L = rec["layer"]
        assert "error" not in rec and "loop_dim_extent_mismatch" not in L and L["loop_iterations"] == 2 * T * E and L["loop_iterations_complete"] is True, L
        assert any(n.startswith("witness_unmodelled:spelling_bound_otherwise") for n in L["witness_unmodelled"]) and L["witness_coverage"] == "partial" and "loop_iterations_omitted" not in L, L

    FOR_ELSE = ('    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                '    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n        else:\n            for j in Dim("E"):\n                Q[t,j]=X[t,j]*0.5\n'
                '    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n    return Y\n')

    def test_for_else_valid_body_witnessed(self, tmp_path):
        """(b') round 9/10: the orelse loop over ``j`` is a SIBLING segment, not a nested iteration -- the body statement keeps its witness
        (no ``spelling_shadowed_by_nested_dim_loop``); v4 coverage is ``partial`` because the orelse ACCESS was not inspected
        (``access_outside_loop_expr``); the counter's own for-else omission is unchanged."""
        L, A = self._layer(tmp_path, self.FOR_ELSE)
        notes = L.get("witness_unmodelled") or []
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial", L
        assert notes and all(n.startswith("witness_unmodelled:access_outside_loop_expr") for n in notes), notes
        assert L["loop_iterations"] == T * E + T * K and not any("witness" in str(o) for o in (L.get("loop_iterations_omitted") or [])), L

    def test_for_else_narrow_body_refuses(self, tmp_path):
        """(b'): a genuinely narrow ``Y[t, j]`` in the BODY of the for-else loop still refuses (the orelse loop does not shadow it)."""
        L, A = self._layer(tmp_path, self.FOR_ELSE.replace('            Q[t,j]=X[t,j]\n        else:', '            Y[t,j]=X[t,j]\n        else:'))
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L

    def test_for_else_orelse_only_narrow_is_boundary(self, tmp_path):
        """SCOPE-CONTRACT-25: a genuinely narrow access ONLY in the orelse is a BOUNDARY -- not visited by the loop counter, outside
        ``loop_expr``, not inspected: no refusal, ``witness_coverage: partial`` with ``access_outside_loop_expr`` (never ``full``)."""
        L, A = self._layer(tmp_path, self.FOR_ELSE.replace('                Q[t,j]=X[t,j]*0.5', '                Y[t,j]=X[t,j]*0.5'))
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial", L
        assert any(n.startswith("witness_unmodelled:access_outside_loop_expr") for n in L.get("witness_unmodelled") or []), L

    CT = {"unconditional_break": (["Q[t,j]=X[t,j]", "Y[t,j]=X[t,j]", "break"], True), "conditional_break": (["Q[t,j]=X[t,j]", "if j>=3:\n                break", "Y[t,j]=X[t,j]"], True),
          "continue_filter": (["Q[t,j]=X[t,j]", "if j>=3:\n                continue", "Y[t,j]=X[t,j]"], True), "return_in_if": (["Q[t,j]=X[t,j]", "Y[t,j]=X[t,j]", "if j==2:\n                return Y"], True),
          "raise_in_try": (["try:\n                raise ValueError()\n            except ValueError:\n                pass", "Y[t,j]=X[t,j]"], False),
          "break_in_nested_if": (["Q[t,j]=X[t,j]", "if j>1:\n                if t>=0:\n                    break", "Y[t,j]=X[t,j]"], True)}

    @pytest.mark.parametrize("name", sorted(CT))
    def test_control_transfer_valid_counts(self, tmp_path, name):
        """(f) control-transfer-22: a break / continue / return / raise anywhere in the governing loop's body means the narrow-looking
        ``Y[t, j]`` may execute only at small ``j`` or never -- these VALID programs are not refused; the governed components are
        unmodelled (``control_transfer_in_enclosing_loop``, ``partial``); the counter's own omission makes the count incomplete (§6)."""
        lines, run = self.CT[name]
        src = self._src(lines, second=False)
        if run:
            rec = observe(tmp_path, prog(src), inputs=INPUTS); assert "error" not in rec, rec; L = rec["layer"]
        else:
            L, A = self._layer(tmp_path, src)
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial", L
        assert any(n.startswith("witness_unmodelled:control_transfer_in_enclosing_loop") for n in L.get("witness_unmodelled") or []), L
        assert L.get("loop_iterations_complete") is False, L
        if name == "return_in_if":   # a return inside the loop is the round-4 rule's refusal (loop_entry_unresolved), not the counter omission
            assert "loop_entry_unresolved" in L and not any(str(o).startswith("control_transfer_in_dim_loop") for o in L.get("loop_iterations_omitted") or []), L
        else:
            assert any(str(o).startswith("control_transfer_in_dim_loop") for o in L.get("loop_iterations_omitted") or []), L

    ANC = {"outer_break": "break", "outer_continue": "continue", "outer_if_break": "if t>=0:\n    break",
           "outer_except_continue": "try:\n    raise ValueError()\nexcept ValueError:\n    continue"}

    def _anc_src(self, text):
        return self.PRE + '    for t in Dim("T"):\n' + "".join("        " + l + "\n" for l in text.splitlines()) + '        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n            Y[t,j]=X[t,j]\n    return Y\n'

    def test_ancestor_while_break_is_conservative(self, tmp_path):
        """A ``break`` inside a nested ``while`` in the outer body leaves the while, not the Dim loop -- the inner nest runs fully and this
        program is INVALID at runtime (parsed only here).  The rule is conservative: the transfer is attributed to the enclosing Dim loop,
        so the witness declines (no refusal, note, ``partial``) -- an accepted over-approximation, not a detection."""
        L, A = self._layer(tmp_path, self._anc_src("while True:\n    break"))
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial", L
        assert any(n.startswith("witness_unmodelled:control_transfer_in_enclosing_loop") for n in L.get("witness_unmodelled") or []), L

    @pytest.mark.parametrize("name", sorted(ANC))
    def test_ancestor_transfer_valid(self, tmp_path, name):
        """(f) v5, ancestor-transfer-26: a transfer in the OUTER ``t`` loop (directly, under an if, in an except handler, inside a
        nested while) can keep the inner ``j`` loop from ever running -- the narrow-looking ``Y[t, j]`` executes 0 (or few) times.  Not
        refused; every candidate of the statement is unmodelled (``control_transfer_in_enclosing_loop``, ``partial``); the outer loop
        carries the counter's own ``control_transfer_in_dim_loop`` omission."""
        rec = observe(tmp_path, prog(self._anc_src(self.ANC[name])), inputs=INPUTS); assert "error" not in rec, rec; L = rec["layer"]
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial", L
        assert any(n.startswith("witness_unmodelled:control_transfer_in_enclosing_loop") for n in L.get("witness_unmodelled") or []), L
        assert L.get("loop_iterations_complete") is False and any(str(o).startswith("control_transfer_in_dim_loop") and "inside for t in Dim('T')" in str(o) for o in L.get("loop_iterations_omitted") or []), L

    def test_transfer_in_sibling_nest_still_refuses(self, tmp_path):
        """(f) v5: a transfer in an EARLIER, non-enclosing nest (break in the K loop of the first nest) does not erase the genuine narrow
        witness of a later independent nest -- the reviewer's separate-loop control."""
        L, A = self._layer(tmp_path, self.PRE + '    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Z[t,r]\n            break\n    for t in Dim("T"):\n        for j in Dim("E"):\n            Y[t,j]=X[t,j]\n    return Y\n')
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L

    def test_control_transfer_outside_governor_still_refuses(self, tmp_path):
        """A transfer in a SIBLING nest does not reach this governor: the genuinely narrow body access still refuses."""
        L, A = self._layer(tmp_path, self._src(["Y[t,j]=X[t,j]"], second=False).replace("    return Y\n", '    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n            break\n    return Y\n'))
        assert re.search(r"Dim\('E'\) of kernel 'k#0' is bound to extent 8 but its loop variable indexes an axis of only 3", L["loop_dim_extent_mismatch"]), L

    @pytest.mark.parametrize("src", [
        PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n        Y[t,0]=Q[t,0]\n    return Y\n',
        PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n        else:\n            Y[t,0]=Q[t,0]\n    return Y\n',
        PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            if j>0:\n                Q[t,j]=X[t,j]\n            Y[t,0]=X[t,0]\n    return Y\n',
        PRE + '    for t in Dim("T"):\n        for j in Dim("E"):\n            def f():\n                return Y[t,0]\n            Q[t,j]=X[t,j]\n    return Y\n',
    ], ids=["outer_level_write", "orelse_access", "if_body_access", "nested_def_access"])
    def test_coverage_partial_for_uninspected_access(self, tmp_path, src):
        """v4 coverage: an access the witness did not inspect (outer-level in-loop statement, orelse, if body, nested def) makes coverage
        ``partial`` with a typed note; nothing is refused and no count changes."""
        L, A = self._layer(tmp_path, src)
        assert "loop_dim_extent_mismatch" not in L and L.get("witness_coverage") == "partial" and L.get("witness_unmodelled"), L
        assert not any(str(o).startswith("witness_unmodelled:") for o in L.get("loop_iterations_omitted") or []), L

    def test_analyzer_surfaces_witness_coverage_informationally(self, tmp_path):
        """B′ pass-through: the flagged valid program's analysis layer carries ``witness_coverage`` / ``witness_unmodelled`` and is
        otherwise identical in kind to the eligible program's layer (no ``status``, no ``loop_iterations_omitted``, same ``cost_basis``)."""
        Lf, Af = self._layer(tmp_path / "flagged", self._src(["j=0", "Y[t,j]=X[t,j]"], second=False))
        Le, Ae = self._layer(tmp_path / "eligible", self._src(["Y[t,j]=X[t,j]"], second=False).replace("Y[t,j]=X[t,j]", "Q[t,j]=X[t,j]"))
        lf, le = Af["layers"]["k#0"], Ae["layers"]["k#0"]
        assert lf.get("witness_coverage") == "partial" and lf.get("witness_unmodelled") == Lf["witness_unmodelled"], lf
        assert le.get("witness_coverage") == "full" and "witness_unmodelled" not in le, le
        for lay in (lf, le):
            assert "status" not in lay and "loop_iterations_omitted" not in lay and lay.get("cost_basis") == le.get("cost_basis"), lay

    def test_boundary_placement(self, tmp_path):
        """Placement: a segment that cannot be placed / a statement the segment does not reproduce -> no witness, typed note."""
        import types
        from solar.einsum_dsl.einsum_converter import _kernel_loop_iterations, _witness_model_v2
        kernel = TestIndexBinding._kernel(tmp_path, self._src(["Y[t,j]=X[t,j]"], second=False))
        kernel.source_text = "def k(X, Z):\n    return Y\n"
        assert _kernel_loop_iterations(kernel)[0] == T * E and kernel.witness_notes == ["witness_unmodelled:segment_not_placed"]
        p = tmp_path / "probe_einsum_lib.py"; p.write_text(H + SIG + self._src(["Q[t,j]=X[t,j]"]))
        info = extract_kernel_info(p)["k"]; stmts = [s.strip() for s in info["loop_expr"].split(";") if s.strip()]
        model = _witness_model_v2(types.SimpleNamespace(**info), [stmts[0].replace("Q[t, j]", "Q[t, j + 0]"), stmts[1]])
        accesses, notes = model[1]
        assert model[0] is None and not notes and accesses[0][0] == "Y" and accesses[0][1][1][1][1] == "K"   # the second statement keeps its (line, K) governor

    def test_legacy_records(self, tmp_path):
        """§3: no return information -> no governor.  Valid scratch-first 44, valid returned-first 12, the genuinely narrow returned-first
        record 32 WITHOUT a witness (the documented boundary; no note either: there is no governor to decline; ``witness_pass`` False ->
        the exporter writes ``witness_coverage: none``)."""
        from solar.einsum_dsl.einsum_converter import _kernel_loop_iterations
        cases = [(self.PRE + '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    for t in Dim("T"):\n        for e in Dim("K"):\n            Y[t,e]=Z[t,e]\n    return Y\n', 44),
                 (self.PRE + '    for t in Dim("T"):\n        for k in Dim("K"):\n            Y[t,k]=Z[t,k]\n    return Y\n', 12),
                 (self.PRE + '    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=X[t,e]\n    return Y\n', 32)]
        for body, n in cases:
            kernel = TestIndexBinding._kernel(tmp_path, body); kernel.returns = None; kernel.returned_names = None
            assert _kernel_loop_iterations(kernel)[0] == n and not getattr(kernel, "witness_notes", None) and kernel.witness_pass is False, (n, kernel.witness_notes)


class TestIndexBinding:
    @staticmethod
    def _kernel(tmp_path, body):
        from solar.einsum_dsl.loader import load_dsl_yaml
        from solar.einsum_dsl.assembler import build_dsl_graph
        dsl = tmp_path / "dsl"; dsl.mkdir(exist_ok=True)
        (dsl / "probe_einsum_lib.py").write_text(H + SIG + body)
        (dsl / "probe_einsum_lib.yaml").write_text(yaml.safe_dump(_probe_metadata(), sort_keys=False))
        graph = build_dsl_graph(load_dsl_yaml(dsl / "probe_einsum_lib.yaml"), None, "probe", kernel_info=extract_kernel_info(dsl / "probe_einsum_lib.py"))
        return next(k for k in graph.kernels if k.id == "k#0")

    def test_unplaceable_record_witnesses_nothing(self, tmp_path):
        """POOL-8 (round 5): the valid sibling program counts 44 placed; the same record with its ``source_text`` removed counts 44 from the
        bindings alone (no pooled-by-spelling mismatch); the placed genuine-narrow sibling control still refuses."""
        from solar.einsum_dsl.einsum_converter import _kernel_loop_iterations, LoopDimExtentMismatch
        valid = ('    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                 '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
                 '    for t in Dim("T"):\n        for e in Dim("K"):\n            Y[t,e]=Z[t,e]\n    return Y\n')
        kernel = self._kernel(tmp_path, valid)
        assert _kernel_loop_iterations(kernel)[0] == T * E + T * K
        kernel.source_text = None
        total, trips, source, outer = _kernel_loop_iterations(kernel)
        assert source == "paths" and total == T * E + T * K and outer == 0 and kernel.witness_notes == ["witness_unmodelled:segment_not_placed"], (total, kernel.witness_notes)
        narrow = self._kernel(tmp_path, '    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
                                        '    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n'
                                        '    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=Q[t,e]\n    return Y\n')
        with pytest.raises(LoopDimExtentMismatch, match=r"bound to extent 8 but its loop variable indexes an axis of only 3"):
            _kernel_loop_iterations(narrow)


class TestCounterControlTransfer:
    """TARGET-BREAK-COUNT-23 (round 10, separate counter commit; WITNESS_MODEL.v5 §6): a Dim loop whose body contains a
    break / continue / raise is no longer a WRONG complete count (a return inside the loop is the round-4 ``loop_entry_unresolved`` rule) -- the figure is unchanged, ``loop_iterations_complete`` is
    False with the typed omission ``control_transfer_in_dim_loop``, and the analyzer bills ``loop_metadata_incomplete``.  B′ untouched:
    this is the COUNTER's own omission, not a witness note."""
    PRE_ = TestWitnessModelV5.PRE; LOOP_ = TestWitnessModelV5.E; COPY_ = TestWitnessModelV5.K

    def _src(self, lines, copy=True):
        return self.PRE_ + self.LOOP_ + "".join("            " + l + "\n" for l in lines) + (self.COPY_ if copy else "") + "    return Y\n"

    def test_plain_control_complete(self, tmp_path):
        """the reviewer's ``plain_control``: 44 complete, no omission -- unchanged by the fix."""
        rec = observe(tmp_path, prog(self._src(["Q[t,j]=X[t,j]"])), inputs=INPUTS)
        L = counted(rec, T * E + T * K); assert "loop_iterations_omitted" not in L, L

    @pytest.mark.parametrize("lines, hits", [
        (["Q[t,j]=X[t,j]", "break"], T + T * K), (["Q[t,j]=X[t,j]", "if j>=3:\n                continue", "Q[t,j]=X[t,j]*0.5"], T * E + 3 * T + T * K),
    ], ids=["break_wide_counter", "continue_filter"])
    def test_control_transfer_incomplete(self, tmp_path, lines, hits):
        """the reviewer's ``break_wide_counter`` (runtime 4 + 12 = 16 assignments; the counter said 44 complete): the figure stays, the
        structure is incomplete with the typed omission; the analyzer bills loop_metadata_incomplete."""
        rec = observe(tmp_path, prog(self._src(lines)), inputs=INPUTS)
        assert "error" not in rec, rec
        L = rec["layer"]
        assert L.get("loop_iterations_complete") is False, L                       # a kernel that returns inside the loop carries no count at all (existing parser behaviour)
        om = [str(o) for o in L.get("loop_iterations_omitted") or []]
        assert any(o.startswith("control_transfer_in_dim_loop (line ") and " inside for j in Dim('E')" in o for o in om), om
        assert not any(o.startswith("witness_unmodelled:") for o in om), om
        lay = rec["analysis"]["layers"]["k#0"] if "analysis" in rec else None
        if lay is not None:
            assert lay.get("incomplete_reason") == "loop_metadata_incomplete" and lay.get("cost_basis") != "loop_iterations", lay

    def test_return_inside_the_loop_is_the_round4_rule(self, tmp_path):
        """A ``return`` inside the loop is already the round-4 refusal ``loop_entry_unresolved`` ("encloses a conditional return"): the
        counter records no control_transfer omission for it (not twice; the pre-existing round-4 contract is unchanged)."""
        rec = observe(tmp_path, prog(self._src(["Q[t,j]=X[t,j]", "if j==2:\n                return Y"])), inputs=INPUTS)
        L = rec["layer"]
        assert "loop_entry_unresolved" in L and not any(str(o).startswith("control_transfer_in_dim_loop") for o in L.get("loop_iterations_omitted") or []), L

    def test_raise_in_try_is_conservative(self, tmp_path):
        """A ``raise`` caught inside the loop body does not end the loop at runtime, but the rule is conservative: the omission is
        recorded (parsed only)."""
        L, A = _isolated_layer(tmp_path, H + SIG + self._src(["try:\n                raise ValueError()\n            except ValueError:\n                pass", "Q[t,j]=X[t,j]"]), _probe_metadata(), "k#0")
        om = [str(o) for o in L.get("loop_iterations_omitted") or []]
        assert L.get("loop_iterations_complete") is False and any(o.startswith("control_transfer_in_dim_loop") and "raise at line" in o for o in om), om

    def test_transfer_inside_nested_def_is_not_the_loop_s(self, tmp_path):
        """A ``return`` inside a nested def in the loop body belongs to that def: no control-transfer omission (the def statement is
        the counter's existing 'not visited' omission)."""
        L, A = _isolated_layer(tmp_path, H + SIG + self._src(["def f():\n                return 0", "Q[t,j]=X[t,j]"]), _probe_metadata(), "k#0")
        om = [str(o) for o in L.get("loop_iterations_omitted") or []]
        assert not any(o.startswith("control_transfer_in_dim_loop") for o in om), om
