# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round 10 of the fallback dispatcher (the review an earlier review, manager hardening rule).

result-boundary-29: every path a candidate/adapter process may have written (report.json,
result archives, the staged receipt) is ingested through ``_ingest_result_file`` -- lstat, open
O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC, fstat (regular file, one link, our uid, size cap,
directory-entry identity), bounded read under the remaining dispatch budget -- BEFORE the private
evidence root is unlocked.  A fifo/symlink/directory/oversize file is ``candidate_result_missing``
with the type named.
reference-27: the report is parsed against an allowlist (unknown keys dropped and counted,
never merged raw); no public record names the private root's path; after a two-attempt schedule no
public record carries a private digest or path.
numeric-feedback-30: attempt-visible state carries pass/fail per output and the tolerance
policy only; the error statistics surface in the final receipt / result.

Round 11 (the review an earlier review): (A) a RESUMABLE outcome (awaiting_agent, pending
re-validation) publishes the attempt-visible view in its receipt and result too; (B) the public
verdict digest is computed over the attempt-visible verdict, the full-verdict digest is private;
(P2) the post-read fstat re-checks size/mode/uid and an exhausted budget refuses before any read;
(CI-FAIL-27) torch._C submodules carrying the PARENT's __file__ (pybind11 >= 3.0 def_submodule,
torch >= 2.9 on the gate runner) are attributed; a submodule naming another file is refused and
the private census diagnostic says which condition failed.

Round 12 (the review an earlier review, NUMERIC-FEEDBACK-30/C): the public compare record copied
every ingest field but the bytes, so ``result_ingest.report.size`` published the serialized length
of the UNREDACTED comparison report and a four-value finite domain identified the hidden error.
Attempt-visible records now carry a FIXED-SHAPE ingest view (ok, kind, cap, timed_out, redacted
problem) and no duration; size/links/time bound stay in the private root and return in the
terminal receipt.  A property test over the WHOLE attempt-visible record checks that dispatches
differing only in the hidden comparison outcome publish identical state, receipt and result.

Round 13 (the review an earlier review): (P1 NUMERIC-FEEDBACK-30/D) a NON-FINITE error statistic
(finite operands: 1e308 - (-1e308)) was dropped by the allowlist parser -- the public
``report_schema.unknown_or_mistyped_dropped`` counter, the absent key and the digest disagreement
(``trusted_verdict_unverifiable`` instead of ``trusted_reference_mismatch``) identified the hidden
value.  The compare child now writes non-finite statistics as the canonical tokens "inf"/"-inf"/"nan"
that the parser accepts (digests agree, nothing waived); the attempt-visible record is a
CONSTANT-SHAPE document (fixed verdict keys, nullable report_schema, ``vacuous_outputs``, kernel ids
and inode facts withheld); every check on the compare child's verdict is ONE public class with ONE
constant message; a failing candidate against a vacuous reference is the mismatch class.
(P2 PUBLIC-PROPERTY-SCOPE-31) ``wall_seconds_elapsed`` and the request's ``wall_seconds_remaining``
are withheld; the whole-surface property below compares raw file bytes (canonical serialisation,
normalised byte identity including key order, size accounting), the public tree listing and the
captured dispatcher log over every statistic class, with the reviewer's two fixtures verbatim.

Helpers are imported from ``test_fallback_dispatcher`` (no test function is imported).
"""
import hashlib
import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from solar.frontend import dispatcher as D
from solar.frontend.dispatcher import Budget, DispatchPolicy, FailureKind, PureSkillProvider, dispatch, resume
from test_fallback_dispatcher import (_agentic_dir_payload, _candidate_body, _cuda_case, _diverge,  # noqa: F401
                                      _phase_dir, _phase_payload, _private_root_of, _private_verdict, _sha,
                                      _wrong_dsl)


def _byte_distinct(consume):
    """A two-attempt fake agent: every attempt writes byte-distinct candidate source
    (a retry with identical bytes is refused by design)."""
    calls = []

    def wrapped(path):
        consume(path)
        rq = json.loads(Path(path).read_text())
        ds = Path(rq["candidate_path"])
        calls.append(str(ds))
        ds.write_text(ds.read_text() + f"\n# attempt {len(calls)}\n")
        rp = Path(path).parent / "TRANSLATION_RESPONSE.json"
        resp = json.loads(rp.read_text())
        resp["candidate_sha256"] = _sha(ds)
        rp.write_text(json.dumps(resp))
    wrapped.calls = calls
    return wrapped


def _private_roots(out_dir):
    return sorted((Path(out_dir) / "agentic").glob(".private-evidence-*"))


def _public_text(out_dir, roots, res=None):
    """Every public JSON record of a dispatch: files outside the private roots (no symlink
    followed) plus the returned result."""
    parts = [] if res is None else [json.dumps(res.to_dict(), default=str)]
    for q in sorted(Path(out_dir).rglob("*.json")):
        if q.is_symlink() or any(r in q.parents for r in roots):
            continue
        parts.append(q.read_text())
    return "\n".join(parts)


def _validate_receipts(res):
    return [x for x in res.failure_receipts if x["stage"] == "agentic_validate"]


# ---------------------------------------------------------------------------
# the ingest helper itself
# ---------------------------------------------------------------------------

def test_ingest_result_file_rules(tmp_path):
    good = tmp_path / "good.bin"
    good.write_bytes(b"abc")
    ok = D._ingest_result_file(good, 10, label="good", seconds=1.0)
    assert ok["ok"] and ok["bytes"] == b"abc" and ok["kind"] == "regular file" and ok["nlink"] == 1
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 11)
    r = D._ingest_result_file(big, 10, label="big", seconds=1.0)
    assert not r["ok"] and r["bytes"] is None and "exceed" in r["problem"] and r["kind"] == "regular file"
    sym = tmp_path / "sym.bin"
    sym.symlink_to(good)
    r = D._ingest_result_file(sym, 10, label="sym", seconds=1.0)
    assert not r["ok"] and r["kind"] == "symlink" and "symlink" in r["problem"]
    fifo = tmp_path / "fifo.bin"
    os.mkfifo(fifo, 0o600)
    t0 = time.monotonic()
    r = D._ingest_result_file(fifo, 10, label="fifo", seconds=5.0)
    assert not r["ok"] and r["kind"] == "fifo" and "fifo" in r["problem"] and time.monotonic() - t0 < 2.0
    r = D._ingest_result_file(tmp_path, 10, label="dir", seconds=1.0)
    assert not r["ok"] and r["kind"] == "directory"
    r = D._ingest_result_file(tmp_path / "absent.bin", 10, label="absent", seconds=1.0)
    assert not r["ok"] and r["kind"] == "absent"
    alias = tmp_path / "alias.bin"
    os.link(good, alias)
    r = D._ingest_result_file(good, 10, label="linked", seconds=1.0)
    assert not r["ok"] and "2 directory entries" in r["problem"] and r["nlink"] == 2
    os.unlink(alias)
    assert D._ingest_result_file(good, 10, label="good", seconds=1.0)["ok"]
    with pytest.raises(ValueError):
        DispatchPolicy(max_result_bytes=0)


def test_ingest_is_bounded_by_the_remaining_budget(tmp_path, monkeypatch):
    """A read that does not return (simulated) cannot hold the parent past the bound."""
    path = tmp_path / "slow.bin"
    path.write_bytes(b"data")
    gate = threading.Event()

    def stuck_read(fd, n):
        gate.wait(10.0)
        return b""
    monkeypatch.setattr(D, "_TRUSTED_READ", stuck_read)
    t0 = time.monotonic()
    r = D._ingest_result_file(path, 10, label="slow", seconds=0.6)
    elapsed = time.monotonic() - t0
    gate.set()
    assert not r["ok"] and r["timed_out"] and "remaining dispatch budget" in r["problem"] and r["bytes"] is None
    assert elapsed < 5.0
    # the bound follows the dispatch deadline when one is set
    token = D._DISPATCH_DEADLINE.set(time.monotonic() + 0.2)
    try:
        assert D._ingest_seconds(600.0) == pytest.approx(D._INGEST_MIN_SECONDS, abs=0.05)
    finally:
        D._DISPATCH_DEADLINE.reset(token)
    assert D._ingest_seconds(7.0) == 7.0


def test_parse_report_allowlist_drops_and_counts():
    raw = {"ok": True, "phase": "candidate", "outputs_sha256": "a" * 64, "output_shapes": {"output": [2, 3]},
           "identity_audit": [], "unknown_key": "x", "argv": ["a", {"nested": 1}], "program_sha256_at_end": "nothex",
           "flags": {"isolated": True, "extra": 1}, "verdict": {"success": True}, "pid": "12"}
    rep, stats = D._parse_report(json.dumps(raw).encode(), "candidate")
    assert rep["ok"] is True and rep["outputs_sha256"] == "a" * 64 and rep["output_shapes"] == {"output": [2, 3]}
    assert "unknown_key" not in rep and "verdict" not in rep and "program_sha256_at_end" not in rep and "pid" not in rep
    assert rep["argv"] == ["a"] and rep["flags"] == {"isolated": True}
    assert stats["parsed"] and stats["dropped"] >= 5
    assert D._parse_report(b"[1, 2]", "candidate") == ({}, {"dropped": 0, "truncated": 0, "parsed": False})
    assert D._parse_report(b"\xff", "candidate")[0] == {}
    assert D._parse_report(None, "compare")[0] == {}


def test_private_root_is_found_by_identity_not_by_name(tmp_path):
    agentic = tmp_path / "agentic"
    root = D._private_root_create(agentic)
    ident = {"id": D._private_root_id(root), **{k: getattr(os.stat(root), "st_" + k) for k in ("dev", "ino")}}
    decoy = agentic / ".private-evidence-decoy"
    decoy.mkdir(0o700)
    (decoy / "PRIVATE_STATE.json").write_text((root / "PRIVATE_STATE.json").read_text())
    assert D._private_root_find(agentic, ident) == root
    assert D._private_root_find(agentic, {**ident, "ino": ident["ino"] + 1}) is None
    assert D._private_root_find(agentic, None) is None
    redacted = D._redact_private_paths({"a": str(root / "validation-1" / "x"), "b": [str(root)]}, root, ident["id"])
    assert redacted == {"a": f"<private-evidence:{ident['id']}>/validation-1/x", "b": [f"<private-evidence:{ident['id']}>"]}


# ---------------------------------------------------------------------------
# dispatches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["report", "outputs"])
def test_fifo_at_a_result_path_is_refused_without_blocking(tmp_path, where):
    """the review RESULT-BOUNDARY-29: a FIFO at the candidate's report path (or outputs path) and a
    zero exit.  The parent never opens it blocking: refused, type named, reference never computed.

    Budget (target pipeline 77147099): the facade's isolated verification starts fresh interpreters
    inside the trusted verify child (two per verification when numba declines the payload-laden
    body), so the phase needs tens of seconds on the runner; 10 s killed it.  The non-blocking proof
    does not rest on the budget: a blocked read shows as ``timed_out`` / a phase at its bound."""
    key = "report_path" if where == "report" else "outputs_path"
    payload = _phase_payload(
        "import os as _os\nfrom pathlib import Path as _P\n"
        f"_p = _P(_f.f_globals['META']['{key}'])\n"
        "if _p.exists(): _p.unlink()\n"
        "_os.mkfifo(_p, 0o600)\n"
        "(_p.parent / 'fifo-ready.txt').write_text('fifo replaces the result; the child exits zero')\n"
        "_os._exit(0)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    policy = DispatchPolicy(budget=Budget(max_attempts=1))
    policy.verify_timeout = 60
    t0 = time.monotonic()
    res = dispatch(r, policy, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "candidate_result_missing" and val["kind"] == FailureKind.UNSUPPORTED and val["scored"]
    assert "fifo" in val["reason"]
    tr = val["validation"]["trusted_reference"]
    label = "report" if where == "report" else "candidate_outputs"
    ing = tr["phases"]["candidate"]["result_ingest"][label]
    assert ing["kind"] == "fifo" and ing["ok"] is False and ing["timed_out"] is False
    assert tr["phases"]["candidate"]["returncode"] == 0 and "reference" not in tr["phases"]
    assert tr["phases"]["candidate"]["report_schema"]["ingest_before_unlock"] is True
    # the candidate phase itself (60 s timeout) plus the ingest never waited on the fifo
    assert tr["seconds"] < 60 and time.monotonic() - t0 < 120


def test_oversize_report_is_refused(tmp_path):
    payload = _phase_payload(
        "import os as _os\nfrom pathlib import Path as _P\n"
        "_p = _P(_f.f_globals['META']['report_path'])\n"
        "_p.write_bytes(b'{' + b' ' * (1 << 20) + b'}')\n"
        "_os._exit(0)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "candidate_result_missing" and "exceed" in val["reason"]
    ing = val["validation"]["trusted_reference"]["phases"]["candidate"]["result_ingest"]["report"]
    assert ing["kind"] == "regular file" and ing["size"] > D._REPORT_BYTES_CAP and not ing["ok"]


def test_candidate_written_report_is_allowlisted_and_counted(tmp_path):
    """A report the candidate writes itself (ok=true, foreign keys, mistyped values) admits
    nothing: only allowlisted typed keys survive, the rest is dropped and counted."""
    payload = _phase_payload(
        "import json as _json, os as _os\nfrom pathlib import Path as _P\n"
        "_rp = _P(_f.f_globals['META']['report_path'])\n"
        "_rp.write_text(_json.dumps({'ok': True, 'phase': 'candidate', 'outputs_sha256': 'nope', "
        "'injected_private_digest': 'f' * 64, 'program_sha256_at_end': {'nested': 1}, 'argv': 'notalist'}))\n"
        "_os._exit(0)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "candidate_result_missing" and val["scored"]
    rec = val["validation"]["trusted_reference"]["phases"]["candidate"]
    assert "injected_private_digest" not in json.dumps(rec) and rec.get("outputs_sha256") is None
    assert rec["report_schema"]["allowlisted"] and rec["report_schema"]["unknown_or_mistyped_dropped"] >= 4
    assert rec["result_ingest"]["candidate_outputs"]["kind"] == "absent"


def test_report_symlink_into_the_private_root_publishes_nothing(tmp_path):
    """the review REFERENCE-27 (round 9 probe, strengthened): attempt 1 mismatches and leaves a private
    verdict; attempt 2 makes its report.json a SYMLINK to the private root's own state file (the
    root found by directory NAME, since no public record carries its path) and exits zero.  The parent
    refuses the link without following it and before unlocking; no private path or digest reaches
    any public record."""
    payload = _agentic_dir_payload(
        "if _croot.parents[1].name != 'attempt_01':\n"
        "    _roots = sorted(_agentic.glob('.private-evidence-*'))\n"
        "    _rp = _P(_f.f_globals['META']['report_path'])\n"
        "    if _rp.exists(): _rp.unlink()\n"
        "    _rp.symlink_to(_roots[-1] / 'PRIVATE_STATE.json')\n"
        "    (_croot / 'report-link-marker.txt').write_text('report.json -> <root>/PRIVATE_STATE.json')\n"
        "    _os._exit(0)\n")
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge,
                               dsl_edit=lambda d: _candidate_body(_wrong_dsl(d), payload))
    res = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider(agent=_byte_distinct(consume)))
    assert not res.verified
    vals = _validate_receipts(res)
    assert [v["reason_code"] for v in vals] == ["trusted_reference_mismatch", "candidate_result_missing"]
    assert "symlink" in vals[1]["reason"]
    roots = _private_roots(tmp_path / "out")
    assert len(roots) == 1
    root = roots[0]
    privs = sorted(root.glob("validation-*/verdict_private.json"))
    assert len(privs) == 1
    secret = json.loads(privs[0].read_text())
    public = _public_text(tmp_path / "out", roots, res)
    assert str(root) not in public and root.name not in public
    assert secret["artifact_sha256"]["trusted_outputs"] not in public and secret["verdict_sha256"] not in public
    state = json.loads((tmp_path / "out" / "DISPATCH_STATE.json").read_text())
    assert state["private_evidence"]["id"] == D._private_root_id(root) and "private_evidence_root" not in state
    assert state["private_evidence"]["path_in_public_records"] is False
    rec = vals[1]["validation"]["trusted_reference"]["phases"]["candidate"]
    assert rec["result_ingest"]["report"]["kind"] == "symlink" and rec["report_schema"]["parsed"] is False
    assert "created_utc" not in json.dumps(rec)          # nothing of PRIVATE_STATE.json was read


def test_error_statistics_are_private_until_the_dispatch_ends(tmp_path):
    """the review NUMERIC-FEEDBACK-30 (manager ruling): a wrong candidate's retry reads the public state
    while it runs and finds pass/fail per output and the tolerance policy, no error statistic; the
    final receipt and the returned result carry the statistics."""
    payload = _agentic_dir_payload(
        "_state = _json.loads((_agentic.parent / 'DISPATCH_STATE.json').read_text())\n"
        "_nums = []\n_passes = []\n"
        "def _walk(_o):\n"
        "    if isinstance(_o, dict):\n"
        "        _v = _o.get('max_abs_error')\n"
        "        if isinstance(_v, (int, float)) and not isinstance(_v, bool): _nums.append(_v)\n"
        "        if isinstance(_o.get('outputs_pass'), dict): _passes.append(_o['outputs_pass'])\n"
        "        for _x in _o.values(): _walk(_x)\n"
        "    elif isinstance(_o, list):\n"
        "        for _x in _o: _walk(_x)\n"
        "_walk(_state)\n"
        "_txt = _json.dumps(_state)\n"
        "(_croot / 'state_probe.json').write_text(_json.dumps({'numbers': _nums, 'passes': _passes, "
        "'text_has_statistic': ('max_abs_error=' in _txt) or ('1.0' in _txt.replace('\"1.0e', ''))}))\n")
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge,
                               dsl_edit=lambda d: _candidate_body(_wrong_dsl(d), payload))
    res = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider(agent=_byte_distinct(consume)))
    assert not res.verified
    vals = _validate_receipts(res)
    assert [v["reason_code"] for v in vals] == ["trusted_reference_mismatch", "trusted_reference_mismatch"]
    probe = json.loads((Path(vals[1]["validation"]["trusted_reference"]["phases"]["candidate"]["phase_dir"])
                        / "state_probe.json").read_text())
    assert probe["numbers"] == [] and {"output": False} in probe["passes"]
    # the final result / receipt carry the statistics; the persisted state never does
    receipt = json.loads(Path(res.receipt_path).read_text())
    for src in (res.to_dict(), receipt):
        fails = [x for x in src["failure_receipts"] if x["stage"] == "agentic_validate"]
        assert all(x["validation"]["trusted_reference"]["max_abs_error"] == pytest.approx(1.0) for x in fails)
        assert all(x["validation"]["trusted_reference"]["per_output_max_abs_error"] == {"output": pytest.approx(1.0)} for x in fails)
        assert all(x["validation"]["trusted_reference"]["outputs_pass"] == {"output": False} for x in fails)
        assert all("_private_numeric_id" not in x["validation"] and "reason_private_id" not in x for x in fails)
    state = json.loads((tmp_path / "out" / "DISPATCH_STATE.json").read_text())
    fails = [x for x in state["failure_receipts"] if x["stage"] == "agentic_validate"]
    assert fails and all(x["validation"]["trusted_reference"]["max_abs_error"] is None for x in fails)
    assert all(x["validation"]["trusted_reference"]["outputs_pass"] == {"output": False} for x in fails)
    assert all(x["validation"]["trusted_reference"]["tolerance"]["rtol"] == 1e-4 for x in fails)
    assert all(x["validation"]["_private_numeric_id"] for x in fails)
    import re as _re
    assert not _re.search(r"max_abs_error=\s*[0-9]", json.dumps(state))      # the facade's text is redacted too
    assert "max_abs_error=<n>" in json.dumps(state)
    # the private store holds them
    root = _private_roots(tmp_path / "out")[0]
    store = json.loads((root / D._NUMERIC_STORE).read_text())
    assert len(store) == 2 and all(e["validation"]["trusted_reference"]["max_abs_error"] == 1.0 for e in store.values())


def test_no_public_record_names_the_private_root_after_two_attempts(tmp_path):
    """fix (d): after a full two-attempt schedule (two byte-distinct mismatching candidates, two
    private verdicts) no public record contains the private root's path, any private digest, or a
    private report path -- only the opaque id."""
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    res = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider(agent=_byte_distinct(consume)))
    assert not res.verified
    roots = _private_roots(tmp_path / "out")
    assert len(roots) == 1
    root = roots[0]
    secrets_ = []
    for p in root.rglob("verdict_private.json"):
        d = json.loads(p.read_text())
        secrets_ += [d["verdict_sha256"], d["artifact_sha256"]["trusted_outputs"], str(p), str(p.parent)]
    assert len(secrets_) == 8      # two validations
    public = _public_text(tmp_path / "out", roots, res)
    for s in secrets_ + [str(root), root.name]:
        assert s not in public, s
    # the opaque id is what public records carry
    tr = _validate_receipts(res)[0]["validation"]["trusted_reference"]
    assert tr["private_evidence_id"] == D._private_root_id(root) and tr["isolation"]["private_evidence_id"] == tr["private_evidence_id"]
    assert all("<private-evidence:" in rec["phase_dir"] for ph, rec in tr["phases"].items() if ph in ("reference", "compare"))


# ---------------------------------------------------------------------------
# Manager CI finding (r9 head an earlier revision): torch's distributed package registers four
# torch._C extension submodules in sys.modules lazily, after the parent extension
# already existed; the census must admit them by descent (name + identity + no own
# file/spec + pre-existing attribute) and refuse a planted torch._C.evil.
# ---------------------------------------------------------------------------

_RUNNER_NAMES = ["_distributed_c10d", "_distributed_rpc", "_distributed_autograd", "_autograd"]


def test_torch_extension_submodules_registered_lazily_are_attributed_to_torch_c(tmp_path):
    """The runner's pattern: torch._C (an extension whose .so is hashed) existed before
    generated code ran; torch's Python code then registers extension submodules it
    exposes as attributes in sys.modules (torch.distributed does this for
    _distributed_c10d, _distributed_rpc, _distributed_autograd, _autograd).  The
    candidate does exactly that for every torch._C module attribute this build has
    not registered yet, and re-registers the runner's four names (same objects)."""
    payload = _phase_payload(
        "import sys as _sys, json as _json, types as _types, torch as _torch\nfrom pathlib import Path as _P\n"
        f"_runner = {_RUNNER_NAMES!r}\n"
        "_present = [n for n in _runner if hasattr(_torch._C, n)]\n"
        "for _n in _present:\n    _sys.modules.pop('torch._C.' + _n, None)\n"
        "for _n in _present:\n    _sys.modules['torch._C.' + _n] = getattr(_torch._C, _n)\n"
        "_lazy = sorted(k for k, v in _torch._C.__dict__.items() if isinstance(v, _types.ModuleType)\n"
        "              and 'torch._C.' + k not in _sys.modules and getattr(v, '__name__', '') == 'torch._C.' + k)\n"
        "for _k in _lazy:\n    _sys.modules['torch._C.' + _k] = getattr(_torch._C, _k)\n"
        "(_P(_f.f_globals['META']['phase_dir']) / 'ext_probe.json').write_text(_json.dumps({'present': _present, 'lazy': _lazy}))\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified", res.failure_receipts
    tr = res.validation["trusted_reference"]
    probe = json.loads((Path(tr["phases"]["candidate"]["phase_dir"]) / "ext_probe.json").read_text())
    assert probe["present"], "this torch build exposes none of the runner's four names"
    root = _private_roots(tmp_path / "out")[0]
    att = json.loads(next(root.glob("validation-*/dependency_attribution.json")).read_text())
    ext = att["candidate"]["extension_submodule_of"]
    # the attribution is in the private receipt only: not in the dispatcher's public records
    # (phase record, DISPATCH_STATE, DISPATCH_RECEIPT, result).  The child's own raw
    # report.json in its working root is a phase working file, not a dispatcher record.
    assert "extension_submodule_of" not in json.dumps(res.to_dict())
    assert "extension_submodule_of" not in (tmp_path / "out" / "DISPATCH_STATE.json").read_text()
    assert "extension_submodule_of" not in Path(res.receipt_path).read_text()
    assert "extension_submodule_of" not in json.dumps(tr["phases"]["candidate"])
    priv = _private_verdict(tr)
    assert priv["dependency_attribution"]["candidate"]["extension_submodule_of"] == ext
    if not probe["lazy"]:
        pytest.skip("this torch build registers every torch._C submodule eagerly; the lazy pattern could not be exercised")
    assert all(ext.get("torch._C." + k) == "torch._C" for k in probe["lazy"]), (probe["lazy"], ext)


def test_planted_torch_c_submodule_stays_untrusted(tmp_path):
    payload = _phase_payload(
        "import sys as _sys, types as _types, torch as _torch\n"
        "_sys.modules['torch._C.evil'] = _types.ModuleType('torch._C.evil')\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "reference_dependency_untrusted" and val["kind"] == FailureKind.INFRASTRUCTURE
    assert "torch._C.evil" in val["reason"] and "registered without passing through the import system" in val["reason"]
    tr = val["validation"]["trusted_reference"]
    assert "reference" not in tr["phases"]
    # the honest torch._C.* names registered by the same import are not what refused it
    assert "torch._C._distributed_c10d" not in val["reason"]


# ---------------------------------------------------------------------------
# Round 11 (the review an earlier review)
# ---------------------------------------------------------------------------

def _torch_c_prelude(body):
    """A prelude for the REFERENCE source: runs in the reference child, under the guard, after
    the snapshot (torch is imported by the reference source itself, as on the gate runner)."""
    return ("import sys as _sys, torch as _torch\n_C = _torch._C\n"
            f"_names = {_RUNNER_NAMES!r}\n" + body + "\n")


def _dependency_attribution(out_dir):
    root = _private_roots(out_dir)[0]
    return json.loads(next(root.glob("validation-*/dependency_attribution.json")).read_text())


def _dispatcher_records(out_dir, res):
    return [json.dumps(res.to_dict(), default=str), (Path(out_dir) / "DISPATCH_STATE.json").read_text(),
            Path(res.receipt_path).read_text()]


def _numbers_under(obj, key):
    """Every positive number stored under *key* (directly, or as the values of a dict)."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            v = x.get(key)
            vals = list(v.values()) if isinstance(v, dict) else [v]
            found.extend(n for n in vals if isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0)
            for y in x.values():
                walk(y)
        elif isinstance(x, list):
            for y in x:
                walk(y)
    walk(obj)
    return found


# -- CI-FAIL-27 root cause: pybind11 >= 3.0 copies the parent's __file__ onto def_submodule modules --

def test_torch_c_submodules_carrying_the_parents_file_are_attributed(tmp_path):
    """On the gate runner torch >= 2.9 (pybind11 3.0, #5584) registers torch._C._distributed_c10d,
    _distributed_rpc, _distributed_autograd and _autograd with __file__ == torch._C.__file__ (they
    are created by Python-called init functions after torch._C finished initialising; the
    submodules created inside the extension's own init carry none).  The reference source
    mirrors that on this torch build: the four are admitted -- name + identity + the parent's
    verified file -- and the basis is recorded in the private receipt only."""
    prelude = _torch_c_prelude(
        "for _n in _names:\n    _m = _sys.modules.get('torch._C.' + _n)\n"
        "    if _m is not None and getattr(_m, '__file__', None) is None:\n        _m.__file__ = _C.__file__\n")
    r, consume, _ = _cuda_case(tmp_path, source_edit=lambda src: prelude + src)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified", res.failure_receipts
    att = _dependency_attribution(tmp_path / "out")["reference"]
    present = [n for n in _RUNNER_NAMES if "torch._C." + n in att["extension_submodule_of"]]
    assert present, att
    for n in present:
        assert att["extension_submodule_of"]["torch._C." + n] == "torch._C"
        assert "parent's verified file" in att["attribution_basis"]["torch._C." + n]
        assert "imported through the guard" in att["attribution_basis"]["torch._C." + n]
    assert att["census_diagnostic"] == {}
    for text in _dispatcher_records(tmp_path / "out", res):
        assert "attribution_basis" not in text and "census_diagnostic" not in text


def test_torch_c_submodule_naming_another_file_is_refused_and_explained(tmp_path):
    """A torch._C.* entry whose __file__ names any file other than the parent's verified file is
    refused; the public reason names the module and the failed condition's code; the private
    census diagnostic carries the parent's census state; no public record carries it."""
    prelude = _torch_c_prelude("_m = _sys.modules['torch._C.' + _names[0]]\n_m.__file__ = _torch.__file__\n")
    r, consume, _ = _cuda_case(tmp_path, source_edit=lambda src: prelude + src)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "reference_dependency_untrusted" and val["kind"] == FailureKind.INFRASTRUCTURE
    assert "module torch._C._distributed_c10d registered without passing through the import system " \
           "[census: own_file_not_the_parent_file]" in val["reason"]
    assert "torch._C._distributed_rpc" not in val["reason"]
    diag = _dependency_attribution(tmp_path / "out")["reference"]["census_diagnostic"]["torch._C._distributed_c10d"]
    assert diag["failed"] == "own_file_not_the_parent_file" and diag["own_file_is_parent_file"] is False
    assert diag["parent"] == "torch._C" and diag["exposed_as"] == "_distributed_c10d" and diag["own_file_absolute"]
    assert diag["parent_in_snapshot"] is False          # torch is imported by the reference source, under the guard
    assert diag["parent_guard_kind"] == "file" and diag["parent_fingerprint_unchanged"] is True
    assert diag["parent_under_root"] is True and diag["parent_origin"].endswith(".so")
    for text in _dispatcher_records(tmp_path / "out", res):
        assert "census_diagnostic" not in text and diag["parent_origin"] not in text and diag["own_file"] not in text


@pytest.mark.parametrize("with_file", [False, True])
def test_planted_torch_c_submodule_stays_untrusted_with_or_without_the_parents_file(tmp_path, with_file):
    body = "_sys.modules['torch._C.evil'] = _types.ModuleType('torch._C.evil')\n"
    if with_file:
        body += "_sys.modules['torch._C.evil'].__file__ = _torch._C.__file__\n"
    payload = _phase_payload("import sys as _sys, types as _types, torch as _torch\n" + body)
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    assert val["reason_code"] == "reference_dependency_untrusted"
    assert "module torch._C.evil registered without passing through the import system " \
           "[census: not_identical_to_a_parent_attribute]" in val["reason"]
    diag = _dependency_attribution(tmp_path / "out")["candidate"]["census_diagnostic"]["torch._C.evil"]
    assert diag["failed"] == "not_identical_to_a_parent_attribute" and diag["exposed_as"] == ""
    assert diag["own_file_absolute"] is with_file


# -- NUMERIC-FEEDBACK-30/A: resumable outcomes withhold the statistics ---------------------------

def test_resumable_receipt_and_result_withhold_the_statistics(tmp_path):
    """An awaiting_agent outcome after a mismatch is resumable: its DISPATCH_RECEIPT, its returned
    result and the state file carry NO error statistic (the next candidate can read all three);
    the terminal receipt carries them."""
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    agent = _byte_distinct(consume)
    first = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider())
    assert first.status == "awaiting_agent"
    agent(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    mid = resume(r.out_dir, PureSkillProvider())
    assert mid.status == "awaiting_agent"
    receipt = json.loads((r.out_dir / "DISPATCH_RECEIPT.json").read_text())
    state = json.loads((r.out_dir / "DISPATCH_STATE.json").read_text())
    assert receipt["numeric_statistics"].startswith("withheld")
    for src in (mid.to_dict(), receipt, state):
        assert _numbers_under(src, "max_abs_error") == [] and _numbers_under(src, "per_output_max_abs_error") == []
        txt = json.dumps(src, default=str)
        assert not re.search(r"max_abs_error=\s*[0-9]", txt) and D._PRIVATE_MARK in txt
        fails = [x for x in src["failure_receipts"] if x["stage"] == "agentic_validate"]
        assert fails and all(x["validation"]["trusted_reference"]["outputs_pass"] == {"output": False} for x in fails)
    agent(Path(mid.resume_token) / "TRANSLATION_REQUEST.json")
    last = resume(r.out_dir, PureSkillProvider())
    assert not last.verified and last.status != "awaiting_agent"
    final = json.loads(Path(last.receipt_path).read_text())
    assert final["numeric_statistics"] == "final"
    nums = _numbers_under(final, "max_abs_error")
    assert len(nums) >= 2 and all(n == pytest.approx(1.0) for n in nums)
    assert _numbers_under(last.to_dict(), "max_abs_error") == nums


# -- NUMERIC-FEEDBACK-30/B: the public digest depends on no withheld field -----------------------

def test_public_verdict_digest_is_over_the_redacted_view(tmp_path):
    """The public verdict digest is recomputable from the public compare record alone and does
    not change with the hidden statistics (guessing them yields no match); the digest over the
    full verdict is in verdict_private.json only."""
    import types as _types
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = _validate_receipts(res)[0]
    tr = val["validation"]["trusted_reference"]
    ph = tr["phases"]["compare"]
    meta = json.loads((_phase_dir(tr, "compare") / "meta.json").read_text())
    fake = _types.SimpleNamespace(env={k: v for k, v in meta["env_expected"]})
    # the launch argv names the private root by its opaque token in public records; a candidate
    # resolves it by enumerating agentic/ (the review did) -- the digest binds the real launch
    root = _private_roots(tmp_path / "out")[0]
    argv = [re.sub(r"<private-evidence:[^>]+>", str(root), a) for a in ph["argv"]]

    def digest(verdict):
        return D._TrustedPhases.digest_for(fake, verdict, {"argv": argv})
    state = json.loads((tmp_path / "out" / "DISPATCH_STATE.json").read_text())
    ph_state = [x for x in state["failure_receipts"] if x["stage"] == "agentic_validate"][0] \
        ["validation"]["trusted_reference"]["phases"]["compare"]
    assert ph_state["verdict"]["max_abs_error"] is None and "<n>" in ph_state["verdict"]["error"]
    assert tr["verdict_sha256"] == digest(ph_state["verdict"])           # from the public view alone
    assert tr["verdict_sha256"] == digest(D._public_verdict_view(ph["verdict"]))
    assert "attempt-visible" in tr["verdict_digest_basis"]
    full = {k: v for k, v in ph["verdict"].items() if k != "artifact_sha256"}
    assert full["max_abs_error"] == pytest.approx(1.0)
    for guess in (0.5, 1.0, 2.0, 4.0):              # the reviewer's finite-domain reconstruction
        probe = dict(full, max_abs_error=guess, per_output_max_abs_error={"output": guess})
        assert digest(probe) != tr["verdict_sha256"]
    priv = _private_verdict(tr)
    assert priv["verdict_sha256"] != tr["verdict_sha256"] and priv["verdict_sha256"] == digest(priv["verdict"])
    for text in _dispatcher_records(tmp_path / "out", res):
        assert priv["verdict_sha256"] not in text


# -- RESULT-BOUNDARY-29: final size re-check; exhausted budget refuses before any read ------------

def test_refstat_rechecks_the_size(tmp_path, monkeypatch):
    p = tmp_path / "result"
    p.write_bytes(b"abc")
    real_read = D._TRUSTED_READ

    def append_after_eof(fd, n):
        b = real_read(fd, n)
        if b == b"":
            with p.open("ab") as fh:
                fh.write(b"x" * 32)
        return b
    monkeypatch.setattr(D, "_TRUSTED_READ", append_after_eof)
    r = D._ingest_result_file(p, 16, label="growing", seconds=1)
    assert not r["ok"] and r["bytes"] is None and "changed size" in r["problem"] and r["size"] == 35


def test_expired_budget_refuses_the_ingest_without_reading(tmp_path, monkeypatch):
    p = tmp_path / "result"
    p.write_bytes(b"abc")
    calls = []
    real_open = D._TRUSTED_OS_OPEN
    monkeypatch.setattr(D, "_TRUSTED_OS_OPEN", lambda *a: calls.append(a) or real_open(*a))
    token = D._DISPATCH_DEADLINE.set(time.monotonic() - 1)
    try:
        bound = D._ingest_seconds(1.0)
        r = D._ingest_result_file(p, 16, label="late", seconds=bound)
    finally:
        D._DISPATCH_DEADLINE.reset(token)
    assert bound == D._INGEST_MIN_SECONDS == 0.5          # the grace, reported as such
    assert not r["ok"] and r["timed_out"] and "exhausted" in r["problem"] and r["seconds_bound"] == 0.0
    assert calls == []                                      # nothing opened, nothing read
    # with budget remaining the grace applies and the file is read
    token = D._DISPATCH_DEADLINE.set(time.monotonic() + 0.2)
    try:
        r2 = D._ingest_result_file(p, 16, label="soon", seconds=D._ingest_seconds(1.0))
    finally:
        D._DISPATCH_DEADLINE.reset(token)
    assert r2["ok"] and r2["bytes"] == b"abc" and r2["seconds_bound"] == 0.5


# ---------------------------------------------------------------------------
# Round 12 -- the review NUMERIC-FEEDBACK-30/C: no metadata of the unredacted comparison report (or of
# any private verdict) in attempt-visible records; the whole attempt-visible record is a function
# of the public representation only.
# ---------------------------------------------------------------------------

_PUBLIC_INGEST_SHAPE = {"label", "ok", "kind", "cap_bytes", "timed_out", "problem", "evidence"}


def _awaiting_after_mismatch(r, agent):
    """dispatch -> agent writes the (wrong) candidate -> resume: the mismatch is validated and the
    dispatch is again awaiting the agent.  Returns (mid result, state text, receipt text)."""
    first = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider())
    assert first.status == "awaiting_agent"
    agent(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    mid = resume(r.out_dir, PureSkillProvider())
    assert mid.status == "awaiting_agent"
    return (mid, (r.out_dir / "DISPATCH_STATE.json").read_text(),
            (r.out_dir / "DISPATCH_RECEIPT.json").read_text())


def test_attempt_visible_ingest_records_carry_no_content_metadata(tmp_path):
    """the review NUMERIC-FEEDBACK-30/C: while the dispatch can still be resumed, every result_ingest
    record in state, receipt and returned result is the fixed public shape -- no size, no link
    count, no time bound, numerals in the problem text redacted -- and no duration is published;
    the terminal receipt restores the full ingest evidence and the durations."""
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    agent = _byte_distinct(consume)
    mid, state_text, receipt_text = _awaiting_after_mismatch(r, agent)
    for src in (json.loads(state_text), json.loads(receipt_text), mid.to_dict()):
        fails = [x for x in src["failure_receipts"] if x["stage"] == "agentic_validate"]
        assert fails
        for x in fails:
            val = x["validation"]
            assert val["seconds"] is None                       # validation duration withheld
            tr = val["trusted_reference"]
            assert tr["seconds"] is None                        # trusted-reference duration withheld
            assert set(tr["phases"]) >= {"candidate", "reference", "compare"}
            for ph, rec in tr["phases"].items():
                for label, ing in rec["result_ingest"].items():
                    assert set(ing) == _PUBLIC_INGEST_SHAPE, (ph, label, sorted(ing))
                    assert ing["cap_bytes"] in (D._REPORT_BYTES_CAP, D._RECEIPT_BYTES_CAP) or ing["cap_bytes"] > 0
                    assert not re.search(r"\d", ing["problem"] or "")
                assert "stderr_tail" not in rec or not re.search(r"\d", rec["stderr_tail"])
            ing = tr["phases"]["compare"]["result_ingest"]["report"]
            assert ing["ok"] is True and ing["kind"] == "regular file" and ing["timed_out"] is False
        # the reviewer's reconstruction target: nothing under these keys anywhere in the record
        for key in ("size", "nlink", "seconds_bound"):
            assert _numbers_under(src, key) == [], key
    # the terminal receipt and result carry the full ingest evidence again
    agent(Path(mid.resume_token) / "TRANSLATION_REQUEST.json")
    last = resume(r.out_dir, PureSkillProvider())
    assert not last.verified and last.status != "awaiting_agent"
    for src in (json.loads(Path(last.receipt_path).read_text()), last.to_dict()):
        fails = [x for x in src["failure_receipts"] if x["stage"] == "agentic_validate"]
        assert len(fails) == 2
        for x in fails:
            tr = x["validation"]["trusted_reference"]
            ing = tr["phases"]["compare"]["result_ingest"]["report"]
            assert ing["ok"] and ing["size"] > 0 and ing["nlink"] == 1 and ing["seconds_bound"] > 0
            assert x["validation"]["seconds"] > 0 and tr["seconds"] > 0
    # and the private root holds them throughout
    root = _private_roots(tmp_path / "out")[0]
    store = json.loads((root / D._NUMERIC_STORE).read_text())
    assert all(e["validation"]["trusted_reference"]["phases"]["compare"]["result_ingest"]["report"]["size"] > 0
               for e in store.values())


# -- the property: attempt-visible records are a function of the public representation only --------

_HIDDEN_SHIFT_FILE = "hidden-shift.txt"
_HIDDEN_SHIFTS = (0.0, 0.875, -15.0, -127.0)            # candidate = x*w + 2.0; error = |1 - shift|
_HIDDEN_ERRORS = (1.0, 0.125, 16.0, 128.0)


def _hidden_shift_source(hidden_path):
    """A reference-source edit whose BYTES are the same for every run: ``run`` adds a shift read
    from *hidden_path* in the REFERENCE phase only (verify/candidate frames see 0.0).  The hidden
    comparison outcome varies with the file; the request, the source bytes, the candidate and
    every public input stay identical."""
    # inline in ``run`` (a second top-level callable would make the reference entry ambiguous)
    body = ("    assert x.shape[1] == 16\n"
            "    import sys as _s\n"
            "    _shift = 0.0\n"
            "    _f = _s._getframe()\n"
            "    while _f is not None:\n"
            "        if _f.f_globals.get('PHASE') == 'reference':\n"
            "            from pathlib import Path as _P\n"
            f"            _shift = float(_P({str(hidden_path)!r}).read_text())\n"
            "            break\n"
            "        _f = _f.f_back\n"
            "    return x * w + 1.0 + _shift")

    def edit(src):
        old = "    assert x.shape[1] == 16\n    return x * w + 1.0"
        assert old in src
        return src.replace(old, body)
    return edit


# Fields the property test normalises, each with the reason it is PROVABLY independent of the
# comparison outcome.  Nothing else is normalised: a difference anywhere else fails the test.
_CLOCK_KEYS = {"utc", "created_utc", "started_utc", "updated_utc", "generated_utc",
               "monotonic_start", "deadline_monotonic", "started_epoch",
               "started_monotonic",                              # the dispatch's own clock readings
               "ctime_ns", "mtime_ns"}                           # inode times of files written before the comparison
# round 13: wall_seconds_elapsed / wall_seconds_remaining are NOT normalised any more -- they are
# withheld (None) in every attempt-visible record, and the whole-surface test asserts that
_KERNEL_KEYS = {"pid", "sid", "pgid", "leader_pid",     # assigned at process launch, before the phase runs
                "starttime", "started_pid"}             # the dispatching process's kernel start time / pid
_FS_KEYS = {"ino"}                                      # inode numbers of files the parent created before the
                                                        # comparison (the inputs, the private root)
_TOKEN_KEYS = {"_private_numeric_id", "reason_private_id",      # secrets.token_hex, drawn by the parent
               "private_evidence_id", "id",                      # the private root's random id
               "request_id"}                                     # sha256(source:created_utc:attempt): a clock reading
# digests that are functions of pre-comparison data only (proved separately, see the test):
#  * receipt_sha256 -- the verify child's receipt (it carries its own ``utc``), emitted in the verify
#    phase, which precedes the candidate, reference and compare phases;
#  * verdict_sha256 -- digest over the PUBLIC verdict view bound to the launch argv, which names the
#    per-dispatch random private root; recomputed from public data in every run by
#    ``_dispatch_with_hidden_shift`` (the public verdict view itself must be identical: it is not
#    normalised).
_BOUND_DIGEST_KEYS = {"receipt_sha256", "verdict_sha256"}
_RANDOM_NAME_RE = re.compile(r"\b(validation|reference)-[0-9a-f]{8}(?:[0-9a-f]{8})?\b|\b(candidate)-[0-9a-z_]{8}\b")
_ROOT_TOKEN_RE = re.compile(r"<private-evidence:[0-9a-f]+>")
_ROOT_NAME_RE = re.compile(r"\.private-evidence-[0-9a-f]{16}\b")       # the private root's random directory name


def _normalise_independent(doc):
    """Replace ONLY the provably comparison-independent fields (and the random names / tokens
    they introduce into path strings) by placeholders; return (normalised doc, touched paths)."""
    tokens = set()

    def collect(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in _TOKEN_KEYS and isinstance(v, str) and v:
                    tokens.add(v)
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
    collect(doc)
    touched = []

    def scrub(text):
        for t in sorted(tokens, key=len, reverse=True):
            text = text.replace(t, "<token>")
        text = _ROOT_TOKEN_RE.sub("<private-evidence:ID>", text)
        return _RANDOM_NAME_RE.sub(lambda m: (m.group(1) or m.group(2)) + "-ID", text)

    def walk(x, path):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                p = path + (k,)
                if k in _CLOCK_KEYS:
                    touched.append("/".join(p)); out[k] = "<clock>"
                elif k in _KERNEL_KEYS:
                    touched.append("/".join(p)); out[k] = "<kernel-id>"
                elif k in _FS_KEYS:
                    touched.append("/".join(p)); out[k] = "<inode>"
                elif k in _BOUND_DIGEST_KEYS and isinstance(v, str):
                    touched.append("/".join(p)); out[k] = "<pre-comparison-bound digest>"
                elif k in _TOKEN_KEYS:
                    touched.append("/".join(p)); out[k] = "<token>"
                elif path[:1] == ("deterministic",) and k in ("seconds", "elapsed_seconds"):
                    touched.append("/".join(p)); out[k] = "<pre-agentic timing>"
                else:
                    out[k] = walk(v, p)
            return out
        if isinstance(x, list):
            return [walk(v, path + (str(i),)) for i, v in enumerate(x)]
        if isinstance(x, str):
            return scrub(x)
        return x
    return walk(doc, ()), touched


def _differing_paths(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (present in one run only)")
            else:
                out += _differing_paths(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (list length {len(a)} vs {len(b)})"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _differing_paths(x, y, f"{path}/{i}")]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"[:300]]


def _dispatch_with_hidden_shift(tmp_path, shift):
    """One two-attempt schedule at the SAME paths as every other run (the case dir is renamed away
    afterwards): returns the attempt-visible texts captured while the dispatch was resumable, and
    the terminal error statistics that prove which hidden outcome this run had."""
    (tmp_path / _HIDDEN_SHIFT_FILE).write_text(repr(float(shift)))
    case = tmp_path / "case"
    case.mkdir()
    r, consume, _ = _cuda_case(case, adapter_edit=_diverge, dsl_edit=_wrong_dsl,
                               source_edit=_hidden_shift_source(tmp_path / _HIDDEN_SHIFT_FILE))
    agent = _byte_distinct(consume)
    mid, state_text, receipt_text = _awaiting_after_mismatch(r, agent)
    # the public verdict digest is a function of PUBLIC data only: the attempt-visible verdict view
    # and the launch binding (argv naming the random private root, env, program hash) -- recomputed
    # here from the public record, so normalising it in the comparison below is justified
    import types as _types
    tr = [x for x in mid.to_dict()["failure_receipts"] if x["stage"] == "agentic_validate"][0] \
        ["validation"]["trusted_reference"]
    ph = tr["phases"]["compare"]
    root = _private_roots(case / "out")[0]
    meta = json.loads((_phase_dir(tr, "compare") / "meta.json").read_text())
    fake = _types.SimpleNamespace(env={k: v for k, v in meta["env_expected"]})
    argv = [re.sub(r"<private-evidence:[^>]+>", str(root), a) for a in ph["argv"]]
    assert tr["verdict_sha256"] == D._TrustedPhases.digest_for(fake, ph["verdict"], {"argv": argv})
    agent(Path(mid.resume_token) / "TRANSLATION_REQUEST.json")
    last = resume(r.out_dir, PureSkillProvider())
    final = json.loads(Path(last.receipt_path).read_text())
    return {"state": state_text, "receipt": receipt_text,
            "result": json.dumps(mid.to_dict(), default=str, sort_keys=True),
            "errors": sorted(set(_numbers_under(final, "max_abs_error"))), "verified": last.verified,
            "source_sha256": _sha(case / "reference.py")}


def test_attempt_visible_records_do_not_depend_on_the_hidden_comparison(tmp_path):
    """PROPERTY (the review NUMERIC-FEEDBACK-30/C ruling), finite domain: dispatches that differ ONLY
    in the hidden comparison outcome -- four distinct FINITE error values, byte-identical
    request/source/candidate, identical paths -- publish attempt-visible DISPATCH_STATE.json,
    intermediate DISPATCH_RECEIPT.json and returned result whose PARSED documents are equal after
    normalising only the fields listed in ``_normalise_independent`` (clock readings, kernel ids,
    inode numbers, pre-comparison-bound digests, random tokens and names).  What this test checks:
    parsed-document equality (dict equality, order-insensitive) and that the normaliser touched the
    same field set in every run.  What it does not check: raw bytes, key order, the public tree, the
    dispatcher log, non-finite or vacuous statistics -- those are the whole-surface property
    ``test_whole_public_surface_does_not_depend_on_the_hidden_comparison`` (round 13)."""
    runs = []
    for i, shift in enumerate(_HIDDEN_SHIFTS):
        runs.append(_dispatch_with_hidden_shift(tmp_path, shift))
        os.rename(tmp_path / "case", tmp_path / f"run-{i}")
    # the four runs really had four distinct hidden outcomes, visible only in the terminal receipts
    assert [r["errors"] for r in runs] == [[pytest.approx(e)] for e in _HIDDEN_ERRORS]
    assert not any(r["verified"] for r in runs)
    assert len({r["source_sha256"] for r in runs}) == 1                     # identical source bytes
    for doc in ("state", "receipt", "result"):
        norms, touched = zip(*(_normalise_independent(json.loads(r[doc])) for r in runs))
        diffs = sorted({p for n in norms[1:] for p in _differing_paths(norms[0], n)})
        assert not diffs, f"{doc}: attempt-visible fields vary with the hidden comparison outcome:\n" + "\n".join(diffs)
        # the normaliser touched only the declared independent fields (same set in every run)
        assert len({tuple(sorted(set(t))) for t in touched}) == 1


# ---------------------------------------------------------------------------
# round 13 (the review an earlier review): constant-shape public document, canonical non-finite
# statistics, one anomaly class, withheld wall-clock telemetry, the whole-surface property
# ---------------------------------------------------------------------------

def _r13_full_record(variant):
    """A full validation record (what ``ValidationVerdict.to_dict()`` yields) in two variants that differ
    in every private-number-dependent way the reviewer's probe exploited: the dropped statistic (key absent,
    counter 2), vacuous outputs, kernel ids and inode facts of different widths, durations, the message."""
    tol = {"rtol": 1e-4, "atol": 1e-5, "trusted_dtype_floor": 8.9e-16, "rtol_effective": 1e-4, "atol_effective": 1e-5}
    verdict = {"success": False, "error": "Output 'output' values differ beyond the tolerance policy (rtol=0.0001, atol=1e-05)",
               "max_abs_error": 1.0, "vacuous_outputs": [], "compared_outputs": ["output"], "outputs_pass": {"output": False},
               "per_output_max_abs_error": {"output": 1.0}, "tolerance": dict(tol),
               "artifact_sha256": {k: "0" * 64 for k in ("inputs", "trusted_outputs", "candidate_outputs")}}
    schema = {"allowlisted": True, "unknown_or_mistyped_dropped": 0, "truncated": 0, "parsed": True, "ingest_before_unlock": True}
    ids = {"pid": 4242, "sid": 4242, "pgid": 4242}
    obs = {"ino": 123456, "ctime_ns": 1793115819232998699, "mtime_ns": 1793115819232998699, "sha256": "a" * 64, "mode": 33024}
    verify = {"status": "fail", "message": "max_abs_error=1.0 beyond rtol", "max_abs_error": 1.0, "vacuous_outputs": []}
    message = "trusted_reference_mismatch: outputs failing: ['output']; error 1.0"
    seconds = 2.5
    if variant == "nonfinite-dropped":
        verdict = {k: v for k, v in verdict.items() if k not in ("max_abs_error", "per_output_max_abs_error")}
        verdict["vacuous_outputs"] = ["output"]
        schema = dict(schema, unknown_or_mistyped_dropped=2)
        ids = {"pid": 123456, "sid": 123456, "pgid": 123456}
        obs = dict(obs, ino=7, ctime_ns=1793115819232998700)
        verify = {"status": "fail", "message": "max_abs_error=inf beyond rtol"}       # non-finite: keys absent
        message = "trusted_verdict_unverifiable: the compare child's verdict digest disagrees with the parent's digest"
        seconds = 31.75
    return {"status": "mismatch", "kind": "unsupported", "message": message, "max_abs_error": 1.0, "receipt_path": None,
            "verify_status": "fail", "seconds": seconds, "tolerance": dict(tol), "tolerance_source": "request",
            "reference_binding": None, "reason_code": "trusted_reference_mismatch", "public_message": None,
            "trusted_reference": {
                "max_abs_error": 1.0, "per_output_max_abs_error": {"output": 1.0}, "seconds": seconds / 2,
                "session_boundaries": {"compare": {"leader_pid": ids["pid"], "clean": True}},
                "input_audit": {"parent_observation": {"before": {"x": dict(obs)}, "after": {"x": dict(obs)}}},
                "phases": {"verify": {**ids, "returncode": 0, "report_schema": dict(schema), "verify": verify},
                           "compare": {**ids, "returncode": 0, "report_schema": dict(schema), "verdict": verdict,
                                       "program_sha256": "f" * 64}}}}


def test_attempt_visible_record_is_a_constant_shape_document():
    """the review NUMERIC-FEEDBACK-30/D ruling (a): the attempt-visible record has a fixed key set in every
    record; statistic keys and vacuous_outputs are present and None; report_schema carries no parse
    counter; kernel ids and inode facts are withheld; a verdict with a constant public message shows
    exactly that text.  Two full records that differ in all of these project to the SAME document,
    byte for byte including key order."""
    a = D._attempt_visible(_r13_full_record("finite"), "k", public_message="C")
    b = D._attempt_visible(_r13_full_record("nonfinite-dropped"), "k", public_message="C")
    assert json.dumps(a) == json.dumps(b), _differing_paths(a, b)
    v = a["trusted_reference"]["phases"]["compare"]["verdict"]
    assert list(v) == list(D._PUBLIC_VERDICT_KEYS)
    assert v["max_abs_error"] is None and v["per_output_max_abs_error"] is None and v["vacuous_outputs"] is None
    assert v["error"] == "Output 'output' values differ beyond the tolerance policy (rtol=<n>, atol=<n>)"
    assert list(v["tolerance"]) == list(D._PUBLIC_TOLERANCE_KEYS) and v["outputs_pass"] == {"output": False}
    assert "artifact_sha256" not in v
    rs = a["trusted_reference"]["phases"]["compare"]["report_schema"]
    assert rs["allowlisted"] is True and rs["ingest_before_unlock"] is True
    assert rs["unknown_or_mistyped_dropped"] is None and rs["truncated"] is None and rs["parsed"] is None
    ph = a["trusted_reference"]["phases"]["compare"]
    assert ph["pid"] is None and ph["sid"] is None and ph["pgid"] is None and ph["returncode"] == 0
    assert a["trusted_reference"]["session_boundaries"]["compare"] == {"leader_pid": None, "clean": True}
    x = a["trusted_reference"]["input_audit"]["parent_observation"]["before"]["x"]
    assert x["ino"] is None and x["ctime_ns"] is None and x["mtime_ns"] is None and x["sha256"] == "a" * 64
    assert a["seconds"] is None and a["trusted_reference"]["seconds"] is None and a["max_abs_error"] is None
    vv = a["trusted_reference"]["phases"]["verify"]["verify"]
    assert vv["max_abs_error"] is None and vv["vacuous_outputs"] is None and vv["message"] == "max_abs_error=<n> beyond rtol"
    assert a["message"] == "C"
    # without a public message the (redacted) private text is shown -- the two variants then differ
    # ONLY in that text, which the product makes constant for the anomaly class (public_message)
    a2 = D._attempt_visible(_r13_full_record("finite"), "k")
    b2 = D._attempt_visible(_r13_full_record("nonfinite-dropped"), "k")
    assert _differing_paths(a2, b2) == [f"/message: {a2['message']!r} vs {b2['message']!r}"[:300]]
    # the public verdict view (the digest basis) is the same fixed shape
    full = _r13_full_record("nonfinite-dropped")["trusted_reference"]["phases"]["compare"]["verdict"]
    view = D._public_verdict_view(full)
    assert list(view) == list(D._PUBLIC_VERDICT_KEYS) and view["max_abs_error"] is None and view["vacuous_outputs"] is None
    assert view == b["trusted_reference"]["phases"]["compare"]["verdict"]


def test_parser_accepts_canonical_nonfinite_statistics_and_the_digests_agree():
    """the review NUMERIC-FEEDBACK-30/D ruling (b): the compare child's canonical tokens parse whole (no drop,
    key present), and the parent's digest over the parsed verdict equals the child's digest over its own
    verdict -- the exact construction of ``_trusted_child.py``.  A JSON Infinity/NaN literal (what the
    child used to write) and any other string are still dropped and counted; digest validation stays."""
    import types
    verdict = {"success": False, "error": "Output 'output' values differ beyond the tolerance policy (rtol=0.0001, atol=1e-05)",
               "max_abs_error": "inf", "vacuous_outputs": [], "compared_outputs": ["output"], "outputs_pass": {"output": False},
               "per_output_max_abs_error": {"output": "inf", "other": "nan", "fine": 0.5, "neg": "-inf"},
               "tolerance": {"rtol": 1e-4, "atol": 1e-5, "trusted_dtype_floor": 8.9e-16, "rtol_effective": 1e-4, "atol_effective": 1e-5},
               "artifact_sha256": {k: "0" * 64 for k in ("inputs", "trusted_outputs", "candidate_outputs")}}
    env = [["CUDA_VISIBLE_DEVICES", ""], ["PATH", "/usr/bin"]]
    child_argv = [str(D._TRUSTED_CHILD_PATH), "compare", "/x/meta.json"]
    child_bound = {"verdict": verdict, "program_sha256": D._TRUSTED_PROGRAM_SHA256, "argv": child_argv, "env": env}
    child_digest = hashlib.sha256(json.dumps(child_bound, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    raw = {"ok": True, "phase": "compare", "verdict": verdict, "verdict_sha256": child_digest, "program_sha256": D._TRUSTED_PROGRAM_SHA256}
    rep, stats = D._parse_report(json.dumps(raw).encode(), "compare")
    assert stats == {"dropped": 0, "truncated": 0, "parsed": True}
    assert rep["verdict"]["max_abs_error"] == "inf"
    assert rep["verdict"]["per_output_max_abs_error"] == {"output": "inf", "other": "nan", "fine": 0.5, "neg": "-inf"}
    fake = types.SimpleNamespace(env=dict(env))
    launch = [sys.executable, "-I", "-S"] + child_argv
    assert D._TrustedPhases.digest_for(fake, rep["verdict"], {"argv": launch}) == child_digest
    # the old representation: a JSON Infinity literal is dropped and counted (and would fail the digest)
    legacy = dict(raw, verdict={**verdict, "max_abs_error": float("inf"), "per_output_max_abs_error": {"output": float("inf")}})
    rep2, stats2 = D._parse_report(json.dumps(legacy).encode(), "compare")
    assert stats2["dropped"] == 2 and "max_abs_error" not in rep2["verdict"] and rep2["verdict"]["per_output_max_abs_error"] == {}
    rep3, stats3 = D._parse_report(json.dumps(dict(raw, verdict={**verdict, "max_abs_error": "huge"})).encode(), "compare")
    assert stats3["dropped"] == 1 and "max_abs_error" not in rep3["verdict"]
    # the child's own canonicalisation, verbatim from the compare phase
    child = D._TRUSTED_CHILD
    assert 'return "inf" if x > 0 else "-inf"' in child and 'max_err = "nan" if nan_seen else canonical_stat(max_err)' in child
    assert 'with np.errstate(all="ignore")' in child


def test_caps_remaining_is_derived_from_the_absolute_deadline(monkeypatch):
    """the review PUBLIC-PROPERTY-SCOPE-31 (c): the request carries no remaining-wall reading; the providers
    derive their bound from deadline_utc (fixed at dispatch start) and their own clock -- also when that
    clock is a substitute offering ``time()`` only (the review controls' Clock has no ``strptime``)."""
    assert D._caps_remaining(None) is None and D._caps_remaining({}) is None
    assert D._caps_remaining({"deadline_utc": None, "wall_seconds_remaining": None}) is None
    soon = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 120))
    r = D._caps_remaining({"deadline_utc": soon, "wall_seconds_remaining": None})
    assert 117.0 <= r <= 120.0
    assert D._caps_remaining({"deadline_utc": "2000-01-01T00:00:00Z"}) == 0.0
    assert D._caps_remaining({"wall_seconds_remaining": 7.5, "deadline_utc": soon}) == 7.5   # older request files
    assert D._caps_remaining({"deadline_utc": "not a time"}) is None

    class _Clock:            # what the reviewer's round-2/round-6 controls stand in for the module's ``time``
        epoch = 100.0

        def time(self):
            return self.epoch

    monkeypatch.setattr(D, "time", _Clock())
    assert D._caps_remaining({"deadline_utc": "1970-01-01T00:01:41Z"}) == 1.0
    assert D._caps_remaining({"deadline_utc": "1970-01-01T00:01:40Z"}) == 0.0
    assert D._caps_remaining({"deadline_utc": "1970-01-01T00:01:39Z"}) == 0.0


# -- the whole-surface property ------------------------------------------------------------------------

_HIDDEN_OUTCOME_FILE = "hidden-reference.txt"
# (name, hidden first element of the trusted reference, the private max_abs_error it produces).  The
# candidate's first element is 1e308 (the reviewer's payload), the other elements are off by exactly 1.0:
#   +1e308  -> 1.0            the review schema-positive, VERBATIM            (finite)
#   -1e308  -> inf            the review schema-negative, VERBATIM: 1e308 - (-1e308) overflows   (+inf)
#    5e307  -> 5e307          a large finite value (different textual width)                   (finite)
#    nan    -> nan            the reference carries a nan                                      (nan)
#    0.0    -> 1e308          the ALL-ZERO reference: vacuous_outputs == ['output'] privately  (vacuous)
# A -inf statistic cannot occur: the statistic is a maximum of absolute differences; a reference carrying
# inf itself gives the same canonical token as the overflow case and is not run (each case is ~70 s).
_HIDDEN_OUTCOMES = (("control-positive", 1e308, 1.0), ("control-negative", -1e308, "inf"),
                    ("finite-large", 5e307, 5e307), ("nan", float("nan"), "nan"),
                    ("vacuous-all-zero", 0.0, 1e308))

# the reviewer's reference edit (checks-76fddbd3/test_round12_schema_channel_corrected.py), verbatim, plus one
# branch for the all-zero reference (hidden value 0.0, never used by the reviewer's two cases)
_R12_REFERENCE_BODY = """    _result = x * w + 1.0
    import sys as _sys
    _frame = _sys._getframe()
    while _frame is not None:
        if _frame.f_globals.get('PHASE') == 'reference':
            from pathlib import Path as _Path
            _result = _result.double()
            _hidden = float(_Path(HIDDEN_PATH).read_text())
            if _hidden == 0.0:
                _result = _result * 0.0
            else:
                _result.reshape(-1)[0] = _hidden
            break
        _frame = _frame.f_back
    return _result"""
# the reviewer's adapter expression (corrected probe), verbatim
_R12_ADAPTER_EDIT = lambda a: a.replace('exec(get_source(), ns)', 'exec(get_source().replace("_result = x * w + 1.0", "_result = x * w + 2.0"), ns)')  # noqa: E731
# the reviewer's candidate payload, verbatim: PREFIX (checks-76fddbd3/test_round9_independent.py) + the probe
# (test_round12_schema_channel_corrected.py).  The probe reads the PUBLIC state on the second attempt and
# picks the hidden value from the parse counter; with the counter withheld it is a blind guess.
_R9_PREFIX = """from pathlib import Path as _P
import os as _os, json as _json, numpy as _np
_g = _f.f_globals
_m = _g["META"]
_croot = _P(_m["phase_dir"])
_agentic = _croot.parents[2]
"""
_R12_PAYLOAD = _R9_PREFIX + """
result=result.astype(_np.float64)
result.reshape(-1)[0]=1e308
_state=_json.loads((_agentic.parent/'DISPATCH_STATE.json').read_text())
_seen=[]
for _rec in _state.get('failure_receipts',[]):
    _tr=(_rec.get('validation') or {}).get('trusted_reference') or {}
    _cp=(_tr.get('phases') or {}).get('compare') or {}
    if _cp:
        _seen.append({'schema':_cp.get('report_schema'),'verdict':_cp.get('verdict')})
if _seen:
    _dropped=_seen[-1]['schema']['unknown_or_mistyped_dropped']
    result=result-1.0
    result.reshape(-1)[0]=-1e308 if _dropped==2 else 1e308
(_croot/'public-schema-observation.json').write_text(_json.dumps({'observations':_seen,'private_files_opened':False}))
"""
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_WITHHELD_NUMBER_KEYS = ("size", "nlink", "seconds_bound", "seconds", "pid", "sid", "pgid", "leader_pid", "ino",
                         "ctime_ns", "mtime_ns", "max_abs_error", "per_output_max_abs_error",
                         "unknown_or_mistyped_dropped", "truncated", "wall_seconds_elapsed", "wall_seconds_remaining")


def _hidden_outcome_source(hidden_path):
    """the reviewer's reference edit with the hidden path bound: the SAME bytes for every run."""
    body = _R12_REFERENCE_BODY.replace("HIDDEN_PATH", repr(str(hidden_path)))

    def edit(src):
        old = "    return x * w + 1.0"
        assert old in src
        return src.replace(old, body)
    return edit


def _stats_under(obj, key):
    """Every value under *key*: numbers and the canonical non-finite tokens."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.append(v)
                    elif isinstance(v, str) and v in ("inf", "-inf", "nan"):
                        out.append(v)
                    elif isinstance(v, dict):
                        out.extend(x for x in v.values() if isinstance(x, (int, float, str)) and not isinstance(x, bool))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return out


def _touched_values(doc):
    """The raw values at the paths ``_normalise_independent`` replaces (same rules), in walk order."""
    out = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                p = path + (k,)
                if k in _CLOCK_KEYS or k in _KERNEL_KEYS or k in _FS_KEYS or k in _TOKEN_KEYS \
                        or (k in _BOUND_DIGEST_KEYS and isinstance(v, str)) \
                        or (path[:1] == ("deterministic",) and k in ("seconds", "elapsed_seconds")):
                    out.append(v)
                else:
                    walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + (str(i),))
    walk(doc, ())
    return out


def _canonical_form(text):
    """(parsed document, the serialiser that reproduces *text* exactly) -- every attempt-visible JSON file
    is written by Python's json from plain values, so its bytes are a function of the parsed document and
    one of these fixed formats; a file that matches none would be a channel outside the document."""
    doc = json.loads(text)
    forms = {"indent1-nl": lambda d: json.dumps(d, indent=1) + "\n", "indent1": lambda d: json.dumps(d, indent=1),
             "compact": lambda d: json.dumps(d)}
    for name, fn in forms.items():
        if fn(doc) == text:
            return doc, name, fn
    return doc, None, None


def _public_tree(out_dir, roots):
    """Every entry under *out_dir* reachable with the private root LOCKED: relative path, type, mode
    bits, link count; size for regular files; link target; the text of JSON files (compared as documents
    by the caller).  The private root itself is listed (type, mode); nothing below it is."""
    out = []
    for p in sorted(Path(out_dir).rglob("*")):
        if any(root in p.parents for root in roots):
            continue
        st = p.lstat()
        e = {"path": str(p.relative_to(out_dir)), "type": stat.S_IFMT(st.st_mode), "mode": stat.S_IMODE(st.st_mode),
             "nlink": st.st_nlink}
        if stat.S_ISREG(st.st_mode):
            e["size"] = st.st_size
            if p.suffix == ".json":
                e["json"] = p.read_text()
        if stat.S_ISLNK(st.st_mode):
            e["link"] = os.readlink(p)
        out.append(e)
    return out


def _dispatch_with_hidden_outcome(tmp_path, hidden_value):
    """One two-attempt schedule at the SAME paths as every other run (the reviewer's fixtures verbatim: the
    reference edit, the adapter edit, the candidate payload).  Returns every attempt-visible surface
    captured while the dispatch was resumable -- the raw texts of DISPATCH_STATE.json, the intermediate
    DISPATCH_RECEIPT.json and the next attempt's TRANSLATION_REQUEST.json, the returned result, the public
    tree, the dispatcher log -- and the terminal facts that prove which hidden outcome this run had."""
    (tmp_path / _HIDDEN_OUTCOME_FILE).write_text(repr(float(hidden_value)))
    case = tmp_path / "case"
    case.mkdir()
    r, consume, _ = _cuda_case(case, adapter_edit=_R12_ADAPTER_EDIT,
                               dsl_edit=lambda d: _candidate_body(_wrong_dsl(d), _phase_payload(_R12_PAYLOAD)),
                               source_edit=_hidden_outcome_source(tmp_path / _HIDDEN_OUTCOME_FILE))
    agent = _byte_distinct(consume)
    logs = []
    first = dispatch(r, DispatchPolicy(budget=Budget(max_attempts=2)), provider=PureSkillProvider(), log=logs.append)
    assert first.status == "awaiting_agent"
    agent(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    mid = resume(r.out_dir, PureSkillProvider(), log=logs.append)
    assert mid.status == "awaiting_agent"
    visible_logs = list(logs)          # the attempt-visible log: everything emitted up to this state
    roots = _private_roots(r.out_dir)
    assert len(roots) == 1
    docs = {"state": (r.out_dir / "DISPATCH_STATE.json").read_text(),
            "receipt": (r.out_dir / "DISPATCH_RECEIPT.json").read_text(),
            "request": (Path(mid.resume_token) / "TRANSLATION_REQUEST.json").read_text(),
            "result": json.dumps(mid.to_dict(), indent=1, default=str) + "\n"}
    tree = _public_tree(r.out_dir, roots)
    # the public verdict digest is recomputable from public data (justifies normalising it, as in round 12)
    import types as _types
    tr = [x for x in mid.to_dict()["failure_receipts"] if x["stage"] == "agentic_validate"][0]["validation"]["trusted_reference"]
    ph = tr["phases"]["compare"]
    meta = json.loads((_phase_dir(tr, "compare") / "meta.json").read_text())
    fake = _types.SimpleNamespace(env={k: v for k, v in meta["env_expected"]})
    argv = [re.sub(r"<private-evidence:[^>]+>", str(roots[0]), a) for a in ph["argv"]]
    assert tr["verdict_sha256"] == D._TrustedPhases.digest_for(fake, ph["verdict"], {"argv": argv})
    # second attempt: the reviewer's probe reads the public state and picks; then the terminal receipt
    agent(Path(mid.resume_token) / "TRANSLATION_REQUEST.json")
    last = resume(r.out_dir, PureSkillProvider(), log=logs.append)
    final = json.loads(Path(last.receipt_path).read_text())
    first_validation = [x for x in final["failure_receipts"] if x["stage"] == "agentic_validate" and x["attempt"] == 1][0]
    cmp_full = first_validation["validation"]["trusted_reference"]["phases"]["compare"]
    probe = sorted(r.out_dir.rglob("public-schema-observation.json"))
    return {"docs": docs, "tree": tree, "logs": visible_logs, "terminal_logs": logs[len(visible_logs):],
            "public_reason_codes": [x["reason_code"] for x in mid.to_dict()["failure_receipts"] if x["stage"] == "agentic_validate"],
            "verified": last.verified, "status": last.status,
            "error": _stats_under(first_validation["validation"]["trusted_reference"], "max_abs_error")[0],
            "terminal_reason": first_validation["reason"], "terminal_reason_code": first_validation["reason_code"],
            "terminal_vacuous": cmp_full["verdict"]["vacuous_outputs"],
            "terminal_dropped": cmp_full["report_schema"]["unknown_or_mistyped_dropped"],
            "terminal_pid": cmp_full["pid"], "terminal_elapsed": final["ledger"]["wall_seconds_elapsed"],
            "probe_observations": [json.loads(p.read_text())["observations"] for p in probe],
            "source_sha256": _sha(case / "reference.py")}


def _scrub_text(text, tokens):
    for t in sorted(tokens, key=len, reverse=True):
        text = text.replace(t, "<token>")
    text = _ROOT_TOKEN_RE.sub("<private-evidence:ID>", text)
    text = _ROOT_NAME_RE.sub(".private-evidence-ID", text)
    text = _RANDOM_NAME_RE.sub(lambda m: (m.group(1) or m.group(2)) + "-ID", text)
    return text


def _redact_messages(doc):
    """Numerals in free-text ``message`` fields -> <n> (the facade's own timings in a raw child report)."""
    if isinstance(doc, dict):
        return {k: (D._redact_numerals(v) if k == "message" and isinstance(v, str) else _redact_messages(v))
                for k, v in doc.items()}
    if isinstance(doc, list):
        return [_redact_messages(v) for v in doc]
    return doc


def _assert_identical_json_surface(name, texts, pre_comparison=False):
    """The raw bytes of one attempt-visible JSON file across runs: (1) each file IS the canonical
    serialisation of its parsed document (no whitespace/ordering/number-format channel outside the
    values); (2) the normalised documents, re-serialised in that same format, are byte-identical --
    key order included; (3) the raw size differences are exactly the width differences of the
    normalised values.  Hence the raw bytes differ only inside the justified fields.
    ``pre_comparison``: a raw child report in the public tree written by a phase that PRECEDES the
    reference and compare phases (verify, inputs, candidate) -- independent of the hidden comparison by
    construction (written before it, never rewritten); its free-text ``message`` fields carry the
    facade's own timings and are redacted before the comparison, so (3) is not applicable there and
    (1)+(2) carry the proof."""
    parsed = [_canonical_form(t) for t in texts]
    assert all(form is not None for _, form, _ in parsed), f"{name}: not a canonical serialisation of its document"
    assert len({form for _, form, _ in parsed}) == 1
    fn = parsed[0][2]
    docs = [(_redact_messages(doc) if pre_comparison else doc) for doc, _, _ in parsed]
    norms, touched = zip(*(_normalise_independent(doc) for doc in docs))
    diffs = sorted({p for n in norms[1:] for p in _differing_paths(norms[0], n)})
    assert not diffs, f"{name}: attempt-visible fields vary with the hidden comparison outcome:\n" + "\n".join(diffs)
    serial = [fn(n) for n in norms]
    assert len(set(serial)) == 1, f"{name}: normalised documents differ in bytes (key order / number format)"
    assert len({tuple(sorted(set(t))) for t in touched}) == 1
    sizes = [len(t.encode()) for t in texts]
    if not pre_comparison:
        widths = [sum(len(json.dumps(v)) for v in _touched_values(doc)) for doc in docs]
        for i in range(1, len(texts)):
            assert sizes[i] - sizes[0] == widths[i] - widths[0], \
                f"{name}: raw size difference {sizes[i] - sizes[0]} not accounted for by the normalised fields ({widths[i] - widths[0]})"
    return sizes, sorted(set(touched[0]))


def test_whole_public_surface_does_not_depend_on_the_hidden_comparison(tmp_path):
    """PROPERTY (the review NUMERIC-FEEDBACK-30/D and PUBLIC-PROPERTY-SCOPE-31 rulings): five dispatches that
    differ ONLY in the hidden first element of the trusted reference -- the reviewer's two reached-comparison
    fixtures verbatim (+1e308: error 1.0; -1e308: error inf by overflow) plus a large finite value, a nan
    and the all-zero (vacuous) reference -- with byte-identical request, source, adapter edit, candidate
    (the reviewer's probe payload) and paths.

    Surfaces compared, while the dispatch is resumable (awaiting the second attempt):
      * RAW BYTES of DISPATCH_STATE.json, the intermediate DISPATCH_RECEIPT.json, the next attempt's
        TRANSLATION_REQUEST.json and every other public JSON file: each is the canonical serialisation of
        its document; the documents are byte-identical (key order included) after normalising ONLY the
        fields listed in ``_normalise_independent`` (clock readings of the dispatch start and of files
        written before the comparison, kernel ids, inode numbers, the two pre-comparison-bound digests,
        random tokens and names); the raw size differences equal the width differences of those values;
      * the returned result (serialised the same way);
      * the public tree listing (paths, types, modes, link counts; sizes of every non-JSON file; the
        private root listed, its contents not; the raw child reports of the pre-comparison phases under
        trusted_reference/ compared as documents with their free-text timings redacted);
      * the dispatcher log lines emitted up to that state (random names/tokens scrubbed; no clock
        reading is in them); the terminal lines differ with the terminal outcome and are not compared.
    Excluded, with reason: file timestamps (mtime/ctime/atime: clock readings; the test does not read
    them); the justified fields above.  Not claimed: identity across different hosts or paths, or a
    timing side channel measured with the attempt's own clock.
    Also asserted: ONE public class (trusted_reference_mismatch) in all five runs including the overflow,
    nan and vacuous references; constant-shape verdict and report_schema; no withheld number in any
    attempt-visible document; wall-clock telemetry withheld until the terminal receipt; the terminal
    receipt restores statistics (canonical tokens), vacuous_outputs, parse counters (0: the canonical
    tokens parsed whole), kernel ids and the elapsed wall; the reviewer's probe, reading identical public
    records, makes the same choice in every run and is right in exactly the one run whose hidden value
    equals its default guess."""
    runs = []
    for i, (name, hidden, _) in enumerate(_HIDDEN_OUTCOMES):
        runs.append(_dispatch_with_hidden_outcome(tmp_path, hidden))
        os.rename(tmp_path / "case", tmp_path / f"run-{i}-{name}")
    names = [n for n, _, _ in _HIDDEN_OUTCOMES]
    # the five runs really had five hidden outcomes, visible only in the terminal receipts
    errors = [r["error"] for r in runs]
    assert errors == [1.0, "inf", 5e307, "nan", 1e308], errors
    assert [r["terminal_vacuous"] for r in runs] == [[], [], [], [], ["output"]]
    assert all(r["terminal_dropped"] == 0 for r in runs)                       # canonical tokens parsed whole
    assert all(isinstance(r["terminal_pid"], int) for r in runs)              # restored in the terminal receipt
    assert all(isinstance(r["terminal_elapsed"], float) and r["terminal_elapsed"] > 0 for r in runs)
    assert all(r["terminal_reason_code"] == "trusted_reference_mismatch" for r in runs)
    assert all("error statistics are in the final DISPATCH_RECEIPT.json" in r["terminal_reason"] for r in runs)
    assert len({r["source_sha256"] for r in runs}) == 1
    # ONE public class for every statistic class; the reviewer's probe saw identical public records
    assert all(r["public_reason_codes"] == ["trusted_reference_mismatch"] for r in runs), [r["public_reason_codes"] for r in runs]
    assert len({json.dumps(r["probe_observations"], sort_keys=True) for r in runs}) == 1
    assert runs[0]["probe_observations"][-1][-1]["schema"]["unknown_or_mistyped_dropped"] is None
    # the probe's second attempt is a blind guess (1e308 when the counter is not 2): right exactly once
    assert [r["verified"] for r in runs] == [True, False, False, False, False], [r["status"] for r in runs]
    # attempt-visible documents: raw bytes, key order, size accounting
    report = {}
    for doc in ("state", "receipt", "request", "result"):
        report[doc] = _assert_identical_json_surface(doc, [r["docs"][doc] for r in runs])
    # no withheld number anywhere in a validation record (the deterministic stage's pre-agentic
    # `seconds` at the top level is public by design and normalised as such)
    for doc in ("state", "receipt"):
        for p in (json.loads(r["docs"][doc]) for r in runs):
            records = [x.get("validation") for x in p.get("failure_receipts", []) + p.get("attempts", [])] + [p.get("validation")]
            assert any(isinstance(v, dict) for v in records)
            for v in records:
                if isinstance(v, dict):
                    for key in _WITHHELD_NUMBER_KEYS:
                        assert _numbers_under(v, key) == [], (doc, key)
    state = json.loads(runs[0]["docs"]["state"])
    assert state["wall_seconds_elapsed"] is None
    assert json.loads(runs[0]["docs"]["receipt"])["ledger"]["wall_seconds_elapsed"] is None
    assert json.loads(runs[0]["docs"]["request"])["caps"]["wall_seconds_remaining"] is None
    val = [x for x in state["failure_receipts"] if x["stage"] == "agentic_validate"][0]["validation"]
    cmp_pub = val["trusted_reference"]["phases"]["compare"]
    assert list(cmp_pub["verdict"]) == list(D._PUBLIC_VERDICT_KEYS) and cmp_pub["verdict"]["vacuous_outputs"] is None
    assert cmp_pub["report_schema"]["unknown_or_mistyped_dropped"] is None and cmp_pub["pid"] is None
    # the public tree: same entries, types, modes, link counts; same sizes for every non-JSON file; JSON
    # files compared as documents (canonical, normalised, size-accounted) like the records above
    tokens = set()
    for r in runs:
        for doc in r["docs"].values():
            def collect(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if k in _TOKEN_KEYS and isinstance(v, str) and v:
                            tokens.add(v)
                        collect(v)
                elif isinstance(x, list):
                    for v in x:
                        collect(v)
            collect(json.loads(doc))
    trees = []
    for r in runs:
        t = {}
        for e in r["tree"]:
            key = _scrub_text(e["path"], tokens)
            t[key] = {k: (_scrub_text(v, tokens) if isinstance(v, str) else v) for k, v in e.items() if k != "json"}
            if "json" in e:
                t[key]["json"] = e["json"]
        trees.append(t)
    assert all(set(t) == set(trees[0]) for t in trees), [sorted(set(t) ^ set(trees[0])) for t in trees]
    assert len(trees[0]) >= 30
    # the reference and compare phases live under the private root: nothing of them is in the public tree;
    # every public file under trusted_reference/ belongs to a phase that precedes the comparison
    assert not any("phase_reference" in k or "phase_compare" in k for k in trees[0])
    for key in sorted(trees[0]):
        entries = [t[key] for t in trees]
        meta = [{k: v for k, v in e.items() if k not in ("json", "size")} for e in entries]
        assert all(m == meta[0] for m in meta), (key, meta)
        if "json" in entries[0]:
            report[f"tree:{key}"] = _assert_identical_json_surface(key, [e["json"] for e in entries],
                                                                   pre_comparison="/trusted_reference/" in key)
        elif "size" in entries[0]:
            assert len({e["size"] for e in entries}) == 1, (key, [e["size"] for e in entries])
    # the dispatcher log: identical lines once random names/tokens are scrubbed
    logs = [[_scrub_text(l, tokens) for l in r["logs"]] for r in runs]
    assert all(l == logs[0] for l in logs), [sorted(set(map(tuple, logs)))][:2]
    assert logs[0] and not any(_UTC_RE.search(l) for l in logs[0])
    assert any("trusted_reference_mismatch" in l for l in logs[0])
    (tmp_path / "whole-surface-report.json").write_text(json.dumps(
        {"cases": names, "errors": [str(e) for e in errors], "verified": [r["verified"] for r in runs],
         "raw_sizes_and_normalised_fields": {k: {"sizes": v[0], "normalised": v[1]} for k, v in report.items()},
         "tree_entries": sorted(trees[0]), "log_lines": logs[0]}, indent=1) + "\n")


def test_facade_execution_error_names_its_refusal(tmp_path, monkeypatch):
    """target pipeline 77147099 (90840700, dispatcher-c): the facade reported ``IR execution FAILED after
    0.0s: PermissionError: ...`` and the verdict was the ONE class without a reason_code -- the attempt-visible
    receipt carried ``reason_code: null``.  The event is now named; the kind is still decided by the exception
    TYPE only (PermissionError -> unsupported/candidate_error, MemoryError -> infrastructure)."""
    dsl = tmp_path / "cand.py"
    dsl.write_text("x = 1\n")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("y = 2\n")
    src = tmp_path / "ref.py"
    src.write_text("z = 3\n")
    cand = D.Candidate(dsl_path=dsl, model_path=adapter, dsl_sha256=D._sha256(dsl), source_sha256=D._sha256(src),
                       adapter_sha256=D._sha256(adapter))
    req = D.DispatchRequest(src, tmp_path / "out", source_type="cuda", tolerance={"rtol": 1e-4, "atol": 1e-5})
    out = tmp_path / "out" / "agentic" / "attempt_01"
    out.mkdir(parents=True)
    for exc_name, kind, status in (("PermissionError", FailureKind.UNSUPPORTED, "candidate_error"),
                                   ("MemoryError", FailureKind.INFRASTRUCTURE, "infrastructure")):
        msg = f"IR execution FAILED after 0.0s: {exc_name}: [Errno 1] Operation not permitted"
        monkeypatch.setattr(D, "_run_verify_phase", lambda *a, _m=msg, **k: (
            {"status": "error", "message": _m, "max_abs_error": None, "vacuous_outputs": [], "execution": None}, None))
        v = D.default_validator(cand, req, DispatchPolicy(), out / "receipt.json", log=lambda *_: None)
        assert (v.status, v.kind, v.verify_status) == (status, kind, "error")
        assert v.reason_code == "ir_execution_failed" and v.message == msg


def test_trusted_child_is_inert_when_imported_as_a_multiprocessing_main(tmp_path, monkeypatch):
    """target pipeline 77147099, layer 2: a spawn/forkserver worker of the facade's isolated verification
    re-imports its parent's main module as ``__mp_main__``.  The trusted child used to run its whole phase
    program at module level and end in ``done()`` -> ``sys.exit`` (the worker exited 0 without a result).
    Imported under any other name it now defines its helpers, records the fact, and neither reads argv, nor
    writes a report, nor exits."""
    import runpy
    meta = tmp_path / "meta.json"
    report = tmp_path / "report.json"
    meta.write_text(json.dumps({"report_path": str(report), "trusted_roots": [], "sys_path": []}))
    monkeypatch.setattr(sys, "argv", [str(D._TRUSTED_CHILD_PATH), "verify", str(meta)])
    ns = runpy.run_path(str(D._TRUSTED_CHILD_PATH), run_name="__mp_main__")      # must not raise SystemExit
    assert ns["IMPORTED_NOT_EXECUTED"] == "__mp_main__"
    assert "PHASE" not in ns and "META" not in ns and "REPORT" not in ns and "ROOTS" not in ns
    assert not report.exists() and not (tmp_path / "report.json.tmp").exists()
    assert callable(ns["done"]) and callable(ns["set_trusted_path"]) and ns["GUARD"] is not None
    # direct execution is untouched: the same argv refuses the missing checkout root through done() -> exit 3
    import subprocess
    proc = subprocess.run([sys.executable, "-I", "-S", str(D._TRUSTED_CHILD_PATH), "verify", str(meta)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 3 and report.exists()
    rep = json.loads(report.read_text())
    assert rep["phase"] == "verify" and rep["ok"] is False and "checkout_root" in rep["error"]


def test_isolation_unavailable_is_an_environment_failure(tmp_path, monkeypatch):
    """target pipeline 77147099 round 2 (the review isolation-fallback-1): when the facade cannot start its
    isolating child it refuses with ``IsolationUnavailableError`` instead of running the program in its
    caller.  The dispatcher carries that refusal as an ENVIRONMENT failure: infrastructure, never
    ``candidate_error``, unscored, the attempt retained, with its own constant reason code.  (The facade
    re-raises the typed refusal; this covers a facade result that folded it into an execution failure.)"""
    dsl = tmp_path / "cand.py"
    dsl.write_text("x = 1\n")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("y = 2\n")
    src = tmp_path / "ref.py"
    src.write_text("z = 3\n")
    cand = D.Candidate(dsl_path=dsl, model_path=adapter, dsl_sha256=D._sha256(dsl), source_sha256=D._sha256(src),
                       adapter_sha256=D._sha256(adapter))
    req = D.DispatchRequest(src, tmp_path / "out", source_type="cuda", tolerance={"rtol": 1e-4, "atol": 1e-5})
    out = tmp_path / "out" / "agentic" / "attempt_01"
    out.mkdir(parents=True)
    msg = ("IR execution FAILED after 0.0s: IsolationUnavailableError: isolation_unavailable: no isolating "
           "child process could be started ([Errno 1] Operation not permitted)")
    monkeypatch.setattr(D, "_run_verify_phase", lambda *a, **k: (
        {"status": "error", "message": msg, "max_abs_error": None, "vacuous_outputs": [], "execution": None}, None))
    v = D.default_validator(cand, req, DispatchPolicy(), out / "receipt.json", log=lambda *_: None)
    assert (v.status, v.kind, v.verify_status, v.reason_code) == ("infrastructure", FailureKind.INFRASTRUCTURE,
                                                                   "error", "isolation_unavailable")
    assert v.message == msg
    rec = D._receipt(v.kind, "agentic_validate", v.message, reason_code=v.reason_code)
    assert rec["scored"] is False and rec["retained_for_retry"] is True
    assert "IsolationUnavailableError" in D._CANDIDATE_INFRA_TYPES


def test_isolation_refusal_reported_by_the_phase_child_is_typed(tmp_path):
    """The verify child reports the facade's refusal with its exception TYPE (no message matching): rc 12 with
    exception.type IsolationUnavailableError, or the candidate child's rc 18, is isolation_unavailable
    (infrastructure).  (minimal split of an earlier change: an earlier revision's rc-17 fresh-attempt assertion is not carried.)"""
    class _Phases:
        private_root = tmp_path
        ingested = {}

        def run(self, phase, meta, results=None):
            return 12, {"ok": False, "returncode": 12, "error": "verify raised IsolationUnavailableError: isolation_unavailable: x",
                        "exception": {"type": "IsolationUnavailableError", "mro": ["IsolationUnavailableError", "OSError", "Exception"]}}

    dsl = tmp_path / "cand.py"
    dsl.write_text("x = 1\n")
    src = tmp_path / "ref.py"
    src.write_text("z = 3\n")
    cand = D.Candidate(dsl_path=dsl, model_path=dsl, dsl_sha256=D._sha256(dsl), source_sha256=D._sha256(src))
    req = D.DispatchRequest(src, tmp_path / "out", source_type="cuda", tolerance={"rtol": 1e-4, "atol": 1e-5})
    summary, fail = D._run_verify_phase(cand, req, DispatchPolicy(), {"rtol": 1e-4, "atol": 1e-5}, tmp_path / "w",
                                        tmp_path / "w" / "receipt.staged.json", log=lambda *_: None, phases=_Phases())
    assert summary is None and fail[:3] == (FailureKind.INFRASTRUCTURE, "infrastructure", "isolation_unavailable")
    assert D._classify_exception_names("IsolationUnavailableError", ["OSError"], 0) == FailureKind.INFRASTRUCTURE
    assert D._classify_exception_names(None, [], D._RC_ISOLATION_UNAVAILABLE) == FailureKind.INFRASTRUCTURE


def test_verify_phase_helper_slow_to_exit_is_reaped_before_the_leader_exits(tmp_path):
    """Target dispatcher-c flake (pipelines 77147099 / 77789291 / 77857565: ``candidate_descendants_alive`` instead of
    ``candidate_result_missing`` in the FIFO [report] case, passing on retry).  The facade's isolated verification inside the
    trusted VERIFY child starts helper processes in the phase's session -- the multiprocessing forkserver and its resource
    tracker -- which exit asynchronously after their owner; the session boundary samples /proc the instant the leader exits and
    refuses ANY survivor, so a helper merely slow to notice its parent's exit turned a healthy validation into an infrastructure
    refusal on a loaded runner.  Deterministic reproduction of the mechanism: the candidate program, executed by the facade in
    the isolated worker of the verify phase, SIGSTOPs the resource tracker of its session -- a helper that can no longer exit on
    its own.  The trusted child now reaps its helpers (verify.shutdown_isolation, bounded, SIGKILL for a stuck one) BEFORE it
    exits: the boundary is clean, the stopped helper is gone, and the dispatch verifies.  Without the repair this dispatch ends
    ``candidate_descendants_alive`` (the recorded flake)."""
    marker = tmp_path / "stopped_helper.pid"
    payload = ("import os as _os, signal as _signal\n"
               "_sid = _os.getsid(0)\n"
               "for _d in _os.listdir('/proc'):\n"
               "    if not _d.isdigit(): continue\n"
               "    try:\n"
               "        _st = open('/proc/%s/stat' % _d).read(); _rest = _st[_st.rindex(')') + 2:].split()\n"
               "        if int(_rest[3]) == _sid and b'resource_tracker' in open('/proc/%s/cmdline' % _d, 'rb').read():\n"
               "            _os.kill(int(_d), _signal.SIGSTOP)\n"
               f"            open({str(marker)!r}, 'a').write(_d + '\\n')\n"
               "    except Exception: pass\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    policy = DispatchPolicy(budget=Budget(max_attempts=1))
    policy.verify_timeout = 60
    res = dispatch(r, policy, provider=PureSkillProvider(agent=consume))
    assert marker.exists(), "the isolated worker of the verify phase did not find a resource tracker in its session"
    stopped = [int(x) for x in marker.read_text().split()]
    assert stopped, "no helper was stopped: the reproduction did not arm"
    assert res.verified, [x.get("reason") for x in res.failure_receipts]
    tr = res.validation["trusted_reference"]
    sb = tr["phases"]["verify"]["session_boundary"]      # the verify phase's boundary record (the generated phases' live under session_boundaries)
    assert sb["clean"] is True and sb["found_after_exit"] == [] and sb["alive_after_kill"] == []
    for pid in stopped:          # the stopped helper was reaped by its owner before the leader exited: it does not exist any more
        assert not Path(f"/proc/{pid}").exists() or open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
