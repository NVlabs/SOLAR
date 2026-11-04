# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``verify._run_isolated`` when the isolating child cannot be STARTED (target pipeline 77147099; the minimal split
of an earlier change).

Inside the dispatcher's trusted verify child every validation carrying a ``verify_timeout`` became a scored
``candidate_error`` when the facade's isolating child failed (target pipeline 77147099).  A start refusal is a fact
about THIS process' environment, never a property of the program under test.  Contract (the review
isolation-fallback-1 / CONTEXT-FALLBACK-3): when the multiprocessing context cannot be obtained or the
isolating child cannot be started the entry is NOT executed -- ``IsolationUnavailableError`` (an OSError) is
raised; the memory budget, the termination guarantee and the caller's state are kept.  There is no thread
fallback and no in-process mode.

Carved from an earlier change an earlier revision (this module); the changes against an earlier revision are marked "minimal split" in place:
(1) a forkserver start refusal IS the typed refusal (an earlier revision accepted a spawn child there; this branch carries no
second start method); (2) the caller-boundary mode test is not carried (the mode does not exist here); (3) a
context-lookup failure is the same typed refusal (an earlier revision, the review CONTEXT-FALLBACK-3); (4) the absence of an
in-process mode is asserted on the public signatures.

Round 2 (the review isolation-type-1): EVERY pre-execution start refusal is the typed refusal, whatever class the
start raised -- the daemon-worker ``AssertionError`` and the spawn bootstrap ``RuntimeError`` included -- tested
with a controlled context and with a REAL daemon worker; an error raised inside a child that started keeps its own
class.  Round 2 (the review dropped-control-2): the generic interpreter-start-refusal control of an earlier revision is
restored verbatim at the end of this module.
"""
import errno
import json
import multiprocessing
import os
import time
import warnings

import numpy as np
import pytest

from solar.einsum_lib import verify
from solar.frontend import dispatcher as D
import test_isolated_verification_openmp as I


class _Refusing:
    """A multiprocessing-like context whose Process.start() the host refuses."""

    _name = "forkserver"

    class Process:
        exitcode = None

        def __init__(self, *a, **k):
            self.pid = None

        def start(self):
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    @staticmethod
    def Pipe(duplex=False):
        return multiprocessing.Pipe(duplex)


class _RefusingSpawn(_Refusing):
    _name = "spawn"


def _child_marker(path):
    from pathlib import Path
    Path(path).write_text(json.dumps({"pid": os.getpid()}))
    return os.getpid()


def _child_raises():
    raise PermissionError(errno.EPERM, "raised INSIDE a child that started")


def _child_sleeps_then_marks(path, seconds):
    time.sleep(seconds)
    from pathlib import Path
    Path(path).write_text(str(os.getpid()))


def _child_allocates(path, nbytes):
    buf = bytearray(nbytes)
    from pathlib import Path
    Path(path).write_text(str(len(buf)))
    return len(buf)


def _run(fn, timeout, *args, mem=None, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = verify._run_isolated(fn, timeout, mem, args=args, **kw)
    return result, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.fixture
def forkserver_refused(monkeypatch):
    monkeypatch.setattr(verify, "_isolation_context", lambda: _Refusing())


@pytest.fixture
def nothing_starts(monkeypatch):
    monkeypatch.setattr(verify, "_isolation_context", lambda: _Refusing())
    # a second start method, where one exists, is refused as well (none on this branch: raising=False)
    monkeypatch.setattr(verify, "_spawn_context", lambda: _RefusingSpawn(), raising=False)


def test_start_refusal_is_a_typed_refusal_without_execution(forkserver_refused, tmp_path):
    """minimal split: an earlier revision accepted "a spawn child or the typed refusal" here; this branch has no second
    start method, so a refused start IS the typed refusal and the entry never runs anywhere."""
    marker = tmp_path / "ran.json"
    with pytest.raises(verify.IsolationUnavailableError, match="isolation_unavailable") as info:
        _run(_child_marker, 60, str(marker))
    assert isinstance(info.value, OSError) and "Operation not permitted" in str(info.value)
    assert not marker.exists()


def test_no_startable_fresh_process_is_a_typed_refusal_without_execution(nothing_starts, tmp_path):
    marker = tmp_path / "ran.json"
    with pytest.raises(verify.IsolationUnavailableError, match="isolation_unavailable") as info:
        _run(_child_marker, 60, str(marker))
    assert isinstance(info.value, OSError) and "Operation not permitted" in str(info.value)
    assert not marker.exists()


def test_refusal_keeps_the_memory_budget_contract(nothing_starts, tmp_path):
    """An explicit budget the refused child would have enforced is never bypassed by running here."""
    marker = tmp_path / "alloc.txt"
    with pytest.raises(verify.IsolationUnavailableError):
        _run(_child_allocates, 5, str(marker), 8 * 1024 * 1024, mem=1024)
    assert not marker.exists()


def test_refusal_does_not_run_anything_that_a_timeout_would_have_to_stop(nothing_starts, tmp_path):
    marker = tmp_path / "late.txt"
    t0 = time.monotonic()
    with pytest.raises(verify.IsolationUnavailableError):
        _run(_child_sleeps_then_marks, 0.05, str(marker), 0.3)
    assert time.monotonic() - t0 < 1.0
    time.sleep(0.45)
    assert not marker.exists()


def test_refusal_leaves_the_callers_random_state_untouched(nothing_starts):
    before = np.random.get_state()
    try:
        np.random.seed(1039)
        state = np.random.get_state()
        with pytest.raises(verify.IsolationUnavailableError):
            _run(np.random.random, 5)
        after = np.random.get_state()
        assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    finally:
        np.random.set_state(before)


def test_spawn_only_refusal_is_the_same_typed_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "_isolation_context", lambda: _RefusingSpawn())
    marker = tmp_path / "ran.json"
    with pytest.raises(verify.IsolationUnavailableError):
        result, texts = _run(_child_marker, 60, str(marker))
    assert not marker.exists()


def test_context_lookup_failure_is_the_same_typed_refusal(monkeypatch, tmp_path):
    """minimal split (an earlier revision, the review context-fallback-3): an OSError while obtaining the multiprocessing
    context is the typed refusal, nothing runs; the deliberate invalid-start-method ``ValueError`` is unchanged."""
    def unavailable():
        raise OSError(errno.EPERM, "controlled context lookup failure")

    monkeypatch.setattr(verify, "_isolation_context", unavailable)
    marker = tmp_path / "ran.json"
    with pytest.raises(verify.IsolationUnavailableError, match="no multiprocessing context") as info:
        _run(_child_marker, 60, str(marker))
    assert isinstance(info.value.__cause__, OSError) and not marker.exists()

    def invalid():
        raise ValueError("isolated verification cannot use start method 'fork'")

    monkeypatch.setattr(verify, "_isolation_context", invalid)
    with pytest.raises(ValueError, match="start method"):
        _run(_child_marker, 60, str(marker))
    assert not marker.exists()


def test_an_error_inside_a_started_child_is_still_the_childs_error():
    """Control: a child that DID start reports its own exception, unchanged."""
    try:
        with pytest.raises(PermissionError, match="INSIDE a child"):
            verify._run_isolated(_child_raises, 60, None)
    except verify.IsolationUnavailableError as exc:
        pytest.skip(f"this host can start no isolating child: {exc}")


def test_no_in_process_mode_exists():
    """minimal split: the fresh-process contract has no opt-out -- neither ``_run_isolated`` nor
    ``execute_translated`` takes an isolation mode, and the thread path is not reachable from a refused start."""
    import inspect
    assert "isolation" not in inspect.signature(verify._run_isolated).parameters
    assert "isolation" not in inspect.signature(verify.execute_translated).parameters
    assert not hasattr(verify, "ISOLATION_MODES") and not hasattr(verify, "_mint_caller_boundary")


# ---------------------------------------------------------------------------------------------------------------
# round 2 -- the review isolation-type-1: a start refusal of ANY class is the typed refusal
# ---------------------------------------------------------------------------------------------------------------

_PLAIN_PROGRAM = "import numpy as np\ndef run_model(inputs):\n    return {'x': np.array([1.0])}\n"


def _refusing_with(exc):
    """A multiprocessing-like context whose Process.start() raises *exc* (whatever its class)."""
    class _Ctx(_Refusing):
        class Process:
            exitcode = None

            def __init__(self, *a, **k):
                self.pid = None

            def start(self):
                raise exc
    return _Ctx()


def _child_asserts():
    raise AssertionError("raised INSIDE a child that started")


@pytest.mark.parametrize("exc", [
    AssertionError("daemonic processes are not allowed to have children"),
    RuntimeError("An attempt has been made to start a new process before the current process has finished its "
                 "bootstrapping phase."),
], ids=["daemon-worker-AssertionError", "spawn-bootstrap-RuntimeError"])
def test_non_oserror_start_refusal_is_the_same_typed_refusal(monkeypatch, tmp_path, exc):
    """The startup boundary types what the start raised, not only an OSError: the entry never runs, the cause and
    its class name are kept, the memory-budget contract holds."""
    monkeypatch.setattr(verify, "_isolation_context", lambda: _refusing_with(exc))
    marker = tmp_path / "ran.json"
    with pytest.raises(verify.IsolationUnavailableError, match="isolation_unavailable") as info:
        _run(_child_marker, 60, str(marker), mem=1024)
    assert info.value.__cause__ is exc and type(exc).__name__ in str(info.value) and str(exc) in str(info.value)
    assert not marker.exists()


def _daemon_probe(out):
    """Runs INSIDE a daemon worker: the real ``Process.start()`` refusal of a daemon (AssertionError) at the
    `_run_isolated` boundary, through ``execute_translated`` in both modes; records what each raised."""
    from pathlib import Path
    marker = Path(out).with_suffix(".marker")
    records = []

    def probe(label, fn):
        try:
            fn()
            records.append({"what": label, "executed": True})
        except BaseException as exc:  # noqa: BLE001 - the class is the observation
            records.append({"what": label, "type": type(exc).__name__,
                            "typed": isinstance(exc, verify.IsolationUnavailableError),
                            "cause": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                            "message": str(exc)[:300]})

    probe("_run_isolated", lambda: verify._run_isolated(_child_marker, 30, None, args=(str(marker),)))
    probe("execute_translated fast=True", lambda: verify.execute_translated(_PLAIN_PROGRAM, {}, fast=True, timeout=5))
    probe("execute_translated fast=False", lambda: verify.execute_translated(_PLAIN_PROGRAM, {}, fast=False, timeout=5))
    Path(out).write_text(json.dumps({"daemon": multiprocessing.current_process().daemon, "marker_exists": marker.exists(),
                                     "records": records}))


def test_real_daemon_start_refusal_is_typed(tmp_path):
    """A REAL daemon worker (the standard multiprocessing refusal: "daemonic processes are not allowed to have
    children") calling the isolating entry points: `_run_isolated` directly and `execute_translated` with the fast
    and the interpreter path.  Every call is the typed refusal with the AssertionError as its cause; nothing ran;
    the dispatcher's type-only classification of the facade-formatted message is infrastructure, unscored,
    retained."""
    out = tmp_path / "daemon.json"
    p = multiprocessing.get_context("spawn").Process(target=_daemon_probe, args=(str(out),), daemon=True)
    p.start()
    p.join(90)
    if p.is_alive():
        p.kill()
        p.join()
        pytest.fail("the daemon probe did not finish in 90 s")
    assert p.exitcode == 0 and out.exists()
    rec = json.loads(out.read_text())
    assert rec["daemon"] is True and rec["marker_exists"] is False
    assert len(rec["records"]) == 3, rec
    for r in rec["records"]:
        assert r.get("typed") is True and r["type"] == "IsolationUnavailableError" and r["cause"] == "AssertionError", r
        assert "daemonic processes are not allowed to have children" in r["message"]
        msg = f"IR execution FAILED after 0.0s: {r['type']}: {r['message']}"
        assert D.classify_candidate_error_message(msg) == D.FailureKind.INFRASTRUCTURE
        receipt = D._receipt(D.FailureKind.INFRASTRUCTURE, "agentic_validate", msg, reason_code="isolation_unavailable")
        assert receipt["scored"] is False and receipt["retained_for_retry"] is True


def test_a_non_oserror_raised_inside_a_started_child_keeps_its_class():
    """Control for the boundary: an exception of the SAME class as a start refusal (AssertionError) raised by a
    child that did start arrives as that class -- it is the program's error, never isolation_unavailable."""
    try:
        with pytest.raises(AssertionError, match="INSIDE a child"):
            verify._run_isolated(_child_asserts, 60, None)
    except verify.IsolationUnavailableError as exc:
        pytest.skip(f"this host can start no isolating child: {exc}")


# ---------------------------------------------------------------------------------------------------------------
# round 2 -- the review dropped-control-2: the generic control of an earlier change an earlier revision, restored VERBATIM
# (tests/test_isolated_verification_retry_state.py::test_interpreter_start_refusal_keeps_environment_classification;
# it needs neither the caller-boundary capability nor a second start method)
# ---------------------------------------------------------------------------------------------------------------

def test_interpreter_start_refusal_keeps_environment_classification(monkeypatch, tmp_path):
    """the reviewer's control: the first (fast) worker starts and declines; the interpreter worker's start is refused.
    The refusal keeps its type through the interpreter wrapper and the facade, and default_validator classifies
    it as an environment failure."""
    monkeypatch.setattr(verify, "_isolation_context", lambda: multiprocessing.get_context("spawn"))
    real = verify._start_isolated; calls = []

    def start(ctx, *args):
        calls.append(ctx.get_start_method())
        if len(calls) > 1:
            raise PermissionError(1, "second isolated worker start unavailable (control)")
        return real(ctx, *args)

    monkeypatch.setattr(verify, "_start_isolated", start)
    err = None
    try:
        verify.execute_translated(I._RNG_PROGRAM_PY, {}, all_inputs={}, fast=True, timeout=30)
    except Exception as e:
        err = e
    assert err is not None and len(calls) > 1
    assert isinstance(err, verify.IsolationUnavailableError), type(err)              # the type survives the wrapper
    msg = f"IR execution FAILED after 0.0s: {type(err).__name__}: {err}"
    dsl = tmp_path / "cand.py"; dsl.write_text("x=1\n")
    src = tmp_path / "ref.py"; src.write_text("x=2\n")
    cand = D.Candidate(dsl_path=dsl, model_path=dsl, dsl_sha256=D._sha256(dsl), source_sha256=D._sha256(src))
    req = D.DispatchRequest(src, tmp_path / "out", source_type="cuda", tolerance={"rtol": 1e-4, "atol": 1e-5})
    monkeypatch.setattr(D, "_run_verify_phase", lambda *a, **k: (
        {"status": "error", "message": msg, "max_abs_error": None, "vacuous_outputs": [], "execution": None}, None))
    out = tmp_path / "out/agentic/attempt_01"; out.mkdir(parents=True)
    val = D.default_validator(cand, req, D.DispatchPolicy(), out / "receipt.json", log=lambda *_: None)
    assert (val.kind, val.status, val.reason_code) == (D.FailureKind.INFRASTRUCTURE, "infrastructure", "isolation_unavailable")


# ---------------------------------------------------------------------------------------------------------------
# helper lifetime: shutdown_isolation() reaps the forkserver and the resource tracker before their owner exits
# (target dispatcher-c flake candidate_descendants_alive on a loaded runner)
# ---------------------------------------------------------------------------------------------------------------

def _helper_pids():
    import multiprocessing.forkserver as fs, multiprocessing.resource_tracker as rt
    return fs._forkserver._forkserver_pid, rt._resource_tracker._pid


def _gone(pid):
    if not os.path.exists(f"/proc/{pid}"):
        return True
    return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"


def test_shutdown_isolation_reaps_the_helpers_and_a_later_run_restarts_them(tmp_path):
    if verify._isolation_context().get_start_method() != "forkserver":
        pytest.skip("the forkserver helpers exist only under the forkserver start method")
    marker = tmp_path / "a.json"
    assert verify._run_isolated(_child_marker, 60, None, args=(str(marker),)) != os.getpid()
    fs_pid, rt_pid = _helper_pids()
    assert fs_pid and rt_pid
    rec = verify.shutdown_isolation()
    assert rec["forkserver"]["reaped"] is True and rec["resource_tracker"]["reaped"] is True
    assert rec["forkserver"]["escalated"] == [] and rec["resource_tracker"]["escalated"] == []    # the stop protocol sufficed
    assert _helper_pids() == (None, None) and _gone(fs_pid) and _gone(rt_pid)
    assert verify.shutdown_isolation() == {"forkserver": None, "resource_tracker": None}          # idempotent
    marker2 = tmp_path / "b.json"
    assert verify._run_isolated(_child_marker, 60, None, args=(str(marker2),)) != os.getpid()      # helpers restart on demand
    assert all(_helper_pids())
    verify.shutdown_isolation()


def test_shutdown_isolation_ends_a_helper_that_cannot_exit(tmp_path):
    """A stopped helper (the deterministic stand-in for one that is slow to notice its owner's exit) cannot honour the stop
    protocol: shutdown escalates to SIGKILL within its bound and still reaps it, so the owner exits with no survivor."""
    import signal
    if verify._isolation_context().get_start_method() != "forkserver":
        pytest.skip("the forkserver helpers exist only under the forkserver start method")
    assert verify._run_isolated(_child_marker, 60, None, args=(str(tmp_path / "c.json"),)) != os.getpid()
    fs_pid, rt_pid = _helper_pids()
    os.kill(rt_pid, signal.SIGSTOP)
    t0 = time.monotonic()
    rec = verify.shutdown_isolation(timeout_sec=0.5)
    assert time.monotonic() - t0 < 10.0
    assert rec["resource_tracker"]["reaped"] is True and "SIGKILL" in rec["resource_tracker"]["escalated"]
    assert rec["forkserver"]["reaped"] is True
    assert _gone(rt_pid) and _gone(fs_pid) and _helper_pids() == (None, None)
