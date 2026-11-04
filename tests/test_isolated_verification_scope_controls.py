# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""the reviewer's round-2 independent contract controls for an earlier change (an earlier review.md), kept as regression
fixtures: a failure to obtain the multiprocessing context must refuse like a start failure (context-fallback-3);
the standalone default is a fresh process.  Control logic verbatim from checks-f38ddc32/test_r2_independent.py; the
reviewer's evidence recording is replaced by a record under pytest's tmp_path.

minimal split of an earlier change (carved from an earlier revision): the three ``test_standalone_cannot_declare_unenforced_caller_boundary``
cases are NOT carried -- this branch has no caller-boundary mode at all (``execute_translated`` takes no ``isolation``
argument; see test_isolated_verification_start_failure.py::test_no_in_process_mode_exists), so the property they
guard (a standalone caller cannot opt out of containment) holds by construction; ``test_spawn_available_is_used_
without_permitting_refusal`` is not carried either (no second start method on this branch).
"""
from pathlib import Path
import os, json, time, errno, multiprocessing, warnings
import numpy as np
import pytest
from solar.einsum_lib import verify as V
import test_isolated_verification_refusal_controls as R


@pytest.fixture
def record(tmp_path):
    def _record(name, value):
        (tmp_path / 'records').mkdir(exist_ok=True)
        (tmp_path / 'records' / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return _record


@pytest.mark.parametrize('case', ['memory', 'timeout', 'rng'])
def test_context_lookup_failure_must_not_select_thread(case, monkeypatch, tmp_path, record):
    def unavailable():
        raise OSError(errno.EPERM, 'controlled context lookup failure')
    monkeypatch.setattr(V, '_isolation_context', unavailable)
    rec = lambda name, data: record('context-' + name, data)
    if case == 'memory': R.test_unstartable_isolation_keeps_memory_containment(None, tmp_path, rec)
    elif case == 'timeout': R.test_unstartable_isolation_does_not_continue_after_timeout(None, tmp_path, rec)
    else: R.test_unstartable_isolation_preserves_caller_random_state(None, rec)


def test_standalone_default_uses_fresh_process(monkeypatch, record):
    monkeypatch.setattr(V, '_isolation_context', lambda: multiprocessing.get_context('spawn'))
    code = "import os,numpy as np\ndef run_model(inputs):\n    return {'pid':np.array([os.getpid()])}\n"
    out, _ = V.execute_translated(code, {}, fast=False, timeout=30)
    record('standalone-default', {'parent_pid': os.getpid(), 'child_pid': int(out['pid'][0])})
    assert int(out['pid'][0]) != os.getpid()


# ---------------------------------------------------------------------------------------------------------------
# round 2 -- the review isolation-type-1: the REAL facade (solar.translate.verify) inside a daemon worker
# ---------------------------------------------------------------------------------------------------------------

def _example_pair():
    import solar
    ex = Path(solar.__file__).resolve().parents[1] / 'skills' / 'verified-translation' / 'example'
    return ex / 'adapter.py', ex / 'solar_model_einsum_lib.py'


def _facade_in_this_process(fast):
    from solar.translate import verify as facade_verify
    adapter, dsl = _example_pair()
    return facade_verify(adapter, dsl, source_type='cuda', fast=fast, timeout=30, quiet=True, log=lambda *_: None)


def _facade_daemon(out):
    """Runs INSIDE a daemon worker: the real facade on the repository's verified example, both engine modes; records
    whether the facade RAISED the typed refusal or RETURNED a result."""
    records = []
    for fast in (True, False):
        try:
            r = _facade_in_this_process(fast)
            records.append({'fast': fast, 'returned_status': r.status, 'message': (r.message or '')[:300]})
        except BaseException as exc:  # noqa: BLE001 - the class is the observation
            records.append({'fast': fast, 'raised_type': type(exc).__name__,
                            'typed': isinstance(exc, V.IsolationUnavailableError),
                            'cause': type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                            'message': str(exc)[:300]})
    Path(out).write_text(json.dumps({'daemon': multiprocessing.current_process().daemon, 'records': records}))


def _facade_plain(out):
    r = _facade_in_this_process(True)
    Path(out).write_text(json.dumps({'daemon': multiprocessing.current_process().daemon, 'status': r.status,
                                     'message': (r.message or '')[:300]}))


def _run_in_child(target, out, daemon, seconds):
    p = multiprocessing.get_context('spawn').Process(target=target, args=(str(out),), daemon=daemon)
    p.start(); p.join(seconds)
    if p.is_alive():
        p.kill(); p.join()
        pytest.fail(f'{target.__name__} did not finish in {seconds} s')
    assert p.exitcode == 0 and Path(out).exists(), (target.__name__, p.exitcode)
    return json.loads(Path(out).read_text())


def test_facade_inside_a_daemon_raises_the_typed_refusal(tmp_path, record):
    """The real facade cannot start its isolating child in a daemon worker: it RAISES IsolationUnavailableError (it
    does not return status=error), in both engine modes, with the daemon refusal as the cause -- so the dispatcher's
    trusted verify child reports the type (rc 12) and the validation is infrastructure / isolation_unavailable,
    unscored, retained."""
    rec = _run_in_child(_facade_daemon, tmp_path / 'facade-daemon.json', True, 120)
    record('facade-daemon', rec)
    assert rec['daemon'] is True and len(rec['records']) == 2
    for r in rec['records']:
        assert r.get('raised_type') == 'IsolationUnavailableError' and r.get('typed') is True, r
        assert r['cause'] == 'AssertionError' and 'daemonic processes' in r['message']
    from solar.frontend import dispatcher as D
    assert D._classify_exception_names('IsolationUnavailableError', ['IsolationUnavailableError', 'OSError'], 12) \
        == D.FailureKind.INFRASTRUCTURE
    receipt = D._receipt(D.FailureKind.INFRASTRUCTURE, 'agentic_validate', rec['records'][0]['message'],
                         reason_code='isolation_unavailable')
    assert receipt['scored'] is False and receipt['retained_for_retry'] is True


def test_the_same_example_verifies_in_a_plain_process(tmp_path, record):
    """Control: the identical facade call in a plain (non-daemon) fresh process verifies -- the refusal above is the
    daemon's start prohibition, not the example."""
    rec = _run_in_child(_facade_plain, tmp_path / 'facade-plain.json', False, 180)
    record('facade-plain', rec)
    assert rec['daemon'] is False and rec['status'] == 'pass', rec
