# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""the reviewer's three independent controls for isolation-fallback-1 (an earlier review.md), kept as
regression fixtures.  Control logic verbatim from checks-3d2ef35c/test_isolation_refusal_independent.py; the
reviewer's evidence recording (REVIEW_PIN-named JSON files under the review tree) is replaced by a record under
pytest's tmp_path so the module runs anywhere.  All three must REFUSE: the entry never executes in the caller.
"""
from pathlib import Path
import os, json, time, errno, warnings, multiprocessing
import numpy as np
import pytest
from solar.einsum_lib import verify as V


class Refusing:
    _name = 'forkserver'
    Pipe = staticmethod(multiprocessing.Pipe)

    class Process:
        def __init__(self, *a, **k): self.pid = None
        def start(self): raise PermissionError(errno.EPERM, 'deliberate start refusal (control)')


class RefusingSpawn(Refusing):
    _name = 'spawn'


def memory_marker(path):
    payload = bytearray(8 * 1024 * 1024)
    Path(path).write_text(json.dumps({'pid': os.getpid(), 'allocated_bytes': len(payload)}))
    time.sleep(.15)
    return {'pid': os.getpid(), 'allocated_bytes': len(payload)}


def delayed_marker(path):
    time.sleep(.3)
    Path(path).write_text(str(os.getpid()))


def random_draw():
    return float(np.random.random())


@pytest.fixture
def unavailable(monkeypatch):
    monkeypatch.setattr(V, '_isolation_context', lambda: Refusing())
    monkeypatch.setattr(V, '_spawn_context', lambda: RefusingSpawn(), raising=False)


@pytest.fixture
def record(tmp_path):
    # the reviewer wrote these under the review tree; here they go to a records/ subdirectory of tmp_path so
    # that a record never shares a path with a control's marker file (memory.json, late.txt)
    def _record(name, data):
        (tmp_path / 'records').mkdir(exist_ok=True)
        (tmp_path / 'records' / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    return _record


def test_unstartable_isolation_keeps_memory_containment(unavailable, tmp_path, record):
    marker = tmp_path / 'memory.json'; result = None; error = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: result = V._run_isolated(memory_marker, 1, 1024, args=(str(marker),))
        except (OSError, MemoryError) as e: error = type(e).__name__
    record('memory', {'result': result, 'error': error, 'parent_pid': os.getpid(), 'marker_exists': marker.exists(), 'requested_bytes': 1024, 'warnings': [str(v.message) for v in w]})
    assert not marker.exists() and result is None, 'unstartable isolator executed in caller despite explicit memory budget'


def test_unstartable_isolation_does_not_continue_after_timeout(unavailable, tmp_path, record):
    marker = tmp_path / 'late.txt'; error = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); t = time.monotonic()
        try: V._run_isolated(delayed_marker, .05, None, args=(str(marker),))
        except (RuntimeError, OSError) as e: error = type(e).__name__
        elapsed = time.monotonic() - t; present_on_return = marker.exists()
        time.sleep(.45)
    record('timeout', {'error': error, 'elapsed_on_return': elapsed, 'present_on_return': present_on_return, 'present_after_wait': marker.exists(), 'parent_pid': os.getpid(), 'marker_pid': marker.read_text() if marker.exists() else None, 'warnings': [str(v.message) for v in w]})
    assert error is not None
    assert not marker.exists(), 'verification work continued in caller after timeout was reported'


def test_unstartable_isolation_preserves_caller_random_state(unavailable, record):
    before = np.random.get_state(); np.random.seed(1039); state = np.random.get_state(); error = None; result = None
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try: result = V._run_isolated(random_draw, 1, None)
            except OSError as e: error = type(e).__name__
        after = np.random.get_state()
        same = state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
        record('rng', {'unchanged': same, 'error': error, 'result': result, 'warnings': [str(v.message) for v in w]})
        assert same, 'fallback consumed caller RNG state, violating isolated-state contract'
    finally: np.random.set_state(before)
