# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``verify._run_isolated`` after the parent has initialised GNU OpenMP.

The CI job runs the whole selection in ONE pytest process in ``.gitlab-ci.yml`` order; the numba ``prange`` modules
run before ``test_density_guard``, so by the time the density guard needs an isolated child the process has used
numba's OpenMP threading layer (and torch's).  libgomp aborts a *forked* child outright -- "fork() called from a process
already using GNU OpenMP, this is unsafe" -- and the parent saw ``EOFError`` on the result pipe.  The isolated child is
now a forkserver/spawn process with a picklable entry; these controls initialise OpenMP in THIS process first and then
drive every isolated path (fast, interpreter, the agent's measured-density pass) plus the kept guard semantics.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from solar.einsum_lib import clear_dims, set_density_profiling, stop_trace
from solar.einsum_lib.core import density_profiling_enabled, density_profiling_source, _density_profiling
from solar.einsum_lib.verify import execute_translated, _run_isolated, _isolation_context

PRANGE_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def gemm(A: Tensor("M,K"), B: Tensor("K,N")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m,n] += A[m,k] * B[k,n]
    return Y

def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["a"], "M,K")
    b = Tensor.from_numpy(inputs["b"], "K,N")
    return {"output": gemm(a, b).data}
'''


def _inputs():
    rng = np.random.RandomState(0)
    return {"a": rng.uniform(-1, 1, (16, 8)), "b": rng.uniform(-1, 1, (8, 12))}


def _initialise_openmp():
    """Use numba's parallel layer (and torch, when importable) in THIS process, as the CI order does."""
    import numba

    @numba.njit(parallel=True)
    def total(n):
        s = 0.0
        for i in numba.prange(n):
            s += i * 0.5
        return s

    assert total(1000) == pytest.approx(0.5 * 999 * 1000 / 2)
    try:
        import torch
        torch.ones(128, 128) @ torch.ones(128, 128)
    except Exception:
        pass
    return numba.threading_layer()


# ---- module-level entries (an isolated child is a fresh process; closures cannot travel) ----

def _parallel_sum(n):
    import numba

    @numba.njit(parallel=True)
    def total(m):
        s = 0.0
        for i in numba.prange(m):
            s += i * 0.5
        return s

    return total(n), numba.threading_layer()


def _child_density_state():
    return (density_profiling_enabled(), density_profiling_source(),
            bool(getattr(_density_profiling, "measure_attention", False)))


def _child_env(name):
    return os.environ.get(name)


def _child_sleep(seconds):
    time.sleep(seconds)
    return "slept"


def _child_allocate(nbytes, hold):
    block = np.ones(nbytes, dtype=np.uint8)  # written, so it is resident
    time.sleep(hold)
    return int(block[-1])


def _child_start_method():
    import multiprocessing as mp
    return mp.current_process()._config.get("start_method") or mp.get_start_method(allow_none=True)


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    clear_dims(); stop_trace()


def test_forking_after_openmp_is_the_failure_mode():
    """Control for the mechanism (not the fix): a FORKED child that runs a parallel kernel after this process used
    numba's OpenMP layer dies before answering.  Skipped when numba did not pick the omp layer here."""
    layer = _initialise_openmp()
    if layer != "omp":
        pytest.skip(f"numba threading layer here is {layer!r}; the libgomp abort needs the omp layer")
    script = (
        "import numba, multiprocessing as mp\n"
        "@numba.njit(parallel=True)\n"
        "def f(n):\n    s = 0.0\n    for i in numba.prange(n):\n        s += i\n    return s\n"
        "f(1000)\n"
        "def child(q):\n    q.put(f(1000))\n"
        "ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=child, args=(q,)); p.start(); p.join(30)\n"
        "print('EXIT', p.exitcode, 'EMPTY', q.empty())\n")
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert "fork() called from a process already using GNU OpenMP" in res.stderr or "EMPTY True" in res.stdout, (res.stdout, res.stderr[-500:])


def test_fast_path_is_isolated_after_openmp_initialised():
    _initialise_openmp()
    inputs = _inputs()
    slow, _ = execute_translated(PRANGE_DSL, {}, all_inputs=inputs, fast=False)
    fast, _ = execute_translated(PRANGE_DSL, {}, all_inputs=inputs, fast=True, timeout=120,
                                 mem_budget_bytes=4 * 10**9)
    np.testing.assert_allclose(fast["output"], inputs["a"] @ inputs["b"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fast["output"], slow["output"], rtol=1e-6, atol=1e-6)


def test_interpreter_path_is_isolated_after_openmp_initialised():
    _initialise_openmp()
    inputs = _inputs()
    out, _ = execute_translated(PRANGE_DSL, {}, all_inputs=inputs, fast=False, timeout=120)
    np.testing.assert_allclose(out["output"], inputs["a"] @ inputs["b"], rtol=1e-9, atol=1e-9)


def test_isolated_child_runs_a_parallel_kernel_on_the_workqueue_layer():
    _initialise_openmp()
    value, layer = _run_isolated(_parallel_sum, 120, None, args=(1000,))
    assert value == pytest.approx(0.5 * 999 * 1000 / 2) and layer == "workqueue"


def test_density_profiling_switch_reaches_the_child():
    """The agent's measured-density pass runs under the guard and reads the parent's switch; a fresh child does not
    inherit it, so ``_run_isolated`` carries it."""
    set_density_profiling(True, "measured_capped", measure_attention=True)
    try:
        assert _run_isolated(_child_density_state, 120, None) == (True, "measured_capped", True)
    finally:
        set_density_profiling(False)
    assert _run_isolated(_child_density_state, 120, None) == (False, "measured", False)


def test_density_guard_api_path_after_openmp(monkeypatch):
    """The failing CI case itself: the agent's measured-density pass after this process used OpenMP."""
    _initialise_openmp()
    from solar.einsum_lib.agent import TranslationAgent
    sys.path.insert(0, os.path.dirname(__file__))
    from test_density_guard import DSL, _inputs as guard_inputs
    monkeypatch.setattr(TranslationAgent, "_init_client", lambda self: None)
    monkeypatch.setattr(TranslationAgent, "_call_llm", lambda *args: DSL)
    agent = TranslationAgent(api_key="test-only", max_retries=1, fast=True, use_fp64=True, verify_timeout=60)
    inputs = guard_inputs()
    set_density_profiling(True, source="measured", measure_attention=True)
    try:
        result = agent.translate("test reference", {k: v.shape for k, v in inputs.items()},
                                 reference_outputs={"output": np.ones((4, 3))}, all_inputs=inputs, full_inputs=inputs)
    finally:
        set_density_profiling(False)
    assert result.verified, result.verification_error
    assert result.metadata["tensors"]["probs#0"].get("sparsity") == 10 / 16


def test_parent_environment_reaches_the_child(monkeypatch):
    """forkserver children would otherwise see the server's snapshot of the environment."""
    monkeypatch.setenv("SOLAR_ISOLATION_PROBE", "from-the-parent")
    assert _run_isolated(_child_env, 120, None, args=("SOLAR_ISOLATION_PROBE",)) == "from-the-parent"
    monkeypatch.delenv("SOLAR_ISOLATION_PROBE")
    assert _run_isolated(_child_env, 120, None, args=("SOLAR_ISOLATION_PROBE",)) is None


def test_closure_entry_is_refused_before_any_process_starts():
    with pytest.raises(TypeError, match="picklable entry"):
        _run_isolated(lambda: 1, 10, None)


def test_timeout_semantics_kept():
    t = time.time()
    with pytest.raises(RuntimeError, match="timed out after 2s"):
        _run_isolated(_child_sleep, 2, None, args=(60,))
    assert time.time() - t < 30


def test_memory_budget_semantics_kept():
    with pytest.raises(MemoryError, match="exceeded memory budget"):
        _run_isolated(_child_allocate, 120, 150 * 10**6, args=(600 * 10**6, 20))
    assert _run_isolated(_child_allocate, 120, 2 * 10**9, args=(10 * 10**6, 0)) == 1


@pytest.mark.parametrize("method", ["forkserver", "spawn"])
def test_both_fresh_start_methods_work(monkeypatch, method):
    import multiprocessing as mp
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable here")
    monkeypatch.setenv("SOLAR_ISOLATION_START_METHOD", method)
    assert _isolation_context().get_start_method() == method
    value, layer = _run_isolated(_parallel_sum, 180, None, args=(1000,))
    assert value == pytest.approx(0.5 * 999 * 1000 / 2) and layer == "workqueue"


def test_fork_cannot_be_selected(monkeypatch):
    monkeypatch.setenv("SOLAR_ISOLATION_START_METHOD", "fork")
    with pytest.raises(ValueError, match="fork is unsafe"):
        _run_isolated(_child_env, 10, None, args=("HOME",))


# ---- isolation-rng-28: the child reproduces the caller's seeded state and runs the program once ----

_RNG_PROGRAM = ('import numpy as np\n'
                'R = np.random.uniform(1, 2)\n'
                'def run_model(inputs):\n    return {"output": np.array([R, np.random.uniform(1, 2)])}\n')
# ``random`` is not re-injected by the lowering, so this program only runs on the interpreter path
_RNG_PROGRAM_PY = ('import numpy as np\nimport random\n'
                   'R = np.random.uniform(1, 2)\nP = random.random()\n'
                   'def run_model(inputs):\n    return {"output": np.array([R, P, np.random.uniform(1, 2), random.random()])}\n')
_COUNT_PROGRAM = ('import os\nimport numpy as np\n'
                  'with open(os.environ["SOLAR_EXEC_COUNT_FILE"], "a") as fh:\n    fh.write("x")\n'
                  'def run_model(inputs):\n    return {"output": np.zeros(1)}\n')


def _seeded(fn):
    import random
    original = (np.random.get_state(), random.getstate())
    try:
        np.random.seed(12345)
        random.seed(777)
        return fn()
    finally:
        np.random.set_state(original[0])
        random.setstate(original[1])


@pytest.mark.parametrize("fast,program", [(False, _RNG_PROGRAM), (True, _RNG_PROGRAM), (False, _RNG_PROGRAM_PY)],
                         ids=["interpreter-numpy", "fast-numpy", "interpreter-numpy+random"])
def test_seeded_random_state_reaches_the_child(fast, program):
    """A module-level NumPy draw, a run-time draw (and, on the interpreter path, Python ``random`` draws): the
    isolated run returns the same numbers as the non-isolated run from the same seed -- the state the former fork
    child inherited.  (A program whose lowering FAILS after drawing is not compared here: the in-process fast path
    consumes the caller's generator before it falls back, which isolation -- like fork -- does not; that in-process
    behaviour predates this MR.)"""
    results = {}
    for label, timeout in (("inprocess", None), ("isolated", 30)):
        results[label] = _seeded(lambda: execute_translated(program, {}, all_inputs={}, fast=fast, timeout=timeout)[0]["output"].tolist())
    assert results["inprocess"] == results["isolated"], results
    assert results["isolated"][0] == pytest.approx(1.929616092817148)      # the reviewer's recorded first draw after seed 12345


@pytest.mark.parametrize("fast", [False, True])
def test_caller_state_is_left_where_the_caller_put_it(fast):
    """The child draws from a COPY: the caller's generator is not advanced by the isolated run (as a fork left it)."""
    import random
    def _next_after_isolated():
        execute_translated(_RNG_PROGRAM, {}, all_inputs={}, fast=fast, timeout=30)
        return (np.random.uniform(1, 2), random.random())
    assert _seeded(_next_after_isolated) == _seeded(lambda: (np.random.uniform(1, 2), random.random()))


@pytest.mark.parametrize("fast", [False, True])
def test_program_module_executes_once_under_isolation(tmp_path, monkeypatch, fast):
    """The parent no longer executes the module before isolating it: a module-level side effect happens once per
    call, exactly as without isolation."""
    counter = tmp_path / "count"
    monkeypatch.setenv("SOLAR_EXEC_COUNT_FILE", str(counter))
    for timeout in (None, 60):
        counter.write_text("")
        execute_translated(_COUNT_PROGRAM, {}, all_inputs={}, fast=fast, timeout=timeout)
        assert counter.read_text() == "x", (fast, timeout, counter.read_text())


def test_syntax_error_is_still_reported_as_a_compilation_failure_under_isolation():
    with pytest.raises(RuntimeError, match="Code compilation failed"):
        execute_translated("def (\n", {}, all_inputs={}, fast=False, timeout=30)


def test_missing_run_model_is_reported_from_the_child():
    with pytest.raises(RuntimeError, match="must define a run_model"):
        execute_translated("import numpy as np\nX = 1\n", {}, all_inputs={}, fast=False, timeout=30)
