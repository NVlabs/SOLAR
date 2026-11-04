# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification utilities for translated einsum_lib code.

Handles execution of generated code and comparison against reference outputs.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os
import threading
import time as _time

import numpy as np


_ISOLATION_PRELOAD = ("numpy", "numba", "solar.einsum_lib.core",
                      "solar.einsum_lib.numba_accel", "solar.einsum_lib.verify")


def _isolation_context():
    """The multiprocessing context for an isolated child: forkserver, else spawn.

    Never ``fork``.  By the time verification runs, the parent has usually
    executed torch (the reference) or numba's parallel layer, both of which
    initialise GNU OpenMP, and libgomp aborts a forked child outright
    ("fork() called from a process already using GNU OpenMP, this is unsafe")
    -- the parent then sees ``EOFError`` on the result pipe.  A forkserver
    child is forked from a fresh, single-threaded server process that never
    ran a parallel region (and has the heavy modules preloaded once); spawn is
    the equivalent fresh interpreter where forkserver does not exist.  Either
    way the child shares no memory with the parent, so the entry and its
    arguments travel pickled -- see :func:`_run_isolated`.

    ``SOLAR_ISOLATION_START_METHOD`` may select ``forkserver`` or ``spawn``
    explicitly (tests exercise both); ``fork`` is refused.
    """
    import multiprocessing as mp

    methods = mp.get_all_start_methods()
    method = os.environ.get("SOLAR_ISOLATION_START_METHOD") or (
        "forkserver" if "forkserver" in methods else "spawn")
    if method == "fork" or method not in methods:
        raise ValueError(
            f"isolated verification cannot use start method {method!r}: "
            f"fork is unsafe after OpenMP initialisation; available: "
            f"{sorted(set(methods) - {'fork'})}")
    ctx = mp.get_context(method)
    if method == "forkserver":
        try:
            ctx.set_forkserver_preload(list(_ISOLATION_PRELOAD))
        except Exception:
            pass
    return ctx


def _isolated_state():
    """Process state an isolated child must reproduce (it inherits no memory):
    the density-profiling switch that the agent's measured pass relies on, and
    the caller's seeded random state (isolation-rng-28) -- NumPy's global
    generator and Python's ``random`` -- so that a program's module-level or
    run-time draws are the same numbers the non-isolated path (and the former
    fork-based child, which inherited them) would have drawn.  The caller's own
    state is left untouched, exactly as a fork left it."""
    import random
    from solar.einsum_lib import core
    dp = core._density_profiling
    return {"density_profiling": (bool(getattr(dp, "active", False)),
                                  str(getattr(dp, "source", "measured")),
                                  bool(getattr(dp, "measure_attention", False))),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate()}


def _apply_isolated_state(state):
    import random
    active, source, measure_attention = state.get(
        "density_profiling", (False, "measured", False))
    if active:
        from solar.einsum_lib import core
        core.set_density_profiling(True, source, measure_attention=measure_attention)
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])


def _isolated_child(send, args_path, result_path, environ):
    """Entry of the isolated child (module-level, hence picklable).

    Restores the parent's environment (a forkserver child would otherwise see
    the server's snapshot of it), pins numba's fork-safe ``workqueue``
    threading layer before any parallel kernel runs, loads the pickled
    ``(fn, args, kwargs, state)`` from the RAM-backed *args_path*, runs *fn*
    and reports through *send* exactly as the fork-based child did: the
    result goes to *result_path*, only ``("ok", None)`` or ``("err", exc)``
    crosses the pipe.
    """
    import pickle

    try:
        os.environ.clear()
        os.environ.update(environ)
        try:
            import numba
            numba.config.THREADING_LAYER = "workqueue"
        except Exception:
            pass
        with open(args_path, "rb") as fh:
            fn, args, kwargs, state = pickle.load(fh)
        try:
            os.unlink(args_path)
        except OSError:
            pass
        _apply_isolated_state(state)
        result = fn(*args, **kwargs)
        try:
            with open(result_path, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            send.send(("ok", None))
        except Exception as exc:  # unpicklable result
            send.send(("err", RuntimeError(f"result not picklable: {exc}")))
    except BaseException as exc:
        try:
            send.send(("err", exc))
        except Exception:  # unpicklable exception (e.g. numba TypingError)
            send.send(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))
    finally:
        send.close()


class IsolationUnavailableError(OSError):
    """No isolating child process could be started, so the entry was NOT executed.

    Raised by :func:`_run_isolated` when the multiprocessing context cannot be
    obtained or the isolating child cannot be started; ``str(exc)`` is
    ``"isolation_unavailable: <what the host refused>"``.  An OSError: a fact
    about the calling process' environment, never about the program under test;
    callers classify it as an infrastructure (environment) failure (an earlier change the review
    isolation-fallback-1 / CONTEXT-FALLBACK-3)."""

    def __init__(self, reason):
        super().__init__(f"isolation_unavailable: {reason}")
        self.reason = str(reason)


def _start_isolated(ctx, args_path, result_path):
    """Pipe + ``Process`` started under *ctx* -> ``(recv, proc)``; the parent
    keeps only the read end.  Raises what ``Process.start()`` raised."""
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_isolated_child,
                       args=(send, args_path, result_path, dict(os.environ)),
                       daemon=True)
    try:
        proc.start()
    except BaseException:
        send.close()
        recv.close()
        raise
    send.close()
    return recv, proc


def shutdown_isolation(timeout_sec=2.0):
    """Stop and REAP the multiprocessing helper processes this process started for
    isolated verification -- the forkserver and its resource tracker -- so that
    none of them outlives this process.

    The dispatcher's session boundary samples ``/proc`` the instant a phase
    leader exits and refuses ANY process of the phase's session found alive
    (``candidate_descendants_alive``); the helpers exit asynchronously after
    their owner, so a helper merely slow to notice its parent's exit turned a
    healthy validation into an infrastructure refusal on a loaded CI runner
    (dispatcher-c, pipelines 77147099 / 77789291 / 77857565).  The owner --
    the trusted phase child -- calls this before it exits.

    Bounded: a helper gets *timeout_sec* to exit after its alive descriptor is
    closed (the helpers' own stop protocol), then SIGTERM (1 s), then SIGKILL
    (3 s, which also ends a stopped helper); every helper is waited for, so no
    zombie remains.  Idempotent; a helper that was never started is reported as
    ``None``; a later :func:`_run_isolated` restarts what it needs.  Only the
    standard library's module-level helper objects are touched; nothing is
    imported here.  Returns ``{name: {pid, reaped, escalated} | None}``.
    """
    import signal as _signal
    import sys as _sys
    record = {}
    targets = []
    fs = _sys.modules.get("multiprocessing.forkserver")
    rt = _sys.modules.get("multiprocessing.resource_tracker")
    if fs is not None:
        targets.append(("forkserver", fs._forkserver, "_forkserver_pid", "_forkserver_alive_fd"))
    if rt is not None:
        targets.append(("resource_tracker", rt._resource_tracker, "_pid", "_fd"))
    for name, obj, pid_attr, fd_attr in targets:
        pid = getattr(obj, pid_attr, None)
        if not pid:
            record[name] = None
            continue
        fd = getattr(obj, fd_attr, None)
        try:
            if fd is not None:
                os.close(fd)          # the helper's stop protocol: EOF on its alive descriptor
        except OSError:
            pass
        setattr(obj, fd_attr, None)
        escalated = []
        reaped = False
        deadline = _time.monotonic() + timeout_sec
        while True:
            try:
                waited, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                waited = pid          # not (or no longer) our child: nothing left to reap
            if waited == pid:
                reaped = True
                break
            if _time.monotonic() > deadline:
                if len(escalated) == 2:
                    break
                sig = _signal.SIGTERM if not escalated else _signal.SIGKILL
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
                escalated.append(sig.name)
                deadline = _time.monotonic() + (1.0 if sig == _signal.SIGTERM else 3.0)
            _time.sleep(0.01)
        setattr(obj, pid_attr, None)
        if name == "forkserver":
            addr = getattr(obj, "_forkserver_address", None)
            obj._forkserver_address = None
            try:
                from multiprocessing import util as _util
                if addr and not _util.is_abstract_socket_namespace(addr):
                    os.unlink(addr)
            except Exception:
                pass
        record[name] = {"pid": pid, "reaped": reaped, "escalated": escalated}
    return record


def _run_isolated(fn, timeout_sec=None, mem_budget_bytes=None, args=(), kwargs=None):
    """Run ``fn(*args, **kwargs)`` in an isolated child process with wall-clock
    and RSS guards.

    OOM guardrail for IR emulation: a bad translation can materialize huge
    intermediates, and in-process execution then gets SIGKILLed by the system
    OOM-killer, taking the whole verification process down (rc=137).  Running
    in a child contains the damage: if the child exceeds *mem_budget_bytes*
    RSS (polled via psutil) it is killed and the parent raises
    ``MemoryError``; if the system OOM-killer SIGKILLs the child, the parent
    survives and reports it as ``MemoryError`` too.

    The child is a forkserver/spawn process (see :func:`_isolation_context`;
    fork is unsafe once the parent has initialised OpenMP), so *fn* must be a
    module-level function (or another picklable callable) and *args* /
    *kwargs* picklable: they are written to a RAM-backed file that the child
    loads, the same path the result takes back.  A closure is refused with a
    ``TypeError`` before any process starts.  The parent's environment and the
    density-profiling switch are reproduced in the child.

    A child that cannot be STARTED is a fact about this process' environment,
    never about *fn* (target pipeline 77147099 / an earlier change: inside the dispatcher's
    trusted verify child every validation with a timeout became a scored
    candidate error).  When the multiprocessing context cannot be obtained or
    the isolating child cannot be started -- whatever the start raised: an
    OSError, the daemon-worker ``AssertionError``, the spawn bootstrap
    ``RuntimeError`` -- the entry is NOT executed: the call raises
    :class:`IsolationUnavailableError` (an OSError).  There is no
    fallback to a thread of this process, so the requested memory budget, the
    termination guarantee and the caller's random / profiling state are never
    silently given up (the review isolation-fallback-1 and
    CONTEXT-FALLBACK-3).  An error raised by a child that DID start propagates
    exactly as before; an invalid start method is still a ``ValueError``.
    """
    import pickle
    import tempfile

    kwargs = dict(kwargs or {})
    try:
        ctx = _isolation_context()
    except ValueError:
        raise
    except Exception as exc:
        # no multiprocessing context at all: the entry is NOT run here (the review context-fallback-3)
        raise IsolationUnavailableError(f"no multiprocessing context could be obtained ({exc})") from exc

    # Arguments and results go through RAM-backed temp files, NOT the pipe:
    # Pipe/pickle streaming of multi-GB payloads (inputs, weight-gradient
    # outputs) runs at ~tens of MB/s and turns verification into a spurious
    # timeout.  A file under /dev/shm moves the same data at memory speed;
    # only the paths and a status token travel through the pipe.
    _shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, args_path = tempfile.mkstemp(suffix=".args.pkl", dir=_shm_dir)
    os.close(fd)
    fd, result_path = tempfile.mkstemp(suffix=".pkl", dir=_shm_dir)
    os.close(fd)

    def _cleanup():
        for path in (args_path, result_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    try:
        with open(args_path, "wb") as fh:
            pickle.dump((fn, tuple(args), kwargs, _isolated_state()), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        _cleanup()
        raise TypeError(
            "_run_isolated needs a picklable entry -- a module-level function "
            "and picklable arguments -- because the isolated child is a fresh "
            f"process, not a fork of the caller: {exc}") from exc

    try:
        recv, proc = _start_isolated(ctx, args_path, result_path)
    except Exception as exc:
        # the isolating child could not be started: the entry is NOT run here -- the memory
        # budget, the termination guarantee and the caller's state are the contract
        # (the review isolation-fallback-1); typed for the callers, never a thread.  EVERY
        # exception out of the start is a pre-execution refusal of this process' environment,
        # whatever its class (the review isolation-type-1): an OSError from fork/exec or the
        # pipe, the daemon-worker ``AssertionError`` ("daemonic processes are not allowed to have
        # children"), the spawn bootstrap ``RuntimeError``; the child has not run, so nothing here
        # is the program's error -- those arrive through the result pipe or the exit code below.
        _cleanup()
        raise IsolationUnavailableError(
            f"no isolating child process could be started ({type(exc).__name__}: {exc})") from exc
    except BaseException:
        _cleanup()
        raise

    ps_child = None
    if mem_budget_bytes:
        try:
            import psutil
            ps_child = psutil.Process(proc.pid)
        except Exception:
            ps_child = None

    deadline = _time.time() + timeout_sec if timeout_sec else None
    got_message = False
    try:
        while True:
            if recv.poll(0.25):
                got_message = True
                break
            if not proc.is_alive():
                break
            if deadline is not None and _time.time() > deadline:
                proc.kill()
                proc.join()
                raise RuntimeError(f"Verification timed out after {timeout_sec}s")
            if ps_child is not None:
                try:
                    rss = ps_child.memory_info().rss
                except Exception:
                    rss = 0
                if rss > mem_budget_bytes:
                    proc.kill()
                    proc.join()
                    raise MemoryError(
                        f"IR execution exceeded memory budget "
                        f"({rss / 1e9:.1f} GB > {mem_budget_bytes / 1e9:.1f} GB); "
                        f"killed before triggering the system OOM-killer")

        if got_message or recv.poll(0.25):
            try:
                status, payload = recv.recv()
            except EOFError:
                status, payload = None, None  # the child died without a word
            if status is not None:
                proc.join()
                if status == "ok":
                    with open(result_path, "rb") as fh:
                        return pickle.load(fh)
                raise payload if isinstance(payload, BaseException) else RuntimeError(str(payload))

        proc.join()
        if proc.exitcode is not None and proc.exitcode < 0:
            sig = -proc.exitcode
            if sig == 9:
                raise MemoryError(
                    "IR execution process was SIGKILLed — most likely the system "
                    "OOM-killer (allocation exceeded available RAM)")
            raise RuntimeError(f"IR execution subprocess died with signal {sig}")
        raise RuntimeError(
            f"IR execution subprocess exited (code {proc.exitcode}) without a result")
    finally:
        _cleanup()


def _run_in_thread(fn, timeout_sec):
    """Fallback: run *fn* in a daemon thread; raise on timeout or exception."""
    result_box: list = []
    error_box: list = []

    def _worker():
        try:
            result_box.append(fn())
        except Exception as exc:
            error_box.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise RuntimeError(f"Verification timed out after {timeout_sec}s")
    if error_box:
        raise error_box[0]
    return result_box[0]

def _softmax(values):
    """Numerically stable softmax over a list of scalars."""
    vals = list(values)
    mx = max(vals)
    exps = [np.exp(v - mx) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


from solar.einsum_lib.core import fp32 as _fp32_cast

_MATH_BUILTINS = {
    "erf": np.vectorize(math.erf),
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "cos": np.cos,
    "sin": np.sin,
    "tanh": np.tanh,
    "floor": np.floor,
    "round": np.round,
    "log2": np.log2,
    "fp32": _fp32_cast,
    "abs": abs,
    "max": max,
    "min": min,
    "softmax": _softmax,
}


@dataclass
class VerificationResult:
    """Result of verifying translated code against a reference."""

    success: bool
    error: Optional[str] = None
    max_abs_error: Optional[float] = None
    output_shapes_match: bool = True
    # Output keys whose REFERENCE was all zeros (or had no finite element):
    # the comparison for them is vacuous (an all-zero translation passes).
    # A pass with a non-empty tuple is weak evidence and is reported as such
    # by every caller; it is not promoted to a failure here (legacy policy,
    # see solar/translate/_runner.py VerifyResult.exit_status).
    vacuous_outputs: tuple = ()


# ---------------------------------------------------------------------------
# Model capture utilities
# ---------------------------------------------------------------------------


@dataclass
class CapturedRun:
    """Result of running the original model once with hooks."""

    reference_outputs: Dict[str, np.ndarray]
    captured_masks: Dict[str, np.ndarray]
    all_inputs: Dict[str, np.ndarray]
    mask_fill_values: Dict[str, float]
    captured_randn: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)


_DEFAULT_MASK_FILL_VALUE = float("-inf")


def capture_model_run(
    model,
    torch_inputs: list,
    numpy_inputs: Dict[str, np.ndarray],
    seed: int = 42,
) -> CapturedRun:
    """Run a PyTorch model once, capturing reference output and mask tensors.

    Installs three categories of hooks to capture boolean/binary masks
    regardless of how the model consumes them:

    1. **masked_fill / masked_fill_** — additive masking.  The mask
       argument uses PyTorch's convention (1 = replace) and is inverted
       to 1 = keep before storage.  The fill value is recorded.
    2. **Tensor.__getitem__** — when a 2-D+ boolean tensor is indexed
       (e.g. ``mask[i]`` in a gather/scatter attention loop), the full
       boolean tensor is captured.  Already in 1 = keep convention.
    3. **torch.where** — the condition tensor is captured when it is
       boolean and 2-D+.  Already in 1 = keep convention.

    Trivial masks (all-keep after normalisation) are filtered and
    duplicates are removed.

    Args:
        model: PyTorch ``nn.Module`` (already in eval mode with weights loaded).
        torch_inputs: Positional input tensors for ``model.forward()``.
        numpy_inputs: The complete ``{name: ndarray}`` dict that will also be
            given to the translated code.  Returned as-is in ``all_inputs``.
        seed: Random seed set before the forward pass (for models with
            internal randomness, e.g. random attention masks).

    Returns:
        A :class:`CapturedRun` with reference outputs, captured masks
        (all normalised to 1 = keep convention), fill values, and the
        full inputs dict.
    """
    import torch

    # (keep_mask, fill_value) — keep_mask already in 1=keep convention
    captured: List[Tuple[np.ndarray, float]] = []

    def _is_binary(arr: np.ndarray) -> bool:
        return arr.dtype == np.bool_ or set(np.unique(arr)).issubset(
            {0, 1, 0.0, 1.0}
        )

    # --- Hook 1: masked_fill / masked_fill_ (1=replace → invert) ----------

    orig_masked_fill = torch.Tensor.masked_fill
    orig_masked_fill_ = torch.Tensor.masked_fill_

    def _record_masked_fill(mask_tensor, fill_value):
        arr = mask_tensor.detach().cpu().numpy()
        if _is_binary(arr):
            keep = 1.0 - arr.astype(np.float64)
            fv = float(fill_value)
            captured.append((keep, fv))

    def _hook_masked_fill(self, mask, value):
        if isinstance(mask, torch.Tensor):
            _record_masked_fill(mask, value)
        return orig_masked_fill(self, mask, value)

    def _hook_masked_fill_(self, mask, value):
        if isinstance(mask, torch.Tensor):
            _record_masked_fill(mask, value)
        return orig_masked_fill_(self, mask, value)

    # --- Hook 2: __getitem__ on bool 2-D+ tensors -------------------------

    orig_getitem = torch.Tensor.__getitem__

    def _hook_getitem(self, key):
        if self.dtype == torch.bool and self.ndim >= 2:
            # Skip slice-based narrowing (e.g. bias[:, :, :seq, :seq]).
            # The actual mask will be captured downstream by torch.where
            # or masked_fill at the correct (possibly capped) size.
            has_slice = isinstance(key, slice)
            if isinstance(key, tuple):
                has_slice = any(isinstance(k, slice) for k in key)
            if not has_slice:
                arr = self.detach().cpu().numpy().astype(np.float64)
                if not np.all(arr > 0.5):
                    captured.append((arr, _DEFAULT_MASK_FILL_VALUE))
        return orig_getitem(self, key)

    # --- Hook 3: torch.where with bool 2-D+ condition ---------------------

    orig_where = torch.where

    def _hook_where(condition, *args, **kwargs):
        if (
            isinstance(condition, torch.Tensor)
            and condition.dtype == torch.bool
            and condition.ndim >= 2
            and len(args) >= 2
        ):
            arr = condition.detach().cpu().numpy().astype(np.float64)
            if not np.all(arr > 0.5):
                fv = _DEFAULT_MASK_FILL_VALUE
                y = args[1]
                if isinstance(y, (int, float)):
                    fv = float(y)
                elif isinstance(y, torch.Tensor) and y.numel() == 1:
                    fv = float(y.item())
                captured.append((arr, fv))
        return orig_where(condition, *args, **kwargs)

    # --- Hook 4: torch.randn / randn_like (internal random state) --------

    orig_randn = torch.randn
    orig_randn_like = torch.randn_like
    captured_randn_tensors: List[np.ndarray] = []

    def _hook_randn(*args, **kwargs):
        result = orig_randn(*args, **kwargs)
        captured_randn_tensors.append(result.detach().cpu().numpy())
        return result

    def _hook_randn_like(*args, **kwargs):
        result = orig_randn_like(*args, **kwargs)
        captured_randn_tensors.append(result.detach().cpu().numpy())
        return result

    # --- Install hooks and run -------------------------------------------

    torch.Tensor.masked_fill = _hook_masked_fill
    torch.Tensor.masked_fill_ = _hook_masked_fill_
    torch.Tensor.__getitem__ = _hook_getitem
    torch.where = _hook_where
    torch.randn = _hook_randn
    torch.randn_like = _hook_randn_like
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            output = model(*torch_inputs)
    finally:
        torch.Tensor.masked_fill = orig_masked_fill
        torch.Tensor.masked_fill_ = orig_masked_fill_
        torch.Tensor.__getitem__ = orig_getitem
        torch.where = orig_where
        torch.randn = orig_randn
        torch.randn_like = orig_randn_like

    # --- Collect reference outputs ----------------------------------------

    def _out_to_numpy(t):
        t = t.detach().cpu()
        if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            t = t.float()
        return t.numpy()

    if isinstance(output, torch.Tensor):
        ref_outputs = {"output": _out_to_numpy(output)}
    elif isinstance(output, dict):
        ref_outputs = {
            k: _out_to_numpy(v) if isinstance(v, torch.Tensor)
            else np.array(v)
            for k, v in output.items()
        }
    elif isinstance(output, (tuple, list)):
        ref_outputs = {
            f"output_{i}": _out_to_numpy(o) if isinstance(o, torch.Tensor)
            else np.array(o)
            for i, o in enumerate(output)
        }
    else:
        ref_outputs = {"output": np.array(output)}

    # --- Deduplicate and filter trivial masks ----------------------------

    masks: Dict[str, np.ndarray] = {}
    fill_values: Dict[str, float] = {}
    seen: set = set()
    for keep_mask, fv in captured:
        if np.all(keep_mask > 0.5):
            continue

        key = (keep_mask.shape, keep_mask.tobytes())
        if key not in seen:
            seen.add(key)
            name = f"captured_mask_{len(masks)}"
            masks[name] = keep_mask
            fill_values[name] = fv

    randn_dict: Dict[str, np.ndarray] = {}
    for i, arr in enumerate(captured_randn_tensors):
        randn_dict[f"captured_randn_{i}"] = arr

    return CapturedRun(
        reference_outputs=ref_outputs,
        captured_masks=masks,
        all_inputs=numpy_inputs,
        mask_fill_values=fill_values,
        captured_randn=randn_dict,
    )


def compute_sparsity(array) -> float:
    """Compute the density (fraction of non-zero elements) of an array.

    Args:
        array: numpy array or boolean mask.

    Returns:
        Float in ``[0, 1]`` — fraction of non-zero elements.
        For a boolean mask, this is the fraction of ``True`` values.
    """
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr) / arr.size)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

_INTEGER_DTYPES = frozenset({"int8", "int32", "int64"})


def _is_integer_dtype(dtype: str) -> bool:
    return dtype in _INTEGER_DTYPES


def _isolated_execute_fast(code, inputs, use_fp64, fail_on_numba_decline, fail_on_untyped_fast_failure=False):
    """Fast-path entry (module-level, picklable for :func:`_run_isolated`).
    Returns ``(result, decline_reasons)``; the caller reports the reasons.

    ``fail_on_numba_decline`` (the policy before an earlier fix commit B, kept behind the
    agent's ``reprompt_on_numba_decline=True``): EVERY fast-path None raises.
    ``fail_on_untyped_fast_failure`` (round 10, agent-routing-scope-31,
    the typed-decline-only adjudication): only an UNTYPED failure
    raises -- generated code crashing in the lowered module, the lowered module
    failing to import, no typed reason at all -- while a TYPED decline
    (``kernel_<code>:...``, see ``numba_accel.is_typed_decline``) returns None
    and the caller falls back to the interpreter.  With both False (the
    non-agent default) every None falls back, as always."""
    from solar.einsum_lib.numba_accel import execute_fast, is_typed_decline, untyped_fast_failure_class

    reasons: list = []
    result = execute_fast(code, inputs, use_fp64=use_fp64, on_fallback=reasons.append)
    if result is None:
        reason = reasons[0] if reasons else "unknown reason"
        if fail_on_numba_decline:
            # During agent retry loops, grinding the ~100x slower interpreter
            # after a numba decline just burns the verify timeout (one decline =
            # a dead cap round).  Fail fast with the decline reason so the LLM
            # can fix the code instead.
            raise RuntimeError(
                f"Numba declined the generated code ({reason}). "
                f"Rewrite so it lowers cleanly: only supported builtins "
                f"(exp/log/sqrt/tanh/cos/sin/floor/log2/round, apply_mask, top_k_mask, "
                f"gather, slice_at, stack, concat), no np.* inside "
                f"@kernel bodies, and a top-level run_model().")
        if fail_on_untyped_fast_failure and not is_typed_decline(reason):
            raise RuntimeError(
                f"The fast path failed without a typed decline "
                f"[{untyped_fast_failure_class(reason)}] ({reason}). "
                f"A typed numba decline would be verified on the interpreter; "
                f"this is a failure of the generated code itself -- fix the code.")
    return result, reasons


def _isolated_run_model(code, inputs):
    """Interpreter entry (module-level, picklable): the isolated child executes
    the program module ONCE -- the parent does not execute it when the run is
    isolated (isolation-rng-28), so module-level side effects and random
    draws happen exactly once, in the child, from the caller's transferred
    random state."""
    import math as _math

    namespace: dict = {**_MATH_BUILTINS, "math": _math, "np": np}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        raise RuntimeError(f"Code compilation failed: {e}") from e
    if "run_model" not in namespace:
        raise RuntimeError("Generated code must define a run_model(inputs) function")
    return namespace["run_model"](inputs)


def execute_translated(
    code: str,
    input_shapes: Dict[str, tuple],
    seed: int = 42,
    input_dtypes: Optional[Dict[str, str]] = None,
    fixed_inputs: Optional[Dict[str, np.ndarray]] = None,
    all_inputs: Optional[Dict[str, np.ndarray]] = None,
    trace: bool = False,
    fast: bool = True,
    shape_only: bool = False,
    use_fp64: bool = False,
    timeout: Optional[int] = None,
    mem_budget_bytes: Optional[int] = None,
    fail_on_numba_decline: bool = False,
    fail_on_untyped_fast_failure: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Execute translated einsum_lib code.

    The generated code must define a ``run_model(inputs: dict) -> dict``
    function that accepts numpy arrays and returns numpy arrays.

    Args:
        code: Generated Python source (must define ``run_model``).
        input_shapes: Mapping of input names to shape tuples.
        seed: Random seed for reproducible inputs (ignored when
            *all_inputs* is provided).
        input_dtypes: Optional mapping of input names to dtype strings.
            Integer dtypes (int8/int32/int64) get random integer arrays;
            everything else gets float64.  Ignored when *all_inputs* is
            provided.
        fixed_inputs: Optional mapping of input names to pre-built numpy
            arrays.  Keys present here bypass random generation.  Ignored
            when *all_inputs* is provided.
        all_inputs: When provided, this complete ``{name: ndarray}`` dict
            is used directly — no random generation occurs.  This is the
            preferred mode when inputs were pre-generated alongside the
            reference outputs via :func:`capture_model_run`.
        trace: If ``True``, enable runtime tracing during execution.
            Uses dummy execution (``trace_only``) so loop bodies are
            skipped — only tensor constructors and kernel/builtin
            hooks fire.  After return, call ``get_trace()`` to read
            the trace entries.
        fast: If ``True`` and ``trace`` is ``False``, attempt to use
            Numba JIT acceleration for faster execution.  Falls back
            to the standard slow path if Numba is unavailable or
            lowering fails.
        timeout: Maximum seconds to allow for execution.  ``None``
            means no limit.  If exceeded, raises ``RuntimeError``.
        mem_budget_bytes: RSS budget for IR execution.  When set (or when
            *timeout* is set), execution runs in an isolated child process
            (forkserver/spawn -- never fork, see :func:`_run_isolated`) that
            is killed if it exceeds the budget — raising ``MemoryError``
            instead of letting the system OOM-killer take the process down.
        fail_on_numba_decline: Raise ``RuntimeError`` on EVERY fast-path
            failure instead of falling back to the interpreter (the agent's
            pre-an earlier fix fail-fast policy, ``reprompt_on_numba_decline=True``).
        fail_on_untyped_fast_failure: Raise ``RuntimeError`` only when the
            fast path failed WITHOUT a typed decline (generated code crashing
            on the fast path, a lowered module that does not import); a typed
            decline still falls back to the interpreter.  The fast agent's
            default since an earlier fix round 10 (typed-decline-only routing); other
            callers keep falling back on every failure.

    Returns:
        ``(outputs, inputs)`` — both are dicts of name to numpy array.

    Raises:
        RuntimeError: On compilation, execution failure, or timeout.
    """
    if all_inputs is not None:
        inputs = dict(all_inputs)
    else:
        fixed = fixed_inputs or {}
        dtypes = input_dtypes or {}
        rng = np.random.RandomState(seed)
        inputs = {}
        for name, shape in input_shapes.items():
            if name in fixed:
                inputs[name] = fixed[name]
            elif _is_integer_dtype(dtypes.get(name, "")):
                high = max(shape) if shape else 100
                inputs[name] = rng.randint(0, high, size=shape).astype(np.int64)
            else:
                inputs[name] = rng.uniform(-0.1, 0.1, size=shape).astype(np.float64)

    if fast and not trace:
        # The interpreter below is correct but roughly 100x slower, so say why
        # we ended up there rather than letting it look like a slow model.
        def _report(reason: str) -> None:
            print(f"  [WARN] Numba declined this code, falling back to the "
                  f"interpreter (much slower): {reason}")

        fast_result, decline_reasons = (
            _run_isolated(_isolated_execute_fast, timeout, mem_budget_bytes,
                          args=(code, inputs, use_fp64, fail_on_numba_decline, fail_on_untyped_fast_failure))
            if (timeout or mem_budget_bytes)
            else _isolated_execute_fast(code, inputs, use_fp64, fail_on_numba_decline, fail_on_untyped_fast_failure)
        )
        for reason in decline_reasons:
            _report(reason)
        if fast_result is not None:
            if not isinstance(fast_result, dict):
                raise RuntimeError(
                    f"run_model() must return a dict, got {type(fast_result).__name__}"
                )
            return fast_result, inputs

    import math as _math
    namespace: dict = {**_MATH_BUILTINS, "math": _math, "np": np}
    isolated = bool((timeout or mem_budget_bytes) and not trace)

    if isolated:
        # The isolated child executes the module itself, once, from the
        # caller's random state (isolation-rng-28); executing it here as
        # well would run module-level side effects twice and advance the
        # caller's generator.  Only the syntax is checked in this process.
        try:
            compile(code, "<translated>", "exec")
        except SyntaxError as e:
            raise RuntimeError(f"Code compilation failed: {e}") from e
    else:
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            raise RuntimeError(f"Code compilation failed: {e}") from e

        if "run_model" not in namespace:
            raise RuntimeError(
                "Generated code must define a run_model(inputs) function"
            )

    if trace:
        from solar.einsum_lib.core import start_trace, _tracer as tracer_obj
        start_trace(trace_only=shape_only)
        tracer_obj.register_inputs(inputs)

    try:
        def _run():
            return namespace["run_model"](inputs)

        if trace:
            # Trace state lives in this process — no subprocess isolation.
            outputs = (
                _run_in_thread(_run, timeout) if timeout
                else namespace["run_model"](inputs)
            )
            # which traced tensors were RETURNED (issue #59): the analyzer
            # bills only these as external outputs
            tracer_obj.record_model_outputs(outputs)
        else:
            outputs = (
                _run_isolated(_isolated_run_model, timeout, mem_budget_bytes,
                              args=(code, inputs))
                if isolated
                else namespace["run_model"](inputs)
            )
    except IsolationUnavailableError:
        # the environment refusal keeps its type through this wrapper (the review isolation-type-5)
        if trace:
            from solar.einsum_lib.core import stop_trace
            stop_trace()
        raise
    except Exception as e:
        if trace:
            from solar.einsum_lib.core import stop_trace
            stop_trace()
        raise RuntimeError(f"run_model() failed: {e}") from e

    if trace:
        from solar.einsum_lib.core import stop_trace
        stop_trace()

    if not isinstance(outputs, dict):
        raise RuntimeError(
            f"run_model() must return a dict, got {type(outputs).__name__}"
        )

    return outputs, inputs


def compare_outputs(
    translated: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    min_tol: float = 0.0,
) -> VerificationResult:
    """Compare translated outputs against reference outputs element-wise.

    Args:
        translated: Outputs from the translated einsum_lib code.
        reference: Outputs from the original program.
        rtol: Relative tolerance for ``np.allclose``.
        atol: Absolute tolerance for ``np.allclose``.
        min_tol: Floor applied to both tolerances.  Lets a caller that knows the
            reference cannot be exact -- because the kernel stores its output in
            a precision narrower than the IR computes in -- raise the tolerance
            to what that precision permits, without having to know what the
            tolerance would otherwise have been.  Zero leaves *rtol* and *atol*
            untouched.

    Returns:
        A ``VerificationResult`` with success status and error details.
    """
    rtol = max(rtol, min_tol)
    atol = max(atol, min_tol)
    t_keys = set(translated.keys())
    r_keys = set(reference.keys())
    if t_keys != r_keys:
        return VerificationResult(
            success=False,
            error=(
                f"Output key mismatch: translated has {t_keys}, "
                f"reference has {r_keys}"
            ),
            output_shapes_match=False,
        )

    max_abs_err = 0.0
    vacuous: list = []
    for key in translated:
        t_out = np.asarray(translated[key], dtype=np.float64)
        r_out = np.asarray(reference[key], dtype=np.float64)

        if t_out.shape != r_out.shape:
            return VerificationResult(
                success=False,
                error=(
                    f"Shape mismatch for '{key}': "
                    f"translated {t_out.shape} vs reference {r_out.shape}"
                ),
                output_shapes_match=False,
            )

        # A size-0 output has nothing to compare: it is a verdict (not
        # verified), never a ValueError from np.max on an empty array.
        if r_out.size == 0:
            return VerificationResult(
                success=False,
                error=(
                    f"Output '{key}' is empty (size 0, shape {r_out.shape}): "
                    f"nothing to compare, so this output cannot be "
                    f"numerically verified; fix the capped inputs/shapes"
                ),
                max_abs_error=None,
                output_shapes_match=True,
            )

        # Vacuous-pass guard: an all-zero reference output compares equal to an
        # all-zero (i.e. broken or trivially-wrong) translation.  Most common
        # cause: capped/degenerate inputs (zero-filled tensors, a mask that
        # blanks everything, seq/window caps that empty a branch).  Warn loudly
        # but do not fail — a few references are all-zero BY DESIGN (e.g. a
        # no-op mask fill); the operator must confirm that from the source.
        finite_ref = r_out[np.isfinite(r_out)] if np.issubdtype(
            np.asarray(r_out).dtype, np.floating) else np.asarray(r_out)
        if finite_ref.size == 0 or not np.any(finite_ref):
            vacuous.append(key)
            print(
                f"WARNING: reference output '{key}' is ALL ZEROS — "
                f"verification of this output is vacuous (an all-zero "
                f"translation would pass). Check capped inputs / masks; "
                f"confirm against the reference source whether zero is "
                f"genuinely expected."
            )

        abs_diff = float(np.max(np.abs(t_out - r_out)))
        max_abs_err = max(max_abs_err, abs_diff)

        if not np.allclose(t_out, r_out, rtol=rtol, atol=atol):
            return VerificationResult(
                success=False,
                error=(
                    f"Output '{key}' values differ: "
                    f"max_abs_error={abs_diff:.6e}, rtol={rtol}, atol={atol}"
                ),
                max_abs_error=abs_diff,
                output_shapes_match=True,
            )

    return VerificationResult(
        success=True,
        max_abs_error=max_abs_err,
        output_shapes_match=True,
        vacuous_outputs=tuple(vacuous),
    )
