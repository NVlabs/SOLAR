# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OFFLINE tests of ``solar.einsum_lib.fallback_contract`` (lite-fallback-contract-v2; round 2 after the reviewer's HOLD:
P2-FALLBACK-UNTYPED-1, P2-FALLBACK-FAILURE-RECEIPT-1, P2-FALLBACK-119-INPUTS-1).

Everything here is CPU, in-process, stubbed: the fast engine's decline is simulated by monkeypatching
``numba_accel.numba_lower`` and by lowered texts that raise the engine's typed decline exception.  Arrays are tiny
(one element where the finding used one; a 2x3x5 @ 5x4 matmul for the real-interpreter cases).  No Lite workload,
reference adapter, saved receipt, candidate, GPU or model API is touched.
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from solar.einsum_lib import numba_accel
from solar.einsum_lib import fallback_contract as fc

BIND = {"engine_pin": "test-pin-0000", "binding": {"out_dir": "/tmp/attempt", "row": 0}}

MATMUL_DSL = """\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    for b in Dim("B"):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("K"):
                    C[b,m,n] += A[b,m,k] * W[k,n]
    return C

def model(x, w):
    return matmul(x, w)

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight")
    out = model(x, w)
    return {"output": out.data}
"""
# a one-element program whose interpreter run echoes its input (the review's [7.0] probe shape)
ECHO_DSL = """\
import numpy as np
def run_model(inputs):
    return {"output": np.array(inputs["x"], copy=True)}
"""
ECHO_DSL_RAISES = """\
def run_model(inputs):
    raise ValueError("interpreter candidate error")
"""
ECHO_DSL_BAD_IMPORT = "raise ImportError('interpreter module import failed')\n"

TYPED_REASON = ("kernel_function_statement_unresolved:matmul: name_bound_to_non_kernel_value: a '.data' attribute at "
                "line 19 is read from 'out' in the function 'run_model'")

PLAIN_LOWERED_OK = """\
import numpy as np
def run_model(inputs):
    return {"output": np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])}
"""
LOWERED_RAISES_TYPED_AT_IMPORT = """\
from solar.einsum_lib.numba_accel import KernelOutputDtypeUnresolved
raise KernelOutputDtypeUnresolved("matmul", "unestablished", "the @kernel decorator in the live module is not einsum_lib's")
"""
LOWERED_RAISES_TYPED_IN_RUN_MODEL = """\
from solar.einsum_lib.numba_accel import KernelOutputDtypeUnresolved
def run_model(inputs):
    inputs.pop("x")          # consume the dict: a re-run MUST get fresh inputs
    raise KernelOutputDtypeUnresolved("matmul", "conflict", "declared bf16, computed fp32")
"""
LOWERED_MUTATES_THEN_DECLINES = """\
from solar.einsum_lib.numba_accel import KernelOutputDtypeUnresolved
def run_model(inputs):
    inputs["x"][...] = 99.0  # partial fast execution mutating its inputs IN PLACE, then a typed decline
    raise KernelOutputDtypeUnresolved("echo", "conflict", "declared bf16, computed fp32")
"""
LOWERED_RAISES_INDEXERROR = """\
def run_model(inputs):
    return {"output": inputs["x"][0, 0, 99]}
"""
LOWERED_NO_RUN_MODEL = "x = 1\n"


def _inputs():
    rng = np.random.RandomState(7)
    return {"x": rng.uniform(-1, 1, size=(2, 3, 5)), "w": rng.uniform(-1, 1, size=(5, 4))}


def _reference(inputs):
    return np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])


class _Factory:
    def __init__(self, base=None):
        self.calls = 0
        self.base = base if base is not None else _inputs()

    def __call__(self):
        self.calls += 1
        return {k: np.array(v, copy=True) for k, v in self.base.items()}


class _Persist:
    """A persist callback that keeps the event sequence and a deep copy of every record (what a harness writes)."""

    def __init__(self):
        self.events = []

    def __call__(self, event, record):
        self.events.append((event, dict(record)))

    @property
    def names(self):
        return [e for e, _ in self.events]

    def last(self, event):
        return [r for e, r in self.events if e == event][-1]


def _stub_lowerer(result, *, typed_reason=None, with_hook=True):
    if with_hook:
        def numba_lower(code, boundscheck=False, use_fp64=True, on_decline=None):
            numba_lower.calls.append({"boundscheck": boundscheck, "use_fp64": use_fp64, "hook": on_decline is not None})
            if result is None and typed_reason is not None and on_decline is not None:
                on_decline(typed_reason)
            return result
    else:
        def numba_lower(code, boundscheck=False, use_fp64=True):   # the earlier revision signature: no hook
            numba_lower.calls.append({"boundscheck": boundscheck, "use_fp64": use_fp64, "hook": False})
            return result
    numba_lower.calls = []
    return numba_lower


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(numba_accel, "HAS_NUMBA", True)
    return numba_accel


def _assert_no_verdict(rec):
    assert rec["verdict"] is None and "pass" not in rec["outcome"].lower()


# ---------------------------------------------------------------------------------------------------------------------
# typed lowering decline -> interpreter (transition persisted first)
# ---------------------------------------------------------------------------------------------------------------------

class TestTypedLoweringDecline:
    def test_typed_decline_routes_to_the_interpreter_and_the_comparator_still_decides(self, engine, monkeypatch):
        stub = _stub_lowerer(None, typed_reason=TYPED_REASON)
        monkeypatch.setattr(engine, "numba_lower", stub)
        persist = _Persist()
        prog = fc.lower_or_fallback(MATMUL_DSL, boundscheck=True, use_fp64=True, persist=persist, **BIND)
        assert prog.backend == fc.BACKEND_INTERPRETER and prog.program_text == MATMUL_DSL and prog.lowered is None
        assert prog.decline_typed and prog.decline_stage == fc.STAGE_LOWERING and prog.decline_reason == TYPED_REASON
        assert stub.calls == [{"boundscheck": True, "use_fp64": True, "hook": True}]       # execute_fast's exact call
        # the transition was persisted BEFORE any interpreter work
        assert persist.names == ["transition-lowering"]
        t = persist.last("transition-lowering")
        assert t["event"] == "transition-lowering" and t["backend_attempt"] == "interpreter" and t["backend_terminal"] is None
        assert t["engine"]["engine_pin"] == "test-pin-0000" and {k: v for k, v in t["binding"].items() if k != "run_dir"} == BIND["binding"] and t["event_utc"] and t["binding"]["run_dir"] == _os.path.realpath(BIND["binding"]["out_dir"])
        assert t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING and t["backend"] == "interpreter"
        assert t["decline_reason"] == TYPED_REASON and t["decline_typed"] is True and t["decline_stage"] == "lowering"
        assert t["history"] == [{"backend": "numba", "stage": "lowering", "reason": TYPED_REASON, "typed": True}]
        _assert_no_verdict(t)
        factory = _Factory()
        ns = prog.exec_module(prog.make_namespace(), "<dsl>")
        out = prog.run_model(ns, factory)
        assert factory.calls == 1
        assert out["output"].dtype == np.float64                                            # interpreter storage
        np.testing.assert_allclose(out["output"], _reference(factory.base), rtol=1e-12, atol=1e-12)
        rec = prog.record()
        assert rec["outcome"] == fc.OUTCOME_INTERPRETER_RAN and rec["backend"] == "interpreter" and rec["fallback_taken"]
        assert rec["executed_program"] == "original_dsl_interpreter"
        assert rec["program_text_sha256"] == rec["source_sha256"] == hashlib.sha256(MATMUL_DSL.encode()).hexdigest()
        assert rec["lowered_sha256"] is None and rec["interpreter_failure"] is None and rec["backend_terminal"] == "interpreter"
        assert persist.names == ["transition-lowering", "interpreter-ran"]
        _assert_no_verdict(rec)

    def test_interpreter_namespace_is_verify_s_namespace(self):
        from solar.einsum_lib.verify import _MATH_BUILTINS
        ns = fc.interpreter_namespace()
        assert set(ns) == set(_MATH_BUILTINS) | {"math", "np"}
        assert ns["np"] is np and all(ns[k] is _MATH_BUILTINS[k] for k in _MATH_BUILTINS)


# ---------------------------------------------------------------------------------------------------------------------
# P2-FALLBACK-UNTYPED-1: ONLY typed declines reroute
# ---------------------------------------------------------------------------------------------------------------------

class TestUntypedDeclinesDoNotReroute:
    def test_no_hook_pin_untyped_decline_is_a_visible_non_pass_without_interpreter(self, engine, monkeypatch):
        stub = _stub_lowerer(None, with_hook=False)                 # an earlier revision: numba_lower(code, boundscheck, use_fp64)
        monkeypatch.setattr(engine, "numba_lower", stub)
        persist = _Persist()
        with pytest.raises(fc.UntypedDecline) as ei:
            fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        rec = ei.value.record
        assert rec["outcome"] == fc.OUTCOME_UNTYPED_DECLINE and rec["backend"] == "numba" and rec["fallback_taken"] is False
        assert rec["decline_typed"] is False and rec["decline_hook_available"] is False and rec["decline_stage"] == "lowering"
        assert rec["decline_reason"].startswith("UNTYPED: numba_lower returned None and this engine pin exposes no on_decline hook")
        assert rec["history"] == [] and persist.names == ["untyped-decline"]
        assert stub.calls == [{"boundscheck": True, "use_fp64": True, "hook": False}]
        _assert_no_verdict(rec)

    def test_silent_hook_untyped_decline_is_a_visible_non_pass_without_interpreter(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None))     # hook present, never called
        persist = _Persist()
        with pytest.raises(fc.UntypedDecline) as ei:
            fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        rec = ei.value.record
        assert rec["outcome"] == fc.OUTCOME_UNTYPED_DECLINE and rec["backend"] == "numba" and rec["decline_hook_available"] is True
        assert rec["decline_reason"].startswith("UNTYPED: numba_lower returned None without reporting a reason")
        assert persist.names == ["untyped-decline"]

    def test_missing_numba_is_an_environment_failure_never_a_fallback(self, monkeypatch):
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", False)
        persist = _Persist()
        with pytest.raises(fc.NumbaUnavailable) as ei:
            fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        rec = ei.value.record
        assert rec["outcome"] == fc.OUTCOME_NUMBA_UNAVAILABLE and rec["backend"] == "numba" and rec["fallback_taken"] is False
        assert persist.names == ["numba-unavailable"]
        assert "require_numba" not in inspect.signature(fc.lower_or_fallback).parameters   # the opt-in is gone
        assert {"engine_pin", "binding"} <= set(inspect.signature(fc.lower_or_fallback).parameters)

    def test_untyped_decline_through_the_one_call_form_carries_its_record(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, with_hook=False))
        persist = _Persist()
        with pytest.raises(fc.UntypedDecline) as ei:
            fc.execute_candidate(ECHO_DSL, _Factory({"x": np.array([7.0])}), persist=persist, **BIND)
        assert ei.value.record["outcome"] == fc.OUTCOME_UNTYPED_DECLINE and persist.names == ["untyped-decline"]


# ---------------------------------------------------------------------------------------------------------------------
# the fast path when nothing declines
# ---------------------------------------------------------------------------------------------------------------------

class TestFastPathUnchanged:
    def test_successful_lowering_stays_on_numba_and_records_it(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        persist = _Persist()
        prog = fc.lower_or_fallback(MATMUL_DSL, use_fp64=True, persist=persist, **BIND)
        assert prog.backend == fc.BACKEND_NUMBA and prog.program_text == PLAIN_LOWERED_OK and prog.outcome == fc.OUTCOME_FAST_LOWERED
        assert prog.decline_reason is None and prog.decline_stage is None and not prog.decline_typed
        factory = _Factory()
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        out = prog.run_model(ns, factory)
        assert factory.calls == 1
        np.testing.assert_allclose(out["output"], _reference(factory.base), rtol=1e-12, atol=1e-12)
        rec = prog.record()
        assert rec["outcome"] == fc.OUTCOME_FAST_RAN and rec["backend"] == "numba" and rec["fallback_taken"] is False
        assert rec["program_text_sha256"] == rec["lowered_sha256"] == hashlib.sha256(PLAIN_LOWERED_OK.encode()).hexdigest()
        assert rec["history"] == [] and persist.names == ["fast-ran"] and rec["backend_terminal"] == "numba" and rec["backend_attempt"] == "numba"
        _assert_no_verdict(rec)

    def test_numba_namespace_is_the_engine_s_execution_namespace(self, engine, monkeypatch):
        seen = {}

        def execution_namespace(use_fp64=True):
            seen["use_fp64"] = use_fp64
            return {"__marker__": True}

        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        monkeypatch.setattr(engine, "execution_namespace", execution_namespace)
        prog = fc.lower_or_fallback(MATMUL_DSL, use_fp64=False, **BIND)
        assert prog.make_namespace() == {"__marker__": True} and seen == {"use_fp64": False}

    def test_a_caller_compiled_code_object_is_used_as_is(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        prog = fc.lower_or_fallback(MATMUL_DSL, **BIND)
        compiled = compile(prog.program_text, "/tmp/executed-lowered.py", "exec")
        ns = prog.exec_module(prog.make_namespace(), "/tmp/executed-lowered.py", compiled=compiled)
        assert ns["run_model"].__code__.co_filename == "/tmp/executed-lowered.py"

    def test_an_index_error_in_the_lowered_candidate_is_not_a_decline_and_propagates(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_INDEXERROR))
        persist = _Persist()
        prog = fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        factory = _Factory()
        with pytest.raises(IndexError):
            prog.run_model(ns, factory)
        assert factory.calls == 1 and persist.names == []
        assert prog.backend == fc.BACKEND_NUMBA and prog.decline_reason is None and prog.record()["fallback_taken"] is False

    def test_a_lowered_module_without_run_model_is_an_error_not_a_fallback(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_NO_RUN_MODEL))
        prog = fc.lower_or_fallback(MATMUL_DSL, **BIND)
        with pytest.raises(RuntimeError, match="no run_model"):
            prog.exec_module(prog.make_namespace(), "<lowered>")
        assert prog.backend == fc.BACKEND_NUMBA

    def test_a_pin_without_the_typed_exception_class_never_treats_errors_as_declines(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_AT_IMPORT))
        monkeypatch.delattr(engine, "KernelOutputDtypeUnresolved")
        prog = fc.lower_or_fallback(MATMUL_DSL, **BIND)
        with pytest.raises(Exception):
            prog.exec_module(prog.make_namespace(), "<lowered>")
        assert prog.backend == fc.BACKEND_NUMBA


# ---------------------------------------------------------------------------------------------------------------------
# typed declines AFTER lowering (module import, run_model): transition persisted first, success path
# ---------------------------------------------------------------------------------------------------------------------

class TestLaterStageDeclinesSucceed:
    def test_module_exec_decline_persists_the_transition_before_the_interpreter_runs(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_AT_IMPORT))
        persist = _Persist()
        prog = fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        assert prog.backend == fc.BACKEND_NUMBA and persist.names == []
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        assert persist.names == ["transition-module_exec"]
        t = persist.last("transition-module_exec")
        assert t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING and t["decline_stage"] == "module_exec" and t["decline_typed"]
        assert "KernelOutputDtypeUnresolved" in t["decline_reason"] and "unestablished" in t["decline_reason"]
        assert t["lowered_sha256"] == hashlib.sha256(LOWERED_RAISES_TYPED_AT_IMPORT.encode()).hexdigest()
        assert t["program_text_sha256"] == t["source_sha256"]
        assert callable(ns["run_model"]) and "np" in ns
        factory = _Factory()
        out = prog.run_model(ns, factory)
        np.testing.assert_allclose(out["output"], _reference(factory.base), rtol=1e-12, atol=1e-12)
        assert prog.record()["outcome"] == fc.OUTCOME_INTERPRETER_RAN and persist.names == ["transition-module_exec", "interpreter-ran"]

    def test_run_model_decline_reruns_the_original_on_fresh_inputs_after_persisting(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_IN_RUN_MODEL))
        persist = _Persist()
        prog = fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        factory = _Factory()
        out = prog.run_model(ns, factory)
        assert factory.calls == 2                                        # one per attempt; the consumed dict was not reused
        assert persist.names == ["transition-run_model", "interpreter-ran"]
        assert persist.last("transition-run_model")["decline_stage"] == "run_model"
        assert "declared bf16, computed fp32" in prog.record()["decline_reason"]
        np.testing.assert_allclose(out["output"], _reference(factory.base), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# P2-FALLBACK-FAILURE-RECEIPT-1: typed decline followed by a FAILED interpreter attempt / resource stop
# ---------------------------------------------------------------------------------------------------------------------

class TestInterpreterFailureRecords:
    def _check_failure(self, ei, persist, stage, exc_name, msg):
        rec = ei.value.record
        assert isinstance(ei.value, fc.InterpreterFailed) and isinstance(ei.value.__cause__, Exception)
        assert rec["outcome"].startswith(f"interpreter_failed:{exc_name}: {msg}")
        assert rec["backend"] == "interpreter" and rec["decline_typed"] and rec["decline_stage"] == stage
        assert rec["interpreter_failure"]["type"] == exc_name and rec["interpreter_failure"]["prior_decline_stage"] == stage
        assert rec["interpreter_failure"]["prior_decline_reason"] == rec["decline_reason"] and rec["decline_reason"]
        assert persist.names == [f"transition-{stage}", "interpreter-failed"]
        t = persist.last(f"transition-{stage}")
        assert t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING and t["interpreter_failure"] is None   # persisted before the attempt
        _assert_no_verdict(rec); _assert_no_verdict(t)

    def test_lowering_decline_then_interpreter_import_failure(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        persist = _Persist()
        prog = fc.lower_or_fallback(ECHO_DSL_BAD_IMPORT, persist=persist, **BIND)
        with pytest.raises(fc.InterpreterFailed) as ei:
            prog.exec_module(prog.make_namespace(), "<dsl>")
        self._check_failure(ei, persist, "lowering", "ImportError", "interpreter module import failed")

    def test_lowering_decline_then_interpreter_run_model_failure(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        persist = _Persist()
        prog = fc.lower_or_fallback(ECHO_DSL_RAISES, persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<dsl>")
        with pytest.raises(fc.InterpreterFailed) as ei:
            prog.run_model(ns, _Factory({"x": np.array([7.0])}))
        self._check_failure(ei, persist, "lowering", "ValueError", "interpreter candidate error")

    def test_module_exec_decline_then_interpreter_failure(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_AT_IMPORT))
        persist = _Persist()
        prog = fc.lower_or_fallback(ECHO_DSL_RAISES, persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        with pytest.raises(fc.InterpreterFailed) as ei:
            prog.run_model(ns, _Factory({"x": np.array([7.0])}))
        self._check_failure(ei, persist, "module_exec", "ValueError", "interpreter candidate error")

    def test_run_model_decline_then_interpreter_failure(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_IN_RUN_MODEL))
        persist = _Persist()
        prog = fc.lower_or_fallback(ECHO_DSL_RAISES, persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<lowered>")
        with pytest.raises(fc.InterpreterFailed) as ei:
            prog.run_model(ns, _Factory({"x": np.array([7.0])}))
        self._check_failure(ei, persist, "run_model", "ValueError", "interpreter candidate error")

    def test_one_call_form_failure_raises_with_the_persisted_record(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_AT_IMPORT))
        persist = _Persist()
        with pytest.raises(fc.InterpreterFailed) as ei:
            fc.execute_candidate(ECHO_DSL_RAISES, _Factory({"x": np.array([7.0])}), persist=persist, **BIND)
        self._check_failure(ei, persist, "module_exec", "ValueError", "interpreter candidate error")
        assert ei.value.record == persist.last("interpreter-failed")          # the exception carries the persisted record

    def test_an_interrupt_during_the_interpreter_attempt_is_persisted_and_re_raised_unchanged(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        persist = _Persist()
        prog = fc.lower_or_fallback("def run_model(inputs):\n    raise KeyboardInterrupt()\n", persist=persist, **BIND)
        ns = prog.exec_module(prog.make_namespace(), "<dsl>")
        with pytest.raises(KeyboardInterrupt):
            prog.run_model(ns, _Factory({"x": np.array([7.0])}))
        assert persist.names == ["transition-lowering", "interpreter-failed"]
        assert persist.last("interpreter-failed")["interpreter_failure"]["type"] == "KeyboardInterrupt"

    def test_resource_stop_record_is_built_from_the_persisted_transition(self, engine, monkeypatch):
        """A SIGKILL during the fallback runs no cleanup: the transition persisted BEFORE the attempt is what survives,
        and the controller derives the distinct RESOURCE_STOP record from it."""
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        persist = _Persist()
        fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)          # the child was killed right after this point
        t = persist.last("transition-lowering")
        rs = fc.resource_stop_record(t, "RSS 2,147,483,648 B > cap during interpreter attempt; killed by controller")
        assert rs["outcome"] == fc.OUTCOME_RESOURCE_STOP and rs["backend"] == "interpreter"
        assert rs["resource_stop"]["prior_decline_reason"] == TYPED_REASON and rs["resource_stop"]["prior_decline_stage"] == "lowering"
        assert rs["decline_reason"] == TYPED_REASON and [e["event"] for e in rs["events"]] == ["transition-lowering", "resource-stop"]
        assert rs["event"] == "resource-stop" and rs["backend_attempt"] == "interpreter" and rs["backend_terminal"] is None and fc.validate_record(rs)
        assert t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING                 # the transition record itself is retained unchanged
        _assert_no_verdict(rs)


# ---------------------------------------------------------------------------------------------------------------------
# P2-FALLBACK-119-INPUTS-1: fresh arrays from an immutable prepared baseline
# ---------------------------------------------------------------------------------------------------------------------

class TestPreparedInputsBaseline:
    def test_mutating_fast_attempt_then_typed_decline_interpreter_sees_the_prepared_values(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_MUTATES_THEN_DECLINES))
        fp64_in = {"x": np.array([7.0])}
        baseline = fc.prepared_inputs(fp64_in)
        persist = _Persist()
        out, rec = fc.execute_candidate(ECHO_DSL, baseline, persist=persist, **BIND)
        assert out["output"].tolist() == [7.0] and out["output"].dtype == np.float64        # not the mutated [99.0]
        assert rec["outcome"] == fc.OUTCOME_INTERPRETER_RAN and rec["decline_stage"] == "run_model"
        assert fp64_in["x"].tolist() == [7.0]                                                # the caller's dict untouched too
        acc = baseline.accounting()
        assert acc["copies_made"] == 2 and acc["bytes_copied"] == 16 and acc["baseline_bytes"] == 8 and acc["peak_extra_bytes_bound"] == 16
        assert baseline.verify()

    def test_shallow_dict_copy_would_have_leaked_the_mutation(self, engine, monkeypatch):
        """The round-1 counterexample (``lambda: dict(fp64_in)``) reproduced: with a factory that only copies the dict,
        the interpreter sees [99.0].  Kept as the control for the baseline fix above."""
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_MUTATES_THEN_DECLINES))
        fp64_in = {"x": np.array([7.0])}
        out, _ = fc.execute_candidate(ECHO_DSL, lambda: dict(fp64_in), **BIND)
        assert out["output"].tolist() == [99.0]

    def test_baseline_arrays_are_read_only_hashed_and_value_preserving(self):
        src = {"a": np.array([[1.5, -2.0]], dtype=np.float32), "i": np.array([3], dtype=np.int64)}
        b = fc.prepared_inputs(src)
        f1 = b.fresh(); f2 = b.fresh()
        assert f1["a"].dtype == np.float32 and f1["a"].shape == (1, 2) and f1["i"].dtype == np.int64
        assert f1["a"] is not f2["a"] and f1["a"].flags.writeable
        f1["a"][0, 0] = 42.0
        assert f2["a"][0, 0] == 1.5 and src["a"][0, 0] == 1.5
        with pytest.raises(ValueError):
            b._base["a"][0, 0] = 0.0                       # the baseline itself is read-only
        assert b.verify() and set(b.digest) == {"a", "i"} and b.digest["a"]["nbytes"] == 8
        b._base["a"].setflags(write=True); b._base["a"][0, 0] = 0.0
        with pytest.raises(RuntimeError, match="immutability violated"):
            b.verify()


# ---------------------------------------------------------------------------------------------------------------------
# environment / signature guards
# ---------------------------------------------------------------------------------------------------------------------

class TestSignatures:
    def test_execute_candidate_on_the_fast_path(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        factory = _Factory()
        out, rec = fc.execute_candidate(MATMUL_DSL, factory, **BIND)
        np.testing.assert_allclose(out["output"], _reference(factory.base), rtol=1e-12, atol=1e-12)
        assert rec["outcome"] == fc.OUTCOME_FAST_RAN and rec["backend"] == "numba" and rec["decline_reason"] is None

    def test_the_real_engine_signature_is_what_the_shim_expects(self):
        params = inspect.signature(numba_accel.numba_lower).parameters
        assert {"code", "boundscheck", "use_fp64", "on_decline"} <= set(params)
        assert hasattr(numba_accel, "execution_namespace") and hasattr(numba_accel, "KernelOutputDtypeUnresolved")

    def test_the_module_uses_absolute_imports_only_so_it_can_sit_beside_a_harness(self):
        import ast, pathlib
        tree = ast.parse(pathlib.Path(fc.__file__).read_text())
        assert [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.level or 0) > 0] == []

    def test_no_record_ever_carries_a_verdict(self):
        import ast, pathlib
        src = pathlib.Path(fc.__file__).read_text()
        assert src.count('"verdict"') >= 2 and 'rec["verdict"] = None' in src and '"verdict": None' in src and '"verdict": None' in src
        assert "PASS" not in src.replace("NON_PASS", "").replace("non-pass", "").replace("NON-PASS", "").replace("never a pass", "").replace("pass/fail", "").replace("a pass", "")


# ---------------------------------------------------------------------------------------------------------------------
# round 3 -- P2-R2-TRANSITION-BINDING-1: required self-binding fields on EVERY persisted record
# ---------------------------------------------------------------------------------------------------------------------

def _required_fields_present(rec):
    for k in fc.REQUIRED_RECORD_FIELDS:
        assert k in rec, k
    assert rec["engine"]["engine_pin"] == "test-pin-0000"
    for k in fc.REQUIRED_ENGINE_FIELDS:
        assert rec["engine"].get(k), k
    import datetime as _dt
    _dt.datetime.fromisoformat(rec["event_utc"])
    assert rec["backend_attempt"] in ("numba", "interpreter") and {k: v for k, v in rec["binding"].items() if k != "run_dir"} == BIND["binding"]
    assert len(rec["source_sha256"]) == 64 and rec["verdict"] is None
    fc.validate_record(rec)


class TestRequiredBindingFields:
    def test_every_transition_and_terminal_record_is_self_bound(self, engine, monkeypatch):
        cases = [(LOWERED_RAISES_TYPED_AT_IMPORT, ECHO_DSL), (LOWERED_RAISES_TYPED_IN_RUN_MODEL, ECHO_DSL), (None, ECHO_DSL)]
        for lowered, src in cases:
            monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(lowered, typed_reason=TYPED_REASON))
            persist = _Persist()
            out, rec = fc.execute_candidate(src, _Factory({"x": np.array([7.0])}), persist=persist, **BIND)
            assert len(persist.events) == 2 and persist.names[0].startswith("transition-") and persist.names[1] == "interpreter-ran"
            for ev, r in persist.events:
                _required_fields_present(r)
                assert r["event"] == ev
            t = persist.events[0][1]
            assert t["backend_attempt"] == "interpreter" and t["backend_terminal"] is None and t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING
            assert (t["lowered_sha256"] is None) == (lowered is None)                      # null only for a lowering-stage decline
            assert persist.events[1][1]["backend_terminal"] == "interpreter"               # the producer of the compared numbers
            _required_fields_present(rec)

    def test_failure_and_untyped_records_are_self_bound_too(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(LOWERED_RAISES_TYPED_AT_IMPORT))
        persist = _Persist()
        with pytest.raises(fc.InterpreterFailed):
            fc.execute_candidate(ECHO_DSL_RAISES, _Factory({"x": np.array([7.0])}), persist=persist, **BIND)
        for _, r in persist.events:
            _required_fields_present(r)
        assert persist.last("interpreter-failed")["backend_attempt"] == "interpreter" and persist.last("interpreter-failed")["backend_terminal"] is None
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, with_hook=False))
        persist = _Persist()
        with pytest.raises(fc.UntypedDecline):
            fc.lower_or_fallback(ECHO_DSL, persist=persist, **BIND)
        _required_fields_present(persist.last("untyped-decline"))
        assert persist.last("untyped-decline")["backend_attempt"] == "numba"

    def test_engine_identity_hashes_the_imported_engine(self):
        import hashlib, pathlib
        e = fc.engine_identity("pin-x", {"use_fp64": True})
        assert e["engine_pin"] == "pin-x" and e["config"] == {"use_fp64": True}
        assert e["numba_accel_sha256"] == hashlib.sha256(pathlib.Path(numba_accel.__file__).read_bytes()).hexdigest()
        assert pathlib.Path(e["solar_package_path"]).name == "solar" and e["python"] and e["executable"]

    def test_missing_pin_or_binding_is_refused_before_anything_runs(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        with pytest.raises(fc.RecordInvalid):
            fc.lower_or_fallback(MATMUL_DSL, engine_pin="", binding={"out_dir": "x"})
        with pytest.raises(fc.RecordInvalid):
            fc.lower_or_fallback(MATMUL_DSL, engine_pin="pin", binding={})
        with pytest.raises(TypeError):
            fc.lower_or_fallback(MATMUL_DSL)                     # keyword-only, mandatory

    def test_validate_record_rejects_each_omission(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        persist = _Persist()
        fc.lower_or_fallback(MATMUL_DSL, persist=persist, **BIND)
        good = persist.last("transition-lowering")
        for k in ("engine", "event_utc", "backend_attempt", "source_sha256", "binding"):
            bad = dict(good); del bad[k]
            with pytest.raises(fc.RecordInvalid):
                fc.validate_record(bad)
        bad = dict(good); bad["engine"] = {**good["engine"], "engine_pin": None}
        with pytest.raises(fc.RecordInvalid):
            fc.validate_record(bad)
        bad = dict(good); bad["verdict"] = "PASS"
        with pytest.raises(fc.RecordInvalid):
            fc.validate_record(bad)
        bad = dict(good); bad["event_utc"] = "not-a-time"
        with pytest.raises(fc.RecordInvalid):
            fc.validate_record(bad)


# ---------------------------------------------------------------------------------------------------------------------
# round 3 -- P2-R2-OUTER-STOP-1: the controller post-processor joins a surviving transition to RESOURCE_STOP
# ---------------------------------------------------------------------------------------------------------------------

import os as _os
import datetime as datetime_mod
import signal as _signal
import subprocess as _subprocess
import sys as _sys
import time as _time
import json as _json
import pathlib as _pathlib

_CHILD = r'''
import sys, json, pathlib, time
sys.path.insert(0, sys.argv[1]); out = pathlib.Path(sys.argv[2]); mode = sys.argv[3]
from solar.einsum_lib import numba_accel, fallback_contract as fc
numba_accel.HAS_NUMBA = True
def numba_lower(code, boundscheck=False, use_fp64=True, on_decline=None):
    if on_decline is not None:
        on_decline("kernel_function_statement_unresolved:echo: name_bound_to_non_kernel_value: stub typed reason")
    return None
numba_accel.numba_lower = numba_lower
persist = lambda ev, rec: (out / ("fallback-" + ev + ".json")).write_text(json.dumps(rec, indent=1))
src = "import pathlib, time\ndef run_model(inputs):\n    pathlib.Path(%r).write_text('entered')\n    time.sleep(120)\n    return {'output': inputs['x']}\n" % str(out / "interpreter-entered")
if mode == "exit-without-terminal":
    src = "import pathlib\ndef run_model(inputs):\n    pathlib.Path(%r).write_text('entered')\n    import os; os._exit(3)\n" % str(out / "interpreter-entered")
if mode == "complete":
    src = "def run_model(inputs):\n    return {'output': inputs['x']}\n"
prog = fc.lower_or_fallback(src, engine_pin="stub-pin-1749ded0", binding={"out_dir": str(out), "row": 999, "attempt": 1}, persist=persist)
ns = prog.exec_module(prog.make_namespace(), "<stub>")
import numpy as np
prog.run_model(ns, lambda: {"x": np.array([7.0])})
'''


def _run_child(tmp_path, mode, solar_root, kill=False):
    out = tmp_path / mode; out.mkdir()
    env = dict(_os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    p = _subprocess.Popen([_sys.executable, "-c", _CHILD, str(solar_root), str(out), mode], env=env, start_new_session=True,
                          stdout=_subprocess.DEVNULL, stderr=_subprocess.PIPE)
    guard = None
    if kill:
        t0 = _time.monotonic()
        while not (out / "interpreter-entered").exists():
            assert p.poll() is None, p.stderr.read().decode()[-2000:]
            assert _time.monotonic() - t0 < 60, "child never entered the interpreter attempt"
            _time.sleep(0.05)
        _os.killpg(p.pid, _signal.SIGKILL)            # the controller's guard kill, mid interpreter attempt
        guard = "RSS_CAP"
    rc = p.wait(timeout=120)
    return out, rc, guard


@pytest.fixture(scope="module")
def solar_root():
    import solar
    return _pathlib.Path(solar.__file__).resolve().parent.parent


class TestOuterPostprocess:
    def test_stub_child_sigkill_during_interpreter_attempt_yields_joined_resource_stop(self, tmp_path, solar_root):
        out, rc, guard = _run_child(tmp_path, "kill", solar_root, kill=True)
        assert rc == -9
        t = _json.loads((out / "fallback-transition-lowering.json").read_text())            # survived the kill, byte-for-byte
        assert t["outcome"] == fc.OUTCOME_INTERPRETER_PENDING and t["backend_attempt"] == "interpreter"
        assert not any(p.name.startswith("fallback-interpreter-ran") for p in out.iterdir())
        attempt = fc.LaunchDescriptor.standalone({"out_dir": str(out), "row": 999, "attempt": 1})   # identity present on BOTH sides (the child bound {out_dir,row,attempt}); a key on one side only is a mismatch
        s = fc.postprocess_attempt(out, attempt=attempt, child_state="GUARD_KILLED_RSS_CAP", child_exit=rc, guard_kill=guard)
        assert s["non_pass"] is True and s["verdict"] is None
        info = s["tags"][""]
        assert info["status"] == fc.OUTCOME_RESOURCE_STOP and info["rejected_events"] == [] and info["scope"] == "parent"
        rs = _json.loads((out / "fallback-resource-stop.json").read_text())
        assert rs["outcome"] == fc.OUTCOME_RESOURCE_STOP and rs["event"] == "resource-stop" and rs["verdict"] is None
        assert rs["decline_reason"] == t["decline_reason"] and rs["decline_typed"] and rs["decline_stage"] == "lowering"
        assert rs["resource_stop"]["prior_decline_reason"] == t["decline_reason"] and rs["resource_stop"]["attempt"]["run_dir"] == _os.path.realpath(str(out))
        assert s["mint_check"]["state"] == "NO_MINT_EXPECTED"
        assert rs["engine"]["engine_pin"] == "stub-pin-1749ded0" and rs["binding"]["row"] == 999 and rs["backend_attempt"] == "interpreter"
        fc.validate_record(rs)
        assert (out / "FALLBACK_POSTPROCESS.json").exists()
        # the transition itself is retained unchanged
        assert _json.loads((out / "fallback-transition-lowering.json").read_text()) == t

    def test_child_exiting_without_terminal_event_is_an_open_transition_non_pass(self, tmp_path, solar_root):
        out, rc, guard = _run_child(tmp_path, "exit-without-terminal", solar_root)
        assert rc == 3 and guard is None
        s = fc.postprocess_attempt(out, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(out), "row": 999, "attempt": 1}), child_state="EXIT_3", child_exit=rc, guard_kill=None)
        assert s["non_pass"] is True and s["tags"][""]["status"] == fc.OUTCOME_OPEN_TRANSITION
        op = _json.loads((out / "fallback-open-transition.json").read_text())
        assert op["outcome"] == fc.OUTCOME_OPEN_TRANSITION and op["verdict"] is None and op["decline_typed"] and op["open_transition"]["child_exit"] == 3
        assert not (out / "fallback-resource-stop.json").exists()

    def test_completed_interpreter_attempt_leaves_the_verdict_to_the_comparator(self, tmp_path, solar_root):
        out, rc, guard = _run_child(tmp_path, "complete", solar_root)
        assert rc == 0
        s = fc.postprocess_attempt(out, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(out), "row": 999, "attempt": 1}), child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s["non_pass"] is False and s["verdict"] is None
        assert s["tags"][""]["status"].startswith("CLOSED_COMPARATOR_DECIDES") and "interpreter-ran" in s["tags"][""]["terminals"]
        assert not (out / "fallback-resource-stop.json").exists() and not (out / "fallback-open-transition.json").exists()

    def test_binding_mismatch_and_missing_events_are_non_pass(self, tmp_path, solar_root):
        out, rc, _ = _run_child(tmp_path, "complete", solar_root)
        s = fc.postprocess_attempt(out, attempt=fc.LaunchDescriptor.standalone({"out_dir": "/somewhere/else", "row": 999}), child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s["non_pass"] is True and s["tags"][""]["status"].startswith("NO_VALID_EVENT+REJECTED:")
        assert all(r["code"] == "IDENTITY_MISMATCH" for r in s["rejected_events"])
        empty = tmp_path / "empty"; empty.mkdir()
        s2 = fc.postprocess_attempt(empty, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(empty)}), child_state="GUARD_KILLED_WALL_CAP", child_exit=-9, guard_kill="WALL_CAP")
        assert s2["non_pass"] is True and s2["tags"][""]["status"].startswith("NO_EXECUTION_EVIDENCE+RESOURCE_STOP_BEFORE_ANY_FALLBACK_EVENT")   # v3: + the unjoined kill record
        s3 = fc.postprocess_attempt(empty, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(empty)}), child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s3["non_pass"] is True and s3["tags"][""]["status"] == "NO_EXECUTION_EVIDENCE" and s3["no_execution_evidence"] is True   # v3 (P2-R6-EMPTY-ATTEMPT-1): fail closed, exit 0 never closes

    def test_119_style_tagged_events_are_grouped_per_case(self, tmp_path, engine, monkeypatch):
        monkeypatch.setattr(engine, "numba_lower", _stub_lowerer(None, typed_reason=TYPED_REASON))
        out = tmp_path / "r119"; out.mkdir()
        persist = lambda tag: (lambda ev, rec: (out / f"{tag}.fallback-{ev}.json").write_text(_json.dumps(rec)))
        BS = {"engine_pin": BIND["engine_pin"], "binding": {**BIND["binding"], "stage": "precheck"}}; BC = {"engine_pin": BIND["engine_pin"], "binding": {**BIND["binding"], "case": "caseA"}}
        prog = fc.lower_or_fallback(ECHO_DSL, persist=persist("precheck"), execution=False, **BS)     # the 119 pre-check: PREPARATION, never an execution transition
        assert prog.preparation_only and prog.record()["outcome"] == fc.OUTCOME_PRECHECK_DECLINE and prog.program_text == ECHO_DSL
        with pytest.raises(fc.RecordInvalid):
            prog.exec_module(prog.make_namespace())
        fc.execute_candidate(ECHO_DSL, _Factory({"x": np.array([7.0])}), persist=persist("caseA"), **BC)   # transition + interpreter-ran under the case tag
        s = fc.postprocess_attempt(out, attempt=fc.LaunchDescriptor.standalone({"out_dir": "/tmp/attempt", "row": 0}), child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert set(s["tags"]) == {"precheck.", "caseA."}
        assert s["tags"]["caseA."]["status"].startswith("CLOSED_COMPARATOR_DECIDES")
        assert s["tags"]["precheck."]["status"] == "PREPARATION_RECORDED:precheck-decline" and s["tags"]["precheck."]["non_pass"] is False
        assert s["non_pass"] is False and s["execution_tags"] == 1 and s["preparation_only_tags"] == 1   # round-3 test encoded the false aggregate; fixed (P2-R3-119-PRECHECK-1)
        assert not (out / "precheck.fallback-open-transition.json").exists()
        assert (out / "precheck.fallback-precheck-decline.json").exists()


# ---------------------------------------------------------------------------------------------------------------------
# round 4 -- P2-R3-ATTEMPT-JOIN-1: every event used for closure is validated against the SAME attempt identity
# (the reviewer's five record probes, the disjoint-binding probe and the killed-with-partial-terminal fixture)
# ---------------------------------------------------------------------------------------------------------------------

REVIEW_SRC = 'def run_model(inputs):\n    return {"out": inputs["x"].copy()}\n'


def _typed_engine(monkeypatch):
    monkeypatch.setattr(numba_accel, "HAS_NUMBA", True)
    monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(None, typed_reason="kernel_output_dtype_unestablished: reviewer synthetic postprocess"))


def _create_pair(monkeypatch, out):
    """A valid transition + a valid interpreter-ran terminal, bound to {out_dir, row 229, attempt 1} (the reviewer's create_pair)."""
    _typed_engine(monkeypatch)
    persist = lambda ev, rec: (out / ("fallback-" + ev + ".json")).write_text(_json.dumps(rec))
    fc.execute_candidate(REVIEW_SRC, lambda: {"x": np.array([7.0])}, engine_pin="pin-r4", binding={"out_dir": str(out), "row": 229, "attempt": 1}, persist=persist)
    return fc.LaunchDescriptor.standalone({"out_dir": str(out), "row": 229, "attempt": 1})


class TestAttemptJoinValidation:
    @pytest.mark.parametrize("alteration", ["binding", "source", "engine", "timestamp", "malformed"])
    def test_a_foreign_or_malformed_terminal_never_closes_the_transition(self, monkeypatch, tmp_path, alteration):
        attempt = _create_pair(monkeypatch, tmp_path)
        path = tmp_path / "fallback-interpreter-ran.json"; terminal = _json.loads(path.read_text())
        if alteration == "binding": terminal["binding"] = {"out_dir": "/different-run", "row": 113, "attempt": 9}
        elif alteration == "source": terminal["source_sha256"] = "a" * 64
        elif alteration == "engine": terminal["engine"]["numba_accel_sha256"] = "b" * 64
        elif alteration == "timestamp": terminal["event_utc"] = "2000-01-01T00:00:00+00:00"
        if alteration == "malformed": path.write_text('{"interrupted_write":')
        else: path.write_text(_json.dumps(terminal))
        s = fc.postprocess_attempt(tmp_path, attempt=attempt, child_state="COMPLETED", child_exit=0, guard_kill=None)
        info = s["tags"][""]
        assert s["non_pass"] is True
        assert info["terminals"] == [] and info["surviving_transition"] == "transition-lowering"
        assert info["status"].startswith(fc.OUTCOME_OPEN_TRANSITION + "+REJECTED:interpreter-ran:")       # the transition stays open; the rejected terminal is named
        assert len(s["rejected_events"]) == 1 and s["rejected_events"][0]["event"] == "interpreter-ran"
        expected = {"binding": "IDENTITY_MISMATCH", "source": "SOURCE_MISMATCH", "engine": "ENGINE_HASH_MISMATCH", "timestamp": "TIME_BEFORE_TRANSITION", "malformed": "UNREADABLE"}[alteration]
        assert s["rejected_events"][0]["code"] == expected
        assert (tmp_path / "fallback-open-transition.json").exists()

    def test_a_malformed_terminal_during_a_kill_does_not_lose_the_joined_stop(self, monkeypatch, tmp_path):
        _create_pair(monkeypatch, tmp_path)
        (tmp_path / "fallback-interpreter-ran.json").write_text('{"interrupted_write":')
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path), "row": 229, "attempt": 1}), child_state="GUARD_KILLED_OFFLINE", child_exit=-9, guard_kill="OFFLINE")
        assert s["non_pass"] is True and s["tags"][""]["status"].startswith(fc.OUTCOME_RESOURCE_STOP + "+REJECTED:interpreter-ran:UNREADABLE")
        rs = _json.loads((tmp_path / "fallback-resource-stop.json").read_text())
        assert rs["outcome"] == fc.OUTCOME_RESOURCE_STOP and rs["decline_typed"] and rs["rejected_events"][0]["event"] == "interpreter-ran"
        fc.validate_record(rs)

    def test_disjoint_binding_is_a_mismatch_never_an_agreement(self, monkeypatch, tmp_path):
        # the reviewer's probe shape: a transition bound to {row: 229} only vs an attempt {out_dir: /different-run, group: C, attempt: 9}.
        # lower_or_fallback now refuses a binding without the run directory at the source ...
        _typed_engine(monkeypatch)
        with pytest.raises(fc.RecordInvalid):
            fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding={"row": 229})
        # ... and a transition FILE carrying such a binding (written by anything else) is rejected by the join, never matched
        persist = lambda ev, rec: (tmp_path / ("fallback-" + ev + ".json")).write_text(_json.dumps(rec))
        prog = fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding={"out_dir": str(tmp_path), "row": 229}, persist=persist)
        t = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text()); t["binding"] = {"row": 229}
        (tmp_path / "fallback-transition-lowering.json").write_text(_json.dumps(t))
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": "/different-run", "group": "C", "attempt": 9}), child_state="GUARD_KILLED_OFFLINE", child_exit=-9, guard_kill="OFFLINE")
        assert s["non_pass"] is True and s["tags"][""]["surviving_transition"] is None and s["rejected_events"][0]["code"] == "SCHEMA"   # the type schema (binding.out_dir) is checked BEFORE identity; either way: never a match
        assert fc.bindings_match({"row": 229}, fc.LaunchDescriptor.standalone({"out_dir": "/different-run", "group": "C", "attempt": 9}))["matches"] is False
        # intersecting keys alone never match: same run_dir but an identity key on one side only
        assert fc.bindings_match({"out_dir": str(tmp_path), "row": 229}, fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path), "row": 229, "attempt": 9}))["matches"] is False
        assert fc.bindings_match({"out_dir": str(tmp_path), "row": 229, "attempt": 1}, fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path), "row": 229, "attempt": 1}))["matches"] is True

    def test_terminal_only_fast_record_is_validated_against_the_mint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", True); monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(PLAIN_LOWERED_OK))
        attempt = fc.mint_attempt(tmp_path, group="C", attempt=1, row=70, source_sha256=hashlib.sha256(MATMUL_DSL.encode()).hexdigest(), engine_pin="pin-r4")
        persist = lambda ev, rec: (tmp_path / ("fallback-" + ev + ".json")).write_text(_json.dumps(rec))
        fc.execute_candidate(MATMUL_DSL, _Factory(), engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path), persist=persist)
        s = fc.postprocess_attempt(tmp_path, attempt=attempt, child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s["non_pass"] is False and s["tags"][""]["terminals"] == ["fast-ran"] and s["mint_check"]["state"] == "MINT_OK" and s["attempt"]["group"] == "C"
        # the same fast record with a foreign source hash is rejected against the mint
        f = _json.loads((tmp_path / "fallback-fast-ran.json").read_text()); f["source_sha256"] = "c" * 64; (tmp_path / "fallback-fast-ran.json").write_text(_json.dumps(f))
        s2 = fc.postprocess_attempt(tmp_path, attempt=attempt, child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s2["non_pass"] is True and s2["rejected_events"][0]["code"] == "SOURCE_MISMATCH"

    def test_mint_and_attempt_binding_put_the_same_identity_on_both_sides(self, tmp_path):
        launch = fc.mint_attempt(tmp_path, group="C", attempt=1, row=229, admission_sha256="d" * 64)
        b = fc.attempt_binding(tmp_path)
        assert b["group"] == "C" and b["attempt"] == 1 and b["row"] == 229 and b["run_dir"] == _os.path.realpath(str(tmp_path))
        assert fc.bindings_match(b, launch)["matches"] is True
        assert fc.bindings_match({k: v for k, v in b.items() if k != "row"}, launch)["matches"] is False   # row on one side only -> mismatch
        assert launch.authority == "parent" and launch.mint_sha256 and fc.check_mint(tmp_path, launch)["state"] == "MINT_OK"
        other = tmp_path / "other"; other.mkdir()
        assert fc.attempt_binding(other, group="D")["group"] == "D"      # no mint there: the caller's own identity is kept (and the outer attempt must carry the same)
        fc.mint_attempt(other, group="D", attempt=2)
        with pytest.raises(fc.RecordInvalid):
            fc.attempt_binding(other, group="E")                        # conflicts with the mint
        (tmp_path / fc.MINT_FILE).write_text("{bad json")
        with pytest.raises(fc.RecordInvalid):
            fc.attempt_binding(tmp_path)                                 # the child cannot bind to a corrupt mint
        assert fc.check_mint(tmp_path, launch)["state"] == "MINT_ALTERED"   # and the parent never trusts it

    def test_a_stale_transition_file_from_another_attempt_is_rejected_and_the_valid_one_survives(self, monkeypatch, tmp_path):
        attempt = _create_pair(monkeypatch, tmp_path)
        t = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())
        stale = dict(t); stale["binding"] = {"out_dir": "/older-run", "row": 229, "attempt": 0}; stale["event"] = "transition-module_exec"; stale["decline_stage"] = "module_exec"
        (tmp_path / "fallback-transition-module_exec.json").write_text(_json.dumps(stale))
        s = fc.postprocess_attempt(tmp_path, attempt=attempt, child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s["tags"][""]["surviving_transition"] == "transition-lowering" and s["tags"][""]["terminals"] == ["interpreter-ran"]
        assert s["non_pass"] is True and s["rejected_events"][0]["event"] == "transition-module_exec" and s["rejected_events"][0]["code"] == "IDENTITY_MISMATCH"                      # a rejected event is always a non-pass diagnostic


class TestPrecheckPreparation:
    def test_four_completed_cases_with_a_declined_precheck_are_not_an_abandoned_attempt(self, monkeypatch, tmp_path):
        """the reviewer's 119 probe: precheck (typed decline, no execution) + determinism call + four cases completed on the interpreter -> non_pass False."""
        _typed_engine(monkeypatch)
        persist = lambda tag: (lambda ev, rec: (tmp_path / f"{tag}.fallback-{ev}.json").write_text(_json.dumps(rec, indent=1)))
        pre = fc.lower_or_fallback(REVIEW_SRC, boundscheck=True, use_fp64=True, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=persist("precheck"), execution=False)
        fc.lower_or_fallback(REVIEW_SRC, boundscheck=True, use_fp64=True, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, stage="precheck-determinism"), execution=False)
        assert pre.record()["outcome"] == fc.OUTCOME_PRECHECK_DECLINE and pre.record()["preparation_only"] is True and pre.record()["backend_attempt"] == "numba"
        results = []
        for name in ["compare_shapeA_base", "compare_shapeA_empty", "compare_shapeB_base", "compare_shapeB_empty"]:
            out, rec = fc.execute_candidate(REVIEW_SRC, fc.prepared_inputs({"x": np.array([7.0])}), boundscheck=True, use_fp64=True, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, case=name), persist=persist(name))
            assert out["out"].tolist() == [7.0] and rec["backend_terminal"] == "interpreter"; results.append(rec)
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path)}), child_state="COMPLETED", child_exit=0, guard_kill=None)
        assert s["non_pass"] is False and s["execution_tags"] == 4 and s["preparation_only_tags"] == 1 and s["preparation_without_execution"] is False
        assert s["tags"]["precheck."]["status"] == "PREPARATION_RECORDED:precheck-decline"
        assert all(s["tags"][f"{n}."]["status"].startswith("CLOSED_COMPARATOR_DECIDES:interpreter-ran") for n in ["compare_shapeA_base", "compare_shapeA_empty", "compare_shapeB_base", "compare_shapeB_empty"])
        assert not list(tmp_path.glob("*open-transition*"))

    def test_a_precheck_decline_alone_without_any_execution_is_flagged(self, monkeypatch, tmp_path):
        _typed_engine(monkeypatch)
        persist = lambda ev, rec: (tmp_path / f"precheck.fallback-{ev}.json").write_text(_json.dumps(rec))
        fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=persist, execution=False)
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path)}), child_state="EXIT_2", child_exit=2, guard_kill=None)
        assert s["preparation_without_execution"] is True and s["non_pass"] is True

    def test_an_abandoned_execution_attempt_is_still_a_non_pass_next_to_a_precheck(self, monkeypatch, tmp_path):
        _typed_engine(monkeypatch)
        persist = lambda tag: (lambda ev, rec: (tmp_path / f"{tag}.fallback-{ev}.json").write_text(_json.dumps(rec)))
        fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=persist("precheck"), execution=False)
        fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, case="caseA"), persist=persist("caseA"))   # execution transition, never completed
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path)}), child_state="EXIT_1", child_exit=1, guard_kill=None)
        assert s["non_pass"] is True and s["tags"]["caseA."]["status"] == fc.OUTCOME_OPEN_TRANSITION and s["tags"]["precheck."]["non_pass"] is False

    def test_untyped_decline_in_the_precheck_stays_a_visible_non_pass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", True); monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(None, with_hook=False))
        persist = lambda ev, rec: (tmp_path / f"precheck.fallback-{ev}.json").write_text(_json.dumps(rec))
        with pytest.raises(fc.UntypedDecline):
            fc.lower_or_fallback(REVIEW_SRC, engine_pin="pin-r4", binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=persist, execution=False)
        s = fc.postprocess_attempt(tmp_path, attempt=fc.LaunchDescriptor.standalone({"out_dir": str(tmp_path)}), child_state="EXIT_2", child_exit=2, guard_kill=None)
        assert s["non_pass"] is True and s["tags"]["precheck."]["status"].startswith("CLOSED_NON_PASS:untyped-decline")


# =====================================================================================================================
# round 5 -- tests DERIVED FROM PROTOCOL.md (round-5 revision): one per allowed row, one per rejection
# class, the attempt-level rules, and the reviewer's round-4 probes.  Toy source and one-element arrays only.
# =====================================================================================================================

R5_SRC = 'def run_model(inputs):\n    return {"out": inputs["x"].copy()}\n'
R5_SHA = hashlib.sha256(R5_SRC.encode()).hexdigest()
R5_PIN = "pin-r5-protocol"


def _r5_engine(monkeypatch, typed=True, lowered=None):
    monkeypatch.setattr(numba_accel, "HAS_NUMBA", True)
    monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(lowered, typed_reason=("kernel_output_dtype_unestablished: protocol toy" if typed else None)))


def _r5_mint(out, **kw):
    kw.setdefault("group", "C"); kw.setdefault("attempt", 1); kw.setdefault("source_sha256", R5_SHA); kw.setdefault("engine_pin", R5_PIN); kw.setdefault("admission_sha256", "a" * 64)
    return fc.mint_attempt(out, **kw)


def _r5_persist(out, tag=""):
    return lambda ev, rec: (out / (tag + "fallback-" + ev + ".json")).write_text(_json.dumps(rec))


def _r5_emit(out, tag="", case=None, stage=None, complete=True, src=R5_SRC):
    extra = {}
    if case: extra["case"] = case
    if stage: extra["stage"] = stage
    b = fc.attempt_binding(out, **extra)
    if stage:
        return fc.lower_or_fallback(src, engine_pin=R5_PIN, binding=b, persist=_r5_persist(out, tag), execution=False)
    if complete:
        return fc.execute_candidate(src, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=b, persist=_r5_persist(out, tag))
    return fc.lower_or_fallback(src, engine_pin=R5_PIN, binding=b, persist=_r5_persist(out, tag))


def _r5_post(out, launch, killed=False, exit_code=None):
    return fc.postprocess_attempt(out, attempt=launch, child_state="GUARD_KILLED_RSS_CAP" if killed else "COMPLETED", child_exit=(-9 if killed else 0) if exit_code is None else exit_code, guard_kill="RSS_CAP" if killed else None)


def _r5_mutate(out, name, fn):
    p = out / name; rec = _json.loads(p.read_text()); fn(rec); p.write_text(_json.dumps(rec)); return rec


class TestProtocolAllowedRows:
    """One test per row of the event/state table (PROTOCOL.md section 3): the record produced by the real emission path
    is VALID and the closure rule of section 5 applies."""

    def test_row_transition_lowering_parent_and_case(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path)
        _r5_emit(tmp_path, complete=False); _r5_emit(tmp_path, tag="caseA.", case="caseA", complete=False)
        s = _r5_post(tmp_path, launch)
        for tag in ("", "caseA."):
            assert s["tags"][tag]["transitions"] == ["transition-lowering"] and s["tags"][tag]["status"] == fc.OUTCOME_OPEN_TRANSITION and s["tags"][tag]["rejected_events"] == []
        assert s["tags"][""]["scope"] == "parent" and s["tags"]["caseA."]["scope"] == "case" and s["non_pass"] is True

    def test_row_transition_module_exec(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, lowered=LOWERED_RAISES_TYPED_AT_IMPORT); launch = _r5_mint(tmp_path)
        prog = fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path), persist=_r5_persist(tmp_path)); prog.exec_module(prog.make_namespace(), "<lowered>")
        s = _r5_post(tmp_path, launch, killed=True)
        assert s["tags"][""]["surviving_transition"] == "transition-module_exec" and s["tags"][""]["status"] == fc.OUTCOME_RESOURCE_STOP and (tmp_path / "fallback-resource-stop.json").exists()

    def test_row_transition_run_model(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, lowered=LOWERED_RAISES_TYPED_IN_RUN_MODEL); launch = _r5_mint(tmp_path)
        out, rec = fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path), persist=_r5_persist(tmp_path))
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["surviving_transition"] == "transition-run_model" and s["tags"][""]["status"] == "CLOSED_COMPARATOR_DECIDES:interpreter-ran" and s["non_pass"] is False

    def test_row_interpreter_ran(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path); _r5_emit(tmp_path)
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["terminals"] == ["interpreter-ran"] and s["tags"][""]["status"] == "CLOSED_COMPARATOR_DECIDES:interpreter-ran" and s["non_pass"] is False and s["verdict"] is None

    def test_row_fast_ran(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path); _r5_emit(tmp_path)
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["terminals"] == ["fast-ran"] and s["tags"][""]["transitions"] == [] and s["tags"][""]["status"] == "CLOSED_COMPARATOR_DECIDES:fast-ran" and s["non_pass"] is False

    def test_row_interpreter_failed(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path, source_sha256=hashlib.sha256(ECHO_DSL_RAISES.encode()).hexdigest())
        with pytest.raises(fc.InterpreterFailed):
            _r5_emit(tmp_path, src=ECHO_DSL_RAISES)
        s = _r5_post(tmp_path, launch, exit_code=1)
        assert s["tags"][""]["terminals"] == ["interpreter-failed"] and s["tags"][""]["status"] == "CLOSED_NON_PASS:interpreter-failed" and s["non_pass"] is True

    def test_row_untyped_decline_parent_case_and_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", True); monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(None, with_hook=False)); launch = _r5_mint(tmp_path)
        for tag, extra in (("", {}), ("caseA.", {"case": "caseA"}), ("precheck.", {"stage": "precheck"})):
            with pytest.raises(fc.UntypedDecline):
                fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, **extra), persist=_r5_persist(tmp_path, tag), execution=not extra.get("stage"))
        s = _r5_post(tmp_path, launch, exit_code=2)
        assert all(s["tags"][t]["status"] == "CLOSED_NON_PASS:untyped-decline" for t in ("", "caseA.", "precheck.")) and s["non_pass"] is True

    def test_row_numba_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", False); launch = _r5_mint(tmp_path)
        with pytest.raises(fc.NumbaUnavailable):
            fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path), persist=_r5_persist(tmp_path))
        s = _r5_post(tmp_path, launch, exit_code=2)
        assert s["tags"][""]["status"] == "CLOSED_NON_PASS:numba-unavailable" and s["non_pass"] is True

    def test_row_precheck_decline_stage_only(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path, stages=["precheck"], cases=["caseA"])
        _r5_emit(tmp_path, tag="precheck.", stage="precheck"); _r5_emit(tmp_path, tag="caseA.", case="caseA")
        s = _r5_post(tmp_path, launch)
        assert s["tags"]["precheck."]["status"] == "PREPARATION_RECORDED:precheck-decline" and s["tags"]["precheck."]["scope"] == "stage" and s["non_pass"] is False and s["missing_cases"] == []


class TestProtocolRejectionClasses:
    """One test per rejection diagnostic (PROTOCOL.md section 4); every rejected record is named and never closes."""

    def _pair(self, monkeypatch, tmp_path, **mint):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path, **mint); _r5_emit(tmp_path); return launch

    def _expect(self, s, code, event="interpreter-ran", status_prefix=fc.OUTCOME_OPEN_TRANSITION):
        assert s["non_pass"] is True, s["tags"]
        hits = [r for r in s["rejected_events"] if r["event"] == event]
        assert hits and hits[0]["code"] == code, (code, s["rejected_events"])
        assert s["tags"][""]["status"].startswith(status_prefix), s["tags"][""]["status"]

    def test_UNREADABLE(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / "fallback-interpreter-ran.json").write_text('{"interrupted":'); self._expect(_r5_post(tmp_path, l), "UNREADABLE")

    def test_NOT_AN_OBJECT(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / "fallback-interpreter-ran.json").write_text('[1, 2, 3]'); self._expect(_r5_post(tmp_path, l), "UNREADABLE")

    @pytest.mark.parametrize("field,value", [("binding", ["wrong-type"]), ("engine", ["wrong-type"]), ("source_sha256", 42), ("event_utc", 12), ("decline_typed", "yes"), ("backend_attempt", 3)])
    def test_SCHEMA_type_errors_never_raise(self, monkeypatch, tmp_path, field, value):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, value))
        s = _r5_post(tmp_path, l)
        code = "TIME_INVALID" if field == "event_utc" else "SCHEMA"; self._expect(s, code)

    def test_CONTRACT_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("contract", "lite-fallback-contract-v4")); self._expect(_r5_post(tmp_path, l), "CONTRACT_MISMATCH")

    def test_UNKNOWN_EVENT(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / "fallback-bogus.json").write_bytes((tmp_path / "fallback-interpreter-ran.json").read_bytes())
        s = _r5_post(tmp_path, l); assert [r["code"] for r in s["rejected_events"] if r["event"] == "bogus"] == ["UNKNOWN_EVENT"] and s["non_pass"] is True

    def test_EVENT_FILENAME_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / "fallback-fast-ran.json").write_bytes((tmp_path / "fallback-interpreter-ran.json").read_bytes()); (tmp_path / "fallback-interpreter-ran.json").unlink()
        self._expect(_r5_post(tmp_path, l), "EVENT_FILENAME_MISMATCH", event="fast-ran")

    @pytest.mark.parametrize("field", ["backend_attempt", "backend_terminal"])
    def test_STATE_NOT_ALLOWED_backend(self, monkeypatch, tmp_path, field):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, "numba")); self._expect(_r5_post(tmp_path, l), "STATE_NOT_ALLOWED")

    def test_STATE_NOT_ALLOWED_outcome_and_decline(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("decline_reason", None)); self._expect(_r5_post(tmp_path, l), "STATE_NOT_ALLOWED")

    def test_SCOPE_NOT_ALLOWED_transition_under_stage(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path)
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=_r5_persist(tmp_path, "precheck."))   # an EXECUTION transition under a stage tag
        s = _r5_post(tmp_path, l); assert s["rejected_events"][0]["code"] == "SCOPE_NOT_ALLOWED" and s["tags"]["precheck."]["status"].startswith("NO_VALID_EVENT") and s["non_pass"] is True

    def test_SUBATTEMPT_MISMATCH_cross_case_copy(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path)
        _r5_emit(tmp_path, tag="caseB-", case="caseB", complete=False); _r5_emit(tmp_path, tag="caseA-", case="caseA")
        (tmp_path / "caseB-fallback-interpreter-ran.json").write_bytes((tmp_path / "caseA-fallback-interpreter-ran.json").read_bytes())
        s = _r5_post(tmp_path, l)
        assert s["tags"]["caseB-"]["status"].startswith(fc.OUTCOME_OPEN_TRANSITION + "+REJECTED:interpreter-ran:SUBATTEMPT_MISMATCH") and s["tags"]["caseA-"]["status"].startswith("CLOSED_COMPARATOR_DECIDES") and s["non_pass"] is True
        assert (tmp_path / "caseB-fallback-open-transition.json").exists()

    def test_SUBATTEMPT_MISMATCH_parent_record_with_case(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r["binding"].__setitem__("case", "caseA")); self._expect(_r5_post(tmp_path, l), "SUBATTEMPT_MISMATCH")

    def test_SUBATTEMPT_MISMATCH_undeclared_case(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path, cases=["caseA"]); _r5_emit(tmp_path, tag="caseZ.", case="caseZ")
        s = _r5_post(tmp_path, l); assert s["rejected_events"][0]["code"] == "SUBATTEMPT_MISMATCH" and "caseA" in s["missing_cases"] and s["non_pass"] is True

    def test_IDENTITY_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r["binding"].__setitem__("attempt", 9)); self._expect(_r5_post(tmp_path, l), "IDENTITY_MISMATCH")

    def test_SOURCE_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("source_sha256", "b" * 64)); self._expect(_r5_post(tmp_path, l), "SOURCE_MISMATCH")

    def test_ENGINE_PIN_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("engine_pin", "other-pin")); self._expect(_r5_post(tmp_path, l), "ENGINE_PIN_MISMATCH")

    def test_ENGINE_HASH_MISMATCH(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("numba_accel_sha256", "b" * 64)); self._expect(_r5_post(tmp_path, l), "ENGINE_HASH_MISMATCH")

    def test_TIME_INVALID(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("event_utc", "not-a-time")); self._expect(_r5_post(tmp_path, l), "TIME_INVALID")

    def test_TIME_BEFORE_MINT(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("event_utc", "2000-01-01T00:00:00+00:00")); self._expect(_r5_post(tmp_path, l), "TIME_BEFORE_MINT")

    def test_TIME_BEFORE_TRANSITION(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path)
        t = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())["event_utc"]
        earlier = (datetime_mod.datetime.fromisoformat(t) - datetime_mod.timedelta(microseconds=1)).isoformat()
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("event_utc", earlier)); self._expect(_r5_post(tmp_path, l), "TIME_BEFORE_TRANSITION")

    def test_ORPHAN_TERMINAL(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / "fallback-transition-lowering.json").unlink()
        s = _r5_post(tmp_path, l); assert s["tags"][""]["state"] == "STATE_CONFLICT" and s["tags"][""]["conflicting_events"] == ["interpreter-ran"] and s["non_pass"] is True   # v6: an orphan closing record is a set-level conflict

    def test_STATE_CONFLICT_fast_ran_with_transition(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path)
        r = _json.loads((tmp_path / "fallback-interpreter-ran.json").read_text()); r.update(event="fast-ran", outcome=fc.OUTCOME_FAST_RAN, backend="numba", backend_attempt="numba", backend_terminal="numba", decline_reason=None, decline_typed=False, decline_stage=None)
        inconsistent = _json.loads(_json.dumps(r))                                             # v3: a fast-ran forged from an interpreter record violates I3/I4/I6/I7/I8 -> FIELD_INCONSISTENT, joined record kept
        r.update(fallback_taken=False, executed_program="numba_lowered", lowered_sha256="c" * 64, program_text_sha256="c" * 64, history=[], events=[{"event": "fast-ran", "event_utc": r["event_utc"]}])   # a CONSISTENT fast-ran row -> the SET conflicts
        (tmp_path / "fallback-fast-ran.json").write_text(_json.dumps(r)); (tmp_path / "fallback-interpreter-ran.json").unlink()
        s = _r5_post(tmp_path, l); assert s["tags"][""]["state"] == "STATE_CONFLICT" and "fast-ran" in s["tags"][""]["conflicting_events"] and s["tags"][""]["status"].startswith(fc.OUTCOME_OPEN_TRANSITION + "+STATE_CONFLICT") and s["non_pass"] is True and (tmp_path / "fallback-open-transition.json").exists()
        (tmp_path / "fallback-fast-ran.json").write_text(_json.dumps(inconsistent)); (tmp_path / "fallback-open-transition.json").unlink()
        s = _r5_post(tmp_path, l); assert s["tags"][""]["state"] == "OPEN" and [x["code"] for x in s["tags"][""]["rejected_events"]] == ["TERMINAL_INCONSISTENT"] and s["non_pass"] is True and (tmp_path / "fallback-open-transition.json").exists()   # v4: fast-ran is a re-emit row

    def test_MINT_ALTERED_attempt_1_to_9(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path)
        m = _json.loads((tmp_path / fc.MINT_FILE).read_text()); m["attempt"] = 9; (tmp_path / fc.MINT_FILE).write_text(_json.dumps(m))
        s = _r5_post(tmp_path, l); assert s["mint_check"]["state"] == "MINT_ALTERED" and s["non_pass"] is True and s["attempt"]["attempt"] == 1       # identity stays the parent's

    def test_MINT_MISSING(self, monkeypatch, tmp_path):
        l = self._pair(monkeypatch, tmp_path); (tmp_path / fc.MINT_FILE).unlink()
        s = _r5_post(tmp_path, l); assert s["mint_check"]["state"] == "MINT_MISSING" and s["non_pass"] is True

    def test_MISSING_CASE(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path, cases=["caseA", "caseB"]); _r5_emit(tmp_path, tag="caseA.", case="caseA")
        s = _r5_post(tmp_path, l); assert s["missing_cases"] == ["caseB"] and s["non_pass"] is True

    def test_dict_attempt_is_refused(self, monkeypatch, tmp_path):
        with pytest.raises(fc.RecordInvalid):
            fc.postprocess_attempt(tmp_path, attempt={"out_dir": str(tmp_path)}, child_state="COMPLETED", child_exit=0, guard_kill=None)


class TestProtocolReview6Round4Probes:
    def test_retained_descriptor_beats_child_replaced_mint(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path)
        m = _json.loads((tmp_path / fc.MINT_FILE).read_text()); m["attempt"] = 9; (tmp_path / fc.MINT_FILE).write_text(_json.dumps(m))
        _r5_emit(tmp_path)                                                 # the child now binds attempt 9 from the altered mint
        s = _r5_post(tmp_path, l)
        assert s["non_pass"] is True and s["mint_check"]["state"] == "MINT_ALTERED" and any(r["code"] == "IDENTITY_MISMATCH" for r in s["rejected_events"])

    def test_retained_source_anchor_beats_altered_disk_anchor(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path); _r5_emit(tmp_path)
        m = _json.loads((tmp_path / fc.MINT_FILE).read_text()); m["source_sha256"] = "b" * 64; (tmp_path / fc.MINT_FILE).write_text(_json.dumps(m))
        for p in tmp_path.glob("fallback-*.json"):
            if p.name == fc.MINT_FILE: continue
            r = _json.loads(p.read_text()); r["source_sha256"] = "b" * 64; p.write_text(_json.dumps(r))
        s = _r5_post(tmp_path, l)
        assert s["non_pass"] is True and all(r["code"] == "SOURCE_MISMATCH" for r in s["rejected_events"]) and s["mint_check"]["state"] == "MINT_ALTERED"

    @pytest.mark.parametrize("field,value", [("binding", ["wrong-type"]), ("engine", ["wrong-type"]), ("source_sha256", 42)])
    def test_invalid_field_types_during_kill_keep_the_joined_stop(self, monkeypatch, tmp_path, field, value):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path); _r5_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, value))
        s = _r5_post(tmp_path, l, killed=True)
        assert s["non_pass"] is True and (tmp_path / "fallback-resource-stop.json").exists() and s["tags"][""]["status"].startswith(fc.OUTCOME_RESOURCE_STOP + "+REJECTED:interpreter-ran:SCHEMA")

    def test_case_identity_cannot_close_a_different_case(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path)
        _r5_emit(tmp_path, tag="caseB-", case="caseB", complete=False); _r5_emit(tmp_path, tag="caseA-", case="caseA")
        (tmp_path / "caseB-fallback-interpreter-ran.json").write_bytes((tmp_path / "caseA-fallback-interpreter-ran.json").read_bytes())
        s = _r5_post(tmp_path, l); assert s["non_pass"] is True and s["tags"]["caseB-"]["status"].startswith(fc.OUTCOME_OPEN_TRANSITION)

    @pytest.mark.parametrize("field", ["backend_attempt", "backend_terminal"])
    def test_interpreter_event_requires_interpreter_backend(self, monkeypatch, tmp_path, field):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path); _r5_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, "numba"))
        s = _r5_post(tmp_path, l); assert s["non_pass"] is True and s["tags"][""]["status"].startswith(fc.OUTCOME_OPEN_TRANSITION + "+REJECTED:interpreter-ran:STATE_NOT_ALLOWED")

    def test_valid_minted_fast_terminal_control(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        l = _r5_mint(tmp_path); _r5_emit(tmp_path); s = _r5_post(tmp_path, l); assert s["non_pass"] is False

    def test_minted_transition_partial_terminal_control(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); l = _r5_mint(tmp_path); _r5_emit(tmp_path, complete=False); (tmp_path / "fallback-interpreter-ran.json").write_text('{"incomplete":')
        s = _r5_post(tmp_path, l, killed=True); assert s["non_pass"] is True and (tmp_path / "fallback-resource-stop.json").exists()

    def test_validate_record_safe_never_raises(self):
        for bad in (None, [], 42, "x", {"contract": None}, {"contract": fc.CONTRACT, "event": 1}, {k: None for k in fc.REQUIRED_RECORD_FIELDS}, {**{k: None for k in fc.REQUIRED_RECORD_FIELDS}, "contract": fc.CONTRACT, "event": "x", "event_utc": "2026-01-01T00:00:00+00:00", "engine": [1]}):
            ok, reasons = fc.validate_record_safe(bad); assert ok is False and reasons and isinstance(reasons[0], str)


# =====================================================================================================================
# round 6 -- tests derived from PROTOCOL v2 (round6/): state-machine COMBINATIONS per scope, schema fuzz generated from
# RECORD_SCHEMA, typed case/stage coverage, and the reviewer's round-5 probes.  Toy source, one-element arrays, stub kills.
# =====================================================================================================================

import itertools as _it
import pathlib as _pathlib2

PROTOCOL_JSON = _pathlib2.Path(fc.__file__).resolve().parent / "fallback_protocol" / "PROTOCOL.json"   # the TRACKED canonical spec (P2-R6-SPEC-PORTABILITY-1)
_P7 = _json.loads(PROTOCOL_JSON.read_text())
_CR_FIELDS = _P7["invariants"]["cross_record_fields_equal"]          # the independent oracle reads the spec, never the shim


def _oracle_consistent(T, S):
    """PROTOCOL v3 C1-C7 restated from the tracked JSON: a closing record T is consistent with the surviving transition S iff
    the listed fields are equal, S's event trail is a prefix of T's and T is not earlier than S."""
    return all(T[k] == S[k] for k in _CR_FIELDS) and T["events"][:len(S["events"])] == S["events"] and T["event_utc"] >= S["event_utc"]


def _r6_fast_failed(monkeypatch, out, event, tag="", align_hook=False, **extra):
    """Emit a REAL untyped-decline / numba-unavailable record (the review fixture shape)."""
    if event == "numba-unavailable":
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", False)
    else:
        monkeypatch.setattr(numba_accel, "HAS_NUMBA", True); monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(None, with_hook=True))   # v5: ONE engine per attempt -- the hook exists, the lowerer returns None without a reason (UNTYPED)
    with pytest.raises(fc.FastStageFailure):
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(out, **extra), persist=_r5_persist(out, tag), execution=not extra.get("stage"))
    if event == "numba-unavailable" and align_hook:   # v5 combination fixtures: the row's hook flag is aligned to the attempt's engine so the SET machine is exercised; a differing flag is a CONTEXT_MISMATCH tested separately
        _r5_mutate(out, tag + "fallback-" + event + ".json", lambda r: r.__setitem__("decline_hook_available", True))
    return _json.loads((out / (tag + "fallback-" + event + ".json")).read_text())


def _r6_make(monkeypatch, out, combo, tag="", killed=False, **extra):
    """Produce the records of `combo` (event names) in scope `tag` using real emission where possible, synthesising the
    rest from a real record; returns the launch descriptor."""
    launch = _r5_mint(out, cases=[extra["case"]] if extra.get("case") else None, stages=[extra["stage"]] if extra.get("stage") else None)
    _r5_engine(monkeypatch)
    base = None
    if any(e.startswith("transition-") for e in combo) or "interpreter-ran" in combo or "interpreter-failed" in combo:
        prog = fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(out, **extra), persist=_r5_persist(out, tag))   # real transition-lowering
        base = _json.loads((out / (tag + "fallback-transition-lowering.json")).read_text())
        if "transition-lowering" not in combo: (out / (tag + "fallback-transition-lowering.json")).unlink()
        for t in ("transition-module_exec", "transition-run_model"):
            if t in combo:   # a CONSISTENT synthetic later-stage transition (I7: lowered set; I8: history stage; I9: own trail)
                r = _json.loads(_json.dumps(base)); st = t[len("transition-"):]; now = fc._utc()
                r.update(event=t, decline_stage=st, event_utc=now, lowered_sha256="c" * 64); r["history"][-1]["stage"] = st; r["events"] = [{"event": t, "event_utc": now}]
                (out / (tag + "fallback-" + t + ".json")).write_text(_json.dumps(r))
        if "interpreter-ran" in combo:
            ns = prog.exec_module(prog.make_namespace(), "<dsl>"); prog.run_model(ns, lambda: {"x": np.array([7.0])})     # real interpreter-ran
        if "interpreter-failed" in combo:   # a CONSISTENT synthetic failure closing the REAL transition-lowering (I5, I9, C1-C6)
            r = _json.loads(_json.dumps(base)); now = fc._utc()
            r.update(event="interpreter-failed", event_utc=now, outcome="interpreter_failed:ValueError: synthetic", interpreter_failure={"type": "ValueError", "message": "synthetic", "prior_decline_reason": base["decline_reason"], "prior_decline_stage": base["decline_stage"]})
            r["events"] = base["events"] + [{"event": "interpreter-failed", "event_utc": now}]
            (out / (tag + "fallback-interpreter-failed.json")).write_text(_json.dumps(r))
    if "fast-ran" in combo:
        monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(out, **extra), persist=_r5_persist(out, tag))
    for e in ("untyped-decline", "numba-unavailable"):
        if e in combo: _r6_fast_failed(monkeypatch, out, e, tag, align_hook=(len(combo) > 1), **extra)
    if "precheck-decline" in combo:
        _r5_engine(monkeypatch); fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(out, **extra), persist=_r5_persist(out, tag), execution=False)
    return launch


EXEC_EVENTS = ["transition-lowering", "transition-module_exec", "transition-run_model", "interpreter-ran", "interpreter-failed", "fast-ran", "untyped-decline", "numba-unavailable"]
EXEC_SINGLES = {("transition-lowering",): "OPEN", ("transition-module_exec",): "OPEN", ("transition-run_model",): "OPEN", ("fast-ran",): "FAST_DONE", ("untyped-decline",): "FAST_FAILED", ("numba-unavailable",): "FAST_FAILED",
                ("interpreter-ran",): "STATE_CONFLICT", ("interpreter-failed",): "STATE_CONFLICT"}   # orphans
EXEC_PAIRS_OK = {("interpreter-ran", "transition-lowering"): "CLOSED_OK", ("interpreter-failed", "transition-lowering"): "CLOSED_FAIL"}


def _expected_exec(out, tag, combo):
    """PROTOCOL v3 sections 3-4 restated INDEPENDENTLY of the shim, from the records on disk: a closing record inconsistent
    with the surviving (latest) transition is TERMINAL_INCONSISTENT and leaves the set; the remaining set maps to one state.
    'any transition + any interpreter-ran = CLOSED_OK' is NOT the definition (the review P2-R6-CLOSING-CONSISTENCY-1)."""
    recs = {e: _json.loads((out / (tag + "fallback-" + e + ".json")).read_text()) for e in combo}
    trans = [e for e in combo if e.startswith("transition-")]
    surviving = max(trans, key=lambda e: recs[e]["event_utc"]) if trans else None
    inconsistent = [e for e in combo if e in ("interpreter-ran", "interpreter-failed") and surviving is not None and not _oracle_consistent(recs[e], recs[surviving])]
    live = sorted(e for e in combo if e not in inconsistent)
    if not live: state = "NO_VALID_EVENT"
    elif live == ["fast-ran"]: state = "FAST_DONE"
    elif len(live) == 1 and live[0] in ("untyped-decline", "numba-unavailable"): state = "FAST_FAILED"
    elif len(trans) == 1 and live == sorted(trans): state = "OPEN"
    elif len(trans) == 1 and live == sorted(trans + ["interpreter-ran"]): state = "CLOSED_OK"
    elif len(trans) == 1 and live == sorted(trans + ["interpreter-failed"]): state = "CLOSED_FAIL"
    else: state = "STATE_CONFLICT"
    conflicting = [e for e in live if e != surviving] if state == "STATE_CONFLICT" else []
    return {"state": state, "surviving": surviving, "inconsistent": inconsistent, "conflicting": conflicting}


class TestProtocolV2StateMachineCombinations:
    @pytest.mark.parametrize("combo", [c for n in (1, 2) for c in _it.combinations(EXEC_EVENTS, n)] + [("transition-lowering", "interpreter-ran", "untyped-decline"), ("transition-lowering", "interpreter-ran", "interpreter-failed"), ("transition-lowering", "transition-run_model", "interpreter-ran"), ("fast-ran", "untyped-decline", "numba-unavailable")])
    @pytest.mark.parametrize("killed", [False, True])
    def test_execution_scope_combination(self, monkeypatch, tmp_path, combo, killed):
        launch = _r6_make(monkeypatch, tmp_path, combo, killed=killed)
        exp = _expected_exec(tmp_path, "", combo)                      # from the on-disk records, independent of the shim
        s = _r5_post(tmp_path, launch, killed=killed)
        info = s["tags"][""]
        assert info["state"] == exp["state"], (combo, exp, info)
        expected_rejected = sorted([(e, "TERMINAL_INCONSISTENT") for e in exp["inconsistent"]] + [(e, "STATE_CONFLICT") for e in exp["conflicting"]])
        assert sorted((r["event"], r["code"]) for r in info["rejected_events"]) == expected_rejected, (combo, exp, info["rejected_events"])
        assert info["closing_event"] == ({"CLOSED_OK": "interpreter-ran", "CLOSED_FAIL": "interpreter-failed", "FAST_DONE": "fast-ran"}.get(exp["state"]) if exp["state"] in ("CLOSED_OK", "CLOSED_FAIL", "FAST_DONE", "FAST_FAILED") else None) or exp["state"] == "FAST_FAILED", (combo, info)
        joined = (tmp_path / "fallback-resource-stop.json").exists() if killed else (tmp_path / "fallback-open-transition.json").exists()
        assert joined is (exp["surviving"] is not None and exp["state"] in ("OPEN", "STATE_CONFLICT")), (combo, "the joined record exists iff a valid transition survives unclosed")
        kill_rec = (tmp_path / "fallback-kill-record.json").exists()
        assert kill_rec is (killed and exp["surviving"] is None and exp["state"] not in ("FAST_DONE", "CLOSED_OK", "CLOSED_FAIL")), (combo, "unjoined kill record iff kill + no surviving transition + no consistent completion")
        assert s["non_pass"] is (killed or exp["state"] not in ("FAST_DONE", "CLOSED_OK") or bool(expected_rejected)), (combo, s["non_pass"])
        if exp["state"] == "STATE_CONFLICT":
            assert "STATE_CONFLICT" in info["status"] and info["conflicting_events"] and info["closing_event"] is None   # no first-match success

    @pytest.mark.parametrize("combo", [("precheck-decline",), ("untyped-decline",), ("numba-unavailable",), ("precheck-decline", "untyped-decline"), ("precheck-decline", "numba-unavailable"), ("untyped-decline", "numba-unavailable")])
    def test_stage_scope_combination(self, monkeypatch, tmp_path, combo):
        launch = _r6_make(monkeypatch, tmp_path, combo, tag="precheck.", stage="precheck")
        s = _r5_post(tmp_path, launch, exit_code=0 if combo == ("precheck-decline",) else 2)
        info = s["tags"]["precheck."]
        exp = "PREPARATION_RECORDED" if combo == ("precheck-decline",) else ("FAST_FAILED" if len(combo) == 1 else "STATE_CONFLICT")
        assert info["state"] == exp and info["scope"] == "stage"
        if exp == "STATE_CONFLICT":   # amendment 1: no transition in a stage scope -> every record of the conflicting set is REJECTED with code STATE_CONFLICT
            assert sorted(r["event"] for r in info["rejected_events"]) == sorted(combo) == sorted(info["conflicting_events"]) and all(r["code"] == "STATE_CONFLICT" for r in info["rejected_events"]) and info["valid_events"] == [], info
        else:
            assert info["rejected_events"] == []
        assert info["non_pass"] is (exp != "PREPARATION_RECORDED")
        assert s["non_pass"] is True and s["preparation_without_execution"] is (exp == "PREPARATION_RECORDED")   # a stage-only attempt never passes: no execution scope exists

    @pytest.mark.parametrize("event", ["transition-lowering", "interpreter-ran", "fast-ran"])
    def test_execution_rows_under_a_stage_tag_are_scope_rejected(self, monkeypatch, tmp_path, event):
        launch = _r5_mint(tmp_path, stages=["precheck"]); _r5_engine(monkeypatch)
        if event == "fast-ran":
            monkeypatch.setattr(numba_accel, "numba_lower", _stub_lowerer(R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
            fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=_r5_persist(tmp_path, "precheck."))
        else:
            prog = fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, stage="precheck"), persist=_r5_persist(tmp_path, "precheck."))
            if event == "interpreter-ran":
                ns = prog.exec_module(prog.make_namespace(), "<dsl>"); prog.run_model(ns, lambda: {"x": np.array([7.0])})
        s = _r5_post(tmp_path, launch)
        assert all(r["code"] == "SCOPE_NOT_ALLOWED" for r in s["rejected_events"]) and s["tags"]["precheck."]["state"] == "NO_VALID_EVENT" and s["non_pass"] is True

    def test_forged_fast_failure_row_plus_real_sigkill_keeps_the_joined_stop(self, tmp_path, solar_root):
        """the reviewer's forged-row probe: the child persists a real typed transition, a forged allowed untyped-decline row, enters the
        interpreter and is SIGKILLed -> STATE_CONFLICT, joined RESOURCE_STOP written."""
        out, rc, guard = _run_child(tmp_path, "kill", solar_root, kill=True)
        t = _json.loads((out / "fallback-transition-lowering.json").read_text())
        forged = dict(t); forged.update(event="untyped-decline", outcome=fc.OUTCOME_UNTYPED_DECLINE, backend="numba", backend_attempt="numba", backend_terminal=None, decline_typed=False, decline_reason="UNTYPED: forged by the child", history=[], events=[{"event": "untyped-decline", "event_utc": t["event_utc"]}])
        sloppy = _json.loads(_json.dumps(forged))                                        # the reviewer's forgery as copied: fallback_taken/executed_program still say interpreter -> FIELD_INCONSISTENT (I3/I4)
        forged.update(fallback_taken=False, executed_program="numba_lowered")            # a field-CONSISTENT forgery -> the SET conflicts (STATE_CONFLICT)
        (out / "fallback-untyped-decline.json").write_text(_json.dumps(forged))
        desc = fc.LaunchDescriptor.standalone({"out_dir": str(out), "row": 999, "attempt": 1})
        s = fc.postprocess_attempt(out, attempt=desc, child_state="GUARD_KILLED_RSS_CAP", child_exit=rc, guard_kill="RSS_CAP")
        info = s["tags"][""]
        assert rc == -9 and info["state"] == "STATE_CONFLICT" and "untyped-decline" in info["conflicting_events"] and s["non_pass"] is True
        assert (out / "fallback-resource-stop.json").exists() and info["status"].startswith("RESOURCE_STOP+STATE_CONFLICT:untyped-decline")
        rs = _json.loads((out / "fallback-resource-stop.json").read_text()); assert rs["conflicting_events"] == ["untyped-decline"] and rs["decline_typed"] is True
        (out / "fallback-untyped-decline.json").write_text(_json.dumps(sloppy)); (out / "fallback-resource-stop.json").unlink()
        s = fc.postprocess_attempt(out, attempt=desc, child_state="GUARD_KILLED_RSS_CAP", child_exit=rc, guard_kill="RSS_CAP"); info = s["tags"][""]
        assert info["state"] == "OPEN" and [x["code"] for x in info["rejected_events"]] == ["FIELD_INCONSISTENT"] and (out / "fallback-resource-stop.json").exists() and not (out / "fallback-kill-record.json").exists()   # either way the real SIGKILL keeps its joined stop


_WRONG_BY_TYPE = {   # INDEPENDENT expectation table (P3-R6-NULLABLE-REQUIRED-1): never derived from the validator under test
    "const": [42, "x", [1], {"a": 1}, True, "a" * 64], "str": [42, [1], {"a": 1}, True, 1.5], "utc": ["x", 42, "2026-13-01", True, [1]],
    "sha256": ["x", "a" * 63, "A" * 64, 42, True], "bool": [42, "true", "x", [1], 1.5], "int": [True, "1", 1.5, [1], "x"],
    "enum": ["zzz", 42, [1], {"a": 1}, 3.5], "object": [42, "x", [1], True], "list": [42, "x", {"a": 1}, True]}


def _null_allowed(spec):
    return bool(spec.get("optional") or spec.get("nullable") or (spec["type"] == "enum" and None in spec["values"]) or (spec["type"] == "const" and spec["value"] is None))


def _wrong_values_for(spec):
    vals = [v for v in _WRONG_BY_TYPE[spec["type"]] if not (spec["type"] == "const" and v == spec["value"]) and not (spec["type"] == "enum" and v in spec["values"])]
    if spec["type"] == "str" and spec.get("nonempty"): vals.append("")
    if not _null_allowed(spec): vals.append(None)
    return vals


def _schema_paths(schema, prefix=""):
    """Every field path incl. nested object fields AND list-item fields (`history[0].stage`, `events[0].event_utc`)."""
    for k, spec in schema.items():
        yield prefix + k, spec
        if spec["type"] == "object" and spec.get("fields") and not spec.get("open"):
            yield from _schema_paths(spec["fields"], prefix + k + ".")
        if spec["type"] == "list" and spec["items"]["type"] == "object":
            yield from _schema_paths(spec["items"]["fields"], prefix + k + "[0].")


def _walk(rec, path):
    import re as _re
    parts = [p for p in _re.split(r"\.|(\[\d+\])", path) if p]
    o = rec
    for p in parts[:-1]:
        o = o[int(p[1:-1])] if p.startswith("[") else o[p]
    last = parts[-1]
    return o, (int(last[1:-1]) if last.startswith("[") else last)


def _set_path(rec, path, value):
    o, k = _walk(rec, path); o[k] = value


def _del_path(rec, path):
    o, k = _walk(rec, path); del o[k]


class TestProtocolV2SchemaFuzz:
    def test_shim_schema_is_the_protocol_schema(self):
        """P2-R6-SPEC-PORTABILITY-1: the canonical spec is TRACKED beside the package and resolved relative to it."""
        tracked = fc.PROTOCOL_DIR / "PROTOCOL.json"                      # resolved relative to the package only (no module-level name, no author-scratch path)
        assert tracked.is_file() and (fc.PROTOCOL_DIR / "PROTOCOL.md").is_file() and fc.PROTOCOL_DIR.parent.name == "einsum_lib"
        P = _json.loads(tracked.read_text())
        assert _json.loads(_json.dumps(fc.RECORD_SCHEMA)) == P["type_schema"]["record"]
        assert _json.loads(_json.dumps(fc.INVARIANTS)) == {"cross_field": P["invariants"]["cross_field"], "cross_record": P["invariants"]["cross_record"]}
        assert _json.loads(_json.dumps(fc.REEMIT)) == {k: P["reemit_and_compare"][k] for k in ("reemit_events", "deterministic_fields", "free_fields")}   # v4: RE-EMIT AND COMPARE tables
        assert _json.loads(_json.dumps(fc.CONTEXT)) == {k: P["context_resolution"][k] for k in ("fields", "precedence", "codes")}                      # v5: context resolution
        assert list(fc.ANCHOR_DOMAIN) == P["context_resolution"]["anchor_domain"]["allowed_expected_engine_keys"]                                       # v6: anchor domain
        assert P["contract"] == fc.CONTRACT and P["spec_version"] == "v6"

    @pytest.mark.parametrize("path", [p for p, _ in _schema_paths(fc.RECORD_SCHEMA)])
    def test_every_field_presence_null_and_wrong_types(self, monkeypatch, tmp_path, path):
        """INDEPENDENT oracle (P3-R6-NULLABLE-REQUIRED-1): deletion rejected unless optional (reason = SCHEMA:<path>); null rejected
        unless nullable / enum-with-null / const-null / optional; wrong values from the fixed table rejected; the valid record passes."""
        spec = dict(_schema_paths(fc.RECORD_SCHEMA))[path]
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        good = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())
        if path.startswith("interpreter_failure.") and good["interpreter_failure"] is None:
            good["interpreter_failure"] = {"type": "ValueError", "message": "m", "prior_decline_reason": None, "prior_decline_stage": None}
        assert fc.validate_record_safe(good) == (True, [])
        # presence: a required key must exist even when nullable; an optional key may be absent
        o, k = _walk(good, path)
        if k in o:
            rec = _json.loads(_json.dumps(good)); _del_path(rec, path)
            ok, reasons = fc.validate_record_safe(rec)
            if spec.get("optional"): assert ok, (path, reasons)
            else: assert ok is False and reasons == ["SCHEMA:" + path], (path, reasons)
        else:
            assert spec.get("optional"), (path, "absent in a real record but not optional")
        # null
        rec = _json.loads(_json.dumps(good)); _set_path(rec, path, None)
        ok, reasons = fc.validate_record_safe(rec)
        assert ok is _null_allowed(spec), (path, spec, reasons)
        # wrong values
        for bad in _wrong_values_for(spec):
            rec = _json.loads(_json.dumps(good)); _set_path(rec, path, bad)
            ok, reasons = fc.validate_record_safe(rec)
            assert ok is False and reasons and (reasons[0].startswith("SCHEMA") or reasons[0] in ("CONTRACT_MISMATCH", "TIME_INVALID")), (path, bad, reasons)
        # a malformed TRANSITION under a kill never crashes closure and leaves an honest unjoined kill record
        bad = _json.loads(_json.dumps(good)); _set_path(bad, path, _wrong_values_for(spec)[0]); (tmp_path / "fallback-transition-lowering.json").write_text(_json.dumps(bad))
        s = _r5_post(tmp_path, launch, killed=True)
        assert s["non_pass"] is True and s["tags"][""]["state"] == "NO_VALID_EVENT" and (tmp_path / "fallback-kill-record.json").exists()
        kr = _json.loads((tmp_path / "fallback-kill-record.json").read_text()); assert kr["outcome"] == "RESOURCE_STOP_UNJOINED" and kr["prior_decline_claimed"] is False

    def test_required_nullable_nested_field_must_be_present(self, monkeypatch, tmp_path):
        """the review P3-R6-NULLABLE-REQUIRED-1 fixture: interpreter_failure without prior_decline_reason (nullable, NOT optional)."""
        _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        rec = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())
        rec.update(event="interpreter-failed", event_utc=fc._utc(), outcome="interpreter_failed:ValueError: synthetic", interpreter_failure={"type": "ValueError", "message": "synthetic", "prior_decline_stage": "lowering"})
        ok, reasons = fc.validate_record_safe(rec)
        assert ok is False and reasons == ["SCHEMA:interpreter_failure.prior_decline_reason"]

    @pytest.mark.parametrize("events", [42, [1], None, "broken"])
    def test_malformed_transition_history_cannot_crash_join(self, monkeypatch, tmp_path, events):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        _r5_mutate(tmp_path, "fallback-transition-lowering.json", lambda r: r.__setitem__("events", events))
        s = _r5_post(tmp_path, launch, killed=True)
        assert s["non_pass"] is True and s["rejected_events"] and s["rejected_events"][0]["code"].startswith("SCHEMA") and (tmp_path / "fallback-kill-record.json").exists()

    def test_list_element_types_are_enforced(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        good = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())
        for bad in ([{"event": 1, "event_utc": good["event_utc"]}], [{"event": "x"}], [{"event": "x", "event_utc": "no"}], ["x"]):
            rec = _json.loads(_json.dumps(good)); rec["events"] = bad; assert fc.validate_record_safe(rec)[0] is False
        for bad in ([{"backend": "numba", "stage": "lowering", "reason": 1, "typed": True}], [{"backend": "cuda", "stage": "lowering", "reason": "r", "typed": True}], [{"backend": "numba"}], [42]):
            rec = _json.loads(_json.dumps(good)); rec["history"] = bad; assert fc.validate_record_safe(rec)[0] is False
        assert fc.validate_record_safe(good)[0]


class TestProtocolV2CaseCoverage:
    def test_preparation_record_never_fulfils_a_required_execution_case(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, cases=["caseA", "caseB"]); _r5_engine(monkeypatch)
        _r5_emit(tmp_path, tag="caseB.", case="caseB")                                                     # caseB really executes
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, stage="caseA"), persist=_r5_persist(tmp_path, "caseA."), execution=False)   # preparation posing as case A
        s = _r5_post(tmp_path, launch)
        assert s["case_coverage"]["caseB"] == "CLOSED_OK" and s["case_coverage"]["caseA"] == "MISSING_CASE" and s["missing_cases"] == ["caseA"] and s["non_pass"] is True
        assert any(r["code"] == "SCOPE_CONFLICT" for r in s["rejected_events"])        # a declared case tag bound as a stage

    def test_case_and_stage_on_one_record_is_rejected_at_validation(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, tag="caseA-", case="caseA")
        for p in tmp_path.glob("caseA-fallback-*.json"):
            r = _json.loads(p.read_text()); r["binding"]["stage"] = "caseA"; p.write_text(_json.dumps(r))
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is True and all(r["code"] == "SCHEMA" and r["reasons"][0].startswith("SCHEMA:binding.xor") for r in s["rejected_events"]) and s["tags"]["caseA-"]["state"] == "NO_VALID_EVENT"
        with pytest.raises(fc.RecordInvalid):
            fc.attempt_binding(tmp_path, case="a", stage="b")

    @pytest.mark.parametrize("combo,expected_state,non_pass", [(("fast-ran",), "FAST_DONE", False), (("transition-lowering", "interpreter-ran"), "CLOSED_OK", False), (("transition-lowering", "interpreter-failed"), "CLOSED_FAIL", True), (("untyped-decline",), "FAST_FAILED", True), (("transition-lowering",), "OPEN", True), (("transition-lowering", "untyped-decline"), "STATE_CONFLICT", True)])
    def test_required_case_takes_its_closure_status(self, monkeypatch, tmp_path, combo, expected_state, non_pass):
        launch = _r6_make(monkeypatch, tmp_path, combo, tag="caseA.", case="caseA")
        s = _r5_post(tmp_path, launch, exit_code=0 if not non_pass else 1)
        assert s["case_coverage"]["caseA"] == expected_state and s["non_pass"] is non_pass and s["missing_cases"] == []

    def test_declared_case_and_stage_names_must_be_disjoint(self, tmp_path):
        with pytest.raises(fc.RecordInvalid):
            fc.mint_attempt(tmp_path, group="C", attempt=1, cases=["x"], stages=["x"])

    def test_declared_stage_used_as_a_case_is_a_scope_conflict(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, stages=["precheck"]); _r5_engine(monkeypatch); _r5_emit(tmp_path, tag="precheck.", case="precheck")
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is True and all(r["code"] == "SCOPE_CONFLICT" for r in s["rejected_events"])


class TestProtocolV2Review6Round5Probes:
    @pytest.mark.parametrize("event", ["untyped-decline", "numba-unavailable"])
    @pytest.mark.parametrize("killed", [True, False])
    def test_allowed_fast_failure_cannot_close_interpreter_transition(self, monkeypatch, tmp_path, event, killed):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False); _r6_fast_failed(monkeypatch, tmp_path, event, align_hook=True)
        s = _r5_post(tmp_path, launch, killed=killed)
        stop = tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")
        assert s["non_pass"] is True and stop.exists() and s["tags"][""]["state"] == "STATE_CONFLICT" and event in s["tags"][""]["conflicting_events"]

    @pytest.mark.parametrize("event", ["untyped-decline", "numba-unavailable", "interpreter-failed"])
    def test_complete_plus_failure_is_a_state_conflict(self, monkeypatch, tmp_path, event):
        launch = _r6_make(monkeypatch, tmp_path, ("transition-lowering", "interpreter-ran", event))
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is True and s["tags"][""]["state"] == "STATE_CONFLICT" and s["tags"][""]["closing_event"] is None

    @pytest.mark.parametrize("field,value", [("backend", 42), ("fallback_allowed", "yes"), ("fallback_taken", None), ("decline_hook_available", []), ("program_text_sha256", "not-sha256"), ("history", 42), ("events", [1]), ("preparation_only", None)])
    def test_declared_record_schema_is_enforced(self, monkeypatch, tmp_path, field, value):
        _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        rec = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text()); rec[field] = value
        ok, reasons = fc.validate_record_safe(rec); assert ok is False and reasons[0].startswith("SCHEMA")

    def test_fast_failure_without_transition_remains_legitimate(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r6_fast_failed(monkeypatch, tmp_path, "untyped-decline")
        s = _r5_post(tmp_path, launch, exit_code=2); assert s["non_pass"] is True and not s["rejected_events"] and s["tags"][""]["state"] == "FAST_FAILED"

    def test_interpreter_failure_with_transition_remains_legitimate(self, monkeypatch, tmp_path):
        launch = _r6_make(monkeypatch, tmp_path, ("transition-lowering", "interpreter-failed"))
        s = _r5_post(tmp_path, launch, exit_code=1); assert s["non_pass"] is True and not s["rejected_events"] and s["tags"][""]["state"] == "CLOSED_FAIL"


# =====================================================================================================================
# round 7 -- tests derived from PROTOCOL v3: fail-closed empty attempt + modes, cross-field / cross-record closing
# invariants (independent table), unjoined kill record independent of other rows, the reviewer's round-6 probes.
# =====================================================================================================================

class TestProtocolV3EmptyAttempt:
    @pytest.mark.parametrize("mode", list(fc.MODES))
    @pytest.mark.parametrize("killed", [False, True])
    @pytest.mark.parametrize("exit_code", [0, 2])
    def test_no_events_is_non_pass_in_every_mode_whatever_the_exit(self, tmp_path, mode, killed, exit_code):
        launch = _r5_mint(tmp_path, mode=mode)
        s = _r5_post(tmp_path, launch, killed=killed, exit_code=(None if killed else exit_code))
        assert s["non_pass"] is True and s["no_execution_evidence"] is True and s["execution_evidence"] is False and s["tags"][""]["state"] == "NO_EXECUTION_EVIDENCE" and s["mode"] == mode
        assert (tmp_path / "fallback-kill-record.json").exists() is killed
        if killed:
            kr = _json.loads((tmp_path / "fallback-kill-record.json").read_text()); assert kr["outcome"] == "RESOURCE_STOP_UNJOINED" and kr["scope_state"] == "NO_EXECUTION_EVIDENCE" and kr["prior_decline_claimed"] is False

    def test_deleted_events_with_retained_mint_is_non_pass(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path)
        for p in tmp_path.glob("fallback-*.json"):
            if p.name != fc.MINT_FILE: p.unlink()
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is True and s["no_execution_evidence"] is True and s["mint_check"]["state"] == "MINT_OK"

    def test_no_fallback_expected_passes_only_with_the_positive_record(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, mode="no-fallback-expected")
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path), persist=_r5_persist(tmp_path))
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is False and s["tags"][""]["state"] == "FAST_DONE" and s["mode_violations"] == [] and s["execution_evidence"] is True

    def test_no_fallback_expected_without_positive_record_is_non_pass(self, tmp_path):
        s = _r5_post(tmp_path, _r5_mint(tmp_path, mode="no-fallback-expected"), exit_code=0)
        assert s["non_pass"] is True and s["no_execution_evidence"] is True

    def test_no_fallback_expected_transition_is_a_mode_violation(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, mode="no-fallback-expected"); _r5_engine(monkeypatch); _r5_emit(tmp_path)   # typed decline -> interpreter completed
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["state"] == "CLOSED_OK" and s["mode_violations"] and s["mode_violations"][0]["code"] == "MODE_VIOLATION" and "+MODE_VIOLATION" in s["tags"][""]["status"] and s["non_pass"] is True

    def test_mint_carries_the_mode_and_refuses_unknown_modes(self, tmp_path):
        launch = _r5_mint(tmp_path, mode="no-fallback-expected")
        assert launch.mode == "no-fallback-expected" and _json.loads((tmp_path / fc.MINT_FILE).read_text())["mode"] == "no-fallback-expected"
        with pytest.raises(fc.RecordInvalid):
            _r5_mint(tmp_path / "other", mode="whatever")
        with pytest.raises(fc.RecordInvalid):
            fc.LaunchDescriptor(run_dir="/r", out_dir="/r", mode="whatever")

    def test_exit_code_never_establishes_closure(self, monkeypatch, tmp_path):
        s0 = _r5_post(tmp_path, _r5_mint(tmp_path), exit_code=0); assert s0["non_pass"] is True
        d = tmp_path / "ok"; d.mkdir(); launch = _r5_mint(d); _r5_engine(monkeypatch); _r5_emit(d)
        assert _r5_post(d, launch, exit_code=0)["non_pass"] is False and _r5_post(d, launch, exit_code=3)["non_pass"] is True


# the review mutations with the cross-field / cross-record invariant ids they violate (independent table; the reasons must name at least one)
_R6_MUTATIONS = [("decline_stage", "module_exec", {"I7", "I8", "C1"}), ("decline_reason", "a different declined kernel", {"I8", "C2"}), ("backend", "numba", {"I1"}),
                 ("executed_program", "not_executed_preparation", {"I4"}), ("fallback_taken", False, {"I3"}), ("fallback_allowed", False, {"I2"}),
                 ("program_text_sha256", "a" * 64, {"I6", "C3"}),
                 ("interpreter_failure", {"type": "ValueError", "message": "explicit failure", "prior_decline_reason": "another reason", "prior_decline_stage": "lowering"}, {"I5"})]


def _ids(rejected):
    return {r.split(":")[1] for x in rejected for r in x["reasons"] if ":" in r}


def _names(ids, path):
    """A re-emit difference names the field path or a sub-key of it (open objects such as engine.config)."""
    return path in ids or any(i.startswith(path + ".") for i in ids)


class TestProtocolV3ClosingInvariants:
    @pytest.mark.parametrize("field,value,ids", _R6_MUTATIONS, ids=[m[0] for m in _R6_MUTATIONS])
    @pytest.mark.parametrize("killed", [False, True])
    def test_closing_record_must_describe_interpreter_execution_and_match_the_transition(self, monkeypatch, tmp_path, field, value, ids, killed):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, value))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert [r["code"] for r in info["rejected_events"]] == ["TERMINAL_INCONSISTENT"] and _ids(info["rejected_events"]) & ids, (field, info["rejected_events"])
        assert info["state"] == "OPEN" and info["closing_event"] is None and s["non_pass"] is True
        assert (tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")).exists(), "a contradictory terminal never suppresses the joined record"
        assert not (tmp_path / "fallback-kill-record.json").exists()

    @pytest.mark.parametrize("ids,mutate", [({"decline_stage", "history", "lowered_sha256"}, lambda r: (r.__setitem__("decline_stage", "module_exec"), r["history"][-1].__setitem__("stage", "module_exec"), r.__setitem__("lowered_sha256", "c" * 64))),
                                            ({"decline_reason", "history"}, lambda r: (r.__setitem__("decline_reason", "other"), r["history"][-1].__setitem__("reason", "other"))),
                                            ({"events"}, lambda r: r.__setitem__("events", r["events"][1:])),
                                            ({"binding.row"}, lambda r: r["binding"].__setitem__("row", 7))], ids=["stage+history+lowered", "reason+history", "trail-prefix", "binding"])   # v4: re-emit differences are named by FIELD PATH
    def test_field_consistent_terminal_still_must_match_the_transition(self, monkeypatch, tmp_path, ids, mutate):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", mutate)
        s = _r5_post(tmp_path, launch); info = s["tags"][""]
        assert info["rejected_events"] and info["rejected_events"][0]["code"] in ("TERMINAL_INCONSISTENT", "IDENTITY_MISMATCH") and info["state"] == "OPEN" and info["closing_event"] is None and s["non_pass"] is True
        if info["rejected_events"][0]["code"] == "TERMINAL_INCONSISTENT": assert _ids(info["rejected_events"]) & ids, info["rejected_events"]

    @pytest.mark.parametrize("event,mutate,inv", [("fast-ran", lambda r: r.__setitem__("program_text_sha256", "b" * 64), "I6"), ("fast-ran", lambda r: r.__setitem__("history", [{"backend": "numba", "stage": "lowering", "reason": "x", "typed": True}]), "I8"),
                                                   ("untyped-decline", lambda r: r.__setitem__("fallback_taken", True), "I3"), ("untyped-decline", lambda r: r.__setitem__("events", []), "I9"),
                                                   ("transition-lowering", lambda r: r.__setitem__("fallback_allowed", False), "I2"), ("transition-lowering", lambda r: r.__setitem__("lowered_sha256", "c" * 64), "I7")], ids=["fast-program", "fast-history", "untyped-taken", "untyped-trail", "transition-allowed", "transition-lowered"])
    def test_non_closing_rows_obey_the_cross_field_invariants(self, monkeypatch, tmp_path, event, mutate, inv):
        launch = _r6_make(monkeypatch, tmp_path, (event,))
        _r5_mutate(tmp_path, "fallback-" + event + ".json", mutate)
        s = _r5_post(tmp_path, launch); info = s["tags"][""]
        code = "TERMINAL_INCONSISTENT" if event == "fast-ran" else "FIELD_INCONSISTENT"      # v4: fast-ran is a re-emit row
        assert [r["code"] for r in info["rejected_events"]] == [code] and inv in _ids(info["rejected_events"]) and info["state"] == "NO_VALID_EVENT" and s["non_pass"] is True

    def test_precheck_preparation_obeys_the_cross_field_invariants(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, stages=["pre"]); _r5_engine(monkeypatch)
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path, stage="pre"), persist=_r5_persist(tmp_path, "pre."), execution=False)
        _r5_mutate(tmp_path, "pre.fallback-precheck-decline.json", lambda r: r.__setitem__("executed_program", "numba_lowered"))
        s = _r5_post(tmp_path, launch); info = s["tags"]["pre."]
        assert [r["code"] for r in info["rejected_events"]] == ["FIELD_INCONSISTENT"] and "I4" in _ids(info["rejected_events"]) and s["non_pass"] is True

    @pytest.mark.parametrize("lowered,stage", [(None, "lowering"), (LOWERED_RAISES_TYPED_AT_IMPORT, "module_exec"), (LOWERED_RAISES_TYPED_IN_RUN_MODEL, "run_model")], ids=["lowering", "module_exec", "run_model"])
    def test_real_consistent_completion_closes_at_every_stage(self, monkeypatch, tmp_path, lowered, stage):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch, lowered=lowered)
        if lowered is not None: monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(tmp_path), persist=_r5_persist(tmp_path))
        s = _r5_post(tmp_path, launch); info = s["tags"][""]
        assert info["rejected_events"] == [] and info["state"] == "CLOSED_OK" and info["surviving_transition"] == "transition-" + stage and s["non_pass"] is False
        ran = _json.loads((tmp_path / "fallback-interpreter-ran.json").read_text()); tr = _json.loads((tmp_path / f"fallback-transition-{stage}.json").read_text())
        assert _oracle_consistent(ran, tr) and fc._cross_field("interpreter-ran", ran) == [] and fc._cross_field("transition-" + stage, tr) == []

    def test_every_real_row_satisfies_the_cross_field_invariants(self, monkeypatch, tmp_path):
        """Every row the shim emits passes I1-I9 by construction (the invariants describe the emission path)."""
        for i, combo in enumerate([("transition-lowering", "interpreter-ran"), ("fast-ran",), ("untyped-decline",), ("numba-unavailable",)]):
            d = tmp_path / str(i); d.mkdir(); _r6_make(monkeypatch, d, combo)
            for e in combo:
                assert fc._cross_field(e, _json.loads((d / ("fallback-" + e + ".json")).read_text())) == [], e
        d = tmp_path / "pre"; d.mkdir(); _r5_mint(d, stages=["s"]); _r5_engine(monkeypatch)
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(d, stage="s"), persist=_r5_persist(d, "s."), execution=False)
        assert fc._cross_field("precheck-decline", _json.loads((d / "s.fallback-precheck-decline.json").read_text())) == []


class TestProtocolV3UnjoinedKill:
    @pytest.mark.parametrize("extra", ["untyped-decline", "numba-unavailable", "interpreter-ran"])
    def test_rejected_transition_plus_other_row_still_records_the_outer_kill(self, monkeypatch, tmp_path, extra):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=(extra == "interpreter-ran"))
        _r5_mutate(tmp_path, "fallback-transition-lowering.json", lambda r: r.__setitem__("events", [1]))
        if extra != "interpreter-ran": _r6_fast_failed(monkeypatch, tmp_path, extra)
        s = _r5_post(tmp_path, launch, killed=True); info = s["tags"][""]
        assert s["non_pass"] is True and (tmp_path / "fallback-kill-record.json").exists() and not (tmp_path / "fallback-resource-stop.json").exists()
        kr = _json.loads((tmp_path / "fallback-kill-record.json").read_text())
        assert kr["outcome"] == "RESOURCE_STOP_UNJOINED" and kr["prior_decline_claimed"] is False and kr["scope_state"] == info["state"] and any(r["event"] == "transition-lowering" and r["code"] == "SCHEMA" for r in kr["rejected_events"])
        assert info["state"] == ("FAST_FAILED" if extra != "interpreter-ran" else "STATE_CONFLICT") and "+RESOURCE_STOP_UNJOINED" in info["status"]

    def test_fast_failure_alone_under_kill_records_the_kill(self, monkeypatch, tmp_path):
        launch = _r6_make(monkeypatch, tmp_path, ("untyped-decline",), killed=True)
        s = _r5_post(tmp_path, launch, killed=True)
        assert s["tags"][""]["state"] == "FAST_FAILED" and (tmp_path / "fallback-kill-record.json").exists() and _json.loads((tmp_path / "fallback-kill-record.json").read_text())["valid_non_closing_events"] == ["untyped-decline"]

    @pytest.mark.parametrize("combo,state", [(("transition-lowering", "interpreter-ran"), "CLOSED_OK"), (("fast-ran",), "FAST_DONE"), (("transition-lowering", "interpreter-failed"), "CLOSED_FAIL")])
    def test_consistent_completion_under_kill_writes_no_kill_record(self, monkeypatch, tmp_path, combo, state):
        launch = _r6_make(monkeypatch, tmp_path, combo, killed=True)
        s = _r5_post(tmp_path, launch, killed=True); info = s["tags"][""]
        assert info["state"] == state and "+GUARD_KILL_AFTER_TERMINAL" in info["status"] and s["non_pass"] is True and not (tmp_path / "fallback-kill-record.json").exists() and not (tmp_path / "fallback-resource-stop.json").exists()

    def test_stage_fast_failure_under_kill_records_the_kill_under_its_tag(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path, stages=["pre"]); _r6_fast_failed(monkeypatch, tmp_path, "untyped-decline", tag="pre.", stage="pre")
        s = _r5_post(tmp_path, launch, killed=True)
        assert s["tags"]["pre."]["state"] == "FAST_FAILED" and (tmp_path / "pre.fallback-kill-record.json").exists()

    def test_surviving_transition_gets_the_joined_record_not_the_unjoined_one(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        s = _r5_post(tmp_path, launch, killed=True)
        assert (tmp_path / "fallback-resource-stop.json").exists() and not (tmp_path / "fallback-kill-record.json").exists() and s["tags"][""]["status"] == fc.OUTCOME_RESOURCE_STOP


class TestProtocolV3Review6Round6Probes:
    """the reviewer's round-6 probe file test_new_v6.py restated with this file's helpers (the sealed originals are run unmodified as well)."""

    @pytest.mark.parametrize("delete_emitted", [False, True])
    def test_empty_parent_attempt_cannot_report_non_pass_false(self, monkeypatch, tmp_path, delete_emitted):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch)
        if delete_emitted:
            _r5_emit(tmp_path)
            for p in tmp_path.glob("fallback-*.json"):
                if p.name != fc.MINT_FILE: p.unlink()
        assert _r5_post(tmp_path, launch)["non_pass"] is True

    @pytest.mark.parametrize("field,value,_ids", _R6_MUTATIONS, ids=[m[0] for m in _R6_MUTATIONS])
    @pytest.mark.parametrize("killed", [False, True])
    def test_closing_record_semantics_must_match_transition(self, monkeypatch, tmp_path, field, value, _ids, killed):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__(field, value))
        s = _r5_post(tmp_path, launch, killed=killed)
        joined = tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")
        assert s["non_pass"] and joined.exists() and s["tags"][""]["closing_event"] is None

    def test_prior_handbuilt_failure_with_required_fields_is_legitimate(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        rec = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text()); now = fc._utc()
        rec.update(event="interpreter-failed", event_utc=now, outcome="interpreter_failed:ValueError: synthetic", interpreter_failure={"type": "ValueError", "message": "synthetic", "prior_decline_reason": rec["decline_reason"], "prior_decline_stage": rec["decline_stage"]})
        rec["events"] = rec["events"] + [{"event": "interpreter-failed", "event_utc": now}]
        (tmp_path / "fallback-interpreter-failed.json").write_text(_json.dumps(rec))
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] and not s["rejected_events"] and s["tags"][""]["state"] == "CLOSED_FAIL"

    def test_clean_matching_completion_control(self, monkeypatch, tmp_path):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path)
        s = _r5_post(tmp_path, launch)
        assert not s["non_pass"] and s["tags"][""]["state"] == "CLOSED_OK" and s["execution_evidence"] is True


# =====================================================================================================================
# round 8 -- PROTOCOL v4 RE-EMIT AND COMPARE: generated per deterministic field for every closing kind x mode x kill,
# free fields inside / outside their constraints, emitter == shared function, the reviewer's round-7 probes.
# =====================================================================================================================

import re as _re8
_REEMIT = _P7["reemit_and_compare"]
_HEX8 = _re8.compile(r"[0-9a-f]{64}")
R8_FAIL_SRC = 'def run_model(inputs):\n    raise ValueError("synthetic interpreter failure")\n'


def _r8_make(monkeypatch, out, kind, mode="fallback-allowed", expected_engine=None, learner_case=None):
    """Real toy records for one closing kind; returns (launch, closing record path).  learner_case="B": a parent-scope fast-ran
    LEARNS the attempt context first and the returned path is case B's fast-ran (PROTOCOL v5: learned fields are fixed)."""
    launch = _r5_mint(out, mode=mode, expected_engine=expected_engine, cases=([learner_case] if learner_case else None), **({"source_sha256": hashlib.sha256(R8_FAIL_SRC.encode()).hexdigest()} if kind == "interpreter-failed" else {}))
    if kind == "fast-ran":
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(out), persist=_r5_persist(out), config={"use_fp64": True, "boundscheck": True})
        if learner_case:
            fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(out, case=learner_case), persist=_r5_persist(out, learner_case + "-"), config={"use_fp64": True, "boundscheck": True})
            return launch, out / (learner_case + "-fallback-fast-ran.json")
    elif kind == "interpreter-ran":
        _r5_engine(monkeypatch)
        fc.execute_candidate(R5_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(out), persist=_r5_persist(out), config={"use_fp64": True, "boundscheck": True})
    else:
        _r5_engine(monkeypatch)
        with pytest.raises(fc.InterpreterFailed):
            fc.execute_candidate(R8_FAIL_SRC, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, binding=fc.attempt_binding(out), persist=_r5_persist(out), config={"use_fp64": True, "boundscheck": True})
    return launch, out / ("fallback-" + kind + ".json")


def _r8_paths(kind, rec):
    """Expand the spec's deterministic field list into the concrete dotted paths present in the record."""
    out = []
    for f in _REEMIT["deterministic_fields"][kind]:
        if f == "engine": out += ["engine." + k for k in rec["engine"]]
        elif f == "binding": out += ["binding." + k for k in rec["binding"]]
        elif f == "interpreter_failure" and isinstance(rec["interpreter_failure"], dict): out += ["interpreter_failure.prior_decline_reason", "interpreter_failure.prior_decline_stage"]
        elif "." in f:
            o, k = _walk(rec, f)
            if isinstance(o, dict) and k in o: out.append(f)
        else: out.append(f)
    return out


def _r8_mutate(value, path):
    """A type-preserving DIFFERENT value for a deterministic field."""
    if isinstance(value, bool): return not value
    if value is None:
        if path.endswith("sha256"): return "d" * 64                                   # a null hash becomes a VALID different hash (not a schema failure)
        if path == "interpreter_failure": return {"type": "ValueError", "message": "m", "prior_decline_reason": None, "prior_decline_stage": None}
        if path == "backend_terminal": return "numba"
        if path.endswith("prior_decline_stage") or path == "decline_stage": return "module_exec"
        return "zzz"
    if isinstance(value, str):
        if _HEX8.fullmatch(value): return "d" * 64
        if path == "event": return "interpreter-ran" if value != "interpreter-ran" else "fast-ran"
        if path == "decline_stage" or path.endswith("prior_decline_stage"): return "module_exec" if value != "module_exec" else "run_model"
        if path == "executed_program": return "original_dsl_interpreter" if value != "original_dsl_interpreter" else "numba_lowered"      # another VALID enum value
        if path in ("backend", "backend_attempt", "backend_terminal"): return "numba" if value != "numba" else "interpreter"
        return value + "x"
    if isinstance(value, int): return value + 1
    if isinstance(value, dict): return {**value, "boundscheck": not value.get("boundscheck", True)} if "boundscheck" in value else {**value, "zz": 1}
    if isinstance(value, list):
        if path == "events": return [{"event": "transition-lowering", "event_utc": value[0]["event_utc"]}] + value   # prepended trail event (the review TRAIL-CLOSURE-1)
        if path == "history": return [] if value else [{"backend": "numba", "stage": "lowering", "reason": "r", "typed": True}]
        return value + ["x"]
    return "zzz"


_REJECTING_CODES = {"TERMINAL_INCONSISTENT", "CONTEXT_MISMATCH", "STATE_NOT_ALLOWED", "IDENTITY_MISMATCH", "SOURCE_MISMATCH", "ENGINE_PIN_MISMATCH", "ENGINE_HASH_MISMATCH", "CONTRACT_MISMATCH", "EVENT_FILENAME_MISMATCH", "SCHEMA", "TIME_BEFORE_TRANSITION", "TIME_BEFORE_MINT", "UNKNOWN_EVENT", "SUBATTEMPT_MISMATCH", "SCOPE_CONFLICT", "STATE_CONFLICT"}


def _r8_cases():
    for kind in _REEMIT["reemit_events"]:
        for mode in fc.MODES:
            for killed in (False, True):
                yield kind, mode, killed


class TestProtocolV4ReemitAndCompare:
    @pytest.mark.parametrize("kind,mode,killed", list(_r8_cases()), ids=[f"{k}-{m}-{'killed' if x else 'clean'}" for k, m, x in _r8_cases()])
    def test_every_deterministic_field_mutation_is_rejected_and_never_closes(self, monkeypatch, tmp_path, kind, mode, killed):
        tag = "B-" if kind == "fast-ran" else ""                                   # v5: the fast case under test is case B; the parent fast-ran has LEARNED the context
        launch, path = _r8_make(monkeypatch, tmp_path, kind, mode, learner_case=("B" if kind == "fast-ran" else None))
        good = _json.loads(path.read_text()); paths = _r8_paths(kind, good); assert len(paths) >= 20, paths
        checked = {}
        for fp in paths:
            rec = _json.loads(_json.dumps(good)); o, k = _walk(rec, fp); o[k] = _r8_mutate(o[k], fp); path.write_text(_json.dumps(rec))
            for stale in ("fallback-resource-stop.json", "fallback-open-transition.json", "fallback-kill-record.json"):
                if (tmp_path / (tag + stale)).exists(): (tmp_path / (tag + stale)).rename(tmp_path / ("stale-" + fp.replace(".", "_") + "-" + tag + stale))
            s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][tag]
            rej = [r for r in info["rejected_events"] if r["event"] == kind]
            assert rej and rej[0]["code"] in _REJECTING_CODES, (kind, fp, info)
            if rej[0]["code"] == "TERMINAL_INCONSISTENT":
                ids = _ids(rej); assert _names(ids, fp) or any(_re8.fullmatch(r"I\d+", i) for i in ids), (kind, fp, rej)
            assert info["closing_event"] is None and info["state"] not in ("CLOSED_OK", "CLOSED_FAIL", "FAST_DONE") and s["non_pass"] is True, (kind, fp, info)
            if kind != "fast-ran":   # the surviving transition keeps its joined record
                assert info["state"] == "OPEN" and (tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")).exists(), (kind, fp, info)
                assert not (tmp_path / "fallback-kill-record.json").exists()
            else:                    # nothing valid survives in case B: the kill is recorded unjoined under B's tag; B is MISSING_CASE
                assert info["state"] == "NO_VALID_EVENT" and (tmp_path / (tag + "fallback-kill-record.json")).exists() is killed and s["case_coverage"]["B"] == "MISSING_CASE", (kind, fp, info)
            checked[fp] = rej[0]["code"]
        path.write_text(_json.dumps(good))                     # the untouched record closes again (control)
        for stale in ("fallback-resource-stop.json", "fallback-open-transition.json", "fallback-kill-record.json"):
            if (tmp_path / (tag + stale)).exists(): (tmp_path / (tag + stale)).rename(tmp_path / ("stale-control-" + tag + stale))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][tag]
        assert info["rejected_events"] == [] and info["state"] == {"interpreter-ran": "CLOSED_OK", "interpreter-failed": "CLOSED_FAIL", "fast-ran": "FAST_DONE"}[kind], (kind, info)
        # the relationships the anchors / schema / state table cannot catch MUST be caught by the re-emit comparison itself
        must = {"fast-ran": {"events", "history", "program_text_sha256", "executed_program", "engine.config", "engine.python", "engine.executable", "engine.solar_package_path", "decline_hook_available"}, "interpreter-ran": {"events", "history", "engine.config", "engine.python", "engine.executable", "engine.solar_package_path", "decline_hook_available", "lowered_sha256", "program_text_sha256", "decline_reason"},
                "interpreter-failed": {"events", "history", "engine.config", "engine.python", "engine.executable", "engine.solar_package_path", "decline_hook_available", "lowered_sha256", "program_text_sha256", "outcome", "interpreter_failure.prior_decline_reason", "interpreter_failure.prior_decline_stage"}}[kind]
        ctx_codes = {"TERMINAL_INCONSISTENT", "CONTEXT_MISMATCH", "ENGINE_HASH_MISMATCH"}                       # v5: learned/anchored context fields are rejected BEFORE admission (CONTEXT_MISMATCH), the rest by the comparison
        assert all(checked.get(m) in ctx_codes for m in must), {m: checked.get(m) for m in must}

    @pytest.mark.parametrize("kind", ["interpreter-ran", "interpreter-failed", "fast-ran"])
    def test_emitter_writes_exactly_the_shared_functions_record(self, monkeypatch, tmp_path, kind):
        """PROTOCOL v4 emitter obligation: the persisted closing record IS expected_closing_record(...) of its inputs."""
        _, path = _r8_make(monkeypatch, tmp_path, kind); rec = _json.loads(path.read_text())
        if kind == "fast-ran":
            exp = fc.expected_closing_record(None, {"binding": rec["binding"], "engine": rec["engine"], "source_sha256": rec["source_sha256"], "decline_hook_available": rec["decline_hook_available"]}, kind, {"event_utc": rec["event_utc"], "lowered_sha256": rec["lowered_sha256"]})
        else:
            tr = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text())
            free = {"event_utc": rec["event_utc"]}
            if kind == "interpreter-failed": free.update(failure_type=rec["interpreter_failure"]["type"], failure_message=rec["interpreter_failure"]["message"])
            exp = fc.expected_closing_record(tr, None, kind, free)
        assert exp == rec and fc._deep_diff(exp, rec) == []

    def test_free_fields_inside_their_constraints_stay_accepted_and_outside_are_rejected(self, monkeypatch, tmp_path):
        # interpreter-failed: another message (<= 1000) with the DERIVED outcome -> accepted; > 1000 -> rejected; non-derived outcome -> rejected
        d = tmp_path / "f"; d.mkdir(); launch, path = _r8_make(monkeypatch, d, "interpreter-failed"); good = _json.loads(path.read_text())
        rec = _json.loads(_json.dumps(good)); rec["interpreter_failure"]["message"] = "m" * 900; rec["outcome"] = "interpreter_failed:ValueError: " + "m" * 400; path.write_text(_json.dumps(rec))
        assert _r5_post(d, launch)["tags"][""]["state"] == "CLOSED_FAIL"
        rec["interpreter_failure"]["message"] = "m" * 1001; path.write_text(_json.dumps(rec)); s = _r5_post(d, launch)
        assert "interpreter_failure.message" in _ids(s["tags"][""]["rejected_events"]) and s["tags"][""]["state"] == "OPEN"
        rec = _json.loads(_json.dumps(good)); rec["outcome"] = "interpreter_failed:TypeError: incompatible"; path.write_text(_json.dumps(rec)); s = _r5_post(d, launch)
        assert "outcome" in _ids(s["tags"][""]["rejected_events"]) and s["tags"][""]["closing_event"] is None   # the review FAILED-CLOSURE-1
        # event_utc: later than the transition -> accepted; in the future -> rejected
        rec = _json.loads(_json.dumps(good)); later = (datetime_mod.datetime.fromisoformat(rec["event_utc"]) + datetime_mod.timedelta(seconds=1)).isoformat(); rec["event_utc"] = later; rec["events"][-1]["event_utc"] = later; path.write_text(_json.dumps(rec))
        assert _r5_post(d, launch)["tags"][""]["state"] == "CLOSED_FAIL"
        future = (datetime_mod.datetime.now(datetime_mod.timezone.utc) + datetime_mod.timedelta(hours=1)).isoformat(); rec["event_utc"] = future; rec["events"][-1]["event_utc"] = future; path.write_text(_json.dumps(rec))
        assert "event_utc" in _ids(_r5_post(d, launch)["tags"][""]["rejected_events"])
        # fast-ran: lowered hash with the derived program hash -> accepted; alone -> rejected; hook flag free; engine fields free WITHOUT expected_engine, deterministic WITH it
        e = tmp_path / "e"; e.mkdir(); launch, path = _r8_make(monkeypatch, e, "fast-ran"); good = _json.loads(path.read_text())
        rec = _json.loads(_json.dumps(good)); rec["lowered_sha256"] = rec["program_text_sha256"] = "e" * 64; path.write_text(_json.dumps(rec)); assert _r5_post(e, launch)["tags"][""]["state"] == "FAST_DONE"
        rec["program_text_sha256"] = good["program_text_sha256"]; path.write_text(_json.dumps(rec)); assert _ids(_r5_post(e, launch)["tags"][""]["rejected_events"]) & {"program_text_sha256", "I6"}
        rec = _json.loads(_json.dumps(good)); rec["decline_hook_available"] = not rec["decline_hook_available"]; path.write_text(_json.dumps(rec)); assert _r5_post(e, launch)["tags"][""]["state"] == "FAST_DONE"   # the single record is the LEARNER (v5)
        rec = _json.loads(_json.dumps(good)); rec["engine"]["python"] = "3.12.0"; path.write_text(_json.dumps(rec)); assert _r5_post(e, launch)["tags"][""]["state"] == "FAST_DONE"      # learner: no expected_engine, no earlier record
        g = tmp_path / "g"; g.mkdir(); launch2, path2 = _r8_make(monkeypatch, g, "fast-ran", expected_engine=good["engine"]); rec = _json.loads(path2.read_text()); rec["engine"]["python"] = "3.12.0"; path2.write_text(_json.dumps(rec))
        assert "engine.python" in _ids(_r5_post(g, launch2)["tags"][""]["rejected_events"])                                                                                     # deterministic once the parent anchors it

    def test_mode_does_not_change_any_record_field(self, monkeypatch, tmp_path):
        a = tmp_path / "a"; a.mkdir(); b = tmp_path / "b"; b.mkdir()
        _, pa = _r8_make(monkeypatch, a, "fast-ran", "fallback-allowed"); _, pb = _r8_make(monkeypatch, b, "fast-ran", "no-fallback-expected")
        ra, rb = _json.loads(pa.read_text()), _json.loads(pb.read_text())
        for r in (ra, rb): r["event_utc"] = r["events"][0]["event_utc"] = "T"; r["binding"]["out_dir"] = r["binding"]["run_dir"] = "D"
        assert ra == rb and fc.expected_closing_record(None, {"binding": {}, "engine": {}, "source_sha256": "a" * 64, "decline_hook_available": True}, "fast-ran", {"event_utc": "t", "lowered_sha256": "b" * 64}, mode="no-fallback-expected")["events"] == [{"event": "fast-ran", "event_utc": "t"}]


class TestProtocolV4Review6Round7Probes:
    """the reviewer's round-7 probe file test_new_v7.py restated with this file's helpers (the sealed originals are run unmodified as well)."""

    @pytest.mark.parametrize("prior", ["transition-lowering", "interpreter-ran", "interpreter-failed", "untyped-decline"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_fast_terminal_cannot_carry_fallback_or_failure_trail(self, monkeypatch, tmp_path, prior, killed):
        launch, path = _r8_make(monkeypatch, tmp_path, "fast-ran", "no-fallback-expected"); rec = _json.loads(path.read_text())
        rec["events"].insert(0, {"event": prior, "event_utc": rec["event_utc"]}); path.write_text(_json.dumps(rec))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert s["rejected_events"] and "events" in _ids(info["rejected_events"]) and info["state"] == "NO_VALID_EVENT" and s["non_pass"] is True
        assert (tmp_path / "fallback-kill-record.json").exists() is killed

    @pytest.mark.parametrize("field", ["config", "python", "executable", "decline_hook_available"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_terminal_preserves_transition_execution_context(self, monkeypatch, tmp_path, field, killed):
        launch, path = _r8_make(monkeypatch, tmp_path, "interpreter-ran"); rec = _json.loads(path.read_text())
        if field == "config": rec["engine"]["config"] = {"use_fp64": False, "boundscheck": False}
        elif field == "python": rec["engine"]["python"] = "3.12.0"
        elif field == "executable": rec["engine"]["executable"] = "/unrelated/python"
        else: rec["decline_hook_available"] = False
        path.write_text(_json.dumps(rec)); s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert [r["code"] for r in info["rejected_events"]] in (["TERMINAL_INCONSISTENT"], ["CONTEXT_MISMATCH"]) and info["state"] == "OPEN" and s["non_pass"] is True   # v5: the transition has LEARNED the context, so the terminal is a CONTEXT_MISMATCH before the comparison
        assert (tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")).exists()
        ids = _ids(info["rejected_events"]); assert _names(ids, "engine." + field) or (field == "decline_hook_available" and ("decline_hook_available" in ids or "I10" in ids)), ids

    @pytest.mark.parametrize("killed", [False, True])
    def test_lowering_typed_decline_requires_available_hook(self, monkeypatch, tmp_path, killed):
        launch, path = _r8_make(monkeypatch, tmp_path, "interpreter-ran")
        for ev in ("transition-lowering", "interpreter-ran"):
            _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r.__setitem__("decline_hook_available", False))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert s["rejected_events"] and s["non_pass"] is True and "I10" in _ids(info["rejected_events"]) and info["surviving_transition"] is None
        assert (tmp_path / "fallback-kill-record.json").exists() is killed

    @pytest.mark.parametrize("killed", [False, True])
    def test_failure_object_matches_failure_outcome(self, monkeypatch, tmp_path, killed):
        launch = _r5_mint(tmp_path); _r5_engine(monkeypatch); _r5_emit(tmp_path, complete=False)
        rec = _json.loads((tmp_path / "fallback-transition-lowering.json").read_text()); now = fc._utc()
        rec.update(event="interpreter-failed", event_utc=now, outcome="interpreter_failed:TypeError: incompatible", interpreter_failure={"type": "ValueError", "message": "different", "prior_decline_stage": rec["decline_stage"], "prior_decline_reason": rec["decline_reason"]})
        rec["events"].append({"event": "interpreter-failed", "event_utc": now}); (tmp_path / "fallback-interpreter-failed.json").write_text(_json.dumps(rec))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert [r["code"] for r in info["rejected_events"]] == ["TERMINAL_INCONSISTENT"] and "outcome" in _ids(info["rejected_events"]) and info["state"] == "OPEN"
        assert (tmp_path / ("fallback-resource-stop.json" if killed else "fallback-open-transition.json")).exists()

    def test_child_mode_in_record_or_mint_cannot_override_parent(self, monkeypatch, tmp_path):
        launch, _ = _r8_make(monkeypatch, tmp_path, "interpreter-ran", "no-fallback-expected")
        for ev in ("transition-lowering", "interpreter-ran"):
            _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r.__setitem__("mode", "fallback-allowed"))
        s = _r5_post(tmp_path, launch); assert s["non_pass"] and s["mode_violations"]
        m = _json.loads((tmp_path / fc.MINT_FILE).read_text()); m["mode"] = "fallback-allowed"; (tmp_path / fc.MINT_FILE).write_text(_json.dumps(m))
        s = _r5_post(tmp_path, launch); assert s["non_pass"] and s["mode_violations"] and s["mint_check"]["state"] == "MINT_ALTERED"

    def test_fast_positive_mode_control_and_fabrication_limit(self, monkeypatch, tmp_path):
        launch, path = _r8_make(monkeypatch, tmp_path, "fast-ran", "no-fallback-expected")
        s = _r5_post(tmp_path, launch); assert not s["non_pass"] and not s["mode_violations"] and s["verdict"] is None
        rec = _json.loads(path.read_text()); path.unlink(); path.write_text(_json.dumps(rec))      # an identical, internally consistent copy is indistinguishable: stated limit, verdict still null
        s = _r5_post(tmp_path, launch); assert not s["non_pass"] and s["verdict"] is None

    def test_retained_i3_control(self, monkeypatch, tmp_path):
        launch, path = _r8_make(monkeypatch, tmp_path, "interpreter-ran")
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r.__setitem__("fallback_taken", False))
        s = _r5_post(tmp_path, launch, killed=True); assert s["non_pass"] and (tmp_path / "fallback-resource-stop.json").exists()


# =====================================================================================================================
# round 9 -- PROTOCOL v5: the attempt context resolved once (anchor -> learned -> free), enforced before admission;
# canonical out_dir rule; the reviewer's round-8 probes.
# =====================================================================================================================

R9_CONFIG = {"use_fp64": True, "boundscheck": True}
R9_ENGINE_MUT = {"python": "3.12.0", "executable": "/different/python", "solar_package_path": "/different/solar", "config": {"use_fp64": False, "boundscheck": False},
                 "numba_accel_sha256": "d" * 64, "verify_sha256": "d" * 64, "core_sha256": "d" * 64, "engine_pin": "other-pin"}


def _r9_emit(out, case=None, src=R5_SRC, config=R9_CONFIG):
    prefix = case + "-" if case else ""
    return fc.execute_candidate(src, lambda: {"x": np.array([7.0])}, engine_pin=R5_PIN, config=config, binding=fc.attempt_binding(out, **({"case": case} if case else {})), persist=_r5_persist(out, prefix))


def _r9_set(rec, field, value):
    if field.startswith("engine."): rec["engine"][field[len("engine."):]] = value
    elif field == "binding.out_dir": rec["binding"]["out_dir"] = value
    else: rec[field] = value


def _r9_other(rec, field):
    if field.startswith("engine."): return R9_ENGINE_MUT[field[len("engine."):]]
    if field == "source_sha256": return "e" * 64
    if field == "decline_hook_available": return not rec["decline_hook_available"]
    return rec["binding"]["out_dir"] + "/../" + _pathlib.Path(rec["binding"]["out_dir"]).name   # a spelling ALIAS of the same directory


class TestProtocolV5ContextResolution:
    @pytest.mark.parametrize("field", [f for f in fc.CONTEXT_FIELDS if f.startswith("engine.") and f != "engine.engine_pin"])
    @pytest.mark.parametrize("typed", [False, True])
    def test_anchor_every_parent_named_engine_field_is_enforced_on_every_row(self, monkeypatch, tmp_path, field, typed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        anchor = fc.engine_identity(R5_PIN, R9_CONFIG); launch = _r5_mint(tmp_path, expected_engine=anchor); _r9_emit(tmp_path)
        events = ["transition-lowering", "interpreter-ran"] if typed else ["fast-ran"]
        for ev in events: _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: _r9_set(r, field, _r9_other(r, field)))
        for killed in (False, True):
            s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
            assert info["state"] == "NO_VALID_EVENT" and s["non_pass"] is True and {r["code"] for r in info["rejected_events"]} <= {"CONTEXT_MISMATCH", "ENGINE_HASH_MISMATCH"}, (field, info)
            assert all(any(x.startswith(("CONTEXT_MISMATCH:" + field, "ENGINE_HASH_MISMATCH:" + field)) for x in r["reasons"]) for r in info["rejected_events"]), info["rejected_events"]
            assert (tmp_path / "fallback-kill-record.json").exists() is killed
            if killed: (tmp_path / "fallback-kill-record.json").rename(tmp_path / "stale-fallback-kill-record.json")
        assert s["engine_context"]["anchored"]["engine." + field.split(".")[1]] == anchor[field.split(".")[1]]

    @pytest.mark.parametrize("field", [f for f in fc.CONTEXT_FIELDS if f not in ("engine.engine_pin", "source_sha256")])
    @pytest.mark.parametrize("killed", [False, True])
    def test_learned_the_first_valid_record_fixes_the_field_for_every_later_record_across_cases(self, monkeypatch, tmp_path, field, killed):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, cases=["A", "B"]); _r9_emit(tmp_path, "A"); _r9_emit(tmp_path, "B")
        _r5_mutate(tmp_path, "B-fallback-fast-ran.json", lambda r: _r9_set(r, field, _r9_other(r, field)))
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["tags"]["A-"]["state"] == "FAST_DONE" and s["engine_context"]["learned_from"].get(field, "").endswith("A-fallback-fast-ran.json"), s["engine_context"]
        b = s["tags"]["B-"]; assert b["state"] == "NO_VALID_EVENT" and s["case_coverage"]["B"] == "MISSING_CASE" and s["non_pass"] is True
        assert b["rejected_events"][0]["code"] in ("CONTEXT_MISMATCH", "ENGINE_HASH_MISMATCH", "IDENTITY_MISMATCH") and (tmp_path / "B-fallback-kill-record.json").exists() is killed, b
        assert any(field in x for x in b["rejected_events"][0]["reasons"]), b["rejected_events"]

    def test_free_before_first_the_learner_itself_passes_and_its_values_become_the_context(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path); _r9_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-fast-ran.json", lambda r: (_r9_set(r, "engine.python", "3.12.0"), _r9_set(r, "decline_hook_available", False)))
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["state"] == "FAST_DONE" and s["engine_context"]["learned"]["engine.python"] == "3.12.0" and s["engine_context"]["learned"]["decline_hook_available"] is False and "engine.python" not in s["engine_context"]["anchored"]

    @pytest.mark.parametrize("typed", [False, True])
    def test_partial_parent_anchor_leaves_unnamed_keys_unspecified(self, monkeypatch, tmp_path, typed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, expected_engine={"config": R9_CONFIG}); _r9_emit(tmp_path); s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is False and s["rejected_events"] == [] and set(s["engine_context"]["anchored"]) == {"engine.engine_pin", "source_sha256", "engine.config"}
        assert {"engine.python", "engine.numba_accel_sha256", "decline_hook_available", "binding.out_dir"} <= set(s["engine_context"]["learned"])
        d2 = tmp_path / "bad"; d2.mkdir(); launch2 = _r5_mint(d2, expected_engine={"config": {"use_fp64": False, "boundscheck": False}}); _r9_emit(d2); s2 = _r5_post(d2, launch2)
        assert s2["non_pass"] is True and all(r["code"] == "CONTEXT_MISMATCH" and any(x.startswith("CONTEXT_MISMATCH:engine.config") for x in r["reasons"]) for r in s2["rejected_events"]) and s2["rejected_events"]

    @pytest.mark.parametrize("typed", [False, True])
    @pytest.mark.parametrize("killed", [False, True])
    def test_outdir_canonical_rule_rejects_an_unrelated_directory_and_accepts_an_alias(self, monkeypatch, tmp_path, typed, killed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path); _r9_emit(tmp_path)
        events = ["transition-lowering", "interpreter-ran"] if typed else ["fast-ran"]
        for ev in events: _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["binding"].__setitem__("out_dir", str(tmp_path / "not-the-minted-run")))   # minted run_dir retained
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][""]
        assert s["non_pass"] is True and info["state"] == "NO_VALID_EVENT" and all(r["code"] == "IDENTITY_MISMATCH" and any("canonicalises" in x for x in r["reasons"]) for r in info["rejected_events"]) and info["rejected_events"]
        assert (tmp_path / "fallback-kill-record.json").exists() is killed
        alias = str(tmp_path) + "/./"                                                  # a spelling alias of the same directory, used by EVERY record of the attempt
        for ev in events: _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["binding"].__setitem__("out_dir", alias))
        if killed: (tmp_path / "fallback-kill-record.json").rename(tmp_path / "stale-fallback-kill-record.json")
        s = _r5_post(tmp_path, launch); info = s["tags"][""]
        assert info["rejected_events"] == [] and info["state"] == ("CLOSED_OK" if typed else "FAST_DONE") and s["engine_context"]["learned"]["binding.out_dir"] == alias

    def test_a_later_record_with_a_different_spelling_is_a_context_mismatch(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, cases=["A", "B"]); _r9_emit(tmp_path, "A"); _r9_emit(tmp_path, "B")
        _r5_mutate(tmp_path, "B-fallback-fast-ran.json", lambda r: r["binding"].__setitem__("out_dir", str(tmp_path) + "/./"))
        s = _r5_post(tmp_path, launch); b = s["tags"]["B-"]
        assert b["state"] == "NO_VALID_EVENT" and b["rejected_events"][0]["code"] == "CONTEXT_MISMATCH" and any(x.startswith("CONTEXT_MISMATCH:binding.out_dir") for x in b["rejected_events"][0]["reasons"])

    def test_expected_fast_record_is_built_from_the_resolved_context_not_the_record(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        anchor = fc.engine_identity(R5_PIN, R9_CONFIG); launch = _r5_mint(tmp_path, expected_engine=anchor); _r9_emit(tmp_path)
        rec = _json.loads((tmp_path / "fallback-fast-ran.json").read_text()); ctx = fc.AttemptContext(launch)
        assert ctx.engine(rec) == anchor and ctx.value("source_sha256", rec) == R5_SHA and not ctx.resolved("decline_hook_available")
        ctx.learn(rec, "p"); assert ctx.resolved("decline_hook_available") and ctx.value("binding.out_dir", rec) == rec["binding"]["out_dir"]
        bad = _json.loads(_json.dumps(rec)); bad["engine"]["python"] = "3.12.0"
        assert ctx.mismatches(bad) == [("CONTEXT_MISMATCH", "engine.python")] and ctx.engine(bad) == anchor      # the resolved context never takes the child's value

    def test_transition_contradicting_the_parent_anchor_is_rejected_before_it_can_be_copied_into_a_closing(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); anchor = fc.engine_identity(R5_PIN, R9_CONFIG); launch = _r5_mint(tmp_path, expected_engine=anchor); _r9_emit(tmp_path)
        for ev in ("transition-lowering", "interpreter-ran"): _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["engine"].__setitem__("executable", "/different/python"))
        s = _r5_post(tmp_path, launch, killed=True); info = s["tags"][""]
        assert info["state"] == "NO_VALID_EVENT" and [r["code"] for r in info["rejected_events"]] == ["CONTEXT_MISMATCH", "CONTEXT_MISMATCH"] and (tmp_path / "fallback-kill-record.json").exists() and not (tmp_path / "fallback-resource-stop.json").exists()


class TestProtocolV5Review6Round8Probes:
    """the reviewer's round-8 probe file test_new_v8.py restated with this file's helpers (the sealed originals are run unmodified as well)."""

    @pytest.mark.parametrize("field", ["python", "executable", "solar_package_path", "config"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_parent_engine_anchor_applies_to_interpreter_transition_and_terminal(self, monkeypatch, tmp_path, field, killed):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path, expected_engine=fc.engine_identity(R5_PIN, R9_CONFIG)); _r9_emit(tmp_path)
        for ev in ("transition-lowering", "interpreter-ran"): _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["engine"].__setitem__(field, R9_ENGINE_MUT[field]))
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["rejected_events"] and s["non_pass"] is True and (tmp_path / "fallback-kill-record.json").exists() is killed

    @pytest.mark.parametrize("field", ["python", "executable", "solar_package_path", "config"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_fast_engine_fields_fixed_by_prior_valid_case(self, monkeypatch, tmp_path, field, killed):
        _r5_engine(monkeypatch, typed=False, lowered=R5_SRC); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, cases=["A", "B"]); _r9_emit(tmp_path, "A"); _r9_emit(tmp_path, "B")
        _r5_mutate(tmp_path, "B-fallback-fast-ran.json", lambda r: r["engine"].__setitem__(field, R9_ENGINE_MUT[field]))
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["rejected_events"] and s["non_pass"] is True and (tmp_path / "B-fallback-kill-record.json").exists() is killed

    @pytest.mark.parametrize("typed", [False, True])
    def test_partial_expected_engine_anchors_only_named_fields(self, monkeypatch, tmp_path, typed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, expected_engine={"config": R9_CONFIG}); _r9_emit(tmp_path); s = _r5_post(tmp_path, launch)
        assert not s["non_pass"] and not s["rejected_events"]

    @pytest.mark.parametrize("typed", [False, True])
    @pytest.mark.parametrize("killed", [False, True])
    def test_outdir_canonical_identity_is_checked(self, monkeypatch, tmp_path, typed, killed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path); _r9_emit(tmp_path)
        for ev in (["transition-lowering", "interpreter-ran"] if typed else ["fast-ran"]): _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["binding"].__setitem__("out_dir", str(tmp_path / "not-the-minted-run")))
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["non_pass"] and s["rejected_events"] and (tmp_path / "fallback-kill-record.json").exists() is killed

    @pytest.mark.parametrize("typed", [False, True])
    def test_full_parent_engine_positive_control(self, monkeypatch, tmp_path, typed):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        launch = _r5_mint(tmp_path, expected_engine=fc.engine_identity(R5_PIN, R9_CONFIG)); _r9_emit(tmp_path); s = _r5_post(tmp_path, launch)
        assert not s["non_pass"] and s["verdict"] is None

    def test_unlisted_additions_and_shared_constructor_and_fabrication_limit(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path); _r9_emit(tmp_path)
        _r5_mutate(tmp_path, "fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("extra", "injected"))
        s = _r5_post(tmp_path, launch, killed=True); assert s["non_pass"] and s["rejected_events"] and (tmp_path / "fallback-resource-stop.json").exists()
        d = tmp_path / "fab"; d.mkdir(); launch2 = _r5_mint(d)
        fc.lower_or_fallback(R5_SRC, engine_pin=R5_PIN, binding=fc.attempt_binding(d), persist=_r5_persist(d), config=R9_CONFIG)
        fake = fc.expected_closing_record(_json.loads((d / "fallback-transition-lowering.json").read_text()), None, "interpreter-ran", {"event_utc": fc._utc()})
        (d / "fallback-interpreter-ran.json").write_text(_json.dumps(fake)); s = _r5_post(d, launch2)
        assert not s["non_pass"] and s["verdict"] is None       # the acknowledged consistency-vs-attestation limit: bytes built from the correctly anchored context


# =====================================================================================================================
# round 10 -- PROTOCOL v6: admission boundary (two-phase resolution to a fixed point), type-exact context equality,
# anchor domain; the reviewer's round-9 probes.
# =====================================================================================================================

def _r10_two_cases(monkeypatch, out, typed=True, config=R9_CONFIG):
    _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
    launch = _r5_mint(out, cases=["A", "B"]); _r9_emit(out, "A", config=config); _r9_emit(out, "B", config=config)
    return launch


class TestProtocolV6AdmissionBoundary:
    @pytest.mark.parametrize("killed", [False, True])
    def test_rejected_orphan_first_record_cannot_poison_a_later_valid_transition(self, monkeypatch, tmp_path, killed):
        launch = _r10_two_cases(monkeypatch, tmp_path)
        (tmp_path / "A-fallback-transition-lowering.json").unlink(); (tmp_path / "B-fallback-interpreter-ran.json").unlink()
        _r5_mutate(tmp_path, "A-fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("python", "FORGED-ORPHAN"))
        s = _r5_post(tmp_path, launch, killed=killed); a, b = s["tags"]["A-"], s["tags"]["B-"]
        assert a["rejected_events"] and a["state"] in ("STATE_CONFLICT", "NO_VALID_EVENT") and s["non_pass"] is True
        assert b["surviving_transition"] == "transition-lowering" and b["state"] == "OPEN" and b["rejected_events"] == []
        assert (tmp_path / ("B-fallback-resource-stop.json" if killed else "B-fallback-open-transition.json")).exists() and not (tmp_path / "B-fallback-kill-record.json").exists()
        ec = s["engine_context"]; assert ec["learned"]["engine.python"] != "FORGED-ORPHAN" and ec["committed_from"]["engine.python"].endswith("B-fallback-transition-lowering.json") and ec["fixed_point"] is True
        assert ec["tentative_discarded"]["engine.python"]["value"] == "FORGED-ORPHAN" and ec["tentative_discarded"]["engine.python"]["from"].endswith("A-fallback-interpreter-ran.json")
        assert len(ec["passes"]) >= 2 and ec["passes"][0]["mode"] == "tentative" and ec["passes"][-1]["mode"] == "frozen"

    @pytest.mark.parametrize("kind", ["fast-ran", "interpreter-ran", "interpreter-failed"])
    def test_without_a_rejected_learner_the_builder_is_invoked_once_by_the_validator(self, monkeypatch, tmp_path, kind):
        launch, path = _r8_make(monkeypatch, tmp_path, kind); calls = []; original = fc.expected_closing_record
        monkeypatch.setattr(fc, "expected_closing_record", lambda *a, **k: (calls.append(a[2]), original(*a, **k))[1])
        s = _r5_post(tmp_path, launch)
        assert calls == [kind] and s["engine_context"]["fixed_point"] is True and len(s["engine_context"]["passes"]) == 1 and "final" in s["engine_context"]["passes"][0]

    @pytest.mark.parametrize("killed", [False, True])
    def test_valid_first_then_rejected_orphan_commits_only_from_the_valid_one(self, monkeypatch, tmp_path, killed):
        launch = _r10_two_cases(monkeypatch, tmp_path)
        (tmp_path / "B-fallback-transition-lowering.json").unlink()                                   # B's terminal is the orphan, A is fully valid
        _r5_mutate(tmp_path, "B-fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("python", "FORGED-ORPHAN"))
        s = _r5_post(tmp_path, launch, killed=killed); a, b = s["tags"]["A-"], s["tags"]["B-"]
        assert a["state"] == "CLOSED_OK" and a["rejected_events"] == [] and b["rejected_events"] and b["state"] in ("STATE_CONFLICT", "NO_VALID_EVENT")
        assert s["engine_context"]["committed_from"]["engine.python"].endswith("A-fallback-transition-lowering.json") and (tmp_path / "B-fallback-kill-record.json").exists() is killed

    def test_multi_case_orphans_in_parent_and_case_a_do_not_fix_case_b(self, monkeypatch, tmp_path):
        _r5_engine(monkeypatch); launch = _r5_mint(tmp_path, cases=["A", "B"]); _r9_emit(tmp_path); _r9_emit(tmp_path, "A"); _r9_emit(tmp_path, "B")
        for pre in ("", "A-"):
            (tmp_path / (pre + "fallback-transition-lowering.json")).unlink(); _r5_mutate(tmp_path, pre + "fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("executable", "/forged/" + (pre or "parent")))
        s = _r5_post(tmp_path, launch)
        assert s["tags"][""]["rejected_events"] and s["tags"]["A-"]["rejected_events"] and s["tags"]["B-"]["state"] == "CLOSED_OK" and s["tags"]["B-"]["rejected_events"] == []
        assert s["engine_context"]["committed_from"]["engine.executable"].endswith("B-fallback-transition-lowering.json") and s["case_coverage"] == {"A": "MISSING_CASE", "B": "CLOSED_OK"}

    def test_resolution_is_deterministic_and_written_only_from_the_final_classification(self, monkeypatch, tmp_path):
        launch = _r10_two_cases(monkeypatch, tmp_path)
        (tmp_path / "A-fallback-transition-lowering.json").unlink(); (tmp_path / "B-fallback-interpreter-ran.json").unlink()
        _r5_mutate(tmp_path, "A-fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("python", "FORGED-ORPHAN"))
        s1 = _r5_post(tmp_path, launch); (tmp_path / "B-fallback-open-transition.json").rename(tmp_path / "stale-B-fallback-open-transition.json")
        s2 = _r5_post(tmp_path, launch)
        strip = lambda s: {k: v for k, v in s.items() if k not in ("postprocess_utc", "written")}
        assert strip(s1)["engine_context"] == strip(s2)["engine_context"] and {t: i["state"] for t, i in s1["tags"].items()} == {t: i["state"] for t, i in s2["tags"].items()}
        assert all(not p.name.startswith("A-fallback-open") for p in tmp_path.iterdir())   # nothing written for the rejected orphan tag in any phase

    @pytest.mark.parametrize("bad", ["schema", "filename", "state", "scope", "identity", "premint", "crossfield", "deep_extra", "future"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_individually_rejected_first_fast_record_does_not_seed(self, monkeypatch, tmp_path, bad, killed):
        launch = _r10_two_cases(monkeypatch, tmp_path, typed=False)
        def mut(r):
            r["engine"]["python"] = "FORGED-FIRST"
            if bad == "schema": r["events"] = 42
            elif bad == "filename": r["event"] = "interpreter-ran"
            elif bad == "state": r["outcome"] = "invalid"
            elif bad == "scope": r["binding"]["case"] = "undeclared"
            elif bad == "identity": r["binding"]["attempt"] = 2
            elif bad in ("premint", "future"):
                stamp = (datetime_mod.datetime.now(datetime_mod.timezone.utc) + datetime_mod.timedelta(seconds=-600 if bad == "premint" else 120)).isoformat(); r["event_utc"] = stamp; r["events"][0]["event_utc"] = stamp
            elif bad == "crossfield": r["fallback_taken"] = True
            elif bad == "deep_extra": r["engine"]["extra"] = "unexpected"
        _r5_mutate(tmp_path, "A-fallback-fast-ran.json", mut)
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["tags"]["A-"]["rejected_events"] and s["non_pass"] is True and s["tags"]["B-"]["state"] == "FAST_DONE" and not s["tags"]["B-"]["rejected_events"]
        assert s["engine_context"]["learned"]["engine.python"] != "FORGED-FIRST" and (tmp_path / "A-fallback-kill-record.json").exists() is killed


_TYPE_CHANGES = {"engine.config": lambda v: {**v, "use_fp64": 1} if isinstance(v, dict) and "use_fp64" in v else {"nested": {"choices": [1, 0]}},
                 "decline_hook_available": lambda v: 1 if v is True else 0,
                 "engine.python": lambda v: 3.1, "engine.executable": lambda v: ["/usr/bin/python"], "engine.solar_package_path": lambda v: {"path": v}}


class TestProtocolV6TypeExactContext:
    @pytest.mark.parametrize("kind", ["parent", "learned"])
    @pytest.mark.parametrize("variant", ["bool_to_int", "nested_bool_to_int"])
    @pytest.mark.parametrize("killed", [False, True])
    def test_config_json_type_change_is_a_context_mismatch(self, monkeypatch, tmp_path, kind, variant, killed):
        cfg = R9_CONFIG if variant == "bool_to_int" else {"nested": {"choices": [True, False]}}
        _r5_engine(monkeypatch)
        if kind == "parent": launch = _r5_mint(tmp_path, expected_engine={"config": cfg}); tag = ""; _r9_emit(tmp_path, config=cfg)
        else: launch = _r5_mint(tmp_path, cases=["A", "B"]); tag = "B-"; _r9_emit(tmp_path, "A", config=cfg); _r9_emit(tmp_path, "B", config=cfg)
        changed = _json.loads(_json.dumps(cfg))
        if variant == "bool_to_int": changed["use_fp64"] = 1
        else: changed["nested"]["choices"] = [1, 0]
        for ev in ("transition-lowering", "interpreter-ran"): _r5_mutate(tmp_path, tag + "fallback-" + ev + ".json", lambda r: r["engine"].__setitem__("config", changed))
        s = _r5_post(tmp_path, launch, killed=killed); info = s["tags"][tag]
        assert info["rejected_events"] and all(r["code"] == "CONTEXT_MISMATCH" and any(x.startswith("CONTEXT_MISMATCH:engine.config") for x in r["reasons"]) for r in info["rejected_events"]) and info["state"] == "NO_VALID_EVENT" and s["non_pass"] is True
        assert (tmp_path / (tag + "fallback-kill-record.json")).exists() is killed

    @pytest.mark.parametrize("field", sorted(_TYPE_CHANGES))
    def test_every_typed_context_field_is_compared_type_exactly(self, monkeypatch, tmp_path, field):
        launch = _r10_two_cases(monkeypatch, tmp_path)
        for ev in ("transition-lowering", "interpreter-ran"): _r5_mutate(tmp_path, "B-fallback-" + ev + ".json", lambda r: _r9_set(r, field, _TYPE_CHANGES[field](_path_get(r, field))))
        s = _r5_post(tmp_path, launch); b = s["tags"]["B-"]
        assert b["rejected_events"] and b["state"] == "NO_VALID_EVENT" and s["tags"]["A-"]["state"] == "CLOSED_OK"
        assert all(r["code"] in ("CONTEXT_MISMATCH", "SCHEMA") for r in b["rejected_events"]), b["rejected_events"]   # a type change the schema already refuses is SCHEMA; otherwise CONTEXT_MISMATCH

    def test_deep_diff_is_the_equality_rule(self):
        assert fc._deep_diff(True, 1, "x") == ["x"] and fc._deep_diff([True, False], [1, 0], "l") == ["l"] and fc._deep_diff({"a": {"b": True}}, {"a": {"b": 1}}, "") == ["a.b"] and fc._deep_diff(1, 1.0, "n") == ["n"] and fc._deep_diff({"a": [1, {"b": "c"}]}, {"a": [1, {"b": "c"}]}, "") == []


def _path_get(rec, path):
    o = rec
    for part in path.split("."):
        o = o.get(part) if isinstance(o, dict) else None
    return o


class TestProtocolV6AnchorDomain:
    @pytest.mark.parametrize("extra", [{"deployment_id": "PARENT-NAMED"}, {"python": "3.10.4", "cluster": "x"}])
    def test_unsupported_expected_engine_key_is_refused_at_descriptor_and_mint(self, tmp_path, extra):
        with pytest.raises(fc.RecordInvalid, match="unsupported expected_engine key"):
            fc.LaunchDescriptor(run_dir=str(tmp_path), out_dir=str(tmp_path), engine_pin=R5_PIN, expected_engine=extra)
        with pytest.raises(fc.RecordInvalid, match="unsupported expected_engine key"):
            _r5_mint(tmp_path, expected_engine=extra)
        assert not (tmp_path / fc.MINT_FILE).exists()                                          # refused BEFORE anything is written

    def test_named_engine_pin_must_match_and_every_enumerated_key_is_accepted(self, monkeypatch, tmp_path):
        with pytest.raises(fc.RecordInvalid, match="engine_pin"):
            _r5_mint(tmp_path, expected_engine={"engine_pin": "other"})
        full = fc.engine_identity(R5_PIN, R9_CONFIG); assert sorted(full) == sorted(fc.ANCHOR_DOMAIN)
        launch = _r5_mint(tmp_path / "ok", expected_engine=full); _r5_engine(monkeypatch); _r9_emit(tmp_path / "ok")
        s = _r5_post(tmp_path / "ok", launch); assert s["non_pass"] is False and set(s["engine_context"]["anchored"]) >= {"engine." + k for k in fc.ANCHOR_DOMAIN}


class TestProtocolV6Review6Round9Probes:
    """the reviewer's round-9 probe file test_new_v9.py restated with this file's helpers (the sealed originals are run unmodified as well)."""

    @pytest.mark.parametrize("killed", [False, True])
    def test_rejected_orphan_first_record_cannot_poison_later_transition(self, monkeypatch, tmp_path, killed):
        launch = _r10_two_cases(monkeypatch, tmp_path)
        (tmp_path / "A-fallback-transition-lowering.json").unlink(); (tmp_path / "B-fallback-interpreter-ran.json").unlink()
        _r5_mutate(tmp_path, "A-fallback-interpreter-ran.json", lambda r: r["engine"].__setitem__("python", "FORGED-ORPHAN"))
        s = _r5_post(tmp_path, launch, killed=killed)
        assert s["tags"]["A-"]["rejected_events"] and s["non_pass"] and s["tags"]["B-"]["surviving_transition"] == "transition-lowering"
        assert (tmp_path / ("B-fallback-resource-stop.json" if killed else "B-fallback-open-transition.json")).exists() and s["engine_context"]["learned"]["engine.python"] != "FORGED-ORPHAN"

    @pytest.mark.parametrize("typed", [False, True])
    def test_every_parent_named_engine_key_is_enforced_or_rejected(self, tmp_path, typed):
        with pytest.raises(fc.RecordInvalid):
            _r5_mint(tmp_path, expected_engine={"deployment_id": "PARENT-NAMED"})

    @pytest.mark.parametrize("typed", [False, True])
    @pytest.mark.parametrize("anchored", [False, True])
    def test_fabricated_first_context_is_an_explicit_non_attestation_limit(self, monkeypatch, tmp_path, typed, anchored):
        _r5_engine(monkeypatch, typed=typed, lowered=(None if typed else R5_SRC)); monkeypatch.setattr(numba_accel, "execution_namespace", lambda use_fp64=True: {})
        actual = fc.engine_identity(R5_PIN, R9_CONFIG); launch = _r5_mint(tmp_path, **({"expected_engine": actual} if anchored else {})); _r9_emit(tmp_path)
        for ev in (["transition-lowering", "interpreter-ran"] if typed else ["fast-ran"]):
            _r5_mutate(tmp_path, "fallback-" + ev + ".json", lambda r: r["engine"].update({"python": "3.12.0-UNOBSERVED", "executable": "/unobserved/python", "solar_package_path": "/unobserved/solar", "numba_accel_sha256": "a" * 64, "verify_sha256": "b" * 64, "core_sha256": "c" * 64, "config": {"unobserved": True}}))
        s = _r5_post(tmp_path, launch)
        assert s["non_pass"] is anchored and bool(s["rejected_events"]) is anchored and s["verdict"] is None
        if not anchored: assert s["engine_context"]["learned"]["engine.python"] == "3.12.0-UNOBSERVED"     # the DISCLOSED consistency-only limit, kept stated


# =====================================================================================================================
# round 11 -- PROTOCOL v6 wording-1 (P3-R10-FIXED-POINT-WORDING-1): the shortcut keeps identical DISPOSITIONS; diagnostics
# of records rejected in the tentative evaluation may name a different code than a frozen pass would.
# =====================================================================================================================

class TestProtocolV6FixedPointWording:
    def test_shortcut_dispositions_equal_a_frozen_pass_while_diagnostics_may_differ(self, monkeypatch, tmp_path):
        """the reviewer's fixture: A's fast record has fallback_taken=True (I3) AND engine.python='BAD-FIRST'; B is valid and commits the context."""
        launch = _r10_two_cases(monkeypatch, tmp_path, typed=False)
        _r5_mutate(tmp_path, "A-fallback-fast-ran.json", lambda r: (r["engine"].__setitem__("python", "BAD-FIRST"), r.__setitem__("fallback_taken", True)))
        s = _r5_post(tmp_path, launch); ec = s["engine_context"]
        assert ec["fixed_point"] is True and len(ec["passes"]) == 1 and "dispositions" in ec["passes"][0]["final"]       # the shortcut applied
        assert s["tags"]["B-"]["state"] == "FAST_DONE" and ec["committed_from"]["engine.python"].endswith("B-fallback-fast-ran.json") and ec["learned"]["engine.python"] != "BAD-FIRST"
        tentative_codes = [(r["event"], r["code"]) for r in s["tags"]["A-"]["rejected_events"]]
        assert tentative_codes == [("fast-ran", "TERMINAL_INCONSISTENT")] and "I3" in _ids(s["tags"]["A-"]["rejected_events"])
        # a frozen re-validation against the COMMITTED context: identical dispositions, different diagnostic for A
        frozen = fc.AttemptContext(launch, frozen=True); frozen.learned.update(_json.loads(_json.dumps(ec["learned"]))); frozen.learned_from.update(ec["learned_from"])
        events = fc.scan_fallback_events(tmp_path)
        for tag in sorted(events, key=lambda t: (t != "", t)):
            r = fc._classify_tag(tag, events[tag], launch, frozen, tentative=False); info = s["tags"][tag]
            for k in ("state", "valid_events", "closing_event", "surviving_transition", "scope"):
                assert r["info"][k] == info[k], (tag, k, r["info"][k], info[k])
            assert [ev for ev, _, _ in r["survivors"]] == sorted(info["valid_events"], key=lambda e: (e not in fc.TRANSITION_EVENTS, e))
        frozen_a = fc._classify_tag("A-", events["A-"], launch, frozen, tentative=False)["rejected"]
        assert [(r["event"], r["code"]) for r in frozen_a] == [("fast-ran", "CONTEXT_MISMATCH")] and any("engine.python" in x for x in frozen_a[0]["reasons"])   # the documented diagnostic difference
