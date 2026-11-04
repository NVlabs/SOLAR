# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-side contract for a FAST (Numba) engine decline: a TYPED decline -> interpreter execution of the SAME
program on fresh copies of the SAME prepared inputs, the caller's comparator unchanged, every transition persisted
BEFORE the interpreter runs with the engine identity, the event time and the attempted backend bound into it, and an
outer post-processor that joins a surviving transition to the controller's RESOURCE_STOP.

Why this module exists (the fallback-contract task, 2026-10-28; rounds 2 and 3 after the reviewer's HOLDs).  The
selected Lite validation harnesses never go through ``TranslationAgent.translate``: nineteen call ``numba_lower``
directly and ``assert lowered is not None`` ("no interpreter fallback"), one (row 119) calls
``execute_translated(fast=True, fail_on_numba_decline=True)``.  A typed decline of the fast engine therefore fails those
harnesses closed with the reason in a traceback and no receipt field.  The entry-point commit that routes an agent's
typed decline to the interpreter (``TranslationAgent(reprompt_on_numba_decline=False)``) does not reach them.

Contract (``lite-fallback-contract-v10``; PROTOCOL v2, superseded by the tracked fallback_protocol/PROTOCOL.md: per-scope state machine, schema-generated validator, typed coverage; this module implements it):

* ONLY a TYPED decline reroutes.  A typed decline is ``numba_lower`` returning ``None`` and reporting a reason through
  its ``on_decline`` hook, or ``KernelOutputDtypeUnresolved`` raised while the LOWERED module is imported or run.
  ``numba_lower`` returning ``None`` with NO reason (an engine pin without the hook, e.g. an earlier revision, or a silent hook)
  is an UNTYPED decline: NO reroute; it is a visible non-pass of the fast stage (:class:`UntypedDecline`, record
  outcome ``UNTYPED_DECLINE``).  Consequence: rows whose accepted engine pin is an earlier revision cannot gain the fallback on
  that pin by this shim alone; a new engine binding for them is a separately admitted and reviewed step.  Numba being
  absent is an environment failure (:class:`NumbaUnavailable`), never a decline and never a fallback.
* Every persisted record is SELF-BOUND: ``event`` name, ``event_utc`` (actual UTC at persistence), ``engine`` (the
  caller's engine pin plus the imported engine's package path and the sha256 of its ``numba_accel.py`` / ``verify.py``
  / ``core.py``), ``binding`` (the caller's attempt binding: run directory, row, case, ...), ``backend_attempt`` (the
  backend about to run or that ran: ``interpreter`` on every transition), ``backend_terminal`` (the backend that
  produced the compared numbers; set only by ``fast-ran`` / ``interpreter-ran``), the mandatory ``source_sha256`` and
  the nullable ``lowered_sha256`` (null for a lowering-stage decline).  :func:`validate_record` checks the required
  fields and is applied before every persistence.
* A TRANSITION EVENT is persisted BEFORE any interpreter work (``persist("transition-<stage>", record)``: the typed
  decline reason, the stage ``lowering`` | ``module_exec`` | ``run_model``, "interpreter attempt starting").  If the
  interpreter attempt then fails, a DISTINCT typed record is persisted (``interpreter-failed``, outcome
  ``interpreter_failed:<Type>: <message>``) carrying the prior decline, and :class:`InterpreterFailed` is raised.  A
  resource kill runs no cleanup: the persisted transition is the retained evidence, and the controller's
  :func:`postprocess_attempt` reads it after the child stopped and writes the DISTINCT joined ``RESOURCE_STOP`` record
  (prior typed decline, attempt binding, null verdict); a transition without a terminal event is always NON-PASS.
  No record here is ever a pass (``verdict`` is always ``None``); the caller's comparator decides on returned outputs.
* The interpreter re-run receives FRESH arrays.  A caller hands either a zero-argument factory that returns fresh
  arrays (the direct harnesses' ``consumer_inputs`` / explicit ``np.array(copy=True)``) or a :class:`PreparedInputs`
  baseline: deep copies made once, held read-only, hashed; every attempt gets value-preserving writable copies and
  the baseline is re-verified after each attempt.
* Anything that is not a typed decline -- ``IndexError`` from ``boundscheck``, a ``ValueError`` in the candidate, a
  missing ``run_model`` -- propagates unchanged: a candidate error is not an engine decline and takes no fallback.

This module performs no comparison, spawns nothing, imports the engine lazily so a harness can pin ``solar`` on
``sys.path`` first, and writes files only through the caller's ``persist`` callback (and, for the controller-side
:func:`postprocess_attempt`, into the attempt's own run directory).  It uses only ``numba_accel.HAS_NUMBA`` /
``numba_lower`` / ``execution_namespace`` (and ``KernelOutputDtypeUnresolved`` where the pin has it) and
``verify._MATH_BUILTINS``, all present at five earlier revisions, so the file can also be
copied beside a harness unchanged (absolute imports only).
"""
from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

CONTRACT = "lite-fallback-contract-v10"
BACKEND_NUMBA = "numba"
BACKEND_INTERPRETER = "interpreter"
STAGE_LOWERING = "lowering"
STAGE_MODULE_EXEC = "module_exec"
STAGE_RUN_MODEL = "run_model"

# record["outcome"] values -- process states, never verdicts
OUTCOME_FAST_LOWERED = "FAST_LOWERED"
OUTCOME_FAST_RAN = "FAST_RAN"
OUTCOME_UNTYPED_DECLINE = "UNTYPED_DECLINE"
OUTCOME_NUMBA_UNAVAILABLE = "NUMBA_UNAVAILABLE"
OUTCOME_INTERPRETER_PENDING = "INTERPRETER_PENDING"
OUTCOME_INTERPRETER_RAN = "INTERPRETER_RAN"
OUTCOME_INTERPRETER_FAILED = "interpreter_failed"     # prefix: interpreter_failed:<Type>: <message>
OUTCOME_RESOURCE_STOP = "RESOURCE_STOP"
OUTCOME_OPEN_TRANSITION = "NON_PASS_OPEN_TRANSITION"  # transition persisted, no terminal event, child not guard-killed
OUTCOME_PRECHECK_DECLINE = "PRECHECK_DECLINE_NO_EXECUTION"  # non-executing preparation (row 119 lowering precheck): never opens an execution transition

TERMINAL_EVENTS = ("fast-ran", "interpreter-ran", "interpreter-failed", "untyped-decline", "numba-unavailable")
PREPARATION_EVENTS = ("precheck-decline",)              # preparation records: neither a transition nor a terminal
EXPECTED_OUTCOME = {"fast-ran": OUTCOME_FAST_RAN, "interpreter-ran": OUTCOME_INTERPRETER_RAN, "untyped-decline": OUTCOME_UNTYPED_DECLINE, "numba-unavailable": OUTCOME_NUMBA_UNAVAILABLE, "precheck-decline": OUTCOME_PRECHECK_DECLINE}
ATTEMPT_IDENTITY_FIELDS = ("group", "attempt", "row")   # compared on BOTH sides: present-and-equal on both, or absent on both; present on one side only is a MISMATCH
MINT_FILE = "fallback-attempt.json"                      # written by the controller BEFORE launch; read by attempt_binding() in the child
COMPLETE_EVENTS = ("fast-ran", "interpreter-ran")     # the comparator decides on these; still no verdict here

# Required on EVERY persisted record (the review P2-R2-TRANSITION-BINDING-1): engine identity, event UTC, attempted backend,
# mandatory original-source hash (lowered hash may be null), typed decline fields, null verdict.
REQUIRED_RECORD_FIELDS = ("contract", "event", "event_utc", "engine", "binding", "backend", "backend_attempt", "backend_terminal", "executed_program", "fallback_allowed", "fallback_taken",
                          "source_sha256", "lowered_sha256", "program_text_sha256", "outcome", "decline_reason", "decline_typed", "decline_stage", "decline_hook_available", "interpreter_failure", "preparation_only", "history", "events", "verdict")
REQUIRED_ENGINE_FIELDS = ("engine_pin", "solar_package_path", "numba_accel_sha256", "verify_sha256", "core_sha256", "python", "executable")

Persist = Callable[[str, dict], None]


class FastStageFailure(RuntimeError):
    """Base: a failure of this contract that carries its persisted typed record (``.record``)."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


class UntypedDecline(FastStageFailure):
    """``numba_lower`` returned None without a typed reason: visible non-pass, no interpreter."""


class NumbaUnavailable(FastStageFailure):
    """Numba is not importable: environment failure, not a typed engine decline, no interpreter."""


class InterpreterFailed(FastStageFailure):
    """A typed decline was followed by a failing interpreter attempt; ``.record`` carries both."""


class RecordInvalid(RuntimeError):
    """A record misses a required binding field (never persisted)."""


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha256_file(p) -> Optional[str]:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def _engine():
    from solar.einsum_lib import numba_accel  # lazy: the caller pins solar on sys.path first
    return numba_accel


def engine_identity(engine_pin: str, config: Optional[dict] = None) -> dict:
    """The engine actually imported, bound into every record: the caller's pin plus file hashes of the imported engine."""
    if not engine_pin or not isinstance(engine_pin, str):
        raise RecordInvalid("engine_pin (the caller's engine/config pin) is required for every fallback record")
    import solar
    from solar.einsum_lib import numba_accel, verify, core
    return {"engine_pin": engine_pin, "solar_package_path": str(Path(solar.__file__).resolve().parent),
            "numba_accel_sha256": _sha256_file(numba_accel.__file__), "verify_sha256": _sha256_file(verify.__file__), "core_sha256": _sha256_file(core.__file__),
            "python": sys.version.split()[0], "executable": sys.executable, "config": dict(config or {})}


_HEX64 = re.compile(r"[0-9a-f]{64}")
STAGES = (STAGE_LOWERING, STAGE_MODULE_EXEC, STAGE_RUN_MODEL)

# PROTOCOL v2 type_schema.record -- the SINGLE SOURCE OF TRUTH (a test asserts PROTOCOL.json carries the identical dict);
# validate_record_safe is GENERATED from it.
RECORD_SCHEMA = {
    "contract": {"type": "const", "value": CONTRACT},
    "event": {"type": "str", "nonempty": True},
    "event_utc": {"type": "utc"},
    "engine": {"type": "object", "fields": {"engine_pin": {"type": "str", "nonempty": True}, "solar_package_path": {"type": "str", "nonempty": True}, "numba_accel_sha256": {"type": "sha256"}, "verify_sha256": {"type": "sha256"}, "core_sha256": {"type": "sha256"}, "python": {"type": "str", "nonempty": True}, "executable": {"type": "str", "nonempty": True}, "config": {"type": "object", "fields": {}, "open": True}}},
    "binding": {"type": "object", "fields": {"out_dir": {"type": "str", "nonempty": True}, "run_dir": {"type": "str", "optional": True}, "group": {"type": "str", "optional": True}, "attempt": {"type": "int", "optional": True}, "row": {"type": "int", "optional": True}, "case": {"type": "str", "optional": True}, "stage": {"type": "str", "optional": True}}, "xor": ["case", "stage"]},
    "outcome": {"type": "str", "nonempty": True},
    "backend": {"type": "enum", "values": ["numba", "interpreter"]},
    "backend_attempt": {"type": "enum", "values": ["numba", "interpreter"]},
    "backend_terminal": {"type": "enum", "values": [None, "numba", "interpreter"]},
    "executed_program": {"type": "enum", "values": ["numba_lowered", "original_dsl_interpreter", "not_executed_preparation"]},
    "fallback_allowed": {"type": "bool"}, "fallback_taken": {"type": "bool"},
    "decline_reason": {"type": "str", "nullable": True}, "decline_typed": {"type": "bool"},
    "decline_stage": {"type": "enum", "values": [None, "lowering", "module_exec", "run_model"]},
    "decline_hook_available": {"type": "bool"},
    "interpreter_failure": {"type": "object", "nullable": True, "fields": {"type": {"type": "str", "nonempty": True}, "message": {"type": "str"}, "prior_decline_reason": {"type": "str", "nullable": True}, "prior_decline_stage": {"type": "enum", "values": [None, "lowering", "module_exec", "run_model"]}}},
    "preparation_only": {"type": "bool"},
    "source_sha256": {"type": "sha256"}, "lowered_sha256": {"type": "sha256", "nullable": True}, "program_text_sha256": {"type": "sha256"},
    "history": {"type": "list", "items": {"type": "object", "fields": {"backend": {"type": "enum", "values": ["numba", "interpreter"]}, "stage": {"type": "enum", "values": ["lowering", "module_exec", "run_model"]}, "reason": {"type": "str"}, "typed": {"type": "bool"}}}},
    "events": {"type": "list", "items": {"type": "object", "fields": {"event": {"type": "str", "nonempty": True}, "event_utc": {"type": "utc"}}}},
    "verdict": {"type": "const", "value": None},
}


# PROTOCOL v3 section 3 -- the cross-field (I1-I9) and cross-record (C1-C7) invariants, embedded VERBATIM from the tracked
# PROTOCOL.json (a test asserts equality).  _cross_field implements I1-I10; the re-emit comparison (_reemit_diff) covers the closing rows; a violation rejects the record
# (TERMINAL_INCONSISTENT for the closing rows, FIELD_INCONSISTENT otherwise) and a rejected record never closes anything.
INVARIANTS = {
    "cross_field": {
        "I1": "backend == backend_attempt",
        "I2": "fallback_allowed is True",
        "I3": "fallback_taken == (backend_attempt == \"interpreter\")",
        "I4": "executed_program == \"not_executed_preparation\" if preparation_only else (\"numba_lowered\" if backend_attempt == \"numba\" else \"original_dsl_interpreter\")",
        "I5": "interpreter_failure is an object iff event == \"interpreter-failed\"; then interpreter_failure.prior_decline_stage == decline_stage and interpreter_failure.prior_decline_reason == decline_reason",
        "I6": "program_text_sha256 == lowered_sha256 for fast-ran (lowered non-null); == source_sha256 for every other row",
        "I7": "lowered_sha256 null for transition-lowering / untyped-decline / numba-unavailable / precheck-decline; non-null for transition-module_exec / transition-run_model / fast-ran; for interpreter-ran / interpreter-failed null iff decline_stage == \"lowering\"",
        "I8": "typed-decline rows (transition-*, interpreter-*, precheck-decline): history non-empty and history[-1] == {backend: \"numba\", stage: decline_stage, reason: decline_reason, typed: true}; fast-ran / untyped-decline / numba-unavailable: history == []",
        "I9": "non-reemit rows (transition-*, untyped-decline, numba-unavailable, precheck-decline): events == [{event: record.event, event_utc: record.event_utc}] EXACTLY (one event, its own); reemit rows: the trail is DERIVED by expected_closing_record (transition.events + exactly one appended event, or exactly [fast-ran])",
        "I10": "decline_typed and decline_stage == \"lowering\" => decline_hook_available is true (a typed LOWERING reason is collected only through the on_decline hook; without the hook the null return is untyped)"
    },
    "cross_record": {
        "subsumed_by": "reemit_and_compare (v4): C1-C7 are implied by field equality with the record derived from the surviving transition; C7 (time) stays a separate anchor (TIME_BEFORE_TRANSITION)",
        "C1": "T.decline_stage == S.decline_stage",
        "C2": "T.decline_reason == S.decline_reason",
        "C3": "T.source_sha256 == S.source_sha256 and T.lowered_sha256 == S.lowered_sha256 and T.program_text_sha256 == S.program_text_sha256",
        "C4": "T.binding == S.binding (every key); imported-engine hashes equal (ENGINE_HASH_MISMATCH)",
        "C5": "T.history == S.history",
        "C6": "T.events[:len(S.events)] == S.events",
        "C7": "T.event_utc >= S.event_utc (TIME_BEFORE_TRANSITION)"
    }
}
# PROTOCOL v4 section 3b -- RE-EMIT AND COMPARE: embedded VERBATIM from the tracked PROTOCOL.json (a test asserts equality).
REEMIT = {
    "reemit_events": [
        "interpreter-ran",
        "interpreter-failed",
        "fast-ran"
    ],
    "deterministic_fields": {
        "interpreter-ran": [
            "contract",
            "event",
            "engine",
            "binding",
            "outcome",
            "backend",
            "backend_attempt",
            "backend_terminal",
            "executed_program",
            "fallback_allowed",
            "fallback_taken",
            "decline_reason",
            "decline_typed",
            "decline_stage",
            "decline_hook_available",
            "interpreter_failure",
            "preparation_only",
            "source_sha256",
            "lowered_sha256",
            "program_text_sha256",
            "history",
            "events",
            "verdict"
        ],
        "interpreter-failed": [
            "contract",
            "event",
            "engine",
            "binding",
            "outcome",
            "backend",
            "backend_attempt",
            "backend_terminal",
            "executed_program",
            "fallback_allowed",
            "fallback_taken",
            "decline_reason",
            "decline_typed",
            "decline_stage",
            "decline_hook_available",
            "interpreter_failure",
            "preparation_only",
            "source_sha256",
            "lowered_sha256",
            "program_text_sha256",
            "history",
            "events",
            "verdict"
        ],
        "fast-ran": [
            "contract",
            "event",
            "engine",
            "binding",
            "outcome",
            "backend",
            "backend_attempt",
            "backend_terminal",
            "executed_program",
            "fallback_allowed",
            "fallback_taken",
            "decline_reason",
            "decline_typed",
            "decline_stage",
            "decline_hook_available",
            "interpreter_failure",
            "preparation_only",
            "source_sha256",
            "program_text_sha256",
            "history",
            "events",
            "verdict"
        ]
    },
    "free_fields": {
        "interpreter-ran": {
            "event_utc": "ISO-8601 UTC; >= transition.event_utc (TIME_BEFORE_TRANSITION) and <= postprocess time + 60 s (TERMINAL_INCONSISTENT:event_utc)"
        },
        "interpreter-failed": {
            "event_utc": "as interpreter-ran",
            "interpreter_failure.type": "non-empty str (schema)",
            "interpreter_failure.message": "str, <= 1000 chars (TERMINAL_INCONSISTENT:interpreter_failure.message); the outcome string is DERIVED from type and message[:400], so a non-derived outcome is TERMINAL_INCONSISTENT:outcome"
        },
        "fast-ran": {
            "event_utc": ">= descriptor.minted_utc (TIME_BEFORE_MINT) and <= postprocess time + 60 s",
            "lowered_sha256": "sha256 (schema); program_text_sha256 is DERIVED equal to it"
        },
        "verdict": "const null in this contract (the comparator owns the verdict outside the record)",
        "context_note": "engine fields, decline_hook_available and the out_dir spelling are NOT free fields: they are the attempt context (context_resolution), anchored by the parent or learned from the first valid record and then fixed"
    }
}
# PROTOCOL v5 section 1b -- the attempt CONTEXT (resolved once: anchor -> learned -> free), embedded VERBATIM from the tracked PROTOCOL.json.
CONTEXT = {
    "fields": [
        "engine.engine_pin",
        "engine.solar_package_path",
        "engine.numba_accel_sha256",
        "engine.verify_sha256",
        "engine.core_sha256",
        "engine.python",
        "engine.executable",
        "engine.config",
        "source_sha256",
        "decline_hook_available",
        "binding.out_dir"
    ],
    "precedence": {
        "1_anchor": "parent-held descriptor values are ANCHORS: engine.engine_pin <- descriptor.engine_pin; source_sha256 <- descriptor.source_sha256; engine.<k> <- descriptor.expected_engine[k] for EVERY key k the parent names (every named field is enforced; UNNAMED keys are unspecified -- never a null requirement); binding.out_dir has no parent spelling anchor, only the canonical rule below; the keys the parent may name are exactly the enumerated engine keys (anchor_domain); any other key is refused at construction",
        "2_learned": "a field not anchored is LEARNED from the first ADMISSIBLE SURVIVING record of the attempt (classification order: parent tag first, then tags in sorted order; transitions before other rows within a tag) and is then FIXED for every later record of the attempt across all cases and stages. ADMISSIBLE SURVIVING = the record passed every per-record check AND remained in the valid set after the scope and state-machine classification of its tag (a record removed as STATE_CONFLICT / SCOPE_CONFLICT or rejected otherwise never commits a value)",
        "3_free": "a context field is free ONLY until the first valid record fixes it; no engine field, the source hash, the hook flag or the out_dir spelling stays free afterwards. The genuinely free record fields are listed in reemit_and_compare.free_fields (event_utc; failure type/message; fast-ran lowered_sha256)"
    },
    "codes": {
        "engine.engine_pin": "ENGINE_PIN_MISMATCH",
        "source_sha256": "SOURCE_MISMATCH",
        "engine.numba_accel_sha256 / verify_sha256 / core_sha256": "ENGINE_HASH_MISMATCH",
        "engine.solar_package_path / python / executable / config and decline_hook_available": "CONTEXT_MISMATCH:<field>",
        "binding.out_dir": "IDENTITY_MISMATCH (canonical rule) / CONTEXT_MISMATCH:binding.out_dir (spelling differs from the learned one)",
        "note": "the three legacy names are the historical spellings of CONTEXT_MISMATCH for those fields and are kept for continuity of the retained tests"
    }
}
CONTEXT_FIELDS = tuple(CONTEXT["fields"])
ANCHOR_DOMAIN = ["engine_pin", "solar_package_path", "numba_accel_sha256", "verify_sha256", "core_sha256", "python", "executable", "config"]   # PROTOCOL v6: the only expected_engine keys a parent may name
_MAX_CONTEXT_ROUNDS = 4
_CONTEXT_CODE = {"engine.engine_pin": "ENGINE_PIN_MISMATCH", "source_sha256": "SOURCE_MISMATCH", "engine.numba_accel_sha256": "ENGINE_HASH_MISMATCH", "engine.verify_sha256": "ENGINE_HASH_MISMATCH", "engine.core_sha256": "ENGINE_HASH_MISMATCH"}   # every other context field -> CONTEXT_MISMATCH:<field>
CLOSING_EVENTS = ("interpreter-ran", "interpreter-failed")
REEMIT_EVENTS = ("interpreter-ran", "interpreter-failed", "fast-ran")      # closing records built ONLY through expected_closing_record (PROTOCOL v4 section 3b)
_FREE_TIME_SKEW_S = 60.0
MODES = ("fallback-allowed", "no-fallback-expected")
PROTOCOL_DIR = Path(__file__).resolve().parent / "fallback_protocol"      # the TRACKED canonical spec (PROTOCOL.md / PROTOCOL.json)


def _is_sha(v) -> bool:
    return isinstance(v, str) and bool(_HEX64.fullmatch(v))


def _utc_ok(v) -> bool:
    try:
        datetime.datetime.fromisoformat(v); return isinstance(v, str)
    except (TypeError, ValueError):
        return False


def _check_value(spec: dict, v, path: str) -> Optional[str]:
    """One schema node; returns the first failing path or None.  Pure type/shape checks, never raises on bad input."""
    t = spec["type"]
    if v is None:
        if t == "const":
            return None if spec["value"] is None else f"SCHEMA:{path}:const"
        if t == "enum":
            return None if None in spec["values"] else f"SCHEMA:{path}:enum"
        return None if spec.get("nullable") else f"SCHEMA:{path}:null"
    if t == "const":
        return None if v == spec["value"] and type(v) == type(spec["value"]) else f"SCHEMA:{path}:const"
    if t == "str":
        if not isinstance(v, str): return f"SCHEMA:{path}:type"
        if spec.get("nonempty") and not v: return f"SCHEMA:{path}:empty"
        return None
    if t == "utc":
        return None if _utc_ok(v) else f"SCHEMA:{path}:utc"
    if t == "sha256":
        return None if _is_sha(v) else f"SCHEMA:{path}:sha256"
    if t == "bool":
        return None if isinstance(v, bool) else f"SCHEMA:{path}:type"
    if t == "int":
        return None if isinstance(v, int) and not isinstance(v, bool) else f"SCHEMA:{path}:type"
    if t == "enum":
        return None if (not isinstance(v, (list, dict)) and v in spec["values"] and type(v) in {type(x) for x in spec["values"] if x is not None}) else f"SCHEMA:{path}:enum"
    if t == "object":
        if not isinstance(v, dict): return f"SCHEMA:{path}:type"
        for k, fs in spec["fields"].items():
            if k not in v:
                if fs.get("optional"): continue
                return f"SCHEMA:{path}.{k}"          # PRESENCE is checked separately from nullability: a required key must exist even when nullable
            if v[k] is None and fs.get("optional"): continue
            r = _check_value(fs, v[k], f"{path}.{k}")
            if r: return r
        xor = spec.get("xor")
        if xor and sum(1 for k in xor if v.get(k) is not None) > 1:
            return f"SCHEMA:{path}.xor:{'+'.join(xor)}"
        return None
    if t == "list":
        if not isinstance(v, list): return f"SCHEMA:{path}:type"
        for n, item in enumerate(v):
            r = _check_value(spec["items"], item, f"{path}[{n}]")
            if r: return r
        return None
    return f"SCHEMA:{path}:unknown_type"


def validate_record_safe(rec) -> "tuple[bool, list[str]]":
    """GENERATED from RECORD_SCHEMA; NEVER raises; stops at the first structural failure; checks every advertised field
    and nested element type BEFORE any use."""
    try:
        if not isinstance(rec, dict):
            return False, ["NOT_AN_OBJECT"]
        for k in REQUIRED_RECORD_FIELDS:
            if k not in rec:
                return False, [f"SCHEMA:{k}"]
        if rec.get("contract") != CONTRACT:
            return False, ["CONTRACT_MISMATCH"]
        for k, spec in RECORD_SCHEMA.items():
            if k not in rec:
                return False, [f"SCHEMA:{k}"]
            r = _check_value(spec, rec[k], k)
            if r:
                return False, ["TIME_INVALID" if r.startswith("SCHEMA:event_utc") else r]
        return True, []
    except Exception as exc:  # the guarantee: never raise
        return False, [f"SCHEMA:exception:{type(exc).__name__}"]


def validate_record(rec: dict) -> dict:
    """Raising wrapper for in-process emission; the post-processor uses :func:`validate_record_safe` only."""
    ok, reasons = validate_record_safe(rec)
    if not ok:
        raise RecordInvalid("; ".join(reasons))
    return rec


def _decline_exception_type():
    return getattr(_engine(), "KernelOutputDtypeUnresolved", None)


def _is_typed_decline(exc: BaseException) -> bool:
    typ = _decline_exception_type()
    return typ is not None and isinstance(exc, typ)


def interpreter_namespace() -> dict:
    """Exactly the namespace ``verify._isolated_run_model`` / ``execute_translated``'s interpreter branch build."""
    from solar.einsum_lib.verify import _MATH_BUILTINS
    return {**_MATH_BUILTINS, "math": math, "np": np}


class PreparedInputs:
    """An IMMUTABLE prepared baseline: deep copies of the caller's arrays, held read-only and hashed once; each
    attempt gets fresh writable value-preserving copies; the baseline is re-verified after every attempt."""

    def __init__(self, inputs: Dict[str, np.ndarray]):
        self._base: Dict[str, np.ndarray] = {}
        self.digest: Dict[str, dict] = {}
        for k, v in inputs.items():
            a = np.array(v, copy=True)
            a.setflags(write=False)
            self._base[k] = a
            self.digest[k] = {"sha256": _sha256_array(a), "dtype": str(a.dtype), "shape": list(a.shape), "nbytes": int(a.nbytes)}
        self.baseline_bytes = sum(a.nbytes for a in self._base.values())
        self.copies_made = 0
        self.bytes_copied = 0

    def fresh(self) -> Dict[str, np.ndarray]:
        out = {k: np.array(v, copy=True) for k, v in self._base.items()}
        self.copies_made += 1
        self.bytes_copied += sum(a.nbytes for a in out.values())
        return out

    def verify(self) -> bool:
        ok = all(_sha256_array(a) == self.digest[k]["sha256"] and not a.flags.writeable for k, a in self._base.items())
        if not ok:
            raise RuntimeError("prepared baseline changed: immutability violated")
        return True

    def accounting(self) -> dict:
        return {"baseline_bytes": int(self.baseline_bytes), "copies_made": self.copies_made, "bytes_copied": int(self.bytes_copied),
                "peak_extra_bytes_bound": int(self.baseline_bytes * 2), "digest": self.digest}


def prepared_inputs(inputs: Dict[str, np.ndarray]) -> PreparedInputs:
    return PreparedInputs(inputs)


InputsSource = Union[PreparedInputs, Callable[[], Dict[str, np.ndarray]]]


def _fresh_inputs(source: InputsSource) -> Dict[str, np.ndarray]:
    return source.fresh() if isinstance(source, PreparedInputs) else source()


def _after_attempt(source: InputsSource) -> None:
    if isinstance(source, PreparedInputs):
        source.verify()


def _jcopy(o):
    return json.loads(json.dumps(o))


def expected_closing_record(transition: Optional[dict], context: Optional[dict], outcome_kind: str, free: dict, mode: Optional[str] = None) -> dict:
    """PROTOCOL v4 section 3b -- RE-EMIT AND COMPARE.  PURE.  The ONLY way a closing record (interpreter-ran / interpreter-failed /
    fast-ran) is built, by the emitter (CandidateProgram) and by the validator (classify_event) alike.
    transition: the persisted transition record (None for fast-ran); context: fast-ran only -- {"binding", "engine", "source_sha256",
    "decline_hook_available"}; free: {"event_utc"} (+ "failure_type", "failure_message" for interpreter-failed; + "lowered_sha256" for
    fast-ran).  mode is accepted for the documented signature only: no record field depends on it."""
    if mode is not None and mode not in MODES:
        raise RecordInvalid(f"unknown mode {mode!r}")
    if outcome_kind not in REEMIT_EVENTS:
        raise RecordInvalid(f"{outcome_kind!r} is not a re-emit closing event")
    event_utc = free["event_utc"]
    if outcome_kind == "fast-ran":
        if transition is not None or context is None:
            raise RecordInvalid("fast-ran is built from the attempt context, never from a transition")
        lowered = free["lowered_sha256"]
        return {"contract": CONTRACT, "event": "fast-ran", "event_utc": event_utc, "engine": _jcopy(context["engine"]), "binding": _jcopy(context["binding"]),
                "outcome": OUTCOME_FAST_RAN, "backend": BACKEND_NUMBA, "backend_attempt": BACKEND_NUMBA, "backend_terminal": BACKEND_NUMBA, "executed_program": "numba_lowered",
                "fallback_allowed": True, "fallback_taken": False, "decline_reason": None, "decline_typed": False, "decline_stage": None,
                "decline_hook_available": bool(context["decline_hook_available"]), "interpreter_failure": None, "preparation_only": False,
                "source_sha256": context["source_sha256"], "lowered_sha256": lowered, "program_text_sha256": lowered, "history": [],
                "events": [{"event": "fast-ran", "event_utc": event_utc}], "verdict": None}
    if transition is None:
        raise RecordInvalid(f"{outcome_kind} is built from the surviving transition")
    rec = _jcopy(transition)
    rec["event"] = outcome_kind; rec["event_utc"] = event_utc
    rec["events"] = [dict(e) for e in transition.get("events", [])] + [{"event": outcome_kind, "event_utc": event_utc}]
    if outcome_kind == "interpreter-ran":
        rec["outcome"] = OUTCOME_INTERPRETER_RAN; rec["backend_terminal"] = BACKEND_INTERPRETER; rec["interpreter_failure"] = None
    else:
        t = str(free["failure_type"]); m = str(free["failure_message"])[:1000]
        rec["outcome"] = f"{OUTCOME_INTERPRETER_FAILED}:{t}: {m[:400]}"; rec["backend_terminal"] = None
        rec["interpreter_failure"] = {"type": t, "message": m, "prior_decline_reason": transition["decline_reason"], "prior_decline_stage": transition["decline_stage"]}
    rec["verdict"] = None
    return rec


def _deep_diff(expected, got, path: str = "") -> List[str]:
    """Dotted paths of every deterministic difference; a dict names the key, a list names the container."""
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return [path or "<record>"]
        out: List[str] = []
        for k in sorted(set(expected) | set(got)):
            sub = f"{path}.{k}" if path else k
            if k not in expected or k not in got:
                out.append(sub)
            else:
                out.extend(_deep_diff(expected[k], got[k], sub))
        return out
    if isinstance(expected, list):
        return [] if (isinstance(got, list) and len(got) == len(expected) and all(not _deep_diff(a, b, path) for a, b in zip(expected, got))) else [path]
    return [] if (expected == got and type(expected) == type(got)) else [path]


@dataclass
class CandidateProgram:
    """The candidate as it will actually be executed, plus the self-bound decline / transition record."""

    source: str
    use_fp64: bool
    boundscheck: bool
    backend: str
    program_text: str
    lowered: Optional[str]
    decline_hook_available: bool
    engine: dict
    binding: dict
    outcome: str = OUTCOME_FAST_LOWERED
    backend_terminal: Optional[str] = None
    decline_reason: Optional[str] = None
    decline_typed: bool = False
    decline_stage: Optional[str] = None
    interpreter_failure: Optional[dict] = None
    preparation_only: bool = False      # a non-executing lowering precheck (row 119): a typed decline is recorded, no execution transition is opened
    history: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    persist: Optional[Persist] = None
    transition_record: Optional[dict] = None   # the PERSISTED transition record: the input of every interpreter closing (PROTOCOL v4 section 3b)

    def _emit_closing(self, outcome_kind: str, free: dict, context: Optional[dict] = None) -> dict:
        """PROTOCOL v4: closing records are built ONLY by expected_closing_record (shared with the validator), then persisted."""
        rec = expected_closing_record(self.transition_record if outcome_kind != "fast-ran" else None, context, outcome_kind, free)
        self.events.append({"event": outcome_kind, "event_utc": rec["event_utc"]})
        if rec["events"] != [dict(e) for e in self.events]:
            raise RecordInvalid(f"emitter trail {self.events!r} differs from the derived trail {rec['events']!r}")
        validate_record(rec)
        if self.persist is not None:
            self.persist(outcome_kind, rec)
        return rec

    @property
    def executed_program(self) -> str:
        if self.preparation_only:
            return "not_executed_preparation"
        return "numba_lowered" if self.backend == BACKEND_NUMBA else "original_dsl_interpreter"

    @property
    def fallback_taken(self) -> bool:
        return self.backend == BACKEND_INTERPRETER

    def record(self, event: Optional[str] = None) -> dict:
        """Receipt fields, self-bound.  No verdict: the caller's comparator decides on returned outputs only."""
        return {
            "contract": CONTRACT,
            "event": event,
            "event_utc": _utc(),
            "engine": dict(self.engine),
            "binding": dict(self.binding),
            "outcome": self.outcome,
            "backend": self.backend,
            "backend_attempt": self.backend,
            "backend_terminal": self.backend_terminal,
            "executed_program": self.executed_program,
            "fallback_allowed": True,
            "fallback_taken": self.fallback_taken,
            "decline_reason": self.decline_reason,
            "decline_typed": self.decline_typed,
            "decline_stage": self.decline_stage,
            "decline_hook_available": self.decline_hook_available,
            "interpreter_failure": self.interpreter_failure,
            "preparation_only": self.preparation_only,
            "source_sha256": _sha256_text(self.source),
            "lowered_sha256": _sha256_text(self.lowered) if self.lowered is not None else None,
            "program_text_sha256": _sha256_text(self.program_text),
            "history": list(self.history),
            "events": [dict(e) for e in self.events],
            "verdict": None,
        }

    def _emit(self, event: str) -> dict:
        rec = self.record(event)
        self.events.append({"event": event, "event_utc": rec["event_utc"]})
        rec["events"] = [dict(e) for e in self.events]
        validate_record(rec)
        if self.persist is not None:
            self.persist(event, rec)
        return rec

    def _switch_to_interpreter(self, stage: str, reason: str) -> None:
        """TYPED decline -> interpreter.  Persists the self-bound transition BEFORE any interpreter work."""
        self.history.append({"backend": self.backend, "stage": stage, "reason": reason, "typed": True})
        self.program_text = self.source
        self.decline_reason = reason
        self.decline_typed = True
        self.decline_stage = stage
        if self.preparation_only:
            # PREPARATION (P2-R3-119-PRECHECK-1): the program text becomes the original source (what an execution attempt
            # WOULD interpret) but NO interpreter attempt is opened: the backend stays the fast engine that declined, the
            # record is a preparation event (precheck-decline), never a transition.
            self.outcome = OUTCOME_PRECHECK_DECLINE
            self._emit("precheck-decline")
            return
        self.backend = BACKEND_INTERPRETER
        self.outcome = OUTCOME_INTERPRETER_PENDING
        self.transition_record = self._emit("transition-" + stage)

    def _interpreter_failed(self, exc: BaseException) -> InterpreterFailed:
        rec = self._emit_closing("interpreter-failed", {"event_utc": _utc(), "failure_type": type(exc).__name__, "failure_message": str(exc)[:1000]})
        self.interpreter_failure = dict(rec["interpreter_failure"]); self.outcome = rec["outcome"]
        return InterpreterFailed(f"interpreter attempt failed after a typed decline at {self.decline_stage} "
                                 f"({self.decline_reason}): {type(exc).__name__}: {exc}", rec)

    def make_namespace(self) -> dict:
        if self.backend == BACKEND_NUMBA:
            return _engine().execution_namespace(use_fp64=self.use_fp64)
        return interpreter_namespace()

    def _exec_interpreter_module(self, filename: str) -> dict:
        try:
            namespace = self.make_namespace()
            exec(compile(self.program_text, filename, "exec"), namespace)  # noqa: S102
            if not callable(namespace.get("run_model")):
                raise RuntimeError("candidate defines no run_model()")
            return namespace
        except Exception as exc:
            raise self._interpreter_failed(exc) from exc
        except BaseException as exc:
            self._interpreter_failed(exc)
            raise

    def exec_module(self, namespace: dict, filename: str = "<candidate>", compiled=None) -> dict:
        if self.preparation_only:
            raise RecordInvalid("a preparation-only program (lower_or_fallback(..., execution=False)) is never executed; use an execution call for the attempt")
        if self.backend == BACKEND_INTERPRETER:
            return self._exec_interpreter_module(filename)
        if compiled is None:
            compiled = compile(self.program_text, filename, "exec")
        try:
            exec(compiled, namespace)  # noqa: S102 -- the caller chose to execute this candidate
        except Exception as exc:
            if _is_typed_decline(exc):
                self._switch_to_interpreter(STAGE_MODULE_EXEC, f"{type(exc).__name__}: {exc}")
                return self._exec_interpreter_module(filename)
            raise
        if not callable(namespace.get("run_model")):
            raise RuntimeError("candidate defines no run_model()")
        return namespace

    def run_model(self, namespace: dict, inputs: InputsSource, filename: str = "<candidate>") -> dict:
        if self.preparation_only:
            raise RecordInvalid("a preparation-only program is never executed")
        if self.backend == BACKEND_NUMBA:
            try:
                outputs = namespace["run_model"](_fresh_inputs(inputs))
            except Exception as exc:
                if not _is_typed_decline(exc):
                    raise
                _after_attempt(inputs)
                self._switch_to_interpreter(STAGE_RUN_MODEL, f"{type(exc).__name__}: {exc}")
                namespace = self._exec_interpreter_module(filename)
            else:
                _after_attempt(inputs)
                if not isinstance(outputs, dict):
                    raise RuntimeError(f"run_model() must return a dict, got {type(outputs).__name__}")
                self.outcome = OUTCOME_FAST_RAN
                self.backend_terminal = BACKEND_NUMBA
                self._emit_closing("fast-ran", {"event_utc": _utc(), "lowered_sha256": _sha256_text(self.lowered)},
                                   context={"binding": dict(self.binding), "engine": dict(self.engine), "source_sha256": _sha256_text(self.source), "decline_hook_available": self.decline_hook_available})
                return outputs
        try:
            outputs = namespace["run_model"](_fresh_inputs(inputs))
            if not isinstance(outputs, dict):
                raise RuntimeError(f"run_model() must return a dict, got {type(outputs).__name__}")
        except Exception as exc:
            raise self._interpreter_failed(exc) from exc
        except BaseException as exc:
            self._interpreter_failed(exc)
            raise
        _after_attempt(inputs)
        self.outcome = OUTCOME_INTERPRETER_RAN
        self.backend_terminal = BACKEND_INTERPRETER
        self._emit_closing("interpreter-ran", {"event_utc": _utc()})
        return outputs


def _fail(prog: CandidateProgram, outcome: str, reason: str, event: str, exc_type):
    prog.outcome = outcome
    prog.decline_reason = reason
    prog.decline_typed = False
    prog.decline_stage = STAGE_LOWERING
    rec = prog._emit(event)
    return exc_type(reason, rec)


def lower_or_fallback(source: str, *, boundscheck: bool = True, use_fp64: bool = True, engine_pin: str,
                      binding: dict, persist: Optional[Persist] = None, config: Optional[dict] = None,
                      execution: bool = True) -> CandidateProgram:
    """Lower ``source`` with the engine's ``numba_lower``; on a TYPED decline hand back the interpreter program
    (self-bound transition persisted first); on an UNTYPED decline raise :class:`UntypedDecline`; without Numba raise
    :class:`NumbaUnavailable`.  ``engine_pin`` (the caller's engine/config pin) and ``binding`` (the attempt binding from
    :func:`attempt_binding`: canonical run directory + the controller-minted identity) are mandatory and go into every
    record.  ``execution=False`` marks a NON-EXECUTING PREPARATION (the row-119 lowering precheck): a typed decline is
    then recorded as ``precheck-decline`` (outcome ``PRECHECK_DECLINE_NO_EXECUTION``) and no execution transition is
    opened; the returned program cannot be executed."""
    engine = engine_identity(engine_pin, config)
    if not isinstance(binding, dict) or not binding.get("out_dir"):
        raise RecordInvalid("binding must carry the attempt's run directory (out_dir); build it with attempt_binding(out_dir, ...)")
    binding = {**binding, "run_dir": canonical_run_dir(binding["out_dir"])}
    eng = _engine()
    has = bool(getattr(eng, "HAS_NUMBA", False))
    hook = has and "on_decline" in inspect.signature(eng.numba_lower).parameters
    prog = CandidateProgram(source=source, use_fp64=use_fp64, boundscheck=boundscheck, backend=BACKEND_NUMBA, program_text=source,
                            lowered=None, decline_hook_available=hook, engine=engine, binding=dict(binding), persist=persist, preparation_only=not execution)
    if not has:
        raise _fail(prog, OUTCOME_NUMBA_UNAVAILABLE, "numba is not installed: environment failure, not a typed engine decline; no fallback",
                    "numba-unavailable", NumbaUnavailable)
    reasons: List[str] = []
    kwargs = {"on_decline": reasons.append} if hook else {}
    lowered = eng.numba_lower(source, boundscheck=boundscheck, use_fp64=use_fp64, **kwargs)
    if lowered is None:
        if reasons:
            prog._switch_to_interpreter(STAGE_LOWERING, reasons[0])
            return prog
        if hook:
            reason = "UNTYPED: numba_lower returned None without reporting a reason through on_decline; no fallback (typed declines only)"
        else:
            reason = ("UNTYPED: numba_lower returned None and this engine pin exposes no on_decline hook "
                      "(numba_lower(code, boundscheck, use_fp64) only; legacy signature); no fallback (typed declines only)")
        raise _fail(prog, OUTCOME_UNTYPED_DECLINE, reason, "untyped-decline", UntypedDecline)
    prog.program_text = lowered
    prog.lowered = lowered
    return prog


def execute_candidate(source: str, inputs: InputsSource, *, boundscheck: bool = True, use_fp64: bool = True, engine_pin: str,
                      binding: dict, filename: str = "<candidate>", persist: Optional[Persist] = None, config: Optional[dict] = None):
    """One-call form (row 119): returns ``(outputs, record)``; every contract failure raises a :class:`FastStageFailure`
    whose ``.record`` is the persisted record; candidate errors on the fast path propagate unchanged."""
    prog = lower_or_fallback(source, boundscheck=boundscheck, use_fp64=use_fp64, engine_pin=engine_pin, binding=binding, persist=persist, config=config)
    namespace = prog.exec_module(prog.make_namespace(), filename)
    outputs = prog.run_model(namespace, inputs, filename)
    return outputs, prog.record("record")


def resource_stop_record(transition_record: dict, reason: str, attempt: Optional[dict] = None) -> dict:
    """The controller's DISTINCT record for a resource kill during the fallback attempt, built from the persisted
    transition (all that survives a kill): carries the prior typed decline and the attempt; outcome RESOURCE_STOP;
    no verdict."""
    rec = dict(transition_record)
    rec["event"] = "resource-stop"
    rec["event_utc"] = _utc()
    rec["outcome"] = OUTCOME_RESOURCE_STOP
    rec["backend_terminal"] = None
    rec["resource_stop"] = {"reason": reason, "during": "interpreter attempt after a typed decline",
                            "prior_decline_reason": transition_record.get("decline_reason"), "prior_decline_stage": transition_record.get("decline_stage"),
                            "transition_event": transition_record.get("event"), "transition_event_utc": transition_record.get("event_utc"), "attempt": dict(attempt or {})}
    rec["events"] = [dict(e) for e in transition_record.get("events", [])] + [{"event": "resource-stop", "event_utc": rec["event_utc"]}]
    rec["verdict"] = None
    return validate_record(rec)


# ---------------------------------------------------------------------------------------------------------------------
# attempt identity (PROTOCOL.md section 1): the parent-held LaunchDescriptor is the only authority; the disk mint lets
# the child bind and is validated against the descriptor afterwards
# ---------------------------------------------------------------------------------------------------------------------

def canonical_run_dir(p) -> str:
    return os.path.realpath(str(p))


@dataclass(frozen=True)
class LaunchDescriptor:
    """Immutable, parent-held identity of ONE attempt.  Created before the launch (``mint_attempt``) or, for a run that no
    controller minted, by ``standalone`` (identity only: no source/engine/time anchor)."""

    run_dir: str
    out_dir: str
    group: Optional[str] = None
    attempt: Optional[int] = None
    row: Optional[int] = None
    admission_sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    engine_pin: Optional[str] = None
    expected_engine: Optional[dict] = None
    harness_sha256: Optional[str] = None
    shim_sha256: Optional[str] = None
    minted_utc: Optional[str] = None
    cases: Optional[tuple] = None
    stages: Optional[tuple] = None
    mint_sha256: Optional[str] = None
    mode: str = "fallback-allowed"       # PROTOCOL v3 section 1: fallback-allowed (default) | no-fallback-expected
    authority: str = "parent"

    def identity(self) -> dict:
        d = {"out_dir": self.out_dir, "run_dir": self.run_dir}
        for k in ATTEMPT_IDENTITY_FIELDS:
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def tag_kind(self, name: str) -> Optional[str]:
        """'case' / 'stage' when the name is declared, else None (kind then comes from the records' binding)."""
        if self.cases and name in self.cases: return "case"
        if self.stages and name in self.stages: return "stage"
        return None

    def __post_init__(self):
        if self.cases and self.stages and set(self.cases) & set(self.stages):
            raise RecordInvalid(f"declared cases and stages overlap: {sorted(set(self.cases) & set(self.stages))}")
        if self.mode not in MODES:
            raise RecordInvalid(f"unknown mode {self.mode!r}; one of {MODES}")
        _check_expected_engine(self.expected_engine, self.engine_pin)      # PROTOCOL v6 anchor domain

    @staticmethod
    def standalone(binding: dict) -> "LaunchDescriptor":
        """Identity-only descriptor for an attempt without a controller mint (tests / standalone launches)."""
        if not isinstance(binding, dict) or not binding.get("out_dir"):
            raise RecordInvalid("standalone descriptor needs out_dir")
        return LaunchDescriptor(run_dir=canonical_run_dir(binding["out_dir"]), out_dir=str(binding["out_dir"]), group=binding.get("group"),
                                attempt=binding.get("attempt"), row=binding.get("row"), authority="standalone")


def mint_attempt(out_dir, *, group: str, attempt: int, row: Optional[int] = None, admission_sha256: Optional[str] = None, source_sha256: Optional[str] = None,
                 engine_pin: Optional[str] = None, expected_engine: Optional[dict] = None, harness_sha256: Optional[str] = None, shim_sha256: Optional[str] = None,
                 cases: Optional[list] = None, stages: Optional[list] = None, mode: str = "fallback-allowed") -> LaunchDescriptor:
    """Controller side, BEFORE the launch: write the disk mint for the child and return the PARENT-HELD descriptor (the
    only authority).  The disk mint's sha256 is kept so it can be verified after the child stopped."""
    _check_expected_engine(expected_engine, engine_pin)                     # PROTOCOL v6 anchor domain: refused BEFORE anything is written
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    minted = _utc()
    mint = {"schema": "lite-fallback-attempt-v2", "contract": CONTRACT, "minted_utc": minted, "out_dir": str(out_dir), "run_dir": canonical_run_dir(out_dir), "group": str(group), "attempt": int(attempt), "row": row,
            "admission_sha256": admission_sha256, "source_sha256": source_sha256, "engine_pin": engine_pin, "harness_sha256": harness_sha256, "shim_sha256": shim_sha256, "cases": list(cases) if cases else None, "stages": list(stages) if stages else None, "mode": mode}
    if mode not in MODES:
        raise RecordInvalid(f"unknown mode {mode!r}; one of {MODES}")
    data = (json.dumps(mint, indent=1) + "\n").encode()
    (out_dir / MINT_FILE).write_bytes(data)
    return LaunchDescriptor(run_dir=mint["run_dir"], out_dir=str(out_dir), group=str(group), attempt=int(attempt), row=row, admission_sha256=admission_sha256, source_sha256=source_sha256, engine_pin=engine_pin,
                            expected_engine=dict(expected_engine) if expected_engine else None, harness_sha256=harness_sha256, shim_sha256=shim_sha256, minted_utc=minted,
                            cases=tuple(cases) if cases else None, stages=tuple(stages) if stages else None, mint_sha256=hashlib.sha256(data).hexdigest(), mode=mode, authority="parent")


def attempt_binding(out_dir, **extra) -> dict:
    """Harness side: canonical run directory + the parent-minted identity (when the mint exists) + the caller's sub-attempt
    tag (``row``, ``case``, ``stage``).  The mint is read here ONLY so the child can bind; it is never an authority."""
    out_dir = Path(out_dir); b = {"out_dir": str(out_dir), "run_dir": canonical_run_dir(out_dir)}
    p = out_dir / MINT_FILE
    if p.exists():
        try:
            mint = json.loads(p.read_bytes())
        except (OSError, ValueError) as exc:
            raise RecordInvalid(f"attempt mint {p} unreadable: {type(exc).__name__}: {exc}") from exc
        if not isinstance(mint, dict) or mint.get("run_dir") != b["run_dir"]:
            raise RecordInvalid(f"attempt mint run_dir {mint.get('run_dir') if isinstance(mint, dict) else mint!r} != this run directory {b['run_dir']!r}")
        for k in ("group", "attempt", "row"):
            if mint.get(k) is not None:
                b[k] = mint[k]
    for k, v in extra.items():
        if k in b and b[k] != v:
            raise RecordInvalid(f"attempt binding conflict on {k!r}: mint {b[k]!r} vs caller {v!r}")
        b[k] = v
    if b.get("case") is not None and b.get("stage") is not None:
        raise RecordInvalid("a binding carries either case or stage, not both")
    return b


def _check_expected_engine(expected_engine: Optional[dict], engine_pin: Optional[str]) -> None:
    """PROTOCOL v6 anchor domain: a parent may name exactly the enumerated engine keys; a named engine_pin must equal the descriptor pin."""
    if expected_engine is None:
        return
    if not isinstance(expected_engine, dict):
        raise RecordInvalid("expected_engine must be a dict of enumerated engine keys")
    bad = sorted(k for k in expected_engine if k not in ANCHOR_DOMAIN)
    if bad:
        raise RecordInvalid(f"unsupported expected_engine key(s) {bad}: the parent may name only {ANCHOR_DOMAIN} (PROTOCOL v6 anchor domain)")
    if "engine_pin" in expected_engine and engine_pin is not None and expected_engine["engine_pin"] != engine_pin:
        raise RecordInvalid(f"expected_engine.engine_pin {expected_engine['engine_pin']!r} != descriptor engine_pin {engine_pin!r}")


class AttemptContext:
    """PROTOCOL v6 section 1b: the attempt context -- parent anchors, values COMMITTED only from admissible surviving evidence
    (two-phase resolution, see postprocess_attempt), plus a TENTATIVE layer used only while classifying in phase 1.  Context
    equality is TYPE-EXACT JSON equality (_deep_diff).  Never built from a child-supplied value except by learning."""

    def __init__(self, descriptor: "LaunchDescriptor", frozen: bool = False):
        self.anchored: Dict[str, object] = {}; self.learned: Dict[str, object] = {}; self.learned_from: Dict[str, str] = {}
        self.tentative: Dict[str, object] = {}; self.tentative_from: Dict[str, str] = {}; self.frozen = frozen
        if descriptor.engine_pin: self.anchored["engine.engine_pin"] = descriptor.engine_pin
        if descriptor.source_sha256: self.anchored["source_sha256"] = descriptor.source_sha256
        _check_expected_engine(descriptor.expected_engine, descriptor.engine_pin)
        for k, v in (descriptor.expected_engine or {}).items():
            self.anchored[f"engine.{k}"] = _jcopy(v)                      # EVERY named (enumerated) key is an anchor; unnamed keys stay unspecified

    def get(self, field: str):
        if field in self.anchored: return self.anchored[field]
        if field in self.learned: return self.learned[field]
        return self.tentative.get(field)

    def resolved(self, field: str) -> bool:
        return field in self.anchored or field in self.learned or field in self.tentative

    def mismatches(self, rec: dict) -> List[tuple]:
        """[(code, field)] for every context field whose resolved value the record contradicts under TYPE-EXACT JSON equality."""
        out = []
        for f in CONTEXT_FIELDS:
            if not self.resolved(f): continue
            if _deep_diff(self.get(f), _path_get(rec, f), f):
                out.append((_CONTEXT_CODE.get(f, "CONTEXT_MISMATCH"), f))
        return out

    def learn(self, rec: dict, source_path: str) -> None:
        """COMMIT: an admissible surviving record fixes every still-uncommitted context field."""
        for f in CONTEXT_FIELDS:
            if f not in self.anchored and f not in self.learned:
                self.learned[f] = _jcopy(_path_get(rec, f)); self.learned_from[f] = source_path

    def learn_tentative(self, rec: dict, source_path: str) -> None:
        """Phase 1 only: an individually valid record supplies TENTATIVE values for still-unresolved fields; discarded unless it survives."""
        if self.frozen: return
        for f in CONTEXT_FIELDS:
            if not self.resolved(f):
                self.tentative[f] = _jcopy(_path_get(rec, f)); self.tentative_from[f] = source_path

    def engine(self, rec: Optional[dict] = None) -> dict:
        """The resolved engine identity (anchored ∪ learned ∪ tentative); a still-unresolved key comes from the record only when the record is the learner."""
        eng = {}
        for f in CONTEXT_FIELDS:
            if f.startswith("engine."):
                k = f[len("engine."):]
                eng[k] = _jcopy(self.get(f)) if self.resolved(f) else (_jcopy(rec["engine"].get(k)) if rec is not None else None)
        return eng

    def value(self, field: str, rec: dict):
        return _jcopy(self.get(field)) if self.resolved(field) else _jcopy(_path_get(rec, field))

    def as_dict(self) -> dict:
        return {"anchored": _jcopy(self.anchored), "learned": _jcopy(self.learned), "learned_from": dict(self.learned_from), "committed_from": dict(self.learned_from)}


def _path_get(rec: dict, path: str):
    o = rec
    for part in path.split("."):
        if not isinstance(o, dict): return None
        o = o.get(part)
    return o


def bindings_match(binding: dict, attempt) -> dict:
    """Identity on BOTH sides: canonical run_dir equal; every identity field present-and-equal on both or absent on both."""
    ident = attempt.identity() if isinstance(attempt, LaunchDescriptor) else dict(attempt)
    reasons: List[str] = []
    rb = binding.get("run_dir") or (canonical_run_dir(binding["out_dir"]) if isinstance(binding.get("out_dir"), str) else None)
    ra = ident.get("run_dir") or (canonical_run_dir(ident["out_dir"]) if isinstance(ident.get("out_dir"), str) else None)
    if not rb or not ra:
        reasons.append("run_dir missing on " + ("both sides" if not rb and not ra else ("the record" if not rb else "the attempt")))
    elif rb != ra:
        reasons.append(f"run_dir differs: record {rb!r} vs attempt {ra!r}")
    if isinstance(binding.get("out_dir"), str) and ra:
        co = canonical_run_dir(binding["out_dir"])            # PROTOCOL v5: canonical(out_dir) == run_dir == descriptor.run_dir (aliases allowed)
        if co != ra or (binding.get("run_dir") and co != binding["run_dir"]):
            reasons.append(f"binding.out_dir {binding['out_dir']!r} canonicalises to {co!r}, not the attempt run_dir {ra!r}")
    for k in ATTEMPT_IDENTITY_FIELDS:
        inb, ina = binding.get(k) is not None, ident.get(k) is not None
        if inb != ina:
            reasons.append(f"{k} present on one side only ({'record' if inb else 'attempt'})")
        elif inb and str(binding[k]) != str(ident[k]):
            reasons.append(f"{k} differs: record {binding[k]!r} vs attempt {ident[k]!r}")
    return {"matches": not reasons, "reasons": reasons}


# ---------------------------------------------------------------------------------------------------------------------
# the event/state table (PROTOCOL.md section 3) and the per-record classification (section 4)
# ---------------------------------------------------------------------------------------------------------------------

KIND_TRANSITION, KIND_COMPLETE, KIND_NON_PASS, KIND_PREPARATION = "transition", "terminal_complete", "terminal_non_pass", "preparation"
# event -> (kind, outcome or ("prefix", p), backend_attempt, backend_terminal, decline_present, decline_typed, decline_stage ("any"|"any_exec"|stage|None), preparation_only ("any"|bool), scopes, requires_transition)
STATE_TABLE = {
    "transition-lowering":    (KIND_TRANSITION,  OUTCOME_INTERPRETER_PENDING, BACKEND_INTERPRETER, None, True, True, STAGE_LOWERING, False, ("parent", "case"), False),
    "transition-module_exec": (KIND_TRANSITION,  OUTCOME_INTERPRETER_PENDING, BACKEND_INTERPRETER, None, True, True, STAGE_MODULE_EXEC, False, ("parent", "case"), False),
    "transition-run_model":   (KIND_TRANSITION,  OUTCOME_INTERPRETER_PENDING, BACKEND_INTERPRETER, None, True, True, STAGE_RUN_MODEL, False, ("parent", "case"), False),
    "interpreter-ran":        (KIND_COMPLETE,    OUTCOME_INTERPRETER_RAN, BACKEND_INTERPRETER, BACKEND_INTERPRETER, True, True, "any_exec", False, ("parent", "case"), True),
    "fast-ran":               (KIND_COMPLETE,    OUTCOME_FAST_RAN, BACKEND_NUMBA, BACKEND_NUMBA, False, False, None, False, ("parent", "case"), False),
    "interpreter-failed":     (KIND_NON_PASS,    ("prefix", OUTCOME_INTERPRETER_FAILED + ":"), BACKEND_INTERPRETER, None, True, True, "any_exec", False, ("parent", "case"), True),
    "untyped-decline":        (KIND_NON_PASS,    OUTCOME_UNTYPED_DECLINE, BACKEND_NUMBA, None, True, False, STAGE_LOWERING, "any", ("parent", "case", "stage"), False),
    "numba-unavailable":      (KIND_NON_PASS,    OUTCOME_NUMBA_UNAVAILABLE, BACKEND_NUMBA, None, True, False, STAGE_LOWERING, "any", ("parent", "case", "stage"), False),
    "precheck-decline":       (KIND_PREPARATION, OUTCOME_PRECHECK_DECLINE, BACKEND_NUMBA, None, True, True, STAGE_LOWERING, True, ("stage",), False),
}
CONTROLLER_OUTPUT_EVENTS = ("resource-stop", "open-transition", "kill-record")   # controller outputs are never events
_EVENT_FILE = re.compile(r"^(?P<tag>.*?)fallback-(?P<event>[a-z_-]+)\.json$")


def _tag_name(tag: str) -> str:
    return tag.rstrip(".-_")


def _state_ok(event: str, rec: dict) -> Optional[str]:
    """None when the record's state tuple is exactly the table row for ``event``; else the diagnostic."""
    row = STATE_TABLE.get(event)
    if row is None:
        return "UNKNOWN_EVENT"
    kind, outcome, b_att, b_term, dec_present, dec_typed, dec_stage, prep, _scopes, _needs = row
    if isinstance(outcome, tuple):
        if not str(rec["outcome"]).startswith(outcome[1]) or not isinstance(rec.get("interpreter_failure"), dict):
            return "STATE_NOT_ALLOWED"
    elif rec["outcome"] != outcome:
        return "STATE_NOT_ALLOWED"
    if rec["backend_attempt"] != b_att or rec["backend_terminal"] != b_term:
        return "STATE_NOT_ALLOWED"
    present = rec["decline_reason"] is not None
    if present != dec_present or rec["decline_typed"] != dec_typed:
        return "STATE_NOT_ALLOWED"
    stage = rec["decline_stage"]
    if dec_stage == "any_exec":
        if stage not in STAGES: return "STATE_NOT_ALLOWED"
    elif dec_stage == "any":
        pass
    elif stage != dec_stage:
        return "STATE_NOT_ALLOWED"
    if prep != "any" and bool(rec.get("preparation_only", False)) != prep:
        return "STATE_NOT_ALLOWED"
    return None


_LOWERED_NULL_ROWS = ("transition-lowering", "untyped-decline", "numba-unavailable", "precheck-decline")
_LOWERED_SET_ROWS = ("transition-module_exec", "transition-run_model", "fast-ran")


def _ts(v: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(v)


def _cross_field(event: str, rec: dict) -> List[str]:
    """PROTOCOL v3 I1-I9 for one structurally valid row record; returns the violated invariant ids (empty = consistent)."""
    bad: List[str] = []
    if rec["backend"] != rec["backend_attempt"]: bad.append("I1")
    if rec["fallback_allowed"] is not True: bad.append("I2")
    if rec["fallback_taken"] != (rec["backend_attempt"] == BACKEND_INTERPRETER): bad.append("I3")
    expected_program = "not_executed_preparation" if rec["preparation_only"] else ("numba_lowered" if rec["backend_attempt"] == BACKEND_NUMBA else "original_dsl_interpreter")
    if rec["executed_program"] != expected_program: bad.append("I4")
    f = rec["interpreter_failure"]
    if (f is not None) != (event == "interpreter-failed"): bad.append("I5")
    elif f is not None and (f.get("prior_decline_stage") != rec["decline_stage"] or f.get("prior_decline_reason") != rec["decline_reason"]): bad.append("I5")
    if event == "fast-ran":
        if rec["lowered_sha256"] is None or rec["program_text_sha256"] != rec["lowered_sha256"]: bad.append("I6")
    elif rec["program_text_sha256"] != rec["source_sha256"]: bad.append("I6")
    lowered_null = rec["lowered_sha256"] is None
    if event in _LOWERED_NULL_ROWS: ok7 = lowered_null
    elif event in _LOWERED_SET_ROWS: ok7 = not lowered_null
    else: ok7 = lowered_null == (rec["decline_stage"] == STAGE_LOWERING)
    if not ok7: bad.append("I7")
    h = rec["history"]
    if event.startswith("transition-") or event in CLOSING_EVENTS or event == "precheck-decline":
        if not h or h[-1] != {"backend": BACKEND_NUMBA, "stage": rec["decline_stage"], "reason": rec["decline_reason"], "typed": True}: bad.append("I8")
    elif h: bad.append("I8")
    if event not in REEMIT_EVENTS and rec["events"] != [{"event": rec["event"], "event_utc": rec["event_utc"]}]: bad.append("I9")   # v4: exactly ONE event, its own (re-emit rows: derived trail)
    if rec["decline_typed"] and rec["decline_stage"] == STAGE_LOWERING and not rec["decline_hook_available"]: bad.append("I10")        # v4: a typed LOWERING reason needs the hook
    return bad


def _reemit_diff(event: str, rec: dict, surviving: Optional[dict], descriptor: LaunchDescriptor, scope: str, name: str, ctx: AttemptContext, now: datetime.datetime) -> List[str]:
    """PROTOCOL v4 section 3b: rebuild the expected closing record with the emitter's own function and list every deterministic
    difference (field paths) plus the free-field constraint violations."""
    bad: List[str] = []
    if _ts(rec["event_utc"]) > now + datetime.timedelta(seconds=_FREE_TIME_SKEW_S): bad.append("event_utc")
    if event == "fast-ran":
        ident = descriptor.identity()
        # PROTOCOL v5: the context is the RESOLVED attempt context; the record's own value enters only for a field still unresolved (the learner)
        binding = {"out_dir": ctx.value("binding.out_dir", rec), "run_dir": descriptor.run_dir}
        for k in ATTEMPT_IDENTITY_FIELDS:
            if ident.get(k) is not None: binding[k] = ident[k]
        if scope == "case": binding["case"] = name
        elif scope == "stage": binding["stage"] = name
        context = {"binding": binding, "engine": ctx.engine(rec), "source_sha256": ctx.value("source_sha256", rec), "decline_hook_available": ctx.value("decline_hook_available", rec)}
        exp = expected_closing_record(None, context, "fast-ran", {"event_utc": rec["event_utc"], "lowered_sha256": rec["lowered_sha256"]})
    else:
        free = {"event_utc": rec["event_utc"]}
        if event == "interpreter-failed":
            f = rec["interpreter_failure"] if isinstance(rec.get("interpreter_failure"), dict) else {}
            free["failure_type"] = f.get("type", ""); free["failure_message"] = f.get("message", "")
            if not isinstance(f.get("message"), str) or len(f.get("message")) > 1000: bad.append("interpreter_failure.message")
        exp = expected_closing_record(surviving, None, event, free)
    bad.extend(d for d in _deep_diff(exp, rec) if d not in bad)
    return bad


def classify_event(event: str, ent: dict, tag: str, descriptor: LaunchDescriptor, surviving: Optional[dict] = None, ctx: Optional[AttemptContext] = None) -> dict:
    """PROTOCOL sections 2-4 for ONE record: returns {"valid", "kind", "scope", "code", "reasons"}; NEVER raises."""
    out = {"valid": False, "kind": None, "scope": None, "code": None, "reasons": []}
    try:
        if "unreadable" in ent:
            out["code"] = "UNREADABLE"; out["reasons"] = [ent["unreadable"]]; return out
        rec = ent["record"]
        ok, reasons = validate_record_safe(rec)
        if not ok:
            out["code"] = reasons[0].split(":")[0] if reasons[0].startswith(("NOT_AN_OBJECT", "CONTRACT_MISMATCH", "TIME_INVALID")) else "SCHEMA"; out["reasons"] = reasons; return out
        if event not in STATE_TABLE:
            out["code"] = "UNKNOWN_EVENT"; out["reasons"] = [f"event {event!r} not in the table"]; return out
        if rec["event"] != event:
            out["code"] = "EVENT_FILENAME_MISMATCH"; out["reasons"] = [f"record.event {rec['event']!r} != filename event {event!r}"]; return out
        st = _state_ok(event, rec)
        if st:
            out["code"] = st; out["reasons"] = [f"state tuple (outcome={rec['outcome']!r}, attempt={rec['backend_attempt']!r}, terminal={rec['backend_terminal']!r}, decline={rec['decline_reason'] is not None}, typed={rec['decline_typed']}, stage={rec['decline_stage']!r}, prep={rec.get('preparation_only', False)}) is not the table row for {event!r}"]; return out
        kind, _o, _a, _t, _d, _ty, _s, _p, scopes, _needs = STATE_TABLE[event]
        out["kind"] = kind
        b = rec["binding"]; name = _tag_name(tag)
        if tag == "":
            if b.get("case") is not None or b.get("stage") is not None:
                out["code"] = "SUBATTEMPT_MISMATCH"; out["reasons"] = ["parent-scope record carries a case/stage"]; return out
            scope = "parent"
        else:
            if b.get("case") is not None and b.get("case") == name:
                scope = "case"
            elif b.get("stage") is not None and b.get("stage") == name:
                scope = "stage"
            else:
                out["code"] = "SUBATTEMPT_MISMATCH"; out["reasons"] = [f"tag {name!r} vs binding case={b.get('case')!r} stage={b.get('stage')!r}"]; return out
            declared_kind = descriptor.tag_kind(name)
            if declared_kind is not None and declared_kind != scope:
                out["code"] = "SCOPE_CONFLICT"; out["reasons"] = [f"tag {name!r} is declared as a {declared_kind} but the record binds it as a {scope}"]; return out
            declared = descriptor.cases if scope == "case" else descriptor.stages
            if declared is not None and name not in declared:
                out["code"] = "SUBATTEMPT_MISMATCH"; out["reasons"] = [f"{scope} {name!r} is not declared by the launch descriptor"]; return out
        out["scope"] = scope
        if scope not in scopes:
            out["code"] = "SCOPE_NOT_ALLOWED"; out["reasons"] = [f"{event} is not allowed under the {scope} scope"]; return out
        bm = bindings_match(b, descriptor)
        if not bm["matches"]:
            out["code"] = "IDENTITY_MISMATCH"; out["reasons"] = bm["reasons"]; return out
        ctx = ctx if ctx is not None else AttemptContext(descriptor)
        mm = ctx.mismatches(rec)          # PROTOCOL v5 section 1b: the RESOLVED attempt context (anchored or learned) is enforced before admission (after scope and identity, before time)
        if mm:
            codes = {c for c, _ in mm}
            out["code"] = next(c for c in ("ENGINE_PIN_MISMATCH", "SOURCE_MISMATCH", "ENGINE_HASH_MISMATCH", "CONTEXT_MISMATCH") if c in codes)
            out["reasons"] = [f"{c}:{f}: record {_path_get(rec, f)!r} != {'anchored' if f in ctx.anchored else 'learned'} {ctx.get(f)!r}" for c, f in mm]; return out
        t_rec = datetime.datetime.fromisoformat(rec["event_utc"])
        if descriptor.minted_utc and t_rec < datetime.datetime.fromisoformat(descriptor.minted_utc):
            out["code"] = "TIME_BEFORE_MINT"; out["reasons"] = [f"{rec['event_utc']} < minted {descriptor.minted_utc}"]; return out
        if surviving is not None and kind != KIND_TRANSITION and t_rec < datetime.datetime.fromisoformat(surviving["event_utc"]):
            out["code"] = "TIME_BEFORE_TRANSITION"; out["reasons"] = [f"{rec['event_utc']} < transition {surviving['event_utc']}"]; return out
        bad = _cross_field(event, rec)          # PROTOCOL v4 I1-I10 (after the identity / source / engine / time anchors, which report first)
        if bad:
            code = "TERMINAL_INCONSISTENT" if event in REEMIT_EVENTS else "FIELD_INCONSISTENT"
            out["code"] = code; out["reasons"] = [f"{code}:{i}" for i in bad]; return out
        if event in REEMIT_EVENTS and (event == "fast-ran" or surviving is not None):
            # PROTOCOL v4 section 3b -- RE-EMIT AND COMPARE: rebuild the record the emitter would have written and require equality
            bad = _reemit_diff(event, rec, surviving, descriptor, scope, name, ctx, datetime.datetime.now(datetime.timezone.utc))
            if bad:
                out["code"] = "TERMINAL_INCONSISTENT"; out["reasons"] = [f"TERMINAL_INCONSISTENT:{i}" for i in bad]; return out
        out["valid"] = True; return out
    except Exception as exc:  # the guarantee
        out["code"] = "SCHEMA"; out["reasons"] = [f"exception during classification: {type(exc).__name__}: {exc}"]; return out


def scan_fallback_events(out_dir) -> Dict[str, Dict[str, dict]]:
    """``{tag: {event: {"path", "record" | "unreadable"}}}``; the mint and the controller outputs are not events."""
    out: Dict[str, Dict[str, dict]] = {}
    for p in sorted(Path(out_dir).iterdir()):
        m = _EVENT_FILE.match(p.name)
        if not m or not p.is_file() or p.name == MINT_FILE or m.group("event") in CONTROLLER_OUTPUT_EVENTS or m.group("event") == "attempt":
            continue
        try:
            rec = json.loads(p.read_bytes())
            ent = {"path": str(p), "record": rec} if isinstance(rec, dict) else {"path": str(p), "unreadable": "NOT_AN_OBJECT"}
        except Exception as exc:
            ent = {"path": str(p), "unreadable": f"{type(exc).__name__}: {exc}"}
        out.setdefault(m.group("tag"), {})[m.group("event")] = ent
    return out


def resource_stop_record(transition_record: dict, reason: str, attempt=None) -> dict:
    """The controller's DISTINCT joined record for a resource kill during the fallback attempt (prior typed decline +
    attempt identity + null verdict), built from the surviving VALID transition."""
    rec = dict(transition_record)
    ident = attempt.identity() if isinstance(attempt, LaunchDescriptor) else dict(attempt or {})
    rec["event"] = "resource-stop"; rec["event_utc"] = _utc(); rec["outcome"] = OUTCOME_RESOURCE_STOP; rec["backend_terminal"] = None
    rec["resource_stop"] = {"reason": reason, "during": "interpreter attempt after a typed decline", "prior_decline_reason": transition_record.get("decline_reason"), "prior_decline_stage": transition_record.get("decline_stage"),
                            "transition_event": transition_record.get("event"), "transition_event_utc": transition_record.get("event_utc"), "attempt": ident}
    rec["events"] = [dict(e) for e in transition_record.get("events", [])] + [{"event": "resource-stop", "event_utc": rec["event_utc"]}]
    rec["verdict"] = None
    return rec


def check_mint(out_dir, descriptor: LaunchDescriptor) -> dict:
    """The disk mint is validated against the parent-held descriptor; it never substitutes identity."""
    p = Path(out_dir) / MINT_FILE
    if descriptor.mint_sha256 is None:
        return {"state": "NO_MINT_EXPECTED" if not p.exists() else "UNEXPECTED_MINT_IGNORED", "ok": True}
    if not p.exists():
        return {"state": "MINT_MISSING", "ok": False}
    try:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception as exc:
        return {"state": "MINT_ALTERED", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    if h != descriptor.mint_sha256:
        return {"state": "MINT_ALTERED", "ok": False, "disk_sha256": h, "parent_sha256": descriptor.mint_sha256}
    return {"state": "MINT_OK", "ok": True, "sha256": h}


_STAGE_ORDER = {STAGE_LOWERING: 0, STAGE_MODULE_EXEC: 1, STAGE_RUN_MODEL: 2}
FAST_FAILED_EVENTS = ("untyped-decline", "numba-unavailable")
TRANSITION_EVENTS = tuple(e for e in STATE_TABLE if e.startswith("transition-"))


def classify_valid_set(scope: str, valid: Dict[str, dict]) -> dict:
    """PROTOCOL v2 sections 4-5: the WHOLE valid set of a scope -> one machine state.  No priority order.
    Returns {"state", "surviving" (transition record or None), "closing" (event or None), "conflicting": [events]}."""
    trans = {e: r for e, r in valid.items() if e in TRANSITION_EVENTS}
    others = {e: r for e, r in valid.items() if e not in TRANSITION_EVENTS}
    surviving = max(trans.values(), key=lambda r: (r["event_utc"], _STAGE_ORDER.get(r["decline_stage"], -1))) if trans else None
    if scope == "stage":
        keys = sorted(valid)
        if keys == ["precheck-decline"]: return {"state": "PREPARATION_RECORDED", "surviving": None, "closing": "precheck-decline", "conflicting": []}
        if len(keys) == 1 and keys[0] in FAST_FAILED_EVENTS: return {"state": "FAST_FAILED", "surviving": None, "closing": keys[0], "conflicting": []}
        if not keys: return {"state": "NO_VALID_EVENT", "surviving": None, "closing": None, "conflicting": []}
        return {"state": "STATE_CONFLICT", "surviving": None, "closing": None, "conflicting": keys}
    keys = sorted(valid)
    if not keys:
        return {"state": "NO_VALID_EVENT", "surviving": None, "closing": None, "conflicting": []}
    if keys == ["fast-ran"]: return {"state": "FAST_DONE", "surviving": None, "closing": "fast-ran", "conflicting": []}
    if len(keys) == 1 and keys[0] in FAST_FAILED_EVENTS: return {"state": "FAST_FAILED", "surviving": None, "closing": keys[0], "conflicting": []}
    if len(trans) == 1 and not others: return {"state": "OPEN", "surviving": surviving, "closing": None, "conflicting": []}
    if len(trans) == 1 and sorted(others) in (["interpreter-ran"], ["interpreter-failed"]):
        ev = sorted(others)[0]; rec = others[ev]
        if datetime.datetime.fromisoformat(rec["event_utc"]) >= datetime.datetime.fromisoformat(surviving["event_utc"]):
            return {"state": "CLOSED_OK" if ev == "interpreter-ran" else "CLOSED_FAIL", "surviving": surviving, "closing": ev, "conflicting": []}
        return {"state": "STATE_CONFLICT", "surviving": surviving, "closing": None, "conflicting": [ev]}
    # everything else: contradictory set; the latest transition (if any) is the surviving evidence, nothing closes
    return {"state": "STATE_CONFLICT", "surviving": surviving, "closing": None, "conflicting": [e for e in keys if e not in trans or (surviving is not None and valid[e] is not surviving)]}


def _kill_record(out_dir: Path, tag: str, descriptor: LaunchDescriptor, child_state: str, child_exit, guard_kill: str, rejected: List[dict], state: str = "NO_VALID_EVENT", valid_events: Optional[List[str]] = None) -> str:
    """RESOURCE_STOP_UNJOINED (PROTOCOL v3 section 7): a guard kill with NO valid surviving transition and NO consistent completion,
    whatever other rows exist -- recorded honestly, no prior decline claimed, nothing manufactured from malformed input."""
    rec = {"contract": CONTRACT, "event": "kill-record", "event_utc": _utc(), "outcome": "RESOURCE_STOP_UNJOINED", "attempt": descriptor.identity(), "tag": tag, "child_state": child_state, "child_exit": child_exit, "guard_kill": guard_kill,
           "scope_state": state, "valid_non_closing_events": list(valid_events or []), "rejected_events": rejected, "prior_decline_claimed": False, "verdict": None}
    p = out_dir / f"{tag}fallback-kill-record.json"; p.write_text(json.dumps(rec, indent=1) + "\n"); return str(p)


def _classify_tag(tag: str, evs: Dict[str, dict], attempt: LaunchDescriptor, ctx: AttemptContext, tentative: bool) -> dict:
    """One tag through the per-record checks, the scope rule and the state machine against ``ctx`` (tentative learning only when
    asked).  PURE: writes nothing.  Returns valid / rejected / scope / machine result / info and the ADMISSIBLE SURVIVORS
    (records still valid after the machine; transitions first) that may COMMIT context values."""
    info = {"events": sorted(evs), "valid_events": [], "rejected_events": [], "conflicting_events": [], "scope": None, "state": None, "surviving_transition": None, "closing_event": None, "status": None, "non_pass": None}
    rejected: List[dict] = []; valid: Dict[str, dict] = {}; scopes: Dict[str, str] = {}
    for ev, ent in evs.items():                                    # pass 1: transitions (reference for the terminals)
        if ev in TRANSITION_EVENTS:
            c = classify_event(ev, ent, tag, attempt, None, ctx)
            if c["valid"]:
                valid[ev] = ent["record"]; scopes[ev] = c["scope"]
                if tentative: ctx.learn_tentative(ent["record"], ent["path"])
            else: rejected.append({"event": ev, "path": ent["path"], "code": c["code"], "reasons": c["reasons"]})
    surviving = max(valid.values(), key=lambda r: (r["event_utc"], _STAGE_ORDER.get(r["decline_stage"], -1))) if valid else None
    for ev, ent in evs.items():                                    # pass 2: everything else
        if ev in TRANSITION_EVENTS: continue
        c = classify_event(ev, ent, tag, attempt, surviving, ctx)
        if c["valid"]:
            valid[ev] = ent["record"]; scopes[ev] = c["scope"]
            if tentative: ctx.learn_tentative(ent["record"], ent["path"])
        else:
            rejected.append({"event": ev, "path": ent["path"], "code": c["code"], "reasons": c["reasons"]})
    kinds = {}
    for ev, sc in scopes.items(): kinds.setdefault(sc, []).append(ev)
    declared = attempt.tag_kind(_tag_name(tag)) if tag else "parent"
    if len(kinds) > 1:                                             # mixed kinds -> SCOPE_CONFLICT for the minority
        keep = declared if declared in kinds else max(kinds, key=lambda k: len(kinds[k]))
        for sc, evl in kinds.items():
            if sc != keep:
                for ev in evl:
                    rejected.append({"event": ev, "path": evs[ev]["path"], "code": "SCOPE_CONFLICT", "reasons": [f"mixed kinds under tag {tag!r}: {sc} record among {keep} records"]}); valid.pop(ev, None)
        kinds = {keep: kinds[keep]}
    scope = declared if declared and declared != "parent" else (next(iter(kinds)) if kinds else ("parent" if tag == "" else None))
    if tag == "": scope = "parent"
    info["scope"] = scope
    m = classify_valid_set("stage" if scope == "stage" else "exec", valid)
    if m["state"] == "STATE_CONFLICT":                             # amendment 1: the conflicting set is REJECTED except the surviving transition
        for ev in m["conflicting"]:
            rejected.append({"event": ev, "path": evs[ev]["path"], "code": "STATE_CONFLICT", "reasons": [f"forbidden in state STATE_CONFLICT of scope {scope}: set {sorted(valid)}" + (f", surviving transition {next(e for e, r in valid.items() if r is m['surviving'])!r}" if m["surviving"] is not None else ", no transition")]})
        for ev in m["conflicting"]: valid.pop(ev, None)
    info.update(state=m["state"], valid_events=sorted(valid), surviving_transition=next((e for e, r in valid.items() if r is m["surviving"]), None), closing_event=m["closing"], conflicting_events=m["conflicting"], rejected_events=rejected,
                transitions=sorted(e for e in valid if e in TRANSITION_EVENTS), terminals=sorted(e for e in valid if e not in TRANSITION_EVENTS and e not in PREPARATION_EVENTS), preparation=sorted(e for e in valid if e in PREPARATION_EVENTS))
    survivors = [(ev, valid[ev], evs[ev]["path"]) for ev in valid if ev in TRANSITION_EVENTS] + [(ev, valid[ev], evs[ev]["path"]) for ev in valid if ev not in TRANSITION_EVENTS]
    return {"info": info, "rejected": rejected, "valid": valid, "scope": scope, "m": m, "survivors": survivors}


def postprocess_attempt(out_dir, *, attempt: LaunchDescriptor, child_state: str, child_exit, guard_kill: Optional[str], filename: str = "FALLBACK_POSTPROCESS.json") -> dict:
    """PROTOCOL v3 sections 2-7, run by the CONTROLLER after the child stopped with its PARENT-HELD descriptor."""
    if not isinstance(attempt, LaunchDescriptor):
        raise RecordInvalid("postprocess_attempt needs the parent-held LaunchDescriptor; a dict or anything read from disk is not an authority")
    out_dir = Path(out_dir)
    summary = {"schema": "lite-fallback-postprocess-v8", "contract": CONTRACT, "postprocess_utc": _utc(), "attempt": attempt.as_dict(), "mode": attempt.mode, "child_state": child_state, "child_exit": child_exit, "guard_kill": guard_kill,
               "mint_check": check_mint(out_dir, attempt), "tags": {}, "written": [], "rejected_events": [], "non_pass": False, "verdict": None, "execution_tags": 0, "preparation_only_tags": 0, "preparation_without_execution": False,
               "execution_evidence": False, "no_execution_evidence": False, "mode_violations": [], "case_coverage": {}, "missing_cases": []}
    if not summary["mint_check"]["ok"]:
        summary["non_pass"] = True
    events = scan_fallback_events(out_dir)
    if not events:
        # PROTOCOL v3 section 6 (P2-R6-EMPTY-ATTEMPT-1): NO execution evidence -> non-pass in EVERY mode, whatever the exit code; the
        # process exit code never establishes closure.  Under a guard kill the honest unjoined kill record is written as well.
        st = "NO_EXECUTION_EVIDENCE" + ("+RESOURCE_STOP_BEFORE_ANY_FALLBACK_EVENT" if guard_kill else "")
        info = {"events": [], "valid_events": [], "rejected_events": [], "conflicting_events": [], "scope": "parent", "state": "NO_EXECUTION_EVIDENCE", "surviving_transition": None, "closing_event": None, "status": st, "non_pass": True}
        if guard_kill:
            summary["written"].append(_kill_record(out_dir, "", attempt, child_state, child_exit, guard_kill, [], state="NO_EXECUTION_EVIDENCE")); info["status"] += "+RESOURCE_STOP_UNJOINED"
        summary["tags"][""] = info; summary["no_execution_evidence"] = True; summary["non_pass"] = True
    order = sorted(events, key=lambda t: (t != "", t))          # PROTOCOL v5/v6 classification order: the parent tag first, then tags in sorted order
    # PROTOCOL v6 section 1b -- ADMISSION BOUNDARY, two-phase resolution:
    #   phase 1: classify every tag with a TENTATIVE context (values from individually valid records), run the machine, collect the admissible survivors;
    #   phase 2: COMMIT learned values only from the survivors (parent tag first, sorted tags, transitions before other rows);
    #   phase 3: re-validate every tag against the committed context FROZEN; repeat 2-3 until the commit is a fixed point (bounded, fail closed);
    #   phase 4: write records from the FINAL classification only.
    committed = AttemptContext(attempt); passes: List[dict] = []
    ctx = AttemptContext(attempt)                                  # phase 1: tentative learning allowed
    results = {tag: _classify_tag(tag, events[tag], attempt, ctx, tentative=True) for tag in order}
    tentative_seen = dict(ctx.tentative); tentative_from = dict(ctx.tentative_from)
    fixed_point = False
    for it in range(_MAX_CONTEXT_ROUNDS):
        new = AttemptContext(attempt, frozen=True)
        for tag in order:
            for ev, rec, path in results[tag]["survivors"]: new.learn(rec, path)
        passes.append({"pass": it + 1, "mode": "tentative" if it == 0 else "frozen", "learned": _jcopy(new.learned), "learned_from": dict(new.learned_from), "survivors": {t: [ev for ev, _, _ in results[t]["survivors"]] for t in order}})
        if it == 0 and not _deep_diff(new.learned, tentative_seen) and new.learned_from == tentative_from:
            fixed_point = True; passes[-1]["final"] = "tentative == committed: no learner was rejected; a frozen re-validation would reach identical admission/survivor/completion dispositions (diagnostics of rejected records are those of the tentative evaluation and may differ in code from a frozen pass)"; committed = new; break
        if it > 0 and not _deep_diff(new.learned, committed.learned) and new.learned_from == committed.learned_from:
            fixed_point = True; break
        committed = new
        results = {tag: _classify_tag(tag, events[tag], attempt, committed, tentative=False) for tag in order}     # phase 3: frozen re-validation
    ctx = committed
    summary["engine_context"] = {**ctx.as_dict(), "tentative_discarded": {f: {"value": v, "from": tentative_from.get(f)} for f, v in tentative_seen.items() if f not in ctx.learned or _deep_diff(ctx.learned[f], v, f)}, "passes": passes, "fixed_point": fixed_point}
    if not fixed_point:
        summary["non_pass"] = True; summary["context_resolution_failed"] = f"no fixed point after {_MAX_CONTEXT_ROUNDS} re-validation rounds"      # fail closed
    for tag in order:                                                # phase 4: statuses and records from the FINAL classification only
        r = results[tag]; evs = events[tag]; info = r["info"]; rejected = r["rejected"]; valid = r["valid"]; scope = r["scope"]; m = r["m"]
        summary["rejected_events"].extend({"tag": tag, **r} for r in rejected)
        suffix = "".join(f"+REJECTED:{r['event']}:{r['code']}" for r in rejected)
        st = m["state"]
        if st == "CLOSED_OK":
            info["status"] = "CLOSED_COMPARATOR_DECIDES:interpreter-ran" + suffix; info["non_pass"] = bool(guard_kill) or child_exit != 0 or bool(rejected)
            if guard_kill: info["status"] += "+GUARD_KILL_AFTER_TERMINAL"
        elif st == "FAST_DONE":
            info["status"] = "CLOSED_COMPARATOR_DECIDES:fast-ran" + suffix; info["non_pass"] = bool(guard_kill) or child_exit != 0 or bool(rejected)
            if guard_kill: info["status"] += "+GUARD_KILL_AFTER_TERMINAL"
        elif st == "CLOSED_FAIL":
            info["status"] = "CLOSED_NON_PASS:interpreter-failed" + suffix; info["non_pass"] = True
            if guard_kill: info["status"] += "+GUARD_KILL_AFTER_TERMINAL"
        elif st == "FAST_FAILED":
            info["status"] = "CLOSED_NON_PASS:" + m["closing"] + suffix; info["non_pass"] = True
        elif st == "PREPARATION_RECORDED":
            info["status"] = "PREPARATION_RECORDED:precheck-decline" + suffix; info["non_pass"] = bool(rejected)
        elif st in ("OPEN", "STATE_CONFLICT") and m["surviving"] is not None:
            # the joined record is ALWAYS written when a valid transition survives without a consistent closing record
            conflict_note = ("STATE_CONFLICT:" + ",".join(m["conflicting"])) if st == "STATE_CONFLICT" else ""
            if guard_kill:
                rs = resource_stop_record(m["surviving"], f"{guard_kill}: child {child_state} during the interpreter attempt (exit {child_exit})", attempt); rs["rejected_events"] = rejected; rs["conflicting_events"] = m["conflicting"]
                p = out_dir / f"{tag}fallback-resource-stop.json"; p.write_text(json.dumps(rs, indent=1) + "\n"); summary["written"].append(str(p))
                info["status"] = OUTCOME_RESOURCE_STOP + (("+" + conflict_note) if conflict_note else "") + suffix
            else:
                op = dict(m["surviving"]); op.update(event="open-transition", event_utc=_utc(), outcome=OUTCOME_OPEN_TRANSITION, backend_terminal=None, verdict=None, rejected_events=rejected, conflicting_events=m["conflicting"],
                                                    open_transition={"child_state": child_state, "child_exit": child_exit, "attempt": attempt.identity(), "reason": "transition persisted but no consistent closing record (" + (conflict_note or "none") + ")"})
                op["events"] = [dict(e) for e in m["surviving"].get("events", [])] + [{"event": "open-transition", "event_utc": op["event_utc"]}]
                p = out_dir / f"{tag}fallback-open-transition.json"; p.write_text(json.dumps(op, indent=1) + "\n"); summary["written"].append(str(p))
                info["status"] = OUTCOME_OPEN_TRANSITION + (("+" + conflict_note) if conflict_note else "") + suffix
            info["non_pass"] = True
        elif st == "STATE_CONFLICT":
            info["status"] = "STATE_CONFLICT:" + ",".join(m["conflicting"]) + suffix; info["non_pass"] = True
        else:  # NO_VALID_EVENT
            info["status"] = "NO_VALID_EVENT" + suffix; info["non_pass"] = True
        # PROTOCOL v3 section 7 (P2-R6-UNJOINED-KILL-1): the unjoined kill record is an obligation INDEPENDENT of which other rows
        # exist -- guard kill + no valid surviving transition (no joined record) + no consistent completion -> RESOURCE_STOP_UNJOINED
        if guard_kill and m["surviving"] is None and st not in ("FAST_DONE", "CLOSED_OK", "CLOSED_FAIL", "PREPARATION_RECORDED"):
            summary["written"].append(_kill_record(out_dir, tag, attempt, child_state, child_exit, guard_kill, rejected, state=st, valid_events=sorted(valid))); info["status"] += "+RESOURCE_STOP_UNJOINED"
        if scope in ("parent", "case") and valid: summary["execution_tags"] += 1
        elif scope == "stage" and m["state"] == "PREPARATION_RECORDED": summary["preparation_only_tags"] += 1   # a stage FAST_FAILED is a visible fast-stage non-pass, not a preparation
        summary["tags"][tag] = info; summary["non_pass"] = summary["non_pass"] or bool(info["non_pass"])
    # required-case coverage: ONLY case-scope execution evidence counts; its closure status applies
    if attempt.cases:
        by_name = {_tag_name(t): i for t, i in summary["tags"].items() if t}
        for c in attempt.cases:
            i = by_name.get(c)
            if i is None or i.get("scope") != "case" or not i.get("valid_events"):
                summary["case_coverage"][c] = "MISSING_CASE"; summary["missing_cases"].append(c); summary["non_pass"] = True
            else:
                summary["case_coverage"][c] = i["state"]
                if i["state"] not in ("FAST_DONE", "CLOSED_OK"): summary["non_pass"] = True
    if events and summary["execution_tags"] == 0 and summary["preparation_only_tags"] > 0:
        summary["preparation_without_execution"] = True; summary["non_pass"] = True
    summary["execution_evidence"] = summary["execution_tags"] > 0
    if not summary["execution_evidence"]:
        summary["no_execution_evidence"] = True; summary["non_pass"] = True       # fail closed: no validated execution evidence in any scope
    if attempt.mode == "no-fallback-expected":
        for tag, info in summary["tags"].items():
            viol = [e for e in info.get("valid_events", []) if e in TRANSITION_EVENTS or e in CLOSING_EVENTS]
            if viol:
                summary["mode_violations"].append({"tag": tag, "events": viol, "code": "MODE_VIOLATION"}); info["status"] = str(info["status"]) + "+MODE_VIOLATION"; info["non_pass"] = True; summary["non_pass"] = True
    p = out_dir / filename; p.write_text(json.dumps(summary, indent=1) + "\n"); summary["written"].append(str(p))
    return summary


__all__ = ["CONTRACT", "BACKEND_NUMBA", "BACKEND_INTERPRETER", "STAGE_LOWERING", "STAGE_MODULE_EXEC", "STAGE_RUN_MODEL",
           "OUTCOME_FAST_LOWERED", "OUTCOME_FAST_RAN", "OUTCOME_UNTYPED_DECLINE", "OUTCOME_NUMBA_UNAVAILABLE", "OUTCOME_INTERPRETER_PENDING",
           "OUTCOME_INTERPRETER_RAN", "OUTCOME_INTERPRETER_FAILED", "OUTCOME_RESOURCE_STOP", "OUTCOME_OPEN_TRANSITION", "TERMINAL_EVENTS",
           "REQUIRED_RECORD_FIELDS", "REQUIRED_ENGINE_FIELDS", "FastStageFailure", "UntypedDecline", "NumbaUnavailable", "InterpreterFailed",
           "RecordInvalid", "engine_identity", "validate_record", "RECORD_SCHEMA", "INVARIANTS", "REEMIT", "REEMIT_EVENTS", "CONTEXT", "CONTEXT_FIELDS", "ANCHOR_DOMAIN", "AttemptContext", "expected_closing_record", "CLOSING_EVENTS", "MODES", "PROTOCOL_DIR", "classify_valid_set", "PreparedInputs", "prepared_inputs", "CandidateProgram", "lower_or_fallback",
           "execute_candidate", "resource_stop_record", "scan_fallback_events", "postprocess_attempt", "interpreter_namespace",
           "OUTCOME_PRECHECK_DECLINE", "PREPARATION_EVENTS", "ATTEMPT_IDENTITY_FIELDS", "MINT_FILE", "STATE_TABLE", "canonical_run_dir", "LaunchDescriptor", "mint_attempt", "attempt_binding", "bindings_match", "classify_event", "check_mint", "validate_record_safe"]
