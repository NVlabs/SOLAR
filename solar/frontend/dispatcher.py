# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic-to-agentic fallback dispatcher.

Route policy (the ONE policy this module implements):

1. The deterministic/legacy route runs first: PyTorch capture
   (``solar.graph.PyTorchProcessor``) -> evidence-based route decision
   (``solar.frontend.router``) -> registered op-to-Einsum lowering ->
   analysis -> perf.  A complete lowering is the result; the dispatcher
   records that this route performs NO numerical validation.
2. An *eligible* deterministic outcome — an unsupported capture/lowering
   (router routes ``agentic``/``unsupported``, a layer is not einsum
   supportable, a stage reports ``status: incomplete``) or a failed
   deterministic numerical validation (when the caller supplies one) —
   records a concise exact reason, organised as ``deterministic_limitation``
   + ``fallback_reason``, and automatically enters the SELECTED v2 agentic
   route (``pure_skill`` or ``pythonic``) under a bounded policy
   (``DispatchPolicy``: attempts, total wall, API requests, cost).  The
   original deterministic failure receipt is retained verbatim and is never
   overwritten.
3. Data-dependent-sparsity op families are agentic-by-default
   (``DATA_DEPENDENT_SPARSITY_OPS`` membership, or a caller-declared
   ``sparsity_family_hint``).  Op-name membership is a CONSERVATIVE ROUTING
   HEURISTIC, not proof that the program is sparse: the mechanical route
   bills these ops dense and cannot establish measured density, so the row
   is routed to the DSL route and its SOL must come from the DSL-route
   sparse (measured-density) analysis.  An explicit caller declaration
   takes precedence over detection and is carried through EVERY
   deterministic outcome (raw sources, capture exceptions, stage errors).
   The deterministic attempt (capture, route decision) is still run and
   retained in the report.
4. A deterministic INFRASTRUCTURE or STAGE error (timeout, OOM, host
   contention, missing/unreadable capture artifact) is NOT eligible: it is
   retained for retry with a durable, unscored receipt and no fallback is
   launched — falling back would hide the fault.
5. Every agentic candidate is validated by the dispatcher itself through
   ``solar.translate.verify`` under an EXPLICIT tolerance policy (request
   ``tolerance`` -> selected ``workload.jsonl`` line -> unresolved; the
   dispatcher never substitutes the facade defaults) and, when a definition
   and dtype are given, bound into a receipt through
   ``solar.translate.receipts.emit_workload_receipt``.  An unresolved
   tolerance policy stays unresolved: the candidate executes (candidate
   defects are still detected) but no pass/fail is admitted.
6. Fail closed: missing provider, missing input, missing budget/reservation
   stop the dispatch with the deterministic failure retained.  Failures are
   classified by STAGE, exception TYPE and HTTP status first and by message
   text last: provider auth/transport faults are ``infrastructure``
   (retained for retry, never scored); candidate-execution exceptions are
   candidate defects (``unsupported``, scored).  Both count against the
   model-attempt bound; only ``unsupported`` is scored.  An
   infrastructure-only validation failure keeps the candidate in a
   pending-infrastructure state and the SAME bytes are re-validated within
   the execution caps without a new model request.
7. Total wall is an ABSOLUTE deadline persisted across resume: generation
   and validation are bounded by the remaining time, time spent awaiting an
   agent counts, and a result that arrives after the deadline is rejected
   as infrastructure (unscored, retained).
8. No external API call is made by default: the Pythonic provider requires
   an EXPLICIT ``api_key`` (checked in the method that calls the facade; the
   dispatcher never reads env/key files) AND a ledger ``reservation`` handle
   passed in by the caller.  Every transport request is authorized and its
   cost PREAUTHORIZED at the actual call boundary (``TransportBudget``
   charged inside ``TranslationAgent._complete``), so the agent's internal
   model fallback cannot exceed the reservation; failed calls stay charged;
   usage is reconciled from the response when the provider exposes it.
   Without an explicit ``cost_per_request_usd`` each request preauthorizes
   the entire remaining reservation (at most one request per reservation).
   The pure-skill provider never calls an API: it writes a durable
   ``TRANSLATION_REQUEST.json`` for the native coding agent, which writes
   the candidate and a ``TRANSLATION_RESPONSE.json``; the dispatcher then
   RESUMES validation on the candidate.
9. Freshness: a candidate whose bytes equal a known saved DSL
   (``known_candidates``) or an earlier attempt is refused at the common
   validation boundary, whichever provider produced it.  Byte inequality is
   NOT authorship proof; the receipt says "byte-distinct from known saved
   DSLs", never "freshly generated".
10. ``sol_source.status`` is ``ready`` only when a sealed receipt exists;
    a numerical pass without definition/dtype is ``blocked`` on the missing
    receipt inputs, because ``sol(strict=True)`` refuses a missing receipt.

Not implemented here (stated, not implied): the Pythonic provider is a thin
shim over the in-process ``solar.einsum_lib.pipeline.translate_file``
(one bounded invocation per dispatcher attempt, ``max_retries=1``); it does
not re-wire the TranslationAgent's prompt/repair loop into the dispatcher's
attempt loop.  Neither problem 119 nor the 20 Lite rows have been run
through this dispatcher.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime
import hashlib
import io
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from solar.frontend.router import RouteDecision, decide_from_graph
from solar.einsum_lib.transport_budget import (TransportBudget,
                                               TransportBudgetExhausted,
                                               permits_for, transport_budget)

__all__ = [
    "AgenticProvider", "Budget", "Candidate", "DATA_DEPENDENT_SPARSITY_OPS",
    "DispatchPolicy", "DispatchRequest", "DispatchResult", "FailureKind",
    "LedgerReservation", "PendingDispatch", "ProviderError", "PureSkillProvider",
    "PythonicProvider", "TranslationRequest", "ValidationVerdict",
    "classify_exception", "classify_message", "data_dependent_sparsity_ops",
    "default_deterministic_stage", "default_validator", "dispatch",
    "resolve_tolerance", "resume",
]

SCHEMA = "solar-dispatch-v4"
REQUEST_SCHEMA = "solar-translation-request-v1"
RESPONSE_SCHEMA = "solar-translation-response-v1"
ROUTES = ("pure_skill", "pythonic")

# Op families whose cost depends on DATA (which experts/tokens/keys are
# selected), not only on shapes.  Membership is a CONSERVATIVE ROUTING
# HEURISTIC (sparse-scope-11): a dense identity gather is also routed
# agentic and labelled an expected mechanical limitation; the label is not
# evidence that the program is sparse, only that the mechanical route cannot
# establish its density.  Exact names, no wildcards; explicit catalog
# declarations (``sparsity_family_hint``) take precedence and are needed
# for masked/variable-length programs that use none of these names.
DATA_DEPENDENT_SPARSITY_OPS = frozenset({
    # MoE / top-k routing and dispatch
    "topk", "argsort", "sort", "bincount", "one_hot", "embedding_bag",
    # data-dependent gathers / scatters / selection
    "gather", "scatter", "scatter_", "scatter_add", "scatter_add_",
    "scatter_reduce", "scatter_reduce_", "index_select", "index_add",
    "index_add_", "index_put", "index_put_", "take", "take_along_dim",
    "masked_select", "masked_scatter", "masked_scatter_", "masked_fill",
    "masked_fill_", "nonzero", "unique", "searchsorted",
})
SOL_SOURCE_MECHANICAL = "mechanical_dense_analysis"
SOL_SOURCE_DSL_DENSE = "dsl_route_dense_analysis"
SOL_SOURCE_DSL_SPARSE = "dsl_route_sparse_measured_density_analysis"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _caps_remaining(caps) -> Optional[float]:
    """Seconds left before a request's absolute deadline, from ``caps["deadline_utc"]`` (the policy
    value fixed at dispatch start) and the caller's own clock.  Round 13 (the review
    PUBLIC-PROPERTY-SCOPE-31): the request no longer carries ``wall_seconds_remaining`` -- a clock
    reading taken after the previous attempt's comparison -- so the providers derive their bound
    here; an explicit numeric value (older request files) is still honoured.  None without a deadline."""
    caps = caps or {}
    rem = caps.get("wall_seconds_remaining")
    if isinstance(rem, (int, float)) and not isinstance(rem, bool):
        return max(0.0, float(rem))
    deadline = caps.get("deadline_utc")
    if not isinstance(deadline, str):
        return None
    # parsed with ``datetime``, not ``time.strptime``: the review controls stand the module's ``time`` in
    # for a clock object offering time/monotonic/gmtime/strftime only, and a reading that raises here
    # instead of bounding cost a late generation its charge (campaign finding on an earlier revision)
    try:
        at = datetime.datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return None
    return max(0.0, float(at) - time.time())


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_dependent_sparsity_ops(graph: dict) -> dict:
    """``{op: count}`` of DATA_DEPENDENT_SPARSITY_OPS compute nodes in an
    extracted ``pytorch_graph.yaml`` document (empty when none)."""
    found: dict = {}
    for node in ((graph or {}).get("layers") or {}).values():
        op = str(node.get("type", "")).lower()
        if op in DATA_DEPENDENT_SPARSITY_OPS:
            found[op] = found.get(op, 0) + 1
    return found


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class FailureKind:
    """Why an attempt did not produce a verified translation."""

    INFRASTRUCTURE = "infrastructure"   # retained for retry, never scored
    UNSUPPORTED = "unsupported"         # scored as unsupported with reason
    INPUT = "input"                     # fail closed: missing/invalid input or policy
    NO_PROVIDER = "no_provider"         # fail closed
    NO_BUDGET = "no_budget"             # fail closed
    EXHAUSTED = "exhausted"             # attempt/wall/request/cost cap reached
    PROTOCOL = "protocol"               # malformed agent response; retained


# Exception types whose occurrence says nothing about the program's
# semantics: the host, the network or the API failed.
_INFRA_EXC = (TimeoutError, MemoryError, ConnectionError, BrokenPipeError,
              subprocess.TimeoutExpired, InterruptedError)
# Exception type NAMES (matched on the class name and its MRO names, so the
# openai/httpx types are recognised without importing them).  Provider
# authentication/permission failures are infrastructure: a provider's
# inability to authenticate is not evidence against a translation.
_INFRA_EXC_NAMES = frozenset({
    "APITimeoutError", "APIConnectionError", "APIStatusError", "APIError",
    "RateLimitError", "InternalServerError", "ServiceUnavailableError",
    "AuthenticationError", "PermissionDeniedError", "UnprocessableEntityError",
    "ReadTimeout", "ConnectTimeout", "ConnectError", "ReadError",
    "RemoteProtocolError", "TimeoutExpired", "TorchviewCapabilityError",
})
_INFRA_HTTP_STATUS = frozenset({401, 403, 408, 425, 429, 500, 502, 503, 504})
_INFRA_PATTERNS = (
    r"\brc=137\b", r"\bKilled\b", r"Cannot allocate memory",
    r"Resource temporarily unavailable", r"\btimed? ?out\b", r"\bTimeout",
    r"\bConnection (reset|refused|aborted|closed|error|timed)",
    r"\bRateLimit", r"\bHTTP (401|403|408|425|429|5\d\d)\b",
    r"\bstatus[_ ]code[ =:]+ *(401|403|408|425|429|5\d\d)\b",
    r"\bUnauthorized\b", r"\bForbidden\b",
    r"\bstream (closed|error|terminated|reset|interrupted)",
    r"\bterminated unexpectedly\b", r"\bcontention\b",
    r"\bOOM\b", r"out of memory", r"Too many open files",
    r"No space left on device",
)
_INFRA_RE = re.compile("|".join(_INFRA_PATTERNS), re.IGNORECASE)
# Candidate-execution exception types that are host faults, not candidate
# defects (type-only: candidate error TEXT is never matched, a DSL may say
# anything).
_CANDIDATE_INFRA_TYPES = frozenset({
    "TimeoutError", "TimeoutExpired", "MemoryError", "BrokenPipeError",
    "InterruptedError", "KeyboardInterrupt", "SystemExit",
    # the facade could not start its isolating child (einsum_lib.verify.IsolationUnavailableError):
    # an environment fact, never the candidate's (target pipeline 77147099, an earlier change round 2)
    "IsolationUnavailableError",
})


def classify_message(message: str) -> str:
    """``infrastructure`` if the text names a host/network/API fault,
    otherwise ``unsupported`` (a semantic / candidate defect).  Text is the
    LAST resort; prefer ``classify_exception`` where an exception exists."""
    return (FailureKind.INFRASTRUCTURE if message and _INFRA_RE.search(message)
            else FailureKind.UNSUPPORTED)


def _exc_names(exc: BaseException) -> set:
    return {type(exc).__name__} | {c.__name__ for c in type(exc).__mro__}


def _http_status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    resp = getattr(exc, "response", None)
    val = getattr(resp, "status_code", None)
    return val if isinstance(val, int) else None


def classify_exception(exc: BaseException, *, stage: str = "provider") -> str:
    """Infrastructure vs unsupported for a raised exception.

    Order (classify-3): stage, then exception type / MRO names, then
    HTTP status, then — for the provider stage only — message text.  A
    missing input file is an INPUT fault.  In the ``candidate`` stage (the
    candidate's own execution) the text is never consulted: only host-fault
    exception types are infrastructure, everything else is a candidate
    defect.
    """
    if isinstance(exc, FileNotFoundError):
        return FailureKind.INPUT
    if isinstance(exc, _INFRA_EXC):
        return FailureKind.INFRASTRUCTURE
    names = _exc_names(exc)
    if stage == "candidate":
        return (FailureKind.INFRASTRUCTURE if names & _CANDIDATE_INFRA_TYPES
                else FailureKind.UNSUPPORTED)
    if names & _INFRA_EXC_NAMES:
        return FailureKind.INFRASTRUCTURE
    status = _http_status(exc)
    if status is not None and status in _INFRA_HTTP_STATUS:
        return FailureKind.INFRASTRUCTURE
    # OSError other than file-not-found: ENOMEM/EIO/EAGAIN are host faults.
    if isinstance(exc, OSError):
        return FailureKind.INFRASTRUCTURE
    return classify_message(f"{type(exc).__name__}: {exc}")


_IR_FAIL_RE = re.compile(r"IR execution FAILED after [\d.]+s: (\w+):")


def classify_candidate_error_message(message: str) -> str:
    """Classify a ``run_verification`` IR-execution failure by the recorded
    exception TYPE only (never by the candidate's message text)."""
    m = _IR_FAIL_RE.search(message or "")
    name = m.group(1) if m else ""
    return (FailureKind.INFRASTRUCTURE if name in _CANDIDATE_INFRA_TYPES
            else FailureKind.UNSUPPORTED)


def _receipt(kind: str, stage: str, reason: str, *, attempt: int = 0,
             exception_type: Optional[str] = None, **extra) -> dict:
    """One failure receipt. ``scored`` says whether it counts as an
    unsupported-semantics verdict; ``retained_for_retry`` says whether the
    same inputs may be retried without changing anything."""
    rec = {
        "kind": kind, "stage": stage, "reason": reason, "attempt": attempt,
        "exception_type": exception_type, "utc": _utc(),
        "scored": kind == FailureKind.UNSUPPORTED,
        "retained_for_retry": kind in (FailureKind.INFRASTRUCTURE,
                                       FailureKind.PROTOCOL),
    }
    rec.update(extra)
    return rec


class ProviderError(RuntimeError):
    """A provider attempt failed; ``kind`` is a FailureKind.  ``extra`` may
    carry ``api_requests`` / ``cost_usd`` actually charged, so a failed call
    is accounted durably."""

    def __init__(self, kind: str, reason: str, **extra):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.extra = extra


class NothingToResume(ValueError):
    """``resume()`` precondition failure (the caller's contract): the dispatch
    has no pending agent request and no pending-infrastructure candidate.
    Every OTHER exception during resume is an operational fault and yields a
    durable unscored ``dispatcher_error`` (failure-17)."""


class PendingDispatch(Exception):
    """The pure-skill request was written and awaits the native agent."""

    def __init__(self, request_dir: Path, request: "TranslationRequest"):
        super().__init__(f"awaiting agent response in {request_dir}")
        self.request_dir = Path(request_dir)
        self.request = request


# ---------------------------------------------------------------------------
# Policy, budget, reservation
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Caps the dispatcher enforces.  ``None`` means uncapped for that axis
    EXCEPT ``max_attempts`` which is always finite."""

    max_attempts: int = 1                      # model attempts (generations)
    wall_seconds: Optional[float] = None       # total wall: ABSOLUTE deadline
    max_requests: Optional[int] = None         # external API transport requests
    max_cost_usd: Optional[float] = None
    # re-validations of the SAME retained candidate after an infrastructure-
    # only validation failure (no new model request)
    max_validation_retries: int = 1

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1 (bounded, non-zero)")
        if self.max_validation_retries < 0:
            raise ValueError("max_validation_retries must be >= 0")


@dataclass
class DispatchPolicy:
    route: str = "pure_skill"                  # selected v2 agentic route
    allow_agentic: bool = True
    budget: Budget = field(default_factory=Budget)
    verify_timeout: Optional[int] = None       # per candidate verification
    mem_budget_gb: Optional[float] = None
    arch: str = "B200"
    precision: str = "fp16"                    # mechanical analysis precision
    # Pythonic route: preauthorized cost per transport request.  None means
    # "preauthorize the entire remaining reservation per request" (at most
    # one request per reservation): conservative, never a silent default.
    cost_per_request_usd: Optional[float] = None
    # transport requests one Pythonic invocation may issue (model fallback
    # and rate-limit retries included); bounded by the reservation anyway
    max_transport_requests_per_attempt: int = 1
    # round 10 (result-boundary-29): the declared byte budget for a
    # result archive a candidate or adapter process writes (its outputs .npz,
    # the input snapshot); a larger file is refused, never read
    max_result_bytes: int = 1 << 30

    def __post_init__(self):
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {self.route!r}")
        if int(self.max_result_bytes) < 1:
            raise ValueError("max_result_bytes must be >= 1")
        if self.cost_per_request_usd is not None and self.cost_per_request_usd <= 0:
            raise ValueError("cost_per_request_usd must be > 0 when given")
        if self.max_transport_requests_per_attempt < 1:
            raise ValueError("max_transport_requests_per_attempt must be >= 1")


class LedgerReservation:
    """In-memory handle onto an externally owned request/cost reservation.

    The dispatcher never creates one: the acceptance owner passes it in, and
    the dispatcher consumes from it at the actual call boundary (one
    request + the preauthorized cost per transport request), reconciling
    the cost afterwards when the provider reports actual usage.
    """

    def __init__(self, requests: int, usd: float, label: str = ""):
        self.requests_total = int(requests)
        self.usd_total = float(usd)
        self.requests_used = 0
        self.usd_used = 0.0
        self.label = label
        self.consumptions: list = []

    @property
    def remaining_requests(self) -> int:
        return self.requests_total - self.requests_used

    @property
    def remaining_usd(self) -> float:
        return self.usd_total - self.usd_used

    @staticmethod
    def _money(value, what: str) -> float:
        """A dollar amount entering the ledger must be a finite non-negative
        number (cost-13): NaN, infinities and negatives never reach
        the durable balance."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{what} must be a finite non-negative dollar amount, got {value!r}")
        return float(value)

    def consume(self, requests: int = 1, usd: float = 0.0, note: str = ""):
        usd = self._money(usd, "consumed usd")
        if requests > self.remaining_requests or usd > self.remaining_usd + 1e-12:
            raise ProviderError(
                FailureKind.NO_BUDGET,
                f"reservation {self.label!r} cannot cover {requests} "
                f"request(s) / ${usd:.4f}: remaining "
                f"{self.remaining_requests} request(s) / ${self.remaining_usd:.4f}")
        self.requests_used += int(requests)
        self.usd_used += float(usd)
        self.consumptions.append({"requests": requests, "usd": usd,
                                  "note": note, "utc": _utc()})

    def reconcile(self, preauthorized_usd: float, actual_usd: float, note: str = ""):
        """Replace a preauthorized charge by the actual reported usage.
        An overrun beyond the remaining funds is recorded, never hidden."""
        actual_usd = self._money(actual_usd, "reconciled actual usd")
        preauthorized_usd = self._money(preauthorized_usd, "reconciled preauthorized usd")
        delta = actual_usd - preauthorized_usd
        self.usd_used += delta
        self.consumptions.append({"requests": 0, "usd": delta,
                                  "note": f"reconcile: {note}", "utc": _utc(),
                                  "overrun": self.usd_used > self.usd_total + 1e-12})

    def to_dict(self) -> dict:
        return {"label": self.label, "requests_total": self.requests_total,
                "requests_used": self.requests_used,
                "usd_total": self.usd_total, "usd_used": self.usd_used,
                "consumptions": list(self.consumptions)}


# ---------------------------------------------------------------------------
# Request / candidate contracts
# ---------------------------------------------------------------------------

@dataclass
class DispatchRequest:
    """What is being translated. ``source_path`` is the pytorch model file
    (``Model`` + ``get_inputs``) or, for adapter contracts, the problem's
    reference; ``definition_dir`` + ``dtype`` enable receipt emission."""

    source_path: Path
    out_dir: Path
    source_type: str = "pytorch"
    definition_dir: Optional[Path] = None
    workload_idx: int = 0
    dtype: Optional[str] = None               # analysis precision for receipt
    original_axes: Optional[dict] = None      # original workload shape
    model_route_label: str = ""               # free-form provenance label
    # caller-declared data-dependent-sparsity family (e.g. "moe",
    # "sparse_attention"): routes agentic-by-default with the reason recorded
    # and is carried through EVERY deterministic outcome
    sparsity_family_hint: Optional[str] = None
    # explicit approved tolerance policy for candidate validation:
    # {"rtol": float, "atol": float[, "required_matched_ratio": float]}.
    # None -> resolved from the selected workload.jsonl line's "tolerance";
    # still None -> UNRESOLVED (no default is ever substituted)
    tolerance: Optional[dict] = None
    # saved DSL candidates that must NOT be presented as byte-distinct work
    known_candidates: tuple = ()
    # optional deterministic numerical validation hook:
    # fn(deterministic_artifacts: dict) -> (ok: bool, reason: str)
    deterministic_validator: Optional[Callable] = None

    def __post_init__(self):
        self.source_path = Path(self.source_path)
        self.out_dir = Path(self.out_dir)
        if self.definition_dir is not None:
            self.definition_dir = Path(self.definition_dir)
        if self.tolerance is not None:
            tol = dict(self.tolerance)
            for key in ("rtol", "atol"):
                if key not in tol:
                    raise ValueError(f"tolerance needs {key!r}: {tol}")
                tol[key] = float(tol[key])
            self.tolerance = tol


def resolve_tolerance(request: DispatchRequest) -> tuple:
    """(tolerance dict | None, source) — request first, then the selected
    ``workload.jsonl`` line's ``tolerance`` object, else unresolved.  The
    workload grammar ``max_rtol``/``max_atol``/``required_matched_ratio``
    (or ``required_match_ratio``) is normalised; nothing is defaulted."""
    if request.tolerance is not None:
        return dict(request.tolerance), "request"
    if request.definition_dir is not None:
        wl = Path(request.definition_dir) / "workload.jsonl"
        if wl.is_file():
            try:
                lines = wl.read_text().splitlines()
                line = json.loads(lines[request.workload_idx])
            except (IndexError, json.JSONDecodeError, OSError):
                return None, "unresolved"
            tol = line.get("tolerance") if isinstance(line, dict) else None
            if isinstance(tol, dict):
                rtol = tol.get("max_rtol", tol.get("rtol"))
                atol = tol.get("max_atol", tol.get("atol"))
                if rtol is not None and atol is not None:
                    out = {"rtol": float(rtol), "atol": float(atol)}
                    ratio = tol.get("required_matched_ratio", tol.get("required_match_ratio"))
                    if ratio is not None:
                        out["required_matched_ratio"] = float(ratio)
                    return out, f"workload[{request.workload_idx}]"
    return None, "unresolved"


@dataclass
class TranslationRequest:
    """The durable request the pure-skill boundary hands the native agent.

    Written as ``TRANSLATION_REQUEST.json``.  The agent must write the
    candidate at ``candidate_path`` (and ``adapter_path`` for adapter
    contracts) plus ``TRANSLATION_RESPONSE.json`` echoing ``request_id``
    and ``source_sha256``, naming ``candidate_sha256`` and its provenance.
    """

    request_id: str
    route: str
    source_path: str
    source_sha256: str
    source_type: str
    definition_dir: Optional[str]
    workload_idx: int
    dtype: Optional[str]
    original_axes: Optional[dict]
    attempt: int
    max_attempts: int
    caps: dict
    candidate_path: str
    adapter_path: Optional[str]
    prior_candidate_sha256s: list
    deterministic_failure: dict
    created_utc: str
    tolerance: Optional[dict] = None
    tolerance_source: str = "unresolved"
    schema: str = REQUEST_SCHEMA
    status: str = "open"
    skill: str = "skills/verified-translation/SKILL.md"
    instructions: str = (
        "Author a NEW DSL translation of source_path following the skill. "
        "Write it to candidate_path (and the adapter to adapter_path when "
        "set). Then write TRANSLATION_RESPONSE.json next to this file with: "
        "schema, request_id, source_sha256 (echo), status "
        "('candidate'|'declined'|'error'), candidate_sha256, "
        "provenance {author, model, route, created_utc}, and for "
        "declined/error a reason and error_kind "
        "('unsupported'|'infrastructure'). Do not copy a prior saved DSL: "
        "candidates whose bytes match prior_candidate_sha256s are refused "
        "(byte inequality is not authorship proof). Respect deadline_utc.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    """A translation candidate a provider produced."""

    dsl_path: Path
    model_path: Path            # what verify() loads: model file or adapter
    dsl_sha256: str
    source_sha256: str          # the source the provider translated
    provenance: dict = field(default_factory=dict)
    api_requests: int = 0
    cost_usd: float = 0.0
    cost_basis: str = ""        # hard_cap_preauthorized | reservation_preauthorized | reported
    adapter_sha256: Optional[str] = None   # bound when model_path is an adapter

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dsl_path"] = str(self.dsl_path)
        d["model_path"] = str(self.model_path)
        return d


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class AgenticProvider:
    """Interface. ``availability`` returns None when the provider can run,
    else the fail-closed reason.  ``generate`` returns a Candidate, raises
    ProviderError(kind=...) or PendingDispatch (pure-skill, async)."""

    route: str = ""
    name: str = ""
    calls_external_api: bool = False

    def availability(self, request: DispatchRequest) -> Optional[str]:
        raise NotImplementedError

    def generate(self, treq: TranslationRequest, workdir: Path,
                 reservation=None) -> Candidate:
        raise NotImplementedError

    def collect(self, request_dir: Path) -> Candidate:
        raise ProviderError(FailureKind.PROTOCOL,
                            f"{self.name} has no asynchronous collect step")


def _candidate_paths(treq_or_req, attempt_dir: Path):
    source_type = treq_or_req.source_type
    dsl = attempt_dir / "solar_model_einsum_lib.py"
    adapter = None if source_type in ("pytorch", "triton") else attempt_dir / "adapter.py"
    return dsl, adapter


class PureSkillProvider(AgenticProvider):
    """Durable claimed task/receipt boundary to the native coding agent.

    ``agent`` (optional) is an in-process consumer ``fn(request_path) ->
    None`` that writes the candidate and response (tests use a fake agent;
    a native agent normally consumes the file out of process).  Without an
    agent, ``poll_seconds > 0`` waits for the response; otherwise
    ``generate`` raises PendingDispatch and the caller resumes later.
    """

    route = "pure_skill"
    name = "pure-skill (native coding agent via TRANSLATION_REQUEST)"
    calls_external_api = False

    def __init__(self, agent: Optional[Callable] = None,
                 poll_seconds: float = 0.0, poll_interval: float = 1.0):
        self.agent = agent
        self.poll_seconds = float(poll_seconds)
        self.poll_interval = float(poll_interval)

    def availability(self, request: DispatchRequest) -> Optional[str]:
        return None   # the boundary itself is always available; it costs nothing

    def generate(self, treq: TranslationRequest, workdir: Path,
                 reservation=None) -> Candidate:
        request_dir = Path(workdir)
        request_dir.mkdir(parents=True, exist_ok=True)
        req_path = request_dir / "TRANSLATION_REQUEST.json"
        req_path.write_text(json.dumps(treq.to_dict(), indent=1) + "\n")
        if self.agent is not None:
            try:
                self.agent(req_path)
            except ProviderError:
                raise
            except Exception as exc:  # the agent hook itself failed
                raise ProviderError(classify_exception(exc),
                                    f"agent hook raised {type(exc).__name__}: {exc}",
                                    exception_type=type(exc).__name__) from exc
            return self.collect(request_dir)
        if self.poll_seconds > 0:
            remaining = _caps_remaining(treq.caps)
            window = self.poll_seconds if remaining is None else min(self.poll_seconds, remaining)
            deadline = time.monotonic() + max(0.0, window)
            resp = request_dir / "TRANSLATION_RESPONSE.json"
            while time.monotonic() < deadline:
                if resp.is_file():
                    return self.collect(request_dir)
                time.sleep(self.poll_interval)
            raise ProviderError(
                FailureKind.INFRASTRUCTURE,
                f"no TRANSLATION_RESPONSE.json within {window:g}s "
                f"poll window in {request_dir}", request_dir=str(request_dir))
        raise PendingDispatch(request_dir, treq)

    def collect(self, request_dir: Path) -> Candidate:
        """Consume the agent's response; bind it to the request; refuse
        anything stale, unbound or byte-identical to a known saved DSL."""
        request_dir = Path(request_dir)
        req_path = request_dir / "TRANSLATION_REQUEST.json"
        resp_path = request_dir / "TRANSLATION_RESPONSE.json"
        if not req_path.is_file():
            raise ProviderError(FailureKind.PROTOCOL,
                                f"request file missing: {req_path}")
        if not resp_path.is_file():
            raise ProviderError(FailureKind.PROTOCOL,
                                f"no TRANSLATION_RESPONSE.json in {request_dir}: "
                                f"the agent has not responded")
        treq = json.loads(req_path.read_text())
        try:
            resp = json.loads(resp_path.read_text())
        except json.JSONDecodeError as exc:
            raise ProviderError(FailureKind.PROTOCOL,
                                f"response malformed: {exc}") from exc
        if not isinstance(resp, dict) or resp.get("schema") != RESPONSE_SCHEMA:
            raise ProviderError(FailureKind.PROTOCOL,
                                f"response schema must be {RESPONSE_SCHEMA!r}")
        if resp.get("request_id") != treq["request_id"]:
            raise ProviderError(
                FailureKind.PROTOCOL,
                f"response request_id {resp.get('request_id')!r} does not "
                f"echo request {treq['request_id']!r}")
        if resp.get("source_sha256") != treq["source_sha256"]:
            raise ProviderError(
                FailureKind.PROTOCOL,
                "response source_sha256 does not echo the request's source: "
                "the agent translated something else (or the source changed)")
        status = resp.get("status")
        if status == "declined":
            raise ProviderError(FailureKind.UNSUPPORTED,
                                f"agent declined: {resp.get('reason') or 'no reason given'}",
                                provenance=resp.get("provenance"))
        if status == "error":
            kind = resp.get("error_kind")
            if kind not in (FailureKind.INFRASTRUCTURE, FailureKind.UNSUPPORTED):
                raise ProviderError(
                    FailureKind.PROTOCOL,
                    f"agent error without a valid error_kind "
                    f"({kind!r}); refusing to guess: {resp.get('reason')}")
            raise ProviderError(kind, f"agent error: {resp.get('reason') or 'no reason given'}",
                                provenance=resp.get("provenance"))
        if status != "candidate":
            raise ProviderError(FailureKind.PROTOCOL,
                                f"response status {status!r} is not one of "
                                f"candidate/declined/error")
        dsl_path = Path(treq["candidate_path"])
        if not dsl_path.is_file():
            raise ProviderError(FailureKind.PROTOCOL,
                                f"response says 'candidate' but {dsl_path} does not exist")
        now = _sha256(dsl_path)
        if resp.get("candidate_sha256") != now:
            raise ProviderError(
                FailureKind.PROTOCOL,
                f"candidate_sha256 in the response ({str(resp.get('candidate_sha256'))[:12]}…) "
                f"does not match the candidate file ({now[:12]}…)")
        if now in set(treq.get("prior_candidate_sha256s") or []):
            raise ProviderError(
                FailureKind.UNSUPPORTED,
                "candidate is not byte-distinct: its bytes equal a known saved "
                "DSL listed in prior_candidate_sha256s; a reused DSL is never "
                "recorded as new work (byte inequality is not authorship proof "
                "either)", provenance=resp.get("provenance"))
        adapter = treq.get("adapter_path")
        if adapter:
            if not Path(adapter).is_file():
                raise ProviderError(FailureKind.PROTOCOL,
                                    f"adapter contract requires {adapter}, which is missing")
            model_path = Path(adapter)
        else:
            model_path = Path(treq["source_path"])
        prov = dict(resp.get("provenance") or {})
        prov.setdefault("route", "pure_skill")
        prov["request_id"] = treq["request_id"]
        prov["response_path"] = str(resp_path)
        prov["freshness"] = "byte-distinct from known saved DSLs (not authorship proof)"
        return Candidate(dsl_path=dsl_path, model_path=model_path, dsl_sha256=now,
                         source_sha256=treq["source_sha256"], provenance=prov,
                         adapter_sha256=_sha256(model_path) if adapter else None)


def _inprocess_translate(source_path, out_dir, *, source_type, api_key, **kw):
    """Default Pythonic translation: the in-process pipeline, so the active
    TransportBudget is charged at every transport request.  ONE bounded
    invocation: max_retries=1, max_cap_rounds=1, rate_limit_retries=1."""
    from solar.einsum_lib.pipeline import translate_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mem = kw.get("mem_budget")
    return translate_file(
        file_path=str(source_path), source_type=source_type, output_dir=str(out),
        max_retries=1, rate_limit_retries=1, rate_limit_delay=0, verify=True,
        api_key=api_key, save_on_fail=True, cap_input_size=False, max_cap_rounds=1,
        verify_timeout=kw.get("verify_timeout"),
        mem_budget_bytes=(int(mem * 1e9) if mem else None),
        llm_model=kw.get("llm_model"))


class PythonicProvider(AgenticProvider):
    """Thin, tested shim over the in-process translation pipeline.

    Explicit-key contract enforced in THIS method (key-8): no key, no
    call, no env/key-file lookup.  Reservation handle required.  Every
    transport request is authorized and its cost preauthorized at the call
    boundary through a ``TransportBudget`` (budget-1): the invocation
    itself prepays one request; the agent's model fallback / rate-limit
    retries charge further permits or fail before the call.  Failed calls
    stay charged and are reported back through ``ProviderError.extra``.
    ``translate_fn`` is injectable so tests never call out; an injected
    function that never reaches the agent still costs the prepaid request.
    """

    route = "pythonic"
    name = "Pythonic (in-process solar.einsum_lib.pipeline.translate_file)"
    calls_external_api = True

    def __init__(self, api_key: Optional[str] = None,
                 translate_fn: Optional[Callable] = None,
                 llm_model: Optional[str] = None,
                 verify_timeout: Optional[int] = None,
                 mem_budget_gb: Optional[float] = None,
                 cost_per_request_usd: Optional[float] = None,
                 max_transport_requests: Optional[int] = None,
                 cost_per_1k_tokens_usd: Optional[float] = None):
        self.api_key = api_key
        self._translate = translate_fn
        self.llm_model = llm_model
        self.verify_timeout = verify_timeout
        self.mem_budget_gb = mem_budget_gb
        self.cost_per_request_usd = cost_per_request_usd
        self.max_transport_requests = max_transport_requests
        # the cost bridge for the in-process pipeline: measured cost =
        # reported total tokens / 1000 * this rate; without a rate the
        # provider's measured cost is UNAVAILABLE (fail closed under a finite
        # cost cap, cost-13)
        self.cost_per_1k_tokens_usd = cost_per_1k_tokens_usd

    def availability(self, request: DispatchRequest) -> Optional[str]:
        if not self.api_key:
            return ("Pythonic route requires an explicit api_key argument; "
                    "the dispatcher never reads OPENAI_API_KEY or key files")
        if request.source_type not in ("pytorch", "raw", "jax", "triton", "cuda"):
            return f"translate_file does not accept source_type {request.source_type!r}"
        return None

    def generate(self, treq: TranslationRequest, workdir: Path,
                 reservation=None) -> Candidate:
        if not self.api_key:
            raise ProviderError(
                FailureKind.NO_PROVIDER,
                "external API call refused: PythonicProvider has no explicit "
                "api_key (it never falls back to OPENAI_API_KEY or key files)")
        if reservation is None:
            raise ProviderError(FailureKind.NO_BUDGET,
                                "external API call refused: no ledger reservation handle")
        caps = dict(getattr(treq, "caps", None) or {})
        res_usd = float(getattr(reservation, "remaining_usd", 0.0) or 0.0)
        res_req = int(getattr(reservation, "remaining_requests", 0) or 0)
        est = self.cost_per_request_usd if self.cost_per_request_usd is not None \
            else caps.get("cost_per_request_usd")
        if est is None:
            est = res_usd      # no estimate: preauthorize the whole reservation
        est = float(est)
        if est <= 0.0:
            raise ProviderError(
                FailureKind.NO_BUDGET,
                f"cannot preauthorize a positive cost for the call: estimate "
                f"${est:.4f}, reservation remaining ${res_usd:.4f}; a zero-dollar "
                f"reservation is not spend authority")
        cap_cost_left = caps.get("max_cost_usd_remaining")
        if cap_cost_left is not None and float(cap_cost_left) + 1e-12 < est:
            raise ProviderError(
                FailureKind.NO_BUDGET,
                f"cost cap leaves ${float(cap_cost_left):.4f}, below the preauthorized "
                f"${est:.4f} per request; no call is made")
        permits = permits_for(caps.get("max_requests_remaining"), res_req, res_usd,
                              cap_cost_left, est,
                              self.max_transport_requests
                              or caps.get("max_transport_requests_per_attempt") or 1)
        if permits < 1:
            raise ProviderError(
                FailureKind.NO_BUDGET,
                f"no transport request can be authorized (requests remaining "
                f"cap={caps.get('max_requests_remaining')}, reservation={res_req}; "
                f"funds ${res_usd:.4f} at ${est:.4f}/request)")
        budget = TransportBudget(permits=permits, cost_per_request_usd=est,
                                 reservation=reservation, prepaid=1)
        # the invocation boundary prepays its first transport request
        try:
            reservation.consume(1, est, note=f"pythonic attempt {treq.attempt} "
                                             f"{treq.request_id} (prepaid request 1)")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(FailureKind.NO_BUDGET,
                                f"reservation refused the prepaid request: {exc}") from exc
        translate = self._translate or _inprocess_translate
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        kw = dict(max_retries=1, max_cap_rounds=1, save_on_fail=True)
        # the request's remaining time bounds the provider's own verification
        # (wall-15): the tightest of the provider field, the policy's
        # verify_timeout and the remaining wall is forwarded
        bounds = [b for b in (self.verify_timeout, caps.get("verify_timeout"),
                              _caps_remaining(caps)) if b is not None]
        if bounds:
            kw["verify_timeout"] = max(1, int(math.ceil(min(float(b) for b in bounds))))
        mem = self.mem_budget_gb if self.mem_budget_gb is not None else caps.get("mem_budget_gb")
        if mem is not None:
            kw["mem_budget"] = mem
        if self.llm_model:
            kw["llm_model"] = self.llm_model
        explicit_bound = (self.cost_per_request_usd is not None
                          or caps.get("cost_per_request_usd") is not None)
        finite_cost_cap = caps.get("max_cost_usd") is not None

        preauth_basis = "hard_cap_preauthorized" if explicit_bound else "reservation_preauthorized"
        # once a measured cost has been RECONCILED against the reservation it
        # is the authoritative debit in BOTH ledgers (accounting-24): a
        # later fault must not restore the preauthorized figure on one side
        reconciled = {"cost": None, "basis": None}

        def charged(cost=None, basis=None):
            """The PRIMITIVE debit of this invocation: requests issued (from
            the budget's call rows, written at call issue) times the
            preauthorized bound, or an already-validated measured cost.
            Total: never raises, never credits (budget-12)."""
            try:
                used = max(1, int(budget.requests_used))
            except Exception:
                used = 1
            bound = used * est
            charged_usd = bound
            if cost is None and reconciled["cost"] is not None:
                cost, basis = reconciled["cost"], basis or reconciled["basis"]
            if cost is not None:
                try:
                    c = float(cost)
                    if math.isfinite(c) and c >= 0.0:
                        charged_usd = c
                except Exception:
                    pass
            try:
                tb = budget.to_dict()
            except Exception as exc:  # pragma: no cover - to_dict is total
                tb = {"serialization_error": f"{type(exc).__name__}: {exc}"}
            return {"api_requests": used, "cost_usd": charged_usd,
                    "cost_basis": basis or preauth_basis, "transport_budget": tb}

        # --- transport: every exception keeps the durable debit -------------
        try:
            with transport_budget(budget):
                res = translate(Path(treq.source_path), workdir, mode="api",
                                source_type=treq.source_type, api_key=self.api_key, **kw)
        except TransportBudgetExhausted as exc:
            raise ProviderError(FailureKind.EXHAUSTED,
                                f"transport budget exhausted inside the provider call: {exc}",
                                exception_type=type(exc).__name__, **charged()) from exc
        except ProviderError as pe:
            pe.extra.update(charged())
            raise
        except Exception as exc:
            raise ProviderError(classify_exception(exc, stage="provider"),
                                f"provider call raised {type(exc).__name__}: {exc}",
                                exception_type=type(exc).__name__,
                                http_status=_http_status(exc), **charged()) from exc
        # --- result processing: inside the accounting boundary too (BUDGET-12) --
        try:
            return self._admit_result(res, treq, workdir, budget, est, reservation,
                                      explicit_bound, finite_cost_cap, charged, reconciled)
        except ProviderError:
            raise
        except Exception as exc:
            # a decoder/usage/result fault after the transport call is a
            # provider-processing fault: unscored, retained, and the request
            # it consumed stays charged
            raise ProviderError(FailureKind.PROTOCOL,
                                f"provider result processing failed after transport: "
                                f"{type(exc).__name__}: {exc}",
                                exception_type=type(exc).__name__, **charged()) from exc

    def _admit_result(self, res, treq, workdir, budget, est, reservation,
                      explicit_bound, finite_cost_cap, charged, reconciled=None) -> Candidate:
        reconciled = reconciled if reconciled is not None else {"cost": None, "basis": None}
        used = max(1, budget.requests_used)
        preauth = used * est
        preauth_basis = "hard_cap_preauthorized" if explicit_bound else "reservation_preauthorized"
        conservative = charged(preauth, preauth_basis)     # never a credit

        # --- measured cost (cost-13) ----------------------------------
        # A reported cost must be a FINITE NON-NEGATIVE number, otherwise the
        # attempt is cost_malformed (infrastructure, unscored) and the
        # pre-transport upper bound stays debited.  Absent/unparseable usage
        # is admissible ONLY under an explicit pre-transport bound
        # (cost_per_request_usd); reserving the remaining balance bounds
        # nothing (cost_unbounded, fail closed).
        try:
            has_cost_attr = hasattr(res, "cost_usd")
            reported = getattr(res, "cost_usd", None) if has_cost_attr else None
        except Exception as exc:
            raise ProviderError(FailureKind.INFRASTRUCTURE,
                                f"cost_malformed: the provider result's cost accessor raised "
                                f"{type(exc).__name__}: {exc}; result not admitted, the pre-transport "
                                f"bound ${preauth:.4f} stays debited",
                                reason_code="cost_malformed", exception_type=type(exc).__name__,
                                **conservative) from exc
        tokens = budget.total_tokens
        measured, basis = None, None
        if reported is not None:
            good = (isinstance(reported, (int, float)) and not isinstance(reported, bool)
                    and math.isfinite(float(reported)) and float(reported) >= 0.0)
            if not good:
                budget.annotate("cost_malformed", reported=repr(reported)[:200],
                                request_id=treq.request_id)
                raise ProviderError(FailureKind.INFRASTRUCTURE,
                                    f"cost_malformed: the provider reported cost {reported!r}, not a "
                                    f"finite non-negative number; result not admitted, the "
                                    f"pre-transport bound ${preauth:.4f} stays debited (never a credit)",
                                    reason_code="cost_malformed", **conservative)
            measured, basis = float(reported), "reported"
        elif tokens is not None and self.cost_per_1k_tokens_usd is not None:
            rate = self.cost_per_1k_tokens_usd
            good = (isinstance(rate, (int, float)) and not isinstance(rate, bool)
                    and math.isfinite(float(rate)) and float(rate) >= 0.0)
            if not good:
                raise ProviderError(FailureKind.INFRASTRUCTURE,
                                    f"cost_malformed: cost_per_1k_tokens_usd {rate!r} is not a finite "
                                    f"non-negative rate; result not admitted",
                                    reason_code="cost_malformed", **conservative)
            measured, basis = tokens / 1000.0 * float(rate), "reported_tokens"
        bound_used = None
        if measured is None:
            if not explicit_bound:
                raise ProviderError(
                    FailureKind.INFRASTRUCTURE,
                    "cost_unbounded: no measurable cost (no finite cost_usd, no parseable token "
                    "usage with a rate) and no explicit pre-transport per-request bound "
                    "(cost_per_request_usd); reserving the remaining balance is not an upper "
                    "bound on the remote charge; result not admitted",
                    reason_code="cost_unbounded", **conservative)
            if has_cost_attr and finite_cost_cap:
                # the provider's own cost channel exists and REPORTS the cost as
                # unavailable: under a finite dollar cap the caller's estimate
                # cannot stand in for the provider's measurement -> fail closed
                raise ProviderError(
                    FailureKind.INFRASTRUCTURE,
                    "cost_unbounded: the provider reported the cost of this call as unavailable "
                    "(cost_usd=None) under a finite cost cap; the pre-transport estimate is not "
                    "the provider's measurement; result not admitted",
                    reason_code="cost_unbounded", **conservative)
            # no provider cost channel at all, but the caller established the
            # bound before the call: admit at the bound and RECORD the bound used
            cost, cost_basis, bound_used = preauth, "hard_cap_preauthorized", est
        else:
            reservation.reconcile(preauth, measured, note=f"{treq.request_id} {basis} usage")
            cost, cost_basis = measured, basis
            reconciled["cost"], reconciled["basis"] = measured, basis
            if measured > preauth + 1e-12:
                # the preauthorized bound was exceeded: recorded as an
                # ANNOTATION (never a request, accounting-20) and refused
                budget.annotate("overrun", preauthorized_usd=preauth, measured_usd=measured,
                                request_id=treq.request_id)
                raise ProviderError(
                    FailureKind.NO_BUDGET,
                    f"cost_cap_exceeded: measured cost ${measured:.4f} exceeds the preauthorized "
                    f"per-request bound ${preauth:.4f} ({preauth_basis}); the overrun is debited "
                    f"and recorded, the result is not admitted",
                    reason_code="cost_cap_exceeded", **charged(cost, cost_basis))
        cap_left = (treq.caps or {}).get("max_cost_usd_remaining")
        if cap_left is not None and cost > float(cap_left) + 1e-12:
            raise ProviderError(
                FailureKind.NO_BUDGET,
                f"cost_cap_exceeded: measured/charged cost ${cost:.4f} exceeds the remaining "
                f"cost cap ${float(cap_left):.4f} (preauthorized ${preauth:.4f}); the overrun is "
                f"recorded and the result is not admitted",
                reason_code="cost_cap_exceeded", **charged(cost, cost_basis))
        exit_code = getattr(res, "exit_code", None)
        if exit_code not in (None, 0, 1):
            log = getattr(res, "log", "") or ""
            raise ProviderError(classify_message(log) if log else FailureKind.INFRASTRUCTURE,
                                f"translation child exited {exit_code}: {log[-600:]}",
                                **charged(cost, cost_basis))
        stem = Path(treq.source_path).stem
        saved = sorted(workdir.glob(f"{stem}_einsum_lib.py")) or sorted(workdir.glob("*_einsum_lib.py"))
        if not saved:
            raise ProviderError(FailureKind.UNSUPPORTED,
                                f"translation produced no saved candidate (status "
                                f"{getattr(res, 'verification_status', '?')})",
                                **charged(cost, cost_basis))
        if len(saved) > 1:
            raise ProviderError(FailureKind.PROTOCOL,
                                f"ambiguous: {len(saved)} *_einsum_lib.py files in {workdir}",
                                **charged(cost, cost_basis))
        dsl = saved[0]
        prov = {"route": "pythonic", "provider": self.name, "model": self.llm_model,
                "child_verification_status": getattr(res, "verification_status", None),
                "child_exit_code": exit_code, "request_id": treq.request_id,
                "created_utc": _utc(), "transport_budget": budget.to_dict(),
                "cost": {"preauthorized_usd": preauth, "charged_usd": cost, "basis": cost_basis,
                         "measured": measured is not None,
                         "bound_used_usd": bound_used,
                         "bound_source": (None if bound_used is None else
                                          ("provider.cost_per_request_usd"
                                           if self.cost_per_request_usd is not None
                                           else "policy.cost_per_request_usd"))},
                "freshness": "byte-distinct from known saved DSLs (not authorship proof)"}
        if treq.source_type not in ("pytorch", "triton"):
            raise ProviderError(FailureKind.UNSUPPORTED,
                                "Pythonic shim validates pytorch/triton sources only; adapter "
                                "contracts need the pure-skill route", provenance=prov,
                                **charged(cost, cost_basis))
        return Candidate(dsl_path=dsl, model_path=Path(treq.source_path), dsl_sha256=_sha256(dsl),
                         source_sha256=treq.source_sha256, provenance=prov,
                         api_requests=used, cost_usd=cost, cost_basis=cost_basis)


# ---------------------------------------------------------------------------
# Deterministic stage (v1: capture -> route -> lower -> analyze -> perf)
# ---------------------------------------------------------------------------

@dataclass
class DeterministicOutcome:
    status: str                      # ok | unsupported | validation_failed | infrastructure_error | input_error
    reasons: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    route_decision: Optional[dict] = None
    receipt: Optional[dict] = None   # failure receipt (retained verbatim)
    numerical_validation: str = "none (mechanical route performs no numerical comparison)"
    seconds: float = 0.0
    # True when the failure is the EXPECTED mechanical limitation for a
    # data-dependent-sparsity family (agentic-by-default), not a defect
    expected_limitation: bool = False
    sparsity_family: dict = field(default_factory=dict)

    @property
    def eligible_for_fallback(self) -> bool:
        return self.status in ("unsupported", "validation_failed")

    def to_dict(self) -> dict:
        return asdict(self)


def _sparsity_family(request: DispatchRequest, graph: Optional[dict]) -> dict:
    fam = data_dependent_sparsity_ops(graph) if graph else {}
    if request.sparsity_family_hint:
        fam = {**fam, "declared": request.sparsity_family_hint}
    return fam


def _apply_declared_family(out: DeterministicOutcome, request: DispatchRequest) -> DeterministicOutcome:
    """Carry the caller's sparsity declaration through an eligible outcome
    (sparse-5): the declaration, not the reached stage, decides."""
    if request.sparsity_family_hint and out.eligible_for_fallback \
            and "declared" not in (out.sparsity_family or {}):
        out.sparsity_family = {**(out.sparsity_family or {}),
                               "declared": request.sparsity_family_hint}
        out.expected_limitation = True
        out.reasons.append(
            f"caller-declared data-dependent sparsity family "
            f"{request.sparsity_family_hint!r}: agentic-by-default (expected "
            f"limitation, not a defect); SOL from the DSL-route sparse "
            f"(measured-density) analysis")
    return out


def default_deterministic_stage(request: DispatchRequest, policy: DispatchPolicy,
                                out_dir: Path, log=print,
                                deadline_seconds: Optional[float] = None) -> DeterministicOutcome:
    """The legacy v1 route, classified.  Only ``pytorch`` sources have a
    mechanical frontend; every other source type is unsupported here by
    construction (recorded as such, so the fallback reason is exact).
    Artifact/stage faults (missing or unreadable graph, router crash) are
    infrastructure: retained, unscored, never a fallback trigger."""
    import yaml

    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reasons: list = []

    def done(status, receipt_reason=None, **artifacts):
        rec = None
        if status != "ok":
            rec = _receipt(
                {"unsupported": FailureKind.UNSUPPORTED,
                 "validation_failed": FailureKind.UNSUPPORTED,
                 "infrastructure_error": FailureKind.INFRASTRUCTURE,
                 "input_error": FailureKind.INPUT}[status],
                "deterministic", receipt_reason or "; ".join(reasons), route="mechanical")
        out = DeterministicOutcome(status=status, reasons=list(reasons),
                                   artifacts=artifacts, receipt=rec,
                                   route_decision=artifacts.get("route_decision"),
                                   seconds=time.perf_counter() - t0)
        return _apply_declared_family(out, request)

    def over_deadline():
        return deadline_seconds is not None and (time.perf_counter() - t0) > deadline_seconds

    if not request.source_path.is_file():
        reasons.append(f"source file not found: {request.source_path}")
        return done("input_error")
    if request.source_type != "pytorch":
        reasons.append(f"no mechanical frontend for source_type {request.source_type!r} "
                       f"(the mechanical route is PyTorch/torchview only)")
        return done("unsupported")

    from solar.common.types import ProcessingConfig
    from solar.graph.pytorch_processor import PyTorchProcessor

    graph_dir = out_dir / "graph"
    capture_log = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture_log):
            proc = PyTorchProcessor(ProcessingConfig(force_rerun=True, safe_mode=False))
            ok = proc.process_model_file(str(request.source_path), str(graph_dir))
    except Exception as exc:
        kind = classify_exception(exc)
        reasons.append(f"capture raised {type(exc).__name__}: {exc}")
        return done("infrastructure_error" if kind == FailureKind.INFRASTRUCTURE
                    else "unsupported", capture_log=capture_log.getvalue()[-2000:])
    text = capture_log.getvalue()
    if over_deadline():
        reasons.append("capture finished after the total wall deadline")
        return done("infrastructure_error", capture_log=text[-2000:])
    if not ok:
        err = next((ln for ln in text.splitlines() if ln.startswith("✗")), text[-400:])
        reasons.append(f"capture failed: {err.strip()}")
        kind = classify_message(text)
        return done("infrastructure_error" if kind == FailureKind.INFRASTRUCTURE
                    else "unsupported", capture_log=text[-2000:])

    graph_path = graph_dir / "pytorch_graph.yaml"
    try:
        graph = yaml.safe_load(graph_path.read_text())
        if not isinstance(graph, dict):
            raise ValueError(f"graph document is not a mapping: {type(graph).__name__}")
    except Exception as exc:
        # capture reported success but its artifact is missing/unreadable:
        # a stage fault, retained unscored (failure-10)
        reasons.append(f"capture reported success but the graph artifact is "
                       f"missing or unreadable ({graph_path}): "
                       f"{type(exc).__name__}: {exc}")
        return done("infrastructure_error", graph=str(graph_path), capture_log=text[-2000:],
                    stage_error={"stage": "graph_read", "exception_type": type(exc).__name__})
    try:
        decision: RouteDecision = decide_from_graph(graph)
        decision_dict = json.loads(decision.to_json())
    except Exception as exc:
        reasons.append(f"route decision raised {type(exc).__name__}: {exc}")
        return done("infrastructure_error", graph=str(graph_path),
                    stage_error={"stage": "route_decision", "exception_type": type(exc).__name__})
    family = _sparsity_family(request, graph)
    if family:
        # Agentic-by-default: the deterministic attempt (capture + route
        # decision) is retained; the limitation is expected, not a defect.
        # Membership is a routing heuristic, not proof of sparsity.
        reasons.append(
            f"data-dependent sparsity family {family}: mechanical lowering bills "
            f"dense and cannot establish measured density; agentic-by-default "
            f"(expected limitation, not a defect; op-name membership is a routing "
            f"heuristic, not proof of sparsity); SOL from the DSL-route sparse "
            f"(measured-density) analysis")
        if decision.route != "mechanical":
            reasons.extend(decision.reasons)
        out = done("unsupported", graph=str(graph_path), route_decision=decision_dict)
        out.expected_limitation, out.sparsity_family = True, family
        return out
    if decision.route != "mechanical":
        reasons.extend(decision.reasons)
        return done("unsupported", graph=str(graph_path), route_decision=decision_dict)

    from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    einsum_dir = out_dir / "einsum"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            einsum = PyTorchToEinsum().convert(str(graph_path), str(einsum_dir),
                                               copy_graph=False, enable_rename=True)
    except Exception as exc:
        kind = classify_exception(exc)
        reasons.append(f"einsum lowering raised {type(exc).__name__}: {exc}")
        return done("infrastructure_error" if kind == FailureKind.INFRASTRUCTURE else "unsupported",
                    graph=str(graph_path), route_decision=decision_dict)
    if einsum is None:
        reasons.append("einsum lowering returned no graph")
        return done("unsupported", graph=str(graph_path), route_decision=decision_dict)
    # 'start' layers are the model inputs (filtered before analysis), not ops
    bad = [name for name, layer in (einsum.get("layers") or {}).items()
           if layer.get("type") != "start" and layer.get("is_einsum_supportable") is False]
    if bad or einsum.get("capture_status") not in (None, "ok"):
        reasons.append(f"layers not einsum-supportable: {bad}" if bad
                       else f"capture_status {einsum.get('capture_status')!r}")
        return done("unsupported", graph=str(graph_path), route_decision=decision_dict,
                    einsum=str(einsum_dir / "einsum_graph_renamed.yaml"))
    einsum_path = einsum_dir / "einsum_graph_renamed.yaml"
    if over_deadline():
        reasons.append("lowering finished after the total wall deadline")
        return done("infrastructure_error", graph=str(graph_path), route_decision=decision_dict)

    from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
    from solar.perf.perf_model import EinsumGraphPerfModel
    analysis_dir, perf_dir = out_dir / "analysis", out_dir / "perf"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = EinsumGraphAnalyzer().analyze_graph(
                einsum_path, analysis_dir, precision=policy.precision, copy_graph=False)
            if analysis is None:
                raise ValueError("analysis returned None")
            if (analysis.get("total") or {}).get("status") == "incomplete":
                raise ValueError(f"analysis total status incomplete: "
                                 f"{(analysis.get('total') or {}).get('incomplete_layers')}")
            perf = EinsumGraphPerfModel().predict(
                analysis_dir / "analysis.yaml", perf_dir, arch_config=policy.arch,
                precision=policy.precision, copy_analysis=False)
            if perf is None:
                raise ValueError("perf returned None")
            wl = perf.get("workload") or {}
            if wl.get("status") == "incomplete" or wl.get("unresolved_ops"):
                raise ValueError(f"perf workload incomplete: status={wl.get('status')} "
                                 f"unresolved_ops={wl.get('unresolved_ops')}")
    except Exception as exc:
        kind = classify_exception(exc)
        reasons.append(f"analysis/perf: {type(exc).__name__}: {exc}")
        return done("infrastructure_error" if kind == FailureKind.INFRASTRUCTURE else "unsupported",
                    graph=str(graph_path), route_decision=decision_dict, einsum=str(einsum_path))

    artifacts = dict(graph=str(graph_path), route_decision=decision_dict,
                     einsum=str(einsum_path), analysis=str(analysis_dir / "analysis.yaml"),
                     perf=str(perf_dir / f"perf_{policy.arch}.yaml"),
                     macs=(perf.get("workload") or {}).get("total_macs"),
                     macs_typed=(perf.get("workload") or {}).get("macs_typed"))    # issue #58: the aggregate's typed components
    if request.deterministic_validator is not None:
        try:
            ok, why = request.deterministic_validator(artifacts)
        except Exception as exc:
            kind = classify_exception(exc)
            reasons.append(f"deterministic validator raised {type(exc).__name__}: {exc}")
            return done("infrastructure_error" if kind == FailureKind.INFRASTRUCTURE
                        else "validation_failed", **artifacts)
        if not ok:
            reasons.append(f"deterministic numerical validation failed: {why}")
            return done("validation_failed", **artifacts)
        return DeterministicOutcome(status="ok", artifacts=artifacts, route_decision=decision_dict,
                                    numerical_validation=f"caller validator passed: {why}",
                                    seconds=time.perf_counter() - t0)
    return done("ok", **artifacts)


# ---------------------------------------------------------------------------
# Candidate validation (the dispatcher's own, never the provider's word)
# ---------------------------------------------------------------------------

@dataclass
class ValidationVerdict:
    status: str                  # verified | mismatch | no_proof | tolerance_unresolved | candidate_error | infrastructure
    kind: str                    # FailureKind or "" when verified
    message: str = ""
    max_abs_error: Optional[float] = None
    receipt_path: Optional[str] = None
    verify_status: str = ""
    seconds: float = 0.0
    tolerance: Optional[dict] = None       # the policy actually applied
    tolerance_source: str = "unresolved"   # request | workload[i] | unresolved
    # reference-14: the executed reference bound to the requested source
    # as a CHECKED assertion (requested_sha256 == executed reference_sha256)
    reference_binding: Optional[dict] = None
    reason_code: Optional[str] = None
    # the dispatcher's OWN execution of the trusted reference from the
    # original source bytes, compared against the candidate (adapter routes)
    trusted_reference: Optional[dict] = None
    # round 13 (the review NUMERIC-FEEDBACK-30/D): when set, the attempt-visible reason and
    # validation.message are exactly this constant text; ``message`` keeps the specific
    # (private) detail for the terminal receipt
    public_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _check_reference_binding(res, candidate: Candidate) -> dict:
    """requested source sha256 == the sha256 of the reference the verifier
    actually EXECUTED (``ExecutionEvidence.reference_sha256``: the text
    ``get_source()`` returned, or the pytorch model text).  An adapter that
    embeds a changed reference verifies against the wrong program; echoed
    metadata cannot catch that, only the executed hash can."""
    ev = getattr(res, "execution", None)
    executed = getattr(ev, "reference_sha256", None) if ev is not None else None
    deps = list(getattr(ev, "local_dependencies", ()) or ()) if ev is not None else []
    unsupported = list(getattr(ev, "unsupported_dependencies", ()) or ()) if ev is not None else []
    executed_source = getattr(ev, "source_sha256", None) if ev is not None else None
    # the file the verifier loaded: the original source (no adapter) or the
    # adapter bytes the candidate was bound to
    loaded_want = candidate.adapter_sha256 or candidate.source_sha256
    out = {"requested_sha256": candidate.source_sha256, "executed_reference_sha256": executed,
           "executed_source_sha256": executed_source, "loaded_file_sha256_expected": loaded_want,
           "adapter_sha256": candidate.adapter_sha256,
           "reference_dependencies": [list(d) for d in deps],
           "unsupported_dependencies": unsupported,
           "checked": executed is not None,
           "bound": executed is not None and executed == candidate.source_sha256
                    and executed_source == loaded_want and not unsupported,
           "note": ("get_source() equality is necessary, not sufficient, for adapter "
                    "routes: the dispatcher executes the trusted reference itself "
                    "(trusted_reference)")}
    return out


# ---------------------------------------------------------------------------
# Trusted reference execution (reference-14)
# ---------------------------------------------------------------------------
# The generated adapter may supply ONLY the candidate side and input
# generation metadata.  The reference the candidate is judged against is
# executed by the dispatcher itself, from the ORIGINAL requested source bytes
# (hash-checked against the request immediately before execution), in a
# fresh interpreter (``python -E``) whose module path excludes the adapter and
# candidate directories.  The comparison candidate-vs-trusted-reference is
# computed by the dispatcher; a reference that cannot be executed fails
# closed (``reference_unexecutable``: infrastructure, unscored).
# The trusted comparator PROGRAM is the installed package module
# solar/frontend/_trusted_child.py (isolation-23, round 7).  It is never a
# string written at runtime and never a copy in the artifact tree: every phase
# launches ``python -I -S <installed module> <phase> <meta>`` in its own session
# and private 0700 directory; the PARENT hashes the module and the stdlib/numpy
# files it imports at import time and re-hashes them before AND after every
# phase (``trusted_program_tampered`` fails closed).  The child's own hash
# report is recorded, never trusted.  ``_TRUSTED_CHILD`` below is the program
# TEXT as read at import (evidence/hash reference only; editing it in memory
# changes nothing the children execute).
_TRUSTED_CHILD_PATH = Path(__file__).resolve().with_name("_trusted_child.py")
# the validation's ONE _TrustedPhases (one trusted-roots computation), published by the
# validator for the phase functions below; a context variable so that those functions
# keep their historical signatures (reviewer controls replace them with fakes)
_ACTIVE_PHASES: "contextvars.ContextVar" = contextvars.ContextVar("solar_trusted_phases", default=None)
_TRUSTED_CHILD = _TRUSTED_CHILD_PATH.read_text()
_TRUSTED_PROGRAM_SHA256 = hashlib.sha256(_TRUSTED_CHILD.encode("utf-8")).hexdigest()
_TRUSTED_PROGRAM_IMPORTS = ("json", "hashlib", "importlib", "importlib.abc", "importlib.machinery",
                            "importlib.util", "types", "inspect", "numpy", "numpy.core.numeric",
                            "numpy.lib.npyio")


def _trusted_program_files() -> dict:
    """``{label: path}`` of the comparator module and the source files of the
    modules it imports (resolved through the pinned interpreter)."""
    import importlib.util as _iu
    files = {"solar.frontend._trusted_child": str(_TRUSTED_CHILD_PATH)}
    for name in _TRUSTED_PROGRAM_IMPORTS:
        try:
            spec = _iu.find_spec(name)
        except Exception:
            spec = None
        origin = getattr(spec, "origin", None)
        if origin and origin not in ("built-in", "frozen") and os.path.isabs(origin):
            files[name] = origin
    return files


def _hash_files(files: dict) -> dict:
    out = {}
    for label, path in files.items():
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[label] = h.hexdigest()
        except Exception as exc:
            out[label] = f"<unreadable: {type(exc).__name__}>"
    return out


_TRUSTED_PROGRAM_FILES = _trusted_program_files()
_TRUSTED_PROGRAM_HASHES = _hash_files(_TRUSTED_PROGRAM_FILES)


def _trusted_program_check() -> Optional[str]:
    """Re-hash the comparator program and its imports against the constants
    recorded at dispatcher import; a description of the first mismatch, or
    None when every file is unchanged."""
    now = _hash_files(_TRUSTED_PROGRAM_FILES)
    for label, want in _TRUSTED_PROGRAM_HASHES.items():
        have = now.get(label)
        if have != want:
            return f"{label} ({_TRUSTED_PROGRAM_FILES[label]}) hashes {str(have)[:12]}…, recorded {want[:12]}… at import"
    if now["solar.frontend._trusted_child"] != _TRUSTED_PROGRAM_SHA256:
        return "comparator module differs from the text read at import"
    return None


_TRUSTED_REFERENCE_DEFAULT_TIMEOUT = 600.0   # per phase, when no verify_timeout bounds it
_RC_ISOLATION_UNAVAILABLE = 18    # the candidate child: the facade could not start its isolating child (typed)

# primitives the trusted gate relies on in the PARENT, captured at import time
# so that nothing executed later in this process can substitute them
_TRUSTED_RUN = subprocess.run
_TRUSTED_POPEN = subprocess.Popen
_TRUSTED_SHA256 = hashlib.sha256
_TRUSTED_JSON_LOADS = json.loads
_TRUSTED_OPEN = open
_TRUSTED_REPLACE = os.replace
_TRUSTED_KILLPG = os.killpg
_TRUSTED_KILL = os.kill
_TRUSTED_LISTDIR = os.listdir
_TRUSTED_OS_OPEN = os.open
_TRUSTED_FSTAT = os.fstat
_TRUSTED_LSTAT = os.lstat
_TRUSTED_READ = os.read
_TRUSTED_PREAD = os.pread
_TRUSTED_CLOSE = os.close
_TRUSTED_GETUID = os.getuid

# round 10 (manager hardening rule, result-boundary-29): every path a
# candidate/adapter process may have written is ingested through ONE helper.
_REPORT_BYTES_CAP = 1 << 20          # a phase report.json
_STDERR_BYTES_CAP = 64 << 10         # the tail kept of a phase's stderr (parent-held descriptor)
_RECEIPT_BYTES_CAP = 8 << 20         # a staged workload receipt written by the verify child
_INGEST_MIN_SECONDS = 0.5            # the ingest always gets at least this long
# monotonic deadline of the running dispatch (set by _Dispatch around the
# validator): the result ingest never outlives the remaining wall budget
_DISPATCH_DEADLINE: "contextvars.ContextVar" = contextvars.ContextVar("solar_dispatch_deadline", default=None)


def _ingest_seconds(phase_timeout: float) -> float:
    """The bound for one result ingest: the remaining dispatch budget, never
    more than the phase timeout, never less than ``_INGEST_MIN_SECONDS``.  The
    floor is a disclosed GRACE that applies while budget remains; an already
    expired budget is refused by ``_ingest_result_file`` before any open or
    read, so the floor never buys a read past the deadline."""
    deadline = _DISPATCH_DEADLINE.get()
    bound = float(phase_timeout)
    if deadline is not None:
        bound = min(bound, float(deadline) - time.monotonic())
    return max(_INGEST_MIN_SECONDS, bound)


def _file_kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "device"
    return "unknown file type"


def _ingest_result_file(path, cap: int, *, label: str, seconds: float) -> dict:
    """Read a file a candidate/adapter process may have written, or refuse it.

    Rule (round 10): ``lstat`` the directory entry; open with
    ``O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC``; ``fstat`` the DESCRIPTOR and
    require a regular file, exactly one link, our own uid, size within *cap*
    and the same (dev, ino) as the directory entry; read at most ``cap + 1``
    bytes.  Any other file type (fifo, socket, device, symlink, directory), an
    oversize file or an identity mismatch is a refusal that NAMES the type.
    After the read the descriptor is ``fstat``-ed again and the metadata that
    must be STABLE is: st_dev, st_ino, st_nlink (1), st_mode (regular file),
    st_uid, and st_size, which must equal the bytes read and stay within *cap*
    (an append past EOF between the read and the re-check refuses; round 11).
    The whole step runs in a helper thread bounded by *seconds* (the remaining
    dispatch budget), so a stuck read can never exceed the wall budget.  The
    bound is never less than ``_INGEST_MIN_SECONDS`` WHILE BUDGET REMAINS -- a
    disclosed grace for a near-expired budget -- but once the dispatch deadline
    (``_DISPATCH_DEADLINE``) has passed nothing is opened or read: the result is
    refused outright.
    -> {"ok", "bytes", "problem", "kind", "size", "nlink", ...}; the bytes are
    None unless ``ok``.  This full record is PRIVATE evidence: attempt-visible
    records carry ``_public_ingest_view`` of it (round 12, the review
    NUMERIC-FEEDBACK-30/C) -- the serialized length of the unredacted comparison
    report identified the hidden statistics through a finite domain."""
    cap = int(cap)
    out = {"label": label, "ok": False, "bytes": None, "problem": None, "kind": None, "size": None,
           "nlink": None, "cap_bytes": cap, "seconds_bound": round(float(seconds), 3), "timed_out": False}
    deadline = _DISPATCH_DEADLINE.get()
    if deadline is not None and time.monotonic() >= float(deadline):
        return {**out, "seconds_bound": 0.0, "timed_out": True,
                "problem": f"{label}: the dispatch budget was exhausted before the result could be ingested; "
                           f"nothing was read and nothing is admitted"}

    def work():
        try:
            try:
                ls = _TRUSTED_LSTAT(str(path))
            except FileNotFoundError:
                out["kind"] = "absent"
                out["problem"] = f"{label}: no file at the result path"
                return
            out["kind"], out["size"], out["nlink"] = _file_kind(ls.st_mode), ls.st_size, ls.st_nlink
            if not stat.S_ISREG(ls.st_mode):
                out["problem"] = f"{label}: a {out['kind']} at the result path, not a regular file"
                return
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
            try:
                fd = _TRUSTED_OS_OPEN(str(path), flags)
            except OSError as exc:
                out["problem"] = f"{label}: not openable without following links ({type(exc).__name__})"
                return
            try:
                st = _TRUSTED_FSTAT(fd)
                out["kind"], out["size"], out["nlink"] = _file_kind(st.st_mode), st.st_size, st.st_nlink
                if not stat.S_ISREG(st.st_mode):
                    problem = f"a {out['kind']} behind the opened descriptor"
                elif st.st_nlink != 1:
                    problem = f"{st.st_nlink} directory entries link to the result inode"
                elif st.st_uid != _TRUSTED_GETUID():
                    problem = "the result inode is owned by another user"
                elif st.st_size > cap:
                    problem = f"{st.st_size} bytes exceed the {cap}-byte cap"
                elif (st.st_dev, st.st_ino) != (ls.st_dev, ls.st_ino):
                    problem = "the directory entry changed between lstat and open"
                else:
                    problem = None
                if problem:
                    out["problem"] = f"{label}: {problem}"
                    return
                chunks, total = [], 0
                while total <= cap:
                    chunk = _TRUSTED_READ(fd, min(1 << 20, cap + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                if total > cap:
                    out["problem"] = f"{label}: more than {cap} bytes were readable"
                    return
                again = _TRUSTED_FSTAT(fd)
                if (again.st_dev, again.st_ino, again.st_nlink, again.st_mode, again.st_uid) != \
                        (st.st_dev, st.st_ino, st.st_nlink, st.st_mode, st.st_uid) or not stat.S_ISREG(again.st_mode):
                    out["problem"] = f"{label}: the result inode changed while it was read"
                    return
                if again.st_size > cap or again.st_size != total:
                    # round 11 (the review RESULT-BOUNDARY-29 final-size gap): the size after
                    # the read must be the size read and within the cap
                    out["size"] = again.st_size
                    out["problem"] = (f"{label}: the result changed size while it was read ({total} bytes read, "
                                      f"{again.st_size} bytes afterwards, cap {cap})")
                    return
                out["bytes"] = b"".join(chunks)
                out["size"] = total
                out["ok"] = True
            finally:
                _TRUSTED_CLOSE(fd)
        except BaseException as exc:  # noqa: BLE001 - any failure is a refusal, never an admission
            out["ok"], out["bytes"] = False, None
            out["problem"] = f"{label}: ingest failed ({type(exc).__name__})"

    worker = threading.Thread(target=work, name="solar-result-ingest", daemon=True)
    worker.start()
    worker.join(max(_INGEST_MIN_SECONDS, float(seconds)))
    if worker.is_alive():
        return {**out, "ok": False, "bytes": None, "timed_out": True,
                "problem": f"{label}: the result ingest did not finish within the remaining dispatch "
                           f"budget ({float(seconds):.1f}s); nothing is admitted"}
    return out


def _read_held_descriptor(fileobj, cap: int) -> str:
    """The tail of a phase's stderr, read from the descriptor the PARENT opened
    before the launch (no candidate-controlled path is opened).  The descriptor
    is checked to still be a regular file."""
    try:
        fd = fileobj.fileno()
        st = _TRUSTED_FSTAT(fd)
        if not stat.S_ISREG(st.st_mode):
            return ""
        start = max(0, st.st_size - cap)
        data = _TRUSTED_PREAD(fd, min(cap, st.st_size), start)
        return data.decode("utf-8", "replace")[-1200:]
    except (OSError, ValueError):
        return ""


# -- allowlisted report schema (reference-27 fix b) ------------------
# Type codes: "bool", "int", "num", "str" (<=4000 chars), "str_s" (<=256),
# "path" (<=4096), "hex" (64 lowercase hex digits), "list_str", "list_list_str",
# "dict_str", "dict_list_int", "dict_bool", "dict_num", "stat" (a finite number or one of
# the compare child's canonical non-finite tokens), "dict_stat", or a nested dict schema.
# Round 13 (the review NUMERIC-FEEDBACK-30/D): the compare child writes a non-finite error
# statistic as "inf" / "-inf" / "nan"; the parser accepts exactly these tokens, so the
# verdict parses whole and the parent's digest agrees with the child's.  A JSON
# Infinity/NaN literal is still dropped (and counted) like any mistyped value.
_NONFINITE_TOKENS = frozenset({"inf", "-inf", "nan"})
_SCHEMA_COMMON = {
    "ok": "bool", "phase": "str_s", "program_path": "path", "program_sha256_at_start": "hex",
    "program_sha256_at_end": "hex", "argv": "list_str", "cwd": "path", "pid": "int", "sid": "int", "pgid": "int",
    "flags": {"isolated": "bool", "no_site": "bool", "ignore_environment": "bool"},
    "error": "str", "violations": "list_str", "fault": "str_s",
    "exception": {"type": "str_s", "mro": "list_str", "message": "str"},
    "sys_path": "list_str", "guard_loaded": "list_str",
}
_SCHEMA_VERDICT = {
    "success": "bool", "error": "str", "max_abs_error": "stat", "vacuous_outputs": "list_str",
    "compared_outputs": "list_str", "outputs_pass": "dict_bool", "per_output_max_abs_error": "dict_stat",
    "tolerance": {"rtol": "num", "atol": "num", "trusted_dtype_floor": "num", "rtol_effective": "num",
                  "atol_effective": "num"},
    "artifact_sha256": {"inputs": "hex", "trusted_outputs": "hex", "candidate_outputs": "hex"},
}
_SCHEMA_PHASE = {
    "verify": {"verify": {
        "status": "str_s", "message": "str", "max_abs_error": "stat", "vacuous_outputs": "list_str",
        "trusted_roots_consumed_by_runner": "bool",
        "execution": {"source_sha256": "hex", "reference_sha256": "hex", "local_dependencies": "list_list_str",
                      "unsupported_dependencies": "list_str", "min_tol": "num"},
        "receipt_path": "path", "receipt_sha256": "hex", "receipt_refused": "str",
        "receipt_error": {"type": "str_s", "message": "str"}}},
    "inputs": {"inputs_sha256": "hex", "names": "list_str", "shapes": "dict_list_int", "dtypes": "dict_str",
               "entry": "str_s", "args": "list_str"},
    "reference": {"entry": "str_s", "framework": "str_s", "outputs_sha256": "hex", "output_shapes": "dict_list_int",
                  "output_dtypes": "dict_str", "source_sha256": "hex",
                  "extension_submodule_of": "dict_str", "package_submodule_of": "dict_str",
                  "attribution_basis": "dict_str", "census_diagnostic": "dict_dict"},
    "candidate": {"outputs_sha256": "hex", "output_shapes": "dict_list_int", "output_dtypes": "dict_str",
                  "identity_audit": "list_str",
                  "extension_submodule_of": "dict_str", "package_submodule_of": "dict_str",
                  "attribution_basis": "dict_str", "census_diagnostic": "dict_dict"},
    "compare": {"verdict": _SCHEMA_VERDICT, "verdict_sha256": "hex", "program_sha256": "hex"},
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _coerce(value, code, stats: dict):
    """One allowlisted value, re-serialised into plain Python; None when the
    value does not fit the declared type (counted in stats["dropped"])."""
    def drop():
        stats["dropped"] += 1
        return None
    if value is None:
        return None
    if isinstance(code, dict):
        if not isinstance(value, dict):
            return drop()
        return _allowlist(value, code, stats)
    if code == "bool":
        return value if isinstance(value, bool) else drop()
    if code == "int":
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else drop()
    if code == "num":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return drop()
        return float(value) if math.isfinite(float(value)) else drop()
    if code == "stat":
        if isinstance(value, str):
            return value if value in _NONFINITE_TOKENS else drop()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return drop()
        return float(value) if math.isfinite(float(value)) else drop()
    if code in ("str", "str_s", "path"):
        if not isinstance(value, str):
            return drop()
        limit = {"str": 4000, "str_s": 256, "path": 4096}[code]
        if len(value) > limit:
            stats["truncated"] += 1
        return value[:limit]
    if code == "hex":
        return value if isinstance(value, str) and _HEX64.match(value) else drop()
    if code == "list_str":
        if not isinstance(value, list):
            return drop()
        return [str(x)[:512] for x in value[:256] if isinstance(x, (str, int, float)) and not isinstance(x, bool)]
    if code == "list_list_str":
        if not isinstance(value, list):
            return drop()
        return [[str(x)[:512] for x in item[:16] if isinstance(x, (str, int, float))]
                for item in value[:256] if isinstance(item, (list, tuple))]
    if code in ("dict_str", "dict_list_int", "dict_bool", "dict_num", "dict_stat"):
        if not isinstance(value, dict):
            return drop()
        out = {}
        for k, v in list(value.items())[:256]:
            if not isinstance(k, str):
                stats["dropped"] += 1
                continue
            k = k[:256]
            if code == "dict_str" and isinstance(v, str):
                out[k] = v[:512]
            elif code == "dict_bool" and isinstance(v, bool):
                out[k] = v
            elif code == "dict_num" and isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and math.isfinite(float(v)):
                out[k] = float(v)
            elif code == "dict_stat" and isinstance(v, str) and v in _NONFINITE_TOKENS:
                out[k] = v
            elif code == "dict_stat" and isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and math.isfinite(float(v)):
                out[k] = float(v)
            elif code == "dict_list_int" and isinstance(v, list) and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in v):
                out[k] = [int(x) for x in v[:64]]
            else:
                stats["dropped"] += 1
        return out
    if code == "dict_dict":
        # {name: {field: scalar}} -- the census diagnostic: str/bool/int/float/None fields only
        if not isinstance(value, dict):
            return drop()
        out = {}
        for k, v in list(value.items())[:256]:
            if not isinstance(k, str) or not isinstance(v, dict):
                stats["dropped"] += 1
                continue
            inner = {}
            for ik, iv in list(v.items())[:64]:
                if not isinstance(ik, str):
                    stats["dropped"] += 1
                elif iv is None or isinstance(iv, (bool, int)):
                    inner[ik[:128]] = iv
                elif isinstance(iv, float):
                    inner[ik[:128]] = iv if math.isfinite(iv) else None
                elif isinstance(iv, str):
                    inner[ik[:128]] = iv[:512]
                else:
                    stats["dropped"] += 1
            out[k[:256]] = inner
        return out
    return drop()


def _allowlist(raw: dict, schema: dict, stats: dict) -> dict:
    out = {}
    for key, value in raw.items():
        if key not in schema:
            stats["dropped"] += 1       # unknown keys are dropped and COUNTED, never copied
            continue
        coerced = _coerce(value, schema[key], stats)
        if coerced is not None or value is None:
            out[key] = coerced
    return out


def _parse_report(data: Optional[bytes], phase: str) -> tuple:
    """A child's report.json bytes -> (allowlisted dict, stats).  Nothing from
    the raw object is ever merged: only known keys with values of the declared
    types are re-serialised.  Unparseable bytes give an empty report."""
    stats = {"dropped": 0, "truncated": 0, "parsed": False}
    if not data:
        return {}, stats
    try:
        raw = _TRUSTED_JSON_LOADS(data.decode("utf-8"))
    except Exception:
        return {}, stats
    if not isinstance(raw, dict):
        return {}, stats
    stats["parsed"] = True
    schema = {**_SCHEMA_COMMON, **_SCHEMA_PHASE.get(phase, {})}
    return _allowlist(raw, schema, stats), stats


def _redact_private_paths(obj, private_root, private_id: Optional[str]):
    """Public records never carry a private-root PATH (reference-27 fix c):
    every string under *obj* that names the root (or anything below it) is
    rewritten to an opaque id; the path itself stays in the root's own state file."""
    if private_root is None:
        return obj
    roots = {str(private_root)}
    try:
        roots.add(os.path.realpath(str(private_root)))
    except OSError:
        pass
    token = f"<private-evidence:{private_id or 'root'}>"

    def walk(x):
        if isinstance(x, str):
            for r in roots:
                if r in x:
                    x = x.replace(r, token)
            return x
        if isinstance(x, dict):
            return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(walk(v) for v in x)
        return x
    return walk(obj)
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_PR_SET_CHILD_SUBREAPER = 36
_GENERATED_PHASES = ("verify", "inputs", "candidate")   # phases that execute adapter/candidate code


def _enable_subreaper() -> bool:
    """Linux: orphaned descendants of our children re-parent to THIS process
    instead of init, so a double-forked or ``setsid``'d survivor of a phase is
    still visible as our child on ``/proc`` and can be killed and reaped."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        return int(libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)) == 0
    except Exception:
        return False


_SUBREAPER = _enable_subreaper()


def _trusted_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    return env


def _proc_table() -> dict:
    """``{pid: {ppid, pgid, sid, state, comm}}`` from ``/proc`` (Linux)."""
    out = {}
    try:
        entries = _TRUSTED_LISTDIR("/proc")
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with _TRUSTED_OPEN(f"/proc/{entry}/stat", "rb") as fh:
                text = fh.read().decode("utf-8", "replace")
            comm = text[text.index("(") + 1:text.rindex(")")]
            rest = text[text.rindex(")") + 2:].split()
            out[int(entry)] = {"ppid": int(rest[1]), "pgid": int(rest[2]), "sid": int(rest[3]),
                               "state": rest[0], "comm": comm}
        except (OSError, ValueError, IndexError):
            continue
    return out


def _session_boundary(leader: int, before: dict) -> dict:
    """The hard phase boundary: kill the phase's whole session / process group
    and VERIFY on ``/proc`` that nothing survived.  A process that is in the
    leader's session or process group, or that re-parented to us (subreaper)
    and did not exist before the launch, is a descendant that outlived the
    phase.  Survivors are killed and reaped; the record says what was found
    after the leader exited and whether anything stayed alive after SIGKILL."""
    me = os.getpid()

    def survivors():
        return {pid: info for pid, info in _proc_table().items()
                if pid not in (me, leader) and info["state"] != "Z"
                and (info["sid"] == leader or info["pgid"] == leader
                     or (info["ppid"] == me and pid not in before))}

    found = survivors()
    record = {"leader_pid": leader, "subreaper": _SUBREAPER, "signals": [],
              "found_after_exit": [dict(pid=p, **i) for p, i in sorted(found.items())]}
    for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 3.0)):
        targets = survivors()
        if not targets:
            break
        try:
            _TRUSTED_KILLPG(leader, sig)
            record["signals"].append(f"killpg({leader}, {sig.name})")
        except OSError:
            pass
        for pid in targets:
            try:
                _TRUSTED_KILL(pid, sig)
                record["signals"].append(f"kill({pid}, {sig.name})")
            except OSError:
                pass
        deadline = time.monotonic() + grace
        while True:
            left = survivors()
            for pid, info in left.items():
                if info["ppid"] == me:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if not left or time.monotonic() > deadline:
                break
            time.sleep(0.02)
    alive = survivors()
    for pid, info in _proc_table().items():          # reap zombies that re-parented to us
        if info["ppid"] == me and info["state"] == "Z" and pid not in before:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
    record["alive_after_kill"] = [dict(pid=p, **i) for p, i in sorted(alive.items())]
    record["clean"] = not found and not alive
    return record


class _TrustedPhases:
    """Launches the trusted comparator program FROM ITS INSTALLED PACKAGE PATH,
    one fresh ``python -I -S`` interpreter per phase, each in its own session
    and its own private 0700 directory; the parent hashes the program (and its
    imports) before AND after every phase, kills the session afterwards and
    verifies no descendant survived (isolation-23, round 7).  The trusted
    roots are computed ONCE here and passed to every child.  Round 9: while a
    phase that executes generated code (verify, inputs, candidate) is alive the
    dispatch's parent-private evidence root (``private_root``) is locked to
    mode 0000 and re-verified (inode, mode, ctime) before it is unlocked; the
    reference and compare phases run inside it.  Nothing but ``meta.json``
    reaches a child: completion is established by the parent."""

    def __init__(self, work: Path, timeout: float, log=print):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.log = log
        self.roots = _trusted_roots()
        self.roots_sha256 = _TRUSTED_SHA256(json.dumps(list(self.roots), sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest()
        self.env = _trusted_env()
        self.records: dict = {}
        self.ingested: dict = {}     # phase -> {label: ingest record incl. bytes} (round 10)
        self.private_root: Optional[Path] = None   # the dispatch's parent-private evidence root
        # evidence copy of the program text (never executed; the control that
        # reviews a dispatch hashes it next to the artifacts)
        evidence = self.work / "trusted_child.py"
        if not evidence.exists():
            evidence.write_text(_TRUSTED_CHILD)
            try:
                os.chmod(evidence, 0o444)
            except OSError:
                pass

    def new_dir(self, phase: str, parent: Optional[Path] = None) -> Path:
        """A fresh private directory for one phase, mode 0700, created NOW (so a
        reference directory is created only after the candidate boundary); the
        reference and compare phases live under the private evidence root."""
        base = Path(parent) if parent is not None else self.work
        n = 0
        while True:
            pdir = base / (f"phase_{phase}" if n == 0 else f"phase_{phase}_{n}")
            if not pdir.exists():
                break
            n += 1
        pdir.mkdir(parents=True, mode=0o700)
        try:
            os.chmod(pdir, 0o700)
        except OSError:
            pass
        return pdir

    def run(self, phase: str, meta: dict, pdir: Optional[Path] = None, results: Optional[dict] = None):
        """-> (rc, report).  rc -3 = program tampered (fail closed), -4 = a
        process of the phase's session outlived it, -5 = the private evidence
        root was touched while generated code ran, -1 = timeout, -2 = launch
        failure.  No data reaches the child other than ``meta.json`` (the parent
        establishes completion itself: there is no nonce).

        Round 10: every file the child may have written is INGESTED through
        ``_ingest_result_file`` (regular file, one link, our uid, size cap,
        directory-entry identity, bounded read under the remaining dispatch
        budget) BEFORE the private evidence root is unlocked; the report is
        parsed against the phase's allowlist schema and never merged raw.
        *results* maps a label to ``(path, cap_bytes)`` for the child's result
        archives; their bytes (or the refusal) are kept in ``self.ingested[phase]``."""
        problem = _trusted_program_check()
        if problem:
            rec = {"ok": False, "returncode": -3, "error": f"trusted_program_tampered: {problem}"}
            self.records[phase] = rec
            return -3, rec
        if pdir is None:
            pdir = self.new_dir(phase)
        meta = {**meta, "report_path": str(pdir / "report.json"), "trusted_roots": self.roots,
                "trusted_roots_sha256": self.roots_sha256, "sys_path": self.roots,
                "checkout_root": str(_CHECKOUT_ROOT), "phase_dir": str(pdir),
                "env_expected": sorted([k, v] for k, v in self.env.items())}
        meta_path = pdir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
        # the INSTALLED module is what executes: no copy in the artifact tree
        argv = [sys.executable, "-I", "-S", str(_TRUSTED_CHILD_PATH), phase, str(meta_path)]
        # while a process that executes generated code is alive the private evidence
        # root is LOCKED (mode 0000): no path below it resolves, for any user
        locked = None
        if phase in _GENERATED_PHASES and self.private_root is not None:
            try:
                locked = _lock_private_root(self.private_root)
            except OSError as exc:
                rec = {"ok": False, "returncode": -5,
                       "error": f"evidence_root_tampered: the private evidence root could not be locked ({exc})"}
                self.records[phase] = rec
                return -5, rec
        before = _proc_table()
        leader = None
        stderr = ""
        ingest = {}
        try:
            with _TRUSTED_OPEN(str(pdir / "stdout.txt"), "wb") as so, _TRUSTED_OPEN(str(pdir / "stderr.txt"), "wb") as se:
                try:
                    proc = _TRUSTED_POPEN(argv, cwd=str(pdir), env=self.env, stdin=subprocess.DEVNULL,
                                          stdout=so, stderr=se, start_new_session=True)
                    leader = proc.pid
                    try:
                        rc = proc.wait(timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        rc = -1
                        try:
                            _TRUSTED_KILLPG(proc.pid, signal.SIGKILL)
                        except OSError:
                            pass
                        proc.wait()
                except Exception as exc:
                    rc, stderr = -2, f"could not start the isolated interpreter: {type(exc).__name__}: {exc}"
                boundary = _session_boundary(leader, before) if leader is not None else None
                # the session is dead and verified gone.  Everything the child may
                # have written is ingested NOW, while the private root is still
                # locked, from the parent's own descriptor (stderr) or through the
                # hardened open (report, result archives) -- never a blocking open
                # of a candidate-controlled path, never a followed link
                if rc == -1:
                    stderr = f"phase {phase} did not finish within {self.timeout:g}s"
                elif leader is not None:
                    stderr = _read_held_descriptor(se, _STDERR_BYTES_CAP)
                seconds = _ingest_seconds(self.timeout)
                ingest["report"] = _ingest_result_file(pdir / "report.json", _REPORT_BYTES_CAP,
                                                       label="report.json", seconds=seconds)
                for label, (rpath, cap) in (results or {}).items():
                    ingest[label] = _ingest_result_file(rpath, int(cap), label=label,
                                                        seconds=_ingest_seconds(self.timeout))
        except Exception as exc:
            rc, stderr = -2, f"could not start the isolated interpreter: {type(exc).__name__}: {exc}"
            boundary = None
        self.ingested[phase] = ingest
        # only now is the private root unlocked, and only after its identity,
        # mode and ctime are re-checked
        lock_problem = _unlock_private_root(self.private_root, locked) if locked is not None else None
        # the PARENT re-hashes the installed program and its imports after the child
        after = _trusted_program_check()
        report_ingest = ingest.get("report") or {}
        report, stats = _parse_report(report_ingest.get("bytes") if report_ingest.get("ok") else None, phase)
        # the full ingest metadata (size, links, time bound) travels in the phase record,
        # which is PRIVATE until the dispatch ends: ``_attempt_visible`` replaces it by the
        # fixed-shape ``_public_ingest_view`` in every attempt-visible record (round 12,
        # the review NUMERIC-FEEDBACK-30/C); the terminal receipt restores it
        result_ingest = {label: {k: v for k, v in rec.items() if k != "bytes"} for label, rec in ingest.items()}
        report = {**report, "returncode": rc, "child_argv": report.get("argv"), "argv": argv, "phase_dir": str(pdir),
                  "program_path": str(_TRUSTED_CHILD_PATH), "program_sha256_parent": _TRUSTED_PROGRAM_SHA256,
                  "program_check_after_phase": after, "self_reported_hash_trusted": False,
                  "session_boundary": boundary, "private_root_locked": locked is not None,
                  "stdin": "devnull (no nonce: completion is established by the parent)",
                  "result_ingest": result_ingest,
                  "report_schema": {"allowlisted": True, "unknown_or_mistyped_dropped": stats["dropped"],
                                    "truncated": stats["truncated"], "parsed": stats["parsed"],
                                    "ingest_before_unlock": True}}
        if not report_ingest.get("ok"):
            # a report that is not a plain regular file of ours (fifo, symlink,
            # oversize, ...) is NO report: the phase did not complete
            report["ok"] = False
            report["result_problem"] = report_ingest.get("problem")
        if stderr and not report.get("ok"):
            report["stderr_tail"] = stderr
        self_start = report.get("program_sha256_at_start")
        self_end = report.get("program_sha256_at_end")
        if after:
            report = {**report, "ok": False, "returncode": -3,
                      "error": f"trusted_program_tampered: {after} (re-hashed by the parent after the {phase} phase)"}
            rc = -3
        elif rc not in (-1, -2) and (self_start not in (None, _TRUSTED_PROGRAM_SHA256)
                                     or self_end not in (None, _TRUSTED_PROGRAM_SHA256)):
            # a self-report is never evidence FOR the verdict, but a self-report
            # that disagrees with the parent's hash is enough to fail closed
            report = {**report, "ok": False, "returncode": -3,
                      "error": "trusted_program_tampered: the child's self-reported program hash differs from the parent's"}
            rc = -3
        elif boundary is not None and not boundary["clean"] and rc != -1:
            what = "candidate_descendants_alive" if phase in _GENERATED_PHASES else "trusted_phase_descendants_alive"
            report = {**report, "ok": False, "returncode": -4,
                      "error": f"{what}: {len(boundary['found_after_exit'])} process(es) of the {phase} session "
                               f"outlived the phase ({[p['comm'] for p in boundary['found_after_exit']]}); "
                               f"{len(boundary['alive_after_kill'])} still alive after SIGKILL; nothing produced "
                               f"by this phase is admitted"}
            rc = -4
        elif lock_problem:
            report = {**report, "ok": False, "returncode": -5,
                      "error": f"evidence_root_tampered: {lock_problem}; nothing produced by this phase is admitted"}
            rc = -5
        # the record kept in the (candidate-readable) artifact tree carries no
        # digest of reference bytes: those stay in the private evidence root
        rec = {k: v for k, v in report.items() if k not in ("sys_path", "guard_loaded") or phase == "reference"}
        for key in _ATTRIBUTION_KEYS:                # dependency attribution + census diagnostic: private receipt only
            rec.pop(key, None)
        if phase == "reference":
            rec.pop("outputs_sha256", None)
        if phase == "compare":
            rec.pop("verdict_sha256", None)
            if isinstance(rec.get("verdict"), dict):
                rec["verdict"] = {k: v for k, v in rec["verdict"].items() if k != "artifact_sha256"}
        self.records[phase] = rec
        return rc, report

    def digest_for(self, verdict: dict, report: dict) -> str:
        """The AUTHORITATIVE verdict digest, computed by the parent from its
        own constants: the import-time program hash (re-verified after the
        phase), the argv it launched and the environment it passed.  The
        compare child's digest is only cross-checked against it."""
        launch = list(report.get("argv") or [])
        child_argv = launch[3:] if len(launch) >= 6 and launch[1:3] == ["-I", "-S"] else launch
        bound = {"verdict": verdict, "program_sha256": _TRUSTED_PROGRAM_SHA256, "argv": child_argv,
                 "env": sorted([k, v] for k, v in self.env.items())}
        return _TRUSTED_SHA256(json.dumps(bound, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _array_fingerprints(path) -> Optional[dict]:
    """``{name: {sha256, shape, dtype}}`` of the arrays in an ``.npz``, read by
    the PARENT with pickling disabled: the per-array frozen state."""
    try:
        import numpy as np
        out = {}
        with np.load(str(path), allow_pickle=False) as z:
            for name in z.files:
                a = np.ascontiguousarray(z[name])
                out[str(name)] = {"sha256": _TRUSTED_SHA256(a.tobytes()).hexdigest(),
                                  "shape": list(a.shape), "dtype": str(a.dtype)}
        return out
    except Exception:
        return None

def _trusted_roots() -> list:
    """Directories whose code the trusted children may import, derived from the
    LIVE interpreter (CI-FAIL-25): (1) its configured locations — ``sysconfig``
    stdlib/platstdlib/purelib/platlib, ``site.getsitepackages()``, the user
    site when ``site.ENABLE_USER_SITE`` holds (``pip install --user`` runner
    images keep torch/numpy there) — each only when the directory EXISTS and is
    actually on the live ``sys.path`` (a configured-but-inactive distro
    dist-packages dir is not admitted), and the prefixes; (2) the directories the
    environment itself declares inside those site dirs (``.pth`` targets,
    ``.egg-link``, PEP 660 editable ``direct_url.json``); (3) by ORIGIN: the
    resolved site directory each of solar's own already-imported third-party
    dependencies was actually loaded from, and every ``sys.path`` entry that
    is an installed site directory (named ``site-packages``/``dist-packages``
    and holding ``*.dist-info``); (4) the pinned checkout.  Never: PYTHONPATH
    entries as such, cwd, or the adapter's/candidate's directories — those are
    excluded explicitly even if a rule above would admit them."""
    import site
    import sysconfig
    roots = []
    # (1) the interpreter's CONFIGURED locations, admitted only when they exist
    # and are actually active on the live sys.path (CI-FAIL-25 addendum: a
    # distro scheme may name a dist-packages directory that is not an active
    # site dir on the runner; a configured-but-inactive directory is not an
    # installed location of THIS interpreter)
    configured = []
    try:
        paths = sysconfig.get_paths()
        configured += [paths.get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")]
    except Exception:  # pragma: no cover
        pass
    for getter in ("getsitepackages", "getusersitepackages"):
        fn = getattr(site, getter, None)
        if fn is None:
            continue
        if getter == "getusersitepackages" and not getattr(site, "ENABLE_USER_SITE", True):
            continue
        try:
            got = fn()
        except Exception:  # pragma: no cover
            continue
        configured += list(got) if isinstance(got, (list, tuple)) else [got]
    on_path = set()
    for entry in sys.path:
        try:
            on_path.add(str(Path(entry or ".").resolve()))
        except Exception:
            continue
    for d in configured:
        try:
            if d and Path(d).is_dir() and str(Path(d).resolve()) in on_path:
                roots.append(d)
        except Exception:
            continue
    roots += [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, str(_CHECKOUT_ROOT)]
    # (3a) origin of the dependencies this process already loaded: wherever the
    # live interpreter actually found them is, by construction, an installed
    # location (resolved through symlinks, as the import guard resolves)
    for name in ("numpy", "torch", "yaml", "numba", "llvmlite", "scipy", "psutil", "torchview", "networkx",
                 "einops", "pandas", "sympy", "mpmath", "typing_extensions", "filelock", "fsspec", "jinja2"):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None)
        if f:
            fp = Path(f).resolve()
            roots.append(str(fp.parent.parent if fp.name == "__init__.py" else fp.parent))
    # (3b) installed site directories present on the live sys.path, whatever
    # mechanism put them there (user site, venv, site customisation)
    for entry in sys.path:
        try:
            ep = Path(entry).resolve()
        except Exception:
            continue
        if ep.is_dir() and ep.name in ("site-packages", "dist-packages") and any(ep.glob("*.dist-info")):
            roots.append(str(ep))
    # (2) the environment's OWN import configuration inside the site dirs
    for site_dir in [r for r in roots if r and str(r).rstrip("/").endswith(("site-packages", "dist-packages"))]:
        sd = Path(site_dir)
        if not sd.is_dir():
            continue
        try:
            for pth in sd.glob("*.pth"):
                for line in pth.read_text(errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "import ", "import\t")) and Path(line).is_dir():
                        roots.append(line)
            for link in sd.glob("*.egg-link"):
                first = link.read_text(errors="replace").splitlines()[:1]
                if first and Path(first[0].strip()).is_dir():
                    roots.append(first[0].strip())
            for du in sd.glob("*.dist-info/direct_url.json"):
                try:
                    info = json.loads(du.read_text())
                except Exception:
                    continue
                if (info.get("dir_info") or {}).get("editable") and str(info.get("url", "")).startswith("file://"):
                    from urllib.parse import unquote, urlparse
                    target = unquote(urlparse(info["url"]).path)
                    if Path(target).is_dir():
                        roots.append(target)
        except Exception:  # pragma: no cover - unreadable site metadata
            pass
    out = []
    for r in roots:
        if r:
            try:
                rr = str(Path(r).resolve())
            except Exception:
                continue
            if rr not in out:
                out.append(rr)
    return out


def _file_sha256(path) -> Optional[str]:
    try:
        h = _TRUSTED_SHA256()
        with _TRUSTED_OPEN(str(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


class _ProgramTampered(Exception):
    """The comparator program (or one of its imports) no longer hashes to the
    import-time constant: nothing may be concluded from any phase."""


_PRIVATE_ROOT_PREFIX = ".private-evidence-"
_REFERENCE_CENSUS = "reference-census.json"
_PRIVATE_STATE = "PRIVATE_STATE.json"     # the root's own state: its id and its path (never in a public record)
# the dispatch's parent-private evidence root, published by _Dispatch for the
# validator and the phase functions (a context variable: signatures unchanged)
_PRIVATE_EVIDENCE_ROOT: "contextvars.ContextVar" = contextvars.ContextVar("solar_private_evidence_root", default=None)


def _private_root_create(agentic_dir: Path) -> Path:
    """Create the dispatch's PARENT-PRIVATE evidence root (round 9, manager ruling
    2): a fresh random name under the dispatch's ``agentic`` directory, created
    EXCLUSIVELY with mode 0700 (``mkdir`` fails if the path exists), outside every
    candidate working root.  Reference material, comparison reports and every
    digest derived from reference bytes live only here; the root is LOCKED
    (mode 0000) while any process that executes generated code is alive."""
    agentic_dir = Path(agentic_dir)
    agentic_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(64):
        path = agentic_dir / (_PRIVATE_ROOT_PREFIX + secrets.token_hex(8))
        try:
            os.mkdir(str(path), 0o700)
        except FileExistsError:
            continue
        os.chmod(str(path), 0o700)
        ident = os.stat(str(path))
        (path / _PRIVATE_STATE).write_text(json.dumps(
            {"id": secrets.token_hex(8), "path": str(path), "dev": ident.st_dev, "ino": ident.st_ino,
             "created_utc": _utc(),
             "note": "public records (DISPATCH_STATE/RECEIPT, phase records) name this root only by its id"},
            indent=1))
        return path
    raise OSError("could not create a private evidence root under %s" % agentic_dir)


def _private_root_id(root) -> Optional[str]:
    """The opaque id of a private root (from its own state file)."""
    if root is None:
        return None
    try:
        info = _TRUSTED_JSON_LOADS((Path(root) / _PRIVATE_STATE).read_text())
        return str(info.get("id")) if isinstance(info, dict) and info.get("id") else None
    except Exception:
        return None


def _private_root_find(agentic_dir: Path, ident: Optional[dict]) -> Optional[Path]:
    """Re-find a dispatch's private root from its recorded inode identity (dev,
    ino) and id: a public record never carries the path, so resume looks the
    root up by identity; a directory with the right name but another inode
    (a decoy) is not it."""
    if not ident:
        return None
    agentic_dir = Path(agentic_dir)
    if not agentic_dir.is_dir():
        return None
    for root in agentic_dir.glob(_PRIVATE_ROOT_PREFIX + "*"):
        try:
            st = os.stat(str(root), follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != _TRUSTED_GETUID():
            continue
        if (st.st_dev, st.st_ino) != (int(ident.get("dev", -1)), int(ident.get("ino", -1))):
            continue
        if ident.get("id") and _private_root_id(root) != ident.get("id"):
            continue
        return root
    return None


def _private_root_for(agentic_dir: Path) -> Path:
    """The dispatch's private root published by ``_Dispatch``; a fresh one for a
    direct validation call (reviewer controls call the phase functions directly)."""
    active = _PRIVATE_EVIDENCE_ROOT.get()
    if active is not None and Path(active).is_dir():
        return Path(active)
    return _private_root_create(agentic_dir)


def _private_subdir(root: Path, prefix: str) -> Path:
    """A fresh, exclusively created 0700 directory inside the private root."""
    for _ in range(64):
        path = Path(root) / f"{prefix}-{secrets.token_hex(4)}"
        try:
            os.mkdir(str(path), 0o700)
        except FileExistsError:
            continue
        os.chmod(str(path), 0o700)
        return path
    raise OSError("could not create a private directory under %s" % root)


def _inode_identity(path) -> dict:
    st = os.stat(str(path), follow_symlinks=False)
    return {"dev": st.st_dev, "ino": st.st_ino, "ctime_ns": st.st_ctime_ns, "mode": st.st_mode & 0o7777}


def _lock_private_root(root: Path) -> dict:
    """Mode 0000: while a process that executes generated code is alive, no path
    below the private root resolves (the same OS user included; the kernel
    checks the owner bits).  The recorded inode identity and ctime detect any
    chmod/replace that happened meanwhile (ctime cannot be set back by a user)."""
    os.chmod(str(root), 0)
    return _inode_identity(root)


def _unlock_private_root(root: Path, locked: dict) -> Optional[str]:
    """Verify the lock survived untouched, then restore owner access; returns a
    description of the problem (the dispatch is refused) or None."""
    try:
        now = _inode_identity(root)
    except OSError as exc:
        return f"the private evidence root is unreadable after the phase ({exc})"
    problem = None
    if (now["dev"], now["ino"]) != (locked["dev"], locked["ino"]):
        problem = "the private evidence root was replaced while generated code ran"
    elif now["mode"] != 0:
        problem = f"the private evidence root's mode changed while generated code ran (0{now['mode']:o})"
    elif now["ctime_ns"] != locked["ctime_ns"]:
        problem = "the private evidence root's inode was modified (ctime) while generated code ran"
    try:
        os.chmod(str(root), 0o700)
    except OSError as exc:
        problem = problem or f"the private evidence root could not be unlocked ({exc})"
    return problem


def _write_private_file(private_dir: Path, prefix: str, data: bytes) -> tuple:
    """Write *data* to a FRESH random filename inside *private_dir* with
    ``O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW`` (mode 0600): a brand-new inode that no
    other path can have been linked to in advance.  After writing, ``fstat`` must
    show exactly one link to a regular file whose directory IS *private_dir*;
    otherwise the material is removed and OSError is raised (the dispatch is
    refused as ``reference_material_retained``).  -> (path, {"dev", "ino"})."""
    private_dir = Path(private_dir)
    want_dir = os.stat(str(private_dir))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    for _ in range(64):
        path = private_dir / f"{prefix}-{secrets.token_hex(8)}"
        try:
            fd = os.open(str(path), flags, 0o600)
        except FileExistsError:
            continue
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            st = os.fstat(fd)
        finally:
            os.close(fd)
        here = os.stat(str(path.parent))
        problem = None
        if st.st_nlink != 1:
            problem = f"{st.st_nlink} links"
        elif not stat.S_ISREG(st.st_mode):
            problem = "not a regular file"
        elif os.path.islink(str(path)):
            problem = "a symbolic link"
        elif (here.st_dev, here.st_ino) != (want_dir.st_dev, want_dir.st_ino):
            problem = "outside the private directory"
        if problem:
            try:
                os.unlink(str(path))
            except OSError:
                pass
            raise OSError(f"reference material not isolated: {path} is {problem}")
        return path, {"dev": st.st_dev, "ino": st.st_ino}
    raise OSError("could not create a private file under %s" % private_dir)


def _census_add(private_root: Path, entry: dict) -> None:
    """Record a reference inode written this dispatch (parent-written, inside the
    private root): the hard-link census consults it before every launch."""
    path = Path(private_root) / _REFERENCE_CENSUS
    try:
        items = _TRUSTED_JSON_LOADS(path.read_text()) if path.is_file() else []
    except Exception:
        items = []
    items.append(entry)
    path.write_text(json.dumps(items, indent=1, default=str))


def _census_load(agentic_dir: Path) -> set:
    """Every (dev, ino) of reference material written during this dispatch, from
    the census files of the private roots under the dispatch's agentic dir."""
    out = set()
    agentic_dir = Path(agentic_dir)
    if not agentic_dir.is_dir():
        return out
    for root in agentic_dir.glob(_PRIVATE_ROOT_PREFIX + "*"):
        try:
            for item in _TRUSTED_JSON_LOADS((root / _REFERENCE_CENSUS).read_text()):
                out.add((int(item["dev"]), int(item["ino"])))
        except Exception:
            continue
    return out


def _seal_reference_material(root: Path, census: Optional[set] = None) -> dict:
    """Defence in depth before every launch of generated code (reference-27):
    remove every entry under *root* that is a raw reference observation by NAME
    (``trusted_outputs.npz``, the canonical name earlier rounds used) or by INODE
    (the census of reference inodes written this dispatch, so a differently
    named hard link is caught).  The proof boundary is the private evidence
    root (exclusive, locked while generated code runs); this walk only removes
    what should not exist.  -> {"removed": [...], "retained": [...]}: anything
    retained refuses the next launch."""
    removed, retained = [], []
    root = Path(root)
    census = set(census or ())
    if root.is_dir():
        targets = {}
        for p in root.rglob("*"):
            try:
                st = os.stat(str(p), follow_symlinks=False)
            except OSError:
                continue
            if p.name == "trusted_outputs.npz" or (not stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) in census):
                targets[str(p)] = (p, stat.S_ISDIR(st.st_mode))
        for key in sorted(targets):
            p, is_dir = targets[key]
            try:
                os.chmod(str(p), 0o700 if is_dir else 0o600)
            except OSError:
                pass
            try:
                os.rmdir(str(p)) if is_dir else os.unlink(str(p))
            except OSError:
                pass
            (retained if (p.exists() or p.is_symlink()) else removed).append(key)
    return {"removed": removed, "retained": retained}


def _active_phases(phases, work: Path, timeout: float, log) -> "_TrustedPhases":
    """The validation's ONE _TrustedPhases: the explicit argument, else the one the
    validator published for this validation, else a fresh one (direct call)."""
    if phases is not None:
        return phases
    active = _ACTIVE_PHASES.get()
    return active if active is not None else _TrustedPhases(work, timeout, log)


def _trusted_reference_check(candidate: Candidate, request: DispatchRequest, policy: DispatchPolicy,
                             tol: dict, adapter_min_tol: float, workdir: Path, log=print,
                             phases: Optional["_TrustedPhases"] = None) -> dict:
    try:
        return _trusted_reference_check_impl(candidate, request, policy, tol, adapter_min_tol, workdir, log, phases)
    except _ProgramTampered as exc:
        return {"status": "infrastructure", "kind": FailureKind.INFRASTRUCTURE, "reason_code": "trusted_program_tampered",
                "message": f"trusted_program_tampered: {exc}; no verdict is derived from any phase",
                "detail": {"program": {"path": str(_TRUSTED_CHILD_PATH), "sha256_at_import": _TRUSTED_PROGRAM_SHA256},
                           "error": str(exc)}}


_ATTRIBUTION_KEYS = ("extension_submodule_of", "package_submodule_of", "attribution_basis", "census_diagnostic")


def _write_attribution(private: Path, attribution: dict) -> None:
    """The dependency census' attribution and diagnostic -- which file-less submodules
    were admitted by descent and on which basis; for every refused one which condition
    failed and the parent's census state -- for the PRIVATE receipt only, written
    after every census-bearing phase so that a refusal is self-explaining (round 11,
    CI-FAIL-27)."""
    try:
        (private / "dependency_attribution.json").write_text(json.dumps(attribution, indent=1, default=str))
    except OSError:
        pass


def _trusted_reference_check_impl(candidate: Candidate, request: DispatchRequest, policy: DispatchPolicy,
                                  tol: dict, adapter_min_tol: float, workdir: Path, log=print,
                                  phases: Optional["_TrustedPhases"] = None) -> dict:
    """The trusted proof for an adapter route, computed entirely outside the
    reach of generated code (reference-14/INPUT-21/TOLERANCE-22/
    ISOLATION-23/REFERENCE-27/INPUT-AUDIT-25, round 9 design ruling).  Fresh
    ``python -I -S`` interpreters, each in its own session and private 0700
    directory, with a killed-and-verified boundary after every phase that
    executed generated code:

    1. ``inputs``    — executes adapter code: produces the input snapshot from
                       ``get_inputs()``.  The parent reads the bytes ONCE and
                       holds them together with per-array hashes: that is the
                       frozen state, and it lives only in the parent.
    2. ``candidate`` — FIRST, in ITS OWN PROCESS: a fresh 0700 working root
                       (mkdtemp under the attempt directory) holding only this
                       launch's inputs (parent-owned ``.npy`` files, mode 0400,
                       mapped READ-ONLY) and the candidate source; cwd is that
                       root; no path to any other attempt, evidence or reference.
                       Outputs come back only as a result file the parent reads
                       after the process EXITED; completion is established by
                       the parent (exit status + parseable result), never by a
                       token available inside the candidate process.  An
                       in-place write faults inside the candidate process
                       (``candidate_mutated_inputs``); after the boundary the
                       parent re-hashes and identity-checks its OWN files.
    3. boundary      — the candidate session is killed and ``/proc`` is checked
                       for survivors (``candidate_descendants_alive`` refuses).
    4. ``reference`` — AFTER the boundary, inside the dispatch's PARENT-PRIVATE
                       evidence root (exclusive 0700 creation, locked to mode
                       0000 while generated code runs): executes the ORIGINAL
                       requested source (hash-checked) under the import guard;
                       the parent copies the outputs to a fresh random name with
                       ``O_EXCL|O_NOFOLLOW`` and verifies ``st_nlink == 1``.
    5. ``compare``   — never runs generated code: reads the parent's private
                       copies by hash and compares under the approved rtol/atol
                       plus a floor derived ONLY from the trusted outputs'
                       dtypes.  The report and every digest of reference bytes
                       stay in the private root; public records carry none.
    The comparator program executes from its installed path; the parent hashes
    it before and after every phase and computes the verdict digest itself.
    Adapter-side tolerance floors are diagnostics only."""
    t0 = time.perf_counter()
    detail = {"source_path": str(request.source_path), "requested_sha256": candidate.source_sha256,
              "adapter_path": str(candidate.model_path), "adapter_sha256": candidate.adapter_sha256,
              "isolation": None, "entry": None, "executed_source_sha256": None,
              "inputs_sha256": None, "trusted_outputs_sha256": None, "candidate_outputs_sha256": None,
              "verdict_sha256": None, "max_abs_error": None, "compared_outputs": None,
              "tolerance": None, "adapter_side_min_tol": float(adapter_min_tol or 0.0),
              "tolerance_widening_rejected": None, "input_audit": None, "report": None,
              "phases": {}, "session_boundaries": {}, "seconds": None}

    def _pub(d):
        # fix (c): no private-root path in any public record
        return _redact_private_paths(d, phases.private_root, _private_root_id(phases.private_root))

    def verdict(status, kind, code, message, public=None):
        detail["seconds"] = time.perf_counter() - t0
        return {"status": status, "kind": kind, "reason_code": code, "message": _pub(message),
                "detail": _pub(detail), "public_message": public}

    # 0. the original bytes on disk ARE the requested bytes (checked again
    #    inside the reference child immediately before execution)
    now = _file_sha256(request.source_path) if Path(request.source_path).is_file() else None
    if now != candidate.source_sha256:
        return verdict("source_changed", FailureKind.PROTOCOL, "source_changed",
                       f"trusted reference refused: the requested source on disk is {str(now)[:12]}…, "
                       f"the candidate was bound to {candidate.source_sha256[:12]}…")
    work = Path(workdir) / "trusted_reference"
    timeout = float(policy.verify_timeout) if policy.verify_timeout else _TRUSTED_REFERENCE_DEFAULT_TIMEOUT
    phases = _active_phases(phases, work, timeout, log)
    agentic_dir = Path(workdir).parent
    # the dispatch's PARENT-PRIVATE evidence root (round 9): exclusive 0700
    # creation, outside every candidate working root, locked (mode 0000) while
    # generated code runs; reference material, comparison reports and every
    # digest derived from reference bytes live only here
    try:
        if phases.private_root is None:
            phases.private_root = _private_root_for(agentic_dir)
        private = _private_subdir(phases.private_root, "validation")
        # evidence copy of the program TEXT next to the private report (never executed)
        (private / "trusted_child.py").write_text(_TRUSTED_CHILD)
        os.chmod(str(private / "trusted_child.py"), 0o444)
    except OSError as exc:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_material_retained",
                       f"reference_material_retained: no private evidence directory could be created ({exc})")
    private_id = _private_root_id(phases.private_root)
    excluded = sorted({str(Path(candidate.model_path).resolve().parent),
                       str(Path(candidate.dsl_path).resolve().parent), str(work.resolve()),
                       str(Path(phases.private_root).resolve())})
    # parent-controlled copies of the artifacts: ALL inside the private directory;
    # the reference copy gets a fresh exclusively created random name
    final = {"inputs": private / "inputs.npz", "trusted_outputs": None,
             "candidate_outputs": private / "candidate_outputs.npz"}
    # the two NON-secret artifacts (the inputs, the candidate's own outputs) are also
    # kept at their historical paths in the attempt's work directory; the reference
    # never is, and (round 10) no alias of it either: a symlink's target would name
    # the private root in the artifact tree
    public = {"inputs": work / "inputs.npz", "candidate_outputs": work / "candidate_outputs.npz"}
    detail["isolation"] = {"python": sys.executable, "flags": ["-I", "-S"], "cwd": str(work), "per_phase_dirs": True,
                           "phase_dir_mode": "0700", "new_session_per_phase": True,
                           "trusted_roots": phases.roots, "trusted_roots_sha256": phases.roots_sha256,
                           "sys_path": phases.roots, "excluded_paths": excluded,
                           "env_keys": sorted(phases.env), "timeout_seconds_per_phase": timeout,
                           "phases": ["inputs", "candidate", "boundary", "reference", "compare"],
                           "program": {"path": str(_TRUSTED_CHILD_PATH), "sha256": _TRUSTED_PROGRAM_SHA256,
                                       "imports_sha256": _TRUSTED_PROGRAM_HASHES, "verified_before_each_phase": True,
                                       "verified_after_each_phase": True, "executed_from_installed_path": True,
                                       "child_self_report_trusted": False},
                           "session_boundary": {"kill": "killpg SIGTERM then SIGKILL",
                                                "verified_on": "/proc: session id, process group, re-parented orphans",
                                                "subreaper": _SUBREAPER},
                           "private_evidence_id": private_id,
                           "private_evidence_path_in_public_records": False,
                           "candidate_process": {
                               "one_process_per_launch": True,
                               "working_root": "a fresh 0700 directory (mkdtemp under the attempt directory) holding "
                                               "ONLY this launch's inputs (.npy, mode 0400) and the candidate source; "
                                               "cwd is that root; no path to any other attempt, evidence or reference",
                               "inputs": "read-only shared mappings (np.load mmap_mode='r' over parent-owned 0400 files); "
                                         "an in-place write faults inside the candidate process",
                               "outputs": "a result file in the working root, read by the parent after the process EXITED",
                               "completion": "established by the PARENT (exit status + parseable result); no token "
                                             "available inside the candidate process plays any role",
                               "nonce": None},
                           "reference_isolation": {
                               "written_with": "O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, mode 0600, fresh random name; fstat "
                                               "nlink==1, regular file, directory identity checked",
                               "location": "private evidence root only (never a path a candidate could pre-link)",
                               "digests": "report_compare.json and every digest of reference bytes stay in the private "
                                          "root; public records carry none, and no public record names the root's path",
                               "result_ingest": "every candidate-written path (report.json, result archives) is opened "
                                                "O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC, fstat-checked (regular file, "
                                                "one link, our uid, size cap, directory-entry identity) and read within "
                                                "the remaining dispatch budget BEFORE the private root is unlocked; the "
                                                "report is parsed against an allowlist schema, never merged raw",
                               "locked_while_generated_code_runs": True,
                               "seal_before_each_launch": "by name and by reference-inode census (defence in depth)"},
                           "frozen_state": "input snapshot bytes and per-array hashes held by the parent; the candidate's "
                                           "files are parent-owned, 0400, hashed and identity-checked after the boundary",
                           "parent_primitives": "subprocess.Popen/os.killpg/hashlib.sha256/json.loads/open/os.replace "
                                                "captured at import; no generated code executes in the parent"}
    detail["private_evidence_id"] = private_id
    result_cap = int(getattr(policy, "max_result_bytes", 1 << 30) or (1 << 30))
    detail["result_caps_bytes"] = {"report": _REPORT_BYTES_CAP, "result_archive": result_cap}

    def run_phase(phase: str, meta: dict, pdir: Optional[Path] = None, results: Optional[dict] = None):
        rc, rep = phases.run(phase, meta, pdir, results=results)
        detail["phases"] = dict(phases.records)
        if rep.get("session_boundary") is not None:
            detail["session_boundaries"][phase] = rep["session_boundary"]
        if rc == -3:
            raise _ProgramTampered(rep.get("error") or "trusted_program_tampered")
        return rc, rep

    def survivors(rep):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "candidate_descendants_alive",
                       str(rep.get("error") or "candidate_descendants_alive"))

    def tampered_root(rep):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "evidence_root_tampered",
                       str(rep.get("error") or "evidence_root_tampered"))

    def unverifiable(message, public=None):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "trusted_verdict_unverifiable", message,
                       public=public)

    # 1. inputs (adapter code) in its own session; the parent keeps the bytes
    pdir_in = phases.new_dir("inputs")
    rc, rep = run_phase("inputs", {"adapter_path": str(Path(candidate.model_path).resolve()),
                                   "inputs_path": str(pdir_in / "inputs.npz")}, pdir_in,
                        results={"inputs": (pdir_in / "inputs.npz", result_cap)})
    if rc == -4:
        return survivors(rep)
    if rc == -5:
        return tampered_root(rep)
    if rc != 0 or not rep.get("ok"):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       f"reference_unexecutable: adapter input metadata unavailable "
                       f"(phase inputs exit {rc}): {rep.get('error') or rep.get('result_problem') or rep.get('stderr_tail')}")
    # the snapshot the adapter process wrote: ingested (hardened) while the
    # private root was still locked; the bytes below are the parent's only copy
    snap = (phases.ingested.get("inputs") or {}).get("inputs") or {}
    if not snap.get("ok"):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       f"reference_unexecutable: the input snapshot was not admitted ({snap.get('problem')})")
    frozen_bytes = snap["bytes"]
    inputs_sha = _TRUSTED_SHA256(frozen_bytes).hexdigest()
    if inputs_sha != rep.get("inputs_sha256"):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       "reference_unexecutable: the input snapshot could not be frozen (hash disagreement)")
    with _TRUSTED_OPEN(str(final["inputs"]), "wb") as fh:
        fh.write(frozen_bytes)
    with _TRUSTED_OPEN(str(public["inputs"]), "wb") as fh:
        fh.write(frozen_bytes)
    frozen_arrays = _array_fingerprints(final["inputs"])
    if _file_sha256(final["inputs"]) != inputs_sha or frozen_arrays is None:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       "reference_unexecutable: the frozen input snapshot is not a readable array archive")
    detail["inputs_sha256"] = inputs_sha
    detail["input_audit"] = {"frozen": frozen_arrays, "inputs_after_available": None, "mutated_inputs": None,
                             "method": "the candidate process receives READ-ONLY shared mappings of parent-owned 0400 "
                                       ".npy files; an in-place write faults in that process (refused "
                                       "candidate_mutated_inputs); after the session boundary the parent re-hashes "
                                       "and identity-checks (inode, ctime, mode) its OWN files against the pre-launch "
                                       "record.  This covers the observed END STATE of the parent's inputs (plus any "
                                       "inode-level write history the kernel records); it is not a trace of every "
                                       "intervening write.  A private copy changed inside the candidate process is not "
                                       "a purity violation of the inputs: outputs are judged by the trusted comparison."}
    names = list(rep.get("names") or [])
    detail["entry"] = rep.get("entry")
    arg_order = names
    output_names = ["output"]
    if request.definition_dir is not None:
        try:
            spec = _TRUSTED_JSON_LOADS(Path(request.definition_dir, "definition.json").read_text())
            spec_inputs = list((spec.get("inputs") or {}).keys())
            if spec_inputs and set(spec_inputs) == set(names):
                arg_order = spec_inputs
            spec_outputs = list((spec.get("outputs") or {}).keys())
            if spec_outputs:
                output_names = spec_outputs
        except Exception:
            pass
    declared = rep.get("args")
    if isinstance(declared, list) and set(declared) == set(names):
        arg_order = declared
    # 2. the candidate FIRST, in ITS OWN PROCESS with a fresh private working root.
    #    Defence in depth: raw reference observations (by name or by inode) that
    #    should not exist anywhere under the dispatch root are removed first.
    sealed = _seal_reference_material(agentic_dir, census=_census_load(agentic_dir))
    detail["sealed_reference_material"] = sealed
    if sealed["retained"]:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_material_retained",
                       f"reference_material_retained: raw trusted outputs of an earlier validation could not be "
                       f"removed before the candidate launch ({sealed['retained']}); the candidate is not launched")
    try:
        croot = Path(tempfile.mkdtemp(prefix="candidate-", dir=str(work)))
        os.chmod(str(croot), 0o700)
        inputs_dir = croot / "inputs"
        inputs_dir.mkdir(mode=0o700)
        with _TRUSTED_OPEN(str(Path(candidate.dsl_path).resolve()), "rb") as fh:
            code = fh.read()
        if _TRUSTED_SHA256(code).hexdigest() != candidate.dsl_sha256:
            return verdict("candidate_bytes_changed", FailureKind.PROTOCOL, "candidate_bytes_changed",
                           "candidate bytes changed before the candidate launch; refusing to run different bytes")
        cand_src = croot / "candidate.py"
        with _TRUSTED_OPEN(str(cand_src), "wb") as fh:
            fh.write(code)
        os.chmod(str(cand_src), 0o400)
        # the arrays the candidate runs on are PARENT-OWNED files (one .npy per
        # array, mode 0400) mapped READ-ONLY by the wrapper; the parent records
        # hash + inode identity now and re-checks them after the session is gone
        import numpy as np
        inputs_npy, pre = {}, {}
        with np.load(str(final["inputs"]), allow_pickle=False) as z:
            for i, name in enumerate(z.files):
                p = inputs_dir / f"{i:03d}.npy"
                np.save(str(p), np.ascontiguousarray(z[name]), allow_pickle=False)
                os.chmod(str(p), 0o400)
                st = os.stat(str(p))
                inputs_npy[str(name)] = str(p)
                pre[str(name)] = {"sha256": _file_sha256(p), "dev": st.st_dev, "ino": st.st_ino,
                                  "ctime_ns": st.st_ctime_ns, "mode": st.st_mode & 0o7777}
    except Exception as exc:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       f"reference_unexecutable: the candidate working root could not be laid out ({exc})")
    if set(inputs_npy) != set(frozen_arrays) or any(v["sha256"] is None for v in pre.values()):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       "reference_unexecutable: the candidate input layout does not match the frozen snapshot")
    rc, rep = run_phase("candidate", {
        "candidate_path": str(cand_src), "candidate_sha256": candidate.dsl_sha256,
        "inputs_npy": inputs_npy, "inputs_npy_sha256": {k: v["sha256"] for k, v in pre.items()},
        "outputs_path": str(croot / "candidate_outputs.npz"),
        "path_prepend": [str(_CHECKOUT_ROOT)],
        "mem_budget_bytes": (int(policy.mem_budget_gb * 1e9) if policy.mem_budget_gb else None)}, croot,
        results={"candidate_outputs": (croot / "candidate_outputs.npz", result_cap)})
    # 3. boundary passed inside run(); a surviving process or a touched private
    #    root refuses everything
    if rc == -4:
        return survivors(rep)
    if rc == -5:
        return tampered_root(rep)
    # the PARENT's observation: its own input files (still 0400), re-hashed and
    # identity-checked after the boundary; nothing the candidate process wrote or
    # reported about the inputs is admissible
    observed, after = [], {}
    for k, p in inputs_npy.items():
        try:
            st = os.stat(p, follow_symlinks=False)
        except OSError:
            observed.append(f"{k}: the parent's input file was removed")
            after[k] = None
            continue
        now_sha = _file_sha256(p)
        after[k] = {"sha256": now_sha, "dev": st.st_dev, "ino": st.st_ino, "ctime_ns": st.st_ctime_ns,
                    "mode": st.st_mode & 0o7777}
        if now_sha != pre[k]["sha256"]:
            observed.append(f"{k}: changed in place")
        elif (st.st_dev, st.st_ino) != (pre[k]["dev"], pre[k]["ino"]):
            observed.append(f"{k}: replaced (the parent's input path is a different inode)")
        elif st.st_ctime_ns != pre[k]["ctime_ns"] or (st.st_mode & 0o7777) != pre[k]["mode"]:
            observed.append(f"{k}: written and restored (the inode's metadata changed while the candidate ran)")
    fault = rep.get("fault") if isinstance(rep.get("fault"), str) else None
    if rc == 10 or fault == "readonly_input_write":
        observed.append(f"inputs: {rep.get('error') or 'attempted in-place write to a read-only input'}")
    # the wrapper's record (did the candidate rebind the input entries it was
    # handed, or write a private buffer it re-pointed an input object at?) comes
    # from the candidate process: it may only REFUSE, never admit
    identity = list(rep.get("identity_audit") or []) if isinstance(rep.get("identity_audit"), list) else []
    mutated = list(dict.fromkeys(observed + identity))
    detail["input_audit"].update(
        inputs_after_available=bool(after) and all(v is not None for v in after.values()),
        mutated_inputs=mutated,
        parent_observation={"method": "the parent re-hashed and identity-checked its own 0400 input files after the "
                                      "session boundary (sha256, inode, ctime, mode)",
                            "before": pre, "after": after, "mutated": observed},
        wrapper_observation={"identity": identity, "use": "refusal only; never admits"},
        write_fault=rep.get("error") if (rc == 10 or fault) else None,
        candidate_reported_not_admissible=sorted(k for k in ("identity_audit", "inputs_after_sha256", "audit") if k in rep))
    phases.records.get("candidate", {})["mutated_inputs"] = mutated
    detail["phases"] = dict(phases.records)
    attribution = {"candidate": {k: rep.get(k) or {} for k in _ATTRIBUTION_KEYS}}
    _write_attribution(private, attribution)
    if mutated:
        return verdict("candidate_mutated_inputs", FailureKind.UNSUPPORTED, "candidate_mutated_inputs",
                       f"candidate_mutated_inputs: the candidate is not a pure function of its inputs "
                       f"({mutated}); the frozen snapshot is held by the parent and the candidate is refused")
    if rc == 8:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_dependency_untrusted",
                       f"candidate_dependency_untrusted: {rep.get('error')}")
    if rc == _RC_ISOLATION_UNAVAILABLE or (rep.get("exception") or {}).get("type") == "IsolationUnavailableError":
        # the facade could not start its isolating child inside the candidate process: an environment
        # fact, typed by the child (exit code / exception type, no message matching), never the candidate's
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "isolation_unavailable",
                       f"isolation_unavailable: {rep.get('error')}")
    if rc in (-1, -2, 9):
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "candidate_execution_failed",
                       f"candidate execution on the frozen trusted inputs failed (exit {rc}): "
                       f"{rep.get('error') or rep.get('stderr_tail') or ''}")
    if rep.get("error") and not rep.get("ok"):
        return verdict("candidate_error", FailureKind.UNSUPPORTED, "candidate_execution_failed",
                       f"candidate execution on the frozen trusted inputs failed (exit {rc}): {rep.get('error')}")
    outs_ingest = (phases.ingested.get("candidate") or {}).get("candidate_outputs") or {}
    if rc != 0 or rep.get("ok") is not True or not outs_ingest.get("ok"):
        why = "; ".join(p for p in (rep.get("result_problem"), outs_ingest.get("problem")) if p) \
            or ("the report does not say ok" if rc == 0 else "no parseable result")
        return verdict("candidate_result_missing", FailureKind.UNSUPPORTED, "candidate_result_missing",
                       f"candidate_result_missing: the candidate process ended (exit {rc}) without an admitted "
                       f"result in its working root ({why}); completion is established by the parent from the "
                       f"exit status and the ingested result files (regular files of ours, one link, within the "
                       f"byte cap, read within the remaining budget), never by anything the candidate reports")
    cand_bytes = outs_ingest["bytes"]
    cand_sha = _TRUSTED_SHA256(cand_bytes).hexdigest()
    if cand_sha != rep.get("outputs_sha256"):
        return unverifiable("candidate outputs hash disagreement between the candidate process and the parent")
    with _TRUSTED_OPEN(str(final["candidate_outputs"]), "wb") as fh:
        fh.write(cand_bytes)
    if _file_sha256(final["candidate_outputs"]) != cand_sha:
        return unverifiable("the parent's copy of the candidate outputs does not hash to the candidate outputs")
    with _TRUSTED_OPEN(str(public["candidate_outputs"]), "wb") as fh:
        fh.write(cand_bytes)
    detail["candidate_outputs_sha256"] = cand_sha
    if _file_sha256(final["inputs"]) != inputs_sha:
        return unverifiable("the frozen input snapshot changed after candidate execution")
    # 4. the trusted reference: AFTER the candidate boundary, inside the private
    #    directory (created only now), on its own copy written by the parent
    pdir_r = phases.new_dir("reference", parent=private)
    with _TRUSTED_OPEN(str(pdir_r / "inputs.npz"), "wb") as fh:
        fh.write(frozen_bytes)
    rc, rep = run_phase("reference", {
        "source_path": str(Path(request.source_path).resolve()), "source_sha256": candidate.source_sha256,
        "inputs_path": str(pdir_r / "inputs.npz"), "inputs_sha256": inputs_sha,
        "outputs_path": str(pdir_r / "trusted_outputs.npz"), "entry": detail["entry"],
        "arg_order": arg_order, "output_names": output_names, "path_prepend": []}, pdir_r)
    detail["report"] = {k: v for k, v in rep.items() if k not in ("outputs_sha256",) + _ATTRIBUTION_KEYS}
    attribution["reference"] = {k: rep.get(k) or {} for k in _ATTRIBUTION_KEYS}
    # manager ruling (CI finding, torch._C.*): which file-less submodules were admitted
    # by descent from a census-verified file module, on which basis, and for every
    # refused one which condition failed -- private receipt only
    _write_attribution(private, attribution)
    detail["dependency_attribution"] = "dependency_attribution.json in the private evidence root (path not published)"
    detail["executed_source_sha256"] = rep.get("source_sha256")
    detail["entry"] = rep.get("entry", detail["entry"])
    if rc == -4:
        return unverifiable(str(rep.get("error")))
    if rc == 8:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_dependency_untrusted",
                       f"reference_dependency_untrusted: {rep.get('error')}")
    if rc != 0 or not rep.get("ok") or not (pdir_r / "trusted_outputs.npz").is_file():
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_unexecutable",
                       f"reference_unexecutable: the trusted reference could not be executed from the "
                       f"original source (exit {rc}): {rep.get('error') or rep.get('stderr_tail')}")
    if rep.get("source_sha256") != candidate.source_sha256:
        return verdict("source_changed", FailureKind.PROTOCOL, "source_changed",
                       "trusted reference refused: the reference child executed different source bytes")
    trusted_sha = _file_sha256(pdir_r / "trusted_outputs.npz")
    if not trusted_sha or trusted_sha != rep.get("outputs_sha256"):
        return unverifiable("trusted outputs hash disagreement between the reference child and the parent")
    with _TRUSTED_OPEN(str(pdir_r / "trusted_outputs.npz"), "rb") as fh:
        trusted_bytes = fh.read()
    try:
        final["trusted_outputs"], ident = _write_private_file(private, "reference", trusted_bytes)
    except OSError as exc:
        return verdict("infrastructure", FailureKind.INFRASTRUCTURE, "reference_material_retained",
                       f"reference_material_retained: {exc}")
    _census_add(phases.private_root, {**ident, "path": str(final["trusted_outputs"]), "validation_dir": str(private)})
    try:
        os.unlink(str(pdir_r / "trusted_outputs.npz"))      # one private copy, census-tracked
    except OSError:
        pass
    try:
        # canonical name inside the private directory: a symlink to the exclusively
        # created copy (same lock, same seal; readers of the private report find it)
        os.symlink(final["trusted_outputs"].name, str(private / "trusted_outputs.npz"))
    except OSError:
        pass
    if _file_sha256(final["trusted_outputs"]) != trusted_sha:
        return unverifiable("the parent's copy of the trusted outputs does not hash to the reference outputs")
    # 5. compare, in a process that never ran generated code, on the parent's
    #    private copies; the report stays in the private directory
    pdir_cmp = phases.new_dir("compare", parent=private)
    rc, rep = run_phase("compare", {
        "inputs_path": str(final["inputs"]), "inputs_sha256": inputs_sha,
        "trusted_outputs_path": str(final["trusted_outputs"]), "trusted_outputs_sha256": trusted_sha,
        "candidate_outputs_path": str(final["candidate_outputs"]), "candidate_outputs_sha256": cand_sha,
        "rtol": float(tol["rtol"]), "atol": float(tol["atol"])}, pdir_cmp)
    try:
        (private / "report_compare.json").write_text(json.dumps(rep, indent=1, default=str))
    except OSError:
        pass
    detail["private_report"] = "report_compare.json in the private evidence root (path not published)"
    detail["public_reference_alias"] = None      # round 10: no alias of the reference in the artifact tree
    v = rep.get("verdict") if isinstance(rep.get("verdict"), dict) else None
    if rc == -4:
        return unverifiable(str(rep.get("error")))
    # round 13 (the review NUMERIC-FEEDBACK-30/D, ruling a): every check on the compare child's
    # VERDICT -- a report the parent cannot use, a parse that dropped or truncated anything, a
    # re-hash or digest disagreement, an inconsistent verdict -- is ONE public failure class with
    # ONE constant public message.  The checks read the private numbers, so WHICH check refused
    # is private (``message``, restored in the terminal receipt; ``verdict_anomalies.json`` in the
    # private root).  Nothing is waived: any anomaly fails closed.
    anomalies = []
    if rc != 0 or not rep.get("ok") or v is None:
        anomalies.append(f"the trusted comparison could not be computed (exit {rc}): "
                         f"{rep.get('error') or rep.get('stderr_tail')}")
        digest = rehash = None
    else:
        stats = rep.get("report_schema") if isinstance(rep.get("report_schema"), dict) else {}
        if stats.get("unknown_or_mistyped_dropped") or stats.get("truncated") or stats.get("parsed") is False:
            anomalies.append(f"the compare child's report did not parse whole against the allowlist "
                             f"(dropped {stats.get('unknown_or_mistyped_dropped')}, truncated {stats.get('truncated')}, "
                             f"parsed {stats.get('parsed')})")
        # the parent re-hashes the persisted arrays and computes the digest itself
        digest = phases.digest_for(v, rep)
        rehash = {k: _file_sha256(final[k]) for k in final}
        if rehash != v.get("artifact_sha256") or \
                rehash != {"inputs": inputs_sha, "trusted_outputs": trusted_sha, "candidate_outputs": cand_sha}:
            anomalies.append("the trusted verdict does not re-hash to the parent's artifacts")
        if digest != rep.get("verdict_sha256"):
            anomalies.append("the compare child's verdict digest disagrees with the parent's digest over the "
                             "program hash, the launch argv and the environment")
        # single source of truth: success is exactly "no comparison error"
        if bool(v.get("success")) != (v.get("error") is None):
            anomalies.append(f"the trusted verdict is internally inconsistent (success={v.get('success')!r}, "
                             f"error={v.get('error')!r})")
    if anomalies:
        try:
            (private / "verdict_anomalies.json").write_text(json.dumps(
                {"anomalies": anomalies, "returncode": rc, "report_schema": rep.get("report_schema"),
                 "verdict": v, "verdict_sha256_child": rep.get("verdict_sha256"), "verdict_sha256_parent": digest,
                 "artifact_sha256_parent": rehash}, indent=1, default=str))
        except OSError:
            pass
        return unverifiable("trusted_verdict_unverifiable: " + "; ".join(anomalies), public=_UNVERIFIABLE_PUBLIC_MESSAGE)
    # the full verdict (artifact digests included) and the parent's digest over it
    # stay PRIVATE: a digest of the reference bytes is a finite-domain oracle for a
    # later candidate.  The public digest binds the program hash, the launch argv,
    # the environment and the outcome, not the reference bytes.
    try:
        (private / "verdict_private.json").write_text(json.dumps(
            {"verdict": v, "verdict_sha256": digest, "artifacts": {k: str(p) for k, p in final.items()},
             "artifact_sha256": rehash, "dependency_attribution": attribution}, indent=1, default=str))
    except OSError:
        pass
    # round 11 (the review NUMERIC-FEEDBACK-30/B): the PUBLIC digest is computed over the
    # attempt-visible representation of the verdict -- numeric statistics emptied,
    # numerals in free text redacted; exactly what the public compare record shows --
    # so it depends on no withheld field.  The digest over the full verdict is in
    # verdict_private.json only.
    detail["verdict_sha256"] = phases.digest_for(_public_verdict_view(v), rep)
    detail["verdict_digest_basis"] = ("attempt-visible verdict (numeric statistics emptied, numerals in text "
                                      "redacted); the full-verdict digest is in the private root only")
    detail["trusted_outputs_sha256"] = None       # private (verdict_private.json), never in the artifact tree
    # numeric error statistics (numeric-feedback-30, manager ruling): the
    # dispatcher keeps them out of attempt-visible state and surfaces them only
    # in the final receipt; attempt-visible state carries pass/fail per output
    # and the tolerance policy
    detail["max_abs_error"] = v.get("max_abs_error")
    detail["per_output_max_abs_error"] = v.get("per_output_max_abs_error")
    detail["outputs_pass"] = v.get("outputs_pass")
    detail["compared_outputs"] = v.get("compared_outputs")
    detail["tolerance"] = v.get("tolerance")
    floor = float((v.get("tolerance") or {}).get("trusted_dtype_floor") or 0.0)
    detail["tolerance_widening_rejected"] = bool(float(adapter_min_tol or 0.0) > floor + 1e-18)
    if detail["tolerance_widening_rejected"]:
        log(f"[dispatch] tolerance_widening_rejected: adapter-side floor {float(adapter_min_tol):.3e} "
            f"ignored; trusted gate uses rtol={tol['rtol']} atol={tol['atol']} floor={floor:.3e}")
    err = v.get("error") or ""
    if err.startswith("Output key mismatch"):
        return verdict("reference_outputs_unaligned", FailureKind.PROTOCOL, "reference_outputs_unaligned", err)
    if not v.get("success"):
        # round 13 (the review NUMERIC-FEEDBACK-30/D): a FAILING candidate is the mismatch class
        # whatever the reference looked like -- vacuousness (an all-zero/non-finite reference) is
        # decided only for a passing candidate, so the public class does not tell an all-zero
        # reference from any other; ``vacuous_outputs`` itself is withheld until the terminal receipt
        failed = sorted(k for k, ok in (v.get("outputs_pass") or {}).items() if ok is False)
        return verdict("trusted_reference_mismatch", FailureKind.UNSUPPORTED, "trusted_reference_mismatch",
                       f"trusted_reference_mismatch: the candidate disagrees with the trusted reference "
                       f"executed from the original source (outputs failing the tolerance policy: "
                       f"{failed or 'see outputs_pass'}; {err}); the adapter's own reference is not trusted; "
                       f"the error statistics are in the final DISPATCH_RECEIPT.json only")
    if v.get("vacuous_outputs"):
        return verdict("no_proof", FailureKind.UNSUPPORTED, "trusted_reference_vacuous",
                       f"trusted reference outputs {v['vacuous_outputs']} are all-zero/non-finite: "
                       f"the comparison is vacuous")
    detail["seconds"] = time.perf_counter() - t0
    return {"status": "ok", "detail": _pub(detail)}

class _VerifySummary:
    """The verify child's report as the validator reads it (attribute access
    mirrors ``VerifyResult`` for the binding/receipt logic)."""

    def __init__(self, summary: dict):
        import types as _types
        self.status = summary.get("status")
        self.message = summary.get("message") or ""
        self.max_abs_error = summary.get("max_abs_error")
        self.vacuous_outputs = list(summary.get("vacuous_outputs") or [])
        ev = summary.get("execution")
        self.execution = None if not isinstance(ev, dict) else _types.SimpleNamespace(
            source_sha256=ev.get("source_sha256"), reference_sha256=ev.get("reference_sha256"),
            local_dependencies=tuple(tuple(d) for d in (ev.get("local_dependencies") or [])),
            unsupported_dependencies=tuple(ev.get("unsupported_dependencies") or []),
            min_tol=float(ev.get("min_tol") or 0.0))
        self.receipt_path = summary.get("receipt_path")
        self.receipt_sha256 = summary.get("receipt_sha256")
        self.receipt_refused = summary.get("receipt_refused")
        self.receipt_error = summary.get("receipt_error")


def _classify_exception_names(name, mro: list, rc: int) -> str:
    """classify_exception, from a child's reported exception type names."""
    names = set(mro or []) | ({name} if name else set())
    if rc in (-1, -2, _RC_ISOLATION_UNAVAILABLE) or "IsolationUnavailableError" in names:
        return FailureKind.INFRASTRUCTURE
    if "FileNotFoundError" in names:
        return FailureKind.INPUT
    if names & _CANDIDATE_INFRA_TYPES or names & _INFRA_EXC_NAMES or "OSError" in names:
        return FailureKind.INFRASTRUCTURE
    return FailureKind.UNSUPPORTED


def _run_verify_phase(candidate: Candidate, request: DispatchRequest, policy: DispatchPolicy, tol,
                      work: Path, staged_receipt: Path, log=print, phases: Optional["_TrustedPhases"] = None):
    """Run the facade verifier (and receipt emission on a pass) in a trusted-
    program child in its own session; the session is killed and verified gone
    afterwards.  -> (summary dict, None) or (None, (kind, status, reason_code, message))."""
    timeout = float(policy.verify_timeout) if policy.verify_timeout else _TRUSTED_REFERENCE_DEFAULT_TIMEOUT
    phases = _active_phases(phases, work, timeout, log)
    meta = {"model_path": str(Path(candidate.model_path).resolve()),
            "model_sha256": candidate.adapter_sha256 or candidate.source_sha256,
            "dsl_path": str(Path(candidate.dsl_path).resolve()), "dsl_sha256": candidate.dsl_sha256,
            "source_type": request.source_type,
            "rtol": None if tol is None else float(tol["rtol"]), "atol": None if tol is None else float(tol["atol"]),
            "mem_budget_gb": policy.mem_budget_gb, "verify_timeout": policy.verify_timeout,
            "definition_dir": None if request.definition_dir is None else str(Path(request.definition_dir).resolve()),
            "dtype": request.dtype, "workload_idx": request.workload_idx, "receipt_path": str(staged_receipt)}
    rc, rep = phases.run("verify", meta, results={"receipt": (staged_receipt, _RECEIPT_BYTES_CAP)})
    exc_rep = rep.get("exception") or {}
    if rc == _RC_ISOLATION_UNAVAILABLE or exc_rep.get("type") == "IsolationUnavailableError" \
            or "IsolationUnavailableError" in (exc_rep.get("mro") or []):
        # the facade could not start its isolating child: an ENVIRONMENT failure, typed (the child reports
        # the exception type; no message matching) -- infrastructure, unscored, the attempt retained
        return None, (FailureKind.INFRASTRUCTURE, "infrastructure", "isolation_unavailable",
                      f"isolation_unavailable: {rep.get('error')}")
    if rc == -3:
        return None, (FailureKind.INFRASTRUCTURE, "infrastructure", "trusted_program_tampered",
                      f"trusted_program_tampered: {rep.get('error')}")
    if rc == -4:
        return None, (FailureKind.INFRASTRUCTURE, "infrastructure", "candidate_descendants_alive",
                      str(rep.get("error") or "candidate_descendants_alive"))
    if rc == 4:
        return None, (FailureKind.PROTOCOL, "candidate_bytes_changed", "candidate_bytes_changed",
                      f"candidate_bytes_changed: {rep.get('error')}")
    if rc != 0 or not rep.get("ok") or not isinstance(rep.get("verify"), dict):
        exc = rep.get("exception") or {}
        kind = _classify_exception_names(exc.get("type"), exc.get("mro") or [], rc)
        return None, (kind, "infrastructure" if kind == FailureKind.INFRASTRUCTURE else "candidate_error",
                      "verify_raised", f"verify child failed (exit {rc}): "
                                       f"{rep.get('error') or rep.get('result_problem') or rep.get('stderr_tail')}")
    summary = dict(rep["verify"])
    summary["phase"] = {k: v for k, v in rep.items()
                        if k in ("program_sha256_at_start", "program_sha256_at_end", "program_sha256_parent",
                                 "program_check_after_phase", "argv", "flags", "session_boundary")}
    return summary, None

def default_validator(candidate: Candidate, request: DispatchRequest,
                      policy: DispatchPolicy, receipt_out: Path, log=print) -> ValidationVerdict:
    token = _ACTIVE_PHASES.set(None)
    try:
        verdict = _default_validator_impl(candidate, request, policy, receipt_out, log)
    finally:
        _ACTIVE_PHASES.reset(token)
    if verdict.status != "verified":
        # a receipt the verify child staged is never left behind unadmitted
        staged = Path(receipt_out).parent / "trusted_reference" / "receipt.staged.json"
        try:
            if staged.is_file():
                staged.unlink()
        except OSError:
            pass
    return verdict


def _default_validator_impl(candidate: Candidate, request: DispatchRequest,
                            policy: DispatchPolicy, receipt_out: Path, log=print) -> ValidationVerdict:
    """Run ``solar.translate.verify`` on the candidate under the EXPLICIT
    tolerance policy and, when the request carries a definition and dtype,
    emit the hash-bound receipt.

    Tolerance (tolerance-6): request -> selected workload line ->
    unresolved.  With an unresolved policy the candidate is still executed
    (so candidate defects are detected and the error statistic recorded),
    but neither pass nor fail is admitted: ``tolerance_unresolved``
    (unscored, actionable).  ``required_matched_ratio`` is recorded as the
    row's match-ratio gate; the facade checker applies all-element
    closeness, which is not relaxed here.  Candidate-execution failures are
    classified by exception TYPE only (classify-3).
    """
    t0 = time.perf_counter()
    tol, tol_src = resolve_tolerance(request)
    # bytes about to execute must be the bytes the candidate was bound to
    # (retry-16: a retry never validates different bytes)
    for label, path, want in (("candidate", candidate.dsl_path, candidate.dsl_sha256),
                              ("adapter", candidate.model_path, candidate.adapter_sha256)):
        if want is None:
            continue
        have = _sha256(path) if Path(path).is_file() else None
        if have != want:
            return ValidationVerdict(
                status="candidate_bytes_changed", kind=FailureKind.PROTOCOL,
                message=f"{label} bytes changed before validation: bound {want[:12]}…, "
                        f"on disk {str(have)[:12]}…; refusing to validate different bytes",
                verify_status="refused", seconds=time.perf_counter() - t0,
                tolerance=tol, tolerance_source=tol_src, reason_code="candidate_bytes_changed")
    # the facade verifier runs in a CHILD (isolation-23): adapter and
    # candidate code never execute in the dispatcher's own process
    work = Path(receipt_out).parent / "trusted_reference"
    staged_receipt = work / "receipt.staged.json"
    timeout = float(policy.verify_timeout) if policy.verify_timeout else _TRUSTED_REFERENCE_DEFAULT_TIMEOUT
    # ONE trusted-roots computation for this validation, published to the phase
    # functions through _ACTIVE_PHASES (their signatures are unchanged)
    phases = _TrustedPhases(work, timeout, log)
    try:
        # the dispatch's parent-private evidence root (locked during the verify,
        # inputs and candidate phases; home of the reference and compare phases)
        phases.private_root = _private_root_for(Path(receipt_out).parent.parent)
    except OSError as exc:
        return ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                 message=f"reference_material_retained: no private evidence root could be created ({exc})",
                                 verify_status="refused", seconds=time.perf_counter() - t0, tolerance=tol,
                                 tolerance_source=tol_src, reason_code="reference_material_retained")
    _ACTIVE_PHASES.set(phases)
    summary, fail = _run_verify_phase(candidate, request, policy, tol, work, staged_receipt, log)
    if fail is not None:
        kind, status, code, message = fail
        return ValidationVerdict(status=status, kind=kind, message=message, verify_status="raised",
                                 seconds=time.perf_counter() - t0, tolerance=tol, tolerance_source=tol_src,
                                 reason_code=code)
    res = _VerifySummary(summary)
    secs = time.perf_counter() - t0
    common = dict(verify_status=res.status, seconds=secs, tolerance=tol, tolerance_source=tol_src,
                  max_abs_error=getattr(res, "max_abs_error", None))
    if res.status == "error":
        kind = classify_candidate_error_message(res.message)
        m = _IR_FAIL_RE.search(res.message or "")
        if m and m.group(1) == "IsolationUnavailableError":
            # the facade could not start its isolating child: an ENVIRONMENT failure -- infrastructure,
            # unscored, the attempt retained for continuation; its own constant reason code (defence in
            # depth: run_verification re-raises the typed refusal, so this branch sees it only from a
            # facade that folded it into an execution-failure result)
            return ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                     message=res.message, reason_code="isolation_unavailable", **common)
        # the facade reported an IR-execution failure ("IR execution FAILED after Ns: Type: ..."):
        # the verdict names the event like every other refusal (target pipeline 77147099 showed
        # this class with reason_code null); the kind is still decided by the exception TYPE only
        return ValidationVerdict(
            status="infrastructure" if kind == FailureKind.INFRASTRUCTURE else "candidate_error",
            kind=kind, message=res.message, reason_code="ir_execution_failed", **common)
    if tol is None:
        return ValidationVerdict(
            status="tolerance_unresolved", kind=FailureKind.INPUT,
            message=(f"tolerance policy unresolved (no request tolerance, no workload "
                     f"tolerance for workload_idx={request.workload_idx}); the candidate "
                     f"executed ({res.status}, max_abs_error={common['max_abs_error']}) "
                     f"but no pass/fail is admitted without an approved policy"),
            **common)
    if res.status == "fail":
        return ValidationVerdict(status="mismatch", kind=FailureKind.UNSUPPORTED,
                                 message=res.message, **common)
    if res.status in ("shape_only", "pass_vacuous"):
        return ValidationVerdict(status="no_proof", kind=FailureKind.UNSUPPORTED,
                                 message=f"{res.status}: {res.message}", **common)
    # res.status == "pass" — bind the EXECUTED reference to the requested source
    # before admitting anything (reference-14)
    binding = _check_reference_binding(res, candidate)
    if not binding["checked"]:
        return ValidationVerdict(status="no_proof", kind=FailureKind.UNSUPPORTED,
                                 message="reference binding unavailable: the verify result carries "
                                         "no execution evidence (reference_sha256); a pass cannot be "
                                         "admitted for the requested source",
                                 reference_binding=binding, reason_code="reference_binding_unavailable",
                                 **common)
    if not binding["bound"]:
        return ValidationVerdict(
            status="adapter_reference_mismatch", kind=FailureKind.PROTOCOL,
            message=(f"adapter_reference_mismatch: the executed reference "
                     f"{str(binding['executed_reference_sha256'])[:12]}… is not the requested source "
                     f"{candidate.source_sha256[:12]}… (unsupported dependencies: "
                     f"{binding['unsupported_dependencies']}); the pass is for a different program "
                     f"and is refused"),
            reference_binding=binding, reason_code="adapter_reference_mismatch", **common)
    trusted = None
    if candidate.adapter_sha256:
        # adapter route: the adapter never supplies both the reference and the
        # assertion identifying it — the dispatcher executes the trusted
        # reference itself (reference-14)
        tr = _trusted_reference_check(candidate, request, policy, tol,
                                      float(getattr(getattr(res, "execution", None), "min_tol", 0.0) or 0.0),
                                      Path(receipt_out).parent, log)
        trusted = tr["detail"]
        if tr["status"] != "ok":
            return ValidationVerdict(status=tr["status"], kind=tr["kind"], message=tr["message"],
                                     reference_binding=binding, trusted_reference=trusted,
                                     reason_code=tr["reason_code"], public_message=tr.get("public_message"),
                                     **common)
    receipt_path = None
    if request.definition_dir is not None and request.dtype:
        # the sealed receipt was emitted by the verify child (where the sealed
        # VerifyResult lives) into a STAGING path; it is admitted only here,
        # after the trusted gate passed, by an atomic rename
        if res.receipt_refused:
            return ValidationVerdict(status="no_proof", kind=FailureKind.UNSUPPORTED,
                                     message=f"receipt refused: {res.receipt_refused}", reference_binding=binding,
                                     trusted_reference=trusted, **common)
        if res.receipt_error:
            # receipt storage/emission fault: infrastructure, unscored; the
            # candidate is retained (failure-17)
            return ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                     message=f"receipt_emission_failed: {res.receipt_error.get('type')}: "
                                             f"{res.receipt_error.get('message')}",
                                     reference_binding=binding, trusted_reference=trusted,
                                     reason_code="receipt_emission_failed", **common)
        # the staged receipt was written by the verify child (a process that ran
        # generated code): it was ingested through the hardened open while the
        # private root was locked; the parent writes its OWN copy from those bytes
        staged = (phases.ingested.get("verify") or {}).get("receipt") or {}
        if not staged.get("ok") or _TRUSTED_SHA256(staged["bytes"]).hexdigest() != res.receipt_sha256:
            return ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                     message="receipt_emission_failed: the verify child produced no admitted receipt "
                                             f"({staged.get('problem') or 'bytes differ from the emitted digest'})",
                                     reference_binding=binding, trusted_reference=trusted,
                                     reason_code="receipt_emission_failed", **common)
        try:
            tmp = Path(str(receipt_out) + f".tmp.{os.getpid()}")
            with _TRUSTED_OPEN(str(tmp), "wb") as fh:
                fh.write(staged["bytes"])
            _TRUSTED_REPLACE(str(tmp), str(receipt_out))
            receipt_path = str(receipt_out)
            try:
                os.unlink(str(staged_receipt))
            except OSError:
                pass
        except Exception as exc:
            return ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                     message=f"receipt_emission_failed: {type(exc).__name__}: {exc}",
                                     reference_binding=binding, trusted_reference=trusted,
                                     reason_code="receipt_emission_failed", **common)
    return ValidationVerdict(status="verified", kind="", message=res.message,
                             receipt_path=receipt_path, reference_binding=binding,
                             trusted_reference=trusted, **common)


# ---------------------------------------------------------------------------
# Dispatch result + core loop
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    status: str
    final_route: str                      # mechanical | pure_skill | pythonic | none
    deterministic: dict
    why: Optional[dict] = None            # deterministic_limitation + fallback_reason
    attempts: list = field(default_factory=list)
    failure_receipts: list = field(default_factory=list)   # append-only
    candidate: Optional[dict] = None
    validation: Optional[dict] = None
    ledger: dict = field(default_factory=dict)
    resume_token: Optional[str] = None    # request dir when awaiting the agent
    receipt_path: Optional[str] = None    # DISPATCH_RECEIPT.json
    source_sha256: str = ""
    # where this row's SOL must come from (mechanical dense / DSL dense /
    # DSL sparse measured-density) and what it needs (receipt, candidate)
    sol_source: Optional[dict] = None
    # a candidate retained in pending-infrastructure validation state
    pending_validation: Optional[dict] = None

    @property
    def verified(self) -> bool:
        return self.status == "agentic_verified"

    def to_dict(self) -> dict:
        return {**asdict(self), "verified": self.verified}


_STATE_FILE = "DISPATCH_STATE.json"
_RECEIPT_FILE = "DISPATCH_RECEIPT.json"
_SCORED_OUTCOMES = frozenset({"mismatch", "no_proof", "candidate_error", FailureKind.UNSUPPORTED})


def _write_json(path: Path, obj) -> Path:
    """Durable write: temp file + rename, so a reader never sees a torn
    receipt and a failed write leaves the previous one intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    return path


_CLOCK_TOLERANCE_SECONDS = 2.0   # epoch vs monotonic delta disagreement allowed across processes


def _proc_starttime(pid: Optional[int] = None) -> Optional[int]:
    """The kernel's process start time (clock ticks since boot, field 22 of
    /proc/<pid>/stat): together with the pid and the boot id it identifies a
    process INCARNATION, which an integer pid alone does not."""
    try:
        text = Path(f"/proc/{pid if pid is not None else 'self'}/stat").read_text()
        rest = text[text.rindex(")") + 2:].split()
        return int(rest[19])
    except Exception:
        return None


def _process_identity() -> dict:
    return {"pid": os.getpid(), "boot_id": _boot_id(), "starttime": _proc_starttime()}


def _boot_id() -> Optional[str]:
    """The kernel boot id (Linux): the monotonic clock is comparable across
    processes only within one boot."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return value or None
    except Exception:
        return None


# numeric-feedback-30 (manager ruling): numeric error statistics never
# enter attempt-visible state (DISPATCH_STATE.json while the dispatch runs);
# they are kept in the private evidence root and surface only in the final
# DISPATCH_RECEIPT.json / the returned result, after the dispatch has ended.
# Round 12 (the review NUMERIC-FEEDBACK-30/C): the same holds for every piece of
# METADATA derived from the unredacted comparison report or a private verdict
# -- the ingest size/link count/time bound, durations, numerals in problem or
# stderr text.  Attempt-visible records carry a fixed-shape ingest view
# (``_public_ingest_view``) and no duration; the full validation record is
# stashed privately (``_stash_numeric``) and restored by ``_with_numeric``.
_NUMERIC_KEYS = frozenset({"max_abs_error", "per_output_max_abs_error"})
_NUMERIC_TEXT_KEYS = frozenset({"message", "reason", "numerical_validation", "error"})
# free-text keys whose numerals are redacted in attempt-visible records: the verdict texts
# above plus the ingest problem, the phase result problem and a failed phase's stderr tail
_REDACT_TEXT_KEYS = _NUMERIC_TEXT_KEYS | frozenset({"problem", "result_problem", "stderr_tail"})
# a numeral is redacted whole, including one that carries a short unit suffix ("0.3s", "2ms",
# "5%": round 12 -- the previous lookahead left the fraction digit of "0.3s" in place); digits
# inside identifiers and digests (sha256, fp32, 62e924be…) are untouched
_NUMERAL_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
                         r"(?:(?![\w])|(?=[a-zA-Z%]{1,3}(?![\w])))")
# round 13: a non-finite statistic spelled as a WORD ("max_abs_error=inf", "nan") is a numeral too
_NONFINITE_WORD_RE = re.compile(r"(?<![\w.])[-+]?(?:inf|nan)(?![\w])", re.IGNORECASE)
_NUMERIC_STORE = "numeric-statistics.json"
_PRIVATE_MARK = "_private_numeric_id"
# the attempt-visible representation of one result ingest: this shape and nothing else
_PUBLIC_INGEST_KEYS = ("label", "ok", "kind", "cap_bytes", "timed_out", "problem", "evidence")
_INGEST_EVIDENCE_NOTE = ("size, link count and time bound are private until the dispatch ends; "
                         "terminal DISPATCH_RECEIPT.json only")


def _redact_numerals(text: str) -> str:
    return _NONFINITE_WORD_RE.sub("<n>", _NUMERAL_RE.sub("<n>", text))


def _public_ingest_view(rec: dict) -> dict:
    """The attempt-visible representation of one ``_ingest_result_file`` record
    (round 12, the review NUMERIC-FEEDBACK-30/C): a FIXED shape -- ok, kind, the
    documented cap, timed_out and the problem text with numerals redacted -- and
    no number that depends on the content of what was ingested.  ``size`` is
    the serialized length of the UNREDACTED report (for the compare phase: a
    function of the hidden statistics), ``nlink`` and ``seconds_bound`` are
    inode and timing facts; all three stay in the private root and return in
    the terminal receipt.  ``cap_bytes`` is the documented cap of the label."""
    problem = rec.get("problem")
    return {"label": rec.get("label"), "ok": bool(rec.get("ok")), "kind": rec.get("kind"),
            "cap_bytes": rec.get("cap_bytes"), "timed_out": bool(rec.get("timed_out")),
            "problem": _redact_numerals(problem) if isinstance(problem, str) else None,
            "evidence": _INGEST_EVIDENCE_NOTE}


# Round 13 (the review NUMERIC-FEEDBACK-30/D, ruling a): the attempt-visible record is a
# CONSTANT-SHAPE document.  Fixed key sets; a key whose value is withheld is present and
# None; nothing optional whose presence depends on content.
_WITHHELD_VERDICT_KEYS = _NUMERIC_KEYS | frozenset({"vacuous_outputs"})   # statistics + which outputs are vacuous
_PUBLIC_VERDICT_KEYS = ("success", "error", "max_abs_error", "vacuous_outputs", "compared_outputs",
                        "outputs_pass", "per_output_max_abs_error", "tolerance")
_PUBLIC_TOLERANCE_KEYS = ("rtol", "atol", "trusted_dtype_floor", "rtol_effective", "atol_effective")
_KERNEL_ID_KEYS = frozenset({"pid", "sid", "pgid", "leader_pid"})       # kernel-assigned, variable width
_FS_ID_KEYS = frozenset({"ino", "ctime_ns", "mtime_ns"})                 # inode facts of parent-owned files
_SCHEMA_COUNTERS_NOTE = ("allowlist parse counters are private until the dispatch ends; "
                         "terminal DISPATCH_RECEIPT.json only")
_UNVERIFIABLE_PUBLIC_MESSAGE = ("trusted_verdict_unverifiable: the compare phase did not yield a verdict the "
                                "parent accepts; which check refused it is recorded in the terminal "
                                "DISPATCH_RECEIPT.json only")


def _public_report_schema_view(stats: dict) -> dict:
    """The attempt-visible view of a phase report's allowlist-parse counters: the same
    five keys as the private record, the counters None (round 13 -- a dropped
    non-finite statistic incremented the public counter and identified the hidden value)."""
    return {"allowlisted": True, "unknown_or_mistyped_dropped": None, "truncated": None, "parsed": None,
            "ingest_before_unlock": bool(stats.get("ingest_before_unlock", True)), "counters": _SCHEMA_COUNTERS_NOTE}


def _public_verdict_view(verdict: dict) -> dict:
    """The comparison verdict as every public record shows it: exactly the eight
    keys of ``_PUBLIC_VERDICT_KEYS`` in that order -- statistics and
    ``vacuous_outputs`` None, ``error`` with numerals redacted, ``tolerance`` with
    exactly its five keys (None when absent) -- and nothing else (no artifact
    digests).  Keys the parser dropped are present and None, so the shape does not
    depend on the content (round 13).  The public verdict digest is computed over
    THIS view (round 11): a digest over the full verdict would commit to the
    withheld statistics and a small candidate set would recover them."""
    out = {}
    for k in _PUBLIC_VERDICT_KEYS:
        val = verdict.get(k)
        if k in _WITHHELD_VERDICT_KEYS:
            out[k] = None
        elif k == "error":
            out[k] = _redact_numerals(val) if isinstance(val, str) else None
        elif k == "tolerance":
            tol = val if isinstance(val, dict) else {}
            out[k] = {t: (tol[t] if isinstance(tol.get(t), (int, float)) and not isinstance(tol.get(t), bool) else None)
                      for t in _PUBLIC_TOLERANCE_KEYS}
        else:
            out[k] = val
    return out


def _attempt_visible(obj, private_id: str, public_message: Optional[str] = None):
    """A copy of a validation record fit for attempt-visible state: numeric
    statistic keys emptied, numerals in free-text fields redacted (verdict
    texts, ingest/result problems, stderr tails), every ``result_ingest``
    record replaced by its fixed-shape public view, every duration
    (``seconds``) withheld, and a mark that names (opaquely) where the full
    record lives.  Round 12 (the review NUMERIC-FEEDBACK-30/C): the attempt-visible
    record is a function of the public representation only -- no size, count,
    hash or timing derived from the unredacted comparison report.  Round 13
    (the review NUMERIC-FEEDBACK-30/D): a CONSTANT-SHAPE document -- every
    ``verdict`` is the fixed eight-key public view, every ``report_schema`` the
    fixed nullable view (no parse counters), ``vacuous_outputs`` is withheld,
    kernel ids (pid/sid/pgid/leader_pid) and inode facts (ino/ctime_ns/mtime_ns)
    are withheld, and a verdict carrying ``public_message`` shows exactly that
    constant text as its message.  The full record is restored in the terminal
    receipt."""
    def walk(x, key=None):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in _WITHHELD_VERDICT_KEYS or k in _KERNEL_ID_KEYS or k in _FS_ID_KEYS:
                    out[k] = None
                elif k == "verdict" and isinstance(v, dict):
                    out[k] = _public_verdict_view(v)
                elif k == "report_schema" and isinstance(v, dict):
                    out[k] = _public_report_schema_view(v)
                elif k == "verify" and isinstance(v, dict):
                    inner = walk(v, k)
                    for withheld in ("max_abs_error", "vacuous_outputs"):
                        inner.setdefault(withheld, None)       # present even when the child omitted it
                    out[k] = inner
                elif k in _REDACT_TEXT_KEYS and isinstance(v, str):
                    out[k] = _redact_numerals(v)
                elif k == "result_ingest" and isinstance(v, dict):
                    out[k] = {label: (_public_ingest_view(rec) if isinstance(rec, dict) else None)
                              for label, rec in v.items()}
                elif k == "seconds" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = None          # durations: timing that may scale with content is withheld
                else:
                    out[k] = walk(v, k)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    vis = walk(obj)
    if isinstance(vis, dict):
        if public_message is not None and "message" in vis:
            vis["message"] = public_message
        vis[_PRIVATE_MARK] = private_id
        vis["numeric_statistics"] = "private until the dispatch ends; final DISPATCH_RECEIPT.json only"
    return vis


class _Dispatch:
    """One dispatch, resumable.  All mutable state lives in ``self.state``
    and is persisted to ``DISPATCH_STATE.json`` after every step."""

    def __init__(self, request: DispatchRequest, policy: DispatchPolicy, provider,
                 deterministic_stage, validator, reservation, log):
        self.request, self.policy, self.provider = request, policy, provider
        self.deterministic_stage = deterministic_stage or default_deterministic_stage
        self.validator = validator or default_validator
        self.reservation, self.log = reservation, log or (lambda *_: None)
        self.out_dir = request.out_dir
        self.state: dict = {}
        self._private_root_path: Optional[Path] = None   # in memory only; never written to a public record
        self._numeric: dict = {}                          # private id -> full validation record (+ message)

    # -- persistence -----------------------------------------------------

    def save(self):
        self.state["updated_utc"] = _utc()
        # round 13 (the review PUBLIC-PROPERTY-SCOPE-31 c): the elapsed wall is a clock reading taken
        # after whatever comparison work preceded this save, and the state file is attempt-visible
        # at all times -- WITHHELD here; the terminal receipt's ledger publishes the figure.  The
        # deadline (``deadline_utc``, fixed at dispatch start) and the caps are policy and stay.
        self.state["wall_seconds_elapsed"] = None
        _write_json(self.out_dir / _STATE_FILE, self._public(self.state))

    def _public(self, obj):
        """Safety net (reference-27 fix c): whatever a record carried, nothing
        persisted or returned names the private root's path.  Returns a redacted
        COPY; the in-memory state keeps its object identity (records held by the
        running step stay attached)."""
        root = self._private_root_path
        if root is None:
            return obj
        return _redact_private_paths(obj, root, _private_root_id(root))

    # -- numeric statistics: private until the dispatch ends ---------------

    def _stash_numeric(self, full: dict, message: str) -> str:
        """Keep the full validation record (with its numeric statistics) in
        memory and in the private evidence root; return its opaque id."""
        key = secrets.token_hex(8)
        entry = {"validation": full, "message": message, "utc": _utc()}
        self._numeric[key] = entry
        root = self._private_root()
        if root is not None:
            path = Path(root) / _NUMERIC_STORE
            try:
                store = _TRUSTED_JSON_LOADS(path.read_text()) if path.is_file() else {}
            except Exception:
                store = {}
            if not isinstance(store, dict):
                store = {}
            store[key] = entry
            try:
                _write_json(path, store)
            except OSError:
                pass
        return key

    def _numeric_entry(self, key: str) -> Optional[dict]:
        if key in self._numeric:
            return self._numeric[key]
        root = self._private_root()
        if root is None:
            return None
        try:
            store = _TRUSTED_JSON_LOADS((Path(root) / _NUMERIC_STORE).read_text())
            entry = store.get(key) if isinstance(store, dict) else None
        except Exception:
            entry = None
        if isinstance(entry, dict):
            self._numeric[key] = entry
        return entry

    def _with_numeric(self, obj):
        """The final-receipt view: every attempt-visible validation record is
        replaced by its full private record (the dispatch has ended)."""
        def walk(x):
            if isinstance(x, dict):
                key = x.get(_PRIVATE_MARK)
                if isinstance(key, str):
                    entry = self._numeric_entry(key)
                    if entry is not None and isinstance(entry.get("validation"), dict):
                        return entry["validation"]          # the full record, as the validator produced it
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list):
                return [walk(v) for v in x]
            return x
        out = walk(obj)
        # failure receipts: the redacted reason gets its full text back
        if isinstance(out, dict):
            for rec in out.get("failure_receipts") or []:
                if not isinstance(rec, dict):
                    continue
                key = rec.get("reason_private_id")
                if isinstance(key, str):
                    entry = self._numeric_entry(key)
                    if entry is not None and isinstance(entry.get("message"), str) and rec.get("reason_redacted"):
                        rec["reason"] = entry["message"]
                    rec.pop("reason_redacted", None)
                    rec.pop("reason_private_id", None)
            pv = out.get("pending_validation")
            if isinstance(pv, dict) and isinstance(pv.get("reason_private_id"), str):
                entry = self._numeric_entry(pv["reason_private_id"])
                if entry is not None and isinstance(entry.get("message"), str):
                    pv["reason"] = entry["message"]
                pv.pop("reason_private_id", None)
        return out

    def _resumable(self, status: str) -> bool:
        """The outcomes ``resume()`` continues from: awaiting the agent, or an
        infrastructure failure with a retained candidate.  Not the end of the
        dispatch: another attempt may still read what this result publishes."""
        return status == "awaiting_agent" or \
            (status == "agentic_infrastructure_error" and bool(self.state.get("pending_validation")))

    def result(self, status: str, final_route: str, **kw) -> DispatchResult:
        st = self.state
        st["status"] = status
        st["final_route"] = final_route
        snapshot = json.loads(json.dumps(
            {"attempts": list(st.get("attempts", [])), "failure_receipts": list(st.get("failure_receipts", [])),
             "validation": st.get("validation"), "pending_validation": st.get("pending_validation")}, default=str))
        # the numeric statistics are restored ONLY on terminal completion (final
        # verdict, exhausted attempts, hard failure).  A RESUMABLE outcome (round 11,
        # the review NUMERIC-FEEDBACK-30/A) -- awaiting the agent, or an infrastructure
        # failure whose retained candidate resume re-validates -- publishes the same
        # attempt-visible view as DISPATCH_STATE.json, in the receipt and in the
        # returned result alike; the next attempt can read both.
        resumable = self._resumable(status)
        full = self._public(snapshot if resumable else self._with_numeric(snapshot))
        res = DispatchResult(
            status=status, final_route=final_route, deterministic=st["deterministic"],
            why=st.get("why"), attempts=full["attempts"],
            failure_receipts=full["failure_receipts"],
            candidate=st.get("candidate"), validation=full["validation"],
            ledger=self.ledger(withhold_elapsed=resumable), resume_token=kw.get("resume_token"),
            source_sha256=st.get("source_sha256", ""),
            sol_source=self._sol_source(status),
            pending_validation=full["pending_validation"])
        rec = self._public({"schema": SCHEMA, **res.to_dict(),
                            "numeric_statistics": ("withheld: resumable outcome (restored in the terminal receipt)"
                                                   if resumable else "final"),
                            "request": st["request"], "policy": st["policy"], "utc": _utc()})
        res.receipt_path = str(_write_json(self.out_dir / _RECEIPT_FILE, rec))
        st["receipt_path"] = res.receipt_path
        self.save()
        return res

    # -- time ------------------------------------------------------------

    def _clock_reading(self) -> tuple:
        """(elapsed_seconds, continuity_failure_or_None) — wall-15.

        Persisted at start: ``started_epoch``, ``started_monotonic`` and the
        process INCARNATION ``started_process`` = (boot id, pid, kernel start
        time).  Continuity is established only when ALL of these are present
        on both sides: a missing boot identity, pid, start time or monotonic
        start is ``clock_continuity_unknown`` (infrastructure, unscored); the
        identity is checked at every RESUME (state reloaded from disk) — the
        object that recorded the start itself never crossed a persistence
        boundary and needs none.  A monotonic reading before the persisted
        start is refused on EVERY branch.  Same incarnation under the same boot: the monotonic delta is
        primary.  Another process under the SAME boot id: the monotonic clock
        is system-wide since boot, so its delta is still valid and must agree
        with the epoch delta within a small tolerance; a different boot id
        means the deadline cannot be carried across the reboot.  A wall clock
        before the persisted start is a rollback.  The elapsed figure is NEVER
        smaller than the epoch delta."""
        st = self.state
        started = st.get("started_epoch")
        if started is None:
            return 0.0, None
        now = time.time()
        epoch_elapsed = now - float(started)
        if epoch_elapsed < -1e-9:
            return 0.0, (f"clock discontinuity: wall clock {now:.3f} is before the persisted start "
                         f"{float(started):.3f} (rollback {-epoch_elapsed:.3f}s)")
        mono_start = st.get("started_monotonic")
        if mono_start is None:
            return max(0.0, epoch_elapsed), (
                "clock_continuity_unknown: no persisted monotonic start; the deadline is never "
                "evaluated from the wall clock alone")
        if getattr(self, "_recorded_start", False):
            # this very object took the persisted readings and never left memory:
            # no persistence boundary was crossed, so the monotonic clock is
            # continuous by construction and no incarnation identity is needed
            # (identity is required at EVERY resume, which reloads the state);
            # a reading before the recorded start is still refused
            mono_elapsed = time.monotonic() - float(mono_start)
            if mono_elapsed < -1e-9:
                return max(0.0, epoch_elapsed), (
                    f"clock_continuity_unknown: monotonic reading {mono_elapsed:.3f}s before the "
                    f"persisted start within the dispatching process")
            st["clock_continuity"] = "same_process_monotonic"
            return max(0.0, epoch_elapsed, mono_elapsed), None
        then = st.get("started_process") or {"pid": st.get("started_pid"), "boot_id": st.get("started_boot_id"),
                                              "starttime": None}
        now_id = _process_identity()
        missing = sorted({k for k in ("boot_id", "pid", "starttime") if then.get(k) is None}
                         | {k for k in ("boot_id", "pid", "starttime") if now_id.get(k) is None})
        if missing:
            what = "boot identity unavailable" if "boot_id" in missing else "process identity incomplete"
            return max(0.0, epoch_elapsed), (
                f"clock_continuity_unknown: {what} ({', '.join(missing)} missing; persisted "
                f"{then.get('boot_id')}/{then.get('pid')}/{then.get('starttime')} -> now "
                f"{now_id.get('boot_id')}/{now_id.get('pid')}/{now_id.get('starttime')}); continuity requires "
                f"boot id, pid, start time and the persisted monotonic start on both sides, and the "
                f"deadline is not evaluated from the wall clock alone")
        if then["boot_id"] != now_id["boot_id"]:
            return max(0.0, epoch_elapsed), (
                f"clock_continuity_unknown: boot id changed since the dispatch started "
                f"({then['boot_id']} -> {now_id['boot_id']}); the monotonic deadline cannot be carried "
                f"across a reboot")
        mono_elapsed = time.monotonic() - float(mono_start)
        if mono_elapsed < -1e-9:
            return max(0.0, epoch_elapsed), (
                f"clock_continuity_unknown: monotonic reading {mono_elapsed:.3f}s before the "
                f"persisted start under the same boot id")
        if then["pid"] == now_id["pid"] and then["starttime"] == now_id["starttime"]:
            continuity = "same_process_monotonic"
        else:
            continuity = "same_boot_monotonic"
            tol = _CLOCK_TOLERANCE_SECONDS + 0.02 * max(mono_elapsed, epoch_elapsed)
            if abs(mono_elapsed - epoch_elapsed) > tol:
                return max(epoch_elapsed, mono_elapsed, 0.0), (
                    f"clock_continuity_unknown: epoch delta {epoch_elapsed:.3f}s and monotonic "
                    f"delta {mono_elapsed:.3f}s disagree beyond {tol:.3f}s across processes")
        st["clock_continuity"] = continuity
        return max(0.0, epoch_elapsed, mono_elapsed), None

    def _clock_discontinuity(self) -> Optional[str]:
        return self._clock_reading()[1]

    def elapsed(self) -> float:
        """Elapsed total wall: never below the epoch delta (see _clock_reading)."""
        return self._clock_reading()[0]

    def remaining(self) -> Optional[float]:
        wall = self.policy.budget.wall_seconds
        if wall is None:
            return None
        return float(wall) - self.elapsed()

    def over_deadline(self) -> Optional[str]:
        disc = self._clock_discontinuity()
        if disc and self.policy.budget.wall_seconds is not None:
            self.state["clock_failure"] = ("clock_continuity_unknown" if "clock_continuity_unknown" in disc
                                           else "clock_discontinuity")
            return f"total wall cap {self.policy.budget.wall_seconds:g}s cannot be evaluated: {disc}"
        self.state.pop("clock_failure", None)
        rem = self.remaining()
        if rem is not None and rem <= 0:
            return (f"total wall cap {self.policy.budget.wall_seconds:g}s exhausted "
                    f"({self.elapsed():.3f}s elapsed since {self.state.get('started_utc')}, "
                    f"deadline {self.state.get('deadline_utc')}; monotonic-primary)")
        return None

    def _over_outcome(self) -> tuple:
        """(kind, status, reason_code) for a deadline stop: a cap that is
        genuinely exhausted is EXHAUSTED; a deadline that cannot be evaluated
        (clock continuity) is INFRASTRUCTURE, unscored."""
        code = self.state.get("clock_failure")
        if code:
            return FailureKind.INFRASTRUCTURE, "agentic_infrastructure_error", code
        return FailureKind.EXHAUSTED, "agentic_exhausted", "wall_cap_exhausted"

    def _bounded_policy(self) -> DispatchPolicy:
        """The policy with an EXISTING verify_timeout tightened to the
        remaining deadline.  A timeout makes the verifier pre-empt the DSL
        in a forked executor, which is unsafe in a process that already
        initialised OpenMP (the documented fork-guard hazard), so the
        dispatcher never introduces one on its own: without a caller-set
        verify_timeout the deadline is enforced post hoc — a validation that
        finishes late is rejected as infrastructure, unscored."""
        rem = self.remaining()
        vt = self.policy.verify_timeout
        if rem is None or vt is None:
            return self.policy
        return dataclasses.replace(self.policy, verify_timeout=max(1, min(vt, int(math.ceil(rem)))))

    # -- receipts / aggregates ------------------------------------------

    def _sol_source(self, status: str) -> Optional[dict]:
        det = self.state.get("deterministic") or {}
        if status == "mechanical_ok":
            return {"source": SOL_SOURCE_MECHANICAL, "status": "available",
                    "perf": (det.get("artifacts") or {}).get("perf"),
                    "numerical_validation": det.get("numerical_validation")}
        if not self.state.get("why"):
            return None
        sparse = bool(det.get("expected_limitation")) or bool(det.get("sparsity_family"))
        src = SOL_SOURCE_DSL_SPARSE if sparse else SOL_SOURCE_DSL_DENSE
        val = self.state.get("validation") or {}
        receipt = val.get("receipt_path")
        if status == "agentic_verified" and receipt:
            sol_status, blocker = "ready", None
        elif status == "agentic_verified":
            sol_status = "blocked"
            blocker = ("missing receipt inputs: the numerical pass was not bound into a "
                       "sealed receipt (definition_dir and dtype are required); "
                       "sol(strict=True) refuses a missing receipt")
        else:
            sol_status, blocker = "blocked", status
        out = {"source": src, "status": sol_status, "blocker": blocker, "receipt": receipt,
               "candidate": (self.state.get("candidate") or {}).get("dsl_path"),
               "tolerance": val.get("tolerance"), "tolerance_source": val.get("tolerance_source"),
               "note": ("run solar.translate.sol(strict=True, receipt=..., "
                        "source_path=..., measure_density=True) on the verified "
                        "candidate; the dispatcher does not run SOL"
                        if sparse else
                        "run solar.translate.sol(strict=True, receipt=..., "
                        "source_path=...) on the verified candidate; the "
                        "dispatcher does not run SOL")}
        if sparse:
            out["sparsity_family"] = det.get("sparsity_family")
            out["sparsity_note"] = ("op-name membership / declaration is a routing "
                                    "heuristic, not proof of sparsity; measured density "
                                    "must be established by the DSL-route analysis")
        return out

    def ledger(self, withhold_elapsed: bool = False) -> dict:
        st = self.state
        attempts = st.get("attempts", [])
        model_attempts = [a for a in attempts if a.get("counted")]
        verified = sum(1 for a in attempts if a.get("outcome") == "verified")
        scorable = sum(1 for a in attempts if a.get("outcome") in _SCORED_OUTCOMES)
        pending = sum(1 for a in attempts if a.get("outcome") == "pending_infrastructure")
        unresolved = sum(1 for a in model_attempts
                         if a.get("outcome") not in _SCORED_OUTCOMES | {"verified", "pending_infrastructure"})
        return {
            # model attempts (generations) vs scorable outcomes vs infrastructure
            "attempts_planned": self.policy.budget.max_attempts,
            "model_attempts_used": len(model_attempts),
            "attempts_used": len(model_attempts),    # legacy name, same number
            "max_attempts": self.policy.budget.max_attempts,
            "scorable_outcomes": scorable + verified,
            "verified": verified,
            "pending_infrastructure": pending,
            "unresolved": unresolved,
            "validation_retries_used": st.get("validation_retries_used", 0),
            "max_validation_retries": self.policy.budget.max_validation_retries,
            # round 13: a resumable result (another attempt can read it) withholds the elapsed wall
            "wall_seconds_elapsed": None if withhold_elapsed else round(self.elapsed(), 3),
            "wall_seconds_cap": self.policy.budget.wall_seconds,
            "started_utc": st.get("started_utc"), "deadline_utc": st.get("deadline_utc"),
            "api_requests_used": st.get("api_requests_used", 0),
            "max_requests": self.policy.budget.max_requests,
            "cost_usd_used": st.get("cost_usd_used", 0.0),
            "max_cost_usd": self.policy.budget.max_cost_usd,
            "cost_per_request_usd": self.policy.cost_per_request_usd,
            "reservation": (self.reservation.to_dict()
                            if hasattr(self.reservation, "to_dict") else
                            (None if self.reservation is None else str(self.reservation))),
        }

    def fail(self, kind: str, stage: str, reason: str, attempt: int = 0, **extra) -> dict:
        rec = _receipt(kind, stage, reason, attempt=attempt, **extra)
        self.state.setdefault("failure_receipts", []).append(rec)
        self.log(f"[dispatch] {stage} {kind}: {reason}")
        self.save()
        return rec

    def _account(self, extra: dict):
        self.state["api_requests_used"] += int(extra.get("api_requests", 0) or 0)
        self.state["cost_usd_used"] += float(extra.get("cost_usd", 0.0) or 0.0)

    # -- steps -----------------------------------------------------------

    def start(self) -> DispatchResult:
        try:
            return self._start()
        except Exception as exc:  # failure-17: never escape without a durable outcome
            return self._crash("start", exc)

    def resume(self) -> DispatchResult:
        try:
            return self._resume()
        except NothingToResume:
            raise          # precondition only: the caller's contract error
        except Exception as exc:
            # a ValueError (or anything else) raised DURING validation/emission
            # is operational: a FRESH durable unscored outcome replaces the
            # obsolete receipt (failure-17)
            return self._crash("resume", exc)

    def _crash(self, where: str, exc: BaseException) -> DispatchResult:
        """Durable, unscored infrastructure outcome for an unexpected dispatcher
        exception; the candidate (if any) is retained in state.  Written via
        the atomic path; if even that fails a minimal receipt is attempted."""
        kind = classify_exception(exc)
        kind = FailureKind.INFRASTRUCTURE if kind in (FailureKind.UNSUPPORTED, FailureKind.INPUT) else kind
        rec = _receipt(kind, f"dispatcher_{where}",
                       f"dispatcher raised {type(exc).__name__}: {exc}; outcome retained unscored",
                       exception_type=type(exc).__name__, traceback=traceback.format_exc()[-3000:],
                       reason_code="dispatcher_exception")
        st = self.state if isinstance(self.state, dict) else {}
        st.setdefault("failure_receipts", []).append(rec)
        st.setdefault("attempts", [])
        st.setdefault("deterministic", {"status": "unknown", "reasons": []})
        try:
            return self.result("dispatcher_error", st.get("final_route", "none"))
        except Exception as exc2:
            minimal = {"schema": SCHEMA, "status": "dispatcher_error", "failure_receipts": [rec],
                       "second_failure": f"{type(exc2).__name__}: {exc2}", "utc": _utc(),
                       "candidate": st.get("candidate"), "pending_validation": st.get("pending_validation")}
            try:
                _write_json(self.out_dir / _RECEIPT_FILE, minimal)
            except Exception:
                pass
            return DispatchResult(status="dispatcher_error", final_route="none",
                                  deterministic=st.get("deterministic") or {},
                                  failure_receipts=[rec], candidate=st.get("candidate"),
                                  pending_validation=st.get("pending_validation"),
                                  receipt_path=str(self.out_dir / _RECEIPT_FILE))

    def _start(self) -> DispatchResult:
        req, pol = self.request, self.policy
        self.out_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        wall = pol.budget.wall_seconds
        self.state = {
            "schema": SCHEMA, "created_utc": _utc(), "started_epoch": now,
            "started_monotonic": time.monotonic(), "started_pid": os.getpid(),
            "started_boot_id": _boot_id(), "started_process": _process_identity(), "started_utc": _utc(),
            "deadline_epoch": None if wall is None else now + float(wall),
            "deadline_utc": None if wall is None else time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + float(wall))),
            "request": {**{k: (str(v) if isinstance(v, Path) else v)
                           for k, v in asdict(req).items() if k != "deterministic_validator"},
                        "known_candidates": [str(p) for p in req.known_candidates]},
            "policy": asdict(pol), "attempts": [], "failure_receipts": [],
            "api_requests_used": 0, "cost_usd_used": 0.0, "validation_retries_used": 0,
        }
        self._recorded_start = True       # see _clock_reading
        if not req.source_path.is_file():
            self.state["deterministic"] = DeterministicOutcome(
                status="input_error", reasons=[f"source file not found: {req.source_path}"]).to_dict()
            self.fail(FailureKind.INPUT, "input", f"source file not found: {req.source_path}")
            return self.result("input_error", "none")
        self.state["source_sha256"] = _sha256(req.source_path)

        # 1. deterministic route first — bounded by the deadline, never escaping
        try:
            try:
                det = self.deterministic_stage(req, pol, self.out_dir / "deterministic", self.log,
                                               deadline_seconds=self.remaining())
            except TypeError:
                det = self.deterministic_stage(req, pol, self.out_dir / "deterministic", self.log)
        except Exception as exc:
            kind = classify_exception(exc)
            det = DeterministicOutcome(
                status="infrastructure_error" if kind != FailureKind.INPUT else "input_error",
                reasons=[f"deterministic stage raised {type(exc).__name__}: {exc}"],
                receipt=_receipt(FailureKind.INFRASTRUCTURE if kind != FailureKind.INPUT else kind,
                                 "deterministic", f"stage raised {type(exc).__name__}: {exc}",
                                 exception_type=type(exc).__name__,
                                 traceback=traceback.format_exc()[-2000:], route="mechanical"))
        det = _apply_declared_family(det, req)
        self.state["deterministic"] = det.to_dict()
        if det.receipt:
            self.state["failure_receipts"].append(det.receipt)   # retained verbatim
        self.save()
        late = self.over_deadline()
        if late:
            # a deterministic outcome that arrives after the deadline is not
            # admitted, success included (wall-15): infrastructure, unscored
            self.fail(FailureKind.INFRASTRUCTURE, "deterministic",
                      f"late result: the deterministic stage finished after the total wall deadline "
                      f"(status {det.status}); not admitted, retained unscored ({late})")
            return self.result("deterministic_infrastructure_error", "none")
        if det.status == "ok":
            return self.result("mechanical_ok", "mechanical")
        if det.status == "input_error":
            return self.result("input_error", "none")
        if det.status == "infrastructure_error":
            # not eligible: a host/stage fault is retained for retry, never hidden by a fallback
            return self.result("deterministic_infrastructure_error", "none")

        # 2. eligible -> why + bounded agentic
        assert det.eligible_for_fallback, det.status
        self.state["why"] = {
            "deterministic_limitation": list(det.reasons),
            "deterministic_status": det.status,
            "expected_limitation": bool(det.expected_limitation),
            "sparsity_family": dict(det.sparsity_family or {}),
            "fallback_reason": (
                (f"data-dependent sparsity family {det.sparsity_family} (routing heuristic / "
                 f"declaration, not proof of sparsity): agentic-by-default (expected "
                 f"limitation, not a defect); "
                 if det.expected_limitation else
                 f"deterministic route {det.status}; ")
                + f"policy selects v2 agentic route {pol.route!r} under caps "
                f"attempts={pol.budget.max_attempts}, wall={pol.budget.wall_seconds}s, "
                f"requests={pol.budget.max_requests}, cost=${pol.budget.max_cost_usd}"),
            "sol_source": SOL_SOURCE_DSL_SPARSE if det.expected_limitation else SOL_SOURCE_DSL_DENSE,
        }
        self.save()
        if not pol.allow_agentic:
            self.fail(FailureKind.NO_PROVIDER, "policy",
                      "agentic fallback disabled by policy (allow_agentic=False)")
            return self.result("deterministic_failed_no_fallback", "none")
        if self.provider is None:
            self.fail(FailureKind.NO_PROVIDER, "provider",
                      f"no provider registered for route {pol.route!r}")
            return self.result("no_provider", "none")
        if getattr(self.provider, "route", None) != pol.route:
            self.fail(FailureKind.NO_PROVIDER, "provider",
                      f"provider {self.provider.name!r} serves route "
                      f"{getattr(self.provider, 'route', None)!r}, policy selected {pol.route!r}")
            return self.result("no_provider", "none")
        unavailable = self.provider.availability(req)
        if unavailable:
            self.fail(FailureKind.NO_PROVIDER, "provider", unavailable)
            return self.result("no_provider", "none")
        if self.provider.calls_external_api:
            if self.reservation is None:
                self.fail(FailureKind.NO_BUDGET, "budget",
                          "route calls an external API but no ledger reservation handle was passed")
                return self.result("no_budget", "none")
            if getattr(self.reservation, "remaining_requests", 1) < 1:
                self.fail(FailureKind.NO_BUDGET, "budget",
                          "reservation has no remaining requests")
                return self.result("no_budget", "none")
            if getattr(self.reservation, "remaining_usd", 1.0) <= 0.0:
                self.fail(FailureKind.NO_BUDGET, "budget",
                          "reservation has no remaining funds; a zero-dollar reservation "
                          "is not spend authority")
                return self.result("no_budget", "none")
        return self.run_attempts()

    def _requests_exceeded(self) -> Optional[str]:
        b = self.policy.budget
        if b.max_requests is not None and self.state["api_requests_used"] >= b.max_requests:
            return f"request cap {b.max_requests} exhausted ({self.state['api_requests_used']} used)"
        if b.max_cost_usd is not None and self.state["cost_usd_used"] >= b.max_cost_usd - 1e-12:
            return f"cost cap ${b.max_cost_usd} exhausted (${self.state['cost_usd_used']:.4f} used)"
        if self.provider.calls_external_api and self.reservation is not None:
            if getattr(self.reservation, "remaining_requests", 1) < 1:
                return "reservation has no remaining requests"
            if getattr(self.reservation, "remaining_usd", 1.0) <= 0.0:
                return "reservation has no remaining funds"
        return None

    def _make_request(self, attempt: int) -> tuple:
        req, pol = self.request, self.policy
        attempt_dir = self.out_dir / "agentic" / f"attempt_{attempt:02d}"
        dsl, adapter = _candidate_paths(req, attempt_dir)
        prior = set()
        for p in list(req.known_candidates) + [dsl]:
            p = Path(p)
            if p.is_file():
                prior.add(_sha256(p))
        for a in self.state["attempts"]:
            if a.get("candidate_sha256"):
                prior.add(a["candidate_sha256"])
        b = pol.budget
        rid = hashlib.sha256(
            f"{self.state['source_sha256']}:{self.state['created_utc']}:{attempt}".encode()
        ).hexdigest()[:16]
        tol, tol_src = resolve_tolerance(req)
        treq = TranslationRequest(
            request_id=rid, route=pol.route, source_path=str(req.source_path.resolve()),
            source_sha256=self.state["source_sha256"], source_type=req.source_type,
            definition_dir=None if req.definition_dir is None else str(req.definition_dir.resolve()),
            workload_idx=req.workload_idx, dtype=req.dtype, original_axes=req.original_axes,
            attempt=attempt, max_attempts=b.max_attempts,
            # round 13 (the review PUBLIC-PROPERTY-SCOPE-31 c): no remaining-wall READING in the request
            # (it is taken after the previous attempt's comparison); the absolute deadline and the cap
            # are policy, fixed at dispatch start -- the agent and the providers derive the remaining
            # time from deadline_utc with their own clock (``_caps_remaining``)
            caps={"wall_seconds_remaining": None, "wall_seconds_cap": b.wall_seconds,
                  "deadline_utc": self.state.get("deadline_utc"),
                  "max_requests": b.max_requests,
                  "max_requests_remaining": (None if b.max_requests is None else
                                             max(0, b.max_requests - self.state["api_requests_used"])),
                  "max_cost_usd": b.max_cost_usd,
                  "max_cost_usd_remaining": (None if b.max_cost_usd is None else
                                             max(0.0, b.max_cost_usd - self.state["cost_usd_used"])),
                  "cost_per_request_usd": pol.cost_per_request_usd,
                  "max_transport_requests_per_attempt": pol.max_transport_requests_per_attempt,
                  "verify_timeout": pol.verify_timeout, "mem_budget_gb": pol.mem_budget_gb},
            candidate_path=str(dsl), adapter_path=None if adapter is None else str(adapter),
            prior_candidate_sha256s=sorted(prior),
            deterministic_failure=dict(self.state.get("why") or {}),
            created_utc=_utc(), tolerance=tol, tolerance_source=tol_src)
        return treq, attempt_dir

    def _late(self, stage: str, attempt: int, request_id: str,
              record: Optional[dict] = None) -> Optional[DispatchResult]:
        """Reject work that finished after the deadline: infrastructure,
        unscored, retained (wall-2).  The attempt record's outcome is set
        BEFORE the result is built (the result is a copy of the state)."""
        over = self.over_deadline()
        if not over:
            return None
        if record is not None:
            record["outcome"] = "late_result"
        self.fail(FailureKind.INFRASTRUCTURE, stage,
                  f"late result: {stage} finished after the total wall deadline; result "
                  f"rejected, retained unscored ({over})", attempt=attempt, request_id=request_id)
        self.fail(FailureKind.EXHAUSTED, "budget", over, attempt=attempt)
        return self.result("agentic_exhausted", self.policy.route)

    def _private_root(self) -> Optional[Path]:
        """The dispatch's parent-private evidence root (round 9): created once per
        dispatch (exclusive, 0700).  Round 10: the persisted (public) state carries
        only its opaque id and inode identity, never its path; the path lives in
        memory and in the root's own state file, and resume re-finds the root by
        identity.  Recreated only if it disappeared."""
        p = self._private_root_path
        if p is not None and Path(p).is_dir():
            return Path(p)
        found = _private_root_find(self.out_dir / "agentic", self.state.get("private_evidence"))
        if found is not None:
            self._private_root_path = found
            return found
        try:
            root = _private_root_create(self.out_dir / "agentic")
        except OSError:
            return None
        st = os.stat(str(root))
        self._private_root_path = root
        self.state["private_evidence"] = {"id": _private_root_id(root), "dev": st.st_dev, "ino": st.st_ino,
                                          "path_in_public_records": False}
        self.state.pop("private_evidence_root", None)
        self.save()
        return root

    def run_attempts(self) -> DispatchResult:
        b = self.policy.budget
        attempt = len(self.state["attempts"]) + 1
        while attempt <= b.max_attempts:
            over = self.over_deadline()
            if over:
                kind, status, code = self._over_outcome()
                self.fail(kind, "budget", over, attempt=attempt, reason_code=code)
                return self.result(status, self.policy.route)
            over = self._requests_exceeded()
            if over:
                self.fail(FailureKind.EXHAUSTED, "budget", over, attempt=attempt)
                return self.result("agentic_exhausted", self.policy.route)
            # raw reference observations of earlier attempts are an oracle for the
            # next candidate (reference-27, retry scope): removed before the
            # request is even issued; anything that cannot be removed refuses the attempt
            self._private_root()
            sealed = _seal_reference_material(self.out_dir / "agentic", census=_census_load(self.out_dir / "agentic"))
            self.state.setdefault("sealed_reference_material", []).append(
                {"before_attempt": attempt, "utc": _utc(), **sealed})
            if sealed["retained"]:
                self.fail(FailureKind.INFRASTRUCTURE, "agentic_request",
                          f"reference_material_retained: raw trusted outputs of an earlier attempt could not be "
                          f"removed before attempt {attempt} ({sealed['retained']})", attempt=attempt,
                          reason_code="reference_material_retained")
                return self.result("agentic_infrastructure_error", self.policy.route)
            treq, attempt_dir = self._make_request(attempt)
            record = {"attempt": attempt, "request_id": treq.request_id,
                      "request_dir": str(attempt_dir), "started_utc": _utc(),
                      "counted": True, "outcome": None}
            self.state["attempts"].append(record)
            self.save()
            try:
                cand = self.provider.generate(treq, attempt_dir, self.reservation)
            except PendingDispatch as pend:
                record["outcome"] = "awaiting_agent"
                record["counted"] = False
                self.state["pending_request_dir"] = str(pend.request_dir)
                return self.result("awaiting_agent", self.policy.route,
                                   resume_token=str(pend.request_dir))
            except ProviderError as pe:
                self._account(getattr(pe, "extra", {}))
                record["outcome"] = pe.kind
                record["reason"] = pe.reason
                record["api_requests"] = int(pe.extra.get("api_requests", 0) or 0)
                self.fail(pe.kind, "agentic_generate", pe.reason, attempt=attempt,
                          request_id=treq.request_id, provenance=pe.extra.get("provenance"),
                          exception_type=pe.extra.get("exception_type"),
                          http_status=pe.extra.get("http_status"),
                          reason_code=pe.extra.get("reason_code"),
                          api_requests=record["api_requests"],
                          cost_usd=pe.extra.get("cost_usd"),
                          cost_basis=pe.extra.get("cost_basis"),
                          transport_budget=pe.extra.get("transport_budget"))
                if pe.kind == FailureKind.NO_BUDGET:
                    return self.result("no_budget", self.policy.route)
                if pe.kind == FailureKind.NO_PROVIDER:
                    return self.result("no_provider", self.policy.route)
                if pe.kind == FailureKind.EXHAUSTED:
                    return self.result("agentic_exhausted", self.policy.route)
                attempt += 1
                continue
            except Exception as exc:  # provider bug or host fault
                kind = classify_exception(exc)
                record["outcome"] = kind
                record["reason"] = f"{type(exc).__name__}: {exc}"
                self.fail(kind, "agentic_generate", record["reason"], attempt=attempt,
                          request_id=treq.request_id, exception_type=type(exc).__name__,
                          traceback=traceback.format_exc()[-2000:])
                attempt += 1
                continue
            # account the call FIRST so the ledger, DISPATCH_STATE and the
            # DISPATCH_RECEIPT agree whatever happens next (an earlier fix-BUDGET-12c)
            self._account({"api_requests": cand.api_requests, "cost_usd": cand.cost_usd})
            record["candidate_sha256"] = cand.dsl_sha256
            record["api_requests"] = int(cand.api_requests)
            record["cost_usd"] = float(cand.cost_usd)
            self.save()
            late = self._late("agentic_generate", attempt, treq.request_id, record=record)
            if late is not None:
                record["outcome"] = "late_result"
                self.save()
                return late
            res = self._validate(cand, record, attempt, accounted=True)
            if res is not None:
                return res
            attempt += 1
        last = self.state["failure_receipts"][-1] if self.state["failure_receipts"] else None
        if last and last["kind"] in (FailureKind.INFRASTRUCTURE, FailureKind.PROTOCOL):
            # the final failure was not scored: say so, keep it retryable
            self.fail(FailureKind.EXHAUSTED, "budget",
                      f"attempt cap {b.max_attempts} reached; last failure was "
                      f"{last['kind']} (retained for retry, not scored)")
            return self.result("agentic_infrastructure_error", self.policy.route)
        self.fail(FailureKind.EXHAUSTED, "budget",
                  f"attempt cap {b.max_attempts} reached without a verified candidate")
        return self.result("agentic_exhausted", self.policy.route)

    def _known_shas(self, exclude_record: dict) -> set:
        known = set()
        for p in self.request.known_candidates:
            p = Path(p)
            if p.is_file():
                known.add(_sha256(p))
        for a in self.state["attempts"]:
            if a is not exclude_record and a.get("candidate_sha256"):
                known.add(a["candidate_sha256"])
        return known

    def _validate(self, cand: Candidate, record: dict, attempt: int,
                  accounted: bool = False) -> Optional[DispatchResult]:
        """Validate a candidate at the common boundary; None means 'continue
        to the next attempt'.  Infrastructure-only verdicts re-validate the
        SAME bytes within the execution caps (no new model request)."""
        if not accounted:
            self._account({"api_requests": cand.api_requests, "cost_usd": cand.cost_usd})
        record["candidate_sha256"] = cand.dsl_sha256
        record["candidate"] = cand.to_dict()
        # source binding at the dispatcher layer: the candidate must claim the
        # same source bytes that are on disk NOW (the verifier binds the bytes
        # it executes; the receipt re-checks them again).
        now = _sha256(self.request.source_path)
        if cand.source_sha256 != now or now != self.state["source_sha256"]:
            record["outcome"] = FailureKind.PROTOCOL
            self.fail(FailureKind.PROTOCOL, "source_binding",
                      f"candidate claims source {cand.source_sha256[:12]}…, the source on disk is "
                      f"{now[:12]}…, the dispatch started on {self.state['source_sha256'][:12]}…: "
                      f"refusing to validate an unbound candidate", attempt=attempt,
                      request_id=record["request_id"])
            return None
        # freshness at the common boundary (fresh-7), every provider
        if cand.dsl_sha256 in self._known_shas(record):
            record["outcome"] = FailureKind.UNSUPPORTED
            self.fail(FailureKind.UNSUPPORTED, "freshness",
                      "candidate bytes equal a known saved DSL or an earlier attempt's "
                      "candidate: not byte-distinct work (byte inequality would not be "
                      "authorship proof either)", attempt=attempt, request_id=record["request_id"],
                      candidate_sha256=cand.dsl_sha256)
            return None
        return self._run_validation(cand, record, attempt)

    def _retries_left(self) -> int:
        """Validation retries remaining from PERSISTED state (retry-16):
        the cap is a total over the dispatch, resumes included."""
        return max(0, self.policy.budget.max_validation_retries
                   - int(self.state.get("validation_retries_used", 0)))

    def _bytes_unchanged(self, cand: Candidate, record: dict, attempt: int, when: str) -> bool:
        """Re-hash the candidate (and adapter) bytes immediately before a
        validation attempt; different bytes are refused (retry-16)."""
        checks = [("candidate", cand.dsl_path, cand.dsl_sha256)]
        if cand.adapter_sha256:
            checks.append(("adapter", cand.model_path, cand.adapter_sha256))
        for label, path, want in checks:
            have = _sha256(path) if Path(path).is_file() else None
            if have != want:
                record["outcome"] = FailureKind.PROTOCOL
                self.fail(FailureKind.PROTOCOL, "agentic_validate",
                          f"candidate_bytes_changed ({when}): {label} bytes bound as {want[:12]}… are "
                          f"now {str(have)[:12]}…; refusing to validate different bytes under the "
                          f"retained attempt", attempt=attempt, request_id=record["request_id"],
                          candidate_sha256=cand.dsl_sha256, reason_code="candidate_bytes_changed")
                self.state.pop("pending_validation", None)
                self.save()
                return False
        return True

    def _run_validation(self, cand: Candidate, record: dict, attempt: int,
                        first_is_retry: bool = False) -> Optional[DispatchResult]:
        receipt_out = Path(record["request_dir"]) / "receipt.json"
        tries = 0
        while True:
            over = self.over_deadline()
            if over:
                self.fail(FailureKind.INFRASTRUCTURE, "agentic_validate",
                          f"validation not started/continued: {over}; candidate retained in "
                          f"pending-infrastructure state", attempt=attempt,
                          request_id=record["request_id"], candidate_sha256=cand.dsl_sha256)
                return self._pending(cand, record, attempt, over)
            if first_is_retry or tries > 0:
                if self._retries_left() <= 0:
                    self.fail(FailureKind.EXHAUSTED, "agentic_validate",
                              f"validation retry budget exhausted ({self.state.get('validation_retries_used', 0)} "
                              f"of max {self.policy.budget.max_validation_retries}); candidate retained, "
                              f"not re-validated", attempt=attempt, request_id=record["request_id"],
                              candidate_sha256=cand.dsl_sha256)
                    return self._pending(cand, record, attempt, "validation retry budget exhausted")
                self.state["validation_retries_used"] = int(self.state.get("validation_retries_used", 0)) + 1
                self.save()
            if not self._bytes_unchanged(cand, record, attempt, "before validation try %d" % (tries + 1)):
                return None
            tries += 1
            token = _PRIVATE_EVIDENCE_ROOT.set(self._private_root())
            rem = self.remaining()
            dl_token = _DISPATCH_DEADLINE.set(None if rem is None else time.monotonic() + max(0.0, rem))
            try:
                verdict = self.validator(cand, self.request, self._bounded_policy(), receipt_out, self.log)
            finally:
                _DISPATCH_DEADLINE.reset(dl_token)
                _PRIVATE_EVIDENCE_ROOT.reset(token)
            # the full record (numeric statistics included) is private until the
            # dispatch ends; attempt-visible state gets the redacted view
            full_record = verdict.to_dict()
            numeric_id = self._stash_numeric(full_record, verdict.message)
            visible = _attempt_visible(full_record, numeric_id, public_message=verdict.public_message)
            # round 13: a verdict with a constant public message shows exactly that text
            visible_message = verdict.public_message or _redact_numerals(verdict.message)
            record["validation"] = visible
            record["validation_tries"] = record.get("validation_tries", 0) + 1
            late = self.over_deadline()
            if late:
                # EVERY late outcome is infrastructure before any scoring
                # (wall-15): a late mismatch is not a model verdict either
                self.fail(FailureKind.INFRASTRUCTURE, "agentic_validate",
                          f"late result: validation finished after the total wall deadline "
                          f"(verdict {verdict.status}); rejected, retained unscored ({late})",
                          attempt=attempt, request_id=record["request_id"],
                          candidate_sha256=cand.dsl_sha256, verdict_status=verdict.status)
                return self._pending(cand, record, attempt, late)
            if verdict.kind == FailureKind.PROTOCOL:
                # bytes changed / binding refused: unscored, the attempt ends
                record["outcome"] = verdict.status
                self.fail(verdict.kind, "agentic_validate", visible_message, attempt=attempt,
                          request_id=record["request_id"], validation=visible, reason_redacted=True, reason_private_id=numeric_id,
                          candidate_sha256=cand.dsl_sha256, reason_code=verdict.reason_code)
                self.state.pop("pending_validation", None)
                self.save()
                return None
            if verdict.status == "verified":
                record["outcome"] = "verified"
                self.state["candidate"] = cand.to_dict()
                self.state["validation"] = visible
                self.state.pop("pending_validation", None)
                self.save()
                return self.result("agentic_verified", self.policy.route)
            if verdict.kind == FailureKind.INFRASTRUCTURE:
                self.fail(verdict.kind, "agentic_validate", visible_message, attempt=attempt,
                          request_id=record["request_id"], validation=visible, reason_redacted=True, reason_private_id=numeric_id,
                          candidate_sha256=cand.dsl_sha256, provenance=cand.provenance,
                          validation_try=tries, reason_code=verdict.reason_code)
                if self._retries_left() > 0 and not self.over_deadline():
                    self.log(f"[dispatch] re-validating the same candidate bytes "
                             f"({cand.dsl_sha256[:12]}…) after an infrastructure-only failure")
                    continue
                return self._pending(cand, record, attempt, visible_message)
            # scored or unresolved verdicts end this attempt
            record["outcome"] = verdict.status
            self.fail(verdict.kind, "agentic_validate", visible_message, attempt=attempt,
                      request_id=record["request_id"], validation=visible, reason_redacted=True, reason_private_id=numeric_id,
                      candidate_sha256=cand.dsl_sha256, provenance=cand.provenance,
                      reason_code=verdict.reason_code)
            if verdict.status == "tolerance_unresolved":
                # not a model answer and not retryable by re-running: the caller
                # must supply the approved tolerance policy
                return self.result("tolerance_unresolved", self.policy.route)
            return None

    def _pending(self, cand: Candidate, record: dict, attempt: int, reason: str) -> DispatchResult:
        record["outcome"] = "pending_infrastructure"
        self.state["pending_validation"] = {
            "attempt": attempt, "request_id": record["request_id"], "request_dir": record["request_dir"],
            "candidate": cand.to_dict(), "reason": reason, "utc": _utc(),
            "note": "same bytes re-validated on resume within the execution caps; no new model request"}
        self.save()
        return self.result("agentic_infrastructure_error", self.policy.route)

    # -- resume ----------------------------------------------------------

    @classmethod
    def load(cls, out_dir: Path, provider, validator, reservation, log) -> "_Dispatch":
        out_dir = Path(out_dir)
        try:
            state = json.loads((out_dir / _STATE_FILE).read_text())
        except FileNotFoundError as exc:
            raise NothingToResume(f"no {_STATE_FILE} in {out_dir}; nothing to resume") from exc
        rq = dict(state["request"])
        request = DispatchRequest(
            source_path=rq["source_path"], out_dir=out_dir, source_type=rq["source_type"],
            definition_dir=rq.get("definition_dir"), workload_idx=rq.get("workload_idx", 0),
            dtype=rq.get("dtype"), original_axes=rq.get("original_axes"),
            model_route_label=rq.get("model_route_label", ""),
            sparsity_family_hint=rq.get("sparsity_family_hint"),
            tolerance=rq.get("tolerance"),
            known_candidates=tuple(rq.get("known_candidates") or ()))
        pol = dict(state["policy"])
        policy = DispatchPolicy(**{**pol, "budget": Budget(**pol["budget"])})
        d = cls(request, policy, provider, None, validator, reservation, log)
        d.state = state
        # absolute deadline persisted: elapsed includes time spent awaiting the agent
        return d

    def _resume(self) -> DispatchResult:
        status = self.state.get("status")
        pending = self.state.get("pending_validation")
        if status == "awaiting_agent":
            return self._resume_awaiting()
        if pending and status == "agentic_infrastructure_error":
            return self._resume_pending(pending)
        raise NothingToResume(f"dispatch in {self.out_dir} is {status!r} with no pending agent "
                              f"request or pending-infrastructure candidate; nothing to resume")

    def _check_provider(self) -> Optional[DispatchResult]:
        if self.provider is None or getattr(self.provider, "route", None) != self.policy.route:
            self.fail(FailureKind.NO_PROVIDER, "provider",
                      f"resume needs a provider for route {self.policy.route!r}")
            return self.result("no_provider", "none")
        return None

    def _source_changed(self, attempt: int) -> Optional[DispatchResult]:
        now = _sha256(self.request.source_path) if self.request.source_path.is_file() else None
        if now != self.state["source_sha256"]:
            self.fail(FailureKind.PROTOCOL, "source_binding",
                      f"source changed while the dispatch was suspended (was "
                      f"{self.state['source_sha256'][:12]}…, now {str(now)[:12]}…)", attempt=attempt)
            return self.result("source_changed", self.policy.route)
        return None

    def _resume_awaiting(self) -> DispatchResult:
        bad = self._check_provider()
        if bad is not None:
            return bad
        record = self.state["attempts"][-1]
        attempt = record["attempt"]
        request_dir = Path(self.state["pending_request_dir"])
        changed = self._source_changed(attempt)
        if changed is not None:
            record["outcome"] = FailureKind.PROTOCOL
            record["counted"] = True
            self.save()
            return changed
        over = self.over_deadline()
        if over:
            # time awaiting the agent counts; a late response is not collected
            record["outcome"] = "late_result"
            record["counted"] = True
            kind, status, code = self._over_outcome()
            self.fail(FailureKind.INFRASTRUCTURE, "agentic_collect",
                      f"deadline passed while awaiting the agent; any response is rejected "
                      f"unscored and retained ({over})", attempt=attempt, request_id=record["request_id"],
                      reason_code=code)
            self.fail(kind, "budget", over, attempt=attempt, reason_code=code)
            return self.result(status, self.policy.route)
        try:
            cand = self.provider.collect(request_dir)
        except ProviderError as pe:
            if pe.kind == FailureKind.PROTOCOL and "has not responded" in pe.reason:
                return self.result("awaiting_agent", self.policy.route, resume_token=str(request_dir))
            record["outcome"] = pe.kind
            record["counted"] = True
            record["reason"] = pe.reason
            self.fail(pe.kind, "agentic_collect", pe.reason, attempt=attempt,
                      request_id=record["request_id"], provenance=pe.extra.get("provenance"))
            self.state.pop("pending_request_dir", None)
            return self.run_attempts()
        record["counted"] = True
        self.state.pop("pending_request_dir", None)
        late = self._late("agentic_collect", attempt, record["request_id"], record=record)
        if late is not None:
            record["outcome"] = "late_result"
            record["candidate_sha256"] = cand.dsl_sha256
            self.save()
            return late
        res = self._validate(cand, record, attempt)
        if res is not None:
            return res
        return self.run_attempts()

    def _resume_pending(self, pending: dict) -> DispatchResult:
        """Re-validate the retained candidate bytes (no new model request)."""
        attempt = int(pending["attempt"])
        record = next((a for a in self.state["attempts"] if a.get("attempt") == attempt),
                      self.state["attempts"][-1])
        changed = self._source_changed(attempt)
        if changed is not None:
            return changed
        c = dict(pending["candidate"])
        dsl_path = Path(c["dsl_path"])
        if not dsl_path.is_file() or _sha256(dsl_path) != c["dsl_sha256"]:
            self.fail(FailureKind.PROTOCOL, "agentic_validate",
                      f"retained candidate bytes changed or vanished ({dsl_path}); refusing to "
                      f"re-validate different bytes under the retained attempt", attempt=attempt,
                      request_id=record.get("request_id"))
            self.state.pop("pending_validation", None)
            return self.result("agentic_infrastructure_error", self.policy.route)
        cand = Candidate(dsl_path=dsl_path, model_path=Path(c["model_path"]),
                         dsl_sha256=c["dsl_sha256"], source_sha256=c["source_sha256"],
                         provenance=dict(c.get("provenance") or {}),
                         api_requests=0, cost_usd=0.0,       # already accounted
                         cost_basis=c.get("cost_basis", ""))
        self.state.pop("pending_validation", None)
        self.save()
        over = self.over_deadline()
        if over:
            kind, _status, code = self._over_outcome()
            self.fail(kind, "budget", over, attempt=attempt, reason_code=code)
            return self._pending(cand, record, attempt, over)
        # a re-validation on resume IS a retry: it consumes the persisted budget
        res = self._run_validation(cand, record, attempt, first_is_retry=True)
        if res is not None:
            return res
        return self.run_attempts()


def dispatch(request: DispatchRequest, policy: Optional[DispatchPolicy] = None,
             provider: Optional[AgenticProvider] = None, *,
             deterministic_stage: Optional[Callable] = None,
             validator: Optional[Callable] = None,
             reservation=None, log: Optional[Callable] = None) -> DispatchResult:
    """Run the route policy once.  Deterministic first; on an eligible
    failure, the selected agentic ``provider`` under ``policy.budget``.

    ``provider=None`` is the DEFAULT and fails closed with ``no_provider``
    once the deterministic route is unsupported — nothing calls out unless
    the caller wires a provider in.  ``reservation`` is required by
    providers that call an external API and is consumed at the actual call
    boundary.  The result and ``<out_dir>/DISPATCH_RECEIPT.json`` retain
    the deterministic outcome, the why, every attempt and every failure
    receipt.
    """
    return _Dispatch(request, policy or DispatchPolicy(), provider,
                     deterministic_stage, validator, reservation, log).start()


def resume(out_dir, provider: AgenticProvider, *, validator: Optional[Callable] = None,
           reservation=None, log: Optional[Callable] = None) -> DispatchResult:
    """Resume a dispatch that returned ``awaiting_agent`` (consume the
    agent's response, validate the candidate, continue the bounded loop) or
    ``agentic_infrastructure_error`` with a pending-infrastructure candidate
    (re-validate the SAME bytes, no new model request).  The absolute wall
    deadline persists: time spent suspended counts."""
    return _Dispatch.load(out_dir, provider, validator, reservation, log).resume()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m solar.frontend.dispatcher",
        description="Deterministic-first translation with bounded agentic fallback.")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="dispatch one source")
    run.add_argument("source")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--source-type", default="pytorch")
    run.add_argument("--route", default="pure_skill", choices=ROUTES)
    run.add_argument("--definition", default=None)
    run.add_argument("--workload-idx", type=int, default=0)
    run.add_argument("--dtype", default=None)
    run.add_argument("--rtol", type=float, default=None,
                     help="explicit approved tolerance (with --atol); otherwise the "
                          "selected workload.jsonl line's tolerance; otherwise unresolved")
    run.add_argument("--atol", type=float, default=None)
    run.add_argument("--max-attempts", type=int, default=1)
    run.add_argument("--max-validation-retries", type=int, default=1)
    run.add_argument("--wall-seconds", type=float, default=None)
    run.add_argument("--max-requests", type=int, default=None)
    run.add_argument("--max-cost-usd", type=float, default=None)
    run.add_argument("--cost-per-request-usd", type=float, default=None,
                     help="pythonic: the explicit per-request HARD bound, preauthorized before "
                          "each transport request and the only basis on which unmeasured usage "
                          "can be admitted; without it each request preauthorizes the whole "
                          "remaining reservation and a result without a measurable cost is "
                          "refused (cost_unbounded)")
    run.add_argument("--cost-per-1k-tokens-usd", type=float, default=None,
                     help="pythonic: rate that turns the provider's reported token usage into "
                          "the measured cost reconciled against the bound")
    run.add_argument("--verify-timeout", type=int, default=None)
    run.add_argument("--mem-budget", type=float, default=None)
    run.add_argument("--arch", default="B200")
    run.add_argument("--precision", default="fp16")
    run.add_argument("--known-candidate", action="append", default=[],
                     help="saved DSL whose bytes must not be presented as new work")
    run.add_argument("--poll-seconds", type=float, default=0.0,
                     help="pure_skill: wait this long for the agent's response "
                          "(default 0: return awaiting_agent and resume later)")
    run.add_argument("--api-key", default=None,
                     help="pythonic: explicit key (never read from env/files here)")
    run.add_argument("--reservation", default=None, metavar="REQUESTS:USD:LABEL",
                     help="pythonic: ledger reservation handle to consume from")
    run.add_argument("--sparsity-family", default=None,
                     help="declare a data-dependent-sparsity family (moe, "
                          "sparse_attention, ...): agentic-by-default, reason recorded, "
                          "carried through every deterministic outcome")
    run.add_argument("--no-agentic", action="store_true")
    rs = sub.add_parser("resume", help="resume an awaiting_agent / pending-infrastructure dispatch")
    rs.add_argument("out_dir")
    rs.add_argument("--api-key", default=None)
    rs.add_argument("--cost-per-1k-tokens-usd", type=float, default=None)
    rs.add_argument("--reservation", default=None, metavar="REQUESTS:USD:LABEL")
    args = p.parse_args(argv)

    def _reservation(spec):
        if not spec:
            return None
        n, usd, *label = spec.split(":")
        return LedgerReservation(int(n), float(usd), label=":".join(label))

    if args.cmd == "run":
        if (args.rtol is None) != (args.atol is None):
            p.error("--rtol and --atol must be given together")
        policy = DispatchPolicy(
            route=args.route, allow_agentic=not args.no_agentic,
            budget=Budget(max_attempts=args.max_attempts, wall_seconds=args.wall_seconds,
                          max_requests=args.max_requests, max_cost_usd=args.max_cost_usd,
                          max_validation_retries=args.max_validation_retries),
            verify_timeout=args.verify_timeout, mem_budget_gb=args.mem_budget,
            arch=args.arch, precision=args.precision,
            cost_per_request_usd=args.cost_per_request_usd)
        provider = (PureSkillProvider(poll_seconds=args.poll_seconds) if args.route == "pure_skill"
                    else PythonicProvider(api_key=args.api_key, verify_timeout=args.verify_timeout,
                                          mem_budget_gb=args.mem_budget,
                                          cost_per_request_usd=args.cost_per_request_usd,
                                          cost_per_1k_tokens_usd=args.cost_per_1k_tokens_usd))
        req = DispatchRequest(source_path=args.source, out_dir=args.out_dir,
                              source_type=args.source_type, definition_dir=args.definition,
                              workload_idx=args.workload_idx, dtype=args.dtype,
                              sparsity_family_hint=args.sparsity_family,
                              tolerance=(None if args.rtol is None else
                                         {"rtol": args.rtol, "atol": args.atol}),
                              known_candidates=tuple(args.known_candidate))
        res = dispatch(req, policy, provider, reservation=_reservation(args.reservation), log=print)
    else:
        state = json.loads((Path(args.out_dir) / _STATE_FILE).read_text())
        route = state["policy"]["route"]
        provider = (PureSkillProvider() if route == "pure_skill"
                    else PythonicProvider(api_key=args.api_key,
                                          cost_per_1k_tokens_usd=args.cost_per_1k_tokens_usd))
        res = resume(args.out_dir, provider, reservation=_reservation(args.reservation), log=print)
    # one parseable summary line (nested JSON stays on it, no pretty-print)
    print("DISPATCH-SUMMARY: " + json.dumps(
        {"status": res.status, "final_route": res.final_route, "why": res.why,
         "sol_source": res.sol_source, "ledger": res.ledger, "resume_token": res.resume_token,
         "receipt": res.receipt_path}, default=str))
    return 0 if res.status in ("mechanical_ok", "agentic_verified") else (
        4 if res.status == "awaiting_agent" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
