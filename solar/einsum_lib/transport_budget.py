# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-invocation transport budget for provider calls (budget-1).

The dispatcher authorizes a bounded number of transport requests and a
bounded preauthorized cost for ONE provider invocation and installs the
budget as the active context.  ``TranslationAgent._complete`` charges the
budget immediately before every ``chat.completions.create`` call — the
actual call boundary — so the agent's internal model fallback and
rate-limit retries cannot issue requests the caller did not authorize.  A
charge the budget cannot cover raises ``TransportBudgetExhausted`` BEFORE
the request is sent.  Failed calls stay charged (they were sent).

Without an active budget the agent behaves as before (no charging): the
budget is the dispatcher's instrument, not a global policy.
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import time
from dataclasses import dataclass, field
from typing import Optional


class TransportBudgetExhausted(RuntimeError):
    """Raised at the call boundary when no transport permit / funds remain."""


def _jsonable(value, depth: int = 0):
    """Total conversion to JSON-safe data: unknown objects are stringified,
    never raised on (budget-12: serialization of the accounting state
    must not be able to fail)."""
    if depth > 6:
        return repr(value)[:500]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k)[:200]: _jsonable(v, depth + 1) for k, v in list(value.items())[:200]}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, depth + 1) for v in list(value)[:200]]
    try:
        return repr(value)[:500]
    except Exception:  # pragma: no cover - a repr that raises
        return f"<unrepresentable {type(value).__name__}>"


def _token_count(value):
    """A reported token count is a non-negative integer (bool excluded);
    anything else is unparsed (returned as None)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


@dataclass
class TransportBudget:
    """Permits and funds for one provider invocation.

    ``permits`` bounds the transport requests this invocation may issue
    (None = unbounded by count; the reservation still bounds it).
    ``cost_per_request_usd`` is PREAUTHORIZED against ``reservation`` for
    every request at the call boundary (``reservation.consume(1, cost)``);
    the first ``prepaid`` permits were already consumed by the invoker.

    Accounting is PRIMITIVE (budget-12): ``calls`` records one row per
    transport request at the moment it is issued, before any response is
    parsed; ``requests_used``/``cost_preauthorized_usd`` derive from those
    rows only.  Usage reconciliation (``record_usage``) is a separate,
    total, exception-free step: an unparseable usage object is stored
    verbatim as ``usage_unparsed`` and never disturbs the debit.  Cost
    annotations (``annotate``) live in ``annotations``, never in ``calls``,
    so a note can never count as a request (accounting-20).
    ``to_dict`` is total: it never raises.
    """

    permits: Optional[int]
    cost_per_request_usd: float
    reservation: object = None
    prepaid: int = 0
    calls: list = field(default_factory=list)
    annotations: list = field(default_factory=list)

    @property
    def requests_used(self) -> int:
        """Transport requests actually issued (annotations never count)."""
        return len(self.calls)

    @property
    def cost_preauthorized_usd(self) -> float:
        return float(self.requests_used) * float(self.cost_per_request_usd)

    def charge(self, model: Optional[str] = None) -> None:
        if self.permits is not None and self.requests_used >= self.permits:
            raise TransportBudgetExhausted(
                f"transport permit cap reached ({self.permits} request(s) "
                f"authorized for this invocation); no further provider call "
                f"is made")
        if self.requests_used >= self.prepaid and self.reservation is not None:
            try:
                self.reservation.consume(
                    1, float(self.cost_per_request_usd),
                    note=f"transport request {self.requests_used + 1} model={model}")
            except Exception as exc:  # the reservation refused: fail closed
                raise TransportBudgetExhausted(
                    f"reservation cannot preauthorize transport request "
                    f"{self.requests_used + 1} (${self.cost_per_request_usd:.4f}): "
                    f"{exc}") from exc
        # the primitive debit row: written at call issue, before any response
        self.calls.append({"n": self.requests_used + 1, "model": _jsonable(model),
                           "cost_preauthorized_usd": float(self.cost_per_request_usd),
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def record_usage(self, model: Optional[str], usage) -> None:
        """Attach the provider's reported usage (token counts) to the most
        recent charged call so the invoker can reconcile cost
        (cost-13).  Total: never raises.  Fields that are not
        non-negative integers are recorded as unparsed, the raw object is
        kept verbatim (stringified) and the debit is untouched."""
        rec = {"model": _jsonable(model), "prompt_tokens": None,
               "completion_tokens": None, "total_tokens": None,
               "usage_unparsed": None}
        try:
            if usage is None:
                rec["usage_unparsed"] = "absent"
            else:
                raw = {}
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    try:
                        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                    except Exception as exc:
                        value = f"<{type(exc).__name__}: {exc}>"
                    raw[name] = value
                    rec[name] = _token_count(value)
                if any(rec[n] is None for n in ("prompt_tokens", "completion_tokens", "total_tokens")):
                    rec["usage_unparsed"] = _jsonable(raw)
        except Exception as exc:  # pragma: no cover - defensive total step
            rec["usage_unparsed"] = f"<{type(exc).__name__}: {exc}>"
        if self.calls:
            self.calls[-1]["usage"] = rec
        else:
            # a usage report without a charged call: an annotation, not a request
            self.annotate("usage_without_call", usage=rec)

    def annotate(self, kind: str, **data) -> None:
        """Record cost/accounting metadata OUTSIDE the request count."""
        self.annotations.append({"kind": str(kind), **_jsonable(data),
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    @property
    def total_tokens(self) -> Optional[int]:
        """Sum of parsed total tokens over charged calls, or None when any
        charged call carries no parseable total (cost cannot be measured)."""
        total = 0
        for c in self.calls:
            u = c.get("usage") if isinstance(c, dict) else None
            tokens = _token_count(u.get("total_tokens")) if isinstance(u, dict) else None
            if tokens is None:
                return None
            total += tokens
        return total

    def to_dict(self) -> dict:
        """Total serialization: never raises, unknown values stringified."""
        try:
            return {"permits": _jsonable(self.permits), "prepaid": _jsonable(self.prepaid),
                    "cost_per_request_usd": _jsonable(self.cost_per_request_usd),
                    "requests_used": self.requests_used,
                    "cost_preauthorized_usd": _jsonable(self.cost_preauthorized_usd),
                    "total_tokens": self.total_tokens,
                    "calls": _jsonable(list(self.calls)),
                    "annotations": _jsonable(list(self.annotations))}
        except Exception as exc:  # pragma: no cover - last line of defence
            return {"requests_used": len(self.calls) if isinstance(self.calls, list) else None,
                    "serialization_error": f"{type(exc).__name__}: {exc}"}


_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "solar_transport_budget", default=None)


def active_transport_budget() -> Optional[TransportBudget]:
    return _ACTIVE.get()


@contextlib.contextmanager
def transport_budget(budget: TransportBudget):
    """Install *budget* as the active transport budget for the block."""
    token = _ACTIVE.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE.reset(token)


def charge_transport(model: Optional[str] = None) -> None:
    """Charge the active budget (no-op when none is installed)."""
    budget = _ACTIVE.get()
    if budget is not None:
        budget.charge(model)


def record_transport_usage(model: Optional[str], usage) -> None:
    """Record a provider usage report on the active budget (no-op when none)."""
    budget = _ACTIVE.get()
    if budget is not None:
        budget.record_usage(model, usage)


def permits_for(max_requests_remaining: Optional[int],
                reservation_requests_remaining: Optional[int],
                reservation_usd_remaining: Optional[float],
                cost_cap_usd_remaining: Optional[float],
                cost_per_request_usd: float,
                per_invocation_cap: Optional[int]) -> int:
    """How many transport requests can be authorized now (>= 0)."""
    bounds = []
    for b in (max_requests_remaining, reservation_requests_remaining, per_invocation_cap):
        if b is not None:
            bounds.append(int(b))
    if cost_per_request_usd > 0:
        for usd in (reservation_usd_remaining, cost_cap_usd_remaining):
            if usd is not None:
                bounds.append(int(math.floor(float(usd) / float(cost_per_request_usd) + 1e-9)))
    return max(0, min(bounds)) if bounds else 10 ** 9
