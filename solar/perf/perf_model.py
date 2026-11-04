# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance model for an analyzed einsum graph.

This module implements the **third stage** of the Solar pipeline:

  `analysis.yaml` + `configs/arch/<ARCH>.yaml`  ->  `perf_<ARCH>.yaml`

Three SOL (Speed-of-Light) roofline models are computed:
1. Unfused: Each op runs in isolation, all tensors from DRAM
2. Fused: Per-op roofline, intermediate tensors excluded from memory cost
3. Fused+Prefetched: Single roofline for entire graph, perfect overlap assumed

See SOL_GUIDE.md for detailed explanation.
"""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.utils import ensure_directory, NoAliasDumper, yaml_safe_load
import logging

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def _exact_count(value, errors=None):
    """An exact non-negative integer MAC count as a Python ``int``, or ``None``.

    THE one validator for every raw count field (``total.macs``, every
    ``layers.<id>.macs``, every ``total.macs_by_compute_dtype.<dtype>``), applied at
    the loader boundary BEFORE any arithmetic or conversion (count-coercion-1,
    ruling 2026-10-26: counts are integers, no tolerance).  Accepted: any
    ``numbers.Integral`` that is not a bool (Python ``int``, ``numpy.int64`` /
    ``uint64`` / ``intp`` ...), converted exactly with ``int()``; and any finite
    ``numbers.Real`` whose value is EXACTLY integral — tested on the ORIGINAL
    object with the type's own arithmetic (``value % 1 == 0``: long double stays
    long double, ``Fraction`` stays rational) and converted with the type's exact
    ``int()`` (count-real-rounding-1: ``float()`` is never called on a count,
    so a wider or exact type is neither rounded into acceptance nor loses a MAC:
    ``numpy.longdouble('2000000.0000000001')`` and ``Fraction(20000000000000001,
    10000000000)`` are rejected, ``numpy.longdouble('9007199254740993')`` yields
    9007199254740993).  ``2000000.0`` / ``numpy.float64(2000000.0)`` remain the
    documented integral-float acceptance.  Rejected (``None``): bool /
    ``numpy.bool_``, NaN, ±inf, negative, fractional, ``str``, ``None``,
    ``decimal.Decimal`` (not a ``numbers.Real``) and every other type.

    ANY exception the object's numeric protocol raises while being validated
    (``int()``, ``%``, ``==`` of a typed subtype: ``TypeError`` / ``ValueError`` /
    ``OverflowError`` / ``ArithmeticError`` / ``RuntimeError`` /
    ``NotImplementedError`` / anything else) means the value is malformed DATA —
    the function returns ``None`` and, when ``errors`` (a list) is given, appends
    ``"<ExceptionType>: <message>"`` so the refusal receipt can record it
    (count-real-exception-1).  Process-control exceptions are never
    swallowed: ``KeyboardInterrupt`` / ``SystemExit`` / ``GeneratorExit`` are not
    ``Exception`` subclasses and ``MemoryError`` is re-raised explicitly.  The
    function is self-contained (its import is local) so it can be extracted and
    exercised on its own, as the independent review does.
    """
    import numbers

    def _protocol_error(exc):
        if errors is not None:
            try:
                errors.append(f"{type(exc).__name__}: {exc}")
            except MemoryError:                     # protocol-message-memory-1: never swallowed, even from __str__
                raise
            except Exception:                       # a message that cannot be rendered is still an error
                errors.append(type(exc).__name__)
        return None

    try:
        if isinstance(value, bool) or getattr(getattr(value, "dtype", None), "kind", None) == "b":   # Python bool / numpy bool (any NumPy version)
            return None
        if isinstance(value, numbers.Integral):
            n = int(value)
            return n if n >= 0 else None
        if isinstance(value, numbers.Real):
            n = int(value)                  # exact for every finite value of every numeric type; raises on NaN / ±inf
            integral = bool(value % 1 == 0)  # integrality on the ORIGINAL object, its own arithmetic — no float()
            if not integral or n < 0:
                return None
            return n
        return None
    except MemoryError:
        raise
    except Exception as exc:                # the numeric protocol failed: malformed data, never an escape from predict()
        return _protocol_error(exc)


def _rejected_count_kind(value) -> str:
    """Why ``_exact_count`` rejected an ORIGINAL object, decided on that object with its own arithmetic
    (traffic-count-validation-1): ``"nonfinite"`` (a real that is NaN / +-inf), ``"negative"`` (a real
    that is ``< 0`` BEFORE any conversion: ``numpy.longdouble('-1e-4000')`` is negative although ``float()``
    of it is ``-0.0``), else ``"not_exact_int"`` (bool, fractional, str, array, null, mapping ...).  Any
    exception the object's protocol raises classifies it as ``not_exact_int``; ``MemoryError`` propagates."""
    import numbers
    try:
        if isinstance(value, bool) or getattr(getattr(value, "dtype", None), "kind", None) == "b":
            return "not_exact_int"
        if isinstance(value, numbers.Real):
            if value != value or value == float("inf") or value == float("-inf"):
                return "nonfinite"
            if value < 0:
                return "negative"
        return "not_exact_int"
    except MemoryError:
        raise
    except Exception:
        return "not_exact_int"


def _validate_counts(analysis: Dict[str, Any]):
    """Validate EVERY raw MAC count field of a loaded analysis, unconditionally.

    Returns ``(counts, malformed)``: ``counts`` holds the validated Python ints
    (``total`` — 0 when ``total.macs`` is absent; ``layers`` — only layers whose
    ``macs`` key is present; ``aggregate`` — ``total.macs_by_compute_dtype``),
    ``malformed`` maps the field path of every rejected field to its RAW value,
    verbatim; ``counts["protocol_errors"]`` maps the paths whose numeric protocol
    RAISED during validation to the exception (``"<Type>: <message>"``).  An absent key is not malformed (a layer without a ``macs`` key is
    simply not a count; ``total.macs`` absent is 0; ``macs_by_compute_dtype``
    absent is "no aggregate"); a PRESENT key with a non-count value is — a present
    ``macs: null``, and a ``macs_by_compute_dtype`` that is present but null or not
    a mapping (malformed as a whole).
    """
    total = analysis.get("total") or {}
    malformed: Dict[str, Any] = {}
    protocol_errors: Dict[str, str] = {}
    counts: Dict[str, Any] = {"total": 0, "layers": {}, "aggregate": {}, "protocol_errors": protocol_errors}

    def _check(path, raw):
        errs: list = []
        n = _exact_count(raw, errors=errs)
        if n is None:
            malformed[path] = raw
            if errs:
                protocol_errors[path] = errs[0]
        return n

    if "macs" in total:
        n = _check("total.macs", total["macs"])
        if n is not None:
            counts["total"] = n
    layers_map = analysis.get("layers") or {}
    if isinstance(layers_map, dict):
        for lid, layer in layers_map.items():
            if isinstance(layer, dict) and "macs" in layer:
                n = _check(f"layers.{lid}.macs", layer["macs"])
                if n is not None:
                    counts["layers"][str(lid)] = n
    if "macs_by_compute_dtype" in total:
        # key PRESENCE is what matters: an absent aggregate is allowed, a present
        # null or non-mapping value is malformed as a whole (count-null-aggregate-1)
        raw_aggregate = total["macs_by_compute_dtype"]
        if not isinstance(raw_aggregate, dict):
            malformed["total.macs_by_compute_dtype"] = raw_aggregate
        else:
            for k, v in raw_aggregate.items():
                n = _check(f"total.macs_by_compute_dtype.{k}", v)
                if n is not None:
                    counts["aggregate"][str(k)] = n
    return counts, malformed


_RECEIPT_MAX_DEPTH = 32


def _receipt_safe(value, _seen=None, _depth=0):
    """A raw count value normalised to EXACT built-in types that YAML (SafeDumper)
    and JSON carry natively (count-integer-types-1, count-real-serialization-1).

    ``type(x) is`` ``int`` / ``float`` / ``str`` / ``bool`` / ``NoneType`` pass
    through unchanged (NaN/inf included).  Everything else is converted, never
    returned as-is: a numpy scalar goes through ``.item()`` and then the same
    rule (a numpy bool becomes the built-in bool, on every NumPy version); an ``int`` subtype (not bool) becomes
    ``int(x)``; a ``float`` subtype or any other ``numbers.Real`` becomes
    ``float(x)`` only when that round-trips exactly in the type's own arithmetic
    (``numpy.float32(0.5)`` → ``0.5``; a long double or ``Fraction`` a float cannot
    carry keeps its full ``repr``); a ``str`` subtype becomes ``str(x)``; mappings
    and sequences are converted element-wise; every other object, and any value
    whose conversion raises, is recorded as its ``repr`` (or a placeholder when
    even ``repr`` fails); a ``repr`` that returns a ``str`` SUBCLASS is normalised
    to the exact built-in ``str`` as well (receipt-repr-subtype-1).  A numpy
    scalar whose ``.item()`` is not a built-in (long double, complex long double,
    void, datetime64 ...) is normalised EXPLICITLY -- an exact float when the round
    trip is exact, else numpy's positional digit string -- never through ``repr``
    (numpy2-longdouble-1).
    ``MemoryError`` is never swallowed.

    A ``str`` subtype is forced to a GENUINE built-in ``str`` through the base
    type's own ``__str__`` (a subtype whose ``__str__`` returns itself must not
    survive: receipt-str-subtype-1); the result is asserted ``type(x) is str``
    or the value falls back to its placeholder.  Containers are expanded with an
    identity set and a depth cap (receipt-cycle-1): a cycle is recorded as
    ``<cycle: <type> id=N>`` and nesting beyond ``_RECEIPT_MAX_DEPTH`` as
    ``<depth N truncated>``, so a self-referential malformed value yields a small
    receipt that the default ``yaml.safe_load`` can read."""
    import numbers

    def placeholder_for(x):
        return f"<unrepresentable {type(x).__name__}>"

    def _repr(x):
        placeholder = placeholder_for(x)
        try:
            out = repr(x)
        except MemoryError:
            raise
        except Exception:
            return placeholder
        if type(out) is not str:
            # a __repr__ may return a str SUBCLASS (receipt-repr-subtype-1): force the genuine
            # built-in type through the base type's own conversion, exactly as the str branch does
            try:
                if not isinstance(out, str):
                    return placeholder
                out = str.__str__(out)
                if type(out) is not str:
                    out = "".join(str.__getitem__(out, i) for i in range(str.__len__(out)))
            except MemoryError:
                raise
            except Exception:
                return placeholder
        assert type(out) is str, type(out)          # the receipt contract: exact built-in str, never a subclass
        return out

    try:
        if value is None or type(value) in (bool, int, float, str):
            return value
        if hasattr(value, "item") and hasattr(value, "dtype"):
            # EVERY numpy scalar (bool_, integer, floating, str_, bytes_ ...) is normalised through .item() BEFORE
            # any repr/str path (CI-FAIL-26: NumPy 2.x spells repr(np.True_) as 'np.True_' and names the type
            # 'bool', so no name test and no repr may be relied on).  .item() is used when it round-trips exactly
            # in the scalar's own arithmetic; a value it cannot carry (a fractional long double) is recorded as
            # str(value): full precision and no module prefix on every NumPy version.
            item = value.item()
            if type(item) in (bool, int, float, str, bytes):
                try:
                    lossless = bool(type(value)(item) == value) or (item != item and value != value)
                except MemoryError:
                    raise
                except Exception:
                    lossless = False
                if lossless:
                    return _receipt_safe(item)
            # numpy2-longdouble-1: .item() of a long double / complex long double / void / datetime64 ...
            # is NOT a built-in (it returns the same numpy type), so the value is normalised EXPLICITLY -- a float
            # when the float round trip is exact in the scalar's own arithmetic, else a plain digit string from
            # numpy's own positional formatter (unique shortest repr, no module prefix on any NumPy version) --
            # and never through repr().  Every path asserts an exact built-in type.
            kind = getattr(getattr(value, "dtype", None), "kind", None)
            try:
                if kind == "f":
                    f = float(value)
                    if f == value:
                        return f
                elif kind in ("i", "u"):
                    return int(value)
            except MemoryError:
                raise
            except Exception:
                pass
            try:
                import numpy as _np
                if kind == "f":
                    out = _np.format_float_positional(value, unique=True, trim="-")
                elif kind == "c":
                    out = (_np.format_float_positional(value.real, unique=True, trim="-") + "+"
                           + _np.format_float_positional(value.imag, unique=True, trim="-") + "j")
                else:
                    out = str(value)
            except MemoryError:
                raise
            except Exception:
                try:
                    out = str(value)                     # numpy's own str(): digits only, never a module prefix
                except MemoryError:
                    raise
                except Exception:
                    return placeholder_for(value)
            out = str.__str__(out) if isinstance(out, str) and type(out) is not str else out
            if type(out) is not str or out.startswith("np."):
                return placeholder_for(value)
            return out
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            f = float(value)
            if f == value:              # exact round trip in the type's own arithmetic
                return f
            return _repr(value)
        if isinstance(value, str):
            out = str.__str__(value)                 # base-type conversion: a sticky subtype __str__ is bypassed
            if type(out) is not str:                     # defensive: rebuild from the base type's own characters
                out = "".join(str.__getitem__(value, i) for i in range(str.__len__(value)))
            if type(out) is not str:
                return _repr(value)
            return out
        if isinstance(value, (dict, list, tuple)):
            seen = _seen if _seen is not None else set()
            if id(value) in seen:
                return f"<cycle: {type(value).__name__} id={id(value)}>"
            if _depth >= _RECEIPT_MAX_DEPTH:
                return f"<depth {_RECEIPT_MAX_DEPTH} truncated: {type(value).__name__}>"
            seen = seen | {id(value)}
            if isinstance(value, dict):
                return {(k if type(k) is str else str(_receipt_safe(k, seen, _depth + 1))): _receipt_safe(v, seen, _depth + 1) for k, v in value.items()}
            return [_receipt_safe(v, seen, _depth + 1) for v in value]
        return _repr(value)
    except MemoryError:
        raise
    except Exception:
        return _repr(value)


class PerfProjectionUnavailable(RuntimeError):
    """The perf projection cannot be done on this architecture profile.

    Raised (Jenny's decision, 2026-10-26: "for H100 without nvfp4 raise error
    saying the perf projection cannot be done") when MAC-BEARING work is declared
    at a compute dtype — a layer's ``compute_dtype`` with ``macs > 0`` in the
    analysis (``torch._scaled_mm`` layers), or MACs the total's
    ``macs_by_compute_dtype`` names at that dtype — that has NO tensor-core rate
    (``MAC_per_cycle_<dtype>_tc``) in the selected architecture profile.  A
    quantized dtype that is only stored/moved (zero-MAC bucket, a copy) needs no
    rate; the workload ``metadata.yaml`` declaration is evidence recorded on the
    refusal, never a trigger by itself (mac-scope-1, LAYER-EVIDENCE-2).  No
    prediction is produced: neither a priced one at another dtype's rate nor
    an incomplete-marked one; the caller gets this exception and
    ``perf_<arch>.REFUSED.yaml`` is left in the output directory instead of
    ``perf_<arch>.yaml``.  Profiles that declare the rate (B200 declares
    ``MAC_per_cycle_nvfp4_tc``) are priced as before; the analysis (MAC /
    traffic accounting) is unaffected.
    """

    def __init__(self, arch: str, dtypes, layers: Dict[str, str], *, source: str = "analysis",
                 layer_macs: Optional[Dict[str, int]] = None, unbound_macs: Optional[Dict[str, int]] = None,
                 metadata_declared: Optional[Dict[str, Any]] = None, reason: str = "perf_projection_unavailable",
                 disagreement: Optional[Dict[str, Any]] = None, total_macs: Optional[int] = None, macs_typed: Optional[Dict[str, Any]] = None,
                 malformed_counts: Optional[Dict[str, Any]] = None, malformed_protocol_errors: Optional[Dict[str, str]] = None,
                 invalid_rate: Optional[Dict[str, Any]] = None, derived: Optional[Dict[str, Any]] = None):
        # reasons: "perf_projection_unavailable" (no rate) | "conflicting compute-dtype evidence" | "non-conserving compute-dtype split"
        #          | "compute_dtype_count_malformed" | "count_not_representable_as_float" | "rate_invalid:<profile key>"
        #          (rate-float-boundary-1) | "derived_quantity_not_finite" (a cycle / intensity / seconds value is inf or nan)
        #          | "traffic_not_exact_int" / "count_not_exact_int" (a raw traffic / count total that is not an exact non-negative
        #          integer on the ORIGINAL object: bool, fractional, string, array ...) | "traffic_negative" / "count_negative"
        #          | "traffic_negative_final" (a FINAL re-priced traffic total is negative; signed intermediate deltas may be)
        #          (traffic-count-validation-1)
        self.reason = reason
        # the architecture-profile value that is not a usable rate (name, raw value, profile, requirement), receipt-safe
        self.invalid_rate = {k: _receipt_safe(v) for k, v in invalid_rate.items()} if invalid_rate else None
        # a derived quantity (cycles, intensity, seconds) that is not finite: name, value and the inputs that produced it
        self.derived = ({"name": str(derived.get("name")), "value": _receipt_safe(derived.get("value")),
                         "inputs": {str(k): _receipt_safe(v) for k, v in (derived.get("inputs") or {}).items()}} if derived else None)
        self.disagreement = dict(disagreement) if disagreement else None   # dtype -> {aggregate_macs | aggregate: "absent", layer_macs_bound, total[, bucket_sum, layers]}
        # key (field path) -> the offending RAW value, verbatim (never truncated or
        # rounded); kept serializable so no receipt/summary writer can fail on it
        self.malformed_counts = {k: _receipt_safe(v) for k, v in malformed_counts.items()} if malformed_counts else None
        # field path -> "<ExceptionType>: <message>" for the fields whose numeric protocol raised during validation
        self.malformed_protocol_errors = {str(k): str(v) for k, v in malformed_protocol_errors.items()} if malformed_protocol_errors else None
        self.total_macs = total_macs
        self.macs_typed = macs_typed            # issue #58: the aggregate's typed components, so the refusal receipt never records a bare aggregate
        self.arch = str(arch)
        self.dtypes = sorted({str(d) for d in ([dtypes] if isinstance(dtypes, str) else dtypes)})
        self.dtype = self.dtypes[0] if len(self.dtypes) == 1 else "+".join(self.dtypes)
        self.layers = dict(layers)                      # layer id -> declared compute dtype (MAC-bearing layers only)
        self.layer_macs = {k: int(v) for k, v in (layer_macs or {}).items()}   # layer id -> its MACs at that dtype
        self.unbound_macs = {k: int(v) for k, v in (unbound_macs or {}).items()}  # dtype -> MACs the total names but no layer carries
        self.metadata_declared = dict(metadata_declared) if metadata_declared else None  # the workload sidecar's declaration, if any
        self.source = source                            # "analysis" (per-layer compute_dtype) or "analysis_total" (split only)
        self.missing_keys = [f"MAC_per_cycle_{d}_tc" for d in self.dtypes] if reason == "perf_projection_unavailable" else []
        self.macs_affected = sum(self.layer_macs.values()) + sum(self.unbound_macs.values())
        parts = []
        if reason.startswith("rate_invalid:") or reason.startswith("rate_protocol_error:"):
            r = self.invalid_rate or {}
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — the architecture profile value "
                            f"{r.get('name')!r} = {r.get('value')!r} is not a usable rate ({r.get('requirement')}"
                            + (f"; its numeric protocol raised {r.get('protocol_error')}" if r.get("protocol_error") else "") + "); "
                            f"no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason in ("traffic_negative", "count_negative", "traffic_not_exact_int", "count_not_exact_int"):
            offending = sorted(k for k in (malformed_protocol_errors or {})) or sorted(malformed_counts or {})
            what = ("is negative" if reason.endswith("_negative") else
                    "is not an exact non-negative integer (bool, fractional, string, array and null are rejected on the ORIGINAL object; nothing is rounded or coerced)")
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — {', '.join(offending)} {what} "
                            f"({', '.join(f'{k} = {_receipt_safe(v)!r}' for k, v in sorted((malformed_counts or {}).items()) if k in offending)}); "
                            f"a count or traffic total is an exact non-negative integer and the zero-byte intensity sentinel applies to exactly zero bytes only; "
                            f"no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason == "traffic_negative_final":
            d = self.derived or {}
            inputs = ", ".join(f"{k} = {v!r}" for k, v in sorted((d.get("inputs") or {}).items()))
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — {d.get('name')} evaluates to {d.get('value')!r} "
                            f"after the declared-boundary re-pricing ({inputs}); a final traffic total cannot be negative (a signed "
                            f"intermediate delta may be); no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason == "derived_quantity_not_finite":
            d = self.derived or {}
            inputs = ", ".join(f"{k} = {v!r}" for k, v in sorted((d.get("inputs") or {}).items()))
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — {d.get('name')} evaluates to {d.get('value')!r} "
                            f"from the accepted counts and rates ({inputs}); no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason in ("scalar_statement_component_malformed", "macs_typed_component_malformed", "macs_typed_malformed", "typed_components_not_conserving"):
            what = (f"the typed MAC record / component {', '.join(sorted(malformed_counts or {}))} is not a mapping of exact non-negative integer counts "
                    f"({', '.join(f'{k} = {_receipt_safe(v)!r}' for k, v in sorted((malformed_counts or {}).items()))}); a PRESENT malformed record refuses, only an ABSENT total.macs_typed is priced whole"
                    if reason != "typed_components_not_conserving" else
                    f"the typed MAC components do not conserve when recomputed from their exact values (the stored components_sum_to_aggregate flag is not trusted): "
                    f"{', '.join(f'{k} = {v!r}' for k, v in sorted((disagreement or {}).items()))}")
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — {what}; issue #58 convention B excludes "
                            f"macs_typed.scalar_statement_multiplies from the tensor-core term, so that count enters pricing and is validated "
                            f"like every other count; no numeric SOL is produced")
            super().__init__(self.message)
            return
        if reason == "count_not_representable_as_float":
            offending = sorted(k for k in (malformed_protocol_errors or {})) or sorted(malformed_counts)
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — the count(s) {', '.join(offending)} cannot be carried "
                            f"by float arithmetic (OverflowError / non-finite); every exact count is kept in this receipt "
                            f"({', '.join(f'{k} = {_receipt_safe(v)!r}' for k, v in sorted(malformed_counts.items()))}) and no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason == "compute_dtype_count_malformed":
            def _one(k, v):
                # receipt-str-subtype-1: the message is built from the receipt-safe
                # form and NEVER fails -- a value whose rendering raises gets a placeholder
                try:
                    shown = repr(_receipt_safe(v))
                except MemoryError:
                    raise
                except Exception:
                    shown = f"<unrenderable {type(v).__name__}>"
                err = self.malformed_protocol_errors.get(k) if self.malformed_protocol_errors else None
                return f"{k} = {shown}" + (f" ({err})" if err else "")
            detail = "; ".join(_one(k, v) for k, v in sorted(malformed_counts.items(), key=lambda kv: str(kv[0])))
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — counts must be exact non-negative "
                            f"integers, found {detail}; nothing is truncated or rounded and no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if reason != "perf_projection_unavailable":
            def _agg(v):
                return f"aggregate {v['aggregate_macs']:,} MACs" if "aggregate_macs" in v else "aggregate absent"
            detail = "; ".join(f"{dt}: {_agg(v)} vs {v['layer_macs_bound']:,} MACs in layers"
                               f"{' ' + ', '.join(sorted(l for l, d in self.layers.items() if d == dt)) if any(d == dt for d in self.layers.values()) else ''}"
                               for dt, v in sorted((self.disagreement or {}).items()))
            self.message = (f"perf projection cannot be done: {reason} on {self.arch} — {detail}"
                            + (f" (total {self.total_macs:,} MACs)" if self.total_macs is not None else "")
                            + "; the analysis is inconsistent and no numeric SOL is produced (ruling 2026-10-26)")
            super().__init__(self.message)
            return
        if self.layers:
            parts.append("layers: " + ", ".join(f"{lid} ({dt}, {self.layer_macs.get(lid, 0):,} MACs)" for lid, dt in sorted(self.layers.items())))
        if self.unbound_macs:
            parts.append("total.macs_by_compute_dtype names " + ", ".join(f"{dt}: {m:,} MACs" for dt, m in sorted(self.unbound_macs.items())) + " that no layer carries")
        if self.metadata_declared:
            parts.append(f"workload metadata declares {self.metadata_declared.get('orig_dtypes')!r}")
        self.message = (f"perf projection cannot be done: {self.arch} declares no {self.dtype} tensor-core rate "
                        f"({', '.join(self.missing_keys)} missing from the architecture profile); " + "; ".join(parts))
        super().__init__(self.message)

    def receipt(self) -> Dict[str, Any]:
        return {"status": "refused", "reason": self.reason, "message": self.message, "arch": self.arch,
                "dtype": self.dtype, "missing_keys": list(self.missing_keys), "layers": dict(self.layers),
                "layer_macs": dict(self.layer_macs), "macs_affected": int(self.macs_affected),
                **({"unbound_macs": dict(self.unbound_macs)} if self.unbound_macs else {}),
                **({"compute_dtype_evidence_disagreement": dict(self.disagreement)} if self.disagreement else {}),
                **({"malformed_counts": dict(self.malformed_counts)} if self.malformed_counts else {}),
                **({"malformed_protocol_errors": dict(self.malformed_protocol_errors)} if self.malformed_protocol_errors else {}),
                **({"invalid_rate": dict(self.invalid_rate)} if self.invalid_rate else {}),
                **({"derived_not_finite": dict(self.derived)} if self.derived and self.reason != "traffic_negative_final" else {}),
                **({"traffic_final_negative": dict(self.derived)} if self.derived and self.reason == "traffic_negative_final" else {}),
                **({"total_macs": int(self.total_macs)} if self.total_macs is not None else {}),
                **({"macs_typed": dict(self.macs_typed)} if isinstance(self.macs_typed, dict) else {}),   # issue #58
                "metadata_declared": self.metadata_declared, "source": self.source}


# Reasons whose layers contribute 0 MACs (their work is EXCLUDED from every
# compute cycle, not accounted from the stored shapes), with the wording the
# prediction qualification uses for each (wording-1).
_ZEROED_WORK = {
    "einsum_operand_evidence_unresolved": "operand shapes could not be established",
    "scaled_mm_evidence_unresolved": ("a scaled_mm whose operand dtypes, packing or output could not be bound: "
                                      "its MACs are excluded, not derived from the stored shapes"),
    "op_unrecognised": ("an operation with no einsum handler and no recognised family (the layer's "
                        "unrecognised_op names it): its work is excluded, not estimated (issue #50)"),
}
_ZEROED_REASONS = frozenset(_ZEROED_WORK)


def _rate_class(rate_key: str) -> str:
    """The hardware rate CLASS an applied ``MAC_per_cycle_*`` key denotes (issue #58, P2-rate-class-1): derived from the
    key the fallback chain actually selected, never assumed from the workload precision."""
    if rate_key.endswith("_tc"):
        return "tensor_core"
    if rate_key.endswith("_sm"):
        return "sm"
    if rate_key.startswith("MAC_per_cycle (default 1.0"):
        return "default_unit"
    return "unknown"


_RATE_CLASS_WORDS = {"tensor_core": "a tensor-core rate", "sm": "the CUDA-core (SM) rate", "default_unit": "the unit default rate 1.0 (no MAC rate key in the profile)", "unknown": "a rate of unknown class"}


SOL_CONVENTION: Dict[str, Any] = {
    # issue #58 convention B (bound convention): what the headline SOL number IS and is NOT.
    "id": "issue58-B",
    "headline": "optimistic contraction-and-memory bound",
    "statement": ("runtime = max(compute_cycles, memory_cycles).  The tensor-core term compute_cycles prices the analysis' contraction MACs "
                  "(quantised-dtype buckets at their tensor-core rates, the workload-precision remainder at the applied MAC rate) plus any "
                  "other non-contraction MACs the analyzer records (macs_typed.non_contraction_other; retained in the term as before).  "
                  "Excluded from compute_cycles: the multiply statements of non-einsum kernels that the per-statement classifier counts as "
                  "MACs (macs_typed.scalar_statement_multiplies) and other_ops.  Both excluded counts are retained in the receipt; the "
                  "existing SM diagnostic compute_sm_cycles prices other_ops only (other_ops / MAC_per_cycle_fp32_sm, unchanged) and does "
                  "not include the scalar statement multiplies."),
    "limits": ["not a bound accounting for all compute resources: the excluded SM-class statement work (scalar statement multiplies, other_ops) has no timing term in the headline",
               "the excluded work is not shown to be hidden by memory traffic or to be free; the exclusion is a bound convention, not a proof",
               "the per-statement MAC classification, every count, tolerance, tie, reference, cap and selection rule are unchanged"],
    "not_claimed": ["scalar work proven hidden", "scalar work free", "elementwise work absorbed by the memory term", "a bound over all compute resources"],
}


def _typed_components(total: Dict[str, Any], total_macs: int, macs_default: int):
    """issue #58 convention B, round 2 (P2-conservation-1 / TYPED-PRESENCE-1): the typed MAC components as exact ints, validated
    and re-conserved HERE, or a refusal description.

    Returns ``(scalar_excluded, record)``.  ``record`` carries ``applied`` and, when applied, the exact breakdown that enters the
    tensor-core term (``contraction_wp``, ``non_contraction_other``).  Rules:
      * ``total.macs_typed`` ABSENT: legacy path -- nothing excluded, ``applied: False`` with an accurate reason;
      * PRESENT but not a mapping (null / false / [] / string ...): refuse ``macs_typed_malformed`` with the path and the raw value;
      * every component that enters pricing (aggregate, contraction.macs, each contraction.by_compute_dtype bucket, scalar count,
        non_contraction_other count when present) must be an exact non-negative integer (``_exact_count``): refuse
        ``macs_typed_component_malformed`` / ``scalar_statement_component_malformed`` with the path and raw value;
      * conservation is RECOMPUTED, never read from the stored ``components_sum_to_aggregate`` flag: aggregate == total.macs,
        contraction + scalar + other == aggregate, sum(by_compute_dtype) == contraction.macs, and the default-rate remainder
        identity contraction[workload_precision] + scalar + other == macs_default (total.macs minus the quantised MACs priced at their
        own rates); any violation -- or a stored flag that is not True while the arithmetic conserves (a contradiction in the
        record) -- refuses ``typed_components_not_conserving`` with every original value in the receipt."""
    if "macs_typed" not in total:
        return 0, {"applied": False, "reason": ("total.macs_typed is absent (the analysis predates typed MAC components): the default-rate "
                                                 "remainder is priced whole, as before convention B; the scalar share is not separable here")}
    typed = total["macs_typed"]
    if not isinstance(typed, dict):
        return 0, {"refuse": True, "reason": "macs_typed_malformed", "malformed_counts": {"total.macs_typed": typed}}
    vals: Dict[str, int] = {}; malformed: Dict[str, Any] = {}; protocol: Dict[str, str] = {}
    def need(path, raw, reason):
        errs: list = []; n = _exact_count(raw, errors=errs)
        if n is None:
            malformed[path] = raw
            if errs: protocol[path] = errs[0]
            return reason
        vals[path] = n; return None
    r = need("total.macs_typed.aggregate", typed.get("aggregate"), "macs_typed_component_malformed")
    contraction = typed.get("contraction")
    if not isinstance(contraction, dict): r = r or "macs_typed_component_malformed"; malformed["total.macs_typed.contraction"] = contraction
    else:
        r = r or need("total.macs_typed.contraction.macs", contraction.get("macs"), "macs_typed_component_malformed")
        buckets = contraction.get("by_compute_dtype")
        if not isinstance(buckets, dict): r = r or "macs_typed_component_malformed"; malformed["total.macs_typed.contraction.by_compute_dtype"] = buckets
        else:
            for k, v in buckets.items(): r = r or need(f"total.macs_typed.contraction.by_compute_dtype.{k}", v, "macs_typed_component_malformed")
    scalar = typed.get("scalar_statement_multiplies")
    if not isinstance(scalar, dict) or "macs" not in scalar:
        r = r or "scalar_statement_component_malformed"; malformed["total.macs_typed.scalar_statement_multiplies.macs"] = (scalar if not isinstance(scalar, dict) else "<macs key absent>")
    else: r = r or need("total.macs_typed.scalar_statement_multiplies.macs", scalar["macs"], "scalar_statement_component_malformed")
    other = typed.get("non_contraction_other")
    if other is not None:
        if not isinstance(other, dict) or "macs" not in other: r = r or "macs_typed_component_malformed"; malformed["total.macs_typed.non_contraction_other.macs"] = (other if not isinstance(other, dict) else "<macs key absent>")
        else: r = r or need("total.macs_typed.non_contraction_other.macs", other["macs"], "macs_typed_component_malformed")
    if malformed:
        return 0, {"refuse": True, "reason": r, "malformed_counts": malformed, "protocol_errors": protocol or None}
    agg = vals["total.macs_typed.aggregate"]; c = vals["total.macs_typed.contraction.macs"]; sc = vals["total.macs_typed.scalar_statement_multiplies.macs"]
    oth = vals.get("total.macs_typed.non_contraction_other.macs", 0)
    bucket_vals = {k[len("total.macs_typed.contraction.by_compute_dtype."):]: v for k, v in vals.items() if k.startswith("total.macs_typed.contraction.by_compute_dtype.")}
    c_wp = bucket_vals.get("workload_precision", 0)
    evidence = {"total_macs": int(total_macs), "macs_typed": typed, "aggregate": agg, "contraction_macs": c, "contraction_by_compute_dtype": bucket_vals, "scalar_statement_multiplies": sc,
                "non_contraction_other_macs": oth, "components_sum": c + sc + oth, "default_rate_remainder": int(macs_default),
                "default_rate_components_sum": c_wp + sc + oth, "components_sum_to_aggregate_flag": typed.get("components_sum_to_aggregate")}
    conserves = (agg == total_macs and c + sc + oth == agg and sum(bucket_vals.values()) == c and c_wp + sc + oth == macs_default)
    # the stored flag is never accepted as proof (the arithmetic above decides); a flag that is not True while the components
    # conserve is itself a contradiction in present evidence and refuses as well
    if not conserves or typed.get("components_sum_to_aggregate") is not True:
        return 0, {"refuse": True, "reason": "typed_components_not_conserving", "disagreement": {**evidence, "recomputed_conservation": conserves}}
    return sc, {"applied": True, "contraction_wp": c_wp, "non_contraction_other": oth, "evidence": evidence}


def _default_rate_remainder_note(total: Dict[str, Any], macs_default: int, rate_key: str, requested_key: Optional[str] = None, precision: Optional[str] = None,
                                 scalar_excluded: int = 0) -> Dict[str, Any]:
    """issue #58: decompose the default-rate MAC remainder from the analysis' typed components (informational).  The displayed
    rate class is derived from the APPLIED key (`rate_key`, exact); `requested_key` is the key the workload precision asked
    for; `rate_is_fallback` says whether the fallback chain substituted another key (P2-rate-class-1)."""
    typed = total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None
    cls = _rate_class(rate_key); words = _RATE_CLASS_WORDS[cls]
    fallback = requested_key is not None and requested_key != rate_key
    rec: Dict[str, Any] = {"macs": int(macs_default), "rate_key": rate_key, "rate_class": cls,
                           **({"requested_rate_key": requested_key} if requested_key else {}), "rate_is_fallback": bool(fallback),
                           **({"precision": precision} if precision else {})}
    how = (f" ({words}; the profile declares no {requested_key}, so the fallback chain selected {rate_key})" if fallback else f" ({words})")
    if typed is None:
        rec["macs_priced_in_tensor_core_term"] = int(macs_default)
        rec["note"] = (f"{macs_default:,} MACs priced at {rate_key}{how}; the analysis predates typed MAC components, so the share of "
                       "scalar multiply statements (non-einsum kernels) inside this remainder is not separable here and nothing is "
                       "excluded from the tensor-core timing term (issue #58 convention B needs the typed components); this is a bound "
                       "convention, not a claim that any work is hidden or free")
        return rec
    contraction_default = int((typed.get("contraction", {}).get("by_compute_dtype", {}) or {}).get("workload_precision", 0))
    scalar = typed.get("scalar_statement_multiplies", {}) or {}
    other = typed.get("non_contraction_other", {}) or {}
    rec.update({"contraction_macs_at_workload_precision": contraction_default,
                "scalar_statement_multiplies": int(scalar.get("macs", 0)), "scalar_statement_layers": dict(scalar.get("layers", {}) or {}),
                **({"non_contraction_other_macs": int(other.get("macs", 0)), "non_contraction_other_layers": dict(other.get("layers", {}) or {})} if other.get("macs") else {}),
                "macs_priced_in_tensor_core_term": int(macs_default) - int(scalar_excluded),
                "note": (f"{macs_default:,} MACs in the default-rate remainder: {contraction_default:,} contraction MACs without a "
                         f"quantised compute dtype priced at {rate_key}{how} + {int(scalar.get('macs', 0)):,} scalar multiply statements of "
                         f"non-einsum kernels"
                         + (f" ({', '.join(f'{k} {int(v):,}' for k, v in sorted((scalar.get('layers') or {}).items()))})" if scalar.get("layers") else "")
                         + (f", of which {int(scalar_excluded):,} are excluded from the tensor-core timing term (issue #58 convention B: the "
                            "typed counts and the SM diagnostic are retained; the headline is an optimistic contraction-and-memory bound, "
                            "not a bound over all compute resources; the exclusion is a convention, not a proof that the scalar work is "
                            "absorbed by memory traffic)")
                         + (f"; priced at {rate_key}: {contraction_default:,} contraction MACs"
                            + (f" + {int(other.get('macs', 0)):,} other non-contraction MACs (retained in the term, as before)" if other.get("macs") else ""))
                         + {"tensor_core": "",
                            "sm": "; the contraction MACs at the workload precision are classified from real-einsum provenance only (the analyzer does not establish the execution unit) and are priced at the CUDA-core (SM) rate the fallback selected",
                            "default_unit": "; priced at the unit default rate because the profile declares no MAC rate key",
                            "unknown": "; the rate class of the applied key is not recognised"}[cls])})
    return rec


def _prediction_qualification(incomplete_layers) -> str:
    """Reason-specific wording for an incomplete prediction (see STATUS_CONTRACT.md).

    ``einsum_operand_evidence_unresolved``, ``scaled_mm_evidence_unresolved``
    and ``op_unrecognised`` layers contribute 0 MACs (their operand shapes, the
    scaled-mm operand dtypes / packing / output, or the operation itself could
    not be established: that work is excluded from the prediction, see
    ``_ZEROED_WORK``); every other recorded reason
    (unresolved provenance, unattached norm parameters, degraded capture, ...)
    keeps the layer's costs from the shapes that are known but leaves them
    unqualified.
    """
    # a layer may carry several causes joined by '+': every cause is qualified.  A compute
    # dtype without a tensor-core rate on the architecture is NOT a qualification: the
    # projection refuses (PerfProjectionUnavailable) before any prediction is built.
    reasons = set()
    for v in ((incomplete_layers or {}).values() if isinstance(incomplete_layers, dict) else ()):
        reasons.update(str(v).split("+"))
    zeroed = reasons & _ZEROED_REASONS
    other = reasons - _ZEROED_REASONS
    zeroed_text = "; ".join(f"{r}: {_ZEROED_WORK[r]}" for r in sorted(zeroed))
    if zeroed and not other:
        return ("partial: layers with unresolved operand evidence contribute 0 MACs (" + zeroed_text +
                "); their work is excluded from the compute cycles; runtimes are lower bounds of the known work")
    if other and not zeroed:
        return ("unqualified: incomplete layers are accounted from their known shapes but their "
                "accounting is not qualified (see incomplete_layers reasons)")
    return ("contains partial/unqualified accounting: evidence-unresolved layers contribute 0 MACs (" + zeroed_text +
            "; their work is excluded from the compute cycles), "
            "other incomplete layers are accounted but unqualified (see incomplete_layers reasons)")


class EinsumGraphPerfModel:
    """Compute SOL-style roofline predictions from `analysis.yaml`.

    Computes three performance models:
    - unfused: Each operation's roofline computed independently, summed
    - fused: Per-op roofline with intermediate tensors excluded
    - fused_prefetched: Single roofline for entire graph (best case)
    """

    def __init__(self, debug: bool = False, dtype_bytes: bool = False) -> None:
        self.debug = debug
        # When True and the analysis carries per-tensor-dtype byte totals,
        # price memory traffic at each tensor's own width (bool 1 B, fp32 4 B,
        # fp8 1 B ...) instead of one bytes_per_element for the whole graph.
        # Default False keeps the "precision is a modelling knob" behaviour
        # (e.g. pricing an fp32-traced model as fp16 end to end).
        self.dtype_bytes = dtype_bytes

    def predict(
        self,
        analysis_path: PathLike,
        output_dir: PathLike,
        *,
        arch_config: str = "H100_PCIe",
        precision: str = DEFAULT_PRECISION,
        copy_analysis: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Predict performance and write `perf_<arch>.yaml`.

        Args:
            analysis_path: Path to `analysis.yaml`.
            output_dir: Directory to write perf outputs into.
            arch_config: Architecture name (e.g., "H100_PCIe") or path to a YAML file.
            precision: Precision used for selecting MAC throughput keys.
            copy_analysis: If True, copy analysis into output dir as `analysis.yaml`.

        Returns:
            Perf dict or None on failure.
        """
        analysis_path = Path(analysis_path)
        out_dir = ensure_directory(output_dir)

        if not analysis_path.exists():
            if self.debug:
                logger.debug(f"analysis not found: {analysis_path}")
            return None

        try:
            with open(analysis_path) as f:
                analysis = yaml_safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                logger.debug(f"failed reading analysis: {exc}")
            return None

        if copy_analysis:
            try:
                dst = out_dir / "analysis.yaml"
                if analysis_path.resolve() != dst.resolve():
                    dst.write_text(analysis_path.read_text())
            except Exception:
                if self.debug:
                    logger.debug("failed copying analysis.yaml")

        arch = self._load_arch_config(arch_config)
        if not arch:
            if self.debug:
                logger.debug(f"failed loading arch config: {arch_config}")
            return None

        arch_name = str(arch.get("name") or Path(str(arch_config)).stem or "arch")

        # Check for quantization metadata (e.g., nvfp4/fp8 conversions)
        quant_metadata = self._load_quant_metadata(analysis_path)
        quant_mac_key = None
        quant_bpe = None
        quant_label = None
        if quant_metadata:
            quant_mac_key, quant_bpe, quant_label = self._resolve_quant_overrides(
                quant_metadata, arch, precision
            )
        # mac-scope-1: the sidecar's declaration ("a quantized dtype appears
        # somewhere in the inputs/reference", run_execbench_problem.detect_quant_dtypes)
        # is NOT compute-dtype evidence.  It is kept as evidence for the refusal
        # record and, when the profile has no rate for it, the MAC-rate override is
        # dropped (the workload precision's rate prices MACs no layer binds to a
        # quantized compute dtype; the bytes override stays — storage/traffic
        # dtypes need no tensor-core rate).  The refusal itself is bound to
        # MAC-bearing layers by the per-layer scan below.
        metadata_declared = None
        if quant_mac_key:
            metadata_declared = {"dtype": quant_mac_key[len("MAC_per_cycle_"):-len("_tc")], "orig_dtypes": quant_label,
                                 "rate_in_profile": arch.get(quant_mac_key) is not None, "key": quant_mac_key}
            if arch.get(quant_mac_key) is None:
                quant_mac_key = None

        total = analysis.get("total") or {}
        metadata = analysis.get("metadata") or {}
        # torchview capability record (issue #45): a degraded capture keeps its
        # marker on the perf output too.
        capture_capability = metadata.get("capture_capability")
        from solar.graph.torchview_capability import degraded_marker
        capture_degraded = degraded_marker(capture_capability) if capture_capability else None
        if total.get("unresolved_ops"):
            # A DIAGNOSTIC analysis (allow_unresolved_loop_counts=True)
            # carries layers whose work was never established; its MAC
            # totals are undercounts.  Never price them into SOL (review
            # I2): the annotation is evidence of a gap, not admission.
            raise ValueError(
                "Cannot predict performance: analysis carries unresolved "
                f"loop counts for {list(total['unresolved_ops'])} "
                f"({analysis_path}). Its totals are incomplete by "
                "construction; fix the DSL loop extents and re-run the "
                "analysis instead of pricing a diagnostic file."
            )

        # Get bytes_per_element: quant override > precision flag > metadata > default
        # The precision flag takes priority over analysis metadata because the user
        # may run perf with a different precision than the analysis was generated with.
        # the metadata width is a raw analysis value: it is validated (finite, > 0) in the rate layer below,
        # after the exact-int refusals, like every architecture rate (rate-float-boundary-1)
        bpe_raw_source: Optional[tuple] = None
        if quant_bpe is not None:
            bytes_per_element = quant_bpe
        elif precision in BYTES_PER_ELEMENT:
            bytes_per_element = BYTES_PER_ELEMENT[precision]
        else:
            bytes_per_element = None
            bpe_raw_source = ("metadata.bytes_per_element", metadata.get("bytes_per_element", 4))
        
        # count-coercion-1 (round 6): EVERY raw MAC count field is validated
        # here, at the loader boundary, BEFORE any conversion or arithmetic — the
        # total, each layer, each aggregate bucket — by the one module-level
        # validator; a malformed field refuses with a structured receipt naming the
        # field path and the raw value (no Python exception escapes predict()).
        # Validated counts are Python ints and stay ints through the conservation
        # decision below; floats enter only for rate arithmetic.
        counts, malformed = _validate_counts(analysis)
        if malformed:
            raw_aggregate = total.get("macs_by_compute_dtype")
            evidence_dtypes = ({str(l.get("compute_dtype")) for l in (analysis.get("layers") or {}).values()
                                if isinstance(l, dict) and isinstance(l.get("compute_dtype"), str)}
                               | ({str(k) for k in raw_aggregate} if isinstance(raw_aggregate, dict) else set()))
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, sorted(evidence_dtypes) or ["<none>"], {}, source="analysis",
                metadata_declared=metadata_declared, reason="compute_dtype_count_malformed", malformed_counts=malformed,
                malformed_protocol_errors=counts["protocol_errors"] or None))
        total_macs: int = counts["total"]                   # exact Python int (validated above)
        # NO raw count is converted to float here: FLOPs, other_ops, the element / byte totals and the
        # declared boundary bytes are converted only in the ONE guarded layer below, after every exact-int
        # refusal decision (conservation-float-order-1, total-float-boundary-1).


        # architecture values are read RAW here; nothing is converted until the rate layer below
        freq_ghz_raw = arch.get("freq_GHz", 1.0)
        dram_bw_raw = arch.get("DRAM_byte_per_cycle", 1.0)

        # MAC throughput: quant override takes priority over precision flag
        if quant_mac_key:
            mac_key = quant_mac_key
        else:
            mac_key = f"MAC_per_cycle_{precision}_tc"
        mac_per_cycle_raw = arch.get(mac_key); mac_rate_key_used = mac_key

        if mac_per_cycle_raw is None:
            # Fallback chain based on precision similarity
            if precision in ["bf16", "bfloat16"]:
                # BF16 has same throughput as FP16 on modern tensor cores
                mac_per_cycle_raw = arch.get("MAC_per_cycle_fp16_tc"); mac_rate_key_used = "MAC_per_cycle_fp16_tc"
            elif precision in ["fp16", "float16", "half"]:
                # Try FP16, then BF16
                mac_per_cycle_raw = arch.get("MAC_per_cycle_bf16_tc"); mac_rate_key_used = "MAC_per_cycle_bf16_tc"

            # Final fallback to FP32
            if mac_per_cycle_raw is None:
                if arch.get("MAC_per_cycle_fp32_tc") is not None:
                    mac_per_cycle_raw = arch.get("MAC_per_cycle_fp32_tc"); mac_rate_key_used = "MAC_per_cycle_fp32_tc"
                elif arch.get("MAC_per_cycle_fp32_sm") is not None:
                    mac_per_cycle_raw = arch.get("MAC_per_cycle_fp32_sm"); mac_rate_key_used = "MAC_per_cycle_fp32_sm"
                else:
                    mac_per_cycle_raw = 1.0; mac_rate_key_used = "MAC_per_cycle (default 1.0: no rate key in the profile)"

        sm_per_cycle_raw = arch.get("MAC_per_cycle_fp32_sm", 0)

        # Tensor-core cycles (matmul/conv MACs).  MACs whose operands were
        # recorded in a quantised compute dtype (a layer's ``compute_dtype`` with
        # macs > 0 — torch._scaled_mm — and/or the analysis total's optional
        # ``macs_by_compute_dtype`` aggregate) are priced at THAT dtype's
        # tensor-core rate when the architecture declares one
        # (MAC_per_cycle_<dtype>_tc).  When it declares none for a MAC-bearing
        # dtype the whole projection REFUSES (PerfProjectionUnavailable, decision
        # 2026-10-26) — no lower bound, no workload-precision pricing.  A dtype
        # with no MACs (storage / traffic only) needs no rate.
        #
        # an earlier fix round 3 (P2-1): the layer evidence is collected FIRST and
        # independently of the aggregate's keys — an absent aggregate, or one
        # that lacks the layers' dtype, never hides MAC-bearing layers; the
        # union of both sources is examined, and a positive remainder the
        # aggregate names beyond the layers is preserved as ``unbound_macs``
        # (P3-1).  Ruling (manager, 2026-10-26 05:03Z, round 4): when per-layer
        # MAC-bearing evidence and the aggregate BOTH exist and CONTRADICT each
        # other for the same dtype, or when the split is non-conserving (buckets
        # exceed the total / a derived remainder is negative), the projection
        # REFUSES with a structured receipt and NO numeric SOL — exactly like a
        # missing rate.  Absence is not contradiction: an absent aggregate with
        # consistent positive layers prices from the layers, a total-only record
        # prices from the total (with the unbound remainder on a missing rate),
        # absent layers with an aggregate price from the aggregate.
        layers_map = analysis.get("layers") or {}
        # compute-dtype evidence from the VALIDATED counts (exact Python ints)
        layer_dtype_macs: Dict[str, Dict[str, int]] = {}
        for lid, layer in layers_map.items():
            if isinstance(layer, dict) and isinstance(layer.get("compute_dtype"), str) and str(lid) in counts["layers"]:
                n = counts["layers"][str(lid)]
                if n > 0:
                    layer_dtype_macs.setdefault(layer["compute_dtype"], {})[str(lid)] = n
        macs_by_dtype: Dict[str, int] = dict(counts["aggregate"])
        compute_dtype_rates: Dict[str, Any] = {}
        rateless_layers: Dict[str, str] = {}
        rateless_layer_macs: Dict[str, int] = {}
        unbound_macs: Dict[str, int] = {}
        evidence_disagreement: Dict[str, Any] = {}
        priced_quantized_macs = 0
        eff_counts: Dict[str, int] = {}
        for dt in sorted(set(layer_dtype_macs) | set(macs_by_dtype)):
            dt_key = f"MAC_per_cycle_{dt}_tc"
            dt_rate = arch.get(dt_key)        # RAW profile value: only `is None` is ever asked of it here (rate-protocol-coverage-1)
            bound_layers = layer_dtype_macs.get(dt, {})
            bound = sum(bound_layers.values())
            agg_macs = macs_by_dtype.get(dt)                      # None when the aggregate lacks this dtype
            eff_macs = agg_macs if agg_macs is not None else bound
            eff_counts[dt] = eff_macs
            if agg_macs is not None and bound and agg_macs != bound:  # exact integer comparison
                # both sources exist and contradict each other: conflicting evidence
                evidence_disagreement[dt] = {"aggregate_macs": agg_macs, "layer_macs_bound": bound,
                                             "total": total_macs, "layers": sorted(bound_layers)}
            if eff_macs == 0 and not bound_layers:
                # a ZERO-MAC bucket -- a quantized tensor that is only stored / moved -- needs no tensor-core rate
                # WHATEVER the profile declares (mac-scope-1; zero-mac-rate-1): a declared rate, valid,
                # zero or malformed, is recorded receipt-safe as unused and is never admitted to the rate guard.
                # (eff_macs == 0 with MAC-bearing layers is the aggregate-vs-layers contradiction handled below.)
                compute_dtype_rates[dt] = {"key": dt_key, "MAC_per_cycle": None, "macs": eff_macs, "status": "traffic_only_no_rate_needed",
                                           **({"declared_rate_not_required": _receipt_safe(dt_rate)} if dt_rate is not None else {})}
                continue
            if dt_rate is not None:
                # exact int kept here: no count is converted to float before every
                # conflict / conservation refusal has run (conservation-float-order-1); the raw rate is
                # admitted through `to_rate` below, after those refusals, before any arithmetic or truth test
                compute_dtype_rates[dt] = {"key": dt_key, "MAC_per_cycle": dt_rate, "macs": eff_macs,   # RAW rate: validated in the rate layer
                                           "layer_macs_bound": bound}
                priced_quantized_macs += eff_macs
                continue
            # no rate for this compute dtype: refuse ONLY for MAC-bearing evidence
            # (mac-scope-1).  A zero-MAC bucket (a quantized tensor that is
            # only moved: storage/traffic dtype) needs no tensor-core rate.
            for lid, m in bound_layers.items():
                rateless_layers[lid] = dt
                rateless_layer_macs[lid] = m
            remainder = agg_macs - bound if agg_macs is not None else 0
            if remainder > 0:
                # the aggregate names MAC-bearing work at this dtype beyond what the
                # layers carry: still MAC evidence, preserved as the unbound remainder
                unbound_macs[dt] = remainder
            if not bound_layers and remainder <= 0:
                compute_dtype_rates[dt] = {"key": dt_key, "MAC_per_cycle": None, "macs": eff_macs,
                                           "status": "traffic_only_no_rate_needed"}
        # a contradiction on a dtype that HAS a rate is the refusal reason itself; on a
        # dtype WITHOUT a rate the missing-rate refusal below carries the disagreement
        # as additional evidence (its receipt keeps layers + unbound remainder)
        rated_conflicts = {dt: v for dt, v in evidence_disagreement.items() if arch.get(f"MAC_per_cycle_{dt}_tc") is not None}
        if rated_conflicts:
            conflict_layers = {lid: dt for dt in rated_conflicts for lid in layer_dtype_macs.get(dt, {})}
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, set(rated_conflicts), conflict_layers, source="analysis",
                layer_macs={lid: layer_dtype_macs[dt][lid] for lid, dt in conflict_layers.items()},
                metadata_declared=metadata_declared, reason="conflicting compute-dtype evidence",
                disagreement=rated_conflicts, total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None)))
        bucket_sum = sum(eff_counts.values())                     # exact: per dtype, the aggregate when present else the layer sum
        if bucket_sum > total_macs or (total_macs - priced_quantized_macs) < 0:   # exact Python ints, no float boundary
            # the split claims more MACs than the analysis total: non-conserving (exact,
            # no half-MAC allowance).  Evidence comes from BOTH sources per dtype, the
            # aggregate marked "absent" where only layers carry the dtype
            # (layer-only-receipt-1).
            evidence = {dt: {**({"aggregate_macs": macs_by_dtype[dt]} if dt in macs_by_dtype else {"aggregate": "absent"}),
                             "layer_macs_bound": sum(layer_dtype_macs.get(dt, {}).values()),
                             "layers": sorted(layer_dtype_macs.get(dt, {})),
                             "total": total_macs, "bucket_sum": bucket_sum} for dt in eff_counts}
            nc_layers = {lid: dt for dt in eff_counts for lid in layer_dtype_macs.get(dt, {})}
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, set(eff_counts), nc_layers, source="analysis" if nc_layers else "analysis_total",
                layer_macs={lid: layer_dtype_macs[dt][lid] for lid, dt in nc_layers.items()},
                metadata_declared=metadata_declared, reason="non-conserving compute-dtype split",
                disagreement=evidence, total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None)))
        macs_default = total_macs - priced_quantized_macs       # exact int remainder (>= 0 by the guard above)
        if rateless_layers or unbound_macs:
            # Jenny's decision (2026-10-26): no rate for a declared compute dtype
            # carrying MACs means NO projection on this architecture — not an
            # incomplete-marked prediction, not a lower bound.  Analysis is
            # untouched; B200, which declares the FP4 rate, prices these layers
            # above.  The sidecar's declaration, if any, rides along as evidence
            # (layer-evidence-2: the layer scan always runs first).
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, set(rateless_layers.values()) | set(unbound_macs), rateless_layers,
                source="analysis" if rateless_layers else "analysis_total",
                layer_macs=rateless_layer_macs, unbound_macs=unbound_macs, metadata_declared=metadata_declared,
                disagreement={dt: v for dt, v in evidence_disagreement.items() if dt not in rated_conflicts} or None,
                total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None)))
        # ── issue #58 convention B (bound convention): the tensor-core timing term prices CONTRACTION MACs only.
        # The multiply statements of non-einsum kernels that the per-statement classifier counts as MACs
        # (total.macs_typed.scalar_statement_multiplies) are excluded from compute_tc_cycles exactly as other_ops are; their
        # typed counts and the informational SM diagnostic are retained unchanged.  This is a bound convention: it does NOT
        # claim the scalar work is hidden or free.  The term still prices the analyzer's other non-contraction MACs
        # (macs_typed.non_contraction_other), as before.  An analysis WITHOUT total.macs_typed is priced whole (as before) and the
        # receipt says so; a PRESENT typed record that is malformed or does not conserve (recomputed, not read from its own flag)
        # REFUSES, since its counts now enter pricing (round 2: P2-conservation-1 / TYPED-PRESENCE-1).
        scalar_excluded, tc_term_record = _typed_components(total, total_macs, macs_default)      # exact Python ints, re-conserved here
        if tc_term_record.get("refuse"):
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                metadata_declared=metadata_declared, reason=tc_term_record["reason"], total_macs=total_macs,
                macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                malformed_counts=tc_term_record.get("malformed_counts"), malformed_protocol_errors=tc_term_record.get("protocol_errors"),
                disagreement=tc_term_record.get("disagreement")))
        macs_tc_default = macs_default - scalar_excluded          # exact int, >= 0 (scalar_excluded <= macs_default checked above)
        # ── ONE guarded conversion layer for every count that enters float arithmetic ──
        # Every conflict / conservation / missing-rate refusal above ran on exact Python ints.  Only
        # now, after acceptance, does ANY count enter float arithmetic, and every such count -- the MAC
        # total, the per-dtype buckets, the default remainder, FLOPs, other_ops, the element / byte
        # totals, the declared boundary bytes, and the byte products / sums derived from them -- passes
        # through ``to_float`` below: an integer a float cannot carry (OverflowError) or a non-finite
        # result (inf / nan) is a refusal with the exact integers in its receipt, never an escaping
        # exception (conservation-float-order-1, total-float-boundary-1).  No raw count is
        # converted anywhere else in predict() (the static audit in tests/test_scaled_mm_accounting.py
        # holds the module to this).
        exact_evidence = {"total.macs": total_macs,
                          **{f"total.macs_by_compute_dtype.{dt}": macs_by_dtype[dt] for dt in macs_by_dtype},
                          **{f"layers.{lid}.macs": counts["layers"][lid] for lid in counts["layers"]}}

        # ── Every raw traffic / count TOTAL (elements, bytes, declared boundary bytes, FLOPs, other_ops) is validated on
        # the ORIGINAL object by the one exact-count validator BEFORE any conversion (traffic-count-validation-1):
        # a bool, a fraction, a numeric string, an array or a null refuses (`traffic_not_exact_int` / `count_not_exact_int`),
        # a negative value refuses on the original object (`traffic_negative` / `count_negative`: a typed long double that is
        # negative but rounds to -0.0 as a float never reaches the zero-byte sentinel), NaN / +-inf keep
        # `count_not_representable_as_float`.  The exact Python int is returned; `to_float` converts it afterwards.
        def exact_total(name: str, raw: Any, *, traffic: bool) -> int:
            errs: list = []
            n = _exact_count(raw, errs)
            if n is not None:
                return n
            kind = _rejected_count_kind(raw)
            reason = ("count_not_representable_as_float" if kind == "nonfinite" else
                      f"{'traffic' if traffic else 'count'}_{kind}")
            detail = (errs[0] if errs else
                      f"{'negative' if kind == 'negative' else 'not a finite exact non-negative integer' if kind == 'nonfinite' else 'not an exact non-negative integer'} "
                      f"{'traffic total' if traffic else 'count'}: {type(raw).__name__}")
            self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                metadata_declared=metadata_declared, reason=reason, total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                malformed_counts={**exact_evidence, name: raw}, malformed_protocol_errors={name: detail}))
            raise AssertionError("unreachable: _refuse raises")  # pragma: no cover

        # ONE presence sweep: every KNOWN traffic / count total that is PRESENT in the analysis total (and every
        # declared-boundary field) is validated here, whether or not the selected pricing path reads it -- key
        # PRESENCE decides (a present `null` is malformed, like a present `macs: null`), never truthiness.  The one
        # documented exception: the analyzer writes `orojenesis_elements: null` as the legacy alias of
        # `unfused_elements`, so that null is "absent".  The pricing paths below read these exact ints only.
        _TRAFFIC_TOTAL_KEYS = ("unfused_elements", "orojenesis_elements", "fused_elements", "fused_prefetched_elements", "weight_elements",
                               "model_io_elements", "intermediate_elements", "orojenesis_bytes", "unfused_bytes", "fused_bytes",
                               "fused_prefetched_bytes", "weight_bytes", "model_io_bytes", "intermediate_bytes", "external_input_bytes",
                               "external_output_bytes", "model_io_bytes_declared", "external_io_bytes_declared_dedup")
        _COUNT_TOTAL_KEYS = ("flops", "other_ops")
        _BOUNDARY_KEYS = ("elements_raw", "bytes_raw_model_io", "footprint_elements_weight", "bytes_footprint_weight")
        boundary_declared = total.get("boundary_declared")
        if not isinstance(boundary_declared, dict):            # presence / type, never truthiness of a raw value
            boundary_declared = {}
        tot_exact: Dict[str, int] = {k: exact_total(f"total.{k}", total[k], traffic=True) for k in _TRAFFIC_TOTAL_KEYS
                                     if k in total and not (k == "orojenesis_elements" and total[k] is None)}
        tot_exact.update({k: exact_total(f"total.{k}", total[k], traffic=False) for k in _COUNT_TOTAL_KEYS if k in total})
        bd_exact: Dict[str, int] = {k: exact_total(f"total.boundary_declared.{k}", boundary_declared[k], traffic=True) for k in _BOUNDARY_KEYS if k in boundary_declared}

        # ── Architecture rates and widths: validated through the refusal path BEFORE any accepted count is
        # consumed (rate-float-boundary-1).  A rate must be a real number (not bool), finite as a float and
        # > 0 (the SM rate may be 0: SM cycles are informational and disabled at 0).  Anything else -- a string,
        # None, nan, inf, a value a float cannot carry, a negative or zero divisor -- refuses with reason
        # `rate_invalid:<key>` and the raw value in the receipt; a numeric projection is never written from it.
        def to_rate(name: str, raw: Any, allow_zero: bool = False) -> float:
            import numbers
            f = None; protocol_error = None
            try:
                ok = isinstance(raw, numbers.Real) and not isinstance(raw, bool) and getattr(getattr(raw, "dtype", None), "kind", None) != "b"
                if ok:
                    bool(raw)                             # the truth protocol of a typed rate is exercised HERE, inside the boundary, never before
                    f = float(raw)
                    ok = (f == f) and f != float("inf") and f != float("-inf") and (f > 0 or (allow_zero and f == 0))
            except MemoryError:                       # resource / process-control exceptions always propagate
                raise
            except (OverflowError, TypeError, ValueError):   # the VALUE is not a usable rate (too large, wrong type, unparsable)
                ok = False
            except Exception as exc:                  # any OTHER numeric-protocol failure of a typed value is malformed DATA, never an escape
                ok = False; protocol_error = type(exc).__name__   # the TYPE NAME only: str(exc) of a typed value may itself raise (rate-protocol-coverage-1)
            if not ok:
                reason = f"rate_protocol_error:{protocol_error}" if protocol_error else f"rate_invalid:{name}"
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="architecture_profile",
                    metadata_declared=metadata_declared, reason=reason, total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                    invalid_rate={"name": name, "value": raw, "profile": arch_name,
                                  "requirement": "a finite real number > 0" + (" (0 allowed: SM cycles are informational)" if allow_zero else ""),
                                  **({"protocol_error": protocol_error} if protocol_error else {})}))
                raise AssertionError("unreachable: _refuse raises")  # pragma: no cover
            return f

        freq_ghz = to_rate("freq_GHz", freq_ghz_raw)
        dram_bw = to_rate("DRAM_byte_per_cycle", dram_bw_raw)
        mac_per_cycle = to_rate(mac_rate_key_used, mac_per_cycle_raw)
        sm_per_cycle = to_rate("MAC_per_cycle_fp32_sm", sm_per_cycle_raw, allow_zero=True)
        for dt, rec in compute_dtype_rates.items():
            if rec.get("MAC_per_cycle") is not None:
                rec["MAC_per_cycle"] = to_rate(rec["key"], rec["MAC_per_cycle"])
        if bpe_raw_source is not None:
            bytes_per_element = to_rate(*bpe_raw_source)
        rates_evidence = {"freq_GHz": freq_ghz, "DRAM_byte_per_cycle": dram_bw, mac_rate_key_used: mac_per_cycle,
                          **{rec["key"]: rec["MAC_per_cycle"] for rec in compute_dtype_rates.values() if rec.get("MAC_per_cycle") is not None}}

        # ── Derived quantities (cycles, intensities, seconds, ratios): every value computed from accepted counts
        # and validated rates passes through `fin` -- a non-finite result (inf / nan, e.g. a 10**308 bucket over a
        # 1e-308 rate) refuses with reason `derived_quantity_not_finite`, the exact counts and the rates in the
        # receipt; a numeric projection is never written from a non-finite intermediate (rate-float-boundary-1).
        def fin(name: str, value: float, *, nonneg: bool = False, **inputs: Any) -> float:
            if value != value or value == float("inf") or value == float("-inf"):
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                    metadata_declared=metadata_declared, reason="derived_quantity_not_finite", total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                    malformed_counts=exact_evidence, derived={"name": name, "value": value, "inputs": {**rates_evidence, **inputs}}))
                raise AssertionError("unreachable: _refuse raises")  # pragma: no cover
            if nonneg and value < 0:
                # a FINAL traffic total (after the declared-boundary re-pricing) can never be negative; the signed
                # intermediate deltas that produce it may be (traffic-count-validation-1)
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                    metadata_declared=metadata_declared, reason="traffic_negative_final", total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                    malformed_counts=exact_evidence, derived={"name": name, "value": value, "inputs": {**rates_evidence, **inputs}}))
                raise AssertionError("unreachable: _refuse raises")  # pragma: no cover
            return value

        def to_float(name: str, raw: Any) -> float:
            """The one count -> float boundary: refuse (receipt with the exact ints) instead of raising.  Its input is
            an exact validated integer (``exact_total`` / ``_validate_counts``) or a product / sum of such floats; a
            value a float cannot carry refuses with ``count_not_representable_as_float``.  The sign check is kept as
            defence in depth (``traffic_negative`` / ``count_negative``: the zero-byte intensity sentinel applies to
            exactly-zero bytes only, intensity-sentinel-1); the ORIGINAL-object validation happens in
            ``exact_total`` BEFORE this conversion (traffic-count-validation-1)."""
            try:
                f = float(raw)
                if f != f or f == float("inf") or f == float("-inf"):
                    raise OverflowError(f"{name} is not finite as a float ({f!r})")
            except OverflowError as exc:
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                    metadata_declared=metadata_declared, reason="count_not_representable_as_float",
                    total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None), malformed_counts={**exact_evidence, name: raw},
                    malformed_protocol_errors={name: f"OverflowError: {exc}"}))
                raise AssertionError("unreachable: _refuse raises")  # pragma: no cover
            if f < 0:
                traffic = ("bytes" in name) or ("elements" in name)
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                    metadata_declared=metadata_declared, reason="traffic_negative" if traffic else "count_negative",
                    total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None), malformed_counts={**exact_evidence, name: raw},
                    malformed_protocol_errors={name: f"negative {'traffic' if traffic else 'count'}: {f!r}"}))
                raise AssertionError("unreachable: _refuse raises")  # pragma: no cover
            return f

        total_macs_f = to_float("total.macs", total_macs)
        macs_default_f = to_float("total.macs - priced quantized MACs (default-rate remainder)", macs_default)
        macs_tc_default_f = to_float("default-rate remainder minus scalar statement multiplies (tensor-core term, issue #58 B)", macs_tc_default)
        scalar_excluded_f = to_float("scalar statement multiplies excluded from the tensor-core term", scalar_excluded)
        compute_tc_cycles = fin("compute_tc_cycles (default-rate term)", macs_tc_default_f / mac_per_cycle if mac_per_cycle > 0 else 0.0,
                                macs=macs_tc_default_f, rate=mac_per_cycle)
        scalar_excluded_cycles = fin("scalar statement multiplies excluded (cycles at the applied rate; informational)",
                                     scalar_excluded_f / mac_per_cycle if mac_per_cycle > 0 else 0.0, macs=scalar_excluded_f, rate=mac_per_cycle)
        for dt, rec in compute_dtype_rates.items():
            rec["macs"] = to_float(f"compute_dtype_rates.{dt}.macs", rec["macs"])
            if rec.get("MAC_per_cycle"):
                compute_tc_cycles = fin(f"compute_tc_cycles (+ {dt} term)", compute_tc_cycles + rec["macs"] / rec["MAC_per_cycle"],
                                        macs=rec["macs"], rate=rec["MAC_per_cycle"])
        total_flops = to_float("total.flops", tot_exact.get("flops", 0))
        total_other_ops = to_float("total.other_ops", tot_exact.get("other_ops", 0))
        # Parse elements from new format, with fallback to old bytes format
        # With dtype_bytes=True and an analysis that carries per-tensor-dtype
        # byte totals (metadata bytes_accounting == "per-tensor-dtype"), memory
        # traffic is priced at those widths and ``precision`` only selects the
        # MAC rate. Otherwise every tensor is priced at bytes_per_element.
        dtype_bytes = (
            getattr(self, "dtype_bytes", False)
            and metadata.get("bytes_accounting") == "per-tensor-dtype"
            and "fused_bytes" in tot_exact
        )
        # New format uses unfused_elements, old format uses orojenesis_elements or _bytes suffix
        unfused_val = tot_exact.get("unfused_elements")      # presence of the validated exact int, never truthiness of a raw value
        if unfused_val is None:
            unfused_val = tot_exact.get("orojenesis_elements")
        if unfused_val is not None:
            # New format: elements -- every total was validated on the ORIGINAL object by the sweep above; converted here
            total_orojenesis_elems = to_float("total.unfused_elements", unfused_val)
            total_fused_elems = to_float("total.fused_elements", tot_exact.get("fused_elements", 0))
            total_fused_prefetched_elems = (to_float("total.fused_prefetched_elements", tot_exact["fused_prefetched_elements"])
                                            if "fused_prefetched_elements" in tot_exact else total_fused_elems)
            total_weight_elems = to_float("total.weight_elements", tot_exact.get("weight_elements", 0))
            total_model_io_elems = to_float("total.model_io_elements", tot_exact.get("model_io_elements", 0))
            total_intermediate_elems = to_float("total.intermediate_elements", tot_exact.get("intermediate_elements", 0))

            # Convert elements to bytes
            total_orojenesis_bytes = to_float("unfused_bytes (elements x width)", total_orojenesis_elems * bytes_per_element)
            total_fused_bytes = to_float("fused_bytes (elements x width)", total_fused_elems * bytes_per_element)
            total_fused_prefetched_bytes = to_float("fused_prefetched_bytes (elements x width)", total_fused_prefetched_elems * bytes_per_element)
            total_weight_bytes = to_float("weight_bytes (elements x width)", total_weight_elems * bytes_per_element)
            total_model_io_bytes = to_float("model_io_bytes (elements x width)", total_model_io_elems * bytes_per_element)
            total_intermediate_bytes = to_float("intermediate_bytes (elements x width)", total_intermediate_elems * bytes_per_element)
        else:
            # Old format: bytes (backward compatibility)
            total_orojenesis_bytes = to_float("total.orojenesis_bytes", tot_exact.get("orojenesis_bytes", 0))
            total_fused_bytes = to_float("total.fused_bytes", tot_exact.get("fused_bytes", 0))
            total_fused_prefetched_bytes = (to_float("total.fused_prefetched_bytes", tot_exact["fused_prefetched_bytes"])
                                            if "fused_prefetched_bytes" in tot_exact else total_fused_bytes)
            total_weight_bytes = to_float("total.weight_bytes", tot_exact.get("weight_bytes", 0))
            total_model_io_bytes = to_float("total.model_io_bytes", tot_exact.get("model_io_bytes", 0))
            total_intermediate_bytes = to_float("total.intermediate_bytes", tot_exact.get("intermediate_bytes", 0))

            # Convert bytes to elements -- each quotient is checked (an accepted tiny width makes it infinite)
            total_orojenesis_elems = fin("unfused.memory_elements (bytes / width)", total_orojenesis_bytes / bytes_per_element, bytes=total_orojenesis_bytes, width=bytes_per_element)
            total_fused_elems = fin("fused.memory_elements (bytes / width)", total_fused_bytes / bytes_per_element, bytes=total_fused_bytes, width=bytes_per_element)
            total_fused_prefetched_elems = fin("fused_prefetched.memory_elements (bytes / width)", total_fused_prefetched_bytes / bytes_per_element, bytes=total_fused_prefetched_bytes, width=bytes_per_element)
            total_weight_elems = fin("memory_breakdown.weight_elements (bytes / width)", total_weight_bytes / bytes_per_element, bytes=total_weight_bytes, width=bytes_per_element)
            total_model_io_elems = fin("memory_breakdown.model_io_elements (bytes / width)", total_model_io_bytes / bytes_per_element, bytes=total_model_io_bytes, width=bytes_per_element)
            total_intermediate_elems = fin("memory_breakdown.intermediate_elements (bytes / width)", total_intermediate_bytes / bytes_per_element, bytes=total_intermediate_bytes, width=bytes_per_element)

        # ── Boundary storage billed at DECLARED widths when present ──
        # The analysis carries model_io_bytes_declared /
        # external_io_bytes_declared_dedup when the caller bound the
        # benchmark definition's input/output STORAGE dtypes (campaign-audit
        # defects 2/3: a single per-graph byte scalar billed BF16 public I/O
        # at fp8's 1 byte and FP32 I/O at fp16's 2 bytes).  Only the
        # boundary component of each traffic class changes; weights,
        # intermediates and every compute rate (incl. legitimate FP8/NVFP4
        # MAC throughput) are untouched.
        boundary_billing = "uniform_precision_width"
        # the validated ORIGINAL integer is the evidence / output record (declared-count-roundtrip-1)
        declared_exact: Optional[int] = tot_exact.get("model_io_bytes_declared")
        if declared_exact is not None:
            declared_raw_f = to_float("total.model_io_bytes_declared", declared_exact)       # a separately guarded float, for the pricing arithmetic only
            # The declared raw total covers EVERY boundary tensor the analyzer
            # billed (model I/O and weight-classified declared inputs, plus
            # unnamed groups such as a recovered gather index).  The
            # re-pricing delta is therefore taken against the uniform-width
            # bytes of exactly those covered elements (`boundary_declared`),
            # not against the model-I/O class: since an earlier decision a declared input
            # typed `weight` lives in weight_bytes, and a class-based delta
            # would add it a second time (M97-INTEG-D09-1 follow-up on the
            # an earlier change composition).  Older analyses without `boundary_declared`
            # keep the class-based delta.
            if "elements_raw" in bd_exact:
                covered_uniform = fin("boundary_declared.elements_raw x width", to_float("total.boundary_declared.elements_raw", bd_exact["elements_raw"]) * bytes_per_element, width=bytes_per_element)
                delta_raw = fin("model_io_bytes_declared - covered uniform bytes", declared_raw_f - covered_uniform)   # a SIGNED intermediate delta: may be negative
                total_model_io_bytes = (to_float("total.boundary_declared.bytes_raw_model_io", bd_exact["bytes_raw_model_io"])
                                        if "bytes_raw_model_io" in bd_exact else declared_raw_f)
                # weight_bytes is the PARAMETER FOOTPRINT (total.weight_elements
                # x width).  Re-price the declared weight inputs' footprint —
                # distinct parameter source per consuming op x full numel, the
                # analyzer's own basis — from the uniform width to the declared
                # width.  Never the raw read count: a parameter in two slots
                # is read twice and an indexed table partially, so a read-based
                # correction mixes traffic into the footprint (M97-WEIGHT-2).
                fp_elems = to_float("total.boundary_declared.footprint_elements_weight", bd_exact.get("footprint_elements_weight", 0))
                if fp_elems:
                    fp_uniform = fin("boundary_declared.footprint_elements_weight x width", fp_elems * bytes_per_element, width=bytes_per_element)
                    fp_declared = to_float("total.boundary_declared.bytes_footprint_weight", bd_exact.get("bytes_footprint_weight", 0))
                    # FINAL re-priced footprint: the signed correction may be negative, the result may not
                    total_weight_bytes = fin("weight_bytes (declared footprint re-priced)", total_weight_bytes + fp_declared - fp_uniform, nonneg=True,
                                             weight_bytes_uniform=total_weight_bytes, bytes_footprint_weight=fp_declared, footprint_uniform=fp_uniform)
            else:
                delta_raw = fin("model_io_bytes_declared - model_io_bytes", declared_raw_f - total_model_io_bytes)   # a SIGNED intermediate delta: may be negative
                total_model_io_bytes = declared_raw_f
            # FINAL re-priced unfused traffic: the signed delta may be negative, the result may not
            total_orojenesis_bytes = fin("unfused_bytes (declared boundary re-priced)", total_orojenesis_bytes + delta_raw, nonneg=True,
                                         unfused_bytes_uniform=total_orojenesis_bytes, delta=delta_raw)
            if "external_io_bytes_declared_dedup" in tot_exact:
                # fused / fused_prefetched elements ARE the deduplicated
                # external-I/O set (see graph_analyzer), so their byte
                # totals are exactly the declared dedup bytes.
                dedup_f = to_float("total.external_io_bytes_declared_dedup", tot_exact["external_io_bytes_declared_dedup"])
                total_fused_bytes = dedup_f
                total_fused_prefetched_bytes = dedup_f
            boundary_billing = "declared_storage_dtypes"

        # a successful projection is the ONE current state for this profile
        # (receipt-lifecycle-3): an earlier refusal receipt is retired by name
        self._retire_refusal_receipt(out_dir, arch_name)
        # ONE combined representation of the prediction's incompleteness, built
        # after every cause is known (qualification-1): the analysis'
        # per-layer reasons and the capture-degraded marker; the machine fields
        # and the qualification text below are derived from it, once.
        perf_incomplete_layers: Dict[str, str] = dict(total.get("incomplete_layers") or {}) if total.get("status") == "incomplete" else {}
        perf_incomplete = bool(perf_incomplete_layers) or bool(capture_degraded)
        qualification_fields: Dict[str, Any] = {}
        if perf_incomplete:
            qualification_fields["status"] = "incomplete"
            if perf_incomplete_layers:
                qualification_fields["incomplete_layers"] = perf_incomplete_layers
                qualification_fields["prediction_qualification"] = _prediction_qualification(perf_incomplete_layers)
            if capture_degraded:
                qualification_fields["capture_degraded"] = capture_degraded
        # SM cycles (elementwise / reduction ops that run on CUDA cores): the informational diagnostic
        # other_ops / MAC_per_cycle_fp32_sm.  It is EXCLUDED from the headline bound by convention (issue #58 B: the
        # headline is an optimistic contraction-and-memory bound, not a bound over all compute resources); that
        # exclusion is not a claim that this work is hidden by the memory term or free.  The diagnostic prices
        # other_ops only -- the scalar statement multiplies excluded above are retained as counts, not added here.
        compute_sm_cycles = fin("compute_sm_cycles", total_other_ops / sm_per_cycle if sm_per_cycle > 0 else 0.0, other_ops=total_other_ops)
        # SOL compute = tensor-core cycles only
        compute_cycles = compute_tc_cycles

        # Memory cycles for each model (using bytes)
        unfused_mem_cycles = fin("unfused.memory_cycles", total_orojenesis_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_orojenesis_bytes)
        fused_mem_cycles = fin("fused.memory_cycles", total_fused_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_fused_bytes)
        fused_prefetched_mem_cycles = fin("fused_prefetched.memory_cycles", total_fused_prefetched_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_fused_prefetched_bytes)

        # Total cycles (roofline: max of compute and memory)
        unfused_total_cycles = max(compute_cycles, unfused_mem_cycles)
        fused_total_cycles = max(compute_cycles, fused_mem_cycles)
        fused_prefetched_total_cycles = max(compute_cycles, fused_prefetched_mem_cycles)

        # Calculate arithmetic intensity for each model (MACs / bytes)
        # arithmetic intensity = MACs / bytes.  `float('inf')` on a ZERO-byte path is an intentional, documented
        # sentinel (no traffic at all), not a derived overflow; a division over positive bytes is checked.
        unfused_ai = float('inf') if total_orojenesis_bytes == 0 else fin("unfused.arithmetic_intensity", total_macs_f / total_orojenesis_bytes, bytes=total_orojenesis_bytes)
        fused_ai = float('inf') if total_fused_bytes == 0 else fin("fused.arithmetic_intensity", total_macs_f / total_fused_bytes, bytes=total_fused_bytes)
        fused_prefetched_ai = float('inf') if total_fused_prefetched_bytes == 0 else fin("fused_prefetched.arithmetic_intensity", total_macs_f / total_fused_prefetched_bytes, bytes=total_fused_prefetched_bytes)

        if dtype_bytes:
            total_orojenesis_bytes = to_float("total.unfused_bytes", tot_exact["unfused_bytes"]) if "unfused_bytes" in tot_exact else total_orojenesis_bytes
            total_fused_bytes = to_float("total.fused_bytes", tot_exact["fused_bytes"])
            total_fused_prefetched_bytes = to_float("total.fused_prefetched_bytes", tot_exact["fused_prefetched_bytes"]) if "fused_prefetched_bytes" in tot_exact else total_fused_bytes
            total_intermediate_bytes = to_float("total.intermediate_bytes", tot_exact["intermediate_bytes"]) if "intermediate_bytes" in tot_exact else total_intermediate_bytes
            total_model_io_bytes = to_float("total.external_input_bytes + total.external_output_bytes",
                                            to_float("total.external_input_bytes", tot_exact.get("external_input_bytes", 0))
                                            + to_float("total.external_output_bytes", tot_exact.get("external_output_bytes", 0)))
            total_weight_bytes = 0.0  # folded into external inputs at their own widths
            unfused_mem_cycles = fin("unfused.memory_cycles", total_orojenesis_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_orojenesis_bytes)
            fused_mem_cycles = fin("fused.memory_cycles", total_fused_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_fused_bytes)
            fused_prefetched_mem_cycles = fin("fused_prefetched.memory_cycles", total_fused_prefetched_bytes / dram_bw if dram_bw > 0 else 0.0, bytes=total_fused_prefetched_bytes)
            unfused_total_cycles = max(compute_cycles, unfused_mem_cycles)
            fused_total_cycles = max(compute_cycles, fused_mem_cycles)
            fused_prefetched_total_cycles = max(compute_cycles, fused_prefetched_mem_cycles)
            # arithmetic intensity = MACs / bytes.  `float('inf')` on a ZERO-byte path is an intentional, documented
            # sentinel (no traffic at all), not a derived overflow; a division over positive bytes is checked.
            unfused_ai = float('inf') if total_orojenesis_bytes == 0 else fin("unfused.arithmetic_intensity", total_macs_f / total_orojenesis_bytes, bytes=total_orojenesis_bytes)
            fused_ai = float('inf') if total_fused_bytes == 0 else fin("fused.arithmetic_intensity", total_macs_f / total_fused_bytes, bytes=total_fused_bytes)
            fused_prefetched_ai = float('inf') if total_fused_prefetched_bytes == 0 else fin("fused_prefetched.arithmetic_intensity", total_macs_f / total_fused_prefetched_bytes, bytes=total_fused_prefetched_bytes)

        # Ridge point: where compute-bound meets memory-bound
        ridge_point = fin("arch.ridge_point", mac_per_cycle / dram_bw if dram_bw > 0 else 0.0)
        # runtimes (ms) and ratios -- every derived value is checked before it is written
        cycles_per_ms = fin("freq_GHz x 1e6 (cycles per millisecond)", freq_ghz * 1e6)    # checked BEFORE it divides: 1e308 GHz must not become a zero quotient
        unfused_runtime_ms = fin("unfused.runtime_ms", unfused_total_cycles / cycles_per_ms, total_cycles=unfused_total_cycles)
        fused_runtime_ms = fin("fused.runtime_ms", fused_total_cycles / cycles_per_ms, total_cycles=fused_total_cycles)
        fused_prefetched_runtime_ms = fin("fused_prefetched.runtime_ms", fused_prefetched_total_cycles / cycles_per_ms, total_cycles=fused_prefetched_total_cycles)
        speedup_fused_vs_unfused = fin("speedup.fused_vs_unfused", (unfused_total_cycles / fused_total_cycles) if fused_total_cycles > 0 else 1.0)
        speedup_fp_vs_unfused = fin("speedup.fused_prefetched_vs_unfused", (unfused_total_cycles / fused_prefetched_total_cycles) if fused_prefetched_total_cycles > 0 else 1.0)
        speedup_fp_vs_fused = fin("speedup.fused_prefetched_vs_fused", (fused_total_cycles / fused_prefetched_total_cycles) if fused_prefetched_total_cycles > 0 else 1.0)
        memred_fused_vs_unfused = fin("memory_reduction.fused_vs_unfused", 1.0 - (total_fused_bytes / total_orojenesis_bytes) if total_orojenesis_bytes > 0 else 0.0)
        memred_fp_vs_unfused = fin("memory_reduction.fused_prefetched_vs_unfused", 1.0 - (total_fused_prefetched_bytes / total_orojenesis_bytes) if total_orojenesis_bytes > 0 else 0.0)
        # FINAL traffic totals (after every re-pricing and the dtype-bytes selection): finite AND non-negative
        # (traffic-count-validation-1); each value here is already a checked float, this is the last gate before the record
        total_orojenesis_bytes = fin("unfused.memory_bytes (final)", total_orojenesis_bytes, nonneg=True)
        total_fused_bytes = fin("fused.memory_bytes (final)", total_fused_bytes, nonneg=True)
        total_fused_prefetched_bytes = fin("fused_prefetched.memory_bytes (final)", total_fused_prefetched_bytes, nonneg=True)
        total_weight_bytes = fin("memory_breakdown.weight_bytes (final)", total_weight_bytes, nonneg=True)
        total_model_io_bytes = fin("memory_breakdown.model_io_bytes (final)", total_model_io_bytes, nonneg=True)
        total_intermediate_bytes = fin("memory_breakdown.intermediate_bytes (final)", total_intermediate_bytes, nonneg=True)
        # weight_elements_unresolved (unresolved-weight-boundary-1): the ORIGINAL value is validated as an exact
        # non-negative integer by the one count validator BEFORE any cast; the float form and the byte product both go
        # through the guarded layer; only then are the exact element count and the checked byte product written.
        unresolved_weight_fields: Dict[str, Any] = {}
        if "weight_elements_unresolved" in total:                       # PRESENCE, never truthiness: a present False / None / [] / {} / "" / array is validated and refused
            w_raw = total["weight_elements_unresolved"]; w_errors: list = []
            w_exact = _exact_count(w_raw, w_errors)
            if w_exact is None:
                self._refuse(out_dir, arch_name, PerfProjectionUnavailable(
                    arch_name, set(compute_dtype_rates) or {precision}, {}, source="analysis_total",
                    metadata_declared=metadata_declared, reason="compute_dtype_count_malformed", total_macs=total_macs, macs_typed=(total.get("macs_typed") if isinstance(total.get("macs_typed"), dict) else None),
                    malformed_counts={**exact_evidence, "total.weight_elements_unresolved": w_raw},
                    malformed_protocol_errors={"total.weight_elements_unresolved": w_errors[0]} if w_errors else None))
            if w_exact > 0:                                             # the valid zero is admitted and, as before, not written
                w_unresolved_f = to_float("total.weight_elements_unresolved", w_exact)
                w_unresolved_bytes = to_float("weight_bytes_unresolved (elements x width)", w_unresolved_f * bytes_per_element)
                unresolved_weight_fields = {"weight_elements_unresolved": w_exact, "weight_bytes_unresolved": int(w_unresolved_bytes)}

        tensor_core_term = {
            "convention": SOL_CONVENTION["id"],
            "applied": bool(tc_term_record.get("applied")),
            **({"reason": tc_term_record["reason"]} if not tc_term_record.get("applied") else {}),
            "macs_priced": int(macs_tc_default) + int(priced_quantized_macs),            # every MAC in compute_tc_cycles
            "default_rate_macs_priced": int(macs_tc_default),
            "quantized_dtype_macs_priced": int(priced_quantized_macs),
            # what the priced MACs are (round 2, P2-term-label-1): contraction (quantised buckets + workload precision)
            # plus the analyzer's other non-contraction MACs, which the term retains as before; only the scalar statement
            # multiplies are excluded.  Without typed components the whole remainder is priced and these two are None.
            "contraction_macs_priced": (int(tc_term_record["contraction_wp"]) + int(priced_quantized_macs)) if tc_term_record.get("applied") else None,
            "non_contraction_other_macs_priced": int(tc_term_record["non_contraction_other"]) if tc_term_record.get("applied") else None,
            "scalar_statement_multiplies_excluded": int(scalar_excluded),
            "excluded_cycles_at_applied_rate": float(scalar_excluded_cycles),
            "rate_key": mac_rate_key_used, "rate_class": _rate_class(mac_rate_key_used),
            "note": ("compute_tc_cycles = default_rate_macs_priced / MAC_per_cycle + sum(quantised bucket / its rate); priced: contraction MACs"
                     + (" and the analyzer's other non-contraction MACs" if tc_term_record.get("applied") and tc_term_record.get("non_contraction_other") else "")
                     + "; excluded: the scalar statement multiplies (their typed count is retained in workload.macs_typed) and other_ops; the SM "
                     "diagnostic compute_sm_cycles prices other_ops only and is unchanged; the headline is an optimistic contraction-and-memory "
                     "bound, not a bound over all compute resources"),
        }
        perf: Dict[str, Any] = {
            "sol_convention": dict(SOL_CONVENTION),
            "arch": {
                "name": arch_name,
                "freq_GHz": freq_ghz,
                "DRAM_byte_per_cycle": dram_bw,
                "mac_per_cycle_key": mac_key,
                # issue #58 (P2-rate-class-1): the key the fallback chain actually APPLIED (`mac_per_cycle_key` is the key the
                # workload precision requested; they differ when the profile lacks it) and its class -- report-only, additive
                "mac_per_cycle_key_applied": mac_rate_key_used,
                "mac_rate_class_applied": _rate_class(mac_rate_key_used),
                "MAC_per_cycle": mac_per_cycle,
                "MAC_per_cycle_fp32_sm": sm_per_cycle,
                **({"compute_dtype_rates": compute_dtype_rates} if compute_dtype_rates else {}),
                "ridge_point": ridge_point,
            },
            "workload": {
                "total_macs": int(total_macs),
                # issue #58: the aggregate's typed components and the equation-faithfulness marker copied from the analysis
                # total, and what the DEFAULT-RATE remainder (total.macs minus the quantised-dtype MACs priced above) is made
                # of -- contraction MACs at the workload precision, the analyzer's other non-contraction MACs (both priced at
                # the APPLIED rate `mac_rate_key_used`: the requested `MAC_per_cycle_<precision>_tc` when the profile declares
                # it, else the key the fallback chain selected) and the scalar multiply statements of non-einsum kernels, which
                # convention B EXCLUDES from the tensor-core term (`tensor_core_term` below).
                **({"macs_typed": total["macs_typed"]} if isinstance(total.get("macs_typed"), dict) else {}),
                **({"equation_unfaithful_layers": list(total["equation_unfaithful_layers"])} if total.get("equation_unfaithful_layers") else {}),
                **({"default_rate_remainder": _default_rate_remainder_note(total, int(macs_default), mac_rate_key_used, mac_key, precision, int(scalar_excluded))} if int(macs_default) > 0 else {}),
                "tensor_core_term": tensor_core_term,
                "total_other_ops": int(total_other_ops),
                "total_flops": int(total_flops),
                "bytes_per_element": "per-tensor-dtype" if dtype_bytes else bytes_per_element,
                "mac_rate_bytes_per_element": bytes_per_element,
                # the prediction's incompleteness: status, the analysis' per-layer
                # reasons, the reason-specific qualification text and the
                # capture-degraded marker — one spread from one combined
                # representation (STATUS_CONTRACT.md; qualification-1).  A
                # compute dtype without a rate never reaches here (refused above).
                **qualification_fields,
                "boundary_billing": boundary_billing,
                **({"unresolved_ops": total.get("unresolved_ops")}
                   if total.get("unresolved_ops") else {}),
                **({"model_io_bytes_declared": declared_exact}      # the validated ORIGINAL integer, never a float round trip (declared-count-roundtrip-1)
                   if declared_exact is not None else {}),
                **({"quant_orig_dtype": quant_label} if quant_label else {}),
                **({"quant_metadata_unbound": {**metadata_declared,
                                               "note": "declared by the workload metadata but no MAC-bearing layer carries this "
                                                       "compute dtype and the profile has no rate for it: MACs are priced at the "
                                                       f"{precision} rate ({mac_key}); the dtype is billed as storage/traffic only"}}
                   if metadata_declared and not metadata_declared["rate_in_profile"] else {}),
            },
            **({"capture_capability": dict(capture_capability)} if capture_capability else {}),
            "unfused": {
                "description": "Each op in isolation, all tensors from DRAM",
                "memory_elements": int(total_orojenesis_elems),
                "memory_bytes": int(total_orojenesis_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(unfused_mem_cycles),
                "total_cycles": int(unfused_total_cycles),
                "runtime_ms": unfused_runtime_ms,
                "arithmetic_intensity": unfused_ai,
                "bottleneck": "compute" if compute_cycles >= unfused_mem_cycles else "memory",
            },
            "fused": {
                "description": "Per-op roofline, intermediate tensors excluded",
                "memory_elements": int(total_fused_elems),
                "memory_bytes": int(total_fused_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(fused_mem_cycles),
                "total_cycles": int(fused_total_cycles),
                "runtime_ms": fused_runtime_ms,
                "arithmetic_intensity": fused_ai,
                "bottleneck": "compute" if compute_cycles >= fused_mem_cycles else "memory",
            },
            "fused_prefetched": {
                "description": "Single roofline for entire graph, perfect overlap",
                "memory_elements": int(total_fused_prefetched_elems),
                "memory_bytes": int(total_fused_prefetched_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(fused_prefetched_mem_cycles),
                "total_cycles": int(fused_prefetched_total_cycles),
                "runtime_ms": fused_prefetched_runtime_ms,
                "arithmetic_intensity": fused_prefetched_ai,
                "bottleneck": "compute" if compute_cycles >= fused_prefetched_mem_cycles else "memory",
            },
            # weight_elements / weight_bytes are the PARAMETER FOOTPRINT of
            # the analysis (distinct parameters per consuming op), not the
            # weight traffic (analysis `weight_read_elements`); the three
            # parts are not a partition of fused traffic when footprint and
            # reads differ (gathers, repeated operands, unknown provenance).
            "memory_breakdown": {
                "auxiliary_non_additive": True,
                "semantics": {
                    "scope": "Mixed diagnostic bases; not a disjoint physical-memory breakdown or a general formula for fused/unfused traffic.",
                    "weight_elements": "Analysis parameter-footprint counter per consuming op, not globally unique storage or weight-read traffic; legacy byte-only inputs retain their width conversion.",
                    "weight_bytes": (
                        "Sentinel zero: weights are folded into external-input bytes in dtype mode; not zero parameter storage or a zero-width dtype."
                        if dtype_bytes else
                        "Parameter-footprint estimate at reporting width, with any declared-boundary adjustment; not weight-read traffic."
                    ),
                    "model_io_elements": "Analysis per-op non-weight external reads plus model outputs, not re-deduplicated when dtype mode changes the byte field; legacy byte-only inputs retain their width conversion.",
                    "model_io_bytes": (
                        "Deduplicated external-input plus external-output bytes, including weights; a different counting basis from model_io_elements."
                        if dtype_bytes else
                        "Per-op model-I/O counter at reporting width, with any declared-boundary override; shared reads may repeat."
                    ),
                    "intermediate_elements": "Summed per-op internal reads and writes, not a resident allocation footprint; legacy byte-only inputs retain their width conversion.",
                    "intermediate_bytes": "Those internal accesses at reporting width or saved per-tensor widths; not an additive complement of deduplicated external traffic.",
                    "traffic_fields": "Use unfused.memory_bytes, fused.memory_bytes and fused_prefetched.memory_bytes for the corresponding traffic scenarios.",
                },
                "weight_elements": int(total_weight_elems),
                "weight_bytes": int(total_weight_bytes),
                **unresolved_weight_fields,
                "model_io_elements": int(total_model_io_elems),
                "model_io_bytes": int(total_model_io_bytes),
                "intermediate_elements": int(total_intermediate_elems),
                "intermediate_bytes": int(total_intermediate_bytes),
            },
            "speedup": {
                "fused_vs_unfused": speedup_fused_vs_unfused,
                "fused_prefetched_vs_unfused": speedup_fp_vs_unfused,
                "fused_prefetched_vs_fused": speedup_fp_vs_fused,
            },
            "memory_reduction": {
                "fused_vs_unfused": memred_fused_vs_unfused,
                "fused_prefetched_vs_unfused": memred_fp_vs_unfused,
            },
        }

        out_path = out_dir / f"perf_{arch_name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(perf, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            logger.debug(f"✅ Wrote perf: {out_path}")

        return perf

    @staticmethod
    def _stale_name(out_dir: Path, pattern: str) -> Path:
        """First unused ``pattern.format(n)`` — historical files are never overwritten."""
        n = 0
        while (out_dir / pattern.format(n)).exists():
            n += 1
        return out_dir / pattern.format(n)

    @classmethod
    def _retire_refusal_receipt(cls, out_dir: Path, arch_name: str) -> Optional[Path]:
        """On a successful projection, move an earlier ``perf_<arch>.REFUSED.yaml`` to
        ``perf_<arch>.REFUSED.stale.<n>.yaml`` (receipt-lifecycle-3): exactly one
        current state per architecture, historical receipts kept under a stale name."""
        receipt = out_dir / f"perf_{arch_name}.REFUSED.yaml"
        if not receipt.exists():
            return None
        target = cls._stale_name(out_dir, f"perf_{arch_name}.REFUSED.stale.{{}}.yaml")
        receipt.rename(target)
        return target

    @classmethod
    def _refuse(cls, out_dir: Path, arch_name: str, exc: "PerfProjectionUnavailable") -> None:
        """Leave an explicit REFUSED receipt instead of ``perf_<arch>.yaml`` and raise.

        A ``perf_<arch>.yaml`` from an earlier run must not survive next to the
        refusal (it would read as this run's prediction), so it is renamed
        ``perf_<arch>.yaml.REFUSED.stale.<n>`` (numbered: never overwritten); an
        earlier refusal receipt is likewise kept as ``perf_<arch>.REFUSED.stale.<n>.yaml``
        so the new receipt is the ONE current state (receipt-lifecycle-3).
        """
        stale = out_dir / f"perf_{arch_name}.yaml"
        if stale.exists():
            stale.rename(cls._stale_name(out_dir, f"perf_{arch_name}.yaml.REFUSED.stale.{{}}"))
        cls._retire_refusal_receipt(out_dir, arch_name)
        # The receipt text is rendered BEFORE the file is opened, so a value the
        # dumper cannot represent can never leave an empty or truncated receipt
        # (count-integer-types-1); the fallback renders every field through
        # the receipt-safe form (ints / repr strings) and records that it did.
        receipt = exc.receipt()
        try:
            text = yaml.dump(receipt, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
        except MemoryError:
            raise
        except Exception:
            safe = _receipt_safe(receipt)                      # every scalar an exact built-in type
            if not isinstance(safe, dict):
                safe = {"receipt_repr": safe}
            safe["receipt_serialization"] = "repr-fallback"
            try:
                text = yaml.dump(safe, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
            except MemoryError:
                raise
            except Exception:
                # last resort, no YAML dumper involved: plain JSON text (JSON is YAML-parseable), every
                # unknown object through repr — a REFUSED file is ALWAYS written (count-real-serialization-1)
                import json
                safe["receipt_serialization"] = "json-fallback"
                text = json.dumps(safe, indent=2, default=lambda o: _receipt_safe(repr(o))) + "\n"
        (out_dir / f"perf_{arch_name}.REFUSED.yaml").write_text(text)
        logger.error("%s", exc.message)
        raise exc

    # Maps orig_dtypes keywords from metadata.yaml to (precision_key, bytes_per_element)
    _QUANT_DTYPE_MAP = {
        "nvfp4": ("nvfp4", 0.5),
        "float4_e2m1fn_x2": ("nvfp4", 0.5),
        "fp4": ("nvfp4", 0.5),
        "fp8": ("fp8", 1),
        "float8_e4m3fn": ("fp8", 1),
        "float8_e5m2": ("fp8", 1),
        "float8_e4m3fnuz": ("fp8", 1),
        "float8_e5m2fnuz": ("fp8", 1),
    }

    def _load_quant_metadata(
        self, analysis_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Search for metadata.yaml near the analysis path.

        Typical layout::

            <model_output>/metadata.yaml
            <model_output>/analysis/analysis.yaml   <-- analysis_path

        We walk up from ``analysis_path`` checking each parent for
        ``metadata.yaml`` (max 3 levels).
        """
        search_dir = analysis_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    return None
            search_dir = search_dir.parent
        return None

    def _resolve_quant_overrides(
        self,
        metadata: Dict[str, Any],
        arch: Dict[str, Any],
        precision: str,
    ) -> tuple:
        """Derive MAC key and bytes_per_element from quantization metadata.

        Scans ``dtype_conversions`` for the *highest-throughput* original
        quantized dtype (nvfp4 > fp8).  Returns the corresponding
        ``(mac_per_cycle_key, bytes_per_element, orig_dtype_label)`` or
        ``(None, None, None)`` if no quantized dtypes are found.
        """
        conversions = metadata.get("dtype_conversions") or []
        if not conversions:
            return None, None, None

        # Priority: nvfp4 > fp8  (pick highest throughput)
        best_precision = None
        best_bpe = None
        best_label = None

        for conv in conversions:
            orig = str(conv.get("orig_dtypes", "")).lower()
            for keyword, (prec, bpe) in self._QUANT_DTYPE_MAP.items():
                if keyword in orig:
                    if best_precision is None or bpe < best_bpe:
                        best_precision = prec
                        best_bpe = bpe
                        best_label = orig
                    break

        if best_precision is None:
            return None, None, None

        mac_key = f"MAC_per_cycle_{best_precision}_tc"
        # NOTE: the sidecar's declaration is not compute-dtype evidence by itself
        # (mac-scope-1).  When the profile lacks ``mac_key`` the caller drops
        # this MAC-rate override (MACs no layer binds to a quantized compute dtype
        # price at the workload precision's rate, recorded as
        # ``workload.quant_metadata_unbound``), keeps the bytes override, and
        # refuses only when the analysis carries MAC-bearing layers / aggregate
        # work at a dtype without a rate.

        if self.debug:
            logger.debug(f"Quant override: orig_dtype={best_label} -> "
                  f"mac_key={mac_key}, bytes_per_element={best_bpe}")

        return mac_key, best_bpe, best_label

    def _load_arch_config(self, arch_config: str) -> Dict[str, Any]:
        """Load an architecture YAML by name or path."""
        # Explicit path.
        cfg_path = Path(arch_config)
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}

        # Shipped config by name: configs/arch/<name>.yaml in a checkout, an
        # unpacked sdist or the packaged copy (solar/_data) of a wheel install.
        from solar.resources import find_resource
        candidate = find_resource("configs", "arch", f"{arch_config}.yaml")
        if candidate is not None:
            with open(candidate) as f:
                return yaml.safe_load(f) or {}

        # Fallback: return empty.
        return {}


__all__ = ["EinsumGraphPerfModel", "PerfProjectionUnavailable"]
