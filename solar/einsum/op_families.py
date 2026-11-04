# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation-name families the mechanical route recognises (issue #50).

``EinsumAnalyzer.get_einsum_op`` routes a recorded operation name through the
handler registry.  An op with no handler used to be lowered to the generic
``ElementwiseSelectEinsumOp`` copy — zero MACs, ``is_einsum_supportable:
true`` — whatever its name, so real work (``torch._scaled_mm`` before an earlier change,
``mm.out`` / ``aten.mm.default`` spellings, any opaque name) vanished without
a marker.  This module holds the two pieces that make the route fail closed:

* :func:`normalise_overload_name` strips the ``aten.`` / ``torch.ops.aten.``
  prefixes and the ``.out`` / ``.default`` (and other aten overload) suffixes
  so that ``mm.out``, ``aten.mm.default``, ``bmm.out``, ``addmm.out`` reach
  the real handlers.  ``out=`` is the aliasing destination the capture already
  split off (an earlier change ``_split_destination``); it is not an operand.
* :data:`RECOGNISED_COPY_FAMILIES` names the handler-less operations whose
  lowering to the zero-MAC copy / select is the intended accounting: views,
  aliases, casts, memory-format copies, masks and selects, scatters (in-place
  writes), creation ops and the gathers.  Everything else without a handler
  raises :class:`UnrecognisedOperationError`; the converter records the name
  on the layer (``unrecognised_op``) with ``is_einsum_supportable: false`` and
  the analyzer reports ``status: incomplete`` / ``incomplete_reason:
  op_unrecognised`` with 0 MACs (INTEG-9 carries it to the total and to the
  perf qualification).  The layer's arithmetic is zeroed, not estimated; its
  recorded-shape traffic remains in the partial, unqualified estimate (which
  is therefore not a lower bound of the complete work).
"""

from __future__ import annotations

import re

__all__ = [
    "UnrecognisedOperationError",
    "RECOGNISED_COPY_FAMILIES",
    "PLACEHOLDER_OPS",
    "IDENTITY_COPY_OP",
    "normalise_overload_name",
    "identity_interpolate_evidence",
    "interpolate_evidence_record",
    "interpolate_record_refusal",
    "callee_identity",
    "parse_recorded_kwargs",
]


class UnrecognisedOperationError(ValueError):
    """No handler and no recognised family for a recorded operation name."""

    def __init__(self, recorded_name: str, normalised_name: str):
        self.recorded_name = str(recorded_name)
        self.normalised_name = str(normalised_name)
        super().__init__(
            f"operation {self.recorded_name!r} (normalised {self.normalised_name!r}) has no einsum "
            f"handler and belongs to no recognised zero-MAC family: it is not lowered to a copy; "
            f"the layer is reported incomplete (op_unrecognised)")


# aten overload tokens that follow the op name: aten.<op>.<overload>
_ATEN_PREFIXES = ("torch.ops.aten.", "ops.aten.", "aten.")
_OVERLOAD_SUFFIXES = (".out", ".default")


def normalise_overload_name(op_name: str) -> str:
    """``aten.mm.default`` / ``torch.ops.aten.mm.default`` -> ``mm``; ``mm.out`` -> ``mm``.

    For an ``aten.``-prefixed name the first component after the prefix is
    the op (``aten.add.Tensor`` -> ``add``, ``aten.sum.dim_IntList`` -> ``sum``);
    for any other name only the ``.out`` / ``.default`` suffixes are removed,
    so recorded names such as ``Model.max`` or ``clamp_4`` are untouched (the
    analyzer's own family rules handle those).  Case-insensitive.
    """
    op = str(op_name).lower()
    for prefix in _ATEN_PREFIXES:
        if op.startswith(prefix):
            rest = op[len(prefix):]
            return rest.split(".")[0] if rest else op
    for suffix in _OVERLOAD_SUFFIXES:
        if op.endswith(suffix) and len(op) > len(suffix):
            return op[: -len(suffix)]
    return op


# Handler-less operations whose zero-MAC copy / select lowering IS the intended
# accounting (the analyzer prices their traffic by family: views and aliases
# are free, casts/copies move bytes, masks/selects bill one op per output,
# scatters write the indexed rows).  Grouped by family; keep in sync with the
# analyzer's ALIAS_OPS / _ZERO_COMPUTE_OPS / _SCATTER_OPS / _MASK_FILL_OPS /
# _CREATION_OUTPUT_OPS sets when a family is added there.
RECOGNISED_COPY_FAMILIES = frozenset({
    # alias / view
    "detach", "detach_", "alias", "view_as", "reshape_as", "swapaxes", "swapdims", "movedim",
    "moveaxis", "as_strided", "unbind", "narrow", "slice", "select", "unfold", "unflatten",
    "expand_as", "broadcast_to", "tensor_split",
    # casts / device moves (zero compute, byte traffic only)
    "to", "type", "type_as", "float", "double", "half", "bfloat16", "int", "long", "short",
    "char", "byte", "bool", "cpu", "cuda",
    # memory-format copies
    "clone", "contiguous", "copy_",
    # masks / selects (one op per output element)
    "where", "masked_fill", "masked_fill_", "__invert__", "logical_not", "bitwise_not",
    "one_hot",
    # boolean / bitwise elementwise (mask algebra; one op per output element)
    "__and__", "__or__", "__xor__", "__rand__", "__ror__", "__rxor__", "__iand__", "__ior__",
    "__ixor__", "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or",
    "bitwise_xor",
    # data movement
    "repeat_interleave", "tile", "roll", "flip", "constant_pad_nd",
    # scatters / in-place writes (read-modify-write of the indexed rows)
    "__setitem__", "scatter", "scatter_", "index_copy", "index_copy_", "index_put", "index_put_",
    "index_add", "index_add_", "scatter_add", "scatter_add_", "scatter_reduce", "scatter_reduce_",
    "index_reduce", "index_reduce_", "masked_scatter", "masked_scatter_", "put_",
    # creation / fill
    "zeros_like", "zeros", "full_like", "full", "empty_like", "empty", "ones_like", "ones",
    "new_zeros", "new_full", "new_ones", "new_empty", "fill_", "zero_",
    # gathers (re-indexing; the analyzer's an earlier decision rule prices them)
    "gather", "take", "take_along_dim",
    # an evidenced identity-size F.interpolate (identity_interpolate_evidence)
    "identity_copy",
})

# Internal placeholder names the converter uses for a layer it has already
# marked unsupportable (a refused scaled_mm, an unrecognised op): they produce
# the rank-preserving equation that keeps the output tensor defined and bill
# nothing.
PLACEHOLDER_OPS = frozenset({"unresolved_op"})

# NOTE (kernel-origin-10): the DSL route's layer type "kernel" is NOT an
# admitted name.  Bodiless DSL builtins are admitted to the generic cost by the
# exporter's explicit provenance (``dsl_route`` + resolved ``builtin`` from
# solar.einsum_dsl.builtin_table), never by the type name.


# ---------------------------------------------------------------------------
# F.interpolate: a narrowly bound identity-copy rule (an earlier review)
# ---------------------------------------------------------------------------

# Internal op name the converter lowers an evidenced identity resize to: the
# zero-arithmetic copy whose traffic is the recorded input read + output write.
IDENTITY_COPY_OP = "identity_copy"

_IDENTITY_INTERPOLATE_MODES = ("linear", "nearest", "nearest-exact")


def _split_top_level(text, sep=","):
    """Split *text* at *sep* occurrences that sit at bracket depth 0 and outside quotes."""
    parts, depth, quote, cur, escaped = [], 0, None, [], False
    for ch in text:
        if quote:
            cur.append(ch)
            if escaped:
                escaped = False          # a backslash-escaped character never closes the quote
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_recorded_kwargs(raw_attributes):
    """DIAGNOSTIC ONLY: top-level keyword bindings of a recorded display string, or None.

    This parser never admits anything (parser-5: a display string can be
    forged by string contents, bracket mismatches or extra slots); the
    identity-copy rule consumes the structured capture-time record instead.
    It is kept for inspecting legacy graphs.

    The capture records a call as the display string ``[[<positional args>],
    {k: v, ...}]``.  Only the LAST top-level ``{...}`` is the keyword dict, and
    only its depth-0 ``key: value`` entries are bindings: a nested
    ``metadata: {mode: 'linear'}`` binds ``metadata``, never ``mode``; a key
    is matched exactly (``recompute_scale_factor`` is not ``scale_factor``);
    a key bound twice makes the record ambiguous and the result is None.
    Values are returned as scalars where they are unambiguous: ``None`` ->
    None, quoted strings -> str, ints/floats -> int/float, a tuple/list of
    ints -> list[int]; anything else (a nested dict, a tensor display, an
    expression) is kept as the raw text wrapped in ``_Opaque``.
    """
    text = str(raw_attributes).strip()
    if not text.endswith("]"):
        return None
    body = text[1:-1] if text.startswith("[") else text
    tops = _split_top_level(body)
    if not tops or not (tops[-1].startswith("{") and tops[-1].endswith("}")):
        return None
    entries = _split_top_level(tops[-1][1:-1])
    out = {}
    for e in entries:
        kv = _split_top_level(e, ":")
        if len(kv) < 2:
            return None
        key = kv[0].strip().strip("'\"")
        val = ":".join(kv[1:]).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in out:
            return None          # malformed or duplicate key: ambiguous record
        out[key] = _scalar(val)
    return out


class _Opaque(str):
    """A recorded value that is not an unambiguous scalar (kept for display only)."""


def _scalar(val):
    if val == "None":
        return None
    if val in ("True", "False"):
        return val == "True"
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
        return val[1:-1]
    if re.fullmatch(r"[+-]?[0-9]+", val):
        return int(val)
    if re.fullmatch(r"[+-]?([0-9]*\.[0-9]+|[0-9]+\.[0-9]*)([eE][+-]?[0-9]+)?", val):
        return float(val)
    if (val.startswith("(") and val.endswith(")")) or (val.startswith("[") and val.endswith("]")):
        items = _split_top_level(val[1:-1])
        if items and all(re.fullmatch(r"[+-]?[0-9]+", i) for i in items):
            return [int(i) for i in items]
    return _Opaque(val)


# The only callables whose structured record may grant the identity-copy rule:
# the trusted torch library function, matched by OBJECT IDENTITY at capture
# time (callee-7).  A same-named user function is never one of them.
_TRUSTED_INTERPOLATE_QUALNAMES = frozenset({"torch.nn.functional.interpolate"})


def callee_identity(func):
    """Identity of the EXECUTED callable, from the object itself (callee-7).

    ``is_torch_library`` is True only when ``func`` *is* (object identity)
    ``torch.nn.functional.interpolate``; ``bound_as`` then names it.  The
    module/qualname of the actual object are recorded for diagnostics but
    never decide anything: a user function named ``interpolate`` in a module
    of any name has ``is_torch_library: False``.
    """
    import torch.nn.functional as F
    trusted = {"torch.nn.functional.interpolate": F.interpolate}
    bound_as = next((name for name, obj in trusted.items() if func is obj), None)
    return {"module": getattr(func, "__module__", None),
            "qualname": getattr(func, "__qualname__", getattr(func, "__name__", None)),
            "type": type(func).__name__,
            "is_torch_library": bound_as is not None,
            "bound_as": bound_as}


# every field a v1 record must carry; a missing key is an incomplete record,
# which is NOT the same as a key bound to None (record-8)
_INTERPOLATE_RECORD_FIELDS = ("schema", "mode", "size", "scale_factor", "align_corners",
                              "recompute_scale_factor", "antialias", "callee_identity")
# callee_identity: ADMISSION fields (validated for type before any membership
# test; a wrong type is `interpolate_record_malformed`) and DIAGNOSTIC fields
# (recorded for the reader, never consulted by the admission rule; their
# absence or type is an observation, not a refusal) (schema-9)
_CALLEE_ADMISSION_FIELDS = {"module": (str,), "qualname": (str,), "is_torch_library": (bool,), "bound_as": (str, type(None))}
_CALLEE_DIAGNOSTIC_FIELDS = {"type": (str,)}


def interpolate_evidence_record(func, args, kwargs):
    """Capture-time, STRUCTURED record of an ``F.interpolate`` call's scalar arguments.

    Built by the capture shim from the ACTUAL argument objects (never from a
    display string): the call is bound to ``torch.nn.functional.interpolate``'s
    signature so positional and keyword spellings and the signature defaults
    are all resolved.  Types are preserved: ``size`` is recorded as
    ``{"type": "int"|"tuple"|"list"|..., "value": ...}``; ``scale_factor`` as
    a float, a list of floats or None; ``mode`` as a str; ``align_corners`` /
    ``recompute_scale_factor`` / ``antialias`` as recorded.  Returns None when
    the call cannot be bound (the admission then refuses).
    """
    import inspect
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return None
    a = bound.arguments

    def _scalar(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (tuple, list)) and all(isinstance(x, (bool, int, float)) for x in v):
            return [x for x in v]
        return {"unrecorded_type": type(v).__name__}

    size = a.get("size")
    if size is None or isinstance(size, (bool, int)):
        size_rec = {"type": type(size).__name__, "value": size}
    elif isinstance(size, (tuple, list)) and all(isinstance(x, int) and not isinstance(x, bool) for x in size):
        size_rec = {"type": type(size).__name__, "value": [int(x) for x in size]}
    else:
        size_rec = {"type": type(size).__name__, "value": {"unrecorded_type": type(size).__name__}}
    return {"schema": "solar-interpolate-evidence-v1", "mode": a.get("mode"), "size": size_rec,
            "scale_factor": _scalar(a.get("scale_factor")), "align_corners": _scalar(a.get("align_corners")),
            "recompute_scale_factor": _scalar(a.get("recompute_scale_factor")), "antialias": _scalar(a.get("antialias")),
            "callee_identity": callee_identity(func)}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def interpolate_record_refusal(evidence):
    """Why a structured record cannot ground the identity-copy rule, or None when it is admissible input.

    Schema validation of the v1 record (record-8): every field in
    :data:`_INTERPOLATE_RECORD_FIELDS` must be PRESENT (``key in record``) with
    the type the builder emits; an absent key is ``interpolate_record_incomplete``
    and is never read as a bound None.  The genuine signature default that
    capture records (``apply_defaults`` -> present ``None``) is a present, bound
    value and is accepted here.  Callee binding (callee-7): the record must
    name a trusted torch library callable by object identity
    (``callee_identity.is_torch_library`` True and ``bound_as`` allow-listed);
    anything else is ``interpolate_callee_untrusted``.

    Record fields fall in two classes.  ADMISSION fields decide the rule and
    are type-validated before they are read: the seven scalar/size fields
    above and, inside ``callee_identity``, ``module`` (str), ``qualname``
    (str), ``is_torch_library`` (bool) and ``bound_as`` (str, or None only
    when ``is_torch_library`` is False).  A wrong type anywhere is
    ``interpolate_record_malformed`` naming the field and the observed type;
    it is established BEFORE any hash or set membership so a malformed nested
    value can never raise out of this function (schema-9).  DIAGNOSTIC
    fields (``callee_identity.type``) are recorded for the reader and never
    consulted by the rule; their absence or type is an observation only.
    Any unexpected exception while validating is itself reported as a
    malformed record (never propagated).
    """
    try:
        return _interpolate_record_refusal(evidence)
    except Exception as exc:  # pragma: no cover - defensive: a refusal, never a crash
        return f"interpolate_record_malformed: {type(exc).__name__} while validating the record: {exc}"


def _interpolate_record_refusal(evidence):
    if not isinstance(evidence, dict):
        return "interpolate_record_missing: no structured record"
    if evidence.get("schema") != "solar-interpolate-evidence-v1":
        return f"interpolate_record_incomplete: schema {evidence.get('schema')!r} is not solar-interpolate-evidence-v1"
    missing = [k for k in _INTERPOLATE_RECORD_FIELDS if k not in evidence]
    if missing:
        return "interpolate_record_incomplete: absent field(s) " + ", ".join(missing)
    mode = evidence["mode"]
    if mode is not None and not isinstance(mode, str):
        return f"interpolate_record_incomplete: mode has type {type(mode).__name__}"
    size = evidence["size"]
    if not (isinstance(size, dict) and isinstance(size.get("type"), str) and "value" in size):
        return "interpolate_record_incomplete: size is not a {type, value} record"
    sf = evidence["scale_factor"]
    if not (sf is None or _is_number(sf) or (isinstance(sf, list) and all(_is_number(x) for x in sf))):
        return f"interpolate_record_incomplete: scale_factor has type {type(sf).__name__}"
    for k in ("align_corners", "recompute_scale_factor", "antialias"):
        v = evidence[k]
        if not (v is None or isinstance(v, (bool, int, float, str, list))):
            return f"interpolate_record_incomplete: {k} has type {type(v).__name__}"
    ci = evidence["callee_identity"]
    if not isinstance(ci, dict):
        return f"interpolate_record_malformed: callee_identity has type {type(ci).__name__} (expected a record)"
    missing_ci = [k for k in _CALLEE_ADMISSION_FIELDS if k not in ci]
    if missing_ci:
        return "interpolate_record_incomplete: callee_identity lacks admission field(s) " + ", ".join(missing_ci)
    # nested TYPE validation before any membership test (schema-9)
    for k, types in _CALLEE_ADMISSION_FIELDS.items():
        if not isinstance(ci[k], types) or (k == "is_torch_library" and type(ci[k]) is not bool):
            return f"interpolate_record_malformed: callee_identity.{k} has type {type(ci[k]).__name__}"
    if ci["is_torch_library"] is True and not isinstance(ci["bound_as"], str):
        return "interpolate_record_malformed: callee_identity.bound_as is None although is_torch_library is True"
    if ci["is_torch_library"] is not True or ci["bound_as"] not in _TRUSTED_INTERPOLATE_QUALNAMES:
        return (f"interpolate_callee_untrusted: executed callable {ci['module']}.{ci['qualname']} "
                f"is not the torch library function (is_torch_library={ci['is_torch_library']!r})")
    return None


def identity_interpolate_evidence(evidence, input_shape, output_shape):
    """Evidence that a captured ``F.interpolate`` call is an identity-size copy, else None.

    Consumes ONLY the structured capture-time record written by
    :func:`interpolate_evidence_record` (``module_args["interpolate_evidence"]``);
    a display string (``raw_attributes``) is never evidence, so graphs captured
    before this record exists refuse (incomplete) rather than being parsed
    (parser-5).  The rule fires when ``mode`` is ``linear`` / ``nearest`` /
    ``nearest-exact``, ``scale_factor`` is ``None``, the recorded output shape
    equals the input shape, and ``size`` is either a scalar int equal to every
    spatial extent or a tuple/list whose length equals the spatial rank with
    every element equal to the corresponding input extent (size-6: a
    one-element sequence on a multi-dimensional input is NOT broadcast; only a
    scalar is).  Before any of that, :func:`interpolate_record_refusal` must
    accept the record: every v1 field present with its type (an absent field is
    never read as None; record-8) and the executed callable bound by
    object identity to ``torch.nn.functional.interpolate`` (a same-named user
    function is refused; callee-7).  A same-size linear resample selects each sample with weights
    1 and 0 and nearest is re-indexing, so the arithmetic is zero and the
    traffic is one read plus one write of the recorded tensor.
    """
    if interpolate_record_refusal(evidence) is not None:
        return None
    if input_shape is None or output_shape is None:
        return None
    if list(input_shape) != list(output_shape) or len(input_shape) < 3:
        return None
    mode = evidence["mode"]
    if not isinstance(mode, str) or mode not in _IDENTITY_INTERPOLATE_MODES:
        return None
    if evidence["scale_factor"] is not None:
        return None
    size_rec = evidence["size"]
    size, stype = size_rec["value"], size_rec["type"]
    spatial = [int(v) for v in list(input_shape)[2:]]
    if stype == "int" and isinstance(size, int) and not isinstance(size, bool):
        if any(s != size for s in spatial):
            return None
        size_list = [size] * len(spatial)
    elif stype in ("tuple", "list") and isinstance(size, list) and all(isinstance(x, int) for x in size):
        if len(size) != len(spatial) or size != spatial:
            return None
        size_list = list(size)
    else:
        return None
    return {"mode": mode, "size": size_list, "size_type": stype, "scale_factor": None,
            "input_shape": [int(v) for v in input_shape], "output_shape": [int(v) for v in output_shape],
            "rule": "identity_size_resize_is_a_copy", "binding": "structured capture-time record (solar-interpolate-evidence-v1)",
            "callee": evidence["callee_identity"]["bound_as"]}
