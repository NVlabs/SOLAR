# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-matched numerical-validation receipts for the strict SOL path.

A receipt records ONE numerical verification run. Its evidence — the SHA-256
of the exact source/DSL bytes imported and executed, the tolerance actually
applied, the checker identity (module path + file hash), the numeric mode and
the validation tensor shapes — is captured INSIDE ``run_verification``
(``solar.translate._runner.ExecutionEvidence``, sealed with a module-private
token), never supplied afterwards by the caller.  ``emit_receipt`` refuses a
result that carries no execution evidence, and refuses to emit when the
current file bytes differ from the bytes that run verified: a passing result
can never be rebound to subsequently changed code (round-4 P1).

The PREPARED REFERENCE is bound too (round-5 P1): ``run_verification`` hashes
the reference source it actually executed (what ``get_source()`` returned,
or the pytorch model file text), and both emission and strict entry re-derive
the CURRENT reference through the same frontend contract and compare — an
adapter whose ``get_source()`` reads an external ``reference.py`` (the
documented normal pattern) cannot keep an old receipt alive after that file
changes.  When the reference cannot be re-derived (no ``get_source()``, or
it fails), strict qualification is refused rather than presumed.

The reference's LOCAL EXECUTION DEPENDENCIES are bound as well (round-6 P1):
modules imported while the reference was prepared/executed whose files live
in the adapter's directory tree are hashed into the evidence and re-checked
at emission and strict entry, so a reference like ``from helper import run``
cannot keep an old receipt alive after ``helper.py`` changes.  Modules newly
imported from OUTSIDE the supported closure (adapter directory, Python
environment, solar checkout) make the closure unprovable: strict
qualification is refused, never presumed.

The hashes describe the code that ACTUALLY RAN (round-7/8/9 P1): before the
reference is prepared, ``run_verification`` drops every cached module whose
file lives outside the trusted environment/checkout (a fresh, controlled
import snapshot, recorded in the evidence as ``evicted_cached_modules``),
and during preparation every untrusted ``.py`` import goes through a
controlled loader that compiles the CURRENT source bytes directly —
bytecode caches are bypassed, so a timestamp-valid stale ``.pyc`` cannot
execute old code while the new source gets bound.  The MAIN adapter/model
file is loaded the same way (round-9 P1): every supported frontend's direct
module load routes through ``solar.einsum_lib.source_load``, and the runner
refuses strict qualification unless the adapter's executed-bytes hash
equals the source hash it bound.  Reference re-derivation at emission and
strict entry imports the adapter through the same source-bytes loader.
The recorded dependency hash IS the hash of the bytes handed to compile();
a local dependency whose executed code cannot be tied to source bytes
(sourceless bytecode, extension module, manual loading) is unsupported and
refuses strict qualification.

``check_strict`` recomputes the current hashes at strict-SOL entry and
refuses missing, malformed, failed, checker-error, stale or mismatched
receipts BEFORE any tracing or analysis runs.  The exact selected workload
(uuid, axes), the selected analysis dtype, and — when the caller states
expectations — the tolerance policy and checker identity are compared, not
merely required to be present.

Verdict kinds are recorded separately and deliberately:

* ``numerical``  — what the verification run established (pass/fail/error);
* ``structural`` / ``accounting`` / ``hardware`` — ``not_certified`` here: a
  numerical pass does NOT certify MAC/access counts or hardware calibration.

Known limits, stated rather than implied: the definition.json hash is bound
at emission (verification does not read it) and re-checked at strict entry;
modules under the Python environment and the solar checkout are trusted as
environment, not hashed per-receipt (the same trust verification itself
requires to run torch and the checker); re-deriving ``get_source()`` imports
the adapter module (the same execution trust already required to verify or
trace it); and nothing inside one Python process can stop an adversary with
code
execution from editing this module itself — the gate defends against stale
and rebound artifacts, not against a hostile interpreter.  The legacy
``sol`` path without a receipt remains available and is documented as
unvalidated.

Validation SHAPES are bound, not claimed (audit finding A6a, schema v5):
``workload.validation_axes`` is DERIVED from the executed
``execution.validation_shapes`` and the definition's declared input shapes;
a caller-supplied ``validation_axes`` is cross-checked against that
derivation and refused when it contradicts it, and ``check_strict``
repeats the cross-check at strict entry from the receipt's own execution
block.  ``workload.capped_validation`` says whether the executed shapes
differ from the selected workload's axes.  Per-tensor dtypes are recorded
as measured (A6b): the source-declared input dtypes, the dtypes fed to the
reference, and the dtypes the reference returned — ``verification_dtype``
remains the IR precision flag.  A pass whose reference outputs were all
zeros is recorded (``error_stats.vacuous_outputs``,
``execution.vacuous_reference_outputs``; A4) and refused by
``check_strict`` unless ``allow_vacuous=True`` is stated explicitly.
Receipts of earlier schemas are not trusted: they could claim validation
at the original axes for a capped run.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

SCHEMA = "solar-receipt-v5"

_REQUIRED = (
    "schema", "verdicts", "error_stats", "hashes", "workload",
    "dtype", "layout_transforms", "tolerance", "checker", "execution", "utc",
)
_REQUIRED_HASHES = ("source_sha256", "dsl_sha256", "definition_sha256",
                    "reference_sha256", "dependencies")
_REQUIRED_WORKLOAD = ("uuid", "axes", "validation_axes",
                      "validation_axes_source", "capped_validation")
_REQUIRED_EXECUTION = ("source_type", "validation_shapes",
                       "vacuous_reference_outputs")


class ReceiptRefusal(RuntimeError):
    """Strict SOL refused: no trusted numerical-validation receipt."""


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _current_reference_text(source_path, source_type: str) -> str:
    """Re-derive the reference source the frontend would prepare RIGHT NOW.

    For ``pytorch`` the reference is the model file itself.  For adapter
    contracts (cuda/jax/triton) it is ``get_source()`` — re-called here so an
    externally read dependency (e.g. a sibling reference.py, the documented
    normal pattern) is captured at its CURRENT bytes.  When the reference
    cannot be re-derived, strict qualification is refused rather than
    presumed: ReceiptRefusal, not a silent pass.
    """
    source_path = Path(source_path)
    if source_type == "pytorch":
        return source_path.read_text(encoding="utf-8")
    # Current SOURCE bytes only — a direct spec_from_file_location load
    # would consult __pycache__, and a timestamp-valid stale .pyc could
    # re-derive an OLD reference for changed adapter source (round-9 P1).
    from solar.einsum_lib.source_load import load_source_module
    try:
        module = load_source_module(
            source_path,
            f"_solar_receipt_ref_{abs(hash(str(source_path)))}")
    except Exception as e:
        raise ReceiptRefusal(
            f"reference provenance cannot be established: adapter "
            f"{source_path.name} failed to load ({type(e).__name__}: {e})"
        ) from e
    if not hasattr(module, "get_source"):
        raise ReceiptRefusal(
            f"reference provenance cannot be established: adapter "
            f"{source_path.name} defines no get_source(); strict receipts "
            f"require a re-derivable (or self-contained) reference")
    try:
        return str(module.get_source())
    except Exception as e:
        raise ReceiptRefusal(
            f"reference provenance cannot be established: get_source() "
            f"raised {type(e).__name__}: {e}") from e


def _check_dependencies(dependencies, when: str):
    """Re-hash the recorded local execution dependencies; refuse changes."""
    for path, then in dict(dependencies).items():
        try:
            now = _sha256(path)
        except OSError as e:
            raise ReceiptRefusal(
                f"helper dependency missing/unreadable ({when}): {path} "
                f"({e}) — the reference's execution dependency closure no "
                f"longer matches the verified run") from e
        if now != then:
            raise ReceiptRefusal(
                f"helper dependency changed since verification ({when}): "
                f"{path} was {then[:12]}…, is now {now[:12]}… — the imported "
                f"reference dependency is bound, not only the reference "
                f"text; re-verify")


def _check_reference(source_path, source_type: str, then: str, when: str):
    now = hashlib.sha256(
        _current_reference_text(source_path, source_type)
        .encode("utf-8")).hexdigest()
    if now != then:
        raise ReceiptRefusal(
            f"reference changed since verification ({when}): the run "
            f"prepared reference {then[:12]}…, the adapter now yields "
            f"{now[:12]}… — the executed reference (e.g. an external "
            f"reference.py) is bound, not only the adapter bytes; re-verify")


def workload_identity(definition_dir, workload_idx: int = 0):
    """(uuid, axes) of the selected ``workload.jsonl`` line, VERBATIM.

    Without a ``workload.jsonl`` the identity is ``definition-only:<idx>``
    and axes is None (strict SOL then compares no axes).  Shared by
    ``sol(strict=True)`` and receipt emission so both sides of the gate
    name the workload the same way.
    """
    wl_file = Path(definition_dir) / "workload.jsonl"
    if wl_file.is_file():
        lines = wl_file.read_text().splitlines()
        wl = json.loads(lines[workload_idx])
        return wl.get("uuid", f"workload-{workload_idx}"), wl.get("axes")
    return f"definition-only:{workload_idx}", None


def _load_definition(definition_path) -> dict:
    try:
        defn = json.loads(Path(definition_path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ReceiptRefusal(
            f"definition unreadable ({Path(definition_path)}): {e}") from e
    return defn if isinstance(defn, dict) else {}


def definition_const_axes(definition_path) -> dict:
    """``{axis: value}`` for the definition's const axes (definition-only
    workload identity)."""
    defn = _load_definition(definition_path)
    return {str(k): v.get("value")
            for k, v in (defn.get("axes") or {}).items()
            if isinstance(v, dict) and v.get("type") == "const"}


def derive_validation_axes(definition_path, validation_shapes) -> dict:
    """Axis values the verification ACTUALLY executed at.

    Binds each declared input ``shape`` symbol that names a definition axis
    to the corresponding executed dimension of the same-named tensor in
    ``validation_shapes`` (``((name, (dims...)), ...)`` as recorded in
    ``ExecutionEvidence``).  Literal and expression dimensions are not
    inverted.  Two tensors binding one axis to different values is a
    contradiction and refuses (ReceiptRefusal) rather than picking one.
    Returns ``{}`` when nothing is derivable (no same-named declared input).
    """
    defn = _load_definition(definition_path)
    axis_names = set((defn.get("axes") or {}).keys())
    executed = {str(name): tuple(dims) for name, dims in
                (dict(validation_shapes) if not isinstance(validation_shapes, dict)
                 else validation_shapes).items()}
    bound: dict = {}
    for name, spec in (defn.get("inputs") or {}).items():
        dims = executed.get(str(name))
        declared = (spec or {}).get("shape") if isinstance(spec, dict) else None
        if dims is None or not declared or len(declared) != len(dims):
            continue
        for sym, value in zip(declared, dims):
            if not (isinstance(sym, str) and sym in axis_names):
                continue
            value = int(value)
            if sym in bound and bound[sym] != value:
                raise ReceiptRefusal(
                    f"executed validation shapes are inconsistent with the "
                    f"definition: axis {sym!r} is {bound[sym]} in one input "
                    f"and {value} in {name!r}; the run cannot be bound to "
                    f"one set of validation axes")
            bound[sym] = value
    return bound


def _resolve_validation_axes(definition_path, validation_shapes,
                             validation_axes: Optional[dict]):
    """(validation_axes, source) honoring derivation first (A6a).

    * derived and no caller value      -> (derived, "executed_shapes")
    * derived and caller value         -> caller value checked key-by-key
                                          against the derivation; any
                                          contradiction refuses
                                          -> (caller, "caller_checked")
    * nothing derivable, caller value  -> (caller, "caller_unchecked")
    * nothing derivable, no caller     -> ReceiptRefusal (never default to
                                          the ORIGINAL workload axes)
    """
    derived = derive_validation_axes(definition_path, validation_shapes)
    if validation_axes is None:
        if not derived:
            raise ReceiptRefusal(
                "validation_axes cannot be derived from the executed shapes "
                "(no definition input shares a name with an executed "
                "tensor) and none were given; pass validation_axes= with "
                "the capped shapes that were actually verified — the "
                "original workload axes are never assumed")
        return dict(derived), "executed_shapes"
    claimed = {str(k): v for k, v in dict(validation_axes).items()}
    if not derived:
        return claimed, "caller_unchecked"
    for k, v in derived.items():
        if k in claimed and int(claimed[k]) != int(v):
            raise ReceiptRefusal(
                f"validation_axes claim {k}={claimed[k]} but the run executed "
                f"{k}={v} (execution.validation_shapes); the receipt binds "
                f"executed shapes, not claims")
    return claimed, "caller_checked"


def _capped(axes: Optional[dict], validation_axes: dict) -> bool:
    """True when any axis shared by the selected workload and the executed
    validation differs: the SOL at ``axes`` rests on a capped verification."""
    axes = dict(axes or {})
    return any(k in axes and int(axes[k]) != int(v)
               for k, v in validation_axes.items())


def _engine_source_record():
    try:
        from solar.engine_source import engine_source
        return engine_source()
    except Exception as exc:  # never refuse a receipt over provenance lookup
        return {"error": str(exc)}


def emit_receipt(verify_result, *, source_path, dsl_path, definition_path,
                 workload_uuid: str, axes: Optional[dict],
                 dtype: str, out_path,
                 validation_axes: Optional[dict] = None,
                 layout_transforms: Optional[list] = None) -> Path:
    """Write a receipt from an actual verification execution.

    ``verify_result`` must be a ``VerifyResult`` returned by
    ``run_verification`` (it carries sealed ``ExecutionEvidence``); a plain
    dict or a hand-constructed ``VerifyResult`` is rejected.  The current
    bytes of ``source_path``/``dsl_path`` must equal the bytes that execution
    verified, otherwise ``ReceiptRefusal`` is raised — tolerance, checker
    identity and hashes are copied from the evidence, never from the caller.

    ``axes`` is the selected workload's axes VERBATIM (``workload.jsonl``
    line; ``sol(strict=True)`` compares it for equality) or None for a
    definition-only identity.  ``validation_axes`` is derived from the
    executed shapes when omitted and cross-checked when given (see
    ``_resolve_validation_axes``); it is never defaulted to ``axes``.
    """
    from solar.translate._runner import ExecutionEvidence, VerifyResult

    ev = getattr(verify_result, "execution", None)
    if not isinstance(verify_result, VerifyResult) or \
            not isinstance(ev, ExecutionEvidence):
        raise TypeError(
            "emit_receipt requires a VerifyResult carrying sealed execution "
            "evidence from an actual run_verification execution; got "
            f"{type(verify_result).__name__} (type alone is not proof of a "
            "run)")
    for label, path, then in (("source", source_path, ev.source_sha256),
                              ("DSL", dsl_path, ev.dsl_sha256)):
        now = _sha256(path)
        if now != then:
            raise ReceiptRefusal(
                f"{label} bytes changed since verification: the run verified "
                f"{then[:12]}…, {Path(path).name} is now {now[:12]}… — "
                f"re-verify instead of rebinding the old result")
    if ev.unsupported_dependencies:
        raise ReceiptRefusal(
            "dependency provenance cannot be established: the reference "
            "imported modules outside the supported closure (adapter "
            "directory / Python environment / solar checkout): "
            f"{list(ev.unsupported_dependencies)} — strict qualification is "
            "refused; use a self-contained or adapter-local reference")
    _check_dependencies(ev.local_dependencies, "at receipt emission")
    _check_reference(source_path, ev.source_type, ev.reference_sha256,
                     "at receipt emission")
    if verify_result.shape_only:
        numerical = "error"
        note = "shape-only run is not a numerical validation"
    elif verify_result.verified:
        numerical = "pass"
        note = verify_result.message
    else:
        numerical = "fail"
        note = verify_result.message
    resolved_axes, axes_source = _resolve_validation_axes(
        definition_path, ev.validation_shapes, validation_axes)
    vacuous = sorted(set(getattr(verify_result, "vacuous_outputs", ()) or ())
                     | set(ev.vacuous_reference_outputs))
    receipt = {
        "schema": SCHEMA,
        "verdicts": {
            "numerical": numerical,
            "structural": "not_certified",
            "accounting": "not_certified",
            "hardware": "not_certified",
        },
        "error_stats": {
            "max_abs_error": verify_result.max_abs_error,
            "message": note,
            # outputs whose reference was all zeros: a pass on them is
            # vacuous (weak evidence); strict entry refuses unless the
            # caller states allow_vacuous=True
            "vacuous_outputs": vacuous,
        },
        "hashes": {
            "source_sha256": ev.source_sha256,
            "dsl_sha256": ev.dsl_sha256,
            "definition_sha256": _sha256(definition_path),
            "reference_sha256": ev.reference_sha256,
            "dependencies": dict(ev.local_dependencies),
        },
        "workload": {
            "uuid": workload_uuid,
            "axes": dict(axes or {}),
            # the axes the verification EXECUTED at (derived from
            # execution.validation_shapes), not the selected workload's
            "validation_axes": resolved_axes,
            "validation_axes_source": axes_source,
            # True when the SOL workload axes differ from the executed
            # validation axes: original-shape SOL on a capped verification
            "capped_validation": _capped(axes, resolved_axes),
        },
        "dtype": dtype,
        "layout_transforms": list(layout_transforms or []),
        "tolerance": {"rtol": ev.rtol, "atol": ev.atol,
                      "min_tol": ev.min_tol},
        "checker": {"identity": ev.checker_identity,
                    "sha256": ev.checker_sha256},
        # Self-binding provenance of the engine that produced this receipt
        # (119-TRACE-PROVENANCE-1): resolved solar.__file__, git commit/blob.
        "engine_source": _engine_source_record(),
        "execution": {
            "source_type": ev.source_type,
            "use_fp64": ev.use_fp64,
            "fast": ev.fast,
            "verification_dtype": ev.verification_dtype,
            "config_overrides": [list(item) for item in ev.config_overrides],
            "evicted_cached_modules": list(ev.evicted_cached_modules),
            "validation_shapes": {name: list(shape)
                                  for name, shape in ev.validation_shapes},
            # measured per-tensor dtypes (A6b); verification_dtype above is
            # the IR precision flag, not the reference's precision
            "input_dtypes": dict(ev.input_dtypes),
            "reference_input_dtypes": dict(ev.reference_input_dtypes),
            "reference_output_dtypes": dict(ev.reference_output_dtypes),
            "vacuous_reference_outputs": list(ev.vacuous_reference_outputs),
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(receipt, indent=1))
    return out_path


def read_receipt(path) -> dict:
    """Load and schema-check a receipt; malformed receipts are refused."""
    path = Path(path)
    if not path.is_file():
        raise ReceiptRefusal(f"receipt missing: {path}")
    try:
        receipt = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ReceiptRefusal(f"receipt malformed: {path}: {e}") from e
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ReceiptRefusal(
            f"receipt has wrong/missing schema (need {SCHEMA!r}; receipts "
            f"from before execution-bound evidence are not trusted): {path}")
    missing = [k for k in _REQUIRED if k not in receipt]
    missing += [f"hashes.{k}" for k in _REQUIRED_HASHES
                if k not in receipt.get("hashes", {})]
    missing += [f"workload.{k}" for k in _REQUIRED_WORKLOAD
                if k not in receipt.get("workload", {})]
    missing += [f"execution.{k}" for k in _REQUIRED_EXECUTION
                if k not in receipt.get("execution", {})]
    if missing:
        raise ReceiptRefusal(f"receipt missing fields {missing}: {path}")
    if not receipt["tolerance"]:
        raise ReceiptRefusal(f"receipt has empty tolerance policy: {path}")
    if not receipt["checker"].get("identity"):
        raise ReceiptRefusal(f"receipt has no checker identity: {path}")
    return receipt


def check_strict(receipt_path, *, source_path, dsl_path, definition_path,
                 workload_uuid: str, axes: Optional[dict] = None,
                 dtype: Optional[str] = None,
                 expected_tolerance: Optional[dict] = None,
                 expected_checker: Optional[str] = None,
                 allow_vacuous: bool = False) -> dict:
    """Validate a receipt against the CURRENT files and selection.

    Recomputes source/DSL/definition hashes from the given paths, requires a
    numerical PASS, and requires the exact workload uuid (and axes/dtype when
    given) to match.  ``expected_tolerance``/``expected_checker`` let the
    caller pin the recorded policy and checker identity; when given they are
    compared field-for-field, not merely required to exist.  A pass whose
    reference outputs were all zeros is refused unless ``allow_vacuous=True``
    (the caller then states that zero is the genuine reference).  The
    receipt's ``validation_axes`` are re-derived from its own executed
    shapes and the current definition and refused on contradiction.  Raises
    ReceiptRefusal otherwise.  Returns the receipt.
    """
    receipt = read_receipt(receipt_path)
    if receipt["verdicts"]["numerical"] != "pass":
        raise ReceiptRefusal(
            f"receipt numerical verdict is "
            f"'{receipt['verdicts']['numerical']}', not 'pass'")
    vacuous = sorted(set(receipt["error_stats"].get("vacuous_outputs") or [])
                     | set(receipt["execution"].get("vacuous_reference_outputs")
                           or []))
    if vacuous and not allow_vacuous:
        raise ReceiptRefusal(
            f"receipt pass is vacuous: reference output(s) {vacuous} were "
            f"all zeros, so an all-zero translation would also have passed; "
            f"strict numerical proof is refused (pass allow_vacuous=True "
            f"only after confirming from the reference source that zero is "
            f"the genuine result)")
    current = {
        "source_sha256": _sha256(source_path),
        "dsl_sha256": _sha256(dsl_path),
        "definition_sha256": _sha256(definition_path),
    }
    for key, now in current.items():
        then = receipt["hashes"][key]
        if then != now:
            raise ReceiptRefusal(
                f"stale/mismatched receipt: {key} was {then[:12]}…, "
                f"current file is {now[:12]}…")
    _check_dependencies(receipt["hashes"]["dependencies"],
                        "at strict-SOL entry")
    source_type = (receipt.get("execution") or {}).get("source_type")
    if not source_type:
        raise ReceiptRefusal("receipt records no execution source_type; the "
                             "reference cannot be re-derived")
    _check_reference(source_path, source_type,
                     receipt["hashes"]["reference_sha256"],
                     "at strict-SOL entry")
    if receipt["workload"]["uuid"] != workload_uuid:
        raise ReceiptRefusal(
            f"receipt validates workload {receipt['workload']['uuid']!r}, "
            f"strict SOL selected {workload_uuid!r}")
    if axes is not None and dict(receipt["workload"]["axes"]) != dict(axes):
        raise ReceiptRefusal("receipt axes differ from the selected shape")
    if dtype is not None and receipt["dtype"] != dtype:
        raise ReceiptRefusal(
            f"receipt dtype {receipt['dtype']!r} != selected {dtype!r}")
    if expected_tolerance is not None and \
            dict(receipt["tolerance"]) != dict(expected_tolerance):
        raise ReceiptRefusal(
            f"receipt tolerance {receipt['tolerance']} != expected "
            f"tolerance {expected_tolerance}")
    if expected_checker is not None and \
            receipt["checker"]["identity"] != expected_checker:
        raise ReceiptRefusal(
            f"receipt checker {receipt['checker']['identity']!r} != "
            f"expected checker {expected_checker!r}")
    # The recorded validation axes must agree with the receipt's OWN
    # executed shapes under the current (hash-matched) definition.
    derived = derive_validation_axes(
        definition_path, receipt["execution"]["validation_shapes"])
    recorded = receipt["workload"]["validation_axes"]
    for k, v in derived.items():
        if k in recorded and int(recorded[k]) != int(v):
            raise ReceiptRefusal(
                f"receipt validation_axes claim {k}={recorded[k]} but its "
                f"execution.validation_shapes executed {k}={v}; the receipt "
                f"is self-contradictory and is refused")
    return receipt


def emit_workload_receipt(verify_result, *, source_path, dsl_path,
                          definition_dir, dtype: str, out_path,
                          workload_idx: int = 0,
                          validation_axes: Optional[dict] = None,
                          layout_transforms: Optional[list] = None) -> Path:
    """Emit a receipt for the workload ``sol(strict=True)`` will select.

    The ONE receipt path shared by ``scripts/verify_saved_ir.py
    --emit-receipt``, ``solar-translate verify --emit-receipt`` and library
    callers (audit finding A5): resolves ``<definition_dir>/definition.json``
    and the ``workload.jsonl`` line ``workload_idx`` exactly as
    ``solar.translate.sol`` does (``workload_identity``), then calls
    ``emit_receipt``.  For a definition-only problem the recorded axes are
    the definition's const axes.  ``dtype`` must equal the ``precision``
    the later SOL run analyzes at.
    """
    definition_dir = Path(definition_dir)
    definition_path = definition_dir / "definition.json"
    if not definition_path.is_file():
        raise ReceiptRefusal(f"definition missing: {definition_path}")
    uuid, axes = workload_identity(definition_dir, workload_idx)
    if axes is None:
        axes = definition_const_axes(definition_path)
    return emit_receipt(verify_result, source_path=source_path,
                        dsl_path=dsl_path, definition_path=definition_path,
                        workload_uuid=uuid, axes=axes, dtype=dtype,
                        out_path=out_path, validation_axes=validation_axes,
                        layout_transforms=layout_transforms)
