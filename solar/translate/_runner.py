# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared verification runner for the translation facade.

This is the core of ``scripts/verify_saved_ir.py`` factored into a library
function so both the script (CLI, byte-compatible output) and
``solar.translate.verify`` (programmatic) drive the same code path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import importlib
import importlib.machinery
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


def coerce_config_value(text: str):
    """Turn a --config value into int, float, bool, None, or str."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


# Appended to the sized copy before the CONFIG overrides: fail loudly when the
# module has no CONFIG dict for --config to override.
_CONFIG_GUARD = """

# ---- injected by scripts/verify_saved_ir.py ----
try:
    CONFIG
except NameError:
    raise SystemExit(
        'verify_saved_ir.py: --config given but this module defines no CONFIG dict'
    )
"""


@contextlib.contextmanager
def sized_model(model_path: Path, overrides: dict, workdir: Path):
    """Yield a copy of *model_path* whose CONFIG carries *overrides*.

    Appending works because module-level statements run at import, before
    ``get_inputs()`` is ever called, and because the baselines read ``CONFIG``
    at call time rather than capturing its values at definition.

    The copy goes next to the original when possible so that any sibling module
    the model imports still resolves; a system temp directory is only the
    fallback for a read-only source tree.
    """
    if not overrides:
        yield model_path
        return

    dest = model_path.parent / f".sized_{model_path.name}"
    try:
        shutil.copy(model_path, dest)
    except OSError:
        dest = workdir / f"sized_{model_path.name}"
        shutil.copy(model_path, dest)
    try:
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(_CONFIG_GUARD)
            for key, value in overrides.items():
                fh.write(f"CONFIG[{key!r}] = {value!r}\n")
        yield dest
    finally:
        dest.unlink(missing_ok=True)


def format_shapes(shapes: dict) -> str:
    """One indented ``name (shape)`` line per tensor, as the script prints them."""
    return "\n".join(f"    {name:<12} {tuple(shape)}" for name, shape in shapes.items())


# Module-private seal: only run_verification can mint ExecutionEvidence, so a
# receipt can only ever be emitted for bytes/options an actual verification
# executed (isinstance on the publicly constructible VerifyResult is not
# proof of a run — round-4 P1).
_EVIDENCE_TOKEN = object()


@dataclass(frozen=True)
class ExecutionEvidence:
    """What one verification execution ACTUALLY ran, captured in-run.

    Hashes are of the exact bytes imported/executed; tolerance, checker and
    numeric mode are the values applied, not caller claims. Sealed: direct
    construction raises, run_verification passes the module-private token.
    """

    source_sha256: str = ""
    dsl_sha256: str = ""
    # sha256 of the prepared reference source the run actually compared
    # against (e.g. the text get_source() returned) — round-5 P1: adapter
    # bytes alone do not pin an externally read reference.
    reference_sha256: str = ""
    source_type: str = ""
    rtol: float = 0.0
    atol: float = 0.0
    min_tol: float = 0.0
    use_fp64: bool = True
    fast: bool = True
    verification_dtype: str = ""
    config_overrides: tuple = ()
    validation_shapes: tuple = ()   # ((tensor_name, (dims...)), ...)
    # Local execution dependencies of the reference (round-6 P1): modules
    # imported while the reference was prepared/executed whose files live in
    # the adapter's directory tree, as ((resolved_path, sha256), ...).
    local_dependencies: tuple = ()
    # Newly imported modules OUTSIDE the adapter dir, the Python environment
    # and the solar checkout: their provenance cannot be established, so
    # strict qualification must be refused.
    unsupported_dependencies: tuple = ()
    # Cached modules outside the trusted environment/checkout that were
    # DROPPED before reference preparation (round-7 P1): the fresh,
    # controlled import snapshot re-executes current on-disk bytes instead
    # of unproven cached code; recorded here for auditability.
    evicted_cached_modules: tuple = ()
    checker_identity: str = ""
    checker_sha256: str = ""
    # Per-tensor dtype provenance (audit finding A6b), measured in-run:
    #   input_dtypes            — the SOURCE-declared dtypes the frontend
    #                             derived (get_inputs()/model arrays as given)
    #   reference_input_dtypes  — the dtypes actually FED to the reference
    #                             (e.g. the cuda adapter receives fp64 casts)
    #   reference_output_dtypes — the dtypes the reference actually RETURNED
    #                             (the pytorch golden runs at native fp32
    #                             even when verification_dtype is fp64)
    # all as ((tensor_name, solar_dtype_string), ...).
    input_dtypes: tuple = ()
    reference_input_dtypes: tuple = ()
    reference_output_dtypes: tuple = ()
    # Reference outputs that were ALL ZEROS (or had no finite element) when
    # the reference was prepared (audit finding A4): a pass on such an
    # output is vacuous.  Sealed here, inside the run, so a receipt cannot
    # drop the fact after the fact.
    vacuous_reference_outputs: tuple = ()
    _token: object = None

    def __post_init__(self):
        if self._token is not _EVIDENCE_TOKEN:
            raise TypeError(
                "ExecutionEvidence is minted only inside run_verification; "
                "it cannot be constructed from caller-supplied claims")


# The solar checkout root (…/solar): modules under it are part of the tool
# itself, not reference dependencies.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def _python_environment_dirs():
    """Every directory the Python ENVIRONMENT installs packages into.

    The interpreter prefixes cover a wheel/conda install, but ``pip install
    --user`` (the CI image, ``install.sh``) places torch, numpy, pytest and
    solar's dependencies under the USER site directory, which is outside
    every prefix.  Evicting those from ``sys.modules`` mid-process evicts
    the host harness itself: on CI the dispatcher's first in-process
    verification dropped ``_pytest.main`` and detached
    ``_pytest.assertion.util`` from its package, and pytest crashed in
    session teardown (``NameError: name 'util' is not defined``; ``Could not
    obtain a node for scope Session``).  Installed-package directories are
    the environment, not reference code; a reference helper never lives in
    site-packages.
    """
    import site
    import sysconfig
    dirs = []
    try:
        paths = sysconfig.get_paths()
        dirs += [paths.get("purelib"), paths.get("platlib")]
    except Exception:  # pragma: no cover - sysconfig scheme unavailable
        pass
    for getter in ("getsitepackages", "getusersitepackages"):
        fn = getattr(site, getter, None)
        if fn is None:
            continue
        try:
            got = fn()
        except Exception:  # pragma: no cover - a broken site module
            continue
        dirs += list(got) if isinstance(got, (list, tuple)) else [got]
    return [d for d in dirs if d]


_TRUSTED_ROOTS_IN_USE = None


def use_trusted_roots(roots, digest=None):
    """Consume ONE trusted-roots computation made by the dispatcher parent
    (CI-FAIL-25, round 7): every later ``_safe_roots()`` in this process
    returns exactly this list instead of recomputing its own from the
    interpreter configuration.  ``digest`` (sha256 of the JSON list) is
    checked when given, so the list that arrives is the list the parent
    hashed into the phase metadata."""
    listed = [str(r) for r in roots]
    if digest is not None:
        have = hashlib.sha256(json.dumps(listed, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
        if have != digest:
            raise ValueError("trusted roots do not hash to the supplied digest")
    global _TRUSTED_ROOTS_IN_USE
    _TRUSTED_ROOTS_IN_USE = tuple(Path(r).resolve() for r in listed)
    return list(_TRUSTED_ROOTS_IN_USE)


def trusted_roots_in_use():
    """The parent-supplied roots this process consumes, or None when it
    derives its own (plain facade use outside the dispatcher)."""
    return None if _TRUSTED_ROOTS_IN_USE is None else list(_TRUSTED_ROOTS_IN_USE)


def _safe_roots():
    if _TRUSTED_ROOTS_IN_USE is not None:
        # the dispatcher parent computed the roots once and passed them down:
        # no second, independent derivation in the verify child
        return list(_TRUSTED_ROOTS_IN_USE)
    # base_prefix/base_exec_prefix matter in a venv: stdlib files resolve
    # under the BASE interpreter, not the venv prefix.  The site-packages
    # directories (user site included) are part of the environment even when
    # they lie outside every prefix (``pip install --user``).
    roots = [Path(r).resolve()
             for r in dict.fromkeys([sys.prefix, sys.exec_prefix,
                                     sys.base_prefix, sys.base_exec_prefix]
                                    + _python_environment_dirs())
             if r]
    roots.append(_CHECKOUT_ROOT)
    return roots


def _resolved_parent(fname):
    if not fname:
        return None
    try:
        return Path(fname).resolve().parent
    except (OSError, RuntimeError):
        return None


def _inside(fname, directory):
    if not fname or directory is None:
        return False
    try:
        path = Path(fname).resolve()
    except (OSError, RuntimeError):
        return False
    return directory in path.parents


def _evict_unproven_cached_modules():
    """Fresh, controlled import snapshot (round-7 P1).

    A cached module executes its OLD loaded code while the dependency scan
    hashes the NEW on-disk bytes — a repeated same-process verification
    could bind bytes it never executed.  Dropping every cached module whose
    file lives outside the trusted environment (sys.prefix/exec_prefix) and
    the solar checkout forces the reference's imports to re-execute the
    current on-disk bytes, so the post-run hashes describe the code that
    actually ran.  Evicted names are recorded in the evidence.
    """
    roots = _safe_roots()
    tv = sys.modules.get("torchview")
    torchview_dir = _resolved_parent(getattr(tv, "__file__", None))
    evicted = []
    for name, mod in list(sys.modules.items()):
        fname = getattr(mod, "__file__", None)
        # torchview is solar's own dependency, patched in-process once at
        # import (ensure_parameter_capture).  Evicting it when it lives
        # outside the trusted roots (--user or source-clone installs) would
        # hand every later trace in this process a pristine, unpatched copy.
        # Only the package's own files are exempt: a reference helper that
        # merely registers as ``torchview.<x>`` (extended __path__) is not.
        if (name == "torchview" or name.startswith("torchview.")) \
                and _inside(fname, torchview_dir):
            continue
        if not fname:
            continue
        try:
            path = Path(fname).resolve()
        except (OSError, RuntimeError):
            continue
        if any(root == path or root in path.parents for root in roots):
            continue
        del sys.modules[name]
        # ``from pkg import child`` resolves through the parent's attribute,
        # which survives the sys.modules deletion; detach it too so a retained
        # parent cannot hand back the stale module.
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if parent is not None and getattr(parent, child, None) is mod:
            try:
                delattr(parent, child)
            except (AttributeError, TypeError):
                pass
        evicted.append(f"{name} ({path})")
    return tuple(sorted(evicted))


class _AuditedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile an untrusted module from its CURRENT source bytes.

    Bypasses bytecode caches entirely (a timestamp-valid stale .pyc can
    otherwise execute old code while the new source gets hashed — round-8
    P1) and records the sha256 of the exact bytes handed to compile(), so
    the receipt binds the code that actually executed.  Subclassing keeps
    exec_module -> get_code dispatch on THIS loader.
    """

    def __init__(self, fullname, path, record):
        super().__init__(fullname, path)
        self._record = record

    def get_code(self, fullname):
        data = self.get_data(self.path)
        self._record[str(Path(self.path).resolve())] = (
            hashlib.sha256(data).hexdigest())
        return self.source_to_code(data, self.path)


class _AuditedSourceFinder:
    """Meta-path finder routing untrusted .py imports through
    _AuditedSourceLoader; trusted-root and non-source imports defer to the
    normal machinery (non-source local dependencies are then refused by the
    dependency scan because their executed code has no recorded source)."""

    def __init__(self, roots, record):
        self._roots = roots
        self._record = record

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path,
                                                        target)
        if spec is None or not spec.origin:
            return None
        try:
            origin = Path(spec.origin).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if origin.suffix != ".py":
            return None
        if any(root == origin or root in origin.parents
               for root in self._roots):
            return None
        spec.loader = _AuditedSourceLoader(fullname, spec.origin,
                                           self._record)
        spec.cached = None
        return spec


def _scan_reference_dependencies(model_path: Path, sized_path: Path,
                                 before_modules: set,
                                 executed_record: dict):
    """Classify the modules present after reference preparation.

    Modules whose file lives in the adapter's directory tree (the supported
    local-helper contract) are hashed; modules NEWLY imported during
    preparation from anywhere else except the Python environment
    (sys.prefix/exec_prefix) and the solar checkout are reported as
    unsupported dependencies — their closure cannot be proven.  Cached
    adapter-dir modules are still caught because ALL of sys.modules is
    scanned against the adapter directory, not only the new imports.
    """
    adapter_dir = Path(model_path).resolve().parent
    own = {Path(model_path).resolve(), Path(sized_path).resolve()}
    safe_roots = _safe_roots()
    deps, unsupported = {}, []
    for name, mod in list(sys.modules.items()):
        fname = getattr(mod, "__file__", None)
        if not fname:
            continue
        try:
            path = Path(fname).resolve()
        except (OSError, RuntimeError):
            continue
        if path in own:
            continue
        if adapter_dir == path.parent or adapter_dir in path.parents:
            executed = executed_record.get(str(path))
            if executed is None:
                unsupported.append(
                    f"{name} ({path}): executed code cannot be tied to "
                    f"current source bytes (bytecode/non-source/manual "
                    f"load)")
            else:
                deps[str(path)] = executed
        elif name not in before_modules and not any(
                root == path or root in path.parents
                for root in safe_roots):
            unsupported.append(f"{name} ({path})")
    return tuple(sorted(deps.items())), tuple(sorted(unsupported))


# Exit status both verify CLIs (scripts/verify_saved_ir.py and
# ``solar-translate verify``) report.  0/1/2 are the legacy contract; 3 is
# returned only when the caller asked for strict numerical proof
# (``--strict``) or a receipt and the run cannot deliver it.
EXIT_VERIFIED = 0
EXIT_FAILED = 1          # MISMATCH or IR execution failure
EXIT_USAGE = 2           # argparse errors, missing input files
EXIT_NO_PROOF = 3        # strict: shape-only / vacuous pass / receipt refused


def check_input_files(model_path, ir_path) -> Optional[str]:
    """Shared missing-input check for both verify CLIs (audit finding A9).

    Returns the exact ``ERROR: ... file not found: <path>`` message the
    script has always printed (exit 2), or None when both files exist.
    """
    for path, what in ((Path(model_path).resolve(), "model"),
                       (Path(ir_path).resolve(), "IR")):
        if not path.is_file():
            return f"ERROR: {what} file not found: {path}"
    return None


_NUMPY_TO_SOLAR = {
    "float16": "fp16", "float32": "fp32", "float64": "fp64",
    "bfloat16": "bf16", "int8": "int8", "int16": "int16", "int32": "int32",
    "int64": "int64", "uint8": "uint8", "bool": "bool",
}


def _dtype_name(value) -> Optional[str]:
    """SOLAR dtype string for an array-like's dtype, or None if it has none."""
    dt = getattr(value, "dtype", None)
    if dt is None:
        return None
    name = getattr(dt, "name", None) or str(dt).replace("torch.", "")
    return _NUMPY_TO_SOLAR.get(name, name)


def _dtype_table(arrays) -> tuple:
    return tuple((str(k), _dtype_name(v)) for k, v in (arrays or {}).items()
                 if _dtype_name(v) is not None)


def _all_zero_outputs(reference_outputs) -> tuple:
    """Names of reference outputs that are all zeros / have no finite element
    (the same rule compare_outputs applies for its ALL ZEROS warning)."""
    import numpy as np
    names = []
    for name, arr in (reference_outputs or {}).items():
        a = np.asarray(arr)
        if a.size == 0:
            names.append(str(name))
            continue
        if np.issubdtype(a.dtype, np.floating):
            finite = a[np.isfinite(a)]
        else:
            finite = a
        if finite.size == 0 or not np.any(finite):
            names.append(str(name))
    return tuple(names)


@dataclass
class VerifyResult:
    """Outcome of one verification run (library form of the script's output).

    ``verified`` keeps its legacy meaning (numerical comparison passed).
    ``status`` names the outcome precisely — ``pass``, ``pass_vacuous``
    (passed, but every output in ``vacuous_outputs`` had an all-zero
    reference: weak evidence), ``shape_only`` (no comparison happened),
    ``fail`` (MISMATCH) or ``error`` (IR execution failed).  ``exit_code``
    is the legacy exit status (shape-only and vacuous passes exit 0);
    ``exit_status(strict=True)`` fails closed with EXIT_NO_PROOF for both.
    """

    verified: bool
    max_abs_error: Optional[float] = None
    message: str = ""
    reference_seconds: Optional[float] = None
    ir_seconds: Optional[float] = None
    shape_only: bool = False
    # present only on results produced by run_verification
    execution: Optional[ExecutionEvidence] = None
    # outputs whose reference was all zeros (comparison vacuous), see A4
    vacuous_outputs: list = field(default_factory=list)
    execution_failed: bool = False

    @property
    def vacuous(self) -> bool:
        return bool(self.vacuous_outputs)

    @property
    def status(self) -> str:
        if self.shape_only:
            return "shape_only"
        if self.verified:
            return "pass_vacuous" if self.vacuous else "pass"
        if self.execution_failed:
            return "error"
        return "fail"

    def exit_status(self, strict: bool = False) -> int:
        """Exit status for this outcome.

        Legacy (``strict=False``): 0 for a pass AND for shape-only (the
        documented historical contract), 1 otherwise.  ``strict=True`` is
        the request for numerical proof: shape-only runs and vacuous passes
        return EXIT_NO_PROOF (3) instead of 0.
        """
        if self.shape_only:
            return EXIT_NO_PROOF if strict else EXIT_VERIFIED
        if self.verified:
            return EXIT_NO_PROOF if (strict and self.vacuous) else EXIT_VERIFIED
        return EXIT_FAILED

    @property
    def exit_code(self) -> int:
        """The legacy exit code scripts/verify_saved_ir.py reports."""
        return self.exit_status(strict=False)


def _frontend(source_type: str):
    """The ``prepare_*`` reference builder for *source_type* (cuda by default)."""
    if source_type == "jax":
        from solar.einsum_lib.jax_frontend import prepare_jax as prepare
    elif source_type == "triton":
        from solar.einsum_lib.triton_frontend import prepare_triton as prepare
    elif source_type == "pytorch":
        from solar.einsum_lib.pipeline import prepare_pytorch as prepare
    else:
        from solar.einsum_lib.cuda_frontend import prepare_cuda as prepare
    return prepare


def run_verification(
    model_path: Path,
    ir_path: Path,
    *,
    source_type: str = "cuda",
    config_overrides: Optional[dict] = None,
    timeout: Optional[int] = None,
    mem_budget_gb: Optional[float] = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    fast: bool = True,
    shape_only: bool = False,
    use_fp64: bool = True,
    quiet: bool = False,
    log: Callable[[str], None] = print,
) -> VerifyResult:
    """Build the reference, execute the saved IR, and compare numerically.

    One typed outcome propagates instead of becoming an execution-failure
    result: ``IsolationUnavailableError`` (no isolating child could be started
    -- an environment fact; the caller classifies it).

    This is the body of ``scripts/verify_saved_ir.py`` with the argument
    parsing removed; ``log`` receives exactly the lines the script prints, so
    the script wrapper keeps byte-compatible output.
    """
    from solar.einsum_lib.verify import IsolationUnavailableError, compare_outputs, execute_translated

    prepare = _frontend(source_type)
    overrides = dict(config_overrides or {})

    with tempfile.TemporaryDirectory(prefix="verify_saved_ir_") as tmp, \
            sized_model(Path(model_path), overrides, Path(tmp)) as sized:
        if overrides:
            log(f"CONFIG overrides: {overrides}")

        # Hash the exact bytes this run imports/executes (round-4 P1: the
        # receipt evidence is bound here, inside the execution, so a result
        # can never be rebound to bytes it did not verify).
        source_sha256 = hashlib.sha256(sized.read_bytes()).hexdigest()

        log(f"Building reference from {sized.name} ...")
        t0 = time.perf_counter()
        # cap_input_size is intentionally False: capping would call the LLM, and
        # --config is the deterministic way to choose sizes here.
        evicted_cached_modules = _evict_unproven_cached_modules()
        _modules_before_prepare = set(sys.modules)
        _executed_record = {}
        _finder = _AuditedSourceFinder(_safe_roots(), _executed_record)
        sys.meta_path.insert(0, _finder)
        importlib.invalidate_caches()
        # The frontends load the adapter/model DIRECTLY (not via sys.meta_path),
        # through solar.einsum_lib.source_load — recording into the same dict
        # covers that path too (round-9 P1: a direct
        # spec_from_file_location load bypassed the finder, so the MAIN
        # adapter could execute a timestamp-valid stale .pyc).
        from solar.einsum_lib.source_load import record_executed_sources
        try:
            with record_executed_sources(_executed_record):
                prepared = prepare(str(sized), True, cap_input_size=False,
                                   use_fp64=use_fp64, debug=not quiet)
        finally:
            try:
                sys.meta_path.remove(_finder)
            except ValueError:
                pass
        ref_secs = time.perf_counter() - t0
        local_dependencies, unsupported_dependencies = (
            _scan_reference_dependencies(Path(model_path), sized,
                                         _modules_before_prepare,
                                         _executed_record))
        # The adapter/model itself must have been compiled from the exact
        # bytes hashed above; anything else (a load path that skipped the
        # audited loader, or a file changed mid-run) cannot be tied to the
        # evidence and refuses strict qualification.
        if _executed_record.get(str(sized.resolve())) != source_sha256:
            unsupported_dependencies += ((
                f"adapter ({sized.resolve()}): executed code cannot be "
                f"tied to the hashed source bytes (bytecode/non-source/"
                f"manual load)"),)

        # Frontends share a 9-tuple prefix.  The CUDA frontend appends a
        # numeric tolerance floor; prepare_pytorch appends the model object —
        # pick up only NUMERIC extras as the floor and ignore the rest.
        (reference_source, input_shapes, input_dtypes, reference_outputs,
         all_inputs, _sparsity, _mask_fill, _buffers,
         verify_shapes), extra = prepared[:9], prepared[9:]
        min_tol = next((float(e) for e in extra
                        if isinstance(e, (int, float)) and not isinstance(e, bool)), 0.0)

        log(f"Reference built in {ref_secs:.1f}s")
        log("Verify shapes:")
        log(format_shapes(verify_shapes))
        for name, arr in (reference_outputs or {}).items():
            log(f"  reference output {name!r}: {arr.shape} {arr.dtype}")

        dsl_bytes = Path(ir_path).read_bytes()
        code = dsl_bytes.decode("utf-8")

        from solar.einsum_lib import verify as _checker_mod
        vacuous_reference_outputs = _all_zero_outputs(reference_outputs)
        evidence = ExecutionEvidence(
            source_sha256=source_sha256,
            dsl_sha256=hashlib.sha256(dsl_bytes).hexdigest(),
            reference_sha256=hashlib.sha256(
                str(reference_source).encode("utf-8")).hexdigest(),
            source_type=source_type,
            rtol=rtol, atol=atol, min_tol=min_tol,
            use_fp64=use_fp64, fast=fast,
            verification_dtype="fp64" if use_fp64 else "fp32",
            config_overrides=tuple(sorted(overrides.items())),
            validation_shapes=tuple(
                (name, tuple(shape)) for name, shape in verify_shapes.items()),
            local_dependencies=local_dependencies,
            unsupported_dependencies=unsupported_dependencies,
            evicted_cached_modules=evicted_cached_modules,
            checker_identity="solar.einsum_lib.verify.compare_outputs",
            checker_sha256=hashlib.sha256(
                Path(_checker_mod.__file__).read_bytes()).hexdigest(),
            input_dtypes=tuple((str(k), str(v))
                               for k, v in (input_dtypes or {}).items()),
            reference_input_dtypes=_dtype_table(all_inputs),
            reference_output_dtypes=_dtype_table(reference_outputs),
            vacuous_reference_outputs=vacuous_reference_outputs,
            _token=_EVIDENCE_TOKEN)

        limit = "no limit" if timeout is None else f"{timeout}s"
        log(f"\nExecuting {Path(ir_path).name} (fast={fast}, timeout={limit}) ...")

        t0 = time.perf_counter()
        try:
            outputs, _ = execute_translated(
                code, verify_shapes,
                input_dtypes=input_dtypes,
                all_inputs=all_inputs,
                fast=fast,
                shape_only=shape_only,
                use_fp64=use_fp64,
                timeout=timeout,
                mem_budget_bytes=(int(mem_budget_gb * 1e9)
                                  if mem_budget_gb else None),
            )
        except IsolationUnavailableError:
            raise                       # typed, structured: the caller classifies it
        except Exception as exc:
            secs = time.perf_counter() - t0
            msg = (f"IR execution FAILED after {secs:.1f}s: "
                   f"{type(exc).__name__}: {exc}")
            log(msg)
            return VerifyResult(verified=False, message=msg,
                                reference_seconds=ref_secs, ir_seconds=secs,
                                execution=evidence, execution_failed=True)
        ir_secs = time.perf_counter() - t0
        log(f"IR executed in {ir_secs:.1f}s")

        for name, arr in (outputs or {}).items():
            log(f"  IR output {name!r}: {getattr(arr, 'shape', '?')}")

        if shape_only:
            log("\nSHAPE-ONLY: numeric comparison skipped.")
            return VerifyResult(verified=False, shape_only=True,
                                message="SHAPE-ONLY: numeric comparison skipped.",
                                reference_seconds=ref_secs, ir_seconds=ir_secs,
                                execution=evidence)

        result = compare_outputs(outputs, reference_outputs,
                                 rtol=rtol, atol=atol, min_tol=min_tol)
        log("")
        if result.success:
            msg = (f"VERIFIED  (max_abs_error={result.max_abs_error:.3e}, "
                   f"reference {ref_secs:.1f}s, IR {ir_secs:.1f}s)")
            vacuous = sorted(set(result.vacuous_outputs)
                             | set(vacuous_reference_outputs))
            if vacuous:
                # The verdict line stays parseable; the qualifier is appended
                # so a log reader sees the weak evidence next to the pass.
                msg += (f"  [VACUOUS: all-zero reference output(s) "
                        f"{', '.join(vacuous)} — weak evidence]")
            log(msg)
            return VerifyResult(verified=True, max_abs_error=result.max_abs_error,
                                message=msg, reference_seconds=ref_secs,
                                ir_seconds=ir_secs, execution=evidence,
                                vacuous_outputs=list(vacuous))
        msg = f"MISMATCH  {result.error}"
        log(msg)
        if result.max_abs_error is not None:
            log(f"  max_abs_error={result.max_abs_error:.6e} "
                f"(rtol={max(rtol, min_tol)}, atol={max(atol, min_tol)})")
        return VerifyResult(verified=False, max_abs_error=result.max_abs_error,
                            message=msg, reference_seconds=ref_secs,
                            ir_seconds=ir_secs, execution=evidence)
