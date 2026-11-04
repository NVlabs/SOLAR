# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-mode translation facade: one artifact contract, two translation modes.

Mode ``agent`` (the DEFAULT) scaffolds a workspace and hands the translation
to a coding agent following ``skills/verified-translation/SKILL.md`` — no LLM
API is ever called and no key is ever looked for.  Mode ``api`` wraps the
LLM pipeline (``solar.cli.toDSL_model``) and runs ONLY when explicitly
selected AND given a key.  Both modes produce the same artifacts
(adapter/DSL -> verify -> dsl_full/einsum_full/analysis_full/perf_full), so
``verify`` and ``sol`` below are mode-agnostic.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ._runner import (EXIT_NO_PROOF, EXIT_USAGE, VerifyResult,
                      check_input_files, run_verification)
from .receipts import workload_identity as _workload_identity

__all__ = [
    "ScaffoldResult", "SolResult", "TranslateResult", "VerifyResult",
    "finalize_verify", "scaffold", "sol", "translate", "verify",
]

_SOLAR_ROOT = Path(__file__).resolve().parents[2]
_SKILL_MD = "skills/verified-translation/SKILL.md"

API_POLICY = (
    "The LLM API pipeline runs only when BOTH hold: mode='api' was explicitly "
    "selected AND an API key was provided (api_key argument or OPENAI_API_KEY "
    "env). The default is mode='agent': the coding agent authors the DSL "
    f"following {_SKILL_MD}; never fall back to the API silently, never hunt "
    "for keys."
)

_ADAPTER_STUB = '''\
"""CUDA-contract adapter for {problem} (manual path) — STUB, fill the TODOs.

Contract (skills/verified-translation/SKILL.md step 2):
  get_source()      -> the reference source VERBATIM (env shims go in the exec
                       namespace, never edit the source string)
  get_inputs()      -> dict of capped-shape torch tensors. Default var axes to
                       the FIRST workload.jsonl line; respect every assert
                       (divisibility, expert>=2*topk, exact factoring). Inputs
                       must be non-degenerate: an all-zero reference output
                       makes verification vacuous (the verifier warns).
  kernel_fn(inputs) -> exec the source, call the entry point, return
                       {{name: np.ndarray}} keyed like the definition outputs.
                       Dtype-agnostic: verify feeds fp64 (no dtype= kwargs;
                       clone golden-side if the reference mutates in place).
"""
from pathlib import Path

import numpy as np
import torch

_REFERENCE = Path({reference!r})


def get_source() -> str:
    return _REFERENCE.read_text()


def get_inputs() -> dict:
    # FILL IN: capped, non-degenerate inputs keyed by the entry point's tensor
    # parameter names (workload line 0 values for var axes unless too large).
    raise NotImplementedError("fill get_inputs() per SKILL.md step 2")


def kernel_fn(inputs: dict) -> dict:
    # FILL IN: exec(get_source(), ns) with any shims, call the entry point,
    # return outputs as numpy arrays (keys must match the DSL's run_model).
    raise NotImplementedError("fill kernel_fn() per SKILL.md step 2")
'''

_NOTES_TEMPLATE = """\
kernel      : {problem}
entry point : <fill: function + exact math, from the reference CODE, not names>
serving     : <fill: original axes from definition.json / workload.jsonl line 0>
capped      : <fill: verification shapes + why each cap is safe>
self-check  : <fill: max_abs_err + expected error class>
verify cmd  : <fill: exact command + tolerance rationale>
quirks      : <fill: reference surprises: names that lie, in-place mutation, ...>
"""

_TRITON_README = """\
Triton source: NO adapter is needed — source_type='triton' takes a
SELF-REFERENCING module as the model argument. The module must define:
  get_inputs() -> {name: np.ndarray}         (names become einsum_lib inputs)
  kernel_fn(inputs) -> {"output": np.ndarray} (golden: runs the real kernel)
  get_source() -> str                         (optional; defaults to the file)

1. If the kernel's default sizes are too large to verify, COPY the module to
   model_capped.py in this directory and edit its size globals only.
2. Write solar_model_einsum_lib.py against the kernel's math (tl.load masks
   with neutral `other=` values are usually value-neutral — follow the code).
3. Verify:  scripts/verify_saved_ir.py <model_capped.py> <dsl.py> \\
                --source-type triton
Contracts and idioms: skills/verified-translation/reference.md.
"""

_PYTORCH_README = """\
PyTorch source (kernelbench-style): NO adapter is needed — the model .py
(defines Model + get_inputs) IS the golden reference.

1. If the model is too large to verify, COPY it to model_capped.py in this
   directory and edit its globals / get_inputs sizes (kernelbench files have
   no CONFIG dict, so --config does not apply). Watch architecture-coupled
   sizes (fc layers pin spatial dims; adaptive pools decouple them).
2. Write solar_model_einsum_lib.py against Model.forward's math.
3. Verify:  scripts/verify_saved_ir.py <model_capped.py> <dsl.py> \\
                --source-type pytorch [--mem-budget 30]
Contracts (input naming x/x_0, dotted->underscore params, BN buffers,
'output' key): skills/verified-translation/reference.md.
"""

# source_type -> (README file name, contents) for sources that need no adapter.
_NO_ADAPTER_READMES = {
    "pytorch": ("README_pytorch.txt", _PYTORCH_README),
    "triton": ("README_triton.txt", _TRITON_README),
}

@dataclass
class ScaffoldResult:
    out_dir: Path
    files: list = field(default_factory=list)
    checklist: str = ""


@dataclass
class SolResult:
    """Original-shape SOL numbers plus their VALIDATION STATUS (A11).

    ``validated`` is True only when ``sol(strict=True)`` admitted a
    hash-matched numerical receipt for exactly these inputs; ``receipt``
    is that receipt's path.  ``validation_status`` is ``validated`` or
    ``unvalidated_legacy``; the same record is written next to the perf
    YAML as ``validation.json`` (``validation_path``) so the artifacts
    carry the label downstream.  The numbers themselves are identical on
    both paths — only the evidence differs.
    """

    fused_ms: Optional[float] = None
    unfused_ms: Optional[float] = None
    macs: Optional[int] = None
    macs_typed: Optional[dict] = None          # issue #58: the aggregate's typed components (perf workload.macs_typed)
    perf_path: Optional[Path] = None
    analysis_path: Optional[Path] = None
    einsum_path: Optional[Path] = None
    dsl_dir: Optional[Path] = None
    log: str = ""
    validated: bool = False
    validation_status: str = "unvalidated_legacy"
    receipt: Optional[Path] = None
    validation_path: Optional[Path] = None

    @property
    def validation_line(self) -> str:
        """The one-line label ``solar-translate sol`` prints."""
        if self.validated:
            return f"validation: receipt {self.receipt}"
        return "validation: none (legacy, unvalidated)"


@dataclass
class TranslateResult:
    mode: str
    out_dir: Path
    checklist: str = ""          # agent mode
    verified: Optional[bool] = None  # api mode
    log: str = ""
    # api mode: the child's exit status and the typed verdict derivation
    # (``verified`` / ``child_failed`` / ``status_disagreement``), see
    # ``translate`` (A2)
    exit_code: Optional[int] = None
    verification_status: str = ""


def _checklist(problem_dir: Path, out_dir: Path, source_type: str) -> str:
    """The agent-mode step list for *source_type*, with workspace paths filled in."""
    if source_type == "triton":
        return f"""\
Agent-mode translation of {problem_dir.name} (triton source) — follow {_SKILL_MD}:
  1. Read the kernel module; see {out_dir}/README_triton.txt for the
     self-referencing contract (get_inputs/kernel_fn in the module, NO adapter;
     model_capped.py copy for capping, verify with --source-type triton).
  2-7. As in SKILL.md steps 3-7 (skip the adapter)."""
    if source_type == "pytorch":
        return f"""\
Agent-mode translation of {problem_dir.name} (pytorch source) — follow {_SKILL_MD}:
  1. Read the model file; see {out_dir}/README_pytorch.txt for the no-adapter
     contract (model_capped.py copy, naming, verify command).
  2-7. As in SKILL.md steps 3-7 (skip the adapter)."""
    return f"""\
Agent-mode translation of {problem_dir.name} — follow {_SKILL_MD}:
  1. Read the reference end to end (names and docstrings lie) + definition
     axes + workload.jsonl line 0.
  2. Fill {out_dir}/adapter.py (get_source/get_inputs/kernel_fn contract).
  3. Write {out_dir}/solar_model_einsum_lib.py per reference.md (language
     cheat-sheet, authoring rules for numerics AND op counting; start from
     skills/verified-translation/example/).
  4. Self-check: skills/verified-translation/example/selfcheck.py {out_dir}
  5. Official verify (ONE at a time):
       scripts/verify_saved_ir.py {out_dir}/adapter.py \\
           {out_dir}/solar_model_einsum_lib.py --source-type {source_type} \\
           --mem-budget 30 [tolerance flags per reference.md]
  6. Einsum graph + SOL at original shapes: solar.translate.sol(...) or the
     trace_saved_ir -> cli_einsum -> analyze -> perf chain (SKILL.md step 6).
  7. Record {out_dir}/NOTES (entry point, caps, self-check error, verify
     command + tolerance rationale, quirks)."""


def scaffold(problem_dir, out_dir, source_type: str = "cuda") -> ScaffoldResult:
    """Create an agent-mode workspace: stubs + NOTES + checklist (no LLM).

    PyTorch and Triton sources get a README instead of an adapter stub: the
    model file (pytorch) or a self-referencing module (triton) is itself the
    golden reference, so a cuda-style adapter stub would mislead.
    """
    problem_dir = Path(problem_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    if source_type in _NO_ADAPTER_READMES:
        readme_name, readme_text = _NO_ADAPTER_READMES[source_type]
        readme = out_dir / readme_name
        readme.write_text(readme_text)
        files.append(readme)
    else:
        adapter = out_dir / "adapter.py"
        if not adapter.exists():  # never clobber in-progress work
            adapter.write_text(_ADAPTER_STUB.format(
                problem=problem_dir.name,
                reference=str(problem_dir / "reference.py")))
            files.append(adapter)

    notes = out_dir / "NOTES"
    if not notes.exists():
        notes.write_text(_NOTES_TEMPLATE.format(problem=problem_dir.name))
        files.append(notes)

    checklist = _checklist(problem_dir, out_dir, source_type)
    return ScaffoldResult(out_dir=out_dir, files=files, checklist=checklist)


def verify(model_path, ir_path, source_type: str = "cuda",
           mem_budget_gb: Optional[float] = None, rtol: Optional[float] = None,
           atol: Optional[float] = None, timeout: Optional[int] = None,
           **kw) -> VerifyResult:
    """Library form of scripts/verify_saved_ir.py (same core, no subprocess).

    The returned ``VerifyResult`` carries the typed outcome (``status``,
    ``vacuous_outputs``) and sealed ``execution`` evidence; pass it to
    ``solar.translate.receipts.emit_workload_receipt`` for a receipt.
    """
    return run_verification(
        Path(model_path), Path(ir_path), source_type=source_type,
        mem_budget_gb=mem_budget_gb,
        rtol=1e-4 if rtol is None else rtol,
        atol=1e-5 if atol is None else atol,
        timeout=timeout, **kw)


def finalize_verify(result: VerifyResult, *, model_path, ir_path,
                    strict: bool = False, emit_receipt=None,
                    definition_dir=None, dtype: Optional[str] = None,
                    workload_idx: int = 0, log=print) -> int:
    """Shared tail of both verify CLIs: optional receipt, then exit status.

    * ``emit_receipt`` (a path) writes a receipt through
      ``receipts.emit_workload_receipt`` — the same artifact on the script
      and the facade CLI (A5).  It requires ``definition_dir`` and
      ``dtype``.  A failed run still gets its (``fail``) receipt; a receipt
      the evidence cannot support prints ``RECEIPT REFUSED: ...`` and
      returns EXIT_NO_PROOF (3), whatever the verdict was.
    * ``strict`` asks for numerical proof: shape-only runs and vacuous
      passes return EXIT_NO_PROOF instead of the legacy 0 (A3/A4).
    """
    rc = result.exit_status(strict=strict)
    if emit_receipt is not None:
        if definition_dir is None or dtype is None:
            raise ValueError("--emit-receipt requires --definition and --dtype")
        from solar.translate.receipts import (ReceiptRefusal,
                                              emit_workload_receipt)
        try:
            out = emit_workload_receipt(
                result, source_path=model_path, dsl_path=ir_path,
                definition_dir=definition_dir, dtype=dtype,
                workload_idx=workload_idx, out_path=emit_receipt)
        except ReceiptRefusal as e:
            log(f"RECEIPT REFUSED: {e}")
            return EXIT_NO_PROOF
        verdict = ("error" if result.shape_only
                   else "pass" if result.verified else "fail")
        qualifier = " (vacuous)" if result.vacuous else ""
        log(f"receipt: {out}  (numerical: {verdict}{qualifier}, "
            f"status: {result.status})")
    return rc


def sol(ir_path, definition_dir, out_dir, arch: str = "B200",
        precision: str = "fp16", fallback_adapter=None, custom_inputs=None,
        measure_density: bool = False, density_overrides: Optional[dict] = None,
        workload_idx: int = 0,
        perf_precision: Optional[str] = None,
        receipt=None, strict: bool = False,
        source_path=None,
        axes: Optional[dict] = None,
        mem_budget_gb: Optional[float] = None,
        measure_timeout: Optional[int] = None,
        measure_attention_density: bool = False,
        allow_unbound_constants: bool = False) -> SolResult:
    """Original-shape trace -> einsum graph -> analysis -> SOL perf.

    The trace step shells out to scripts/trace_saved_ir.py (it owns the
    density trust ladder); the einsum/analysis/perf stages are library calls.
    ``axes`` (``--axis NAME=VALUE`` overrides), ``mem_budget_gb``,
    ``measure_timeout``, ``measure_attention_density`` and
    ``allow_unbound_constants`` are forwarded to the trace script verbatim
    (A7); when omitted the script's own defaults apply, as before.

    strict=True enforces the receipt gate BEFORE any tracing/analysis: the
    ``receipt`` (written by solar.translate.receipts.emit_receipt from an
    actual verify() run) must carry a numerical PASS and hash-match the
    CURRENT ``source_path``/``ir_path``/definition.json bytes and the exact
    selected workload; otherwise ReceiptRefusal is raised and nothing runs.
    The default (strict=False) is the documented LEGACY path: it performs no
    numerical validation and its outputs must not be presented as validated.
    Either way the result and ``<out_dir>/validation.json`` state which
    path produced the numbers (A11).
    """
    import yaml

    receipt_sha256 = None
    if strict:
        from solar.translate.receipts import ReceiptRefusal, check_strict
        if receipt is None:
            raise ReceiptRefusal("strict SOL requires a receipt path")
        if source_path is None:
            raise ReceiptRefusal(
                "strict SOL requires source_path (the source the receipt "
                "validated)")
        wl_uuid, wl_axes = _workload_identity(definition_dir, workload_idx)
        # dtype is always compared against the SELECTED analysis precision —
        # a receipt for one precision cannot admit a run billed at another.
        check_strict(receipt, source_path=source_path, dsl_path=ir_path,
                     definition_path=Path(definition_dir) / "definition.json",
                     workload_uuid=wl_uuid, axes=wl_axes, dtype=precision)
        import hashlib
        receipt = Path(receipt).resolve()
        receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()

    from solar.analysis import EinsumGraphAnalyzer
    from solar.common.utils import ensure_directory
    from solar.einsum_dsl.cli_einsum import translate_with_einsum
    from solar.perf import EinsumGraphPerfModel

    out_dir = Path(out_dir).resolve()
    dsl_dir = out_dir / "dsl_full"
    einsum_dir = out_dir / "einsum_full"
    analysis_dir = out_dir / "analysis_full"
    perf_dir = out_dir / "perf_full"

    from solar.resources import resource_path
    cmd = [sys.executable, str(resource_path("scripts", "trace_saved_ir.py")),
           str(ir_path), "--definition", str(definition_dir),
           "--out-dir", str(dsl_dir), "--workload-idx", str(workload_idx)]
    if fallback_adapter:
        cmd += ["--fallback-adapter", str(fallback_adapter)]
    if custom_inputs:
        cmd += ["--custom-py", str(custom_inputs)]
    if measure_density:
        cmd += ["--measure-density"]
    for name, val in (density_overrides or {}).items():
        cmd += ["--density-override", f"{name}={val}"]
    for name, val in (axes or {}).items():
        cmd += ["--axis", f"{name}={val}"]
    if mem_budget_gb is not None:
        cmd += ["--mem-budget", str(mem_budget_gb)]
    if measure_timeout is not None:
        cmd += ["--measure-timeout", str(measure_timeout)]
    if measure_attention_density:
        cmd += ["--measure-attention-density"]
    if allow_unbound_constants:
        cmd += ["--allow-unbound-constants"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_SOLAR_ROOT)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"trace failed (rc={proc.returncode}): {shlex.join(cmd)}\n{log[-2000:]}")

    einsum_path = Path(translate_with_einsum(dsl_dir, einsum_dir))

    # Boundary STORAGE dtypes are the benchmark definition's declared
    # input/output dtypes — authoritative for boundary traffic billing,
    # independent of the analysis/compute precision (campaign-audit
    # defects 2/3).  DSL-internal boundary tensors fall back to recorded
    # trace dtypes inside the analyzer; anything unresolvable fails loudly.
    defn = json.loads(
        (Path(definition_dir) / "definition.json").read_text())
    boundary_dtypes = {}
    for group in ("inputs", "outputs"):
        for name, spec in (defn.get(group) or {}).items():
            if isinstance(spec, dict) and spec.get("dtype"):
                boundary_dtypes[str(name)] = str(spec["dtype"])
    # Graph output tensors carry DSL kernel names; pass the declared
    # outputs with their RESOLVED element counts so the analyzer can bind
    # them by size (refusing ambiguous widths, never guessing).
    _, _wl_axes = _workload_identity(definition_dir, workload_idx)
    _axes = {k: v.get("value") for k, v in (defn.get("axes") or {}).items()
             if isinstance(v, dict) and v.get("type") == "const"}
    _axes.update(_wl_axes or {})
    # Declared shapes use the benchmark definition grammar: ints, axis
    # names, and bounded arithmetic expressions (e.g. "num_heads *
    # head_dim").  Resolve them with the established bounded resolver —
    # unknown axis names and disallowed constructs keep loud ValueErrors
    # (campaign review C12-P2: int() rejected valid expressions).
    from solar.translate._dimspec import resolve_dimension
    declared_outputs = []
    for name, spec in (defn.get("outputs") or {}).items():
        if not (isinstance(spec, dict) and spec.get("dtype")):
            continue
        elems = 1
        for d in spec.get("shape") or []:
            elems *= resolve_dimension(d, _axes)
        declared_outputs.append({"name": str(name),
                                 "dtype": str(spec["dtype"]),
                                 "elements": int(elems)})

    analysis = EinsumGraphAnalyzer(debug=False).analyze_graph(
        einsum_path, ensure_directory(analysis_dir),
        precision=precision, copy_graph=False,
        boundary_storage_dtypes=boundary_dtypes,
        declared_output_storage=declared_outputs)
    if analysis is None:
        raise RuntimeError("analysis failed")
    analysis_path = analysis_dir / "analysis.yaml"

    # perf_precision lets the perf stage bill at a different precision than
    # the analysis stage (campaign convention: Quant analyzes fp16, bills
    # fp8/nvfp4 — see scripts/regen_all_sol.sh).
    EinsumGraphPerfModel(debug=False).predict(
        analysis_path, ensure_directory(perf_dir),
        arch_config=arch, precision=perf_precision or precision,
        copy_analysis=False)
    perf_path = perf_dir / f"perf_{arch}.yaml"

    perf = yaml.safe_load(perf_path.read_text())
    if (perf.get("workload") or {}).get("unresolved_ops"):
        # Unreachable through the analyzer/perf refusals above; kept as the
        # facade's own admission guard so no validation label (strict OR
        # legacy) is ever attached to incomplete counts (review I2).
        raise ValueError(
            "Refusing SOL: perf workload carries unresolved loop counts "
            f"{perf['workload']['unresolved_ops']}; incomplete counts are "
            "not admitted on either validation path.")
    # Validation label next to the artifacts (A11).  The perf YAML itself is
    # the perf stage's artifact and is left byte-for-byte as written.
    validation = {
        "status": "validated" if strict else "unvalidated_legacy",
        "strict": bool(strict),
        "receipt": str(receipt) if strict else None,
        "receipt_sha256": receipt_sha256,
        "perf_path": str(perf_path),
        "precision": precision,
        "perf_precision": perf_precision or precision,
        "workload_idx": workload_idx,
        "note": ("numbers admitted by a hash-matched numerical receipt for "
                 "exactly these inputs" if strict else
                 "no numerical validation was performed on this path; do "
                 "not present these numbers as validated"),
    }
    validation_path = out_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=1) + "\n")
    return SolResult(
        fused_ms=perf["fused"]["runtime_ms"],
        unfused_ms=perf["unfused"]["runtime_ms"],
        macs=perf["workload"]["total_macs"],
        macs_typed=perf["workload"].get("macs_typed"),
        perf_path=perf_path, analysis_path=analysis_path,
        einsum_path=einsum_path, dsl_dir=dsl_dir, log=log,
        validated=bool(strict), validation_status=validation["status"],
        receipt=receipt if strict else None,
        validation_path=validation_path)


def translate(source_path, out_dir, mode: str = "agent",
              source_type: str = "cuda", api_key: Optional[str] = None,
              **kw) -> TranslateResult:
    """Translate a source into the einsum_lib DSL.

    mode='agent' (DEFAULT): scaffold the workspace and return the skill
    checklist — no LLM call, no key lookup, ever.
    mode='api': explicit opt-in; requires a key (api_key or OPENAI_API_KEY).
    """
    source_path = Path(source_path)
    out_dir = Path(out_dir)

    if mode == "agent":
        problem_dir = source_path if source_path.is_dir() else source_path.parent
        res = scaffold(problem_dir, out_dir, source_type=source_type)
        return TranslateResult(mode="agent", out_dir=res.out_dir,
                               checklist=res.checklist)

    if mode != "api":
        raise ValueError(f"unknown mode {mode!r}: use 'agent' (default) or 'api'")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"mode='api' without an API key. {API_POLICY}")

    if not source_path.is_file():
        raise ValueError("mode='api' needs a source FILE (toDSL_model input)")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "solar.cli.toDSL_model", str(source_path),
           "--output-dir", str(out_dir), "--source-type", source_type]
    for flag, val in kw.items():
        opt = "--" + flag.replace("_", "-")
        if val is True:
            cmd.append(opt)
        elif val not in (None, False):
            cmd += [opt, str(val)]
    env = dict(os.environ, OPENAI_API_KEY=key)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=_SOLAR_ROOT, env=env)
    log = proc.stdout + proc.stderr
    # The verdict is the child's EXIT STATUS: toDSL_model exits 0 if and
    # only if its TranslationResult was verified (A2).  Marker text is never
    # trusted on its own — it can appear inside echoed code — and an exact
    # UNVERIFIED marker line that disagrees with exit 0 fails closed.
    lines = {ln.strip() for ln in log.splitlines()}
    if proc.returncode != 0:
        verified, status = False, "child_failed"
    elif "GENERATED CODE (UNVERIFIED):" in lines:
        verified, status = False, "status_disagreement"
    else:
        verified, status = True, "verified"
    return TranslateResult(mode="api", out_dir=out_dir, verified=verified,
                           log=log, exit_code=proc.returncode,
                           verification_status=status)
