# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One source-to-SOL API: mechanical first, then execution-validated IR.

Mode names describe frontends, not different definitions of speed of light.
The historical ``solar.translate`` scaffold API remains compatible.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4


class UnsupportedError(RuntimeError):
    """The mechanical frontend cannot faithfully lower this input."""


class ValidationError(RuntimeError):
    """An IR failed execution or numerical comparison; no SOL was emitted."""


class PipelineError(RuntimeError):
    """Infrastructure or analysis failure, never an implicit fallback trigger."""


class AgentTranslationRequired(UnsupportedError):
    """A coding agent must author IR and resubmit it with ``ir_path``."""


@dataclass
class EstimateResult:
    requested_mode: str
    used_mode: str
    fused_ms: float
    unfused_ms: float
    macs: float
    output_dir: str
    source_sha256: str
    arch_sha256: str
    precision: str
    validation: dict
    fallback_reason: Optional[str] = None
    work_contract: str = "as-written"
    macs_typed: Optional[dict] = None          # issue #58: the aggregate's typed components (perf workload.macs_typed)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arch_path(arch):
    p = Path(arch)
    if not p.is_file():
        from solar.resources import find_resource
        p = find_resource("configs", "arch", str(arch) + ".yaml") or p
    if not p.is_file():
        raise FileNotFoundError(f"Unknown hardware configuration: {arch}")
    return p.resolve()


def _mechanical(source, out, source_type):
    # Never enable the legacy converter's unverified LLM handler generator.
    if source_type != "pytorch":
        raise UnsupportedError(f"v0 supports PyTorch Model/get_inputs files, not {source_type}")
    tree = ast.parse(source.read_text())
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    if not ({"Model", "ReferenceModel"} & names) or "get_inputs" not in names:
        raise UnsupportedError("v0 needs Model or ReferenceModel and get_inputs()")
    from solar.common.types import ProcessingConfig
    from solar.graph import PyTorchProcessor
    from solar.einsum import PyTorchToEinsum

    log = io.StringIO()
    graph_dir = out / "graph"
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        ok = PyTorchProcessor(ProcessingConfig(force_rerun=True, debug=True)).process_model_file(
            str(source), str(graph_dir))
    (out / "mechanical.log").write_text(log.getvalue())
    if not ok:
        message = log.getvalue()
        # Do not turn OOMs, missing packages, invalid environments or arbitrary
        # Python exceptions into unsupported-language fallback.
        # "UnsupportedFrontendOperand" is the recovery shim's typed
        # fail-closed diagnostic (accounting-incomplete operand capture) —
        # a genuine unsupported-source signal, so it must route agentic
        # instead of surfacing as a pipeline error.
        unsupported = ("not implemented for", "data-dependent", "data dependent",
                       "unsupported operation", "Cannot copy out of meta tensor",
                       "Could not get source code", "No Model or ReferenceModel",
                       "UnsupportedFrontendOperand")
        if any(s.lower() in message.lower() for s in unsupported):
            raise UnsupportedError("v0 tracing unsupported; see mechanical.log: " + message[-1200:])
        raise PipelineError("v0 tracing failed; see mechanical.log: " + message[-1200:])
    graph = PyTorchToEinsum(enable_agent=False).convert(graph_dir / "pytorch_graph.yaml", out / "einsum")
    problem = mechanical_graph_unsupported(graph)
    if problem:
        raise UnsupportedError(problem)
    return out / "einsum" / "einsum_graph.yaml", {"kind": "mechanical", "numeric_comparison": False}


_BOUNDARY_LAYER_TYPES = frozenset({"start", "end", "input", "output"})


def mechanical_graph_unsupported(graph) -> Optional[str]:
    """Why a converted einsum graph cannot stand as the mechanical (v0) result, or None.

    Boundary layers (``start`` / ``end`` / ``input`` / ``output``) carry no
    operation and are ignored for the supportability scan, but a graph that
    consists ONLY of them is not a successful translation: an operation node
    that vanished before conversion (tensor-node-2) would otherwise
    pass silently.  Any remaining layer that is marked ``is_einsum_supportable:
    false`` (e.g. ``unrecognised_op``, a refused scaled_mm) or has no equation
    is listed by name.
    """
    if not graph or not graph.get("layers"):
        return "v0 produced no operation graph"
    layers = graph["layers"]
    operation_layers = {k: v for k, v in layers.items() if v.get("type") not in _BOUNDARY_LAYER_TYPES}
    if not operation_layers:
        return ("v0 produced no operation layers (only " + ", ".join(sorted({str(v.get("type")) for v in layers.values()}))
                + "): the recorded operations did not reach the converter")
    unsupported = [f"{k}: {v.get('type')}" + (f" (unrecognised_op)" if v.get("unrecognised_op") else "")
                   for k, v in operation_layers.items()
                   if v.get("is_einsum_supportable") is False or not v.get("einsum_equation")]
    if unsupported:
        return "v0 cannot lower: " + ", ".join(unsupported)
    return None


def _translate_api(source, out, source_type, options):
    from solar.einsum_lib.pipeline import translate_file
    result = translate_file(str(source), source_type=source_type, output_dir=str(out),
                            verify=True, cap_input_size=False, use_fp64=False,
                            save_on_fail=False, **options)
    if not result.verified or not result.code:
        raise ValidationError(result.verification_error or "LLM translation did not verify")
    return out / f"{source.stem}_einsum_lib.py"


def _validated_ir_entry(source, source_type, code, ir, rtol, atol):
    """Isolated-child entry (module-level, picklable for ``verify._run_isolated``):
    reference capture, IR execution, comparison and metadata tracing under
    one process guard."""
    from solar.translate._runner import _frontend
    from solar.einsum_lib.verify import execute_translated, compare_outputs
    from solar.einsum_lib.core import get_trace
    from solar.einsum_lib.metadata import build_metadata

    prepared = _frontend(source_type)(str(source), True, cap_input_size=False,
                                      use_fp64=False, debug=False)
    (_, shapes, dtypes, reference, inputs, _, _, _, verify_shapes) = prepared[:9]
    if shapes != verify_shapes:
        raise ValidationError("Capped validation is not accepted by the unified API")
    floor = next((float(e) for e in prepared[9:]
                  if isinstance(e, (float, int)) and not isinstance(e, bool)), 0.0)
    if not reference:
        raise ValidationError("No reference outputs to validate")
    # The enclosing process guard covers reference capture, IR execution,
    # and metadata tracing. A second guard here would try to start a child
    # from a daemonic multiprocessing child, which Python does not permit.
    actual, _ = execute_translated(code, shapes, all_inputs=inputs, input_dtypes=dtypes,
                                   fast=True, use_fp64=False)
    check = compare_outputs(actual, reference, rtol=rtol, atol=atol, min_tol=floor)
    if not check.success:
        raise ValidationError(check.error or "Numerical mismatch")
    # The shape-only pass derives metadata; it cannot authorize a result.
    execute_translated(code, shapes, all_inputs=inputs, input_dtypes=dtypes,
                       trace=True, shape_only=True, fast=False)
    return build_metadata(get_trace()), {
        "kind": "output-comparison", "numeric_comparison": True,
        "max_abs_error": check.max_abs_error, "rtol": max(rtol, floor),
        "atol": max(atol, floor), "shapes": shapes, "dtypes": dtypes,
        "ir_sha256": _sha(ir), "scope": "supplied get_inputs case; not all inputs",
        "work_preservation": "requires structural audit; output equality alone is insufficient",
    }


def _validated_ir(source, ir, out, source_type, timeout, mem_budget_gb, rtol, atol):
    """Identical validation and deterministic lifting for API and skill IR."""
    from solar.translate._runner import _frontend
    from solar.einsum_lib.verify import _run_isolated
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_dsl.cli_einsum import translate_with_einsum

    budget = int(mem_budget_gb * 1e9)
    code = ir.read_text()
    metadata, validation = _run_isolated(_validated_ir_entry, timeout, budget,
                                         args=(source, source_type, code, ir, rtol, atol))
    dsl = out / "dsl"
    dsl.mkdir()
    (dsl / "model_einsum_lib.py").write_text(code)
    write_metadata_yaml(metadata, str(dsl / "model_einsum_lib.yaml"))
    return Path(translate_with_einsum(dsl, out / "einsum")), validation


def _analyze(graph, out, arch, precision):
    from solar.analysis import EinsumGraphAnalyzer
    from solar.perf import EinsumGraphPerfModel
    analysis_dir = out / "analysis"
    analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(graph, analysis_dir, precision=precision,
                                                   copy_graph=False)
    if analysis is None:
        raise PipelineError("Analysis did not produce a result")
    perf = EinsumGraphPerfModel(dtype_bytes=True).predict(
        analysis_dir / "analysis.yaml", out / "perf", arch_config=str(arch),
        precision=precision, copy_analysis=False)
    if not perf:
        raise PipelineError("Performance analysis did not produce a result")
    values = (perf["fused"]["runtime_ms"], perf["unfused"]["runtime_ms"],
              perf["workload"]["total_macs"])
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
        raise PipelineError("Non-finite or negative SOL result")
    return values + (perf["workload"].get("macs_typed"),)      # issue #58: typed components ride along (report-only)


def estimate(source_path, out_dir, *, mode="auto", source_type="pytorch", arch="B200",
             precision="fp16", ir_path=None, api_key_file=None, llm_model=None,
             max_retries=3, timeout=300, mem_budget_gb=16, rtol=1e-4, atol=1e-5):
    """Estimate as-written SOL, with explicit provenance and failure semantics.

    ``auto`` tries v0, falling back ONLY on UnsupportedError to v1.
    ``v0`` never calls an LLM. ``v1`` translates through the Python LLM API.
    ``v2`` consumes coding-agent IR, or raises AgentTranslationRequired with
    a task packet path. Both verified modes use the same numerical gate.

    Execute only trusted source files: references and generated IR are Python.
    Each call writes a unique run directory; failed calls cannot reuse stale SOL.
    """
    if mode not in {"auto", "v0", "v1", "v2"}:
        raise ValueError("mode must be auto, v0, v1 or v2")
    if source_type not in {"pytorch", "cuda", "jax", "triton"}:
        raise ValueError("source_type must be pytorch, cuda, jax or triton")
    if timeout <= 0 or mem_budget_gb <= 0 or max_retries < 1:
        raise ValueError("timeout, memory budget and max_retries must be positive")
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    source = Path(source_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("source_path must be a Python model or reference adapter file")
    if ir_path is not None and mode != "v2":
        raise ValueError("ir_path is only accepted with mode='v2'")
    arch = _arch_path(arch)
    out = Path(out_dir).resolve() / ("run-" + uuid4().hex[:12])
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps({
        "source": str(source), "source_sha256": _sha(source), "mode": mode,
        "source_type": source_type, "arch": str(arch), "arch_sha256": _sha(arch),
        "precision": precision, "work_contract": "as-written",
    }, indent=2))
    fallback = None
    used = mode
    try:
        if mode in {"auto", "v0"}:
            try:
                graph, validation = _mechanical(source, out, source_type)
                used = "v0"
            except UnsupportedError as exc:
                if mode == "v0":
                    raise
                fallback = str(exc)
                used = "v1"
        if used == "v1":
            translation = out / "translation"
            ir = _translate_api(source, translation, source_type, {
                "api_key_file": api_key_file, "llm_model": llm_model,
                "max_retries": max_retries, "rate_limit_retries": 2,
                "rate_limit_delay": 15, "verify_timeout": timeout,
                "mem_budget_bytes": int(mem_budget_gb * 1e9),
            })
        elif used == "v2":
            if ir_path is None:
                packet = out / "AGENT_TASK.md"
                packet.write_text(
                    f"Follow skills/solar-estimate/SKILL.md.\nReference: {source}\n"
                    f"Source type: {source_type}\nWrite executable Affine Loop IR preserving "
                    "all as-written work, then call estimate(..., mode='v2', ir_path=...).\n"
                    "Do not alter reference, tolerances, hardware or work sizes.\n")
                raise AgentTranslationRequired(f"Agent IR required; task packet: {packet}")
            ir = Path(ir_path).resolve(strict=True)
        if used in {"v1", "v2"}:
            graph, validation = _validated_ir(source, ir, out, source_type, timeout,
                                               mem_budget_gb, rtol, atol)
        fused, unfused, macs, macs_typed = _analyze(graph, out, arch, precision)
        result = EstimateResult(mode, used, fused, unfused, macs, str(out), _sha(source),
                                _sha(arch), precision, validation, fallback, macs_typed=macs_typed)
        (out / "result.json").write_text(json.dumps(asdict(result), indent=2, allow_nan=False))
        return result
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps({
            "type": type(exc).__name__, "message": str(exc), "mode": mode,
            "used_mode": used, "fallback_reason": fallback,
        }, indent=2))
        raise
