#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the SOLAR pipeline on one SOL-ExecBench problem at one workload shape.

SOL-ExecBench stores a problem as ``definition.json`` (axes, input/output
tensors, PyTorch ``run()`` reference) plus ``workload.jsonl`` (one concrete
binding of the ``var`` axes per line).  SOLAR consumes a Python file that
defines ``Model`` and ``get_inputs()``.  This script bridges the two:

1. Resolve all axes for the chosen workload (const, var, expr).
2. Generate a SOLAR model file that wraps the reference ``run()`` in a
   ``Model`` and allocates inputs with the resolved shapes and dtypes.
3. Run the four SOLAR stages (graph -> einsum -> analysis -> perf).
4. Print the SOL (speed-of-light) runtime and write ``sol_summary.json``.

Example::

    python scripts/run_execbench_problem.py \
        /home/scratch.jennyhuang_research/SOL-ExecBench/data/benchmark/L1/<problem> \
        --workload-index 0 --arch-config B200

With no problem argument, the first L1 problem (sorted by name) is used.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_ROOT = SOLAR_ROOT.parent / "SOL-ExecBench" / "data" / "benchmark"

# SOL-ExecBench dtype string -> (torch dtype expr, SOLAR precision key)
DTYPE_MAP = {
    "float64": ("torch.float64", "fp64"),
    "float32": ("torch.float32", "fp32"),
    "float16": ("torch.float16", "fp16"),
    "bfloat16": ("torch.bfloat16", "bf16"),
    "float8_e4m3fn": ("torch.float8_e4m3fn", "fp8"),
    "float8_e5m2": ("torch.float8_e5m2", "fp8"),
    "float4_e2m1fn_x2": ("torch.float4_e2m1fn_x2", "nvfp4"),
    "float4_e2m1": ("torch.uint8", "nvfp4"),
    "int64": ("torch.int64", "int8"),
    "int32": ("torch.int32", "int8"),
    "int16": ("torch.int16", "int8"),
    "int8": ("torch.int8", "int8"),
    "bool": ("torch.bool", "int8"),
}
FLOAT_DTYPES = {"float64", "float32", "float16", "bfloat16"}

# SOL-ExecBench targets Python >= 3.12; some references use enum.StrEnum
# (3.11+). Backfill it so the reference can be imported on older interpreters.
PY_COMPAT_SHIM = """\
import enum as _enum_compat
if not hasattr(_enum_compat, "StrEnum"):
    class _StrEnum(str, _enum_compat.Enum):
        def __str__(self):
            return str(self.value)
    _enum_compat.StrEnum = _StrEnum
del _enum_compat
import torch as _torch_compat
if not hasattr(_torch_compat, "float4_e2m1fn_x2"):
    # torch < 2.8 has no packed FP4 dtype. Alias to uint8 (same 1-byte packed
    # element) so shape-level tracing works; SOLAR's nvfp4 rates come from
    # metadata.yaml, not from this dtype.
    _torch_compat.float4_e2m1fn_x2 = _torch_compat.uint8


def _solar_scaled_mm(mat_a, mat_b, scale_a=None, scale_b=None, bias=None,
                     scale_result=None, out_dtype=None, use_fast_accum=False, **_kw):
    # torch._scaled_mm has no CPU/meta kernel. SOLAR only needs shapes/dataflow,
    # so replace it with a plain matmul. Packed FP4 operands hold 2 elements per
    # byte along K; unpack so the K extent (and hence MAC count) is correct.
    import torch as _t
    _packed = {_t.uint8, getattr(_t, "float4_e2m1fn_x2", _t.uint8)}
    a, b = mat_a, mat_b
    if a.dtype in _packed:
        a = a.view(_t.uint8).repeat_interleave(2, dim=-1)
    if b.dtype in _packed:
        b = b.view(_t.uint8).repeat_interleave(2, dim=-2)
    out = _t.matmul(a.to(_t.float32), b.to(_t.float32))
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.to(out_dtype or _t.bfloat16)


_torch_compat._scaled_mm = _solar_scaled_mm
del _torch_compat
"""


# --------------------------------------------------------------------------- #
# Problem loading
# --------------------------------------------------------------------------- #
def pick_default_problem(bench_root: Path) -> Path:
    l1 = bench_root / "L1"
    problems = sorted(p for p in l1.iterdir() if (p / "definition.json").exists())
    if not problems:
        sys.exit(f"No problems found under {l1}. Run SOL-ExecBench/scripts/download_data.sh first.")
    return problems[0]


def load_workload(problem_dir: Path, index: int | None, uuid: str | None) -> Dict[str, Any]:
    lines = [ln for ln in (problem_dir / "workload.jsonl").read_text().splitlines() if ln.strip()]
    workloads = [json.loads(ln) for ln in lines]
    if uuid is not None:
        for w in workloads:
            if w.get("uuid") == uuid:
                return w
        sys.exit(f"Workload uuid {uuid} not found in {problem_dir / 'workload.jsonl'}")
    index = index or 0
    if index >= len(workloads):
        sys.exit(f"Workload index {index} out of range; problem has {len(workloads)} workloads.")
    return workloads[index]


def resolve_axes(definition: Dict[str, Any], workload: Dict[str, Any]) -> Dict[str, int]:
    """Bind const, var and expr axes to concrete ints."""
    # Seed with every workload axis: some problems bind extra axes (e.g. a mode
    # flag consumed by the custom input generator) that the definition omits.
    resolved: Dict[str, int] = {k: int(v) for k, v in workload["axes"].items()}
    pending: Dict[str, str] = {}
    for name, spec in definition["axes"].items():
        kind = spec["type"]
        if kind == "const":
            resolved[name] = int(spec["value"])
        elif kind == "var":
            if name not in workload["axes"]:
                sys.exit(f"Workload does not bind var axis '{name}'")
            resolved[name] = int(workload["axes"][name])
        elif kind == "expr":
            pending[name] = spec["expression"]
        else:
            sys.exit(f"Unknown axis type '{kind}' for axis '{name}'")
    # Iteratively evaluate expressions (they may reference other exprs).
    while pending:
        progressed = False
        for name, expr in list(pending.items()):
            try:
                resolved[name] = int(eval(expr, {"__builtins__": {}}, dict(resolved)))  # noqa: S307
                del pending[name]
                progressed = True
            except NameError:
                continue
        if not progressed:
            sys.exit(f"Could not resolve expr axes: {pending}")
    return resolved


def resolve_dim(dim: Any, axes: Dict[str, int]) -> int:
    """A shape entry is an axis name or a literal integer (e.g. "1" for a singleton dim)."""
    if isinstance(dim, int):
        return dim
    if dim in axes:
        return axes[dim]
    try:
        return int(dim)
    except ValueError:
        sys.exit(f"Shape entry {dim!r} is neither a known axis nor an integer")


def scalar_inputs(definition: Dict[str, Any], workload: Dict[str, Any]) -> Dict[str, Any]:
    """Inputs with shape null are Python scalars.

    Returns a dict of the ones with a literal ``scalar`` value in the workload.
    Scalars marked ``custom`` are produced by the definition's
    ``custom_inputs_entrypoint`` at trace time instead.
    """
    out: Dict[str, Any] = {}
    for name, spec in definition["inputs"].items():
        if spec["shape"] is None:
            desc = workload["inputs"].get(name, {})
            kind = desc.get("type")
            if kind == "scalar":
                out[name] = desc["value"]
            elif kind == "custom":
                if not definition.get("custom_inputs_entrypoint"):
                    sys.exit(f"Scalar input '{name}' is 'custom' but definition has no custom_inputs_entrypoint")
            else:
                sys.exit(f"Scalar input '{name}' has unsupported workload type {kind!r}")
    return out


QUANT_DTYPE_RE = re.compile(r"torch\.(float8_e4m3fn|float8_e5m2|float8_e4m3fnuz|float8_e5m2fnuz|float4_e2m1fn_x2)\b")


def detect_quant_dtypes(definition: Dict[str, Any]) -> List[str]:
    """Quantized dtypes used anywhere in the problem (inputs or inside the reference).

    SOL-ExecBench Quant problems often take bf16 inputs and cast to fp8/fp4
    inside ``run()``, so the input dtypes alone under-report the precision.
    """
    found = set(QUANT_DTYPE_RE.findall(definition["reference"]))
    for spec in definition["inputs"].values():
        if spec["dtype"] in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2", "float4_e2m1"):
            found.add(spec["dtype"])
    return sorted(found)


BLOB_ROOTS = [SOLAR_ROOT.parent / "SOL-ExecBench", DEFAULT_BENCH_ROOT.parent]


def resolve_blob_path(rel: str) -> Path:
    """Resolve a workload safetensors path against the SOL-ExecBench repo roots."""
    rel_p = Path(rel)
    if rel_p.is_absolute() and rel_p.exists():
        return rel_p
    for root in BLOB_ROOTS:
        cand = root / rel_p
        if cand.exists():
            return cand
        # Paths may already start at 'data/...' or at 'blob/...'; try suffix matches.
        parts = rel_p.parts
        for i in range(1, len(parts)):
            cand = root / Path(*parts[i:])
            if cand.exists():
                return cand
    sys.exit(f"safetensors blob not found: {rel} (searched {BLOB_ROOTS}). "
             "Run SOL-ExecBench/scripts/download_data.sh to fetch flashinfer-trace.")


def write_quant_metadata(out_base: Path, quant_dtypes: List[str]) -> None:
    """Write metadata.yaml so SOLAR's analysis/perf stages apply fp8/nvfp4 rates.

    Both stages walk up to 3 directories from their input file looking for
    this file; ``out_base`` is the parent of ``analysis/`` and ``perf/``.
    """
    meta_path = out_base / "metadata.yaml"
    if not quant_dtypes:
        if meta_path.exists():
            meta_path.unlink()
        return
    conversions = [{
        "function": "run",
        "operation": "dtype_annotation",
        "orig_dtypes": ("nvfp4 " if dt.startswith("float4") else "fp8 ") + dt,
        "new_dtypes": "traced as-is",
        "count": 1,
        "reason": "SOL-ExecBench quantized dtype detected in definition/reference by run_execbench_problem.py",
    } for dt in quant_dtypes]
    meta_path.write_text(yaml.safe_dump({"dtype_conversions": conversions}, sort_keys=False))


# --------------------------------------------------------------------------- #
# Model-file generation
# --------------------------------------------------------------------------- #

def load_setup_config(path: Optional[Path]) -> Dict[str, Any]:
    """Read a setup YAML (see configs/execbench/leaderboard_b200.yaml).

    Returns ``{}`` when no path is given. The returned dict carries the file's
    ``name``, ``sha256`` and ``path`` so results can cite exactly which policy
    produced them.
    """
    if path is None:
        return {}
    import hashlib
    text = Path(path).read_text()
    cfg = yaml.safe_load(text) or {}
    cfg["_path"] = str(Path(path).resolve())
    cfg["_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("problem_overrides", {}) 
    return cfg


def problem_override(cfg: Dict[str, Any], problem: str) -> Dict[str, Any]:
    """Per-problem entry of the setup config (exact name or unique prefix)."""
    overrides = cfg.get("problem_overrides") or {}
    if problem in overrides:
        return dict(overrides[problem] or {})
    hits = [k for k in overrides if problem.startswith(k) or k.startswith(problem)]
    return dict(overrides[hits[0]] or {}) if len(hits) == 1 else {}


def pick_precision(definition: Dict[str, Any], override: str | None) -> str:
    if override:
        return override
    counts: Dict[str, int] = {}
    tensor_specs = [sp for sp in definition["inputs"].values() if sp["shape"] is not None]
    if not tensor_specs:
        # scalars-only problem: bytes are governed by what gets written
        tensor_specs = list(definition["outputs"].values())
    for spec in tensor_specs:
        prec = DTYPE_MAP.get(spec["dtype"], ("", "fp16"))[1]
        counts[prec] = counts.get(prec, 0) + 1
    if not counts:
        return "fp16"
    # Prefer the narrowest floating type present; SOLAR applies one precision
    # for the tensor-core MAC rate and bytes/element.
    for prec in ("nvfp4", "fp8", "bf16", "fp16", "fp32", "fp64", "int8"):
        if prec in counts:
            return prec
    return max(counts, key=counts.get)


def write_scalars_only_model(definition, workload, axes, scalars, out_path: Path) -> None:
    """Synthetic output-write model for problems whose inputs are all Python scalars."""
    scalar_names = list(definition["inputs"].keys())
    values = []
    for n in scalar_names:
        v = scalars.get(n)
        if v is None:
            v = 1.0  # custom-generated scalar: the value is irrelevant for shapes
        values.append(float(v) if not isinstance(v, bool) else float(v))
    outs = []
    for name, spec in definition["outputs"].items():
        shape = [resolve_dim(a, axes) for a in (spec["shape"] or [])]
        torch_dtype = DTYPE_MAP[spec["dtype"]][0]
        outs.append((name, shape, torch_dtype))
    body = "\n".join(
        f"        outs.append((vecs[0].expand({shape!r}) * 1.0).to({dt}))  # {name}" if shape
        else f"        outs.append((vecs[0] * 1.0).reshape(()).to({dt}))  # {name} (0-d)"
        for name, shape, dt in outs)
    src = f'''# Auto-generated by SOLAR/scripts/run_execbench_problem.py (scalars-only fallback).
# Problem : {definition["name"]}
# Workload: {workload.get("uuid")}
# The reference takes only Python scalars {scalar_names!r} and builds its outputs
# internally, which torchview cannot trace. This model reproduces the problem's
# speed of light (one write per output) with traceable ops.
import torch
import torch.nn as nn

_SCALAR_VALUES = {values!r}


class Model(nn.Module):
    def forward(self, *vecs):
        outs = []
{body}
        return outs[0] if len(outs) == 1 else tuple(outs)


def get_inputs():
    return [torch.tensor([v], dtype=torch.float32) for v in _SCALAR_VALUES]


def get_init_inputs():
    return []
'''
    out_path.write_text(src)


def generate_model_file(
    definition: Dict[str, Any],
    workload: Dict[str, Any],
    axes: Dict[str, int],
    scalars: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write ``reference_impl.py`` (verbatim reference) and ``model.py`` (SOLAR wrapper).

    The reference lives in its own module because its helper names (e.g. a
    custom input generator called ``get_inputs``) can collide with the
    ``Model`` / ``get_inputs`` names SOLAR looks for in the model file.
    """
    ref_path = out_path.parent / "reference_impl.py"
    ref_path.write_text(PY_COMPAT_SHIM + definition["reference"])

    input_names: List[str] = list(definition["inputs"].keys())
    tensor_names = [n for n in input_names if definition["inputs"][n]["shape"] is not None]
    if not tensor_names:
        # Scalars-only problem: torchview records only ops on descendants of
        # traced tensor inputs, and these references build every tensor from
        # Python scalars (arange/ones with device='cuda'), so tracing run()
        # yields an empty graph. The speed of light for such a problem is
        # writing its outputs (inputs are scalars, compute is elementwise), so
        # emit a synthetic model: each scalar becomes a size-1 vector input and
        # each output is that vector broadcast to the output's resolved shape
        # and cast to its dtype. SOLAR then counts one full write per output
        # and (via the broadcast access region) a 1-element read.
        write_scalars_only_model(definition, workload, axes, scalars, out_path)
        return
    custom_fn = definition.get("custom_inputs_entrypoint")
    uses_custom = any(workload["inputs"].get(n, {}).get("type") == "custom" for n in input_names)

    alloc_lines: List[str] = []
    for name in tensor_names:
        spec = definition["inputs"][name]
        shape = [resolve_dim(a, axes) for a in spec["shape"]]
        torch_dtype, _ = DTYPE_MAP[spec["dtype"]]
        kind = workload["inputs"].get(name, {"type": "random"}).get("type", "random")
        desc = workload["inputs"].get(name, {})
        if kind == "custom":
            alloc_lines.append(f"    tensors[{name!r}] = _custom[{name!r}]")
        elif kind == "safetensors":
            # Real data (page tables, indptr arrays) that references validate/index with.
            st_path = resolve_blob_path(desc["path"])
            alloc_lines.append(
                f"    tensors[{name!r}] = _load_safetensor({str(st_path)!r}, {desc['tensor_key']!r})")
        elif spec["dtype"] in FLOAT_DTYPES:
            alloc_lines.append(f"    tensors[{name!r}] = torch.randn({shape}, dtype={torch_dtype})")
        else:
            # randn does not support int/bool/fp8 dtypes; zeros is enough for
            # shape/dtype tracing (safetensors inputs are also stubbed this way).
            alloc_lines.append(f"    tensors[{name!r}] = torch.zeros({shape}, dtype={torch_dtype})")

    if uses_custom:
        if not custom_fn:
            sys.exit("Workload has 'custom' inputs but definition has no custom_inputs_entrypoint")
        custom_block = "\n".join([
            "    axes_and_scalars = dict(_AXES)",
            "    axes_and_scalars.update(_STATIC_SCALARS)",
            "    # SOLAR calls get_inputs() once for its zero-memory meta-device trace and",
            "    # again (CPU) only if that trace fails on data-dependent ops. Generate on",
            "    # meta for the first call so large shapes don't materialise GBs of real",
            "    # tensors; fall back to CPU for retries or if the generator itself needs",
            "    # real values (e.g. .item()).",
            "    _CALLS[0] += 1",
            "    _custom = None",
            "    if _CALLS[0] == 1:",
            "        try:",
            f"            _custom = _ref.{custom_fn}(axes_and_scalars, torch.device('meta'))",
            "        except Exception:",
            "            _custom = None",
            "    if _custom is None:",
            f"        _custom = _ref.{custom_fn}(axes_and_scalars, torch.device('cpu'))",
            "    for _k in _CUSTOM_SCALARS:",
            "        _v = _custom[_k]",
            "        _SCALARS[_k] = _v.item() if isinstance(_v, torch.Tensor) and _v.numel() == 1 and _v.device.type != 'meta' else _v",
            "",
        ])
    else:
        custom_block = "    _custom = {}\n"

    custom_scalars = [n for n in input_names
                      if definition["inputs"][n]["shape"] is None
                      and workload["inputs"].get(n, {}).get("type") == "custom"]

    # Rebuild the positional argument list for run(): tensors in definition
    # order with scalar values spliced back into their slots.
    arg_exprs = []
    t_idx = 0
    for name in input_names:
        if definition["inputs"][name]["shape"] is None:
            arg_exprs.append(f"_SCALARS[{name!r}]")
        else:
            arg_exprs.append(f"tensors[{t_idx}]")
            t_idx += 1

    src = f'''# Auto-generated by SOLAR/scripts/run_execbench_problem.py.
# Problem : {definition["name"]}
# Workload: {workload.get("uuid")}
# Axes    : {axes}
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_impl as _ref  # SOL-ExecBench reference (verbatim)


def _load_safetensor(path, key):
    from safetensors.torch import load_file
    return load_file(path)[key]

_AXES = {axes!r}
_STATIC_SCALARS = {scalars!r}
_CUSTOM_SCALARS = {custom_scalars!r}
_SCALARS = dict(_STATIC_SCALARS)  # custom scalars are filled in by get_inputs()
_PARAM_ORDER = {input_names!r}
_CALLS = [0]  # get_inputs() call counter: 1st call -> meta device, later calls -> CPU


class Model(nn.Module):
    """Wraps the SOL-ExecBench reference ``run()`` so SOLAR can trace it."""

    def __init__(self):
        super().__init__()

    def forward(self, *tensors):
        return _ref.run({", ".join(arg_exprs)})


def get_inputs():
    torch.manual_seed(0)
{custom_block}
    tensors = {{}}
{chr(10).join(alloc_lines)}
    return [tensors[n] for n in {tensor_names!r}]


def get_init_inputs():
    return []
'''
    out_path.write_text(src)


# --------------------------------------------------------------------------- #
# SOLAR pipeline
# --------------------------------------------------------------------------- #
def run_stage(cmd: List[str], verbose: bool) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=SOLAR_ROOT, text=True, capture_output=not verbose)
    if result.returncode != 0:
        if not verbose:
            sys.stdout.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        sys.exit(f"Stage failed with exit code {result.returncode}: {cmd[2]}")


def _run_stages_in_process(graph_path: Optional[Path], einsum_out: Path, einsum_graph: Path,
                           analysis_out: Path, perf_out: Path, arch: str, precision: str,
                           dtype_bytes: bool) -> None:
    """Stages 2-4 through the Python API in this process.

    The subprocess route pays ~6 s of torch import per stage; for the
    re-analysis sweeps (thousands of shapes, traces reused) that overhead
    dominates. Behaviour matches the CLIs (no graph copy, no rank rename).
    """
    # Import the checked-out package, not a stale site-packages install.
    if str(SOLAR_ROOT) not in sys.path:
        sys.path.insert(0, str(SOLAR_ROOT))
    from solar.analysis import EinsumGraphAnalyzer
    from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    from solar.perf import EinsumGraphPerfModel
    if graph_path is not None:
        print("==> Stage 2: einsum conversion (in-process)")
        if PyTorchToEinsum().convert(str(graph_path), str(einsum_out), copy_graph=False) is None:
            sys.exit("Stage failed: einsum conversion")
    print("==> Stage 3: hardware-independent analysis (in-process)")
    if EinsumGraphAnalyzer().analyze_graph(str(einsum_graph), str(analysis_out),
                                           precision=precision) is None:
        sys.exit("Stage failed: analysis")
    print(f"==> Stage 4: SOL perf prediction ({arch}, {precision}, in-process)")
    if EinsumGraphPerfModel(dtype_bytes=dtype_bytes).predict(
            analysis_out / "analysis.yaml", perf_out, arch_config=arch, precision=precision) is None:
        sys.exit("Stage failed: perf prediction")


def run_solar(model_file: Path, out_base: Path, arch: str, precision: str, verbose: bool,
              reuse_einsum: Optional[Path] = None, dtype_bytes: bool = True,
              reuse_graph: Optional[Path] = None, in_process: bool = False) -> Path:
    graph_out = out_base / "graph"
    einsum_out = out_base / "einsum"
    analysis_out = out_base / "analysis"
    perf_out = out_base / "perf"
    for d in (graph_out, einsum_out, analysis_out, perf_out):
        d.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if reuse_einsum is not None:
        # Stages 1-2 (trace + einsum conversion) do not depend on the arch or
        # precision policy; reuse a previous run's einsum graph and only redo
        # analysis + perf (seconds instead of minutes per shape).
        print(f"==> Stages 1-2 skipped: reusing {reuse_einsum}")
        einsum_graph = reuse_einsum
    else:
        if reuse_graph is not None:
            # The trace (stage 1) is the expensive, policy-independent part;
            # reuse it and redo einsum conversion so converter fixes apply.
            print(f"==> Stage 1 skipped: reusing {reuse_graph}")
            graph_path = reuse_graph
        else:
            print("==> Stage 1: PyTorch graph extraction")
            run_stage([py, "-m", "solar.cli.process_model", "--model-file", str(model_file),
                       "--output-dir", str(graph_out), "--force-rerun"], verbose)
            graph_path = graph_out / "pytorch_graph.yaml"
        einsum_graph = einsum_out / "einsum_graph_renamed.yaml"
        if in_process:
            _run_stages_in_process(graph_path, einsum_out, einsum_graph, analysis_out, perf_out,
                                   arch, precision, dtype_bytes)
            graph_path = None
        else:
            print("==> Stage 2: einsum conversion")
            run_stage([py, "-m", "solar.cli.toeinsum_model", "--graph-path", str(graph_path),
                       "--output-dir", str(einsum_out), "--no-copy-graph"], verbose)
    if in_process and reuse_einsum is not None:
        _run_stages_in_process(None, einsum_out, einsum_graph, analysis_out, perf_out,
                               arch, precision, dtype_bytes)
    if in_process:
        perf_files = sorted(perf_out.glob("perf_*.yaml"), key=lambda p: p.stat().st_mtime)
        if not perf_files:
            sys.exit(f"No perf_*.yaml produced under {perf_out}")
        return perf_files[-1]
    print("==> Stage 3: hardware-independent analysis")
    run_stage([py, "-m", "solar.cli.analyze_model",
               "--einsum-graph-path", str(einsum_graph),
               "--output-dir", str(analysis_out), "--precision", precision], verbose)
    print(f"==> Stage 4: SOL perf prediction ({arch}, {precision})")
    run_stage([py, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(analysis_out / "analysis.yaml"),
               "--output-dir", str(perf_out), "--arch-config", arch, "--precision", precision]
              + (["--dtype-bytes"] if dtype_bytes else []), verbose)

    # The perf file is named after the `name` field inside the arch YAML,
    # which need not match the CLI argument, so locate it by glob.
    perf_files = sorted(perf_out.glob("perf_*.yaml"), key=lambda p: p.stat().st_mtime)
    if not perf_files:
        sys.exit(f"No perf_*.yaml produced under {perf_out}")
    return perf_files[-1]


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem_dir", nargs="?", type=Path,
                        help="SOL-ExecBench problem dir (definition.json + workload.jsonl). "
                             "Default: first L1 problem under --bench-root.")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT,
                        help=f"SOL-ExecBench data/benchmark dir (default: {DEFAULT_BENCH_ROOT})")
    parser.add_argument("--workload-index", type=int, default=0, help="Line index into workload.jsonl (default 0)")
    parser.add_argument("--workload-uuid", help="Select workload by uuid instead of index")
    parser.add_argument("--setup-config", type=Path,
                        help="Setup YAML (arch, fp32 policy, byte accounting, per-problem precision overrides); "
                             "see configs/execbench/leaderboard_b200.yaml. Explicit CLI flags take precedence. "
                             "Name, sha256 and the effective settings are recorded in sol_summary.json.")
    parser.add_argument("--arch-config", default=None,
                        help="SOLAR arch config name under configs/arch or a YAML path (default: B200, "
                             "the SOL-ExecBench leaderboard target, or the setup config's arch_config)")
    parser.add_argument("--precision", help="Override SOLAR precision key (fp16, bf16, fp8, ...). "
                                            "Default: inferred from the definition's input dtypes.")
    parser.add_argument("--fp32-as", default=None, choices=["fp32", "tf32", "fp16"],
                        help="How to price problems whose inferred precision is fp32. 'fp32' (default) uses "
                             "4 B/elem and the CUDA-core rate; 'tf32' the TF32 tensor-core rate with 4 B/elem; "
                             "'fp16' reproduces the leaderboard reference table, which priced every non-quant "
                             "problem at 16-bit tensor-core rate and 2 B/elem.")
    parser.add_argument("--uniform-bytes", action="store_true", default=None,
                        help="Legacy byte accounting: one bytes_per_element for every tensor. Default prices each "
                             "tensor at its own dtype width (bool masks 1 B, fp32 outputs of fp8 problems 4 B ...).")
    parser.add_argument("--reuse-from", type=Path,
                        help="Results root of a previous run. By default reuses that run's traced graph "
                             "(<problem>/<uuid>/graph/pytorch_graph.yaml) and redoes einsum conversion, analysis "
                             "and perf; with --reuse-einsum also reuses the einsum graph and redoes only analysis + perf.")
    parser.add_argument("--reuse-einsum", action="store_true",
                        help="With --reuse-from: also reuse the einsum graph (only valid if the converter is unchanged).")
    parser.add_argument("--in-process", action="store_true",
                        help="Run stages 2-4 through the Python API in this process (skips ~6 s of torch "
                             "import per stage; intended for --reuse-from re-analysis sweeps).")
    parser.add_argument("--out-root", type=Path,
                        help="Root for artifacts (default: SOLAR/out/execbench); each problem gets <root>/<problem>/<uuid>/")
    parser.add_argument("--output-dir", type=Path, help="Where to write artifacts (default: SOLAR/out/execbench/<problem>/<uuid>)")
    parser.add_argument("--t-k", type=float, help="Optional measured kernel latency (ms) to compute a SOL-Score")
    parser.add_argument("--t-b", type=float, help="Optional baseline latency (ms) for the SOL-Score")
    parser.add_argument("-v", "--verbose", action="store_true", help="Stream SOLAR stage output")
    args = parser.parse_args()

    problem_dir = args.problem_dir or pick_default_problem(args.bench_root)
    definition = json.loads((problem_dir / "definition.json").read_text())
    workload = load_workload(problem_dir, args.workload_index, args.workload_uuid)
    axes = resolve_axes(definition, workload)
    scalars = scalar_inputs(definition, workload)

    # Effective setup: setup config supplies defaults, explicit CLI flags win,
    # per-problem overrides in the config apply unless --precision was given.
    setup = load_setup_config(args.setup_config)
    override = problem_override(setup, definition["name"]) if setup else {}
    if args.arch_config is None:
        args.arch_config = setup.get("arch_config", "B200")
    if args.fp32_as is None:
        args.fp32_as = override.get("fp32_as", setup.get("fp32_as", "fp32"))
    if args.uniform_bytes is None:
        args.uniform_bytes = (setup.get("bytes_accounting", "per-tensor-dtype") == "uniform")
    inferred_precision = pick_precision(definition, None)
    precision = pick_precision(definition, args.precision)
    override_applied = None
    if args.precision is None and override.get("precision"):
        precision = str(override["precision"])
        override_applied = {"precision": precision, "reason": override.get("reason", "")}
    elif args.precision is None and precision == "fp32" and args.fp32_as != "fp32":
        precision = args.fp32_as

    out_root = args.out_root or (SOLAR_ROOT / "out" / "execbench")
    out_base = args.output_dir or (out_root / definition["name"] / str(workload.get("uuid", args.workload_index)))
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Problem : {definition['name']}")
    print(f"Dir     : {problem_dir}")
    print(f"Workload: {workload.get('uuid')}  axes={workload['axes']}")
    print(f"Resolved axes: {axes}")
    print(f"Precision: {precision} (inferred {inferred_precision}, fp32-as {args.fp32_as}"
          + (f", override: {override_applied['precision']}" if override_applied else "") + f")   Arch: {args.arch_config}")
    if setup:
        print(f"Setup   : {setup.get('name')} ({setup.get('_sha256')}) {setup.get('_path')}")
    print(f"Output  : {out_base}")

    quant_dtypes = detect_quant_dtypes(definition)
    write_quant_metadata(out_base, quant_dtypes)
    if quant_dtypes:
        print(f"Quantized dtypes detected: {quant_dtypes} -> SOLAR will apply the matching TC rate/byte width")

    model_file = out_base / "model.py"
    generate_model_file(definition, workload, axes, scalars, model_file)
    print(f"Generated SOLAR model file: {model_file}")

    reuse_einsum = None
    reuse_graph = None
    if args.reuse_from:
        prev = args.reuse_from / definition["name"] / str(workload.get("uuid", args.workload_index))
        if args.reuse_einsum and (prev / "einsum" / "einsum_graph_renamed.yaml").exists():
            reuse_einsum = prev / "einsum" / "einsum_graph_renamed.yaml"
        elif (prev / "graph" / "pytorch_graph.yaml").exists():
            reuse_graph = prev / "graph" / "pytorch_graph.yaml"
        else:
            print(f"Note: nothing to reuse under {prev}; running all stages")
    perf_path = run_solar(model_file, out_base, args.arch_config, precision, args.verbose,
                          reuse_einsum=reuse_einsum, dtype_bytes=not args.uniform_bytes, reuse_graph=reuse_graph,
                          in_process=args.in_process)
    perf = yaml.safe_load(perf_path.read_text())

    summary = {
        "problem": definition["name"],
        "problem_dir": str(problem_dir),
        "workload_uuid": workload.get("uuid"),
        "workload_axes": workload["axes"],
        "resolved_axes": axes,
        "arch": perf.get("arch", {}).get("name", args.arch_config),
        "precision": precision,
        "inferred_precision": inferred_precision,
        "fp32_policy": args.fp32_as,
        "setup_config": {
            "name": setup.get("name"), "sha256": setup.get("_sha256"), "path": setup.get("_path"),
            "arch_config": args.arch_config, "fp32_as": args.fp32_as,
            "bytes_accounting": "uniform" if args.uniform_bytes else "per-tensor-dtype",
            "precision_override": override_applied,
        } if setup else None,
        "scalars_only_fallback": not any(sp["shape"] is not None for sp in definition["inputs"].values()),
        "quant_dtypes": quant_dtypes,
        "perf_mac_key": perf.get("arch", {}).get("mac_per_cycle_key"),
        "perf_bytes_per_element": perf.get("workload", {}).get("bytes_per_element"),
        "bytes_accounting": "per-tensor-dtype" if not args.uniform_bytes else "uniform",
        "total_macs": perf.get("workload", {}).get("total_macs"),
        "total_flops": perf.get("workload", {}).get("total_flops"),
        "sol_ms": {
            mode: perf.get(mode, {}).get("runtime_ms") for mode in ("unfused", "fused", "fused_prefetched")
        },
        "bottleneck": {
            mode: perf.get(mode, {}).get("bottleneck") for mode in ("unfused", "fused", "fused_prefetched")
        },
        "memory_bytes": {
            mode: perf.get(mode, {}).get("memory_bytes") for mode in ("unfused", "fused", "fused_prefetched")
        },
        "perf_yaml": str(perf_path),
        "reused_einsum_from": str(reuse_einsum) if reuse_einsum else None,
        "reused_graph_from": str(reuse_graph) if reuse_graph else None,
    }

    if args.t_k is not None and args.t_b is not None:
        sys.path.insert(0, str(SOLAR_ROOT.parent / "SOL-ExecBench" / "src"))
        from sol_execbench.sol_score import sol_score  # type: ignore

        t_sol = summary["sol_ms"]["fused"]
        summary["sol_score"] = sol_score(args.t_k, args.t_b, t_sol)

    (out_base / "sol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== SOL summary ===")
    print(f"MACs            : {summary['total_macs']}")
    for mode, ms in summary["sol_ms"].items():
        print(f"SOL {mode:<16}: {ms} ms  ({summary['bottleneck'][mode]}-bound)")
    if "sol_score" in summary:
        print(f"SOL-Score (t_k={args.t_k}, t_b={args.t_b}): {summary['sol_score']:.4f}")
    print(f"Summary written : {out_base / 'sol_summary.json'}")


if __name__ == "__main__":
    main()
