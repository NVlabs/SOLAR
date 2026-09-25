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
from typing import Any, Dict, List

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
def pick_precision(definition: Dict[str, Any], override: str | None) -> str:
    if override:
        return override
    counts: Dict[str, int] = {}
    for spec in definition["inputs"].values():
        if spec["shape"] is None:
            continue
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
        # torchview only records ops applied to (descendants of) the traced
        # input tensors; tensors created inside run() from scalars are
        # invisible to it, so the graph would be empty. Fail with a clear
        # message rather than an "einsum_graph has no layers" error later.
        sys.exit("Problem has no tensor inputs (scalars only). torchview cannot record ops on "
                 "tensors created inside run(), so SOLAR has nothing to trace here.")
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
            "    # Custom inputs are materialised on CPU (not meta): references often do",
            "    # data-dependent indexing (index_add, topk, cu_seqlens) that needs values.",
            f"    _custom = _ref.{custom_fn}(axes_and_scalars, torch.device('cpu'))",
            "    for _k in _CUSTOM_SCALARS:",
            "        _SCALARS[_k] = _custom[_k]",
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


def run_solar(model_file: Path, out_base: Path, arch: str, precision: str, verbose: bool) -> Path:
    graph_out = out_base / "graph"
    einsum_out = out_base / "einsum"
    analysis_out = out_base / "analysis"
    perf_out = out_base / "perf"
    for d in (graph_out, einsum_out, analysis_out, perf_out):
        d.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print("==> Stage 1: PyTorch graph extraction")
    run_stage([py, "-m", "solar.cli.process_model", "--model-file", str(model_file),
               "--output-dir", str(graph_out), "--force-rerun"], verbose)
    print("==> Stage 2: einsum conversion")
    run_stage([py, "-m", "solar.cli.toeinsum_model", "--graph-path", str(graph_out / "pytorch_graph.yaml"),
               "--output-dir", str(einsum_out), "--no-copy-graph"], verbose)
    print("==> Stage 3: hardware-independent analysis")
    run_stage([py, "-m", "solar.cli.analyze_model",
               "--einsum-graph-path", str(einsum_out / "einsum_graph_renamed.yaml"),
               "--output-dir", str(analysis_out), "--precision", precision], verbose)
    print(f"==> Stage 4: SOL perf prediction ({arch}, {precision})")
    run_stage([py, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(analysis_out / "analysis.yaml"),
               "--output-dir", str(perf_out), "--arch-config", arch, "--precision", precision], verbose)

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
    parser.add_argument("--arch-config", default="B200",
                        help="SOLAR arch config name under configs/arch or a YAML path (default: B200, "
                             "the SOL-ExecBench leaderboard target)")
    parser.add_argument("--precision", help="Override SOLAR precision key (fp16, bf16, fp8, ...). "
                                            "Default: inferred from the definition's input dtypes.")
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
    precision = pick_precision(definition, args.precision)

    out_base = args.output_dir or (SOLAR_ROOT / "out" / "execbench" / definition["name"] / str(workload.get("uuid", args.workload_index)))
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Problem : {definition['name']}")
    print(f"Dir     : {problem_dir}")
    print(f"Workload: {workload.get('uuid')}  axes={workload['axes']}")
    print(f"Resolved axes: {axes}")
    print(f"Precision: {precision}   Arch: {args.arch_config}")
    print(f"Output  : {out_base}")

    quant_dtypes = detect_quant_dtypes(definition)
    write_quant_metadata(out_base, quant_dtypes)
    if quant_dtypes:
        print(f"Quantized dtypes detected: {quant_dtypes} -> SOLAR will apply the matching TC rate/byte width")

    model_file = out_base / "model.py"
    generate_model_file(definition, workload, axes, scalars, model_file)
    print(f"Generated SOLAR model file: {model_file}")

    perf_path = run_solar(model_file, out_base, args.arch_config, precision, args.verbose)
    perf = yaml.safe_load(perf_path.read_text())

    summary = {
        "problem": definition["name"],
        "problem_dir": str(problem_dir),
        "workload_uuid": workload.get("uuid"),
        "workload_axes": workload["axes"],
        "resolved_axes": axes,
        "arch": perf.get("arch", {}).get("name", args.arch_config),
        "precision": precision,
        "quant_dtypes": quant_dtypes,
        "perf_mac_key": perf.get("arch", {}).get("mac_per_cycle_key"),
        "perf_bytes_per_element": perf.get("workload", {}).get("bytes_per_element"),
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
