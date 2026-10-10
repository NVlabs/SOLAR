#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-run only the perf stage of previous SOL-ExecBench results under a new setup.

The trace, einsum conversion and hardware-independent analysis do not depend
on the arch config or the precision policy; only ``predict_perf_model`` does.
Changing the fp32 policy (TF32 -> 16-bit rate), the arch, or a per-problem
precision override therefore needs no re-tracing and no re-analysis: this
script reads every ``<problem>/<uuid>/sol_summary.json`` + ``analysis/analysis.yaml``
under ``--from-root``, re-prices it with the setup config (or explicit flags)
and writes a complete result tree under ``--to-root`` (analysis + einsum are
symlinked, perf and sol_summary.json are new). One process, seconds per shape.

    python scripts/reprice_execbench.py --from-root out/execbench_v9 --to-root out/execbench_v11 \
        --setup-config configs/execbench/leaderboard_b200.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("run_execbench_problem", ROOT / "scripts" / "run_execbench_problem.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)  # type: ignore[union-attr]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--from-root", type=Path, required=True)
ap.add_argument("--to-root", type=Path, required=True)
ap.add_argument("--setup-config", type=Path)
ap.add_argument("--arch-config", default=None)
ap.add_argument("--fp32-as", default=None, choices=["fp32", "tf32", "fp16"])
ap.add_argument("--precision", default=None, help="Force one precision for every shape (rarely wanted)")
ap.add_argument("--only-changed", action="store_true",
                help="Only write shapes whose effective precision or arch differs from the source result")
ap.add_argument("--bench-root", type=Path, default=runner.DEFAULT_BENCH_ROOT)
args = ap.parse_args()

from solar.perf import EinsumGraphPerfModel  # noqa: E402

setup = runner.load_setup_config(args.setup_config)
arch = args.arch_config or setup.get("arch_config", "B200")
fp32_as_default = args.fp32_as or setup.get("fp32_as", "fp16")
dtype_bytes = setup.get("bytes_accounting", "per-tensor-dtype") != "uniform"

defs: dict = {}
def definition_for(problem: str):
    if problem not in defs:
        hits = list(args.bench_root.glob(f"*/{problem}/definition.json"))
        defs[problem] = json.loads(hits[0].read_text()) if hits else None
    return defs[problem]

t0 = time.time()
n_total = n_written = n_same = n_skipped = 0
for summ_path in sorted(args.from_root.glob("*/*/sol_summary.json")):
    n_total += 1
    d = json.loads(summ_path.read_text())
    src_dir = summ_path.parent
    analysis = src_dir / "analysis" / "analysis.yaml"
    if not analysis.exists():
        n_skipped += 1
        continue
    definition = definition_for(d["problem"])
    inferred = d.get("inferred_precision") or (runner.pick_precision(definition, None) if definition else d["precision"])
    override = runner.problem_override(setup, d["problem"]) if setup else {}
    fp32_as = override.get("fp32_as", fp32_as_default)
    if args.precision:
        precision = args.precision
    elif override.get("precision"):
        precision = str(override["precision"])
    elif inferred == "fp32" and fp32_as != "fp32":
        precision = fp32_as
    else:
        precision = inferred
    unchanged = (precision == d["precision"] and arch == d.get("arch", arch)
                 and (("per-tensor-dtype" if dtype_bytes else "uniform") == d.get("bytes_accounting")))
    if args.only_changed and unchanged:
        n_same += 1
        continue
    dst = args.to_root / d["problem"] / d["workload_uuid"]
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("analysis", "einsum", "graph", "model.py", "reference_impl.py", "metadata.yaml"):
        s, t = src_dir / sub, dst / sub
        if s.exists() and not t.exists():
            os.symlink(os.path.relpath(s, dst), t)
    perf_dir = dst / "perf"
    perf_dir.mkdir(exist_ok=True)
    perf = EinsumGraphPerfModel(dtype_bytes=dtype_bytes).predict(analysis, perf_dir, arch_config=arch, precision=precision)
    if perf is None:
        n_skipped += 1
        print(f"[skip] {d['problem']}/{d['workload_uuid'][:8]}: perf model failed", flush=True)
        continue
    perf_files = sorted(perf_dir.glob("perf_*.yaml"), key=lambda p: p.stat().st_mtime)
    new = dict(d)
    new.update({
        "precision": precision,
        "inferred_precision": inferred,
        "fp32_policy": fp32_as,
        "arch": perf.get("arch", {}).get("name", arch),
        "perf_mac_key": perf.get("arch", {}).get("mac_per_cycle_key"),
        "perf_bytes_per_element": perf.get("workload", {}).get("bytes_per_element"),
        "bytes_accounting": "per-tensor-dtype" if dtype_bytes else "uniform",
        "total_macs": perf.get("workload", {}).get("total_macs"),
        "total_flops": perf.get("workload", {}).get("total_flops"),
        "sol_ms": {m: perf.get(m, {}).get("runtime_ms") for m in ("unfused", "fused", "fused_prefetched")},
        "bottleneck": {m: perf.get(m, {}).get("bottleneck") for m in ("unfused", "fused", "fused_prefetched")},
        "memory_bytes": {m: perf.get(m, {}).get("memory_bytes") for m in ("unfused", "fused", "fused_prefetched")},
        "perf_yaml": str(perf_files[-1]) if perf_files else None,
        "repriced_from": str(summ_path),
        "setup_config": {
            "name": setup.get("name"), "sha256": setup.get("_sha256"), "path": setup.get("_path"),
            "arch_config": arch, "fp32_as": fp32_as,
            "bytes_accounting": "per-tensor-dtype" if dtype_bytes else "uniform",
            "precision_override": ({"precision": precision, "reason": override.get("reason", "")}
                                   if override.get("precision") and not args.precision else None),
        } if setup else None,
    })
    (dst / "sol_summary.json").write_text(json.dumps(new, indent=2) + "\n")
    n_written += 1
    if n_written % 200 == 0:
        print(f"  {n_written} repriced ({time.time() - t0:.0f}s)", flush=True)

print(f"{n_total} source shapes: {n_written} repriced, {n_same} unchanged (skipped), {n_skipped} without analysis "
      f"-> {args.to_root}  arch={arch} fp32_as={fp32_as_default} bytes={'per-tensor-dtype' if dtype_bytes else 'uniform'} "
      f"({time.time() - t0:.0f}s)")
