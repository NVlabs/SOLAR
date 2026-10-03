#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export SOLAR's SOL-ExecBench results next to the leaderboard reference table.

Reads every ``<problem>/<uuid>/sol_summary.json`` under ``--results-root`` (one or more
roots; later roots override earlier ones for the same (problem, uuid)) and the reference
``sol_latencies.csv`` (columns ``artifact_id,workload_uuid,subset,reference_latency_ms,
optimized_baseline_latency_ms,sol_latency_ms``) and writes two files:

* ``<out>_compare.csv`` — one row per reference workload: all reference columns plus
  SOLAR's fused / unfused / prefetched SOL, MACs, bytes, bottleneck, precision, the
  ours/ref ratio (with and without the reference's ~0.4 us launch floor) and a status.
* ``<out>_solar.csv`` — a drop-in replacement for ``sol_latencies.csv`` whose
  ``sol_latency_ms`` is SOLAR's fused SOL (no floor added); rows SOLAR could not produce
  keep the reference value and are flagged in ``sol_source``.

Usage::

    python scripts/export_execbench_csv.py \
        --ref /home/scratch.jennyhuang_research/llm4arch/sol-bench/data/sol_latencies.csv \
        --results-root out/execbench_allshapes -o out/execbench_allshapes/sol_latencies
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOOR = 0.0004

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True)
ap.add_argument("--results-root", type=Path, action="append", required=True)
ap.add_argument("-o", "--out-prefix", type=Path, required=True)
ap.add_argument("--tol", type=float, default=0.10)
args = ap.parse_args()

ours = {}
for root in args.results_root:
    for f in glob.glob(f"{root}/*/*/sol_summary.json"):
        d = json.loads(Path(f).read_text())
        ours[(d["problem"], d["workload_uuid"])] = d

ref_rows = list(csv.DictReader(open(args.ref)))
ref_fields = list(ref_rows[0].keys())

compare_fields = ref_fields + [
    "solar_sol_fused_ms", "solar_sol_fused_plus_floor_ms", "solar_sol_unfused_ms",
    "solar_sol_fused_prefetched_ms", "solar_macs", "solar_fused_bytes", "solar_unfused_bytes",
    "solar_bottleneck", "solar_precision", "solar_fp32_policy", "solar_quant_dtypes",
    "solar_scalars_only_fallback", "ratio_fused_over_ref", "ratio_fused_plus_floor_over_ref",
    "status",
]
compare_rows, solar_rows = [], []
ratios = []
n_missing = 0
for r in ref_rows:
    key = (r["artifact_id"], r["workload_uuid"])
    d = ours.get(key)
    row = dict(r)
    srow = dict(r)
    if d is None:
        n_missing += 1
        row.update({k: "" for k in compare_fields if k not in r})
        row["status"] = "no-solar-result"
        srow["sol_source"] = "reference (SOLAR result missing)"
    else:
        ref_ms = float(r["sol_latency_ms"]) if r["sol_latency_ms"] else float("nan")
        fused = d["sol_ms"]["fused"]
        ratio = fused / ref_ms if ref_ms > 0 else float("inf")
        ratio_f = (fused + FLOOR) / ref_ms if ref_ms > 0 else float("inf")
        best = ratio_f if abs(math.log(ratio_f)) <= abs(math.log(ratio)) else ratio
        if abs(best - 1) <= args.tol:
            status = "match"
        elif best > 1:
            status = "solar-higher"
        else:
            status = "solar-lower"
        ratios.append(best)
        row.update({
            "solar_sol_fused_ms": f"{fused:.6g}",
            "solar_sol_fused_plus_floor_ms": f"{fused + FLOOR:.6g}",
            "solar_sol_unfused_ms": f"{d['sol_ms']['unfused']:.6g}",
            "solar_sol_fused_prefetched_ms": f"{d['sol_ms']['fused_prefetched']:.6g}",
            "solar_macs": d["total_macs"],
            "solar_fused_bytes": d["memory_bytes"]["fused"],
            "solar_unfused_bytes": d["memory_bytes"]["unfused"],
            "solar_bottleneck": d["bottleneck"]["fused"],
            "solar_precision": d.get("precision"),
            "solar_fp32_policy": d.get("fp32_policy", "fp32"),
            "solar_quant_dtypes": ",".join(d.get("quant_dtypes") or []),
            "solar_scalars_only_fallback": int(bool(d.get("scalars_only_fallback"))),
            "ratio_fused_over_ref": f"{ratio:.4f}",
            "ratio_fused_plus_floor_over_ref": f"{ratio_f:.4f}",
            "status": status,
        })
        srow["sol_latency_ms"] = f"{fused:.6g}"
        srow["sol_source"] = "SOLAR fused (no launch floor)"
    compare_rows.append(row)
    solar_rows.append(srow)

cmp_path = Path(f"{args.out_prefix}_compare.csv")
with open(cmp_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=compare_fields)
    w.writeheader(); w.writerows(compare_rows)
sol_path = Path(f"{args.out_prefix}_solar.csv")
with open(sol_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=ref_fields + ["sol_source"])
    w.writeheader(); w.writerows(solar_rows)

n = len(ref_rows)
print(f"{n} reference workloads; SOLAR results for {n - n_missing}, missing {n_missing}")
if ratios:
    logs = [math.log(x) for x in ratios if 0 < x < math.inf]
    print(f"ours/ref (floor accepted either way): median {statistics.median(ratios):.3f}, "
          f"geo-mean {math.exp(statistics.mean(logs)):.3f}")
    from collections import Counter
    c = Counter(r["status"] for r in compare_rows)
    for k in ("match", "solar-higher", "solar-lower", "no-solar-result"):
        print(f"  {k:<16} {c.get(k, 0):5d}  ({100 * c.get(k, 0) / n:.0f}%)")
    per = {}
    for r, x in zip([r for r in compare_rows if r["status"] != "no-solar-result"], ratios):
        per.setdefault(r["subset"], []).append(x)
    print("per subset median ratio:", {k: round(statistics.median(v), 3) for k, v in sorted(per.items())})
print(f"wrote {cmp_path}\nwrote {sol_path}")
