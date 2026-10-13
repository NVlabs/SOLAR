#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-check SOLAR SOL estimates from run_execbench_problem.py against a comparison table you provide.

``--ref`` is a CSV with at least ``artifact_id, workload_uuid, subset, sol_latency_ms``;
``reference_latency_ms`` and ``optimized_baseline_latency_ms`` are carried through when present.
Nothing is shipped with this repository: the table is whatever per-workload SOL you want to check
against. Every ``sol_summary.json`` under ``--results-root`` is matched on (problem name, workload
uuid) and the fused SOL is compared.

Usage::

    python scripts/crosscheck_execbench_ref.py \
        --ref my_comparison.csv \
        --results-root out/execbench -o out/execbench/crosscheck.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True, help="comparison CSV (see module docstring)")
ap.add_argument("--results-root", type=Path, action="append",
                help="Root(s) holding <problem>/<uuid>/sol_summary.json. May be given several times; later "
                     "roots override earlier ones for the same (problem, uuid). Default: out/execbench")
ap.add_argument("--floor-ms", type=float, default=0.0,
                help="Fixed latency (ms) added to our SOL before comparing, if the comparison table "
                     "includes a launch floor (default 0).")
ap.add_argument("-o", "--output", type=Path)
ap.add_argument("--tol", type=float, default=0.10, help="relative tolerance for 'match' (default 10%%)")
ap.add_argument("--mode", default="fused", choices=["fused", "unfused", "fused_prefetched"])
args = ap.parse_args()

ref = {}
_ref_reader = csv.DictReader(open(args.ref))
_missing = [c for c in ("artifact_id", "workload_uuid", "subset", "sol_latency_ms") if c not in (_ref_reader.fieldnames or [])]
if _missing:
    raise SystemExit(f"--ref {args.ref}: expected columns artifact_id, workload_uuid, subset, sol_latency_ms (missing: {_missing})")
for r in _ref_reader:
    if r.get("sol_latency_ms"):
        ref[(r["artifact_id"], r["workload_uuid"])] = (r["subset"], float(r["sol_latency_ms"]),
                                                       r.get("optimized_baseline_latency_ms"),
                                                       r.get("reference_latency_ms"))

roots = args.results_root or [ROOT / "out" / "execbench"]
summaries = {}
for root in roots:
    for f in sorted(root.glob("*/*/sol_summary.json")):
        d = json.loads(f.read_text())
        summaries[(d["problem"], d["workload_uuid"])] = d
rows = []
for key, d in sorted(summaries.items()):
    ours = d["sol_ms"][args.mode] + args.floor_ms
    if key not in ref:
        rows.append({"problem": d["problem"], "uuid": d["workload_uuid"], "subset": "?",
                     "ours_ms": ours, "ref_ms": None, "ratio": None, "status": "no-ref"})
        continue
    subset, ref_ms, t_b, t_ref = ref[key]
    ratio = ours / ref_ms if ref_ms > 0 else math.inf
    status = "match" if abs(ratio - 1) <= args.tol else ("ours-higher" if ratio > 1 else "ours-lower")
    rows.append({"problem": d["problem"], "uuid": d["workload_uuid"], "subset": subset,
                 "ours_ms": ours, "ref_ms": ref_ms, "ratio": ratio, "status": status,
                 "precision": d.get("precision"), "fp32_policy": d.get("fp32_policy", "fp32"),
                 "quant": ",".join(d.get("quant_dtypes") or []),
                 "macs": d.get("total_macs"), "bottleneck": d["bottleneck"][args.mode],
                 "ref_baseline_ms": t_b, "ref_reference_ms": t_ref})

if args.output:
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["problem"])
        w.writeheader(); w.writerows(rows)

matched = [r for r in rows if r["ratio"] is not None]
print(f"{len(rows)} results, {len(matched)} with a reference SOL, {len(rows) - len(matched)} without"
      f"  (mode={args.mode}, floor added to ours={args.floor_ms} ms)")
if matched:
    ratios = [r["ratio"] for r in matched]
    logs = [math.log(x) for x in ratios if x > 0]
    print(f"ratio ours/ref: median {statistics.median(ratios):.3f}, geo-mean {math.exp(statistics.mean(logs)):.3f}, "
          f"min {min(ratios):.3f}, max {max(ratios):.3f}")
    for st in ("match", "ours-higher", "ours-lower"):
        n = sum(1 for r in matched if r["status"] == st)
        print(f"  {st:<12} {n:>4}  ({100 * n / len(matched):.0f}%)  [tol ±{args.tol:.0%}]")
    by_subset = {}
    for r in matched:
        by_subset.setdefault(r["subset"], []).append(r["ratio"])
    print("\nper subset (median ratio, n):")
    for s, v in sorted(by_subset.items()):
        print(f"  {s:<18} {statistics.median(v):.3f}  n={len(v)}")
    print("\nlargest deviations:")
    for r in sorted(matched, key=lambda r: abs(math.log(r["ratio"])) if r["ratio"] > 0 else 99, reverse=True)[:15]:
        print(f"  {r['ratio']:8.3f}x  ours={r['ours_ms']:.5g} ms  ref={r['ref_ms']:.5g} ms  {r['subset']}/{r['problem']}  [{r['precision']} {r['bottleneck']}]")
