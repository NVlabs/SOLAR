#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a case-by-case evidence table for SOL deviations vs the leaderboard reference.

For each problem (workload 0) it joins:
  * ours      — run_execbench_problem.py sol_summary.json (+ optional alternative-policy root
                that overrides rows whose default-policy precision is fp32)
  * old       — the pre-fix llm4arch SOLAR perf_B200.yaml, when present (same MAC/byte model
                lineage as the reference table)
  * reference — sol_latencies.csv (sol_latency_ms, optimized baseline, reference latency)
and derives structural flags from the traced graph (MoE/gather ops, full-tensor casts,
sequential loops) plus a first-pass automatic verdict. The verdict is a *suggestion*; the
final judgement is made by reading the row.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OLD_ROOTS = sorted(glob.glob("/home/scratch.jennyhuang_research/llm4arch/solar/solexecbench_output*"))
FLOOR = 0.0004

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True)
ap.add_argument("--results-root", type=Path, default=ROOT / "out" / "execbench")
ap.add_argument("--policy-root", type=Path, help="root of the --fp32-as fp16 run; overrides fp32 rows")
ap.add_argument("--tol", type=float, default=0.10)
ap.add_argument("-o", "--output", type=Path, default=ROOT / "out" / "execbench" / "deviations.csv")
args = ap.parse_args()

ref = {}
for r in csv.DictReader(open(args.ref)):
    ref[(r["artifact_id"], r["workload_uuid"])] = r

def load_root(root):
    out = {}
    for f in glob.glob(f"{root}/*/*/sol_summary.json"):
        d = json.loads(Path(f).read_text()); d["_dir"] = str(Path(f).parent)
        out[(d["problem"], d["workload_uuid"])] = d
    return out

ours = load_root(args.results_root)
policy = load_root(args.policy_root) if args.policy_root else {}

def old_perf(subset, name):
    for r in OLD_ROOTS:
        fs = glob.glob(f"{r}/{subset}/{name}/**/perf_B200.yaml", recursive=True)
        if fs:
            y = yaml.safe_load(open(fs[0]))
            return {"old_macs": y["workload"]["total_macs"], "old_fused_bytes": y["fused"]["memory_bytes"],
                    "old_fused_ms": y["fused"]["runtime_ms"], "old_bpe": y["workload"].get("bytes_per_element")}
    return {}

GATHER_OPS = ("index_select", "index_add", "index_add_", "index_copy", "gather", "scatter", "scatter_add",
              "embedding", "bincount", "nonzero", "where", "topk", "sort", "argsort", "cumsum", "repeat_interleave")

def graph_flags(d):
    g = Path(d["_dir"]) / "graph" / "pytorch_graph.yaml"
    flags = {"n_ops": 0, "gather_ops": 0, "getitem_ops": 0, "full_cast_ops": 0, "moe_like": False}
    if not g.exists():
        return flags
    L = yaml.safe_load(open(g))["layers"]
    types = [str(x.get("type", "")).lower() for x in L.values()]
    flags["n_ops"] = sum(1 for t in types if not t.endswith("tensor"))
    flags["gather_ops"] = sum(1 for t in types if t in GATHER_OPS)
    flags["getitem_ops"] = sum(1 for t in types if t in ("__getitem__", "getitem"))
    flags["full_cast_ops"] = sum(1 for n, x in L.items() if str(x.get("type", "")).lower() in ("to", "float", "half", "bfloat16")
                                 and x.get("input_shapes") and math.prod(x["input_shapes"][0] or [1]) > 50_000_000)
    name = d["problem"].lower()
    flags["moe_like"] = any(k in name for k in ("moe", "expert", "routing", "dispatch", "topk", "router"))
    return flags

rows = []
for key, d in sorted(ours.items()):
    rr = ref.get(key)
    if not rr:
        continue
    used = d
    policy_used = "fp32"
    if d.get("precision") == "fp32" and key in policy:
        used = policy[key]; policy_used = used.get("fp32_policy", "fp16")
    ref_ms = float(rr["sol_latency_ms"])
    ours_ms = used["sol_ms"]["fused"]
    ours_unfused = used["sol_ms"]["unfused"]
    # The reference table adds a ~0.4 us launch floor to most rows but not all; accept either form.
    ratio_raw = ours_ms / ref_ms if ref_ms > 0 else math.inf
    ratio_floor = (ours_ms + FLOOR) / ref_ms if ref_ms > 0 else math.inf
    ratio = ratio_floor if abs(math.log(ratio_floor)) <= abs(math.log(ratio_raw)) else ratio_raw
    sub = rr["subset"]
    row = {"subset": sub, "problem": d["problem"], "uuid": key[1], "precision_used": used.get("precision"),
           "fp32_policy": policy_used, "quant": ",".join(d.get("quant_dtypes") or []),
           "ours_fused_ms": ours_ms, "ours_unfused_ms": ours_unfused, "ours_plus_floor_ms": ours_ms + FLOOR,
           "ref_sol_ms": ref_ms, "ratio": ratio,
           "ours_macs": used["total_macs"], "ours_fused_bytes": used["memory_bytes"]["fused"],
           "ours_unfused_bytes": used["memory_bytes"]["unfused"], "bottleneck": used["bottleneck"]["fused"],
           "ref_baseline_ms": rr.get("optimized_baseline_latency_ms"), "ref_reference_ms": rr.get("reference_latency_ms")}
    row.update(old_perf(sub, d["problem"]))
    row.update(graph_flags(used))
    # first-pass verdict suggestion (order matters: most specific evidence first)
    dev = abs(math.log(ratio)) if 0 < ratio < math.inf else 99
    same_as_old = (row.get("old_macs") is not None
                   and abs(row["old_macs"] - row["ours_macs"]) <= 0.02 * max(1, row["ours_macs"])
                   and abs(row["old_fused_bytes"] - row["ours_fused_bytes"]) <= 0.02 * max(1, row["ours_fused_bytes"]))
    if dev <= math.log(1 + args.tol):
        v = "match"
    elif ref_ms <= FLOOR + 1e-9 and ours_ms < 5 * FLOOR:
        v = "match (both at launch floor)"
    elif row["quant"].startswith("float4") and ratio > 1.3:
        v = "nvfp4: reference counted packed FP4 pairs as one MAC (ref MACs 2x low); ours unpacks"
    elif row["full_cast_ops"] and ratio > 5:
        v = "reference code casts the whole KV cache before gathering; ours traces that faithfully (ref counts gathered pages only)"
    elif row["moe_like"] and row["gather_ops"] > 0 and ratio > 1:
        v = "MoE/gather: ours counts precise per-expert access regions; reference assumed minimum loads (ref optimistic)"
    elif ref_ms <= 1e-4 and ours_ms > 10 * ref_ms:
        v = "reference entry below any launch floor (< 0.1 us): reference value looks erroneous"
    elif ratio < 1 and ours_ms < ref_ms < ours_unfused:
        v = "reference between our fused and unfused SOL: manual model assumed a multi-kernel decomposition (less fusion); ours is the fully-fused lower bound"
    elif same_as_old and ratio > 1 and row["bottleneck"] == "memory":
        v = "ours == old SOLAR, memory-bound: reference undercounts external I/O (verified on L1/033: ref = 2 of 3 tensors)"
    elif same_as_old:
        v = "ours == old SOLAR; reference value was adjusted by hand / other source"
    elif row.get("old_macs") is not None and row["ours_macs"] < 0.6 * row["old_macs"]:
        v = "ours lower MACs than old SOLAR: check (likely old bug, e.g. dense-vs-depthwise)"
    elif row.get("old_macs") is not None and row["ours_macs"] > 1.6 * row["old_macs"]:
        v = "ours counts more MACs than the manual model: reference skipped work the reference code performs (e.g. full-K projection, both attention-backward matmuls)"
    else:
        v = "investigate"
    row["verdict_suggestion"] = v
    rows.append(row)

with open(args.output, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

from collections import Counter
c = Counter(r["verdict_suggestion"] for r in rows)
print(f"{len(rows)} problems; verdict suggestions:")
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")
print("\nrows not 'match' (sorted by |log ratio|):")
for r in sorted(rows, key=lambda r: -abs(math.log(r["ratio"])) if 0 < r["ratio"] < math.inf else -99):
    if r["verdict_suggestion"].startswith("match"):
        continue
    old = f"old={r['old_fused_ms']:.4g}" if r.get("old_fused_ms") is not None else "old=n/a"
    print(f"  {r['ratio']:8.3f}x ours={r['ours_fused_ms']:.4g} ref={r['ref_sol_ms']:.4g} {old} {r['subset']}/{r['problem'][:48]}"
          f" [{r['precision_used']} {r['bottleneck']} gather={r['gather_ops']} moe={int(r['moe_like'])}] -> {r['verdict_suggestion'][:60]}")
print(f"\nwrote {args.output}")
