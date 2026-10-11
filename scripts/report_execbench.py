#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summaries of a SOL-ExecBench comparison CSV (``<out>_compare.csv`` from export_execbench_csv.py --ref).

Writes next to the CSV:
* ``SUMMARY.md``                 shape-level buckets, per-subset agreement, problems above/below the
                                 comparison SOL with optional causes, and the lower-bound check against
                                 measured latencies when the comparison CSV carries them
                                 (``reference_latency_ms``, ``optimized_baseline_latency_ms``);
* ``report_data.json``           the same numbers plus per-shape points for plotting.

    python scripts/report_execbench.py out/execbench_full/sol_table_compare.csv --label "full" [--notes causes.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics as st
from pathlib import Path

# Optional hand-checked causes per problem, loaded from --notes (JSON: {"<problem>": "<cause>"}).
NOTES: dict = {}


def best_ratio(r):
    rf = max(float(r["ratio_fused_over_ref"]), 1e-9)
    rff = max(float(r["ratio_fused_plus_floor_over_ref"]), 1e-9)
    return rff if abs(math.log(rff)) <= abs(math.log(rf)) else rf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare_csv", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--notes", type=Path, help='optional JSON {"<problem>": "<cause>"} shown in the cause column')
    args = ap.parse_args()
    if args.notes:
        NOTES.update(json.loads(args.notes.read_text()))
    out_dir = args.compare_csv.parent
    rows = list(csv.DictReader(open(args.compare_csv)))
    have = [r for r in rows if r["status"] != "no-solar-result"]
    for r in have:
        r["best"] = best_ratio(r)
        r["sol"] = float(r["solar_sol_fused_ms"])
        r["ref_sol"] = float(r["sol_latency_ms"])
        r["ref_impl"] = float(r["reference_latency_ms"]) if r.get("reference_latency_ms") else float("nan")
        r["baseline"] = float(r["optimized_baseline_latency_ms"]) if r.get("optimized_baseline_latency_ms") else float("nan")
        r["measured"] = min(x for x in (r["ref_impl"], r["baseline"]) if not math.isnan(x)) if not (math.isnan(r["ref_impl"]) and math.isnan(r["baseline"])) else float("nan")
    n = len(have)

    def frac(c):
        k = sum(1 for r in have if c(r)); return k, 100 * k / n

    buckets = [
        ("within +-10 %", lambda r: abs(r["best"] - 1) <= 0.10),
        ("above by 10-50 %", lambda r: 1.10 < r["best"] <= 1.5),
        ("above by 1.5-2.1x", lambda r: 1.5 < r["best"] <= 2.1),
        ("above by more than 2.1x", lambda r: r["best"] > 2.1),
        ("below by 10-50 %", lambda r: 1 / 1.5 <= r["best"] < 0.90),
        ("below by more than 1.5x", lambda r: r["best"] < 1 / 1.5),
    ]
    subsets = sorted({r["subset"] for r in have})
    per_subset = {}
    for s in subsets:
        rs = [r for r in have if r["subset"] == s]; m = len(rs)
        per_subset[s] = {
            "shapes": m, "median": st.median(r["best"] for r in rs),
            "within10": sum(1 for r in rs if abs(r["best"] - 1) <= 0.1),
            "above10": sum(1 for r in rs if r["best"] > 1.1),
            "above50": sum(1 for r in rs if r["best"] > 1.5),
            "below10": sum(1 for r in rs if r["best"] < 0.9),
            "above_measured": sum(1 for r in rs if not math.isnan(r["measured"]) and r["sol"] > r["measured"]),
        }
    viol_ref = [r for r in have if not math.isnan(r["ref_impl"]) and r["sol"] > r["ref_impl"]]
    viol_base = [r for r in have if not math.isnan(r["baseline"]) and r["sol"] > r["baseline"]]
    has_ref_impl = any(not math.isnan(r["ref_impl"]) for r in have)
    has_baseline = any(not math.isnan(r["baseline"]) for r in have)
    conf = collections.Counter(r.get("solar_confidence") or "n/a" for r in have)

    per = collections.defaultdict(list)
    for r in have:
        per[(r["subset"], r["artifact_id"])].append(r)
    problems = []
    for (s, p), rs in per.items():
        med = st.median(r["best"] for r in rs)
        confs = collections.Counter(r.get("solar_confidence") or "n/a" for r in rs)
        level = "low" if confs.get("low") else "medium" if confs.get("medium") else ("high" if confs.get("high") else "n/a")
        reason = next((r.get("solar_confidence_reason") for r in rs if r.get("solar_confidence") == level), "")
        problems.append({
            "subset": s, "problem": p, "shapes": len(rs), "median_ratio": med,
            "min_ratio": min(r["best"] for r in rs), "max_ratio": max(r["best"] for r in rs),
            "median_sol_us": st.median(r["sol"] for r in rs) * 1e3,
            "median_ref_sol_us": st.median(r["ref_sol"] for r in rs) * 1e3,
            "median_measured_us": st.median(r["measured"] for r in rs if not math.isnan(r["measured"])) * 1e3 if any(not math.isnan(r["measured"]) for r in rs) else None,
            "max_sol_over_measured": max((r["sol"] / r["measured"] for r in rs if not math.isnan(r["measured"]) and r["measured"] > 0), default=None),
            "bottleneck": rs[0]["solar_bottleneck"], "precision": rs[0]["solar_precision"],
            "confidence": level, "confidence_reason": reason,
            "cause": NOTES.get(p, ""),
        })
    problems.sort(key=lambda t: -t["median_ratio"])
    above = [t for t in problems if t["median_ratio"] > 1.1]
    below = [t for t in problems if t["median_ratio"] < 0.9]

    # ---- SUMMARY.md ---------------------------------------------------------
    L = []
    L.append(f"# SOL-ExecBench x Solar — {args.label or args.compare_csv.parent.name}\n")
    L.append(f"Shapes with a SOL: {n} of {len(rows)}. Ratio = Solar fused SOL / comparison SOL, whichever of (with, without the launch floor) is closer to 1.\n")
    L.append("## Agreement with the comparison SOL table\n")
    L.append("| | value |\n|---|---|")
    L.append(f"| median ours/ref over shapes | {st.median(r['best'] for r in have):.3f} |")
    for name, c in buckets:
        k, pc = frac(c); L.append(f"| {name} | {k} ({pc:.1f} %) |")
    L.append("")
    L.append("| subset | shapes | median | within 10 % | above 10 % | above 50 % | below 10 % |\n|---|---|---|---|---|---|---|")
    for s in subsets:
        d = per_subset[s]
        L.append(f"| {s} | {d['shapes']} | {d['median']:.3f} | {d['within10']} ({100*d['within10']/d['shapes']:.0f} %) | {d['above10']} | {d['above50']} | {d['below10']} |")
    L.append("")
    L.append("## Lower-bound check against measured latencies in the comparison CSV\n")
    if not (has_ref_impl or has_baseline):
        L.append("* n/a: the comparison CSV has no `reference_latency_ms` / `optimized_baseline_latency_ms` columns")
    if has_ref_impl:
        L.append(f"* SOL above `reference_latency_ms`: **{len(viol_ref)}** shapes")
    if has_baseline:
        L.append(f"* SOL above `optimized_baseline_latency_ms`: **{len(viol_base)}** shapes")
    for r in sorted(viol_base, key=lambda r: -(r["sol"] / r["baseline"]))[:30]:
        L.append(f"  * {r['subset']}/{r['artifact_id']} {r['workload_uuid'][:8]}: SOL {r['sol']*1e3:.1f} us vs baseline {r['baseline']*1e3:.1f} us")
    L.append("")
    L.append("## Confidence of the SOL (per shape)\n")
    L.append("| level | shapes | meaning |\n|---|---|---|")
    L.append(f"| high | {conf.get('high', 0)} | static dense/streaming kernels; the traced shapes determine the work exactly |")
    L.append(f"| medium | {conf.get('medium', 0)} | structural sparsity (triangular / masked operands, broadcast or gathered-output shortcuts) discounted statically |")
    L.append(f"| low | {conf.get('low', 0)} | data-dependent behaviour (index/mask inputs, gathers, scatters, top-k) or in-kernel quantization; the trace fixes one input instance |")
    L.append("")
    L.append(f"## Problems with median above the comparison SOL by more than 10 % ({len(above)} of {len(problems)})\n")
    L.append("| subset | problem | shapes | median | min | max | bottleneck | precision | confidence | cause |\n|---|---|---|---|---|---|---|---|---|---|")
    for t in above:
        L.append(f"| {t['subset']} | {t['problem']} | {t['shapes']} | {t['median_ratio']:.2f} | {t['min_ratio']:.2f} | {t['max_ratio']:.2f} | {t['bottleneck']} | {t['precision']} | {t['confidence']} | {t['cause']} |")
    L.append("")
    L.append(f"## Problems with median below the comparison SOL by more than 10 % ({len(below)})\n")
    L.append("| subset | problem | shapes | median | min | max | bottleneck | precision | confidence | cause |\n|---|---|---|---|---|---|---|---|---|---|")
    for t in below:
        L.append(f"| {t['subset']} | {t['problem']} | {t['shapes']} | {t['median_ratio']:.2f} | {t['min_ratio']:.2f} | {t['max_ratio']:.2f} | {t['bottleneck']} | {t['precision']} | {t['confidence']} | {t['cause']} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(L) + "\n")

    # ---- report_data.json ---------------------------------------------------
    points = [{
        "subset": r["subset"], "problem": r["artifact_id"], "uuid": r["workload_uuid"][:8],
        "sol_us": r["sol"] * 1e3, "ref_sol_us": r["ref_sol"] * 1e3,
        "measured_us": None if math.isnan(r["measured"]) else r["measured"] * 1e3,
        "baseline_us": None if math.isnan(r["baseline"]) else r["baseline"] * 1e3,
        "ref_impl_us": None if math.isnan(r["ref_impl"]) else r["ref_impl"] * 1e3,
        "confidence": r.get("solar_confidence") or "n/a", "precision": r["solar_precision"],
        "bottleneck": r["solar_bottleneck"],
    } for r in have]
    data = {
        "label": args.label, "shapes": n, "shapes_total": len(rows),
        "median_ratio": st.median(r["best"] for r in have),
        "buckets": [{"name": name, "shapes": frac(c)[0], "pct": frac(c)[1]} for name, c in buckets],
        "per_subset": per_subset,
        "violations": {"reference_impl": len(viol_ref) if has_ref_impl else None,
                       "optimized_baseline": len(viol_base) if has_baseline else None},
        "confidence": dict(conf),
        "problems": problems, "points": points,
    }
    (out_dir / "report_data.json").write_text(json.dumps(data))
    print(f"shapes {n}; median {data['median_ratio']:.3f}; within10 {frac(buckets[0][1])[0]}; "
          f"violations ref={len(viol_ref) if has_ref_impl else 'n/a'} base={len(viol_base) if has_baseline else 'n/a'}; confidence {dict(conf)}; "
          f"above>10% {len(above)} problems; wrote {out_dir/'SUMMARY.md'} and report_data.json")


if __name__ == "__main__":
    main()
