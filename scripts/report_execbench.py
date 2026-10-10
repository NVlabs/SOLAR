#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summaries of a SOL-ExecBench comparison CSV (from export_execbench_csv.py).

Writes next to the CSV:
* ``SUMMARY.md``                 shape-level buckets, per-subset agreement, problems above/below the
                                 leaderboard SOL table with causes, and the lower-bound check against
                                 the measured implementations (reference impl, optimized baseline);
* ``report_data.json``           the same numbers plus per-shape points for plotting.

    python scripts/report_execbench.py out/execbench_final/sol_latencies_compare.csv --label "final"
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics as st
from pathlib import Path

# Hand-checked causes for problems whose median sits away from the leaderboard SOL table.
NOTES = {
    "023_fp8_mamba2_ssm_discretization": "reference row is 38 ns, below any launch floor (reference error)",
    "066_masked_softmax_with_attention_dropout_backward": "232 MB of declared fp32 I/O cannot take the reference's 5 us (reference error)",
    "081_joint_attention_context_projection": "reference counts half the MACs of the (4096+77) x 2432 x 4864 projection",
    "025_moe_expert_parallel_execution_backward": "weights read + grads written at 4 B = 51.6 GB, exactly the hand count; reference used 2 B",
    "021_cross_attention_text_video_conditioning_backward": "reference MAC undercount on the 10-output backward; fp32 memory at 4 B",
    "007_hyena_fft_size_padding_rfft": "complex64 rfft output priced at 8 B; reference 2 B (1 us problem)",
    "008_nvfp4_multimodal_embedding_projection": "boolean-mask indexing on the meta trace yields an upper-bound row count (known gap)",
    "026_nvfp4_mamba2_out_projection": "reference counts a packed FP4 pair as one MAC / 0.5 B",
    "013_fused_residual_rms_norm_backward": "reference counts fewer input bytes than the declared fp32 inputs",
    "014_fp8_yarn_rope_embedding": "reference rows at 1e-6 ms level; fp32 cos/sin tables priced at 4 B",
    "011_fp8_moe_gate_routing": "reference prices the bf16 hidden states at 1 B (fp8) although the input is declared bf16",
    "087_embedding_with_initial_layernorm_backward": "sparse index_add_ into the grad table; only touched rows are written",
    "031_flux_timestep_guidance_projection_embedding": "57 MB of fp32 F.linear weights (plain-tensor operands) now read; reference omitted them",
    "086_sam_hq_mask_decoder_iou_hypernetwork_fusion": "F.linear weights on plain tensors now read; reference omitted them",
    "036_flux_output_norm_projection_chain": "F.linear weights on plain tensors now read; reference omitted them",
    "045_audio_encoder_to_language_model_multimodal_fusion": "projector computed only for the gathered audio-token rows",
    "043_mamba_chunk_scan_with_segsum": "tril-masked einsums halved; C*B^T computed once per group",
    "060_chunk_gated_delta_rule_linear_attention": "reference counts dense triangular chunk products",
    "020_nvfp4_linear_layer": "reference counts a packed FP4 pair as one MAC",
    "029_nvfp4_fused_mlp_silu_gating": "reference counts a packed FP4 pair as one MAC",
    "018_nvfp4_attention_output_projection_with_residual": "reference counts a packed FP4 pair as one MAC",
    "024_nvfp4_attention_output_projection": "reference counts a packed FP4 pair as one MAC",
}


def best_ratio(r):
    rf = max(float(r["ratio_fused_over_ref"]), 1e-9)
    rff = max(float(r["ratio_fused_plus_floor_over_ref"]), 1e-9)
    return rff if abs(math.log(rff)) <= abs(math.log(rf)) else rf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare_csv", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
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
    L.append(f"Shapes with a SOL: {n} of {len(rows)}. Ratio = Solar fused SOL / leaderboard SOL, whichever of (with, without the 0.4 us launch floor) is closer to 1.\n")
    L.append("## Agreement with the leaderboard SOL table\n")
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
    L.append("## Lower-bound check against the measured implementations\n")
    L.append(f"* SOL above the measured reference implementation: **{len(viol_ref)}** shapes")
    L.append(f"* SOL above the measured optimized baseline: **{len(viol_base)}** shapes")
    for r in sorted(viol_base, key=lambda r: -(r["sol"] / r["baseline"]))[:30]:
        L.append(f"  * {r['subset']}/{r['artifact_id']} {r['workload_uuid'][:8]}: SOL {r['sol']*1e3:.1f} us vs baseline {r['baseline']*1e3:.1f} us")
    L.append("")
    L.append("## Confidence of the SOL (per shape)\n")
    L.append("| level | shapes | meaning |\n|---|---|---|")
    L.append(f"| high | {conf.get('high', 0)} | static dense/streaming kernels; the traced shapes determine the work exactly |")
    L.append(f"| medium | {conf.get('medium', 0)} | structural sparsity (triangular / masked operands, broadcast or gathered-output shortcuts) discounted statically |")
    L.append(f"| low | {conf.get('low', 0)} | data-dependent behaviour (index/mask inputs, gathers, scatters, top-k) or in-kernel quantization; the trace fixes one input instance |")
    L.append("")
    L.append(f"## Problems with median above the leaderboard SOL by more than 10 % ({len(above)} of {len(problems)})\n")
    L.append("| subset | problem | shapes | median | min | max | bottleneck | precision | confidence | cause |\n|---|---|---|---|---|---|---|---|---|---|")
    for t in above:
        L.append(f"| {t['subset']} | {t['problem']} | {t['shapes']} | {t['median_ratio']:.2f} | {t['min_ratio']:.2f} | {t['max_ratio']:.2f} | {t['bottleneck']} | {t['precision']} | {t['confidence']} | {t['cause']} |")
    L.append("")
    L.append(f"## Problems with median below the leaderboard SOL by more than 10 % ({len(below)})\n")
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
        "violations": {"reference_impl": len(viol_ref), "optimized_baseline": len(viol_base)},
        "confidence": dict(conf),
        "problems": problems, "points": points,
    }
    (out_dir / "report_data.json").write_text(json.dumps(data))
    print(f"shapes {n}; median {data['median_ratio']:.3f}; within10 {frac(buckets[0][1])[0]}; "
          f"violations ref={len(viol_ref)} base={len(viol_base)}; confidence {dict(conf)}; "
          f"above>10% {len(above)} problems; wrote {out_dir/'SUMMARY.md'} and report_data.json")


if __name__ == "__main__":
    main()
