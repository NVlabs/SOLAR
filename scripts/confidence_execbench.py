#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confidence level of each SOL-ExecBench SOL, from the patterns in its graph.

Solar's SOL is a static lower bound. Three patterns make it less certain:

* **dynamic** (confidence *low*): data-dependent behaviour the trace cannot
  resolve exactly: index/mask inputs (int32/int64/bool tensors), gathers,
  scatters, embedding lookups, top-k / sort / nonzero, tensor-indexed
  ``__getitem__``. The traced shapes fix how many rows are touched for one
  input instance; another instance may touch more or fewer.
* **quantization** (confidence *low*): quant problems whose reference computes
  scales and casts to fp8 / fp4 inside the kernel (``amax``/``clamp``/``round``
  chains, casts to float8/uint8). The quantizer's compute and the FP4 packing
  conventions are modelled, not measured.
* **structural sparsity** (confidence *medium*): triangular / masked operands
  (``tril``/``triu``, additive -inf masks) whose MACs are discounted
  statically, and broadcast / gathered-output shortcuts.

Everything else is *high*. A shape takes the lowest level of any pattern it
contains. Output: one CSV row per shape with the level and the triggers.

    python scripts/confidence_execbench.py --results-root out/a --results-root out/b -o out/b/confidence.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_ROOT = ROOT.parent / "SOL-ExecBench" / "data" / "benchmark"

DYNAMIC_OPS = {
    "index_select", "gather", "take", "take_along_dim", "embedding", "embedding_bag",
    "index_add", "index_add_", "scatter", "scatter_", "scatter_add", "scatter_add_",
    "scatter_reduce", "scatter_reduce_", "index_copy", "index_copy_", "index_put",
    "index_put_", "masked_scatter", "masked_scatter_", "masked_select", "topk",
    "nonzero", "argsort", "sort", "bucketize", "searchsorted", "unique", "bincount",
}
SPARSE_OPS = {"tril", "triu"}
QUANT_SIGNATURE = {"amax", "clamp", "round", "floor"}
INDEX_DTYPES = {"int64", "int32", "int16", "bool", "uint8"}


def classify(problem: str, definition: dict | None, summary: dict, layers: dict, analysis_layers: dict):
    reasons = []
    level = "high"
    types = {str(v.get("type", "")).lower() for v in layers.values() if isinstance(v, dict)}

    # --- dynamic / data dependent -----------------------------------------
    dyn = sorted(types & DYNAMIC_OPS)
    if definition:
        ins = definition.get("inputs") or {}
        items = ins.items() if isinstance(ins, dict) else ((i.get("name"), i) for i in ins)
        idx_inputs = [n for n, sp in items if sp.get("shape") is not None and str(sp.get("dtype")) in INDEX_DTYPES]
        if idx_inputs:
            reasons.append("index/mask inputs: " + ",".join(idx_inputs[:4]))
    for lid, v in layers.items():
        if not isinstance(v, dict) or str(v.get("type", "")).lower() != "__getitem__":
            continue
        ins = (v.get("tensor_names") or {}).get("inputs") or []
        dts = (v.get("tensor_dtypes") or {}).get("inputs") or []
        if len(ins) >= 2 or any(str(d).replace("torch.", "") in INDEX_DTYPES for d in dts[1:]):
            dyn.append("tensor-indexed __getitem__")
            break
    if dyn:
        reasons.append("data-dependent ops: " + ",".join(sorted(set(dyn))[:6]))
    if dyn or any(r.startswith("index/mask inputs") for r in reasons):
        level = "low"

    # --- quantization overhead --------------------------------------------
    quant = summary.get("quant_dtypes") or []
    if quant:
        sig = sorted(types & QUANT_SIGNATURE)
        narrow_casts = any(
            any(("float8" in str(d) or "uint8" in str(d) or "float4" in str(d))
                for d in ((v.get("tensor_dtypes") or {}).get("outputs") or []))
            for v in layers.values() if isinstance(v, dict) and str(v.get("type", "")).lower() == "to")
        if sig or narrow_casts:
            reasons.append("in-kernel quantization (" + ",".join(quant) + "; " + ",".join(sig or ["narrow casts"]) + ")")
            level = "low"

    # --- structural sparsity ----------------------------------------------
    sparse = sorted(types & SPARSE_OPS)
    fr = [l.get("mac_sparsity_fraction", 1.0) for l in analysis_layers.values() if isinstance(l, dict)]
    n_disc = sum(1 for f in fr if f is not None and f < 1.0)
    if sparse or n_disc:
        reasons.append(f"structural sparsity ({','.join(sparse) or 'masked'}; {n_disc} einsums discounted)")
        if level == "high":
            level = "medium"
    return level, "; ".join(reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, action="append", required=True,
                    help="Result roots; later roots override earlier ones for the same (problem, uuid)")
    ap.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    latest = {}
    for root in args.results_root:
        for f in glob.glob(f"{root}/*/*/sol_summary.json"):
            d = json.loads(Path(f).read_text())
            latest[(d["problem"], d["workload_uuid"])] = Path(f).parent
    defs = {}
    rows = []
    for (problem, uuid), d in sorted(latest.items()):
        if problem not in defs:
            hits = list(args.bench_root.glob(f"*/{problem}/definition.json"))
            defs[problem] = json.loads(hits[0].read_text()) if hits else None
        summary = json.loads((d / "sol_summary.json").read_text())
        eg = d / "einsum" / "einsum_graph_renamed.yaml"
        an = d / "analysis" / "analysis.yaml"
        try:
            layers = yaml.load(eg.read_text(), Loader=yaml.CSafeLoader).get("layers") or {}
        except Exception:
            layers = {}
        try:
            analysis_layers = yaml.load(an.read_text(), Loader=yaml.CSafeLoader).get("layers") or {}
        except Exception:
            analysis_layers = {}
        level, why = classify(problem, defs[problem], summary, layers, analysis_layers)
        rows.append({"artifact_id": problem, "workload_uuid": uuid, "confidence": level, "confidence_reason": why})
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} shapes classified", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["artifact_id", "workload_uuid", "confidence", "confidence_reason"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    c = Counter(r["confidence"] for r in rows)
    print(f"{len(rows)} shapes: " + ", ".join(f"{k}={c[k]}" for k in ("high", "medium", "low")) + f" -> {args.out}")


if __name__ == "__main__":
    main()
