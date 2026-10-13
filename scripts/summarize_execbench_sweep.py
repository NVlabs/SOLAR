#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge sweep_*.jsonl files (later files override earlier) and classify failures."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

files = [Path(a) for a in sys.argv[1:]] or sorted((Path(__file__).resolve().parent.parent / "out/execbench").glob("sweep_*.jsonl"))
rows = {}
for f in files:
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line); rows[(r["subset"], r["problem"])] = r

def classify(err: str) -> str:
    e = err or ""
    if "no tensor inputs" in e: return "scalars-only problem (nothing for SOLAR to trace)"
    if "TIMEOUT" in e: return "timeout (long sequential Python loop in reference)"
    m = re.search(r"tensor 'Model_(where|topk|split|chunk|unbind)[^']*' has inconsistent", e)
    if m: return f"SOLAR einsum: multi-output op '{m.group(1)}' rank inconsistency"
    if "has inconsistent rank tuples" in e: return "SOLAR einsum: rank inconsistency on intermediate (strided slice / view)"
    if "producer was mis-attributed" in e or "Conflicts:" in e: return "SOLAR stage-1: producer mis-attribution (shape conflict)"
    if "einsum_graph has no layers" in e: return "SOLAR einsum: no traceable layers"
    if "Failed to run torchgraph" in e: return "torchview trace failed (see per-problem log)"
    return "other: " + e[-160:]

by_subset = defaultdict(lambda: [0, 0])
buckets = defaultdict(list)
for (subset, prob), r in sorted(rows.items()):
    by_subset[subset][0 if r["ok"] else 1] += 1
    if not r["ok"]:
        buckets[classify(r.get("error", ""))].append(f"{subset}/{prob}")

tot_ok = sum(v[0] for v in by_subset.values()); tot = sum(sum(v) for v in by_subset.values())
print(f"Overall: {tot_ok}/{tot} passed\n")
print(f"{'subset':<18}{'passed':>8}{'failed':>8}")
for s, (ok, bad) in by_subset.items(): print(f"{s:<18}{ok:>8}{bad:>8}")
print("\nFailure classes:")
for cls, probs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"\n[{len(probs)}] {cls}")
    for p in probs: print(f"    {p}")
