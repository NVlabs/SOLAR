#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run run_execbench_problem.py over every SOL-ExecBench problem (one workload each)
and write a pass/fail table to out/execbench/sweep_<ts>.{jsonl,md}."""
import argparse, json, resource, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT.parent / "SOL-ExecBench" / "data" / "benchmark"

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", nargs="+", default=["L1", "L2", "Quant", "FlashInfer-Bench"])
ap.add_argument("--workload-index", type=int, default=0)
ap.add_argument("--arch-config", default="B200")
ap.add_argument("--timeout", type=int, default=900)
ap.add_argument("--limit", type=int)
ap.add_argument("--resume", type=Path, help="Skip problems already marked ok in a previous sweep .jsonl")
ap.add_argument("--only-failed", type=Path, help="Re-run only the problems marked failed in a previous sweep .jsonl")
ap.add_argument("--mem-gb", type=float, default=40.0,
                help="Per-problem virtual address-space cap (RLIMIT_AS) in GB; 0 disables. "
                     "Keeps one runaway trace from taking down the whole machine.")
args = ap.parse_args()

def _limit_mem():
    if args.mem_gb > 0:
        cap = int(args.mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = ROOT / "out" / "execbench"; out_dir.mkdir(parents=True, exist_ok=True)
jsonl = out_dir / f"sweep_{ts}.jsonl"
rows = []
for subset in args.subsets:
    probs = sorted(p for p in (BENCH / subset).iterdir() if (p / "definition.json").exists())
    if args.resume:
        done = {(r["subset"], r["problem"]) for r in map(json.loads, filter(str.strip, args.resume.read_text().splitlines())) if r["ok"]}
        probs = [p for p in probs if (subset, p.name) not in done]
    if args.only_failed:
        failed = {(json.loads(l)["subset"], json.loads(l)["problem"]) for l in args.only_failed.read_text().splitlines() if l.strip() and not json.loads(l)["ok"]}
        probs = [p for p in probs if (subset, p.name) in failed]
    if args.limit: probs = probs[: args.limit]
    for p in probs:
        t0 = time.time()
        cmd = [sys.executable, str(ROOT / "scripts/run_execbench_problem.py"), str(p),
               "--workload-index", str(args.workload_index), "--arch-config", args.arch_config]
        try:
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout,
                               preexec_fn=_limit_mem)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            def _s(b): return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
            rc, out, err = -9, _s(e.stdout), f"TIMEOUT after {args.timeout}s"
        if rc in (-9, 137) and not err.startswith("TIMEOUT"):
            err += " | KILLED (likely exceeded --mem-gb cap or OOM)"
        row = {"subset": subset, "problem": p.name, "ok": rc == 0, "rc": rc, "secs": round(time.time() - t0, 1)}
        if rc == 0:
            summ = None
            for line in out.splitlines():
                if line.startswith("Summary written :"):
                    summ = json.loads(Path(line.split(":", 1)[1].strip()).read_text())
            if summ:
                row.update(sol_fused_ms=summ["sol_ms"]["fused"], sol_unfused_ms=summ["sol_ms"]["unfused"],
                           macs=summ["total_macs"], precision=summ["precision"])
        else:
            tail = [l for l in (out + "\n" + err).splitlines() if l.strip()]
            row["error"] = " | ".join(tail[-6:])[-600:]
        rows.append(row)
        with open(jsonl, "a") as f: f.write(json.dumps(row) + "\n")
        print(f"[{'OK ' if row['ok'] else 'FAIL'}] {subset}/{p.name} ({row['secs']}s)" + ("" if row["ok"] else f"\n      {row['error'][-300:]}"), flush=True)

n_ok = sum(r["ok"] for r in rows)
print(f"\n{n_ok}/{len(rows)} passed. Details: {jsonl}")
