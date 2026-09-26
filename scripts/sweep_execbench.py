#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run run_execbench_problem.py over every SOL-ExecBench problem (one workload each)
and write a pass/fail table to out/execbench/sweep_<ts>.{jsonl,md}."""
import argparse, json, os, resource, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT.parent / "SOL-ExecBench" / "data" / "benchmark"

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", nargs="+", default=["L1", "L2", "Quant", "FlashInfer-Bench"])
ap.add_argument("--workload-index", type=int, default=0)
ap.add_argument("--all-workloads", action="store_true", help="Run every line of each problem's workload.jsonl, not just --workload-index")
ap.add_argument("--jobs", type=int, default=1, help="Parallel problem runs (each is its own process; size --mem-gb so jobs*mem-gb fits in RAM)")
ap.add_argument("--arch-config", default="B200")
ap.add_argument("--timeout", type=int, default=900)
ap.add_argument("--limit", type=int)
ap.add_argument("--resume", type=Path, help="Skip problems already marked ok in a previous sweep .jsonl")
ap.add_argument("--out-root", type=Path, help="Passed to the runner: artifact root (default SOLAR/out/execbench)")
ap.add_argument("--runner-args", default="", help="Extra arguments appended to every run_execbench_problem.py call, e.g. '--fp32-as fp16'")
ap.add_argument("--only-failed", type=Path, help="Re-run only the problems marked failed in a previous sweep .jsonl")
ap.add_argument("--allow-cuda", action="store_true",
                help="Let child processes see the GPU. Off by default: SOLAR traces on meta/CPU, and "
                     "CUDA initialisation reserves tens of GB of virtual address space that would "
                     "trip the --mem-gb RLIMIT_AS cap on large graphs.")
ap.add_argument("--mem-gb", type=float, default=40.0,
                help="Per-problem virtual address-space cap (RLIMIT_AS) in GB; 0 disables. "
                     "Keeps one runaway trace from taking down the whole machine.")
args = ap.parse_args()

child_env = dict(os.environ)
if not args.allow_cuda:
    child_env["CUDA_VISIBLE_DEVICES"] = ""

def _limit_mem():
    if args.mem_gb > 0:
        cap = int(args.mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = args.out_root or (ROOT / "out" / "execbench"); out_dir.mkdir(parents=True, exist_ok=True)
jsonl = out_dir / f"sweep_{ts}.jsonl"
lock = threading.Lock()

def _read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def _key(r):
    return (r["subset"], r["problem"], int(r.get("workload_index", 0)))

done = set()
if args.resume:
    done = {_key(r) for r in _read_rows(args.resume) if r["ok"]}
failed_keys = None
if args.only_failed:
    failed_keys = {_key(r) for r in _read_rows(args.only_failed) if not r["ok"]}

tasks = []
for subset in args.subsets:
    probs = sorted(p for p in (BENCH / subset).iterdir() if (p / "definition.json").exists())
    if args.limit: probs = probs[: args.limit]
    for p in probs:
        if args.all_workloads:
            n = sum(1 for l in (p / "workload.jsonl").read_text().splitlines() if l.strip())
            idxs = list(range(n))
        else:
            idxs = [args.workload_index]
        for idx in idxs:
            k = (subset, p.name, idx)
            if k in done: continue
            if failed_keys is not None and k not in failed_keys: continue
            tasks.append((subset, p, idx))
print(f"{len(tasks)} runs queued ({args.jobs} parallel, cap {args.mem_gb} GB each, timeout {args.timeout}s)", flush=True)

def run_task(subset, p, idx):
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts/run_execbench_problem.py"), str(p),
           "--workload-index", str(idx), "--arch-config", args.arch_config]
    if args.out_root:
        cmd += ["--out-root", str(args.out_root)]
    if args.runner_args.strip():
        cmd += args.runner_args.split()
    try:
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout,
                           preexec_fn=_limit_mem, env=child_env)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        def _s(b): return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
        rc, out, err = -9, _s(e.stdout), f"TIMEOUT after {args.timeout}s"
    if rc in (-9, 137) and not err.startswith("TIMEOUT"):
        err += " | KILLED (likely exceeded --mem-gb cap or OOM)"
    row = {"subset": subset, "problem": p.name, "workload_index": idx, "ok": rc == 0, "rc": rc,
           "secs": round(time.time() - t0, 1)}
    if rc == 0:
        summ = None
        for line in out.splitlines():
            if line.startswith("Summary written :"):
                summ = json.loads(Path(line.split(":", 1)[1].strip()).read_text())
        if summ:
            row.update(uuid=summ["workload_uuid"], sol_fused_ms=summ["sol_ms"]["fused"],
                       sol_unfused_ms=summ["sol_ms"]["unfused"], macs=summ["total_macs"],
                       precision=summ["precision"])
    else:
        tail = [l for l in (out + "\n" + err).splitlines() if l.strip()]
        row["error"] = " | ".join(tail[-6:])[-600:]
    return row

rows = []
def _record(row):
    with lock:
        rows.append(row)
        with open(jsonl, "a") as f: f.write(json.dumps(row) + "\n")
        tag = f"{row['subset']}/{row['problem']}#{row['workload_index']}"
        print(f"[{'OK ' if row['ok'] else 'FAIL'}] {tag} ({row['secs']}s)"
              + ("" if row["ok"] else f"\n      {row['error'][-300:]}"), flush=True)

if args.jobs <= 1:
    for t in tasks: _record(run_task(*t))
else:
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_task, *t) for t in tasks]
        for fut in as_completed(futs): _record(fut.result())

n_ok = sum(r["ok"] for r in rows)
print(f"\n{n_ok}/{len(rows)} passed. Details: {jsonl}")
