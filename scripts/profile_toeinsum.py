#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample where the einsum conversion spends its time on a large pytorch_graph.yaml.

Runs PyTorchToEinsum.convert in a worker thread and, every ``--interval`` seconds,
records the worker's current Python stack. After ``--duration`` seconds (or when the
conversion finishes) it prints the most frequent innermost frames and the most frequent
frames at any depth inside the ``solar`` package: a poor man's sampling profiler that
does not need py-spy and survives being killed by a timeout.

Usage::

    python scripts/profile_toeinsum.py <pytorch_graph.yaml> --duration 600 --interval 2
"""
from __future__ import annotations

import argparse
import collections
import sys
import tempfile
import threading
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("graph")
ap.add_argument("--duration", type=float, default=600)
ap.add_argument("--interval", type=float, default=2.0)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()

# Import the repo's package, not a pip-installed SOLAR: running a script from
# scripts/ puts that directory first on sys.path, so insert the repo root ahead.
import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum  # noqa: E402
import solar  # noqa: E402
print("profiling package:", os.path.dirname(solar.__file__), flush=True)

out_dir = tempfile.mkdtemp(prefix="solar_prof_")
done = threading.Event()
err: list = []


def work():
    try:
        PyTorchToEinsum().convert(args.graph, out_dir, copy_graph=False)
    except BaseException as e:  # noqa: BLE001
        err.append(repr(e))
    finally:
        done.set()


t = threading.Thread(target=work, daemon=True)
t.start()
innermost = collections.Counter()
anywhere = collections.Counter()
samples = 0
t0 = time.time()
while not done.is_set() and time.time() - t0 < args.duration:
    time.sleep(args.interval)
    frame = sys._current_frames().get(t.ident)
    if frame is None:
        continue
    stack = traceback.extract_stack(frame)
    solar_frames = [f for f in stack if "/solar/" in f.filename]
    if not solar_frames:
        continue
    samples += 1
    f = solar_frames[-1]
    innermost[f"{f.filename.split('/solar/')[-1]}:{f.lineno} {f.name}"] += 1
    seen = set()
    for f in solar_frames:
        key = f"{f.filename.split('/solar/')[-1]} {f.name}"
        if key not in seen:
            anywhere[key] += 1
            seen.add(key)

elapsed = time.time() - t0
print(f"{'finished' if done.is_set() else 'still running'} after {elapsed:.0f}s, {samples} samples"
      + (f", error: {err[0]}" if err else ""))
print("\nInnermost solar frame (share of samples):")
for k, v in innermost.most_common(args.top):
    print(f"  {100 * v / max(samples, 1):5.1f}%  {k}")
print("\nFunction on stack at any depth (share of samples):")
for k, v in anywhere.most_common(args.top):
    print(f"  {100 * v / max(samples, 1):5.1f}%  {k}")
