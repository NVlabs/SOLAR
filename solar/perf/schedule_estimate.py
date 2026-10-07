# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in schedule estimate for an explicitly timed task DAG.

Durations are supplied by the caller. Resources are exclusive; this module
neither derives kernel costs from einsums nor predicts CUDA occupancy.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    duration_ns: int
    stream: str
    deps: tuple[str, ...]
    resource: str | None


def estimate_schedule(plan: dict) -> dict:
    """Return critical path and a deterministic exclusive-resource schedule."""
    if (plan.get("schema_version"), plan.get("unit"), plan.get("node_kind"),
            plan.get("duration_source"), plan.get("stream_semantics")) != (
                1, "ns", "synthetic_task", "synthetic", "explicit_order_only"):
        raise ValueError("unsupported schedule schema or cost source")
    rows = plan.get("nodes")
    order = plan.get("stream_order")
    resources = plan.get("resources", [])
    if not isinstance(rows, list) or not isinstance(order, dict):
        raise ValueError("nodes and stream_order are required")
    if not isinstance(resources, list) or len(resources) != len(set(resources)) or any(
            not isinstance(name, str) or not name for name in resources):
        raise ValueError("resources must contain unique names")

    tasks: dict[str, Task] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each node must be an object")
        name, duration, stream = row.get("id"), row.get("duration_ns"), row.get("stream")
        deps, resource = row.get("deps"), row.get("resource")
        if not isinstance(name, str) or not name or name in tasks:
            raise ValueError("node ids must be unique nonempty strings")
        if type(duration) is not int or duration < 0:
            raise ValueError("duration_ns must be a nonnegative integer")
        if not isinstance(stream, str) or not stream:
            raise ValueError("each node needs a stream")
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps) or len(deps) != len(set(deps)):
            raise ValueError("deps must contain unique node ids")
        if resource is not None and resource not in resources:
            raise ValueError(f"unknown resource: {resource}")
        tasks[name] = Task(duration, stream, tuple(deps), resource)

    predecessors = {name: set(task.deps) for name, task in tasks.items()}
    if any(not deps <= tasks.keys() for deps in predecessors.values()):
        raise ValueError("unknown predecessor")
    seen: set[str] = set()
    for stream, names in order.items():
        if not isinstance(stream, str) or not isinstance(names, list):
            raise ValueError("invalid stream_order")
        for index, name in enumerate(names):
            if name not in tasks or name in seen or tasks[name].stream != stream:
                raise ValueError("stream_order must cover each node exactly once")
            seen.add(name)
            if index:
                predecessors[name].add(names[index - 1])
    if seen != tasks.keys():
        raise ValueError("stream_order must cover each node exactly once")

    successors = {name: [] for name in tasks}
    degree = {name: len(deps) for name, deps in predecessors.items()}
    for name, deps in predecessors.items():
        for dep in deps:
            successors[dep].append(name)
    ready = [name for name, count in degree.items() if count == 0]
    heapq.heapify(ready)
    finish_earliest: dict[str, int] = {}
    while ready:
        name = heapq.heappop(ready)
        finish_earliest[name] = max((finish_earliest[dep] for dep in predecessors[name]), default=0) + tasks[name].duration_ns
        for nxt in successors[name]:
            degree[nxt] -= 1
            if degree[nxt] == 0:
                heapq.heappush(ready, nxt)
    if len(finish_earliest) != len(tasks):
        raise ValueError("dependency or stream-order cycle")

    completed: set[str] = set()
    started: set[str] = set()
    busy: set[str] = set()
    running: list[tuple[int, str]] = []
    starts: dict[str, int] = {}
    finishes: dict[str, int] = {}
    now = 0
    while len(completed) < len(tasks):
        while running and running[0][0] == now:
            _, name = heapq.heappop(running)
            completed.add(name)
            if tasks[name].resource is not None:
                busy.remove(tasks[name].resource)
        for name in sorted(tasks):
            task = tasks[name]
            if name in started or not predecessors[name] <= completed or task.resource in busy:
                continue
            started.add(name)
            starts[name] = now
            finishes[name] = now + task.duration_ns
            heapq.heappush(running, (finishes[name], name))
            if task.resource is not None:
                busy.add(task.resource)
        if running:
            now = running[0][0]
        elif len(completed) < len(tasks):
            raise ValueError("unschedulable plan")

    return {
        "schema_version": 1,
        "status": "ok",
        "node_kind": "synthetic_task",
        "duration_source": "synthetic",
        "resource_model": "exclusive",
        "critical_path_ns": max(finish_earliest.values(), default=0),
        "makespan_ns": max(finishes.values(), default=0),
        "per_node": {name: {"start_ns": starts[name], "finish_ns": finishes[name],
                            "stream": tasks[name].stream} for name in sorted(tasks)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(estimate_schedule(json.loads(args.plan.read_text())), indent=2))


if __name__ == "__main__":
    main()
