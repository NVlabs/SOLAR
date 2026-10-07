# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

module_path = Path(__file__).resolve().parents[1] / "solar/perf/schedule_estimate.py"
spec = importlib.util.spec_from_file_location("schedule_estimate", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
estimate_schedule = module.estimate_schedule


def plan(nodes, stream_order, resources=None):
    return {
        "schema_version": 1,
        "unit": "ns",
        "node_kind": "synthetic_task",
        "duration_source": "synthetic",
        "stream_semantics": "explicit_order_only",
        "nodes": nodes,
        "stream_order": stream_order,
        "resources": resources or [],
    }


def node(name, duration, stream, deps=None, resource=None):
    return {"id": name, "duration_ns": duration, "stream": stream,
            "deps": deps or [], "resource": resource}


def test_serial_and_fork_join():
    serial = plan([node("a", 2, "s"), node("b", 3, "s"), node("c", 5, "s")],
                  {"s": ["a", "b", "c"]})
    assert estimate_schedule(serial)["critical_path_ns"] == 10
    assert estimate_schedule(serial)["makespan_ns"] == 10

    fork = plan([node("a", 2, "s0"), node("b", 3, "s1", ["a"]),
                 node("c", 5, "s2", ["a"]), node("d", 1, "s0", ["b", "c"])],
                {"s0": ["a", "d"], "s1": ["b"], "s2": ["c"]})
    result = estimate_schedule(fork)
    assert (result["critical_path_ns"], result["makespan_ns"]) == (8, 8)
    assert result["per_node"]["d"]["start_ns"] == 7


def test_exclusive_resource_and_stream_order():
    independent = plan([node("a", 3, "s0", resource="gpu"),
                        node("b", 5, "s1", resource="gpu")],
                       {"s0": ["a"], "s1": ["b"]}, ["gpu"])
    result = estimate_schedule(independent)
    assert (result["critical_path_ns"], result["makespan_ns"]) == (5, 8)

    same_stream = plan([node("a", 3, "s"), node("b", 5, "s")],
                       {"s": ["a", "b"]})
    assert estimate_schedule(same_stream)["critical_path_ns"] == 8


def test_empty_zero_duration_and_order_independence():
    assert estimate_schedule(plan([], {}))["makespan_ns"] == 0
    original = plan([node("a", 0, "s0"), node("b", 3, "s1", ["a"])],
                    {"s0": ["a"], "s1": ["b"]})
    reversed_rows = copy.deepcopy(original)
    reversed_rows["nodes"].reverse()
    assert estimate_schedule(original) == estimate_schedule(reversed_rows)
    assert estimate_schedule(original)["makespan_ns"] == 3


@pytest.mark.parametrize("change", [
    lambda p: p["nodes"].append(node("a", 1, "s0")),
    lambda p: p["nodes"][0].update(duration_ns=-1),
    lambda p: p["nodes"][0].pop("duration_ns"),
    lambda p: p["nodes"][0].update(deps=["missing"]),
    lambda p: p["nodes"][0].update(deps=["b"]),
    lambda p: p["stream_order"]["s0"].append("a"),
    lambda p: p["nodes"][0].update(resource="unknown"),
])
def test_invalid_plan_rejected(change):
    base = plan([node("a", 1, "s0"), node("b", 1, "s1", ["a"])],
                {"s0": ["a"], "s1": ["b"]})
    change(base)
    with pytest.raises(ValueError):
        estimate_schedule(base)
