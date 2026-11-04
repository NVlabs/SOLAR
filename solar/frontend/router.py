# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-based frontend route decision: mechanical vs agentic.

``decide_from_graph`` inspects an extracted ``pytorch_graph.yaml`` document
and routes it ``mechanical`` ONLY on positive evidence; anything it cannot
positively support routes ``agentic`` (execution-validated DSL translation —
Pythonic or pure-skill, both remain available) with recorded reasons.  A
registered/known operator name alone is NOT sufficient, and neither is a
plausible operand count (round-4 P2): positive routing requires

* schema/connectivity integrity — every input/output edge resolves to a node
  that exists in the graph, operand shape lists are well-formed positive-int
  shapes (0-d scalars allowed), and the edge count equals the recorded
  operand shape count (repeated operands appear once per occurrence in both);
* bounded per-op semantic evidence — matmul contraction and output dims must
  agree, add/mul operands must be broadcast-compatible with the recorded
  output, relu/softmax must preserve shape, reshape-family ops must preserve
  element count and transpose/permute the dim multiset.

The capability set is BOUNDED and explicit (``SUPPORTED_OPS`` /
``STRUCTURAL_OPS``); it describes what the mechanical lowering has evidence
for today, not universal support.  Decisions are deterministic, offline, and
record their evidence so it can be carried into results; no credentials are
read and no network or paid API is touched.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Bounded initial mechanical capability set: ops with regression-backed
# operand accounting in this checkout. Extending it requires evidence
# (tests), not registry presence.
SUPPORTED_OPS = frozenset({
    "matmul", "linear", "add", "mul", "sum", "softmax", "relu",
})
# Shape plumbing that carries no arithmetic accounting of its own — still
# held to connectivity and shape-preservation checks below.
STRUCTURAL_OPS = frozenset({
    "view", "reshape", "transpose", "permute", "contiguous", "flatten",
})
_TENSOR_NODE_TYPES = frozenset({
    "input-tensor", "output-tensor", "auxiliary-tensor", "hidden-tensor",
    "parameter-tensor",
})
# ops that mathematically require at least two operands
_MIN_OPERANDS = {"matmul": 2, "add": 2, "mul": 2, "linear": 2}
_ELEMENT_PRESERVING = frozenset({"view", "reshape", "flatten", "contiguous"})
_DIM_PRESERVING = frozenset({"transpose", "permute"})


@dataclass(frozen=True)
class RouteDecision:
    route: str                      # mechanical | agentic | unsupported
    reasons: tuple                  # why (always non-empty)
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"route": self.route, "reasons": list(self.reasons),
             "evidence": self.evidence}, indent=1)


def _valid_shape(shape) -> bool:
    """A well-formed recorded shape: ints >= 1; [] is a 0-d scalar."""
    return (isinstance(shape, (list, tuple)) and
            all(isinstance(d, int) and not isinstance(d, bool) and d >= 1
                for d in shape))


def _broadcast(shapes):
    """Numpy-style broadcast of recorded shapes, or None if incompatible."""
    out = ()
    for shape in shapes:
        merged = []
        a, b = list(out), list(shape)
        for i in range(1, max(len(a), len(b)) + 1):
            da = a[-i] if i <= len(a) else 1
            db = b[-i] if i <= len(b) else 1
            if da != db and 1 not in (da, db):
                return None
            merged.append(max(da, db))
        out = tuple(reversed(merged))
    return out


def _matmul_expected(a, b):
    """The torch.matmul output shape for operands a @ b, or None if the
    operands are invalid (contraction or batch-broadcast mismatch)."""
    a, b = tuple(a), tuple(b)
    if not a or not b:
        return None                      # 0-d operands: not a contraction
    if len(a) == 1 and len(b) == 1:
        return () if a[0] == b[0] else None
    if len(a) == 1:                      # (k) @ (..., k, n) -> (..., n)
        if a[0] != b[-2]:
            return None
        return b[:-2] + (b[-1],)
    if len(b) == 1:                      # (..., m, k) @ (k) -> (..., m)
        if a[-1] != b[0]:
            return None
        return a[:-1]
    if a[-1] != b[-2]:
        return None
    batch = _broadcast([a[:-2], b[:-2]])
    if batch is None:
        return None                      # batch dims do not broadcast
    return tuple(batch) + (a[-2], b[-1])


def _sum_output_ok(inp, out):
    """out must be a legal torch.sum result shape for input inp: full
    reduction, keepdim (reduced dims -> 1), or reduced dims removed (an
    order-preserving subsequence)."""
    inp, out = tuple(inp), tuple(out)
    if out == ():
        return True
    if len(out) > len(inp):
        return False
    if len(out) == len(inp):
        return all(o == i or o == 1 for o, i in zip(out, inp))
    it = iter(inp)
    return all(any(o == dim for dim in it) for o in out)


def _check_semantics(op, name, shapes, out_shapes, input_types):
    """Bounded per-op semantic evidence; returns a list of defect reasons."""
    reasons = []
    out = tuple(out_shapes[0]) if out_shapes else None
    if op == "matmul":
        if len(shapes) != 2:
            return [f"unsupported matmul form ({name}): "
                    f"{len(shapes)} operands"]
        a, b = (tuple(s) for s in shapes)
        expected = _matmul_expected(a, b)
        if expected is None:
            reasons.append(
                f"invalid matmul operands ({name}): {list(a)} @ {list(b)} "
                f"(contraction or batch broadcast mismatch)")
        elif out is not None and out != expected:
            reasons.append(
                f"matmul output shape inconsistent with operands ({name}): "
                f"{list(out)} recorded, {list(expected)} expected from "
                f"{list(a)} @ {list(b)}")
    elif op == "linear":
        # Explicit orientation contract (torch.nn.functional.linear):
        # input (..., in_f) @ weight (out_f, in_f) [+ bias (out_f,)].
        # Every recorded operand must be used exactly once under an
        # unambiguous role assignment — extra, duplicate or uncovered
        # operands route agentic, never silently drop (round-6 P2).
        n = len(shapes)
        if n not in (2, 3):
            return [f"unsupported linear form ({name}): {n} operands "
                    f"(supported: input, weight, optional bias)"]
        types = list(input_types or [])
        if types and len(types) != n:
            return [f"linear operand role tags do not cover its operands "
                    f"({name}): {n} operands, {len(types)} tags"]
        if types:
            w_idx = [i for i, tag in enumerate(types) if tag == "weight"]
            b_idx = [i for i, tag in enumerate(types) if tag == "bias"]
            x_idx = [i for i in range(n) if i not in w_idx and i not in b_idx]
            if len(w_idx) != 1 or len(b_idx) > 1 or len(x_idx) != 1:
                return [f"ambiguous/unsupported linear operand roles "
                        f"({name}): {types} — need exactly one input, one "
                        f"weight and at most one bias, each operand used "
                        f"exactly once"]
            x = tuple(shapes[x_idx[0]])
            w = tuple(shapes[w_idx[0]])
            bias = tuple(shapes[b_idx[0]]) if b_idx else None
        else:
            x, w = tuple(shapes[0]), tuple(shapes[1])
            bias = tuple(shapes[2]) if n == 3 else None
        if len(w) != 2 or not x or x[-1] != w[1]:
            reasons.append(
                f"invalid linear contraction ({name}): input {list(x)} "
                f"needs weight (out_features, {x[-1] if x else '?'}), "
                f"got {list(w)}")
        elif bias is not None and bias != (w[0],):
            reasons.append(
                f"invalid linear bias ({name}): {list(bias)} does not match "
                f"out_features {w[0]}")
        elif out is not None and out != x[:-1] + (w[0],):
            reasons.append(
                f"linear output shape inconsistent ({name}): {list(out)} "
                f"recorded, {list(x[:-1] + (w[0],))} expected")
    elif op == "sum":
        if not _sum_output_ok(shapes[0], out if out is not None else ()):
            reasons.append(
                f"'sum' output shape {list(out)} is not a reduction of "
                f"{list(shapes[0])} ({name})")
    elif op in ("add", "mul"):
        bc = _broadcast(shapes)
        if bc is None:
            reasons.append(
                f"'{op}' operands are not broadcast-compatible ({name}): "
                f"{[list(s) for s in shapes]}")
        elif out is not None and out != bc:
            reasons.append(
                f"'{op}' output shape {list(out)} inconsistent with "
                f"broadcast result {list(bc)} ({name})")
    elif op in ("softmax", "relu"):
        if out is not None and out != tuple(shapes[0]):
            reasons.append(
                f"'{op}' must preserve its operand shape ({name}): "
                f"{list(shapes[0])} -> {list(out)}")
    return reasons


def decide_from_graph(graph: dict) -> RouteDecision:
    layers = (graph or {}).get("layers") or {}
    compute = {name: node for name, node in layers.items()
               if str(node.get("type", "")).lower() not in _TENSOR_NODE_TYPES}
    if not compute:
        return RouteDecision(
            "unsupported",
            ("no compute nodes extracted: nothing to support mechanically",),
            {"compute_nodes": 0})

    reasons = []
    ops_seen = {}
    for name, node in compute.items():
        op = str(node.get("type", "")).lower()
        ops_seen[op] = ops_seen.get(op, 0) + 1
        if op not in SUPPORTED_OPS and op not in STRUCTURAL_OPS:
            reasons.append(
                f"unknown handler semantics for '{op}' ({name}): not in the "
                f"bounded supported set")
            continue

        shapes = node.get("input_shapes") or []
        out_shapes = node.get("output_shapes") or []
        conns = node.get("connections") or {}
        inputs = conns.get("inputs") or []
        outputs = conns.get("outputs") or []

        # -- schema/connectivity integrity (round-4 P2) --
        bad_shapes = [s for s in list(shapes) + list(out_shapes)
                      if not _valid_shape(s)]
        if bad_shapes:
            reasons.append(
                f"malformed shape entries for '{op}' ({name}): {bad_shapes}")
            continue
        dangling = [e for e in list(inputs) + list(outputs)
                    if e not in layers]
        if dangling:
            reasons.append(
                f"edge of '{op}' ({name}) does not resolve to a graph node: "
                f"{dangling}")
            continue
        if not shapes or not inputs:
            reasons.append(
                f"operand accounting incomplete for '{op}' ({name}): "
                f"{len(shapes)} input shapes, {len(inputs)} input edges")
            continue
        if not out_shapes or not outputs:
            reasons.append(
                f"output accounting incomplete for '{op}' ({name}): "
                f"{len(out_shapes)} output shapes, {len(outputs)} output "
                f"edges")
            continue
        if len(inputs) != len(shapes):
            reasons.append(
                f"operand edge/shape count mismatch for '{op}' ({name}): "
                f"{len(shapes)} operand shapes but {len(inputs)} input "
                f"edges (each operand occurrence must carry its edge)")
            continue
        if len(shapes) < _MIN_OPERANDS.get(op, 1):
            reasons.append(
                f"operand accounting incomplete for '{op}' ({name}): "
                f"{len(shapes)} operands recorded, "
                f"needs >= {_MIN_OPERANDS[op]}")
            continue

        # -- bounded per-op semantic evidence --
        if op in _ELEMENT_PRESERVING:
            if math.prod(shapes[0]) != math.prod(out_shapes[0]):
                reasons.append(
                    f"'{op}' changes element count ({name}): "
                    f"{list(shapes[0])} -> {list(out_shapes[0])}")
        elif op in _DIM_PRESERVING:
            if sorted(shapes[0]) != sorted(out_shapes[0]):
                reasons.append(
                    f"'{op}' output dims are not a permutation of its "
                    f"operand dims ({name}): {list(shapes[0])} -> "
                    f"{list(out_shapes[0])}")
        else:
            reasons.extend(_check_semantics(
                op, name, shapes, out_shapes, node.get("input_types")))

    if reasons:
        return RouteDecision("agentic", tuple(reasons),
                             {"compute_nodes": len(compute), "ops": ops_seen})
    return RouteDecision(
        "mechanical",
        ("all compute nodes are in the bounded supported set with validated "
         "connectivity, complete operand accounting and per-op semantic "
         "evidence",),
        {"compute_nodes": len(compute), "ops": ops_seen,
         "supported_set": sorted(SUPPORTED_OPS),
         "operand_check": "every op node's edges resolve, each operand "
                          "occurrence carries shape+edge, and bounded per-op "
                          "shape semantics hold"})


def decide(graph_path) -> RouteDecision:
    import yaml
    graph = yaml.safe_load(Path(graph_path).read_text())
    return decide_from_graph(graph)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph_path", help="pytorch_graph.yaml to route")
    p.add_argument("--out", help="write the decision JSON here")
    args = p.parse_args(argv)
    decision = decide(args.graph_path)
    text = decision.to_json()
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
