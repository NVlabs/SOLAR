# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check structural and einsum equivalence between two einsum_graph.yaml files.

Usage:
    python3 -m solar.common.check_einsum_graph_equivalence graph_a.yaml graph_b.yaml

Two graphs are equivalent when:
  1. They have the same BFS-topological structure (same node count and connectivity pattern).
  2. Each paired node has the same tensor shapes (inputs and outputs).
  3. Each paired node's einsum equation is canonically equivalent — i.e. identical
     after renaming all dimension labels to A, B, C, ... in first-appearance order.

Nodes are matched by BFS traversal order starting from 'start', so node names
and dimension label names may differ freely between the two graphs.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import yaml
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> dict:
    """Load an einsum_graph.yaml and return the raw dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "layers" not in data:
        raise ValueError(f"No 'layers' key found in {path}")
    return data


# ---------------------------------------------------------------------------
# Einsum canonicalisation
# ---------------------------------------------------------------------------

def _parse_tokens(operand_str: str) -> list:
    """Parse an einsum operand string into tokens ([A-Z][0-9]* or parenthesized groups).

    Mirrors the logic of solar.common.utils.parse_dim_tokens without the
    duplicate-rank validation so we don't need to import the full module.
    """
    tokens = []
    i = 0
    s = operand_str.strip()
    while i < len(s):
        if s[i] == "(":
            j = i + 1
            depth = 1
            while j < len(s) and depth > 0:
                depth += s[j] == "("
                depth -= s[j] == ")"
                j += 1
            tokens.append(s[i:j].upper())
            i = j
            continue
        if not s[i].isalpha():
            i += 1
            continue
        letter = s[i].upper()
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        tokens.append(letter + s[i:j])
        i = j
    return tokens


def canonical_equation(equation: str) -> str | None:
    """Return a canonical form of an einsum equation by relabelling dims.

    Dims are renamed A, B, C, ... (then A0, A1, ...) in the order they first
    appear, scanning input operands left-to-right then the output.

    Examples:
        "B0B1K,NK->B0B1N"  →  "ABC,DC->ABD"
        "BSH,H2H->BSH2"    →  "ABC,DC->ABD"   (same canonical form — equivalent!)
        "->ABC"            →  "->ABC"
        None or ""         →  None
    """
    if not equation:
        return None

    if "->" not in equation:
        return None

    lhs, rhs = equation.split("->", 1)

    # Parse all operands
    input_operands = [_parse_tokens(op) for op in lhs.split(",") if op.strip()]
    output_tokens = _parse_tokens(rhs)

    # Build renaming map in first-appearance order
    all_tokens_in_order: list = []
    for operand in input_operands:
        for t in operand:
            if t not in all_tokens_in_order:
                all_tokens_in_order.append(t)
    for t in output_tokens:
        if t not in all_tokens_in_order:
            all_tokens_in_order.append(t)

    # Generate short canonical names: A, B, C, ..., Z, A0, A1, ...
    def _nth_name(n: int) -> str:
        import string
        alpha = string.ascii_uppercase
        if n < 26:
            return alpha[n]
        return alpha[(n - 26) % 26] + str((n - 26) // 26)

    rename = {tok: _nth_name(i) for i, tok in enumerate(all_tokens_in_order)}

    def _rewrite(tokens: list) -> str:
        return "".join(rename[t] for t in tokens)

    input_parts = [_rewrite(op) for op in input_operands]
    output_part = _rewrite(output_tokens)
    return ",".join(input_parts) + "->" + output_part


# ---------------------------------------------------------------------------
# BFS traversal
# ---------------------------------------------------------------------------

def bfs_order(layers: dict) -> list:
    """Return node IDs in BFS order starting from 'start'.

    Follows connections.outputs edges. Nodes with no predecessor other than
    'start' are visited first. Maintains insertion-stable BFS.
    """
    if "start" not in layers:
        raise ValueError("No 'start' node found in graph")

    visited: set = set()
    order: list = []
    queue: deque = deque(["start"])

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        order.append(node_id)

        node = layers.get(node_id, {})
        for succ in node.get("connections", {}).get("outputs", []):
            if succ not in visited and succ in layers:
                queue.append(succ)

    return order


# ---------------------------------------------------------------------------
# Node-level comparison
# ---------------------------------------------------------------------------

def _shapes(node: dict) -> tuple:
    """Return ((input_shapes...), (output_shapes...)) as a hashable tuple."""
    ts = node.get("tensor_shapes", {})
    ins = tuple(tuple(s) for s in ts.get("inputs", []))
    outs = tuple(tuple(s) for s in ts.get("outputs", []))
    return (ins, outs)


class NodeDiff:
    """Record of differences between two matched nodes."""

    def __init__(self, id_a: str, id_b: str):
        self.id_a = id_a
        self.id_b = id_b
        self.issues: list = []

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def __str__(self) -> str:
        header = f"  [{self.id_a}] vs [{self.id_b}]"
        if self.ok:
            return header + " ✓"
        return header + "\n" + "\n".join(f"    - {iss}" for iss in self.issues)


def compare_nodes(id_a: str, node_a: dict, id_b: str, node_b: dict) -> NodeDiff:
    diff = NodeDiff(id_a, id_b)

    # Shapes
    shapes_a = _shapes(node_a)
    shapes_b = _shapes(node_b)
    if shapes_a != shapes_b:
        diff.issues.append(
            f"tensor_shapes differ:\n"
            f"      A: inputs={list(shapes_a[0])} outputs={list(shapes_a[1])}\n"
            f"      B: inputs={list(shapes_b[0])} outputs={list(shapes_b[1])}"
        )

    # Einsum equations (canonical comparison)
    eq_a = node_a.get("einsum_equation")
    eq_b = node_b.get("einsum_equation")
    canon_a = canonical_equation(eq_a) if eq_a else None
    canon_b = canonical_equation(eq_b) if eq_b else None

    if canon_a is None and canon_b is None:
        pass  # both absent — ok for non-einsum nodes
    elif canon_a is None and canon_b is not None:
        diff.issues.append(f"A has no einsum_equation; B has: {eq_b!r} (canonical: {canon_b!r})")
    elif canon_a is not None and canon_b is None:
        diff.issues.append(f"A has einsum_equation: {eq_a!r} (canonical: {canon_a!r}); B has none")
    elif canon_a != canon_b:
        diff.issues.append(
            f"einsum_equations differ (after canonicalisation):\n"
            f"      A: {eq_a!r}  →  canonical: {canon_a!r}\n"
            f"      B: {eq_b!r}  →  canonical: {canon_b!r}"
        )

    # Fan-in / fan-out counts (topology check)
    conn_a = node_a.get("connections", {})
    conn_b = node_b.get("connections", {})
    fanin_a = len(conn_a.get("inputs", []))
    fanin_b = len(conn_b.get("inputs", []))
    fanout_a = len(conn_a.get("outputs", []))
    fanout_b = len(conn_b.get("outputs", []))

    if fanin_a != fanin_b:
        diff.issues.append(f"fan-in differs: A={fanin_a}, B={fanin_b}")
    if fanout_a != fanout_b:
        diff.issues.append(f"fan-out differs: A={fanout_a}, B={fanout_b}")

    return diff


# ---------------------------------------------------------------------------
# Top-level equivalence check
# ---------------------------------------------------------------------------

class EquivalenceReport:
    def __init__(self, path_a: Path, path_b: Path):
        self.path_a = path_a
        self.path_b = path_b
        self.node_diffs: list = []     # list[NodeDiff] for matched pairs
        self.extra_a: list = []        # node IDs in A but not matched
        self.extra_b: list = []        # node IDs in B but not matched
        self.errors: list = []         # fatal load/parse errors

    @property
    def equivalent(self) -> bool:
        return (
            not self.errors
            and not self.extra_a
            and not self.extra_b
            and all(d.ok for d in self.node_diffs)
        )

    def print_report(self, verbose: bool = False) -> None:
        logger.info(f"\nGraph A: {self.path_a}")
        logger.info(f"Graph B: {self.path_b}")
        logger.info("")

        if self.errors:
            for e in self.errors:
                print(f"ERROR: {e}")
            return

        ok_count = sum(1 for d in self.node_diffs if d.ok)
        bad_count = sum(1 for d in self.node_diffs if not d.ok)
        total = len(self.node_diffs)

        logger.info(f"Matched nodes:  {total}  ({ok_count} equivalent, {bad_count} with differences)")
        if self.extra_a:
            logger.info(f"Extra in A:     {self.extra_a}")
        if self.extra_b:
            logger.info(f"Extra in B:     {self.extra_b}")

        if bad_count == 0 and not self.extra_a and not self.extra_b:
            logger.info("\n✓ Graphs are equivalent.")
        else:
            logger.info("\n✗ Graphs differ:")
            for d in self.node_diffs:
                if not d.ok or verbose:
                    logger.debug(str(d))

        logger.info("")


def check_equivalence(
    path_a: str | Path,
    path_b: str | Path,
    verbose: bool = False,
) -> EquivalenceReport:
    """Compare two einsum_graph.yaml files.

    Nodes are matched by BFS traversal order from 'start'. For each matched
    pair the check verifies tensor shapes and canonical einsum equation.
    Extra nodes (present in one graph but not matched) are reported.

    Returns:
        EquivalenceReport with .equivalent bool and .print_report() method.
    """
    report = EquivalenceReport(Path(path_a), Path(path_b))

    try:
        graph_a = load_graph(Path(path_a))
        graph_b = load_graph(Path(path_b))
    except Exception as e:
        report.errors.append(str(e))
        return report

    layers_a = graph_a["layers"]
    layers_b = graph_b["layers"]

    try:
        order_a = bfs_order(layers_a)
        order_b = bfs_order(layers_b)
    except ValueError as e:
        report.errors.append(str(e))
        return report

    # Pair nodes by position in BFS order
    len_a, len_b = len(order_a), len(order_b)
    matched = min(len_a, len_b)

    for i in range(matched):
        id_a, id_b = order_a[i], order_b[i]
        node_a = layers_a[id_a]
        node_b = layers_b[id_b]
        diff = compare_nodes(id_a, node_a, id_b, node_b)
        report.node_diffs.append(diff)

    report.extra_a = order_a[matched:]
    report.extra_b = order_b[matched:]

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check structural and einsum equivalence between two einsum_graph.yaml files."
    )
    parser.add_argument("graph_a", help="Path to first einsum_graph.yaml")
    parser.add_argument("graph_b", help="Path to second einsum_graph.yaml")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print all node comparisons, not just failures",
    )
    args = parser.parse_args()

    report = check_equivalence(args.graph_a, args.graph_b, verbose=args.verbose)
    report.print_report(verbose=args.verbose)

    sys.exit(0 if report.equivalent else 1)


if __name__ == "__main__":
    main()
