# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Visualize the final AF workload graph (``af_einsum_graph.yaml``) as a PDF.

This is the AccelForge-consumed view of the workload, AFTER the union-find
canonicalization in ``af_graph_builder.py``. It carries:

- a flat list of einsums with canonical-rank ``tensor_accesses``,
- a workload-level ``rank_sizes`` table,
- per-tensor ``bits_per_value`` annotations.

Producer/consumer wiring is implicit in the AF format — a tensor flows from
its (unique) producer einsum (the one whose ``tensor_access`` has
``output: true``) to every consumer einsum that names it. This visualizer
reconstructs that wiring and renders an ops-graph view where:

- each einsum is a node showing its name, copy/real flag, and per-access
  ``(tensor, projection)`` lines,
- each edge is a producer→consumer link labeled with the carried tensor
  and its projection on the consumer side,
- a sidebar node lists ``rank_sizes`` so projections are readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


PathLike = Union[str, Path]


def _proj_str(proj: Any) -> str:
    """Compact rendering of a projection (list or dict)."""
    if isinstance(proj, list):
        return "[" + ", ".join(map(str, proj)) + "]"
    if isinstance(proj, dict):
        return "{" + ", ".join(f"{k}:{v}" for k, v in proj.items()) + "}"
    return str(proj)


class AFGraphVisualizer:
    """Render ``af_einsum_graph.yaml`` as a PDF ops graph."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def save_graph_pdf(
        self,
        af_graph_path: PathLike,
        output_path: PathLike,
        *,
        title: Optional[str] = None,
    ) -> Path:
        """Load an AF workload YAML and render it as PDF.

        Args:
            af_graph_path: Path to ``af_einsum_graph.yaml``.
            output_path: Destination PDF path.
            title: Optional title rendered above the graph.

        Returns:
            Path to the produced PDF.
        """
        af_path = Path(af_graph_path)
        if not af_path.exists():
            raise FileNotFoundError(f"AF graph not found: {af_path}")
        with open(af_path) as f:
            doc = yaml.safe_load(f) or {}
        workload = doc.get("workload") or {}

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._render_with_graphviz(workload, out_path, title)
        if self._debug:
            print(f"Saved AF graph PDF: {out_path}")
        return out_path

    def _render_with_graphviz(
        self,
        workload: Dict[str, Any],
        output_path: Path,
        title: Optional[str],
    ) -> None:
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "graphviz package required. Install with: pip install graphviz"
            ) from e

        einsums: List[Dict[str, Any]] = list(workload.get("einsums") or [])
        rank_sizes: Dict[str, Any] = dict(workload.get("rank_sizes") or {})

        # Producer index: tensor_name -> einsum_name that writes it.
        producer: Dict[str, str] = {}
        for e in einsums:
            for ta in e.get("tensor_accesses") or []:
                if ta.get("output") and ta.get("name") is not None:
                    producer[ta["name"]] = e["name"]

        dot = Digraph(name="af_workload", format="pdf", engine="dot")
        dot.attr(rankdir="LR", nodesep="0.4", ranksep="0.9")
        if title:
            dot.attr(label=title, labelloc="t", fontsize="16")
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontname="Helvetica",
            fontsize="10",
        )
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # Einsum nodes.
        for e in einsums:
            name = e.get("name", "<unnamed>")
            is_copy = bool(e.get("is_copy_operation"))
            fillcolor = "lightyellow" if is_copy else "lightblue"
            dot.node(name, label=self._einsum_label(e), fillcolor=fillcolor)

        # External-source nodes (tensors with no producer in this graph,
        # i.e. true workload inputs / persistent tensors).
        ext_seen: set = set()
        for e in einsums:
            for ta in e.get("tensor_accesses") or []:
                if ta.get("output"):
                    continue
                tn = ta.get("name")
                if tn is None or tn in producer or tn in ext_seen:
                    continue
                ext_seen.add(tn)
                dot.node(
                    f"_ext_{tn}",
                    label=f"{tn}\\n(source)",
                    shape="ellipse",
                    fillcolor="lightgreen",
                )

        # Edges: producer -> consumer for every non-output tensor access.
        for e in einsums:
            dst = e.get("name")
            for ta in e.get("tensor_accesses") or []:
                if ta.get("output"):
                    continue
                tn = ta.get("name")
                if tn is None:
                    continue
                src = producer.get(tn, f"_ext_{tn}")
                bits = ta.get("bits_per_value")
                edge_label = f"{tn} {_proj_str(ta.get('projection'))}"
                if bits is not None:
                    edge_label += f"\\n[{bits}b]"
                dot.edge(src, dst, label=edge_label)

        # Sidebar: rank_sizes table.
        if rank_sizes:
            rs_lines = [f"{k} = {v}" for k, v in rank_sizes.items()]
            rs_label = "rank_sizes\\l" + "\\l".join(rs_lines) + "\\l"
            with dot.subgraph(name="cluster_rank_sizes") as c:
                c.attr(label="", style="invis")
                c.node(
                    "_rank_sizes_table",
                    label=rs_label,
                    shape="note",
                    fillcolor="white",
                    style="filled",
                )

        output_stem = output_path.with_suffix("")
        dot.render(str(output_stem), cleanup=True)

    @staticmethod
    def _einsum_label(e: Dict[str, Any]) -> str:
        """Multi-line label: einsum name + each tensor access."""
        lines = [e.get("name", "<unnamed>")]
        if e.get("is_copy_operation"):
            lines.append("(copy)")
        for ta in e.get("tensor_accesses") or []:
            arrow = "→" if ta.get("output") else "←"
            lines.append(f"{arrow} {ta.get('name')} {_proj_str(ta.get('projection'))}")
        return "\n".join(lines)


def save_af_graph_pdf(
    af_graph_path: PathLike,
    output_path: PathLike,
    *,
    title: Optional[str] = None,
    debug: bool = False,
) -> Path:
    """Convenience wrapper around :class:`AFGraphVisualizer.save_graph_pdf`."""
    return AFGraphVisualizer(debug=debug).save_graph_pdf(
        af_graph_path, output_path, title=title
    )


__all__ = ["AFGraphVisualizer", "save_af_graph_pdf"]
