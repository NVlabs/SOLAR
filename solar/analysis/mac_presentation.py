# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed presentation of an analysis' MAC aggregate (issue #58, report provenance).

``total.macs`` is an AGGREGATE: tensor-core contraction MACs (by compute dtype) plus the multiply statements of
non-einsum kernels that the per-statement classifier counts as MACs.  Every report path that shows the aggregate
must show it as such; this module is the single place that renders it, so the AST check in
``tests/test_mac_report_provenance.py`` can hold every writer to it.  Report-only: nothing here changes a count.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def typed_mac_components(total: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ``macs_typed`` block of an analysis total (``None`` for a record that predates typed components)."""
    typed = (total or {}).get("macs_typed")
    return typed if isinstance(typed, dict) and isinstance(typed.get("aggregate"), int) else None


def describe_macs(total: Dict[str, Any]) -> str:
    """One line: the aggregate followed by its typed components, e.g.
    ``881,134,080 MACs aggregate = 880,803,840 contraction (nvfp4 880,803,840) + 330,240 scalar statement multiplies
    (group_top2_sum#0 327,680, group_topk_mask#0 2,560)``.  A record without typed components is labelled as an
    untyped aggregate rather than presented as homogeneous MACs."""
    total = total or {}
    macs = total.get("macs", 0)
    typed = typed_mac_components(total)
    if typed is None:
        return f"{_n(macs)} MACs (untyped aggregate: contraction MACs and scalar multiply statements not separated by this analysis)"
    c = typed.get("contraction", {}) or {}
    by_dtype = c.get("by_compute_dtype", {}) or {}
    parts = [f"{_n(c.get('macs', 0))} contraction" + (f" ({', '.join(f'{k} {_n(v)}' for k, v in sorted(by_dtype.items()))})" if by_dtype else "")]
    s = typed.get("scalar_statement_multiplies", {}) or {}
    if s.get("macs"):
        layers = s.get("layers", {}) or {}
        parts.append(f"{_n(s['macs'])} scalar statement multiplies" + (f" ({', '.join(f'{k} {_n(v)}' for k, v in sorted(layers.items()))})" if layers else ""))
    o = typed.get("non_contraction_other", {}) or {}
    if o.get("macs"):
        parts.append(f"{_n(o['macs'])} other non-contraction MACs ({', '.join(sorted((o.get('layers') or {}).keys()))})")
    tail = "" if typed.get("components_sum_to_aggregate", True) else "  [components do not sum to the aggregate]"
    unf = total.get("equation_unfaithful_layers") or []
    if unf:
        tail += f"  [equation not faithful for {', '.join(unf)}: schedule fields descriptive only]"
    return f"{_n(typed['aggregate'])} MACs aggregate = " + " + ".join(parts) + tail


def _n(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)
