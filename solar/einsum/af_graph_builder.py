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

"""Principled AccelForge einsum-graph builder for Solar.

Replaces the historical multi-pass rank renamer + AF emit + conflict mint
(``EinsumRankRenamer._process_node``, ``_build_af_einsum_graph_renamed``,
``_build_accelforge_graph``) with a single topologically-ordered union-find
traversal over the stage-2 einsum-graph dict produced by ``_build_einsum_graph``.

Algorithm
=========
A *rank* is a named tensor dimension identified by its physical size. Two
operand positions belong to the same rank iff:

1. **Within-layer same atomic label + same size**: e.g. matmul's
   ``Input=[A,B]``, ``Weight=[B,C]``, ``Output=[A,C]`` — the ``B`` in
   Input and Weight is the same rank (the reduction dim).
2. **Cross-layer producer→consumer position match + same size**: the
   predecessor's output dim ``i`` and the successor's input dim ``i`` are
   the same physical tensor dim.

Two positions with the same label but different sizes are SEPARATE ranks
(handles reduction-with-keepdim cases such as
``Min(dim=1, keepdim=True)``: ``ABCD->ABCD`` with B sized 64 → 1).

Composite labels like ``P+R`` (conv kernel stencil) stay as their own
component in the union-find; the *iterator expression* in their projection
uses the canonical names of the sub-atoms.

The emitted AccelForge graph contains:
- ``rank_sizes``: one entry per equivalence class (canonical name ``R0``,
  ``R1``, ... assigned in topological order of first appearance)
- ``einsums``: one per real layer, with dict-form projections (list form
  when the rank/iter is the trivial identity) referencing canonical names
- top-level ``renames``, ``bits_per_value``, ``persistent_tensors`` blocks

Invariants guaranteed
=====================
1. Every rank name maps to exactly one size — no ``Rk`` reuse across
   layers with different physical sizes.
2. Every tensor accessed in multiple einsums has the same rank tuple.
3. Names are ISL-safe (only ``[A-Za-z0-9_]``).
4. Composite-label positions get their own rank; iterators reference the
   canonical names of sub-atoms.

Usage
=====
From a dict (typical solar internal use)::

    af = build_af_graph_from_dict(einsum_graph)

From an einsum_graph.yaml on disk::

    af = build_af_graph_from_yaml("einsum_graph.yaml",
                                   output_path="af_einsum_graph.yaml")
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Op-type classification
# ---------------------------------------------------------------------------

# Solar layer-types treated as explicit copy operations in the AF graph.
# Only used to fast-path the "start" entry-point pseudo-node here — non-start
# layers from this set are emitted without ``is_copy_operation`` so AF lets
# the mapper handle them normally (setting True triggers AF's "No backing
# TensorHolder" on shape-changing views; setting False explodes the pmapping
# join space for decoder-class graphs).
_OUTPUT_ROLE_PATTERN = re.compile(r"^Output(?:_\d+)?$")
_INPUT_ROLE_PATTERN = re.compile(r"^(?:Input|Weight)(?:_\d+)?$")
_NON_ID_CHAR = re.compile(r"[^A-Za-z0-9_]")


def _is_output_role(role: str) -> bool:
    return bool(_OUTPUT_ROLE_PATTERN.match(role))


def _is_input_role(role: str) -> bool:
    return bool(_INPUT_ROLE_PATTERN.match(role))


def _parse_atoms(label: str) -> List[str]:
    """Split a (possibly-composite) dim label like ``P+R`` into atoms ``[P, R]``."""
    if "+" not in label:
        return [label]
    return [tok.strip() for tok in label.split("+") if tok.strip()]


def _sanitize(name: str) -> str:
    """Make a tensor/einsum name safe for ISL identifiers.

    ISL identifiers must match ``[A-Za-z_][A-Za-z0-9_]*``. Solar emits names
    like ``Model.parameter-tensor`` (dots and hyphens). Replace every
    non-identifier character with underscore; prepend ``_`` if the name
    begins with a digit.
    """
    if not isinstance(name, str):
        return name
    s = _NON_ID_CHAR.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _bits_from_dtype(dtype_str: str) -> Optional[int]:
    """Translate a torch dtype string (e.g. ``'torch.float16'``) to bit width."""
    if not isinstance(dtype_str, str):
        return None
    s = dtype_str.replace("torch.", "").lower()
    mapping = {
        "float64": 64, "double": 64, "complex128": 128, "complex64": 64,
        "float32": 32, "tf32": 32,
        "bfloat16": 16, "float16": 16, "half": 16,
        "int64": 64, "long": 64, "int32": 32, "int": 32,
        "int16": 16, "short": 16, "int8": 8, "uint8": 8, "byte": 8,
        "bool": 1,
    }
    return mapping.get(s)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisKey:
    """Unique identifier for one dim of one operand in one layer."""
    layer: str
    role: str
    pos: int


@dataclass
class Axis:
    key: AxisKey
    label: str  # raw label as emitted by solar (e.g. "B" or "P+R")
    size: int


# ---------------------------------------------------------------------------
# Union-find with size-checked unions
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[AxisKey, AxisKey] = {}
        self._first_seen_order: Dict[AxisKey, int] = {}
        self._counter = 0

    def add(self, x: AxisKey) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._first_seen_order[x] = self._counter
            self._counter += 1

    def find(self, x: AxisKey) -> AxisKey:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        cur = x
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, x: AxisKey, y: AxisKey) -> AxisKey:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        # Stable canonical: keep the one seen earliest (topological priority).
        if self._first_seen_order[rx] <= self._first_seen_order[ry]:
            self._parent[ry] = rx
            return rx
        self._parent[rx] = ry
        return ry


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class BuildContext:
    layers: Dict[str, dict]
    """Topologically-ordered dict from einsum_graph stage-2 output."""

    axes: Dict[AxisKey, Axis] = field(default_factory=dict)
    """All (layer, role, pos) → Axis."""

    uf: UnionFind = field(default_factory=UnionFind)
    """Union-find over AxisKey."""

    canonical_name: Dict[AxisKey, str] = field(default_factory=dict)
    """Axis → canonical name (R0, R1, ...)."""

    rank_sizes: Dict[str, int] = field(default_factory=dict)
    """canonical_name → size."""

    role_to_shape_index: Dict[Tuple[str, str], Tuple[str, int]] = field(default_factory=dict)
    """(layer, role) → ('inputs' or 'outputs', index)."""

    diagnostics: List[str] = field(default_factory=list)


def _build_role_to_shape_index(layers: Dict[str, dict],
                                ctx: BuildContext) -> None:
    """Map (layer, role) → (which-tensor-shapes-list, index-into-list).

    Walks operand keys in YAML insertion order. Input-like roles consume
    successive ``tensor_shapes.inputs`` entries; output-like roles consume
    ``tensor_shapes.outputs``. Unknown role names (e.g. "start" pseudo-node)
    pick whichever bucket still has slots. When the indexed slot is
    out-of-range we still assign an index so the emitter can classify; the
    size is inferred later in ``_collect_axes`` from same-layer atom labels.
    """
    for layer_name, L in layers.items():
        operands = L.get("operands") or {}
        shapes_in = L.get("tensor_shapes", {}).get("inputs") or []
        shapes_out = L.get("tensor_shapes", {}).get("outputs") or []
        next_in = 0
        next_out = 0
        for role in operands:
            kind: Optional[str] = None
            if _is_output_role(role):
                kind = "outputs"
            elif _is_input_role(role):
                kind = "inputs"
            else:
                if next_in < len(shapes_in):
                    kind = "inputs"
                elif next_out < len(shapes_out):
                    kind = "outputs"
                else:
                    kind = "inputs"
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}: role {role!r} unclassified "
                        f"with no remaining shape slots; defaulting to input."
                    )
            if kind == "outputs":
                ctx.role_to_shape_index[(layer_name, role)] = ("outputs", next_out)
                next_out += 1
            else:
                ctx.role_to_shape_index[(layer_name, role)] = ("inputs", next_in)
                next_in += 1


def _collect_axes(ctx: BuildContext) -> None:
    """Populate ctx.axes for every (layer, role, pos).

    When solar's ``tensor_shapes`` is missing an entry for a role (e.g.
    matmul Weight: 2 input operands but only 1 input shape recorded),
    infer the size from same-layer atom labels — any atom appearing
    atomically in another operand of the same layer carries its size.
    Composite labels (``P+R``) are skipped (need atom-level info).
    """
    for layer_name, L in ctx.layers.items():
        operands = L.get("operands") or {}
        shapes_in = L.get("tensor_shapes", {}).get("inputs") or []
        shapes_out = L.get("tensor_shapes", {}).get("outputs") or []

        label_size: Dict[str, int] = {}
        for role, dims in operands.items():
            ki = ctx.role_to_shape_index.get((layer_name, role))
            if ki is None:
                continue
            kind, idx = ki
            shapes_list = shapes_in if kind == "inputs" else shapes_out
            if idx >= len(shapes_list):
                continue
            shape = shapes_list[idx]
            for pos, label in enumerate(dims):
                if pos >= len(shape):
                    break
                if "+" not in label and label not in label_size:
                    label_size[label] = int(shape[pos])

        for role, dims in operands.items():
            key_to_kind = ctx.role_to_shape_index.get((layer_name, role))
            shape: Optional[List[int]] = None
            if key_to_kind is not None:
                kind, idx = key_to_kind
                shapes_list = shapes_in if kind == "inputs" else shapes_out
                if idx < len(shapes_list):
                    shape = shapes_list[idx]
            if shape is None:
                inferred: List[Optional[int]] = []
                for label in dims:
                    if "+" in label:
                        inferred.append(None)
                    else:
                        inferred.append(label_size.get(label))
                if all(s is not None for s in inferred):
                    shape = inferred  # type: ignore[assignment]
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}, role {role!r}: inferred shape "
                        f"{shape} from same-layer atom labels."
                    )
                else:
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}, role {role!r}: no tensor_shape "
                        f"and couldn't infer; skipping."
                    )
                    continue
            n = min(len(dims), len(shape))
            for pos in range(n):
                key = AxisKey(layer_name, role, pos)
                ctx.axes[key] = Axis(key=key, label=dims[pos], size=int(shape[pos]))
                ctx.uf.add(key)


def _within_layer_union(ctx: BuildContext) -> None:
    """Phase 1: union axes within a layer that share (atomic label, size).

    Composite labels (containing ``+``) stay as their own identity. Grouping
    by ``(label, size)`` ensures that when a layer like ``div`` has three
    "B"-labeled positions with sizes ``[64, 1, 64]``, the two size-64
    entries unify even though they're not adjacent in YAML order.
    """
    by_layer: Dict[str, List[AxisKey]] = defaultdict(list)
    for k in ctx.axes:
        by_layer[k.layer].append(k)
    for _layer, keys in by_layer.items():
        groups: Dict[Tuple[str, int], List[AxisKey]] = defaultdict(list)
        for k in keys:
            ax = ctx.axes[k]
            groups[(ax.label, ax.size)].append(k)
        for _key, group in groups.items():
            for a in group[1:]:
                ctx.uf.union(group[0], a)


def _input_like_roles_in_order(operands: dict, layer_name: str,
                                role_to_shape_index: dict,
                                tensor_types_inputs: Optional[List[str]] = None,
                                skip_weight_typed: bool = False) -> List[str]:
    """Ordered list of input-like roles (those mapping to tensor_shapes.inputs).

    Catches solar's custom input roles (``Target`` for loss functions,
    ``Hidden_in``/``Cell_in`` for RNN/LSTM/GRU, etc.) — anything mapped to
    an ``inputs`` slot is treated as an input regardless of name.

    When ``skip_weight_typed`` is True and ``tensor_types_inputs`` is
    available, drop roles tagged as ``"weight"`` — these don't consume
    a predecessor (they're emitted as fresh ``W{n}`` tensors). Without
    skipping, the cross-layer union pairs a weight role with the next
    predecessor and mis-unions the pred's output axes with the scalar
    weight's axes — the L2/84 (`Gemm + BatchNorm + scale*x + Softmax`)
    crash signature.
    """
    out = []
    input_pos = 0
    for role in operands:
        idx = role_to_shape_index.get((layer_name, role))
        if idx is None or idx[0] != "inputs":
            continue
        if skip_weight_typed and tensor_types_inputs is not None:
            role_type = (
                tensor_types_inputs[input_pos]
                if input_pos < len(tensor_types_inputs)
                else None
            )
            input_pos += 1
            if role_type == "weight":
                continue
        else:
            input_pos += 1
        out.append(role)
    return out


def _primary_output_role(pred_operands: dict,
                          role_to_shape_index: dict,
                          pred_name: str) -> Optional[str]:
    """Return the predecessor's primary output role name."""
    for cand in pred_operands:
        if _is_output_role(cand):
            return cand
    # Fallback for pseudo-nodes (e.g. "start").
    for cand in pred_operands:
        idx = role_to_shape_index.get((pred_name, cand))
        if idx is not None and idx[0] == "outputs":
            return cand
    return None


def _cross_layer_union(ctx: BuildContext) -> None:
    """Phase 2: union axes across producer→consumer connections (pos-wise)."""
    for layer_name, L in ctx.layers.items():
        preds = (L.get("connections") or {}).get("inputs") or []
        operands = L.get("operands") or {}
        # connections.inputs lists only the non-weight predecessors. Match
        # that by skipping weight-typed roles when pairing roles ↔ preds —
        # otherwise the cross-layer union for `mul(scale, x)` mis-pairs
        # ``scale`` with batch_norm and never unions batch_norm's actual
        # axes with the mul input's, producing distinct canonical names
        # across einsums that AF's pydantic schema rejects.
        tensor_types_inputs = (L.get("tensor_types") or {}).get("inputs") or []
        input_roles = _input_like_roles_in_order(
            operands, layer_name,
            ctx.role_to_shape_index,
            tensor_types_inputs=tensor_types_inputs,
            skip_weight_typed=True,
        )
        for i, role in enumerate(input_roles):
            if i >= len(preds):
                break
            pred = preds[i]
            pred_layer = ctx.layers.get(pred)
            if pred_layer is None:
                continue
            pred_operands = pred_layer.get("operands") or {}
            pred_output_role = _primary_output_role(
                pred_operands, ctx.role_to_shape_index, pred
            )
            if pred_output_role is None:
                continue
            pred_dims = pred_operands.get(pred_output_role, [])
            cur_dims = operands.get(role, [])
            n = min(len(pred_dims), len(cur_dims))
            for pos in range(n):
                a = AxisKey(pred, pred_output_role, pos)
                b = AxisKey(layer_name, role, pos)
                if a not in ctx.axes or b not in ctx.axes:
                    continue
                if ctx.axes[a].size == ctx.axes[b].size:
                    ctx.uf.union(a, b)


def _assign_canonical_names(ctx: BuildContext) -> None:
    """Phase 3: assign R0, R1, ... to equivalence classes in topological order."""
    counter = 0
    seen_roots: Dict[AxisKey, str] = {}
    for layer_name, L in ctx.layers.items():
        operands = L.get("operands") or {}
        for role, dims in operands.items():
            for pos in range(len(dims)):
                key = AxisKey(layer_name, role, pos)
                if key not in ctx.axes:
                    continue
                root = ctx.uf.find(key)
                if root not in seen_roots:
                    name = f"R{counter}"
                    counter += 1
                    seen_roots[root] = name
                    ctx.rank_sizes[name] = ctx.axes[key].size
                else:
                    existing = ctx.rank_sizes[seen_roots[root]]
                    if existing != ctx.axes[key].size:
                        ctx.diagnostics.append(
                            f"size mismatch in component {seen_roots[root]}: "
                            f"{existing} vs {ctx.axes[key].size} at {key}"
                        )
                ctx.canonical_name[key] = seen_roots[root]


# ---------------------------------------------------------------------------
# AF YAML emission
# ---------------------------------------------------------------------------


def _build_iter_expr_for_layer(ctx: BuildContext, layer_name: str) -> Dict[str, str]:
    """Per-layer ``atom_letter → lowercase canonical_iter_var`` map.

    Used by ``_projection_for_axis`` to rewrite composite labels (``P+R``)
    into iterator expressions over canonical rank names (e.g. ``r5+r7``).
    """
    operands = ctx.layers[layer_name].get("operands") or {}
    atomic_to_iter: Dict[str, str] = {}
    for role, dims in operands.items():
        for pos, label in enumerate(dims):
            atoms = _parse_atoms(label)
            if len(atoms) == 1:
                atom = atoms[0]
                key = AxisKey(layer_name, role, pos)
                canonical = ctx.canonical_name.get(key)
                if canonical is not None and atom not in atomic_to_iter:
                    atomic_to_iter[atom] = canonical.lower()
    # Second pass: atoms only referenced inside composites without an
    # atomic anchor — synthesize a fallback iterator name.
    for role, dims in operands.items():
        for pos, label in enumerate(dims):
            atoms = _parse_atoms(label)
            for atom in atoms:
                if atom not in atomic_to_iter:
                    atomic_to_iter[atom] = (
                        f"x_{layer_name.replace('.', '_')}_{atom}".lower()
                    )
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}: atom {atom!r} in composite "
                        f"{label!r} has no atomic anchor; using fallback iter."
                    )
    return atomic_to_iter


def _projection_for_axis(ctx: BuildContext, key: AxisKey,
                          atomic_iter_map: Dict[str, str]) -> str:
    """Iterator expression for one operand position."""
    label = ctx.axes[key].label
    atoms = _parse_atoms(label)
    if len(atoms) == 1:
        return ctx.canonical_name[key].lower()
    return "+".join(atomic_iter_map[a] for a in atoms)


def _bits_for_role(L: dict, role: str,
                    ctx_idx: Optional[Tuple[str, int]] = None) -> Optional[int]:
    """Return bits-per-value for one operand role with sensible fallbacks."""
    dtypes = L.get("tensor_dtypes") or {}
    if ctx_idx is None:
        for kind in ("outputs", "inputs"):
            dlist = dtypes.get(kind) or []
            if dlist:
                return _bits_from_dtype(dlist[0])
        return None
    kind, idx = ctx_idx
    dtype_list = dtypes.get(kind) or []
    if idx < len(dtype_list):
        return _bits_from_dtype(dtype_list[idx])
    if dtype_list:
        return _bits_from_dtype(dtype_list[0])
    other = "outputs" if kind == "inputs" else "inputs"
    other_list = dtypes.get(other) or []
    if other_list:
        return _bits_from_dtype(other_list[0])
    return None


def _emit_af_workload(ctx: BuildContext, model_name: str) -> dict:
    einsums: List[dict] = []
    weight_counter = [0]

    def next_weight_name() -> str:
        weight_counter[0] += 1
        return f"W{weight_counter[0]}"

    for layer_name, L in ctx.layers.items():
        operands = L.get("operands") or {}
        if not operands:
            continue
        sanitized_name = _sanitize(layer_name)
        preds = (L.get("connections") or {}).get("inputs") or []
        atomic_iter_map = _build_iter_expr_for_layer(ctx, layer_name)
        tensor_accesses: List[dict] = []

        rename_target: Dict[str, str] = {}
        primary_input_set = False
        primary_weight_set = False

        # Solar annotates each input role as "input" or "weight" in
        # tensor_types.inputs (positional, matching operands' input-role
        # iteration order). connections.inputs only lists the non-weight
        # producers, so we must skip "weight"-typed roles when stepping
        # through preds — otherwise a layer like ``mul(scale, x)`` where
        # operand 0 is a weight and operand 1 is the predecessor tensor
        # ends up assigning the predecessor's name to the weight slot
        # (size 1) and a synthetic W{n} to the tensor slot (full rank),
        # producing pydantic "inconsistent ranks" errors downstream.
        tensor_types_inputs = (L.get("tensor_types") or {}).get("inputs") or []
        input_role_index = 0  # position within input-typed roles only
        pred_index = 0
        for role, dims in operands.items():
            ctx_idx = ctx.role_to_shape_index.get((layer_name, role))
            tensor_name: str
            is_output_access = False
            af_rename_key: Optional[str] = None
            role_kind = ctx_idx[0] if ctx_idx is not None else None
            if role_kind == "outputs":
                is_output_access = True
                if role == "Output" or role == "Output_0":
                    tensor_name = sanitized_name
                elif role.startswith("Output_"):
                    n_str = role.split("_", 1)[1] if "_" in role else "0"
                    tensor_name = f"{sanitized_name}_{n_str}"
                else:
                    tensor_name = sanitized_name
                af_rename_key = "output"
            elif role_kind == "inputs":
                # Honor solar's tensor_types tagging when it disagrees with
                # the simple "any input consumes a pred" assumption.
                role_type = (
                    tensor_types_inputs[input_role_index]
                    if input_role_index < len(tensor_types_inputs)
                    else None
                )
                is_weight_role = (role_type == "weight")
                if is_weight_role:
                    tensor_name = next_weight_name()
                    af_rename_key = "weight"
                elif pred_index < len(preds):
                    tensor_name = _sanitize(preds[pred_index])
                    pred_index += 1
                    # Match OLD-pipeline convention: the FIRST consumed
                    # pred maps to "input", subsequent preds map to "weight".
                    af_rename_key = "input" if not primary_input_set else "weight"
                else:
                    tensor_name = next_weight_name()
                    af_rename_key = "weight"
                input_role_index += 1
            else:
                tensor_name = sanitized_name
                is_output_access = True
                af_rename_key = "output"

            projection: Dict[str, str] = {}
            for pos in range(len(dims)):
                key = AxisKey(layer_name, role, pos)
                if key not in ctx.axes:
                    continue
                canonical = ctx.canonical_name[key]
                expr = _projection_for_axis(ctx, key, atomic_iter_map)
                projection[canonical] = expr
            can_demote = all(
                isinstance(v, str) and v.isidentifier() and k == v.upper()
                for k, v in projection.items()
            )
            access: Dict[str, Any] = {
                "name": tensor_name,
                "projection": list(projection.values()) if can_demote else dict(projection),
            }
            if is_output_access:
                access["output"] = True
            bits = _bits_for_role(L, role, ctx_idx)
            if bits is not None:
                access["bits_per_value"] = bits
            tensor_accesses.append(access)

            if af_rename_key == "input" and not primary_input_set:
                rename_target["input"] = tensor_name
                primary_input_set = True
            elif af_rename_key == "weight" and not primary_weight_set:
                rename_target["weight"] = tensor_name
                primary_weight_set = True
            elif af_rename_key == "output" and "output" not in rename_target:
                rename_target["output"] = tensor_name

        # Dedup multi-read input accesses (e.g. residual ``Add(x, x)`` lists
        # the same predecessor twice). AF requires unique tensor names in
        # an einsum's tensor_accesses, and reading the same tensor twice
        # with the same projection has the same memory cost as reading it
        # once — keep one access. Output accesses are never deduped.
        seen_in: set = set()
        deduped: list = []
        for ta in tensor_accesses:
            if ta.get("output"):
                deduped.append(ta)
                continue
            # Hash the (name, projection) pair so distinct-projection
            # multi-reads (rare; would require an alias if encountered)
            # still surface as a separate access.
            proj = ta["projection"]
            key = (ta["name"], tuple(proj) if isinstance(proj, list)
                    else tuple(sorted(proj.items())))
            if key in seen_in:
                continue
            seen_in.add(key)
            deduped.append(ta)
        tensor_accesses = deduped

        # Entry-point pseudo-nodes ("start") have outputs but no inputs and
        # no predecessors. Synthesize a source-input access so AF has a
        # tensor to read from at the model boundary.
        has_input = any(not ta.get("output") for ta in tensor_accesses)
        has_output = any(ta.get("output") for ta in tensor_accesses)
        is_entry_point = has_output and not has_input and not preds
        if is_entry_point:
            out_ta = next(ta for ta in tensor_accesses if ta.get("output"))
            synth_name = f"{sanitized_name}_in"
            synth: Dict[str, Any] = {
                "name": synth_name,
                "projection": (
                    out_ta["projection"]
                    if isinstance(out_ta["projection"], list)
                    else dict(out_ta["projection"])
                ),
            }
            if "bits_per_value" in out_ta:
                synth["bits_per_value"] = out_ta["bits_per_value"]
            tensor_accesses.insert(0, synth)

        if is_entry_point:
            einsum_renames = {
                "input": "Inputs()",
                "output": "Outputs()",
                "weight": "Nothing()",
            }
        else:
            einsum_renames = {
                "input": rename_target.get("input", "Nothing()"),
                "output": rename_target.get("output", sanitized_name),
                "weight": rename_target.get("weight", "Nothing()"),
            }

        einsum_entry: Dict[str, Any] = {
            "name": sanitized_name,
            "tensor_accesses": tensor_accesses,
            "renames": einsum_renames,
        }
        if is_entry_point:
            einsum_entry["is_copy_operation"] = True
        einsums.append(einsum_entry)

    # Pin a canonical rank order per tensor. Union-find guarantees rank
    # IDENTITY is consistent across multiple accesses of the same tensor,
    # but the order in the projection may differ if Solar wrote different
    # operand orderings. Pin the first occurrence and rewrite mismatches.
    canonical_rank_order: Dict[str, List[str]] = {}
    for e in einsums:
        for ta in e["tensor_accesses"]:
            name = ta["name"]
            proj = ta["projection"]
            if isinstance(proj, list):
                ranks = [v.upper() for v in proj]
            elif isinstance(proj, dict):
                ranks = list(proj.keys())
            else:
                continue
            if name not in canonical_rank_order:
                canonical_rank_order[name] = ranks
            else:
                target_ranks = canonical_rank_order[name]
                if ranks != target_ranks:
                    iter_map: Dict[str, str] = {}
                    if isinstance(proj, list):
                        for r, v in zip(ranks, proj):
                            iter_map[r] = v
                    else:
                        for r, v in proj.items():
                            iter_map[r] = v
                    if all(r in iter_map for r in target_ranks):
                        ta["projection"] = {r: iter_map[r] for r in target_ranks}

    all_bits = [
        ta.get("bits_per_value")
        for e in einsums for ta in e["tensor_accesses"]
        if ta.get("bits_per_value") is not None
    ]
    default_bits = max(all_bits) if all_bits else 32

    workload = {
        "rank_sizes": dict(ctx.rank_sizes),
        "bits_per_value": {"All": default_bits},
        "persistent_tensors": "weight - Intermediates",
        "einsums": einsums,
    }
    renames = {
        "einsums": [
            {
                "name": "default",
                "tensor_accesses": [
                    {"name": "input", "source": "Inputs & Intermediates",
                     "expected_count": 1},
                    {"name": "output", "source": "Outputs", "expected_count": 1},
                    {"name": "weight", "source": "~(input | output)",
                     "expected_count": 1},
                ],
            }
        ]
    }
    return {"workload": workload, "renames": renames}


def _ghost_scalar_outputs(af: dict) -> None:
    """Give TERMINAL reduction-to-scalar outputs one bounded rank.

    Solar emits ops like ``Model.cross_entropy``, ``Model.mean``,
    ``Model.smooth_l1_loss``, ``Model.kl_div`` with empty output
    projection. An empty-projection output that is a graph TERMINAL (no
    consumer) propagates through AF's join_pmappings as an unconstrained
    schedule, breaking upstream multi-input joins ("No mappings found for
    start <--> start_1"). We give such terminals a single rank borrowed
    from a non-output input access so AF can bound the operation space.

    Two refinements over the historical behavior:

    * **Consumed scalars are left empty.** A scalar output that feeds a
      downstream op (e.g. ``norm`` in ``x / x.norm()``) is a genuine
      intermediate; the consumer reads it with an empty projection too, so
      promoting only one side would make the same tensor carry two rank
      tuples (caught by ``_validate_graph_invariants``). Leaving it empty
      keeps producer and consumer consistent — AF bounds it via the join
      with the consumer's other (ranked) operand.
    * **Smallest rank, not first.** Among the input ranks we pick the
      one with the smallest size. AF rejects an unbounded fresh rank, and
      a fresh size-1 rank is rejected too (ISL "Shape infty"); reusing a
      bounded input rank is required. Choosing the smallest minimizes the
      (already negligible) spurious output-write traffic.
    """
    einsums = af["workload"]["einsums"]
    rank_sizes = af["workload"].get("rank_sizes") or {}
    consumed = {
        ta["name"]
        for e in einsums for ta in e["tensor_accesses"]
        if not ta.get("output")
    }
    for e in einsums:
        if e.get("is_copy_operation"):
            continue
        for out_ta in e["tensor_accesses"]:
            if not out_ta.get("output"):
                continue
            proj = out_ta["projection"]
            is_empty = (isinstance(proj, list) and len(proj) == 0) or (
                isinstance(proj, dict) and len(proj) == 0)
            if not is_empty or out_ta["name"] in consumed:
                continue
            best_iter = None
            best_size = None
            for ta in e["tensor_accesses"]:
                if ta.get("output"):
                    continue
                p = ta["projection"]
                pairs = (
                    [(v.upper(), v) for v in p] if isinstance(p, list)
                    else list(p.items())
                )
                for rank, it in pairs:
                    sz = rank_sizes.get(rank.upper() if isinstance(rank, str) else rank)
                    if sz is not None and (best_size is None or sz < best_size):
                        best_size, best_iter = sz, it
            if best_iter is not None:
                out_ta["projection"] = [best_iter]


# ---------------------------------------------------------------------------
# Shape-op elision (graph rewrite that runs BEFORE union-find)
# ---------------------------------------------------------------------------

# Solar layer types that are pure shape views — no actual compute / memory
# traffic. When ``is_real_einsum: false`` co-occurs with one of these types
# we drop the op and rewire downstream readers to access the physical root.
_SHAPE_OP_TYPES: Set[str] = {
    "transpose", "permute", "contiguous", "squeeze", "unsqueeze",
    "expand", "__getitem__", "__get__", "view", "reshape",
}


@dataclass
class _Alias:
    root_tensor: str
    """Tensor name in the surviving (root) producer's output."""

    root_layer: str
    """Name of the layer that produces ``root_tensor``."""

    root_role: str
    """Role under which ``root_layer`` exposes ``root_tensor`` (typically 'Output')."""

    root_dims: List[str]
    """Operand labels of the root producer's output, in order."""

    root_shape: List[int]
    """Shape of the root producer's output, in order."""

    in_to_out: List[List[int]]
    """For each input position of the elided shape-op, the list of output
    positions it maps to. Empty list means the input position was dropped
    (squeeze); multiple means it was broadcast (expand). Lengths and
    indices reference the shape-op's OWN input/output positions, not the
    physical root's (those are reachable transitively via this struct's
    ``root_dims``)."""

    out_to_in: List[Optional[int]]
    """For each output position of the elided shape-op, the input position
    it came from, or None if introduced (unsqueeze)."""


def _derive_pos_mapping(layer_name: str, L: dict,
                         in_dims: List[str], in_shape: List[int],
                         out_dims: List[str], out_shape: List[int]
                         ) -> Optional[Tuple[List[Optional[int]],
                                              List[List[int]]]]:
    """Return (out_to_in, in_to_out) for an elide-able shape op.

    Returns None when the rewrite is not safe (e.g. genuine reshape, or
    shape ambiguity we can't resolve from operands alone)."""
    n_in = len(in_dims)
    n_out = len(out_dims)
    op_type = L.get("type")

    # contiguous / identity-view (shapes match positionally).
    if n_in == n_out and in_shape == out_shape:
        return [i for i in range(n_in)], [[i] for i in range(n_in)]

    # Pure label permutation (same multiset of labels, same multiset of sizes).
    # transpose / permute usually fall here. solar's transpose has identical
    # labels with reordered sizes ("AB->AB" but shape transposed) — so we
    # PREFER shape-based matching over label-based.
    if n_in == n_out and sorted(in_shape) == sorted(out_shape):
        # Try label-based permutation first when the labels are a true
        # multiset permutation distinct from identity.
        if sorted(in_dims) == sorted(out_dims) and in_dims != out_dims:
            used: Set[int] = set()
            o2i: List[Optional[int]] = []
            for j, lbl in enumerate(out_dims):
                hit = None
                for i, ilbl in enumerate(in_dims):
                    if i in used:
                        continue
                    if ilbl == lbl and in_shape[i] == out_shape[j]:
                        hit = i
                        break
                if hit is None:
                    o2i = None  # type: ignore[assignment]
                    break
                used.add(hit)
                o2i.append(hit)
            if o2i is not None:
                i2o: List[List[int]] = [[] for _ in range(n_in)]
                for j, i in enumerate(o2i):
                    if i is not None:
                        i2o[i].append(j)
                return o2i, i2o
        # Shape-based positional permutation. Greedy unique match by size;
        # bail out if sizes aren't unique enough to derive an unambiguous
        # permutation (caller will emit the op normally).
        used2: Set[int] = set()
        o2i2: List[Optional[int]] = []
        ok = True
        for j in range(n_out):
            hit = None
            for i in range(n_in):
                if i in used2:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit = i
                    break
            if hit is None:
                ok = False
                break
            used2.add(hit)
            o2i2.append(hit)
        if ok:
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i2):
                if i is not None:
                    i2o[i].append(j)
            return o2i2, i2o
        return None

    # squeeze: input has size-1 dims that are dropped in output.
    if op_type == "squeeze" and n_out <= n_in:
        ok = True
        used3: Set[int] = set()
        o2i3: List[Optional[int]] = []
        for j in range(n_out):
            hit = None
            for i in range(n_in):
                if i in used3:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit = i
                    break
            if hit is None:
                ok = False
                break
            used3.add(hit)
            o2i3.append(hit)
        if ok:
            # Every unmatched input position must be size 1 (the dropped axes).
            if all(in_shape[i] == 1 for i in range(n_in) if i not in used3):
                i2o = [[] for _ in range(n_in)]
                for j, i in enumerate(o2i3):
                    if i is not None:
                        i2o[i].append(j)
                return o2i3, i2o
        return None

    # unsqueeze: output has size-1 dims that aren't in input.
    if op_type == "unsqueeze" and n_in <= n_out:
        ok = True
        used4: Set[int] = set()
        o2i4: List[Optional[int]] = []
        for j in range(n_out):
            if out_shape[j] == 1:
                # Either it's a true unsqueeze-introduced axis, or a
                # preserved size-1 from input — prefer to match an unused
                # size-1 input first so order is stable.
                hit = None
                for i in range(n_in):
                    if i in used4:
                        continue
                    if in_shape[i] == 1:
                        hit = i
                        break
                if hit is not None:
                    used4.add(hit)
                    o2i4.append(hit)
                else:
                    o2i4.append(None)
                continue
            hit2 = None
            for i in range(n_in):
                if i in used4:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit2 = i
                    break
            if hit2 is None:
                ok = False
                break
            used4.add(hit2)
            o2i4.append(hit2)
        if ok and len(used4) == n_in:
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i4):
                if i is not None:
                    i2o[i].append(j)
            return o2i4, i2o
        return None

    # expand: a size-1 input dim is broadcast to a larger output dim.
    if op_type == "expand" and n_in == n_out:
        o2i5: List[Optional[int]] = list(range(n_in))
        i2o: List[List[int]] = [[j] for j in range(n_in)]
        return o2i5, i2o

    # __getitem__ / __get__: only safe when the access selects every
    # element along every axis (no-op). Detect by exact shape equality.
    if op_type in ("__getitem__", "__get__"):
        if n_in == n_out and in_shape == out_shape:
            return [i for i in range(n_in)], [[i] for i in range(n_in)]
        # Same total size but different shape — could be transpose-like
        # (e.g. `.T` is emitted as __get__ with shape swap). Try the
        # permutation derivation above.
        try:
            prod_in = 1
            for s in in_shape:
                prod_in *= s
            prod_out = 1
            for s in out_shape:
                prod_out *= s
        except Exception:
            return None
        if prod_in == prod_out and n_in == n_out \
                and sorted(in_shape) == sorted(out_shape):
            used5: Set[int] = set()
            o2i6: List[Optional[int]] = []
            ok = True
            for j in range(n_out):
                hit = None
                for i in range(n_in):
                    if i in used5:
                        continue
                    if in_shape[i] == out_shape[j]:
                        hit = i
                        break
                if hit is None:
                    ok = False
                    break
                used5.add(hit)
                o2i6.append(hit)
            if ok:
                i2o = [[] for _ in range(n_in)]
                for j, i in enumerate(o2i6):
                    if i is not None:
                        i2o[i].append(j)
                return o2i6, i2o
        return None

    # view / reshape: only elide when it's a pure no-op (same shape) OR a
    # squeeze-or-unsqueeze of size-1 dims. Axis collapse / split is left
    # to AF as a real op.
    if op_type in ("view", "reshape"):
        # Identity reshape (shape unchanged).
        if n_in == n_out and in_shape == out_shape:
            return [i for i in range(n_in)], [[i] for i in range(n_in)]
        # Reshape that only adds/drops size-1 dims (product preserved).
        # Build the mapping by walking the non-unit dim sequences in both
        # sides — they must match in order.
        in_nonunit = [(i, s) for i, s in enumerate(in_shape) if s != 1]
        out_nonunit = [(j, s) for j, s in enumerate(out_shape) if s != 1]
        if len(in_nonunit) == len(out_nonunit) \
                and all(a[1] == b[1] for a, b in zip(in_nonunit, out_nonunit)):
            o2i7: List[Optional[int]] = [None] * n_out
            for (i, _), (j, _) in zip(in_nonunit, out_nonunit):
                o2i7[j] = i
            # Pair leftover size-1 input dims to leftover size-1 output
            # dims in order; remaining are introduced (None on out side)
            # or dropped (no entry on out side).
            unmatched_in = [i for i in range(n_in)
                             if i not in {x for x in o2i7 if x is not None}]
            unmatched_out = [j for j in range(n_out) if o2i7[j] is None]
            for k in range(min(len(unmatched_in), len(unmatched_out))):
                o2i7[unmatched_out[k]] = unmatched_in[k]
            # Validate every input dim is either matched (one or more
            # output slot points to it) or is a dropped size-1.
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i7):
                if i is not None:
                    i2o[i].append(j)
            for i in range(n_in):
                if not i2o[i] and in_shape[i] != 1:
                    return None
            return o2i7, i2o
        return None

    return None


def _build_shape_op_aliases(layers: Dict[str, dict]
                             ) -> Tuple[Dict[str, _Alias], Set[str], List[str]]:
    """Return (alias_table, elided_set, diagnostics).

    Walks layers in topological order and records, for each elide-able
    pure-shape-op layer, an Alias entry keyed by the layer's primary
    output tensor name. Chained shape ops compose via the table — when
    an op's predecessor is itself elided, we look up the predecessor's
    alias and propagate the root forward.
    """
    aliases: Dict[str, _Alias] = {}
    elided: Set[str] = set()
    diagnostics: List[str] = []

    for name, L in layers.items():
        if L.get("is_real_einsum", True):
            continue
        op_type = L.get("type")
        if op_type not in _SHAPE_OP_TYPES:
            continue

        operands = L.get("operands") or {}
        in_roles = [r for r in operands
                     if _is_input_role(r)
                     or (not _is_output_role(r) and r != "start")]
        out_roles = [r for r in operands if _is_output_role(r)]
        # Skip multi-input or no-output shape ops.
        if len(in_roles) != 1 or len(out_roles) != 1:
            continue
        preds = (L.get("connections") or {}).get("inputs") or []
        if len(preds) != 1:
            continue

        in_role = in_roles[0]
        out_role = out_roles[0]
        in_dims = list(operands.get(in_role) or [])
        out_dims = list(operands.get(out_role) or [])
        shapes = L.get("tensor_shapes", {}) or {}
        in_shape_list = shapes.get("inputs") or []
        out_shape_list = shapes.get("outputs") or []
        if not in_shape_list or not out_shape_list:
            continue
        in_shape = list(in_shape_list[0])
        out_shape = list(out_shape_list[0])

        # Defensive: total size must be preserved (expand explicitly excepted).
        if op_type != "expand":
            try:
                pi = 1
                for s in in_shape:
                    pi *= int(s)
                po = 1
                for s in out_shape:
                    po *= int(s)
            except Exception:
                diagnostics.append(
                    f"layer {name!r}: non-integer shape; emit normally")
                continue
            if pi != po:
                diagnostics.append(
                    f"layer {name!r}: shape product mismatch "
                    f"(in={in_shape}, out={out_shape}); emit normally")
                continue

        # Detect partial __getitem__ (slice with stride/length != full
        # range). For now, only elide when shapes equate or are a pure
        # permutation (handled inside _derive_pos_mapping).
        # Compute the rewrite.
        derived = _derive_pos_mapping(name, L, in_dims, in_shape,
                                       out_dims, out_shape)
        if derived is None:
            diagnostics.append(
                f"layer {name!r} ({op_type}): could not derive a safe "
                f"projection rewrite (in={in_dims}/{in_shape}, "
                f"out={out_dims}/{out_shape}); emit normally")
            continue
        out_to_in, in_to_out = derived

        # Resolve predecessor's primary output tensor + role.
        pred_name = preds[0]
        pred_layer = layers.get(pred_name)
        if pred_layer is None:
            continue
        pred_operands = pred_layer.get("operands") or {}
        pred_out_role: Optional[str] = None
        for cand in pred_operands:
            if _is_output_role(cand):
                pred_out_role = cand
                break
        if pred_out_role is None:
            # pseudo-node (e.g. "start") — fall back to its first operand.
            if pred_operands:
                pred_out_role = next(iter(pred_operands))
        if pred_out_role is None:
            continue

        pred_output_tensor_name = (
            (pred_layer.get("tensor_names") or {}).get("outputs") or [None]
        )[0]
        pred_out_shape_list = (pred_layer.get("tensor_shapes") or {}).get("outputs") or []
        if not pred_out_shape_list:
            continue
        pred_out_shape = list(pred_out_shape_list[0])
        pred_out_dims = list(pred_operands.get(pred_out_role) or [])

        my_output_tensor_name = (
            (L.get("tensor_names") or {}).get("outputs") or [None]
        )[0]
        if my_output_tensor_name is None:
            continue

        # If the predecessor is itself elided, follow the chain.
        if pred_output_tensor_name in aliases:
            up = aliases[pred_output_tensor_name]
            # Compose: out_to_in points at our input positions; those map
            # via the predecessor's alias.out_to_in to the chain root.
            composed: List[Optional[int]] = []
            for j in range(len(out_to_in)):
                k = out_to_in[j]
                if k is None:
                    composed.append(None)
                else:
                    if k < len(up.out_to_in):
                        composed.append(up.out_to_in[k])
                    else:
                        composed.append(None)
            composed_i2o: List[List[int]] = [[] for _ in range(len(up.root_dims))]
            for j, i in enumerate(composed):
                if i is not None and 0 <= i < len(composed_i2o):
                    composed_i2o[i].append(j)
            aliases[my_output_tensor_name] = _Alias(
                root_tensor=up.root_tensor,
                root_layer=up.root_layer,
                root_role=up.root_role,
                root_dims=list(up.root_dims),
                root_shape=list(up.root_shape),
                in_to_out=composed_i2o,
                out_to_in=composed,
            )
        else:
            aliases[my_output_tensor_name] = _Alias(
                root_tensor=pred_output_tensor_name,
                root_layer=pred_name,
                root_role=pred_out_role,
                root_dims=pred_out_dims,
                root_shape=pred_out_shape,
                in_to_out=in_to_out,
                out_to_in=out_to_in,
            )
        elided.add(name)

    return aliases, elided, diagnostics


def _apply_shape_op_elision(layers: Dict[str, dict]
                             ) -> Tuple[Dict[str, dict], List[str]]:
    """Return a rewritten layers dict with shape ops elided.

    For each elide-able layer S:
      - drop S from the output dict
      - for every consumer C whose ``connections.inputs`` lists S, replace
        the entry with the chain root
      - for the consumer operand reading from S, rewrite its dim list to
        align positionally with the root producer's output dims (insert a
        synthetic label for unsqueeze-introduced output dims that the
        consumer carried; drop labels for squeeze-dropped axes; reorder
        for transposes/permutations)
    """
    aliases, elided, diags = _build_shape_op_aliases(layers)
    if not elided:
        return layers, diags

    new_layers: Dict[str, dict] = {}
    rewrite_seq = 0

    for name, L in layers.items():
        if name in elided:
            continue
        new_L = copy.deepcopy(L)
        preds = list((new_L.get("connections") or {}).get("inputs") or [])
        operands = new_L.get("operands") or {}
        tensor_names = (new_L.get("tensor_names") or {}).get("inputs") or []
        tensor_shapes = (new_L.get("tensor_shapes") or {}).get("inputs") or []
        tensor_dtypes_in = (new_L.get("tensor_dtypes") or {}).get("inputs") or []

        # Every role that isn't an explicit output role is an input-like slot
        # (covers Input, Input_1, Weight, Target, Hidden_in, etc.).
        input_roles = [r for r in operands if not _is_output_role(r)]
        # Walk input slot k (1:1 with preds[k] / tensor_names[k] / shapes[k]).
        for k in range(len(preds)):
            pred_name = preds[k]
            # Walk the alias chain: if pred itself produces an elided
            # tensor we substitute its root.
            in_tensor_name = (tensor_names[k] if k < len(tensor_names) else None)
            if in_tensor_name is None or in_tensor_name not in aliases:
                continue
            alias = aliases[in_tensor_name]
            preds[k] = alias.root_layer
            if k < len(tensor_names):
                tensor_names[k] = alias.root_tensor
            if k < len(tensor_shapes):
                tensor_shapes[k] = list(alias.root_shape)
            # Rewrite operand labels for this input slot.
            if k < len(input_roles):
                role = input_roles[k]
                cur_dims = list(operands.get(role) or [])
                if len(cur_dims) != len(alias.out_to_in):
                    # Defensive: skip this rewrite (shouldn't happen given
                    # solar's positional convention).
                    diags.append(
                        f"layer {name!r}: alias-rewrite skipped for role "
                        f"{role!r} — operand width {len(cur_dims)} != alias "
                        f"width {len(alias.out_to_in)}.")
                    continue
                new_dims: List[Optional[str]] = [None] * len(alias.root_dims)
                for j, i in enumerate(alias.out_to_in):
                    if i is None or i < 0 or i >= len(new_dims):
                        continue
                    # When multiple output positions map to the same input
                    # (broadcast) — pick the first; the iter at the root
                    # axis is the same dim across the broadcast.
                    if new_dims[i] is None:
                        new_dims[i] = cur_dims[j]
                # Slot in synthetic labels for root positions that the
                # consumer doesn't iterate (because the shape-op squeezed
                # them out / they're size-1).
                for i, label in enumerate(new_dims):
                    if label is None:
                        new_dims[i] = f"squeeze_{name}_{rewrite_seq}_{i}"
                        rewrite_seq += 1
                operands[role] = [str(x) for x in new_dims]

        new_L["operands"] = operands
        # Write rewritten connections / tensor_names / shapes / dtypes back.
        if (new_L.get("connections") or {}).get("inputs") is not None:
            new_L["connections"]["inputs"] = preds
        if (new_L.get("tensor_names") or {}).get("inputs") is not None:
            new_L["tensor_names"]["inputs"] = tensor_names
        if (new_L.get("tensor_shapes") or {}).get("inputs") is not None:
            new_L["tensor_shapes"]["inputs"] = tensor_shapes

        # Rewrite outgoing-connection lists: each elided layer's name in a
        # successor's connections.outputs should be replaced by the
        # surviving successor or the root layer's successors. We only
        # touch `connections.outputs` for symmetry; AF doesn't read it.
        outs = list((new_L.get("connections") or {}).get("outputs") or [])
        outs = [o for o in outs if o not in elided]
        if (new_L.get("connections") or {}).get("outputs") is not None:
            new_L["connections"]["outputs"] = outs

        _ = tensor_dtypes_in  # currently no dtype rewrite needed; root retains
        new_layers[name] = new_L

    return new_layers, diags


# ---------------------------------------------------------------------------
# Operand normalization (multi-input correctness gate)
# ---------------------------------------------------------------------------


def _normalize_operands(layers: Dict[str, dict]) -> List[str]:
    """Make ``operands`` reflect the true tensor slot count per layer.

    Solar's shape-handlers sometimes emit a single ``Input`` operand role
    for ops that actually consume N tensors (cat, concat, stack). Every
    downstream pass — context indexing, axis collection, cross-layer
    union, AF emit — iterates ``operands`` and would silently drop the
    extra preds. We fix that here, once, by making operands the canonical
    projection-shape map: one role per real input/output tensor slot.

    Source of truth (per AF builder contract):
      - ``tensor_shapes.inputs[k]``, ``tensor_types.inputs[k]`` — slot k's
        shape and role (``"input"`` vs ``"weight"``)
      - ``tensor_shapes.outputs[k]`` — output slot k's shape

    For each layer:
      - If existing input-role count < slot count, synthesize ``Input_k``
        (or ``Weight_k`` per tensor_types) for the missing slots.
      - Same for output roles.

    Synthesized labels are derived from the first existing same-kind role
    as a template, with a per-slot suffix on dims whose size differs from
    the template's size at the same position. Equal-sized dims keep the
    template's label so the union-find correctly merges them (e.g.
    add(x,x) has all dims unified; cat's cat-axis is split). Without a
    template, fresh labels are minted per-dim.

    Mutates ``layers`` in place. Returns diagnostic strings for the
    synthesized roles.
    """
    diags: List[str] = []

    for name, L in layers.items():
        operands = L.get("operands")
        if not operands:
            continue
        in_shapes = (L.get("tensor_shapes") or {}).get("inputs") or []
        in_types = (L.get("tensor_types") or {}).get("inputs") or []
        out_shapes = (L.get("tensor_shapes") or {}).get("outputs") or []

        # Classify each existing role by the same rule
        # ``_build_role_to_shape_index`` uses, so unconventionally-named
        # roles like the entry-point ``start`` (which functions as an
        # output) are recognized and don't trigger spurious synthesis.
        in_roles_existing: List[str] = []
        out_roles_existing: List[str] = []
        n_in_max = max(len(in_shapes), len(in_types))
        for role in operands:
            if _is_output_role(role):
                out_roles_existing.append(role)
            elif _is_input_role(role):
                in_roles_existing.append(role)
            else:
                # Default: fill remaining input slots first, then outputs.
                if len(in_roles_existing) < n_in_max:
                    in_roles_existing.append(role)
                elif len(out_roles_existing) < len(out_shapes):
                    out_roles_existing.append(role)
                else:
                    in_roles_existing.append(role)

        def _synthesize(slot: int, role_name: str, slot_shape: List[int],
                         tmpl_dims: List[str], tmpl_shape: List[int],
                         suffix: str) -> None:
            while role_name in operands:
                role_name += "_x"
            if (tmpl_dims and tmpl_shape and slot_shape
                    and len(tmpl_dims) == len(slot_shape)
                    and len(tmpl_shape) == len(slot_shape)):
                labels = list(tmpl_dims)
                for d in range(len(labels)):
                    if int(slot_shape[d]) != int(tmpl_shape[d]):
                        labels[d] = f"{labels[d]}_{suffix}{slot}"
            elif slot_shape:
                labels = [f"{role_name}_d{d}" for d in range(len(slot_shape))]
            else:
                diags.append(
                    f"layer {name!r}: cannot synthesize role {role_name!r} "
                    f"— no shape info available for slot {slot}.")
                return
            operands[role_name] = labels
            diags.append(
                f"layer {name!r}: synthesized {role_name}={labels} "
                f"for missing slot {slot} (shape={slot_shape}).")

        # Input slots — only synthesize NON-WEIGHT slots. Weight slots not
        # declared in operands are still correctly emitted as ``W{n}`` by
        # ``_emit_af_workload`` (it inspects tensor_types positionally);
        # adding phantom Weight_k roles for tensors that don't participate
        # in the current einsum's compute inflates the AF mapper's
        # pmapping space and causes OOMs (L2/13: ConvTranspose3d with
        # bias). Real multi-input ops (cat/concat/stack) only need
        # additional non-weight roles.
        tmpl_in_dims = (list(operands.get(in_roles_existing[0]) or [])
                        if in_roles_existing else [])
        tmpl_in_shape = list(in_shapes[0]) if in_shapes else []
        for slot in range(len(in_roles_existing), len(in_types) or n_in_max):
            if slot < len(in_types) and in_types[slot] == "weight":
                continue
            slot_shape = list(in_shapes[slot]) if slot < len(in_shapes) else []
            _synthesize(slot, f"Input_{slot}", slot_shape,
                        tmpl_in_dims, tmpl_in_shape, suffix="s")

        # Output slots.
        tmpl_out_dims = (list(operands.get(out_roles_existing[0]) or [])
                         if out_roles_existing else [])
        tmpl_out_shape = list(out_shapes[0]) if out_shapes else []
        for slot in range(len(out_roles_existing), len(out_shapes)):
            slot_shape = list(out_shapes[slot])
            _synthesize(slot, f"Output_{slot}", slot_shape,
                        tmpl_out_dims, tmpl_out_shape, suffix="o")

    return diags


# ---------------------------------------------------------------------------
# Post-emit coverage check (safety net against future silent drops)
# ---------------------------------------------------------------------------


def _validate_af_coverage(af: dict, layers: Dict[str, dict]) -> None:
    """Assert every non-weight Solar pred appears in its consumer's AF einsum.

    Walks ``tensor_types.inputs`` to determine which preds correspond to
    non-weight slots — weight preds (parameter-tensor) are synthesized as
    ``W{n}`` in the AF emit and don't appear by their original name. Each
    non-weight pred's sanitized layer name must appear in the consumer's
    non-output tensor_accesses. Raises on any drop.

    This is the safety net for the cat/concat/stack class of bugs where
    solar emits a unary operand block for a multi-input op and the AF
    emit silently drops the extra preds.
    """
    einsums_by_name = {e["name"]: e for e in af["workload"]["einsums"]}
    errors: List[str] = []
    for layer_name, L in layers.items():
        sanitized = _sanitize(layer_name)
        e = einsums_by_name.get(sanitized)
        if e is None:
            continue  # elided by shape-op pass, not a coverage error.
        preds = (L.get("connections") or {}).get("inputs") or []
        if not preds:
            continue
        in_types = (L.get("tensor_types") or {}).get("inputs") or []
        # Walk slots: pred_index advances only for non-weight slots, mirroring
        # the AF emit's pred-consumption rule. Each non-weight pred must
        # appear by its sanitized name in the consumer's reads.
        expected: List[str] = []
        pred_index = 0
        for slot, slot_type in enumerate(in_types):
            if slot_type == "weight":
                continue
            if pred_index < len(preds):
                expected.append(_sanitize(preds[pred_index]))
            pred_index += 1
        # When tensor_types is shorter than preds (older graphs), assume
        # remaining preds are non-weight.
        for k in range(len(in_types), len(preds)):
            expected.append(_sanitize(preds[k]))
        read_names = {ta["name"] for ta in e["tensor_accesses"]
                      if not ta.get("output")}
        missing = [p for p in expected if p not in read_names]
        if missing:
            errors.append(
                f"layer {layer_name!r} (einsum {sanitized!r}): non-weight "
                f"preds {missing} missing from AF tensor_accesses; "
                f"reads={sorted(read_names)}")
    if errors:
        raise RuntimeError(
            "AF graph coverage check failed:\n  " + "\n  ".join(errors))


def _ranks_of_projection(proj: Any) -> Tuple[str, ...]:
    """Ordered uppercase canonical-rank names referenced by a projection."""
    if isinstance(proj, list):
        return tuple(str(v).upper() for v in proj)
    if isinstance(proj, dict):
        return tuple(str(k).upper() for k in proj.keys())
    return ()


def _validate_graph_invariants(af: dict) -> None:
    """Hard-fail correctness gate on the emitted AF workload.

    Checks the invariants the union-find construction is supposed to
    guarantee, so any future regression surfaces immediately instead of
    silently producing a wrong energy number:

    1. **One rank tuple per tensor.** Every access of a given tensor name
       must reference the same ordered rank tuple. Catches the ghost-scalar
       producer/consumer mismatch, cross-layer rank divergence, and the
       multi-input (cat/concat) ordering class.
    2. **Bounded ranks.** Every rank referenced in a projection exists in
       ``rank_sizes`` (so AF's ISL can bound the operation space).
    3. **No orphan reads.** Every non-output tensor read is either produced
       by some einsum, a synthesized weight (``W<n>``), or a synthetic
       model-input source (``*_in`` read inside an entry-point copy einsum).

    Raises ``RuntimeError`` listing every violation.
    """
    workload = af["workload"]
    einsums = workload["einsums"]
    rank_sizes = workload.get("rank_sizes") or {}
    errors: List[str] = []

    tuples_by_name: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    producers: Set[str] = set()
    for e in einsums:
        for ta in e["tensor_accesses"]:
            tuples_by_name[ta["name"]].add(_ranks_of_projection(ta["projection"]))
            if ta.get("output"):
                producers.add(ta["name"])

    for name, tset in tuples_by_name.items():
        if len(tset) > 1:
            errors.append(
                f"tensor {name!r} has inconsistent rank tuples across "
                f"accesses: {sorted(tset)}")

    for e in einsums:
        for ta in e["tensor_accesses"]:
            for r in _ranks_of_projection(ta["projection"]):
                if r not in rank_sizes:
                    errors.append(
                        f"einsum {e['name']!r} tensor {ta['name']!r} references "
                        f"rank {r!r} absent from rank_sizes")

    for e in einsums:
        is_copy = e.get("is_copy_operation")
        for ta in e["tensor_accesses"]:
            if ta.get("output"):
                continue
            nm = str(ta["name"])
            if nm in producers or re.match(r"^W\d+$", nm):
                continue
            if is_copy and nm.endswith("_in"):
                continue
            # torchview placeholders for tensors created by untraced ops
            # (e.g. BERT position-ids from ``torch.arange``) surface as
            # ``*hidden-tensor`` / ``*auxiliary-tensor`` and are legitimate
            # producerless external sources, not dropped edges (a dropped
            # edge has a producer and is repaired in pytorch_to_einsum).
            if re.search(r"(hidden|auxiliary)[-_]tensor", nm):
                continue
            errors.append(
                f"einsum {e['name']!r} reads {nm!r} which has no producer "
                f"(orphan read)")

    if errors:
        raise RuntimeError(
            "AF graph invariant check failed:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Topological-order normalization
# ---------------------------------------------------------------------------


def _topological_sort_layers(layers: Dict[str, dict]) -> Dict[str, dict]:
    """Return ``layers`` in topological order (Kahn's algorithm).

    Solar's stage-2 output isn't always topo-sorted — e.g. RMSNorm emits
    ``Model.div`` before ``Model.sqrt`` even though div depends on sqrt.
    AF requires producer-before-consumer order; we enforce it here.
    Ties broken by insertion order — stable & deterministic.
    """
    in_deg: Dict[str, int] = {name: 0 for name in layers}
    deps_of: Dict[str, Set[str]] = {name: set() for name in layers}
    successors_of: Dict[str, List[str]] = {name: [] for name in layers}
    for name, L in layers.items():
        preds = (L.get("connections") or {}).get("inputs") or []
        for p in preds:
            if p in layers and p not in deps_of[name]:
                deps_of[name].add(p)
                successors_of[p].append(name)
                in_deg[name] += 1
    ready = [n for n in layers if in_deg[n] == 0]
    result: Dict[str, dict] = {}
    while ready:
        cur = ready.pop(0)
        result[cur] = layers[cur]
        for succ in successors_of[cur]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                ready.append(succ)
    # Cycle / missing node fallback — keep all data, surface to AF.
    if len(result) != len(layers):
        for name in layers:
            if name not in result:
                result[name] = layers[name]
    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_af_graph_from_dict(einsum_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Build the AccelForge einsum graph from an in-memory stage-2 dict.

    The expected input shape is what ``PyTorchToEinsum._build_einsum_graph``
    returns: a dict containing ``"layers": {layer_name: {operands, ...}}``
    plus optional metadata like ``"model_name"``.
    """
    layers = einsum_graph.get("layers") or {}
    if not layers:
        raise ValueError("einsum_graph has no layers")

    layers = _topological_sort_layers(layers)
    layers, elision_diags = _apply_shape_op_elision(layers)
    # Normalize ``operands`` so every real tensor slot has exactly one role
    # — this is the correctness gate that fixes silent multi-input drops
    # (cat/concat/stack) at the AF boundary.
    norm_diags = _normalize_operands(layers)

    ctx = BuildContext(layers=layers)
    if elision_diags:
        ctx.diagnostics.extend(elision_diags)
    if norm_diags:
        ctx.diagnostics.extend(norm_diags)
    _build_role_to_shape_index(layers, ctx)
    _collect_axes(ctx)
    _within_layer_union(ctx)
    _cross_layer_union(ctx)
    _assign_canonical_names(ctx)
    af = _emit_af_workload(ctx, einsum_graph.get("model_name", "model"))
    _ghost_scalar_outputs(af)
    # Note: the historical ``_correct_output_bits`` post-emit dtype repair
    # is no longer needed here — torchview's fp32-override-on-bf16
    # quirk is now repaired at the ``layers`` stage by
    # ``PyTorchToEinsum._repair_torchview_quirks`` (sub-pass C), so the
    # ``tensor_dtypes`` we ingest are already correct.
    # Post-emit coverage: every Solar pred must end up read by its
    # consumer's AF einsum. Raises on any silent drop.
    _validate_af_coverage(af, layers)
    # Hard-fail correctness gate: rank-tuple consistency, bounded ranks,
    # no orphan reads.
    _validate_graph_invariants(af)
    # Re-derive top-level ``bits_per_value: {All: <max>}`` from the
    # post-emit per-access bits so the energy fallback for tensors without
    # an explicit ``bits_per_value`` annotation matches the widest precision
    # actually used by the workload.
    corrected = [ta.get("bits_per_value")
                  for e in af["workload"]["einsums"]
                  for ta in e["tensor_accesses"]
                  if ta.get("bits_per_value") is not None]
    if corrected:
        af["workload"]["bits_per_value"]["All"] = max(corrected)
    if ctx.diagnostics:
        af["_diagnostics"] = list(ctx.diagnostics)
    return af


def build_af_graph_from_yaml(einsum_graph_yaml: Union[Path, str],
                              output_path: Optional[Union[Path, str]] = None
                              ) -> Dict[str, Any]:
    """Build the AccelForge einsum graph from a stage-2 YAML on disk.

    Args:
        einsum_graph_yaml: Path to ``einsum_graph.yaml`` (stage-2 output).
        output_path: Optional path to write the resulting AF YAML to.

    Returns:
        Dict with ``workload`` and ``renames`` keys.
    """
    path = Path(einsum_graph_yaml)
    with open(path) as f:
        graph = yaml.safe_load(f)
    af = build_af_graph_from_dict(graph)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        af_to_write = {k: v for k, v in af.items() if not k.startswith("_")}
        with open(out, "w") as f:
            yaml.dump(af_to_write, f, default_flow_style=False, sort_keys=False)
    return af


# Backward-compatible alias matching the e1 builder's name.
build_af_einsum_graph = build_af_graph_from_yaml
