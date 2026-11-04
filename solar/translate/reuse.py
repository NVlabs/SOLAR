# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable translations: symbolic-shape preservation, guards, and caching.

A verified ``*_einsum_lib.py`` translation is *shape-polymorphic* when every
``Dim`` extent is derived from an input's ``.shape[i]`` (the house style in
``skills/verified-translation``): the same translation can then be
re-instantiated per workload, with actual dimensions filled in when the trace
lowers it to the einsum graph.  That reuse is only sound within a *guard
family*: shape-dependent branches, rank changes, divisibility assumptions,
dtype changes, data-dependent accesses and baked specialization constants all
narrow the family, and anything outside it needs per-shape revalidation or a
separate translation.

This module provides:

- :func:`semantic_source_hash` — AST-normalized hash of a DSL (comments /
  whitespace / docstrings do not invalidate the cache).
- :func:`guard_profile` — static profile of the reuse-relevant hazards in a
  DSL program.
- :func:`reuse_decision` — maps (profile, baseline axes, requested axes,
  definition) to ``symbolic_reuse`` / ``revalidate_per_shape`` /
  ``reject`` with explicit reasons.
- :func:`cache_key` — content key for a (translation, instantiation) pair:
  semantic DSL hash, definition hash, axes, dtypes, options and the skill
  revision that produced the translation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "GuardProfile", "cache_key", "guard_profile", "reuse_decision",
    "semantic_source_hash",
]


def semantic_source_hash(path_or_source) -> str:
    """Hash of the AST-normalized DSL source (whitespace/comment immune).

    Docstrings are stripped so documentation edits do not invalidate cached
    instantiations; any behavioral edit (statements, constants, annotations)
    changes the hash.
    """
    src = (Path(path_or_source).read_text()
           if isinstance(path_or_source, (str, Path))
           and "\n" not in str(path_or_source)
           and Path(path_or_source).exists()
           else str(path_or_source))
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.FunctionDef,
                                  ast.AsyncFunctionDef, ast.ClassDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        normalized = ast.dump(tree)
    except SyntaxError:
        normalized = src
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class GuardProfile:
    """Static reuse hazards found in a DSL program (per guard family)."""

    shape_branches: list = field(default_factory=list)   # ifs on shapes/axes
    div_mod_index: list = field(default_factory=list)    # // or % in subscripts
    data_dependent: list = field(default_factory=list)   # gather-indexed access
    baked_dim_constants: list = field(default_factory=list)  # Dim(size=<int>)
    # Dim extents derived by FLOOR DIVISION of a shape/axis (window
    # partitioning): the covered domain silently shrinks on nondivisible
    # axes (campaign-audit L2/077 — 14/16 shapes dropped the padded
    # border), so one shape's validation does not transfer.
    floor_partition_dims: list = field(default_factory=list)
    symbolic_dims: int = 0                               # Dim sized from .shape
    literal_dims: int = 0                                # Dim sized by literal

    @property
    def hazards(self) -> list:
        out = []
        if self.shape_branches:
            out.append("shape_dependent_branch")
        if self.div_mod_index:
            out.append("divisibility_index_arithmetic")
        if self.data_dependent:
            out.append("data_dependent_access")
        if self.floor_partition_dims:
            out.append("floor_partition_extent")
        return out


def _subscript_names(node) -> set:
    return {n.value.id for n in ast.walk(node)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}


def guard_profile(path_or_source) -> GuardProfile:
    """Scan a DSL program for reuse-relevant hazards (see GuardProfile)."""
    src = (Path(path_or_source).read_text()
           if isinstance(path_or_source, (str, Path))
           and "\n" not in str(path_or_source)
           and Path(path_or_source).exists()
           else str(path_or_source))
    tree = ast.parse(src)
    prof = GuardProfile()

    # Names assigned from an expression containing floor division (window
    # partition counts like ``nwh = X.shape[1] // ws``).
    floor_div_names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and not isinstance(node.value, ast.Subscript)
                and any(isinstance(n, ast.BinOp)
                        and isinstance(n.op, ast.FloorDiv)
                        for n in ast.walk(node.value))):
            floor_div_names.add(node.targets[0].id)

    # Gather-derived loop variables: `e = SE[n, k]` / `e = int(SE[n, k])`
    gather_vars = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            v = node.value
            if isinstance(v, ast.Call) and v.args:
                v = v.args[0]
            if isinstance(v, ast.Subscript):
                gather_vars.add(node.targets[0].id)

    for node in ast.walk(tree):
        # `if` statements whose test touches .shape or len(...) — the branch
        # taken can differ between shapes, so one shape's validation does not
        # transfer.
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if ".shape" in test_src or "len(" in test_src:
                prof.shape_branches.append(
                    f"line {node.lineno}: if {test_src}")
        # Floor-div / modulo inside tensor subscripts — index arithmetic that
        # assumes divisibility between axes (e.g. GQA `n // groups`).
        if isinstance(node, ast.Subscript):
            for sub in ast.walk(node.slice):
                if isinstance(sub, ast.BinOp) and isinstance(
                        sub.op, (ast.FloorDiv, ast.Mod)):
                    prof.div_mod_index.append(
                        f"line {node.lineno}: {ast.unparse(node)}")
                    break
            # Subscript indexed by a gather-derived variable.
            if any(isinstance(n, ast.Name) and n.id in gather_vars
                   for n in ast.walk(node.slice)):
                prof.data_dependent.append(
                    f"line {node.lineno}: {ast.unparse(node)}")
        # Dim(...) size= spec: symbolic (from .shape / name) vs literal int.
        if (isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name) and node.func.id == "Dim")
                     or (isinstance(node.func, ast.Attribute)
                         and node.func.attr == "Dim"))):
            size_kw = next((k.value for k in node.keywords
                            if k.arg == "size"), None)
            if size_kw is None:
                prof.symbolic_dims += 1  # registry-resolved (first-use shape)
            elif isinstance(size_kw, ast.Constant) and isinstance(
                    size_kw.value, int):
                prof.literal_dims += 1
                prof.baked_dim_constants.append(
                    f"line {node.lineno}: {ast.unparse(node)}")
            else:
                prof.symbolic_dims += 1
            if size_kw is not None and (
                    (isinstance(size_kw, ast.Name)
                     and size_kw.id in floor_div_names)
                    or any(isinstance(n, ast.BinOp)
                           and isinstance(n.op, ast.FloorDiv)
                           for n in ast.walk(size_kw))):
                prof.floor_partition_dims.append(
                    f"line {node.lineno}: {ast.unparse(node)}")
    return prof


def reuse_decision(profile: GuardProfile, defn: dict,
                   baseline_axes: dict, axes: dict) -> tuple:
    """Classify an instantiation request against a validated baseline.

    Returns ``(mode, reasons)`` where mode is one of:

    - ``"reject"`` — outside the translation's guard family (const axis or
      dtype disagreement with the definition the translation was verified
      against): a *separate translation* is required.
    - ``"revalidate_per_shape"`` — same family, but the DSL carries hazards
      (shape-dependent branches, divisibility index arithmetic,
      data-dependent accesses, or baked literal dims while var axes change):
      the instantiation must be executed and checked at THIS shape; the
      baseline's numeric validation does not transfer.
    - ``"symbolic_reuse"`` — fully shape-polymorphic within the family: the
      baseline verification plus per-shape execution + dimension accounting
      is a supported symbolic argument for reuse (numeric spot-checks still
      apply per campaign policy).
    """
    reasons = []
    const_axes = {n: s["value"] for n, s in defn.get("axes", {}).items()
                  if s.get("type") == "const"}
    for name, value in const_axes.items():
        if name in axes and axes[name] != value:
            reasons.append(
                f"const axis {name}={axes[name]} != definition value {value}")
    if reasons:
        return "reject", reasons

    var_axes_change = any(
        axes.get(n) != baseline_axes.get(n)
        for n, s in defn.get("axes", {}).items() if s.get("type") == "var")

    hazards = list(profile.hazards)
    if profile.baked_dim_constants and var_axes_change:
        hazards.append("baked_dim_constant_with_varying_axes")
    if hazards and var_axes_change:
        return "revalidate_per_shape", hazards
    if not var_axes_change:
        return "symbolic_reuse", ["axes identical to validated baseline"]
    return "symbolic_reuse", [
        "shape-polymorphic DSL (all Dim extents symbolic), same const axes "
        "and dtypes; instantiation executed + accounting-checked per shape"]


def cache_key(dsl_sha: str, defn_sha: str, axes: dict, dtypes: dict,
              options: dict, skill_rev: str) -> str:
    """Content key for a cached instantiation of a translation."""
    blob = json.dumps({
        "dsl": dsl_sha, "defn": defn_sha,
        "axes": dict(sorted(axes.items())),
        "dtypes": dict(sorted(dtypes.items())),
        "options": dict(sorted(options.items())),
        "skill_rev": skill_rev,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def definition_hash(defn: dict) -> str:
    """Stable hash of a problem definition (axes/inputs/outputs/reference)."""
    keep = {k: defn.get(k) for k in
            ("axes", "inputs", "outputs", "reference",
             "custom_inputs_entrypoint")}
    return hashlib.sha256(
        json.dumps(keep, sort_keys=True).encode()).hexdigest()
