# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic binding of DSL module constants to benchmark definition axes.

Campaign-audit L2/058: a DSL saved from a capped verification session baked
``_G = 2`` while the definition declared ``n_groups = 8``; the full-shape
re-trace executed and copied the DSL verbatim, and the saved graph cannot
even express the group count, so no artifact-level check existed.

Contract: when the definition declares const axes, every module-level int
constant that reaches a ``Dim`` extent must carry explicit provenance in
the DSL:

    _AXIS_BINDINGS = {"_G": "n_groups", "_HD": "head_dim"}   # semantic
    _FREE_CONSTANTS = ["_TILE"]                              # independent

Bound constants are CHECKED against the definition's axis values —
name-similarity is never used, and a constant the author does not declare
fails the full-shape trace instead of silently keeping its capped value.
Legitimate independent tiling constants are preserved by declaring them
free (recorded, not checked).
"""

from __future__ import annotations

import ast
from typing import Any, Dict


class ConstantBindingError(ValueError):
    """A required constant binding is missing, dangling, or mismatched."""


def _literal_dict(node: ast.expr):
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            if isinstance(v, ast.Constant) and (
                    isinstance(v.value, str) or v.value is None):
                out[k.value] = v.value
            else:
                return None
        return out
    return None


def _literal_str_seq(node: ast.expr):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = []
        for e in node.elts:
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                return None
            vals.append(e.value)
        return vals
    return None


def _dim_extent_const_names(tree: ast.Module, int_consts: Dict[str, int]):
    """Module int constants reachable (transitively through local assigns)
    from any ``Dim(...)`` size expression."""
    assign_rhs_names: Dict[str, set] = {}
    size_names: set = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            assign_rhs_names[node.targets[0].id] = {
                n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        if (isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name) and node.func.id == "Dim")
                     or (isinstance(node.func, ast.Attribute)
                         and node.func.attr == "Dim"))):
            for kw in node.keywords:
                if kw.arg == "size":
                    size_names |= {n.id for n in ast.walk(kw.value)
                                   if isinstance(n, ast.Name)}
            if len(node.args) >= 2:  # positional size
                size_names |= {n.id for n in ast.walk(node.args[1])
                               if isinstance(n, ast.Name)}
    # transitive closure backward through local assignments
    reach = set(size_names)
    changed = True
    while changed:
        changed = False
        for name in list(reach):
            for dep in assign_rhs_names.get(name, ()):
                if dep not in reach:
                    reach.add(dep)
                    changed = True
    return sorted(reach & set(int_consts))


def verify_constant_bindings(code: str, defn: dict, *,
                             allow_unbound: bool = False) -> Dict[str, Any]:
    """Check the DSL's Dim-extent module constants against *defn*.

    Returns a provenance record {const_name: {...}}.  Raises
    ConstantBindingError on a missing or dangling binding or a value
    mismatch (unless ``allow_unbound`` downgrades the missing-binding case
    to a recorded warning — mismatches always fail).
    """
    tree = ast.parse(code)
    int_consts: Dict[str, int] = {}
    axis_bindings: Dict[str, Any] = {}
    free_constants: list = []
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            tname = stmt.targets[0].id
            if isinstance(stmt.value, ast.Constant) and isinstance(
                    stmt.value.value, int) and not isinstance(
                    stmt.value.value, bool):
                int_consts[tname] = stmt.value.value
            elif tname == "_AXIS_BINDINGS":
                d = _literal_dict(stmt.value)
                if d is None:
                    raise ConstantBindingError(
                        "_AXIS_BINDINGS must be a literal {str: str|None} dict")
                axis_bindings = d
            elif tname == "_FREE_CONSTANTS":
                s = _literal_str_seq(stmt.value)
                if s is None:
                    raise ConstantBindingError(
                        "_FREE_CONSTANTS must be a literal list/tuple/set of "
                        "str")
                free_constants = s

    const_axes = {name: spec.get("value")
                  for name, spec in (defn.get("axes") or {}).items()
                  if isinstance(spec, dict) and spec.get("type") == "const"}

    used = _dim_extent_const_names(tree, int_consts)
    provenance: Dict[str, Any] = {}
    problems = []
    for name in used:
        value = int_consts[name]
        if name in free_constants:
            provenance[name] = {"value": value, "binding": "free",
                                "source": "_FREE_CONSTANTS"}
            continue
        if name in axis_bindings:
            axis = axis_bindings[name]
            if axis is None:
                provenance[name] = {"value": value, "binding": "free",
                                    "source": "_AXIS_BINDINGS: None"}
                continue
            if axis not in const_axes:
                problems.append(
                    f"{name} is bound to definition axis {axis!r}, which is "
                    f"not a declared const axis")
                continue
            if int(const_axes[axis]) != int(value):
                problems.append(
                    f"baked constant {name}={value} != definition axis "
                    f"{axis}={const_axes[axis]} (capped value survived into "
                    f"the full-shape DSL)")
                continue
            provenance[name] = {"value": value, "binding": axis,
                                "definition_value": const_axes[axis],
                                "source": "_AXIS_BINDINGS"}
            continue
        if const_axes:
            msg = (f"module constant {name}={value} feeds a Dim extent but "
                   f"carries no provenance; declare it in _AXIS_BINDINGS "
                   f"(semantic, checked against the definition) or "
                   f"_FREE_CONSTANTS (independent tiling)")
            if allow_unbound:
                provenance[name] = {"value": value, "binding": "unbound",
                                    "source": "allow_unbound", "warning": msg}
            else:
                problems.append(msg)
        else:
            provenance[name] = {"value": value, "binding": "free",
                                "source": "definition-has-no-const-axes"}

    # Dangling declared bindings are also author errors.
    for name, axis in axis_bindings.items():
        if name not in int_consts and axis is not None:
            problems.append(
                f"_AXIS_BINDINGS names {name!r}, which is not a module int "
                f"constant")

    if problems:
        raise ConstantBindingError(
            "constant binding verification failed:\n  - "
            + "\n  - ".join(problems))
    return provenance
