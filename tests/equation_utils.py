# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for rank-letter-independent einsum equation assertions.

normalize_equation() renames rank tokens in first-seen order (A, B, C, ...)
so that structurally equivalent equations compare equal regardless of the
original token names.  For example:

    normalize_equation("MK,KN->MN")          == "AB,BC->AC"
    normalize_equation("XY,XY->XY")          == "AB,AB->AB"
    normalize_equation("BC(P+R)(Q+S),OCRS->BOPQ")
                                              == "AB(C+D)(E+F),GBDF->AGCE"

Supported grammar: simple alphabetic rank tokens and parenthesised sums or
differences of rank tokens and integer constants, e.g. ``(P+R)`` or ``(X+2)``.
Integer constants inside parentheses are never renamed, so ``(X+2)`` and
``(X+3)`` stay distinct.  A standalone top-level ``1`` is NOT supported:
``parse_dim_tokens`` drops it, so it would silently compare equal to its
absence; such equations raise ValueError instead.
"""

import re
from typing import List

from solar.common.utils import parse_dim_tokens
from solar.einsum.ops.shape_ops import generate_dim_labels


def _extract_atoms(token: str) -> List[str]:
    """Extract atomic rank names from a (possibly compound) token."""
    if token.startswith("("):
        inner = token[1:-1]
        atoms = (a.strip() for a in re.split(r"[+\-]", inner))
        return [a for a in atoms if a and not a.isdigit()]
    if token == "1":
        return []
    return [token]


def normalize_equation(equation: str) -> str:
    """Normalize rank token names to canonical first-seen order.

    Equations in the supported grammar (see module docstring) that differ only
    by a consistent renaming of rank tokens normalize to the same string;
    equations that differ in structure or in an integer constant do not.
    """
    if "->" not in equation:
        return equation
    if re.search(r"\d", re.sub(r"\([^)]*\)", "", equation)):
        raise ValueError(f"standalone numeric token unsupported: {equation!r}")

    lhs, rhs = equation.split("->", 1)
    all_parts = lhs.split(",") + [rhs]

    seen_atoms: List[str] = []
    for part in all_parts:
        tokens = parse_dim_tokens(part.strip()) if part.strip() else []
        for token in tokens:
            for atom in _extract_atoms(token):
                if atom not in seen_atoms:
                    seen_atoms.append(atom)

    canonical = generate_dim_labels(len(seen_atoms))
    mapping = dict(zip(seen_atoms, canonical))

    def _replace_token(token: str) -> str:
        if token.startswith("("):
            inner = token[1:-1]
            parts = re.split(r"([+\-])", inner)
            normalized = "".join(
                mapping.get(p.strip(), p) if p.strip() not in ("+", "-") else p
                for p in parts
            )
            return f"({normalized})"
        if token == "1":
            return "1"
        return mapping.get(token, token)

    result_parts: List[str] = []
    for part in all_parts:
        tokens = parse_dim_tokens(part.strip()) if part.strip() else []
        result_parts.append("".join(_replace_token(t) for t in tokens))

    num_inputs = len(lhs.split(","))
    return ",".join(result_parts[:num_inputs]) + "->" + result_parts[num_inputs]
