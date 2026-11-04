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

"""Generic resolver for declared shape dimensions in problem definitions.

Moved verbatim (2026-11-04) from the retired benchmark package
``solar.benchmark.solbenchv3.parser`` (``SolBenchV3Parser.resolve_dimension``
and its module-level helpers), because ``solar.translate.sol`` resolves the
declared output shapes of a ``definition.json`` with it.  Behaviour is
unchanged: the same inputs give the same integers and the same ``ValueError``
messages.  A dimension spec may be an ``int``/``float`` literal, an axis name,
a bounded arithmetic expression over axis names (``"num_heads * head_dim"``),
``{"expr": "..."}``, an operator dict (``{"mul": [...]}``) or an infix list
(``["num_heads", "*", "head_dim"]``).  No function calls, attribute access or
other constructs are evaluated.
"""

import ast
import functools
import operator
from typing import Any, Dict, List, Union


# Operators allowed in shape expressions
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expr(expr_str: str, namespace: Dict[str, int]) -> int:
    """Safely evaluate an arithmetic expression over axis names.

    Uses Python's ``ast`` module to parse the expression and walks the tree
    with a strict whitelist of allowed node types.  Only integer arithmetic
    on axis names is permitted — no function calls, attribute access, etc.

    Args:
        expr_str: Expression like ``"num_heads * head_dim"`` or ``"head_dim // 2"``.
        namespace: Mapping of axis name → concrete int value.

    Returns:
        The evaluated integer result.

    Raises:
        ValueError: If the expression contains disallowed constructs or
            references an undefined axis name.
    """
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse shape expression: {expr_str!r}") from exc

    def _eval(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return int(node.value)
            raise ValueError(f"Non-numeric constant in expression: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in namespace:
                return namespace[node.id]
            raise ValueError(
                f"Unknown axis name {node.id!r} in expression {expr_str!r}. "
                f"Available axes: {sorted(namespace)}"
            )
        if isinstance(node, ast.BinOp):
            op_fn = _ALLOWED_BINOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed operator {type(node.op).__name__} in {expr_str!r}")
            return int(op_fn(_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp):
            op_fn = _ALLOWED_UNARYOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed unary operator {type(node.op).__name__} in {expr_str!r}")
            return int(op_fn(_eval(node.operand)))
        raise ValueError(
            f"Disallowed AST node {type(node).__name__} in expression {expr_str!r}"
        )

    return _eval(tree)


def _resolve_dict_op(
    d: Dict[str, Any], resolved_axes: Dict[str, int]
) -> int:
    """Resolve operator-based dict dimension specs.

    Supported formats::

        {"mul": ["batch_size", "seq_len"]}
        {"multiply": ["a", "b"]}
        {"div": ["hidden_size", 2]}
        {"divide": ["hidden_size", 2]}
        {"add": ["past_len", "new_len"]}

    Operands can be axis names (str), integer literals, or nested dicts.
    """
    # Determine the operator
    _OP_KEYS = {
        "mul": operator.mul,
        "multiply": operator.mul,
        "div": operator.floordiv,
        "divide": operator.floordiv,
        "add": operator.add,
        "sub": operator.sub,
    }

    for key, op_fn in _OP_KEYS.items():
        if key in d:
            operands = d[key]
            if not isinstance(operands, list) or len(operands) < 2:
                raise ValueError(f"Dict op {key!r} requires a list of ≥2 operands: {d}")
            values = [
                resolve_dimension(op, resolved_axes)
                for op in operands
            ]
            return int(functools.reduce(op_fn, values))

    raise ValueError(f"Unsupported dict dimension format: {d}")


_INFIX_OPS = {"+", "-", "*", "//", "/", "%"}


def _resolve_list_expr(
    lst: List[Any], resolved_axes: Dict[str, int]
) -> int:
    """Resolve infix list-format dimension specs.

    Example: ``["num_heads", "*", "q_head_dim"]`` → ``num_heads * q_head_dim``.
    """
    parts: List[str] = []
    for item in lst:
        if isinstance(item, str) and item.strip() in _INFIX_OPS:
            parts.append(item.strip())
        else:
            # Resolve operand to int, then stringify
            parts.append(str(resolve_dimension(item, resolved_axes)))
    return _safe_eval_expr(" ".join(parts), resolved_axes)


def resolve_dimension(
    dim_spec: Union[int, str, Dict[str, Any]],
    resolved_axes: Dict[str, int],
) -> int:
    """Resolve a single shape dimension to a concrete integer.

    Handles:
        - ``int`` literal → returned as-is
        - ``str`` simple axis name → dict lookup
        - ``str`` arithmetic expression → ``_safe_eval_expr()``
        - ``dict`` with ``"expr"`` key → evaluate the expression string
    """
    if isinstance(dim_spec, (int, float)):
        return int(dim_spec)

    if isinstance(dim_spec, dict):
        if "expr" in dim_spec:
            return _safe_eval_expr(dim_spec["expr"], resolved_axes)
        # Operator-based dict formats: {"mul": [...]}, {"div": [...]}, etc.
        return _resolve_dict_op(dim_spec, resolved_axes)

    if isinstance(dim_spec, list):
        # Infix list format: ["num_heads", "*", "q_head_dim"]
        return _resolve_list_expr(dim_spec, resolved_axes)

    if isinstance(dim_spec, str):
        # Fast path: plain axis name
        if dim_spec in resolved_axes:
            return resolved_axes[dim_spec]
        # Expression path (contains operators or is multi-word)
        return _safe_eval_expr(dim_spec, resolved_axes)

    raise ValueError(f"Unsupported dimension type {type(dim_spec)}: {dim_spec}")

