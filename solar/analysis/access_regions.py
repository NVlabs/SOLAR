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

"""Static access-region analysis for external tensor reads.

The fused / fused_prefetched SOL model requires the *unique element
footprint* of every external tensor: each external byte is read from DRAM
at most once, so N ops reading regions R_1..R_N of one base tensor
contribute |R_1 ∪ ... ∪ R_N| elements, not Σ|R_i| and not max|R_i|.

This module resolves access regions statically from the graph metadata
that TorchView records in ``raw_attributes`` (index expressions for
``__getitem__``) and from multi-output partition layers (``chunk`` /
``split`` / ``tensor_split`` output shapes).  A region is only produced
when it can be *proven*:

- the index expression parses to basic indexing (ints, step-1 slices,
  ``Ellipsis``, ``None``) with static bounds, or a static list of ints;
- the resulting region lies inside the base tensor bounds;
- the region's element count equals the element count independently
  derived from the op's output shape.

Any failure yields ``None`` and the caller falls back to the conservative
lower-bound accounting (per-tensor ``max`` of reads), which preserves the
SOL lower-bound property.  Regions are axis-aligned boxes (one half-open
interval per dimension); the exact size of a union of boxes is computed
by recursive coordinate compression.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence, Tuple

# One half-open interval per dimension: ((s0, e0), (s1, e1), ...).
Box = Tuple[Tuple[int, int], ...]

# Ops whose output is a view of a sub-region of their first input.
SLICE_VIEW_OPS = frozenset({"__getitem__", "narrow", "slice", "select"})

# Multi-output ops that partition their input along one dimension.
PARTITION_OPS = frozenset({"chunk", "split", "tensor_split"})

# Cap on elementary cells visited by the box-union recursion.  Groups are
# per external tensor, so box counts are O(number of consumers); real
# graphs stay far below this.  On overflow the caller falls back to the
# conservative per-box maximum.
_UNION_CELL_BUDGET = 2_000_000

# Cap on boxes emitted for a static integer-list (fancy) index.
_MAX_FANCY_BOXES = 4096


class _TensorArg:
    """Sentinel for a ``Tensor(...)`` argument in raw_attributes."""

    __slots__ = ()


class _Unsupported:
    """Sentinel for any argument construct outside the static whitelist."""

    __slots__ = ()


_TENSOR_SENTINEL = "__SOLAR_TENSOR_ARG__"


def _mask_tensor_reprs(raw: str) -> Optional[str]:
    """Replace every balanced ``Tensor(...)`` span with a sentinel name.

    Handles nested parentheses (``Tensor(shape=(2, 8), ...)``) and quoted
    strings (``device='cuda:0'``).  Returns None on unbalanced input.
    """
    out: List[str] = []
    i = 0
    n = len(raw)
    while i < n:
        j = raw.find("Tensor(", i)
        if j < 0:
            out.append(raw[i:])
            break
        out.append(raw[i:j])
        k = j + len("Tensor(")
        depth = 1
        quote = ""
        while k < n and depth > 0:
            ch = raw[k]
            if quote:
                if ch == quote:
                    quote = ""
            elif ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            k += 1
        if depth != 0:
            return None
        out.append(_TENSOR_SENTINEL)
        i = k
    return "".join(out)


def _node_to_value(node: ast.AST) -> Any:
    """Convert a whitelisted AST node to a Python value.

    Anything outside the whitelist becomes ``_Unsupported`` (kept in place
    so container structure is preserved for validation by the caller).
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == _TENSOR_SENTINEL:
            return _TensorArg()
        if node.id == "Ellipsis":
            return Ellipsis
        if node.id == "None":
            return None
        if node.id in ("True", "False"):
            return node.id == "True"
        if node.id == "inf":
            return float("inf")
        if node.id == "nan":
            return float("nan")
        return _Unsupported()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _node_to_value(node.operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand
        return _Unsupported()
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_node_to_value(el) for el in node.elts]
    if isinstance(node, ast.Dict):
        result: Dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            elif isinstance(key_node, ast.Name):
                # torchview stringifies kwargs with bare keys: "{dim: 1}".
                key = key_node.id
            else:
                return _Unsupported()
            result[key] = _node_to_value(value_node)
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice"
        and not node.keywords
        and len(node.args) <= 3
    ):
        args = [_node_to_value(a) for a in node.args]
        if all(a is None or (isinstance(a, int) and not isinstance(a, bool)) for a in args):
            return slice(*args)
        return _Unsupported()
    return _Unsupported()


def parse_call_attributes(raw: Any) -> Optional[Tuple[List[Any], Dict[str, Any]]]:
    """Parse a torchview ``raw_attributes`` string into (args, kwargs).

    The recorded format is ``[[<arg0>, <arg1>, ...], {<kwargs>}]`` where
    tensor arguments appear as ``Tensor(shape=..., dtype=...)``.  Tensor
    arguments become ``_TensorArg`` sentinels; constructs outside the
    static whitelist become ``_Unsupported`` sentinels.  Returns None when
    the overall structure cannot be parsed.
    """
    if not isinstance(raw, str) or not raw:
        return None
    masked = _mask_tensor_reprs(raw)
    if masked is None:
        return None
    try:
        tree = ast.parse(masked.strip(), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    root = tree.body
    if not isinstance(root, (ast.List, ast.Tuple)) or len(root.elts) != 2:
        return None
    args_node, kwargs_node = root.elts
    if not isinstance(args_node, (ast.List, ast.Tuple)):
        return None
    if not isinstance(kwargs_node, ast.Dict):
        return None
    args = [_node_to_value(el) for el in args_node.elts]
    kwargs = _node_to_value(kwargs_node)
    if isinstance(kwargs, _Unsupported):
        return None
    return args, kwargs


def _is_static_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_int_index(index: int, dim_size: int) -> Optional[int]:
    if index < 0:
        index += dim_size
    if 0 <= index < dim_size:
        return index
    return None


def _basic_index_box(entries: Sequence[Any], base_shape: Sequence[int]) -> Optional[Box]:
    """Resolve a basic-indexing entry sequence to one box over base_shape.

    Supported entries: static ints, step ±1 slices with static bounds,
    one ``Ellipsis``, ``None`` (newaxis, consumes no base dim).  Slices
    with |step| != 1 select non-contiguous elements and are rejected (the
    caller's fallback already counts them exactly via the output size).
    """
    rank = len(base_shape)
    consuming = [e for e in entries if e is not None and e is not Ellipsis]
    num_ellipsis = sum(1 for e in entries if e is Ellipsis)
    if num_ellipsis > 1 or len(consuming) > rank:
        return None

    # Expand Ellipsis into full slices for the unconsumed middle dims.
    expanded: List[Any] = []
    for entry in entries:
        if entry is Ellipsis:
            expanded.extend([slice(None)] * (rank - len(consuming)))
        elif entry is not None:
            expanded.append(entry)
    expanded.extend([slice(None)] * (rank - len(expanded)))
    if len(expanded) != rank:
        return None

    intervals: List[Tuple[int, int]] = []
    for entry, dim_size in zip(expanded, base_shape):
        if not _is_static_int(dim_size) or dim_size <= 0:
            return None
        if _is_static_int(entry):
            idx = _normalize_int_index(entry, dim_size)
            if idx is None:
                return None
            intervals.append((idx, idx + 1))
        elif isinstance(entry, slice):
            start, stop, step = entry.indices(dim_size)
            if step == 1:
                lo, hi = start, stop
            elif step == -1:
                lo, hi = stop + 1, start + 1
            else:
                return None
            if lo >= hi:
                return None  # empty slice: no read; fall back
            intervals.append((lo, hi))
        else:
            return None
    return tuple(intervals)


def box_size(box: Box) -> int:
    size = 1
    for lo, hi in box:
        size *= hi - lo
    return size


def _fancy_int_list_boxes(
    indices: Sequence[Any], base_shape: Sequence[int]
) -> Optional[List[Box]]:
    """Boxes for a static integer-list gather along dim 0 (``x[[0, 2, 2]]``)."""
    rank = len(base_shape)
    if rank == 0 or not indices or len(indices) > _MAX_FANCY_BOXES:
        return None
    if not all(_is_static_int(i) for i in indices):
        return None
    rest: List[Tuple[int, int]] = []
    for dim_size in base_shape[1:]:
        if not _is_static_int(dim_size) or dim_size <= 0:
            return None
        rest.append((0, dim_size))
    boxes: List[Box] = []
    for raw_idx in indices:
        idx = _normalize_int_index(raw_idx, base_shape[0])
        if idx is None:
            return None
        boxes.append(((idx, idx + 1), *rest))
    return boxes


def _getitem_boxes(
    index_arg: Any, base_shape: Sequence[int], expected_elems: int
) -> Optional[List[Box]]:
    """Resolve a ``__getitem__`` index argument to validated boxes.

    torchview stringifies tuple indices as lists, so ``x[1, 0:4]`` and a
    fancy list index ``x[[1, 3]]`` both arrive as lists.  Both readings
    are attempted; each is validated against ``expected_elems`` (the
    element count from the op's recorded output shape, which for basic
    indexing counts every selected element with its multiplicity).  An
    ambiguous match (both readings validate with different footprints) is
    rejected.
    """
    if isinstance(index_arg, list):
        entries: Sequence[Any] = index_arg
    else:
        entries = [index_arg]

    candidates: List[List[Box]] = []

    box = _basic_index_box(entries, base_shape)
    if box is not None and box_size(box) == expected_elems:
        candidates.append([box])

    if isinstance(index_arg, list):
        fancy = _fancy_int_list_boxes(index_arg, base_shape)
        if fancy is not None and sum(box_size(b) for b in fancy) == expected_elems:
            candidates.append(fancy)

    if not candidates:
        return None
    if len(candidates) == 2:
        sizes = {union_size(c) for c in candidates}
        if len(sizes) != 1:
            return None
    return candidates[0]


def slice_op_boxes(
    op_type: str,
    raw_attributes: Any,
    base_shape: Sequence[int],
    expected_elems: int,
) -> Optional[List[Box]]:
    """Resolve the access region of a slice-view op on its first input.

    Returns validated boxes over ``base_shape``, or None when the region
    cannot be proven (dynamic/advanced indexing, parse failure, shape or
    size inconsistency).
    """
    if op_type not in SLICE_VIEW_OPS:
        return None
    if not base_shape or not all(_is_static_int(d) and d > 0 for d in base_shape):
        return None
    if expected_elems <= 0:
        return None
    parsed = parse_call_attributes(raw_attributes)
    if parsed is None:
        return None
    args, kwargs = parsed
    if not args or not isinstance(args[0], _TensorArg):
        return None
    scalars = args[1:]

    boxes: Optional[List[Box]] = None
    if op_type == "__getitem__":
        if len(scalars) != 1:
            return None
        boxes = _getitem_boxes(scalars[0], base_shape, expected_elems)
    elif op_type in ("narrow", "select", "slice"):
        params = _positional_and_kw(
            scalars,
            kwargs,
            {
                "narrow": ("dim", "start", "length"),
                "select": ("dim", "index"),
                "slice": ("dim", "start", "end", "step"),
            }[op_type],
        )
        if params is None:
            return None
        box = _dim_sub_box(op_type, params, base_shape)
        if box is not None and box_size(box) == expected_elems:
            boxes = [box]

    if boxes is None:
        return None
    for box in boxes:
        if len(box) != len(base_shape):
            return None
        for (lo, hi), dim_size in zip(box, base_shape):
            if not (0 <= lo < hi <= dim_size):
                return None
    return boxes


def _positional_and_kw(
    scalars: Sequence[Any], kwargs: Dict[str, Any], names: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Bind positional + keyword scalar args to parameter names."""
    if len(scalars) > len(names):
        return None
    params: Dict[str, Any] = dict(zip(names, scalars))
    for key, value in kwargs.items():
        if key not in names or key in params:
            return None
        params[key] = value
    return params


def _dim_sub_box(
    op_type: str, params: Dict[str, Any], base_shape: Sequence[int]
) -> Optional[Box]:
    """Box for narrow/select/slice: full on all dims except ``dim``."""
    dim = params.get("dim", 0)
    if not _is_static_int(dim):
        return None
    rank = len(base_shape)
    if dim < 0:
        dim += rank
    if not (0 <= dim < rank):
        return None
    dim_size = base_shape[dim]

    if op_type == "select":
        index = params.get("index")
        if not _is_static_int(index):
            return None
        idx = _normalize_int_index(index, dim_size)
        if idx is None:
            return None
        lo, hi = idx, idx + 1
    elif op_type == "narrow":
        start, length = params.get("start"), params.get("length")
        if not (_is_static_int(start) and _is_static_int(length)):
            return None
        if start < 0:
            start += dim_size
        if length < 0 or not (0 <= start and start + length <= dim_size):
            return None
        lo, hi = start, start + length
    else:  # slice(dim, start, end, step)
        step = params.get("step", 1)
        if step is None:
            step = 1
        if step != 1:
            return None
        start = params.get("start")
        end = params.get("end")
        if start is None:
            start = 0
        if end is None:
            end = dim_size
        if not (_is_static_int(start) and _is_static_int(end)):
            return None
        lo, hi, _ = slice(start, end, 1).indices(dim_size)
    if lo >= hi:
        return None
    return tuple(
        (lo, hi) if d == dim else (0, base_shape[d]) for d in range(rank)
    )


def partition_output_box(
    op_type: str,
    raw_attributes: Any,
    base_shape: Sequence[int],
    output_shapes: Sequence[Sequence[int]],
    output_index: int,
) -> Optional[Box]:
    """Box covered by output ``output_index`` of a chunk/split/tensor_split.

    The partition is reconstructed from the recorded output shapes: all
    outputs must equal the base shape except along one dimension whose
    sizes sum to the base size (a full, ordered partition).  The split
    dimension is taken from the recorded ``dim`` argument when statically
    available and validated; otherwise it is inferred, requiring a unique
    candidate.
    """
    if op_type not in PARTITION_OPS:
        return None
    rank = len(base_shape)
    if rank == 0 or not all(_is_static_int(d) and d > 0 for d in base_shape):
        return None
    if not output_shapes or not (0 <= output_index < len(output_shapes)):
        return None
    for shape in output_shapes:
        if len(shape) != rank or not all(_is_static_int(d) and d > 0 for d in shape):
            return None

    def _valid_dim(d: int) -> bool:
        if not all(
            shape[j] == base_shape[j]
            for shape in output_shapes
            for j in range(rank)
            if j != d
        ):
            return False
        return sum(shape[d] for shape in output_shapes) == base_shape[d]

    declared_dim: Optional[int] = None
    parsed = parse_call_attributes(raw_attributes)
    if parsed is not None:
        args, kwargs = parsed
        # chunk(input, chunks, dim) / split(input, size, dim) /
        # tensor_split(input, sections, dim): dim is 3rd positional or kwarg.
        candidate = kwargs.get("dim") if isinstance(kwargs, dict) else None
        if candidate is None and len(args) >= 3:
            candidate = args[2]
        if _is_static_int(candidate):
            declared_dim = candidate + rank if candidate < 0 else candidate

    if declared_dim is not None and 0 <= declared_dim < rank and _valid_dim(declared_dim):
        dim = declared_dim
    else:
        candidates = [d for d in range(rank) if _valid_dim(d)]
        if len(candidates) != 1:
            return None
        dim = candidates[0]

    offset = sum(shape[dim] for shape in output_shapes[:output_index])
    extent = output_shapes[output_index][dim]
    return tuple(
        (offset, offset + extent) if d == dim else (0, base_shape[d])
        for d in range(rank)
    )


def union_size(boxes: Sequence[Box], cell_budget: int = _UNION_CELL_BUDGET) -> Optional[int]:
    """Exact element count of the union of axis-aligned boxes.

    All boxes must share one rank.  Returns None when ranks are mixed or
    the recursion exceeds ``cell_budget`` elementary cells (the caller
    must then fall back to a conservative bound).
    """
    unique = list(dict.fromkeys(boxes))
    if not unique:
        return 0
    rank = len(unique[0])
    if any(len(b) != rank for b in unique):
        return None
    if rank == 0:
        return 1

    budget = [cell_budget]

    def recurse(active: List[Box]) -> Optional[int]:
        if len(active) == 1:
            return box_size(active[0])
        bounds = sorted({b[0][0] for b in active} | {b[0][1] for b in active})
        total = 0
        for lo, hi in zip(bounds, bounds[1:]):
            budget[0] -= 1
            if budget[0] < 0:
                return None
            covering = list(
                dict.fromkeys(b[1:] for b in active if b[0][0] <= lo and b[0][1] >= hi)
            )
            if not covering:
                continue
            if len(covering[0]) == 0:
                total += hi - lo
                continue
            sub = recurse(covering)
            if sub is None:
                return None
            total += (hi - lo) * sub
        return total

    return recurse(unique)
