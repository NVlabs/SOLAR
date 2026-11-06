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

"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        tensor_names: {inputs: [...], outputs: [...]}
        tensor_shapes: {inputs: [...], outputs: [...]}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

import ast
import json
import warnings
import re
import string
import copy
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import warnings

import networkx as nx
import yaml

from solar.common.types import TensorShapes
from solar.common.utils import (
    ensure_directory,
    NoAliasDumper,
    validate_tensor_names_match_shapes,
    LocalDumper,
    flowify,
    yaml_safe_load,
)
from solar.einsum.af_graph_builder import AFExportRefused, build_af_graph_from_dict
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.op_families import (
    IDENTITY_COPY_OP,
    UnrecognisedOperationError,
    identity_interpolate_evidence,
    interpolate_record_refusal,
)
from solar.einsum.einsum_to_taco import add_taco_expressions
from solar.einsum.ops.base import conv_call_kwargs
from solar.einsum.expansions import (
    expand_gru, expand_lstm, expand_mha, expand_rnn_relu, expand_rnn_tanh, expand_sdpa,
)
from solar.einsum.operands import SIGNATURES, Operand, bind_operands, resolve_operands
from solar.einsum.ops.base import EinsumOp, EinsumOperand
from solar.einsum.ops.registry import get_global_registry

logger = logging.getLogger(__name__)


_CONV_TRANSPOSE_TYPES = frozenset({"convtranspose1d", "convtranspose2d", "convtranspose3d"})
_CONV_TRANSPOSE_ARG_ORDER = ("stride", "padding", "output_padding", "groups", "dilation")
_CONV_TRANSPOSE_DEFAULTS = {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1}


def _split_top_level(text: str) -> List[str]:
    """Split ``a, b: [1, 2], c`` on commas that are not nested in brackets/parens."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _conv_arg_value(name: str, v: Any, dims: int) -> Optional[Any]:
    """Validated per-dim list (or int for ``groups``) for an EXPLICITLY present
    argument value; None when the value is not a usable integer geometry."""
    if isinstance(v, bool) or v is None:
        return None
    if name == "groups":
        return int(v) if isinstance(v, int) and v >= 1 else None
    if isinstance(v, (list, tuple)):
        if not v or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return None
        vv = [int(x) for x in v]
        if len(vv) == 1 and dims > 1:
            vv = vv * dims
        if len(vv) != dims:
            return None
    elif isinstance(v, int):
        vv = [int(v)] * dims
    else:
        return None
    floor = 1 if name in ("stride", "dilation") else 0
    return vv if all(x >= floor for x in vv) else None


def parse_conv_transpose_args(raw_attributes: Any, dims: int) -> Optional[Dict[str, Any]]:
    """Parse ``F.conv_transpose{1,2,3}d`` arguments out of torchview call attributes.

    The attribute string is ``[[<positional args>], {<kwargs>}]`` with tensors
    rendered as ``Tensor(shape=(...), dtype=...)`` and kwargs keys unquoted.
    Positional order is ``(input, weight, bias, stride, padding,
    output_padding, groups, dilation)``; keyword arguments override.  Returns
    ``{"stride", "padding", "output_padding", "dilation": [int]*dims, "groups": int}``.

    Contract (independent review args-1): a default is supplied ONLY
    for a parameter proven omitted from a successfully parsed call.  An
    argument that is explicitly present but cannot be decoded (``opaque``,
    ``[s0]``), is ``None``, or is not a valid integer geometry makes the whole
    call unparseable: the function returns None and the caller reports
    ``conv_args_unavailable: unparsed_call_attributes`` / ``min_work:
    not_representable``.  A wholly unparseable or empty string is None too.
    Export-only metadata for the minimum-work bound (issue #44).
    """
    text = str(raw_attributes or "").strip()
    if not text:
        return None
    split = text.rfind("], {")
    if split < 0 or not text.endswith("}]"):
        return None                                   # not the recorded "[[...], {...}]" form
    positional_text, keyword_text = text[:split + 1] + "]", text[split + 3:-1]
    # Tensor(shape=(...), dtype=...) carries one nested parenthesis level.
    positional_text = re.sub(r"Tensor\([^()]*(?:\([^()]*\)[^()]*)*\)", "None", positional_text)
    positional_text = re.sub(r"torch\.\w+", "None", positional_text)
    try:
        positional = ast.literal_eval(positional_text)
        if isinstance(positional, (list, tuple)) and len(positional) == 1 and isinstance(positional[0], (list, tuple)):
            positional = positional[0]
    except Exception:
        return None                                   # the positional part did not parse: nothing is proven omitted
    if not isinstance(positional, (list, tuple)):
        return None
    body = keyword_text.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    if len(positional) < 2 and not body.strip():
        return None                                   # no call recorded beyond (at most) one tensor
    present: Dict[str, Any] = {}
    # input/weight may be passed positionally or as keywords; geometry follows after them positionally
    rest = list(positional)[2:]
    if rest and rest[0] is None:                      # bias slot (tensor placeholder or None)
        rest = rest[1:]
    for name, val in zip(_CONV_TRANSPOSE_ARG_ORDER, rest):
        present[name] = val
    for item in _split_top_level(body):
        if not item.strip():
            continue
        if ":" not in item:
            return None                               # malformed kwargs entry
        key, value = item.split(":", 1)
        key = key.strip().strip("'\"")
        if key not in _CONV_TRANSPOSE_ARG_ORDER:
            continue
        try:
            present[key] = ast.literal_eval(value.strip())
        except Exception:
            return None                               # explicitly present, undecodable (e.g. `opaque`, `[s0]`)
    out: Dict[str, Any] = {}
    for name in ("stride", "padding", "output_padding", "dilation"):
        if name in present:
            vv = _conv_arg_value(name, present[name], dims)
            if vv is None:
                return None                           # explicitly present but None / invalid
        else:
            vv = [_CONV_TRANSPOSE_DEFAULTS[name]] * dims      # proven omitted from a parsed call
        out[name] = vv
    if "groups" in present:
        g = _conv_arg_value("groups", present["groups"], dims)
        if g is None:
            return None
        out["groups"] = g
    else:
        out["groups"] = _CONV_TRANSPOSE_DEFAULTS["groups"]
    return out


PathLike = Union[str, Path]

# Operation categories for einsum supportability classification
_REAL_EINSUM_OPS = frozenset({
    "matmul", "mm", "bmm", "linear",
    "conv1d", "conv2d", "conv3d",
    "convtranspose1d", "convtranspose2d", "convtranspose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "attention", "sdpa",
    "einsum",
})

_BINARY_ELEMENTWISE_OPS = frozenset({
    "add", "sub", "mul", "div", "pow",
    "add_", "sub_", "mul_", "div_",
    "__add__", "__sub__", "__mul__", "__truediv__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
})

_UNARY_ELEMENTWISE_OPS = frozenset({
    "relu", "sigmoid", "tanh", "gelu", "selu", "elu", "mish",
    "softmax", "log_softmax", "softplus", "hardswish", "hardsigmoid",
    "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos",
    "clamp", "clamp_", "relu_",
    "dropout", "dropout_",
})

_REDUCTION_OPS = frozenset({
    "sum", "mean", "prod", "max", "min", "amax", "amin",
    "argmax", "argmin", "logsumexp", "norm",
})

_NORM_OPS = frozenset({
    "batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d",
    "layer_norm", "layernorm", "rms_norm", "rmsnorm", "group_norm", "groupnorm",
    "instance_norm", "instancenorm", "normalize",
})

_POOLING_OPS = frozenset({
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
})

_SHAPE_OPS = frozenset({
    "view", "reshape", "flatten", "unflatten",
    "squeeze", "unsqueeze", "expand", "repeat",
    "transpose", "permute", "t", "contiguous",
    "cat", "concat", "stack", "split", "chunk",
    "__getitem__", "getitem", "select", "index_select",
})

_MATRIX_OPS = frozenset({"diag", "diagonal", "tril", "triu"})

_EMBEDDING_OPS = frozenset({"embedding"})

_RNN_OPS = frozenset({"gru", "lstm", "rnn"})

_TRIVIAL_OPS = frozenset({
    "clone", "detach", "copy_", "to", "type", "float", "half",
    "hidden-tensor", "output-tensor", "auxiliary-tensor",
    "roll", "pad", "unfold", "fold",
})

_ATTENTION_OPS = frozenset({
    "multi_head_attention_forward", "multihead_attention",
    "flex_attention",
})

_ALL_SUPPORTABLE_OPS = (
    _REAL_EINSUM_OPS | _BINARY_ELEMENTWISE_OPS | _UNARY_ELEMENTWISE_OPS |
    _REDUCTION_OPS | _NORM_OPS | _POOLING_OPS | _SHAPE_OPS |
    _MATRIX_OPS | _EMBEDDING_OPS | _RNN_OPS | _TRIVIAL_OPS | _ATTENTION_OPS
)

# torchview's tensor-side node names (the capture writes ``node_class`` too;
# this set only serves graphs recorded without one)
_TENSOR_NODE_TYPES = frozenset({
    "hidden-tensor", "output-tensor", "auxiliary-tensor", "parameter-tensor", "input-tensor",
})

_UNSUPPORTABLE_OPS = frozenset({
    "if", "while", "for", "return", "raise",
    "print", "assert", "pass",
})


def _product(shape: List[int]) -> int:
    """Compute product of dimensions in a shape.

    Args:
        shape: List of dimension sizes.

    Returns:
        Product of all dimensions (1 for empty shape).
    """
    result = 1
    for dim in shape:
        result *= int(dim)
    return int(result)



# ---------------------------------------------------------------------------
# Unbound returned-output footprint (issue #53; return-stride-4 / -REGION-5)
# ---------------------------------------------------------------------------
_RETURNED_REGION_MAX_RUNS = 1 << 20


def _returned_leaf_byte_intervals(ident: Dict[str, Any], leaf: Dict[str, Any]):
    """Byte intervals ``[start, end)`` within the leaf's storage that the
    returned view addresses, from storage offset / shape / stride / element
    size (the same element-address enumeration a signed-address oracle
    performs, expressed as runs along the unit-stride dimension so large
    contiguous views cost one interval).  Signed strides are normalised
    first (negative-stride-7): a negative stride is reflected onto its
    minimum address (``offset += (shape-1) * stride``; ``stride = |stride|``),
    which addresses exactly the same elements, so the regions are exact for
    any integer strides including zero (broadcast) and negative ones.  The
    installed PyTorch cannot construct negative-stride views; the case is
    reachable only through serialised metadata.  Returns ``(intervals,
    method)``:

    * ``byte_region_union``: exact element-wise regions (clipped to the
      storage by the caller's merge);
    * ``bounding_interval``: a view with more than ``_RETURNED_REGION_MAX_RUNS``
      non-contiguous runs is over-approximated by its bounding extent
      (never an under-count);
    * ``contiguous_assumed_legacy_identity``: a round-2 identity without
      shape/stride (legacy graph) is taken as contiguous ``[offset, offset+numel)``.
    """
    esize = int(ident.get("element_size") or 0)
    offset = int(ident.get("storage_offset") or 0)
    shape = ident.get("shape")
    if shape is None:
        shape = leaf.get("shape")
    stride = ident.get("stride")
    if esize <= 0:
        return [], "byte_region_union"
    if not isinstance(shape, (list, tuple)):
        numel = int(ident.get("numel") or 0)
        return ([(offset * esize, (offset + numel) * esize)] if numel > 0 else []), "contiguous_assumed_legacy_identity"
    shape = [int(d) for d in shape]
    numel = 1
    for d in shape:
        numel *= d
    if numel <= 0:
        return [], "byte_region_union"
    if stride is None:
        return [(offset * esize, (offset + numel) * esize)], "contiguous_assumed_legacy_identity"
    stride = [int(d) for d in stride]
    # negative-stride-7: reflect negative strides onto their minimum
    # address before building forward intervals (same element set).
    for st, sh in zip(stride, shape):
        if st < 0 and sh > 1:
            offset += (sh - 1) * st
    dims = [(abs(st), sh) for st, sh in zip(stride, shape) if sh > 1]
    if not dims:
        return [(offset * esize, (offset + 1) * esize)], "byte_region_union"
    # contiguous (row-major with any dimension order) -> one interval
    ordered = sorted(dims)  # by stride ascending
    contiguous = ordered[0][0] == 1
    expect = 1
    if contiguous:
        for st, sh in ordered:
            if st != expect:
                contiguous = False
                break
            expect *= sh
    if contiguous:
        return [(offset * esize, (offset + numel) * esize)], "byte_region_union"
    # runs along the unit-stride dimension (if any), one interval per run
    run_len = 1
    outer = ordered
    if ordered[0][0] == 1:
        run_len = ordered[0][1]
        outer = ordered[1:]
    n_runs = numel // run_len
    if n_runs > _RETURNED_REGION_MAX_RUNS:
        extent = 1 + sum((sh - 1) * st for st, sh in dims)
        return [(offset * esize, (offset + extent) * esize)], "bounding_interval"
    starts = [offset]
    for st, sh in outer:
        starts = [b + k * st for b in starts for k in range(sh)]
    return [(b * esize, (b + run_len) * esize) for b in starts], "byte_region_union"


def _merge_byte_intervals(intervals, storage_nbytes=None):
    """Union of half-open byte intervals, clipped to ``[0, storage_nbytes)``
    when the storage size is known."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in intervals):
        if storage_nbytes is not None:
            b = min(b, int(storage_nbytes))
        a = max(a, 0)
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _interval_total(intervals) -> int:
    return int(sum(max(0, int(b) - int(a)) for a, b in intervals))


def _subtract_byte_intervals(intervals, minus, work=None):
    """``intervals`` minus ``minus`` (half-open byte intervals; ``minus`` merged and sorted).

    subtract-cost-2: a MONOTONE two-pointer sweep.  ``intervals`` is taken in ascending
    order (sorted here if it is not already), ``minus`` is consumed by a pointer that only ever
    advances: a subtrahend that ends inside the current interval can never meet a later
    (disjoint, ascending) interval, so it is skipped for good; one that extends beyond it is
    kept as the pointer for the next interval.  O(n + m) comparisons instead of restarting the
    whole B scan per A interval.  Every comparison is counted into ``work["ops"]`` and, when
    ``work["budget"]`` is set, the budget is enforced HERE (:class:`_GridWorkExceeded`), so both
    callers -- the period-grid sweep and the interval-only path -- fall back conservatively
    instead of continuing.  ``minus`` that is not sorted/merged is merged first (defensive)."""
    A = [(int(a), int(b)) for a, b in intervals if int(b) > int(a)]
    if any(A[i][1] > A[i + 1][0] for i in range(len(A) - 1)):      # not ascending-disjoint: merge (callers pass merged lists)
        A = _merge_byte_intervals(A)
    B = [(int(a), int(b)) for a, b in minus]
    if any(B[i][1] > B[i + 1][0] for i in range(len(B) - 1)) or any(b <= a for a, b in B):
        B = _merge_byte_intervals(B)
    budget = work.get("budget") if work is not None else None
    ops = 0
    out = []
    j = 0
    m = len(B)
    for a, b in A:
        cur = a
        while j < m and B[j][1] <= cur:          # subtrahends entirely before this interval: skipped for good
            j += 1
            ops += 1
        k = j
        while k < m and B[k][0] < b:
            ops += 1
            ma, mb = B[k]
            if ma > cur:
                out.append((cur, ma))
            if mb > cur:
                cur = mb
            if mb >= b:                            # extends beyond this interval: it is the pointer for the next one
                break
            k += 1
        j = k
        if cur < b:
            out.append((cur, b))
        if budget is not None and work.get("ops", 0) + ops > budget:
            work["ops"] = work.get("ops", 0) + ops
            raise _GridWorkExceeded()
    if work is not None:
        work["ops"] = work.get("ops", 0) + ops
    return out


# ---------------------------------------------------------------------------
# bound-envelope-1: proven coverage only.  A returned view's byte set is
# represented EXACTLY whenever it can be -- an interval list (contiguous or a
# bounded enumeration of runs) or a strided LATTICE (one outer stride: the
# bytes ``start + k*stride .. +run`` for ``k < count``, e.g. ``y[::2]`` of any
# size) -- and only exact regions are ever subtracted as "already charged
# through a bound output".  A region that is merely a bounding interval (several
# outer strides above the enumeration cap) or a legacy contiguity assumption is
# an over-approximation: it may still be BILLED (an over-count is the permitted
# side) but it never covers anything and never makes an unbound leaf an alias.
# Unions / differences of lattices and intervals are computed analytically as
# rectangles on a common period grid (period x offset-within-period), never by
# enumerating runs.
# ---------------------------------------------------------------------------
_RETURNED_LATTICE_MAX_SPLIT = 64        # a lattice may be re-expressed on a coarser common period at most this many times


def _returned_leaf_layout(ident: Dict[str, Any], leaf: Dict[str, Any]):
    """``(esize, offset_elems, dims, run_len, outer, numel)`` of a returned view after
    normalising negative strides -- the same layout :func:`_returned_leaf_byte_intervals`
    enumerates (``dims`` = (|stride|, size) for sizes > 1 sorted by stride; ``outer`` =
    the dims beyond a unit-stride run of ``run_len`` elements).  ``None`` when shape /
    stride are not recorded."""
    esize = int(ident.get("element_size") or 0)
    offset = int(ident.get("storage_offset") or 0)
    shape = ident.get("shape")
    if shape is None:
        shape = leaf.get("shape")
    stride = ident.get("stride")
    if esize <= 0 or not isinstance(shape, (list, tuple)) or stride is None:
        return None
    shape = [int(d) for d in shape]
    stride = [int(d) for d in stride]
    numel = 1
    for d in shape:
        numel *= d
    for st, sh in zip(stride, shape):
        if st < 0 and sh > 1:
            offset += (sh - 1) * st
    dims = sorted((abs(st), sh) for st, sh in zip(stride, shape) if sh > 1)
    run_len = 1
    outer = dims
    if dims and dims[0][0] == 1:
        run_len = dims[0][1]
        outer = dims[1:]
    return esize, offset, dims, run_len, outer, numel


def _returned_leaf_region(ident: Dict[str, Any], leaf: Dict[str, Any]) -> Dict[str, Any]:
    """The byte region of a returned leaf as the accounting uses it:

    ``{"intervals": [...] | None, "lattice": (start, stride, count, run) | None,
    "exact": bool, "method": str, "bytes": int}`` -- ``bytes`` is the exact
    byte count of an exact region and the (over-approximating) extent of an
    approximate one.  Methods: ``byte_region_union`` (exact intervals),
    ``strided_lattice_exact`` (exact lattice above the enumeration cap),
    ``bounding_interval`` and ``contiguous_assumed_legacy_identity``
    (approximate: billed conservatively, never subtracted, never an alias)."""
    intervals, method = _returned_leaf_byte_intervals(ident, leaf)
    if method == "byte_region_union":
        # region-bytes-1: ``bytes`` is the size of the MERGED interval set (a zero-stride broadcast
        # addresses its one element once: shape [9] / stride [0] / width 4 is 4 bytes, not 36); the raw
        # per-run list is kept as ``intervals`` for the set algebra, which merges it again.
        region = {"intervals": list(intervals), "lattice": None, "exact": True, "method": method,
                  "bytes": _interval_total(_merge_byte_intervals(intervals))}
        # grid-cost-1 (a): a regular strided view -- one outer stride over a unit run, runs not
        # overlapping -- is ALSO recorded as the arithmetic progression it is, so the period-grid algebra
        # can represent it as a lattice (a handful of rectangles) instead of one rectangle per run, even
        # far below the enumeration cap.  ``lattice`` stays None here: the region is exact as intervals.
        layout = _returned_leaf_layout(ident, leaf)
        if layout is not None:
            esize, offset, dims, run_len, outer, numel = layout
            if len(outer) == 1 and outer[0][1] > 1 and outer[0][0] * esize >= run_len * esize > 0:
                region["progression"] = (offset * esize, outer[0][0] * esize, outer[0][1], run_len * esize)
        return region
    if method == "bounding_interval":
        layout = _returned_leaf_layout(ident, leaf)
        if layout is not None:
            esize, offset, dims, run_len, outer, numel = layout
            if len(outer) == 1:
                st, cnt = outer[0]
                start_b, stride_b, run_b = offset * esize, st * esize, run_len * esize
                if stride_b == 0:
                    return {"intervals": [(start_b, start_b + run_b)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": run_b}
                if stride_b < run_b:                               # overlapping runs: one contiguous span
                    end_b = start_b + (cnt - 1) * stride_b + run_b
                    return {"intervals": [(start_b, end_b)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": end_b - start_b}
                return {"intervals": None, "lattice": (start_b, stride_b, cnt, run_b), "exact": True,
                        "method": "strided_lattice_exact", "bytes": cnt * run_b}
    return {"intervals": list(intervals), "lattice": None, "exact": False, "method": method, "bytes": _interval_total(_merge_byte_intervals(intervals))}


def _interval_rects(a: int, b: int, period: int):
    """Half-open byte interval -> rectangles ``(p0, p1, o0, o1)`` on the period grid."""
    a, b = int(a), int(b)
    if b <= a:
        return []
    pa, pb = a // period, (b - 1) // period
    if pa == pb:
        return [(pa, pa + 1, a % period, (b - 1) % period + 1)]
    rects = [(pa, pa + 1, a % period, period)]
    if pb > pa + 1:
        rects.append((pa + 1, pb, 0, period))
    rects.append((pb, pb + 1, 0, (b - 1) % period + 1))
    return rects


def _lattice_rects(lattice, period: int):
    """Strided lattice -> rectangles on a period grid that is a multiple of its stride
    (``None`` when the stride does not divide the period or the split is too large)."""
    start, stride, count, run = (int(v) for v in lattice)
    if stride <= 0 or period % stride:
        return None
    m = period // stride
    if m > _RETURNED_LATTICE_MAX_SPLIT:
        return None
    rects = []
    for j in range(min(m, count)):
        first = start + j * stride
        k_count = (count - j + m - 1) // m
        p0, o0 = first // period, first % period
        if o0 + run <= period:
            rects.append((p0, p0 + k_count, o0, o0 + run))
        else:                                                      # the run wraps into the next period
            rects.append((p0, p0 + k_count, o0, period))
            rects.append((p0 + 1, p0 + 1 + k_count, 0, o0 + run - period))
    return rects


def _merge_offsets(offsets):
    out = []
    for a, b in sorted(offsets):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# grid-cost-1 (c): the period-grid algebra is bounded.  One "operation" is one rectangle
# active on one sweep segment; beyond the budget the group footprint falls back CONSERVATIVELY
# (unbound billed by extent, nothing subtracted, no alias) instead of continuing -- it never hangs.
_RETURNED_GRID_MAX_WORK = 1 << 16


class _GridWorkExceeded(Exception):
    """Raised by the grid sweep when the rectangle-segment work budget is exhausted."""


def _coalesce_runs(intervals):
    """Merged, sorted, disjoint byte intervals -> maximal arithmetic progressions of equal-length
    runs ``(start, stride, count, run)`` (``count == 1`` is a plain interval).  An enumerated regular
    strided view (one interval per run) collapses to ONE progression; a two-outer-stride view to
    one progression per outer index (grid-cost-1 (a))."""
    merged = _merge_byte_intervals(intervals)
    out = []
    i, n = 0, len(merged)
    while i < n:
        a, b = merged[i]
        run = b - a
        j = i + 1
        stride = None
        while j < n:
            c, d = merged[j]
            if d - c != run:
                break
            st = c - merged[j - 1][0]
            if stride is None:
                stride = st
            elif st != stride:
                break
            j += 1
        if j - i >= 2:
            out.append((a, stride, j - i, run))
        else:
            out.append((a, run, 1, run))
        i = j
    return out


def _rect_difference_area(rects_a, rects_b, work=None) -> int:
    """Area of (union of ``rects_a``) minus (union of ``rects_b``) on the period grid.

    Incremental sweep of the period axis (grid-cost-1 (b)): every rectangle is read
    ONCE into add / remove events at its period bounds; between consecutive event points the
    ACTIVE rectangles' offset ranges (A merged, minus B merged) times the segment width are
    accumulated.  Work is proportional to the number of (rectangle, segment) pairs in which the
    rectangle is active -- O((n + m) log(n + m)) for n interval rectangles that each span one
    period -- instead of rescanning every rectangle on every segment.  ``work`` (a dict with
    ``ops`` and optional ``budget``) counts those pairs; exceeding ``budget`` raises
    :class:`_GridWorkExceeded`.  ``rects_b`` empty gives the union area of A."""
    add_at: Dict[int, list] = {}
    rem_at: Dict[int, list] = {}
    rects = []
    for side, group in (("a", rects_a), ("b", rects_b)):
        for r in group:
            p0, p1, o0, o1 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            if p1 <= p0 or o1 <= o0:
                continue
            idx = len(rects)
            rects.append((side, o0, o1))
            add_at.setdefault(p0, []).append(idx)
            rem_at.setdefault(p1, []).append(idx)
    if work is not None:
        work["rectangles"] = work.get("rectangles", 0) + len(rects)
        budget = work.get("budget")
        if budget is not None and work.get("ops", 0) + len(rects) > budget:
            work["ops"] = work.get("ops", 0) + len(rects)
            raise _GridWorkExceeded()
    points = sorted(set(add_at) | set(rem_at))
    active_a: Dict[int, tuple] = {}
    active_b: Dict[int, tuple] = {}
    area = 0
    for pa, pb in zip(points, points[1:]):
        for idx in rem_at.get(pa, ()):
            active_a.pop(idx, None); active_b.pop(idx, None)
        for idx in add_at.get(pa, ()):
            side, o0, o1 = rects[idx]
            (active_a if side == "a" else active_b)[idx] = (o0, o1)
        if not active_a:
            continue
        if work is not None:
            work["ops"] = work.get("ops", 0) + len(active_a) + len(active_b)
            budget = work.get("budget")
            if budget is not None and work["ops"] > budget:
                raise _GridWorkExceeded()
        a_off = _merge_offsets(active_a.values())
        if active_b:
            b_off = _merge_offsets(active_b.values())
            remaining = _subtract_byte_intervals([(a, b) for a, b in a_off], [(a, b) for a, b in b_off], work)
        else:
            remaining = [(a, b) for a, b in a_off]
        area += _interval_total(remaining) * (pb - pa)
    return int(area)


def _region_extent(region):
    """Bounding byte interval of a region (lattice or intervals); ``None`` when empty."""
    lat = region.get("lattice")
    if lat is not None:
        start, stride, count, run = (int(v) for v in lat)
        return (start, start + (count - 1) * stride + run)
    ivs = region.get("intervals") or []
    if not ivs:
        return None
    return (min(int(a) for a, b in ivs), max(int(b) for a, b in ivs))


def _lcm(a: int, b: int) -> int:
    from math import gcd
    return a * b // gcd(a, b)


def _returned_group_footprint(unbound: list, bound: list, storage_nbytes) -> Dict[str, Any]:
    """Footprint of one storage's UNBOUND returned regions after removing the bytes PROVEN
    to be charged through BOUND regions of the same storage.

    * exact unbound regions and exact bound regions take part exactly (intervals and
      lattices on a common period grid, analytically);
    * an approximate unbound region is billed by its bounding extent (over-count: the
      permitted side) and is listed in ``approximate_unbound``;
    * an approximate bound region is NEVER subtracted (an upper bound on touched bytes is
      no proof those bytes were charged) and is listed in ``unproven_bound``.

    Returns ``union_bytes`` (what the unbound leaves bill), ``covered_bytes`` (what was
    subtracted), ``remainder_regions`` (interval list when the result is an interval set,
    else ``None`` with ``period`` recording the lattice grid), ``unbound_exact``,
    ``coverage_exact`` and the two lists above.  On the period grid the work is bounded by
    ``_RETURNED_GRID_MAX_WORK`` rectangle-segment operations (grid-cost-1); beyond it the
    result is the conservative ``complexity_fallback`` (extent-billed, nothing subtracted,
    neither side exact)."""
    approximate_unbound = [r.get("path") for r in unbound if not r["exact"]]
    unproven_bound = [r.get("path") for r in bound if not r["exact"]]
    bound_exact = [r for r in bound if r["exact"]]
    lattices = [r["lattice"] for r in unbound + bound_exact if r["lattice"] is not None]
    if not lattices:
        work = {"ops": 0, "rectangles": 0, "budget": _RETURNED_GRID_MAX_WORK}
        unbound_merged = _merge_byte_intervals([iv for r in unbound for iv in (r["intervals"] or [])], storage_nbytes)
        coverage = _merge_byte_intervals([iv for r in bound_exact for iv in (r["intervals"] or [])], storage_nbytes)
        try:
            remainder = _subtract_byte_intervals(unbound_merged, coverage, work)
        except _GridWorkExceeded:
            # subtract-cost-2: the interval-only path is budgeted too; beyond it, fail CLOSED exactly like the
            # grid path -- extents billed, nothing subtracted, neither side exact, no alias, marker kept.
            extents = _merge_byte_intervals([e for e in (_region_extent(r) for r in unbound) if e is not None], storage_nbytes)
            return {"union_bytes": int(_interval_total(extents)), "covered_bytes": 0, "remainder_regions": [(int(a), int(b)) for a, b in extents], "period": None,
                    "unbound_exact": False, "coverage_exact": False,
                    "approximate_unbound": [r.get("path") for r in unbound], "unproven_bound": [r.get("path") for r in bound],
                    "complexity_fallback": {"work_ops": int(work["ops"]), "rectangles": 0, "budget": int(work["budget"]),
                                            "note": "interval-subtraction work budget exceeded: unbound regions billed by bounding extent, bound coverage not subtracted, no alias"}}
        union_bytes = _interval_total(remainder)
        return {"union_bytes": int(union_bytes), "covered_bytes": int(_interval_total(unbound_merged) - union_bytes),
                "remainder_regions": [(int(a), int(b)) for a, b in remainder], "period": None,
                "unbound_exact": not approximate_unbound, "coverage_exact": not unproven_bound,
                "approximate_unbound": approximate_unbound, "unproven_bound": unproven_bound,
                "grid_work": {"ops": int(work["ops"]), "rectangles": 0, "budget": int(work["budget"])}}
    period = 1
    for lat in lattices:
        period = _lcm(period, int(lat[1]))
    work = {"ops": 0, "rectangles": 0, "budget": _RETURNED_GRID_MAX_WORK}

    def rects_of(region):
        """Rectangles of one region on the common period grid.  An explicit lattice, or a regular strided view
        recorded as a ``progression``, or any arithmetic progression found in the merged interval list, becomes
        LATTICE rectangles (at most 128 per progression) when its stride divides the period -- never one
        rectangle per run (grid-cost-1 (a)); only irregular leftovers become per-interval rectangles,
        and those are counted against the work budget."""
        if region["lattice"] is not None:
            return _lattice_rects(region["lattice"], period)
        merged = _merge_byte_intervals(region["intervals"] or [], storage_nbytes)
        progressions = _coalesce_runs(merged)
        prog = region.get("progression")
        if prog is not None and len(progressions) > 1:
            start, stride, count, run = (int(v) for v in prog)
            if storage_nbytes is None or start + (count - 1) * stride + run <= int(storage_nbytes):
                progressions = [(start, stride, count, run)]
        rects = []
        for start, stride, count, run in progressions:
            lat_rects = _lattice_rects((start, stride, count, run), period) if count > 1 else None
            if lat_rects is not None:
                new = lat_rects
            else:                                            # irregular or non-dividing stride: per-run rectangles
                if work["rectangles"] + len(rects) + 2 * count > work["budget"]:
                    raise _GridWorkExceeded()                # do not even build them: the budget decides first
                new = [rc for k in range(count) for rc in _interval_rects(start + k * stride, start + k * stride + run, period)]
            rects.extend(new)
        work["rectangles"] += len(rects)
        if work["rectangles"] > work["budget"]:
            raise _GridWorkExceeded()
        return rects

    try:
        a_rects, b_rects = [], []
        for r in unbound:
            rc = rects_of(r)
            if rc is None:                                        # lattice not representable on the common grid: bill its extent
                lat = r["lattice"]; a_rects.extend(_interval_rects(lat[0], lat[0] + (lat[2] - 1) * lat[1] + lat[3], period))
                approximate_unbound.append(r.get("path"))
            else:
                a_rects.extend(rc)
        for r in bound_exact:
            rc = rects_of(r)
            if rc is None:                                        # cannot prove coverage from it: exclude
                unproven_bound.append(r.get("path"))
            else:
                b_rects.extend(rc)
        total_unbound = _rect_difference_area(a_rects, [], work)
        union_bytes = _rect_difference_area(a_rects, b_rects, work)
    except _GridWorkExceeded:
        # grid-cost-1 (c): the budget is exhausted -- fail CLOSED.  Every unbound region is billed by its
        # bounding extent (an over-count: the permitted side), nothing is subtracted, and neither side is exact,
        # so the leaf keeps its declaration and incomplete marker and can never become an alias.
        extents = _merge_byte_intervals([e for e in (_region_extent(r) for r in unbound) if e is not None], storage_nbytes)
        union_bytes = _interval_total(extents)
        return {"union_bytes": int(union_bytes), "covered_bytes": 0, "remainder_regions": [(int(a), int(b)) for a, b in extents], "period": int(period),
                "unbound_exact": False, "coverage_exact": False,
                "approximate_unbound": [r.get("path") for r in unbound], "unproven_bound": [r.get("path") for r in bound],
                "complexity_fallback": {"work_ops": int(work["ops"]), "rectangles": int(work["rectangles"]), "budget": int(work["budget"]),
                                        "note": "period-grid work budget exceeded: unbound regions billed by bounding extent, bound coverage not subtracted, no alias"}}
    return {"union_bytes": int(union_bytes), "covered_bytes": int(total_unbound - union_bytes), "remainder_regions": None, "period": int(period),
            "unbound_exact": not approximate_unbound, "coverage_exact": not unproven_bound,
            "approximate_unbound": approximate_unbound, "unproven_bound": unproven_bound,
            "grid_work": {"ops": int(work["ops"]), "rectangles": int(work["rectangles"]), "budget": int(work["budget"])}}


class PyTorchToEinsum:
    """Convert PyTorch computation graphs to einsum representation.

    This converter transforms pytorch_graph.yaml files into einsum_graph.yaml
    files, translating PyTorch operations into einsum notation where possible.

    Attributes:
        debug: Whether to print debug information.
        enable_agent: Whether to use LLM agent for unknown operations.
        api_key: API key for LLM agent.
        cache_dir: Directory for caching generated handlers.
    """

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: Optional[str] = None,
        cache_dir: str = "./solar_handlers_cache",
    ) -> None:
        """Initialize the converter.

        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key for LLM agent.
            cache_dir: Directory for caching generated handlers.
        """
        self._debug = debug
        self._enable_agent = enable_agent
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._einsum_analyzer = EinsumAnalyzer(debug=debug)

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    @property
    def einsum_analyzer(self) -> EinsumAnalyzer:
        """The einsum analyzer instance."""
        return self._einsum_analyzer

    def _parse_einsum_from_raw_attributes(
        self,
        module_args: Dict[str, Any],
    ) -> Optional[str]:
        """Parse einsum equation from raw_attributes in module_args.

        For torch.einsum operations, the raw_attributes field contains the
        einsum equation string as the first argument.

        Example raw_attributes:
            "[[\'bijl,lk->bijk\', Tensor(...), Tensor(...)], {}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Solar-compatible einsum equation (uppercase) or None if not found.
        """
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None

        # Try to extract the einsum equation string from raw_attributes
        # Pattern: first string argument in the list, e.g., 'bijl,lk->bijk'
        import re

        # Match quoted string that looks like an einsum equation (contains -> and comma)
        # Handles both single and double quotes
        pattern = r"['\"]([a-zA-Z0-9,\s]+->[\s]*[a-zA-Z0-9]+)['\"]"
        match = re.search(pattern, raw_attrs)

        if match:
            equation = match.group(1).strip()
            # Convert to Solar format (uppercase)
            return self._convert_einsum_to_solar_format(equation)

        return None

    def _convert_einsum_to_solar_format(self, equation: str) -> str:
        """Convert a lowercase einsum equation to Solar's uppercase format.

        Solar uses uppercase letters for dimension labels, with optional
        numeric suffixes for batch dimensions (e.g., B0, B1).

        Example:
            'bijl,lk->bijk' -> 'B0IJL,LK->B0IJK'

        Args:
            equation: Lowercase einsum equation string.

        Returns:
            Uppercase einsum equation string.
        """
        if not equation or "->" not in equation:
            return equation

        # Split into inputs and output
        parts = equation.split("->")
        if len(parts) != 2:
            return equation.upper()

        lhs, rhs = parts[0].strip(), parts[1].strip()

        # Collect all unique dimension letters
        all_dims = set()
        for char in lhs + rhs:
            if char.isalpha():
                all_dims.add(char.lower())

        # Create mapping: lowercase letter -> uppercase with optional number
        # We'll use simple uppercase for now, but could add batch numbering
        dim_map = {d: d.upper() for d in sorted(all_dims)}

        # Apply mapping to equation
        result_lhs = ""
        for char in lhs:
            if char.isalpha():
                result_lhs += dim_map.get(char.lower(), char.upper())
            else:
                result_lhs += char

        result_rhs = ""
        for char in rhs:
            if char.isalpha():
                result_rhs += dim_map.get(char.lower(), char.upper())
            else:
                result_rhs += char

        return f"{result_lhs}->{result_rhs}"

    def _parse_reduction_args_from_raw_attributes(
        self,
        module_args: Dict[str, Any],
    ) -> Tuple[Optional[List[int]], bool]:
        """Parse reduction arguments (dim, keepdim) from raw_attributes.

        For reduction operations like sum/mean/max/min, the raw_attributes field
        contains the dim and keepdim arguments.

        Example raw_attributes:
            "[[Tensor(...)], {dim: 1}]"
            "[[Tensor(...)], {dim: 1, keepdim: True}]"
            "[[Tensor(...)], {dim: [1, 2]}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Tuple of (reduction_dims, keepdim). reduction_dims is a list of ints or None.
        """
        # First check parsed dim/keepdim fields (from _parse_torchview_attributes)
        if "dim" in module_args:
            dim_val = module_args["dim"]
            reduce_dims = [dim_val] if isinstance(dim_val, int) else list(dim_val)
            keepdim = bool(module_args.get("keepdim", False))
            return reduce_dims, keepdim

        # Then try raw_attributes string (regex parsing)
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None, False

        reduce_dims = None
        keepdim = False

        # Match dim: [<numbers>] first (list case)
        list_dim_pattern = r"dim:\s*\[([^\]]+)\]"
        match = re.search(list_dim_pattern, raw_attrs)
        if match:
            dims_str = match.group(1)
            reduce_dims = [int(d.strip()) for d in dims_str.split(",")]
        else:
            # Pattern for single dim: dim: 1 or dim: -1
            single_dim_pattern = r"dim:\s*(-?\d+)"
            match = re.search(single_dim_pattern, raw_attrs)
            if match:
                reduce_dims = [int(match.group(1))]

        # Match keepdim: True or keepdim: False
        keepdim_pattern = r"keepdim:\s*(True|False)"
        match = re.search(keepdim_pattern, raw_attrs)
        if match:
            keepdim = match.group(1) == "True"

        return reduce_dims, keepdim

    def _tensor_arg_shapes_from_raw(
        self,
        module_args: Dict[str, Any],
    ) -> List[Optional[Tuple[int, ...]]]:
        """Shapes of the positional ``Tensor`` arguments recorded by torchview.

        torchview stores the real call signature in ``raw_attributes``, e.g.::

            [[Tensor(shape=(112, 64, 512, 512), dtype=torch.float32),
              Tensor(shape=(), dtype=torch.float32)], {p: 'fro'}]

        Returns the shape tuple of every ``Tensor(shape=(...))`` occurrence in
        order; a scalar tensor yields ``()`` and an unparseable shape yields
        ``None``. This is ground truth for an op's true tensor arity — used by
        the dropped-edge repair in ``_build_op_graph`` to detect ops whose
        recorded ``input_shapes`` undercount their actual tensor inputs
        (torchview can drop a scalar-tensor edge, e.g. ``x / x.norm()``).
        """
        raw = module_args.get("raw_attributes", "") if module_args else ""
        if not raw:
            return []
        shapes: List[Optional[Tuple[int, ...]]] = []
        for m in re.finditer(r"Tensor\(shape=\(([^)]*)\)", raw):
            body = m.group(1).strip().rstrip(",").strip()
            if not body:
                shapes.append(())
                continue
            try:
                shapes.append(tuple(int(d.strip()) for d in body.split(",") if d.strip()))
            except ValueError:
                shapes.append(None)
        return shapes

    # ----- torchview-quirk constants ---------------------------------------
    # Layer types whose output dtype passes through unchanged from the first
    # input (no real compute, no precision change). Used by the dtype repair
    # in ``_repair_torchview_quirks``.
    _SHAPE_OP_TYPES_FOR_DTYPE: "set[str]" = {
        "view", "reshape", "flatten", "unflatten", "squeeze", "unsqueeze",
        "expand", "repeat", "transpose", "permute", "t", "contiguous",
        "cat", "concat", "stack", "split", "chunk",
        "__getitem__", "getitem", "select", "index_select",
        "narrow", "slice", "movedim", "swapaxes", "swapdims",
        "view_as", "reshape_as", "broadcast_to", "expand_as",
        "detach", "alias", "ravel", "unbind", "diagonal",
    }
    # Ops whose recorded output dtype must be kept as-is by the dtype repair.
    _CAST_OP_TYPES: "set[str]" = {
        "to", "type", "type_as", "float", "double", "half", "bfloat16",
        "int", "long", "short", "bool", "byte", "char",
        # Creation / index ops choose their dtype independently of their
        # inputs (``zeros_like(x, dtype=uint8)``, ``argmax`` -> int64).
        "zeros_like", "ones_like", "full_like", "empty_like", "rand_like",
        "randn_like", "randint_like", "zeros", "ones", "full", "empty",
        "arange", "linspace", "argmax", "argmin", "argsort", "nonzero",
        "bucketize", "searchsorted", "one_hot",
    }
    # Comparison / logical ops always produce ``torch.bool`` (1 B); the
    # widest-input rule would price an fp32 mask at 4 B.
    _BOOL_OP_TYPES: "set[str]" = {
        "eq", "ne", "gt", "lt", "ge", "le",
        "__eq__", "__ne__", "__gt__", "__lt__", "__ge__", "__le__",
        "__and__", "__or__", "__xor__", "__invert__",
        "__rand__", "__ror__", "__rxor__",
        "logical_and", "logical_or", "logical_not", "logical_xor",
        "isnan", "isinf", "isfinite", "isclose", "isin", "signbit",
        "any", "all", "equal", "allclose",
    }
    _DTYPE_BITS: "dict[str, int]" = {
        "float64": 64, "double": 64, "complex128": 128, "complex64": 64,
        "float32": 32, "tf32": 32,
        "bfloat16": 16, "float16": 16, "half": 16,
        "int64": 64, "long": 64, "int32": 32, "int": 32,
        "int16": 16, "short": 16, "int8": 8, "uint8": 8, "byte": 8,
        "bool": 1,
    }

    @classmethod
    def _bits_of_dtype(cls, dtype_str: Optional[str]) -> int:
        if not dtype_str:
            return 32
        return cls._DTYPE_BITS.get(
            str(dtype_str).replace("torch.", "").lower(), 32
        )

    # Ops whose tensor operands are all "real" (weights / activations): a
    # tensor argument recorded in ``raw_attributes`` but absent from the
    # traced edges is an untraced constant (e.g. a Sobel kernel built with
    # ``torch.tensor`` inside ``forward``) and must still be read from DRAM.
    _CONST_OPERAND_OPS: "set[str]" = {
        "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "linear", "matmul", "bmm", "addmm", "addmv", "baddbmm", "addbmm", "mm", "mv", "dot",
        "embedding",
    }

    def _repair_torchview_quirks(
        self,
        layers: Dict[str, Any],
        op_ids: List[str],
        tensor_ids: List[str],
        auxiliary_ids: Optional[List[str]] = None,
    ) -> None:
        """Single pass repairing every known torchview tracing quirk.

        torchview is great at recording shapes and types but consistently
        drops a small number of well-known patterns. Three repairs run
        here in dependency order so every downstream pass — per-op
        handlers, op-graph build, AF emission — sees a clean ``layers``
        dict and never has to second-guess torchview:

        **(A) Dropped scalar-tensor edges** (FrobeniusNorm pattern).
        torchview can omit a tensor edge entirely when one operand is a
        scalar tensor produced by a reduction (canonical: ``x / x.norm()``).
        Detected by comparing each op's ``raw_attributes`` tensor-arg count
        against its recorded ``input_shapes``; missing slots are wired in
        from a matching dangling ``hidden-tensor`` dead-end (one producer,
        no consumer) of the same shape.

        **(B) Split tensor-node pairs** (cumsum_exclusive zeros_like).
        torchview occasionally splits a single physical tensor flowing
        producer→consumer into two disconnected nodes — an orphan (no
        producer, has consumer) and a dead-end (has producer, no consumer)
        of matching ``(shape, dtype)``. We rewire the orphan's
        ``connections.inputs`` to the dead-end's producer so the normal
        edge build picks up the producer→consumer edge, and record the
        producer in ``self._tensor_to_producer_op`` for the converter's
        hidden-tensor resolution path.

        **(C) Wrong output dtypes** (fp32 override on bf16/fp16 outputs).
        torchview reports ``torch.float32`` on FunctionNode outputs even
        when all inputs are narrower. Walk the layers in insertion (topo)
        order; for each non-shape op set the output dtype to the widest
        input dtype and propagate forward through downstream tensor nodes
        and the consumer ops' ``input_dtypes``. Shape ops pass through the
        first input's dtype.

        Mutates ``layers`` in place. Initializes
        ``self._tensor_to_producer_op``. Conservative: only unambiguous
        1:1 matches are applied for (A) and (B); ambiguous candidates are
        left untouched.
        """
        self._tensor_to_producer_op = {}
        # tensor_id -> position among its producer's ordered outputs.  Only
        # populated for slots > 0; consumers of slot 0 keep the plain
        # ``<producer>.Output`` name.  Without this, every consumer of a
        # multi-output op (chunk/split) was attributed to output 0, losing
        # which partition slice it actually reads.
        self._tensor_to_producer_slot = {}
        op_id_set = set(op_ids)

        # --- (A+B prep) Index orphan / dead-end tensor nodes --------------
        # The shape/dtype of a tensor node is normally its own
        # ``output_shapes[0]`` / ``output_dtypes[0]``. But torchview's
        # ``hidden-tensor`` placeholder records empty lists for both, so we
        # fall back to the producer's ``output_shapes[0]`` when the tensor
        # itself has no shape recorded.
        orphans_by_key: Dict[Tuple[Tuple[int, ...], str], List[str]] = defaultdict(list)
        dangling_by_key: Dict[Tuple[Tuple[int, ...], str], List[Tuple[str, str]]] = defaultdict(list)
        hidden_dangling_by_shape: Dict[Tuple[int, ...], List[Tuple[str, str, str]]] = defaultdict(list)
        for tensor_id in tensor_ids:
            tdata = layers.get(tensor_id) or {}
            conns = tdata.get("connections") or {}
            producers_ = [p for p in (conns.get("inputs") or []) if p in op_id_set]
            consumers_ = [c for c in (conns.get("outputs") or []) if c in op_id_set]
            shapes = tdata.get("output_shapes") or tdata.get("input_shapes") or []
            dtypes = tdata.get("output_dtypes") or tdata.get("input_dtypes") or []
            if not shapes and len(producers_) == 1:
                pdata = layers.get(producers_[0]) or {}
                shapes = pdata.get("output_shapes") or []
                dtypes = pdata.get("output_dtypes") or []
            if not shapes:
                continue
            sh = tuple(shapes[0]) if shapes[0] is not None else ()
            dt = str(dtypes[0]) if dtypes else ""
            key = (sh, dt)
            if not producers_ and consumers_:
                orphans_by_key[key].append(tensor_id)
            elif (len(producers_) == 1 and not consumers_
                  and (tdata.get("type") or "").lower() != "output-tensor"):
                # Model outputs legitimately end the graph; they are not
                # the dead-end half of a split tensor node.
                dangling_by_key[key].append((tensor_id, producers_[0]))
                if (tdata.get("type") or "").lower() == "hidden-tensor":
                    hidden_dangling_by_shape[sh].append((tensor_id, producers_[0], dt))

        # --- (A) Dropped scalar-tensor edges ------------------------------
        consumed: set[tuple[str, str]] = set()
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            arg_shapes = [s for s in self._tensor_arg_shapes_from_raw(
                odata.get("module_args") or {}) if s is not None]
            if not arg_shapes:
                continue
            recorded = [tuple(s) for s in (odata.get("input_shapes") or [])
                        if s is not None]
            missing = Counter(arg_shapes) - Counter(recorded)
            if not missing:
                continue
            in_dtypes = odata.get("input_dtypes") or []
            default_dt = str(in_dtypes[0]) if in_dtypes else "torch.float32"
            op_type_l = (odata.get("type") or "").lower()
            for sh, cnt in missing.items():
                for _ in range(cnt):
                    candidates = [(t, p, d) for (t, p, d)
                                  in hidden_dangling_by_shape.get(sh, [])
                                  if (t, p) not in consumed]
                    if (not candidates and len(sh) >= 1
                            and op_type_l in self._CONST_OPERAND_OPS):
                        # (A') No traced tensor anywhere carries this
                        # operand: it is an untraced constant. Synthesize an
                        # auxiliary-tensor source so the op keeps its full
                        # arity (conv keeps its kernel, matmul its weight)
                        # and the bytes are counted as a model input.
                        self._synthesize_constant_operand(
                            layers, op_id, odata, list(sh), default_dt,
                            auxiliary_ids)
                        continue
                    if len(candidates) != 1:
                        continue  # ambiguous or none
                    tensor_id, producer_op, dt = candidates[0]
                    if producer_op == op_id:
                        continue
                    consumed.add((tensor_id, producer_op))
                    odata.setdefault("input_shapes", []).append(list(sh))
                    odata.setdefault("input_dtypes", []).append(dt or default_dt)
                    odata.setdefault("input_types", []).append("input")
                    oc = odata.setdefault("connections", {}).setdefault("inputs", [])
                    if tensor_id not in oc:
                        oc.append(tensor_id)
                    tdata = layers.get(tensor_id) or {}
                    tout = tdata.setdefault("connections", {}).setdefault("outputs", [])
                    if op_id not in tout:
                        tout.append(op_id)

        # --- (B) Split tensor-node pairs ----------------------------------
        # torchview records the result of a creation op (``zeros_like``) or
        # an in-place update as one dead-end node and hands consumers a
        # second, producer-less node of the same shape. One orphan / one
        # dead-end is the common case; several accumulators of one shape
        # (``grad_w = zeros_like(w)`` per expert weight) produce equal-size
        # groups, which are paired in trace order (the dead-end precedes
        # the orphan that replaces it).
        for key, orphan_ids in orphans_by_key.items():
            de_list = [(t, p) for (t, p) in dangling_by_key.get(key, [])
                       if (t, p) not in consumed]
            if not de_list or len(orphan_ids) != len(de_list):
                continue
            for orphan_id, (_, producer_op) in zip(orphan_ids, de_list):
                self._tensor_to_producer_op[orphan_id] = producer_op
                # Rewire the orphan's ``connections.inputs`` to the producer so
                # the normal edge build picks up producer→consumer naturally.
                orphan_data = layers.get(orphan_id) or {}
                ocon_in = orphan_data.setdefault("connections", {}).setdefault("inputs", [])
                if producer_op not in ocon_in:
                    ocon_in.append(producer_op)

        # --- (C) Output-dtype correction ----------------------------------
        self._repair_dtypes(layers, op_id_set)

    # ---- dtype repair -----------------------------------------------------
    _TENSOR_ARG_RE = re.compile(r"Tensor\(shape=\(([^)]*)\),\s*dtype=(torch\.\w+)\)")
    _DTYPE_TOKEN_RE = re.compile(r"(?<![\w.])(torch\.[a-z0-9_]+)\b")
    _DTYPE_NAMES = {
        "float64", "double", "float32", "float", "float16", "half", "bfloat16",
        "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
        "float4_e2m1fn_x2", "int64", "long", "int32", "int", "int16", "short",
        "int8", "uint8", "bool", "complex64", "complex128", "cfloat", "cdouble",
    }
    _DTYPE_ALIASES = {
        "torch.float": "torch.float32", "torch.double": "torch.float64",
        "torch.half": "torch.float16", "torch.long": "torch.int64",
        "torch.int": "torch.int32", "torch.short": "torch.int16",
        "torch.cfloat": "torch.complex64", "torch.cdouble": "torch.complex128",
    }
    _METHOD_CASTS = {
        "float": "torch.float32", "double": "torch.float64", "half": "torch.float16",
        "bfloat16": "torch.bfloat16", "int": "torch.int32", "long": "torch.int64",
        "short": "torch.int16", "bool": "torch.bool", "byte": "torch.uint8",
        "char": "torch.int8",
    }
    _GENERIC_CASTS = {"to", "type", "type_as", "astype"}
    _CREATION_OPS = {
        "zeros_like", "ones_like", "full_like", "empty_like", "rand_like",
        "randn_like", "randint_like", "zeros", "ones", "full", "empty",
        "arange", "linspace", "eye", "rand", "randn", "randint", "tensor",
    }
    _INDEX_OUTPUT_OPS = {"argmax", "argmin", "argsort", "nonzero", "bucketize",
                         "searchsorted", "argwhere", "count_nonzero"}
    _VALUE_INDEX_OPS = {"topk", "sort", "max", "min", "mode", "kthvalue", "median"}
    _INT_DTYPES = {"torch.int64", "torch.int32", "torch.int16", "torch.int8",
                   "torch.uint8", "torch.bool"}

    @classmethod
    def _norm_dtype(cls, d: Optional[str]) -> Optional[str]:
        if not d:
            return None
        d = str(d)
        if not d.startswith("torch."):
            d = "torch." + d
        return cls._DTYPE_ALIASES.get(d, d)

    @classmethod
    def _raw_call(cls, odata: Dict[str, Any]) -> Tuple[str, str, List[Tuple[Tuple[int, ...], str]]]:
        """Split ``raw_attributes`` into (args text, kwargs text, tensor args)."""
        raw = odata.get("raw_attributes")
        if raw is None:
            raw = (odata.get("module_args") or {}).get("raw_attributes")
        raw = str(raw or "")
        if not raw:
            return "", "", []
        if "], {" in raw:
            args_part, kwargs_part = raw.rsplit("], {", 1)
        else:
            args_part, kwargs_part = raw, ""
        tensors = []
        for m in cls._TENSOR_ARG_RE.finditer(args_part):
            shape = tuple(int(x) for x in re.findall(r"-?\d+", m.group(1)))
            tensors.append((shape, cls._norm_dtype(m.group(2))))
        return args_part, kwargs_part, tensors

    @classmethod
    def _kwarg_dtype(cls, kwargs_part: str) -> Optional[str]:
        m = re.search(r"\bdtype\s*:\s*(torch\.[a-z0-9_]+)", kwargs_part)
        return cls._norm_dtype(m.group(1)) if m else None

    @classmethod
    def _free_dtype_tokens(cls, args_part: str) -> List[str]:
        """``torch.X`` dtype tokens in the args that are not inside ``Tensor(...)``."""
        stripped = cls._TENSOR_ARG_RE.sub("T", args_part)
        out = []
        for m in cls._DTYPE_TOKEN_RE.finditer(stripped):
            tok = m.group(1)
            if tok[len("torch."):] in cls._DTYPE_NAMES:
                out.append(cls._norm_dtype(tok))
        return out

    @classmethod
    def _cast_target(cls, layer_type: str, args_part: str, kwargs_part: str,
                     tensors: List[Tuple[Tuple[int, ...], str]]) -> Optional[str]:
        """Target dtype of an explicit cast, or None for a device-only move."""
        if layer_type in cls._METHOD_CASTS:
            return cls._METHOD_CASTS[layer_type]
        kw = cls._kwarg_dtype(kwargs_part)
        if kw:
            return kw
        toks = cls._free_dtype_tokens(args_part)
        if toks:
            return toks[0]
        if layer_type == "type_as" or (layer_type == "to" and len(tensors) >= 2):
            # ``x.to(other)`` / ``x.type_as(other)`` adopt the other tensor's dtype.
            return tensors[1][1]
        return None

    @classmethod
    def _promote(cls, dtypes: List[str]) -> Optional[str]:
        """PyTorch-style promotion: floats beat ints beat bool; widest wins."""
        ds = [cls._norm_dtype(d) for d in dtypes if d]
        if not ds:
            return None
        floats = [d for d in ds if d not in cls._INT_DTYPES]
        if floats:
            return max(floats, key=cls._bits_of_dtype)
        ints = [d for d in ds if d != "torch.bool"]
        if ints:
            return max(ints, key=cls._bits_of_dtype)
        return "torch.bool"

    def _repair_dtypes(self, layers: Dict[str, Any], op_id_set: "set[str]") -> None:
        """Rebuild every tensor dtype from the call arguments torchview recorded.

        torchview's per-node ``input_dtypes`` / ``output_dtypes`` are not
        trustworthy: for models without parameters (every SOL-ExecBench
        problem) missing dtypes are padded with ``torch.float32``, so a
        ``.to(torch.bfloat16)`` output, an fp16 model input or a bool mask
        all read as 4 B tensors. The ``raw_attributes`` string, however,
        carries each tensor argument as ``Tensor(shape=..., dtype=...)`` and
        each cast target literally, so dtypes are rebuilt from it:

        * op inputs: the raw tensor argument with the same shape (in order);
          the matched source tensor node is corrected too, so model inputs
          and weights get their real width;
        * casts (``to``/``type``/``float``/``half``/...): the dtype kwarg,
          a free ``torch.X`` token, the method name, or the other tensor's
          dtype; a device-only ``.to(device)`` keeps the input dtype;
        * creation ops (``zeros_like(x, dtype=...)``, ``arange``...): the
          dtype kwarg, else the tensor argument's dtype;
        * comparisons / logical ops: ``torch.bool``; index-producing ops:
          ``torch.int64`` (second output of ``topk``/``sort``/``max(dim)``);
        * everything else: PyTorch promotion over the inputs (floats beat
          ints beat bool, widest wins), so ``embedding(ids, table)`` and
          ``where(mask, a, b)`` take the data dtype, not the index width.
        """
        corrected: Dict[str, str] = {}
        for layer_id, ldata in layers.items():
            if layer_id in op_id_set:
                continue
            outd = ldata.get("output_dtypes") or ldata.get("input_dtypes") or []
            if outd:
                corrected[layer_id] = self._norm_dtype(outd[0])

        def _set_tensor_dtype(tid: str, dtype: str) -> None:
            corrected[tid] = dtype
            tdata = layers.get(tid)
            if tdata is None or tid in op_id_set:
                return
            for key in ("output_dtypes", "input_dtypes"):
                n = len(tdata.get(key) or [])
                if n:
                    tdata[key] = [dtype] * n

        for layer_id, odata in layers.items():
            if layer_id not in op_id_set:
                continue
            layer_type = (odata.get("type") or "").lower()
            in_tensors = (odata.get("connections") or {}).get("inputs") or []
            in_shapes = [tuple(int(x) for x in s) if isinstance(s, (list, tuple)) else None
                         for s in (odata.get("input_shapes") or [])]
            in_dtypes = [self._norm_dtype(d) for d in (odata.get("input_dtypes") or [])]
            while len(in_dtypes) < len(in_tensors):
                in_dtypes.append(None)
            args_part, kwargs_part, raw_tensors = self._raw_call(odata)

            # (1) inputs: raw tensor args matched by shape, else propagated dtype.
            used = [False] * len(raw_tensors)
            for i, tid in enumerate(in_tensors):
                shape = in_shapes[i] if i < len(in_shapes) else None
                raw_dtype = None
                if shape is not None:
                    for j, (rshape, rdtype) in enumerate(raw_tensors):
                        if not used[j] and rshape == shape:
                            used[j] = True
                            raw_dtype = rdtype
                            break
                if raw_dtype:
                    in_dtypes[i] = raw_dtype
                    if tid in layers and tid not in op_id_set:
                        _set_tensor_dtype(tid, raw_dtype)
                elif tid in corrected:
                    in_dtypes[i] = corrected[tid]
            in_dtypes = [d or "torch.float32" for d in in_dtypes]
            if in_dtypes:
                odata["input_dtypes"] = in_dtypes

            # (2) outputs.
            n_out = len(odata.get("output_dtypes") or []) or 1
            recorded = [self._norm_dtype(d) for d in (odata.get("output_dtypes") or [])]
            data_dtype = self._promote(in_dtypes) or (recorded[0] if recorded else "torch.float32")
            out_dtypes: List[str]
            if odata.get("output_dtypes_recorded") and recorded and all(recorded):
                # The capture shim recorded the ACTUAL executed result
                # dtypes per output slot — authoritative evidence; the
                # heuristic reconstruction below exists for graphs captured
                # before this evidence (default-padded output_dtypes).  A
                # cast's target dtype in particular exists nowhere else
                # when the call carries no raw_attributes (the
                # sparse-update BF16-billed-as-FP32 gate).
                out_dtypes = list(recorded)
                if len(out_dtypes) < n_out:
                    out_dtypes += [out_dtypes[-1]] * (n_out - len(out_dtypes))
            elif layer_type in self._METHOD_CASTS or layer_type in self._GENERIC_CASTS:
                target = self._cast_target(layer_type, args_part, kwargs_part, raw_tensors)
                out_dtypes = [target or (in_dtypes[0] if in_dtypes else data_dtype)] * n_out
            elif layer_type in self._BOOL_OP_TYPES:
                out_dtypes = ["torch.bool"] * n_out
            elif layer_type in self._INDEX_OUTPUT_OPS:
                out_dtypes = ["torch.int64"] * n_out
            elif layer_type in self._VALUE_INDEX_OPS and n_out >= 2:
                out_dtypes = [data_dtype] + ["torch.int64"] * (n_out - 1)
            elif layer_type in self._CREATION_OPS:
                kw = self._kwarg_dtype(kwargs_part) or (self._free_dtype_tokens(args_part) or [None])[0]
                if kw:
                    chosen = kw
                elif raw_tensors:
                    chosen = raw_tensors[0][1]
                elif in_dtypes:
                    chosen = in_dtypes[0]
                else:
                    chosen = recorded[0] if recorded else "torch.float32"
                out_dtypes = [chosen] * n_out
            elif layer_type in self._SHAPE_OP_TYPES_FOR_DTYPE:
                out_dtypes = [in_dtypes[0] if in_dtypes else data_dtype] * n_out
            else:
                out_dtypes = [data_dtype] * n_out
            odata["output_dtypes"] = out_dtypes
            for k, tid in enumerate((odata.get("connections") or {}).get("outputs") or []):
                d = out_dtypes[min(k, len(out_dtypes) - 1)]
                _set_tensor_dtype(tid, d)

    def _validate_tensor_shape_consistency(
        self,
        einsum_graph: Dict[str, Any],
    ) -> None:
        """Assert that every tensor name reused across einsums has a single shape.

        Walks every einsum's tensor_names + tensor_shapes (inputs and outputs)
        and builds a (name → set of shape tuples) map. Any name with >1 distinct
        shape indicates an emit bug — for example, the cumsum_exclusive case
        where ``Model.cat`` mis-attributed input 0 to ``Model.cumsum``, causing
        the name ``Model.cumsum.Output`` to claim both ``[32768, 1]`` and
        ``[32768, 32767]`` shapes.

        Surfaces the violation at the solar boundary (with a list of every
        einsum that referenced the conflicting tensor) rather than letting
        the inconsistency propagate to AccelForge.
        """
        shapes_by_name: Dict[str, Dict[Tuple[int, ...], List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for layer_name, m in (einsum_graph.get("layers") or {}).items():
            tnames = m.get("tensor_names") or {}
            tshapes = m.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                names = tnames.get(side) or []
                shapes = tshapes.get(side) or []
                n = min(len(names), len(shapes))
                for i in range(n):
                    if shapes[i] is None:
                        continue
                    key = tuple(shapes[i])
                    shapes_by_name[names[i]][key].append(
                        f"{layer_name}.{side}[{i}]"
                    )
        conflicts: List[str] = []
        for name, by_shape in shapes_by_name.items():
            if len(by_shape) > 1:
                listed = "; ".join(
                    f"{list(shape)} via {refs}" for shape, refs in by_shape.items()
                )
                conflicts.append(f"  '{name}': {listed}")
        if conflicts:
            raise ValueError(
                "Pre-AF tensor-shape inconsistency detected: a tensor name "
                "is referenced with multiple distinct shapes across einsums. "
                "This usually means a producer was mis-attributed in stage-1. "
                "Conflicts:\n" + "\n".join(conflicts)
            )

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
        enable_rename: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Convert a PyTorch graph to einsum representation.

        This method:
        1. Loads the PyTorch graph
        2. Builds an operation-only graph (collapsing tensor nodes)
        3. Converts operations to einsum notation
        4. Writes einsum_graph.yaml
        5. Optionally renames ranks using BFS and writes einsum_graph_renamed.yaml

        Args:
            pytorch_graph_path: Path to pytorch_graph.yaml (or legacy JSON).
            output_dir: Directory to write output files.
            copy_graph: If True, copy input graph to output directory.
            expand_complex_ops: If True, attempt to expand complex operations.
            enable_rename: If True, run BFS rank renaming; otherwise copy einsum_graph.yaml as renamed.

        Returns:
            The einsum graph dictionary, or None on failure.
        """
        src = Path(pytorch_graph_path)
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self._debug:
                logger.debug(f"PyTorch graph not found: {src}")
            return None

        pytorch_graph = self._load_pytorch_graph(src)
        if not pytorch_graph:
            return None

        if copy_graph:
            self._copy_input_graph(src, out_dir, pytorch_graph)

        # Build operation-only graph (collapse tensor nodes)
        op_graph, start_nodes_info, param_nodes_info = self._build_op_graph(pytorch_graph)

        # Optional complex operation expansion
        if expand_complex_ops:
            op_graph = self._expand_complex_ops(op_graph)

        # Build einsum graph dictionary
        einsum_graph = self._build_einsum_graph(
            pytorch_graph, op_graph, start_nodes_info, param_nodes_info
        )

        # Add TACO expressions to all layers
        einsum_graph = add_taco_expressions(einsum_graph)

        # Pre-AF validation: every tensor name reused across einsums must
        # have a single, consistent shape. Catches emit-side inconsistencies
        # (e.g. the cat-input mis-attribution that mapped two distinct
        # shapes onto one tensor name in kernbench L1/92 cumsum_exclusive)
        # before they reach AccelForge's pydantic schema, where the error
        # message points only at the symptom rather than the cause.
        self._validate_tensor_shape_consistency(einsum_graph)
        # Carry the torchview capability record (issue #45).  A degraded
        # capture (only produced under --allow-degraded-capture) marks every
        # layer not supportable so no stage presents it as trusted accounting.
        capability = pytorch_graph.get("capture_capability")
        if capability is not None:
            from solar.graph.torchview_capability import degraded_marker
            einsum_graph["capture_capability"] = dict(capability)
            marker = degraded_marker(capability)
            einsum_graph["capture_status"] = "degraded" if marker else "ok"
            if marker:
                for layer in einsum_graph.get("layers", {}).values():
                    layer["is_einsum_supportable"] = False
                    layer["capture_degraded"] = marker

        # Write einsum_graph.yaml
        out_path = out_dir / "einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                einsum_graph, f,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False
            )

        if self._debug:
            logger.info(f"✅ Wrote einsum graph: {out_path}")

        # EinsumRankRenamer was removed (see einsum_rank_renamer.py's
        # tombstone): the AccelForge graph below is the principled
        # successor of the rename pass.  einsum_graph_renamed.yaml is kept
        # as a stable downstream path name and is now always a copy.
        renamed_path = out_dir / "einsum_graph_renamed.yaml"
        import shutil
        shutil.copy2(out_path, renamed_path)
        if self._debug:
            logger.info(f"✅ Copied einsum graph as renamed: {renamed_path}")

        # Principled AccelForge graph emission via one union-find pass
        # over the stage-2 einsum graph. Replaces the historical
        # ``_build_af_einsum_graph_renamed`` + ``_build_accelforge_graph``
        # pipeline, which suffered from a phase-skew bug where the conflict
        # mint mutated ``dims`` in-place without persisting back into
        # ``ops["Output"]`` (causing the "rank-size collision / Rk reuse"
        # failures on DenseNet/Mamba2/RNN/LSTM/GRU/etc.).
        try:
            af_graph = build_af_graph_from_dict(einsum_graph)
        except AFExportRefused as exc:
            # Fail closed (state-consumer-p2-5): no AF workload is serialized for a graph
            # whose predecessor slots the builder could only fill by inventing
            # a weight.  The refusal is recorded in the einsum graph files
            # (analysis and perf still run, qualified by the layers' own
            # incomplete markers) and a stale AF file from an earlier run is
            # never left behind.
            einsum_graph["af_export_refused"] = {
                "reason": str(exc),
                "layers": {lid: dict(info) for lid, info in exc.layers.items()},
                "incomplete_reasons": dict(exc.incomplete_reasons),
            }
            # state-consumer-p2-8: a refusal the BUILDER discovered (unmarked graph, positional
            # correspondence) carries its evidence only in the exception; copy it into
            # the layer's own qualification fields so the persisted stage-2 graph, the
            # analysis layer/total and the perf workload carry status "incomplete"
            # through the existing marker path (STATUS_CONTRACT / INTEG-9), exactly as
            # a converter-marked layer does.
            for lid, info in exc.layers.items():
                layer = (einsum_graph.get("layers") or {}).get(lid)
                if layer is None:
                    continue
                layer["status"] = "incomplete"
                layer.setdefault("incomplete_reason", info.get("incomplete_reason") or "predecessor_slots_unaligned")
                if not layer.get("predecessor_slots_unaligned"):
                    layer["predecessor_slots_unaligned"] = [dict(s) for s in (info.get("predecessor_slots_unaligned") or [])]
                    layer["predecessor_slots_unaligned_source"] = "af_builder_positional_correspondence"
            for p in (out_dir / "einsum_graph.yaml", out_dir / "einsum_graph_renamed.yaml"):
                with open(p, "w") as f:
                    yaml.dump(einsum_graph, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
            stale = out_dir / "af_einsum_graph.yaml"
            if stale.exists():
                stale.rename(out_dir / "af_einsum_graph.REFUSED.yaml.stale")
            logger.warning("%s", exc)
            return einsum_graph
        # Strip internal diagnostics before serializing — AF rejects unknown
        # top-level fields.
        af_to_write = {k: v for k, v in af_graph.items() if not k.startswith("_")}
        out_path = out_dir / "af_einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                af_to_write, f,
                sort_keys=False,
                default_flow_style=False
            )

        if self._debug:
            logger.info(f"✅ Wrote AccelForge graph: {out_path}")
            diagnostics = af_graph.get("_diagnostics") or []
            for d in diagnostics:
                logger.info(f"af_graph diagnostic: {d}")

        return einsum_graph

    # Backward compatibility alias
    convert_graph = convert

    def _copy_input_graph(
        self,
        src: Path,
        out_dir: Path,
        pytorch_graph: Dict[str, Any],
    ) -> None:
        """Copy input graph to output directory."""
        try:
            dst = out_dir / "pytorch_graph.yaml"
            if src.suffix.lower() in {".yaml", ".yml"}:
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            elif not dst.exists():
                with open(dst, "w") as f:
                    yaml.dump(
                        pytorch_graph, f,
                        Dumper=NoAliasDumper,
                        sort_keys=False,
                        default_flow_style=False
                    )
        except Exception:
            if self._debug:
                logger.debug("Failed to copy/write canonical pytorch_graph.yaml")

    def _load_pytorch_graph(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load PyTorch graph from YAML or JSON file.

        Args:
            path: Path to the graph file.

        Returns:
            The graph dictionary, or None on failure.
        """
        try:
            suffix = path.suffix.lower()

            if suffix in {".yaml", ".yml"}:
                with open(path) as f:
                    data = yaml_safe_load(f)
            elif suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
            else:
                if self._debug:
                    logger.debug(f"Unsupported file extension: {path.suffix}")
                return None

            if isinstance(data, dict) and "layers" in data:
                return data
            if isinstance(data, list):
                return self._convert_node_list(data, model_name=path.stem)

            if self._debug:
                logger.debug(f"Unexpected structure in {path}")
            return None

        except Exception as exc:
            if self._debug:
                logger.debug(f"Failed to load PyTorch graph: {exc}")
            return None

    def _convert_node_list(
        self,
        nodes: List[Dict[str, Any]],
        *,
        model_name: str,
    ) -> Dict[str, Any]:
        """Convert legacy node list format to structured graph dictionary."""
        layers: Dict[str, Any] = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("name") or "unknown"
            layers[node_id] = {
                "type": node.get("node_type", node.get("type", "unknown")),
                "node_class": node.get("node_class", "UnknownNode"),
                "input_shapes": node.get("input_shapes", []) or [],
                "output_shapes": node.get("output_shapes", []) or [],
                "weight_nodes": node.get("weight_nodes", []) or [],
                "weight_shapes": node.get("weight_shapes", []) or [],
                "module_args": node.get("module_args", {}) or {},
                "connections": {
                    "inputs": node.get("input_nodes", []) or [],
                    "outputs": node.get("output_nodes", []) or [],
                },
            }
        return {"model_name": model_name, "layers": layers}

    def _consumed_output_slots(self, node_id: str, node_data: Dict[str, Any]) -> List[int]:
        """Recorded output slots of ``node_id`` that have at least one consumer in the capture
        (an op, or a model output).  Slot k is the k-th tensor in the node's ``connections.outputs``;
        synthetic graphs without tensor nodes report only slot 0 when the node has successors."""
        layers = getattr(self, "_pt_layers", None) or {}
        outs = list((node_data.get("connections") or {}).get("outputs") or [])
        consumed: List[int] = []
        for k, tid in enumerate(outs):
            t = layers.get(tid)
            if t is None:
                continue
            ttype = str(t.get("type", "")).lower()
            if ttype in ("output-tensor",):
                consumed.append(k); continue
            if (t.get("connections") or {}).get("outputs"):
                consumed.append(k)
        return consumed

    def _build_op_graph(
        self,
        pytorch_graph: Dict[str, Any],
    ) -> Tuple[nx.DiGraph, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build operation-only graph by collapsing tensor nodes.

        The input PyTorch graph is typically bipartite (TensorNodes and
        Function/Module nodes). This method collapses tensors and connects
        producer operations to consumer operations.

        Args:
            pytorch_graph: The PyTorch graph dictionary.

        Returns:
            Tuple of (operation graph, start node information, parameter node info).
        """
        layers = pytorch_graph.get("layers") or {}
        # kept for queries about recorded output-slot consumption (expansions, an earlier fix P2-2)
        self._pt_layers = layers
        tensor_ids, op_ids, auxiliary_ids, parameter_ids = self._partition_nodes(layers)

        # Repair every known torchview tracing quirk in one place: dropped
        # scalar edges, orphan/dead-end tensor pairs, and fp32-overridden
        # output dtypes. After this call, ``layers`` is the cleaned source
        # of truth for downstream graph construction and handlers.
        self._repair_torchview_quirks(layers, op_ids, tensor_ids, auxiliary_ids)

        # In-place view ops (``unsqueeze_``, ``squeeze_``, ``transpose_``,
        # ``t_``, ...) alias their operand's storage: torchview records the
        # op's ``input_shapes`` AFTER the in-place reshape, while the tensor
        # node feeding it keeps the shape its producer wrote. Left alone, the
        # producer's output name is referenced with two shapes and the pre-AF
        # consistency check refuses the graph (BigBird: 49 ``unsqueeze_``
        # nodes, [1,12,1,64,64] vs [1,12,64,64]). Re-anchor the op's input
        # shape to the tensor node's recorded shape so the op is modelled as
        # its functional view (same element count, new rank).
        self._alias_inplace_view_input_shapes(layers, op_ids, tensor_ids)

        graph = nx.DiGraph()
        for op_id in op_ids:
            graph.add_node(op_id, **(layers.get(op_id) or {}))
        # Membership tests below run once per tensor edge; a list makes the
        # edge build O(tensors x ops), which dominated conversion time on
        # traces with thousands of ops (per-batch-element loops).
        op_id_set = set(op_ids)

        # Collect auxiliary tensor info for start nodes (model inputs only)
        start_nodes_info = self._collect_start_node_info(
            layers, auxiliary_ids, op_ids
        )

        # Collect parameter tensor info separately (model weights)
        param_nodes_info = self._collect_start_node_info(
            layers, parameter_ids, op_ids
        )

        for tensor_id in tensor_ids:
            tensor_data = layers.get(tensor_id) or {}
            conns = tensor_data.get("connections") or {}
            producers = list(conns.get("inputs") or [])
            consumers = list(conns.get("outputs") or [])

            if len(producers) == 1 and producers[0] in op_id_set:
                self._tensor_to_producer_op[tensor_id] = producers[0]
                producer_outputs = list(
                    ((layers.get(producers[0]) or {}).get("connections") or {}).get("outputs")
                    or []
                )
                producer_out_shapes = (
                    (layers.get(producers[0]) or {}).get("output_shapes") or []
                )
                if tensor_id in producer_outputs:
                    slot = producer_outputs.index(tensor_id)
                    # Only record slots the producer declares a shape for, so
                    # consumer names always match a produced tensor name.
                    if 0 < slot < max(len(producer_out_shapes), 1):
                        self._tensor_to_producer_slot[tensor_id] = slot

            for producer in producers:
                for consumer in consumers:
                    if producer in op_id_set and consumer in op_id_set:
                        if producer != consumer:
                            graph.add_edge(producer, consumer)

        # Fallback: use direct connections if no tensor nodes
        if not tensor_ids:
            for op_id in op_ids:
                conns = (layers.get(op_id) or {}).get("connections") or {}
                outputs = list(conns.get("outputs") or [])
                for out_id in outputs:
                    if out_id in op_id_set and out_id != op_id:
                        graph.add_edge(op_id, out_id)

        return graph, start_nodes_info, param_nodes_info

    # In-place variants of rank/axis-reordering views that keep the element
    # count (``x.unsqueeze_(d)`` returns ``x`` itself with a new shape).
    # ``resize_`` / ``as_strided_`` can change the element count and are not
    # aliases of a functional view; they are left to the consistency check.
    _INPLACE_VIEW_BASES = frozenset({
        "unsqueeze", "squeeze", "transpose", "t", "swapaxes", "swapdims",
    })

    @classmethod
    def _inplace_view_base(cls, op_type: Any) -> Optional[str]:
        """``"unsqueeze_"`` -> ``"unsqueeze"``; None for anything else."""
        name = str(op_type or "").strip()
        if not name.endswith("_") or name.endswith("__"):
            return None
        base = name[:-1].rsplit(".", 1)[-1]
        return base if base in cls._INPLACE_VIEW_BASES else None

    def _alias_inplace_view_input_shapes(
        self,
        layers: Dict[str, Any],
        op_ids: List[str],
        tensor_ids: List[str],
    ) -> int:
        """Record an in-place view op's input at the connected tensor's shape.

        For every op whose type is an in-place view (see
        ``_INPLACE_VIEW_BASES``), each recorded ``input_shapes[i]`` that
        differs from the shape of the tensor node wired into slot ``i`` but
        has the same element count is replaced by that tensor's shape.  A
        slot whose element count differs is not an alias of a view and is left
        for ``_validate_tensor_shape_consistency``.  Returns the number of
        slots rewritten.
        """
        def _numel(shape: Any) -> Optional[int]:
            if not isinstance(shape, (list, tuple)):
                return None
            total = 1
            for d in shape:
                if not isinstance(d, int) or isinstance(d, bool) or d < 0:
                    return None
                total *= d
            return total

        tensor_id_set = set(tensor_ids)
        rewritten = 0
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            if self._inplace_view_base(odata.get("type")) is None:
                continue
            in_ids = list((odata.get("connections") or {}).get("inputs") or [])
            in_shapes = odata.get("input_shapes")
            if not isinstance(in_shapes, list):
                continue
            for i, tid in enumerate(in_ids):
                if i >= len(in_shapes) or tid not in tensor_id_set:
                    continue
                tdata = layers.get(tid) or {}
                tshapes = tdata.get("output_shapes") or tdata.get("input_shapes") or []
                if not tshapes or tshapes[0] is None or in_shapes[i] is None:
                    continue
                pre = list(tshapes[0])
                post = list(in_shapes[i])
                if pre == post:
                    continue
                if _numel(pre) is None or _numel(pre) != _numel(post):
                    continue
                in_shapes[i] = pre
                rewritten += 1
                if self._debug:
                    logger.info(f"[in-place view alias] {op_id} ({odata.get('type')}): "
                        f"input[{i}] {post} -> {pre} (shape of {tid})")
        return rewritten

    def _synthesize_constant_operand(
        self,
        layers: Dict[str, Any],
        op_id: str,
        odata: Dict[str, Any],
        shape: List[int],
        dtype: str,
        auxiliary_ids: Optional[List[str]],
    ) -> str:
        """Add an ``auxiliary-tensor`` node feeding ``op_id`` for an untraced operand."""
        k = 0
        aux_id = f"{op_id}.const"
        while aux_id in layers:
            k += 1
            aux_id = f"{op_id}.const_{k}"
        layers[aux_id] = {
            "type": "auxiliary-tensor",
            "node_class": "TensorNode",
            "input_shapes": [],
            "output_shapes": [list(shape)],
            "input_dtypes": [dtype],
            "output_dtypes": [dtype],
            "input_types": [],
            "output_types": ["output"],
            "module_args": {
                "hierarchical_name": aux_id,
                "synthesized": "untraced constant operand (from raw_attributes)",
            },
            "connections": {"inputs": [], "outputs": [op_id]},
        }
        odata.setdefault("input_shapes", []).append(list(shape))
        odata.setdefault("input_dtypes", []).append(dtype)
        odata.setdefault("input_types", []).append("input")
        odata.setdefault("connections", {}).setdefault("inputs", []).append(aux_id)
        if auxiliary_ids is not None:
            auxiliary_ids.append(aux_id)
        if self._debug:
            logger.info(f"[repair A'] {op_id}: synthesized constant operand {aux_id} shape={shape}")
        return aux_id

    def _partition_nodes(
        self,
        layers: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Partition nodes into tensor, operation, and auxiliary categories.

        Args:
            layers: The layers dictionary from the PyTorch graph.

        Returns:
            Tuple of (tensor_ids, op_ids, auxiliary_tensor_ids, parameter_tensor_ids).
        """
        tensor_ids: List[str] = []
        op_ids: List[str] = []
        auxiliary_ids: List[str] = []

        parameter_ids: List[str] = []

        for node_id, data in (layers or {}).items():
            node_class = (data.get("node_class") or "").lower()
            node_type = (data.get("type") or "").lower()

            # A recorded node class is authoritative: TensorNode is a
            # tensor-side node, FunctionNode / ModuleNode is an operation
            # whatever its name (``aten.add.Tensor``, ``mul.Tensor`` or an
            # opaque ``*_tensor_kernel`` used to vanish here before reaching
            # a handler or the fail-closed refusal, tensor-node-2).
            # Without a node class, only torchview's tensor-node names
            # (``hidden-tensor`` etc.) are tensor-side.
            if node_class:
                is_tensor_side = "tensornode" in node_class
            else:
                is_tensor_side = node_type in _TENSOR_NODE_TYPES or node_type.endswith("-tensor")
            if is_tensor_side:
                if node_type == "auxiliary-tensor":
                    auxiliary_ids.append(node_id)
                elif node_type == "parameter-tensor":
                    parameter_ids.append(node_id)
                else:
                    tensor_ids.append(node_id)
            else:
                op_ids.append(node_id)

        return tensor_ids, op_ids, auxiliary_ids, parameter_ids

    def _collect_start_node_info(
        self,
        layers: Dict[str, Any],
        auxiliary_ids: List[str],
        op_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Collect information about auxiliary tensors to create start nodes."""
        start_nodes_info: List[Dict[str, Any]] = []
        op_id_set = set(op_ids)

        for idx, aux_id in enumerate(auxiliary_ids):
            aux_data = layers.get(aux_id) or {}
            conns = aux_data.get("connections") or {}
            output_shapes = aux_data.get("output_shapes") or []
            consumers = list(conns.get("outputs") or [])
            # Filter to only include operation nodes
            valid_consumers = [c for c in consumers if c in op_id_set]

            output_dtypes = aux_data.get("output_dtypes") or []

            start_nodes_info.append({
                "original_id": aux_id,
                "index": idx,
                "output_shapes": output_shapes,
                "output_dtypes": output_dtypes,
                "consumers": valid_consumers,
            })

        return start_nodes_info

    def _expand_complex_ops(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations using GraphExpander (best-effort)."""
        if not graph.nodes:
            return graph

        try:
            from solar.einsum.graph_expander import GraphExpander

            expander = GraphExpander(
                debug=self._debug,
                enable_agent=self._enable_agent,
                api_key=self._api_key,
                cache_dir=self._cache_dir,
            )
            return expander.expand(graph)
        except Exception:
            return graph

    def _build_einsum_graph(
        self,
        pytorch_graph: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        param_nodes_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build einsum graph dictionary from operation graph."""
        result: Dict[str, Any] = {
            "model_name": pytorch_graph.get("model_name", "pytorch_model"),
            "layers": {},
        }

        # Add start nodes from auxiliary tensors (model inputs only)
        self._start_node_targets = None
        start_node_id_map = self._add_start_nodes(result, start_nodes_info)

        # Combine start + param info for ID mapping in _convert_operation.
        # Parameter nodes don't get their own einsum layers.
        all_source_nodes_info = list(start_nodes_info) + list(param_nodes_info or [])

        # Map parameter node original IDs so _convert_operation can find them
        for info in (param_nodes_info or []):
            original_id = info["original_id"]
            start_node_id_map[original_id] = original_id
        # Parameter identity: the capture's ``parameter-tensor`` node ids.  A
        # ``weight`` slot fed by one of them records that id in the layer's
        # ``parameter_sources`` (see _convert_operation); the analyzer counts
        # the parameter footprint once per distinct id per consuming op and
        # treats a ``weight`` slot without one as provenance unknown
        # (parameter-identity-1 / provenance-1).
        self._parameter_node_ids = {str(info["original_id"]) for info in (param_nodes_info or [])}

        # Map hidden-tensor IDs to their producer op so all downstream
        # code (connections, tensor_names) resolves them automatically.
        for tensor_id, producer_op in self._tensor_to_producer_op.items():
            if tensor_id not in start_node_id_map:
                start_node_id_map[tensor_id] = producer_op
        # ``x in start_node_id_map.values()`` is a linear scan over every
        # tensor in the graph; done once per op input it made
        # _convert_operation quadratic (93% of einsum-stage time on a
        # 4.8k-node trace). Cache the value set for the per-op lookups.
        self._start_node_targets = set(start_node_id_map.values())

        # Track node ID remapping for split/expanded operations
        # Maps original node_id -> final output node_id
        node_id_remap: Dict[str, str] = {}

        # Track expanded nodes' input mappings
        # Maps original node_id -> {input_index -> subgraph_node_id}
        expanded_input_map: Dict[str, Dict[int, str]] = {}
        expanded_output_slots: Dict[str, Dict[int, str]] = {}

        # Convert each operation to einsum representation
        for node_id in op_graph.nodes():
            node_data = dict(op_graph.nodes[node_id] or {})
            self._validate_input_types_alignment(node_id, node_data)

            # Check if this is a linear layer with bias that should be split
            if self._should_split_linear_with_bias(node_data):
                matmul_layer, add_layer = self._split_linear_with_bias(
                    node_id, node_data, op_graph, all_source_nodes_info, start_node_id_map
                )
                result["layers"][node_id] = matmul_layer
                add_node_id = f"{node_id}.bias_add"
                result["layers"][add_node_id] = add_layer
                # Remap: original node_id outputs now come from add_node_id
                node_id_remap[node_id] = add_node_id

            # Check if this is a group-wise conv that needs reshape expansion
            elif self._should_expand_groupwise_conv(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_groupwise_conv(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            elif self._expansion_for(node_data) is not None:
                expand = self._expansion_for(node_data)
                try:
                    expansion = expand(
                        node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                    )
                    subgraph_layers, final_node_id, input_mapping = expansion[:3]
                    output_slots = dict(expansion[3]) if len(expansion) > 3 else {}
                except ValueError as exc:
                    # Operands the expansion needs were not recorded; keep the
                    # node as one opaque layer rather than drop it.
                    logger.warning("%s: not expanded (%s)", node_id, exc)
                    result["layers"][node_id] = self._convert_operation(
                        node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                    )
                else:
                    for sub_id, sub_layer in subgraph_layers.items():
                        result["layers"][sub_id] = sub_layer
                    # Remap: original node_id outputs now come from final subgraph node
                    node_id_remap[node_id] = final_node_id
                    # Store input mapping for predecessor updates
                    expanded_input_map[node_id] = input_mapping
                    if output_slots:
                        expanded_output_slots[node_id] = output_slots

            else:
                layer_dict = self._convert_operation(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                result["layers"][node_id] = layer_dict

        # Fix connections for split/expanded operations
        self._fix_split_connections(result, node_id_remap, expanded_input_map, expanded_output_slots)

        # Record which einsum layers feed the model's declared outputs
        # (torchview ``output-tensor`` nodes). An op whose result is neither
        # consumed nor returned is dead code or has an untraced in-place
        # consumer (``result[mask] = v``); the analyzer uses this list to
        # avoid charging a DRAM write for such tensors.
        output_ops: List[str] = []
        model_outputs: List[Dict[str, Any]] = []
        pt_layers = pytorch_graph.get("layers") or {}
        for node_id, node_data in pt_layers.items():
            if str(node_data.get("type", "")).lower() != "output-tensor":
                continue
            shapes = node_data.get("input_shapes") or []
            dtypes = node_data.get("input_dtypes") or []
            for i, src in enumerate((node_data.get("connections") or {}).get("inputs") or []):
                dtype = dtypes[i] if i < len(dtypes) else None
                # Einsum-side name of the tensor handed to the output: the
                # analyzer uses it to recognise reads of an output buffer
                # whose producer torchview lost (orphan accumulators).
                first_src = node_id_remap.get(src, src)
                tensor_name = f"{first_src}.Output"
                # an expanded node's consumed state output (h_n / c_n) has its own producer
                _slot = self._tensor_to_producer_slot.get(src)
                _prod = self._tensor_to_producer_op.get(src)
                if _slot and _prod in expanded_output_slots and _slot in expanded_output_slots[_prod]:
                    first_src = expanded_output_slots[_prod][_slot]
                    tensor_name = f"{first_src}.Output"
                first_type = str((pt_layers.get(src) or {}).get("type", "")).lower()
                is_input = first_type in ("input-tensor", "auxiliary-tensor")
                # The output may be reached through tensor nodes
                # (``hidden-tensor``) rather than directly from the op;
                # follow them to the producing op, taking the (repaired)
                # dtype of the tensor on the way.
                hops = 0
                while src is not None and src not in result["layers"] and hops < 8:
                    tnode = pt_layers.get(src)
                    if tnode is None:
                        break
                    d = (tnode.get("output_dtypes") or tnode.get("input_dtypes") or [None])[0]
                    if d:
                        dtype = d
                    ins = (tnode.get("connections") or {}).get("inputs") or []
                    src = node_id_remap.get(ins[0], ins[0]) if ins else None
                    hops += 1
                src = node_id_remap.get(src, src) if src is not None else None
                if src in result["layers"] and src not in output_ops:
                    output_ops.append(src)
                model_outputs.append({
                    "op": src if src in result["layers"] else None,
                    "tensor": tensor_name,
                    "is_input": is_input,
                    "shape": list(shapes[i]) if i < len(shapes) and isinstance(shapes[i], (list, tuple)) else None,
                    "dtype": dtype,
                })
        # Returned outputs the capture recorded from forward()'s actual return
        # value (issue #53).  A tensor leaf torchview never traced (allocated
        # in forward and filled in place, e.g. ``zeros(n); out[1:] = ...``)
        # has no output-tensor node: declare it as a model output with its
        # shape/dtype (the analyzer charges the write once) and flag it so
        # the accounting fails closed instead of silently pricing only the
        # bound outputs.
        returned = pytorch_graph.get("returned_outputs")
        if isinstance(returned, list) and returned:
            result["returned_outputs"] = [dict(r) for r in returned]
            result["model_output_provenance"] = pytorch_graph.get(
                "model_output_provenance") or "pytorch-return-v1"
            # return-hook-error-2: an extraction failure is an explicit
            # incomplete reason for the analyzer (never a silent legacy look).
            unrecorded = [r for r in returned if isinstance(r, dict) and r.get("kind") == "unrecorded"]
            if unrecorded or pytorch_graph.get("returned_outputs_unrecorded"):
                result["returned_outputs_unrecorded"] = True
                result["returned_outputs_error"] = pytorch_graph.get("returned_outputs_error") or "; ".join(
                    str(r.get("error")) for r in unrecorded)
            # return-alias-1 / -STRIDE-4 / -REGION-5: the footprint of the
            # UNBOUND returned leaves is the per-STORAGE union of their byte
            # regions (offset / shape / stride / element size), billed once per
            # storage.  Exact aliases collapse to one region; overlapping views
            # (``b[:5]``, ``b[3:]``) bill their union, not their sum; disjoint
            # views (``b[:4]``, ``b[4:]``) bill both; same offset/numel with
            # different strides (``b[:4]``, ``b[::2]``) bills the union of the
            # bytes actually addressed.  Every leaf keeps its own provenance
            # record in ``returned_outputs``; the declared model output carries
            # the paths it stands for.  Deduplicating by storage alone would
            # under-bill disjoint views, so the union is over regions, never a
            # single whole-storage charge.
            # real-alias-1: BOUND leaves (declared above through their traced
            # output-tensor nodes) join the per-storage grouping too, so a plain
            # alias of a traced output (``y, y.as_subclass(torch.Tensor)``) is not
            # charged a second time: the unbound footprint of a storage is the
            # union of its unbound regions MINUS the regions already charged
            # through bound outputs of the same storage.  Distinct storages and
            # disjoint regions are unaffected.  A leaf whose storage identity is an
            # ``object_fallback`` can never be matched to another leaf.
            groups: Dict[str, Dict[str, Any]] = {}
            for i, leaf in enumerate(returned):
                if not isinstance(leaf, dict) or leaf.get("kind") != "tensor":
                    continue
                ident = leaf.get("tensor_identity") or {}
                bound = leaf.get("bound_node") is not None
                sid = ident.get("storage_id") or f"leaf:{i}"
                if str(sid).startswith("obj:") or ident.get("storage_identity") == "object_fallback":
                    sid = f"{sid}:leaf{i}"   # object identity never aliases across leaves
                region = _returned_leaf_region(ident, leaf)
                region["path"] = leaf.get("path")
                method = region["method"]
                g = groups.setdefault(str(sid), {"first_index": None, "leaves": [], "regions": [], "bound_regions": [],
                                                 "bound_leaves": [], "methods": set(), "dtypes": set(), "storage_nbytes": None})
                rec_leaf = {"index": i, "path": leaf.get("path"), "shape": leaf.get("shape"),
                            "dtype": leaf.get("dtype"), "storage_key": ident.get("storage_key"),
                            "storage_identity": ident.get("storage_identity"),
                            "storage_offset": ident.get("storage_offset"), "stride": ident.get("stride"),
                            "element_size": ident.get("element_size"), "region_bytes": int(region["bytes"]),
                            "region_method": method, "region_exact": bool(region["exact"]), "bound_node": leaf.get("bound_node")}
                if ident.get("storage_nbytes"):
                    g["storage_nbytes"] = int(ident["storage_nbytes"])
                if bound:
                    g["bound_leaves"].append(rec_leaf); g["bound_regions"].append(region)
                    continue
                if g["first_index"] is None:
                    g["first_index"] = i
                g["leaves"].append(rec_leaf)
                g["regions"].append(region)
                g["methods"].add(method)
                g["dtypes"].add(str(leaf.get("dtype")))
            footprint: Dict[str, Any] = {}
            for sid, g in sorted(groups.items(), key=lambda kv: (kv[1]["first_index"] is None, kv[1]["first_index"] or 0)):
                if not g["leaves"]:
                    # only bound leaves on this storage: charged through their traced
                    # outputs above; record any bound/bound overlap as an observation
                    # (baseline-owned traced-output accounting, not changed here)
                    if len(g["bound_leaves"]) > 1:
                        raw = sum(l["region_bytes"] for l in g["bound_leaves"])
                        obs = _returned_group_footprint(g["bound_regions"], [], g["storage_nbytes"])
                        footprint[sid] = {"union_bytes": 0, "regions": [], "leaves": [], "bound_leaves": g["bound_leaves"],
                                          "bound_region_union_bytes": int(obs["union_bytes"]), "bound_region_sum_bytes": raw,
                                          "bound_region_union_exact": bool(obs["unbound_exact"]),
                                          "method": "bound_only", "storage_nbytes": g["storage_nbytes"],
                                          "note": "bound leaves sharing a storage are charged through their traced output nodes; overlap recorded, not re-billed here"}
                    continue
                # bound-envelope-1: subtract ONLY proven coverage.  Exact regions
                # (intervals, strided lattices) are combined analytically; an approximate
                # unbound region bills its bounding extent (over-count); an approximate
                # bound region covers nothing.  See _returned_group_footprint.
                fp = _returned_group_footprint(g["regions"], g["bound_regions"], g["storage_nbytes"])
                union_bytes = fp["union_bytes"]
                covered_bytes = fp["covered_bytes"]
                merged = fp["remainder_regions"]
                if union_bytes == 0 and covered_bytes > 0 and fp["unbound_exact"] and fp["coverage_exact"]:
                    # every byte of the unbound leaves is PROVEN charged through a bound
                    # output of the same storage (exact regions on both sides): an exact
                    # (or sub-) alias of a traced output -- its producer IS the traced op,
                    # nothing to declare.  An approximate region on either side never
                    # reaches here: the leaf stays declared, with its incomplete marker.
                    # (An EMPTY unbound leaf covers no bytes and is not an alias: it
                    # falls through and is declared with a zero footprint + marker.)
                    footprint[sid] = {"union_bytes": 0, "regions": [], "leaves": g["leaves"], "bound_leaves": g["bound_leaves"],
                                      "covered_by_bound_bytes": int(covered_bytes), "method": "alias_of_bound_output",
                                      "coverage_exact": True, "storage_nbytes": g["storage_nbytes"],
                                      "note": "unbound returned leaf fully covered by bound output(s) of the same storage; charged once through the traced output"}
                    continue
                esizes = {int(l["element_size"]) for l in g["leaves"] if l.get("element_size")}
                if len(g["dtypes"]) == 1 and len(esizes) == 1 and union_bytes % next(iter(esizes)) == 0:
                    dtype = next(iter(g["dtypes"]))
                    shape = [union_bytes // next(iter(esizes))]
                else:
                    # mixed element widths in one storage: bill the union as bytes
                    dtype = "torch.uint8"
                    shape = [union_bytes]
                i = g["first_index"]
                method = "byte_region_union" if g["methods"] <= {"byte_region_union"} else "+".join(sorted(g["methods"]))
                model_outputs.append({
                    "op": None,
                    "tensor": f"returned_output_{i}",
                    "is_input": False,
                    "shape": shape,
                    "dtype": dtype,
                    "unbound_returned": True,
                    "returned_path": g["leaves"][0]["path"],
                    "returned_paths": [l["path"] for l in g["leaves"]],
                    "returned_leaf_indices": [l["index"] for l in g["leaves"]],
                    "storage_id": sid,
                    "footprint_bytes": int(union_bytes),
                    "footprint_method": method,
                })
                footprint[sid] = {"union_bytes": int(union_bytes),
                                  "regions": [[int(a), int(b)] for a, b in merged] if merged is not None else None,
                                  "leaves": g["leaves"], "bound_leaves": g["bound_leaves"], "covered_by_bound_bytes": int(covered_bytes),
                                  "method": method, "storage_nbytes": g["storage_nbytes"],
                                  "unbound_exact": bool(fp["unbound_exact"]), "coverage_exact": bool(fp["coverage_exact"]),
                                  **({"lattice_period_bytes": fp["period"]} if fp["period"] else {}),
                                  **({"grid_work": fp["grid_work"]} if fp.get("grid_work") else {}),
                                  **({"complexity_fallback": fp["complexity_fallback"]} if fp.get("complexity_fallback") else {}),
                                  **({"approximate_unbound": fp["approximate_unbound"]} if fp["approximate_unbound"] else {}),
                                  **({"unproven_bound": fp["unproven_bound"],
                                      "note": "bound region(s) known only as a bounding interval / legacy assumption: not subtracted (an upper bound on touched bytes is no proof those bytes were charged)"}
                                     if fp["unproven_bound"] else {})}
            if footprint:
                result["returned_footprint"] = footprint
        if output_ops:
            result["model_output_ops"] = output_ops
        if model_outputs:
            result["model_outputs"] = model_outputs

        # Contiguity recorded at capture for model inputs and parameters
        # (an earlier decision round 3, an earlier decision review input-2), keyed by einsum
        # tensor name so the analyzer's reshape_as decision can use it.
        tensor_layouts = self._collect_tensor_layouts(pt_layers, result, start_node_id_map)
        if tensor_layouts:
            result["tensor_layouts"] = tensor_layouts

        return result

    @staticmethod
    def _collect_tensor_layouts(
        pt_layers: Dict[str, Any],
        result: Dict[str, Any],
        start_node_id_map: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """Recorded layouts of model inputs / parameters by einsum tensor name.

        ``PyTorchProcessor._patch_tensor_layouts`` records ``module_args.layout``
        (``contiguous``, ``stride`` when unambiguous, ``evidence``) on the
        auxiliary-/input-tensor and parameter-tensor nodes of
        ``pytorch_graph.yaml``.  A start layer maps through its start id; a
        parameter maps through the consuming layer's positional weight operand
        (``<layer>.Weight[_k]``).  Nothing is inferred here: a tensor without a
        record is absent and the analyzer treats its layout as unknown.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for pt_id, node in pt_layers.items():
            if str(node.get("type", "")).lower() not in ("auxiliary-tensor", "input-tensor"):
                continue
            layout = (node.get("module_args") or {}).get("layout")
            start_id = start_node_id_map.get(pt_id)
            if isinstance(layout, dict) and start_id:
                out[f"{start_id}.Output"] = dict(layout)
        for layer_id, layer in result["layers"].items():
            node = pt_layers.get(layer_id)
            if not node:
                continue
            pt_inputs = node.get("input_nodes") or (node.get("connections") or {}).get("inputs") or []
            names = (layer.get("tensor_names") or {}).get("inputs") or []
            types = (layer.get("tensor_types") or {}).get("inputs") or []
            if len(names) != len(pt_inputs):
                continue
            for i, name in enumerate(names):
                if i >= len(types) or str(types[i]).lower() != "weight":
                    continue
                pnode = pt_layers.get(pt_inputs[i]) or {}
                if str(pnode.get("type", "")).lower() != "parameter-tensor":
                    continue
                layout = (pnode.get("module_args") or {}).get("layout")
                if isinstance(layout, dict):
                    out[name] = dict(layout)
        return out

    def _expansion_for(self, node_data: Dict[str, Any]):
        """The subgraph expansion for this node type, or None."""
        if self._should_expand_mha(node_data):
            return self._expand_mha
        if self._should_expand_lstm(node_data):
            return self._expand_lstm
        if self._should_expand_gru(node_data):
            return self._expand_gru
        if self._should_expand_rnn(node_data):
            return self._expand_rnn_tanh if self._rnn_nonlinearity(node_data) == "tanh" else self._expand_rnn_relu
        if self._should_expand_sdpa(node_data):
            return self._expand_sdpa
        return None

    @staticmethod
    def _rnn_nonlinearity(node_data: Dict[str, Any]) -> str:
        return "relu" if "relu" in str(node_data.get("type", "")).lower() else "tanh"

    def _should_expand_rnn(self, node_data: Dict[str, Any]) -> bool:
        """torch.rnn_tanh / torch.rnn_relu (nn.RNN) get the LSTM/GRU expansion (an earlier fix P2-1)."""
        node_type = str(node_data.get("type", "")).lower()
        return node_type in {"rnn_tanh", "rnn_relu"}

    def _should_expand_mha(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a multi_head_attention_forward that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"multi_head_attention_forward", "multihead_attention"}

    def _should_expand_sdpa(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a scaled_dot_product_attention that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"scaled_dot_product_attention", "sdpa", "attention"}

    def _should_expand_lstm(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is an LSTM that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"lstm"}

    def _should_expand_gru(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a GRU that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"gru"}

    def _expand_sdpa(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand scaled_dot_product_attention; see :func:`solar.einsum.expansions.expand_sdpa`."""
        return expand_sdpa(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _start_targets(self, start_node_id_map: Dict[str, str]) -> "set[str]":
        """Set of start-node / producer ids that inputs may map to (cached per conversion)."""
        cached = getattr(self, "_start_node_targets", None)
        if cached is None:
            cached = set(start_node_id_map.values())
            self._start_node_targets = cached
        return cached

    def _ordered_input_producers(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> List[str]:
        """Producer ids of ``node_id``'s inputs in PyTorch argument order.

        Walks ``connections.inputs`` (tensor ids, in call order) and maps
        each to the op / start node that produces it: start-node map first
        (also carries hidden-tensor -> producer entries), then op-graph
        nodes, then ``_tensor_to_producer_op``.  Inputs that cannot be
        resolved are filled positionally from the still-unassigned
        predecessors when the counts match.  Any predecessor or consumed
        start node not yet listed is appended (sorted) so no edge is lost.
        Used by the op expanders (SDPA) that previously relied on sorted
        predecessor names, which does not reflect argument order.
        """
        raw_inputs = list((node_data.get("connections") or {}).get("inputs") or [])
        preds = list(op_graph.predecessors(node_id))
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        ordered: List[Optional[str]] = []
        for tid in raw_inputs:
            if tid in start_node_id_map:
                ordered.append(start_node_id_map[tid])
            elif tid in op_graph.nodes:
                ordered.append(tid)
            elif tensor_to_producer.get(tid) in op_graph.nodes:
                ordered.append(tensor_to_producer[tid])
            else:
                ordered.append(None)
        unresolved = [i for i, x in enumerate(ordered) if x is None]
        remaining = [p for p in preds if p not in ordered]
        if unresolved and len(unresolved) == len(remaining):
            for i, p in zip(unresolved, remaining):
                ordered[i] = p
        result = [x for x in ordered if x is not None]
        for p in sorted(preds):
            if p not in result:
                result.append(p)
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in result:
                    result.append(start_id)
        return result

    def _should_expand_groupwise_conv(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a group-wise convolution that needs reshape expansion.

        Currently only conv1d / conv2d are expanded via the reshape pass.
        Conv3d and conv-transpose variants rely on the AF graph builder's
        union-find canonicalization to handle the C_out vs C_out/groups
        split, since the reshape path doesn't survive the parameter-tensor
        weight (still 4-/5-dim original) <-> 5-/6-dim grouped einsum mismatch.
        """
        node_type = str(node_data.get("type", "")).lower()
        if node_type not in ("conv1d", "conv2d"):
            return False

        module_args = node_data.get("module_args") or {}
        groups = int(conv_call_kwargs(module_args).get("groups") or 1)
        if groups <= 1:
            return False

        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        in_channels = int(
            module_args.get(
                "in_channels",
                input_shapes[0][1] if input_shapes and len(input_shapes[0]) > 1 else 0,
            )
        )
        out_channels = int(
            module_args.get(
                "out_channels",
                output_shapes[0][1] if output_shapes and len(output_shapes[0]) > 1 else 0,
            )
        )

        # Depthwise conv is handled directly by the conv handler.
        if groups == in_channels and groups == out_channels:
            return False
        return True

    def _expand_groupwise_conv(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: "nx.DiGraph",
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand group-wise conv into input view, grouped conv, and output view."""
        module_args = node_data.get("module_args") or {}
        groups = int(conv_call_kwargs(module_args).get("groups") or 1)
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        node_type = str(node_data.get("type", "conv2d")).lower()
        is_2d = node_type == "conv2d"

        input_shape = list(input_shapes[0]) if len(input_shapes) > 0 else []
        weight_shape = list(input_shapes[1]) if len(input_shapes) > 1 else []
        output_shape = list(output_shapes[0]) if output_shapes else []

        B = input_shape[0]
        C_in = input_shape[1]
        O_total = weight_shape[0] if weight_shape else output_shape[1]
        I = C_in // groups
        O_pg = O_total // groups

        if is_2d:
            H, W = input_shape[2], input_shape[3]
            KH, KW = weight_shape[2], weight_shape[3]
            H_out, W_out = output_shape[2], output_shape[3]
            reshaped_input = [B, groups, I, H, W]
            reshaped_weight = [groups, O_pg, I, KH, KW]
            reshaped_output = [B, groups, O_pg, H_out, W_out]
            reshape_in_eq = "ABCD->AE0E1CD"
            reshape_in_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "E1", "C", "D"],
            }
            reshape_out_eq = "ABCDE->AF0DE"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D", "E"],
                "Output": ["A", "F0", "D", "E"],
            }
            fallback_conv_equation = "BGI(P+R)(Q+S),GOIRS->BGOPQ"
        else:
            L = input_shape[2]
            KL = weight_shape[2]
            L_out = output_shape[2]
            reshaped_input = [B, groups, I, L]
            reshaped_weight = [groups, O_pg, I, KL]
            reshaped_output = [B, groups, O_pg, L_out]
            reshape_in_eq = "ABC->ADE0C"
            reshape_in_operands = {
                "Input": ["A", "B", "C"],
                "Output": ["A", "D", "E0", "C"],
            }
            reshape_out_eq = "ABCD->AE0D"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "D"],
            }
            fallback_conv_equation = "BGI(P+R),GOIR->BGOP"

        raw_input_connections = list((node_data.get("connections") or {}).get("inputs") or [])
        if not raw_input_connections:
            raw_input_connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in raw_input_connections:
                    raw_input_connections.append(original_id)

        # Strict tensor→producer resolution (see notes in the twin block of
        # `_convert_operation`). Fall back to op_graph.predecessors only when
        # exactly enough unmatched predecessors remain to fill all deferred
        # slots — raise on ambiguity rather than silently positional-guess.
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        op_predecessors = list(op_graph.predecessors(node_id))
        input_connections: List[Optional[str]] = []
        assigned_preds: set = set()
        deferred_indices: List[int] = []
        input_types_raw = list(node_data.get("input_types") or [])
        for idx, conn_id in enumerate(raw_input_connections):
            mapped = start_node_id_map.get(conn_id, conn_id)
            itype = str(input_types_raw[idx]).lower() if idx < len(input_types_raw) else "input"
            if itype == "weight":
                input_connections.append(mapped)
                continue
            if mapped in self._start_targets(start_node_id_map) or mapped in op_graph.nodes:
                input_connections.append(mapped)
                assigned_preds.add(mapped)
                continue
            producer = tensor_to_producer.get(conn_id)
            if producer is not None and producer in op_graph.nodes:
                input_connections.append(producer)
                assigned_preds.add(producer)
                continue
            input_connections.append(None)
            deferred_indices.append(idx)
        if deferred_indices:
            unmatched_preds = [p for p in op_predecessors if p not in assigned_preds]
            if len(unmatched_preds) == len(deferred_indices):
                for d_idx, pred in zip(deferred_indices, unmatched_preds):
                    input_connections[d_idx] = pred
            elif len(unmatched_preds) == 0:
                # Genuinely producerless input — fall through to literal name.
                for d_idx in deferred_indices:
                    input_connections[d_idx] = start_node_id_map.get(
                        raw_input_connections[d_idx], raw_input_connections[d_idx]
                    )
            else:
                raise ValueError(
                    f"_convert_operation({node_id!r}, conv path): ambiguous "
                    f"predecessor resolution; deferred={len(deferred_indices)}, "
                    f"unmatched_preds={len(unmatched_preds)}."
                )

        output_connections = list(op_graph.successors(node_id))
        if not output_connections:
            raw_outs = list((node_data.get("connections") or {}).get("outputs") or [])
            output_connections = [c for c in raw_outs if c in op_graph.nodes]

        input_types = node_data.get("input_types") or []
        activation_conn = None
        weight_conn = None
        for idx, conn in enumerate(input_connections):
            itype = input_types[idx] if idx < len(input_types) else "input"
            if str(itype).lower() == "weight" or "parameter-tensor" in conn:
                if weight_conn is None:
                    weight_conn = conn
            elif activation_conn is None:
                activation_conn = conn
        if activation_conn is None and input_connections:
            activation_conn = input_connections[0]
        if weight_conn is None and len(input_connections) > 1:
            weight_conn = input_connections[1]

        reshape_in_id = f"{node_id}.reshape_input"
        conv_id = f"{node_id}.groupwise_conv"
        reshape_out_id = f"{node_id}.reshape_output"

        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = input_dtypes[1] if len(input_dtypes) > 1 else activation_dtype
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        weight_tensor_name = f"{weight_conn}.Output" if weight_conn else f"{conv_id}.Weight"

        reshape_in_layer = {
            "type": "view",
            "einsum_equation": reshape_in_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{activation_conn}.Output" if activation_conn else f"{reshape_in_id}.Input"],
                "outputs": [f"{reshape_in_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(input_shape)],
                "outputs": [reshaped_input],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype],
                "outputs": [activation_dtype],
            },
            "operands": reshape_in_operands,
            "connections": {
                "inputs": [activation_conn] if activation_conn else [],
                "outputs": [conv_id],
            },
        }

        stride = self._as_list(module_args.get("stride"), [1, 1] if is_2d else [1])
        padding = self._as_list(module_args.get("padding"), [0, 0] if is_2d else [0])
        dilation = self._as_list(module_args.get("dilation"), [1, 1] if is_2d else [1])

        conv_ts = TensorShapes(
            inputs=[reshaped_input, reshaped_weight],
            outputs=[reshaped_output],
        )
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op(
                node_type,
                conv_ts,
                module_args=module_args,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            conv_equation = einsum_op.equation
            conv_operands = {operand.name: operand.dims for operand in einsum_op.operands}
        except Exception:
            conv_equation = fallback_conv_equation
            if is_2d:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R", "Q+S"],
                    "Weight": ["G", "O", "I", "R", "S"],
                    "Output": ["B", "G", "O", "P", "Q"],
                }
            else:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R"],
                    "Weight": ["G", "O", "I", "R"],
                    "Output": ["B", "G", "O", "P"],
                }

        conv_layer = {
            "type": node_type,
            "einsum_equation": conv_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{reshape_in_id}.Output", weight_tensor_name],
                "outputs": [f"{conv_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_input, reshaped_weight],
                "outputs": [reshaped_output],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype, weight_dtype],
                "outputs": [output_dtype],
            },
            "parameter_sources": {
                "inputs": [None, str(weight_conn)
                           if weight_conn is not None
                           and str(weight_conn) in (getattr(self, "_parameter_node_ids", None) or set())
                           else None],
            },
            "operands": conv_operands,
            "connections": {
                "inputs": [reshape_in_id, weight_conn] if weight_conn else [reshape_in_id],
                "outputs": [reshape_out_id],
            },
        }

        reshape_out_layer = {
            "type": "view",
            "einsum_equation": reshape_out_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{conv_id}.Output"],
                "outputs": [f"{reshape_out_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_output],
                "outputs": [list(output_shape)],
            },
            "tensor_dtypes": {
                "inputs": [output_dtype],
                "outputs": [output_dtype],
            },
            "operands": reshape_out_operands,
            "connections": {
                "inputs": [conv_id],
                "outputs": output_connections,
            },
        }

        subgraph = {
            reshape_in_id: reshape_in_layer,
            conv_id: conv_layer,
            reshape_out_id: reshape_out_layer,
        }

        input_mapping = {}
        if activation_conn:
            input_mapping[0] = reshape_in_id
        if weight_conn:
            input_mapping[1] = conv_id

        return subgraph, reshape_out_id, input_mapping

    @staticmethod
    def _as_list(value: Any, default: List[int]) -> List[Any]:
        """Normalize scalar/list convolution args to a list."""
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _expand_mha(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand multi_head_attention_forward; see :func:`solar.einsum.expansions.expand_mha`."""
        return expand_mha(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _expand_lstm(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand torch.lstm; see :func:`solar.einsum.expansions.expand_lstm`."""
        return expand_lstm(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _expand_gru(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand torch.gru; see :func:`solar.einsum.expansions.expand_gru`."""
        return expand_gru(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _expand_rnn_tanh(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand torch.rnn_tanh; see :func:`solar.einsum.expansions.expand_rnn_tanh`."""
        return expand_rnn_tanh(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _expand_rnn_relu(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand torch.rnn_relu; see :func:`solar.einsum.expansions.expand_rnn_relu`."""
        return expand_rnn_relu(self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)

    def _linear_operands(self, node_data: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
        """Input slots of ``linear(input, weight, bias)``, or None if there is no bias.

        Slots are bound to the signature of ``F.linear`` from the recorded
        call (``F.linear(x, bias=b, weight=w)`` binds the same as
        ``F.linear(x, w, b)``); labels only say whether a tensor is a
        parameter. Shapes that contradict the roles return None, so the node
        is converted unsplit rather than mis-split.
        """
        if str(node_data.get("type", "")).lower() != "linear":
            return None
        shapes = node_data.get("input_shapes") or []
        types = list(node_data.get("input_types") or [])
        slots = [
            Operand(i, "", list(shape), None, str(types[i]).lower() if i < len(types) else "input", None)
            for i, shape in enumerate(shapes)
        ]
        roles = bind_operands(node_data, slots, SIGNATURES["linear"])
        x, weight, bias = (roles.get(r) for r in ("input", "weight", "bias"))
        if not all(isinstance(o, Operand) for o in (x, weight, bias)):
            return None
        if (len(weight.shape) != 2 or len(bias.shape) != 1
                or x.shape[-1:] != weight.shape[1:] or bias.shape != weight.shape[:1]):
            return None
        return x.slot, weight.slot, bias.slot

    # ``bias + contraction`` ops split like ``addmm`` into a contraction layer
    # (MACs) and a ``bias_add`` layer (other_ops).  Value: the type of the
    # contraction layer, whose handler bills it from the two matrices.
    _BIASED_CONTRACTION_LAYER_TYPE = {
        "addmm": "matmul",
        "addmv": "matmul",
        "baddbmm": "bmm",
        "addbmm": "addbmm",
    }

    @staticmethod
    def _addmm_positional_operands(
        input_shapes: List[Any], node_type: str = "addmm"
    ) -> Optional[Tuple[List[Any], List[Any], List[Any]]]:
        """Return (bias, mat1, mat2) shapes for a splittable addmm-family op, else None.

        Requires exactly three recorded tensor shapes: ``addmm`` takes 2D
        mat1/mat2 whose inner extents agree and a rank-1 or rank-2 bias;
        ``addmv`` a 2D matrix and a vector; ``baddbmm``/``addbmm`` two 3D
        batches with matching batch and inner extents.  The bias must
        broadcast (right-aligned) to the output.  Provenance (parameter vs
        runtime input) plays no role: the slots are positional.
        """
        if len(input_shapes) != 3 or not all(isinstance(s, list) for s in input_shapes):
            return None
        bias_shape, mat1_shape, mat2_shape = input_shapes
        if node_type == "addmm":
            if len(mat1_shape) != 2 or len(mat2_shape) != 2 or mat1_shape[1] != mat2_shape[0]:
                return None
            if len(bias_shape) not in (1, 2):
                return None
        elif node_type == "addmv":
            if len(mat1_shape) != 2 or len(mat2_shape) != 1 or mat1_shape[1] != mat2_shape[0]:
                return None
            if len(bias_shape) > 1:
                return None
            if bias_shape and bias_shape[0] not in (1, mat1_shape[0]):
                return None
        elif node_type in ("baddbmm", "addbmm"):
            if (len(mat1_shape) != 3 or len(mat2_shape) != 3 or mat1_shape[0] != mat2_shape[0]
                    or mat1_shape[2] != mat2_shape[1]):
                return None
            out = ([mat1_shape[0]] if node_type == "baddbmm" else []) + [mat1_shape[1], mat2_shape[2]]
            if len(bias_shape) > len(out):
                return None
            for extent, full in zip(bias_shape, out[len(out) - len(bias_shape):]):
                if extent not in (1, full):
                    return None
        else:
            return None
        return bias_shape, mat1_shape, mat2_shape

    def _should_split_linear_with_bias(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a linear/addmm layer with bias that should be split.

        ``torch.addmm(bias, x, W)`` (Hugging Face ``Conv1D``) is a linear layer
        with its bias passed first and the weight stored ``[in, out]``; it is
        split exactly like ``nn.Linear`` (matmul layer + ``bias_add`` layer) so
        both are billed by one convention.  A ``linear`` node binds its
        operands to the ``F.linear`` signature (:meth:`_linear_operands`).
        """
        node_type = str(node_data.get("type", "")).lower()
        if node_type in self._BIASED_CONTRACTION_LAYER_TYPE:
            # torch.addmm(bias, mat1, mat2) (and addmv/baddbmm/addbmm) bind
            # their operands by position, so the split needs all three recorded
            # as tensors with matrices of the right rank; anything else is left
            # to the op's handler, which refuses what it cannot bind instead of
            # guessing.
            return self._addmm_positional_operands(
                node_data.get("input_shapes") or [], node_type) is not None
        return self._linear_operands(node_data) is not None

    def _validate_input_types_alignment(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Ensure input_types aligns 1:1 with input_shapes for op nodes.

        When the torchview graph collapses multiple tensor inputs into
        fewer connection nodes (e.g. ``cat`` receiving two tensors via a
        single hidden-tensor node), ``input_types`` will be shorter than
        ``input_shapes``.  Pad with ``'input'`` to restore alignment,
        since the missing entries are always activation (non-weight)
        tensors.

        If ``input_types`` is *longer* than ``input_shapes``, that
        indicates a graph construction bug and we raise immediately.
        """
        input_shapes = node_data.get("input_shapes") or []
        input_types = node_data.get("input_types") or []
        if len(input_types) < len(input_shapes):
            input_types = list(input_types) + ["input"] * (len(input_shapes) - len(input_types))
            node_data["input_types"] = input_types
        elif len(input_types) > len(input_shapes):
            node_type = node_data.get("type", "unknown")
            raise ValueError(
                f"Node '{node_id}' (type={node_type}) has more input_types "
                f"({len(input_types)}) than input_shapes ({len(input_shapes)}). "
                "This indicates a graph construction bug."
            )

    def _split_linear_with_bias(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split a linear layer with bias into matmul + add operations.

        Returns:
            Tuple of (matmul_layer_dict, add_layer_dict)
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        # Resolve every operand the way any other op is resolved (start nodes,
        # producer ops and their output slots), then pick x, weight and bias
        # by position. Entries are (slot, tensor name, shape, type, producer);
        # producer is None for parameters.
        node_type = str(node_data.get("type", "")).lower()
        biased = node_type in self._BIASED_CONTRACTION_LAYER_TYPE
        if biased:
            # torch.addmm(bias, mat1, mat2): ordered semantic slots, bound by
            # position whatever the operands' provenance (nn.Parameter, buffer
            # or runtime input).  _should_split_linear_with_bias has already
            # established that all three are recorded; the node's own ordered
            # operand list must line up with the recorded shapes.
            node_connections = (node_data.get("connections") or {}).get("inputs") or []
            if len(node_connections) != 3 or len(input_shapes) != 3:
                raise ValueError(
                    f"{node_type} '{node_id}': cannot bind (bias, mat1, mat2) by position, "
                    f"{len(node_connections)} ordered operands for {len(input_shapes)} recorded shapes")
        entries = resolve_operands(
            self, node_id, node_data, op_graph, start_nodes_info, start_node_id_map
        )
        if biased:
            bias_entry, activation_entry, weight_entry = entries[0], entries[1], entries[2]
        else:
            activation_entry, weight_entry, bias_entry = (
                entries[i] for i in self._linear_operands(node_data)
            )
        output_connections = sorted(op_graph.successors(node_id))

        # Extract dtypes from the original node for propagation to sub-nodes.
        input_types = node_data.get("input_types") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (input_dtypes[i] for i, t in enumerate(input_types)
             if str(t).lower() == "weight" and i < len(input_dtypes)),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        weight_shape = list(weight_entry.shape)
        bias_shape = list(bias_entry.shape)

        # Intermediate shape (output of matmul, input to add)
        matmul_output_shape = output_shapes[0] if output_shapes else []

        # === MATMUL LAYER ===
        # Get einsum equation for matmul
        matmul_input_shapes_for_equation: List[List[Any]] = []
        matmul_input_shapes_for_equation.append(list(activation_entry.shape))
        matmul_input_shapes_for_equation.append(list(weight_entry.shape))
        matmul_ts = TensorShapes(
            inputs=matmul_input_shapes_for_equation,
            outputs=list(node_data.get("output_shapes") or []),
        )
        # nn.Linear stores the weight [out, in] (equation ...K,NK->...N);
        # torch.addmm stores it [in, out] (MK,KN->MN), so the matmul layer of an
        # addmm split is typed and billed as a matmul.
        matmul_op_name = self._BIASED_CONTRACTION_LAYER_TYPE.get(node_type, "linear")
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op(matmul_op_name, matmul_ts)
            matmul_equation = einsum_op.equation
            matmul_operands = {o.name: list(o.dims) for o in einsum_op.operands}
        except Exception:
            if node_type in ("addmv", "baddbmm", "addbmm"):
                # No synthesised equation for the new family: an operand set the
                # handler refuses must surface, not be billed as another shape.
                raise
            # Fallback equation
            if matmul_op_name == "matmul":
                matmul_equation = "MK,KN->MN"
                matmul_operands = {"Input": ["M", "K"], "Weight": ["K", "N"], "Output": ["M", "N"]}
            else:
                batch_dims = len(input_shapes[0]) - 1 if input_shapes else 0
                batch_letters = [f"B{i}" for i in range(batch_dims)]
                input_str = ''.join(batch_letters + ["K"])
                weight_str = "NK"
                output_str = ''.join(batch_letters + ["N"])
                matmul_equation = f"{input_str},{weight_str}->{output_str}"
                # Parse the fallback equation back into the operand structure.
                # Tokens are an uppercase letter optionally followed by digits (e.g. B0).
                _toks = lambda s: re.findall(r"[A-Z]\d*", s)
                matmul_operands = {
                    "Input": _toks(input_str),
                    "Weight": _toks(weight_str),
                    "Output": _toks(output_str),
                }

        add_node_id = f"{node_id}.bias_add"

        matmul_input_names: List[str] = []
        matmul_input_shapes_list: List[List[Any]] = []
        matmul_connection_inputs: List[str] = []

        for entry in (activation_entry, weight_entry):
            matmul_input_names.append(entry.name)
            matmul_input_shapes_list.append(list(entry.shape))
            if entry.producer is not None:
                matmul_connection_inputs.append(entry.producer)

        matmul_tensor_names = {
            "inputs": matmul_input_names,
            "outputs": [f"{node_id}.Output"],
        }
        matmul_tensor_types = {
            "inputs": [activation_entry.type, weight_entry.type],
            "outputs": ["output"],
        }
        matmul_tensor_shapes = {
            "inputs": matmul_input_shapes_list,
            "outputs": [list(matmul_output_shape)] if matmul_output_shape else [],
        }

        matmul_layer: Dict[str, Any] = {
            # Keep type as linear so MACs are computed by LinearHandler; an
            # addmm split is a matmul (weight [in, out]) and is billed as such.
            "type": matmul_op_name,
            "einsum_equation": matmul_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            # Operands drive the AF graph builder; without them the layer
            # is silently skipped from the AF einsums list.
            "operands": matmul_operands,
            "tensor_names": matmul_tensor_names,
            "tensor_types": matmul_tensor_types,
            "tensor_shapes": matmul_tensor_shapes,
            "parameter_sources": {"inputs": [None, weight_entry.parameter_source]},
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": matmul_connection_inputs,
                "outputs": [add_node_id],  # Output goes to bias_add
            },
        }

        if weight_shape:
            matmul_layer["additional_info"] = {
                "weights": [{"name": "Weight", "shape": list(weight_shape)}]
            }

        # === ADD (BIAS) LAYER ===
        # Generate einsum equation for bias add (broadcast add).  The bias is
        # aligned with the output from the right (NumPy broadcasting): an axis
        # of full extent shares the output's label, a singleton axis gets its
        # own label so the bias keeps its actual rank and storage extent in
        # the expression (``AB,B->AB`` for nn.Linear's [N]; ``AB,CB->AB`` for
        # an addmm bias [1, N]; ``AB,AB->AB`` for [M, N]).  A bias that does
        # not broadcast to the output is emitted as a copy, as before.
        add_equation = ""
        if matmul_output_shape:
            labels = string.ascii_uppercase[:len(matmul_output_shape)]
            bias_labels = self._broadcast_bias_labels(list(matmul_output_shape), bias_shape, labels)
            if bias_labels is not None:
                add_equation = f"{labels},{bias_labels}->{labels}"
                add_operands = {
                    "Input": list(labels),
                    "Weight": list(bias_labels),
                    "Output": list(labels),
                }
            else:
                add_equation = f"{labels}->{labels}"
                add_operands = {
                    "Input": list(labels),
                    "Output": list(labels),
                }
        else:
            add_operands = None

        add_input_names = [f"{node_id}.Output"]
        add_input_shapes_list = [list(matmul_output_shape)] if matmul_output_shape else []
        add_connection_inputs = [node_id]

        if bias_entry and bias_shape:
            add_input_names.append(bias_entry.name)
            add_input_shapes_list.append(list(bias_shape))
            if bias_entry.producer is not None:
                add_connection_inputs.append(bias_entry.producer)

        add_tensor_names = {
            "inputs": add_input_names,
            "outputs": [f"{add_node_id}.Output"],
        }
        add_tensor_types = {
            "inputs": ["input"] + ([bias_entry.type] if len(add_input_names) > 1 else []),
            "outputs": ["output"],
        }
        add_tensor_shapes = {
            "inputs": add_input_shapes_list,
            "outputs": [list(output_shapes[0])] if output_shapes else [],
        }

        add_layer: Dict[str, Any] = {
            "type": "add",
            "einsum_equation": add_equation,
            "elementwise_op": "add",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": add_operands,
            "tensor_names": add_tensor_names,
            "tensor_types": add_tensor_types,
            "tensor_shapes": add_tensor_shapes,
            "parameter_sources": {
                "inputs": [None] + ([bias_entry.parameter_source] if len(add_input_names) > 1 else []),
            },
            "tensor_dtypes": {
                "inputs": [out_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": add_connection_inputs,
                "outputs": output_connections,  # Original outputs
            },
        }

        # Add bias info
        if bias_shape:
            add_layer["additional_info"] = {
                "weights": [{"name": "Bias", "shape": list(bias_shape)}]
            }

        return matmul_layer, add_layer

    @staticmethod
    def _broadcast_bias_labels(output_shape: List[Any], bias_shape: Optional[List[Any]], out_labels: str) -> Optional[str]:
        """Labels for a bias broadcast against ``output_shape`` (right-aligned), or None.

        A bias axis equal to the aligned output axis reuses that axis's label;
        a singleton axis gets a fresh label (so ``M``/``N`` are never claimed
        by a size-1 axis and the bias keeps its own rank); any other extent
        does not broadcast and yields ``None``.
        """
        if not bias_shape or len(bias_shape) > len(output_shape):
            return None
        fresh = (c for c in string.ascii_uppercase if c not in out_labels)
        offset = len(output_shape) - len(bias_shape)
        labels = []
        for i, extent in enumerate(bias_shape):
            out_extent = output_shape[offset + i]
            if extent == out_extent:
                labels.append(out_labels[offset + i])
            elif extent == 1:
                labels.append(next(fresh))
            else:
                return None
        return "".join(labels)

    def _fix_split_connections(
        self,
        result: Dict[str, Any],
        node_id_remap: Dict[str, str],
        expanded_input_map: Optional[Dict[str, Dict[int, str]]] = None,
        expanded_output_slots: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """Fix connections for layers that reference split/expanded operations.

        When an operation is split/expanded:
        1. Downstream layers that consume the output should reference the final node
        2. Upstream layers (predecessors) should have their outputs updated to
           reference the correct subgraph entry node

        Args:
            result: The einsum graph dictionary being built.
            node_id_remap: Maps original node_id -> final output node_id.
            expanded_input_map: Maps original node_id -> {input_index -> subgraph_node_id}.
        """
        if expanded_input_map is None:
            expanded_input_map = {}
        if expanded_output_slots is None:
            expanded_output_slots = {}

        if not node_id_remap and not expanded_input_map:
            return

        # First pass: Update predecessor outputs for expanded operations
        for original_node_id, input_mapping in expanded_input_map.items():
            # Find all layers that have the original_node_id in their outputs
            for layer_id, layer_data in result["layers"].items():
                connections = layer_data.get("connections", {})
                outputs = connections.get("outputs", [])

                if original_node_id in outputs:
                    # This layer was a predecessor to the expanded node
                    # Find which input index this layer corresponds to
                    # by looking at the subgraph's inputs
                    new_outputs = []
                    for out in outputs:
                        if out == original_node_id:
                            # Determine which subgraph node this layer feeds into
                            # based on which input it provides
                            # We need to find the correct entry node
                            target_node = self._find_entry_node_for_predecessor(
                                result, layer_id, original_node_id, input_mapping
                            )
                            new_outputs.append(target_node)
                        else:
                            new_outputs.append(out)
                    connections["outputs"] = new_outputs

        # Second pass: Update downstream references
        for layer_id, layer_data in result["layers"].items():
            connections = layer_data.get("connections", {})
            inputs = connections.get("inputs", [])

            # Update input connections to reference final output node
            new_inputs = []
            for inp in inputs:
                # BUGFIX: Don't remap if the current layer is itself the target of the remapping
                # (e.g., don't replace Model.linear -> Model.linear.bias_add in Model.linear.bias_add's own inputs)
                # This prevents creating self-loops in split layers like bias_add
                if inp in node_id_remap and node_id_remap[inp] != layer_id:
                    new_inputs.append(node_id_remap[inp])
                else:
                    new_inputs.append(inp)
            connections["inputs"] = new_inputs

            # Update tensor_names inputs
            tensor_names = layer_data.get("tensor_names", {})
            if tensor_names:
                input_names = tensor_names.get("inputs", [])
                new_input_names = []
                slot_producers_used: List[str] = []   # expanded state outputs this layer reads (an earlier fix P2-2)
                finals_still_read: set = set()
                for name in input_names:
                    for old_id, new_id in node_id_remap.items():
                        # Keep split node self-inputs stable (e.g. bias_add should
                        # consume Model.linear.Output, not its own output).
                        if new_id == layer_id:
                            continue
                        if name == f"{old_id}.Output" or name.startswith(f"{old_id}.Output_"):
                            slots = expanded_output_slots.get(old_id) or {}
                            suffix = name[len(old_id) + 1:]           # "Output" or "Output_k"
                            k = int(suffix[len("Output_"):]) if suffix.startswith("Output_") and suffix[len("Output_"):].isdigit() else 0
                            if k in slots:
                                # a consumed h_n / c_n of an expanded recurrent node: its own producer
                                name = f"{slots[k]}.Output"
                                slot_producers_used.append(slots[k])
                            else:
                                name = name.replace(f"{old_id}.", f"{new_id}.", 1)
                                finals_still_read.add(new_id)
                            break
                    new_input_names.append(name)
                tensor_names["inputs"] = new_input_names
                if slot_producers_used:
                    # state-consumer-p2-3: rebuild the predecessor list IN OPERAND ORDER
                    # from the renamed tensor names — the producer of each
                    # consumed output is substituted into the slot where the
                    # original node sat for that output (sequence slot -> final
                    # sequence node, h_n slot -> h_n producer, c_n slot -> c_n
                    # producer), never appended: a consumer reading both
                    # outputs keeps one predecessor per distinct operand, so the
                    # AF builder pairs its roles with the right predecessors.
                    # state-consumer-p2-4: one predecessor entry PER predecessor-consuming
                    # slot, duplicates allowed (cat(h, h, c) reads h_n.stack
                    # twice) — the AF builder pairs non-weight roles with
                    # predecessors positionally, so deduplicating here made it
                    # pair the second h slot with c and invent a weight for
                    # the last slot.  Weight / unresolved slots carry no
                    # predecessor by contract.
                    layers_here = result["layers"]
                    slot_types = (layer_data.get("tensor_types") or {}).get("inputs") or []
                    ordered: List[str] = []
                    unaligned: List[Dict[str, Any]] = []
                    for i, name in enumerate(new_input_names):
                        stype = str(slot_types[i]).lower() if i < len(slot_types) else "input"
                        if stype in ("weight", "unresolved"):
                            continue
                        producer = name.rsplit(".Output", 1)[0] if ".Output" in name else None
                        if producer is None or producer == layer_id or producer not in layers_here:
                            unaligned.append({"slot": i, "name": name})
                            continue
                        ordered.append(producer)
                    connections["inputs"] = ordered
                    # Invariant: every connected non-weight slot has exactly one
                    # predecessor entry.  A slot whose producer is not a graph
                    # layer cannot be aligned positionally; refuse by marking
                    # the layer incomplete instead of letting the AF builder
                    # invent a weight for it (STATUS_CONTRACT rule 1).
                    if unaligned:
                        layer_data["status"] = "incomplete"
                        layer_data.setdefault("incomplete_reason", "predecessor_slots_unaligned")
                        layer_data["predecessor_slots_unaligned"] = unaligned
                        logger.warning("%s: %d input slot(s) have no graph producer after the expansion rewiring: %s",
                                       layer_id, len(unaligned), unaligned)

        # Third pass (an earlier fix P2-2): the final node and the state producers of an expanded
        # node were both handed every successor; keep only the layers that read them.
        if expanded_output_slots:
            readers: Dict[str, List[str]] = {}
            for layer_id, layer_data in result["layers"].items():
                for inp in (layer_data.get("connections", {}) or {}).get("inputs", []) or []:
                    readers.setdefault(inp, []).append(layer_id)
            for original_node_id, slots in expanded_output_slots.items():
                for lid in [node_id_remap.get(original_node_id)] + list(slots.values()):
                    layer = result["layers"].get(lid)
                    if layer is None:
                        continue
                    outs = (layer.get("connections", {}) or {}).get("outputs", []) or []
                    layer["connections"]["outputs"] = [o for o in outs if o in readers.get(lid, [])]

    def _find_entry_node_for_predecessor(
        self,
        result: Dict[str, Any],
        predecessor_id: str,
        original_node_id: str,
        input_mapping: Dict[int, str],
    ) -> str:
        """Find which subgraph entry node a predecessor should connect to.

        Args:
            result: The einsum graph dictionary.
            predecessor_id: ID of the predecessor layer.
            original_node_id: ID of the original (expanded) node.
            input_mapping: Maps input index -> subgraph node that receives it.

        Returns:
            The subgraph node ID that this predecessor should connect to.
        """
        # Look at the subgraph nodes to find which one has this predecessor in its inputs
        for subgraph_node_id in input_mapping.values():
            if subgraph_node_id in result["layers"]:
                subgraph_layer = result["layers"][subgraph_node_id]
                subgraph_inputs = subgraph_layer.get("connections", {}).get("inputs", [])
                if predecessor_id in subgraph_inputs:
                    return subgraph_node_id

        # Default: return the first entry node (qk_matmul for SDPA)
        if input_mapping:
            return input_mapping.get(0, list(input_mapping.values())[0])

        return original_node_id

    def _add_start_nodes(
        self,
        result: Dict[str, Any],
        start_nodes_info: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Add start nodes to the einsum graph."""
        start_node_id_map: Dict[str, str] = {}

        for info in start_nodes_info:
            idx = info["index"]
            start_id = "start" if idx == 0 else f"start_{idx}"
            original_id = info["original_id"]
            start_node_id_map[original_id] = start_id

            output_shapes = info.get("output_shapes") or []
            consumers = info.get("consumers", [])

            # Build tensor_names
            output_names = [f"{start_id}.Output"]
            for i in range(1, len(output_shapes)):
                output_names.append(f"{start_id}.Output_{i}")

            tensor_names = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": output_names,
            }

            # Build tensor_shapes
            tensor_shapes = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": [list(s) for s in output_shapes],
            }

            # Generate einsum equation
            equation = ""
            operands = {}
            if output_shapes and len(output_shapes[0]) > 0:
                dims = len(output_shapes[0])
                labels = [f"{c}{idx}" for c in string.ascii_uppercase[:dims]]
                equation = f"->{''.join(labels)}"
                operands = {start_id: labels}

            layer_dict: Dict[str, Any] = {
                "type": "start",
                "einsum_equation": equation,
                "elementwise_op": "copy",
                "reduction_op": "none",
                "is_real_einsum": False,
                "is_einsum_supportable": False,
                "tensor_names": tensor_names,
                "tensor_types": {
                    "inputs": [],
                    "outputs": ["input" for _ in output_names],
                },
                "tensor_shapes": tensor_shapes,
                "operands": operands,
                "connections": {
                    "inputs": [],
                    "outputs": consumers,
                },
            }

            # Propagate dtype info for start nodes
            output_dtypes = info.get("output_dtypes") or []
            if output_dtypes:
                layer_dict["tensor_dtypes"] = {
                    "inputs": [],
                    "outputs": output_dtypes,
                }

            result["layers"][start_id] = layer_dict

        return start_node_id_map


    def _convert_operation(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Convert a single operation to einsum representation."""
        node_type_raw = node_data.get("type", "unknown")
        node_type = self._einsum_analyzer._get_operation_from_name(str(node_type_raw))

        einsum_degraded = None
        if node_type == "einsum":
            if not (node_data.get("module_args") or {}).get("raw_attributes"):
                # torchview recorded no call attributes for this FunctionNode.
                # That happens with torchview releases before upstream #128
                # (0.2.7 on PyPI), where collect_attributes is not propagated
                # to the input/hidden recorder tensors; solar/patches/
                # torchview-collect-attributes-inputs.patch (applied by
                # validate-install) or a torchview >= #128 provides them.
                # Without them neither the equation nor the uncaptured
                # operands can be recovered: the layer degrades to the
                # generic elementwise fallback with MACs 0.  Say so
                # explicitly instead of failing silently.
                einsum_degraded = "no_call_attributes"
                if not getattr(self, "_warned_no_call_attributes", False):
                    self._warned_no_call_attributes = True
                    warnings.warn(
                        f"torchview recorded no call attributes for {node_id} (einsum); the "
                        "equation and uncaptured operands cannot be recovered and the layer is "
                        "costed as a generic elementwise op (MACs 0).  Requires torchview with "
                        "collect_attributes propagation (upstream #128) or solar/patches/"
                        "torchview-collect-attributes-inputs.patch.",
                        RuntimeWarning, stacklevel=2)
            # torchview only records operands it traced from the model inputs;
            # parameter-derived operands (e.g. rearranged nn.Parameters) are
            # absent from input_shapes although raw_attributes lists every
            # operand.  Restore them so shapes, traffic and MACs see all
            # operands of the equation (issue #32 / an earlier decision).
            node_data = self._complete_einsum_operands(node_id, node_data)

        ts = TensorShapes(
            inputs=list(node_data.get("input_shapes") or []),
            outputs=list(node_data.get("output_shapes") or []),
        )

        # Get module_args for operations like transpose/permute
        module_args = node_data.get("module_args", {})

        # Try to get einsum representation
        equation = ""
        operands = {}
        elementwise_op = "mul"
        reduction_op = "add"
        is_real_einsum = True
        is_einsum_supportable = True
        scaled_mm_rejection: Optional[str] = None   # a refused torch._scaled_mm (evidence-1)
        unrecognised_op: Optional[str] = None        # no handler, no recognised family (issue #50)
        interpolate_identity_copy: Optional[Dict[str, Any]] = None   # evidenced identity-size F.interpolate
        interpolate_refusal: Optional[str] = None       # why a present structured record was NOT admitted

        # Special handling for torch.einsum operations - parse the equation from raw_attributes
        if node_type == "einsum":
            parsed_equation = self._parse_einsum_from_raw_attributes(module_args)
            if parsed_equation:
                equation = parsed_equation
                elementwise_op = "mul"
                reduction_op = "add"
                is_real_einsum = True
                is_einsum_supportable = True
                lhs, rhs = parsed_equation.split("->")
                lhs_parts = lhs.split(",")
                operands = {"Input": list(lhs_parts[0]), "Output": list(rhs)}
                if len(lhs_parts) == 2:
                    operands["Weight"] = list(lhs_parts[1])
                else:
                    # Every operand of a multi-operand torch.einsum is a role of
                    # its own (issue #32 / an earlier decision): the AF builder maps roles to
                    # tensor_shapes.inputs positionally and must not have to
                    # synthesize the missing ones (INTEG-5: a four-operand SSD
                    # einsum lost one operand in the AF coverage check).
                    for i, part in enumerate(lhs_parts[1:], 1):
                        operands[f"Input_{i}"] = list(part)
            else:
                # Fallback: try to get from analyzer
                try:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args
                    )
                    equation = einsum_op.equation
                    elementwise_op = einsum_op.elementwise_op
                    reduction_op = einsum_op.reduction_op
                    is_real_einsum = einsum_op.is_real_einsum
                    is_einsum_supportable = einsum_op.is_einsum_supportable
                    operands = {operand.name: operand.dims for operand in einsum_op.operands}
                except Exception:
                    equation = ""
                    is_einsum_supportable = True
        else:
            # For reduction operations, parse dim and keepdim from raw_attributes
            # Based on PyTorch docs: https://docs.pytorch.org/docs/stable/nn.functional.html
            # These operations support dim and keepdim parameters:
            # - sum, mean, prod: standard reductions
            # - max, min, amax, amin: value reductions
            # - argmax, argmin: index reductions
            # - logsumexp, norm: special reductions
            # - std, var: statistical reductions
            # - all, any: boolean reductions
            reduce_dims = None
            keepdim = False
            reduction_ops_with_dim = {
                "sum", "mean", "prod",
                "max", "min", "amax", "amin",
                "argmax", "argmin",
                "logsumexp", "norm",
                "std", "var",
                "all", "any",
                "nansum", "nanmean",
            }
            if node_type in reduction_ops_with_dim:
                reduce_dims, keepdim = self._parse_reduction_args_from_raw_attributes(module_args)

            try:
                # Pass module_args, reduce_dims, and keepdim to the analyzer
                if reduce_dims is not None:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args, dims=reduce_dims, keepdim=keepdim
                    )
                elif "scaled_mm" in str(node_type).lower():
                    # the recorded operand dtypes decide the packed-vs-logical
                    # reduction extent (float4_e2m1fn_x2 packs two per byte); a
                    # handler refusal (missing / inconsistent dtypes or shapes) is
                    # recorded on the layer and the op is NOT lowered to the
                    # generic supportable copy (evidence-1)
                    try:
                        einsum_op = self._einsum_analyzer.get_einsum_op(
                            node_type, ts, module_args=module_args,
                            input_dtypes=list(node_data.get("input_dtypes") or []),
                        )
                    except Exception as exc:
                        scaled_mm_rejection = str(exc)
                        raise
                elif node_type == "interpolate":
                    # F.interpolate is mode/geometry dependent and has no
                    # handler.  Only an EVIDENCED identity-size resize
                    # (mode linear/nearest, explicit size == input extent, no
                    # scale factor, output shape == input shape, all read from
                    # the recorded call attributes) is lowered to the
                    # zero-arithmetic copy whose traffic is the recorded read
                    # and write; anything else stays unrecognised.
                    _in0 = ts.inputs[0] if ts.inputs else None
                    _out0 = ts.outputs[0] if ts.outputs else None
                    _ev = module_args.get("interpolate_evidence")
                    _ev_err = module_args.get("interpolate_evidence_error")
                    try:
                        if isinstance(_ev_err, dict):
                            # the capture recorded that the evidence builder failed:
                            # there is no evidence to weigh (capture-evidence-11)
                            raise TypeError(f"capture-time evidence builder failed: {_ev_err.get('type')}: {_ev_err.get('message')}")
                        interpolate_identity_copy = identity_interpolate_evidence(_ev, _in0, _out0)
                        if interpolate_identity_copy is None and _ev is not None:
                            # a record exists but did not ground the rule: say why
                            # (schema incomplete / malformed / untrusted callee / geometry)
                            interpolate_refusal = interpolate_record_refusal(_ev) or "interpolate_geometry_not_identity"
                    except Exception as _ev_exc:
                        # ANY failure while weighing interpolation evidence is
                        # recorded and lands on the unrecognised/incomplete path,
                        # never on the permissive supportability fallback
                        # (schema-9)
                        interpolate_identity_copy = None
                        interpolate_refusal = f"interpolate_evidence_error: {type(_ev_exc).__name__}: {_ev_exc}"
                        raise UnrecognisedOperationError(str(node_type), "interpolate") from _ev_exc
                    if interpolate_identity_copy is None:
                        einsum_op = self._einsum_analyzer.get_einsum_op(
                            node_type, ts, module_args=module_args, keepdim=keepdim
                        )
                    else:
                        einsum_op = self._einsum_analyzer.get_einsum_op(
                            IDENTITY_COPY_OP, ts, module_args=module_args
                        )
                else:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args, keepdim=keepdim
                    )
                equation = einsum_op.equation
                elementwise_op = einsum_op.elementwise_op
                reduction_op = einsum_op.reduction_op
                is_real_einsum = einsum_op.is_real_einsum
                is_einsum_supportable = einsum_op.is_einsum_supportable
                operands = {operand.name: operand.dims for operand in einsum_op.operands}
            except Exception as exc:
                equation = ""
                is_einsum_supportable = self._is_operation_supportable(node_type)
                if node_type == "interpolate" and not isinstance(exc, UnrecognisedOperationError):
                    # belt and braces: an interpolate layer never reaches the
                    # permissive fallback on an exception (schema-9)
                    if interpolate_refusal is None:
                        interpolate_refusal = f"interpolate_evidence_error: {type(exc).__name__}: {exc}"
                    exc = UnrecognisedOperationError(str(node_type), "interpolate")
                if isinstance(exc, UnrecognisedOperationError):
                    # Issue #50: no handler and no recognised family.  The op is
                    # NOT supportable and is recorded by its captured name; the
                    # analyzer reports it incomplete (op_unrecognised) with 0
                    # MACs.  A rank-preserving placeholder equation keeps the
                    # output tensor defined for the AF graph and bills nothing.
                    unrecognised_op = str(node_type)
                    is_einsum_supportable = False
                    is_real_einsum = False
                    elementwise_op = "copy"
                    reduction_op = "none"
                    try:
                        _placeholder = self._einsum_analyzer.get_einsum_op("unresolved_op", ts, module_args=module_args)
                        equation = _placeholder.equation
                        operands = {operand.name: operand.dims for operand in _placeholder.operands}
                    except Exception:
                        equation = ""
                if scaled_mm_rejection is not None:
                    # a refused scaled_mm is NOT supportable and carries the
                    # refusal; the analyzer reports it incomplete with 0 MACs
                    # (scaled_mm_evidence_unresolved), never the stored-shape
                    # product.  A rank-preserving placeholder equation keeps the
                    # layer's output tensor produced for the AF graph (its
                    # consumers would otherwise read an orphan); it is not a
                    # contraction and bills nothing.
                    try:
                        _placeholder = self._einsum_analyzer.get_einsum_op("unresolved_op", ts, module_args=module_args)
                        equation = _placeholder.equation
                        operands = {operand.name: operand.dims for operand in _placeholder.operands}
                    except Exception:
                        equation = ""
                    is_einsum_supportable = False
                    is_real_einsum = False
                    elementwise_op = "copy"
                    reduction_op = "none"

                # Set default ops based on node type
                if node_type in {"add", "sub", "mul", "div"}:
                    elementwise_op = node_type
                    reduction_op = "none"
                    is_real_einsum = False
                elif node_type in {"sum", "mean"}:
                    elementwise_op = "copy"
                    reduction_op = "add"
                    is_real_einsum = False
                elif node_type == "prod":
                    elementwise_op = "copy"
                    reduction_op = "mul"
                    is_real_einsum = False
                elif node_type in {"max", "min"}:
                    elementwise_op = "copy"
                    reduction_op = node_type
                    is_real_einsum = False

        # Build input connections preserving the original PyTorch argument order.
        # Then map tensor-node IDs to start-node IDs where applicable.
        raw_input_connections = list((node_data.get("connections") or {}).get("inputs") or [])
        if not raw_input_connections:
            raw_input_connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in raw_input_connections:
                    raw_input_connections.append(original_id)

        # Resolve each input tensor to its producer op. Priority:
        #   1. weight slots → keep the mapped value as-is
        #   2. tensors that map directly to a start node or op node
        #   3. hidden tensors with a known producer in _tensor_to_producer_op
        #      (populated by _build_op_graph, including torchview-orphan
        #      reconciliation for shape-donor patterns like zeros_like)
        # Any remaining unresolved inputs are filled from op_graph.predecessors
        # only if the count matches exactly — i.e. there is a unique 1-to-1
        # assignment. Mismatched counts raise, replacing the legacy positional
        # fallback that silently mis-attributed the cat inputs in kernbench
        # L1/92 cumsum_exclusive.
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        op_predecessors = list(op_graph.predecessors(node_id))
        input_connections: List[Optional[str]] = []
        assigned_preds: set = set()
        deferred_indices: List[int] = []
        input_types_raw = list(node_data.get("input_types") or [])
        for i, conn_id in enumerate(raw_input_connections):
            mapped = start_node_id_map.get(conn_id, conn_id)
            itype = str(input_types_raw[i]).lower() if i < len(input_types_raw) else "input"
            if itype in ("weight", "unresolved"):   # unresolved einsum operands (issue #46) have no producer to resolve
                input_connections.append(mapped)
                continue
            if mapped in self._start_targets(start_node_id_map) or mapped in op_graph.nodes:
                input_connections.append(mapped)
                assigned_preds.add(mapped)
                continue
            producer = tensor_to_producer.get(conn_id)
            if producer is not None and producer in op_graph.nodes:
                input_connections.append(producer)
                assigned_preds.add(producer)
                continue
            input_connections.append(None)
            deferred_indices.append(i)
        if deferred_indices:
            unmatched_preds = [p for p in op_predecessors if p not in assigned_preds]
            if len(unmatched_preds) == len(deferred_indices):
                # Unambiguous positional match.
                for idx, pred in zip(deferred_indices, unmatched_preds):
                    input_connections[idx] = pred
            elif len(unmatched_preds) == 0:
                # No op-graph candidates — the input is genuinely producerless
                # (e.g. BERT's position-ids tensor created by an untraced
                # `torch.arange`). Fall through to the literal tensor id, which
                # downstream code handles as a phantom name. NOT the L1/92
                # pattern, which has unmatched_preds != 0.
                for idx in deferred_indices:
                    input_connections[idx] = start_node_id_map.get(
                        raw_input_connections[idx], raw_input_connections[idx]
                    )
            else:
                # Truly ambiguous: there are candidate predecessors but the
                # count does not match. This was the kernbench L1/92
                # cumsum_exclusive failure: 2 orphan inputs mapped to 1
                # predecessor. Refuse to guess.
                unresolved = [raw_input_connections[i] for i in deferred_indices]
                raise ValueError(
                    f"_convert_operation({node_id!r}): cannot uniquely "
                    f"resolve inputs {unresolved}. deferred="
                    f"{len(deferred_indices)}, unmatched_preds="
                    f"{len(unmatched_preds)}. Producer attribution is "
                    f"ambiguous — usually a torchview tracing gap that "
                    f"_build_op_graph's reconciliation pass couldn't "
                    f"shape-match uniquely."
                )

        # Which output slot of its producer each input tensor is (0 for the
        # primary output).  Consumers of chunk/split slot k must reference
        # ``<producer>.Output_<k>``, not the primary output.  A recorded slot
        # is only trusted when the input actually resolved to that producer.
        tensor_slots = getattr(self, "_tensor_to_producer_slot", {})
        input_slots: List[int] = []
        for i, conn_id in enumerate(raw_input_connections):
            if i >= len(input_connections):
                break
            slot = tensor_slots.get(conn_id, 0)
            if slot and input_connections[i] != tensor_to_producer.get(conn_id):
                slot = 0
            input_slots.append(slot)

        # Take tensor input/output types directly from PyTorch graph by index.
        pytorch_input_types = list(node_data.get("input_types") or [])
        if len(pytorch_input_types) < len(input_connections):
            pytorch_input_types.extend(["input"] * (len(input_connections) - len(pytorch_input_types)))

        # Inject input_types into node_data so downstream functions can use it.
        node_data_with_types = dict(node_data)
        node_data_with_types["input_types"] = pytorch_input_types

        output_connections = sorted(list(op_graph.successors(node_id)))

        # Build tensor_names using input_types
        tensor_names = self._build_tensor_names(
            node_id, node_data_with_types, input_connections, output_connections,
            input_slots=input_slots,
        )
        pytorch_output_types = list(node_data.get("output_types") or [])
        if len(pytorch_output_types) < len(tensor_names.get("outputs", [])):
            pytorch_output_types.extend(
                ["output"] * (len(tensor_names.get("outputs", [])) - len(pytorch_output_types))
            )
        tensor_types = {
            "inputs": list(pytorch_input_types[: len(tensor_names.get("inputs", []))]),
            "outputs": list(pytorch_output_types[: len(tensor_names.get("outputs", []))]),
        }

        # Build tensor_shapes: shapes matching tensor_names order
        tensor_shapes = self._build_tensor_shapes(node_data)

        # Validate tensor_names and tensor_shapes match
        is_valid, error_msg = validate_tensor_names_match_shapes(tensor_names, tensor_shapes)
        if not is_valid:
            # Fix mismatch by aligning counts
            tensor_names, tensor_shapes = self._align_tensor_names_and_shapes(
                tensor_names, tensor_shapes, node_data
            )
            tensor_types["inputs"] = tensor_types["inputs"][: len(tensor_names.get("inputs", []))]
            tensor_types["outputs"] = tensor_types["outputs"][: len(tensor_names.get("outputs", []))]

        # Parameter provenance per input slot: the capture ``parameter-tensor``
        # node that supplies a ``weight`` slot, else None.  The slot's TYPE
        # says the capture saw a parameter; the SOURCE says which one, so the
        # same nn.Parameter in two slots of one call is one footprint, and a
        # ``weight`` label with no parameter node behind it (an operand an
        # older converter restored from raw_attributes) is provenance unknown
        # rather than a parameter (parameter-identity-1 / PROVENANCE-1).
        parameter_node_ids = getattr(self, "_parameter_node_ids", None) or set()
        parameter_sources: List[Optional[str]] = []
        for i in range(len(tensor_names.get("inputs", []))):
            slot_type = str(pytorch_input_types[i]).lower() if i < len(pytorch_input_types) else "input"
            conn = input_connections[i] if i < len(input_connections) else None
            parameter_sources.append(
                str(conn) if slot_type == "weight" and conn is not None and str(conn) in parameter_node_ids else None
            )

        # Build additional_info for weight/bias metadata
        additional_info = self._build_additional_info(node_data)

        raw_attributes = module_args.get("raw_attributes")
        access = self._build_access_metadata(
            node_type,
            raw_attributes,
            tensor_names,
            tensor_shapes,
        )

        # Filter out weight connections from connections.inputs
        # (parameter nodes don't exist as layers in the einsum graph)
        activation_connections = [
            c for i, c in enumerate(input_connections)
            if not (
                i < len(pytorch_input_types)
                and str(pytorch_input_types[i]).lower() in ("weight", "unresolved")
            )
        ]
        unresolved_operands = list((module_args or {}).get("einsum_operands_unresolved") or [])

        # Propagate dtype info from pytorch graph so downstream stages
        # (graph_analyzer) can detect non-standard dtypes like torch.bool.
        input_dtypes = list(node_data.get("input_dtypes") or [])
        output_dtypes = list(node_data.get("output_dtypes") or [])
        tensor_dtypes: Dict[str, Any] = {}
        if input_dtypes:
            tensor_dtypes["inputs"] = input_dtypes
        if output_dtypes:
            tensor_dtypes["outputs"] = output_dtypes

        result: Dict[str, Any] = {
            "type": node_type,
            "einsum_equation": equation,
            "elementwise_op": elementwise_op,
            "reduction_op": reduction_op,
            "is_real_einsum": is_real_einsum,
            "is_einsum_supportable": is_einsum_supportable,
            # issue #46: einsum operands whose provenance could not be
            # established are typed 'unresolved'; the layer is incomplete.
            **({"status": "incomplete", "einsum_operands_unresolved": unresolved_operands}
               if unresolved_operands else {}),
            "tensor_names": tensor_names,
            "tensor_types": tensor_types,
            "tensor_shapes": tensor_shapes,
            "operands": operands,
            "connections": {
                "inputs": activation_connections,
                "outputs": output_connections,
            },
        }

        if tensor_dtypes:
            result["tensor_dtypes"] = tensor_dtypes
        result["parameter_sources"] = {"inputs": parameter_sources}

        if additional_info:
            result["additional_info"] = additional_info

        if access:
            result["access"] = access

        # Pass through raw_attributes from module_args if present
        if raw_attributes:
            result["raw_attributes"] = raw_attributes
        # Identity binding of the call's tensor arguments (an earlier decision round 5)
        call_operands = module_args.get("call_operands")
        if isinstance(call_operands, dict):
            result["call_operands"] = dict(call_operands)
        if scaled_mm_rejection is not None:
            result["scaled_mm_rejection"] = scaled_mm_rejection
        if unrecognised_op is not None:
            result["unrecognised_op"] = unrecognised_op
        if interpolate_identity_copy is not None:
            result["interpolate_identity_copy"] = dict(interpolate_identity_copy)
        if interpolate_refusal is not None:
            result["interpolate_refusal"] = interpolate_refusal
        if einsum_degraded:
            result["einsum_degraded"] = einsum_degraded

        # Normalisation layers whose parameters could not be attached at capture
        # (issue #34): carry the machine-readable marker and mark the operation
        # not supportable so no downstream consumer treats the activation-only
        # traffic as an ordinary result.
        unattached = module_args.get("norm_params_unattached")
        if unattached:
            result["norm_params_unattached"] = unattached
            result["is_einsum_supportable"] = False

        return result

    def _build_access_metadata(
        self,
        node_type: str,
        raw_attributes: Any,
        tensor_names: Dict[str, List[str]],
        tensor_shapes: Dict[str, List[List[int]]],
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Emit normalized access-region metadata for view-like ops.

        The analyzer can consume this structured form directly while older
        graphs continue to fall back to raw_attributes parsing. Imports stay
        local to avoid making einsum conversion depend on analysis package
        initialization at module import time.
        """
        from solar.analysis.access_regions import (
            PARTITION_OPS,
            SLICE_VIEW_OPS,
            boxes_to_list,
            partition_output_box,
            slice_op_boxes,
        )

        def _valid_shape(shape: Any) -> bool:
            return isinstance(shape, list) and all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
                for dim in shape
            )

        def _shape_product(shape: List[int]) -> int:
            result = 1
            for dim in shape:
                result *= int(dim)
            return int(result)

        def _entry(
            index: int,
            tensor_name: str,
            base_tensor: str,
            base_shape: List[int],
            boxes: List[Any],
            source: str,
        ) -> Dict[str, Any]:
            return {
                "index": int(index),
                "tensor": str(tensor_name),
                "kind": "region",
                "base_tensor": str(base_tensor),
                "base_shape": [int(dim) for dim in base_shape],
                "boxes": boxes_to_list(boxes),
                "source": source,
            }

        op_type = str(node_type).lower()
        input_names = list((tensor_names or {}).get("inputs") or [])
        output_names = list((tensor_names or {}).get("outputs") or [])
        input_shapes = list((tensor_shapes or {}).get("inputs") or [])
        output_shapes = list((tensor_shapes or {}).get("outputs") or [])
        access: Dict[str, List[Dict[str, Any]]] = {"inputs": [], "outputs": []}

        if (
            op_type in SLICE_VIEW_OPS
            and input_names
            and output_names
            and input_shapes
            and output_shapes
            and _valid_shape(input_shapes[0])
            and isinstance(output_shapes[0], list)
        ):
            expected_elems = _shape_product(output_shapes[0])
            boxes = slice_op_boxes(
                op_type,
                raw_attributes,
                input_shapes[0],
                expected_elems=expected_elems,
            )
            if boxes is not None:
                access["inputs"].append(
                    _entry(
                        0,
                        input_names[0],
                        input_names[0],
                        input_shapes[0],
                        boxes,
                        "raw_attributes",
                    )
                )
                access["outputs"].append(
                    _entry(
                        0,
                        output_names[0],
                        input_names[0],
                        input_shapes[0],
                        boxes,
                        "raw_attributes",
                    )
                )

        if (
            op_type in PARTITION_OPS
            and input_names
            and input_shapes
            and _valid_shape(input_shapes[0])
            and all(_valid_shape(shape) for shape in output_shapes)
        ):
            for output_index, output_name in enumerate(output_names):
                box = partition_output_box(
                    op_type,
                    raw_attributes,
                    input_shapes[0],
                    output_shapes,
                    output_index,
                )
                if box is None:
                    continue
                access["outputs"].append(
                    _entry(
                        output_index,
                        output_name,
                        input_names[0],
                        input_shapes[0],
                        [box],
                        "partition_shapes",
                    )
                )

        return {key: value for key, value in access.items() if value} or None

    def _complete_einsum_operands(
        self,
        node_id: str,
        node_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return node_data with every einsum operand present in input_shapes.

        ``raw_attributes`` of a ``torch.einsum`` node lists the equation and
        all operand tensors in positional order.  When torchview captured
        fewer inputs than operands, the captured shapes are aligned to the
        operand slots in order (by shape equality) and the uncaptured slots
        are filled from raw_attributes.  Uncaptured operands have no
        producer in the graph, so their provenance is unknown: they are typed
        ``unresolved`` (external reads named ``<node>.Unresolved[_k]``, never
        defaulted to ``weight``; issue #46 / INTEG-2), listed in
        ``module_args['einsum_operands_unresolved']`` and the converted
        layer carries ``status: incomplete``.  Provenance would make them
        ``weight`` (parameter/buffer node) or ``input`` (produced by an op or a
        graph input).  If the captured shapes cannot be aligned unambiguously
        the node is returned unchanged.
        """
        from solar.einsum.ops.einsum_ops import parse_einsum_operands_from_raw_attributes

        module_args = node_data.get("module_args") or {}
        operands = parse_einsum_operands_from_raw_attributes(module_args.get("raw_attributes"))
        input_shapes = [list(s) for s in (node_data.get("input_shapes") or [])]
        if not operands or len(operands) <= len(input_shapes):
            return node_data
        conns = list((node_data.get("connections") or {}).get("inputs") or [])
        types = list(node_data.get("input_types") or [])
        dtypes = list(node_data.get("input_dtypes") or [])
        if conns and len(conns) != len(input_shapes):
            logger.warning(
                "einsum %s: %d captured inputs but %d connections; operands not completed",
                node_id, len(input_shapes), len(conns))
            return node_data

        slot_of_captured: Dict[int, int] = {}
        j = 0
        for i, (shape, _dtype) in enumerate(operands):
            if j < len(input_shapes) and list(shape) == input_shapes[j]:
                slot_of_captured[j] = i
                j += 1
        if j != len(input_shapes):
            logger.warning(
                "einsum %s: captured input shapes %s do not align with raw operands %s; not completed",
                node_id, input_shapes, [s for s, _ in operands])
            return node_data
        captured_at = {slot: j for j, slot in slot_of_captured.items()}

        new_shapes: List[List[int]] = []
        new_types: List[str] = []
        new_dtypes: List[str] = []
        new_conns: List[str] = []
        unresolved_slots: List[int] = []
        for i, (shape, dtype) in enumerate(operands):
            if i in captured_at:
                j = captured_at[i]
                new_shapes.append(input_shapes[j])
                new_types.append(types[j] if j < len(types) else "input")
                new_dtypes.append(dtypes[j] if j < len(dtypes) else (dtype or ""))
                new_conns.append(conns[j] if j < len(conns) else f"{node_id}.operand_{i}")
            else:
                # No producer in the graph: the operand's provenance is unknown
                # (issue #46 / INTEG-2).  It is NOT a weight by default: it is
                # typed ``unresolved`` (an external read with no provenance),
                # recorded in ``einsum_operands_unresolved`` and the layer is
                # marked incomplete downstream.
                new_shapes.append(list(shape))
                new_types.append("unresolved")
                new_dtypes.append(dtype or (dtypes[0] if dtypes else ""))
                new_conns.append(f"{node_id}.operand_{i}")
                unresolved_slots.append(i)

        completed = dict(node_data)
        completed["input_shapes"] = new_shapes
        completed["input_types"] = new_types
        if dtypes or any(d for _, d in operands):
            completed["input_dtypes"] = new_dtypes
        completed["connections"] = dict(node_data.get("connections") or {})
        completed["connections"]["inputs"] = new_conns
        completed["module_args"] = dict(module_args)
        completed["module_args"]["einsum_operands_unresolved"] = list(unresolved_slots)
        if getattr(self, "debug", False):
            logger.debug(f"einsum {node_id}: completed operands "
                  f"{[i for i in range(len(operands)) if i not in captured_at]} from raw_attributes")
        return completed

    def _build_tensor_names(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        input_connections: List[str],
        output_connections: List[str],
        input_slots: Optional[List[int]] = None,
    ) -> Dict[str, List[str]]:
        """Build tensor names matching input_shapes/output_shapes order.

        Uses input_types to name weight inputs as <node_id>.Weight
        and activation inputs as <predecessor_id>.Output.  When
        ``input_slots`` marks an input as output slot k > 0 of its
        producer (multi-output ops like chunk/split), the name is
        <predecessor_id>.Output_<k>, matching the producer's own
        output naming.
        """
        input_names: List[str] = []
        output_names: List[str] = []
        input_types = node_data.get("input_types") or []

        weight_idx = 0
        unresolved_idx = 0
        for i, pred_id in enumerate(input_connections):
            itype = input_types[i] if i < len(input_types) else 'input'
            if itype == 'weight':
                name = f"{node_id}.Weight" if weight_idx == 0 else f"{node_id}.Weight_{weight_idx}"
                input_names.append(name)
                weight_idx += 1
            elif itype == 'unresolved':
                # an einsum operand with no producer (issue #46): not a graph layer
                name = f"{node_id}.Unresolved" if unresolved_idx == 0 else f"{node_id}.Unresolved_{unresolved_idx}"
                input_names.append(name)
                unresolved_idx += 1
            else:
                slot = input_slots[i] if input_slots and i < len(input_slots) else 0
                suffix = ".Output" if slot == 0 else f".Output_{slot}"
                input_names.append(f"{pred_id}{suffix}")

        # Output tensors
        output_names.append(f"{node_id}.Output")
        output_shapes = node_data.get("output_shapes") or []
        for i in range(1, len(output_shapes)):
            output_names.append(f"{node_id}.Output_{i}")

        return {
            "inputs": input_names,
            "outputs": output_names,
        }

    def _build_tensor_shapes(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, List[List[int]]]:
        """Build tensor shapes matching input_shapes/output_shapes order.

        All inputs (activation + weight) are already in input_shapes in arg order.
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        return {
            "inputs": [list(s) for s in input_shapes],
            "outputs": [list(s) for s in output_shapes],
        }

    def _align_tensor_names_and_shapes(
        self,
        tensor_names: Dict[str, List[str]],
        tensor_shapes: Dict[str, List[List[int]]],
        node_data: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[List[int]]]]:
        """Align tensor_names and tensor_shapes to have matching counts.

        When there's a mismatch (e.g., weight_nodes vs weight_shapes have different lengths),
        this method aligns them by using the shapes as the source of truth and generating
        placeholder names if needed, or trimming excess names.
        """
        input_names = tensor_names.get("inputs", [])
        output_names = tensor_names.get("outputs", [])
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])

        # Align inputs
        if len(input_names) != len(input_shapes):
            # Use shapes as source of truth
            if len(input_shapes) > len(input_names):
                # Add placeholder names for missing entries
                node_id = node_data.get("id", "unknown")
                for i in range(len(input_names), len(input_shapes)):
                    input_names.append(f"{node_id}.Input_{i}")
            else:
                # Trim excess names
                input_names = input_names[:len(input_shapes)]

        # Align outputs
        if len(output_names) != len(output_shapes):
            if len(output_shapes) > len(output_names):
                node_id = node_data.get("id", "unknown")
                for i in range(len(output_names), len(output_shapes)):
                    output_names.append(f"{node_id}.Output_{i}")
            else:
                output_names = output_names[:len(output_shapes)]

        return (
            {"inputs": input_names, "outputs": output_names},
            {"inputs": input_shapes, "outputs": output_shapes},
        )

    def _build_additional_info(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build additional_info metadata (weight info is in tensor_names/tensor_shapes).

        For transposed convolutions the call arguments (stride, padding,
        output_padding, dilation, groups) are exported as ``conv_args`` so
        the analyzer can report the minimum-work bound next to the dense
        ``macs`` (issue #44).  Export-only: equation, shapes, traffic and the
        dense count do not depend on it.  When torchview recorded no call
        attributes (releases before upstream #128 without
        solar/patches/torchview-collect-attributes-inputs.patch) the layer is
        marked ``conv_args_unavailable`` and the analyzer reports
        ``min_work: not_representable``; never a silent number.
        """
        info: Dict[str, Any] = {}
        node_type = self._einsum_analyzer._get_operation_from_name(str(node_data.get("type", "")))
        if node_type in _CONV_TRANSPOSE_TYPES:
            dims = int(node_type[-2])
            raw = (node_data.get("module_args") or {}).get("raw_attributes")
            if not raw:
                info["conv_args_unavailable"] = "no_call_attributes"
                if not getattr(self, "_warned_no_conv_attributes", False):
                    self._warned_no_conv_attributes = True
                    warnings.warn(
                        f"torchview recorded no call attributes for {node_data.get('id', node_type)} "
                        f"({node_type}); conv_args cannot be exported and the minimum-work bound is "
                        "not representable.  Requires torchview with collect_attributes propagation "
                        "(upstream #128) or solar/patches/torchview-collect-attributes-inputs.patch.",
                        RuntimeWarning, stacklevel=2)
            else:
                args = parse_conv_transpose_args(raw, dims)
                if args is None:
                    info["conv_args_unavailable"] = "unparsed_call_attributes"
                else:
                    info["conv_args"] = args
        return info

    def _is_operation_supportable(self, op_type: str) -> bool:
        """Check if an operation can be expressed with extended einsum."""
        op = op_type.lower()

        # Check against known supportable operations
        if op in _ALL_SUPPORTABLE_OPS:
            return True

        # Check for suffixed matches
        for supported_op in _ALL_SUPPORTABLE_OPS:
            if op.endswith(f".{supported_op}"):
                return True

        # Check prefixed patterns
        if any(op.startswith(prefix) for prefix in ["torch.", "nn.", "functional."]):
            stripped = op.split(".")[-1]
            return stripped in _ALL_SUPPORTABLE_OPS

        # Default: supportable unless explicitly unsupportable
        return op not in _UNSUPPORTABLE_OPS







# Backward compatibility alias
PyTorchEinsumConverter = PyTorchToEinsum


__all__ = [
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
]
