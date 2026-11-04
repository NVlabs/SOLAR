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

"""Analyze an einsum graph into hardware-independent metrics.

This module implements the **second stage** of the Solar pipeline:

  `einsum_graph.yaml`  ->  `analysis.yaml`

The output `analysis.yaml` is intended to be hardware-independent and includes:
- per-layer: macs, flops (= 2 * macs), unfused_elements, fused_elements
- totals across the graph

Memory Access Models (in elements, multiply by bytes_per_element for bytes):
- unfused_elements: All tensor accesses (inputs + outputs) per op, summed
- orojenesis_elements: Set to None (orojenesis runs not enabled)
- fused_elements: Deduplicated external I/O (weights + model inputs/outputs),
    same tensor read by multiple ops counted once. Equal to fused_prefetched.
- fused_prefetched_elements: Same as fused_elements (deduplicated external I/O)

Note: input_elements includes all inputs to an operation (including weights/biases).
Weights are treated as inputs since they are just another operand to the computation.

External (non-intermediate) traffic is reported in labelled parts, per layer
and in the totals (issue #36 / an earlier decision):
- weight_elements: the PARAMETER footprint — numel of each DISTINCT
    parameter a layer consumes, once per consuming op whatever the number
    of slots it occupies.  Provenance is the capture's parameter identity:
    the converter records in ``parameter_sources.inputs[i]`` the
    ``parameter-tensor`` node that supplies a ``weight`` slot.  A ``weight``
    slot without a recorded source is provenance UNKNOWN: its numel is
    reported in ``weight_elements_unresolved`` / ``weight_provenance_unresolved``
    (layer ``status: incomplete``, reason ``weight_provenance_unresolved``),
    its read is model I/O.  Graphs saved without ``parameter_sources``
    carry only the label; it is trusted except for explicit ``einsum``
    operands (the pre-#46 converter typed untraced activations ``weight``).
    An input beyond the typed list has no provenance and is never a weight.
- weight_read_elements: the DRAM reads of the counted parameters under each
    op's read model.  The two quantities differ whenever a parameter is not
    read exactly once in full: embedding gathers read rows (fewer or more
    elements than the table, an earlier decision), a parameter in two slots is read twice.
- model_io_elements: the remaining external reads (model inputs and reads of
    unknown provenance) plus model outputs.
For every layer
``weight_read_elements + model_io_elements + intermediate_elements == unfused_elements``
and ``fused_elements == weight_read_elements + model_io_elements``.
``total.weight_elements``, ``total.weight_read_elements`` and
``total.model_io_elements`` are plain layer sums; ``total.unfused_elements``
and ``total.fused_elements`` also carry graph-level model I/O that no layer
owns (declared outputs, gather indices) and the fused total is the
deduplicated external footprint, so the identity is a per-layer statement.
The split is a relabel of the same element counts: no traffic total changes.

Note: "start" nodes are filtered out before analysis as they represent model inputs,
not actual computation. Their outputs are treated as external inputs to the graph.

See SOL_GUIDE.md for detailed explanation of the three SOL models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import math
import numbers
import re

import yaml

from solar.einsum import EinsumAnalyzer
from solar.einsum.ops.base import compute_cost_from_equation, optimal_dense_binary_schedule, validate_equation_operands
from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.types import TensorShapes
from solar.common.utils import ensure_directory, NoAliasDumper, parse_dim_tokens, yaml_safe_load
from solar.analysis.access_regions import (
    PARTITION_OPS,
    SLICE_VIEW_OPS,
    Box,
    access_entry_to_boxes,
    box_size,
    partition_output_box,
    slice_op_boxes,
    union_size,
)
import logging

logger = logging.getLogger(__name__)

# Storage widths for boundary-tensor billing, accepting both numpy/torch
# dtype spellings (benchmark definitions) and SOLAR precision names (DSL
# tags).  Boundary STORAGE width is a property of the declared tensor, not
# of the per-graph compute precision.
_STORAGE_WIDTHS: Dict[str, float] = {
    "float64": 8, "fp64": 8, "double": 8, "int64": 8,
    "float32": 4, "fp32": 4, "tf32": 4, "int32": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float8": 1, "fp8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "uint8": 1, "bool": 1,
    "nvfp4": 0.5, "fp4": 0.5,
}


PathLike = Union[str, Path]


def _product(shape: List[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


def _einsum_operand_shapes_from_raw(layer: Dict[str, Any], equation: str, ts: TensorShapes) -> Optional[List[List[int]]]:
    """Positional operand shapes of an explicit torch.einsum from its recorded call.

    ``raw_attributes`` lists the equation and every operand tensor in call
    order; the saved ``tensor_shapes`` may carry fewer (the Mamba2 SSD einsum
    kept only the last two of four).  The saved shapes are aligned to the raw
    operands in order by shape equality (the same rule the earlier fix's converter-side
    completion uses); if every saved shape aligns and the raw count equals the
    equation's operand count, the raw shapes are returned in operand order,
    else None.  Evidence restoration for the cost only: the saved graph (and
    its traffic) is not changed here.
    """
    raw = _layer_raw_attributes(layer)
    if not raw or "->" not in (equation or ""):
        return None
    raw_shapes = [[int(x) for x in m.group(1).split(",") if x.strip()] for m in _RAW_TENSOR_RE.finditer(str(raw))]
    n_operands = len(equation.split("->", 1)[0].split(","))
    if len(raw_shapes) != n_operands:
        return None
    saved = [list(s) for s in (ts.inputs or []) if isinstance(s, list)]
    if len(saved) >= len(raw_shapes):
        return None
    j = 0
    for shape in raw_shapes:
        if j < len(saved) and shape == saved[j]:
            j += 1
    if j != len(saved):
        return None
    return raw_shapes


def _layer_raw_attributes(layer: Dict[str, Any]) -> Any:
    raw = layer.get("raw_attributes")
    if raw is None:
        raw = (layer.get("module_args") or {}).get("raw_attributes")
    return raw


def _layer_access_entry(
    layer: Dict[str, Any],
    direction: str,
    slot_index: int,
) -> Optional[Dict[str, Any]]:
    """Return one structured ``access`` entry for an input/output slot."""
    access = layer.get("access")
    if not isinstance(access, dict):
        return None
    entries = access.get(direction)
    if isinstance(entries, dict):
        entry = entries.get(slot_index)
        if entry is None:
            entry = entries.get(str(slot_index))
        return entry if isinstance(entry, dict) else None
    if not isinstance(entries, list):
        return None

    has_explicit_index = any(isinstance(entry, dict) and "index" in entry for entry in entries)
    if has_explicit_index:
        for entry in entries:
            if isinstance(entry, dict) and entry.get("index") == slot_index:
                return entry
        return None

    if 0 <= slot_index < len(entries) and isinstance(entries[slot_index], dict):
        return entries[slot_index]
    return None


def _access_base_size(access: Dict[str, Any]) -> int:
    base_shape = access.get("base_shape")
    if not isinstance(base_shape, list):
        return 0
    try:
        return _product([int(dim) for dim in base_shape])
    except (TypeError, ValueError):
        return 0


def _canonical_access_boxes(
    access: Optional[Dict[str, Any]],
    canonical: str,
    expected_elems: int,
) -> Optional[List[Box]]:
    if not isinstance(access, dict):
        return None
    base_tensor = access.get("base_tensor")
    if base_tensor is not None and str(base_tensor) != canonical:
        return None
    return access_entry_to_boxes(access, expected_elems=expected_elems)


_CREATION_OPS_ZERO_READ = frozenset({
    "zeros_like", "ones_like", "full_like", "empty_like", "rand_like",
    "randn_like", "randint_like", "zeros", "ones", "full", "empty",
    "arange", "linspace", "eye",
})
_LAZY_UNARY_OPS = frozenset({
    "to", "type", "type_as", "float", "half", "bfloat16", "double",
    "clone", "contiguous", "copy_", "detach",
})
_GATHER_CONSUMER_OPS = frozenset(SLICE_VIEW_OPS) | frozenset({
    "index_select", "gather", "take", "take_along_dim", "embedding",
})


# --------------------------------------------------------------------------- #
# Structured sparsity (triangular / masked operands)
# --------------------------------------------------------------------------- #
_RAW_TENSOR_RE = re.compile(r"Tensor\(shape=\(([^)]*)\),\s*dtype=torch\.\w+\)")
_SPARSE_VIEW_OPS = frozenset({
    "view", "reshape", "permute", "transpose", "t", "expand", "expand_as",
    "unsqueeze", "squeeze", "flatten", "unflatten", "contiguous", "clone",
    "detach", "to", "float", "half", "bfloat16", "type", "type_as", "repeat",
    "broadcast_to", "movedim", "swapaxes",
})
_ZERO_PRESERVING_UNARY = frozenset({
    "abs", "neg", "__neg__", "relu", "silu", "gelu", "tanh", "sqrt", "rsqrt_zero",
    "square", "sign", "sin", "sinh", "asinh", "atan", "erf", "round", "floor",
    "ceil", "trunc", "dropout", "leaky_relu", "hardtanh", "relu6", "mish",
})
_MASK_FILL_OPS = frozenset({"masked_fill", "masked_fill_"})
_NOT_OPS = frozenset({"__invert__", "logical_not", "bitwise_not"})
_MUL_OPS = frozenset({"mul", "__mul__", "__rmul__", "multiply"})
_DIV_OPS = frozenset({"div", "__truediv__", "divide", "true_divide"})
_ADD_OPS = frozenset({"add", "__add__", "__radd__", "sub", "__sub__", "__rsub__", "subtract"})
_NEG_FILL_THRESHOLD = -1e4   # finfo.min / -1e9 style fills vanish under exp/softmax


def _raw_call_tokens(raw: Any) -> Tuple[List[Any], Dict[str, str]]:
    """Top-level positional tokens ('T' for a tensor, else the literal) and kwargs."""
    text = str(raw or "")
    if not text:
        return [], {}
    if "], {" in text:
        args_part, kw_part = text.rsplit("], {", 1)
    else:
        args_part, kw_part = text, ""
    args_part = _RAW_TENSOR_RE.sub(" T ", args_part)
    args_part = args_part.strip().lstrip("[").lstrip("[")
    tokens: List[Any] = []
    depth = 0
    cur = ""
    for ch in args_part:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth <= 0:
            if cur.strip():
                tokens.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        tokens.append(cur.strip().rstrip("]"))
    kwargs: Dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*:\s*([^,}]+)", kw_part):
        kwargs[m.group(1)] = m.group(2).strip()
    return tokens, kwargs


def _scalar_kind(tok: Optional[str]) -> Optional[str]:
    """'Z' for a zero literal, 'N' for -inf / very negative, None otherwise."""
    if tok is None:
        return None
    t = str(tok).strip().strip("'\"").replace("float(", "").replace(")", "")
    if t in ("T",):
        return None
    try:
        v = float(t)
    except ValueError:
        if "inf" in t and t.startswith("-"):
            return "N"
        return None
    if v == 0.0:
        return "Z"
    if v <= _NEG_FILL_THRESHOLD:
        return "N"
    return None


def _triangular_density(shape: List[Any], op: str, diagonal: int) -> float:
    if not isinstance(shape, list) or len(shape) < 2:
        return 1.0
    n, m = int(shape[-2]), int(shape[-1])
    if n <= 0 or m <= 0:
        return 1.0
    kept = 0
    for i in range(n):
        if op == "tril":
            kept += min(m, max(0, i + diagonal + 1))
        else:
            kept += max(0, m - max(0, i + diagonal))
    return max(0.0, min(1.0, kept / float(n * m)))


def _macs_from_equation(equation: str, input_shapes: List[List[int]]) -> int:
    """Compute MACs for an einsum equation by multiplying all unique dim sizes.

    For a real einsum (mul+add reduction), MACs = product of all unique
    dimension sizes across all input tensors.  This correctly accounts for
    reduction dimensions that don't appear in the output.  (an earlier revision lineage;
    restored with the DSL kernel cost basis — TARGET-DENSITY-GUARD-1.)

    Example: BSH,H2H->BSH2  with shapes [2,32,64],[64,64]
        All dims: B=2, S=32, H=64, H2=64  ->  MACs = 2*32*64*64 = 262144
    """
    try:
        if "->" not in equation:
            return 0
        lhs = equation.split("->")[0]
        operands = [op.strip() for op in lhs.split(",") if op.strip()]
        if len(operands) != len(input_shapes):
            return 0
        dim_sizes: dict = {}
        for operand_str, shape in zip(operands, input_shapes):
            tokens = parse_dim_tokens(operand_str, validate=False)
            if len(tokens) != len(shape):
                return 0
            for token, size in zip(tokens, shape):
                if token not in dim_sizes:
                    dim_sizes[token] = int(size)
        result = 1
        for size in dim_sizes.values():
            result *= size
        return result
    except Exception:
        return 0


def _structured_sparsity(
    layers_in: Dict[str, Any],
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, Set[str]],
) -> Dict[str, float]:
    """MAC fraction per einsum layer implied by triangular / masked operands.

    Tracks, per tensor, the density of "live" entries and the kind of the
    dead ones: ``Z`` (exact zeros, from ``tril``/``triu``/``masked_fill(0)``
    /``x * mask``) or ``N`` (``-inf`` / finfo.min fills, which ``exp`` and
    ``softmax`` turn into zeros). Every MAC of an einsum uses exactly one
    element of each operand, so an operand with live density d lets a
    kernel skip the fraction 1 - d of the MACs; likewise an output that is
    only ever multiplied by a mask of density d (``(C B^T) * L`` in the
    Mamba chunk scan, ``softmax(QK^T + causal)``) only needs the fraction d
    of its entries. The fraction is the minimum over the sparse operands
    and the needed output density; dense graphs are unaffected.
    """
    dens: Dict[str, Tuple[str, float]] = {}   # tensor name -> (kind, live density)

    def get(name: Optional[str]) -> Optional[Tuple[str, float]]:
        return dens.get(name) if name else None

    for _ in range(2):   # tolerate non-topological layer order
        for lid, layer in layers_in.items():
            op = str(layer.get("type", "")).lower()
            names = layer.get("tensor_names") or {}
            ins = list(names.get("inputs") or [])
            outs = list(names.get("outputs") or [])
            shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
            if not outs:
                continue
            toks, kw = _raw_call_tokens(_layer_raw_attributes(layer))
            scalars = [t for t in toks if t != "T"]
            res: Optional[Tuple[str, float]] = None
            if op in ("zeros", "zeros_like", "new_zeros"):
                res = ("Z", 0.0)                       # nothing live yet
            elif op in ("full", "full_like", "new_full"):
                fill = _scalar_kind(kw.get("fill_value") or (scalars[-1] if scalars else None))
                if fill:
                    res = (fill, 0.0)                  # e.g. full((S, S), -inf)
            elif op in ("tril", "triu"):
                diag = 0
                if "diagonal" in kw:
                    try:
                        diag = int(float(kw["diagonal"]))
                    except ValueError:
                        diag = 0
                elif scalars:
                    try:
                        diag = int(float(scalars[0]))
                    except (ValueError, OverflowError):
                        diag = 0
                kept = _triangular_density(shapes[0] if shapes else None, op, diag)
                src = get(ins[0] if ins else None)
                orphan_input = bool(ins) and ins[0] not in tensor_producers
                consumers = {str((layers_in.get(c) or {}).get("type", "")).lower()
                             for o in outs for c in (tensor_consumers.get(o) or ())}
                if src and src[0] == "N" and src[1] == 0.0:
                    # triu(full(-inf), 1): the zeroed triangle is the live
                    # part of an additive mask; the kept -inf entries kill.
                    res = ("N", 1.0 - kept)
                elif (orphan_input and consumers and consumers <= _ADD_OPS):
                    # The filled constant (torch.full(..., -inf)) is built in
                    # forward and not traced; a triangular constant that is
                    # only ever ADDED to scores is the HF-style additive
                    # causal mask, whose kept triangle is the -inf part.
                    res = ("N", 1.0 - kept)
                elif src and src[0] == "Z":
                    res = ("Z", min(src[1], kept))
                else:
                    res = ("Z", kept)
            elif op in _NOT_OPS:
                a = get(ins[0] if ins else None)
                if a and a[0] == "Z":
                    res = ("Z", 1.0 - a[1])
            elif op in _MASK_FILL_OPS and len(ins) >= 2:
                mask = get(ins[1])
                fill = _scalar_kind(kw.get("value") or (scalars[0] if scalars else None))
                if mask and mask[0] == "Z" and fill:
                    x = get(ins[0])
                    d = 1.0 - mask[1]
                    if x and x[0] == fill:
                        d = min(d, x[1])
                    res = (fill, d)
            elif op == "where" and len(toks) >= 3:
                cond = get(ins[0]) if ins else None
                if cond and cond[0] == "Z":
                    a_kind = _scalar_kind(toks[1]) if toks[1] != "T" else None
                    b_kind = _scalar_kind(toks[2]) if toks[2] != "T" else None
                    if b_kind:            # where(cond, x, 0): live where cond is True
                        res = (b_kind, cond[1])
                    elif a_kind:          # where(cond, 0, x): live where cond is False
                        res = (a_kind, 1.0 - cond[1])
            elif op in _MUL_OPS:
                tens = [get(n) for n in ins]
                zs = [t for t in tens if t and t[0] == "Z"]
                ns = [t for t in tens if t and t[0] == "N"]
                if zs:
                    res = ("Z", min(t[1] for t in zs))
                elif ns and len(ins) == 1:
                    res = ns[0]           # -inf * scalar keeps the pattern
            elif op in _DIV_OPS:
                a = get(ins[0] if ins else None)
                if a:
                    res = a
            elif op in _ADD_OPS:
                tens = [get(n) for n in ins]
                ns = [t for t in tens if t and t[0] == "N"]
                if ns and len(ins) == 1:
                    res = ns[0]           # x + c keeps -inf
                elif ns and len(ins) >= 2 and all(t is None or t[0] == "N" or True for t in tens):
                    # -inf + finite = -inf: the union of the -inf patterns
                    res = ("N", min(t[1] for t in ns))
            elif op in ("exp", "softmax", "log_softmax", "exp_", "softmax_"):
                a = get(ins[0] if ins else None)
                if a and a[0] == "N":
                    res = ("Z", a[1])
            elif op in _SPARSE_VIEW_OPS:
                a = get(ins[0] if ins else None)
                if a:
                    res = a
            elif op in _ZERO_PRESERVING_UNARY:
                a = get(ins[0] if ins else None)
                if a and a[0] == "Z":
                    res = a
            for o in outs:
                if res is not None:
                    dens[o] = res
                else:
                    dens.pop(o, None)

    def needed_density(tensor: str, depth: int = 0) -> float:
        """Fraction of ``tensor`` entries some consumer actually uses."""
        consumers = tensor_consumers.get(tensor) or set()
        if not consumers or depth > 6:
            return 1.0
        need = 0.0
        gather_need = 0.0
        for cid in consumers:
            c = layers_in.get(cid) or {}
            cop = str(c.get("type", "")).lower()
            cins = list((c.get("tensor_names") or {}).get("inputs") or [])
            couts = list((c.get("tensor_names") or {}).get("outputs") or [])
            frac = 1.0
            if cop in _GATHER_CONSUMER_OPS and cins and cins[0] == tensor:
                # Only the gathered / sliced rows of the product are used
                # (projector applied to every window, 80 of 1720 rows kept).
                src = _tensor_elems(layers_in, tensor_producers, tensor)
                got = sum(_product(sh) for sh in ((c.get("tensor_shapes") or {}).get("outputs") or [])
                          if isinstance(sh, list))
                frac = min(1.0, got / src) if src > 0 and got > 0 else 1.0
                # A slice that is itself gathered further on (y[i][pos]) only
                # needs what its own consumers need.
                if couts:
                    frac *= max(needed_density(o, depth + 1) for o in couts)
                gather_need += frac
                continue
            cshapes_in = (c.get("tensor_shapes") or {}).get("inputs") or []
            cshape_out = ((c.get("tensor_shapes") or {}).get("outputs") or [None])[0]
            my_shape = None
            for n_, sh_ in zip(cins, cshapes_in):
                if n_ == tensor:
                    my_shape = sh_
            full_operand = my_shape is not None and my_shape == cshape_out
            if cop in _SPARSE_VIEW_OPS:
                frac = max(needed_density(o, depth + 1) for o in couts) if couts else 1.0
            elif cop in _MUL_OPS and len(cins) >= 2:
                others = [get(n) for n in cins if n != tensor]
                zs = [t for t in others if t and t[0] == "Z"]
                frac = min(t[1] for t in zs) if zs else (
                    max(needed_density(o, depth + 1) for o in couts) if couts and full_operand else 1.0)
            elif cop in _ADD_OPS and len(cins) >= 2 and any(
                    (get(n) or ("", 1.0))[0] == "N" for n in cins if n != tensor):
                # scores + causal_mask(-inf): entries that become -inf are
                # never needed.
                ns = [get(n) for n in cins if n != tensor and get(n) and get(n)[0] == "N"]
                frac = min(t[1] for t in ns)
            elif (cop in _ADD_OPS or cop in _DIV_OPS or cop in _MUL_OPS or cop in _ZERO_PRESERVING_UNARY
                  or cop in ("sigmoid", "log", "rsqrt", "pow", "__pow__", "clamp", "clip")) and full_operand:
                # Elementwise: element j of the input feeds element j of the output only
                # (bias add after a linear whose rows are then gathered).
                frac = max(needed_density(o, depth + 1) for o in couts) if couts else 1.0
            elif cop in _MASK_FILL_OPS and len(cins) >= 2 and cins[0] == tensor:
                toks, kw = _raw_call_tokens(_layer_raw_attributes(c))
                scalars = [t for t in toks if t != "T"]
                mask = get(cins[1])
                fill = _scalar_kind(kw.get("value") or (scalars[0] if scalars else None))
                frac = 1.0 - mask[1] if (mask and mask[0] == "Z" and fill) else 1.0
            elif cop == "where" and len(cins) >= 2 and cins[0] != tensor:
                cond = get(cins[0])
                toks, _ = _raw_call_tokens(_layer_raw_attributes(c))
                if cond and cond[0] == "Z" and len(toks) >= 3:
                    # tensor is the 2nd tensor arg (x) or the 3rd (y)
                    idx = cins.index(tensor)
                    if idx == 1 and len(toks) > 2 and toks[2] != "T" and _scalar_kind(toks[2]):
                        frac = cond[1]
                    elif idx == 2 and toks[1] != "T" and _scalar_kind(toks[1]):
                        frac = 1.0 - cond[1]
            need = max(need, frac)
            if need >= 1.0:
                return 1.0
        return min(1.0, max(need, gather_need))

    replicated = _replicated_dims(layers_in, tensor_producers)
    fractions: Dict[str, float] = {}
    for lid, layer in layers_in.items():
        if not layer.get("is_real_einsum"):
            continue
        names = layer.get("tensor_names") or {}
        frac = 1.0
        for n in names.get("inputs") or []:
            t = get(n)
            if t and t[0] == "Z":
                frac = min(frac, t[1])
        for o in names.get("outputs") or []:
            frac = min(frac, needed_density(o))
        frac /= _einsum_redundancy(layer, replicated)
        if frac < 1.0:
            fractions[lid] = max(frac, 0.0)
    return fractions


def _tensor_elems(layers_in: Dict[str, Any], tensor_producers: Dict[str, str], tensor: str) -> int:
    pid = tensor_producers.get(tensor)
    layer = layers_in.get(pid) or {}
    names = (layer.get("tensor_names") or {}).get("outputs") or []
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    for n, sh in zip(names, shapes):
        if n == tensor and isinstance(sh, list):
            return _product(sh)
    return 0


_EXPAND_OPS = frozenset({"expand", "expand_as", "broadcast_to"})
_REPL_PASSTHRU = frozenset({
    "to", "float", "half", "bfloat16", "double", "type", "type_as", "clone",
    "contiguous", "detach", "abs", "neg", "exp", "relu", "silu", "gelu", "tanh",
    "sigmoid", "sqrt", "rsqrt", "square", "log", "softmax", "cumsum",
})
_REPL_BINARY = frozenset({"mul", "__mul__", "__rmul__", "add", "__add__", "__radd__",
                          "sub", "__sub__", "__rsub__", "div", "__truediv__", "where",
                          "masked_fill", "maximum", "minimum", "pow", "__pow__"})
_EINSUM_TOKEN_RE = re.compile(r"[A-Za-z]\d*")


def _reshape_dim_map(in_shape: List[int], out_shape: List[int]) -> Dict[int, int]:
    """Input dim -> output dim for dims that survive a reshape one-to-one."""
    m: Dict[int, int] = {}
    i = j = 0
    while i < len(in_shape) and j < len(out_shape):
        pi, pj = int(in_shape[i]), int(out_shape[j])
        i0, j0 = i, j
        while pi != pj:
            if pi < pj:
                i += 1
                if i >= len(in_shape):
                    return m
                pi *= int(in_shape[i])
            else:
                j += 1
                if j >= len(out_shape):
                    return m
                pj *= int(out_shape[j])
        if i == i0 and j == j0:
            m[i] = j
        i += 1
        j += 1
    return m


def _replicated_dims(
    layers_in: Dict[str, Any], tensor_producers: Dict[str, str]
) -> Dict[str, Dict[int, int]]:
    """Per tensor: {dim: replication factor} for dims that only repeat values.

    ``expand`` of a size-1 dim, ``repeat`` / ``repeat_interleave`` create
    dims along which every slice is a copy (GQA K/V repeated per group,
    Mamba-2 B/C expanded from one group to every head). The map follows
    casts, elementwise ops whose every operand is replicated the same way,
    unsqueeze/squeeze/permute/transpose and reshapes that keep the dim.
    """
    rep: Dict[str, Dict[int, int]] = {}

    def shp(layer, key, i=0):
        shapes = (layer.get("tensor_shapes") or {}).get(key) or []
        return [int(x) for x in shapes[i]] if i < len(shapes) and isinstance(shapes[i], list) else None

    for _ in range(2):
        for lid, layer in layers_in.items():
            op = str(layer.get("type", "")).lower()
            names = layer.get("tensor_names") or {}
            ins = list(names.get("inputs") or [])
            outs = list(names.get("outputs") or [])
            if not outs:
                continue
            if (layer.get("additional_info") or {}).get(
                    "synthetic_recurrence_placeholder"):
                # Explicit lowering provenance: this broadcast only wires an
                # internally synthesized recurrent placeholder (RNN h0 across
                # timesteps).  Its repeated values are a lowering artifact,
                # not reusable work — recording no replication keeps the
                # downstream GEMMs billed at full as-written cost.
                for o in outs:
                    rep.pop(o, None)
                continue
            out_shape = shp(layer, "outputs")
            in_shape = shp(layer, "inputs")
            res: Dict[int, int] = {}
            toks, kw = _raw_call_tokens(_layer_raw_attributes(layer))
            scalars = []
            for t in toks:
                if t == "T":
                    continue
                try:
                    v = float(t)
                except ValueError:
                    continue
                if math.isfinite(v):
                    scalars.append(int(v))
            src = rep.get(ins[0], {}) if ins else {}
            if op in _EXPAND_OPS and in_shape is not None and out_shape is not None:
                off = len(out_shape) - len(in_shape)
                for j, oj in enumerate(out_shape):
                    i = j - off
                    if i < 0:
                        if oj > 1:
                            res[j] = oj
                    elif in_shape[i] == 1 and oj > 1:
                        res[j] = oj
                    elif i in src:
                        res[j] = src[i]
            elif op == "repeat_interleave" and in_shape is not None and out_shape is not None:
                dim = kw.get("dim")
                try:
                    dim = int(float(dim)) if dim is not None else (scalars[1] if len(scalars) > 1 else None)
                except (ValueError, OverflowError):
                    dim = None
                reps = scalars[0] if scalars else None
                if dim is not None and reps and len(in_shape) == len(out_shape):
                    dim = dim % len(out_shape)
                    res = dict(src)
                    res[dim] = res.get(dim, 1) * reps
            elif op == "repeat" and in_shape is not None and out_shape is not None and scalars:
                off = len(out_shape) - len(in_shape)
                for j, oj in enumerate(out_shape):
                    k = j - (len(out_shape) - len(scalars))
                    f = scalars[k] if 0 <= k < len(scalars) else 1
                    i = j - off
                    base = src.get(i, 1) if i >= 0 else 1
                    if f > 1 or base > 1:
                        res[j] = base * f
            elif op in ("unsqueeze",) and in_shape is not None and out_shape is not None and scalars:
                d = scalars[0] % len(out_shape)
                res = {(i + 1 if i >= d else i): f for i, f in src.items()}
            elif op in ("squeeze",) and in_shape is not None and out_shape is not None:
                kept = [i for i, x in enumerate(in_shape) if x != 1 or len(in_shape) == len(out_shape)]
                if len(kept) == len(out_shape):
                    pos = {i: j for j, i in enumerate(kept)}
                    res = {pos[i]: f for i, f in src.items() if i in pos}
            elif op in ("permute",) and len(scalars) == len(out_shape or []):
                perm = [d % len(out_shape) for d in scalars]
                res = {j: src[i] for j, i in enumerate(perm) if i in src}
            elif op in ("transpose", "swapaxes", "swapdims") and len(scalars) >= 2 and out_shape is not None:
                a, b = scalars[0] % len(out_shape), scalars[1] % len(out_shape)
                res = {}
                for i, f in src.items():
                    res[b if i == a else a if i == b else i] = f
            elif op == "t" and out_shape is not None and len(out_shape) == 2:
                res = {1 - i: f for i, f in src.items()}
            elif op in ("view", "reshape", "flatten", "unflatten") and in_shape is not None and out_shape is not None:
                m = _reshape_dim_map(in_shape, out_shape)
                res = {m[i]: f for i, f in src.items() if i in m}
            elif op in _REPL_PASSTHRU and len(ins) >= 1:
                res = dict(src)
            elif op in _REPL_BINARY and out_shape is not None and len(ins) >= 1:
                res = {}
                for j, oj in enumerate(out_shape):
                    fs = []
                    ok = True
                    for k, n in enumerate(ins):
                        ish = shp(layer, "inputs", k)
                        if ish is None:
                            ok = False
                            break
                        i = j - (len(out_shape) - len(ish))
                        if i < 0 or ish[i] == 1:
                            fs.append(oj)          # broadcast operand: trivially replicated
                        elif i in rep.get(n, {}):
                            fs.append(rep[n][i])
                        else:
                            ok = False
                            break
                    if ok and fs and min(fs) > 1 and oj > 1:
                        res[j] = min(fs)
            for o in outs:
                if res:
                    rep[o] = res
                else:
                    rep.pop(o, None)
    return rep


def _einsum_redundancy(layer: Dict[str, Any], replicated: Dict[str, Dict[int, int]]) -> float:
    """Factor by which an einsum's MACs repeat identical work.

    For every index letter whose every operand is replicated along it with
    the same factor, the products along that index are copies: a kernel
    computes them once (Mamba-2 ``C B^T`` with one B/C group expanded to 16
    heads is 16x redundant). Returns the product of such factors (>= 1).
    """
    eq = str(layer.get("einsum_equation") or "")
    if "->" not in eq:
        return 1.0
    lhs = eq.split("->")[0]
    subs = [_EINSUM_TOKEN_RE.findall(part) for part in lhs.split(",")]
    names = (layer.get("tensor_names") or {}).get("inputs") or []
    shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    if len(subs) != len(names) or len(subs) != len(shapes):
        return 1.0
    per_letter: Dict[str, List[Optional[int]]] = {}
    for toks, name, sh in zip(subs, names, shapes):
        if not isinstance(sh, list) or len(toks) != len(sh):
            return 1.0
        r = replicated.get(name, {})
        for d, tok in enumerate(toks):
            if int(sh[d]) <= 1:
                continue
            per_letter.setdefault(tok, []).append(r.get(d))
    factor = 1.0
    for tok, fs in per_letter.items():
        if fs and all(f is not None for f in fs) and len(set(fs)) == 1:
            factor *= fs[0]
    return max(factor, 1.0)


_WRITE_PASSTHRU = frozenset({
    "to", "float", "half", "bfloat16", "double", "type", "type_as", "clone",
    "contiguous", "detach", "view", "reshape", "permute", "transpose", "t",
    "unsqueeze", "squeeze", "flatten", "unflatten",
})
_CREATION_OUTPUT_OPS = frozenset({"zeros_like", "zeros", "full_like", "full", "empty_like",
                                  "empty", "ones_like", "ones", "new_zeros", "new_full"})


def _output_write_caps(
    layers_in: Dict[str, Any], tensor_producers: Dict[str, str]
) -> Dict[str, int]:
    """Smaller-than-shape write footprints for layers that end an output.

    * ``mask.expand(B, H, T, S).contiguous()``: the reference materialises a
      broadcast; the minimal implementation returns the view and writes only
      the base ``[T, S]`` tensor.
    * ``grad = zeros(...); grad.index_add_(rows); grad.to(bf16)``: a sparse
      update of a zero-initialised buffer; only the updated rows are written
      (the benchmark harness provides the zeroed buffer).
    Returned values cap the external-output write of the given layer.
    """
    caps: Dict[str, int] = {}

    def first_input_layer(layer):
        ins = (layer.get("tensor_names") or {}).get("inputs") or []
        return tensor_producers.get(ins[0]) if ins else None

    def in_sizes(layer):
        return [_product(sh) for sh in ((layer.get("tensor_shapes") or {}).get("inputs") or [])
                if isinstance(sh, list)]

    def passthru(layer) -> bool:
        t = str(layer.get("type", "")).lower()
        if t in _WRITE_PASSTHRU or t in _ZERO_PRESERVING_UNARY:
            return True
        # Elementwise op with a single tensor operand (scalar multiply/add):
        # same footprint in and out.
        if t in _MUL_OPS or t in _ADD_OPS or t in _DIV_OPS:
            return len((layer.get("tensor_names") or {}).get("inputs") or []) == 1
        return False

    for lid, layer in layers_in.items():
        op = str(layer.get("type", "")).lower()
        cur_id, cur = lid, layer
        hops = 0
        while cur is not None and passthru(cur) and hops < 12:
            nid = first_input_layer(cur)
            cur_id, cur = nid, layers_in.get(nid)
            hops += 1
        if cur is None:
            continue
        ctype = str(cur.get("type", "")).lower()
        if ctype in _EXPAND_OPS and (cur_id != lid or op in _EXPAND_OPS):
            sizes = in_sizes(cur)
            if sizes:
                caps[lid] = sizes[0]
            continue
        # Sparse update chain: scatters into a creation-op buffer.
        total = 0
        seen = 0
        while hops < 64:
            if cur is None:
                # Producer-less target (torchview orphan of the zero buffer,
                # or a model input updated in place): only the scattered
                # rows are written.
                if seen:
                    caps[lid] = total
                break
            t = str(cur.get("type", "")).lower()
            if passthru(cur):
                pass
            elif t in ("__setitem__", "scatter", "scatter_", "index_copy", "index_copy_",
                       "index_put", "index_put_", "index_add", "index_add_", "scatter_add",
                       "scatter_add_", "scatter_reduce", "scatter_reduce_", "masked_scatter",
                       "masked_scatter_", "put_"):
                sizes = in_sizes(cur)
                total += max(sorted(sizes)[:-1]) if len(sizes) >= 2 else (sizes[0] if sizes else 0)
                seen += 1
            elif t in _CREATION_OUTPUT_OPS:
                if seen:
                    caps[lid] = total
                break
            else:
                break
            nid = first_input_layer(cur)
            cur = layers_in.get(nid)
            hops += 1
    return caps


def _canonical_external_tensor(
    tensor_name: str,
    tensor_producers: Dict[str, str],
    layers_in: Dict[str, Any],
    transparent_layer_ids: Set[str],
) -> str:
    """Trace a tensor name backward through transparent view layers.

    Returns the base tensor name (start-node output, weight, or orphan)
    when the whole producer chain is transparent.  When the chain cannot
    be followed (real producer, dangling view, cycle), the original name
    is returned so the tensor keeps its own dedup group; that reproduces
    the previous per-name accounting and never merges unrelated tensors.
    """
    seen: Set[str] = set()
    current = tensor_name
    while current in tensor_producers:
        if current in seen:
            return tensor_name
        seen.add(current)
        producer_id = tensor_producers[current]
        if producer_id not in transparent_layer_ids:
            return tensor_name
        producer_inputs = (
            (layers_in.get(producer_id, {}).get("tensor_names") or {}).get("inputs") or []
        )
        if not producer_inputs or not producer_inputs[0]:
            return tensor_name
        current = str(producer_inputs[0])
    return current


def _partition_output_index(output_names: List[str], tensor_name: str) -> Optional[int]:
    """Position of ``tensor_name`` among a partition op's ordered outputs.

    The recorded output order must match the ``.Output`` / ``.Output_<k>``
    suffix convention; misassigning a partition slot to the wrong region
    could overcount the union, so any deviation rejects the fast path.
    """
    for position, name in enumerate(output_names):
        suffix = str(name).rsplit(".", 1)[-1]
        if position == 0:
            if suffix != "Output":
                return None
        elif suffix != f"Output_{position}":
            return None
    try:
        return output_names.index(tensor_name)
    except ValueError:
        return None


def _resolve_read_region(
    op_type: str,
    input_index: int,
    tensor_name: str,
    mem_read: int,
    input_shapes: List[Any],
    input_sizes: List[int],
    layer: Dict[str, Any],
    layers_in: Dict[str, Any],
    tensor_producers: Dict[str, str],
    transparent_layer_ids: Set[str],
) -> Tuple[str, Optional[List[Box]], int]:
    """Resolve one external read into (group key, proven boxes, full size).

    The group key is the canonical base tensor name.  ``boxes`` is a list
    of proven access boxes over the base tensor, or None when the region
    cannot be established statically (the caller then applies the
    conservative per-group ``max`` accounting).  The returned full size is
    a candidate for the base tensor's total element count; it is 0 when
    the event provides no trustworthy candidate (e.g. reads through views,
    whose local tensor size need not equal the base size).
    """
    canonical = _canonical_external_tensor(
        tensor_name, tensor_producers, layers_in, transparent_layer_ids
    )

    if tensor_name not in tensor_producers:
        # Direct read of the base tensor: its recorded size is the base size.
        full_candidate = int(input_sizes[input_index]) if input_index < len(input_sizes) else 0
        boxes = _canonical_access_boxes(
            _layer_access_entry(layer, "inputs", input_index),
            canonical,
            int(mem_read),
        )
        if boxes is None and op_type in SLICE_VIEW_OPS and input_index == 0:
            base_shape = input_shapes[0] if input_shapes else None
            if isinstance(base_shape, list):
                boxes = slice_op_boxes(
                    op_type,
                    _layer_raw_attributes(layer),
                    base_shape,
                    expected_elems=int(mem_read),
                )
        return canonical, boxes, full_candidate

    # Read through a view chain.  The only statically provable case is a
    # direct consumer of one output of a partition op (chunk/split) whose
    # input is the base tensor itself.
    producer_id = tensor_producers[tensor_name]
    producer = layers_in.get(producer_id) or {}
    producer_type = str(producer.get("type", "")).lower()
    if producer_type in PARTITION_OPS:
        p_names = producer.get("tensor_names") or {}
        p_shapes = producer.get("tensor_shapes") or {}
        p_inputs = p_names.get("inputs") or []
        base_name = str(p_inputs[0]) if p_inputs and p_inputs[0] else ""
        base_shape = (p_shapes.get("inputs") or [None])[0]
        output_names = [str(n) for n in (p_names.get("outputs") or [])]
        output_shapes = p_shapes.get("outputs") or []
        if (
            base_name
            and base_name == canonical
            and base_name not in tensor_producers
            and isinstance(base_shape, list)
            and all(isinstance(s, list) for s in output_shapes)
        ):
            output_index = _partition_output_index(output_names, tensor_name)
            if output_index is not None:
                access = _layer_access_entry(producer, "outputs", output_index)
                boxes = _canonical_access_boxes(access, canonical, int(mem_read))
                if boxes is not None:
                    full_size = _access_base_size(access) if access is not None else 0
                    return canonical, boxes, full_size or _product(base_shape)

                box = partition_output_box(
                    producer_type,
                    _layer_raw_attributes(producer),
                    base_shape,
                    output_shapes,
                    output_index,
                )
                if box is not None and box_size(box) == int(mem_read):
                    return canonical, [box], _product(base_shape)

    return canonical, None, 0


# torch dtype string -> bytes per (stored) element. Packed FP4 (float4_e2m1fn_x2)
# stores two values per byte, but the tensor's element count already counts packed
# bytes, so it is 1 byte per element here.
_DTYPE_BYTES = {
    # DSL trace tags carry declared logical widths, independent of the
    # graph-wide compute precision and of NumPy's emulation storage.
    "fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "nvfp4": 0.5,
    "float64": 8, "double": 8, "int64": 8, "long": 8, "complex64": 8,
    "float32": 4, "float": 4, "int32": 4, "int": 4, "tf32": 4,
    "bfloat16": 2, "float16": 2, "half": 2, "int16": 2, "short": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1,
    "int8": 1, "uint8": 1, "byte": 1, "bool": 1, "float4_e2m1fn_x2": 1,
}


def _dtype_bytes(dtype: Any, default: float) -> float:
    """Bytes per element for a torch dtype string; ``default`` when unknown/missing."""
    if not dtype:
        return default
    key = str(dtype).replace("torch.", "").strip().lower()
    return float(_DTYPE_BYTES.get(key, default))


def _group_footprints(
    groups: Dict[str, Dict[str, Any]],
    base_full_sizes: Dict[str, int],
    debug: bool = False,
) -> Dict[str, int]:
    """Per-base-tensor unique read footprint (elements); see _sum_group_footprints."""
    out: Dict[str, int] = {}
    for name, group in groups.items():
        boxes: List[Box] = group["boxes"]
        counted: List[int] = list(group["counted"])
        full_candidates: Set[int] = {c for c in group["full"] if c > 0}
        if name in base_full_sizes and base_full_sizes[name] > 0:
            full_candidates.add(int(base_full_sizes[name]))
        if len(full_candidates) > 1:
            counted.extend(box_size(b) for b in boxes)
            footprint = min(max(counted, default=0), min(full_candidates))
            if debug:
                logger.debug(f"external tensor '{name}' has conflicting full "
                      f"sizes {sorted(full_candidates)}; using conservative bound")
        else:
            union = union_size(boxes) if boxes else 0
            if union is None:
                counted.extend(box_size(b) for b in boxes)
                union = 0
            footprint = max(union, max(counted, default=0))
            if full_candidates:
                footprint = min(footprint, full_candidates.pop())
        out[name] = int(footprint)
    return out


def _sum_group_footprints(
    groups: Dict[str, Dict[str, Any]],
    base_full_sizes: Dict[str, int],
    debug: bool = False,
) -> int:
    """Unique-footprint total across per-base-tensor read groups.

    Per group the footprint is ``min(full, max(union(boxes), max(counted)))``:
    exact when every access resolved to a proven box, and a valid lower
    bound otherwise (any single access's element count, and the union of
    proven regions, are both lower bounds of the true unique footprint).
    Inconsistent full-size metadata degrades the group to the conservative
    bound capped at the smallest candidate.
    """
    total = 0
    for name, group in groups.items():
        boxes: List[Box] = group["boxes"]
        counted: List[int] = list(group["counted"])
        full_candidates: Set[int] = {c for c in group["full"] if c > 0}
        if name in base_full_sizes and base_full_sizes[name] > 0:
            full_candidates.add(int(base_full_sizes[name]))

        if len(full_candidates) > 1:
            # Conflicting size metadata: distrust the boxes, keep the bound.
            counted.extend(box_size(b) for b in boxes)
            footprint = min(max(counted, default=0), min(full_candidates))
            if debug:
                logger.debug(f"external tensor '{name}' has conflicting full "
                    f"sizes {sorted(full_candidates)}; using conservative bound")
        else:
            union = union_size(boxes) if boxes else 0
            if union is None:
                # Mixed ranks or budget overflow: degrade boxes to counts.
                counted.extend(box_size(b) for b in boxes)
                union = 0
            footprint = max(union, max(counted, default=0))
            if full_candidates:
                footprint = min(footprint, full_candidates.pop())
        total += int(footprint)
    return total


# Pure aliases of an existing storage (issue #35, an earlier decision).  They do no ALU
# work and move no bytes, so in unfused mode they are billed zero other_ops
# and zero DRAM traffic; graph connectivity through them is preserved and the
# fused model still sees a model input or output that passes through one.
# Real copies (clone, copy_, to/type casts) stay charged, and contiguous()
# keeps its pre-existing view treatment because the captured graph cannot
# prove whether it copied.  The convention is documented in
# docs/EINSUM_GUIDE.md ("Alias, View and Copy Operations").
ALIAS_OPS = frozenset({
    "detach", "detach_", "alias",
    "view_as",
    "swapaxes", "swapdims", "movedim", "moveaxis",
    "as_strided", "unbind",
})

# Layout-dependent aliases (an earlier decision review P2 reshape-1): PyTorch
# documents reshape_as as "a view OR a new tensor".  It is billed as an alias
# only when the source operand's contiguous layout is ESTABLISHED from the
# captured graph (see _source_layout_established); otherwise the copy is
# charged (source read + output write, zero ALU work).  Its second operand
# supplies a shape only and is never a data read.
LAYOUT_DEPENDENT_ALIAS_OPS = frozenset({"reshape_as"})

# Layout provenance for the reshape_as decision (an earlier decision review P2
# layout-2 / input-2).  A positive "layout established" proof comes
# only from (a) contiguity RECORDED AT CAPTURE for a model input or parameter
# (``tensor_layouts`` in einsum_graph.yaml), (b) an operation that allocates a
# fresh row-major result whatever its inputs' strides, (c) an operation whose
# RECORDED call arguments guarantee it (``contiguous()`` without a
# memory_format, ``clone(memory_format=torch.contiguous_format)``), or (d) a
# layout-preserving / layout-inheriting operation whose data inputs are all
# established.  Operation names alone never prove a layout: ``clone`` keeps a
# dense transposed layout, ``einsum('ij->ji')`` returns the transposed view,
# ``contiguous``/``to`` with ``memory_format=channels_last`` are channels-last.
# Everything unknown is charged as a copy, never reported as established.
_LAYOUT_PRESERVING_VIEWS = frozenset({
    "view", "reshape", "reshape_as", "view_as", "flatten", "unflatten",
    "squeeze", "unsqueeze", "detach", "detach_", "alias",
})
# Fresh row-major output regardless of the inputs' strides (new allocation).
_FRESH_ROW_MAJOR_OPS = frozenset({
    "matmul", "mm", "bmm", "linear", "addmm", "embedding",
})
# Output follows the inputs' memory format (TensorIterator / memory-format
# propagation, including convolutions and concatenations): established only
# when every data input is.
_LAYOUT_INHERITING_OPS = frozenset({
    "add", "sub", "mul", "div", "pow", "add_", "sub_", "mul_", "div_",
    "__add__", "__sub__", "__mul__", "__truediv__", "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
    "relu", "sigmoid", "tanh", "gelu", "selu", "elu", "mish", "softmax", "log_softmax", "softplus",
    "hardswish", "hardsigmoid", "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos",
    "clamp", "clamp_", "relu_", "dropout", "dropout_",
    "layer_norm", "layernorm", "batch_norm", "batchnorm", "group_norm", "groupnorm",
    "conv1d", "conv2d", "conv3d",
    "convtranspose1d", "convtranspose2d", "convtranspose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "cat", "concat", "stack",
})
# Output layout depends on a recorded ``memory_format`` argument (default /
# preserve_format inherit the source layout; contiguous_format establishes
# it; channels_last & co. do not).  Without recorded call attributes the
# layout is unknown.
_MEMORY_FORMAT_OPS = frozenset({
    "contiguous", "clone", "to", "type", "type_as", "float", "half", "bfloat16", "int",
})


def _memory_format_evidence(layer: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """``(memory_format, recorded)`` from the layer's recorded call attributes.

    ``("default", True)`` when the call was recorded without a memory_format
    argument, ``("channels_last", True)`` and so on when it was, and
    ``(None, False)`` when torchview recorded no call attributes at all.
    """
    raw = layer.get("raw_attributes")
    if not raw:
        return None, False
    m = re.search(r"memory_format['\"]?\s*[:=]\s*(?:torch\.)?(\w+)", str(raw))
    return (m.group(1) if m else "default"), True


_TENSOR_ARG_RE = re.compile(r"Tensor\(shape=\(([^)]*)\)")


def _recorded_positional_tensor_shapes(layer: Dict[str, Any]) -> Optional[List[List[int]]]:
    """Shapes of the tensor arguments recorded positionally in ``raw_attributes``
    (``[[<positional>], {<kwargs>}]``), or None when no call was recorded."""
    raw = layer.get("raw_attributes")
    if not raw:
        return None
    text = str(raw)
    split = text.rfind("], {")
    positional = text[:split + 1] if split >= 0 else text
    shapes: List[List[int]] = []
    for m in _TENSOR_ARG_RE.finditer(positional):
        dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
        try:
            shapes.append([int(d) for d in dims])
        except ValueError:
            shapes.append([])
    return shapes


def _has_destination_argument(layer: Dict[str, Any]) -> Optional[bool]:
    """True when the recorded call passed ``out=`` (a destination tensor), False
    when a call was recorded without one, None when no call was recorded.

    The identity record written at capture (``call_operands.destination``) is
    authoritative when present; the recorded call text is the fallback."""
    co = layer.get("call_operands")
    if isinstance(co, dict) and "receiver_bound" in co:
        return "destination" in co
    raw = layer.get("raw_attributes")
    if not raw:
        return None
    return re.search(r"(?:^|[\s{,])out\s*[:=]\s*Tensor", str(raw)) is not None


def _reshape_as_receiver_present(layer: Dict[str, Any]) -> Optional[bool]:
    """Is ``reshape_as``'s receiver a connected graph operand bound by tensor
    identity?  True / False from the capture's ``call_operands.receiver_bound``
    record (``tensor_layouts.bind_call_operands``); None when no identity
    record exists (no call recorded, or a graph written before the record
    existed), which is *unverified*, never a positive proof.

    Counts and shapes of the recorded arguments are not evidence
    (input-4): ``self.b.reshape_as(other=other)`` records the buffer
    positionally and a same-shaped ``other`` under a keyword, and only
    ``other`` survives as an operand.  A buffer or plain-tensor attribute
    receiver never becomes a graph operand, so it is unbound whatever its
    shape, and the shape-reference tensor's layout is never borrowed.
    """
    co = layer.get("call_operands")
    if not isinstance(co, dict) or "receiver_bound" not in co:
        return None
    return _reshape_as_receiver_state(layer) == "bound"


def _reshape_as_receiver_state(layer: Dict[str, Any]) -> str:
    """``bound`` (identity-bound receiver whose binding-time layout record still
    describes the executed receiver), ``mutated`` (identity-bound, but the
    executed storage/strides differ from the record, or the executed receiver
    is not contiguous: ``.data`` assignment, ``set_``, ``as_strided_``,
    ``resize_`` (storage-5)), ``unbound`` (not a connected graph
    operand) or ``unverified`` (no identity record)."""
    co = layer.get("call_operands")
    if not isinstance(co, dict) or "receiver_bound" not in co:
        return "unverified"
    if co.get("receiver_bound") is not True:
        return "unbound"
    rl = co.get("receiver_layout")
    if not isinstance(rl, dict):
        return "unverified"                    # identity alone (a record without the validity sub-record) never authorizes reuse
    if rl.get("layout_valid") is False:
        return "mutated"                       # storage and/or strides differ from the binding-time record
    return "bound"


def _is_overlapping_expand(layer: Dict[str, Any]) -> bool:
    """True for an ``expand``/``expand_as`` that broadcast a size-1 axis (stride-0 result)."""
    if str(layer.get("type", "")).lower() not in ("expand", "expand_as"):
        return False
    shapes = layer.get("tensor_shapes") or {}
    ins, outs = shapes.get("inputs") or [], shapes.get("outputs") or []
    if not ins or not outs or not isinstance(ins[0], list) or not isinstance(outs[0], list):
        return False
    src, dst = list(ins[0]), list(outs[0])
    if len(dst) < len(src):
        return False
    pad = len(dst) - len(src)
    return any(int(dst[pad + i]) > 1 and int(src[i]) == 1 for i in range(len(src))) or any(int(d) > 1 for d in dst[:pad])

_CONV_TRANSPOSE_TYPES = frozenset({
    "convtranspose1d", "convtranspose2d", "convtranspose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
})


def convtranspose_min_work_macs(
    input_shape: Optional[List[int]],
    weight_shape: Optional[List[int]],
    output_shape: Optional[List[int]],
    conv_args: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Minimum executable MACs of a transposed convolution (issue #44).

    Per spatial dim an input index ``i`` and tap ``r`` write output index
    ``o = i*stride - padding + r*dilation``; the product contributes a MAC
    only if ``0 <= o < out``.  min_work = N * C_in * (C_out/groups) *
    prod_dims #{(i, r) surviving}.  The reviewed dense convention (``macs``,
    issue #31) counts every (i, r); the two are equal iff no tap is
    cropped, i.e. padding 0 in every dim.  Returns None when the shapes or
    ``conv_args`` needed for the integer index arithmetic are missing (the
    caller then reports ``min_work: not_representable``).
    """
    if not conv_args or input_shape is None or weight_shape is None or output_shape is None:
        return None
    dims = len(input_shape) - 2
    if dims < 1 or len(weight_shape) != dims + 2 or len(output_shape) != dims + 2:
        return None
    def per_dim(name: str) -> Optional[List[int]]:
        v = conv_args.get(name)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = [int(x) for x in v]
            return v if len(v) == dims else (v * dims if len(v) == 1 else None)
        return [int(v)] * dims
    stride, padding, dilation = per_dim("stride"), per_dim("padding"), per_dim("dilation")
    if stride is None or padding is None or dilation is None:
        return None
    pairs = 1
    for ax in range(dims):
        n_in, k, n_out = int(input_shape[2 + ax]), int(weight_shape[2 + ax]), int(output_shape[2 + ax])
        s, p, d = stride[ax], padding[ax], dilation[ax]
        count = 0
        for i in range(n_in):
            base = i * s - p
            for r in range(k):
                if 0 <= base + r * d < n_out:
                    count += 1
        pairs *= count
    return int(input_shape[0]) * int(input_shape[1]) * int(weight_shape[1]) * pairs
# ── __getitem__ index classification and embedding_bag metadata (issue #38 / an earlier decision) ──
_TENSOR_PLACEHOLDER_RE = re.compile(r"Tensor\(shape=\(([^)]*)\)(?:,\s*dtype=([\w.]+))?\)")


def _split_top_level(text: str) -> List[str]:
    """Split a bracketed component list on top-level commas."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _getitem_index_text(raw_attributes: Any) -> Optional[str]:
    """Return the rendered index argument of ``x[index]`` from torchview call attributes."""
    text = str(raw_attributes or "").strip()
    if not text:
        return None
    split = text.rfind("], {")
    positional = text[:split + 1] if split >= 0 else text
    # "[[<args>], {kwargs}]" -> keep the positional list "[<args>]" and unwrap it once
    if positional.startswith("[["):
        positional = positional[1:]
    if positional.startswith("[") and positional.endswith("]"):
        positional = positional[1:-1]
    comps = _split_top_level(positional)
    return comps[1] if len(comps) >= 2 else None


def classify_getitem_index(
    raw_attributes: Any,
    input_shape: Optional[List[int]],
    output_shape: Optional[List[int]],
    with_dtype: bool = False,
) -> Dict[str, Any]:
    """Classify a ``__getitem__`` from its recorded index argument.

    Returns ``{"kind": ..., "index_elements": int}`` (plus ``"index_dtype":
    str|None`` when ``with_dtype`` is set: the recorded dtype of a tensor
    index, None when absent or mixed) with kind one of ``basic`` (ints/slices/None/Ellipsis, or a 0-dim
    integer tensor: a zero-copy view), ``tensor`` (an integer/bool tensor
    index, including a 0-dim bool tensor, which materializes ``[1, *shape]``
    or ``[0, *shape]`` (index-2): a gather, the index numel taken from
    the rendering and its dtype reported), ``sequence`` (a Python list index:
    a gather with no index tensor read), ``unknown`` (rendering not
    understood: view accounting is kept and the flag is exported) or
    ``unavailable`` (no call attributes recorded).  An all-int list is a
    basic tuple of integer indices when the output has lost that many
    dims, and a list index when the output keeps the rank with that many
    rows; torchview renders nested lists beyond its depth as ``...``.
    """
    idx = _getitem_index_text(raw_attributes)
    if idx is None:
        return {"kind": "unavailable", "index_elements": 0, **({"index_dtype": None} if with_dtype else {})}
    kind, index_elements = "basic", 0
    index_dtypes: List[str] = []

    def tensor_numel(m: "re.Match") -> int:
        dims = [int(d) for d in m.group(1).split(",") if d.strip()]
        n = 1
        for d in dims:
            n *= d
        return n if dims else 1

    def is_scalar_view(m: "re.Match") -> bool:
        # a 0-dim INTEGER tensor selects like an int (view); a 0-dim BOOL
        # tensor is advanced indexing that materializes the output
        return m.group(1).strip() == "" and str(m.group(2) or "").replace("torch.", "") != "bool"

    def component_kind(c: str) -> str:
        c = c.strip()
        m = _TENSOR_PLACEHOLDER_RE.fullmatch(c)
        if m:
            return "basic" if is_scalar_view(m) else "tensor"
        if c.startswith("["):
            return "sequence"
        if c.startswith("slice(") or c in ("None", "Ellipsis") or re.fullmatch(r"-?\d+", c):
            return "basic"
        return "unknown"

    def result(kind: str, elements: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": kind, "index_elements": elements}
        if with_dtype:
            dts = sorted(set(index_dtypes))
            out["index_dtype"] = (dts[0] if len(dts) == 1 else None) if kind == "tensor" else None
        return out

    m = _TENSOR_PLACEHOLDER_RE.fullmatch(idx)
    if m:
        if is_scalar_view(m):
            return result("basic", 0)
        if m.group(2):
            index_dtypes.append(str(m.group(2)))
        return result("tensor", tensor_numel(m))
    if idx.startswith("[") and idx.endswith("]"):
        comps = _split_top_level(idx[1:-1])
        kinds = [component_kind(c) for c in comps]
        if comps and all(re.fullmatch(r"-?\d+", c.strip()) for c in comps):
            # tuple of ints (view) vs list index (gather): decide from the shapes
            if input_shape is not None and output_shape is not None:
                if list(output_shape) == [len(comps)] + list(input_shape)[1:]:
                    return result("sequence", 0)
                if list(output_shape) == list(input_shape)[len(comps):]:
                    return result("basic", 0)
            return result("unknown", 0)
        for c, k in zip(comps, kinds):
            if k == "tensor":
                kind = "tensor"
                tm = _TENSOR_PLACEHOLDER_RE.fullmatch(c.strip())
                index_elements += tensor_numel(tm)
                if tm.group(2):
                    index_dtypes.append(str(tm.group(2)))
            elif k == "sequence" and kind != "tensor":
                kind = "sequence"
            elif k == "unknown" and kind == "basic":
                kind = "unknown"
        return result(kind, index_elements)
    return result(component_kind(idx), 0)


def embedding_bag_metadata(raw_attributes: Any) -> Dict[str, Any]:
    """mode / offsets / per_sample_weights of an ``embedding_bag`` call from its attributes."""
    text = str(raw_attributes or "")
    out: Dict[str, Any] = {"mode": None, "offsets_elements": 0, "per_sample_weights_elements": 0, "available": bool(text)}
    m = re.search(r"mode:\s*'([a-z]+)'", text)
    if m:
        out["mode"] = m.group(1)
    for key in ("offsets", "per_sample_weights"):
        mm = re.search(key + r":\s*Tensor\(shape=\(([^)]*)\)", text)
        if mm:
            n = 1
            for d in [int(x) for x in mm.group(1).split(",") if x.strip()]:
                n *= d
            out[key + "_elements"] = n
    return out


class LoopCountEvidence(NamedTuple):
    """The ONE gate for a DSL kernel's loop-count evidence (an earlier review round; shared
    with the other-ops basis of issue #52 — import and reuse, do not
    re-derive).  ``present`` is False only for a layer that carries no
    ``loop_iterations`` KEY at all (a bodiless builtin such as ``gather``:
    nothing to judge); a key whose value is ``None`` is present and malformed
    (round 6).
    ``trusted`` requires all of: ``loop_iterations`` a positive INTEGER —
    ``numbers.Integral`` excluding ``bool``; a float (even an integral one such
    as 24.0), inf / nan, a string, None, zero, a negative value or any other
    type is ``loop_count_malformed:<repr>`` and is never truncated or coerced
    (shared-gate-type-8) —, ``loop_iterations_complete`` exactly True,
    and no ``loop_iterations_omitted``.  The helper never raises.  ``equation_faithful`` is reported, not gated
    on: the loop count is the witness precisely when the exported equation is
    NOT faithful (a gathered-slot kernel's equation is dense over the gathered
    axis; its proven count is the correction), so gating on it would revert
    that correction.  ``count`` is 0 whenever the evidence is not trusted."""
    present: bool
    count: int
    trusted: bool
    reasons: List[str]
    equation_faithful: Optional[bool]


def dsl_loop_count_evidence(layer: Dict[str, Any]) -> LoopCountEvidence:
    # ``present`` is about the KEY, not the value (round 6): a layer that carries
    # ``loop_iterations`` at all -- even as an explicit null -- has count
    # evidence to judge, and a null is malformed, never "absent".  Only a layer
    # without the key (and without omissions / an explicit incomplete flag),
    # such as a bodiless builtin, is unjudged and keeps the unmarked behaviour.
    has_count_key = "loop_iterations" in layer
    raw = layer.get("loop_iterations")
    omitted = list(layer.get("loop_iterations_omitted") or [])
    complete = layer.get("loop_iterations_complete")
    faithful = layer.get("equation_faithful")
    present = has_count_key or bool(omitted) or complete is False
    if not present:
        return LoopCountEvidence(False, 0, False, [], faithful if isinstance(faithful, bool) else None)
    reasons: List[str] = []
    count = 0
    if has_count_key:
        # A bool-excluded integral value > 0 is the ONLY accepted form.  Nothing
        # is coerced: 7.9 is not 7, 24.0 is not 24, "24" is not 24, inf / nan
        # are not numbers of trips.  ``repr`` is computed defensively so the
        # helper cannot raise on an exotic object.
        if isinstance(raw, numbers.Integral) and not isinstance(raw, bool) and raw > 0:
            count = int(raw)
        else:
            try:
                shown = repr(raw)
            except Exception:                                   # pragma: no cover - defensive
                shown = f"<unrepresentable {type(raw).__name__}>"
            reasons.append(f"loop_count_malformed:{shown}")
    if count > 0 and complete is True and not omitted:
        return LoopCountEvidence(True, int(count), True, [], faithful if isinstance(faithful, bool) else None)
    if omitted:
        reasons.extend(omitted)
    if not reasons:
        reasons.append("loop_iterations_complete not recorded" if complete is not True else "loop_iterations missing or not positive")
    return LoopCountEvidence(True, 0, False, reasons, faithful if isinstance(faithful, bool) else None)



def compute_dtype_of_operands(dtypes) -> Optional[str]:
    """The compute dtype a tensor-core contraction over operands recorded /
    declared with these logical dtypes is billed at: the highest-throughput
    quantised dtype among them -- ``"nvfp4"`` for any float4 (e2m1) operand,
    else ``"fp8"`` for any float8 operand -- or ``None`` (the workload
    precision, no bucket).  ONE rule for both frontend routes: the torch-capture
    route applies it to ``torch._scaled_mm``'s matrix operands, the DSL route to
    the tensors a MAC statement multiplies; the metadata override orders the
    same way (nvfp4 over fp8).  Unknown labels are not a quantised claim."""
    labels = [str(d or "").lower().replace("torch.", "") for d in (dtypes or [])]
    if any(d.startswith("float4_e2m1") or d in ("nvfp4", "fp4") for d in labels):
        return "nvfp4"
    if any(d.startswith("float8") or d == "fp8" for d in labels):
        return "fp8"
    return None


class EinsumGraphAnalyzer:
    """Analyze `einsum_graph.yaml` and write `analysis.yaml`."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)

    @staticmethod
    def _recorded_layout_reason(tensor_name: Optional[str], tensor_layouts: Dict[str, Any]) -> Optional[str]:
        """Reason string when the capture RECORDED ``tensor_name`` as contiguous, else None.

        Model inputs and parameters have no producer in the graph; the
        processor records their contiguity at capture (shape-matched, see
        ``PyTorchProcessor._collect_tensor_layouts``).  No record, or a
        recorded non-contiguous layout, is not a proof.
        """
        rec = tensor_layouts.get(tensor_name or "") if tensor_layouts else None
        if isinstance(rec, dict) and rec.get("contiguous") is True:
            return "contiguous layout recorded at capture"
        return None

    @staticmethod
    def _source_layout_established(layer_id: str, layers: Dict[str, Any], tensor_producers: Dict[str, str],
                                   memo: Dict[str, Optional[str]],
                                   tensor_layouts: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Return a reason string when ``layer_id``'s output is known to be
        contiguous from recorded evidence, else None (unknown: charged as copy).

        Evidence: a model input or parameter whose contiguity was recorded at
        capture; a fresh row-major allocation (``_FRESH_ROW_MAJOR_OPS``); a
        ``memory_format`` argument recorded in the call attributes
        (``_MEMORY_FORMAT_OPS``); a layout-preserving view of an established
        source; a layout-inheriting op whose data inputs are all established.
        Everything else (transposes, permutes, slices, expands, as_strided,
        unbind/chunk/split, einsum, ops without recorded arguments, unknown
        ops) is NOT established, so a dependent ``reshape_as`` is charged.
        """
        if layer_id in memo:
            return memo[layer_id]
        memo[layer_id] = None  # cycle guard
        layer = layers.get(layer_id) or {}
        op = str(layer.get("type", "")).lower()
        inputs = (layer.get("tensor_names") or {}).get("inputs") or []
        tensor_layouts = tensor_layouts or {}

        def source_status(index: int) -> Optional[str]:
            if index >= len(inputs):
                return None
            producer = tensor_producers.get(inputs[index])
            if producer is None:
                return EinsumGraphAnalyzer._recorded_layout_reason(inputs[index], tensor_layouts)
            return EinsumGraphAnalyzer._source_layout_established(producer, layers, tensor_producers, memo,
                                                                   tensor_layouts)

        reason: Optional[str] = None
        if op in _FRESH_ROW_MAJOR_OPS:
            # Fresh allocation only when the recorded call supplied no
            # destination: ``mm(a, b, out=dst)`` writes into ``dst`` and keeps
            # its strides (layout-3).  Without a recorded call the
            # allocation cannot be established.
            has_out = _has_destination_argument(layer)
            if has_out is False:
                reason = f"{op} allocates a fresh row-major output (no destination argument recorded)"
            else:
                reason = None
        elif op in _LAYOUT_PRESERVING_VIEWS:
            if op in LAYOUT_DEPENDENT_ALIAS_OPS and _reshape_as_receiver_present(layer) is not True:
                # an unresolved receiver: this node's output layout is UNKNOWN
                # for every downstream proof; never recurse through it to the
                # shape-reference operand that survived (input-4b)
                reason = None
            else:
                src = source_status(0)
                reason = f"{op} preserves layout of its source ({src})" if src else None
        elif op in _MEMORY_FORMAT_OPS:
            fmt, recorded = _memory_format_evidence(layer)
            if not recorded:
                reason = None                      # no recorded call arguments: layout unknown
            elif fmt == "contiguous_format":
                reason = f"{op}(memory_format=contiguous_format)"
            elif fmt not in ("default", "preserve_format"):
                reason = None                      # channels_last & co.: not row-major
            elif op == "contiguous":
                reason = "contiguous() without memory_format"
            else:
                # clone / dtype casts with the default or preserve_format: same layout as the source
                src = source_status(0)
                if src:
                    reason = f"{op} preserves the layout of its contiguous source ({src})"
                elif op == "clone" and inputs:
                    producer = tensor_producers.get(inputs[0])
                    if producer and _is_overlapping_expand(layers.get(producer) or {}):
                        reason = "clone of an overlapping expand result allocates a contiguous tensor"
        elif op in _LAYOUT_INHERITING_OPS:
            # Inheritance is a proof only from a recorded call without a
            # destination: ``add(x, x, out=dst)`` writes into ``dst`` and keeps
            # its strides whatever the inputs' layouts (layout-4); without
            # a recorded call the destination cannot be excluded.
            if _has_destination_argument(layer) is False:
                statuses = [source_status(i) for i in range(len(inputs))]
                if inputs and all(statuses):
                    reason = f"{op} inherits the layout of contiguous inputs (no destination argument recorded)"
        memo[layer_id] = reason
        return reason

    def analyze_graph(
        self,
        einsum_graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
        boundary_storage_dtypes: Optional[Dict[str, str]] = None,
        declared_output_storage: Optional[List[Dict[str, Any]]] = None,
        boundary_input_binding: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze an einsum graph and write `analysis.yaml`.

        Args:
            einsum_graph_path: Path to `einsum_graph.yaml`.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the einsum graph into output dir using the
                canonical name `einsum_graph.yaml`.
            boundary_storage_dtypes: Authoritative STORAGE dtypes for boundary
                (external input/output) tensors, keyed by tensor name —
                normally the benchmark definition's declared input/output
                dtypes.  The single per-graph ``precision`` width is a
                compute/representation choice and must not set boundary
                traffic (campaign-audit defects 2/3: BF16 public I/O billed
                1 byte under an fp8 override; FP32 I/O billed 2 bytes under
                fp16).  When given, every boundary tensor must resolve to a
                width — from this mapping, through an explicit
                ``boundary_input_binding``, or from the graph's recorded
                ``tensor_dtypes`` trace evidence for inputs the graph's
                ``tensor_origins`` map records as internal constants —
                otherwise ValueError: storage types are never silently
                invented.  A PUBLIC external input whose authoritative
                binding cannot be established REFUSES (campaign reviews
                C12-P1/C19-P1): its verification/trace dtype is not
                public storage evidence merely because an alias lacks a
                direct definition-name match, and a name spelling such as
                ``__input_0`` is never provenance — it is a valid Python
                identifier a user may register as a public input.
            boundary_input_binding: Explicit trace-to-definition input
                mapping {graph tensor name -> definition input name}, for
                graphs whose public input tensors carry aliases (e.g.
                positional ``x_0``) instead of the definition's names.
                A supplied binding is AUTHORITATIVE for its input and is
                honored before direct-name matching (a graph alias may
                collide with another declared input's name — review
                C19-P2); a binding targeting a name absent from
                ``boundary_storage_dtypes`` is dangling and refuses,
                regardless of the tensor's name, recorded origin, or any
                direct-name match.  Inputs without a binding entry keep
                plain direct-name resolution.
                Internal-constant provenance stays distinct: tensors the
                graph's ``tensor_origins`` map (propagated from
                ``einsum_lib.metadata.build_metadata``'s actual trace
                provenance) records as ``internal-constant`` bill at
                their recorded trace dtype.  Resolution provenance is
                recorded in
                ``metadata.boundary_storage`` and declared byte totals in
                ``total.model_io_bytes_declared`` (raw per-op basis) and
                ``total.external_io_bytes_declared_dedup`` (deduplicated,
                the fused-model basis).

        Returns:
            Analysis dict, or None on failure.
        """
        graph_path = Path(einsum_graph_path)   # issue #63: a distinct local -- `src` is reused below for per-layer sources and returned outputs
        # Stage 2.5 fallback: prefer einsum_graph_reordered.yaml if present in the same dir
        reordered = graph_path.parent / "einsum_graph_reordered.yaml"
        if graph_path.name == "einsum_graph.yaml" and reordered.exists():
            if self.debug:
                logger.debug(f"using reordered graph {reordered}")
            graph_path = reordered
        out_dir = ensure_directory(output_dir)

        if not graph_path.exists():
            if self.debug:
                logger.debug(f"einsum graph not found: {graph_path}")
            return None

        try:
            with open(graph_path) as f:
                graph = yaml_safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                logger.debug(f"failed reading einsum graph: {exc}")
            return None
        # Contiguity recorded at capture for model inputs / parameters (an earlier decision round 3)
        tensor_layouts: Dict[str, Any] = dict(graph.get("tensor_layouts") or {})

        if copy_graph:
            try:
                dst = out_dir / "einsum_graph.yaml"
                if graph_path.resolve() != dst.resolve():
                    dst.write_text(graph_path.read_text())
            except Exception:
                if self.debug:
                    logger.debug("failed to copy einsum_graph.yaml")

        all_layers: Dict[str, Any] = graph.get("layers") or {}
        # Layers feeding the model's declared outputs (written by the
        # converter from torchview's ``output-tensor`` nodes). Older graphs
        # lack the key; then every consumer-less op is treated as an output.
        model_output_ops: Set[str] = set(graph.get("model_output_ops") or [])
        # Declared outputs with shape/dtype; any whose producer is not
        # charged as an external write below (untraced in-place producer,
        # or an intermediate that is also returned) is added explicitly.
        model_outputs: List[Dict[str, Any]] = list(graph.get("model_outputs") or [])
        declared_output_tensors: Set[str] = {
            str(mo["tensor"]) for mo in model_outputs if mo.get("tensor")
        }
        # A returned model INPUT (is_input, recorded provenance) stays an
        # external read for its consumers; only producer-less declared
        # outputs that are not inputs are treated as in-model buffers.
        orphan_buffer_tensors: Set[str] = {
            str(mo["tensor"]) for mo in model_outputs
            if mo.get("tensor") and not mo.get("is_input")
        }
        capture_capability = graph.get("capture_capability")
        from solar.graph.torchview_capability import degraded_marker
        capture_degraded = degraded_marker(capture_capability) if capture_capability else None
        element_size = BYTES_PER_ELEMENT.get(precision, 4)

        # Override precision/element_size from quant metadata if available
        quant_precision = self._resolve_quant_precision(graph_path)
        if quant_precision:
            element_size = BYTES_PER_ELEMENT.get(quant_precision, element_size)
            precision = quant_precision
            if self.debug:
                logger.debug(f"Quant override: precision={precision}, bytes_per_element={element_size}")

        # Filter out "start" nodes - they represent model inputs, not computation
        # Keep track of start node IDs for reference
        _BOOL_DTYPES = {"torch.bool", "bool"}
        start_node_ids: Set[str] = set()
        bool_start_node_ids: Set[str] = set()
        layers_in: Dict[str, Any] = {}

        for layer_id, layer in all_layers.items():
            op_type = str(layer.get("type", "")).lower()
            if op_type == "start":
                start_node_ids.add(layer_id)
                # Detect bool-typed start nodes from tensor_dtypes
                out_dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
                if out_dtypes and all(str(d) in _BOOL_DTYPES for d in out_dtypes):
                    bool_start_node_ids.add(layer_id)
            else:
                layers_in[layer_id] = layer

        if self.debug:
            logger.debug(f"Filtered out {len(start_node_ids)} start nodes")
            if bool_start_node_ids:
                logger.debug(f"Found {len(bool_start_node_ids)} bool-typed start nodes: {bool_start_node_ids}")
            logger.debug(f"Analyzing {len(layers_in)} computation nodes")

        # Full element count of every start-node output tensor, keyed by
        # tensor name.  Used as the authoritative base size when capping
        # deduplicated external reads of a model input.
        start_output_sizes: Dict[str, int] = {}
        for sid in start_node_ids:
            s_names = (all_layers[sid].get("tensor_names") or {}).get("outputs") or []
            s_shapes = (all_layers[sid].get("tensor_shapes") or {}).get("outputs") or []
            for s_name, s_shape in zip(s_names, s_shapes):
                if s_name and isinstance(s_shape, list):
                    start_output_sizes[str(s_name)] = _product(s_shape)

        # Build tensor producer/consumer maps using tensor_names from the
        # einsum graph.  A tensor is intermediate if it is produced by one op
        # AND consumed by another op.
        all_layer_ids: Set[str] = set(layers_in.keys())

        # Zero-copy view layers are transparent for memory accounting.  The
        # fused model should see through them when deciding whether a tensor is
        # graph-internal or external.
        _TRANSPARENT_OPS = {
            "expand", "expand_as",
            "view", "reshape", "contiguous",
            "transpose", "permute", "t",
            "unsqueeze", "squeeze", "flatten",
            "unfold", "unflatten",
            "chunk", "split", "tensor_split",
        }
        transparent_layer_ids: Set[str] = set()
        for layer_id, layer in layers_in.items():
            if str(layer.get("type", "")).lower() in _TRANSPARENT_OPS:
                transparent_layer_ids.add(layer_id)

        tensor_producers: Dict[str, str] = {}   # tensor_name -> producer_layer_id
        layout_memo: Dict[str, Optional[str]] = {}  # layer_id -> layout reason (reshape_as decisions)
        tensor_consumers: Dict[str, Set[str]] = {}  # tensor_name -> set of consumer_layer_ids

        # Pass 1: gather all produced tensor names (order-independent).
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
            for oname in (t_names.get("outputs") or []):
                tensor_producers[oname] = layer_id

        # Pass 2: gather consumers for tensors produced somewhere in graph.
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
            for iname in (t_names.get("inputs") or []):
                if iname in tensor_producers:
                    tensor_consumers.setdefault(iname, set()).add(layer_id)

        # Identify intermediate tensors: produced by one op AND consumed by another
        intermediate_tensors: Set[str] = set()
        for tensor_name in tensor_producers:
            if tensor_name in tensor_consumers and len(tensor_consumers[tensor_name]) > 0:
                intermediate_tensors.add(tensor_name)

        def _trace_source_through_views(layer_id: str) -> str:
            """Trace backward through transparent view layers to the real source."""
            visited: Set[str] = set()
            current = layer_id
            while current in transparent_layer_ids and current not in visited:
                visited.add(current)
                conns = (layers_in[current].get("connections") or {}).get("inputs") or []
                if not conns:
                    break
                current = conns[0]
            return current

        def _has_real_consumer(layer_id: str) -> bool:
            """Return true if output reaches a non-transparent graph layer."""
            visited: Set[str] = set()
            queue = [layer_id]
            while queue:
                lid = queue.pop(0)
                if lid in visited:
                    continue
                visited.add(lid)
                conns = (layers_in.get(lid, {}).get("connections") or {}).get("outputs") or []
                for out_id in conns:
                    if out_id in transparent_layer_ids:
                        queue.append(out_id)
                    elif out_id in all_layer_ids:
                        return True
            return False

        def _reaches_model_output(layer_id: str) -> bool:
            """True if the layer (or a view chain from it) is a declared model output."""
            visited: Set[str] = set()
            queue = [layer_id]
            while queue:
                lid = queue.pop(0)
                if lid in visited:
                    continue
                visited.add(lid)
                if lid in model_output_ops:
                    return True
                conns = (layers_in.get(lid, {}).get("connections") or {}).get("outputs") or []
                queue.extend(out_id for out_id in conns if out_id in transparent_layer_ids)
            return False

        # Lazy casts/copies feeding gathers: ``cache.to(fp32)[pages]`` in
        # the reference code converts the whole KV cache, but a kernel only
        # touches the gathered pages. When every real consumer of a unary
        # elementwise op (through views) is a slice/gather, cap the op's
        # read of its input at the total gathered size.
        lazy_read_cap: Dict[str, int] = {}
        for lid, layer in layers_in.items():
            if str(layer.get("type", "")).lower() not in _LAZY_UNARY_OPS:
                continue
            total = 0
            found = False
            only_gathers = True
            seen: Set[str] = set()
            queue = [lid]
            while queue and only_gathers:
                cur = queue.pop()
                for oname in (layers_in[cur].get("tensor_names") or {}).get("outputs") or []:
                    for cid in tensor_consumers.get(oname) or ():
                        if cid in seen:
                            continue
                        seen.add(cid)
                        clayer = layers_in.get(cid) or {}
                        ctype = str(clayer.get("type", "")).lower()
                        if ctype in _GATHER_CONSUMER_OPS:
                            found = True
                            for shp in (clayer.get("tensor_shapes") or {}).get("outputs") or []:
                                total += _product(shp) if isinstance(shp, list) else 0
                        elif cid in transparent_layer_ids:
                            queue.append(cid)
                        else:
                            only_gathers = False
                            break
            if found and only_gathers:
                lazy_read_cap[lid] = total
        # Structured sparsity: triangular / masked operands let a kernel skip
        # MACs (Mamba chunk scan, causal attention). Fraction per einsum layer.
        mac_fraction = _structured_sparsity(layers_in, tensor_producers, tensor_consumers)
        write_caps = _output_write_caps(layers_in, tensor_producers)
        if self.debug and mac_fraction:
            logger.debug(f"structured-sparsity MAC fractions: {mac_fraction}")
        external_output_layers: Set[str] = set()
        external_output_written: Set[str] = set()   # ... with a non-zero DRAM write

        if self.debug:
            logger.debug(f"Found {len(intermediate_tensors)} intermediate tensors")
            for t in sorted(intermediate_tensors)[:10]:
                logger.debug(f"- {t}")
            if transparent_layer_ids:
                logger.debug(f"{len(transparent_layer_ids)} transparent view layers")

        # TEMPORARY FIX: Propagate bool-ness from start nodes through graph.
        # A computation layer is "bool" if ALL its inputs come from bool
        # sources (bool start nodes or other bool layers).
        _bool_layers: Set[str] = set()
        if bool_start_node_ids:
            # Process layers in topological-ish order (inputs before outputs)
            # by iterating until no more layers are added.
            changed = True
            while changed:
                changed = False
                for layer_id, layer in layers_in.items():
                    if layer_id in _bool_layers:
                        continue
                    conns = layer.get("connections") or {}
                    inp_ids = list(conns.get("inputs") or [])
                    if not inp_ids:
                        continue
                    # All inputs must be bool (start or layer)
                    if all(
                        inp in bool_start_node_ids or inp in _bool_layers
                        for inp in inp_ids
                    ):
                        _bool_layers.add(layer_id)
                        changed = True

            if self.debug and _bool_layers:
                logger.debug(f"Skipping memory for {len(_bool_layers)} bool-derived layers: {sorted(_bool_layers)}")

        # Detect orphaned subgraphs: chains rooted at tensors created
        # outside RecorderTensor tracking (e.g. torch.zeros() with no
        # subclass args).  The source-tracing code (Step 4 below)
        # classifies inputs from non-existent nodes as external DRAM I/O.
        # For dead-end layers whose input traces to a genuinely orphaned
        # source, this phantom traffic should be zeroed.
        #
        # A layer's input is "orphaned" when the source traces to a
        # non-existent ID AND the layer produces no live output.
        # For scatter/setitem ops, if the TARGET (first input) traces to
        # an orphaned source AND the output is dead-end, the write is
        # phantom too.
        #
        # Pre-compute which layers are dead ends (no live output path).
        _dead_end_layers: Set[str] = set()
        for layer_id in layers_in:
            if not _has_real_consumer(layer_id):
                _dead_end_layers.add(layer_id)

        # Track which layers are flagged as orphaned (for metadata).
        _orphaned_layers: Set[str] = set()

        layers_out: Dict[str, Any] = {}
        total_macs = 0          # tensor-core MACs (matmul, conv)
        total_min_work_macs = 0  # macs with transposed convs at their minimum-work bound (issue #44)
        min_work_not_representable = 0
        macs_by_compute_dtype: Dict[str, int] = {}   # nvfp4 / fp8 MACs (recorded operand dtypes)
        total_macs_dense_binary_schedule = 0   # dense; multi-operand einsums at their cheapest binary schedule
        total_scalar_multiplies_binary_schedule = 0
        macs_incomplete_layers: List[str] = []
        operand_evidence_warning_layers: List[str] = []
        multi_operand_einsum_layers: List[str] = []
        # issue #58 (report provenance): typed MAC components beside the aggregate -- tensor-core CONTRACTION MACs by
        # compute dtype (the workload-precision bucket is what the perf model prices at the default rate) vs SCALAR
        # MULTIPLY STATEMENTS the per-statement classifier counts as MACs in non-einsum kernels -- and the layers whose
        # exported equation is NOT faithful to the kernel (their binary schedule is descriptive only).  Every recorded
        # count is preserved: the components sum to ``macs``; nothing is re-priced here.
        contraction_macs_by_dtype: Dict[str, int] = {}
        scalar_statement_mac_layers: Dict[str, int] = {}
        non_contraction_other_mac_layers: Dict[str, int] = {}
        equation_unfaithful_layers: List[str] = []
        total_flops = 0         # 2 * total_macs
        total_other_ops = 0     # CUDA-core elementwise/reduction ops
        # issue #52: DSL kernel layers grouped by the basis of their count
        # (only emitted when the graph has DSL kernels)
        dsl_cost_basis_layers: Dict[str, List[str]] = {}
        total_unfused_elems = 0       # Σ (all input + output elems) per op
        total_intermediate_elems = 0  # Σ intermediate activation elems
        total_weight_elems = 0        # Σ parameter numel of weight-typed external inputs per op
        total_weight_read_elems = 0   # Σ DRAM read elements of those inputs per op (traffic)
        total_weight_unresolved_elems = 0            # Σ numel of weight-typed slots with unknown provenance
        weight_provenance_unresolved_layers: List[str] = []

        # Deduplicated external (non-intermediate) tensor tracking for the
        # fused / fused_prefetched model.  Fused semantics: each external
        # byte is read from DRAM at most once, so per base tensor the DRAM
        # traffic is the *unique element footprint* of all its reads:
        #   - fan-out (full tensor read by several ops) counts once;
        #   - disjoint slices of a stacked tensor (per-layer KV cache) sum;
        #   - repeated or overlapping slices count each element once.
        # Reads are grouped by canonical base tensor (traced through
        # transparent views).  Each read contributes either a statically
        # proven access box (see solar.analysis.access_regions) whose union
        # is computed exactly, or — when the region cannot be proven — a
        # bare element count that participates via max(), the conservative
        # lower-bound accounting.  Group footprints are capped at the base
        # tensor size, so the result never exceeds the true footprint and
        # the SOL lower-bound property is preserved.
        # Per canonical base tensor: proven boxes, unproven read counts,
        # and full-size candidates observed at direct-read sites.
        external_read_groups: Dict[str, Dict[str, Any]] = {}
        unique_external_outputs: Dict[str, int] = {}
        unique_external_output_bpe: Dict[str, float] = {}
        # NONRETURNED-OUTPUT-BILLING-1 (issue #59): consumer-less results that the
        # program does NOT return are internal stores (removable under fusion);
        # recorded here separately, never as boundary traffic
        nonreturned_store_bytes = 0.0
        nonreturned_store_layers: List[str] = []
        total_unfused_bytes = 0.0
        total_intermediate_bytes = 0.0
        # Per-name max-dedup external INPUT elements: basis of boundary
        # storage billing (external_io_bytes_declared_dedup) and binding
        # membership — kept alongside the canonical-base read groups, which
        # serve the fused byte model, not boundary billing.
        unique_external_inputs: Dict[str, int] = {}
        weight_boundary_names: set = set()   # external input names whose reads are weight-classified (an earlier decision)
        weight_boundary_footprint: Dict[str, int] = {}   # per such name: parameter FOOTPRINT elements counted (distinct source per consuming op x numel)
        # Raw (per-op, fan-out counted) external element sums by name: the
        # basis of total.model_io_elements, needed to bill boundary storage
        # at declared widths on the same basis.
        raw_external_input_elems: Dict[str, int] = {}
        raw_external_output_elems: Dict[str, int] = {}
        # Public boundary input name -> canonical base tensor of its read
        # group.  Boundary billing iterates public NAMES (view aliases
        # included) while read groups are keyed by canonical BASE; the
        # width correction must land on the group the alias resolves to,
        # exactly once per group (review C03-P1).  "" marks a name whose
        # canonical resolution conflicted across reads — refused at
        # billing, never guessed.
        external_input_canonical: Dict[str, str] = {}
        # DSL representation dtypes recorded in the graph (trace evidence),
        # by tensor name — fallback width source for DSL-internal boundary
        # tensors that no definition declares.
        graph_tensor_dtypes: Dict[str, str] = {}
        for _layer in all_layers.values():
            _names = _layer.get("tensor_names") or {}
            _dtypes = _layer.get("tensor_dtypes") or {}
            for io in ("inputs", "outputs"):
                for n, d in zip(_names.get(io) or [], _dtypes.get(io) or []):
                    if n and d:
                        graph_tensor_dtypes.setdefault(str(n), str(d))
        # Recorded trace provenance by tensor name (exporter
        # ``tensor_origins``: registered-input vs internal-constant).  The
        # ONLY admissible internal-constant evidence — a ``__<role>_<n>``
        # name spelling is a valid public identifier, never provenance
        # (review C19-P1).
        graph_tensor_origins: Dict[str, str] = {
            str(k): str(v)
            for k, v in (graph.get("tensor_origins") or {}).items()}

        for layer_id, layer in layers_in.items():
            op_type = str(layer.get("type", "unknown"))
            equation = str(layer.get("einsum_equation", "") or "")
            if "is_real_einsum" not in layer:
                raise ValueError(
                    f"Layer '{layer_id}' (type={op_type}) is missing 'is_real_einsum' field. "
                    f"All layers in the einsum graph must specify is_real_einsum: true/false."
                )
            is_real_einsum = bool(layer["is_real_einsum"])
            tensor_shapes: Dict[str, Any] = layer.get("tensor_shapes") or {}
            tensor_types: Dict[str, Any] = layer.get("tensor_types") or {}
            tensor_names: Dict[str, Any] = layer.get("tensor_names") or {}
            connections: Dict[str, Any] = layer.get("connections") or {}
            input_layer_ids = list(connections.get("inputs") or [])
            output_layer_ids = list(connections.get("outputs") or [])

            ts = TensorShapes(
                inputs=tensor_shapes.get("inputs", []),
                outputs=tensor_shapes.get("outputs", []),
            )

            # Operand evidence (F6 / R5): can the equation's operands be bound to
            # the recorded shapes?  For an explicit torch.einsum the saved
            # tensor_shapes may be incomplete; the recorded call (raw_attributes)
            # restores the positional shapes when the saved ones align to it.
            # Unresolved evidence on an explicit einsum marks the layer's MACs
            # incomplete (0, flagged) instead of pricing a misbound product; on
            # handler-emitted equations the problems are recorded as warnings
            # and the cost is unchanged.
            operand_evidence: Dict[str, Any] = {}
            macs_incomplete = False
            # torch._scaled_mm: only the two matrix operands are einsum operands;
            # the scale tensors (and an optional bias) are extra reads that the
            # contraction does not index.  The equation is validated and priced
            # over the matrix operands; the handler needs the recorded dtypes
            # to unpack a float4_e2m1fn_x2 reduction extent.
            _is_scaled_mm = "scaled_mm" in str(op_type).lower()
            _in_dtypes_rec = list((layer.get("tensor_dtypes") or {}).get("inputs") or [])
            _cost_kwargs: Dict[str, Any] = {"input_dtypes": _in_dtypes_rec} if _is_scaled_mm else {}

            def _eq_ts() -> TensorShapes:
                # evaluated at each use: ``ts`` may be replaced below by the
                # operands restored from raw_attributes (explicit torch.einsum)
                return TensorShapes(inputs=list(ts.inputs[:2]), outputs=list(ts.outputs)) if _is_scaled_mm else ts

            macs_incomplete_reason = "einsum_operand_evidence_unresolved"
            scaled_mm_rejection: Optional[str] = None
            if _is_scaled_mm:
                # a scaled_mm whose handler refused at conversion (recorded on the
                # layer) or that reached the analyzer without an equation is
                # incomplete: its MACs are NOT the stored-shape product
                # (evidence-1)
                rej = layer.get("scaled_mm_rejection")
                if rej or not equation or not is_real_einsum:
                    scaled_mm_rejection = str(rej) if rej else "no handler equation recorded for the scaled_mm layer"
                    macs_incomplete = True
                    macs_incomplete_reason = "scaled_mm_evidence_unresolved"
            # Issue #50: an op the converter could not route (no handler, no
            # recognised family) is recorded by name; its work is excluded,
            # never estimated from the stored shapes (0 MACs, 0 other_ops).
            unrecognised_op = layer.get("unrecognised_op")
            if unrecognised_op:
                macs_incomplete = True
                macs_incomplete_reason = "op_unrecognised"
            if is_real_einsum and equation and not macs_incomplete:
                problems = validate_equation_operands(equation, _eq_ts())
                if problems and op_type == "einsum":
                    restored = _einsum_operand_shapes_from_raw(layer, equation, ts)
                    if restored is not None:
                        operand_evidence["operand_shapes_source"] = (
                            f"raw_attributes (positional; saved tensor_shapes carried "
                            f"{len(ts.inputs)} of {len(restored)} operands)")
                        operand_evidence["operand_shapes"] = restored
                        ts = TensorShapes(inputs=restored, outputs=ts.outputs)
                        problems = validate_equation_operands(equation, ts)
                if problems:
                    operand_evidence["operand_evidence_problems"] = problems
                    if op_type == "einsum":
                        macs_incomplete = True

            ops_cost = 0
            if _is_scaled_mm and not macs_incomplete:
                # the scaled_mm handler is the only MAC source for the op: a
                # refusal (unrecorded / mixed dtypes, inconsistent shapes) is an
                # incomplete layer, never the generic equation product
                try:
                    _smm_op = self.einsum_analyzer.get_einsum_op(op_type, _eq_ts(), **_cost_kwargs)
                    ops_cost = int(_smm_op.get_compute_cost(_eq_ts()))
                except Exception as exc:
                    scaled_mm_rejection = str(exc)
                    macs_incomplete = True
                    macs_incomplete_reason = "scaled_mm_evidence_unresolved"
                    ops_cost = 0
            # DSL kernels (einsum_lib route): the cost basis is the kernel's own
            # loop nest as the converter counted it (``loop_iterations``), then
            # the exported equation, then the operand shapes — the an earlier revision
            # precedence the lineage port dropped together with the measured-
            # sparsity discount that scales it (TARGET-DENSITY-GUARD-1).
            # Equation-derived counts inflate multiplicatively when the exported
            # equation misrepresents the loops (derived indices like k//128 or
            # n*D+d become independent dims; fused residual/scale inputs widen
            # the index union); a kernel that visits only gathered slots is
            # billed for those slots and not re-discounted.
            # The loop count is used ONLY when the converter PROVED the loop nest
            # complete (``loop_iterations_complete: true``: every executed loop
            # is a sized Dim() loop).  A positive count whose completeness is
            # not proven — a range()/while loop the parser could not represent,
            # the standalone-index fallback, a legacy artifact without the
            # proof, or a malformed value — is NOT trusted: the layer is billed
            # by the equation (then the shapes) and marked
            # ``loop_metadata_incomplete`` so the INTEG-9 contract carries it
            # (loop-undercount-1).  Never the maximum of the two bases:
            # that would undo the gathered-slot correction (8,192 -> 1,024).
            # A kernel without any count but with recorded omissions (its only
            # loop is a ``while``: loop-marker-5) is incomplete in the
            # same way; a bodiless builtin carries neither field and is not
            # marked.
            # ONE gate (``dsl_loop_count_evidence``, an earlier review round) decides whether
            # the count is trusted; the other-ops basis (issue #52)
            # consumes the same helper — the count is selected ONCE here,
            # density-discounted ONCE below, and only then split into MACs /
            # other_ops (loop-evidence-1, density-composition-1); the
            # selected basis travels with it as ``cost_basis``.
            _loop_iters = 0
            _loop_meta_incomplete = False
            _loop_omitted: list = []
            if op_type == "kernel":
                _loop_ev = dsl_loop_count_evidence(layer)
                if _loop_ev.present:
                    if _loop_ev.trusted:
                        _loop_iters = _loop_ev.count
                    else:
                        _loop_meta_incomplete = True
                        _loop_omitted = list(_loop_ev.reasons)
            # issue #52: the basis the kernel's count was SELECTED from
            # (loop_iterations | equation_product | shape_generic |
            # incomplete_evidence); None for mechanical-route layers, which gain
            # no receipt field
            cost_basis: Optional[str] = None
            # issue #59: "per_statement" when MACs / other_ops come from the
            # converter's per-statement class sums (loop_mac_statements /
            # loop_other_statements), else None (whole-kernel flag or handler)
            unit_basis: Optional[str] = None
            # legacy-token-16: the exception type when judging a DSL
            # kernel's admission evidence raised; such a layer is incomplete.
            _admission_error: Optional[str] = None
            _dsl_route_layer = op_type == "kernel"
            try:
                if macs_incomplete or _is_scaled_mm:
                    if op_type == "kernel":
                        cost_basis = "incomplete_evidence"
                elif op_type == "kernel" and _loop_iters > 0:
                    ops_cost = _loop_iters
                    cost_basis = "loop_iterations"
                elif op_type == "kernel" and equation and is_real_einsum:
                    ops_cost = _macs_from_equation(equation, tensor_shapes.get("inputs", []))
                    cost_basis = "equation_product"
                elif is_real_einsum and equation:
                    ops_cost = int(
                        self.einsum_analyzer.get_compute_cost(op_type, _eq_ts(), equation=equation, **_cost_kwargs)
                    )
                elif op_type == "einsum" and equation:
                    # Explicit torch.einsum: the handler needs the equation to
                    # cost the (possibly multi-operand) contraction chain; the
                    # shapes alone cannot tell which indices are contracted.
                    ops_cost = int(self.einsum_analyzer.get_compute_cost(
                        op_type, ts, equation=equation
                    ))
                elif op_type == "kernel":
                    # A DSL node with no trusted loop count and no real-einsum
                    # equation.  The generic shape fallback is admitted ONLY by
                    # explicit DSL provenance, never by the type name
                    # (kernel-origin-10):
                    #   * exporter marker `dsl_route: true` with `kernel_body`
                    #     (a parsed @kernel body / legacy loop_expr: the target's
                    #     generic cost + the loop_metadata_incomplete marker above)
                    #     or a resolved `builtin` from the DSL builtin table;
                    #   * a graph exported before the marker existed is recognised
                    #     by fields only the DSL exporter writes (loop_expr /
                    #     loop_iterations* / equation_faithful) and treated as a
                    #     kernel body (the target's accounting for saved graphs).
                    # Anything else -- a torch-route record that merely bears the
                    # name "kernel" (the converter marks it unrecognised_op before
                    # it gets here) or a bodiless record with no resolved builtin --
                    # is reported incomplete (dsl_kernel_unresolved), never priced.
                    from solar.einsum_dsl.builtin_table import DSL_TENSOR_BUILTINS as _dsl_table
                    _dsl = layer.get("dsl_route") is True
                    _legacy_dsl = (not _dsl) and any(k in layer for k in (
                        "loop_expr", "loop_iterations", "loop_iterations_complete",
                        "loop_iterations_omitted", "equation_faithful"))
                    _dsl_builtin = layer.get("builtin") if _dsl else None
                    if _dsl:
                        _dsl_body = bool(layer.get("kernel_body"))
                    elif _legacy_dsl:
                        # A pre-marker record identifies the DSL route by its
                        # fields, but the old exporter wrote ``loop_expr: null``
                        # for unresolved bodiless kernels too, so field PRESENCE is
                        # not body evidence (legacy-origin-12).  Admit the
                        # generic cost only on POSITIVE evidence: a non-null
                        # loop_expr; the ops the old exporter INFERRED from a
                        # body (`_infer_ops` yields a reduction_op string --
                        # "add"/"sub"/"none" -- only when a loop_expr existed,
                        # and (None, None) for a bodiless kernel); a positive
                        # numeric loop_iterations the converter proved complete;
                        # or a resolved table builtin.
                        # Typed, parse-validated predicates (legacy-schema-14):
                        # truthy non-strings, empty strings and unknown op tokens
                        # are NOT evidence.  The predicates come from the exporter
                        # itself (its parser and its body-inference vocabulary).
                        # The predicates are total over artifact value types; should
                        # one still raise, admission FAILS CLOSED below with the
                        # exception type recorded (legacy-token-16) -- never a
                        # complete-looking zero.
                        from solar.einsum_dsl import einsum_converter as _dsl_exporter
                        try:
                            _li = layer.get("loop_iterations")
                            _li_ok = (isinstance(_li, int) and not isinstance(_li, bool) and _li > 0
                                      and layer.get("loop_iterations_complete") is True)
                            _legacy_builtin = layer.get("builtin")
                            if isinstance(_legacy_builtin, str) and _legacy_builtin in _dsl_table:
                                _dsl_builtin = _legacy_builtin
                            _dsl_body = (_dsl_exporter.loop_expr_is_body(layer.get("loop_expr"))
                                         or _dsl_exporter.ops_prove_body(layer.get("elementwise_op"),
                                                                         layer.get("reduction_op"))
                                         or _li_ok)
                        except MemoryError:  # resource exhaustion propagates (legacy-memory-17, the earlier change contract)
                            raise
                        except Exception as _adm_exc:  # fail closed, record the type
                            _dsl_body = False
                            _dsl_builtin = None
                            _admission_error = type(_adm_exc).__name__
                    else:
                        _dsl_body = False
                    if _admission_error is None and _dsl_builtin == "gather":
                        # TARGET-GATHER-COST-1: a built-in gather is a re-indexing,
                        # one address op per gathered (output) element; never a
                        # dense product of the operand extents.  Traffic is routed
                        # to the gather rule below (index + gathered rows + output).
                        ops_cost = int(sum(math.prod(int(d) for d in shp) for shp in (tensor_shapes.get("outputs") or []) if isinstance(shp, list)))
                        cost_basis = "gather_elements"
                    elif _admission_error is None and (_dsl_body or (_dsl_builtin is not None and _dsl_builtin in _dsl_table)):
                        ops_cost = int(self.einsum_analyzer.get_compute_cost(
                            op_type, ts, raw_attributes=layer.get("raw_attributes"),
                            dsl_builtin=_dsl_builtin, dsl_kernel_body=_dsl_body))
                        cost_basis = "shape_generic"
                    else:
                        ops_cost = 0
                        macs_incomplete = True
                        macs_incomplete_reason = ("dsl_kernel_unresolved_legacy" if _legacy_dsl
                                                  else "dsl_kernel_unresolved")
                        cost_basis = "incomplete_evidence"
                else:
                    # Recurrent handlers read the recorded call attributes
                    # (num_layers, bidirectional, batch_first) when present.
                    ops_cost = int(self.einsum_analyzer.get_compute_cost(
                        op_type, ts, raw_attributes=layer.get("raw_attributes")))
                    if op_type == "kernel":
                        cost_basis = "shape_generic"
            except MemoryError:  # resource exhaustion is never turned into a zero (legacy-memory-17)
                raise
            except Exception as _cost_exc:
                ops_cost = 0
                if _dsl_route_layer:
                    # A DSL-route kernel whose cost evaluation raised is NOT a
                    # complete zero: mark it incomplete with the exception type
                    # (legacy-token-16 (b)); the status propagates to the
                    # total and the perf workload like every other refusal.
                    macs_incomplete = True
                    macs_incomplete_reason = "dsl_kernel_unresolved_legacy" if layer.get("dsl_route") is not True else "dsl_kernel_unresolved"
                    cost_basis = "incomplete_evidence"
                    _admission_error = _admission_error or type(_cost_exc).__name__
            # Explicit torch.einsum: chain steps that sum nothing away are
            # elementwise multiplies, reported as scalar_multiplies and billed
            # as other_ops, never as MACs (independent review P2-2 units).
            einsum_scalar_multiplies = 0
            if op_type == "einsum" and equation:
                try:
                    _eop = self.einsum_analyzer.get_einsum_op(op_type, ts, equation=equation)
                    if hasattr(_eop, "scalar_multiplies"):
                        einsum_scalar_multiplies = int(_eop.scalar_multiplies(ts))
                except Exception:
                    einsum_scalar_multiplies = 0

            # Both labelled estimates for a multi-operand einsum (maintainers' ruling):
            # the as-written single contraction over all indices (macs_dense,
            # the existing convention) and the cheapest ordinary dense binary
            # schedule.  Neither is a proven universal lower bound; no
            # convention is chosen here.
            binary_schedule: Optional[Dict[str, Any]] = None
            as_written_dense: Optional[int] = None
            if is_real_einsum and equation and not macs_incomplete:
                binary_schedule = optimal_dense_binary_schedule(equation, _eq_ts())
                if binary_schedule:
                    # computed explicitly (not aliased to macs_dense) so the label
                    # stays true when a handler prices `macs` by another
                    # convention, e.g. an earlier fix's left-to-right chain for torch.einsum
                    try:
                        as_written_dense = int(compute_cost_from_equation(equation, _eq_ts()))
                    except Exception:
                        as_written_dense = None

            # Zero-compute operations: no ALU work, only pointer/metadata
            # manipulation or pure memory copies.
            #
            # View/reshape ops: pointer manipulation, zero cost
            # Slice/select ops: pointer offset, zero cost
            # Scatter/index ops: in-place writes, zero compute
            # Embedding: table lookup, zero MACs
            # Memory ops (cat, repeat, stack, chunk, split): move data but
            #   have zero *compute* cost — bounded by memory bandwidth, not
            #   SM throughput.  Their memory cost is already captured by
            #   input_elems/output_elems; assigning them other_ops would
            _ZERO_COMPUTE_OPS = {
                # Embedding
                "embedding", "embedding_bag",
                # View / reshape (pointer manipulation)
                "expand", "expand_as",
                "view", "reshape", "contiguous",
                "transpose", "permute", "t",
                "unsqueeze", "squeeze", "flatten",
                "unfold", "unflatten",
                # Slice / select (pointer offset)
                "__getitem__", "narrow", "slice", "select",
                # Scatter / in-place write
                "__setitem__", "scatter", "scatter_",
                "index_copy", "index_copy_",
                "index_put", "index_put_",
                # Memory-only ops (data movement, zero ALU compute)
                "cat", "concat", "stack",
                "chunk", "split", "tensor_split",
                "repeat", "repeat_interleave", "tile",
                "roll", "flip",
                "pad", "constant_pad_nd",
                "clone", "copy_",
                # Type conversion (zero compute)
                "to", "type", "type_as", "float", "half", "bfloat16", "int",
            }
            if op_type in _ZERO_COMPUTE_OPS or op_type in ALIAS_OPS or op_type in LAYOUT_DEPENDENT_ALIAS_OPS:
                ops_cost = 0
                is_real_einsum = False

            macs_dense = ops_cost if is_real_einsum else 0
            sparsity_fraction = mac_fraction.get(layer_id, 1.0)

            # ── Zeros are free: measured-sparsity compute discount ──
            # (an earlier revision; dropped by the lineage port at the lineage base, restored with the
            # an earlier revision semantics — TARGET-DENSITY-GUARD-1.)  When a kernel input
            # carries a measured density < 1 AND that input participates
            # MULTIPLICATIVELY in the loop body (a zero kills the whole product
            # term), the effective work is the dense loop count scaled by the
            # density (e.g. a top-k expert mask makes all-but-k/E of the expert
            # MACs free).  Additive/guard uses (e.g. `-1e9 * mask`) do NOT
            # discount compute.  Positional param_names map densities to
            # loop_expr names; without them we fall back conservatively to the
            # min density of sparse inputs.  ``tensor_sparsity`` comes from the
            # einsum_lib DSL route only — measured density under profiling,
            # explicit ``sparsity=`` annotations and analytic builtin metadata
            # (e.g. ``top_k_mask``) — never from captured PyTorch graphs, which
            # are therefore unaffected; ``macs_dense`` keeps the undiscounted
            # as-written count and the structured ``mac_sparsity_fraction``
            # applies after this discount, as before.  The discount (and the
            # ``compute_density`` field) applies to DSL @kernel bodies only: a
            # bodiless builtin layer (``gather`` / ``slice_at`` / ... are
            # exported as ``type: kernel`` with ``loop_expr: null``) is not a
            # compute op and keeps the target's accounting (historical-scope-2).
            # ``_density_scope`` is the one gate for BOTH the compute discount
            # here and the Step-2b memory discount below (historical-scope-2).
            _density_scope = op_type == "kernel" and bool(is_real_einsum or layer.get("loop_expr"))
            compute_density = 1.0
            if _density_scope and ops_cost > 0:
                _tsp = layer.get("tensor_sparsity") or {}
                _in_sp = _tsp.get("inputs") or []
                if any(sp is not None and sp < 1.0 for sp in _in_sp):
                    _loop_expr = str(layer.get("loop_expr") or "")
                    _pnames = layer.get("param_names") or []
                    # Converter-computed (full kernel AST) — the authoritative
                    # source; loop_expr keeps only innermost statements and can
                    # miss outer-level gating like `Y[n,e] = acc * M[n,e]`.
                    _mult = layer.get("multiplicative_params")
                    _densities: List[float] = []
                    for _i, _sp in enumerate(_in_sp):
                        if _sp is None or not (0.0 <= _sp < 1.0):
                            continue
                        if _mult is not None and _i < len(_pnames):
                            if _pnames[_i] in _mult:
                                _densities.append(float(_sp))
                        elif _loop_expr and _i < len(_pnames) and _pnames[_i]:
                            _pn = re.escape(str(_pnames[_i]))
                            if re.search(
                                    r"\*\s*" + _pn + r"\s*\[|"
                                    + _pn + r"\s*\[[^\]]*\]\s*\*",
                                    _loop_expr):
                                _densities.append(float(_sp))
                        else:
                            _densities.append(float(_sp))
                    if _densities:
                        compute_density = max(min(_densities), 0.0)
                        ops_cost = int(math.ceil(ops_cost * compute_density))
            # Compute dtype of the MAC-bearing operands, from the RECORDED dtypes
            # of the two matrix inputs: ``nvfp4`` (float4_e2m1fn_x2) or ``fp8``
            # (float8_*).  The perf model prices those MACs at the matching
            # tensor-core rate when the architecture declares one and REFUSES
            # the whole projection otherwise (PerfProjectionUnavailable; decision
            # 2026-10-26 — never the workload precision's rate, never an
            # incomplete-marked lower bound).  Other layers carry no field and
            # keep the workload precision.  The analysis itself is unaffected.
            compute_dtype: Optional[str] = None
            # scoped to the scaled_mm contract only (scope-1): an ordinary
            # matmul with fp8-recorded inputs keeps the workload precision's rate;
            # a general dtype-driven rate policy is a separate decision
            if _is_scaled_mm and is_real_einsum and ops_cost > 0 and not macs_incomplete:
                compute_dtype = compute_dtype_of_operands(_in_dtypes_rec[:2])
            # Split the ONE validated, density-discounted count into MACs and
            # other_ops (issue #52 / density-composition-1).  For a DSL
            # kernel ``ops_cost`` is already the count the shared gate
            # selected (validated loop evidence, else the equation product,
            # else the generic shape cost) with the measured-density discount
            # applied once above — raw ``loop_iterations`` is never re-read
            # here: a non-reduction kernel like 119 ``scale_by_routing``
            # (TH,TK->TKH, 68,157,440 statement executions) is billed its
            # validated count, not the 5.8e14 rank product, and an X*M kernel
            # at density 0.25 bills 2 of 8, not 8.
            _mac_n = layer.get("loop_mac_statements")
            _oth_n = layer.get("loop_other_statements")
            _ext_n = layer.get("loop_extra_multiplies")
            _stm_n = layer.get("loop_statements")
            _cls_uncertain = layer.get("class_evidence_uncertain")
            _classes_present = any(k in layer for k in ("loop_mac_statements", "loop_other_statements", "loop_extra_multiplies"))
            _class_invalid = False
            _per_statement = False
            if (op_type == "kernel" and _classes_present and cost_basis == "loop_iterations"
                    and layer.get("recompute_fingerprint") is not True):
                # class-guard-2: PRESENT class evidence must be valid and must
                # reconcile exactly against the converter's own statement record --
                # three non-negative integers, extras no larger than the other
                # work, mac + other - extras == the full executed statement count
                # (>= when the classes are a per-class maximum over branches), and
                # the trusted innermost count between the MACs and that total.
                # Anything else fails closed (class_evidence_invalid); only ABSENT
                # classes (legacy artifacts) keep the whole-kernel split.
                def _nn_int(v):
                    return isinstance(v, int) and not isinstance(v, bool) and v >= 0
                ok = _nn_int(_mac_n) and _nn_int(_oth_n) and _nn_int(_ext_n) and _nn_int(_stm_n) and _oth_n >= _ext_n
                if ok:
                    _executed = _mac_n + _oth_n - _ext_n
                    ok = (_executed >= _stm_n if _cls_uncertain else _executed == _stm_n) and _mac_n <= _loop_iters <= _executed
                if ok:
                    _per_statement = True
                else:
                    _class_invalid = True
            # DSL-COMPUTE-DTYPE-BINDING-1: the converter's MAC-operand record names, per
            # MAC statement group, the operands it contracts BY ROLE -- payloads (the
            # contracted elements), block scales, explicit conversions -- each payload
            # bound to its POSITION among the kernel inputs and its DECLARED logical
            # dtype.  The record is evidence only when it reconciles with the graph's
            # own provenance: valid structure, counts summing to loop_mac_statements,
            # every payload position inside the node's declared inputs with the SAME
            # dtype label the graph declares, at least two payloads / conversions per
            # group.  A contraction's compute dtype is proven only when ALL its
            # payloads (and conversions) agree; differing payload dtypes with a
            # quantised participant, or groups resolving to different compute dtypes,
            # refuse (compute_dtype_mixed); an operand whose declared dtype could not
            # be established (a cast, an arithmetic decode, a runtime scalar, a
            # conditional contracting different operands) refuses
            # (compute_dtype_unproven).  When the node DECLARES a quantised input,
            # missing / malformed / inconsistent evidence refuses
            # (compute_dtype_evidence_invalid) -- never a silent workload-rate default;
            # a node without quantised inputs and without usable evidence keeps the
            # workload precision (legacy graphs).  (the review DTYPE-CONTRACTION-1,
            # DTYPE-EVIDENCE-2, DTYPE-BINDING-3.)
            compute_dtype_operands: Optional[Dict[str, Any]] = None
            _cd_dtype: Optional[str] = None
            _cd_fail: Optional[str] = None
            _declared_in = [None if d is None else str(d) for d in ((layer.get("tensor_dtypes") or {}).get("inputs") or [])]
            _declared_quant = any(compute_dtype_of_operands([d]) for d in _declared_in if d)
            _rec_present = "loop_mac_operands" in layer
            # Round 3 (the review DTYPE-EVIDENCE-BINDING-5, DTYPE-ZERO-RECONCILIATION-6, DTYPE-ROLE-MULTIPLICITY-7): a
            # PRESENT receipt is always validated -- structure, total (== the class MAC count, 0 included) and every
            # entry reconciled against the parser's OCCURRENCE CHANNEL (loop_mac_reads: name AND input position AND
            # role per tensor-element occurrence, the conversion occurrences from the supported set, the names
            # read) -- the receipt's own operand map is never the sole binding evidence.  Cardinality is counted on
            # occurrences.  An absent record on a kernel that declares a quantised input and executes MACs refuses;
            # an absent / empty record on a genuine zero-MAC kernel, or on a non-quantised legacy graph, is valid.
            _SUPPORTED_CONVERSIONS = {"fp32"}
            if _per_statement and (_rec_present or (_declared_quant and _mac_n > 0)):
                _rec = layer.get("loop_mac_operands")
                _reads = layer.get("loop_mac_reads")

                def _nn(v):
                    return isinstance(v, int) and not isinstance(v, bool) and v >= 0

                def _entry_ok(e):
                    return (isinstance(e, dict) and isinstance(e.get("name"), str)
                            and (e.get("input") is None or (_nn(e.get("input"))))
                            and (e.get("dtype") is None or isinstance(e.get("dtype"), str)))

                def _full(e):
                    return (isinstance(e, dict) and all(isinstance(e.get(k), list) for k in ("payloads", "scales", "conversions"))
                            and isinstance(e.get("operands"), dict))

                def _rec_ok(e):
                    return (_full(e) and _nn(e.get("macs")) and all(_entry_ok(x) for x in e["payloads"] + e["scales"])
                            and all(isinstance(c, str) for c in e["conversions"]))

                def _occ_ok(o):
                    return (isinstance(o, dict) and isinstance(o.get("name"), str) and (o.get("input") is None or _nn(o.get("input")))
                            and isinstance(o.get("index"), str) and o.get("role") in ("payload", "scale"))

                def _read_ok(r):
                    return (isinstance(r, dict) and _nn(r.get("macs")) and isinstance(r.get("occurrences"), list)
                            and all(_occ_ok(o) for o in r["occurrences"]) and isinstance(r.get("conversions"), list)
                            and all(isinstance(c, str) for c in r["conversions"]) and isinstance(r.get("names"), list))

                def _bound(e, ops):
                    """The entry's position lies in the declared inputs, carries the declared label, and the record's
                    own operand map names it with that label."""
                    i, d = e.get("input"), e.get("dtype")
                    return (i is not None and i < len(_declared_in) and d is not None and _declared_in[i] is not None
                            and d.lower() == _declared_in[i].lower() and compute_dtype_of_operands([d]) == compute_dtype_of_operands([_declared_in[i]])
                            and e.get("name") in ops and str(ops.get(e.get("name"))).lower() == d.lower())

                def _source_binding_failure():
                    """DTYPE-OCCURRENCE-CHANNEL-8: RE-DERIVE the MAC operand occurrences from the kernel source bound to the
                    node (its text + sha256, the module text + sha256 at graph level, the declared signature) through the
                    parser's own pure function and compare them with the serialized channel.  None when they agree; else
                    the reason.  The channel is a validated cache, never the proof: two consistently rewritten channels
                    cannot agree with a source they contradict."""
                    import hashlib as _hl
                    ks = layer.get("kernel_source")
                    ms = graph.get("dsl_module_source") if isinstance(graph, dict) else None
                    if not isinstance(ks, dict) or not isinstance(ms, dict):
                        return "no source bound to the node"
                    text, mtext = ks.get("text"), ms.get("text")
                    if not isinstance(text, str) or not isinstance(mtext, str) or not isinstance(ks.get("name"), str):
                        return "bound source incomplete"
                    # DTYPE-KERNEL-IDENTITY-10: the function re-derived is the one THIS node traced -- its identity is
                    # the node key (`<function>#<n>`, written at conversion) and the converter's own record; the name
                    # the receipt asserts must agree with both, else nothing below is trusted
                    traced = str(layer_id).split("#")[0]
                    recorded = layer.get("kernel_function")
                    if recorded is not None and str(recorded) != traced:
                        return "the converter's kernel record disagrees with the node identity"
                    if ks["name"] != traced:
                        return "kernel_source names a function that is not this node's traced kernel"
                    if _hl.sha256(text.encode("utf-8")).hexdigest() != ks.get("sha256"):
                        return "kernel source hash mismatch"
                    if _hl.sha256(mtext.encode("utf-8")).hexdigest() != ms.get("sha256") or ks.get("module_sha256") != ms.get("sha256"):
                        return "module source hash mismatch"
                    if text not in mtext:
                        return "kernel text is not a segment of the module text"
                    try:
                        from solar.einsum_dsl.ast_parser import kernel_source_binding
                        bound = kernel_source_binding(mtext, traced)              # selected by the node's identity, never by the receipt
                    except Exception as exc:  # unparseable / missing kernel: no binding
                        return f"bound source not derivable ({type(exc).__name__})"
                    if bound.get("sha256") != ks.get("sha256"):
                        return "the module's kernel differs from the bound kernel text"
                    sig = bound.get("signature") or []
                    if [(s.get("name"), s.get("position"), s.get("dtype")) for s in sig] != [(s.get("name"), s.get("position"), s.get("dtype")) for s in (ks.get("signature") or [])]:
                        return "signature differs from the bound source"
                    for s in sig:
                        i, d = s.get("position"), s.get("dtype")
                        if s.get("tensor") and d is not None and isinstance(i, int) and i < len(_declared_in) and _declared_in[i] is not None and str(d).lower() != str(_declared_in[i]).lower():
                            return "declared input dtype differs from the source annotation"
                    pos = {s.get("name"): s.get("position") for s in sig}
                    derived = {(tuple((o["name"], pos.get(o["name"]), o["index"], o["role"]) for o in sp.get("occurrences", [])),
                                tuple(sp.get("conversions", [])), tuple(sorted(sp.get("names", [])))) for sp in bound.get("specs", [])}
                    channel = {(tuple((o["name"], o["input"], o["index"], o["role"]) for o in r["occurrences"]),
                                tuple(r["conversions"]), tuple(sorted(r["names"]))) for r in _reads}
                    return None if derived == channel else "occurrences differ from the bound source"

                if not _rec_present or not isinstance(_rec, list):
                    _cd_fail = "compute_dtype_evidence_invalid"                   # declared quantised, MACs, no receipt
                elif not all(_full(e) for e in _rec):
                    # a legacy-shaped record (no role lists): no binding evidence; on a quantised node it refuses,
                    # on a non-quantised node the workload precision stands (legacy graphs)
                    if _declared_quant:
                        _cd_fail = "compute_dtype_evidence_invalid"
                else:
                    _structural = (all(_rec_ok(e) for e in _rec) and sum(e["macs"] for e in _rec) == _mac_n
                                   and isinstance(_reads, list) and len(_reads) == len(_rec) and all(_read_ok(r) for r in _reads)
                                   and all(r["macs"] == e["macs"] for r, e in zip(_reads, _rec)))
                    if not _structural:
                        _cd_fail = "compute_dtype_evidence_invalid"               # a present receipt that does not reconcile
                    elif _source_binding_failure() is not None:
                        _cd_fail = "compute_dtype_evidence_invalid"               # the channel does not re-derive from the bound source
                    else:
                        compute_dtype_operands = {}
                        _cd_seen = set()
                        for e, r in zip(_rec, _reads):
                            for n, d in e["operands"].items():
                                compute_dtype_operands[str(n)] = None if d is None else str(d)
                            # the FULL entry is validated before any zero-count fast path (DTYPE-ZERO-PAYLOAD-SKIP-9)
                            if e["macs"] == 0:
                                if e["payloads"] or e["scales"] or e["conversions"] or r["occurrences"] or r["conversions"]:
                                    _cd_fail = "compute_dtype_evidence_invalid"   # a zero-MAC entry carrying contraction operands is a contradiction
                                    _cd_seen.add("__unresolved__")
                                continue
                            if e.get("unproven") or e.get("arms_differ"):
                                _cd_fail = _cd_fail or ("compute_dtype_unproven" if _declared_quant else None)
                                _cd_seen.add("__unproven__")
                                continue
                            # the receipt against the occurrence channel: name + position + role, one entry per occurrence
                            _occ_pay = sorted((o["name"], o["input"]) for o in r["occurrences"] if o["role"] == "payload")
                            _occ_sc = sorted((o["name"], o["input"]) for o in r["occurrences"] if o["role"] == "scale")
                            _rec_pay = sorted((x["name"], x.get("input")) for x in e["payloads"])
                            _rec_sc = sorted((x["name"], x.get("input")) for x in e["scales"])
                            _conv_ok = (sorted(e["conversions"]) == sorted(r["conversions"])
                                        and all(c.lower() in _SUPPORTED_CONVERSIONS for c in e["conversions"]))
                            _names_ok = set(e["operands"]) == ({o["name"] for o in r["occurrences"]} | set(r["names"]))
                            if not (_occ_pay == _rec_pay and _occ_sc == _rec_sc and _conv_ok and _names_ok):
                                _cd_fail = "compute_dtype_evidence_invalid"       # rewritten, duplicated, unused or unknown entries
                                _cd_seen.add("__unresolved__")
                                continue
                            if not all(_bound(x, e["operands"]) for x in e["payloads"] + e["scales"]) or len(e["payloads"]) + len(e["conversions"]) < 2:
                                if _declared_quant:
                                    _cd_fail = _cd_fail or "compute_dtype_evidence_invalid"
                                _cd_seen.add("__unresolved__")
                                continue
                            _pay = sorted({str(x["dtype"]).lower() for x in e["payloads"]} | {str(c).lower() for c in e["conversions"]})
                            if len(_pay) > 1 and any(compute_dtype_of_operands([d]) for d in _pay):
                                _cd_fail = _cd_fail or "compute_dtype_mixed"      # a quantised payload against another dtype
                                _cd_seen.add("__mixed__")
                                continue
                            _cd_seen.add(compute_dtype_of_operands(_pay) if len(_pay) == 1 else None)
                        if _cd_fail is None:
                            _quant_seen = {d for d in _cd_seen if d and not str(d).startswith("__")}
                            if len(_quant_seen) > 1 or (_quant_seen and len(_cd_seen) > 1):
                                _cd_fail = "compute_dtype_mixed"                  # MAC groups at different tensor-core rates
                            elif len(_quant_seen) == 1 and len(_cd_seen) == 1:
                                _cd_dtype = next(iter(_quant_seen))
            if _cls_uncertain:
                _cd_dtype = None                                              # uncertain classes carry no quantised claim
                if _cd_fail in ("compute_dtype_unproven", "compute_dtype_mixed"):
                    _cd_fail = None                                           # the class path already holds the layer
            _fail_closed = "class_evidence_invalid" if _class_invalid else _cd_fail
            if _fail_closed:
                ops_cost = 0
                macs_dense = 0
                macs_incomplete = True
                macs_incomplete_reason = _fail_closed
                cost_basis = "incomplete_evidence"
            if _fail_closed:
                macs = 0
                other_ops = einsum_scalar_multiplies
            elif _per_statement:
                # issue #59 unit contract: MACs are the multiply-accumulate
                # statements' trips, other_ops the remaining statements' trips
                # (plus standalone multiplies of multi-factor MAC statements),
                # each scaled by the density discount applied to ops_cost above
                # (ops_cost == loop_iterations when undiscounted; the innermost
                # count of a reduction kernel may exclude outer statements, so
                # the scale is the discount ratio, not ops_cost itself).
                if _loop_iters and ops_cost != _loop_iters:
                    # density-discounted: scale both parts by the same integer ratio
                    macs_raw = (int(_mac_n) * int(ops_cost)) // int(_loop_iters)
                    other_raw = (int(_oth_n) * int(ops_cost)) // int(_loop_iters)
                else:
                    macs_raw, other_raw = int(_mac_n), int(_oth_n)      # exact integers, no float round-trip
                macs = macs_raw if sparsity_fraction == 1.0 else int(round(macs_raw * sparsity_fraction))
                other_ops = other_raw + einsum_scalar_multiplies
                macs_dense = int(_mac_n)
                unit_basis = "per_statement"
                if _cd_dtype and macs > 0:
                    compute_dtype = _cd_dtype
            elif is_real_einsum:
                # exact for the common dense case: a float multiply would round a count above 2**53 (independent review note)
                macs = int(ops_cost) if sparsity_fraction == 1.0 else int(round(ops_cost * sparsity_fraction))
                other_ops = einsum_scalar_multiplies
            else:
                # non-real (stored graphs may carry an explicit einsum with the
                # flag false): all of its work is CUDA-core work, reduction
                # steps and elementwise steps alike (independent review P2-3)
                macs = 0
                other_ops = ops_cost + einsum_scalar_multiplies

            flops = int(2 * macs)

            input_shapes = tensor_shapes.get("inputs") or []
            output_shapes = tensor_shapes.get("outputs") or []
            input_type_list = tensor_types.get("inputs") or []
            output_type_list = tensor_types.get("outputs") or []
            # Parameter identity recorded by the converter (capture
            # parameter-tensor node per weight slot); absent on graphs saved
            # by older converters and on synthetic graphs.
            has_parameter_sources = isinstance(layer.get("parameter_sources"), dict)
            input_source_list = (layer.get("parameter_sources") or {}).get("inputs") or []

            # ── Step 1: Compute per-tensor sizes from shapes ──
            input_sizes: List[int] = []
            output_sizes: List[int] = []
            for shp in input_shapes:
                input_sizes.append(_product(shp) if isinstance(shp, list) else 0)
            for shp in output_shapes:
                output_sizes.append(_product(shp) if isinstance(shp, list) else 0)

            # memory_reads[i]  = DRAM read elements for input tensor i
            # memory_writes[i] = DRAM write elements for output tensor i
            # Initialised to raw tensor sizes; special-case ops override below.
            memory_reads: List[int] = list(input_sizes)
            memory_writes: List[int] = list(output_sizes)
            input_name_list_early = tensor_names.get("inputs") or []

            # ── Step 2: Override memory_reads/writes for special-case ops ──

            _ZERO_COPY_VIEW_OPS = {
                "expand", "expand_as",
                "view", "reshape", "contiguous",
                "transpose", "permute", "t",
                "unsqueeze", "squeeze", "flatten",
                "unfold", "unflatten",
                # chunk/split return views into the source tensor
                "chunk", "split", "tensor_split",
            }
            # Shared with the access-region analysis, which relies on the
            # invariant that these ops' read equals their output size.
            _SLICE_VIEW_OPS = SLICE_VIEW_OPS
            _SCATTER_OPS = {
                "__setitem__", "scatter", "scatter_",
                "index_copy", "index_copy_",
                "index_put", "index_put_",
                # Accumulating scatters touch only the indexed rows of the
                # target (read-modify-write of the source footprint).
                "index_add", "index_add_", "scatter_add", "scatter_add_",
                "scatter_reduce", "scatter_reduce_", "index_reduce",
                "index_reduce_", "masked_scatter", "masked_scatter_", "put_",
            }

            # ── Gathers (issue #38 / an earlier decision) ──
            # torch.gather, index_select, take/take_along_dim, embedding and
            # advanced indexing (x[idx] with a tensor or Python-list index)
            # are re-indexings, not compute:
            #   * every index tensor is read fully;
            #   * the source is read only at the gathered elements, i.e.
            #     numel(output) elements as written (repeated indices are
            #     charged each time; this analyzer has no unique-element view);
            #   * the output is written fully;
            #   * other_ops = one per OUTPUT element (address computation),
            #     never per source element; MACs are zero.
            # The source operand is the last input for embedding (indices,
            # table) and the first input otherwise (source, index...).
            # __getitem__ is classified from its recorded index argument
            # (classify_getitem_index): ints/slices/None/Ellipsis and 0-dim
            # tensors are zero-copy views; tensor and list indices are
            # gathers even when the index does not survive as an input
            # shape (its numel is then taken from the attributes, and the
            # index read is charged on the single captured input slot).
            # embedding_bag is NOT a plain gather: it gathers numel(indices)
            # rows and reduces them (see the bag rule below).
            _GATHER_OPS = {"gather", "index_select", "take", "take_along_dim", "embedding"}
            gather_extra_fields: Dict[str, Any] = {}
            extra_index_reads = 0
            _recovered_slot: Optional[int] = None   # index slot of a recovered (attribute-only) gather index
            extra_index_dtype: Optional[str] = None   # dtype of an index recovered from the attributes
            layer_incomplete_reason: Optional[str] = None
            if _loop_meta_incomplete:
                layer_incomplete_reason = "loop_metadata_incomplete"
            elif _cls_uncertain and (_per_statement or layer.get("recompute_fingerprint") is True):
                # TARGET-KB-BRANCH-CLASS-4: the per-class maximum over branches is
                # billed, but the kernel's units are not proven -> incomplete.  The
                # recompute billing path bills by the SM/loop convention but keeps
                # the refusal (recompute-refusal-30): the marker reaches layer,
                # total and perf status regardless of the billing path
                layer_incomplete_reason = str(_cls_uncertain)
            if op_type == "__getitem__":
                _in0 = input_shapes[0] if input_shapes and isinstance(input_shapes[0], list) else None
                _out0 = output_shapes[0] if output_shapes and isinstance(output_shapes[0], list) else None
                _cls = classify_getitem_index(layer.get("raw_attributes"), _in0, _out0, with_dtype=True)
                gather_extra_fields["index_kind"] = _cls["kind"]
                if _cls["kind"] == "unavailable":
                    # no call attributes: fall back to the input-count predicate
                    is_gather = len(input_sizes) >= 2
                elif _cls["kind"] in ("tensor", "sequence"):
                    is_gather = True
                    if len(input_sizes) < 2:
                        # the index did not survive as an input: its elements are
                        # read in addition to the source, at the INDEX's own
                        # width (typing-1), never at the source's dtype
                        extra_index_reads = int(_cls["index_elements"])
                        extra_index_dtype = _cls.get("index_dtype")
                        if extra_index_reads and not extra_index_dtype:
                            layer_incomplete_reason = "gather_index_dtype_unknown"
                        if extra_index_reads:
                            gather_extra_fields["recovered_index"] = {"elements": extra_index_reads, "dtype": extra_index_dtype}
                else:  # basic or unknown -> view accounting (unknown is flagged)
                    is_gather = False
            else:
                # a built-in DSL gather (table, index) is a gather like torch.gather (TARGET-GATHER-COST-1)
                is_gather = op_type in _GATHER_OPS or (op_type == "kernel" and layer.get("builtin") == "gather")
            is_bag = op_type == "embedding_bag"
            if is_gather:
                total_output = sum(output_sizes)
                src = len(input_sizes) - 1 if op_type == "embedding" else 0
                memory_reads = list(input_sizes)
                if input_sizes:
                    memory_reads[src] = int(total_output)
                if extra_index_reads:
                    # a recovered (attribute-only) index is a SEPARATE read slot
                    # with its own dtype: it is classified, priced and grouped
                    # like any other external read below (typing-2), never
                    # folded into the source's slot at the source's width
                    _recovered_slot = len(memory_reads)
                    memory_reads.append(int(extra_index_reads))
                memory_writes = list(output_sizes)
                macs = 0
                flops = 0
                other_ops = int(total_output)
                is_real_einsum = False
            elif is_bag:
                # EmbeddingBag: gather numel(indices) rows of D elements, then
                # reduce each bag.  Reads: indices (+ offsets, + per-sample
                # weights) fully, gathered = numel(indices) * D table elements
                # (as written); writes: the reduced output; ops: one address
                # op per gathered element + the reduction (sum/max: gathered -
                # output; mean: + output divisions) + one multiply per gathered
                # element when per_sample_weights are given.  mode comes from
                # the recorded attributes; without them "sum" is assumed and
                # flagged (bag_mode: unknown).
                _meta = embedding_bag_metadata(layer.get("raw_attributes"))
                total_output = sum(output_sizes)
                n_idx = input_sizes[0] if input_sizes else 0
                table = input_shapes[-1] if input_shapes and isinstance(input_shapes[-1], list) else None
                d = int(table[-1]) if table else (int(output_shapes[0][-1]) if output_shapes and isinstance(output_shapes[0], list) else 0)
                gathered = int(n_idx) * d
                mode = _meta["mode"] or "sum"
                memory_reads = list(input_sizes)
                if input_sizes:
                    memory_reads[0] = int(n_idx) + int(_meta["offsets_elements"]) + int(_meta["per_sample_weights_elements"])
                    memory_reads[-1] = gathered
                memory_writes = list(output_sizes)
                reduce_ops = max(gathered - int(total_output), 0)
                if mode == "mean":
                    reduce_ops += int(total_output)
                if _meta["per_sample_weights_elements"]:
                    reduce_ops += gathered
                other_ops = gathered + reduce_ops
                macs = 0
                flops = 0
                is_real_einsum = False
                gather_extra_fields["bag_mode"] = mode if _meta["mode"] else "unknown"
                gather_extra_fields["gathered_elements"] = gathered

            # Bool-typed tensors (masks) used to be zeroed here because a
            # single graph-wide bytes_per_element priced them at 2 B. Bytes
            # are now taken from each tensor's own dtype (1 B for bool), so
            # masks count at their true size.

            # View/reshape ops produce zero-copy aliases — they never
            # materialize data to DRAM.  The downstream consumer accounts
            # for the actual read, so these ops contribute 0 memory.
            if op_type in _ZERO_COPY_VIEW_OPS:
                memory_reads = [0] * len(input_sizes)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Slicing/selection ops return a view into the source tensor.
            # The actual memory read is the output slice size, not the
            # full source.  Set read = output size so the downstream
            # consumer accounts for reading the slice.
            elif op_type in _SLICE_VIEW_OPS and not is_gather:
                # (a __getitem__ with a tensor or list index was handled as a gather above)
                out_total = sum(output_sizes)
                memory_reads = [out_total] if input_sizes else []
                memory_reads += [0] * max(0, len(input_sizes) - 1)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Scatter/index-write ops (__setitem__, scatter, index_copy)
            # write a slice into a large target tensor.  Memory cost is
            # the values being written, not the full target.  The smallest
            # input shape is typically the values/indices; use that as
            # the write cost and set output to the same (in-place update).
            elif op_type in _SCATTER_OPS:
                if len(input_sizes) >= 2:
                    slice_elems = max(sorted(input_sizes)[:-1])
                elif input_sizes:
                    slice_elems = min(input_sizes)
                elif output_sizes:
                    slice_elems = min(output_sizes)
                else:
                    slice_elems = 0
                memory_reads = [0] * len(input_sizes)
                memory_writes = [slice_elems] if output_sizes else []
                memory_writes += [0] * max(0, len(output_sizes) - 1)
                other_ops = 0

            # Creation ops (zeros_like, full_like, arange ...) only take
            # shape/dtype from their argument; nothing is read from DRAM.
            if op_type in _CREATION_OPS_ZERO_READ:
                memory_reads = [0] * len(input_sizes)
                other_ops = 0

            if layer_id in lazy_read_cap and memory_reads:
                memory_reads[0] = min(memory_reads[0], lazy_read_cap[layer_id])

            # Orphaned dead-end layers: ALL inputs trace (through views)
            # to non-existent sources AND the layer has no live output.
            # For scatter/setitem: if the TARGET (first input) traces to a
            # non-existent source and the output is dead-end, the write is
            # phantom — zero all memory.
            # Standalone if (not elif) so it overrides any prior op-type branch.
            if layer_id in _dead_end_layers and input_layer_ids:
                _is_orphan = False
                _SCATTER_TARGET_OPS_INLINE = _SCATTER_OPS

                def _source_is_orphan(cid: str) -> bool:
                    src = _trace_source_through_views(cid) if cid in transparent_layer_ids else cid
                    return src not in all_layer_ids and src not in start_node_ids

                if all(_source_is_orphan(c) for c in input_layer_ids):
                    _is_orphan = True

                if (
                    not _is_orphan
                    and op_type in _SCATTER_TARGET_OPS_INLINE
                    and _source_is_orphan(input_layer_ids[0])
                ):
                    _is_orphan = True

                if _is_orphan:
                    memory_reads = [0] * len(input_sizes)
                    memory_writes = [0] * len(output_sizes)
                    other_ops = 0
                    _orphaned_layers.add(layer_id)

            # ── Step 2b: Apply sparsity (density) discount ──
            # (an earlier revision; restored — TARGET-DENSITY-GUARD-1.)  When
            # tensor_sparsity is present, scale element counts by the density
            # fraction.  This reduces DRAM traffic for sparse tensors (e.g.
            # upper-triangular matrices, masked attention).  The field comes from
            # the einsum_lib DSL route only (measured, annotated or analytic
            # density); captured PyTorch graphs are unaffected.  Like the
            # compute discount, it applies to DSL @kernel BODIES only: a
            # bodiless builtin (``gather`` / ``slice_at`` ...) keeps the
            # target's memory accounting (historical-scope-2).
            tensor_sparsity: Dict[str, Any] = layer.get("tensor_sparsity") or {}
            input_sparsities = (tensor_sparsity.get("inputs") or []) if _density_scope else []
            output_sparsities = (tensor_sparsity.get("outputs") or []) if _density_scope else []
            for i, sp in enumerate(input_sparsities):
                if sp is not None and i < len(memory_reads):
                    memory_reads[i] = int(memory_reads[i] * sp)
            for i, sp in enumerate(output_sparsities):
                if sp is not None and i < len(memory_writes):
                    memory_writes[i] = int(memory_writes[i] * sp)

            # ── Step 3: Derive totals from corrected per-tensor counts ──
            input_elems = int(sum(memory_reads))
            output_elems = int(sum(memory_writes))
            unfused_elems = input_elems + output_elems

            # Alias ops (detach, view_as, ...): no bytes move in unfused mode.
            # memory_reads/memory_writes keep the raw sizes so Step 4 still
            # classifies a model input or output that passes through the alias
            # (the DRAM traffic belongs to the boundary, not to the alias).
            is_alias = op_type in ALIAS_OPS
            alias_qualification: Optional[str] = "alias (pure alias op)" if is_alias else None
            receiver_read_elems = 0                 # reshape_as receiver absent from the operands: read charged at output size
            capture_incomplete: Optional[str] = None
            incomplete_reason: Optional[str] = None  # STATUS_CONTRACT rule 1 (machine-readable)
            if op_type in LAYOUT_DEPENDENT_ALIAS_OPS:
                # The shape-reference operand(s) supply a shape only: never a data read.
                for i in range(1, len(memory_reads)):
                    memory_reads[i] = 0
                src_name = input_name_list_early[0] if input_name_list_early else None
                receiver_present = _reshape_as_receiver_present(layer)
                receiver_state = _reshape_as_receiver_state(layer)
                if receiver_present is not True:
                    # The receiver is not the first captured operand (buffer /
                    # plain-tensor attribute receivers are recorded in the call
                    # but never become graph operands) or no call was recorded:
                    # never borrow the shape-reference tensor's layout.  Charge
                    # the receiver read at the output's size plus the write.
                    reason = None
                    src_name = None
                    receiver_read_elems = int(sum(output_sizes))
                    for i in range(len(memory_reads)):
                        memory_reads[i] = 0
                elif src_name is not None and src_name in tensor_producers:
                    reason = self._source_layout_established(
                        tensor_producers[src_name], layers_in, tensor_producers, layout_memo, tensor_layouts)
                else:
                    # model input / parameter: only a layout RECORDED at capture is a proof
                    reason = self._recorded_layout_reason(src_name, tensor_layouts)
                _at_call = ((layer.get("call_operands") or {}).get("receiver_layout") or {}).get("at_call") or {}
                executed_noncontiguous = reason is not None and _at_call.get("contiguous") is False
                if executed_noncontiguous:
                    # the capture observed the receiver's strides at the executed call: a
                    # non-contiguous receiver cannot be a free reshape whatever the
                    # provenance proof says (a hidden tensor mutated in place, storage-5)
                    reason = None
                if reason:
                    is_alias = True
                    alias_qualification = f"alias (layout established: {reason})"
                else:
                    # a materialising (or unprovable) reshape_as: source read + output write, zero ALU work
                    is_alias = False
                    rec = tensor_layouts.get(src_name or "") if src_name is not None and src_name not in tensor_producers else None
                    if executed_noncontiguous:
                        alias_qualification = ("copy_charged (receiver executed non-contiguous at the recorded call: "
                                               "provenance proof refused: charged as copy)")
                        capture_incomplete = "reshape_as receiver executed non-contiguous although its provenance proved a contiguous layout (in-place mutation)"
                        incomplete_reason = "reshape_as_receiver_storage_mutated"
                    elif receiver_state == "mutated":
                        # identity-bound receiver whose storage/strides no longer match the
                        # binding-time record (or that executed non-contiguous): the recorded
                        # layout is stale, the copy is charged and the result qualified
                        alias_qualification = ("copy_charged (receiver storage/strides changed after the layout record was bound: "
                                               "recorded layout no longer describes the executed receiver: charged as copy)")
                        capture_incomplete = "reshape_as receiver storage or strides mutated after capture binding (.data / set_ / as_strided_ / resize_)"
                        incomplete_reason = "reshape_as_receiver_storage_mutated"
                    elif receiver_present is not True:
                        alias_qualification = ("copy_charged (receiver not a connected graph operand bound by identity: layout unknown: charged as copy)"
                                               if receiver_state == "unbound" else
                                               "copy_charged (no identity record of the call: receiver unverified: layout unknown: charged as copy)")
                        capture_incomplete = ("reshape_as receiver is not a connected graph operand (buffer / plain tensor attribute / keyword-only operands)"
                                              if receiver_state == "unbound" else "reshape_as call has no identity record (not recorded)")
                        incomplete_reason = "reshape_as_receiver_unbound" if receiver_state == "unbound" else "reshape_as_receiver_unverified"
                    elif isinstance(rec, dict) and rec.get("contiguous") is False:
                        alias_qualification = "copy_charged (layout recorded non-contiguous at capture: charged as copy)"
                    else:
                        alias_qualification = "copy_charged (layout unknown: charged as copy)"
                    other_ops = 0
                input_elems = int(sum(memory_reads)) + receiver_read_elems
                output_elems = int(sum(memory_writes))
                unfused_elems = input_elems + output_elems
            if is_alias:
                input_elems = 0
                output_elems = 0
                unfused_elems = 0
                other_ops = 0

            # Per-tensor byte widths from the recorded dtypes (fallback: the
            # graph-wide element_size). A single bytes_per_element misprices
            # mixed graphs: bool masks (1 B) at 2 B, fp32 outputs of an fp8
            # problem at 1 B, packed FP4 inputs next to bf16 activations.
            _in_dt = (layer.get("tensor_dtypes") or {}).get("inputs") or []
            _out_dt = (layer.get("tensor_dtypes") or {}).get("outputs") or []
            in_bpe = [_dtype_bytes(_in_dt[i] if i < len(_in_dt) else None, element_size)
                      for i in range(len(memory_reads))]
            out_bpe = [_dtype_bytes(_out_dt[o] if o < len(_out_dt) else None, element_size)
                       for o in range(len(memory_writes))]
            if _recovered_slot is not None and _recovered_slot < len(in_bpe):
                # the recovered index slot is priced at the INDEX's recorded
                # width (unknown dtype: the source width, and the layer is
                # marked incomplete, see layer_incomplete_reason)
                in_bpe[_recovered_slot] = _dtype_bytes(extra_index_dtype, element_size) if extra_index_dtype else in_bpe[0]
            input_bytes = sum(r * b for r, b in zip(memory_reads, in_bpe))
            output_bytes = sum(w * b for w, b in zip(memory_writes, out_bpe))
            unfused_bytes = input_bytes + output_bytes
            if receiver_read_elems:
                _rb = _dtype_bytes(_out_dt[0] if _out_dt else None, element_size)
                input_bytes += receiver_read_elems * _rb
                unfused_bytes = input_bytes + output_bytes
            if is_alias:
                # An alias moves no bytes in unfused mode either; boundary
                # traffic through it is billed in Step 4 from memory_reads /
                # memory_writes, which keep the raw sizes (an earlier decision convention).
                input_bytes = 0.0
                output_bytes = 0.0
                unfused_bytes = 0.0

            # ── Step 4: Classify inputs as external vs graph-internal ──
            # Uses memory_reads (already corrected) so no re-scanning needed.
            # Classify each input tensor:
            #   - graph-internal  → intermediate (fusable, skip in fused model)
            #   - other           → external: weights and model inputs (DRAM read);
            #                       labelled by tensor type as weight_elements
            #                       ("weight") or model_io_elements (the rest)
            #
            # graph-internal = produced by a non-view op in the graph.
            # Transparent views are traced back to their source. The tensor
            # type is not consulted for externality: an operand in a weight
            # role can still be computed in the graph, e.g. F.linear(x, w * 2).
            # The weight / model-input split only labels the external reads;
            # the external total (and so fused/unfused) is unchanged
            # (issue #36 / an earlier decision).
            input_name_list = input_name_list_early
            graph_internal_input_elems = 0   # intermediate activations from other ops
            external_input_elems = 0         # weights + model-level inputs (always DRAM)
            weight_input_elems = 0           # parameter numel of weight-typed external inputs
            weight_read_input_elems = 0      # their DRAM read elements (traffic basis)
            model_input_elems_raw = 0        # model-level (non-weight) external inputs (DRAM)
            graph_internal_input_bytes = 0.0
            external_input_bytes = 0.0
            counted_parameter_ids: set = set()        # parameters already in this op's footprint
            weight_unresolved_slots: List[int] = []   # weight-typed input slots whose provenance is unknown
            weight_unresolved_elems = 0               # their numel (excluded from the footprint)

            for i, mem_read in enumerate(memory_reads):
                if mem_read <= 0:
                    continue
                iname = input_name_list[i] if i < len(input_name_list) else ""

                if iname in tensor_producers:
                    producer_id = tensor_producers[iname]
                    source_id = _trace_source_through_views(producer_id)
                    is_graph_internal = source_id in all_layer_ids and source_id not in transparent_layer_ids
                elif orphan_buffer_tensors and (
                    iname in orphan_buffer_tensors
                    or _canonical_external_tensor(
                        iname, tensor_producers, layers_in, transparent_layer_ids
                    ) in orphan_buffer_tensors
                ):
                    # A producer-less tensor that is a declared model output
                    # is a buffer created inside the model (torchview split
                    # the in-place accumulator into an orphan node); reading
                    # it back is on-chip traffic, not a DRAM input.
                    is_graph_internal = True
                else:
                    is_graph_internal = False

                if is_graph_internal:
                    graph_internal_input_elems += mem_read
                    graph_internal_input_bytes += mem_read * in_bpe[i]
                else:
                    external_input_elems += mem_read
                    external_input_bytes += mem_read * in_bpe[i]
                    # Parameter provenance is the converter's "weight" type
                    # (torchview parameter-tensor nodes and the earlier decision
                    # attached parameters/buffers).  An input beyond the
                    # typed list has NO provenance and is model I/O: the
                    # SDPA decomposition's sub-layers carry empty
                    # tensor_types, and defaulting them to weight reported
                    # softmax/scale operands as weights (INTEG-3).
                    itype = input_type_list[i] if i < len(input_type_list) else ""
                    if itype == "weight":
                        # weight_elements is the PARAMETER footprint (numel),
                        # independent of how much of it this op reads
                        # (embedding gathers read rows: an earlier decision); the read
                        # itself is kept in weight_read_elements.
                        #
                        # Provenance beyond the label (provenance-1):
                        # with the converter's `parameter_sources`, a slot
                        # counts only when a capture parameter-tensor node
                        # supplies it, once per distinct node per op (the
                        # same nn.Parameter in two slots is one footprint:
                        # parameter-identity-1).  A `weight` slot with
                        # no recorded source is provenance UNKNOWN: it is an
                        # external read of the model-I/O boundary, excluded
                        # from the footprint and reported in
                        # `weight_provenance_unresolved` (status incomplete).
                        # Graphs without `parameter_sources` (older saved
                        # graphs, synthetic graphs) carry only the label; it
                        # is trusted except for explicit `einsum` operands,
                        # the one path on which a pre-#46 converter typed
                        # untraced activations `weight` (INTEG-2).
                        numel_i = input_sizes[i] if i < len(input_sizes) else mem_read
                        if has_parameter_sources:
                            source_i = input_source_list[i] if i < len(input_source_list) else None
                            provenance = "parameter" if source_i else "unknown"
                        else:
                            source_i = None
                            provenance = "unknown" if op_type == "einsum" else "label"
                        if provenance == "unknown":
                            weight_unresolved_slots.append(int(i))
                            weight_unresolved_elems += int(numel_i)
                            model_input_elems_raw += mem_read
                        else:
                            weight_read_input_elems += mem_read
                            if iname:
                                weight_boundary_names.add(iname)
                            if source_i is not None:
                                if source_i not in counted_parameter_ids:
                                    counted_parameter_ids.add(source_i)
                                    weight_input_elems += numel_i
                                    if iname:
                                        weight_boundary_footprint[iname] = weight_boundary_footprint.get(iname, 0) + int(numel_i)
                            else:
                                weight_input_elems += numel_i
                                if iname:
                                    weight_boundary_footprint[iname] = weight_boundary_footprint.get(iname, 0) + int(numel_i)
                    else:
                        model_input_elems_raw += mem_read
                    if i == _recovered_slot:
                        # a recovered gather index has no tensor name of its own:
                        # one external read group per layer, at the index width,
                        # so the fused footprint keeps it (typing-2)
                        group = external_read_groups.setdefault(
                            f"{layer_id}::recovered_index", {"boxes": [], "counted": [], "full": set(), "bpe": 0.0}
                        )
                        group["bpe"] = max(float(group.get("bpe") or 0.0), in_bpe[i])
                        group["counted"].append(int(mem_read))
                        # declared-boundary billing needs the group's own
                        # evidence: the recovered index dtype (None when the
                        # attributes did not record it) and the owning layer
                        # (M97-INTEG-D09-1)
                        group["dtype"] = extra_index_dtype
                        group["layer"] = layer_id
                        group["unnamed"] = True
                    elif iname:
                        canonical, boxes, full_candidate = _resolve_read_region(
                            op_type,
                            i,
                            iname,
                            int(mem_read),
                            input_shapes,
                            input_sizes,
                            layer,
                            layers_in,
                            tensor_producers,
                            transparent_layer_ids,
                        )
                        group = external_read_groups.setdefault(
                            canonical, {"boxes": [], "counted": [], "full": set(), "bpe": 0.0}
                        )
                        group["bpe"] = max(float(group.get("bpe") or 0.0), in_bpe[i])
                        if full_candidate > 0:
                            group["full"].add(int(full_candidate))
                        if boxes:
                            group["boxes"].extend(boxes)
                        else:
                            group["counted"].append(int(mem_read))
                        # Boundary-billing bases (per declared input NAME, not
                        # canonical base): max-dedup and raw fan-out sums.
                        unique_external_inputs[iname] = max(
                            unique_external_inputs.get(iname, 0), int(mem_read)
                        )
                        if external_input_canonical.setdefault(
                                iname, canonical) != canonical:
                            external_input_canonical[iname] = ""
                        raw_external_input_elems[iname] = (
                            raw_external_input_elems.get(iname, 0) + mem_read)

            if receiver_read_elems:
                # The unbound reshape_as receiver is an external tensor the
                # graph never saw: its read enters the Step 4 external-read
                # accounting once (input-4c), priced at the output's
                # dtype width (reshape_as returns the receiver's dtype), so the
                # fused/prefetched totals and external_input_bytes carry it.
                _rbpe = float(out_bpe[0]) if out_bpe else float(element_size)
                external_input_elems += receiver_read_elems
                external_input_bytes += receiver_read_elems * _rbpe
                # INTEG-15: under the earlier decision split the receiver read is a model-level
                # (non-weight) external read, so the per-layer model_io_elements and the
                # fused per-layer total carry it too
                model_input_elems_raw += receiver_read_elems
                group = external_read_groups.setdefault(
                    f"{layer_id}.receiver(unbound)", {"boxes": [], "counted": [], "full": set(), "bpe": 0.0}
                )
                group["bpe"] = max(float(group.get("bpe") or 0.0), _rbpe)
                group["counted"].append(int(receiver_read_elems))

            intermediate_input_elems = int(graph_internal_input_elems)
            weight_elems = int(weight_input_elems)
            weight_read_elems = int(weight_read_input_elems)
            model_input_elems = int(model_input_elems_raw)
            input_is_intermediate = graph_internal_input_elems > 0

            # Classify outputs: intermediate if consumed by a real
            # non-transparent op, or by views that lead to one.
            output_name_list = tensor_names.get("outputs") or []
            output_is_intermediate = False
            for oname in output_name_list:
                for consumer_id in (tensor_consumers.get(oname) or set()):
                    if consumer_id not in transparent_layer_ids:
                        output_is_intermediate = True
                        break
                    if _has_real_consumer(consumer_id):
                        output_is_intermediate = True
                        break
                if output_is_intermediate:
                    break
            if (graph.get("model_output_provenance") == "dsl-return-v1"
                    and len(output_name_list) == 1
                    and output_name_list[0] in declared_output_tensors):
                # A DSL node has one owned output. Returning it requires the
                # producer's existing write even when a later op consumes it.
                # Charge it here once; the fallback below would add a second
                # unfused write for an already traced intermediate producer.
                output_is_intermediate = False
            nonreturned_store = False
            if (not output_is_intermediate and (model_output_ops or model_outputs)
                    and not _reaches_model_output(layer_id)):
                # Consumer-less result that is not a declared model output:
                # dead code, or consumed by an op torchview does not trace
                # (in-place ``result[mask] = v``). A fused kernel never
                # writes it to DRAM, so it is intermediate traffic.
                output_is_intermediate = True
                nonreturned_store = True
                nonreturned_store_bytes += float(output_bytes)
                nonreturned_store_layers.append(layer_id)

            # Intermediate output elems: written to cache (fused) not DRAM
            intermediate_output_elems = output_elems if output_is_intermediate else 0
            boundary_output_elems = int(sum(memory_writes)) if is_alias else output_elems
            # Total intermediate elems for this layer (inputs + outputs); an
            # alias moves nothing through the cache either.
            layer_intermediate_elems = 0 if is_alias else (intermediate_input_elems + intermediate_output_elems)
            layer_intermediate_bytes = 0.0 if is_alias else (graph_internal_input_bytes + (
                output_bytes if output_is_intermediate else 0.0))

            # Model output elems: final graph outputs that must go to DRAM
            model_output_elems = boundary_output_elems if not output_is_intermediate else 0
            # Per-op model I/O: non-weight external inputs + model outputs
            # (no intermediates, no weights — those are weight_elems)
            model_io_elems = model_input_elems + model_output_elems

            # Track unique external outputs for deduplication.  Each output
            # tensor counts its own write elements: assigning the layer's
            # total (sum over all outputs) to every output name would count
            # a multi-output op (e.g. max(dim) returning values + indices)
            # once per output, overcounting DRAM writes and breaking the
            # SOL lower bound.
            if not output_is_intermediate:
                if layer_id not in transparent_layer_ids:
                    external_output_layers.add(layer_id)
                for oi, oname in enumerate(output_name_list):
                    write_elems = (
                        int(memory_writes[oi]) if oi < len(memory_writes) else 0
                    )
                    if layer_id in write_caps:
                        write_elems = min(write_elems, int(write_caps[layer_id]))
                    unique_external_outputs[oname] = max(
                        unique_external_outputs.get(oname, 0), write_elems
                    )
                    if write_elems > 0:
                        external_output_written.add(layer_id)
                    unique_external_output_bpe[oname] = max(
                        unique_external_output_bpe.get(oname, 0.0),
                        out_bpe[oi] if oi < len(out_bpe) else element_size,
                    )
                    # Raw basis too bills each output slot its OWN write
                    # count: charging the whole-layer output sum to every
                    # name duplicates multi-output writes (max(dim) values
                    # + indices) in the declared raw total (C23-P3).
                    raw_external_output_elems[oname] = (
                        raw_external_output_elems.get(oname, 0)
                        + int(write_elems))

            # Per-op fused elements: only non-intermediate DRAM traffic
            # (weights + model I/O)
            fused_elems = int(weight_read_elems + model_io_elems)

            _unattached = layer.get("norm_params_unattached")
            # ── Transposed convolutions: minimum-work bound (issue #44) ──
            # `macs` keeps the dense input-based convention (#31 / an earlier change); the
            # bound counts only padding-surviving taps and is reported as a
            # separate field.  Without exported conv_args it is not
            # representable and is reported as such, never as `macs`.
            min_work_fields: Dict[str, Any] = {}
            if op_type in _CONV_TRANSPOSE_TYPES:
                _conv_args = (layer.get("additional_info") or {}).get("conv_args")
                _in0 = input_shapes[0] if input_shapes and isinstance(input_shapes[0], list) else None
                _in1 = input_shapes[1] if len(input_shapes) > 1 and isinstance(input_shapes[1], list) else None
                _out0 = output_shapes[0] if output_shapes and isinstance(output_shapes[0], list) else None
                _mw = convtranspose_min_work_macs(_in0, _in1, _out0, _conv_args) if _conv_args else None
                if _mw is None:
                    min_work_fields["min_work"] = "not_representable"
                    min_work_not_representable += 1
                else:
                    min_work_fields["min_work_macs"] = int(_mw)
                    total_min_work_macs += int(_mw)
            else:
                total_min_work_macs += macs

            layers_out[layer_id] = {
                "type": op_type,
                **({"status": "incomplete", "capture_degraded": capture_degraded, "incomplete_reason": "capture_degraded"} if capture_degraded else {}),
                "einsum_equation": equation,
                "is_real_einsum": is_real_einsum,
                # issue #34: incomplete_reason is the cross-marker field every incomplete layer carries
                **({"status": "incomplete", "norm_params_unattached": str(_unattached), "incomplete_reason": str(_unattached)} if _unattached else {}),
                # converter-side refusal (state-consumer-p2-4): an input slot of an
                # expanded-node consumer could not be aligned with a graph
                # predecessor; the AF builder must not invent a weight for it
                **({"status": "incomplete", "incomplete_reason": layer.get("incomplete_reason") or "predecessor_slots_unaligned",
                    "predecessor_slots_unaligned": list(layer["predecessor_slots_unaligned"])}
                   if layer.get("predecessor_slots_unaligned") else {}),
                # weight-typed slots with no parameter provenance: their
                # numel is excluded from weight_elements and their read is
                # model I/O; the layer is incomplete (STATUS_CONTRACT rule 1;
                # a MAC-evidence reason below takes precedence).
                **({"weight_provenance_unresolved": list(weight_unresolved_slots),
                    "weight_elements_unresolved": int(weight_unresolved_elems),
                    "status": "incomplete",
                    "incomplete_reason": "weight_provenance_unresolved"}
                   if weight_unresolved_slots else {}),
                "macs": macs,
                # issue #58 (report provenance): what KIND of MACs the count holds -- a tensor-core contraction, or scalar
                # multiply statements the per-statement classifier counts as MACs in a non-einsum kernel -- so no reader
                # has to take an aggregate for homogeneous hardware MACs
                **({"macs_kind": ("contraction" if is_real_einsum else "scalar_statement_multiplies" if unit_basis == "per_statement" else "non_contraction")} if macs else {}),
                # issue #58: the converter's verdict that the exported equation does NOT describe the kernel (e.g. a
                # declared-dims fallback for a dependent-index contraction) travels into the analysis record; the count is
                # the loop-count witness either way (dsl_loop_count_evidence reports the flag, never gates on it)
                **({"equation_faithful": False} if layer.get("equation_faithful") is False else {}),
                "macs_dense": macs_dense,
                "mac_sparsity_fraction": sparsity_fraction,
                **({"compute_dtype": compute_dtype} if compute_dtype else {}),
                # the layer's MAC partition by compute dtype (one bucket per layer: a mixed kernel fails closed)
                **({"macs_by_compute_dtype": {compute_dtype: int(macs)}} if compute_dtype else {}),
                # the declared logical dtypes of the tensors the MAC statements multiply (DSL route): kept
                # beside a quantised binding and beside a mixed refusal; a workload-precision kernel carries
                # no compute-dtype field (its operands stay in the graph's tensor_dtypes)
                **({"compute_dtype_operands": compute_dtype_operands}
                   if compute_dtype_operands and (compute_dtype or _cd_fail == "compute_dtype_mixed") else {}),
                **({"scaled_mm": {"k_stored": int(ts.inputs[0][1]), "k_logical": int(ops_cost // (int(ts.inputs[0][0]) * int(ts.outputs[0][1]))) if ts.outputs and int(ts.inputs[0][0]) * int(ts.outputs[0][1]) else None,
                                  "scale_shapes": [list(s) for s in ts.inputs[2:4]], "scales_mac_bearing": False}}
                   if _is_scaled_mm and is_real_einsum and not macs_incomplete and len(ts.inputs) >= 2 and len(ts.inputs[0]) == 2 else {}),
                **({"macs_incomplete": True, "status": "incomplete",
                    "incomplete_reason": macs_incomplete_reason} if macs_incomplete else {}),
                **({"admission_error": _admission_error} if _admission_error else {}),
                **({"scaled_mm_rejection": scaled_mm_rejection} if scaled_mm_rejection else {}),
                **({"unrecognised_op": str(unrecognised_op)} if unrecognised_op else {}),
                **operand_evidence,
                **({
                    "macs_as_written_all_index_dense": as_written_dense,
                    "macs_dense_binary_schedule": binary_schedule["macs"],
                    "scalar_multiplies_dense_binary_schedule": binary_schedule["scalar_multiplies"],
                    "binary_schedule": binary_schedule["schedule"],
                    # issue #58: the schedule fields are labelled alternatives of the EXPORTED equation, never the selected
                    # cost basis; when that equation is not faithful to the kernel they describe nothing the kernel computes
                    "binary_schedule_applicability": ("descriptive_alternative_of_unfaithful_equation" if layer.get("equation_faithful") is False else "descriptive_alternative_not_selected_basis"),
                } if binary_schedule else {}),
                "other_ops": other_ops,
                "flops": flops,
                # issue #52: which count this DSL kernel's cost was SELECTED
                # from in the shared calculation (loop_iterations |
                # equation_product | shape_generic); absent on mechanical-route
                # layers.  The basis is tracked where ops_cost is chosen, so a
                # gathered MAC kernel billed from validated loop evidence says
                # loop_iterations (cost-basis-composition-1).
                **({"cost_basis": cost_basis} if cost_basis else {}),
                **({"unit_basis": unit_basis} if unit_basis else {}),
                **({"gather_index_statements": int(layer["gather_index_statements"])}
                   if isinstance(layer.get("gather_index_statements"), int) and not isinstance(layer.get("gather_index_statements"), bool) and layer.get("gather_index_statements") > 0 else {}),
                **({"singleton_scale_reads": int(layer["singleton_scale_reads"])}
                   if isinstance(layer.get("singleton_scale_reads"), int) and not isinstance(layer.get("singleton_scale_reads"), bool) and layer.get("singleton_scale_reads") > 0 else {}),
                **({"compute_density": compute_density}
                   if compute_density < 1.0 else {}),
                **({"loop_iterations_omitted": list(_loop_omitted)} if _loop_meta_incomplete else {}),
                # WITNESS_MODEL.v3 §2 (B′, an earlier change round 9): the narrow-access witness's coverage is passed through as
                # INFORMATIONAL fields only -- a declined witness changes no count, no completeness, no reason, no basis.
                **({"witness_coverage": layer["witness_coverage"]} if layer.get("witness_coverage") in ("full", "partial", "none") else {}),
                **({"witness_unmodelled": [str(n) for n in layer["witness_unmodelled"]]} if isinstance(layer.get("witness_unmodelled"), list) and layer.get("witness_unmodelled") else {}),
                **min_work_fields,
                **gather_extra_fields,
                **({"status": "incomplete", "incomplete_reason": layer_incomplete_reason} if layer_incomplete_reason else {}),
                **({"scalar_multiplies": einsum_scalar_multiplies} if einsum_scalar_multiplies else {}),
                # issue #46: an einsum with operands of unknown provenance is
                # costed from its shapes but reported incomplete.
                **({"status": "incomplete", "einsum_operands_unresolved": list(layer["einsum_operands_unresolved"]), "incomplete_reason": "einsum_operands_unresolved"}
                   if layer.get("einsum_operands_unresolved") else {}),
                "unfused_elements": unfused_elems,
                "orojenesis_elements": None,
                "fused_elements": fused_elems,
                "unfused_bytes": int(unfused_bytes),
                "intermediate_bytes": int(layer_intermediate_bytes),
                "bytes_per_element": {"inputs": in_bpe, "outputs": out_bpe},
                "tensor_shapes": {
                    "inputs": [s for s in input_shapes if isinstance(s, list)],
                    "outputs": [s for s in output_shapes if isinstance(s, list)],
                },
                "tensor_sizes": {
                    "inputs": input_sizes,
                    "outputs": output_sizes,
                },
                "memory_elements": {
                    "inputs": [0] * len(memory_reads) if is_alias else memory_reads,
                    "outputs": [0] * len(memory_writes) if is_alias else memory_writes,
                },
                # the converter's parameter identity per input slot, when recorded
                **({"parameter_sources": {"inputs": list(input_source_list)}} if has_parameter_sources else {}),
                "tensor_types": {
                    "inputs": list(input_type_list),
                    "outputs": list(output_type_list),
                },
                "input_elements": input_elems,
                "output_elements": output_elems,
                "intermediate_elements": layer_intermediate_elems,
                "weight_elements": weight_elems,
                "weight_read_elements": weight_read_elems,
                "model_io_elements": model_io_elems,
                "model_output_elements": int(model_output_elems),
                **({"nonreturned_output_elements": int(output_elems)} if nonreturned_store else {}),
                "input_is_intermediate": input_is_intermediate,
                "output_is_intermediate": output_is_intermediate,
                "is_orphaned": layer_id in _orphaned_layers,
                "connections": {"inputs": input_layer_ids, "outputs": output_layer_ids},
                **({"alias_qualification": alias_qualification} if alias_qualification else {}),
                **({"call_operands": dict(layer["call_operands"])} if isinstance(layer.get("call_operands"), dict) else {}),
                **({"status": "incomplete", "capture_incomplete": capture_incomplete,
                    "incomplete_reason": incomplete_reason or "capture_incomplete"} if capture_incomplete else {}),
            }

            total_macs += macs
            if compute_dtype:
                macs_by_compute_dtype[compute_dtype] = macs_by_compute_dtype.get(compute_dtype, 0) + int(macs)
            if macs:                                                       # issue #58: typed components of the aggregate
                if is_real_einsum:
                    _bucket = compute_dtype or "workload_precision"
                    contraction_macs_by_dtype[_bucket] = contraction_macs_by_dtype.get(_bucket, 0) + int(macs)
                elif unit_basis == "per_statement":
                    scalar_statement_mac_layers[layer_id] = int(macs)
                else:
                    non_contraction_other_mac_layers[layer_id] = int(macs)
            if layer.get("equation_faithful") is False:
                equation_unfaithful_layers.append(layer_id)
            if macs_incomplete:
                macs_incomplete_layers.append(layer_id)
            elif "operand_evidence_problems" in operand_evidence:
                operand_evidence_warning_layers.append(layer_id)
            if binary_schedule:
                multi_operand_einsum_layers.append(layer_id)
                total_macs_dense_binary_schedule += int(binary_schedule["macs"])
                total_scalar_multiplies_binary_schedule += int(binary_schedule["scalar_multiplies"])
            else:
                total_macs_dense_binary_schedule += int(macs_dense)
            total_other_ops += other_ops
            total_flops += flops
            if cost_basis:
                dsl_cost_basis_layers.setdefault(cost_basis, []).append(layer_id)
            total_unfused_elems += unfused_elems
            total_intermediate_elems += layer_intermediate_elems
            total_unfused_bytes += unfused_bytes
            total_intermediate_bytes += layer_intermediate_bytes
            total_weight_elems += weight_elems
            total_weight_read_elems += weight_read_elems
            if weight_unresolved_slots:
                total_weight_unresolved_elems += int(weight_unresolved_elems)
                weight_provenance_unresolved_layers.append(layer_id)

        # Deduplicated graph-level external I/O.  Per canonical base tensor,
        # DRAM reads are the unique element footprint of all accesses: the
        # exact union of statically proven access regions, with unproven
        # reads folded in via the conservative max() lower bound, capped at
        # the base tensor size.  Used for both fused and fused_prefetched
        # totals.
        # Declared model outputs not charged above: produced by an op
        # torchview did not trace (in-place ``out[idx] = v``), or an
        # intermediate that is also returned. They still have to be
        # written once at their declared shape and dtype.
        for k, mo in enumerate(model_outputs):
            op = mo.get("op")
            shape = mo.get("shape")
            if mo.get("is_input") or not isinstance(shape, list):
                # A model input returned after in-place updates: only the
                # traced writes into it count, never the whole buffer.
                continue
            # Skip only if some layer actually wrote this output: a view or
            # slice at the end of an untraced chain is "external" but writes
            # nothing, so the declared output would otherwise be lost.
            src = op
            if op in transparent_layer_ids:
                src = _trace_source_through_views(op)
            if op in external_output_written or src in external_output_written:
                continue
            elems = _product(shape)
            if op in write_caps:
                elems = min(elems, int(write_caps[op]))
            if elems <= 0:
                continue
            key = f"declared_output_{k}"
            bpe = _dtype_bytes(mo.get("dtype"), element_size)
            unique_external_outputs[key] = elems
            unique_external_output_bpe[key] = bpe
            # The write happens in every execution model, so the unfused
            # totals carry it too (keeps fused <= unfused).
            total_unfused_elems += elems
            total_unfused_bytes += elems * bpe
            if self.debug:
                logger.debug(f"declared output {k} ({op}) not produced by a traced "
                      f"external write; charging {elems} elements")

        # Returned outputs the capture could not bind to a traced tensor
        # (issue #53): their footprint is charged above from the declared
        # shape/dtype, but the work that produced them is untraced, so the
        # accounting is NOT qualified -- fail closed (status incomplete on
        # the total; the perf model reports it unqualified).
        unbound_returned_outputs: Dict[str, str] = {}
        for k, mo in enumerate(model_outputs):
            if mo.get("unbound_returned"):
                unbound_returned_outputs[f"declared_output_{k}"] = "returned_output_unbound"
        # return-hook-error-2: the capture could not record the returned
        # outputs at all -- the external-output footprint is unknown; fail closed.
        if graph.get("returned_outputs_unrecorded"):
            unbound_returned_outputs["returned_outputs"] = "returned_outputs_unrecorded"

        group_footprints = _group_footprints(
            external_read_groups, start_output_sizes, debug=self.debug
        )
        unique_external_input_elems = sum(group_footprints.values())
        total_fused_prefetched_elems = int(
            unique_external_input_elems
            + sum(unique_external_outputs.values())
        )
        # Same footprints priced at each tensor's own dtype width.
        unique_external_input_bytes = sum(
            fp * float(external_read_groups[name].get("bpe") or element_size)
            for name, fp in group_footprints.items()
        )
        unique_external_output_bytes = sum(
            elems * unique_external_output_bpe.get(name, element_size)
            for name, elems in unique_external_outputs.items()
        )
        total_fused_bytes = int(unique_external_input_bytes + unique_external_output_bytes)
        # fused_elements == fused_prefetched_elements (same dedup logic)
        total_fused_elems = total_fused_prefetched_elems

        # model_io_elements / weight_elements: raw per-op sums (may double-
        # count shared inputs).  Kept for diagnostic / per-layer inspection;
        # weight_elements feeds perf's memory_breakdown.
        total_model_io_elems = sum(
            layer.get("model_io_elements", 0)
            for layer in layers_out.values()
        )

        # ── Boundary STORAGE billing at declared widths (opt-in) ──
        # The campaign audit confirmed two defect classes from the single
        # per-graph byte scalar: BF16 public I/O billed at fp8's 1 byte
        # (Quant/001) and FP32 I/O billed at fp16's 2 bytes (L1/007).
        boundary_storage: Optional[Dict[str, Any]] = None
        model_io_bytes_declared: Optional[int] = None
        external_io_bytes_declared_dedup: Optional[int] = None
        boundary_declared: Optional[Dict[str, int]] = None
        if boundary_storage_dtypes is not None:
            boundary_storage = {}
            unresolved = []
            raw_total = 0.0
            dedup_total = 0.0
            # Canonical base -> resolved width already applied to its read
            # group: each group's footprint is billed and width-corrected
            # exactly ONCE across all public aliases; aliases of one
            # stored tensor must agree on the width (review C03-P1).
            group_width_billed: Dict[str, float] = {}
            # What the declared totals COVER (for the perf model's re-pricing
            # delta): every boundary tensor, weight-classified inputs
            # included — since an earlier decision a declared boundary input the
            # converter typed `weight` sits in weight_elements, not in
            # model_io_elements, so a delta taken against the model-I/O class
            # alone double-counts it (observed: +32 B on the fixture).
            covered_raw_elems = 0
            covered_dedup_elems = 0
            covered_weight_raw_elems = 0
            declared_raw_model_io = 0.0
            declared_raw_weight = 0.0
            # The parameter FOOTPRINT of declared weight inputs, on its own
            # basis (distinct parameter source per consuming op x full numel,
            # exactly what weight_elements counts) at the declared width —
            # never the raw read count (a parameter in two slots is read
            # twice; an indexed table is read partially) (M97-WEIGHT-2).
            covered_weight_footprint_elems = 0
            declared_footprint_weight = 0.0
            names = sorted(set(unique_external_inputs)
                           | set(unique_external_outputs))
            for name in names:
                declared = boundary_storage_dtypes.get(name)
                has_binding = (name in unique_external_inputs
                               and boundary_input_binding is not None
                               and name in boundary_input_binding)
                # A SUPPLIED explicit binding is authoritative for its
                # input and is validated FIRST: a graph alias may collide
                # with another declared input's name, and direct-name
                # matching would then silently bill the wrong declaration
                # and silently ignore the caller's mapping (review
                # C19-P2).  An invalid binding refuses; it is never
                # ignored.  Without a binding entry for the name, the
                # direct-name resolution below is unchanged.
                if has_binding:
                    bound = boundary_input_binding[name]
                    if bound in boundary_storage_dtypes:
                        # Explicit trace-to-definition input binding: the
                        # authoritative declared storage survives
                        # public-input aliases (C12-P1).
                        declared = boundary_storage_dtypes[bound]
                        source = f"definition(bound:{name}->{bound})"
                    else:
                        # Dangling binding: refuses REGARDLESS of the
                        # tensor's name or recorded origin — it never
                        # falls through to trace billing (C19-P1).
                        unresolved.append(
                            f"{name} (dangling boundary_input_binding to "
                            f"{bound!r}: not a declared definition input "
                            "— refused regardless of tensor name)")
                        continue
                elif declared is not None:
                    source = "definition"
                elif name in unique_external_outputs and declared_output_storage:
                    # Graph output tensors carry DSL kernel names (e.g.
                    # "scale#0"), not the definition's output names.  Bind
                    # by resolved element count: admissible only when every
                    # size-matching declared output agrees on ONE width —
                    # otherwise the mapping is ambiguous and we refuse
                    # (never pick heuristically).
                    elems = unique_external_outputs.get(name, 0)
                    cands = [o for o in declared_output_storage
                             if int(o.get("elements", -1)) == int(elems)]
                    widths = {_STORAGE_WIDTHS.get(str(o["dtype"]).lower())
                              for o in cands}
                    if cands and len(widths) == 1 and None not in widths:
                        declared = str(cands[0]["dtype"])
                        source = ("definition-output(size-matched: "
                                  + ",".join(str(o["name"]) for o in cands)
                                  + ")")
                    elif cands:
                        unresolved.append(
                            f"{name} (ambiguous declared-output widths "
                            f"{sorted(str(o['dtype']) for o in cands)})")
                        continue
                    else:
                        declared = graph_tensor_dtypes.get(name)
                        source = "trace"
                elif (name in unique_external_inputs
                        and graph_tensor_origins.get(name)
                        == "internal-constant"):
                    # The graph RECORDS this tensor as internally created
                    # by the trace (propagated provenance, not a name
                    # spelling): its recorded trace dtype is admissible
                    # storage evidence.
                    declared = graph_tensor_dtypes.get(name)
                    source = "trace(internal-constant)"
                elif name in unique_external_inputs:
                    # PUBLIC (or provenance-unrecorded) external input with
                    # no definition name and no explicit binding: its
                    # trace/verification dtype is NOT storage evidence
                    # (C12-P1), and a __<role>_<n> name spelling is not
                    # internal-constant provenance (C19-P1) — refuse,
                    # never guess.
                    unresolved.append(
                        f"{name} (public input: no definition dtype, no "
                        "explicit boundary_input_binding to a declared "
                        "input, and no recorded internal-constant "
                        "provenance in the graph's tensor_origins — "
                        "verification/trace dtype is not public storage "
                        "evidence)")
                    continue
                else:
                    declared = graph_tensor_dtypes.get(name)
                    source = "trace"
                width = _STORAGE_WIDTHS.get(
                    str(declared).lower()) if declared else None
                if width is None:
                    unresolved.append(f"{name} (dtype={declared!r})")
                    continue
                raw_elems = (raw_external_input_elems.get(name, 0)
                             + raw_external_output_elems.get(name, 0))
                # Dedup INPUT elements are the canonical access-region
                # union footprint — the same proven basis as the fused
                # byte model (disjoint slices sum, overlaps count once,
                # capped at the base size).  A per-name max silently drops
                # disjoint reads of the same tensor (C23-P2).  Boundary
                # names may be public VIEW ALIASES of the canonical base
                # the groups are keyed by, so the group is reached through
                # the recorded alias association, never a name-equality
                # lookup (C03-P1), and is billed once across its aliases.
                in_dedup = 0
                in_w = float(element_size)
                if name in unique_external_inputs:
                    canon = external_input_canonical.get(name, name)
                    if canon == "":
                        unresolved.append(
                            f"{name} (reads resolve to conflicting "
                            "canonical base tensors — the width "
                            "correction target is ambiguous)")
                        continue
                    in_w = float((external_read_groups.get(canon) or {}).get(
                        "bpe") or element_size)
                    billed = group_width_billed.get(canon)
                    if billed is None:
                        group_width_billed[canon] = width
                        in_dedup = group_footprints.get(
                            canon, unique_external_inputs.get(name, 0))
                    elif billed != width:
                        unresolved.append(
                            f"{name} (conflicting declared widths for "
                            f"canonical base {canon!r}: {billed} vs "
                            f"{width} bytes/element — aliases of one "
                            "stored tensor must agree)")
                        continue
                    # else: the group's footprint and width correction
                    # were already billed via another alias (in_dedup 0).
                dedup_elems = in_dedup + unique_external_outputs.get(name, 0)
                raw_total += raw_elems * width
                dedup_total += dedup_elems * width
                covered_raw_elems += raw_elems
                covered_dedup_elems += dedup_elems
                if name in weight_boundary_names:
                    declared_raw_weight += raw_elems * width
                    covered_weight_raw_elems += raw_elems
                    fp = int(weight_boundary_footprint.get(name, 0))
                    declared_footprint_weight += fp * width
                    covered_weight_footprint_elems += fp
                else:
                    declared_raw_model_io += raw_elems * width
                # Resolve the per-tensor-dtype byte totals at the SAME
                # authoritative width (C23-P1): the boundary component of
                # every traffic class is re-priced from its recorded trace
                # width to the resolved width on its own basis — raw
                # fan-out sums for the unfused totals, canonical dedup
                # footprints for the fused totals.  Trace-sourced widths
                # resolve to the recorded width (delta 0); weights,
                # intermediates and compute rates are untouched.
                out_w = float(unique_external_output_bpe.get(
                    name, element_size))
                total_unfused_bytes += (
                    (width - in_w) * raw_external_input_elems.get(name, 0)
                    + (width - out_w) * raw_external_output_elems.get(name, 0))
                unique_external_input_bytes += (width - in_w) * in_dedup
                unique_external_output_bytes += (
                    (width - out_w) * unique_external_outputs.get(name, 0))
                boundary_storage[name] = {
                    "dtype": str(declared), "bytes_per_element": width,
                    "source": source, "elements_raw": int(raw_elems),
                    "elements_dedup": int(dedup_elems),
                }
            # Every external read group the analyzer classified must take
            # part in the declared totals, not only the NAMED unique inputs:
            # a gather index recovered from the call attributes has no tensor
            # name, so it never enters unique_external_inputs, and skipping
            # it silently made the declared raw/dedup totals SMALLER than the
            # undeclared traffic (M97-INTEG-D09-1, composition with an earlier change).  A
            # recovered index cannot be bound to a definition name; its only
            # admissible width evidence is the dtype the attributes recorded
            # for it (the same evidence that priced its undeclared read).
            # Unknown dtype -> billed at the trace width it already carries
            # AND marked incomplete on its layer (never a silent smaller
            # total; STATUS_CONTRACT rule 1), the total block and perf copy
            # the marker.
            named_canonicals = {external_input_canonical.get(n, n) for n in unique_external_inputs}
            for canon, group in external_read_groups.items():
                if canon in group_width_billed or canon in named_canonicals or not group.get("unnamed"):
                    continue
                g_dtype = group.get("dtype")
                g_width = _STORAGE_WIDTHS.get(str(g_dtype).lower().replace("torch.", "")) if g_dtype else None
                trace_w = float(group.get("bpe") or element_size)
                g_raw = int(sum(group.get("counted") or []))
                g_dedup = int(group_footprints.get(canon, g_raw))
                if g_width is None:
                    g_width = trace_w
                    g_source = "trace(recovered-index; dtype unrecorded -> width unresolved)"
                    lid = group.get("layer")
                    if lid in layers_out:
                        # rule 1: status + a reason; an earlier reason on the
                        # layer (e.g. gather_index_dtype_unknown) is kept, the
                        # boundary detail field is added either way
                        layers_out[lid]["status"] = "incomplete"
                        layers_out[lid].setdefault("incomplete_reason", "boundary_width_unresolved")
                        layers_out[lid]["boundary_width_unresolved"] = [canon]
                else:
                    g_source = "trace(recovered-index)"
                group_width_billed[canon] = g_width
                raw_total += g_raw * g_width
                dedup_total += g_dedup * g_width
                covered_raw_elems += g_raw
                covered_dedup_elems += g_dedup
                declared_raw_model_io += g_raw * g_width
                total_unfused_bytes += (g_width - trace_w) * g_raw
                unique_external_input_bytes += (g_width - trace_w) * g_dedup
                boundary_storage[canon] = {
                    "dtype": str(g_dtype), "bytes_per_element": g_width,
                    "source": g_source, "elements_raw": g_raw,
                    "elements_dedup": g_dedup,
                }
            if unresolved:
                raise ValueError(
                    "boundary storage dtype cannot be established for "
                    f"{unresolved}: declared definition dtypes and recorded "
                    "trace dtypes are the only admissible evidence — "
                    "storage types are never silently invented")
            model_io_bytes_declared = int(raw_total)
            external_io_bytes_declared_dedup = int(dedup_total)
            boundary_declared = {
                "elements_raw": int(covered_raw_elems),
                "elements_dedup": int(covered_dedup_elems),
                "elements_raw_weight": int(covered_weight_raw_elems),
                "bytes_raw_model_io": int(declared_raw_model_io),
                "bytes_raw_weight": int(declared_raw_weight),
                # parameter footprint of the declared weight inputs (what
                # perf's weight_bytes is), separate from the raw reads above
                "footprint_elements_weight": int(covered_weight_footprint_elems),
                "bytes_footprint_weight": int(declared_footprint_weight),
            }
            # The fused byte total is the external I/O byte sum, now at
            # resolved widths — recompute so the per-tensor-dtype fields
            # and the declared dedup field agree (C23-P1/P2).
            total_fused_bytes = int(
                unique_external_input_bytes + unique_external_output_bytes)

        analysis: Dict[str, Any] = {
            "layers": layers_out,
            "total": {
                "num_layers": len(layers_out),
                "num_start_nodes_filtered": len(start_node_ids),
                **({"status": "incomplete", "incomplete_layers": {
                        **{lid: (l.get("incomplete_reason") or "incomplete") for lid, l in layers_out.items() if l.get("status") == "incomplete"},
                        **unbound_returned_outputs}}
                   if (any(l.get("status") == "incomplete" for l in layers_out.values()) or unbound_returned_outputs) else {}),
                **({"returned_outputs_unbound": [
                        {"key": key, "tensor": mo.get("tensor"), "path": mo.get("returned_path"),
                         "shape": mo.get("shape"), "dtype": mo.get("dtype"),
                         # return-region-5: one declared output per storage,
                         # standing for every unbound leaf of that storage at the
                         # union of their byte regions
                         **({"paths": mo.get("returned_paths")} if mo.get("returned_paths") else {}),
                         **({"footprint_bytes": mo.get("footprint_bytes"),
                             "footprint_method": mo.get("footprint_method")} if mo.get("footprint_bytes") is not None else {})}
                        for key, mo in ((f"declared_output_{k}", mo) for k, mo in enumerate(model_outputs))
                        if mo.get("unbound_returned")]}
                   if unbound_returned_outputs else {}),
                "macs": int(total_macs),
                **({"min_work_macs": int(total_min_work_macs)} if min_work_not_representable == 0
                   else {"min_work": "not_representable",
                         "min_work_not_representable_layers": int(min_work_not_representable)}),
                # MACs whose operands were recorded in a quantised dtype (nvfp4 /
                # fp8): the perf model prices them at that dtype's tensor-core
                # rate; the remainder keeps the workload precision.
                **({"macs_by_compute_dtype": {k: int(v) for k, v in sorted(macs_by_compute_dtype.items())}} if macs_by_compute_dtype else {}),
                # issue #58 (report provenance): the aggregate ``macs`` decomposed into typed components that sum to it --
                # a reader must never present the aggregate as homogeneous hardware MACs (e.g. 881,134,080 = 880,803,840
                # nvfp4 contraction + 327,680 + 2,560 scalar multiply statements).  Report-only; no count changes.
                "macs_typed": {
                    "aggregate": int(total_macs),
                    "contraction": {"macs": int(sum(contraction_macs_by_dtype.values())),
                                    "by_compute_dtype": {k: int(v) for k, v in sorted(contraction_macs_by_dtype.items())}},
                    "scalar_statement_multiplies": {"macs": int(sum(scalar_statement_mac_layers.values())),
                                                    "layers": {k: int(v) for k, v in sorted(scalar_statement_mac_layers.items())},
                                                    "note": "multiply statements of non-einsum kernels counted as MACs by the per-statement classifier (unit_basis per_statement); not tensor-core contraction work"},
                    **({"non_contraction_other": {"macs": int(sum(non_contraction_other_mac_layers.values())),
                                                  "layers": {k: int(v) for k, v in sorted(non_contraction_other_mac_layers.items())}}}
                       if non_contraction_other_mac_layers else {}),
                    "components_sum_to_aggregate": int(sum(contraction_macs_by_dtype.values()) + sum(scalar_statement_mac_layers.values()) + sum(non_contraction_other_mac_layers.values())) == int(total_macs),
                },
                # issue #58: layers whose exported equation is not faithful to the kernel (parallel to
                # multi_operand_einsum_layers); the schedule totals below are descriptive alternatives of the exported
                # equations and never the selected cost basis -- for such layers they describe nothing the kernel computes
                "equation_unfaithful_layers": list(equation_unfaithful_layers),
                "binary_schedule_applicability": ("descriptive_alternative_not_selected_basis; equation_unfaithful_layers describe nothing the kernel computes" if equation_unfaithful_layers else "descriptive_alternative_not_selected_basis"),
                # F6 / R5 evidence fields.  `macs` stays the as-written single
                # contraction (times structured sparsity); the dense binary
                # schedule total is the labelled alternative, not a replacement.
                "macs_dense_binary_schedule": int(total_macs_dense_binary_schedule),
                "scalar_multiplies_dense_binary_schedule": int(total_scalar_multiplies_binary_schedule),
                "multi_operand_einsum_layers": multi_operand_einsum_layers,
                "macs_incomplete_layers": macs_incomplete_layers,
                "operand_evidence_warning_layers": operand_evidence_warning_layers,
                "other_ops": int(total_other_ops),
                "flops": int(total_flops),
                **({"dsl_cost_basis_layers": {k: list(v) for k, v in sorted(dsl_cost_basis_layers.items())}}
                   if dsl_cost_basis_layers else {}),
                "unfused_elements": int(total_unfused_elems),
                "orojenesis_elements": None,
                "fused_elements": int(total_fused_elems),
                "fused_prefetched_elements": total_fused_prefetched_elems,
                "weight_elements": int(total_weight_elems),
                "weight_read_elements": int(total_weight_read_elems),
                **({"weight_elements_unresolved": int(total_weight_unresolved_elems),
                    "weight_provenance_unresolved_layers": list(weight_provenance_unresolved_layers)}
                   if weight_provenance_unresolved_layers else {}),
                "model_io_elements": int(total_model_io_elems),
                "intermediate_elements": int(total_intermediate_elems),
                # Byte totals at per-tensor dtype widths (the perf model
                # prefers these over elements x bytes_per_element).
                "fused_bytes": total_fused_bytes,
                "fused_prefetched_bytes": total_fused_bytes,
                "unfused_bytes": int(total_unfused_bytes),
                "intermediate_bytes": int(total_intermediate_bytes),
                "external_input_bytes": int(unique_external_input_bytes),
                "external_output_bytes": int(unique_external_output_bytes),
                # non-returned consumer-less stores (internal; not boundary traffic)
                "nonreturned_store_bytes": int(nonreturned_store_bytes),
                "nonreturned_store_layers": list(nonreturned_store_layers),
                "num_intermediate_tensors": len(intermediate_tensors),
                "num_orphaned_layers": len(_orphaned_layers),
                # The INTEG-9 status spread appears ONCE in this block (above, next to
                # num_start_nodes_filtered); an earlier landing had merged a second copy here.
                **({"status": "incomplete", "capture_degraded": capture_degraded} if capture_degraded else {}),
                **({"model_io_bytes_declared": model_io_bytes_declared,
                    "external_io_bytes_declared_dedup":
                        external_io_bytes_declared_dedup,
                    "boundary_declared": boundary_declared}
                   if model_io_bytes_declared is not None else {}),
            },
            "metadata": {
                "precision": precision,
                "bytes_per_element": element_size,
                "bytes_accounting": "per-tensor-dtype",
                "source_graph": str(graph_path),   # issue #63: the selected input graph path, never a per-layer / returned-output local
                **({"capture_capability": dict(capture_capability)} if capture_capability else {}),
                **({"boundary_storage": boundary_storage}
                   if boundary_storage is not None else {}),
            },
        }

        out_path = out_dir / "analysis.yaml"
        with open(out_path, "w") as f:
            yaml.dump(analysis, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            logger.debug(f"✅ Wrote analysis: {out_path}")

        return analysis

    # Maps metadata orig_dtypes keywords to Solar precision names
    _QUANT_DTYPE_MAP = {
        "nvfp4": "nvfp4",
        "float4_e2m1fn_x2": "nvfp4",
        "fp8": "fp8",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
    }

    def _resolve_quant_precision(self, einsum_graph_path: Path) -> Optional[str]:
        """Search for metadata.yaml near the einsum graph and return quant precision.

        Walks up from the einsum_graph_path looking for metadata.yaml
        (max 3 levels). Picks highest-throughput quant dtype (nvfp4 > fp8).
        """
        search_dir = einsum_graph_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        meta = yaml.safe_load(f) or {}
                except Exception:
                    return None

                best = None
                for conv in meta.get("dtype_conversions") or []:
                    orig = str(conv.get("orig_dtypes", "")).lower()
                    for keyword, prec in self._QUANT_DTYPE_MAP.items():
                        if keyword in orig:
                            if best is None or BYTES_PER_ELEMENT.get(prec, 99) < BYTES_PER_ELEMENT.get(best, 99):
                                best = prec
                            break
                return best
            search_dir = search_dir.parent
        return None


__all__ = ["EinsumGraphAnalyzer"]
