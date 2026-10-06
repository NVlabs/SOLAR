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

Note: "start" nodes are filtered out before analysis as they represent model inputs,
not actual computation. Their outputs are treated as external inputs to the graph.

See SOL_GUIDE.md for detailed explanation of the three SOL models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import math
import re

import yaml

from solar.einsum import EinsumAnalyzer
from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.types import TensorShapes
from solar.common.utils import ensure_directory, NoAliasDumper, yaml_safe_load
from solar.analysis.access_regions import (
    PARTITION_OPS,
    SLICE_VIEW_OPS,
    Box,
    box_size,
    partition_output_box,
    slice_op_boxes,
    union_size,
)


PathLike = Union[str, Path]


def _product(shape: List[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


def _layer_raw_attributes(layer: Dict[str, Any]) -> Any:
    raw = layer.get("raw_attributes")
    if raw is None:
        raw = (layer.get("module_args") or {}).get("raw_attributes")
    return raw


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
                    pos = [i for i, t in enumerate(toks) if t == "T"]
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
        boxes: Optional[List[Box]] = None
        if op_type in SLICE_VIEW_OPS and input_index == 0:
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
                print(f"Debug: external tensor '{name}' has conflicting full "
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
                print(
                    f"Debug: external tensor '{name}' has conflicting full "
                    f"sizes {sorted(full_candidates)}; using conservative bound"
                )
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


class EinsumGraphAnalyzer:
    """Analyze `einsum_graph.yaml` and write `analysis.yaml`."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)

    def analyze_graph(
        self,
        einsum_graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Analyze an einsum graph and write `analysis.yaml`.

        Args:
            einsum_graph_path: Path to `einsum_graph.yaml`.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the einsum graph into output dir using the
                canonical name `einsum_graph.yaml`.

        Returns:
            Analysis dict, or None on failure.
        """
        src = Path(einsum_graph_path)
        # Stage 2.5 fallback: prefer einsum_graph_reordered.yaml if present in the same dir
        reordered = src.parent / "einsum_graph_reordered.yaml"
        if src.name == "einsum_graph.yaml" and reordered.exists():
            if self.debug:
                print(f"Debug: using reordered graph {reordered}")
            src = reordered
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self.debug:
                print(f"Debug: einsum graph not found: {src}")
            return None

        try:
            with open(src) as f:
                graph = yaml_safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                print(f"Debug: failed reading einsum graph: {exc}")
            return None

        if copy_graph:
            try:
                dst = out_dir / "einsum_graph.yaml"
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            except Exception:
                if self.debug:
                    print("Debug: failed to copy einsum_graph.yaml")

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
        element_size = BYTES_PER_ELEMENT.get(precision, 4)

        # Override precision/element_size from quant metadata if available
        quant_precision = self._resolve_quant_precision(src)
        if quant_precision:
            element_size = BYTES_PER_ELEMENT.get(quant_precision, element_size)
            precision = quant_precision
            if self.debug:
                print(f"  Quant override: precision={precision}, bytes_per_element={element_size}")

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
            print(f"Debug: Filtered out {len(start_node_ids)} start nodes")
            if bool_start_node_ids:
                print(f"Debug: Found {len(bool_start_node_ids)} bool-typed start nodes: {bool_start_node_ids}")
            print(f"Debug: Analyzing {len(layers_in)} computation nodes")

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
            print(f"Debug: structured-sparsity MAC fractions: {mac_fraction}")
        external_output_layers: Set[str] = set()
        external_output_written: Set[str] = set()   # ... with a non-zero DRAM write

        if self.debug:
            print(f"Debug: Found {len(intermediate_tensors)} intermediate tensors")
            for t in sorted(intermediate_tensors)[:10]:
                print(f"  - {t}")
            if transparent_layer_ids:
                print(f"Debug: {len(transparent_layer_ids)} transparent view layers")

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
                print(f"Debug: Skipping memory for {len(_bool_layers)} bool-derived layers: {sorted(_bool_layers)}")

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
        total_flops = 0         # 2 * total_macs
        total_other_ops = 0     # CUDA-core elementwise/reduction ops
        total_unfused_elems = 0       # Σ (all input + output elems) per op
        total_intermediate_elems = 0  # Σ intermediate activation elems

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
        total_unfused_bytes = 0.0
        total_intermediate_bytes = 0.0

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

            ops_cost = 0
            try:
                if is_real_einsum and equation:
                    ops_cost = int(
                        self.einsum_analyzer.get_compute_cost(op_type, ts, equation=equation)
                    )
                else:
                    ops_cost = int(self.einsum_analyzer.get_compute_cost(op_type, ts))
            except Exception:
                ops_cost = 0

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
            if op_type in _ZERO_COMPUTE_OPS:
                ops_cost = 0
                is_real_einsum = False

            macs_dense = ops_cost if is_real_einsum else 0
            sparsity_fraction = mac_fraction.get(layer_id, 1.0)
            if is_real_einsum:
                macs = int(round(ops_cost * sparsity_fraction))
                other_ops = 0
            else:
                macs = 0
                other_ops = ops_cost

            flops = int(2 * macs)

            input_shapes = tensor_shapes.get("inputs") or []
            output_shapes = tensor_shapes.get("outputs") or []
            input_type_list = tensor_types.get("inputs") or []
            output_type_list = tensor_types.get("outputs") or []

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

            # For embedding (table lookup), only the gathered rows are read
            # from the weight matrix, not the entire vocabulary table.
            # Input shapes are [indices_shape, weight_shape] where weight is
            # [vocab_size, embedding_dim].  The actual DRAM read is just the
            # rows selected by indices — which equals the output shape
            # [batch, seq, embedding_dim].  Use min(input, output) to handle
            # both small token counts (gathered rows << full table) and large
            # token counts (most/all rows accessed, full table is the bound).
            if op_type in ("embedding", "embedding_bag"):
                total_output = sum(output_sizes)
                gathered = min(sum(input_sizes), total_output)
                memory_reads = [0] * len(input_sizes)
                if input_sizes:
                    memory_reads[-1] = gathered
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

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
            elif op_type in _SLICE_VIEW_OPS:
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

            # ── Step 3: Derive totals from corrected per-tensor counts ──
            input_elems = int(sum(memory_reads))
            output_elems = int(sum(memory_writes))
            unfused_elems = input_elems + output_elems

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
            input_bytes = sum(r * b for r, b in zip(memory_reads, in_bpe))
            output_bytes = sum(w * b for w, b in zip(memory_writes, out_bpe))
            unfused_bytes = input_bytes + output_bytes

            # ── Step 4: Classify inputs as external vs graph-internal ──
            # Uses memory_reads (already corrected) so no re-scanning needed.
            # Classify each input tensor:
            #   - graph-internal  → intermediate (fusable, skip in fused model)
            #   - other           → external: weights and model inputs (DRAM read)
            #
            # graph-internal = produced by a non-view op in the graph.
            # Transparent views are traced back to their source. The tensor
            # type is not consulted: an operand in a weight role can still be
            # computed in the graph, e.g. F.linear(x, w * 2).
            input_name_list = tensor_names.get("inputs") or []
            graph_internal_input_elems = 0   # intermediate activations from other ops
            external_input_elems = 0         # weights + model-level inputs (always DRAM)
            graph_internal_input_bytes = 0.0
            external_input_bytes = 0.0

            for i, mem_read in enumerate(memory_reads):
                if mem_read <= 0:
                    continue
                iname = input_name_list[i] if i < len(input_name_list) else ""

                if iname in tensor_producers:
                    producer_id = tensor_producers[iname]
                    source_id = _trace_source_through_views(producer_id)
                    is_graph_internal = source_id in all_layer_ids and source_id not in transparent_layer_ids
                elif declared_output_tensors and (
                    iname in declared_output_tensors
                    or _canonical_external_tensor(
                        iname, tensor_producers, layers_in, transparent_layer_ids
                    ) in declared_output_tensors
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
                    if iname:
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

            intermediate_input_elems = int(graph_internal_input_elems)
            model_input_elems = int(external_input_elems)
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
            if (not output_is_intermediate and (model_output_ops or model_outputs)
                    and not _reaches_model_output(layer_id)):
                # Consumer-less result that is not a declared model output:
                # dead code, or consumed by an op torchview does not trace
                # (in-place ``result[mask] = v``). A fused kernel never
                # writes it to DRAM, so it is intermediate traffic.
                output_is_intermediate = True

            # Intermediate output elems: written to cache (fused) not DRAM
            intermediate_output_elems = output_elems if output_is_intermediate else 0
            # Total intermediate elems for this layer (inputs + outputs)
            layer_intermediate_elems = intermediate_input_elems + intermediate_output_elems
            layer_intermediate_bytes = graph_internal_input_bytes + (
                output_bytes if output_is_intermediate else 0.0)

            # Model output elems: final graph outputs that must go to DRAM
            model_output_elems = output_elems if not output_is_intermediate else 0
            # Per-op model I/O: external inputs + model outputs (no intermediates)
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

            # Per-op fused elements: only non-intermediate DRAM traffic
            fused_elems = int(model_io_elems)

            layers_out[layer_id] = {
                "type": op_type,
                "einsum_equation": equation,
                "is_real_einsum": is_real_einsum,
                "macs": macs,
                "macs_dense": macs_dense,
                "mac_sparsity_fraction": sparsity_fraction,
                "other_ops": other_ops,
                "flops": flops,
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
                    "inputs": memory_reads,
                    "outputs": memory_writes,
                },
                "tensor_types": {
                    "inputs": list(input_type_list),
                    "outputs": list(output_type_list),
                },
                "input_elements": input_elems,
                "output_elements": output_elems,
                "intermediate_elements": layer_intermediate_elems,
                "model_io_elements": model_io_elems,
                "input_is_intermediate": input_is_intermediate,
                "output_is_intermediate": output_is_intermediate,
                "is_orphaned": layer_id in _orphaned_layers,
                "connections": {"inputs": input_layer_ids, "outputs": output_layer_ids},
            }

            total_macs += macs
            total_other_ops += other_ops
            total_flops += flops
            total_unfused_elems += unfused_elems
            total_intermediate_elems += layer_intermediate_elems
            total_unfused_bytes += unfused_bytes
            total_intermediate_bytes += layer_intermediate_bytes

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
                print(f"Debug: declared output {k} ({op}) not produced by a traced "
                      f"external write; charging {elems} elements")

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

        # model_io_elements: raw per-op sum (may double-count shared inputs).
        # Kept for diagnostic / per-layer inspection.
        total_model_io_elems = sum(
            layer.get("model_io_elements", 0)
            for layer in layers_out.values()
        )

        analysis: Dict[str, Any] = {
            "layers": layers_out,
            "total": {
                "num_layers": len(layers_out),
                "num_start_nodes_filtered": len(start_node_ids),
                "macs": int(total_macs),
                "other_ops": int(total_other_ops),
                "flops": int(total_flops),
                "unfused_elements": int(total_unfused_elems),
                "orojenesis_elements": None,
                "fused_elements": int(total_fused_elems),
                "fused_prefetched_elements": total_fused_prefetched_elems,
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
                "num_intermediate_tensors": len(intermediate_tensors),
                "num_orphaned_layers": len(_orphaned_layers),
            },
            "metadata": {
                "precision": precision,
                "bytes_per_element": element_size,
                "bytes_accounting": "per-tensor-dtype",
                "source_graph": str(src),
            },
        }

        out_path = out_dir / "analysis.yaml"
        with open(out_path, "w") as f:
            yaml.dump(analysis, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            print(f"✅ Wrote analysis: {out_path}")

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
