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

"""Subgraph expansions of fused PyTorch ops.

``scaled_dot_product_attention``, ``multi_head_attention_forward``,
``torch.lstm`` and ``torch.gru`` are single trace nodes that hide several
einsums.  Each expansion here resolves the node's operands by signature
(:mod:`solar.einsum.operands`), so a weight, bias, mask or initial state
passed as a plain or computed tensor keeps its real producer and is counted
exactly like an ``nn.Parameter``, then emits the einsums that make up the
op.  Every emitted layer carries the full layer schema the analyzer expects.

Each ``expand_*`` returns ``(layers, final_node_id, input_mapping)``:
``layers`` in emission order, ``final_node_id`` the layer whose output
replaces the original node's, and ``input_mapping`` from recorded input
slot to the layer that consumes it (used to rewire predecessors).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from solar.einsum.operands import (
    SIGNATURES,
    Operand,
    bind_operands,
    resolve_operands,
    tensor_operands,
)

Expansion = Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]


class Tensor(NamedTuple):
    """An input edge of an emitted layer."""

    name: str
    shape: List[int]
    dtype: Optional[str]
    type: str = "input"
    producer: Optional[str] = None

    @staticmethod
    def of(op: Operand) -> "Tensor":
        return Tensor(op.name, list(op.shape), op.dtype, op.type, op.producer)

    @property
    def elems(self) -> int:
        return math.prod(self.shape)


class _Builder:
    """Emits layers for one expanded node and tracks their wiring."""

    def __init__(self, node_id: str, output_connections: List[str]):
        self.node_id = node_id
        self.output_connections = list(output_connections)
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.input_mapping: Dict[int, str] = {}
        self._consumers: Dict[str, List[str]] = {}

    def add(
        self,
        suffix: str,
        op_type: str,
        equation: str,
        inputs: Sequence[Tensor],
        out_shape: List[int],
        out_dtype: Optional[str],
        *,
        real: bool,
        elementwise_op: str = "mul",
        reduction_op: str = "none",
        additional_info: Optional[Dict[str, Any]] = None,
        operands: Sequence[Operand] = (),
    ) -> Tensor:
        """Emit ``<node>.<suffix>`` and return its output as a Tensor.

        ``operands`` are the recorded slots this layer consumes, so the
        original node's predecessors can be rewired onto it.
        """
        lid = f"{self.node_id}.{suffix}"
        lhs, rhs = equation.split("->")
        in_letters = lhs.split(",")
        names = [t.name for t in inputs]
        ops: Dict[str, List[str]] = {}
        for i, (t, letters) in enumerate(zip(inputs, in_letters)):
            key = ("Weight" if t.type == "weight" else "Input")
            if key in ops:
                key = f"{key}_{i}"
            ops[key] = list(letters)
        ops["Output"] = list(rhs)
        connections = []
        for t in inputs:
            if t.producer and t.producer not in connections:
                connections.append(t.producer)
                self._consumers.setdefault(t.producer, []).append(lid)
        self.layers[lid] = {
            "type": op_type,
            "einsum_equation": equation,
            "elementwise_op": elementwise_op,
            "reduction_op": reduction_op,
            "is_real_einsum": real,
            "is_einsum_supportable": True,
            "operands": ops,
            "tensor_names": {"inputs": names, "outputs": [f"{lid}.Output"]},
            "tensor_types": {"inputs": [t.type for t in inputs], "outputs": ["output"]},
            "tensor_shapes": {"inputs": [list(t.shape) for t in inputs], "outputs": [list(out_shape)]},
            "tensor_dtypes": {"inputs": [t.dtype for t in inputs], "outputs": [out_dtype]},
            "connections": {"inputs": connections, "outputs": []},
        }
        if additional_info:
            self.layers[lid]["additional_info"] = dict(additional_info)
        for op in operands:
            self.input_mapping.setdefault(op.slot, lid)
        return Tensor(f"{lid}.Output", list(out_shape), out_dtype, "input", lid)

    def finish(self, final: Tensor) -> Expansion:
        """Wire intra-subgraph outputs and hand the node's successors to ``final``."""
        for lid, layer in self.layers.items():
            layer["connections"]["outputs"] = list(self._consumers.get(lid, []))
        assert final.producer in self.layers
        self.layers[final.producer]["connections"]["outputs"] = list(self.output_connections)
        return self.layers, final.producer, self.input_mapping


# --- equation helpers ------------------------------------------------------

_SPARE = "UWXYZRTVNMLJ"


def _broadcast_letters(ref: str, ref_shape: Sequence[int], shape: Sequence[int], used: str) -> str:
    """Letters for ``shape`` broadcast against ``ref`` (right aligned).

    Dims equal to the reference share its letter; size-1 and otherwise
    differing dims (e.g. a mask's ``B*H``) get a fresh letter so no letter
    binds two sizes.
    """
    spare = iter(c for c in _SPARE if c not in used and c not in ref)
    out = []
    for i, size in enumerate(shape):
        j = len(ref) - len(shape) + i
        if 0 <= j < len(ref) and ref_shape[j] == size:
            out.append(ref[j])
        else:
            out.append(next(spare))
    return "".join(out)


def _lead_letters(n: int) -> str:
    """Letters for ``n`` leading batch-like dims; the 4-D case keeps ``BH``."""
    pool = "BHEFGIJLMNPR"
    if n > len(pool):
        raise ValueError(f"too many leading dims: {n}")
    return pool[:n]


def _common_dtype(ops: Sequence[Optional[Operand]], fallback: str = "torch.float32") -> str:
    for op in ops:
        if op is not None and op.dtype:
            return str(op.dtype)
    return fallback


def _require(op: Any, what: str, node_id: str) -> Operand:
    if not isinstance(op, Operand):
        raise ValueError(f"{node_id}: {what} was not recorded in the trace")
    return op


# --- scaled_dot_product_attention ------------------------------------------

def expand_sdpa(converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map) -> Expansion:
    """``softmax(Q K^T * scale + mask) V`` with any number of leading dims.

    ``attn_mask`` (bool or additive, broadcastable) is read once by the
    mask layer; it was previously dropped when passed by keyword or wired
    into the wrong slot when passed positionally.
    """
    ops = resolve_operands(converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)
    roles = bind_operands(node_data, ops, SIGNATURES["scaled_dot_product_attention"])
    q = _require(roles.get("query"), "query", node_id)
    k = _require(roles.get("key"), "key", node_id)
    v = _require(roles.get("value"), "value", node_id)
    mask = roles.get("attn_mask") if isinstance(roles.get("attn_mask"), Operand) else None
    if len(q.shape) < 2 or len(k.shape) < 2 or len(v.shape) < 2:
        raise ValueError(f"{node_id}: SDPA inputs must be at least 2-D, got {q.shape}, {k.shape}, {v.shape}")

    lead_shape = list(q.shape[:-2])
    lead = _lead_letters(len(lead_shape))
    Q, D, K, V = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1]
    scores_shape = lead_shape + [Q, K]
    out_shapes = node_data.get("output_shapes") or []
    out_shape = list(out_shapes[0]) if out_shapes else lead_shape + [Q, V]
    act = _common_dtype([q, k, v])
    out_dtype = str((node_data.get("output_dtypes") or [act])[0])

    b = _Builder(node_id, sorted(op_graph.successors(node_id)))
    scores = b.add("qk_matmul", "matmul", f"{lead}QD,{lead}KD->{lead}QK", [Tensor.of(q), Tensor.of(k)],
                   scores_shape, act, real=True, reduction_op="add", operands=[q, k])
    scale = roles.get("scale")
    scaled = b.add("scale", "mul", f"{lead}QK->{lead}QK", [scores], scores_shape, act, real=False,
                   additional_info={"scale_factor": scale if isinstance(scale, (int, float)) else f"1/sqrt({D})"})
    if mask is not None:
        mask_letters = _broadcast_letters(f"{lead}QK", scores_shape, mask.shape, used="QKD")
        scaled = b.add("mask", "add", f"{lead}QK,{mask_letters}->{lead}QK", [scaled, Tensor.of(mask)],
                       scores_shape, act, real=False, elementwise_op="add", operands=[mask])
    probs = b.add("softmax", "softmax", f"{lead}QK->{lead}QK", [scaled], scores_shape, act, real=False,
                  elementwise_op="softmax", additional_info={"dim": -1})
    out = b.add("av_matmul", "matmul", f"{lead}QK,{lead}KV->{lead}QV", [probs, Tensor.of(v)], out_shape,
                out_dtype, real=True, reduction_op="add", operands=[v])
    return b.finish(out)


# --- multi_head_attention_forward -------------------------------------------

def _slices(b: _Builder, suffix: str, src: Optional[Tensor], n: int, part_shape: List[int],
            operands: Sequence[Operand] = ()) -> List[Optional[Tensor]]:
    """``n`` equal blocks of ``src`` along dim 0 (in_proj rows, initial states per cell).

    A parameter is addressed as ``n`` named weight blocks; any other tensor
    goes through one ``chunk`` layer with ``n`` output slots, so the source
    is read once however many cells consume it.
    """
    if src is None:
        return [None] * n
    if src.type == "weight":
        return [Tensor(f"{src.name}[{i}]", list(part_shape), src.dtype, "weight") for i in range(n)]
    lid = f"{b.node_id}.{suffix}"
    letters = "ZBHNK"[: len(src.shape)]
    out_letters = letters if len(part_shape) == len(src.shape) else letters[1:]
    b.add(suffix, "chunk", f"{letters}->{out_letters}", [src], part_shape, src.dtype,
          real=False, additional_info={"chunks": n, "dim": 0}, operands=operands)
    layer = b.layers[lid]
    layer["tensor_names"]["outputs"] = [f"{lid}.Output"] + [f"{lid}.Output_{i}" for i in range(1, n)]
    layer["tensor_shapes"]["outputs"] = [list(part_shape)] * n
    layer["tensor_types"]["outputs"] = ["output"] * n
    layer["tensor_dtypes"]["outputs"] = [src.dtype] * n
    return [Tensor(name, list(part_shape), src.dtype, "input", lid) for name in layer["tensor_names"]["outputs"]]


def expand_mha(converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map) -> Expansion:
    """Input projections, per-head attention and the output projection.

    Self-attention (query is key is value) uses the packed ``in_proj``
    weight in one einsum; otherwise each of q/k/v is projected with its own
    row block of that weight (or with ``q/k/v_proj_weight``).  Biases and
    masks are read where PyTorch reads them.
    """
    ops = resolve_operands(converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)
    roles = bind_operands(node_data, ops, SIGNATURES["multi_head_attention_forward"])
    q = _require(roles.get("query"), "query", node_id)
    k = roles.get("key") if isinstance(roles.get("key"), Operand) else q
    v = roles.get("value") if isinstance(roles.get("value"), Operand) else k
    if len(q.shape) != 3:
        raise ValueError(f"{node_id}: expected [L, B, D] query, got {q.shape}")
    Lq, B, D = q.shape
    Lk = k.shape[0]
    heads = roles.get("num_heads") if isinstance(roles.get("num_heads"), int) else 1
    Dh = D // heads if heads and D % heads == 0 else D
    heads = D // Dh
    act = _common_dtype([q, k, v])
    out_shapes = node_data.get("output_shapes") or []
    out_shape = list(out_shapes[0]) if out_shapes else [Lq, B, D]
    out_dtype = str((node_data.get("output_dtypes") or [act])[0])

    def opt(name: str) -> Optional[Operand]:
        val = roles.get(name)
        return val if isinstance(val, Operand) else None

    in_w, in_b = opt("in_proj_weight"), opt("in_proj_bias")
    out_w, out_b = opt("out_proj_weight"), opt("out_proj_bias")
    sep_w = [opt("q_proj_weight"), opt("k_proj_weight"), opt("v_proj_weight")]

    b = _Builder(node_id, sorted(op_graph.successors(node_id)))

    def project(suffix: str, x: Operand, w: Tensor, bias: Optional[Tensor], n_out: int) -> Tensor:
        y = b.add(suffix, "linear", "SBK,NK->SBN", [Tensor.of(x), w], [x.shape[0], B, n_out], act,
                  real=True, reduction_op="add", operands=[x] + [o for o in (w_op(w), bias_op(bias)) if o])
        if bias is not None:
            y = b.add(f"{suffix}_bias_add", "add", "SBN,N->SBN", [y, bias], y.shape, act, real=False,
                      elementwise_op="add", operands=[o for o in (bias_op(bias),) if o])
        return y

    def w_op(t: Optional[Tensor]) -> Optional[Operand]:
        return next((o for o in (in_w, out_w, *sep_w) if o and t and t.name == o.name), None)

    def bias_op(t: Optional[Tensor]) -> Optional[Operand]:
        return next((o for o in (in_b, out_b) if o and t and t.name == o.name), None)

    def heads_view(suffix: str, x: Tensor, L: int) -> Tensor:
        return b.add(suffix, "view", "SBN->BHSD", [x], [B, heads, L, Dh], act, real=False)

    packed = (q.name == k.name == v.name) and in_w is not None
    if packed:
        qkv = project("in_proj", q, Tensor.of(in_w), Tensor.of(in_b) if in_b else None, 3 * D)
        q_h = b.add("q_view", "view", "SBN->BHQD", [qkv], [B, heads, Lq, Dh], act, real=False)
        k_h = b.add("k_view", "view", "SBN->BHKD", [qkv], [B, heads, Lk, Dh], act, real=False)
        v_h = b.add("v_view", "view", "SBN->BHKD", [qkv], [B, heads, Lk, Dh], act, real=False)
    else:
        if in_w is not None:
            ws = _slices(b, "in_proj_weight_chunk", Tensor.of(in_w), 3, [D, D], operands=[in_w])
        else:
            ws = [Tensor.of(w) if w else None for w in sep_w]
        bs = _slices(b, "in_proj_bias_chunk", Tensor.of(in_b) if in_b else None, 3, [D],
                     operands=[in_b] if in_b else [])
        outs = []
        for suffix, x, w, bias in zip(("q_proj", "k_proj", "v_proj"), (q, k, v), ws, bs):
            if w is None:
                raise ValueError(f"{node_id}: no projection weight recorded for {suffix}")
            outs.append(project(suffix, x, w, bias, D))
        q_h = heads_view("q_view", outs[0], Lq)
        k_h = heads_view("k_view", outs[1], Lk)
        v_h = heads_view("v_view", outs[2], Lk)

    scores_shape = [B, heads, Lq, Lk]
    scores = b.add("qk_matmul", "matmul", "BHQD,BHKD->BHQK", [q_h, k_h], scores_shape, act, real=True,
                   reduction_op="add")
    scaled = b.add("scale", "mul", "BHQK->BHQK", [scores], scores_shape, act, real=False,
                   additional_info={"scale_factor": f"1/sqrt({Dh})"})
    for name in ("attn_mask", "key_padding_mask"):
        mask = opt(name)
        if mask is None:
            continue
        ref, ref_shape = ("BHQK", scores_shape)
        if name == "key_padding_mask" and len(mask.shape) == 2:
            letters = "BK"
        else:
            letters = _broadcast_letters(ref, ref_shape, mask.shape, used="QKD")
        scaled = b.add(name, "add", f"BHQK,{letters}->BHQK", [scaled, Tensor.of(mask)], scores_shape, act,
                       real=False, elementwise_op="add", operands=[mask])
    probs = b.add("softmax", "softmax", "BHQK->BHQK", [scaled], scores_shape, act, real=False,
                  elementwise_op="softmax", additional_info={"dim": -1})
    ctx = b.add("av_matmul", "matmul", "BHQK,BHKD->BHQD", [probs, v_h], [B, heads, Lq, Dh], act, real=True,
                reduction_op="add")
    merged = b.add("out_view", "view", "BHQD->QBN", [ctx], [Lq, B, D], act, real=False)
    if out_w is None:
        raise ValueError(f"{node_id}: out_proj_weight was not recorded in the trace")
    out = b.add("out_proj", "linear", "SBK,NK->SBN", [merged, Tensor.of(out_w)], [Lq, B, D],
                out_dtype if out_b is None else act, real=True, reduction_op="add", operands=[out_w])
    if out_b is not None:
        out = b.add("out_proj_bias_add", "add", "SBN,N->SBN", [out, Tensor.of(out_b)], out_shape, out_dtype,
                    real=False, elementwise_op="add", operands=[out_b])
    return b.finish(out)


# --- lstm / gru -------------------------------------------------------------

def _expand_rnn(kind: str, converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map) -> Expansion:
    """Per layer and direction: ``x W_ih^T + h W_hh^T + b_ih + b_hh`` over all steps.

    ``params`` is the flat ``[W_ih, W_hh, b_ih, b_hh] * layers * directions``
    list ``torch.lstm``/``torch.gru`` receive.  A traced initial state is
    read once (one chunk per cell, expanded over the steps); the zero state
    the modules create in forward is a creation op and costs nothing.
    """
    gates = {"lstm": 4, "gru": 3}[kind]
    ops = resolve_operands(converter, node_id, node_data, op_graph, start_nodes_info, start_node_id_map)
    roles = bind_operands(node_data, ops, SIGNATURES[kind])
    x = _require(roles.get("input"), "input", node_id)
    params = tensor_operands(roles.get("params"))
    hx = tensor_operands(roles.get("hx"))
    h0 = hx[0] if hx else None
    c0 = hx[1] if len(hx) > 1 else None
    has_biases = bool(roles.get("has_biases", len(params) % 4 == 0 and len(params) >= 4))
    per_cell = 4 if has_biases else 2
    if len(params) < 2 or len(params) % per_cell:
        raise ValueError(f"{node_id}: expected {per_cell} tensors per cell in params, got {len(params)}")
    num_dirs = 2 if roles.get("bidirectional") else 1
    num_layers = roles.get("num_layers") if isinstance(roles.get("num_layers"), int) else len(params) // per_cell // num_dirs
    batch_first = bool(roles.get("batch_first"))
    if len(x.shape) != 3:
        raise ValueError(f"{node_id}: expected a 3-D {kind} input, got {x.shape}")
    S, B = (x.shape[1], x.shape[0]) if batch_first else (x.shape[0], x.shape[1])
    seq = "BS" if batch_first else "SB"
    H = params[1].shape[1]
    G = gates * H
    act = _common_dtype([x])
    out_shapes = node_data.get("output_shapes") or []
    out_dtype = str((node_data.get("output_dtypes") or [act])[0])

    b = _Builder(node_id, sorted(op_graph.successors(node_id)))
    n_cells = num_layers * num_dirs

    def untraced(op: Optional[Operand]) -> bool:
        # No producer and not a parameter: the zero state nn.LSTM/nn.GRU
        # create in forward, which torchview cannot attribute.
        return op is None or (op.producer is None and not op.is_parameter)

    if untraced(h0):
        h_src = b.add("h0_zeros", "zeros", "->ZBH", [], [n_cells, B, H], act, real=False, elementwise_op="zeros")
    else:
        h_src = Tensor.of(h0)
    h0_cells = _slices(b, "h0_unbind", h_src, n_cells, [B, H], operands=[] if untraced(h0) else [h0])
    if untraced(c0):
        c0 = None
    layer_in: Tensor = Tensor.of(x)
    for layer in range(num_layers):
        dir_outs = []
        for d in range(num_dirs):
            cell = layer * num_dirs + d
            tag = f"l{layer}" + (f"d{d}" if num_dirs > 1 else "")
            w_ih, w_hh = params[cell * per_cell], params[cell * per_cell + 1]
            biases = params[cell * per_cell + 2: cell * per_cell + 4] if has_biases else []
            I = layer_in.shape[-1]
            ih = b.add(f"{tag}.ih_linear", "linear", f"{seq}I,GI->{seq}G", [layer_in, Tensor.of(w_ih)],
                       [x.shape[0], x.shape[1], G], act, real=True, reduction_op="add",
                       operands=[w_ih] + ([x] if layer == 0 else []))
            h_in = b.add(f"{tag}.h0_expand", "expand", f"BH->{seq}H", [h0_cells[cell]],
                         [x.shape[0], x.shape[1], H], act, real=False)
            hh = b.add(f"{tag}.hh_linear", "linear", f"{seq}H,GH->{seq}G", [h_in, Tensor.of(w_hh)],
                       [x.shape[0], x.shape[1], G], act, real=True, reduction_op="add", operands=[w_hh])
            gate_inputs = [ih, hh] + [Tensor.of(p) for p in biases]
            gate_letters = [f"{seq}G", f"{seq}G"] + ["G"] * len(biases)
            if kind == "lstm" and c0 is not None and layer == 0 and d == 0:
                gate_inputs.append(Tensor.of(c0))
                gate_letters.append("ZBH")
            out = b.add(f"{tag}.gates", "add", ",".join(gate_letters) + f"->{seq}H", gate_inputs,
                        [x.shape[0], x.shape[1], H], act, real=False, elementwise_op="add",
                        additional_info={"gates": gates},
                        operands=biases + ([c0] if kind == "lstm" and c0 is not None and layer == 0 and d == 0 else []))
            dir_outs.append(out)
        if num_dirs == 1:
            layer_in = dir_outs[0]
        else:
            layer_in = b.add(f"l{layer}.cat", "cat", f"{seq}H,{seq}H->{seq}N", dir_outs,
                             [x.shape[0], x.shape[1], num_dirs * H], act, real=False, elementwise_op="cat",
                             additional_info={"dim": -1})
    final = layer_in
    if out_shapes and list(out_shapes[0]) != final.shape:
        raise ValueError(f"{node_id}: {kind} output {out_shapes[0]} does not match expanded {final.shape}")
    b.layers[final.producer]["tensor_dtypes"]["outputs"] = [out_dtype]
    return b.finish(final)


def expand_lstm(*args) -> Expansion:
    return _expand_rnn("lstm", *args)


def expand_gru(*args) -> Expansion:
    return _expand_rnn("gru", *args)
