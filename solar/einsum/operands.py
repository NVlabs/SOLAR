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

"""Operands of a traced call, by role.

torchview records a call's tensor arguments in signature order (positional
tensors first, then keyword tensors) and labels each slot ``weight`` when it
is an ``nn.Parameter`` and ``input`` otherwise.  A converter that needs
*roles* -- which slot is the query, the attention mask, ``W_hh`` -- must
not read them off the labels or the shapes: a weight computed on the fly
or passed as a plain tensor is labelled ``input`` and sits wherever the
caller put it.

Two steps give every multi-operand op the same treatment ``F.linear`` gets:

1. :func:`resolve_operands` converts the node once with the generic
   machinery, so each slot carries its tensor name, shape, dtype, label and
   producer layer (output slots of chunk/split included).
2. :func:`bind_operands` zips those slots with the callee's signature,
   using the recorded call (``raw_attributes``) to see which parameters
   were tensors, which were ``None`` or scalars, and which were lists of
   tensors (``torch.lstm``'s ``params``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from solar.analysis.access_regions import _TensorArg, parse_call_attributes


class Operand(NamedTuple):
    """One recorded tensor argument of a traced call."""

    slot: int
    name: str                 # tensor name in the einsum graph
    shape: List[int]
    dtype: Optional[str]
    type: str                 # "weight" (nn.Parameter) or "input"
    producer: Optional[str]   # layer id that produces it; None for parameters

    @property
    def is_parameter(self) -> bool:
        return self.type == "weight"


# Parameter names in signature order. Only tensor-valued parameters consume
# recorded slots; the rest are reported by value so callers can read flags
# (``num_heads``, ``has_biases``, ``batch_first``) from the same binding.
SIGNATURES: Dict[str, Sequence[str]] = {
    "linear": ("input", "weight", "bias"),
    "scaled_dot_product_attention": (
        "query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa",
    ),
    "multi_head_attention_forward": (
        "query", "key", "value", "embed_dim_to_check", "num_heads", "in_proj_weight",
        "in_proj_bias", "bias_k", "bias_v", "add_zero_attn", "dropout_p", "out_proj_weight",
        "out_proj_bias", "training", "key_padding_mask", "need_weights", "attn_mask",
        "use_separate_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
        "static_k", "static_v", "average_attn_weights", "is_causal",
    ),
    "lstm": ("input", "hx", "params", "has_biases", "num_layers", "dropout", "train",
             "bidirectional", "batch_first"),
    "gru": ("input", "hx", "params", "has_biases", "num_layers", "dropout", "train",
            "bidirectional", "batch_first"),
}
# Parameters that are never tensors; a trace without a call record binds
# its slots to the remaining (tensor) parameters in order.
SCALAR_PARAMS = frozenset({
    "dropout_p", "is_causal", "scale", "enable_gqa", "embed_dim_to_check", "num_heads",
    "add_zero_attn", "training", "need_weights", "use_separate_proj_weight",
    "average_attn_weights", "has_biases", "num_layers", "dropout", "train", "bidirectional",
    "batch_first",
})
SIGNATURES["multihead_attention"] = SIGNATURES["multi_head_attention_forward"]
SIGNATURES["sdpa"] = SIGNATURES["scaled_dot_product_attention"]
SIGNATURES["attention"] = SIGNATURES["scaled_dot_product_attention"]

_TENSOR_SHAPE_RE = re.compile(r"Tensor\(shape=\(([^)]*)\)")


def resolve_operands(
    converter: Any,
    node_id: str,
    node_data: Dict[str, Any],
    op_graph: Any,
    start_nodes_info: List[Dict[str, Any]],
    start_node_id_map: Dict[str, str],
) -> List[Operand]:
    """Every recorded input slot of ``node_id`` with its resolved source.

    Parameters keep ``producer=None`` and the ``<node>.Weight[_k]`` name the
    generic conversion gives them; activations reference the layer (and
    output slot) that produces them, start nodes included.
    """
    layer = converter._convert_operation(
        node_id, node_data, op_graph, start_nodes_info, start_node_id_map
    )
    names = list(layer["tensor_names"]["inputs"])
    types = list(layer["tensor_types"]["inputs"])
    dtypes = list(node_data.get("input_dtypes") or [])
    raw_types = [str(t).lower() for t in (node_data.get("input_types") or [])]
    producers = iter(layer["connections"]["inputs"])
    operands = []
    n_weights = 0
    for i, shape in enumerate(node_data.get("input_shapes") or []):
        if i < len(names) and i < len(types):
            name, itype = names[i], types[i]
        else:
            # Slot without a recorded connection (synthetic graphs): name it
            # the way _build_tensor_names would.
            itype = raw_types[i] if i < len(raw_types) else "input"
            name = (f"{node_id}.Weight" if n_weights == 0 else f"{node_id}.Weight_{n_weights}") \
                if itype == "weight" else f"{node_id}.Input_{i}"
        if itype == "weight":
            n_weights += 1
        producer = None if itype == "weight" else next(producers, None)
        if producer == node_id:
            # A tensor the trace could only attribute to the node itself
            # (torch.lstm's zero initial state shares a shape with its h_n
            # output). Treat it as an untraced external input.
            producer, name = None, f"{node_id}.Input_{i}"
        elif producer is not None and producer not in op_graph.nodes and producer not in start_node_id_map.values():
            # A torchview placeholder (``hidden-tensor``) with no producing
            # layer: an untraced creation inside forward.
            producer = None
        operands.append(Operand(
            i, name, list(shape), dtypes[i] if i < len(dtypes) else None, itype, producer,
        ))
    return operands


def bind_operands(
    node_data: Dict[str, Any],
    operands: Sequence[Operand],
    signature: Sequence[str],
) -> Dict[str, Any]:
    """Map recorded operands onto ``signature`` by position.

    Returns ``{parameter: value}`` where a tensor parameter is an
    :class:`Operand` (or None when it was not recorded, e.g. a keyword
    tensor an older trace dropped), a list parameter is a list of
    operands, and a scalar parameter is its literal value.  Without a
    recorded call, parameters go to the weight/bias/params parameters in
    signature order and activations to the remaining tensor parameters.
    """
    raw = (node_data.get("module_args") or {}).get("raw_attributes")
    parsed = parse_call_attributes(raw)
    if parsed is None:
        return _bind_without_record(operands, signature)
    args, kwargs = parsed
    # Rank of each Tensor(...) in text order; 0-d tensors are not recorded
    # as slots, so they must not consume one.
    ranks = iter(len([d for d in dims.split(",") if d.strip()])
                 for dims in _TENSOR_SHAPE_RE.findall(str(raw)))
    queue = list(operands)

    def take(value: Any) -> Any:
        if isinstance(value, _TensorArg):
            if next(ranks, 1) == 0:
                return None
            return queue.pop(0) if queue else None
        if isinstance(value, (list, tuple)):
            return [take(v) for v in value]
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value

    roles = {}
    for i, name in enumerate(signature):
        value = args[i] if i < len(args) else kwargs.get(name)
        roles[name] = take(value)
    return roles


def tensor_operands(value: Any) -> List[Operand]:
    """Flatten a bound value (operand, list of operands, None) to operands."""
    if isinstance(value, Operand):
        return [value]
    if isinstance(value, (list, tuple)):
        return [o for v in value for o in tensor_operands(v)]
    return []


def _bind_without_record(operands: Sequence[Operand], signature: Sequence[str]) -> Dict[str, Any]:
    """Positional binding by label when the trace carries no call record."""
    roles: Dict[str, Any] = {name: None for name in signature}
    tensor_params = [p for p in signature if p not in SCALAR_PARAMS]
    weight_params = [p for p in tensor_params if p.endswith(("weight", "bias", "params"))]
    act_params = [p for p in tensor_params if p not in weight_params]
    weights = [o for o in operands if o.is_parameter]
    for name, op in zip(act_params, (o for o in operands if not o.is_parameter)):
        roles[name] = op
    for name in weight_params:
        if not weights:
            break
        if name == "params":
            roles[name], weights = list(weights), []
        else:
            roles[name] = weights.pop(0)
    return roles
