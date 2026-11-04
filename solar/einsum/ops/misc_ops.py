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

"""Handlers for miscellaneous operations.

This module provides einsum handlers for:
- embedding
- gru, lstm, rnn
- cross_entropy, nll_loss
- clone, detach, copy_, to
- roll, pad, unfold, fold
"""

import string
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes

logger = logging.getLogger(__name__)


class EmbeddingHandler(EinsumOpHandler):
    """Handler for embedding operations."""

    supported_ops = ["embedding"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for embedding lookup."""
        if tensor_shapes.num_inputs < 2:
            raise ValueError(
                f"Embedding requires 2 inputs (indices, weight). "
                f"Got {tensor_shapes.num_inputs}"
            )

        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1]

        return self._generate_embedding_einsum(input_shape, weight_shape)

    def _generate_embedding_einsum(
        self,
        indices_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for embedding lookup.

        Embedding: indices [*] x weight [V, D] -> output [*, D]
        This is essentially a gather operation.
        """
        batch_dims = len(indices_shape)
        batch_labels = string.ascii_uppercase[:batch_dims]

        operands = [
            EinsumOperand("Input", list(batch_labels), is_output=False),
            EinsumOperand("Weight", ["V", "D"], is_output=False),
            EinsumOperand("Output", list(batch_labels) + ["D"], is_output=True),
        ]

        equation = f"{batch_labels},VD->{batch_labels}D"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="embedding",
            is_real_einsum=False,
            elementwise_op="embedding",  # Index lookup into weight table
            reduction_op="none",
        )


# ── Recurrent layers: exact MACs over the time loop, layer stack and directions ──
#
# torch.lstm / torch.gru / torch.rnn_tanh / torch.rnn_relu are recorded as ONE
# node whose operands are: x, the initial state(s) (3-D: h0 [, c0]), then for
# every (layer, direction) the flat weights w_ih [G*H, in_l], w_hh [G*H, H]
# and, with biases, b_ih [G*H], b_hh [G*H].  A single einsum over those
# operands cannot express the recurrence, so the handlers below keep their
# descriptive equation and supply the exact count (defect an earlier decision, issue #33):
#
#   MACs = sum over (layer, direction) of  S * B * (G*H) * (in_l + H)
#
# i.e. per time step the input projection (ih) and the recurrent projection
# (hh) of all G gates; the time loop runs S steps; every layer and every
# direction runs it (in_0 = input size, in_l = D*H for l >= 1).
#
# Two routes price a recurrent node (an earlier fix P3-1):
#
# * DEFAULT — the converter EXPANDS torch.lstm / torch.gru / torch.rnn_tanh /
#   torch.rnn_relu into per-(layer, direction) ih/hh linear layers, a gates
#   add and the synthetic initial-state placeholder (solar.einsum.expansions).
#   There the MACs above come out of the expanded linear layers, the gate
#   arithmetic is charged as other_ops of the gates layer, and the
#   intermediate tensors between the sub-layers carry their own traffic.
#   This is the route real captures take; the handlers below are not
#   consulted for an expanded node.
# * RAW HANDLER — an unexpanded whole node (the expansion refused because the
#   recorded operands could not be bound, or canonical metadata fed straight
#   to EinsumAnalyzer/graph analysis).  Only this route uses the conventions
#   that follow: biases, gate nonlinearities and the elementwise state
#   updates are not charged as separate ops (linear/conv handlers do not
#   charge their bias add either); traffic is not changed by the handler:
#   each weight tensor is read once per layer and the activations once (the
#   generic per-tensor accounting of the single node), a per-step re-read of
#   weights or states would be a different (unfused) convention and is not
#   applied.  Identical MACs on both routes do not imply identical traffic or
#   other_ops (F4).
#
# When the recorded operand list does not match the flat-weights convention
# (e.g. packed sequences, or an unexpected operand order) the handler falls
# back to the generic equation cost and logs a warning naming the operation,
# so the limitation is visible rather than silently wrong.

_RNN_GATES = {"rnn": 1, "gru": 3, "lstm": 4}

# torch.lstm(input, hx, params, has_biases, num_layers, dropout, train,
#            bidirectional, batch_first) -- the recorded raw_attributes end with
# the six scalar arguments followed by the empty kwargs dict.
_RNN_CALL_TAIL = re.compile(
    r"\],\s*(True|False),\s*(\d+),\s*([0-9.eE+\-]+),\s*(True|False),\s*(True|False),\s*(True|False)\s*\]\s*,\s*\{\}\s*\]\s*$")


def rnn_call_attributes(raw_attributes: Any) -> Optional[Dict[str, Any]]:
    """``{"has_biases", "num_layers", "bidirectional", "batch_first"}`` parsed from the recorded
    ``torch.lstm/gru/rnn_*`` call text, or None when the text is absent or not of that shape
    (packed-sequence calls have no ``batch_first`` argument and are not parsed)."""
    if not isinstance(raw_attributes, str):
        return None
    m = _RNN_CALL_TAIL.search(raw_attributes.strip())
    if not m:
        return None
    return {"has_biases": m.group(1) == "True", "num_layers": int(m.group(2)),
            "bidirectional": m.group(5) == "True", "batch_first": m.group(6) == "True"}


def recurrent_mac_cost(op_name: str, tensor_shapes: TensorShapes,
                       raw_attributes: Any = None) -> Tuple[Optional[int], Dict[str, Any]]:
    """Exact MACs of a recurrent op from its recorded operands; ``(None, info)`` when the operand
    list cannot be interpreted (``info["reason"]`` says why)."""
    gates = _RNN_GATES[op_name]
    inputs = [list(s) for s in (tensor_shapes.inputs or []) if s is not None]
    if not inputs:
        return None, {"reason": "no input shapes"}
    x = inputs[0]
    mats = [s for s in inputs[1:] if len(s) == 2]
    states = [s for s in inputs[1:] if len(s) == 3]
    if len(mats) < 2 or len(mats) % 2:
        return None, {"reason": "weights are not (w_ih, w_hh) pairs: %d matrices" % len(mats)}
    pairs = [(mats[i], mats[i + 1]) for i in range(0, len(mats), 2)]
    gh, hidden = int(pairs[0][1][0]), int(pairs[0][1][1])
    if hidden <= 0 or gh != gates * hidden:
        return None, {"reason": "w_hh %r does not match %d gates" % (pairs[0][1], gates)}
    for w_ih, w_hh in pairs:
        if int(w_ih[0]) != gh or [int(v) for v in w_hh] != [gh, hidden]:
            return None, {"reason": "inconsistent weight shapes %r / %r" % (w_ih, w_hh)}
    attrs = rnn_call_attributes(raw_attributes)
    if attrs is not None:
        directions = 2 if attrs["bidirectional"] else 1
        layers = attrs["num_layers"]
        if layers * directions != len(pairs):
            return None, {"reason": "num_layers*directions=%d but %d weight pairs" % (layers * directions, len(pairs))}
        batch_first = attrs["batch_first"]
        source = "call attributes"
    else:
        outputs = [list(s) for s in (tensor_shapes.outputs or []) if s is not None]
        if outputs and outputs[0] and int(outputs[0][-1]) % hidden == 0 and int(outputs[0][-1]) // hidden in (1, 2):
            directions = int(outputs[0][-1]) // hidden
        elif len(pairs) > 1 and int(pairs[1][0][1]) % hidden == 0 and int(pairs[1][0][1]) // hidden in (1, 2):
            directions = int(pairs[1][0][1]) // hidden        # in_1 = D*H
        else:
            directions = 1
        if len(pairs) % directions:
            return None, {"reason": "%d weight pairs not divisible by %d directions" % (len(pairs), directions)}
        layers = len(pairs) // directions
        if len(x) == 3 and states:
            batch = int(states[0][1])
            # PyTorch default is seq-first; batch_first only when the first dim
            # (and not the second) matches the recorded batch of the state.
            batch_first = int(x[0]) == batch and int(x[1]) != batch
        else:
            batch_first = False
        source = "inferred from shapes"
    if len(x) == 3:
        batch, seq = (int(x[0]), int(x[1])) if batch_first else (int(x[1]), int(x[0]))
    elif len(x) == 2:                                  # unbatched (S, I)
        seq, batch = int(x[0]), 1
    else:
        return None, {"reason": "input rank %d" % len(x)}
    macs = 0
    per_pair = []
    for w_ih, w_hh in pairs:
        in_l = int(w_ih[1])
        m = seq * batch * gh * (in_l + hidden)
        per_pair.append({"in": in_l, "macs": m})
        macs += m
    info = {"seq_len": seq, "batch": batch, "hidden": hidden, "gates": gates, "layers": layers,
            "directions": directions, "batch_first": batch_first, "source": source, "per_layer_direction": per_pair}
    return int(macs), info


def _attach_recurrent_cost(op: EinsumOp, op_name: str, tensor_shapes: TensorShapes, kwargs: Dict[str, Any]) -> EinsumOp:
    macs, info = recurrent_mac_cost(op_name, tensor_shapes, kwargs.get("raw_attributes"))
    if macs is None:
        logger.warning("%s: recurrent operand list not interpreted (%s); keeping the generic equation cost "
                       "(known undercount, defect D-04)", op_name, info.get("reason"))
        return op
    op.compute_cost_override = macs
    return op


class GRUHandler(EinsumOpHandler):
    """Handler for GRU operations."""

    supported_ops = ["gru"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for GRU."""
        input_shape = tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else [1, 1, 64]
        hidden = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else [1, input_shape[1], 64]
        w_ih = tensor_shapes.inputs[2] if tensor_shapes.num_inputs > 2 else [192, input_shape[-1]]
        w_hh = tensor_shapes.inputs[3] if tensor_shapes.num_inputs > 3 else [192, 64]

        op = self._generate_gru_einsum(input_shape, hidden, w_ih, w_hh)
        return _attach_recurrent_cost(op, "gru", tensor_shapes, kwargs)

    def _generate_gru_einsum(
        self,
        input_shape: TensorShape,
        hidden_shape: TensorShape,
        weight_ih_shape: TensorShape,
        weight_hh_shape: TensorShape,
        num_layers: int = 1
    ) -> EinsumOp:
        """Generate einsum for GRU.

        GRU has 3 gates (reset, update, new) each with ih and hh weights.
        """
        operands = [
            EinsumOperand("Input", ["S", "B", "I"], is_output=False),
            EinsumOperand("Weight_ih", ["G", "I"], is_output=False),
            EinsumOperand("Weight_hh", ["G", "H"], is_output=False),
            EinsumOperand("Hidden", ["B", "H"], is_output=False),
            EinsumOperand("Output", ["S", "B", "H"], is_output=True),
        ]

        equation = "SBI,GI,GH,BH->SBH"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="gru",
            elementwise_op="mul",
            reduction_op="add",
        )


class LSTMHandler(EinsumOpHandler):
    """Handler for LSTM operations."""

    supported_ops = ["lstm"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for LSTM."""
        input_shape = tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else [1, 1, 64]
        hidden = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else [1, input_shape[1], 64]
        cell = tensor_shapes.inputs[2] if tensor_shapes.num_inputs > 2 else hidden
        w_ih = tensor_shapes.inputs[3] if tensor_shapes.num_inputs > 3 else [256, input_shape[-1]]
        w_hh = tensor_shapes.inputs[4] if tensor_shapes.num_inputs > 4 else [256, 64]

        op = self._generate_lstm_einsum(input_shape, hidden, cell, w_ih, w_hh)
        return _attach_recurrent_cost(op, "lstm", tensor_shapes, kwargs)

    def _generate_lstm_einsum(
        self,
        input_shape: TensorShape,
        hidden_shape: TensorShape,
        cell_shape: TensorShape,
        weight_ih_shape: TensorShape,
        weight_hh_shape: TensorShape,
        num_layers: int = 1
    ) -> EinsumOp:
        """Generate einsum for LSTM.

        LSTM has 4 gates (input, forget, cell, output) each with ih and hh weights.
        """
        operands = [
            EinsumOperand("Input", ["S", "B", "I"], is_output=False),
            EinsumOperand("Weight_ih", ["G", "I"], is_output=False),
            EinsumOperand("Weight_hh", ["G", "H"], is_output=False),
            EinsumOperand("Hidden", ["B", "H"], is_output=False),
            EinsumOperand("Cell", ["B", "H"], is_output=False),
            EinsumOperand("Output", ["S", "B", "H"], is_output=True),
        ]

        equation = "SBI,GI,GH,BH,BH->SBH"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="lstm",
            elementwise_op="mul",
            reduction_op="add",
        )


class RNNHandler(EinsumOpHandler):
    """Handler for basic RNN operations (the recorded ``torch.rnn_tanh`` /
    ``torch.rnn_relu`` names normalise to ``rnn``; see
    ``EinsumAnalyzer._get_operation_from_name``)."""

    supported_ops = ["rnn"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for basic RNN."""
        input_shape = tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else [1, 1, 64]

        operands = [
            EinsumOperand("Input", ["S", "B", "I"], is_output=False),
            EinsumOperand("Weight_ih", ["H", "I"], is_output=False),
            EinsumOperand("Weight_hh", ["H", "H"], is_output=False),
            EinsumOperand("Hidden", ["B", "H"], is_output=False),
            EinsumOperand("Output", ["S", "B", "H"], is_output=True),
        ]

        equation = "SBI,HI,HH,BH->SBH"

        op = EinsumOp(
            operands=operands,
            equation=equation,
            name="rnn",
            elementwise_op="mul",
            reduction_op="add",
        )
        return _attach_recurrent_cost(op, "rnn", tensor_shapes, kwargs)


class CrossEntropyHandler(EinsumOpHandler):
    """Handler for cross-entropy loss."""

    supported_ops = ["cross_entropy", "crossentropy", "nll_loss", "nllloss"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for cross-entropy loss."""
        pred = tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else None
        target = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None

        if pred is None:
            raise ValueError(f"Missing predictions shape for {op_name}")

        if target is None:
            target = [pred[0]] if pred else None

        reduction = kwargs.get("reduction", "mean")

        return self._generate_cross_entropy_einsum(pred, target, reduction)

    def _generate_cross_entropy_einsum(
        self,
        predictions_shape: TensorShape,
        targets_shape: Optional[TensorShape],
        reduction: str = "mean"
    ) -> EinsumOp:
        """Generate einsum for cross-entropy loss."""
        dims = len(predictions_shape)
        labels = string.ascii_uppercase[:dims]

        # Output depends on reduction
        if reduction == "none":
            out_labels = labels[0]  # batch dim only
        else:
            out_labels = ""  # scalar output

        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Target", [labels[0]], is_output=False),
            EinsumOperand("Output", list(out_labels) if out_labels else [], is_output=True),
        ]

        target_eq = labels[0]
        equation = f"{labels},{target_eq}->{out_labels}"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="cross_entropy",
            is_real_einsum=False,
            elementwise_op="cross_entropy",  # Cross-entropy loss function
            reduction_op="add" if reduction != "none" else "none",
        )


class PairwiseLossHandler(EinsumOpHandler):
    """Handler for pairwise loss functions (MSE, L1, Huber, etc.).

    These loss functions take two tensors of the same shape (predictions and targets)
    and produce a scalar output (when reduction='mean' or 'sum').
    """

    supported_ops = [
        "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
        "binary_cross_entropy", "kl_div", "cosine_embedding_loss",
    ]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for pairwise loss."""
        pred = tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else None
        target = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        output = tensor_shapes.outputs[0] if tensor_shapes.num_outputs > 0 else None

        if pred is None:
            raise ValueError(f"Missing predictions shape for {op_name}")

        # Determine reduction based on output shape
        # If output is empty list or None, it's a scalar (mean/sum reduction)
        reduction = "mean"
        if output is not None and len(output) > 0:
            reduction = "none"

        return self._generate_pairwise_loss_einsum(pred, target, reduction, op_name)

    def _generate_pairwise_loss_einsum(
        self,
        predictions_shape: TensorShape,
        targets_shape: Optional[TensorShape],
        reduction: str,
        op_name: str
    ) -> EinsumOp:
        """Generate einsum for pairwise loss."""
        dims = len(predictions_shape)
        labels = string.ascii_uppercase[:dims]

        # Second input has same shape
        target_labels = labels

        # Output depends on reduction
        if reduction == "none":
            out_labels = labels  # same shape as input
        else:
            out_labels = ""  # scalar output

        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Target", list(target_labels), is_output=False),
            EinsumOperand("Output", list(out_labels) if out_labels else [], is_output=True),
        ]

        equation = f"{labels},{target_labels}->{out_labels}"

        # Normalize op_name (remove _loss suffix for elementwise_op)
        normalized_op = op_name.replace("_loss", "").replace("_", "")

        return EinsumOp(
            operands=operands,
            equation=equation,
            name=op_name,
            is_real_einsum=False,
            elementwise_op=normalized_op,  # e.g., "mse", "l1", "smoothl1", "huber"
            reduction_op="add" if reduction != "none" else "none",
        )


class TrivialOpsHandler(EinsumOpHandler):
    """Handler for trivial/identity operations."""

    supported_ops = [
        "clone", "detach", "copy_", "to", "type", "float", "half",
        "hidden-tensor", "output-tensor", "auxiliary-tensor",
        "roll", "pad", "unfold", "fold",
    ]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for trivial operations."""
        if tensor_shapes.num_inputs < 1:
            raise ValueError(f"Missing Input shape for {op_name}")

        input_shape = tensor_shapes.inputs[0]

        return self._generate_identity_einsum(input_shape, op_name)

    def _generate_identity_einsum(
        self,
        input_shape: TensorShape,
        op_name: str
    ) -> EinsumOp:
        """Generate identity einsum."""
        dims = len(input_shape)
        labels = string.ascii_uppercase[:dims]

        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]

        equation = f"{labels}->{labels}"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name=op_name,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(EmbeddingHandler)
_registry.register_handler(GRUHandler)
_registry.register_handler(LSTMHandler)
_registry.register_handler(RNNHandler)
_registry.register_handler(CrossEntropyHandler)
_registry.register_handler(PairwiseLossHandler)
_registry.register_handler(TrivialOpsHandler)


__all__ = [
    "EmbeddingHandler",
    "GRUHandler",
    "LSTMHandler",
    "RNNHandler",
    "CrossEntropyHandler",
    "PairwiseLossHandler",
    "TrivialOpsHandler",
]
