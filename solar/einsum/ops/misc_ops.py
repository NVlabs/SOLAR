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
from typing import Any, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes


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
        
        return self._generate_gru_einsum(input_shape, hidden, w_ih, w_hh)
    
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
        
        return self._generate_lstm_einsum(input_shape, hidden, cell, w_ih, w_hh)
    
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
    """Handler for basic RNN operations."""
    
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
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="rnn",
            elementwise_op="mul",
            reduction_op="add",
        )


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

