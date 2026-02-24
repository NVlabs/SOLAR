"""Handlers for loss function operations.

This module provides einsum handlers for:
- kl_div: KL divergence loss
- cross_entropy: Cross entropy loss
- nll_loss: Negative log likelihood loss
- mse_loss: Mean squared error loss
- l1_loss: L1 loss
- bce_loss: Binary cross entropy loss
- etc.

Loss functions typically:
- Take input tensors of some shape
- Output a scalar (when reduction='mean', 'sum', or 'batchmean')
- Or output element-wise loss (when reduction='none')

Based on PyTorch documentation:
https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions
"""

import re
import string
from typing import Any, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class LossHandler(EinsumOpHandler):
    """Handler for loss function operations.
    
    Based on PyTorch documentation:
    https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions
    
    Key behavior:
    - Default reduction='mean' or 'batchmean' -> scalar output
    - reduction='none' -> output same shape as input
    - reduction='sum' -> scalar output
    """
    
    supported_ops = [
        # KL divergence
        "kl_div",
        # Cross entropy variants
        "cross_entropy",
        "nll_loss",
        "binary_cross_entropy",
        "bce_loss",
        "bce_with_logits_loss",
        # Distance-based losses
        "mse_loss",
        "l1_loss",
        "smooth_l1_loss",
        "huber_loss",
        # Embedding losses
        "cosine_embedding_loss",
        "hinge_embedding_loss",
        "margin_ranking_loss",
        "triplet_margin_loss",
        # Sequence losses
        "ctc_loss",
        # Other losses
        "poisson_nll_loss",
        "soft_margin_loss",
        "multi_margin_loss",
        "multilabel_margin_loss",
        "multilabel_soft_margin_loss",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for loss function operation.
        
        Loss functions compute a loss value from predictions and targets.
        The output shape depends on the reduction parameter.
        """
        input_shape = self._get_input_shape(shapes)
        output_shape = self._get_output_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Parse reduction mode from raw_attributes
        reduction = self._parse_reduction_mode(kwargs)
        
        # Normalize op name
        op_type = op_name.lower()
        
        return self._generate_loss_einsum(
            input_shape, output_shape, op_type, reduction
        )
    
    def _parse_reduction_mode(self, kwargs: Any) -> str:
        """Parse reduction mode from kwargs or raw_attributes.
        
        Returns one of: 'mean', 'sum', 'batchmean', 'none'
        Default is 'mean'.
        """
        # Try direct kwarg first
        reduction = kwargs.get("reduction")
        if reduction:
            return reduction
        
        # Try to extract from raw_attributes
        raw_attrs = kwargs.get("raw_attributes", "")
        if raw_attrs:
            # Match reduction: 'mean' or reduction: "batchmean" etc.
            match = re.search(r"reduction:\s*['\"]?(\w+)['\"]?", str(raw_attrs))
            if match:
                return match.group(1)
        
        # Default
        return "mean"
    
    def _generate_loss_einsum(
        self,
        input_shape: TensorShape,
        output_shape: Optional[TensorShape],
        op_type: str = "mse_loss",
        reduction: str = "mean"
    ) -> EinsumOp:
        """Generate einsum for loss operations.
        
        Args:
            input_shape: Input tensor shape (predictions).
            output_shape: Output tensor shape (from graph, may be scalar []).
            op_type: Type of loss function.
            reduction: Reduction mode ('mean', 'sum', 'batchmean', 'none').
            
        Returns:
            EinsumOp for the loss operation.
        """
        ndims = len(input_shape)
        input_labels = list(string.ascii_uppercase[:ndims])
        
        # Determine output labels based on reduction and actual output shape
        if output_shape is not None and len(output_shape) == 0:
            # Scalar output (reduction applied)
            output_labels = []
        elif reduction == "none":
            # Element-wise loss - same shape as input
            output_labels = input_labels.copy()
        else:
            # Reduction modes: mean, sum, batchmean -> scalar
            output_labels = []
        
        operands = [
            EinsumOperand("Input", input_labels, is_output=False),
            EinsumOperand("Output", output_labels, is_output=True),
        ]
        
        equation = f"{''.join(input_labels)}->{''.join(output_labels)}"
        
        # Map loss type to elementwise operation
        elementwise_op_map = {
            "kl_div": "kl_div",
            "cross_entropy": "cross_entropy",
            "nll_loss": "nll_loss",
            "mse_loss": "squared_diff",  # (pred - target)^2
            "l1_loss": "abs_diff",       # |pred - target|
            "smooth_l1_loss": "smooth_l1",
            "huber_loss": "huber",
            "bce_loss": "bce",
            "binary_cross_entropy": "bce",
            "bce_with_logits_loss": "bce_logits",
            "cosine_embedding_loss": "cosine",
            "hinge_embedding_loss": "hinge",
            "margin_ranking_loss": "margin",
            "triplet_margin_loss": "triplet",
            "ctc_loss": "ctc",
            "poisson_nll_loss": "poisson_nll",
            "soft_margin_loss": "soft_margin",
            "multi_margin_loss": "multi_margin",
            "multilabel_margin_loss": "multilabel_margin",
            "multilabel_soft_margin_loss": "multilabel_soft_margin",
        }
        
        # Reduction operation for loss (typically add then divide)
        reduction_op = "add" if reduction in {"mean", "sum", "batchmean"} else "none"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            # Loss functions are not real einsum operations
            is_real_einsum=False,
            elementwise_op=elementwise_op_map.get(op_type, op_type),
            reduction_op=reduction_op,
            is_einsum_supportable=True,
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(LossHandler)


__all__ = ["LossHandler"]
