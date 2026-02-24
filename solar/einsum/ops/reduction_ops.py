"""Handlers for reduction operations.

This module provides einsum handlers for:
- sum, mean, prod
- max, min, amax, amin
- argmax, argmin
- logsumexp, norm
"""

import string
from typing import Any, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class ReductionHandler(EinsumOpHandler):
    """Handler for reduction operations.
    
    Based on PyTorch documentation:
    https://docs.pytorch.org/docs/stable/nn.functional.html
    
    All these operations support dim and keepdim parameters.
    """
    
    supported_ops = [
        # Standard reductions
        "sum", "mean", "prod",
        # Value reductions
        "max", "min", "amax", "amin",
        # Index reductions (return indices, not values)
        "argmax", "argmin",
        # Special reductions
        "logsumexp", "norm",
        # Statistical reductions
        "std", "var",
        # Boolean reductions
        "all", "any",
        # NaN-aware reductions
        "nansum", "nanmean",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for reduction operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Get reduction dimensions
        dims = kwargs.get("dims")
        if dims is None:
            dims = kwargs.get("reduce_dims")
        
        keepdim = bool(kwargs.get("keepdim", False))
        
        # Normalize op name
        op_type = op_name.lower()
        if op_type in {"amax", "amin"}:
            op_type = op_type[1:]  # amax -> max, amin -> min
        
        return self._generate_reduction_einsum(
            input_shape, op_type, dims, keepdim
        )
    
    def _generate_reduction_einsum(
        self,
        shape: TensorShape,
        op_type: str = "sum",
        dims: Optional[List[int]] = None,
        keepdim: bool = False
    ) -> EinsumOp:
        """Generate einsum for reduction operations.
        
        Args:
            shape: Input tensor shape.
            op_type: Type of reduction (sum, mean, max, etc.).
            dims: Dimensions to reduce along.
            keepdim: Whether to keep reduced dimensions (size 1).
            
        Returns:
            EinsumOp for the reduction operation.
            
        When keepdim=True, reduced dimensions are kept with a special marker.
        For example, sum over dim 1 with keepdim=True:
            Input: ABC -> Output: A1C (where 1 represents the kept dimension)
        In einsum notation, we use the same label but mark it as reduced:
            ABC->A[B]C where [B] indicates B is reduced but kept
        For simplicity, we keep the label in output when keepdim=True.
        """
        ndims = len(shape)
        input_labels = list(string.ascii_uppercase[:ndims])
        
        # Normalize dims to handle negative indices
        if dims is not None:
            normalized_dims = []
            for d in dims:
                if d < 0:
                    d = ndims + d
                normalized_dims.append(d)
            dims = normalized_dims
        
        # Determine output labels based on reduction dims and keepdim
        if dims is None:
            # Reduce all dimensions
            if keepdim:
                # Keep all dims but they become size 1
                output_labels = input_labels.copy()
            else:
                output_labels = []
        else:
            if keepdim:
                # Keep all labels, reduced dims will have size 1
                output_labels = input_labels.copy()
            else:
                # Remove reduced dimensions from output
                output_labels = []
                for i, label in enumerate(input_labels):
                    if i not in dims:
                        output_labels.append(label)
        
        operands = [
            EinsumOperand("Input", input_labels, is_output=False),
            EinsumOperand("Output", output_labels, is_output=True),
        ]
        
        equation = f"{''.join(input_labels)}->{''.join(output_labels)}"
        
        # Map reduction op_type to appropriate reduction_op
        reduction_op_map = {
            # Standard reductions
            "sum": "add",
            "mean": "add",  # Mean is sum then divide
            "prod": "mul",
            # Value reductions
            "max": "max",
            "min": "min",
            "amax": "max",
            "amin": "min",
            # Index reductions
            "argmax": "max",
            "argmin": "min",
            # Special reductions
            "logsumexp": "add",
            "norm": "add",
            # Statistical reductions
            "std": "add",  # std involves sum of squared differences
            "var": "add",  # var involves sum of squared differences
            # Boolean reductions
            "all": "and",
            "any": "or",
            # NaN-aware reductions
            "nansum": "add",
            "nanmean": "add",
        }
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op=reduction_op_map.get(op_type, "add"),
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(ReductionHandler)


__all__ = ["ReductionHandler"]

