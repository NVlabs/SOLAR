"""Handlers for elementwise operations.

This module provides einsum handlers for:
- Unary elementwise: relu, sigmoid, tanh, gelu, softmax, abs, exp, log, etc.
- Binary elementwise: add, sub, mul, div
"""

import string
from typing import Any, List

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class UnaryElementwiseHandler(EinsumOpHandler):
    """Handler for unary elementwise operations."""
    
    supported_ops = [
        "relu", "leaky_relu", "prelu", "rrelu",
        "sigmoid", "tanh", "gelu", "selu", "elu", "celu", "mish", "silu",
        "softmax", "log_softmax", "softplus", "softsign", "hardswish", "hardsigmoid", "hardtanh",
        "abs", "neg", "exp", "log", "log2", "log10", "sqrt", "rsqrt", "sin", "cos", "tan",
        "clamp", "clamp_", "relu_", "leaky_relu_",
        "dropout", "dropout_",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for unary elementwise operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_elementwise_einsum(input_shape, op_name)
    
    def _generate_elementwise_einsum(
        self,
        shape: TensorShape,
        op_type: str = "elementwise"
    ) -> EinsumOp:
        """Generate einsum for unary elementwise operations.
        
        Args:
            shape: Input tensor shape.
            op_type: Type of elementwise operation (e.g., relu, sigmoid, tanh).
            
        Returns:
            EinsumOp for the elementwise operation.
        """
        dims = len(shape)
        labels = string.ascii_uppercase[:dims]
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]
        
        equation = f"{labels}->{labels}"
        
        # Normalize op name (remove trailing underscore for inplace ops)
        normalized_op = op_type.rstrip("_")
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op=normalized_op,  # Use actual operation name
            reduction_op="none",
        )


class BinaryElementwiseHandler(EinsumOpHandler):
    """Handler for binary elementwise operations."""
    
    supported_ops = [
        "add", "sub", "mul", "div", "pow",
        "add_", "sub_", "mul_", "div_",
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Get second input shape - try multiple keys
        input_1_shape = (
            shapes.get("Input_1") or 
            shapes.get("Weight") 
        )
        
        # Normalize op name (remove underscores and dunder)
        op_type = op_name.lower().rstrip("_")
        if op_type.startswith("__"):
            op_type = op_type[2:]
        if op_type.startswith("r"):
            op_type = op_type[1:]  # __radd__ -> add
        
        if input_1_shape is not None:
            einsum_op = self._generate_binary_elementwise_einsum(
                input_shape, input_1_shape, op_type
            )
            # Build tensor_shapes for validation
            output_shape = self._get_output_shape(shapes)
            tensor_shapes = {
                "inputs": [list(input_shape), list(input_1_shape)],
                "outputs": [list(output_shape)] if output_shape else []
            }
            # Validate and fix if needed
            return self._validate_einsum(einsum_op, tensor_shapes)
        
        # Fallback to unary (scalar broadcast case)
        einsum_op = self._generate_unary_elementwise_einsum(input_shape, op_type)
        output_shape = self._get_output_shape(shapes)
        tensor_shapes = {
            "inputs": [list(input_shape)],
            "outputs": [list(output_shape)] if output_shape else []
        }
        return self._validate_einsum(einsum_op, tensor_shapes)
    
    def _generate_binary_elementwise_einsum(
        self,
        input_shape: TensorShape,
        input_1_shape: TensorShape,
        op_type: str = "add"
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operations with broadcasting.
        
        Handles NumPy-style broadcasting where shapes are aligned from the right.
        For example:
            [32768, 32768] * [32768] -> [32768, 32768]
            einsum: AB,B->AB (second input broadcasts along first dim)
        
        Args:
            input_shape: Shape of first input tensor.
            input_1_shape: Shape of second input tensor.
            op_type: Type of binary operation.
            
        Returns:
            EinsumOp for the binary elementwise operation.
        """
        if not input_shape or not input_1_shape:
            raise ValueError(f"Input shapes cannot be empty for {op_type}")
        
        # Handle broadcasting: compute output shape
        max_dims = max(len(input_shape), len(input_1_shape))
        
        # Pad shorter shape with 1s at the front (broadcasting aligns from right)
        padded_input = [1] * (max_dims - len(input_shape)) + list(input_shape)
        padded_input_1 = [1] * (max_dims - len(input_1_shape)) + list(input_1_shape)
        
        # Compute broadcast output shape
        output_shape = []
        for d1, d2 in zip(padded_input, padded_input_1):
            if d1 == d2 or d1 == 1 or d2 == 1:
                output_shape.append(max(d1, d2))
            else:
                raise ValueError(
                    f"Incompatible shapes for broadcasting: {input_shape} and {input_1_shape}"
                )
        
        # Generate dimension labels for output (full rank)
        output_labels = list(string.ascii_uppercase[:max_dims])
        
        # Build dimension labels for each input based on actual rank
        # Align from the right (broadcasting semantics)
        input_labels = output_labels[-(len(input_shape)):] if input_shape else []
        input_1_labels = output_labels[-(len(input_1_shape)):] if input_1_shape else []
        
        equation = f"{''.join(input_labels)},{''.join(input_1_labels)}->{''.join(output_labels)}"
        
        operands = [
            EinsumOperand("Input", input_labels, is_output=False),
            EinsumOperand("Input_1", input_1_labels, is_output=False),
            EinsumOperand("Output", output_labels, is_output=True),
        ]
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=op_type,
            is_real_einsum=False,
            elementwise_op=op_type,
            reduction_op="none",
        )
    
    def _generate_unary_elementwise_einsum(
        self,
        shape: TensorShape,
        op_type: str
    ) -> EinsumOp:
        """Generate einsum for scalar broadcast case."""
        dims = len(shape)
        labels = string.ascii_uppercase[:dims]
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]
        
        equation = f"{labels}->{labels}"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op=op_type,
            reduction_op="none",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(UnaryElementwiseHandler)
_registry.register_handler(BinaryElementwiseHandler)


__all__ = ["UnaryElementwiseHandler", "BinaryElementwiseHandler"]

