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

"""Handlers for convolution operations.

This module provides einsum handlers for:
- conv1d, conv2d, conv3d
- convtranspose1d, convtranspose2d, convtranspose3d
"""

from typing import Any, List, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes


class Conv1dHandler(EinsumOpHandler):
    """Handler for 1D convolution."""
    
    supported_ops = ["conv1d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 1D convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1,)))
        padding = tuple(kwargs.get("padding", (0,)))
        dilation = tuple(kwargs.get("dilation", (1,)))
        
        return self._generate_conv1d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int] = (1,),
        padding: Tuple[int] = (0,),
        dilation: Tuple[int] = (1,)
    ) -> EinsumOp:
        """Generate einsum for 1D convolution.

        For standard conv: BC(P+R),OCR->BOP (reduces over C and R)
        For depthwise conv (weight C_per_group=1): BO(P+R),O1R->BOP
          (no cross-channel reduction, each channel is independent)

        Depthwise is detected when weight_shape[1] == 1 and O == C,
        meaning groups=C (each channel has its own filter).
        """
        B, C, L = input_shape
        O, C_per_group, KL = weight_shape

        L_out = (L + 2 * padding[0] - dilation[0] * (KL - 1) - 1) // stride[0] + 1

        if C_per_group == 1 and O == C:
            # Depthwise convolution: groups=C, each channel independent
            # MACs = B * O * L_out * KL (no cross-channel reduction)
            operands = [
                EinsumOperand("Input", ["B", "O", "P+R"], is_output=False),
                EinsumOperand("Weight", ["O", "1", "R"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P"], is_output=True),
            ]
            equation = "BO(P+R),O1R->BOP"
        else:
            # Standard convolution: reduces over C and R
            # MACs = B * O * L_out * C * KL
            operands = [
                EinsumOperand("Input", ["B", "C", "P+R"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P"], is_output=True),
            ]
            equation = "BC(P+R),OCR->BOP"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="conv1d",
            elementwise_op="mul",
            reduction_op="add",
        )


class Conv2dHandler(EinsumOpHandler):
    """Handler for 2D convolution."""
    
    supported_ops = ["conv2d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 2D convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1, 1)))
        padding = tuple(kwargs.get("padding", (0, 0)))
        dilation = tuple(kwargs.get("dilation", (1, 1)))
        
        return self._generate_conv2d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1)
    ) -> EinsumOp:
        """Generate einsum for 2D convolution.
        
        Uses sliding window format: BC(P+R)(Q+S),OCRS->BOPQ
        where P,Q are output spatial positions and R,S are kernel positions.
        The input spatial dimensions are expressed as (P+R) and (Q+S) to show
        the sliding window relationship that can be flattened into loops.
        
        This maps directly to the nested loop structure:
            for b in B:
                for o in O:
                    for p in P:
                        for q in Q:
                            for c in C:
                                for r in R:
                                    for s in S:
                                        out[b,o,p,q] += in[b,c,p+r,q+s] * w[o,c,r,s]
        """
        B, C, H, W = input_shape
        O, C_per_group, KH, KW = weight_shape

        H_out = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1

        if C_per_group == 1 and O == C:
            # Depthwise conv2d: groups=C, each channel independent
            # MACs = B * O * H_out * W_out * KH * KW
            operands = [
                EinsumOperand("Input", ["B", "O", "P+R", "Q+S"], is_output=False),
                EinsumOperand("Weight", ["O", "1", "R", "S"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P", "Q"], is_output=True),
            ]
            equation = "BO(P+R)(Q+S),O1RS->BOPQ"
        else:
            # Standard or group-wise conv2d
            # The weight C dimension is C_per_group (= C/groups).
            # MACs = B * O * H_out * W_out * C_per_group * KH * KW
            operands = [
                EinsumOperand("Input", ["B", "C", "P+R", "Q+S"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R", "S"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P", "Q"], is_output=True),
            ]
            equation = "BC(P+R)(Q+S),OCRS->BOPQ"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="conv2d",
            elementwise_op="mul",
            reduction_op="add",
        )


class Conv3dHandler(EinsumOpHandler):
    """Handler for 3D convolution."""
    
    supported_ops = ["conv3d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 3D convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1, 1, 1)))
        padding = tuple(kwargs.get("padding", (0, 0, 0)))
        dilation = tuple(kwargs.get("dilation", (1, 1, 1)))
        
        return self._generate_conv3d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1)
    ) -> EinsumOp:
        """Generate einsum for 3D convolution.
        
        Uses sliding window format: BC(P+T)(Q+R)(U+S),OCTRS->BOPQU
        where P,Q,U are output spatial positions and T,R,S are kernel positions.
        The input spatial dimensions are expressed as (P+T), (Q+R), (U+S) to show
        the sliding window relationship that can be flattened into loops.
        """
        B, C, D, H, W = input_shape
        O, _, KD, KH, KW = weight_shape
        
        D_out = (D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) // stride[0] + 1
        H_out = (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
        W_out = (W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) // stride[2] + 1
        
        # Sliding window format: Input[B,C,P+T,Q+R,U+S] * Weight[O,C,T,R,S] -> Output[B,O,P,Q,U]
        # P,Q,U are output positions, T,R,S are kernel positions
        operands = [
            EinsumOperand("Input", ["B", "C", "P+T", "Q+R", "U+S"], is_output=False),
            EinsumOperand("Weight", ["O", "C", "T", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "O", "P", "Q", "U"], is_output=True),
        ]
        
        # Sliding window einsum: BC(P+T)(Q+R)(U+S),OCTRS->BOPQU
        equation = "BC(P+T)(Q+R)(U+S),OCTRS->BOPQU"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="conv3d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose1dHandler(EinsumOpHandler):
    """Handler for 1D transposed convolution."""
    
    supported_ops = ["convtranspose1d", "conv_transpose1d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 1D transposed convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3]
        
        return self._generate_convtranspose1d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 1D transposed convolution.
        
        Uses sliding window format: BC(P-R),CKR->BKP
        For transposed conv, output position P maps to input position (P-R).
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-R"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P"], is_output=True),
        ]
        
        equation = "BC(P-R),CKR->BKP"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose1d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose2dHandler(EinsumOpHandler):
    """Handler for 2D transposed convolution."""
    
    supported_ops = ["convtranspose2d", "conv_transpose2d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 2D transposed convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3, 3]
        
        return self._generate_convtranspose2d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 2D transposed convolution.
        
        Uses sliding window format: BC(P-R)(Q-S),CKRS->BKPQ
        For transposed conv, output positions P,Q map to input positions (P-R),(Q-S).
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-R", "Q-S"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q"], is_output=True),
        ]
        
        equation = "BC(P-R)(Q-S),CKRS->BKPQ"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose2d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose3dHandler(EinsumOpHandler):
    """Handler for 3D transposed convolution."""
    
    supported_ops = ["convtranspose3d", "conv_transpose3d"]
    
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 3D transposed convolution."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3, 3, 3]
        
        return self._generate_convtranspose3d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 3D transposed convolution.
        
        Uses sliding window format: BC(P-T)(Q-R)(U-S),CKTRS->BKPQU
        For transposed conv, output positions P,Q,U map to input positions (P-T),(Q-R),(U-S).
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-T", "Q-R", "U-S"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "T", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q", "U"], is_output=True),
        ]
        
        equation = "BC(P-T)(Q-R)(U-S),CKTRS->BKPQU"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose3d",
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(Conv1dHandler)
_registry.register_handler(Conv2dHandler)
_registry.register_handler(Conv3dHandler)
_registry.register_handler(ConvTranspose1dHandler)
_registry.register_handler(ConvTranspose2dHandler)
_registry.register_handler(ConvTranspose3dHandler)


__all__ = [
    "Conv1dHandler",
    "Conv2dHandler",
    "Conv3dHandler",
    "ConvTranspose1dHandler",
    "ConvTranspose2dHandler",
    "ConvTranspose3dHandler",
]

