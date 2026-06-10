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
        module_args = kwargs.get("module_args", {})
        groups = int(module_args.get("groups", 1)) if module_args else 1
        in_channels = int(module_args.get("in_channels", input_shape[1])) if module_args else input_shape[1]
        out_channels = int(module_args.get("out_channels", weight_shape[0])) if module_args else weight_shape[0]
        
        return self._generate_conv1d_einsum(
            input_shape, weight_shape, stride, padding, dilation,
            groups, in_channels, out_channels,
        )
    
    def _generate_conv1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int] = (1,),
        padding: Tuple[int] = (0,),
        dilation: Tuple[int] = (1,),
        groups: int = 1,
        in_channels: int = 0,
        out_channels: int = 0,
    ) -> EinsumOp:
        """Generate einsum for 1D convolution.

        Three cases based on groups vs in_channels/out_channels:
        - Standard (groups==1): BC(P+R),OCR->BOP
        - Depthwise (groups==in_channels==out_channels): BO(P+R),OCR->BOP
        - Group-wise (otherwise): BGI(P+R),GOIR->BGOP  (expects reshaped tensors)
        """
        if groups == 1:
            operands = [
                EinsumOperand("Input", ["B", "C", "P+R"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P"], is_output=True),
            ]
            equation = "BC(P+R),OCR->BOP"
        elif groups == in_channels and groups == out_channels:
            operands = [
                EinsumOperand("Input", ["B", "O", "P+R"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P"], is_output=True),
            ]
            equation = "BO(P+R),OCR->BOP"
        else:
            operands = [
                EinsumOperand("Input", ["B", "G", "I", "P+R"], is_output=False),
                EinsumOperand("Weight", ["G", "O", "I", "R"], is_output=False),
                EinsumOperand("Output", ["B", "G", "O", "P"], is_output=True),
            ]
            equation = "BGI(P+R),GOIR->BGOP"

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
        module_args = kwargs.get("module_args", {})
        groups = int(module_args.get("groups", 1)) if module_args else 1
        in_channels = int(module_args.get("in_channels", input_shape[1])) if module_args else input_shape[1]
        out_channels = int(module_args.get("out_channels", weight_shape[0])) if module_args else weight_shape[0]
        
        return self._generate_conv2d_einsum(
            input_shape, weight_shape, stride, padding, dilation,
            groups, in_channels, out_channels,
        )
    
    def _generate_conv2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        in_channels: int = 0,
        out_channels: int = 0,
    ) -> EinsumOp:
        """Generate einsum for 2D convolution.

        Three cases based on groups vs in_channels/out_channels:
        - Standard (groups==1): BC(P+R)(Q+S),OCRS->BOPQ
        - Depthwise (groups==in_channels==out_channels): BO(P+R)(Q+S),OCRS->BOPQ
        - Group-wise (otherwise): BGI(P+R)(Q+S),GOIRS->BGOPQ  (expects reshaped tensors)
        """
        if groups == 1:
            operands = [
                EinsumOperand("Input", ["B", "C", "P+R", "Q+S"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R", "S"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P", "Q"], is_output=True),
            ]
            equation = "BC(P+R)(Q+S),OCRS->BOPQ"
        elif groups == in_channels and groups == out_channels:
            operands = [
                EinsumOperand("Input", ["B", "O", "P+R", "Q+S"], is_output=False),
                EinsumOperand("Weight", ["O", "C", "R", "S"], is_output=False),
                EinsumOperand("Output", ["B", "O", "P", "Q"], is_output=True),
            ]
            equation = "BO(P+R)(Q+S),OCRS->BOPQ"
        else:
            operands = [
                EinsumOperand("Input", ["B", "G", "I", "P+R", "Q+S"], is_output=False),
                EinsumOperand("Weight", ["G", "O", "I", "R", "S"], is_output=False),
                EinsumOperand("Output", ["B", "G", "O", "P", "Q"], is_output=True),
            ]
            equation = "BGI(P+R)(Q+S),GOIRS->BGOPQ"

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

