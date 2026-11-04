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

MAC accounting
--------------
Forward convolutions are counted from the sliding-window equation
(``BC(P+R),OCR->BOP``): every output position sees ``C * prod(kernel)``
taps, so the generic :meth:`EinsumOp.get_compute_cost` product over
``B, O, P, C, R`` is exact.

Transposed convolutions are *scatter* operations: every input element is
multiplied with all ``(C_out / groups) * prod(kernel)`` taps, so the exact
dense count is

    MACs = N * C_in * prod(input_spatial) * (C_out / groups) * prod(kernel)
         = numel(input) * numel(weight) / C_in

independent of stride, padding, output_padding and dilation.  The
sliding-window equation ``BC(P-R),CKR->BKP`` is kept for traffic, renaming
and export purposes, but its generic product (``B, C, K, P, R`` with ``P``
the *output* length) iterates output positions and taps independently.
For the same resolved channel and kernel dimensions that output-based
product exceeds the exact count by ``prod(output_spatial) /
prod(input_spatial)``; this equals ``prod(stride)`` only when the output is
exactly ``stride`` times the input (e.g. 8x for KernelBench L2/20,
``ConvTranspose3d`` stride 2, padding 1, output_padding 1).  Padding,
dilation and output_padding change the ratio: a length-7 input with
``k=4, stride=2, padding=1, dilation=2`` gives a length-17 output and a
ratio of 17/7.  See :class:`ConvTransposeEinsumOp`.

Forward-convolution MAC accounting (issue #42)
----------------------------------------------
The sliding-window equations (``BC(P+R),OCR->BOP`` and the 2D/3D forms)
label the input channels ``C`` with the *total* input channel count, while
the weight's second dimension is ``C_in / groups``.  The generic
:meth:`EinsumOp.get_compute_cost` resolves ``C`` from the input tensor, so
grouped convolutions with ``1 < groups < C_in`` were over-counted by
``groups``x, and depthwise layers that do not take the dedicated depthwise
equation (channel multiplier ``C_out > C_in``, or any 3D depthwise) by
``C_in``x.  The forward handlers therefore return :class:`ConvEinsumOp`,
whose cost is the exact

    MACs = N * C_out * prod(output_spatial) * (C_in / groups) * prod(kernel)

read from the input, weight (``[C_out, C_in/groups, *kernel]``) and output
shapes.  Dense (``groups == 1``) layers in 1D/2D/3D and plain depthwise
layers in 1D/2D (where the handlers emit the dedicated depthwise equation)
give the same value as the generic product, so they are unchanged; plain
3D depthwise had no depthwise equation and is a corrected case.  The
equations are not modified: traffic, rank renaming and the TACO/Timeloop
exports are driven by the equation and the tensor shapes and stay as they
were (equation preservation is not numerical acceptance of the grouped
lowering).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
    conv_call_kwargs,
)


def _nd(value, nd: int) -> tuple:
    """Broadcast a scalar / 1-tuple conv argument (stride, padding, dilation) to ``nd`` dims."""
    t = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    return t * nd if len(t) == 1 and nd > 1 else t
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes


def conv_macs(input_shape: TensorShape, weight_shape: TensorShape, output_shape: TensorShape) -> int:
    """Exact dense MAC count of a forward convolution.

    ``input_shape`` is ``[N, C_in, *in_spatial]``, ``weight_shape`` the torch
    layout ``[C_out, C_in // groups, *kernel]`` and ``output_shape``
    ``[N, C_out, *out_spatial]``.  Every output element accumulates
    ``(C_in // groups) * prod(kernel)`` products, so

        MACs = N * C_out * prod(out_spatial) * (C_in // groups) * prod(kernel)

    which is the formula ``torch.utils.flop_counter.conv_flop_count`` uses
    for forward convolutions (times two for FLOPs).  Padding, stride and
    dilation enter only through the output spatial size.
    """
    macs = int(output_shape[0]) * int(output_shape[1]) * int(weight_shape[1])
    for d in output_shape[2:]:
        macs *= int(d)
    for d in weight_shape[2:]:
        macs *= int(d)
    return macs


@dataclass
class ConvEinsumOp(EinsumOp):
    """EinsumOp for forward convolutions with the exact grouped MAC count.

    Equation and operands are the unchanged sliding-window form; only
    :meth:`get_compute_cost` is overridden (see the module docstring).  The
    exact count needs the input, weight and output shapes; if the positional
    shapes do not provide a weight of matching rank or an output, the
    generic resolver is used as before.
    """

    def _torch_layout(self) -> bool:
        """True when the equation is the torch-layout sliding-window form
        (input ``[N, C, *spatial]``, weight ``[C_out, C_in/g, *k]``).  The
        the lineage base handlers also emit a group-wise form over RESHAPED tensors
        (``BGI(P+R),GOIR->BGOP``: input ``[B, G, I, *]``, weight
        ``[G, O, I, *]``); there every rank is an explicit label, the generic
        product is exact and :func:`conv_macs`'s layout assumption does not
        hold (INTEG-7)."""
        for op in self.operands:
            if not op.is_output and op.name == "Input":
                return "G" not in op.dims
        return True

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        inputs = list(tensor_shapes.inputs)
        outputs = list(tensor_shapes.outputs)
        input_shape = inputs[0] if inputs else None
        weight_shape = inputs[1] if len(inputs) > 1 else None
        output_shape = outputs[0] if outputs else None
        if (not self._torch_layout()
                or input_shape is None or weight_shape is None or output_shape is None
                or len(input_shape) < 3 or len(weight_shape) != len(input_shape)
                or len(output_shape) != len(input_shape)):
            return super().get_compute_cost(tensor_shapes)
        return conv_macs(input_shape, weight_shape, output_shape)


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

        module_args = kwargs.get("module_args", {})
        # Functional F.conv*d carries stride/padding/dilation/groups only in
        # raw_attributes; nn.Conv*d exposes them in module_args. Both paths
        # are normalised by conv_call_kwargs.
        call = conv_call_kwargs(module_args)
        stride = _nd(kwargs.get("stride") or call.get("stride") or (1,), 1)
        padding = _nd(kwargs.get("padding") or call.get("padding") or (0,), 1)
        dilation = _nd(kwargs.get("dilation") or call.get("dilation") or (1,), 1)
        groups = int(call.get("groups") or 1)
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

        return ConvEinsumOp(
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

        module_args = kwargs.get("module_args", {})
        # Functional F.conv*d carries stride/padding/dilation/groups only in
        # raw_attributes; nn.Conv*d exposes them in module_args. Both paths
        # are normalised by conv_call_kwargs.
        call = conv_call_kwargs(module_args)
        stride = _nd(kwargs.get("stride") or call.get("stride") or (1, 1), 2)
        padding = _nd(kwargs.get("padding") or call.get("padding") or (0, 0), 2)
        dilation = _nd(kwargs.get("dilation") or call.get("dilation") or (1, 1), 2)
        groups = int(call.get("groups") or 1)
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

        return ConvEinsumOp(
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

        module_args = kwargs.get("module_args", {})
        # Functional F.conv*d carries stride/padding/dilation/groups only in
        # raw_attributes; nn.Conv*d exposes them in module_args. Both paths
        # are normalised by conv_call_kwargs.
        call = conv_call_kwargs(module_args)
        stride = _nd(kwargs.get("stride") or call.get("stride") or (1, 1, 1), 3)
        padding = _nd(kwargs.get("padding") or call.get("padding") or (0, 0, 0), 3)
        dilation = _nd(kwargs.get("dilation") or call.get("dilation") or (1, 1, 1), 3)
        groups = int(call.get("groups") or 1)
        in_channels = int(module_args.get("in_channels", input_shape[1])) if module_args else input_shape[1]
        out_channels = int(module_args.get("out_channels", weight_shape[0])) if module_args else weight_shape[0]

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

        return ConvEinsumOp(
            operands=operands,
            equation=equation,
            name="conv3d",
            elementwise_op="mul",
            reduction_op="add",
        )


def convtranspose_macs(input_shape: TensorShape, weight_shape: TensorShape) -> int:
    """Exact dense MAC count of a transposed convolution.

    ``input_shape`` is ``[N, C_in, *spatial]`` and ``weight_shape`` is the
    torch layout ``[C_in, C_out // groups, *kernel]``.  Every input element
    is multiplied with every ``(C_out // groups) * prod(kernel)`` tap, so

        MACs = N * C_in * prod(spatial) * (C_out // groups) * prod(kernel)

    Stride, padding, output_padding and dilation only move the taps around
    (padding may crop some of them from the stored output); they do not
    change the number of multiplies a dense implementation performs.  This
    is the same formula ``torch.utils.flop_counter.conv_flop_count`` uses
    for ``transposed=True`` (it reports 2x for FLOPs).
    """
    macs = int(input_shape[0]) * int(input_shape[1]) * int(weight_shape[1])
    for d in input_shape[2:]:
        macs *= int(d)
    for d in weight_shape[2:]:
        macs *= int(d)
    return macs


@dataclass
class ConvTransposeEinsumOp(EinsumOp):
    """EinsumOp for transposed convolutions with the exact input-based MAC count.

    The equation and operands are the unchanged sliding-window form
    (``BC(P-R),CKR->BKP``), so byte/traffic accounting, rank renaming and
    the TACO/Timeloop exports are untouched.  Only :meth:`get_compute_cost`
    is overridden: the generic resolver would take ``P`` from the output
    shape and count ``B*C*K*P*R``, i.e. it assumes every output position has
    ``prod(kernel)`` valid taps.  For the same resolved channel and kernel
    dimensions that count is ``prod(output_spatial) / prod(input_spatial)``
    times the exact one.  The ratio equals ``prod(stride)`` only when the
    output is exactly ``stride`` times the input, as in KernelBench L2/20
    (stride 2, padding 1, output_padding 1: 8x for the 3D layer); with other
    padding/dilation/output_padding it differs (e.g. 17/7 for a length-7
    input, ``k=4, stride=2, padding=1, dilation=2``, output length 17), and
    at stride 1 it is above 1 whenever the output is longer than the input.

    ``fallback_weight_shape`` is the weight shape the handler saw (possibly
    a synthesized placeholder) and is used only when the positional shapes
    passed to :meth:`get_compute_cost` carry no weight of matching rank.
    """

    fallback_weight_shape: Optional[List[int]] = None

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        inputs = list(tensor_shapes.inputs)
        input_shape = inputs[0] if inputs else None
        if input_shape is None or len(input_shape) < 3:
            return super().get_compute_cost(tensor_shapes)
        weight_shape = inputs[1] if len(inputs) > 1 else None
        if weight_shape is None or len(weight_shape) != len(input_shape):
            # e.g. the only extra positional input is the bias, or the
            # graph carried no weight tensor at all.
            weight_shape = self.fallback_weight_shape
        if weight_shape is None or len(weight_shape) != len(input_shape):
            return super().get_compute_cost(tensor_shapes)
        return convtranspose_macs(input_shape, weight_shape)


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

        MACs = B * C * L_in * K_per_group * KL (exact input-based count,
        see :func:`convtranspose_macs`); the equation is unchanged.
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-R"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P"], is_output=True),
        ]

        equation = "BC(P-R),CKR->BKP"

        return ConvTransposeEinsumOp(
            operands=operands,
            equation=equation,
            name="convtranspose1d",
            elementwise_op="mul",
            reduction_op="add",
            fallback_weight_shape=[int(d) for d in weight_shape],
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

        MACs = B * C * H_in * W_in * K_per_group * KH * KW (exact input-based
        count, see :func:`convtranspose_macs`); the equation is unchanged.
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-R", "Q-S"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q"], is_output=True),
        ]

        equation = "BC(P-R)(Q-S),CKRS->BKPQ"

        return ConvTransposeEinsumOp(
            operands=operands,
            equation=equation,
            name="convtranspose2d",
            elementwise_op="mul",
            reduction_op="add",
            fallback_weight_shape=[int(d) for d in weight_shape],
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

        MACs = B * C * D_in * H_in * W_in * K_per_group * KD * KH * KW (exact
        input-based count, see :func:`convtranspose_macs`); the equation is
        unchanged.
        """
        # Sliding window format for transposed conv
        operands = [
            EinsumOperand("Input", ["B", "C", "P-T", "Q-R", "U-S"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "T", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q", "U"], is_output=True),
        ]

        equation = "BC(P-T)(Q-R)(U-S),CKTRS->BKPQU"

        return ConvTransposeEinsumOp(
            operands=operands,
            equation=equation,
            name="convtranspose3d",
            elementwise_op="mul",
            reduction_op="add",
            fallback_weight_shape=[int(d) for d in weight_shape],
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
    "ConvTransposeEinsumOp",
    "convtranspose_macs",
    "ConvEinsumOp",
    "conv_macs",
    "Conv1dHandler",
    "Conv2dHandler",
    "Conv3dHandler",
    "ConvTranspose1dHandler",
    "ConvTranspose2dHandler",
    "ConvTranspose3dHandler",
]
