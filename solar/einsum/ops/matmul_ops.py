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

"""Handlers for matrix multiplication operations.

This module provides einsum handlers for:
- matmul (matrix multiplication)
- linear (fully connected layer)
- bmm (batch matrix multiplication)
- mm (2D matrix multiplication)
- addmm (bias + 2D matrix multiplication, e.g. Hugging Face Conv1D)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes


class MatmulHandler(EinsumOpHandler):
    """Handler for matmul operations."""

    supported_ops = ["matmul", "mm"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None

        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")

        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]

        return self._generate_matmul_einsum(input_shape, weight_shape)

    def _generate_matmul_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication.

        Handles broadcasting where inputs can have different batch dimensions.
        PyTorch matmul broadcasting rules:
        - [B, M, K] x [K, N] -> [B, M, N]  (2D weight broadcasts)
        - [M, K] x [B, K, N] -> [B, M, N]  (2D input broadcasts)
        - [B, M, K] x [B, K, N] -> [B, M, N]  (same batch dims)

        Args:
            input_shape: Shape of first input tensor.
            other_shape: Shape of second input tensor.

        Returns:
            EinsumOp for the matmul operation.
        """
        # Handle different matmul cases based on actual tensor shapes
        # PyTorch matmul with 1D operands:
        # - [K] x [K, N] -> [N]       (vector-matrix)
        # - [M, K] x [K] -> [M]       (matrix-vector)
        # - [K] x [K] -> []           (dot product)
        input_ndim = len(input_shape)
        other_ndim = len(other_shape)

        if input_ndim == 1 and other_ndim == 1:
            # Dot product: [K] x [K] -> scalar
            operands = [
                EinsumOperand("Input", ["K"], is_output=False),
                EinsumOperand("Weight", ["K"], is_output=False),
                EinsumOperand("Output", [], is_output=True),
            ]
            equation = "K,K->"
        elif input_ndim == 1 and other_ndim == 2:
            # Vector-matrix: [K] x [K, N] -> [N]
            operands = [
                EinsumOperand("Input", ["K"], is_output=False),
                EinsumOperand("Weight", ["K", "N"], is_output=False),
                EinsumOperand("Output", ["N"], is_output=True),
            ]
            equation = "K,KN->N"
        elif input_ndim == 2 and other_ndim == 1:
            # Matrix-vector: [M, K] x [K] -> [M]
            operands = [
                EinsumOperand("Input", ["M", "K"], is_output=False),
                EinsumOperand("Weight", ["K"], is_output=False),
                EinsumOperand("Output", ["M"], is_output=True),
            ]
            equation = "MK,K->M"
        elif input_ndim == 1 and other_ndim > 2:
            # Vector with batched matrix: [K] x [B..., K, N] -> [B..., N]
            batch_dims = other_ndim - 2
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            dims_b = batch_letters + ["K", "N"]
            dims_out = batch_letters + ["N"]
            operands = [
                EinsumOperand("Input", ["K"], is_output=False),
                EinsumOperand("Weight", dims_b, is_output=False),
                EinsumOperand("Output", dims_out, is_output=True),
            ]
            equation = f"K,{''.join(dims_b)}->{''.join(dims_out)}"
        elif input_ndim == 2 and other_ndim == 2:
            # Standard 2D matmul: [M, K] x [K, N] -> [M, N]
            operands = [
                EinsumOperand("Input", ["M", "K"], is_output=False),
                EinsumOperand("Weight", ["K", "N"], is_output=False),
                EinsumOperand("Output", ["M", "N"], is_output=True),
            ]
            equation = "MK,KN->MN"
        elif input_ndim > 2 and other_ndim == 2:
            # Batched input with 2D weight: [B0, ..., M, K] x [K, N] -> [B0, ..., M, N]
            batch_dims = input_ndim - 2
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            dims_a = batch_letters + ["M", "K"]
            dims_b = ["K", "N"]  # No batch dims for weight
            dims_out = batch_letters + ["M", "N"]

            operands = [
                EinsumOperand("Input", dims_a, is_output=False),
                EinsumOperand("Weight", dims_b, is_output=False),
                EinsumOperand("Output", dims_out, is_output=True),
            ]
            equation = f"{''.join(dims_a)},{''.join(dims_b)}->{''.join(dims_out)}"
        elif input_ndim == 2 and other_ndim > 2:
            # 2D input with batched weight: [M, K] x [B0, ..., K, N] -> [B0, ..., M, N]
            batch_dims = other_ndim - 2
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            dims_a = ["M", "K"]  # No batch dims for input
            dims_b = batch_letters + ["K", "N"]
            dims_out = batch_letters + ["M", "N"]

            operands = [
                EinsumOperand("Input", dims_a, is_output=False),
                EinsumOperand("Weight", dims_b, is_output=False),
                EinsumOperand("Output", dims_out, is_output=True),
            ]
            equation = f"{''.join(dims_a)},{''.join(dims_b)}->{''.join(dims_out)}"
        else:
            # Both have batch dimensions - use the actual batch dims from each tensor
            input_batch_dims = input_ndim - 2
            other_batch_dims = other_ndim - 2
            output_batch_dims = max(input_batch_dims, other_batch_dims)

            # Generate batch letters for output (max of both)
            output_batch_letters = [f"B{i}" for i in range(output_batch_dims)]

            # Input batch dims (may be fewer than output)
            input_batch_letters = output_batch_letters[-input_batch_dims:] if input_batch_dims > 0 else []
            dims_a = input_batch_letters + ["M", "K"]

            # Other batch dims (may be fewer than output)
            other_batch_letters = output_batch_letters[-other_batch_dims:] if other_batch_dims > 0 else []
            dims_b = other_batch_letters + ["K", "N"]

            dims_out = output_batch_letters + ["M", "N"]

            operands = [
                EinsumOperand("Input", dims_a, is_output=False),
                EinsumOperand("Weight", dims_b, is_output=False),
                EinsumOperand("Output", dims_out, is_output=True),
            ]
            equation = f"{''.join(dims_a)},{''.join(dims_b)}->{''.join(dims_out)}"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="matmul",
            elementwise_op="mul",
            reduction_op="add",
        )


class LinearHandler(EinsumOpHandler):
    """Handler for linear (fully connected) layers."""

    supported_ops = ["linear"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for linear layer."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None

        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")

        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]

        return self._generate_linear_einsum(input_shape, weight_shape)

    def _generate_linear_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for linear layer.

        Args:
            input_shape: Shape of input tensor.
            weight_shape: Shape of weight tensor.

        Returns:
            EinsumOp for the linear operation.
        """
        # Linear: input @ weight.T
        batch_dims = len(input_shape) - 1
        batch_letters = [f"B{i}" for i in range(batch_dims)]

        input_dims = batch_letters + ["K"]
        weight_dims = ["N", "K"]  # Weight is [out_features, in_features]
        output_dims = batch_letters + ["N"]

        operands = [
            EinsumOperand("Input", input_dims, is_output=False),
            EinsumOperand("Weight", weight_dims, is_output=False),
            EinsumOperand("Output", output_dims, is_output=True),
        ]

        input_str = ''.join(input_dims)
        weight_str = ''.join(weight_dims)
        output_str = ''.join(output_dims)
        equation = f"{input_str},{weight_str}->{output_str}"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="linear",
            elementwise_op="mul",
            reduction_op="add",
        )


class BmmHandler(EinsumOpHandler):
    """Handler for batch matrix multiplication."""

    supported_ops = ["bmm"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None

        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")

        return self._generate_bmm_einsum(input_shape, weight_shape)

    def _generate_bmm_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication.

        bmm: [B, M, K] x [B, K, N] -> [B, M, N]
        """
        operands = [
            EinsumOperand("Input", ["B", "M", "K"], is_output=False),
            EinsumOperand("Weight", ["B", "K", "N"], is_output=False),
            EinsumOperand("Output", ["B", "M", "N"], is_output=True),
        ]

        equation = "BMK,BKN->BMN"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="bmm",
            elementwise_op="mul",
            reduction_op="add",
        )


class AddmmHandler(EinsumOpHandler):
    """Handler for ``torch.addmm(bias, mat1, mat2)``.

    ``addmm`` is ``bias[N] + mat1[M, K] @ mat2[K, N]`` with the weight stored
    ``[in, out]`` (Hugging Face ``Conv1D``).  It is billed exactly like a
    matmul with a bias: M*K*N MACs, with bias, mat1 and mat2 read and the
    output written.  The converter normally splits it into a matmul layer and
    a ``bias_add`` layer (the nn.Linear convention); this handler covers the
    direct path and must agree with that split on MACs and traffic.
    """

    supported_ops = ["addmm"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for addmm (bias first, then mat1 and mat2).

        Operands are bound by position (``bias, mat1, mat2``) and never by
        provenance.  The matrix work is M*K*N from mat1/mat2 alone: a bias
        dimension that broadcasts (extent 1) is labelled with its own rank
        (``P`` over M, ``Q`` over N) so it never redefines M or N, while the
        bias operand keeps its actual storage extent for traffic.  Two
        recorded 2D matrices (no bias captured) are a checked matmul; any
        other operand count, rank or extent mismatch is refused.  A missing
        matrix is never synthesised.
        """
        inputs = list(tensor_shapes.inputs)
        if len(inputs) == 2:
            mat1_shape, mat2_shape = self._checked_2d_pair(op_name, inputs[0], inputs[1])
            return MatmulHandler(debug=self.debug)._generate_matmul_einsum(mat1_shape, mat2_shape)
        if len(inputs) != 3:
            raise ValueError(
                f"{op_name} expects (bias, mat1, mat2) or exactly two 2D matrices, "
                f"got {len(inputs)} input shape(s)")

        bias_shape, mat1_shape, mat2_shape = inputs
        mat1_shape, mat2_shape = self._checked_2d_pair(op_name, mat1_shape, mat2_shape)
        m, n = mat1_shape[0], mat2_shape[1]
        if not isinstance(bias_shape, (list, tuple)) or len(bias_shape) not in (1, 2):
            raise ValueError(f"{op_name} expects a rank-1 or rank-2 bias, got {bias_shape!r}")
        bias_dims: List[str] = []
        axes = [("N", n, "Q")] if len(bias_shape) == 1 else [("M", m, "P"), ("N", n, "Q")]
        for extent, (label, full, broadcast_label) in zip(bias_shape, axes):
            extent = int(extent)
            if extent == full:
                bias_dims.append(label)
            elif extent == 1:
                bias_dims.append(broadcast_label)
            else:
                raise ValueError(
                    f"{op_name} bias {list(bias_shape)} does not broadcast to the "
                    f"[{m}, {n}] output")

        operands = [
            EinsumOperand("Bias", bias_dims, is_output=False),
            EinsumOperand("Input", ["M", "K"], is_output=False),
            EinsumOperand("Weight", ["K", "N"], is_output=False),
            EinsumOperand("Output", ["M", "N"], is_output=True),
        ]
        equation = f"{''.join(bias_dims)},MK,KN->MN"
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="addmm",
            elementwise_op="mul",
            reduction_op="add",
        )

    @staticmethod
    def _checked_2d_pair(op_name: str, mat1_shape: Any, mat2_shape: Any) -> Tuple[List[int], List[int]]:
        """Return (mat1, mat2) as 2D int lists with matching inner extents, else refuse."""
        for label, shape in (("mat1", mat1_shape), ("mat2", mat2_shape)):
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                rank = len(shape) if isinstance(shape, (list, tuple)) else None
                raise ValueError(
                    f"{op_name} expects a 2D {label}, got rank {rank} ({shape!r})")
        mat1 = [int(d) for d in mat1_shape]
        mat2 = [int(d) for d in mat2_shape]
        if mat1[1] != mat2[0]:
            raise ValueError(
                f"{op_name} inner dimensions disagree: mat1 {mat1} vs mat2 {mat2}")
        return mat1, mat2


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(MatmulHandler)
_registry.register_handler(LinearHandler)
_registry.register_handler(BmmHandler)
_registry.register_handler(AddmmHandler)




# ---------------------------------------------------------------------------
# torch._scaled_mm (FP8 / NVFP4 block-scaled matmul)
# ---------------------------------------------------------------------------

def _dtype_key(dtype: Any) -> str:
    return str(dtype or "").lower().replace("torch.", "").strip()


# Storage dtypes that pack more than one logical element per stored element
# along the reduction dimension.  ``float4_e2m1fn_x2`` stores two E2M1
# values per byte: a tensor recorded as [M, 4096] float4_e2m1fn_x2 holds a
# logical [M, 8192] operand (route 195 reference: ``unpack_e2m1_x2``, and
# the NVFP4 block scales recorded next to it: numel(scale_a) = M * K/16).
_PACKED_REDUCTION_FACTOR: Dict[str, int] = {"float4_e2m1fn_x2": 2}


@dataclass
class ScaledMmOp(EinsumOp):
    """``torch._scaled_mm(A, B, scale_a, scale_b, ...)``: a 2-D matmul whose
    MACs are counted over the LOGICAL reduction extent.

    ``m``, ``n`` are the output extents; ``k_logical`` the reduction extent
    after unpacking (``k_stored * pack_factor``).  The scale tensors are not
    operands of the contraction (they scale rows / blocks of A and B); they
    are additional reads for the traffic model and carry no MACs.
    """
    m: int = 0
    n: int = 0
    k_stored: int = 0
    k_logical: int = 0
    pack_factor: int = 1
    compute_dtype: Optional[str] = None
    scale_shapes: Optional[List[Any]] = None

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        return int(self.m) * int(self.k_logical) * int(self.n)


class ScaledMmHandler(EinsumOpHandler):
    """Handler for ``torch._scaled_mm`` (and its ``.out`` / ``scaled_mm`` spellings).

    Operand layout follows the recorded call: ``A [M, K]`` and ``B`` either
    ``[K, N]`` (the usual column-major second operand, e.g. a transposed
    weight) or ``[N, K]``; the orientation is taken from which extent of B
    equals A's reduction extent, and the output shape (when recorded) must
    agree.  Inputs 2 and 3 are the scales (per-tensor scalars or NVFP4 /
    MX block scales), input 4 an optional bias: never MAC-bearing.

    MACs = M * K_logical * N where K_logical = K_stored * pack_factor and the
    pack factor comes from the RECORDED dtype of the matrix operands
    (``float4_e2m1fn_x2`` -> 2; fp8 / bf16 -> 1).  Traffic stays at the
    recorded (stored) shapes and dtype widths: 1 B per packed FP4 pair, 1 B
    per fp8 element, the output at its own dtype.
    """

    supported_ops = ["scaled_mm", "_scaled_mm", "_scaled_mm.out", "scaled_mm.out"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        input_dtypes: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> EinsumOp:
        ins = list(tensor_shapes.inputs)
        if len(ins) < 2 or ins[0] is None or ins[1] is None:
            raise ValueError(f"{op_name}: needs the two matrix operands, got {len(ins)} input shapes")
        a, b = list(ins[0]), list(ins[1])
        if len(a) != 2 or len(b) != 2:
            raise ValueError(f"{op_name}: matrix operands must be 2-D, got {a} and {b}")
        m, k_stored = int(a[0]), int(a[1])
        if int(b[0]) == k_stored:
            n = int(b[1]); weight_dims = ["K", "N"]; equation = "MK,KN->MN"
        elif int(b[1]) == k_stored:
            n = int(b[0]); weight_dims = ["N", "K"]; equation = "MK,NK->MN"
        else:
            raise ValueError(f"{op_name}: no reduction extent shared between A {a} and B {b}")
        out = tensor_shapes.outputs[0] if tensor_shapes.outputs else None
        if out is not None and [int(x) for x in out] != [m, n]:
            raise ValueError(f"{op_name}: recorded output {list(out)} is not [M, N] = {[m, n]}")

        dtypes = [_dtype_key(d) for d in (input_dtypes or [])]
        matrix_dtypes = [d for d in dtypes[:2] if d]
        if len(matrix_dtypes) < 2:
            # The stored shapes alone do not establish the reduction extent:
            # [M, 4096] is a logical K of 4096 for fp8 / bf16 and of 8192 for
            # float4_e2m1fn_x2.  Without BOTH recorded matrix dtypes nothing is
            # assumed (evidence-1); the caller reports the layer incomplete.
            raise ValueError(f"{op_name}: the matrix operand dtypes are not recorded (A: {dtypes[0] if dtypes else None!r}, "
                             f"B: {dtypes[1] if len(dtypes) > 1 else None!r}); the packing of the reduction extent cannot be established")
        factors = {_PACKED_REDUCTION_FACTOR.get(d, 1) for d in matrix_dtypes}
        if len(factors) > 1:
            raise ValueError(f"{op_name}: matrix operands disagree on packing: {matrix_dtypes}")
        pack = factors.pop()
        compute_dtype = None
        if any(d.startswith("float4_e2m1") for d in matrix_dtypes):
            compute_dtype = "nvfp4"
        elif any(d.startswith("float8") for d in matrix_dtypes):
            compute_dtype = "fp8"

        operands = [
            EinsumOperand("Input", ["M", "K"], is_output=False),
            EinsumOperand("Weight", weight_dims, is_output=False),
            EinsumOperand("Output", ["M", "N"], is_output=True),
        ]
        return ScaledMmOp(
            operands=operands, equation=equation, name="scaled_mm",
            is_real_einsum=True, elementwise_op="mul", reduction_op="add",
            m=m, n=n, k_stored=k_stored, k_logical=k_stored * pack, pack_factor=pack,
            compute_dtype=compute_dtype, scale_shapes=[list(s) for s in ins[2:4] if s is not None],
        )

_registry.register_handler(ScaledMmHandler)
__all__ = ["MatmulHandler", "LinearHandler", "BmmHandler", "AddmmHandler", "ScaledMmHandler", "ScaledMmOp"]
