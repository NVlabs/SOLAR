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

"""Handlers for shape/view operations.

This module provides einsum handlers for:
- view, reshape, flatten, unflatten
- squeeze, unsqueeze, expand, repeat
- transpose, permute, t, contiguous
- cat, concat, stack, split, chunk
- getitem, select, index_select
"""

import string
from typing import Any, Dict, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


def generate_dim_labels(num_dims: int, prefix: str = "") -> List[str]:
    """Generate dimension labels that support more than 26 dimensions.
    
    Uses a char+number scheme:
    - First 26 dims: A, B, C, ..., Z (or with prefix: I0, I1, ..., I25)
    - Beyond 26: A0, A1, ..., Z0, Z1, ... (or I26, I27, ...)
    
    Args:
        num_dims: Number of dimension labels to generate.
        prefix: Optional uppercase prefix for labels (e.g., "I" for input, "O" for output).
        
    Returns:
        List of dimension label strings.
    """
    if prefix:
        # Use uppercase prefix + number scheme: I0, I1, I2, ... or O0, O1, O2, ...
        prefix_upper = prefix.upper()
        return [f"{prefix_upper}{i}" for i in range(num_dims)]
    
    # Use letter-based scheme with numbers for overflow
    labels = []
    for i in range(num_dims):
        if i < 26:
            # First 26: A, B, C, ..., Z
            labels.append(string.ascii_uppercase[i])
        else:
            # Beyond 26: A0, A1, ..., Z0, Z1, A2, ...
            letter_idx = (i - 26) % 26
            number = (i - 26) // 26
            labels.append(f"{string.ascii_uppercase[letter_idx]}{number}")
    return labels


class TensorManipulationHandler(EinsumOpHandler):
    """Handler for tensor manipulation operations."""
    
    # Operations that change the layout (different input/output ranks)
    RESHAPE_OPS = {
        "view", "reshape", "flatten", "unflatten",
        "squeeze", "unsqueeze", "expand", "repeat",
        "cat", "concat", "stack", "split", "chunk",
        "__getitem__", "getitem", "select", "index_select",
    }
    
    # Operations that reorder dimensions (same ranks, different order)
    TRANSPOSE_OPS = {"transpose", "permute", "t", "contiguous"}
    
    supported_ops = list(RESHAPE_OPS | TRANSPOSE_OPS)
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for tensor manipulation operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        output_shape = self._get_output_shape(shapes) or input_shape
        
        # Get module_args for explicit permutation info
        module_args = kwargs.get('module_args', {})
        
        # Transpose/permute operations use same dimension labels (reordered)
        if op_name.lower() in self.TRANSPOSE_OPS:
            return self._generate_transpose_einsum(
                input_shape, output_shape, op_name, module_args
            )
        
        # Reshape operations - try to preserve dimension labels where possible
        return self._generate_reshape_einsum(
            input_shape, output_shape, op_name, module_args
        )
    
    def _generate_transpose_einsum(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        operation_name: str,
        module_args: Dict[str, Any] = None,
    ) -> EinsumOp:
        """Generate einsum for transpose/permute operations.
        
        For transpose/permute, input and output use the SAME dimension
        labels because they represent the same data, just reordered.
        
        Uses explicit permutation from module_args if available,
        otherwise infers from matching shapes.
        
        Example: transpose [2,4,32,16] -> [2,32,4,16]
        - Input dims: ABCD with shapes [2,4,32,16]
        - Output dims: ACBD (reordered to match [2,32,4,16])
        - Equation: ABCD->ACBD
        """
        in_dims = len(input_shape)
        out_dims = len(output_shape)
        
        # Generate labels for input dimensions
        in_label_list = generate_dim_labels(in_dims)
        
        # For transpose, output should have same number of dims
        if in_dims != out_dims:
            # Fall back to reshape behavior if dims don't match
            return self._generate_reshape_einsum(input_shape, output_shape, operation_name)
        
        # Try to use explicit permutation from module_args
        out_label_list = self._apply_permutation_from_args(
            in_label_list, module_args
        )
        
        # Fall back to shape-based inference if no explicit permutation
        if out_label_list is None:
            out_label_list = self._match_transpose_dims(
                input_shape, output_shape, in_label_list
            )
        
        # Join labels for equation string
        in_labels = "".join(in_label_list)
        out_labels = "".join(out_label_list)
        
        operands = [
            EinsumOperand("Input", in_label_list, is_output=False),
            EinsumOperand("Output", out_label_list, is_output=True),
        ]
        
        equation = f"{in_labels}->{out_labels}"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=operation_name,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )
    
    def _apply_permutation_from_args(
        self,
        in_labels: List[str],
        module_args: Dict[str, Any] = None,
    ) -> Optional[List[str]]:
        """Apply explicit permutation from module_args if available.
        
        Args:
            in_labels: Input dimension labels.
            module_args: Module arguments that may contain permutation info.
            
        Returns:
            Reordered labels if permutation found, None otherwise.
        """
        if not module_args:
            return None
        
        # Check for explicit permute_dims (full permutation order)
        if 'permute_dims' in module_args:
            perm = module_args['permute_dims']
            if len(perm) == len(in_labels):
                return [in_labels[i] for i in perm]
        
        # Check for transpose_dims (swap two dimensions)
        if 'transpose_dims' in module_args:
            dims = module_args['transpose_dims']
            if len(dims) == 2:
                dim0, dim1 = dims
                if 0 <= dim0 < len(in_labels) and 0 <= dim1 < len(in_labels):
                    out_labels = list(in_labels)
                    out_labels[dim0], out_labels[dim1] = out_labels[dim1], out_labels[dim0]
                    return out_labels
        
        # Check for dim0/dim1 (transpose two dimensions)
        if 'dim0' in module_args and 'dim1' in module_args:
            dim0 = module_args['dim0']
            dim1 = module_args['dim1']
            if 0 <= dim0 < len(in_labels) and 0 <= dim1 < len(in_labels):
                out_labels = list(in_labels)
                out_labels[dim0], out_labels[dim1] = out_labels[dim1], out_labels[dim0]
                return out_labels
        
        return None
    
    def _match_transpose_dims(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        in_labels: List[str],
    ) -> List[str]:
        """Match output dimensions to input dimensions by shape values.
        
        This infers the permutation by matching shape values.
        For ambiguous cases (same shape values), maintains relative order.
        
        Args:
            input_shape: Input tensor shape
            output_shape: Output tensor shape  
            in_labels: Labels for input dimensions
            
        Returns:
            Labels for output dimensions (reordered input labels)
        """
        # Build a mapping of shape value -> list of (index, label) pairs
        shape_to_labels: Dict[int, List[tuple]] = {}
        for i, (size, label) in enumerate(zip(input_shape, in_labels)):
            if size not in shape_to_labels:
                shape_to_labels[size] = []
            shape_to_labels[size].append((i, label))
        
        # Track which input labels have been used
        used_counts: Dict[int, int] = {size: 0 for size in shape_to_labels}
        
        # Build output labels by matching shapes
        out_labels = []
        for out_size in output_shape:
            if out_size in shape_to_labels:
                idx = used_counts[out_size]
                if idx < len(shape_to_labels[out_size]):
                    _, label = shape_to_labels[out_size][idx]
                    out_labels.append(label)
                    used_counts[out_size] += 1
                else:
                    # Fallback: reuse first matching label
                    _, label = shape_to_labels[out_size][0]
                    out_labels.append(label)
            else:
                # Shape not found - shouldn't happen for valid transpose
                out_labels.append(in_labels[len(out_labels)] if len(out_labels) < len(in_labels) else "X")
        
        return out_labels
    
    def _match_reshape_dims(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        in_labels: List[str],
    ) -> List[str]:
        """Match output dimensions to input dimensions for reshape/view operations.
        
        Preserves dimension labels where the size is unchanged at the same position
        or can be matched. New labels are generated for reshaped dimensions.
        
        Strategy:
        1. Match from the start: if input[i] == output[i], keep the label
        2. Match from the end: if input[-j] == output[-j], keep the label
        3. For unmatched dimensions, generate new labels with "R" prefix
        
        Args:
            input_shape: Input tensor shape
            output_shape: Output tensor shape
            in_labels: Labels for input dimensions
            
        Returns:
            Labels for output dimensions
        """
        in_dims = len(input_shape)
        out_dims = len(output_shape)
        
        # Initialize output labels as None (to be filled)
        out_labels: List[Optional[str]] = [None] * out_dims
        used_in_labels: set = set()
        
        # Match from the start
        start_match = 0
        for i in range(min(in_dims, out_dims)):
            if input_shape[i] == output_shape[i]:
                out_labels[i] = in_labels[i]
                used_in_labels.add(in_labels[i])
                start_match = i + 1
            else:
                break
        
        # Match from the end
        end_match = 0
        for j in range(1, min(in_dims, out_dims) + 1):
            in_idx = in_dims - j
            out_idx = out_dims - j
            
            # Don't overlap with start matches
            if out_idx < start_match or in_idx < start_match:
                break
                
            if input_shape[in_idx] == output_shape[out_idx]:
                if in_labels[in_idx] not in used_in_labels:
                    out_labels[out_idx] = in_labels[in_idx]
                    used_in_labels.add(in_labels[in_idx])
                    end_match = j
            else:
                break
        
        # Generate new labels for unmatched dimensions
        new_label_idx = 0
        for i in range(out_dims):
            if out_labels[i] is None:
                # Generate a new label that doesn't conflict
                while True:
                    new_label = f"R{new_label_idx}"
                    new_label_idx += 1
                    if new_label not in used_in_labels and new_label not in out_labels:
                        break
                out_labels[i] = new_label
        
        return out_labels  # type: ignore
    
    def _generate_reshape_einsum(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        operation_name: str,
        module_args: Dict[str, Any] = None,
    ) -> EinsumOp:
        """Generate einsum for reshape/view operations.
        
        For view/reshape operations, we try to preserve dimension labels where
        the dimension size is unchanged. This helps track data flow through reshapes.
        
        Example: view [2,32,64] -> [2,32,4,16]
        - Input: A,B,C with sizes [2,32,64]
        - Output: A,B,D,E with sizes [2,32,4,16]  (A,B preserved since sizes match)
        - Equation: ABC->ABDE
        
        For dimensions that change, we use new labels with "R" prefix (reshaped).
        """
        in_dims = len(input_shape)
        out_dims = len(output_shape)
        
        # Generate input labels
        in_label_list = generate_dim_labels(in_dims)
        
        # Try to match output dimensions to input dimensions
        out_label_list = self._match_reshape_dims(
            input_shape, output_shape, in_label_list
        )
        
        # Join labels for equation string
        in_labels = "".join(in_label_list)
        out_labels = "".join(out_label_list)
        
        operands = [
            EinsumOperand("Input", in_label_list, is_output=False),
            EinsumOperand("Output", out_label_list, is_output=True),
        ]
        
        equation = f"{in_labels}->{out_labels}"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=operation_name,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


class MatrixStructureHandler(EinsumOpHandler):
    """Handler for matrix structure operations."""
    
    supported_ops = ["diag", "diagonal", "tril", "triu"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for matrix structure operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_matrix_structure_einsum(input_shape, op_name)
    
    def _generate_matrix_structure_einsum(
        self,
        input_shape: TensorShape,
        operation: str
    ) -> EinsumOp:
        """Generate einsum for matrix structure ops.
        
        Note: diag operation creates a diagonal matrix where conceptually
        the same dimension appears twice. However, to comply with our
        "no repeated ranks" convention, we use distinct dimension names
        (A and B) for the output, even though they have the same size.
        """
        dims = len(input_shape)
        labels = string.ascii_uppercase[:dims]
        
        if operation == "diag":
            if dims == 1:
                # 1D -> 2D diagonal matrix: A -> AB (not AA)
                # The diagonal constraint is implicit, not expressed in einsum
                output_labels = "AB"
            else:
                # 2D+ -> 1D diagonal extraction
                output_labels = labels[0]
        elif operation == "transpose" and dims >= 2:
            # Swap last two dims
            output_labels = labels[:-2] + labels[-1] + labels[-2]
        else:
            # tril, triu preserve shape
            output_labels = labels
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(output_labels), is_output=True),
        ]
        
        equation = f"{labels}->{output_labels}"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=operation,
            is_real_einsum=False,  # diag is not a real einsum (has implicit constraint)
            elementwise_op="copy",
            reduction_op="none",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(TensorManipulationHandler)
_registry.register_handler(MatrixStructureHandler)


__all__ = ["TensorManipulationHandler", "MatrixStructureHandler"]
