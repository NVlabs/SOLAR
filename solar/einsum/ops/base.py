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

"""Base classes for einsum operation handlers.

This module defines the core data structures and abstract base class
for all einsum operation handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import re

from solar.common.types import TensorShape, TensorShapes
from solar.common.utils import validate_einsum_ranks_match_shapes

logger = logging.getLogger(__name__)


@dataclass
class EinsumOperand:
    """Represents an operand in an einsum operation."""
    name: str
    dims: List[str]
    is_output: bool = False
    stride: Optional[Dict[str, int]] = None
    dilation: Optional[Dict[str, int]] = None
    
    def to_timeloop_dataspace(self) -> Dict[str, Any]:
        """Convert to timeloop dataspace format."""
        dataspace = {
            'name': self.name,
            'projection': self.dims,
        }
        if self.is_output:
            dataspace['read_write'] = 'true'
        return dataspace


@dataclass
class EinsumOp:
    """Represents an einsum operation.
    
    The extended einsum representation supports different elementwise and reduction
    operations beyond the standard multiply-add semantics:
    
    - elementwise_op: The operation applied element-wise (default: 'mul')
      Examples: 'mul' for matmul, 'add' for element-wise add, 'max' for max pooling
    - reduction_op: The operation used to reduce/aggregate (default: 'add')
      Examples: 'add' for sum, 'max' for max reduction, 'none' for no reduction
    - is_real_einsum: True if this is a standard tensor contraction (mul+add)
    - is_einsum_supportable: True if the operation can be expressed with extended einsum
    
    Standard einsum (matmul): elementwise_op='mul', reduction_op='add'
    Element-wise add:        elementwise_op='add', reduction_op='none'
    Max pooling:             elementwise_op='copy', reduction_op='max'
    """
    operands: List[EinsumOperand]
    equation: str
    name: str
    is_real_einsum: bool = True
    elementwise_op: str = "mul"  # 'mul', 'add', 'sub', 'div', 'max', 'min', 'copy'
    reduction_op: str = "add"    # 'add', 'max', 'min', 'mul', 'none'
    is_einsum_supportable: bool = True  # Can this op be expressed with extended einsum?

    @property
    def input_operands(self) -> List[EinsumOperand]:
        """Get input operands."""
        return [op for op in self.operands if not op.is_output]
    
    @property
    def output_operands(self) -> List[EinsumOperand]:
        """Get output operands."""
        return [op for op in self.operands if op.is_output]
    
    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        """Calculate compute cost from einsum rank dimensions.
        
        Collects unique rank dimension sizes from ALL operands (input + output).
        Compound dims like 'P+R' are split into atoms and resolved from other
        operands. No op-specific special cases — purely driven by einsum equation.
        
        Total cost = product of all unique resolved rank dimension sizes.
        """
        return compute_cost_from_equation(self.equation, tensor_shapes)
    
    def to_torch_einsum(self, tensor_names: Optional[List[str]] = None) -> str:
        """Convert to torch.einsum format."""
        input_operands = self.input_operands
        
        if tensor_names is None:
            tensor_names = [op.name for op in input_operands]
        elif len(tensor_names) != len(input_operands):
            raise ValueError(
                f"Number of tensor names ({len(tensor_names)}) must match "
                f"number of input operands ({len(input_operands)})"
            )
        
        equation_str = f"'{self.equation}'"
        tensor_args = ', '.join(tensor_names)
        return f"torch.einsum({equation_str}, {tensor_args})"


def _parse_dim_atoms(dim: str) -> List[str]:
    """Parse a possibly compound dim into atomic rank names.
    
    'P+R' -> ['P', 'R']
    'B'   -> ['B']
    'P+R0' -> ['P', 'R0']
    """
    return [d.strip() for d in re.split(r'[+\-]', dim) if d.strip()]


def _parse_equation_operand(operand: str) -> List[str]:
    """Parse one einsum operand into rank tokens.

    Supports single-letter ranks with optional digits and parenthesized
    compound ranks, e.g. ``BGI(P+R)`` -> ``["B", "G", "I", "P+R"]``.
    """
    tokens: List[str] = []
    i = 0
    while i < len(operand):
        if operand[i] == "(":
            j = operand.index(")", i)
            tokens.append(operand[i + 1 : j])
            i = j + 1
        elif operand[i].isalpha():
            token = operand[i]
            i += 1
            while i < len(operand) and operand[i].isdigit():
                token += operand[i]
                i += 1
            tokens.append(token)
        else:
            i += 1
    return tokens


def compute_cost_from_equation(equation: str, tensor_shapes: TensorShapes) -> int:
    """Calculate compute cost from an einsum equation and tensor shapes.

    The cost model is the product of every unique rank used by the equation.
    Compound ranks like ``P+R`` are split into atoms; kernel atoms such as
    ``R`` are resolved from concrete input operands, and output-position atoms
    such as ``P`` are resolved from the output shapes.
    """
    if not equation or "->" not in equation:
        return 0

    lhs, rhs = equation.split("->", 1)
    input_tokens = [_parse_equation_operand(operand) for operand in lhs.split(",")]
    output_tokens = _parse_equation_operand(rhs)

    all_ranks: Dict[str, Optional[int]] = {}
    for tokens in input_tokens:
        for token in tokens:
            for atom in _parse_dim_atoms(token):
                all_ranks.setdefault(atom, None)
    for token in output_tokens:
        for atom in _parse_dim_atoms(token):
            all_ranks.setdefault(atom, None)

    def _resolve(tokens_by_operand: List[List[str]], shapes: List[TensorShape]) -> None:
        for idx, tokens in enumerate(tokens_by_operand):
            if idx >= len(shapes):
                break
            shape = shapes[idx]
            for dim_offset, token in enumerate(tokens):
                atoms = _parse_dim_atoms(token)
                if len(atoms) == 1 and dim_offset < len(shape):
                    atom = atoms[0]
                    if all_ranks.get(atom) is None:
                        all_ranks[atom] = int(shape[dim_offset])

    _resolve(input_tokens, tensor_shapes.inputs)
    _resolve([output_tokens], tensor_shapes.outputs)

    total_ops = 1
    for value in all_ranks.values():
        if value is not None and value > 0:
            total_ops *= value
    return int(total_ops)


class EinsumOpHandler(ABC):
    """Abstract base class for einsum operation handlers.
    
    Each handler is responsible for converting one or more related operation
    types to einsum notation. Handlers receive TensorShapes (positional)
    and should access inputs/outputs by index, not by name.
    """
    
    supported_ops: List[str] = []
    
    def __init__(self, debug: bool = False):
        """Initialize the handler.
        
        Args:
            debug: Enable debug output.
        """
        self.debug = debug
    
    @abstractmethod
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate an einsum operation for the given operation.
        
        Args:
            op_name: Normalized operation name.
            tensor_shapes: Positional input/output shapes.
            **kwargs: Additional operation-specific parameters.
            
        Returns:
            EinsumOp representing the operation.
        """
        pass
    
    def can_handle(self, op_name: str) -> bool:
        """Check if this handler can process the given operation.
        
        Args:
            op_name: Normalized operation name.
            
        Returns:
            True if this handler supports the operation.
        """
        return op_name.lower() in [op.lower() for op in self.supported_ops]
    
    def _validate_einsum(
        self, 
        einsum_op: "EinsumOp", 
        tensor_shapes: Dict[str, List[List[int]]]
    ) -> "EinsumOp":
        """Validate that einsum ranks match tensor shapes.
        
        If validation fails, logs a warning and attempts to fix the equation
        by regenerating it based on actual shapes.
        
        Args:
            einsum_op: The generated EinsumOp to validate.
            tensor_shapes: Dictionary with "inputs" and "outputs" keys containing shape lists.
                          Format: {"inputs": [[shape1], [shape2]], "outputs": [[output_shape]]}
            
        Returns:
            The validated (and possibly corrected) EinsumOp.
        """
        is_valid, error_msg = validate_einsum_ranks_match_shapes(
            einsum_op.equation, tensor_shapes
        )
        
        if not is_valid:
            logger.warning(
                f"Einsum rank mismatch for {einsum_op.name}: {error_msg}. "
                f"Equation: {einsum_op.equation}, tensor_shapes: {tensor_shapes}"
            )
            # Try to fix by regenerating equation from shapes
            corrected_op = self._try_fix_einsum_ranks(einsum_op, tensor_shapes)
            if corrected_op is not None:
                return corrected_op
        
        return einsum_op
    
    def _try_fix_einsum_ranks(
        self, 
        einsum_op: "EinsumOp", 
        tensor_shapes: Dict[str, List[List[int]]]
    ) -> Optional["EinsumOp"]:
        """Attempt to fix einsum equation to match actual tensor shapes.
        
        This is a best-effort fix that regenerates the equation based on
        actual tensor ranks.
        
        Args:
            einsum_op: The EinsumOp with mismatched ranks.
            tensor_shapes: Dictionary with "inputs" and "outputs" keys containing shape lists.
            
        Returns:
            Corrected EinsumOp if fix was possible, None otherwise.
        """
        import string
        
        # Get actual shapes from tensor_shapes
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])
        
        if not input_shapes or not output_shapes:
            return None
        
        input_shape = input_shapes[0] if input_shapes else None
        input_1_shape = input_shapes[1] if len(input_shapes) > 1 else None
        output_shape = output_shapes[0] if output_shapes else None
        
        if input_shape is None or output_shape is None:
            return None
        
        input_rank = len(input_shape)
        output_rank = len(output_shape)
        
        # Generate labels based on actual ranks
        input_labels = string.ascii_uppercase[:input_rank]
        output_labels = string.ascii_uppercase[:output_rank]
        
        # For binary ops, handle second input
        if input_1_shape is not None:
            input_1_rank = len(input_1_shape)
            
            # Handle broadcasting: use output labels for the larger tensor
            if input_1_rank < input_rank:
                # Second input is smaller, use suffix of output labels (broadcast from right)
                input_1_labels = output_labels[-input_1_rank:] if input_1_rank > 0 else ""
            elif input_1_rank > input_rank:
                # First input is smaller, use suffix of output labels
                input_labels = output_labels[-input_rank:] if input_rank > 0 else ""
                input_1_labels = output_labels
            else:
                input_1_labels = input_labels
            
            new_equation = f"{input_labels},{input_1_labels}->{output_labels}"
            
            # Update operands
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Input_1", list(input_1_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]
        else:
            new_equation = f"{input_labels}->{output_labels}"
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]
        
        logger.info(f"Fixed einsum equation: {einsum_op.equation} -> {new_equation}")
        
        return EinsumOp(
            operands=new_operands,
            equation=new_equation,
            name=einsum_op.name,
            is_real_einsum=einsum_op.is_real_einsum,
            elementwise_op=einsum_op.elementwise_op,
            reduction_op=einsum_op.reduction_op,
            is_einsum_supportable=einsum_op.is_einsum_supportable,
        )

@dataclass
class AFOperand:
    """One tensor access in AccelForge (AF) einsum YAML.

    - dims_lowercase: Labels in the current (renamed) einsum equation. The projection
      targets AccelForge uses for this access.
    - dims_uppercase: Optional parallel list of labels from the input graph node's
      tensor notation. If None, don't do renaming and use the dims_lowercase list as is.
    """
    name: str
    dims_lowercase: List[str]
    dims_uppercase: Optional[List[str]] = None
    is_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to AF yaml."""
        operand = dict()
        operand['name'] = self.name
        operand['projection'] = (
            [d.lower() for d in self.dims_lowercase]
            if self.dims_uppercase is None
            else {d.upper(): pr.lower() for d, pr in zip(self.dims_uppercase, self.dims_lowercase)}
        )
        if self.is_output:
            operand['output'] = True
        return operand

@dataclass
class AFOp:
    """Represents an operation in AccelForge (AF) einsum format.

    - tensor_accesses: List of tensor accesses.
    - is_copy_operation: Used to copy input tensors into memory.
    """
    name: str
    tensor_accesses: List[AFOperand]
    is_copy_operation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to AF yaml."""
        op_dict = dict()
        op_dict['name'] = self.name
        if self.is_copy_operation:
            op_dict['is_copy_operation'] = True
        op_dict['tensor_accesses'] = [operand.to_dict() for operand in self.tensor_accesses]
        return op_dict


__all__ = [
    "EinsumOperand",
    "EinsumOp",
    "EinsumOpHandler",
    "compute_cost_from_equation",
]
