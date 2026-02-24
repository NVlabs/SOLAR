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

"""Handlers for cumulative/scan operations.

This module provides einsum handlers for:
- cumsum: cumulative sum along a dimension
- cumprod: cumulative product along a dimension
- cummax: cumulative maximum along a dimension
- cummin: cumulative minimum along a dimension

These operations are NOT reductions - they preserve the input shape.
Based on PyTorch documentation:
https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html

For example, cumsum on input of size N returns output of size N:
    y[i] = x[0] + x[1] + ... + x[i]
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


class CumulativeHandler(EinsumOpHandler):
    """Handler for cumulative/scan operations.
    
    Based on PyTorch documentation:
    https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html
    
    Key property: input shape == output shape
    
    These are sequential scan operations that cannot be efficiently
    expressed as standard einsum but we represent them for analysis.
    """
    
    supported_ops = [
        # Cumulative operations - preserve input shape
        "cumsum",      # Cumulative sum
        "cumprod",     # Cumulative product
        "cummax",      # Cumulative maximum
        "cummin",      # Cumulative minimum
        # Scan operations
        "scan",
        "prefix_sum",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for cumulative operation.
        
        Since cumulative operations preserve input shape, the einsum
        equation has identical input and output dimensions.
        """
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Get the dimension along which cumulative op is performed
        dim = kwargs.get("dim")
        if dim is None:
            # Try to extract from raw_attributes
            raw_attrs = kwargs.get("raw_attributes", "")
            if raw_attrs:
                import re
                match = re.search(r"dim:\s*(-?\d+)", raw_attrs)
                if match:
                    dim = int(match.group(1))
        
        # Normalize op name
        op_type = op_name.lower()
        
        return self._generate_cumulative_einsum(input_shape, op_type, dim)
    
    def _generate_cumulative_einsum(
        self,
        shape: TensorShape,
        op_type: str = "cumsum",
        dim: Optional[int] = None
    ) -> EinsumOp:
        """Generate einsum for cumulative operations.
        
        Args:
            shape: Input tensor shape.
            op_type: Type of cumulative op (cumsum, cumprod, cummax, cummin).
            dim: Dimension along which to perform the operation.
            
        Returns:
            EinsumOp for the cumulative operation.
            
        The key difference from reduction operations:
        - Reductions: ABC->AB (remove dimension C)
        - Cumulative: ABC->ABC (preserve all dimensions)
        
        The einsum representation shows that output has same rank and
        dimensions as input.
        """
        ndims = len(shape)
        labels = list(string.ascii_uppercase[:ndims])
        
        # Input and output have identical labels (shape preserved)
        input_labels = labels.copy()
        output_labels = labels.copy()
        
        operands = [
            EinsumOperand("Input", input_labels, is_output=False),
            EinsumOperand("Output", output_labels, is_output=True),
        ]
        
        equation = f"{''.join(input_labels)}->{''.join(output_labels)}"
        
        # Map cumulative op_type to appropriate reduction_op
        # Note: the reduction_op indicates what operation is accumulated,
        # not that dimensions are reduced
        reduction_op_map = {
            "cumsum": "add",      # Accumulate sums
            "cumprod": "mul",     # Accumulate products
            "cummax": "max",      # Accumulate max
            "cummin": "min",      # Accumulate min
            "scan": "add",        # Generic scan (assume sum)
            "prefix_sum": "add",  # Prefix sum
        }
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            # Not a real einsum - sequential dependency
            is_real_einsum=False,
            # Copy operation (each output element copies accumulated value)
            elementwise_op="copy",
            # The type of accumulation
            reduction_op=reduction_op_map.get(op_type, "add"),
            # Can be represented but not efficiently parallelizable
            is_einsum_supportable=True,
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(CumulativeHandler)


__all__ = ["CumulativeHandler"]
