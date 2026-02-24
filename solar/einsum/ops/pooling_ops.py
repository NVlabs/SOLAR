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

"""Handlers for pooling operations.

This module provides einsum handlers for:
- max_pool1d, max_pool2d, max_pool3d
- avg_pool1d, avg_pool2d, avg_pool3d
- adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
- adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d
"""

import string
from typing import Any, List, Optional, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class PoolingHandler(EinsumOpHandler):
    """Handler for pooling operations."""
    
    supported_ops = [
        "max_pool1d", "max_pool2d", "max_pool3d",
        "avg_pool1d", "avg_pool2d", "avg_pool3d",
        "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
        "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for pooling operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        kernel_size = kwargs.get("kernel_size", (2, 2))
        stride = kwargs.get("stride")
        
        return self._generate_pooling_einsum(
            input_shape, op_name, kernel_size, stride
        )
    
    def _generate_pooling_einsum(
        self,
        input_shape: TensorShape,
        pool_type: str,
        kernel_size: Tuple[int, ...] = (2, 2),
        stride: Optional[Tuple[int, ...]] = None
    ) -> EinsumOp:
        """Generate einsum for pooling operations.
        
        Pooling reduces spatial dimensions via local reduction.
        """
        dims = len(input_shape)
        labels = string.ascii_uppercase[:dims]
        
        # Pooling preserves batch and channel dims, reduces spatial
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),  # Simplified
        ]
        
        equation = f"{labels}->{labels}"
        
        # Determine reduction op based on pool type
        if "max" in pool_type.lower():
            reduction_op = "max"
        else:  # avg_pool
            reduction_op = "add"  # avg = sum / count
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=pool_type,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op=reduction_op,
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(PoolingHandler)


__all__ = ["PoolingHandler"]

