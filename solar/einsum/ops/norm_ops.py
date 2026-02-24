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

"""Handlers for normalization operations.

This module provides einsum handlers for:
- batch_norm (1d, 2d, 3d)
- layer_norm
- group_norm
- instance_norm
- normalize
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


class NormalizationHandler(EinsumOpHandler):
    """Handler for normalization operations."""
    
    supported_ops = [
        "batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d",
        "layer_norm", "layernorm",
        "group_norm", "groupnorm",
        "instance_norm", "instancenorm",
        "normalize",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for normalization operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_normalization_einsum(input_shape, op_name)
    
    def _generate_normalization_einsum(
        self,
        input_shape: TensorShape,
        norm_type: str
    ) -> EinsumOp:
        """Generate einsum for normalization.
        
        Normalization is approximated as elementwise for einsum analysis
        since the main computation is the affine transformation γx + β.
        The elementwise_op is set to the norm type to indicate the actual operation.
        """
        dims = len(input_shape)
        labels = string.ascii_uppercase[:dims]
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]
        
        equation = f"{labels}->{labels}"
        
        # Normalize op name (remove trailing underscore for inplace ops)
        normalized_norm = norm_type.rstrip("_") 
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=norm_type,
            is_real_einsum=False,
            elementwise_op=normalized_norm,  # e.g., "batchnorm", "batchnorm2d", "layernorm"
            reduction_op="none",
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(NormalizationHandler)


__all__ = ["NormalizationHandler"]

