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

"""Handlers for attention operations.

This module provides einsum handlers for:
- scaled_dot_product_attention (SDPA)
- flex_attention
- multi_head_attention_forward

Based on PyTorch's scaled_dot_product_attention:
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

The SDPA operation is expanded into a subgraph of simpler operations:
1. Q @ K^T (matmul) -> attention scores
2. scores * scale (elementwise mul)
3. softmax(scores) -> attention weights
4. weights @ V (matmul) -> output
"""

from typing import Any, Dict, List, Optional, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class ScaledDotProductAttentionHandler(EinsumOpHandler):
    """Handler for scaled dot-product attention.
    
    This handler can either:
    1. Generate a single fused einsum equation (BHQD,BHKD,BHKV->BHQV)
    2. Expand into a subgraph of operations for detailed analysis
    
    The expansion follows PyTorch's reference implementation:
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value
    """
    
    supported_ops = [
        "scaled_dot_product_attention", "sdpa", "attention",
    ]
    
    # Mark this as an expandable operation
    is_expandable = True
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for scaled dot-product attention.
        
        Gets shapes from input_shapes (Q, K, V) and output_shapes.
        No weights are involved in SDPA.
        """
        # Get shapes from the new format (input_shapes list)
        input_shapes = shapes.get("input_shapes", [])
        output_shapes = shapes.get("output_shapes", [])
        
        # Also try legacy format
        if not input_shapes:
            inputs = shapes.get("inputs", [])
            input_shapes = inputs if inputs else []
        
        # Try to get from Input, Input_1, Input_2 keys (legacy)
        if not input_shapes:
            query = shapes.get("Input")
            key = shapes.get("Input_1")
            value = shapes.get("Input_2")
            if query and key and value:
                input_shapes = [query, key, value]
        
        if len(input_shapes) < 3:
            raise ValueError(
                f"SDPA requires 3 input shapes (Q, K, V). Got {len(input_shapes)}: {shapes}"
            )
        
        query_shape = input_shapes[0]
        key_shape = input_shapes[1]
        value_shape = input_shapes[2]
        
        is_causal = kwargs.get("is_causal", False)
        
        return self._generate_sdpa_einsum(query_shape, key_shape, value_shape, is_causal)
    
    def _generate_sdpa_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape,
        is_causal: bool = False
    ) -> EinsumOp:
        """Generate fused einsum for scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Shape notation (following PyTorch docs):
            Query: (N, ..., Hq, L, E)  -> simplified to (B, H, Q, D)
            Key:   (N, ..., H, S, E)   -> simplified to (B, H, K, D)
            Value: (N, ..., H, S, Ev)  -> simplified to (B, H, K, V)
            Output:(N, ..., Hq, L, Ev) -> simplified to (B, H, Q, V)
        
        Where:
            B = batch (N and any extra batch dims)
            H = number of heads
            Q = query sequence length (L)
            K = key/value sequence length (S)
            D = embedding dimension (E)
            V = value embedding dimension (Ev)
        """
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]
        
        # Fused equation: combines Q@K^T and result@V
        equation = "BHQD,BHKD,BHKV->BHQV"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="scaled_dot_product_attention",
            elementwise_op="mul",
            reduction_op="add",
        )
    
    def create_subgraph(
        self,
        node_id: str,
        node_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Expand SDPA into a subgraph of simpler operations.
        
        Based on PyTorch's reference implementation:
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight = torch.softmax(attn_weight, dim=-1)
            return attn_weight @ value
        
        Subgraph structure:
            1. qk_matmul: Q @ K^T -> attention scores [B,H,Q,K]
            2. scale: scores * (1/sqrt(d_k)) -> scaled scores [B,H,Q,K]
            3. softmax: softmax(scaled_scores, dim=-1) -> attention weights [B,H,Q,K]
            4. av_matmul: weights @ V -> output [B,H,Q,V]
        
        Args:
            node_id: Original node identifier.
            node_data: Node data containing shapes and arguments.
            
        Returns:
            Dictionary of subgraph nodes.
        """
        input_shapes = node_data.get("input_shapes", [])
        output_shapes = node_data.get("output_shapes", [])
        module_args = node_data.get("module_args", {})
        
        if len(input_shapes) < 3:
            raise ValueError(f"SDPA requires 3 inputs (Q, K, V). Got: {input_shapes}")
        
        query_shape = list(input_shapes[0])  # [B, H, Q, D]
        key_shape = list(input_shapes[1])    # [B, H, K, D]
        value_shape = list(input_shapes[2])  # [B, H, K, V]
        output_shape = list(output_shapes[0]) if output_shapes else None
        
        # Infer dimensions
        B = query_shape[0]  # batch
        H = query_shape[1]  # heads
        Q = query_shape[2]  # query sequence length
        D = query_shape[3]  # embedding dim
        K = key_shape[2]    # key sequence length
        V = value_shape[3]  # value embedding dim
        
        # Intermediate shapes
        scores_shape = [B, H, Q, K]  # Q @ K^T
        
        subgraph: Dict[str, Dict[str, Any]] = {}
        
        # 1. Q @ K^T -> attention scores
        # Einsum: BHQD,BHKD->BHQK (K is transposed, so D is contracted)
        qk_node_id = f"{node_id}.qk_matmul"
        subgraph[qk_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQD,BHKD->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "input_shapes": [query_shape, key_shape],
            "output_shapes": [scores_shape],
            "weight_shapes": [],
            "weight_nodes": [],
            "module_args": {"operation": "Q @ K^T"},
            "connections": {
                "inputs": [],  # Will be connected by caller
                "outputs": [f"{node_id}.scale"],
            },
        }
        
        # 2. Scale by 1/sqrt(d_k)
        # This is an elementwise multiplication
        scale_node_id = f"{node_id}.scale"
        subgraph[scale_node_id] = {
            "type": "mul",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "input_shapes": [scores_shape],
            "output_shapes": [scores_shape],
            "weight_shapes": [],
            "weight_nodes": [],
            "module_args": {"scale_factor": f"1/sqrt({D})"},
            "connections": {
                "inputs": [qk_node_id],
                "outputs": [f"{node_id}.softmax"],
            },
        }
        
        # 3. Softmax over K dimension (dim=-1)
        softmax_node_id = f"{node_id}.softmax"
        subgraph[softmax_node_id] = {
            "type": "softmax",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "softmax",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "input_shapes": [scores_shape],
            "output_shapes": [scores_shape],
            "weight_shapes": [],
            "weight_nodes": [],
            "module_args": {"dim": -1},
            "connections": {
                "inputs": [scale_node_id],
                "outputs": [f"{node_id}.av_matmul"],
            },
        }
        
        # 4. Attention weights @ V -> output
        # Einsum: BHQK,BHKV->BHQV
        av_node_id = f"{node_id}.av_matmul"
        final_output_shape = output_shape if output_shape else [B, H, Q, V]
        subgraph[av_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQK,BHKV->BHQV",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "input_shapes": [scores_shape, value_shape],
            "output_shapes": [final_output_shape],
            "weight_shapes": [],
            "weight_nodes": [],
            "module_args": {"operation": "attn_weights @ V"},
            "connections": {
                "inputs": [softmax_node_id],  # Also needs value input
                "outputs": [],  # Will be connected by caller
            },
        }
        
        return subgraph
    
    def get_subgraph_input_mapping(
        self,
        node_id: str,
    ) -> Dict[str, List[int]]:
        """Get mapping of subgraph nodes to original input indices.
        
        For SDPA:
            - qk_matmul needs inputs 0 (Q) and 1 (K)
            - av_matmul needs input 2 (V) in addition to softmax output
        
        Returns:
            Dict mapping subgraph node IDs to list of original input indices.
        """
        return {
            f"{node_id}.qk_matmul": [0, 1],  # Q, K
            f"{node_id}.av_matmul": [2],      # V (softmax output is internal)
        }


class FlexAttentionHandler(EinsumOpHandler):
    """Handler for flex_attention (similar to SDPA)."""
    
    supported_ops = ["flex_attention"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        inputs = shapes.get("inputs", [])
        query = inputs[0] if inputs else None
        key = inputs[1] if len(inputs) > 1 else None
        value = inputs[2] if len(inputs) > 2 else None
        
        if not query or not key or not value:
            raise ValueError(f"Missing Q/K/V shapes for attention. Got: {shapes}")
        
        return self._generate_flex_attention_einsum(query, key, value)
    
    def _generate_flex_attention_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]
        
        equation = "BHQD,BHKD,BHKV->BHQV"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="flex_attention",
            elementwise_op="mul",
            reduction_op="add",
        )


class MultiHeadAttentionHandler(EinsumOpHandler):
    """Handler for multi-head attention forward."""
    
    supported_ops = ["multi_head_attention_forward", "multihead_attention"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for multi-head attention."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_mha_einsum(input_shape)
    
    def _generate_mha_einsum(self, input_shape: TensorShape) -> EinsumOp:
        """Generate einsum for multi-head attention.
        
        MHA combines Q, K, V projections with attention computation.
        We represent it as a single attention-like operation.
        """
        operands = [
            EinsumOperand("Input", ["B", "S", "D"], is_output=False),
            EinsumOperand("Output", ["B", "S", "D"], is_output=True),
        ]
        
        equation = "BSD->BSD"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="multi_head_attention_forward",
            is_real_einsum=False,  # Composite operation
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(ScaledDotProductAttentionHandler)
_registry.register_handler(FlexAttentionHandler)
_registry.register_handler(MultiHeadAttentionHandler)


__all__ = [
    "ScaledDotProductAttentionHandler",
    "FlexAttentionHandler",
    "MultiHeadAttentionHandler",
]

