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

"""Tests for extended einsum definitions and multi-node expansion.

This module tests:
1. Extended einsum operations with elementwise_op and reduction_op
2. Binary elementwise operations (add, sub, mul, div)
3. Operations that expand to multiple einsum nodes (softmax, attention, etc.)
"""

import pytest
from solar.einsum import EinsumAnalyzer
from solar.einsum.ops import EinsumOp, EinsumOperand


class TestExtendedEinsumDefinition:
    """Tests for elementwise_op and reduction_op in EinsumOp."""
    
    def test_matmul_ops(self):
        """Test matmul has mul/add semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64], "Weight": [64, 128]}
        einsum_op = analyzer.get_einsum_op("matmul", shapes)
        
        assert einsum_op.elementwise_op == "mul"
        assert einsum_op.reduction_op == "add"
        assert einsum_op.is_real_einsum is True
    
    def test_linear_ops(self):
        """Test linear has mul/add semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [8, 16, 64], "Weight": [128, 64]}
        
        einsum_op = analyzer.get_einsum_op("linear", shapes)
        
        assert einsum_op.elementwise_op == "mul"
        assert einsum_op.reduction_op == "add"
        assert einsum_op.is_real_einsum is True
    
    def test_conv2d_ops(self):
        """Test conv2d has mul/add semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [1, 3, 224, 224], "Weight": [64, 3, 7, 7]}
        einsum_op = analyzer.get_einsum_op("conv2d", shapes)
        
        assert einsum_op.elementwise_op == "mul"
        assert einsum_op.reduction_op == "add"
        assert einsum_op.is_real_einsum is True
    
    def test_add_ops(self):
        """Test add has add/none semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64], "Input_1": [32, 64]}
        einsum_op = analyzer.get_einsum_op("add", shapes)
        
        assert einsum_op.elementwise_op == "add"
        assert einsum_op.reduction_op == "none"
        assert einsum_op.is_real_einsum is False
    
    def test_sub_ops(self):
        """Test sub has sub/none semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64], "Input_1": [32, 64]}
        einsum_op = analyzer.get_einsum_op("sub", shapes)
        
        assert einsum_op.elementwise_op == "sub"
        assert einsum_op.reduction_op == "none"
        assert einsum_op.is_real_einsum is False
    
    def test_mul_ops(self):
        """Test mul has mul/none semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64], "Input_1": [32, 64]}
        einsum_op = analyzer.get_einsum_op("mul", shapes)
        
        assert einsum_op.elementwise_op == "mul"
        assert einsum_op.reduction_op == "none"
        assert einsum_op.is_real_einsum is False
    
    def test_div_ops(self):
        """Test div has div/none semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64], "Input_1": [32, 64]}
        einsum_op = analyzer.get_einsum_op("div", shapes)
        
        assert einsum_op.elementwise_op == "div"
        assert einsum_op.reduction_op == "none"
        assert einsum_op.is_real_einsum is False
    
    def test_sum_reduction_ops(self):
        """Test sum has copy/add semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64, 128]}
        einsum_op = analyzer.get_einsum_op("sum", shapes, dims=[1])
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "add"
        assert einsum_op.is_real_einsum is False
    
    def test_mean_reduction_ops(self):
        """Test mean has copy/add semantics (mean = sum / count)."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64, 128]}
        einsum_op = analyzer.get_einsum_op("mean", shapes, dims=[1])
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "add"
        assert einsum_op.is_real_einsum is False
    
    def test_prod_reduction_ops(self):
        """Test prod has copy/mul semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64, 128]}
        einsum_op = analyzer.get_einsum_op("prod", shapes, dims=[1])
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "mul"
        assert einsum_op.is_real_einsum is False
    
    def test_max_reduction_ops(self):
        """Test max has copy/max semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64, 128]}
        einsum_op = analyzer.get_einsum_op("max", shapes, dims=[1])
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "max"
        assert einsum_op.is_real_einsum is False
    
    def test_min_reduction_ops(self):
        """Test min has copy/min semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64, 128]}
        einsum_op = analyzer.get_einsum_op("min", shapes, dims=[1])
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "min"
        assert einsum_op.is_real_einsum is False
    
    def test_relu_ops(self):
        """Test relu has copy/none semantics."""
        analyzer = EinsumAnalyzer()
        
        shapes = {"Input": [32, 64]}
        einsum_op = analyzer.get_einsum_op("relu", shapes)
        
        assert einsum_op.elementwise_op == "copy"
        assert einsum_op.reduction_op == "none"
        assert einsum_op.is_real_einsum is False


class TestBinaryElementwiseEinsum:
    """Tests for binary elementwise einsum generation."""
    
    def test_add_same_shape(self):
        """Test add with same shape tensors."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.generate_binary_elementwise_einsum(
            [32, 64], [32, 64], "add"
        )
        
        assert einsum_op.equation == "AB,AB->AB"
        assert einsum_op.name == "add"
        assert len(einsum_op.input_operands) == 2
        assert len(einsum_op.output_operands) == 1
    
    def test_add_broadcasting(self):
        """Test add with broadcasting."""
        analyzer = EinsumAnalyzer()
        
        # [32, 64] + [64] -> broadcasts to [32, 64]
        einsum_op = analyzer.generate_binary_elementwise_einsum(
            [32, 64], [1, 64], "add"
        )
        
        assert "AB,AB->AB" == einsum_op.equation
        assert einsum_op.elementwise_op == "add"
    
    def test_mul_same_shape(self):
        """Test mul with same shape tensors."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.generate_binary_elementwise_einsum(
            [8, 16, 32], [8, 16, 32], "mul"
        )
        
        assert einsum_op.equation == "ABC,ABC->ABC"
        assert einsum_op.elementwise_op == "mul"
        assert einsum_op.reduction_op == "none"
    
    def test_sub_same_shape(self):
        """Test sub with same shape tensors."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.generate_binary_elementwise_einsum(
            [16, 32], [16, 32], "sub"
        )
        
        assert einsum_op.equation == "AB,AB->AB"
        assert einsum_op.elementwise_op == "sub"
    
    def test_div_same_shape(self):
        """Test div with same shape tensors."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.generate_binary_elementwise_einsum(
            [64, 128], [64, 128], "div"
        )
        
        assert einsum_op.equation == "AB,AB->AB"
        assert einsum_op.elementwise_op == "div"


class TestMultiNodeExpansion:
    """Tests for operations that expand to multiple einsum nodes.
    
    These tests verify the concept of multi-node expansion where a single
    PyTorch operation (like softmax, attention, normalization) gets
    decomposed into multiple simpler einsum operations.
    """
    
    def test_softmax_expansion_concept(self):
        """Test that softmax can be decomposed into multiple ops.
        
        Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        Decomposition:
        1. max_x = max(x, dim=-1)           # reduction (max)
        2. x_shifted = x - max_x            # binary elementwise (sub)
        3. exp_x = exp(x_shifted)           # unary elementwise (exp)
        4. sum_exp = sum(exp_x, dim=-1)     # reduction (add)
        5. output = exp_x / sum_exp         # binary elementwise (div)
        """
        analyzer = EinsumAnalyzer()
        
        # Step 1: max reduction
        max_op = analyzer.get_einsum_op("max", {"Input": [32, 64]}, dims=[1])
        assert max_op.reduction_op == "max"
        
        # Step 2: subtraction
        sub_op = analyzer.get_einsum_op("sub", {"Input": [32, 64], "Input_1": [32, 1]})
        assert sub_op.elementwise_op == "sub"
        
        # Step 3: exp (unary)
        exp_shapes = {"Input": [32, 64]}
        exp_op = analyzer.generate_elementwise_einsum([32, 64], "exp")
        assert exp_op.elementwise_op == "copy"
        assert exp_op.reduction_op == "none"
        
        # Step 4: sum reduction
        sum_op = analyzer.get_einsum_op("sum", {"Input": [32, 64]}, dims=[1])
        assert sum_op.reduction_op == "add"
        
        # Step 5: division
        div_op = analyzer.get_einsum_op("div", {"Input": [32, 64], "Input_1": [32, 1]})
        assert div_op.elementwise_op == "div"
    
    def test_layer_norm_expansion_concept(self):
        """Test that layer normalization can be decomposed.
        
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        
        Decomposition:
        1. mean_x = mean(x, dim=-1)         # reduction (add)
        2. x_centered = x - mean_x          # binary elementwise (sub)
        3. var_x = mean(x_centered^2)       # mul then reduction
        4. std_x = sqrt(var_x + eps)        # unary elementwise
        5. x_norm = x_centered / std_x      # binary elementwise (div)
        6. output = x_norm * gamma + beta   # binary elementwise (mul, add)
        """
        analyzer = EinsumAnalyzer()
        
        # Step 1: mean reduction
        mean_op = analyzer.get_einsum_op("mean", {"Input": [8, 16, 64]}, dims=[2])
        assert mean_op.reduction_op == "add"
        
        # Step 2: subtraction (centering)
        sub_op = analyzer.get_einsum_op("sub", {"Input": [8, 16, 64], "Input_1": [8, 16, 1]})
        assert sub_op.elementwise_op == "sub"
        
        # Step 5: division (normalize)
        div_op = analyzer.get_einsum_op("div", {"Input": [8, 16, 64], "Input_1": [8, 16, 1]})
        assert div_op.elementwise_op == "div"
        
        # Step 6a: scale by gamma
        mul_op = analyzer.get_einsum_op("mul", {"Input": [8, 16, 64], "Input_1": [64]})
        assert mul_op.elementwise_op == "mul"
        
        # Step 6b: add beta
        add_op = analyzer.get_einsum_op("add", {"Input": [8, 16, 64], "Input_1": [64]})
        assert add_op.elementwise_op == "add"
    
    def test_attention_expansion_concept(self):
        """Test that attention can be decomposed.
        
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Decomposition:
        1. scores = Q @ K^T                     # matmul
        2. scaled_scores = scores / sqrt(d_k)  # binary elementwise (div) or scale
        3. weights = softmax(scaled_scores)    # softmax (multi-node)
        4. output = weights @ V                 # matmul
        """
        analyzer = EinsumAnalyzer()
        
        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
        
        # Step 1: Q @ K^T -> scores
        qk_shapes = {
            "Input": [batch_size, num_heads, seq_len, head_dim],  # Q
            "Weight": [batch_size, num_heads, seq_len, head_dim],  # K (will be transposed)
        }
        qk_op = analyzer.get_einsum_op("matmul", qk_shapes)
        assert qk_op.elementwise_op == "mul"
        assert qk_op.reduction_op == "add"
        
        # Step 2: Scale by 1/sqrt(d_k)
        scale_shapes = {
            "Input": [batch_size, num_heads, seq_len, seq_len],  # scores
            "Input_1": [1],  # scalar scale factor
        }
        div_op = analyzer.get_einsum_op("div", scale_shapes)
        assert div_op.elementwise_op == "div"
        
        # Step 4: weights @ V -> output
        wv_shapes = {
            "Input": [batch_size, num_heads, seq_len, seq_len],  # weights
            "Weight": [batch_size, num_heads, seq_len, head_dim],  # V
        }
        wv_op = analyzer.get_einsum_op("matmul", wv_shapes)
        assert wv_op.elementwise_op == "mul"
        assert wv_op.reduction_op == "add"
    
    def test_gelu_expansion_concept(self):
        """Test that GELU can be decomposed.
        
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        This is a complex function that in principle could be decomposed
        but is typically treated as a single fused kernel.
        """
        analyzer = EinsumAnalyzer()
        
        # GELU as a single unary operation
        gelu_op = analyzer.get_einsum_op("gelu", {"Input": [32, 64]})
        assert gelu_op.elementwise_op == "copy"  # Treated as unary transform
        assert gelu_op.reduction_op == "none"


class TestEinsumOpYamlSerialization:
    """Test that EinsumOp can be properly serialized for YAML output."""
    
    def test_einsum_op_to_dict(self):
        """Test converting EinsumOp to dict for YAML."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.get_einsum_op("add", {"Input": [32, 64], "Input_1": [32, 64]})
        
        # Simulate what pytorch_einsum_converter does
        output = {
            "type": einsum_op.name,
            "einsum_equation": einsum_op.equation,
            "elementwise_op": einsum_op.elementwise_op,
            "reduction_op": einsum_op.reduction_op,
            "is_real_einsum": einsum_op.is_real_einsum,
        }
        
        assert output["type"] == "add"
        assert output["elementwise_op"] == "add"
        assert output["reduction_op"] == "none"
        assert output["is_real_einsum"] is False
    
    def test_matmul_op_to_dict(self):
        """Test matmul serialization."""
        analyzer = EinsumAnalyzer()
        
        einsum_op = analyzer.get_einsum_op("matmul", {"Input": [32, 64], "Weight": [64, 128]})
        
        output = {
            "type": einsum_op.name,
            "einsum_equation": einsum_op.equation,
            "elementwise_op": einsum_op.elementwise_op,
            "reduction_op": einsum_op.reduction_op,
            "is_real_einsum": einsum_op.is_real_einsum,
        }
        
        assert output["type"] == "matmul"
        assert output["einsum_equation"] == "MK,KN->MN"
        assert output["elementwise_op"] == "mul"
        assert output["reduction_op"] == "add"
        assert output["is_real_einsum"] is True

