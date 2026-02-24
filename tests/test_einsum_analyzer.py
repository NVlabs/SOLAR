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

"""Tests for einsum analyzer."""

import pytest
import numpy as np
from solar.einsum import EinsumAnalyzer
from solar.einsum.ops import EinsumOp, EinsumOperand
from solar.common.types import ShapeDict


class TestEinsumOperand:
    """Tests for EinsumOperand dataclass."""
    
    def test_basic_creation(self):
        """Test creating basic einsum operand."""
        operand = EinsumOperand(name="Input", dims=["A", "B", "C"])
        
        assert operand.name == "Input"
        assert operand.dims == ["A", "B", "C"]
        assert operand.is_output is False
    
    def test_weight_operand(self):
        """Test creating weight operand."""
        operand = EinsumOperand(name="Weight", dims=["O", "I", "H", "W"])
        assert operand.name == "Weight"
    
    def test_output_operand(self):
        """Test creating output operand."""
        operand = EinsumOperand(name="Output", dims=["N", "O", "H", "W"], is_output=True)
        
        assert operand.is_output is True


class TestEinsumOp:
    """Tests for EinsumOp dataclass."""
    
    def test_basic_creation(self):
        """Test creating basic einsum operation."""
        operands = [
            EinsumOperand("A", ["A", "B"]),
            EinsumOperand("B", ["B", "C"]),
            EinsumOperand("Output", ["A", "C"], is_output=True),
        ]
        
        op = EinsumOp(
            equation="AB,BC->AC",
            operands=operands,
            name="matmul",
        )
        
        assert op.equation == "AB,BC->AC"
        assert len(op.operands) == 3
    
    def test_get_compute_cost(self):
        """Test compute cost calculation."""
        op = EinsumOp(
            equation="AB,BC->AC",
            operands=[
                EinsumOperand("A", ["A", "B"]),
                EinsumOperand("B", ["B", "C"]),
                EinsumOperand("Output", ["A", "C"], is_output=True),
            ],
            name="matmul",
        )
        
        shapes = {"A": [2, 3], "B": [3, 4]}
        cost = op.get_compute_cost(shapes)
        
        # Compute cost should be 2 * 3 * 4 = 24
        assert cost == 24


class TestEinsumAnalyzer:
    """Tests for EinsumAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return EinsumAnalyzer(debug=True)
    
    def test_matmul(self, analyzer):
        """Test matrix multiplication einsum generation."""
        input_shape = [2, 3]
        weight_shape = [3, 4]
        
        op = analyzer.generate_matmul_einsum(input_shape, weight_shape)
        
        assert "->" in op.equation
        assert op.get_compute_cost({"Input": input_shape, "Weight": weight_shape}) == 2 * 3 * 4
    
    def test_conv2d(self, analyzer):
        """Test 2D convolution einsum generation."""
        input_shape = [1, 3, 224, 224]
        weight_shape = [64, 3, 7, 7]
        
        op = analyzer.generate_conv2d_einsum(
            input_shape, weight_shape,
            stride=[2, 2], padding=[3, 3]
        )
        
        assert "->" in op.equation
    
    def test_elementwise(self, analyzer):
        """Test elementwise operation einsum generation."""
        shape = [2, 3, 4]
        
        op = analyzer.generate_elementwise_einsum(shape, "relu")
        
        assert op.equation == "ABC->ABC"
        assert op.get_compute_cost({"Input": shape}) == 2 * 3 * 4
    
    def test_reduction(self, analyzer):
        """Test reduction operation einsum generation."""
        shape = [2, 3, 4, 5]
        
        # Test sum reduction along dim 1
        op = analyzer.generate_reduction_einsum(shape, "sum", dims=[1])
        assert op.equation == "ABCD->ACD"
        
        # Test mean reduction all dims
        op = analyzer.generate_reduction_einsum(shape, "mean", dims=None)
        assert op.equation == "ABCD->"
        
        # Test prod reduction with keepdim
        op = analyzer.get_reduction_einsum_op(
            "torch.prod",
            {"Input": shape},
            reduce_dims=[2],
            keepdim=True
        )
        assert op.equation == "ABCD->ABD"
    
    def test_torch_prod(self, analyzer):
        """Test torch.prod support."""
        # Full reduction
        shapes = {"Input": [2, 3, 4]}
        equation = analyzer.get_torch_einsum_equation("torch.prod", shapes=shapes)
        assert equation == "ABC->"
        
        cost = analyzer.get_compute_cost("torch.prod", shapes=shapes)
        assert cost == 2 * 3 * 4
        
        # Partial reduction
        op = analyzer.get_reduction_einsum_op(
            "torch.prod",
            shapes,
            reduce_dims=[1],
            keepdim=False
        )
        assert op.equation == "ABC->AC"
    
    def test_get_operation_from_name(self, analyzer):
        """Test operation name mapping."""
        # Test convolution variants
        assert analyzer._get_operation_from_name("Conv2d") == "conv2d"
        assert analyzer._get_operation_from_name("torch.nn.Conv2d") == "conv2d"
        
        # Test linear/matmul
        assert analyzer._get_operation_from_name("Linear") == "linear"
        assert analyzer._get_operation_from_name("torch.matmul") == "matmul"
        
        # Test elementwise
        assert analyzer._get_operation_from_name("ReLU") == "relu"
        assert analyzer._get_operation_from_name("torch.sigmoid") == "sigmoid"
        
        # Test reduction
        assert analyzer._get_operation_from_name("torch.sum") == "sum"
        assert analyzer._get_operation_from_name("torch.prod") == "prod"
    
    def test_compute_cost_calculation(self, analyzer):
        """Test compute cost calculations for various operations."""
        # Matrix multiplication
        shapes = {"Input": [10, 20], "Weight": [20, 30]}
        cost = analyzer.get_compute_cost("matmul", shapes)
        assert cost == 10 * 20 * 30
        
        # Convolution (simplified test)
        shapes = {
            "Input": [1, 3, 32, 32],
            "Weight": [16, 3, 3, 3]
        }
        cost = analyzer.get_compute_cost("conv2d", shapes, stride=[1, 1], padding=[1, 1])
        # Output will be [1, 16, 32, 32]
        # Cost per output element: 3 * 3 * 3 = 27
        # Total: 16 * 32 * 32 * 27
        expected = 16 * 32 * 32 * 3 * 3 * 3
        assert cost == expected
    
    def test_memory_cost_calculation(self, analyzer):
        """Test memory cost calculations."""
        shapes = {
            "Input": [10, 20],
            "Weight": [20, 30],
            "Output": [10, 30]
        }
        
        memory = analyzer.get_memory_cost(shapes)
        
        assert memory["Input"] == 10 * 20
        assert memory["Weight"] == 20 * 30
        assert memory["Output"] == 10 * 30
        assert memory["total"] == 10*20 + 20*30 + 10*30


class TestIntegration:
    """Integration tests for einsum analyzer."""
    
    def test_full_model_analysis(self):
        """Test analyzing a full model."""
        analyzer = EinsumAnalyzer(debug=False)
        
        # Simulate a simple model: Conv -> ReLU -> Linear
        operations = [
            ("conv2d", {
                "Input": [1, 3, 224, 224],
                "Weight": [64, 3, 7, 7]
            }, {"stride": [2, 2], "padding": [3, 3]}),
            ("relu", {
                "Input": [1, 64, 112, 112]
            }, {}),
            ("linear", {
                "Input": [1, 64 * 112 * 112],
                "Weight": [1000, 64 * 112 * 112]
            }, {})
        ]
        
        total_compute = 0
        total_memory = 0
        
        for op_name, shapes, kwargs in operations:
            compute = analyzer.get_compute_cost(op_name, shapes, **kwargs)
            memory = analyzer.get_memory_cost(shapes)
            
            total_compute += compute
            total_memory += memory["total"]
        
        assert total_compute > 0
        assert total_memory > 0
    
    def test_kernelbench_compatibility(self):
        """Test analyzer works with kernelbench models."""
        analyzer = EinsumAnalyzer()
        
        kernelbench_ops = [
            "Conv2d", "Linear", "ReLU", "BatchNorm2d", "MaxPool2d"
        ]
        
        for op in kernelbench_ops:
            normalized = analyzer._get_operation_from_name(op)
            assert normalized is not None
