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

"""Test transpose/permute einsum generation using shape-based inference.

Since torchview's FunctionNode doesn't capture function arguments (args/kwargs),
we use shape-based inference to determine the transpose permutation.

This test verifies that the TensorManipulationHandler correctly infers
the permutation from input/output shapes for transpose, permute, t, and
contiguous operations.
"""

import pytest
import torch
import torch.nn as nn

from typing import List, Optional

from solar.common.types import TensorShapes
from solar.common.utils import parse_dim_tokens
from solar.einsum.ops.shape_ops import TensorManipulationHandler, generate_dim_labels


def _ts(inp, out=None):
    """Shorthand to build TensorShapes from a single input/output pair."""
    return TensorShapes(inputs=[inp], outputs=[out] if out else [])


def _parse_unary_equation_ranks(equation: str):
    """Parse a unary einsum equation into (input_ranks, output_ranks)."""
    assert "->" in equation, f"Invalid equation (missing ->): {equation}"
    lhs, rhs = equation.split("->", 1)
    lhs_operands = [s.strip() for s in lhs.split(",") if s.strip()]
    assert len(lhs_operands) == 1, f"Expected unary equation, got: {equation}"
    in_ranks = parse_dim_tokens(lhs_operands[0])
    out_ranks = parse_dim_tokens(rhs.strip())
    return in_ranks, out_ranks


def _assert_unary_permute_equation(
    equation: str,
    input_shape: List[int],
    output_shape: List[int],
    expected_perm: Optional[List[int]] = None,
) -> None:
    """Assert a unary permutation equation, independent of rank letters.

    Checks:
    - output ranks are a permutation of input ranks
    - applying the permutation to input_shape yields output_shape
    - (optional) the exact permutation matches expected_perm
    """
    in_ranks, out_ranks = _parse_unary_equation_ranks(equation)
    assert len(in_ranks) == len(input_shape)
    assert len(out_ranks) == len(output_shape)
    assert set(in_ranks) == set(out_ranks)

    perm = [in_ranks.index(r) for r in out_ranks]
    assert [input_shape[i] for i in perm] == output_shape
    if expected_perm is not None:
        assert perm == expected_perm


class TestGenerateDimLabels:
    """Tests for dimension label generation."""
    
    def test_generate_labels_basic(self):
        """Test basic label generation."""
        labels = generate_dim_labels(4)
        assert labels == ["A", "B", "C", "D"]
    
    def test_generate_labels_with_prefix(self):
        """Test label generation with prefix."""
        labels = generate_dim_labels(3, prefix="I")
        assert labels == ["I0", "I1", "I2"]
    
    def test_generate_labels_overflow(self):
        """Test label generation beyond 26 dims."""
        labels = generate_dim_labels(28)
        assert labels[0] == "A"
        assert labels[25] == "Z"
        assert labels[26] == "A0"
        assert labels[27] == "B0"


class TestTransposeShapeInference:
    """Tests for transpose einsum generation using shape-based inference."""
    
    @pytest.fixture
    def handler(self):
        return TensorManipulationHandler()
    
    def test_transpose_swap_dims_1_2(self, handler):
        """Test transpose(1, 2) on 4D tensor."""
        # [2, 32, 4, 16] -> [2, 4, 32, 16] (swap dims 1 and 2)
        input_shape = [2, 32, 4, 16]
        output_shape = [2, 4, 32, 16]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("transpose", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 2, 1, 3],
        )
        assert result.elementwise_op == "copy"
        assert result.reduction_op == "none"
        assert result.is_real_einsum is False
    
    def test_transpose_swap_dims_2_3(self, handler):
        """Test transpose(2, 3) on 4D tensor."""
        # [2, 4, 32, 16] -> [2, 4, 16, 32] (swap dims 2 and 3)
        input_shape = [2, 4, 32, 16]
        output_shape = [2, 4, 16, 32]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("transpose", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 1, 3, 2],
        )
    
    def test_transpose_2d_matrix(self, handler):
        """Test transpose on 2D matrix."""
        # [32, 64] -> [64, 32]
        input_shape = [32, 64]
        output_shape = [64, 32]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("transpose", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[1, 0],
        )
    
    def test_t_operation(self, handler):
        """Test t() operation (2D transpose)."""
        # t() is equivalent to transpose(0, 1) for 2D tensors
        input_shape = [32, 64]
        output_shape = [64, 32]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("t", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[1, 0],
        )
    
    def test_contiguous_identity(self, handler):
        """Test contiguous operation (should be identity)."""
        input_shape = [2, 4, 32, 16]
        output_shape = [2, 4, 32, 16]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("contiguous", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 1, 2, 3],
        )
    
    def test_permute_reorder_all_dims(self, handler):
        """Test permute that reorders all dimensions."""
        # permute(0, 2, 1, 3): [2, 32, 4, 16] -> [2, 4, 32, 16]
        input_shape = [2, 32, 4, 16]
        output_shape = [2, 4, 32, 16]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("permute", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 2, 1, 3],
        )
    
    def test_transpose_with_duplicate_dims(self, handler):
        """Test transpose with duplicate dimension sizes."""
        # [2, 32, 32, 16] -> [2, 32, 32, 16] (swap identical dims)
        # This is ambiguous, but should still produce valid output
        input_shape = [2, 32, 32, 16]
        output_shape = [2, 32, 32, 16]
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("transpose", shapes)
        
        # Should be identity since shapes are the same
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 1, 2, 3],
        )
    
    def test_transpose_batch_head_swap(self, handler):
        """Test attention-style transpose: [B, S, H, D] -> [B, H, S, D]."""
        input_shape = [2, 32, 4, 16]
        output_shape = [2, 4, 32, 16]
        shapes = _ts(input_shape, output_shape)  # [B,S,H,D] -> [B,H,S,D]
        result = handler.generate_einsum("transpose", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 2, 1, 3],
        )


class TestReshapeOperations:
    """Tests for reshape/view operations (different input/output ranks)."""
    
    @pytest.fixture
    def handler(self):
        return TensorManipulationHandler()
    
    def test_view_expand_dims(self, handler):
        """Test view that expands dimensions.
        
        Preserved dims keep the same rank token; reshaped dims get new rank tokens.
        [2,32,64] -> [2,32,4,16]: dims 0 and 1 are preserved, dim 2 is split.
        """
        # [2, 32, 64] -> [2, 32, 4, 16]
        shapes = _ts([2, 32, 64], [2, 32, 4, 16])
        result = handler.generate_einsum("view", shapes)
        
        in_ranks, out_ranks = _parse_unary_equation_ranks(result.equation)
        assert len(in_ranks) == 3
        assert len(out_ranks) == 4
        # Based on shape: dims 0 and 1 remain unchanged, so their rank tokens must match.
        assert out_ranks[:2] == in_ranks[:2]
        assert set(in_ranks).intersection(out_ranks) == set(in_ranks[:2])
    
    def test_reshape_collapse_dims(self, handler):
        """Test reshape that collapses dimensions.
        
        [2,32,4,16] -> [2,32,64]: dims 0 and 1 are preserved, dims 2 and 3 collapse.
        """
        # [2, 32, 4, 16] -> [2, 32, 64]
        shapes = _ts([2, 32, 4, 16], [2, 32, 64])
        result = handler.generate_einsum("reshape", shapes)
        
        in_ranks, out_ranks = _parse_unary_equation_ranks(result.equation)
        assert len(in_ranks) == 4
        assert len(out_ranks) == 3
        # Based on shape: dims 0 and 1 remain unchanged, so their rank tokens must match.
        assert out_ranks[:2] == in_ranks[:2]
        assert set(in_ranks).intersection(out_ranks) == set(in_ranks[:2])
    
    def test_flatten(self, handler):
        """Test flatten operation.
        
        [2,32,64] -> [2,2048]: dim 0 is preserved, dims 1 and 2 flatten.
        """
        # [2, 32, 64] -> [2, 2048]
        shapes = _ts([2, 32, 64], [2, 2048])
        result = handler.generate_einsum("flatten", shapes)
        
        in_ranks, out_ranks = _parse_unary_equation_ranks(result.equation)
        assert len(in_ranks) == 3
        assert len(out_ranks) == 2
        # Based on shape: dim 0 remains unchanged, so its rank token must match.
        assert out_ranks[0] == in_ranks[0]
        assert set(in_ranks).intersection(out_ranks) == {in_ranks[0]}


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def handler(self):
        return TensorManipulationHandler()
    
    def test_missing_input_shape(self, handler):
        """Test error when Input shape is missing."""
        shapes = TensorShapes(inputs=[], outputs=[[2, 4, 32, 16]])
        
        with pytest.raises(ValueError, match="Missing Input shape"):
            handler.generate_einsum("transpose", shapes)
    
    def test_missing_output_shape_uses_input(self, handler):
        """Test that missing Output shape defaults to Input shape."""
        shapes = _ts([2, 4, 32, 16])
        result = handler.generate_einsum("contiguous", shapes)
        
        # Should be identity
        in_ranks, out_ranks = _parse_unary_equation_ranks(result.equation)
        assert in_ranks == out_ranks
    
    def test_high_rank_tensor(self, handler):
        """Test with high-rank tensor (>4 dims)."""
        input_shape = [2, 3, 4, 5, 6, 7]
        output_shape = [2, 3, 4, 5, 7, 6]  # swap last two
        shapes = _ts(input_shape, output_shape)
        result = handler.generate_einsum("transpose", shapes)
        
        _assert_unary_permute_equation(
            result.equation,
            input_shape=input_shape,
            output_shape=output_shape,
            expected_perm=[0, 1, 2, 3, 5, 4],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

