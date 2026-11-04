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

from solar.common.types import TensorShapes
from solar.common.utils import parse_dim_tokens
from solar.einsum.ops.shape_ops import TensorManipulationHandler, generate_dim_labels


def _ts(inp, out=None):
    """Shorthand to build TensorShapes from a single input/output pair."""
    return TensorShapes(inputs=[inp], outputs=[out] if out else [])


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
        shapes = _ts([2, 32, 4, 16], [2, 4, 32, 16])
        result = handler.generate_einsum("transpose", shapes)

        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"
        assert result.elementwise_op == "copy"
        assert result.reduction_op == "none"
        assert result.is_real_einsum is False

    def test_transpose_swap_dims_2_3(self, handler):
        """Test transpose(2, 3) on 4D tensor."""
        # [2, 4, 32, 16] -> [2, 4, 16, 32] (swap dims 2 and 3)
        shapes = _ts([2, 4, 32, 16], [2, 4, 16, 32])
        result = handler.generate_einsum("transpose", shapes)

        assert result.equation == "ABCD->ABDC", \
            f"Expected ABCD->ABDC, got {result.equation}"

    def test_transpose_2d_matrix(self, handler):
        """Test transpose on 2D matrix."""
        # [32, 64] -> [64, 32]
        shapes = _ts([32, 64], [64, 32])
        result = handler.generate_einsum("transpose", shapes)

        assert result.equation == "AB->BA", \
            f"Expected AB->BA, got {result.equation}"

    def test_t_operation(self, handler):
        """Test t() operation (2D transpose)."""
        # t() is equivalent to transpose(0, 1) for 2D tensors
        shapes = _ts([32, 64], [64, 32])
        result = handler.generate_einsum("t", shapes)

        assert result.equation == "AB->BA", \
            f"Expected AB->BA, got {result.equation}"

    def test_contiguous_identity(self, handler):
        """Test contiguous operation (should be identity)."""
        shapes = _ts([2, 4, 32, 16], [2, 4, 32, 16])
        result = handler.generate_einsum("contiguous", shapes)

        assert result.equation == "ABCD->ABCD", \
            f"Expected ABCD->ABCD, got {result.equation}"

    def test_permute_reorder_all_dims(self, handler):
        """Test permute that reorders all dimensions."""
        # permute(0, 2, 1, 3): [2, 32, 4, 16] -> [2, 4, 32, 16]
        shapes = _ts([2, 32, 4, 16], [2, 4, 32, 16])
        result = handler.generate_einsum("permute", shapes)

        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"

    def test_transpose_with_duplicate_dims(self, handler):
        """Test transpose with duplicate dimension sizes."""
        # [2, 32, 32, 16] -> [2, 32, 32, 16] (swap identical dims)
        # This is ambiguous, but should still produce valid output
        shapes = _ts([2, 32, 32, 16], [2, 32, 32, 16])
        result = handler.generate_einsum("transpose", shapes)

        # Should be identity since shapes are the same
        assert result.equation == "ABCD->ABCD", \
            f"Expected ABCD->ABCD for identical shapes, got {result.equation}"

    def test_transpose_batch_head_swap(self, handler):
        """Test attention-style transpose: [B, S, H, D] -> [B, H, S, D]."""
        shapes = _ts([2, 32, 4, 16], [2, 4, 32, 16])  # [B,S,H,D] -> [B,H,S,D]
        result = handler.generate_einsum("transpose", shapes)

        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"


def _parse_unary_equation_ranks(equation: str):
    """Parse a unary einsum equation into (input_ranks, output_ranks)."""
    assert "->" in equation, f"Invalid equation (missing ->): {equation}"
    lhs, rhs = equation.split("->", 1)
    lhs_operands = [s.strip() for s in lhs.split(",")]
    assert len(lhs_operands) == 1 and lhs_operands[0], \
        f"Expected unary equation, got: {equation}"
    assert rhs.strip(), f"Empty output operand: {equation}"
    in_ranks = parse_dim_tokens(lhs_operands[0])
    out_ranks = parse_dim_tokens(rhs.strip())
    return in_ranks, out_ranks


def _assert_reshape_tokens(in_ranks, out_ranks, n_preserved):
    """Rank tokens of a reshape: unique per operand, the first ``n_preserved``
    dims share tokens, every changed output dim gets a FRESH token."""
    assert len(set(in_ranks)) == len(in_ranks), f"duplicate input ranks {in_ranks}"
    assert len(set(out_ranks)) == len(out_ranks), f"duplicate output ranks {out_ranks}"
    assert out_ranks[:n_preserved] == in_ranks[:n_preserved]
    assert set(out_ranks[n_preserved:]).isdisjoint(in_ranks), \
        f"changed dims must not reuse input ranks: {in_ranks} -> {out_ranks}"


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
        _assert_reshape_tokens(in_ranks, out_ranks, 2)

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
        _assert_reshape_tokens(in_ranks, out_ranks, 2)

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
        _assert_reshape_tokens(in_ranks, out_ranks, 1)


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
        assert result.equation == "ABCD->ABCD"

    def test_high_rank_tensor(self, handler):
        """Test with high-rank tensor (>4 dims)."""
        shapes = _ts([2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 7, 6])  # swap last two
        result = handler.generate_einsum("transpose", shapes)

        assert result.equation == "ABCDEF->ABCDFE", \
            f"Expected ABCDEF->ABCDFE, got {result.equation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
