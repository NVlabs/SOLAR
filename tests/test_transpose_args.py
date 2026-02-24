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

from solar.einsum.ops.shape_ops import TensorManipulationHandler, generate_dim_labels


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
        shapes = {
            "Input": [2, 32, 4, 16],
            "Output": [2, 4, 32, 16],
        }
        result = handler.generate_einsum("transpose", shapes)
        
        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"
        assert result.elementwise_op == "copy"
        assert result.reduction_op == "none"
        assert result.is_real_einsum is False
    
    def test_transpose_swap_dims_2_3(self, handler):
        """Test transpose(2, 3) on 4D tensor."""
        # [2, 4, 32, 16] -> [2, 4, 16, 32] (swap dims 2 and 3)
        shapes = {
            "Input": [2, 4, 32, 16],
            "Output": [2, 4, 16, 32],
        }
        result = handler.generate_einsum("transpose", shapes)
        
        assert result.equation == "ABCD->ABDC", \
            f"Expected ABCD->ABDC, got {result.equation}"
    
    def test_transpose_2d_matrix(self, handler):
        """Test transpose on 2D matrix."""
        # [32, 64] -> [64, 32]
        shapes = {
            "Input": [32, 64],
            "Output": [64, 32],
        }
        result = handler.generate_einsum("transpose", shapes)
        
        assert result.equation == "AB->BA", \
            f"Expected AB->BA, got {result.equation}"
    
    def test_t_operation(self, handler):
        """Test t() operation (2D transpose)."""
        # t() is equivalent to transpose(0, 1) for 2D tensors
        shapes = {
            "Input": [32, 64],
            "Output": [64, 32],
        }
        result = handler.generate_einsum("t", shapes)
        
        assert result.equation == "AB->BA", \
            f"Expected AB->BA, got {result.equation}"
    
    def test_contiguous_identity(self, handler):
        """Test contiguous operation (should be identity)."""
        shapes = {
            "Input": [2, 4, 32, 16],
            "Output": [2, 4, 32, 16],
        }
        result = handler.generate_einsum("contiguous", shapes)
        
        assert result.equation == "ABCD->ABCD", \
            f"Expected ABCD->ABCD, got {result.equation}"
    
    def test_permute_reorder_all_dims(self, handler):
        """Test permute that reorders all dimensions."""
        # permute(0, 2, 1, 3): [2, 32, 4, 16] -> [2, 4, 32, 16]
        shapes = {
            "Input": [2, 32, 4, 16],
            "Output": [2, 4, 32, 16],
        }
        result = handler.generate_einsum("permute", shapes)
        
        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"
    
    def test_transpose_with_duplicate_dims(self, handler):
        """Test transpose with duplicate dimension sizes."""
        # [2, 32, 32, 16] -> [2, 32, 32, 16] (swap identical dims)
        # This is ambiguous, but should still produce valid output
        shapes = {
            "Input": [2, 32, 32, 16],
            "Output": [2, 32, 32, 16],
        }
        result = handler.generate_einsum("transpose", shapes)
        
        # Should be identity since shapes are the same
        assert result.equation == "ABCD->ABCD", \
            f"Expected ABCD->ABCD for identical shapes, got {result.equation}"
    
    def test_transpose_batch_head_swap(self, handler):
        """Test attention-style transpose: [B, S, H, D] -> [B, H, S, D]."""
        shapes = {
            "Input": [2, 32, 4, 16],  # [B, S, H, D]
            "Output": [2, 4, 32, 16],  # [B, H, S, D]
        }
        result = handler.generate_einsum("transpose", shapes)
        
        assert result.equation == "ABCD->ACBD", \
            f"Expected ABCD->ACBD, got {result.equation}"


class TestReshapeOperations:
    """Tests for reshape/view operations (different input/output ranks)."""
    
    @pytest.fixture
    def handler(self):
        return TensorManipulationHandler()
    
    def test_view_expand_dims(self, handler):
        """Test view that expands dimensions."""
        # [2, 32, 64] -> [2, 32, 4, 16]
        shapes = {
            "Input": [2, 32, 64],
            "Output": [2, 32, 4, 16],
        }
        result = handler.generate_einsum("view", shapes)
        
        # Should use different labels for input/output
        assert "->" in result.equation
        assert result.equation.startswith("I0I1I2->"), \
            f"Expected input labels I0I1I2, got {result.equation}"
        assert "O0O1O2O3" in result.equation, \
            f"Expected output labels O0O1O2O3, got {result.equation}"
    
    def test_reshape_collapse_dims(self, handler):
        """Test reshape that collapses dimensions."""
        # [2, 32, 4, 16] -> [2, 32, 64]
        shapes = {
            "Input": [2, 32, 4, 16],
            "Output": [2, 32, 64],
        }
        result = handler.generate_einsum("reshape", shapes)
        
        assert result.equation == "I0I1I2I3->O0O1O2", \
            f"Expected I0I1I2I3->O0O1O2, got {result.equation}"
    
    def test_flatten(self, handler):
        """Test flatten operation."""
        # [2, 32, 64] -> [2, 2048]
        shapes = {
            "Input": [2, 32, 64],
            "Output": [2, 2048],
        }
        result = handler.generate_einsum("flatten", shapes)
        
        assert result.equation == "I0I1I2->O0O1", \
            f"Expected I0I1I2->O0O1, got {result.equation}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def handler(self):
        return TensorManipulationHandler()
    
    def test_missing_input_shape(self, handler):
        """Test error when Input shape is missing."""
        shapes = {"Output": [2, 4, 32, 16]}
        
        with pytest.raises(ValueError, match="Missing Input shape"):
            handler.generate_einsum("transpose", shapes)
    
    def test_missing_output_shape_uses_input(self, handler):
        """Test that missing Output shape defaults to Input shape."""
        shapes = {"Input": [2, 4, 32, 16]}
        result = handler.generate_einsum("contiguous", shapes)
        
        # Should be identity
        assert result.equation == "ABCD->ABCD"
    
    def test_high_rank_tensor(self, handler):
        """Test with high-rank tensor (>4 dims)."""
        shapes = {
            "Input": [2, 3, 4, 5, 6, 7],
            "Output": [2, 3, 4, 5, 7, 6],  # Swap last two dims
        }
        result = handler.generate_einsum("transpose", shapes)
        
        assert result.equation == "ABCDEF->ABCDFE", \
            f"Expected ABCDEF->ABCDFE, got {result.equation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

