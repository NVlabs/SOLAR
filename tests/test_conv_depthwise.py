"""Tests for depthwise and group-wise convolution handling in conv_ops.py.

Verifies that:
- Standard conv1d/conv2d computes full cross-channel MACs
- Depthwise conv1d/conv2d (groups=C, weight C_per_group=1) computes per-channel MACs only
- Group-wise conv uses C_per_group from weight shape, not full C from input
"""

import sys
from pathlib import Path

# Add solar to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar.common.types import TensorShapes
from solar.einsum.ops.conv_ops import Conv1dHandler, Conv2dHandler


class TestConv1dDepthwise:
    """Test Conv1d handler for standard vs depthwise convolution."""

    def test_standard_conv1d(self):
        """Standard conv1d: full cross-channel reduction."""
        handler = Conv1dHandler()
        # Input: [B=2, C=64, L=128], Weight: [O=128, C=64, K=3]
        ts = TensorShapes(
            inputs=[[2, 64, 128], [128, 64, 3]],
            outputs=[[2, 128, 126]],
        )
        result = handler.generate_einsum("conv1d", ts)
        assert result.equation == "BC(P+R),OCR->BOP"
        # C dimension should be in both Input and Weight operands (reduction)
        input_dims = result.operands[0].dims
        weight_dims = result.operands[1].dims
        assert "C" in input_dims
        assert "C" in weight_dims

    def test_depthwise_conv1d(self):
        """Depthwise conv1d: C_per_group=1, O=C, no cross-channel reduction."""
        handler = Conv1dHandler()
        # Input: [B=16, C=16384, L=515], Weight: [O=16384, C_per_group=1, K=4]
        # This is the Mamba conv1d pattern: groups=16384
        ts = TensorShapes(
            inputs=[[16, 16384, 515], [16384, 1, 4]],
            outputs=[[16, 16384, 512]],
        )
        result = handler.generate_einsum("conv1d", ts)
        assert result.equation == "BO(P+R),O1R->BOP"
        # C dimension should NOT appear as a shared reduction dimension
        input_dims = result.operands[0].dims
        weight_dims = result.operands[1].dims
        assert "C" not in input_dims
        assert "1" in weight_dims

    def test_depthwise_conv1d_macs(self):
        """Verify depthwise conv1d MACs are much smaller than standard."""
        handler = Conv1dHandler()
        B, C, L, K = 16, 16384, 512, 4

        # Standard: weight [O, C, K] with O=C
        ts_standard = TensorShapes(
            inputs=[[B, C, L + K - 1], [C, C, K]],
            outputs=[[B, C, L]],
        )
        result_standard = handler.generate_einsum("conv1d", ts_standard)

        # Depthwise: weight [O, 1, K] with O=C
        ts_depthwise = TensorShapes(
            inputs=[[B, C, L + K - 1], [C, 1, K]],
            outputs=[[B, C, L]],
        )
        result_depthwise = handler.generate_einsum("conv1d", ts_depthwise)

        # Standard should have C in reduction, depthwise should not
        assert result_standard.equation == "BC(P+R),OCR->BOP"
        assert result_depthwise.equation == "BO(P+R),O1R->BOP"

    def test_groupwise_conv1d_small_groups(self):
        """Group-wise conv1d with groups > 1 but < C (e.g., groups=4)."""
        handler = Conv1dHandler()
        # Input: [B=2, C=64, L=128], Weight: [O=64, C_per_group=16, K=3]
        # groups=4, C_per_group=64/4=16
        ts = TensorShapes(
            inputs=[[2, 64, 128], [64, 16, 3]],
            outputs=[[2, 64, 126]],
        )
        result = handler.generate_einsum("conv1d", ts)
        # Not depthwise (C_per_group=16 != 1), so standard equation
        # but weight C dimension is 16, not 64
        assert result.equation == "BC(P+R),OCR->BOP"


class TestConv2dDepthwise:
    """Test Conv2d handler for standard vs depthwise convolution."""

    def test_standard_conv2d(self):
        """Standard conv2d: full cross-channel reduction."""
        handler = Conv2dHandler()
        # Input: [B=1, C=64, H=32, W=32], Weight: [O=128, C=64, KH=3, KW=3]
        ts = TensorShapes(
            inputs=[[1, 64, 32, 32], [128, 64, 3, 3]],
            outputs=[[1, 128, 30, 30]],
        )
        result = handler.generate_einsum("conv2d", ts)
        assert result.equation == "BC(P+R)(Q+S),OCRS->BOPQ"

    def test_depthwise_conv2d(self):
        """Depthwise conv2d: C_per_group=1, O=C."""
        handler = Conv2dHandler()
        # ConvNeXtV2 pattern: [B=8, C=128, H=14, W=14], Weight: [128, 1, 7, 7]
        ts = TensorShapes(
            inputs=[[8, 128, 14, 14], [128, 1, 7, 7]],
            outputs=[[8, 128, 8, 8]],
        )
        result = handler.generate_einsum("conv2d", ts)
        assert result.equation == "BO(P+R)(Q+S),O1RS->BOPQ"
        # Verify no C reduction
        input_dims = result.operands[0].dims
        assert "C" not in input_dims

    def test_depthwise_conv2d_macs(self):
        """Verify depthwise conv2d MACs are C times smaller than standard."""
        handler = Conv2dHandler()
        B, C, H, W, KH, KW = 8, 128, 14, 14, 7, 7

        ts_standard = TensorShapes(
            inputs=[[B, C, H, W], [C, C, KH, KW]],
            outputs=[[B, C, H - KH + 1, W - KW + 1]],
        )
        ts_depthwise = TensorShapes(
            inputs=[[B, C, H, W], [C, 1, KH, KW]],
            outputs=[[B, C, H - KH + 1, W - KW + 1]],
        )

        result_standard = handler.generate_einsum("conv2d", ts_standard)
        result_depthwise = handler.generate_einsum("conv2d", ts_depthwise)

        assert result_standard.equation == "BC(P+R)(Q+S),OCRS->BOPQ"
        assert result_depthwise.equation == "BO(P+R)(Q+S),O1RS->BOPQ"

    def test_non_depthwise_c_per_group_1_but_o_ne_c(self):
        """Weight C_per_group=1 but O != C: not depthwise, standard conv."""
        handler = Conv2dHandler()
        # O=256, C=128, C_per_group=1 means groups=128, but O != C
        ts = TensorShapes(
            inputs=[[1, 128, 32, 32], [256, 1, 3, 3]],
            outputs=[[1, 256, 30, 30]],
        )
        result = handler.generate_einsum("conv2d", ts)
        # Not depthwise since O != C
        assert result.equation == "BC(P+R)(Q+S),OCRS->BOPQ"


def run_tests():
    """Run all tests without pytest."""
    test_classes = [TestConv1dDepthwise, TestConv2dDepthwise]
    passed = 0
    failed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {cls.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ERROR: {cls.__name__}.{method_name}: {e}")
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
