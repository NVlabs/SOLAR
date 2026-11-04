# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for standard, depthwise, and group-wise convolution handling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.einsum.ops.conv_ops import Conv1dHandler, Conv2dHandler
from equation_utils import normalize_equation

class TestConv1dStandard:
    def test_standard_conv1d(self):
        handler = Conv1dHandler()
        ts = TensorShapes(inputs=[[2, 64, 128], [128, 64, 3]], outputs=[[2, 128, 126]])
        result = handler.generate_einsum("conv1d", ts)
        assert normalize_equation(result.equation) == "AB(C+D),EBD->AEC"

class TestConv1dDepthwise:
    def test_depthwise_conv1d(self):
        handler = Conv1dHandler()
        ts = TensorShapes(inputs=[[16, 16384, 515], [16384, 1, 4]], outputs=[[16, 16384, 512]])
        result = handler.generate_einsum("conv1d", ts, module_args={"groups": 16384})
        assert normalize_equation(result.equation) == "AB(C+D),BED->ABC"

    def test_depthwise_conv1d_macs(self):
        handler = Conv1dHandler()
        B, C, L, K = 16, 16384, 512, 4
        ts_std = TensorShapes(inputs=[[B, C, L+K-1], [C, C, K]], outputs=[[B, C, L]])
        ts_dw = TensorShapes(inputs=[[B, C, L+K-1], [C, 1, K]], outputs=[[B, C, L]])
        r_std = handler.generate_einsum("conv1d", ts_std)
        r_dw = handler.generate_einsum("conv1d", ts_dw, module_args={"groups": C})
        assert normalize_equation(r_std.equation) == "AB(C+D),EBD->AEC"
        assert normalize_equation(r_dw.equation) == "AB(C+D),BED->ABC"

class TestConv1dGroupwise:
    def test_groupwise_conv1d(self):
        handler = Conv1dHandler()
        G, I, O_pg, B, K, L, L_out = 4, 16, 16, 2, 3, 128, 126
        ts = TensorShapes(
            inputs=[[B, G, I, L], [G, O_pg, I, K]],
            outputs=[[B, G, O_pg, L_out]])
        result = handler.generate_einsum(
            "conv1d", ts,
            module_args={"groups": G, "in_channels": G*I, "out_channels": G*O_pg})
        assert normalize_equation(result.equation) == "ABC(D+E),BFCE->ABFD"
        cost = result.get_compute_cost(ts)
        assert cost == B * G * O_pg * L_out * I * K

class TestConv2dStandard:
    def test_standard_conv2d(self):
        handler = Conv2dHandler()
        ts = TensorShapes(inputs=[[1, 64, 32, 32], [128, 64, 3, 3]], outputs=[[1, 128, 30, 30]])
        result = handler.generate_einsum("conv2d", ts)
        assert normalize_equation(result.equation) == "AB(C+D)(E+F),GBDF->AGCE"

class TestConv2dDepthwise:
    def test_depthwise_conv2d(self):
        handler = Conv2dHandler()
        ts = TensorShapes(inputs=[[8, 128, 14, 14], [128, 1, 7, 7]], outputs=[[8, 128, 8, 8]])
        result = handler.generate_einsum("conv2d", ts, module_args={"groups": 128})
        assert normalize_equation(result.equation) == "AB(C+D)(E+F),BGDF->ABCE"

    def test_depthwise_conv2d_macs(self):
        handler = Conv2dHandler()
        B, C, H, W, KH, KW = 8, 128, 14, 14, 7, 7
        ts_std = TensorShapes(inputs=[[B,C,H,W],[C,C,KH,KW]], outputs=[[B,C,H-KH+1,W-KW+1]])
        ts_dw = TensorShapes(inputs=[[B,C,H,W],[C,1,KH,KW]], outputs=[[B,C,H-KH+1,W-KW+1]])
        r_std = handler.generate_einsum("conv2d", ts_std)
        r_dw = handler.generate_einsum("conv2d", ts_dw, module_args={"groups": C})
        assert normalize_equation(r_std.equation) == "AB(C+D)(E+F),GBDF->AGCE"
        assert normalize_equation(r_dw.equation) == "AB(C+D)(E+F),BGDF->ABCE"

    def test_groupwise_c_per_group_1_but_o_ne_c(self):
        handler = Conv2dHandler()
        G, I, O_pg, B = 128, 1, 2, 1
        H, W, KH, KW, H_out, W_out = 32, 32, 3, 3, 30, 30
        ts = TensorShapes(
            inputs=[[B,G,I,H,W],[G,O_pg,I,KH,KW]],
            outputs=[[B,G,O_pg,H_out,W_out]])
        result = handler.generate_einsum(
            "conv2d", ts,
            module_args={"groups": G, "in_channels": G*I, "out_channels": G*O_pg})
        assert "G" in result.equation
        cost = result.get_compute_cost(ts)
        assert cost == B * G * O_pg * H_out * W_out * I * KH * KW

class TestConv2dGroupwise:
    def test_groupwise_conv2d(self):
        handler = Conv2dHandler()
        G, I, O_pg, B = 4, 32, 64, 1
        H, W, KH, KW, H_out, W_out = 32, 32, 3, 3, 32, 32
        ts = TensorShapes(
            inputs=[[B,G,I,H,W],[G,O_pg,I,KH,KW]],
            outputs=[[B,G,O_pg,H_out,W_out]])
        result = handler.generate_einsum(
            "conv2d", ts,
            module_args={"groups": G, "in_channels": G*I, "out_channels": G*O_pg})
        assert normalize_equation(result.equation) == "ABC(D+E)(F+G),BHCEG->ABHDF"
        cost = result.get_compute_cost(ts)
        assert cost == B * G * O_pg * H_out * W_out * I * KH * KW

    def test_groupwise_conv2d_analyzer_equation_macs(self):
        analyzer = EinsumAnalyzer()
        G, I, O_pg, B = 4, 32, 64, 1
        H, W, KH, KW, H_out, W_out = 32, 32, 3, 3, 32, 32
        ts = TensorShapes(
            inputs=[[B,G,I,H,W],[G,O_pg,I,KH,KW]],
            outputs=[[B,G,O_pg,H_out,W_out]])
        cost = analyzer.get_compute_cost(
            "conv2d", ts, equation="BGI(P+R)(Q+S),GOIRS->BGOPQ")
        assert cost == B * G * O_pg * H_out * W_out * I * KH * KW

def run_tests():
    classes = [TestConv1dStandard, TestConv1dDepthwise, TestConv1dGroupwise,
               TestConv2dStandard, TestConv2dDepthwise, TestConv2dGroupwise]
    passed = failed = 0
    for cls in classes:
        inst = cls()
        for name in dir(inst):
            if name.startswith("test_"):
                try:
                    getattr(inst, name)()
                    passed += 1
                    print("  PASS: " + cls.__name__ + "." + name)
                except AssertionError as e:
                    failed += 1
                    print("  FAIL: " + cls.__name__ + "." + name + ": " + str(e))
                except Exception as e:
                    failed += 1
                    print("  ERROR: " + cls.__name__ + "." + name + ": " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
