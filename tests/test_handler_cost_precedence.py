# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A handler-provided MAC count takes precedence over the equation's rank product.

The lineage snapshot the lineage rewrite made ``EinsumAnalyzer.get_compute_cost`` return
``compute_cost_from_equation(equation, ts)`` BEFORE building the handler op
whenever the graph analyzer passed a saved equation (every ``is_real_einsum``
layer).  That bypassed every handler that models its own MAC count: the
reviewed ConvTranspose count for level2/20 became 115,964,116,992
instead of 14,495,514,624 (8x: each ``(P-T)`` compound rank contributes both
atoms to the product).  These tests pin the precedence at the analyzer and
through ``EinsumGraphAnalyzer`` with the saved level2/20 layer.
"""
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.base import EinsumOp, compute_cost_from_equation
from solar.einsum.ops.conv_ops import ConvTransposeEinsumOp, convtranspose_macs

# KernelBench level2/20 ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd,
# layer `Model.conv_transpose3d` exactly as einsum_graph_renamed.yaml saves it.
L2_20_INPUT = [16, 32, 16, 32, 32]
L2_20_WEIGHT = [32, 64, 3, 3, 3]
L2_20_BIAS = [64]
L2_20_OUTPUT = [16, 64, 32, 64, 64]
L2_20_EQUATION = "BC(P-T)(Q-R)(U-S),CKTRS->BKPQU"
# hand count: N*C_in*D_in*H_in*W_in * C_out * kD*kH*kW (every input element
# scatters to every kernel tap of every output channel)
L2_20_MACS = 16 * 32 * 16 * 32 * 32 * 64 * 27
assert L2_20_MACS == 14_495_514_624
L2_20_LINEAGE_RANK_PRODUCT = 115_964_116_992  # what the lineage rewrite reported

L2_20_LAYER = {
    "type": "convtranspose3d",
    "einsum_equation": L2_20_EQUATION,
    "elementwise_op": "mul",
    "reduction_op": "add",
    "is_real_einsum": True,
    "is_einsum_supportable": True,
    "tensor_names": {
        "inputs": ["start.Output", "Model.conv_transpose3d.Weight", "Model.conv_transpose3d.Weight_1"],
        "outputs": ["Model.conv_transpose3d.Output"],
    },
    "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
    "tensor_shapes": {"inputs": [L2_20_INPUT, L2_20_WEIGHT, L2_20_BIAS], "outputs": [L2_20_OUTPUT]},
    "operands": {
        "Input": ["B", "C", "P-T", "Q-R", "U-S"],
        "Weight": ["C", "K", "T", "R", "S"],
        "Output": ["B", "K", "P", "Q", "U"],
    },
    "connections": {"inputs": [], "outputs": []},
    "tensor_dtypes": {"inputs": ["torch.float32"] * 3, "outputs": ["torch.float32"]},
    "taco_expression": "O0[b,k,p,q,u] = In0[b,c,(p-t),(q-r),(u-s)] * In1[c,k,t,r,s]",
}


# GPT-2 (KernelBench level4/7, bs32 seq256) `Model.scaled_dot_product_attention.qk_matmul`
# as einsum_graph_renamed.yaml saves it.  The lineage CHANGED this layer in the
# fix direction (402,653,184 -> 1,610,612,736 MACs; x12 layers = GPT-2's
# +14,495,514,624): the matmul handler's shape-only labels under-count the
# broadcast-batch QK^T, the emitted equation is decisive.  F3 ruling: preserve.
GPT2_QK_INPUT = [32, 12, 256, 64]
GPT2_QK_OUTPUT = [32, 12, 256, 256]
GPT2_QK_EQUATION = "BHQD,BHKD->BHQK"
GPT2_QK_MACS = 32 * 12 * 256 * 256 * 64
assert GPT2_QK_MACS == 1_610_612_736
GPT2_QK_HANDLER_ONLY_MACS = 402_653_184  # the pre-lineage (the earlier baseline) number
GPT2_QK_LAYER = {
    "type": "matmul",
    "einsum_equation": GPT2_QK_EQUATION,
    "elementwise_op": "mul",
    "reduction_op": "add",
    "is_real_einsum": True,
    "is_einsum_supportable": True,
    "operands": {"Input": ["B", "H", "Q", "D"], "Input_1": ["B", "H", "K", "D"], "Output": ["B", "H", "Q", "K"]},
    "tensor_names": {"inputs": ["Model.transpose_2.Output", "Model.cat.Output"],
                     "outputs": ["Model.scaled_dot_product_attention.qk_matmul.Output"]},
    "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
    "tensor_shapes": {"inputs": [GPT2_QK_INPUT, GPT2_QK_INPUT], "outputs": [GPT2_QK_OUTPUT]},
    "tensor_dtypes": {"inputs": ["torch.float32", "torch.float32"], "outputs": ["torch.float32"]},
    "connections": {"inputs": [], "outputs": []},
    "taco_expression": "O0[b,h,q,k] = In0[b,h,q,d] * In1[b,h,k,d]",
}


def _ts() -> TensorShapes:
    return TensorShapes(inputs=[L2_20_INPUT, L2_20_WEIGHT, L2_20_BIAS], outputs=[L2_20_OUTPUT])


def test_hand_count_matches_the_reviewed_handler():
    assert convtranspose_macs(L2_20_INPUT, L2_20_WEIGHT) == L2_20_MACS
    assert ConvTransposeEinsumOp.get_compute_cost is not EinsumOp.get_compute_cost


def test_equation_rank_product_is_not_a_mac_count_for_convtranspose():
    """Documents the defect: the saved equation's product is 8x the MAC count."""
    assert compute_cost_from_equation(L2_20_EQUATION, _ts()) == L2_20_LINEAGE_RANK_PRODUCT
    assert L2_20_LINEAGE_RANK_PRODUCT == 8 * L2_20_MACS


@pytest.mark.parametrize(
    "equation",
    [L2_20_EQUATION, "XY(A-B)(C-D)(E-F),YZBDF->XZACE"],  # saved form and a pure label renaming of it
)
def test_analyzer_prefers_handler_cost_over_passed_equation(equation):
    cost = EinsumAnalyzer().get_compute_cost("convtranspose3d", _ts(), equation=equation)
    assert cost == L2_20_MACS, f"equation {equation!r} bypassed the ConvTranspose handler: {cost}"


def test_handler_precedence_requires_the_same_representation():
    """A saved equation of a different structure (e.g. the converter's expanded grouped-conv
    form, or a renamed graph whose compound ranks were collapsed) is priced as written: the
    handler op rebuilt from the shapes describes another representation."""
    analyzer = EinsumAnalyzer()
    op = analyzer.get_einsum_op("convtranspose3d", _ts())
    assert analyzer._handler_owns_compute_cost(op) is True
    assert analyzer._handler_owns_compute_cost(op, L2_20_EQUATION) is True
    assert analyzer._handler_owns_compute_cost(op, "XY(A-B)(C-D)(E-F),YZBDF->XZACE") is True
    assert analyzer._handler_owns_compute_cost(op, "ACDBE,CLIHJ->ALKFG") is False      # compounds collapsed
    assert analyzer._handler_owns_compute_cost(op, "BGI(P+R)(Q+S),GOIRS->BGOPQ") is False  # another op's form
    assert analyzer.get_compute_cost("convtranspose3d", _ts(), equation="ACDBE,CLIHJ->ALKFG") == \
        compute_cost_from_equation("ACDBE,CLIHJ->ALKFG", _ts())
    assert EinsumAnalyzer._canonical_equation("B0B1MK,B0B1KN->B0B1MN") == EinsumAnalyzer._canonical_equation("PQAB,PQBC->PQAC")


def test_analyzer_without_equation_is_unchanged():
    assert EinsumAnalyzer().get_compute_cost("convtranspose3d", _ts()) == L2_20_MACS


def test_equation_still_drives_ops_without_a_handler_cost():
    """Raw torch.einsum has no handler MAC model: the equation product stays."""
    ts = TensorShapes(inputs=[[8, 256, 512, 256], [256, 768]], outputs=[[8, 256, 512, 768]])
    assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation="BIJL,LK->BIJK") == 8 * 256 * 512 * 256 * 768


def test_equation_is_the_fallback_when_no_handler_op_can_be_built():
    ts = TensorShapes(inputs=[], outputs=[[4, 5]])
    assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation="AB,BC->AC") == 4 * 5
    with pytest.raises(ValueError):
        EinsumAnalyzer().get_compute_cost("no_such_op_xyz", ts)


def test_graph_analyzer_bills_level2_20_convtranspose_with_the_handler(tmp_path: Path):
    graph = {"model_name": "level2_20_convtranspose", "layers": {"Model.conv_transpose3d": L2_20_LAYER}}
    gpath = tmp_path / "einsum_graph_renamed.yaml"
    gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    out = tmp_path / "analysis"
    assert EinsumGraphAnalyzer().analyze_graph(str(gpath), str(out), precision="fp32", copy_graph=False) is not None
    analysis = yaml.safe_load((out / "analysis.yaml").read_text())
    layer = analysis["layers"]["Model.conv_transpose3d"]
    assert layer["macs"] == L2_20_MACS, layer["macs"]
    assert layer["macs_dense"] == L2_20_MACS
    assert analysis["total"]["macs"] == L2_20_MACS
    assert analysis["total"]["flops"] == 2 * L2_20_MACS


def test_sdpa_qk_matmul_keeps_the_equation_cost_f3():
    """F3: the matmul handler does not model its own MACs, so the emitted equation stays decisive."""
    ts = TensorShapes(inputs=[GPT2_QK_INPUT, GPT2_QK_INPUT], outputs=[GPT2_QK_OUTPUT])
    analyzer = EinsumAnalyzer()
    assert type(analyzer.get_einsum_op("matmul", ts)).get_compute_cost is EinsumOp.get_compute_cost
    assert analyzer.get_compute_cost("matmul", ts, equation=GPT2_QK_EQUATION) == GPT2_QK_MACS
    # documents why the equation must win here: the shape-only handler labels under-count
    assert analyzer.get_compute_cost("matmul", ts) == GPT2_QK_HANDLER_ONLY_MACS


def test_graph_analyzer_bills_gpt2_qk_matmul_from_the_equation_f3(tmp_path: Path):
    graph = {"model_name": "gpt2_qk", "layers": {"Model.scaled_dot_product_attention.qk_matmul": GPT2_QK_LAYER}}
    gpath = tmp_path / "einsum_graph_renamed.yaml"
    gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    out = tmp_path / "analysis"
    assert EinsumGraphAnalyzer().analyze_graph(str(gpath), str(out), precision="fp32", copy_graph=False) is not None
    analysis = yaml.safe_load((out / "analysis.yaml").read_text())
    assert analysis["layers"]["Model.scaled_dot_product_attention.qk_matmul"]["macs"] == GPT2_QK_MACS
    assert analysis["total"]["macs"] == GPT2_QK_MACS
