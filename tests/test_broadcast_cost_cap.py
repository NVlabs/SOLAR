# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic fallback op bills a selection cost: one operation per output element.

For an unhandled op whose rank changes, the analyzer emits ``AB->O0O1O2O3O4``
(is_real_einsum False, elementwise copy, no reduction) and the lineage
snapshot the lineage rewrite costed it with the rank product of ALL labels: KernelBench
level3/48 Mamba2's ``Model.where`` ([64, 64] condition broadcast to
[2048, 8, 2, 64, 64]) was billed 549,755,813,888 other_ops, 4096x its
134,217,728 outputs (Finding 3 / F6-where).  Maintainers' ruling: a defined
selection cost per output element with broadcast alignment, narrowed to the
elementwise/select fallback; reductions, real einsums and handled ops are not
capped.  Reads were already at the actual operand sizes and stay so.
"""
from pathlib import Path

import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.base import (
    ElementwiseSelectEinsumOp,
    EinsumOp,
    EinsumOperand,
    compute_cost_from_equation,
)

MAMBA2_COND = [64, 64]
MAMBA2_OUT = [2048, 8, 2, 64, 64]
MAMBA2_OUT_NUMEL = 2048 * 8 * 2 * 64 * 64
assert MAMBA2_OUT_NUMEL == 134_217_728
MAMBA2_LINEAGE_OTHER_OPS = 549_755_813_888  # = 64 * 64 * output numel

# level3/48 Mamba2ReturnY `Model.where`, as einsum_graph_renamed.yaml saves it.
MAMBA2_WHERE_LAYER = {
    "type": "where",
    "einsum_equation": "AB->O0O1O2O3O4",
    "elementwise_op": "copy",
    "reduction_op": "none",
    "is_real_einsum": False,
    "is_einsum_supportable": True,
    "tensor_names": {"inputs": ["Model.le.Output"], "outputs": ["Model.where.Output"]},
    "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
    "tensor_shapes": {"inputs": [MAMBA2_COND], "outputs": [MAMBA2_OUT]},
    "operands": {"Input": ["A", "B"], "Output": ["O0", "O1", "O2", "O3", "O4"]},
    "connections": {"inputs": [], "outputs": []},
    "tensor_dtypes": {"inputs": ["torch.bool"], "outputs": ["torch.float32"]},
    "taco_expression": "O0[o0,o1,o2,o3,o4] = In0[a,b]",
}


def _fallback(equation: str, in_dims, out_dims, cls=ElementwiseSelectEinsumOp) -> EinsumOp:
    return cls(
        operands=[EinsumOperand("Input", list(in_dims)), EinsumOperand("Output", list(out_dims), is_output=True)],
        equation=equation, name="where", is_real_einsum=False, elementwise_op="copy", reduction_op="none")


OUT5 = ["O0", "O1", "O2", "O3", "O4"]


def test_rank_product_documents_the_defect():
    ts = TensorShapes(inputs=[MAMBA2_COND], outputs=[MAMBA2_OUT])
    assert compute_cost_from_equation("AB->O0O1O2O3O4", ts) == MAMBA2_LINEAGE_OTHER_OPS
    assert MAMBA2_LINEAGE_OTHER_OPS == 64 * 64 * MAMBA2_OUT_NUMEL


def test_broadcast_aligned_selection_costs_one_op_per_output_element():
    ts = TensorShapes(inputs=[MAMBA2_COND], outputs=[MAMBA2_OUT])
    assert _fallback("AB->O0O1O2O3O4", "AB", OUT5).get_compute_cost(ts) == MAMBA2_OUT_NUMEL


def test_same_element_count_view_costs_one_op_per_output_element():
    # an in-place unsqueeze_ reaching the fallback: [1,12,64,64] -> [1,12,1,64,64]
    ts = TensorShapes(inputs=[[1, 12, 64, 64]], outputs=[[1, 12, 1, 64, 64]])
    assert _fallback("ABCD->O0O1O2O3O4", "ABCD", OUT5).get_compute_cost(ts) == 12 * 64 * 64


def test_unaligned_inputs_keep_the_rank_product():
    # neither a broadcast of the output nor the same element count: legacy rule
    ts = TensorShapes(inputs=[[3, 5]], outputs=[[4, 7]])
    op = _fallback("AB->O0O1", "AB", ["O0", "O1"])
    assert op.get_compute_cost(ts) == compute_cost_from_equation("AB->O0O1", ts) == 3 * 5 * 4 * 7


def test_unresolved_output_shape_keeps_the_rank_product():
    op = _fallback("AB->O0O1", "AB", ["O0", "O1"])
    assert op.get_compute_cost(TensorShapes(inputs=[[3, 4]], outputs=[])) == 12
    ts0 = TensorShapes(inputs=[[3, 4]], outputs=[[3, 0]])
    assert op.get_compute_cost(ts0) == compute_cost_from_equation("AB->O0O1", ts0) == 36


def test_handled_ops_and_reductions_are_not_capped():
    """Narrow scope: a plain EinsumOp (handled op) and reductions keep the rank product."""
    ts = TensorShapes(inputs=[MAMBA2_COND], outputs=[MAMBA2_OUT])
    assert _fallback("AB->O0O1O2O3O4", "AB", OUT5, cls=EinsumOp).get_compute_cost(ts) == MAMBA2_LINEAGE_OTHER_OPS
    red = ElementwiseSelectEinsumOp(
        operands=[EinsumOperand("Input", ["A", "B", "C"]), EinsumOperand("Output", ["A", "B"], is_output=True)],
        equation="ABC->AB", name="max", is_real_einsum=False, elementwise_op="copy", reduction_op="max")
    assert red.get_compute_cost(TensorShapes(inputs=[[4, 5, 6]], outputs=[[4, 5]])) == 4 * 5 * 6
    real = EinsumOp(
        operands=[EinsumOperand("A", ["A", "B"]), EinsumOperand("B", ["B", "C"]),
                  EinsumOperand("Output", ["A", "C"], is_output=True)],
        equation="AB,BC->AC", name="matmul")
    assert real.get_compute_cost(TensorShapes(inputs=[[4, 5], [5, 6]], outputs=[[4, 6]])) == 4 * 5 * 6


def test_analyzer_generic_fallback_for_where_is_the_selection_op():
    ts = TensorShapes(inputs=[MAMBA2_COND], outputs=[MAMBA2_OUT])
    analyzer = EinsumAnalyzer()
    op = analyzer.get_einsum_op("where", ts)
    assert isinstance(op, ElementwiseSelectEinsumOp)
    assert op.is_real_einsum is False and op.reduction_op == "none"
    assert analyzer.get_compute_cost("where", ts) == MAMBA2_OUT_NUMEL
    # a fallback is not a handler-provided MAC model: a passed equation still wins for it
    assert analyzer._handler_owns_compute_cost(op) is False


def test_graph_analyzer_bills_mamba2_where_at_output_numel(tmp_path: Path):
    graph = {"model_name": "mamba2_where", "layers": {"Model.where": MAMBA2_WHERE_LAYER}}
    gpath = tmp_path / "einsum_graph_renamed.yaml"
    gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    out = tmp_path / "analysis"
    assert EinsumGraphAnalyzer().analyze_graph(str(gpath), str(out), precision="fp32", copy_graph=False) is not None
    analysis = yaml.safe_load((out / "analysis.yaml").read_text())
    layer = analysis["layers"]["Model.where"]
    assert layer["macs"] == 0
    assert layer["other_ops"] == MAMBA2_OUT_NUMEL, layer["other_ops"]
    assert layer["other_ops"] <= layer["output_elements"]
    # reads/writes at the actual operand sizes: the [64, 64] condition and the output
    assert layer["input_elements"] == 64 * 64
    assert layer["output_elements"] == MAMBA2_OUT_NUMEL
    assert layer["unfused_elements"] == 64 * 64 + MAMBA2_OUT_NUMEL
    assert analysis["total"]["other_ops"] == MAMBA2_OUT_NUMEL


# ── R3 follow-up: the BigBird sidecar's two further fallback shapes ─────────────

BIGBIRD_UNBIND_IN = [12, 60, 64, 512]
BIGBIRD_UNBIND_OUTS = [[60, 64, 512]] * 12
BIGBIRD_UNBIND_NUMEL = 12 * 60 * 64 * 512
BIGBIRD_UNBIND_LINEAGE_OTHER_OPS = 46_385_646_796_800  # rank product of in and out labels


def _multi_out(name: str, equation: str, in_dims, out_dims) -> ElementwiseSelectEinsumOp:
    return ElementwiseSelectEinsumOp(
        operands=[EinsumOperand("Input", list(in_dims)), EinsumOperand("Output", list(out_dims), is_output=True)],
        equation=equation, name=name, is_real_einsum=False, elementwise_op="copy", reduction_op="none")


def test_multi_output_partition_costs_one_op_per_output_element():
    ts = TensorShapes(inputs=[BIGBIRD_UNBIND_IN], outputs=BIGBIRD_UNBIND_OUTS)
    assert compute_cost_from_equation("ABCD->O0O1O2", ts) == BIGBIRD_UNBIND_LINEAGE_OTHER_OPS
    op = _multi_out("unbind", "ABCD->O0O1O2", "ABCD", ["O0", "O1", "O2"])
    assert op.get_compute_cost(ts) == BIGBIRD_UNBIND_NUMEL
    assert EinsumAnalyzer().get_compute_cost("unbind", ts) == BIGBIRD_UNBIND_NUMEL


def test_multi_output_that_is_not_a_partition_keeps_the_rank_product():
    ts = TensorShapes(inputs=[[4, 4]], outputs=[[4, 4], [4, 4], [4, 4]])  # 48 outputs from 16 inputs
    op = _multi_out("foo_op", "AB->O0O1", "AB", ["O0", "O1"])
    assert op.get_compute_cost(ts) == compute_cost_from_equation("AB->O0O1", ts)


def test_factory_fill_costs_one_op_per_output_element():
    # x.new_ones((4, 5, 6)) on a [2, 3] donor: the donor does not align with the output
    ts = TensorShapes(inputs=[[2, 3]], outputs=[[4, 5, 6]])
    op = _multi_out("new_ones", "AB->O0O1O2", "AB", ["O0", "O1", "O2"])
    assert op.get_compute_cost(ts) == 4 * 5 * 6
    assert compute_cost_from_equation("AB->O0O1O2", ts) == 2 * 3 * 4 * 5 * 6
    assert EinsumAnalyzer().get_compute_cost("new_ones", ts) == 4 * 5 * 6
    # a non-factory op with the same unaligned shapes keeps the rank product
    assert _multi_out("foo_op", "AB->O0O1O2", "AB", ["O0", "O1", "O2"]).get_compute_cost(ts) == 2 * 3 * 4 * 5 * 6
