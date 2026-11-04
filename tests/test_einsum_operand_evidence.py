# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-operand einsum evidence: operand cardinality, contradictions, both estimates (F6 / R5).

KernelBench level3/48 Mamba2's SSD einsum ``bclhn,bcshn,bhcls,bcshp->bclhp`` is
saved with only TWO of its four operand shapes (L and X_blocks, positionally
bound to the C_blocks and B_blocks slots: the resolver read C=8, L=2, H=64,
N=64 and then C=2 from X without noticing).  The lineage's cost helper
returned the rank product of that misbound evidence, 549,755,813,888 dense
(279,172,874,240 after triangular density 0.5078125).  With the four recorded
operand shapes the as-written single contraction is 137,438,953,472 dense MACs
and the cheapest ordinary dense binary schedule is 10,737,418,240 MACs plus
134,217,728 scalar multiplies (maintainers' ruling).  Neither is a universal lower bound;
both are exposed, labelled, and ``macs`` keeps the as-written convention.
"""
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer, _einsum_operand_shapes_from_raw
from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.base import (
    compute_cost_from_equation,
    optimal_dense_binary_schedule,
    resolve_equation_extents,
    validate_equation_operands,
)

SSD_EQUATION = "BCLHN,BCSHN,BHCLS,BCSHP->BCLHP"
SSD_RAW = ("[['bclhn,bcshn,bhcls,bcshp->bclhp', Tensor(shape=(2048, 2, 64, 8, 16), dtype=torch.float32), "
           "Tensor(shape=(2048, 2, 64, 8, 16), dtype=torch.float32), Tensor(shape=(2048, 8, 2, 64, 64), dtype=torch.float32), "
           "Tensor(shape=(2048, 2, 64, 8, 64), dtype=torch.float32)], {}]")
C_BLOCKS = [2048, 2, 64, 8, 16]
B_BLOCKS = [2048, 2, 64, 8, 16]
L_MASK = [2048, 8, 2, 64, 64]
X_BLOCKS = [2048, 2, 64, 8, 64]
Y_OUT = [2048, 2, 64, 8, 64]
SSD_SAVED_INPUTS = [L_MASK, X_BLOCKS]            # what einsum_graph_renamed.yaml carries
SSD_ALL_INPUTS = [C_BLOCKS, B_BLOCKS, L_MASK, X_BLOCKS]
# B=2048 C=2 L=64 H=8 N=16 S=64 P=64
SSD_AS_WRITTEN_DENSE = 2048 * 2 * 64 * 8 * 16 * 64 * 64
assert SSD_AS_WRITTEN_DENSE == 137_438_953_472
SSD_MISBOUND_DENSE = 549_755_813_888
SSD_SCHEDULE_MACS = 2_147_483_648 + 8_589_934_592   # C.B over N, then (.L) with X over S
SSD_SCHEDULE_SCALAR = 2048 * 8 * 2 * 64 * 64        # scaling the LS tensor by L
assert SSD_SCHEDULE_MACS == 10_737_418_240 and SSD_SCHEDULE_SCALAR == 134_217_728

SSD_LAYER = {
    "type": "einsum",
    "einsum_equation": SSD_EQUATION,
    "elementwise_op": "mul", "reduction_op": "add",
    "is_real_einsum": True, "is_einsum_supportable": True,
    "tensor_names": {"inputs": ["Model.exp_1.Output", "Model.reshape.Output"], "outputs": ["Model.einsum.Output"]},
    "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
    "tensor_shapes": {"inputs": SSD_SAVED_INPUTS, "outputs": [Y_OUT]},
    "connections": {"inputs": [], "outputs": []},
    "tensor_dtypes": {"inputs": ["torch.float32", "torch.float32"], "outputs": ["torch.float32"]},
    "raw_attributes": SSD_RAW,
}


def _saved_ts() -> TensorShapes:
    return TensorShapes(inputs=SSD_SAVED_INPUTS, outputs=[Y_OUT])


def _full_ts() -> TensorShapes:
    return TensorShapes(inputs=SSD_ALL_INPUTS, outputs=[Y_OUT])


def test_saved_ssd_evidence_is_misbound_and_the_helper_priced_it_anyway():
    problems = validate_equation_operands(SSD_EQUATION, _saved_ts())
    assert any(p.startswith("operand cardinality: 2 input shape(s) for 4") for p in problems), problems
    assert any(p.startswith("rank C: 8 (operand 0) vs 2 (operand 1)") for p in problems), problems
    assert compute_cost_from_equation(SSD_EQUATION, _saved_ts()) == SSD_MISBOUND_DENSE


def test_four_recorded_operands_bind_consistently():
    assert validate_equation_operands(SSD_EQUATION, _full_ts()) == []
    assert compute_cost_from_equation(SSD_EQUATION, _full_ts()) == SSD_AS_WRITTEN_DENSE


def test_bias_slots_and_broadcast_ones_are_not_problems():
    conv = TensorShapes(inputs=[[16, 32, 16, 32, 32], [32, 64, 3, 3, 3], [64]], outputs=[[16, 64, 32, 64, 64]])
    assert validate_equation_operands("BC(P-T)(Q-R)(U-S),CKTRS->BKPQU", conv) == []
    bcast = TensorShapes(inputs=[[4, 1, 6], [4, 5, 6]], outputs=[[4, 5, 6]])
    assert validate_equation_operands("ABC,ABC->ABC", bcast) == []
    assert validate_equation_operands("AB,BC->AC", TensorShapes(inputs=[[4, 5], [6, 7]], outputs=[[4, 7]])) == [
        "rank B: 5 (operand 0) vs 6 (operand 1)"]
    assert validate_equation_operands("ABC,BC->AC", TensorShapes(inputs=[[4, 5], [5, 6]], outputs=[[4, 6]])) == [
        "operand 0: 3 label(s) ABC vs shape [4, 5]"]


def test_raw_attributes_restore_the_positional_operand_shapes():
    assert _einsum_operand_shapes_from_raw(SSD_LAYER, SSD_EQUATION, _saved_ts()) == SSD_ALL_INPUTS
    # saved shapes that do not align to the recorded call are not restored
    misaligned = TensorShapes(inputs=[[7, 7], X_BLOCKS], outputs=[Y_OUT])
    assert _einsum_operand_shapes_from_raw(SSD_LAYER, SSD_EQUATION, misaligned) is None
    # a complete saved layer is left alone
    assert _einsum_operand_shapes_from_raw(SSD_LAYER, SSD_EQUATION, _full_ts()) is None
    assert _einsum_operand_shapes_from_raw({**SSD_LAYER, "raw_attributes": None}, SSD_EQUATION, _saved_ts()) is None


def test_optimal_dense_binary_schedule_for_the_ssd():
    sched = optimal_dense_binary_schedule(SSD_EQUATION, _full_ts())
    assert sched is not None
    assert sched["macs"] == SSD_SCHEDULE_MACS, sched
    assert sched["scalar_multiplies"] == SSD_SCHEDULE_SCALAR, sched
    assert sched["objective_2macs_plus_scalar"] == 2 * SSD_SCHEDULE_MACS + SSD_SCHEDULE_SCALAR
    assert sched["num_operands"] == 4
    # the as-written single contraction is 12.8x the schedule's MACs: both are reported, neither chosen
    assert SSD_AS_WRITTEN_DENSE == 12.8 * SSD_SCHEDULE_MACS
    # misbound evidence yields no schedule
    assert optimal_dense_binary_schedule(SSD_EQUATION, _saved_ts()) is None
    # two operands: no schedule to choose (the single contraction is the only binary tree)
    assert optimal_dense_binary_schedule("AB,BC->AC", TensorShapes(inputs=[[4, 5], [5, 6]], outputs=[[4, 6]])) is None


def test_three_operand_chain_schedule():
    # BCLHN,BHCL,BCLHP->BCHPN (Mamba2 einsum_1): scale by the decay (no sum), then contract over L
    ts = TensorShapes(inputs=[[2048, 2, 64, 8, 16], [2048, 8, 2, 64], [2048, 2, 64, 8, 64]], outputs=[[2048, 2, 8, 64, 16]])
    sched = optimal_dense_binary_schedule("BCLHN,BHCL,BCLHP->BCHPN", ts)
    assert sched["macs"] == 2048 * 2 * 64 * 8 * 16 * 64
    assert sched["scalar_multiplies"] == 2048 * 2 * 64 * 8 * 16
    assert compute_cost_from_equation("BCLHN,BHCL,BCLHP->BCHPN", ts) == 2048 * 2 * 64 * 8 * 16 * 64


def _analyze(tmp_path: Path, layer: dict) -> dict:
    gpath = tmp_path / "einsum_graph_renamed.yaml"
    gpath.write_text(yaml.safe_dump({"model_name": "ssd", "layers": {"Model.einsum": layer}}, sort_keys=False))
    out = tmp_path / "analysis"
    assert EinsumGraphAnalyzer().analyze_graph(str(gpath), str(out), precision="fp32", copy_graph=False) is not None
    return yaml.safe_load((out / "analysis.yaml").read_text())


def test_graph_analyzer_restores_operands_and_reports_both_estimates(tmp_path: Path):
    a = _analyze(tmp_path, SSD_LAYER)
    layer = a["layers"]["Model.einsum"]
    assert layer["operand_shapes_source"].startswith("raw_attributes (positional; saved tensor_shapes carried 2 of 4")
    assert layer["operand_shapes"] == SSD_ALL_INPUTS
    assert "macs_incomplete" not in layer and "operand_evidence_problems" not in layer
    # `macs` follows the analyzer's cost convention for the restored operands: on this
    # head the as-written product; with an earlier fix's torch.einsum handler its pairwise chain.
    convention = EinsumAnalyzer().get_compute_cost("einsum", _full_ts(), equation=SSD_EQUATION)
    assert layer["macs_dense"] == convention
    assert layer["macs"] == convention                       # no structured sparsity in this one-layer graph
    assert layer["macs_as_written_all_index_dense"] == SSD_AS_WRITTEN_DENSE
    assert layer["macs_dense_binary_schedule"] == SSD_SCHEDULE_MACS
    assert layer["scalar_multiplies_dense_binary_schedule"] == SSD_SCHEDULE_SCALAR
    assert "sum over" in layer["binary_schedule"]
    total = a["total"]
    assert total["macs"] == convention
    assert total["macs_dense_binary_schedule"] == SSD_SCHEDULE_MACS
    assert total["scalar_multiplies_dense_binary_schedule"] == SSD_SCHEDULE_SCALAR
    assert total["multi_operand_einsum_layers"] == ["Model.einsum"]
    assert total["macs_incomplete_layers"] == [] and total["operand_evidence_warning_layers"] == []
    assert "status" not in total and "status" not in layer   # complete evidence carries no status marker


def test_graph_analyzer_marks_unrestorable_einsum_incomplete_instead_of_pricing_it(tmp_path: Path):
    layer = {k: v for k, v in SSD_LAYER.items() if k != "raw_attributes"}
    a = _analyze(tmp_path, layer)
    out = a["layers"]["Model.einsum"]
    assert out["macs"] == 0 and out["macs_dense"] == 0
    assert out["macs_incomplete"] is True
    assert any(p.startswith("operand cardinality") for p in out["operand_evidence_problems"])
    assert "macs_dense_binary_schedule" not in out
    assert a["total"]["macs"] == 0
    assert a["total"]["macs_incomplete_layers"] == ["Model.einsum"]
    # INTEG-9 unified status contract (integ9-status-unification/STATUS_CONTRACT.md)
    assert out["status"] == "incomplete" and out["incomplete_reason"] == "einsum_operand_evidence_unresolved"
    assert a["total"]["status"] == "incomplete"
    assert a["total"]["incomplete_layers"] == {"Model.einsum": "einsum_operand_evidence_unresolved"}
    from solar.perf import EinsumGraphPerfModel
    perf = EinsumGraphPerfModel().predict(tmp_path / "analysis" / "analysis.yaml", tmp_path / "perf", arch_config="H100_PCIe", precision="fp32")
    wl = perf["workload"] if isinstance(perf, dict) else yaml.safe_load(next((tmp_path / "perf").glob("*.yaml")).read_text())["workload"]
    assert wl["status"] == "incomplete" and wl["incomplete_layers"] == {"Model.einsum": "einsum_operand_evidence_unresolved"}


def test_handler_emitted_equations_only_warn(tmp_path: Path):
    layer = {
        "type": "matmul", "einsum_equation": "AB,BC->AC", "elementwise_op": "mul", "reduction_op": "add",
        "is_real_einsum": True, "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["x", "w"], "outputs": ["y"]},
        "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
        "tensor_shapes": {"inputs": [[4, 5], [6, 7]], "outputs": [[4, 7]]},
        "connections": {"inputs": [], "outputs": []},
    }
    a = _analyze(tmp_path, layer)
    out = a["layers"]["Model.einsum"]
    assert out["operand_evidence_problems"] == ["rank B: 5 (operand 0) vs 6 (operand 1)"]
    assert "macs_incomplete" not in out
    assert out["macs"] == compute_cost_from_equation("AB,BC->AC", TensorShapes(inputs=[[4, 5], [6, 7]], outputs=[[4, 7]]))
    assert a["total"]["operand_evidence_warning_layers"] == ["Model.einsum"]
    assert a["total"]["macs_incomplete_layers"] == []


# ── P2-2 (an earlier change review): broadcast / diagonal / output validation and real broadcast step sizes ──

def _ts(shapes, out):
    return TensorShapes(inputs=shapes, outputs=[out])


def test_valid_broadcast_resolves_the_non_singleton_extent_and_costs_real_step_sizes():
    ts = _ts([[1], [3], [3]], [3])
    extents, problems = resolve_equation_extents("i,i,i->i", ts)
    assert problems == [] and extents == {"i": 3}
    sched = optimal_dense_binary_schedule("i,i,i->i", ts)
    assert sched["macs"] == 0 and sched["scalar_multiplies"] == 6, sched   # two 3-element Hadamard steps, not 2


def test_singleton_only_intermediates_cost_their_own_broadcast_extent():
    """Round-2 P2-2: [1] x [1] is one element, then x [3] is three: 1 + 3 = 4, not 2 x 3."""
    ts = _ts([[1], [1], [3]], [3])
    assert validate_equation_operands("i,i,i->i", ts) == []
    sched = optimal_dense_binary_schedule("i,i,i->i", ts)
    assert sched["macs"] == 0 and sched["scalar_multiplies"] == 4, sched
    ts2 = _ts([[2, 1], [1, 1], [2, 3]], [2, 3])
    assert validate_equation_operands("ij,ij,ij->ij", ts2) == []
    sched2 = optimal_dense_binary_schedule("ij,ij,ij->ij", ts2)
    assert sched2["macs"] == 0 and sched2["scalar_multiplies"] == 2 + 6, sched2   # [2,1]x[1,1] -> 2 elements, then x[2,3] -> 6


def test_invalid_broadcast_is_refused_and_gets_no_schedule():
    ts = _ts([[1], [2], [3]], [3])
    problems = validate_equation_operands("i,i,i->i", ts)
    assert "rank i: 2 (operand 1) vs 3 (operand 2)" in problems, problems   # (the output [3] vs resolved 2 is reported too)
    assert optimal_dense_binary_schedule("i,i,i->i", ts) is None


def test_diagonal_within_one_operand_requires_equal_extents():
    ts = _ts([[1, 3], [3], [3]], [3])
    problems = validate_equation_operands("ii,i,i->i", ts)
    assert problems == ["rank i: diagonal extents 1 vs 3 within operand 0"], problems
    assert optimal_dense_binary_schedule("ii,i,i->i", ts) is None
    assert validate_equation_operands("ii,i,i->i", _ts([[3, 3], [3], [3]], [3])) == []


def test_output_extent_is_validated_against_the_resolved_extent():
    assert validate_equation_operands("i,i->i", _ts([[1], [3]], [1])) == ["output rank i: 1 vs resolved 3 (operand 1)"]
    assert validate_equation_operands("i,i->i", _ts([[1], [3]], [3])) == []
    assert validate_equation_operands("ij,jk->ik", _ts([[2, 3], [3, 4]], [2, 5])) == ["output rank k: 5 vs resolved 4 (operand 1)"]


# ── P2-1 (an earlier change review): missing and contradictory evidence through conversion -> analysis -> perf ──

def _raw(eq, shapes):
    return "[['" + eq + "', " + ", ".join(f"Tensor(shape={tuple(s)}, dtype=torch.float32)" for s in shapes) + "], {}]"


def _start(name, shape, consumer):
    dims = [f"A{i}" for i in range(len(shape))]
    return {"type": "start", "is_real_einsum": False, "is_einsum_supportable": False, "elementwise_op": "copy", "reduction_op": "none",
            "einsum_equation": "->" + "".join(dims), "operands": {"Output": dims}, "tensor_names": {"inputs": [], "outputs": [name + ".Output"]},
            "tensor_shapes": {"inputs": [], "outputs": [shape]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
            "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]}, "connections": {"inputs": [], "outputs": [consumer]}}


def _convert_einsum_node(shapes, out_shape, raw_attributes):
    """The converter's own operation path (the reviewer's probe shape), no capture."""
    import networkx as nx
    from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    names = [f"s{i}" for i in range(len(shapes))]
    g = nx.DiGraph(); g.add_node("Model.op"); g.add_edges_from([(n, "Model.op") for n in names])
    node = {"type": "einsum", "input_shapes": shapes, "input_types": ["input"] * len(shapes), "input_dtypes": ["torch.float32"] * len(shapes),
            "output_shapes": [out_shape], "output_dtypes": ["torch.float32"], "connections": {"inputs": names, "outputs": []},
            "module_args": {"raw_attributes": raw_attributes}}
    layer = PyTorchToEinsum()._convert_operation("Model.op", node, g, [], {})
    return {"model_name": "probe", "layers": {**{n: _start(n, shapes[i], "Model.op") for i, n in enumerate(names)}, "Model.op": layer}}


def _analysis_and_perf(tmp_path: Path, graph):
    from solar.perf import EinsumGraphPerfModel
    gpath = tmp_path / "einsum_graph_renamed.yaml"; gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
    assert a is not None
    p = EinsumGraphPerfModel().predict(analysis_path=tmp_path / "analysis" / "analysis.yaml", output_dir=tmp_path / "perf", arch_config="H100_PCIe", precision="fp32")
    wl = p["workload"] if isinstance(p, dict) else yaml.safe_load(next((tmp_path / "perf").glob("*.yaml")).read_text())["workload"]
    return a, wl


@pytest.mark.parametrize("case,shapes,raw", [
    ("missing", [[2, 3, 4], [2, 5, 6]], "[['bij,bjk,bkl->bil'], {}]"),                          # 2 of 3 operands, no shapes to restore from
    ("contradictory", [[2, 3, 4], [2, 5, 6]], _raw("bij,bjk->bik", [[2, 3, 4], [2, 5, 6]])),  # j: 4 vs 5
])
def test_unresolved_evidence_is_incomplete_through_conversion_analysis_and_perf(tmp_path: Path, case, shapes, raw):
    graph = _convert_einsum_node(shapes, [2, 3, 6], raw)
    assert graph["layers"]["Model.op"]["type"] == "einsum" and graph["layers"]["Model.op"]["is_real_einsum"]
    a, wl = _analysis_and_perf(tmp_path, graph)
    layer = a["layers"]["Model.op"]
    assert layer["macs"] == 0 and layer["macs_incomplete"] is True
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "einsum_operand_evidence_unresolved"
    assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"] == {"Model.op": "einsum_operand_evidence_unresolved"}
    assert wl["status"] == "incomplete" and wl["incomplete_layers"] == {"Model.op": "einsum_operand_evidence_unresolved"}
    assert wl["prediction_qualification"].startswith("partial")
    assert wl["total_macs"] == 0


def test_complete_evidence_through_conversion_analysis_and_perf_carries_no_status(tmp_path: Path):
    shapes = [[2, 3, 4], [2, 4, 5], [2, 5, 6]]
    graph = _convert_einsum_node(shapes, [2, 3, 6], _raw("bij,bjk,bkl->bil", shapes))
    a, wl = _analysis_and_perf(tmp_path, graph)
    assert "status" not in a["total"] and "status" not in wl and "prediction_qualification" not in wl
    layer = a["layers"]["Model.op"]
    assert layer["macs"] == a["total"]["macs"] == wl["total_macs"] > 0
    # as-written and schedule estimates present and labelled
    assert layer["macs_as_written_all_index_dense"] == 2 * 3 * 4 * 5 * 6
    assert layer["macs_dense_binary_schedule"] == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6


def test_invalid_broadcast_evidence_is_incomplete_end_to_end(tmp_path: Path):
    shapes = [[1], [2], [3]]
    graph = _convert_einsum_node(shapes, [3], _raw("i,i,i->i", shapes))
    a, wl = _analysis_and_perf(tmp_path, graph)
    assert a["layers"]["Model.op"]["status"] == "incomplete" and wl["status"] == "incomplete"
    assert "macs_dense_binary_schedule" not in a["layers"]["Model.op"]


# ── round-2 P3-1: reason-specific prediction qualification ──────────────────────

def _perf_of_total(tmp_path: Path, incomplete_layers):
    from solar.perf import EinsumGraphPerfModel
    layers = {"L": {"type": "matmul", "einsum_equation": "AB,BC->AC", "is_real_einsum": True, "macs": 60, "macs_dense": 60, "other_ops": 0, "flops": 120,
                    "unfused_elements": 47, "fused_elements": 47, "unfused_bytes": 188, "intermediate_bytes": 0, "bytes_per_element": {"inputs": [4, 4], "outputs": [4]},
                    "tensor_shapes": {"inputs": [[3, 4], [4, 5]], "outputs": [[3, 5]]}, "input_elements": 32, "output_elements": 15, "intermediate_elements": 0,
                    "model_io_elements": 47, "input_is_intermediate": False, "output_is_intermediate": False, "is_orphaned": False, "connections": {"inputs": [], "outputs": []}}}
    total = {"num_layers": 1, "num_start_nodes_filtered": 0, "macs": 60, "other_ops": 0, "flops": 120, "unfused_elements": 47, "orojenesis_elements": None, "fused_elements": 47,
             "fused_prefetched_elements": 47, "model_io_elements": 47, "intermediate_elements": 0, "fused_bytes": 188, "fused_prefetched_bytes": 188, "unfused_bytes": 188, "intermediate_bytes": 0,
             "external_input_bytes": 128, "external_output_bytes": 60, "num_intermediate_tensors": 0, "num_orphaned_layers": 0}
    if incomplete_layers is not None:
        total.update({"status": "incomplete", "incomplete_layers": incomplete_layers})
    tmp_path.mkdir(parents=True, exist_ok=True)
    apath = tmp_path / "analysis.yaml"
    apath.write_text(yaml.safe_dump({"layers": layers, "total": total, "metadata": {"precision": "fp32", "bytes_per_element": 4, "bytes_accounting": "per-tensor-dtype"}}, sort_keys=False))
    p = EinsumGraphPerfModel().predict(analysis_path=apath, output_dir=tmp_path / "perf", arch_config="H100_PCIe", precision="fp32")
    return p["workload"] if isinstance(p, dict) else yaml.safe_load(next((tmp_path / "perf").glob("*.yaml")).read_text())["workload"]


def test_prediction_qualification_is_reason_specific(tmp_path: Path):
    zero = _perf_of_total(tmp_path / "z", {"E": "einsum_operand_evidence_unresolved"})
    assert zero["status"] == "incomplete" and zero["prediction_qualification"].startswith("partial: layers with unresolved operand evidence contribute 0 MACs")
    prov = _perf_of_total(tmp_path / "p", {"E": "einsum_operands_unresolved"})
    assert prov["status"] == "incomplete" and prov["prediction_qualification"].startswith("unqualified: incomplete layers are accounted from their known shapes")
    assert "0 MACs" not in prov["prediction_qualification"]
    mixed = _perf_of_total(tmp_path / "m", {"E": "einsum_operand_evidence_unresolved", "F": "norm_params_unattached"})
    assert mixed["prediction_qualification"].startswith("contains partial/unqualified accounting")
    assert "status" not in _perf_of_total(tmp_path / "c", None)
