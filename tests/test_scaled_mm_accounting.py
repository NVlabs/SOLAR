# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``torch._scaled_mm`` is a matmul: bill it over the LOGICAL reduction extent.

Finding NVFP4-MECHANICAL-ACCOUNTING-1 (route 195, NVFP4 grouped-query
attention): the capture records ``torch._scaled_mm(A [1024, 4096]
float4_e2m1fn_x2, B [4096, 8192] float4_e2m1fn_x2, scale_a [524288] fp8,
scale_b [4194304] fp8) -> [1024, 8192] bf16`` faithfully, but the engine had
no handler for the op: ``_get_operation_from_name`` matched neither
``matmul`` nor ``mm`` and the unknown-op fallback emitted a one-input copy
(``AB->AB``, ``elementwise_op: copy``, ``is_real_einsum: false``) — zero MACs
for a 1024 x 8192 x 8192 projection, four times per model, marked supportable.

Conventions pinned here (see docs/EINSUM_GUIDE.md "torch._scaled_mm"):

* equation ``MK,KN->MN`` (or ``MK,NK->MN`` when B is recorded ``[N, K]``), a
  real einsum whose handler owns the MAC count;
* MACs = M * K_logical * N with K_logical = K_stored * pack_factor, the pack
  factor taken from the RECORDED dtype of the matrix operands:
  ``float4_e2m1fn_x2`` stores two E2M1 values per byte (the route-195
  reference unpacks weights with ``unpack_e2m1_x2``; the NVFP4 block scales
  recorded next to the operands have numel = M * K_logical / 16), fp8 / bf16
  store one;
* traffic at the recorded (stored) shapes and dtype widths: 1 B per packed FP4
  pair, 1 B per fp8 element, scale tensors read at their own width, the output
  at its dtype; the scales are additional reads and carry no MACs;
* the layer records ``compute_dtype`` (``nvfp4`` / ``fp8``); the perf model
  prices those MACs at ``MAC_per_cycle_<dtype>_tc`` when the architecture
  declares it (Blackwell) and otherwise REFUSES the projection
  (``PerfProjectionUnavailable``, decision 2026-10-26: "for H100 without nvfp4
  raise error saying the perf projection cannot be done") instead of
  inheriting the workload precision's rate or emitting an incomplete-marked
  prediction (H100 declares no FP4 rate).  The analysis is unaffected.
"""
import decimal
import fractions
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.einsum.ops.matmul_ops import ScaledMmHandler, ScaledMmOp
import solar.perf.perf_model as perf_model_module
from solar.perf.perf_model import EinsumGraphPerfModel, PerfProjectionUnavailable, _exact_count, _receipt_safe
from solar.resources import arch_config_path

# The ExecBench result-repricing tool (scripts/reprice_execbench.py) is not part of the public distribution; the tests that
# drive that tool as a subprocess or read its source are skipped when it is absent and run unchanged where it ships.
_REPRICE_TOOL = Path(__file__).resolve().parents[1] / "scripts" / "reprice_execbench.py"
_needs_reprice_tool = pytest.mark.skipif(not _REPRICE_TOOL.exists(), reason="scripts/reprice_execbench.py is not shipped in this distribution")


FP4 = "torch.float4_e2m1fn_x2"
FP8 = "torch.float8_e4m3fn"
BF16 = "torch.bfloat16"
FP32 = "torch.float32"

# route 195 q / o projection: A [1024, 4096] packed FP4 -> logical K = 8192
M, KP, N = 1024, 4096, 8192
K = 2 * KP
BLOCK = 16
Q_MACS = M * K * N                       # 68,719,476,736
KV_MACS = M * K * 1024                   # 8,589,934,592
Q_SCALES = [[M * K // BLOCK], [N * K // BLOCK]]          # [524288], [4194304]
Q_BYTES = M * KP + KP * N + M * K // BLOCK + N * K // BLOCK + M * N * 2   # 59,244,544
Q_ELEMS = M * KP + KP * N + M * K // BLOCK + N * K // BLOCK + M * N       # 50,855,936
ROUTE195_TOTAL = 2 * Q_MACS + 2 * KV_MACS                # 154,618,822,656


def _q_shapes():
    return TensorShapes(inputs=[[M, KP], [KP, N]] + Q_SCALES, outputs=[[M, N]])


# ── handler ────────────────────────────────────────────────────────────────

class TestScaledMmHandler:
    def test_route195_projection_binds_the_logical_reduction_extent(self):
        op = ScaledMmHandler().generate_einsum("_scaled_mm", _q_shapes(), input_dtypes=[FP4, FP4, FP8, FP8])
        assert isinstance(op, ScaledMmOp) and op.is_real_einsum and op.equation == "MK,KN->MN"
        assert (op.elementwise_op, op.reduction_op) == ("mul", "add")
        assert (op.m, op.n, op.k_stored, op.k_logical, op.pack_factor, op.compute_dtype) == (M, N, KP, K, 2, "nvfp4")
        assert [o.name for o in op.operands] == ["Input", "Weight", "Output"]
        assert op.get_compute_cost(_q_shapes()) == Q_MACS == 68_719_476_736
        # the recorded NVFP4 block scales corroborate the logical K: numel = M * K / 16
        assert Q_SCALES[0][0] == M * op.k_logical // BLOCK and Q_SCALES[1][0] == N * op.k_logical // BLOCK

    def test_fp8_per_tensor_scaled_mm_keeps_the_stored_extent(self):
        ts = TensorShapes(inputs=[[64, 32], [32, 80], [1], [1]], outputs=[[64, 80]])
        op = ScaledMmHandler().generate_einsum("_scaled_mm", ts, input_dtypes=[FP8, FP8, FP32, FP32])
        assert (op.k_logical, op.pack_factor, op.compute_dtype) == (32, 1, "fp8")
        assert op.get_compute_cost(ts) == 64 * 32 * 80

    def test_second_operand_recorded_as_n_by_k(self):
        ts = TensorShapes(inputs=[[M, KP], [N, KP]] + Q_SCALES, outputs=[[M, N]])
        op = ScaledMmHandler().generate_einsum("_scaled_mm", ts, input_dtypes=[FP4, FP4, FP8, FP8])
        assert op.equation == "MK,NK->MN" and op.get_compute_cost(ts) == Q_MACS

    @pytest.mark.parametrize("dtypes", [None, [], [FP4], [FP4, None, FP8, FP8], [None, FP4, FP8, FP8]], ids=["none", "empty", "only-A", "B-missing", "A-missing"])
    def test_unrecorded_matrix_dtypes_are_refused_not_defaulted(self, dtypes):
        """The stored shapes alone do not establish K (4096 stored is K=4096 for fp8, 8192 for FP4x2):
        without both recorded matrix dtypes the handler refuses (evidence-1)."""
        with pytest.raises(ValueError, match="dtypes are not recorded"):
            ScaledMmHandler().generate_einsum("_scaled_mm", _q_shapes(), input_dtypes=dtypes)

    def test_bf16_recorded_dtypes_keep_the_stored_extent(self):
        op = ScaledMmHandler().generate_einsum("_scaled_mm", _q_shapes(), input_dtypes=[BF16, BF16, FP32, FP32])
        assert (op.k_logical, op.pack_factor, op.compute_dtype) == (KP, 1, None)
        assert op.get_compute_cost(_q_shapes()) == M * KP * N

    @pytest.mark.parametrize("inputs,outputs,dtypes", [
        ([[M, KP, 1], [KP, N]], [[M, N]], [FP4, FP4]),                 # 3-D matrix operand
        ([[M, KP], [KP + 1, N]], [[M, N]], [FP4, FP4]),                # no shared reduction extent
        ([[M, KP], [KP, N]], [[M, N + 1]], [FP4, FP4]),                # output disagrees
        ([[M, KP], [KP, N]], [[M, N]], [FP4, FP8]),                    # mixed packing
        ([[M, KP]], [[M, N]], [FP4]),                                  # second operand missing
    ])
    def test_refuses_inconsistent_operands_instead_of_guessing(self, inputs, outputs, dtypes):
        with pytest.raises(ValueError):
            ScaledMmHandler().generate_einsum("_scaled_mm", TensorShapes(inputs=inputs, outputs=outputs), input_dtypes=dtypes)

    @pytest.mark.parametrize("name", ["_scaled_mm", "torch._scaled_mm", "_scaled_mm.out", "scaled_mm", "Model._scaled_mm_3"])
    def test_every_recorded_spelling_reaches_the_handler(self, name):
        op = EinsumAnalyzer().get_einsum_op(name, _q_shapes(), input_dtypes=[FP4, FP4, FP8, FP8])
        assert isinstance(op, ScaledMmOp) and op.elementwise_op != "copy"

    def test_handler_cost_takes_precedence_over_the_stored_equation_product(self):
        ts = TensorShapes(inputs=[[M, KP], [KP, N]], outputs=[[M, N]])
        assert EinsumAnalyzer().get_compute_cost("_scaled_mm", ts, equation="MK,KN->MN", input_dtypes=[FP4, FP4]) == Q_MACS
        assert Q_MACS != M * KP * N  # the stored-shape rank product is half the logical count


# ── synthetic einsum graph through analyzer + perf ─────────────────────────

def _start(name, shape, dtype, consumer):
    labels = [f"S{name[-1] if name[-1].isdigit() else 0}{i}" for i in range(len(shape))]
    return {"type": "start", "einsum_equation": "->" + "".join(labels), "elementwise_op": "copy", "reduction_op": "none",
            "is_real_einsum": False, "is_einsum_supportable": False,
            "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
            "tensor_shapes": {"inputs": [], "outputs": [list(shape)]}, "operands": {name: labels},
            "connections": {"inputs": [], "outputs": [consumer]}, "tensor_dtypes": {"inputs": [], "outputs": [dtype]}}


def _scaled_mm_graph(tmp_path, a, b, scales, out, in_dtypes, out_dtype, equation="MK,KN->MN", weight_dims=("K", "N")):
    """One ``scaled_mm`` einsum layer fed by four start nodes (the converter's own layout)."""
    lid = "Model._scaled_mm"
    starts = {f"start{'' if i == 0 else '_' + str(i)}": (shape, dt)
              for i, (shape, dt) in enumerate(zip([a, b] + list(scales), in_dtypes))}
    layers = {s: _start(s, shape, dt, lid) for s, (shape, dt) in starts.items()}
    layers[lid] = {
        "type": "scaled_mm", "einsum_equation": equation, "elementwise_op": "mul", "reduction_op": "add",
        "is_real_einsum": True, "is_einsum_supportable": True,
        "tensor_names": {"inputs": [f"{s}.Output" for s in starts], "outputs": [f"{lid}.Output"]},
        "tensor_types": {"inputs": ["input"] * len(starts), "outputs": ["output"]},
        "tensor_shapes": {"inputs": [list(a), list(b)] + [list(s) for s in scales], "outputs": [list(out)]},
        "operands": {"Input": ["M", "K"], "Weight": list(weight_dims), "Output": ["M", "N"]},
        "connections": {"inputs": list(starts), "outputs": []},
        "tensor_dtypes": {"inputs": list(in_dtypes), "outputs": [out_dtype]},
    }
    graph = {"model_name": "scaled_mm_control", "layers": layers, "model_output_ops": [lid],
             "model_outputs": [{"tensor": f"{lid}.Output", "op": lid, "shape": list(out), "dtype": out_dtype}]}
    path = tmp_path / "einsum_graph.yaml"
    path.write_text(yaml.safe_dump(graph, sort_keys=False))
    analysis = EinsumGraphAnalyzer().analyze_graph(str(path), str(tmp_path / "analysis"), precision="bf16", copy_graph=False)
    assert analysis is not None
    return analysis, analysis["layers"][lid]


def _perf(tmp_path, arch):
    p = EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / f"perf_{arch}"), arch_config=arch, precision="bf16")
    assert p is not None
    return p


class TestSyntheticScaledMm:
    def test_nvfp4_projection_hand_control(self, tmp_path):
        analysis, layer = _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        assert layer["macs"] == Q_MACS and layer["is_real_einsum"] is True and layer.get("status") is None
        assert layer["unfused_bytes"] == Q_BYTES == 59_244_544          # A 4,194,304 + B 33,554,432 + scales 524,288 + 4,194,304 + out 16,777,216
        assert layer["unfused_elements"] == Q_ELEMS
        assert layer["compute_dtype"] == "nvfp4"
        assert layer["scaled_mm"] == {"k_stored": KP, "k_logical": K, "scale_shapes": Q_SCALES, "scales_mac_bearing": False}
        assert "operand_evidence_problems" not in layer                 # the scales are not equation operands
        assert analysis["total"]["macs"] == Q_MACS and analysis["total"]["macs_by_compute_dtype"] == {"nvfp4": Q_MACS}

    def test_fp8_per_tensor_control(self, tmp_path):
        analysis, layer = _scaled_mm_graph(tmp_path, [64, 32], [32, 80], [[1], [1]], [64, 80], [FP8, FP8, FP32, FP32], BF16)
        assert layer["macs"] == 64 * 32 * 80 and layer["compute_dtype"] == "fp8"
        assert layer["unfused_bytes"] == 64 * 32 + 32 * 80 + 4 + 4 + 64 * 80 * 2   # 14,856
        assert layer["scaled_mm"]["k_logical"] == 32 and analysis["total"]["macs_by_compute_dtype"] == {"fp8": 64 * 32 * 80}

    def test_transposed_second_operand_through_the_analyzer(self, tmp_path):
        _, layer = _scaled_mm_graph(tmp_path, [M, KP], [N, KP], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16, equation="MK,NK->MN", weight_dims=("N", "K"))
        assert layer["macs"] == Q_MACS

    def test_h100_has_no_fp4_rate_so_the_projection_is_refused(self, tmp_path):
        """Decision 2026-10-26 (was: an incomplete-marked prediction with ``rate_unavailable:nvfp4``)."""
        analysis, layer = _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        out = tmp_path / "perf_H100_PCIe"
        with pytest.raises(PerfProjectionUnavailable) as info:
            EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(out), arch_config="H100_PCIe", precision="bf16")
        exc = info.value
        assert exc.message == ("perf projection cannot be done: H100_PCIe declares no nvfp4 tensor-core rate "
                               f"(MAC_per_cycle_nvfp4_tc missing from the architecture profile); layers: Model._scaled_mm (nvfp4, {Q_MACS:,} MACs)")
        assert str(exc) == exc.message
        assert (exc.arch, exc.dtype, exc.layers, exc.source) == ("H100_PCIe", "nvfp4", {"Model._scaled_mm": "nvfp4"}, "analysis")
        assert exc.layer_macs == {"Model._scaled_mm": Q_MACS} and exc.macs_affected == Q_MACS and exc.metadata_declared is None
        assert not (out / "perf_H100_PCIe.yaml").exists()                                 # NO perf artifact
        receipt = yaml.safe_load((out / "perf_H100_PCIe.REFUSED.yaml").read_text())
        assert receipt["status"] == "refused" and receipt["reason"] == "perf_projection_unavailable" and receipt["message"] == exc.message
        assert receipt["layers"] == {"Model._scaled_mm": "nvfp4"} and receipt["missing_keys"] == ["MAC_per_cycle_nvfp4_tc"]
        assert receipt["layer_macs"] == {"Model._scaled_mm": Q_MACS} and receipt["macs_affected"] == Q_MACS and receipt["source"] == "analysis"
        # the analysis (MAC / traffic accounting) is unaffected by the refusal
        assert layer["macs"] == Q_MACS and analysis["total"]["macs_by_compute_dtype"] == {"nvfp4": Q_MACS}
        assert yaml.safe_load((tmp_path / "analysis" / "analysis.yaml").read_text())["total"]["macs"] == Q_MACS

    def test_b200_prices_the_nvfp4_rate(self, tmp_path):
        _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        p = _perf(tmp_path, "B200"); w = p["workload"]
        assert w.get("status") is None and "incomplete_layers" not in w
        assert p["arch"]["compute_dtype_rates"]["nvfp4"]["MAC_per_cycle"] == 2_415_000
        assert p["fused"]["compute_tc_cycles"] == int(Q_MACS / 2_415_000) == 28_455   # cycles are reported as integers

    def test_fp8_on_h100_uses_the_fp8_rate(self, tmp_path):
        d = 4096   # large enough for the integer cycle count to be meaningful
        _scaled_mm_graph(tmp_path, [d, d], [d, d], [[1], [1]], [d, d], [FP8, FP8, FP32, FP32], BF16)
        p = _perf(tmp_path, "H100_PCIe"); w = p["workload"]
        assert w.get("status") is None
        assert p["arch"]["compute_dtype_rates"]["fp8"]["MAC_per_cycle"] == 378_000
        assert p["fused"]["compute_tc_cycles"] == int(d ** 3 / 378_000) == 181_797   # not the bf16 rate (363,595)


# ── route-195 capture: the copy lowering must not recur ────────────────────


def test_ci_unit_lane_selects_this_module():
    ci = (Path(__file__).resolve().parents[1] / ".gitlab-ci.yml").read_text()
    assert "tests/test_scaled_mm_accounting.py" in ci


ZEROED_PREFIX = "partial: layers with unresolved operand evidence contribute 0 MACs"


def _assert_zeroed_work_text(text):
    """wording-1: a refused scaled_mm is ZEROED work — the qualification says its MACs are excluded,
    never that it is 'accounted from its known shapes'."""
    assert text.startswith(ZEROED_PREFIX), text
    assert "scaled_mm_evidence_unresolved" in text and "its MACs are excluded" in text and "excluded from the compute cycles" in text, text
    assert "unqualified" not in text and "accounted from their known shapes" not in text, text


def _refused_case(tmp_path, **kw):
    """A scaled_mm layer the handler refuses, through the analyzer and the perf model (both archs)."""
    analysis, layer = _scaled_mm_graph(tmp_path, **kw)
    h = _perf(tmp_path, "H100_PCIe"); b = _perf(tmp_path, "B200")
    return analysis, layer, h, b


class TestRefusalPropagation:
    """evidence-1: a scaled_mm the handler cannot bind is incomplete through layer, total and
    perf — never the stored-shape rank product, never an assumed packing."""

    @pytest.mark.parametrize("kw,fragment", [
        (dict(a=[M, KP], b=[KP, N], scales=Q_SCALES, out=[M, N], in_dtypes=[None, None, FP8, FP8], out_dtype=BF16), "dtypes are not recorded"),
        (dict(a=[M, KP], b=[KP, N], scales=Q_SCALES, out=[M, N], in_dtypes=[FP4, None, FP8, FP8], out_dtype=BF16), "dtypes are not recorded"),
        (dict(a=[M, KP], b=[KP, N], scales=Q_SCALES, out=[M, N], in_dtypes=[FP4, FP8, FP8, FP8], out_dtype=BF16), "disagree on packing"),
        (dict(a=[M, KP], b=[KP, N], scales=Q_SCALES, out=[M, N + 1], in_dtypes=[FP4, FP4, FP8, FP8], out_dtype=BF16), "is not \\[M, N\\]"),
    ], ids=["both-dtypes-missing", "second-dtype-missing", "mixed-packing", "output-contradicts-N"])
    def test_refusal_is_incomplete_through_analysis_total_and_perf(self, tmp_path, kw, fragment):
        import re
        analysis, layer, h, b = _refused_case(tmp_path, **kw)
        assert layer["macs"] == 0 and layer["status"] == "incomplete" and layer["incomplete_reason"] == "scaled_mm_evidence_unresolved"
        assert re.search(fragment, layer["scaled_mm_rejection"]), layer["scaled_mm_rejection"]
        assert "compute_dtype" not in layer and "scaled_mm" not in layer
        assert analysis["total"]["macs"] == 0 and "macs_by_compute_dtype" not in analysis["total"]
        assert analysis["total"]["status"] == "incomplete" and analysis["total"]["incomplete_layers"] == {"Model._scaled_mm": "scaled_mm_evidence_unresolved"}
        assert "Model._scaled_mm" in analysis["total"]["macs_incomplete_layers"]
        for p in (h, b):
            w = p["workload"]
            assert w["status"] == "incomplete" and w["incomplete_layers"] == {"Model._scaled_mm": "scaled_mm_evidence_unresolved"}
            _assert_zeroed_work_text(w["prediction_qualification"])                       # wording-1, H100 and B200
            assert "rate_unavailable" not in w["prediction_qualification"]
            assert p["fused"]["compute_tc_cycles"] == 0 and "compute_dtype_rates" not in p["arch"]

    def test_zeroed_work_wording_is_reason_specific(self):
        """The pure classifier: the scaled-mm refusal is zeroed work alone, with einsum-evidence zeroed work, and
        next to an 'other' reason (mixed), and it never reads as accounted-from-known-shapes."""
        from solar.perf.perf_model import _prediction_qualification, _ZEROED_REASONS
        assert "scaled_mm_evidence_unresolved" in _ZEROED_REASONS
        _assert_zeroed_work_text(_prediction_qualification({"Model._scaled_mm": "scaled_mm_evidence_unresolved"}))
        both = _prediction_qualification({"Model._scaled_mm": "scaled_mm_evidence_unresolved", "Model.einsum": "einsum_operand_evidence_unresolved"})
        _assert_zeroed_work_text(both)
        assert "einsum_operand_evidence_unresolved: operand shapes could not be established" in both
        mixed = _prediction_qualification({"Model._scaled_mm": "scaled_mm_evidence_unresolved", "Model.norm": "weight_provenance_unresolved"})
        assert mixed.startswith("contains partial/unqualified accounting") and "scaled_mm_evidence_unresolved" in mixed
        assert "its MACs are excluded" in mixed and "other incomplete layers are accounted but unqualified" in mixed
        other = _prediction_qualification({"Model.norm": "weight_provenance_unresolved"})
        assert other.startswith("unqualified: incomplete layers are accounted from their known shapes") and "scaled_mm" not in other


class TestComputeDtypeScope:
    """scope-1: compute_dtype belongs to the scaled_mm contract only; an ordinary matmul with
    fp8-recorded inputs keeps the workload precision's rate."""

    def test_ordinary_fp8_mm_keeps_the_workload_rate(self, tmp_path):
        a, b, n = [1024, 4096], [4096, 2048], 2048
        lid = "Model.mm"
        starts = {"start": (a, FP8), "start_1": (b, FP8)}
        layers = {s: _start(s, shape, dt, lid) for s, (shape, dt) in starts.items()}
        layers[lid] = {"type": "mm", "einsum_equation": "MK,KN->MN", "elementwise_op": "mul", "reduction_op": "add", "is_real_einsum": True, "is_einsum_supportable": True,
                       "tensor_names": {"inputs": ["start.Output", "start_1.Output"], "outputs": [f"{lid}.Output"]}, "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
                       "tensor_shapes": {"inputs": [a, b], "outputs": [[1024, n]]}, "operands": {"Input": ["M", "K"], "Weight": ["K", "N"], "Output": ["M", "N"]},
                       "connections": {"inputs": list(starts), "outputs": []}, "tensor_dtypes": {"inputs": [FP8, FP8], "outputs": [BF16]}}
        path = tmp_path / "einsum_graph.yaml"
        path.write_text(yaml.safe_dump({"model_name": "fp8_mm", "layers": layers, "model_output_ops": [lid], "model_outputs": [{"tensor": f"{lid}.Output", "op": lid, "shape": [1024, n], "dtype": BF16}]}, sort_keys=False))
        analysis = EinsumGraphAnalyzer().analyze_graph(str(path), str(tmp_path / "analysis"), precision="bf16", copy_graph=False)
        layer = analysis["layers"][lid]
        assert layer["macs"] == 1024 * 4096 * 2048 == 8_589_934_592 and "compute_dtype" not in layer
        assert "macs_by_compute_dtype" not in analysis["total"]
        p = _perf(tmp_path, "H100_PCIe")
        assert p["arch"]["MAC_per_cycle"] == 189_000 and "compute_dtype_rates" not in p["arch"]
        assert p["fused"]["compute_tc_cycles"] == int(8_589_934_592 / 189_000) == 45_449
        assert p["workload"].get("status") is None


class TestCombinedQualification:
    """qualification-1 after the 2026-10-26 decision: a scaled_mm layer that is already incomplete
    (einsum_operands_unresolved) REFUSES on an architecture without its rate (the refusal precedes every
    qualification) and keeps its own cause, derived once from the combined representation, on B200."""

    def test_existing_reason_refuses_on_h100_and_is_kept_alone_on_b200(self, tmp_path):
        # seed the analyzer's own provenance marker on the layer (the earlier producer)
        analysis, layer = _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        import yaml
        g = yaml.safe_load(open(tmp_path / "einsum_graph.yaml")); g["layers"]["Model._scaled_mm"]["einsum_operands_unresolved"] = ["Weight"]
        (tmp_path / "einsum_graph.yaml").write_text(yaml.safe_dump(g, sort_keys=False))
        analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum_graph.yaml"), str(tmp_path / "analysis"), precision="bf16", copy_graph=False)
        layer = analysis["layers"]["Model._scaled_mm"]
        assert layer["macs"] == Q_MACS and layer["incomplete_reason"] == "einsum_operands_unresolved" and layer["compute_dtype"] == "nvfp4"
        with pytest.raises(PerfProjectionUnavailable) as info:
            _perf(tmp_path, "H100_PCIe")
        assert info.value.layers == {"Model._scaled_mm": "nvfp4"}
        b = _perf(tmp_path, "B200"); wb = b["workload"]
        assert wb["incomplete_layers"] == {"Model._scaled_mm": "einsum_operands_unresolved"} and "rate_unavailable" not in wb["prediction_qualification"]
        assert b["fused"]["compute_tc_cycles"] == int(Q_MACS / 2_415_000)

    def test_no_rate_text_survives_in_any_qualification(self):
        # the rate marker is no longer a qualification: nothing in the wording helper mentions it
        from solar.perf.perf_model import _prediction_qualification
        for reasons in ({"L": "einsum_operands_unresolved"}, {"L": "scaled_mm_evidence_unresolved"}, {"L": "a+scaled_mm_evidence_unresolved"}):
            assert "rate" not in _prediction_qualification(reasons)


# ── the projection refuses without a rate for a declared dtype (decision 2026-10-26) ───────────────

def _mm_layer(lid, a, b, n, dtypes, out_dtype, starts):
    return {"type": "mm", "einsum_equation": "MK,KN->MN", "elementwise_op": "mul", "reduction_op": "add", "is_real_einsum": True, "is_einsum_supportable": True,
            "tensor_names": {"inputs": [f"{s}.Output" for s in starts], "outputs": [f"{lid}.Output"]}, "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
            "tensor_shapes": {"inputs": [a, b], "outputs": [[a[0], n]]}, "operands": {"Input": ["M", "K"], "Weight": ["K", "N"], "Output": ["M", "N"]},
            "connections": {"inputs": list(starts), "outputs": []}, "tensor_dtypes": {"inputs": list(dtypes), "outputs": [out_dtype]}}


def _mixed_graph(tmp_path, with_fp4=True):
    """A bf16 matmul (priced at the workload rate) next to an nvfp4 scaled_mm (no H100 rate)."""
    mm, smm = "Model.mm", "Model._scaled_mm"
    layers = {}
    layers.update({s: _start(s, shape, BF16, mm) for s, shape in {"start": [256, 512], "start_1": [512, 128]}.items()})
    layers[mm] = _mm_layer(mm, [256, 512], [512, 128], 128, [BF16, BF16], BF16, ["start", "start_1"])
    if with_fp4:
        fp4_starts = {"start_2": ([M, KP], FP4), "start_3": ([KP, N], FP4), "start_4": (Q_SCALES[0], FP8), "start_5": (Q_SCALES[1], FP8)}
        layers.update({s: _start(s, shape, dt, smm) for s, (shape, dt) in fp4_starts.items()})
        layers[smm] = {"type": "scaled_mm", "einsum_equation": "MK,KN->MN", "elementwise_op": "mul", "reduction_op": "add", "is_real_einsum": True, "is_einsum_supportable": True,
                       "tensor_names": {"inputs": [f"{s}.Output" for s in fp4_starts], "outputs": [f"{smm}.Output"]}, "tensor_types": {"inputs": ["input"] * 4, "outputs": ["output"]},
                       "tensor_shapes": {"inputs": [[M, KP], [KP, N]] + [list(x) for x in Q_SCALES], "outputs": [[M, N]]},
                       "operands": {"Input": ["M", "K"], "Weight": ["K", "N"], "Output": ["M", "N"]}, "connections": {"inputs": list(fp4_starts), "outputs": []},
                       "tensor_dtypes": {"inputs": [FP4, FP4, FP8, FP8], "outputs": [BF16]}}
    outs = [mm] + ([smm] if with_fp4 else [])
    graph = {"model_name": "mixed", "layers": layers, "model_output_ops": outs,
             "model_outputs": [{"tensor": f"{o}.Output", "op": o, "shape": layers[o]["tensor_shapes"]["outputs"][0], "dtype": BF16} for o in outs]}
    (tmp_path / "einsum_graph.yaml").write_text(yaml.safe_dump(graph, sort_keys=False))
    analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum_graph.yaml"), str(tmp_path / "analysis"), precision="bf16", copy_graph=False)
    assert analysis is not None
    return analysis


class TestProjectionRefusal:
    MM_MACS = 256 * 512 * 128

    def test_mixed_graph_refuses_on_h100_and_names_only_the_fp4_layers(self, tmp_path):
        analysis = _mixed_graph(tmp_path)
        assert analysis["total"]["macs"] == self.MM_MACS + Q_MACS and analysis["total"]["macs_by_compute_dtype"] == {"nvfp4": Q_MACS}
        with pytest.raises(PerfProjectionUnavailable) as info:
            _perf(tmp_path, "H100_PCIe")
        assert info.value.layers == {"Model._scaled_mm": "nvfp4"} and "Model.mm" not in info.value.message
        assert not (tmp_path / "perf_H100_PCIe" / "perf_H100_PCIe.yaml").exists()
        # the whole projection is refused: no partial prediction for the bf16 layer either
        assert sorted(p.name for p in (tmp_path / "perf_H100_PCIe").iterdir()) == ["analysis.yaml", "perf_H100_PCIe.REFUSED.yaml"]
        # B200 prices both layers: the bf16 one at the workload rate, the FP4 one at the nvfp4 rate
        b = _perf(tmp_path, "B200")
        assert b["workload"].get("status") is None
        assert b["fused"]["compute_tc_cycles"] == int(self.MM_MACS / 603_751 + Q_MACS / 2_415_000)

    def test_non_fp4_layer_on_h100_is_priced_as_before(self, tmp_path):
        analysis = _mixed_graph(tmp_path, with_fp4=False)
        assert "macs_by_compute_dtype" not in analysis["total"]
        p = _perf(tmp_path, "H100_PCIe")
        assert p["workload"].get("status") is None and p["arch"]["MAC_per_cycle"] == 189_000
        assert p["fused"]["compute_tc_cycles"] == int(self.MM_MACS / 189_000) and (tmp_path / "perf_H100_PCIe" / "perf_H100_PCIe.yaml").exists()

    def test_metadata_declaration_alone_does_not_refuse_but_is_recorded(self, tmp_path):
        """mac-scope-1: the ``metadata.yaml`` dtype_conversions declaration ("a quantized dtype appears somewhere
        in the inputs/reference") is not compute-dtype evidence.  A bf16 matmul with an nvfp4 sidecar prices on H100
        at the workload precision's rate, the unbound declaration is recorded, and no MAC-rate override is applied;
        B200 keeps the pre-existing override."""
        _mixed_graph(tmp_path, with_fp4=False)
        (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"dtype_conversions": [{"orig_dtypes": "torch.float4_e2m1fn_x2", "new_dtypes": "torch.float32"}]}))
        p = _perf(tmp_path, "H100_PCIe")
        assert p["arch"]["mac_per_cycle_key"] == "MAC_per_cycle_bf16_tc" and p["arch"]["MAC_per_cycle"] == 189_000
        assert p["fused"]["compute_tc_cycles"] == int(self.MM_MACS / 189_000) and p["workload"].get("status") is None
        unbound = p["workload"]["quant_metadata_unbound"]
        assert unbound["dtype"] == "nvfp4" and unbound["rate_in_profile"] is False and unbound["orig_dtypes"] == "torch.float4_e2m1fn_x2"
        assert "no MAC-bearing layer" in unbound["note"] and (tmp_path / "perf_H100_PCIe" / "perf_H100_PCIe.yaml").exists()
        b = _perf(tmp_path, "B200")
        assert b["arch"]["mac_per_cycle_key"] == "MAC_per_cycle_nvfp4_tc" and b["arch"]["MAC_per_cycle"] == 2_415_000
        assert b["workload"]["quant_orig_dtype"] == "torch.float4_e2m1fn_x2" and b["workload"].get("status") is None and "quant_metadata_unbound" not in b["workload"]

    def test_metadata_declaration_rides_along_as_evidence_on_a_layer_bound_refusal(self, tmp_path):
        """layer-evidence-2: with the sidecar present the per-layer scan still runs first and the refusal
        names the MAC-bearing FP4 layer with its MACs; the metadata declaration is identified separately."""
        _mixed_graph(tmp_path)
        (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"dtype_conversions": [{"orig_dtypes": "nvfp4 float4_e2m1fn_x2", "new_dtypes": "traced as-is"}]}))
        with pytest.raises(PerfProjectionUnavailable) as info:
            _perf(tmp_path, "H100_PCIe")
        exc = info.value
        assert exc.source == "analysis" and exc.layers == {"Model._scaled_mm": "nvfp4"} and exc.layer_macs == {"Model._scaled_mm": Q_MACS}
        assert exc.metadata_declared == {"dtype": "nvfp4", "orig_dtypes": "nvfp4 float4_e2m1fn_x2", "rate_in_profile": False, "key": "MAC_per_cycle_nvfp4_tc"}
        assert "workload metadata declares 'nvfp4 float4_e2m1fn_x2'" in exc.message and f"Model._scaled_mm (nvfp4, {Q_MACS:,} MACs)" in exc.message
        receipt = yaml.safe_load((tmp_path / "perf_H100_PCIe" / "perf_H100_PCIe.REFUSED.yaml").read_text())
        assert receipt["layers"] == {"Model._scaled_mm": "nvfp4"} and receipt["metadata_declared"]["dtype"] == "nvfp4"

    def test_metadata_declared_fp8_on_h100_still_prices(self, tmp_path):
        _mixed_graph(tmp_path, with_fp4=False)
        (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"dtype_conversions": [{"orig_dtypes": "torch.float8_e4m3fn", "new_dtypes": "torch.float32"}]}))
        p = _perf(tmp_path, "H100_PCIe")
        assert p["arch"]["mac_per_cycle_key"] == "MAC_per_cycle_fp8_tc" and p["arch"]["MAC_per_cycle"] == 378_000

    def test_stale_prediction_is_quarantined_by_the_refusal(self, tmp_path):
        _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        out = tmp_path / "perf_H100_PCIe"; out.mkdir(); (out / "perf_H100_PCIe.yaml").write_text("stale: true\n")
        with pytest.raises(PerfProjectionUnavailable):
            _perf(tmp_path, "H100_PCIe")
        assert not (out / "perf_H100_PCIe.yaml").exists() and (out / "perf_H100_PCIe.yaml.REFUSED.stale.0").read_text() == "stale: true\n"

    def test_cli_exits_non_zero_with_the_message_and_writes_no_perf_artifact(self, tmp_path):
        import subprocess, sys, os
        _scaled_mm_graph(tmp_path, [M, KP], [KP, N], Q_SCALES, [M, N], [FP4, FP4, FP8, FP8], BF16)
        out = tmp_path / "cli_perf"
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run([sys.executable, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(tmp_path / "analysis" / "analysis.yaml"),
                            "--output-dir", str(out), "--arch-config", "H100_PCIe", "--precision", "bf16"], capture_output=True, text=True, env=env, cwd=str(root))
        assert r.returncode == 3, (r.returncode, r.stdout[-800:], r.stderr[-800:])
        assert "perf projection cannot be done: H100_PCIe declares no nvfp4 tensor-core rate" in r.stdout
        assert not (out / "perf_H100_PCIe.yaml").exists() and (out / "perf_H100_PCIe.REFUSED.yaml").exists()
        r = subprocess.run([sys.executable, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(tmp_path / "analysis" / "analysis.yaml"),
                            "--output-dir", str(tmp_path / "cli_perf_b200"), "--arch-config", "B200", "--precision", "bf16"], capture_output=True, text=True, env=env, cwd=str(root))
        assert r.returncode == 0 and (tmp_path / "cli_perf_b200" / "perf_B200.yaml").exists()


# ── the reviewer's round-1 controls, verbatim (mac-scope-1 / LAYER-EVIDENCE-2 / RECEIPT-LIFECYCLE-3) ────────

CONTROLS = Path(__file__).parent / "fixtures" / "perf_refusal_controls"


def _control(tmp_path, name):
    """Stage one reviewer control: ``<run>/analysis/analysis.yaml`` (+ ``metadata.yaml`` sidecar next to it)."""
    run = tmp_path / name; (run / "analysis").mkdir(parents=True)
    (run / "analysis" / "analysis.yaml").write_bytes((CONTROLS / name / "analysis.yaml").read_bytes())
    if (CONTROLS / name / "metadata.yaml").exists():
        (run / "metadata.yaml").write_bytes((CONTROLS / name / "metadata.yaml").read_bytes())
    return run


def _predict(run, arch, precision="bf16"):
    return EinsumGraphPerfModel(dtype_bytes=True).predict(str(run / "analysis" / "analysis.yaml"), str(run / "perf"), arch_config=arch, precision=precision)


class TestReview4Controls:
    @pytest.mark.parametrize("name,arch", [
        ("fp4_traffic_only_no_metadata", "H100_PCIe"),            # 0 MACs, 16 B, FP4 copy: traffic only
        ("fp4_traffic_only_with_metadata", "H100_PCIe"),          # ... plus the nvfp4 sidecar: still traffic only
        ("zero_mac_dtype_entry", "H100_PCIe"),                    # macs_by_compute_dtype {nvfp4: 0}: zero-MAC bucket
        ("bf16_macs_plus_fp4_traffic_metadata", "H100_PCIe"),     # BF16 MACs + an FP4 copy + the sidecar
        ("fp8_traffic_only_missing_rate", "profile_H100_without_fp8.yaml"),  # fp8 sidecar, 0 MACs, profile without fp8
        ("mixed_b200", "B200"),                                   # B200 declares the rate
        ("then_successful", "H100_PCIe"),
    ])
    def test_controls_that_must_price(self, tmp_path, name, arch):
        run = _control(tmp_path, name)
        arch_arg = str(CONTROLS / arch) if arch.endswith(".yaml") else arch
        p = _predict(run, arch_arg)
        assert p is not None and p["workload"].get("status") is None
        files = sorted(x.name for x in (run / "perf").iterdir())
        assert not any("REFUSED" in f for f in files), files
        before = yaml.safe_load((CONTROLS / name / "analysis.yaml").read_text())
        assert yaml.safe_load((run / "analysis" / "analysis.yaml").read_text()) == before          # analysis untouched
        if name == "zero_mac_dtype_entry":
            assert p["arch"]["compute_dtype_rates"]["nvfp4"] == {"key": "MAC_per_cycle_nvfp4_tc", "MAC_per_cycle": None, "macs": 0.0, "status": "traffic_only_no_rate_needed"}
        if name == "bf16_macs_plus_fp4_traffic_metadata":
            assert p["arch"]["mac_per_cycle_key"] == "MAC_per_cycle_bf16_tc" and p["fused"]["compute_tc_cycles"] == int(1_000_000 / 189_000)
            assert p["workload"]["quant_metadata_unbound"]["dtype"] == "nvfp4"
        if name == "fp8_traffic_only_missing_rate":
            assert p["workload"]["quant_metadata_unbound"]["dtype"] == "fp8" and p["workload"]["total_macs"] == 0
        if name == "mixed_b200":
            assert p["arch"]["compute_dtype_rates"]["nvfp4"]["MAC_per_cycle"] == 2_415_000

    @pytest.mark.parametrize("name", ["mixed_analysis_only", "mixed_analysis_plus_metadata", "first_refused"])
    def test_controls_that_must_refuse_name_the_mac_bearing_layers(self, tmp_path, name):
        run = _control(tmp_path, name)
        with pytest.raises(PerfProjectionUnavailable) as info:
            _predict(run, "H100_PCIe")
        exc = info.value
        assert exc.layers == {"q": "nvfp4", "k": "nvfp4"} and exc.layer_macs == {"q": 1_000_000, "k": 1_000_000} and exc.macs_affected == 2_000_000
        assert exc.source == "analysis" and "bf16" not in exc.layers
        assert (exc.metadata_declared is not None) == (CONTROLS / name / "metadata.yaml").exists()
        receipt = yaml.safe_load((run / "perf" / "perf_H100_PCIe.REFUSED.yaml").read_text())
        assert receipt["layers"] == exc.layers and receipt["layer_macs"] == exc.layer_macs and not (run / "perf" / "perf_H100_PCIe.yaml").exists()

    def test_unbound_mac_bearing_bucket_still_refuses_without_an_invented_layer(self, tmp_path):
        """The total names FP4 MACs that no layer carries (inconsistent analysis): MAC evidence, recorded as unbound."""
        run = _control(tmp_path, "zero_mac_dtype_entry")
        a = yaml.safe_load((run / "analysis" / "analysis.yaml").read_text()); a["total"]["macs_by_compute_dtype"] = {"nvfp4": 5000}; a["total"]["macs"] = 5000
        (run / "analysis" / "analysis.yaml").write_text(yaml.safe_dump(a, sort_keys=False))
        with pytest.raises(PerfProjectionUnavailable) as info:
            _predict(run, "H100_PCIe")
        assert info.value.layers == {} and info.value.unbound_macs == {"nvfp4": 5000} and info.value.source == "analysis_total"
        assert "that no layer carries" in info.value.message and "<total" not in info.value.message

    def test_refusal_then_success_leaves_exactly_one_current_state(self, tmp_path):
        """receipt-lifecycle-3: an FP4 refusal followed by a supported BF16 projection into the same output
        directory retires the receipt to a numbered stale name; a later refusal quarantines the numeric YAML."""
        first = _control(tmp_path, "first_refused"); second = _control(tmp_path, "then_successful")
        out = tmp_path / "shared_perf"
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel(dtype_bytes=True).predict(str(first / "analysis" / "analysis.yaml"), str(out), arch_config="H100_PCIe", precision="bf16")
        assert sorted(x.name for x in out.iterdir()) == ["analysis.yaml", "perf_H100_PCIe.REFUSED.yaml"]
        p = EinsumGraphPerfModel(dtype_bytes=True).predict(str(second / "analysis" / "analysis.yaml"), str(out), arch_config="H100_PCIe", precision="bf16")
        assert p is not None
        assert sorted(x.name for x in out.iterdir()) == ["analysis.yaml", "perf_H100_PCIe.REFUSED.stale.0.yaml", "perf_H100_PCIe.yaml"]
        # refuse again: the numeric prediction is quarantined under a numbered name, the stale receipt is kept
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel(dtype_bytes=True).predict(str(first / "analysis" / "analysis.yaml"), str(out), arch_config="H100_PCIe", precision="bf16")
        assert sorted(x.name for x in out.iterdir()) == ["analysis.yaml", "perf_H100_PCIe.REFUSED.stale.0.yaml", "perf_H100_PCIe.REFUSED.yaml", "perf_H100_PCIe.yaml.REFUSED.stale.0"]
        # and succeed once more: the second receipt is retired as .stale.1, nothing is overwritten
        EinsumGraphPerfModel(dtype_bytes=True).predict(str(second / "analysis" / "analysis.yaml"), str(out), arch_config="H100_PCIe", precision="bf16")
        assert sorted(x.name for x in out.iterdir()) == ["analysis.yaml", "perf_H100_PCIe.REFUSED.stale.0.yaml", "perf_H100_PCIe.REFUSED.stale.1.yaml", "perf_H100_PCIe.yaml", "perf_H100_PCIe.yaml.REFUSED.stale.0"]


class TestRepriceSummary:
    @_needs_reprice_tool
    def test_refusal_rewrites_the_destination_summary_and_exits_3(self, tmp_path):
        """stale-summary-4: a destination sol_summary.json with a prior numeric SOL is quarantined and replaced by
        an explicit refusal record (state REFUSED, numeric fields null); the run continues and exits 3."""
        import subprocess, sys, os
        root = Path(__file__).resolve().parents[1]
        src = tmp_path / "from" / "mixed_problem" / "uuid-0001"; (src / "analysis").mkdir(parents=True)
        (src / "analysis" / "analysis.yaml").write_bytes((CONTROLS / "mixed_analysis_only" / "analysis.yaml").read_bytes())
        (src / "sol_summary.json").write_text(json.dumps({"problem": "mixed_problem", "workload_uuid": "uuid-0001", "precision": "bf16", "inferred_precision": "bf16", "arch": "B200",
                                                           "bytes_accounting": "per-tensor-dtype", "sol_ms": {"fused": 1.5}}))
        ok = tmp_path / "from" / "bf16_problem" / "uuid-0002"; (ok / "analysis").mkdir(parents=True)
        (ok / "analysis" / "analysis.yaml").write_bytes((CONTROLS / "then_successful" / "analysis.yaml").read_bytes())
        (ok / "sol_summary.json").write_text(json.dumps({"problem": "bf16_problem", "workload_uuid": "uuid-0002", "precision": "bf16", "inferred_precision": "bf16", "arch": "B200",
                                                          "bytes_accounting": "per-tensor-dtype", "sol_ms": {"fused": 0.5}}))
        to = tmp_path / "to"; dst = to / "mixed_problem" / "uuid-0001"; dst.mkdir(parents=True)
        prior = {"problem": "mixed_problem", "workload_uuid": "uuid-0001", "arch": "H100_PCIe", "sol_ms": {"unfused": 9.0, "fused": 9.0, "fused_prefetched": 9.0}, "perf_yaml": "old/perf_H100_PCIe.yaml"}
        (dst / "sol_summary.json").write_text(json.dumps(prior))
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, str(root / "scripts" / "reprice_execbench.py"), "--from-root", str(tmp_path / "from"), "--to-root", str(to),
                            "--arch-config", "H100_PCIe", "--precision", "bf16", "--bench-root", str(tmp_path / "nobench")], capture_output=True, text=True, env=env, cwd=str(root))
        assert r.returncode == 3, (r.returncode, r.stdout[-1500:], r.stderr[-1500:])
        assert "[refused] mixed_problem/uuid-000" in r.stdout and "1 refused" in r.stdout
        new = json.loads((dst / "sol_summary.json").read_text())
        assert new["state"] == "REFUSED" and new["arch"] == "H100_PCIe"
        assert new["sol_ms"] == {"unfused": None, "fused": None, "fused_prefetched": None} and new["total_macs"] is None and new["perf_yaml"] is None
        ref = new["refusal"]
        assert ref["reason"] == "perf_projection_unavailable" and ref["missing_rate_keys"] == ["MAC_per_cycle_nvfp4_tc"] and ref["dtypes"] == ["nvfp4"]
        assert ref["layers"] == {"q": "nvfp4", "k": "nvfp4"} and ref["layer_macs"] == {"q": 1_000_000, "k": 1_000_000} and ref["evidence_source"] == "analysis"
        assert "perf projection cannot be done: H100_PCIe declares no nvfp4" in ref["message"]
        rc = ref["receipt"]
        assert rc["path"].endswith("perf_H100_PCIe.REFUSED.yaml") and len(rc["sha256"]) == 64 and rc["analysis_sha256"] and rc["profile"] == "H100_PCIe" and rc["run_utc"]
        assert ref["quarantined_prior_summary"].endswith("sol_summary.stale.0.json")
        assert json.loads((dst / "sol_summary.stale.0.json").read_text()) == prior                          # preserved, not deleted
        assert not (dst / "perf" / "perf_H100_PCIe.yaml").exists() and (dst / "perf" / "perf_H100_PCIe.REFUSED.yaml").exists()
        # the other row was repriced normally in the same run
        other = json.loads((to / "bf16_problem" / "uuid-0002" / "sol_summary.json").read_text())
        assert "state" not in other and other["sol_ms"]["fused"] is not None and other["arch"] == "H100_PCIe"


# ── the owning lane's round-2 fresh controls (P2-1 / P2-2 / P3-1), verbatim except fixture/output paths ──────────────────────
#    (attribution: the "fresh controls" file is the owning lane's, not the reviewer's)

def _fresh(tmp_path, name, data, meta=True, arch="H100_PCIe"):
    run = tmp_path / name; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True); ap.write_text(yaml.safe_dump(data, sort_keys=False))
    if meta: (run / "metadata.yaml").write_bytes((CONTROLS / "mixed_analysis_plus_metadata" / "metadata.yaml").read_bytes())
    h = hashlib.sha256(ap.read_bytes()).hexdigest(); d = {"analysis": data, "arch": arch}
    try:
        p = EinsumGraphPerfModel(dtype_bytes=True).predict(ap, run / "perf", arch_config=arch, precision="bf16"); d.update(state="PRICED", perf=p)
    except PerfProjectionUnavailable as exc: d.update(state="REFUSED", receipt=exc.receipt())
    assert hashlib.sha256(ap.read_bytes()).hexdigest() == h
    return d


def _fresh_analysis(name="mixed_analysis_plus_metadata"):
    return yaml.safe_load((CONTROLS / name / "analysis.yaml").read_text())


class TestRound2FreshControls:
    @pytest.mark.parametrize("meta", [False, True])
    def test_layer_compute_evidence_without_aggregate_bucket(self, tmp_path, meta):
        a = _fresh_analysis(); a["total"].pop("macs_by_compute_dtype")
        d = _fresh(tmp_path, "layer-only-" + str(meta), a, meta)
        assert d["state"] == "REFUSED", "MAC-bearing nvfp4 layers must refuse even when the aggregate bucket is absent"
        assert d["receipt"]["layers"] == {"q": "nvfp4", "k": "nvfp4"}
        assert d["receipt"]["layer_macs"] == {"q": 1_000_000, "k": 1_000_000} and d["receipt"]["macs_affected"] == 2_000_000 and "unbound_macs" not in d["receipt"]

    def test_partially_bound_total_preserves_unbound_remainder(self, tmp_path):
        a = _fresh_analysis(); a["total"]["macs_by_compute_dtype"]["nvfp4"] = 4_000_000; a["total"]["macs"] = 5_000_000; a["total"]["flops"] = 10_000_000
        d = _fresh(tmp_path, "partial-aggregate", a)
        assert d["state"] == "REFUSED"
        assert d["receipt"]["macs_affected"] == 4_000_000
        assert d["receipt"]["unbound_macs"] == {"nvfp4": 2_000_000} and d["receipt"]["layers"] == {"q": "nvfp4", "k": "nvfp4"}
        assert "that no layer carries" in d["receipt"]["message"]

    def test_layer_only_missing_dtype_is_not_hidden_by_other_bucket(self, tmp_path):
        a = _fresh_analysis(); a["layers"]["q"]["compute_dtype"] = "nvfp4"; a["layers"]["k"]["compute_dtype"] = "fp8"; a["total"]["macs_by_compute_dtype"] = {"fp8": 1_000_000}
        d = _fresh(tmp_path, "layer-other-bucket", a)
        assert d["state"] == "REFUSED"
        assert d["receipt"]["layers"] == {"q": "nvfp4"}

    def test_two_missing_rates_aggregate_layers_before_refusal(self, tmp_path):
        a = _fresh_analysis(); a["layers"]["k"]["compute_dtype"] = "fp8"; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 1_000_000, "fp8": 1_000_000}
        d = _fresh(tmp_path, "two-missing-rates", a, arch=str(CONTROLS / "profile_H100_without_fp8.yaml"))
        assert d["state"] == "REFUSED"
        assert d["receipt"]["layers"] == {"q": "nvfp4", "k": "fp8"} and d["receipt"]["macs_affected"] == 2_000_000

    def test_rate_present_disagreement_refuses_with_a_structured_receipt(self, tmp_path):
        """Ruling 2026-10-26: B200 (declares the rate), aggregate 4,000,000 vs layers 2,000,000 — contradiction REFUSES,
        no numeric SOL (round 3 priced from the aggregate with an advisory field; that is withdrawn)."""
        a = _fresh_analysis(); a["total"]["macs_by_compute_dtype"]["nvfp4"] = 4_000_000; a["total"]["macs"] = 5_000_000
        d = _fresh(tmp_path, "b200-disagreement", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED"
        rc = d["receipt"]
        assert rc["reason"] == "conflicting compute-dtype evidence" and rc["missing_keys"] == []
        assert rc["compute_dtype_evidence_disagreement"] == {"nvfp4": {"aggregate_macs": 4_000_000, "layer_macs_bound": 2_000_000, "total": 5_000_000, "layers": ["k", "q"]}}
        assert rc["layers"] == {"q": "nvfp4", "k": "nvfp4"} and rc["total_macs"] == 5_000_000
        assert "conflicting compute-dtype evidence on B200" in rc["message"] and "4,000,000 MACs vs 2,000,000 MACs" in rc["message"]
        assert not (tmp_path / "b200-disagreement" / "perf" / "perf_B200.yaml").exists() and (tmp_path / "b200-disagreement" / "perf" / "perf_B200.REFUSED.yaml").exists()

    @pytest.mark.parametrize("aggregate", [0, 1_000_000, 10_000_000])
    def test_any_contradicting_aggregate_refuses_including_zero_and_overfull(self, tmp_path, aggregate):
        a = _fresh_analysis(); a["total"]["macs_by_compute_dtype"]["nvfp4"] = aggregate
        d = _fresh(tmp_path, f"b200-contradiction-{aggregate}", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "conflicting compute-dtype evidence"
        assert d["receipt"]["compute_dtype_evidence_disagreement"]["nvfp4"]["aggregate_macs"] == aggregate

    def test_non_conserving_split_without_layer_contradiction_refuses(self, tmp_path):
        """buckets exceed the total with NO layer evidence at all (total-only record): non-conserving, refuses."""
        a = _fresh_analysis(); a["layers"] = {"bf16": {"macs": 1_000_000}}; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 10_000_000}; a["total"]["macs"] = 3_000_000
        d = _fresh(tmp_path, "b200-nonconserving", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "non-conserving compute-dtype split"
        assert d["receipt"]["compute_dtype_evidence_disagreement"]["nvfp4"] == {"aggregate_macs": 10_000_000, "layer_macs_bound": 0, "layers": [], "total": 3_000_000, "bucket_sum": 10_000_000}

    def test_absence_is_not_contradiction_total_only_record_prices(self, tmp_path):
        a = _fresh_analysis(); a["layers"] = {"bf16": {"macs": 1_000_000}}; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 2_000_000}
        d = _fresh(tmp_path, "b200-total-only", a, meta=False, arch="B200")
        assert d["state"] == "PRICED" and d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"] == {"key": "MAC_per_cycle_nvfp4_tc", "MAC_per_cycle": 2_415_000.0, "macs": 2_000_000.0, "layer_macs_bound": 0}
        assert d["perf"]["fused"]["compute_tc_cycles"] == int(1_000_000 / 603_751 + 2_000_000 / 2_415_000)

    def test_layers_without_aggregate_price_at_their_own_rate_on_b200(self, tmp_path):
        a = _fresh_analysis(); a["total"].pop("macs_by_compute_dtype")
        d = _fresh(tmp_path, "b200-layer-only", a, meta=False, arch="B200")
        assert d["state"] == "PRICED" and "compute_dtype_evidence_disagreement" not in d["perf"]["workload"]
        assert d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]["macs"] == 2_000_000.0
        assert d["perf"]["fused"]["compute_tc_cycles"] == int(1_000_000 / 603_751 + 2_000_000 / 2_415_000)


class TestRepriceSuccessor:
    @staticmethod
    def _stage(root, name, fixture, arch="B200", state=None, refusal=None):
        source = root / name / "mixed" / "uuid"; (source / "analysis").mkdir(parents=True)
        (source / "analysis" / "analysis.yaml").write_bytes((CONTROLS / fixture / "analysis.yaml").read_bytes())
        d = {"problem": "mixed", "workload_uuid": "uuid", "precision": "bf16", "inferred_precision": "bf16", "arch": arch, "bytes_accounting": "per-tensor-dtype", "sol_ms": {"fused": 12345}}
        if state: d["state"] = state
        if refusal: d["refusal"] = refusal
        (source / "sol_summary.json").write_text(json.dumps(d) + "\n"); return source.parent.parent

    @staticmethod
    def _reprice(tmp_path, src, dst, arch="H100_PCIe"):
        import subprocess, sys, os
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-B", str(root / "scripts" / "reprice_execbench.py"), "--from-root", str(src), "--to-root", str(dst), "--arch-config", arch, "--precision", "bf16", "--bench-root", str(tmp_path / "empty-bench")],
                           capture_output=True, text=True, env=env, cwd=str(root), timeout=120)
        sf = dst / "mixed" / "uuid" / "sol_summary.json"
        return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "summary": json.loads(sf.read_text()) if sf.exists() else None}

    @_needs_reprice_tool
    def test_batch_refusal_identity_and_successor_state(self, tmp_path):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        src = self._stage(tmp_path, "source", "mixed_analysis_only"); first = tmp_path / "first-destination"
        d = self._reprice(tmp_path, src, first)
        assert d["exit_code"] == 3 and d["summary"]["state"] == "REFUSED", (d["stdout"][-500:], d["stderr"][-800:])
        ref = d["summary"]["refusal"]; rc = ref["receipt"]
        assert rc["engine_head"] == subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
        assert rc["analysis_sha256"] == hashlib.sha256((src / "mixed" / "uuid" / "analysis" / "analysis.yaml").read_bytes()).hexdigest()
        assert rc["profile_sha256"] == hashlib.sha256(arch_config_path("H100_PCIe").read_bytes()).hexdigest()
        assert rc["sha256"] == hashlib.sha256(Path(rc["path"]).read_bytes()).hexdigest()
        # the prior numeric summary (sol_ms fused 12345 on B200) is recorded as history on the refusal
        assert d["summary"]["prior_outcomes"][-1]["state"] == "PRICED" and d["summary"]["prior_outcomes"][-1]["sol_ms"] == {"fused": 12345}
        d = self._reprice(tmp_path, first, tmp_path / "supported-destination", arch="B200")
        assert d["exit_code"] == 0 and d["summary"]["sol_ms"]["fused"] is not None, (d["stdout"][-500:], d["stderr"][-800:])
        assert d["summary"].get("state") != "REFUSED", "Successful B200 projection inherited the source H100 REFUSED state"
        assert "refusal" not in d["summary"]
        # ... and the refusal is kept as history, not as the current outcome
        hist = d["summary"]["prior_outcomes"]
        assert hist[-1]["state"] == "REFUSED" and hist[-1]["arch"] == "H100_PCIe" and hist[-1]["refusal"]["missing_rate_keys"] == ["MAC_per_cycle_nvfp4_tc"]
        assert d["summary"]["arch"] == "B200" and d["summary"]["total_macs"] == 3_000_000

    @_needs_reprice_tool
    def test_numeric_output_then_refusal_keeps_the_numeric_history(self, tmp_path):
        src = self._stage(tmp_path, "source", "mixed_analysis_only"); ok = tmp_path / "b200-destination"
        d = self._reprice(tmp_path, src, ok, arch="B200")
        assert d["exit_code"] == 0 and "state" not in d["summary"]
        d = self._reprice(tmp_path, ok, tmp_path / "h100-destination")
        assert d["exit_code"] == 3 and d["summary"]["state"] == "REFUSED" and d["summary"]["sol_ms"]["fused"] is None
        assert d["summary"]["prior_outcomes"][-1]["state"] == "PRICED" and d["summary"]["prior_outcomes"][-1]["arch"] == "B200"


# ── the owning lane's round-3 policy / history probes (test_new.py), verbatim except paths and the engine-head literal (compared to the live HEAD) ──

import subprocess as _sp

def _engine_head():
    return _sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()


class TestRound3PolicyProbes:
    @pytest.mark.parametrize('aggregate', [0, 1_000_000, 4_000_000, 10_000_000])
    def test_supported_disagreement_is_qualified_or_refused(self, tmp_path, aggregate):
        a = _fresh_analysis(); a['total']['macs_by_compute_dtype']['nvfp4'] = aggregate
        if aggregate == 4_000_000: a['total'].update(macs=5_000_000, flops=10_000_000)
        d = _fresh(tmp_path, 'disagreement-' + str(aggregate), a, meta=False, arch='B200')
        if d['state'] == 'REFUSED':
            assert d['receipt']['reason'] == 'conflicting compute-dtype evidence'          # the ruling: refuse, structured receipt
            return
        w = d['perf']['workload']; e = w.get('compute_dtype_evidence_disagreement')
        assert e == {'nvfp4': {'aggregate_macs': aggregate, 'layer_macs_bound': 2_000_000, 'priced': 'aggregate'}}
        # Review recommendation: disagreement must not emit an unqualified current SOL.
        assert w.get('status') == 'incomplete' or w.get('prediction_qualification'), 'aggregate/layer disagreement leaves ordinary unqualified numeric SOL'

    def test_disagreement_cannot_create_negative_compute_cycles(self, tmp_path):
        a = _fresh_analysis(); a['total']['macs_by_compute_dtype']['nvfp4'] = 10_000_000
        d = _fresh(tmp_path, 'negative-default-compute', a, meta=False, arch='B200')
        if d['state'] == 'REFUSED': return
        assert d['perf']['fused']['compute_cycles'] >= 0, 'dtype bucket larger than total subtracts excess work at default rate'

    @staticmethod
    def _current(d, refused):
        summary = d['summary']; assert d['exit_code'] == (3 if refused else 0), (d['stdout'][-400:], d['stderr'][-600:])
        if refused:
            assert summary['state'] == 'REFUSED' and summary['refusal'] and all(x is None for x in summary['sol_ms'].values()) and summary['perf_yaml'] is None
            rec = summary['refusal']['receipt']; assert rec['engine_head'] == _engine_head(); assert rec['sha256'] == hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest()
        else:
            assert 'state' not in summary and 'refusal' not in summary and all(x is not None for x in summary['sol_ms'].values())
            assert Path(summary['perf_yaml']).is_file()
        return summary

    @_needs_reprice_tool
    def test_complete_numeric_refused_numeric_refused_numeric_history(self, tmp_path):
        src = TestRepriceSuccessor._stage(tmp_path, 'lifecycle-source', 'mixed_analysis_only'); sp = src / 'mixed' / 'uuid' / 'sol_summary.json'; seed = json.loads(sp.read_text()); seed.pop('sol_ms'); sp.write_text(json.dumps(seed))
        outputs = []; previous = src
        for index, (arch, refused) in enumerate([('B200', False), ('H100_PCIe', True), ('B200', False), ('H100_PCIe', True), ('B200', False)]):
            dst = tmp_path / ('lifecycle-' + str(index)); d = TestRepriceSuccessor._reprice(tmp_path, previous, dst, arch=arch); summary = self._current(d, refused)
            history = summary.get('prior_outcomes', []); assert len(history) == index
            for i, old in enumerate(outputs):
                h = history[i]; was_refused = old.get('state') == 'REFUSED'; assert h['state'] == ('REFUSED' if was_refused else 'PRICED') and h['arch'] == old['arch']
                assert json.loads(Path(h['summary']).read_text()) == old
                if was_refused:
                    assert h['refusal'] == old['refusal']; rec = h['refusal']['receipt']; assert hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest() == rec['sha256']
                else: assert h['sol_ms'] == old['sol_ms'] and h['perf_yaml'] == old['perf_yaml']
            outputs.append(summary); previous = dst

    @_needs_reprice_tool
    def test_supported_disagreement_survives_batch_qualification(self, tmp_path):
        src = TestRepriceSuccessor._stage(tmp_path, 'disagree-summary-source', 'mixed_analysis_only'); ap = src / 'mixed' / 'uuid' / 'analysis' / 'analysis.yaml'; a = _fresh_analysis(); a['total']['macs_by_compute_dtype']['nvfp4'] = 1_000_000
        ap.write_text(yaml.safe_dump(a, sort_keys=False)); d = TestRepriceSuccessor._reprice(tmp_path, src, tmp_path / 'disagree-summary-output', arch='B200')
        # ruling: the contradiction REFUSES through the batch lifecycle (exit 3, REFUSED summary with the structured receipt)
        s = self._current(d, True)
        assert s.get('state') in ('INCOMPLETE', 'REFUSED') or s.get('compute_dtype_evidence_disagreement') or s.get('fully_qualified') is False, 'summary exports ordinary numeric SOL with no disagreement/qualification field'
        assert s['refusal']['reason'] == 'conflicting compute-dtype evidence' and s['refusal']['compute_dtype_evidence_disagreement']['nvfp4']['aggregate_macs'] == 1_000_000 and s['refusal']['missing_rate_keys'] == []


# ── the reviewer's round-4 controls (test_review2.py), verbatim except helpers (count-coercion-1 / LAYER-ONLY-RECEIPT-1) ──

import math


class TestReview2Controls:
    @pytest.mark.parametrize('case', ['equal', 'layers_only', 'total_only', 'aggregate_only', 'absent_dtype_in_layers'])
    def test_noncontradictory_evidence_prices(self, tmp_path, case):
        a = _fresh_analysis()
        if case == 'layers_only': a['total'].pop('macs_by_compute_dtype')
        if case == 'total_only': a['layers'] = {}; a['total'].pop('macs_by_compute_dtype')
        if case == 'aggregate_only': a['layers'] = {}
        if case == 'absent_dtype_in_layers': a['layers'] = {'bf16': {'macs': 1_000_000}}
        d = _fresh(tmp_path, case, a, meta=False, arch='B200'); assert d['state'] == 'PRICED'
        assert 'compute_dtype_evidence_disagreement' not in d['perf']['workload']
        if case in ['aggregate_only', 'absent_dtype_in_layers']:
            r = d['perf']['arch']['compute_dtype_rates']['nvfp4']; assert r['macs'] == 2_000_000 and r['layer_macs_bound'] == 0

    @pytest.mark.parametrize('name,aggregate', [('integer_plus_one', 2_000_001), ('ulp_above', math.nextafter(2_000_000.0, math.inf)), ('ulp_below', math.nextafter(2_000_000.0, -math.inf)), ('fraction_above', 2_000_000.25), ('fraction_below', 1_999_999.75)])
    def test_explicit_unequal_counts_refuse(self, tmp_path, name, aggregate):
        a = _fresh_analysis(); a['total']['macs_by_compute_dtype']['nvfp4'] = aggregate
        d = _fresh(tmp_path, name, a, meta=False, arch='B200')
        assert d['state'] == 'REFUSED', 'Manager ruling has no truncation waiver for different counts'
        rc = d['receipt']
        if name == 'integer_plus_one':
            assert rc['reason'] == 'conflicting compute-dtype evidence' and rc['compute_dtype_evidence_disagreement']['nvfp4']['aggregate_macs'] == 2_000_001
        else:
            assert rc['reason'] == 'compute_dtype_count_malformed' and rc['malformed_counts'] == {'total.macs_by_compute_dtype.nvfp4': aggregate}   # RAW value, verbatim
            assert repr(aggregate) in rc['message'] and '1,999,999' not in rc['message']

    def test_negative_fractional_default_remainder_refuses(self, tmp_path):
        a = _fresh_analysis(); a['layers'] = {}; a['total']['macs_by_compute_dtype']['nvfp4'] = 3_000_000.25
        d = _fresh(tmp_path, 'negative_fractional_remainder', a, meta=False, arch='B200')
        assert d['state'] == 'REFUSED', 'Manager ruling rejects any negative derived remainder'
        assert d['receipt']['reason'] == 'compute_dtype_count_malformed'

    def test_layer_only_overfull_receipt_retains_evidence(self, tmp_path):
        a = _fresh_analysis(); a['total'].pop('macs_by_compute_dtype'); a['layers']['q']['macs'] = 4_000_000
        d = _fresh(tmp_path, 'layer_only_overfull', a, meta=False, arch='B200'); assert d['state'] == 'REFUSED'
        assert d['receipt'].get('compute_dtype_evidence_disagreement'), 'Non-conserving layers-only evidence disappeared from receipt'
        rc = d['receipt']
        assert rc['reason'] == 'non-conserving compute-dtype split' and rc['dtype'] == 'nvfp4' and rc['source'] == 'analysis'
        assert rc['layers'] == {'q': 'nvfp4', 'k': 'nvfp4'} and rc['layer_macs'] == {'q': 4_000_000, 'k': 1_000_000} and rc['macs_affected'] == 5_000_000
        assert rc['compute_dtype_evidence_disagreement'] == {'nvfp4': {'aggregate': 'absent', 'layer_macs_bound': 5_000_000, 'layers': ['k', 'q'], 'total': 3_000_000, 'bucket_sum': 5_000_000}}
        assert 'aggregate absent vs 5,000,000 MACs in layers k, q (total 3,000,000 MACs)' in rc['message']

    @_needs_reprice_tool
    @pytest.mark.parametrize('case', ['contradiction', 'nonconserving'])
    def test_refusal_successor_batch_lifecycle(self, tmp_path, case):
        src = TestRepriceSuccessor._stage(tmp_path, 'batch-' + case, 'mixed_analysis_only'); ap = src / 'mixed' / 'uuid' / 'analysis' / 'analysis.yaml'; a = _fresh_analysis()
        if case == 'contradiction': a['total']['macs_by_compute_dtype']['nvfp4'] = 1_000_000
        else: a['layers'] = {}; a['total']['macs_by_compute_dtype']['nvfp4'] = 10_000_000
        ap.write_text(yaml.safe_dump(a, sort_keys=False)); dst = tmp_path / ('refused-' + case); d = TestRepriceSuccessor._reprice(tmp_path, src, dst, arch='B200')
        s = d['summary']; assert d['exit_code'] == 3 and s['state'] == 'REFUSED' and s['perf_yaml'] is None
        assert all(v is None for v in s['sol_ms'].values()); assert s['refusal']['compute_dtype_evidence_disagreement']
        receipt = s['refusal']['receipt']; assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest() == receipt['sha256']
        (dst / 'mixed' / 'uuid' / 'analysis' / 'analysis.yaml').write_text(yaml.safe_dump(_fresh_analysis(), sort_keys=False))
        ok = TestRepriceSuccessor._reprice(tmp_path, dst, tmp_path / ('success-' + case), arch='B200'); z = ok['summary']
        assert ok['exit_code'] == 0 and 'state' not in z and 'refusal' not in z and z['sol_ms']['fused'] is not None
        assert z['prior_outcomes'][-1]['state'] == 'REFUSED' and z['prior_outcomes'][-1]['refusal'] == s['refusal']
        assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest() == receipt['sha256']


class TestMalformedCounts:
    @pytest.mark.parametrize('where,raw', [('layer', 1_000_000.5), ('layer', True), ('layer', -1), ('layer', float('nan')), ('layer', float('inf')), ('layer', '1000000'),
                                           ('aggregate', 2_000_000.25), ('aggregate', -2_000_000), ('aggregate', float('inf')), ('total', 3_000_000.5), ('total', -3_000_000)])
    def test_every_malformed_count_refuses_with_the_raw_value(self, tmp_path, where, raw):
        a = _fresh_analysis()
        if where == 'layer': a['layers']['q']['macs'] = raw; key = 'layers.q.macs'
        elif where == 'aggregate': a['total']['macs_by_compute_dtype']['nvfp4'] = raw; key = 'total.macs_by_compute_dtype.nvfp4'
        else: a['total']['macs'] = raw; key = 'total.macs'
        d = _fresh(tmp_path, f"malformed-{where}", a, meta=False, arch='B200')
        assert d['state'] == 'REFUSED' and d['receipt']['reason'] == 'compute_dtype_count_malformed'
        got = d['receipt']['malformed_counts']
        assert list(got) == [key] and (got[key] == raw or (isinstance(raw, float) and math.isnan(raw) and isinstance(got[key], float) and math.isnan(got[key])))
        assert d['receipt']['missing_keys'] == [] and 'nothing is truncated or rounded' in d['receipt']['message']
        assert not (tmp_path / f"malformed-{where}" / 'perf' / 'perf_B200.yaml').exists()

    def test_integral_floats_are_exact_integers_and_price(self, tmp_path):
        a = _fresh_analysis(); a['total']['macs_by_compute_dtype']['nvfp4'] = 2_000_000.0; a['layers']['q']['macs'] = 1_000_000.0
        d = _fresh(tmp_path, 'integral-floats', a, meta=False, arch='B200')
        assert d['state'] == 'PRICED' and d['perf']['arch']['compute_dtype_rates']['nvfp4']['macs'] == 2_000_000.0

    def test_exact_conservation_rejects_one_mac_over(self, tmp_path):
        a = _fresh_analysis(); a['layers'] = {}; a['total']['macs_by_compute_dtype']['nvfp4'] = 3_000_001
        d = _fresh(tmp_path, 'one-over', a, meta=False, arch='B200')
        assert d['state'] == 'REFUSED' and d['receipt']['reason'] == 'non-conserving compute-dtype split'
        a = _fresh_analysis(); a['layers'] = {}; a['total']['macs_by_compute_dtype']['nvfp4'] = 3_000_000
        d = _fresh(tmp_path, 'exactly-full', a, meta=False, arch='B200')
        assert d['state'] == 'PRICED' and d['perf']['fused']['compute_tc_cycles'] == int(3_000_000 / 2_415_000)


def _fresh_typed(tmp_path, name, data, monkeypatch, arch="B200"):
    """Like ``_fresh`` but the loader hands ``predict()`` the in-memory ``data`` (with its
    typed scalars: numpy, Decimal) instead of the YAML text, which cannot carry them —
    the loader-boundary control the review asked for (count-integer-types-1)."""
    run = tmp_path / name; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True)
    ap.write_text(yaml.safe_dump(_fresh_analysis("mixed_analysis_only"), sort_keys=False))
    orig = perf_model_module.yaml_safe_load
    monkeypatch.setattr(perf_model_module, "yaml_safe_load", lambda stream: data if Path(getattr(stream, "name", "")).resolve() == ap.resolve() else orig(stream))
    d = {"arch": arch}
    try:
        p = EinsumGraphPerfModel(dtype_bytes=True).predict(ap, run / "perf", arch_config=arch, precision="bf16", copy_analysis=False); d.update(state="PRICED", perf=p)
    except PerfProjectionUnavailable as exc: d.update(state="REFUSED", receipt=exc.receipt())
    rf = run / "perf" / f"perf_{arch}.REFUSED.yaml"
    d["refusal_file"] = yaml.safe_load(rf.read_text()) if rf.exists() else None; d["refusal_file_bytes"] = rf.stat().st_size if rf.exists() else None
    d["numeric_perf_exists"] = (run / "perf" / f"perf_{arch}.yaml").exists()
    return d


class TestRound6CountBoundary:
    """the review round 5 (count-coercion-1 class, COUNT-INTEGER-TYPES-1, COUNT-EXACT-TOTAL-1):
    ONE validator over EVERY raw count field at the loader boundary, numpy integers
    accepted, Python ints through the conservation decision, receipts never empty."""

    @pytest.mark.parametrize("value,expected", [(2_000_000, 2_000_000), (np.int64(2_000_000), 2_000_000), (np.uint64(2_000_000), 2_000_000), (np.intp(2_000_000), 2_000_000),
                                                (2_000_000.0, 2_000_000), (np.float64(2_000_000.0), 2_000_000), (np.float32(1024.0), 1024), (0, 0), (2**53 + 1, 2**53 + 1), (2**64 + 7, 2**64 + 7),
                                                (True, None), (np.bool_(True), None), (-1, None), (np.int64(-1), None), (2_000_000.25, None), (float("nan"), None), (float("inf"), None), (float("-inf"), None),
                                                ("2000000", None), ("not-a-count", None), (None, None), (decimal.Decimal("2000000"), None), ([2_000_000], None), ({"macs": 1}, None)])
    def test_the_one_validator(self, value, expected):
        got = _exact_count(value)
        assert got == expected and (got is None or type(got) is int)

    @pytest.mark.parametrize("name,value", [("fraction", 3_000_000.25), ("bool", True), ("negative", -1), ("nan", float("nan")), ("inf", float("inf")), ("numeric_str", "3000000"), ("bad_str", "not-a-count"), ("null", None)])
    def test_total_only_malformed_refuses_structurally_with_no_evidence_present(self, tmp_path, name, value):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = value
        d = _fresh(tmp_path, "total-only-" + name, a, meta=False, arch="B200")       # no exception escapes predict(): the helper would propagate anything but the refusal
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed", d
        got = d["receipt"]["malformed_counts"]; assert list(got) == ["total.macs"]
        assert got["total.macs"] == value or (isinstance(value, float) and math.isnan(value) and math.isnan(got["total.macs"]))
        assert d["receipt"]["dtype"] == "<none>" and d["receipt"]["missing_keys"] == []
        assert not (tmp_path / ("total-only-" + name) / "perf" / "perf_B200.yaml").exists()

    @pytest.mark.parametrize("value", ["not-a-count", None, -1, float("nan")])
    def test_malformed_total_with_typed_evidence_present_is_malformed_not_non_conserving(self, tmp_path, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs"] = value
        d = _fresh(tmp_path, "typed-total-" + repr(value), a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed"
        assert list(d["receipt"]["malformed_counts"]) == ["total.macs"] and d["receipt"]["dtype"] == "nvfp4"

    def test_every_layer_is_validated_even_without_a_compute_dtype(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"]["bf16_only"] = {"macs": 1_000_000.5}
        d = _fresh(tmp_path, "plain-layer-fraction", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["malformed_counts"] == {"layers.bf16_only.macs": 1_000_000.5}
        a = _fresh_analysis("mixed_analysis_only"); a["layers"]["q"]["macs"] = "1000000"; a["total"]["macs_by_compute_dtype"]["nvfp4"] = None; a["total"]["macs"] = True
        d = _fresh(tmp_path, "all-three-malformed", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["malformed_counts"] == {"layers.q.macs": "1000000", "total.macs_by_compute_dtype.nvfp4": None, "total.macs": True}

    def test_aggregate_that_is_not_a_mapping_is_malformed_as_a_whole(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs_by_compute_dtype"] = [2_000_000]
        d = _fresh(tmp_path, "aggregate-list", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["malformed_counts"] == {"total.macs_by_compute_dtype": [2_000_000]}

    def test_absent_count_keys_are_not_malformed(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"]["plain"] = {"flops": 2}; a["total"].pop("macs_by_compute_dtype"); a["layers"] = {k: v for k, v in a["layers"].items() if k == "plain"}
        d = _fresh(tmp_path, "absent-keys", a, meta=False, arch="B200")
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000

    @pytest.mark.parametrize("total,bucket,expected", [(2**53 + 3, 2**53 + 4, "REFUSED"), (2**53 + 1, 2**53 + 1, "PRICED"), (2**53 + 1, 2**53 + 2, "REFUSED"), (2**64 + 2**53, 2**64 + 2**53, "PRICED"), (2**64 + 2**53, 2**64 + 2**53 + 1, "REFUSED")],
                             ids=["one-over-2^53+3", "equal-2^53+1", "one-over-2^53+1", "equal-above-2^64", "one-over-above-2^64"])
    def test_exact_integer_conservation_beyond_the_float_mantissa(self, tmp_path, total, bucket, expected):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs"] = total; a["total"]["macs_by_compute_dtype"]["nvfp4"] = bucket
        d = _fresh(tmp_path, f"exact-{total}-{bucket}", a, meta=False, arch="B200")
        assert d["state"] == expected, (total, bucket, d.get("receipt", {}).get("reason"))
        if expected == "REFUSED":
            r = d["receipt"]; e = r["compute_dtype_evidence_disagreement"]["nvfp4"]
            assert r["reason"] == "non-conserving compute-dtype split"
            assert r["total_macs"] == total and e["total"] == total and e["bucket_sum"] == bucket and e["aggregate_macs"] == bucket   # exact ints, not float-rounded
            assert all(type(x) is int for x in (r["total_macs"], e["total"], e["bucket_sum"], e["aggregate_macs"]))
        else:
            assert d["perf"]["workload"]["total_macs"] == total and type(d["perf"]["workload"]["total_macs"]) is int

    def test_layer_contradiction_is_exact_beyond_the_float_mantissa(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"]["q"]["macs"] = 2**53; a["layers"]["k"]["macs"] = 1; a["total"]["macs_by_compute_dtype"]["nvfp4"] = 2**53; a["total"]["macs"] = 2**53 + 1
        d = _fresh(tmp_path, "layer-contradiction-2^53", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "conflicting compute-dtype evidence"
        e = d["receipt"]["compute_dtype_evidence_disagreement"]["nvfp4"]; assert e["aggregate_macs"] == 2**53 and e["layer_macs_bound"] == 2**53 + 1

    @pytest.mark.parametrize("typ", [np.int64, np.uint64, np.intp, np.float64], ids=["int64", "uint64", "intp", "float64"])
    def test_numpy_scalars_at_the_loader_boundary_price(self, tmp_path, monkeypatch, typ):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs_by_compute_dtype"]["nvfp4"] = typ(2_000_000); a["layers"]["q"]["macs"] = typ(1_000_000); a["total"]["macs"] = typ(3_000_000)
        d = _fresh_typed(tmp_path, "np-" + typ.__name__ + str(np.dtype(typ).itemsize), a, monkeypatch)
        assert d["state"] == "PRICED", d
        assert d["perf"]["workload"]["total_macs"] == 3_000_000 and type(d["perf"]["workload"]["total_macs"]) is int
        assert d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]["macs"] == 2_000_000 and d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]["layer_macs_bound"] == 2_000_000

    @pytest.mark.parametrize("raw,shown", [(decimal.Decimal("2000000"), "Decimal('2000000')"), (np.bool_(True), None), (np.float32(0.5), 0.5)], ids=["decimal", "np-bool", "np-float32-fraction"])
    def test_typed_malformed_values_refuse_with_a_complete_receipt_file(self, tmp_path, monkeypatch, raw, shown):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs_by_compute_dtype"]["nvfp4"] = raw
        d = _fresh_typed(tmp_path, "typed-malformed", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed", d
        assert d["refusal_file"] is not None and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"]
        f = d["refusal_file"]; assert f["status"] == "refused" and f["reason"] == "compute_dtype_count_malformed" and list(f["malformed_counts"]) == ["total.macs_by_compute_dtype.nvfp4"]
        got = f["malformed_counts"]["total.macs_by_compute_dtype.nvfp4"]
        if shown is not None: assert got == shown
        else: assert got in (True, repr(raw))          # a numpy bool is recorded as its repr when the dumper cannot carry it natively
        assert "receipt_serialization" not in f         # the receipt-safe form is applied at construction; the writer's fallback was not needed
        assert json.dumps(d["receipt"])                   # the reprice summary writer (JSON) can carry it too

    def test_receipt_writer_never_leaves_an_empty_file(self, tmp_path):
        class Opaque:
            def __repr__(self): return "Opaque<count>"
        exc = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts={"total.macs": Opaque()})
        assert exc.malformed_counts == {"total.macs": "Opaque<count>"} and "Opaque<count>" in exc.message
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel._refuse(tmp_path, "B200", exc)
        text = (tmp_path / "perf_B200.REFUSED.yaml").read_text(); assert text and yaml.safe_load(text)["malformed_counts"] == {"total.macs": "Opaque<count>"}
        # and even a receipt the dumper rejects outright is rendered through the fallback, never truncated
        exc2 = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts={"total.macs": "x"})
        exc2.metadata_declared = {"dtype": np.int64(4)}          # a typed value the safe dumper cannot represent
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel._refuse(tmp_path, "B200", exc2)
        f = yaml.safe_load((tmp_path / "perf_B200.REFUSED.yaml").read_text())
        assert f["receipt_serialization"] == "repr-fallback" and f["metadata_declared"] == {"dtype": 4} and f["reason"] == "compute_dtype_count_malformed"
        assert (tmp_path / "perf_B200.REFUSED.stale.0.yaml").exists()


class TestRound7RealExactness:
    """the review round 6 (count-real-rounding-1, count-null-aggregate-1): no float()
    on a count before validation — integrality on the original object, exact int();
    a PRESENT null aggregate is malformed."""

    @pytest.mark.parametrize("value,expected", [
        (np.longdouble("2000000.0000000001"), None), (np.longdouble("9007199254740993"), 2**53 + 1), (np.longdouble(2) ** 70, 2**70), (np.longdouble(2) ** 70 + np.longdouble(2) ** 10, 2**70 + 2**10),
        (np.longdouble("inf"), None), (np.longdouble("nan"), None), (np.longdouble("-1"), None), (np.longdouble("-0.0"), 0),
        (fractions.Fraction(20000000000000001, 10000000000), None), (fractions.Fraction(3), 3), (fractions.Fraction(-2), None), (fractions.Fraction(2**60, 1), 2**60),
        (np.float16(2.5), None), (np.float16(2048.0), 2048), (np.float32(16777216.0), 16777216), (np.float32("nan"), None), (2.0 ** 70, 2**70), (2000000.0000000002, None)],
        ids=["ld-fraction", "ld-2^53+1", "ld-2^70", "ld-2^70+2^10", "ld-inf", "ld-nan", "ld-neg", "ld-negzero", "frac-fraction", "frac-3", "frac-neg", "frac-2^60", "f16-fraction", "f16-2048", "f32-2^24", "f32-nan", "f64-2^70", "f64-fraction"])
    def test_validator_never_rounds_a_real(self, value, expected):
        if isinstance(value, np.longdouble) and expected is not None and expected > 2**53:
            assert np.finfo(np.longdouble).nmant >= 63, "this control needs an 80-bit long double"
        got = _exact_count(value)
        assert got == expected and (got is None or type(got) is int)

    @pytest.mark.parametrize("field", ["total", "aggregate", "layer"])
    def test_fractional_longdouble_refuses_with_its_full_repr(self, tmp_path, monkeypatch, field):
        raw = np.longdouble("2000000.0000000001"); assert raw != np.floor(raw)
        a = _fresh_analysis("mixed_analysis_only")
        path = {"total": "total.macs", "aggregate": "total.macs_by_compute_dtype.nvfp4", "layer": "layers.q.macs"}[field]
        if field == "total": a["total"]["macs"] = raw
        elif field == "aggregate": a["total"]["macs_by_compute_dtype"]["nvfp4"] = raw
        else: a["layers"]["q"]["macs"] = raw
        d = _fresh_typed(tmp_path, "ld-" + field, a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed", d
        assert d["receipt"]["malformed_counts"] == {path: str(raw)} and "0000000001" in d["refusal_file"]["malformed_counts"][path]   # not rounded to 2000000.0; str(): no NumPy-2 'np.longdouble(...)' prefix

    def test_integral_longdouble_total_keeps_every_mac(self, tmp_path, monkeypatch):
        raw = np.longdouble("9007199254740993"); assert int(raw) == 2**53 + 1
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs"] = raw; a["total"]["macs_by_compute_dtype"]["nvfp4"] = np.longdouble("9007199254740993")
        d = _fresh_typed(tmp_path, "ld-integral", a, monkeypatch)
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 2**53 + 1 and type(d["perf"]["workload"]["total_macs"]) is int
        a["total"]["macs_by_compute_dtype"]["nvfp4"] = np.longdouble("9007199254740994")       # one MAC over, in long double
        d = _fresh_typed(tmp_path, "ld-one-over", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "non-conserving compute-dtype split" and d["receipt"]["compute_dtype_evidence_disagreement"]["nvfp4"]["bucket_sum"] == 2**53 + 2

    def test_fraction_is_never_coerced_before_validation(self, tmp_path, monkeypatch):
        raw = fractions.Fraction(20000000000000001, 10000000000); assert raw.denominator != 1
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = raw
        d = _fresh_typed(tmp_path, "fraction-total", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["malformed_counts"] == {"total.macs": repr(raw)} and d["refusal_file"]["malformed_counts"]["total.macs"] == repr(raw)
        a["total"]["macs"] = fractions.Fraction(3_000_000)
        d = _fresh_typed(tmp_path, "fraction-integral", a, monkeypatch)
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000

    def test_present_null_aggregate_is_malformed_absent_is_allowed(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs_by_compute_dtype"] = None
        d = _fresh(tmp_path, "null-aggregate", a, meta=False, arch="B200")
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed" and d["receipt"]["malformed_counts"] == {"total.macs_by_compute_dtype": None}
        assert not (tmp_path / "null-aggregate" / "perf" / "perf_B200.yaml").exists()
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype")
        d = _fresh(tmp_path, "absent-aggregate", a, meta=False, arch="B200")
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000
        counts, bad = perf_model_module._validate_counts({"total": {"macs": 1, "macs_by_compute_dtype": None}})
        assert bad == {"total.macs_by_compute_dtype": None} and counts["aggregate"] == {}

    def test_absent_layer_macs_key_is_ignored_present_null_refuses(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {"q": {"compute_dtype": "nvfp4"}}; a["total"].pop("macs_by_compute_dtype")
        counts, bad = perf_model_module._validate_counts(a); assert counts["layers"] == {} and not bad
        d = _fresh(tmp_path, "layer-key-absent", a, meta=False, arch="B200"); assert d["state"] == "PRICED" and not d["perf"]["arch"].get("compute_dtype_rates")
        a["layers"] = {"q": {"compute_dtype": "nvfp4", "macs": None}}
        d = _fresh(tmp_path, "layer-null", a, meta=False, arch="B200"); assert d["state"] == "REFUSED" and d["receipt"]["malformed_counts"] == {"layers.q.macs": None}


class _ModRaises(float):
    """A numbers.Real whose numeric protocol raises (the review round 7)."""
    exc_type = RuntimeError
    def __mod__(self, other): raise self.exc_type("numeric protocol raises " + self.exc_type.__name__)


class _CustomError(Exception):
    pass


class TestRound8NumericProtocol:
    """the review round 7 (count-real-exception-1, count-real-serialization-1): any
    numeric-protocol exception is malformed data (process control propagates); receipt scalars are
    exact built-in types; a REFUSED file is always written."""

    @pytest.mark.parametrize("exc_type", [ArithmeticError, TypeError, ValueError, RuntimeError, NotImplementedError, _CustomError, OSError], ids=lambda t: t.__name__)
    def test_raising_numeric_protocol_is_malformed_with_the_exception_recorded(self, tmp_path, monkeypatch, exc_type):
        cls = type("ModRaises" + exc_type.__name__, (_ModRaises,), {"exc_type": exc_type}); raw = cls(3_000_000.0)
        errs = []; assert _exact_count(raw, errors=errs) is None and errs == [f"{exc_type.__name__}: numeric protocol raises {exc_type.__name__}"]
        assert _exact_count(raw) is None                                   # the one-argument form the independent review extracts
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = raw
        d = _fresh_typed(tmp_path, "mod-raises", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed", d
        assert d["receipt"]["malformed_counts"] == {"total.macs": 3_000_000.0} and type(d["receipt"]["malformed_counts"]["total.macs"]) is float
        assert d["receipt"]["malformed_protocol_errors"] == {"total.macs": f"{exc_type.__name__}: numeric protocol raises {exc_type.__name__}"}
        assert exc_type.__name__ in d["receipt"]["message"]
        assert d["refusal_file_bytes"] > 0 and d["refusal_file"]["malformed_protocol_errors"]["total.macs"].startswith(exc_type.__name__) and "receipt_serialization" not in d["refusal_file"]
        assert not d["numeric_perf_exists"]

    @pytest.mark.parametrize("exc_type", [MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit], ids=lambda t: t.__name__)
    def test_process_control_exceptions_propagate(self, exc_type):
        cls = type("ModRaises" + exc_type.__name__, (_ModRaises,), {"exc_type": exc_type})
        with pytest.raises(exc_type):
            _exact_count(cls(3.0))

    def test_raising_protocol_in_every_field_is_reported_per_path(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs"] = _ModRaises(3_000_000.0); a["layers"]["q"]["macs"] = _ModRaises(1_000_000.0); a["total"]["macs_by_compute_dtype"]["nvfp4"] = _ModRaises(2_000_000.0)
        d = _fresh_typed(tmp_path, "mod-raises-all", a, monkeypatch)
        assert d["state"] == "REFUSED" and set(d["receipt"]["malformed_protocol_errors"]) == {"total.macs", "layers.q.macs", "total.macs_by_compute_dtype.nvfp4"}
        assert all(v.startswith("RuntimeError: ") for v in d["receipt"]["malformed_protocol_errors"].values()) and d["refusal_file"]["malformed_counts"] == {"total.macs": 3_000_000.0, "layers.q.macs": 1_000_000.0, "total.macs_by_compute_dtype.nvfp4": 2_000_000.0}

    def test_float_and_int_subtypes_with_a_working_protocol(self, tmp_path, monkeypatch):
        class MyFloat(float): pass
        class MyInt(int): pass
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = MyFloat(3_000_000.0)
        d = _fresh_typed(tmp_path, "myfloat-integral", a, monkeypatch); assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000 and type(d["perf"]["workload"]["total_macs"]) is int
        a["total"]["macs"] = MyInt(3_000_000)
        d = _fresh_typed(tmp_path, "myint", a, monkeypatch); assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000
        a["total"]["macs"] = MyFloat(2.5)
        d = _fresh_typed(tmp_path, "myfloat-fraction", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["refusal_file_bytes"] > 0 and d["refusal_file"]["malformed_counts"] == {"total.macs": 2.5} and type(d["receipt"]["malformed_counts"]["total.macs"]) is float
        assert "receipt_serialization" not in d["refusal_file"]          # normalised at construction; the SafeDumper took it directly

    def test_receipt_safe_yields_exact_builtin_types(self):
        class MyFloat(float): pass
        class MyInt(int): pass
        class MyStr(str): pass
        class Unreprable:
            def __repr__(self): raise RuntimeError("no repr")
        cases = [(MyFloat(2.5), float, 2.5), (MyInt(7), int, 7), (MyStr("x"), str, "x"), (np.float32(0.5), float, 0.5), (np.int64(5), int, 5), (np.uint64(2**63), int, 2**63), (np.float64(1.0), float, 1.0),
                 (np.longdouble("2000000.0000000001"), str, str(np.longdouble("2000000.0000000001"))), (fractions.Fraction(1, 3), str, "Fraction(1, 3)"), (decimal.Decimal("2000000"), str, "Decimal('2000000')"),
                 (np.bool_(True), bool, True), (True, bool, True), (None, type(None), None), (3, int, 3), (2.5, float, 2.5), ("s", str, "s"), (_ModRaises(3.0), float, 3.0)]   # a list: 1.0 and True would collide as dict keys
        for raw, typ, val in cases:
            got = _receipt_safe(raw); assert type(got) is typ and got == val, (repr(raw), got)
        assert _receipt_safe(Unreprable()) == "<unrepresentable Unreprable>"
        assert _receipt_safe({"k": MyFloat(1.5), "l": [MyInt(1), np.int32(2)]}) == {"k": 1.5, "l": [1, 2]} and type(_receipt_safe([np.int32(2)])[0]) is int
        assert _receipt_safe(float("nan")) != _receipt_safe(float("nan")) and _receipt_safe(float("inf")) == float("inf")
        for raw, _, _ in cases: yaml.safe_dump({"v": _receipt_safe(raw)}); json.dumps({"v": _receipt_safe(raw)})

    def test_refusal_file_is_written_even_when_every_yaml_dump_fails(self, tmp_path, monkeypatch):
        exc = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts={"total.macs": _ModRaises(3.0)}, malformed_protocol_errors={"total.macs": "RuntimeError: x"})
        monkeypatch.setattr(perf_model_module.yaml, "dump", lambda *a, **k: (_ for _ in ()).throw(yaml.representer.RepresenterError("always")))
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel._refuse(tmp_path, "B200", exc)
        text = (tmp_path / "perf_B200.REFUSED.yaml").read_text(); assert text
        f = json.loads(text); assert f["receipt_serialization"] == "json-fallback" and f["reason"] == "compute_dtype_count_malformed" and f["malformed_counts"] == {"total.macs": 3.0} and f["malformed_protocol_errors"] == {"total.macs": "RuntimeError: x"}
        assert yaml.safe_load(text)["reason"] == "compute_dtype_count_malformed"     # JSON text is YAML-parseable
        # a non-YAMLError failure of the dumper also reaches the fallbacks
        monkeypatch.setattr(perf_model_module.yaml, "dump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dumper broke")))
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel._refuse(tmp_path, "B200", exc)
        assert json.loads((tmp_path / "perf_B200.REFUSED.yaml").read_text())["receipt_serialization"] == "json-fallback" and (tmp_path / "perf_B200.REFUSED.stale.0.yaml").exists()

    def test_one_shot_representer_fault_still_uses_the_yaml_repr_fallback(self, tmp_path, monkeypatch):
        exc = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts={"total.macs": -1})
        real = perf_model_module.yaml.dump; calls = []
        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 1: raise yaml.representer.RepresenterError("one shot")
            return real(*a, **k)
        monkeypatch.setattr(perf_model_module.yaml, "dump", flaky)
        with pytest.raises(PerfProjectionUnavailable):
            EinsumGraphPerfModel._refuse(tmp_path, "B200", exc)
        f = yaml.safe_load((tmp_path / "perf_B200.REFUSED.yaml").read_text()); assert f["receipt_serialization"] == "repr-fallback" and len(calls) == 2 and f["malformed_counts"] == {"total.macs": -1}

    @_needs_reprice_tool
    def test_reprice_summary_carries_the_protocol_error(self):
        exc = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts={"total.macs": _ModRaises(3.0)}, malformed_protocol_errors={"total.macs": "RuntimeError: x"})
        assert json.loads(json.dumps(exc.receipt()))["malformed_protocol_errors"] == {"total.macs": "RuntimeError: x"}
        src = (Path(__file__).resolve().parents[1] / "scripts" / "reprice_execbench.py").read_text(); assert "malformed_protocol_errors" in src


class TestRound9ReceiptContracts:
    """the review round 8 (receipt-str-subtype-1, conservation-float-order-1,
    protocol-message-memory-1, receipt-cycle-1): receipt scalars are genuine built-in
    strings and message construction never raises; every refusal decision runs on exact ints before
    any float conversion, and the conversion itself is guarded; MemoryError from a protocol
    exception's message propagates; cyclic malformed containers give a compact, readable receipt."""

    def test_sticky_str_subtype_with_failing_repr_is_refused_with_a_plain_str(self, tmp_path, monkeypatch):
        class Sticky(str):
            def __str__(self): return self
        class StickyBadRepr(Sticky):
            def __repr__(self): raise _CustomError("str subtype repr unavailable")
        safe = perf_model_module._receipt_safe(StickyBadRepr("bad"))
        assert type(safe) is str and safe == "bad"
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = StickyBadRepr("bad")
        d = _fresh_typed(tmp_path, "sticky-str", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        assert d["refusal_file"]["malformed_counts"] == {"total.macs": "bad"} and "total.macs = 'bad'" in d["receipt"]["message"]

    def test_message_construction_is_failure_safe(self):
        class Unrenderable:
            def __repr__(self): raise _CustomError("no repr")
        monkey = {"total.macs": Unrenderable()}
        orig = perf_model_module._receipt_safe
        try:
            # make the receipt-safe form of this value an object whose repr fails, so the message
            # formatter's own rendering raises: the message must still build with a placeholder
            perf_model_module._receipt_safe = lambda v, *a, **k: v if isinstance(v, Unrenderable) else orig(v, *a, **k)
            exc = PerfProjectionUnavailable("B200", ["nvfp4"], {}, reason="compute_dtype_count_malformed", malformed_counts=monkey)
        finally:
            perf_model_module._receipt_safe = orig
        assert "total.macs = <unrenderable Unrenderable>" in exc.message

    def test_non_conserving_huge_integers_refuse_before_any_float(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs"] = 10**400; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 10**400 + 1}
        d = _fresh_typed(tmp_path, "huge-nonconserving", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "non-conserving compute-dtype split" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        e = d["refusal_file"]["compute_dtype_evidence_disagreement"]["nvfp4"]
        assert type(e["total"]) is int and e["total"] == 10**400 and e["aggregate_macs"] == 10**400 + 1 and e["bucket_sum"] == 10**400 + 1

    def test_conserving_huge_integers_refuse_at_the_guarded_float_boundary(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs"] = 10**400; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 10**400}
        d = _fresh_typed(tmp_path, "huge-conserving", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "count_not_representable_as_float" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        mc = d["refusal_file"]["malformed_counts"]
        assert mc["total.macs"] == 10**400 and mc["total.macs_by_compute_dtype.nvfp4"] == 10**400 and type(mc["total.macs"]) is int
        assert all(v.startswith("OverflowError") for v in d["refusal_file"]["malformed_protocol_errors"].values())

    def test_memory_error_from_a_protocol_message_propagates(self):
        class MessageMemory(Exception):
            def __str__(self): raise MemoryError("message allocation failed")
        class ModMemoryMessage(float):
            def __mod__(self, other): raise MessageMemory()
        with pytest.raises(MemoryError):
            _exact_count(ModMemoryMessage(3.0), errors=[])

    @pytest.mark.parametrize("kind", ["list", "dict"])
    def test_cyclic_malformed_container_gets_a_compact_readable_receipt(self, tmp_path, monkeypatch, kind):
        raw = [] if kind == "list" else {}
        (raw.append(raw) if kind == "list" else raw.__setitem__("self", raw))
        safe = perf_model_module._receipt_safe(raw)
        assert "<cycle: " in json.dumps(safe)
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype"); a["total"]["macs"] = raw
        d = _fresh_typed(tmp_path, "cycle-" + kind, a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "compute_dtype_count_malformed" and not d["numeric_perf_exists"], d
        assert 0 < d["refusal_file_bytes"] < 16384                    # compact, not a half-megabyte expansion
        rf = tmp_path / "cycle-" + kind if False else tmp_path / ("cycle-" + kind) / "perf" / "perf_B200.REFUSED.yaml"
        assert yaml.safe_load(rf.read_text())["reason"] == "compute_dtype_count_malformed"   # default pure-Python reader
        assert perf_model_module.yaml_safe_load(rf.open())["reason"] == "compute_dtype_count_malformed"   # SOLAR's loader

    def test_deep_nesting_is_truncated_not_expanded(self):
        deep = cur = []
        for _ in range(200):
            nxt = []; cur.append(nxt); cur = nxt
        safe = perf_model_module._receipt_safe(deep)
        assert "<depth 32 truncated" in json.dumps(safe)


class TestRound10FloatBoundary:
    """the review round 9 (conservation-float-order-1 PARTIAL, total-float-boundary-1,
    receipt-repr-subtype-1): NO raw count is converted to float before every exact-int refusal
    decision; afterwards EVERY count that enters float arithmetic (the MAC total, buckets, remainder,
    FLOPs, other_ops, element / byte totals, declared boundary bytes, derived byte products) goes through
    the one guarded ``to_float`` layer, which refuses with the exact integers on OverflowError / inf / nan;
    ``repr()`` results are normalised to the exact built-in ``str``."""

    HUGE = 10**400

    @staticmethod
    def _refused_unrepresentable(d, offending):
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "count_not_representable_as_float", d.get("receipt", d)
        assert d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"]
        mc = d["refusal_file"]["malformed_counts"]
        assert type(mc["total.macs"]) is int and mc["total.macs"] == 3_000_000, mc        # exact ints retained for the MAC evidence
        assert offending in mc and offending in d["refusal_file"]["malformed_protocol_errors"], (offending, sorted(mc))
        # round 13: the field carries the ORIGINAL object's own protocol error from the exact-count validator --
        # OverflowError for +-inf / a huge int, ValueError for NaN (int(nan)); the reason is unchanged
        assert d["refusal_file"]["malformed_protocol_errors"][offending].startswith(("OverflowError", "ValueError"))
        return mc

    def test_huge_flops_never_convert_before_the_conservation_refusal(self, tmp_path, monkeypatch):
        # the review round 9: float(total.flops) used to raise before the non-conserving split could refuse
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}
        a["total"]["macs"] = self.HUGE; a["total"]["flops"] = 2 * self.HUGE; a["total"]["macs_by_compute_dtype"] = {"nvfp4": self.HUGE + 1}
        d = _fresh_typed(tmp_path, "huge-flops-nonconserving", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "non-conserving compute-dtype split" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        e = d["refusal_file"]["compute_dtype_evidence_disagreement"]["nvfp4"]
        assert type(e["total"]) is int and e["total"] == self.HUGE and e["aggregate_macs"] == self.HUGE + 1

    def test_huge_flops_on_a_conserving_split_refuse_at_the_layer_with_exact_ints(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}
        a["total"]["macs"] = self.HUGE; a["total"]["flops"] = 2 * self.HUGE; a["total"]["macs_by_compute_dtype"] = {"nvfp4": self.HUGE}
        d = _fresh_typed(tmp_path, "huge-flops-conserving", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "count_not_representable_as_float" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        mc = d["refusal_file"]["malformed_counts"]; assert type(mc["total.macs"]) is int and mc["total.macs"] == self.HUGE

    def test_wide_partition_total_refuses_at_the_total_conversion(self, tmp_path, monkeypatch):
        # the review round 9 total-float-boundary-1: buckets are float-representable, the TOTAL is not
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}
        a["total"]["macs"] = 2 * 10**308; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 10**308, "fp8": 10**308}
        d = _fresh_typed(tmp_path, "wide-partition", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["receipt"]["reason"] == "count_not_representable_as_float" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d
        mc = d["refusal_file"]["malformed_counts"]
        assert type(mc["total.macs"]) is int and mc["total.macs"] == 2 * 10**308 and mc["total.macs_by_compute_dtype.fp8"] == 10**308
        assert "total.macs" in d["refusal_file"]["malformed_protocol_errors"]

    @pytest.mark.parametrize("site, mutate", [
        ("total.flops", lambda t: t.__setitem__("flops", 10**400)),
        ("total.other_ops", lambda t: t.__setitem__("other_ops", 10**400)),
        ("total.unfused_elements", lambda t: t.__setitem__("unfused_elements", 10**400)),
        ("total.fused_elements", lambda t: t.__setitem__("fused_elements", 10**400)),
        ("total.fused_prefetched_elements", lambda t: t.__setitem__("fused_prefetched_elements", 10**400)),
        ("total.weight_elements", lambda t: t.__setitem__("weight_elements", 10**400)),
        ("total.model_io_elements", lambda t: t.__setitem__("model_io_elements", 10**400)),
        ("total.intermediate_elements", lambda t: t.__setitem__("intermediate_elements", 10**400)),
        ("total.orojenesis_bytes", lambda t: (t.pop("unfused_elements"), t.__setitem__("orojenesis_bytes", 10**400))),
        ("total.weight_bytes", lambda t: (t.pop("unfused_elements"), t.__setitem__("weight_bytes", 10**400))),
        ("total.model_io_bytes_declared", lambda t: t.__setitem__("model_io_bytes_declared", 10**400)),
        ("total.boundary_declared.elements_raw", lambda t: (t.__setitem__("model_io_bytes_declared", 100), t.__setitem__("boundary_declared", {"elements_raw": 10**400}))),
        ("total.boundary_declared.bytes_footprint_weight", lambda t: (t.__setitem__("model_io_bytes_declared", 100), t.__setitem__("boundary_declared", {"elements_raw": 4, "footprint_elements_weight": 2, "bytes_footprint_weight": 10**400}))),
        ("total.external_io_bytes_declared_dedup", lambda t: (t.__setitem__("model_io_bytes_declared", 100), t.__setitem__("external_io_bytes_declared_dedup", 10**400))),
        ("total.unfused_bytes", lambda t: t.__setitem__("unfused_bytes", 10**400)),                 # dtype_bytes block (per-tensor-dtype fixture)
        ("total.fused_bytes", lambda t: t.__setitem__("fused_bytes", 10**400)),
        ("total.intermediate_bytes", lambda t: t.__setitem__("intermediate_bytes", 10**400)),
        ("total.external_input_bytes", lambda t: t.__setitem__("external_input_bytes", 10**400)),
        ("total.external_input_bytes + total.external_output_bytes", lambda t: (t.__setitem__("external_input_bytes", 10**308), t.__setitem__("external_output_bytes", 10**308))),
        ("total.weight_elements_unresolved", lambda t: t.__setitem__("weight_elements_unresolved", 10**400)),
        ("unfused_bytes (elements x width)", lambda t: t.__setitem__("unfused_elements", 10**308)),   # float-representable count, product with the width is inf
    ])
    def test_every_count_site_refuses_through_the_one_layer(self, tmp_path, monkeypatch, site, mutate):
        a = _fresh_analysis("mixed_analysis_only"); mutate(a["total"])
        d = _fresh_typed(tmp_path, "site-" + site.replace("/", "_").replace(" ", "_")[:40], a, monkeypatch)
        mc = self._refused_unrepresentable(d, site)
        raw = mc[site]
        assert (type(raw) is int and raw >= 10**308) or raw == float("inf"), raw    # the offending value verbatim (exact int) or the inf it produced

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"])
    def test_non_finite_float_counts_refuse_not_propagate(self, tmp_path, monkeypatch, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["flops"] = value
        d = _fresh_typed(tmp_path, "nonfinite-flops", a, monkeypatch)
        self._refused_unrepresentable(d, "total.flops")

    def test_representable_counts_still_price_identically(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs"] = 2**53 + 1; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 2**53 + 1}; a["layers"] = {}
        d = _fresh_typed(tmp_path, "exact-price", a, monkeypatch)
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 2**53 + 1 and d["perf"]["workload"]["total_flops"] == 6_000_000
        assert d["perf"]["fused"]["arithmetic_intensity"] == float(2**53 + 1) / 16.0

    def test_repr_returning_a_str_subclass_is_normalised_to_builtin_str(self):
        class Sticky(str):
            def __str__(self): return self
            def __repr__(self): raise _CustomError("sticky repr")
        class ReprSubtype:
            def __repr__(self): return Sticky("opaque-value")
        safe = perf_model_module._receipt_safe(ReprSubtype())
        assert type(safe) is str and safe == "opaque-value"
        class ReprNotStr:
            def __repr__(self): return 42          # a broken __repr__ (TypeError in repr()) -> placeholder, still an exact str
        safe2 = perf_model_module._receipt_safe(ReprNotStr())
        assert type(safe2) is str and safe2.startswith("<unrepresentable ReprNotStr")


def _run_arch(tmp_path, name, data, arch="B200", arch_changes=None, monkeypatch=None):
    """predict() on an in-memory analysis against a PRIVATE copy of an architecture profile with ``arch_changes``
    applied (the supported explicit-YAML path); returns state / receipt / disk receipt / numeric-file presence."""
    run = tmp_path / name; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True)
    ap.write_text(yaml.safe_dump(_fresh_analysis("mixed_analysis_only") if monkeypatch is not None else data, sort_keys=False))   # typed objects cannot be dumped; the loader hands them over
    cfg = arch
    if arch_changes is not None:
        src = arch_config_path(arch)   # the shipped profile, wherever this solar is installed (checkout, sdist or wheel)
        prof = yaml.safe_load(src.read_text()); prof.update(arch_changes)
        cfg = run / "architecture.yaml"; cfg.write_text(yaml.safe_dump(prof, sort_keys=False))
    if monkeypatch is not None:   # typed loader: hand predict() the in-memory objects (nan / inf / huge ints survive)
        orig = perf_model_module.yaml_safe_load
        monkeypatch.setattr(perf_model_module, "yaml_safe_load", lambda stream: data if Path(getattr(stream, "name", "")).resolve() == ap.resolve() else orig(stream))
    d = {"arch": arch}
    try:
        p = EinsumGraphPerfModel(dtype_bytes=True).predict(ap, run / "perf", arch_config=str(cfg), precision="bf16", copy_analysis=False); d.update(state="PRICED", perf=p)
    except PerfProjectionUnavailable as exc: d.update(state="REFUSED", receipt=exc.receipt(), reason=exc.reason)
    rf = run / "perf" / f"perf_{arch}.REFUSED.yaml"
    d["refusal_file"] = yaml.safe_load(rf.read_text()) if rf.exists() else None; d["refusal_file_bytes"] = rf.stat().st_size if rf.exists() else None
    d["numeric_perf_exists"] = (run / "perf" / f"perf_{arch}.yaml").exists()
    return d


class TestRound11RateAndDerivedBoundary:
    """the review round 10 (unresolved-weight-boundary-1, rate-float-boundary-1): the unresolved weight
    count is validated as an exact integer before its first cast and its byte product is checked; every architecture
    rate / width is validated (finite, > 0) through the refusal path with a `rate_invalid:<key>` receipt AFTER the exact
    refusals and BEFORE any accepted count is consumed; every derived quantity (cycles, intensity, seconds, ratios)
    is checked and a non-finite one refuses -- a numeric projection is never written from a non-finite rate or value."""

    @staticmethod
    def _refused(d):
        assert d["state"] == "REFUSED" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d.get("receipt", d)
        return d["refusal_file"]

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 2.5, -4, "many", True], ids=["inf", "-inf", "nan", "fractional", "negative", "str", "bool"])
    def test_unresolved_weight_original_value_is_validated_before_any_cast(self, tmp_path, monkeypatch, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["weight_elements_unresolved"] = value
        disk = self._refused(_run_arch(tmp_path, "w-" + str(type(value).__name__), a, monkeypatch=monkeypatch))
        assert "total.weight_elements_unresolved" in disk["malformed_counts"] and type(disk["malformed_counts"]["total.macs"]) is int

    @pytest.mark.parametrize("power, expect", [(307, "PRICED"), (308, "REFUSED"), (400, "REFUSED")])
    def test_unresolved_weight_product_is_checked_before_its_cast(self, tmp_path, monkeypatch, power, expect):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["weight_elements_unresolved"] = 10**power
        d = _run_arch(tmp_path, f"w-10e{power}", a, monkeypatch=monkeypatch)
        if expect == "PRICED":
            assert d["state"] == "PRICED" and d["perf"]["memory_breakdown"]["weight_elements_unresolved"] == 10**307   # exact, never through float
            assert math.isfinite(float(d["perf"]["memory_breakdown"]["weight_bytes_unresolved"]))
        else:
            disk = self._refused(d); assert disk["reason"] == "count_not_representable_as_float"
            name = "total.weight_elements_unresolved" if power == 400 else "weight_bytes_unresolved (elements x width)"
            assert name in disk["malformed_protocol_errors"] and disk["malformed_counts"]["total.macs"] == 3_000_000

    @pytest.mark.parametrize("key", ["MAC_per_cycle_nvfp4_tc", "MAC_per_cycle_bf16_tc", "freq_GHz", "DRAM_byte_per_cycle", "MAC_per_cycle_fp32_sm"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400, -1.0, "fast", True], ids=["nan", "inf", "huge", "negative", "str", "bool"])
    def test_every_architecture_rate_is_validated_with_a_receipt(self, tmp_path, key, value):
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(_run_arch(tmp_path, f"rate-{key}", a, arch_changes={key: value}))
        assert disk["reason"] == f"rate_invalid:{key}" and disk["invalid_rate"]["name"] == key and disk["source"] == "architecture_profile"
        assert disk["invalid_rate"]["profile"] == "B200" and "malformed_counts" not in disk     # a profile defect, not a count defect

    @pytest.mark.parametrize("key", ["MAC_per_cycle_nvfp4_tc", "MAC_per_cycle_bf16_tc", "freq_GHz", "DRAM_byte_per_cycle"])
    def test_zero_is_not_a_divisor_rate_but_the_sm_rate_may_be_zero(self, tmp_path, key):
        a = _fresh_analysis("mixed_analysis_only")
        # round 13 (rate-protocol-coverage-1): a DECLARED rate is only ever asked `is None`; a declared zero -- for the
        # compute-dtype key as for every other key -- is admitted through the rate guard and refused as an invalid rate
        # (it used to be read as "no rate" through the truth protocol: `if dt_rate`)
        disk = self._refused(_run_arch(tmp_path, f"zero-{key}", a, arch_changes={key: 0})); assert disk["reason"] == f"rate_invalid:{key}"
        d = _run_arch(tmp_path, "zero-sm", a, arch_changes={"MAC_per_cycle_fp32_sm": 0})
        assert d["state"] == "PRICED" and d["perf"]["unfused"]["compute_sm_cycles"] == 0

    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), 10**400], ids=["nan", "inf", "huge"])
    def test_exact_evidence_refusals_precede_the_rate_refusal(self, tmp_path, rate):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs_by_compute_dtype"] = {"nvfp4": a["total"]["macs"] + 1}
        disk = self._refused(_run_arch(tmp_path, "nonconserving-badrate", a, arch_changes={"MAC_per_cycle_nvfp4_tc": rate}))
        assert disk["reason"] == "non-conserving compute-dtype split" and disk["compute_dtype_evidence_disagreement"]["nvfp4"]["total"] == 3_000_000

    def test_infinite_rate_never_writes_a_numeric_projection(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(_run_arch(tmp_path, "inf-rate", a, arch_changes={"MAC_per_cycle_nvfp4_tc": float("inf")}))
        assert disk["reason"] == "rate_invalid:MAC_per_cycle_nvfp4_tc"

    def test_finite_rate_arithmetic_overflow_refuses_as_a_derived_quantity(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"]["macs"] = 10**308; a["total"]["macs_by_compute_dtype"] = {"nvfp4": 10**308}; a["total"]["flops"] = 0
        disk = self._refused(_run_arch(tmp_path, "tiny-rate", a, arch_changes={"MAC_per_cycle_nvfp4_tc": 1e-308}, monkeypatch=monkeypatch))
        assert disk["reason"] == "derived_quantity_not_finite" and disk["derived_not_finite"]["name"].startswith("compute_tc_cycles")
        assert disk["derived_not_finite"]["inputs"]["MAC_per_cycle_nvfp4_tc"] == 1e-308 and disk["malformed_counts"]["total.macs"] == 10**308

    @pytest.mark.parametrize("field, changes, derived", [
        ("unfused_bytes", {"DRAM_byte_per_cycle": 1e-308}, "unfused.memory_cycles"),
        ("freq", {"freq_GHz": 1e-320}, "runtime_ms"),
    ])
    def test_memory_cycles_and_runtime_overflow_refuse(self, tmp_path, monkeypatch, field, changes, derived):
        a = _fresh_analysis("mixed_analysis_only")
        if field == "unfused_bytes":
            a["total"]["unfused_bytes"] = 10**308
        else:
            a["total"]["unfused_bytes"] = 1e300
        disk = self._refused(_run_arch(tmp_path, "derived-" + field, a, arch_changes=changes, monkeypatch=monkeypatch))
        assert disk["reason"] == "derived_quantity_not_finite" and derived in disk["derived_not_finite"]["name"]

    def test_zero_byte_path_keeps_the_documented_intensity_sentinel(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["fused_bytes"] = 0; a["total"]["fused_elements"] = 0
        d = _run_arch(tmp_path, "zero-bytes", a)
        assert d["state"] == "PRICED" and d["perf"]["fused"]["arithmetic_intensity"] == float("inf") and d["perf"]["fused"]["memory_cycles"] == 0

    def test_representable_inputs_price_identically_to_round_10(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only")
        d = _run_arch(tmp_path, "baseline", a)
        assert d["state"] == "PRICED" and d["perf"]["workload"]["total_macs"] == 3_000_000 and d["perf"]["arch"]["MAC_per_cycle"] == 603751.0
        assert d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]["MAC_per_cycle"] == 2415000.0 and type(d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]["macs"]) is float


class TestCIFail26NumpyScalarNormalisation:
    """CI-FAIL-26 (exact-head CI on NumPy 2.x): every numpy scalar is normalised through .item() BEFORE any repr/str
    path, so the receipt holds exact built-in types on NumPy 1.26 and 2.x alike; the 2.x spellings (repr 'np.True_',
    type name 'bool', repr 'np.float64(2.5)') are simulated with duck-typed scalars because the installed NumPy cannot
    be switched at test time."""

    @staticmethod
    def _np2(name, kind, item_of=None, repr_of=None):
        """a NumPy-2-looking scalar type: .dtype.kind, .item(), a module-prefixed repr, single-argument constructor
        (so the helper's own lossless round-trip test `type(v)(v.item()) == v` behaves as it does for real scalars)"""
        dt = type("dtype", (), {"kind": kind})()
        cls = type(name, (), {})
        cls.dtype = dt
        cls.__init__ = lambda self, v: setattr(self, "v", v)
        cls.item = lambda self: (item_of or (lambda v: v))(self.v)
        cls.__repr__ = lambda self: (repr_of or (lambda v: f"np.{name}({v!r})"))(self.v)
        cls.__str__ = lambda self: str(self.v)
        cls.__eq__ = lambda self, o: isinstance(o, cls) and o.v == self.v
        cls.__hash__ = lambda self: hash(self.v)
        return cls

    def test_real_numpy_scalars_become_exact_builtins(self):
        assert _receipt_safe(np.bool_(True)) is True and _receipt_safe(np.bool_(False)) is False
        assert type(_receipt_safe(np.int64(5))) is int and type(_receipt_safe(np.float32(0.5))) is float and _receipt_safe(np.str_("x")) == "x"
        got = _receipt_safe(np.longdouble("2000000.0000000001")); assert type(got) is str and got == str(np.longdouble("2000000.0000000001")) and "0000000001" in got and not got.startswith("np.")
        assert _exact_count(np.bool_(True)) is None and _exact_count(np.int64(3)) == 3

    def test_numpy2_spellings_never_reach_the_receipt(self):
        Bool2 = self._np2("bool", "b", repr_of=lambda v: "np.True_" if v else "np.False_")      # NumPy 2.x: type name 'bool', repr 'np.True_'
        two_bool = Bool2(True); assert repr(two_bool) == "np.True_"                              # the 2.x spelling that must NOT leak
        assert _receipt_safe(two_bool) is True
        two_float = self._np2("float64", "f")(2.5); assert repr(two_float) == "np.float64(2.5)" and _receipt_safe(two_float) == 2.5 and type(_receipt_safe(two_float)) is float
        two_int = self._np2("int64", "i")(7); assert _receipt_safe(two_int) == 7 and type(_receipt_safe(two_int)) is int
        lossy = self._np2("longdouble", "f", item_of=lambda v: 2000000.0)("2000000.0000000001")   # .item() would round the value away
        got = _receipt_safe(lossy); assert type(got) is str and got == "2000000.0000000001"
        assert _exact_count(two_bool) is None, "a numpy bool is malformed data whatever NumPy calls its type"

    def test_receipt_with_numpy_scalars_dumps_without_module_prefixes(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs_by_compute_dtype"]["nvfp4"] = np.bool_(True)
        d = _fresh_typed(tmp_path, "np-bool-receipt", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["refusal_file"]["malformed_counts"] == {"total.macs_by_compute_dtype.nvfp4": True}
        assert "np." not in (tmp_path / "np-bool-receipt" / "perf" / "perf_B200.REFUSED.yaml").read_text()


class TestRound12BoundaryCompletion:
    """the review round 11 / re-bind (unresolved-weight-boundary-1 presence admission, rate-float-boundary-1
    intermediates + protocol errors, intensity-sentinel-1 negative traffic, static-conversion-scope-1
    dataflow test with negative fixtures, numpy2-longdouble-1 explicit normalisation)."""

    @staticmethod
    def _refused(d):
        assert d["state"] == "REFUSED" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d.get("receipt", d)
        return d["refusal_file"]

    # -- presence admission --------------------------------------------------------------------------------------
    @pytest.mark.parametrize("value", [False, None, [], {}, "", np.array([1, 2]), -1, 2.5, "7"], ids=["false", "null", "list", "dict", "emptystr", "array", "negative", "fractional", "str"])
    def test_present_unresolved_weight_is_always_validated(self, tmp_path, monkeypatch, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["weight_elements_unresolved"] = value
        disk = self._refused(_run_arch(tmp_path, "w-present", a, monkeypatch=monkeypatch))
        assert disk["reason"] == "compute_dtype_count_malformed" and "total.weight_elements_unresolved" in disk["malformed_counts"]

    @pytest.mark.parametrize("value, written", [(0, False), (5, True), (np.int64(3), True)], ids=["zero", "five", "np-int"])
    def test_valid_unresolved_weight_counts_price(self, tmp_path, monkeypatch, value, written):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["weight_elements_unresolved"] = value
        d = _run_arch(tmp_path, "w-valid", a, monkeypatch=monkeypatch)
        assert d["state"] == "PRICED"
        assert ("weight_elements_unresolved" in d["perf"]["memory_breakdown"]) is written
        if written:
            assert d["perf"]["memory_breakdown"]["weight_elements_unresolved"] == int(value) and d["perf"]["memory_breakdown"]["weight_bytes_unresolved"] == 2 * int(value)

    def test_absent_unresolved_weight_is_not_written(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["total"].pop("weight_elements_unresolved", None)
        d = _run_arch(tmp_path, "w-absent", a); assert d["state"] == "PRICED" and "weight_elements_unresolved" not in d["perf"]["memory_breakdown"]

    # -- intermediate products / quotients -------------------------------------------------------------------------
    def test_huge_finite_frequency_cannot_zero_the_runtimes(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(_run_arch(tmp_path, "freq-1e308", a, arch_changes={"freq_GHz": 1e308}))
        assert disk["reason"] == "derived_quantity_not_finite" and disk["derived_not_finite"]["name"].startswith("freq_GHz x 1e6")

    def test_legacy_bytes_over_a_tiny_width_refuse_as_a_derived_quantity(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["metadata"]["bytes_per_element"] = 1e-308
        a["total"].pop("unfused_elements", None); a["total"].pop("orojenesis_elements", None); a["total"]["orojenesis_bytes"] = 32
        run = tmp_path / "legacy-width"; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True); ap.write_text(yaml.safe_dump(a, sort_keys=False))
        try:
            EinsumGraphPerfModel(dtype_bytes=False).predict(ap, run / "perf", arch_config="B200", precision="custom", copy_analysis=False); state = "PRICED"
        except PerfProjectionUnavailable as exc:
            state = "REFUSED"; reason = exc.reason
        assert state == "REFUSED" and reason == "derived_quantity_not_finite" and (run / "perf" / "perf_B200.REFUSED.yaml").exists() and not (run / "perf" / "perf_B200.yaml").exists()

    @pytest.mark.parametrize("exc_type", [RuntimeError, NotImplementedError, ArithmeticError, LookupError])
    def test_rate_numeric_protocol_errors_refuse_with_a_receipt(self, tmp_path, monkeypatch, exc_type):
        class BadRate(float):
            def __float__(self): raise exc_type("control rate conversion")
        a = _fresh_analysis("mixed_analysis_only")
        orig = EinsumGraphPerfModel._load_arch_config
        monkeypatch.setattr(EinsumGraphPerfModel, "_load_arch_config", lambda self, name: {**orig(self, name), "freq_GHz": BadRate(1.5)})
        disk = self._refused(_run_arch(tmp_path, "bad-rate-protocol", a))
        assert disk["reason"] == f"rate_protocol_error:{exc_type.__name__}" and disk["invalid_rate"]["name"] == "freq_GHz" and exc_type.__name__ in disk["invalid_rate"]["protocol_error"]

    @pytest.mark.parametrize("exc_type", [MemoryError, KeyboardInterrupt, SystemExit])
    def test_rate_resource_and_control_exceptions_propagate(self, tmp_path, monkeypatch, exc_type):
        class BadRate(float):
            def __float__(self): raise exc_type("control rate conversion")
        a = _fresh_analysis("mixed_analysis_only")
        orig = EinsumGraphPerfModel._load_arch_config
        monkeypatch.setattr(EinsumGraphPerfModel, "_load_arch_config", lambda self, name: {**orig(self, name), "freq_GHz": BadRate(1.5)})
        run = tmp_path / "ctl"; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True); ap.write_text(yaml.safe_dump(a, sort_keys=False))
        with pytest.raises(exc_type):
            EinsumGraphPerfModel(dtype_bytes=True).predict(ap, run / "perf", arch_config="B200", precision="bf16", copy_analysis=False)

    # -- negative traffic / sentinel ------------------------------------------------------------------------------
    @pytest.mark.parametrize("field", ["unfused_bytes", "fused_bytes", "fused_prefetched_bytes", "intermediate_bytes", "external_input_bytes"])
    def test_negative_traffic_refuses_instead_of_hitting_the_sentinel(self, tmp_path, field):
        a = _fresh_analysis("mixed_analysis_only"); a["total"][field] = -16
        disk = self._refused(_run_arch(tmp_path, "neg-" + field, a))
        assert disk["reason"] == "traffic_negative" and any(field in k for k in disk["malformed_counts"]) and disk["malformed_counts"]["total.macs"] == 3_000_000

    @pytest.mark.parametrize("field", ["unfused_elements", "fused_elements", "weight_elements"])
    def test_negative_elements_refuse(self, tmp_path, field):
        a = _fresh_analysis("mixed_analysis_only"); a["total"][field] = -4
        disk = self._refused(_run_arch(tmp_path, "neg-" + field, a)); assert disk["reason"] == "traffic_negative"

    def test_negative_flops_or_ops_refuse_as_a_negative_count(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["flops"] = -1
        disk = self._refused(_run_arch(tmp_path, "neg-flops", a)); assert disk["reason"] == "count_negative"

    def test_zero_bytes_keep_the_documented_sentinel(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["fused_bytes"] = 0; a["total"]["fused_elements"] = 0
        d = _run_arch(tmp_path, "zero-bytes", a); assert d["state"] == "PRICED" and d["perf"]["fused"]["arithmetic_intensity"] == float("inf") and d["perf"]["fused"]["memory_bytes"] == 0

    # -- numpy long double ----------------------------------------------------------------------------------------
    def test_real_longdouble_is_a_plain_digit_string_on_every_numpy_spelling(self):
        raw = np.longdouble("2000000.0000000001"); assert type(raw.item()) is type(raw)
        got = _receipt_safe(raw); assert type(got) is str and got == str(raw) == "2000000.0000000001" and not got.startswith("np.")
        import unittest.mock
        orig = repr
        def np2repr(v): return "np.longdouble(" + orig(str(v)) + ")" if type(v) is np.longdouble else orig(v)
        with unittest.mock.patch.object(perf_model_module, "repr", np2repr, create=True):
            assert _receipt_safe(raw) == "2000000.0000000001"
        assert _receipt_safe(np.longdouble(2) ** 70) == float(2 ** 70) and type(_receipt_safe(np.longdouble("3.5"))) is float
        assert _receipt_safe(np.clongdouble(1.5 + 2j)) == "1.5+2j" and _receipt_safe(np.datetime64("2026-10-26")) == "2026-10-26"
        for v in (np.bool_(True), np.int64(5), np.uint8(7), np.float32(0.5), np.str_("x"), np.bytes_(b"ab"), raw):
            assert type(_receipt_safe(v)) in (bool, int, float, str), v

    def test_longdouble_count_receipt_end_to_end(self, tmp_path, monkeypatch):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs"] = np.longdouble("2000000.0000000001")
        d = _fresh_typed(tmp_path, "ld-e2e", a, monkeypatch)
        assert d["state"] == "REFUSED" and d["refusal_file"]["malformed_counts"]["total.macs"] == "2000000.0000000001"


class TestRound13TrafficAndRateBoundary:
    """the review round 12 (verdict on an earlier revision): declared-count-roundtrip-1 (the ORIGINAL declared integer is
    the record), traffic-count-validation-1 (every raw traffic / count total validated on the ORIGINAL object
    before any conversion; FINAL re-priced traffic non-negative; signed intermediate deltas allowed),
    rate-protocol-coverage-1 (truth protocol inside the boundary; exception TYPE NAME only),
    static-conversion-scope-1 (taint-based audit; the reviewer's four mutations as negative fixtures)."""

    @staticmethod
    def _refused(d, reason=None):
        assert d["state"] == "REFUSED" and d["refusal_file_bytes"] > 0 and not d["numeric_perf_exists"], d.get("receipt", d)
        if reason is not None:
            assert d["refusal_file"]["reason"] == reason, d["refusal_file"]["reason"]
        return d["refusal_file"]

    @staticmethod
    def _total_only():
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype", None); return a

    # -- declared count round trip --------------------------------------------------------------------------------
    @pytest.mark.parametrize("count", [2**53 + 1, 2**53 + 3, 2**64 + 7])
    def test_declared_boundary_record_keeps_the_exact_original_integer(self, tmp_path, count):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = count
        d = _run_arch(tmp_path, "declared-exact", a)
        assert d["state"] == "PRICED"
        got = d["perf"]["workload"]["model_io_bytes_declared"]
        assert type(got) is int and got == count, got                      # not the float round trip (...992 / ...996)
        assert d["perf"]["workload"]["boundary_billing"] == "declared_storage_dtypes"

    def test_declared_boundary_record_is_exact_on_disk_too(self, tmp_path):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = 2**53 + 1
        d = _run_arch(tmp_path, "declared-disk", a)
        on_disk = yaml.safe_load((tmp_path / "declared-disk" / "perf" / "perf_B200.yaml").read_text())
        assert on_disk["workload"]["model_io_bytes_declared"] == 2**53 + 1 and type(on_disk["workload"]["model_io_bytes_declared"]) is int

    # -- original-object traffic validation -----------------------------------------------------------------------
    TRAFFIC_FIELDS = ["unfused_bytes", "fused_bytes", "fused_prefetched_bytes", "unfused_elements", "fused_elements",
                      "fused_prefetched_elements", "weight_elements", "model_io_elements", "intermediate_elements", "intermediate_bytes",
                      "external_input_bytes", "external_output_bytes"]

    @pytest.mark.parametrize("value", [0.5, True, "16"], ids=["fraction", "bool", "numeric_string"])
    @pytest.mark.parametrize("field", ["unfused_bytes", "fused_bytes", "fused_prefetched_bytes"])
    def test_review3_traffic_bytes_must_be_exact_integers(self, tmp_path, field, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"][field] = value
        disk = self._refused(_run_arch(tmp_path, "traffic-" + field, a), "traffic_not_exact_int")
        assert disk["malformed_counts"]["total." + field] == value and disk["malformed_counts"]["total.macs"] == 3_000_000
        assert "total." + field in disk["malformed_protocol_errors"]

    @pytest.mark.parametrize("field", TRAFFIC_FIELDS)
    def test_every_traffic_total_is_validated_on_the_original_object(self, tmp_path, field):
        a = _fresh_analysis("mixed_analysis_only"); a["total"][field] = "8"
        self._refused(_run_arch(tmp_path, "traffic-str-" + field, a), "traffic_not_exact_int")

    @pytest.mark.parametrize("value", [None, [], {}, np.array([1, 2])], ids=["null", "list", "dict", "array"])
    def test_present_non_count_traffic_refuses(self, tmp_path, monkeypatch, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["fused_bytes"] = value
        self._refused(_run_arch(tmp_path, "traffic-nc", a, monkeypatch=monkeypatch), "traffic_not_exact_int")

    def test_typed_negative_longdouble_refuses_before_it_rounds_to_negative_zero(self, tmp_path, monkeypatch):
        raw = np.longdouble("-1e-4000"); assert raw < 0 and float(raw) == 0
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["fused_bytes"] = raw
        disk = self._refused(_fresh_typed(tmp_path, "ld-neg", a, monkeypatch), "traffic_negative")
        assert "total.fused_bytes" in disk["malformed_counts"]

    @pytest.mark.parametrize("field", ["flops", "other_ops"])
    @pytest.mark.parametrize("value", [True, "6", 0.5], ids=["bool", "numeric_string", "fraction"])
    def test_counts_are_validated_on_the_original_object_too(self, tmp_path, field, value):
        a = _fresh_analysis("mixed_analysis_only"); a["total"][field] = value
        self._refused(_run_arch(tmp_path, "count-" + field, a), "count_not_exact_int")

    @pytest.mark.parametrize("field", ["model_io_bytes_declared", "external_io_bytes_declared_dedup"])
    def test_declared_boundary_fields_are_validated_on_the_original_object(self, tmp_path, field):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = 64; a["total"][field] = 0.5
        self._refused(_run_arch(tmp_path, "declared-" + field, a), "traffic_not_exact_int")

    @pytest.mark.parametrize("value, expected", [(16, 16), (16.0, 16), (np.int64(16), 16), (np.float64(16.0), 16), (0, 0)], ids=["int", "integral-float", "np-int", "np-integral-float", "zero"])
    def test_integral_traffic_values_still_price(self, tmp_path, monkeypatch, value, expected):
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["fused_bytes"] = value; a["total"]["fused_elements"] = value
        d = _run_arch(tmp_path, "traffic-ok", a, monkeypatch=monkeypatch)
        assert d["state"] == "PRICED" and d["perf"]["fused"]["memory_bytes"] == expected

    # -- final re-priced traffic ----------------------------------------------------------------------------------
    def test_repriced_unfused_traffic_cannot_be_negative(self, tmp_path):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = 0; a["total"]["boundary_declared"] = {"elements_raw": 100}
        disk = self._refused(_run_arch(tmp_path, "final-unfused", a), "traffic_negative_final")
        assert disk["traffic_final_negative"]["name"].startswith("unfused_bytes") and disk["traffic_final_negative"]["value"] < 0
        assert "derived_not_finite" not in disk

    def test_repriced_weight_footprint_cannot_be_negative(self, tmp_path):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = 0
        a["total"]["boundary_declared"] = {"elements_raw": 0, "footprint_elements_weight": 100, "bytes_footprint_weight": 0}
        disk = self._refused(_run_arch(tmp_path, "final-weight", a), "traffic_negative_final")
        assert disk["traffic_final_negative"]["name"].startswith("weight_bytes")

    def test_signed_intermediate_delta_is_allowed_when_the_final_total_stays_nonnegative(self, tmp_path):
        a = self._total_only(); a["total"]["model_io_bytes_declared"] = 4; a["total"]["boundary_declared"] = {"elements_raw": 10}   # delta -16 on 32 uniform bytes
        d = _run_arch(tmp_path, "signed-delta", a)
        assert d["state"] == "PRICED" and d["perf"]["workload"]["model_io_bytes_declared"] == 4

    # -- rate protocol coverage -----------------------------------------------------------------------------------
    def test_dtype_rate_truth_protocol_runs_inside_the_boundary(self, tmp_path):
        class BadBool(float):
            def __bool__(self): raise RuntimeError("dtype rate truth failed")
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(self._run_typed_arch(tmp_path, "bad-bool", a, {"MAC_per_cycle_nvfp4_tc": BadBool(2.0)}))
        assert disk["reason"] == "rate_protocol_error:RuntimeError" and disk["invalid_rate"]["name"] == "MAC_per_cycle_nvfp4_tc"

    def test_unrenderable_rate_exception_still_refuses_with_the_type_name_only(self, tmp_path):
        class BadMessage(RuntimeError):
            def __str__(self): raise ValueError("rate message rendering failed")
        class BadRate(float):
            def __float__(self): raise BadMessage()
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(self._run_typed_arch(tmp_path, "bad-msg", a, {"freq_GHz": BadRate(1.5)}))
        assert disk["reason"] == "rate_protocol_error:BadMessage" and disk["invalid_rate"]["protocol_error"] == "BadMessage"

    def test_zero_declared_dtype_rate_is_an_invalid_rate_not_a_missing_one(self, tmp_path):
        a = _fresh_analysis("mixed_analysis_only")
        disk = self._refused(_run_arch(tmp_path, "zero-rate", a, arch_changes={"MAC_per_cycle_nvfp4_tc": 0}), "rate_invalid:MAC_per_cycle_nvfp4_tc")
        assert disk["invalid_rate"]["value"] == 0

    @staticmethod
    def _run_typed_arch(tmp_path, name, data, architecture):
        """predict() with TYPED architecture values (a YAML cannot carry a float subtype): the loader is patched."""
        import unittest.mock
        run = tmp_path / name; ap = run / "analysis" / "analysis.yaml"; ap.parent.mkdir(parents=True); ap.write_text(yaml.safe_dump(data, sort_keys=False))
        orig = EinsumGraphPerfModel._load_arch_config
        d = {}
        with unittest.mock.patch.object(EinsumGraphPerfModel, "_load_arch_config", lambda self, n: {**orig(self, n), **architecture}):
            try:
                p = EinsumGraphPerfModel(dtype_bytes=True).predict(ap, run / "perf", arch_config="B200", precision="bf16", copy_analysis=False); d.update(state="PRICED", perf=p)
            except PerfProjectionUnavailable as exc: d.update(state="REFUSED", receipt=exc.receipt())
        rf = run / "perf" / "perf_B200.REFUSED.yaml"
        d["refusal_file"] = yaml.safe_load(rf.read_text()) if rf.exists() else None; d["refusal_file_bytes"] = rf.stat().st_size if rf.exists() else None
        d["numeric_perf_exists"] = (run / "perf" / "perf_B200.yaml").exists()
        return d


class TestRound14ZeroMacRateAndStaticAliases:
    """the review round 13 (verdict on an earlier revision): zero-mac-rate-1 (a declared rate -- zero or otherwise -- is required
    only for MAC-bearing dtype buckets; zero-MAC buckets are traffic-only whatever the profile says) and
    static-conversion-scope-1 (mapping defaults, container mutation, import aliases, unresolved callees)."""

    @staticmethod
    def _total_only():
        a = _fresh_analysis("mixed_analysis_only"); a["layers"] = {}; a["total"].pop("macs_by_compute_dtype", None); return a

    # -- zero-MAC buckets ----------------------------------------------------------------------------------------
    @pytest.mark.parametrize("representation", ["aggregate_only", "zero_layer_and_aggregate"])
    @pytest.mark.parametrize("rate", [0, 2_415_000.0, "not-a-rate", None], ids=["zero", "valid", "string", "absent"])
    def test_zero_mac_dtype_bucket_prices_whatever_rate_the_profile_declares(self, tmp_path, representation, rate):
        a = self._total_only(); a["total"]["macs_by_compute_dtype"] = {"nvfp4": 0}
        if representation == "zero_layer_and_aggregate":
            a["layers"] = {"storage_copy": {"compute_dtype": "nvfp4", "macs": 0}}
        # B200 declares MAC_per_cycle_nvfp4_tc: "absent" is exercised on H100_PCIe (no NVFP4 rate), the declared variants on B200
        d = _run_arch(tmp_path, "zero-mac", a, arch="B200" if rate is not None else "H100_PCIe", arch_changes={"MAC_per_cycle_nvfp4_tc": rate} if rate is not None else None)
        assert d["state"] == "PRICED" and d["numeric_perf_exists"], d.get("receipt")
        assert d["perf"]["workload"]["total_macs"] == 3_000_000
        rec = d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"]
        assert rec["status"] == "traffic_only_no_rate_needed" and rec["MAC_per_cycle"] is None and rec["macs"] == 0
        assert ("declared_rate_not_required" in rec) is (rate is not None)        # the H100 (no rate) entry is byte-for-byte the round-7 control
        if rate is not None:
            assert rec["declared_rate_not_required"] == rate

    def test_zero_mac_bucket_on_h100_keeps_the_exact_control_entry(self, tmp_path):
        a = self._total_only(); a["total"]["macs_by_compute_dtype"] = {"nvfp4": 0}
        d = _run_arch(tmp_path, "zero-mac-h100", a, arch="H100_PCIe")
        assert d["state"] == "PRICED"
        assert d["perf"]["arch"]["compute_dtype_rates"]["nvfp4"] == {"key": "MAC_per_cycle_nvfp4_tc", "MAC_per_cycle": None, "macs": 0.0, "status": "traffic_only_no_rate_needed"}

    @pytest.mark.parametrize("rate", [0, -1.0, "x", float("nan")], ids=["zero", "negative", "string", "nan"])
    def test_positive_mac_bucket_with_an_invalid_declared_rate_refuses(self, tmp_path, rate):
        a = _fresh_analysis("mixed_analysis_only")
        d = _run_arch(tmp_path, "pos-mac-bad-rate", a, arch_changes={"MAC_per_cycle_nvfp4_tc": rate})
        assert d["state"] == "REFUSED" and d["refusal_file"]["reason"] == "rate_invalid:MAC_per_cycle_nvfp4_tc" and not d["numeric_perf_exists"]

    def test_zero_aggregate_with_positive_layers_is_still_a_contradiction(self, tmp_path):
        """eff_macs == 0 from the aggregate but MAC-bearing layers exist: the exact-evidence refusal is unchanged."""
        a = _fresh_analysis("mixed_analysis_only"); a["total"]["macs_by_compute_dtype"] = {"nvfp4": 0}
        d = _run_arch(tmp_path, "zero-agg-pos-layers", a)
        assert d["state"] == "REFUSED" and d["refusal_file"]["reason"] == "conflicting compute-dtype evidence"
        d2 = _run_arch(tmp_path, "zero-agg-pos-layers-h100", a, arch="H100_PCIe")
        assert d2["state"] == "REFUSED" and d2["refusal_file"]["reason"] == "perf_projection_unavailable"
