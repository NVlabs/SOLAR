# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""loop-undercount-1 (independent review): the DSL kernel loop-count cost basis is used only when the
converter PROVES the loop nest complete; otherwise the equation basis is billed and the layer is marked
``loop_metadata_incomplete`` (carried by the INTEG-9 contract).  Never a blind maximum of the two bases.

Also historical-scope-2: the measured-density compute AND memory discounts (and ``compute_density``) apply to
DSL @kernel bodies only, never to a bodiless builtin layer; and the malformed-``loop_iterations`` fall-through control.

Round 3 (independent review): loop-completeness-4 — the completeness proof IS the counted traversal
(``_loop_paths`` records every loop it does not represent and every container it does not descend into: a
``for ... else`` body, ``while``, ``with``, ``try``, comprehensions); loop-marker-5 — omissions are carried
and consumed with no numeric count at all (a kernel whose only loop is a ``while``); the reviewer's nine extended
controls (``tests/fixtures/loop_evidence/probe_extended_v2.py``, verbatim) natively and as the verbatim script.
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_sparse_cost import DENSE_DSL, SLOT_DSL, _traced_metadata, _analyze, _moe_inputs, N, TOPK, H  # noqa: E402

from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.verify import execute_translated
from solar.perf.perf_model import EinsumGraphPerfModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE_V2 = pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "probe_extended_v2.py"


@pytest.fixture(autouse=True)
def _clean():
    clear_dims(); stop_trace()
    yield
    stop_trace(); clear_dims()


INPUTS = {"x": np.arange(1, 9, dtype=np.float32).reshape(2, 4), "w": np.arange(1, 13, dtype=np.float32).reshape(3, 4)}
RANGE_INNER = DENSE_DSL.replace('for h in Dim("H"):', "for h in range(X.shape[1]):")


def _graph_layer(tmp_path, name):
    g = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())
    return g["layers"][name]


class TestLoopBasisOnlyWhenProvenComplete:
    def test_all_dim_control_bills_the_loop_count(self, tmp_path):
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        a = _analyze(DENSE_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "linear#0"); layer = a["layers"]["linear#0"]
        assert raw["loop_iterations"] == 24 and raw["loop_iterations_complete"] is True and "loop_iterations_omitted" not in raw
        assert layer["macs"] == 24 and "status" not in layer and "status" not in a["total"]

    def test_inner_range_loop_is_unproven_equation_basis_with_marker(self, tmp_path):
        """The reviewer's case: ``for h in range(X.shape[1])`` is dropped by the Dim path walk (count 6) while the
        faithful equation NH,FH->NF is 24 MACs; the executed output equals X @ W.T."""
        out, _ = execute_translated(RANGE_INNER, {}, all_inputs=INPUTS, trace=False, fast=False)
        np.testing.assert_allclose(out["output"], INPUTS["x"] @ INPUTS["w"].T)
        md = _traced_metadata(RANGE_INNER, INPUTS, profile=False)
        a = _analyze(RANGE_INNER, md, tmp_path)
        raw = _graph_layer(tmp_path, "linear#0"); layer = a["layers"]["linear#0"]
        assert raw["loop_iterations"] == 6 and raw["loop_iterations_complete"] is False
        assert raw["loop_iterations_omitted"] == ["for h in range(X.shape[1]) (line 10): not a Dim() loop"]
        assert layer["macs"] == 24 and layer["macs_dense"] == 24                       # equation basis, never the 6
        assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == raw["loop_iterations_omitted"]
        assert a["total"]["status"] == "incomplete" and a["total"]["incomplete_layers"] == {"linear#0": "loop_metadata_incomplete"}

    def test_gathered_slot_kernel_keeps_its_loop_count(self, tmp_path):
        """The correction the loop basis exists for: the slot-dense routed kernel visits N*K*H, not N*E*H (8,192)."""
        md = _traced_metadata(SLOT_DSL, _moe_inputs(), profile=True)
        a = _analyze(SLOT_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "routed#0"); layer = a["layers"]["routed#0"]
        assert raw["loop_iterations_complete"] is True
        assert layer["macs"] == N * TOPK * H == 1024 and "compute_density" not in layer and "status" not in layer

    def test_derived_index_kernel_keeps_its_loop_count(self, tmp_path):
        """A derived index (``k // 2``) makes the equation's index union over-count; the complete Dim nest is the work."""
        code = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def half(X: Tensor("N,K", dtype="fp32"), W: Tensor("J,D", role="weight", dtype="fp32")) -> Tensor("N,D"):
    Y = Tensor(N=X.shape[0], D=W.shape[1], dtype="fp32")
    for n in Dim("N", parallel=True):
        for d in Dim("D"):
            for k in Dim("K"):
                Y[n, d] += X[n, k] * W[k // 2, d]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "J,D", dtype="fp32")
    return {"output": half(x, w).data}
'''
        inputs = {"x": np.ones((2, 8), dtype=np.float32), "w": np.ones((4, 3), dtype=np.float32)}
        md = _traced_metadata(code, inputs, profile=False)
        a = _analyze(code, md, tmp_path)
        raw = _graph_layer(tmp_path, "half#0"); layer = a["layers"]["half#0"]
        assert raw["loop_iterations"] == 2 * 3 * 8 and raw["loop_iterations_complete"] is True
        assert layer["macs"] == 48 and "status" not in layer

    def test_malformed_loop_iterations_falls_through_to_the_equation_with_a_marker(self, tmp_path):
        """A non-integer ``loop_iterations`` in a saved graph: no exception, equation basis (24), marked incomplete."""
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        g["layers"]["linear#0"]["loop_iterations"] = "twenty-four"
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))
        layer = a["layers"]["linear#0"]
        assert layer["macs"] == 24 and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == ["loop_count_malformed:'twenty-four'"]

    def test_unflagged_positive_count_is_not_trusted(self, tmp_path):
        """A saved graph that carries a positive ``loop_iterations`` without the completeness proof (older
        converter output) is billed by the equation and marked — the count is not taken on faith."""
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        g["layers"]["linear#0"]["loop_iterations"] = 6                       # a wrong count ...
        del g["layers"]["linear#0"]["loop_iterations_complete"]             # ... with no proof
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))
        layer = a["layers"]["linear#0"]
        assert layer["macs"] == 24 and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == ["loop_iterations_complete not recorded"]


class TestDiscountAppliesToKernelBodiesOnly:
    def test_builtin_gather_gets_no_compute_density_and_keeps_the_target_accounting(self, tmp_path):
        """The reviewer's gather: two rows of a [16,4] table with measured density 0.5.  A bodiless builtin is not
        a compute op: no compute_density.  Its work and traffic follow the gather rule (issue #59,
        TARGET-GATHER-COST-1): one address op per gathered element (8), reads = gathered rows (8) + index (2),
        output 8 -- never the table's dense extent product (the former 64) and never discounted by density."""
        code = '''
from solar.einsum_lib import Tensor, clear_dims, gather
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs['x'], 'E,H', dtype='fp32')
    i = Tensor.from_numpy(inputs['i'], 'N', dtype='int32')
    return {'output': gather(x, i, 'E').data}
'''
        x = np.ones((16, 4), dtype=np.float32); x[8:] = 0
        md = _traced_metadata(code, {"x": x, "i": np.array([1, 2], dtype=np.int32)}, profile=True)
        a = _analyze(code, md, tmp_path)
        layer = a["layers"]["gather#0"]; raw = _graph_layer(tmp_path, "gather#0")
        assert raw["loop_expr"] is None and raw["is_real_einsum"] is False and raw["tensor_sparsity"]["inputs"][0] == 0.5
        assert "compute_density" not in layer and layer["macs"] == 0 and layer["other_ops"] == 8
        assert layer["cost_basis"] == "gather_elements"
        # historical-scope-2 (round 3): the measured density of the table is not applied to a bodiless
        # builtin's reads either; the gather rule reads the gathered rows and the index, writes the output.
        assert layer["memory_elements"]["inputs"] == [8, 2] and layer["memory_elements"]["outputs"] == [8]
        assert layer["unfused_elements"] == 18
        # and a builtin is not marked by the no-count omission path (loop-marker-5)
        assert "loop_iterations" not in raw and "loop_iterations_omitted" not in raw
        assert "status" not in layer and "status" not in a["total"]

    def test_kernel_body_keeps_both_discounts(self, tmp_path):
        """Control for the gate: a DSL kernel body WITH a measured-density input is still discounted on both sides
        (compute via compute_density, memory via Step 2b) — the scope restriction removes nothing from kernel bodies."""
        x = np.array([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=np.float32)            # density 0.5 on X
        inputs = {"x": x, "w": INPUTS["w"]}
        md = _traced_metadata(DENSE_DSL, inputs, profile=True)
        a = _analyze(DENSE_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "linear#0"); layer = a["layers"]["linear#0"]
        assert raw["tensor_sparsity"]["inputs"][0] == 0.5 and raw["loop_iterations_complete"] is True
        assert layer["compute_density"] == 0.5 and layer["macs"] == 12 and layer["macs_dense"] == 24
        assert layer["memory_elements"]["inputs"] == [4, 12]                      # 8 * 0.5, 12 * 1.0
        assert "status" not in layer


# ─── Round 3: the reviewer's nine extended controls (probe_extended_v2.py) ───────────────────────────────────────
_MAC = '                Y[n, f] += X[n, h] * W[f, h]'
_INNER = '            for h in Dim("H"):\n' + _MAC
_DOT = '''
from solar.einsum_lib import Tensor, kernel, clear_dims
@kernel
def dot(X: Tensor("H", dtype="fp32"), W: Tensor("H", dtype="fp32")) -> Tensor("O"):
    Y = Tensor(O=1, dtype="fp32")
    for h in range(X.shape[0]):
        Y[0] += X[h] * W[h]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "H", dtype="fp32")
    return {"output": dot(x,w).data}
'''
_DOT_INPUTS = {"x": np.arange(1, 5, dtype=np.float32), "w": np.arange(2, 6, dtype=np.float32)}
# name -> (code, inputs, executed MACs, output factor, expected loop_iterations, expected omission fragments)
EXTENDED_CONTROLS = {
    "all_dim": (DENSE_DSL, INPUTS, 24, 1, 24, None),
    "inner_range": (RANGE_INNER, INPUTS, 24, 1, 6, ["for h in range(X.shape[1]) (line 10): not a Dim() loop"]),
    "all_range": (DENSE_DSL.replace('for n in Dim("N", parallel=True):', 'for n in range(X.shape[0]):')
                  .replace('for f in Dim("F"):', 'for f in range(W.shape[0]):')
                  .replace('for h in Dim("H"):', 'for h in range(X.shape[1]):'), INPUTS, 24, 1, 24,
                  ["for n in range(X.shape[0]) (line 8): not a Dim() loop", "for f in range(W.shape[0]) (line 9): not a Dim() loop",
                   "for h in range(X.shape[1]) (line 10): not a Dim() loop", "standalone-index fallback: no complete Dim() loop structure"]),
    "mixed_while": (DENSE_DSL.replace(_MAC, _MAC + '\n                repeat = 0\n                while repeat < 1:\n'
                                      '                    Y[n, f] += X[n, h] * W[f, h]\n                    repeat += 1'),
                    INPUTS, 48, 2, 48, ["while repeat < 1 (line 13): not visited by the loop counter"]),
    "try_dim": (DENSE_DSL.replace(_INNER, '            try:\n                for h in Dim("H"):\n'
                                  '                    Y[n, f] += X[n, h] * W[f, h]\n            finally:\n                pass'),
                INPUTS, 24, 1, 6, ["try (line 10): not visited by the loop counter"]),
    "with_dim": ('from contextlib import nullcontext\n' + DENSE_DSL.replace(
        _INNER, '            with nullcontext():\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]'),
        INPUTS, 24, 1, 6, ["with (line 11): not visited by the loop counter"]),
    "for_else_dim": (DENSE_DSL.replace(_MAC, _MAC + '\n            else:\n                for h in Dim("H"):\n'
                                       '                    Y[n, f] += X[n, h] * W[f, h]'),
                     INPUTS, 48, 2, 24, ["for ... else body (line 13): not visited by the loop counter"]),
    "single_range": (_DOT, _DOT_INPUTS, 4, 1, 4, ["for h in range(X.shape[0]) (line 6): not a Dim() loop",
                                                  "standalone-index fallback: no complete Dim() loop structure"]),
    "single_while": (_DOT.replace('    for h in range(X.shape[0]):', '    h = 0\n    while h < X.shape[0]:')
                     .replace('        Y[0] += X[h] * W[h]', '        Y[0] += X[h] * W[h]\n        h += 1'),
                     _DOT_INPUTS, 4, 1, None, ["while h < X.shape[0] (line 7): not visited by the loop counter"]),
}


class TestCountedTraversalIsTheProof:
    """loop-completeness-4 / -MARKER-5: every structure the counter does not visit fails closed — incomplete
    at the layer, the analysis total and the perf workload, equation basis, never a blind maximum — and the all-Dim
    control stays exact and unmarked."""

    @pytest.mark.parametrize("name", list(EXTENDED_CONTROLS))
    def test_control(self, name, tmp_path):
        code, inputs, executed, factor, expect_count, expect_omitted = EXTENDED_CONTROLS[name]
        out, _ = execute_translated(code, {}, all_inputs=inputs, trace=False, fast=False)
        np.testing.assert_allclose(out["output"], inputs["x"] @ inputs["w"].T * factor)      # the program's real work
        md = _traced_metadata(code, inputs, profile=False)
        a = _analyze(code, md, tmp_path)
        lid = next(k for k in a["layers"] if k != "start")
        raw = _graph_layer(tmp_path, lid); layer = a["layers"][lid]
        perf = EinsumGraphPerfModel().predict(tmp_path / "analysis" / "analysis.yaml", tmp_path / "perf",
                                              arch_config=str(ROOT / "configs" / "arch" / "H100_PCIe.yaml"))
        assert perf["workload"]["total_macs"] == a["total"]["macs"]
        assert raw.get("loop_iterations") == expect_count
        if expect_omitted is None:
            assert raw["loop_iterations_complete"] is True and "loop_iterations_omitted" not in raw
            assert layer["macs"] == executed and "status" not in layer
            assert "status" not in a["total"] and "status" not in perf["workload"]
        else:
            assert raw["loop_iterations_complete"] is False and raw["loop_iterations_omitted"] == expect_omitted
            assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
            assert layer["loop_iterations_omitted"] == expect_omitted
            assert a["total"]["status"] == "incomplete" and perf["workload"]["status"] == "incomplete"
            if expect_count == 6:
                assert layer["macs"] != 6                    # the partial count (6 of 24) is never billed

    def test_parser_records_the_omission_without_any_count(self):
        """loop-marker-5 at the source: the sole ``while`` loop has no Dim() path, no count, and one omission."""
        src = pathlib.Path(__file__).parent / "_tmp_single_while_einsum_lib.py"
        try:
            src.write_text(EXTENDED_CONTROLS["single_while"][0])
            info = extract_kernel_info(src)["dot"]
        finally:
            src.unlink()
        assert info["loop_paths"] is None and info["loop_paths_complete"] is False
        assert info["loop_paths_omitted"] == ["while h < X.shape[0] (line 7): not visited by the loop counter"]

    def test_containers_the_counter_never_visits_are_omissions_even_when_they_hold_dim_loops(self):
        """The round-2 walker ACCEPTED Dim() loops inside for-else / with / try; the counter never visits them, so
        the counted paths (24 for for-else, 6 for try/with) are not the kernel's work and may not be certified."""
        for name, expect in (("for_else_dim", "for ... else body"), ("try_dim", "try (line"), ("with_dim", "with (line")):
            src = pathlib.Path(__file__).parent / f"_tmp_{name}_einsum_lib.py"
            try:
                src.write_text(EXTENDED_CONTROLS[name][0])
                info = extract_kernel_info(src)["linear"]
            finally:
                src.unlink()
            assert info["loop_paths_complete"] is False, name
            assert len(info["loop_paths_omitted"]) == 1 and info["loop_paths_omitted"][0].startswith(expect), name

    def test_comprehension_and_match_are_omissions(self):
        src = pathlib.Path(__file__).parent / "_tmp_comp_einsum_lib.py"
        code = DENSE_DSL.replace(_MAC, '                s = sum(X[n, k] for k in range(1))\n' + _MAC)
        try:
            src.write_text(code)
            info = extract_kernel_info(src)["linear"]
        finally:
            src.unlink()
        assert info["loop_paths_complete"] is False
        assert info["loop_paths_omitted"] == ["comprehension at line 11: not visited by the loop counter"]

    def test_reviewer_probe_extended_v2_verbatim(self, tmp_path):
        """The reviewer's script, byte-for-byte (sha256 b6f334be…), run from the repo root: exit 0 = all nine
        qualification checks satisfied or explicitly incomplete."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, str(PROBE_V2), str(tmp_path / "probe")], cwd=str(ROOT), env=env,
                             capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        records = json.loads((tmp_path / "probe" / "RESULT.json").read_text())
        assert len(records) == 9 and all(r["qualification_check_passed"] for r in records)
        assert all(r.get("numerical_output_correct") for r in records)
        marked = {r["case"]: r["all_three_incomplete"] for r in records}
        assert marked == {"all_dim": False, **{c: True for c in marked if c != "all_dim"}}


# 119-SOL-GATHER-1: the shape of agentic 119's ``slot_proj_bwd_act`` (two accumulation chains per gathered slot).
TWO_CHAIN_GATHER_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def slot_proj_bwd_act(GGP: Tensor("T,K,I", dtype="fp32"),
                      GUO: Tensor("T,K,I", dtype="fp32"),
                      SE: Tensor("T,K", dtype="int64"),
                      WG: Tensor("E,I,H", role="weight", dtype="fp32"),
                      WU: Tensor("E,I,H", role="weight", dtype="fp32")) -> Tensor("T,H"):
    Y = Tensor(T=GGP.shape[0], H=WG.shape[2], dtype="fp32")
    for t in Dim("T", parallel=True):
        for k in Dim("K"):
            e = int(SE[t, k])
            for h in Dim("H"):
                for i in Dim("I"):
                    Y[t, h] += GGP[t, k, i] * WG[e, i, h]
                    Y[t, h] += GUO[t, k, i] * WU[e, i, h]
    return Y

def run_model(inputs):
    clear_dims()
    ggp = Tensor.from_numpy(inputs["ggp"], "T,K,I", dtype="fp32")
    guo = Tensor.from_numpy(inputs["guo"], "T,K,I", dtype="fp32")
    se = Tensor.from_numpy(inputs["se"], "T,K", dtype="int64")
    wg = Tensor.from_numpy(inputs["wg"], "E,I,H", dtype="fp32")
    wu = Tensor.from_numpy(inputs["wu"], "E,I,H", dtype="fp32")
    return {"output": slot_proj_bwd_act(ggp, guo, se, wg, wu).data}
'''
T2, K2, I2, H2, E2 = 4, 2, 3, 5, 6


def _two_chain_inputs():
    rng = np.random.default_rng(119)
    return {"ggp": rng.standard_normal((T2, K2, I2)).astype(np.float32),
            "guo": rng.standard_normal((T2, K2, I2)).astype(np.float32),
            "se": rng.integers(0, E2, size=(T2, K2)).astype(np.int64),
            "wg": rng.standard_normal((E2, I2, H2)).astype(np.float32),
            "wu": rng.standard_normal((E2, I2, H2)).astype(np.float32)}


def _two_chain_reference(inp):
    """Independent count and value from the indexed-loop semantics: one MAC per executed ``+=``."""
    y = np.zeros((T2, H2)); macs = 0
    for t in range(T2):
        for k in range(K2):
            e = int(inp["se"][t, k])
            for h in range(H2):
                for i in range(I2):
                    y[t, h] += inp["ggp"][t, k, i] * inp["wg"][e, i, h]; macs += 1
                    y[t, h] += inp["guo"][t, k, i] * inp["wu"][e, i, h]; macs += 1
    return y, macs


class TestMultiChainGatheredSlot:
    """119-SOL-GATHER-1 (independent review of the fresh gathered-attention candidate): a gathered-slot kernel with TWO accumulation chains per
    slot (119's ``slot_proj_bwd_act``) is billed at its executed work. The target bills the as-written equation dense over
    the gathered E axis and blind to the chain count (720 here = E/2 x; 128x on 119, 256x for single-chain gathers).
    The expected count is the executed indexed loop itself: no ratio, no threshold."""

    def test_two_chain_gathered_slot_kernel_is_billed_per_executed_accumulation(self, tmp_path):
        inp = _two_chain_inputs()
        y_ref, macs_ref = _two_chain_reference(inp)
        assert macs_ref == 2 * T2 * K2 * I2 * H2                                     # 240: two chains x the slot loops
        out, _ = execute_translated(TWO_CHAIN_GATHER_DSL, {}, all_inputs=inp, trace=False, fast=False)
        np.testing.assert_allclose(out["output"], y_ref, rtol=1e-5, atol=1e-5)
        md = _traced_metadata(TWO_CHAIN_GATHER_DSL, inp, profile=False)
        a = _analyze(TWO_CHAIN_GATHER_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "slot_proj_bwd_act#0"); layer = a["layers"]["slot_proj_bwd_act#0"]
        assert raw["loop_iterations"] == macs_ref and raw["loop_iterations_complete"] is True
        assert layer["macs"] == macs_ref and "status" not in layer and "compute_density" not in layer
        assert a["total"]["macs"] == macs_ref and "status" not in a["total"]
        # the as-written dense-over-E figure the target bills (every expert visited per slot) is NOT the layer's cost
        assert layer["macs"] != 2 * T2 * K2 * I2 * H2 * E2 and layer["macs"] != T2 * K2 * I2 * H2 * E2

# ─── Round 4 (independent review): branches are exclusive, not additive; recompute fingerprint restored ───────
_INNER_H = '            for h in Dim("H"):\n                Y[n, f] += X[n, h] * W[f, h]'
# the reviewer's dead-branch control, built exactly as probe_additional.py builds it
DEAD_IF_ELSE = DENSE_DSL.replace(_INNER_H, '            if False:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]')
LIVE_IF_ELSE = DENSE_DSL.replace(_INNER_H, '            if True:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]')
DATA_IF = DENSE_DSL.replace(_INNER_H, '            if X[n, 0] > 0:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]')
STATIC_CHAIN = DENSE_DSL.replace(_INNER_H, '            if False:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]\n            elif True:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]')
DATA_CHAIN = DENSE_DSL.replace(_INNER_H, '            if X[n, 0] > 100:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]\n            elif X[n, 0] > 0:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]')
ONE_ARM_IF = DENSE_DSL.replace(_INNER_H, '            if X[n, 0] > 0:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]')
NESTED_FUNCTION = DENSE_DSL.replace(_INNER_H, '            def accumulate():\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            accumulate()')
# an ``if`` that gates ASSIGNMENTS only (no loop under it): the as-written weight is the larger branch, never the sum
GATED_ACCUMULATION = DENSE_DSL.replace('                Y[n, f] += X[n, h] * W[f, h]',
                                       '                if X[n, h] > 0:\n                    Y[n, f] += X[n, h] * W[f, h]\n                else:\n                    Y[n, f] += X[n, h] * W[f, h]')


class TestBranchesAreExclusiveNotAdditive:
    """loop-completeness-4 (round 4): a loop nest under an ``if`` branch is never added to its sibling branch's
    nest and is never certified (no test is constant-folded): the kernel is incomplete at layer / total / perf with the
    branch recorded.  Executed work is asserted independently (``X @ W.T``)."""

    @pytest.mark.parametrize("name,code,expect_fragment", [
        ("dead_if_else", DEAD_IF_ELSE, "if False (line 10): loop nest under a branch (body 1 / else 1 paths)"),
        ("live_if_else", LIVE_IF_ELSE, "if True (line 10): loop nest under a branch (body 1 / else 1 paths)"),
        ("data_if_else", DATA_IF, "if X[n, 0] > 0 (line 10): loop nest under a branch (body 1 / else 1 paths)"),
        ("static_chain", STATIC_CHAIN, "if False (line 10): loop nest under a branch (body 1 / else 0 paths)"),   # the elif level is its own omission and yields no paths
        ("data_chain", DATA_CHAIN, "if X[n, 0] > 100 (line 10): loop nest under a branch (body 1 / else 0 paths)"),
        ("one_arm_if", ONE_ARM_IF, "if X[n, 0] > 0 (line 10): loop nest under a branch (body 1 / else 0 paths)"),
        ("nested_function", NESTED_FUNCTION, "FunctionDef accumulate (line 10): not visited by the loop counter"),
    ])
    def test_branch_nest_fails_closed(self, name, code, expect_fragment, tmp_path):
        out, _ = execute_translated(code, {}, all_inputs=INPUTS, trace=False, fast=False)
        np.testing.assert_allclose(out["output"], INPUTS["x"] @ INPUTS["w"].T)          # 24 MACs really executed
        md = _traced_metadata(code, INPUTS, profile=False)
        a = _analyze(code, md, tmp_path)
        raw = _graph_layer(tmp_path, "linear#0"); layer = a["layers"]["linear#0"]
        perf = EinsumGraphPerfModel().predict(tmp_path / "analysis" / "analysis.yaml", tmp_path / "perf",
                                              arch_config=str(ROOT / "configs" / "arch" / "H100_PCIe.yaml"))
        assert raw["loop_iterations_complete"] is False
        assert any(o.startswith(expect_fragment) for o in raw["loop_iterations_omitted"]), raw["loop_iterations_omitted"]
        assert raw.get("loop_iterations") != 48                                           # never the sum of both arms
        assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert a["total"]["status"] == "incomplete" and perf["workload"]["status"] == "incomplete"

    def test_chain_records_each_level_that_holds_a_nest(self):
        """The ``elif`` is a nested ``if`` in the ``else``: each level holding a nest is its own omission."""
        src = pathlib.Path(__file__).parent / "_tmp_chain_einsum_lib.py"
        try:
            src.write_text(DATA_CHAIN); info = extract_kernel_info(src)["linear"]
        finally:
            src.unlink()
        assert info["loop_paths_complete"] is False
        assert [o.split(" (line")[0] for o in info["loop_paths_omitted"]] == ["if X[n, 0] > 0", "if X[n, 0] > 100"]

    def test_gated_accumulation_keeps_the_as_written_weight(self, tmp_path):
        """An ``if`` with no loop under it only gates assignments: weight = larger branch (1), not the sum (2); the
        nest is complete and its 24 trips are billed once (the as-written dense convention; density, not the count,
        discounts masks).  Whether those trips are MACs or SM ops is the loop-expression extractor's classification
        (target code: it does not look inside the ``if``, so the accumulation is SM work here as on the plain target)."""
        md = _traced_metadata(GATED_ACCUMULATION, INPUTS, profile=False)
        a = _analyze(GATED_ACCUMULATION, md, tmp_path)
        raw = _graph_layer(tmp_path, "linear#0"); layer = a["layers"]["linear#0"]
        assert raw["loop_iterations"] == 24 and raw["loop_iterations_complete"] is True
        assert layer["macs"] + layer["other_ops"] == 24 and "status" not in layer

    def test_reviewer_probe_additional_verbatim(self, tmp_path):
        """probe_additional.py byte-for-byte (sha256 59a9e9c6…): nested_function incomplete, dead_if_else exact or
        incomplete, recomputed_moment classified as SM work, two_chain_gather exact — exit 0, every case satisfied."""
        (tmp_path / "head").mkdir()
        (tmp_path / "head" / "solar").symlink_to(ROOT)                    # the probe reads <root>/head/solar/tests/...
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "probe_additional.py"),
                              str(tmp_path), "head"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        records = json.loads((tmp_path / "runs" / "additional" / "head" / "RESULT.json").read_text())
        assert [r["case"] for r in records] == ["nested_function", "dead_if_else", "recomputed_moment", "two_chain_gather"]
        assert all(r["satisfied"] and r.get("numerical_output_correct") for r in records)
        by = {r["case"]: r for r in records}
        assert by["dead_if_else"]["all_three_incomplete"] and by["nested_function"]["all_three_incomplete"]
        assert by["two_chain_gather"]["layer"]["macs"] == 240 and not by["two_chain_gather"]["all_three_incomplete"]


REPEATED_INNER_PRODUCT = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
@kernel
def repeated_inner_product(X: Tensor("N,H",dtype="fp32"), W: Tensor("N,H",dtype="fp32")) -> Tensor("N,H"):
    Y=Tensor(N=X.shape[0],H=X.shape[1],dtype="fp32")
    for n in Dim("N"):
        for h in Dim("H"):
            acc=0.0
            for j in Dim("H"):
                acc += X[n,j]*W[n,j]
            Y[n,h]=acc
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N,H",dtype="fp32")
    w=Tensor.from_numpy(inputs["w"],"N,H",dtype="fp32")
    return {"output":repeated_inner_product(x,w).data}
'''


class TestRecomputeFingerprintRunsForCountedKernels:
    """recompute-7: the recompute-per-element fingerprint (a real-einsum nest that trips more than 1.5x the union
    of its tensors' dim extents is SM work, not a tensor-core contraction) runs for POSITIVE counts, as in round 2;
    round 3 had nested it under the no-count ``elif``."""

    def test_two_input_repeated_inner_product_is_sm_work(self, tmp_path):
        inputs = {"x": np.arange(8, dtype=np.float32).reshape(2, 4), "w": np.arange(8, 16, dtype=np.float32).reshape(2, 4)}
        out, _ = execute_translated(REPEATED_INNER_PRODUCT, {}, all_inputs=inputs, trace=False, fast=False)
        np.testing.assert_allclose(out["output"], np.repeat((inputs["x"] * inputs["w"]).sum(axis=1)[:, None], 4, axis=1))
        md = _traced_metadata(REPEATED_INNER_PRODUCT, inputs, profile=False)
        a = _analyze(REPEATED_INNER_PRODUCT, md, tmp_path)
        raw = _graph_layer(tmp_path, "repeated_inner_product#0"); layer = a["layers"]["repeated_inner_product#0"]
        assert raw["loop_iterations"] == 32 and raw["loop_iterations_complete"] is True and raw["is_real_einsum"] is False
        assert layer["macs"] == 0 and layer["other_ops"] == 32 and "status" not in layer

    def test_reviewer_probe_recompute2_verbatim(self, tmp_path):
        """probe_recompute2.py byte-for-byte (sha256 d04102db…): round-2 behaviour, 0 MACs / 32 SM other_ops."""
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "probe_recompute2.py"),
                              str(tmp_path), "head"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
        r = json.loads((tmp_path / "runs" / "recompute2" / "head" / "RESULT.json").read_text())
        layer = r["analysis"]["layers"]["repeated_inner_product#0"]
        assert r["numerical_output_correct"] and layer["macs"] == 0 and layer["other_ops"] == 32 and not layer.get("is_real_einsum")
        assert "status" not in layer and r["perf"]["workload"]["total_macs"] == 0

    def test_no_count_omission_path_still_propagates(self, tmp_path):
        """The restored placement keeps MARKER-5: the sole-``while`` kernel still carries its omission without a count."""
        code, inputs = EXTENDED_CONTROLS["single_while"][0], EXTENDED_CONTROLS["single_while"][1]
        md = _traced_metadata(code, inputs, profile=False)
        a = _analyze(code, md, tmp_path)
        raw = _graph_layer(tmp_path, "dot#0")
        assert "loop_iterations" not in raw and raw["loop_iterations_complete"] is False
        assert a["layers"]["dot#0"]["incomplete_reason"] == "loop_metadata_incomplete"

# ─── Round-4 addendum (independent review of the loop-evidence fix, shared with this branch): one loop-evidence gate; outer-level statements ──
from solar.analysis.graph_analyzer import dsl_loop_count_evidence, LoopCountEvidence  # noqa: E402

OUTER_STATEMENTS_DSL = '''
from solar.einsum_lib import Tensor,Dim,kernel,clear_dims
@kernel
def probe(X:Tensor("T,H"))->Tensor("T,H"):
 Y=Tensor(T=X.shape[0],H=X.shape[1])
 for t in Dim("T",parallel=True):
  for h in Dim("H"):
   s=X[t,h]+1
   for j in Dim("J",size=3):
    s=s+2
   Y[t,h]=s
 return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"T,H")
 return {"out":probe(x).data}
'''


class TestSharedLoopCountGate:
    """(a) ONE gate for the loop-count evidence — ``dsl_loop_count_evidence`` — used by the MAC basis here and by
    an earlier fix's other-ops basis: trusted iff a positive integer count, ``loop_iterations_complete`` True and no omissions;
    ``equation_faithful`` is reported, never gated on (the gathered-slot count IS the correction of an unfaithful
    equation)."""

    @pytest.mark.parametrize("layer,present,trusted,count", [
        ({}, False, False, 0),                                                          # bodiless builtin: nothing to judge
        ({"loop_iterations": 24, "loop_iterations_complete": True}, True, True, 24),
        ({"loop_iterations": 24, "loop_iterations_complete": True, "equation_faithful": False}, True, True, 24),   # slot kernels
        ({"loop_iterations": 24, "loop_iterations_complete": False, "loop_iterations_omitted": ["x"]}, True, False, 0),
        ({"loop_iterations": 24, "loop_iterations_complete": True, "loop_iterations_omitted": ["x"]}, True, False, 0),  # omission wins
        ({"loop_iterations": 24}, True, False, 0),                                      # flag not recorded (legacy)
        ({"loop_iterations": True, "loop_iterations_complete": True}, True, False, 0),  # bool is malformed
        ({"loop_iterations": "24", "loop_iterations_complete": True}, True, False, 0),  # a string is never a count (round 5: tolerance withdrawn)
        ({"loop_iterations": "abc", "loop_iterations_complete": True}, True, False, 0),
        ({"loop_iterations": 0, "loop_iterations_complete": True}, True, False, 0),
        ({"loop_iterations": np.int64(24), "loop_iterations_complete": True}, True, True, 24),   # an integral type is an integer
        ({"loop_iterations_complete": False, "loop_iterations_omitted": ["while"]}, True, False, 0),   # no count, omission carried
        ({"loop_iterations": None, "loop_iterations_complete": True}, True, False, 0),   # round 6: an explicit null is PRESENT and malformed
        ({"loop_iterations": None}, True, False, 0),
        ({"loop_iterations_complete": True}, False, False, 0),                           # key absent: unjudged, as before
    ])
    def test_gate(self, layer, present, trusted, count):
        ev = dsl_loop_count_evidence(layer)
        assert isinstance(ev, LoopCountEvidence)
        assert (ev.present, ev.trusted, ev.count) == (present, trusted, count)
        assert ev.trusted or ev.count == 0
        assert ev.trusted or not ev.present or ev.reasons

    def test_untrusted_reasons_name_the_cause(self):
        assert dsl_loop_count_evidence({"loop_iterations": True, "loop_iterations_complete": True}).reasons == ["loop_count_malformed:True"]
        assert dsl_loop_count_evidence({"loop_iterations": 5}).reasons == ["loop_iterations_complete not recorded"]
        assert dsl_loop_count_evidence({"loop_iterations": 5, "loop_iterations_complete": False, "loop_iterations_omitted": ["a", "b"]}).reasons == ["a", "b"]

    def test_mac_side_uses_the_gate(self, tmp_path):
        """The analyzer's MAC basis goes through the helper: a layer the gate rejects is billed by the equation with
        the gate's reasons; one it trusts is billed by the count (reviewer's unflagged / malformed controls kept)."""
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        g["layers"]["linear#0"]["loop_iterations"] = True
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        layer = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))["layers"]["linear#0"]
        assert layer["macs"] == 24 and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == ["loop_count_malformed:True"]


class TestOuterLevelStatementsAreCounted:
    """(b) statement-coverage-1: assignments at non-innermost loop levels are executed in-loop work.  They are
    recorded by the parser (``loop_outer_statements``), sized by the converter (``loop_statements`` = innermost weighted
    trips + outer statements), and — for a kernel that is NOT a reduction — become its count (assignment-execution
    convention).  A reduction kernel keeps ``loop_iterations`` = its innermost MAC trips (the gathered-slot 1,024)."""

    def test_parser_records_outer_statements(self):
        src = pathlib.Path(__file__).parent / "_tmp_outer_einsum_lib.py"
        try:
            src.write_text(OUTER_STATEMENTS_DSL); info = extract_kernel_info(src)["probe"]
        finally:
            src.unlink()
        assert info["loop_paths"] == [([["T", None], ["H", None], ["J", 3]], 1)] or info["loop_paths"] == [([("T", None), ("H", None), ("J", 3)], 1)]
        assert info["loop_outer_statements"] == [([("T", None), ("H", None)], 2)] or info["loop_outer_statements"] == [([["T", None], ["H", None]], 2)]
        assert info["loop_paths_complete"] is True

    def test_two_by_four_kernel_counts_forty(self, tmp_path):
        """The reviewer's kernel: 2x4 outputs, ``s = X + 1`` and ``Y = s`` at the h level (16) plus 3 inner trips (24)
        = 40 executed assignments; numerically 8 per element."""
        inputs = {"x": np.ones((2, 4))}
        out, _ = execute_translated(OUTER_STATEMENTS_DSL, {}, all_inputs=inputs, trace=False, fast=False)
        assert np.all(out["out"] == 8)
        md = _traced_metadata(OUTER_STATEMENTS_DSL, inputs, profile=False)
        a = _analyze(OUTER_STATEMENTS_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "probe#0"); layer = a["layers"]["probe#0"]
        assert raw["is_real_einsum"] is False
        assert raw["loop_statements"] == 40 and raw["loop_outer_statements"] == 16 and raw["loop_iterations"] == 40
        assert raw["loop_iterations_complete"] is True and "status" not in layer
        assert layer["macs"] == 0 and layer["other_ops"] == 40

    def test_reduction_kernel_keeps_its_mac_trips_and_exports_the_statement_total(self, tmp_path):
        """SLOT_DSL ``routed``: three statements per slot at the k level (``e = int(SE)``, ``acc = 0.0``,
        ``Y[n, e] = acc * RW``) = 3 * N * TOPK outer statements; MACs stay N * TOPK * H = 1,024."""
        inputs = _moe_inputs()
        md = _traced_metadata(SLOT_DSL, inputs, profile=True)
        a = _analyze(SLOT_DSL, md, tmp_path)
        raw = _graph_layer(tmp_path, "routed#0"); layer = a["layers"]["routed#0"]
        assert raw["is_real_einsum"] is True and raw["loop_iterations"] == N * TOPK * H == 1024
        assert raw["loop_outer_statements"] == 3 * N * TOPK and raw["loop_statements"] == 1024 + 3 * N * TOPK
        assert layer["macs"] == 1024 and "compute_density" not in layer

    def test_recompute_control_keeps_round_two_count(self, tmp_path):
        """The reclassified (fingerprint) kernel keeps its trip count of 32 as other_ops, as the round-4 task pins;
        its full statement total (acc init + write at the h level: 48) is exported alongside."""
        inputs = {"x": np.arange(8, dtype=np.float32).reshape(2, 4), "w": np.arange(8, 16, dtype=np.float32).reshape(2, 4)}
        md = _traced_metadata(REPEATED_INNER_PRODUCT, inputs, profile=False)
        a = _analyze(REPEATED_INNER_PRODUCT, md, tmp_path)
        raw = _graph_layer(tmp_path, "repeated_inner_product#0"); layer = a["layers"]["repeated_inner_product#0"]
        assert raw["loop_iterations"] == 32 and raw["loop_statements"] == 48 and raw["is_real_einsum"] is False
        assert layer["macs"] == 0 and layer["other_ops"] == 32

    def test_reviewer_statement_probes_verbatim(self, tmp_path):
        """the reviewer's probes.py (byte-for-byte, sha256 1119a04f…) run from a scratch copy: outer
        statements 40 exact; two statements 16 exact; the ``range(7)`` nest is refused (incomplete, not 56); an
        explicitly incomplete count is rejected by the gate; the density-0.25 mask kernel bills 2."""
        probe = pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "statement_probes.py"
        (tmp_path / "probes.py").write_bytes(probe.read_bytes()); (tmp_path / "evidence").mkdir()   # the probe writes evidence/ next to itself
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, str(tmp_path / "probes.py"), "head"], cwd=str(tmp_path), env=env,
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
        rec = json.loads((tmp_path / "evidence" / "probes-head.json").read_text())
        assert rec["outer_statements"]["exact_count_pass"] and rec["outer_statements"]["graph_fields"]["loop_iterations"] == 40
        assert rec["outer_statements"]["graph_fields"]["loop_iterations_complete"] is True
        assert rec["two_statements"]["exact_count_pass"]
        assert rec["range7"]["analysis"]["status"] == "incomplete" and rec["range7"]["analysis"]["other_ops"] != 56
        assert rec["explicit_incomplete"]["gate_rejects_incomplete_count"]
        assert rec["mask"]["discount_pass"] and rec["mask"]["analysis"]["compute_density"] == 0.25


# ─── Round 5 (independent review): the shared gate trusts only a positive bool-excluded integer ─────────────────
class TestSharedGateRejectsMalformedCounts:
    """shared-gate-type-8: a fractional float was truncated (7.9 -> 7 MACs billed, no marker) and +inf raised
    OverflowError out of the helper.  Now: anything that is not a ``numbers.Integral`` (bool excluded) > 0 is
    ``loop_count_malformed:<repr>`` — untrusted, count 0, never coerced, never raised — and the analyzer bills the
    generic basis with the incomplete marker through layer / total / perf."""

    REVIEWER_VALUES = [7.9, float("inf"), "24", 24.0, float("nan"), -1, True]          # the reviewer's seven, verbatim
    OWN_VALUES = [24 + 0j, [24], float("-inf"), np.float64(24.0)]                      # own: complex, list, -inf, numpy float

    @pytest.mark.parametrize("value", REVIEWER_VALUES + OWN_VALUES, ids=repr)
    def test_untrusted_with_a_reason_and_no_exception(self, value):
        ev = dsl_loop_count_evidence({"loop_iterations": value, "loop_iterations_complete": True})
        assert ev.present and not ev.trusted and ev.count == 0
        assert ev.reasons == [f"loop_count_malformed:{value!r}"]

    @pytest.mark.parametrize("value", [24, np.int64(24), 10 ** 15], ids=repr)
    def test_positive_integers_are_trusted(self, value):
        ev = dsl_loop_count_evidence({"loop_iterations": value, "loop_iterations_complete": True})
        assert ev.trusted and ev.count == int(value) and ev.reasons == []

    @pytest.mark.parametrize("value,expect_reason", [(7.9, "loop_count_malformed:7.9"), (float("inf"), "loop_count_malformed:inf"),
                                                     (float("nan"), "loop_count_malformed:nan"), (24.0, "loop_count_malformed:24.0")], ids=repr)
    def test_analyzer_and_perf_fall_back_with_the_marker(self, value, expect_reason, tmp_path):
        """End to end on the 24-MAC dense kernel: the malformed count is not billed (7.9 -> 7 was the defect); the
        equation basis (24) is used and the marker reaches layer, analysis total and perf workload."""
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        g["layers"]["linear#0"]["loop_iterations"] = value; g["layers"]["linear#0"]["loop_iterations_complete"] = True
        g["layers"]["linear#0"].pop("loop_iterations_omitted", None)
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))
        layer = a["layers"]["linear#0"]
        assert layer["macs"] == 24 and layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == [expect_reason] and a["total"]["status"] == "incomplete"
        perf = EinsumGraphPerfModel().predict(tmp_path / "analysis2" / "analysis.yaml", tmp_path / "perf2",
                                              arch_config=str(ROOT / "configs" / "arch" / "H100_PCIe.yaml"))
        assert perf["workload"]["status"] == "incomplete" and perf["workload"]["total_macs"] == 24

    def test_reviewer_extended_script_verbatim(self, tmp_path):
        """the reviewer's ``extended.py`` byte-for-byte (sha256 2f99f64e…): the four branch cases incomplete, the no-loop
        branch a conservative bound of 24, and the nine gate schema cases all satisfying the positive-integer contract
        (``trusted == (type(value) is int and value > 0)``) with no exception."""
        probe = pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "extended_round4.py"
        (tmp_path / "extended.py").write_bytes(probe.read_bytes()); (tmp_path / "evidence").mkdir()
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, str(tmp_path / "extended.py")], cwd=str(ROOT), env=env,
                             capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
        rec = json.loads((tmp_path / "evidence" / "extended-results.json").read_text())
        assert all(rec[c]["passed"] for c in ("if_true_both", "if_elif_else", "else_only", "data_dependent_reduction"))
        assert rec["no_loop_branch"]["sound_assignment_upper_bound"] and rec["no_loop_branch"]["complete"] is True
        gate = {g["case"]: g for g in rec["gate"]}
        assert all(g["positive_integer_contract_pass"] for g in gate.values()), {k: v.get("exception") for k, v in gate.items()}
        assert not any("exception" in g for g in gate.values())
        assert gate["fractional_float"]["billed"]["macs"] == 24 and gate["fractional_float"]["billed"]["status"] == "incomplete"
        assert gate["infinity"]["billed"]["status"] == "incomplete" and gate["infinity"]["perf_status"] == "incomplete"


# ─── Round 6 (independent review): presence is the KEY, an explicit null is malformed ────────────────────────────
class TestSharedGateNullCount:
    """shared-gate-type-8 residual: ``{'loop_iterations': None, ...}`` was treated as ABSENT evidence (present
    False, no reason, generic 24 MACs with no marker).  Presence is now the existence of the ``loop_iterations`` key;
    a present None is ``loop_count_malformed:None`` and the marker reaches layer / total / perf.  A genuinely absent
    key keeps the unmarked behaviour (bodiless builtins)."""

    def test_present_null_is_malformed_not_absent(self):
        ev = dsl_loop_count_evidence({"loop_iterations": None, "loop_iterations_complete": True})
        assert ev.present and not ev.trusted and ev.count == 0 and ev.reasons == ["loop_count_malformed:None"]

    def test_absent_key_is_unjudged(self):
        ev = dsl_loop_count_evidence({"loop_iterations_complete": True, "equation_faithful": True})
        assert not ev.present and not ev.trusted and ev.count == 0 and ev.reasons == []

    def test_null_in_a_saved_graph_bills_the_generic_basis_with_the_marker(self, tmp_path):
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        g["layers"]["linear#0"]["loop_iterations"] = None; g["layers"]["linear#0"]["loop_iterations_complete"] = True
        g["layers"]["linear#0"].pop("loop_iterations_omitted", None)
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        assert "loop_iterations: null" in gpath.read_text()                       # the key is really present, as null
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        a = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))
        layer = a["layers"]["linear#0"]
        assert layer["macs"] == 24 and layer["status"] == "incomplete" and layer["incomplete_reason"] == "loop_metadata_incomplete"
        assert layer["loop_iterations_omitted"] == ["loop_count_malformed:None"] and a["total"]["status"] == "incomplete"
        perf = EinsumGraphPerfModel().predict(tmp_path / "analysis2" / "analysis.yaml", tmp_path / "perf2",
                                              arch_config=str(ROOT / "configs" / "arch" / "H100_PCIe.yaml"))
        assert perf["workload"]["status"] == "incomplete" and perf["workload"]["total_macs"] == 24

    def test_absent_key_in_a_saved_graph_stays_unmarked(self, tmp_path):
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        for k in ("loop_iterations", "loop_iterations_complete", "loop_iterations_omitted"):
            g["layers"]["linear#0"].pop(k, None)
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        layer = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))["layers"]["linear#0"]
        assert layer["macs"] == 24 and "status" not in layer

    def test_huge_trusted_count_is_billed_exactly(self, tmp_path):
        """The reviewer's precision note: ``2**80 + 37`` must not be rounded through a float multiply when the
        sparsity fraction is 1.0 (exact-integer path)."""
        md = _traced_metadata(DENSE_DSL, INPUTS, profile=False)
        _analyze(DENSE_DSL, md, tmp_path)
        gpath = next((tmp_path / "einsum").glob("*.yaml")); g = yaml.safe_load(gpath.read_text())
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        # (issue #59) a tampered count that disagrees with PRESENT per-statement classes is invalid evidence and
        # fails closed; the exact-integer path is exercised without classes (legacy record) and with classes that
        # agree with the count
        lin = g["layers"]["linear#0"]
        lin["loop_iterations"] = 2 ** 80 + 37; lin["loop_iterations_complete"] = True
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        layer = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis_invalid"))["layers"]["linear#0"]
        assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "class_evidence_invalid" and layer["macs"] == 0
        for k in ("loop_mac_statements", "loop_other_statements", "loop_extra_multiplies"):
            lin.pop(k, None)
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        layer = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis2"))["layers"]["linear#0"]
        assert layer["macs"] == 2 ** 80 + 37 and "status" not in layer
        lin.update(loop_mac_statements=2 ** 80 + 37, loop_other_statements=0, loop_extra_multiplies=0, loop_statements=2 ** 80 + 37)
        # the MAC-operand record must agree with the count too (DSL-COMPUTE-DTYPE-BINDING-1)
        lin["loop_mac_operands"] = [{"macs": 2 ** 80 + 37, "operands": (lin.get("loop_mac_operands") or [{}])[0].get("operands", {})}]
        gpath.write_text(yaml.safe_dump(g, sort_keys=False))
        layer = EinsumGraphAnalyzer().analyze_graph(str(gpath), str(tmp_path / "analysis3"))["layers"]["linear#0"]
        assert layer["macs"] == 2 ** 80 + 37 and layer["unit_basis"] == "per_statement" and "status" not in layer

    def test_reviewer_round5_gate_file_verbatim(self, tmp_path):
        """The maintainers' ``test_gate.py`` (shipped copy) with its own saved dense graph: 21 helper-contract
        values (incl. ``none``), 13 layer/total/perf contracts, the absent and the omissions controls — all pass."""
        fx = pathlib.Path(__file__).parent / "fixtures" / "loop_evidence" / "round5_gate"
        (tmp_path / "inputs").mkdir(); (tmp_path / "source" / "head").mkdir(parents=True)
        (tmp_path / "test_gate.py").write_bytes((fx / "test_gate.py").read_bytes())
        (tmp_path / "round4-dense-graph.yaml").write_bytes((fx / "round4-dense-graph.yaml").read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path / "test_gate.py"), "-q", "-p", "no:cacheprovider"],
                             cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]
        assert "36 passed" in res.stdout, res.stdout[-400:]
