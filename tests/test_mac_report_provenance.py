# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""issue #58 -- report provenance of the MAC aggregate and applicability of the schedule / equation fields.

An analysis ``total.macs`` is an AGGREGATE of tensor-core contraction MACs and of multiply statements the per-statement
classifier counts as MACs in non-einsum kernels (Lite 203: 881,134,080 = 880,803,840 nvfp4 contraction + 327,680 +
2,560 scalar statements).  The analyzer now emits the typed decomposition beside the aggregate (``total.macs_typed``,
per-layer ``macs_kind``), carries the converter's ``equation_faithful: false`` verdict into the analysis layer and a
total-level ``equation_unfaithful_layers`` list, and labels the binary-schedule fields with an applicability tag -- so an
excerpt that copies the total verbatim (the Lite analyzer child's ``a_child_fs.py`` keep list) can never show an
unfaithful descriptive schedule as a qualified one.  Every recorded count is preserved.  Since issue #58 convention B (2026-10-30)
the typed scalar component is excluded from the tensor-core timing term (tests/test_issue58_convention_b.py); the report paths here are unchanged.
"""
import json
import pathlib

import numpy as np
import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.analysis.mac_presentation import describe_macs, typed_mac_components

FX = pathlib.Path(__file__).parent / "fixtures"
GOLD_INDEX = json.loads((FX / "graph_analyzer_source_graph" / "INDEX.json").read_text())["cases"]
H = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n\n"


def _analyze(graph, out):
    a = EinsumGraphAnalyzer().analyze_graph(str(graph), str(out), precision="fp32", copy_graph=False)
    assert isinstance(a, dict)
    return a


def _dsl_graph(tmp_path, code, inputs):
    """Execute a toy DSL program, build its metadata, convert, and return the graph path (the statement-unit tests' recipe)."""
    from solar.einsum_lib import build_metadata, get_trace
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_lib.verify import execute_translated
    from solar.einsum_dsl.cli_einsum import translate_with_einsum
    execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, shape_only=False, fast=False)
    md = build_metadata(get_trace()); dsl = tmp_path / "dsl"; dsl.mkdir(exist_ok=True)
    (dsl / "toy_einsum_lib.py").write_text(code); write_metadata_yaml(md, str(dsl / "toy_einsum_lib.yaml"))
    return pathlib.Path(translate_with_einsum(str(dsl), str(tmp_path / "einsum")))


def _check_typed(a):
    t = a["total"]; typed = t["macs_typed"]
    c = typed["contraction"]["macs"]; s = typed["scalar_statement_multiplies"]["macs"]; o = typed.get("non_contraction_other", {}).get("macs", 0)
    assert typed["aggregate"] == t["macs"] == c + s + o and typed["components_sum_to_aggregate"] is True, typed
    assert sum(typed["contraction"]["by_compute_dtype"].values()) == c
    for lid, L in a["layers"].items():
        if L.get("macs"):
            assert L["macs_kind"] in ("contraction", "scalar_statement_multiplies", "non_contraction"), (lid, L["macs_kind"])
            assert (L["macs_kind"] == "contraction") == bool(L["is_real_einsum"]), lid
        else:
            assert "macs_kind" not in L, lid
    # the typed contraction buckets agree with the existing compute-dtype aggregate where one exists
    for dt, n in (t.get("macs_by_compute_dtype") or {}).items():
        assert typed["contraction"]["by_compute_dtype"].get(dt) == n, (dt, n)
    return typed


@pytest.mark.parametrize("name", sorted(GOLD_INDEX))
def test_typed_components_sum_to_the_aggregate_on_fixture_graphs(tmp_path, name):
    """Every analyzable in-repo fixture graph: contraction + scalar-statement (+ other) == macs; every MAC-bearing layer is typed."""
    a = _analyze(FX / GOLD_INDEX[name]["graph"], tmp_path / name)
    typed = _check_typed(a)
    assert typed["scalar_statement_multiplies"]["macs"] == 0           # these fixtures hold contractions only
    assert a["total"]["equation_unfaithful_layers"] == [] and a["total"]["binary_schedule_applicability"] == "descriptive_alternative_not_selected_basis"


MASKED = H + '''@kernel
def masked_sum(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        acc = 0.0
        for k in Dim("K"):
            if k < 2:
                acc += X[m, k] * W[m, k]
        Y[m] = acc
    return Y

def run_model(inputs):
    clear_dims()
    return {"output": masked_sum(Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32"), Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")).data}
'''
INPUTS = {"x": np.arange(1, 25, dtype=np.float32).reshape(6, 4), "w": np.ones((6, 4), dtype=np.float32)}


def test_scalar_statement_multiplies_are_typed_not_homogeneous_macs(tmp_path):
    """A non-einsum kernel whose multiply statements the classifier counts as MACs: the layer is ``scalar_statement_multiplies``
    and the total's typed block names it with its count; the aggregate itself is unchanged (report-only)."""
    a = _analyze(_dsl_graph(tmp_path, MASKED, INPUTS), tmp_path / "analysis")
    L = next(v for k, v in a["layers"].items() if k.startswith("masked_sum#"))
    assert L["is_real_einsum"] is False and L["macs"] > 0 and L["macs_kind"] == "scalar_statement_multiplies" and L["unit_basis"] == "per_statement"
    typed = _check_typed(a)
    assert typed["contraction"]["macs"] == 0 and typed["scalar_statement_multiplies"]["macs"] == L["macs"] == a["total"]["macs"]
    assert list(typed["scalar_statement_multiplies"]["layers"]) == [next(k for k in a["layers"] if k.startswith("masked_sum#"))]
    assert "scalar statement multiplies" in describe_macs(a["total"]) and "aggregate" in describe_macs(a["total"])


def _unfaithful_copy(src_graph, dst, layer_id):
    raw = yaml.safe_load(pathlib.Path(src_graph).read_text()); raw["layers"][layer_id]["equation_faithful"] = False
    dst.write_text(yaml.safe_dump(raw, sort_keys=False)); return dst


def test_equation_unfaithful_marker_reaches_layer_and_total(tmp_path):
    """The converter's ``equation_faithful: false`` on a graph node appears on the analysis layer and in the total-level
    ``equation_unfaithful_layers`` (parallel to ``multi_operand_einsum_layers``); the count is the loop-count witness either way."""
    src = FX / "rnn_state_consumers/round8_matrix_valid.yaml"; base = _analyze(src, tmp_path / "base")
    target = next(k for k, v in base["layers"].items() if v["is_real_einsum"])
    a = _analyze(_unfaithful_copy(src, tmp_path / "einsum_graph_unfaithful.yaml", target), tmp_path / "unfaithful")
    assert a["layers"][target]["equation_faithful"] is False and "equation_faithful" not in base["layers"][target]
    assert a["total"]["equation_unfaithful_layers"] == [target] and base["total"]["equation_unfaithful_layers"] == []
    assert a["total"]["binary_schedule_applicability"].startswith("descriptive_alternative_not_selected_basis; equation_unfaithful_layers")
    for k in ("macs", "other_ops", "flops", "macs_typed"):                 # report-only: the counts do not move
        assert a["total"][k] == base["total"][k], k
    assert a["layers"][target]["macs"] == base["layers"][target]["macs"]


THREE_OPERAND = H + '''@kernel
def chain3(A: Tensor("T,H", dtype="fp32"), B: Tensor("H,S", dtype="fp32"), C: Tensor("S,E", dtype="fp32")) -> Tensor("T,E"):
    Y = Tensor(T=A.shape[0], E=C.shape[1], dtype="fp32")
    for t in Dim("T", parallel=True):
        for e in Dim("E"):
            for h in Dim("H"):
                for s in Dim("S"):
                    Y[t, e] += A[t, h] * B[h, s] * C[s, e]
    return Y

def run_model(inputs):
    clear_dims()
    return {"output": chain3(Tensor.from_numpy(inputs["a"], "T,H", dtype="fp32"), Tensor.from_numpy(inputs["b"], "H,S", dtype="fp32"), Tensor.from_numpy(inputs["c"], "S,E", dtype="fp32")).data}
'''
INPUTS3 = {"a": np.ones((2, 3), dtype=np.float32), "b": np.ones((3, 4), dtype=np.float32), "c": np.ones((4, 5), dtype=np.float32)}


def test_schedule_fields_carry_the_applicability_tag(tmp_path):
    """A multi-operand einsum layer emits the labelled binary-schedule alternatives; they now carry an applicability tag,
    and when the exported equation is not faithful the tag says so -- a reader of the layer alone cannot mistake the
    schedule for a qualified cost basis."""
    g = _dsl_graph(tmp_path, THREE_OPERAND, INPUTS3); a = _analyze(g, tmp_path / "faithful")
    lid = next(k for k in a["layers"] if k.startswith("chain3#")); L = a["layers"][lid]
    if "macs_dense_binary_schedule" not in L:
        pytest.skip("this tree emits no binary schedule for the three-operand kernel; nothing to tag")
    assert L["binary_schedule_applicability"] == "descriptive_alternative_not_selected_basis" and lid in a["total"]["multi_operand_einsum_layers"]
    b = _analyze(_unfaithful_copy(g, tmp_path / "einsum_graph_unfaithful3.yaml", lid), tmp_path / "unfaithful")
    assert b["layers"][lid]["equation_faithful"] is False and b["layers"][lid]["binary_schedule_applicability"] == "descriptive_alternative_of_unfaithful_equation"
    assert b["total"]["equation_unfaithful_layers"] == [lid] and b["total"]["macs"] == a["total"]["macs"]


def test_excerpt_that_copies_the_total_verbatim_keeps_the_markers(tmp_path):
    """K9: the Lite analyzer child copies a fixed subset of layer fields but the TOTAL verbatim; the total-level markers and the
    typed components therefore reach an excerpt-only reader even when the per-layer schedule fields are dropped."""
    src = FX / "rnn_state_consumers/round8_matrix_valid.yaml"; base = _analyze(src, tmp_path / "base")
    target = next(k for k, v in base["layers"].items() if v["is_real_einsum"])
    a = _analyze(_unfaithful_copy(src, tmp_path / "g.yaml", target), tmp_path / "unf")
    keep = ("einsum_equation", "is_real_einsum", "macs", "other_ops", "status")              # the child's style of keep list
    excerpt = {"layers": {k: {f: v[f] for f in keep if f in v} for k, v in a["layers"].items()}, "total": dict(a["total"])}
    assert "equation_faithful" not in excerpt["layers"][target]                                # dropped by the keep list ...
    assert excerpt["total"]["equation_unfaithful_layers"] == [target]                           # ... but the total names it
    assert excerpt["total"]["macs_typed"]["aggregate"] == excerpt["total"]["macs"]
    assert "equation not faithful for " + target in describe_macs(excerpt["total"])


def test_describe_macs_renders_the_lite203_decomposition_exactly():
    total = {"macs": 881134080, "macs_typed": {"aggregate": 881134080, "contraction": {"macs": 880803840, "by_compute_dtype": {"nvfp4": 880803840}},
                                                "scalar_statement_multiplies": {"macs": 330240, "layers": {"group_top2_sum#0": 327680, "group_topk_mask#0": 2560}}, "components_sum_to_aggregate": True},
             "equation_unfaithful_layers": ["router_scaled_mm#0"]}
    assert describe_macs(total) == ("881,134,080 MACs aggregate = 880,803,840 contraction (nvfp4 880,803,840) + 330,240 scalar statement multiplies "
                                    "(group_top2_sum#0 327,680, group_topk_mask#0 2,560)  [equation not faithful for router_scaled_mm#0: schedule fields descriptive only]")
    assert describe_macs({"macs": 42}) == "42 MACs (untyped aggregate: contraction MACs and scalar multiply statements not separated by this analysis)"
    assert typed_mac_components({"macs": 42}) is None


# ---- report paths: no untyped aggregate, and the perf receipt's default-rate remainder is decomposed -----------------------------------
import ast as _ast

REPORT_MODULES = ["solar/cli/analyze_model.py", "solar/cli/translate.py", "solar/estimate.py", "solar/translate/__init__.py", "solar/frontend/dispatcher.py", "solar/perf/perf_model.py"]
REPORT_SCRIPTS = ["scripts/run_execbench_problem.py", "scripts/reprice_execbench.py"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
# The two ExecBench report scripts are not part of the public distribution: their AST checks are skipped when absent.
REPORT_SCRIPT_PARAMS = [pytest.param(rel, marks=pytest.mark.skipif(not (ROOT / rel).exists(), reason=rel + " is not shipped in this distribution")) for rel in REPORT_SCRIPTS]


def _report_violations(path):
    """Statements that PRINT a MAC figure must render it through ``describe_macs``; dict literals that record ``total_macs``
    must record ``macs_typed`` beside it (an excerpt writer never emits the aggregate alone)."""
    src = path.read_text(); tree = _ast.parse(src); bad = []
    parents = {c: p for p in _ast.walk(tree) for c in _ast.iter_child_nodes(p)}
    def stmt_of(n):
        while n is not None and not isinstance(n, _ast.stmt): n = parents.get(n)
        return n
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "print":
            text = _ast.get_source_segment(src, n) or ""
            if ("macs" in text or "MAC" in text) and "describe_macs" not in text:
                st = stmt_of(n); seg = _ast.get_source_segment(src, st) or text
                if "describe_macs" not in seg and ("'macs'" in seg or '"macs"' in seg or "total_macs" in seg or ".macs" in seg or "MACs" in seg):
                    bad.append(f"{path.name}:{n.lineno}: prints a MAC figure without describe_macs: {text[:80]}")
        if isinstance(n, _ast.Dict):
            keys = [k.value for k in n.keys if isinstance(k, _ast.Constant) and isinstance(k.value, str)]
            if "total_macs" in keys:
                outer = n                                                    # a `**({...} if ...)` splat is its own Dict node:
                q = parents.get(n)                                           # judge the outermost enclosing dict literal
                while q is not None:
                    if isinstance(q, _ast.Dict): outer = q
                    q = parents.get(q)
                seg = _ast.get_source_segment(src, outer) or ""
                if "macs_typed" not in seg:
                    bad.append(f"{path.name}:{n.lineno}: records total_macs without macs_typed")
    return bad


@pytest.mark.parametrize("rel", REPORT_MODULES + REPORT_SCRIPT_PARAMS)
def test_no_report_path_prints_an_untyped_mac_aggregate(rel):
    """AST check over every report / excerpt writer: a MAC figure is printed only through describe_macs and recorded only
    beside its typed components."""
    assert _report_violations(ROOT / rel) == [], _report_violations(ROOT / rel)


def test_ast_check_catches_a_bare_print_and_a_bare_record(tmp_path):
    bad = tmp_path / "bad.py"; bad.write_text("total = {}\nprint(f'MACs: {total.get(\"macs\", 0)}')\nsummary = {'total_macs': 1, 'sol_ms': 2}\n")
    v = _report_violations(bad); assert len(v) == 2 and "without describe_macs" in v[0] and "without macs_typed" in v[1], v
    good = tmp_path / "good.py"; good.write_text("from solar.analysis.mac_presentation import describe_macs\nprint(f'MACs: {describe_macs(total)}')\ns = {'total_macs': 1, 'macs_typed': t}\n")
    assert _report_violations(good) == []


MIXED = H + '''@kernel
def dense(A: Tensor("M,K", dtype="fp32"), B: Tensor("K,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m, n] += A[m, k] * B[k, n]
    return Y

@kernel
def masked_sum(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        acc = 0.0
        for k in Dim("K"):
            if k < 2:
                acc += X[m, k] * W[m, k]
        Y[m] = acc
    return Y

def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["a"], "M,K", dtype="fp32"); b = Tensor.from_numpy(inputs["b"], "K,N", dtype="fp32")
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32"); w = Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")
    return {"dense": dense(a, b).data, "masked": masked_sum(x, w).data}
'''
MIXED_INPUTS = {"a": np.ones((6, 4), dtype=np.float32), "b": np.ones((4, 5), dtype=np.float32), "x": np.ones((6, 4), dtype=np.float32), "w": np.ones((6, 4), dtype=np.float32)}


def test_perf_receipt_decomposes_the_default_rate_remainder_and_prices_only_its_contraction_macs(tmp_path):
    """The perf receipt carries macs_typed and a default_rate_remainder decomposition (contraction MACs at the workload precision
    + scalar statement multiplies with their layers).

    Fixture justification (issue #58 convention B, authorised 2026-10-30): this test formerly asserted that the typed fields change
    NO cycle ("report-only").  Under convention B the typed scalar component is EXCLUDED from the tensor-core timing term, so the
    prediction made from the typed analysis differs from the one made with the typed fields stripped by exactly
    scalar_statement_multiplies / applied rate (on H100 fp32 the few hundred toy MACs truncate to the same integer cycles -- the exact
    relation is asserted at unit rates in tests/test_issue58_convention_b.py).  Every memory figure, every count and the SM
    diagnostic are identical with or without the typed fields; the stripped (legacy) analysis is priced whole and labelled so."""
    import yaml as _yaml
    from solar.perf.perf_model import EinsumGraphPerfModel
    g = _dsl_graph(tmp_path, MIXED, MIXED_INPUTS); a = _analyze(g, tmp_path / "analysis")
    typed = a["total"]["macs_typed"]; assert typed["contraction"]["macs"] > 0 and typed["scalar_statement_multiplies"]["macs"] > 0
    arch = ROOT / "configs" / "arch" / "H100_SXM.yaml"            # the python root holds configs/arch
    assert arch.exists(), arch
    perf = EinsumGraphPerfModel(dtype_bytes=True).predict(tmp_path / "analysis" / "analysis.yaml", tmp_path / "perf", arch_config=str(arch), precision="fp32", copy_analysis=False)
    wl = perf["workload"]; assert wl["macs_typed"] == typed and wl["total_macs"] == typed["aggregate"]
    rem = wl["default_rate_remainder"]; assert rem["macs"] == wl["total_macs"] and rem["contraction_macs_at_workload_precision"] == typed["contraction"]["macs"]
    assert rem["scalar_statement_multiplies"] == typed["scalar_statement_multiplies"]["macs"] and set(rem["scalar_statement_layers"]) == set(typed["scalar_statement_multiplies"]["layers"])
    assert "scalar multiply statements" in rem["note"] and rem["rate_key"].startswith("MAC_per_cycle_")
    assert rem["macs_priced_in_tensor_core_term"] == typed["contraction"]["macs"] == wl["tensor_core_term"]["macs_priced"]
    stripped = dict(a); stripped["total"] = {k: v for k, v in a["total"].items() if k not in ("macs_typed", "equation_unfaithful_layers", "binary_schedule_applicability")}
    stripped["layers"] = {k: {f: v for f, v in L.items() if f not in ("macs_kind", "equation_faithful", "binary_schedule_applicability")} for k, L in a["layers"].items()}
    (tmp_path / "stripped").mkdir(); (tmp_path / "stripped" / "analysis.yaml").write_text(_yaml.safe_dump(stripped, sort_keys=False))
    perf0 = EinsumGraphPerfModel(dtype_bytes=True).predict(tmp_path / "stripped" / "analysis.yaml", tmp_path / "perf0", arch_config=str(arch), precision="fp32", copy_analysis=False)
    rate = perf["arch"]["MAC_per_cycle"]; scalar = typed["scalar_statement_multiplies"]["macs"]
    for mode in ("unfused", "fused", "fused_prefetched"):
        for k in ("memory_elements", "memory_bytes", "memory_cycles", "compute_sm_cycles", "arithmetic_intensity"):
            assert perf[mode][k] == perf0[mode][k], (mode, k)
        assert perf0[mode]["compute_tc_cycles"] == int(wl["total_macs"] / rate) and perf[mode]["compute_tc_cycles"] == int((wl["total_macs"] - scalar) / rate), mode
    assert perf0["workload"]["tensor_core_term"]["applied"] is False and perf0["workload"]["tensor_core_term"]["macs_priced"] == wl["total_macs"]
    assert "macs_typed" not in perf0["workload"] and "predates typed" in perf0["workload"]["default_rate_remainder"]["note"]
