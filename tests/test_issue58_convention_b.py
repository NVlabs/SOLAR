# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""issue #58 convention B (bound convention): scalar statement multiplies of non-einsum kernels are EXCLUDED
from the tensor-core timing term; their typed counts and the informational SM diagnostic are retained unchanged; the headline is
labelled an optimistic contraction-and-memory bound.  Nothing here claims scalar work is hidden or free.

The MIXED fixture (dense einsum + masked_sum scalar kernel) is the one the report-only round named for this test: the tensor-core
term must drop by exactly scalar / rate; a contraction-only workload is unchanged; an analysis that predates typed components is
priced whole and labelled so; a malformed or non-conserving typed component refuses (counts that enter pricing are validated).
"""
import copy
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from solar.perf.arch_rates import DEFAULT_SOL_MODE, resolve_arch_rates
from solar.perf.perf_model import EinsumGraphPerfModel, PerfProjectionUnavailable
from test_mac_report_provenance import H, MIXED, MIXED_INPUTS, ROOT, _analyze, _dsl_graph

ARCH = ROOT / "configs" / "arch"
HEADLINE = "optimistic contraction-and-memory bound"
UNIT_PROFILE = {"name": "unit_rates", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "MAC_per_cycle_fp16_tc": 1, "MAC_per_cycle_fp32_sm": 1}

DENSE_ONLY = H + '''@kernel
def dense(A: Tensor("M,K", dtype="fp32"), B: Tensor("K,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m, n] += A[m, k] * B[k, n]
    return Y

def run_model(inputs):
    clear_dims()
    return {"dense": dense(Tensor.from_numpy(inputs["a"], "M,K", dtype="fp32"), Tensor.from_numpy(inputs["b"], "K,N", dtype="fp32")).data}
'''


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mixed_b")
    g = _dsl_graph(tmp, MIXED, MIXED_INPUTS); a = _analyze(g, tmp / "analysis")
    assert a["total"]["macs_typed"]["scalar_statement_multiplies"]["macs"] > 0
    return a, tmp / "analysis" / "analysis.yaml"


@pytest.fixture(scope="module")
def dense_only(tmp_path_factory):
    import numpy as np
    tmp = tmp_path_factory.mktemp("dense_b")
    g = _dsl_graph(tmp, DENSE_ONLY, {"a": np.ones((6, 4), dtype=np.float32), "b": np.ones((4, 5), dtype=np.float32)})
    a = _analyze(g, tmp / "analysis")
    assert a["total"]["macs_typed"]["scalar_statement_multiplies"]["macs"] == 0
    return a, tmp / "analysis" / "analysis.yaml"


def _profile(profile, out):
    """A shipped profile by name, or the unit-rate profile (every rate 1): cycles == MACs, so exact-cycle assertions are not
    truncated to 0 by the toy fixture's few hundred MACs."""
    if profile == "unit":
        p = out / "unit_rates.yaml"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(UNIT_PROFILE)); return p
    return ARCH / profile


def _predict(analysis_path, out, profile="unit", precision="fp16"):
    perf = EinsumGraphPerfModel(dtype_bytes=True).predict(analysis_path, out, arch_config=str(_profile(profile, out.parent / "profiles")), precision=precision, copy_analysis=False)
    assert isinstance(perf, dict)
    return perf


def _rate(profile, key):
    """The rate ``_predict`` applies for ``key``: the unit profile's bare key, or a shipped profile resolved at the default SOL mode
    (issue #14: shipped profiles carry explicit ``_dense`` / ``_sparse`` keys only, so the bare key is read through the resolver)."""
    if profile == "unit":
        return UNIT_PROFILE[key]
    return resolve_arch_rates(yaml.safe_load((ARCH / profile).read_text()), DEFAULT_SOL_MODE)[key]


def _strip_typed(a):
    s = copy.deepcopy(a)
    s["total"] = {k: v for k, v in a["total"].items() if k not in ("macs_typed", "equation_unfaithful_layers", "binary_schedule_applicability")}
    s["layers"] = {k: {f: v for f, v in L.items() if f not in ("macs_kind", "equation_faithful", "binary_schedule_applicability")} for k, L in a["layers"].items()}
    return s


def _write(tmp_path, name, analysis):
    d = tmp_path / name; d.mkdir(); p = d / "analysis.yaml"; p.write_text(yaml.safe_dump(analysis, sort_keys=False)); return p


def test_tensor_core_term_prices_contraction_macs_only(mixed, tmp_path):
    a, path = mixed; typed = a["total"]["macs_typed"]; rate = _rate("unit", "MAC_per_cycle_fp16_tc")
    perf = _predict(path, tmp_path / "perf")
    scalar = typed["scalar_statement_multiplies"]["macs"]; contraction = typed["contraction"]["macs"]
    assert contraction + scalar == perf["workload"]["total_macs"]
    for mode in ("unfused", "fused", "fused_prefetched"):
        assert perf[mode]["compute_tc_cycles"] == int(contraction / rate), mode          # the scalar statements are not in the term
        assert perf[mode]["compute_cycles"] == perf[mode]["compute_tc_cycles"]
    term = perf["workload"]["tensor_core_term"]
    assert term["applied"] is True and term["convention"] == "issue58-B"
    assert term["scalar_statement_multiplies_excluded"] == scalar and term["macs_priced"] == contraction
    assert term["excluded_cycles_at_applied_rate"] == pytest.approx(scalar / rate)
    assert term["rate_key"] == "MAC_per_cycle_fp16_tc" == perf["arch"]["mac_per_cycle_key_applied"] and term["rate_class"] == "tensor_core"
    assert term["default_rate_macs_priced"] == contraction and term["quantized_dtype_macs_priced"] == 0


def test_exclusion_is_exactly_scalar_over_rate_against_the_untyped_prediction(mixed, tmp_path):
    a, path = mixed; typed = a["total"]["macs_typed"]; rate = _rate("unit", "MAC_per_cycle_fp16_tc")
    perf = _predict(path, tmp_path / "perf"); perf0 = _predict(_write(tmp_path, "stripped", _strip_typed(a)), tmp_path / "perf0")
    total = perf["workload"]["total_macs"]; scalar = typed["scalar_statement_multiplies"]["macs"]
    assert perf0["fused"]["compute_tc_cycles"] == total and perf["fused"]["compute_tc_cycles"] == total - scalar and scalar > 0
    assert perf0["fused"]["compute_tc_cycles"] - perf["fused"]["compute_tc_cycles"] == scalar == int(perf["workload"]["tensor_core_term"]["excluded_cycles_at_applied_rate"])
    # and on a shipped profile the same relation holds at its rate
    h = _predict(path, tmp_path / "h100", "H100_SXM.yaml"); h0 = _predict(_write(tmp_path, "stripped_h", _strip_typed(a)), tmp_path / "h100_0", "H100_SXM.yaml")
    r = _rate("H100_SXM.yaml", "MAC_per_cycle_fp16_tc")
    assert h0["fused"]["compute_tc_cycles"] == int(total / r) and h["fused"]["compute_tc_cycles"] == int((total - scalar) / r)
    assert h["workload"]["tensor_core_term"]["excluded_cycles_at_applied_rate"] == pytest.approx(scalar / r)
    # nothing but the compute-derived fields moves; every memory figure is identical
    for mode in ("unfused", "fused", "fused_prefetched"):
        moved = {k for k in perf[mode] if perf[mode][k] != perf0[mode][k]}
        assert moved <= {"compute_tc_cycles", "compute_cycles", "total_cycles", "runtime_ms", "bottleneck"}, (mode, moved)
        assert perf[mode]["memory_bytes"] == perf0[mode]["memory_bytes"] and perf[mode]["memory_cycles"] == perf0[mode]["memory_cycles"]


def test_typed_counts_and_sm_diagnostic_are_retained_unchanged(mixed, tmp_path):
    a, path = mixed; perf = _predict(path, tmp_path / "perf"); wl = perf["workload"]
    assert wl["macs_typed"] == a["total"]["macs_typed"] and wl["total_macs"] == a["total"]["macs"]
    assert wl["total_other_ops"] == a["total"]["other_ops"]
    sm = _rate("unit", "MAC_per_cycle_fp32_sm")
    for mode in ("unfused", "fused", "fused_prefetched"):
        assert perf[mode]["compute_sm_cycles"] == int(a["total"]["other_ops"] / sm)            # the SM diagnostic prices other_ops exactly as before
    rem = wl["default_rate_remainder"]
    assert rem["macs"] == wl["total_macs"] and rem["scalar_statement_multiplies"] == a["total"]["macs_typed"]["scalar_statement_multiplies"]["macs"]
    assert rem["macs_priced_in_tensor_core_term"] == rem["contraction_macs_at_workload_precision"]
    assert "excluded from the tensor-core timing term" in rem["note"] and "free" not in rem["note"] and "hidden" not in rem["note"]


def test_contraction_only_workload_is_unchanged(dense_only, tmp_path):
    a, path = dense_only; rate = _rate("unit", "MAC_per_cycle_fp16_tc")
    perf = _predict(path, tmp_path / "perf"); term = perf["workload"]["tensor_core_term"]
    assert term["applied"] is True and term["scalar_statement_multiplies_excluded"] == 0 and term["macs_priced"] == a["total"]["macs"]
    assert perf["fused"]["compute_tc_cycles"] == int(a["total"]["macs"] / rate)


def test_analysis_without_typed_components_is_priced_whole_and_labelled(mixed, tmp_path):
    a, path = mixed; rate = _rate("unit", "MAC_per_cycle_fp16_tc")
    perf0 = _predict(_write(tmp_path, "stripped", _strip_typed(a)), tmp_path / "perf0"); term = perf0["workload"]["tensor_core_term"]
    assert term["applied"] is False and "predates typed" in term["reason"] and term["scalar_statement_multiplies_excluded"] == 0
    assert term["macs_priced"] == perf0["workload"]["total_macs"] and perf0["fused"]["compute_tc_cycles"] == int(perf0["workload"]["total_macs"] / rate)
    assert perf0["sol_convention"]["headline"] == HEADLINE


def test_headline_is_labelled_an_optimistic_contraction_and_memory_bound(mixed, tmp_path):
    a, path = mixed; perf = _predict(path, tmp_path / "perf"); conv = perf["sol_convention"]
    assert conv["id"] == "issue58-B" and conv["headline"] == HEADLINE
    text = json.dumps(conv)                                   # json: no line wrapping, so substrings are checked verbatim
    assert "not a bound accounting for all compute resources" in text and "other_ops" in text and "scalar statement multiplies" in text
    assert "scalar work proven hidden" in conv["not_claimed"] and "scalar work free" in conv["not_claimed"]
    on_disk = yaml.safe_load((tmp_path / "perf" / "perf_unit_rates.yaml").read_text())
    assert on_disk["sol_convention"]["headline"] == HEADLINE and on_disk["workload"]["tensor_core_term"]["applied"] is True


def test_cli_prints_the_headline_label(mixed, tmp_path):
    a, path = mixed
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(path), "--arch-config", str(ARCH / "H100_SXM.yaml"),
                        "--precision", "fp16", "--output-dir", str(tmp_path / "cli"), "--no-copy-analysis", "--dtype-bytes"],
                       capture_output=True, text=True, env=env, timeout=170)
    assert r.returncode == 0, r.stdout + r.stderr
    # round 2 (P3-diagnostic-text-1): the line names what is priced and what is excluded, and says the SM diagnostic prices other_ops only
    assert HEADLINE in r.stdout and "scalar statement multiplies and" in r.stdout and "compute_sm_cycles prices other_ops only" in r.stdout, r.stdout


@pytest.mark.parametrize("mutate, reason", [
    (lambda t: t["scalar_statement_multiplies"].__setitem__("macs", "many"), "scalar_statement_component_malformed"),
    (lambda t: t["scalar_statement_multiplies"].__setitem__("macs", None), "scalar_statement_component_malformed"),
    (lambda t: t["scalar_statement_multiplies"].__setitem__("macs", True), "scalar_statement_component_malformed"),
    (lambda t: t.__setitem__("components_sum_to_aggregate", False), "typed_components_not_conserving"),
    (lambda t: t["scalar_statement_multiplies"].__setitem__("macs", t["aggregate"] + 1), "typed_components_not_conserving"),
], ids=["string", "null", "bool", "not-conserving-flag", "scalar-exceeds-remainder"])
def test_malformed_or_non_conserving_typed_component_refuses(mixed, tmp_path, mutate, reason):
    a, _ = mixed; bad = copy.deepcopy(a); mutate(bad["total"]["macs_typed"])
    p = _write(tmp_path, "bad", bad)
    with pytest.raises(PerfProjectionUnavailable) as ei:
        _predict(p, tmp_path / "perf")
    assert ei.value.reason == reason
    receipt = yaml.safe_load((tmp_path / "perf" / "perf_unit_rates.REFUSED.yaml").read_text())
    assert receipt["reason"] == reason and not (tmp_path / "perf" / "perf_unit_rates.yaml").exists()
    assert receipt["macs_typed"]["aggregate"] == a["total"]["macs"]          # the refusal still records the typed components
