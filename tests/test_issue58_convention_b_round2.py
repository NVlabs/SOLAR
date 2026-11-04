# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""issue #58 convention B, round 2: regressions for the independently reviewed defects; each probe is a product assertion:

P2-conservation-1  the stored ``components_sum_to_aggregate`` flag is not arithmetic proof: a typed record whose components do not
                         sum to the aggregate, or whose aggregate differs from ``total.macs``, REFUSES with the original values (never a
                         zero tensor-core term, never an accepted 121-vs-120 aggregate);
P2-typed-presence-1 a PRESENT malformed ``total.macs_typed`` (null / false / [] / string) REFUSES with its path and value; only a
                         genuinely ABSENT field takes the legacy 'priced whole' path;
P2-term-label-1    the tensor-core term prices contraction MACs AND the analyzer's other non-contraction MACs; the receipt says so;
P2-hidden-wording-1 / P3-diagnostic-text-1  no 'already carried by memory' / 'double-count' claim in the active convention text;
                         the SM diagnostic prices other_ops only and the prose says so.
"""
import copy
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from solar.perf.perf_model import EinsumGraphPerfModel, PerfProjectionUnavailable, SOL_CONVENTION
from test_mac_report_provenance import MIXED, MIXED_INPUTS, ROOT, _analyze, _dsl_graph

UNIT_PROFILE = {"name": "unit_rates", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "MAC_per_cycle_fp16_tc": 1, "MAC_per_cycle_fp32_sm": 1}
NOTE = "multiply statements of non-einsum kernels counted as MACs by the per-statement classifier (unit_basis per_statement); not tensor-core contraction work"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A real analysis (MIXED fixture) whose total is then overwritten per probe, exactly as the reviewer did."""
    tmp = tmp_path_factory.mktemp("r2"); g = _dsl_graph(tmp, MIXED, MIXED_INPUTS); return _analyze(g, tmp / "analysis")


def _typed(aggregate, contraction, scalar, other=None, flag=True, scalar_layers=None, other_layers=None):
    t = {"aggregate": aggregate, "contraction": {"macs": contraction, "by_compute_dtype": ({"workload_precision": contraction} if contraction else {})},
         "scalar_statement_multiplies": {"macs": scalar, "layers": scalar_layers or {}, "note": NOTE}}
    if other is not None: t["non_contraction_other"] = {"macs": other, "layers": other_layers or {}}
    t["components_sum_to_aggregate"] = flag
    return t


def _with_total(base, total_macs, typed="ABSENT", other_ops=0):
    a = copy.deepcopy(base); t = {k: v for k, v in a["total"].items() if k not in ("macs_typed", "macs_by_compute_dtype")}
    t["macs"] = total_macs; t["other_ops"] = other_ops
    if typed != "ABSENT": t["macs_typed"] = typed
    a["total"] = t
    for L in a["layers"].values():
        L.pop("compute_dtype", None)
    return a


def _predict(tmp_path, analysis, name="case"):
    d = tmp_path / name; d.mkdir(); p = d / "analysis.yaml"; p.write_text(yaml.safe_dump(analysis, sort_keys=False))
    prof = d / "unit_rates.yaml"; prof.write_text(yaml.safe_dump(UNIT_PROFILE))
    return EinsumGraphPerfModel(dtype_bytes=True).predict(p, d / "perf", arch_config=str(prof), precision="fp16", copy_analysis=False), d / "perf"


def _refusal(tmp_path, analysis, name="case"):
    with pytest.raises(PerfProjectionUnavailable) as ei:
        _predict(tmp_path, analysis, name)
    receipt = yaml.safe_load((tmp_path / name / "perf" / "perf_unit_rates.REFUSED.yaml").read_text())
    assert not (tmp_path / name / "perf" / "perf_unit_rates.yaml").exists()
    return ei.value, receipt


# ── P2-conservation-1 ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_stale_true_flag_with_components_summing_to_240_refuses_instead_of_zero_tc_cycles(base, tmp_path):
    exc, receipt = _refusal(tmp_path, _with_total(base, 120, _typed(120, 120, 120, flag=True)))
    assert exc.reason == "typed_components_not_conserving" == receipt["reason"]
    d = receipt["compute_dtype_evidence_disagreement"]
    assert d["total_macs"] == 120 and d["aggregate"] == 120 and d["contraction_macs"] == 120 and d["scalar_statement_multiplies"] == 120 and d["components_sum"] == 240
    assert d["components_sum_to_aggregate_flag"] is True              # the stored flag was true and is recorded, not trusted


def test_typed_aggregate_121_against_total_120_refuses(base, tmp_path):
    exc, receipt = _refusal(tmp_path, _with_total(base, 120, _typed(121, 120, 0, flag=True)))
    assert exc.reason == "typed_components_not_conserving"
    d = receipt["compute_dtype_evidence_disagreement"]; assert d["aggregate"] == 121 and d["total_macs"] == 120


def test_valid_typed_control_is_priced(base, tmp_path):
    perf, _ = _predict(tmp_path, _with_total(base, 120, _typed(120, 120, 0)))
    assert perf["fused"]["compute_tc_cycles"] == 120 and perf["workload"]["tensor_core_term"]["applied"] is True


# ── P2-typed-presence-1 ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("container", [None, False, [], "broken"], ids=["null", "false", "list", "string"])
def test_present_malformed_macs_typed_container_refuses_with_path_and_value(base, tmp_path, container):
    exc, receipt = _refusal(tmp_path, _with_total(base, 120, container))
    assert exc.reason == "macs_typed_malformed" == receipt["reason"]
    assert "total.macs_typed" in receipt["malformed_counts"] and receipt["malformed_counts"]["total.macs_typed"] == (container if container != [] else [])
    assert receipt["total_macs"] == 120


def test_absent_macs_typed_is_the_only_legacy_path(base, tmp_path):
    perf, _ = _predict(tmp_path, _with_total(base, 120, "ABSENT"))
    term = perf["workload"]["tensor_core_term"]
    assert term["applied"] is False and "absent" in term["reason"] and "no total.macs_typed" not in term["reason"].replace("absent", "")
    assert perf["fused"]["compute_tc_cycles"] == 120


@pytest.mark.parametrize("mutate, reason, path", [
    (lambda t: t["contraction"].__setitem__("macs", "x"), "macs_typed_component_malformed", "total.macs_typed.contraction.macs"),
    (lambda t: t.__setitem__("aggregate", None), "macs_typed_component_malformed", "total.macs_typed.aggregate"),
    (lambda t: t["contraction"]["by_compute_dtype"].__setitem__("workload_precision", 1.5), "macs_typed_component_malformed", "total.macs_typed.contraction.by_compute_dtype.workload_precision"),
    (lambda t: t.__setitem__("non_contraction_other", {"macs": True}), "macs_typed_component_malformed", "total.macs_typed.non_contraction_other.macs"),
], ids=["contraction-string", "aggregate-null", "bucket-fractional", "other-bool"])
def test_every_typed_component_that_enters_pricing_is_validated(base, tmp_path, mutate, reason, path):
    t = _typed(120, 120, 0); mutate(t)
    exc, receipt = _refusal(tmp_path, _with_total(base, 120, t))
    assert exc.reason == reason and path in receipt["malformed_counts"]


# ── P2-term-label-1 ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_other_non_contraction_macs_are_priced_and_named(base, tmp_path):
    perf, out = _predict(tmp_path, _with_total(base, 120, _typed(120, 0, 0, other=120, other_layers={"dense#0": 120})))
    term = perf["workload"]["tensor_core_term"]
    assert perf["fused"]["compute_tc_cycles"] == 120 and term["macs_priced"] == 120
    assert term["contraction_macs_priced"] == 0 and term["non_contraction_other_macs_priced"] == 120 and term["scalar_statement_multiplies_excluded"] == 0
    text = json.dumps(perf["workload"]) + json.dumps(perf["sol_convention"])
    assert "only component" not in text and "contraction MACs only" not in text and "CONTRACTION MACs only" not in text
    assert "non_contraction_other" in json.dumps(perf["sol_convention"]) and "120" in perf["workload"]["default_rate_remainder"]["note"] and "other non-contraction" in perf["workload"]["default_rate_remainder"]["note"]
    on_disk = yaml.safe_load((out / "perf_unit_rates.yaml").read_text()); assert on_disk["workload"]["tensor_core_term"]["non_contraction_other_macs_priced"] == 120


def test_mixed_scalar_and_other_components_price_only_the_other(base, tmp_path):
    perf, _ = _predict(tmp_path, _with_total(base, 36, _typed(36, 0, 24, other=12, scalar_layers={"masked_sum#0": 24}, other_layers={"dense#0": 12}), other_ops=12))
    term = perf["workload"]["tensor_core_term"]
    assert perf["fused"]["compute_tc_cycles"] == 12 and term["scalar_statement_multiplies_excluded"] == 24 and term["non_contraction_other_macs_priced"] == 12


# ── P3-diagnostic-text-1 / P2-hidden-wording-1 ─────────────────────────────────────────────────────────────────
def test_sm_diagnostic_prices_other_ops_only_and_the_prose_says_so(base, tmp_path):
    a = _with_total(base, 24, _typed(24, 0, 24, scalar_layers={"masked_sum#0": 24}), other_ops=12)
    perf, out = _predict(tmp_path, a)
    assert perf["fused"]["compute_sm_cycles"] == 12                      # 12 other_ops at SM rate 1; the 24 scalar multiplies are NOT in it
    conv = json.dumps(perf["sol_convention"]); term = json.dumps(perf["workload"]["tensor_core_term"])
    for banned in ("both are retained as typed counts and in the informational SM diagnostic", "in the informational SM diagnostic", "and in compute_sm_cycles"):
        assert banned not in conv and banned not in term, banned
    assert "other_ops only" in conv or "prices other_ops only" in conv or "other_ops, not the scalar" in conv
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, "-m", "solar.cli.predict_perf_model", "--analysis-path", str(tmp_path / "case" / "analysis.yaml"), "--arch-config", str(tmp_path / "case" / "unit_rates.yaml"),
                        "--precision", "fp16", "--output-dir", str(tmp_path / "cli"), "--no-copy-analysis", "--dtype-bytes"], capture_output=True, text=True, env=env, timeout=170)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "optimistic contraction-and-memory bound" in r.stdout and "both retained as counts and in compute_sm_cycles" not in r.stdout
    assert "compute_sm_cycles" in r.stdout and "other_ops only" in r.stdout


def test_no_absorption_claim_in_the_active_convention_text():
    banned = ("already carried by the memory", "already captured by", "would double-count", "memory-bound in practice")
    pm = (ROOT / "solar" / "perf" / "perf_model.py").read_text()
    for b in banned: assert b not in pm, b
    guide = (ROOT / "docs" / "SOL_GUIDE.md").read_text(); head = guide.split("## 1. Unfused SOL")[0]   # roofline basics + convention
    for b in banned: assert b not in head, b
    assert "as above" not in head.split("### Convention")[1] if "### Convention" in head else True
    readme = (ROOT / "README.md").read_text(); cli = (ROOT / "solar" / "cli" / "predict_perf_model.py").read_text()
    for b in ("in the informational SM diagnostic", "and in compute_sm_cycles"): assert b not in readme and b not in cli and b not in json.dumps(SOL_CONVENTION), b
    assert "all priced at the APPLIED rate" not in pm
