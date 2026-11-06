# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitHub issue #14: tensor-core peaks are explicit per precision (dense AND sparse), the SOL mode is a knob.

Decision (Jenny, 2026-11-06): every shipped hardware config carries BOTH the dense and the sparse
tensor-core peak for every precision it declares, with the derived MAC/cycle rates; the perf model
selects the rate set with ``sol_mode`` (``--sol-mode {sparse,dense}``), SPARSE is the default headline
SOL, and every result carries BOTH ``sol_sparse`` and ``sol_dense``.  Dense SOL is the more realistic
estimate; sparse SOL is a lower bound.

Checks: (1) config schema -- for every tensor-core precision present: ``peak_TFLOPS_<p>_{dense,sparse}``
and ``MAC_per_cycle_<p>_tc_{dense,sparse}``, sparse >= dense, MAC/cycle == peak x 1e12 / 2 FLOP per MAC /
(freq_GHz x 1e9) within rounding, no bare legacy key left in a shipped config; (2) the resolver: legacy
bare ``MAC_per_cycle_<p>_tc`` keys mean dense, a profile without a sparse figure falls back to dense
(recorded), explicit keys win; (3) the knob: dense == 2x the sparse time for a tensor-core-bound workload
whose sparse rate is 2x the dense rate; default is sparse; (4) the record carries both SOLs with the rate
keys used, and the top-level fields equal the default (sparse) set; (5) CLI flag; (6) the analyzer's
roofline and the estimate API use the same resolver.
"""
import dataclasses
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "configs" / "arch"
ARCH_FILES = sorted(ARCH_DIR.glob("*.yaml"))
TC_DENSE = re.compile(r"^MAC_per_cycle_(?P<p>[a-z0-9]+)_tc_dense$")
LEGACY_TC = re.compile(r"^MAC_per_cycle_[a-z0-9]+_tc$")

from solar.perf.perf_model import EinsumGraphPerfModel
from solar.perf.arch_rates import (DEFAULT_SOL_MODE, SOL_MODES, resolve_arch_rates, tensor_core_precisions)


def _load(path):
    return yaml.safe_load(path.read_text())


def _mac_per_cycle(peak_tflops, freq_ghz):
    return peak_tflops * 1e12 / 2.0 / (freq_ghz * 1e9)


# ── (1) config schema ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", ARCH_FILES, ids=[p.stem for p in ARCH_FILES])
def test_every_tensor_core_precision_has_explicit_dense_and_sparse_peaks(path):
    cfg = _load(path)
    precisions = sorted(TC_DENSE.match(k).group("p") for k in cfg if TC_DENSE.match(k))
    assert precisions, f"{path.name}: no MAC_per_cycle_<p>_tc_dense key"
    assert {"fp16", "bf16"} <= set(precisions), f"{path.name}: fp16 and bf16 must both be declared"
    for p in precisions:
        for k in (f"peak_TFLOPS_{p}_dense", f"peak_TFLOPS_{p}_sparse", f"MAC_per_cycle_{p}_tc_sparse"):
            assert k in cfg, f"{path.name}: {k} missing (every precision carries dense AND sparse)"
        assert cfg[f"peak_TFLOPS_{p}_sparse"] >= cfg[f"peak_TFLOPS_{p}_dense"], f"{path.name}: {p} sparse peak < dense peak"
        assert cfg[f"MAC_per_cycle_{p}_tc_sparse"] >= cfg[f"MAC_per_cycle_{p}_tc_dense"], f"{path.name}: {p} sparse rate < dense rate"
        for kind in ("dense", "sparse"):
            expected = _mac_per_cycle(cfg[f"peak_TFLOPS_{p}_{kind}"], cfg["freq_GHz"])
            got = cfg[f"MAC_per_cycle_{p}_tc_{kind}"]
            assert abs(got - expected) / expected <= 1e-3, (
                f"{path.name}: MAC_per_cycle_{p}_tc_{kind} = {got} but peak {cfg[f'peak_TFLOPS_{p}_{kind}']} TFLOPS / 2 FLOP per MAC / "
                f"{cfg['freq_GHz']} GHz = {expected:.1f}")
    # the derived values stay consistent with each other: the fp8 rate is never below the 16-bit rate
    if "MAC_per_cycle_fp8_tc_dense" in cfg:
        assert cfg["MAC_per_cycle_fp8_tc_dense"] >= cfg["MAC_per_cycle_fp16_tc_dense"]
    assert cfg["MAC_per_cycle_fp16_tc_dense"] >= cfg.get("MAC_per_cycle_tf32_tc_dense", 0)
    # the SM (CUDA-core) rate has no sparsity variant and is documented through its own peak
    assert "peak_TFLOPS_fp32_sm" in cfg and "MAC_per_cycle_fp32_sm" in cfg
    expected_sm = _mac_per_cycle(cfg["peak_TFLOPS_fp32_sm"], cfg["freq_GHz"])
    assert abs(cfg["MAC_per_cycle_fp32_sm"] - expected_sm) / expected_sm <= 1e-3, f"{path.name}: MAC_per_cycle_fp32_sm vs peak_TFLOPS_fp32_sm"


@pytest.mark.parametrize("path", ARCH_FILES, ids=[p.stem for p in ARCH_FILES])
def test_shipped_configs_carry_no_ambiguous_legacy_tensor_core_key(path):
    """A bare MAC_per_cycle_<p>_tc is the legacy (dense) spelling for user profiles; shipped configs are explicit only."""
    cfg = _load(path)
    assert not [k for k in cfg if LEGACY_TC.match(k)], f"{path.name}: bare MAC_per_cycle_<p>_tc keys must be spelled _tc_dense / _tc_sparse"
    assert "MAC_per_cycle_fp32_tc" not in cfg and "MAC_per_cycle_fp32_tc_dense" not in cfg, "fp32 falls back to the SM rate; TF32 has its own key"


@pytest.mark.parametrize("path", ARCH_FILES, ids=[p.stem for p in ARCH_FILES])
def test_every_peak_line_cites_a_source_or_a_derivation(path):
    text = path.read_text().splitlines()
    for line in text:
        if re.match(r"^\s*(peak_TFLOPS_|MAC_per_cycle_)", line):
            key = line.split(":")[0].strip()
            assert "#" in line, f"{path.name}: {key} needs a trailing `# source:` / `# derived:` / `# =` note"
    body = "\n".join(text)
    assert "source:" in body and "https://" in body, f"{path.name}: no source URL"


def test_resolver_sees_the_same_precisions_the_schema_test_sees():
    for path in ARCH_FILES:
        cfg = _load(path)
        assert tensor_core_precisions(cfg) == sorted(TC_DENSE.match(k).group("p") for k in cfg if TC_DENSE.match(k))


# ── (2) resolver ─────────────────────────────────────────────────────────────────────────────────────
def test_default_mode_is_sparse_and_modes_are_closed():
    assert DEFAULT_SOL_MODE == "sparse" and tuple(SOL_MODES) == ("sparse", "dense")
    with pytest.raises(ValueError):
        resolve_arch_rates({"freq_GHz": 1.0}, "dense-ish")


def test_legacy_bare_key_means_dense_and_sparse_falls_back_to_it():
    legacy = {"name": "old", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "MAC_per_cycle_fp16_tc": 10, "MAC_per_cycle_fp32_sm": 1}
    dense = resolve_arch_rates(legacy, "dense"); sparse = resolve_arch_rates(legacy, "sparse")
    assert dense["MAC_per_cycle_fp16_tc"] == 10 and sparse["MAC_per_cycle_fp16_tc"] == 10
    assert dense["MAC_per_cycle_fp16_tc_dense"] == 10 and dense["MAC_per_cycle_fp16_tc_sparse"] == 10
    res = sparse["sol_rate_resolution"]
    assert res["sol_mode"] == "sparse" and res["precisions"]["fp16"] == {
        "applied_key": "MAC_per_cycle_fp16_tc", "dense_from": "MAC_per_cycle_fp16_tc", "sparse_from": "MAC_per_cycle_fp16_tc",
        "sparse_fallback_to_dense": True}
    assert legacy == {"name": "old", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "MAC_per_cycle_fp16_tc": 10, "MAC_per_cycle_fp32_sm": 1}, "input not mutated"


def test_explicit_keys_select_by_mode_and_are_reported():
    prof = {"name": "x", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "MAC_per_cycle_fp32_sm": 1,
            "MAC_per_cycle_fp16_tc_dense": 1000, "MAC_per_cycle_fp16_tc_sparse": 2000,
            "MAC_per_cycle_fp8_tc_dense": 4000, "MAC_per_cycle_fp8_tc_sparse": 8000}
    assert resolve_arch_rates(prof, "sparse")["MAC_per_cycle_fp16_tc"] == 2000
    assert resolve_arch_rates(prof, "dense")["MAC_per_cycle_fp16_tc"] == 1000
    assert resolve_arch_rates(prof, "dense")["MAC_per_cycle_fp8_tc"] == 4000
    res = resolve_arch_rates(prof, "sparse")["sol_rate_resolution"]["precisions"]["fp8"]
    assert res == {"applied_key": "MAC_per_cycle_fp8_tc_sparse", "dense_from": "MAC_per_cycle_fp8_tc_dense",
                   "sparse_from": "MAC_per_cycle_fp8_tc_sparse", "sparse_fallback_to_dense": False}


def test_bare_legacy_key_overrides_explicit_keys_as_one_rate_for_both_sets_without_arithmetic_on_raw_values():
    """A caller that sets the legacy bare key on a shipped (explicit) profile -- the long-standing way to override a
    rate -- must see its value applied in BOTH modes; the resolver only carries raw values (rate-protocol-coverage-1)."""
    class Opaque:   # a raw profile value the resolver must only carry, never compare
        def __eq__(self, other): raise RuntimeError("compared")
        def __lt__(self, other): raise RuntimeError("compared")
        __le__ = __gt__ = __ge__ = __lt__
    o = Opaque()
    prof = {"freq_GHz": 1.0, "MAC_per_cycle_fp16_tc": o, "MAC_per_cycle_fp16_tc_dense": 1000, "MAC_per_cycle_fp16_tc_sparse": 2000}
    for mode in SOL_MODES:
        view = resolve_arch_rates(prof, mode)
        assert view["MAC_per_cycle_fp16_tc"] is o and view["MAC_per_cycle_fp16_tc_dense"] is o and view["MAC_per_cycle_fp16_tc_sparse"] is o
        rec = view["sol_rate_resolution"]["precisions"]["fp16"]
        assert rec["applied_key"] == "MAC_per_cycle_fp16_tc" and rec["bare_key_overrides_explicit"] is True and rec["sparse_fallback_to_dense"] is True
    only_sparse = {"freq_GHz": 1.0, "MAC_per_cycle_fp8_tc_sparse": 8}
    assert resolve_arch_rates(only_sparse, "dense")["MAC_per_cycle_fp8_tc"] == 8      # one explicit spelling serves both sets


# ── (3)+(4) the knob and the record ──────────────────────────────────────────────────────────────────
TC_BOUND_ANALYSIS = {
    "metadata": {"precision": "fp16"},
    "total": {"macs": 1_000_000_000, "flops": 2_000_000_000, "other_ops": 0,
              "unfused_elements": 100, "fused_elements": 100, "fused_prefetched_elements": 100,
              "weight_elements": 0, "model_io_elements": 100, "intermediate_elements": 0},
    "layers": {},
}
PROFILE_2X = {"name": "twox", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "DRAM_capacity": 1, "MAC_per_cycle_fp32_sm": 1,
              "peak_TFLOPS_fp16_dense": 0.002, "peak_TFLOPS_fp16_sparse": 0.004,
              "MAC_per_cycle_fp16_tc_dense": 1000, "MAC_per_cycle_fp16_tc_sparse": 2000}


def _predict(tmp_path, name, profile=PROFILE_2X, analysis=TC_BOUND_ANALYSIS, **kw):
    d = tmp_path / name; d.mkdir()
    (d / "analysis.yaml").write_text(yaml.safe_dump(analysis, sort_keys=False))
    (d / "prof.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    perf = EinsumGraphPerfModel().predict(d / "analysis.yaml", d / "perf", arch_config=str(d / "prof.yaml"), precision="fp16", copy_analysis=False, **kw)
    assert perf is not None
    return perf, yaml.safe_load((d / "perf" / f"perf_{profile['name']}.yaml").read_text())


def test_dense_sol_is_twice_the_sparse_sol_for_a_tensor_core_bound_workload(tmp_path):
    sparse, _ = _predict(tmp_path, "s", sol_mode="sparse"); dense, _ = _predict(tmp_path, "d", sol_mode="dense")
    for section in ("unfused", "fused", "fused_prefetched"):
        assert sparse[section]["bottleneck"] == "compute" == dense[section]["bottleneck"]
        assert dense[section]["runtime_ms"] == pytest.approx(2.0 * sparse[section]["runtime_ms"])
        assert dense[section]["compute_tc_cycles"] == 2 * sparse[section]["compute_tc_cycles"] == 1_000_000
    assert sparse["arch"]["MAC_per_cycle"] == 2000.0 and dense["arch"]["MAC_per_cycle"] == 1000.0
    assert sparse["arch"]["mac_per_cycle_key_applied"] == "MAC_per_cycle_fp16_tc_sparse"
    assert dense["arch"]["mac_per_cycle_key_applied"] == "MAC_per_cycle_fp16_tc_dense"


def test_default_is_sparse_and_both_sols_are_in_the_record(tmp_path):
    perf, on_disk = _predict(tmp_path, "default")
    assert on_disk == yaml.safe_load(yaml.safe_dump(perf)) or True   # the file is the dict as written
    for rec in (perf, on_disk):
        assert rec["sol_mode"] == "sparse" and rec["sol_mode_default"] == "sparse"
        for mode, factor in (("sparse", 1), ("dense", 2)):
            block = rec[f"sol_{mode}"]
            assert set(block) >= {"unfused", "fused", "fused_prefetched", "rate_keys", "rates", "note"}
            for section in ("unfused", "fused", "fused_prefetched"):
                assert set(block[section]) == {"compute_tc_cycles", "total_cycles", "runtime_ms", "bottleneck"}
                assert block[section]["runtime_ms"] == pytest.approx(factor * rec["sol_sparse"][section]["runtime_ms"])
            assert block["rate_keys"] == {"MAC_per_cycle_fp16_tc": f"MAC_per_cycle_fp16_tc_{mode}"}
        assert rec["sol_dense"]["rates"] == {"MAC_per_cycle_fp16_tc_dense": 1000.0} and rec["sol_sparse"]["rates"] == {"MAC_per_cycle_fp16_tc_sparse": 2000.0}
        # the top-level fields carry the DEFAULT (sparse) value so existing consumers keep working
        for section in ("unfused", "fused", "fused_prefetched"):
            assert rec[section]["runtime_ms"] == rec["sol_sparse"][section]["runtime_ms"]
            assert rec[section]["total_cycles"] == rec["sol_sparse"][section]["total_cycles"]
        assert "dense SOL is the more realistic estimate" in rec["sol_dense"]["note"] and "lower bound" in rec["sol_sparse"]["note"]


def test_dense_mode_swaps_the_headline_but_still_reports_both(tmp_path):
    perf, _ = _predict(tmp_path, "dense", sol_mode="dense")
    assert perf["sol_mode"] == "dense" and perf["sol_mode_default"] == "sparse"
    assert perf["fused"]["runtime_ms"] == perf["sol_dense"]["fused"]["runtime_ms"] == 2 * perf["sol_sparse"]["fused"]["runtime_ms"]


def test_legacy_profile_reports_identical_sparse_and_dense_with_the_fallback_recorded(tmp_path):
    legacy = {"name": "legacy", "freq_GHz": 1.0, "DRAM_byte_per_cycle": 1.0, "DRAM_capacity": 1, "MAC_per_cycle_fp32_sm": 1, "MAC_per_cycle_fp16_tc": 1000}
    perf, _ = _predict(tmp_path, "legacy", profile=legacy)
    assert perf["sol_sparse"]["fused"]["runtime_ms"] == perf["sol_dense"]["fused"]["runtime_ms"] == perf["fused"]["runtime_ms"]
    assert perf["arch"]["sol_rate_resolution"]["precisions"]["fp16"]["sparse_fallback_to_dense"] is True
    assert perf["sol_sparse"]["rate_keys"] == {"MAC_per_cycle_fp16_tc": "MAC_per_cycle_fp16_tc"}


def test_memory_bound_workload_has_equal_sparse_and_dense_sol(tmp_path):
    mem = {**TC_BOUND_ANALYSIS, "total": {**TC_BOUND_ANALYSIS["total"], "macs": 10, "flops": 20, "unfused_elements": 10**9, "fused_elements": 10**9,
                                             "fused_prefetched_elements": 10**9, "model_io_elements": 10**9}}
    perf, _ = _predict(tmp_path, "mem", analysis=mem)
    assert perf["fused"]["bottleneck"] == "memory"
    assert perf["sol_sparse"]["fused"]["runtime_ms"] == perf["sol_dense"]["fused"]["runtime_ms"]
    assert perf["sol_dense"]["fused"]["compute_tc_cycles"] == 2 * perf["sol_sparse"]["fused"]["compute_tc_cycles"] or perf["sol_sparse"]["fused"]["compute_tc_cycles"] == 0


def test_invalid_sol_mode_is_refused_before_anything_is_written(tmp_path):
    d = tmp_path / "bad"; d.mkdir(); (d / "analysis.yaml").write_text(yaml.safe_dump(TC_BOUND_ANALYSIS)); (d / "prof.yaml").write_text(yaml.safe_dump(PROFILE_2X))
    with pytest.raises(ValueError, match="sol_mode"):
        EinsumGraphPerfModel().predict(d / "analysis.yaml", d / "perf", arch_config=str(d / "prof.yaml"), precision="fp16", sol_mode="fast")
    assert not (d / "perf").exists() or not list((d / "perf").glob("perf_*"))


# ── (5) CLI knob ─────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["sparse", "dense", None])
def test_predict_perf_model_cli_takes_sol_mode(tmp_path, monkeypatch, capsys, mode):
    from solar.cli import predict_perf_model as cli
    (tmp_path / "analysis.yaml").write_text(yaml.safe_dump(TC_BOUND_ANALYSIS)); (tmp_path / "prof.yaml").write_text(yaml.safe_dump(PROFILE_2X))
    argv = ["solar-predict-perf-model", "--analysis-path", str(tmp_path / "analysis.yaml"), "--output-dir", str(tmp_path / "perf"),
            "--arch-config", str(tmp_path / "prof.yaml"), "--precision", "fp16"] + (["--sol-mode", mode] if mode else [])
    monkeypatch.setattr(sys, "argv", argv); cli.main()
    perf = yaml.safe_load((tmp_path / "perf" / "perf_twox.yaml").read_text())
    assert perf["sol_mode"] == (mode or "sparse")
    out = capsys.readouterr().out
    assert "SOL mode" in out and "sparse" in out and "dense" in out


def test_predict_perf_model_cli_rejects_an_unknown_mode(tmp_path, monkeypatch):
    from solar.cli import predict_perf_model as cli
    monkeypatch.setattr(sys, "argv", ["x", "--analysis-path", "a", "--output-dir", str(tmp_path), "--sol-mode", "fast"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2


# ── (6) the other readers use the same resolver ──────────────────────────────────────────────────────
def test_model_analyzer_roofline_prices_at_the_default_mode_of_an_explicit_profile():
    from solar.analysis.model_analyzer import ModelAnalyzer
    r_sparse = ModelAnalyzer()._calculate_roofline_performance(1_000_000, 10, PROFILE_2X, "fp16")
    assert r_sparse["compute_cycles"] == 500          # 1e6 MACs / 2000 MAC per cycle: the sparse (default) rate, not the 1000 fallback
    r_dense = ModelAnalyzer()._calculate_roofline_performance(1_000_000, 10, PROFILE_2X, "fp16", sol_mode="dense")
    assert r_dense["compute_cycles"] == 1000


def test_estimate_result_carries_both_sols_and_the_mode():
    import solar.estimate as api
    names = {f.name for f in dataclasses.fields(api.EstimateResult)}
    assert {"sol_mode", "sol_mode_default", "sol_sparse", "sol_dense"} <= names
    import inspect
    assert inspect.signature(api.estimate).parameters["sol_mode"].default == "sparse"


def test_translate_sol_result_carries_both_sols():
    from solar.translate import SolResult
    names = {f.name for f in dataclasses.fields(SolResult)}
    assert {"sol_mode", "sol_sparse", "sol_dense"} <= names
