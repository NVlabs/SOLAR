# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arch configs: MAC_per_cycle values are MACs, i.e. half the datasheet FLOP rate.

perf_model divides total_macs (and total_other_ops) by MAC_per_cycle_*; one FMA is
one MAC and two FLOPs, so a config that stores the FLOP rate halves every
compute-bound SOL time.  The lineage snapshot the lineage rewrite doubled every
MAC_per_cycle key of H100_PCIe.yaml (fp32 12,750 -> 25,500 = 102 TFLOPS at
2 GHz, twice the datasheet) and deleted the comment that says why.  This test
pins each arch's fp32 SM rate: MAC_per_cycle_fp32_sm x freq_GHz x 2 FLOP/MAC
must equal the documented FP32 TFLOPS within 1 %.
"""
from pathlib import Path

import pytest
import yaml

ARCH_DIR = Path(__file__).resolve().parent.parent / "configs" / "arch"

# arch -> (documented dense FP32 TFLOPS, source of the figure)
DATASHEET_FP32_TFLOPS = {
    "H100_PCIe": (51.0, "NVIDIA H100 PCIe datasheet: 51 TFLOPS FP32"),
    "H100_SXM": (67.0, "NVIDIA H100 SXM datasheet: 67 TFLOPS FP32"),
    "H200_SXM": (67.0, "NVIDIA H200 datasheet: compute rates equal H100 SXM, 67 TFLOPS FP32"),
    "L40S": (91.6, "NVIDIA L40S datasheet: 91.6 TFLOPS FP32"),
    "A6000": (38.71, "NVIDIA RTX A6000 datasheet: 38.7 TFLOPS FP32"),
    # B200.yaml's header states these are the repository's existing roofline
    # assumptions normalised to a nominal 1.5 GHz clock, "not a new validation
    # against public data"; the documented aggregate in the file is 0.121
    # PFLOPS fp32.  The check pins the file to its own documented figure.
    "B200": (120.75, "B200.yaml header/line comment: 0.121 PFLOPS fp32 (repository assumption, not validated)"),
}

# Jetson_Thor_T5000.yaml documents "15 FP32 TFLOPS (SM only)" but stores
# 9,554 MAC/cycle = 15e12 / 1.57e9, i.e. the FLOP rate as the MAC rate
# (encodes 30 TFLOPS).  Pre-existing at the earlier baseline, unchanged by the lineage;
# recorded here as a strict expected failure so the inconsistency stays
# visible until its owner settles the figure.
JETSON_DOCUMENTED_FP32_TFLOPS = 15.0

# H100_PCIe, per key: documented dense rate (NVIDIA H100 architecture whitepaper
# v1.04, Table 1, PCIe column, dense figures) -> MAC/cycle = rate / (2 FLOP per
# MAC x 2.0e9 Hz, the yaml's declared conversion clock).  int8/fp8 keep the
# repository's existing 1512 TOPS contract value (378,000; the whitepaper's 1513
# would give 378,250): a 0.07 % rounding, not the 2x unit error.
H100_PCIE_DOCUMENTED_DENSE_TOPS = {
    "MAC_per_cycle_fp32_sm": (51.0, "FP32 51 TFLOPS (whitepaper gives 51.2; datasheet 51)"),
    "MAC_per_cycle_tf32_tc": (378.0, "TF32 Tensor Core 378 TFLOPS dense"),
    "MAC_per_cycle_fp16_tc": (756.0, "FP16 Tensor Core 756 TFLOPS dense"),
    "MAC_per_cycle_bf16_tc": (756.0, "BF16 Tensor Core 756 TFLOPS dense"),
    "MAC_per_cycle_fp8_tc": (1513.0, "FP8 Tensor Core 1513 TFLOPS dense; yaml keeps the 1512 contract value"),
    "MAC_per_cycle_int8_tc": (1513.0, "INT8 Tensor Core 1513 TOPS dense; yaml keeps the 1512 contract value"),
}
H100_PCIE_CONVERSION_GHZ = 2.0

# H100_PCIe: every key pinned to the values the earlier baseline carried (the lineage
# doubled all six).
H100_PCIE_MACS_PER_CYCLE = {
    "MAC_per_cycle_fp32_sm": 12750,
    "MAC_per_cycle_int8_tc": 378000,
    "MAC_per_cycle_fp8_tc": 378000,
    "MAC_per_cycle_fp16_tc": 189000,
    "MAC_per_cycle_bf16_tc": 189000,
    "MAC_per_cycle_tf32_tc": 94500,
}


def _load(name: str) -> dict:
    with open(ARCH_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _fp32_tflops(cfg: dict) -> float:
    return float(cfg["MAC_per_cycle_fp32_sm"]) * float(cfg["freq_GHz"]) * 2.0 / 1000.0


@pytest.mark.parametrize("arch", sorted(DATASHEET_FP32_TFLOPS))
def test_fp32_sm_rate_is_half_the_documented_flop_rate(arch):
    cfg = _load(arch)
    documented, source = DATASHEET_FP32_TFLOPS[arch]
    got = _fp32_tflops(cfg)
    assert abs(got - documented) / documented <= 0.01, (
        f"{arch}: MAC_per_cycle_fp32_sm x clock x 2 = {got:.2f} TFLOPS, documented "
        f"{documented} ({source}); the yaml must store MACs = FLOPs / 2")


@pytest.mark.xfail(strict=True, reason="pre-existing at the earlier baseline (not a lineage change): Jetson stores the FP32 FLOP rate as MAC_per_cycle_fp32_sm (encodes 30 TFLOPS against its documented 15)")
def test_jetson_thor_fp32_sm_rate_matches_its_documented_figure():
    got = _fp32_tflops(_load("Jetson_Thor_T5000"))
    assert abs(got - JETSON_DOCUMENTED_FP32_TFLOPS) / JETSON_DOCUMENTED_FP32_TFLOPS <= 0.01


@pytest.mark.parametrize("key", sorted(H100_PCIE_DOCUMENTED_DENSE_TOPS))
def test_h100_pcie_each_key_is_documented_dense_rate_over_two_flop_per_mac(key):
    """dense FLOP/s / (2 FLOP/MAC x 2.0e9 Hz) per key, within 1 %."""
    cfg = _load("H100_PCIe")
    assert cfg["freq_GHz"] == H100_PCIE_CONVERSION_GHZ
    documented_tops, source = H100_PCIE_DOCUMENTED_DENSE_TOPS[key]
    expected_macs_per_cycle = documented_tops * 1e12 / (2.0 * H100_PCIE_CONVERSION_GHZ * 1e9)
    assert abs(cfg[key] - expected_macs_per_cycle) / expected_macs_per_cycle <= 0.01, (
        f"{key} = {cfg[key]} MAC/cycle encodes {cfg[key] * 2 * H100_PCIE_CONVERSION_GHZ / 1000:.1f} "
        f"dense TFLOPS; documented {documented_tops} ({source})")


def test_h100_pcie_has_no_fp32_tc_key():
    """The fp32 route falls back to the SM rate; TF32 needs its own precision contract."""
    assert "MAC_per_cycle_fp32_tc" not in _load("H100_PCIe")


def test_h100_pcie_mac_rates_are_the_dense_baseline_values():
    cfg = _load("H100_PCIe")
    for key, value in H100_PCIE_MACS_PER_CYCLE.items():
        assert cfg[key] == value, f"{key}: {cfg[key]} != {value} (lineage doubled it)"


def test_h100_pcie_states_the_mac_convention():
    text = (ARCH_DIR / "H100_PCIe.yaml").read_text()
    assert "MACs" in text and "half the advertised dense number" in text


def test_every_arch_yaml_is_covered_or_declared():
    names = {p.stem for p in ARCH_DIR.glob("*.yaml")}
    assert names == set(DATASHEET_FP32_TFLOPS) | {"Jetson_Thor_T5000"}, names
