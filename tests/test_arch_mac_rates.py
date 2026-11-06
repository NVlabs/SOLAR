# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arch configs: MAC_per_cycle values are MACs, i.e. half the datasheet FLOP rate.

perf_model divides total_macs (and total_other_ops) by MAC_per_cycle_*; one FMA is
one MAC and two FLOPs, so a config that stores the FLOP rate halves every
compute-bound SOL time.  An earlier lineage rewrite doubled every MAC_per_cycle key
of H100_PCIe.yaml (fp32 12,750 -> 25,500 = 102 TFLOPS at 2 GHz, twice the
datasheet) and deleted the comment that says why; the A6000 and Jetson Thor files
historically stored the FLOP rate per cycle as the MAC rate (GitHub issue #14:
"twice the tensor-core peak").  This test pins each arch's fp32 SM rate
(MAC_per_cycle_fp32_sm x freq_GHz x 2 FLOP/MAC == the documented FP32 TFLOPS
within 1 %) and, for H100_PCIe, every tensor-core key of BOTH peak sets
(issue #14: ``MAC_per_cycle_<p>_tc_dense`` / ``_sparse``), with the unit rule
``MAC_per_cycle = TFLOPS x 1e12 / 2 / (freq_GHz x 1e9) = TFLOPS x 1000 / 2 / freq_GHz``.
"""
from pathlib import Path

import pytest
import yaml

ARCH_DIR = Path(__file__).resolve().parent.parent / "configs" / "arch"

# arch -> (documented dense FP32 TFLOPS on the CUDA cores, source of the figure)
DATASHEET_FP32_TFLOPS = {
    "H100_PCIe": (51.0, "NVIDIA H100 datasheet, H100 PCIe column: 51 TFLOPS FP32"),
    "H100_SXM": (67.0, "NVIDIA H100 product page, H100 SXM: 67 TFLOPS FP32"),
    "H200_SXM": (67.0, "NVIDIA H200 product page: compute rates equal H100 SXM, 67 TFLOPS FP32"),
    "L40S": (91.6, "NVIDIA L40S datasheet: 91.6 TFLOPS FP32"),
    "A6000": (38.71, "NVIDIA RTX A6000 datasheet: 38.7 TFLOPS FP32 (10,752 CUDA cores x 2 x 1.8 GHz = 38.71)"),
    "B200": (75.0, "NVIDIA HGX B200 datasheet: FP32 600 TFLOPS per 8 GPUs = 75 per GPU (issue #14 re-basing)"),
    # issue #14: Jetson previously stored the FLOP rate (9,554 = 15e12 / 1.57e9, encoding 30 TFLOPS) and was pinned here as a
    # strict expected failure; the file now stores MACs (4,777) and the check passes like every other arch.
    "Jetson_Thor_T5000": (15.0, "NVIDIA Jetson AGX Thor specification: 15 TFLOPS FP32 (CUDA cores)"),
}

# H100_PCIe, per tensor-core precision: (dense TFLOPS, sparse TFLOPS) as printed on the NVIDIA H100 datasheet
# (H100 PCIe column; the printed tensor-core figure is "with sparsity", dense is half).  MAC/cycle = TFLOPS x 1000 / 2 / 2.0.
H100_PCIE_DOCUMENTED_TOPS = {
    "tf32": (378.0, 756.0, "TF32 Tensor Core 756 TFLOPS with sparsity"),
    "fp16": (756.5, 1513.0, "FP16 Tensor Core 1,513 TFLOPS with sparsity"),
    "bf16": (756.5, 1513.0, "BFLOAT16 Tensor Core 1,513 TFLOPS with sparsity"),
    "fp8": (1513.0, 3026.0, "FP8 Tensor Core 3,026 TFLOPS with sparsity"),
    "int8": (1513.0, 3026.0, "INT8 Tensor Core 3,026 TOPS with sparsity"),
    "fp64": (51.0, 51.0, "FP64 Tensor Core 51 TFLOPS (no sparse figure: sparse == dense)"),
}
H100_PCIE_CONVERSION_GHZ = 2.0

# H100_PCIe: every tensor-core key pinned exactly (issue #14 re-derivation from the printed peaks; the former
# "1512 TOPS contract" rounding 378,000 is gone: 1513 -> 378,250 dense).
H100_PCIE_MACS_PER_CYCLE = {
    "MAC_per_cycle_fp32_sm": 12750,
    "MAC_per_cycle_tf32_tc_dense": 94500, "MAC_per_cycle_tf32_tc_sparse": 189000,
    "MAC_per_cycle_fp16_tc_dense": 189125, "MAC_per_cycle_fp16_tc_sparse": 378250,
    "MAC_per_cycle_bf16_tc_dense": 189125, "MAC_per_cycle_bf16_tc_sparse": 378250,
    "MAC_per_cycle_fp8_tc_dense": 378250, "MAC_per_cycle_fp8_tc_sparse": 756500,
    "MAC_per_cycle_int8_tc_dense": 378250, "MAC_per_cycle_int8_tc_sparse": 756500,
    "MAC_per_cycle_fp64_tc_dense": 12750, "MAC_per_cycle_fp64_tc_sparse": 12750,
}

# issue #14 provenance (LM correction C5): the two historically sparse-equivalent files keep a History note quoting the
# legacy numbers, so the relabelling is visible in the file itself.
LEGACY_PROVENANCE = {
    "A6000": ("86016", "172032", "History"),
    "Jetson_Thor_T5000": ("329299", "659236", "9554", "History"),
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
    assert abs(cfg["peak_TFLOPS_fp32_sm"] - documented) / documented <= 0.01, f"{arch}: peak_TFLOPS_fp32_sm disagrees with {source}"


def test_jetson_thor_fp32_sm_rate_matches_its_documented_figure():
    """Formerly a strict xfail (the file stored 9,554 = the FLOP rate); issue #14 corrected it to 4,777 MAC/cycle."""
    got = _fp32_tflops(_load("Jetson_Thor_T5000"))
    assert abs(got - 15.0) / 15.0 <= 0.01


@pytest.mark.parametrize("precision", sorted(H100_PCIE_DOCUMENTED_TOPS))
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_h100_pcie_each_key_is_documented_rate_over_two_flop_per_mac(precision, kind):
    """TFLOPS x 1e12 / (2 FLOP/MAC x 2.0e9 Hz) per key and peak set, within 0.1 %."""
    cfg = _load("H100_PCIe")
    assert cfg["freq_GHz"] == H100_PCIE_CONVERSION_GHZ
    dense, sparse, source = H100_PCIE_DOCUMENTED_TOPS[precision]
    documented_tops = dense if kind == "dense" else sparse
    key = f"MAC_per_cycle_{precision}_tc_{kind}"
    expected_macs_per_cycle = documented_tops * 1e12 / (2.0 * H100_PCIE_CONVERSION_GHZ * 1e9)
    assert abs(cfg[key] - expected_macs_per_cycle) / expected_macs_per_cycle <= 1e-3, (
        f"{key} = {cfg[key]} MAC/cycle encodes {cfg[key] * 2 * H100_PCIE_CONVERSION_GHZ / 1000:.1f} "
        f"{kind} TFLOPS; documented {documented_tops} ({source})")
    assert cfg[f"peak_TFLOPS_{precision}_{kind}"] == documented_tops


def test_h100_pcie_has_no_fp32_tc_key():
    """The fp32 route falls back to the SM rate; TF32 needs its own precision contract."""
    cfg = _load("H100_PCIe")
    assert not [k for k in cfg if k.startswith("MAC_per_cycle_fp32_tc")]


def test_h100_pcie_mac_rates_are_the_pinned_values():
    cfg = _load("H100_PCIe")
    for key, value in H100_PCIE_MACS_PER_CYCLE.items():
        assert cfg[key] == value, f"{key}: {cfg[key]} != {value}"
    assert not [k for k in cfg if k.startswith("MAC_per_cycle_") and k.endswith("_tc")], "no bare legacy keys in the shipped file"


def test_h100_pcie_states_the_mac_convention():
    text = (ARCH_DIR / "H100_PCIe.yaml").read_text()
    assert "MACs" in text and "2 FLOP per MAC" in text and "NOT FLOPs" in text
    assert "peak_TFLOPS x 1000 / 2 / freq_GHz" in text, "the unit rule is stated in the file"
    assert "structured-sparsity" in text and "lower bound" in text and "realistic bound" in text


@pytest.mark.parametrize("arch", sorted(LEGACY_PROVENANCE))
def test_historically_sparse_files_keep_their_provenance_note(arch):
    text = (ARCH_DIR / f"{arch}.yaml").read_text()
    for token in LEGACY_PROVENANCE[arch]:
        assert token in text, f"{arch}.yaml: provenance note must quote the legacy value {token}"


def test_every_arch_yaml_is_covered_or_declared():
    names = {p.stem for p in ARCH_DIR.glob("*.yaml")}
    assert names == set(DATASHEET_FP32_TFLOPS), names
