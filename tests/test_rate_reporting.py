# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""issue #58 / P2-rate-class-1 -- the perf receipt names the rate CLASS actually applied to the default MAC remainder.

The default remainder (``total.macs`` minus the quantised-dtype MACs) is priced at ``MAC_per_cycle_<precision>_tc`` only when the
profile declares that key; otherwise the existing fallback chain selects another key -- for fp32 on H100 / H200 / L40S / B200 /
Jetson / H100_PCIe that is ``MAC_per_cycle_fp32_sm``, a CUDA-core rate.  The receipt's ``default_rate_remainder`` must derive its
displayed class from the APPLIED key, keep the exact key, say whether it was a fallback, and never call an SM rate a tensor-core
rate.  Report-only: no rate is changed to make a label true; every cycle figure is the fallback chain's as before.
"""
import pathlib

import pytest
import yaml

from solar.perf.perf_model import EinsumGraphPerfModel, _rate_class
from test_mac_report_provenance import MIXED, MIXED_INPUTS, ROOT, _analyze, _dsl_graph

ARCH = ROOT / "configs" / "arch"


@pytest.fixture(scope="module")
def mixed_analysis(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mixed")
    g = _dsl_graph(tmp, MIXED, MIXED_INPUTS); a = _analyze(g, tmp / "analysis")
    assert a["total"]["macs_typed"]["scalar_statement_multiplies"]["macs"] > 0
    return tmp / "analysis" / "analysis.yaml"


def _predict(analysis, out, arch, precision):
    perf = EinsumGraphPerfModel(dtype_bytes=True).predict(analysis, out, arch_config=str(arch), precision=precision, copy_analysis=False)
    assert isinstance(perf, dict), (arch, precision)
    return perf


@pytest.mark.parametrize("profile, precision, applied_key, cls, fallback", [
    ("H100_SXM.yaml", "fp32", "MAC_per_cycle_fp32_sm", "sm", True),          # the reviewer's reproduction: no fp32 tensor-core key -> SM fallback
    ("B200.yaml", "fp32", "MAC_per_cycle_fp32_sm", "sm", True),
    ("A6000.yaml", "fp32", "MAC_per_cycle_fp32_tc", "tensor_core", False),   # native fp32 tensor-core key
    ("H100_SXM.yaml", "fp16", "MAC_per_cycle_fp16_tc", "tensor_core", False),
    ("A6000.yaml", "bf16", "MAC_per_cycle_bf16_tc", "tensor_core", False),
], ids=["h100-fp32-sm-fallback", "b200-fp32-sm-fallback", "a6000-fp32-tc", "h100-fp16-tc", "a6000-bf16-tc"])
def test_receipt_names_the_applied_rate_class(mixed_analysis, tmp_path, profile, precision, applied_key, cls, fallback):
    prof = yaml.safe_load((ARCH / profile).read_text())
    assert (applied_key in prof) and ((f"MAC_per_cycle_{precision}_tc" in prof) is not fallback), profile
    perf = _predict(mixed_analysis, tmp_path / "perf", ARCH / profile, precision)
    r = perf["workload"]["default_rate_remainder"]
    assert r["rate_key"] == applied_key == perf["arch"]["mac_per_cycle_key_applied"] and r["rate_class"] == cls == perf["arch"]["mac_rate_class_applied"]
    assert perf["arch"]["mac_per_cycle_key"] == f"MAC_per_cycle_{precision}_tc"      # the pre-existing field records the REQUESTED key
    assert r["requested_rate_key"] == f"MAC_per_cycle_{precision}_tc" and r["rate_is_fallback"] is fallback and r["precision"] == precision
    assert r["scalar_statement_multiplies"] > 0 and r["contraction_macs_at_workload_precision"] > 0 and r["macs"] == perf["workload"]["total_macs"]
    if cls == "sm":
        assert "tensor-core rate" not in r["note"] and "CUDA-core (SM) rate" in r["note"] and "fallback chain selected MAC_per_cycle_fp32_sm" in r["note"], r["note"]
    else:
        assert "tensor-core rate" in r["note"] and "fallback" not in r["note"], r["note"]
    # P2-r2-contraction-class-1: the receipt never asserts a hardware work class the analyzer did not establish
    assert "tensor-core-class" not in r["note"] and "under-rated" not in r["note"] and "the contraction MACs are tensor-core-class units" not in r["note"], r["note"]
    assert "contraction MACs" in r["note"] and "workload precision" in r["note"] if cls == "sm" else True
    # the applied rate is the one the cycles were computed with (report derived from the arithmetic, not the other way round)
    assert perf["arch"]["MAC_per_cycle"] == prof[applied_key]                      # the applied key is the one the cycles were computed with


def test_profile_without_any_mac_rate_key_reports_the_unit_default_class(mixed_analysis, tmp_path):
    prof = yaml.safe_load((ARCH / "H100_SXM.yaml").read_text())
    for k in [k for k in prof if str(k).startswith("MAC_per_cycle_")]:
        del prof[k]
    p = tmp_path / "no_rates.yaml"; p.write_text(yaml.safe_dump(prof, sort_keys=False))
    perf = _predict(mixed_analysis, tmp_path / "perf", p, "fp32")
    r = perf["workload"]["default_rate_remainder"]
    assert r["rate_class"] == "default_unit" and r["rate_is_fallback"] is True and r["rate_key"].startswith("MAC_per_cycle (default 1.0")
    assert perf["arch"]["mac_per_cycle_key_applied"] == r["rate_key"] and perf["arch"]["mac_rate_class_applied"] == "default_unit" and perf["arch"]["MAC_per_cycle"] == 1.0
    assert "unit default rate 1.0" in r["note"] and "tensor-core rate" not in r["note"]


def test_rate_class_derivation_is_by_key_suffix_only():
    assert _rate_class("MAC_per_cycle_fp32_tc") == "tensor_core" and _rate_class("MAC_per_cycle_nvfp4_tc") == "tensor_core"
    assert _rate_class("MAC_per_cycle_fp32_sm") == "sm" and _rate_class("MAC_per_cycle (default 1.0: no rate key in the profile)") == "default_unit"
    assert _rate_class("something_else") == "unknown"
