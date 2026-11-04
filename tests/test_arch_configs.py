# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Every shipped architecture config must be loadable by the perf model.

The roofline in ``EinsumGraphPerfModel`` reads a fixed set of keys from
``configs/arch/<name>.yaml``; a config that is missing one of them, or whose
``name`` does not match its file name, silently degrades to zero throughput.
These checks run over whichever configs are present so a new one is covered
the moment it is added.
"""

from pathlib import Path

import pytest
import yaml

from solar.perf.perf_model import EinsumGraphPerfModel

ARCH_DIR = Path(__file__).resolve().parents[1] / "configs" / "arch"
ARCH_FILES = sorted(ARCH_DIR.glob("*.yaml"))

# Keys the perf model reads unconditionally (memory roofline, clock and the
# CUDA-core rate). A 16-bit tensor-core rate must exist under either name
# (fp16 falls back to bf16 and vice versa); SRAM_* and the other
# MAC_per_cycle_* keys are optional (SRAM_* is informational, the rest are
# selected by precision).
REQUIRED_KEYS = (
    "DRAM_capacity",
    "DRAM_byte_per_cycle",
    "freq_GHz",
    "MAC_per_cycle_fp32_sm",
)


def test_arch_dir_has_configs():
    assert ARCH_FILES, f"no arch configs under {ARCH_DIR}"


@pytest.mark.parametrize("path", ARCH_FILES, ids=[p.stem for p in ARCH_FILES])
def test_arch_config_is_complete(path):
    cfg = yaml.safe_load(path.read_text())
    assert isinstance(cfg, dict)
    assert cfg.get("name") == path.stem, "config name must match the file name"
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"{path.name} is missing {missing}"
    for key, value in cfg.items():
        if key == "name":
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{path.name}: {key} must be numeric, got {value!r}")
        assert value > 0, f"{path.name}: {key} must be positive"
    half = cfg.get("MAC_per_cycle_fp16_tc", cfg.get("MAC_per_cycle_bf16_tc"))
    assert half is not None, f"{path.name} has neither fp16 nor bf16 tensor-core rate"
    # Tensor-core rates are ordered by width: 16-bit >= tf32, fp8 >= 16-bit when present.
    assert half >= cfg.get("MAC_per_cycle_tf32_tc", 0)
    if "MAC_per_cycle_fp8_tc" in cfg:
        assert cfg["MAC_per_cycle_fp8_tc"] >= half


@pytest.mark.parametrize("path", ARCH_FILES, ids=[p.stem for p in ARCH_FILES])
def test_perf_model_resolves_config_by_name(path):
    model = EinsumGraphPerfModel()
    cfg = model._load_arch_config(path.stem)
    assert cfg and cfg["name"] == path.stem
