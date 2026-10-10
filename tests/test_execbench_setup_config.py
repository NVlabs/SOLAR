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

"""The SOL-ExecBench setup config makes a run's pricing policy reproducible.

`configs/execbench/leaderboard_b200.yaml` fixes the arch, the fp32 policy,
the byte accounting and per-problem precision overrides (with a stated
reason). The runner records the file's name and sha256 plus the effective
settings in every sol_summary.json.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_execbench_problem", ROOT / "scripts" / "run_execbench_problem.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_execbench_problem"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shipped_setup_loads_with_hash_and_reasons(runner):
    cfg = runner.load_setup_config(ROOT / "configs" / "execbench" / "leaderboard_b200.yaml")
    assert cfg["name"] == "leaderboard_b200"
    # fp32 problems are priced at the 16-bit tensor-core rate: the measured
    # optimized baselines of the fp32 attention/decoder problems run bf16 math,
    # and a SOL must stay below every implementation.
    assert cfg["arch_config"] == "B200" and cfg["fp32_as"] == "fp16"
    assert cfg["bytes_accounting"] == "per-tensor-dtype"
    assert len(cfg["_sha256"]) == 16
    for name, ov in (cfg["problem_overrides"] or {}).items():
        assert ov["precision"] in ("bf16", "fp16", "tf32"), name
        assert ov.get("reason"), f"override for {name} must state a reason"


def test_problem_override_exact_and_prefix(runner):
    cfg = {"problem_overrides": {"067_flash_attention_gqa_ultralong": {"precision": "bf16", "reason": "x"},
                                 "002_decoder_layer_full_block": {"precision": "bf16", "reason": "y"}}}
    assert runner.problem_override(cfg, "067_flash_attention_gqa_ultralong")["precision"] == "bf16"
    assert runner.problem_override(cfg, "067_flash")["precision"] == "bf16"      # unique prefix
    assert runner.problem_override(cfg, "019_decoder_layer_fused_attention_mlp") == {}
    assert runner.problem_override({}, "anything") == {}


def test_no_config_is_empty(runner):
    assert runner.load_setup_config(None) == {}


def test_pick_precision_still_infers_from_definition(runner):
    definition = {"inputs": {"x": {"dtype": "float32", "shape": ["n"]},
                             "w": {"dtype": "float32", "shape": ["n", "n"]}},
                  "outputs": {"y": {"dtype": "float32", "shape": ["n"]}}}
    assert runner.pick_precision(definition, None) == "fp32"
    assert runner.pick_precision(definition, "bf16") == "bf16"
    definition["inputs"]["w"]["dtype"] = "bfloat16"
    assert runner.pick_precision(definition, None) == "bf16"   # narrowest floating class wins


def test_default_fp32_policy_is_fp16_without_config(runner):
    # No setup config: the runner/reprice fall back to the 16-bit MAC rate for
    # all-fp32 problems (memory still 4 B/elem), the same as the shipped config.
    import inspect
    src = inspect.getsource(runner)
    assert 'setup.get("fp32_as", "fp16")' in src

