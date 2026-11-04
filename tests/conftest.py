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

"""Pytest configuration for Solar tests."""

import os

import pytest
from pathlib import Path
from typing import Any, Dict, List

# tests/fixtures/** is harness DATA, not a test tree.  Several vendored controls there are named test_*.py: they share
# basenames across directories, import their siblings by bare name or read reviewer environment settings at import
# time, so collecting them directly errors.  The tests that own them copy them into a run directory and execute them
# explicitly; tests/fixtures/COVERAGE_MAP.json records that consumer for every such file (tests/test_fixture_coverage_map.py).
collect_ignore = ["fixtures"]


@pytest.fixture(autouse=True)
def _public_inference_endpoint_for_mocked_agents(monkeypatch):
    """The public tree ships no inference endpoint (see
    ``solar.einsum_lib.agent``).  Tests that construct a mocked
    ``TranslationAgent`` need ``SOLAR_API_BASE_URL`` set; a reserved
    ``.invalid`` host guarantees nothing can be contacted even if a mock is
    missing.  An endpoint already set in the caller's environment is kept.
    Tests that clear ``os.environ`` still see the documented refusals."""
    monkeypatch.setenv("SOLAR_API_BASE_URL",
                       os.environ.get("SOLAR_API_BASE_URL",
                                      "https://solar-tests.invalid/v1"))
    # likewise no default model ships; mocked agents get a placeholder name
    monkeypatch.setenv("SOLAR_LLM_MODEL", os.environ.get("SOLAR_LLM_MODEL", "solar-tests-model"))


@pytest.fixture
def sample_node_data() -> Dict[str, Any]:
    """Sample node data for testing."""
    return {
        "node_id": "test_node",
        "node_type": "conv2d",
        "input_shapes": [[1, 3, 224, 224]],
        "output_shapes": [[1, 64, 112, 112]],
        "input_types": ["input", "weight"],
        "output_types": ["output"],
        "module_args": {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": [7, 7],
            "stride": [2, 2],
            "padding": [3, 3]
        }
    }


@pytest.fixture
def sample_torchview_nodes() -> List[Dict[str, Any]]:
    """Sample torchview graph nodes."""
    return [
        {
            "node_id": "Model.conv1",
            "node_type": "conv2d",
            "node_class": "FunctionNode",
            "input_nodes": ["Model.input"],
            "output_nodes": ["Model.relu1"],
            "input_shapes": [[1, 3, 224, 224]],
            "output_shapes": [[1, 64, 112, 112]],
            "input_types": ["input", "weight", "weight"],
            "output_types": ["output"],
            "module_args": {"kernel_size": [7, 7], "stride": [2, 2]}
        },
        {
            "node_id": "Model.relu1",
            "node_type": "relu",
            "node_class": "FunctionNode",
            "input_nodes": ["Model.conv1"],
            "output_nodes": ["Model.pool1"],
            "input_shapes": [[1, 64, 112, 112]],
            "output_shapes": [[1, 64, 112, 112]],
            "input_types": ["input"],
            "output_types": ["output"],
            "module_args": {"inplace": False}
        }
    ]


@pytest.fixture
def kernelbench_sample_path(tmp_path_factory) -> Path:
    """Path to a sample kernelbench model file."""
    base_dir = tmp_path_factory.mktemp("solar_test_data")
    sample_file = base_dir / "kernelbench" / "level1" / "1_simple_model.py"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))

def get_inputs():
    return [torch.randn(1, 3, 224, 224)]
""")
    return sample_file


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for testing."""
    return """
def create_custom_op_subgraph(node_id: str, node_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    subgraph = {}

    input_shapes = node_data.get("input_shapes", [])
    output_shapes = node_data.get("output_shapes", [])

    # Decompose into basic operations
    subgraph[f"{node_id}_matmul"] = {
        "node_type": "matmul",
        "input_shapes": input_shapes,
        "output_shapes": output_shapes
    }

    return subgraph
"""


def pytest_runtest_logreport(report):
    """CI diagnosability (round 9 of an earlier change): when ``SOLAR_CI_STREAM_FAILURES`` is set the
    failure representation is printed the moment a test fails, so a job that is later
    killed by the runner's timeout still names every failure and its reason in its trace
    (the round-8 timeout traces carried 24 bare ``F`` markers and nothing else)."""
    import os
    import sys
    if not os.environ.get("SOLAR_CI_STREAM_FAILURES"):
        return
    if report.when in ("setup", "call", "teardown") and report.failed:
        text = getattr(report, "longreprtext", "") or str(report.longrepr)
        sys.stderr.write("\n=== FAILED (%s) %s ===\n%s\n=== end %s ===\n" % (report.when, report.nodeid, text[-6000:], report.nodeid))
        sys.stderr.flush()
