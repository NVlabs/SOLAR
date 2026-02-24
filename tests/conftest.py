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

import pytest
from pathlib import Path
from typing import Any, Dict, List


@pytest.fixture
def sample_node_data() -> Dict[str, Any]:
    """Sample node data for testing."""
    return {
        "node_id": "test_node",
        "node_type": "conv2d",
        "input_shapes": [[1, 3, 224, 224]],
        "output_shapes": [[1, 64, 112, 112]],
        "weight_shapes": [[64, 3, 7, 7]],
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
            "weight_nodes": ["weight", "bias"],
            "weight_shapes": [[64, 3, 7, 7], [64]],
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
            "weight_nodes": [],
            "weight_shapes": [],
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
