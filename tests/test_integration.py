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

"""Integration tests for Solar package with kernelbench."""

import pytest
from pathlib import Path
import tempfile

from solar.graph import BenchmarkProcessor
from solar.analysis import EinsumGraphAnalyzer
from solar.einsum import PyTorchToEinsum
from solar.perf import EinsumGraphPerfModel
from solar.common.types import ProcessingConfig


def create_sample_kernelbench_model(path: Path) -> None:
    """Create a sample kernelbench model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def get_inputs():
    return [torch.randn(1, 3, 224, 224)]
""")


class TestKernelbenchIntegration:
    """Integration tests for kernelbench models."""
    
    def test_full_kernelbench_pipeline(self):
        """Test complete pipeline for kernelbench model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample model
            model_dir = Path(tmpdir) / "kernelbench" / "level1"
            model_path = model_dir / "1_test_model.py"
            create_sample_kernelbench_model(model_path)
            
            # Process the model
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            success = processor.process_file(
                str(model_path)
            )
            
            assert success is True
            
            # Check output files
            kernel_output = output_dir / "level1" / "1"
            assert kernel_output.exists()
            
            # Check extracted layer nodes
            pytorch_graph = kernel_output / "pytorch_graph.yaml"
            assert pytorch_graph.exists()

            # PyTorch graph -> einsum graph (also generates einsum_graph_renamed.yaml).
            converter = PyTorchToEinsum(debug=False, enable_agent=False)
            einsum_graph = converter.convert_graph(pytorch_graph, kernel_output)
            assert einsum_graph is not None
            assert (kernel_output / "einsum_graph.yaml").exists()
            assert (kernel_output / "einsum_graph_renamed.yaml").exists()

            # Einsum graph -> analysis (use renamed graph).
            analyzer = EinsumGraphAnalyzer(debug=False)
            analysis = analyzer.analyze_graph(
                kernel_output / "einsum_graph_renamed.yaml", kernel_output, precision="fp32"
            )
            assert analysis is not None
            assert (kernel_output / "analysis.yaml").exists()
            assert analysis["total"]["num_layers"] > 0
            assert analysis["total"]["macs"] > 0

            # analysis -> perf prediction.
            perf_model = EinsumGraphPerfModel(debug=False)
            perf = perf_model.predict(
                kernel_output / "analysis.yaml",
                kernel_output,
                arch_config="H100_PCIe",
                precision="fp32",
            )
            assert perf is not None
            assert (kernel_output / "perf_H100_PCIe.yaml").exists()
    
    def test_kernelbench_batch_processing(self):
        """Test processing multiple kernelbench models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple models
            model_dir = Path(tmpdir) / "kernelbench" / "level1"
            for i in range(1, 4):
                model_path = model_dir / f"{i}_model.py"
                create_sample_kernelbench_model(model_path)
            
            # Process all models
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            results = processor.process_directory(
                str(tmpdir),
                level="level1",
                kernel_ids=None  # Process all
            )
            
            # Check all were processed
            assert len(results) == 3
            assert all(results.values())
            
            # Check output directories
            for i in range(1, 4):
                kernel_output = output_dir / "level1" / str(i)
                assert kernel_output.exists()


