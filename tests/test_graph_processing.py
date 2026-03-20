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

"""Tests for graph processing modules."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from solar.graph import BenchmarkProcessor, PyTorchProcessor, TorchviewProcessor
from solar.common.types import ProcessingConfig, NodeInfo


class TestTorchviewProcessor:
    """Tests for TorchviewProcessor."""
    
    def test_process_graph(self, tmp_path, sample_torchview_nodes):
        """Test processing a torchview graph."""
        processor = TorchviewProcessor(debug=True)
        
        # Create minimal computation graph (avoid Mock auto-creating attributes)
        mock_graph = Mock(spec=["edge_list", "node_hierarchy"])
        mock_graph.edge_list = []
        mock_graph.node_hierarchy = {}
        
        # Process graph
        result = processor.process_graph(
            mock_graph,
            str(tmp_path),
            "test_kernel",
            original_model=None
        )
        
        # Check output file was created
        assert (tmp_path / "pytorch_graph.yaml").exists()
    
    def test_extract_node_info(self):
        """Test extracting information from a node."""
        processor = TorchviewProcessor()
        
        # Create mock node with proper attributes
        mock_node = Mock()
        mock_node.name = "conv2d"
        mock_node.operation = None
        mock_node.op_name = None
        mock_node.inputs = []
        mock_node.outputs = []
        # Mock input_shape and output_shape to return None (no shapes available)
        mock_node.input_shape = None
        mock_node.output_shape = None
        mock_node.tensor_shape = None
        
        # Extract info
        node_info = processor._extract_node_info(mock_node, "test_node")
        
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "test_node"
        assert node_info.type == "conv2d"
    
    def test_infer_parameter_name(self):
        """Test parameter name inference."""
        processor = TorchviewProcessor()
        
        # Test convolution
        assert processor._infer_parameter_name("conv2d", 0, [1, 3, 224, 224]) == "input"
        assert processor._infer_parameter_name("conv2d", 1, [64, 3, 7, 7]) == "weight"
        assert processor._infer_parameter_name("conv2d", 2, [64]) == "bias"
        
        # Test linear
        assert processor._infer_parameter_name("linear", 0, [32, 784]) == "input"
        assert processor._infer_parameter_name("linear", 1, [10, 784]) == "weight"
        assert processor._infer_parameter_name("linear", 2, [10]) == "bias"


class TestPyTorchProcessor:
    """Tests for PyTorchProcessor."""
    
    def test_process_kernelbench_file(self, kernelbench_sample_path, tmp_path):
        """Test processing a kernelbench file."""
        config = ProcessingConfig(
            output_dir=str(tmp_path),
            save_graph=False,
            force_rerun=True,
            debug=True
        )
        processor = PyTorchProcessor(config)
        
        output_dir = tmp_path / "single_model_kernelbench"
        success = processor.process_model_file(str(kernelbench_sample_path), str(output_dir))
        
        # Check success
        assert success is True
        
        # Check output was created
        assert output_dir.exists()
        assert (output_dir / "pytorch_graph.yaml").exists()
    
    @patch('torchview.draw_graph')
    def test_generate_torchview_graph(self, mock_draw_graph):
        """Test torchview graph generation."""
        processor = PyTorchProcessor()
        
        # Create mock model and inputs
        mock_model = Mock()
        mock_model.to_empty = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        
        mock_inputs = [Mock()]
        
        # Mock torchview return
        mock_graph = Mock()
        mock_draw_graph.return_value = mock_graph
        
        # Generate graph
        result = processor._generate_torchview_graph(mock_model, mock_inputs)
        
        assert result is mock_graph
        mock_draw_graph.assert_called_once()
    
    def test_is_rnn_model(self):
        """Test RNN model detection."""
        processor = PyTorchProcessor()
        
        # Test with RNN attributes
        mock_rnn = Mock()
        mock_rnn.hidden = Mock()
        assert processor._is_rnn_model(mock_rnn) is True
        
        # Test with module names
        mock_model = Mock()
        mock_model.named_modules = Mock(return_value=[("lstm_layer", Mock())])
        assert processor._is_rnn_model(mock_model) is True
        
        # Test non-RNN
        mock_model = Mock(spec=["named_modules"])
        mock_model.named_modules = Mock(return_value=[("conv_layer", Mock())])
        assert processor._is_rnn_model(mock_model) is False
    
class TestBenchmarkProcessor:
    """Tests for BenchmarkProcessor (kernelbench conventions)."""

    def test_filter_by_kernel_ids(self):
        """Test filtering files by kernel IDs."""
        processor = BenchmarkProcessor()

        # Create mock file paths
        files = [
            Path("1_model.py"),
            Path("2_model.py"),
            Path("3_model.py"),
            Path("10_model.py")
        ]

        filtered = processor._filter_by_kernel_ids(files, [1, 3])
        assert len(filtered) == 2
        assert Path("1_model.py") in filtered
        assert Path("3_model.py") in filtered


class TestIntegration:
    """Integration tests for the full processing pipeline."""
    
    def test_kernelbench_pipeline(self, kernelbench_sample_path, tmp_path):
        """Test full kernelbench processing pipeline."""
        config = ProcessingConfig(
            output_dir=str(tmp_path),
            save_graph=False,
            force_rerun=True,
            debug=False
        )
        processor = BenchmarkProcessor(config)
        
        # Process directory
        results = processor.process_directory(
            str(kernelbench_sample_path.parent.parent.parent),
            level="level1",
            kernel_ids=[1]
        )
        
        # Check results
        assert len(results) > 0
        assert all(isinstance(v, bool) for v in results.values())


class TestDtypeExtraction:
    """Tests for dtype extraction and validation."""
    
    def test_dtype_counts_match_shape_counts(self, tmp_path):
        """Test that input_dtypes and output_dtypes counts match input_shapes and output_shapes counts."""
        import torch
        import torch.nn as nn
        try:
            from torchview import draw_graph
            TORCHVIEW_AVAILABLE = True
        except ImportError:
            pytest.skip("torchview not available")
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        x = torch.randn(2, 10)
        
        processor = TorchviewProcessor(debug=False)
        graph = draw_graph(model, input_data=x, expand_nested=True)
        nodes = processor.process_graph(graph, str(tmp_path), "test_model", model)
        
        # Load the saved YAML to verify
        import yaml
        yaml_path = tmp_path / "pytorch_graph.yaml"
        assert yaml_path.exists(), "YAML file should be created"
        
        with open(yaml_path) as f:
            graph_data = yaml.safe_load(f)
        
        # Check each node
        for node_id, node_data in graph_data.get("layers", {}).items():
            input_shapes = node_data.get("input_shapes", [])
            output_shapes = node_data.get("output_shapes", [])
            input_dtypes = node_data.get("input_dtypes", [])
            output_dtypes = node_data.get("output_dtypes", [])
            
            # Verify counts match
            assert len(input_dtypes) == len(input_shapes), (
                f"Node {node_id}: input_dtypes count ({len(input_dtypes)}) "
                f"does not match input_shapes count ({len(input_shapes)})"
            )
            
            assert len(output_dtypes) == len(output_shapes), (
                f"Node {node_id}: output_dtypes count ({len(output_dtypes)}) "
                f"does not match output_shapes count ({len(output_shapes)})"
            )
            
            # Verify dtype format (should be strings)
            for dtype in input_dtypes:
                assert isinstance(dtype, str), f"Node {node_id}: input_dtype should be string, got {type(dtype)}"
            
            for dtype in output_dtypes:
                assert isinstance(dtype, str), f"Node {node_id}: output_dtype should be string, got {type(dtype)}"
    
    def test_dtype_fields_present_in_yaml(self, tmp_path):
        """Test that input_dtypes and output_dtypes fields are present in YAML output."""
        import torch
        import torch.nn as nn
        try:
            from torchview import draw_graph
            TORCHVIEW_AVAILABLE = True
        except ImportError:
            pytest.skip("torchview not available")
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        x = torch.randn(2, 10)
        
        processor = TorchviewProcessor(debug=False)
        graph = draw_graph(model, input_data=x, expand_nested=True)
        nodes = processor.process_graph(graph, str(tmp_path), "test_model", model)
        
        # Load the saved YAML
        import yaml
        yaml_path = tmp_path / "pytorch_graph.yaml"
        
        with open(yaml_path) as f:
            graph_data = yaml.safe_load(f)
        
        # Check that all nodes have dtype fields
        for node_id, node_data in graph_data.get("layers", {}).items():
            assert "input_dtypes" in node_data, f"Node {node_id} missing input_dtypes field"
            assert "output_dtypes" in node_data, f"Node {node_id} missing output_dtypes field"
            assert isinstance(node_data["input_dtypes"], list), f"Node {node_id} input_dtypes should be list"
            assert isinstance(node_data["output_dtypes"], list), f"Node {node_id} output_dtypes should be list"
