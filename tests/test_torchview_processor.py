"""Tests for TorchviewProcessor to verify extracted graph matches expected format.

This module tests that the TorchviewProcessor correctly extracts:
- node_id: Hierarchical node identifier
- node_type: Operation type
- node_class: Actual node class (TensorNode, FunctionNode, ModuleNode)
- input_nodes: List of input node IDs (connections from predecessors)
- output_nodes: List of output node IDs (connections to successors)
- input_shapes: List of input tensor shapes
- output_shapes: List of output tensor shapes
- weight_nodes: List of parameter names
- weight_shapes: List of parameter shapes
- module_args: Dictionary of module configuration arguments
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
import yaml


# Skip tests if torchview is not available
try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False

from solar.graph.torchview_processor import TorchviewProcessor
from solar.common.types import NodeInfo


class SimpleLinearModel(nn.Module):
    """Simple model with a single linear layer."""
    
    def __init__(self, in_features: int = 64, out_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoLayerModel(nn.Module):
    """Model with two linear layers and a ReLU."""
    
    def __init__(self, in_features: int = 64, hidden: int = 32, out_features: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """Model with a convolution layer."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvTransposeModel(nn.Module):
    """Model with a transposed convolution layer."""
    
    def __init__(self, in_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=3, stride=2, padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)


class AttentionModel(nn.Module):
    """Simple attention model for testing multi-head operations."""
    
    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Simple attention without reshaping for heads
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.out_proj(context)


@pytest.fixture
def processor():
    """Create a TorchviewProcessor instance."""
    return TorchviewProcessor(debug=False)


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestNodeInfoFields:
    """Test that extracted NodeInfo objects have all required fields."""
    
    def test_node_info_has_all_fields(self, processor):
        """Verify NodeInfo has all expected fields."""
        model = SimpleLinearModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            assert len(nodes) > 0, "Should extract at least one node"
            
            for node in nodes:
                # Check all required fields exist
                assert hasattr(node, 'node_id'), "Missing node_id"
                assert hasattr(node, 'node_type'), "Missing node_type"
                assert hasattr(node, 'node_class'), "Missing node_class"
                assert hasattr(node, 'input_nodes'), "Missing input_nodes"
                assert hasattr(node, 'output_nodes'), "Missing output_nodes"
                assert hasattr(node, 'input_shapes'), "Missing input_shapes"
                assert hasattr(node, 'output_shapes'), "Missing output_shapes"
                assert hasattr(node, 'weight_nodes'), "Missing weight_nodes"
                assert hasattr(node, 'weight_shapes'), "Missing weight_shapes"
                assert hasattr(node, 'module_args'), "Missing module_args"
                
                # Check types
                assert isinstance(node.node_id, str)
                assert isinstance(node.node_type, str)
                assert isinstance(node.node_class, str)
                assert isinstance(node.input_nodes, list)
                assert isinstance(node.output_nodes, list)
                assert isinstance(node.input_shapes, list)
                assert isinstance(node.output_shapes, list)
                assert isinstance(node.weight_nodes, list)
                assert isinstance(node.weight_shapes, list)
                assert isinstance(node.module_args, dict)
    
    def test_node_class_values(self, processor):
        """Verify node_class is one of expected values."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            valid_classes = {'TensorNode', 'FunctionNode', 'ModuleNode'}
            for node in nodes:
                assert node.node_class in valid_classes, \
                    f"Invalid node_class: {node.node_class}"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestConnectionExtraction:
    """Test that input_nodes and output_nodes are correctly extracted."""
    
    def test_connections_are_extracted(self, processor):
        """Verify connections between nodes are extracted."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # At least some nodes should have connections
            nodes_with_inputs = [n for n in nodes if n.input_nodes]
            nodes_with_outputs = [n for n in nodes if n.output_nodes]
            
            assert len(nodes_with_inputs) > 0, "No nodes have input connections"
            assert len(nodes_with_outputs) > 0, "No nodes have output connections"
    
    def test_connection_consistency(self, processor):
        """Verify connection consistency: if A outputs to B, B should input from A."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            node_map = {n.node_id: n for n in nodes}
            
            for node in nodes:
                for output_id in node.output_nodes:
                    if output_id in node_map:
                        target_node = node_map[output_id]
                        assert node.node_id in target_node.input_nodes, \
                            f"Inconsistent connection: {node.node_id} -> {output_id}"
    
    def test_connection_node_ids_are_valid(self, processor):
        """Verify all connection IDs refer to existing nodes."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            all_node_ids = {n.node_id for n in nodes}
            
            for node in nodes:
                for input_id in node.input_nodes:
                    assert input_id in all_node_ids, \
                        f"Invalid input_node reference: {input_id}"
                for output_id in node.output_nodes:
                    assert output_id in all_node_ids, \
                        f"Invalid output_node reference: {output_id}"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestShapeExtraction:
    """Test that input_shapes and output_shapes are correctly extracted."""
    
    def test_linear_shapes(self, processor):
        """Verify shapes are extracted for linear layers."""
        model = SimpleLinearModel(in_features=64, out_features=32)
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find linear node
            linear_nodes = [n for n in nodes if 'linear' in n.node_type.lower()]
            assert len(linear_nodes) > 0, "Should find linear node"
            
            linear_node = linear_nodes[0]
            assert len(linear_node.input_shapes) > 0, "Linear should have input shapes"
            assert len(linear_node.output_shapes) > 0, "Linear should have output shapes"
            
            # Check shape dimensions
            assert linear_node.input_shapes[0][-1] == 64, "Input last dim should be 64"
            assert linear_node.output_shapes[0][-1] == 32, "Output last dim should be 32"
    
    def test_conv_shapes(self, processor):
        """Verify shapes are extracted for conv layers."""
        model = ConvModel(in_channels=3, out_channels=16)
        x = torch.randn(2, 3, 32, 32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find conv node
            conv_nodes = [n for n in nodes if 'conv' in n.node_type.lower()]
            assert len(conv_nodes) > 0, "Should find conv node"
            
            conv_node = conv_nodes[0]
            assert len(conv_node.input_shapes) > 0, "Conv should have input shapes"
            assert len(conv_node.output_shapes) > 0, "Conv should have output shapes"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestWeightExtraction:
    """Test that weight_nodes and weight_shapes are correctly extracted."""
    
    def test_linear_weights(self, processor):
        """Verify weights are extracted for linear layers."""
        model = SimpleLinearModel(in_features=64, out_features=32)
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find linear node
            linear_nodes = [n for n in nodes if 'linear' in n.node_type.lower()]
            assert len(linear_nodes) > 0, "Should find linear node"
            
            linear_node = linear_nodes[0]
            assert 'weight' in linear_node.weight_nodes, "Should have weight parameter"
            
            # Check weight shape
            weight_idx = linear_node.weight_nodes.index('weight')
            weight_shape = linear_node.weight_shapes[weight_idx]
            assert weight_shape == [32, 64], f"Weight shape should be [32, 64], got {weight_shape}"
    
    def test_conv_weights(self, processor):
        """Verify weights are extracted for conv layers."""
        model = ConvModel(in_channels=3, out_channels=16)
        x = torch.randn(2, 3, 32, 32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find conv node
            conv_nodes = [n for n in nodes if 'conv' in n.node_type.lower()]
            assert len(conv_nodes) > 0, "Should find conv node"
            
            conv_node = conv_nodes[0]
            assert 'weight' in conv_node.weight_nodes, "Should have weight parameter"
            
            # Check weight shape
            weight_idx = conv_node.weight_nodes.index('weight')
            weight_shape = conv_node.weight_shapes[weight_idx]
            assert weight_shape == [16, 3, 3, 3], f"Weight shape should be [16, 3, 3, 3], got {weight_shape}"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestModuleArgsExtraction:
    """Test that module_args are correctly extracted."""
    
    def test_linear_module_args(self, processor):
        """Verify module_args are extracted for linear layers."""
        model = SimpleLinearModel(in_features=64, out_features=32)
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find linear node
            linear_nodes = [n for n in nodes if 'linear' in n.node_type.lower()]
            assert len(linear_nodes) > 0, "Should find linear node"
            
            linear_node = linear_nodes[0]
            args = linear_node.module_args
            
            assert 'module_type' in args, "Should have module_type"
            assert args['module_type'] == 'Linear', f"module_type should be Linear, got {args['module_type']}"
            assert args.get('in_features') == 64, "in_features should be 64"
            assert args.get('out_features') == 32, "out_features should be 32"
    
    def test_conv_module_args(self, processor):
        """Verify module_args are extracted for conv layers."""
        model = ConvModel(in_channels=3, out_channels=16)
        x = torch.randn(2, 3, 32, 32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Find conv node
            conv_nodes = [n for n in nodes if 'conv' in n.node_type.lower()]
            assert len(conv_nodes) > 0, "Should find conv node"
            
            conv_node = conv_nodes[0]
            args = conv_node.module_args
            
            assert 'module_type' in args, "Should have module_type"
            assert 'Conv' in args['module_type'], f"module_type should contain Conv, got {args['module_type']}"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestYAMLOutput:
    """Test that the YAML output matches expected format."""
    
    def test_yaml_structure(self, processor):
        """Verify YAML output has correct structure."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            processor.process_graph(graph, tmpdir, "test_model", model)
            
            yaml_path = Path(tmpdir) / "pytorch_graph.yaml"
            assert yaml_path.exists(), "YAML file should be created"
            
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            
            assert 'model_name' in data, "YAML should have model_name"
            assert 'layers' in data, "YAML should have layers"
            assert isinstance(data['layers'], dict), "layers should be a dict"
    
    def test_yaml_layer_fields(self, processor):
        """Verify each layer in YAML has all required fields."""
        model = SimpleLinearModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            processor.process_graph(graph, tmpdir, "test_model", model)
            
            yaml_path = Path(tmpdir) / "pytorch_graph.yaml"
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            
            required_fields = [
                'type', 'node_class', 'input_shapes', 'output_shapes',
                'weight_nodes', 'weight_shapes', 'module_args', 'connections'
            ]
            
            for layer_id, layer_data in data['layers'].items():
                for field in required_fields:
                    assert field in layer_data, \
                        f"Layer {layer_id} missing field: {field}"
                
                # Check connections structure
                assert 'inputs' in layer_data['connections'], \
                    f"Layer {layer_id} connections missing inputs"
                assert 'outputs' in layer_data['connections'], \
                    f"Layer {layer_id} connections missing outputs"
    
    def test_yaml_connections_populated(self, processor):
        """Verify YAML connections are populated (not all empty)."""
        model = TwoLayerModel()
        x = torch.randn(2, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            processor.process_graph(graph, tmpdir, "test_model", model)
            
            yaml_path = Path(tmpdir) / "pytorch_graph.yaml"
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            
            # Count layers with connections
            layers_with_inputs = 0
            layers_with_outputs = 0
            
            for layer_id, layer_data in data['layers'].items():
                if layer_data['connections']['inputs']:
                    layers_with_inputs += 1
                if layer_data['connections']['outputs']:
                    layers_with_outputs += 1
            
            assert layers_with_inputs > 0, \
                "At least some layers should have input connections"
            assert layers_with_outputs > 0, \
                "At least some layers should have output connections"


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestAttentionModel:
    """Test extraction of attention-like models with multiple linear layers."""
    
    def test_attention_linear_layers(self, processor):
        """Verify all linear layers in attention model are extracted."""
        model = AttentionModel(hidden_size=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Should have 4 linear layers: q, k, v, out projections
            linear_nodes = [n for n in nodes if 'linear' in n.node_type.lower()]
            assert len(linear_nodes) >= 4, \
                f"Should have at least 4 linear nodes, got {len(linear_nodes)}"
    
    def test_attention_matmul_operations(self, processor):
        """Verify matmul operations are extracted."""
        model = AttentionModel(hidden_size=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = draw_graph(model, input_data=x, expand_nested=True)
            nodes = processor.process_graph(graph, tmpdir, "test_model", model)
            
            # Should have matmul operations for Q@K and attn@V
            matmul_nodes = [n for n in nodes if 'matmul' in n.node_type.lower()]
            assert len(matmul_nodes) >= 2, \
                f"Should have at least 2 matmul nodes, got {len(matmul_nodes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

