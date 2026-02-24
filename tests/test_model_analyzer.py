"""Tests for model analyzer with LLM agent support."""

import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

from solar.analysis.model_analyzer import ModelAnalyzer
from solar.einsum.llm_agent import AgentConfig, NodeTypeConversionAgent
from solar.einsum.node_type_registry import NodeTypeHandler, NodeTypeRegistry
from solar.common.types import AnalysisResult


class TestModelAnalyzer:
    """Tests for ModelAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance without LLM agent."""
        return ModelAnalyzer(debug=True, enable_agent=False)
    
    @pytest.fixture
    def analyzer_with_agent(self):
        """Create analyzer instance with mock LLM agent."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            analyzer = ModelAnalyzer(
                debug=True,
                enable_agent=True,
                api_key='test_key'
            )
            # Mock the agent
            analyzer.agent = Mock(spec=NodeTypeConversionAgent)
            return analyzer
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.debug is True
        assert analyzer.enable_agent is False
        assert analyzer.agent is None
        assert isinstance(analyzer.registry, NodeTypeRegistry)
    
    def test_builtin_handlers_registered(self, analyzer):
        """Test built-in handlers are registered."""
        # Check some common handlers
        handlers = [
            "matmul", "linear", "conv2d", "relu", "sum", "prod"
        ]
        
        for handler_name in handlers:
            handler = analyzer.registry.get_handler(handler_name)
            assert handler is not None
            assert isinstance(handler, NodeTypeHandler)
    
    def test_convert_torchview_to_model_info(self, analyzer, sample_torchview_nodes):
        """Test converting torchview nodes to model info."""
        model_info = analyzer._convert_torchview_to_model_info(sample_torchview_nodes)
        
        assert "model_name" in model_info
        assert "layers" in model_info
        assert len(model_info["layers"]) == 2
        
        # Check first layer
        first_layer = model_info["layers"]["Model.conv1"]
        assert first_layer["type"] == "conv2d"
        assert first_layer["input_shapes"] == [[1, 3, 224, 224]]
        assert first_layer["output_shapes"] == [[1, 64, 112, 112]]
    
    def test_build_graph_from_model_info(self, analyzer):
        """Test building NetworkX graph from model info."""
        model_info = {
            "model_name": "test_model",
            "layers": {
                "conv1": {
                    "type": "conv2d",
                    "input_shapes": [[1, 3, 224, 224]],
                    "output_shapes": [[1, 64, 112, 112]],
                    "connections": {
                        "inputs": [],
                        "outputs": ["relu1"]
                    }
                },
                "relu1": {
                    "type": "relu",
                    "input_shapes": [[1, 64, 112, 112]],
                    "output_shapes": [[1, 64, 112, 112]],
                    "connections": {
                        "inputs": ["conv1"],
                        "outputs": []
                    }
                }
            }
        }
        
        graph = analyzer._build_graph_from_model_info(model_info)
        
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) == 2
        assert "conv1" in graph.nodes
        assert "relu1" in graph.nodes
        assert graph.has_edge("conv1", "relu1")
    
    def test_extract_shapes_from_node(self, analyzer):
        """Test extracting shapes from node data."""
        node_data = {
            "input_shapes": [[1, 3, 224, 224]],
            "output_shapes": [[1, 64, 112, 112]],
            "weight_shapes": [[64, 3, 7, 7], [64]]
        }
        
        shapes = analyzer._extract_shapes_from_node(node_data)
        
        assert shapes["Input"] == [1, 3, 224, 224]
        assert shapes["Output"] == [1, 64, 112, 112]
        assert shapes["Weight_0"] == [64, 3, 7, 7]
        assert shapes["Weight_1"] == [64]
    
    def test_analyze_node(self, analyzer):
        """Test analyzing a single node."""
        node_data = {
            "type": "conv2d",
            "input_shapes": [[1, 3, 32, 32]],
            "output_shapes": [[1, 16, 30, 30]],
            "weight_shapes": [[16, 3, 3, 3]],
            "module_args": {"stride": [1, 1], "padding": [0, 0]}
        }
        
        analysis = analyzer._analyze_node("test_conv", node_data)
        
        assert analysis is not None
        assert analysis["node_id"] == "test_conv"
        assert analysis["node_type"] == "conv2d"
        assert analysis["compute_macs"] > 0
        assert "memory_elements" in analysis
        assert "einsum_equation" in analysis
    
    @patch('solar.analysis.model_analyzer.NodeTypeConversionAgent')
    def test_handle_unknown_node_type(self, mock_agent_class, analyzer_with_agent):
        """Test handling unknown node type with LLM agent."""
        # Setup mock agent
        mock_agent = analyzer_with_agent.agent
        mock_agent.generate_conversion_code.return_value = (
            "def create_custom_op_subgraph(node_id, node_data): return {}",
            {"source": "generated"}
        )
        
        node_data = {
            "type": "unknown_op",
            "input_shapes": [[1, 10]],
            "output_shapes": [[1, 20]]
        }
        
        result = analyzer_with_agent._handle_unknown_node_type("unknown_op", node_data)
        
        mock_agent.generate_conversion_code.assert_called_once()
    
    def test_expand_complex_operations(self, analyzer):
        """Test expanding complex operations into subgraphs."""
        # Create a graph with an expandable node
        graph = nx.DiGraph()
        graph.add_node("attention", type="attention", 
                      input_shapes=[[1, 10, 512]], 
                      output_shapes=[[1, 10, 512]])
        
        # Mock expansion strategy
        analyzer.expansion_strategy.should_expand = Mock(return_value=True)
        
        # Test expansion
        expanded = analyzer.expand_complex_operations_in_graph(graph)
        
        # The mock doesn't actually expand, so graph should be same
        assert len(expanded.nodes) == len(graph.nodes)
    
    def test_calculate_roofline_performance(self, analyzer):
        """Test roofline performance calculation."""
        arch_config = {
            "freq_GHz": 1.8,
            "MAC_per_cycle_fp32_tc": 1000,
            "DRAM_byte_per_cycle": 500
        }
        
        perf = analyzer._calculate_roofline_performance(
            compute_macs=1000000,
            memory_elements=10000,
            arch_config=arch_config,
            precision="fp32"
        )
        
        assert "compute_cycles" in perf
        assert "memory_cycles" in perf
        assert "runtime_ms" in perf
        assert "bottleneck" in perf
        assert perf["bottleneck"] in ["compute", "memory"]
    
    def test_save_analysis_json(self, analyzer, tmp_path):
        """Test saving analysis to JSON."""
        analysis = AnalysisResult(
            layers={"conv1": {"compute_macs": 100}},
            total={"compute_macs": 100, "num_layers": 1},
            roofline_performance={"runtime_ms": 0.1},
            metadata={"arch_config": "H100_PCIe"}
        )
        
        output_path = tmp_path / "analysis.json"
        analyzer.save_analysis(analysis, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["total"]["compute_macs"] == 100
        assert data["metadata"]["arch_config"] == "H100_PCIe"
    
    def test_save_analysis_yaml(self, analyzer, tmp_path):
        """Test saving analysis to YAML."""
        analysis = AnalysisResult(
            layers={"conv1": {"compute_macs": 100}},
            total={"compute_macs": 100, "num_layers": 1},
            roofline_performance={"runtime_ms": 0.1},
            metadata={"arch_config": "A6000"}
        )
        
        output_path = tmp_path / "analysis.yaml"
        analyzer.save_analysis(analysis, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = yaml.safe_load(f)
        
        assert data["total"]["compute_macs"] == 100
        assert data["metadata"]["arch_config"] == "A6000"
    
    def test_analyze_model_torchview(self, analyzer, tmp_path):
        """Test analyzing a model from torchview graph."""
        # Create test graph file
        graph_file = tmp_path / "test_graph.json"
        graph_data = [
            {
                "node_id": "conv1",
                "node_type": "conv2d",
                "input_shapes": [[1, 3, 32, 32]],
                "output_shapes": [[1, 16, 30, 30]],
                "weight_shapes": [[16, 3, 3, 3]],
                "module_args": {"stride": [1, 1], "padding": [0, 0]}
            },
            {
                "node_id": "relu1",
                "node_type": "relu",
                "input_shapes": [[1, 16, 30, 30]],
                "output_shapes": [[1, 16, 30, 30]]
            }
        ]
        graph_file.write_text(json.dumps(graph_data))
        
        # Analyze
        result = analyzer.analyze_model(
            str(graph_file),
            graph_type="torchview_graph",
            arch_config="H100_PCIe",
            precision="fp32"
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.total["num_layers"] > 0
        assert result.total["compute_macs"] > 0


class TestKernelbenchCompatibility:
    """Test compatibility with kernelbench models."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer for testing."""
        return ModelAnalyzer(debug=False, enable_agent=False)
    
    def test_kernelbench_node_types(self, analyzer):
        """Test handling kernelbench node type naming conventions."""
        kernelbench_types = [
            ("Conv2d", "conv2d"),
            ("Linear", "linear"),
            ("BatchNorm2d", "batch_norm"),
            ("ReLU", "relu"),
            ("MaxPool2d", "max_pool2d")
        ]
        
        for kb_type, expected in kernelbench_types:
            node_data = {
                "type": kb_type,
                "input_shapes": [[1, 3, 32, 32]],
                "output_shapes": [[1, 3, 32, 32]]
            }
            
            analysis = analyzer._analyze_node(f"test_{kb_type}", node_data)
            assert analysis is None or isinstance(analysis, dict)
    
    def test_mixed_naming_conventions(self, analyzer):
        """Test handling mixed naming conventions in same model."""
        model_info = {
            "model_name": "mixed_model",
            "layers": {
                "layer1": {
                    "type": "Conv2d",
                    "input_shapes": [[1, 3, 32, 32]],
                    "output_shapes": [[1, 16, 30, 30]],
                    "connections": {"outputs": ["layer2"]}
                },
                "layer2": {
                    "type": "relu",
                    "input_shapes": [[1, 16, 30, 30]],
                    "output_shapes": [[1, 16, 30, 30]],
                    "connections": {"inputs": ["layer1"], "outputs": ["layer3"]}
                },
                "layer3": {
                    "type": "Linear",
                    "input_shapes": [[1, 14400]],
                    "output_shapes": [[1, 10]],
                    "connections": {"inputs": ["layer2"]}
                }
            }
        }
        
        graph = analyzer._build_graph_from_model_info(model_info)
        
        assert len(graph.nodes) == 3
        assert graph.has_edge("layer1", "layer2")
        assert graph.has_edge("layer2", "layer3")
