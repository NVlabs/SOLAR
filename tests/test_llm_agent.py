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

"""Tests for LLM agent and node type registry."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from solar.einsum.llm_agent import AgentConfig, NodeTypeConversionAgent
from solar.einsum.node_type_registry import (
    NodeTypeHandler,
    NodeTypeRegistry,
    NodeTypeHandlerFactory,
    DefaultNodeExpansionStrategy
)


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig(api_key="test_key")
        
        assert config.api_key == "test_key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.2
        assert config.max_tokens == 2000
        assert config.cache_dir == "./llm_handlers_cache"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            api_key="test_key",
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000,
            cache_dir="/tmp/cache"
        )
        
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.cache_dir == "/tmp/cache"


class TestNodeTypeConversionAgent:
    """Tests for NodeTypeConversionAgent."""
    
    @pytest.fixture
    def agent(self, tmp_path):
        """Create agent with mocked OpenAI."""
        config = AgentConfig(
            api_key="test_key",
            cache_dir=str(tmp_path / "cache")
        )
        
        with patch('solar.einsum.llm_agent.openai'):
            agent = NodeTypeConversionAgent(config)
            agent.client = Mock()
            return agent
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.config.api_key == "test_key"
        assert Path(agent.config.cache_dir).exists()
    
    def test_create_conversion_prompt(self, agent):
        """Test prompt creation."""
        node_data = {
            "input_shapes": [[1, 10, 512]],
            "output_shapes": [[1, 10, 512]],
            "module_args": {"num_heads": 8}
        }
        
        prompt = agent._create_conversion_prompt("attention", node_data)
        
        assert "attention" in prompt
        assert "input_shapes" in prompt
        assert "num_heads" in prompt
        assert "def create_attention_subgraph" in prompt
    
    def test_extract_code(self, agent):
        """Test code extraction from LLM response."""
        # Test with code block
        response = """
Here's the function:
```python
def create_test_subgraph(node_id, node_data):
    return {}
```
"""
        code = agent._extract_code(response)
        assert code.startswith("def create_test_subgraph")
        
        # Test without code block
        response = "def create_test_subgraph(node_id, node_data): return {}"
        code = agent._extract_code(response)
        assert code.startswith("def create_test_subgraph")
    
    def test_validate_code(self, agent):
        """Test code validation."""
        # Valid code
        code = "def create_attention_subgraph(node_id, node_data): return {}"
        validated = agent._validate_code(code, "attention")
        assert validated == code
        
        # Invalid code (not a function)
        code = "class TestClass: pass"
        with pytest.raises(ValueError):
            agent._validate_code(code, "attention")
        
        # Wrong function name (should be fixed)
        code = "def wrong_name(node_id, node_data): return {}"
        validated = agent._validate_code(code, "attention")
        assert "create_attention_subgraph" in validated
    
    def test_generate_fallback(self, agent):
        """Test fallback generation."""
        fallback = agent._generate_fallback("custom_op")
        
        assert "def create_custom_op_subgraph" in fallback
        assert "identity" in fallback
        assert "return subgraph" in fallback
    
    def test_caching(self, agent):
        """Test code caching."""
        node_type = "test_op"
        code = "def create_test_op_subgraph(node_id, node_data): return {}"
        
        # Cache the code
        agent._cache_code(node_type, code)
        
        # Check it can be retrieved
        cached = agent._check_cache(node_type)
        assert cached == code
        
        # Check file exists
        cache_file = Path(agent.config.cache_dir) / f"{node_type}.py"
        assert cache_file.exists()
    
    @patch('solar.einsum.llm_agent.OpenAI')
    def test_generate_conversion_code(self, mock_openai_class, agent):
        """Test full code generation flow."""
        # Setup mock response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="""
def create_custom_op_subgraph(node_id: str, node_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {"test": {"node_type": "identity"}}
"""))]
        
        mock_client.chat.completions.create.return_value = mock_response
        agent.client = mock_openai_class.return_value
        agent.client.chat.completions.create.return_value = mock_response
        
        node_data = {"input_shapes": [[1, 10]]}
        code, metadata = agent.generate_conversion_code("custom_op", node_data)
        
        assert "def create_custom_op_subgraph" in code
        assert metadata["source"] == "generated"
        assert not metadata["is_fallback"]


class TestNodeTypeHandler:
    """Tests for NodeTypeHandler."""
    
    def test_basic_handler(self):
        """Test basic handler creation."""
        handler = NodeTypeHandler(
            node_type="test_op",
            create_subgraph_func=lambda node_id, node_data: {},
            generate_einsum_func=None
        )
        
        assert handler.node_type == "test_op"
        assert handler.can_expand() is True
        assert handler.can_generate_einsum() is False
    
    def test_handler_with_einsum(self):
        """Test handler with einsum generation."""
        handler = NodeTypeHandler(
            node_type="matmul",
            create_subgraph_func=None,
            generate_einsum_func=lambda x, y: "AB,BC->AC"
        )
        
        assert handler.can_expand() is False
        assert handler.can_generate_einsum() is True
        
        equation = handler.generate_einsum_func([2, 3], [3, 4])
        assert equation == "AB,BC->AC"


class TestNodeTypeRegistry:
    """Tests for NodeTypeRegistry."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create registry instance."""
        return NodeTypeRegistry(cache_dir=str(tmp_path / "handlers"))
    
    def test_register_handler(self, registry):
        """Test registering a handler."""
        handler = NodeTypeHandler(
            node_type="custom",
            create_subgraph_func=lambda x, y: {}
        )
        
        registry.register("custom", handler)
        
        retrieved = registry.get_handler("custom")
        assert retrieved is handler
    
    def test_list_handlers(self, registry):
        """Test listing registered handlers."""
        # Register some handlers
        for i in range(3):
            handler = NodeTypeHandler(f"handler_{i}", None)
            registry.register(f"handler_{i}", handler)
        
        handlers = registry.list_handlers()
        assert len(handlers) == 3
        assert "handler_0" in handlers
    
    def test_list_expandable(self, registry):
        """Test listing expandable handlers."""
        # Register expandable handler
        expandable = NodeTypeHandler(
            "expandable",
            create_subgraph_func=lambda x, y: {}
        )
        registry.register("expandable", expandable)
        
        # Register non-expandable handler
        non_expandable = NodeTypeHandler(
            "non_expandable",
            create_subgraph_func=None
        )
        registry.register("non_expandable", non_expandable)
        
        expandable_list = registry.list_expandable()
        assert "expandable" in expandable_list
        assert "non_expandable" not in expandable_list
    
    def test_save_and_load_handler(self, registry):
        """Test saving and loading generated handlers."""
        handler = NodeTypeHandler(
            node_type="saved_handler",
            create_subgraph_func=lambda x, y: {"test": {}},
            metadata={"source": "generated"}
        )
        
        # Save handler
        registry.save_generated_handler("saved_handler", handler)
        
        # Check file exists
        handler_file = Path(registry.cache_dir) / "saved_handler_handler.json"
        assert handler_file.exists()
    
    def test_handler_factory(self):
        """Test NodeTypeHandlerFactory."""
        # Create from methods
        handler = NodeTypeHandlerFactory.create_handler_from_methods(
            node_type="test",
            create_subgraph_method=lambda x, y: {},
            generate_einsum_method=lambda x: "A->A"
        )
        
        assert handler.node_type == "test"
        assert handler.can_expand() is True
        assert handler.can_generate_einsum() is True
        
        # Create from code
        code = """
def create_test_subgraph(node_id, node_data):
    return {"node": {"type": "identity"}}
"""
        handler = NodeTypeHandlerFactory.create_handler_from_code(
            "test", code, {}
        )
        
        assert handler.node_type == "test"
        assert handler.can_expand() is True


class TestDefaultNodeExpansionStrategy:
    """Tests for DefaultNodeExpansionStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        registry = NodeTypeRegistry()
        return DefaultNodeExpansionStrategy(registry)
    
    def test_should_expand_complex_ops(self, strategy):
        """Test expansion decision for complex operations."""
        # Attention should be expanded
        node_data = {"type": "attention"}
        assert strategy.should_expand("node1", node_data) is True
        
        # Multi-head attention should be expanded
        node_data = {"type": "multi_head_attention"}
        assert strategy.should_expand("node2", node_data) is True
        
        # LSTM should be expanded
        node_data = {"type": "lstm"}
        assert strategy.should_expand("node3", node_data) is True
    
    def test_should_not_expand_basic_ops(self, strategy):
        """Test expansion decision for basic operations."""
        # Conv2d should not be expanded
        node_data = {"type": "conv2d"}
        assert strategy.should_expand("node1", node_data) is False
        
        # ReLU should not be expanded
        node_data = {"type": "relu"}
        assert strategy.should_expand("node2", node_data) is False
        
        # MatMul should not be expanded
        node_data = {"type": "matmul"}
        assert strategy.should_expand("node3", node_data) is False


class TestKernelbenchCompatibility:
    """Test compatibility with kernelbench models."""
    
    @patch('solar.einsum.llm_agent.openai')
    def test_handle_kernelbench_unknown_ops(self, mock_openai, tmp_path):
        """Test handling unknown ops from kernelbench."""
        config = AgentConfig(
            api_key="test_key",
            cache_dir=str(tmp_path / "cache")
        )
        agent = NodeTypeConversionAgent(config)
        
        # Kernelbench-style unknown op
        kb_node = {
            "node_type": "CustomKernelOp",
            "input_shapes": [[32, 512]],
            "module_args": {"custom_param": 42}
        }
        
        # Should generate appropriate prompt
        prompt = agent._create_conversion_prompt("CustomKernelOp", kb_node)
        assert "CustomKernelOp" in prompt
        assert "custom_param" in prompt
