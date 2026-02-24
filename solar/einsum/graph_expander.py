"""Expand complex operations in computation graphs.

This module provides functionality to expand complex operations (like attention,
multi-head attention, etc.) into their constituent subgraphs of simpler operations.

Example:
    >>> from solar.einsum.graph_expander import GraphExpander
    >>> expander = GraphExpander()
    >>> expanded_graph = expander.expand(graph)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.node_type_registry import (
    DefaultNodeExpansionStrategy,
    NodeTypeHandler,
    NodeTypeHandlerFactory,
    NodeTypeRegistry,
)


class GraphExpander:
    """Expand complex operations into subgraphs of simpler operations.
    
    This class handles the expansion of complex operations like attention
    mechanisms into their constituent parts (matmul, softmax, etc.).
    
    Attributes:
        debug: Whether to print debug information.
    """

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: Optional[str] = None,
        cache_dir: str = "./solar_handlers_cache",
    ) -> None:
        """Initialize the graph expander.
        
        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key for LLM agent.
            cache_dir: Directory for caching generated handlers.
        """
        self._debug = debug
        self._enable_agent = enable_agent
        self._api_key = api_key
        self._cache_dir = cache_dir
        
        # Initialize components
        self._einsum_analyzer = EinsumAnalyzer(debug=debug)
        self._registry = NodeTypeRegistry(cache_dir=cache_dir)
        self._expansion_strategy = DefaultNodeExpansionStrategy(
            self._registry, debug=debug
        )
        
        # Register built-in handlers
        self._register_builtin_handlers()
        
        # Initialize LLM agent if enabled
        self._agent = None
        if enable_agent:
            self._init_agent(api_key)

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    @property
    def registry(self) -> NodeTypeRegistry:
        """The node type registry."""
        return self._registry

    def _init_agent(self, api_key: Optional[str]) -> None:
        """Initialize the LLM agent.
        
        Args:
            api_key: OpenAI API key.
        """
        try:
            import os
            from solar.einsum.llm_agent import (
                AgentConfig,
                NodeTypeConversionAgent,
                get_api_key_interactive,
            )
            
            # Get API key if not provided
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("\n🤖 LLM Agent is enabled for handling unknown node types.")
                    api_key = get_api_key_interactive()

            if api_key:
                agent_config = AgentConfig(
                    api_key=api_key,
                    cache_dir=self._cache_dir,
                )
                self._agent = NodeTypeConversionAgent(agent_config)
                if self._debug:
                    print(f"✅ LLM Agent initialized with model: {agent_config.model}")
            else:
                print("⚠️ No API key provided. Agent will be disabled.")
                self._agent = None
        except Exception as e:
            print(f"⚠️ Failed to initialize LLM agent: {e}")
            print("Continuing without dynamic handler generation.")
            self._agent = None

    def _register_builtin_handlers(self) -> None:
        """Register built-in node type handlers."""
        handlers = {
            # Matrix operations
            "matmul": {
                "einsum": self._einsum_analyzer.generate_matmul_einsum
            },
            "linear": {
                "einsum": self._einsum_analyzer.generate_linear_einsum
            },
            
            # Convolution operations
            "conv1d": {
                "einsum": self._einsum_analyzer.generate_conv1d_einsum
            },
            "conv2d": {
                "einsum": self._einsum_analyzer.generate_conv2d_einsum
            },
            "conv3d": {
                "einsum": self._einsum_analyzer.generate_conv3d_einsum
            },
            
            # Elementwise operations
            "relu": {
                "einsum": lambda shape, **kw: self._einsum_analyzer.generate_elementwise_einsum(shape, "relu")
            },
            "sigmoid": {
                "einsum": lambda shape, **kw: self._einsum_analyzer.generate_elementwise_einsum(shape, "sigmoid")
            },
            "tanh": {
                "einsum": lambda shape, **kw: self._einsum_analyzer.generate_elementwise_einsum(shape, "tanh")
            },
            
            # Reduction operations
            "sum": {
                "einsum": lambda shape, dims=None, **kw: self._einsum_analyzer.generate_reduction_einsum(shape, "sum", dims)
            },
            "mean": {
                "einsum": lambda shape, dims=None, **kw: self._einsum_analyzer.generate_reduction_einsum(shape, "mean", dims)
            },
            "prod": {
                "einsum": lambda shape, dims=None, **kw: self._einsum_analyzer.generate_reduction_einsum(shape, "prod", dims)
            },
        }
        
        # Register each handler
        for node_type, methods in handlers.items():
            handler = NodeTypeHandlerFactory.create_handler_from_methods(
                node_type=node_type,
                create_subgraph_method=methods.get("create"),
                generate_einsum_method=methods.get("einsum"),
                metadata={"source": "builtin"}
            )
            self._registry.register(node_type, handler)
        
        if self._debug:
            print(f"📦 Registered {len(handlers)} built-in node handlers")

    def expand(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations in the graph.
        
        Args:
            graph: Input computation graph.
            
        Returns:
            Graph with complex operations expanded into subgraphs.
        """
        if self._debug:
            print("\n🔄 Expanding complex operations...")
        
        expanded_graph = graph.copy()
        nodes_to_expand = []
        
        # Identify nodes to expand
        for node_id in expanded_graph.nodes():
            node_data = expanded_graph.nodes[node_id]
            if self._expansion_strategy.should_expand(node_id, node_data):
                nodes_to_expand.append(node_id)
        
        if self._debug:
            print(f"  Found {len(nodes_to_expand)} nodes to expand")
        
        # Expand each node
        for node_id in nodes_to_expand:
            node_data = expanded_graph.nodes[node_id].copy()
            node_type = node_data.get("type", node_data.get("node_type", ""))
            
            # Try to expand
            subgraph = self._expand_node(node_id, node_data)
            
            if subgraph:
                # Get connections
                predecessors = list(expanded_graph.predecessors(node_id))
                successors = list(expanded_graph.successors(node_id))
                
                # Remove original node
                expanded_graph.remove_node(node_id)
                
                # Add subgraph nodes
                for sub_id, sub_data in subgraph.items():
                    expanded_graph.add_node(sub_id, **sub_data)
                
                # Connect subgraph
                self._connect_subgraph(
                    expanded_graph, subgraph, predecessors, successors
                )

                if self._debug:
                    print(f"  ✅ Expanded {node_id} ({node_type}) into {len(subgraph)} nodes")
            else:
                if self._debug:
                    print(f"  ⚠️ Could not expand {node_id} ({node_type})")
        
        return expanded_graph

    # Backward compatibility alias
    expand_complex_operations_in_graph = expand

    def _expand_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Expand a single node into a subgraph.
        
        Args:
            node_id: Node identifier.
            node_data: Node data.
            
        Returns:
            Subgraph dictionary or None.
        """
        node_type = node_data.get("type", node_data.get("node_type", ""))
        
        # Check registry for handler
        handler = self._registry.get_handler(node_type)
        
        if handler and handler.can_expand():
            return handler.create_subgraph_func(node_id, node_data)
        
        # Try LLM agent if enabled
        if self._agent:
            return self._handle_unknown_node_type(node_type, node_data)
        
        return None

    def _handle_unknown_node_type(
        self,
        node_type: str,
        node_data: Dict[str, Any],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Handle unknown node type using LLM agent.
        
        Args:
            node_type: Unknown node type.
            node_data: Node data.
            
        Returns:
            Subgraph dictionary or None.
        """
        if not self._agent:
            return None
        
        print(f"\n🤖 Unknown node type: {node_type}")
        print("  Generating handler with LLM agent...")
        
        try:
            # Generate handler code
            code, metadata = self._agent.generate_conversion_code(
                node_type, node_data
            )
            
            # Create and register handler
            handler = NodeTypeHandlerFactory.create_handler_from_code(
                node_type, code, metadata
            )
            self._registry.register(node_type, handler)
            self._registry.save_generated_handler(node_type, handler)
            
            print(f"  ✅ Handler generated and registered")
            
            # Use the new handler
            if handler.can_expand():
                return handler.create_subgraph_func(
                    node_data.get("node_id", node_type), node_data
                )
                
        except Exception as e:
            print(f"  ❌ Failed to generate handler: {e}")
        
        return None

    def _connect_subgraph(
        self,
        graph: nx.DiGraph,
        subgraph: Dict[str, Dict[str, Any]],
        predecessors: List[str],
        successors: List[str],
    ) -> None:
        """Connect a subgraph to the main graph.
        
        Args:
            graph: Main graph.
            subgraph: Subgraph nodes.
            predecessors: Predecessor nodes.
            successors: Successor nodes.
        """
        if not subgraph:
            return
        
        # Simple connection: first node gets inputs, last gives outputs
        sub_ids = list(subgraph.keys())
        if sub_ids:
            # Connect predecessors to first subgraph node
            for pred in predecessors:
                if pred in graph:
                    graph.add_edge(pred, sub_ids[0])
            
            # Connect last subgraph node to successors
            for succ in successors:
                if succ in graph:
                    graph.add_edge(sub_ids[-1], succ)
            
            # Connect subgraph nodes sequentially
            for i in range(len(sub_ids) - 1):
                graph.add_edge(sub_ids[i], sub_ids[i + 1])


__all__ = [
    "GraphExpander",
]

