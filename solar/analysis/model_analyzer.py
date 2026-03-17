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

"""Model analyzer with LLM agent support for dynamic node handling.

This module provides comprehensive model analysis capabilities with
support for unknown node types through LLM-based code generation.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from solar.einsum import EinsumAnalyzer
from solar.einsum.llm_agent import AgentConfig, NodeTypeConversionAgent, get_api_key_interactive
from solar.einsum.node_type_registry import (
    DefaultNodeExpansionStrategy,
    NodeTypeHandler,
    NodeTypeHandlerFactory,
    NodeTypeRegistry,
)
from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.types import AnalysisResult, NodeInfo, TensorShapes
from solar.common.utils import convert_numpy_types, ensure_directory, format_number


class ModelAnalyzer:
    """Advanced model analyzer with LLM agent support.
    
    This analyzer can handle both known and unknown node types,
    using an LLM agent to dynamically generate handlers for new operations.
    """
    
    def __init__(self,
                 debug: bool = False,
                 enable_agent: bool = False,
                 api_key: Optional[str] = None,
                 cache_dir: str = "./solar_handlers_cache"):
        """Initialize the model analyzer.
        
        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key (will prompt if not provided and agent enabled).
            cache_dir: Directory for caching generated handlers.
        """
        self.debug = debug
        self.enable_agent = enable_agent
        self.cache_dir = cache_dir
        
        # Initialize components
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)
        self.registry = NodeTypeRegistry(cache_dir=cache_dir)
        self.expansion_strategy = DefaultNodeExpansionStrategy(
            self.registry, debug=debug
        )
        
        # Register built-in handlers
        self._register_builtin_handlers()
        
        # Initialize LLM agent if enabled
        self.agent = None
        if enable_agent:
            self._init_agent(api_key)
    
    def _init_agent(self, api_key: Optional[str]) -> None:
        """Initialize the LLM agent.
        
        Args:
            api_key: OpenAI API key.
        """
        try:
            # Get API key if not provided.
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("\n🤖 LLM Agent is enabled for handling unknown node types.")
                    api_key = get_api_key_interactive()

            if api_key:
                agent_config = AgentConfig(
                    api_key=api_key,
                    cache_dir=self.cache_dir,
                )
                self.agent = NodeTypeConversionAgent(agent_config)
                if self.debug:
                    print(f"✅ LLM Agent initialized with model: {agent_config.model}")
            else:
                print("⚠️ No API key provided. Agent will be disabled.")
                self.agent = None
        except Exception as e:
            print(f"⚠️ Failed to initialize LLM agent: {e}")
            print("Continuing without dynamic handler generation.")
            self.agent = None
    
    def _register_builtin_handlers(self) -> None:
        """Register built-in node type handlers."""
        # This would include all the built-in handlers
        # For brevity, I'll include a subset here
        
        handlers = {
            # Matrix operations
            "matmul": {
                "einsum": self.einsum_analyzer.generate_matmul_einsum
            },
            "linear": {
                "einsum": self.einsum_analyzer.generate_linear_einsum
            },
            
            # Convolution operations
            "conv1d": {
                "einsum": self.einsum_analyzer.generate_conv1d_einsum
            },
            "conv2d": {
                "einsum": self.einsum_analyzer.generate_conv2d_einsum
            },
            "conv3d": {
                "einsum": self.einsum_analyzer.generate_conv3d_einsum
            },
            
            # Elementwise operations
            "relu": {
                "einsum": lambda shape, **kw: self.einsum_analyzer.generate_elementwise_einsum(shape, "relu")
            },
            "sigmoid": {
                "einsum": lambda shape, **kw: self.einsum_analyzer.generate_elementwise_einsum(shape, "sigmoid")
            },
            "tanh": {
                "einsum": lambda shape, **kw: self.einsum_analyzer.generate_elementwise_einsum(shape, "tanh")
            },
            
            # Reduction operations
            "sum": {
                "einsum": lambda shape, dims=None, **kw: self.einsum_analyzer.generate_reduction_einsum(shape, "sum", dims)
            },
            "mean": {
                "einsum": lambda shape, dims=None, **kw: self.einsum_analyzer.generate_reduction_einsum(shape, "mean", dims)
            },
            "prod": {
                "einsum": lambda shape, dims=None, **kw: self.einsum_analyzer.generate_reduction_einsum(shape, "prod", dims)
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
            self.registry.register(node_type, handler)
        
        if self.debug:
            print(f"📦 Registered {len(handlers)} built-in node handlers")
    
    def analyze_model(self,
                     model_path: str,
                     graph_type: str = "torchview_graph",
                     arch_config: str = "H100_PCIe",
                     precision: str = DEFAULT_PRECISION) -> AnalysisResult:
        """Analyze a model from a graph file.
        
        Args:
            model_path: Path to the model graph file.
            graph_type: Type of graph ("einsum_graph" or "torchview_graph").
            arch_config: Architecture configuration name.
            precision: Precision for calculations.
            
        Returns:
            AnalysisResult with comprehensive analysis.
        """
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Analyzing model: {model_path}")
            print(f"Graph type: {graph_type}")
            print(f"Architecture: {arch_config}")
            print(f"Precision: {precision}")
            print(f"{'='*60}")
        
        # Load the graph
        if graph_type == "torchview_graph":
            with open(model_path) as f:
                nodes_data = json.load(f)
            model_info = self._convert_torchview_to_model_info(nodes_data)
        else:
            with open(model_path) as f:
                model_info = yaml.safe_load(f)
        
        # Build graph
        graph = self._build_graph_from_model_info(model_info)
        
        # Expand complex operations
        if self.enable_agent or self.registry.list_expandable():
            graph = self.expand_complex_operations_in_graph(graph)
        
        # Analyze each layer
        layers_analysis = {}
        total_compute = 0
        total_memory = 0
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            analysis = self._analyze_node(node_id, node_data)
            
            if analysis:
                layers_analysis[node_id] = analysis
                total_compute += analysis.get("compute_macs", 0)
                total_memory += analysis.get("memory_elements", {}).get("total", 0)
        
        # Calculate operational intensity
        op_intensity = total_compute / total_memory if total_memory > 0 else 0
        
        # Get architecture configuration
        arch_cfg = self._load_arch_config(arch_config)
        
        # Calculate roofline performance
        roofline = self._calculate_roofline_performance(
            total_compute, total_memory, arch_cfg, precision
        )
        
        return AnalysisResult(
            layers=layers_analysis,
            total={
                "compute_macs": total_compute,
                "memory_elements": total_memory,
                "operational_intensity": op_intensity,
                "num_layers": len(layers_analysis)
            },
            roofline_performance=roofline,
            metadata={
                "arch_config": arch_config,
                "precision": precision,
                "graph_type": graph_type,
                "agent_enabled": self.agent is not None
            }
        )

    def _load_arch_config(self, arch_config: str) -> Dict[str, Any]:
        """Load an architecture YAML by name or path.
        
        Supports either:
        - A full path to a YAML file
        - A config name looked up under `solar/configs/arch/<name>.yaml`
        """
        # Explicit path.
        cfg_path = Path(arch_config)
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}

        # Look under solar root: solar/configs/arch/<name>.yaml
        solar_root = Path(__file__).resolve().parents[2]
        candidate = solar_root / "configs" / "arch" / f"{arch_config}.yaml"
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f) or {}

        # Backward-compatible fallback.
        fallback = solar_root / "configs" / "arch" / "H100_PCIe.yaml"
        if fallback.exists():
            with open(fallback) as f:
                return yaml.safe_load(f) or {}

        return {}
    
    def _convert_torchview_to_model_info(self,
                                        nodes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert torchview graph to model info format.
        
        Args:
            nodes_data: List of node dictionaries from torchview.
            
        Returns:
            Model info dictionary.
        """
        model_info = {
            "model_name": "torchview_model",
            "layers": {}
        }
        
        for node_info in nodes_data:
            node_id = node_info.get("node_id", "")
            layer_info = {
                "type": node_info.get("node_type", "unknown"),
                "input_shapes": node_info.get("input_shapes", []),
                "output_shapes": node_info.get("output_shapes", []),
                "weight_shapes": node_info.get("weight_shapes", []),
                "module_args": node_info.get("module_args", {}),
                "connections": {
                    "inputs": node_info.get("input_nodes", []),
                    "outputs": node_info.get("output_nodes", [])
                }
            }
            model_info["layers"][node_id] = layer_info
        
        return model_info
    
    def _build_graph_from_model_info(self,
                                    model_info: Dict[str, Any]) -> nx.DiGraph:
        """Build a directed graph from model info.
        
        Args:
            model_info: Model information dictionary.
            
        Returns:
            NetworkX directed graph.
        """
        graph = nx.DiGraph()
        
        for layer_id, layer_info in model_info.get("layers", {}).items():
            # Add node with all its data
            graph.add_node(layer_id, **layer_info)

            # Add edges based on connections
            connections = layer_info.get("connections", {})
            for output_id in connections.get("outputs", []):
                if output_id in model_info.get("layers", {}):
                    graph.add_edge(layer_id, output_id)
        
        return graph
    
    def expand_complex_operations_in_graph(self,
                                          graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations into subgraphs.
        
        Args:
            graph: Input graph.
            
        Returns:
            Expanded graph.
        """
        if self.debug:
            print("\n🔄 Expanding complex operations...")
        
        expanded_graph = graph.copy()
        nodes_to_expand = []
        
        # Identify nodes to expand
        for node_id in expanded_graph.nodes():
            node_data = expanded_graph.nodes[node_id]
            if self.expansion_strategy.should_expand(node_id, node_data):
                nodes_to_expand.append(node_id)
        
        if self.debug:
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

                if self.debug:
                    print(f"  ✅ Expanded {node_id} ({node_type}) into {len(subgraph)} nodes")
            else:
                if self.debug:
                    print(f"  ⚠️ Could not expand {node_id} ({node_type})")
        
        return expanded_graph
    
    def _expand_node(self,
                    node_id: str,
                    node_data: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Expand a single node into a subgraph.
        
        Args:
            node_id: Node identifier.
            node_data: Node data.
            
        Returns:
            Subgraph dictionary or None.
        """
        node_type = node_data.get("type", node_data.get("node_type", ""))
        
        # Check registry for handler
        handler = self.registry.get_handler(node_type)
        
        if handler and handler.can_expand():
            return handler.create_subgraph_func(node_id, node_data)
        
        # Try LLM agent if enabled
        if self.agent:
            return self._handle_unknown_node_type(node_type, node_data)
        
        return None
    
    def _handle_unknown_node_type(self,
                                 node_type: str,
                                 node_data: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Handle unknown node type using LLM agent.
        
        Args:
            node_type: Unknown node type.
            node_data: Node data.
            
        Returns:
            Subgraph dictionary or None.
        """
        if not self.agent:
            return None
        
        print(f"\n🤖 Unknown node type: {node_type}")
        print("  Generating handler with LLM agent...")
        
        try:
            # Generate handler code
            code, metadata = self.agent.generate_conversion_code(
                node_type, node_data
            )
            
            # Create and register handler
            handler = NodeTypeHandlerFactory.create_handler_from_code(
                node_type, code, metadata
            )
            self.registry.register(node_type, handler)
            self.registry.save_generated_handler(node_type, handler)
            
            print(f"  ✅ Handler generated and registered")
            
            # Use the new handler
            if handler.can_expand():
                return handler.create_subgraph_func(
                    node_data.get("node_id", node_type), node_data
                )
                
        except Exception as e:
            print(f"  ❌ Failed to generate handler: {e}")
        
        return None
    
    def _connect_subgraph(self,
                                   graph: nx.DiGraph,
                         subgraph: Dict[str, Dict[str, Any]],
                         predecessors: List[str],
                         successors: List[str]) -> None:
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
    
    def _analyze_node(self,
                     node_id: str,
                     node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single node.
        
        Args:
            node_id: Node identifier.
            node_data: Node data.
            
        Returns:
            Analysis results or None.
        """
        node_type = node_data.get("type", node_data.get("node_type", ""))
        
        # Skip certain node types
        if node_type in ["input-tensor", "output-tensor", "parameter"]:
            return None
        
        analysis = {
            "node_id": node_id,
            "node_type": node_type,
            "compute_macs": 0,
            "memory_elements": {"total": 0}
        }
        
        # Get handler for einsum generation
        handler = self.registry.get_handler(node_type)
        
        if handler and handler.can_generate_einsum():
            try:
                # Prepare shapes.
                shapes = self._extract_shapes_from_node(node_data)

                # Generate einsum.
                einsum_op = self._call_einsum_generator(
                    handler, node_type, node_data, shapes
                )

                if einsum_op:
                    # Positional shapes for compute cost (matches EinsumOp operand order).
                    tensor_shapes = TensorShapes(
                        inputs=list(node_data.get("input_shapes", []) or [])
                        + list(node_data.get("weight_shapes", []) or []),
                        outputs=list(node_data.get("output_shapes", []) or []),
                    )
                    # Calculate costs.
                    analysis["compute_macs"] = einsum_op.get_compute_cost(tensor_shapes)
                    analysis["memory_elements"] = self.einsum_analyzer.get_memory_cost(shapes)
                    analysis["einsum_equation"] = einsum_op.equation
            except Exception as e:
                if self.debug:
                    print(f"  Error analyzing {node_id}: {e}")
        
        return analysis
    
    def _extract_shapes_from_node(self,
                                 node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shapes from node data.
        
        Args:
            node_data: Node data dictionary.
            
        Returns:
            Dictionary of tensor shapes.
        """
        shapes = {}
        
        # Input shapes
        input_shapes = node_data.get("input_shapes", [])
        if input_shapes:
            if len(input_shapes) == 1:
                shapes["Input"] = input_shapes[0]
            else:
                for i, shape in enumerate(input_shapes):
                    shapes[f"Input_{i}"] = shape
        
        # Weight shapes
        weight_shapes = node_data.get("weight_shapes", [])
        if weight_shapes:
            if len(weight_shapes) == 1:
                shapes["Weight"] = weight_shapes[0]
            else:
                for i, shape in enumerate(weight_shapes):
                    shapes[f"Weight_{i}"] = shape
        
        # Output shapes
        output_shapes = node_data.get("output_shapes", [])
        if output_shapes:
            if len(output_shapes) == 1:
                shapes["Output"] = output_shapes[0]
            else:
                for i, shape in enumerate(output_shapes):
                    shapes[f"Output_{i}"] = shape
        
        return shapes
    
    def _call_einsum_generator(
        self,
        handler: NodeTypeHandler,
        node_type: str,
        node_data: Dict[str, Any],
        shapes: Dict[str, Any],
    ) -> Optional[Any]:
        """Call the einsum generator for a handler.

        Args:
            handler: Node type handler.
            node_type: Node type name.
            node_data: Node data.
            shapes: Tensor shapes.

        Returns:
            Einsum operation or None.
        """
        if not handler.generate_einsum_func:
            return None

        # Different calling conventions for different operations.
        if node_type in ["matmul", "linear"]:
            if "Input" in shapes and "Weight" in shapes:
                return handler.generate_einsum_func(shapes["Input"], shapes["Weight"])
            return None

        if node_type.startswith("conv"):
            if "Input" in shapes and "Weight" in shapes:
                module_args = node_data.get("module_args", {}) or {}
                conv_args = {}
                for key in ["stride", "padding", "dilation"]:
                    if key in module_args:
                        conv_args[key] = module_args[key]
                return handler.generate_einsum_func(shapes["Input"], shapes["Weight"], **conv_args)
            return None

        # Generic call: elementwise / reduction, etc.
        if "Input" in shapes:
            return handler.generate_einsum_func(shapes["Input"])

        return None
    
    def _calculate_roofline_performance(self,
                                       compute_macs: int,
                                       memory_elements: int,
                                       arch_config: Dict[str, Any],
                                       precision: str) -> Dict[str, Any]:
        """Calculate roofline model performance.
        
        Args:
            compute_macs: Total MAC operations.
            memory_elements: Total memory elements.
            arch_config: Architecture configuration.
            precision: Precision setting.
            
        Returns:
            Roofline performance metrics.
        """
        # Get architecture parameters
        freq_ghz = arch_config.get("freq_GHz", 1.0)
        mac_key = f"MAC_per_cycle_{precision}_tc"
        mac_per_cycle = arch_config.get(mac_key, arch_config.get("MAC_per_cycle_fp32_tc", 1000))
        dram_bw = arch_config.get("DRAM_byte_per_cycle", 1000)
        
        # Calculate memory bytes
        bytes_per_element = BYTES_PER_ELEMENT.get(precision, 4)
        memory_bytes = memory_elements * bytes_per_element
        
        # Calculate time
        compute_cycles = compute_macs / mac_per_cycle
        memory_cycles = memory_bytes / dram_bw
        total_cycles = max(compute_cycles, memory_cycles)
        
        # Runtime in ms
        runtime_ms = total_cycles / (freq_ghz * 1e6)
        
        # Determine bottleneck
        bottleneck = "compute" if compute_cycles > memory_cycles else "memory"
        
        # Utilization
        compute_util = min(compute_cycles / total_cycles, 1.0) if total_cycles > 0 else 0
        memory_util = min(memory_cycles / total_cycles, 1.0) if total_cycles > 0 else 0
        
        return {
            "compute_cycles": int(compute_cycles),
            "memory_cycles": int(memory_cycles),
            "total_cycles": int(total_cycles),
            "runtime_ms": runtime_ms,
            "bottleneck": bottleneck,
            "compute_utilization": compute_util,
            "memory_utilization": memory_util,
            "operational_intensity": compute_macs / memory_elements if memory_elements > 0 else 0
        }
    
    def save_analysis(self,
                     analysis: AnalysisResult,
                     output_path: str) -> None:
        """Save analysis results to file.
        
        Args:
            analysis: Analysis results.
            output_path: Output file path.
        """
        # Convert to dictionary
        data = {
            "layers": analysis.layers,
            "total": analysis.total,
            "roofline_performance": analysis.roofline_performance,
            "metadata": analysis.metadata
        }
        
        # Convert numpy types
        data = convert_numpy_types(data)
        
        # Save based on file extension
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        
        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        if self.debug:
            print(f"✅ Analysis saved to {output_path}")
    
    def print_summary(self, analysis: AnalysisResult) -> None:
        """Print analysis summary.
        
        Args:
            analysis: Analysis results.
        """
        print("\n" + "="*60)
        print("MODEL ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Layers: {analysis.total['num_layers']}")
        print(f"Total Compute: {format_number(analysis.total['compute_macs'])} MACs")
        print(f"Total Memory: {format_number(analysis.total['memory_elements'])} elements")
        print(f"Operational Intensity: {analysis.total['operational_intensity']:.2f} MACs/element")
        
        if analysis.roofline_performance:
            perf = analysis.roofline_performance
            print(f"\nRoofline Performance:")
            print(f"  Runtime: {perf['runtime_ms']:.2f} ms")
            print(f"  Bottleneck: {perf['bottleneck']}")
            print(f"  Compute Utilization: {perf['compute_utilization']*100:.1f}%")
            print(f"  Memory Utilization: {perf['memory_utilization']*100:.1f}%")
        
        if analysis.metadata.get("agent_enabled"):
            print(f"\n🤖 LLM Agent: Enabled")
        
        print("="*60)