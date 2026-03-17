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

"""Type definitions for the Solar package.

This module defines common types used throughout the Solar package,
following Google's Python style guide for type annotations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Type aliases for better readability
TensorShape = List[int]
NodeDict = Dict[str, Any]
EdgeList = List[Tuple[str, str]]


@dataclass
class TensorShapes:
    """Positional tensor shapes for an operation.
    with ordered lists of shapes matching the einsum operand order.
    """
    inputs: List[TensorShape] = field(default_factory=list)
    outputs: List[TensorShape] = field(default_factory=list)

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def input_rank(self, idx: int) -> int:
        """Rank (ndim) of input tensor at position idx."""
        return len(self.inputs[idx]) if idx < len(self.inputs) else 0

    def output_rank(self, idx: int) -> int:
        """Rank (ndim) of output tensor at position idx."""
        return len(self.outputs[idx]) if idx < len(self.outputs) else 0



@dataclass
class NodeInfo:
    """Information about a single node in the computation graph.
    
    Attributes:
        node_id: Unique identifier for the node.
        type: Type of operation (e.g., 'matmul', 'conv2d').
        node_class: Class of the node (e.g., 'FunctionNode', 'ModuleNode').
        input_nodes: List of input node IDs (in positional arg order).
        output_nodes: List of output node IDs.
        input_shapes: Shapes of input tensors (in positional arg order).
        output_shapes: Shapes of output tensors.
        input_dtypes: Data types of input tensors (one per input shape).
        output_dtypes: Data types of output tensors (one per output shape).
        input_types: Type classification per input: 'input' or 'weight'.
        output_types: Type classification per output: 'output'.
        module_args: Module-specific arguments.
    """
    node_id: str
    type: str
    node_class: str = "UnknownNode"
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shapes: List[TensorShape] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    output_dtypes: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    module_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> NodeDict:
        """Convert NodeInfo to a dictionary representation."""
        return {
            "type": self.type,
            "node_class": self.node_class,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "module_args": self.module_args,
            "connections": {
                "inputs": self.input_nodes,
                "outputs": self.output_nodes,
            },
        }


@dataclass
class GraphInfo:
    """Information about a computation graph.
    
    Attributes:
        nodes: List of nodes in the graph.
        edges: List of edges between nodes.
        total_nodes: Total number of nodes.
        graph_class: Class of the graph object.
        metadata: Additional metadata about the graph.
    """
    nodes: List[NodeInfo]
    edges: EdgeList = field(default_factory=list)
    total_nodes: int = 0
    graph_class: str = "ComputationGraph"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EinsumOperation:
    """Represents an einsum operation.
    
    Attributes:
        equation: The einsum equation string.
        operand_names: Names of the operands.
        operand_dims: Dimensions for each operand.
        is_output: Whether this is an output operand.
        compute_cost: Number of operations required.
        memory_cost: Memory elements accessed.
    """
    equation: str
    operand_names: List[str]
    operand_dims: List[List[str]]
    is_output: List[bool] = field(default_factory=list)
    compute_cost: Optional[int] = None
    memory_cost: Optional[Dict[str, int]] = None
    is_real_einsum: bool = True


@dataclass
class AnalysisResult:
    """Result of model analysis.
    
    Attributes:
        layers: Layer-by-layer analysis results.
        total: Total compute and memory statistics.
        fusion_analysis: Results of fusion analysis.
        roofline_performance: Roofline model results.
        metadata: Additional analysis metadata.
    """
    layers: Dict[str, Dict[str, Any]]
    total: Dict[str, Union[int, float]]
    fusion_analysis: Optional[Dict[str, Any]] = None
    roofline_performance: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureConfig:
    """Hardware architecture configuration.
    
    Attributes:
        name: Architecture name (e.g., 'H100_PCIe').
        freq_ghz: Frequency in GHz.
        mac_per_cycle: MAC operations per cycle for different precisions.
        dram_bandwidth: DRAM bandwidth in bytes per cycle.
        sram_capacity: SRAM capacity in bytes.
        power_per_chip: Power consumption per chip in watts.
    """
    name: str
    freq_ghz: float
    mac_per_cycle: Dict[str, int]
    dram_bandwidth: float
    sram_capacity: int
    power_per_chip: float = 1000.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Configuration for processing models.
    
    Attributes:
        save_graph: Whether to save graph visualizations.
        force_rerun: Force reprocessing even if output exists.
        batch_size: Number of models to process in parallel.
        timeout: Timeout for processing in seconds.
        output_dir: Directory for output files.
        debug: Enable debug output.
    """
    save_graph: bool = False
    force_rerun: bool = False
    batch_size: int = 5
    timeout: int = 600
    output_dir: str = "outputs"
    debug: bool = False
    safe_mode: bool = False
