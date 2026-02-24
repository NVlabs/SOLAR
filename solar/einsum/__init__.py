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

"""Einsum conversion module for Solar.

This module provides tools for converting PyTorch operations and graphs
to einsum notation.

Key components:
- `ops/` - Operation handlers for different PyTorch operations
- `EinsumAnalyzer` - Core einsum analysis
    - `PyTorchToEinsum` - Convert PyTorch graphs to einsum graphs
    - `EinsumRankRenamer` - Rename dimension ranks using BFS
    - `EinsumToTimeloop` - Convert einsum graphs to Timeloop workload format
    - `EinsumToTaco` - Convert einsum equations to TACO expressions
    - `EinsumGraphVisualizer` - Visualize einsum graphs as PDF
    - `GraphExpander` - Expand complex operations into subgraphs
- `BenchmarkEinsumConverter` - Convert benchmark suites to einsum graphs
- `llm_agent` - LLM-based dynamic handler generation
- `node_type_registry` - Registry for node type handlers

File naming convention:
    - `pytorch_to_einsum.py` - PyTorch graph to einsum conversion
    - `einsum_rank_renamer.py` - Dimension rank renaming
    - `einsum_to_timeloop.py` - Einsum to Timeloop conversion
    - `einsum_to_taco.py` - Einsum to TACO expression conversion
    - `einsum_graph_visualizer.py` - Graph visualization to PDF
    - `graph_expander.py` - Complex operation expansion
"""

from solar.einsum.analyzer import EinsumAnalyzer

# Main converters (new names)
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.einsum.einsum_rank_renamer import EinsumRankRenamer, rename_einsum_ranks
from solar.einsum.einsum_to_timeloop import EinsumToTimeloop, convert_to_timeloop
from solar.einsum.einsum_to_taco import EinsumToTaco, generate_taco_expression, add_taco_expressions
from solar.einsum.einsum_graph_visualizer import EinsumGraphVisualizer, save_einsum_graph_pdf
from solar.einsum.graph_expander import GraphExpander

# Backward compatibility aliases
from solar.einsum.pytorch_to_einsum import PyTorchEinsumConverter
from solar.einsum.einsum_rank_renamer import EinsumGraphRenamer
from solar.einsum.einsum_to_timeloop import TimeloopFormatter

# Benchmark converter
from solar.einsum.benchmark_converter import BenchmarkEinsumConverter

# LLM agent
from solar.einsum.llm_agent import (
    AgentConfig,
    NodeTypeConversionAgent,
    get_api_key_interactive,
)

# Node type registry
from solar.einsum.node_type_registry import (
    DefaultNodeExpansionStrategy,
    NodeExpansionStrategy,
    NodeTypeHandler,
    NodeTypeHandlerFactory,
    NodeTypeRegistry,
)

__all__ = [
    # Core analyzer
    "EinsumAnalyzer",
    # Main converters (new names)
    "PyTorchToEinsum",
    "EinsumRankRenamer",
    "EinsumToTimeloop",
    "EinsumToTaco",
    "EinsumGraphVisualizer",
    "GraphExpander",
    # Backward compatibility aliases
    "PyTorchEinsumConverter",
    "EinsumGraphRenamer",
    "TimeloopFormatter",
    # Convenience functions
    "convert_to_timeloop",
    "rename_einsum_ranks",
    "save_einsum_graph_pdf",
    "generate_taco_expression",
    "add_taco_expressions",
    # Benchmark converter
    "BenchmarkEinsumConverter",
    # LLM agent
    "AgentConfig",
    "NodeTypeConversionAgent",
    "get_api_key_interactive",
    # Node type registry
    "NodeTypeHandler",
    "NodeTypeHandlerFactory",
    "NodeTypeRegistry",
    "NodeExpansionStrategy",
    "DefaultNodeExpansionStrategy",
]
