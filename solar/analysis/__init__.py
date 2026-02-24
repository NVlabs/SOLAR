"""Analysis module for Solar.

This module provides hardware-independent analysis tools for einsum graphs.

Key components:
- `EinsumGraphAnalyzer` - Graph-level analysis (MACs, memory, etc.)
- `ModelAnalyzer` - Model analysis with LLM agent support

For einsum conversion, see `solar.einsum`.
For performance modeling, see `solar.perf`.
"""

# Local analysis modules
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.analysis.model_analyzer import ModelAnalyzer

__all__ = [
    "EinsumGraphAnalyzer",
    "ModelAnalyzer",
]
