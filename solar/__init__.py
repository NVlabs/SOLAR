"""Solar: PyTorch Model Analysis Toolkit.

This package provides tools for analyzing PyTorch model graphs,
generating einsum representations, and performing performance analysis.

Package structure:
- solar.einsum: Einsum conversion (ops, converters, analyzer)
- solar.perf: Performance modeling
- solar.analysis: Hardware-independent analysis (graph analyzer, model analyzer)
- solar.graph: PyTorch graph processing
"""

__version__ = "1.2.0"

# Einsum conversion and analysis
from solar.einsum import (
    EinsumAnalyzer,
    PyTorchToEinsum,
    PyTorchEinsumConverter,  # Backward compatibility
    BenchmarkEinsumConverter,
)

# Hardware-independent analysis
from solar.analysis import (
    EinsumGraphAnalyzer,
    ModelAnalyzer,
)

# Performance modeling
from solar.perf import EinsumGraphPerfModel

# Graph processing
from solar.graph import PyTorchProcessor, TorchviewProcessor

# Core types
from solar.einsum.ops import EinsumOp, EinsumOperand

__all__ = [
    # Einsum
    "EinsumAnalyzer",
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
    "BenchmarkEinsumConverter",
    # Analysis
    "EinsumGraphAnalyzer",
    "ModelAnalyzer",
    # Performance
    "EinsumGraphPerfModel",
    # Graph processing
    "PyTorchProcessor",
    "TorchviewProcessor",
    # Types
    "EinsumOp",
    "EinsumOperand",
]
