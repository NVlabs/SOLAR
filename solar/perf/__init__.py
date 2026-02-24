"""Performance modeling module for Solar.

This module provides tools for predicting performance of einsum graphs
on different hardware architectures.

Key components:
- `EinsumGraphPerfModel` - Roofline-based performance predictions
"""

from solar.perf.perf_model import EinsumGraphPerfModel

__all__ = [
    "EinsumGraphPerfModel",
]

