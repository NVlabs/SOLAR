"""Einsum operation handlers.

This package contains modular handlers for converting different PyTorch
operations to einsum notation.
"""

from solar.einsum.ops.registry import (
    EinsumOpRegistry,
    register_einsum_op,
    get_global_registry,
)
from solar.einsum.ops.base import EinsumOpHandler, EinsumOp, EinsumOperand

__all__ = [
    "EinsumOpRegistry",
    "register_einsum_op",
    "get_global_registry",
    "EinsumOpHandler",
    "EinsumOp",
    "EinsumOperand",
]

