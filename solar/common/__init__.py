"""Common utilities and types for the Solar package."""

from solar.common.types import (
    NodeInfo,
    GraphInfo,
    AnalysisResult,
    EinsumOperation,
    ShapeDict,
    TensorShape,
)
from solar.common.constants import (
    DEFAULT_PRECISION,
    SUPPORTED_OPERATIONS,
    ARCHITECTURE_CONFIGS,
)
from solar.common.utils import (
    format_number,
    setup_safe_environment,
    load_module_from_file,
)
from solar.common.einsum_graph_check import (
    EinsumGraphChecker,
    ValidationError,
    ValidationResult,
    check_einsum_graph,
    check_einsum_graph_file,
)

__all__ = [
    # Types
    "NodeInfo",
    "GraphInfo", 
    "AnalysisResult",
    "EinsumOperation",
    "ShapeDict",
    "TensorShape",
    # Constants
    "DEFAULT_PRECISION",
    "SUPPORTED_OPERATIONS",
    "ARCHITECTURE_CONFIGS",
    # Utils
    "format_number",
    "setup_safe_environment",
    "load_module_from_file",
    # Einsum graph checker
    "EinsumGraphChecker",
    "ValidationError",
    "ValidationResult",
    "check_einsum_graph",
    "check_einsum_graph_file",
]
