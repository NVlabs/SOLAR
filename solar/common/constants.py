"""Constants used throughout the Solar package.

This module defines constants following Google's Python style guide conventions.
"""

from typing import Dict, FrozenSet, List

# Default settings
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_OUTPUT_DIR = "outputs"

# Precision settings
BYTES_PER_ELEMENT = {
    "fp32": 4,
    "fp16": 2, 
    "bf16": 2,
    "int8": 1,
    "fp64": 8,
}

# Supported operations for einsum analysis
SUPPORTED_OPERATIONS: FrozenSet[str] = frozenset({
    # Matrix operations
    "matmul", "bmm", "linear", "addmm",
    
    # Convolution operations
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    
    # Attention operations
    "scaled_dot_product_attention", "flex_attention",
    
    # Normalization operations
    "batch_norm", "layer_norm", "group_norm", "instance_norm",
    
    # Reduction operations
    "sum", "mean", "max", "min", "prod",
    "torch.sum", "torch.mean", "torch.max", "torch.min", "torch.prod",
    
    # Elementwise operations
    "add", "mul", "div", "sub", "pow",
    "relu", "gelu", "sigmoid", "tanh", "softmax",
    
    # Pooling operations
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "max_pool1d", "max_pool2d", "max_pool3d",
    
    # Other operations
    "transpose", "reshape", "flatten", "view",
})

# Architecture configurations directory
ARCHITECTURE_CONFIGS = {
    "H100_PCIe": {
        "freq_GHz": 1.41,
        "MAC_per_cycle_fp32_tc": 378000,
        "MAC_per_cycle_fp16_tc": 756000,
        "MAC_per_cycle_int8": 1513000,
        "DRAM_byte_per_cycle": 2000,
        "SRAM_capacity": 52428800,
    },
    "A6000": {
        "freq_GHz": 1.8,
        "MAC_per_cycle_fp32_tc": 43008,
        "MAC_per_cycle_bf16_tc": 86016,
        "MAC_per_cycle_int8": 172032,
        "DRAM_byte_per_cycle": 768,
        "SRAM_capacity": 48000000,
    },
    "H100_fp32": {
        "freq_GHz": 1.0,
        "MAC_per_cycle_fp32_tc": 189000,
        "MAC_per_cycle_int8": 1513000,
        "DRAM_byte_per_cycle": 2000,
        "SRAM_capacity": 52428800,
    },
}

# Node type mappings for graph processing
NODE_TYPE_MAPPINGS = {
    "MatmulNode": "matmul",
    "ConvNode": "conv2d",
    "LinearNode": "linear",
    "AddNode": "add",
    "MulNode": "mul",
    "ReluNode": "relu",
    "BatchNormNode": "batch_norm",
    "SoftmaxNode": "softmax",
}

# Attribute names to check for module extraction
MODULE_ATTR_NAMES = [
    "module",
    "pytorch_module",
    "op",
    "operation",
    "target",
    "_module",
    "wrapped_module",
]

# Geometric attributes for convolution operations
GEOMETRIC_ATTRS = frozenset({
    "kernel_size",
    "stride", 
    "padding",
    "dilation",
    "output_padding",
    "normalized_shape",
    "output_size",
})

# Boolean attributes for modules
BOOLEAN_ATTRS = frozenset({
    "inplace",
    "affine",
    "elementwise_affine",
    "track_running_stats",
    "ceil_mode",
    "count_include_pad",
    "return_indices",
    "sparse",
})

# Environment variables for safe execution
SAFE_ENV_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_VERBOSE": "0",
    "PYTORCH_DISABLE_CUDA": "1",
    "USE_OPENMP": "0",
}

# File patterns for different graph types
GRAPH_FILE_PATTERNS = {
    "einsum_graph": "einsum_graph.yaml",
    "torchview_graph": "pytorch_graph.yaml",
}

# Analysis output file names
ANALYSIS_OUTPUT_FILES = {
    "analysis": "analysis.yaml",
    "summary": "summary.txt",
    "performance": "perf_{arch}.yaml",
    "graph": "model_graph.yaml",
}

# Kernel directory patterns
KERNEL_DIR_PATTERNS = {
    "kernelbench": r"^\d+$",  # Simple numeric: 1, 2, 3
}
