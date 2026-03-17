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

"""Constants used throughout the Solar package.

This module defines constants following Google's Python style guide conventions.
"""

from typing import FrozenSet

# Default settings
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_OUTPUT_DIR = "outputs"

# Precision settings
BYTES_PER_ELEMENT = {
    "fp32": 4,
    "tf32": 4,
    "fp16": 2, 
    "bf16": 2,
    "int8": 1,
    "fp64": 8,
    "fp8": 1,
    "nvfp4": 0.5,
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
