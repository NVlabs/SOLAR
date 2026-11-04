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

"""Solar: PyTorch Model Analysis Toolkit.

This package provides tools for analyzing PyTorch model graphs,
generating einsum representations, and performing performance analysis.

Package structure:
- solar.einsum: Einsum conversion (ops, converters, analyzer)
- solar.perf: Performance modeling
- solar.analysis: Hardware-independent analysis (graph analyzer, model analyzer)
- solar.graph: PyTorch graph processing
"""

__version__ = "2.0.0"

# Einsum conversion and analysis
from solar.einsum import (
    EinsumAnalyzer,
    PyTorchToEinsum,
    PyTorchEinsumConverter,  # Backward compatibility
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

# Einsum lib (v2 loop-nest representation)
from solar.einsum_lib import Tensor, Dim

__all__ = [
    # Einsum
    "EinsumAnalyzer",
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
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
    # Einsum lib (v2)
    "Tensor",
    "Dim",
]
