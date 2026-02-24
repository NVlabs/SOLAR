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
