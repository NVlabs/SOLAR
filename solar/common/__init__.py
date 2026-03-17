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

"""Common utilities and types for the Solar package."""

from solar.common.types import (
    NodeInfo,
    GraphInfo,
    AnalysisResult,
    EinsumOperation,
    TensorShape,
)
from solar.common.constants import (
    DEFAULT_PRECISION,
    SUPPORTED_OPERATIONS,
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
    "TensorShape",
    # Constants
    "DEFAULT_PRECISION",
    "SUPPORTED_OPERATIONS",
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
