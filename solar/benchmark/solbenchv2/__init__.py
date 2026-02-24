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

"""SolBench v2 support module for Solar analysis.

This module provides utilities to parse and process kernels from
sol-bench/data/benchmark for Solar analysis.
"""

from .converter import convert_solbenchv2_file
from .parser import parse_solbenchv2_file

__all__ = ["convert_solbenchv2_file", "parse_solbenchv2_file"]
