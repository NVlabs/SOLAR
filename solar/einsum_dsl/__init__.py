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

"""einsum_dsl: translate toDSL_model output into einsum_graph.yaml format."""

from .model import DSLTensor, DSLInput, DSLKernel, DSLGraph
from .loader import load_dsl_yaml
from .ast_parser import extract_loop_exprs
from .assembler import build_dsl_graph
from .exporter import dsl_graph_to_dict, write_einsum_yaml
from .cli import translate
from .cli_einsum import translate_with_einsum
from .einsum_converter import build_kernel_einsum, build_start_einsum, annotate_einsum_equations

__all__ = [
    "DSLTensor", "DSLInput", "DSLKernel", "DSLGraph",
    "load_dsl_yaml",
    "extract_loop_exprs",
    "build_dsl_graph",
    "dsl_graph_to_dict", "write_einsum_yaml",
    "translate",
    "translate_with_einsum",
    "build_kernel_einsum", "build_start_einsum", "annotate_einsum_equations",
]
