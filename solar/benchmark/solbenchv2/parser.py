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

"""Parser for SolBench v2 kernel files.

Parses Python files from sol-bench/data/benchmark to extract:
- ReferenceModel class or reference_backward function
- get_inputs() function
- get_init_inputs() function
- Global constants for shapes
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SolBenchV2Kernel:
    """Parsed information from a SolBench v2 kernel file."""
    
    filepath: str
    filename: str
    level: str  # L1, L2, or Quant
    
    # Detected methods/classes
    has_reference_model: bool = False
    has_reference_backward: bool = False
    has_get_inputs: bool = False
    has_get_init_inputs: bool = False
    has_launch_reference_implementation: bool = False
    
    # Extracted code
    reference_model_code: str = ""
    reference_backward_code: str = ""
    get_inputs_code: str = ""
    get_init_inputs_code: str = ""
    
    # Global constants
    global_constants: Dict[str, Any] = field(default_factory=dict)
    
    # Imports
    imports: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if kernel has all required components for Solar analysis."""
        has_reference = self.has_reference_model or self.has_reference_backward
        return (
            self.has_get_inputs and 
            has_reference and 
            self.has_launch_reference_implementation
        )


def extract_global_constants(tree: ast.AST) -> Dict[str, Any]:
    """Extract global constant assignments from AST."""
    constants = {}
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Check if it's likely a constant (UPPER_CASE)
                    if name.isupper() or name.startswith("_"):
                        try:
                            # Try to evaluate simple constants
                            if isinstance(node.value, ast.Constant):
                                constants[name] = node.value.value
                            elif isinstance(node.value, ast.Num):  # Python 3.7 compat
                                constants[name] = node.value.n
                            elif isinstance(node.value, ast.Str):  # Python 3.7 compat
                                constants[name] = node.value.s
                        except Exception:
                            pass
    
    return constants


def extract_imports(tree: ast.AST, source_code: str) -> List[str]:
    """Extract import statements from source code."""
    imports = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Get the actual source line
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            lines = source_code.split('\n')[start_line:end_line]
            imports.append('\n'.join(lines))
    
    return imports


def extract_node_source(node: ast.AST, source_code: str) -> str:
    """Extract source code for an AST node."""
    lines = source_code.split('\n')
    start_line = node.lineno - 1
    end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
    return '\n'.join(lines[start_line:end_line])


def parse_solbenchv2_file(filepath: str) -> SolBenchV2Kernel:
    """
    Parse a SolBench v2 kernel file and extract relevant information.
    
    Args:
        filepath: Path to the Python file
        
    Returns:
        SolBenchV2Kernel with parsed information
    """
    filepath = Path(filepath)
    
    # Determine level from parent directory
    level = filepath.parent.name
    if level not in ["L1", "L2", "Quant"]:
        level = "unknown"
    
    kernel = SolBenchV2Kernel(
        filepath=str(filepath),
        filename=filepath.name,
        level=level,
    )
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        
        # Extract imports
        kernel.imports = extract_imports(tree, source_code)
        
        # Extract global constants
        kernel.global_constants = extract_global_constants(tree)
        
        # Walk through top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "ReferenceModel":
                    kernel.has_reference_model = True
                    kernel.reference_model_code = extract_node_source(node, source_code)
            
            elif isinstance(node, ast.FunctionDef):
                if node.name == "reference_backward":
                    kernel.has_reference_backward = True
                    kernel.reference_backward_code = extract_node_source(node, source_code)
                elif node.name == "get_inputs":
                    kernel.has_get_inputs = True
                    kernel.get_inputs_code = extract_node_source(node, source_code)
                elif node.name == "get_init_inputs":
                    kernel.has_get_init_inputs = True
                    kernel.get_init_inputs_code = extract_node_source(node, source_code)
                elif node.name == "launch_reference_implementation":
                    kernel.has_launch_reference_implementation = True
                    
    except SyntaxError as e:
        print(f"Warning: Syntax error in {filepath}: {e}")
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
    
    return kernel
