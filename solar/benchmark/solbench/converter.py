#!/usr/bin/env python3
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

"""
Convert SolBench model files to KernelBench-compatible format.

This module parses SolBench Python files and generates new Python files
in the KernelBench format with:
- class Model(nn.Module)
- def get_inputs()
- def get_init_inputs()
"""

from __future__ import annotations

import re
import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# Default configuration for SolBench models
# Use 32 as default for unknown dimensions - model-specific values from docstring will override
DEFAULT_CONFIG = {
    "hidden_size": 32,  # Will be overridden by model config if available
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 32,
    "intermediate_size": 32,
    "rms_norm_eps": 1e-05,
    "attention_bias": False,
    "mlp_bias": False,
    "dtype": "bfloat16",
    "batch": 32,
    "seq_len": 32,
    # Additional common sizes
    "default_dim": 32,
    "vocab_size": 32,
    "max_position_embeddings": 32,
}


@dataclass
class ModelInfo:
    """Parsed information from a SolBench model file."""
    filename: str
    index: str
    name: str
    class_name: str
    docstring: str
    op_type: str
    priority: str
    description: str
    config: Dict[str, Any]
    optimization_notes: str
    fusion_opportunities: List[str]
    forward_args: List[Tuple[str, str]]  # (arg_name, type_hint)
    forward_return_type: str
    source_code: str
    init_params: List[str]  # Parameters in __init__


class SolBenchConverter:
    """Convert SolBench models to KernelBench format."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, debug: bool = False):
        """
        Initialize converter.
        
        Args:
            config: Configuration overrides for default values
            debug: Enable debug output
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.debug = debug
    
    def parse_model_file(self, filepath: Path) -> ModelInfo:
        """Parse a SolBench model file and extract information."""
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        # Parse filename
        filename = filepath.name
        match = re.match(r'^(\d+)_(.+)\.py$', filename)
        if match:
            index = match.group(1)
            name = match.group(2)
        else:
            index = "0000"
            name = filename.replace('.py', '')
        
        # Parse docstring
        docstring_match = re.search(r'^"""(.*?)"""', source_code, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else ""
        
        # Extract metadata from docstring
        metadata = self._parse_docstring(docstring)
        
        # Find the main class
        class_match = re.search(r'class\s+(\w+)\s*\(nn\.Module\)', source_code)
        class_name = class_match.group(1) if class_match else "Model"
        
        # Parse forward method arguments
        forward_args, return_type = self._parse_forward_signature(source_code)
        
        # Parse __init__ to find parameters
        init_params = self._parse_init_params(source_code)
        
        return ModelInfo(
            filename=filename,
            index=index,
            name=name,
            class_name=class_name,
            docstring=docstring,
            op_type=metadata.get("op_type", "fused_op"),
            priority=metadata.get("priority", "medium"),
            description=metadata.get("description", ""),
            config=metadata.get("config", {}),
            optimization_notes=metadata.get("optimization_notes", ""),
            fusion_opportunities=metadata.get("fusion_opportunities", []),
            forward_args=forward_args,
            forward_return_type=return_type,
            source_code=source_code,
            init_params=init_params,
        )
    
    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse metadata from module docstring."""
        result = {}
        
        # Extract Operation Type
        match = re.search(r'Operation Type:\s*(.+)', docstring)
        if match:
            result["op_type"] = match.group(1).strip()
        
        # Extract Priority
        match = re.search(r'Priority:\s*(.+)', docstring)
        if match:
            result["priority"] = match.group(1).strip()
        
        # Extract Description
        match = re.search(r'Description:\s*(.+?)(?=\n\n|Configuration|Optimization|Kernel|$)', 
                         docstring, re.DOTALL)
        if match:
            result["description"] = match.group(1).strip()
        
        # Extract Configuration Constants
        match = re.search(r'Configuration Constants:\s*(\{[^}]+\})', docstring, re.DOTALL)
        if match:
            try:
                config_str = match.group(1).strip()
                # Handle JSON-style booleans
                config_str = config_str.replace('true', 'True').replace('false', 'False')
                result["config"] = ast.literal_eval(config_str)
            except:
                result["config"] = {}
        
        # Extract Optimization Notes
        match = re.search(r'Optimization Notes:\s*(.+?)(?=\n\n|Kernel|$)', docstring, re.DOTALL)
        if match:
            result["optimization_notes"] = match.group(1).strip()
        
        # Extract Kernel Fusion Opportunities
        match = re.search(r'Kernel Fusion Opportunities:\s*(.+?)(?="""|\n\n[A-Z]|$)', 
                         docstring, re.DOTALL)
        if match:
            opportunities_text = match.group(1).strip()
            opportunities = [line.strip().lstrip('- ') 
                           for line in opportunities_text.split('\n') 
                           if line.strip().startswith('-')]
            result["fusion_opportunities"] = opportunities
        
        return result
    
    def _parse_forward_signature(self, source_code: str) -> Tuple[List[Tuple[str, str]], str]:
        """Parse forward method signature to get arguments and return type."""
        # Match forward method with various formatting
        pattern = r'def\s+forward\s*\(\s*self\s*,?\s*([^)]*)\)\s*(?:->\s*([^:]+))?:'
        match = re.search(pattern, source_code, re.DOTALL)
        
        if not match:
            return [], "torch.Tensor"
        
        args_str = match.group(1).strip()
        return_type = match.group(2).strip() if match.group(2) else "torch.Tensor"
        
        # Parse arguments
        args = []
        if args_str:
            # Split by comma, handling nested brackets
            arg_parts = self._split_args(args_str)
            for arg in arg_parts:
                arg = arg.strip()
                if not arg or arg == 'self':
                    continue
                
                # Parse "name: Type" or "name: Type = default"
                if ':' in arg:
                    parts = arg.split(':', 1)
                    arg_name = parts[0].strip()
                    type_hint = parts[1].split('=')[0].strip()
                else:
                    arg_name = arg.split('=')[0].strip()
                    type_hint = "torch.Tensor"
                
                if arg_name and arg_name != 'self':
                    args.append((arg_name, type_hint))
        
        return args, return_type
    
    def _split_args(self, args_str: str) -> List[str]:
        """Split argument string by commas, respecting brackets."""
        args = []
        current = ""
        depth = 0
        
        for char in args_str:
            if char in '([{':
                depth += 1
                current += char
            elif char in ')]}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args
    
    def _parse_init_params(self, source_code: str) -> List[str]:
        """Parse __init__ method to find self.xxx assignments."""
        params = []
        
        # Find __init__ method
        init_match = re.search(r'def\s+__init__\s*\([^)]*\):[^\n]*\n((?:\s+[^\n]+\n)*)', source_code)
        if init_match:
            init_body = init_match.group(1)
            # Find self.xxx = assignments
            for match in re.finditer(r'self\.(\w+)\s*=', init_body):
                param = match.group(1)
                if param not in params:
                    params.append(param)
        
        return params
    
    def _extract_imports(self, source_code: str) -> List[str]:
        """Extract all import statements from source code.
        
        Args:
            source_code: Python source code string
            
        Returns:
            List of all import statements from the original file
        """
        imports = []
        lines = source_code.strip().split("\n")
        
        for line in lines:
            stripped = line.strip()
            # Copy all import statements
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        
        return imports
    
    def _infer_tensor_shape(self, arg_name: str, type_hint: str, docstring: str) -> List[str]:
        """Infer tensor shape from argument name, type hint, and docstring."""
        # Try to find shape in docstring comments
        # Look for patterns like "arg_name: [batch, seq_len, hidden_size]"
        # Escape special regex characters in arg_name
        escaped_arg_name = re.escape(arg_name)
        shape_pattern = rf'{escaped_arg_name}[^[]*\[([^\]]+)\]'
        match = re.search(shape_pattern, docstring, re.IGNORECASE)
        
        if match:
            shape_str = match.group(1)
            # Parse shape dimensions
            dims = [d.strip() for d in shape_str.split(',')]
            return dims
        
        # Infer from common naming patterns
        name_lower = arg_name.lower()
        
        if 'hidden_states' in name_lower or 'input_embeds' in name_lower:
            return ['batch', 'seq_len', 'hidden_size']
        elif 'attention_mask' in name_lower or 'mask' in name_lower:
            return ['batch', 'seq_len']
        elif 'position_ids' in name_lower:
            return ['batch', 'seq_len']
        elif 'query' in name_lower or 'key' in name_lower or 'value' in name_lower:
            return ['batch', 'num_attention_heads', 'seq_len', 'head_dim']
        elif 'weight' in name_lower:
            return ['hidden_size', 'hidden_size']
        elif 'gate' in name_lower:
            return ['batch', 'intermediate_size', 'seq_len']
        elif 'input' in name_lower or 'x' == name_lower:
            return ['batch', 'seq_len', 'hidden_size']
        elif 'output' in name_lower:
            return ['batch', 'seq_len', 'hidden_size']
        else:
            # Default shape
            return ['batch', 'seq_len', 'hidden_size']
    
    def _resolve_dim(self, dim: str, config: Optional[Dict[str, Any]] = None) -> int:
        """Resolve a dimension name to a concrete value.
        
        Args:
            dim: Dimension name or expression to resolve
            config: Config to use (defaults to self.config if not provided)
        """
        cfg = config if config is not None else self.config
        dim_lower = dim.lower().strip()
        
        # Direct config lookup
        if dim_lower in cfg:
            return cfg[dim_lower]
        
        # Common aliases - use provided config values
        aliases = {
            'batch': cfg.get('batch', self.config['batch']),
            'batch_size': cfg.get('batch', self.config['batch']),
            'b': cfg.get('batch', self.config['batch']),
            'seq_len': cfg.get('seq_len', self.config['seq_len']),
            'sequence_length': cfg.get('seq_len', self.config['seq_len']),
            'seq': cfg.get('seq_len', self.config['seq_len']),
            's': cfg.get('seq_len', self.config['seq_len']),
            'hidden_size': cfg.get('hidden_size', self.config['hidden_size']),
            'hidden': cfg.get('hidden_size', self.config['hidden_size']),
            'd': cfg.get('hidden_size', self.config['hidden_size']),
            'h': cfg.get('hidden_size', self.config['hidden_size']),
            'num_attention_heads': cfg.get('num_attention_heads', self.config['num_attention_heads']),
            'num_heads': cfg.get('num_attention_heads', self.config['num_attention_heads']),
            'n_heads': cfg.get('num_attention_heads', self.config['num_attention_heads']),
            'num_key_value_heads': cfg.get('num_key_value_heads', self.config['num_key_value_heads']),
            'head_dim': cfg.get('head_dim', self.config['head_dim']),
            'intermediate_size': cfg.get('intermediate_size', self.config['intermediate_size']),
            'vocab_size': cfg.get('vocab_size', self.config['vocab_size']),
        }
        
        if dim_lower in aliases:
            return aliases[dim_lower]
        
        # Try to parse as integer
        try:
            return int(dim)
        except ValueError:
            pass
        
        # Try to evaluate expressions like "hidden_size * 2"
        try:
            # Replace known variables using merged config
            expr = dim_lower
            for key, val in cfg.items():
                expr = re.sub(rf'\b{key}\b', str(val), expr)
            return int(eval(expr))
        except:
            pass
        
        # Default
        return cfg.get('default_dim', self.config['default_dim'])
    
    def convert_model(self, model_info: ModelInfo) -> str:
        """Convert a parsed model to KernelBench format."""
        # Merge configs (model-specific values from docstring override defaults)
        # This preserves known constants from the original model
        merged_config = {**self.config, **model_info.config}
        
        # Generate get_inputs function
        input_tensors = []
        input_shapes = []
        
        for arg_name, type_hint in model_info.forward_args:
            # Infer shape
            shape_dims = self._infer_tensor_shape(arg_name, type_hint, model_info.docstring)
            
            # Resolve dimensions to concrete values using merged config
            # This ensures model-specific constants (e.g., hidden_size: 4096) are preserved
            shape = [self._resolve_dim(dim, merged_config) for dim in shape_dims]
            
            input_tensors.append(arg_name)
            input_shapes.append(shape)
        
        # Determine dtype
        dtype = merged_config.get('dtype', 'bfloat16')
        dtype_torch = f"torch.{dtype}"
        
        # Generate the converted code
        code = self._generate_code(model_info, input_tensors, input_shapes, dtype_torch, merged_config)
        
        return code
    
    def _generate_code(
        self, 
        model_info: ModelInfo, 
        input_tensors: List[str],
        input_shapes: List[List[int]],
        dtype_torch: str,
        config: Dict[str, Any]
    ) -> str:
        """Generate the KernelBench-compatible Python code."""
        
        # Extract all imports from original source code
        original_imports = self._extract_imports(model_info.source_code)
        
        # Build imports - start with __future__ annotations, then all original imports
        imports = ["from __future__ import annotations", ""]
        imports.extend(original_imports)
        
        # Build docstring
        docstring = f'''"""
SolBench Model: {model_info.name}
Operation Type: {model_info.op_type}
Priority: {model_info.priority}

Description:
{model_info.description}

Original class: {model_info.class_name}
    
Configuration:
{self._format_config(config)}
"""'''
        
        # Extract the original class definition
        original_class = self._extract_class_definition(model_info.source_code, model_info.class_name)
        
        # Create Model wrapper class
        model_class = self._create_model_wrapper(model_info, original_class, config)
        
        # Generate get_inputs function
        get_inputs = self._generate_get_inputs(input_tensors, input_shapes, dtype_torch)
        
        # Generate get_init_inputs function
        get_init_inputs = "def get_init_inputs():\n    return []  # No special initialization inputs needed"
        
        # Combine all parts
        code_parts = [
            '\n'.join(imports),
            '',
            docstring,
            '',
            '# Original implementation',
            original_class,
            '',
            '# KernelBench-compatible wrapper',
            model_class,
            '',
            '# Input generation',
            get_inputs,
            '',
            get_init_inputs,
        ]
        
        return '\n'.join(code_parts)
    
    def _format_config(self, config: Dict[str, Any]) -> str:
        """Format config dict as string."""
        lines = []
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        return '\n'.join(lines)
    
    def _extract_single_class(self, source_code: str, class_name: str) -> str:
        """Extract a single class definition from source code."""
        # Find class start
        class_pattern = rf'class\s+{re.escape(class_name)}\s*\([^)]*\):'
        match = re.search(class_pattern, source_code)
        
        if not match:
            return ""
        
        start = match.start()
        
        # Find class end (next class definition or end of file)
        remaining = source_code[match.end():]
        
        # Find the end by tracking indentation
        lines = remaining.split('\n')
        class_lines = [source_code[start:match.end()]]
        
        for line in lines:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Non-indented non-empty line = end of class
                if not line.startswith('#'):
                    break
            class_lines.append(line)
        
        return '\n'.join(class_lines).rstrip()
    
    def _find_all_classes(self, source_code: str) -> List[Tuple[str, str]]:
        """Find all class definitions in source code.
        
        Returns:
            List of (class_name, parent_class) tuples
        """
        classes = []
        pattern = r'class\s+(\w+)\s*\(([^)]+)\):'
        for match in re.finditer(pattern, source_code):
            class_name = match.group(1)
            parent = match.group(2).strip()
            classes.append((class_name, parent))
        return classes
    
    def _find_dependencies(self, source_code: str, main_class: str) -> List[str]:
        """Find all classes that the main class depends on.
        
        This includes:
        - Classes referenced in the main class body
        - Function subclasses (torch.autograd.Function)
        - Any other supporting classes
        """
        dependencies = []
        all_classes = self._find_all_classes(source_code)
        
        # Get main class code to check for references
        main_code = self._extract_single_class(source_code, main_class)
        
        for class_name, parent in all_classes:
            if class_name == main_class:
                continue
            
            # Include Function subclasses (used for custom autograd)
            if 'Function' in parent:
                dependencies.append(class_name)
                continue
            
            # Check if this class is referenced in the main class
            if re.search(rf'\b{re.escape(class_name)}\b', main_code):
                dependencies.append(class_name)
        
        return dependencies
    
    def _extract_class_definition(self, source_code: str, class_name: str) -> str:
        """Extract the original class definition and all its dependencies from source code."""
        # Find all dependencies
        dependencies = self._find_dependencies(source_code, class_name)
        
        # Extract all dependent classes first
        class_parts = []
        for dep_class in dependencies:
            dep_code = self._extract_single_class(source_code, dep_class)
            if dep_code:
                class_parts.append(dep_code)
        
        # Extract main class
        main_code = self._extract_single_class(source_code, class_name)
        if main_code:
            class_parts.append(main_code)
        else:
            return f"# Could not extract original class {class_name}"
        
        return '\n\n'.join(class_parts)
    
    def _create_model_wrapper(self, model_info: ModelInfo, original_class: str, config: Dict[str, Any]) -> str:
        """Create a Model wrapper class."""
        class_name = model_info.class_name
        
        # Build forward args string
        forward_args = ', '.join([f"{name}: torch.Tensor" for name, _ in model_info.forward_args])
        forward_call_args = ', '.join([name for name, _ in model_info.forward_args])
        
        wrapper = f'''class Model(nn.Module):
    """
    KernelBench-compatible wrapper for {class_name}.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.model = {class_name}()
    
    def forward(self, {forward_args}) -> torch.Tensor:
        """
        Forward pass through the wrapped model.
        """
        result = self.model({forward_call_args})
        # Handle tuple returns by returning first element
        if isinstance(result, tuple):
            return result[0]
        return result'''
        
        return wrapper
    
    def _generate_get_inputs(
        self, 
        input_tensors: List[str], 
        input_shapes: List[List[int]], 
        dtype_torch: str
    ) -> str:
        """Generate the get_inputs function."""
        lines = ["def get_inputs():"]
        
        tensor_vars = []
        for i, (name, shape) in enumerate(zip(input_tensors, input_shapes)):
            shape_str = ', '.join(str(d) for d in shape)
            var_name = name.replace('-', '_')
            lines.append(f"    {var_name} = torch.randn({shape_str}, dtype={dtype_torch})")
            tensor_vars.append(var_name)
        
        if tensor_vars:
            return_str = ', '.join(tensor_vars)
            lines.append(f"    return [{return_str}]")
        else:
            lines.append("    return []")
        
        return '\n'.join(lines)
    
    def convert_file(self, input_path: Path, output_path: Path) -> bool:
        """Convert a single SolBench file to KernelBench format."""
        try:
            model_info = self.parse_model_file(input_path)
            converted_code = self.convert_model(model_info)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(converted_code)
            
            if self.debug:
                print(f"Converted: {input_path.name} -> {output_path}")
            
            return True
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def convert_directory(
        self, 
        input_dir: Path, 
        output_dir: Path,
        max_files: Optional[int] = None
    ) -> Tuple[int, int]:
        """Convert all SolBench files in a directory.
        
        Returns:
            Tuple of (success_count, failure_count)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        py_files = sorted(input_dir.glob("*.py"))
        
        if max_files:
            py_files = py_files[:max_files]
        
        success = 0
        failure = 0
        
        for py_file in py_files:
            # Output filename: same as input
            output_file = output_dir / py_file.name
            
            if self.convert_file(py_file, output_file):
                success += 1
            else:
                failure += 1
        
        return success, failure


def main():
    """CLI for converting SolBench models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SolBench models to KernelBench format")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory with SolBench files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for converted files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to convert")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    # Config overrides
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size (default 32, use value from model config if available)")
    
    args = parser.parse_args()
    
    config = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
    }
    
    converter = SolBenchConverter(config=config, debug=args.debug)
    success, failure = converter.convert_directory(
        args.input_dir, 
        args.output_dir,
        max_files=args.max_files
    )
    
    print(f"\nConversion complete: {success} succeeded, {failure} failed")


if __name__ == "__main__":
    main()
