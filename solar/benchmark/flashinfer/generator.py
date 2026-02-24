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

"""Generate kernelbench-style Python files from FlashInfer definitions."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from solar.benchmark.flashinfer.parser import FlashInferParser


# Map FlashInfer dtypes to PyTorch dtypes
DTYPE_MAP = {
    "float16": "torch.float16",
    "float32": "torch.float32",
    "bfloat16": "torch.bfloat16",
    "int8": "torch.int8",
    "int32": "torch.int32",
    "int64": "torch.int64",
    # FP8 types - fall back to float16 for compatibility (FP8 requires special hardware)
    "float8_e4m3fn": "torch.float16",
    "float8_e5m2": "torch.float16",
}


class FlashInferModelGenerator:
    """Generate kernelbench-style model files from FlashInfer definitions."""

    def __init__(self, parser: FlashInferParser, output_dir: Path):
        """Initialize generator.
        
        Args:
            parser: FlashInferParser instance
            output_dir: Base output directory for generated files
        """
        self.parser = parser
        self.output_dir = Path(output_dir)

    def _extract_forward_code(self, reference: str) -> str:
        """Extract the forward computation from reference code.
        
        Args:
            reference: Reference Python code string
            
        Returns:
            Code for the forward method body
        """
        # Parse the reference code to extract the run function body
        lines = reference.strip().split("\n")
        in_signature = False
        in_body = False
        run_lines = []
        indent = 0
        
        for line in lines:
            if "def run(" in line:
                in_signature = True
                # Get the indent level
                match = re.match(r"^(\s*)def run", line)
                if match:
                    indent = len(match.group(1))
                # Check if signature ends on same line
                if "):" in line:
                    in_signature = False
                    in_body = True
                continue
            
            # Skip lines that are part of the multi-line function signature
            if in_signature:
                if "):" in line:
                    in_signature = False
                    in_body = True
                continue
            
            if in_body:
                if line.strip() and not line.startswith(" " * (indent + 1)) and not line.startswith("\t"):
                    # End of function
                    if not line.strip().startswith("#"):
                        break
                run_lines.append(line)
        
        # Remove leading indent and return
        if run_lines:
            # Find minimum indent
            min_indent = float("inf")
            for line in run_lines:
                if line.strip():
                    spaces = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, spaces)
            if min_indent == float("inf"):
                min_indent = 0
            
            # Remove that indent
            result = []
            for line in run_lines:
                if line.strip():
                    result.append(line[min_indent:])
                else:
                    result.append("")
            return "\n".join(result)
        
        return "pass"

    def _get_input_params(self, inputs: Dict[str, Any]) -> str:
        """Generate forward method parameters from inputs spec."""
        params = ["self"]
        for name in inputs.keys():
            params.append(f"{name}: torch.Tensor")
        return ", ".join(params)

    def _get_return_statement(self, reference: str) -> str:
        """Extract return variable name from reference code."""
        # Look for return statement
        for line in reference.split("\n"):
            if "return " in line:
                match = re.search(r"return\s+(\w+)", line)
                if match:
                    return match.group(1)
        return "output"

    def _resolve_safetensors_path(self, raw_path: str) -> str:
        """Resolve a workload safetensors path to an absolute local path.

        Workloads may reference paths like "./blob/workloads/.../*.safetensors".
        We try:
        - relative to the provided trace_dir
        - HuggingFace cache (download if needed)
        """
        rel = raw_path[2:] if raw_path.startswith("./") else raw_path

        # 1) Relative to trace directory
        candidate = (self.parser.trace_dir / rel).resolve()
        if candidate.exists():
            return str(candidate)

        # 2) HuggingFace dataset cache
        # Prefer local-only first (user may already have blobs cached).
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(
                repo_id="flashinfer-ai/flashinfer-trace",
                repo_type="dataset",
                filename=rel,
                local_files_only=True,
            )
        except Exception:
            return hf_hub_download(
                repo_id="flashinfer-ai/flashinfer-trace",
                repo_type="dataset",
                filename=rel,
            )

    def generate_base_model(
        self, 
        definition: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Generate base model file with variable shapes as args.
        
        Args:
            definition: Parsed definition dict
            output_path: Path to write the generated Python file
        """
        name = definition["name"]
        description = definition.get("description", f"FlashInfer {name} operation")
        op_type = definition.get("op_type", "unknown")
        inputs = definition["inputs"]
        outputs = definition["outputs"]
        axes = definition["axes"]
        reference = definition.get("reference", "")
        
        # Determine variable axes
        var_axes = {k: v for k, v in axes.items() if v.get("type") == "var"}
        const_axes = {k: v["value"] for k, v in axes.items() if v.get("type") == "const"}
        
        # Check if reference code uses math module
        needs_math = 'math.' in reference
        
        # Build the model code
        code_lines = [
            '"""',
            f'{description}',
            f'',
            f'Op type: {op_type}',
            f'Generated from FlashInfer trace definition.',
            '"""',
            '',
            'import torch',
            'import torch.nn as nn',
        ]
        
        if needs_math:
            code_lines.append('import math')
        
        code_lines.append('')
        
        # Add constants
        for const_name, const_val in const_axes.items():
            code_lines.append(f'{const_name.upper()} = {const_val}')
        
        if const_axes:
            code_lines.append('')
        
        # Add variable shape defaults
        for var_name in var_axes.keys():
            code_lines.append(f'{var_name.upper()}_DEFAULT = 64  # Variable, override via args')
        
        if var_axes:
            code_lines.append('')
        
        code_lines.extend([
            '',
            'class Model(nn.Module):',
            f'    """',
            f'    {description}',
            f'    """',
            '    def __init__(self):',
            '        super(Model, self).__init__()',
            '',
        ])
        
        # Generate forward method
        forward_params = self._get_input_params(inputs)
        code_lines.append(f'    def forward({forward_params}) -> torch.Tensor:')
        code_lines.append(f'        """')
        code_lines.append(f'        Forward pass.')
        code_lines.append(f'        """')
        
        # Add the forward body from reference
        forward_body = self._extract_forward_code(reference)
        for line in forward_body.split("\n"):
            if line.strip():
                code_lines.append(f'        {line}')
            else:
                code_lines.append('')
        
        code_lines.append('')
        code_lines.append('')
        
        # Generate get_inputs function
        code_lines.append('def get_inputs():')
        code_lines.append('    """Generate random input tensors."""')
        
        input_tensors = []
        for input_name, input_spec in inputs.items():
            shape = input_spec.get("shape")
            dtype = DTYPE_MAP.get(input_spec.get("dtype", "float32"), "torch.float32")
            orig_dtype = input_spec.get("dtype", "float32")
            
            # Skip scalar inputs (shape is None) - they need special handling
            if shape is None:
                # Generate scalar placeholder
                if "int" in orig_dtype:
                    code_lines.append(f'    {input_name} = 0  # Scalar int')
                else:
                    code_lines.append(f'    {input_name} = 1.0  # Scalar float')
                input_tensors.append(input_name)
                continue
            
            # Build shape string
            shape_parts = []
            for dim in shape:
                if dim in const_axes:
                    shape_parts.append(str(const_axes[dim]))
                elif dim in var_axes:
                    shape_parts.append(f'{dim.upper()}_DEFAULT')
                else:
                    shape_parts.append(dim.upper())
            
            shape_str = ", ".join(shape_parts)
            
            # Use randint for integer types, randn for float types
            if "int" in orig_dtype:
                code_lines.append(f'    {input_name} = torch.randint(0, 100, ({shape_str},), dtype={dtype})')
            else:
                code_lines.append(f'    {input_name} = torch.randn({shape_str}, dtype={dtype})')
            input_tensors.append(input_name)
        
        code_lines.append(f'    return [{", ".join(input_tensors)}]')
        code_lines.append('')
        code_lines.append('')
        
        # Generate get_init_inputs function
        code_lines.append('def get_init_inputs():')
        code_lines.append('    """Return initialization inputs (none needed)."""')
        code_lines.append('    return []')
        code_lines.append('')
        
        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))

    def generate_workload_instance(
        self,
        definition: Dict[str, Any],
        workload: Dict[str, Any],
        output_path: Path,
        row_id: int
    ) -> None:
        """Generate model file for a specific workload instance.
        
        Args:
            definition: Parsed definition dict
            workload: Workload dict with concrete axis values
            output_path: Path to write the generated Python file
            row_id: Row index in the workload file
        """
        name = definition["name"]
        description = definition.get("description", f"FlashInfer {name} operation")
        op_type = definition.get("op_type", "unknown")
        inputs = definition["inputs"]
        outputs = definition["outputs"]
        axes = definition["axes"]
        reference = definition.get("reference", "")
        
        workload_info = workload.get("workload", {})
        workload_axes = workload_info.get("axes", {})
        workload_inputs = workload_info.get("inputs", {}) or {}
        workload_uuid = workload_info.get("uuid", "unknown")
        
        # Resolve all axes to concrete values
        resolved_axes = {}
        for axis_name, axis_spec in axes.items():
            if axis_spec.get("type") == "const":
                resolved_axes[axis_name] = axis_spec["value"]
            elif axis_name in workload_axes:
                resolved_axes[axis_name] = workload_axes[axis_name]
            else:
                # For workload instances, variable axes must be provided by the JSONL.
                # This keeps shapes faithful to trace/workload metadata and avoids silent fallbacks.
                raise ValueError(
                    f"Missing required axis '{axis_name}' for {op_type}/{name} (row_id={row_id}). "
                    f"Expected it in workload['workload']['axes']."
                )
        
        # Check if reference code uses math module
        needs_math = 'math.' in reference

        # Some workloads provide concrete inputs (e.g. kv_indptr/kv_indices) via safetensors blobs.
        needs_safetensors = any(
            isinstance(spec, dict) and spec.get("type") == "safetensors"
            for spec in workload_inputs.values()
        )
        
        # Build the model code
        code_lines = [
            '"""',
            f'{description}',
            f'',
            f'Op type: {op_type}',
            f'Workload UUID: {workload_uuid}',
            f'Row ID: {row_id}',
            f'Axes: {workload_axes}',
            f'Generated from FlashInfer trace workload.',
            '"""',
            '',
            'import torch',
            'import torch.nn as nn',
        ]
        
        if needs_math:
            code_lines.append('import math')

        if needs_safetensors:
            code_lines.append('from safetensors.torch import load_file as _st_load_file')
        
        code_lines.append('')
        
        # Add resolved constants
        for axis_name, axis_val in resolved_axes.items():
            code_lines.append(f'{axis_name.upper()} = {axis_val}')
        
        code_lines.extend([
            '',
            '',
            'class Model(nn.Module):',
            f'    """',
            f'    {description}',
            f'    """',
            '    def __init__(self):',
            '        super(Model, self).__init__()',
            '',
        ])
        
        # Generate forward method
        forward_params = self._get_input_params(inputs)
        code_lines.append(f'    def forward({forward_params}) -> torch.Tensor:')
        code_lines.append(f'        """')
        code_lines.append(f'        Forward pass.')
        code_lines.append(f'        """')
        
        # Add the forward body from reference
        forward_body = self._extract_forward_code(reference)
        for line in forward_body.split("\n"):
            if line.strip():
                code_lines.append(f'        {line}')
            else:
                code_lines.append('')
        
        code_lines.append('')
        code_lines.append('')
        
        # Generate get_inputs function with concrete shapes
        code_lines.append('def get_inputs():')
        code_lines.append('    """Generate random input tensors with concrete shapes."""')
        
        input_tensors = []
        safetensors_var_by_path: Dict[str, str] = {}
        for input_name, input_spec in inputs.items():
            shape = input_spec.get("shape")
            dtype = DTYPE_MAP.get(input_spec.get("dtype", "float32"), "torch.float32")
            orig_dtype = input_spec.get("dtype", "float32")
            workload_input_spec = workload_inputs.get(input_name, {"type": "random"}) or {"type": "random"}
            input_type = workload_input_spec.get("type", "random")
            
            # Skip scalar inputs (shape is None) - they need special handling
            if shape is None:
                # Prefer workload-provided scalar (e.g., sm_scale)
                if input_type == "scalar" and "value" in workload_input_spec:
                    code_lines.append(f'    {input_name} = {workload_input_spec["value"]}  # Scalar from trace')
                else:
                    # Fallback placeholder
                    if "int" in orig_dtype:
                        code_lines.append(f'    {input_name} = 0  # Scalar int')
                    else:
                        code_lines.append(f'    {input_name} = 1.0  # Scalar float')
                input_tensors.append(input_name)
                continue

            # Workload specifies a concrete tensor from a safetensors blob.
            if input_type == "safetensors":
                raw_path = workload_input_spec.get("path")
                tensor_key = workload_input_spec.get("tensor_key", input_name)
                if not raw_path:
                    raise ValueError(f"Missing safetensors path for input '{input_name}'")

                abs_path = self._resolve_safetensors_path(str(raw_path))
                st_var = safetensors_var_by_path.get(abs_path)
                if st_var is None:
                    st_var = f"_st_{len(safetensors_var_by_path)}"
                    safetensors_var_by_path[abs_path] = st_var
                    code_lines.append(f'    {st_var} = _st_load_file(r"{abs_path}")')
                code_lines.append(f'    {input_name} = {st_var}["{tensor_key}"]')
                # Best-effort dtype normalization to match definition.
                code_lines.append(f'    {input_name} = {input_name}.to({dtype})')
                input_tensors.append(input_name)
                continue
            
            # Build shape string with concrete values
            shape_parts = []
            for dim in shape:
                if dim in resolved_axes:
                    shape_parts.append(str(resolved_axes[dim]))
                else:
                    shape_parts.append(dim.upper())
            
            shape_str = ", ".join(shape_parts)
            
            # Use randint for integer types, randn for float types
            if "int" in orig_dtype:
                code_lines.append(f'    {input_name} = torch.randint(0, 100, ({shape_str},), dtype={dtype})')
            else:
                code_lines.append(f'    {input_name} = torch.randn({shape_str}, dtype={dtype})')
            input_tensors.append(input_name)
        
        code_lines.append(f'    return [{", ".join(input_tensors)}]')
        code_lines.append('')
        code_lines.append('')
        
        # Generate get_init_inputs function
        code_lines.append('def get_init_inputs():')
        code_lines.append('    """Return initialization inputs (none needed)."""')
        code_lines.append('    return []')
        code_lines.append('')
        
        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))
        
        # Write uuid.yaml with workload metadata
        uuid_path = output_path.parent / "uuid.yaml"
        uuid_data = {
            "uuid": workload_uuid,
            "definition": name,
            "op_type": op_type,
            "row_id": row_id,
            "axes": workload_axes,
            "resolved_axes": resolved_axes,
        }
        with open(uuid_path, "w") as f:
            yaml.dump(uuid_data, f, default_flow_style=False, sort_keys=False)

    def generate_all_for_definition(
        self, 
        op_type: str, 
        name: str,
        max_workloads: Optional[int] = None
    ) -> List[Path]:
        """Generate base model and all workload instances for a definition.
        
        Args:
            op_type: Operation type (e.g., "gemm")
            name: Definition name (e.g., "gemm_n128_k2048")
            max_workloads: Maximum number of workloads to generate (None = all)
            
        Returns:
            List of paths to generated files
        """
        definition, workloads = self.parser.get_definition_and_workloads(op_type, name)
        
        generated_files = []
        
        # Generate base model
        base_dir = self.output_dir / op_type / name
        base_path = base_dir / f"{name}.py"
        self.generate_base_model(definition, base_path)
        generated_files.append(base_path)
        
        # Generate workload instances
        if max_workloads is not None:
            workloads = workloads[:max_workloads]
        
        for row_id, workload in enumerate(workloads):
            instance_path = base_dir / str(row_id) / f"{name}.py"
            self.generate_workload_instance(definition, workload, instance_path, row_id)
            generated_files.append(instance_path)
        
        return generated_files

    def generate_all(self, max_workloads_per_def: Optional[int] = None) -> List[Path]:
        """Generate models for all definitions and workloads.
        
        Args:
            max_workloads_per_def: Max workloads per definition (None = all)
            
        Returns:
            List of all generated file paths
        """
        all_files = []
        
        for op_type in self.parser.list_op_types():
            for name in self.parser.list_definitions(op_type):
                files = self.generate_all_for_definition(
                    op_type, name, max_workloads_per_def
                )
                all_files.extend(files)
        
        return all_files
