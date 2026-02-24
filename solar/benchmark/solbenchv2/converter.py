"""Converter for SolBench v2 kernels to Solar-compatible format.

Converts kernel files from sol-bench/data/benchmark to a format compatible
with Solar's process_model CLI, wrapping ReferenceModel or reference_backward
into a standard Model class.
"""

import re
from pathlib import Path
from typing import Optional

from .parser import parse_solbenchv2_file, SolBenchV2Kernel


TEMPLATE_REFERENCE_MODEL = '''"""Solar-compatible wrapper for SolBench v2 kernel.

Original file: {original_file}
Level: {level}
Kernel type: ReferenceModel class

This file wraps the ReferenceModel class from the original benchmark
for Solar analysis.
"""
from __future__ import annotations

{imports}

# ===== Original ReferenceModel =====
{reference_model_code}

# ===== Solar-compatible Model wrapper =====
class Model(ReferenceModel):
    """Model wrapper for Solar analysis.
    
    Inherits from ReferenceModel to use the same forward pass.
    """
    pass


{get_inputs_code}


{get_init_inputs_code}
'''

TEMPLATE_REFERENCE_BACKWARD = '''"""Solar-compatible wrapper for SolBench v2 kernel.

Original file: {original_file}
Level: {level}
Kernel type: reference_backward function

This file wraps the reference_backward function from the original benchmark
for Solar analysis.
"""
from __future__ import annotations

{imports}

# ===== Original reference_backward function =====
{reference_backward_code}

# ===== Solar-compatible Model wrapper =====
class Model(torch.nn.Module):
    """Model wrapper for Solar analysis of backward pass.
    
    Wraps reference_backward as a forward pass for graph extraction.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        """Forward pass calls reference_backward for graph extraction."""
        return reference_backward(*args, **kwargs)


{get_inputs_code}


def get_init_inputs():
    """Return empty init inputs since Model has no parameters."""
    return []
'''


def convert_solbenchv2_file(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert a SolBench v2 kernel file to Solar-compatible format.
    
    Args:
        input_path: Path to the original kernel file
        output_path: Optional output path. If not provided, returns the converted code.
        
    Returns:
        The converted source code
    """
    kernel = parse_solbenchv2_file(input_path)
    
    if not kernel.is_valid:
        raise ValueError(
            f"Invalid kernel: {input_path}\n"
            f"  has_get_inputs: {kernel.has_get_inputs}\n"
            f"  has_reference: {kernel.has_reference_model or kernel.has_reference_backward}\n"
            f"  has_launch_reference_implementation: {kernel.has_launch_reference_implementation}"
        )
    
    # Prepare imports
    imports = "\n".join(kernel.imports)
    
    # Ensure torch is imported for backward template
    if "import torch" not in imports:
        imports = "import torch\nimport torch.nn as nn\n" + imports
    
    # Generate get_init_inputs if not present
    get_init_inputs_code = kernel.get_init_inputs_code if kernel.has_get_init_inputs else """
def get_init_inputs():
    \"\"\"Return initialization inputs for the model.\"\"\"
    return []
"""
    
    # Choose template based on kernel type
    if kernel.has_reference_model:
        converted = TEMPLATE_REFERENCE_MODEL.format(
            original_file=kernel.filepath,
            level=kernel.level,
            imports=imports,
            reference_model_code=kernel.reference_model_code,
            get_inputs_code=kernel.get_inputs_code,
            get_init_inputs_code=get_init_inputs_code,
        )
    else:
        converted = TEMPLATE_REFERENCE_BACKWARD.format(
            original_file=kernel.filepath,
            level=kernel.level,
            imports=imports,
            reference_backward_code=kernel.reference_backward_code,
            get_inputs_code=kernel.get_inputs_code,
        )
    
    # Clean up multiple blank lines
    converted = re.sub(r'\n{3,}', '\n\n', converted)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(converted)
        return str(output_path)
    
    return converted


def convert_solbenchv2_directory(
    input_dir: str,
    output_dir: str,
    level: Optional[str] = None,
    max_files: Optional[int] = None,
) -> dict:
    """
    Convert all valid kernels in a SolBench v2 directory.
    
    Args:
        input_dir: Path to sol-bench/data/benchmark
        output_dir: Output directory for converted files
        level: Optional filter for L1, L2, or Quant
        max_files: Optional limit on number of files to convert
        
    Returns:
        Dictionary with conversion results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    results = {
        "converted": [],
        "skipped": [],
        "failed": [],
    }
    
    # Collect files
    levels = [level] if level else ["L1", "L2", "Quant"]
    files = []
    
    for lvl in levels:
        level_dir = input_dir / lvl
        if level_dir.exists():
            files.extend(sorted(level_dir.glob("*.py")))
    
    if max_files:
        files = files[:max_files]
    
    print(f"Converting {len(files)} files...")
    
    for py_file in files:
        try:
            kernel = parse_solbenchv2_file(str(py_file))
            
            if not kernel.is_valid:
                results["skipped"].append({
                    "file": str(py_file),
                    "reason": "Missing required methods",
                })
                continue
            
            # Create output path maintaining level structure
            rel_path = py_file.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            convert_solbenchv2_file(str(py_file), str(output_path))
            results["converted"].append(str(output_path))
            
        except Exception as e:
            results["failed"].append({
                "file": str(py_file),
                "error": str(e),
            })
    
    return results
