#!/usr/bin/env python3.10
"""
Postprocess SolBench v2 kernels for Solar compatibility.

This script detects and fixes common issues in benchmark files:
- Case 1: Remove device=<xx> specifications from tensors/models
- Case 2: Replace Triton _fused_fma with pure PyTorch implementation

Usage:
    python3.10 postprocess_solbenchv2.py --input-dir <benchmark_dir> --output-dir <output_dir>
    python3.10 postprocess_solbenchv2.py --file <single_file> --output-dir <output_dir>
"""

import argparse
import os
import re
import shutil
import yaml
from pathlib import Path
from typing import Tuple


# PyTorch replacement for Triton _fused_fma
FUSED_FMA_PYTORCH = '''
def _fused_fma(y, x, s, BLOCK=128):
    """
    Fused multiply-add operation (y = y + x * s) - Pure PyTorch version.
    
    Replaces Triton kernel with equivalent PyTorch operations.
    
    Args:
        y: Accumulator tensor (modified in-place)
        x: Input tensor to multiply
        s: Scale tensor
        BLOCK: Unused (kept for API compatibility)
    
    Returns:
        y (modified in-place)
    """
    y.add_(x * s)
    return y
'''

# Replace torch._scaled_mm with regular matmul for Solar graph extraction
# Converts NVFP4 and FP8 to int8 for meta device compatibility, then performs matmul
#
# Meta device dtype support research findings:
# -----------------------------------------------
# NVFP4 / float4_e2m1fn_x2:
#   - Tensor creation on meta: YES (shape/dtype metadata stored)
#   - copy_ (type conversion): NOT IMPLEMENTED -> .to() fails
#   - addmm/matmul: NOT IMPLEMENTED -> torch.matmul fails
#   - Conclusion: NOT SUPPORTED for operations on meta device
#
# FP8 / float8_e4m3fn, float8_e5m2:
#   - Tensor creation on meta: YES
#   - copy_ (type conversion): PARTIALLY IMPLEMENTED (version-dependent)
#   - addmm/matmul on meta: NOT IMPLEMENTED in most PyTorch versions
#   - addmm on CPU: NOT IMPLEMENTED ("addmm_impl_cpu_" not implemented for Float8_e4m3fn)
#   - _scaled_mm on CPU: PARTIALLY (scalar scales only, block-wise fails)
#   - Conclusion: NOT RELIABLY SUPPORTED for operations on meta or CPU
#
# int8 / torch.int8:
#   - Full meta device support (standard dtype since PyTorch inception)
#   - Full CPU support for creation, conversion, and matmul
#   - Conclusion: FULLY SUPPORTED on meta and CPU
#
# Standard float types (float32, float16, bfloat16):
#   - Full meta and CPU support
#   - Conclusion: FULLY SUPPORTED everywhere
#
SCALED_MM_TO_MATMUL_FIX = '''
# Replace torch._scaled_mm with regular matmul for Solar compatibility
# All NVFP4/FP8 dtypes have been replaced with int8 in source code
# Accepts **kwargs to handle use_fast_accum and other _scaled_mm args

def _scaled_mm_to_matmul(mat_a, mat_b, scale_a=None, scale_b=None, bias=None, out_dtype=None, **kwargs):
    """
    Replace torch._scaled_mm with regular matmul for Solar graph extraction.
    Accepts **kwargs for compatibility with torch._scaled_mm args like use_fast_accum.
    """
    # Convert int8 to float32 for matmul (int8 matmul not supported on meta/cpu)
    if mat_a.dtype == torch.int8:
        mat_a = mat_a.to(torch.float32)
    if mat_b.dtype == torch.int8:
        mat_b = mat_b.to(torch.float32)
    
    result = torch.matmul(mat_a, mat_b)
    
    if bias is not None:
        result = result + bias
    
    if out_dtype is not None:
        result = result.to(out_dtype)
    
    return result
'''

# Injected at the top of every postprocessed Quant file that has int8 dtypes
# This is separate from SCALED_MM_TO_MATMUL_FIX so it applies even without _scaled_mm
SAFE_PARAMETER_FIX = '''
# Safe wrapper for nn.Parameter that handles int8 tensors
# int8 tensors cannot require gradients, so we convert to float32
import torch.nn as nn
_orig_nn_Parameter = nn.Parameter

def _safe_nn_Parameter(data, requires_grad=True):
    """Safe nn.Parameter that converts int8 to float32 (int8 cannot require gradients)."""
    if hasattr(data, "dtype") and data.dtype == torch.int8:
        data = data.to(torch.float32)
        requires_grad = False
    return _orig_nn_Parameter(data, requires_grad=requires_grad)

nn.Parameter = _safe_nn_Parameter

# Patch register_buffer to handle meta device: ensure buffers are never None
_orig_register_buffer = nn.Module.register_buffer

def _safe_register_buffer(self, name, tensor, persistent=True):
    """Safe register_buffer that ensures buffers are not None on meta device."""
    if tensor is None:
        _orig_register_buffer(self, name, tensor, persistent)
        return
    # Convert int8 to float32 for meta device compatibility
    if hasattr(tensor, "dtype") and tensor.dtype == torch.int8:
        tensor = tensor.to(torch.float32)
    _orig_register_buffer(self, name, tensor, persistent)

nn.Module.register_buffer = _safe_register_buffer
'''

# Replaces CuBLASRefBlockwiseGemm.scaled_mm with a simple matmul
# The blockwise for-loop fails on meta device (slicing/clone/contiguous on meta tensors)
BLOCKWISE_GEMM_SIMPLE_FIX = '''
# Replace blockwise GEMM with simple matmul for meta device compatibility
# The for-loop in qgemm_blockwise_2d does tile-by-tile processing which fails on meta device
# For Solar graph extraction, we only need the shapes, not the blockwise detail

def _simple_blockwise_scaled_mm(mat_a, mat_b, scale_a=None, scale_b=None, scale_recipe_a=None,
                                 scale_recipe_b=None, bias=None, output_dtype=None,
                                 global_decode_a=None, global_decode_b=None, use_fast_accum=True, **kwargs):
    """Simple matmul replacement for CuBLASRefBlockwiseGemm.scaled_mm.
    Ignores block structure and scales. Just does mat_a @ mat_b.T for shape extraction."""
    if mat_a.dtype == torch.int8:
        mat_a = mat_a.to(torch.float32)
    if mat_b.dtype == torch.int8:
        mat_b = mat_b.to(torch.float32)
    # mat_a: (M, K), mat_b: (N, K) -> result: (M, N)
    # Use transpose(-2, -1) instead of .t() to handle any number of dimensions
    if mat_a.shape[-1] == mat_b.shape[-1]:
        result = torch.matmul(mat_a, mat_b.transpose(-2, -1))
    elif mat_a.shape[-1] == mat_b.shape[-2] if mat_b.dim() >= 2 else mat_b.shape[0]:
        result = torch.matmul(mat_a, mat_b)
    else:
        result = torch.matmul(mat_a, mat_b.transpose(-2, -1))
    if bias is not None:
        result = result + bias
    if output_dtype is not None:
        result = result.to(output_dtype)
    return result
'''


def detect_device_specs(content: str) -> list:
    """
    Detect device=<xx> specifications in the content.
    
    Patterns detected:
    - device="cuda"
    - device='cuda'
    - device=torch.device("cuda")
    - device=qx.device (variable references are OK, not removed)
    """
    # Pattern for literal device specifications
    patterns = [
        r'device\s*=\s*["\']cuda["\']',
        r'device\s*=\s*["\']cuda:\d+["\']',
        r'device\s*=\s*["\']cpu["\']',
        r'device\s*=\s*torch\.device\(["\']cuda["\']\)',
        r'device\s*=\s*torch\.device\(["\']cuda:\d+["\']\)',
        r'device\s*=\s*torch\.device\(["\']cpu["\']\)',
    ]
    
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            matches.append((match.start(), match.end(), match.group()))
    
    return matches


def remove_device_specs(content: str) -> Tuple[str, int]:
    """
    Remove device=<xx> specifications from content.
    
    Removes:
    - device="cuda" / device='cuda' in function calls
    - device=torch.device("cuda")
    - torch.set_default_device("cuda")
    - .cuda() method calls
    - torch.cuda.synchronize()
    
    Also hoists torch.set_default_dtype() from main() to module level.
    
    Returns:
        Tuple of (modified_content, num_changes)
    """
    changes = 0
    
    # Patterns to remove (with optional trailing comma)
    patterns = [
        # device="cuda" or device='cuda' with optional trailing comma
        (r',\s*device\s*=\s*["\']cuda(:\d+)?["\']', ''),
        (r'device\s*=\s*["\']cuda(:\d+)?["\']\s*,\s*', ''),
        (r'device\s*=\s*["\']cuda(:\d+)?["\']', ''),
        # device="cpu" 
        (r',\s*device\s*=\s*["\']cpu["\']', ''),
        (r'device\s*=\s*["\']cpu["\']\s*,\s*', ''),
        (r'device\s*=\s*["\']cpu["\']', ''),
        # device=torch.device("cuda")
        (r',\s*device\s*=\s*torch\.device\(["\']cuda(:\d+)?["\']\)', ''),
        (r'device\s*=\s*torch\.device\(["\']cuda(:\d+)?["\']\)\s*,\s*', ''),
        (r'device\s*=\s*torch\.device\(["\']cuda(:\d+)?["\']\)', ''),
        # torch.set_default_device("cuda") - remove entire line
        (r'^\s*torch\.set_default_device\s*\(["\']cuda(:\d+)?["\']\)\s*$', '', re.MULTILINE),
        (r'^\s*torch\.set_default_device\s*\(["\']cpu["\']\)\s*$', '', re.MULTILINE),
        (r'torch\.set_default_device\s*\(["\']cuda(:\d+)?["\']\)\s*\n', '\n'),
        (r'torch\.set_default_device\s*\(["\']cpu["\']\)\s*\n', '\n'),
        # .cuda() method calls -> remove (tensor stays on default device)
        (r'\.cuda\(\)', ''),
        # torch.cuda.synchronize() -> remove entire line
        (r'^\s*torch\.cuda\.synchronize\(\)\s*$', '', re.MULTILINE),
        (r'torch\.cuda\.synchronize\(\)\s*\n', '\n'),
    ]
    
    for pattern_info in patterns:
        if len(pattern_info) == 3:
            pattern, replacement, flags = pattern_info
        else:
            pattern, replacement = pattern_info
            flags = 0
        
        new_content, n = re.subn(pattern, replacement, content, flags=flags)
        if n > 0:
            changes += n
            content = new_content
    
    # Hoist torch.set_default_dtype() from main()/if __name__ to module level
    # This ensures the dtype is set before model class is defined
    dtype_match = re.search(r'torch\.set_default_dtype\s*\((torch\.\w+)\)', content)
    if dtype_match:
        dtype_val = dtype_match.group(1)
        # Check if it's inside main() or if __name__ (not at module level)
        dtype_pos = dtype_match.start()
        before = content[:dtype_pos]
        if 'def main' in before[before.rfind('\ndef '):] if '\ndef ' in before else False:
            # It's inside main() - add at module level
            # Find first import or class/def
            insert_pos = 0
            for p in ['\nclass ', '\ndef ']:
                pos = content.find(p)
                if pos != -1 and (insert_pos == 0 or pos < insert_pos):
                    insert_pos = pos
            if insert_pos > 0:
                hoist_line = f'\ntorch.set_default_dtype({dtype_val})\n'
                content = content[:insert_pos] + hoist_line + content[insert_pos:]
                changes += 1
        # Also check if set_default_dtype is ONLY in main/if __name__ block
        # If there's no module-level set_default_dtype, add one
        lines = content.split('\n')
        has_module_level = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('torch.set_default_dtype') and not line.startswith(' ') and not line.startswith('\t'):
                has_module_level = True
                break
        if not has_module_level:
            # Add at module level before first class/def
            insert_pos = 0
            for p in ['\nclass ', '\ndef ']:
                pos = content.find(p)
                if pos != -1 and (insert_pos == 0 or pos < insert_pos):
                    insert_pos = pos
            if insert_pos > 0:
                hoist_line = f'\ntorch.set_default_dtype({dtype_val})\n'
                content = content[:insert_pos] + hoist_line + content[insert_pos:]
                changes += 1
    
    return content, changes


def detect_triton_fused_fma(content: str) -> bool:
    """Detect if content uses Triton _fused_fma kernel."""
    # Check for Triton kernel definition
    has_triton_kernel = '@triton.jit' in content and '_fused_fma_kernel' in content
    # Check for _fused_fma function that calls the kernel
    has_fma_func = 'def _fused_fma(' in content and '_fused_fma_kernel[grid]' in content
    
    return has_triton_kernel or has_fma_func


def detect_quantized_type_usage(content: str) -> dict:
    """
    Detect NVFP4 and FP8 quantized type usage in the content.
    
    Both NVFP4 and FP8 are NOT supported on meta device for operations
    like matmul, copy_, addmm. They need to be converted to int8.
    
    Returns:
        Dictionary with detection results for both NVFP4 and FP8
    """
    result = {
        'has_nvfp4': False,
        'has_fp8': False,
        'functions_with_nvfp4': [],
        'functions_with_fp8': [],
    }
    
    # NVFP4 type patterns
    fp4_patterns = [
        r'float4_e2m1fn_x2',
        r'Float4_e2m1fn_x2',
        r'torch\.float4_e2m1fn_x2',
    ]
    
    # FP8 type patterns
    fp8_patterns = [
        r'float8_e4m3fn(?!uz)',
        r'float8_e5m2(?!fn)',
        r'float8_e4m3fnuz',
        r'float8_e5m2fnuz',
        r'torch\.float8_e4m3fn',
        r'torch\.float8_e5m2',
    ]
    
    lines = content.split('\n')
    
    # Detect NVFP4
    for pattern in fp4_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            result['has_nvfp4'] = True
            break
    
    if result['has_nvfp4']:
        seen_funcs = set()
        for i, line in enumerate(lines):
            if any(re.search(p, line, re.IGNORECASE) for p in fp4_patterns):
                for j in range(i, max(0, i-30), -1):
                    stripped = lines[j].strip()
                    if stripped.startswith('def '):
                        func_name = re.match(r'def\s+(\w+)', stripped)
                        if func_name and func_name.group(1) not in seen_funcs:
                            result['functions_with_nvfp4'].append(func_name.group(1))
                            seen_funcs.add(func_name.group(1))
                        break
    
    # Detect FP8
    for pattern in fp8_patterns:
        if re.search(pattern, content):
            result['has_fp8'] = True
            break
    
    if result['has_fp8']:
        seen_funcs = set()
        for i, line in enumerate(lines):
            if any(re.search(p, line) for p in fp8_patterns):
                for j in range(i, max(0, i-30), -1):
                    stripped = lines[j].strip()
                    if stripped.startswith('def '):
                        func_name = re.match(r'def\s+(\w+)', stripped)
                        if func_name and func_name.group(1) not in seen_funcs:
                            result['functions_with_fp8'].append(func_name.group(1))
                            seen_funcs.add(func_name.group(1))
                        break
    
    return result


def replace_scaled_mm_with_matmul(content: str) -> Tuple[str, bool, dict]:
    """
    Replace torch._scaled_mm calls with regular matmul for Solar compatibility.
    
    Converts quantized scaled_mm to standard matmul operations.
    Detects NVFP4 usage and tracks conversions.
    
    Returns:
        Tuple of (modified_content, was_changed, conversion_metadata)
    """
    conversion_metadata = {
        'scaled_mm_replacements': [],
        'nvfp4_to_int8_conversions': [],
        'fp8_to_int8_conversions': [],
    }
    
    # Check if file uses torch._scaled_mm
    has_scaled_mm = 'torch._scaled_mm' in content
    
    if not has_scaled_mm:
        return content, False, conversion_metadata
    
    # Check if already replaced
    if '_scaled_mm_to_matmul' in content:
        return content, False, conversion_metadata
    
    # Detect quantized type usage (NVFP4 and FP8)
    quant_info = detect_quantized_type_usage(content)
    if quant_info['has_nvfp4']:
        conversion_metadata['nvfp4_to_int8_conversions'] = quant_info['functions_with_nvfp4']
    if quant_info['has_fp8']:
        conversion_metadata['fp8_to_int8_conversions'] = quant_info['functions_with_fp8']
    
    # Find insertion point (after imports, before first class/function)
    import_end = -1
    for pattern in ['\nclass ', '\ndef ', '\n# =====']:
        pos = content.find(pattern)
        if pos != -1:
            if import_end == -1 or pos < import_end:
                import_end = pos
    
    if import_end == -1:
        # No clear insertion point, add at end of imports section
        # Look for last import statement
        import_lines = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(i)
        
        if import_lines:
            import_end = sum(len(l) + 1 for l in lines[:import_lines[-1] + 1])
        else:
            # Fallback: add after first non-empty line
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    import_end = sum(len(l) + 1 for l in lines[:i+1])
                    break
    
    if import_end != -1:
        # Insert the matmul replacement function
        content = content[:import_end] + '\n' + SCALED_MM_TO_MATMUL_FIX + content[import_end:]
        
        # Replace torch._scaled_mm calls with _scaled_mm_to_matmul
        # Pattern: torch._scaled_mm(...) -> _scaled_mm_to_matmul(...)
        # Use word boundary to avoid partial matches
        matches = list(re.finditer(r'\btorch\._scaled_mm\s*\(', content))
        for match in matches:
            # Find the function this call is in
            lines = content[:match.start()].split('\n')
            for i in range(len(lines)-1, max(0, len(lines)-20), -1):
                if lines[i].strip().startswith('def '):
                    func_name = re.match(r'def\s+(\w+)', lines[i])
                    if func_name:
                        conversion_metadata['scaled_mm_replacements'].append({
                            'function': func_name.group(1),
                            'location': match.start(),
                        })
                    break
        
        content = re.sub(
            r'\btorch\._scaled_mm\s*\(',
            '_scaled_mm_to_matmul(',
            content
        )
        
        return content, True, conversion_metadata
    
    return content, False, conversion_metadata


def replace_triton_fused_fma(content: str) -> Tuple[str, bool]:
    """
    Replace Triton _fused_fma with pure PyTorch implementation.
    
    Returns:
        Tuple of (modified_content, was_changed)
    """
    if not detect_triton_fused_fma(content):
        return content, False
    
    # Remove Triton kernel definition
    # Pattern: @triton.jit\ndef _fused_fma_kernel(...): ... (until next def or class)
    kernel_pattern = r'@triton\.jit\s*\ndef _fused_fma_kernel\([^)]*\):\s*"""[^"]*"""\s*[^@]*?(?=\ndef |\nclass |\n@)'
    content, n1 = re.subn(kernel_pattern, '\n', content, flags=re.DOTALL)
    
    # Remove original _fused_fma function
    # Pattern: def _fused_fma(...): ... (until next def or class at same indent)
    fma_pattern = r'\ndef _fused_fma\([^)]*\):\s*"""[\s\S]*?"""[\s\S]*?(?=\ndef |\nclass |\n[A-Z])'
    content, n2 = re.subn(fma_pattern, FUSED_FMA_PYTORCH, content, flags=re.DOTALL)
    
    # If patterns didn't match exactly, try simpler approach
    if n2 == 0 and 'def _fused_fma(' in content:
        # Find and replace the function more aggressively
        lines = content.split('\n')
        new_lines = []
        skip_until_next_def = False
        in_fused_fma = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def _fused_fma('):
                in_fused_fma = True
                skip_until_next_def = True
                # Insert our replacement
                new_lines.append(FUSED_FMA_PYTORCH)
                continue
            
            if skip_until_next_def:
                # Check if we've reached the next function/class definition
                if (line.strip().startswith('def ') or 
                    line.strip().startswith('class ') or
                    line.strip().startswith('@')) and not line.strip().startswith('def _fused_fma'):
                    skip_until_next_def = False
                    in_fused_fma = False
                    new_lines.append(line)
                # Skip lines inside _fused_fma
                continue
            
            new_lines.append(line)
        
        if in_fused_fma or skip_until_next_def:
            # Function was at the end of file
            pass
        
        content = '\n'.join(new_lines)
        n2 = 1
    
    # Also remove the Triton kernel if still present
    if '@triton.jit' in content and '_fused_fma_kernel' in content:
        lines = content.split('\n')
        new_lines = []
        skip_kernel = False
        
        for i, line in enumerate(lines):
            if '@triton.jit' in line:
                # Check if next non-empty line is _fused_fma_kernel
                for j in range(i+1, min(i+5, len(lines))):
                    if 'def _fused_fma_kernel' in lines[j]:
                        skip_kernel = True
                        break
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        break
            
            if skip_kernel:
                if line.strip().startswith('def ') and '_fused_fma_kernel' not in line:
                    skip_kernel = False
                    new_lines.append(line)
                elif line.strip().startswith('class '):
                    skip_kernel = False
                    new_lines.append(line)
                elif line.strip().startswith('@') and '@triton' not in line:
                    skip_kernel = False
                    new_lines.append(line)
                continue
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        n1 = 1
    
    return content, (n1 > 0 or n2 > 0)


def remove_triton_imports(content: str) -> Tuple[str, int]:
    """Remove Triton import statements if no longer needed."""
    changes = 0
    
    # Check if Triton is still used elsewhere
    triton_usage = re.findall(r'triton\.|@triton', content)
    
    if len(triton_usage) == 0:
        # Safe to remove imports
        patterns = [
            r'import triton\n',
            r'import triton\.language as tl\n',
            r'from triton import [^\n]+\n',
        ]
        for pattern in patterns:
            new_content, n = re.subn(pattern, '', content)
            if n > 0:
                changes += n
                content = new_content
    
    return content, changes


def replace_quantized_dtypes_with_int8(content: str) -> Tuple[str, int, dict]:
    """
    Case 4: Replace all NVFP4 and FP8 dtype references in source code with int8.
    
    This handles:
      - torch.float4_e2m1fn_x2 -> torch.int8
      - torch.float8_e4m3fn -> torch.int8
      - torch.float8_e5m2 -> torch.int8
      - torch.float8_e4m3fnuz -> torch.int8
      - torch.float8_e5m2fnuz -> torch.int8
    
    Also handles string forms in dtype= and .to() and .view():
      - dtype=torch.float8_e4m3fn -> dtype=torch.int8
      - .view(torch.float4_e2m1fn_x2) -> .view(torch.int8)
      - .to(torch.float8_e4m3fn) -> .to(torch.int8)
    
    Returns:
        Tuple of (modified_content, num_replacements, dtype_change_records)
    """
    total_replacements = 0
    dtype_changes = []
    
    # Map of unsupported quantized dtype patterns -> replacement
    dtype_replacements = [
        # NVFP4 types
        ('torch.float4_e2m1fn_x2', 'torch.int8', 'nvfp4 float4_e2m1fn_x2'),
        # FP8 types
        ('torch.float8_e4m3fnuz', 'torch.int8', 'fp8 float8_e4m3fnuz'),
        ('torch.float8_e5m2fnuz', 'torch.int8', 'fp8 float8_e5m2fnuz'),
        ('torch.float8_e4m3fn', 'torch.int8', 'fp8 float8_e4m3fn'),
        ('torch.float8_e5m2', 'torch.int8', 'fp8 float8_e5m2'),
    ]
    
    for old_dtype, new_dtype, dtype_label in dtype_replacements:
        count = content.count(old_dtype)
        if count > 0:
            # Find functions containing these dtype references
            lines = content.split('\n')
            funcs_affected = set()
            for i, line in enumerate(lines):
                if old_dtype in line:
                    # Walk backward to find enclosing function
                    for j in range(i, max(0, i - 40), -1):
                        stripped = lines[j].strip()
                        if stripped.startswith('def '):
                            func_match = re.match(r'def\s+(\w+)', stripped)
                            if func_match:
                                funcs_affected.add(func_match.group(1))
                            break
                        elif stripped.startswith('class '):
                            class_match = re.match(r'class\s+(\w+)', stripped)
                            if class_match:
                                funcs_affected.add(class_match.group(1))
                            break
            
            content = content.replace(old_dtype, new_dtype)
            total_replacements += count
            
            dtype_changes.append({
                'orig_dtype': dtype_label,
                'new_dtype': 'int8',
                'count': count,
                'functions': list(funcs_affected),
            })
    
    # Also replace bare type references (without torch. prefix) in dtype= contexts
    # e.g. float8_e4m3fn without torch. prefix — less common but possible
    bare_replacements = [
        (r"(?<!\.)float4_e2m1fn_x2", 'int8', 'nvfp4 float4_e2m1fn_x2'),
        (r"(?<!\.)float8_e4m3fnuz", 'int8', 'fp8 float8_e4m3fnuz'),
        (r"(?<!\.)float8_e5m2fnuz", 'int8', 'fp8 float8_e5m2fnuz'),
        (r"(?<!\.)float8_e4m3fn", 'int8', 'fp8 float8_e4m3fn'),
        (r"(?<!\.)float8_e5m2(?!fn)", 'int8', 'fp8 float8_e5m2'),
    ]
    
    for pattern, replacement, dtype_label in bare_replacements:
        # Only replace if in non-comment, non-string context
        # Simple approach: skip lines that are only comments or docstrings
        new_content, n = re.subn(pattern, replacement, content)
        if n > 0:
            content = new_content
            total_replacements += n
            # Check if we already recorded this dtype
            existing = [d for d in dtype_changes if d['orig_dtype'] == dtype_label]
            if existing:
                existing[0]['count'] += n
            else:
                dtype_changes.append({
                    'orig_dtype': dtype_label,
                    'new_dtype': 'int8',
                    'count': n,
                    'functions': [],
                })
    
    return content, total_replacements, dtype_changes


def fix_nn_parameter_int8(content: str) -> Tuple[str, int]:
    """
    Case 5: Fix nn.Parameter wrapping int8 tensors.
    
    int8 tensors cannot require gradients, so nn.Parameter(int8_tensor) fails.
    
    Instead of modifying multi-line nn.Parameter calls with fragile regex,
    we inject a _safe_nn_Parameter wrapper that monkey-patches nn.Parameter
    at runtime to auto-convert int8 to float32.
    
    Returns:
        Tuple of (modified_content, num_fixes)
    """
    # Count nn.Parameter calls
    count = len(re.findall(r'nn\.Parameter\(', content))
    
    if count == 0:
        return content, 0
    
    # Check if the safe parameter wrapper is already present
    if '_safe_nn_Parameter' in content:
        return content, count
    
    # Inject SAFE_PARAMETER_FIX at the top (after imports, before first class/def)
    import_end = -1
    for pattern in ['\nclass ', '\ndef ']:
        pos = content.find(pattern)
        if pos != -1:
            if import_end == -1 or pos < import_end:
                import_end = pos
    
    if import_end != -1:
        content = content[:import_end] + '\n' + SAFE_PARAMETER_FIX + content[import_end:]
    
    return content, count


def replace_blockwise_gemm_calls(content: str) -> Tuple[str, int]:
    """
    Case 7: Replace CuBLASRefBlockwiseGemm.scaled_mm calls with simple matmul.
    
    The blockwise GEMM uses a for-loop with tile slicing that fails on meta device.
    Replace call sites like self.gemm_ref.scaled_mm(...) with _simple_blockwise_scaled_mm(...).
    
    Returns:
        Tuple of (modified_content, num_replacements)
    """
    # Check if blockwise GEMM is used
    if 'qgemm_blockwise_2d' not in content and 'CuBLASRefBlockwiseGemm' not in content:
        return content, 0
    
    # Already replaced
    if '_simple_blockwise_scaled_mm' in content:
        return content, 0
    
    replacements = 0
    
    # Inject the simple replacement function
    import_end = -1
    for pattern in ['\nclass ', '\ndef ']:
        pos = content.find(pattern)
        if pos != -1:
            if import_end == -1 or pos < import_end:
                import_end = pos
    
    if import_end != -1:
        content = content[:import_end] + '\n' + BLOCKWISE_GEMM_SIMPLE_FIX + content[import_end:]
    
    # Replace call sites: self.gemm_ref.scaled_mm( -> _simple_blockwise_scaled_mm(
    # Pattern matches any self.<name>.scaled_mm( call
    pattern = r'self\.\w+\.scaled_mm\s*\('
    matches = list(re.finditer(pattern, content))
    
    if matches:
        # Replace from end to start to preserve offsets
        for match in reversed(matches):
            content = content[:match.start()] + '_simple_blockwise_scaled_mm(' + content[match.end():]
            replacements += 1
    
    return content, replacements


def fix_forward_signature_mismatch(content: str) -> Tuple[str, int]:
    """
    Case 8: Fix forward() signature mismatch with get_inputs().
    
    Some Quant benchmarks have forward(self, x, scale_input, scale_weight) but
    get_inputs() only returns (x,). Solar calls forward(*get_inputs()) which fails.
    
    Fix: Make extra forward args optional with default None.
    e.g. forward(self, x, scale_input=None, scale_weight=None)
    
    Returns:
        Tuple of (modified_content, num_fixes)
    """
    fixes = 0
    
    # First find get_inputs and count how many values it returns
    get_inputs_match = re.search(r'def get_inputs\s*\([^)]*\)[^:]*:.*?return\s+\(?([^)]+)\)?', 
                                  content, re.DOTALL)
    
    if not get_inputs_match:
        return content, 0
    
    # Count comma-separated items in return (handle trailing comma)
    return_body = get_inputs_match.group(1).strip().rstrip(',').strip()
    num_inputs = return_body.count(',') + 1 if return_body else 0
    
    # Find forward method definitions (multi-line support with re.DOTALL)
    # Match: def forward(\n        self,\n        arg1: type,\n        arg2: type,\n    ) -> type:
    forward_pattern = r'(def forward\s*\(\s*self\s*,\s*)(.*?)(\)\s*(?:->.*?)?:)'
    forward_matches = list(re.finditer(forward_pattern, content, re.DOTALL))
    
    for match in reversed(forward_matches):
        args_str = match.group(2)
        
        # Split args by comma, handling multi-line and type annotations
        # Remove leading/trailing whitespace and newlines
        args_clean = re.sub(r'\s+', ' ', args_str).strip().rstrip(',').strip()
        
        if not args_clean:
            continue
        
        # Split on commas that are not inside brackets/parens
        # Simple approach: split on ', ' pattern after collapsing whitespace
        args = [a.strip() for a in args_clean.split(',') if a.strip()]
        
        # Filter out empty args from trailing commas
        args = [a for a in args if a]
        
        # If forward has more args than get_inputs provides, make extras optional
        if len(args) > num_inputs:
            new_args = []
            for i, arg in enumerate(args):
                if i < num_inputs:
                    new_args.append(arg)
                else:
                    # Make optional with default None (if not already has =)
                    if '=' in arg:
                        new_args.append(arg)
                    elif ':' in arg:
                        # Has type hint: add = None after type
                        new_args.append(f'{arg} = None')
                    else:
                        new_args.append(f'{arg}=None')
            
            # Rebuild with same formatting style as original
            if '\n' in args_str:
                # Multi-line: preserve indentation
                indent = '        '
                new_args_str = ',\n'.join(f'{indent}{a}' for a in new_args)
                new_args_str = '\n' + new_args_str + ',\n    '
            else:
                new_args_str = ', '.join(new_args)
            
            new_match = match.group(1) + new_args_str + match.group(3)
            content = content[:match.start()] + new_match + content[match.end():]
            
            # Add guard at the start of the forward body:
            # When optional scale args are None, skip quantization and do simple linear
            body_start = content.find(':', match.start() + len(new_match) - 10)
            if body_start != -1:
                after_colon = content[body_start + 1:]
                docstring_end = body_start + 1
                stripped_after = after_colon.lstrip()
                if stripped_after.startswith('"""') or stripped_after.startswith("'''"):
                    quote = stripped_after[:3]
                    close_idx = after_colon.find(quote, after_colon.find(quote) + 3)
                    if close_idx != -1:
                        docstring_end = body_start + 1 + close_idx + 3
                        next_newline = content.find('\n', docstring_end)
                        if next_newline != -1:
                            docstring_end = next_newline
                
                # Build guard: if any optional arg is None, do simple linear and return
                optional_arg_names = []
                for arg_str in new_args[num_inputs:]:
                    arg_name = re.match(r'(\w+)', arg_str.strip())
                    if arg_name:
                        optional_arg_names.append(arg_name.group(1))
                
                if optional_arg_names:
                    # Get first required input arg name (the main tensor)
                    first_input = re.match(r'(\w+)', new_args[0].strip())
                    first_input_name = first_input.group(1) if first_input else 'x'
                    
                    conditions = ' or '.join(f'{n} is None' for n in optional_arg_names)
                    guard_code = (
                        f'\n        # Guard: skip quantization if scale args not provided\n'
                        f'        if {conditions}:\n'
                        f'            # Simple linear pass without quantization\n'
                        f'            if hasattr(self, "weight"):\n'
                        f'                return torch.nn.functional.linear({first_input_name}, self.weight)\n'
                        f'            elif hasattr(self, "o_proj"):\n'
                        f'                return self.o_proj({first_input_name})\n'
                        f'            else:\n'
                        f'                return {first_input_name}\n'
                    )
                    content = content[:docstring_end] + guard_code + content[docstring_end:]
            
            fixes += 1
    
    return content, fixes


def auto_call_quantize_weights(content: str) -> Tuple[str, int]:
    """
    Case 9: Auto-call quantize_weights() at end of ReferenceModel.__init__ only.
    
    Some models register buffers as None in __init__ and fill them in quantize_weights().
    Solar never calls quantize_weights(), leaving buffers as None -> NoneType errors.
    
    Fix: Add self.quantize_weights() call at end of ReferenceModel.__init__() ONLY.
    Does NOT modify __init__ of other classes (BlockWiseScalerNVFP4 etc.).
    
    Returns:
        Tuple of (modified_content, num_fixes)
    """
    # Check if quantize_weights method exists
    if 'def quantize_weights(self' not in content:
        return content, 0
    
    # Check if it's already called in any __init__
    if 'self.quantize_weights()' in content:
        return content, 0
    
    # Find ONLY the ReferenceModel class and its __init__
    fixes = 0
    lines = content.split('\n')
    new_lines = []
    in_reference_model = False
    in_init = False
    init_indent = ''
    class_indent = ''
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        stripped = line.strip()
        
        # Detect ReferenceModel class
        if stripped.startswith('class ReferenceModel'):
            in_reference_model = True
            class_indent = line[:len(line) - len(line.lstrip())]
            continue
        
        # Detect leaving ReferenceModel (another class at same indent level)
        if in_reference_model and not in_init:
            if stripped.startswith('class ') and not line.startswith(class_indent + ' ') and not line.startswith(class_indent + '\t'):
                if line[:len(line) - len(line.lstrip())] == class_indent:
                    in_reference_model = False
                    continue
        
        # Detect __init__ inside ReferenceModel
        if in_reference_model and stripped.startswith('def __init__(self'):
            in_init = True
            init_indent = line[:len(line) - len(line.lstrip())] + '    '
            continue
        
        if in_init:
            # Check if we've left __init__ (next def at same indent as __init__)
            if stripped and not line.startswith(init_indent) and not line.startswith(init_indent + ' '):
                if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('@'):
                    # Insert quantize_weights() call before this line
                    new_lines.insert(-1, f'{init_indent}self.quantize_weights()')
                    new_lines.insert(-1, '')
                    in_init = False
                    fixes += 1
    
    if in_init:
        # __init__ was the last method - add at the end
        new_lines.append(f'{init_indent}self.quantize_weights()')
        fixes += 1
    
    content = '\n'.join(new_lines)
    return content, fixes


def replace_quantize_functions(content: str) -> Tuple[str, int]:
    """
    Case 6: Replace quantize/quantize_weights/apply_scaling functions
    that produce FP4/FP8 tensors with stubs returning int8 tensors.
    
    These functions create quantized tensors during model __init__ which
    fail on CPU/meta device. Replace them with simple int8 tensor creators.
    
    Returns:
        Tuple of (modified_content, num_replacements)
    """
    replacements = 0
    
    # Replace .view(torch.int8) that might fail if tensor is packed FP4
    # The dtype replacement (Case 4) already changed torch.float4_e2m1fn_x2 -> torch.int8
    # But .view(torch.int8) on a tensor that's already int8 is a no-op, which is fine
    
    # Replace quantize_to_fp4 / quantize_weights / _quantize style functions
    # These typically:
    # 1. Take a float tensor
    # 2. Quantize it to FP4/FP8 format
    # 3. Return quantized tensor + scale factors
    #
    # We replace the body to return int8 tensors of same shape + float32 scales
    
    # Pattern: find function definitions that do quantization
    quant_func_patterns = [
        r'def quantize_to_fp4\(',
        r'def quantize_weights\(',
        r'def _quantize_to_fp4\(',
        r'def quantize\(',
    ]
    
    for pattern in quant_func_patterns:
        if re.search(pattern, content):
            replacements += 1
    
    # Instead of rewriting function bodies (too complex and error-prone),
    # we rely on Case 4 (dtype replacement) which already converts all
    # FP4/FP8 type references to int8. The quantize functions will then
    # produce int8 tensors naturally since their dtype targets are now int8.
    
    return content, replacements
def write_conversion_metadata(metadata: dict, output_path: str) -> None:
    """Write dtype conversion metadata to YAML file under output/graph/."""
    # Format metadata for YAML
    yaml_data = {
        'dtype_conversions': [],
    }
    
    # Add scaled_mm -> matmul conversions
    for conv in metadata.get('scaled_mm_replacements', []):
        yaml_data['dtype_conversions'].append({
            'function': conv.get('function', 'unknown'),
            'operation': 'matmul',
            'orig_dtypes': 'nvfp4 float4_e2m1fn_x2',
            'new_dtypes': 'int8',
            'reason': 'nvfp4 not supported on meta device, converted to int8',
        })
    
    # Add NVFP4 -> int8 conversions detected in functions
    for func in metadata.get('nvfp4_to_int8_conversions', []):
        yaml_data['dtype_conversions'].append({
            'function': func,
            'operation': 'dtype_cast',
            'orig_dtypes': 'nvfp4 float4_e2m1fn_x2',
            'new_dtypes': 'int8',
            'reason': 'nvfp4 not supported on meta device faketensor, converted to int8',
        })
    
    # Add FP8 -> int8 conversions detected in functions
    for func in metadata.get('fp8_to_int8_conversions', []):
        yaml_data['dtype_conversions'].append({
            'function': func,
            'operation': 'dtype_cast',
            'orig_dtypes': 'fp8 float8_e4m3fn',
            'new_dtypes': 'int8',
            'reason': 'fp8 not reliably supported on meta device, converted to int8',
        })
    
    # Add source-level dtype replacements (Case 4)
    for change in metadata.get('source_dtype_replacements', []):
        for func in change.get('functions', ['global']):
            yaml_data['dtype_conversions'].append({
                'function': func,
                'operation': 'source_dtype_replacement',
                'orig_dtypes': change['orig_dtype'],
                'new_dtypes': change['new_dtype'],
                'count': change['count'],
                'reason': 'not supported on meta/cpu device, replaced in source code',
            })
    
    if yaml_data['dtype_conversions']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    else:
        # Create empty metadata file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)


def postprocess_file(input_path: str, output_dir: str) -> dict:
    """
    Postprocess a single benchmark file.
    
    Args:
        input_path: Path to input file
        output_dir: Output directory
        
    Returns:
        Dictionary with processing results
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    filename = input_path.stem
    
    result = {
        'file': str(input_path),
        'case1_device_removed': 0,
        'case2_fma_replaced': False,
        'case3_scaled_mm_replaced': False,
        'case4_dtype_replaced': 0,
        'triton_imports_removed': 0,
        'modified': False,
        'orig_file': None,
        'src_file': None,
    }
    
    # Read original content
    with open(input_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    content = original_content
    
    # Case 1: Remove device specifications
    content, device_changes = remove_device_specs(content)
    result['case1_device_removed'] = device_changes
    
    # Case 2: Replace Triton _fused_fma
    content, fma_replaced = replace_triton_fused_fma(content)
    result['case2_fma_replaced'] = fma_replaced
    
    # Case 3: Replace torch._scaled_mm with regular matmul
    content, scaled_mm_replaced, conversion_metadata = replace_scaled_mm_with_matmul(content)
    result['case3_scaled_mm_replaced'] = scaled_mm_replaced
    result['conversion_metadata'] = conversion_metadata
    
    # Case 4: Replace all NVFP4/FP8 dtype references with int8 in source code
    content, dtype_replacements, dtype_changes = replace_quantized_dtypes_with_int8(content)
    result['case4_dtype_replaced'] = dtype_replacements
    if dtype_changes:
        if 'conversion_metadata' not in result:
            result['conversion_metadata'] = {}
        result['conversion_metadata']['source_dtype_replacements'] = dtype_changes
    
    # Case 5: Fix nn.Parameter wrapping int8 tensors (int8 can't require gradients)
    content, param_fixes = fix_nn_parameter_int8(content)
    result['case5_param_fixed'] = param_fixes
    
    # Case 6: Track quantize function replacements (handled by Case 4 dtype changes)
    content, quant_func_count = replace_quantize_functions(content)
    result['case6_quantize_funcs'] = quant_func_count
    
    # Case 7: Replace blockwise GEMM loops with simple matmul
    content, gemm_replacements = replace_blockwise_gemm_calls(content)
    result['case7_blockwise_gemm'] = gemm_replacements
    
    # Case 8: Fix forward() signature mismatch with get_inputs()
    content, sig_fixes = fix_forward_signature_mismatch(content)
    result['case8_forward_sig_fixed'] = sig_fixes
    
    # Case 9: Auto-call quantize_weights() at end of __init__
    content, quant_init_fixes = auto_call_quantize_weights(content)
    result['case9_quantize_init'] = quant_init_fixes
    
    # Clean up Triton imports if no longer needed
    if fma_replaced:
        content, import_changes = remove_triton_imports(content)
        result['triton_imports_removed'] = import_changes
    
    # Determine if file was modified
    result['modified'] = (device_changes > 0 or fma_replaced or result['case3_scaled_mm_replaced'] 
                          or dtype_replacements > 0 or param_fixes > 0 or gemm_replacements > 0
                          or sig_fixes > 0 or quant_init_fixes > 0)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result['modified']:
        # Save original to orig_<filename>.py
        orig_path = output_dir / f"orig_{filename}.py"
        with open(orig_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        result['orig_file'] = str(orig_path)
        
        # Save modified to src_<filename>.py
        src_path = output_dir / f"src_{filename}.py"
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(content)
        result['src_file'] = str(src_path)
    else:
        # Just copy to src_<filename>.py
        src_path = output_dir / f"src_{filename}.py"
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(content)
        result['src_file'] = str(src_path)
    
    # Write conversion metadata if there were any dtype changes
    cm = result.get('conversion_metadata', {})
    has_conversions = (
        cm.get('scaled_mm_replacements') or
        cm.get('nvfp4_to_int8_conversions') or
        cm.get('fp8_to_int8_conversions') or
        cm.get('source_dtype_replacements')
    )
    if has_conversions:
        metadata_path = output_dir / "metadata.yaml"
        write_conversion_metadata(cm, str(metadata_path))
        result['metadata_file'] = str(metadata_path)
    
    return result


def postprocess_directory(input_dir: str, output_dir: str, level: str = None) -> list:
    """
    Postprocess all files in a benchmark directory.
    
    Args:
        input_dir: Path to benchmark directory (e.g., sol-bench/data/benchmark)
        output_dir: Output directory
        level: Optional level filter (L1, L2, Quant)
        
    Returns:
        List of processing results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    results = []
    
    # Determine levels to process
    levels = [level] if level else ['L1', 'L2', 'Quant']
    
    for lvl in levels:
        level_dir = input_dir / lvl
        if not level_dir.exists():
            print(f"Warning: Level directory not found: {level_dir}")
            continue
        
        level_output = output_dir / lvl
        
        py_files = sorted(level_dir.glob('*.py'))
        print(f"Processing {len(py_files)} files in {lvl}...")
        
        for py_file in py_files:
            result = postprocess_file(str(py_file), str(level_output))
            result['level'] = lvl
            results.append(result)
            
            status = "MODIFIED" if result['modified'] else "OK"
            changes = []
            if result['case1_device_removed'] > 0:
                changes.append(f"device:{result['case1_device_removed']}")
            if result['case2_fma_replaced']:
                changes.append("fma")
            
            change_str = f" [{','.join(changes)}]" if changes else ""
            print(f"  {status}: {py_file.name}{change_str}")
    
    return results


def generate_summary(results: list, output_path: str):
    """Generate CSV summary of postprocessing results."""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'level', 'file', 'modified', 'case1_device_removed', 
            'case2_fma_replaced', 'case3_scaled_mm_replaced', 'case4_dtype_replaced',
            'triton_imports_removed', 'src_file', 'orig_file'
        ])
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'level': r.get('level', ''),
                'file': Path(r['file']).name,
                'modified': r['modified'],
                'case1_device_removed': r['case1_device_removed'],
                'case2_fma_replaced': r['case2_fma_replaced'],
                'case3_scaled_mm_replaced': r.get('case3_scaled_mm_replaced', False),
                'case4_dtype_replaced': r.get('case4_dtype_replaced', 0),
                'triton_imports_removed': r['triton_imports_removed'],
                'src_file': Path(r['src_file']).name if r['src_file'] else '',
                'orig_file': Path(r['orig_file']).name if r['orig_file'] else '',
            })
    
    print(f"\nSummary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess SolBench v2 kernels for Solar compatibility"
    )
    parser.add_argument(
        '--input-dir',
        help='Path to benchmark directory (sol-bench/data/benchmark)'
    )
    parser.add_argument(
        '--file',
        help='Process a single file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--level',
        choices=['L1', 'L2', 'Quant'],
        help='Process only specific level'
    )
    parser.add_argument(
        '--summary',
        help='Path to output CSV summary file'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Process single file
        result = postprocess_file(args.file, args.output_dir)
        print(f"Processed: {args.file}")
        print(f"  Modified: {result['modified']}")
        print(f"  Device specs removed: {result['case1_device_removed']}")
        print(f"  FMA replaced: {result['case2_fma_replaced']}")
        print(f"  Output: {result['src_file']}")
        if result['orig_file']:
            print(f"  Original saved: {result['orig_file']}")
        results = [result]
    elif args.input_dir:
        # Process directory
        results = postprocess_directory(args.input_dir, args.output_dir, args.level)
        
        # Print summary
        total = len(results)
        modified = sum(1 for r in results if r['modified'])
        device_changes = sum(r['case1_device_removed'] for r in results)
        fma_replaced = sum(1 for r in results if r['case2_fma_replaced'])
        
        print(f"\n{'='*60}")
        print("POSTPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files:        {total}")
        print(f"Modified files:     {modified}")
        print(f"Device specs removed: {device_changes} (across all files)")
        print(f"FMA replacements:   {fma_replaced} files")
        print(f"{'='*60}")
    else:
        parser.error("Either --input-dir or --file must be specified")
        return 1
    
    # Generate summary CSV if requested
    if args.summary:
        generate_summary(results, args.summary)
    
    return 0


if __name__ == '__main__':
    exit(main())
