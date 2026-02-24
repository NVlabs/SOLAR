# Quantized Kernel Postprocessing for Solar

This document describes all conversions applied by `postprocess.py` to make
SolBench v2 Quant kernels compatible with Solar's graph extraction pipeline.

Solar uses `device=meta` (shape-only) then falls back to `device=cpu` for
torchview graph extraction. Neither meta nor CPU supports NVFP4/FP8 operations.

## Error Summary from Quant Benchmarks

| Error | Root Cause | Count | Fix |
|---|---|---|---|
| `unexpected keyword argument 'use_fast_accum'` | `_scaled_mm_to_matmul` missing kwargs | ~20 | Case 3: add `**kwargs` |
| `Only Tensors of floating point and complex dtype can require gradients` | `nn.Parameter(int8_tensor)` | ~1 | Case 5: runtime monkey-patch |
| `Torch not compiled with CUDA enabled` | `.cuda()` calls in source | ~1 | Case 1: remove `.cuda()` |
| `Failed to run torchgraph see error message` (blockwise GEMM) | for-loop tile slicing on meta | ~18 | Case 7: simple matmul |
| `forward() missing required positional argument` | get_inputs/forward mismatch | ~2 | Case 8: add `=None` defaults |
| `Input type (BFloat16) and bias type (float) should be the same` | set_default_dtype in main() | ~1 | Case 1: hoist dtype to module level |
| `t() expects <= 2 dimensions, but self is 3D` | `.t()` on 3D tensor | ~1 | Case 7: use `.transpose(-2,-1)` |
| `'NoneType' object has no attribute 'shape'` | buffers registered as None | ~1 | Case 9: auto-call quantize_weights |
| `SyntaxError: invalid syntax` | old Case 5 regex broke multi-line | ~3 | Case 5: runtime monkey-patch |
| `'torch.dtype' object has no attribute 'float'` | old Case 5 .float() on dtype | ~2 | Case 5: runtime monkey-patch |
| `'BlockWiseScalerNVFP4' has no attribute 'quantize_weights'` | Case 9 added to wrong class | ~1 | Case 9: only target ReferenceModel |
| `'NoneType' has no attribute 'unsqueeze'` | Case 8 made scale None | ~2 | Case 8: inject early-return guard |
| `IndexError: Dimension out of range` | dummy torch.ones(1) too small | ~1 | Case 8: use F.linear instead of dummy |

## Postprocessing Cases

### Case 1: Remove Device Specifications

Removes all CUDA device references so Solar can set meta/cpu automatically.

**What is removed:**
- `device="cuda"` / `device='cuda'` in function parameters
- `device=torch.device("cuda")`
- `torch.set_default_device("cuda")` — entire lines
- `.cuda()` method calls — e.g. `tensor.cuda()` becomes `tensor`
- `torch.cuda.synchronize()` — entire lines

**What is hoisted to module level:**
- `torch.set_default_dtype(torch.bfloat16)` — if found inside `main()`,
  hoisted to module level so the dtype is set before model class definition.
  This prevents dtype mismatches (e.g., Conv1d bias float32 vs input bfloat16).

**Example:**
```python
# Before
x = torch.randn(3, 4, device="cuda")
weight = weight.cuda()
torch.set_default_device("cuda")
torch.cuda.synchronize()

def main():
    torch.set_default_dtype(torch.bfloat16)  # only in main()

# After
x = torch.randn(3, 4)
weight = weight
# (device lines removed)

torch.set_default_dtype(torch.bfloat16)  # hoisted to module level
```

### Case 2: Replace Triton _fused_fma

Replaces Triton JIT kernel `_fused_fma` with pure PyTorch equivalent.

**What is replaced:**
- `@triton.jit def _fused_fma_kernel(...)` — removed
- `def _fused_fma(...)` using Triton grid — replaced with `y.add_(x * s)`
- Triton imports removed if no longer needed

**Example:**
```python
# Before (Triton)
@triton.jit
def _fused_fma_kernel(y_ptr, x_ptr, s_ptr, ...):
    ...

def _fused_fma(y, x, s, BLOCK=128):
    _fused_fma_kernel[grid](y, x, s, ...)
    return y

# After (PyTorch)
def _fused_fma(y, x, s, BLOCK=128):
    y.add_(x * s)
    return y
```

### Case 3: Replace torch._scaled_mm with matmul

Replaces `torch._scaled_mm()` with `_scaled_mm_to_matmul()` wrapper.

**What is replaced:**
- `torch._scaled_mm(mat_a, mat_b, scale_a, scale_b, ...)` → `_scaled_mm_to_matmul(...)`
- The wrapper converts int8 inputs to float32 for matmul
- Accepts `**kwargs` for compatibility with `use_fast_accum` and other args
- Scale factors are ignored (scales already baked into dtype conversion)

**Example:**
```python
# Before
result = torch._scaled_mm(
    qx_block, qw_block.t(),
    scale_a=one, scale_b=one,
    out_dtype=torch.float32,
    use_fast_accum=True,
)

# After
result = _scaled_mm_to_matmul(
    qx_block, qw_block.t(),
    scale_a=one, scale_b=one,
    out_dtype=torch.float32,
    use_fast_accum=True,  # accepted via **kwargs, ignored
)
```

### Case 4: Replace Quantized Dtype References with int8

Replaces ALL NVFP4 and FP8 dtype references in source code with `torch.int8`.

**What is replaced:**

| Original | Replaced With |
|---|---|
| `torch.float4_e2m1fn_x2` | `torch.int8` |
| `torch.float8_e4m3fn` | `torch.int8` |
| `torch.float8_e5m2` | `torch.int8` |
| `torch.float8_e4m3fnuz` | `torch.int8` |
| `torch.float8_e5m2fnuz` | `torch.int8` |

This covers ALL usage patterns:
- `.to(dtype=torch.float8_e4m3fn)` → `.to(dtype=torch.int8)`
- `.view(torch.float4_e2m1fn_x2)` → `.view(torch.int8)`
- `dtype=torch.float8_e4m3fn` in tensor creation → `dtype=torch.int8`

**Why int8:**
- Meta device: full support for creation, indexing, copy_, matmul (via float cast)
- CPU device: full support
- 8-bit width: similar to FP8, reasonable proxy for FP4

### Case 5: Fix nn.Parameter with int8 Tensors

int8 tensors cannot require gradients. `nn.Parameter(int8_tensor)` raises:
`RuntimeError: Only Tensors of floating point and complex dtype can require gradients`

**Approach:** Instead of fragile regex on multi-line `nn.Parameter(...)` calls,
injects a runtime monkey-patch `_safe_nn_Parameter` that auto-converts int8 to
float32 at runtime. Also patches `register_buffer` to convert int8 to float32.

**What is injected:**
```python
_orig_nn_Parameter = nn.Parameter

def _safe_nn_Parameter(data, requires_grad=True):
    if hasattr(data, "dtype") and data.dtype == torch.int8:
        data = data.to(torch.float32)
        requires_grad = False
    return _orig_nn_Parameter(data, requires_grad=requires_grad)

nn.Parameter = _safe_nn_Parameter
```

This handles ALL `nn.Parameter` calls correctly regardless of multi-line
formatting or nested expressions.

### Case 6: Quantize Function Tracking

Tracks quantize functions that produce FP4/FP8 tensors. These are handled
implicitly by Case 4 — since all FP4/FP8 dtype targets are replaced with int8,
the quantize functions naturally produce int8 tensors.

**Functions tracked:**
- `quantize_to_fp4()`
- `quantize_weights()`
- `_quantize_to_fp4()`
- `quantize()`

### Case 7: Replace Blockwise GEMM Loop with Simple Matmul

The `CuBLASRefBlockwiseGemm.scaled_mm` method uses a for-loop that iterates
over tiles, doing tensor slicing, clone, contiguous, and per-tile matmul.
This fails on meta device because meta tensors can't be sliced with
dynamic ranges and cloned properly.

**What is replaced:**
- `self.gemm_ref.scaled_mm(...)` → `_simple_blockwise_scaled_mm(...)`
- The replacement converts int8→float32 and does a single `torch.matmul`
- Uses `.transpose(-2, -1)` instead of `.t()` to handle 3D+ tensors
- Block structure, scales, and tile loop are all bypassed

**Example:**
```python
# Before (blockwise GEMM with tile loop - fails on meta)
output = self.gemm_ref.scaled_mm(
    mat_a=attn_output_fp8,
    mat_b=weight_fp8,
    scale_a=scale_input,
    scale_recipe_a=ScalingType.BlockWise1x128,
    scale_b=scale_weight_cublas,
    scale_recipe_b=ScalingType.BlockWise128x128,
    bias=None,
    output_dtype=self.dtype,
    use_fast_accum=True,
)

# After (simple matmul - works on meta)
output = _simple_blockwise_scaled_mm(
    mat_a=attn_output_fp8,
    mat_b=weight_fp8,
    scale_a=scale_input,
    scale_recipe_a=ScalingType.BlockWise1x128,
    scale_b=scale_weight_cublas,
    scale_recipe_b=ScalingType.BlockWise128x128,
    bias=None,
    output_dtype=self.dtype,
    use_fast_accum=True,
)
```

### Case 8: Fix forward() Signature Mismatch with get_inputs()

Some Quant benchmarks have `forward(self, x, scale_input, scale_weight)` but
`get_inputs()` only returns `(x,)`. Solar calls `forward(*get_inputs())` which
fails with `missing required positional argument`.

**What is fixed:**
- Extra forward args beyond what `get_inputs()` provides are made optional
- Adds `=None` defaults to extra parameters

**Example:**
```python
# Before (fails: forward expects 3 args, get_inputs returns 2)
def forward(
    self,
    attn_output: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor,
) -> torch.Tensor:

# After (extra args default to None)
def forward(
    self,
    attn_output: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor = None,
) -> torch.Tensor:
```

Note: Uses `re.DOTALL` to handle multi-line forward signatures with type annotations.

Also injects a guard at the start of the forward body that bypasses the
quantization path entirely when optional scale args are None. Instead of
creating dummy tensors (which fail due to dimension mismatches), it does
a simple `F.linear()` and returns early:

```python
# Injected guard: skip quantization if scale args not provided
if scale_weight is None:
    if hasattr(self, "weight"):
        return torch.nn.functional.linear(attn_output, self.weight)
    elif hasattr(self, "o_proj"):
        return self.o_proj(attn_output)
    else:
        return attn_output
```

### Case 9: Auto-call quantize_weights() at End of ReferenceModel.__init__

Some models register buffers as `None` in `__init__` and fill them later via a
separate `quantize_weights()` method. Solar never calls `quantize_weights()`,
leaving buffers as `None` which causes `'NoneType' has no attribute 'shape'`.

**Important:** Only targets `ReferenceModel.__init__`, NOT other classes like
`BlockWiseScalerNVFP4` which don't have `quantize_weights()`.

**What is fixed:**
- Detects `def quantize_weights(self` method in the model class
- Adds `self.quantize_weights()` call at the end of `__init__()`

**Example:**
```python
# Before (__init__ registers None buffers, quantize_weights never called)
class ReferenceModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.register_buffer('q_proj_weight_fp4', None)
        self.register_buffer('q_proj_scale', None)
    
    def quantize_weights(self):
        self.q_proj_weight_fp4 = self.scaler.quantize(...)
        self.q_proj_scale = ...

# After (quantize_weights auto-called)
class ReferenceModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.register_buffer('q_proj_weight_fp4', None)
        self.register_buffer('q_proj_scale', None)
        self.quantize_weights()
    
    def quantize_weights(self):
        self.q_proj_weight_fp4 = self.scaler.quantize(...)
        self.q_proj_scale = ...
```

## Progress Summary

| Run | Passed | Failed | Skipped | Total |
|---|---|---|---|---|
| Run 1 (no postprocessing) | 1 | 50 | 0 | 51 |
| Run 2 (Cases 1-4) | 7 | 25 | 19 | 51 |
| Run 3 (Cases 1-6) | 7 | 24 | 20 | 51 |
| Run 4 (Cases 1-7) | 30 | 21 | 0 | 51 |
| Run 5 (Cases 1-8) | 46 | 5 | 0 | 51 |
| Run 6 (Cases 1-9) | 48 | 3 | 0 | 51 |
| Run 7 (Cases 1-9 v2) | 49 | 2 | 0 | 51 |
| Run 8 (Cases 1-9 v3) | TBD | TBD | TBD | 51 |

## Metadata Output

All conversions are recorded in `output/graph/metadata.yaml`:

```yaml
dtype_conversions:
  - function: scaled_mm
    operation: matmul
    orig_dtypes: nvfp4 float4_e2m1fn_x2
    new_dtypes: int8
    reason: nvfp4 not supported on meta device, converted to int8
  - function: _create_scale_factors
    operation: source_dtype_replacement
    orig_dtypes: fp8 float8_e4m3fn
    new_dtypes: int8
    count: 3
    reason: not supported on meta/cpu device, replaced in source code
```

## Meta Device Dtype Support Reference

| Dtype | Tensor Creation | copy_ / .to() | matmul / addmm | indexing |
|---|---|---|---|---|
| float32 | YES | YES | YES | YES |
| float16 | YES | YES | YES | YES |
| bfloat16 | YES | YES | YES | YES |
| int8 | YES | YES | YES (via float cast) | YES |
| float8_e4m3fn | YES | NO | NO | NO |
| float8_e5m2 | YES | NO | NO | NO |
| float4_e2m1fn_x2 | YES | NO | NO | NO |

## Files

- `postprocess.py` — Main postprocessing script
- `QUANT_CONVERSION.md` — This document
- `metadata.yaml` — Per-kernel conversion metadata (generated in output/graph/)
