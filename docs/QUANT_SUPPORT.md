<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quantized Kernel Support in Solar

Solar uses `torch.device('meta')` for shape-only graph extraction via TorchView.
Neither `meta` nor `cpu` devices support NVFP4 or FP8 operations.
The postprocessing script (`solar/solar/benchmark/solbenchv2/postprocess.py`)
rewrites benchmark source code so Solar can extract computation graphs from
quantized kernels without altering the mathematical structure.

## Meta Device Dtype Support

| Dtype | Tensor Creation | `.to()` / `copy_()` | matmul / addmm | Conclusion |
|---|---|---|---|---|
| `float4_e2m1fn_x2` (NVFP4) | Yes | Not implemented | Not implemented | **Not supported** |
| `float8_e4m3fn` (FP8) | Yes | Partial | Not implemented | **Not reliably supported** |
| `float8_e5m2` (FP8) | Yes | Partial | Not implemented | **Not reliably supported** |
| `int8` | Yes | Yes | Yes (via float32 cast) | **Fully supported** |
| `float32` / `float16` / `bfloat16` | Yes | Yes | Yes | **Fully supported** |

## What the Postprocessor Does

The postprocessor applies **source-level rewrites** to benchmark `.py` files.
It does **not** change the model architecture, layer count, or tensor shapes.
It only makes the code runnable on `meta`/`cpu` so TorchView can trace the graph.

### Case 1 — Remove Device Specifications

Strips `device="cuda"`, `.cuda()`, `torch.set_default_device("cuda")`, and
`torch.cuda.synchronize()`. Hoists `torch.set_default_dtype()` from `main()`
to module level when found.

### Case 2 — Replace Triton `_fused_fma`

Replaces the `@triton.jit` kernel `_fused_fma_kernel` and its Python wrapper
with a pure PyTorch equivalent: `y.add_(x * s)`. Removes Triton imports if
no longer needed.

### Case 3 — Replace `torch._scaled_mm`

Replaces `torch._scaled_mm(a, b, scale_a, scale_b, ...)` with a wrapper
`_scaled_mm_to_matmul(a, b, ...)` that:
- Converts `int8` inputs to `float32` before `torch.matmul`.
- Accepts `**kwargs` for compatibility with `use_fast_accum` etc.
- Applies bias and `out_dtype` if provided.

### Case 4 — Replace Quantized Dtypes with `int8`

All NVFP4 and FP8 dtype references are replaced at source level:

| Original | Replacement |
|---|---|
| `torch.float4_e2m1fn_x2` | `torch.int8` |
| `torch.float8_e4m3fn` | `torch.int8` |
| `torch.float8_e5m2` | `torch.int8` |
| `torch.float8_e4m3fnuz` | `torch.int8` |
| `torch.float8_e5m2fnuz` | `torch.int8` |

This covers `dtype=`, `.to()`, `.view()`, tensor creation, and bare references.

### Case 5 — Fix `nn.Parameter` with `int8` Tensors

Injects a runtime monkey-patch that replaces `nn.Parameter` with
`_safe_nn_Parameter`. The wrapper converts `int8` data to `float32` and sets
`requires_grad=False` (int8 tensors cannot require gradients). Also patches
`register_buffer` to convert int8 buffers.

### Case 6 — Track Quantize Functions

Identifies functions like `quantize_to_fp4` and `quantize_weights` for
metadata tracking. Their output dtypes are already converted by Case 4.

### Case 7 — Replace Blockwise GEMM Loop

Replaces `self.<name>.scaled_mm(...)` calls (tile-by-tile for-loops that fail
on meta device) with `_simple_blockwise_scaled_mm` — a direct
`torch.matmul(a, b.transpose(-2, -1))`.

### Case 8 — Fix `forward()` Signature Mismatch

When `forward(self, x, scale_a, scale_b)` has more args than `get_inputs()`
returns, makes the extra args optional (`=None`). Injects an early-return
guard that performs `F.linear(x, self.weight)` when scale args are `None`.

### Case 9 — Auto-call `quantize_weights()`

Adds `self.quantize_weights()` at the end of `ReferenceModel.__init__()` when
the method exists but is never called. This fills buffers that are registered
as `None` in `__init__`.

### Case 10 — Fix PEP 604 Union Syntax

Replaces `X | None` with `Optional[X]` for Python 3.8/3.9 compatibility.
Ensures `from typing import Optional` is present.

## Does Postprocessing Change Functionality?

**No.** The rewrites preserve the mathematical structure of each kernel:

| Aspect | Preserved? | Detail |
|---|---|---|
| Layer count | Yes | No layers added or removed |
| Tensor shapes | Yes | `int8` has same shape as original dtype |
| Computation graph topology | Yes | Same ops, same connections |
| Einsum equations | Yes | Same rank structure |
| MACs / FLOPs | Yes | Same dimension products |
| Memory element counts | Yes | Same tensor element counts (dtype width is separate) |

What **is** lost (by design):
- Actual quantized arithmetic precision (FP4/FP8 → int8 → float32 for matmul).
- Block-wise tiling structure (replaced with single matmul).
- Scale factor arithmetic (scales ignored in replacement wrappers).

These are acceptable because Solar analyzes **shape-level computation graphs**,
not numerical precision. The postprocessor metadata (`metadata.yaml`) records
all dtype conversions so downstream tools can account for the original precision.

## Output Metadata

When dtype conversions occur, `metadata.yaml` is written alongside the
processed file:

```yaml
dtype_conversions:
- function: forward
  operation: source_dtype_replacement
  orig_dtypes: fp8 float8_e4m3fn
  new_dtypes: int8
  count: 4
  reason: not supported on meta/cpu device, replaced in source code
```

## Running Quant Postprocessing

```bash
# Full pipeline: generate + postprocess + solar analysis (1 workload per kernel)
cd solar
bash scripts/run_solbenchv3_quant_rewrite.sh

# With more workloads
bash scripts/run_solbenchv3_quant_rewrite.sh --max-workloads 3

# Specific kernel
bash scripts/run_solbenchv3_quant_rewrite.sh --kernel nvfp4_cross_attention

# Postprocess only (no solar pipeline)
bash scripts/run_solbenchv3.sh --level Quant --postprocess-only --max-workloads 1

# Debug mode
bash scripts/run_solbenchv3_quant_rewrite.sh --debug
```

## References

- Postprocessor source: `solar/solar/benchmark/solbenchv2/postprocess.py`
- Detailed case-by-case doc: `solar/solar/benchmark/solbenchv2/QUANT_CONVERSION.md`
- Runner script: `solar/scripts/run_solbenchv3_quant_rewrite.sh`
- Main solbench v3 runner: `solar/scripts/run_solbenchv3.sh`
