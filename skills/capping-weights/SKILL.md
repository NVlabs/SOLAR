---
name: capping-weights
description: Use when a solar verification fails on architectural/weight scale rather than shape bugs — "Estimated N GB needed ... but budget is M GB" with uncappable const weights, verify timeouts on huge-hidden or many-expert MoE problems, or OOM from weight tensors that --cap-input-size cannot shrink (it only caps batch/seq/spatial).
---

# Capping weights (const-axis override)

## Overview

`--cap-input-size` shrinks only FREE dims (batch/seq/spatial) and deliberately
preserves architectural dims tied to weights. Problems whose **const weights
alone** exceed the memory budget (e.g. 3×16384×53248 fp32 ≈ 10.5 GB) or whose
verification runs >20 min can therefore never pass — no shape proposal fixes a
weight floor. The fix: shrink the **const axes themselves** for verification.

**Why it's sound:** the reference `run()` is shape-generic; when adapter
weights and reference inputs shrink *consistently*, the numeric check still
proves the translation implements the same math. Scale-truth caveat below.

## How

```bash
$PY scripts/gen_solexecbench_adapter.py <problem_dir> adapter.py \
    --const-override hidden_size=2048 --const-override intermediate_size=6656
# then translate/verify adapter.py as usual (toDSL_model or manual + verify_saved_ir)
```

Real example: L2/004 (`hidden 16384, intermediate 53248` → 10.5 GB weights,
budget-floor ERROR) verifies fine at 2048/6656 (0.16 GB).

## Choosing capped values

| Constraint | Rule |
|---|---|
| Divisibility | keep `hidden = num_heads × head_dim`; cap heads, keep head_dim |
| Ratios | preserve e.g. `intermediate/hidden` (53248/16384 = 3.25 → 6656/2048) |
| Degeneracy | keep every axis ≥ 2 (1 hides transpose/broadcast bugs); experts ≥ top_k routed |
| Hardcoded splits | if reference reshapes by literal sizes, those axes can't change — check `run()` first |

## Scale-truth caveat (IMPORTANT)

Trace/analysis after a const-override reflects the **capped** model — SOL
FLOPs/latency are NOT the real workload's. To get true-scale analysis after a
capped verification: re-run the trace at the ORIGINAL shapes in **shape-only
mode** (zero-memory placeholders; see `_shape_only_zeros` in
`einsum_lib/core.py`) — the verified DSL is shape-generic, so the full-size
trace costs no RAM. The tool is `scripts/trace_saved_ir.py`:

```bash
$PY scripts/trace_saved_ir.py <dsl>_einsum_lib.py --definition <problem_dir> \
    --out-dir OUT [--fallback-adapter adapter.py] [--custom-py extra.py]
```

Never report capped-run SOL numbers as full-model numbers.

## When NOT to use

- Failure is a shape bug ("Capped shapes crashed the model") → fix capping, not weights.
- Verification passes at true scale within budget → don't cap; fewer moving parts.

**REQUIRED BACKGROUND:** the `verified-translation` skill (pipeline, flags,
verify_saved_ir).
