# verified-translation — reference

Heavy detail for `solar.cli.toDSL_model` and the batch runners. Read SKILL.md
first; come here for exact contracts, flags, and env knobs.

## Per-source input contracts

The `model` argument is a `.py` file whose required symbols depend on
`--source-type`:

| source-type | File must define | Golden reference |
|---|---|---|
| `pytorch` (default) | `Model` (nn.Module) class + `get_inputs()` | the model itself, run on the inputs |
| `raw` | plain function code; pass `--input-shapes '{"x": [2,3,4]}'` (JSON, required) | executing the raw code |
| `jax` | JAX program (jaxbench-style); CONFIG capping supported via `shape_derivable_config_fields` | `jax` execution (uses `jax.eval_shape` for shape-only discovery when cappable) |
| `triton` | Triton kernel module | kernel execution |
| `cuda` | **adapter**: `get_source() -> str` (code the LLM translates), `get_inputs() -> dict[str, np.ndarray]`, `kernel_fn(inputs) -> dict` (golden outputs). Never compiles/launches CUDA itself — `kernel_fn` may run the real kernel (rigorous, needs GPU) or an equivalent torch computation (torch-golden, CPU-only) | whatever `kernel_fn` returns |

## Key flags (full list: `--help`)

| Flag | Default | Notes |
|---|---|---|
| `--max-retries` | 5 | LLM translation attempts |
| `--no-verify` | off | translate only |
| `--cap-input-size` | off | LLM shrinks free dims (batch/seq/spatial) for verification; trace pass keeps FULL shapes so SOL analysis is unaffected |
| `--max-cap-rounds` | 2 | re-propose shapes + fresh reference on failure |
| `--mem-budget GB` | auto (system RAM) | memory guard + capping-LLM target. THE way to bound memory |
| `--verify-timeout s` | none | per-verification-attempt cap; failed attempt → retry |
| `--fp64` / `--fp32` | fp64 upcast for fp32+ models | `--fp32` = native precision; bf16/fp16/fp8 auto-skip upcast |
| `--save-on-fail` | off | keep generated code on failure (debugging) |
| `--debug` | off | print LLM code before each attempt |
| `--no-numba` | off | pure-python verification (slow; use if numba misbehaves) |
| `--llm-model` | `SOLAR_LLM_MODEL` (required) | a model name your `SOLAR_API_BASE_URL` endpoint serves; no default ships |
| `--api-key-file` | — | else `OPENAI_API_KEY`; no implicit key file |

## Post-verification pipeline (per verified model)

```bash
$PY -m solar.einsum_dsl.cli        OUT/dsl --output-dir OUT/graph
$PY -m solar.einsum_dsl.cli_einsum OUT/dsl --output-dir OUT/einsum
$PY -m solar.cli.analyze_model      --einsum-graph-path OUT/einsum/*_einsum_graph.yaml --output-dir OUT/analysis
$PY -m solar.cli.predict_perf_model --analysis-path OUT/analysis/analysis.yaml \
     --output-dir OUT/perf --arch-config H100_PCIe --precision fp8
```
Arch configs: `configs/arch/*.yaml` (A6000, B200, H100_PCIe, H100_SXM, H200_SXM, Jetson_Thor_T5000, L40S).
SOL output: `perf_<arch>.yaml` with unfused / fused / fused_prefetched `runtime_ms`.

## Problem-definition adapters

For a problem directory holding `definition.json`, `reference.py` and
`workload.jsonl`, `scripts/gen_solexecbench_adapter.py <problem_dir> adapter.py`
writes a capped PyTorch adapter: `var` axes come from the selected
`workload.jsonl` line (`--workload-idx`, default 0), `const` axes from the
definition, and `--const-override NAME=VALUE` caps a constant (see the
`capping-weights` skill). Batch campaign runners are not shipped; verify one
problem at a time.

## Verification internals (for debugging)

- Success marker depends on the route. Manual route (`scripts/verify_saved_ir.py`,
  `solar-translate verify`): `VERIFIED  (max_abs_error=…)` (two spaces), mismatch
  `MISMATCH  …`; it never prints `GENERATED CODE (VERIFIED)`. API route
  (`toDSL_model`): `GENERATED CODE (VERIFIED)`; mismatch `GENERATED CODE (UNVERIFIED)`.
- Reference runs at native precision; IR truncates bf16/fp16 at kernel
  boundaries; per-dtype tolerance headroom (fp32: 256×eps, bf16/fp16: 4×eps).
- Numba may "decline" generated code (falls back to interpreter, much slower —
  a cause of timeouts). `--no-numba` forces interpreter.
- DSL builtins available to generated code: `Tensor`, `Dim`, `kernel`,
  `clear_dims`, `apply_mask`, `top_k_mask`, `slice_at`, `stack`, `concat`,
  `gather` (defined in `einsum_lib/core.py`) plus `cos`/`sin` (mapped straight
  to `np.cos`/`np.sin` in `numba_accel.py`). Adding a core builtin requires
  BOTH the `core.py` definition AND registration in
  `numba_accel.execution_namespace()` — a namespace import of a name missing
  from core breaks every verification with `cannot import name '<X>'`.


## einsum_lib DSL — language template

A DSL file defines `@kernel` functions (plain loop nests) + `model()` wiring +
`run_model(inputs) -> dict`. **Complete verified example in `example/`**
(`adapter.py` + `solar_model_einsum_lib.py`, FlashInfer rmsnorm, VERIFIED 3.4e-7).

```python
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask, top_k_mask

@kernel
def linear(X: Tensor("B,S,H", dtype="fp32"),
           W: Tensor("F,H", role="weight", dtype="fp32")) -> Tensor("B,S,F"):
    Y = Tensor(B=X.shape[0], S=X.shape[1], F=W.shape[0], dtype="fp32")  # fill=v optional
    for b in Dim("B", parallel=True):     # prange; scratch Tensors inside are private
        for s in Dim("S"):
            for f in Dim("F"):
                for h in Dim("H"):        # reduction loop
                    Y[b, s, f] += X[b, s, h] * W[f, h]
    return Y

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["hidden_states"], "B,S,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["weight"], "F,H", dtype="fp32")
    return {"output": linear(x, w).data}
```

- **Dims**: `Dim("X")` iterates the registered size (from any tensor declaring X);
  `Dim("X", size=n, parallel=True)` — `size` must be an int literal, module
  constant, `T.shape[i]`, or a local `n = T.shape[i]` bind (never a bare
  runtime parameter). `parallel=True` only on non-colliding outer loops.
- **dtypes**: fp64 fp32 tf32 fp16 bf16 fp8 nvfp4 int8 int32 int64. bf16/fp16 in
  the RETURN annotation auto-truncate the result (assign-form calls only);
  fp8/nvfp4 tags affect byte accounting, not values.
- **Scalar fns in kernel bodies**: exp log sqrt tanh cos sin floor log2 round
  abs max min, plus `** // %` arithmetic. Nothing else (numba must lower it).
- **Builtins (call at model level, not inside loops)**: `apply_mask(t, mask,
  masked_value)`, `top_k_mask(t, k, axis=-1)`, `gather(table, index, axis)`,
  `slice_at`, `stack`, `concat` — zero-compute in analysis.
- **Int gathers inside kernels**: `e = SE[n, k]` then `W[e, i, h]` (int64
  index tensors; keep the writing loop serial if writes can collide).
- Adapter contract (`adapter.py`): see SKILL.md step 2 and `example/adapter.py`.


## PyTorch sources (kernelbench-style) — contracts and capping

`scripts/verify_saved_ir.py <model.py> <dsl.py> --source-type pytorch` — the
model file (defines `Model` + `get_inputs`) is the golden; no adapter.

- **Input naming in `run_model(inputs)`**: activations are `"x"` (single) or
  `"x_0"`, `"x_1"`… (multiple); parameters use dotted names with dots →
  underscores (`network.0.weight` → `network_0_weight`); BN/`named_buffers()`
  tensors (`*_running_mean/var`) arrive as inputs; int module attributes are
  injected as scalar inputs. Single tensor output must be keyed `"output"`.
- **Capping**: kernelbench files have no CONFIG dict, so `--config` does not
  apply — copy the model file (e.g. `model_capped.py`) and edit its globals /
  `get_inputs` sizes; verify against the copy. Watch for architecture-coupled
  sizes (e.g. `fc1 = 256*6*6` pins the 224 input; adaptive pools decouple it).
- **Self-check**: `example/selfcheck.py` assumes an adapter and interpreted
  execution — too slow for convnets. For pytorch, self-check = golden Model on
  capped `get_inputs()` vs `execute_translated(code,
  {k: v.shape for k, v in inputs.items()}, all_inputs=inputs,
  fast=True, use_fp64=True)` from `solar.einsum_lib.verify`, where `inputs`
  is the named NumPy dictionary (`x`, or `x_0`/`x_1`, plus parameters/buffers).
  Unpack its `(outputs, inputs_used)` return value before comparing outputs.
- **Tolerance for deep fp32 references**: reference-side fp32 accumulation
  noise scales with output magnitude — expect ~1e-6 RELATIVE to output peak.
  If default atol fails, first diagnose against an fp64-torch golden (DSL
  should match it to ~1e-11); then set `--atol ≈ 5e-6 × max|output|` and
  record the rationale in NOTES.
- **Dims rebind per kernel call**: the same kernel (generic `conv2d`,
  `linear`) may be called with different sizes for the same dim names —
  the registry is scoped per call.
- Builtin signature: `concat(tensors, along_dim)`.


### Measured-density field lessons (validated on the cuTile campaign)

- **Attention measurements are guarded by default**: `--measure-density`
  keeps recognized probability tensors dense. To opt into zero-skipping
  attention accounting, also pass `--measure-attention-density` (both the
  manual trace and API CLI support it). Softmax underflow or causal zeros
  could otherwise discount P@V work and probability bytes and change the
  convention used in the dense-attention suite. The guard recognizes
  softmax/probability names; for opaque kernel names mark the actual returned
  Tensor with `role="attention_probability"`. It is a naming/role guard,
  not a semantic proof. Explicit `sparsity=` or `--density-override` still
  wins. See the [complete v2 flow](../../docs/SOLAR_V2_FLOW.md) for scope,
  provenance, and runnable density examples.
- **Banded SOL for sliding-window attention**: override the probability
  tensor with the analytic serving-scale band density
  (CAUSAL sliding window: sum_i min(i+1, W) / S^2 — e.g. S=4096, W=1024 →
  0.21875; do not confuse with the symmetric two-sided band `|i-j| < W`,
  density `((2W-1)S - W(W-1))/S^2` for `1 <= W <= S`, used
  for banded bmm-style masks):
  `--density-override "probs#0=0.21875"`. This bills the attn@V matmul and
  probs bytes banded; the QK^T side stays dense (faithful when the kernel
  computes scores with an elementwise guard).
- **Arithmetic-guard kernels hide their sparsity from the hook** (indicator
  expressions like `max(0,min(1,m-k+1))` are not tensors). When the guarded
  operand is mathematically zero anyway (tril/band weights), zero it in
  `run_model` (`np.tril(inputs["a"])`) — numerics identical, sparsity becomes
  tensor-real, measured density then bills the specialized kernel's true
  work/bytes. Re-verify after (it is a DSL edit).
- **Override names must match the TRACE's tensor names** — inputs are often
  renamed `__const_N` in the yaml; a non-matching override is skipped with a
  WARNING listing available names (check stderr).
- Structurally-compact formulations (slot-dense MoE, compact masked/ragged
  outputs) need no density at all — prefer them when possible.

## Work-minimal translation & static masks

The loop nest you write is the ground truth Solar counts and prices
(`loop_iterations` in the einsum graph). Translate in the most efficient form
that is still functionally correct; verify numerics after every efficiency
edit, and if an efficient form cannot be made to match, keep the plain form
that does — correctness wins.

### Work-minimal translation

- **Loop what the source computes.** A fused projection whose rows are split
  (`chunk`/slices) is one loop per half over `size=W.shape[0] // 2`, not two
  full-axis loops and not one full-axis loop whose half is discarded.
  Never loop a whole axis and throw part of the result away.
- **Resolvable extents.** `Dim("I", size=<int arithmetic over .shape of the
  kernel's own inputs and integer constants>)`. The converter evaluates it at
  each call site; an extent it cannot resolve is refused or mis-counted
  (depending on the checkout). Do not rely on index-based inference: a bare
  loop index such as `W[i, h]` pins the loop to the WHOLE `W` axis whenever
  the extent is not resolvable — the L1/029 half-range projection was billed
  2x its gate half that way (`tests/test_work_minimal_translation.py`).
- **Compute once.** Hoist loop-invariant work, reuse shared intermediates,
  read a single copy instead of materializing repeats; add no pass the source
  does not perform. Fusing an elementwise epilogue (bias, activation, mask
  multiply) into the producing nest is fine on the manual path.
- "Do not optimize" (API prompts) forbids tricks that leave the work unchanged
  — unrolling, constant folding, tiling. It does not ask for a naive nest.

### Static masks

A mask is static when shapes and constants in the source fully determine it:
causal/triangular, sliding window or band, block-diagonal, fixed padding or
valid length, strides/dilations, Triton boundary masks (`offs < N`). Use the
most efficient form that matches the numerics, in this order:

1. **Compact loops** over the live range when it is a loop range (a window of
   W keys loops `Dim("W", size=W)`; a padded tail is left out of the extent).
   Exact at any scale; no density knob needed.
2. **Mask tensor with analytic density** when the live region is not a
   rectangular range (causal triangle, irregular band): build the 0/1 mask in
   `run_model` from shapes/constants and set
   `Tensor.from_numpy(mask, "S,T", sparsity=<analytic density at the ORIGINAL
   shape>)` and multiply by it INSIDE the producing kernel (`acc * M[s, t]`).
   The analyzer then makes zeros free in that kernel (MACs scale by the min
   density of the multiplicative inputs; DRAM bytes by each tensor's
   density). `apply_mask(…, masked_value=0.0)` on a kernel's OUTPUT is not an
   equivalent: it does not discount the producing kernel, it only gives the
   mask result (and what consumes it) the mask's density (pinned in
   `tests/test_work_minimal_translation.py`). Closed forms at the ORIGINAL S:
   square causal `(S+1)/(2S)`; causal window W `sum_i min(i+1, W)/S^2`;
   symmetric band `|s-t| < W` on an S×S mask, valid for `1 <= W <= S`:
   `((2W-1)S - W(W-1))/S^2` (1 for `W >= S`; the untruncated `(2W-1)/S`
   over-counts the edges: S=4, W=2 is 0.625, not 0.75). An explicit
   `sparsity=` beats any measurement (trust-ladder rung "override"), and it is
   the right source here: a capped-shape measurement of a scale-coupled mask
   differs from the original-scale density.
3. **Never an arithmetic guard in the kernel body** (`max(0, min(1, s - t))`):
   its sparsity is invisible to the analyzer and the work is billed dense
   (pinned in `tests/test_work_minimal_translation.py`). Hoist it into the
   mask tensor of (2).

For attention, the pre-softmax mask zeroes the PROBABILITIES: put the analytic
density on the probability tensor (`role="attention_probability"` with
`sparsity=`, or `--density-override`) as described in "Measured-density field
lessons"; the default guard keeps recognized probability tensors dense unless
you do. This changes attention SOL relative to the dense-attention convention,
so apply it deliberately and note the mask, its formula and the density in the
run notes. A **data-dependent** mask — a runtime attention/padding mask
input, a learned or routed selection — is not static: never claim sparsity for
it; keep the dense form and use the measured-density ladder below.

## Manual-path DSL authoring rules (numerics)

- **Follow the code, not the name** — references named dropout/grouped/moe often
  contain none of it; docstrings lie about dims.
- **Dtype-agnostic goldens**: verify inputs arrive fp64. No `dtype=` kwargs,
  `zeros(..., dtype=x.dtype)`; wrap exec'd `run()` with
  `torch.set_default_dtype` only if internal `linspace`-style ops need it.
- **Masks are arithmetic**: causal/ragged guards via `max(0, min(1, expr))`;
  fill `-inf`/`-1e30` so `exp` underflows to exact 0. Softmax = rowmax /
  exp-sum / normalize trio.
- **Single-copy dispatch, never materialize repeats**: GQA reads
  `K[b, t, (n//G)*D + d]` (or `h % HK` for tile order); MoE is SLOT-DENSE —
  `e = SE[n,k]` int gather into one expert weight table (derive SE from
  `top_k_mask` by rank-counting, ascending expert order keeps the fp
  accumulation identical to dense).
- **Sequential recurrences are sequential loops** (`for c in Dim("C")` carrying
  a state tensor); per-iteration scratch tensors are private per prange.
- **QDQ (fp8/nvfp4) grid-snap idiom** (bit-exact vs torch casts):
  `e = floor(log2(max(abs(x),1e-300)))`, `step = 2.0**(e - mant)`,
  `q = round(x/step)*step`, clamp. fp32 pre-snap mant=23 (torch's double
  rounding); e4m3 mant=3, e≥-6, ±448; e2m1 mant=1, e≥0, ±6; bf16 mant=7.
  Snap after every inexact fp32 op in the reference's scale chain.
- **Scalar precision contract** (DSL-SIGMOID-FP32-BOUNDARY-1). Two execution
  modes, described separately:
  * **fp64 emulation** — the interpreter and the `use_fp64=True` Numba
    lowering: floating tensors are float64 (`core._numpy_dtype`), scalar
    math runs in double, and the framework rounds ONLY `bf16`/`fp16` results
    (`core._truncate_to_dtype`); an `fp32` tag is a storage-width label with
    no compute effect. **This is the accepted validation precision**
    (Jenny, 2026-10-26): emulation may be in fp64; an NVFP4 candidate is
    validated in fp64 after dequantization and then carries its declared
    `nvfp4` types into the SOL/perf projection. fp32 bit-fidelity of the
    emulation is not a release requirement.
  * **native fp32-carrier lowering** — `use_fp64=False` (the
    `execute_translated` default fast path): float32 buffers, and compiled
    scalar ops follow their operand types with operand-driven promotion
    (`(X[0]+X[1])-X[0]` with `X = [2**24, 1]` gives 1 in fp64 emulation and
    0 here; `exp(1) - e` gives 0 and -8.25e-08). It is a separate mode, not
    the fp64 contract; compare it against the actual runtime it stands for.
  The grid snap after **every** inexact op of a reference fp32 chain is
  explicit **rounding-boundary emulation**, not a bit-exactness guarantee:
  it reproduces where torch's fp32 chain rounds (observed exact agreement
  with torch's fp32 `sigmoid` on logits [10, 20] at 1e-3, both modes), but
  torch's transcendental implementations are not correctly rounded
  everywhere (on float32-rounded inputs over [-20, 10], 179 of 30,001 points
  in the fp64 paths — interpreter and `use_fp64=True` lowering — and 183 in
  the native fp32-carrier lowering differ from a literal torch fp32 step
  chain by up to 3 ulps), so any other
  domain must be compared against the actual reference/runtime, never
  assumed. A single final snap is NOT the idiom: `sigmoid(17)` gives
  `0.9999999403953552` where torch's fp32 `1/(1+exp(-17))` rounds
  `1+exp(-17)` to 1.0 and returns `1.0`; the single-snap form first
  saturates at 17.329 against 16.636 for the per-op form and torch, one ulp
  apart on ~40% of [10, 20] — enough to move top-k tie membership.
  `tests/test_dsl_scalar_precision_contract.py` pins the two modes and the
  idiom; its "up to ten differing torch points" allowance is a
  host-dependence guard for vectorised `expf`, not an exact-output gate, and
  waives nothing in an earlier repair.
- **Broadcast/scatter outputs**: if the reference returns `expand()` views or
  does `clone(); index_add_`, output COMPACT tensors and have the adapter
  slice the golden (assert stride 0 / untouched-rows-equal first).
- **Numba gotchas**: never alias `Tensor.from_numpy`; int64 gathers with
  colliding writes stay serial; bf16 truncation fires only on
  `var = kernel(...)` assigns with the dtype in the RETURN annotation.

## Manual-path DSL authoring rules (op counting / SOL)

The converter counts work from the kernel's `Dim()` loop nest — keep it
resolvable and honest:

- Loop dims must size statically: registered names, `size=<int|module const>`,
  `size=X.shape[i]`, or a local `n = X.shape[i]` bind. NEVER `size=<runtime
  param>` alone.
- No flat derived indices coexisting with their factors in contraction kernels
  (`W[h, n*D+d]` → ship the weight `[H,N,D]`); block scales index with
  derived `k//B` (fine — derived indices don't multiply into counts).
- One accumulation chain per kernel where practical; a body with two `+=`
  statements counts 2x per trip (intended for fused gate+up).
- Norms as sum/apply trios, not inline per-element mean/var recomputation
  (inline-stat kernels get reclassified off the tensor cores).
- Weight tensors: `role="weight"`; quantized storage widths via dtype tags
  (`fp8`, `nvfp4`) for byte accounting.

## Tolerances (official verify)

| Kernel class | Flags | Rationale |
|---|---|---|
| default | (none) rtol 1e-4 / atol 1e-5 | fp64-vs-fp32-golden noise |
| quant QDQ (fp8/nvfp4) | `--rtol 0.02 --atol 0.005` | benchmark's own tolerance; bf16-output tie flips are irreducible |
| bf16-output attention | `--rtol 0.01 --atol 2e-3..5e-3` | one bf16 ulp; the fp32 lse output pins the logic at ~1e-6 |

## SOL comparison & reconciliation tooling

Collection and join tooling is not shipped: re-run the graph, analysis and
perf stages per verified DSL (per-category precision) and join the perf YAMLs
against your reference on the workload-line-0 identity (a measured reference
usually has a timer floor, 0.4 µs in ours). Known reference-side
conventions when reconciling: nvfp4 billed at 2x the B200 rate, weights-resident
compute rooflines for decoder blocks, unfused-at-bf16-bytes for some quant
kernels, producer-outputs-as-inputs not billed.

## Sparse masks & measured density (zeros-are-free SOL)

Two ways to make sparse workloads cost what they actually compute:

- **Slot-dense restructure (preferred for MoE)**: loop tokens × topk with an
  `e = SE[n,k]` gather — the loop nest IS the activated work, no modeling
  knob needed (structural, exact at any scale).
- **Mask + measured density**: for data-dependent sparsity you cannot
  restructure away (banded sliding-window attention, token dropping,
  threshold activations), keep the dense-form DSL with a multiplicative mask
  (`top_k_mask` / `apply_mask(…, 0.0)` / a 0-1 mask input) and trace with
  `scripts/trace_saved_ir.py … --measure-density`: a full-value instrumented
  run at the adapter's capped shapes measures EXACT per-tensor densities
  (`count_nonzero/size`) and injects them into the original-shape yaml. The
  analyzer then (a) discounts DRAM traffic by each tensor's density and
  (b) makes zero elements FREE in compute: a kernel's MACs/ops scale by the
  min density of inputs that participate MULTIPLICATIVELY in its body
  (additive/guard uses like `-1e9 * mask` do not discount). Per-op
  `compute_density` is recorded in analysis.yaml for audit.
- **Density trust ladder** (how capped-shape measurements reach the
  original-shape yaml — each harvested tensor lands on exactly one rung):
  1. **Analytic** — densities the original-shape trace computes itself
     (`top_k_mask` sets k/n from the ORIGINAL axis extent) always win; a
     conflicting capped measurement is discarded with a note.
  2. **Shape-invariant** — the tensor's shape is identical at harvest and
     original scale → the measurement transfers exactly; injected as
     `sparsity_source: measured`.
  3. **Auto-revert (non-cap revalidation)** — shapes differ (band W/S,
     ratio axes): the tool estimates the fp64 footprint of a full-value run
     at ORIGINAL shapes (3x safety) and, if it fits `--mem-budget` (default
     30 GB), reruns the instrumented trace at original scale with real
     random values → injects the ORIGINAL-scale density
     (`sparsity_source: revalidated_original`).
  4. **Refuse-dense** — the original-scale run doesn't fit (or fails/times
     out): the density is NOT injected (SOL stays dense — a conservative
     upper bound; zeros-are-free only on evidence) and a loud block prints
     the capped measurement, both shapes, and the exact remedy:
     `--density-override NAME=VALUE` (override always wins, any rung).
  A summary line reports the rung counts: `N analytic / N shape-invariant /
  N revalidated / N REFUSED / N overridden`.
- **Patterns with no rung** (value- AND scale-dependent, original run too
  big): compute the density analytically from the definition when possible
  (symmetric two-sided band: `(2W-1)/S`; CAUSAL sliding window:
  `sum_i min(i+1, W) / S^2` — see the field lessons above), or measure
  at 2-3 capped sizes and check the trend
  before overriding — never guess; dense is the honest fallback.
- **API pipeline**: `toDSL_model --measure-density` instruments generated
  translations the same way; with `--cap-input-size` densities carry
  `sparsity_source: measured_capped` (capped-scale — apply the ladder via
  trace_saved_ir or override before trusting scale-coupled ones).
- Tests: `tests/test_sparse_cost.py` (hook exactness, MoE k/E MAC discount,
  SWA band discount, dense regression, and all four ladder rungs).
