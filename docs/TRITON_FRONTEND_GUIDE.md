<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Triton Frontend Guide

This guide documents how SOLAR v2 ingests standalone Triton kernels, and generalizes
what that took into a recipe for adding frontends for other kernel languages.

## Overview

SOLAR v2 translates source code into `solar.einsum_lib` IR with an LLM, verifies the
translation numerically, then runs graph extraction, einsum conversion, analysis, and
roofline perf prediction. The PyTorch and JAX frontends obtain their computation graph
by introspecting the framework — `torch.fx` traces a module, JAX gives you a jaxpr.

Triton offers no such handle. A `@triton.jit` kernel is device code built from raw
pointer arithmetic, block masks, and explicit tiling; there is no object graph to walk
and nothing meaningful to trace on the host. The frontend therefore does the simplest
possible thing: hand the LLM the source text, and obtain a golden reference by running
the real kernel on a real GPU.

That constraint turned out to be a feature. `solar/einsum_lib/triton_frontend.py` is
roughly 200 lines and imports neither `torch` nor `triton`. All language-specific code
lives in the example file, which is why the same path extends to other kernel
languages without pipeline changes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Self-referencing frontend flow                       │
└──────────────────────────────────────────────────────────────────────────┘

              example.py  (the only file that imports triton/torch)
              ┌───────────────┬──────────────┬─────────────────┐
              │ get_inputs()  │ kernel_fn()  │ get_source()    │
              └───────┬───────┴──────┬───────┴────────┬────────┘
                      │              │                │
                      ▼              ▼                ▼
              ┌───────────────┐ ┌──────────┐  ┌───────────────┐
              │ shapes+dtypes │ │  GPU     │  │ source text   │
              │ (from arrays) │ │ reference│  │ for the LLM   │
              └───────┬───────┘ └────┬─────┘  └───────┬───────┘
                      │              │                │
                      └──────────────┼────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │   prepare_triton    │
                          └──────────┬──────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ TranslationAgent (LLM)        │
                     │   source ──► einsum_lib IR    │
                     └───────────────┬───────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ verify: IR (CPU, fp64)  vs    │
                     │         kernel_fn (GPU)       │
                     └───────────────┬───────────────┘
                                     ▼
                       graph ─► einsum ─► analysis ─► perf
                          (entirely language-agnostic)
```

Everything below `prepare_triton` is shared with the PyTorch and JAX frontends and
required no modification.

## The contract

A source file is *self-referencing*: it brings its own inputs and its own reference
implementation. Two functions are required, one is optional.

### `get_inputs() -> dict[str, np.ndarray]` (required)

Returns concrete input arrays; seeded random is fine. Shapes and dtypes are **derived
from these arrays**, never declared separately, so there is no second source of truth
to drift. The dict keys become the einsum_lib input names.

### `kernel_fn(inputs: dict) -> dict[str, np.ndarray]` (required)

Moves inputs to CUDA, launches the real kernel, returns outputs as numpy. This is the
only code that touches Triton or CUDA, and it lives in the example file rather than the
pipeline.

Return **every** output the kernel produces, not just the one you care about. SOLAR
derives the modelled work from the IR that reproduces these outputs; LigerRoPE
under-reported its cost by half until `kernel_fn` returned both `q_out` and `k_out`.

### `get_source() -> str` (optional)

Returns the source the LLM should translate. Without it, the adapter file itself is
used. With it, a thin adapter can call `inspect.getsource()` on an installed package so
the LLM sees real library code instead of glue. This is how the Liger-Kernel examples
test genuine production kernels with no vendored copies.

### Minimal example

```python
import numpy as np

_KERNEL = None


def _get_kernel():
    """Import and cache lazily so the pipeline never imports triton itself."""
    global _KERNEL
    if _KERNEL is None:
        import torch, triton, triton.language as tl

        @triton.jit
        def _scale_kernel(X_ptr, Y_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(X_ptr + offs, mask=mask)
            tl.store(Y_ptr + offs, x * 2.0, mask=mask)

        _KERNEL = (torch, triton, _scale_kernel)
    return _KERNEL


_NP_TO_TORCH = {np.float64: "float64", np.float32: "float32", np.float16: "float16"}


def get_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal((4096, 4096)).astype(np.float32)}


def kernel_fn(inputs):
    torch, triton, kern = _get_kernel()
    # Preserve the caller's dtype: fp64 during verification, fp32 when benchmarking.
    dt = getattr(torch, _NP_TO_TORCH[np.dtype(inputs["x"].dtype).type])
    x = torch.from_numpy(np.ascontiguousarray(inputs["x"])).to(device="cuda", dtype=dt)
    y = torch.empty_like(x)
    n = x.numel()
    kern[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return {"output": y.cpu().numpy()}
```

## Running the pipeline

Pass `--source-type triton` to the translation step; the rest of the pipeline is
unchanged. A typical sequence (no Triton example source ships in this release):

```bash
python3 -m solar.cli.toDSL_model "${MODEL_FILE}" \
  --output-dir "${DSL_DIR}" --source-type triton --save-on-fail --cap-input-size

python3 -m solar.einsum_dsl.cli        "${DSL_DIR}"
python3 -m solar.einsum_dsl.cli_einsum "${DSL_DIR}"
python3 -m solar.cli.analyze_model     --einsum-graph-path ... --output-dir ...
python3 -m solar.cli.predict_perf_model --analysis-path ... --arch-config H100_PCIe --precision tf32
```

`--precision` has to name the precision the kernel *executes*, not the one it verifies
at, because it sets both the tensor-core peak and `bytes_per_element`. The `tl.dot`
examples pass `--precision fp16` for that reason.

A hybrid model — a PyTorch `nn.Module` that calls Triton ops internally — should use
`--source-type pytorch` instead. The frontends compose; `LigerLlamaBlock` is an
`nn.Module` calling three real Liger Triton kernels and runs through the PyTorch path
with no special handling.

## Precision handling

This is the bulk of the work in practice, and the part most likely to recur in a new
language.

The `einsum_lib` core stores every float tensor as float64 (see `core._numpy_dtype`),
so the generated IR always computes in fp64. A GPU kernel usually cannot. The governing
principle:

> **One code path. Precision varies only at the host boundary.**

Verification and benchmarking must exercise the *same* kernel, or you are no longer
testing the LLM's ability to translate that kernel. Do not write a numpy or PyTorch
reimplementation to serve as the reference.

Three consequences:

1. **Make `kernel_fn` dtype-polymorphic.** Derive the compute dtype from the incoming
   array dtype rather than hardcoding one. Verification hands you fp64, benchmarking
   hands you fp32; a helper like `_compute_dtype()` maps those onto whatever the kernel
   can actually run.

2. **Respect each op's precision floor.** `tl.dot` has no fp64 path at all and accepts
   only fp16/bf16/fp32. `tl.exp` and `tl.sqrt` reject fp16 outright. Where an op cannot
   meet the ambient precision, upcast on load, compute in fp32, and store back at the
   tensor's dtype — the accumulator stays `tl.float32` while the output follows the
   input.

3. **Force IEEE for verification where the hardware would otherwise cheat.** On H100,
   `tl.dot` in fp32 silently uses TF32 (about 10 mantissa bits), which leaves a ~3e-3
   gap against the fp64 IR and fails verification on numerical grounds alone. Setting
   `TRITON_F32_DEFAULT=ieee` closes it to ~1e-5.

### The caching hazard this creates

`TRITON_F32_DEFAULT` is a compile-time flag that Triton does **not** include in its JIT
cache key. A cache shared between an IEEE verification run and a TF32 benchmark run
will serve IEEE-compiled kernels to the benchmark, silently disabling tensor cores. The
observed cost was 4-7x on `tl.dot` kernels, which looks exactly like a kernel
performance regression and wasted real debugging time.

Use separate cache directories for the two workflows. Any language with a JIT and
environment-driven compile flags deserves the same audit.

## Input capping

`--cap-input-size` reduces every dimension to at most 64 for verification, because the
IR executes on CPU under numpy/numba and full LLM-scale shapes are impractically slow.
The original shapes are still what the LLM sees and what the perf model uses, so SOL
remains representative; only the numeric check is scaled down.

**Capping can change the code path.** Kernels with hardcoded tile sizes will read out
of bounds once the tensor is smaller than a tile — TritonAttention's `BLOCK_M = 128`
against a capped 64-length sequence produced a 2.03 absolute error that presented as a
translation failure. Derive tile sizes from the runtime shape:

```python
N_CTX = Q.shape[2]
BLOCK_M = min(128, N_CTX)
BLOCK_N = min(64, N_CTX)
```

Any tiled-kernel language has this hazard.

## Environment

Reference generation needs a real GPU, and the toolchains write large caches. Redirect
them to scratch before running, as the example scripts do:

```bash
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/nv_compute_cache"
export TMPDIR="${CACHE_ROOT}/tmp"
# plus TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, XDG_CACHE_HOME, HF_HOME, TORCH_HOME
```

Filling the home directory partition with Triton cache is a common first failure.

## Adding a frontend for another language

The contract is deliberately framework-neutral: anything callable from Python that
returns numpy satisfies it. A CUDA C++ kernel behind pybind11, a CUTLASS instantiation,
a TVM module, a Halide pipeline, or a Mojo function would all plug in the same way.

**What you must supply**, per language:

1. Source text for the LLM.
2. Concrete inputs as numpy arrays.
3. A callable that runs the real implementation and returns numpy outputs.

If your language can be reached from Python, you may not need a new frontend module at
all — `prepare_triton` already only assumes those three things, and the name is the
only Triton-specific part of it.

**What you must reason about fresh**, per language:

- *Precision floor.* Can the implementation meet the fp64 IR partway, and which
  primitives refuse which dtypes? Plan the upcast/downcast points before writing
  examples.
- *Capping sensitivity.* Does shrinking shapes alter the code path — tiles, vector
  widths, specialization thresholds?
- *Hardware requirement.* If the reference needs a GPU, verification is no longer a
  pure-CPU step, which affects where the pipeline can run.
- *Compile caching.* Does the toolchain key its cache on all flags that change
  generated code? If not, isolate caches per workflow.
- *Prompt coverage.* The LLM must know which builtins exist in einsum_lib. Bare `exp`,
  `log`, and `sqrt` inside `@kernel` bodies had to be documented explicitly in
  `docs/einsum_lib_prompt.md` before translations stopped failing on them.

**What does not transfer.** The `run_python` inspection tool exists so the LLM can
expand a PyTorch model object's submodules. For a source-text language there is nothing
to introspect — the source is already complete — and left unrestricted the LLM will
instead use it to iteratively debug its own generated IR, burning tool rounds. It is
scoped to read-only inspection with a bounded round count in `agent.py`.

## Known gotchas

- Do not reintroduce a separate reference implementation. An earlier revision let an
  adapter define `reference_fn` — a pure-numpy fp64 reference used in place of the
  kernel — which made verification pass without ever testing the kernel. It has been
  removed; `kernel_fn` is the single source of truth for both verification and
  benchmarking.
- Unfused SOL is translation-sensitive. A re-translation of TritonVAE moved its unfused
  bound from 154.33us to 113.17us by emitting fewer DRAM-round-tripped intermediates,
  while the fused bound held steady. Prefer the fused number for stable comparisons.
- Verification tolerance has to account for the precision floor of the target language,
  not the IR's. An fp32 `tl.dot` will never match fp64 to fp64 tolerances.

## See also

- `solar/einsum_lib/triton_frontend.py` — the frontend itself
- `docs/einsum_lib_prompt.md` — what the LLM is told about the target IR
- `docs/VERIFICATION_GUIDE.md` — the current verification entry points (`solar-translate verify`, `scripts/verify_saved_ir.py`) and the retired kernelbench graph verifier
