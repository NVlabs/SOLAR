---
name: verified-translation
description: Use when translating a model/kernel (PyTorch, JAX, Triton, CUDA/CuTeDSL, raw) into solar's einsum_lib DSL, verifying a translation numerically, generating einsum graphs or SOL predictions from one, running benchmark campaigns (kernelbench, SOL-ExecBench, JAXBench), or debugging UNVERIFIED results, OOM/timeout failures, or SOL-vs-reference gaps.
---

# Verified Translation (solar)

## Mode selection (read this first)

**Default — manual path: YOU are the translator.** No API calls. You read the
source, author the DSL, and a deterministic verifier checks it numerically.
Proven on 235/235 SOL-ExecBench kernels; everything below is reproducible from
the repo alone.

**Opt-in — LLM API pipeline** (`solar.cli.toDSL_model`): use ONLY when BOTH
hold: (1) the user explicitly asked for the API/LLM pipeline, and (2) an API
key is available (`OPENAI_API_KEY` or an explicit `--api-key-file`) and an
endpoint is configured (`SOLAR_API_BASE_URL`). Never fall back to the API
silently; never hunt for keys.

`$PY` below = a python >=3.10 with solar installed — see the Environment
section of `AGENTS.md` for this machine's interpreter and env gotchas.
Run everything from the repository root.

The [v1/v2 flow index](../../docs/SOLAR_FLOW.md) separates the implementations.
The [v2 guide](../../docs/SOLAR_V2_FLOW.md) includes a complete runnable smoke
example and the `--measure-density` / `--measure-attention-density` policy.

## Manual path — one problem (reproducible recipe)

Benchmark problems provide `definition.json`, `reference.py`, `workload.jsonl`.
Work in an output directory of your own (`M` below, for example
`out/<problem>/manual/`; the scripts take explicit paths).

**PyTorch sources (kernelbench etc.)** need NO adapter: the model `.py`
(defines `Model` + `get_inputs`) is passed directly as the `model` argument
with `--source-type pytorch`; the golden is the Model itself. Skip step 2,
write the DSL against `Model.forward`'s math, and self-check by running the
Model on `get_inputs()`.

**dtype tags**: author verification DSLs with `fp32` tags (matching the
example) unless the reference itself computes in bf16/fp16 (then the RETURN
annotation tag must reproduce its truncation). Storage-width tags
(`fp8`/`nvfp4` on weights) matter only for SOL byte accounting — set them for
step 6, they do not change values.

1. **Read** `reference.py` end to end (names and docstrings lie — follow the
   code) and `definition.json` axes + the first `workload.jsonl` line.
2. **Write `$M/adapter.py`** — three functions:
   - `get_source()` → the reference source VERBATIM (env shims like
     `enum.StrEnum` go in the exec namespace, never edit the source string);
   - `get_inputs()` → dict of capped-shape torch tensors. Default var axes to
     the FIRST workload.jsonl line's values, capping only what is too large;
     respect every assert (block-128 divisibility, experts ≥ 2·topk, exact
     factoring for reshape/split); keep hardcoded reference dims uncapped —
     see the `capping-weights` skill for const-override strategy. Inputs must
     be non-degenerate: after capping, if any reference output comes out all
     zeros the verifier prints a WARNING and the pass is vacuous — fix the
     inputs (or confirm from the source that zero is genuinely expected);
   - `kernel_fn(inputs)` → exec the source, call the entry point, return
     `{name: np.ndarray}`. Must be dtype-agnostic (verify sends fp64): no
     `dtype=` kwargs; clone golden-side if the reference mutates in place.
3. **Write `$M/solar_model_einsum_lib.py`** — start from the verified template
   in `example/` (adapter + DSL pair) and the language cheat-sheet + authoring
   rules in reference.md (arithmetic masks, single-copy GQA/slot-dense MoE
   dispatch, QDQ grid-snap, resolvable loop dims — the rules cover BOTH
   numerics and correct op counting). Write the most efficient translation
   that is still functionally correct — the loop nest is what Solar bills:
   - **Work-minimal translation**: loop each `Dim` over the extent the source
     really computes (a half-range projection loops the half, not the whole
     axis), size sub-ranges with a resolvable `Dim(size=<int arithmetic over
     .shape>)`, compute each value once, never materialize repeats. "Do not
     optimize" means no tricks that leave the work unchanged, not a naive
     translation. Correctness wins: if an efficient form cannot match the
     source numerics, use the plain form that does.
   - **Static masks** (fully determined by shapes/constants in the source:
     causal, window/band, block-diagonal, fixed padding/valid length): use
     compact loops over the live range where it is a loop range; otherwise a
     real 0/1 mask tensor built in `run_model` with its ANALYTIC original-scale
     `sparsity=`, multiplied into the kernel — never an arithmetic guard inside
     the kernel body (billed dense). Data-dependent masks stay dense. Details
     and density formulas: reference.md "Work-minimal translation & static
     masks".
4. **Self-check** before official verify: run `example/selfcheck.py <M>` (a
   ready template — execs the DSL in a verify-style namespace, NEVER imports
   it, feeds fp64 inputs, compares against `kernel_fn`). Judge the error
   RELATIVE to output magnitude: a fp32 golden gives ~1e-7 relative, so large
   outputs legitimately show ~1e-5 absolute. Output dicts are compared BY KEY —
   `run_model` and `kernel_fn` must return the same key names (use the
   definition's output names).
5. **Official verification** (deterministic, no LLM; run ONE at a time):
   ```bash
   $PY scripts/verify_saved_ir.py $M/adapter.py $M/solar_model_einsum_lib.py \
       --source-type cuda --mem-budget 30 [tolerance flags]
   # pytorch sources: pass the (capped copy of the) model .py instead of an
   # adapter, with --source-type pytorch — contracts in reference.md
   ```
   Tolerances by class (table in reference.md): default rtol 1e-4/atol 1e-5;
   quant QDQ `--rtol 0.02 --atol 0.005`; bf16-output attention `--rtol 0.01`.
   If the guard fork hangs (minutes wall-clock, seconds CPU): drop
   `--mem-budget`. Success line: `VERIFIED (max_abs_error=…)`.

   What exit 0 does NOT prove (both `verify_saved_ir.py` and `solar-translate verify`):
   - Exit codes: 0 verified, 1 mismatch or IR execution failure, 2 usage error,
     3 no numerical proof under `--strict` (shape-only run or vacuous pass), or a
     requested receipt was refused (exit 3 whether or not `--strict` was given).
   - `--shape-only` executes for shapes only and compares nothing; it still exits 0
     unless `--strict` is given. Never report it as verification.
   - `WARNING: reference output '…' is ALL ZEROS` is printed to stdout only. The
     run still passes (exit 0 without `--strict`) and an all-zero translation
     would pass too, so check capped inputs and masks before trusting the result.
   - No receipt is written by default. `--emit-receipt PATH` (with `--definition DIR`
     and `--dtype`) writes the numerical-validation receipt the strict SOL gate
     consumes; a refused receipt exits 3. Use `--strict` for anything that feeds SOL.
6. **Einsum graph + SOL at original shapes**:
   ```bash
   $PY scripts/trace_saved_ir.py $M/solar_model_einsum_lib.py \
       --definition <bench>/<cat>/<p> --out-dir $M/dsl_full \
       --fallback-adapter $M/adapter.py [--custom-py $M/full_inputs.py]
   $PY -m solar.einsum_dsl.cli_einsum $M/dsl_full --output-dir $M/einsum_full
   $PY -m solar.cli.analyze_model --einsum-graph-path $M/einsum_full/*_einsum_graph.yaml \
       --output-dir $M/analysis_full --no-copy-graph
   $PY -m solar.cli.predict_perf_model --analysis-path $M/analysis_full/analysis.yaml \
       --output-dir $M/perf_full --arch-config B200 --precision fp16|fp8|nvfp4
   ```
   Write `full_inputs.py` (`full_inputs(axes) -> {name: np.ndarray}`, fixed
   seeds, zero-stride placeholders via `as_strided`) when inputs are
   value-dependent (indptr/page tables/grids) or pre-decomposed layouts.
7. **Record** a `$M/NOTES` file: entry point, caps, self-check error, verify
   command + tolerance rationale, any reference quirks found.

## Manual path — campaign mode (many problems)

- **Author in parallel, verify sequentially.** If your platform supports
  subagents/parallel workers, fan them out to write
  adapters+DSLs+self-checks (give each the recipe above + reference.md rules
  + a VERIFIED problem as template); official verifications run one at a
  time: one `scripts/verify_saved_ir.py` process per output tree, fed from a
  queue file your own driver consumes.
- Never run two heavy processes against the same output tree; never use
  `ulimit -v` (use `--mem-budget`).
- **SOL sweep + reference comparison** after verification: re-run the
  graph, analysis and perf stages per verified DSL (reference.md) with the
  precision each category needs, then join the perf YAMLs against your
  reference. Known reference-side conventions when reconciling gaps are
  listed in reference.md.

## Opt-in API pipeline (only if user explicitly requested + key provided)

```bash
$PY -m solar.cli.toDSL_model MODEL.py --output-dir OUT/dsl \
    --source-type pytorch|raw|jax|triton|cuda --cap-input-size \
    --mem-budget 10 --verify-timeout 600 --save-on-fail --debug
```
Batch runs: isolate each problem in its own process (`setsid -w timeout …`),
bound memory with `--mem-budget` (never `ulimit -v`), skip existing results,
and run a single instance per output directory.
On persistent API-path failures (cap-round exhaustion, converter declines),
switch to the manual path rather than retrying indefinitely.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `API key required` | You are in API mode without explicit user opt-in → use the manual path. If the user DID opt in: `SOLAR_API_BASE_URL` plus `OPENAI_API_KEY` or `--api-key-file` |
| `OOM(rc=137)` / `Cannot allocate memory` | Real RAM exhaustion → lower `--mem-budget`; check for a racing second run. Never `ulimit -v` (torch VA reservations break) |
| Verify hangs: minutes wall, ~0 CPU | fork mem-guard deadlock → rerun without `--mem-budget` |
| `TIMEOUT(rc=124)` | raise `STEP_TIMEOUT`/`VERIFY_TIMEOUT` (batch) or `--timeout` |
| `Numba declined the generated code` | Manual route: not an error. It prints `[WARN] Numba declined …` and falls back to the interpreter (~100x slower, a common cause of timeouts). API route (fast mode) fails fast with this message. Either way rewrite per the message: supported builtins only, no inner imports, no from_numpy aliasing |
| MISMATCH ~1e-1 | wrong algorithm variant (RoPE flavor, GQA repeat order, raw-vs-cumsum gates) → re-read the reference |
| MISMATCH ~1e-3..1e-2 on quant/bf16 kernels | irreducible output-grid noise → use the class tolerance (reference.md) |
| `Estimated N GB … budget is M GB` at minimal shapes | const weights exceed budget → `capping-weights` skill (`--const-override`) |
| Trace fails: dim-count mismatch / `read-only` | DSL wants pre-decomposed inputs → `--custom-py full_inputs.py`; placeholders must be writable zero-stride |
| `cannot import name 'X' from …core` | builtin wired in `numba_accel.execution_namespace()` but missing in `core.py` — implement both (mirror `apply_mask`) |
| `CONDA_PATH_BACKUP: unbound variable` | `set -u` + conda env.sh → use `set -o pipefail` only |
