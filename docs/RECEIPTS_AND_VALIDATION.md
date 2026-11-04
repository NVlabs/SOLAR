<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Receipts and validation: what the outputs mean

SOLAR keeps four verdicts apart. A result is only as strong as the weakest
verdict you need for the claim you are making.

| Verdict | Where it comes from | Pass means | Pass does not mean |
|---|---|---|---|
| **Numerical** | `verify`, `estimate` modes v1/v2, receipt `verdicts.numerical` | On the frontend-prepared inputs, every named output of the DSL satisfied `np.allclose` against the reference at the effective tolerance | Agreement at other shapes or data; that the reference itself is right; that the DSL does the same work |
| **Structural** | Original-shape trace + DSL compiler (`dsl_full/*.yaml`, `einsum_full/*.yaml`) | Every tensor has original dimensions and every kernel loop extent resolved (`total.unresolved_ops` empty) | That loop bodies model what a kernel executes (masks, guards, gathers need review) |
| **Accounting** | `analysis.yaml` | MACs, other ops, element counts and densities follow the as-written convention over the resolved loop nests | Real memory traffic, cache behaviour, or that zeros were skipped on hardware |
| **Hardware** | `perf_<arch>.yaml` | A whole-graph roofline under one arch config, precision, fusion and boundary-billing convention | Measured latency, implementation efficiency, or a feasible schedule |

Receipts store `structural`, `accounting` and `hardware` as `not_certified`
by construction. Nothing in the current tooling upgrades them automatically.

## Evidence classes: measured, native, emulated, mocked, structural

Keep these apart when you report a result:

- **Measured / native**: the reference ran the real implementation (a PyTorch
  model on `get_inputs()`, a Triton kernel on a GPU) and the DSL was compared
  against its outputs. This is the only class that is a numerical verdict.
- **Emulated reference**: a `cuda` adapter whose `kernel_fn` computes the
  kernel's math with PyTorch on the CPU (the shipped RMSNorm example does
  this). The comparison is still numerical, but against an emulation whose
  faithfulness to the kernel is the adapter author's claim, recorded in
  `execution.source_type` and the adapter's `NOTES`.
- **Mocked**: the test suite exercises the Pythonic route with a mocked
  inference client (`tests/release_agentic/`, `tests/test_translation_agent.py`).
  Those tests check control flow and verdict handling; they verify no
  translation and establish nothing numerically.
- **Structural**: `--shape-only` runs, the original-shape trace, the DSL
  compiler and `analysis.yaml`. They never compare values.

## The verifier line and exit status

`VERIFIED  (max_abs_error=3.411e-07, reference 0.1s, IR 1.5s)`

- `VERIFIED` / `MISMATCH` / `IR execution FAILED`: the `np.allclose(dsl,
  ref, rtol, atol)` result over all named outputs after both sides are cast
  to fp64, with key-set and shape equality required. `rtol`/`atol` are the
  caller's values (defaults `1e-4`/`1e-5`) raised to the frontend's dtype
  floor when one applies; see the
  [numerical contract](SOLAR_V2_CONTRACTS.md#numerical-check-what-the-code-actually-does).
- `max_abs_error`: informational. The pass criterion scales with the
  reference magnitude, so a small `max_abs_error` on small outputs is not
  stronger evidence than a larger one on large outputs.
- `[VACUOUS: ...]` appended to the line names outputs whose **reference was
  all zeros**. An all-zero DSL output would have passed on them, so the pass
  is weak evidence. By default (legacy, unchanged) such a run still exits 0;
  the result records `vacuous_outputs` and `status: pass_vacuous`.
- `SHAPE-ONLY`: `--shape-only` executed both sides for shapes only; no
  values were compared. By default (legacy, unchanged) it exits 0 with
  `status: shape_only`.
- The preceding `[DEBUG]` lines list the derived input shapes and dtypes and
  the reference output dtypes. `Running kernel_fn on GPU for reference` is a
  legacy label; adapters such as the shipped RMSNorm example run on the CPU.
- A timeout, an OOM, a checker exception or a Numba decline on the API path
  is an **error**, not a numerical mismatch and not a pass. Keep it in the
  denominator as errored. On the manual path (`verify_saved_ir.py`,
  `solar-translate verify`) a Numba decline is **not** an error: the runner
  prints `[WARN] Numba declined ... falling back to the interpreter` and
  continues with the (much slower) interpreter, which can still VERIFY. The
  API path (`toDSL_model`) fails fast on a decline. Same checker, different
  policy; both are recorded in the output.

Exit status table (`scripts/verify_saved_ir.py`, `solar-translate verify`):

| Exit | Meaning |
|---|---|
| 0 | VERIFIED; also, by default, `--shape-only` and vacuous passes (legacy) |
| 1 | MISMATCH or IR execution failure |
| 2 | usage error, missing input file |
| 3 | `--strict` only: no numerical proof (shape-only run, vacuous pass) or receipt refused |

**Recommendation:** pass `--strict` whenever you claim a numerical proof.
The default exit codes are kept so existing batch recipes keep working;
`--strict` is the switch that makes "exit 0" mean "numerically verified".
Read the line and the status, not only the exit code.

## Receipt fields (`solar-receipt-v5`)

A receipt is written from the `VerifyResult` of an actual run by
`solar.translate.receipts.emit_receipt` (library),
`solar-translate verify --emit-receipt PATH --definition DIR --dtype D`
or `scripts/verify_saved_ir.py --emit-receipt ...` (both go through the one
shared `emit_workload_receipt` path), and consumed by
`solar.translate.sol(strict=True)` or `solar-translate sol --strict --receipt
PATH --source-path SRC` through `check_strict`.

| Field | Meaning |
|---|---|
| `schema` | `solar-receipt-v5`; every earlier schema is refused (see upgrade below) |
| `verdicts.numerical` | `pass`, `fail` or `error` (`error` also for shape-only runs); strict SOL requires `pass` |
| `verdicts.structural`, `.accounting`, `.hardware` | always `not_certified` |
| `error_stats.max_abs_error`, `.message` | the run's reported error and verdict line |
| `error_stats.vacuous_outputs` | outputs whose reference was all zeros; a non-empty list makes the pass vacuous and `check_strict` refuses it unless the caller states `allow_vacuous=True` |
| `hashes.source_sha256`, `.dsl_sha256` | SHA-256 of the adapter/model and DSL **bytes that were imported and executed**, captured inside the run |
| `hashes.reference_sha256` | SHA-256 of the prepared reference text the run compared against (what `get_source()` returned, or the PyTorch model file); re-derived and compared again at emission and at strict entry |
| `hashes.dependencies` | `{path: sha256}` for adapter-directory modules imported while preparing the reference; re-hashed at emission and strict entry |
| `hashes.definition_sha256` | SHA-256 of `definition.json`, bound at emission (the verifier does not read it) |
| `workload.uuid`, `.axes` | the workload the receipt is issued for (`workload.jsonl` line, verbatim) |
| `workload.validation_axes` | the axis values the verification **actually executed at**, derived from `execution.validation_shapes` and the definition's declared input shapes. A caller-supplied value is cross-checked against that derivation and refused on contradiction; it is never defaulted to the original `axes` |
| `workload.validation_axes_source` | `executed_shapes` (every axis derived from the executed shapes; a caller value that merely agrees adds nothing), `executed_shapes+caller_unchecked` (derived axes plus caller-supplied axes that cannot be derived) or `caller_unchecked` (nothing derivable; caller value only). A caller value contradicting a derived axis is refused, and a partial caller value can never erase a derived axis (review finding R10) |
| `workload.validation_axes_evidence` | `{executed_shapes: [...], caller_unchecked: [...]}`: which axis names are proven by `execution.validation_shapes` and which are caller annotations only |
| `workload.capped_validation` | `true` when the executed validation axes differ from the selected workload's axes: the SOL at `axes` rests on a capped verification |
| `dtype` | the analysis precision the receipt admits; strict SOL compares it with the selected `precision` |
| `layout_transforms` | caller-declared adapter layout or dtype transformations (empty unless supplied) |
| `tolerance.rtol`, `.atol`, `.min_tol` | the policy actually applied, copied from the run, never from the caller |
| `checker.identity`, `.sha256` | the comparison function's module path and file hash |
| `execution.source_type`, `.use_fp64`, `.fast`, `.verification_dtype`, `.config_overrides`, `.evicted_cached_modules` | how the run executed: frontend contract, fp64 or fp32 lowering of the DSL, Numba fast path, the DSL precision flag, capping overrides, cached modules dropped to guarantee fresh imports |
| `execution.validation_shapes` | the tensor shapes actually compared (measured) |
| `execution.input_dtypes`, `.reference_input_dtypes`, `.reference_output_dtypes` | per-tensor dtypes as declared by the source, as fed to the reference at its call boundary (for a PyTorch model the torch tensor dtypes, which can differ from the NumPy storage the DSL reads; review finding R11), and as the reference returned (measured). `verification_dtype` is the DSL precision flag, not the reference's precision |
| `execution.vacuous_reference_outputs` | reference outputs measured all-zero during the run |
| `utc` | emission time |

Refusals (`ReceiptRefusal`) are raised, not logged, and nothing downstream
runs. Observed with the shipped example: a receipt for `fp16` refused a
strict run at `fp8` (`receipt dtype 'fp16' != selected 'fp8'`), and a DSL
copy with one appended comment refused with `stale/mismatched receipt:
dsl_sha256 ...`.

### What a receipt is, and is not

A receipt makes a numerical verdict **portable and bound to its inputs**:
`check_strict` recomputes the SHA-256 of the current source, DSL,
definition, re-derived reference and adapter-local dependencies and refuses
when any differs from what ran, and it re-derives `validation_axes` from the
receipt's own `execution.validation_shapes` and refuses a self-contradictory
receipt.

A receipt is **not** an authenticated document. The JSON is plain text with
no signature or MAC; the input hashes cover the files the receipt refers to,
not the receipt's own verdict fields. Anyone who can write the file can
change `verdicts.numerical` or `error_stats` and `check_strict` will trust
them. The guarantee is "this verdict cannot be silently rebound to changed
code or a different workload", not "this verdict could not have been
edited". Keep receipts with the run logs that produced them, and treat a
receipt from an untrusted source as a claim to re-verify, not as proof.

What a receipt also does **not** defend against: edits to the gate module
itself, changes in modules under the Python environment or the solar
installation (they are trusted as environment), and a wrong reference. It
does not make a verdict stronger than the run that produced it.

### Receipt schema upgrade from v4

Receipts of schema `solar-receipt-v4` and earlier are refused by
`read_receipt` with `receipt has wrong/missing schema (need
'solar-receipt-v5'; receipts from before execution-bound evidence are not
trusted)`. The reason is not cosmetic: a v4 receipt could record
`workload.validation_axes` equal to the original workload axes for a run
that executed capped shapes, carried no per-tensor dtype tables and no
vacuity record. Nothing migrates an old receipt: its verdict was not bound
to the fields v5 requires, so copying it forward would assert evidence that
was never recorded.

To re-establish a result, re-run the verification and emit a fresh receipt:

```bash
solar-translate verify ADAPTER DSL --source-type cuda --strict \
    --emit-receipt receipt.json --definition PROBLEM_DIR --workload-idx 0 --dtype fp16
solar-translate sol DSL --definition PROBLEM_DIR --out-dir OUT --arch B200 --precision fp16 \
    --fallback-adapter ADAPTER --strict --receipt receipt.json --source-path ADAPTER
```

`--strict` on `verify` exits 3 instead of 0 for a shape-only or vacuous run,
and on `sol` refuses to trace unless the receipt hash-matches and carries a
numerical pass. Historical results stay what they were: the old files are
not rewritten, and their verdicts are not re-labelled as v5.

## `sol` validation labelling

`solar.translate.sol` and `solar-translate sol` state which gate applied:
the CLI prints `validation: receipt <path>` or `validation: none (legacy,
unvalidated)`, `SolResult.validated` / `validation_status` carry the same
information, and `<out_dir>/validation.json` is written next to the perf
output so the artifacts can be told apart later. A legacy (`strict=False`)
SOL is unvalidated by design; the perf YAML itself is unchanged.

## `estimate` validation dict

`result.json` from `solar.estimate` carries `validation.kind` (`mechanical`
or `output-comparison`), `numeric_comparison`, and for comparisons
`max_abs_error`, the effective `rtol`/`atol`, `shapes`, `dtypes`,
`ir_sha256`, `scope` and `work_preservation`. The last two are deliberate
warnings written into every result: the comparison covers the supplied
`get_inputs()` case only, and output equality does not establish that the IR
preserves the as-written work. `fallback_reason` records why `auto` left
`v0`, verbatim from the typed `UnsupportedError`.

## Analysis fields that carry verdict information

| Field | Where | Meaning |
|---|---|---|
| `total.macs`, `total.other_ops`, `total.flops` | `analysis.yaml` | Instruction counts from the loop nests (mechanical: from handler equations). One FMA is one MAC and two FLOPs |
| `total.unfused_elements`, `fused_elements`, `fused_prefetched_elements`, `model_io_elements`, `intermediate_elements`, `weight_elements` | `analysis.yaml` | Element totals under the unfused and fused conventions ([SOL_GUIDE.md](SOL_GUIDE.md)) |
| `total.unresolved_ops`, per-layer `loop_count_unresolved` | `analysis.yaml` | Loop nests whose work could not be established. Non-empty means the structural verdict failed; the SOL below it is an undercount, not a bound |
| `sparsity`, `sparsity_source`, per-op `compute_density` | einsum graph and `analysis.yaml` | **Density** (`count_nonzero / size`) and where it came from: explicit annotation, `analytic`, `measured`, `revalidated_original`, `measured_capped`; a discount applied here is a convention, not proof of skipped hardware work |
| `tensor_origins`, `metadata.boundary_storage` | einsum graph, `analysis.yaml` | Recorded provenance (`registered-input` vs `internal-constant`) and which storage dtype each boundary tensor was billed at and why |
| `workload.boundary_billing` | `perf_<arch>.yaml` | `declared_storage_dtypes` (facade `sol`, from the benchmark definition) or `uniform_precision_width` (manual chain, mechanical route). Compare numbers only under the same convention |
| `workload.bytes_per_element`, `mac_rate_bytes_per_element` | `perf_<arch>.yaml` | The width used for uniform billing and for MAC-rate selection |
| `unfused.*`, `fused.*`, `fused_prefetched.*` | `perf_<arch>.yaml` | `memory_bytes`, `compute_cycles`, `memory_cycles`, `total_cycles`, `runtime_ms`, `arithmetic_intensity`, `bottleneck` for each memory convention; `fused` and `fused_prefetched` are currently identical |

## Example: the shipped RMSNorm case

| Step | Evidence class | Observed (documentation smoke at the exported internal snapshot; re-run at the exact public candidate pending) |
|---|---|---|
| `translate verify` at B=8, H=128 | numerical, against a CPU PyTorch emulation of the kernel | `VERIFIED (max_abs_error=3.411e-07)`, tolerance `1e-4`/`1e-5`, floor `8.88e-16` |
| `emit_receipt` for workload `rmsnorm-documentation-smoke-b24` | numerical, portable | `verdicts.numerical: pass`, others `not_certified`; `validation_axes` derived from the executed B=8 shapes, `capped_validation: true` |
| `sol(strict=True)` at B=24, fp16 | structural + accounting + hardware under stated conventions | `total.macs 0`, `total.other_ops 6144`, `unresolved_ops` empty, `boundary_billing: declared_storage_dtypes`, `unfused.memory_bytes 37472` |

The B=8 numerical pass does not certify the B=24 graph numerically. The B=24
counts are a structural result under the as-written convention. The roofline
numbers are bounds under B200 fp16 assumptions with float32 boundary
storage. Stating all three separately is the intended way to report a SOLAR
result.
