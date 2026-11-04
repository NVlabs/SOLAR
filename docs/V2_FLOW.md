<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `solar.translate`: library facade for the agentic route

This page documents only the Python facade and its artifact layout. For route
selection read [SOLAR_FLOW.md](SOLAR_FLOW.md); for the runnable agentic
commands read [SOLAR_V2_FLOW.md](SOLAR_V2_FLOW.md); for what each stage does
and does not guarantee read [SOLAR_V2_CONTRACTS.md](SOLAR_V2_CONTRACTS.md);
for the unified `solar.estimate` API read [PYTHON_API.md](PYTHON_API.md).

## Two modes, one artifact contract

```python
from solar import translate as T

# Agent mode (default): create a workspace + checklist. No LLM call, no key lookup.
res = T.scaffold(problem_dir, out_dir, source_type="cuda")     # ScaffoldResult
res = T.translate(problem_dir_or_file, out_dir)                # same, mode="agent"

# Mode-agnostic deterministic steps (used by both workflows):
v = T.verify(adapter_or_model, dsl, source_type="cuda",
             mem_budget_gb=None, rtol=None, atol=None, timeout=None)  # VerifyResult
s = T.sol(dsl, definition_dir, out_dir, arch="B200", precision="fp16",
          fallback_adapter=None, custom_inputs=None,
          measure_density=False, density_overrides=None,
          workload_idx=0, perf_precision=None,
          receipt=None, strict=False, source_path=None)              # SolResult

# API mode (explicit opt-in): toDSL_model with an inference-API LLM.
T.translate(source_file, out_dir, mode="api", source_type="pytorch", api_key=KEY)
```

Behaviour verified against `solar/translate/__init__.py` at source snapshot
the exported internal snapshot:

- `scaffold` writes `adapter.py` (stub) for `cuda`, or
  `README_pytorch.txt` / `README_triton.txt` for sources that need no adapter,
  plus `NOTES`, and returns the step checklist as text. For `jax` it falls
  through to the same CUDA-style adapter stub, which the JAX frontend does not
  consume; use the JAX module contract from the contracts page instead. An
  existing `adapter.py` or `NOTES` is preserved; the generated `README_*.txt`
  files are rewritten on every call, so do not keep edits in them. It does
  **not** run an agent or produce a DSL.
- `verify` is the library form of `scripts/verify_saved_ir.py` (same code
  path, same output lines). `rtol=None`/`atol=None` mean `1e-4`/`1e-5`. The
  returned `VerifyResult` has `verified`, `max_abs_error`, `message`,
  `reference_seconds`, `ir_seconds`, `shape_only` and a sealed `execution`
  evidence object that `emit_receipt` consumes.
- `sol` shells out to `scripts/trace_saved_ir.py` for the original-shape
  trace with this checkout pinned on the subprocess `PYTHONPATH`, then runs
  the DSL compiler, analyzer and perf model as library calls into
  `dsl_full/`, `einsum_full/`, `analysis_full/`, `perf_full/perf_<arch>.yaml`.
  It binds the definition's declared input/output dtypes as boundary storage
  widths (`boundary_billing: declared_storage_dtypes`). With `strict=False`
  (default) it does not call `verify` and does not check for a prior
  verification. With `strict=True` it requires `receipt` and `source_path`
  and raises `ReceiptRefusal` before tracing unless the receipt hash-matches
  the current files, the selected workload and `precision`.
  `perf_precision` lets the perf stage bill at a different precision than the
  analysis stage. `sol()` forwards `axes`, `mem_budget_gb`, `measure_timeout`,
  `measure_attention_density` and `allow_unbound_constants` to the trace
  script. `SolResult.validated` / `validation_status` / `receipt` and
  `<out_dir>/validation.json` state which gate applied.
- `translate(mode="api")` requires a key via `api_key=` or `OPENAI_API_KEY`
  and raises otherwise; it never searches for key files, and the endpoint
  must come from `SOLAR_API_BASE_URL`. It runs `solar.cli.toDSL_model` in a
  subprocess and takes `verified` from the child's **exit status** (0 only
  when its result was verified); an exact `GENERATED CODE (UNVERIFIED):`
  line that disagrees with exit 0 fails closed. `TranslateResult.exit_code`
  and `verification_status` are exposed.

CLI: `python3 -m solar.cli.translate {scaffold|verify|sol|api}` or the
`solar-translate` console script; a bare problem directory means `scaffold`.
`verify` accepts `--shape-only --strict --emit-receipt PATH --definition DIR
--workload-idx N --dtype D`; `sol` accepts `--strict --receipt PATH
--source-path SRC --workload-idx --perf-precision --axis --mem-budget
--measure-timeout --measure-attention-density --allow-unbound-constants`.
Exit status: 0 verified, 1 mismatch/failure, 2 usage or missing file, 3 no
numerical proof under `--strict` or receipt refused. `scripts/verify_saved_ir.py`
and `scripts/trace_saved_ir.py` remain thin wrappers over the same internals
with the same flags.

## Receipts

```python
from solar.translate.receipts import emit_receipt, check_strict, read_receipt, ReceiptRefusal

path = emit_receipt(v, source_path=..., dsl_path=..., definition_path=...,
                    workload_uuid=..., axes={...}, dtype="fp16", out_path="receipt.json",
                    validation_axes=None,      # derived from the executed shapes; a value you pass is cross-checked
                    layout_transforms=None)
receipt = check_strict(path, source_path=..., dsl_path=..., definition_path=...,
                       workload_uuid=..., axes={...}, dtype="fp16",
                       expected_tolerance=None, expected_checker=None)
```

`emit_receipt` accepts only a `VerifyResult` returned by an actual
`run_verification`/`verify` call and refuses when the current source or DSL
bytes, the re-derived reference, or any adapter-local dependency differ from
what ran, or when `validation_axes` can neither be derived nor were given.
`emit_workload_receipt(v, ..., definition_dir=..., workload_idx=0)` is the
shared path the CLI `--emit-receipt` uses. `check_strict(...,
allow_vacuous=False)` is what `sol(strict=True)` calls; it refuses vacuous
passes unless told otherwise. Field meanings and the v4 to v5 upgrade are in
[RECEIPTS_AND_VALIDATION.md](RECEIPTS_AND_VALIDATION.md).

## Artifact layout per problem directory

| File | Written by | Meaning |
|---|---|---|
| `adapter.py` (cuda) or `model_capped.py` (pytorch/triton) | author | golden reference: verbatim source plus capped, non-degenerate inputs |
| `solar_model_einsum_lib.py` | author | the DSL under verification (loop-nest programs; see the skill's reference) |
| `verify.log` | a manual `tee` | captured verifier output; the `verify` command itself prints and does not write it |
| `receipt.json` | `emit_receipt` / `verify --emit-receipt` | hash-bound numerical verdict for one verification run (`solar-receipt-v5`) |
| `validation.json` | `sol` | which gate applied to the SOL artifacts (`receipt <path>` or legacy unvalidated) |
| `dsl_full/` -> `einsum_full/` -> `analysis_full/` -> `perf_full/perf_<arch>.yaml` | `sol` or the manual chain | original-shape trace (+ `constant_provenance.json`) -> einsum graph -> counts -> SOL |
| `full_inputs.py` (optional) | author | `full_inputs(axes)` for value-dependent or pre-decomposed inputs at original shapes |
| `NOTES` | `scaffold` template, filled by author | entry point, caps, self-check error, tolerance rationale, quirks |

## Evidence status

Earlier versions of this page listed acceptance counts (a 235/235
SOL-ExecBench tally, facade regression matrices and fresh-agent
reproducibility results). Those are **historical reports from earlier
commits and hosts** and were not re-run for this base. Where cited elsewhere
they describe numerical verification at capped verification shapes for
workload row 0 with per-class, author-chosen tolerances; they do not certify
original-shape operation counts, memory traffic or hardware assumptions, and
the coding agent that authored those translations consumed model tokens
outside SOLAR. Current, reproducible evidence for this checkout is the
RMSNorm smoke flow and the focused test run recorded in
[SOLAR_V2_CONTRACTS.md](SOLAR_V2_CONTRACTS.md#observations-at-this-base).

## Where things live

- Skill (agent-mode instruction set, vendor-neutral): `skills/verified-translation/`
  (symlinked into `.claude/skills/` and `.agents/skills/`).
- Facade: `solar/translate/` (`__init__.py`, `_runner.py`, `receipts.py`,
  `reuse.py`); CLI: `solar/cli/translate.py`.
- Campaign tooling is not part of this candidate; aggregate per-problem
  outputs with your own scripts.
- Tests: `tests/test_translate_facade.py`, `tests/test_translate_reuse.py`,
  `tests/test_router_receipts.py`, `tests/test_skills.py`,
  `tests/test_sparse_cost.py`, `tests/test_density_guard.py`.
