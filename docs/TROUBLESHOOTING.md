<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Symptoms are grouped by where they appear. "Both routes" means the shared
analysis and perf stages.

## Installation and environment

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'solar.einsum_dsl.constants_binding'` (or another `solar.*` module) from `scripts/*.py` | The script imported a **different** installed `solar` than the checkout you are in. Scripts invoked by path put their own directory, not the current directory, first on `sys.path`. Install this checkout (`python3 -m pip install -e .` from the repository root) or export `PYTHONPATH=$PWD` before running scripts by path. `python3 -m solar.cli.<name>` from the repository root and the `solar.translate.sol` facade (which pins its subprocess) are not affected. Check with `python3 -c "import solar; print(solar.__file__)"` |
| `python3` is Python 3.6/3.8 and imports crash | SOLAR needs Python 3.10+. Point `PY`/`python3` at a 3.10+ interpreter with torch installed |
| `No module named numba` or verification is very slow | Numba is optional (`pip install numba`, listed in `requirements.txt`). Without it the DSL runs interpreted; convnet-sized verifications can take minutes to hours |
| `No module named jax`/`flax` | Only needed for `--source-type jax`; install `jax[cpu] flax` |
| `solar-translate` or `solar-totimeloop: command not found` | Your install predates the console-script entries; reinstall (`pip install -e .` or the wheel) or use `python3 -m solar.cli.translate` / `python3 -m solar.cli.totimeloop`. There is no `solar-estimate` script: use `python3 -m solar.cli.estimate` |
| `ResourceNotFound: SOLAR resource 'configs/arch/...' not found. Searched ...` | You are running from a wheel built without the packaged resources, or `SOLAR_RESOURCE_ROOT` points elsewhere. Install a wheel built from the sdist, or set `SOLAR_RESOURCE_ROOT` to a checkout; `python3 -m solar.resources` shows the roots in use |
| `CONDA_PATH_BACKUP: unbound variable` | `set -u` plus a conda activation script. Use `set -o pipefail` only in scripts that source environment files |

## Mechanical route

| Symptom | Cause and fix |
|---|---|
| `total.macs` is a power-of-two fraction of the hand count and a `weight` is missing from `tensor_types` | A learned parameter was dropped by the graph extractor. At this base `solar/graph/_torchview_param_shim.py` captures bare `nn.Parameter` operands and fails closed when it cannot; a derived view such as `self.weight.t()` is still an untraced constant. Compare with a hand count and prefer explicit inputs or the agentic route for such models |
| `UnsupportedFrontendOperandError` during `process_model` | The shim refused to emit a graph with incomplete operand accounting (keyword tensor arguments it could not bind, or an uncaptured parameter). This is the intended fail-closed path; route the model agentic |
| Router says `agentic` but `toeinsum_model` succeeds | They apply different tests. The router needs positive operand evidence for every node (for example a `mul` by a Python float records one tensor operand and is declined); the converter may still lower it. Read `route.json` `reasons`, then decide whether the mechanical counts are trustworthy for that node |
| `einsum_graph_renamed.yaml` equals `einsum_graph.yaml` | Expected at this base; `--enable-rename` does not change counts |
| `--enable-llm-agent` asks for an API key | That option generates handlers through an inference API. It is not the v2 execution-validated route; leave it off unless you intend it |

## Agentic route: verification

| Symptom | Cause and fix |
|---|---|
| `API key required` / `The LLM API pipeline runs only when BOTH hold ...` (exit 2) | You invoked `translate api` or `toDSL_model` without opting in. The default workflow is the pure-skill route; if the Pythonic route is intended, pass `--api-key` / set `OPENAI_API_KEY` (`toDSL_model` also reads `--api-key-file`). Never commit keys |
| `No inference endpoint configured: pass base_url= or set SOLAR_API_BASE_URL ...` | SOLAR ships no inference endpoint and never falls back to a public provider. Set `SOLAR_API_BASE_URL` to your OpenAI-compatible endpoint (or `base_url=` in Python); `--llm-model` must name a model it serves |
| `GENERATED CODE (UNVERIFIED)` / exit 1 from `toDSL_model --source-type raw` | `raw` source text has no reference, so execution alone is never a verification. Pass `--no-verify` if you only want the DSL, or use a source type with a reference |
| `AgentTranslationRequired: Agent IR required; task packet: .../AGENT_TASK.md` | `estimate(mode="v2")` without `ir_path`. Have the coding agent author the DSL per `skills/verified-translation/SKILL.md` (the packet's `skills/solar-estimate/SKILL.md` reference does not exist at this base) and resubmit with `ir_path` |
| `ValidationError: Capped validation is not accepted by the unified API` | `estimate` requires the prepared shapes to equal the verification shapes. Use `solar.translate.verify` plus `sol` with a definition when verification must be capped |
| `UNVERIFIED`, error ~1e-1 | Wrong algorithm variant (RoPE flavour, GQA repeat order, raw-vs-cumsum gates). Re-read the reference code, not its names |
| `UNVERIFIED`, error ~1e-3..1e-2 on quantized or bf16 kernels | Irreducible output-grid noise. Use the class tolerance from the skill's reference (`--rtol 0.02 --atol 0.005` for QDQ; `--rtol 0.01` for bf16-output attention) and record the rationale in `NOTES` |
| Mismatch only on a model with a scalar learned parameter | The PyTorch frontend skips 0-d parameters when loading synthesized values, so the reference keeps the stored scalar and the DSL receives a random one. Compare against the stored scalar or avoid 0-d parameters in the capped copy |
| `WARNING` about an all-zero reference output, `[VACUOUS: ...]` on the verdict line | The comparison is vacuous: an all-zero DSL output would also have passed. Choose non-degenerate inputs. By default the run still exits 0 (`status: pass_vacuous`); `--strict` exits 3 and a receipt from it is refused by strict SOL unless `allow_vacuous=True` is stated |
| exit 3 from `verify --strict` / `verify_saved_ir.py --strict` | No numerical proof: the run was `--shape-only`, the pass was vacuous, or the receipt was refused. This is the intended refusal; read the printed reason |
| `OOM(rc=137)` / `Cannot allocate memory` | Real RAM exhaustion. Lower `--mem-budget`, check for a second run on the same host. Never use `ulimit -v` (torch reserves large virtual address ranges) |
| Verify hangs for minutes with ~0 CPU | Memory-guard fork deadlock. Rerun without `--mem-budget` |
| `Terminating: fork() called from a process already using GNU OpenMP` then `EOFError` in `_run_isolated` | A forked verification worker started after OpenMP was initialised in the parent (typically a pytest session that ran a Numba parallel kernel first). Run that test file in its own process. At this base `numba_accel` pins the fork-safe `workqueue` threading layer at import, which removes the known in-suite case; the message can still appear in other parents |
| `TIMEOUT(rc=124)` | Raise `--timeout` (verify) or `STEP_TIMEOUT`/`VERIFY_TIMEOUT` (batch scripts) |
| `Numba declined the generated code` (API path, fails fast) / `[WARN] Numba declined this code, falling back to the interpreter` (manual path, continues) | Rewrite per the message: supported builtins only, no inner imports, no `Tensor.from_numpy` aliasing. On the manual path the interpreter result is still a valid verdict, only slower; `--no-numba` (API) / `--no-fast` (manual) force the interpreter |
| `cannot import name 'X' from ...core` | A builtin is registered in `numba_accel.execution_namespace()` but missing in `core.py`; implement both |
| `[DEBUG] Running kernel_fn on GPU for reference` on a CPU-only adapter | Legacy log label. The shipped example runs on the CPU; the line is not evidence of GPU execution |

## Agentic route: receipts and SOL

| Symptom | Cause and fix |
|---|---|
| `ReceiptRefusal: strict SOL requires a receipt path` / `requires source_path` | `sol(strict=True)` needs both `receipt` and `source_path` |
| `ReceiptRefusal: stale/mismatched receipt: dsl_sha256 ...` (or `source_sha256`, `definition_sha256`) | The file changed after verification. Re-verify and re-emit; do not edit the receipt |
| `ReceiptRefusal: reference changed since verification` / `helper dependency changed` | The adapter's `get_source()` output or an adapter-local module changed. Re-verify |
| `ReceiptRefusal: receipt dtype 'fp16' != selected 'fp8'` | One receipt admits one analysis precision. Verify and emit per precision you bill |
| `ReceiptRefusal: receipt has wrong/missing schema (need 'solar-receipt-v5' ...)` | The receipt predates schema v5 (its validation axes and dtypes were not bound to the run). It is not migrated: re-run `verify --emit-receipt` to obtain a v5 receipt |
| `ReceiptRefusal: validation_axes cannot be derived ... and none were given` | No definition input shares a name with an executed tensor. Pass `validation_axes=` with the capped shapes you actually verified; the original workload axes are never assumed |
| `ReceiptRefusal: receipt pass is vacuous: reference output(s) [...] were all zeros` | Strict SOL refuses a vacuous pass. Fix the inputs and re-verify, or state `allow_vacuous=True` only after confirming from the reference source that zero is the genuine result |
| `REFUSED: strict SOL: ...` (exit 3) from `solar-translate sol --strict` | Same refusals as above, surfaced by the CLI; without `--strict` the CLI proceeds unvalidated and says so |
| `ReceiptRefusal: dependency provenance cannot be established ...` | The reference imported modules from outside the adapter directory, Python environment and solar checkout. Make the reference self-contained or adapter-local |
| `TypeError: emit_receipt requires a VerifyResult carrying sealed execution evidence` | Receipts are only issued for actual `verify`/`run_verification` results; a reconstructed result is rejected by design |
| `trace failed (rc=...)` from `sol` | Read the appended log: usually an unresolved axis (`--axis NAME=VALUE`, or `workload_idx`), a missing `--fallback-adapter`/`custom_inputs` for value-dependent inputs, or a constant without provenance (`_AXIS_BINDINGS`/`_FREE_CONSTANTS`) |
| `total.unresolved_ops` is non-empty / `loop_count_unresolved` on a layer | A loop extent could not be established. The SOL is an undercount; fix the `Dim` sizing (registered names, `size=X.shape[i]`, module constants) and re-trace |
| Trace fails with a dim-count mismatch or `read-only` | The DSL wants pre-decomposed inputs: supply `--custom-py full_inputs.py` with writable zero-stride placeholders |
| `--density-override` was ignored (`WARNING` lists available names) | Override names must match the trace's tensor names (often `__const_N` or `name#0`). Check stderr for the list |
| `GUARDED` / `REFUSED` lines in a measured-density run | The attention guard kept a probability tensor dense, or the original-scale revalidation did not fit the budget. Both are conservative; use an analytic `--density-override` only with a justified value |
| `sol` and the manual chain give different `memory_bytes` for the same graph | `sol` bills boundary tensors at the definition's declared dtypes (`boundary_billing: declared_storage_dtypes`); the manual chain bills at the uniform precision width. Compare under one convention |
| `totimeloop` on a v2 graph | Runs at this base but the export is unvalidated; do not present it as a runnable Timeloop workload |

## Tests

| Symptom | Cause and fix |
|---|---|
| `tests/test_param_shim_installs.py` skips everything | The installed torchview already carries the parameter patch, or the repository patch did not apply; the shim's clean-install path is not exercised in that environment |
| `tests/test_estimate.py` cap-escalation cases fail with a missing key | Older versions read a repository-root key file for offline fixtures; this release reads no implicit key file and the fixtures stub their credentials and endpoint. If it recurs, run the file alone and report the environment |
| Mid-suite `EOFError` in `_run_isolated` | See the OpenMP fork entry above; run the file alone |
