<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOLAR 2.0.0 release notes

SOLAR 2.0.0 is the first public release of the v2 program-to-Einsum
speed-of-light (SOL) analysis: the mechanical PyTorch route from 1.x plus the
agentic route (a verified einsum_lib DSL authored by a coding agent or by an
explicitly configured inference API), one shared analysis and performance
backend, and receipts that bind a numerical verdict to the bytes that ran.

## What v2 is

SOLAR v2 turns a program into an Einsum model through one of two frontend
routes and derives speed-of-light roofline bounds from a shared backend:

- **Mechanical lowering** of supported PyTorch graphs (torchview extraction,
  registered operator handlers), with an evidence-based route decision
  (`python3 -m solar.frontend.router`) and in-process capture of bare
  `nn.Parameter` operands.
- **Execution-validated agentic translation** for everything else: a coding
  agent following `skills/verified-translation` (pure skill) or the
  inference-API pipeline (`toDSL_model`, `translate api`; Pythonic) authors an
  executable `einsum_lib` DSL that is compared numerically against an
  executable reference before analysis.
- **Receipts** (`solar.translate.receipts`, schema `solar-receipt-v5`) bind a
  numerical verdict to the exact source, DSL, reference and dependency bytes
  that ran, record the axes and dtypes actually executed and whether the
  reference was all zeros; `sol(strict=True)` and `solar-translate sol
  --strict` refuse stale, rebound, mismatched, vacuous or self-contradictory
  receipts before any tracing. Both manual commands (`solar-translate verify`,
  `scripts/verify_saved_ir.py`) can emit a receipt (`--emit-receipt`).
  Earlier `solar-receipt-v4` receipts are refused and are not migrated:
  re-verify to obtain a v5 receipt ([details](docs/RECEIPTS_AND_VALIDATION.md)).
- **Honest verification state** on the agentic route: a run without a
  reference is reported `UNVERIFIED`, the API wrapper's verdict is the child's
  exit status, size-0 outputs are a failed verdict, all-zero references are
  recorded as vacuous, and `--strict` turns shape-only and vacuous passes into
  exit 3. The legacy exit-0 defaults for shape-only and vacuous runs are
  intentionally unchanged.
- **Installed layout**: the sdist and wheel carry the hardware configs,
  prompt references, skills, verification/tracing scripts, torchview patch
  and examples; `solar.resources` / `solar-resource` locate them, and
  `--arch-config B200` by name works from a wheel install. Package metadata
  has one source (`pyproject.toml`), Apache-2.0 throughout, eight console
  scripts, extras `[verify]`, `[jax]`, `[all]`.
- **Unified API** `solar.estimate.estimate` with modes `auto`, `v0`
  (mechanical only, never calls an LLM), `v1` (API translation) and `v2`
  (agent-authored IR), typed failures and a unique run directory per call.
- **Accounting repairs** carried from the campaign audits: declared boundary
  storage dtypes billed as such by the facade, exact integer loop extents with
  unresolved loops reported in `total.unresolved_ops` instead of zeroed,
  recorded tensor origins instead of name heuristics, and recurrence
  placeholder provenance so the replication discount never applies to real
  per-step work.

## Packaging

- Version `2.0.0` (`solar.__version__`, single source of truth for the
  distribution metadata).  The torch floor is `>=2.7.1`, the lowest version
  with recorded full-suite evidence for this tree; `torchview>=0.2.6` is
  patched by `install.sh` (see `patches/README.md`).
- Console scripts: `solar-process-model`, `solar-toeinsum-model`,
  `solar-analyze-model`, `solar-predict-perf-model`, `solar-totimeloop`,
  `solar-translate`, `solar-toDSL-model`, `solar-resource`.
- Optional extras: `solar[verify]` (numba, psutil) for fast DSL execution and
  the RSS memory guard; `solar[jax]` (jax, jaxlib, flax) for JAX/Flax sources;
  `solar[all]`.
- Workflow resources (hardware configs, docs, skills, verification and tracing
  scripts, patches, examples) are packaged into the wheel and located with
  `solar-resource <relative/path>` (`solar.resources`) in a checkout, an
  unpacked sdist or a wheel install alike.
- SOLAR ships no inference endpoint and reads no implicit key file: the
  Pythonic agentic route runs only with `SOLAR_API_BASE_URL` set and a key
  given explicitly (`--api-key`, `--api-key-file` or `OPENAI_API_KEY`).
- Apache-2.0 (`LICENSE`); third-party notices in `THIRD_PARTY_NOTICES`; an
  SPDX header on every shipped source and configuration file.

## Documentation

New or rewritten for v2: `README.md`, `docs/DISPATCHER.md`, `docs/SOLAR_FLOW.md`,
`docs/SOLAR_V1_FLOW.md`, `docs/SOLAR_V2_FLOW.md`, `docs/SOLAR_V2_CONTRACTS.md`,
`docs/SOLAR_V2_OVERVIEW.md`, `docs/V2_FLOW.md`, `docs/PYTHON_API.md`,
`docs/RECEIPTS_AND_VALIDATION.md`, `docs/TROUBLESHOOTING.md`,
`examples/README.md`, `AGENTS.md`. Every documented offline command was
executed at this snapshot on x86_64 (Python 3.10, CPU only); API-backed
commands were not executed and are marked as such.

Removed after that snapshot (2026-11-04): the legacy kernelbench einsum-graph
verifier `python -m solar.cli.verify_einsum` / `solar.cli.verify_einsum_model`.
It depended on the separate `solar_verifier` project, which this tree never
contained, and failed with `ModuleNotFoundError: No module named 'verify'`
before validating its input. `docs/VERIFICATION_GUIDE.md` now points at the
current entry points (`solar-translate verify`, `scripts/verify_saved_ir.py`);
nothing replaces the removed command.

## Examples

- `examples/TwoInputMatmul/`: new audited mechanical example (262,144 MACs,
  routes `mechanical`).
- `examples/Matmul/` (learned parameter, 1,048,576 MACs) and
  `examples/Attention/` (1,310,720 MACs) are checked against hand counts;
  Attention routes `agentic` under the router's operand-evidence rule.
- `skills/verified-translation/example/`: RMSNorm adapter + DSL, verified at
  B=8, H=128 (`max_abs_error=3.411e-07`), traced at B=24.

## Removed in the release cleanup (2026-11-04)

Removed from the release branch under the maintainers' instruction to ship no
benchmark code, generated outputs or unused material. None of the remaining
commands changed behaviour; numerical and accounting code is untouched.

- Benchmark packaging and batch drivers: `solar/benchmark/` (SolBench v3
  parser and generator), `solar/einsum/benchmark_converter.py`,
  `solar/graph/benchmark_processor.py`, the batch console scripts
  `solar-process` and `solar-toeinsum` with `solar/cli/process.py` and
  `solar/cli/toeinsum.py`. Replacement: the single-model commands
  `solar-process-model`, `solar-toeinsum-model`, `solar-analyze-model`,
  `solar-predict-perf-model`, `solar-totimeloop` (or
  `examples/<Name>/run_solar.sh`). The declared-shape resolver that
  `solar.translate.sol` used from the benchmark parser now lives in
  `solar/translate/_dimspec.py` with the same grammar and the same refusals.
- Dead modules: `solar/cli/process_backward.py` and
  `solar/graph/backward_processor.py` (backward-graph extraction; unreferenced
  and broken) and `solar/einsum/af_graph_visualizer.py` (no caller). No
  replacement.
- Campaign artefacts: `tests/fixtures/allshapes_lock/` (review-round scripts no
  test loaded), `tests/test_l1_violation_fixes.py` (needed a private sol-bench
  checkout), `tests/evidence/`.
- Benchmark-derived test inputs: the SOL-ExecBench-Lite and kernelbench-derived
  DSL fixtures and the programs lowered from them, together with the test
  cases that consumed them, moved to the maintainers' private regression
  harness (see `RELEASE_CHECKLIST.md`, "Public / private test boundary");
  `examples_verified/lite_window_attention/` and its test likewise. The
  public-by-construction geometry controls in `tests/release_portable/` remain.
- Documentation: `README_V2.md` (its dispatcher contract is now
  `docs/DISPATCHER.md`, its validation table `docs/SOLAR_V2_OVERVIEW.md`),
  `docs/USAGE.md` (superseded by `README.md` and `docs/SOLAR_V1_FLOW.md`),
  `docs/QUANT_SUPPORT.md` (described tooling that is not shipped),
  `docs/logo/solar_icon.png` (unreferenced, no provenance record).

Added: `examples/TwoInputMatmul/` (the example the documentation already
described), `tests/test_dimspec.py`, `docs/DISPATCHER.md`,
`RELEASE_CHECKLIST.md`, `REUSE.toml` and `LICENSES/` (licence placement for
header-free files), and the fallback-protocol specification files are now
packaged in both the sdist and the wheel.

## Known limitations

- No universal dispatcher. The caller chooses the route; the router and
  `estimate(mode="auto")` apply different, bounded supportability tests and
  can disagree (Attention: converter lowers, router declines).
- The router checks connectivity and shape semantics only; it does not check
  operator arguments that leave shapes unchanged and is not a correctness
  certificate.
- Mechanical route is PyTorch/torchview only. JAX, Triton and CUDA are
  agentic-route source contracts, not mechanical frontends.
- Learned-parameter capture is implemented by an in-process shim; its
  clean-install behaviour against unpatched torchview was not exercised for
  this snapshot. Derived parameter views (`self.weight.t()`) are still dropped.
- The CLI `translate sol` performs no numerical validation itself; without
  `--strict --receipt --source-path` it is unvalidated by design and says so
  (`validation: none (legacy, unvalidated)`, `validation.json`).
- `estimate` validates only the reference's own `get_inputs()` case, refuses
  capped validation, and has no benchmark-definition workload concept. Its
  mode `v2` task packet names a skill file that does not exist; use
  `skills/verified-translation/SKILL.md`.
- Pythonic route gaps: the `jax` API branch does not run (undefined
  `jax_reference_is_fp64`, unsupported keyword); `raw` source has no
  reference and is now reported `UNVERIFIED` (exit 1) unless `--no-verify`
  is passed; `translate api` lists `raw` but has no `--input-shapes`; scalar
  (0-d) PyTorch parameters are skipped when loading synthesized values.
- Policy defaults kept open on purpose (exposed on the result, refused only
  under `--strict`): shape-only exit 0, the all-zero-reference pass, and the
  direct `TranslationAgent` fp32 tolerance (atol 1e-4, recorded and warned;
  every CLI route uses rtol 1e-4 / atol 1e-5).
- There is no `solar-estimate` console script; use `python3 -m
  solar.cli.estimate` or the `solar.estimate` Python API.
- Helper scripts invoked by path import whichever `solar` is installed.
- Test isolation with an editable torchview: a verification run
  (`solar.translate._runner`) evicts cached modules whose files lie outside
  `sys.prefix` and the SOLAR checkout so the reference's imports re-execute
  current bytes; torchview's own package files are exempt by resolved package
  origin and an evicted child is detached from its retained parent
  (`tests/test_verify_keeps_patched_torchview.py`). The full suite is
  expected to pass in one process with the pinned patched torchview installed
  in site-packages; with torchview installed editable from a sibling clone
  (what `install.sh` does) run the suite in a fresh process per module if
  mechanical-extraction tests lose raw-attribute capture.
- The Pythonic route needs an explicitly configured OpenAI-compatible
  endpoint (`SOLAR_API_BASE_URL` or `base_url=`) and key; none ships with
  SOLAR, and model names passed with `--llm-model` must be ones that
  endpoint serves. The deterministic skill route needs neither.
- `torch>=2.7.1` is the lowest version with recorded full-suite evidence; no
  run on other torch versions is recorded for this tree.
- The sdist carries the canonical `skills/` but not the per-agent discovery
  symlinks (`.claude/skills`, `.agents/skills`), which are a git-checkout
  convenience.
- The maintainers' window-attention padding/crop regression test and its
  fixture bundle are not in this tree: the fixtures derive from benchmark
  content outside the public allowlist. This tree therefore makes no
  padding/crop coverage claim; a reviewed portable fixture is an open
  obligation, not a silent skip.
- DSL-to-Timeloop export of v2 graphs ran for the RMSNorm example but is
  unvalidated against Timeloop; `fused` and `fused_prefetched` SOL are
  currently identical (`docs/SOL_GUIDE.md`).
- Historical campaign tallies (for example 235/235 SOL-ExecBench
  verifications at capped shapes) are not re-run per checkout and certify
  neither counts nor SOL. No universal-correctness or full-benchmark claim
  is made for this release.
