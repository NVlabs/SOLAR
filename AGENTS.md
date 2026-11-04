<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md: SOLAR

Guidance for coding agents working with SOLAR, from a source checkout or from
an installed wheel. Human-facing documentation starts at `README.md`.

## What SOLAR is

Program-to-Einsum speed-of-light (SOL) performance analysis with two
explicitly selected frontend routes feeding one shared analysis/perf backend
(`docs/SOLAR_FLOW.md`):

1. **Mechanical route** (supported PyTorch graphs; no agent, no API):
   torchview graph -> registered einsum handlers -> analysis -> perf ->
   optional Timeloop export (`solar-process-model`, `solar-toeinsum-model`,
   `solar-analyze-model`, `solar-predict-perf-model`, `solar-totimeloop`;
   route check with `python -m solar.frontend.router`; `docs/SOLAR_V1_FLOW.md`).
2. **Agentic route** (programs the mechanical route cannot lower: other
   PyTorch operators, JAX, Triton, CUDA): an executable `einsum_lib` DSL is
   authored, numerically **verified** against an executable reference
   (`solar-translate verify` / `scripts/verify_saved_ir.py`), optionally
   bound into a receipt (`solar.translate.receipts`, schema
   `solar-receipt-v5`), traced at original shapes, compiled to an einsum
   graph and fed to the same backend (`docs/SOLAR_V2_FLOW.md`). Two workflows:
   - **pure skill** (default): YOU author the DSL following
     `skills/verified-translation/SKILL.md`; SOLAR's commands call no LLM;
   - **Pythonic** (explicit opt-in only): `solar-toDSL-model`,
     `solar-translate api`, or `solar.estimate` mode `v1` have an
     inference-API model write the DSL. They require an explicitly configured
     endpoint *and* key and never run implicitly. `raw` source text on that
     path has no reference, so it is reported `UNVERIFIED` unless you pass
     `--no-verify`.

`solar.estimate.estimate` is the unified Python entry (`docs/PYTHON_API.md`):
mode `v0` mechanical only, `v1` API translation, `v2` agent-written IR,
`auto` = `v0` with fallback to `v1` only on a typed unsupported error.

## Environment and layouts

- **Python 3.10+ with torch.** Check which copy you are running:
  `python -c "import solar, solar.resources; print(solar.__file__); print(solar.resources.describe())"`.
- **Two supported layouts.** In a source checkout (or an unpacked sdist) run
  commands from its root; `scripts/`, `configs/`, `skills/` are relative to
  it. From a wheel install there is no checkout: the same resources are
  packaged and located with `solar-resource <relative/path>`
  (`python -m solar.resources`), for example
  `python "$(solar-resource scripts/verify_saved_ir.py)" adapter.py dsl.py --source-type cuda`.
  `SOLAR_RESOURCE_ROOT=<dir>` overrides the lookup. A missing resource
  raises `ResourceNotFound` naming every location searched; nothing falls
  back silently.
- **Scripts invoked by path import whichever `solar` is installed.** If more
  than one SOLAR is installed, use `python -m solar.cli.<name>` from the
  checkout you mean or export `PYTHONPATH=<checkout>`.
- **Optional dependencies**: `pip install "solar[verify]"` for `numba` (fast
  DSL execution) and `psutil` (the `--mem-budget` RSS guard);
  `"solar[jax]"` for JAX/Flax sources. Without numba the DSL runs
  interpreted (slow, same results).
- **Inference endpoint and key (Pythonic route only).** SOLAR ships no
  endpoint. Set `SOLAR_API_BASE_URL` to your OpenAI-compatible endpoint (or
  pass `base_url=` in Python); the agent refuses to start without one and
  never falls back to a public provider. The key comes from `--api-key`,
  `--api-key-file` or `OPENAI_API_KEY`. The model comes from `--llm-model` or
  `SOLAR_LLM_MODEL` (no default ships) and must be a name
  your endpoint serves. Never commit keys or endpoint files.
- **Compute**: verification executes the reference and the DSL. Keep it to
  capped shapes on shared hosts and bound memory with `--mem-budget` (an RSS
  guard), never `ulimit -v`.

## Layout

| Path | What |
|---|---|
| `solar/cli/` | console-script entry points (`python -m solar.cli.<name>` works too) |
| `solar/frontend/router.py` | evidence-based mechanical-vs-agentic decision on an extracted graph |
| `solar/translate/` | facade (`scaffold`/`verify`/`sol`), `receipts.py`, `_runner.py` |
| `solar/estimate.py` | unified source-to-SOL API |
| `solar/einsum_lib/` | DSL: `core.py` (Tensor/kernel/builtins), `verify.py`, `numba_accel.py`, `pipeline.py`, `agent.py`, per-source frontends, `source_load.py` |
| `solar/einsum_dsl/` | DSL -> einsum graph compiler |
| `solar/graph/`, `solar/einsum/`, `solar/analysis/`, `solar/perf/` | mechanical extraction (+ parameter shim), handlers, shared analysis, perf model |
| `solar/resources.py` | locates packaged resources (`solar-resource`) |
| `scripts/` | `verify_saved_ir.py`, `trace_saved_ir.py`, `diff_ir_stages.py`, `gen_solexecbench_adapter.py` (adapter generation for capped verification workloads from a user-supplied problem definition; no benchmark runner/configs) |
| `configs/arch/` | hardware configs (A6000, B200, H100_PCIe, H100_SXM, H200_SXM, L40S, Jetson_Thor_T5000) |
| `examples/`, `examples_jax/` | example models; `examples/README.md` lists checked counts |
| `tests/` | pytest; run files that fork workers in their own process |

## Skills (vendor-neutral)

Canonical skill files live in `skills/` (plain markdown + runnable example);
`.claude/skills/` and `.agents/skills/` are symlinks for per-agent discovery
in a checkout. From a wheel: `solar-resource skills/verified-translation/SKILL.md`.

| Skill | Use when |
|---|---|
| `skills/verified-translation/` | translating/verifying models or kernels into the einsum_lib DSL, einsum graphs, SOL |
| `skills/capping-weights/` | verification inputs/weights exceed memory: const-override capping strategy |

Test the skills themselves: `python -m pytest tests/test_skills.py
tests/release_agentic/test_skill_claims.py` (layout, frontmatter, every
command/flag the skill names, the shipped example must VERIFY). Behavioral
re-test after skill edits: give a FRESH agent only the skill files and an
unseen problem; it must reach VERIFIED without reading existing solutions.

## Conventions

- Outputs are human-readable YAML (no anchors/aliases).
- Report numerical, structural, accounting and hardware status separately;
  never collapse them into `valid`. A receipt certifies only the numerical
  verdict, bound to the bytes that ran (`docs/RECEIPTS_AND_VALIDATION.md`).
- Read the verdict line and the exit status, not one of them: `--shape-only`
  exits 0 by default (`status: shape_only`), an all-zero reference is a
  recorded vacuous pass, and only `--strict` turns both into exit 3.
  Recommend `--strict` whenever a numerical proof is claimed.
- Batch scripts isolate each problem with `setsid -w timeout ...` so
  per-problem OOM/timeout cannot kill the loop; memory is bounded via
  `--mem-budget`, **never** `ulimit -v`.
- DSL builtins (`apply_mask`, `top_k_mask`, `gather`, `slice_at`, ...) live in
  `einsum_lib/core.py` and must also be registered in
  `numba_accel.execution_namespace()`; keep the two in sync.
- Never add private benchmark reference data, credentials, endpoints or
  host-specific paths to tracked files.
