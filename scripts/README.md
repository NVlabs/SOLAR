<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOLAR scripts

Command-line tools shipped with SOLAR v2.  The mechanical pipeline itself is
driven through the `solar-*` console scripts (see `docs/SOLAR_V1_FLOW.md`); the
scripts here serve the agentic (DSL) route described in `docs/SOLAR_V2_FLOW.md`
and `skills/verified-translation/SKILL.md`.

| Script | Purpose |
|---|---|
| `verify_saved_ir.py` | Deterministic numerical re-verification of a saved `einsum_lib` DSL program against its executable reference (`--strict` for a proof-grade exit status; `--mem-budget` RSS guard). |
| `trace_saved_ir.py` | Trace a verified DSL program at the original (full) shapes with zero-memory placeholders and emit the einsum graph inputs; no model is executed. |
| `diff_ir_stages.py` | Localise a verification mismatch to the pipeline stage where it first appears. |
| `gen_solexecbench_adapter.py` | Generate a SOLAR-compatible PyTorch adapter from a user-supplied problem definition for CAPPED verification workloads (the `capping-weights` skill's `--const-override` path). No benchmark runner, datasets or benchmark configs are shipped. |
| `run_tests.sh` | Run the test suite (`all` runs everything under `tests/`; `quick` runs two smoke tests). |

Benchmark-campaign runners, result collectors and comparison
helpers used during SOLAR's development are intentionally not part of the
public distribution.
