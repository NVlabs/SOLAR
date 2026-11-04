<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Verification Guide

This page names the verification entry points that exist in this tree and
records the retirement of the earlier kernelbench einsum-graph verifier that
this page used to describe.

## Current entry points

Numerical verification in SOLAR checks an executable `einsum_lib` DSL against
an executable reference. The inputs are the DSL file and its source or
adapter, not a mechanical-route `einsum_graph.yaml`.

| Entry point | What it is | Where it is documented |
|---|---|---|
| `solar-translate verify` (`python -m solar.cli.translate verify`) | "verify a saved DSL against its source (library form of `scripts/verify_saved_ir.py`; same flags, same exit status: 0 verified, 1 mismatch/failure, 2 usage/missing file, 3 only with `--strict`/`--emit-receipt` when no numerical proof can be delivered)" -- quoted from `solar-translate --help` | [docs/SOLAR_V2_FLOW.md](SOLAR_V2_FLOW.md), [docs/V2_FLOW.md](V2_FLOW.md) |
| `scripts/verify_saved_ir.py <model-or-adapter.py> <dsl.py> --source-type ...` | the manual route's verifier; success line `VERIFIED (max_abs_error=...)`; tolerance classes and what exit 0 does not prove are listed in the skill | [skills/verified-translation/SKILL.md](../skills/verified-translation/SKILL.md) step 5, [reference.md](../skills/verified-translation/reference.md) |
| `solar/einsum_lib/verify.py` (`execute_translated`, `compare_outputs`, `VerificationResult`) | the library the two commands above run | implementation map in [docs/SOLAR_V2_FLOW.md](SOLAR_V2_FLOW.md) |
| receipts (`--emit-receipt`, `sol --strict`) | bind a numerical verdict to the bytes that ran | [docs/RECEIPTS_AND_VALIDATION.md](RECEIPTS_AND_VALIDATION.md) |

Run one official verification at a time and use RSS guards (`--mem-budget`),
not `ulimit -v`; see [docs/SOLAR_V2_FLOW.md](SOLAR_V2_FLOW.md).

## Retired: the kernelbench einsum-graph verifier

Earlier revisions of this guide described `python -m solar.cli.verify_einsum`,
`python -m solar.cli.verify_einsum_model` and the
`einsum_verification/einsum_verification.yaml` result files they wrote. Those
commands executed a mechanical-route `einsum_graph.yaml` through the separate
`solar_verifier` project (`verify.run_benchmark_test`), which is not part of
this tree and was
never part of the installed `solar` package. With that checkout absent,
`verify_model_output()` raised `ModuleNotFoundError: No module named 'verify'`
before validating its input, so the commands could not run.

On 2026-11-04 the two CLI modules, their `sys.path` bootstrap were
removed, and the `Verification:` output lines that the mechanical examples'
`run_solar.sh` scripts printed for the never-written result file were dropped.
This is a removal, not a replacement: the entry points above verify an
`einsum_lib` DSL, and no command in this tree verifies an `einsum_graph.yaml`
against a PyTorch reference.
