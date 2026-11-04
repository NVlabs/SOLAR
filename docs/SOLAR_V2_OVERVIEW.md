<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Why SOLAR v2 has two frontend paths

## Decision

Use mechanical graph/compiler lowering for supported programs and agentic
translation for unsupported programs. Keep one shared Einsum analysis backend.
Within the agentic path, support both a Python orchestrator and a skill-driven
coding agent. These differ in who owns the control loop, not in the required
validation evidence.

Route selection is implemented and evidence-based:
`solar.frontend.router.decide_from_graph` routes mechanical only when every
compute node is in a bounded supported set, every edge resolves, each operand
occurrence carries shape + edge, and full per-op shape semantics hold
(matmul batch/vector output recomputation, linear orientation/output/bias,
legal reductions, broadcast and shape-preservation rules); it records its
reasons/evidence (`tests/test_router_receipts.py`). The strict SOL path
additionally enforces receipts whose evidence — including the prepared
reference source actually executed and the adapter-local modules it
imported, all compiled from current source bytes through a controlled
loader that bypasses bytecode caches (the adapter/model file itself
included; executed bytes must equal the hashed source bytes) — is captured
inside the verification execution and re-derived/hash-matched again at
emission and strict entry, refusing unprovable dependency closures
(`solar.translate.receipts`); the default path remains legacy/unvalidated.

## Options considered

| Option | Benefit | Cost or limitation |
|---|---|---|
| Mechanical lowering only | Fast and repeatable for supported semantics | Unsupported operators, languages and irregular programs block coverage |
| Agentic translation for every program | Flexible source interpretation | Unnecessary inference cost, variable translations and more verification work |
| Mechanical first, validated agentic alternative | Reuse reliable lowering and spend agent effort on unsupported cases | Route selection, adapters and shared artifact contracts must remain explicit |

## Reasons and evidence

The repository already contains a PyTorch/torchview mechanical pipeline and a
separate executable `einsum_lib` path. Both feed graph analysis and hardware
prediction. Reusing that backend makes workload, precision and access assumptions
reviewable across routes. It does not prove the routes produce equivalent costs.

Execution checks catch many translation mistakes but are limited by inputs,
shape caps, reference fidelity, numerical contracts and adapters. A 28x28 window
attention check cannot exercise padding for 14x14 windows. A checker requiring
99% matched elements can accept a narrow strip of incorrect outputs. Neither
check detects a downstream compiler that omits outer-loop multiplicities.

Therefore validation has independent semantic, count/access and hardware gates.
A source-to-output pass must not be promoted into a certified SOL bound.

## Consequences

**Easier:** supported graphs avoid agent generation; unsupported cases can use
an executable reference; both agentic workflows reuse deterministic tools;
independent stage artifacts permit localized diagnosis and replay.

**Harder:** adapters must preserve native layouts and precision or disclose their
transformations; shape families require boundary cases; compiler counting and
memory models need independent tests; unsupported accounting must remain visible.

**Watch:** API fallback must be explicit, caps must not remove the behavior being
tested, original-shape sparse density must not be inferred from capped inputs,
and structural success must not overwrite numerical failures in reports.

## What must be validated

| Check | Evidence required | What a pass does not prove |
|---|---|---|
| Source contract | Shapes, dtypes, layouts, scalars, outputs, executable reference, adapter transformations | That a shape cap preserves all branches |
| Numerical semantics | Frontend-prepared inputs (PyTorch synthesizes parameters with seed 42 at native dtype), named outputs, the effective tolerance including any automatic dtype floor, DSL fp64 (or `--fp32`) lowering with fp16/bf16 output rounding; see the [numerical contract](SOLAR_V2_CONTRACTS.md#numerical-check-what-the-code-actually-does) | Untested shapes, fp64 reference arithmetic, or exact equality under a percentage-match contract |
| Graph and counts | Original dimensions, every loop extent resolved (`unresolved_ops` empty), independent MAC and scalar counts, aliases and accesses | Correct traffic or a feasible hardware schedule |
| Hardware model | Precision and throughput compatibility, bandwidth, cold or warm inputs, fusion and sparse conventions, boundary storage widths | Measured latency or implementation efficiency |

Keep failed, errored, skipped and timed-out cases in the denominator. A run
that produces a YAML file is not thereby validated. Report numeric,
structural, counting and hardware status separately; do not collapse them
into `valid`.

Two earlier campaign cases illustrate these boundaries. Window attention
tested on a 28x28 cap never exercised padding for 14x14 windows. An FP4
attention adapter used split layouts and CPU emulation; that pass did not
cover a later checker feeding native layouts. Correct numerical output can
also coexist with an incorrect loop-to-Einsum operation count.

## Sparse and irregular programs

The DSL includes gathers, index arithmetic, masks and reductions; it is not
restricted to a pure affine subset. Dense contractions with explicit masks can
represent sparse semantics, but cost analysis must separately establish which
products execute and which bytes are stored or accessed. A zero value is not
proof of zero memory traffic. MoE routing and sparse attention need
access-set, representation, dispatch and scalar-work evidence, with
original-shape density. See the
[density policy](SOLAR_V2_FLOW.md#4-choose-the-density-convention-explicitly).

## Status

This is the v2 architecture and documentation contract. Route selection is
explicit (`solar.estimate.estimate(mode="auto")` adds only the typed fallback
from the mechanical route to the API route); a universal automatic router is
not a released implementation. The mechanical route is PyTorch/torchview-based.
The Python-orchestrated and skill-driven agentic workflows exist, but neither
implies arbitrary-language or all-workload certification.

See [implementation contracts](SOLAR_V2_CONTRACTS.md),
[route selection](SOLAR_FLOW.md), and [runnable agentic flow](SOLAR_V2_FLOW.md).
