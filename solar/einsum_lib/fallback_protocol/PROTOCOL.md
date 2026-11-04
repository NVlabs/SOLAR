<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# lite-fallback-contract-v10 PROTOCOL v6 (written before the round-10 implementation; 2026-10-28T13:03Z; wording-1 2026-10-28T13:41:22Z: fixed-point shortcut claim narrowed to dispositions)

Maintained with the shim (the fallback-contract module), 2026-10-28. Machine-readable twin: `PROTOCOL.json`. CANONICAL LOCATION: both files are TRACKED in the
package at `solar/einsum_lib/fallback_protocol/PROTOCOL.md` / `PROTOCOL.json` (resolved relative to the package, never an
author-scratch path). The shim embeds `RECORD_SCHEMA`, `INVARIANTS`,
`REEMIT` and `CONTEXT` verbatim; tests assert equality with the tracked JSON.

**Change vs v5 (the review round 9).** The context-resolution rule of v5 stands (CONTEXT-AUTHORITY-1 and OUTDIR-IDENTITY-1
closed). v6 adds three things to section 1b: (a) an explicit ADMISSION BOUNDARY for learning — values are committed only
from admissible surviving evidence, through a two-phase resolution to a deterministic fixed point, so a record the state
machine later rejects never poisons later cases or erases their valid transitions; (b) context equality is TYPE-EXACT
JSON equality; (c) the anchor domain is the enumerated engine keys and an unsupported parent-named key is refused at
descriptor construction. The disclosed non-attestation limit (section 7) is kept as stated.

## 1. Authority model (unchanged from v5)

`LaunchDescriptor` is the only authority for identity, source hash, engine pin, the parent-named engine fields, time
anchor and MODE; the disk mint is validation-only; the mode never changes a record field; standalone descriptors are a
limited identity-only mode.

## 1b. Attempt context — resolved once, enforced before admission

**Context fields**: `engine.engine_pin`, `engine.solar_package_path`, `engine.numba_accel_sha256`, `engine.verify_sha256`,
`engine.core_sha256`, `engine.python`, `engine.executable`, `engine.config`, `source_sha256`, `decline_hook_available`,
`binding.out_dir` (spelling).

**Precedence, per field:**
1. **ANCHOR** — the parent-held descriptor: `engine.engine_pin ← descriptor.engine_pin`; `source_sha256 ←
   descriptor.source_sha256`; `engine.<k> ← descriptor.expected_engine[k]` for every key the parent names. **Anchor
   domain (v6)**: the parent may name exactly the enumerated engine keys (`engine_pin`, `solar_package_path`,
   `numba_accel_sha256`, `verify_sha256`, `core_sha256`, `python`, `executable`, `config`); any other key is REFUSED at
   descriptor construction (`LaunchDescriptor` / `mint_attempt` raise `RecordInvalid`, so a controller preflight fails
   loudly); a named `engine_pin` must equal `descriptor.engine_pin`. Unnamed keys are unspecified — never a null
   requirement. "Every key the parent names" is therefore literally enforced, because only enforceable keys can be named.
2. **LEARNED** — a field not anchored is learned from the first ADMISSIBLE SURVIVING record of the attempt
   (classification order: the parent tag first, then tags in sorted order; within a tag transitions before other rows)
   and is then FIXED for every later record across all cases and stages. *Admissible surviving* = the record passed every
   per-record check AND remained in the valid set after the scope and state-machine classification of its tag; a record
   removed as `STATE_CONFLICT` / `SCOPE_CONFLICT`, or rejected otherwise, never commits a value.
3. **FREE** — only until then. The genuinely free record fields are those of section 3b (`event_utc`, failure `type` /
   `message`, fast-ran `lowered_sha256`).

**Admission boundary — two-phase resolution (v6, P2-R9-LEARN-REJECTED-1).** Learning has an explicit boundary: context
values are COMMITTED only from admissible surviving evidence; a rejected record discards its values and never poisons
later cases or erases their valid transitions.
* *Phase 1 (tentative)*: every tag is classified against the committed context extended by TENTATIVE values learned from
  individually valid records in classification order; the state machine runs; the admissible survivors of every tag are
  collected; nothing is written.
* *Phase 2 (commit)*: a fresh context = anchors + values learned ONLY from the admissible survivors, in classification
  order.
* *Phase 3 (re-validate)*: when the committed values equal the tentative values (no learner was rejected) the phase-1
  classification is already the fixed point and is final: a frozen re-validation would reach IDENTICAL admission,
  survivor and completion DISPOSITIONS (per tag: state, valid-event set, closing event, surviving transition, learner
  provenance), so none is run and the shared builder is invoked once by the validator. The DIAGNOSTICS of records
  rejected in that case are those of the tentative evaluation and may name a different code than a frozen pass would
  (e.g. `TERMINAL_INCONSISTENT:I3` where a frozen pass would report `CONTEXT_MISMATCH:engine.python` first); the
  disposition is the same (P3-R10-FIXED-POINT-WORDING-1). Otherwise every tag is classified again against the COMMITTED
  context, FROZEN (no tentative learning; unresolved fields pass); the state machine runs. If committing from the new survivors yields the same learned values,
  the resolution is at its FIXED POINT and this classification is final; otherwise phases 2–3 repeat with the new commit.
  The procedure is deterministic and bounded (4 iterations); failing to converge is recorded
  (`context_resolution.fixed_point = false`) and the attempt is non-pass (fail closed).
* *Phase 4 (write)*: joined stop / open-transition / unjoined kill records and the summary are written from the FINAL
  classification only.
Consequence: an orphan interpreter terminal (no surviving transition) is individually valid but is removed by the state
machine as `STATE_CONFLICT`; its values are discarded; a later valid transition in another case keeps its joined
stop / open-transition record. The summary carries `engine_context = {anchored, learned, learned_from, committed_from,
tentative_discarded, passes, fixed_point}`; the same set of record files yields the same committed context and the same
final classification.

**Type-exact equality (v6, P2-R9-CONTEXT-TYPE-1).** Context equality is TYPE-EXACT JSON equality: recursive over dicts
(same key set) and lists (same length, element-wise); scalar values must have the same JSON type (bool is never equal to
int: `1` is never `true`, nested `[true,false]` is never `[1,0]`); no implicit normalisation — the same discipline as the
re-emit comparison. A transition contradicting an anchored or learned value by type alone is `CONTEXT_MISMATCH` and is
never admitted, so the shared builder never inherits the altered value.

**Enforcement point** (unchanged): after the schema / event / state-row / scope / identity checks (identity, incl. the
canonical out_dir rule, reports first) and before the time and re-emit checks, for EVERY row. Codes: `ENGINE_PIN_MISMATCH`,
`SOURCE_MISMATCH`, `ENGINE_HASH_MISMATCH` (legacy names), `CONTEXT_MISMATCH:<field>` otherwise. A rejected record never
closes and never suppresses the joined stop / open-transition / unjoined kill record.

**Out_dir rule** (unchanged from v5): `canonical_run_dir(record.binding.out_dir) == record.binding.run_dir == descriptor.run_dir`
before validity for every row (`IDENTITY_MISMATCH`); aliases pass; the spelling is a learned context field.

**Case-specific configuration**: NONE is declared.

## 2. Scopes and required-case coverage (unchanged)

## 3. Event rows and cross-field invariants I1–I10 (unchanged from v4/v5)

## 3b. RE-EMIT AND COMPARE (closing records) — accepted method, unchanged from v5

`expected_closing_record(transition, context, outcome_kind, free)` is the only builder; the validator rebuilds and
requires equality on every deterministic field (`TERMINAL_INCONSISTENT:<field>`); the fast-ran context comes from the
RESOLVED (committed) context; free fields: `event_utc`, failure `type` / `message`, fast-ran `lowered_sha256`.

**Precedence of diagnostics**: scope and identity first, then the resolved context, then the time anchors, then
cross-field I1–I10, then the re-emit comparison.

## 4–6. State machines and attempt level (unchanged)

## 7. Guarantees

* Joined record whenever a valid transition survives unclosed; unjoined kill record whenever a guard kill has no valid
  surviving transition and no consistent completion; nothing manufactured from malformed input.
* Re-emit and compare: a closing record not equal, field for field, to the emitter's record never establishes completion.
* Context resolution: one resolved context, enforced on every row before admission; unnamed parent keys never null
  requirements; the expected record built from the resolved context.
* **Admission boundary (v6)**: learned values are committed only from admissible surviving evidence; rejected records
  never poison later cases nor erase their valid transitions; the final classification is reached at a deterministic
  fixed point and records are written only then; context equality is type-exact.
* **Stated limit (kept)**: a structurally valid FIRST record, or a consistently built first transition/terminal pair, can
  claim unobserved values for every UNANCHORED context field; the validator records them as learned and cannot tell them
  from the emitted bytes. This is the disclosed consistency-only, non-attestation limit: the contract establishes
  consistency of the evidence, not attestation of the child; a controller that wants independent proof supplies
  `expected_engine` for the fields it knows; the comparator owns the verdict (always null in these records).

## 8. Type schema and validation (unchanged)

## 9. Harness and controller obligations

API unchanged except that an unsupported `expected_engine` key is now refused at construction. The delivered 229
controller candidate selects `fallback-allowed`, holds the descriptor in memory and supplies no `expected_engine`, so the
engine context of its attempt is learned from the first admissible surviving record (stated). Controller v10 differs
from v9 only in the shim/harness hash bindings and the version label.

## 10. Tests derived

Two-phase learning (orphan then valid; valid then orphan; multi-case; kills during each phase; the 18 individually
rejected-first controls; determinism; summary provenance fields); type-exact context per field (parent anchor and learned
case, nested values, clean and killed; the reviewer's typed-config cases); anchor domain (unsupported key refused at descriptor
and mint; pin must match; every enumerated key accepted); the reviewer's round-9 probes (36 cases); the retained 457 tests;
schema / invariants / REEMIT / CONTEXT equality with the tracked JSON.
