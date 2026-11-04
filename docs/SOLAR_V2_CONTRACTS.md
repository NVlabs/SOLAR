<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOLAR v2 implementation contracts

This page separates implemented entry points from release requirements.
Code pointers refer to the shipped source; test evidence belongs to
its exact run/commit, not to all future checkouts.

## Frontend contracts

| Stage | Implementation | Input | Output and limitations |
|---|---|---|---|
| Mechanical graph extraction | [process_model](../solar/cli/process_model.py) | PyTorch source/model and inputs | torchview graph; not a general compiler for every source language |
| Mechanical lowering | [toeinsum_model](../solar/cli/toeinsum_model.py) | Supported graph/operator semantics | Einsum graph; unknown-op API handler generation is optional and does not add source execution validation |
| Agent workspace | [scaffold and translate](../solar/translate/__init__.py) | Problem directory/source contract | Adapter stub/checklist; caller or coding agent must author the DSL |
| Python agentic flow | [toDSL_model](../solar/cli/toDSL_model.py), [pipeline](../solar/einsum_lib/pipeline.py) | Source, adapter/input configuration, explicit inference configuration | Candidate generation, checks and repair; failures remain failures |
| Saved DSL verification | [verify runner](../solar/translate/_runner.py) | Executable reference, DSL, actual test inputs and tolerance | Verdict/error for those inputs; capped or emulated contracts must be disclosed |
| Original-shape trace | [trace_saved_ir](../scripts/trace_saved_ir.py) | DSL, definition, selected workload, optional custom layout inputs | Structural metadata from placeholders; not a numerical result |
| DSL compiler | [cli_einsum](../solar/einsum_dsl/cli_einsum.py) | DSL source and trace metadata | Einsum graph; source loop multiplicities need independent regression checks |
| Shared analysis | [analyze_model](../solar/cli/analyze_model.py) | Einsum graph with dimensions/roles/precision | Work and memory model; non-MAC and irregular-access coverage must be examined |
| Hardware model | [predict_perf_model](../solar/cli/predict_perf_model.py) | Analysis and architecture/precision settings | Conditional SOL estimate, not a measured schedule |

## Workflow ownership

In the Python-orchestrated flow, Python calls `TranslationAgent` and owns the
bounded generation/verification/repair loop. In the pure skill flow, the coding
agent owns that loop and invokes deterministic tools. A human may inspect or
reproduce the same commands. Both require the same source fidelity and evidence.
The skill route does not mean the surrounding coding agent uses no model tokens.

The CLI facade exposes `scaffold`, `verify`, `sol`, and explicit `api` commands.
`sol(..., strict=True, receipt=..., source_path=...)` ENFORCES the receipt
gate before any tracing/analysis. Receipt evidence is bound INSIDE the
verification execution: `run_verification` captures the SHA-256 of the exact
source/DSL bytes it imported and executed, the PREPARED REFERENCE source it
actually compared against (what `get_source()` returned, or the pytorch
model file text), the tolerance it applied, the checker identity (module +
file hash), the numeric mode and the validation tensor shapes into a sealed
`ExecutionEvidence` that callers cannot construct. `emit_receipt` refuses a
result without that evidence and refuses to emit when the current file bytes
differ from the verified bytes — a passing `VerifyResult` cannot be rebound
to subsequently changed code. Both emission and strict entry also RE-DERIVE
the current reference through the same frontend contract and compare it: an
adapter whose `get_source()` reads an external `reference.py` (the
documented normal pattern) cannot keep an old receipt alive after that file
changes, and a reference that cannot be re-derived refuses strict
qualification instead of presuming it. The reference's LOCAL EXECUTION
DEPENDENCIES are bound the same way: modules imported during reference
preparation whose files live in the adapter's directory tree are hashed into
the evidence and re-checked at emission and strict entry (so
`from helper import run` cannot outlive a changed `helper.py`), while
modules newly imported from outside the supported closure (adapter
directory, Python environment, solar checkout) make the closure unprovable
and refuse strict qualification. The hashes describe the code that actually
ran: cached modules outside the trusted environment/checkout are dropped
before each reference preparation (a fresh, controlled import snapshot,
recorded as `evicted_cached_modules`), and untrusted `.py` imports are
loaded through a controlled loader that compiles the current source bytes
directly — bytecode caches are bypassed, so neither a cached module object
nor a timestamp-valid stale `.pyc` can execute old code bound to new
hashes. The MAIN adapter/model file goes through the same source-bytes
loader: every supported frontend's direct module load (cuda, triton,
pytorch, jax — `solar.einsum_lib.source_load`) compiles the current source
bytes rather than consulting `__pycache__`, and the runner additionally
refuses strict qualification unless the adapter's executed-bytes hash
EQUALS the source hash it bound — a load path that skips the audited
loader, or a file changed mid-run, cannot qualify. Receipt emission and
strict entry re-derive the reference through the same source-bytes loader,
so a stale adapter `.pyc` cannot resurrect an old `get_source()` either.
The recorded dependency hash is of the bytes handed to `compile()`;
a local dependency whose executed code cannot be tied to source bytes
(sourceless bytecode, extension modules, manual loading) refuses strict
qualification.
`check_strict` then recomputes current hashes at strict-SOL entry and
compares the exact workload (uuid/axes) and the SELECTED analysis precision
(dtype, always compared by `sol`); `expected_tolerance`/`expected_checker`
pin the recorded policy field-for-field when stated. Missing, malformed,
failed, checker-error, stale, rebound or mismatched receipts raise
`ReceiptRefusal` and nothing runs (`tests/test_router_receipts.py`).
Receipts record numerical verdicts WITH error stats separately from
structural/accounting/hardware verdicts, which stay `not_certified` — a
numerical pass does not certify MAC/access counts or hardware calibration.
Remaining stated limits: the definition.json hash is bound at emission (the
verifier does not read it); modules under the Python environment and the
solar checkout are trusted as environment, not hashed per-receipt (the same
trust verification itself needs to run torch and the checker); and the gate
defends against stale/rebound artifacts, not against an adversary editing
the gate module itself. The
DEFAULT `sol()` (strict=False) remains the legacy path: it performs no
numerical validation and its outputs must not be presented as validated.

`solar.frontend.router.decide_from_graph` routes an extracted graph
`mechanical` only on positive evidence: every compute node in a BOUNDED
supported set (`SUPPORTED_OPS` — not registry presence), with graph
integrity (every input/output edge resolves to an existing node; well-formed
positive-int shapes, 0-d scalars allowed), complete per-occurrence operand
accounting (edge count equals operand shape count, so repeated operands stay
valid, and expected operand arity holds), and bounded per-op semantic
evidence: the FULL torch.matmul output shape is computed and compared
(vector cases and batch broadcasting included, not just the final two dims);
linear uses the explicit `(out_features, in_features)` weight orientation
with output and bias validated, and its operand roles must be unambiguous —
exactly one input, one weight, at most one bias, every recorded operand used
exactly once (extra, duplicate or uncovered role tags route agentic, never
silently drop); sum outputs must be a legal reduction of the
operand (full, keepdim, or removed dims); add/mul operands must broadcast to
the recorded output; relu/softmax preserve shape; reshape-family ops
preserve element count; transpose/permute preserve the dim multiset.
Anything else routes
`agentic` (Pythonic or pure-skill, both preserved) or `unsupported`, with
recorded reasons and evidence carried in the decision; it is offline and
deterministic, reads no credentials and calls no paid APIs. Mechanical graph
artifacts also carry `provenance` (source sha256 + capture-implementation
identity): resume reuses an artifact only when both match, so a stale pre-fix
artifact can never turn an unsupported source into supported output.

### Campaign-audit repairs (accounting/billing contracts)

`sol()` binds the benchmark definition's declared INPUT/OUTPUT storage
dtypes as the authoritative boundary-traffic widths: analysis records
per-tensor provenance (`metadata.boundary_storage`, definition vs trace
evidence; declared outputs are bound to graph output tensors by resolved
element count, refusing ambiguous widths; declared shapes resolve through
the bounded definition shape-expression grammar, so `H*1` bills like `H`
while unknown axes keep loud errors) and the perf model bills the
boundary component of every traffic class at those widths
(`boundary_billing: declared_storage_dtypes`). The per-graph precision
scalar remains the compute/representation choice (MAC-rate selection,
including FP8/NVFP4 quant throughput, is unchanged); a boundary tensor
whose width cannot be established refuses — storage types are never
invented. A PUBLIC external input binds to a declared definition dtype
either directly by name or through an explicit trace-to-definition
`boundary_input_binding`; an aliased public input without such a binding
refuses rather than falling back to its verification/trace dtype, and a
dangling binding (to a name the definition does not declare) refuses
regardless of the tensor's name. Internal constants are identified only
by RECORDED provenance: `einsum_lib.metadata.build_metadata` tags each
tensor's actual origin (`registered-input` for anything from the public
inputs dict — including one a user legally names `__input_0`, since the
`__<role>_<n>` spelling is a valid identifier, not a reserved namespace —
vs `internal-constant` for trace-created tensors), the exporter carries
it into the graph's `tensor_origins` map, and only a recorded
`internal-constant` keeps its trace dtype as storage evidence; a name
prefix alone is never provenance (campaign reviews C12-P1/C12-P2,
C19-P1). DSL kernel loop extents built from integer expressions
(including floor-divided shapes) resolve exactly when the call's recorded
inputs provably correspond to the kernel's tensor parameters (the reviewed
correspondence guard, the internal converter); an explicit loop nest whose extent cannot
be established is REFUSED at conversion time
(`einsum_dsl/einsum_converter._kernel_loop_iterations` raises) rather than
undercounted. There is NO `loop_count_unresolved` marker and the analyzer
emits no `total.unresolved_ops` on this branch: the older
unresolved-reporting implementation was dropped in favour of the
fail-closed converter; the perf stage's refusal of a `total.unresolved_ops`
field is retained as an inert guard. The standalone-index fallback refuses
kernels with compound subscript indices. Floor-partition extents as a reuse
hazard (`floor_partition_extent`) are delivered by a separate change
(`solar/translate/reuse.py`); this branch depends on it for
`tests/test_campaign_repairs.py::TestWindowCounts::test_floor_partition_is_a_reuse_hazard`.

Every external read group the analyzer classifies takes part in both declared totals, including unnamed groups: a gather index recovered from the call attributes is billed at the dtype the attributes recorded for it; an unrecorded dtype is billed at its trace width and marks the layer `incomplete` (reason `boundary_width_unresolved`), never a silently smaller total. `total.boundary_declared` records what the totals cover (`elements_raw`, `elements_dedup`, `elements_raw_weight`, `bytes_raw_model_io`, `bytes_raw_weight` for the raw reads; `footprint_elements_weight`, `bytes_footprint_weight` for the parameter footprint of the declared weight inputs); the perf model takes its traffic re-pricing delta against the covered read elements, so a declared boundary input the converter typed `weight` (reported under `weight_elements` since an earlier decision) is billed exactly once, and re-prices `weight_bytes` on the footprint basis only (distinct parameter per consuming op x full numel x declared width), never on read counts.

PENDING (disclosed scope exclusion, NOT closure or supersession): the
semantic constant-binding checks for the full-shape trace (L2/058) are not
wired on this branch. `einsum_dsl/constants_binding.py`
(`verify_constant_bindings`, `ConstantBindingError`) is carried as a helper
with no production caller, and `scripts/trace_saved_ir.py` has no binding
hook, so `_AXIS_BINDINGS` entries are NOT checked against the definition's
values, `_FREE_CONSTANTS` is not consulted, no `constant_provenance.json` is
written, and a baked capped constant (`_G = 2` against `n_groups=8`) does
NOT refuse the trace. The four `TestConstantBindings` cases that exercise
these behaviours are marked pending (strict xfail) in
`tests/test_campaign_repairs.py`; the three helper-level cases pass.
Full-shape campaign acceptance cannot rely on these checks until a
follow-up wires them. These repairs do NOT regenerate or revalidate any
historical campaign export.

## Result contract

Each result should carry the source/adapter/DSL hashes, workload ID and original
axes, validation axes and seeds, dtype/layout/scalar transformations, tolerance
and error statistics, compiler revision, architecture hash, selected throughput
precision, density/access assumptions, commands, producer and failure statuses.
This is a reporting requirement; existing legacy results do not uniformly emit
all fields and must be audited before publication.

Keep four independent verdicts: numerical semantics, graph structure, work/access
accounting, and hardware-model scope. A timeout or checker error is not a pass
and is not automatically a demonstrated numerical mismatch. A numerical pass
under a 99%-matched contract is not an all-element equivalence proof. Adapter
emulation and native execution are separate evidence classes.

## Release acceptance and present gaps

- Exercise both frontend routes with reproducible smoke examples and check
  outputs/commands against the actual implementation.
- Cover shape boundaries, masks, routing and native-to-adapter layout transforms.
- Independently test operation counts, storage/access sets and precision choice;
  report unresolved operations rather than granting them zero cost.
- Preserve full denominators, failed attempts and original artifacts.
- Label manual routing and unenforced verification receipts honestly until those
  mechanisms are implemented and tested.
- Do not advertise new mechanical compiler frontends, universal language support,
  direct DSL Timeloop export or a fully calibrated sparse SOL without evidence.

The existing facade tests are in
[test_translate_facade.py](../tests/test_translate_facade.py). They exercise
scaffolding, saved verification and the downstream chain; their success does not
certify the benchmark corpus or the missing automatic router.

## Findings from the documentation smoke audit

The mechanical commands completed for `examples/Matmul/Matmul.py`, but the
result was not accounting-valid: the model's parameter has shape [4,32,64],
its explicit input [64,128], and expected work 1,048,576 MACs. The extracted
graph retained only the explicit input; the downstream result reported
262,144 MACs and omitted the parameter's 8,192 elements. Successful command
execution did not establish supported lowering for this case. This was a
release-blocking example/accounting defect, separate from the Lite window fix.

REPAIRED (branch `solar_verified_mechanical_recovery_20261021`): the cause was
that bare `nn.Parameter` operands of function calls are dropped by an
unpatched torchview — the repo's `patches/torchview-parameter-tensors.patch`
was never applied to the installed package, so the frontend's
`parameter-tensor` support was unreachable in a clean environment.
`solar/graph/_torchview_param_shim.py` now applies the same behavior at
import time (no-op when the installed torchview is already patched).
`examples/Matmul/Matmul.py` yields 1,048,576 MACs and 32,768 unfused elements;
`tests/test_learned_param_matmul.py` pins the learned-parameter cases
(weight-first, weight-second positional order) and the unchanged explicit
two-input path. Scope: direct `nn.Parameter` operands only — derived views
(e.g. `self.weight.t()`) are still untraced constants, and this repair does
not by itself make the corpus fully validated.

Consequently, the presence of a registered operator handler is necessary but
not sufficient to call a graph supported. Check operand capture, broadcast
geometry and independent counts for the selected workload before trusting
mechanical lowering. Do not describe the current corpus as fully validated.
