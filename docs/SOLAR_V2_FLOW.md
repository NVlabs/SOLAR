<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOLAR v2 agentic frontend: verification and original-shape SOL

This is one of the [two frontend paths](SOLAR_FLOW.md). Supported programs can
use [mechanical lowering](SOLAR_V1_FLOW.md) without agent generation. This guide
covers the agentic alternative and its two control flows:

- **Skill-driven:** the coding agent follows the skill, authors the adapter/DSL,
  invokes the verifier, and repairs failures. `translate scaffold` only creates
  a workspace; it does not launch that agent.
- **Python-orchestrated:** `solar.cli.toDSL_model` / `TranslationAgent` (or
  explicit `translate api`) own generation, verification and bounded repair.
  Inference API access is an explicit choice.

Either flow can emit a receipt from its passing `verify()` run
(`solar.translate.receipts.emit_receipt`): the hashes — including the
prepared reference source the run actually executed (e.g. what
`get_source()` read from an external `reference.py`) and the adapter-local
modules that reference imported — tolerance, checker identity and validation
shapes are captured inside the verification execution itself (cached
non-environment modules are dropped before each run, and untrusted imports
AND the adapter/model file itself are compiled from current source bytes,
bypassing bytecode caches — the runner refuses strict qualification unless
the adapter's executed bytes equal the hashed source bytes, so the
recorded hashes describe the code that actually executed); emission and
strict entry re-derive the current reference through the same source-bytes
loader, re-hash the dependencies, and
refuse bytes changed after the run (an unprovable dependency closure refuses
strict qualification), and
`sol(..., strict=True, receipt=..., source_path=...)` refuses stale, rebound,
mismatched, failed or missing receipts (including a precision different from
the receipt's dtype) before any tracing. The plain `sol` default stays
legacy/unvalidated.

Both use shared execution checks and the DSL-to-Einsum compiler. The caller must
preserve the validation gate: `translate sol` does not enforce a prior verified
receipt. See [implementation contracts](SOLAR_V2_CONTRACTS.md).

Run from the repository root, with `PY` pointing to Python 3.10+ and Solar installed. See
[AGENTS.md](../AGENTS.md) for x86_64 versus aarch64 environments; an x86 conda
interpreter cannot run on a Grace host.

## 1. Read the source and establish a reference

Read the implementation, workload axes, shapes, dtypes, and output contract.
For PyTorch, pass a model defining `Model` and `get_inputs()` directly. Other
frontends use an adapter supplying `get_source()`, `get_inputs()`, and
`kernel_fn(inputs) -> {output_name: numpy_array}`. The golden must represent
the actual mathematical contract; source names and comments are insufficient.
CUDA/cuTile source can be translated without executing its GPU implementation
when an appropriate executable reference is available.

Cap large inputs for verification while preserving divisibility, routing,
grouping, and architectural relationships. Record caps and keep original
workload dimensions separately. All-zero outputs can make a comparison
vacuous; inspect the verifier's warnings.

## 2. Author and numerically verify the DSL

The default workflow has the coding agent author `@kernel` loop functions and
`run_model(inputs)`. Its deterministic SOLAR tools make no inference API call; the surrounding coding agent may use model tokens. Use named `Tensor`/`Dim`
dimensions, supported builtins, and shape-derived extents. Follow the
[verified-translation skill](../skills/verified-translation/SKILL.md) and its
[language/reference guide](../skills/verified-translation/reference.md).

The shipped RMSNorm example supports a complete deterministic smoke run:

```bash
PY=python3
M=skills/verified-translation/example
OUT=$(mktemp -d /tmp/solar-v2-rmsnorm.XXXXXX)

$PY "$M/selfcheck.py" "$M"
$PY scripts/verify_saved_ir.py "$M/adapter.py" "$M/solar_model_einsum_lib.py" \
    --source-type cuda --mem-budget 30
```

Require `VERIFIED` and a successful exit. The example verifies at B=8, H=128
against a PyTorch reference, then traces the included illustrative original
workload B=24, H=128. `--source-type cuda` selects the adapter contract here;
this smoke test does not require GPU execution. For a PyTorch model instead:

```bash
$PY scripts/verify_saved_ir.py model.py model_einsum_lib.py \
    --source-type pytorch --mem-budget 30
```

Default tolerances are rtol=1e-4, atol=1e-5. Quantized or narrow-output cases
need a documented tolerance choice. Verification tests the chosen inputs; it
does not prove all inputs, operation counts, or hardware performance.

## 3. Trace original shapes and run the shared backend

Continue the same shell session:

```bash
$PY scripts/trace_saved_ir.py "$M/solar_model_einsum_lib.py" \
    --definition "$M" --out-dir "$OUT/dsl" --fallback-adapter "$M/adapter.py"
$PY -m solar.einsum_dsl.cli_einsum "$OUT/dsl" --output-dir "$OUT/einsum"
$PY -m solar.cli.analyze_model \
    --einsum-graph-path "$OUT/einsum/solar_model_einsum_graph.yaml" \
    --output-dir "$OUT/analysis" --no-copy-graph
$PY -m solar.cli.predict_perf_model --analysis-path "$OUT/analysis/analysis.yaml" \
    --output-dir "$OUT/perf" --arch-config B200 --precision fp16
```

Inspect `dsl/*_einsum_lib.yaml` to confirm original dimensions, then
`analysis/analysis.yaml` and `perf/perf_B200.yaml`. The trace uses broadcast
placeholders, so full-size tensors need not be allocated. `--definition`
expects `definition.json` plus an optional `workload.jsonl`; select the row
with `--workload-idx`. Set unresolved axes explicitly with `--axis NAME=VALUE`
instead of relying on the script's default variable-axis size of 4.

Custom index arrays or decomposed layouts can use
`--custom-py full_inputs.py`, defining `full_inputs(axes)`. A fallback adapter
cannot safely supply capped arrays whose shapes depend on changed axes.
Do not use verification-scale graphs to predict original workload latency.

**Timeloop limitation:** direct export of a v2 DSL graph is not currently a
supported step. The existing exporter expects v1 `shapes` and connection-based
tensor identities; v2 emits `tensor_shapes`, explicit `tensor_names`, and a
synthetic start node. The RMSNorm graph currently fails export on repeated
start-node ranks. Merely skipping that node produces missing bounds and
incorrect intermediate identities. Use the [v1 export path](SOLAR_V1_FLOW.md)
for compatible PyTorch models; v2 needs a separately validated schema adapter
before its exported workloads can be called runnable. No Timeloop simulation
or hardware mapping is performed by the v2 smoke flow above.

## 4. Choose the density convention explicitly

The YAML field `sparsity` stores **density**: `count_nonzero / size`. A value
of 0.25 means 25% nonzero. This can discount eligible multiplicative work and
tensor bytes; it does not prove a GPU kernel skipped those instructions or
stored a compressed representation.

| Mode | Flags on `trace_saved_ir.py` | Behavior |
|---|---|---|
| Default | none | No measured-density pass; explicit/analytic sparse annotations remain |
| Measured, attention guarded | `--measure-density` | Profile real inputs and outputs; recognized attention probabilities stay dense |
| Measured attention too | `--measure-density --measure-attention-density` | Allow attention-probability measurements; changes the dense-attention convention |
| Deliberate per-tensor density | `--density-override 'probs#0=0.25'` | Force this trace tensor's density, including a guarded one |

Append the desired flags to step 3's trace command, use a fresh output tree,
and rerun conversion, analysis, and performance prediction. Measurement needs
`--fallback-adapter`, real values, and additional compute. Tune
`--measure-timeout` (default 300 seconds) and `--mem-budget` (default 30 GB).

The attention guard recognizes tokenized `softmax`, `prob`, `probs`,
`probabilities`, `attn_weights`, and `attention_weights` names. It is a
conservative naming heuristic, not a semantic classifier. Mark opaque output
tensors with `role="attention_probability"` to protect them explicitly:

```python
P = Tensor(B=X.shape[0], S=X.shape[1], T=X.shape[1],
           role="attention_probability")
```

The role must be on the constructed/returned tensor, not only its annotation.
For input probability tensors, use a recognizable input dictionary key or the
same role on `Tensor.from_numpy`. Arbitrarily named unmarked tensors cannot
be recognized automatically. Softmax naming also conservatively guards a
router softmax; analytic `top_k_mask` densities and MoE routing remain usable.
The guard logs `GUARDED` when it suppresses a sparse measurement. Explicit
`sparsity=` annotations and `--density-override` remain deliberate overrides.

The manual trace applies this precedence:

1. Explicit per-tensor override wins.
2. Analytic density from the original-shape trace is preserved (e.g. top-k k/E).
3. Measurements with matching tensor shapes are copied.
4. Changed-shape measurements are remeasured at original shapes if the
   estimated footprint fits the memory budget and execution completes.
5. If revalidation fails or exceeds the estimate/timeout, keep the tensor dense
   and log `REFUSED` with an override remedy.

The same attention policy applies to both measurement passes. Shape agreement
is evidence about scale, not a guarantee of representative data: the current
original-scale pass synthesizes floating inputs with a fixed random seed.
For data-dependent sparsity, inspect representativeness and prefer justified
analytic overrides. The manual memory budget is an allocation estimate
(3 × fp64 traced footprint), not a hard RSS cap on the measurement pass.

## 5. Optional API-generated translation

Only choose this path when API generation is intended and credentials are
available. The endpoint is `SOLAR_API_BASE_URL`; the model is `--llm-model` or
`SOLAR_LLM_MODEL` (no default ships); the key is `--api-key`,
`--api-key-file` or `OPENAI_API_KEY` (no implicit key file); never commit
credentials.

```bash
$PY -m solar.cli.toDSL_model model.py --source-type pytorch \
    --output-dir output/v2_api/dsl --cap-input-size \
    --mem-budget 10 --verify-timeout 300 --measure-density
# To include probability measurements, also add --measure-attention-density.
$PY -m solar.einsum_dsl.cli_einsum output/v2_api/dsl \
    --output-dir output/v2_api/einsum
```

Continue with the shared analyze/predict commands using the emitted
graph name. The API path supports pytorch/raw/jax/triton/cuda source types.
`--no-verify` cannot be combined with `--measure-density`.

After successful verification, density mode runs a separate interpreted
full-value trace under the verifier's timeout/RSS guard. This is needed even
when Numba performed verification. Analytic metadata stays authoritative;
only measurements with matching original tensor shapes are exported.
Changed-shape measurements are refused; use `trace_saved_ir.py` with a
definition and adapter for the full original-scale ladder. Profiling failure
does not invalidate a numerical verification, but the output then has no
measured-density discounts. Keep that warning in the run record.

## Implementation map

| Responsibility | v2 implementation |
|---|---|
| Manual numerical verification | `scripts/verify_saved_ir.py`, `solar/einsum_lib/verify.py` |
| API orchestration / source preparation | `solar/cli/toDSL_model.py`, `solar/einsum_lib/pipeline.py` |
| API generation, retry, verify, trace | `solar/einsum_lib/agent.py` |
| Tensor/Dim semantics, tracing, density guard | `solar/einsum_lib/core.py` |
| Numba lowering and execution | `solar/einsum_lib/numba_accel.py` |
| Original-shape trace and density trust ladder | `scripts/trace_saved_ir.py` |
| Metadata and same-shape density merge | `solar/einsum_lib/metadata.py` |
| DSL loading, AST parsing, graph/einsum conversion | `solar/einsum_dsl/` |
| Analysis and performance | Same backend as [v1](SOLAR_V1_FLOW.md) |

The optional `solar.einsum_dsl.cli` emits topology/loop expressions without
einsum equations. `cli_einsum` directly emits the graph needed by analysis;
running the topology-only stage first is unnecessary.

## Validation and batch work

```bash
$PY -m pytest tests/test_density_guard.py tests/test_sparse_cost.py \
    tests/test_skills.py tests/test_einsum_lib.py tests/test_numba_accel.py \
    tests/test_graph_analyzer_ordering.py tests/test_perf_quant.py -q
```

These checks cover density modes, same/original-scale tracing, analytic and
explicit overrides, MoE accounting, API metadata with mocked generation,
DSL semantics, Numba, and the shared backend. They make no inference API calls.
The example above additionally exercises the runnable end-to-end CLI flow.

Batch campaign runners, result collectors and reference-comparison helpers used
during development are not part of this distribution (see `scripts/README.md`).
Run one official verification at a time, keep a separate output tree per
problem set and never run two writers against the same tree.

Use RSS guards for numerical verification, not `ulimit -v`. A failed
verification or `UNVERIFIED` file is a debugging artifact and must not enter
a verified results table. After numerical changes reverify; after counting
or density changes regenerate analysis and predictions, even if numerics are
unchanged.
