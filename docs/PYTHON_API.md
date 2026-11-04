<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Python API

SOLAR exposes four Python surfaces. All run offline unless stated.

| Surface | Module | Use it for |
|---|---|---|
| Unified source-to-SOL | `solar.estimate` | One call from a source file to a SOL result with explicit mode, provenance and failure semantics |
| Agentic facade | `solar.translate`, `solar.translate.receipts` | Workspace scaffolding, saved-DSL verification, receipts, original-shape SOL against a benchmark definition |
| Route decision | `solar.frontend.router` | Evidence-based mechanical-versus-agentic decision on an extracted graph |
| Mechanical stage classes | `solar.graph`, `solar.einsum`, `solar.analysis`, `solar.perf` | Driving the mechanical stages from Python |

## `solar.estimate.estimate`

```python
from solar.estimate import estimate, EstimateResult
from solar.estimate import UnsupportedError, ValidationError, PipelineError, AgentTranslationRequired

r: EstimateResult = estimate(
    source_path, out_dir, *,
    mode="auto",            # auto | v0 | v1 | v2
    source_type="pytorch",  # pytorch | cuda | jax | triton
    arch="B200",            # name under configs/arch/ or a path to a yaml file
    precision="fp16",
    ir_path=None,           # v2 only: the agent-authored *_einsum_lib.py
    api_key_file=None, llm_model=None, max_retries=3,   # v1 only
    timeout=300, mem_budget_gb=16, rtol=1e-4, atol=1e-5)
```

CLI: `python3 -m solar.cli.estimate SOURCE --out-dir OUT [--mode ...] [--source-type ...]
[--arch ...] [--precision ...] [--ir-path ...] [--api-key-file ...] [--llm-model ...]
[--max-retries N] [--timeout S] [--mem-budget-gb G]`. It prints the result as JSON.

### Modes

| Mode | What runs | LLM call | Numerical gate |
|---|---|---|---|
| `v0` | Mechanical frontend: `PyTorchProcessor` -> `PyTorchToEinsum(enable_agent=False)` -> analysis -> perf | never | none (`validation.kind = "mechanical"`, `numeric_comparison = false`) |
| `v1` | `solar.einsum_lib.pipeline.translate_file` (the `toDSL_model` pipeline) with `cap_input_size=False`, fp32 lowering, then the shared validated-IR path | yes, requires a key (`api_key_file` or the resolver in `agent.py`) | yes |
| `v2` | The supplied `ir_path` through the shared validated-IR path | never | yes |
| `auto` | `v0`; on `UnsupportedError` only, `v1` | only on fallback | as the mode used |

The supported-or-not decision in `auto`/`v0` is the converter's own
per-layer supportability (`is_einsum_supportable` and an einsum equation for
every layer) plus a bounded list of tracing failure messages. It does **not**
call `solar.frontend.router`; the two can disagree (see
[route selection](SOLAR_FLOW.md)). `v0` rejects non-PyTorch source types and
files without `Model`/`ReferenceModel` and `get_inputs()`.

The shared validated-IR path (`v1`, `v2`) prepares the reference through the
same per-source frontend as the saved verifier, **refuses capped
validation** (the prepared shapes must equal the verification shapes),
executes the IR with fp32 lowering, compares with `rtol`/`atol` raised to the
frontend's dtype floor, then traces the IR shape-only for metadata and
compiles it. The SOL is therefore computed for the reference's own
`get_inputs()` case; `estimate` has no notion of a benchmark definition or
workload row. Use `solar.translate.sol` when the analysis workload differs
from the verification inputs.

Mode `v2` without `ir_path` writes `AGENT_TASK.md` in the run directory and
raises `AgentTranslationRequired`. At this base that packet tells the agent
to follow `skills/solar-estimate/SKILL.md`, which does not exist in the
checkout; follow `skills/verified-translation/SKILL.md`, write the DSL
against the reference named in the packet, and call `estimate(...,
mode="v2", ir_path=...)`.

### Results and failures

Every call creates `out_dir/run-<12 hex>/` containing `request.json`
(source and arch SHA-256, mode, precision, `work_contract: as-written`),
stage directories, and either `result.json` or `failure.json`
(`type`, `message`, `mode`, `used_mode`, `fallback_reason`). Failed calls
raise; a failed run can never be mistaken for a stale success because the
run directory is unique.

`EstimateResult` fields: `requested_mode`, `used_mode`, `fused_ms`,
`unfused_ms`, `macs`, `output_dir`, `source_sha256`, `arch_sha256`,
`precision`, `validation`, `fallback_reason`, `work_contract`. For `v1`/`v2`
the `validation` dict carries `kind: output-comparison`, `numeric_comparison:
true`, `max_abs_error`, the effective `rtol`/`atol`, `shapes`, `dtypes`,
`ir_sha256`, `scope: "supplied get_inputs case; not all inputs"` and
`work_preservation: "requires structural audit; output equality alone is
insufficient"`. Read those two strings literally: a numerical pass is scoped
to one input case and says nothing about whether the IR preserves the
as-written work.

Observed at this base (CPU, offline):

| Call | `used_mode` | `macs` | `validation` |
|---|---|---|---|
| `estimate("examples/TwoInputMatmul/TwoInputMatmul.py", out, mode="v0")` | `v0` | 262144 | `mechanical`, no comparison |
| `estimate("examples/Matmul/Matmul.py", out, mode="v0")` | `v0` | 1048576 | `mechanical`, no comparison |
| `estimate(RMSNORM/"adapter.py", out, mode="v2", source_type="cuda", ir_path=RMSNORM/"solar_model_einsum_lib.py")` | `v2` | 0 | `output-comparison`, `max_abs_error 4.77e-07`, shapes B=8, H=128 |

Exceptions: `UnsupportedError` (mechanical frontend cannot faithfully lower
the input), `ValidationError` (IR failed execution or comparison; no SOL
emitted), `PipelineError` (infrastructure or analysis failure; never a
fallback trigger), `AgentTranslationRequired` (subclass of `UnsupportedError`).

Execute only trusted source files: references and IR are Python and are
executed.

## `solar.translate` and receipts

```python
from solar import translate as T
from solar.translate.receipts import emit_receipt, check_strict, read_receipt, ReceiptRefusal

T.scaffold(problem_dir, out_dir, source_type="cuda")                          # ScaffoldResult
v = T.verify(model_or_adapter, dsl, source_type="cuda", mem_budget_gb=8,
             rtol=None, atol=None, timeout=None)                              # VerifyResult
receipt = emit_receipt(v, source_path=..., dsl_path=..., definition_path=...,
                       workload_uuid=..., axes=..., dtype="fp16", out_path=...)
s = T.sol(dsl, definition_dir, out_dir, arch="B200", precision="fp16",
          fallback_adapter=..., receipt=receipt, strict=True, source_path=...)  # SolResult
T.translate(source_file, out_dir, mode="api", source_type="pytorch", api_key=KEY)  # explicit API
```

Signatures, defaults and the artifact layout are in [SOLAR_V2_FLOW.md](SOLAR_V2_FLOW.md);
what each receipt field means is in
[RECEIPTS_AND_VALIDATION.md](RECEIPTS_AND_VALIDATION.md). `SolResult` carries
`fused_ms`, `unfused_ms`, `macs`, `perf_path`, `analysis_path`,
`einsum_path`, `dsl_dir`, `log`, and the validation label `validated`,
`validation_status` (`validated` or `unvalidated_legacy`), `receipt`,
`validation_path` (`<out_dir>/validation.json`) and `validation_line`.
`VerifyResult` carries `status` (`pass`, `pass_vacuous`, `shape_only`,
`fail`, `error`), `vacuous_outputs` and `exit_status(strict=False)`;
`TranslateResult` (API mode) carries `exit_code` and `verification_status`.
The same receipt can be emitted from the command line with
`solar-translate verify --emit-receipt` and consumed with
`solar-translate sol --strict --receipt --source-path`.

## `solar.frontend.router`

```python
from solar.frontend.router import decide, decide_from_graph, RouteDecision, SUPPORTED_OPS

d: RouteDecision = decide("out/graph/pytorch_graph.yaml")   # or decide_from_graph(yaml_dict)
d.route      # "mechanical" | "agentic" | "unsupported"
d.reasons    # tuple of strings, never empty
d.evidence   # {"compute_nodes": n, "ops": {...}, "supported_set": [...], ...}
d.to_json()
```

CLI: `python3 -m solar.frontend.router GRAPH.yaml [--out route.json]`. The
decision is deterministic and offline. What it checks and does not check is
listed in [route selection](SOLAR_FLOW.md#three-places-where-supported-is-decided).

## Mechanical stage classes

```python
from solar.graph import PyTorchProcessor
from solar.einsum import PyTorchToEinsum, EinsumGraphVisualizer, EinsumToTimeloop
from solar.analysis import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel

PyTorchProcessor().process_model_file("model.py", output_dir="out/graph")
PyTorchToEinsum().convert("out/graph/pytorch_graph.yaml", "out/einsum", copy_graph=False)
EinsumGraphAnalyzer().analyze_graph("out/einsum/einsum_graph_renamed.yaml", "out/analysis",
                                    precision="fp16", copy_graph=False)
EinsumGraphPerfModel().predict("out/analysis/analysis.yaml", "out/perf",
                               arch_config="B200", precision="fp16")
EinsumGraphVisualizer().save_graph_pdf("out/einsum/einsum_graph_renamed.yaml", "out/einsum/graph.pdf")
EinsumToTimeloop().convert("out/einsum/einsum_graph_renamed.yaml", "out/timeloop/timeloop_graph.yaml")
```

`PyTorchToEinsum(enable_agent=True)` turns on the unknown-operator handler
generator, which calls an inference API and adds no execution validation;
`solar.estimate` never enables it. `EinsumGraphPerfModel(dtype_bytes=True)`
prices traffic at each tensor's own dtype width (the `estimate` default);
the CLI equivalent is `--dtype-bytes`. `EinsumGraphAnalyzer.analyze_graph`
also accepts `boundary_storage_dtypes` and `declared_output_storage`, which
`solar.translate.sol` fills from a benchmark definition.
