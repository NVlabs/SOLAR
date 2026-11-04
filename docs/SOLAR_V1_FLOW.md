<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Solar v1: classic PyTorch graph pipeline

Run from the repository root with Solar installed (`python -m pip install -e .`). Set `PY`
to your environment's Python 3.10 or newer. On the shared research hosts, use
the architecture-specific interpreter listed in [AGENTS.md](../AGENTS.md).

## Input contract

Provide a Python file defining a `torch.nn.Module` named `Model` and
`get_inputs()` returning model inputs. If construction requires arguments,
provide `get_init_inputs()`. Traceable operations need supported graph-node
handlers. Unknown or incorrectly expanded operations need inspection before
trusting the resulting counts.

## Commands

```bash
PY=python3
MODEL=examples/Attention/Attention.py
OUT=output/v1_attention

$PY -m solar.cli.process_model --model-file "$MODEL" --output-dir "$OUT/graph"
$PY -m solar.cli.toeinsum_model --graph-path "$OUT/graph/pytorch_graph.yaml" \
    --output-dir "$OUT/einsum" --enable-rename
$PY -m solar.cli.analyze_model \
    --einsum-graph-path "$OUT/einsum/einsum_graph_renamed.yaml" \
    --output-dir "$OUT/analysis" --no-copy-graph
$PY -m solar.cli.predict_perf_model --analysis-path "$OUT/analysis/analysis.yaml" \
    --output-dir "$OUT/perf" --arch-config B200 --precision fp16
$PY -m solar.cli.totimeloop \
    --einsum-graph-path "$OUT/einsum/einsum_graph_renamed.yaml" \
    --output-dir "$OUT/timeloop"
```

The final Timeloop stage is optional. It exports a workload representation; it
does not run Timeloop or establish a valid hardware mapping. `--enable-rename`
is required here because later commands select the renamed graph. Without it,
use `einsum_graph.yaml`. Add `--save-graph` to the conversion command for an
optional PDF (requires graph visualization dependencies).

No LLM is needed for supported operators. The separate
`--enable-llm-agent` conversion option can request handlers for unsupported
operators and requires API configuration; it does not add v2 verification.

## Implementation

| Stage | Entry point / implementation | Output |
|---|---|---|
| Extract | `solar/cli/process_model.py`, `solar/graph/` | `graph/pytorch_graph.yaml` |
| Convert | `solar/cli/toeinsum_model.py`, `solar/einsum/`, `solar/einsum/ops/` | `einsum/einsum_graph.yaml`, optional renamed graph |
| Analyze | `solar/cli/analyze_model.py`, `solar/analysis/graph_analyzer.py` | `analysis/analysis.yaml`: per-layer and total work/bytes |
| Predict | `solar/cli/predict_perf_model.py`, `solar/perf/perf_model.py`, `configs/arch/` | `perf/perf_B200.yaml` |
| Export | `solar/cli/totimeloop.py`, `solar/einsum/` | `timeloop/timeloop_graph.yaml` |

The analyzer consumes shape, dtype, role, and sparsity metadata. Performance
prediction applies architecture and precision to those counts and reports
unfused, fused, and fused-plus-prefetched estimates. In the current model,
`compute_cycles` prices the contraction MACs plus any other non-contraction MACs the
analyzer records; `compute_sm_cycles` is a diagnostic that prices `other_ops` only;
the multiply statements of non-einsum DSL kernels that are counted as MACs are
excluded from the tensor-core term and retained as counts (issue #58 convention B;
`sol_convention` and `workload.tensor_core_term` in `perf_<arch>.yaml`).  The
headline is an optimistic contraction-and-memory bound, not a bound over all compute
resources.  Elementwise-only cases therefore rely on the modeled memory cost. This is
an accounting convention, not proof that elementwise or scalar instructions are
hidden or free (see `docs/SOL_GUIDE.md`, "Convention").

## Install / preflight contract: the torchview capability gate

The mechanical route needs two behaviours from the installed torchview that the released 0.2.7 wheel does not
provide by itself: **parameter tensors** (a bare `nn.Parameter` passed to a torch function becomes a
`parameter-tensor` node, so weights are billed and their traffic counted; `patches/torchview-parameter-tensors.patch`)
and **call attributes** (`collect_attributes=True` reaches every FunctionNode, so stride/padding/dilation/groups,
transpose dims, softmax dim, einsum equations and norm parameters are readable; upstream #128,
`patches/torchview-collect-attributes-inputs.patch`). Without them a capture *looks* ordinary but is wrong: weights
vanish, conv MACs are 0, addmm MACs collapse (issue #45).

**Contract.** Two things are kept apart. (1) The **installed dependency**: the public install (`install.sh`, which
clones torchview at `edbe1fa` and applies the parameter-tensors patch) is the versioned artifact that must reproduce
both behaviours; the record names it (`torchview_version`, `torchview_file`, `source`). (2) The **effective engine
behaviour**: `solar.graph.pytorch_processor` installs the lineage's in-memory parameter recorder shim
(`_torchview_param_shim`) when it is imported, and wraps call-attribute propagation around every probe and capture
(`_patch_torchview_collect_attributes`). So a plain released wheel can be made capable by the engine; the record then
says `source: released+shim`, which is **not** the same as a patched install. The gate is a behavioural preflight,
not the install contract: it **verifies**, at `PyTorchProcessor` construction and under the processor's own draw
context, what a real capture will see (`solar/graph/torchview_capability.py`: an `nn.Linear` whose output feeds a
functional `softmax(h + 1.0, dim=-1)`, plus a functional `conv2d` with kwargs feeding `relu(c * 2.0)`, traced on CPU,
no files; call attributes count as present only when every FunctionNode including the hidden-tensor consumers
carries them) and

* records the result in every output: `capture_capability` (`parameter_tensors`, `call_attributes`,
  `torchview_version`, `torchview_file`, `source: patched|released+shim|released|unknown`, `probe` incl.
  `chain_consumers_with_attributes` and the probe's `draw_context`, `dynamo_warmup` (the processor imported
  `torch._dynamo` before capturing, so its import-time builtin probe cannot be recorded as model ops), `reason`) and
  `capture_status` in `pytorch_graph.yaml` and `einsum_graph*.yaml`, `metadata.capture_capability` in
  `analysis.yaml`, and `capture_capability` in `perf_<arch>.yaml`; capability records are cached per draw context and
  every entry point (`get_torchview_capability()`, the constructor, `_check_torchview_parameter_support()`) uses
  the one canonical engine context by default; `draw_context=None` is the explicit bare probe and never stands
  in for the processor's;
* reuses a saved `pytorch_graph.yaml` only when its `provenance` matches the source bytes, the sha256 of the complete
  capture implementation (processor, node writer, capability module, recorder shim, installed torchview recorder and
  driver modules) **and** the effective capability identity, and the saved `capture_status` is `ok`; a graph without a
  record, with a different record or a degraded one is re-captured;
* **fails closed** (`TorchviewCapabilityError`, `solar-process-model` exit 3 with the record on stderr) when either
  capability is missing;
* with the explicit `--allow-degraded-capture` flag (`ProcessingConfig(allow_degraded_capture=True)`) proceeds, but
  every einsum layer is marked `is_einsum_supportable: false` with `capture_degraded`, every analysis layer and
  `total` carry `status: incomplete`, and the perf `workload` carries `status: incomplete`. Such outputs are
  diagnostics, never trusted accounting.

CI's `validate-install` job applies both patches to the wheel before running the suite so the suite exercises a
capable capture; that job is **not** the install contract for users, which is the paragraph above (the engine
refuses an incapable torchview). `tests/test_torchview_capability_gate.py` covers the capable path, the fail-closed
path, the override markers in all four stages, and, when the released-wheel profile interpreter is present on the
host, the released wheel end to end.

## Validation

Inspect warnings, unsupported operators, ranks, and layer shapes. A successful
trace is not a numerical equivalence certificate. Compare model outputs with
a reference independently when correctness is in question, or use the
[v2 flow](SOLAR_V2_FLOW.md).

Relevant backend checks:

```bash
$PY -m pytest tests/test_graph_analyzer_ordering.py tests/test_perf_quant.py -q
```
