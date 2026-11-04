<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSL-Based Einsum Generation Path

For the complete current manual and API recipes (verification through SOL),
see [Solar v2 flow](SOLAR_V2_FLOW.md). Agent-authored DSL is the default;
API-generated translation is opt-in. This page details the compiler stages.

This guide describes the **verified einsum generation path**: an alternative to the
original PyTorch-trace-based pipeline that produces `einsum_graph.yaml` from
LLM-generated DSL kernel descriptions. It also explains how to validate the output
using `check_einsum_graph_equivalence.py`.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Output Directory Structure](#output-directory-structure)
4. [Running the Pipeline](#running-the-pipeline)
5. [Module Reference](#module-reference)
6. [Einsum Equation Derivation](#einsum-equation-derivation)
7. [Validation with Graph Equivalence Checker](#validation-with-graph-equivalence-checker)
8. [Expected Differences from the Original Path](#expected-differences-from-the-original-path)

---

## Overview

The original solar pipeline extracts einsum graphs by tracing a PyTorch model and
converting each traced op to an einsum equation via hard-coded node handlers
(`solar/einsum/ops/`). This works well for standard PyTorch modules but is
tightly coupled to PyTorch's internal op names.

The **DSL path** inverts this: an LLM rewrites the model as a set of
`@kernel`-decorated functions in `solar.einsum_lib` DSL notation, where each
kernel's loop body *explicitly encodes* the computation. The einsum equation is then
derived deterministically from the tensor dimension names — no op-name lookup tables
needed.

```
Original path:
  PyTorch model ──► torch.fx trace ──► node handler lookup ──► einsum_graph.yaml

DSL path (verified):
  PyTorch model ──► LLM (toDSL_model) ──► *_einsum_lib.{yaml,py}
                                        ──► einsum_dsl translator
                                        ──► graph/  (topology, loop_expr)
                                        ──► einsum/ (+ einsum equations)
```

---

## Pipeline Stages

### Stage 0 — DSL Generation (`toDSL_model`)

The LLM rewrites the PyTorch model as a `solar.einsum_lib` DSL file pair:

| File | Contents |
|---|---|
| `*_einsum_lib.py` | `@kernel` functions with explicit for-loop bodies |
| `*_einsum_lib.yaml` | Tensor metadata (dims, dtype, role) + graph topology |

**Run:**
```bash
python3 -m solar.cli.toDSL_model examples/Attention/Attention.py \
    --output-dir examples/Attention/output_verified/dsl/
```

---

### Stage 1 — Graph Stage (`cli`)

Reads the DSL pair and writes `einsum_graph.yaml` with:
- Full graph topology (start node + kernel nodes, connections)
- Tensor names, types, shapes
- `loop_expr` — innermost loop body extracted by AST parsing (e.g. `Y[b, s, h2] += X[b, s, h] * W[h2, h]`)
- `einsum_equation: null` (filled in Stage 2)

**Run:**
```bash
python3 -m solar.einsum_dsl.cli examples/Attention/output_verified/dsl/
# writes → examples/Attention/output_verified/graph/Attention_einsum_graph.yaml
```

---

### Stage 2 — Einsum Stage (`cli_einsum`)

Reads the same DSL pair and writes a second `einsum_graph.yaml` with
`einsum_equation` filled in for every node. Equations are derived from tensor
dimension names (see [Einsum Equation Derivation](#einsum-equation-derivation)).

**Run:**
```bash
python3 -m solar.einsum_dsl.cli_einsum examples/Attention/output_verified/dsl/
# writes → examples/Attention/output_verified/einsum/Attention_einsum_graph.yaml
```

---

## Output Directory Structure

After running all stages for an example, the `output_verified/` tree looks like:

```
examples/Attention/output_verified/
├── dsl/
│   ├── Attention_einsum_lib.py      # LLM-generated @kernel functions
│   └── Attention_einsum_lib.yaml    # tensor metadata + graph topology
├── graph/
│   └── Attention_einsum_graph.yaml  # topology + loop_expr, einsum_equation: null
└── einsum/
    └── Attention_einsum_graph.yaml  # topology + loop_expr + einsum equations
```

This mirrors the original pipeline's `output/` layout:

```
examples/Attention/output/
├── graph/
│   └── pytorch_graph.yaml
└── einsum/
    ├── einsum_graph.yaml
    └── einsum_graph_renamed.yaml    # ← compare against output_verified/einsum/
```

---

## Running the Pipeline

No example in this release ships a `run_solar_verified.sh` (the first stage needs
an explicitly configured inference endpoint and key). From the repository root the
three stages are:

```bash
python3 -m solar.cli.toDSL_model examples/Attention/Attention.py --output-dir out/dsl --source-type pytorch
python3 -m solar.einsum_dsl.cli out/dsl
python3 -m solar.einsum_dsl.cli_einsum out/dsl --output-dir out/einsum
```

The stages are:
1. `toDSL_model` — API DSL generation (requires `SOLAR_API_BASE_URL` and a key via `OPENAI_API_KEY` or `--api-key-file`)
2. `solar.einsum_dsl.cli` — graph stage
3. `solar.einsum_dsl.cli_einsum` — einsum stage

To skip the LLM step when DSL files already exist, run stages 1 and 2 directly:

```bash
python3 -m solar.einsum_dsl.cli      examples/Attention/output_verified/dsl/
python3 -m solar.einsum_dsl.cli_einsum examples/Attention/output_verified/dsl/
```

---

## Module Reference

All new code lives under `solar/einsum_dsl/`:

| Module | Role |
|---|---|
| `model.py` | `DSLTensor`, `DSLInput`, `DSLKernel`, `DSLGraph` dataclasses |
| `loader.py` | `load_dsl_yaml()` — parse `*_einsum_lib.yaml` |
| `ast_parser.py` | `extract_loop_exprs()` — AST-parse `*_einsum_lib.py`, extract innermost loop bodies per `@kernel` function |
| `assembler.py` | `build_dsl_graph()` — combine YAML + loop exprs into `DSLGraph` |
| `exporter.py` | `dsl_graph_to_dict()`, `write_einsum_yaml()` — serialise to einsum_graph format |
| `einsum_converter.py` | `build_kernel_einsum()`, `build_start_einsum()`, `annotate_einsum_equations()` — derive equations from dim names |
| `cli.py` | Entry point for the graph stage (`python3 -m solar.einsum_dsl.cli`) |
| `cli_einsum.py` | Entry point for the einsum stage (`python3 -m solar.einsum_dsl.cli_einsum`) |

---

## Einsum Equation Derivation

Equations are derived from the dimension names declared in each tensor's `dims`
field in `*_einsum_lib.yaml`. No parsing of the loop body is needed for the
equation itself — the loop expression is retained separately as `loop_expr` for
human reference and future verification.

### Dim-name → token mapping

The solar einsum format requires tokens matching `[A-Z][0-9]*`
(single capital letter, optional integer suffix). DSL dim names are mapped as
follows:

| DSL dim name | Rule | Token |
|---|---|---|
| `B`, `S`, `H`, `D` | Single letter → use as-is | `B`, `S`, `H`, `D` |
| `H2`, `B0`, `S1` | Letter + pure digits → keep | `H2`, `B0`, `S1` |
| `Sk` | Multi-char alpha → first letter; add counter if conflict | `S0` |
| `Nh` | Multi-char alpha → first letter; no conflict → | `N` |

Tokens are assigned in first-appearance order across all tensors of a kernel
(input tensors in YAML order, then output tensor). This makes the assignment
deterministic.

### Examples

| Operation | Tensor dims | Derived equation |
|---|---|---|
| Linear projection | `X: B,S,H` · `W: H2,H` → `Y: B,S,H2` | `BSH,H2H->BSH2` |
| Attention scores | `Q: B,Nh,S,D` · `K: B,Nh,Sk,D` → `Y: B,Nh,S,Sk` | `BNSD,BNS0D->BNSS0` |
| Softmax | `X: B,Nh,S,Sk` → `P: B,Nh,S,Sk` | `BNSS0->BNSS0` |
| Attention output | `A: B,Nh,S,Sk` · `V: B,Nh,Sk,D` → `Y: B,Nh,S,D` | `BNSS0,BNS0D->BNSD` |

---

## Validation with Graph Equivalence Checker

`solar/common/check_einsum_graph_equivalence.py` compares two
`einsum_graph.yaml` files and reports whether they are structurally and
mathematically equivalent.

### What it checks

For each pair of matched nodes (matched by BFS traversal order from `start`):

1. **Tensor shapes** — all input and output shapes must match exactly.
2. **Einsum equation (canonical form)** — equations are considered equivalent if
   they produce the same canonical form after renaming all dimension labels to
   `A, B, C, ...` in first-appearance order.

   ```
   B0B1K,NK->B0B1N   →  canonical:  ABC,DC->ABD
   BSH,H2H->BSH2      →  canonical:  ABC,DC->ABD   ✓ equivalent
   ```

3. **Fan-in / fan-out** — the number of predecessor and successor connections must
   match, verifying that the local topology is the same.

### Running the checker

```bash
# Compare original pipeline output vs DSL-derived einsum output
python3 -m solar.common.check_einsum_graph_equivalence \
    examples/Attention/output/einsum/einsum_graph_renamed.yaml \
    examples/Attention/output_verified/einsum/Attention_einsum_graph.yaml \
    --verbose
```

**Sample output (fully equivalent):**
```
Graph A: examples/Attention/output/einsum/einsum_graph_renamed.yaml
Graph B: examples/Attention/output_verified/einsum/Attention_einsum_graph.yaml

Matched nodes:  8  (8 equivalent, 0 with differences)

✓ Graphs are equivalent.
```

**Sample output (mismatch):**
```
Graph A: ...
Graph B: ...

Matched nodes:  8  (7 equivalent, 1 with differences)

✗ Graphs differ:
  [Model.linear] vs [linear_q_proj#0]
    - tensor_shapes differ:
        A: inputs=[[2, 32, 64], [64, 64]] outputs=[[2, 32, 64]]
        B: inputs=[[2, 32, 64], [32, 64]] outputs=[[2, 32, 64]]
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Graphs are equivalent |
| `1` | Differences found or load error |

This makes the checker suitable for use in CI scripts:

```bash
python3 -m solar.common.check_einsum_graph_equivalence graph_a.yaml graph_b.yaml \
    || echo "FAIL: einsum graphs differ"
```

### Programmatic use

```python
from solar.common.check_einsum_graph_equivalence import check_equivalence

report = check_equivalence(
    "output/einsum/einsum_graph_renamed.yaml",
    "output_verified/einsum/Attention_einsum_graph.yaml",
)
report.print_report(verbose=True)

if report.equivalent:
    print("OK")
else:
    print(f"Extra nodes in A: {report.extra_a}")
    print(f"Extra nodes in B: {report.extra_b}")
    for diff in report.node_diffs:
        if not diff.ok:
            print(diff)
```

---

## Expected Differences from the Original Path

The original solar pipeline inserts additional intermediate nodes during
PyTorch tracing that have no direct counterpart in the DSL graph:

| Original node type | Reason present in original | Present in DSL graph? |
|---|---|---|
| `view` / `reshape` | PyTorch traces explicit tensor reshapes | No — folded into `@kernel` reshape functions |
| `transpose` / `permute` | PyTorch traces axis permutations | No — folded into kernels |
| `mul` (scalar) | Attention scale `* 1/sqrt(d)` is a separate traced op | No — folded into `scale_scores` kernel |

Because of this, `extra_a` (nodes in the original graph not matched in the DSL
graph) will typically list these intermediate nodes. This is expected and does
**not** indicate an error — the DSL representation is more coarse-grained by
design, grouping reshape/scale steps into named kernels.

The nodes that **must** match are the compute-intensive real-einsum nodes:
linear projections, attention score computation, and attention output. If any of
these differ in shape or canonical equation, that indicates a problem in the DSL
generation or equation derivation.
