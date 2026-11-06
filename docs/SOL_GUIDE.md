<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOL (Speed-of-Light) Performance Models Guide

This guide explains the three roofline-based performance models used in Solar for predicting DNN execution time.


## SOL assumptions

SOL is an optimistic analytical estimate under the following assumptions. Report
these as **SOL assumptions**, separately from numerical validation and the source
version used to produce a result.

- **Peak hardware rates.** Use the selected hardware profile's compute rates and
  DRAM bandwidth for the declared computation precision and memory accounting mode.
- **Sparse vs dense tensor-core peak (`--sol-mode`, issue #14).** Every shipped
  hardware profile carries BOTH vendor peaks per precision
  (`peak_TFLOPS_<p>_dense` / `peak_TFLOPS_<p>_sparse`, with the derived
  `MAC_per_cycle_<p>_tc_dense` / `_sparse` = TFLOPS x 1e12 / 2 FLOP per MAC /
  (freq_GHz x 1e9), i.e. TFLOPS x 1000 / 2 / freq_GHz). The headline SOL is
  priced at one set, selected by `--sol-mode {sparse,dense}` (default
  **sparse**). The sparse SOL *assumes* the workload is eligible for the
  vendor's 2:4 structured-sparsity tensor-core rate; it is a lower bound that
  no dense kernel can reach, and it does not claim that a given workload
  satisfies 2:4 sparsity. The **dense SOL is the more realistic estimate**
  (the realistic bound) for dense work. Every perf record carries both as
  `sol_sparse` and `sol_dense` (unfused / fused / fused_prefetched runtimes
  and the rate keys used) next to `sol_mode` and `sol_mode_default`; the
  top-level sections hold the selected mode's numbers. A user profile that
  spells a rate with the legacy bare key `MAC_per_cycle_<p>_tc` declares a
  dense rate; sparse mode then falls back to it and records
  `sparse_fallback_to_dense`. Where the vendor publishes no sparse figure
  (FP32 on the CUDA cores, FP64) the two sets are equal.
- **Memory scenario.** Fused SOL counts each required external storage region once
  for reads and each required output write once, assuming perfect fusion and reuse
  of intermediates on chip. Unfused SOL includes intermediate accesses. Capacity,
  spills and scheduling are outside the fused minimum-traffic scenario.
- **Compute-memory overlap.** The current implementation uses one whole-graph
  roofline, `max(total_compute_cycles, total_memory_cycles)`. Fused and
  fused-prefetched currently use the same traffic totals and produce the same value.
- **Logical workload and precision.** SOL accounting uses the original logical
  dimensions and declared data types. Record the selected physical-width or
  uniform-width memory scenario. Smaller validation shapes or FP64 emulation do
  not substitute their sizes or carrier precision into the SOL calculation.
- **Scalar-work convention.** The headline prices contraction MACs and other
  recorded non-contraction MACs. It excludes `scalar_statement_multiplies` and
  `other_ops` from headline timing while retaining their counts; the existing
  `compute_sm_cycles` diagnostic prices `other_ops` only. This is the stated
  optimistic contraction-and-memory bound, without a claim that excluded work is
  hidden or free. The detailed convention below also covers older analysis records.
- **Program scenario.** Work counts describe the selected program and its declared
  specialization or algorithm. They do not establish the globally optimal algorithm.

Reports should show **SOL assumptions**, **numerical validation scope**, and
**source/version validation status** separately. Numerical scope records the actual
shapes, reference/carrier precision and emulation mode. Version status records the
source and configuration hashes and completed, failed or pending checks. Stated
SOL assumptions alone are not a failed check; incorrect accounting, failed
comparisons and missing validation remain explicit. These estimates are analytical
SOL values rather than measured kernel latencies.

## Overview

Solar computes three SOL (Speed-of-Light) performance estimates based on the roofline model. Each model makes different assumptions about memory access patterns:

| Model | Memory Accesses | Roofline Application | Use Case |
|-------|-----------------|---------------------|----------|
| **Unfused** | Every tensor access of every op (inputs + weights + outputs), intermediates included | One whole-graph roofline on the graph-wide totals | Baseline / worst case |
| **Fused** | Unique required external reads (model inputs + weights) + required output writes; intermediates excluded | One whole-graph roofline on the graph-wide totals | Compulsory-I/O bound under perfect fusion |
| **Fused+Prefetched** | Same traffic as fused | One whole-graph roofline on the graph-wide totals | Same number as fused (see Gap 2) |

> **Implementation note**: All three models apply a single whole-graph roofline, `max(total_compute_cycles, total_memory_cycles)`, to **graph-wide sums**; no per-op roofline is applied anywhere in the code, old or new. They differ only in how the memory bytes are totaled. `fused` and `fused_prefetched` produce identical memory totals and therefore identical runtimes. See [Section 2](#2-fused-sol) for the fused-traffic principle and its implementation, [Section 6](#6-known-implementation-gaps) for the gaps.

## Roofline Model Basics

The roofline model predicts performance based on two hardware limits:
- **Compute bound**: Limited by peak compute throughput (MACs/second)
- **Memory bound**: Limited by memory bandwidth (bytes/second)

```
runtime_cycles = max(compute_cycles, memory_cycles)
```

Where:
- `compute_cycles = priced_macs / MAC_per_cycle_tc`, where the priced MACs are the
  contraction MACs (quantised-dtype buckets at their own tensor-core rates, the
  workload-precision remainder at the applied MAC rate) plus any other
  non-contraction MACs the analyzer records (`macs_typed.non_contraction_other`)
- `memory_cycles = total_memory_bytes / DRAM_byte_per_cycle`
- **Arithmetic Intensity** = total_macs / total_memory_bytes

`total_other_ops / MAC_per_cycle_fp32_sm` is reported as `compute_sm_cycles` (an
informational diagnostic that prices `other_ops` only) and is excluded from
`compute_cycles` by convention.  That exclusion is not a claim that elementwise or
reshape work is hidden by the memory term or free: the headline is an optimistic
contraction-and-memory bound, not a bound over all compute resources (see
"Convention" below).

### Convention: an optimistic contraction-and-memory bound (issue #58, option B)

The headline SOL number is `max(compute_cycles, memory_cycles)`.  The compute term
prices the analysis' **contraction MACs** (quantised-dtype buckets at their
tensor-core rates, the workload-precision remainder at the applied MAC rate) **plus
any other non-contraction MACs** the analyzer records
(`total.macs_typed.non_contraction_other`, retained in the term as before).  Two
kinds of SM-class statement work are excluded from `compute_cycles` and retained as
counts:

- `other_ops` (elementwise / reduction / scale statements); the informational
  diagnostic `compute_sm_cycles` prices these, and only these, at
  `MAC_per_cycle_fp32_sm`, unchanged;
- the multiply statements of non-einsum DSL kernels that the per-statement
  classifier counts as MACs (`total.macs_typed.scalar_statement_multiplies`).  Until
  convention B they were priced inside the default-rate remainder at the
  tensor-core rate while the same statement class in `other_ops` was excluded; the
  perf model now subtracts them from the tensor-core term
  (`workload.tensor_core_term.scalar_statement_multiplies_excluded`), keeps the typed
  counts (`workload.macs_typed`) and does not add them to the SM diagnostic.

Limits, stated once and carried in every `perf_<arch>.yaml` as `sol_convention`:
the number is **not a bound accounting for all compute resources** (the excluded
SM-class statement work has no timing term in the headline); the exclusion is a
convention, **not a proof that the excluded work is hidden by memory traffic or
free**; no count, tolerance, tie, reference, cap or selection rule changes.  An
analysis whose `total.macs_typed` is **absent** is priced whole, as before, and the
receipt says so (`tensor_core_term.applied: false`).  A `total.macs_typed` that is
present but not a mapping refuses (`macs_typed_malformed`, path and value in the
receipt); a component that is not an exact count refuses
(`macs_typed_component_malformed`, `scalar_statement_component_malformed`); and
conservation is recomputed from the exact values, never read from the stored
`components_sum_to_aggregate` flag: aggregate = `total.macs`, contraction + scalar +
other = aggregate, the dtype buckets sum to the contraction count, and the
default-rate remainder equals contraction-at-workload-precision + scalar + other,
or the prediction refuses (`typed_components_not_conserving`) with every original
value in the receipt.

Static before/after accounting for the Lite 203 recorded aggregate (880,803,840
NVFP4 contraction MACs, 330,240 scalar statement multiplies, 1,850,374,150
`other_ops`; B200 profile, bf16 workload precision; no run, no new prediction):

| term | before | after |
|---|---:|---:|
| `compute_tc_cycles` | 880,803,840 / 2,415,000 + 330,240 / 603,751 = 365.27 | 880,803,840 / 2,415,000 = 364.72 |
| scalar statement multiplies (count) | 330,240 | 330,240 |
| `compute_sm_cycles` (1,850,374,150 / 40,251) | 45,970.9 | 45,970.9 |
| fused `memory_cycles` (27,903,492 B / 5,111.6) | 5,458.9 | 5,458.9 |
| fused `total_cycles` | 5,458.9 | 5,458.9 |

The tensor-core term falls by 330,240 / 603,751 = 0.547 cycles; the memory term
dominates, so the headline is unchanged for this memory-dominated case.  Lite 203
remains not qualified; nothing here says its scalar work is free.

> **Units.** `MAC_per_cycle_*` in `configs/arch/*.yaml` are MACs (multiply-accumulate
> pairs) per cycle, **not** FLOPs per cycle. Two different counts are summed in two
> different units: `total_macs` counts **MACs**, one multiply plus one accumulate per
> count (the contractions: matmul, convolution, the reducing steps of an einsum);
> `total_other_ops` counts **single scalar operations** of the elementwise, reduction and
> non-reducing steps (an einsum's non-reducing steps are reported as `scalar_multiplies`
> and summed into `other_ops`, never into `macs`; `graph_analyzer.py` L2185–L2191 at
> an earlier revision). Neither is a count of emitted hardware instructions. The two-FLOP
> conversion applies to MACs only: `flops = 2 × macs` (L2193), and `total_other_ops` is
> not part of it. Vendor datasheets quote FLOPs, so an arch config entry must be half the
> advertised dense MAC-equivalent rate. Entering the datasheet number directly makes
> every compute-bound prediction 2x optimistic.
>
> The conversion is `MAC_per_cycle = dense FLOP/s ÷ (2 FLOP/MAC × freq_GHz × 1e9)` at
> the config's declared `freq_GHz`, which is the model's conversion scale, not a
> measured clock. For `H100_PCIe.yaml` (`freq_GHz: 2`): 51 TFLOPS dense fp32 →
> `MAC_per_cycle_fp32_sm: 12750`; 378 / 756 / 1513 dense tera-ops/s (TF32 / FP16+BF16 /
> FP8+INT8, Tensor Core) → 94,500 / 189,000 / 378,250 (the config rounds the last to
> 378,000 from 1512). These H100 PCIe values, and the comment that records the rule and
> the datasheet figure each value came from, are on the target **through the
> separately reviewed lineage SOL-regression repair**; at
> the lineage base the H100 config carried doubled values (25,500 fp32, 756,000 fp8), which is
> the units error this box describes, not a faster part. The rule applies to every arch
> yaml; whether a given file carries the comment and the datasheet receipts is a per-file
> fact to check in the config, not a guarantee of this guide. **H100 PCIe** has no
> `fp32_tc` key, so `--precision fp32` prices MACs at its `_sm` rate; that fallback
> statement is scoped to this config (`A6000.yaml`, for example, declares
> `MAC_per_cycle_fp32_tc`). TF32 throughput is a separate precision contract that is
> never selected automatically.

---

## 1. Unfused SOL

### Description
All tensor accesses (inputs, weights, outputs) for every operation are assumed to come from DRAM. This produces the highest memory traffic estimate.

### Memory Calculation ([`graph_analyzer.py` L1525–L2110](../solar/analysis/graph_analyzer.py#L1525-L2110), per-layer loop)
```
Per layer:  unfused_elements_i = Σ memory_reads_i + Σ memory_writes_i      (special cases: aliases 0,
                                                                              embedding = gathered rows,
                                                                              slices = accessed region)
            unfused_bytes_i    = Σ_k memory_reads_i[k] × width(dtype_i[k]) + Σ_k memory_writes_i[k] × width(dtype)
Total:      unfused_elements   = Σ_i unfused_elements_i   (+ declared model outputs no traced op wrote)
            unfused_bytes      = Σ_i unfused_bytes_i
```

`width(dtype)` is the physical byte width of each tensor's recorded dtype (see
[Physical byte widths](#physical-byte-widths)). The historical implementation used one
`bytes_per_element` for the whole graph (`unfused_elements × bytes_per_element`); the
perf model still does so unless the per-dtype mode is selected.

### Roofline Application ([`perf_model.py` L267–L268](../solar/perf/perf_model.py#L267-L268))
```
unfused_total_cycles = max(compute_cycles, unfused_bytes / DRAM_byte_per_cycle)
runtime_ms           = unfused_total_cycles / (freq_GHz × 1e6)
```

A single whole-graph roofline is applied to the graph-wide totals. Intermediate tensors are counted as DRAM traffic (read by the consumer, written by the producer).

### When to Use
- Baseline performance estimate
- No kernel fusion
- Memory-bound workloads with poor data reuse
- Debugging / understanding memory bottlenecks

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Input + Weight, Write Output → DRAM
Layer 2 (ReLU):   Read Input (from DRAM), Write Output → DRAM
Layer 3 (Linear): Read Input (from DRAM) + Weight, Write Output → DRAM
```

---

## 2. Fused SOL

### Description
Intermediate tensor accesses are **excluded** from memory cost. What is counted is the
**compulsory external I/O of the whole graph under perfect fusion**: every required
external storage region (model inputs and weights, at the region actually accessed) read
**once**, plus every required output written once, each priced at its stored dtype width.

#### The principle (what the number means)

```
fused_bytes = Σ_unique required external read regions ( elements × width(stored dtype) )
            + Σ required output writes                ( elements × width(stored dtype) )
```

This is an **optimistic compulsory-I/O bound under perfect fusion and reuse**. It is
*not* an achievable runtime and *not* a capacity-aware traffic estimate: finite on-chip
storage, spills, scheduling and dependency constraints can force more traffic, and
nothing in this model checks whether the intermediates (reported separately as
`intermediate_elements` / `intermediate_bytes`) fit anywhere. A smaller number produced by
an incomplete accounting (a missing input read, an unresolved access region, an unknown
dtype) is a weaker bound, not a tighter one; such gaps are reported as incompleteness
(see the known gaps and the [embedding index-read issue](#6-known-implementation-gaps)).

#### The historical description versus the baseline implementation

Older **guide text** described the fused total as a **per-layer sum** at one **uniform
width**:

```
Per layer:  model_io_elems_i = external_input_elems_i + model_output_elems_i
Total:      fused_elements   = Σ_i model_io_elems_i          (old guide text only)
            fused_bytes      = fused_elements × bytes_per_element
```

The baseline implementation did not do that. At the earlier baseline
(`graph_analyzer.py` L669–L684) the analyzer already summed **unique external inputs
and outputs by tensor identity** for both `fused_elements` and `fused_prefetched_elements`,
and already kept the raw per-layer sum only as the separate `model_io_elements`
diagnostic ("may double-count shared inputs"). What the lineage (the lineage base) refined is
the **storage and access-region handling** (canonical base tensor behind views, slices and
aliases; union of proven access boxes instead of whole-tensor footprints) and the **width
handling** (per-tensor recorded dtype instead of one uniform `bytes_per_element`, so a bool
mask is no longer priced at the fp32 width). The GPT-2 change between the two is therefore
an access-region and width effect, not the introduction of deduplication.
`model_io_elements` is still reported per layer and in the total; it is **diagnostic** and
is **not** ordered with respect to the deduplicated fused traffic in either direction.
Since the earlier decision split the field counts, per layer, the layer's **non-weight** external
reads plus its model-output writes, while parameter reads are reported separately as
`weight_read_elements` (`graph_analyzer.py` at an earlier revision: L2643/L2659 vs L2645 classify
each external read by parameter provenance, L2778 `model_io_elems = model_input_elems +
model_output_elems`, L2814 `fused_elems = weight_read_elems + model_io_elems` per layer;
L3009 sums the layer values into `total.model_io_elements`, whereas L2988–L3004 build
`total.fused_elements` from the **deduplicated** read footprints plus the unique output
writes; on the current target a later revision the same statements sit at L3012 and
L2991–L3007 because an earlier change added three lines after the per-layer fused equation, and the
earlier anchors are unchanged). Exact relationship: `total.model_io_elements` = Σ per-layer non-weight external
reads + Σ per-layer model-output writes (shared inputs counted once per consumer, weights
excluded); `total.fused_elements` = Σ unique external read footprints (model inputs **and**
weights, each storage once) + Σ unique output writes. Sharing pushes the diagnostic
**above** the fused total; excluding weights pushes it **below**; neither effect dominates
in general. Worked example (captured on an earlier revision; the run record is not part of
this tree): two bias-free `Linear(2, 4)` on one shared `[1, 2]` input, both
`[1, 4]` outputs returned. Per layer `model_io_elements = 2 + 4 = 6` (the shared input
counted by each consumer, the weight excluded), so `total.model_io_elements = 12`, while
`total.fused_elements = 2 + 8 + 8 + 4 + 4 = 26` (the input once, both `[4, 2]` weights,
both outputs) and `total.weight_read_elements = 16`. The diagnostic is smaller here. The
opposite order can occur when shared-read duplication exceeds the included weight footprints
(for `n` consumers of a shared `I`-element input with distinct `W`-element weights and
distinct returned outputs, diagnostic − fused = `(n − 1)·I − n·W`); for example, two
parameter-free pointwise operations (say `relu` and `tanh`) reading the same `[100]` input
and returning two distinct `[100]` outputs give, per layer, `model_io_elements = 100 + 100 =
200` (the shared input counted by each consumer), so `total.model_io_elements = 2 × (100 +
100) = 400`, while
`total.fused_elements = 100 + 100 + 100 = 300` (the input once, both outputs, no weight):
the diagnostic is larger. (A `Linear(100, 1)` pair does **not** reverse the order: its
diagnostic is `2 × (100 + 1) = 202` against a fused `100 + 100 + 100 + 1 + 1 = 302`, and
with `n` such layers `101n` against `100 + 101n`.) Both derivations follow from the
classification and deduplication equations above by hand; only the 12/26 case was
captured. Never sum layer values to reproduce `fused_*`.

#### The current implementation ([`graph_analyzer.py` L1525](../solar/analysis/graph_analyzer.py#L1525) read groups, [L854](../solar/analysis/graph_analyzer.py#L854) access regions, [L956](../solar/analysis/graph_analyzer.py#L956) dtype widths, [L2210–L2236](../solar/analysis/graph_analyzer.py#L2210-L2236) totals; anchors at an earlier revision)

```
external reads:  group every external read by its canonical base tensor (views, slices and
                 aliases resolve to the storage they read); footprint per group =
                 min(full size, max(union of proven access boxes, largest single access))
                 (exact when every access resolved to a box, a lower bound otherwise)
output writes:   one entry per written external tensor name (max over writers); declared
                 model outputs that no traced op wrote are charged once
fused_elements = fused_prefetched_elements = Σ footprints + Σ unique output writes
fused_bytes    = Σ footprint × width(dtype) + Σ output elements × width(dtype)
external_input_bytes / external_output_bytes report the two terms separately
```

The elements total and the bytes total are computed from the **same** footprints; they
differ only by the per-tensor widths (a graph whose boundary tensors are all fp32 has
`fused_bytes = 4 × fused_elements`). For GPT-2 at bs32/seq256 the saved difference from
`4 × fused_elements` comes from **recorded int64 accesses**: 256 counted int64 elements at
`Model.add_1`, 1,024 B over uniform fp32 pricing. It does **not** come from the token
indices: the `[32, 256]` int64 token-ID read at `Model.embedding` is recorded as zero, so
its 65,536 B are *missing* from the total ([Gap 3](#gap-3-known-holes-in-the-compulsory-io-accounting),
issue #47). These are arithmetic attributions of the saved artifact, not a corrected or
certified whole-model total. Canonical identities, true returned outputs, overlapping or disjoint regions,
shared or tied weights and storage widths all have to be *established* from the captured
graph for the number to be a complete bound; an access region handled by the maximum
read-count rule is a bound, not a measured footprint.

### Roofline Application ([`perf_model.py` L267–L268](../solar/perf/perf_model.py#L267-L268), [L173–L177](../solar/perf/perf_model.py#L173-L177) per-dtype mode)
```
fused_total_cycles = max(compute_cycles, fused_bytes / DRAM_byte_per_cycle)
runtime_ms         = fused_total_cycles / (freq_GHz × 1e6)
```

A single whole-graph roofline is applied to the graph-wide totals; there is no per-op
roofline. Intermediate tensors are assumed to stay on chip. Which `fused_bytes` enters the
roofline depends on the perf-model mode, see [Physical byte widths](#physical-byte-widths).

### When to Use
- Operator fusion scenarios (e.g., cuDNN fused kernels)
- When intermediate tensors fit in L2 cache
- As the optimistic compulsory-I/O bound for perfectly fused execution; it is **not** a
  realistic runtime estimate (see the fused-traffic principle above and
  [Gap 1](#gap-1-all-three-models-use-one-whole-graph-roofline))

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Model_Input + Weight, intermediate stays in cache
Layer 2 (ReLU):   No DRAM access (intermediate in cache)
Layer 3 (Linear): Read Weight, Write Model_Output
```

---

## 3. Fused+Prefetched SOL

### Description
A **single roofline** is applied to the entire graph. Total compute and total memory accesses (weights + model I/O) are aggregated, assuming perfect overlap between compute and memory operations.

### Memory Calculation ([`graph_analyzer.py` L2210–L2236](../solar/analysis/graph_analyzer.py#L2210-L2236))
```
fused_prefetched_elements = fused_elements          (same deduplicated footprints)
fused_prefetched_bytes    = fused_bytes
```

> **Note**: `fused_prefetched_*` is assigned from the same deduplicated totals as `fused_*`
> (older **guide text** described a per-layer `model_io_elems` sum for both; the earlier baseline
> implementation already assigned both from the deduplicated unique-external sums). The
> two models produce the same result. See [Section 6](#6-known-implementation-gaps).

### Roofline Application ([`perf_model.py` L268](../solar/perf/perf_model.py#L268))
```
fused_prefetched_total_cycles = max(compute_cycles, fused_prefetched_bytes / DRAM_byte_per_cycle)
runtime_ms                    = fused_prefetched_total_cycles / (freq_GHz × 1e6)
```

### When to Use
- Best-case performance estimate
- Highly optimized implementations with prefetching
- Compute-bound workloads
- Upper bound on achievable performance

### Example
For a simple `Linear → ReLU → Linear` network:
```
Total MACs   = MACs(Linear1) + MACs(Linear2)          (ReLU is other_ops, excluded from compute_cycles)
Total Memory = Model_Input + Weight1 + Weight2 + Model_Output   (each once, at its stored width)
Single whole-graph roofline applied to (Total MACs, Total Memory)
```

---

## 4. Comparison Summary

| Aspect | Unfused | Fused | Fused+Prefetched |
|--------|---------|-------|------------------|
| Intermediate tensors | Counted | Excluded | Excluded |
| External tensors | Once per consuming op | Once per unique storage region | Once per unique storage region |
| Roofline granularity | Whole graph (graph-wide sums) | Whole graph (graph-wide sums) | Whole graph (graph-wide sums) |
| Memory assumption | All from DRAM | Perfect fusion/reuse, no capacity limit | Perfect fusion/reuse, no capacity limit |
| What the number is | Pessimistic traffic model | Optimistic compulsory-I/O bound | Same as fused (*) |

(*) In the current implementation, fused and fused_prefetched produce identical results.
None of the three is a measured or achievable runtime.

## 5. Output Fields

### analysis.yaml
```yaml
total:
  macs: 1000000                    # Total multiply-accumulate operations
  flops: 2000000                   # Total floating-point operations (2 × MACs)
  other_ops: 50000                 # Total elementwise/reduction ops (CUDA-core work)
  unfused_elements: 25000000       # Unfused memory elements (all tensor I/O)
  fused_elements: 10000000         # Deduplicated external I/O elements (intermediates excluded)
  fused_prefetched_elements: 10000000  # Same as fused_elements
  model_io_elements: 2500000       # Model input/output elements (non-weight external I/O)
  intermediate_elements: 15000000  # Intermediate tensor elements (never added to fused traffic)
  weight_elements: 7500000         # Parameter footprint: distinct parameters per consuming op
  weight_read_elements: 7500000    # DRAM reads of those parameters (traffic basis)
  # weight_elements_unresolved / weight_provenance_unresolved_layers appear
  # only when a weight-typed slot has no parameter provenance (status incomplete)
  fused_bytes: 40000000            # Same footprints priced at each tensor's stored dtype width
  fused_prefetched_bytes: 40000000
  unfused_bytes: 100000000
  intermediate_bytes: 60000000
  external_input_bytes: 30000000   # unique external reads (model inputs + weights)
  external_output_bytes: 10000000  # required output writes
metadata:
  precision: fp32                  # selects the MAC rate; also the fallback width for unknown dtypes
  bytes_per_element: 4
  bytes_accounting: per-tensor-dtype
```

Per layer, `model_io_elements`, `input_elements`, `output_elements`, `memory_elements`
and `bytes_per_element` (per operand) are diagnostic views of that layer's own accesses;
the deduplication happens only in `total`.


`weight_read_elements + model_io_elements + intermediate_elements == unfused_elements`
holds per layer, and `weight_read_elements + model_io_elements` is the per-layer
`fused_elements`.  In the totals, `weight_elements`, `weight_read_elements` and
`model_io_elements` are plain layer sums (`total.model_io_elements` is NOT the
deduplicated footprint), while `total.unfused_elements` / `total.fused_elements`
also carry graph-level model I/O no layer owns (declared outputs, gather
indices) and the fused total is the deduplicated external footprint.
`weight_elements` is the parameter FOOTPRINT (each distinct parameter once per
consuming op, by the converter's `parameter_sources` identity); it can differ
from `weight_read_elements` — gathers read rows (fewer or more elements than
the table), a parameter in two slots is read twice — and
`perf.memory_breakdown.weight_elements` carries this footprint, so the
breakdown's three parts are not a partition of fused traffic whenever
footprint and reads differ.  See EINSUM_GUIDE.md, "Tensor Types", for the
provenance rules and the unknown-provenance marker.

### Physical byte widths

The analyzer prices every tensor at the physical width of its **recorded** dtype
([`graph_analyzer.py` L944–L962](../solar/analysis/graph_analyzer.py#L944-L962)):

| Recorded dtype | Bytes per element |
|---|---:|
| fp64, int64, complex64 | 8 |
| fp32, int32, tf32 (stored as fp32) | 4 |
| bf16, fp16, int16 | 2 |
| bool, int8, uint8, fp8 (`e4m3`, `e5m2` variants) | 1 |
| packed FP4 `float4_e2m1fn_x2` | 1 per **already-packed** element (two 4-bit values); never halved again |

A tensor whose dtype or storage provenance was not recorded is priced at the `--precision`
fallback width and that is an **explicit incompleteness** of the result, not a convention.
This table does not convert data: fp32 tensors stay 4 B, and no TF32 throughput is
selected automatically (see the units box above).

The perf model has two modes:

* **Default (uniform width).** `memory_bytes = elements × bytes_per_element(--precision)`
  for all three models. The KB regeneration recipe (`analyze_model --precision fp32` and
  `predict_perf_model --precision fp32`) runs in this mode, so its SOL numbers price every
  tensor at 4 B regardless of the per-dtype `fused_bytes` in `analysis.yaml`.
* **Per-dtype (`predict_perf_model --dtype-bytes`).** When the analysis carries
  `bytes_accounting: per-tensor-dtype`, the per-tensor-width totals (`unfused_bytes`,
  `fused_bytes`, `intermediate_bytes`, `external_*_bytes`) enter the roofline and
  `--precision` only picks the MAC rate ([`perf_model.py` L225–L247](../solar/perf/perf_model.py#L225-L247), [L173–L177](../solar/perf/perf_model.py#L173-L177)).
  Display caveat inherited from an earlier change: in this mode the weight bytes are folded into the
  external-input bytes at their own widths and `memory_breakdown.weight_bytes` is set to
  **0** ([`perf_model.py` L403–L411](../solar/perf/perf_model.py#L403-L411) at an earlier revision and
  a later revision), so that breakdown is **not** a parameter-byte inventory; `weight_elements`
  (above) remains the footprint count.
* **Declared boundary widths (the one exception in the default mode).** When the
  analysis carries `model_io_bytes_declared` / `external_io_bytes_declared_dedup` (the
  caller bound the benchmark definition's input/output storage dtypes), declared boundary
  inputs and outputs, including declared parameter inputs, use their declared storage
  widths instead of the uniform `--precision` width; declared weight footprints are
  repriced on the parameter-footprint basis (`footprint_elements_weight` /
  `bytes_footprint_weight`, M97-WEIGHT-2: a 32-element parameter used in two slots, uniform
  4 B and declared 1 B, moves `weight_bytes` from 128 B to `128 + (32 − 32 × 4) = 32` B
  although it is read twice, 64 raw reads), while non-boundary weights and intermediates
  retain uniform pricing; the perf output records `boundary_billing:
  declared_storage_dtypes` ([`perf_model.py` L258–L305](../solar/perf/perf_model.py#L258-L305),
  footprint repricing at L285–L294). Under `--dtype-bytes` the per-tensor totals of
  L403–L411 then replace these byte totals.

### perf_<arch>.yaml
```yaml
sol_convention:                    # issue #58 option B: what the headline is and is not
  id: issue58-B
  headline: optimistic contraction-and-memory bound
  statement: ...                   # compute term = contraction MACs + other non-contraction MACs; scalar statement multiplies and other_ops excluded
  limits: [...]                    # not a bound over all compute resources; no "hidden/free" claim
workload:
  total_macs: 1000000
  macs_typed: {...}                # contraction by compute dtype + scalar statement multiplies (counts, unchanged)
  default_rate_remainder:          # what the default-rate remainder is made of
    macs: 1000000
    macs_priced_in_tensor_core_term: 999000
    scalar_statement_multiplies: 1000
  tensor_core_term:                # the term's composition
    convention: issue58-B
    applied: true                  # false (with reason) when the analysis predates macs_typed
    macs_priced: 999000
    contraction_macs_priced: 999000
    non_contraction_other_macs_priced: 0
    scalar_statement_multiplies_excluded: 1000
    excluded_cycles_at_applied_rate: 0.0017
    rate_key: MAC_per_cycle_fp16_tc
    rate_class: tensor_core
unfused:
  memory_bytes: 50000000
  compute_cycles: 2645
  memory_cycles: 49049
  total_cycles: 49049
  runtime_ms: 0.025
  arithmetic_intensity: 0.04
  bottleneck: memory

fused:
  memory_bytes: 20000000
  compute_cycles: 2645
  memory_cycles: 19619
  total_cycles: 19619
  runtime_ms: 0.010
  arithmetic_intensity: 0.1
  bottleneck: memory

fused_prefetched:
  memory_bytes: 20000000
  compute_cycles: 2645
  memory_cycles: 19619
  total_cycles: 19619
  runtime_ms: 0.010
  arithmetic_intensity: 0.1
  bottleneck: memory

speedup:
  fused_vs_unfused: 2.5
  fused_prefetched_vs_unfused: 2.5
  fused_prefetched_vs_fused: 1.0
```

## 6. Known Implementation Gaps

The current implementation has these discrepancies from the design intent:

### Gap 1: All three models use one whole-graph roofline

The original design intended unfused and fused to use **per-op roofline sums**:
```
Intended unfused:          Σ_i max(compute_i, memory_i)   (per-op, summed)
Intended fused:            Σ_i max(compute_i, fused_memory_i)
Intended fused_prefetched: max(Σ compute, Σ fused_memory)  (whole-graph)
```

No version of the code, before or after the lineage base, has applied a per-op roofline. The
implementation applies the maximum to graph-wide sums for all three
([`perf_model.py` L267–L268](../solar/perf/perf_model.py#L267-L268)):
```
Actual unfused:            max(Σ compute, Σ unfused_memory)
Actual fused:              max(Σ compute, Σ fused_memory)
Actual fused_prefetched:   max(Σ compute, Σ fused_memory)
```

A per-op sum is always >= the whole-graph roofline (`Σ max(c_i, m_i) >= max(Σ c_i, Σ m_i)`), so the unfused and fused estimates are more optimistic than intended. Any prose elsewhere that describes per-op rooflines describes the intent, not the code.

### Gap 2: Fused and fused_prefetched are identical

`fused_prefetched_elements` / `fused_prefetched_bytes` are assigned from the same
deduplicated totals as `fused_elements` / `fused_bytes`
([`graph_analyzer.py` L2210–L2236](../solar/analysis/graph_analyzer.py#L2210-L2236)).
The baseline the earlier baseline already assigned both from the same deduplicated unique-external
sums (only older guide text described a per-layer `model_io_elems` sum). The runtimes are
therefore identical.

### Gap 3: Known holes in the compulsory-I/O accounting

The fused total is a complete compulsory-I/O bound only when every required external
read is present. Known omissions at the lineage base, and their state on the current target:

* **Embedding token-index reads were zeroed at the lineage base.** There, for `embedding` /
  `embedding_bag`, the analyzer set every input read to zero before assigning the
  gathered-row read of the weight, so the external index tensor was never charged: GPT-2
  bs32/seq256 omitted the `[32, 256]` int64 token ids (8,192 elements, 65,536 B) and its
  saved fused total of 541,847,040 elements was not a complete external-I/O minimum
  (**issue #47**). The earlier decision gather rule merged into the target reads every index
  tensor fully, the source only at the gathered elements, and bills one address
  computation per output element
  ([`graph_analyzer.py` L1724–L1760](../solar/analysis/graph_analyzer.py#L1724-L1760); see
  "Gather, index_select and embedding traffic" in `EINSUM_GUIDE.md`). The saved GPT-2
  artifact quoted in this guide predates that rule and is not re-certified here.
* The `model_io_elements` diagnostic and per-layer `memory_elements` are not deduplicated
  and must not be summed to reproduce `fused_*`.

Multi-stream concurrency and programmatic dependent launch are outside this model; the SOL figures assume one stream.

## Practical Guidance

1. **Use Unfused** when:
   - Evaluating baseline performance
   - No fusion optimizations available
   - Debugging memory bottlenecks

2. **Use Fused** when:
   - Estimating performance with standard fusion (cuDNN, etc.)
   - Intermediate tensors fit in L2 cache
   - As the optimistic compulsory-I/O bound for perfectly fused execution; it is **not** a
     realistic runtime estimate (see "Fused SOL" and Gap 1)

3. **Use Fused+Prefetched** when:
   - Estimating best-case performance
   - Highly optimized custom kernels
   - Setting performance targets

## References

- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures.
- NVIDIA cuDNN documentation on operator fusion
- PyTorch compile / TorchInductor fusion strategies
