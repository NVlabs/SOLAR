<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hardware profiles (`configs/arch/<name>.yaml`)

A profile is a flat YAML mapping of numbers read by the perf model
(`solar-predict-perf-model --arch-config <name>|<path>`). The roofline uses
`freq_GHz`, `DRAM_byte_per_cycle`, `MAC_per_cycle_fp32_sm` and the tensor-core
rate of the selected precision; `DRAM_capacity` and `SRAM_capacity` are
informational.

## Dense and sparse tensor-core peaks (issue #14)

Vendor datasheets print tensor-core rates either "with sparsity" (the 2:4
structured-sparsity rate, twice the dense rate) or as `dense | sparse` pairs.
Every shipped profile therefore carries, for every tensor-core precision `<p>`
it declares (`tf32`, `fp16`, `bf16`, `fp8`, `int8`, `nvfp4`, `fp64`):

| key | meaning |
|---|---|
| `peak_TFLOPS_<p>_dense` | the dense peak as published (TFLOPS; TOPS for integer types) |
| `peak_TFLOPS_<p>_sparse` | the structured-sparsity peak as published |
| `MAC_per_cycle_<p>_tc_dense` | `peak_TFLOPS_<p>_dense x 1e12 / 2 FLOP per MAC / (freq_GHz x 1e9)` |
| `MAC_per_cycle_<p>_tc_sparse` | the same derivation from the sparse peak |
| `peak_TFLOPS_fp32_sm`, `MAC_per_cycle_fp32_sm` | the CUDA-core (SM) FP32 rate; no sparsity variant |

Units: `MAC_per_cycle = TFLOPS x 1e12 / 2 / freq_Hz = TFLOPS x 1000 / 2 / freq_GHz`.
`MAC_per_cycle_*` values are MACs per cycle, not FLOPs (one FMA is one MAC and
two FLOPs). `freq_GHz` is a nominal conversion clock: predicted runtimes depend
only on the peaks. Every value line carries a `# source:` (the datasheet or
product page and the printed figure) or a `# derived:` note with the rule;
where the vendor publishes no sparse figure (FP32 on the CUDA cores, FP64) the
two sets are equal and the comment says so. There is no `fp32` tensor-core key:
the fp32 route falls back to the SM rate.

## Which set prices the SOL: `--sol-mode`

`solar-predict-perf-model --sol-mode {sparse,dense}` (Python:
`EinsumGraphPerfModel().predict(..., sol_mode=...)`, `solar.estimate.estimate(...,
sol_mode=...)`, `solar.translate.sol(..., sol_mode=...)`) selects the peak set
the top-level `unfused` / `fused` / `fused_prefetched` numbers are priced at.
The default is **sparse**.

- The **sparse SOL assumes** the workload is eligible for the vendor's 2:4
  structured-sparsity tensor-core rate. It is a lower bound that no dense
  kernel can reach; it does not claim that any particular workload satisfies
  2:4 sparsity.
- The **dense SOL is the more realistic estimate** (the realistic bound) for
  dense work.

Every perf record carries both, whatever the mode: `sol_sparse` and
`sol_dense` (per scenario `compute_tc_cycles`, `total_cycles`, `runtime_ms`,
`bottleneck`, plus `rate_keys` and `rates`), `sol_mode`, `sol_mode_default`,
and `arch.sol_rate_resolution` (which explicit key each bare rate came from).
`result.json` of `solar.estimate` carries the same two blocks.

## Legacy profiles

A user profile that spells a tensor-core rate with the bare key
`MAC_per_cycle_<p>_tc` declares a **dense** rate (that was always its meaning).
It has no sparse figure, so sparse mode prices it at the dense rate and records
`sparse_fallback_to_dense: true` for that precision; its sparse and dense SOLs
are then equal. A bare key present next to explicit `_dense` / `_sparse` keys
is treated as a legacy single-rate override: it is applied in BOTH modes and the
resolution record notes `bare_key_overrides_explicit: true` (existing tests and
user scripts inject bare keys to override one rate, and they keep working).

## Provenance of the shipped figures

| profile | source (specification-table excerpts under `sources/arch_datasheets/`, see `SOURCES.json`) | notes |
|---|---|---|
| H100_PCIe | NVIDIA H100 Tensor Core GPU datasheet PDF (NVIDIA-authored; the copy was fetched from a partner mirror, see `SOURCES.json`), "H100 PCIe" column (80 GB, 2 TB/s); tensor-core rows "Shown with sparsity. Specifications 1/2 lower without sparsity" (the nvidia.com page now lists SXM and NVL only) | dense = half of the printed figure; FP64 TC sparse == dense |
| H100_SXM | nvidia.com/en-us/data-center/h100/, "H100 SXM" column | as above |
| H200_SXM | nvidia.com/en-us/data-center/h200/ | compute equals H100 SXM; 141 GB, 4.8 TB/s |
| L40S | NVIDIA L40S datasheet, `dense | sparse` pairs as printed | FP16/BF16 362.05 / 733 is not an exact 2x (ratio 2.0246); both kept as printed |
| B200 | nvidia.com HGX page, HGX B200 column (per 8-GPU system; "Specification in Sparse. Dense is 1/2 sparse spec shown", FP4 printed "Sparse \| Dense") and the DGX B200 page (1,440 GB, 64 TB/s) | per-GPU = system figure / 8; replaces the former unsourced "existing SOLAR model rates"; 180 GB, 8 TB/s per GPU |
| A6000 | GA102 architecture whitepaper v2, Table 3 RTX A6000 column, pairs printed dense/sparse (FP16/BF16 154.8/309.6, TF32 77.4/154.8, INT8 309.7/619.4); RTX A6000 datasheet (FP32 38.7, "Tensor performance 309.7 TFLOPS" with sparsity) | the former file stored FLOP/cycle as MAC/cycle (= the sparse MAC rate); relabelled, dense = half; estimated fp32 TC key dropped |
| Jetson_Thor_T5000 | NVIDIA Jetson Thor page (2070 TFLOPS FP4 sparse, 1.57 GHz, 273 GB/s) and the NVIDIA Jetson Thor technical blog, Table 1 ("1035 TFLOPS (Dense FP4 \| Sparse FP8 \| Sparse INT8)", "517 TFLOPs (Dense FP8 \| Sparse FP16)") | FP16 dense, BF16, TF32 and INT8 dense are **derived** (unpublished); FP32 15 TFLOPS is **unverified-from-source** (module data sheet unreachable, HTTP 404); the former file stored FLOP/cycle as MAC/cycle; FP32 SM corrected from 9554 to 4777 |

Figures that no fetched NVIDIA-published document states are marked
`unverified-from-source` in `SOURCES.json` and in the yaml comment;
maintainer-relayed numbers are never cited as sources.
