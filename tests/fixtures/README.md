<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# tests/fixtures

Saved inputs for the regression tests: captured graphs, saved einsum graphs,
lowered programs and control probes. Everything under this directory is part
of SOLAR and licensed under the Apache License 2.0 (`LICENSE` at the
repository root).

## Header-free fixture files (licence sidecar)

Benchmark-row-derived DSL inputs and the programs lowered from them (SOL-ExecBench Lite rows,
kernelbench-derived tiny models) are not part of the public tree since 2026-11-04; the regression
checks that consume them run privately against the built package (see `RELEASE_CHECKLIST.md`).

Two exact sets of first-party fixture files carry no SPDX header line; this statement covers
their licensing placement (Apache License 2.0, `LICENSE` at the repository root), exactly as
`patches/README.md` covers the patch containers.

### Checksum-bound files (recorded checksums / byte-for-byte test claims)

These files keep their recorded bytes exactly; their checksums are asserted by the tests and
recorded in the sidecar manifests next to them.

| file (relative to `tests/fixtures/`) |
|---|
| `loop_evidence/extended_round4.py` |
| `loop_evidence/statement_probes.py` |
| `loop_evidence/probe_additional.py` |
| `loop_evidence/probe_extended_v2.py` |
| `loop_evidence/probe_recompute2.py` |
| `rnn_state_consumers/round4_lstm_einsum_hhc_saved.yaml` |
| `rnn_state_consumers/round4_lstm_einsum_hhc_unmarked.yaml` |
| `rnn_state_consumers/round8_matrix_permuted.yaml` |
| `rnn_state_consumers/round8_matrix_valid.yaml` |
| `rnn_state_consumers/round8_mixed_boundary_explicit_edge.yaml` |
| `rnn_state_consumers/round8_mixed_boundary_start_slot.yaml` |
| `rnn_state_consumers/round8_mixed_boundary_without_edge.yaml` |
| `rnn_state_consumers/round4_lstm_capture_pytorch_graph.yaml` |
| `issue62/analysis.yaml` |
| `issue62/perf_dtype_H100_PCIe.yaml` |
| `issue62/perf_uniform_H100_PCIe.yaml` |

These three synthetic issue62 golden fixtures carry their Apache License 2.0 attribution through this sidecar and the repository-root LICENSE. They have no inline SPDX header because the regression tests verify their recorded bytes and compare generated reports against them byte for byte. Their original fixture-record hashes and sizes remain unchanged. This is a license-placement exception for exactly these files:
- `tests/fixtures/issue62/analysis.yaml`
- `tests/fixtures/issue62/perf_dtype_H100_PCIe.yaml`
- `tests/fixtures/issue62/perf_uniform_H100_PCIe.yaml`

### Fixture sources held by their consuming tests (57 files)

Verbatim control modules, sealed round artifacts and lowered carriers that the tests load by
path or compare byte-for-byte. Where a docstring or comment of one of these files named
internal review material, this release carries neutral wording instead, and every SHA-256 the
consuming tests record for such a file was updated in the same change; kernel bodies and
executable statements are unchanged. Listed exactly, by directory:

**`accounting_units/`** (0 files)


**`loop_dim_extent/`** (18 files)

- `loop_dim_extent/round4/checks-00ce6d18/test_legacy_trace.py`
- `loop_dim_extent/round4/checks-00ce6d18/test_round3_edges.py`
- `loop_dim_extent/round4/checks-cf6251bc/runs/order-head/keyword_order/dsl/probe_einsum_lib.py`
- `loop_dim_extent/round4/checks-cf6251bc/runs/order-head/keyword_order/dsl/probe_einsum_lib.yaml`
- `loop_dim_extent/round4/checks-cf6251bc/runs/order-head/last_wins_swapped/dsl/probe_einsum_lib.py`
- `loop_dim_extent/round4/checks-cf6251bc/runs/order-head/last_wins_swapped/dsl/probe_einsum_lib.yaml`
- `loop_dim_extent/round5/checks-9b85d77c/test_round4_edges.py`
- `loop_dim_extent/round6/checks-ea874283/test_round5_binop.py`
- `loop_dim_extent/round6/checks-ea874283/test_round5_edges.py`
- `loop_dim_extent/round7/checks-ed9be3d0/test_round6_edges.py`
- `loop_dim_extent/round8/checks-2f97bdbf/test_round5_edges.py`
- `loop_dim_extent/round8/checks-2f97bdbf/test_round7_edges.py`
- `loop_dim_extent/round8/checks-2f97bdbf/test_round7_nested_complete_v2.py`
- `loop_dim_extent/test-output/provenance_head/test_parsed_body_wins_over_bui0/dsl/shadow_einsum_lib.py`
- `loop_dim_extent/test-output/provenance_head/test_parsed_body_wins_over_bui0/dsl/shadow_einsum_lib.yaml`
- `loop_dim_extent/test_independent_dim_extent.py`
- `loop_dim_extent/test_loop_dimension.py`
- `loop_dim_extent/test_registry_order.py`

**`lowering_final_rounding/`** (15 files)

- `lowering_final_rounding/review3_controls/class_global_rebind.py`
- `lowering_final_rounding/review3_controls/foreign-intp-mutant.py`
- `lowering_final_rounding/review3_controls/mini-after.py`
- `lowering_final_rounding/review3_controls/mini-before.py`
- `lowering_final_rounding/review3_controls/numeric-mutant.py`
- `lowering_final_rounding/review3_controls/test_followup_round7.py`
- `lowering_final_rounding/review3_controls/test_independent_round7.py`
- `lowering_final_rounding/review3_controls/test_isolation_direct.py`
- `lowering_final_rounding/review3_controls/test_reflective_attributes.py`
- `lowering_final_rounding/review3_controls/test_review4_round9.py`
- `lowering_final_rounding/review3_controls/test_review7_round4.py`
- `lowering_final_rounding/review3_controls/test_review7_round5.py`
- `lowering_final_rounding/review3_controls/test_round6_adversarial.py`
- `lowering_final_rounding/review3_controls/test_round8_direct.py`
- `lowering_final_rounding/review3_controls/try_prefix_rebind.py`

**`perf_refusal_controls/`** (16 files)

- `perf_refusal_controls/bf16_macs_plus_fp4_traffic_metadata/analysis.yaml`
- `perf_refusal_controls/bf16_macs_plus_fp4_traffic_metadata/metadata.yaml`
- `perf_refusal_controls/first_refused/analysis.yaml`
- `perf_refusal_controls/first_refused/metadata.yaml`
- `perf_refusal_controls/fp4_traffic_only_no_metadata/analysis.yaml`
- `perf_refusal_controls/fp4_traffic_only_with_metadata/analysis.yaml`
- `perf_refusal_controls/fp4_traffic_only_with_metadata/metadata.yaml`
- `perf_refusal_controls/fp8_traffic_only_missing_rate/analysis.yaml`
- `perf_refusal_controls/fp8_traffic_only_missing_rate/metadata.yaml`
- `perf_refusal_controls/mixed_analysis_only/analysis.yaml`
- `perf_refusal_controls/mixed_analysis_plus_metadata/analysis.yaml`
- `perf_refusal_controls/mixed_analysis_plus_metadata/metadata.yaml`
- `perf_refusal_controls/mixed_b200/analysis.yaml`
- `perf_refusal_controls/profile_H100_without_fp8.yaml`
- `perf_refusal_controls/then_successful/analysis.yaml`
- `perf_refusal_controls/zero_mac_dtype_entry/analysis.yaml`

**`returned_outputs_review7/`** (8 files)

- `returned_outputs_review7/probe_grid_cost.py`
- `returned_outputs_review7/test_live_alias_review2.py`
- `returned_outputs_review7/test_review2_regions.py`
- `returned_outputs_review7/test_review7_aliases.py`
- `returned_outputs_review7/test_review7_fallback_identity.py`
- `returned_outputs_review7/test_review7_large_disjoint.py`
- `returned_outputs_review7/test_review7_large_stride.py`
- `returned_outputs_review7/test_round4_review2.py`

All other files under `tests/fixtures/` carry the standard SPDX header; nothing outside these
two lists is exempt.
