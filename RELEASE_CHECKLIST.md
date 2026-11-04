<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release checklist

Status of the standard release checks for the official release candidate, recorded
by the producer named in each row. A check is marked PASS only when it was executed
and its evidence is recorded; NOT RUN means exactly that. Updated 2026-11-04 for the
official release candidate of this version (SOLAR 2.0.0).

| Check | Status | Producer | Evidence |
|---|---|---|---|
| Clean `git status` on the release tree | **PASS** | author | 0 modified/untracked paths |
| sdist builds offline from a clean `git archive` of the release tree (`setuptools.build_meta.build_sdist`) | **PASS** | author | candidate sdist (`solar-2.0.0.tar.gz`) from the completed packaging checks, 483 regular files; final candidate artefact binding pending |
| wheel builds from the unpacked sdist (`build_wheel`) | **PASS** | author | candidate wheel (`solar-2.0.0-py3-none-any.whl`) from the completed packaging checks, 168 members, 8 console scripts; final candidate artefact binding pending |
| archive member scan: no benchmark packaging, batch CLIs, dead modules, campaign fixtures, `examples_verified`, logo or retired docs in sdist or wheel; no tests/caches in the wheel | **PASS** | author | every `no_*` check true (artifact_checks.json) |
| `solar/einsum_lib/fallback_protocol/PROTOCOL.json` + `PROTOCOL.md` present in BOTH artefacts | **PASS** | author | member checks true; files present next to `fallback_contract.py` in both installs; `tests/test_fallback_contract.py` 510 passed against the installed wheel and the installed sdist |
| licence files in wheel metadata and sdist (`LICENSE`, `THIRD_PARTY_NOTICES`, `REUSE.toml`, `LICENSES/Apache-2.0.txt`); METADATA says Apache-2.0 and never MIT | **PASS** | author | artifact_checks.json |
| isolated install of the wheel (venv, `pip --no-index --no-deps`); 8 console scripts `--help`; `python -m solar.resources`; mechanical pipeline on the packaged TwoInputMatmul outside the checkout; `solar-translate verify` on the shipped RMSNorm example | **PASS** | author | 262144 MACs / 14336 elements / 28672 bytes; `VERIFIED (max_abs_error=3.411e-07)`; INSTALL_SMOKE-*.log |
| isolated install of the sdist (same checks) | **PASS** | author | INSTALL_SMOKE_SDIST-*.log (venv created `--without-pip` so the host pip 25.2 / setuptools 80 build it; the first two attempts failed for host reasons and their logs are kept) |
| removed surfaces absent from the installed packages with the host editable-finder leak excluded | **PASS** | author | INSTALL_SMOKE_SDIST_AND_LEAKPROOF-*.log, INSTALL_SMOKE_SDIST-*.log |
| `tests/release_packaging/test_installed_resources.py` (builds its own wheel) | **PASS** | author | 10 passed |
| bounded tests of changed dependencies: `test_graph_processing`, `test_integration`, `test_packaging_metadata`, `test_dimspec`, `test_translate_facade`, `test_campaign_repairs`, `test_skills`, `test_skill_claims`, `test_fixture_coverage_map`, `test_alias_view_accounting` | **PASS** | author | 61 + 39 (4 xfail) + 103 passed |
| public test modules split from the benchmark-derived fixtures still pass | **PASS** | author | wave 1: 1324 passed (7 modules + coverage map, skills, packaging); wave 2: 133 passed (access_witness, af, weight_elements, coverage map, scalar_index_af); the restored scalar-argument control: 21 passed in its module |
| `scripts/run_tests.sh quick` and `scripts/run_tests.sh examples` (repaired runner) | **PASS** | author | 2 passed; 5 examples passed with all expected outputs |
| private regression checks (the removed fixture cases, both waves) against the installed wheel, with import-origin proof | **PASS** | author | private harness RUNS-* and overlay-2 RUNS-*: every original consumer module passes against the installed package; the former checkout-layout assumption of `test_scaled_mm_accounting.py` rounds 11-14 is gone (architecture profiles now resolve through `solar.resources.arch_config_path`; overlay-1 385 passed) |
| full unit/regression suite | **NOT RUN here** | CI | the GitHub and GitLab pipelines on the MR run it; no claim is made beyond what they show |
| `pylint --errors-only` (dev tool declared in pyproject) | **FAIL (pre-existing debt, reduced)** | author | 44 -> 38 messages; the 6 that disappeared belong to removed modules or the `Set` import fix; the rest are pre-existing (mostly `no-member` on dynamic attributes) |
| `pylint` default | **FAIL (pre-existing debt, reduced)** | author | 6308 -> 6099 messages on `solar/` |
| `mypy --ignore-missing-imports solar` | **FAIL (pre-existing debt, reduced)** | author | 499 -> 461 errors; no `name-defined` left |
| ruff 0.17.3 (`check --isolated --no-cache`, reviewer-identical binary and command; added read-only check) | **FAIL (pre-existing debt)** | author | baseline 5941 total / 3258 in `solar/`; after 4359 / 3030; 1584 baseline diagnostics were in removed files (recorded separately); both package F821 sites gone |
| content scan of the tree for internal hosts, paths, credentials | **PASS** | author | only public datasheet URLs in `configs/arch`; API-key references are documented parameters, not secrets |
| markdown links resolve | **PASS** | author | 0 broken relative links over every `.md` |
| licence placement for every tracked file | **PASS (inventory), with reported gaps** | author | LICENSE_COVERAGE record: inline SPDX header, REUSE.toml annotation, or licence text; benchmark-derived inputs are no longer in the tree |
| numerical / E04 / benchmark evidence | **NOT RERUN** | author | no numerical or accounting module changed, so the prior numerical/E04 evidence is retained; package bindings DID change (solar/translate/_dimspec.py added and the facade import retargeted, solar/einsum/pytorch_to_einsum.py annotation, removed modules, packaging and test files), so reuse of that evidence awaits the release manager's rebinding to this head; it is not claimed here |
| independent review of this exact tree | **NOT RUN by the author** | independent reviewer | performed as a separate gate before publication |
| public push | **NOT DONE** | local manager | separate gate |

## Public / private test boundary

The public tree ships no benchmark-row-derived fixture input (SOL-ExecBench Lite rows
and candidates, kernelbench-derived tiny models) and no benchmark-derived saved model
captures. Authored generic synthetic controls (for example
`tests/fixtures/rnn_state_consumers/round4_lstm_capture_pytorch_graph.yaml`, described in
that directory's README) are deliberately retained.
The regression cases that consumed them were moved, unchanged, to a private,
hash-manifested harness maintained alongside the private source history and are run
against each built official package. Every retained fixture was audited for
provenance and dispositioned (FIXTURE_PROVENANCE_AUDIT record). The public modules
keep every self-contained case. Removed public cases, by module:

- `tests/test_loop_dim_extent_round4.py`: class TestReview6KBFixtures; PINS: the 8 kb/* entries (test_fixture_pins keeps the two public round4 pins); the self-contained control test_scalar_args_are_not_recorded_for_bools_or_tensors is kept as the public class TestScalarArgRecording
- `tests/test_loop_dim_extent_round7.py`: class TestKbReformerConcatLastControl; class TestLite58StableScatterControl; class TestLite229SwigluSlotsControl; helper _lite_fixture; module constants KBREF, LITE58, LITE229, KBREF_PINS, KBREF_SOURCE_SHA256, LITE_PINS, LITE_SOURCE_SHA256
- `tests/test_lowering_final_rounding.py`: class TestSavedRowsStatic (45 cases over the 20 saved Lite rows); helpers _changed_lines, _added_truncations, _strip_guards; module constants _LITE20_ADDED_TRUNCATIONS, _LITE20_UNCHANGED, _TARGET_PIN
- `tests/test_numba_narrow_carrier.py`: test_saved_case_228_program_returns_a_float32_output_carrier_and_float32_lse; test_saved_rows_70_107_122_return_bf16_rounded_outputs_whose_carrier_is_now_float32 (3 params); module constant FIX; helpers _returned_exprs, _last_assignment, _value_is_bf16_rounded (only used by the removed cases)
- `tests/test_prescan_allocation_coverage.py`: TestLiteralSpellingsStillLower.test_saved_fixtures_lower_again_under_the_round9_narrowing
- `tests/test_scaled_mm_accounting.py`: fixture route195; class TestRoute195Capture; class TestRoute195Receipt; TestRefusalPropagation.test_converter_records_the_refusal_instead_of_a_supportable_copy; module constant FIXTURE
- `tests/test_statement_unit_classification.py`: test_mamba_tiny_candidate_executes_four_macs; test_gru_tiny_candidate_executes_five_statements; test_lstm_candidate_tiny_mac_field_is_the_contraction_total; test_reformer_candidate_tiny_bias_and_scale_statements_are_other_work; test_lstm_candidate_tiny_external_output_is_the_returned_fc_only; test_round7_every_slot_of_every_accepted_kernel_is_visited (parametrised over ROUND7_CORPUS, which globbed the removed dsl_*.py and would otherwise SKIP on an empty parameter set); helpers _fixture_pipeline, _candidate_source, _lstm_inputs, _reformer_inputs; module constants FIX, ROUND7_CORPUS, MAMBA_TINY_SHA, GRU_TINY_SHA, LSTM_CANDIDATE_SHA, REFORMER_CANDIDATE_SHA
- `tests/test_loop_dim_extent_access_witness.py`: class TestReview6ControlsVerbatim (3 cases); class TestLite203TopkSelect (2 cases); helpers _pinned, _convert_dir; module constants CTRL, LITE, PINS, LITE_SOURCE_SHA256, T203/E203/K203, FALSE_E_MISMATCH (R4S_PINS kept)
- `tests/test_af_unresolved_operands.py`: test_mamba2_v2_capture_with_unresolved_operands_builds_and_keeps_one_rank_tuple_per_tensor; module constant FIXTURE, unused import gzip
- `tests/test_weight_elements.py`: class TestGPT2Fixture (4 cases); class TestMamba2Provenance (4 cases); helper _analyze_fixture; module constant FIXTURES; unused import gzip

## Known debt carried into the release (disclosed, not suppressed)

- Static analysis: the counts above are pre-existing debt of the shipped code; no
  suppressions were added and no code was reformatted. Fixes were limited to the
  undefined name `Set` (replaced by the builtin `set[tuple[str, str]]` annotation),
  imports left unused by the test split, and modules that were removed. The
  diagnostics in new or moved files (`solar/translate/_dimspec.py`, moved verbatim
  with its typing-style debt; `tests/test_dimspec.py`; `examples/TwoInputMatmul`)
  are new or moved, not pre-existing in place (LINT_ATTRIBUTION record).
- CI provisioning: the GitLab unit-tests and regression-tests jobs install the
  declared build backend (`setuptools>=61`) with `python3 -m pip install --user`
  before pytest; that writes into the runner account's persistent shared `~/.local`
  (the provisioning pattern validate-install already uses), it is not a
  job-isolated step.
- The numerical control for the portable window padding/crop fixture
  (`tests/release_portable/pending_numerical_window_padding_crop.py`) remains PENDING
  an independent run, as before this cleanup.

## TODO before publication

- [ ] Independent review of this exact tree.
- [ ] CI green on both pipelines for this head.
- [ ] Bind retained validation evidence to the final release candidate, documenting
      which checks remain applicable after the changes.
- [ ] Public push by the local manager.
