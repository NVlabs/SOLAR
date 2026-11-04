# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Issue #62 -- perf_model ``memory_breakdown`` report METADATA (L170-AUX-MEMBREAK-MIXED-BASIS-1, P3 presentation).

The three auxiliary categories (weight / model_io / intermediate) mix counting bases: ``*_elements`` are raw per-op
counters in both modes; under dtype storage mode ``weight_bytes`` is a folded-category sentinel 0 and ``model_io_bytes`` is
the deduplicated external boundary, while under uniform width both are raw per-op accesses.  They are therefore NOT a general
additive physical partition.  The repair adds two keys INSIDE ``memory_breakdown`` -- ``auxiliary_non_additive: true`` and a
per-label ``semantics`` dict (mode-aware texts for ``weight_bytes`` / ``model_io_bytes``) -- and changes NOTHING else.

This module proves that claim:
* golden byte identity: the report the patched model writes for a fixed analysis, with the two metadata keys removed, is
  BYTE-IDENTICAL to the report the unpatched baseline wrote for the same analysis (fixtures/issue62/; pinned baseline revision, rebound from
  an earlier baseline after a later change added nine report-provenance leaves -- FIXTURE_RECORD.json records both);
* every pre-existing field equal and type-exact; no numeric token added anywhere in the YAML text;
* metadata content: the flag is a real bool, the texts follow round 8's LABEL_DISPOSITION (F_traffic / diagnostic_overlapping /
  defect-with-explanation; dtype-mode weight_bytes = folded sentinel; model_io_bytes basis per mode);
* YAML / JSON round trips of the new metadata;
* every in-repo consumer of ``memory_breakdown`` (CONSUMER_AUDIT) still produces IDENTICAL output with the metadata present.
The fixture is synthetic (two layers, one repeated external read); no sealed Lite 170 file is read.

Issue #58 convention B (an earlier change, target an earlier revision) reconciliation.  The approved convention-B report contract ADDS, for every report,
the top-level ``sol_convention`` block and ``workload.tensor_core_term``, adds ``workload.default_rate_remainder.macs_priced_in_tensor_core_term``
and APPENDS one sentence to ``workload.default_rate_remainder.note``; it changes no count, no cycle and no other field.  The golden
fixtures are the pre-convention baseline and are NOT refreshed (their bytes are protected): the golden tests below compare the report
with the issue-62 metadata AND the convention-B additions removed, and separately assert that the convention-B additions are EXACTLY
the authorized keys, types, values and numeric tokens (``CONVENTION_B``), so an unauthorized new field, changed value or numeric token
still fails.  The one altered pre-existing leaf is the note: its new text must equal the golden text plus the exact authorized suffix.
"""
import copy
import csv
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from solar.common.utils import NoAliasDumper
from solar.perf import EinsumGraphPerfModel

SOLAR_ROOT = Path(__file__).resolve().parent.parent          # <checkout>/solar
FIX = Path(__file__).resolve().parent / "fixtures" / "issue62"
ANALYSIS = FIX / "analysis.yaml"
GOLDEN = {"uniform": FIX / "perf_uniform_H100_PCIe.yaml", "dtype": FIX / "perf_dtype_H100_PCIe.yaml"}

# Internal benchmark collectors (scripts/collect_sol_b200.py, scripts/collect_perf_results.py, solar.benchmark.solbenchv3/v4
# collectors) and examples/compare_examples.py are not part of the public distribution; the consumer-compatibility tests
# that drive them are skipped when the consumer is absent and run unchanged where it ships.
def _needs(*rel):
    missing = [r for r in rel if not (SOLAR_ROOT / r).exists()]
    return pytest.mark.skipif(bool(missing), reason="not shipped in this distribution: " + ", ".join(missing))
MODES = ("uniform", "dtype")
META_KEYS = ("auxiliary_non_additive", "semantics")
NUMERIC_KEYS = ("weight_elements", "weight_bytes", "model_io_elements", "model_io_bytes", "intermediate_elements", "intermediate_bytes")
SEMANTICS_KEYS = ("scope",) + NUMERIC_KEYS + ("traffic_fields",)
# the round-8 label disposition record, mirrored here as the expectation the texts must follow
DISPOSITION = {"weight_elements": "diagnostic_overlapping", "model_io_elements": "diagnostic_overlapping",
               "intermediate_elements": "diagnostic_overlapping", "intermediate_bytes": "diagnostic_overlapping",
               "weight_bytes": "defect", "model_io_bytes": "defect"}
F_FIELDS = ("unfused.memory_bytes", "fused.memory_bytes", "fused_prefetched.memory_bytes")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# ---------------------------------------------------------------------------------------------------------------------
# issue #58 convention B (an earlier change): the ONLY report changes authorized on top of the pre-convention golden baseline.
# For the issue-62 fixture the analysis carries no total.macs_typed, so convention B is NOT applied (nothing excluded):
# every count below is derived from the golden's own numbers (total_macs 160 priced whole, 0 quantised, 0 excluded).
# ---------------------------------------------------------------------------------------------------------------------
# PROJECTION of the approved convention-B report changes, each justified by its perf_model.py source at an earlier revision
# (solar/solar/perf/perf_model.py sha256 d6cc7c9f…); the tests below check these POSITIVELY and everything else for INVARIANCE.
#   sol_convention                      : perf_model.py:531-546 SOL_CONVENTION constant, embedded verbatim at :1406 ("sol_convention": dict(SOL_CONVENTION))
#   workload.tensor_core_term           : perf_model.py:1384-1404 (dict), emitted at :1432; values from :1003 (_typed_components) and :1011
#                                         (macs_tc_default = macs_default - scalar_excluded), :1164/:1166 (compute_tc_cycles, scalar_excluded_cycles)
#   workload.default_rate_remainder.macs_priced_in_tensor_core_term : perf_model.py:626 (typed components absent branch) -- == macs_default
#   workload.default_rate_remainder.note (ALTERED, suffix appended)  : perf_model.py:627-630 (typed components absent branch)
#   compute_cycles = compute_tc_cycles unchanged                     : perf_model.py:1306; with scalar_excluded == 0 the term equals the pre-convention one
CONVENTION_B_ID = "issue58-B"                                                   # perf_model.py:533
CONVENTION_B_HEADLINE = "optimistic contraction-and-memory bound"              # perf_model.py:534
# The AUTHORIZED sol_convention block, copied VERBATIM from solar/solar/perf/perf_model.py:531-546 at an earlier revision (SOL_CONVENTION; the
# report embeds it at :1406).  Pinned here as literals on purpose (the review P2 live-convention-oracle-1): the test must NOT import
# the live producer constant as its own oracle, so a change to the statement, a limit or a disclaimer in the producer FAILS here.
SOL_CONVENTION_AUTHORIZED = {
    "id": CONVENTION_B_ID,
    "headline": CONVENTION_B_HEADLINE,
    "statement": ("runtime = max(compute_cycles, memory_cycles).  The tensor-core term compute_cycles prices the analysis' contraction MACs "
                  "(quantised-dtype buckets at their tensor-core rates, the workload-precision remainder at the applied MAC rate) plus any "
                  "other non-contraction MACs the analyzer records (macs_typed.non_contraction_other; retained in the term as before).  "
                  "Excluded from compute_cycles: the multiply statements of non-einsum kernels that the per-statement classifier counts as "
                  "MACs (macs_typed.scalar_statement_multiplies) and other_ops.  Both excluded counts are retained in the receipt; the "
                  "existing SM diagnostic compute_sm_cycles prices other_ops only (other_ops / MAC_per_cycle_fp32_sm, unchanged) and does "
                  "not include the scalar statement multiplies."),                                                              # :535-541
    "limits": ["not a bound accounting for all compute resources: the excluded SM-class statement work (scalar statement multiplies, other_ops) has no timing term in the headline",
               "the excluded work is not shown to be hidden by memory traffic or to be free; the exclusion is a bound convention, not a proof",
               "the per-statement MAC classification, every count, tolerance, tie, reference, cap and selection rule are unchanged"],   # :542-544
    "not_claimed": ["scalar work proven hidden", "scalar work free", "elementwise work absorbed by the memory term", "a bound over all compute resources"],  # :545
}
# sha256 of json.dumps(SOL_CONVENTION_AUTHORIZED, sort_keys=True) at an earlier revision, for the record: 84795c4d78f29034bb575012ceef959a515fd688befc3eb7aefe85ee1a45b188
SOL_CONVENTION_KEYS = {"id": str, "headline": str, "statement": str, "limits": list, "not_claimed": list}        # perf_model.py:531-546 (key order)
TENSOR_CORE_TERM_KEYS = {"convention": str, "applied": bool, "reason": str, "macs_priced": int, "default_rate_macs_priced": int,
                         "quantized_dtype_macs_priced": int, "contraction_macs_priced": type(None), "non_contraction_other_macs_priced": type(None),
                         "scalar_statement_multiplies_excluded": int, "excluded_cycles_at_applied_rate": float, "rate_key": str, "rate_class": str, "note": str}  # :1384-1404
# perf_model.py:566-567 -- the reason recorded when total.macs_typed is absent (this fixture)
TC_TERM_REASON_ABSENT = ("total.macs_typed is absent (the analysis predates typed MAC components): the default-rate remainder is priced whole, "
                         "as before convention B; the scalar share is not separable here")
# perf_model.py:1399-1403 -- the note when convention B is not applied (no "other non-contraction MACs" clause)
TC_TERM_NOTE_NOT_APPLIED = ("compute_tc_cycles = default_rate_macs_priced / MAC_per_cycle + sum(quantised bucket / its rate); priced: contraction MACs; "
                            "excluded: the scalar statement multiplies (their typed count is retained in workload.macs_typed) and other_ops; the SM "
                            "diagnostic compute_sm_cycles prices other_ops only and is unchanged; the headline is an optimistic contraction-and-memory "
                            "bound, not a bound over all compute resources")
REMAINDER_ADDED_KEY = "macs_priced_in_tensor_core_term"                           # perf_model.py:626
NOTE_PATH = "workload.default_rate_remainder.note"
# perf_model.py:627-630 -- the one sentence convention B appends to the pre-existing default-rate-remainder note when total.macs_typed is absent
NOTE_SUFFIX = (" and nothing is excluded from the tensor-core timing term (issue #58 convention B needs the typed components); this is a bound "
               "convention, not a claim that any work is hidden or free")
# numeric tokens the authorized additions put into the YAML text (and nowhere else): 58 x3 (sol_convention.id, tensor_core_term.convention,
# "#58" in the note suffix); 32 x2 ("fp32" in tensor_core_term.rate_key and in sol_convention.statement); 160 x3 (macs_priced,
# default_rate_macs_priced, macs_priced_in_tensor_core_term == total_macs); 0 x2 (quantized_dtype_macs_priced, scalar_statement_multiplies_excluded);
# 0.0 x1 (excluded_cycles_at_applied_rate)
CONVENTION_B_TOKENS = {"58": 3, "32": 2, "160": 3, "0": 2, "0.0": 1}


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def _predict(tmp_path: Path, mode: str):
    model = EinsumGraphPerfModel(debug=False, dtype_bytes=(mode == "dtype"))
    out = tmp_path / f"perf_{mode}"
    p = model.predict(str(ANALYSIS), str(out), arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert p is not None, "EinsumGraphPerfModel.predict failed on the issue-62 fixture"
    yml = out / "perf_H100_PCIe.yaml"
    assert yml.is_file()
    return p, yml


def _strip_metadata_text(text: str):
    """Remove the lines of the two metadata keys (written first inside the memory_breakdown mapping)."""
    lines = text.splitlines(keepends=True)
    i = lines.index("memory_breakdown:\n")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("  weight_elements:"))
    return "".join(lines[: i + 1] + lines[j:]), "".join(lines[i + 1 : j])


def _without_metadata(perf: dict) -> dict:
    q = copy.deepcopy(perf)
    for k in META_KEYS:
        q["memory_breakdown"].pop(k, None)
    return q


def _block_end(lines, start, indent):
    """Index of the first line after `start` whose indentation is <= `indent` (a sibling or parent key); list items / folded
    continuations are deeper and stay inside the block."""
    for k in range(start + 1, len(lines)):
        if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip(" "))) <= indent:
            return k
    return len(lines)


def _cut(lines, header, indent):
    i = lines.index(header); j = _block_end(lines, i, indent)
    return lines[:i] + lines[j:], lines[i:j]


def _strip_convention_b_text(text: str):
    """Remove the three convention-B additions from the YAML text: the top-level `sol_convention` mapping, the `tensor_core_term`
    mapping inside `workload` and the `macs_priced_in_tensor_core_term` line inside `default_rate_remainder`.  Returns the stripped
    text and the removed text (kept for the exactness assertions)."""
    lines = text.splitlines(keepends=True); removed = []
    lines, r = _cut(lines, "sol_convention:\n", 0); removed += r
    lines, r = _cut(lines, "  tensor_core_term:\n", 2); removed += r
    i = next(k for k, l in enumerate(lines) if l.startswith(f"    {REMAINDER_ADDED_KEY}:")); removed.append(lines.pop(i))
    return "".join(lines), "".join(removed)


def _strip_note_text(text: str):
    """Remove the `note` scalar of `default_rate_remainder` (convention B appends a sentence to it, which re-flows and re-quotes the
    folded scalar, so its bytes are compared at the dict level: golden note + NOTE_SUFFIX).  Returns (stripped, removed)."""
    lines = text.splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if l.startswith("    note:")); j = _block_end(lines, i, 4)
    return "".join(lines[:i] + lines[j:]), "".join(lines[i:j])


def _without_convention_b(perf: dict) -> dict:
    q = copy.deepcopy(perf)
    q.pop("sol_convention", None); q["workload"].pop("tensor_core_term", None); q["workload"]["default_rate_remainder"].pop(REMAINDER_ADDED_KEY, None)
    return q


def _walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, d


def _numeric_tokens(text: str):
    return _NUM.findall(text)


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    base = tmp_path_factory.mktemp("issue62")
    return {mode: _predict(base, mode) for mode in MODES}


# ---------------------------------------------------------------------------------------------------------------------
# 1. golden byte identity against the unpatched baseline
# ---------------------------------------------------------------------------------------------------------------------
class TestGoldenByteIdentity:
    def test_fixture_files_are_the_recorded_ones(self):
        import hashlib
        rec = json.loads((FIX / "FIXTURE_RECORD.json").read_text())["files"]
        for name, meta in rec.items():
            assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == meta["sha256"], name

    @pytest.mark.parametrize("mode", MODES)
    def test_report_minus_metadata_is_byte_identical_to_the_baseline_report(self, reports, mode):
        """Report minus the issue-62 metadata minus the convention-B additions == the pre-convention golden, byte for byte, except
        the one convention-B-altered leaf (the remainder note, compared exactly at the dict level below)."""
        _, yml = reports[mode]
        new_text = yml.read_text()
        stripped, removed = _strip_metadata_text(new_text)
        stripped, removed_b = _strip_convention_b_text(stripped)
        golden_text = GOLDEN[mode].read_text()
        stripped_wo_note, new_note = _strip_note_text(stripped)
        golden_wo_note, golden_note = _strip_note_text(golden_text)
        assert stripped_wo_note == golden_wo_note, "a pre-existing byte of the report changed"
        meta = yaml.safe_load("memory_breakdown:\n" + removed)["memory_breakdown"]
        assert set(meta) == set(META_KEYS), "only the two metadata keys were added"
        # the removed convention-B text is exactly the authorized structure (parsed from the removed lines themselves)
        conv = yaml.safe_load(removed_b.replace("  tensor_core_term:", "tensor_core_term:").replace(f"    {REMAINDER_ADDED_KEY}:", f"{REMAINDER_ADDED_KEY}:"))
        assert set(conv) == {"sol_convention", "tensor_core_term", REMAINDER_ADDED_KEY}
        golden = yaml.safe_load(golden_text)
        assert conv["sol_convention"] == SOL_CONVENTION_AUTHORIZED and list(conv["sol_convention"]) == list(SOL_CONVENTION_KEYS)   # the saved YAML carries exactly the pinned literals
        assert set(conv["tensor_core_term"]) == set(TENSOR_CORE_TERM_KEYS) and conv[REMAINDER_ADDED_KEY] == golden["workload"]["total_macs"]
        # the only altered pre-existing leaf: the folded note re-flows, so compare its parsed text exactly
        n_new = yaml.safe_load("x:\n" + new_note.replace("    note:", "  note:", 1))["x"]["note"]
        n_old = yaml.safe_load("x:\n" + golden_note.replace("    note:", "  note:", 1))["x"]["note"]
        assert n_new == n_old + NOTE_SUFFIX, "the remainder note changed other than by the authorized convention-B suffix"

    @pytest.mark.parametrize("mode", MODES)
    def test_every_preexisting_field_is_equal_and_type_exact(self, reports, mode):
        p, _ = reports[mode]
        golden = yaml.safe_load(GOLDEN[mode].read_text())
        new = dict(_walk(_without_convention_b(_without_metadata(p))))
        old = dict(_walk(golden))
        assert new.keys() == old.keys()
        for path, v in old.items():
            if path == NOTE_PATH:
                assert new[path] == v + NOTE_SUFFIX and type(new[path]) is str, path
                continue
            assert new[path] == v and type(new[path]) is type(v), path
        assert set(p) == set(golden) | {"sol_convention"}
        assert set(p["workload"]) == set(golden["workload"]) | {"tensor_core_term"}
        assert set(p["workload"]["default_rate_remainder"]) == set(golden["workload"]["default_rate_remainder"]) | {REMAINDER_ADDED_KEY}
        assert set(p["memory_breakdown"]) == set(golden["memory_breakdown"]) | set(META_KEYS)

    @pytest.mark.parametrize("mode", MODES)
    def test_no_numeric_token_is_added_anywhere_in_the_yaml_text(self, reports, mode):
        """Outside the authorized convention-B additions the numeric tokens are the golden's, in order; the additions contribute
        exactly CONVENTION_B_TOKENS (each one accounted for in the constant's comment) and nothing else."""
        from collections import Counter
        _, yml = reports[mode]
        new_text = yml.read_text(); golden_text = GOLDEN[mode].read_text()
        stripped, removed_b = _strip_convention_b_text(new_text)
        stripped_wo_note, new_note = _strip_note_text(stripped)
        golden_wo_note, golden_note = _strip_note_text(golden_text)
        assert _numeric_tokens(stripped_wo_note) == _numeric_tokens(golden_wo_note)
        # the note keeps the golden's tokens in order and gains only the "#58" of the suffix
        assert _numeric_tokens(new_note) == _numeric_tokens(golden_note) + ["58"]
        assert Counter(_numeric_tokens(removed_b)) + Counter(["58"]) == Counter(CONVENTION_B_TOKENS)

    @pytest.mark.parametrize("mode", MODES)
    def test_convention_b_additions_are_exactly_the_authorized_ones(self, reports, mode):
        """The full leaf delta between the report (minus issue-62 metadata) and the pre-convention golden is EXACTLY the authorized
        convention-B contract: no leaf removed, one leaf altered (the note suffix), the added leaves with these keys, types and values."""
        p, _ = reports[mode]
        golden = yaml.safe_load(GOLDEN[mode].read_text())
        new = dict(_walk(_without_metadata(p))); old = dict(_walk(golden))
        assert set(old) - set(new) == set(), "a pre-existing leaf disappeared"
        assert {k for k in set(new) & set(old) if new[k] != old[k] or type(new[k]) is not type(old[k])} == {NOTE_PATH}
        added = set(new) - set(old)
        assert added == {f"sol_convention.{k}" for k in SOL_CONVENTION_KEYS} | {f"workload.tensor_core_term.{k}" for k in TENSOR_CORE_TERM_KEYS} | {f"workload.default_rate_remainder.{REMAINDER_ADDED_KEY}"}
        conv, term = p["sol_convention"], p["workload"]["tensor_core_term"]
        for k, t in SOL_CONVENTION_KEYS.items():
            assert type(conv[k]) is t, k
        for k, t in TENSOR_CORE_TERM_KEYS.items():
            assert type(term[k]) is t, k
        total = golden["workload"]["total_macs"]; remainder = p["workload"]["default_rate_remainder"]
        # POSITIVE checks against the pinned literals (copied from perf_model.py at an earlier revision, lines cited at the constants); the live
        # producer constant is deliberately NOT imported here (the review P2 live-convention-oracle-1)
        assert conv == SOL_CONVENTION_AUTHORIZED, {k: (conv.get(k), SOL_CONVENTION_AUTHORIZED[k]) for k in SOL_CONVENTION_AUTHORIZED if conv.get(k) != SOL_CONVENTION_AUTHORIZED[k]}
        assert conv["statement"] == SOL_CONVENTION_AUTHORIZED["statement"] and conv["limits"] == SOL_CONVENTION_AUTHORIZED["limits"] and conv["not_claimed"] == SOL_CONVENTION_AUTHORIZED["not_claimed"]
        assert conv["id"] == CONVENTION_B_ID and conv["headline"] == CONVENTION_B_HEADLINE
        assert list(conv) == list(SOL_CONVENTION_KEYS) and len(conv["limits"]) == 3 and len(conv["not_claimed"]) == 4
        assert all(isinstance(x, str) and x for x in conv["limits"] + conv["not_claimed"])
        assert term["convention"] == CONVENTION_B_ID and term["applied"] is False                     # :1385-1386; no total.macs_typed in the fixture
        assert term["reason"] == TC_TERM_REASON_ABSENT and term["note"] == TC_TERM_NOTE_NOT_APPLIED      # :566-567, :1399-1403
        assert term["scalar_statement_multiplies_excluded"] == 0                                        # :1003 -> 0 when macs_typed is absent
        assert term["default_rate_macs_priced"] == remainder["macs"] - term["scalar_statement_multiplies_excluded"] == total     # :1011, :1389
        assert term["quantized_dtype_macs_priced"] == 0                                                  # no quantised compute dtypes in the fixture
        assert term["macs_priced"] == term["default_rate_macs_priced"] + term["quantized_dtype_macs_priced"] == total            # :1388
        assert remainder[REMAINDER_ADDED_KEY] == remainder["macs"] == total                              # :626 (== macs_default)
        rate = golden["arch"]["MAC_per_cycle"]
        assert term["excluded_cycles_at_applied_rate"] == term["scalar_statement_multiplies_excluded"] / rate == 0.0             # :1166
        assert term["contraction_macs_priced"] is None and term["non_contraction_other_macs_priced"] is None                      # :1394-1395 (not applied)
        assert term["rate_key"] == golden["arch"]["mac_per_cycle_key_applied"] and term["rate_class"] == golden["arch"]["mac_rate_class_applied"]  # :1398
        assert remainder["note"] == golden["workload"]["default_rate_remainder"]["note"] + NOTE_SUFFIX                           # :627-630
        # the headline accounting is untouched: every cycle / byte / runtime leaf equals the golden (already covered leaf-by-leaf above)
        for sect in ("unfused", "fused", "fused_prefetched"):
            for k in ("compute_tc_cycles", "compute_sm_cycles", "compute_cycles", "memory_cycles", "total_cycles", "memory_bytes", "runtime_ms"):
                assert p[sect][k] == golden[sect][k] and type(p[sect][k]) is type(golden[sect][k]), f"{sect}.{k}"

    @pytest.mark.parametrize("mode", MODES)
    def test_metadata_keys_come_first_and_numeric_keys_keep_their_order(self, reports, mode):
        p, _ = reports[mode]
        assert list(p["memory_breakdown"]) == list(META_KEYS) + list(NUMERIC_KEYS)
        for k in NUMERIC_KEYS:
            assert type(p["memory_breakdown"][k]) is int

    def test_the_fixture_shows_why_the_categories_are_not_additive(self, reports):
        """Uniform: category sum == unfused F (as for Lite 170).  Dtype: sum != unfused F by the repeated external read;
        model_io_bytes == fused F by assignment; weight_bytes is the folded sentinel 0 with a non-zero footprint."""
        u, _ = reports["uniform"]; d, _ = reports["dtype"]
        mu, md = u["memory_breakdown"], d["memory_breakdown"]
        assert (mu["weight_bytes"], mu["model_io_bytes"], mu["intermediate_bytes"]) == (80, 416, 320)
        assert mu["weight_bytes"] + mu["model_io_bytes"] + mu["intermediate_bytes"] == 816 == u["unfused"]["memory_bytes"]
        assert (md["weight_bytes"], md["model_io_bytes"], md["intermediate_bytes"]) == (0, 264, 160)
        assert md["weight_bytes"] + md["model_io_bytes"] + md["intermediate_bytes"] == 424 != d["unfused"]["memory_bytes"] == 488
        assert d["unfused"]["memory_bytes"] - 424 == 64 == 32 * 2          # x (32 bf16 elements) read by both layers
        assert md["model_io_bytes"] == d["fused"]["memory_bytes"] == d["fused_prefetched"]["memory_bytes"] == 264
        assert md["weight_bytes"] == 0 and md["weight_elements"] == mu["weight_elements"] == 20
        for k in ("weight_elements", "model_io_elements", "intermediate_elements"):
            assert md[k] == mu[k]                                           # element counters are raw in BOTH modes


# ---------------------------------------------------------------------------------------------------------------------
# 2. metadata content (follows round 8's LABEL_DISPOSITION)
# ---------------------------------------------------------------------------------------------------------------------
class TestMetadataContent:
    @pytest.mark.parametrize("mode", MODES)
    def test_flag_is_a_real_bool_true(self, reports, mode):
        p, _ = reports[mode]
        assert p["memory_breakdown"]["auxiliary_non_additive"] is True

    @pytest.mark.parametrize("mode", MODES)
    def test_semantics_has_one_text_per_numeric_label_plus_scope_and_traffic_fields(self, reports, mode):
        s = reports[mode][0]["memory_breakdown"]["semantics"]
        assert tuple(s) == SEMANTICS_KEYS
        assert all(isinstance(v, str) and v.strip() and not _NUM.search(v) for v in s.values()), "texts are non-empty and carry no digits"

    def test_dtype_mode_texts_name_the_folded_sentinel_and_the_deduplicated_boundary(self, reports):
        s = reports["dtype"][0]["memory_breakdown"]["semantics"]
        assert s["weight_bytes"].startswith("Sentinel zero: weights are folded into external-input bytes in dtype mode")
        assert "not zero parameter storage or a zero-width dtype" in s["weight_bytes"]
        assert s["model_io_bytes"].startswith("Deduplicated external-input plus external-output bytes, including weights")
        assert "different counting basis from model_io_elements" in s["model_io_bytes"]

    def test_uniform_mode_texts_name_the_per_op_raw_basis(self, reports):
        s = reports["uniform"][0]["memory_breakdown"]["semantics"]
        assert s["weight_bytes"].startswith("Parameter-footprint estimate at reporting width")
        assert "not weight-read traffic" in s["weight_bytes"]
        assert s["model_io_bytes"].startswith("Per-op model-I/O counter at reporting width")
        assert "shared reads may repeat" in s["model_io_bytes"]

    def test_defect_labels_are_mode_aware_and_diagnostic_labels_are_not(self, reports):
        su = reports["uniform"][0]["memory_breakdown"]["semantics"]; sd = reports["dtype"][0]["memory_breakdown"]["semantics"]
        for label, cls in DISPOSITION.items():
            if cls == "defect":
                assert su[label] != sd[label], label                 # basis differs by mode: the text says which
            else:
                assert su[label] == sd[label], label                 # raw per-op diagnostic in both modes
        for k in ("scope", "traffic_fields"):
            assert su[k] == sd[k]

    @pytest.mark.parametrize("mode", MODES)
    def test_diagnostic_overlapping_texts_deny_the_additive_or_traffic_reading(self, reports, mode):
        s = reports[mode][0]["memory_breakdown"]["semantics"]
        assert "not globally unique storage or weight-read traffic" in s["weight_elements"]
        assert "not re-deduplicated" in s["model_io_elements"]
        assert "not a resident allocation footprint" in s["intermediate_elements"]
        assert "not an additive complement of deduplicated external traffic" in s["intermediate_bytes"]
        assert s["scope"].startswith("Mixed diagnostic bases; not a disjoint physical-memory breakdown")

    @pytest.mark.parametrize("mode", MODES)
    def test_traffic_fields_text_points_at_the_three_F_fields(self, reports, mode):
        p, _ = reports[mode]
        for f in F_FIELDS:
            assert f in p["memory_breakdown"]["semantics"]["traffic_fields"]
            sec, key = f.split("."); assert type(p[sec][key]) is int   # the F fields exist, are ints and (see golden test) unchanged


# ---------------------------------------------------------------------------------------------------------------------
# 3. YAML / JSON round trip of the new metadata
# ---------------------------------------------------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize("mode", MODES)
    def test_on_disk_yaml_loads_equal_to_the_returned_dict(self, reports, mode):
        p, yml = reports[mode]
        loaded = yaml.safe_load(yml.read_text())
        assert loaded == p and loaded["memory_breakdown"]["auxiliary_non_additive"] is True

    @pytest.mark.parametrize("mode", MODES)
    def test_yaml_round_trip_with_the_writers_dumper(self, reports, mode):
        mb = reports[mode][0]["memory_breakdown"]
        text = yaml.dump(mb, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
        back = yaml.safe_load(text)
        assert back == mb and back["auxiliary_non_additive"] is True and back["semantics"] == mb["semantics"]
        assert list(back) == list(mb)

    @pytest.mark.parametrize("mode", MODES)
    def test_json_round_trip(self, reports, mode):
        p, _ = reports[mode]
        back = json.loads(json.dumps(p))
        assert back == p and back["memory_breakdown"]["auxiliary_non_additive"] is True
        assert json.loads(json.dumps(yaml.safe_load(yaml.safe_dump(p["memory_breakdown"])))) == p["memory_breakdown"]


# ---------------------------------------------------------------------------------------------------------------------
# 4. consumer compatibility: every in-repo reader of memory_breakdown produces identical output with the metadata present
# ---------------------------------------------------------------------------------------------------------------------
def _perf_tree(root: Path, layout, perf_file: Path, arch="H100_PCIe"):
    """Lay the given perf report out as the consumer expects it; returns the root."""
    dst = root.joinpath(*layout) / f"perf_{arch}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(perf_file.read_text())
    return root


class TestConsumerCompatibility:
    @_needs("scripts/collect_sol_b200.py")
    @pytest.mark.parametrize("mode", MODES)
    def test_scripts_collect_sol_b200(self, reports, tmp_path, mode):
        """solar/scripts/collect_sol_b200.py: mb["weight_bytes"] / mb["model_io_bytes"] / mb["intermediate_bytes"]."""
        _, new_yml = reports[mode]
        csvs = {}
        for tag, src in (("golden", GOLDEN[mode]), ("new", new_yml)):
            root = _perf_tree(tmp_path / tag, ("L1", "p001", "manual", "perf_full"), src)
            out = tmp_path / f"{tag}.csv"
            env = dict(os.environ, SOL_RESULTS_DIR=str(tmp_path / f"results_{tag}"))
            r = subprocess.run([sys.executable, str(SOLAR_ROOT / "scripts" / "collect_sol_b200.py"), "--output-root", str(root), "--arch", "H100_PCIe", "--out", str(out)],
                               cwd=str(SOLAR_ROOT), env=env, capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stdout + r.stderr
            csvs[tag] = out.read_bytes()
        assert csvs["new"] == csvs["golden"]
        rows = list(csv.DictReader((tmp_path / "new.csv").open()))
        assert len(rows) == 1 and int(rows[0]["weight_bytes"]) == reports[mode][0]["memory_breakdown"]["weight_bytes"]

    @_needs("solar/benchmark/solbenchv3/collect_perf.py", "solar/benchmark/solbenchv4/collect_perf.py")
    @pytest.mark.parametrize("module,kwargs", [
        ("solar.benchmark.solbenchv3.collect_perf", {}),
        ("solar.benchmark.solbenchv3.collect_perf_quant", {}),
        ("solar.benchmark.solbenchv4.collect_perf", {}),
        ("solar.benchmark.solbenchv4.collect_perf_l1", {}),
        ("solar.benchmark.solbenchv4.collect_perf_quant", {}),
    ])
    @pytest.mark.parametrize("mode", MODES)
    def test_benchmark_collectors(self, reports, tmp_path, module, kwargs, mode):
        """solbenchv3/v4 collect_perf*.py: mb.get(<named key>, "")."""
        _, new_yml = reports[mode]
        mod = importlib.import_module(module)
        outs = {}
        for tag, src in (("golden", GOLDEN[mode]), ("new", new_yml)):
            root = _perf_tree(tmp_path / tag, ("kernel_a", "0", "perf"), src)
            out_csv = tmp_path / f"{tag}.csv"
            rows = mod.collect(output_dir=str(root), out_csv=str(out_csv), arch="H100_PCIe", **kwargs)
            outs[tag] = (rows, out_csv.read_bytes() if out_csv.exists() else None)
        assert outs["new"] == outs["golden"]
        rows, _ = outs["new"]
        assert len(rows) == 1 and str(rows[0]["weight_bytes"]) == str(reports[mode][0]["memory_breakdown"]["weight_bytes"])

    @_needs("scripts/collect_perf_results.py")
    @pytest.mark.parametrize("mode", MODES)
    def test_scripts_collect_perf_results(self, reports, tmp_path, mode):
        """solar/scripts/collect_perf_results.py: reads unfused/fused/fused_prefetched/workload/speedup; not the breakdown."""
        spec = importlib.util.spec_from_file_location("collect_perf_results_issue62", SOLAR_ROOT / "scripts" / "collect_perf_results.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        res = {}
        for tag, src in (("golden", GOLDEN[mode]), ("new", reports[mode][1])):
            root = _perf_tree(tmp_path / tag, ("level1", "1_kernel", "perf"), src)
            res[tag] = mod.collect_results(root, "H100_PCIe")
        assert res["new"] == res["golden"] and len(res["new"]) == 1

    @_needs("examples/compare_examples.py")
    @pytest.mark.parametrize("mode", MODES)
    def test_examples_compare_examples_extract_perf_stats(self, reports, mode):
        sys.path.insert(0, str(SOLAR_ROOT / "examples"))
        try:
            from compare_examples import extract_perf_stats
        finally:
            sys.path.pop(0)
        assert extract_perf_stats(reports[mode][1]) == extract_perf_stats(GOLDEN[mode]) is not None

    @pytest.mark.parametrize("mode", MODES)
    def test_execbench_runner_summary_fields_and_facade_read(self, reports, mode):
        """scripts/run_execbench_problem.py (sol_summary.json fields) and solar/translate/__init__.py (workload.unresolved_ops)."""
        new = reports[mode][0]; old = yaml.safe_load(GOLDEN[mode].read_text())
        def summary(perf):
            return {"perf_mac_key": perf.get("arch", {}).get("mac_per_cycle_key"), "perf_bytes_per_element": perf.get("workload", {}).get("bytes_per_element"),
                    "total_macs": perf.get("workload", {}).get("total_macs"), "total_flops": perf.get("workload", {}).get("total_flops"),
                    "sol_ms": {m: perf.get(m, {}).get("runtime_ms") for m in ("unfused", "fused", "fused_prefetched")},
                    "bottleneck": {m: perf.get(m, {}).get("bottleneck") for m in ("unfused", "fused", "fused_prefetched")},
                    "memory_bytes": {m: perf.get(m, {}).get("memory_bytes") for m in ("unfused", "fused", "fused_prefetched")}}
        assert summary(new) == summary(old)
        assert (new.get("workload") or {}).get("unresolved_ops") == (old.get("workload") or {}).get("unresolved_ops") is None

    @pytest.mark.parametrize("mode", MODES)
    def test_flashinfer_parser_and_legacy_compare_baseline_reads(self, reports, mode):
        new = yaml.safe_load(reports[mode][1].read_text()); old = yaml.safe_load(GOLDEN[mode].read_text())
        for perf in (new, old):
            assert (perf.get("unfused", {}).get("runtime_ms"), perf.get("fused", {}).get("runtime_ms"), perf.get("fused_prefetched", {}).get("runtime_ms")) == \
                   (old["unfused"]["runtime_ms"], old["fused"]["runtime_ms"], old["fused_prefetched"]["runtime_ms"])
            assert perf.get("performance_projection", {}) == {} and perf.get("cycles", 0) == 0     # legacy/compare_baseline.py keys

    @pytest.mark.parametrize("mode", MODES)
    def test_shell_awk_and_inline_python_readers(self, reports, mode):
        """examples*/run_all.sh and bench_gpu_only.sh: awk on the unfused:/fused: runtime_ms lines; d['fused']['runtime_ms']."""
        for section in ("unfused", "fused"):
            prog = "/^%s:/{found=1} found && /runtime_ms:/{print $2; exit}" % section
            got = [subprocess.run(["awk", prog, str(f)], capture_output=True, text=True, check=True).stdout for f in (reports[mode][1], GOLDEN[mode])]
            assert got[0] == got[1] and got[0].strip()
        new = yaml.safe_load(reports[mode][1].read_text()); old = yaml.safe_load(GOLDEN[mode].read_text())
        assert new["fused"]["runtime_ms"] == old["fused"]["runtime_ms"] and new["fused"].get("runtime_ms", 0) == old["fused"].get("runtime_ms", 0)

    @pytest.mark.parametrize("mode", MODES)
    def test_in_repo_test_access_patterns(self, reports, mode):
        """test_campaign_repairs / test_weight_elements / test_scaled_mm_accounting: named keys and ONE-key membership."""
        mb = reports[mode][0]["memory_breakdown"]
        assert mb["weight_elements"] == 20 and mb["model_io_elements"] == 104 and mb["intermediate_elements"] == 80
        assert mb["weight_elements"] + mb["model_io_elements"] + mb["intermediate_elements"] == yaml.safe_load(ANALYSIS.read_text())["total"]["unfused_elements"]
        assert ("weight_elements_unresolved" in mb) is False and "weight_bytes_unresolved" not in mb

    def test_mr112_saved_report_fixture_is_untouched_and_still_reads(self):
        fx = SOLAR_ROOT / "tests" / "fixtures" / "mr112" / "historical-unresolved-dsl.json"
        d = json.loads(fx.read_text())
        assert "raw" in d
        saved = json.loads(fx.read_text())
        found = [v for k, v in _walk(saved) if k.endswith("memory_breakdown.weight_bytes")]
        assert found == [0]      # the saved report carries no metadata; nothing rewrites saved files


# ---------------------------------------------------------------------------------------------------------------------
# 5. the unresolved-weight keys (present only when the analysis carries weight_elements_unresolved) still follow the metadata
# ---------------------------------------------------------------------------------------------------------------------
def test_unresolved_weight_fields_keep_their_place_after_the_metadata(tmp_path):
    a = yaml.safe_load(ANALYSIS.read_text())
    a["total"]["weight_elements_unresolved"] = 7
    ap = tmp_path / "analysis.yaml"; ap.write_text(yaml.safe_dump(a, sort_keys=False))
    p = EinsumGraphPerfModel().predict(str(ap), str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    mb = p["memory_breakdown"]
    assert list(mb) == list(META_KEYS) + ["weight_elements", "weight_bytes", "weight_elements_unresolved", "weight_bytes_unresolved",
                                           "model_io_elements", "model_io_bytes", "intermediate_elements", "intermediate_bytes"]
    assert mb["weight_elements_unresolved"] == 7 and mb["weight_bytes_unresolved"] == 28 and mb["auxiliary_non_additive"] is True
