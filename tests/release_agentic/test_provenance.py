# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source / shape / dtype provenance across the two routes.

What a verification run knew (validation shapes, precision, reference
bytes) versus what the receipt and downstream SOL artifacts claim.

ADJUDICATION (release adjudication of the agentic contract, archive of the verbatim
audit file: an earlier revision).  Tests that pinned a DEFECT's old behavior are
rewritten to pin the repaired contract; strict xfails whose contract is now
delivered have the marker removed.  Policy-gated findings keep their
original xfail/pinning tests (A3 exit code, A10 tolerance): the legacy
default is deliberately preserved and only exposed/gated (see
test_repairs.py).
"""
from __future__ import annotations

import json
import sys

import pytest

from conftest import CUDA, SOLAR, lib_verify, run

DSL = CUDA / "solar_model_einsum_lib.py"
ADAPTER = CUDA / "adapter.py"
DEFN = CUDA / "definition.json"


@pytest.fixture(scope="module")
def verified(torch):
    res = lib_verify(ADAPTER, DSL, source_type="cuda")
    assert res.verified, res.message
    return res


def test_evidence_records_actual_validation_shapes_and_reference_hash(verified):
    ev = verified.execution
    shapes = dict(ev.validation_shapes)
    assert shapes == {"x": (4, 16), "w": (16,)}          # capped B=4, not 32
    assert ev.source_type == "cuda"
    assert len(ev.reference_sha256) == 64 and len(ev.dsl_sha256) == 64
    assert ev.unsupported_dependencies == ()
    assert ev.checker_identity == "solar.einsum_lib.verify.compare_outputs"


# ---------------------------------------------------------------------------
# A6a: receipt validation_axes defaults to the ORIGINAL workload axes
# ---------------------------------------------------------------------------

def _emit(verified, tmp_path, **kw):
    from solar.translate.receipts import emit_receipt, read_receipt
    out = emit_receipt(verified, source_path=ADAPTER, dsl_path=DSL,
                       definition_path=DEFN,
                       workload_uuid="release-agentic-scale-rows-b32",
                       axes={"B": 32, "H": 16}, dtype="fp16",
                       out_path=tmp_path / "receipt.json", **kw)
    return read_receipt(out)


@pytest.mark.finding("A6")
def test_A6_fixed_receipt_records_validation_at_executed_axes(verified, tmp_path):
    """Repaired: validation_axes omitted -> derived from the executed shapes
    (was: defaulted to the ORIGINAL workload axes)."""
    r = _emit(verified, tmp_path)                         # validation_axes omitted
    assert r["workload"]["axes"] == {"B": 32, "H": 16}
    assert r["workload"]["validation_axes"] == {"B": 4, "H": 16}    # <- fact
    assert r["workload"]["validation_axes_source"] == "executed_shapes"
    assert r["workload"]["capped_validation"] is True
    assert r["execution"]["validation_shapes"]["x"] == [4, 16]


@pytest.mark.finding("A6")
def test_A6_contract_validation_axes_agree_with_executed_shapes(verified, tmp_path):
    r = _emit(verified, tmp_path)
    assert r["workload"]["validation_axes"]["B"] == 4


@pytest.mark.finding("A6")
def test_A6_strict_entry_admits_the_consistent_receipt(verified, tmp_path):
    """Positive control: the derived (consistent) receipt passes the strict
    cross-check; the forged contradiction is refused in test_repairs.py."""
    from solar.translate.receipts import check_strict
    _emit(verified, tmp_path)
    r = check_strict(tmp_path / "receipt.json", source_path=ADAPTER, dsl_path=DSL,
                     definition_path=DEFN,
                     workload_uuid="release-agentic-scale-rows-b32",
                     axes={"B": 32, "H": 16}, dtype="fp16")
    assert r["verdicts"]["numerical"] == "pass"


def test_explicit_validation_axes_are_recorded_when_given(verified, tmp_path):
    r = _emit(verified, tmp_path, validation_axes={"B": 4, "H": 16})
    assert r["workload"]["validation_axes"] == {"B": 4, "H": 16}


# ---------------------------------------------------------------------------
# A6b: dtype provenance — verification_dtype is the IR flag; per-tensor
# source dtypes and reference output dtypes are not in the evidence
# ---------------------------------------------------------------------------

@pytest.mark.finding("A6")
def test_A6_fixed_receipt_binds_measured_dtypes_beside_the_flag(verified, tmp_path):
    """Repaired: the flag fields stay what they were, and the measured
    per-tensor dtypes are recorded beside them (were absent)."""
    r = _emit(verified, tmp_path)
    assert r["execution"]["verification_dtype"] == "fp64"   # use_fp64 flag
    assert r["dtype"] == "fp16"                              # caller's claim
    assert r["execution"]["input_dtypes"] == {"x": "fp32", "w": "fp32"}
    assert r["execution"]["reference_input_dtypes"] == {"x": "fp64", "w": "fp64"}
    assert r["execution"]["reference_output_dtypes"] == {"output": "fp64"}


@pytest.mark.finding("A6")
def test_A6_contract_receipt_binds_per_tensor_dtypes(verified, tmp_path):
    r = _emit(verified, tmp_path)
    assert r["execution"]["input_dtypes"] == {"x": "fp32", "w": "fp32"}


def test_pytorch_reference_runs_native_precision_while_flag_says_fp64(torch):
    """Route asymmetry: the cuda adapter receives fp64-cast inputs, the
    pytorch golden runs at native (fp32) precision.  Both record
    verification_dtype='fp64'.  Pinned from the pipeline docstring."""
    from conftest import PT
    from solar.einsum_lib.pipeline import prepare_pytorch
    p = prepare_pytorch(str(PT / "model.py"), True, use_fp64=True)
    ref = p[3]
    assert ref["output"].dtype.name == "float32"
    res = lib_verify(PT / "model.py", PT / "model_einsum_lib.py",
                     source_type="pytorch")
    assert res.execution.verification_dtype == "fp64"


# ---------------------------------------------------------------------------
# Strict receipt gate: library path works; shape mismatch and dtype mismatch
# refuse; the pure-skill route has no way to produce a receipt (A5)
# ---------------------------------------------------------------------------

def test_strict_gate_refuses_other_workload_axes_and_other_dtype(verified, tmp_path):
    from solar.translate.receipts import ReceiptRefusal, check_strict
    _emit(verified, tmp_path)
    common = dict(source_path=ADAPTER, dsl_path=DSL, definition_path=DEFN,
                  workload_uuid="release-agentic-scale-rows-b32")
    with pytest.raises(ReceiptRefusal, match="axes differ"):
        check_strict(tmp_path / "receipt.json", axes={"B": 64, "H": 16},
                     dtype="fp16", **common)
    with pytest.raises(ReceiptRefusal, match="dtype"):
        check_strict(tmp_path / "receipt.json", axes={"B": 32, "H": 16},
                     dtype="fp8", **common)


@pytest.mark.finding("A5")
def test_A5_fixed_verify_writes_no_receipt_unless_asked(torch, tmp_path):
    """Repaired: both CLIs can emit a receipt (--emit-receipt, see
    test_repairs.py); without the flag they still write nothing into the
    fixture directory and mention no receipt."""
    from conftest import cli_verify, script_verify
    before = {p.name for p in CUDA.iterdir()}
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet")
    c = cli_verify(ADAPTER, DSL, "--source-type", "cuda")
    assert s.returncode == 0 and c.returncode == 0
    assert {p.name for p in CUDA.iterdir()} == before
    assert "receipt" not in (s.stdout + c.stdout).lower()
    for cmd in ([sys.executable, "-m", "solar.cli.translate", "verify", "--help"],
                [sys.executable, SOLAR / "scripts" / "verify_saved_ir.py", "--help"]):
        h = run(cmd)
        assert "--emit-receipt" in h.stdout and "--definition" in h.stdout
        assert "--dtype" in h.stdout and "--workload-idx" in h.stdout


@pytest.mark.finding("A5")
def test_A5_contract_cli_verify_can_emit_receipt():
    h = run([sys.executable, "-m", "solar.cli.translate", "verify", "--help"])
    assert "receipt" in h.stdout.lower()


# ---------------------------------------------------------------------------
# SOL artifacts: legacy sol() carries no verification provenance; strict
# sol() on the same fixture passes the gate and binds the dtype
# ---------------------------------------------------------------------------

def _keys(doc) -> set:
    """Every mapping key at any depth of a YAML/JSON document: the provenance
    check is structural, not a word search over descriptive prose."""
    found = set()
    if isinstance(doc, dict):
        for k, v in doc.items():
            found.add(str(k))
            found |= _keys(v)
    elif isinstance(doc, (list, tuple)):
        for v in doc:
            found |= _keys(v)
    return found


def test_sol_legacy_and_strict_produce_same_numbers_but_only_strict_was_gated(
        verified, tmp_path):
    from solar.translate import sol
    from solar.translate.receipts import emit_receipt
    receipt = emit_receipt(verified, source_path=ADAPTER, dsl_path=DSL,
                           definition_path=DEFN,
                           workload_uuid="release-agentic-scale-rows-b32",
                           # strict sol compares axes against the workload.jsonl
                           # line VERBATIM (var axes only): {"B": 32}, not
                           # {"B": 32, "H": 16} — PARITY row G1 / README.
                           axes={"B": 32},
                           validation_axes={"B": 4}, dtype="fp16",
                           out_path=tmp_path / "receipt.json")
    legacy = sol(DSL, CUDA, tmp_path / "legacy", arch="B200", precision="fp16",
                 fallback_adapter=ADAPTER)
    strict = sol(DSL, CUDA, tmp_path / "strict", arch="B200", precision="fp16",
                 fallback_adapter=ADAPTER, strict=True, receipt=receipt,
                 source_path=ADAPTER)
    assert legacy.macs == strict.macs
    assert legacy.fused_ms == strict.fused_ms
    import yaml
    perf = yaml.safe_load(legacy.perf_path.read_text())
    an = yaml.safe_load(legacy.analysis_path.read_text())
    # Elementwise multiply-add is NOT a MAC in SOLAR accounting: it lands in
    # other_ops.  The trace ran at the ORIGINAL B=32 (not the B=4 validation):
    # declared fp32 boundary bytes are (32*16 + 16)*4 in and 32*16*4 out.
    assert legacy.macs == 0
    assert perf["workload"]["total_other_ops"] > 0          # observed 512, see A13
    assert an["total"]["external_input_bytes"] == (32 * 16 + 16) * 4
    assert an["total"]["external_output_bytes"] == 32 * 16 * 4
    assert perf["workload"]["boundary_billing"] == "declared_storage_dtypes"
    # The perf YAML is the perf stage's artifact and is left as written; the
    # validation label lives on SolResult and in validation.json (A11).  The
    # check is STRUCTURAL: no receipt/validation field at any depth of either
    # perf document and no pointer to the receipt file (a word inside the
    # descriptive SOL_CONVENTION prose is not a provenance pointer), and the
    # strict run writes the same perf document as the legacy run.
    strict_perf = yaml.safe_load(strict.perf_path.read_text())
    provenance_keys = {"receipt", "receipt_sha256", "receipt_path", "validation",
                       "validated", "validation_status"}
    assert _keys(perf).isdisjoint(provenance_keys), sorted(_keys(perf) & provenance_keys)
    assert _keys(strict_perf).isdisjoint(provenance_keys), sorted(_keys(strict_perf) & provenance_keys)
    assert str(receipt.resolve()) not in legacy.perf_path.read_text()
    assert str(receipt.resolve()) not in strict.perf_path.read_text()
    assert strict_perf == perf
    assert legacy.validated is False and legacy.receipt is None
    assert legacy.validation_status == "unvalidated_legacy"
    assert strict.validated is True and strict.receipt == receipt.resolve()
    assert strict.validation_status == "validated"
    assert json.loads(legacy.validation_path.read_text())["status"] == "unvalidated_legacy"
    assert json.loads(strict.validation_path.read_text())["receipt"] == str(receipt.resolve())


@pytest.mark.finding("A11")
def test_sol_cli_prints_validation_status(tmp_path):
    """Repaired: the legacy CLI run labels itself unvalidated (was: no
    marker at all)."""
    p = run([sys.executable, "-m", "solar.cli.translate", "sol", DSL,
             "--definition", CUDA, "--out-dir", tmp_path / "o",
             "--fallback-adapter", ADAPTER, "--arch", "B200",
             "--precision", "fp16"])
    assert p.returncode == 0, p.stderr[-1500:]
    assert p.stdout.startswith("SOL fused=")
    assert "validation: none (legacy, unvalidated)" in p.stdout
    assert "verified" not in p.stdout.lower()
