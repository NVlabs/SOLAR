# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the agentic validation/receipt repairs (lane
the agentic-contract release adjudication).

Findings: the release adjudication's findings list (finding ids below).
Every test is offline (synthetic fixtures, mocked LLM client, no GPU).

Policy decisions that are NOT changed here (recorded, gated behind opt-in):
  * A3  shape-only still exits 0 by default; ``--strict`` / exit_status(strict=True) -> 3
  * A4  an all-zero reference still passes by default (exit 0) but is now
        recorded (result.vacuous_outputs, evidence, receipt) and refused by the
        strict receipt gate; ``--strict`` -> 3
  * A10 TranslationAgent(use_fp64=False) still applies atol=1e-4; the applied
        tolerance is now recorded on the result and warned about
"""
from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import CUDA, PT, ZERO, cli_verify, lib_verify, run, script_verify

DSL = CUDA / "solar_model_einsum_lib.py"
ADAPTER = CUDA / "adapter.py"
DEFN = CUDA / "definition.json"

IDENTITY_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
@kernel
def ident(A: Tensor("M,N")) -> Tensor("M,N"):
    C = Tensor(M=A.shape[0], N=A.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            C[m,n] = A[m,n] * 2.0
    return C
def model(x):
    return ident(x)
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N")
    return {"output": model(x).data}
'''


def _mock_agent(**kw):
    with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
        mock_mod.OpenAI.return_value = MagicMock()
        from solar.einsum_lib.agent import TranslationAgent
        kw.setdefault("api_key", "sk-offline-test")
        kw.setdefault("fast", False)
        kw.setdefault("max_retries", 1)
        agent = TranslationAgent(**kw)
    agent._call_llm = lambda messages: "```python\n" + IDENTITY_DSL + "\n```"
    return agent


# ---------------------------------------------------------------------------
# A1: no reference -> not verified, with a typed status; the code/metadata
# are still returned (structural execution happened), nothing is retried.
# ---------------------------------------------------------------------------

@pytest.mark.finding("A1")
def test_A1_no_reference_is_executed_but_unverified():
    agent = _mock_agent(verify=True)
    res = agent.translate("def f(x): return x", {"x": (2, 3)})
    assert res.verified is False
    assert res.verification_status == "executed_no_reference"
    assert "no reference" in res.verification_error.lower()
    assert res.code.strip()                 # structural result still delivered
    assert res.attempts == 1                # not retried: nothing failed


@pytest.mark.finding("A1")
def test_A1_positive_control_with_reference_verified_status():
    agent = _mock_agent(verify=True)
    x = np.ones((2, 3))
    res = agent.translate("def f(x): return x", {"x": (2, 3)},
                          reference_outputs={"output": 2 * x}, all_inputs={"x": x})
    assert res.verified is True
    assert res.verification_status == "verified"
    assert res.vacuous_outputs == ()
    assert res.applied_tolerance["atol"] == 1e-4      # A10: fp32 default unchanged


@pytest.mark.finding("A1")
def test_A1_negative_control_mismatch_status():
    agent = _mock_agent(verify=True)
    x = np.ones((2, 3))
    res = agent.translate("def f(x): return x", {"x": (2, 3)},
                          reference_outputs={"output": x}, all_inputs={"x": x})
    assert res.verified is False
    assert res.verification_status == "mismatch"
    assert "values differ" in res.verification_error


@pytest.mark.finding("A4")
def test_A4_agent_vacuous_pass_is_recorded_not_promoted():
    """Legacy default preserved: an all-zero reference still passes, but the
    result says so."""
    agent = _mock_agent(verify=True)
    x = np.zeros((2, 3))
    res = agent.translate("def f(x): return x", {"x": (2, 3)},
                          reference_outputs={"output": x}, all_inputs={"x": x})
    assert res.verified is True
    assert res.verification_status == "verified_vacuous"
    assert res.vacuous_outputs == ("output",)


@pytest.mark.finding("A10")
def test_A10_fp64_agent_uses_documented_default_tolerance():
    agent = _mock_agent(verify=True, use_fp64=True)
    x = np.ones((2, 3))
    res = agent.translate("def f(x): return x", {"x": (2, 3)},
                          reference_outputs={"output": 2 * x}, all_inputs={"x": x})
    assert res.verified
    assert res.applied_tolerance == {"rtol": 1e-4, "atol": 1e-5, "min_tol": 0.0}


# ---------------------------------------------------------------------------
# A2: api-mode verdict follows the child's exit status; a marker that
# disagrees with the exit status never promotes, and fails closed.
# ---------------------------------------------------------------------------

def _fake_run(stdout: str, rc: int):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    return fake


@pytest.mark.finding("A2")
@pytest.mark.parametrize("log, rc, expect, status", [
    ("# GENERATED CODE (VERIFIED): not really\n  [FAILED] attempts=1\n", 1,
     False, "child_failed"),
    ("GENERATED CODE (VERIFIED):\n...\n[VERIFIED] attempts=1\n", 0,
     True, "verified"),
    ("[VERIFIED] attempts=1\n", 0, True, "verified"),       # quiet child
    ("GENERATED CODE (UNVERIFIED):\n...\n", 0, False, "status_disagreement"),
    ("", 137, False, "child_failed"),
])
def test_A2_api_verdict_is_exit_status_not_marker(tmp_path, monkeypatch,
                                                  log, rc, expect, status):
    import solar.translate as T
    src = tmp_path / "m.py"
    src.write_text("x = 1\n")
    monkeypatch.setattr(T.subprocess, "run", _fake_run(log, rc))
    res = T.translate(src, tmp_path / "out", mode="api", api_key="sk-x")
    assert res.verified is expect
    assert res.exit_code == rc
    assert res.verification_status == status


# ---------------------------------------------------------------------------
# A3 / A4 on the runner: typed status, strict exit, legacy exit preserved
# ---------------------------------------------------------------------------

@pytest.mark.finding("A3")
def test_A3_shape_only_status_and_strict_exit(torch):
    r = lib_verify(ADAPTER, DSL, source_type="cuda", shape_only=True)
    assert r.status == "shape_only" and not r.verified
    assert r.exit_code == 0                      # legacy default preserved
    assert r.exit_status(strict=True) == 3       # strict proof requested: fail closed
    assert not r.vacuous_outputs


@pytest.mark.finding("A3")
def test_A3_script_and_cli_strict_shape_only_exit_3(torch):
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                      "--shape-only", "--strict")
    c = cli_verify(ADAPTER, DSL, "--source-type", "cuda", "--shape-only",
                   "--strict")
    assert s.returncode == 3 and c.returncode == 3
    assert "SHAPE-ONLY" in s.stdout and "SHAPE-ONLY" in c.stdout
    # without --strict the legacy exit status is unchanged
    s0 = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                       "--shape-only")
    assert s0.returncode == 0


@pytest.mark.finding("A4")
def test_A4_vacuous_pass_is_typed_and_strict_fails_closed(torch):
    r = lib_verify(ZERO / "adapter.py", ZERO / "solar_model_einsum_lib.py",
                   source_type="cuda")
    assert r.verified is True and r.exit_code == 0        # legacy preserved
    assert r.vacuous is True
    assert r.vacuous_outputs == ["output"]
    assert r.status == "pass_vacuous"
    assert "vacuous" in r.message.lower()
    assert r.exit_status(strict=True) == 3
    # the all-zero fact is sealed in the execution evidence (minted in-run)
    assert r.execution.vacuous_reference_outputs == ("output",)
    # measured, not flagged: the cuda adapter is fed fp64 casts and this
    # fixture's torch reference returns fp64
    assert dict(r.execution.reference_output_dtypes) == {"output": "fp64"}
    assert dict(r.execution.input_dtypes) == {"x": "fp32"}


@pytest.mark.finding("A4")
def test_A4_nontrivial_pass_is_not_vacuous(torch):
    r = lib_verify(ADAPTER, DSL, source_type="cuda")
    assert r.verified and r.status == "pass" and r.vacuous is False
    assert r.vacuous_outputs == [] and r.execution.vacuous_reference_outputs == ()
    assert r.exit_status(strict=True) == 0


@pytest.mark.finding("A4")
def test_A4_receipt_records_vacuity_and_strict_gate_refuses(torch, tmp_path):
    from solar.translate.receipts import (ReceiptRefusal, check_strict,
                                          emit_receipt, read_receipt)
    r = lib_verify(ZERO / "adapter.py", ZERO / "solar_model_einsum_lib.py",
                   source_type="cuda")
    out = emit_receipt(r, source_path=ZERO / "adapter.py",
                       dsl_path=ZERO / "solar_model_einsum_lib.py",
                       definition_path=ZERO / "definition.json",
                       workload_uuid="definition-only:0", axes={"M": 3, "N": 4},
                       dtype="fp32", out_path=tmp_path / "r.json")
    rec = read_receipt(out)
    assert rec["verdicts"]["numerical"] == "pass"           # verdict kind unchanged
    assert rec["error_stats"]["vacuous_outputs"] == ["output"]
    assert rec["execution"]["vacuous_reference_outputs"] == ["output"]
    common = dict(source_path=ZERO / "adapter.py",
                  dsl_path=ZERO / "solar_model_einsum_lib.py",
                  definition_path=ZERO / "definition.json",
                  workload_uuid="definition-only:0", axes={"M": 3, "N": 4},
                  dtype="fp32")
    with pytest.raises(ReceiptRefusal, match="vacuous"):
        check_strict(out, **common)
    # explicit, documented escape hatch for references that are zero by design
    assert check_strict(out, allow_vacuous=True, **common)["verdicts"]["numerical"] == "pass"


# ---------------------------------------------------------------------------
# A6: validation_axes derived from executed shapes; contradictions refused;
# capped validation distinguished; per-tensor dtypes bound
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def verified(torch):
    res = lib_verify(ADAPTER, DSL, source_type="cuda")
    assert res.verified, res.message
    return res


def _emit(verified, tmp_path, **kw):
    from solar.translate.receipts import emit_receipt, read_receipt
    kw.setdefault("axes", {"B": 32, "H": 16})
    out = emit_receipt(verified, source_path=ADAPTER, dsl_path=DSL,
                       definition_path=DEFN,
                       workload_uuid="release-agentic-scale-rows-b32",
                       dtype="fp16", out_path=tmp_path / "receipt.json", **kw)
    return read_receipt(out)


@pytest.mark.finding("A6")
def test_A6_validation_axes_derived_from_executed_shapes(verified, tmp_path):
    r = _emit(verified, tmp_path)
    assert r["workload"]["axes"] == {"B": 32, "H": 16}
    assert r["workload"]["validation_axes"] == {"B": 4, "H": 16}
    assert r["workload"]["validation_axes_source"] == "executed_shapes"
    assert r["workload"]["capped_validation"] is True
    assert r["execution"]["validation_shapes"]["x"] == [4, 16]


@pytest.mark.finding("A6")
def test_A6_explicit_validation_axes_are_cross_checked(verified, tmp_path):
    from solar.translate.receipts import ReceiptRefusal
    r = _emit(verified, tmp_path, validation_axes={"B": 4, "H": 16})
    assert r["workload"]["validation_axes"] == {"B": 4, "H": 16}
    assert r["workload"]["validation_axes_source"] == "caller_checked"
    with pytest.raises(ReceiptRefusal, match="validation_axes"):
        _emit(verified, tmp_path, validation_axes={"B": 32, "H": 16})


@pytest.mark.finding("A6")
def test_A6_uncapped_validation_is_not_flagged_capped(verified, tmp_path):
    r = _emit(verified, tmp_path, axes={"B": 4})
    assert r["workload"]["capped_validation"] is False


@pytest.mark.finding("A6")
def test_A6_strict_entry_cross_checks_validation_axes(verified, tmp_path):
    """A receipt whose validation_axes contradict its own executed shapes is
    refused at strict entry even though every hash matches."""
    from solar.translate.receipts import ReceiptRefusal, check_strict
    _emit(verified, tmp_path)
    p = tmp_path / "receipt.json"
    rec = json.loads(p.read_text())
    rec["workload"]["validation_axes"]["B"] = 32          # forged claim
    p.write_text(json.dumps(rec))
    with pytest.raises(ReceiptRefusal, match="validation_axes"):
        check_strict(p, source_path=ADAPTER, dsl_path=DSL, definition_path=DEFN,
                     workload_uuid="release-agentic-scale-rows-b32",
                     axes={"B": 32, "H": 16}, dtype="fp16")


@pytest.mark.finding("A6")
def test_A6_receipt_binds_per_tensor_dtypes(verified, tmp_path):
    r = _emit(verified, tmp_path)
    ex = r["execution"]
    assert ex["input_dtypes"] == {"x": "fp32", "w": "fp32"}          # source-declared
    assert ex["reference_input_dtypes"] == {"x": "fp64", "w": "fp64"}  # fed to kernel_fn
    assert ex["reference_output_dtypes"] == {"output": "fp64"}
    assert ex["verification_dtype"] == "fp64"


@pytest.mark.finding("A6")
def test_A6_pytorch_reference_dtypes_are_measured_not_flagged(torch):
    r = lib_verify(PT / "model.py", PT / "model_einsum_lib.py", source_type="pytorch")
    assert r.verified
    ev = r.execution
    assert ev.verification_dtype == "fp64"                       # IR flag
    assert dict(ev.reference_output_dtypes) == {"output": "fp32"}  # golden ran native


# ---------------------------------------------------------------------------
# A5: both verify CLIs emit a receipt through the one shared path
# ---------------------------------------------------------------------------

@pytest.mark.finding("A5")
def test_A5_script_and_cli_emit_identical_receipts(torch, tmp_path):
    from solar.translate.receipts import read_receipt
    rs, rc_ = tmp_path / "script.json", tmp_path / "cli.json"
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                      "--emit-receipt", rs, "--definition", CUDA, "--dtype", "fp16")
    c = cli_verify(ADAPTER, DSL, "--source-type", "cuda",
                   "--emit-receipt", rc_, "--definition", CUDA, "--dtype", "fp16")
    assert s.returncode == 0, s.stdout[-1500:] + s.stderr[-1500:]
    assert c.returncode == 0, c.stdout[-1500:] + c.stderr[-1500:]
    assert "receipt:" in s.stdout and "receipt:" in c.stdout
    a, b = read_receipt(rs), read_receipt(rc_)
    for k in ("verdicts", "hashes", "workload", "dtype", "tolerance", "checker",
              "execution"):
        assert a[k] == b[k], k
    assert a["workload"]["uuid"] == "release-agentic-scale-rows-b32"
    assert a["workload"]["axes"] == {"B": 32}                  # workload line VERBATIM
    assert a["workload"]["validation_axes"] == {"B": 4, "H": 16}
    assert a["workload"]["capped_validation"] is True


@pytest.mark.finding("A5")
def test_A5_script_receipt_admits_strict_sol(torch, tmp_path):
    from solar.translate import sol
    r = tmp_path / "r.json"
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                      "--emit-receipt", r, "--definition", CUDA, "--dtype", "fp16")
    assert s.returncode == 0
    res = sol(DSL, CUDA, tmp_path / "strict", arch="B200", precision="fp16",
              fallback_adapter=ADAPTER, strict=True, receipt=r, source_path=ADAPTER)
    assert res.validated is True and res.validation_status == "validated"
    assert res.receipt == r.resolve()


@pytest.mark.finding("A5")
def test_A5_failed_run_emits_fail_receipt_and_exit_1(torch, tmp_path):
    r = tmp_path / "r.json"
    s = script_verify(PT / "model.py", PT / "model_wrong_einsum_lib.py",
                      "--source-type", "pytorch", "--quiet",
                      "--emit-receipt", r, "--definition", CUDA, "--dtype", "fp16")
    assert s.returncode == 1
    from solar.translate.receipts import read_receipt
    assert read_receipt(r)["verdicts"]["numerical"] == "fail"


@pytest.mark.finding("A5")
def test_A5_receipt_request_needs_definition_and_dtype(torch, tmp_path):
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                      "--emit-receipt", tmp_path / "r.json")
    assert s.returncode == 2 and "--definition" in s.stderr
    c = cli_verify(ADAPTER, DSL, "--source-type", "cuda",
                   "--emit-receipt", tmp_path / "r.json", "--definition", CUDA)
    assert c.returncode == 2 and "--dtype" in c.stderr


@pytest.mark.finding("A5")
def test_A5_receipt_refusal_exits_3(torch, tmp_path):
    """The verification itself passes, but the receipt cannot be bound: the
    definition declares no input sharing a name with an executed tensor, so
    validation_axes are underivable and (no --validation-axes exists on the
    CLI) the original workload axes are NEVER assumed -> RECEIPT REFUSED,
    exit 3, no receipt file."""
    import shutil
    d = tmp_path / "underivable"
    d.mkdir()
    shutil.copy(ADAPTER, d / "adapter.py")
    shutil.copy(DSL, d / "solar_model_einsum_lib.py")
    shutil.copy(CUDA / "workload.jsonl", d / "workload.jsonl")
    defn = json.loads(DEFN.read_text())
    defn["inputs"] = {"inp": defn["inputs"]["x"], "wt": defn["inputs"]["w"]}
    (d / "definition.json").write_text(json.dumps(defn))
    s = script_verify(d / "adapter.py", d / "solar_model_einsum_lib.py",
                      "--source-type", "cuda", "--quiet",
                      "--emit-receipt", tmp_path / "r.json", "--definition", d,
                      "--dtype", "fp16")
    assert "VERIFIED" in s.stdout, s.stdout + s.stderr
    assert s.returncode == 3
    assert "RECEIPT REFUSED" in s.stdout + s.stderr
    assert "validation_axes cannot be derived" in s.stdout + s.stderr
    assert not (tmp_path / "r.json").exists()


# ---------------------------------------------------------------------------
# A9: CLI verify matches the script on missing inputs
# ---------------------------------------------------------------------------

@pytest.mark.finding("A9")
def test_A9_cli_and_script_agree_on_missing_input():
    nope = "/nonexistent/release_agentic/model.py"
    s = script_verify(nope, PT / "model_einsum_lib.py", "--source-type", "pytorch")
    c = cli_verify(nope, PT / "model_einsum_lib.py", "--source-type", "pytorch")
    assert s.returncode == c.returncode == 2
    assert "ERROR: model file not found" in s.stderr
    assert "ERROR: model file not found" in c.stderr
    assert "Traceback" not in c.stderr


# ---------------------------------------------------------------------------
# A7 / A11: sol CLI strict/receipt switches and a validation status line
# ---------------------------------------------------------------------------

@pytest.mark.finding("A7")
def test_A7_sol_cli_exposes_strict_and_forwarded_flags():
    p = run([sys.executable, "-m", "solar.cli.translate", "sol", "--help"])
    assert p.returncode == 0
    for flag in ("--strict", "--receipt", "--source-path", "--workload-idx",
                 "--perf-precision", "--axis", "--mem-budget", "--measure-timeout",
                 "--measure-attention-density", "--allow-unbound-constants"):
        assert flag in p.stdout, flag


@pytest.mark.finding("A11")
def test_A11_legacy_sol_is_labelled_unvalidated(tmp_path):
    from solar.translate import sol
    res = sol(DSL, CUDA, tmp_path / "legacy", arch="B200", precision="fp16",
              fallback_adapter=ADAPTER)
    assert res.validated is False
    assert res.validation_status == "unvalidated_legacy"
    assert res.receipt is None
    side = json.loads((tmp_path / "legacy" / "validation.json").read_text())
    assert side["status"] == "unvalidated_legacy" and side["receipt"] is None
    assert side["strict"] is False


@pytest.mark.finding("A11")
def test_A11_sol_cli_prints_validation_line_both_ways(torch, tmp_path):
    p = run([sys.executable, "-m", "solar.cli.translate", "sol", DSL,
             "--definition", CUDA, "--out-dir", tmp_path / "legacy",
             "--fallback-adapter", ADAPTER, "--arch", "B200", "--precision", "fp16"])
    assert p.returncode == 0, p.stderr[-1500:]
    assert p.stdout.startswith("SOL fused=")
    assert "validation: none (legacy, unvalidated)" in p.stdout

    r = tmp_path / "r.json"
    s = script_verify(ADAPTER, DSL, "--source-type", "cuda", "--quiet",
                      "--emit-receipt", r, "--definition", CUDA, "--dtype", "fp16")
    assert s.returncode == 0
    q = run([sys.executable, "-m", "solar.cli.translate", "sol", DSL,
             "--definition", CUDA, "--out-dir", tmp_path / "strict",
             "--fallback-adapter", ADAPTER, "--arch", "B200", "--precision", "fp16",
             "--strict", "--receipt", r, "--source-path", ADAPTER])
    assert q.returncode == 0, q.stderr[-1500:]
    assert f"validation: receipt {r.resolve()}" in q.stdout
    side = json.loads((tmp_path / "strict" / "validation.json").read_text())
    assert side["status"] == "validated" and side["strict"] is True
    assert side["receipt_sha256"]


@pytest.mark.finding("A11")
def test_A11_sol_cli_strict_refusal_exits_3_before_tracing(tmp_path):
    q = run([sys.executable, "-m", "solar.cli.translate", "sol", DSL,
             "--definition", CUDA, "--out-dir", tmp_path / "o",
             "--fallback-adapter", ADAPTER, "--strict",
             "--receipt", tmp_path / "none.json", "--source-path", ADAPTER])
    assert q.returncode == 3
    assert "REFUSED" in q.stderr and "receipt missing" in q.stderr
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# A12: empty outputs get a verdict, never a traceback
# ---------------------------------------------------------------------------

@pytest.mark.finding("A12")
def test_A12_empty_output_is_a_failed_verdict():
    from solar.einsum_lib.verify import compare_outputs
    r = compare_outputs({"y": np.zeros((0, 4))}, {"y": np.zeros((0, 4))})
    assert r.success is False
    assert "empty" in r.error and "'y'" in r.error
    assert r.max_abs_error is None
    assert r.output_shapes_match is True


@pytest.mark.finding("A12")
def test_A12_empty_reference_with_nonempty_translation_is_shape_mismatch():
    from solar.einsum_lib.verify import compare_outputs
    r = compare_outputs({"y": np.zeros((1, 4))}, {"y": np.zeros((0, 4))})
    assert not r.success and "Shape mismatch" in r.error


@pytest.mark.finding("A12")
def test_A12_nonempty_controls_unchanged():
    from solar.einsum_lib.verify import compare_outputs
    ok = compare_outputs({"y": np.ones((2, 2))}, {"y": np.ones((2, 2))})
    assert ok.success and ok.max_abs_error == 0.0 and ok.vacuous_outputs == ()
    zero = compare_outputs({"y": np.zeros((2, 2))}, {"y": np.zeros((2, 2))})
    assert zero.success and zero.vacuous_outputs == ("y",)
