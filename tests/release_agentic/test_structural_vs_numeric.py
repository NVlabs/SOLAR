# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrappers that let STRUCTURAL success look like NUMERICAL validity.

Each defect is pinned twice: a plain test that reproduces today's behavior
(so the finding is machine-checked) and a strict xfail that states the
contract.  No inference API is called: the TranslationAgent's client is a
MagicMock and ``_call_llm`` is replaced by a canned response.

ADJUDICATION (release adjudication of the agentic contract, archive of the verbatim
audit file: an earlier revision).  Tests that pinned a DEFECT's old behavior are
rewritten to pin the repaired contract; strict xfails whose contract is now
delivered have the marker removed.  Policy-gated findings keep their
original xfail/pinning tests (A3 exit code, A10 tolerance): the legacy
default is deliberately preserved and only exposed/gated (see
test_repairs.py).
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import ZERO, lib_verify

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
# A1: TranslationAgent(verify=True) with no reference_outputs -> verified=True
# ---------------------------------------------------------------------------

@pytest.mark.finding("A1")
def test_A1_fixed_agent_reports_unverified_without_any_reference():
    """Repaired: execution-only is reported as such (was verified=True)."""
    agent = _mock_agent(verify=True)
    res = agent.translate("def f(x): return x", {"x": (2, 3)})   # no reference
    assert res.verified is False
    assert res.verification_status == "executed_no_reference"
    assert "no reference outputs" in res.verification_error


@pytest.mark.finding("A1")
def test_A1_contract_execution_only_is_not_verified():
    agent = _mock_agent(verify=True)
    res = agent.translate("def f(x): return x", {"x": (2, 3)})
    assert res.verified is False


def test_A1_control_with_reference_the_wrong_code_fails():
    """Control: the same mocked agent DOES fail when a reference is given."""
    agent = _mock_agent(verify=True)
    x = np.ones((2, 3))
    res = agent.translate("def f(x): return x", {"x": (2, 3)},
                          reference_outputs={"output": x}, all_inputs={"x": x})
    assert res.verified is False
    assert "values differ" in (res.verification_error or "")


def test_A1_no_verify_is_honest_at_agent_level():
    """``verify=False`` returns verified=False (not part of the defect)."""
    agent = _mock_agent(verify=False)
    res = agent.translate("def f(x): return x", {"x": (2, 3)})
    assert res.verified is False


# ---------------------------------------------------------------------------
# A2: translate(mode="api") decides by a substring of the child's log
# ---------------------------------------------------------------------------

def _fake_run(stdout: str, rc: int):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    return fake


@pytest.mark.finding("A2")
def test_A2_fixed_api_wrapper_ignores_marker_when_child_failed(tmp_path, monkeypatch):
    """Repaired: a failed child (rc=1) whose echo contains the marker text
    inside a code comment is NOT reported verified (was True)."""
    import solar.translate as T
    src = tmp_path / "m.py"
    src.write_text("x = 1\n")
    log = ("# GENERATED CODE (VERIFIED): not really\n"
           "  [FAILED] attempts=1\n")
    monkeypatch.setattr(T.subprocess, "run", _fake_run(log, rc=1))
    res = T.translate(src, tmp_path / "out", mode="api", api_key="sk-x")
    assert res.verified is False
    assert res.exit_code == 1 and res.verification_status == "child_failed"


@pytest.mark.finding("A2")
def test_A2_contract_api_wrapper_follows_exit_status(tmp_path, monkeypatch):
    import solar.translate as T
    src = tmp_path / "m.py"
    src.write_text("x = 1\n")
    log = "# GENERATED CODE (VERIFIED): not really\n"
    monkeypatch.setattr(T.subprocess, "run", _fake_run(log, rc=1))
    assert T.translate(src, tmp_path / "out", mode="api",
                       api_key="sk-x").verified is False


@pytest.mark.finding("A2")
def test_A2_fixed_passing_child_is_verified_even_without_marker(
        tmp_path, monkeypatch):
    """Repaired converse: rc=0 with the marker absent (quiet output) -> True;
    toDSL_model exits 0 iff its result was verified, so the exit status is
    the contract (was False)."""
    import solar.translate as T
    src = tmp_path / "m.py"
    src.write_text("x = 1\n")
    monkeypatch.setattr(T.subprocess, "run", _fake_run("[VERIFIED] attempts=1\n", rc=0))
    res = T.translate(src, tmp_path / "out", mode="api", api_key="sk-x")
    assert res.verified is True and res.verification_status == "verified"


# ---------------------------------------------------------------------------
# A3: shape-only exits 0 (script + library exit_code)
# ---------------------------------------------------------------------------

@pytest.mark.finding("A3")
def test_A3_current_shape_only_exit_code_is_zero():
    """POLICY (preserved, not fixed): the legacy default exit status of a
    shape-only run stays 0 for compatibility; the typed status and the
    opt-in strict exit (3) are pinned in test_repairs.py."""
    from solar.translate._runner import VerifyResult
    assert VerifyResult(verified=False, shape_only=True).exit_code == 0
    assert VerifyResult(verified=False, shape_only=True).status == "shape_only"


@pytest.mark.finding("A3")
@pytest.mark.xfail(strict=True, reason="FINDING A3 (policy-gated, open): the "
                   "legacy default exit status is preserved by decision; "
                   "exit_status(strict=True) == 3 is the gated contract")
def test_A3_contract_shape_only_is_not_exit_zero():
    from solar.translate._runner import VerifyResult
    assert VerifyResult(verified=False, shape_only=True).exit_code != 0


# ---------------------------------------------------------------------------
# A4: all-zero reference passes; nothing machine-readable records it
# ---------------------------------------------------------------------------

@pytest.mark.finding("A4")
def test_A4_fixed_vacuous_pass_still_passes_but_receipt_records_it(torch, tmp_path):
    """Repaired (policy preserved): the legacy verdict stays pass/exit 0,
    but the result, the sealed evidence and the receipt now record the
    vacuity (was: nothing machine-readable)."""
    from solar.translate.receipts import emit_receipt, read_receipt
    res = lib_verify(ZERO / "adapter.py", ZERO / "solar_model_einsum_lib.py",
                     source_type="cuda")
    assert res.verified is True
    assert res.max_abs_error == 0.0
    out = emit_receipt(res, source_path=ZERO / "adapter.py",
                       dsl_path=ZERO / "solar_model_einsum_lib.py",
                       definition_path=ZERO / "definition.json",
                       workload_uuid="definition-only:0", axes={"M": 3, "N": 4},
                       dtype="fp32", out_path=tmp_path / "r.json")
    r = read_receipt(out)
    assert r["verdicts"]["numerical"] == "pass"
    assert r["error_stats"]["vacuous_outputs"] == ["output"]
    assert r["execution"]["vacuous_reference_outputs"] == ["output"]


@pytest.mark.finding("A4")
def test_A4_contract_vacuous_pass_is_flagged(torch):
    res = lib_verify(ZERO / "adapter.py", ZERO / "solar_model_einsum_lib.py",
                     source_type="cuda")
    assert getattr(res, "vacuous", None) is True or "vacuous" in res.message.lower()


# ---------------------------------------------------------------------------
# A10: Pythonic TranslationAgent default tolerance differs from the CLI/skill
# ---------------------------------------------------------------------------

@pytest.mark.finding("A10")
def test_A10_current_agent_default_tolerance_is_wider_than_documented():
    """POLICY (preserved, not fixed): the fp32-execution tolerance of a direct
    TranslationAgent is unchanged; it is now recorded on the result
    (applied_tolerance) and warned about (test_repairs.py)."""
    import solar.einsum_lib.agent as A
    seen = {}
    orig = A.compare_outputs

    def spy(t, r, **kw):
        seen.update(kw)
        return orig(t, r, **kw)
    with patch.object(A, "compare_outputs", spy):
        agent = _mock_agent(verify=True)           # default use_fp64=False
        x = np.ones((2, 3))
        agent.translate("def f(x): return x", {"x": (2, 3)},
                        reference_outputs={"output": 2 * x}, all_inputs={"x": x})
    assert seen == {"atol": 1e-4, "rtol": 1e-4}   # documented default is atol 1e-5


@pytest.mark.finding("A10")
@pytest.mark.xfail(strict=True, reason="FINDING A10 (policy-gated, open): "
                   "direct TranslationAgent use applies atol=1e-4 by decision; "
                   "the applied tolerance is recorded, not silently changed")
def test_A10_contract_agent_default_matches_documented_tolerance():
    import solar.einsum_lib.agent as A
    seen = {}
    orig = A.compare_outputs

    def spy(t, r, **kw):
        seen.update(kw)
        return orig(t, r, **kw)
    with patch.object(A, "compare_outputs", spy):
        agent = _mock_agent(verify=True)
        x = np.ones((2, 3))
        agent.translate("def f(x): return x", {"x": (2, 3)},
                        reference_outputs={"output": 2 * x}, all_inputs={"x": x})
    assert seen.get("atol", 1e-5) == 1e-5
