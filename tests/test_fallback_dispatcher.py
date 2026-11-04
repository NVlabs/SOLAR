# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic-to-agentic fallback dispatcher (solar.frontend.dispatcher).

CPU only, no external API: providers are faked/mocked, the native coding
agent is a fake that consumes TRANSLATION_REQUEST.json.  Numerical
validation of candidates is REAL (``solar.translate.verify`` on the
release_agentic pt_matmul fixture) so receipt/source binding is exercised
at the layer that owns it.  No comparator, tolerance or protected
assertion is touched.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from solar.frontend import dispatcher as D
from solar.frontend.dispatcher import (
    Budget, DeterministicOutcome, DispatchPolicy, DispatchRequest,
    FailureKind, LedgerReservation, ProviderError, PureSkillProvider,
    PythonicProvider, RESPONSE_SCHEMA, classify_exception, classify_message,
    dispatch, resume,
)

SOLAR = Path(__file__).resolve().parents[1]
PT = SOLAR / "tests" / "release_agentic" / "fixtures" / "pt_matmul"
GOOD_DSL = PT / "model_einsum_lib.py"
WRONG_DSL = PT / "model_wrong_einsum_lib.py"
CRASH_DSL = PT / "model_crash_einsum_lib.py"

pytest.importorskip("torch")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# helpers: fixture copies, fake deterministic stages, fake agents
# ---------------------------------------------------------------------------

@pytest.fixture
def model(tmp_path):
    """A private copy of the pt_matmul model (tests may mutate it)."""
    m = tmp_path / "src" / "model.py"
    m.parent.mkdir()
    m.write_text((PT / "model.py").read_text())
    return m


@pytest.fixture
def definition(tmp_path):
    """A definition for pt_matmul so receipts can be emitted."""
    d = tmp_path / "defn"
    d.mkdir()
    (d / "definition.json").write_text(json.dumps({
        "name": "pt_matmul_dispatcher_test",
        "axes": {"M": {"type": "var"}, "K": {"type": "const", "value": 4},
                 "N": {"type": "const", "value": 3}},
        "inputs": {"x": {"shape": ["M", "K"], "dtype": "float32"},
                   "w": {"shape": ["K", "N"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["M", "N"], "dtype": "float32"}},
    }))
    (d / "workload.jsonl").write_text(
        json.dumps({"uuid": "pt-matmul-m2", "axes": {"M": 2},
                    "tolerance": {"max_rtol": 1e-4, "max_atol": 1e-5,
                                  "required_matched_ratio": 1.0}}) + "\n")
    return d


def det_stage(status, reasons=("unknown handler semantics for 'custom_op' (Model.op)",)):
    """A fake deterministic stage with a fixed classified outcome."""
    def stage(request, policy, out_dir, log, **_):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rec = None
        if status != "ok":
            rec = D._receipt(
                FailureKind.INFRASTRUCTURE if status == "infrastructure_error"
                else FailureKind.UNSUPPORTED, "deterministic", "; ".join(reasons),
                route="mechanical")
        return DeterministicOutcome(status=status, reasons=list(reasons), receipt=rec)
    return stage


def fake_agent(dsl_src, *, suffix="", response=None, before=None):
    """An in-process native-agent stand-in: consumes the request, writes the
    candidate (bytes of dsl_src + suffix) and a bound response."""
    def agent(req_path):
        treq = json.loads(Path(req_path).read_text())
        if before:
            before(treq)
        cand = Path(treq["candidate_path"])
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text(Path(dsl_src).read_text() + suffix)
        resp = {"schema": RESPONSE_SCHEMA, "request_id": treq["request_id"],
                "source_sha256": treq["source_sha256"], "status": "candidate",
                "candidate_sha256": _sha(cand),
                "provenance": {"author": "fake-agent", "model": "none",
                               "route": "pure_skill", "created_utc": "test"}}
        resp.update(response or {})
        (Path(req_path).parent / "TRANSLATION_RESPONSE.json").write_text(json.dumps(resp))
    return agent


TOL = {"rtol": 1e-4, "atol": 1e-5}   # the tests' EXPLICIT tolerance policy


def req(model, tmp_path, **kw):
    # an explicit policy unless a definition (whose workload line carries the
    # row tolerance) is given: the dispatcher never substitutes a default
    if "tolerance" not in kw and kw.get("definition_dir") is None:
        kw["tolerance"] = dict(TOL)
    kw.setdefault("source_type", "pytorch")
    return DispatchRequest(source_path=model, out_dir=tmp_path / "out", **kw)


def receipt_kinds(res):
    return [r["kind"] for r in res.failure_receipts]


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

class TestClassification:
    @pytest.mark.parametrize("exc, kind", [
        (TimeoutError("x"), FailureKind.INFRASTRUCTURE),
        (MemoryError(), FailureKind.INFRASTRUCTURE),
        (ConnectionError("reset"), FailureKind.INFRASTRUCTURE),
        (subprocess.TimeoutExpired("cmd", 5), FailureKind.INFRASTRUCTURE),
        (FileNotFoundError("model.py"), FailureKind.INPUT),
        (ValueError("dim K not registered"), FailureKind.UNSUPPORTED),
        (TypeError("bad operand"), FailureKind.UNSUPPORTED),
        (RuntimeError("process killed rc=137"), FailureKind.INFRASTRUCTURE),
        (RuntimeError("Cannot allocate memory"), FailureKind.INFRASTRUCTURE),
        (RuntimeError("HTTP 503 upstream"), FailureKind.INFRASTRUCTURE),
        (RuntimeError("MISMATCH values differ"), FailureKind.UNSUPPORTED),
    ])
    def test_exception_kinds(self, exc, kind):
        assert classify_exception(exc) == kind

    def test_sdk_exception_recognised_by_name_without_import(self):
        APITimeoutError = type("APITimeoutError", (Exception,), {})
        assert classify_exception(APITimeoutError("slow")) == FailureKind.INFRASTRUCTURE

    def test_messages(self):
        assert classify_message("IR execution FAILED after 3.0s: TimeoutError: timed out") \
            == FailureKind.INFRASTRUCTURE
        assert classify_message("IR execution FAILED after 0.1s: NameError: name 'Q' is not defined") \
            == FailureKind.UNSUPPORTED
        assert classify_message("") == FailureKind.UNSUPPORTED

    def test_receipt_scoring_flags(self):
        infra = D._receipt(FailureKind.INFRASTRUCTURE, "s", "r")
        unsup = D._receipt(FailureKind.UNSUPPORTED, "s", "r")
        assert infra["retained_for_retry"] and not infra["scored"]
        assert unsup["scored"] and not unsup["retained_for_retry"]


# ---------------------------------------------------------------------------
# policy / budget validation
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_attempts_bounded_nonzero(self):
        with pytest.raises(ValueError):
            Budget(max_attempts=0)

    def test_route_must_be_known(self):
        with pytest.raises(ValueError):
            DispatchPolicy(route="magic")

    def test_reservation_fails_closed_when_overdrawn(self):
        r = LedgerReservation(1, 1.0, label="t")
        r.consume(1, 0.5)
        with pytest.raises(ProviderError) as e:
            r.consume(1)
        assert e.value.kind == FailureKind.NO_BUDGET
        assert r.remaining_requests == 0 and r.to_dict()["consumptions"]


# ---------------------------------------------------------------------------
# 1. success: deterministic route completes (REAL v1 on pt_matmul)
# ---------------------------------------------------------------------------

def test_success_mechanical_route_no_fallback(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(precision="fp32", arch="H100_PCIe"),
                   provider=None)
    assert res.status == "mechanical_ok" and res.final_route == "mechanical"
    assert res.attempts == [] and res.failure_receipts == []
    assert res.why is None
    det = res.deterministic
    assert det["status"] == "ok"
    assert det["route_decision"]["route"] == "mechanical"
    assert Path(det["artifacts"]["perf"]).is_file()
    assert det["artifacts"]["macs"] == 2 * 4 * 3
    assert "no numerical comparison" in det["numerical_validation"]
    rec = json.loads(Path(res.receipt_path).read_text())
    assert rec["schema"] == D.SCHEMA and rec["status"] == "mechanical_ok"


def test_failed_deterministic_validation_is_eligible_for_fallback(model, tmp_path):
    """A caller-supplied deterministic numerical validation that fails makes
    the (otherwise complete) mechanical result ineligible and falls back."""
    r = req(model, tmp_path,
            deterministic_validator=lambda artifacts: (False, "reference mismatch 0.31"))
    res = dispatch(r, DispatchPolicy(precision="fp32", arch="H100_PCIe"),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)))
    assert res.deterministic["status"] == "validation_failed"
    assert res.why["deterministic_status"] == "validation_failed"
    assert any("reference mismatch 0.31" in r for r in res.why["deterministic_limitation"])
    assert res.status == "agentic_verified"
    assert res.failure_receipts[0]["stage"] == "deterministic"
    assert Path(res.deterministic["artifacts"]["perf"]).is_file()  # artifacts retained


# ---------------------------------------------------------------------------
# 2. deterministic unsupported (REAL v1) -> reasons recorded -> fail closed w/o provider
# ---------------------------------------------------------------------------

UNSUPPORTED_MODEL = textwrap.dedent('''
    import torch, torch.nn as nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter((torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1.0) / 10.0)
        def forward(self, x):
            return torch.sigmoid(x @ self.w)   # sigmoid is outside the bounded supported set
    def get_inputs():
        return [(torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1.0) / 7.0]
''')


def test_unsupported_capture_records_exact_why_and_fails_closed_without_provider(tmp_path):
    m = tmp_path / "unsupported_model.py"
    m.write_text(UNSUPPORTED_MODEL)
    res = dispatch(req(m, tmp_path), DispatchPolicy(precision="fp32"), provider=None)
    assert res.status == "no_provider" and res.final_route == "none"
    det = res.deterministic
    assert det["status"] == "unsupported"
    assert det["route_decision"]["route"] == "agentic"
    assert any("sigmoid" in r and "unknown handler semantics" in r for r in det["reasons"])
    # why = deterministic limitation + fallback reason
    assert res.why["deterministic_limitation"] == det["reasons"]
    assert "pure_skill" in res.why["fallback_reason"]
    # original failure receipt retained, then the fail-closed receipt
    assert receipt_kinds(res) == [FailureKind.UNSUPPORTED, FailureKind.NO_PROVIDER]
    assert res.failure_receipts[0]["stage"] == "deterministic"


def test_missing_source_fails_closed_as_input_error(tmp_path):
    res = dispatch(req(tmp_path / "nope.py", tmp_path), provider=PureSkillProvider(agent=lambda p: None))
    assert res.status == "input_error" and res.attempts == []
    assert receipt_kinds(res) == [FailureKind.INPUT]


def test_deterministic_infrastructure_error_is_not_eligible(model, tmp_path):
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("infrastructure_error", ("capture timed out",)))
    assert res.status == "deterministic_infrastructure_error"
    assert res.attempts == [] and res.why is None
    assert res.failure_receipts[0]["retained_for_retry"] and not res.failure_receipts[0]["scored"]


# ---------------------------------------------------------------------------
# 2b. data-dependent sparsity families: agentic-by-default, expected limitation
# ---------------------------------------------------------------------------

MOE_LIKE_MODEL = textwrap.dedent('''
    import torch, torch.nn as nn
    class Model(nn.Module):
        """Top-k routing + gather: data-dependent sparsity (MoE-like)."""
        def __init__(self):
            super().__init__()
            self.router = nn.Parameter((torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1.0) / 20.0)
        def forward(self, x):
            logits = x @ self.router                  # (T, E)
            top_val, top_idx = torch.topk(logits, 2, dim=-1)
            return torch.gather(logits, 1, top_idx) + top_val
    def get_inputs():
        return [(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1.0) / 7.0]
''')


def test_moe_like_ops_are_agentic_by_default_expected_limitation(tmp_path):
    m = tmp_path / "moe_like.py"
    m.write_text(MOE_LIKE_MODEL)
    res = dispatch(req(m, tmp_path), DispatchPolicy(precision="fp32"),
                   provider=PureSkillProvider(), deterministic_stage=None)
    det = res.deterministic
    assert det["status"] == "unsupported" and det["expected_limitation"] is True
    assert "topk" in det["sparsity_family"] and "gather" in det["sparsity_family"]
    # the deterministic attempt is retained in the report (capture + route decision)
    assert Path(det["artifacts"]["graph"]).is_file()
    assert det["route_decision"]["route"] in ("agentic", "unsupported")
    assert any("expected limitation, not a defect" in r for r in det["reasons"])
    # why + SOL source: DSL-route sparse measured-density analysis
    assert res.why["expected_limitation"] is True
    assert "agentic-by-default" in res.why["fallback_reason"]
    assert res.why["sol_source"] == D.SOL_SOURCE_DSL_SPARSE
    assert res.status == "awaiting_agent"
    assert res.sol_source["source"] == D.SOL_SOURCE_DSL_SPARSE and res.sol_source["status"] == "blocked"
    treq = json.loads((Path(res.resume_token) / "TRANSLATION_REQUEST.json").read_text())
    assert treq["deterministic_failure"]["sol_source"] == D.SOL_SOURCE_DSL_SPARSE


def test_declared_sparsity_family_routes_agentic_even_when_lowering_would_succeed(model, tmp_path):
    """Catalog says MoE: agentic-by-default with the reason recorded, while the
    deterministic attempt (here a mechanical route decision) stays in the report."""
    r = req(model, tmp_path, sparsity_family_hint="moe")
    res = dispatch(r, DispatchPolicy(precision="fp32", arch="H100_PCIe"),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)))
    det = res.deterministic
    assert det["status"] == "unsupported" and det["expected_limitation"] is True
    assert det["sparsity_family"] == {"declared": "moe"}
    assert det["route_decision"]["route"] == "mechanical"      # attempt retained
    assert res.status == "agentic_verified"
    assert res.sol_source["source"] == D.SOL_SOURCE_DSL_SPARSE
    # verified but no definition/dtype -> no sealed receipt -> NOT ready (sol-9)
    assert res.sol_source["status"] == "blocked" and "missing receipt inputs" in res.sol_source["blocker"]
    assert res.sol_source["candidate"] and res.sol_source["receipt"] is None
    assert "measure_density=True" in res.sol_source["note"]


def test_sol_source_is_mechanical_or_dsl_dense_otherwise(model, tmp_path):
    ok = dispatch(req(model, tmp_path / "a"), DispatchPolicy(precision="fp32", arch="H100_PCIe"))
    assert ok.sol_source["source"] == D.SOL_SOURCE_MECHANICAL and ok.sol_source["perf"]
    dense = dispatch(req(model, tmp_path / "b"), provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                     deterministic_stage=det_stage("unsupported"))
    assert dense.status == "agentic_verified"
    assert dense.sol_source["source"] == D.SOL_SOURCE_DSL_DENSE
    assert dense.sol_source["status"] == "blocked"      # no receipt inputs were given
    assert dense.why["expected_limitation"] is False


def test_data_dependent_sparsity_op_set_is_bounded_and_detected():
    g = {"layers": {"a": {"type": "topk"}, "b": {"type": "matmul"}, "c": {"type": "gather"},
                    "d": {"type": "gather"}}}
    assert D.data_dependent_sparsity_ops(g) == {"topk": 1, "gather": 2}
    assert D.data_dependent_sparsity_ops({"layers": {"b": {"type": "matmul"}}}) == {}
    assert "matmul" not in D.DATA_DEPENDENT_SPARSITY_OPS


# ---------------------------------------------------------------------------
# 3. deterministic failure -> agentic success (pure-skill, real validation)
# ---------------------------------------------------------------------------

def test_deterministic_failure_then_agentic_success(model, tmp_path, definition):
    r = req(model, tmp_path, definition_dir=definition, dtype="fp32")
    res = dispatch(r, DispatchPolicy(route="pure_skill", budget=Budget(max_attempts=2)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_verified" and res.final_route == "pure_skill"
    assert len(res.attempts) == 1 and res.attempts[0]["outcome"] == "verified"
    # original deterministic failure retained verbatim and first
    assert res.failure_receipts[0]["stage"] == "deterministic"
    assert res.failure_receipts[0]["reason"].startswith("unknown handler semantics")
    assert res.deterministic["status"] == "unsupported"
    # candidate + validation + hash-bound receipt
    assert res.candidate["dsl_sha256"] == _sha(res.candidate["dsl_path"])
    assert res.candidate["source_sha256"] == _sha(model) == res.source_sha256
    assert res.candidate["provenance"]["author"] == "fake-agent"
    assert res.validation["status"] == "verified" and res.validation["max_abs_error"] < 1e-5
    receipt = json.loads(Path(res.validation["receipt_path"]).read_text())
    assert receipt["verdicts"]["numerical"] == "pass"
    assert res.sol_source["status"] == "ready" and res.sol_source["receipt"] == res.validation["receipt_path"]
    assert res.validation["tolerance_source"].startswith("workload[") and res.validation["tolerance"]["rtol"] == 1e-4
    assert receipt["tolerance"]["rtol"] == 1e-4 and receipt["tolerance"]["atol"] == 1e-5
    assert receipt["hashes"]["source_sha256"] == _sha(model)
    assert receipt["hashes"]["dsl_sha256"] == res.candidate["dsl_sha256"]
    assert receipt["workload"]["uuid"] == "pt-matmul-m2"
    # the request the agent consumed carried the full contract
    treq = json.loads((Path(res.attempts[0]["request_dir"]) / "TRANSLATION_REQUEST.json").read_text())
    for key in ("source_sha256", "source_type", "dtype", "route", "attempt", "max_attempts",
                "caps", "prior_candidate_sha256s", "deterministic_failure", "candidate_path"):
        assert key in treq
    assert treq["deterministic_failure"]["deterministic_limitation"] == res.deterministic["reasons"]


# ---------------------------------------------------------------------------
# 4. agentic failure after N attempts; mismatch is scored unsupported
# ---------------------------------------------------------------------------

def test_agentic_failure_after_n_attempts(model, tmp_path):
    calls = []

    def agent(p):
        calls.append(p)
        fake_agent(WRONG_DSL, suffix=f"\n# attempt {len(calls)}\n")(p)

    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=3)),
                   provider=PureSkillProvider(agent=agent), deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert len(calls) == 3 and len(res.attempts) == 3
    assert [a["outcome"] for a in res.attempts] == ["mismatch"] * 3
    kinds = receipt_kinds(res)
    assert kinds[0] == FailureKind.UNSUPPORTED            # deterministic, retained
    assert kinds[1:4] == [FailureKind.UNSUPPORTED] * 3    # three scored mismatches
    assert kinds[4] == FailureKind.EXHAUSTED
    assert all(r["scored"] for r in res.failure_receipts[1:4])
    assert all("MISMATCH" in r["reason"] for r in res.failure_receipts[1:4])
    assert res.ledger["attempts_used"] == 3 and res.ledger["max_attempts"] == 3
    assert res.candidate is None


def test_identical_candidate_bytes_across_attempts_are_not_fresh(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=2)),
                   provider=PureSkillProvider(agent=fake_agent(WRONG_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    # attempt 1 mismatched; attempt 2 reproduced the same bytes -> refused by the
    # provider boundary (its sha is in prior_candidate_sha256s), scored unsupported
    assert res.attempts[0]["outcome"] == "mismatch"
    assert res.attempts[1]["outcome"] == FailureKind.UNSUPPORTED
    assert "not byte-distinct" in res.attempts[1]["reason"]


# ---------------------------------------------------------------------------
# 5. no provider / no budget -> fail closed, deterministic failure retained
# ---------------------------------------------------------------------------

def test_no_provider_fails_closed_retaining_deterministic_failure(model, tmp_path):
    res = dispatch(req(model, tmp_path), provider=None, deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_provider" and res.attempts == []
    assert receipt_kinds(res) == [FailureKind.UNSUPPORTED, FailureKind.NO_PROVIDER]
    assert res.deterministic["reasons"]


def test_route_mismatch_is_no_provider(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic"),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_provider"
    assert "serves route 'pure_skill'" in res.failure_receipts[-1]["reason"]


def test_allow_agentic_false_never_dispatches(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(allow_agentic=False),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "deterministic_failed_no_fallback" and res.attempts == []


def test_pythonic_without_explicit_key_is_no_provider(model, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-must-be-ignored")
    calls = []
    prov = PythonicProvider(api_key=None, translate_fn=lambda *a, **k: calls.append(1))
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic"), provider=prov,
                   reservation=LedgerReservation(20, 50.0), deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_provider" and not calls
    assert "never reads OPENAI_API_KEY" in res.failure_receipts[-1]["reason"]


def test_pythonic_without_reservation_is_no_budget(model, tmp_path):
    calls = []
    prov = PythonicProvider(api_key="explicit", translate_fn=lambda *a, **k: calls.append(1))
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic"), provider=prov,
                   reservation=None, deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_budget" and not calls and res.attempts == []
    assert receipt_kinds(res) == [FailureKind.UNSUPPORTED, FailureKind.NO_BUDGET]


def test_pythonic_with_empty_reservation_is_no_budget(model, tmp_path):
    calls = []
    prov = PythonicProvider(api_key="explicit", translate_fn=lambda *a, **k: calls.append(1))
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic"), provider=prov,
                   reservation=LedgerReservation(0, 50.0), deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_budget" and not calls


# ---------------------------------------------------------------------------
# Pythonic shim with a mocked translate(): consumes exactly one request
# ---------------------------------------------------------------------------

def _mock_translate(dsl_src, verified=True):
    from solar.translate import TranslateResult

    def translate(source_path, out_dir, mode, source_type, api_key, **kw):
        assert mode == "api" and api_key == "explicit"
        assert kw["max_retries"] == 1 and kw["max_cap_rounds"] == 1
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(source_path).stem}_einsum_lib.py").write_text(Path(dsl_src).read_text())
        return TranslateResult(mode="api", out_dir=out, verified=verified, log="mock",
                               exit_code=0 if verified else 1,
                               verification_status="verified" if verified else "child_failed")
    return translate


def test_pythonic_shim_success_consumes_one_reservation_request(model, tmp_path, definition):
    reservation = LedgerReservation(20, 50.0, label="acceptance-ledger")
    prov = PythonicProvider(api_key="explicit", translate_fn=_mock_translate(GOOD_DSL))
    r = req(model, tmp_path, definition_dir=definition, dtype="fp32")
    # an explicit pre-transport bound is required for admission when the
    # provider exposes no cost channel (cost-13)
    res = dispatch(r, DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                     budget=Budget(max_attempts=2, max_requests=5)),
                   provider=prov, reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_verified" and res.final_route == "pythonic"
    assert res.candidate["provenance"]["cost"]["bound_used_usd"] == 0.1
    assert reservation.requests_used == 1 and res.ledger["api_requests_used"] == 1
    assert res.ledger["reservation"]["label"] == "acceptance-ledger"
    assert res.candidate["provenance"]["route"] == "pythonic"
    assert res.validation["receipt_path"] and Path(res.validation["receipt_path"]).is_file()


def test_pythonic_child_claim_is_not_trusted_dispatcher_revalidates(model, tmp_path):
    """The child says 'verified' but saved a wrong DSL: the dispatcher's own
    verification decides, and the mismatch is scored."""
    reservation = LedgerReservation(3, 50.0)
    prov = PythonicProvider(api_key="explicit", translate_fn=_mock_translate(WRONG_DSL, verified=True))
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1)),
                   provider=prov, reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert res.attempts[0]["outcome"] == "mismatch"
    assert res.attempts[0]["candidate"]["provenance"]["child_verification_status"] == "verified"
    assert reservation.requests_used == 1


def test_pythonic_api_timeout_is_infrastructure_and_still_consumes(model, tmp_path):
    def translate(*a, **k):
        raise TimeoutError("read timed out")
    reservation = LedgerReservation(5, 50.0)
    prov = PythonicProvider(api_key="explicit", translate_fn=translate)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", budget=Budget(max_attempts=2),
                                                       cost_per_request_usd=1.0),
                   provider=prov, reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_infrastructure_error"
    assert res.ledger["api_requests_used"] == 2 and res.ledger["cost_usd_used"] == 2.0
    infra = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"]
    assert len(infra) == 2 and all(r["kind"] == FailureKind.INFRASTRUCTURE for r in infra)
    assert all(r["retained_for_retry"] and not r["scored"] for r in infra)
    assert reservation.requests_used == 2   # a request that timed out still cost a request


# ---------------------------------------------------------------------------
# 6. limit exhaustion: wall, request and cost caps
# ---------------------------------------------------------------------------

def test_wall_cap_exhaustion_before_any_attempt(model, tmp_path):
    calls = []
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=3, wall_seconds=0.0)),
                   provider=PureSkillProvider(agent=lambda p: calls.append(p)),
                   deterministic_stage=det_stage("unsupported"))
    # with a zero cap even the deterministic outcome arrives late: rejected as
    # infrastructure before any fallback, nothing is generated
    assert res.status == "deterministic_infrastructure_error" and not calls and res.attempts == []
    assert res.failure_receipts[-1]["kind"] == FailureKind.INFRASTRUCTURE
    assert "late result" in res.failure_receipts[-1]["reason"] and "wall cap" in res.failure_receipts[-1]["reason"]


def test_wall_cap_stops_between_attempts(model, tmp_path):
    def slow_agent(p):
        time.sleep(0.3)
        fake_agent(WRONG_DSL, suffix=f"\n# {time.time()}\n")(p)
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=5, wall_seconds=0.2)),
                   provider=PureSkillProvider(agent=slow_agent), deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert len(res.attempts) == 1 and res.attempts[0]["outcome"] == "late_result"
    late = [r for r in res.failure_receipts if "late result" in r["reason"]]
    assert late and late[0]["kind"] == FailureKind.INFRASTRUCTURE and not late[0]["scored"]
    assert "wall cap" in res.failure_receipts[-1]["reason"]
    assert res.candidate is None


def test_request_cap_exhaustion_on_pythonic(model, tmp_path):
    reservation = LedgerReservation(20, 50.0)
    prov = PythonicProvider(api_key="explicit", translate_fn=_mock_translate(WRONG_DSL))
    res = dispatch(req(model, tmp_path),
                   DispatchPolicy(route="pythonic", budget=Budget(max_attempts=5, max_requests=1),
                                  cost_per_request_usd=1.0),
                   provider=prov, reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert len(res.attempts) == 1 and reservation.requests_used == 1
    assert "request cap 1 exhausted" in res.failure_receipts[-1]["reason"]


def test_reservation_runs_dry_mid_dispatch_is_no_budget(model, tmp_path):
    reservation = LedgerReservation(1, 50.0)
    prov = PythonicProvider(api_key="explicit", translate_fn=_mock_translate(WRONG_DSL))
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", budget=Budget(max_attempts=5),
                                                       cost_per_request_usd=1.0),
                   provider=prov, reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert len(res.attempts) == 1 and reservation.remaining_requests == 0
    assert "no remaining requests" in res.failure_receipts[-1]["reason"]


# ---------------------------------------------------------------------------
# 7. infrastructure vs unsupported classification through the dispatcher
# ---------------------------------------------------------------------------

class RaisingProvider(PureSkillProvider):
    def __init__(self, exc):
        super().__init__(agent=lambda p: (_ for _ in ()).throw(exc))


def test_provider_timeout_is_infrastructure_retained_not_scored(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=2)),
                   provider=RaisingProvider(TimeoutError("agent host contention")),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_infrastructure_error"
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"]
    assert [r["kind"] for r in gen] == [FailureKind.INFRASTRUCTURE] * 2
    assert all(r["retained_for_retry"] and not r["scored"] for r in gen)
    assert res.failure_receipts[-1]["kind"] == FailureKind.EXHAUSTED
    assert "not scored" in res.failure_receipts[-1]["reason"]


def test_candidate_crash_is_scored_unsupported(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=1)),
                   provider=PureSkillProvider(agent=fake_agent(CRASH_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert res.attempts[0]["outcome"] == "candidate_error"
    val = [r for r in res.failure_receipts if r["stage"] == "agentic_validate"][0]
    assert val["kind"] == FailureKind.UNSUPPORTED and val["scored"]
    assert "IR execution FAILED" in val["reason"]


def test_validation_timeout_is_infrastructure(model, tmp_path):
    def validator(cand, request, policy, receipt_out, log):
        return D.ValidationVerdict(status="infrastructure", kind=FailureKind.INFRASTRUCTURE,
                                   message="IR execution FAILED after 900.0s: TimeoutError: timed out",
                                   verify_status="error")
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=1)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), validator=validator,
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_infrastructure_error"
    val = [r for r in res.failure_receipts if r["stage"] == "agentic_validate" and r["kind"] == FailureKind.INFRASTRUCTURE]
    assert len(val) == 2 and all(v["retained_for_retry"] for v in val)
    assert res.attempts[0]["outcome"] == "pending_infrastructure"
    assert res.pending_validation["candidate"]["dsl_sha256"] == res.attempts[0]["candidate_sha256"]
    assert res.ledger["validation_retries_used"] == 1 and res.ledger["pending_infrastructure"] == 1
    assert res.ledger["model_attempts_used"] == 1 and res.ledger["scorable_outcomes"] == 0


def test_agent_declined_is_scored_unsupported_with_reason(model, tmp_path):
    def agent(p):
        treq = json.loads(Path(p).read_text())
        (Path(p).parent / "TRANSLATION_RESPONSE.json").write_text(json.dumps({
            "schema": RESPONSE_SCHEMA, "request_id": treq["request_id"],
            "source_sha256": treq["source_sha256"], "status": "declined",
            "reason": "data-dependent gather with unbounded index range"}))
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(agent=agent),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.UNSUPPORTED and "unbounded index range" in gen["reason"]


def test_agent_error_without_kind_is_protocol_not_guessed(model, tmp_path):
    def agent(p):
        treq = json.loads(Path(p).read_text())
        (Path(p).parent / "TRANSLATION_RESPONSE.json").write_text(json.dumps({
            "schema": RESPONSE_SCHEMA, "request_id": treq["request_id"],
            "source_sha256": treq["source_sha256"], "status": "error", "reason": "boom"}))
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(agent=agent),
                   deterministic_stage=det_stage("unsupported"))
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.PROTOCOL and not gen["scored"] and gen["retained_for_retry"]


# ---------------------------------------------------------------------------
# 8. receipt / source binding at the responsible layer
# ---------------------------------------------------------------------------

def test_response_must_echo_request_id_and_source_sha(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=2)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL, response={"request_id": "stale"})),
                   deterministic_stage=det_stage("unsupported"))
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"]
    assert gen and all(r["kind"] == FailureKind.PROTOCOL and "request_id" in r["reason"] for r in gen)
    assert res.status == "agentic_infrastructure_error"   # protocol faults are retained, never scored

    res2 = dispatch(req(model, tmp_path / "b"), DispatchPolicy(),
                    provider=PureSkillProvider(agent=fake_agent(GOOD_DSL, response={"source_sha256": "0" * 64})),
                    deterministic_stage=det_stage("unsupported"))
    gen2 = [r for r in res2.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen2["kind"] == FailureKind.PROTOCOL and "source_sha256" in gen2["reason"]
    assert res2.candidate is None


def test_candidate_sha_in_response_must_match_file(model, tmp_path):
    res = dispatch(req(model, tmp_path),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL, response={"candidate_sha256": "f" * 64})),
                   deterministic_stage=det_stage("unsupported"))
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.PROTOCOL and "candidate_sha256" in gen["reason"]


def test_old_saved_dsl_is_never_marked_freshly_generated(model, tmp_path):
    """The acceptance runs' saved DSL is a known candidate: an agent that hands
    it back is refused as not fresh, even though it would verify."""
    saved = tmp_path / "saved" / "old_einsum_lib.py"
    saved.parent.mkdir()
    saved.write_text(GOOD_DSL.read_text())
    r = req(model, tmp_path, known_candidates=(saved,))
    res = dispatch(r, provider=PureSkillProvider(agent=fake_agent(saved)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted" and res.candidate is None
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert "not byte-distinct" in gen["reason"] and "never recorded as new work" in gen["reason"]
    assert "freshly generated" not in json.dumps(res.to_dict())
    treq = json.loads((Path(res.attempts[0]["request_dir"]) / "TRANSLATION_REQUEST.json").read_text())
    assert _sha(saved) in treq["prior_candidate_sha256s"]


def test_source_changed_during_generation_refuses_validation(model, tmp_path):
    def mutate(treq):
        model.write_text(model.read_text() + "\n# mutated while the agent worked\n")
    agent = fake_agent(GOOD_DSL, before=mutate)   # response echoes the ORIGINAL sha
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(agent=agent),
                   deterministic_stage=det_stage("unsupported"))
    binding = [r for r in res.failure_receipts if r["stage"] == "source_binding"]
    assert binding and binding[0]["kind"] == FailureKind.PROTOCOL
    assert res.candidate is None and res.validation is None


def test_receipt_emission_refusal_is_no_proof(model, tmp_path, definition):
    """A definition whose input names do not match the executed tensors makes
    the receipt underivable: the pass is recorded as no_proof, not verified."""
    (definition / "definition.json").write_text(json.dumps({
        "name": "mismatched", "axes": {"M": {"type": "var"}},
        "inputs": {"hidden": {"shape": ["M", 4], "dtype": "float32"}}}))
    r = req(model, tmp_path, definition_dir=definition, dtype="fp32")
    res = dispatch(r, provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_exhausted"
    assert res.attempts[0]["outcome"] == "no_proof"
    assert "receipt refused" in res.attempts[0]["validation"]["message"]


# ---------------------------------------------------------------------------
# 9. pure-skill request / consume / resume cycle (asynchronous native agent)
# ---------------------------------------------------------------------------

def test_pure_skill_request_consume_resume_cycle(model, tmp_path, definition):
    r = req(model, tmp_path, definition_dir=definition, dtype="fp32",
            original_axes={"M": 2080, "K": 4, "N": 3})
    policy = DispatchPolicy(budget=Budget(max_attempts=2, wall_seconds=600))
    # 1. dispatch writes the durable request and returns awaiting_agent
    res = dispatch(r, policy, provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    assert res.status == "awaiting_agent" and res.resume_token
    request_dir = Path(res.resume_token)
    treq = json.loads((request_dir / "TRANSLATION_REQUEST.json").read_text())
    assert treq["schema"] == D.REQUEST_SCHEMA and treq["status"] == "open"
    assert treq["source_sha256"] == _sha(model)
    assert treq["original_axes"] == {"M": 2080, "K": 4, "N": 3} and treq["dtype"] == "fp32"
    assert treq["route"] == "pure_skill" and treq["attempt"] == 1 and treq["max_attempts"] == 2
    # round 13 (the review PUBLIC-PROPERTY-SCOPE-31): the request carries the POLICY (absolute deadline,
    # cap) and no remaining-wall reading; providers derive the remaining time from deadline_utc
    assert treq["caps"]["wall_seconds_remaining"] is None and treq["caps"]["wall_seconds_cap"] == 600
    assert treq["caps"]["deadline_utc"] and 0 < D._caps_remaining(treq["caps"]) <= 600
    assert treq["deterministic_failure"]["deterministic_limitation"]
    assert not (request_dir / "TRANSLATION_RESPONSE.json").exists()
    # 2. resume before the agent responded: still awaiting, nothing counted
    res_b = resume(tmp_path / "out", PureSkillProvider())
    assert res_b.status == "awaiting_agent" and res_b.ledger["attempts_used"] == 0
    # 3. the (fake) native agent consumes the request out of band
    fake_agent(GOOD_DSL)(request_dir / "TRANSLATION_REQUEST.json")
    # 4. resume validates the FRESH candidate and emits the receipt
    res_c = resume(tmp_path / "out", PureSkillProvider())
    assert res_c.status == "agentic_verified"
    assert res_c.ledger["attempts_used"] == 1
    assert res_c.candidate["dsl_sha256"] == _sha(treq["candidate_path"])
    assert res_c.candidate["provenance"]["request_id"] == treq["request_id"]
    receipt = json.loads(Path(res_c.validation["receipt_path"]).read_text())
    assert receipt["verdicts"]["numerical"] == "pass"
    assert receipt["hashes"]["dsl_sha256"] == res_c.candidate["dsl_sha256"]
    # the deterministic failure survived the whole cycle, first and verbatim
    assert res_c.failure_receipts[0]["stage"] == "deterministic"
    assert res_c.deterministic["reasons"] == res.deterministic["reasons"]
    # the final receipt on disk says the same
    rec = json.loads(Path(res_c.receipt_path).read_text())
    assert rec["status"] == "agentic_verified" and rec["why"]["deterministic_limitation"]


def test_resume_refuses_when_source_changed_while_awaiting(model, tmp_path):
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "awaiting_agent"
    fake_agent(GOOD_DSL)(Path(res.resume_token) / "TRANSLATION_REQUEST.json")
    model.write_text(model.read_text() + "\n# edited\n")
    res2 = resume(tmp_path / "out", PureSkillProvider())
    assert res2.status == "source_changed"
    assert res2.failure_receipts[-1]["stage"] == "source_binding"
    assert res2.candidate is None


def test_resume_after_mismatch_continues_bounded_loop(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=2)),
                   provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(WRONG_DSL)(Path(res.resume_token) / "TRANSLATION_REQUEST.json")
    res2 = resume(tmp_path / "out", PureSkillProvider())
    # attempt 1 mismatched; attempt 2's request is written and awaits the agent
    assert res2.status == "awaiting_agent"
    assert res2.attempts[0]["outcome"] == "mismatch" and res2.attempts[1]["outcome"] == "awaiting_agent"
    treq2 = json.loads((Path(res2.resume_token) / "TRANSLATION_REQUEST.json").read_text())
    assert treq2["attempt"] == 2
    assert res2.attempts[0]["candidate_sha256"] in treq2["prior_candidate_sha256s"]
    fake_agent(GOOD_DSL)(Path(res2.resume_token) / "TRANSLATION_REQUEST.json")
    res3 = resume(tmp_path / "out", PureSkillProvider())
    assert res3.status == "agentic_verified" and res3.ledger["attempts_used"] == 2


def test_resume_of_finished_dispatch_is_refused(model, tmp_path):
    dispatch(req(model, tmp_path), provider=None, deterministic_stage=det_stage("unsupported"))
    with pytest.raises(ValueError, match="nothing to resume"):
        resume(tmp_path / "out", PureSkillProvider())


def test_poll_window_without_response_is_infrastructure(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=1)),
                   provider=PureSkillProvider(poll_seconds=0.2, poll_interval=0.05),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_infrastructure_error"
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.INFRASTRUCTURE and "poll window" in gen["reason"]


# ---------------------------------------------------------------------------
# CLI smoke (real v1 success path; exit codes)
# ---------------------------------------------------------------------------

def test_cli_run_mechanical_success_exit_0(model, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "solar.frontend.dispatcher", "run", str(model),
         "--out-dir", str(tmp_path / "cli"), "--precision", "fp32", "--arch", "H100_PCIe",
         "--no-agentic"], capture_output=True, text=True, cwd=SOLAR)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("DISPATCH-SUMMARY: ")][-1].split(": ", 1)[1])
    assert summary["status"] == "mechanical_ok"


def test_cli_run_unsupported_without_agent_awaits_exit_4(tmp_path):
    m = tmp_path / "unsupported_model.py"
    m.write_text(UNSUPPORTED_MODEL)
    proc = subprocess.run(
        [sys.executable, "-m", "solar.frontend.dispatcher", "run", str(m),
         "--out-dir", str(tmp_path / "cli"), "--precision", "fp32"],
        capture_output=True, text=True, cwd=SOLAR)
    assert proc.returncode == 4, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("DISPATCH-SUMMARY: ")][-1].split(": ", 1)[1])
    assert summary["status"] == "awaiting_agent"
    assert Path(summary["resume_token"], "TRANSLATION_REQUEST.json").is_file()
    assert any("sigmoid" in r for r in summary["why"]["deterministic_limitation"])


# ---------------------------------------------------------------------------
# round-2 repairs (the review findings): transport budget, pending-infra resume,
# status-code classification, declared family carry-through, stage faults
# ---------------------------------------------------------------------------

from solar.einsum_lib.transport_budget import (  # noqa: E402
    TransportBudget, TransportBudgetExhausted, active_transport_budget,
    charge_transport, permits_for, transport_budget,
)


class TestTransportBudget:
    def test_charges_at_the_call_boundary_and_fails_closed(self):
        r = LedgerReservation(3, 3.0, label="t")
        b = TransportBudget(permits=2, cost_per_request_usd=1.0, reservation=r, prepaid=1)
        r.consume(1, 1.0, note="prepaid")              # the invocation prepays request 1
        with transport_budget(b):
            charge_transport("m")                      # request 1: prepaid, no new debit
            assert r.requests_used == 1 and b.requests_used == 1
            charge_transport("m")                      # request 2: debited at the boundary
            assert r.requests_used == 2 and r.usd_used == 2.0
            with pytest.raises(TransportBudgetExhausted):
                charge_transport("m")                  # permit cap: refused BEFORE a call
        assert active_transport_budget() is None
        assert b.requests_used == 2 and b.cost_preauthorized_usd == 2.0

    def test_reservation_shortfall_refuses_before_the_call(self):
        r = LedgerReservation(5, 1.5, label="t")
        b = TransportBudget(permits=None, cost_per_request_usd=1.0, reservation=r, prepaid=0)
        b.charge("m")
        with pytest.raises(TransportBudgetExhausted, match="cannot preauthorize"):
            b.charge("m")
        assert r.requests_used == 1 and r.usd_used == 1.0

    def test_permits_for_takes_the_tightest_bound(self):
        assert permits_for(3, 10, 10.0, None, 1.0, 5) == 3
        assert permits_for(None, 2, 10.0, None, 1.0, 5) == 2
        assert permits_for(None, 10, 1.5, None, 1.0, 5) == 1
        assert permits_for(None, 10, 10.0, 0.5, 1.0, 5) == 0
        assert permits_for(None, None, None, None, 0.0, 1) == 1

    def test_no_active_budget_is_a_no_op(self):
        charge_transport("m")   # must not raise


def test_agent_inner_fallback_is_capped_by_the_transport_budget(model, tmp_path):
    """The real TranslationAgent._complete with a fake transport: a 503 would
    fall back to the next model; the budget refuses the second request
    before it is sent and the reservation shows exactly one debit."""
    from solar.einsum_lib.agent import TranslationAgent
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        exc = RuntimeError("offline simulated HTTP 503")
        exc.status_code = 503
        raise exc

    def translate(src, out, **kw):
        a = TranslationAgent.__new__(TranslationAgent)
        a.model, a.temperature, a.max_tokens, a.thinking = "fake-model", 0, 10, False
        a.rate_limit_retries, a.rate_limit_delay = 3, 0
        a.client = type("C", (), {})()
        a.client.chat = type("Ch", (), {})()
        a.client.chat.completions = type("Co", (), {"create": staticmethod(completion)})()
        a._complete([])

    reservation = LedgerReservation(10, 10.0, label="t")
    res = dispatch(req(model, tmp_path),
                   DispatchPolicy(route="pythonic", budget=Budget(max_attempts=1, max_requests=1),
                                  cost_per_request_usd=1.0),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert len(calls) == 1                      # the fallback call was refused before sending
    assert reservation.requests_used == 1 and reservation.usd_used == 1.0
    assert res.ledger["api_requests_used"] == 1 and res.ledger["cost_usd_used"] == 1.0
    gen = [r for r in res.failure_receipts if r["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.EXHAUSTED and not gen["scored"]
    assert gen["api_requests"] == 1
    assert res.status == "agentic_exhausted"


def test_reported_usage_is_reconciled_against_the_preauthorization(model, tmp_path):
    def translate(*a, **k):
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = 0.25
        return r
    reservation = LedgerReservation(5, 10.0, label="t")
    res = dispatch(req(model, tmp_path),
                   DispatchPolicy(route="pythonic", cost_per_request_usd=2.0),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_verified"
    assert res.candidate["cost_basis"] == "reported" and res.candidate["cost_usd"] == 0.25
    assert reservation.usd_used == pytest.approx(0.25) and reservation.requests_used == 1
    assert res.ledger["cost_usd_used"] == pytest.approx(0.25)


def test_whole_reservation_is_preauthorized_without_an_estimate(model, tmp_path):
    """No cost_per_request_usd: one request preauthorizes the whole remaining
    reservation, so a second attempt is refused for want of funds."""
    reservation = LedgerReservation(5, 3.0, label="t")
    res = dispatch(req(model, tmp_path),
                   DispatchPolicy(route="pythonic", budget=Budget(max_attempts=3)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=_mock_translate(WRONG_DSL)),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert reservation.usd_used == 3.0 and reservation.requests_used == 1
    assert len(res.attempts) == 1 and "no remaining funds" in res.failure_receipts[-1]["reason"]


class TestClassificationByStageTypeStatus:
    def test_http_status_on_a_generic_exception_is_infrastructure(self):
        exc = RuntimeError("provider rejected")
        exc.status_code = 403
        assert classify_exception(exc) == FailureKind.INFRASTRUCTURE

    def test_auth_type_name_is_infrastructure(self):
        E = type("AuthenticationError", (Exception,), {})
        assert classify_exception(E("bad key")) == FailureKind.INFRASTRUCTURE

    def test_candidate_stage_never_uses_text(self):
        assert classify_exception(ValueError("Connection is not a tensor"),
                                  stage="candidate") == FailureKind.UNSUPPORTED
        assert classify_exception(RuntimeError("HTTP 503 inside the DSL message"),
                                  stage="candidate") == FailureKind.UNSUPPORTED
        assert classify_exception(TimeoutError("x"), stage="candidate") == FailureKind.INFRASTRUCTURE

    def test_ir_failure_message_classified_by_type_only(self):
        assert D.classify_candidate_error_message(
            "IR execution FAILED after 0.1s: RuntimeError: run_model() failed: Connection reset") \
            == FailureKind.UNSUPPORTED
        assert D.classify_candidate_error_message(
            "IR execution FAILED after 9.0s: TimeoutError: timed out") == FailureKind.INFRASTRUCTURE


def test_pending_infrastructure_candidate_is_retained_under_the_persisted_retry_budget(model, tmp_path):
    calls = []

    def validator(cand, request, policy, receipt_out, log):
        calls.append(cand.dsl_sha256)
        if len(calls) <= 2:
            return D.ValidationVerdict("infrastructure", FailureKind.INFRASTRUCTURE,
                                       "host timeout", verify_status="error")
        return D.ValidationVerdict("verified", "", "VERIFIED", verify_status="pass")

    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_attempts=1, max_validation_retries=1)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), validator=validator,
                   deterministic_stage=det_stage("unsupported"))
    # first verdict infrastructure -> the one budgeted retry runs on the SAME
    # bytes -> still infrastructure -> pending; the candidate is retained
    assert res.status == "agentic_infrastructure_error" and res.pending_validation
    assert len(calls) == 2 and len(set(calls)) == 1
    assert res.ledger["validation_retries_used"] == 1 and res.ledger["model_attempts_used"] == 1
    # resume: the TOTAL retry budget is spent, so no further validation runs; the
    # receipt never shows used > max and the candidate stays retained
    res2 = resume(tmp_path / "out", PureSkillProvider(), validator=validator)
    assert res2.status == "agentic_infrastructure_error" and len(calls) == 2
    assert res2.pending_validation["candidate"]["dsl_sha256"] == calls[0]
    assert res2.ledger["validation_retries_used"] == 1 <= res2.ledger["max_validation_retries"]
    assert res2.ledger["model_attempts_used"] == 1 and res2.ledger["scorable_outcomes"] == 0


def test_declared_family_survives_raw_source_and_stage_exception(model, tmp_path, monkeypatch):
    raw = dispatch(req(model, tmp_path / "a", source_type="raw", sparsity_family_hint="moe"),
                   provider=PureSkillProvider(), deterministic_stage=None)
    assert raw.deterministic["expected_limitation"] is True
    assert raw.deterministic["sparsity_family"] == {"declared": "moe"}
    assert raw.sol_source["source"] == D.SOL_SOURCE_DSL_SPARSE

    def boom(request, policy, out_dir, log, **_):
        raise ValueError("unsupported dynamic capture")
    exc = dispatch(req(model, tmp_path / "b", sparsity_family_hint="sparse_attention"),
                   provider=PureSkillProvider(), deterministic_stage=boom)
    # a stage that RAISES is an infrastructure/stage fault: durable receipt, no fallback
    assert exc.status == "deterministic_infrastructure_error"
    assert exc.failure_receipts[0]["stage"] == "deterministic" and not exc.failure_receipts[0]["scored"]
    assert Path(exc.receipt_path).is_file()


def test_missing_capture_artifact_gets_a_durable_unscored_receipt(model, tmp_path, monkeypatch):
    from solar.graph.pytorch_processor import PyTorchProcessor
    monkeypatch.setattr(PyTorchProcessor, "process_model_file", lambda *a, **k: True)
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider())
    assert res.status == "deterministic_infrastructure_error"
    assert res.deterministic["artifacts"]["stage_error"]["stage"] == "graph_read"
    assert res.failure_receipts[0]["kind"] == FailureKind.INFRASTRUCTURE
    assert Path(res.receipt_path).is_file() and res.attempts == []


def test_tolerance_unresolved_admits_no_verdict(model, tmp_path):
    res = dispatch(req(model, tmp_path, tolerance=None),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "tolerance_unresolved" and res.candidate is None
    v = res.attempts[0]["validation"]
    assert v["status"] == "tolerance_unresolved" and v["tolerance_source"] == "unresolved"
    assert res.ledger["scorable_outcomes"] == 0


# ---------------------------------------------------------------------------
# round-3 repairs (the review round 2): reference binding, charging, cost cap,
# deadline, retry re-hash / persisted cap, durable failure receipts
# ---------------------------------------------------------------------------

CUDA_FX = SOLAR / "tests" / "release_agentic" / "fixtures" / "cuda_scale"


def test_reference_binding_is_a_checked_assertion_on_the_receipt(model, tmp_path, definition):
    r = req(model, tmp_path, definition_dir=definition, dtype="fp32")
    res = dispatch(r, provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_verified"
    b = res.validation["reference_binding"]
    assert b["checked"] is True and b["bound"] is True
    assert b["requested_sha256"] == _sha(model) == b["executed_reference_sha256"]


def test_pass_without_execution_evidence_is_not_admitted(model, tmp_path, monkeypatch):
    # the verifier runs in a child now: fake the verify PHASE's summary (a pass
    # that carries no execution evidence)
    monkeypatch.setattr(D, "_run_verify_phase",
                        lambda *a, **k: ({"status": "pass", "message": "fake", "max_abs_error": 0.0, "execution": None}, None))
    c = D.Candidate(GOOD_DSL, model, _sha(GOOD_DSL), _sha(model))
    v = D.default_validator(c, req(model, tmp_path), DispatchPolicy(), tmp_path / "r.json")
    assert v.status == "no_proof" and v.reason_code == "reference_binding_unavailable"


def test_adapter_with_changed_reference_is_refused_with_both_hashes(tmp_path):
    """The reviewer's REFERENCE-14 case through the real verifier: the adapter
    embeds x*w+2 for a requested x*w+1 reference; the pass is for a different
    program and is refused with adapter_reference_mismatch (unscored)."""
    import ast as _ast, shutil
    adapter_src = (CUDA_FX / "adapter.py").read_text()
    tree = _ast.parse(adapter_src)
    reference = next(_ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, _ast.Assign) and any(getattr(t, "id", None) == "_REFERENCE_SRC" for t in n.targets))
    original = tmp_path / "reference.py"
    original.write_text(reference)
    defn = tmp_path / "definition"
    defn.mkdir()
    for name in ("definition.json", "workload.jsonl"):
        shutil.copy2(CUDA_FX / name, defn / name)

    def consume(path):
        rq = json.loads(Path(path).read_text())
        Path(rq["adapter_path"]).write_text(adapter_src.replace("return x * w + 1.0", "return x * w + 2.0"))
        Path(rq["candidate_path"]).write_text((CUDA_FX / "solar_model_einsum_lib.py").read_text()
                                              .replace("X[b, h] * W[h] + 1.0", "X[b, h] * W[h] + 2.0"))
        resp = {"schema": RESPONSE_SCHEMA, "request_id": rq["request_id"], "source_sha256": rq["source_sha256"],
                "candidate_sha256": _sha(rq["candidate_path"]), "status": "candidate",
                "provenance": {"author": "fake", "model": "none"}}
        (Path(path).parent / "TRANSLATION_RESPONSE.json").write_text(json.dumps(resp))

    r = DispatchRequest(original, tmp_path / "out", source_type="cuda", definition_dir=defn, dtype="fp32",
                        tolerance={"rtol": 1e-4, "atol": 1e-5})
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified and res.candidate is None
    v = res.attempts[0]["validation"]
    assert v["status"] == "adapter_reference_mismatch" and v["reason_code"] == "adapter_reference_mismatch"
    assert v["reference_binding"]["requested_sha256"] == _sha(original)
    assert v["reference_binding"]["executed_reference_sha256"] not in (None, _sha(original))
    assert v["reference_binding"]["adapter_sha256"] == _sha(Path(res.attempts[0]["request_dir"]) / "adapter.py")
    rec = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert rec["kind"] == FailureKind.PROTOCOL and not rec["scored"]


def test_terminal_transport_call_is_charged(model, tmp_path):
    from solar.einsum_lib.agent import TranslationAgent, _FALLBACK_MODELS
    calls = []

    def create(**kw):
        calls.append(kw["model"])
        e = RuntimeError("offline HTTP 503")
        e.status_code = 503
        raise e

    def translate(src, out, **kw):
        a = TranslationAgent.__new__(TranslationAgent)
        a.model, a.temperature, a.max_tokens, a.thinking = "m0", 0, 10, False
        a.rate_limit_retries, a.rate_limit_delay = 1, 0
        a.client = type("C", (), {})()
        a.client.chat = type("Ch", (), {})()
        a.client.chat.completions = type("Co", (), {"create": staticmethod(create)})()
        a._complete([])

    n_models = len(["m0"] + [m for m in _FALLBACK_MODELS if m != "m0"])
    reservation = LedgerReservation(10, 10.0)
    res = dispatch(req(model, tmp_path),
                   DispatchPolicy(route="pythonic", cost_per_request_usd=1.0,
                                  budget=Budget(max_attempts=1), max_transport_requests_per_attempt=n_models),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    # the terminal call after all fallback models would be call n_models+1: refused
    assert len(calls) == n_models and reservation.requests_used == n_models
    assert res.ledger["api_requests_used"] == n_models


def test_post_transport_processing_fault_keeps_the_debit_unscored(model, tmp_path):
    class Bad:
        @property
        def cost_usd(self):
            raise ValueError("decoder")

    def translate(src, out, **kw):
        _mock_translate(GOOD_DSL)(src, out, **kw)
        return Bad()
    reservation = LedgerReservation(3, 3.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=1.0,
                                                       budget=Budget(max_attempts=1)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    # a cost accessor that raises is a malformed cost report: infrastructure,
    # unscored, the pre-transport bound stays debited (cost-13)
    assert gen["kind"] == FailureKind.INFRASTRUCTURE and not gen["scored"] and gen["api_requests"] == 1
    assert gen["reason_code"] == "cost_malformed" and gen["cost_usd"] == 1.0
    assert res.ledger["api_requests_used"] == 1 == reservation.requests_used
    disk = json.loads(Path(res.receipt_path).read_text())
    assert disk["ledger"]["api_requests_used"] == 1


def test_measured_overrun_is_cost_cap_exceeded(model, tmp_path):
    def translate(*a, **k):
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = 0.25
        return r
    reservation = LedgerReservation(2, 1.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_cost_usd=0.1)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.status == "no_budget" and not res.verified
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.NO_BUDGET and gen["reason_code"] == "cost_cap_exceeded"
    assert reservation.usd_used == pytest.approx(0.25) and res.ledger["cost_usd_used"] == pytest.approx(0.25)


def test_reported_unavailable_cost_fails_closed_under_finite_cap_only(model, tmp_path):
    def translate(*a, **k):
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = None
        return r
    capped = dispatch(req(model, tmp_path / "a"), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                                budget=Budget(max_cost_usd=1.0)),
                      provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                      reservation=LedgerReservation(2, 1.0), deterministic_stage=det_stage("unsupported"))
    assert not capped.verified
    gen = [x for x in capped.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.INFRASTRUCTURE and gen["reason_code"] == "cost_unbounded" and not gen["scored"]
    # without a finite cost cap the explicit per-request hard cap is the bound
    uncapped = dispatch(req(model, tmp_path / "b"), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1),
                        provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                        reservation=LedgerReservation(2, 1.0), deterministic_stage=det_stage("unsupported"))
    assert uncapped.verified and uncapped.candidate["cost_basis"] == "hard_cap_preauthorized"


def test_token_usage_times_rate_is_the_measured_cost(model, tmp_path):
    from solar.einsum_lib.transport_budget import charge_transport, record_transport_usage

    def translate(*a, **k):
        charge_transport("m")
        record_transport_usage("m", {"prompt_tokens": 1500, "completion_tokens": 500, "total_tokens": 2000})
        return _mock_translate(GOOD_DSL)(*a, **k)
    reservation = LedgerReservation(2, 1.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.5,
                                                       budget=Budget(max_cost_usd=1.0)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate, cost_per_1k_tokens_usd=0.01),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert res.verified and res.candidate["cost_basis"] == "reported_tokens"
    assert res.candidate["cost_usd"] == pytest.approx(0.02) and reservation.usd_used == pytest.approx(0.02)


_REAL_TIME = time


class _Clock:
    epoch, mono = 100.0, 1000.0
    def time(self): return self.epoch
    def monotonic(self): return self.mono
    def perf_counter(self): return self.mono
    gmtime = staticmethod(_REAL_TIME.gmtime)
    strftime = staticmethod(_REAL_TIME.strftime)
    sleep = staticmethod(_REAL_TIME.sleep)


def test_late_mismatch_is_unscored_and_late_mechanical_rejected(model, tmp_path, monkeypatch):
    Clock = _Clock
    clock = Clock()
    monkeypatch.setattr(D, "time", clock)

    def validator(*a):
        clock.mono = 1001.5
        return D.ValidationVerdict("mismatch", FailureKind.UNSUPPORTED, "late mismatch")
    res = dispatch(req(model, tmp_path / "a"), DispatchPolicy(budget=Budget(wall_seconds=1)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), validator=validator,
                   deterministic_stage=det_stage("unsupported"))
    assert not res.verified and res.ledger["scorable_outcomes"] == 0
    assert all(not r["scored"] for r in res.failure_receipts if r["stage"] == "agentic_validate")

    clock.mono = 1000.0          # fresh start for the second dispatch

    def late_ok(*a, **k):
        clock.mono = 1002.0
        return D.DeterministicOutcome("ok")
    res2 = dispatch(req(model, tmp_path / "b"), DispatchPolicy(budget=Budget(wall_seconds=1)),
                    provider=PureSkillProvider(), deterministic_stage=late_ok)
    assert res2.status == "deterministic_infrastructure_error" and res2.sol_source is None


def test_clock_rollback_is_a_discontinuity(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    clock.epoch = 90.0          # rolled back, monotonic barely moved
    clock.mono = 1000.5
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any("clock discontinuity" in r["reason"] for r in res.failure_receipts)


def test_retry_rehashes_and_refuses_changed_bytes(model, tmp_path):
    seen = []

    def validator(c, *a):
        seen.append(_sha(c.dsl_path))
        if len(seen) == 1:
            c.dsl_path.write_text(c.dsl_path.read_text() + "\n# mutated\n")
            return D.ValidationVerdict("infrastructure", FailureKind.INFRASTRUCTURE, "interrupted")
        return D.ValidationVerdict("verified", "", "VERIFIED", verify_status="pass")
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_validation_retries=1)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), validator=validator,
                   deterministic_stage=det_stage("unsupported"))
    assert not res.verified and len(seen) == 1
    rec = [r for r in res.failure_receipts if r.get("reason_code") == "candidate_bytes_changed"]
    assert rec and rec[0]["kind"] == FailureKind.PROTOCOL and not rec[0]["scored"]


def test_retry_budget_persists_across_resumes(model, tmp_path):
    calls = []

    def validator(*a):
        calls.append(1)
        return D.ValidationVerdict("infrastructure", FailureKind.INFRASTRUCTURE, "host")
    res = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(max_validation_retries=1)),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), validator=validator,
                   deterministic_stage=det_stage("unsupported"))
    for _ in range(3):
        res = resume(tmp_path / "out", PureSkillProvider(), validator=validator)
    assert len(calls) == 2                       # initial + the one budgeted retry, never more
    assert res.ledger["validation_retries_used"] == 1 <= res.ledger["max_validation_retries"]
    assert res.pending_validation and res.status == "agentic_infrastructure_error"
    assert any("retry budget exhausted" in r["reason"] for r in res.failure_receipts)


def test_receipt_writer_fault_is_durable_and_retains_candidate(model, tmp_path, definition, monkeypatch):
    import solar.translate.receipts as R
    # emission happens in the verify child now; the storage fault is injected at
    # the parent's admission step (the atomic rename of the staged receipt)
    def eio(*a, **k):
        raise OSError("EIO")
    monkeypatch.setattr(D, "_TRUSTED_REPLACE", eio)
    res = dispatch(req(model, tmp_path, definition_dir=definition, dtype="fp32"),
                   provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)), deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_infrastructure_error" and Path(res.receipt_path).is_file()
    assert res.pending_validation and res.pending_validation["candidate"]["dsl_sha256"]
    val = [r for r in res.failure_receipts if r.get("reason_code") == "receipt_emission_failed"]
    assert val and not val[0]["scored"]


def test_unexpected_dispatcher_exception_leaves_a_durable_receipt(model, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("validator plumbing exploded")
    res = dispatch(req(model, tmp_path), provider=PureSkillProvider(agent=fake_agent(GOOD_DSL)),
                   validator=boom, deterministic_stage=det_stage("unsupported"))
    assert res.status == "dispatcher_error" and Path(res.receipt_path).is_file()
    disk = json.loads(Path(res.receipt_path).read_text())
    assert disk["status"] == "dispatcher_error"
    assert not any(r["scored"] for r in res.failure_receipts if r["stage"].startswith("dispatcher"))


def test_remaining_wall_bounds_the_provider_verify_timeout(model, tmp_path):
    seen = []

    def translate(*a, **k):
        seen.append(k)
        return _mock_translate(GOOD_DSL)(*a, **k)
    # a fake validator: this test checks the forwarded bound, not the verifier
    # (and a real verification under a tight wall is timing-dependent)
    ok = lambda *a: D.ValidationVerdict("verified", "", "VERIFIED", verify_status="pass")
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       verify_timeout=10, budget=Budget(wall_seconds=5)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate), validator=ok,
                   reservation=LedgerReservation(1, 1.0), deterministic_stage=det_stage("unsupported"))
    assert res.verified and seen[0]["verify_timeout"] <= 5 < 10


# ---------------------------------------------------------------------------
# round-4 repairs (the review round 3): the dispatcher executes the trusted
# reference itself; primitive debits; finite costs or fail closed;
# annotations outside the request count; boot-id clock continuity; the
# narrow resume exemption
# ---------------------------------------------------------------------------

def _cuda_case(tmp_path, *, adapter_edit=None, dsl_edit=None, source_edit=None, extra_files=None):
    """A cuda-contract dispatch on the cuda_scale fixture: the ORIGINAL
    reference is written as its own file, the fake agent writes the adapter
    (optionally edited) and the DSL (optionally edited)."""
    import ast as _ast, shutil
    adapter_src = (CUDA_FX / "adapter.py").read_text()
    tree = _ast.parse(adapter_src)
    reference = next(_ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, _ast.Assign) and any(getattr(t, "id", None) == "_REFERENCE_SRC" for t in n.targets))
    if source_edit:
        new_reference = source_edit(reference)
        # the adapter's get_source() must still echo the original exactly
        adapter_src = adapter_src.replace(repr(reference), repr(new_reference)) \
            if repr(reference) in adapter_src else adapter_src
        adapter_src = adapter_src.replace(
            "_REFERENCE_SRC = (", "_REFERENCE_SRC = " + repr(new_reference) + "\n_UNUSED = (")
        reference = new_reference
    original = tmp_path / "reference.py"
    original.write_text(reference)
    defn = tmp_path / "definition"
    defn.mkdir()
    for name in ("definition.json", "workload.jsonl"):
        shutil.copy2(CUDA_FX / name, defn / name)
    dsl_src = (CUDA_FX / "solar_model_einsum_lib.py").read_text()

    def consume(path):
        rq = json.loads(Path(path).read_text())
        adir = Path(rq["adapter_path"]).parent
        for name, text in (extra_files or {}).items():
            (adir / name).write_text(text)
        Path(rq["adapter_path"]).write_text(adapter_edit(adapter_src) if adapter_edit else adapter_src)
        Path(rq["candidate_path"]).write_text(dsl_edit(dsl_src) if dsl_edit else dsl_src)
        resp = {"schema": RESPONSE_SCHEMA, "request_id": rq["request_id"], "source_sha256": rq["source_sha256"],
                "candidate_sha256": _sha(rq["candidate_path"]), "status": "candidate",
                "provenance": {"author": "fake", "model": "none"}}
        (Path(path).parent / "TRANSLATION_RESPONSE.json").write_text(json.dumps(resp))

    r = DispatchRequest(original, tmp_path / "out", source_type="cuda", definition_dir=defn, dtype="fp32",
                        tolerance={"rtol": 1e-4, "atol": 1e-5})
    return r, consume, original


def test_trusted_reference_is_executed_by_the_dispatcher_in_isolation(tmp_path):
    r, consume, original = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    assert tr["executed_source_sha256"] == _sha(original) == res.source_sha256
    assert tr["entry"] == "run" and tr["isolation"]["flags"] == ["-I", "-S"]
    adapter_dir = str(Path(res.candidate["model_path"]).resolve().parent)
    assert adapter_dir in tr["isolation"]["excluded_paths"]
    assert all(adapter_dir not in p for p in tr["report"]["sys_path"])
    assert tr["compared_outputs"] == ["output"] and tr["max_abs_error"] <= 1e-6
    assert res.validation["reference_binding"]["bound"] is True


def test_adapter_kernel_divergence_with_identical_get_source_is_caught(tmp_path):
    """the reviewer's REFERENCE-14 round-3 case: get_source() is byte-identical to
    the requested source but kernel_fn executes a modified copy (+2.0) and the
    DSL matches it.  The dispatcher's own trusted reference disagrees."""
    r, consume, _ = _cuda_case(
        tmp_path,
        adapter_edit=lambda a: a.replace("exec(get_source(), ns)",
                                         'exec(get_source().replace("return x * w + 1.0", "return x * w + 2.0"), ns)'),
        dsl_edit=lambda d: d.replace("X[b, h] * W[h] + 1.0", "X[b, h] * W[h] + 2.0"))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified and res.candidate is None
    assert res.sol_source["status"] != "ready" and res.sol_source["candidate"] is None
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["validation"]["status"] == "trusted_reference_mismatch"
    assert val["kind"] == FailureKind.UNSUPPORTED and val["scored"]
    tr = val["validation"]["trusted_reference"]
    assert tr["max_abs_error"] == pytest.approx(1.0)
    assert val["validation"]["reference_binding"]["bound"] is True     # necessary, not sufficient


def test_ambiguous_reference_entry_fails_closed_unless_declared(tmp_path):
    edit = lambda src: src + "\n\ndef helper(t):\n    return t\n"
    r, consume, _ = _cuda_case(tmp_path, source_edit=edit)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "reference_unexecutable" and val["kind"] == FailureKind.INFRASTRUCTURE
    assert not val["scored"] and res.pending_validation is not None       # candidate retained
    # the adapter may DECLARE the entry (metadata only): then the trusted
    # reference executes and the candidate verifies against it
    (tmp_path / "declared").mkdir()
    r2, consume2, _ = _cuda_case(tmp_path / "declared", source_edit=edit,
                                 adapter_edit=lambda a: a + '\nREFERENCE_ENTRY = "run"\n')
    res2 = dispatch(r2, provider=PureSkillProvider(agent=consume2))
    assert res2.status == "agentic_verified" and res2.validation["trusted_reference"]["entry"] == "run"


def test_reference_helper_smuggled_through_the_adapter_dir_is_not_importable(tmp_path):
    """An adversarial adapter puts a helper next to itself and extends
    sys.path so the verifier's in-process reference imports it; the
    dispatcher's isolated trusted reference cannot, and fails closed."""
    edit = lambda src: src.replace("    return x * w + 1.0",
                                   "    import adapter_helper\n    return x * w + adapter_helper.OFFSET")
    r, consume, _ = _cuda_case(
        tmp_path, source_edit=edit,
        adapter_edit=lambda a: a.replace("    import torch\n    ns = {}",
                                         "    import torch, sys, os\n    sys.path.insert(0, os.path.dirname(__file__))\n    ns = {}"),
        extra_files={"adapter_helper.py": "OFFSET = 1.0\n"})
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    # the trusted child never has the adapter directory on its path, so the
    # helper is simply unresolvable there (the adapter's sys.path trick only
    # affected the verifier's in-process run); an explicit run-time path
    # re-insertion is covered by the guard test below
    assert val["reason_code"] in ("reference_unexecutable", "reference_dependency_untrusted")
    assert not val["scored"]
    assert "adapter_helper" in val["validation"]["trusted_reference"]["report"]["error"]


def test_transport_budget_debits_are_primitive_and_serialization_is_total():
    from solar.einsum_lib.transport_budget import TransportBudget
    b = TransportBudget(permits=3, cost_per_request_usd=0.1, prepaid=1)
    b.charge("m")
    b.record_usage("m", {"total_tokens": "unavailable"})
    assert b.requests_used == 1 and b.total_tokens is None
    assert b.calls[-1]["usage"]["usage_unparsed"]
    b.record_usage("m", object())                      # never raises
    b.record_usage("m", {"total_tokens": -1})
    assert b.total_tokens is None and b.requests_used == 1
    b.annotate("overrun", preauthorized_usd=0.1, measured_usd=0.25)
    assert b.requests_used == 1 and len(b.annotations) == 1
    json.dumps(b.to_dict())                             # total
    b.calls.append({"n": 2, "usage": {"total_tokens": float("nan")}, "weird": object()})
    json.dumps(b.to_dict())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.25, "0.1", True])
def test_reported_cost_must_be_a_finite_non_negative_number(model, tmp_path, bad):
    def translate(*a, **k):
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = bad
        return r
    reservation = LedgerReservation(2, 1.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1, max_cost_usd=1.0)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert not res.verified
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["reason_code"] == "cost_malformed" and gen["kind"] == FailureKind.INFRASTRUCTURE
    assert not gen["scored"] and gen["cost_usd"] == pytest.approx(0.1)        # the bound, never a credit
    assert reservation.usd_used == pytest.approx(0.1) and res.ledger["cost_usd_used"] == pytest.approx(0.1)
    assert math.isfinite(reservation.usd_used)


def test_absent_usage_without_an_explicit_bound_is_cost_unbounded(model, tmp_path):
    reservation = LedgerReservation(2, 1.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", budget=Budget(max_attempts=1)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=_mock_translate(GOOD_DSL)),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert not res.verified
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["reason_code"] == "cost_unbounded" and gen["kind"] == FailureKind.INFRASTRUCTURE and not gen["scored"]
    # the whole remaining reservation was preauthorized and stays debited
    assert gen["cost_usd"] == pytest.approx(1.0) and reservation.usd_used == pytest.approx(1.0)


def test_explicit_bound_admits_unmeasured_usage_and_records_the_bound(model, tmp_path):
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1, max_cost_usd=1.0)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=_mock_translate(GOOD_DSL)),
                   reservation=LedgerReservation(2, 1.0), deterministic_stage=det_stage("unsupported"))
    assert res.status == "agentic_verified"
    cost = res.candidate["provenance"]["cost"]
    assert cost["bound_used_usd"] == 0.1 and cost["measured"] is False
    assert cost["bound_source"] == "policy.cost_per_request_usd" and res.candidate["cost_basis"] == "hard_cap_preauthorized"


def test_provider_declaring_cost_unavailable_under_a_finite_cap_is_refused_despite_the_bound(model, tmp_path):
    def translate(*a, **k):
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = None
        return r
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1, max_cost_usd=1.0)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=LedgerReservation(2, 1.0), deterministic_stage=det_stage("unsupported"))
    assert not res.verified
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["reason_code"] == "cost_unbounded"


def test_overrun_is_an_annotation_never_a_transport_request(model, tmp_path):
    from solar.einsum_lib.transport_budget import charge_transport

    def translate(*a, **k):
        charge_transport("m")                 # the one real transport request
        r = _mock_translate(GOOD_DSL)(*a, **k)
        r.cost_usd = 0.25
        return r
    reservation = LedgerReservation(2, 1.0)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["reason_code"] == "cost_cap_exceeded"
    tb = gen["transport_budget"]
    assert tb["requests_used"] == 1 and [a["kind"] for a in tb["annotations"]] == ["overrun"]
    assert res.ledger["api_requests_used"] == 1 == reservation.requests_used == gen["api_requests"]
    assert reservation.usd_used == pytest.approx(0.25) == res.ledger["cost_usd_used"]


def _cross_process(monkeypatch):
    realos = D.os

    class Proxy:
        def __getattr__(self, name):
            return getattr(realos, name)

        def getpid(self):
            return realos.getpid() + 100000
    monkeypatch.setattr(D, "os", Proxy())


def test_cross_process_resume_keeps_monotonic_continuity_under_the_same_boot(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    _cross_process(monkeypatch)
    # epoch says 3 s, monotonic says 6 s: continuity cannot be established
    clock.epoch, clock.mono = 103.0, 1006.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified and res.status == "agentic_infrastructure_error"
    assert any(r.get("reason_code") == "clock_continuity_unknown" for r in res.failure_receipts)
    assert all(not r["scored"] for r in res.failure_receipts if r["stage"] != "deterministic")


def test_cross_process_resume_with_consistent_clocks_proceeds(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=30)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    _cross_process(monkeypatch)
    clock.epoch, clock.mono = 103.0, 1003.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert res.status == "agentic_verified"
    state = json.loads((tmp_path / "out" / "DISPATCH_STATE.json").read_text())
    assert state["clock_continuity"] == "same_boot_monotonic" and state["started_boot_id"]


def test_cross_process_resume_after_a_reboot_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=30)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    _cross_process(monkeypatch)
    monkeypatch.setattr(D, "_boot_id", lambda: "00000000-0000-0000-0000-000000000000")
    clock.epoch, clock.mono = 103.0, 1003.0            # consistent, but another boot
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any(r.get("reason_code") == "clock_continuity_unknown" and "boot id" in r["reason"]
               for r in res.failure_receipts)


def test_nothing_to_resume_is_the_only_resume_exemption(model, tmp_path):
    done = dispatch(req(model, tmp_path / "done"), provider=PureSkillProvider(),
                    deterministic_stage=det_stage("ok"))
    assert done.status == "mechanical_ok"
    with pytest.raises(D.NothingToResume):
        resume(tmp_path / "done" / "out", PureSkillProvider())
    with pytest.raises(D.NothingToResume):
        resume(tmp_path / "missing", PureSkillProvider())
    assert issubclass(D.NothingToResume, ValueError)
    # a ValueError raised DURING validation on resume is operational: fresh
    # durable dispatcher_error, the candidate retained, never re-raised
    first = dispatch(req(model, tmp_path / "v"), DispatchPolicy(), provider=PureSkillProvider(),
                     deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")

    def fail(*a):
        raise ValueError("decoder failure inside validation")
    res = resume(tmp_path / "v" / "out", PureSkillProvider(), validator=fail)
    assert res.status == "dispatcher_error" and not res.verified
    disk = json.loads((tmp_path / "v" / "out" / "DISPATCH_RECEIPT.json").read_text())
    assert disk["status"] == "dispatcher_error"
    assert disk["failure_receipts"][-1]["reason_code"] == "dispatcher_exception"
    assert not disk["failure_receipts"][-1]["scored"]
    assert disk["attempts"][-1]["candidate"]["dsl_sha256"] == _sha(GOOD_DSL)


def test_ledger_reservation_rejects_non_finite_or_negative_amounts():
    res = LedgerReservation(2, 1.0)
    with pytest.raises(ValueError):
        res.consume(1, float("nan"))
    with pytest.raises(ValueError):
        res.reconcile(0.1, -0.2)
    with pytest.raises(ValueError):
        res.reconcile(0.1, float("inf"))
    assert res.usd_used == 0.0 and res.requests_used == 0


# ---------------------------------------------------------------------------
# round-5 repairs (the review round 4): proof integrity — frozen inputs, trusted
# tolerance, comparison inside the trusted children, import guard for the
# whole execution, unknown boot identity, reconciled cost coherence
# ---------------------------------------------------------------------------

def _diverge(a):
    return a.replace("exec(get_source(), ns)",
                     'exec(get_source().replace("return x * w + 1.0", "return x * w + 2.0"), ns)')


def _wrong_dsl(d):
    return d.replace("X[b, h] * W[h] + 1.0", "X[b, h] * W[h] + 2.0")


def _inputs_refused(mutated):
    """Round 9: the candidate's inputs are READ-ONLY mappings of the parent's 0400
    files, so an in-place write faults inside the candidate process (refused from
    the fault); a write that did reach the parent's file is 'changed in place'."""
    return any(("changed in place" in m) or ("read-only input" in m) for m in mutated)


def _private_root_of(tr):
    """Round 10: no public record names the private root's path; the test finds the
    dispatch's root under the agentic dir (the newest one) and checks its id against
    the opaque id the public record carries."""
    agentic = Path(tr["isolation"]["cwd"]).parents[1]
    roots = sorted(agentic.glob(".private-evidence-*"), key=lambda p: p.stat().st_mtime)
    assert roots, "no private evidence root under the agentic dir"
    root = roots[-1]
    state = json.loads((root / "PRIVATE_STATE.json").read_text())
    assert tr["isolation"]["private_evidence_id"] == state["id"] == tr["private_evidence_id"]
    return root


def _phase_dir(tr, ph):
    """A phase directory as a path: the public record names private phase dirs by the
    opaque token <private-evidence:ID>/...; the test resolves it through the root it finds."""
    p = tr["phases"][ph]["phase_dir"]
    if p.startswith("<private-evidence:"):
        p = str(_private_root_of(tr)) + p.split(">", 1)[1]
    return Path(p)


def _private_verdict(tr):
    root = _private_root_of(tr)
    files = sorted(root.glob("validation-*/verdict_private.json"), key=lambda p: p.stat().st_mtime)
    assert files, "no private verdict under the private root"
    return json.loads(files[-1].read_text())


def test_trusted_inputs_are_frozen_before_any_candidate_code_runs(tmp_path):
    """the review INPUT-21: a wrong DSL that mutates inputs["x"] to compensate."""
    def dsl(d):
        return _wrong_dsl(d).replace(
            'return {"output": model(x, w).data}',
            'result = model(x, w).data\n    inputs["x"][:] += 1.0 / inputs["w"]\n    return {"output": result}')
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=dsl)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_mutated_inputs" and val["kind"] == FailureKind.UNSUPPORTED
    tr = val["validation"]["trusted_reference"]
    # the snapshot was hashed BEFORE the candidate ran and is unchanged on disk
    inputs = Path(tr["isolation"]["cwd"]) / "inputs.npz"
    assert tr["inputs_sha256"] == _sha(inputs)
    assert _inputs_refused(tr["phases"]["candidate"]["mutated_inputs"])


def test_trusted_tolerance_ignores_adapter_side_dtype_floor(tmp_path):
    """the review TOLERANCE-22: an adapter casting its output to fp16 cannot
    widen the approved gate; the widening is recorded and the wrong DSL fails."""
    def edit(a):
        a = a.replace("exec(get_source(), ns)",
                      'exec(get_source().replace("return x * w + 1.0", "return (x * w + 1.0) * 1.001"), ns)')
        return a.replace(".numpy()}", ".numpy().astype(np.float16)}")
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=edit,
                               dsl_edit=lambda d: d.replace("X[b, h] * W[h] + 1.0", "(X[b, h] * W[h] + 1.0) * 1.001"))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_reference_mismatch"
    tr = val["validation"]["trusted_reference"]
    assert tr["tolerance_widening_rejected"] is True and tr["adapter_side_min_tol"] > 1e-3
    assert tr["tolerance"]["trusted_dtype_floor"] < 1e-6          # derived from the fp32 trusted outputs
    assert tr["tolerance"]["rtol_effective"] == pytest.approx(1e-4)


def test_parent_comparator_monkeypatch_cannot_fake_the_trusted_pass(tmp_path):
    """the review ISOLATION-23(a): the adapter replaces the parent's comparator;
    the comparison runs in a child that never executed generated code."""
    import solar.einsum_lib.verify as V
    original = V.compare_outputs
    edit = lambda a: _diverge(a) + (
        "\nimport solar.einsum_lib.verify as _v\nfrom types import SimpleNamespace as _NS\n"
        "_v.compare_outputs=lambda *args,**kwargs:_NS(success=True,max_abs_error=0.0,vacuous_outputs=(),error=None)\n")
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=edit, dsl_edit=_wrong_dsl)
    try:
        res = dispatch(r, provider=PureSkillProvider(agent=consume))
    finally:
        V.compare_outputs = original
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_reference_mismatch"
    tr = val["validation"]["trusted_reference"]
    assert tr["max_abs_error"] == pytest.approx(1.0)
    # the verdict is digest-bound to the persisted arrays, which the parent re-hashed
    cwd = Path(tr["isolation"]["cwd"])
    # round 9: no digest of the reference bytes is public; the parent's PRIVATE record
    # binds the arrays it re-hashed (round 10: no alias of the reference in the artifact tree)
    assert tr["verdict_sha256"] and tr["trusted_outputs_sha256"] is None
    priv = _private_verdict(tr)
    assert priv["artifact_sha256"]["trusted_outputs"] == _sha(priv["artifacts"]["trusted_outputs"])
    assert not (cwd / "trusted_outputs.npz").exists() and not (cwd / "trusted_outputs.npz").is_symlink()
    assert priv["artifact_sha256"]["candidate_outputs"] == tr["candidate_outputs_sha256"] == _sha(cwd / "candidate_outputs.npz")


def test_reference_import_side_effect_is_refused_during_execution(tmp_path):
    """the review ISOLATION-23(b): the reference re-adds the excluded adapter dir
    at run time and imports a generated helper; the guard refuses the import."""
    def source_edit(s):
        return s.replace("    return x * w + 1.0",
                         "    import sys\n    sys.path.insert(0, __import__('os').path.dirname(__ADIR__))\n"
                         "    import injected_helper\n    return x * w + injected_helper.OFFSET")
    r, consume, _ = _cuda_case(tmp_path, source_edit=lambda s: source_edit(s).replace(
        "__ADIR__", repr(str(tmp_path / "out" / "agentic" / "attempt_01" / "adapter.py"))),
        dsl_edit=_wrong_dsl, extra_files={"injected_helper.py": "OFFSET = 2.0\n"})
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "reference_dependency_untrusted" and not val["scored"]
    assert "injected_helper" in val["validation"]["trusted_reference"]["report"]["error"]


def test_trusted_verdict_records_every_hash_and_the_guarded_roots(tmp_path):
    r, consume, original = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    cwd = Path(tr["isolation"]["cwd"])
    assert tr["inputs_sha256"] == _sha(cwd / "inputs.npz")
    assert tr["trusted_outputs_sha256"] is None            # private (round 9)
    priv = _private_verdict(tr)
    assert priv["artifact_sha256"]["trusted_outputs"] == _sha(priv["artifacts"]["trusted_outputs"])
    assert tr["candidate_outputs_sha256"] == _sha(cwd / "candidate_outputs.npz")
    assert tr["verdict_sha256"] and tr["executed_source_sha256"] == _sha(original)
    assert tr["isolation"]["phases"] == ["inputs", "candidate", "boundary", "reference", "compare"]
    # round 7: the reference phase directory is created only AFTER the candidate
    # session boundary, and every phase ran from the installed program path
    phases = tr["phases"]
    assert all(Path(rec["argv"][3]) == D._TRUSTED_CHILD_PATH for rec in phases.values())
    assert not list(cwd.rglob("phase_*/trusted_child.py"))
    assert all(phases[ph]["session_boundary"]["clean"] for ph in ("inputs", "candidate", "reference", "compare"))
    assert tr["input_audit"]["mutated_inputs"] == [] and tr["input_audit"]["inputs_after_available"]
    adapter_dir = str(Path(res.candidate["model_path"]).resolve().parent)
    assert adapter_dir not in tr["isolation"]["trusted_roots"]
    assert all(not p.startswith(adapter_dir) for p in tr["report"]["sys_path"])
    assert tr["tolerance_widening_rejected"] is False


def test_unknown_boot_identity_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    monkeypatch.setattr(D, "_boot_id", lambda: None)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    _cross_process(monkeypatch)
    clock.epoch, clock.mono = 103.0, 1006.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified and res.status == "agentic_infrastructure_error"
    assert any(r.get("reason_code") == "clock_continuity_unknown" and "boot identity unavailable" in r["reason"]
               for r in res.failure_receipts)


def test_post_reconciliation_fault_keeps_the_measured_cost_in_both_ledgers(model, tmp_path):
    from solar.einsum_lib.transport_budget import charge_transport

    class Result:
        cost_usd = 0.03
        verification_status = "verified"

        @property
        def exit_code(self):
            raise ValueError("decoder")

    def translate(*a, **k):
        charge_transport("m")
        _mock_translate(GOOD_DSL)(*a, **k)
        return Result()
    reservation = LedgerReservation(1, 0.1)
    res = dispatch(req(model, tmp_path), DispatchPolicy(route="pythonic", cost_per_request_usd=0.1,
                                                       budget=Budget(max_attempts=1, max_requests=1, max_cost_usd=0.1)),
                   provider=PythonicProvider(api_key="explicit", translate_fn=translate),
                   reservation=reservation, deterministic_stage=det_stage("unsupported"))
    assert not res.verified
    gen = [x for x in res.failure_receipts if x["stage"] == "agentic_generate"][0]
    assert gen["kind"] == FailureKind.PROTOCOL and gen["cost_usd"] == pytest.approx(0.03)
    assert gen["cost_basis"] == "reported"
    assert res.ledger["cost_usd_used"] == pytest.approx(0.03) == pytest.approx(reservation.usd_used)
    assert res.ledger["api_requests_used"] == 1 == reservation.requests_used


# ---------------------------------------------------------------------------
# round-6 repairs (the review round 5): the comparator is a read-only program
# outside the candidate's writable boundary; no generated code runs in the
# parent; input audit by identity and content; synthetic module audit;
# process-incarnation clock identity
# ---------------------------------------------------------------------------

def _candidate_body(d, payload):
    return d.replace('return {"output": model(x, w).data}',
                     'result = model(x, w).data\n' + ''.join('    ' + l + '\n' for l in payload.splitlines())
                     + '    return {"output": result}')


def test_no_generated_code_executes_in_the_dispatcher_process(tmp_path, monkeypatch):
    """The facade verifier is monkeypatched to RAISE in this process: a dispatch
    still verifies, because verification (and the adapter) run in children."""
    import solar.translate as T

    def boom(*a, **k):
        raise AssertionError("verifier executed in the dispatcher process")
    monkeypatch.setattr(T, "verify", boom)
    r, consume, _ = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    assert tr["isolation"]["flags"] == ["-I", "-S"] and tr["isolation"]["per_phase_dirs"]
    prog = tr["isolation"]["program"]
    assert prog["sha256"] == D._TRUSTED_PROGRAM_SHA256 and prog["verified_before_each_phase"]
    assert prog["verified_after_each_phase"] and prog["executed_from_installed_path"]
    assert prog["child_self_report_trusted"] is False
    assert Path(prog["path"]).name == "_trusted_child.py"
    for ph in ("inputs", "reference", "candidate", "compare"):
        rec = tr["phases"][ph]
        assert rec["program_sha256_at_start"] == rec["program_sha256_at_end"] == D._TRUSTED_PROGRAM_SHA256
        assert rec["program_sha256_parent"] == D._TRUSTED_PROGRAM_SHA256 and rec["program_check_after_phase"] is None
        assert rec["flags"] == {"isolated": True, "no_site": True, "ignore_environment": True}
        if ph == "candidate":
            # round 9: the candidate runs in its own per-launch working root
            # (mkdtemp "candidate-*", 0700) directly under the trusted_reference dir.
            assert Path(rec["phase_dir"]).name.startswith("candidate-")
            assert Path(rec["phase_dir"]).parent == Path(tr["isolation"]["cwd"])
        else:
            assert Path(rec["phase_dir"]).name.startswith(f"phase_{ph}")


def test_in_memory_program_edit_changes_nothing_the_children_run(tmp_path, monkeypatch):
    """the reviewer's parent_script attack: the adapter edits the in-memory program
    text.  The children execute hash-verified copies of the installed module."""
    patch = ("    import sys as _sys\n    _d = _sys.modules.get('solar.frontend.dispatcher')\n"
             "    if _d is not None:\n        _d._TRUSTED_CHILD = _d._TRUSTED_CHILD.replace('success = error is None', 'success = True')\n")
    old = D._TRUSTED_CHILD
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=lambda a: _diverge(a).replace("    import torch\n", patch + "    import torch\n"),
                               dsl_edit=_wrong_dsl)
    try:
        res = dispatch(r, provider=PureSkillProvider(agent=consume))
    finally:
        D._TRUSTED_CHILD = old
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_reference_mismatch"
    assert val["validation"]["trusted_reference"]["max_abs_error"] == pytest.approx(1.0)


def test_candidate_rewriting_the_installed_program_is_refused_unscored_by_the_parent(tmp_path):
    """the reviewer's candidate_script attack under round 7: there is no per-phase copy,
    so the candidate rewrites the INSTALLED comparator it is running from.  The
    parent re-hashes the installed program after the phase, refuses the dispatch
    as trusted_program_tampered (unscored, retained) and admits nothing from the
    phase.  The test restores the installed bytes afterwards (a tampered
    installation is, correctly, refused by every later validation)."""
    installed = D._TRUSTED_CHILD_PATH
    original, mode = installed.read_bytes(), installed.stat().st_mode
    payload = ("import sys as _sys\nfrom pathlib import Path as _Path\n_f = _sys._getframe()\n"
               "while _f is not None:\n    if _f.f_globals.get('PHASE') == 'candidate' and 'META' in _f.f_globals:\n"
               "        _p = _Path(_f.f_globals['__file__'])\n        import os as _os\n        _os.chmod(_p, 0o644)\n"
               "        _p.write_text(_p.read_text().replace('success = error is None', 'success = True'))\n        break\n"
               "    _f = _f.f_back")
    try:
        r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=lambda d: _candidate_body(_wrong_dsl(d), payload))
        res = dispatch(r, provider=PureSkillProvider(agent=consume))
        assert not res.verified
        val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
        assert val["reason_code"] == "trusted_program_tampered" and not val["scored"]
        assert "re-hashed by the parent after the candidate phase" in val["reason"]
        assert res.pending_validation is not None            # retained, unscored
        assert installed.read_bytes() != original             # the attack did reach the installed file ...
        assert D._trusted_program_check()                      # ... and the parent sees it until it is repaired
    finally:
        os.chmod(installed, mode | 0o200)
        installed.write_bytes(original)
        os.chmod(installed, mode)
    assert D._trusted_program_check() is None


def test_program_hash_mismatch_fails_closed_before_any_phase(tmp_path, monkeypatch):
    monkeypatch.setitem(D._TRUSTED_PROGRAM_HASHES, "solar.frontend._trusted_child", "0" * 64)
    assert D._trusted_program_check()
    r, consume, _ = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_program_tampered" and not val["scored"]
    assert res.pending_validation is not None            # retained, unscored


@pytest.mark.parametrize("mode", ["replace", "snapshot_module", "inplace"])
def test_input_replacement_and_patched_numpy_are_refused(tmp_path, mode):
    payload = {"inplace": 'inputs["x"][:] += 1.0',
               "replace": 'inputs["x"] = inputs["x"] + 1.0',
               "snapshot_module": 'import numpy as _np\n_np.array_equal = lambda *a, **k: True\ninputs["x"][:] += 1.0'}[mode]
    phase = ("import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get('PHASE') == 'candidate':\n"
             + ''.join('        ' + l + '\n' for l in payload.splitlines()) + "        break\n    _f = _f.f_back")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, phase))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified and Path(res.receipt_path).is_file()
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_mutated_inputs"
    mutated = val["validation"]["trusted_reference"]["phases"]["candidate"]["mutated_inputs"]
    assert any(m.startswith("x: replaced") for m in mutated) if mode == "replace" else _inputs_refused(mutated)


def test_synthetic_module_registration_is_reported(tmp_path):
    payload = 'import sys as _sys, types as _types\n_sys.modules["planted_synthetic"] = _types.ModuleType("planted_synthetic")'
    phase = ("import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get('PHASE') == 'candidate':\n"
             + ''.join('        ' + l + '\n' for l in payload.splitlines()) + "        break\n    _f = _f.f_back")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, phase))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "reference_dependency_untrusted"
    assert "planted_synthetic" in val["reason"]


def test_unadmitted_staged_receipt_is_discarded(tmp_path):
    r, consume, _ = _cuda_case(tmp_path, adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    work = Path(res.attempts[0]["request_dir"]) / "trusted_reference"
    assert not (work / "receipt.staged.json").exists()
    assert not (Path(res.attempts[0]["request_dir"]) / "receipt.json").exists()


def test_reused_pid_after_reboot_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    monkeypatch.setattr(D, "_boot_id", lambda: "old-boot")
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    # same integer pid, new boot: the monotonic clock restarted (2 s) while 3 s of
    # epoch passed; an integer pid is not process identity
    monkeypatch.setattr(D, "_boot_id", lambda: "new-boot")
    clock.epoch, clock.mono = 103.0, 2.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any(r.get("reason_code") == "clock_continuity_unknown" for r in res.failure_receipts)


def test_same_pid_missing_monotonic_start_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    state = tmp_path / "out" / "DISPATCH_STATE.json"
    saved = json.loads(state.read_text())
    saved["started_monotonic"] = None
    state.write_text(json.dumps(saved))
    clock.epoch, clock.mono = 103.0, 1006.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any(r.get("reason_code") == "clock_continuity_unknown" and "monotonic start" in r["reason"]
               for r in res.failure_receipts)


def test_process_identity_includes_the_kernel_start_time():
    ident = D._process_identity()
    assert ident["pid"] == D.os.getpid() and ident["boot_id"] and isinstance(ident["starttime"], int)
    assert D._proc_starttime(ident["pid"]) == ident["starttime"]


# ---------------------------------------------------------------------------
# CI-FAIL-25 (round 7 prep): trusted roots are derived from the LIVE
# interpreter — a `pip install --user` runner keeps every package in the user
# site, and the trusted children must still import them
# ---------------------------------------------------------------------------

def _user_site_layout(tmp_path, monkeypatch):
    """Every installed package importable ONLY through a user-site-like directory
    (symlinked package dirs + dist-info); the interpreter's real site-packages is
    absent from the configured site dirs and from sys.path."""
    import site
    import sysconfig
    import sys as _sys
    real = Path(sysconfig.get_paths()["purelib"])
    usersite = tmp_path / "userbase" / "lib" / "python3.10" / "site-packages"
    usersite.mkdir(parents=True)
    for p in real.iterdir():
        if p.name.endswith((".dist-info", ".egg-info")) or (p.is_dir() and not p.name.startswith("_")) or p.suffix == ".py":
            try:
                (usersite / p.name).symlink_to(p)
            except FileExistsError:
                pass
    paths = sysconfig.get_paths()
    monkeypatch.setattr(site, "getsitepackages", lambda *a, **k: [])
    monkeypatch.setattr(site, "getusersitepackages", lambda *a, **k: str(usersite))
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {**paths, "purelib": str(usersite), "platlib": str(usersite)})
    # a real --user layout has the user site ON sys.path and the absent real site off it
    monkeypatch.setattr(_sys, "path", [str(usersite)] + [p for p in _sys.path if str(real) not in p])
    return real, usersite


def test_trusted_roots_follow_the_live_interpreter_not_pythonpath_or_cwd(tmp_path, monkeypatch):
    """Expectations derived from the LIVE interpreter, never from a distro path
    (the GitLab runner's configured dist-packages dir is not an active site
    dir there, CI-FAIL-25 addendum): every existing sysconfig/site dir that is
    actually on sys.path is a root; a configured dir that is not on sys.path
    is not; origins of loaded dependencies and the checkout are roots;
    PYTHONPATH and cwd never are."""
    import numpy
    import site
    import sysconfig
    paths = sysconfig.get_paths()
    on_path = {str(Path(p or ".").resolve()) for p in sys.path}
    configured = [paths.get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")] + list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        configured.append(site.getusersitepackages())
    live = {str(Path(d).resolve()) for d in configured if d and Path(d).is_dir() and str(Path(d).resolve()) in on_path}
    roots0 = D._trusted_roots()
    assert live and live <= set(roots0)
    inactive = tmp_path / "inactive-dist-packages"
    inactive.mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {**paths, "purelib": str(inactive)})
    assert str(inactive.resolve()) not in D._trusted_roots()  # configured but not on sys.path: not a root
    monkeypatch.undo()
    real, usersite = _user_site_layout(tmp_path, monkeypatch)
    roots = D._trusted_roots()
    assert str(usersite.resolve()) in roots                  # the configured, active user site
    assert str(Path(numpy.__file__).resolve().parent.parent) in roots   # the ORIGIN numpy was actually loaded from
    assert str(D._CHECKOUT_ROOT) in roots
    stray = tmp_path / "stray_pythonpath_dir"
    stray.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(stray))
    monkeypatch.chdir(stray)
    roots2 = D._trusted_roots()
    assert str(stray.resolve()) not in roots2 and str(tmp_path.resolve()) not in roots2


def test_packages_only_in_a_user_site_still_verify(tmp_path, monkeypatch):
    """The GitLab runner layout (CI-FAIL-25): with the round-6 derivation the
    trusted children could not import the environment and every adapter-route
    validation refused; the origin-based derivation verifies."""
    _user_site_layout(tmp_path, monkeypatch)
    (tmp_path / "case").mkdir()
    r, consume, _ = _cuda_case(tmp_path / "case")
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified", [x.get("reason_code") for x in res.failure_receipts]
    tr = res.validation["trusted_reference"]
    assert tr["isolation"]["flags"] == ["-I", "-S"]
    assert any("userbase" in p for p in tr["isolation"]["trusted_roots"])


# ---------------------------------------------------------------------------
# Round 7 (the review an earlier review): hard phase boundary, reference computed
# AFTER the candidate, comparator executed from the installed path and hashed by
# the parent around every phase, frozen state parent-only, one trusted-roots
# computation, complete process incarnation for resume.
# ---------------------------------------------------------------------------

def _phase_payload(payload):
    return ("import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get('PHASE') == 'candidate':\n"
            + ''.join('        ' + l + '\n' for l in payload.splitlines()) + "        break\n    _f = _f.f_back")


def test_candidate_descendant_outliving_the_phase_is_refused(tmp_path):
    """A watcher forked by the candidate that escapes the session (setsid) is found
    by the boundary (child subreaper), killed, and the dispatch is refused before
    any reference is computed."""
    marker = tmp_path / "sleeper.pid"
    payload = ("import os as _os, time as _time\n"
               "_pid = _os.fork()\n"
               "if _pid == 0:\n"
               "    _os.setsid()\n"
               "    _os.close(1); _os.close(2)\n"
               f"    open({str(marker)!r}, 'w').write(str(_os.getpid()))\n"
               "    _time.sleep(120)\n"
               "    _os._exit(0)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, _phase_payload(payload)))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_descendants_alive" and not val["scored"]
    tr = val["validation"]["trusted_reference"]
    sb = tr["session_boundaries"]["candidate"]
    assert not sb["clean"] and sb["found_after_exit"] and not sb["alive_after_kill"]
    assert "reference" not in tr["phases"] and "compare" not in tr["phases"]
    pid = int(marker.read_text())
    for _ in range(100):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(0.05)
    assert not Path(f"/proc/{pid}").exists()


def test_no_reference_output_exists_while_the_candidate_runs(tmp_path):
    payload = ("from pathlib import Path as _P\nimport json as _json\n"
               "_work = _P(_f.f_globals['META']['phase_dir']).parent\n"
               "_found = sorted(str(p) for p in _work.rglob('trusted_outputs.npz')) + sorted(str(p) for p in _work.glob('phase_reference*'))\n"
               "(_P(_f.f_globals['META']['phase_dir']) / 'probe.json').write_text(_json.dumps(_found))\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, _phase_payload(payload)))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    assert json.loads((_phase_dir(tr, "candidate") / "probe.json").read_text()) == []
    assert tr["isolation"]["phases"] == ["inputs", "candidate", "boundary", "reference", "compare"]
    cand_end = (_phase_dir(tr, "candidate") / "report.json").stat().st_mtime
    ref_dir = _phase_dir(tr, "reference")
    assert ref_dir.stat().st_mtime >= cand_end and (ref_dir.stat().st_mode & 0o777) == 0o700
    assert all(tr["session_boundaries"][ph]["clean"] for ph in ("inputs", "candidate", "reference", "compare"))


def test_parent_rehashes_the_installed_program_after_each_phase(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = D._trusted_program_check

    def flaky():
        calls["n"] += 1
        return real() if calls["n"] < 4 else "solar.frontend._trusted_child changed (simulated after-phase mutation)"
    monkeypatch.setattr(D, "_trusted_program_check", flaky)
    r, consume, _ = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_program_tampered" and "re-hashed by the parent after" in val["reason"]


def test_runner_consumes_the_parents_trusted_roots(tmp_path, monkeypatch):
    import solar.translate._runner as R
    roots = [str(tmp_path / "siteA"), sys.prefix]
    digest = hashlib.sha256(json.dumps(roots, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    monkeypatch.setattr(R, "_TRUSTED_ROOTS_IN_USE", None)
    with pytest.raises(ValueError):
        R.use_trusted_roots(roots, "0" * 64)
    assert R.trusted_roots_in_use() is None
    R.use_trusted_roots(roots, digest)
    assert R._safe_roots() == [Path(p).resolve() for p in roots] == R.trusted_roots_in_use()
    monkeypatch.setattr(R, "_TRUSTED_ROOTS_IN_USE", None)
    assert R._safe_roots() != [Path(p).resolve() for p in roots]
    # end to end: one computation in the parent, hashed into every phase's metadata
    (tmp_path / "case").mkdir()
    r, consume, _ = _cuda_case(tmp_path / "case")
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    want = hashlib.sha256(json.dumps(tr["isolation"]["trusted_roots"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert tr["isolation"]["trusted_roots_sha256"] == want
    for ph in ("verify", "inputs", "candidate", "reference", "compare"):
        meta = json.loads((_phase_dir(tr, ph) / "meta.json").read_text())
        assert meta["trusted_roots"] == tr["isolation"]["trusted_roots"] and meta["trusted_roots_sha256"] == want


def test_user_site_only_dependency_verifies_under_a_real_pythonuserbase(tmp_path):
    """CI-FAIL-25 end to end with NO site/sysconfig monkeypatch: a package importable
    only through an enabled PYTHONUSERBASE is imported by the reference; the parent
    derives the roots once and the verify child's runner consumes them."""
    base = tmp_path / "userbase"
    site_dir = base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site_dir / "userbase_only_dep").mkdir(parents=True)
    (site_dir / "userbase_only_dep" / "__init__.py").write_text("VALUE = 19\n")
    (tmp_path / "case").mkdir()
    script = textwrap.dedent(f"""
        import json, pathlib, sys, site
        sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
        import userbase_only_dep
        assert site.ENABLE_USER_SITE
        assert pathlib.Path(userbase_only_dep.__file__).resolve().is_relative_to(pathlib.Path(site.getusersitepackages()).resolve())
        import test_fallback_dispatcher as T
        from solar.frontend.dispatcher import dispatch, PureSkillProvider
        edit = lambda s: "import userbase_only_dep\\nassert userbase_only_dep.VALUE == 19\\n" + s
        r, consume, _ = T._cuda_case(pathlib.Path({str(tmp_path / "case")!r}), source_edit=edit)
        res = dispatch(r, provider=PureSkillProvider(agent=consume))
        tr = (res.validation or {{}}).get("trusted_reference") or {{}}
        print("RESULT " + json.dumps({{"status": res.status, "codes": [x.get("reason_code") for x in res.failure_receipts],
                                      "roots": (tr.get("isolation") or {{}}).get("trusted_roots")}}))
    """)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONNOUSERSITE")}
    env.update(PYTHONUSERBASE=str(base), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    # On a runner whose ONLY package location is the real user site (pip install --user,
    # CI-FAIL-25), overriding PYTHONUSERBASE would hide torch/numpy from the subprocess and
    # the test would fail for a reason unrelated to the dispatcher (round-9 CI diagnosis).
    # The real user site therefore stays reachable through PYTHONPATH: the trusted roots
    # admit it as an installed site directory / dependency origin, never as PYTHONPATH.
    import site as _site
    real_user_site = _site.getusersitepackages()
    if isinstance(real_user_site, str) and Path(real_user_site).is_dir():
        env["PYTHONPATH"] = real_user_site
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=900, cwd=str(SOLAR))
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads([l for l in out.stdout.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])
    assert got["status"] == "agentic_verified", got
    assert str(site_dir.resolve()) in got["roots"]


def test_same_process_monotonic_rollback_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    clock.epoch, clock.mono = 103.0, 2.0     # same process, same boot: the monotonic reading went BACKWARDS
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any(r.get("reason_code") == "clock_continuity_unknown" and "before the persisted start" in r["reason"]
               for r in res.failure_receipts)


def test_incomplete_process_identity_is_continuity_unknown(model, tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(D, "time", clock)
    first = dispatch(req(model, tmp_path), DispatchPolicy(budget=Budget(wall_seconds=5)),
                     provider=PureSkillProvider(), deterministic_stage=det_stage("unsupported"))
    fake_agent(GOOD_DSL)(Path(first.resume_token) / "TRANSLATION_REQUEST.json")
    monkeypatch.setattr(D, "_proc_starttime", lambda pid=None: None)
    clock.epoch, clock.mono = 103.0, 1003.0
    res = resume(tmp_path / "out", PureSkillProvider())
    assert not res.verified
    assert any(r.get("reason_code") == "clock_continuity_unknown" and "process identity incomplete" in r["reason"]
               for r in res.failure_receipts)


def test_mutation_audit_state_is_parent_only(tmp_path):
    """No frame reachable from the candidate carries the audit state; an in-place
    change is caught by the PARENT comparing the serialised inputs with its frozen
    hashes."""
    payload = ("_names = set()\n_g2 = _f\nwhile _g2 is not None:\n"
               "    _names |= {k for k in _g2.f_globals if k in ('frozen_bytes', 'orig_objs', '_tobytes')}\n    _g2 = _g2.f_back\n"
               "from pathlib import Path as _P\nimport json as _json\n"
               "(_P(_f.f_globals['META']['phase_dir']) / 'globals_probe.json').write_text(_json.dumps(sorted(_names)))\n"
               "inputs['x'][:] += 1.0\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, _phase_payload(payload)))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_mutated_inputs"
    tr = val["validation"]["trusted_reference"]
    assert json.loads((_phase_dir(tr, "candidate") / "globals_probe.json").read_text()) == []
    assert _inputs_refused(tr["input_audit"]["mutated_inputs"]) and tr["input_audit"]["inputs_after_available"]
    assert _inputs_refused(tr["phases"]["candidate"]["mutated_inputs"])


def test_comparator_runs_from_the_installed_path_without_copies(tmp_path):
    r, consume, _ = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    work = Path(tr["isolation"]["cwd"])
    assert not list(work.rglob("phase_*/trusted_child.py"))
    for ph, rec in tr["phases"].items():
        assert Path(rec["argv"][3]) == D._TRUSTED_CHILD_PATH and rec["argv"][1:3] == ["-I", "-S"]
        assert rec["program_sha256_parent"] == D._TRUSTED_PROGRAM_SHA256 and rec["program_check_after_phase"] is None
        assert rec["self_reported_hash_trusted"] is False and rec["session_boundary"]["clean"]
        assert (_phase_dir(tr, ph).stat().st_mode & 0o777) == 0o700


# ---------------------------------------------------------------------------
# Round 9 (the review an earlier review, manager design ruling): one candidate
# process per launch on read-only mappings, completion established by the parent,
# reference material only in a parent-private evidence root locked while generated
# code runs.
# ---------------------------------------------------------------------------

def _agentic_dir_payload(body):
    """Candidate-frame payload with ``_agentic`` bound to the dispatch's agentic dir
    (the candidate's working root is attempt/trusted_reference/candidate-*)."""
    return _phase_payload("from pathlib import Path as _P\nimport json as _json, os as _os, stat as _stat\n"
                          "_croot = _P(_f.f_globals['META']['phase_dir'])\n_agentic = _croot.parents[2]\n" + body)


def test_candidate_early_exit_without_result_is_candidate_result_missing(tmp_path):
    """Completion is the PARENT's determination: a candidate process that ends before
    writing its result is refused and scored, whatever it did before exiting."""
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, _phase_payload("import os as _os\n_os._exit(0)\n")))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_result_missing" and val["kind"] == FailureKind.UNSUPPORTED and val["scored"]
    tr = val["validation"]["trusted_reference"]
    assert tr["phases"]["candidate"]["returncode"] == 0 and tr["phases"]["candidate"]["stdin"].startswith("devnull")
    assert "reference" not in tr["phases"] and "compare" not in tr["phases"]


def test_private_evidence_root_is_locked_while_the_candidate_runs(tmp_path):
    payload = _agentic_dir_payload(
        "_probe = []\n"
        "for _d in sorted(_agentic.glob('.private-evidence-*')):\n"
        "    _rec = {'mode': oct(_d.stat().st_mode & 0o777)}\n"
        "    try:\n        _os.listdir(_d); _rec['listed'] = True\n"
        "    except OSError as _e:\n        _rec['listed'] = False; _rec['error'] = type(_e).__name__\n"
        "    _probe.append(_rec)\n"
        "(_croot / 'lock_probe.json').write_text(_json.dumps(_probe))\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    probe = json.loads((_phase_dir(tr, "candidate") / "lock_probe.json").read_text())
    assert len(probe) == 1 and probe[0] == {"mode": "0o0", "listed": False, "error": "PermissionError"}
    root = _private_root_of(tr)
    assert (root.stat().st_mode & 0o777) == 0o700                     # unlocked again after the boundary
    assert tr["phases"]["candidate"]["private_root_locked"] and tr["phases"]["verify"]["private_root_locked"]
    assert tr["phases"]["inputs"]["private_root_locked"] and not tr["phases"]["compare"]["private_root_locked"]


def test_reference_material_lives_only_in_the_private_root(tmp_path):
    r, consume, _ = _cuda_case(tmp_path)
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    root = _private_root_of(tr)
    priv = _private_verdict(tr)
    private = Path(priv["artifacts"]["trusted_outputs"]).parent
    assert private.parent == root
    ref_sha = priv["artifact_sha256"]["trusted_outputs"]
    # no regular file outside the private root hashes to the reference (the
    # candidate's OWN outputs legitimately do on an honest, verified dispatch), and
    # (round 10) there is no entry of the canonical name outside it at all: no alias
    attempt = Path(res.attempts[0]["request_dir"])
    for p in attempt.rglob("*"):
        if root in p.parents or p.is_symlink() or not p.is_file() or p.name == "candidate_outputs.npz":
            continue
        assert _sha(p) != ref_sha, p
    assert not [p for p in (tmp_path / "out").rglob("trusted_outputs.npz") if root not in p.parents]
    assert tr["trusted_outputs_sha256"] is None and tr["verdict_sha256"] != priv["verdict_sha256"]
    assert str(root) not in json.dumps(res.to_dict())
    assert priv["verdict_sha256"] not in json.dumps(res.to_dict())
    assert list((tmp_path / "out").rglob("report_compare.json")) == [private / "report_compare.json"]
    assert (private / "trusted_child.py").read_text() == D._TRUSTED_CHILD
    # the reference copy is a fresh exclusively created regular file with one link
    ref = Path(priv["artifacts"]["trusted_outputs"])
    assert ref.parent == private and ref.name.startswith("reference-") and ref.stat().st_nlink == 1
    assert (ref.stat().st_mode & 0o777) == 0o600 and (private / "trusted_outputs.npz").resolve() == ref.resolve()
    # with a MISMATCHING candidate (its own output digest differs from the reference's)
    # no digest of the reference bytes appears in any public record
    (tmp_path / "wrong").mkdir()
    r2, consume2, _ = _cuda_case(tmp_path / "wrong", adapter_edit=_diverge, dsl_edit=_wrong_dsl)
    res2 = dispatch(r2, provider=PureSkillProvider(agent=consume2))
    assert not res2.verified
    val = [x for x in res2.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "trusted_reference_mismatch"
    tr2 = val["validation"]["trusted_reference"]
    priv2 = _private_verdict(tr2)
    ref2 = priv2["artifact_sha256"]["trusted_outputs"]
    assert ref2 != tr2["candidate_outputs_sha256"]
    root2 = _private_root_of(tr2)
    public = json.dumps(res2.to_dict()) + (tmp_path / "wrong" / "out" / "DISPATCH_STATE.json").read_text()
    for q in (tmp_path / "wrong" / "out").rglob("*.json"):
        if root2 not in q.parents and not q.is_symlink():
            public += q.read_text()
    assert ref2 not in public and priv2["verdict_sha256"] not in public and str(root2) not in public


def test_rewrite_and_restore_of_a_parent_input_file_is_refused(tmp_path):
    """The parent's purity check covers the end state AND the inode history the kernel
    records: a candidate that reopens the parent's 0400 input file (owner chmod),
    writes it and restores the bytes/mode leaves the hash equal but the ctime changed."""
    payload = _phase_payload(
        "from pathlib import Path as _P\nimport os as _os\n"
        "_p = _P(_f.f_globals['META']['inputs_npy']['x'])\n_b = _p.read_bytes()\n"
        "_os.chmod(_p, 0o600)\n_p.write_bytes(_b)\n_os.chmod(_p, 0o400)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "candidate_mutated_inputs" and val["scored"]
    mutated = val["validation"]["trusted_reference"]["input_audit"]["mutated_inputs"]
    assert any(m.startswith("x: written and restored") for m in mutated), mutated


def test_evidence_root_touched_during_the_candidate_is_refused(tmp_path):
    payload = _agentic_dir_payload(
        "for _d in _agentic.glob('.private-evidence-*'):\n    _os.chmod(_d, 0o700)\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert not res.verified
    val = [x for x in res.failure_receipts if x["stage"] == "agentic_validate"][0]
    assert val["reason_code"] == "evidence_root_tampered" and not val["scored"]
    tr = val["validation"]["trusted_reference"]
    assert "reference" not in tr["phases"] and tr["phases"]["candidate"]["returncode"] == -5


def test_candidate_metadata_carries_no_token_and_stdin_is_closed(tmp_path):
    payload = _phase_payload(
        "import sys as _sys, json as _json\nfrom pathlib import Path as _P\n"
        "_m = _f.f_globals['META']\n"
        "(_P(_m['phase_dir']) / 'meta_probe.json').write_text(_json.dumps({'keys': sorted(_m), 'stdin': _sys.stdin.read()}))\n")
    r, consume, _ = _cuda_case(tmp_path, dsl_edit=lambda d: _candidate_body(d, payload))
    res = dispatch(r, provider=PureSkillProvider(agent=consume))
    assert res.status == "agentic_verified"
    tr = res.validation["trusted_reference"]
    probe = json.loads((_phase_dir(tr, "candidate") / "meta_probe.json").read_text())
    assert probe["stdin"] == "" and not any(("token" in k or "nonce" in k or "audit" in k) for k in probe["keys"])
    assert set(probe["keys"]) >= {"candidate_path", "inputs_npy", "inputs_npy_sha256", "outputs_path", "report_path"}
    assert "inputs_path" not in probe["keys"]      # only this launch's inputs, laid out by the parent
    croot = _phase_dir(tr, "candidate")
    assert croot.parent == Path(tr["isolation"]["cwd"]) and croot.name.startswith("candidate-")
    assert (croot.stat().st_mode & 0o777) == 0o700
    assert sorted(p.name for p in (croot / "inputs").iterdir()) == ["000.npy", "001.npy"]
    assert all((p.stat().st_mode & 0o777) == 0o400 for p in (croot / "inputs").iterdir())
