# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact unsupported / error behavior of each agentic entry point.

These pin what the pinned checkout DOES, so the release notes can state it
and a later change is visible.  Where two routes disagree on the same
condition the disagreement is recorded as a finding and the DESIRED
behavior is a strict xfail (PARITY.md rows E1-E7).

ADJUDICATION (release adjudication of the agentic contract, archive of the verbatim
audit file: an earlier revision).  Tests that pinned a DEFECT's old behavior are
rewritten to pin the repaired contract; strict xfails whose contract is now
delivered have the marker removed.  Policy-gated findings keep their
original xfail/pinning tests (A3 exit code, A10 tolerance): the legacy
default is deliberately preserved and only exposed/gated (see
test_repairs.py).
"""
from __future__ import annotations

import sys

import pytest

from conftest import CUDA, EXAMPLE, PT, cli_verify, run, script_verify

NOPE = "/nonexistent/release_agentic/model.py"


def test_script_missing_file_exits_2_with_message():
    p = script_verify(NOPE, PT / "model_einsum_lib.py", "--source-type", "pytorch")
    assert p.returncode == 2
    assert "ERROR: model file not found" in p.stderr


@pytest.mark.finding("A9")
def test_cli_missing_file_fixed_matches_script(torch):
    """Repaired: the facade CLI shares the script's existence check
    (was: FileNotFoundError traceback, exit 1)."""
    p = cli_verify(NOPE, PT / "model_einsum_lib.py", "--source-type", "pytorch")
    assert p.returncode == 2
    assert "ERROR: model file not found" in p.stderr
    assert "Traceback" not in p.stderr


@pytest.mark.finding("A9")
def test_cli_missing_file_should_match_script_contract():
    p = cli_verify(NOPE, PT / "model_einsum_lib.py", "--source-type", "pytorch")
    assert p.returncode == 2 and "not found" in p.stderr


def test_unknown_source_type_is_refused_by_argparse_on_both_clis():
    s = script_verify(PT / "model.py", PT / "model_einsum_lib.py",
                      "--source-type", "raw")
    c = cli_verify(PT / "model.py", PT / "model_einsum_lib.py",
                   "--source-type", "raw")
    assert s.returncode == 2 and "invalid choice: 'raw'" in s.stderr
    assert c.returncode == 2 and "invalid choice: 'raw'" in c.stderr


def test_script_default_source_type_differs_from_facade_default(torch):
    """Script default is ``jax``; facade default is ``cuda``.

    Omitting ``--source-type`` on the script for an adapter therefore fails
    deep inside the JAX frontend with a conversion traceback instead of a
    contract message.  SKILL.md always spells the flag; this pins why.
    """
    p = script_verify(CUDA / "adapter.py", CUDA / "solar_model_einsum_lib.py",
                      "--quiet")
    assert p.returncode == 1
    assert "jax_frontend" in p.stderr
    assert "VERIFIED" not in p.stdout


def test_api_mode_without_key_refuses_everywhere(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from solar.translate import API_POLICY, translate
    src = tmp_path / "model.py"
    src.write_text("x = 1\n")
    with pytest.raises(RuntimeError) as err:
        translate(src, tmp_path / "out", mode="api")
    assert API_POLICY.split(".")[0] in str(err.value)
    p = run([sys.executable, "-m", "solar.cli.translate", "api", src,
             "--out-dir", tmp_path / "out"])
    assert p.returncode == 2
    assert "mode='api'" in p.stderr
    assert not (tmp_path / "out").exists()   # nothing was created


def test_toDSL_model_without_key_fails_before_any_translation(torch, tmp_path):
    """The orchestration CLI refuses when no key resolves (exit != 0, no
    artifacts).  SOLAR reads no implicit key file, so only the environment
    and explicit arguments can supply one."""
    p = run([sys.executable, "-m", "solar.cli.toDSL_model", PT / "model.py",
             "--source-type", "pytorch", "--output-dir", tmp_path / "dsl",
             "--max-retries", "1"], timeout=300)
    assert p.returncode != 0
    assert "API key required" in (p.stdout + p.stderr)
    assert not (tmp_path / "dsl").exists()


def test_scaffold_does_not_translate_or_verify(tmp_path):
    """``translate scaffold`` creates files and a checklist only."""
    problem = tmp_path / "prob"
    problem.mkdir()
    (problem / "reference.py").write_text("def run(x):\n    return x\n")
    p = run([sys.executable, "-m", "solar.cli.translate", "scaffold", problem,
             "--out-dir", tmp_path / "ws"])
    assert p.returncode == 0
    assert (tmp_path / "ws" / "adapter.py").is_file()
    assert "NotImplementedError" in (tmp_path / "ws" / "adapter.py").read_text()
    assert not (tmp_path / "ws" / "solar_model_einsum_lib.py").exists()
    assert "VERIFIED" not in p.stdout


@pytest.mark.finding("A7")
def test_sol_cli_has_strict_receipt_switch():
    """Repaired (PARITY row G2): the CLI ``sol`` exposes the receipt gate and
    the library-only selectors."""
    p = run([sys.executable, "-m", "solar.cli.translate", "sol", "--help"])
    assert p.returncode == 0
    for flag in ("--strict", "--receipt", "--source-path", "--workload-idx",
                 "--perf-precision"):
        assert flag in p.stdout, f"{flag} missing"


def test_sol_library_strict_requires_receipt_and_source(tmp_path):
    from solar.translate import sol
    from solar.translate.receipts import ReceiptRefusal
    with pytest.raises(ReceiptRefusal, match="requires a receipt"):
        sol(CUDA / "solar_model_einsum_lib.py", CUDA, tmp_path / "o", strict=True)
    with pytest.raises(ReceiptRefusal, match="requires source_path"):
        sol(CUDA / "solar_model_einsum_lib.py", CUDA, tmp_path / "o",
            strict=True, receipt=tmp_path / "none.json")
    assert not (tmp_path / "o").exists()   # refused BEFORE any tracing


def test_sol_library_strict_refuses_missing_receipt_before_tracing(tmp_path):
    from solar.translate import sol
    from solar.translate.receipts import ReceiptRefusal
    with pytest.raises(ReceiptRefusal, match="receipt missing"):
        sol(CUDA / "solar_model_einsum_lib.py", CUDA, tmp_path / "o",
            strict=True, receipt=tmp_path / "none.json",
            source_path=CUDA / "adapter.py")
    assert not (tmp_path / "o").exists()


@pytest.mark.finding("A12")
def test_compare_outputs_empty_output_is_a_verdict():
    """Repaired: zero-size outputs produce a failed verdict (nothing to
    compare), never a ValueError traceback."""
    import numpy as np
    from solar.einsum_lib.verify import compare_outputs
    r = compare_outputs({"y": np.zeros((0, 4))}, {"y": np.zeros((0, 4))})
    assert r.success is False and "empty" in r.error


def test_compare_outputs_nan_never_passes_and_inf_matches_inf():
    import numpy as np
    from solar.einsum_lib.verify import compare_outputs
    nan = compare_outputs({"y": np.array([np.nan, 1.0])},
                          {"y": np.array([np.nan, 1.0])})
    assert not nan.success
    inf = compare_outputs({"y": np.array([np.inf, 1.0])},
                          {"y": np.array([np.inf, 1.0])})
    assert inf.success and inf.max_abs_error == 0.0
