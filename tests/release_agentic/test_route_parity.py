# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route parity: the same fixture must yield the same verdict, error
statistic and exit status on every verification entry point.

Routes under one contract (release/agentic_contract/PARITY.md, rows V1-V6):
  * pure-skill : scripts/verify_saved_ir.py            (subprocess)
  * Pythonic   : solar.translate.verify (library)      (in-process)
  * Pythonic   : python -m solar.cli.translate verify  (subprocess)
"""
from __future__ import annotations

import math

import pytest

from conftest import (CUDA, PT, ZERO, cli_verify, lib_verify,
                      parse_verified_line, script_verify)


@pytest.mark.parametrize("src_type, model, ir", [
    ("pytorch", PT / "model.py", PT / "model_einsum_lib.py"),
    ("cuda", CUDA / "adapter.py", CUDA / "solar_model_einsum_lib.py"),
])
def test_correct_translation_verifies_identically_on_all_routes(
        torch, src_type, model, ir):
    s = script_verify(model, ir, "--source-type", src_type, "--quiet")
    c = cli_verify(model, ir, "--source-type", src_type)
    l = lib_verify(model, ir, source_type=src_type)

    assert s.returncode == 0, s.stdout[-1500:] + s.stderr[-1500:]
    assert c.returncode == 0, c.stdout[-1500:] + c.stderr[-1500:]
    assert l.verified and l.exit_code == 0

    sv, se = parse_verified_line(s.stdout)
    cv, ce = parse_verified_line(c.stdout)
    assert sv == cv == "VERIFIED"
    assert l.message.startswith("VERIFIED")
    # Same checker, same inputs, same fp64 execution: identical statistic.
    assert math.isclose(se, ce, rel_tol=0, abs_tol=0), (se, ce)
    # the printed line carries 4 significant digits
    assert math.isclose(l.max_abs_error, se, rel_tol=1e-3), (l.max_abs_error, se)
    # The documented default tolerance applies (never relaxed here).
    assert l.max_abs_error < 1e-5
    assert l.execution.rtol == 1e-4 and l.execution.atol == 1e-5


def test_wrong_translation_mismatches_identically_on_all_routes(torch):
    model, ir = PT / "model.py", PT / "model_wrong_einsum_lib.py"
    s = script_verify(model, ir, "--source-type", "pytorch", "--quiet")
    c = cli_verify(model, ir, "--source-type", "pytorch")
    l = lib_verify(model, ir, source_type="pytorch")

    assert s.returncode == 1 and c.returncode == 1
    assert not l.verified and l.exit_code == 1
    assert parse_verified_line(s.stdout)[0] == "MISMATCH"
    assert parse_verified_line(c.stdout)[0] == "MISMATCH"
    assert l.message.startswith("MISMATCH")
    assert "values differ" in l.message
    # The wrong DSL halves the product: error is O(output/2), not noise.
    assert l.max_abs_error > 1e-2


def test_wrong_output_key_is_a_mismatch_not_a_pass(torch):
    """Outputs are compared BY KEY (SKILL.md step 4 contract)."""
    model, ir = PT / "model.py", PT / "model_badkey_einsum_lib.py"
    s = script_verify(model, ir, "--source-type", "pytorch", "--quiet")
    l = lib_verify(model, ir, source_type="pytorch")
    assert s.returncode == 1
    assert "Output key mismatch" in s.stdout
    assert not l.verified and "Output key mismatch" in l.message
    assert l.max_abs_error is None   # no statistic was computed


def test_execution_failure_is_reported_not_passed(torch):
    model, ir = PT / "model.py", PT / "model_crash_einsum_lib.py"
    s = script_verify(model, ir, "--source-type", "pytorch", "--quiet")
    c = cli_verify(model, ir, "--source-type", "pytorch")
    l = lib_verify(model, ir, source_type="pytorch")
    assert s.returncode == 1 and c.returncode == 1
    assert parse_verified_line(s.stdout)[0] == "FAILED"
    assert parse_verified_line(c.stdout)[0] == "FAILED"
    assert not l.verified and l.message.startswith("IR execution FAILED")
    assert l.execution is not None   # evidence exists even for failures


def test_shape_only_is_not_a_numeric_verdict_on_any_route(torch):
    """``--shape-only`` skips the comparison; the library result says so."""
    l = lib_verify(CUDA / "adapter.py", CUDA / "solar_model_einsum_lib.py",
                   source_type="cuda", shape_only=True)
    assert l.shape_only and not l.verified
    assert l.max_abs_error is None
    assert l.message.startswith("SHAPE-ONLY")


def test_all_zero_reference_warns_on_every_route(torch):
    """The vacuous-pass guard WARNS on stdout for script and library.

    Both routes print the warning; neither records it (see FINDING A4 in
    test_structural_vs_numeric.py).
    """
    model, ir = ZERO / "adapter.py", ZERO / "solar_model_einsum_lib.py"
    s = script_verify(model, ir, "--source-type", "cuda", "--quiet")
    c = cli_verify(model, ir, "--source-type", "cuda")
    assert s.returncode == 0 and c.returncode == 0
    assert "ALL ZEROS" in s.stdout
    assert "ALL ZEROS" in c.stdout


def test_library_and_cli_share_default_tolerance_and_source_type_policy():
    """Facade library and facade CLI agree on defaults (rtol/atol/source)."""
    import argparse
    from solar.cli import translate as cli
    from solar.translate import verify as lib_verify_fn
    import inspect
    sig = inspect.signature(lib_verify_fn)
    assert sig.parameters["source_type"].default == "cuda"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    cli._add_verify(sub)
    ns = parser.parse_args(["verify", "m.py", "i.py"])
    assert ns.source_type == "cuda"
    assert ns.rtol == 1e-4 and ns.atol == 1e-5
