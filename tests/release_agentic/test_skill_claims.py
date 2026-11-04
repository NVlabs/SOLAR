# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skill instructions versus implementation: every command, flag, default,
success marker and contract sentence the pure-skill route relies on must
match the pinned code.  Pure text/AST checks plus the shipped example.
"""
from __future__ import annotations

import re
import sys

import pytest

from conftest import EXAMPLE, SOLAR, run

SKILL = SOLAR / "skills" / "verified-translation" / "SKILL.md"
REF = SOLAR / "skills" / "verified-translation" / "reference.md"
CAP = SOLAR / "skills" / "capping-weights" / "SKILL.md"


def _text(p):
    return p.read_text(encoding="utf-8")


def test_every_script_the_skill_names_exists():
    text = _text(SKILL) + _text(REF) + _text(CAP)
    for rel in sorted(set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)", text))):
        assert (SOLAR / rel).is_file(), f"skill references missing {rel}"
    for rel in ("example/selfcheck.py", "example/adapter.py",
                "example/solar_model_einsum_lib.py"):
        assert (SKILL.parent / rel).is_file()


def test_every_flag_the_skill_passes_to_verify_saved_ir_is_accepted():
    """Flags appearing after verify_saved_ir.py in the skill text must parse."""
    text = _text(SKILL) + _text(REF)
    flags = set()
    for m in re.finditer(r"verify_saved_ir\.py[^\n`\\]*(?:\\\n[^\n`\\]*)*", text):
        flags |= set(re.findall(r"--[a-z][a-z0-9-]*", m.group(0)))
    flags.discard("--tolerance")   # prose placeholder "[tolerance flags]"
    h = run([sys.executable, SOLAR / "scripts" / "verify_saved_ir.py", "--help"])
    assert h.returncode == 0
    for f in sorted(flags):
        assert f in h.stdout, f"SKILL passes {f}, script --help lacks it"


def test_skill_always_spells_source_type_for_the_script():
    """The script default is jax; a cuda/pytorch command without the flag
    fails with a frontend traceback (test_error_behavior).  Every runnable
    verify command in the skill therefore carries --source-type."""
    for m in re.finditer(r"verify_saved_ir\.py[^\n`\\]*(?:\\\n[^\n`\\]*)*", _text(SKILL)):
        cmd = m.group(0)
        if "$M/" in cmd or "model.py" in cmd or "<model" in cmd:
            assert "--source-type" in cmd, cmd


def test_documented_success_line_matches_runner_output():
    """SKILL.md: 'Success line: VERIFIED (max_abs_error=…)'."""
    assert "VERIFIED (max_abs_error=" in _text(SKILL)
    runner = (SOLAR / "solar" / "translate" / "_runner.py").read_text()
    assert 'f"VERIFIED  (max_abs_error=' in runner


def test_reference_md_marker_section_distinguishes_manual_and_api_routes():
    """reference.md 'Verification internals' must not present the API-only
    'GENERATED CODE (VERIFIED)' marker as THE success marker: the manual
    route prints 'VERIFIED  (max_abs_error=…)' and never that string."""
    sec = _text(REF).split("## Verification internals")[1].split("\n## ")[0]
    assert "GENERATED CODE (VERIFIED)" in sec
    assert "VERIFIED  (max_abs_error" in sec or "VERIFIED (max_abs_error" in sec
    assert "toDSL_model" in sec or "API" in sec


def test_skill_states_exit_code_and_receipt_limits():
    """The skill must tell the agent what exit 0 does NOT mean on this
    checkout: shape-only exits 0, the all-zero warning is stdout-only, and
    the manual route emits no receipt (strict SOL needs Python)."""
    text = _text(SKILL)
    assert "--shape-only" in text and "exit" in text
    assert "ALL ZEROS" in text
    assert "emit_receipt" in text or "receipt" in text
    assert "strict" in text


def test_skill_tolerance_table_matches_runner_defaults():
    import inspect
    from solar.translate._runner import run_verification
    sig = inspect.signature(run_verification)
    assert sig.parameters["rtol"].default == 1e-4
    assert sig.parameters["atol"].default == 1e-5
    assert "rtol 1e-4/atol 1e-5" in _text(SKILL)
    assert "rtol 1e-4 / atol 1e-5" in _text(REF)


def test_skill_numba_decline_row_matches_both_routes():
    """Manual route: the runner never sets fail_on_numba_decline, so a decline
    falls back to the interpreter with a WARN; API route fails fast."""
    runner = (SOLAR / "solar" / "translate" / "_runner.py").read_text()
    assert "fail_on_numba_decline" not in runner
    agent = (SOLAR / "solar" / "einsum_lib" / "agent.py").read_text()
    assert "fail_on_numba_decline=self.fast" in agent
    row = [l for l in _text(SKILL).splitlines() if "Numba declined" in l]
    assert row, "troubleshooting row missing"
    assert "interpreter" in row[0] or "falls back" in row[0]


def test_selfcheck_template_uses_an_existing_private_hook():
    """example/selfcheck.py reaches into verify._MATH_BUILTINS."""
    from solar.einsum_lib import verify
    assert hasattr(verify, "_MATH_BUILTINS")
    assert "_MATH_BUILTINS" in (EXAMPLE / "selfcheck.py").read_text()


def test_pytorch_naming_contract_in_reference_md_matches_frontend(torch):
    """reference.md: single activation is 'x', params dotted->underscore,
    single output keyed 'output'."""
    from conftest import PT
    from solar.einsum_lib.pipeline import prepare_pytorch
    src, shapes, dtypes, ref, *_ = prepare_pytorch(str(PT / "model.py"), True,
                                                    use_fp64=True)
    assert set(shapes) == {"x", "w"}
    assert set(ref) == {"output"}
    t = _text(REF)
    assert '`"x"`' in t and 'keyed `"output"`' in t


def test_capping_skill_command_and_hooks_exist():
    """capping-weights: its `--const-override` belongs to
    scripts/gen_solexecbench_adapter.py (not toDSL_model) and must parse there;
    the core hook it cites must exist."""
    text = _text(CAP)
    assert "scripts/gen_solexecbench_adapter.py" in text
    h = run([sys.executable, SOLAR / "scripts" / "gen_solexecbench_adapter.py", "--help"])
    assert h.returncode == 0
    assert "--const-override" in h.stdout
    from solar.einsum_lib import core
    assert hasattr(core, "_shape_only_zeros")
    # The true-scale trace tool is scripts/trace_saved_ir.py; the skill must
    # name it rather than only a private core hook.
    assert "trace_saved_ir.py" in text
