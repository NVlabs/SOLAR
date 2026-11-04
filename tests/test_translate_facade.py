# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dual-mode translation facade (solar.translate).

Covers: agent-mode scaffold (the default), the library verify() on the
skill's shipped verified example, the sol() chain on the same example, the
mode='api' key policy, and the CLI smoke.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SOLAR = Path(__file__).resolve().parents[1]
EXAMPLE = SOLAR / "skills" / "verified-translation" / "example"


# ---------------------------------------------------------------------------
# scaffold (agent mode)
# ---------------------------------------------------------------------------

def test_scaffold_creates_stub_notes_checklist(tmp_path):
    from solar.translate import scaffold
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "reference.py").write_text("def run(x):\n    return x\n")
    out = tmp_path / "manual"

    res = scaffold(problem, out)
    assert (out / "adapter.py").is_file()
    assert (out / "NOTES").is_file()
    stub = (out / "adapter.py").read_text()
    for fn in ("get_source", "get_inputs", "kernel_fn"):
        assert fn in stub
    assert "SKILL.md" in res.checklist
    assert "verify_saved_ir.py" in res.checklist


def test_scaffold_pytorch_has_no_adapter(tmp_path):
    from solar.translate import scaffold
    problem = tmp_path / "p"
    problem.mkdir()
    out = tmp_path / "manual"
    res = scaffold(problem, out, source_type="pytorch")
    assert not (out / "adapter.py").exists()
    assert (out / "README_pytorch.txt").is_file()
    assert "model_capped.py" in (out / "README_pytorch.txt").read_text()
    assert "SKILL.md" in res.checklist


def test_scaffold_never_clobbers_existing_adapter(tmp_path):
    from solar.translate import scaffold
    problem = tmp_path / "p"
    problem.mkdir()
    (problem / "reference.py").write_text("x = 1\n")
    out = tmp_path / "manual"
    out.mkdir()
    (out / "adapter.py").write_text("# in-progress work\n")
    scaffold(problem, out)
    assert (out / "adapter.py").read_text() == "# in-progress work\n"


# ---------------------------------------------------------------------------
# verify (library form) on the shipped verified example
# ---------------------------------------------------------------------------

def test_verify_skill_example():
    pytest.importorskip("torch")
    from solar.translate import verify
    res = verify(EXAMPLE / "adapter.py", EXAMPLE / "solar_model_einsum_lib.py",
                 source_type="cuda", quiet=True)
    assert res.verified
    assert res.max_abs_error is not None and res.max_abs_error < 1e-5
    assert res.message.startswith("VERIFIED")
    assert res.exit_code == 0


# ---------------------------------------------------------------------------
# sol chain on the example, against a minimal definition fixture
# ---------------------------------------------------------------------------

def test_sol_skill_example(tmp_path):
    pytest.importorskip("torch")
    from solar.translate import sol
    defn = {
        "name": "rmsnorm_example",
        "axes": {
            "B": {"type": "const", "value": 8},
            "H": {"type": "const", "value": 128},
        },
        "inputs": {
            "hidden_states": {"shape": ["B", "H"], "dtype": "float32"},
            "weight": {"shape": ["H"], "dtype": "float32"},
        },
    }
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "definition.json").write_text(json.dumps(defn))

    res = sol(EXAMPLE / "solar_model_einsum_lib.py", problem,
              tmp_path / "out", arch="B200", precision="fp16",
              fallback_adapter=EXAMPLE / "adapter.py")
    assert res.perf_path.is_file()
    assert res.fused_ms is not None and res.fused_ms > 0
    assert res.unfused_ms >= res.fused_ms  # unfused is the worst case
    assert res.analysis_path.is_file()
    assert res.einsum_path.is_file()


# ---------------------------------------------------------------------------
# translate(): mode policy
# ---------------------------------------------------------------------------

def test_translate_agent_default_never_needs_keys(tmp_path, monkeypatch):
    from solar.translate import translate
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    problem = tmp_path / "p"
    problem.mkdir()
    (problem / "reference.py").write_text("x = 1\n")
    res = translate(problem, tmp_path / "manual")  # mode defaults to agent
    assert res.mode == "agent"
    assert "SKILL.md" in res.checklist
    assert (tmp_path / "manual" / "adapter.py").is_file()


def test_translate_api_without_key_raises_policy(tmp_path, monkeypatch):
    from solar.translate import translate, API_POLICY
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    src = tmp_path / "model.py"
    src.write_text("x = 1\n")
    with pytest.raises(RuntimeError) as err:
        translate(src, tmp_path / "out", mode="api")
    assert "mode='api'" in str(err.value)
    assert API_POLICY.split(".")[0] in str(err.value)


def test_translate_unknown_mode_rejected(tmp_path):
    from solar.translate import translate
    with pytest.raises(ValueError):
        translate(tmp_path, tmp_path / "out", mode="llm")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "solar.cli.translate", "--help"],
        capture_output=True, text=True, cwd=SOLAR, timeout=120)
    assert proc.returncode == 0
    for word in ("scaffold", "verify", "sol", "api"):
        assert word in proc.stdout


def test_cli_bare_problem_dir_scaffolds(tmp_path):
    problem = tmp_path / "prob"
    problem.mkdir()
    (problem / "reference.py").write_text("x = 1\n")
    proc = subprocess.run(
        [sys.executable, "-m", "solar.cli.translate", str(problem)],
        capture_output=True, text=True, cwd=SOLAR, timeout=120)
    assert proc.returncode == 0
    assert "SKILL.md" in proc.stdout
    assert (problem / "manual" / "adapter.py").is_file()


def test_cli_api_without_key_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    src = tmp_path / "model.py"
    src.write_text("x = 1\n")
    import os
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "solar.cli.translate", "api", str(src),
         "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=SOLAR, timeout=120, env=env)
    assert proc.returncode == 2
    assert "mode='api'" in proc.stderr


def test_scaffold_triton_self_referencing_contract(tmp_path):
    """Triton sources take a self-referencing module — scaffold must emit the
    triton README, never a cuda-style adapter stub (v2 field bug)."""
    from solar.translate import scaffold
    src = Path(__file__).resolve().parents[1] / "examples_triton/TritonSoftmax/TritonSoftmax.py"
    r = scaffold(src, tmp_path, source_type="triton")
    names = sorted(f.name for f in r.files)
    assert "README_triton.txt" in names
    assert "adapter.py" not in names
    assert (tmp_path / "README_triton.txt").read_text().count("kernel_fn")
    assert "README_triton" in r.checklist or "self-referencing" in r.checklist.lower()
