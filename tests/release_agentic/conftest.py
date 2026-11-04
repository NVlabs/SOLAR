# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the release_agentic contract tests.

Everything here is OFFLINE: synthetic fixtures under ``fixtures/``, a mocked
LLM client where the Pythonic route is exercised, no inference API, no GPU.
The tests compare the Pythonic (library / ``solar-translate`` CLI /
``toDSL_model`` orchestration) and pure-skill (``scripts/verify_saved_ir.py``
+ skill instructions) routes against ONE documented contract
(release/agentic_contract/PARITY.md).

Conventions
-----------
* A test that pins behavior the contract REQUIRES and the pinned checkout
  does NOT deliver is marked ``xfail(strict=True)`` with the finding id
  (the release adjudication's findings list).  Strict xfail flips to a hard
  failure once the owner fixes the defect, which is the signal to retire
  the marker; it never silently passes.
* Tests never relax a tolerance and never edit core/analyzer/IR files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOLAR = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PT = FIXTURES / "pt_matmul"
CUDA = FIXTURES / "cuda_scale"
ZERO = FIXTURES / "cuda_zero"
EXAMPLE = SOLAR / "skills" / "verified-translation" / "example"
VERIFY_SCRIPT = SOLAR / "scripts" / "verify_saved_ir.py"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "finding(id): links a test to a release-adjudication finding id")


@pytest.fixture(scope="session", autouse=True)
def _solar_is_this_checkout():
    """Refuse to run against a different checkout.

    The conda environment's editable install points at another tree; an
    unpinned ``import solar`` would silently test the core owner's working
    copy instead of the pinned release candidate.
    """
    import solar
    got = Path(solar.__file__).resolve()
    assert SOLAR.resolve() in got.parents, (
        f"solar imported from {got}, not from the checkout under test "
        f"({SOLAR}); run with PYTHONPATH={SOLAR} or from that directory")


def subprocess_env() -> dict:
    """Environment for child processes: pin THIS checkout, no bytecode."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SOLAR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("OPENAI_API_KEY", None)   # the suite never has credentials
    return env


def run(cmd: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True,
                          text=True, cwd=SOLAR, env=subprocess_env(),
                          timeout=timeout)


def script_verify(model, ir, *extra) -> subprocess.CompletedProcess:
    """Pure-skill route: the command SKILL.md step 5 teaches."""
    return run([sys.executable, VERIFY_SCRIPT, model, ir, *extra])


def cli_verify(model, ir, *extra) -> subprocess.CompletedProcess:
    """Pythonic facade CLI: ``solar-translate verify``."""
    return run([sys.executable, "-m", "solar.cli.translate", "verify",
                model, ir, *extra])


def lib_verify(model, ir, **kw):
    """Pythonic library: ``solar.translate.verify`` (in-process)."""
    from solar.translate import verify
    kw.setdefault("quiet", True)
    return verify(model, ir, **kw)


def parse_verified_line(stdout: str):
    """(verdict, max_abs_error) from the documented success/mismatch lines."""
    for line in stdout.splitlines():
        if line.startswith("VERIFIED"):
            err = float(line.split("max_abs_error=")[1].split(",")[0])
            return "VERIFIED", err
        if line.startswith("MISMATCH"):
            return "MISMATCH", None
        if line.startswith("IR execution FAILED"):
            return "FAILED", None
        if line.startswith("SHAPE-ONLY"):
            return "SHAPE-ONLY", None
    return None, None


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def workload_identity():
    wl = json.loads((CUDA / "workload.jsonl").read_text().splitlines()[0])
    return wl["uuid"], wl["axes"]
