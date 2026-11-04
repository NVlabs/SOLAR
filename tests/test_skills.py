# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the agent skills (skills/) — layout, frontmatter, and the shipped
verified example.

Layered per skill-testing practice:
  1. structural: canonical location, cross-agent symlinks, AGENTS.md pointer
  2. contract:   SKILL.md frontmatter (name/description rules, CSO)
  3. behavioral (cheap): the example adapter+DSL pair actually VERIFIES via
     scripts/verify_saved_ir.py — the same command the skill teaches.

The full behavioral test (a fresh agent following only the skill on an unseen
problem) is a manual protocol documented in skills/verified-translation/ — it
requires an agent harness and is not run here.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

SOLAR = Path(__file__).resolve().parents[1]
SKILLS = SOLAR / "skills"
SKILL_NAMES = ["verified-translation", "capping-weights"]


# ---------------------------------------------------------------------------
# 1. Layout: neutral canonical dir + per-agent discovery symlinks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SKILL_NAMES)
def test_canonical_skill_exists(name):
    assert (SKILLS / name / "SKILL.md").is_file()


@pytest.mark.parametrize("agent_dir", [".claude/skills", ".agents/skills"])
@pytest.mark.parametrize("name", SKILL_NAMES)
def test_agent_symlinks_resolve(agent_dir, name):
    link = SOLAR / agent_dir / name
    assert link.is_symlink(), f"{link} must be a symlink to the canonical skill"
    assert (link / "SKILL.md").resolve() == (SKILLS / name / "SKILL.md").resolve()


def test_agents_md_points_at_skills():
    text = (SOLAR / "AGENTS.md").read_text()
    assert "skills/verified-translation" in text


# ---------------------------------------------------------------------------
# 2. Frontmatter contract (agentskills.io: name + description; CSO style)
# ---------------------------------------------------------------------------

def _frontmatter(path: Path) -> dict:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, f"{path} missing YAML frontmatter"
    fm = {}
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^\S+:", line):
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
        elif key:  # folded continuation
            fm[key] += " " + line.strip()
    return fm


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_frontmatter(name):
    fm = _frontmatter(SKILLS / name / "SKILL.md")
    assert fm.get("name") == name, "frontmatter name must match directory"
    assert re.fullmatch(r"[a-z0-9-]+", fm["name"])
    desc = fm.get("description", "")
    assert desc.startswith("Use when"), "description = triggering conditions (CSO)"
    assert len(f"name: {fm['name']} description: {desc}") <= 1024


def test_description_has_no_workflow_summary():
    # CSO: the description must not teach the workflow (or agents skip the body)
    desc = _frontmatter(SKILLS / "verified-translation" / "SKILL.md")["description"]
    for phrase in ("step", "first", "then", "default mode", "authors the DSL"):
        assert phrase.lower() not in desc.lower(), f"workflow leak in description: {phrase!r}"


# ---------------------------------------------------------------------------
# 3. Behavioral: the shipped example pair must VERIFY with the taught command
# ---------------------------------------------------------------------------

EXAMPLE = SKILLS / "verified-translation" / "example"


def test_example_files_present():
    for f in ("adapter.py", "solar_model_einsum_lib.py", "selfcheck.py"):
        assert (EXAMPLE / f).is_file()


def test_example_pair_verifies():
    torch = pytest.importorskip("torch")  # noqa: F841
    proc = subprocess.run(
        [sys.executable, str(SOLAR / "scripts" / "verify_saved_ir.py"),
         str(EXAMPLE / "adapter.py"), str(EXAMPLE / "solar_model_einsum_lib.py"),
         "--source-type", "cuda"],
        capture_output=True, text=True, timeout=600, cwd=SOLAR,
    )
    assert "VERIFIED" in proc.stdout, f"example failed to verify:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


def test_all_zero_reference_warns():
    """The vacuous-pass guard: an all-zero reference output must WARN."""
    import numpy as np
    sys.path.insert(0, str(SOLAR))
    from solar.einsum_lib.verify import compare_outputs
    import io, contextlib
    ref = {"y": np.zeros((4, 4))}
    got = {"y": np.zeros((4, 4))}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare_outputs(got, ref)
    assert res.success
    assert "ALL ZEROS" in buf.getvalue()
