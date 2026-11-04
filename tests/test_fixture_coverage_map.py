# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tests/fixtures/** is harness data: the files named ``test_*.py`` under it are vendored controls that normal tests copy into
a run directory and execute explicitly (or read byte-for-byte).  They are not collected directly (``collect_ignore`` in
tests/conftest.py).  ``tests/fixtures/COVERAGE_MAP.json`` names, for every such file, the normal test modules that consume
it.  The rule is FILE-LEVEL: a consumer must name the control file itself (its path relative to tests/fixtures/ or its bare
filename as a whole token); naming only an ancestor directory does not count.  An orphan (a control with no such consumer)
FAILS this module; the negative self-test below proves the checker detects one."""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
TESTS_PARENT = HERE.parent
FIX = HERE / "fixtures"
MAP_PATH = FIX / "COVERAGE_MAP.json"


def _is_test_module(p: pathlib.Path) -> bool:
    return p.suffix == ".py" and (p.name.startswith("test_") or p.name.endswith("_test.py"))


def _names_file(text: str, control: str) -> bool:
    """True when ``text`` names the control FILE: its path relative to tests/fixtures/ or its bare filename as a whole token."""
    sub = control[len("tests/fixtures/"):]
    base = sub.split("/")[-1]
    return sub in text or re.search(r"(?<![\w/])" + re.escape(base) + r"(?![\w])", text) is not None


def check_coverage(mapping: dict, root: pathlib.Path) -> list:
    """Return the list of problems (empty when the map is complete, current and every control has a real file-level consumer).
    ``root`` is the directory that contains ``tests/``."""
    problems = []
    fixtures = root / "tests" / "fixtures"
    on_disk = sorted(str(p.relative_to(root)) for p in fixtures.rglob("*.py") if _is_test_module(p))
    mapped = sorted(mapping.get("entries", {}))
    for m in sorted(set(on_disk) - set(mapped)):
        problems.append(("missing_from_map", m))
    for m in sorted(set(mapped) - set(on_disk)):
        problems.append(("stale_in_map", m))
    for m, e in mapping.get("entries", {}).items():
        real = []
        for c in e.get("consumers", []):
            p = root / c
            if not p.is_file() or c.startswith("tests/fixtures/") or not _is_test_module(p):
                problems.append(("consumer_not_a_normal_test", m, c)); continue
            if not _names_file(p.read_text(encoding="utf-8", errors="replace"), m):
                problems.append(("consumer_does_not_name_the_file", m, c)); continue
            real.append(c)
        if not real:
            problems.append(("orphan", m))
    if mapping.get("uncovered"):
        problems.append(("map_declares_uncovered", tuple(mapping["uncovered"])))
    return problems


def test_conftest_does_not_collect_fixtures():
    text = (HERE / "conftest.py").read_text()
    assert re.search(r'^collect_ignore\s*=\s*\[\s*"fixtures"\s*\]', text, re.M), "tests/conftest.py must declare collect_ignore = [\"fixtures\"]"


def test_coverage_map_is_complete_and_every_control_has_a_file_level_consumer():
    mapping = json.loads(MAP_PATH.read_text())
    assert check_coverage(mapping, TESTS_PARENT) == []


def test_checker_fails_on_a_synthetic_orphan(tmp_path):
    """Negative self-test: a control whose only listed consumer names merely its ancestor directory is an ORPHAN."""
    root = tmp_path / "checkout"
    (root / "tests" / "fixtures" / "ctrl_dir").mkdir(parents=True)
    (root / "tests" / "fixtures" / "ctrl_dir" / "test_control.py").write_text("def test_x():\n    assert True\n")
    (root / "tests" / "test_consumer_dir_only.py").write_text("FIX = 'tests/fixtures/ctrl_dir'  # names the directory, never the file\n")
    (root / "tests" / "test_consumer_real.py").write_text("NAME = 'ctrl_dir/test_control.py'\n")
    orphan_map = {"entries": {"tests/fixtures/ctrl_dir/test_control.py": {"consumers": ["tests/test_consumer_dir_only.py"]}}, "uncovered": []}
    problems = check_coverage(orphan_map, root)
    assert ("consumer_does_not_name_the_file", "tests/fixtures/ctrl_dir/test_control.py", "tests/test_consumer_dir_only.py") in problems
    assert ("orphan", "tests/fixtures/ctrl_dir/test_control.py") in problems
    good_map = {"entries": {"tests/fixtures/ctrl_dir/test_control.py": {"consumers": ["tests/test_consumer_real.py"]}}, "uncovered": []}
    assert check_coverage(good_map, root) == []
    unlisted = {"entries": {}, "uncovered": []}
    assert ("missing_from_map", "tests/fixtures/ctrl_dir/test_control.py") in check_coverage(unlisted, root)
