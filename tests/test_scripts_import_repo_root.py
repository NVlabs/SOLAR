# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-invoked scripts under solar/scripts import the checkout they live in
(119-TRACE-PROVENANCE-1), not whatever editable install is on sys.path.

Each script is run as ``python <path> --help`` from a foreign cwd with a DECOY
``solar`` package first on PYTHONPATH and ``SOLAR_ENGINE_SOURCE_PROBE=exit``;
the decoy records a marker file if imported.  The probe line the script emits
must name this checkout's package, and the decoy must never be imported.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOLAR_DIR = Path(__file__).resolve().parent.parent          # <checkout>/solar
PKG_DIR = SOLAR_DIR / "solar"                                # <checkout>/solar/solar
SCRIPTS = sorted(p for p in (SOLAR_DIR / "scripts").glob("*.py")
                 if any(l.lstrip().startswith(("from solar", "import solar")) for l in p.read_text().splitlines()))


def _decoy(tmp_path: Path) -> Path:
    d = tmp_path / "decoy_site"
    (d / "solar").mkdir(parents=True)
    (d / "solar" / "__init__.py").write_text(
        "import os\n"
        "open(os.environ['DECOY_MARKER'], 'w').write('decoy solar imported')\n"
        "raise ImportError('DECOY solar package imported')\n")
    return d


def _run(script: Path, tmp_path: Path, args=("--help",)):
    decoy = _decoy(tmp_path)
    marker = tmp_path / "decoy_marker"
    foreign = tmp_path / "foreign_cwd"
    foreign.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(decoy) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["DECOY_MARKER"] = str(marker)
    env["SOLAR_ENGINE_SOURCE_PROBE"] = "exit"
    env.pop("PYTHONSAFEPATH", None)
    proc = subprocess.run([sys.executable, str(script), *args], cwd=str(foreign), env=env,
                          capture_output=True, text=True, timeout=300)
    return proc, marker


def _probe_record(stderr: str):
    for line in stderr.splitlines():
        if line.startswith("engine_source "):
            return json.loads(line[len("engine_source "):])
    return None


def test_inventory_is_complete():
    names = {p.name for p in SCRIPTS}
    assert names == {"diff_ir_stages.py", "gen_solexecbench_adapter.py", "trace_saved_ir.py", "verify_saved_ir.py"}
    assert len(SCRIPTS) == 4


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_imports_its_own_checkout(script, tmp_path):
    proc, marker = _run(script, tmp_path)
    assert not marker.exists(), f"{script.name} imported the DECOY solar package first on PYTHONPATH"
    assert "DECOY" not in proc.stderr, proc.stderr[-800:]
    rec = _probe_record(proc.stderr)
    assert rec is not None, f"{script.name} emitted no engine_source probe line; stderr: {proc.stderr[-800:]}"
    assert Path(rec["solar_file"]).resolve() == (PKG_DIR / "__init__.py").resolve(), rec
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-800:])
    assert "git_commit" in rec and "git_blob_init" in rec and "init_sha256" in rec


def test_harness_detects_leakage_on_an_unpinned_script(tmp_path):
    """Control: a script WITHOUT the repo-root insertion does import the decoy."""
    bad = tmp_path / "unpinned.py"
    bad.write_text("import solar\n")
    proc, marker = _run(bad, tmp_path, args=())
    assert marker.exists()
    assert proc.returncode != 0 and "DECOY" in proc.stderr


def test_engine_source_record_binds_this_checkout():
    from solar.engine_source import engine_source
    rec = engine_source()
    assert Path(rec["solar_file"]).resolve() == (PKG_DIR / "__init__.py").resolve()
    assert rec["init_sha256"] and len(rec["init_sha256"]) == 64
    if rec["git_toplevel"]:
        assert rec["git_commit"] and len(rec["git_commit"]) == 40
        assert rec["git_blob_init"] and len(rec["git_blob_init"]) == 40
