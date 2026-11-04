# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/diff_ir_stages.py must import and answer --help (issue #21).

When ``scripts/verify_saved_ir.py`` became a thin wrapper over ``solar.translate``
(an earlier revision) its private helpers ``_coerce`` and ``_sized_model`` moved to
``solar.translate._runner`` as ``coerce_config_value`` and ``sized_model``, but
``scripts/diff_ir_stages.py`` kept importing them from the script and failed at
start with ``ImportError``.  These controls import the script module directly,
run ``--help`` in a subprocess and check that the two helpers are the facade's
own functions with their an earlier revision behaviour.  No model, reference or DSL program
is executed.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "diff_ir_stages.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("diff_ir_stages_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_module_imports_and_binds_the_facade_helpers():
    from solar.translate._runner import coerce_config_value, sized_model

    module = _load_script_module()
    assert module._coerce is coerce_config_value
    assert module._sized_model is sized_model
    assert callable(module.main)


def test_help_exits_zero_without_executing_anything():
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage:")
    assert "--config" in proc.stdout and "--stages" in proc.stdout
    assert "ImportError" not in proc.stderr


@pytest.mark.parametrize("text,expected", [
    ("1", 1), ("1.5", 1.5), ("true", True), ("False", False), ("none", None), ("null", None), ("abc", "abc"),
])
def test_coerce_keeps_the_e117755_contract(text, expected):
    module = _load_script_module()
    value = module._coerce(text)
    assert value == expected and type(value) is type(expected)


def test_sized_model_writes_and_removes_the_sized_copy(tmp_path):
    module = _load_script_module()
    model = tmp_path / "model.py"
    model.write_text("CONFIG = {'batch': 8}\n")
    with module._sized_model(model, {"batch": 1, "seq_len": 128}, tmp_path) as sized:
        assert sized != model and sized.exists()
        text = sized.read_text()
        assert "CONFIG['batch'] = 1" in text and "CONFIG['seq_len'] = 128" in text
        assert "defines no CONFIG dict" in text  # the guard the facade appends
    assert not sized.exists()
    with module._sized_model(model, {}, tmp_path) as same:
        assert same == model
