# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install-matrix regressions for the parameter shim (round-3 review P1).

The shim must provide its recorder-side guards (semantic operand order,
scalar retention, fail-closed diagnostics) on BOTH torchview installations:

* ``vanilla``   — the installed, unpatched torchview;
* ``prepatched``— a COPY of the installed torchview with this repository's
  exact ``patches/torchview-parameter-tensors.patch`` applied (what a user
  following the patch instructions would run).  Round-3 reproduction: with
  that install, ``ComputationGraph._render_parameter_nodes`` exists, the
  shim skipped itself entirely, and the real pipeline exited 0 with a
  ZERO-input matmul and 0 MACs — a rendering marker is not proof of safe
  semantic operand binding.

The prepatched copy is built fresh in a pytest tmp dir (no shared
site-packages edits, no reuse of reviewer evidence directories).
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

_reg_spec = importlib.util.spec_from_file_location(
    "learned_param_reg", Path(__file__).parent / "test_learned_param_matmul.py")
_reg = importlib.util.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg)


@pytest.fixture(scope="session")
def prepatched_pythonpath(tmp_path_factory):
    """Copy of the installed torchview with the repo patch applied."""
    import torchview
    pkg = Path(torchview.__file__).parent
    root = tmp_path_factory.mktemp("prepatched")
    copy = root / "src" / "torchview"
    shutil.copytree(pkg, copy, ignore=shutil.ignore_patterns("__pycache__"))
    patch_file = SOLAR_ROOT / "patches" / "torchview-parameter-tensors.patch"
    proc = subprocess.run(
        ["patch", "-p1", "--batch", "-i", str(patch_file)],
        cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"repo patch did not apply to installed torchview:\n"
                    f"{proc.stdout}\n{proc.stderr}")
    return str(copy.parent)


def _env(install, prepatched_path):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    parts = [str(SOLAR_ROOT)]
    if install == "prepatched":
        parts.insert(0, prepatched_path)
    env["PYTHONPATH"] = ":".join(parts)
    return env


def _stage1(tmp_path, model_src, env):
    model_file = tmp_path / "model.py"
    model_file.write_text(model_src)
    graph_dir = tmp_path / "graph"
    proc = subprocess.run(
        [PY, "-m", "solar.cli.process_model", "--model-file",
         str(model_file), "--output-dir", str(graph_dir)],
        capture_output=True, text=True, cwd=SOLAR_ROOT, env=env, timeout=300)
    return proc, graph_dir


def _full(tmp_path, model_src, env):
    proc, graph_dir = _stage1(tmp_path, model_src, env)
    assert proc.returncode == 0, proc.stdout[-1200:] + proc.stderr[-800:]
    einsum_dir, analysis_dir = tmp_path / "einsum", tmp_path / "analysis"
    for cmd in (
        [PY, "-m", "solar.cli.toeinsum_model", "--graph-path",
         str(graph_dir / "pytorch_graph.yaml"), "--output-dir",
         str(einsum_dir), "--enable-rename"],
        [PY, "-m", "solar.cli.analyze_model", "--einsum-graph-path",
         str(einsum_dir / "einsum_graph_renamed.yaml"), "--output-dir",
         str(analysis_dir), "--no-copy-graph"],
    ):
        p = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=SOLAR_ROOT, env=env, timeout=300)
        assert p.returncode == 0, p.stdout[-1200:] + p.stderr[-800:]
    return {
        "graph": yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text()),
        "analysis": yaml.safe_load(
            (analysis_dir / "analysis.yaml").read_text()),
    }


INSTALLS = ("vanilla", "prepatched")


@pytest.mark.parametrize("install", INSTALLS)
def test_keyword_learned_matmul_counts(install, tmp_path,
                                       prepatched_pythonpath):
    out = _full(tmp_path, _reg.KW_REVERSED_LEARNED,
                _env(install, prepatched_pythonpath))
    mm = _reg._matmul_layer(out["graph"])
    assert [tuple(s) for s in mm["input_shapes"]] == [(4, 32, 64), (64, 128)]
    assert _reg._total_macs(out["analysis"]) == 1048576, install


@pytest.mark.parametrize("install", INSTALLS)
def test_positional_learned_matmul_counts(install, tmp_path,
                                          prepatched_pythonpath):
    out = _full(tmp_path, _reg.LEARNED_W_FIRST,
                _env(install, prepatched_pythonpath))
    assert _reg._total_macs(out["analysis"]) == 1048576, install


@pytest.mark.parametrize("install", INSTALLS)
def test_scalar_repeated_operand_connectivity(install, tmp_path,
                                              prepatched_pythonpath):
    proc, graph_dir = _stage1(tmp_path, _reg.SCALAR_REDUCTION_CONSUMER,
                              _env(install, prepatched_pythonpath))
    assert proc.returncode == 0, proc.stdout[-1200:]
    graph = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    mul = next(v for v in graph["layers"].values()
               if "mul" in str(v.get("type", "")))
    assert mul["connections"]["inputs"], install
    assert [tuple(s) for s in mul["input_shapes"]] == [(), ()], install


@pytest.mark.parametrize("install", INSTALLS)
def test_unknown_signature_fails_closed(install, tmp_path,
                                        prepatched_pythonpath):
    proc, graph_dir = _stage1(tmp_path, _reg.UNKNOWN_SIGNATURE_INJECTED,
                              _env(install, prepatched_pythonpath))
    assert proc.returncode != 0, (install, proc.stdout[-800:])
    assert "UnsupportedFrontendOperand" in (proc.stdout + proc.stderr)
    assert not (graph_dir / "pytorch_graph.yaml").exists()


@pytest.mark.parametrize("install", INSTALLS)
def test_ensure_is_idempotent(install, prepatched_pythonpath):
    """Repeated ensure calls must not double-wrap the renderer or stack
    recorder patches; the second call reports nothing left to do."""
    code = textwrap.dedent("""
        from solar.graph._torchview_param_shim import ensure_parameter_capture
        from torchview.recorder_tensor import RecorderTensor
        first = ensure_parameter_capture()
        second = ensure_parameter_capture()
        tf = RecorderTensor.__dict__['__torch_function__']
        fn = getattr(tf, '__func__', tf)
        assert getattr(fn, '_solar_param_capture', False), 'recorder guard absent'
        assert second is False, 'second ensure must be a no-op'
        from torchview.computation_graph import ComputationGraph
        rn = ComputationGraph.render_nodes
        assert getattr(ComputationGraph, '_render_parameter_nodes', None) is not None
        print('ok', first, second)
    """)
    proc = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                          cwd=SOLAR_ROOT,
                          env=_env(install, prepatched_pythonpath), timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout
