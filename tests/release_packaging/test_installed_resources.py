# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed-resource discovery, exercised from OUTSIDE the checkout.

PKG3/PKG-B2: a wheel install has no ``configs/``, ``docs/``, ``skills/`` or
``scripts/`` next to the package.  ``setup.py`` copies RESOURCE_TREES into
``solar/_data`` and ``solar.resources`` resolves them there.  The module-level
fixture builds a wheel from this tree with the host's setuptools (no network,
no build isolation), installs it into a temporary target directory and runs
every check in a subprocess whose cwd and PYTHONPATH exclude the checkout.
The in-process tests pin the lookup order with a fake layout.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # import solar.resources from THIS tree without importing solar


def _load_resources_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_solar_resources_under_test", ROOT / "solar" / "resources.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# In-process: lookup order with a fake layout (no torch, no build)
# --------------------------------------------------------------------------

def _fake_layout(tmp_path: Path, with_checkout: bool, with_data: bool):
    pkg = tmp_path / "site" / "solar"
    pkg.mkdir(parents=True)
    shutil.copy(ROOT / "solar" / "resources.py", pkg / "resources.py")
    if with_checkout:
        # resources.py sits in <checkout>/solar/, markers at <checkout>/
        (tmp_path / "site" / "pyproject.toml").write_text("")
        (tmp_path / "site" / "configs" / "arch").mkdir(parents=True)
        (tmp_path / "site" / "configs" / "arch" / "X.yaml").write_text("from: checkout\n")
    if with_data:
        (pkg / "_data" / "configs" / "arch").mkdir(parents=True)
        (pkg / "_data" / "configs" / "arch" / "X.yaml").write_text("from: packaged\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_fake_{with_checkout}_{with_data}", pkg / "resources.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_packaged_copy_is_used_when_no_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLAR_RESOURCE_ROOT", raising=False)
    mod = _fake_layout(tmp_path, with_checkout=False, with_data=True)
    assert mod.resource_path("configs", "arch", "X.yaml").read_text() == "from: packaged\n"
    assert mod.arch_config_path("X").read_text() == "from: packaged\n"
    assert mod.list_arch_configs() == ["X"]


def test_checkout_wins_over_packaged_copy(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLAR_RESOURCE_ROOT", raising=False)
    mod = _fake_layout(tmp_path, with_checkout=True, with_data=True)
    assert mod.resource_path("configs", "arch", "X.yaml").read_text() == "from: checkout\n"


def test_env_root_wins_over_everything(tmp_path, monkeypatch):
    mod = _fake_layout(tmp_path, with_checkout=True, with_data=True)
    env_root = tmp_path / "env_root"
    (env_root / "configs" / "arch").mkdir(parents=True)
    (env_root / "configs" / "arch" / "X.yaml").write_text("from: env\n")
    monkeypatch.setenv("SOLAR_RESOURCE_ROOT", str(env_root))
    assert mod.resource_path("configs", "arch", "X.yaml").read_text() == "from: env\n"
    kinds = [k for k, _ in mod.resource_roots()]
    assert kinds[0].startswith("env:")


def test_missing_resource_names_every_searched_root(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLAR_RESOURCE_ROOT", raising=False)
    mod = _fake_layout(tmp_path, with_checkout=False, with_data=False)
    with pytest.raises(mod.ResourceNotFound) as exc:
        mod.resource_path("configs", "arch", "B200.yaml")
    msg = str(exc.value)
    assert "configs/arch/B200.yaml" in msg and "checkout ->" in msg and "packaged ->" in msg
    assert "SOLAR_RESOURCE_ROOT" in msg
    assert mod.find_resource("configs", "arch", "B200.yaml") is None


def test_resource_trees_exist_in_this_tree():
    mod = _load_resources_module()
    for rel in mod.RESOURCE_TREES:
        assert (ROOT / rel).exists(), rel
    for required in ("configs", "skills", "scripts/verify_saved_ir.py", "scripts/trace_saved_ir.py",
                     "patches", "docs", "examples", "AGENTS.md"):
        assert required in mod.RESOURCE_TREES, required


# --------------------------------------------------------------------------
# Real wheel, installed into a temp dir, exercised from outside the checkout
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    work = tmp_path_factory.mktemp("solar-wheel")
    dist = work / "dist"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SOLAR_RESOURCE_ROOT")}
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                            "--no-index", "-w", str(dist), str(ROOT)],
                           capture_output=True, text=True, env=env, cwd=str(work), timeout=600)
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable on this host (pip wheel rc={build.returncode}):\n{build.stderr[-1500:]}")
    wheels = sorted(dist.glob("solar-*.whl"))
    assert len(wheels) == 1, wheels
    site = work / "site"
    inst = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                           "--target", str(site), str(wheels[0])],
                          capture_output=True, text=True, env=env, cwd=str(work), timeout=600)
    assert inst.returncode == 0, inst.stderr[-2000:]
    return {"wheel": wheels[0], "site": site, "work": work}


def _outside(installed_wheel, code: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``code`` with cwd outside the checkout and only the installed copy importable."""
    env = {k: v for k, v in os.environ.items() if k not in ("SOLAR_RESOURCE_ROOT",)}
    env["PYTHONPATH"] = str(installed_wheel["site"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(installed_wheel["work"]), env=env, timeout=600)


def test_wheel_carries_resources_and_no_cache_namespace(installed_wheel):
    names = zipfile.ZipFile(installed_wheel["wheel"]).namelist()
    assert not any(n.startswith(("solar_handlers_cache/", "node_handlers_cache/", "tests/")) for n in names)
    for member in ("solar/_data/configs/arch/B200.yaml", "solar/_data/skills/verified-translation/SKILL.md",
                   "solar/_data/scripts/verify_saved_ir.py", "solar/_data/scripts/trace_saved_ir.py",
                   "solar/_data/docs/einsum_lib_prompt.md", "solar/_data/patches/torchview-parameter-tensors.patch",
                   "solar/_data/AGENTS.md", "solar/_data/RESOURCE_MANIFEST.json"):
        assert member in names, member
    top = [n for n in names if n.endswith("top_level.txt")][0]
    assert zipfile.ZipFile(installed_wheel["wheel"]).read(top).decode().split() == ["solar"]


def test_installed_lookup_resolves_inside_the_install(installed_wheel):
    code = (
        "import json, solar.resources as r, solar\n"
        "p = r.resource_path('configs','arch','B200.yaml')\n"
        "print(json.dumps({'solar': solar.__file__, 'cfg': str(p), 'kinds': [k for k,_ in r.resource_roots()],"
        " 'skill': str(r.resource_path('skills','verified-translation','SKILL.md')),"
        " 'trace': str(r.resource_path('scripts','trace_saved_ir.py')), 'archs': r.list_arch_configs()}))\n"
    )
    proc = _outside(installed_wheel, code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    site = str(installed_wheel["site"])
    assert out["solar"].startswith(site), out
    for key in ("cfg", "skill", "trace"):
        assert out[key].startswith(site), out
    assert out["kinds"] == ["packaged"], out
    assert set(out["archs"]) >= {"A6000", "B200", "H100_PCIe", "H100_SXM", "H200_SXM", "L40S", "Jetson_Thor_T5000"}


def test_installed_arch_config_by_name_loads_same_bytes(installed_wheel):
    code = (
        "import solar.resources as r, hashlib\n"
        "print(hashlib.sha256(r.arch_config_path('B200').read_bytes()).hexdigest())\n"
    )
    proc = _outside(installed_wheel, code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    import hashlib
    assert proc.stdout.strip() == hashlib.sha256((ROOT / "configs" / "arch" / "B200.yaml").read_bytes()).hexdigest()


def test_installed_perf_model_resolves_arch_by_name(installed_wheel):
    """PKG-B2: ``--arch-config B200`` from a wheel install must load the config."""
    code = (
        "from solar.perf.perf_model import EinsumGraphPerfModel\n"
        "cfg = EinsumGraphPerfModel._load_arch_config(EinsumGraphPerfModel.__new__(EinsumGraphPerfModel), 'B200')\n"
        "assert cfg, 'empty config'\n"
        "from solar.analysis.model_analyzer import ModelAnalyzer\n"
        "cfg2 = ModelAnalyzer._load_arch_config(ModelAnalyzer.__new__(ModelAnalyzer), 'B200')\n"
        "assert cfg2 == cfg, 'analyzer and perf model disagree'\n"
        "print('ok', sorted(cfg)[:3])\n"
    )
    proc = _outside(installed_wheel, code)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_installed_lookup_fails_loudly_without_data(installed_wheel):
    """Remove the packaged copy: the lookup must raise, never fall back silently."""
    site = installed_wheel["site"]
    data = site / "solar" / "_data"
    backup = installed_wheel["work"] / "_data_backup"
    shutil.move(str(data), str(backup))
    try:
        code = (
            "import solar.resources as r\n"
            "try:\n    r.resource_path('configs','arch','B200.yaml')\nexcept r.ResourceNotFound as e:\n"
            "    print('REFUSED', str(e)[:80]); raise SystemExit(3)\nprint('UNEXPECTED')\n"
        )
        proc = _outside(installed_wheel, code)
        assert proc.returncode == 3 and "REFUSED" in proc.stdout, (proc.stdout, proc.stderr[-1500:])
    finally:
        shutil.move(str(backup), str(data))
