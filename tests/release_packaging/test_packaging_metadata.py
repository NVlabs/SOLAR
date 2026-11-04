# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline packaging contract checks (no network, no provider calls, no torch import).

These tests pin the packaging defects fixed for the public candidate so they
cannot silently regress:

* one version source (``solar/__init__.py``), consistent with the built metadata;
* Apache-2.0 license metadata matching the LICENSE file (the old setup.py
  carried an MIT classifier);
* no placeholder project/contact metadata;
* every documented console script is declared and resolves to a module that
  defines ``main``;
* ``requirements.txt`` and ``pyproject.toml`` agree on the core dependencies;
* the install scripts only reference torchview patches that exist, and every
  shipped patch is applied by the install scripts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

MECHANICAL_SCRIPTS = {
    "solar-process-model": "solar.cli.process_model",
    "solar-toeinsum-model": "solar.cli.toeinsum_model",
    "solar-analyze-model": "solar.cli.analyze_model",
    "solar-predict-perf-model": "solar.cli.predict_perf_model",
    "solar-totimeloop": "solar.cli.totimeloop",
}
AGENTIC_SCRIPTS = {
    "solar-translate": "solar.cli.translate",
    "solar-toDSL-model": "solar.cli.toDSL_model",
}
RESOURCE_SCRIPTS = {"solar-resource": "solar.resources"}


@pytest.fixture(scope="module")
def project() -> dict:
    """Expanded pyproject [project] table as setuptools sees it (dynamic fields resolved)."""
    import setuptools
    if int(setuptools.__version__.split(".")[0]) < 61:
        pytest.fail(f"setuptools>=61 is required to read pyproject.toml (found {setuptools.__version__}); "
                    "this matches [build-system].requires")
    from setuptools.config.pyprojecttoml import read_configuration

    cfg = read_configuration(str(PYPROJECT), expand=True)
    return cfg["project"]


def _static_version() -> str:
    tree = ast.parse((ROOT / "solar" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("solar/__init__.py defines no __version__")


def test_single_version_source(project):
    assert project["version"] == _static_version()
    setup_py = (ROOT / "setup.py").read_text()
    assert "version=" not in setup_py, "setup.py must not carry its own version"
    # setup.py carries no metadata: only the build_py resource hook (cmdclass).
    assert re.search(r"setuptools\.setup\(\s*(cmdclass=\{[^}]*\}\s*)?\)", setup_py), (
        "setup.py must be a metadata-free shim (cmdclass only)")
    for forbidden in ("name=", "install_requires", "entry_points", "classifiers", "author"):
        assert forbidden not in setup_py.split("setuptools.setup(")[1], f"setup.py carries {forbidden}"


def test_license_metadata_matches_license_file(project):
    first_lines = (ROOT / "LICENSE").read_text().splitlines()[:3]
    assert any("Apache License" in line for line in first_lines)
    assert project["license"]["text"] == "Apache-2.0"
    classifiers = project["classifiers"]
    assert "License :: OSI Approved :: Apache Software License" in classifiers
    assert not any("MIT" in c for c in classifiers)


def test_no_placeholder_metadata(project):
    blob = PYPROJECT.read_text() + (ROOT / "setup.py").read_text()
    assert "example.com" not in blob
    assert "github.com/example" not in blob
    assert project["authors"] == [{"name": "NVIDIA Corporation"}]
    assert project["urls"]["Repository"] == "https://github.com/NVlabs/SOLAR"


def test_console_scripts_declared_and_resolvable(project):
    scripts = project["scripts"]
    expected = {**MECHANICAL_SCRIPTS, **AGENTIC_SCRIPTS, **RESOURCE_SCRIPTS}
    missing = {k: v for k, v in expected.items() if scripts.get(k) != f"{v}:main"}
    assert not missing, f"missing/mismatched console scripts: {missing}"
    for name, target in scripts.items():
        module, func = target.split(":")
        path = ROOT / Path(*module.split(".")).with_suffix(".py")
        assert path.is_file(), f"{name}: {path} does not exist"
        tree = ast.parse(path.read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body), (
            f"{name}: {module} defines no top-level {func}()"
        )


def test_documented_cli_commands_are_declared(project):
    scripts = set(project["scripts"])
    documented = set()
    for doc in [ROOT / "README.md", ROOT / "docs" / "USAGE.md", ROOT / "AGENTS.md"]:
        if doc.is_file():
            documented |= set(re.findall(r"^\s*(solar-[a-zA-Z-]+)\b", doc.read_text(), flags=re.M))
    undeclared = documented - scripts
    assert not undeclared, f"README/USAGE document undeclared commands: {sorted(undeclared)}"


def _requirements_specs() -> dict[str, str]:
    specs = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
        specs[name] = line
    return specs


def _spec_name(spec: str) -> str:
    return re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()


def test_requirements_cover_pyproject_dependencies(project):
    reqs = _requirements_specs()
    for dep in project["dependencies"]:
        name = _spec_name(dep)
        assert name in reqs, f"{dep} missing from requirements.txt"
        assert reqs[name].replace(" ", "") == dep.replace(" ", ""), (
            f"specifier drift for {name}: requirements.txt={reqs[name]!r} pyproject={dep!r}"
        )
    for extra in ("verify", "jax"):
        for dep in project["optional-dependencies"][extra]:
            assert _spec_name(dep) in reqs, f"[{extra}] {dep} missing from requirements.txt"


def test_install_scripts_and_patches_agree():
    patches_dir = ROOT / "patches"
    shipped = {p.name for p in patches_dir.glob("*.patch")}
    referenced = set()
    for script in ("install.sh", "install_uv.sh"):
        text = (ROOT / script).read_text()
        for name in re.findall(r"patches/([A-Za-z0-9_.-]+\.patch)", text):
            referenced.add(name)
            assert (patches_dir / name).is_file(), f"{script} references missing patch {name}"
    assert shipped == referenced, f"shipped {shipped} vs referenced {referenced}"
    for script in ("install.sh", "install_uv.sh"):
        text = (ROOT / script).read_text()
        assert "already applied or conflicts detected" not in text, (
            f"{script} must fail loudly when a patch does not apply"
        )


def test_torch_floor_is_the_evidenced_one(project):
    """The floor is the lowest torch with recorded full-suite evidence (2.7.1).
    Raising it needs evidence that an older version fails; lowering it needs a
    recorded run on the lower version (see pyproject.toml comment)."""
    torch_spec = [d for d in project["dependencies"] if _spec_name(d) == "torch"]
    assert torch_spec == ["torch>=2.7.1"], torch_spec


def test_package_discovery_excludes_generated_caches(tmp_path):
    """PKG4: the wheel must never pick up runtime handler caches written next
    to the package (solar_handlers_cache/, node_handlers_cache/) or tests."""
    import setuptools
    if int(setuptools.__version__.split(".")[0]) < 61:
        pytest.fail(f"setuptools>=61 is required (found {setuptools.__version__})")
    from setuptools.config.pyprojecttoml import read_configuration
    from setuptools.discovery import PEP420PackageFinder, PackageFinder

    cfg = read_configuration(str(PYPROJECT), expand=False)["tool"]["setuptools"]["packages"]["find"]
    assert cfg.get("namespaces") is False, "namespace discovery would admit cache dirs"
    (tmp_path / "solar").mkdir(); (tmp_path / "solar" / "__init__.py").write_text("")
    (tmp_path / "solar" / "cli").mkdir(); (tmp_path / "solar" / "cli" / "__init__.py").write_text("")
    (tmp_path / "solar_handlers_cache").mkdir(); (tmp_path / "solar_handlers_cache" / "unknown_op.py").write_text("")
    (tmp_path / "node_handlers_cache").mkdir(); (tmp_path / "node_handlers_cache" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir(); (tmp_path / "tests" / "__init__.py").write_text("")
    finder = PEP420PackageFinder if cfg.get("namespaces") else PackageFinder
    found = set(finder.find(str(tmp_path), include=cfg["include"], exclude=cfg.get("exclude", [])))
    assert found == {"solar", "solar.cli"}, found
    # And the bare pattern the baseline used would have admitted the cache:
    assert "solar_handlers_cache" in set(PEP420PackageFinder.find(str(tmp_path), include=["solar*"]))


def test_sdist_manifest_grafts_every_resource_tree():
    """MANIFEST.in must carry every RESOURCE_TREE so a wheel built from the
    sdist can package them (setup.py fails the build otherwise)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_res", ROOT / "solar" / "resources.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    manifest = (ROOT / "MANIFEST.in").read_text()
    grafted = set(re.findall(r"^graft\s+(\S+)", manifest, flags=re.M))
    included = set(re.findall(r"^include\s+(.+)$", manifest, flags=re.M))
    included = {f for line in included for f in line.split()}
    for rel in mod.RESOURCE_TREES:
        top = rel.split("/")[0]
        assert top in grafted or rel in included, f"MANIFEST.in does not carry {rel}"
        assert (ROOT / rel).exists(), f"RESOURCE_TREES names a missing path {rel}"
