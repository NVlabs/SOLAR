# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verifier's module eviction must never evict the Python environment.

``_evict_unproven_cached_modules`` drops cached modules that live outside
the trusted roots so a reference re-executes its current on-disk bytes.  A
``pip install --user`` environment (the CI image, ``install.sh``) keeps
torch, numpy, pytest and solar's dependencies under the USER site directory,
outside every interpreter prefix.  Before the fix, the first in-process
verification in such a process evicted ``_pytest.*`` (detaching
``_pytest.assertion.util`` from its package) and torch itself; pytest then
crashed in session teardown (ci-fail-19).

The tests use stand-in modules whose ``__file__`` points into the
environment directories, so they hold whichever way the machine is laid out.
"""
import site
import sys
import sysconfig
import types
from pathlib import Path

import pytest

import solar.translate._runner as R


def _stand_in(monkeypatch, name, file):
    mod = types.ModuleType(name)
    mod.__file__ = str(file)
    # a copy: the eviction must not damage the real interpreter state
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    sys.modules[name] = mod
    return mod


def test_trusted_roots_cover_site_packages_and_user_site():
    roots = R._safe_roots()
    for d in [sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"],
              site.getusersitepackages(), *site.getsitepackages()]:
        d = Path(d).resolve()
        assert any(root == d or root in d.parents for root in roots), d


def test_user_site_module_survives_eviction(monkeypatch, tmp_path):
    usersite = tmp_path / "userbase" / "lib" / "site-packages"
    usersite.mkdir(parents=True)
    # The REAL environment stays trusted (the harness may itself live in the
    # real user site): the fake user site is added, the real one is kept as
    # an ordinary site directory.  Eviction detaches evicted children from
    # retained parents, so untrusting the real harness would damage live
    # module objects even on a copy of sys.modules.
    real_sites = list(site.getsitepackages()) + [site.getusersitepackages()]
    monkeypatch.setattr(site, "getsitepackages", lambda *a, **k: real_sites)
    monkeypatch.setattr(site, "getusersitepackages", lambda *a, **k: str(usersite))
    mod = _stand_in(monkeypatch, "_pytest_stand_in.main", usersite / "_pytest_stand_in" / "main.py")

    evicted = R._evict_unproven_cached_modules()

    assert sys.modules.get("_pytest_stand_in.main") is mod
    assert not any(e.startswith("_pytest_stand_in") for e in evicted)


def test_purelib_module_survives_eviction(monkeypatch):
    purelib = Path(sysconfig.get_paths()["purelib"])
    mod = _stand_in(monkeypatch, "env_stand_in_pkg", purelib / "env_stand_in_pkg" / "__init__.py")

    evicted = R._evict_unproven_cached_modules()

    assert sys.modules.get("env_stand_in_pkg") is mod
    assert not any(e.startswith("env_stand_in_pkg") for e in evicted)


def test_host_test_harness_survives_eviction(monkeypatch):
    """Whatever the layout of THIS machine, the modules pytest runs on must
    still be registered, and intact, after an eviction."""
    import _pytest.assertion
    import _pytest.main
    import pluggy
    held = {n: sys.modules[n] for n in list(sys.modules)
            if n == "pytest" or n == "pluggy" or n.startswith(("_pytest.", "pluggy."))}
    monkeypatch.setattr(sys, "modules", dict(sys.modules))

    R._evict_unproven_cached_modules()

    for name, mod in held.items():
        assert sys.modules.get(name) is mod, f"{name} was evicted"
    assert _pytest.assertion.util is sys.modules["_pytest.assertion.util"]
    assert sys.modules["_pytest.main"].Session is _pytest.main.Session


def test_reference_helper_outside_the_environment_is_still_evicted(monkeypatch, tmp_path):
    helper = tmp_path / "adapter_dir" / "ref_helper.py"
    helper.parent.mkdir()
    helper.write_text("")
    _stand_in(monkeypatch, "ref_helper", helper)

    evicted = R._evict_unproven_cached_modules()

    assert "ref_helper" not in sys.modules
    assert any(e.startswith("ref_helper ") for e in evicted)
