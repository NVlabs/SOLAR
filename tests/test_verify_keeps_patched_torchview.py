# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""verify() must not evict solar's own patched torchview from sys.modules.

``_evict_unproven_cached_modules`` drops every cached module outside the
trusted roots so a reference re-executes its current on-disk bytes.  torchview
is not reference code: solar imports it and patches it in-process once
(``ensure_parameter_capture``).  When torchview is installed outside
sys.prefix and the solar checkout (a ``pip install --user`` copy or a source
clone) the eviction used to drop it, the next import produced a pristine
torchview, and every later trace in the same process silently lost its
parameter operands, dtypes and raw attributes.

The tests use stand-in modules located outside the trusted roots, so they
behave the same whichever way torchview is installed on the machine.
"""
import sys
import types

import pytest

import solar.translate._runner as R


def _stand_in(monkeypatch, tmp_path, name):
    path = tmp_path / (name.replace(".", "_") + ".py")
    path.write_text("")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    # Work on a copy so the eviction cannot damage the real interpreter state.
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    sys.modules[name] = mod
    return mod


def _stand_in_torchview(monkeypatch, tmp_path):
    """A cached ``torchview`` package living outside the trusted roots."""
    pkg = tmp_path / "site" / "torchview"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "recorder_tensor.py").write_text("")
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    top = types.ModuleType("torchview")
    top.__file__ = str(pkg / "__init__.py")
    sub = types.ModuleType("torchview.recorder_tensor")
    sub.__file__ = str(pkg / "recorder_tensor.py")
    sys.modules["torchview"] = top
    sys.modules["torchview.recorder_tensor"] = sub
    return top, sub


def test_torchview_survives_eviction_when_outside_trusted_roots(
        monkeypatch, tmp_path):
    top, sub = _stand_in_torchview(monkeypatch, tmp_path)

    R._evict_unproven_cached_modules()

    assert sys.modules.get("torchview") is top
    assert sys.modules.get("torchview.recorder_tensor") is sub


def test_external_helper_registered_under_torchview_namespace_is_evicted(
        monkeypatch, tmp_path):
    # A reference helper that extends torchview.__path__ is not the patched
    # dependency: it must be evicted so the receipt binds the bytes it runs.
    top, _ = _stand_in_torchview(monkeypatch, tmp_path)
    ext = tmp_path / "external"
    ext.mkdir()
    (ext / "review_reference.py").write_text("")
    helper = types.ModuleType("torchview.review_reference")
    helper.__file__ = str(ext / "review_reference.py")
    sys.modules["torchview.review_reference"] = helper

    evicted = R._evict_unproven_cached_modules()

    assert "torchview.review_reference" not in sys.modules
    assert any(e.startswith("torchview.review_reference ") for e in evicted)
    assert sys.modules.get("torchview") is top


def test_lookalike_package_name_is_still_evicted(monkeypatch, tmp_path):
    mod = _stand_in(monkeypatch, tmp_path, "torchview_lookalike")

    evicted = R._evict_unproven_cached_modules()

    assert sys.modules.get("torchview_lookalike") is None
    assert any(e.startswith("torchview_lookalike ") for e in evicted)


def test_unrelated_untrusted_module_is_still_evicted(monkeypatch, tmp_path):
    _stand_in(monkeypatch, tmp_path, "ref_helper_mod")

    evicted = R._evict_unproven_cached_modules()

    assert "ref_helper_mod" not in sys.modules
    assert any(e.startswith("ref_helper_mod ") for e in evicted)


@pytest.mark.parametrize("spelling", ["from_submodule", "from_package"])
@pytest.mark.parametrize("changed_on_disk", [True, False])
def test_external_torchview_namespace_helper_never_yields_a_receipt(
        monkeypatch, tmp_path, request, changed_on_disk, spelling):
    """End to end: a cached external ``torchview.*`` helper must be re-executed
    from its current bytes AND must not qualify for a receipt, whether or not
    the numbers happen to match, and whichever way the reference imports it
    (``from torchview.helper import run`` / ``from torchview import helper``,
    the latter resolving through the parent package attribute).  Uses the
    real (patched) torchview."""
    import importlib
    import json

    torchview = pytest.importorskip("torchview")
    import test_router_receipts as fx
    from solar.translate import finalize_verify, verify
    from solar.translate.receipts import ReceiptRefusal

    name = "torchview.review_reference"
    child = "review_reference"

    def _scrub():  # leave neither the cache entry nor the attribute behind
        sys.modules.pop(name, None)
        if hasattr(torchview, child):
            delattr(torchview, child)

    _scrub()
    request.addfinalizer(_scrub)
    real = {n: m for n, m in sys.modules.items()
            if n == "torchview" or n.startswith("torchview.")}
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    good = fx._reference_run_text()
    helper = external / "review_reference.py"
    helper.write_text(good)
    monkeypatch.setattr(torchview, "__path__", list(torchview.__path__))
    torchview.__path__.append(str(external))
    stale = importlib.import_module(name)   # cached, as a stale helper would be
    assert getattr(torchview, child) is stale

    adapter = fx._HELPER_ADAPTER.format(helper=name, extra_path="")
    if spelling == "from_package":
        adapter = adapter.replace(
            "from torchview.review_reference import run\\n",
            "from torchview import review_reference as _m\\nrun = _m.run\\n")
        assert "_m.run" in adapter
    source = project / "adapter.py"
    source.write_text(adapter)
    dsl = project / "solar_model_einsum_lib.py"
    dsl.write_bytes((fx.EXAMPLE / "solar_model_einsum_lib.py").read_bytes())
    definition = project / "definition.json"
    definition.write_text(json.dumps(fx.DEFN))
    helper.write_text(good + fx._ZERO_WRAP if changed_on_disk else good)

    result = verify(source, dsl, source_type="cuda", quiet=True)

    assert result.verified is (not changed_on_disk)
    assert any(e.startswith(name + " ")
               for e in result.execution.evicted_cached_modules)
    assert any(e.startswith(name + " ")
               for e in result.execution.unsupported_dependencies)
    assert all(sys.modules.get(k) is m for k, m in real.items())

    receipt = project / "receipt.json"
    with pytest.raises(ReceiptRefusal, match="provenance"):
        fx._emit(result, project, receipt, source_path=source,
                 dsl_path=dsl, definition_path=definition)
    assert not receipt.exists()
    code = finalize_verify(
        result, model_path=source, ir_path=dsl, strict=False,
        emit_receipt=receipt, definition_dir=project, dtype="fp16")
    assert code == 3 and not receipt.exists()
