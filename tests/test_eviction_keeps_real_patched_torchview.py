# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The eviction step must keep the REAL torchview that the mechanical frontend patched.

``tests/test_verify_keeps_patched_torchview.py`` proves the exemption with
stand-in modules.  This control binds the actual interpreter state instead:
whichever way torchview is installed on this machine (inside ``sys.prefix``
as a wheel, or as an editable/source clone outside it, which is what
``install.sh`` produces), the module objects that
``solar.graph.pytorch_processor`` imported and patched must be the very
objects still registered in ``sys.modules`` after
``_evict_unproven_cached_modules`` runs.  Before the exemption, an editable
torchview outside the trusted roots was evicted here and later mechanical
extraction in the same process silently lost ``raw_attributes`` capture (the
order-dependent failures preserved in the release evidence).

No DSL program, reference or model is executed: the eviction function is
called directly.  On a wheel install torchview lies inside the prefix and is
never a candidate, so the assertions hold trivially; the test records which
layout it exercised in its own output so a log shows what was covered.
"""
import sys

import pytest

import solar.translate._runner as R


def _torchview_modules():
    return {n: m for n, m in sys.modules.items()
            if n == "torchview" or n.startswith("torchview.")}


def test_real_patched_torchview_survives_eviction(capsys):
    torchview = pytest.importorskip("torchview")
    import solar.graph.pytorch_processor as pp  # imports and patches torchview

    before = _torchview_modules()
    assert "torchview" in before and len(before) >= 2
    held = {name: getattr(pp, name) for name in dir(pp)
            if getattr(getattr(pp, name), "__name__", "").startswith("torchview")}
    inside_prefix = any(R._inside(torchview.__file__, root) for root in R._safe_roots())
    layout = "inside trusted roots (wheel/site-packages)" if inside_prefix \
        else "outside trusted roots (editable or source clone)"

    evicted = R._evict_unproven_cached_modules()

    after = _torchview_modules()
    assert set(after) >= set(before), sorted(set(before) - set(after))
    for name, mod in before.items():
        assert after[name] is mod, f"{name} was replaced by a re-imported object"
    for name, obj in held.items():
        assert sys.modules.get(obj.__name__) is obj, f"pytorch_processor.{name} holds a stale module"
    assert not any(e.split(" ")[0] == "torchview" or e.startswith("torchview.") for e in evicted)
    print(f"torchview layout exercised: {layout}: {torchview.__file__}")
