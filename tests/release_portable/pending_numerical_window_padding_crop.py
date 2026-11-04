# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PENDING INDEPENDENT EXECUTION: numerical padding/crop control for the portable window fixture.

This module is deliberately NOT collected by default (its name does not match
``test_*.py``).  It executes the fixture DSL in
``fixtures/window_padding_crop_einsum_lib.py`` through
``solar.einsum_lib.verify.execute_translated`` and compares the padded
windows, the per-window projection and the crop/unpartition against a
reference built only from public PyTorch calls (``F.pad``, reshape/permute
window partition, ``F.linear``, ``[:, :h, :w]`` crop).

It was authored under a release task that forbids DSL
execution, so it has not been run.  Its first execution and the receipt for it
belong to an independent owner.  Until that run is recorded, the tree makes no
numerical padding/crop claim (the static controls in
``test_window_padding_crop_counts.py`` cover geometry and counts only).

Run explicitly, from the checkout root in the approved sourced environment:

    PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \\
      python -m pytest -q -p no:cacheprovider \\
      tests/release_portable/pending_numerical_window_padding_crop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from solar.einsum_lib.verify import execute_translated

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_padding_crop_fixture as fx  # noqa: E402

# Partial windows on one or both axes, the divisible boundary, and one exact window.
SHAPES = [(1, 1, 1), (1, 4, 7), (2, 6, 6), (1, 3, 3), (2, 5, 9)]


def _torch_reference(torch, x, wt, p, g):
    """Public-API reference: zero-pad bottom/right, partition, project, unpartition, crop."""
    F = torch.nn.functional
    b, h, w, c = x.shape
    ws = fx.WS
    pad_h, pad_w = g["padded_h"] - h, g["padded_w"] - w
    xp = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # (C: none, W: right, H: bottom)
    windows = (xp.reshape(b, g["nh"], ws, g["nw"], ws, c)
                 .permute(0, 1, 3, 2, 4, 5)
                 .reshape(g["windows"], ws * ws, c))
    proj = F.linear(windows, wt)
    unpart = (p.reshape(b, g["nh"], g["nw"], ws, ws, c)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, g["padded_h"], g["padded_w"], c))
    crop = unpart[:, :h, :w] + x
    return windows, proj, crop


@pytest.mark.parametrize("b,h,w", SHAPES)
def test_padding_projection_and_crop_match_the_torch_reference(b, h, w):
    torch = pytest.importorskip("torch")
    g = fx.geometry(b, h, w)
    rng = np.random.default_rng(20261025)
    inputs = {
        "x": rng.normal(size=(b, h, w, fx.C)),
        "wt": rng.normal(size=(fx.F, fx.C)),
        # Independent window buffer so the crop mapping is checked, not just the identity.
        "p": rng.normal(size=(g["windows"], g["tokens"], fx.C)),
    }
    actual, _ = execute_translated(fx.DSL_PATH.read_text(),
                                   {k: v.shape for k, v in inputs.items()},
                                   all_inputs=inputs, fast=False)
    windows, proj, crop = _torch_reference(
        torch, torch.from_numpy(inputs["x"]), torch.from_numpy(inputs["wt"]),
        torch.from_numpy(inputs["p"]), g)
    assert actual["windows"].shape == (g["windows"], g["tokens"], fx.C)
    assert actual["proj"].shape == (g["windows"], g["tokens"], fx.F)
    assert actual["crop"].shape == (b, h, w, fx.C)
    np.testing.assert_allclose(actual["windows"], windows.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual["proj"], proj.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual["crop"], crop.numpy(), rtol=1e-12, atol=1e-12)
    if g["partial"]:
        # Padded positions are exactly zero in the DSL output.
        mask = np.ones((b, g["padded_h"], g["padded_w"], fx.C), bool)
        mask[:, :h, :w] = False
        padded_tokens = (mask.reshape(b, g["nh"], fx.WS, g["nw"], fx.WS, fx.C)
                             .transpose(0, 1, 3, 2, 4, 5)
                             .reshape(g["windows"], g["tokens"], fx.C))
        assert np.all(actual["windows"][padded_tokens] == 0.0)
