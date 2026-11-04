# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable window partition (zero padding), per-window projection and crop.

Public-safe fixture for ``tests/release_portable``.  Authored from the public
DSL documentation only: the padded window partition follows the SAM-style
pattern already used by ``tests/test_campaign_repairs.py`` and the crop
follows the scatter/crop pattern in ``docs/einsum_lib_prompt.md``.  Window
size, channel and feature counts are hand-chosen constants; no benchmark
definition, benchmark case, reference program or measured value is involved.

The window size is a module constant exposed through ``_FREE_CONSTANTS`` so
the ceil-divided window counts resolve statically at every call site.  The
same kernel with the window size as an ``int`` parameter is refused by the
converter (see the tests); that refusal is the intended behaviour.

Geometry (hand-derived): for an input ``B,H,W,C`` and window ``WS``,
``NH = ceil(H / WS)``, ``NW = ceil(W / WS)``; the partition writes
``B * NH * NW`` windows of ``WS * WS`` tokens, zero where the window extends
past ``H`` or ``W``; the projection scales with that padded window count; the
crop reads the padded buffer back into the original ``B,H,W,C`` domain and
adds the residual, so its loop count is the original domain, not the padded
one.
"""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_WS = 3
_FREE_CONSTANTS = ["_WS"]


@kernel
def window_partition_pad(X: Tensor("B,H,W,C", dtype="fp32")) -> Tensor("BW,L,C"):
    """Non-overlapping WSxWS windows; the bottom/right border is zero-padded."""
    nh = (X.shape[1] + _WS - 1) // _WS
    nw = (X.shape[2] + _WS - 1) // _WS
    P = Tensor(BW=X.shape[0] * nh * nw, L=_WS * _WS, C=X.shape[3], dtype="fp32")
    for b in Dim("B", parallel=True):
        for wh in Dim("NH", size=nh):
            for ww in Dim("NW", size=nw):
                for i in Dim("I", size=_WS):
                    for j in Dim("J", size=_WS):
                        for c in Dim("C"):
                            if wh * _WS + i < X.shape[1]:
                                if ww * _WS + j < X.shape[2]:
                                    P[(b * nh + wh) * nw + ww, i * _WS + j, c] = X[b, wh * _WS + i, ww * _WS + j, c]
    return P


@kernel
def project_windows(P: Tensor("BW,L,C", dtype="fp32"),
                    Wt: Tensor("F,C", role="weight", dtype="fp32")) -> Tensor("BW,L,F"):
    """Per-token projection of every padded window (work scales with BW)."""
    Q = Tensor(BW=P.shape[0], L=P.shape[1], F=Wt.shape[0], dtype="fp32")
    for bw in Dim("BW", parallel=True):
        for l in Dim("L"):
            for f in Dim("F"):
                for c in Dim("C"):
                    Q[bw, l, f] += P[bw, l, c] * Wt[f, c]
    return Q


@kernel
def window_unpartition_crop(P: Tensor("BW,L,C", dtype="fp32"),
                            X: Tensor("B,H,W,C", dtype="fp32")) -> Tensor("B,H,W,C"):
    """Scatter the windows back, crop to the original H,W and add the residual."""
    nh = (X.shape[1] + _WS - 1) // _WS
    nw = (X.shape[2] + _WS - 1) // _WS
    Y = Tensor(B=X.shape[0], H=X.shape[1], W=X.shape[2], C=X.shape[3], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            for w in Dim("W"):
                for c in Dim("C"):
                    Y[b, h, w, c] = P[(b * nh + h // _WS) * nw + w // _WS, (h % _WS) * _WS + w % _WS, c] + X[b, h, w, c]
    return Y


def model(x, wt, p):
    windows = window_partition_pad(x)
    return windows, project_windows(windows, wt), window_unpartition_crop(p, x)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,H,W,C", dtype="fp32")
    wt = Tensor.from_numpy(inputs["wt"], "F,C", dtype="fp32")
    p = Tensor.from_numpy(inputs["p"], "BW,L,C", dtype="fp32")
    windows, proj, crop = model(x, wt, p)
    return {"windows": windows.data, "proj": proj.data, "crop": crop.data}
