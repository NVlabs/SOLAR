# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor-core rate selection: dense vs sparse peaks (GitHub issue #14).

A hardware profile (``configs/arch/<name>.yaml``) declares, per tensor-core
precision ``<p>``, BOTH peak figures the vendor publishes --
``peak_TFLOPS_<p>_dense`` / ``peak_TFLOPS_<p>_sparse`` -- and the derived rates
``MAC_per_cycle_<p>_tc_dense`` / ``MAC_per_cycle_<p>_tc_sparse``
(= peak x 1e12 / 2 FLOP per MAC / (freq_GHz x 1e9)).  The perf model prices at
ONE of the two sets, chosen by ``sol_mode``:

* ``"sparse"`` (the DEFAULT headline SOL): the structured-sparsity peak; a
  lower bound that no dense kernel can reach;
* ``"dense"``: the dense peak; the more realistic estimate for dense work.

Every projection reports both (``sol_sparse`` / ``sol_dense`` in the perf
record); the mode only decides which set the top-level fields carry.

Backward compatibility: a profile that spells a rate with the legacy bare key
``MAC_per_cycle_<p>_tc`` declares a DENSE rate (that was always its meaning).
Such a profile has no sparse figure, so sparse mode falls back to the dense
rate for that precision and the fallback is recorded
(``sol_rate_resolution.precisions.<p>.sparse_fallback_to_dense``).  A bare key
set beside explicit keys (a caller's legacy override on a shipped profile) is
the single rate for both sets and is recorded as ``bare_key_overrides_explicit``.
The SM (CUDA-core) rate ``MAC_per_cycle_fp32_sm`` has no sparsity variant.

The resolver copies raw profile values by KEY PRESENCE only; it never compares,
converts or tests a rate value (the perf model validates every rate it applies
at its rate boundary, where a malformed value becomes a structured refusal).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

SOL_MODES = ("sparse", "dense")
DEFAULT_SOL_MODE = "sparse"

_TC_KEY = re.compile(r"^MAC_per_cycle_(?P<p>[A-Za-z0-9]+)_tc(?:_(?P<kind>dense|sparse))?$")

SOL_MODE_NOTES = {
    "sparse": ("sparse SOL: tensor-core MACs priced at the vendor's structured-sparsity peak; a lower bound that dense "
               "kernels cannot reach; the default headline SOL"),
    "dense": ("dense SOL: tensor-core MACs priced at the dense peak; dense SOL is the more realistic estimate for dense "
              "work (sparse SOL is a lower bound)"),
}


def check_sol_mode(sol_mode: Any) -> str:
    if not isinstance(sol_mode, str) or sol_mode not in SOL_MODES:
        raise ValueError(f"sol_mode must be one of {SOL_MODES}, got {sol_mode!r}")
    return sol_mode


def tensor_core_precisions(profile: Dict[str, Any]) -> List[str]:
    """Sorted precisions ``<p>`` for which the profile declares any ``MAC_per_cycle_<p>_tc[_dense|_sparse]`` key."""
    found = set()
    for key in profile:
        m = _TC_KEY.match(str(key))
        if m:
            found.add(m.group("p"))
    return sorted(found)


def resolve_arch_rates(profile: Dict[str, Any], sol_mode: str = DEFAULT_SOL_MODE) -> Dict[str, Any]:
    """A NEW flat profile view where ``MAC_per_cycle_<p>_tc`` holds the rate for ``sol_mode``.

    The view keeps every key of ``profile`` (explicit ``_dense`` / ``_sparse`` keys included), fills the
    missing explicit spelling for every tensor-core precision from what the profile does declare, sets
    the bare ``MAC_per_cycle_<p>_tc`` key to the selected set's value and adds ``sol_rate_resolution``:
    ``{"sol_mode", "precisions": {<p>: {"applied_key", "dense_from", "sparse_from",
    "sparse_fallback_to_dense"[, "bare_key_overrides_explicit"]}}}``.  ``profile`` is not mutated.

    Precedence: a BARE ``MAC_per_cycle_<p>_tc`` key, when present, is the legacy single-rate spelling and
    is used for BOTH sets (it is a dense figure; sparse falls back to it) -- also when explicit keys are
    present beside it, so a caller that overrides the legacy key on a shipped profile still sees its
    override applied (recorded as ``bare_key_overrides_explicit``).  Without a bare key the explicit keys
    are used; a profile with only one explicit spelling uses it for both sets.  Shipped profiles carry
    explicit keys only (tests/test_arch_configs.py).
    """
    mode = check_sol_mode(sol_mode)
    view: Dict[str, Any] = dict(profile)
    precisions: Dict[str, Dict[str, Any]] = {}
    for p in tensor_core_precisions(profile):
        bare, dense_k, sparse_k = f"MAC_per_cycle_{p}_tc", f"MAC_per_cycle_{p}_tc_dense", f"MAC_per_cycle_{p}_tc_sparse"
        if bare in profile:                         # legacy single rate: dense, and the sparse fallback
            dense_from = sparse_from = bare
        else:
            dense_from = dense_k if dense_k in profile else sparse_k
            sparse_from = sparse_k if sparse_k in profile else dense_k
        view[dense_k] = profile[dense_from]
        view[sparse_k] = profile[sparse_from]
        applied = dense_from if mode == "dense" else sparse_from
        view[bare] = profile[applied]
        precisions[p] = {"applied_key": applied, "dense_from": dense_from, "sparse_from": sparse_from,
                         "sparse_fallback_to_dense": sparse_from != sparse_k,
                         **({"bare_key_overrides_explicit": True} if bare in profile and (dense_k in profile or sparse_k in profile) else {})}
    view["sol_rate_resolution"] = {"sol_mode": mode, "precisions": precisions}
    return view


def applied_key(view: Dict[str, Any], bare_key: str) -> str:
    """The explicit profile key the view's bare ``bare_key`` was taken from (``bare_key`` itself for non-tensor-core keys)."""
    m = _TC_KEY.match(bare_key)
    if m and m.group("kind") is None:
        rec = (view.get("sol_rate_resolution") or {}).get("precisions", {}).get(m.group("p"))
        if rec:
            return rec["applied_key"]
    return bare_key


def mode_key(view: Dict[str, Any], bare_key: str, mode: str) -> str:
    """The explicit key holding ``bare_key``'s rate for ``mode`` (``*_from`` of the resolution); non-tensor-core keys map to themselves."""
    m = _TC_KEY.match(bare_key)
    if m and m.group("kind") is None:
        rec = (view.get("sol_rate_resolution") or {}).get("precisions", {}).get(m.group("p"))
        if rec:
            return rec["dense_from"] if mode == "dense" else rec["sparse_from"]
    return bare_key


__all__ = ["SOL_MODES", "DEFAULT_SOL_MODE", "SOL_MODE_NOTES", "check_sol_mode", "tensor_core_precisions",
           "resolve_arch_rates", "applied_key", "mode_key"]
