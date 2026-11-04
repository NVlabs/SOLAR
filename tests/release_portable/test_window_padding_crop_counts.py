# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable analytic controls: window partition padding, per-window projection, crop.

The public tree withholds the maintainers' window-attention regression test
because its fixture bundle derives from benchmark content outside the public
allowlist.  These controls cover the same contract boundaries with content
derived only from the public DSL documentation (the scatter/crop pattern in
``docs/einsum_lib_prompt.md`` and the padded window partition already used
by ``tests/test_campaign_repairs.py``) and from hand-derived geometry.  No
benchmark definition, reference, workload row or measured value is used.

Everything in this module is static: the fixture DSL is parsed with the
public ``ast_parser`` and the converter's pure counting functions are applied
to hand-built call-site shapes.  No DSL program, model, reference or GPU
executes here.  The numerical padding/crop control that does execute the DSL
is delivered separately as ``pending_numerical_window_padding_crop.py`` in
this directory; it is not collected by default and its execution is pending
an independent run (see README.md).
"""
from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

import pytest

from solar.einsum_dsl import einsum_converter as ec
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.translate.reuse import guard_profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_padding_crop_fixture as fx  # noqa: E402

# Hand-derived geometry for window size 3 (literal expectations, not computed
# from the fixture): ceil-divided window counts per axis, total padded windows
# and the padded spatial extent.  Cases cover a single partial window, partial
# windows on both axes, the divisible boundary (no padding), exactly one full
# window, one divisible and one partial axis, and a larger synthetic shape.
EXPECTED = {
    (1, 1, 1): dict(nh=1, nw=1, windows=1, padded=(3, 3)),
    (1, 4, 7): dict(nh=2, nw=3, windows=6, padded=(6, 9)),
    (2, 6, 6): dict(nh=2, nw=2, windows=8, padded=(6, 6)),
    (1, 3, 3): dict(nh=1, nw=1, windows=1, padded=(3, 3)),
    (2, 5, 9): dict(nh=2, nw=3, windows=12, padded=(6, 9)),
    (2, 50, 70): dict(nh=17, nw=24, windows=816, padded=(51, 72)),
}
CASES = sorted(EXPECTED)
TOKENS = fx.WS * fx.WS


def _info():
    return extract_kernel_info(fx.DSL_PATH)


class TestGeometry:
    @pytest.mark.parametrize("b,h,w", CASES)
    def test_hand_derived_geometry_matches_literals(self, b, h, w):
        g = fx.geometry(b, h, w)
        e = EXPECTED[(b, h, w)]
        assert (g["nh"], g["nw"], g["windows"]) == (e["nh"], e["nw"], e["windows"])
        assert (g["padded_h"], g["padded_w"]) == e["padded"]
        assert g["tokens"] == TOKENS
        assert g["partial"] == (h % fx.WS != 0 or w % fx.WS != 0)


class TestStaticParse:
    def test_partition_loop_path_is_one_weighted_nest_with_ceil_specs(self):
        info = _info()["window_partition_pad"]
        assert info["param_names"] == ["X"]
        assert len(info["loop_paths"]) == 1
        path, weight = info["loop_paths"][0]
        assert weight == 1
        assert [d for d, _ in path] == ["B", "NH", "NW", "I", "J", "C"]
        specs = dict(path)
        assert specs["B"] is None and specs["C"] is None
        assert specs["I"] == fx.WS and specs["J"] == fx.WS
        # (X.shape[axis] + WS - 1) // WS, resolved at the call site.
        assert specs["NH"] == ("binop", "//", ("binop", "-", ("binop", "+", ("shape", "X", 1), fx.WS), 1), fx.WS)
        assert specs["NW"] == ("binop", "//", ("binop", "-", ("binop", "+", ("shape", "X", 2), fx.WS), 1), fx.WS)

    @pytest.mark.parametrize("b,h,w", CASES)
    def test_ceil_specs_evaluate_to_hand_derived_window_counts(self, b, h, w):
        specs = dict(_info()["window_partition_pad"]["loop_paths"][0][0])
        shapes = {"X": [b, h, w, fx.C]}
        e = EXPECTED[(b, h, w)]
        # binop_shapes: the correspondence the converter requires for
        # arithmetic specs (the internal converter) is the hand-built parameter map itself.
        assert ec._eval_size_spec(specs["NH"], shapes, shapes) == e["nh"]
        assert ec._eval_size_spec(specs["NW"], shapes, shapes) == e["nw"]

    @pytest.mark.parametrize("b,h,w", CASES)
    def test_declared_partition_output_is_padded_window_count_by_tokens(self, b, h, w):
        e = EXPECTED[(b, h, w)]
        shape = fx.declared_output_shape("window_partition_pad", {"X": [b, h, w, fx.C]})
        assert shape == [e["windows"], TOKENS, fx.C]

    @pytest.mark.parametrize("b,h,w", CASES)
    def test_declared_crop_output_is_the_original_shape(self, b, h, w):
        e = EXPECTED[(b, h, w)]
        shape = fx.declared_output_shape(
            "window_unpartition_crop",
            {"P": [e["windows"], TOKENS, fx.C], "X": [b, h, w, fx.C]})
        assert shape == [b, h, w, fx.C]


class TestConverterCounts:
    @pytest.mark.parametrize("b,h,w", CASES)
    def test_partition_counts_the_padded_domain(self, b, h, w):
        e = EXPECTED[(b, h, w)]
        k = fx.kernel_stand_in("window_partition_pad", {"X": [b, h, w, fx.C]})
        padded = e["windows"] * TOKENS * fx.C
        assert ec._kernel_loop_iterations(k, {})[:3] == (padded, padded, "paths")
        assert padded == b * e["padded"][0] * e["padded"][1] * fx.C

    @pytest.mark.parametrize("b,h,w", CASES)
    def test_projection_scales_with_padded_window_count(self, b, h, w):
        e = EXPECTED[(b, h, w)]
        k = fx.kernel_stand_in(
            "project_windows",
            {"P": [e["windows"], TOKENS, fx.C], "Wt": [fx.F, fx.C]})
        expected = e["windows"] * TOKENS * fx.F * fx.C
        assert ec._kernel_loop_iterations(k, {})[:3] == (expected, expected, "paths")
        floored = b * (h // fx.WS) * (w // fx.WS) * TOKENS * fx.F * fx.C
        if h % fx.WS or w % fx.WS:
            assert expected > floored  # a floored translation undercounts
        else:
            assert expected == floored  # divisible boundary: padding is empty

    @pytest.mark.parametrize("b,h,w", CASES)
    def test_crop_counts_the_original_domain_not_the_padded_one(self, b, h, w):
        e = EXPECTED[(b, h, w)]
        k = fx.kernel_stand_in(
            "window_unpartition_crop",
            {"P": [e["windows"], TOKENS, fx.C], "X": [b, h, w, fx.C]})
        crop = b * h * w * fx.C
        assert ec._kernel_loop_iterations(k, {})[:3] == (crop, crop, "paths")
        padded = b * e["padded"][0] * e["padded"][1] * fx.C
        if h % fx.WS or w % fx.WS:
            assert crop < padded
        else:
            assert crop == padded


# Same partition kernel with the window size passed as an ``int`` parameter.
# The parser cannot bind a scalar parameter, so NH/NW carry no size spec and
# the converter must refuse rather than count the partial nest.
REFUSED_INT_PARAM_DSL = '''from solar.einsum_lib import Tensor, Dim, kernel


@kernel
def window_partition_pad(X: Tensor("B,H,W,C", dtype="fp32"), ws: int) -> Tensor("BW,L,C"):
    nh = (X.shape[1] + ws - 1) // ws
    nw = (X.shape[2] + ws - 1) // ws
    P = Tensor(BW=X.shape[0] * nh * nw, L=ws * ws, C=X.shape[3], dtype="fp32")
    for b in Dim("B", parallel=True):
        for wh in Dim("NH", size=nh):
            for ww in Dim("NW", size=nw):
                for i in Dim("I", size=ws):
                    for j in Dim("J", size=ws):
                        for c in Dim("C"):
                            if wh * ws + i < X.shape[1]:
                                if ww * ws + j < X.shape[2]:
                                    P[(b * nh + wh) * nw + ww, i * ws + j, c] = X[b, wh * ws + i, ww * ws + j, c]
    return P
'''


class TestHardRefusal:
    def test_window_size_as_int_parameter_is_refused_not_undercounted(self, tmp_path):
        p = tmp_path / "int_param_window.py"
        p.write_text(REFUSED_INT_PARAM_DSL)
        info = extract_kernel_info(p)["window_partition_pad"]
        path, _ = info["loop_paths"][0]
        # an earlier change round 4 (TARGET-KB-MAXPOOL): a size over a scalar (int) parameter is no longer dropped to None at
        # parse time; it is a deferred spec naming the scalar leaf, resolved from the call's recorded
        # ``scalar_args``.  A trace that did not record them (this stand-in, like every sealed trace) is still
        # REFUSED, naming the parameter -- never undercounted.
        assert set(ec._scalar_leaves(dict(path)["NH"])) == {"ws"} and set(ec._scalar_leaves(dict(path)["NW"])) == {"ws"}
        k = fx.kernel_stand_in("window_partition_pad", {"X": [2, 5, 9, fx.C]}, info=info, source=p)
        with pytest.raises(ValueError, match="NH.*Expose its extent"):
            ec._kernel_loop_iterations(k, {})
        k.loop_nest = info["loop_nest"]
        with pytest.raises(ValueError, match=r"'NH' is sized by the scalar argument\(s\) 'ws'.*Expose its extent"):
            ec._kernel_loop_iterations(k, {})
        # the same call with the int argument recorded: ws=3 -> nh=2, nw=3; B*NH*NW*I*J*C statement iterations
        k.scalar_args = {"ws": 3}
        full = 2 * 2 * 3 * 3 * 3 * fx.C
        assert ec._kernel_loop_iterations(k, {})[:3] == (full, full, "paths")


class TestReuseGuard:
    def test_partition_and_crop_are_per_shape_revalidation_hazards(self):
        prof = guard_profile(fx.DSL_PATH)
        assert "floor_partition_extent" in prof.hazards
        # NH and NW of the partition kernel; the crop kernel sizes no Dim by them.
        assert len(prof.floor_partition_dims) == 2
        assert all('Dim(' in d for d in prof.floor_partition_dims)
        # The two padding guards compare against X.shape.
        assert len(prof.shape_branches) == 2
        # The crop reads the padded buffer through // and % index arithmetic.
        assert prof.div_mod_index and all("P[" in d for d in prof.div_mod_index)


class TestFixtureIntegrity:
    def test_fixture_sources_compile_without_executing(self):
        for path in (fx.DSL_PATH, HERE / "pending_numerical_window_padding_crop.py"):
            compile(path.read_text(), str(path), "exec")

    def test_pending_numerical_control_is_not_collected_by_default(self):
        name = "pending_numerical_window_padding_crop.py"
        assert (HERE / name).is_file()
        assert not fnmatch.fnmatch(name, "test_*.py") and not fnmatch.fnmatch(name, "*_test.py")
        root = HERE.parent.parent
        assert not re.search(r"^\[tool\.pytest", (root / "pyproject.toml").read_text(), re.M)
        assert not (root / "pytest.ini").exists() and not (root / "setup.cfg").exists()

    def test_fixture_declares_the_hand_derived_constants_only(self):
        src = fx.DSL_PATH.read_text()
        assert "_WS = 3" in src and '_FREE_CONSTANTS = ["_WS"]' in src
        for forbidden in ("examples_verified", "workload", "definition.json", "uuid"):
            assert forbidden not in src
