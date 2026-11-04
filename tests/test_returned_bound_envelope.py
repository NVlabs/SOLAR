# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""bound-envelope-1 (the review, round 3): subtract ONLY proven-covered byte regions.

Round 3 subtracted every bound leaf's interval from the unbound footprint; above the enumeration cap a leaf's region
was a BOUNDING interval that includes byte holes, so ``y[::2], y[1:2]`` (one hole byte returned unbound) billed
4,194,308 B, labelled the plain return ``alias_of_bound_output`` and dropped the incomplete status, and the disjoint
``y[::2], y[1::2]`` billed 4,194,312 B instead of 8,388,616 B.  Now a region is exact -- an interval list or a strided
LATTICE ``start + k*stride .. +run`` -- or it is approximate; unions and differences are computed analytically on a
common period grid; an approximate unbound region bills its extent (over-count), an approximate bound region covers
nothing and never makes an alias.  the reviewer's two real-capture counterexamples run verbatim (fixtures), plus analytic
controls that need no capture.
"""
import os
import pathlib
import subprocess
import sys

import pytest

from solar.einsum.pytorch_to_einsum import (
    _returned_leaf_region, _returned_group_footprint, _lattice_rects, _interval_rects,
    _rect_difference_area, _returned_leaf_byte_intervals, _subtract_byte_intervals, _merge_byte_intervals,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "returned_outputs_review7"
N = (1 << 20) + 1          # above the enumeration cap of 1 << 20 runs


def _ident(offset, shape, stride, esize=4, nbytes=None):
    return {"storage_offset": offset, "shape": shape, "stride": stride, "element_size": esize, "storage_nbytes": nbytes}


def _enum_bytes(offset, shape, stride, esize=4):
    """Brute-force oracle for small views."""
    import itertools
    out = set()
    for idx in itertools.product(*(range(n) for n in shape)):
        start = (offset + sum(i * s for i, s in zip(idx, stride))) * esize
        out.update(range(start, start + esize))
    return out


class TestRegionRepresentation:
    def test_large_stride2_view_is_an_exact_lattice_not_a_bounding_interval(self):
        r = _returned_leaf_region(_ident(0, [N], [2]), {})
        assert r["exact"] and r["method"] == "strided_lattice_exact" and r["lattice"] == (0, 8, N, 4) and r["bytes"] == 4 * N
        # the legacy helper keeps its documented bounding behaviour (the reviewer's region controls rely on it)
        iv, m = _returned_leaf_byte_intervals(_ident(0, [N], [2]), {})
        assert m == "bounding_interval" and iv == [(0, 8 * N - 4)]

    def test_small_views_stay_exact_intervals(self):
        r = _returned_leaf_region(_ident(1, [1], [1]), {})
        assert r["exact"] and r["intervals"] == [(4, 8)] and r["lattice"] is None

    def test_two_outer_strides_above_the_cap_remain_approximate(self):
        big = 1 << 11
        r = _returned_leaf_region(_ident(0, [big, big], [2 * big * 2, 2]), {})
        assert not r["exact"] and r["method"] == "bounding_interval"

    def test_legacy_identity_is_approximate(self):
        r = _returned_leaf_region({"storage_offset": 3, "numel": 4, "element_size": 4}, {"shape": [4]})
        assert not r["exact"] and r["method"] == "contiguous_assumed_legacy_identity"

    def test_overlapping_runs_collapse_to_one_interval(self):
        # stride (in bytes) shorter than the run: the lattice IS a contiguous span
        r = _returned_leaf_region(_ident(0, [N, 3], [2, 1]), {})
        assert r["exact"] and r["intervals"] == [(0, (2 * (N - 1) + 3) * 4)]


class TestLatticeAlgebra:
    @pytest.mark.parametrize("lattice,period", [((0, 8, 5, 4), 8), ((4, 8, 5, 4), 8), ((0, 8, 5, 4), 16), ((12, 8, 7, 4), 24), ((6, 8, 5, 4), 8)])
    def test_lattice_rectangles_cover_exactly_the_lattice_bytes(self, lattice, period):
        start, stride, count, run = lattice
        expected = {start + k * stride + b for k in range(count) for b in range(run)}
        got = {p * period + o for (p0, p1, o0, o1) in _lattice_rects(lattice, period) for p in range(p0, p1) for o in range(o0, o1)}
        assert got == expected

    @pytest.mark.parametrize("a,b,period", [(3, 29, 8), (0, 8, 8), (8, 16, 8), (5, 6, 8), (0, 100, 24)])
    def test_interval_rectangles_cover_exactly_the_interval(self, a, b, period):
        got = {p * period + o for (p0, p1, o0, o1) in _interval_rects(a, b, period) for p in range(p0, p1) for o in range(o0, o1)}
        assert got == set(range(a, b))

    def test_difference_area_matches_brute_force(self):
        import random
        rnd = random.Random(7)
        for _ in range(200):
            period = rnd.choice([8, 16, 24])
            A = [(rnd.randrange(0, 6), rnd.randrange(6, 12), rnd.randrange(0, period), rnd.randrange(0, period)) for _ in range(rnd.randrange(1, 4))]
            B = [(rnd.randrange(0, 6), rnd.randrange(6, 12), rnd.randrange(0, period), rnd.randrange(0, period)) for _ in range(rnd.randrange(0, 3))]
            A = [(p0, p1, min(o0, o1), max(o0, o1) + 1) for p0, p1, o0, o1 in A]; B = [(p0, p1, min(o0, o1), max(o0, o1) + 1) for p0, p1, o0, o1 in B]
            cells = lambda R: {(p, o) for (p0, p1, o0, o1) in R for p in range(p0, p1) for o in range(o0, o1)}
            assert _rect_difference_area(A, B) == len(cells(A) - cells(B)), (A, B)

    def test_lattice_with_undividing_stride_is_not_representable(self):
        assert _lattice_rects((0, 12, 5, 4), 8) is None and _lattice_rects((0, 8, 5, 4), 8 * 65) is None


def _region(offset, shape, stride, path, esize=4, nbytes=None):
    r = _returned_leaf_region(_ident(offset, shape, stride, esize, nbytes), {}); r["path"] = path; return r


class TestProvenCoverageOnly:
    def test_hole_byte_next_to_a_large_bound_stride_is_billed_and_stays_unbound(self):
        """the review case 1 analytically: bound y[::2] (lattice), unbound y[1:2] (4 B in a hole): remainder 4 B,
        nothing covered, both sides exact, so the leaf is declared (not an alias)."""
        fp = _returned_group_footprint([_region(1, [1], [1], "return[1]")], [_region(0, [N], [2], "return[0]")], 8 * N)
        assert fp["union_bytes"] == 4 and fp["covered_bytes"] == 0 and fp["unbound_exact"] and fp["coverage_exact"] and fp["period"] == 8

    def test_two_large_disjoint_stride2_views_sum(self):
        """the review case 2: even / odd views of one storage are disjoint -> full 4 N bytes unbound."""
        fp = _returned_group_footprint([_region(1, [N], [2], "return[1]")], [_region(0, [N], [2], "return[0]")], 8 * N)
        assert fp["union_bytes"] == 4 * N and fp["covered_bytes"] == 0

    def test_same_large_lattice_twice_is_fully_covered(self):
        fp = _returned_group_footprint([_region(0, [N], [2], "return[1]")], [_region(0, [N], [2], "return[0]")], 8 * N)
        assert fp["union_bytes"] == 0 and fp["covered_bytes"] == 4 * N and fp["unbound_exact"] and fp["coverage_exact"]

    def test_large_lattice_partially_covered_by_a_contiguous_bound_prefix(self):
        """bound y[:2N/2] contiguous over the first half of the storage, unbound y[::2] over all of it: exactly the
        second half's even elements remain."""
        half = N // 2 + 1                       # elements of y[::2] inside the first (N) floats
        fp = _returned_group_footprint([_region(0, [N], [2], "return[1]")], [_region(0, [N], [1], "return[0]")], 8 * N)
        assert fp["covered_bytes"] == 4 * half and fp["union_bytes"] == 4 * (N - half)

    def test_approximate_bound_region_covers_nothing(self):
        """A bound region known only as a bounding interval (two outer strides above the cap) cannot prove coverage:
        the unbound leaf keeps its full bytes and the record names the unproven bound leaf."""
        big = 1 << 11
        bound = _region(0, [big, big], [4 * big, 2], "return[0]"); assert not bound["exact"]
        fp = _returned_group_footprint([_region(1, [1], [1], "return[1]")], [bound], None)
        assert fp["union_bytes"] == 4 and fp["covered_bytes"] == 0 and fp["unproven_bound"] == ["return[0]"] and not fp["coverage_exact"]

    def test_approximate_unbound_region_is_billed_by_its_extent_and_still_reduced_by_exact_coverage(self):
        """An approximate UNBOUND region may be over-billed (its bounding extent) and exact bound coverage may still be
        subtracted from it: the result can only over-count."""
        big = 1 << 11
        unb = _region(0, [big, big], [4 * big, 2], "return[1]"); assert not unb["exact"]
        extent = unb["bytes"]
        fp = _returned_group_footprint([unb], [_region(0, [2 * big], [1], "return[0]")], None)
        assert fp["union_bytes"] == extent - 8 * big and fp["approximate_unbound"] == ["return[1]"] and not fp["unbound_exact"]

    def test_mixed_strides_on_a_common_period(self):
        """stride-2 and stride-4 views of one storage (period 16): exact union / coverage without enumeration."""
        a = _region(0, [N], [4], "return[1]")          # bytes 0..4 every 16
        b = _region(0, [N], [2], "return[0]")          # bytes 0..4 every 8 -> covers all of a that lies inside b's extent
        fp = _returned_group_footprint([a], [b], None)
        inside = sum(1 for k in range(N) if 4 * k < 2 * N)   # elements of a whose index lies within b's element range
        assert fp["period"] == 16 and fp["covered_bytes"] == 4 * inside and fp["union_bytes"] == 4 * (N - inside)

    def test_small_exact_cases_agree_with_the_interval_path(self):
        """No lattice involved: the pre-existing interval algebra answers (partial overlap 16 B of 20 B)."""
        fp = _returned_group_footprint([_region(1, [3], [1], "u")], [_region(0, [2], [1], "b")], 16)
        assert fp["period"] is None and fp["union_bytes"] == 8 and fp["covered_bytes"] == 4 and fp["remainder_regions"] == [(8, 16)]


class TestReview7FilesVerbatim:
    @pytest.mark.parametrize("filename", ["test_review7_large_stride.py", "test_review7_large_disjoint.py"])
    def test_counterexample_file(self, tmp_path, filename):
        """the reviewer's real-capture counterexamples byte-for-byte: 4,194,312 B with incomplete / unqualified status;
        8,388,616 B for the disjoint pair."""
        run = tmp_path / "run"; run.mkdir()
        (run / filename).write_bytes((FIX / filename).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / filename), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "1 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]


class TestReview7FallbackIdentityVerbatim:
    def test_fallback_identity_file(self, tmp_path):
        """the reviewer's storage-identity fallback controls byte-for-byte (two cases): an identity alias whose storage accessor
        is unavailable stays unqualified (object_fallback, conservative 64 B, status incomplete); the data-pointer fallback
        distinguishes real allocations.  The vendored file is copied into a run directory and executed there (it is never
        collected in place: tests/fixtures is harness data)."""
        run = tmp_path / "run"; run.mkdir(); filename = "test_review7_fallback_identity.py"
        (run / filename).write_bytes((FIX / filename).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / filename), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "2 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]


class TestReview2RegionsVerbatim:
    def test_regions_file(self, tmp_path):
        """the reviewer's byte-interval region controls byte-for-byte (eight cases: small oracle, signed-stride metadata exact or
        rejected, torch negative stride not constructible, ...).  The file writes its receipts under ``REVIEW_RUN_OUT``,
        which is pointed at a directory under ``tmp_path``; copied into a run directory and executed there."""
        run = tmp_path / "run"; run.mkdir(); out = tmp_path / "review_run_out"; out.mkdir(); filename = "test_review2_regions.py"
        (run / filename).write_bytes((FIX / filename).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_RUN_OUT"] = str(out)
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / filename), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "8 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]
        assert any(out.iterdir()), "the control wrote no receipt under REVIEW_RUN_OUT"


class TestGridCostBounded:
    """the review round 4, grid-cost-1 / region-bytes-1: the period-grid algebra is sub-quadratic and
    bounded.  A regular strided view is re-expressed as a lattice before any rectangle exists (even far below the
    enumeration cap), the union / difference sweep reads every rectangle once (event sweep), total rectangle-segment
    work is capped by ``_RETURNED_GRID_MAX_WORK`` with a CONSERVATIVE fallback beyond it (extent-billed, nothing
    subtracted, no alias), and an exact region's byte count is the size of its merged interval set."""

    def test_review2_mixed_lattice_and_32768_intervals_completes_fast_and_exact(self):
        """the reviewer's probe: bound base[::2] (lattice of 1,048,577 elements) with unbound base[1:2K:2], K = 32768
        (32,768 exact intervals, disjoint from the lattice): 131,072 B, nothing covered, period 8, well under 2 s."""
        import time, torch
        N = (1 << 20) + 1; K = 32768
        base = torch.empty(2 * N, dtype=torch.float32)
        def region(t, path):
            d = {"shape": list(t.shape), "stride": list(t.stride()), "storage_offset": t.storage_offset(), "element_size": t.element_size(), "storage_nbytes": t.untyped_storage().nbytes()}
            r = _returned_leaf_region(d, {}); r["path"] = path; return r
        b = region(base[::2], "bound"); u = region(base[1:2 * K:2], "unbound")
        assert b["method"] == "strided_lattice_exact" and u["method"] == "byte_region_union" and len(u["intervals"]) == K
        assert u["progression"] == (4, 8, K, 4)                       # the regular view is known as a lattice before any rectangle
        t0 = time.monotonic(); fp = _returned_group_footprint([u], [b], base.untyped_storage().nbytes()); dt = time.monotonic() - t0
        assert fp["union_bytes"] == 4 * K and fp["covered_bytes"] == 0 and fp["period"] == 8 and fp["unbound_exact"] and fp["coverage_exact"]
        assert fp["grid_work"]["rectangles"] <= 8 and fp["grid_work"]["ops"] <= 16, fp["grid_work"]
        assert dt < 2.0, dt

    @pytest.mark.parametrize("K", [256, 512, 1024])
    def test_review2_smaller_controls_keep_their_values(self, K):
        import torch
        N = (1 << 20) + 1; base = torch.empty(2 * N, dtype=torch.float32)
        def region(t, path):
            d = {"shape": list(t.shape), "stride": list(t.stride()), "storage_offset": t.storage_offset(), "element_size": t.element_size(), "storage_nbytes": t.untyped_storage().nbytes()}
            r = _returned_leaf_region(d, {}); r["path"] = path; return r
        fp = _returned_group_footprint([region(base[1:2 * K:2], "u")], [region(base[::2], "b")], base.untyped_storage().nbytes())
        assert fp["union_bytes"] == 4 * K and fp["covered_bytes"] == 0 and fp["period"] == 8

    def test_sweep_work_is_linear_in_the_raw_rectangle_count(self):
        """the reviewer's deterministic counter, extended to n = 256 .. 32768 raw interval rectangles against the big
        lattice: the sweep reads A once (visits == n) and performs ~2 n rectangle-segment operations -- not n^2 + 2 n."""
        class CountedList(list):
            def __init__(self, *a): super().__init__(*a); self.visits = 0; self.iterations = 0
            def __iter__(self): self.iterations += 1; self.visits += len(self); return super().__iter__()
        rows = []
        for n in (256, 1024, 4096, 32768):
            a = CountedList(rc for k in range(n) for rc in _interval_rects(8 * k + 4, 8 * k + 8, 8))
            b = CountedList(_lattice_rects((0, 8, (1 << 20) + 1, 4), 8)); work = {"ops": 0}
            assert _rect_difference_area(a, b, work) == 4 * n
            rows.append((n, a.visits, work["ops"]))
        for n, visits, ops in rows:
            assert visits == n and ops <= 3 * n, rows                   # linear, not quadratic
        (n0, _, o0), (n1, _, o1) = rows[0], rows[-1]
        assert o1 / o0 <= 1.5 * (n1 / n0), rows                           # sub-quadratic growth across the sweep

    def test_difference_area_sweep_matches_brute_force(self):
        import random
        rnd = random.Random(2026)
        for _ in range(300):
            period = rnd.choice([8, 16, 24])
            def rects(k):
                out = []
                for _ in range(k):
                    p0, p1 = sorted((rnd.randrange(0, 12), rnd.randrange(0, 12)))
                    o0, o1 = sorted((rnd.randrange(0, period), rnd.randrange(0, period)))
                    out.append((p0, p1 + 1, o0, o1 + 1))
                return out
            A, B = rects(rnd.randrange(1, 6)), rects(rnd.randrange(0, 5))
            cells = lambda R: {(p, o) for (p0, p1, o0, o1) in R for p in range(p0, p1) for o in range(o0, o1)}
            assert _rect_difference_area(A, B) == len(cells(A) - cells(B)), (A, B)
            assert _rect_difference_area(A, []) == len(cells(A))

    def test_coalesce_runs_finds_the_progressions(self):
        from solar.einsum.pytorch_to_einsum import _coalesce_runs
        assert _coalesce_runs([(8 * k + 4, 8 * k + 8) for k in range(1000)]) == [(4, 8, 1000, 4)]
        assert _coalesce_runs([(0, 4), (8, 12), (16, 20), (40, 44), (48, 52), (100, 108)]) == [(0, 8, 3, 4), (40, 8, 2, 4), (100, 8, 1, 8)]
        assert _coalesce_runs([(0, 4), (4, 8)]) == [(0, 8, 1, 8)]        # adjacent runs merge first
        # a two-outer-stride view below the cap: one progression per outer index, exact on the grid
        r2 = _returned_leaf_region(_ident(0, [300, 300], [4096, 2]), {})
        assert r2["exact"] and len(r2["intervals"]) == 90000 and len(_coalesce_runs(r2["intervals"])) == 300
        fp = _returned_group_footprint([r2], [_region(0, [N], [2], "b")], None)
        assert fp["union_bytes"] == 0 and fp["covered_bytes"] == 4 * 90000 and fp["unbound_exact"] and fp["coverage_exact"]
        assert fp["grid_work"]["ops"] < 4000, fp["grid_work"]

    def test_work_budget_falls_back_conservatively_never_hangs(self):
        """70,000 irregular 4-byte intervals mixed with a lattice exceed the rectangle budget: the result is the
        extent of the unbound regions (an over-count), nothing is subtracted, neither side is exact (so no alias),
        and the record says why.  Completes in well under a second."""
        import random, time
        from solar.einsum.pytorch_to_einsum import _RETURNED_GRID_MAX_WORK
        rnd = random.Random(1); starts = sorted(set(rnd.randrange(0, 10**7) * 4 for _ in range(70000)))
        irregular = {"intervals": [(a, a + 4) for a in starts], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": 4 * len(starts), "path": "irregular"}
        t0 = time.monotonic(); fp = _returned_group_footprint([irregular], [_region(0, [N], [2], "b")], None); dt = time.monotonic() - t0
        assert dt < 5.0 and "complexity_fallback" in fp and fp["complexity_fallback"]["budget"] == _RETURNED_GRID_MAX_WORK
        assert fp["union_bytes"] == starts[-1] + 4 - starts[0] and fp["covered_bytes"] == 0
        assert not fp["unbound_exact"] and not fp["coverage_exact"] and fp["approximate_unbound"] == ["irregular"] and fp["unproven_bound"] == ["b"]
        assert fp["union_bytes"] >= 4 * len(starts)                       # never an under-count

    def test_fallback_in_the_converter_keeps_the_declaration_and_marker(self, tmp_path):
        """A captured skeleton whose unbound leaf is rewritten (serialized metadata) into an irregular-looking large
        two-outer view with a non-dividing stride next to a bound lattice: the converter bills the extent, keeps
        ``returned_output_unbound`` and never certifies an alias."""
        import yaml
        fix = FIX / "test_review7_aliases.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("r7", fix); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        pg, _, _ = mod.capture(tmp_path, "y[::2], y[1:2].as_subclass(torch.Tensor)")
        u = pg["returned_outputs"][1]; ident = u["tensor_identity"]
        ident.update(shape=[700, 300], stride=[9000, 6], storage_offset=1, storage_nbytes=40_000_000, numel=700 * 300); u.update(shape=[700, 300], nbytes=4 * 700 * 300)
        pg["returned_outputs"][0]["tensor_identity"]["storage_nbytes"] = 40_000_000
        eg, total = mod.convert(tmp_path, pg)
        fp = next(v for v in eg["returned_footprint"].values() if v.get("leaves"))
        assert fp["method"] != "alias_of_bound_output" and fp["union_bytes"] >= 4 * 700 * 300
        assert total["status"] == "incomplete" and "returned_output_unbound" in total["incomplete_layers"].values()

    def test_zero_stride_region_bytes_are_the_merged_set(self):
        """region-bytes-1: shape [9] / stride [0] / width 4 addresses one element: 4 bytes, not 36."""
        r = _returned_leaf_region(_ident(0, [9], [0], nbytes=4), {})
        assert r["exact"] and r["bytes"] == 4 and _returned_leaf_region(_ident(0, [N], [0], nbytes=4), {})["bytes"] == 4
        r3 = _returned_leaf_region(_ident(0, [3, 2], [0, 1]), {}); assert r3["bytes"] == 8 and len(r3["intervals"]) == 6   # six raw runs, two distinct elements
        fp = _returned_group_footprint([r], [], 4); assert fp["union_bytes"] == 4

    def test_region_bytes_equal_the_oracle_for_small_views(self):
        import itertools
        for shape in ([1], [4], [2, 3], [3, 2, 2]):
            for stride in itertools.product(range(4), repeat=len(shape)):
                r = _returned_leaf_region(_ident(2, shape, list(stride)), {})
                assert r["bytes"] == len(_enum_bytes(2, shape, list(stride))), (shape, stride, r)

    def test_small_exact_cases_still_take_the_interval_path(self):
        """No lattice region in the group: the linear interval algebra answers exactly as before (period None)."""
        fp = _returned_group_footprint([_region(1, [3], [2], "u")], [_region(0, [2], [1], "b")], 40)
        assert fp["period"] is None and fp["union_bytes"] == 8 and fp["covered_bytes"] == 4


class TestReview2Round4FilesVerbatim:
    """the reviewer's round-4 files byte-for-byte (checks-bff9342e): the 18 lattice / approximation / oracle controls,
    the pair control, and the mixed lattice / 32,768-interval cost probe that formerly hit the 30-second timer."""
    @pytest.mark.parametrize("filename", ["test_round4_review2.py", "test_live_alias_review2.py"])
    def test_review2_file(self, tmp_path, filename):
        run = tmp_path / "run"; run.mkdir(); (run / "out").mkdir()
        for f in (filename, "test_review7_aliases.py"):
            (run / f).write_bytes((FIX / f).read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_RUN_OUT"] = str(run / "out")
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / filename), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and " passed" in res.stdout and "failed" not in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]

    def test_review2_grid_cost_probe_completes(self, tmp_path):
        run = tmp_path / "run"; run.mkdir(); (run / "out").mkdir()
        (run / "probe_grid_cost.py").write_bytes((FIX / "probe_grid_cost.py").read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_RUN_OUT"] = str(run / "out")
        res = subprocess.run([sys.executable, str(run / "probe_grid_cost.py")], cwd=str(run), env=env, capture_output=True, text=True, timeout=600)
        import json
        rows = json.loads((run / "out" / "RESULT.json").read_text())
        assert res.returncode == 0 and [r["k"] for r in rows] == [256, 512, 1024, 32768] and all(r["state"] == "COMPLETED" for r in rows), res.stdout[-2000:] + res.stderr[-2000:]
        assert all(r["result"]["union_bytes"] == 4 * r["k"] and r["seconds"] < 2.0 for r in rows), rows


class TestSubtractCostBounded:
    """the review round 5, subtract-cost-2: `_subtract_byte_intervals` is a monotone two-pointer sweep over
    ascending merged interval lists (O(n + m)); every comparison is counted into the work budget and the budget is
    enforced inside the helper, so both callers -- the period-grid sweep and the interval-only path -- fall back
    conservatively (extent-billed, nothing subtracted, no alias, marker kept) instead of hanging."""

    @staticmethod
    def _real(t, path):
        d = {"shape": list(t.shape), "stride": list(t.stride()), "storage_offset": t.storage_offset(), "element_size": t.element_size(), "storage_nbytes": t.untyped_storage().nbytes()}
        r = _returned_leaf_region(d, {}); r["path"] = path; return r

    @staticmethod
    def _meta(n, stride, start=0, width=4, path="m"):
        r = _returned_leaf_region({"shape": [n], "stride": [stride], "storage_offset": start, "element_size": width}, {}); r["path"] = path; return r

    def test_review2_two_enumerated_32768_run_views_complete_fast_and_exact(self):
        """the reviewer's interval probe: base[1::2] unbound against base[::2] bound, 32,768 runs each, interval-only
        path (no lattice): 131,072 B, nothing covered, exact, well under 2 s (formerly the 30-second timer)."""
        import time, torch
        k = 32768; base = torch.empty(2 * k, dtype=torch.float32)
        u = self._real(base[1::2], "u"); b = self._real(base[::2], "b"); assert u["lattice"] is None and b["lattice"] is None
        t0 = time.monotonic(); fp = _returned_group_footprint([u], [b], base.untyped_storage().nbytes()); dt = time.monotonic() - t0
        assert fp["union_bytes"] == 4 * k and fp["covered_bytes"] == 0 and fp["period"] is None and fp["unbound_exact"] and fp["coverage_exact"]
        assert fp["grid_work"]["ops"] <= 2 * k and dt < 2.0, (fp["grid_work"], dt)

    @pytest.mark.parametrize("K", [16384, 21845])
    def test_review2_grid_offsets_never_hang(self, K):
        """the reviewer's grid-offset probe (serialized metadata; the anchor is a lattice of stride 2K): completes within
        the budget -- exactly, or by the conservative fallback -- never a 30-second hang."""
        import time
        N = (1 << 20) + 1
        u = self._meta(K, 2, 1, path="u"); b = self._meta(K, 2, path="bound_intervals"); anchor = self._meta(N, 2 * K, path="bound_lattice")
        t0 = time.monotonic(); fp = _returned_group_footprint([u], [b, anchor], None); dt = time.monotonic() - t0
        assert dt < 5.0, dt
        if fp["unbound_exact"] and fp["coverage_exact"]:
            assert fp["union_bytes"] == 4 * K and fp["covered_bytes"] == 0
        else:
            assert "complexity_fallback" in fp and fp["union_bytes"] >= 4 * K and fp["covered_bytes"] == 0
            assert fp["approximate_unbound"] == ["u"] and set(fp["unproven_bound"]) == {"bound_intervals", "bound_lattice"}

    def test_subtraction_work_is_linear_with_integer_wrappers(self):
        """the reviewer's integer-wrapper counter: int() conversions in the subtractor are ~2(n + m), not 2 n^2."""
        class IntegerCounter:
            def __init__(self, n): self.n = n; self.visits = 0
            def __int__(self): self.visits += 1; return self.n
        rows = []
        for n in (32, 64, 128, 256, 1024, 4096, 32768):
            minus = [(IntegerCounter(8 * k), IntegerCounter(8 * k + 4)) for k in range(n)]
            a = [(8 * k + 4, 8 * k + 8) for k in range(n)]
            work = {"ops": 0}
            assert _subtract_byte_intervals(a, minus, work) == a
            visits = sum(x.visits + y.visits for x, y in minus); rows.append((n, visits, work["ops"]))
        for n, visits, ops in rows:
            assert visits <= 2 * n and ops <= 2 * n, rows                 # linear in n + m, never quadratic
        assert rows[-1][1] / rows[0][1] <= 1.1 * (rows[-1][0] / rows[0][0]), rows

    def test_subtraction_matches_brute_force_including_unsorted_and_overlapping_inputs(self):
        import random
        rnd = random.Random(6487560)
        for _ in range(600):
            A = [(x, x + rnd.randrange(1, 9)) for x in [rnd.randrange(300) for _ in range(rnd.randrange(1, 25))]]
            B = [(x, x + rnd.randrange(1, 9)) for x in [rnd.randrange(300) for _ in range(rnd.randrange(0, 25))]]
            truth = {p for a, b in A for p in range(a, b)} - {p for a, b in B for p in range(a, b)}
            for AA, BB in ((_merge_byte_intervals(A), _merge_byte_intervals(B)), (A, B)):
                pts = [p for a, b in _subtract_byte_intervals(AA, BB) for p in range(a, b)]
                assert set(pts) == truth and len(pts) == len(set(pts)), (A, B)       # exact and disjoint output

    def test_budget_is_enforced_inside_the_helper(self):
        from solar.einsum.pytorch_to_einsum import _GridWorkExceeded
        a = [(i * 8, i * 8 + 4) for i in range(100)]; b = [(i * 8 + 4, i * 8 + 8) for i in range(100)]
        with pytest.raises(_GridWorkExceeded):
            _subtract_byte_intervals(a, b, {"ops": 0, "budget": 50})
        work = {"ops": 0, "budget": 10_000}; assert _subtract_byte_intervals(a, b, work) == a and 0 < work["ops"] <= 200

    def test_interval_only_path_falls_back_conservatively_beyond_the_budget(self):
        """Two exact interval regions with ~70,000 runs each (no lattice) exceed the budget on the interval-only path:
        extent-billed, nothing subtracted, both flags false, complexity_fallback recorded -- no hang."""
        import time
        from solar.einsum.pytorch_to_einsum import _RETURNED_GRID_MAX_WORK
        n = 70_000
        u = {"intervals": [(8 * k + 4, 8 * k + 8) for k in range(n)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": 4 * n, "path": "u"}
        b = {"intervals": [(8 * k, 8 * k + 4) for k in range(n)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": 4 * n, "path": "b"}
        t0 = time.monotonic(); fp = _returned_group_footprint([u], [b], None); dt = time.monotonic() - t0
        assert dt < 5.0 and fp["complexity_fallback"]["budget"] == _RETURNED_GRID_MAX_WORK and fp["complexity_fallback"]["work_ops"] > _RETURNED_GRID_MAX_WORK
        assert fp["union_bytes"] == 8 * n - 4 and fp["covered_bytes"] == 0 and not fp["unbound_exact"] and not fp["coverage_exact"] and fp["period"] is None   # extent of bytes 4 .. 8n
        assert fp["approximate_unbound"] == ["u"] and fp["unproven_bound"] == ["b"]

    def test_interval_only_path_stays_exact_within_the_budget(self):
        n = 30_000
        u = {"intervals": [(8 * k + 4, 8 * k + 8) for k in range(n)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": 4 * n, "path": "u"}
        b = {"intervals": [(8 * k, 8 * k + 4) for k in range(n)], "lattice": None, "exact": True, "method": "byte_region_union", "bytes": 4 * n, "path": "b"}
        fp = _returned_group_footprint([u], [b], None)
        assert fp["union_bytes"] == 4 * n and fp["covered_bytes"] == 0 and fp["unbound_exact"] and fp["coverage_exact"] and "complexity_fallback" not in fp and fp["grid_work"]["ops"] <= 2 * n

    def test_converter_fallback_on_the_interval_only_path_keeps_the_declaration(self, tmp_path):
        """Serialized metadata on a captured skeleton: two enumerated views (no lattice) whose subtraction exceeds the
        budget -> the converter bills the extent, keeps `returned_output_unbound`, certifies no alias."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("r7", FIX / "test_review7_aliases.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        pg, _, _ = mod.capture(tmp_path, "y[::2], y[1:2].as_subclass(torch.Tensor)")
        n = 70_000
        for leaf, start in ((pg["returned_outputs"][0], 0), (pg["returned_outputs"][1], 1)):
            leaf["tensor_identity"].update(shape=[n], stride=[2], storage_offset=start, numel=n, element_size=4, storage_nbytes=8 * n + 8); leaf.update(shape=[n], nbytes=4 * n)
        eg, total = mod.convert(tmp_path, pg)
        fp = next(v for v in eg["returned_footprint"].values() if v.get("leaves"))
        assert fp["method"] != "alias_of_bound_output" and fp["union_bytes"] >= 4 * n and fp.get("complexity_fallback")
        assert total["status"] == "incomplete" and "returned_output_unbound" in total["incomplete_layers"].values()
