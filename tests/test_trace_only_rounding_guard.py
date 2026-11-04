# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the trace-only rounding guard in ``einsum_lib.core.kernel`` (issue #67).

Defect: the ``@kernel`` wrapper applied ``_truncate_to_dtype`` to every bf16/fp16 output even in a
shape-only trace (``start_trace(trace_only=True)``).  There the output is the zero-storage
placeholder from ``_shape_only_zeros`` (no kernel body runs); ``astype(float32)`` and the torch
bf16/fp64 casts materialised it at the ORIGINAL logical dims and the output-provenance binding kept
the copies alive -- the 13 Lite all-shapes memory failures (attention-score tensors of 1-11 GiB).

Guard: under ``_is_trace_only()`` the placeholder is kept as the traced output (rounding zeros is
the identity, nothing reads the values) and the trace entry says so explicitly; every non-trace
path still rounds exactly as before.
"""
import tracemalloc

import numpy as np
import pytest

from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, start_trace, stop_trace, get_trace
from solar.einsum_lib import core
from solar.einsum_lib.metadata import build_metadata


@pytest.fixture(autouse=True)
def _clean():
    clear_dims()
    yield
    stop_trace()
    clear_dims()


def _bf16(x):
    import torch
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float64).item()


def _counting(monkeypatch):
    calls = []
    orig = core._truncate_to_dtype

    def spy(arr, dtype_str):
        calls.append((tuple(arr.shape), dtype_str, core.is_shape_only_placeholder(arr)))
        return orig(arr, dtype_str)

    monkeypatch.setattr(core, "_truncate_to_dtype", spy)
    return calls


@kernel
def fill_bf16(X: Tensor("B,M", dtype="fp32")) -> Tensor("B,M", dtype="bf16"):
    Y = Tensor(B=X.shape[0], M=X.shape[1], dtype="bf16")
    for b in Dim("B"):
        for m in Dim("M"):
            Y[b, m] = X[b, m] + 1.005859375        # 1 + 3*2^-9: NOT bf16-representable
    return Y


@kernel
def fill_fp16(X: Tensor("B,M", dtype="fp32")) -> Tensor("B,M", dtype="fp16"):
    Y = Tensor(B=X.shape[0], M=X.shape[1], dtype="fp16")
    for b in Dim("B"):
        for m in Dim("M"):
            Y[b, m] = X[b, m] + 1.00048828125      # 1 + 2^-11: NOT fp16-representable
    return Y


@kernel
def scores(Q: Tensor("B,N,S,D", dtype="bf16"), K: Tensor("B,N,T,D", dtype="bf16")) -> Tensor("B,N,S,T", dtype="bf16"):
    Y = Tensor(B=Q.shape[0], N=Q.shape[1], S=Q.shape[2], T=K.shape[2], dtype="bf16")
    for b in Dim("B"):
        for n in Dim("N"):
            for s in Dim("S"):
                for t in Dim("T"):
                    acc = 0.0
                    for d in Dim("D"):
                        acc += Q[b, n, s, d] * K[b, n, t, d]
                    Y[b, n, s, t] = acc
    return Y


@kernel
def rowsum(P: Tensor("B,N,S,T", dtype="bf16")) -> Tensor("B,N,S", dtype="fp32"):
    Y = Tensor(B=P.shape[0], N=P.shape[1], S=P.shape[2], dtype="fp32")
    for b in Dim("B"):
        for n in Dim("N"):
            for s in Dim("S"):
                acc = 0.0
                for t in Dim("T"):
                    acc += P[b, n, s, t]
                Y[b, n, s] = acc
    return Y


def _model(inputs):
    q = Tensor.from_numpy(inputs["q"], "B,N,S,D", dtype="bf16")
    k = Tensor.from_numpy(inputs["k"], "B,N,T,D", dtype="bf16")
    p = scores(q, k)
    return {"p": p.data, "r": rowsum(p).data}


class TestNonTraceRoundingPreserved:
    """(a) real execution still rounds narrow-dtype kernel outputs -- the guard changes nothing here."""

    def test_bf16_output_is_rounded(self, monkeypatch):
        calls = _counting(monkeypatch)
        x = Tensor.from_numpy(np.zeros((2, 3)), "B,M", dtype="fp32")
        y = fill_bf16(x)
        assert y.data.dtype == np.float64
        # 1 + 3*2^-9 lies between the bf16 neighbours 1.0 and 1.0078125 (spacing 2^-7 at 1.0), above the
        # midpoint 1.00390625: round-to-nearest gives 1.0078125.  Exact values, compared as Python floats.
        assert [float(v) for v in y.data.ravel()] == [1.0078125] * 6
        assert _bf16(1.005859375) == 1.0078125 and 1.0078125 != 1.005859375
        assert calls == [((2, 3), "bf16", False)]

    def test_fp16_output_is_rounded(self, monkeypatch):
        calls = _counting(monkeypatch)
        x = Tensor.from_numpy(np.zeros((2, 3)), "B,M", dtype="fp32")
        y = fill_fp16(x)
        assert y.data.dtype == np.float64
        # 1 + 2^-11 is the exact tie between the fp16 neighbours 1.0 and 1.0009765625 (spacing 2^-10 at 1.0):
        # ties-to-even gives 1.0.  Compare as Python floats only -- `np.float16(x) != x` with a Python float
        # is promotion-dependent (NumPy 1.x legacy: float64 compare, True; NumPy 2.x NEP 50: weak Python
        # scalar cast to float16, False), which is what failed in CI.
        assert [float(v) for v in y.data.ravel()] == [1.0] * 6
        assert float(np.float16(1.00048828125)) == 1.0 and 1.0 != 1.00048828125
        assert calls == [((2, 3), "fp16", False)]

    def test_full_trace_with_execution_still_rounds(self, monkeypatch):
        """trace=True without trace_only (loop bodies execute): rounding applies exactly as without tracing."""
        calls = _counting(monkeypatch)
        start_trace(trace_only=False)
        x = Tensor.from_numpy(np.zeros((2, 3)), "B,M", dtype="fp32")
        y = fill_bf16(x)
        stop_trace()
        assert np.all(y.data == _bf16(1.005859375))
        assert calls == [((2, 3), "bf16", False)]
        assert not any(e.get("trace_only_unrounded") for e in get_trace() if e.get("type") == "tensor")


class TestTraceOnlyPlaceholderKept:
    """(b) shape-only trace: no large allocation, placeholder identity and provenance explicit."""

    LOGICAL = (64, 64, 4096, 4096)   # 2^36 elements: 256 GiB in fp32 -- cannot be materialised

    def test_no_large_allocation_and_provenance(self, monkeypatch):
        calls = _counting(monkeypatch)
        B, N, S, D = self.LOGICAL[0], self.LOGICAL[1], self.LOGICAL[2], 128
        inputs = {"q": core._shape_only_zeros((B, N, S, D), np.float64), "k": core._shape_only_zeros((B, N, S, D), np.float64)}
        start_trace(trace_only=True)
        core._tracer.register_inputs(inputs)
        tracemalloc.start()
        try:
            out = _model(inputs)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        core._tracer.record_model_outputs(out)
        stop_trace()
        assert peak < 64 * 1024 * 1024, peak                      # placeholders only; the logical bf16 output is 2^36 elements
        p = out["p"]
        assert p.shape == self.LOGICAL and core.is_shape_only_placeholder(p)
        assert calls == []                                        # rounding deferred: nothing was truncated under trace_only
        entries = get_trace()
        tens = {e["id"]: e for e in entries if e.get("type") == "tensor"}
        p_entry = next(e for e in tens.values() if tuple(e["shape"]) == self.LOGICAL)
        assert p_entry["trace_only_unrounded"] == "bf16"           # provenance: placeholder kept, rounding deferred to execution
        assert all("trace_only_unrounded" not in e for e in tens.values() if e["dtype"] == "fp32")
        mo = next(e for e in entries if e.get("type") == "model_outputs")
        assert sorted(leaf["path"][0] for leaf in mo["outputs"]) == ["p", "r"]
        assert {leaf["tensor_id"] for leaf in mo["outputs"]} == {p_entry["id"], next(e["id"] for e in tens.values() if e["dtype"] == "fp32" and e["role"] is None)}
        assert not any(e.get("type") == "model_outputs_unrecorded" for e in entries)

    def test_metadata_carries_no_new_field_and_matches_executed_trace(self):
        """The sidecar built from a trace-only run is identical to the one from an executed (full) trace of the
        same program at a feasible shape: the guard adds provenance to the trace entries only."""
        B, N, S, D = 1, 2, 4, 8
        rng = np.random.RandomState(0)
        real = {"q": rng.standard_normal((B, N, S, D)), "k": rng.standard_normal((B, N, S, D))}
        start_trace(trace_only=False)
        core._tracer.register_inputs(real)
        core._tracer.record_model_outputs(_model(real))
        stop_trace()
        md_exec = build_metadata(get_trace())
        clear_dims()
        ph = {"q": core._shape_only_zeros((B, N, S, D), np.float64), "k": core._shape_only_zeros((B, N, S, D), np.float64)}
        start_trace(trace_only=True)
        core._tracer.register_inputs(ph)
        core._tracer.record_model_outputs(_model(ph))
        stop_trace()
        md_trace = build_metadata(get_trace())
        assert md_trace == md_exec
        assert "trace_only_unrounded" not in repr(md_trace)
