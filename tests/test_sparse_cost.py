# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured-sparsity SOL: density profiling hooks + zeros-are-free analyzer.

Covers the feature end to end on toy DSLs:
  (a) profiling hook — a top-k mask's measured density is exactly k/E and a
      masked kernel OUTPUT's measured density matches;
  (b) analyzer MAC discount — a dense-form MoE kernel multiplied by a top-k
      mask costs dense_macs * k/E (zero elements are free);
  (c) sliding-window toy — a banded mask measures W-band density and discounts
      attention-score MACs accordingly;
  (d) regression — a dense DSL analyzes identically with profiling off and on.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SOLAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR))

from solar.einsum_lib import (  # noqa: E402
    build_metadata, get_trace, set_density_profiling,
)
from solar.einsum_lib.verify import execute_translated  # noqa: E402


# ---------------------------------------------------------------------------
# Toy DSLs
# ---------------------------------------------------------------------------

MOE_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, top_k_mask

@kernel
def masked_expert(X: Tensor("N,H", dtype="fp32"),
                  M: Tensor("N,E", dtype="fp32"),
                  W: Tensor("E,H", role="weight", dtype="fp32")) -> Tensor("N,E"):
    Y = Tensor(N=X.shape[0], E=M.shape[1], dtype="fp32")
    for n in Dim("N", parallel=True):
        for e in Dim("E"):
            acc = 0.0
            for h in Dim("H"):
                acc += X[n, h] * W[e, h]
            Y[n, e] = acc * M[n, e]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,H", dtype="fp32")
    s = Tensor.from_numpy(inputs["scores"], "N,E", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "E,H", dtype="fp32")
    m = top_k_mask(s, 2)
    y = masked_expert(x, m, w)
    return {"output": y.data}
'''

SWA_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def banded_scores(Q: Tensor("S,D", dtype="fp32"),
                  K: Tensor("T,D", dtype="fp32"),
                  B: Tensor("S,T", dtype="fp32")) -> Tensor("S,T"):
    Y = Tensor(S=Q.shape[0], T=K.shape[0], dtype="fp32")
    for s in Dim("S", parallel=True):
        for t in Dim("T"):
            acc = 0.0
            for d in Dim("D"):
                acc += Q[s, d] * K[t, d]
            Y[s, t] = acc * B[s, t]
    return Y

def run_model(inputs):
    clear_dims()
    q = Tensor.from_numpy(inputs["q"], "S,D", dtype="fp32")
    k = Tensor.from_numpy(inputs["k"], "T,D", dtype="fp32")
    b = Tensor.from_numpy(inputs["band"], "S,T", dtype="fp32")
    return {"output": banded_scores(q, k, b).data}
'''

DENSE_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def linear(X: Tensor("N,H", dtype="fp32"),
           W: Tensor("F,H", role="weight", dtype="fp32")) -> Tensor("N,F"):
    Y = Tensor(N=X.shape[0], F=W.shape[0], dtype="fp32")
    for n in Dim("N", parallel=True):
        for f in Dim("F"):
            for h in Dim("H"):
                Y[n, f] += X[n, h] * W[f, h]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "F,H", dtype="fp32")
    return {"output": linear(x, w).data}
'''

N, E, H = 16, 8, 32
TOPK = 2
S, T, D, W_BAND = 32, 32, 16, 4


def _moe_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((N, H)),
        "scores": rng.standard_normal((N, E)),
        "w": rng.standard_normal((E, H)),
    }


def _swa_inputs():
    rng = np.random.default_rng(1)
    band = np.zeros((S, T))
    for i in range(S):
        band[i, max(0, i - W_BAND + 1): i + 1] = 1.0
    return {
        "q": rng.standard_normal((S, D)),
        "k": rng.standard_normal((T, D)),
        "band": band,
    }


def _traced_metadata(code, inputs, profile):
    set_density_profiling(profile)
    try:
        execute_translated(
            code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs,
            trace=True, shape_only=False, fast=False,
        )
    finally:
        set_density_profiling(False)
    return build_metadata(get_trace())


def _analyze(code, md, tmp_path, stem="toy_einsum_lib"):
    """dsl (py+yaml) -> einsum graph -> analysis dict (the real pipeline)."""
    import yaml
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_dsl.cli_einsum import translate_with_einsum
    from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

    dsl_dir = tmp_path / "dsl"
    dsl_dir.mkdir(parents=True, exist_ok=True)
    (dsl_dir / f"{stem}.py").write_text(code)
    write_metadata_yaml(md, str(dsl_dir / f"{stem}.yaml"))
    graph_path = translate_with_einsum(str(dsl_dir), str(tmp_path / "einsum"))
    analyzer = EinsumGraphAnalyzer()
    out = analyzer.analyze_graph(str(graph_path), str(tmp_path / "analysis"))
    assert out is not None and "layers" in out, "analysis failed"
    return out


# ---------------------------------------------------------------------------
# (a) profiling hook measures exact densities
# ---------------------------------------------------------------------------

def test_profiling_measures_topk_density():
    md = _traced_metadata(MOE_DSL, _moe_inputs(), profile=True)
    tensors = md["tensors"]
    assert math.isclose(tensors["top_k_mask#0"]["sparsity"], TOPK / E)
    # kernel OUTPUT density back-patched by the profiler (zeros where mask=0)
    assert math.isclose(tensors["masked_expert#0"]["sparsity"], TOPK / E,
                        rel_tol=1e-9)


def test_no_density_without_profiling():
    md = _traced_metadata(MOE_DSL, _moe_inputs(), profile=False)
    # top_k_mask's own analytic k/E is still present (pre-existing semantics)…
    assert math.isclose(md["tensors"]["top_k_mask#0"]["sparsity"], TOPK / E)
    # …but the kernel output carries NO measured density.
    assert "sparsity" not in md["tensors"]["masked_expert#0"]


# ---------------------------------------------------------------------------
# (b) analyzer: zeros are free for the masked MoE kernel
# ---------------------------------------------------------------------------

def test_moe_mac_discount(tmp_path):
    inputs = _moe_inputs()
    dense_macs = N * E * H

    md = _traced_metadata(MOE_DSL, inputs, profile=True)
    analysis = _analyze(MOE_DSL, md, tmp_path)
    layer = analysis["layers"]["masked_expert#0"]
    assert layer["compute_density"] == pytest.approx(TOPK / E)
    assert layer["macs"] == pytest.approx(dense_macs * TOPK / E, rel=0.01)

    md_dense = _traced_metadata(MOE_DSL, inputs, profile=False)
    # analytic top_k_mask density alone still reaches the consumer kernel via
    # its INPUT sparsity — the discount applies even without profiling when the
    # mask is a direct input; scrub sparsity to get the true dense baseline.
    for meta in md_dense["tensors"].values():
        meta.pop("sparsity", None)
    analysis_dense = _analyze(MOE_DSL, md_dense, tmp_path / "dense")
    layer_dense = analysis_dense["layers"]["masked_expert#0"]
    assert "compute_density" not in layer_dense
    assert layer_dense["macs"] == dense_macs


# ---------------------------------------------------------------------------
# (c) sliding-window toy: banded mask discounts score MACs by the band density
# ---------------------------------------------------------------------------

def test_swa_band_discount(tmp_path):
    inputs = _swa_inputs()
    band_density = float(np.count_nonzero(inputs["band"])) / (S * T)
    assert band_density < 0.5  # the toy band is genuinely sparse

    md = _traced_metadata(SWA_DSL, inputs, profile=True)
    assert md["tensors"]["band"]["sparsity"] == pytest.approx(band_density)

    analysis = _analyze(SWA_DSL, md, tmp_path)
    layer = analysis["layers"]["banded_scores#0"]
    assert layer["compute_density"] == pytest.approx(band_density)
    assert layer["macs"] == pytest.approx(S * T * D * band_density, rel=0.01)


# ---------------------------------------------------------------------------
# (d) regression: dense DSL identical with profiling off and on
# ---------------------------------------------------------------------------

def test_dense_unchanged_by_profiling(tmp_path):
    rng = np.random.default_rng(2)
    inputs = {"x": rng.standard_normal((N, H)), "w": rng.standard_normal((E, H))}

    md_off = _traced_metadata(DENSE_DSL, inputs, profile=False)
    md_on = _traced_metadata(DENSE_DSL, inputs, profile=True)
    # randn data is fully dense -> profiler tags nothing
    assert all("sparsity" not in m for m in md_on["tensors"].values())

    a_off = _analyze(DENSE_DSL, md_off, tmp_path / "off")
    a_on = _analyze(DENSE_DSL, md_on, tmp_path / "on")
    l_off = a_off["layers"]["linear#0"]
    l_on = a_on["layers"]["linear#0"]
    assert l_off["macs"] == l_on["macs"] == N * E * H
    assert "compute_density" not in l_on


# ---------------------------------------------------------------------------
# Density trust ladder (scripts/trace_saved_ir.py --measure-density)
# ---------------------------------------------------------------------------

LADDER_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, top_k_mask

_W = 2  # sliding-window half-width + 1 (band offsets -1, 0, +1)

@kernel
def band_mask(X: Tensor("S", dtype="fp32")) -> Tensor("S,T"):
    # |s - t| < _W band: density ~ (3S-2)/S^2 -- SCALE-COUPLED in S.
    Y = Tensor(S=X.shape[0], T=X.shape[0], dtype="fp32")
    for s in Dim("S"):
        for t in Dim("T", size=X.shape[0]):
            Y[s, t] = max(0.0, min(1.0, _W - abs(s - t)))
    return Y

@kernel
def thresh(G: Tensor("E", dtype="fp32")) -> Tensor("E"):
    # index-based: nonzero only for e >= 12 -> density 4/16, shape const in E.
    Y = Tensor(E=G.shape[0], dtype="fp32")
    for e in Dim("E"):
        Y[e] = max(0.0, min(1.0, e - 11))
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "S", dtype="fp32")
    g = Tensor.from_numpy(inputs["g"], "E", dtype="fp32")
    bm = band_mask(x)
    th = thresh(g)
    tk = top_k_mask(x, 4)   # ANALYTIC k/S: 4/32 at original, 4/8 capped
    return {"bm": bm.data, "th": th.data, "tk": tk.data}
'''

LADDER_ADAPTER = '''
import numpy as np

def get_source():
    return ""

def get_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal(8).astype(np.float32),   # S capped 32 -> 8
            "g": rng.standard_normal(16).astype(np.float32)}  # E const 16

def kernel_fn(inputs):
    return {}
'''

LADDER_DEFN = {
    "name": "ladder_toy",
    "axes": {"S": {"type": "var"}, "E": {"type": "const", "value": 16}},
    "inputs": {
        "x": {"shape": ["S"], "dtype": "float32"},
        "g": {"shape": ["E"], "dtype": "float32"},
    },
}


def _run_ladder(tmp_path, extra_args):
    import json
    import subprocess
    pdir = tmp_path / "problem"
    pdir.mkdir(exist_ok=True)
    (pdir / "definition.json").write_text(json.dumps(LADDER_DEFN))
    (pdir / "workload.jsonl").write_text(
        json.dumps({"uuid": "t0", "axes": {"S": 32}}) + "\n")
    (tmp_path / "dsl.py").write_text(LADDER_DSL)
    (tmp_path / "adapter.py").write_text(LADDER_ADAPTER)
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SOLAR / "scripts" / "trace_saved_ir.py"),
         str(tmp_path / "dsl.py"), "--definition", str(pdir),
         "--out-dir", str(out), "--fallback-adapter", str(tmp_path / "adapter.py"),
         "--measure-density", *extra_args],
        capture_output=True, text=True, timeout=300, cwd=SOLAR)
    assert proc.returncode == 0, proc.stderr[-3000:]
    import yaml
    md = yaml.safe_load((out / "dsl.yaml").read_text())
    return md.get("tensors") or {}, proc.stderr


def test_ladder_auto_revert_and_shape_invariant(tmp_path):
    """Scale-coupled band density revalidated at ORIGINAL shapes (rung 3);
    const-shape threshold density injected directly (rung 2); top_k_mask
    analytic k/S_original wins over the capped measurement (rung 1)."""
    tensors, err = _run_ladder(tmp_path, [])
    # rung 3: band at S=32 -> (3*32-2)/32^2, NOT the capped (3*8-2)/64
    assert tensors["band_mask#0"]["sparsity"] == pytest.approx(94 / 1024, abs=1e-6)
    assert tensors["band_mask#0"]["sparsity_source"] == "revalidated_original"
    assert "revalidated at original shapes" in err
    # rung 2: shape-invariant threshold (E const) = 4/16
    assert tensors["thresh#0"]["sparsity"] == pytest.approx(4 / 16, abs=1e-6)
    assert tensors["thresh#0"]["sparsity_source"] == "measured"
    # rung 1: analytic top-k at ORIGINAL S: 4/32 (capped run saw 4/8)
    assert tensors["top_k_mask#0"]["sparsity"] == pytest.approx(4 / 32, abs=1e-6)
    assert "keeping analytic" in err


def test_ladder_refuses_when_budget_too_small(tmp_path):
    """Rung 4: no budget for the original-shape run -> scale-coupled density
    REFUSED (stays dense) with the --density-override remedy printed."""
    tensors, err = _run_ladder(tmp_path, ["--mem-budget", "1e-9"])
    assert "sparsity" not in tensors["band_mask#0"]
    assert "REFUSED" in err
    assert "--density-override band_mask#0=" in err
    # shape-invariant and analytic rungs unaffected
    assert tensors["thresh#0"]["sparsity"] == pytest.approx(0.25, abs=1e-6)
    assert tensors["top_k_mask#0"]["sparsity"] == pytest.approx(0.125, abs=1e-6)


def test_ladder_override_wins(tmp_path):
    tensors, err = _run_ladder(
        tmp_path, ["--density-override", "band_mask#0=0.5",
                   "--mem-budget", "1e-9"])
    assert tensors["band_mask#0"]["sparsity"] == pytest.approx(0.5)
    assert tensors["band_mask#0"]["sparsity_source"] == "override"
    assert "0 REFUSED" in err  # override removed it from the refusal path


# ---------------------------------------------------------------------------
# (e) slot-dense MoE must NOT be double-discounted: the routed kernel multiplies
#     by the sparse routing weights RW[n, e], but e is GATHERED (e = SE[n, k]),
#     so the loop already visits only the non-zero entries.
# ---------------------------------------------------------------------------

SLOT_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, top_k_mask

@kernel
def renorm(M: Tensor("N,E", dtype="fp32")) -> Tensor("N,E"):
    Y = Tensor(N=M.shape[0], E=M.shape[1], dtype="fp32")
    for n in Dim("N", parallel=True):
        for e in Dim("E"):
            Y[n, e] = M[n, e] * 0.5
    return Y

@kernel
def slot_indices(M: Tensor("N,E", dtype="fp32")) -> Tensor("N,K"):
    Y = Tensor(N=M.shape[0], K=2, dtype="int64")
    for n in Dim("N", parallel=True):
        for k in Dim("K", size=2):
            c = 0
            idx = 0
            for e in Dim("E"):
                m = int(M[n, e])
                hit = m * max(0, 1 - abs(c - k))
                idx = idx + hit * (e - idx)
                c = c + m
            Y[n, k] = idx
    return Y

@kernel
def routed(X: Tensor("N,H", dtype="fp32"),
           RW: Tensor("N,E", dtype="fp32"),
           SE: Tensor("N,K", dtype="int64"),
           W: Tensor("E,H", role="weight", dtype="fp32")) -> Tensor("N,E"):
    Y = Tensor(N=X.shape[0], E=RW.shape[1], dtype="fp32")
    for n in Dim("N", parallel=True):
        for k in Dim("K", size=SE.shape[1]):
            e = int(SE[n, k])
            acc = 0.0
            for h in Dim("H"):
                acc += X[n, h] * W[e, h]
            Y[n, e] = acc * RW[n, e]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,H", dtype="fp32")
    s = Tensor.from_numpy(inputs["scores"], "N,E", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "E,H", dtype="fp32")
    m = top_k_mask(s, 2)
    rw = renorm(m)
    se = slot_indices(m)
    y = routed(x, rw, se, w)
    return {"output": y.data}
'''


def test_slot_dense_not_double_discounted(tmp_path):
    inputs = _moe_inputs()
    md = _traced_metadata(SLOT_DSL, inputs, profile=True)
    # RW really is sparse (measured k/E) ...
    rw_meta = [m for name, m in md["tensors"].items() if name.startswith("renorm")]
    assert rw_meta and rw_meta[0]["sparsity"] == pytest.approx(TOPK / E)
    analysis = _analyze(SLOT_DSL, md, tmp_path)
    layer = analysis["layers"]["routed#0"]
    # ... but the routed kernel's loops already visit only the gathered slots:
    # activated-only MACs = N * TOPK * H, and NO extra density discount.
    assert "compute_density" not in layer
    assert layer["macs"] == N * TOPK * H


# ---------------------------------------------------------------------------
# Convention-protected attention guard (an earlier revision): probability tensors from
# softmax-named kernels stay DENSE under profiling unless explicitly opted in.
# ---------------------------------------------------------------------------

def test_guard_blocks_softmax_named_output(capsys):
    """A kernel named like softmax/probs does NOT get density-tagged."""
    import numpy as np
    from solar.einsum_lib import (Tensor, Dim, kernel, clear_dims,
                                  set_density_profiling)
    clear_dims()
    set_density_profiling(True)
    try:
        S = 16
        x = np.tril(np.ones((S, S))); x = x / x.sum(1, keepdims=True)

        @kernel
        def softmax_probs(X: Tensor("S,T", dtype="fp32")) -> Tensor("S,T"):
            Y = Tensor(S=X.shape[0], T=X.shape[1], dtype="fp32")
            for s in Dim("S"):
                for u in Dim("T"):
                    Y[s, u] = X[s, u] + 0.0
            return Y

        xt = Tensor.from_numpy(x, "S,T", dtype="fp32")
        out = softmax_probs(xt)
        assert out.sparsity is None, "guarded probs tensor must stay dense"
        assert "GUARDED" in capsys.readouterr().err
    finally:
        set_density_profiling(False)


def test_guard_opt_in_measures_attention():
    """measure_attention=True bypasses the guard: density IS tagged."""
    import numpy as np
    from solar.einsum_lib import (Tensor, Dim, kernel, clear_dims,
                                  set_density_profiling)
    clear_dims()
    set_density_profiling(True, measure_attention=True)
    try:
        S = 16
        x = np.tril(np.ones((S, S))); x = x / x.sum(1, keepdims=True)

        @kernel
        def softmax_probs2(X: Tensor("S,T", dtype="fp32")) -> Tensor("S,T"):
            Y = Tensor(S=X.shape[0], T=X.shape[1], dtype="fp32")
            for s in Dim("S"):
                for u in Dim("T"):
                    Y[s, u] = X[s, u] + 0.0
            return Y

        xt = Tensor.from_numpy(x, "S,T", dtype="fp32")
        out = softmax_probs2(xt)
        expected = (S + 1) / (2 * S)  # causal density 17/32
        assert out.sparsity is not None
        assert abs(out.sparsity - expected) < 1e-9
    finally:
        set_density_profiling(False)


def test_guard_role_annotation_blocks_any_name():
    """role='attention_probability' guards regardless of tensor/kernel name."""
    import numpy as np
    from solar.einsum_lib import Tensor, clear_dims, set_density_profiling
    clear_dims()
    set_density_profiling(True)
    try:
        S = 16
        x = np.tril(np.ones((S, S))); x = x / x.sum(1, keepdims=True)
        t = Tensor.from_numpy(x, "S,T", dtype="fp32",
                              role="attention_probability")
        assert t.sparsity is None
    finally:
        set_density_profiling(False)


def test_guard_does_not_block_unrelated_sparse_tensors():
    """Control: an innocuously-named sparse tensor (MoE-style renormalized
    routing weights — also probability-like!) is still density-tagged;
    the guard is deliberately naming/role-based, not value-signature-based."""
    import numpy as np
    from solar.einsum_lib import Tensor, clear_dims, set_density_profiling
    clear_dims()
    set_density_profiling(True)
    try:
        N, E, K = 32, 16, 4
        rw = np.zeros((N, E))
        for n in range(N):
            idx = np.random.default_rng(n).choice(E, K, replace=False)
            w = np.random.default_rng(n + 1).random(K)
            rw[n, idx] = w / w.sum()   # fractional, rows sum to 1
        t = Tensor.from_numpy(rw, "N,E", dtype="fp32")
        assert t.sparsity is not None
        assert abs(t.sparsity - K / E) < 1e-9
    finally:
        set_density_profiling(False)
