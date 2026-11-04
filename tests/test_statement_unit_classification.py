# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-statement MAC / other_ops classification (issue #59, TARGET-KB-MAC-UNITS-6)
and the built-in gather cost route (TARGET-GATHER-COST-1).

Unit contract (release decision ``mac_unit_convention``): MAC = one
multiply-accumulate; standalone adds / multiplies / scales / masks / copies are
``other_ops``; dtype MAC rates price MACs only.  The classification is made per
executed assignment statement by ``solar.einsum_dsl.statement_class`` -- the
single code path the DSL parser records, the converter sums per class and the
graph analyzer bills -- never by a whole-kernel ``is_real_einsum`` flag.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer  # noqa: E402
from solar.einsum_dsl.statement_class import MAC, OTHER, classify_source_statement  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the classifier (shared code path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stmt,loop_vars,expected", [
    ("Y[m, n] += X[m, k] * W[n, k]", "mnk", (MAC, 0)),                 # contraction
    ("acc += X[n, j] * X[n, j]", "nj", (MAC, 0)),                      # self-product (variance)
    ("acc += X[m, k] * s", "mk", (MAC, 0)),                            # scaled accumulate: FMA
    ("Y[i] += A[i, k] * Bm[i, k] * Cm[i, k]", "ik", (MAC, 1)),         # 3 value factors: 1 MAC + 1 multiply
    ("St[b] += Bb[b, l] * exp(Ac[b] - Ac[l]) * X[b, l]", "bl", (MAC, 1)),
    ("Y[m, n] = Y[m, n] + X[m, k] * W[n, k]", "mnk", (MAC, 0)),        # written-out accumulation
    ("Y[b, n] += X[b, k] * (k <= n)", "bnk", (OTHER, 0)),              # masked self-reduction: no MAC
    ("Y[b, n] += X[b, k]", "bnk", (OTHER, 0)),                          # plain add-accumulate
    ("Y[m, n] = X[m, n] + B[n]", "mn", (OTHER, 0)),                    # bias add
    ("Y[m, n] = acc * s + B[n]", "mn", (OTHER, 0)),                    # scale + bias epilogue
    ("Y[m, n] = X[m, n] * W[m, n]", "mn", (OTHER, 0)),                 # elementwise product, not accumulated
    ("S[q, k] = S[q, k] * scale + M[q, k]", "qk", (OTHER, 0)),         # score scale / mask
    ("Y[n, h] = acc", "nh", (OTHER, 0)),                               # copy
    ("acc = 0.0", "", (OTHER, 0)),                                     # initialisation
    ("Y[n] += X[n] * 2.0", "n", (OTHER, 0)),                           # constant factor only
    ("Y[n] += X[n] * k", "nk", (OTHER, 0)),                            # loop index is not a value factor
    ("Ms[b, h, s] += K[b, h, s, a] * K[b, h, s, a] / n", "bhsa", (MAC, 0)),   # constant divisor scales the product
    ("Y[i] += -(A[i, k] * B[k])", "ik", (MAC, 0)),                            # negated product
    ("not valid python", "", (OTHER, 0)),
])
def test_classify_statement(stmt, loop_vars, expected):
    assert classify_source_statement(stmt, set(loop_vars)) == expected


# ---------------------------------------------------------------------------
# 2. tiny kernels through the real pipeline, one per class
# ---------------------------------------------------------------------------

def _traced_analysis(code, inputs, tmp_path, stem="toy_einsum_lib", shape_only=False):
    from solar.einsum_lib import build_metadata, get_trace
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_lib.verify import execute_translated
    from solar.einsum_dsl.cli_einsum import translate_with_einsum

    execute_translated(code, {k: v.shape for k, v in inputs.items() if hasattr(v, "shape")},
                       all_inputs=inputs, trace=True, shape_only=shape_only, fast=False)
    md = build_metadata(get_trace())
    dsl_dir = tmp_path / "dsl"
    dsl_dir.mkdir(parents=True, exist_ok=True)
    (dsl_dir / f"{stem}.py").write_text(code)
    write_metadata_yaml(md, str(dsl_dir / f"{stem}.yaml"))
    graph_path = translate_with_einsum(str(dsl_dir), str(tmp_path / "einsum"))
    graph = yaml.safe_load(Path(graph_path).read_text())
    out = EinsumGraphAnalyzer().analyze_graph(str(graph_path), str(tmp_path / "analysis"),
                                              precision="fp32", copy_graph=False)
    assert out is not None and "layers" in out
    return graph, out


def _kernel_layer(out, name):
    return next(v for k, v in out["layers"].items() if k.startswith(name + "#"))


HEADER = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n\n"

M, N, K = 4, 3, 5
BIAS_ADD = HEADER + '''
@kernel
def bias_add(X: Tensor("M,N", dtype="fp32"), B: Tensor("N", role="weight", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            Y[m, n] = X[m, n] + B[n]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N", dtype="fp32")
    b = Tensor.from_numpy(inputs["b"], "N", role="weight", dtype="fp32")
    return {"output": bias_add(x, b).data}
'''

MATMUL_SCALE_BIAS = HEADER + '''
@kernel
def matmul_scale_bias(X: Tensor("M,K", dtype="fp32"), W: Tensor("N,K", role="weight", dtype="fp32"),
                      B: Tensor("N", role="weight", dtype="fp32"), s: float) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += X[m, k] * W[n, k]
            Y[m, n] = acc * s + B[n]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "N,K", role="weight", dtype="fp32")
    b = Tensor.from_numpy(inputs["b"], "N", role="weight", dtype="fp32")
    return {"output": matmul_scale_bias(x, w, b, 0.5).data}
'''

CUMSUM_MASKED = HEADER + '''
@kernel
def cumsum_masked(X: Tensor("B,K", dtype="fp32")) -> Tensor("B,K"):
    Y = Tensor(B=X.shape[0], K=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for n in Dim("K"):
            for k in Dim("K"):
                Y[b, n] += X[b, k] * (1.0 if k <= n else 0.0)
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,K", dtype="fp32")
    return {"output": cumsum_masked(x).data}
'''

VARIANCE = HEADER + '''
@kernel
def second_moment(X: Tensor("N,J", dtype="fp32")) -> Tensor("N"):
    V = Tensor(N=X.shape[0], dtype="fp32")
    inv = 1.0 / X.shape[1]
    for n in Dim("N", parallel=True):
        acc = 0.0
        for j in Dim("J"):
            acc += X[n, j] * X[n, j]
        V[n] = acc * inv
    return V

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,J", dtype="fp32")
    return {"output": second_moment(x).data}
'''

THREE_FACTOR = HEADER + '''
@kernel
def three_factor(A: Tensor("I,K", dtype="fp32"), Bm: Tensor("I,K", dtype="fp32"), Cm: Tensor("I,K", dtype="fp32")) -> Tensor("I"):
    Y = Tensor(I=A.shape[0], dtype="fp32")
    for i in Dim("I", parallel=True):
        for k in Dim("K"):
            Y[i] += A[i, k] * Bm[i, k] * Cm[i, k]
    return Y

def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs["a"], "I,K", dtype="fp32")
    b = Tensor.from_numpy(inputs["b"], "I,K", dtype="fp32")
    c = Tensor.from_numpy(inputs["c"], "I,K", dtype="fp32")
    return {"output": three_factor(a, b, c).data}
'''

GATHER = HEADER + '''
from solar.einsum_lib import gather

def run_model(inputs):
    clear_dims()
    wte = Tensor.from_numpy(inputs["wte"], "V,D", role="weight", dtype="fp32")
    ids = Tensor.from_numpy(inputs["ids"], "B,S", dtype="int64")
    return {"output": gather(wte, ids, axis="V").data}
'''


def _rng():
    return np.random.default_rng(59)


def test_bias_add_is_other_work_only(tmp_path):
    graph, out = _traced_analysis(BIAS_ADD, {"x": _rng().standard_normal((M, N)).astype(np.float32),
                                             "b": _rng().standard_normal((N,)).astype(np.float32)}, tmp_path)
    layer = _kernel_layer(out, "bias_add")
    assert (layer["macs"], layer["other_ops"]) == (0, M * N)
    g = next(v for k, v in graph["layers"].items() if k.startswith("bias_add#"))
    assert (g["loop_mac_statements"], g["loop_other_statements"]) == (0, M * N)


def test_matmul_with_scale_bias_epilogue_splits_units(tmp_path):
    # the Matmul_Scaling pattern: K MACs per output; the accumulator init and the
    # scale+bias epilogue are one statement each per output -> other_ops
    graph, out = _traced_analysis(MATMUL_SCALE_BIAS, {"x": _rng().standard_normal((M, K)).astype(np.float32),
                                                      "w": _rng().standard_normal((N, K)).astype(np.float32),
                                                      "b": _rng().standard_normal((N,)).astype(np.float32)}, tmp_path)
    layer = _kernel_layer(out, "matmul_scale_bias")
    assert layer["macs"] == M * N * K
    assert layer["other_ops"] == 2 * M * N
    assert layer["flops"] == 2 * M * N * K
    g = next(v for k, v in graph["layers"].items() if k.startswith("matmul_scale_bias#"))
    assert (g["loop_mac_statements"], g["loop_other_statements"]) == (M * N * K, 2 * M * N)


def test_masked_cumsum_is_not_a_mac(tmp_path):
    B, KK = 2, 4
    graph, out = _traced_analysis(CUMSUM_MASKED, {"x": _rng().standard_normal((B, KK)).astype(np.float32)}, tmp_path)
    layer = _kernel_layer(out, "cumsum_masked")
    assert (layer["macs"], layer["other_ops"]) == (0, B * KK * KK)


def test_self_product_accumulation_is_a_mac(tmp_path):
    NN, J = 3, 6
    graph, out = _traced_analysis(VARIANCE, {"x": _rng().standard_normal((NN, J)).astype(np.float32)}, tmp_path)
    layer = _kernel_layer(out, "second_moment")
    assert layer["macs"] == NN * J                  # the square-accumulate
    assert layer["other_ops"] == 2 * NN             # acc init + the scaled write


def test_three_factor_accumulation_is_one_mac_plus_one_multiply(tmp_path):
    I, KK = 3, 4
    arr = lambda: _rng().standard_normal((I, KK)).astype(np.float32)
    graph, out = _traced_analysis(THREE_FACTOR, {"a": arr(), "b": arr(), "c": arr()}, tmp_path)
    layer = _kernel_layer(out, "three_factor")
    assert (layer["macs"], layer["other_ops"]) == (I * KK, I * KK)


# ---------------------------------------------------------------------------
# 3. built-in gather: work = gathered elements, traffic = index + gathered rows + output
# ---------------------------------------------------------------------------

def test_builtin_gather_is_costed_as_a_gather(tmp_path):
    V, D, B, S = 5, 3, 2, 2
    inputs = {"wte": _rng().standard_normal((V, D)).astype(np.float32),
              "ids": np.array([[0, 4], [2, 2]], dtype=np.int64)}
    graph, out = _traced_analysis(GATHER, inputs, tmp_path)
    layer = _kernel_layer(out, "gather")
    assert (layer["macs"], layer["other_ops"]) == (0, B * S * D)          # one address op per gathered element
    assert layer["unfused_elements"] == B * S + B * S * D + B * S * D     # index + gathered rows + output
    assert layer.get("cost_basis") == "gather_elements"
    assert layer.get("status") != "incomplete"


# ---------------------------------------------------------------------------
# 4. the reviewer's tiny candidates, bound byte-for-byte: loop EXTENT defects
#    (the call-local extent fix of the DSL loop-dimension MR); strict xfail
#    flips to a failure the moment that fix lands so the marker is removed then
# ---------------------------------------------------------------------------


# (was a strict xfail pending the call-local loop-extent MR; an earlier change is merged into this history and the extent is right)

# (was a strict xfail pending the loop-extent lineage; an earlier change is merged into this history and the extent is right)

# ---------------------------------------------------------------------------
# 5. the LSTM and Reformer candidates at tiny shapes (MAC-UNIT-CONVENTION-1 extension): pure-bias statements and
#    scale / mask statements are other_ops; the MAC field equals the contraction-only total
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. NONRETURNED-OUTPUT-BILLING-1: only tensors reachable from the returned values are external outputs; a
#    non-returned store is internal (removable under fusion) and recorded separately, never as boundary traffic
# ---------------------------------------------------------------------------

UNUSED_STORE = HEADER + '''
@kernel
def scale(X: Tensor("M,N", dtype="fp32"), s: float) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            Y[m, n] = X[m, n] * s
    return Y

@kernel
def store_copy(X: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            Y[m, n] = X[m, n]
    return Y

@kernel
def row_sum(X: Tensor("M,N", dtype="fp32")) -> Tensor("M"):
    R = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            R[m] += X[m, n]
    return R

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,N", dtype="fp32")
    y = scale(x, 2.0)
    unused = store_copy(y)      # computed, never returned, never consumed
    return {"output": row_sum(y).data}
'''


def test_non_returned_store_is_not_an_external_output(tmp_path):
    graph, out = _traced_analysis(UNUSED_STORE, {"x": _rng().standard_normal((M, N)).astype(np.float32)}, tmp_path)
    assert graph.get("model_output_provenance") == "dsl-return-v1"
    returned = {mo["tensor"] for mo in graph["model_outputs"]}
    assert returned == {"row_sum#0"}
    store = _kernel_layer(out, "store_copy")
    assert store["nonreturned_output_elements"] == M * N        # recorded separately ...
    assert store["model_output_elements"] == 0                   # ... never as boundary traffic
    assert out["total"]["external_output_bytes"] == M * 4        # only the returned row sums
    assert out["total"]["nonreturned_store_bytes"] == M * N * 4
    assert out["total"].get("status") != "incomplete"


# ---------------------------------------------------------------------------
# 7. round 2 (the review): the unrecorded-return marker reaches the analysis; present-but-invalid class evidence
#    fails closed; compound scalar divisors; branches of different class composition
# ---------------------------------------------------------------------------

def _perf(tmp_path, analysis_dir):
    from solar.perf.perf_model import EinsumGraphPerfModel
    return EinsumGraphPerfModel().predict(str(Path(analysis_dir) / "analysis.yaml"), str(tmp_path / "perf"),
                                          arch_config="H100_PCIe", precision="fp32")


@pytest.mark.parametrize("name,expr", [("copy", '{"o": row_sum(y).data.copy()}'), ("view", '{"o": y.data[:, ::2]}')])
def test_unrecorded_return_marker_reaches_analysis_and_pricing(tmp_path, name, expr):
    # return-provenance-1 (the review fixtures): a return that cannot be bound to a traced identity is carried
    # sidecar -> DSL graph -> exported graph -> analysis (total incomplete) -> perf (workload incomplete)
    code = UNUSED_STORE.replace('{"output": row_sum(y).data}', expr)
    graph, out = _traced_analysis(code, {"x": np.ones((M, N), np.float32)}, tmp_path)
    md = yaml.safe_load((tmp_path / "dsl" / "toy_einsum_lib.yaml").read_text())
    assert md.get("returned_outputs_unrecorded") and graph.get("returned_outputs_unrecorded") == md["returned_outputs_unrecorded"]
    assert out["total"]["status"] == "incomplete"
    assert out["total"]["incomplete_layers"].get("returned_outputs") == "returned_outputs_unrecorded"
    p = _perf(tmp_path, tmp_path / "analysis")
    assert p["workload"]["status"] == "incomplete"


@pytest.mark.parametrize("expr,expected_bytes", [
    ('{"a": row_sum(y).data}', M * 4),            # returned kernel output
    ('{"a": y.data, "b": y.data}', M * N * 4),    # one identity returned twice: charged once
    ('{"a": inputs["x"]}', 0),                    # a returned INPUT is already charged as the external read
])
def test_return_provenance_positive_controls(tmp_path, expr, expected_bytes):
    code = UNUSED_STORE.replace('{"output": row_sum(y).data}', expr)
    graph, out = _traced_analysis(code, {"x": np.ones((M, N), np.float32)}, tmp_path)
    assert "returned_outputs_unrecorded" not in graph and graph.get("model_output_provenance") == "dsl-return-v1"
    assert out["total"].get("status") != "incomplete"
    assert out["total"]["external_output_bytes"] == expected_bytes
    if expected_bytes == 0:
        assert out["total"]["external_input_bytes"] == M * N * 4 and out["total"]["fused_bytes"] == M * N * 4


def _reanalyze(graph, tmp_path):
    p = tmp_path / "graph.yaml"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(graph, sort_keys=False))
    return EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)


@pytest.mark.parametrize("variant,fields", [
    ("mac_gt_count", {"loop_mac_statements": 61, "loop_other_statements": 0}),
    ("sum_lt_count", {"loop_mac_statements": 1, "loop_other_statements": 1}),
    ("negative", {"loop_mac_statements": -1, "loop_other_statements": 84}),
    ("string", {"loop_mac_statements": "60", "loop_other_statements": 24}),
    ("missing_other", {"loop_mac_statements": 60}),
    ("inflated_inside_guard", {"loop_mac_statements": 60, "loop_other_statements": 10 ** 12}),
    ("extras_inconsistent", {"loop_mac_statements": 60, "loop_other_statements": 24, "loop_extra_multiplies": 6}),   # 60+24-6 != 84
])
def test_present_but_invalid_class_evidence_fails_closed(tmp_path, variant, fields):
    # class-guard-2 (the review probes verbatim + one extra): known-but-invalid classes never fall through to
    # the whole-kernel split; the layer, the total and the perf workload are incomplete (class_evidence_invalid)
    inp = {"x": np.ones((M, K), np.float32), "w": np.ones((N, K), np.float32), "b": np.ones(N, np.float32)}
    graph, _ = _traced_analysis(MATMUL_SCALE_BIAS, inp, tmp_path / "original")
    lid = next(k for k in graph["layers"] if k.startswith("matmul_scale_bias#"))
    for k in ("loop_mac_statements", "loop_other_statements", "loop_extra_multiplies"):
        graph["layers"][lid].pop(k, None)
    graph["layers"][lid].update(fields)
    a = _reanalyze(graph, tmp_path / "changed"); v = a["layers"][lid]
    assert v["status"] == "incomplete" and v["incomplete_reason"] == "class_evidence_invalid", (variant, v.get("macs"), v.get("other_ops"))
    assert v["macs"] == 0 and v["other_ops"] == 0 and v["cost_basis"] == "incomplete_evidence" and "unit_basis" not in v
    assert a["total"]["status"] == "incomplete"
    assert _perf(tmp_path, tmp_path / "changed" / "analysis")["workload"]["status"] == "incomplete"


def test_absent_class_evidence_keeps_the_legacy_split_without_a_marker(tmp_path):
    # a record with NO class fields at all (legacy artifact) is the allowed legacy case: whole-kernel flag, complete
    inp = {"x": np.ones((M, K), np.float32), "w": np.ones((N, K), np.float32), "b": np.ones(N, np.float32)}
    graph, _ = _traced_analysis(MATMUL_SCALE_BIAS, inp, tmp_path / "original")
    lid = next(k for k in graph["layers"] if k.startswith("matmul_scale_bias#"))
    for k in ("loop_mac_statements", "loop_other_statements", "loop_extra_multiplies"):
        graph["layers"][lid].pop(k, None)
    a = _reanalyze(graph, tmp_path / "legacy"); v = a["layers"][lid]
    assert "status" not in v and "unit_basis" not in v and v["macs"] == M * N * K      # the old whole-kernel split
    assert a["total"].get("status") != "incomplete"


def test_class_evidence_reconciles_against_the_full_statement_count(tmp_path):
    graph, _ = _traced_analysis(THREE_FACTOR, {k: np.ones((3, 4), np.float32) for k in ("a", "b", "c")}, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("three_factor#"))
    # mac + other == statements + extra multiplies, and the extras are recorded
    assert g["loop_mac_statements"] == 12 and g["loop_other_statements"] == 12 and g["loop_extra_multiplies"] == 12
    assert g["loop_mac_statements"] + g["loop_other_statements"] - g["loop_extra_multiplies"] == g["loop_statements"]


@pytest.mark.parametrize("stmt,loop_vars,expected", [
    ("Y[i] += X[i, k] * W[k] / n", "ik", (MAC, 0)),
    ("Y[i] += X[i, k] * W[k] / (n + 1)", "ik", (MAC, 0)),
    ("Y[i] += X[i, k] * W[k] / (2 * 3)", "ik", (MAC, 0)),
    ("Y[i] += X[i, k] * W[k] / (k + 1)", "ik", (MAC, 0)),
    ("Y[i] += X[i, k] * W[k] / sqrt(n)", "ik", (MAC, 0)),
    ("Y[i] += X[i, k] * W[k] / T[k]", "ik", (MAC, 1)),           # a tensor-access denominator is a value factor
    ("Y[i] += X[i, k] / T[k]", "ik", (MAC, 0)),                  # product of two value factors through a division
    ("Y[i] += X[i, k] / (k + 1)", "ik", (OTHER, 0)),             # one value factor scaled: not a MAC
])
def test_divisor_forms(stmt, loop_vars, expected):
    assert classify_source_statement(stmt, set(loop_vars)) == expected


@pytest.mark.parametrize("denom", ["6", "(2 * 3)", "(k + 1)", "n"])
def test_scaled_contraction_bills_the_contraction_macs(tmp_path, denom):
    # divisor-class-3 (the review pipeline forms): the base bills 6 MACs for every scalar divisor
    code = HEADER + '''
@kernel
def scaled(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    n = X.shape[1]
    for m in Dim("M"):
        for k in Dim("K"):
            Y[m] += X[m, k] * W[m, k] / DENOM
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")
    return {"output": scaled(x, w).data}
'''.replace("DENOM", denom)
    graph, out = _traced_analysis(code, {"x": np.ones((2, 3), np.float32), "w": np.ones((2, 3), np.float32)}, tmp_path)
    layer = _kernel_layer(out, "scaled")
    assert (layer["macs"], layer["other_ops"]) == (6, 0), (denom, layer["macs"], layer["other_ops"])


BRANCH_KERNEL = HEADER + '''
@kernel
def branch(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M"):
        for k in Dim("K"):
            if X[m, k] > 0:
                Y[m] += X[m, k]
            else:
                Y[m] += X[m, k] * W[m, k]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")
    return {"output": branch(x, w).data}
'''


@pytest.mark.parametrize("sign", [-1, 1])
def test_branches_of_different_class_are_accounted_per_class_maximum_and_marked(tmp_path, sign):
    # TARGET-KB-BRANCH-CLASS-4 (the review probes): the branches differ in class (add vs multiply-accumulate); the
    # kernel bills the per-class maximum over the branches (6 MACs AND 6 other) and is marked incomplete
    # (branch_class_uncertain) -- never one branch's classes silently
    inp = {"x": np.full((2, 3), sign, np.float32), "w": np.full((2, 3), 2, np.float32)}
    graph, out = _traced_analysis(BRANCH_KERNEL, inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("branch#"))
    assert g["class_evidence_uncertain"] == "branch_class_uncertain"
    layer = _kernel_layer(out, "branch")
    assert (layer["macs"], layer["other_ops"]) == (6, 6)
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "branch_class_uncertain"
    assert out["total"]["status"] == "incomplete"


def test_branches_of_equal_class_composition_are_not_marked(tmp_path):
    code = BRANCH_KERNEL.replace("Y[m] += X[m, k]\n", "Y[m] += X[m, k] * W[m, k] * 0.5\n")
    graph, out = _traced_analysis(code, {"x": np.ones((2, 3), np.float32), "w": np.ones((2, 3), np.float32)}, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("branch#"))
    assert "class_evidence_uncertain" not in g
    layer = _kernel_layer(out, "branch")
    assert (layer["macs"], layer["other_ops"]) == (6, 0) and "status" not in layer


# ---------------------------------------------------------------------------
# 8. round 3 (the review): value PROVENANCE decides what a factor is -- shape / size / len metadata, literals,
#    loop indices and index arithmetic are static scales; a tensor NAME subscripted by a loop variable is a
#    tensor value; a local alias inherits the provenance of what it was assigned; an inline conditional is
#    classified per arm (equal arms exact, differing arms per-class maximum + branch_class_uncertain); anything
#    unproven is never a complete per-statement classification (class_evidence_unproven)
# ---------------------------------------------------------------------------

from solar.einsum_dsl.statement_class import classify_source_records  # noqa: E402

TENSORS = frozenset({"X", "W", "Y"})


@pytest.mark.parametrize("stmt,expected,flags", [
    ("Y[m] += X[m, k] / W.shape[1]", [(OTHER, 0)], set()),                 # shape-divisor-5
    ("Y[m] += X[m, k] / (W.shape[1] + 1)", [(OTHER, 0)], set()),
    ("Y[m] += X[m, k] * W.shape[1]", [(OTHER, 0)], set()),                 # shape metadata is a static scale either side
    ("Y[m] += X[m, k] / X.size(1)", [(OTHER, 0)], set()),
    ("Y[m] += X[m, k] / len(X)", [(OTHER, 0)], set()),
    ("Y[m] += X[m, k] / W[m, k]", [(MAC, 0)], set()),                      # tensor denominator: a value
    ("Y[m] += X[m, k] * W[k, m]", [(MAC, 0)], set()),
    ("Y[m] += X[m, k] if X[m, k] > 0 else X[m, k] * W[m, k]", [(MAC, 0), (OTHER, 0)], {"branch_class_uncertain"}),   # ifexp-class-6
    ("Y[m] += X[m, k] * W[m, k] if X[m, k] > 0 else X[m, k] * W[m, k] * 0.5", [(MAC, 0)], set()),                    # equal arms: exact
    ("Y[m] += (X[m, k] * W[m, k] if X[m, k] > 0 else X[m, k] * W[m, k] * 2.0) if k > 0 else X[m, k] * W[m, k]", [(MAC, 0)], set()),  # nested, all MAC
    ("Y[m] += (X[m, k] if X[m, k] > 0 else X[m, k] * W[m, k]) if k > 0 else X[m, k]", [(MAC, 0), (OTHER, 0)], {"branch_class_uncertain"}),
    ("Y[m] += X[m, k] / cfg[k]", [(MAC, 0)], {"class_evidence_unproven"}),  # a subscript on a non-tensor name: unproven
    ("Y[m] += X[m, k] * W[0, 0]", [(MAC, 0)], {"class_evidence_unproven"}), # tensor name without a loop-variable index
])
def test_round3_provenance_records(stmt, expected, flags):
    records, got = classify_source_records(stmt, "mk", tensor_names=TENSORS)
    assert records == expected and got == flags


def test_alias_resolution_in_records():
    body = "denom = W[m, k]\nY[m] += X[m, k] / denom"
    records, flags = classify_source_records(body, "mk", tensor_names=TENSORS)
    assert records == [(OTHER, 0), (MAC, 0)] and flags == set()          # TARGET-KB-DIVISOR-ALIAS-7
    body = "n = X.shape[1]\nY[m] += X[m, k] / n"
    assert classify_source_records(body, "mk", tensor_names=TENSORS) == ([(OTHER, 0), (OTHER, 0)], set())
    body = "t = X[m, k] * 2.0\nY[m] += t * W[m, k]"
    assert classify_source_records(body, "mk", tensor_names=TENSORS) == ([(OTHER, 0), (MAC, 0)], set())
    body = "Y[m] += X[m, k] / unknown_name"
    assert classify_source_records(body, "mk", tensor_names=TENSORS) == ([(MAC, 0)], {"class_evidence_unproven"})


def _probe(body):
    return HEADER + '''
@kernel
def probe(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M"):
        for k in Dim("K"):
BODY
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")
    return {"output": probe(x, w).data}
'''.replace("BODY", "\n".join("            " + l for l in body.splitlines()))


def _probe_run(body, x, w, tmp_path):
    inp = {"x": np.full((2, 3), x, np.float32), "w": np.full((2, 3), w, np.float32)}
    graph, out = _traced_analysis(_probe(body), inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("probe#"))
    return g, _kernel_layer(out, "probe"), out["total"]


@pytest.mark.parametrize("denom", ["W.shape[1]", "(W.shape[1] + 1)"])
def test_shape_denominator_is_a_scale(tmp_path, denom):
    # the reviewer's shape-divisor-5 cases: a scalar-scaled addition, 0 MACs / 6 other, complete
    g, layer, total = _probe_run("Y[m] += X[m, k] / " + denom, 6, 2, tmp_path)
    assert (layer["macs"], layer["other_ops"]) == (0, 6) and "status" not in layer and total.get("status") != "incomplete"
    assert "class_evidence_uncertain" not in g


@pytest.mark.parametrize("sign", [-1, 1])
def test_inline_conditional_of_differing_arms_is_per_class_maximum_and_marked(tmp_path, sign):
    g, layer, total = _probe_run("Y[m] += X[m, k] if X[m, k] > 0 else X[m, k] * W[m, k]", sign, 2, tmp_path)
    assert g["class_evidence_uncertain"] == "branch_class_uncertain"
    assert (layer["macs"], layer["other_ops"]) == (6, 6)
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "branch_class_uncertain"
    assert total["status"] == "incomplete"


def test_inline_conditional_of_equal_mac_arms_is_exact(tmp_path):
    g, layer, total = _probe_run("Y[m] += X[m, k] * W[m, k] if X[m, k] > 0 else X[m, k] * W[m, k] * 0.5", -1, 2, tmp_path)
    assert "class_evidence_uncertain" not in g
    assert (layer["macs"], layer["other_ops"]) == (6, 0) and "status" not in layer


def test_tensor_denominator_local_alias_is_a_mac_plus_the_alias_statement(tmp_path):
    # TARGET-KB-DIVISOR-ALIAS-7: the alias carries the tensor element's provenance; the local assignment
    # is the only other work
    g, layer, total = _probe_run("denom = W[m, k]\nY[m] += X[m, k] / denom", 4, 2, tmp_path)
    assert (layer["macs"], layer["other_ops"]) == (6, 6) and "status" not in layer
    g, layer, total = _probe_run("Y[m] += X[m, k] / W[m, k]", 4, 2, tmp_path)
    assert (layer["macs"], layer["other_ops"]) == (6, 0) and "status" not in layer


def test_unproven_provenance_is_never_a_complete_classification(tmp_path):
    g, layer, total = _probe_run("Y[m] += X[m, k] * W[0, 0]", 1, 2, tmp_path)
    assert g["class_evidence_uncertain"] == "class_evidence_unproven"
    assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "class_evidence_unproven"
    assert total["status"] == "incomplete" and total["incomplete_layers"][next(k for k in total["incomplete_layers"] if k.startswith("probe#"))] == "class_evidence_unproven"
    # the counts are still reported (the statement MAY be a MAC) -- they are just not complete evidence
    assert (layer["macs"], layer["other_ops"]) == (6, 0)


# ---------------------------------------------------------------------------
# 9. round 4 (the review): the structural STATIC / VALUE / UNPROVEN rule
#    call-provenance-8, rebind-provenance-9, unproven-join-10,
#    bool-value-11
# ---------------------------------------------------------------------------
UNPROVEN_FLAG = {"class_evidence_unproven"}


@pytest.mark.parametrize("body,expected,flags", [
    # CALL-PROVENANCE-8: a call is STATIC only for an unshadowed pure-static builtin on proven-static arguments
    ("Y[m] += X[m, k] * int(W[m, k])", [(MAC, 0)], set()),                       # a cast of a tensor value is a value
    ("Y[m] += X[m, k] * float(W[m, k])", [(MAC, 0)], set()),
    ("Y[m] += X[m, k] * int(W.shape[1])", [(OTHER, 0)], set()),                  # a cast of shape metadata stays static
    ("Y[m] += X[m, k] * abs(W[m, k])", [(MAC, 0)], set()),
    ("Y[m] += X[m, k] * exp(W[m, k])", [(MAC, 0)], set()),                       # a DSL math function of a value is a value
    ("Y[m] += X[m, k] / sqrt(W.shape[1])", [(OTHER, 0)], set()),                 # a DSL math function of a static is static
    ("abs = lambda v: W[m, k]\nY[m] += X[m, k] * abs(k)", [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),   # shadowed builtin
    ("f = lambda v: W[m, k]\nY[m] += X[m, k] / f(k)", [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),       # unknown callable
    ("Y[m] += X[m, k] / g(k)", [(MAC, 0)], UNPROVEN_FLAG),                       # unknown callable, static argument
    ("Y[m] += X[m, k] / W.data[m, k].item()", [(MAC, 0)], UNPROVEN_FLAG),        # method on a tensor-derived receiver
    ("Y[m] += X[m, k] / W.data.max()", [(MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] / np.exp(W[m, k])", [(MAC, 0)], UNPROVEN_FLAG),            # module-qualified call: not proven
    ("Y[m] += X[m, k] / X.size(1)", [(OTHER, 0)], set()),                        # size() of a tensor NAME stays static
    ("Y[m] += X[m, k] / len(X)", [(OTHER, 0)], set()),
    # UNPROVEN-JOIN-10: unproven dominates every join
    ("Y[m] += X[m, k] / (W[m, k] + W[0, 0])", [(MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] / max(W[m, k], W[0, 0])", [(MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] * (W[m, k] if k > 0 else W[0, 0])", [(MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] * (W[m, k] or W[0, 0])", [(MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] * W[m, W[0, 0]]", [(MAC, 0)], UNPROVEN_FLAG),              # an unproven index
    # BOOL-VALUE-11: or / and return an operand
    ("Y[m] += X[m, k] * (W[m, k] or 1.0)", [(MAC, 0)], set()),
    ("Y[m] += X[m, k] * (W[m, k] and W[k, m])", [(MAC, 0)], set()),
    ("Y[m] += X[m, k] * ((k <= m) or (m == 0))", [(OTHER, 0)], set()),           # a boolean of comparisons is a mask
    ("Y[m] += X[m, k] * (not W[m, k])", [(OTHER, 0)], set()),                     # an explicit boolean is a mask
    ("Y[m] += X[m, k] * bool(W[m, k])", [(OTHER, 0)], set()),
    # REBIND-PROVENANCE-9: every binding form re-derives; differing classes are unproven; flow order
    ("denom, junk = W[m, k], 0\nY[m] += X[m, k] / denom", [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),   # unpacking is not a proven binding form
    ("denom = 1.0\ndenom, junk = W[m, k], 0\nY[m] += X[m, k] / denom", [(OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),
    ("denom = 1.0\ndenom = junk = W[m, k]\nY[m] += X[m, k] / denom", [(OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),
    ("denom = junk = W[m, k]\nY[m] += X[m, k] / denom", [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),   # chained target: outside the grammar (round 5)
    ("j = k\nk = W[m, k]\nY[m] += X[m, j] * k", [(OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),   # a rebound loop variable
    ("j = k\nY[m] += X[m, j] * W[j, m]", [(OTHER, 0), (MAC, 0)], set()),        # an index alias is a loop index
    ("denom = W[m, k]\ndenom = 2.0\nY[m] += X[m, k] / denom", [(OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),
    ("Y[m] += X[m, k] / denom\ndenom = W[m, k]", [(MAC, 0), (OTHER, 0)], UNPROVEN_FLAG),  # use before binding
    ("a, *rest = W[m, k], 0, 1\nY[m] += X[m, k] / a", [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG),   # starred unpacking: unsupported
    ("acc = 0.0\nacc += X[m, k] * W[m, k]\nY[m] = acc * 2.0", [(OTHER, 0), (MAC, 0), (OTHER, 0)], set()),   # an accumulator is a runtime scalar
])
def test_round4_structural_rule_records(body, expected, flags):
    records, got = classify_source_records(body, "mk", tensor_names=TENSORS)
    assert (records, got) == (expected, flags)


# the reviewer's 21 full-pipeline cases (checks-b022bfb0/test_review2_round3.py): the correct exact class
# OR an incomplete analysis is acceptable; the cases the review saw pass keep their exact classes.
REVIEW2_CASES = [
    # label, body, output per row (x=4, w=2), exact (macs, other) or None when an incomplete marker is required
    ("int_element", "Y[m] += X[m,k] * int(W[m,k])", 24, (6, 0)),
    ("int_shape", "Y[m] += X[m,k] * int(W.shape[1])", 36, (0, 6)),
    ("abs_element", "Y[m] += X[m,k] * abs(W[m,k])", 24, (6, 0)),
    ("method_item", "Y[m] += X[m,k] / W.data[m,k].item()", 6, None),
    ("method_indexed_arg", "Y[m] += X[m,k] / W.data.__getitem__((m,k))", 6, None),
    ("method_max", "Y[m] += X[m,k] / W.data.max()", 6, None),
    ("nested_data_subscript", "Y[m] += X[m,k] / W.data[m,k]", 6, None),
    ("nested_row_subscript", "Y[m] += X[m,k] / W.data[m][k]", 6, None),
    ("literal_tensor_subscript", "Y[m] += X[m,k] / W[0,0]", 6, None),
    ("mixed_unproven_sum", "Y[m] += X[m,k] / (W[m,k] + W[0,0])", 3, None),
    ("mixed_unproven_call", "Y[m] += X[m,k] / max(W[m,k], W[0,0])", 6, None),
    ("tuple_fresh", "denom, junk = W[m,k], 0\nY[m] += X[m,k] / denom", 6, None),
    ("tuple_rebind", "denom = 1.0\ndenom, junk = W[m,k], 0\nY[m] += X[m,k] / denom", 6, None),
    ("chain_rebind", "denom = 1.0\ndenom = junk = W[m,k]\nY[m] += X[m,k] / denom", 6, None),
    ("loop_rebind", "j = k\nk = W[m,k]\nY[m] += X[m,j] * k", 24, None),
    ("shadowed_abs", "abs = lambda v: W[m,k]\nY[m] += X[m,k] * abs(k)", 24, None),
    ("unknown_callable", "f = lambda v: W[m,k]\nY[m] += X[m,k] / f(k)", 6, None),
    ("mixed_bindings", "denom = W[m,k]\ndenom = 2.0\nY[m] += X[m,k] / denom", 6, None),
    ("simple_alias", "denom = W[m,k]\nY[m] += X[m,k] / denom", 6, (6, 6)),
    ("arithmetic_alias", "denom = W[m,k] + 1\nY[m] += X[m,k] / denom", 4, (6, 6)),
    ("boolean_value", "Y[m] += X[m,k] * (W[m,k] or 1.0)", 24, (6, 0)),
]


@pytest.mark.parametrize("label,body,output,exact", REVIEW2_CASES, ids=[c[0] for c in REVIEW2_CASES])
def test_review2_round3_pipeline(label, body, output, exact, tmp_path):
    from solar.einsum_lib.verify import execute_translated
    inp = {"x": np.full((2, 3), 4, np.float32), "w": np.full((2, 3), 2, np.float32)}
    outputs, _ = execute_translated(_probe(body), {k: v.shape for k, v in inp.items()}, all_inputs=inp, trace=True, fast=False)
    assert np.array_equal(outputs["output"], np.full(2, output, np.float32))
    g, layer, total = _probe_run(body, 4, 2, tmp_path)
    if exact is None:
        # the value provenance is not proven: counted as a value, flagged, incomplete through the analysis
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), (label, layer["macs"], layer["other_ops"])
        assert layer["status"] == "incomplete" and total["status"] == "incomplete", label
    else:
        assert (layer["macs"], layer["other_ops"]) == exact, (label, layer["macs"], layer["other_ops"], total.get("status"))
        assert "status" not in layer and total.get("status") != "incomplete", label
        assert "class_evidence_uncertain" not in g, label


# ---------------------------------------------------------------------------
# 10. round 5 (the review): the SUPPORTED GRAMMAR -- anything outside it makes the
#     whole kernel class_evidence_unproven; every entry point through provenance();
#     a real fixpoint over the loop nest; module-namespace shadowing
#     conditional-binding-12, module-callee-13, provenance-bypass-14,
#     shape-index-15, TARGET-KB-LOOP-PROVENANCE-16
# ---------------------------------------------------------------------------
def _probe5(body, preamble="", scalar=False, after_loop=""):
    """the reviewer's round-4 probe program: optional module preamble, scalar parameter, post-loop statements."""
    code = HEADER + preamble + '''
@kernel
def probe(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32")SCALAR) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    for m in Dim("M"):
        for k in Dim("K"):
BODY
AFTER
    return Y
def run_model(inputs):
    global GLOBAL_W
    GLOBAL_W = inputs["w"]
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "M,K", dtype="fp32")
    return {"output": probe(x,wARG).data}
'''
    return (code.replace("SCALAR", ", s: float" if scalar else "").replace("ARG", ", 2.0" if scalar else "")
            .replace("BODY", "\n".join("            " + l for l in body.splitlines()))
            .replace("AFTER", "\n".join("        " + l for l in after_loop.splitlines())))


def _probe5_run(code, tmp_path, expected):
    from solar.einsum_lib.verify import execute_translated
    inp = {"x": np.full((2, 3), 4, np.float32), "w": np.full((2, 3), 2, np.float32)}
    outputs, _ = execute_translated(code, {k: v.shape for k, v in inp.items()}, all_inputs=inp, trace=True, fast=False)
    exp = np.array(expected if isinstance(expected, list) else [expected] * 2, np.float32)
    assert np.array_equal(outputs["output"], exp), (outputs["output"], exp)
    graph, out = _traced_analysis(code, inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("probe#"))
    return g, _kernel_layer(out, "probe"), out["total"]


# the reviewer's round-4 structural probes (checks-e7965891/test_review2_round4.py + followup): exact MACs or incomplete
REVIEW2_ROUND4 = [
    ("star_builtin", _probe5("Y[m] += X[m,k] * max(*(W[m,k],1.0))"), 24, None),
    ("star_static", _probe5("Y[m] += X[m,k] * max(*(1.0,2.0))"), 24, None),
    ("double_star_empty", _probe5("Y[m] += X[m,k] * round(W[m,k], **{})"), 24, None),
    ("comprehension_factor", _probe5("Y[m] += X[m,k] * sum([W[m,j] for j in range(3)])"), 72, None),
    ("comprehension_alias", _probe5("ds = [W[m,j] for j in range(3)]\nY[m] += X[m,k] * ds[k]"), 24, None),
    ("walrus_factor", _probe5("Y[m] += X[m,k] * (s := W[m,k])"), 24, None),
    ("ifexp_walrus_test", _probe5("Y[m] += X[m,k]*W[m,k] if (s := W[0,0]) else X[m,k]*W[m,k]"), 24, None),
    ("ifexp_known_test", _probe5("Y[m] += X[m,k]*W[m,k] if W[m,k]>0 else X[m,k]*W[m,k]"), 24, 6),
    ("nested_ifexp_test", _probe5("f = lambda z: True\nY[m] += (X[m,k]*W[m,k] if f(k) else X[m,k]*W[m,k]) if k>=0 else X[m,k]*W[m,k]"), 24, None),
    ("ifexp_unknown_arm", _probe5("f = lambda z: W[m,k]\nY[m] += f(k) if k>=0 else X[m,k]"), 6, None),
    ("nonproduct_call", _probe5("f = lambda z: W[m,k]\nY[m] += f(k)"), 6, None),
    ("ifexp_product_value_unproven", _probe5("Y[m] += X[m,k] * (W[m,k] if k>=0 else W[0,0])"), 24, None),
    ("shape_unknown_index", _probe5("Y[m] += X[m,k] / W.shape[int(W[0,0])-1]"), 4, None),
    ("shape_unknown_callable_index", _probe5("f = lambda z: 1\nY[m] += X[m,k] * W.shape[f(k)]"), 36, None),
    ("tensor_unknown_callable_index", _probe5("f = lambda z: 1\nY[m] += X[m,k] * W[m,f(k)]"), 24, None),
    ("global_unknown_name", _probe5("Y[m] += X[m,k] * GLOBAL_VALUE", "GLOBAL_VALUE = 2.0\n"), 24, 0),           # issue #61 LITERAL-CONSTANT: a once-bound float literal is a scale (VALUE arithmetic, 0 MAC); the reviewer's round-4 file expected a refusal
    ("global_shadowed_abs", _probe5("Y[m] += X[m,k] * abs(k)", "def abs(z):\n    return GLOBAL_W[0,z]\n"), 24, None),
    ("global_shadowed_exp", _probe5("Y[m] += X[m,k] * exp(k)", "def exp(z):\n    return GLOBAL_W[0,z]\n"), 24, None),
    ("module_qualified", _probe5("Y[m] += X[m,k] * np.float64(W[m,k])", "import numpy as np\n"), 24, None),
    ("chained_method", _probe5("Y[m] += X[m,k] / W.data[m,k].astype(\"float32\").item()"), 6, None),
    ("index_after_loop", _probe5("Y[m] += X[m,k]", after_loop="Y[m] += X[m,0] * k"), 20, None),
    ("short_circuit_walrus_parameter", _probe5("if X[m,k] < 0 and (s := 1.0):\n    tmp = 0\nelse:\n    tmp = 0\nY[m] += X[m,k] * s", scalar=True), 24, None),
    ("parameter_rebound", _probe5("Y[m] += X[m,k] * s\ns = 1.0", scalar=True), [16, 12], None),
    ("conditional_walrus_rhs", _probe5("flag = (s := 1.0) if X[m,k]<0 else 0\nY[m] += X[m,k]*s", scalar=True), 24, None),
    ("pure_static_builtin", _probe5("Y[m] += X[m,k] * max(k,1)"), 16, 0),
    ("nested_ifexp_static_known", _probe5("Y[m] += X[m,k]*(W[m,k] if k>0 else (W[m,k] if m>0 else W[m,k]))"), 24, 6),
    ("statement_if_unknown_test", _probe5("f = lambda z: True\nif f(k):\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,k]"), 24, None),
]


@pytest.mark.parametrize("label,code,expected,macs", REVIEW2_ROUND4, ids=[c[0] for c in REVIEW2_ROUND4])
def test_review2_round4_structural_pipeline(label, code, expected, macs, tmp_path):
    g, layer, total = _probe5_run(code, tmp_path, expected)
    if macs is None:
        assert total["status"] == "incomplete", (label, layer["macs"], layer["other_ops"], g.get("class_evidence_uncertain"))
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), (label, g.get("class_evidence_uncertain"))
    else:
        assert total.get("status") != "incomplete" and layer["macs"] == macs, (label, layer["macs"], layer["other_ops"], total.get("status"))
        assert "class_evidence_uncertain" not in g, label


LOOP_CARRIED = HEADER + '''
@kernel
def probe(X: Tensor("K", dtype="fp32"), W: Tensor("K", dtype="fp32")) -> Tensor("O"):
    Y = Tensor(O=1, dtype="fp32")
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    e = 0.0
    for k in Dim("K"):
        Y[0] += X[k] * a
        a = b
        b = c
        c = d
        d = e
        e = W[k]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "K", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "K", dtype="fp32")
    return {"output": probe(x,w).data}
'''


def test_review2_loop_carried_chain_reaches_a_fixpoint_or_refuses(tmp_path):
    # TARGET-KB-LOOP-PROVENANCE-16: tensor values reach `a` after five trips; the analysis must converge or refuse
    from solar.einsum_lib.verify import execute_translated
    inp = {"x": np.full(8, 4, np.float32), "w": np.full(8, 2, np.float32)}
    out, _ = execute_translated(LOOP_CARRIED, {k: v.shape for k, v in inp.items()}, all_inputs=inp, trace=True, fast=False)
    assert out["output"].tolist() == [24.0]
    graph, a = _traced_analysis(LOOP_CARRIED, inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("probe#"))
    assert a["total"]["status"] == "incomplete" and "class_evidence_unproven" in g["class_evidence_uncertain"]


# grammar / fixpoint at the records level (classify_source_records builds the context from the text)
@pytest.mark.parametrize("body,flags", [
    ("Y[m] += X[m, k] * (s := W[m, k])", UNPROVEN_FLAG),                              # walrus anywhere
    ("if X[m, k] < 0 and (s := 1.0):\n    t = 0\nelse:\n    t = 0\nY[m] += X[m, k] * s", UNPROVEN_FLAG),
    ("Y[m] += X[m, k] * W[m, k] if (s := W[0, 0]) else X[m, k] * W[m, k]", UNPROVEN_FLAG),  # IfExp test through provenance
    ("Y[m] += X[m, k] * W[m, k] if W[m, k] > 0 else X[m, k] * W[m, k]", set()),            # a comparison test is a mask
    ("Y[m] += g(k)", UNPROVEN_FLAG),                                                   # non-product accumulation of an unknown call
    ("t = g(k)\nY[m] += X[m, k] * W[m, k]", UNPROVEN_FLAG),                            # a non-accumulation assign through provenance
    ("Y[m] += X[m, k] / W.shape[int(W[0, 0]) - 1]", UNPROVEN_FLAG),                     # shape index must be a literal
    ("Y[m] += X[m, k] * W.shape[k]", UNPROVEN_FLAG),
    ("Y[m] += X[m, k] / W.shape[1]", set()),
    ("Y[m] += X[m, k] * sum([W[m, j] for j in range(3)])", UNPROVEN_FLAG),              # comprehension
    ("Y[m] += X[m, k] * max(*(1.0, 2.0))", UNPROVEN_FLAG),                              # starred args
    ("Y[m] += X[m, k] / W.data[m, k].item()", UNPROVEN_FLAG),                           # attribute call on a tensor receiver
    ("Y[m] += X[m, k] * max(k, 1)", set()),                                             # pure-static builtin on statics
    ("a = 0.0\nb = 0.0\nY[m] += X[m, k] * a\na = b\nb = W[m, k]", UNPROVEN_FLAG),      # loop-carried chain: the fixpoint finds the tensor
])
def test_round5_grammar_records(body, flags):
    records, got = classify_source_records(body, "mk", tensor_names=TENSORS)
    assert got == flags, (records, got)


def test_round5_module_level_shadowing_of_a_builtin_refuses():
    from solar.einsum_dsl.statement_class import module_bound_names
    mod = ast.parse("import numpy as np\nGLOBAL = 2.0\ndef abs(z):\n    return GLOBAL\nfrom math import exp\n")
    assert {"np", "GLOBAL", "abs", "exp"} <= module_bound_names(mod)
    mod2 = ast.parse("from m import *\n")
    assert "*" in module_bound_names(mod2)


def test_round5_every_classification_entry_point_routes_through_provenance():
    # static call-graph walk of the classifier: every function that can return a record must call provenance()
    # itself or through a callee (no path returns a complete class without consulting it)
    import inspect
    from solar.einsum_dsl import statement_class as sc
    tree = ast.parse(inspect.getsource(sc))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {name: {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)} for name, fn in funcs.items()}

    def reaches_provenance(name, seen=()):
        if name == "provenance":
            return True
        return any(reaches_provenance(c, seen + (name,)) for c in calls.get(name, ()) if c in funcs and c not in seen)

    for entry in ("classify_records", "_classify_rhs", "_classify_product", "classify_statement", "classify_source_statement", "classify_source_records"):
        assert reaches_provenance(entry), entry
    # and dynamically: a complete (unflagged) record is never produced without a provenance() call on the statement
    import solar.einsum_dsl.statement_class as mod
    calls_seen = []
    real = mod.provenance

    def spy(node, lv, ctx, *a, **kw):
        calls_seen.append(node)
        return real(node, lv, ctx, *a, **kw)

    mod.provenance = spy
    try:
        for body in ("Y[m] += X[m, k] * W[m, k]", "Y[m] = X[m, k] + 1.0", "acc = 0.0", "Y[m] += X[m, k]"):
            calls_seen.clear()
            records, flags = classify_source_records(body, "mk", tensor_names=TENSORS)
            assert calls_seen, body
    finally:
        mod.provenance = real


# --- the BLOCK-INDEX PRODUCTION (manager decision): k // _BLK with a module-level int literal is STATIC ---
BLOCK_MODULE = ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_BLK = 16\n_HD = 128\n_G = _NH // _KV\n_F = 1.5\n_TWICE = 4\n_TWICE = 8\n")


@pytest.mark.parametrize("body,expected,flags", [
    ("acc += XQ[m, k] * XS[m, k // _BLK] * (WV[n, k] * WS[n, k // _BLK])", [(MAC, 2)], set()),   # the scaled contraction: 4 value factors = 1 MAC + 2 multiplies
    ("sx = X[m, k] * EN[m, k // _BLK]", [(OTHER, 0)], set()),                                  # the quantisation
    ("Y[m, k // _BLK] = max(Y[m, k // _BLK], abs(X[m, k]))", [(OTHER, 0)], set()),              # the block amax
    ("ws = W[n, k] * WS[n, k // _BLK]\nacc += X[m, k] * XS[m, k // _BLK] * ws", [(OTHER, 0), (MAC, 1)], set()),   # row 205's scale alias
    ("Y[m] += X[m, k] * W[m, (k + 1) // _BLK]", [(MAC, 0)], set()),                            # STATIC arithmetic on the left
    ("Y[m] += X[m, k] * W[m, k // 16]", [(MAC, 0)], set()),                                    # a literal divisor was always static
    ("Y[m] += X[m, k] * W[m, k // _F]", [(MAC, 0)], UNPROVEN_FLAG),                            # a float literal is not the production
    ("Y[m] += X[m, k] * W[m, k // _G]", [(MAC, 0)], UNPROVEN_FLAG),                            # bound to an expression, not a literal
    ("Y[m] += X[m, k] * W[m, k // _TWICE]", [(MAC, 0)], UNPROVEN_FLAG),                        # bound twice
    # issue #61 (LITERAL-CONSTANT production): the once-bound int literal is STATIC in every STATIC role, so the
    # four forms below -- refused by the block-index production alone (round 5/6) -- are now proven
    ("Y[m] += X[m, k] * W[m, k * _BLK]", [(MAC, 0)], set()),                                   # index arithmetic with the literal
    ("Y[m] += X[m, k] * W[m, _HD // 2 + k]", [(MAC, 0)], set()),                               # the literal as a dividend
    ("Y[m] += X[m, k] * _BLK", [(OTHER, 0)], set()),                                           # the literal as a factor: a scale (one value factor)
    ("Y[m] += X[m, k] * W[m, X[m, k] // _BLK]", [(MAC, 0)], set()),                            # a VALUE dividend over the literal: a declared gather index
])
def test_block_index_production_records(body, expected, flags):
    records, got = classify_source_records(body, "mnk", tensor_names=set(TENSORS) | {"XQ", "XS", "WV", "WS", "EN"}, module=BLOCK_MODULE)
    assert (records, got) == (expected, flags)


def test_block_index_production_provenance_roles():
    from solar.einsum_dsl.statement_class import build_context, provenance, STATIC, TENSOR
    body = ast.parse("acc += XQ[m, k] * XS[m, k // _BLK] * (WV[n, k] * WS[n, k // _BLK])").body
    ctx = build_context(body, ("XQ", "XS", "WV", "WS"), (), "mnk", module=BLOCK_MODULE)
    rhs = body[0].value
    factors = [f for f, _ in __import__("solar.einsum_dsl.statement_class", fromlist=["_product_factors"])._product_factors(rhs)]
    provs = [provenance(f, {"m", "n", "k"}, ctx) for f in factors]
    assert provs == [TENSOR, TENSOR, TENSOR, TENSOR]                 # four value factors: payloads and block scales alike
    from solar.einsum_dsl.statement_class import SLOT_INDEX, UNPROVEN
    assert provenance(factors[1].slice.elts[1], {"m", "n", "k"}, ctx, SLOT_INDEX) == STATIC    # k // _BLK is a static INDEX
    assert provenance(factors[1].slice.elts[1], {"m", "n", "k"}, ctx) == STATIC               # ... and a static value: STATIC arithmetic of a loop index and a literal constant (issue #61)
    assert provenance(ast.parse("_BLK").body[0].value, {"m", "n", "k"}, ctx) == STATIC   # the bare once-bound int literal is STATIC (issue #61 LITERAL-CONSTANT)
    assert provenance(ast.parse("_F").body[0].value, {"m", "n", "k"}, ctx) == STATIC     # a float literal is STATIC in a VALUE slot ...
    assert provenance(ast.parse("_F").body[0].value, {"m", "n", "k"}, ctx, SLOT_INDEX) == UNPROVEN   # ... and never an index or an extent
    assert provenance(ast.parse("_G").body[0].value, {"m", "n", "k"}, ctx) == UNPROVEN   # expression-bound: still outside every production


BLOCK_KERNEL = HEADER + '''
_BLK = 2

@kernel
def nvfp4_linear(XQ: Tensor("M,K", dtype="nvfp4"), XS: Tensor("M,NB", dtype="fp8"),
                 WV: Tensor("N,K", role="weight", dtype="nvfp4"), WS: Tensor("N,NB", role="weight", dtype="fp8")) -> Tensor("M,N"):
    Y = Tensor(M=XQ.shape[0], N=WV.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += XQ[m, k] * XS[m, k // _BLK] * (WV[n, k] * WS[n, k // _BLK])
            Y[m, n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    xq = Tensor.from_numpy(inputs["xq"], "M,K", dtype="nvfp4")
    xs = Tensor.from_numpy(inputs["xs"], "M,NB", dtype="fp8")
    wv = Tensor.from_numpy(inputs["wv"], "N,K", role="weight", dtype="nvfp4")
    ws = Tensor.from_numpy(inputs["ws"], "N,NB", role="weight", dtype="fp8")
    return {"output": nvfp4_linear(xq, xs, wv, ws).data}
'''


def test_block_index_production_pipeline(tmp_path):
    # the production end to end: the block-scale contraction is in the grammar, exact 2 MACs + 2 multiplies per trip
    inp = {"xq": np.full((2, 4), 1.0, np.float32), "xs": np.full((2, 2), 1.0, np.float32),
           "wv": np.full((3, 4), 1.0, np.float32), "ws": np.full((3, 2), 1.0, np.float32)}
    graph, out = _traced_analysis(BLOCK_KERNEL, inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("nvfp4_linear#"))
    layer = _kernel_layer(out, "nvfp4_linear")
    assert "class_evidence_uncertain" not in g and "class_grammar_violations" not in g
    # 24 MACs; other = 2 standalone multiplies per trip (48) + 6 `acc = 0.0` + 6 `Y[m, n] = acc`
    assert (layer["macs"], layer["other_ops"]) == (2 * 3 * 4, 2 * 3 * 4 * 2 + 2 * 3 + 2 * 3) and out["total"].get("status") != "incomplete"


# ---------------------------------------------------------------------------
# 11. round 6 (the review): slot-aware provenance, the GATHER-INDEX production (manager ruling),
#     a scope-complete module collector, every expression slot visited, AnnAssign counted
#     block-context-17, module-scope-18, unchecked-expressions-19,
#     static-role-20, TARGET-KB-ANNASSIGN-COUNT-21
# ---------------------------------------------------------------------------
def _probe6(body, preamble="", alloc='Tensor(M=X.shape[0], dtype="fp32")', kdim='Dim("K")', ret="return Y"):
    code = _probe5(body, preamble)
    return code.replace('Tensor(M=X.shape[0], dtype="fp32")', alloc).replace('Dim("K")', kdim).replace("    return Y\n", "    " + ret + "\n")


# the reviewer's round-5 fixtures (checks-2ee10c5b/test_review2_round5*.py): refuse? -> incomplete through analysis
# Two of the reviewer's expected refusals are complete by the manager's round-6 rulings and are marked so:
#   dynamic_gather_index -> the GATHER-INDEX production (complete, flagged gather); annotated_assign -> AnnAssign is Assign (6 MAC / 6 other)
# Eight more are complete by the issue #61 LITERAL-CONSTANT ruling (a once-bound int literal is STATIC in every STATIC role; a
# literal factor is a scale): block_factor, block_local_alias_factor, block_bare_factor, block_dividend, block_multiplier,
# tensor_size_bare_module, tensor_size_bad_multiplier, dim_unproven_size.  block_module_* and module_if_* keep refusing.
REVIEW2_ROUND5 = [
    ("block_subscript", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 2\n"), 24, False, None),
    ("block_factor", _probe6("Y[m] += X[m,k]*(k//_BLK)", "_BLK = 2\n"), 4, False, (0, 6)),                      # issue #61: STATIC arithmetic of a loop index and a literal constant is a scale (0 MAC / 6 other)
    ("block_local_alias_factor", _probe6("q=k//_BLK\nY[m] += X[m,k]*q", "_BLK = 2\n"), 4, False, (0, 12)),      # issue #61: the local is STATIC (0 MAC / 12 other)
    ("block_module_if_rebind", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 1\nif True:\n    _BLK = 2\n"), 24, True, None),
    ("block_module_for_rebind", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 1\nfor _BLK in range(2,3):\n    pass\n"), 24, True, None),
    ("block_module_chained_rebind", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 1\n_BLK = spare = 2\n"), 24, True, None),
    ("block_module_expression", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 1+1\n"), 24, True, None),
    ("block_module_float", _probe6("Y[m] += X[m,k]*W[m,int(k//_BLK)]", "_BLK = 2.0\n"), 24, True, None),
    ("block_module_twice", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK = 1\n_BLK = 2\n"), 24, True, None),
    ("block_local", _probe6("_BLK = 2\nY[m] += X[m,k]*W[m,k//_BLK]"), 24, False, None),
    ("block_bare_factor", _probe6("Y[m] += X[m,k]*_BLK", "_BLK = 2\n"), 24, False, (0, 6)),                      # issue #61: the literal is a scale (VALUE arithmetic, 0 MAC)
    ("block_dividend", _probe6("Y[m] += X[m,k]*W[m,_BLK//2]", "_BLK = 2\n"), 24, False, None),                   # issue #61: literal index arithmetic
    ("block_multiplier", _probe6("Y[m] += X[m,k]*W[m,k*_BLK]", "_BLK = 1\n"), 24, False, None),                  # issue #61: literal index arithmetic
    ("module_if_def", _probe6("Y[m] += X[m,k]*abs(k)", "if True:\n    def abs(z):\n        return GLOBAL_W[0,z]\n"), 24, True, None),
    ("module_if_assign", _probe6("Y[m] += X[m,k]*abs(k)", "def shadow(z):\n    return GLOBAL_W[0,z]\nif True:\n    abs = shadow\n"), 24, True, None),
    ("tensor_size_bare_module", _probe6("Y[m] += X[m,k]*W[m,k]", "_SIZE = 2\n", alloc='Tensor(M=_SIZE, dtype="fp32")'), 24, False, None),          # issue #61: an int literal is a size
    ("tensor_size_bad_multiplier", _probe6("Y[m] += X[m,k]*W[m,k]", "_SIZE = 1\n", alloc='Tensor(M=X.shape[0]*_SIZE, dtype="fp32")'), 24, False, None),   # issue #61: size arithmetic with an int literal
    ("tensor_size_block", _probe6("Y[m] += X[m,k]*W[m,k]", "_SIZE = 1\n", alloc='Tensor(M=X.shape[0]//_SIZE, dtype="fp32")'), 24, False, None),
    ("dynamic_gather_index", _probe6("Y[m] += X[m,k]*W[m,int(W[m,k])-1]"), 24, False, "gather"),          # manager: GATHER-INDEX production
    ("written_out_valid", _probe6("Y[m] = Y[m] + X[m,k]*W[m,k]"), 24, False, None),
    ("written_out_unproven", _probe6("Y[m] = Y[m] + X[m,k]*W[0,0]"), 24, True, None),
    ("nested_value_test", _probe6("Y[m] += X[m,k]*(W[m,k] if W[m,k] else W[m,k])"), 24, True, None),
    ("root_value_test", _probe6("Y[m] += X[m,k]*W[m,k] if W[m,k] else X[m,k]*W[m,k]"), 24, True, None),
    ("return_unproven_test", _probe6("Y[m] += X[m,k]*W[m,k]", ret="return Y if W[0,0]>0 else Y"), 24, True, None),
    ("dim_unproven_size", _probe6("Y[m] += X[m,k]*W[m,k]", "_COUNT = 3\n", kdim='Dim("K",size=_COUNT)'), 24, False, None),              # issue #61: an int literal is a loop bound
    ("block_import", _probe6("Y[m] += X[m,k]*W[m,k//_BLK]", "from os import SEEK_CUR as _BLK\n"), 24, True, None),
    ("nested_block_index", _probe6("Y[m] += X[m,k]*W[m,(k//_BLK)//_B2]", "_BLK = 1\n_B2 = 2\n"), 24, False, None),
    ("annotated_assign", _probe6("a: float = W[m,k]\nY[m] += X[m,k]*a"), 24, False, (6, 6)),                  # manager: AnnAssign is Assign
    ("written_scalar_valid", _probe6("acc=0.0\nacc=acc+X[m,k]*W[m,k]\nY[m]+=acc"), 24, False, None),
    ("written_unknown_test", _probe6("Y[m] = Y[m] + (X[m,k]*W[m,k] if W[0,0]>0 else X[m,k]*W[m,k])"), 24, True, None),
]


@pytest.mark.parametrize("label,code,expected,refuse,extra", REVIEW2_ROUND5, ids=[c[0] for c in REVIEW2_ROUND5])
def test_review2_round5_structural_pipeline(label, code, expected, refuse, extra, tmp_path):
    g, layer, total = _probe5_run(code, tmp_path, expected)
    assert (total.get("status") == "incomplete") == refuse, (label, layer["macs"], layer["other_ops"], total.get("status"), g.get("class_evidence_uncertain"), g.get("class_grammar_violations"))
    if refuse:
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), label
    else:
        assert "class_evidence_uncertain" not in g, (label, g.get("class_evidence_uncertain"))
    if extra == "gather":
        assert g.get("gather_index_statements") == 6, (label, g.get("gather_index_statements"))      # the row exposes its reliance on the gather rule
    elif extra is not None:
        assert (layer["macs"], layer["other_ops"]) == extra, (label, layer["macs"], layer["other_ops"])
    if extra is None and not refuse:
        assert "gather_index_statements" not in g, label


# the GATHER-INDEX production: the three Lite shapes the review cites stay complete and are flagged gather
LITE58_HISTOGRAM = HEADER + '''
@kernel
def histogram(F: Tensor("N", dtype="int32")) -> Tensor("E"):
    Y = Tensor(E=4, dtype="int32")
    for n in Dim("N"):
        Y[F[n]] += 1
    return Y
def run_model(inputs):
    clear_dims()
    f = Tensor.from_numpy(inputs["f"], "N", dtype="int32")
    return {"output": histogram(f).data}
'''
LITE108_CHUNK = HEADER + '''
@kernel
def chunk_scores(QS: Tensor("N,T,D", dtype="fp32"), KS: Tensor("N,T,D", dtype="fp32"), ROW: Tensor("PR", dtype="int32"), COL: Tensor("PR", dtype="int32")) -> Tensor("N,PR"):
    Y = Tensor(N=QS.shape[0], PR=ROW.shape[0], dtype="fp32")
    for n in Dim("N"):
        for p in Dim("PR"):
            ti = ROW[p]
            tj = COL[p]
            acc = 0.0
            for d in Dim("D"):
                acc += QS[n, ti, d] * KS[n, tj, d]
            Y[n, p] = acc
    return Y
def run_model(inputs):
    clear_dims()
    qs = Tensor.from_numpy(inputs["qs"], "N,T,D", dtype="fp32")
    ks = Tensor.from_numpy(inputs["ks"], "N,T,D", dtype="fp32")
    row = Tensor.from_numpy(inputs["row"], "PR", dtype="int32")
    col = Tensor.from_numpy(inputs["col"], "PR", dtype="int32")
    return {"output": chunk_scores(qs, ks, row, col).data}
'''
LITE119_SLOT = HEADER + '''
@kernel
def slot_in_proj(X: Tensor("N,H", dtype="fp32"), IDX: Tensor("N,K", dtype="int64"), W: Tensor("E,I,H", role="weight", dtype="fp32")) -> Tensor("N,K,I"):
    Y = Tensor(N=X.shape[0], K=IDX.shape[1], I=W.shape[1], dtype="fp32")
    for n in Dim("N"):
        for k in Dim("K"):
            e = IDX[n, k]
            for i in Dim("I"):
                acc = 0.0
                for h in Dim("H"):
                    acc += X[n, h] * W[e, i, h]
                Y[n, k, i] = acc
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "N,H", dtype="fp32")
    idx = Tensor.from_numpy(inputs["idx"], "N,K", dtype="int64")
    w = Tensor.from_numpy(inputs["w"], "E,I,H", role="weight", dtype="fp32")
    return {"output": slot_in_proj(x, idx, w).data}
'''


@pytest.mark.parametrize("label,code,inputs,kernel,expect", [
    ("lite58_histogram_store", LITE58_HISTOGRAM, {"f": np.array([0, 2, 2, 3], np.int32)}, "histogram", {"macs": 0, "other_ops": 4, "gather": 4}),
    ("lite108_chunk_scores_reads", LITE108_CHUNK, {"qs": np.ones((2, 3, 2), np.float32), "ks": np.ones((2, 3, 2), np.float32), "row": np.array([0, 2], np.int32), "col": np.array([1, 0], np.int32)}, "chunk_scores", {"macs": 2 * 2 * 2, "gather": 2 * 2 * 2}),
    ("lite119_slot_in_proj_weight_read", LITE119_SLOT, {"x": np.ones((2, 3), np.float32), "idx": np.array([[0, 1], [1, 0]], np.int64), "w": np.ones((2, 2, 3), np.float32)}, "slot_in_proj", {"macs": 2 * 2 * 2 * 3, "gather": 2 * 2 * 2 * 3}),
])
def test_gather_index_production_keeps_the_lite_shapes_complete_and_flagged(label, code, inputs, kernel, expect, tmp_path):
    graph, out = _traced_analysis(code, inputs, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith(kernel + "#"))
    layer = _kernel_layer(out, kernel)
    assert "class_evidence_uncertain" not in g and "class_grammar_violations" not in g and out["total"].get("status") != "incomplete", (label, g.get("class_evidence_uncertain"), g.get("class_grammar_violations"))
    assert layer["macs"] == expect["macs"], (label, layer["macs"], layer["other_ops"])
    if "other_ops" in expect:
        assert layer["other_ops"] == expect["other_ops"], (label, layer["other_ops"])
    assert g["gather_index_statements"] == expect["gather"], (label, g.get("gather_index_statements"))


@pytest.mark.parametrize("label,code,expected", [
    ("value_as_tensor_size", _probe6("Y[m] += X[m,k]*W[m,k]", alloc='Tensor(M=int(W[0,0]), dtype="fp32")'), 24),
    ("value_as_dim_size", _probe6("Y[m] += X[m,k]*W[m,k]", kdim='Dim("K", size=int(W[0,0])+1)'), 24),
    ("value_as_ifexp_test", _probe6("Y[m] += X[m,k]*W[m,k] if W[m,k] else X[m,k]*W[m,k]"), 24),
    ("value_as_shape_index", _probe6("Y[m] += X[m,k]*W.shape[int(W[m,k])-1]"), 36),
    ("value_index_on_non_tensor", _probe6("cfg = 2.0\nY[m] += X[m,k]*W[m,k]*cfg[int(W[m,k])-1]"), 24),
])
def test_gather_index_production_negatives_refuse(label, code, expected, tmp_path):
    from solar.einsum_lib.verify import execute_translated
    inp = {"x": np.full((2, 3), 4, np.float32), "w": np.full((2, 3), 2, np.float32)}
    try:
        execute_translated(code, {k: v.shape for k, v in inp.items()}, all_inputs=inp, trace=True, fast=False)
    except Exception:
        pytest.skip(f"{label}: the DSL cannot execute this form; the records-level table covers it")
    graph, out = _traced_analysis(code, inp, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("probe#"))
    assert out["total"]["status"] == "incomplete" and "class_evidence_unproven" in g["class_evidence_uncertain"], (label, g)


# records-level: slots and the gather production
BLOCK6_MODULE = ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_BLK = 2\n")


@pytest.mark.parametrize("body,expected,flags,gather", [
    ("Y[m] += X[m, k] * W[m, k // _BLK]", [(MAC, 0)], set(), 0),                       # index slot: the production
    ("Y[m] += X[m, k] * (k // _BLK)", [(OTHER, 0)], set(), 0),                         # value slot: STATIC arithmetic of a loop index and a literal constant -> a scale (issue #61)
    ("q = k // _BLK\nY[m] += X[m, k] * q", [(OTHER, 0), (OTHER, 0)], set(), 0),          # a Name bound from it: STATIC likewise (issue #61)
    ("Y[m] += X[m, k] * W[m, int(W[m, k]) - 1]", [(MAC, 0)], set(), 1),                # GATHER-INDEX: a proven VALUE index
    ("e = IDX[m, k]\nY[m] += X[m, k] * W[e, k]", [(OTHER, 0), (MAC, 0)], set(), 1),    # a local bound to a tensor element
    ("Y[W[m, k]] += X[m, k]", [(OTHER, 0)], set(), 1),                                  # a gather STORE
    ("Y[m] += X[m, k] * W[m, cfg[k]]", [(MAC, 0)], UNPROVEN_FLAG, 0),                   # an UNPROVEN index stays refused
    ("Y[m] += X[m, k] * W[m, k] if W[m, k] else X[m, k] * W[m, k]", [(MAC, 0)], UNPROVEN_FLAG, 0),      # a VALUE test (root)
    ("Y[m] += X[m, k] * (W[m, k] if W[m, k] else W[m, k])", [(MAC, 0)], UNPROVEN_FLAG, 0),              # a VALUE test (nested)
    ("Y[m] += X[m, k] * W.shape[int(W[m, k]) - 1]", [(MAC, 0)], UNPROVEN_FLAG, 0),                       # a VALUE .shape index: unproven, counted and flagged
    ("a: float = W[m, k]\nY[m] += X[m, k] * a", [(OTHER, 0), (MAC, 0)], set(), 0),                      # AnnAssign is an Assign
])
def test_round6_slots_and_gather_records(body, expected, flags, gather):
    from solar.einsum_dsl.statement_class import build_context, classify_records
    mod = ast.parse(body)
    ctx = build_context(mod.body, set(TENSORS) | {"IDX"}, (), "mk", module=BLOCK6_MODULE)
    records, got = [], set()
    for s in mod.body:
        r, f = classify_records(s, "mk", ctx); records += r; got |= f
    assert (records, got) == (expected, flags), (records, got)
    assert len(ctx.gather_statements) == gather, ctx.gather_statements


def test_round6_module_collector_is_scope_complete():
    from solar.einsum_dsl.statement_class import module_bound_names, module_literal_names, module_bindings
    mod = ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_A = 2\n_B = 1\nif True:\n    _B = 2\n_C = 1\nfor _C in range(2, 3):\n    pass\n"
                    "_D = spare = 2\n_E = 1 + 1\n_F = 2.0\n_G = 2\ndel _G\nif True:\n    def abs(z):\n        return z\n    exp = 3\ntry:\n    import os as _H\nexcept Exception:\n    pass\n_I = 2\n")
    bound = module_bound_names(mod)
    assert {"_A", "_B", "_C", "_D", "spare", "_E", "_F", "_G", "abs", "exp", "_H", "_I"} <= bound
    assert "Tensor" not in bound and "Dim" not in bound                      # the exact DSL import is the vocabulary
    assert module_literal_names(mod, ints_only=True) == {"_A", "_I"}         # bound once, to an int literal, never rebound / deleted / conditional
    kinds = module_bindings(mod)
    assert kinds["_B"]["count"] == 2 and kinds["_G"]["deleted"] and kinds["abs"]["kinds"] == {"def"}
    other = ast.parse("from unrelated.einsum_lib import Tensor, Dim\nfrom solar.einsum_lib import exp\n")
    assert {"Tensor", "Dim"} <= module_bound_names(other)                    # a suffix match is not the DSL vocabulary
    assert "exp" not in module_bound_names(other)                            # the DSL's own exp is its vocabulary


def test_round6_every_expression_slot_gets_a_provenance_visit():
    # the spy is installed BEFORE the context is built; one kernel exercises every slot
    import solar.einsum_dsl.statement_class as mod
    from solar.einsum_dsl import ast_parser as ap
    src = HEADER + '''
_BLK = 2

@kernel
def probe(X: Tensor("M,K", dtype="fp32"), W: Tensor("M,K", dtype="fp32"), s: float) -> Tensor("M"):
    Y = Tensor(M=X.shape[0], dtype="fp32")
    n = X.shape[1]
    for m in Dim("M"):
        for k in Dim("K", size=n):
            if k > 0:
                Y[m] += X[m, k] * W[m, k // _BLK] * abs(s)
            else:
                Y[m] += X[m, k] * (W[m, k] if m > 0 else W[m, k]) * abs(s)
    return Y if n > 0 else Y
'''
    tree = ast.parse(src); fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    seen = []
    real = mod.provenance
    def spy(node, lv, ctx, *a, **kw):
        seen.append(ast.unparse(node)); return real(node, lv, ctx, *a, **kw)
    mod.provenance = spy
    try:
        ctx = ap._statement_context(fn, tree)
        flags = set()
        ap._loop_paths(fn.body, ap._local_size_consts(fn, ap._module_int_consts(tree), ["X", "W", "s"]), (), [], [], [], frozenset(), flags, ctx)
    finally:
        mod.provenance = real
    assert not ctx.violations and not flags, (ctx.violations, flags)
    for must in ("X.shape[0]", "X.shape[1]", "n", "k > 0", "m > 0", "n > 0", "X[m, k]", "W[m, k // _BLK]", "k // _BLK", "abs(s)", "s", "W[m, k]", "Y if n > 0 else Y"):
        assert must in seen, (must, sorted(set(seen))[:40])


@pytest.mark.parametrize("label,code,expected,refuse", [
    ("int_param_tensor_size", _probe5("Y[m] += X[m,k]*W[m,k]", scalar=False).replace('W: Tensor("M,K", dtype="fp32")) -> Tensor("M")', 'W: Tensor("M,K", dtype="fp32"), rows: int) -> Tensor("M")').replace("probe(x,w)", "probe(x,w,2)").replace('Tensor(M=X.shape[0], dtype="fp32")', 'Tensor(M=rows, dtype="fp32")'), 24, False),
    ("size_local_from_shape_and_int_param", _probe5("Y[m] += X[m,k]*W[m,k]").replace('W: Tensor("M,K", dtype="fp32")) -> Tensor("M")', 'W: Tensor("M,K", dtype="fp32"), half: int) -> Tensor("M")').replace("probe(x,w)", "probe(x,w,1)").replace('    Y = Tensor(M=X.shape[0], dtype="fp32")', '    rows = X.shape[0] // half\n    Y = Tensor(M=rows, dtype="fp32")'), 24, False),
    ("float_param_tensor_size", _probe5("Y[m] += X[m,k]*W[m,k]", scalar=True).replace('Tensor(M=X.shape[0], dtype="fp32")', 'Tensor(M=int(s), dtype="fp32")'), 24, True),
    ("int_param_dim_size", _probe5("Y[m] += X[m,k]*W[m,k]").replace('W: Tensor("M,K", dtype="fp32")) -> Tensor("M")', 'W: Tensor("M,K", dtype="fp32"), cols: int) -> Tensor("M")').replace("probe(x,w)", "probe(x,w,3)").replace('Dim("K")', 'Dim("K", size=cols)'), 24, False),
])
def test_round6_size_slot_accepts_the_dsl_size_inputs(label, code, expected, refuse, tmp_path):
    # an `int` parameter and a local bound to STATIC size arithmetic are sizes (the DSL's extent inputs); a float parameter is not
    g, layer, total = _probe5_run(code, tmp_path, expected)
    assert (total.get("status") == "incomplete") == refuse, (label, layer["macs"], layer["other_ops"], total.get("status"), g.get("class_grammar_violations"), g.get("class_evidence_uncertain"))


# ---------------------------------------------------------------------------
# 12. an earlier fix round 7 (the reviewer's round-6 verdict on an earlier revision): the SIZE production's binding conditions, a fixpoint key
#     over the WHOLE abstract state, every statement validated by slot (inside or outside loops, used or unused), the
#     module collector's definition-time expressions, AnnAssign normalised to Assign, a tensor OBJECT as the only
#     subscript receiver, per-statement gather receipts including control tests, refusal markers surviving the
#     recompute billing convention, and a generic visitation proof over a corpus
# ---------------------------------------------------------------------------

def _probe7_sized(body="Y[m] += X[m,k]*W[m,k]", params="rows: int, scale: float", args="2, 2.0", prefix="", suffix="",
                  size="rows", preamble=""):
    """the reviewer's round-6 ``sized`` probe: extra kernel parameters, a prefix before / suffix after the loops, a Tensor size."""
    code = _probe5(body, preamble)
    code = (code.replace('W: Tensor("M,K", dtype="fp32")) ->', 'W: Tensor("M,K", dtype="fp32"), ' + params + ") ->")
            .replace("probe(x,w)", f"probe(x,w, {args})"))
    code = code.replace('    Y = Tensor(M=X.shape[0], dtype="fp32")',
                        "".join("    " + x + "\n" for x in prefix.splitlines()) + f'    Y = Tensor(M={size}, dtype="fp32")')
    if suffix:
        code = code.replace("    return Y", "".join("    " + x + "\n" for x in suffix.splitlines()) + "    return Y")
    return code


def _probe7_accum(annotated):
    code = _probe5(("acc: float = " if annotated else "acc = ") + "acc + X[m,k]*W[m,k]", after_loop="Y[m] = acc")
    return code.replace('    for m in Dim("M"):\n', '    for m in Dim("M"):\n        acc = W[m,0]\n')


def _probe7_carry(n):
    """the reviewer's loop-carried SIZE chain: ``a0 .. a{n-1}`` start as the int extent; each trip shifts a float one name closer."""
    return HEADER + '''
@kernel
def probe(X: Tensor("M,K",dtype="fp32"), W: Tensor("M,K",dtype="fp32"), rows: int, scale: float) -> Tensor("M"):
    Y=Tensor(M=X.shape[0],dtype="fp32")
''' + "".join(f"    a{i} = rows\n" for i in range(n)) + '''    for r in Dim("R",size=16):
        for m in Dim("M"):
            for k in Dim("K"):
                Z=Tensor(S=int(a0),dtype="fp32")
''' + "".join(f"                a{i}=a{i+1}\n" for i in range(n - 1)) + f"                a{n-1}=scale\n" + '''                Y[m]+=X[m,k]*W[m,k]
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"M,K",dtype="fp32")
    w=Tensor.from_numpy(inputs["w"],"M,K",dtype="fp32")
    return {"output":probe(x,w,2,2.0).data}
'''


def _probe7_recompute(repeated):
    code = _probe5("Y[m] += X[m,k]*W[m,k]*_BAD", "_BAD = 1 + 1\n")   # issue #61: `_BAD = 2` is a LITERAL CONSTANT (STATIC) now; the expression-bound form keeps this fixture's unproven factor
    if repeated:
        start = code.index('    for m in Dim("M"):\n')
        end = code.index("    return Y", start)
        code = code[:start] + '    for r in Dim("R",size=16):\n' + "".join("    " + x + "\n" for x in code[start:end].splitlines()) + code[end:]
    return code


# label, code, numerical output, refuse?, exact (macs, other_ops, gather_index_statements) when complete
ROUND7_PIPELINE = [
    # SIZE-BINDING-23: "never rebound" is a whole-body condition; the annotation must resolve to the builtin `int`
    ("size_param_rebound_after", _probe7_sized(suffix="rows = rows + 1"), 24, True, None),
    ("size_param_augmented_after", _probe7_sized(suffix="rows += 1"), 24, True, None),
    ("size_param_rebound_before", _probe7_sized(prefix="rows = rows"), 24, True, None),
    ("size_param_walrus_after", _probe7_sized(suffix="(rows := rows)"), 24, True, None),
    ("size_annotation_shadow", _probe7_sized(preamble="int = float\n"), 24, True, None),
    ("size_param_never_rebound", _probe7_sized(), 24, False, (6, 0, 0)),
    ("size_keyword_only_never_rebound", _probe7_sized(params="*, rows: int, scale: float", args="rows=2, scale=2.0"), 24, False, (6, 0, 0)),
    # SIZE-FIXPOINT-24: a float circulated into a Tensor size through a carry chain refuses, whatever the chain length
    ("size_carry_4", _probe7_carry(4), 384, True, None),
    ("size_carry_12", _probe7_carry(12), 384, True, None),
    ("size_carry_32", _probe7_carry(32), 384, True, None),
    # OUTER-SLOTS-25: an outer statement-if TEST and an outer UNUSED binding go through the same slot validation
    ("size_conditional_unproven", _probe7_sized(prefix="if _FLAG:\n    extent = rows\nelse:\n    extent = rows", size="extent", preamble="_FLAG = True\n"), 24, True, None),
    ("size_conditional_proven", _probe7_sized(prefix="if scale > 0:\n    extent = rows\nelse:\n    extent = rows", size="extent"), 24, False, (6, 0, 0)),
    ("unused_unproven_assignment", _probe7_sized(prefix="unused = _BAD", preamble="_BAD = 1 + 1\n"), 24, True, None),            # issue #61: expression-bound (`_BAD = 2` would be a literal constant)
    ("unused_unproven_assignment_after", _probe7_sized(suffix="unused = _BAD", preamble="_BAD = 1 + 1\n"), 24, True, None),      # issue #61: expression-bound
    ("outer_unproven_store_index", _probe7_sized(suffix="Y[_BAD - 2] = Y[0]", preamble="_BAD = 1 + 1\n"), [24, 24], True, None),   # issue #61: expression-bound
    # MODULE-DEFAULTS-26: definition-time expressions bind in the enclosing (module) scope
    ("module_default_shadow", _probe5("Y[m] += X[m,k]*abs(k)", "def shadow(z):\n    return GLOBAL_W[0,z]\ndef unused(arg=(abs := shadow)):\n    pass\n"), 24, True, None),
    ("module_default_block_rebind", _probe5("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK=1\ndef unused(arg=(_BLK := 2)):\n    pass\n"), 24, True, None),
    ("module_decorator_shadow", _probe5("Y[m] += X[m,k]*abs(k)", "def shadow(z):\n    return GLOBAL_W[0,z]\ndef deco(f):\n    return f\n@deco if (abs := shadow) else deco\ndef unused():\n    pass\n"), 24, True, None),
    ("module_kwdefault_block_rebind", _probe5("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK=1\ndef unused(*, arg=(_BLK := 2)):\n    pass\n"), 24, True, None),
    ("module_return_annotation_block_rebind", _probe5("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK=1\ndef unused() -> (_BLK := 2):\n    pass\n"), 24, True, None),
    ("module_class_base_block_rebind", _probe5("Y[m] += X[m,k]*W[m,k//_BLK]", "_BLK=1\nclass Unused(object if (_BLK := 2) else object):\n    pass\n"), 24, True, None),
    # ANNASSIGN-CLASS-27: a valued AnnAssign IS an Assign (accumulator detection, binding, flow, classification)
    ("annotated_accumulation", _probe7_accum(True), 26, False, (6, 4, 0)),
    ("plain_accumulation", _probe7_accum(False), 26, False, (6, 4, 0)),
    # GATHER-RECEIPT-29: gather receipts are per statement; a control test addressing through a gather is one evaluation per trip
    ("gather_conditional_two_stmts", _probe5("if k >= 0:\n    Y[m] += X[m,k]*W[m,int(W[m,k])-1]\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,k]\n    Y[m] += X[m,k]*W[m,k]"), 48, False, (12, 0, 6)),
    ("gather_conditional_else_only", _probe5("if k < 0:\n    Y[m] += X[m,k]*W[m,k]\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,int(W[m,k])-1]\n    Y[m] += X[m,k]*W[m,k]"), 48, False, (12, 0, 6)),
    ("gather_if_condition_only", _probe5("if W[m,int(W[m,k])-1] > 0:\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,k]"), 24, False, (6, 0, 6)),
    ("gather_if_condition_and_body", _probe5("if W[m,int(W[m,k])-1] > 0:\n    Y[m] += X[m,k]*W[m,int(W[m,k])-1]\nelse:\n    Y[m] += X[m,k]*W[m,k]"), 24, False, (6, 0, 12)),
    ("gather_outer_if_condition", _probe5("Y[m] += X[m,k]*W[m,k]", after_loop="if W[m,int(W[m,0])-1] > 0:\n    Y[m] = Y[m] + 0.0\nelse:\n    Y[m] = Y[m] + 0.0"), 24, False, (6, 2, 2)),
    # RECOMPUTE-REFUSAL-30: the recompute billing convention never clears a provenance / grammar refusal
    ("recompute_single", _probe7_recompute(False), 48, True, None),
    ("recompute_repeated", _probe7_recompute(True), 768, True, None),
]


@pytest.mark.parametrize("label,code,expected,refuse,exact", ROUND7_PIPELINE, ids=[c[0] for c in ROUND7_PIPELINE])
def test_round7_pipeline(label, code, expected, refuse, exact, tmp_path):
    g, layer, total = _probe5_run(code, tmp_path, expected)
    perf = _perf(tmp_path, tmp_path / "analysis")
    assert (total.get("status") == "incomplete") == refuse, (label, layer["macs"], layer["other_ops"], total.get("status"), g.get("class_grammar_violations"), g.get("class_evidence_uncertain"))
    assert (perf["workload"].get("status") == "incomplete") == refuse, (label, perf["workload"].get("status"))
    if refuse:
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), (label, g.get("class_evidence_uncertain"), g.get("class_grammar_violations"))
    else:
        assert (layer["macs"], layer["other_ops"], layer.get("gather_index_statements", 0)) == exact, (label, layer["macs"], layer["other_ops"], layer.get("gather_index_statements"))
        assert g.get("gather_index_statements", 0) == exact[2], (label, g.get("gather_index_statements"))


def test_round7_size_parameter_conditions_are_whole_body_and_resolved():
    from solar.einsum_dsl.statement_class import build_context, STATIC, SLOT_SIZE, provenance, body_bound_names
    body = ast.parse("Y = Tensor(M=rows, dtype='fp32')\nfor m in Dim('M'):\n    Y[m] = 1.0\nrows = rows + 1\n").body
    assert body_bound_names(body) >= {"Y", "m", "rows"}
    ctx = build_context(body, ["Y0"], ["rows"], int_params=["rows"])
    assert ctx.int_params == set()                                           # rebound AFTER the use: still not a size source
    ctx.select(body[0])
    assert provenance(body[0].value.keywords[0].value, set(), ctx, SLOT_SIZE) != STATIC
    clean = ast.parse("Y = Tensor(M=rows, dtype='fp32')\n").body
    ctx = build_context(clean, [], ["rows"], int_params=["rows"])
    assert ctx.int_params == {"rows"}
    shadow = ast.parse("int = float\n")
    ctx = build_context(clean, [], ["rows"], int_params=["rows"], module=shadow)
    assert ctx.int_params == set()                                           # the annotation `int` no longer resolves to the builtin
    for form in ("rows += 1", "(rows := 2)", "for rows in Dim('R'):\n    pass", "del rows", "with x as rows:\n    pass", "def rows():\n    pass", "import rows"):
        assert "rows" in body_bound_names(ast.parse(form).body), form


def test_round7_fixpoint_key_covers_every_state_component():
    from solar.einsum_dsl.statement_class import freeze_state, build_context
    a = ({"x": "scalar"}, {"m"}, {"x": "static"})
    b = ({"x": "scalar"}, {"m"}, {"x": "scalar"})
    assert freeze_state(a) == freeze_state(({"x": "scalar"}, {"m"}, {"x": "static"})) and freeze_state(a) != freeze_state(b)
    # components added later (lists, nested dicts, tuples) are frozen by VALUE, never by identity
    c1 = ({"x": "scalar"}, set(), {}, {"q": ["W", "X"]}, {"q": ("W",)})
    c2 = ({"x": "scalar"}, set(), {}, {"q": ["W", "X"]}, {"q": ("W",)})
    assert freeze_state(c1) == freeze_state(c2) and freeze_state(c1) != freeze_state(({"x": "scalar"}, set(), {}, {"q": ["W"]}, {"q": ("W",)}))
    fn = next(n for n in ast.parse(_probe7_carry(12)).body if isinstance(n, ast.FunctionDef) and n.name == "probe")
    ctx = build_context(fn.body, ["X", "W"], ["rows", "scale"], int_params=["rows"])
    assert ctx.converged                                                       # the chain converges (bound = bindings + 2 >= chain length)
    z = next(s for s in ast.walk(fn) if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "Z")
    ctx.select(z)
    assert ctx.size_provenance("a0") != "static"                               # the float reached a0's SIZE view


def test_round7_every_statement_is_validated_by_slot():
    from solar.einsum_dsl.statement_class import build_context
    mod = ast.parse("_FLAG = True\n_BAD = 1 + 1\n")      # issue #61: `_BAD = 2` is a LITERAL CONSTANT now; `_FLAG = True` (a bool) never was
    outer_if = ast.parse("if _FLAG:\n    extent = rows\nelse:\n    extent = rows\nY = Tensor(M=extent, dtype='fp32')\n").body
    ctx = build_context(outer_if, ["X"], ["rows"], module=mod, int_params=["rows"])
    assert any("if test" in v for v in ctx.violations), ctx.violations
    unused = ast.parse("unused = _BAD\nY = Tensor(M=rows, dtype='fp32')\n").body
    ctx = build_context(unused, ["X"], ["rows"], module=mod, int_params=["rows"])
    assert any("assigned value not proven" in v for v in ctx.violations), ctx.violations
    store = ast.parse("Y[_BAD - 2] = 1.0\n").body
    ctx = build_context(store, ["Y"], [], module=mod)
    assert any("store index not proven" in v for v in ctx.violations), ctx.violations
    carried = ast.parse("h = 0.0\nfor t in Dim('T'):\n    x = h\n    h = x + 1.0\n").body   # a loop-carried name is proven by the fixpoint, not refused by pass 1
    ctx = build_context(carried, [], [], module=mod)
    assert ctx.violations == [], ctx.violations


def test_round7_module_collector_sees_definition_time_expressions():
    from solar.einsum_dsl.statement_class import module_bindings, module_literal_names
    mod = ast.parse("_BLK = 1\ndef shadow(z):\n    return z\ndef unused(arg=(abs := shadow), *, kw=(exp := shadow)) -> (ret := 1):\n    local = (walrus_in_body := 1)\n"
                    "@(deco := shadow)\ndef decorated(a: (ann := 1) = (_BLK := 2)):\n    pass\nclass K(object if (base := 1) else object, metaclass=(meta := type)):\n    pass\n")
    info = module_bindings(mod)
    for name in ("abs", "exp", "ret", "deco", "ann", "base", "meta"):
        assert name in info and "walrus" in info[name]["kinds"], (name, info.get(name))
    assert "walrus_in_body" not in info                                        # the function BODY is the nested scope
    assert info["_BLK"]["count"] == 2 and "_BLK" not in module_literal_names(mod, ints_only=True)


def test_round7_annotated_assignment_is_classified_as_an_assignment():
    from solar.einsum_dsl.statement_class import classify_source_records
    plain = "acc = W[m,0]\nacc = acc + X[m,k]*W[m,k]\nY[m] = acc"
    annotated = "acc = W[m,0]\nacc: float = acc + X[m,k]*W[m,k]\nY[m] = acc"
    assert classify_source_records(plain, "mk", tensor_names=["X", "W", "Y"]) == classify_source_records(annotated, "mk", tensor_names=["X", "W", "Y"])
    assert classify_source_records(annotated, "mk", tensor_names=["X", "W", "Y"]) == ([(OTHER, 0), (MAC, 0), (OTHER, 0)], set())


def test_round7_only_a_tensor_object_is_a_subscript_receiver():
    from solar.einsum_dsl.statement_class import classify_source_records, UNPROVEN_FLAG
    records, flags = classify_source_records("a = X[m,k]\nY[m] += X[m,k]*a[int(W[m,k])-1]", "mk", tensor_names=["X", "W", "Y"])
    assert UNPROVEN_FLAG in flags, (records, flags)                             # a tensor ELEMENT is a numeric value, not a receiver
    records, flags = classify_source_records("a = X[m,k]\na[k] = 1.0", "mk", tensor_names=["X", "Y"])
    assert UNPROVEN_FLAG in flags, (records, flags)                             # nor a store receiver
    records, flags = classify_source_records("A = Tensor(M=X.shape[0], dtype='fp32')\nY[m] += X[m,k]*A[int(W[m,k])-1]", "mk", tensor_names=["X", "W", "Y"])
    assert flags == set(), (records, flags)                                     # an allocation is a tensor object: a declared gather


def test_round7_gather_receipts_are_per_record_and_include_control_tests():
    from solar.einsum_dsl import ast_parser as ap
    from solar.einsum_dsl.statement_class import build_context
    src = ("if k >= 0:\n    Y[m] += X[m,k]*W[m,int(W[m,k])-1]\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,k]\n    Y[m] += X[m,k]*W[m,k]\n"
           "if W[m,int(W[m,k])-1] > 0:\n    Y[m] += X[m,k]*W[m,k]\nelse:\n    Y[m] += X[m,k]*W[m,k]\n")
    body = ast.parse(src).body
    ctx = build_context(body, ["X", "W", "Y"], (), "mk")
    flags = set()
    out, controls, scontrols = ap._classify_assignments(body, frozenset("mk"), flags, ctx)   # (records, gather controls, singleton controls) -- issue #61
    assert scontrols == 0
    assert [r[2] for r in out] == [True, False, False], out                     # one gather-bearing record of the first if, none of the second (records carry the singleton flag and the dtype operands too on this stack)
    assert [r[3] for r in out] == [False, False, False], out
    assert controls == 1 and flags == set(), (controls, flags)                  # the second if's TEST is one gather evaluation per trip


def test_round7_recompute_keeps_the_refusal_markers(tmp_path):
    g, layer, total = _probe5_run(_probe7_recompute(True), tmp_path, 768)
    assert g.get("recompute_fingerprint") is True and "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), g
    assert total.get("status") == "incomplete" and "probe" in str(total.get("incomplete_layers")), total


def _kernel_slot_nodes(fn):
    """Every expression node of a kernel body that is a classified SLOT: a Load-context expression, except the
    callee of a Call (and the receiver of a method callee), the receiver Name of a Subscript / Attribute and the
    Tuple shell of a multi-index slice (its elements are the slots)."""
    skip, nodes = set(), []
    for s in fn.body:
        for n in ast.walk(s):
            if isinstance(n, ast.Call):
                skip.add(id(n.func))
                if isinstance(n.func, ast.Attribute):
                    skip.add(id(n.func.value))
            elif isinstance(n, ast.Subscript):
                if isinstance(n.value, ast.Name):
                    skip.add(id(n.value))
                if isinstance(n.slice, ast.Tuple):
                    skip.add(id(n.slice))
            elif isinstance(n, ast.Attribute):
                skip.add(id(n.value))
            if isinstance(n, ast.expr) and not isinstance(getattr(n, "ctx", None), (ast.Store, ast.Del)):
                nodes.append(n)
    return [n for n in nodes if id(n) not in skip]


def _visitation_gaps(tree):
    """``{kernel: (refused, [unvisited slot nodes])}`` for every @kernel of *tree*: the provenance spy collects the
    node ids it visited (with a slot) while the parser builds the context and walks the loop paths."""
    import solar.einsum_dsl.statement_class as mod
    from solar.einsum_dsl import ast_parser as ap
    out = {}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and ap._is_kernel_decorated(n)]:
        seen = {}
        real = mod.provenance

        def spy(node, lv, ctx, slot=mod.SLOT_VALUE, *a, **kw):
            seen[id(node)] = slot
            return real(node, lv, ctx, slot, *a, **kw)
        mod.provenance = spy
        try:
            ctx = ap._statement_context(fn, tree)
            flags = set()
            consts = ap._local_size_consts(fn, ap._module_int_consts(tree), [a.arg for a in fn.args.posonlyargs + fn.args.args])
            ap._loop_paths(fn.body, consts, (), [], [], [], frozenset(), flags, ctx)
        finally:
            mod.provenance = real
        refused = bool(ctx.violations) or "class_evidence_unproven" in flags
        out[fn.name] = (refused, [ast.unparse(n) + f" (line {getattr(n, 'lineno', '?')})" for n in _kernel_slot_nodes(fn) if id(n) not in seen])
    return out


def test_round7_every_slot_of_the_author_fixtures_is_visited():
    programs = [code for _, code, _, refuse, _ in ROUND7_PIPELINE if not refuse] + [code for _, code, _, refuse, _ in REVIEW2_ROUND5 if not refuse]
    programs += [code for _, code, _, macs in REVIEW2_ROUND4 if macs is not None] + [LITE58_HISTOGRAM, LITE108_CHUNK, LITE119_SLOT]
    checked = 0
    for code in programs:
        for name, (refused, gaps) in _visitation_gaps(ast.parse(code)).items():
            assert not refused, (name, code)
            assert not gaps, (name, gaps[:5], code)
            checked += 1
    assert checked >= 20, checked


# ---------------------------------------------------------------------------
# 13. an earlier fix round 7 addition (manager, implementation gap of the GATHER-INDEX ruling found by the Lite backlog, PLAN.md 2b):
#     a tensor Subscript whose indices are each STATIC, a loop index or a proven VALUE is admitted whether or not a loop
#     variable is mentioned (a pure gather through VALUE-derived locals); a literal-only index set stays UNPROVEN
# ---------------------------------------------------------------------------

LITE170_QKV_PART_PROJ = HEADER + '''
@kernel
def qkv_part_proj(X: Tensor("B,QH,QW,C", dtype="fp32"), Wq: Tensor("F3,C", role="weight", dtype="fp32"), bias: Tensor("F3", role="weight", dtype="fp32"), off: int, hd: int) -> Tensor("B,N,QH,QW,D"):
    Y = Tensor(B=X.shape[0], N=2, QH=X.shape[1], QW=X.shape[2], D=hd, dtype="fp32")
    for b in Dim("B"):
        for n in Dim("N"):
            for qh in Dim("QH"):
                for qw in Dim("QW"):
                    for d in Dim("D"):
                        row = off + n * hd + d
                        acc = bias[row]
                        for c in Dim("C"):
                            acc += X[b, qh, qw, c] * Wq[row, c]
                        Y[b, n, qh, qw, d] = acc
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,QH,QW,C", dtype="fp32")
    wq = Tensor.from_numpy(inputs["wq"], "F3,C", role="weight", dtype="fp32")
    bias = Tensor.from_numpy(inputs["bias"], "F3", role="weight", dtype="fp32")
    return {"output": qkv_part_proj(x, wq, bias, 0, 2).data}
'''
LITE229_SLOT_ROUTING_WEIGHT = HEADER + '''
@kernel
def slot_routing_weight(SLOT: Tensor("N,C", dtype="int32"), W: Tensor("T,EG", dtype="fp32")) -> Tensor("N"):
    Y = Tensor(N=SLOT.shape[0], dtype="fp32")
    for n in Dim("N"):
        t = SLOT[n, 0]
        ge = SLOT[n, 1]
        Y[n] = W[t, ge] * SLOT[n, 3]
    return Y
def run_model(inputs):
    clear_dims()
    slot = Tensor.from_numpy(inputs["slot"], "N,C", dtype="int32")
    w = Tensor.from_numpy(inputs["w"], "T,EG", dtype="fp32")
    return {"output": slot_routing_weight(slot, w).data}
'''


@pytest.mark.parametrize("label,code,inputs,kernel,expected,expect", [
    # the 170 / 171 qkv_part_proj statement shape (171 nests two more window loops around the same three statements): bias[row] and Wq[row, c] read through a VALUE-derived local
    ("lite170_171_qkv_part_proj", LITE170_QKV_PART_PROJ, {"x": np.ones((1, 1, 2, 2), np.float32), "wq": np.ones((4, 2), np.float32), "bias": np.ones(4, np.float32)}, "qkv_part_proj", 3.0, {"macs": 16, "other_ops": 24, "gather": 8 + 16}),
    # 229 slot_routing_weight: W[t, ge] with both indices tensor-element VALUES and no loop variable in the index set (the Lite
    # program's N1 = N + 1 sentinel row is dropped here: `Tensor(N=SLOT.shape[0] - 1)` is a loop-extent question, not a class one)
    ("lite229_slot_routing_weight", LITE229_SLOT_ROUTING_WEIGHT, {"slot": np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]], np.int32), "w": np.full((2, 2), 2, np.float32)}, "slot_routing_weight", 2.0, {"macs": 0, "other_ops": 9, "gather": 3}),
])
def test_round7_pure_gather_through_value_locals_is_admitted(label, code, inputs, kernel, expected, expect, tmp_path):
    from solar.einsum_lib.verify import execute_translated
    outputs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    assert np.all(outputs["output"] == expected), outputs["output"]
    graph, out = _traced_analysis(code, inputs, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith(kernel + "#"))
    layer = _kernel_layer(out, kernel)
    assert "class_evidence_uncertain" not in g and "class_grammar_violations" not in g and out["total"].get("status") != "incomplete", (label, g.get("class_evidence_uncertain"), g.get("class_grammar_violations"))
    assert (layer["macs"], layer["other_ops"], g.get("gather_index_statements")) == (expect["macs"], expect["other_ops"], expect["gather"]), (label, layer["macs"], layer["other_ops"], g.get("gather_index_statements"))


def test_round7_pure_gather_records_and_the_literal_only_rule():
    from solar.einsum_dsl.statement_class import build_context, classify_records, UNPROVEN_FLAG
    body = ast.parse("t = SLOT[n, 0]\nge = SLOT[n, 1]\nY[n] = W[t, ge] * SLOT[n, 3]\n").body
    ctx = build_context(body, ["SLOT", "W", "Y"], (), "n")
    out = [classify_records(s, "n", ctx) for s in body]
    assert out == [([(OTHER, 0)], set())] * 3 and ctx.violations == [], (out, ctx.violations)
    assert ctx.gather_statements == {id(body[2])}                                  # W[t, ge]: a gather with no loop variable in its index set
    body = ast.parse("row = off + n * hd + d\nacc = bias[row]\nacc += X[b, c] * Wq[row, c]\n").body
    ctx = build_context(body, ["X", "Wq", "bias"], ["off", "hd"], "nbdc")
    flags = set()
    for s in body:
        flags |= classify_records(s, "nbdc", ctx)[1]
    assert flags == set() and ctx.violations == [] and ctx.gather_statements == {id(body[1]), id(body[2])}, (flags, ctx.violations)
    for text in ("Y[m] += X[m, k] * W[0, 0]", "s = GM[0]\nY[m] += X[m, k] * s", "Y[m] += X[m, k] / W[W.shape[0] - 1, 0]"):
        body = ast.parse(text).body
        ctx = build_context(body, ["X", "W", "Y", "GM"], (), "mk")
        got = set()
        for s in body:
            got |= classify_records(s, "mk", ctx)[1]
        assert UNPROVEN_FLAG in got, text                                           # a literal-only / STATIC-only index set without a loop variable stays unproven


# ---------------------------------------------------------------------------
# 14. an earlier fix round 8 (the reviewer's round-7 verdict on an earlier revision): one branch-aware receipt algebra for control and
#     assignment gathers, a subscript / attribute store binds nothing, lambda bodies are nested scope
# ---------------------------------------------------------------------------

_G = "W[m,int(W[m,k])-1]"
_MAC = "Y[m] += X[m,k]*W[m,k]"
_GMAC = f"Y[m] += X[m,k]*{_G}"


def _branch(test, a, b):
    return "if " + test + ":\n" + "".join("    " + x + "\n" for x in a.splitlines()) + "else:\n" + "".join("    " + x + "\n" for x in b.splitlines())


def _inner():
    return _branch(_G + " > 0", _MAC, _MAC)


ROUND8_PIPELINE = [
    # GATHER-CONTROL-JOIN-31: per branch (controls + gather records), per-branch MAXIMUM, plus the enclosing test's own evaluation
    ("nested_branch_controls", _probe5(_branch("k >= 0", _inner(), _inner())), 24, False, (6, 0, 6)),
    ("nested_branch_controls_outer_gather", _probe5(_branch(_G + " > 0", _inner(), _inner())), 24, False, (6, 0, 12)),
    ("branch_control_assignment_join", _probe5(_branch("k >= 0", _branch(_G + " > 0", _MAC + "\n" + _MAC, _MAC + "\n" + _MAC), _GMAC + "\n" + _GMAC)), 48, False, (12, 0, 12)),
    ("branch_control_assignment_join_swapped", _probe5(_branch("k >= 0", _GMAC + "\n" + _GMAC, _branch(_G + " > 0", _MAC + "\n" + _MAC, _MAC + "\n" + _MAC))), 48, False, (12, 0, 12)),
    ("test_and_both_branches_gather", _probe5(_branch(_G + " > 0", _GMAC, _GMAC)), 24, False, (6, 0, 12)),
    ("outer_control_no_assignment", _probe5(_MAC, after_loop="if W[m,int(W[m,0])-1] > 0:\n    pass\nelse:\n    pass"), 24, False, (6, 0, 2)),
    ("outer_control_no_assignment_nested", _probe5(_MAC, after_loop="if W[m,int(W[m,0])-1] > 0:\n    if W[m,int(W[m,0])-1] > 0:\n        pass\n    else:\n        pass\nelse:\n    pass"), 24, False, (6, 0, 4)),
    # SIZE-READ-AS-BINDING-32: a subscript store reads its receiver and index; it binds nothing
    ("size_read_in_store_index", _probe7_sized("Y[rows-2+m] += X[m,k]*W[m,k]"), 24, False, (6, 0, 6)),
    ("size_read_in_store_index_assign", _probe7_sized("Y[rows-2+m] = Y[rows-2+m] + X[m,k]*W[m,k]"), 24, False, (6, 0, 6)),
    # MODULE-LAMBDA-SCOPE-33: a lambda BODY is nested scope (its walrus never executes at definition time); its DEFAULTS are not
    ("lambda_body_local_not_module", _probe5("Y[m] += X[m,k]*abs(k)", "def unused(arg=(lambda: (abs := 1))):\n    pass\n"), 12, False, (0, 6, 0)),
    ("lambda_default_local_body_not_module", _probe5("Y[m] += X[m,k]*abs(k)", "def unused(arg=(lambda z=1: (abs := z))):\n    pass\n"), 12, False, (0, 6, 0)),
    ("module_lambda_body_not_module", _probe5("Y[m] += X[m,k]*abs(k)", "f = lambda: (abs := 1)\n"), 12, False, (0, 6, 0)),
    ("lambda_nested_default_shadow", _probe5("Y[m] += X[m,k]*abs(k)", "def shadow(z):\n    return GLOBAL_W[0,z]\ndef unused(arg=(lambda x=(abs := shadow): x)):\n    pass\n"), 24, True, None),
    ("decorator_comprehension_shadow", _probe5("Y[m] += X[m,k]*abs(k)", "def shadow(z):\n    return GLOBAL_W[0,z]\ndef deco(arg):\n    return lambda f:f\n@deco([(abs := shadow) for item in [0]])\ndef unused():\n    pass\n"), 24, True, None),
    ("comprehension_lambda_body_not_module", _probe5("Y[m] += X[m,k]*abs(k)", "fs = [lambda: (abs := 1) for item in [0]]\n"), 12, False, (0, 6, 0)),
]


@pytest.mark.parametrize("label,code,expected,refuse,exact", ROUND8_PIPELINE, ids=[c[0] for c in ROUND8_PIPELINE])
def test_round8_pipeline(label, code, expected, refuse, exact, tmp_path):
    test_round7_pipeline(label, code, expected, refuse, exact, tmp_path)


def test_round8_control_and_assignment_gathers_share_one_branch_algebra():
    from solar.einsum_dsl import ast_parser as ap
    from solar.einsum_dsl.statement_class import build_context
    def run(src):
        body = ast.parse(src).body
        ctx = build_context(body, ["X", "W", "Y"], (), "mk")
        flags = set()
        out, controls, _scontrols = ap._classify_assignments(body, frozenset("mk"), flags, ctx)   # the third value: singleton controls (issue #61)
        assert flags == set(), flags
        return [r[2] for r in out], controls
    # one gather-bearing inner test in each exclusive branch: ONE control evaluation, not two
    assert run(_branch("k >= 0", _inner(), _inner())) == ([False], 1)
    # the outer test gathers too: its own evaluation adds to the branch maximum
    assert run(_branch(_G + " > 0", _inner(), _inner())) == ([False], 2)
    # one branch: a control; the other: two gathering assignments -> the per-branch maximum of (controls + records) is 2
    g, c = run(_branch("k >= 0", _branch(_G + " > 0", _MAC + "\n" + _MAC, _MAC + "\n" + _MAC), _GMAC + "\n" + _GMAC))
    assert sum(g) + c == 2, (g, c)
    # a gather test with pass arms is still one evaluation
    assert run("if W[m,int(W[m,k])-1] > 0:\n    pass\nelse:\n    pass") == ([], 1)


def test_round8_a_store_binds_nothing():
    from solar.einsum_dsl.statement_class import body_bound_names, build_context, _target_names
    assert _target_names(ast.parse("Y[rows - 2 + m] = 1").body[0].targets[0]) == []
    assert _target_names(ast.parse("Y.attr = 1").body[0].targets[0]) == []
    assert _target_names(ast.parse("a, (b, *c) = v").body[0].targets[0]) == ["a", "b", "c"]
    body = ast.parse("Y = Tensor(M=rows, dtype='fp32')\nfor m in Dim('M'):\n    Y[rows - 2 + m] += X[m, 0]\n    Y[m].x = 1\n").body
    assert body_bound_names(body) == {"Y", "m"}
    ctx = build_context(body, ["X"], ["rows"], int_params=["rows"])
    assert ctx.int_params == {"rows"} and ctx.violations == [], (ctx.int_params, ctx.violations)


def test_round8_module_collector_respects_lambda_scope():
    from solar.einsum_dsl.statement_class import module_bindings, enclosing_scope_walrus_targets
    mod = ast.parse("def unused(arg=(lambda: (abs := 1)), kw=(lambda z=1: (exp := z))):\n    pass\n"
                    "def shadowed(arg=(lambda x=(round := 2): x)):\n    pass\n"
                    "f = lambda: (floor := 1)\n"
                    "g = [(min := 1) for item in [0]]\n"           # a comprehension walrus binds in the containing scope (PEP 572)
                    "h = [lambda: (max := 1) for item in [0]]\n"
                    "class K(metaclass=type, token=(len := 3)):\n    pass\n")
    info = module_bindings(mod)
    assert "abs" not in info and "exp" not in info and "floor" not in info and "max" not in info, sorted(info)
    for name in ("round", "min", "len"):
        assert name in info and "walrus" in info[name]["kinds"], (name, sorted(info))
    expr = ast.parse("(lambda q=(a := 1): (b := q))").body[0].value
    assert enclosing_scope_walrus_targets(expr) == ["a"]


# ---------------------------------------------------------------------------
# 15. issue #61 (manager ruling, round 1): the LITERAL-CONSTANT production and the
#     SINGLETON-SCALE-READ production -- positives, negatives, records, receipts
# ---------------------------------------------------------------------------
# P1 positives: once-bound int / float literal module names are STATIC in STATIC roles; a literal
# multiplier is VALUE arithmetic whose class is unchanged.  (output, refuse?, (macs, other) or None)
ISSUE61_P1 = [
    ("p1_scale_multiplier_value_arithmetic", _probe6("t = X[m,k] * _SCALE\nY[m] += t * W[m,k]", "_SCALE = 0.5\n"), 12, False, (6, 6)),
    ("p1_scale_in_mac_is_a_scale", _probe6("Y[m] += X[m,k] * W[m,k] * _SCALE", "_SCALE = 0.5\n"), 12, False, (6, 0)),
    ("p1_negative_literal_in_sum", _probe6("Y[m] += X[m,k] * W[m,k] + _NEG", "_NEG = -1.0\n"), 21, False, (0, 6)),
    ("p1_tensor_size_int_literal", _probe6("Y[m] += X[m,k]*W[m,k]", "_E = 2\n", alloc='Tensor(M=_E, dtype="fp32")'), 24, False, (6, 0)),
    ("p1_size_arithmetic_with_literal", _probe6("Y[m] += X[m,k]*W[m,k]", "_C = 2\n", alloc='Tensor(M=(X.shape[0] + _C - 1) // _C * _C, dtype="fp32")'), 24, False, (6, 0)),
    ("p1_dim_size_int_literal_loop_bound", _probe6("Y[m] += X[m,k]*W[m,k]", "_TOPK = 3\n", kdim='Dim("K", size=_TOPK)'), 24, False, (6, 0)),
    ("p1_float_literal_fill", _probe6("Y[m] += X[m,k]*W[m,k]", "_INF = 1.2676506002282294e+30\n", alloc='Tensor(M=X.shape[0], dtype="fp32", fill=_INF * -1.0)'), -(2.0 ** 100), False, (6, 0)),   # row 140's _MASK_FILL (2**100, exact in fp32 and fp64)
    ("p1_float_literal_test", _probe6("v = X[m,k]\nif v > _EPS:\n    Y[m] += v * W[m,k]\nelse:\n    Y[m] += v * W[m,k]", "_EPS = 0.5\n"), 24, False, (6, 6)),
    ("p1_nonfinite_float_literal_fill", _probe6("Y[m] += X[m,k]*W[m,k]", "_INF = 1e309\n", alloc='Tensor(M=X.shape[0], dtype="fp32", fill=_INF * -1.0)'), -np.inf, False, (6, 0)),   # manager addition: 1e309 is the +inf literal (row 228's rewrite); float("inf") is a Call
    ("p1_nonfinite_float_literal_test", _probe6("v = X[m,k]\nif v > -_INF:\n    Y[m] += v * W[m,k]\nelse:\n    Y[m] += v * W[m,k]", "_INF = 1e309\n"), 24, False, (6, 6)),
    ("p1_int_literal_index_arithmetic", _probe6("Y[m] += X[m,k]*W[m, (k + _HD) % 3]", "_HD = 3\n"), 24, False, (6, 0)),
    # P1 negatives: outside the production -> the kernel refuses
    ("p1_expression_bound_refuses", _probe6("Y[m] += X[m,k]*W[m, k // _G]", "_NH = 2\n_KV = 1\n_G = _NH // _KV\n"), 24, True, None),
    ("p1_call_bound_inf_fill_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_INF = float(\"inf\")\n", alloc='Tensor(M=X.shape[0], dtype="fp32", fill=_INF * -1.0)'), -np.inf, True, None),
    ("p1_float_literal_size_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_ROWS = 2.0\n", alloc='Tensor(M=int(_ROWS), dtype="fp32")'), 24, True, None),
    ("p1_float_literal_dim_size_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_COLS = 3.0\n", kdim='Dim("K", size=int(_COLS))'), 24, True, None),
    ("p1_float_literal_in_index_refuses", _probe6("Y[m] += X[m,k]*W[m, int(k * _ONE)]", "_ONE = 1.0\n"), 24, True, None),
    # the boundary holds through STATIC ALIASES (the reviewer's re-stack note): a local whose STATIC value derives from a float literal
    ("p1_float_alias_in_index_refuses", _probe6("f = _ONE\nY[m] += X[m,k]*W[m, int(k * f)]", "_ONE = 1.0\n"), 24, True, None),
    ("p1_float_alias_arithmetic_in_size_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_HALF = 0.5\n", alloc='f = _HALF * 4\n    Y = Tensor(M=int(f), dtype="fp32")'), 24, True, None),
    ("p1_float_alias_rebound_to_int_literal_is_static", _probe6("f = _ONE\nf = 1\nY[m] += X[m,k]*W[m, k * f]", "_ONE = 1.0\n"), 24, False, (6, 12)),   # flow order: the rebinding re-derives the mark
    # ISSUE61-FLOAT-SELF-TRANSFER-5: a self-transfer keeps the float-derived mark (the value is evaluated before the destination is rebound)
    ("p1_float_alias_identity_self_refuses", _probe6("f = _ONE\nf = f\nY[m] += X[m,k]*W[m, int(k * f)]", "_ONE = 1.0\n"), 24, True, None),
    ("p1_float_alias_arithmetic_self_refuses", _probe6("f = _ONE\nf = f * 1\nY[m] += X[m,k]*W[m, int(k * f)]", "_ONE = 1.0\n"), 24, True, None),
    ("p1_float_alias_double_cast_self_refuses", _probe6("f = _ONE\nf = int(f)\nf = int(f)\nY[m] += X[m,k]*W[m, k * f]", "_ONE = 1.0\n"), 24, True, None),
    ("p1_float_alias_self_cast_size_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_HALF = 0.5\n", alloc='f = _HALF\n    f = int(f * 4)\n    Y = Tensor(M=int(f * 1), dtype="fp32")'), 24, True, None),
    ("p1_float_alias_self_cast_dim_size_refuses", _probe6("Y[m] += X[m,k]*W[m,k]", "_ONE = 1.0\n", alloc='f = _ONE\n    f = int(f)\n    Y = Tensor(M=X.shape[0], dtype="fp32")', kdim='Dim("K", size=int(f * 3))'), 24, True, None),
    ("p1_twice_bound_refuses", _probe6("Y[m] += X[m,k]*W[m,k]*_S", "_S = 1.0\n_S = 1.0\n"), 24, True, None),
    ("p1_conditionally_bound_refuses", _probe6("Y[m] += X[m,k]*W[m,k]*_S", "_S = 1.0\nif True:\n    _S = 1.0\n"), 24, True, None),
    ("p1_deleted_and_rebound_refuses", _probe6("Y[m] += X[m,k]*W[m,k]*_S", "_S = 1.0\ndel _S\n_S = 1.0\n"), 24, True, None),
    ("p1_annotated_binding_refuses", _probe6("Y[m] += X[m,k]*W[m,k]*_S", "_S: float = 1.0\n"), 24, True, None),
    ("p1_shadowed_builtin_literal_refuses", _probe6("Y[m] += X[m,k]*W[m,k]*abs", "abs = 1.0\n"), 24, True, None),
    ("p1_kernel_local_literal_is_the_local", _probe6("_S = 1.0\nY[m] += X[m,k]*W[m,k]*_S", "_S = 2.0\n"), 24, False, (6, 6)),   # the kernel's own literal binding (flow order), never the module value
]


@pytest.mark.parametrize("label,code,expected,refuse,extra", ISSUE61_P1, ids=[c[0] for c in ISSUE61_P1])
def test_issue61_literal_constant_production_pipeline(label, code, expected, refuse, extra, tmp_path):
    g, layer, total = _probe5_run(code, tmp_path, expected)
    assert (total.get("status") == "incomplete") == refuse, (label, layer["macs"], layer["other_ops"], total.get("status"), g.get("class_evidence_uncertain"), g.get("class_grammar_violations"))
    if refuse:
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), label
    else:
        assert "class_evidence_uncertain" not in g and "singleton_scale_reads" not in g, (label, g.get("class_evidence_uncertain"))
    if extra is not None:
        assert (layer["macs"], layer["other_ops"]) == extra, (label, layer["macs"], layer["other_ops"])


LITE_HEAD_OFFSET = HEADER + '''
_HD = 2

@kernel
def split_heads(X: Tensor("M,F", dtype="fp32")) -> Tensor("M,H,D"):
    Y = Tensor(M=X.shape[0], H=X.shape[1] // _HD, D=_HD, dtype="fp32")
    for m in Dim("M"):
        for h in Dim("H", size=X.shape[1] // _HD):
            for j in Dim("J", size=_HD):
                Y[m, h, j] = X[m, h * _HD + j]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,F", dtype="fp32")
    return {"output": split_heads(x).data}
'''


def test_issue61_head_offset_index_size_and_loop_bound_are_static(tmp_path):
    # the Lite head split: the literal in index arithmetic (h * _HD + j), a Tensor size (D=_HD) and a loop bound (size=_HD)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    graph, out = _traced_analysis(LITE_HEAD_OFFSET, {"x": x}, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("split_heads#"))
    layer = _kernel_layer(out, "split_heads")
    assert "class_evidence_uncertain" not in g and "class_grammar_violations" not in g and out["total"].get("status") != "incomplete", g
    assert (layer["macs"], layer["other_ops"]) == (0, 8), (layer["macs"], layer["other_ops"])


# P2: the SINGLETON-SCALE-READ production -- a literal-0 read of a declared input bound to a one-element tensor
SINGLETON_PROGRAM = HEADER + '''
@kernel
def decode_scale(BM: Tensor("P", dtype="fp32"), GM: Tensor("ONE", dtype="fp32")) -> Tensor("P"):
    Y = Tensor(P=BM.shape[0], dtype="fp32")
    for p in Dim("P"):
        ges = 2688.0 / GM[0]
        Y[p] = BM[p] * ges
    return Y
def run_model(inputs):
    clear_dims()
    bm = Tensor.from_numpy(inputs["bm"], "P", dtype="fp32")
    gm = Tensor.from_numpy(inputs["gm"], "ONE", dtype="fp32")
    return {"output": decode_scale(bm, gm).data}
'''
SINGLETON_INPUTS = {"bm": np.array([1.0, 2.0, 3.0, 4.0], np.float32), "gm": np.array([2688.0], np.float32)}


def _singleton_variant(body, sig='GM: Tensor("ONE", dtype="fp32")', bind='Tensor.from_numpy(inputs["gm"], "ONE", dtype="fp32")', alloc=None):
    # "<loop body>\n@@after@@\n<statements after the loop>": the second part is inserted at the loop's own indentation
    body, _, after = body.partition("\n@@after@@\n")
    code = SINGLETON_PROGRAM.replace("        ges = 2688.0 / GM[0]\n        Y[p] = BM[p] * ges\n",
                                     "".join("        " + l + "\n" for l in body.splitlines()) + "".join("    " + l + "\n" for l in after.splitlines()))
    code = code.replace('GM: Tensor("ONE", dtype="fp32")', sig).replace('Tensor.from_numpy(inputs["gm"], "ONE", dtype="fp32")', bind)
    if alloc:
        code = code.replace('    Y = Tensor(P=BM.shape[0], dtype="fp32")\n', '    Y = Tensor(P=BM.shape[0], dtype="fp32")\n    ' + alloc + "\n")
    return code


def _singleton_run(code, inputs, tmp_path, expected):
    from solar.einsum_lib.verify import execute_translated
    outputs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    assert np.allclose(outputs["output"], np.asarray(expected, np.float32)), (outputs["output"], expected)
    graph, out = _traced_analysis(code, inputs, tmp_path)
    g = next(v for k, v in graph["layers"].items() if k.startswith("decode_scale#"))
    return g, _kernel_layer(out, "decode_scale"), out


ISSUE61_P2 = [
    # label, body, inputs, expected output, refuse?, (macs, other), singleton_scale_reads
    ("p2_lite_decode_scale_read_is_value", "ges = 2688.0 / GM[0]\nY[p] = BM[p] * ges", SINGLETON_INPUTS, [1, 2, 3, 4], False, (0, 8), 4),
    ("p2_singleton_multiplier_is_an_fma", "Y[p] += BM[p] * GM[0]", SINGLETON_INPUTS, [2688, 5376, 8064, 10752], False, (4, 0), 4),
    ("p2_singleton_divisor_is_a_scale", "Y[p] += BM[p] / GM[0]", SINGLETON_INPUTS, [1 / 2688, 2 / 2688, 3 / 2688, 4 / 2688], False, (0, 4), 4),
    ("p2_derived_local_is_value", "g = GM[0]\ninv = 1.0 / g\nY[p] = BM[p] * inv", SINGLETON_INPUTS, [1 / 2688, 2 / 2688, 3 / 2688, 4 / 2688], False, (0, 12), 4),
    ("p2_store_through_singleton_refuses", "GM[0] = BM[p]\nY[p] = BM[p] * GM[0]", SINGLETON_INPUTS, [1, 4, 9, 16], True, None, None),
    ("p2_loop_index_read_is_the_ordinary_rule", "Y[p] = BM[p] * GM[p]", {"bm": SINGLETON_INPUTS["bm"], "gm": np.full(4, 2.0, np.float32)}, [2, 4, 6, 8], False, (0, 4), None),
    ("p2_nonzero_literal_index_refuses", "Y[p] = BM[p] * GM[1]", {"bm": SINGLETON_INPUTS["bm"], "gm": np.array([1.0, 3.0], np.float32)}, [3, 6, 9, 12], True, None, None),
    # ISSUE61-SINGLETON-REBIND-3: a singleton INPUT is eligible only if never rebound anywhere in the body
    ("p2_input_rebound_after_the_loop_refuses", "Y[p] += BM[p] * GM[0]\n@@after@@\nGM = GM", SINGLETON_INPUTS, [2688, 5376, 8064, 10752], True, None, None),
    ("p2_input_rebound_conditionally_refuses", "Y[p] += BM[p] * GM[0]\nif p < 0:\n    GM = GM", SINGLETON_INPUTS, [2688, 5376, 8064, 10752], True, None, None),
    # ISSUE61-SINGLETON-RECEIPT-4: the receipt is per record, counts control evaluations and outer reads
    ("p2_receipt_branch_statements", "if p >= 0:\n    Y[p] += BM[p] * GM[0]\n    Y[p] += BM[p] * BM[p]\nelse:\n    Y[p] += BM[p] * BM[p]\n    Y[p] += BM[p] * BM[p]", SINGLETON_INPUTS, [2689, 5380, 8073, 10768], False, (8, 0), 4),
    ("p2_receipt_condition_only", "if GM[0] > 0:\n    Y[p] += BM[p] * BM[p]\nelse:\n    Y[p] += BM[p] * BM[p]", SINGLETON_INPUTS, [1, 4, 9, 16], False, (4, 0), 4),
    ("p2_receipt_condition_and_statement", "if GM[0] > 0:\n    Y[p] += BM[p] * GM[0]\nelse:\n    Y[p] += BM[p] * BM[p]", SINGLETON_INPUTS, [2688, 5376, 8064, 10752], False, (4, 0), 8),
]


@pytest.mark.parametrize("label,body,inputs,expected,refuse,extra,single", ISSUE61_P2, ids=[c[0] for c in ISSUE61_P2])
def test_issue61_singleton_scale_read_production_pipeline(label, body, inputs, expected, refuse, extra, single, tmp_path):
    sig, bind = ('GM: Tensor("K", dtype="fp32")', 'Tensor.from_numpy(inputs["gm"], "K", dtype="fp32")') if label in ("p2_loop_index_read_is_the_ordinary_rule", "p2_nonzero_literal_index_refuses") else ('GM: Tensor("ONE", dtype="fp32")', 'Tensor.from_numpy(inputs["gm"], "ONE", dtype="fp32")')
    g, layer, out = _singleton_run(_singleton_variant(body, sig, bind), inputs, tmp_path, expected)
    assert (out["total"].get("status") == "incomplete") == refuse, (label, g.get("class_evidence_uncertain"), g.get("class_grammar_violations"), out["total"].get("status"))
    if refuse:
        assert "class_evidence_unproven" in (g.get("class_evidence_uncertain") or ""), label
        return
    assert "class_evidence_uncertain" not in g and "class_grammar_violations" not in g, (label, g)
    assert (layer["macs"], layer["other_ops"]) == extra, (label, layer["macs"], layer["other_ops"])
    if single is None:
        assert "singleton_scale_reads" not in g and "singleton_scale_reads" not in layer, label
    else:
        assert g.get("singleton_scale_reads") == single and layer.get("singleton_scale_reads") == single, (label, g.get("singleton_scale_reads"), layer.get("singleton_scale_reads"))   # the receipt reaches the analysis layer
    assert "gather_index_statements" not in g, label


@pytest.mark.parametrize("label,code,inputs,expected", [
    # a literal-0 read of a NON-singleton declared input: the shape is the production's, the binding does not prove the extent
    ("p2_w00_on_mk_input_refuses_at_the_binding", _probe6("Y[m] += X[m,k]*W[0,0]"), {"x": np.full((2, 3), 4, np.float32), "w": np.full((2, 3), 2, np.float32)}, 24),
    ("p2_state_read_on_two_element_input_refuses", _singleton_variant("Y[p] = BM[p] * GM[0]", 'GM: Tensor("S", dtype="fp32")', 'Tensor.from_numpy(inputs["gm"], "S", dtype="fp32")'), {"bm": SINGLETON_INPUTS["bm"], "gm": np.array([2.0, 3.0], np.float32)}, [2, 4, 6, 8]),
    ("p2_singleton_local_allocation_refuses", _singleton_variant("Y[p] = BM[p] * T[0]", alloc='T = Tensor(ONE=1, dtype="fp32", fill=2.0)'), SINGLETON_INPUTS, [2, 4, 6, 8]),
    ("p2_rank_mismatch_one_index_on_declared_rank_two_refuses", _singleton_variant("Y[p] = BM[p] * G2[0]", 'G2: Tensor("A,B", dtype="fp32")', 'Tensor.from_numpy(inputs["gm"].reshape(1, 1), "A,B", dtype="fp32")'), {"bm": SINGLETON_INPUTS["bm"], "gm": np.array([2.0], np.float32)}, [2, 4, 6, 8]),
])
def test_issue61_singleton_negatives_refuse(label, code, inputs, expected, tmp_path):
    from solar.einsum_lib.verify import execute_translated
    try:
        outputs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    except Exception:
        pytest.skip(f"{label}: the DSL cannot execute this form; the records-level table covers it")
    assert np.allclose(np.asarray(outputs["output"], np.float64).reshape(-1), np.asarray(expected, np.float64).reshape(-1)), (label, outputs["output"])
    graph, out = _traced_analysis(code, inputs, tmp_path)
    kname = "probe" if "probe(" in code else "decode_scale"
    g = next(v for k, v in graph["layers"].items() if k.startswith(kname + "#"))
    assert out["total"]["status"] == "incomplete" and "class_evidence_unproven" in g["class_evidence_uncertain"], (label, g)
    assert "singleton_scale_reads" not in g, label
    if "binding" in label or "two_element" in label:
        assert "not proven" in (g.get("class_grammar_violations") or ""), (label, g.get("class_grammar_violations"))   # the converter names the binding


ISSUE61_MODULE = ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_HD = 128\n_SCALE = 0.125\n_E = 256\n_NEG = -1.0e30\n_G = _HD // 2\n_INF = float('inf')\n_TWICE = 1\n_TWICE = 2\nif True:\n    _COND = 3\nabs = 2\n")


@pytest.mark.parametrize("body,ranks,expected,flags,single", [
    # LITERAL-CONSTANT: records level
    ("Y[h] += X[h, k] * V[h * _HD + j]", {}, [(MAC, 0)], set(), {}),                           # index arithmetic
    ("t = X[h, k] * _SCALE\nY[h] += t", {}, [(OTHER, 0), (OTHER, 0)], set(), {}),              # VALUE arithmetic, class unchanged
    ("Y[h] += X[h, k] * W[h, k] * _SCALE", {}, [(MAC, 0)], set(), {}),                         # a scale in a MAC
    ("Y[h] += X[h, k] * W[h, k] + _NEG", {}, [(OTHER, 0)], set(), {}),                         # a unary-minus literal
    ("Z = Tensor(E=_E, dtype='fp32')\nY[h] += Z[h] * X[h, k]", {}, [(OTHER, 0), (MAC, 0)], set(), {}),             # an int literal size
    ("Z = Tensor(E=_SCALE, dtype='fp32')\nY[h] += Z[h] * X[h, k]", {}, [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),  # a float literal size refuses
    ("Z = Tensor(E=X.shape[0], dtype='fp32', fill=_NEG)\nY[h] += Z[h] * X[h, k]", {}, [(OTHER, 0), (MAC, 0)], set(), {}),   # a float literal fill
    ("Y[h] += X[h, k] * W[h, int(k * _SCALE)]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),            # a float literal inside an index refuses
    ("f = _SCALE\nY[h] += X[h, k] * W[h, int(k * f)]", {}, [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),   # ... and through a STATIC alias
    ("f = _SCALE\ng = f * 2\nZ = Tensor(E=int(g), dtype='fp32')\nY[h] += Z[h] * X[h, k]", {}, [(OTHER, 0), (OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),   # an alias of an alias in a size
    ("f = _SCALE\nf = 2\nY[h] += X[h, k] * W[h, k * f]", {}, [(OTHER, 0), (OTHER, 0), (MAC, 0)], set(), {}),   # rebound to an int literal: the mark is re-derived (flow order)
    ("f = _SCALE\nf = f\nY[h] += X[h, k] * W[h, int(k * f)]", {}, [(OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),      # a self-transfer keeps the mark (SELF-TRANSFER-5)
    ("f = _SCALE\nf = int(f)\nf = int(f)\nY[h] += X[h, k] * W[h, k * f]", {}, [(OTHER, 0), (OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),   # a double self-cast too
    ("f = _SCALE\nf = f * 1\nZ = Tensor(E=int(f), dtype='fp32')\nY[h] += Z[h] * X[h, k]", {}, [(OTHER, 0), (OTHER, 0), (OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),   # ... and into a size
    ("f = _SCALE\nY[h] += X[h, k] * f", {}, [(OTHER, 0), (OTHER, 0)], set(), {}),                      # the alias is a scale in a VALUE role
    ("Y[h] += X[h, k] * W[h, _G]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                         # expression-bound
    ("Y[h] += X[h, k] * _INF", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                             # call-bound
    ("Y[h] += X[h, k] * W[h, k // _TWICE]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                # bound twice
    ("Y[h] += X[h, k] * W[h, k + _COND]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                  # conditionally bound
    ("Y[h] += X[h, k] * abs", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                              # a literal that shadows a whitelisted builtin
    ("Y[h] += X[h, k] * _E[k]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                            # a literal as a tensor receiver
    ("Y[h] += X[h, k] * W[h, k] * _HD\n_HD = 3", {}, [(MAC, 1), (OTHER, 0)], UNPROVEN_FLAG, {}),   # rebound in the kernel: the module value is never admitted (an unproven factor counts as a value)
    # SINGLETON-SCALE-READ: records level (the declared input ranks are the kernel signature)
    ("ges = 2688.0 / GM[0]\nY[h] += X[h, k] * ges", {"GM": 1}, [(OTHER, 0), (MAC, 0)], set(), {"GM": 1}),
    ("Y[h] += X[h, k] * GM[0]", {"GM": 1}, [(MAC, 0)], set(), {"GM": 1}),                      # a runtime scalar multiplier: an FMA
    ("Y[h] += X[h, k] / GM[0]", {"GM": 1}, [(OTHER, 0)], set(), {"GM": 1}),                    # a runtime scalar divisor: a scale
    ("Y[h] += X[h, k] * W[0, 0]", {"W": 2}, [(MAC, 0)], set(), {"W": 2}),                      # the shape is proven here; the extent at the binding
    ("Y[h] += X[h, k] * W[0, 0]", {}, [(MAC, 0)], UNPROVEN_FLAG, {}),                          # not a declared input: refused as before
    ("Y[h] += X[h, k] * G[0]", {"G": 2}, [(MAC, 0)], UNPROVEN_FLAG, {}),                       # rank mismatch
    ("Y[h] += X[h, k] * GM[1]", {"GM": 1}, [(MAC, 0)], UNPROVEN_FLAG, {}),                     # a non-zero literal
    ("GM[0] = X[h, k]\nY[h] += X[h, k]", {"GM": 1}, [(OTHER, 0), (OTHER, 0)], UNPROVEN_FLAG, {}),   # a store through the singleton form
    ("T = Tensor(ONE=1, dtype='fp32')\nY[h] += X[h, k] * T[0]", {}, [(OTHER, 0), (MAC, 0)], UNPROVEN_FLAG, {}),   # a local allocation
    ("Y[h] += X[h, k] * GM[k]", {"GM": 1}, [(MAC, 0)], set(), {}),                             # a loop index: the ordinary rule
])
def test_issue61_records(body, ranks, expected, flags, single):
    from solar.einsum_dsl.statement_class import build_context, classify_records
    mod = ast.parse(body)
    ctx = build_context(mod.body, set(TENSORS) | {"V", "GM", "G"}, (), "hjk", module=ISSUE61_MODULE, input_ranks=ranks)
    records, got = [], set()
    for s in mod.body:
        r, f = classify_records(s, "hjk", ctx); records += r; got |= f
    assert (records, got) == (expected, flags), (records, got)
    assert ctx.singleton_reads == single, ctx.singleton_reads
    assert bool(ctx.singleton_statements) == bool(single), ctx.singleton_statements


def test_issue61_module_literal_kinds_and_kernel_shadowing():
    from solar.einsum_dsl.statement_class import module_literal_kinds, body_bound_names, build_context
    assert module_literal_kinds(ISSUE61_MODULE) == {"_HD": "int", "_SCALE": "float", "_E": "int", "_NEG": "float", "abs": "int"}
    assert module_literal_kinds(ast.parse("_A = True\n_B = 1j\n_C = +2\n_D = -2.5\n_E = 'x'\n_F, _G = 1, 2\n_H = _I = 3\n_J += 1\n")) == {"_C": "int", "_D": "float"}
    assert module_literal_kinds(ast.parse("_INF = 1e309\n_NINF = -1e309\n_CALL = float('inf')\n")) == {"_INF": "float", "_NINF": "float"}   # a non-finite float LITERAL is admitted; a call is not
    assert module_literal_kinds(ast.parse("from os import *\n_A = 1\n")) == {}
    body = ast.parse("for _HD in Dim('K'):\n    pass\nwith open(f) as _E:\n    pass\n").body
    assert body_bound_names(body) >= {"_HD", "_E"}
    ctx = build_context(body, ("X",), (), "hjk", module=ISSUE61_MODULE)
    assert "_HD" not in ctx.literal_names and "_E" not in ctx.literal_names and "abs" not in ctx.literal_names and ctx.literal_names["_SCALE"] == "float"
    ctx = build_context(ast.parse("Y[h] += X[h, k]").body, ("X", "Y", "_NEG"), ("_SCALE",), "hjk", module=ISSUE61_MODULE)
    assert "_NEG" not in ctx.literal_names and "_SCALE" not in ctx.literal_names                 # parameters shadow the module literal


def test_issue61_outer_singleton_read_counts_once_per_call(tmp_path):
    # ISSUE61-SINGLETON-RECEIPT-4: a read outside every loop is one evaluation per kernel call (1, not 0)
    code = _singleton_variant("Y[p] += BM[p] * scale", alloc="scale = GM[0]")
    g, layer, out = _singleton_run(code, SINGLETON_INPUTS, tmp_path, [2688, 5376, 8064, 10752])
    assert out["total"].get("status") != "incomplete" and (layer["macs"], layer["other_ops"]) == (4, 0), (g, layer)
    assert g.get("singleton_scale_reads") == 1 and layer.get("singleton_scale_reads") == 1, (g.get("singleton_scale_reads"), layer.get("singleton_scale_reads"))
    code = _singleton_variant("Y[p] += BM[p] * scale", alloc="scale = GM[0]\n    if GM[0] > 0:\n        scale = scale * 1.0")
    g, layer, out = _singleton_run(code, SINGLETON_INPUTS, tmp_path, [2688, 5376, 8064, 10752])
    assert out["total"].get("status") != "incomplete" and g.get("singleton_scale_reads") == 2, g.get("singleton_scale_reads")   # an outer assignment + an outer control test
    # ISSUE61-OUTER-SINGLETON-BRANCH-6: exclusive outer arms take the per-branch MAXIMUM, the test's own evaluation once
    for alloc, count in [
        ("if BM.shape[0] > 0:\n        scale = GM[0]\n    else:\n        scale = GM[0]", 1),                                                           # one read per arm: 1, not 2
        ("if BM.shape[0] > 0:\n        if BM.shape[0] > 1:\n            scale = GM[0]\n        else:\n            scale = GM[0]\n    else:\n        scale = GM[0]", 1),   # three leaves: 1, not 3
        ("if GM[0] > 0:\n        scale = GM[0]\n    else:\n        scale = GM[0]", 2),                                                                 # a reading test + one read per arm: 2, not 3
        ("scale = GM[0]\n    if GM[0] > 0:\n        scale = GM[0]\n        scale = GM[0]\n    else:\n        scale = GM[0]", 4),                        # 1 + test 1 + max(2, 1)
    ]:
        g, layer, out = _singleton_run(_singleton_variant("Y[p] += BM[p] * scale", alloc=alloc), SINGLETON_INPUTS, tmp_path, [2688, 5376, 8064, 10752])
        assert out["total"].get("status") != "incomplete" and g.get("singleton_scale_reads") == count and layer.get("singleton_scale_reads") == count, (alloc, g.get("singleton_scale_reads"), layer.get("singleton_scale_reads"))


def test_issue61_singleton_eligibility_is_never_rebound_anywhere():
    from solar.einsum_dsl.statement_class import build_context, classify_records, UNPROVEN_FLAG
    for text in ("Y[h] += X[h, k] * GM[0]\nGM = GM", "GM = GM\nY[h] += X[h, k] * GM[0]", "Y[h] += X[h, k] * GM[0]\nif h < 0:\n    GM = X"):
        body = ast.parse(text).body
        ctx = build_context(body, set(TENSORS) | {"GM"}, (), "hk", input_ranks={"GM": 1})
        flags = set()
        for s in body:
            flags |= classify_records(s, "hk", ctx)[1]
        assert UNPROVEN_FLAG in flags and not ctx.singleton_reads and "GM" in ctx.body_rebound, (text, flags, ctx.singleton_reads)


def test_issue61_float_alias_mark_is_flow_state_in_the_fixpoint_key():
    from solar.einsum_dsl.statement_class import build_context, freeze_state
    body = ast.parse("f = _SCALE\nfor k in Dim('K'):\n    if k > 0:\n        f = 2\n    else:\n        f = _SCALE\n    Y[h] += X[h, k] * W[h, int(k * f)]\n").body
    ctx = build_context(body, set(TENSORS), (), "h", module=ISSUE61_MODULE)
    assert "f" in ctx.float_static and ctx.converged                                     # the join keeps the mark (float-derived on one side)
    assert any(len(v) >= 4 and isinstance(v[3], set) for v in ctx._at.values())        # the mark is a snapshot component (component 3; this stack appends the dtype alias records after it) ...
    k1 = freeze_state(ctx._at)
    marked = next(sid for sid, v in ctx._at.items() if v[3])                            # a statement before which f is float-derived
    ctx._at[marked] = ctx._at[marked][:3] + (set(),) + tuple(ctx._at[marked][4:])
    assert freeze_state(ctx._at) != k1                                                   # ... that the whole-state key sees
    ctx = build_context(ast.parse("f = _SCALE\nf = 2\nY[h] += X[h, k] * W[h, k * f]").body, set(TENSORS), (), "hk", module=ISSUE61_MODULE)
    assert "f" not in ctx.float_static                                                   # a rebinding re-derives the mark


def test_issue61_singleton_extent_is_proven_from_the_declared_binding():
    from types import SimpleNamespace as NS
    from solar.einsum_dsl.einsum_converter import _singleton_reads_proven
    def kernel(dims, order=True, tpc=2, reads={"GM": 1}, inputs=2):
        ins = [NS(tensor=NS(dims={"P": 4})), NS(tensor=NS(dims=dims))][:inputs]
        return NS(param_names=["BM", "GM"], inputs=ins, tensor_param_count=tpc, inputs_in_signature_order=order, loop_singleton_reads=reads)
    assert _singleton_reads_proven(kernel({"ONE": 1})) == (True, "")
    assert _singleton_reads_proven(NS(loop_singleton_reads=None)) == (True, "")
    ok, why = _singleton_reads_proven(kernel({"S": 2})); assert not ok and "not a rank-1 singleton" in why
    ok, why = _singleton_reads_proven(kernel({"A": 1, "B": 1})); assert not ok and "not proven" in why            # rank mismatch
    ok, why = _singleton_reads_proven(kernel({"ONE": 1}, order=False)); assert not ok and "signature" in why     # a legacy call-order trace
    ok, why = _singleton_reads_proven(kernel({"ONE": 1}, tpc=None)); assert not ok                                # no clean tensor prefix
    ok, why = _singleton_reads_proven(kernel({"ONE": 1}, inputs=1)); assert not ok                                # fewer recorded inputs than the prefix
