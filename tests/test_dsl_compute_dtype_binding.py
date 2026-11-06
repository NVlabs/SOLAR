# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSL-COMPUTE-DTYPE-BINDING-1 (issue #59): on the DSL route the analyzer
partitions MACs by the contraction's declared compute dtype and emits
``macs_by_compute_dtype`` on the layer and on the total, so the perf model
prices NVFP4 / FP8 MACs at the profile's matching tensor-core rate (B200
declares an NVFP4 rate; H100 declares none and holds).

The compute dtype of a MAC statement is derived from the declared logical
dtypes of the tensors it multiplies, by the SAME rule the torch-capture route
applies to ``torch._scaled_mm`` operands and the metadata override applies to
recorded conversions: the highest-throughput quantised dtype among them
(``nvfp4`` over ``fp8``); no quantised operand -> the workload precision, no
bucket.  The original logical dtypes are preserved beside the partition
(``compute_dtype_operands``).  A kernel whose MAC statements resolve to
different compute dtypes cannot be priced at one rate and fails closed
(``compute_dtype_mixed``); MAC-operand evidence that does not reconcile with
the per-statement MAC count fails closed (``compute_dtype_evidence_invalid``).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer, compute_dtype_of_operands  # noqa: E402
from solar.perf.perf_model import EinsumGraphPerfModel  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the shared rule (torch-capture route and DSL route call the same function)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtypes,expected", [
    (["torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2"], "nvfp4"),     # scaled_mm matrix operands
    (["torch.float8_e4m3fn", "torch.float8_e4m3fn"], "fp8"),
    (["nvfp4", "fp8", "nvfp4", "fp8"], "nvfp4"),                          # DSL block-scaled GEMM operands
    (["fp8", "fp8"], "fp8"),
    (["bf16", "bf16"], None),                                              # workload precision: no bucket
    (["bf16", "bf16", "fp8"], "fp8"),                                      # any quantised operand (torch rule)
    (["fp32", None], None),
    ([], None),
])
def test_compute_dtype_rule(dtypes, expected):
    assert compute_dtype_of_operands(dtypes) == expected


# ---------------------------------------------------------------------------
# 2. tiny kernels through the real pipeline
# ---------------------------------------------------------------------------

def _traced_analysis(code, inputs, tmp_path, stem="toy_einsum_lib"):
    from solar.einsum_lib import build_metadata, get_trace
    from solar.einsum_lib.metadata import write_metadata_yaml
    from solar.einsum_lib.verify import execute_translated
    from solar.einsum_dsl.cli_einsum import translate_with_einsum

    execute_translated(code, {k: v.shape for k, v in inputs.items() if hasattr(v, "shape")},
                       all_inputs=inputs, trace=True, shape_only=False, fast=False)
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


def _kernel_id(out, name):
    return next(k for k in out["layers"] if k.startswith(name + "#"))


HEADER = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n\n"
M, N, K, BLK = 4, 3, 4, 2
NB = K // BLK

NVFP4_GEMM = HEADER + '''
@kernel
def nvfp4_linear(XQ: Tensor("M,K", dtype="nvfp4"), XS: Tensor("M,NB", dtype="fp8"),
                 WV: Tensor("N,K", role="weight", dtype="nvfp4"),
                 WS: Tensor("N,NB", role="weight", dtype="fp8")) -> Tensor("M,N"):
    Y = Tensor(M=XQ.shape[0], N=WV.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += (XQ[m, k] * XS[m, k // 2]) * (WV[n, k] * WS[n, k // 2])
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

FP8_GEMM = HEADER + '''
@kernel
def fp8_linear(X: Tensor("M,K", dtype="fp8"), W: Tensor("N,K", role="weight", dtype="fp8")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += X[m, k] * W[n, k]
            Y[m, n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp8")
    w = Tensor.from_numpy(inputs["w"], "N,K", role="weight", dtype="fp8")
    return {"output": fp8_linear(x, w).data}
'''

BF16_GEMM = HEADER + '''
@kernel
def bf16_linear(X: Tensor("M,K", dtype="bf16"), W: Tensor("N,K", role="weight", dtype="bf16")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += X[m, k] * W[n, k]
            Y[m, n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="bf16")
    w = Tensor.from_numpy(inputs["w"], "N,K", role="weight", dtype="bf16")
    return {"output": bf16_linear(x, w).data}
'''

# one body, two MAC statements at different compute dtypes: not priceable at one rate
MIXED_GEMM = HEADER + '''
@kernel
def mixed_linear(XQ: Tensor("M,K", dtype="nvfp4"), WV: Tensor("N,K", role="weight", dtype="nvfp4"),
                 X: Tensor("M,K", dtype="bf16"), W: Tensor("N,K", role="weight", dtype="bf16")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += XQ[m, k] * WV[n, k]
                acc += X[m, k] * W[n, k]
            Y[m, n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    xq = Tensor.from_numpy(inputs["xq"], "M,K", dtype="nvfp4")
    wv = Tensor.from_numpy(inputs["wv"], "N,K", role="weight", dtype="nvfp4")
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="bf16")
    w = Tensor.from_numpy(inputs["w"], "N,K", role="weight", dtype="bf16")
    return {"output": mixed_linear(xq, wv, x, w).data}
'''

# bf16 matrices with an fp8 per-block scale INSIDE the accumulation: the scale is a
# MAC operand, so the shared "any quantised operand" rule binds the statement to fp8
# (the convention recorded in the module docstring; the operands stay visible)
SCALED_BF16_GEMM = HEADER + '''
@kernel
def scaled_bf16_linear(X: Tensor("M,K", dtype="bf16"), W: Tensor("N,K", role="weight", dtype="bf16"),
                       S: Tensor("N,NB", role="weight", dtype="fp8")) -> Tensor("M,N"):
    Y = Tensor(M=X.shape[0], N=W.shape[0], dtype="fp32")
    for m in Dim("M", parallel=True):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += X[m, k] * W[n, k] * S[n, k // 2]
            Y[m, n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="bf16")
    w = Tensor.from_numpy(inputs["w"], "N,K", role="weight", dtype="bf16")
    s = Tensor.from_numpy(inputs["s"], "N,NB", role="weight", dtype="fp8")
    return {"output": scaled_bf16_linear(x, w, s).data}
'''


def _rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def nvfp4_case(tmp_path_factory):
    r = _rng()
    inputs = {"xq": r.standard_normal((M, K)).astype(np.float32), "xs": r.standard_normal((M, NB)).astype(np.float32),
              "wv": r.standard_normal((N, K)).astype(np.float32), "ws": r.standard_normal((N, NB)).astype(np.float32)}
    tmp = tmp_path_factory.mktemp("nvfp4")
    graph, out = _traced_analysis(NVFP4_GEMM, inputs, tmp)
    return tmp, graph, out


def test_nvfp4_gemm_layer_and_total_are_partitioned(nvfp4_case):
    _, graph, out = nvfp4_case
    layer = _kernel_layer(out, "nvfp4_linear")
    assert layer["macs"] == M * N * K
    assert layer["unit_basis"] == "per_statement"
    assert layer["compute_dtype"] == "nvfp4"
    assert layer["macs_by_compute_dtype"] == {"nvfp4": M * N * K}
    # the original logical dtypes of the MAC operands are preserved beside the partition
    assert layer["compute_dtype_operands"] == {"XQ": "nvfp4", "XS": "fp8", "WV": "nvfp4", "WS": "fp8"}
    assert out["total"]["macs_by_compute_dtype"] == {"nvfp4": M * N * K}
    assert out["total"].get("status") != "incomplete"
    # the dequant multiplies of the four-factor MAC statement and the per-(m,n) init / store are other work
    assert layer["other_ops"] == 2 * M * N * K + 2 * M * N
    # the graph still carries the declared logical dtypes of every operand
    node = next(v for k, v in graph["layers"].items() if k.startswith("nvfp4_linear#"))
    assert node["tensor_dtypes"]["inputs"] == ["nvfp4", "fp8", "nvfp4", "fp8"]


def test_nvfp4_gemm_holds_on_h100_and_prices_on_b200(nvfp4_case):
    tmp, _, out = nvfp4_case
    lid = _kernel_id(out, "nvfp4_linear")
    analysis = str(tmp / "analysis" / "analysis.yaml")
    # the merged target carries an earlier change's pricing contract (perf_model.py, untouched by this branch): a quantised bucket
    # without a declared tensor-core rate REFUSES the projection (PerfProjectionUnavailable) instead of the stack-base
    # `rate_unavailable:nvfp4` hold this test asserted on an earlier revision / an earlier revision (round-6 reconciliation onto an earlier revision)
    from solar.perf.perf_model import PerfProjectionUnavailable
    with pytest.raises(PerfProjectionUnavailable, match="nvfp4"):
        EinsumGraphPerfModel().predict(analysis, str(tmp / "perf_H100_PCIe"), arch_config="H100_PCIe", precision="bf16")
    b = EinsumGraphPerfModel().predict(analysis, str(tmp / "perf_B200"), arch_config="B200", precision="bf16")
    assert b["workload"].get("status") != "incomplete"
    assert lid not in (b["workload"].get("incomplete_layers") or {})
    rate = b["arch"]["compute_dtype_rates"]["nvfp4"]                 # the merged an earlier change perf model adds `layer_macs_bound` beside these three
    # issue #14: the applied key is the explicit sparse (default sol_mode) spelling; B200 nvfp4 sparse 18 PFLOPS -> 6,000,000 MAC/cycle
    assert {k: rate[k] for k in ("key", "MAC_per_cycle", "macs")} == {"key": "MAC_per_cycle_nvfp4_tc_sparse", "MAC_per_cycle": 6_000_000.0, "macs": float(M * N * K)}
    # every MAC is in the nvfp4 bucket: no MAC is priced at the bf16 rate
    assert b["workload"]["total_macs"] == M * N * K


def test_fp8_gemm_is_bound_to_fp8_and_priced_on_h100(tmp_path):
    r = _rng()
    inputs = {"x": r.standard_normal((M, K)).astype(np.float32), "w": r.standard_normal((N, K)).astype(np.float32)}
    _, out = _traced_analysis(FP8_GEMM, inputs, tmp_path)
    layer = _kernel_layer(out, "fp8_linear")
    assert layer["compute_dtype"] == "fp8"
    assert layer["macs_by_compute_dtype"] == {"fp8": M * N * K}
    assert layer["compute_dtype_operands"] == {"X": "fp8", "W": "fp8"}
    assert out["total"]["macs_by_compute_dtype"] == {"fp8": M * N * K}
    h = EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf_H100_PCIe"),
                                       arch_config="H100_PCIe", precision="bf16")
    assert h["workload"].get("status") != "incomplete"
    assert h["arch"]["compute_dtype_rates"]["fp8"]["MAC_per_cycle"] == 756_500.0   # issue #14: H100_PCIe fp8 sparse (default) 3026 TFLOPS


def test_bf16_gemm_carries_no_compute_dtype_fields(tmp_path):
    r = _rng()
    inputs = {"x": r.standard_normal((M, K)).astype(np.float32), "w": r.standard_normal((N, K)).astype(np.float32)}
    _, out = _traced_analysis(BF16_GEMM, inputs, tmp_path)
    layer = _kernel_layer(out, "bf16_linear")
    assert layer["macs"] == M * N * K
    for key in ("compute_dtype", "macs_by_compute_dtype", "compute_dtype_operands"):
        assert key not in layer
    assert "macs_by_compute_dtype" not in out["total"]
    h = EinsumGraphPerfModel().predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf_H100_PCIe"),
                                       arch_config="H100_PCIe", precision="bf16")
    assert h["workload"].get("status") != "incomplete"
    assert "compute_dtype_rates" not in h["arch"]


def test_mixed_compute_dtypes_in_one_kernel_fail_closed(tmp_path):
    r = _rng()
    inputs = {"xq": r.standard_normal((M, K)).astype(np.float32), "wv": r.standard_normal((N, K)).astype(np.float32),
              "x": r.standard_normal((M, K)).astype(np.float32), "w": r.standard_normal((N, K)).astype(np.float32)}
    _, out = _traced_analysis(MIXED_GEMM, inputs, tmp_path)
    lid = _kernel_id(out, "mixed_linear")
    layer = out["layers"][lid]
    assert layer["macs"] == 0 and layer["macs_incomplete"] is True
    assert layer["incomplete_reason"] == "compute_dtype_mixed"
    assert layer["compute_dtype_operands"] == {"XQ": "nvfp4", "WV": "nvfp4", "X": "bf16", "W": "bf16"}
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in layer
    assert out["total"]["status"] == "incomplete"
    assert out["total"]["incomplete_layers"][lid] == "compute_dtype_mixed"
    assert "macs_by_compute_dtype" not in out["total"]


def test_fp8_block_scale_on_bf16_payloads_is_not_an_fp8_contraction(tmp_path):
    # the review DTYPE-CONTRACTION-1 (fp8_scale_only): a scale tensor is not a payload; the two payloads are bf16
    r = _rng()
    inputs = {"x": r.standard_normal((M, K)).astype(np.float32), "w": r.standard_normal((N, K)).astype(np.float32),
              "s": r.standard_normal((N, NB)).astype(np.float32)}
    graph, out = _traced_analysis(SCALED_BF16_GEMM, inputs, tmp_path)
    layer = _kernel_layer(out, "scaled_bf16_linear")
    assert layer["macs"] == M * N * K and "status" not in layer
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in layer
    assert "macs_by_compute_dtype" not in out["total"] and out["total"].get("status") != "incomplete"
    rec = graph["layers"][_node(graph, "scaled_bf16_linear")]["loop_mac_operands"]
    assert [p["name"] for p in rec[0]["payloads"]] == ["X", "W"] and [s["name"] for s in rec[0]["scales"]] == ["S"]


# ---------------------------------------------------------------------------
# 3. evidence guards: the converter's MAC-operand record must reconcile with the
#    per-statement MAC count; a record without classes carries nothing
# ---------------------------------------------------------------------------

def _reanalyze(graph, tmp_path, name="re"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(graph, sort_keys=False))
    return EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / f"{name}_analysis"), precision="fp32", copy_graph=False)


def _node(graph, name):
    return next(k for k in graph["layers"] if k.startswith(name + "#"))


def test_mac_operand_record_is_exported_by_the_converter(nvfp4_case):
    _, graph, _ = nvfp4_case
    node = graph["layers"][_node(graph, "nvfp4_linear")]
    assert node["loop_mac_statements"] == M * N * K
    assert node["loop_mac_operands"] == [{
        "macs": M * N * K,
        "payloads": [{"name": "XQ", "input": 0, "dtype": "nvfp4"}, {"name": "WV", "input": 2, "dtype": "nvfp4"}],
        "scales": [{"name": "XS", "input": 1, "dtype": "fp8"}, {"name": "WS", "input": 3, "dtype": "fp8"}],
        "conversions": [],
        "operands": {"XQ": "nvfp4", "XS": "fp8", "WV": "nvfp4", "WS": "fp8"}}]


def test_mac_operand_record_that_does_not_reconcile_fails_closed(nvfp4_case, tmp_path):
    _, graph, _ = nvfp4_case
    g = yaml.safe_load(yaml.safe_dump(graph))
    nid = _node(g, "nvfp4_linear")
    g["layers"][nid]["loop_mac_operands"] = [{"macs": M * N * K - 1, "operands": {"XQ": "nvfp4", "WV": "nvfp4"}}]
    out = _reanalyze(g, tmp_path)
    layer = out["layers"][nid]
    assert layer["macs"] == 0 and layer["incomplete_reason"] == "compute_dtype_evidence_invalid"
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in out["total"]
    assert out["total"]["incomplete_layers"][nid] == "compute_dtype_evidence_invalid"


@pytest.mark.parametrize("bad", [
    [{"macs": "48", "operands": {"XQ": "nvfp4"}}],         # count not an int
    [{"macs": -1, "operands": {"XQ": "nvfp4"}}, {"macs": M * N * K + 1, "operands": {"WV": "nvfp4"}}],
    [{"macs": M * N * K}],                                   # operands missing
    "nvfp4",                                                 # not a list
])
def test_malformed_mac_operand_record_fails_closed(nvfp4_case, tmp_path, bad):
    _, graph, _ = nvfp4_case
    g = yaml.safe_load(yaml.safe_dump(graph))
    nid = _node(g, "nvfp4_linear")
    g["layers"][nid]["loop_mac_operands"] = bad
    out = _reanalyze(g, tmp_path)
    assert out["layers"][nid]["macs"] == 0
    assert out["layers"][nid]["incomplete_reason"] == "compute_dtype_evidence_invalid"


def test_quantised_node_without_the_operand_record_holds(nvfp4_case, tmp_path):
    # the review DTYPE-EVIDENCE-2: the node declares nvfp4 inputs; an absent record is not a workload-rate default
    _, graph, _ = nvfp4_case
    g = copy.deepcopy(graph)
    nid = _node(g, "nvfp4_linear")
    g["layers"][nid].pop("loop_mac_operands")
    out = _reanalyze(g, tmp_path)
    layer = out["layers"][nid]
    assert layer["macs"] == 0 and layer["macs_incomplete"] is True and layer["incomplete_reason"] == "compute_dtype_evidence_invalid"
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in out["total"] and out["total"]["status"] == "incomplete"


def test_legacy_operand_record_on_a_quantised_node_holds(nvfp4_case, tmp_path):
    # a record in the round-1 shape (an operand map without roles / positions) is not evidence for a quantised node
    _, graph, _ = nvfp4_case
    g = copy.deepcopy(graph)
    nid = _node(g, "nvfp4_linear")
    g["layers"][nid]["loop_mac_operands"] = [{"macs": M * N * K, "operands": {"XQ": "int8", "WV": "bf16"}}]
    out = _reanalyze(g, tmp_path)
    layer = out["layers"][nid]
    assert layer["macs"] == 0 and layer["incomplete_reason"] == "compute_dtype_evidence_invalid"
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in out["total"]


def test_ci_unit_lane_selects_this_module():
    ci = next(p / ".gitlab-ci.yml" for p in Path(__file__).resolve().parents if (p / ".gitlab-ci.yml").is_file()).read_text()
    assert "tests/test_dsl_compute_dtype_binding.py" in ci


# ---------------------------------------------------------------------------
# operand naming on the provenance model (rebased onto the earlier fix round-3 head)
# ---------------------------------------------------------------------------
import ast as _ast  # noqa: E402

from solar.einsum_dsl.statement_class import build_context, mac_operand_names  # noqa: E402


def _operands(body: str, tensors=("X", "W", "V"), scalars=("s",), loop_vars=("m", "k")):
    mod = _ast.parse(body.strip())
    ctx = build_context(mod.body, tensors, scalars, loop_vars)
    stmt = mod.body[-1]
    return mac_operand_names(stmt, loop_vars, ctx)


@pytest.mark.parametrize("body, expected", [
    ("acc += X[m, k] * W[k]", ["X", "W"]),
    ("acc += X[m, k] * W[k] * s", ["X", "W"]),                 # the runtime scalar is not a tensor operand
    ("acc += X[m, k] * W[k] / W.shape[1]", ["X", "W"]),        # shape metadata is not an operand
    ("acc += X[m, k] * W[k] * (k <= m)", ["X", "W"]),           # a mask is not an operand
    ("acc = acc + W[k] * X[m, k]", ["W", "X"]),                 # source order
    ("acc += X[m, k] * 2", []),                                 # not a MAC
    ("Y[m] = X[m, k] * W[k]", []),                              # not accumulated: not a MAC
    ("acc += (X[m, k] * W[k]) if k < m else (X[m, k] * V[k])", ["X", "W", "V"]),   # both arms
    ("a = X[m, k]\nacc += a * W[k]", ["X", "W"]),               # alias resolves to its source tensor
    ("a = X[m, k] * s\nacc += a * W[k]", ["X", "W"]),
    ("denom = W[m, k]\nacc += X[m, k] / denom", ["X", "W"]),    # tensor divisor is an operand
])
def test_mac_operand_names_follow_value_provenance(body, expected):
    assert _operands(body) == expected


def test_mac_operand_names_without_context_treat_subscripted_names_as_tensors():
    stmt = _ast.parse("acc += A[i] * B[i] * c").body[0]
    assert mac_operand_names(stmt, ("i",)) == ["A", "B"]


# ---------------------------------------------------------------------------
# 6. round 2 (the review pre-MR verdict): within-contraction payload agreement,
#    evidence validated against the graph's declared dtypes, reaching alias dtypes
#    (DTYPE-CONTRACTION-1, DTYPE-EVIDENCE-2, DTYPE-BINDING-3)
# ---------------------------------------------------------------------------
def _contract_program(dtypes, expr, prelude=""):
    """the reviewer's probe kernel: `acc += <expr>` over M,K with the named operand dtypes."""
    params = ", ".join(f'{n}: Tensor("M,K", dtype={dt!r})' for n, dt in dtypes.items())
    binds = "\n".join(f'    {n.lower()} = Tensor.from_numpy(inputs[{n.lower()!r}], "M,K", dtype={dt!r})' for n, dt in dtypes.items())
    body = "\n".join("            " + l for l in (prelude.splitlines() + ["acc += " + expr]))
    return ("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, fp32\n\n@kernel\n"
            f"def contract({params}) -> Tensor(\"M\"):\n    Y = Tensor(M=X.shape[0], dtype=\"fp32\")\n"
            "    for m in Dim(\"M\", parallel=True):\n        acc = 0.0\n        for k in Dim(\"K\"):\n" + body +
            "\n        Y[m] = acc\n    return Y\n\ndef run_model(inputs):\n    clear_dims()\n" + binds +
            f"\n    return {{\"output\": contract({', '.join(n.lower() for n in dtypes)}).data}}\n")


def _contract_run(dtypes, expr, prelude, expected, tmp_path):
    from solar.einsum_lib.verify import execute_translated
    inputs = {n.lower(): np.full((2, 3), 2.0 if n == "X" else 1.0, dtype=np.float64) for n in dtypes}
    code = _contract_program(dtypes, expr, prelude)
    outs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    np.testing.assert_allclose(outs["output"], np.full(2, expected), rtol=0, atol=0)
    graph, out = _traced_analysis(code, inputs, tmp_path)
    lid = _kernel_id(out, "contract")
    return graph["layers"][_node(graph, "contract")], out["layers"][lid], out["total"]


# label, operand dtypes, expression, prelude, output per row, expectation:
#   ("hold", reason) -> macs_incomplete with that reason, no bucket; ("none",) -> 6 MACs, no bucket, complete;
#   ("fp8",) -> 6 MACs bound to fp8, complete
ROUND2_CASES = [
    ("mixed_payload", {"X": "nvfp4", "W": "bf16"}, "X[m,k]*W[m,k]", "", 6.0, ("hold", "compute_dtype_mixed")),
    ("fp8_scale_only", {"X": "bf16", "W": "bf16", "S": "fp8"}, "X[m,k]*W[m,k]*S[m,k]", "", 6.0, ("hold", "compute_dtype_mixed")),   # S indexed like a payload: not a block scale
    ("fp8_block_scale_only", {"X": "bf16", "W": "bf16", "S": "fp8"}, "X[m,k]*W[m,k]*S[m,k//2]", "", 6.0, ("none",)),           # a block scale is not a payload: bf16 contraction
    ("dequant_inline_cast", {"X": "fp8", "W": "fp32", "S": "fp8"}, "fp32(X[m,k]*S[m,k])*W[m,k]", "", 6.0, ("none",)),
    ("dequant_local_cast", {"X": "fp8", "W": "fp32", "S": "fp8"}, "a*W[m,k]", "a = fp32(X[m,k]*S[m,k])", 6.0, ("none",)),
    ("conditional_dtypes", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "(X[m,k]*W[m,k]) if k >= 0 else (X[m,k]*Q[m,k])", "", 6.0, ("hold", "compute_dtype_unproven")),
    ("stale_alias_dtype", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "X[m,k]*a", "a = Q[m,k]\na = W[m,k]", 6.0, ("none",)),   # the reaching binding is W (bf16)
    ("rebound_tensor_object", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "X[m,k]*Q[m,k]", "Q = W", 6.0, ("none",)),
    ("int_element_provenance", {"X": "fp8", "W": "fp8"}, "X[m,k]*int(W[m,k])", "", 6.0, ("hold", "compute_dtype_unproven")),
    ("bool_value_provenance", {"X": "fp8", "W": "fp8"}, "X[m,k]*(W[m,k] or 1.0)", "", 6.0, ("hold", "compute_dtype_unproven")),
    ("tuple_alias_provenance", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "X[m,k]*a", "a = W[m,k]\na, junk = Q[m,k], 0", 6.0, ("hold", "class_evidence_unproven")),
    ("unproven_join", {"X": "fp8", "W": "fp8"}, "X[m,k]*(W[m,k]+W[0,0])", "", 12.0, ("hold", "class_evidence_unproven")),
    ("honest_fp8", {"X": "fp8", "W": "fp8"}, "X[m,k]*W[m,k]", "", 6.0, ("fp8",)),
    ("honest_bf16", {"X": "bf16", "W": "bf16"}, "X[m,k]*W[m,k]", "", 6.0, ("none",)),
    ("nvfp4_block_scales", {"X": "nvfp4", "W": "nvfp4", "S": "fp8", "T": "fp8"}, "X[m,k]*S[m,k//2]*(W[m,k]*T[m,k//2])", "", 6.0, ("nvfp4",)),
    ("scale_alias_product", {"X": "nvfp4", "W": "nvfp4", "S": "fp8"}, "X[m,k]*S[m,k//2]*ws", "ws = W[m,k] * S[m,k//2]", 6.0, ("nvfp4",)),
    ("decoded_packed_weight", {"X": "nvfp4", "P": "int8", "S": "fp8"}, "X[m,k]*S[m,k//2]*ws", "c = P[m,k//2] % 256.0\nws = (c % 16.0) * S[m,k//2]", 6.0, ("hold", "compute_dtype_unproven")),
]


@pytest.mark.parametrize("label,dtypes,expr,prelude,output,expect", ROUND2_CASES, ids=[c[0] for c in ROUND2_CASES])
def test_round2_contraction_dtype(label, dtypes, expr, prelude, output, expect, tmp_path):
    node, layer, total = _contract_run(dtypes, expr, prelude, output, tmp_path)
    if expect[0] == "hold":
        assert (layer.get("macs_incomplete") is True or layer.get("status") == "incomplete") and total["status"] == "incomplete", (label, layer)
        assert layer["incomplete_reason"] == expect[1], (label, layer.get("incomplete_reason"))
        assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in total, label
    else:
        assert layer["macs"] == 6 and "status" not in layer and total.get("status") != "incomplete", (label, layer)
        if expect[0] == "none":
            assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in total, (label, layer.get("compute_dtype"))
        else:
            assert layer["compute_dtype"] == expect[0] and total["macs_by_compute_dtype"] == {expect[0]: 6}, label


@pytest.mark.parametrize("mode", ["absent", "empty_record", "legacy_operands_only", "wrong_dtype", "wrong_input", "unknown_name", "unknown_dtype", "null_dtype", "one_payload"])
def test_round2_evidence_against_declared_dtypes_fails_closed(mode, tmp_path):
    # the review DTYPE-EVIDENCE-2: the graph declares nvfp4 inputs; evidence that is missing, malformed or inconsistent
    # with the declared dtypes holds instead of pricing six unbucketed MACs at the workload rate
    node, _, _ = _contract_run({"X": "nvfp4", "W": "nvfp4"}, "X[m,k]*W[m,k]", "", 6.0, tmp_path)
    graph = yaml.safe_load((tmp_path / "einsum" / "toy_einsum_graph.yaml").read_text()) if (tmp_path / "einsum" / "toy_einsum_graph.yaml").exists() else None
    if graph is None:
        graph = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())
    nid = _node(graph, "contract"); n = graph["layers"][nid]
    good = n["loop_mac_operands"][0]
    if mode == "absent":
        n.pop("loop_mac_operands")
    elif mode == "empty_record":
        n["loop_mac_operands"] = [{"macs": 6, "payloads": [], "scales": [], "conversions": [], "operands": {}}]
    elif mode == "legacy_operands_only":
        n["loop_mac_operands"] = [{"macs": 6, "operands": {"X": "nvfp4", "W": "nvfp4"}}]
    elif mode == "wrong_dtype":
        n["loop_mac_operands"] = [dict(good, payloads=[dict(p, dtype="bf16") for p in good["payloads"]])]
    elif mode == "wrong_input":
        n["loop_mac_operands"] = [dict(good, payloads=[dict(p, input=p["input"] + 7) for p in good["payloads"]])]
    elif mode == "unknown_name":
        n["loop_mac_operands"] = [dict(good, payloads=[{"name": "Z", "input": 0, "dtype": "nvfp4"}, good["payloads"][1]])]
    elif mode == "unknown_dtype":
        n["loop_mac_operands"] = [dict(good, payloads=[dict(p, dtype="mystery") for p in good["payloads"]])]
    elif mode == "null_dtype":
        n["loop_mac_operands"] = [dict(good, payloads=[dict(p, dtype=None) for p in good["payloads"]])]
    elif mode == "one_payload":
        n["loop_mac_operands"] = [dict(good, payloads=good["payloads"][:1])]
    out = _reanalyze(graph, tmp_path, "altered")
    lid = _kernel_id(out, "contract"); layer = out["layers"][lid]
    assert out["total"]["status"] == "incomplete" and layer["macs_incomplete"] is True, (mode, layer)
    assert layer["incomplete_reason"] == "compute_dtype_evidence_invalid", (mode, layer["incomplete_reason"])
    assert "macs_by_compute_dtype" not in out["total"], mode


def test_round2_legacy_non_quantized_graph_keeps_the_workload_precision(tmp_path):
    # a graph whose declared inputs are not quantized carries no quantised claim: an absent record is the legacy path
    node, _, _ = _contract_run({"X": "bf16", "W": "bf16"}, "X[m,k]*W[m,k]", "", 6.0, tmp_path)
    graph = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())
    graph["layers"][_node(graph, "contract")].pop("loop_mac_operands")
    out = _reanalyze(graph, tmp_path, "legacy")
    layer = _kernel_layer(out, "contract")
    assert layer["macs"] == 6 and "status" not in layer and out["total"].get("status") != "incomplete"
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in out["total"]


def test_quantised_kernel_without_mac_statements_contracts_nothing(tmp_path):
    # a dequantisation kernel (fp8 inputs, no accumulated product) has no compute-dtype claim to prove:
    # its empty operand record is not missing evidence (the reviewer's separate-kernel dequant positive control)
    code = HEADER.replace("clear_dims\n", "clear_dims, fp32\n") + '''
@kernel
def dequant(X: Tensor("M,K", dtype="fp8"), S: Tensor("M,K", dtype="fp8")) -> Tensor("M,K"):
    D = Tensor(M=X.shape[0], K=X.shape[1], dtype="fp32")
    for m in Dim("M", parallel=True):
        for k in Dim("K"):
            D[m, k] = fp32(X[m, k] * S[m, k])
    return D

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp8")
    s = Tensor.from_numpy(inputs["s"], "M,K", dtype="fp8")
    return {"output": dequant(x, s).data}
'''
    inputs = {"x": np.full((2, 3), 2.0), "s": np.ones((2, 3))}
    graph, out = _traced_analysis(code, inputs, tmp_path)
    node = graph["layers"][_node(graph, "dequant")]
    assert node["loop_mac_statements"] == 0 and node["loop_mac_operands"] == []
    layer = _kernel_layer(out, "dequant")
    assert layer["macs"] == 0 and "status" not in layer and out["total"].get("status") != "incomplete"
    assert "compute_dtype" not in layer and "macs_by_compute_dtype" not in out["total"]


# ---------------------------------------------------------------------------
# 7. the BLOCK-INDEX PRODUCTION of the supported grammar (an earlier fix round 5, manager decision): with a
#    module-level `_BLK`, the block-scale tensors are SCALES (not payloads) and the contraction keeps
#    its declared payload dtype for the binding
# ---------------------------------------------------------------------------
BLOCK_NVFP4_GEMM = HEADER + '''
_BLK = 2

@kernel
def nvfp4_linear(XQ: Tensor("M,K", dtype="nvfp4"), XS: Tensor("M,NB", dtype="fp8"),
                 WV: Tensor("N,K", role="weight", dtype="nvfp4"),
                 WS: Tensor("N,NB", role="weight", dtype="fp8")) -> Tensor("M,N"):
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


def test_block_index_production_roles_and_binding(tmp_path):
    r = _rng()
    inputs = {"xq": r.standard_normal((M, K)).astype(np.float32), "xs": r.standard_normal((M, NB)).astype(np.float32),
              "wv": r.standard_normal((N, K)).astype(np.float32), "ws": r.standard_normal((N, NB)).astype(np.float32)}
    graph, out = _traced_analysis(BLOCK_NVFP4_GEMM, inputs, tmp_path)
    node = graph["layers"][_node(graph, "nvfp4_linear")]
    assert "class_evidence_uncertain" not in node and "class_grammar_violations" not in node     # in the grammar
    rec = node["loop_mac_operands"]
    assert [(p["name"], p["dtype"]) for p in rec[0]["payloads"]] == [("XQ", "nvfp4"), ("WV", "nvfp4")]   # payloads keep their declared dtype
    assert [(s["name"], s["dtype"]) for s in rec[0]["scales"]] == [("XS", "fp8"), ("WS", "fp8")]        # block scales are scales, not payloads
    assert rec[0]["conversions"] == [] and rec[0]["macs"] == M * N * K
    layer = _kernel_layer(out, "nvfp4_linear")
    assert layer["macs"] == M * N * K and layer["compute_dtype"] == "nvfp4" and "status" not in layer
    assert out["total"]["macs_by_compute_dtype"] == {"nvfp4": M * N * K} and out["total"].get("status") != "incomplete"


def test_block_index_production_specs_records():
    from solar.einsum_dsl.statement_class import build_context, mac_operand_specs
    module = _ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_BLK = 16\n")
    body = _ast.parse("acc += XQ[m, k] * XS[m, k // _BLK] * (WV[n, k] * WS[n, k // _BLK])").body
    ctx = build_context(body, ("XQ", "XS", "WV", "WS"), (), "mnk", module=module)
    specs = mac_operand_specs(body[0], "mnk", ctx)
    assert specs["payloads"] == ["XQ", "WV"] and specs["scales"] == ["XS", "WS"] and specs["conversions"] == []
    assert not specs["unproven"] and not specs["arms_differ"]
    # row 205's form: the weight scale enters through a local product alias
    body = _ast.parse("ws = WV[n, k] * WS[n, k // _BLK]\nacc += XQ[m, k] * XS[m, k // _BLK] * ws").body
    ctx = build_context(body, ("XQ", "XS", "WV", "WS"), (), "mnk", module=module)
    specs = mac_operand_specs(body[1], "mnk", ctx)
    assert specs["payloads"] == ["XQ", "WV"] and specs["scales"] == ["XS", "WS"] and not specs["unproven"]


def test_block_scale_rule_keeps_the_gqa_group_index_as_a_payload():
    # row 207's qk_scaled_mm: K is read through the head-group index h // _G (the production's shape) but it is a
    # PAYLOAD -- a scale must share every other index with its payload and floor-divide an index the payload has plainly
    from solar.einsum_dsl.statement_class import build_context, mac_operand_specs
    module = _ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_BLK = 16\n_G = 5\n")
    body = _ast.parse("acc += QQ[b, h, s, d] * QS[b, h, s, d // _BLK] * (KK[b, h // _G, t, d] * KS[b, h // _G, t, d // _BLK])").body
    ctx = build_context(body, ("QQ", "QS", "KK", "KS"), (), "bhstd", module=module)
    specs = mac_operand_specs(body[0], "bhstd", ctx)
    assert specs["payloads"] == ["QQ", "KK"] and specs["scales"] == ["QS", "KS"] and not specs["unproven"]
    # a floor-divided index with no payload that indexes it plainly is a payload access, not a scale
    body = _ast.parse("acc += X[m, k] * W[m, k // _BLK]").body
    ctx = build_context(body, ("X", "W"), (), "mk", module=module)
    assert mac_operand_specs(body[0], "mk", ctx)["scales"] == ["W"]      # W shares m and floor-divides k that X has plainly
    body = _ast.parse("acc += X[m, k] * W[n, k // _BLK]").body
    ctx = build_context(body, ("X", "W"), (), "mnk", module=module)
    sp = mac_operand_specs(body[0], "mnk", ctx)
    assert sp["payloads"] == ["X", "W"] and sp["scales"] == []           # n is not an index of X: W is not X's scale


# ---------------------------------------------------------------------------
# 7. round 3 (the review pre-MR verdict on an earlier revision): alias operands frozen at the defining assignment, the receipt
#    reconciled against the parser's independent occurrence channel, present records validated at 0 MAC, and
#    occurrence lists (not deduplicated names) for the two-factor test (DTYPE-ALIAS-CAPTURE-4, DTYPE-EVIDENCE-BINDING-5,
#    DTYPE-ZERO-RECONCILIATION-6, DTYPE-ROLE-MULTIPLICITY-7)
# ---------------------------------------------------------------------------

ROUND3_CASES = [
    # ROLE-MULTIPLICITY-7: one entry per multiplicative factor OCCURRENCE
    ("two_conversions", {"X": "fp8", "W": "fp8"}, "fp32(X[m,k])*fp32(W[m,k])", "", 6.0, ("none",)),
    ("nested_conversions", {"X": "fp8", "W": "fp8"}, "fp32(fp32(X[m,k]))*fp32(W[m,k])", "", 6.0, ("none",)),
    ("same_payload_twice", {"X": "fp8"}, "X[m,k]*X[m,k]", "", 12.0, ("fp8",)),
    ("same_payload_thrice", {"X": "fp8"}, "X[m,k]*X[m,k]*X[m,k]", "", 24.0, ("fp8",)),
    # ALIAS-CAPTURE-4: an alias keeps the operands it captured AT ITS DEFINITION
    ("alias_chain_late_rebind", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "X[m,k]*b", "a = Q[m,k]\nb = a\na = W[m,k]", 6.0, ("hold", "compute_dtype_mixed")),
    ("alias_capture_quantized", {"X": "nvfp4", "W": "nvfp4", "Q": "bf16"}, "X[m,k]*b", "a = Q[m,k]\nb = a\na = W[m,k]", 6.0, ("hold", "compute_dtype_mixed")),
    ("alias_capture_quantized_control", {"X": "nvfp4", "W": "nvfp4", "Q": "bf16"}, "X[m,k]*Q[m,k]", "", 6.0, ("hold", "compute_dtype_mixed")),
    ("alias_capture_rebound_to_quantized", {"X": "bf16", "W": "fp8", "Q": "bf16"}, "X[m,k]*b", "a = Q[m,k]\nb = a\na = W[m,k]", 6.0, ("none",)),      # b still reads Q (bf16)
    # a tensor PARAMETER rebound inside the loop has no single reaching object at the loop head (the param on trip 1, W
    # afterwards): the class path refuses the read `Q[m, k]` itself (the reviewer's alias_late_rebind accepts any hold)
    ("alias_object_late_rebind", {"X": "bf16", "W": "bf16", "Q": "fp8"}, "X[m,k]*a", "a = Q[m,k]\nQ = W", 6.0, ("hold", "class_evidence_unproven")),
    # a product alias captured W (nvfp4) and the S scale when it was defined; rebinding the LOCAL afterwards does not reach the capture
    ("alias_product_chain_late_rebind", {"X": "nvfp4", "W": "nvfp4", "S": "fp8", "T": "bf16"}, "X[m,k]*S[m,k//2]*w2", "ws = W[m,k] * S[m,k//2]\nw2 = ws\nws = T[m,k]", 6.0, ("nvfp4",)),
]


@pytest.mark.parametrize("label,dtypes,expr,prelude,output,expect", ROUND3_CASES, ids=[c[0] for c in ROUND3_CASES])
def test_round3_contraction_dtype(label, dtypes, expr, prelude, output, expect, tmp_path):
    test_round2_contraction_dtype(label, dtypes, expr, prelude, output, expect, tmp_path)


def test_round3_specs_are_occurrence_lists_frozen_at_definition():
    from solar.einsum_dsl.statement_class import build_context, mac_operand_specs
    body = _ast.parse("acc += fp32(X[m, k]) * fp32(W[m, k])").body
    sp = mac_operand_specs(body[0], "mk", build_context(body, ("X", "W"), (), "mk"))
    assert sp["conversions"] == ["fp32", "fp32"] and sp["occurrences"] == [] and sp["names"] == ["X", "W"], sp
    body = _ast.parse("acc += X[m, k] * X[m, k]").body
    sp = mac_operand_specs(body[0], "mk", build_context(body, ("X",), (), "mk"))
    assert [(o["name"], o["role"], o["index"]) for o in sp["occurrences"]] == [("X", "payload", "m, k"), ("X", "payload", "m, k")] and sp["payloads"] == ["X"], sp
    # the alias chain: b captured Q when it was defined; a's later rebinding to W does not reach b
    body = _ast.parse("a = Q[m, k]\nb = a\na = W[m, k]\nacc += X[m, k] * b").body
    ctx = build_context(body, ("X", "W", "Q"), (), "mk")
    sp = mac_operand_specs(body[3], "mk", ctx)
    assert [(o["name"], o["role"]) for o in sp["occurrences"]] == [("X", "payload"), ("Q", "payload")] and not sp["unproven"], sp
    ctx.select(body[3])
    assert ctx.alias_spec("b") is not None and ctx.alias_spec("b") == ctx.alias_spec("b")      # a frozen, value-comparable record
    # a tensor-object alias rebound after the capture
    body = _ast.parse("a = Q[m, k]\nQ = W\nacc += X[m, k] * a").body
    sp = mac_operand_specs(body[2], "mk", build_context(body, ("X", "W", "Q"), (), "mk"))
    assert [o["name"] for o in sp["occurrences"]] == ["X", "Q"], sp
    # a scale captured through a product alias keeps its role facts
    module = _ast.parse("from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n_BLK = 2\n")
    body = _ast.parse("ws = W[m, k] * S[m, k // _BLK]\nW = T\nacc += X[m, k] * S[m, k // _BLK] * ws").body
    sp = mac_operand_specs(body[2], "mk", build_context(body, ("X", "W", "S", "T"), (), "mk", module=module))
    assert sp["payloads"] == ["X", "W"] and sp["scales"] == ["S"] and [o["role"] for o in sp["occurrences"]] == ["payload", "scale", "payload", "scale"], sp


def test_round3_converter_exports_the_occurrence_channel_beside_the_receipt(tmp_path):
    node, layer, total = _contract_run({"X": "nvfp4", "W": "nvfp4", "S": "fp8"}, "X[m,k]*W[m,k]*S[m,k//2]", "", 6.0, tmp_path)
    rec, reads = node["loop_mac_operands"], node["loop_mac_reads"]
    assert len(rec) == len(reads) == 1 and reads[0]["macs"] == rec[0]["macs"] == 6
    assert [(o["name"], o["input"], o["index"], o["role"]) for o in reads[0]["occurrences"]] == [("X", 0, "m, k", "payload"), ("W", 1, "m, k", "payload"), ("S", 2, "m, k // 2", "scale")]
    assert reads[0]["conversions"] == [] and reads[0]["names"] == ["X", "W", "S"]
    assert [(p["name"], p["input"], p["dtype"]) for p in rec[0]["payloads"]] == [("X", 0, "nvfp4"), ("W", 1, "nvfp4")]
    assert [(s["name"], s["input"], s["dtype"]) for s in rec[0]["scales"]] == [("S", 2, "fp8")]
    assert layer["compute_dtype"] == "nvfp4" and total["macs_by_compute_dtype"] == {"nvfp4": 6}


def _altered_graph(tmp_path, dtypes, expr, prelude=""):
    _contract_run(dtypes, expr, prelude, 6.0 if len(dtypes) > 1 else 12.0, tmp_path)
    graph = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())
    return graph, graph["layers"][_node(graph, "contract")]


ROUND3_EVIDENCE_MODES = ["permute_same_labels", "unused_input", "synthetic_conversion", "unknown_conversion", "zero_count_with_positive_record",
                         "payload_as_scale", "duplicate_payload", "missing_reads_channel", "reads_length_mismatch", "extra_operand_name",
                         "unknown_conversion_bf16_node"]


@pytest.mark.parametrize("mode", ROUND3_EVIDENCE_MODES)
def test_round3_receipt_reconciles_against_the_occurrence_channel(mode, tmp_path):
    # the review EVIDENCE-BINDING-5 / ZERO-RECONCILIATION-6: graph-evidence boundary probes (unchanged source); every
    # altered receipt is caught against the parser's occurrence channel, never against its own map alone
    dtypes = {"X": "bf16", "W": "bf16"} if mode == "unknown_conversion_bf16_node" else {"X": "nvfp4", "W": "nvfp4", "Q": "nvfp4"}
    graph, n = _altered_graph(tmp_path, dtypes, "X[m,k]*W[m,k]")
    rec = n["loop_mac_operands"][0]
    if mode == "permute_same_labels":
        rec["payloads"][0]["input"], rec["payloads"][1]["input"] = rec["payloads"][1]["input"], rec["payloads"][0]["input"]
    elif mode == "unused_input":
        rec["payloads"][1].update(name="Q", input=2, dtype="nvfp4"); rec["operands"].pop("W"); rec["operands"]["Q"] = "nvfp4"
    elif mode in ("synthetic_conversion", "unknown_conversion", "unknown_conversion_bf16_node"):
        rec["payloads"] = []; rec["scales"] = []; rec["operands"] = {}
        rec["conversions"] = ["nvfp4", "nvfp4"] if mode == "synthetic_conversion" else ["unrecognized", "unrecognized"]
    elif mode == "zero_count_with_positive_record":
        n["loop_other_statements"] += n["loop_mac_statements"]; n["loop_mac_statements"] = 0
    elif mode == "payload_as_scale":
        rec["scales"].append(rec["payloads"].pop()); rec["conversions"] = ["nvfp4"]
    elif mode == "duplicate_payload":
        rec["payloads"] = [copy.deepcopy(rec["payloads"][0]), copy.deepcopy(rec["payloads"][0])]
    elif mode == "missing_reads_channel":
        n.pop("loop_mac_reads")
    elif mode == "reads_length_mismatch":
        n["loop_mac_reads"] = n["loop_mac_reads"] + [copy.deepcopy(n["loop_mac_reads"][0])]
    elif mode == "extra_operand_name":
        rec["operands"]["Q"] = "nvfp4"
    out = _reanalyze(graph, tmp_path, "altered")
    layer = out["layers"][_kernel_id(out, "contract")]
    assert out["total"]["status"] == "incomplete" and layer["macs_incomplete"] is True, (mode, layer)
    assert layer["incomplete_reason"] == "compute_dtype_evidence_invalid", (mode, layer["incomplete_reason"])
    assert "macs_by_compute_dtype" not in out["total"], mode


def test_round3_mixed_evidence_cannot_be_promoted(tmp_path):
    # the original X nvfp4 x W bf16 hold cannot be turned into six NVFP4 MACs by pointing the receipt at an unused NVFP4 input
    graph, n = _altered_graph(tmp_path, {"X": "nvfp4", "W": "bf16", "Q": "nvfp4"}, "X[m,k]*W[m,k]")
    rec = n["loop_mac_operands"][0]
    rec["payloads"][1] = {"name": "Q", "input": 2, "dtype": "nvfp4"}; rec["operands"] = {"X": "nvfp4", "Q": "nvfp4"}
    out = _reanalyze(graph, tmp_path, "altered")
    layer = out["layers"][_kernel_id(out, "contract")]
    assert out["total"]["status"] == "incomplete" and layer["incomplete_reason"] == "compute_dtype_evidence_invalid" and "macs_by_compute_dtype" not in out["total"], layer


def test_round3_zero_mac_positive_controls_stay_valid(tmp_path):
    # an EMPTY record on a genuine zero-MAC kernel is the valid positive control (the dequant kernel of section 6)
    dequant_only = HEADER + '''
@kernel
def dequant(X: Tensor("M,K", dtype="fp8"), S: Tensor("M,K", dtype="fp8")) -> Tensor("M,K"):
    Y = Tensor(M=X.shape[0], K=X.shape[1], dtype="fp32")
    for m in Dim("M"):
        for k in Dim("K"):
            Y[m, k] = X[m, k] * S[m, k]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "M,K", dtype="fp8")
    s = Tensor.from_numpy(inputs["s"], "M,K", dtype="fp8")
    return {"output": dequant(x, s).data}
'''
    graph, out = _traced_analysis(dequant_only, {"x": np.ones((2, 3), np.float32), "s": np.ones((2, 3), np.float32)}, tmp_path)
    node = graph["layers"][_node(graph, "dequant")]
    assert node.get("loop_mac_statements") == 0 and node.get("loop_mac_operands") == [] and node.get("loop_mac_reads") == []
    layer = _kernel_layer(out, "dequant")
    assert layer["macs"] == 0 and "status" not in layer and out["total"].get("status") != "incomplete"
    # a quantised OUTPUT alone, absent record: the bf16 contraction keeps the workload precision (the reviewer's output_only_absent)
    code = _contract_program({"X": "bf16", "W": "bf16"}, "X[m,k]*W[m,k]").replace('Tensor(M=X.shape[0], dtype="fp32")', 'Tensor(M=X.shape[0], dtype="nvfp4")')
    from solar.einsum_lib.verify import execute_translated
    inputs = {"x": np.ones((2, 3)), "w": np.ones((2, 3))}
    execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    graph, _ = _traced_analysis(code, inputs, tmp_path / "oo")
    graph["layers"][_node(graph, "contract")].pop("loop_mac_operands"); graph["layers"][_node(graph, "contract")].pop("loop_mac_reads", None)
    out = _reanalyze(graph, tmp_path / "oo", "absent")
    assert out["total"].get("status") != "incomplete" and "macs_by_compute_dtype" not in out["total"], out["total"]


# ---------------------------------------------------------------------------
# 8. round 4 (the reviewer's verdict on an earlier revision): the occurrence channel is a validated CACHE -- the analyzer re-derives the
#    operand occurrences from the kernel source bound to the node (text + sha256 + signature, module text at graph level)
#    and refuses on any difference; a present entry is fully validated BEFORE the zero-count fast path
#    (DTYPE-OCCURRENCE-CHANNEL-8, DTYPE-ZERO-PAYLOAD-SKIP-9)
# ---------------------------------------------------------------------------
import hashlib as _hashlib


def _graph_and_node(tmp_path, dtypes, expr, prelude="", expected=6.0):
    _contract_run(dtypes, expr, prelude, expected, tmp_path)
    graph = yaml.safe_load(next((tmp_path / "einsum").glob("*.yaml")).read_text())
    return graph, graph["layers"][_node(graph, "contract")]


def _assert_invalid(out, label):
    layer = out["layers"][_kernel_id(out, "contract")]
    assert out["total"]["status"] == "incomplete" and layer["macs_incomplete"] is True, (label, layer)
    assert layer["incomplete_reason"] == "compute_dtype_evidence_invalid", (label, layer["incomplete_reason"])
    assert "macs_by_compute_dtype" not in out["total"], label


def test_round4_converter_binds_the_kernel_source_to_the_node(tmp_path):
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "nvfp4"}, "X[m,k]*W[m,k]")
    ks, ms = n["kernel_source"], graph["dsl_module_source"]
    assert ks["name"] == "contract" and ks["text"].startswith("def contract(") and ks["text"] in ms["text"]   # the def segment (decorators are module statements)
    assert ks["sha256"] == _hashlib.sha256(ks["text"].encode()).hexdigest() and ks["module_sha256"] == ms["sha256"] == _hashlib.sha256(ms["text"].encode()).hexdigest()
    assert [(s["name"], s["position"], s["dtype"]) for s in ks["signature"]] == [("X", 0, "nvfp4"), ("W", 1, "nvfp4")]


def test_round4_source_rederivation_is_a_pure_function_of_the_bound_text(tmp_path):
    from solar.einsum_dsl.ast_parser import kernel_source_binding
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "nvfp4", "S": "fp8"}, "X[m,k]*W[m,k]*S[m,k//2]")
    b = kernel_source_binding(graph["dsl_module_source"]["text"], "contract")
    assert [(s["name"], s["position"], s["dtype"]) for s in b["signature"]] == [("X", 0, "nvfp4"), ("W", 1, "nvfp4"), ("S", 2, "fp8")]
    assert len(b["specs"]) == 1 and [(o["name"], o["role"]) for o in b["specs"][0]["occurrences"]] == [("X", "payload"), ("W", "payload"), ("S", "scale")]
    # the same text re-derived twice gives the same record; the channel on the node is exactly that record
    assert kernel_source_binding(graph["dsl_module_source"]["text"], "contract") == b
    r = n["loop_mac_reads"][0]
    assert [(o["name"], o["input"], o["index"], o["role"]) for o in r["occurrences"]] == [(o["name"], {s["name"]: s["position"] for s in b["signature"]}.get(o["name"]), o["index"], o["role"]) for o in b["specs"][0]["occurrences"]]


ROUND4_BINDING_MODES = ["two_channels_unused_quantized_input", "two_channels_invented_fp32_conversions", "edited_source_rehashed", "edited_kernel_hash_only",
                        "edited_module_hash_only", "no_kernel_source_on_quantised_node", "no_module_source", "kernel_text_not_in_module", "signature_dtype_edited"]


@pytest.mark.parametrize("mode", ROUND4_BINDING_MODES)
def test_round4_channels_are_validated_against_the_bound_source(mode, tmp_path):
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "bf16", "Q": "nvfp4"}, "X[m,k]*W[m,k]")
    rec, reads, ks = n["loop_mac_operands"][0], n["loop_mac_reads"][0], n["kernel_source"]
    if mode == "two_channels_unused_quantized_input":
        rec["payloads"][1] = {"name": "Q", "input": 2, "dtype": "nvfp4"}; rec["operands"].pop("W"); rec["operands"]["Q"] = "nvfp4"
        reads["occurrences"][1].update(name="Q", input=2); reads["names"] = ["Q" if x == "W" else x for x in reads["names"]]
    elif mode == "two_channels_invented_fp32_conversions":
        rec["payloads"] = []; rec["scales"] = []; rec["conversions"] = ["fp32", "fp32"]; reads["occurrences"] = []; reads["conversions"] = ["fp32", "fp32"]
    elif mode == "edited_source_rehashed":
        # the bound text is edited to the program the channels claim, and re-hashed consistently; the module text is not
        new = ks["text"].replace("X[m,k]*W[m,k]", "X[m,k]*Q[m,k]"); ks["text"] = new; ks["sha256"] = _hashlib.sha256(new.encode()).hexdigest()
        rec["payloads"][1] = {"name": "Q", "input": 2, "dtype": "nvfp4"}; rec["operands"].pop("W"); rec["operands"]["Q"] = "nvfp4"
        reads["occurrences"][1].update(name="Q", input=2); reads["names"] = ["Q" if x == "W" else x for x in reads["names"]]
    elif mode == "edited_kernel_hash_only":
        ks["sha256"] = "0" * 64
    elif mode == "edited_module_hash_only":
        ks["module_sha256"] = "0" * 64
    elif mode == "no_kernel_source_on_quantised_node":
        n.pop("kernel_source")
    elif mode == "no_module_source":
        graph.pop("dsl_module_source")
    elif mode == "kernel_text_not_in_module":
        graph["dsl_module_source"]["text"] = graph["dsl_module_source"]["text"].replace("X[m,k]*W[m,k]", "X[m,k]*Q[m,k]")
        graph["dsl_module_source"]["sha256"] = _hashlib.sha256(graph["dsl_module_source"]["text"].encode()).hexdigest(); ks["module_sha256"] = graph["dsl_module_source"]["sha256"]
    elif mode == "signature_dtype_edited":
        ks["signature"][1]["dtype"] = "nvfp4"
    _assert_invalid(_reanalyze(graph, tmp_path, "altered"), mode)


def test_round4_consistently_edited_module_and_channels_change_the_program_not_the_proof(tmp_path):
    # editing the SOURCE itself (module text, kernel text, both hashes) to the program the channels describe is a different
    # program: the analyzer then agrees with the channels -- the binding anchors the receipt to the source, nothing more
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "bf16", "Q": "nvfp4"}, "X[m,k]*W[m,k]")
    rec, reads, ks, ms = n["loop_mac_operands"][0], n["loop_mac_reads"][0], n["kernel_source"], graph["dsl_module_source"]
    ms["text"] = ms["text"].replace("X[m,k]*W[m,k]", "X[m,k]*Q[m,k]"); ms["sha256"] = _hashlib.sha256(ms["text"].encode()).hexdigest()
    ks["text"] = ks["text"].replace("X[m,k]*W[m,k]", "X[m,k]*Q[m,k]"); ks["sha256"] = _hashlib.sha256(ks["text"].encode()).hexdigest(); ks["module_sha256"] = ms["sha256"]
    rec["payloads"][1] = {"name": "Q", "input": 2, "dtype": "nvfp4"}; rec["operands"].pop("W"); rec["operands"]["Q"] = "nvfp4"
    reads["occurrences"][1].update(name="Q", input=2); reads["names"] = ["Q" if x == "W" else x for x in reads["names"]]
    out = _reanalyze(graph, tmp_path, "altered"); layer = out["layers"][_kernel_id(out, "contract")]
    assert layer["macs"] == 6 and layer["compute_dtype"] == "nvfp4" and out["total"].get("status") != "incomplete", layer


def test_round4_zero_mac_entry_with_payload_occurrences_is_a_contradiction(tmp_path):
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "nvfp4"}, "X[m,k]*W[m,k]")
    n["loop_other_statements"] += n["loop_mac_statements"]; n["loop_mac_statements"] = 0
    n["loop_mac_operands"][0]["macs"] = 0; n["loop_mac_reads"][0]["macs"] = 0
    assert n["loop_mac_operands"][0]["payloads"] and n["loop_mac_reads"][0]["occurrences"]
    _assert_invalid(_reanalyze(graph, tmp_path, "altered"), "zero_mac_nonempty_payload")


def test_round4_legacy_graph_without_source_binding_keeps_the_workload_precision(tmp_path):
    graph, n = _graph_and_node(tmp_path, {"X": "bf16", "W": "bf16"}, "X[m,k]*W[m,k]")
    for k in ("loop_mac_operands", "loop_mac_reads", "kernel_source"):
        n.pop(k)
    graph.pop("dsl_module_source")
    out = _reanalyze(graph, tmp_path, "legacy"); layer = _kernel_layer(out, "contract")
    assert layer["macs"] == 6 and "status" not in layer and out["total"].get("status") != "incomplete" and "compute_dtype" not in layer


# ---------------------------------------------------------------------------
# 9. round 5 (the reviewer's verdict on an earlier revision): the re-derived source function is bound to the kernel IDENTITY the node
#    carries (the traced kernel name of the node key / converter record), never selected by kernel_source["name"]
#    (DTYPE-KERNEL-IDENTITY-10)
# ---------------------------------------------------------------------------

def _program_with_spare(dtypes, expr, spare_expr, spare_name="spare"):
    """the reviewer's shape: the called `contract` kernel plus an UNUSED same-signature kernel computing *spare_expr*."""
    code = _contract_program(dtypes, expr)
    seg = code[code.index("@kernel"):code.index("\ndef run_model")]
    spare = seg.replace("def contract(", f"def {spare_name}(").replace(expr, spare_expr)
    return code.replace("\ndef run_model", "\n" + spare + "\ndef run_model")


def _bind_node_to(graph, node, code, name):
    from solar.einsum_dsl.ast_parser import kernel_source_binding
    import ast as _a
    fn = next(x for x in _a.walk(_a.parse(code)) if isinstance(x, _a.FunctionDef) and x.name == name)
    seg = _a.get_source_segment(code, fn); b = kernel_source_binding(code, name)
    node["kernel_source"] = {"name": name, "text": seg, "sha256": _hashlib.sha256(seg.encode()).hexdigest(), "module_sha256": _hashlib.sha256(code.encode()).hexdigest(), "signature": b["signature"]}


def _replace_w_by_q(node):
    e, r = node["loop_mac_operands"][0], node["loop_mac_reads"][0]
    e["payloads"][1] = {"name": "Q", "input": 2, "dtype": "nvfp4"}; e["operands"].pop("W"); e["operands"]["Q"] = "nvfp4"
    r["occurrences"][1].update(name="Q", input=2); r["names"] = ["Q" if x == "W" else x for x in r["names"]]


def _run_program(code, inputs, tmp_path, expected):
    from solar.einsum_lib.verify import execute_translated
    outs, _ = execute_translated(code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs, trace=True, fast=False)
    np.testing.assert_allclose(outs["output"], np.full(2, expected), rtol=0, atol=0)
    graph, out = _traced_analysis(code, inputs, tmp_path)
    return graph, out


def test_round5_unused_kernel_cannot_supply_identity_for_the_called_kernel(tmp_path):
    # the reviewer's fixture, as-is semantics: module text / hash, node key `contract#0`, loop expression, declared inputs and
    # class counts unchanged; only kernel_source (spare's genuine source, signature, hash) and the two channels are rewritten
    code = _program_with_spare({"X": "nvfp4", "W": "bf16", "Q": "nvfp4"}, "X[m,k]*W[m,k]", "X[m,k]*Q[m,k]")
    inputs = {"x": np.full((2, 3), 2.0), "w": np.full((2, 3), 3.0), "q": np.full((2, 3), 5.0)}
    graph, out = _run_program(code, inputs, tmp_path, 18.0)
    assert out["total"]["status"] == "incomplete"                                   # the real contraction is mixed
    n = graph["layers"][_node(graph, "contract")]; module_before = copy.deepcopy(graph["dsl_module_source"])
    _bind_node_to(graph, n, code, "spare"); _replace_w_by_q(n)
    assert graph["dsl_module_source"] == module_before and n["kernel_source"]["module_sha256"] == module_before["sha256"]
    _assert_invalid(_reanalyze(graph, tmp_path, "borrowed"), "unused kernel identity")
    # the positive control: the correct binding analyses the mixed contraction exactly as before (holds, compute_dtype_mixed)
    graph2, _ = _run_program(code, inputs, tmp_path / "pos", 18.0)
    out2 = _reanalyze(graph2, tmp_path / "pos", "same"); layer = out2["layers"][_kernel_id(out2, "contract")]
    assert out2["total"]["status"] == "incomplete" and layer["incomplete_reason"] == "compute_dtype_mixed", layer


@pytest.mark.parametrize("mode", ["kernel_source_name_differs_from_node_key", "own_kernel_renamed_body_unchanged", "kernel_function_record_differs"])
def test_round5_kernel_identity_mismatches_refuse(mode, tmp_path):
    code = _program_with_spare({"X": "nvfp4", "W": "nvfp4", "Q": "nvfp4"}, "X[m,k]*W[m,k]", "X[m,k]*W[m,k]")   # spare has the SAME body
    inputs = {"x": np.full((2, 3), 2.0), "w": np.full((2, 3), 3.0), "q": np.full((2, 3), 5.0)}
    graph, out = _run_program(code, inputs, tmp_path, 18.0)
    assert out["total"].get("status") != "incomplete" and out["total"]["macs_by_compute_dtype"] == {"nvfp4": 6}
    n = graph["layers"][_node(graph, "contract")]
    if mode == "kernel_source_name_differs_from_node_key":
        _bind_node_to(graph, n, code, "spare")                                       # a genuine, identical-body kernel -- but not this node's
    elif mode == "own_kernel_renamed_body_unchanged":
        n["kernel_source"]["name"] = "spare"                                         # text / hashes stay contract's
    else:
        n["kernel_function"] = "spare"                                               # the converter's record disagrees with the key
    _assert_invalid(_reanalyze(graph, tmp_path, "altered"), mode)


@pytest.mark.parametrize("same_body", [True, False])
def test_round5_duplicate_kernel_names(same_body, tmp_path):
    # two @kernel defs with the same name: identical bodies pass; differing bodies refuse (the module's `contract` is not
    # provably the traced one), even though Python's last definition is the one that ran
    code = _program_with_spare({"X": "nvfp4", "W": "nvfp4"}, "X[m,k]*W[m,k]", "X[m,k]*W[m,k]" if same_body else "X[m,k]*X[m,k]", spare_name="contract")
    inputs = {"x": np.full((2, 3), 2.0), "w": np.full((2, 3), 3.0)}
    graph, out = _run_program(code, inputs, tmp_path, 18.0 if same_body else 12.0)
    if same_body:
        assert out["total"].get("status") != "incomplete" and out["total"]["macs_by_compute_dtype"] == {"nvfp4": 6}, out["total"]
    else:
        _assert_invalid(out, "duplicate names, different bodies")


def test_round5_converter_records_the_traced_kernel_function(tmp_path):
    graph, n = _graph_and_node(tmp_path, {"X": "nvfp4", "W": "nvfp4"}, "X[m,k]*W[m,k]")
    assert n["kernel_function"] == "contract" == n["kernel_source"]["name"] and _node(graph, "contract").split("#")[0] == "contract"
