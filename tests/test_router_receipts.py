# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend router + strict SOL receipts: implemented-behavior tests.

Replaces the earlier staged (importorskip-gated) acceptance file with real
tests against solar.frontend.router and solar.translate.receipts:

* the router routes mechanical ONLY on positive evidence (bounded supported
  set + complete operand accounting); the captured pre-fix defective graph
  and a round-3-style zero-input graph must never be mechanical;
* strict receipts are emitted from an actual VerifyResult, carry separate
  numerical/structural/accounting/hardware verdicts and actual error stats,
  and strict sol() refuses missing/malformed/failed/stale/mismatched
  receipts BEFORE tracing or analysis;
* stale mechanical artifacts without matching source/capture provenance are
  reprocessed, never reused.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLAR_ROOT))

from solar.frontend.router import (  # noqa: E402
    SUPPORTED_OPS, RouteDecision, decide_from_graph,
)
from solar.translate.receipts import (  # noqa: E402
    ReceiptRefusal, check_strict, emit_receipt, read_receipt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "router"
EXAMPLE = SOLAR_ROOT / "skills" / "verified-translation" / "example"
PY = sys.executable


def _tensor(name, out_shapes, outputs):
    return {"type": name, "node_class": "TensorNode", "input_shapes": [],
            "output_shapes": out_shapes, "connections":
            {"inputs": [], "outputs": outputs}}


def _graph(op_inputs, op_shapes, op="matmul", op_outputs=("Model.out",),
           out_shape=(4, 32, 128), extra_layers=None):
    """An internally consistent graph: every edge resolves to a real node."""
    layers = {}
    for edge, shape in zip(op_inputs, op_shapes):
        layers.setdefault(
            edge, _tensor("auxiliary-tensor", [list(shape)], ["Model.op"]))
    layers["Model.op"] = {
        "type": op, "node_class": "FunctionNode",
        "input_shapes": [list(s) for s in op_shapes],
        "output_shapes": [list(out_shape)],
        "connections": {"inputs": list(op_inputs),
                        "outputs": list(op_outputs)}}
    layers["Model.out"] = {
        "type": "output-tensor", "node_class": "TensorNode",
        "input_shapes": [list(out_shape)], "output_shapes": [],
        "connections": {"inputs": ["Model.op"], "outputs": []}}
    layers.update(extra_layers or {})
    return {"model_name": "m", "layers": layers}


class TestRouter:
    def test_prefix_defect_fixture_never_mechanical(self):
        graph = yaml.safe_load(
            (FIXTURES / "prefix_defect_matmul_graph.yaml").read_text())
        d = decide_from_graph(graph)
        assert d.route != "mechanical"
        assert any("operand accounting incomplete" in r for r in d.reasons)

    def test_zero_input_round3_graph_never_mechanical(self):
        d = decide_from_graph(_graph([], []))
        assert d.route == "agentic"
        assert any("operand accounting incomplete" in r for r in d.reasons)

    def test_single_operand_matmul_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.x"], [[64, 128]]))
        assert d.route == "agentic"

    def test_complete_matmul_routes_mechanical_with_evidence(self):
        d = decide_from_graph(_graph(["Model.w", "Model.x"],
                                     [[4, 32, 64], [64, 128]]))
        assert d.route == "mechanical"
        assert d.evidence["ops"] == {"matmul": 1}
        assert "operand_check" in d.evidence
        assert d.evidence["supported_set"] == sorted(SUPPORTED_OPS)

    def test_unknown_op_routes_agentic_with_reason(self):
        d = decide_from_graph(_graph(["a", "b"], [[2, 2], [2, 2]],
                                     op="custom_fancy_op"))
        assert d.route == "agentic"
        assert any("unknown handler semantics" in r for r in d.reasons)

    def test_empty_graph_unsupported(self):
        d = decide_from_graph({"model_name": "m", "layers": {}})
        assert d.route == "unsupported"
        assert d.reasons

    def test_deterministic(self):
        g = _graph(["a", "b"], [[4, 32, 64], [64, 128]])
        assert decide_from_graph(g) == decide_from_graph(g)

    def test_decision_is_recorded_type(self):
        d = decide_from_graph(_graph([], []))
        assert isinstance(d, RouteDecision)
        assert json.loads(d.to_json())["route"] == d.route


class TestRouterIntegrity:
    """Round-4 P2: positive routing must validate graph schema/connectivity
    and bounded per-op semantics, not only op names and operand counts."""

    def test_valid_real_matmul_graph_routes_mechanical(self):
        graph = yaml.safe_load(
            (FIXTURES / "valid_matmul_graph.yaml").read_text())
        d = decide_from_graph(graph)
        assert d.route == "mechanical", d.reasons

    def test_valid_repeated_operand_scalar_graph_routes_mechanical(self):
        """s = x.sum(); s * s — repeated 0-d operands are valid evidence and
        must stay mechanical (repeated operands preserved, not rejected)."""
        graph = yaml.safe_load(
            (FIXTURES / "valid_scalar_reduction_graph.yaml").read_text())
        d = decide_from_graph(graph)
        assert d.route == "mechanical", d.reasons

    def test_dangling_input_edges_never_mechanical(self):
        g = _graph(["Model.a", "Model.b"], [[4, 8], [8, 2]],
                   out_shape=(4, 2))
        g["layers"]["Model.op"]["connections"]["inputs"] = [
            "missing-a", "missing-b"]
        d = decide_from_graph(g)
        assert d.route != "mechanical"
        assert any("does not resolve" in r for r in d.reasons)

    def test_dangling_output_edge_never_mechanical(self):
        g = _graph(["Model.a", "Model.b"], [[4, 8], [8, 2]],
                   op_outputs=("missing-output",), out_shape=(4, 2))
        d = decide_from_graph(g)
        assert d.route != "mechanical"
        assert any("does not resolve" in r for r in d.reasons)

    def test_edge_shape_count_mismatch_never_mechanical(self):
        g = _graph(["Model.a", "Model.b"], [[4, 8], [8, 2]],
                   out_shape=(4, 2))
        g["layers"]["Model.op"]["connections"]["inputs"] = ["Model.a"]
        d = decide_from_graph(g)
        assert d.route != "mechanical"
        assert any("mismatch" in r for r in d.reasons)

    def test_incompatible_matmul_contraction_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a", "Model.b"], [[4, 8], [7, 2]],
                   out_shape=(4, 2)))
        assert d.route != "mechanical"
        assert any("contraction" in r for r in d.reasons)

    def test_matmul_output_shape_inconsistent_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a", "Model.b"], [[4, 8], [8, 2]],
                   out_shape=(4, 9)))
        assert d.route != "mechanical"

    def test_zero_operand_reshape_never_mechanical(self):
        d = decide_from_graph(
            _graph([], [], op="reshape", out_shape=()))
        assert d.route != "mechanical"
        assert any("reshape" in r for r in d.reasons)

    def test_reshape_element_count_change_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a"], [[4, 8]], op="reshape", out_shape=(5, 8)))
        assert d.route != "mechanical"
        assert any("element" in r for r in d.reasons)

    def test_valid_reshape_routes_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a"], [[4, 8]], op="reshape", out_shape=(2, 16)))
        assert d.route == "mechanical", d.reasons

    def test_broadcast_incompatible_add_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a", "Model.b"], [[4, 8], [3, 8]],
                   op="add", out_shape=(4, 8)))
        assert d.route != "mechanical"
        assert any("broadcast" in r for r in d.reasons)

    def test_elementwise_output_shape_mismatch_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a"], [[4, 8]], op="relu", out_shape=(4, 9)))
        assert d.route != "mechanical"

    def test_malformed_shape_entries_never_mechanical(self):
        d = decide_from_graph(
            _graph(["Model.a", "Model.b"], [[4, 0], [0, 2]],
                   out_shape=(4, 2)))
        assert d.route != "mechanical"
        assert any("malformed" in r for r in d.reasons)


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

DEFN = {
    "name": "rmsnorm_example",
    "axes": {"B": {"type": "const", "value": 8},
             "H": {"type": "const", "value": 128}},
    "inputs": {"hidden_states": {"shape": ["B", "H"], "dtype": "float32"},
               "weight": {"shape": ["H"], "dtype": "float32"}},
}


@pytest.fixture(scope="module")
def verified(tmp_path_factory):
    """One real verify() run on the shipped example + a definition dir."""
    pytest.importorskip("torch")
    from solar.translate import verify
    res = verify(EXAMPLE / "adapter.py", EXAMPLE / "solar_model_einsum_lib.py",
                 source_type="cuda", quiet=True)
    assert res.verified
    problem = tmp_path_factory.mktemp("problem")
    (problem / "definition.json").write_text(json.dumps(DEFN))
    return res, problem


@pytest.fixture(scope="module")
def failed_run(tmp_path_factory):
    """A GENUINE failing verification: the example DSL with zeroed output.

    Round-4 contract: failure receipts come from actual runs too — a
    hand-constructed VerifyResult carries no execution evidence and cannot
    emit a receipt at all.
    """
    pytest.importorskip("torch")
    from solar.translate import verify
    root = tmp_path_factory.mktemp("failed_run")
    source = root / "adapter.py"
    source.write_bytes((EXAMPLE / "adapter.py").read_bytes())
    dsl = root / "solar_model_einsum_lib.py"
    dsl.write_text(
        (EXAMPLE / "solar_model_einsum_lib.py").read_text().replace(
            'return {"output": out.data}',
            'return {"output": out.data * 0}'))
    res = verify(source, dsl, source_type="cuda", quiet=True)
    assert not res.verified
    (root / "definition.json").write_text(json.dumps(DEFN))
    return res, source, dsl, root


def _emit(res, problem, out, **over):
    kw = dict(source_path=EXAMPLE / "adapter.py",
              dsl_path=EXAMPLE / "solar_model_einsum_lib.py",
              definition_path=problem / "definition.json",
              workload_uuid="definition-only:0",
              axes={"B": 8, "H": 128}, dtype="fp16", out_path=out)
    kw.update(over)
    return emit_receipt(res, **kw)


class TestReceipts:
    def test_emit_requires_actual_verify_result(self, tmp_path, verified):
        _, problem = verified
        with pytest.raises(TypeError):
            _emit({"verified": True}, problem, tmp_path / "r.json")

    def test_verdict_kinds_recorded_separately(self, tmp_path, verified):
        res, problem = verified
        r = read_receipt(_emit(res, problem, tmp_path / "r.json"))
        assert r["verdicts"]["numerical"] == "pass"
        for kind in ("structural", "accounting", "hardware"):
            assert r["verdicts"][kind] == "not_certified"
        assert r["error_stats"]["max_abs_error"] is not None

    def test_matching_receipt_admits_strict_sol(self, tmp_path, verified):
        res, problem = verified
        from solar.translate import sol
        receipt = _emit(res, problem, tmp_path / "r.json")
        out = sol(EXAMPLE / "solar_model_einsum_lib.py", problem,
                  tmp_path / "out", arch="B200", precision="fp16",
                  fallback_adapter=EXAMPLE / "adapter.py",
                  receipt=receipt, strict=True,
                  source_path=EXAMPLE / "adapter.py")
        assert out.perf_path.is_file() and out.fused_ms > 0

    def test_failed_verdict_refuses(self, tmp_path, failed_run):
        res, source, dsl, root = failed_run
        receipt = _emit(res, root, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=root / "definition.json")
        assert read_receipt(receipt)["verdicts"]["numerical"] == "fail"
        with pytest.raises(ReceiptRefusal, match="not 'pass'"):
            check_strict(receipt, source_path=source, dsl_path=dsl,
                         definition_path=root / "definition.json",
                         workload_uuid="definition-only:0")

    def test_checker_error_shape_only_refuses(self, tmp_path, verified):
        from solar.translate import verify
        _, problem = verified
        res = verify(EXAMPLE / "adapter.py",
                     EXAMPLE / "solar_model_einsum_lib.py",
                     source_type="cuda", quiet=True, shape_only=True)
        assert res.shape_only
        receipt = _emit(res, problem, tmp_path / "r.json")
        assert read_receipt(receipt)["verdicts"]["numerical"] == "error"
        with pytest.raises(ReceiptRefusal):
            check_strict(receipt, source_path=EXAMPLE / "adapter.py",
                         dsl_path=EXAMPLE / "solar_model_einsum_lib.py",
                         definition_path=problem / "definition.json",
                         workload_uuid="definition-only:0")

    @pytest.mark.parametrize("mutate", ["source", "dsl", "definition",
                                        "uuid", "axes", "dtype"])
    def test_every_identity_field_is_enforced(self, tmp_path, verified,
                                              mutate):
        res, problem = verified
        receipt = _emit(res, problem, tmp_path / "r.json")
        kw = dict(source_path=EXAMPLE / "adapter.py",
                  dsl_path=EXAMPLE / "solar_model_einsum_lib.py",
                  definition_path=problem / "definition.json",
                  workload_uuid="definition-only:0",
                  axes={"B": 8, "H": 128}, dtype="fp16")
        if mutate in ("source", "dsl", "definition"):
            other = tmp_path / f"other_{mutate}.bin"
            other.write_text("changed bytes")
            kw[{"source": "source_path", "dsl": "dsl_path",
                "definition": "definition_path"}[mutate]] = other
            expect = "stale/mismatched"
        elif mutate == "uuid":
            kw["workload_uuid"] = "definition-only:7"
            expect = "selected"
        elif mutate == "axes":
            kw["axes"] = {"B": 16, "H": 128}
            expect = "axes differ"
        else:
            kw["dtype"] = "fp8"
            expect = "dtype"
        with pytest.raises(ReceiptRefusal, match=expect):
            check_strict(receipt, **kw)

    @pytest.mark.parametrize("breakage", ["missing", "malformed",
                                          "no_tolerance", "no_checker"])
    def test_broken_receipts_refuse(self, tmp_path, verified, breakage):
        res, problem = verified
        path = tmp_path / "r.json"
        if breakage == "missing":
            pass  # never write it
        elif breakage == "malformed":
            path.write_text("{not json")
        else:
            _emit(res, problem, path)
            doc = json.loads(path.read_text())
            if breakage == "no_tolerance":
                doc["tolerance"] = {}
            else:
                doc["checker"] = {"identity": ""}
            path.write_text(json.dumps(doc))
        with pytest.raises(ReceiptRefusal):
            read_receipt(path) if breakage != "missing" else read_receipt(path)

    def test_strict_sol_refuses_before_tracing(self, tmp_path, failed_run):
        """Refusal happens BEFORE any pipeline stage: no outputs appear."""
        from solar.translate import sol
        res, source, dsl, root = failed_run
        receipt = _emit(res, root, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=root / "definition.json")
        out_dir = tmp_path / "out"
        with pytest.raises(ReceiptRefusal):
            sol(dsl, root, out_dir, receipt=receipt, strict=True,
                source_path=source)
        assert not (out_dir / "dsl_full").exists()

    def test_strict_sol_requires_receipt_and_source(self, tmp_path, verified):
        from solar.translate import sol
        _, problem = verified
        with pytest.raises(ReceiptRefusal, match="requires a receipt"):
            sol(EXAMPLE / "solar_model_einsum_lib.py", problem,
                tmp_path / "o1", strict=True,
                source_path=EXAMPLE / "adapter.py")
        with pytest.raises(ReceiptRefusal, match="source_path"):
            sol(EXAMPLE / "solar_model_einsum_lib.py", problem,
                tmp_path / "o2", strict=True, receipt=tmp_path / "r.json")


class TestRouterShapeSemantics:
    """Round-5 P2: full supported matmul batch/vector semantics, explicit
    linear orientation/output/bias contract, and reduction output shapes."""

    # -- adversarial (reviewer's probe cases, fresh constructions) --
    def test_nonbroadcast_matmul_batches_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.a", "Model.b"],
                                     [[2, 4, 8], [3, 8, 5]],
                                     out_shape=(2, 4, 5)))
        assert d.route != "mechanical"
        assert any("matmul" in r for r in d.reasons)

    def test_matmul_wrong_batch_output_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.a", "Model.b"],
                                     [[2, 4, 8], [2, 8, 5]],
                                     out_shape=(3, 4, 5)))
        assert d.route != "mechanical"

    def test_linear_wrong_output_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.x", "Model.w"],
                                     [[4, 8], [5, 8]], op="linear",
                                     out_shape=(4, 999)))
        assert d.route != "mechanical"
        assert any("linear" in r for r in d.reasons)

    def test_linear_wrong_orientation_never_mechanical(self):
        # weight contract is (out_features, in_features): (8, 5) cannot
        # consume an input whose feature dim is 8
        d = decide_from_graph(_graph(["Model.x", "Model.w"],
                                     [[4, 8], [8, 5]], op="linear",
                                     out_shape=(4, 5)))
        assert d.route != "mechanical"

    def test_linear_bad_bias_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.x", "Model.w", "Model.c"],
                                     [[4, 8], [5, 8], [7]], op="linear",
                                     out_shape=(4, 5)))
        assert d.route != "mechanical"

    def test_sum_grows_output_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.a"], [[4, 8]], op="sum",
                                     out_shape=(4, 9)))
        assert d.route != "mechanical"
        assert any("sum" in r for r in d.reasons)

    def test_sum_rank_growth_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.a"], [[4, 8]], op="sum",
                                     out_shape=(4, 8, 2)))
        assert d.route != "mechanical"

    def test_vector_matmul_mismatch_never_mechanical(self):
        d = decide_from_graph(_graph(["Model.a", "Model.b"], [[8], [9]],
                                     out_shape=()))
        assert d.route != "mechanical"

    # -- valid positives: supported forms must stay mechanical --
    @pytest.mark.parametrize("a,b,out", [
        ([2, 4, 8], [2, 8, 5], (2, 4, 5)),    # equal batch
        ([1, 4, 8], [2, 8, 5], (2, 4, 5)),    # broadcast batch
        ([4, 8], [8], (4,)),                  # mat @ vec
        ([8], [8, 5], (5,)),                  # vec @ mat
        ([8], [8], ()),                       # vec @ vec -> scalar
    ])
    def test_valid_matmul_forms_route_mechanical(self, a, b, out):
        d = decide_from_graph(_graph(["Model.a", "Model.b"], [a, b],
                                     out_shape=out))
        assert d.route == "mechanical", d.reasons

    def test_valid_linear_with_bias_routes_mechanical(self):
        d = decide_from_graph(_graph(["Model.x", "Model.w", "Model.c"],
                                     [[4, 8], [5, 8], [5]], op="linear",
                                     out_shape=(4, 5)))
        assert d.route == "mechanical", d.reasons

    @pytest.mark.parametrize("out", [(), (8,), (4,), (1, 8), (4, 1)])
    def test_valid_sum_reductions_route_mechanical(self, out):
        d = decide_from_graph(_graph(["Model.a"], [[4, 8]], op="sum",
                                     out_shape=out))
        assert d.route == "mechanical", d.reasons


def _typed_linear_graph(shapes, types, out=(4, 5)):
    g = _graph([f"Model.t{i}" for i in range(len(shapes))], shapes,
               op="linear", out_shape=out)
    g["layers"]["Model.op"]["input_types"] = list(types)
    return g


class TestLinearOperandRoles:
    """Round-6 P2: every recorded linear operand must be used exactly once
    under an unambiguous supported signature; reordering must never drop
    operands or raise."""

    def test_four_operand_linear_never_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[4, 8], [5, 8], [5], [999]],
            ["activation", "weight", "bias", "activation"]))
        assert d.route != "mechanical"
        assert any("linear" in r for r in d.reasons)

    def test_untagged_extra_operand_never_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[4, 8], [5, 8], [999]],
            ["activation", "weight", "activation"]))
        assert d.route != "mechanical"

    def test_weight_bias_without_activation_routes_agentic_no_crash(self):
        d = decide_from_graph(_typed_linear_graph(
            [[5, 8], [5]], ["weight", "bias"]))
        assert d.route != "mechanical"
        assert d.reasons

    def test_duplicate_weight_tags_never_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[4, 8], [5, 8], [5, 8]], ["activation", "weight", "weight"]))
        assert d.route != "mechanical"

    def test_role_tags_not_covering_operands_never_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[4, 8], [5, 8], [5]], ["activation", "weight"]))
        assert d.route != "mechanical"

    def test_valid_typed_linear_with_bias_routes_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[4, 8], [5, 8], [5]], ["activation", "weight", "bias"]))
        assert d.route == "mechanical", d.reasons

    def test_valid_typed_weight_first_routes_mechanical(self):
        d = decide_from_graph(_typed_linear_graph(
            [[5, 8], [4, 8]], ["weight", "activation"]))
        assert d.route == "mechanical", d.reasons


# Adapter mirroring the DOCUMENTED external-reference pattern (the example
# adapter's own docstring: "In real adapters, read it from the benchmark
# checkout instead") — get_source() reads a sibling reference.py.
_EXT_ADAPTER = """\
from pathlib import Path
import numpy as np

_REF = Path(__file__).parent / "reference.py"
B, H = 8, 128


def get_source():
    return _REF.read_text()


def get_inputs():
    rng = np.random.RandomState(0)
    return {"hidden_states": rng.standard_normal((B, H)).astype(np.float32),
            "weight": (1.0 + 0.1 * rng.standard_normal(H)).astype(np.float32)}


def kernel_fn(inputs):
    import torch
    ns = {}
    exec(get_source(), ns)
    t = {k: torch.from_numpy(np.ascontiguousarray(v))
         for k, v in inputs.items()}
    return {"output": ns["run"](t["hidden_states"], t["weight"]).numpy()}
"""

_ZERO_WRAP = ("\n_orig_run = run\n"
              "def run(hidden_states, weight):\n"
              "    return _orig_run(hidden_states, weight) * 0\n")


def _external_reference_root(tmp_path):
    """Fresh root with the external-reference adapter; returns a passing
    VerifyResult plus the paths (reference.py is the mutable dependency)."""
    pytest.importorskip("torch")
    import importlib.util
    from solar.translate import verify
    spec = importlib.util.spec_from_file_location(
        "_example_adapter_for_ref", EXAMPLE / "adapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = tmp_path / "reference.py"
    ref.write_text(mod.get_source())
    source = tmp_path / "adapter.py"
    source.write_text(_EXT_ADAPTER)
    dsl = tmp_path / "solar_model_einsum_lib.py"
    dsl.write_bytes((EXAMPLE / "solar_model_einsum_lib.py").read_bytes())
    (tmp_path / "definition.json").write_text(json.dumps(DEFN))
    res = verify(source, dsl, source_type="cuda", quiet=True)
    assert res.verified
    return res, source, dsl, ref


class TestReferenceBinding:
    """Round-5 P1: the receipt must bind the reference text the verification
    actually prepared/executed, not only the adapter/DSL bytes — a changed
    external reference.py must defeat emission AND strict entry."""

    def test_reference_hash_recorded_in_receipt(self, tmp_path, verified):
        res, problem = verified
        r = read_receipt(_emit(res, problem, tmp_path / "r.json"))
        assert len(r["hashes"]["reference_sha256"]) == 64

    def test_reference_change_refused_at_emit(self, tmp_path):
        res, source, dsl, ref = _external_reference_root(tmp_path)
        ref.write_text(ref.read_text() + _ZERO_WRAP)
        with pytest.raises(ReceiptRefusal, match="reference"):
            _emit(res, tmp_path, tmp_path / "r.json",
                  source_path=source, dsl_path=dsl,
                  definition_path=tmp_path / "definition.json")
        assert not (tmp_path / "r.json").exists()

    def test_reference_change_refused_at_strict_entry(self, tmp_path):
        from solar.translate import sol
        res, source, dsl, ref = _external_reference_root(tmp_path)
        receipt = _emit(res, tmp_path, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=tmp_path / "definition.json")
        ref.write_text(ref.read_text() + _ZERO_WRAP)
        with pytest.raises(ReceiptRefusal, match="reference"):
            check_strict(receipt, source_path=source, dsl_path=dsl,
                         definition_path=tmp_path / "definition.json",
                         workload_uuid="definition-only:0")
        out_dir = tmp_path / "strict-sol"
        with pytest.raises(ReceiptRefusal):
            sol(dsl, tmp_path, out_dir, precision="fp16",
                fallback_adapter=source, receipt=receipt, strict=True,
                source_path=source)
        assert not (out_dir / "dsl_full").exists()

    def test_unprovable_reference_provenance_refused(self, tmp_path):
        from solar.translate.receipts import _current_reference_text
        no_get_source = tmp_path / "adapter.py"
        no_get_source.write_text("def get_inputs():\n    return {}\n")
        with pytest.raises(ReceiptRefusal, match="provenance"):
            _current_reference_text(no_get_source, "cuda")


_HELPER_ADAPTER = """\
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
{extra_path}
import numpy as np

_REFERENCE_SRC = "from {helper} import run\\n"
B, H = 8, 128


def get_source():
    return _REFERENCE_SRC


def get_inputs():
    rng = np.random.RandomState(0)
    return {{"hidden_states": rng.standard_normal((B, H)).astype(np.float32),
             "weight": (1.0 + 0.1 * rng.standard_normal(H)).astype(np.float32)}}


def kernel_fn(inputs):
    import torch
    ns = {{}}
    exec(get_source(), ns)
    t = {{k: torch.from_numpy(np.ascontiguousarray(v))
          for k, v in inputs.items()}}
    return {{"output": ns["run"](t["hidden_states"], t["weight"]).numpy()}}
"""


def _reference_run_text():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_example_adapter_for_helper", EXAMPLE / "adapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_source()


def _helper_root(tmp_path, helper_name, helper_dir=None, padding=""):
    """Adapter whose REFERENCE imports a local helper module (the reviewer's
    round-6 pattern): adapter bytes and reference text never change while the
    helper does.  Returns (result, source, dsl, helper_path)."""
    pytest.importorskip("torch")
    import sys as _sys
    from solar.translate import verify
    helper_dir = helper_dir or tmp_path
    helper = helper_dir / f"{helper_name}.py"
    helper.write_text(_reference_run_text() + "\n" + padding)
    extra = ""
    if helper_dir != tmp_path:
        extra = f"sys.path.insert(0, {str(helper_dir)!r})"
    source = tmp_path / "adapter.py"
    source.write_text(_HELPER_ADAPTER.format(helper=helper_name,
                                             extra_path=extra))
    dsl = tmp_path / "solar_model_einsum_lib.py"
    dsl.write_bytes((EXAMPLE / "solar_model_einsum_lib.py").read_bytes())
    (tmp_path / "definition.json").write_text(json.dumps(DEFN))
    try:
        res = verify(source, dsl, source_type="cuda", quiet=True)
    finally:
        _sys.modules.pop(helper_name, None)
    assert res.verified
    return res, source, dsl, helper


class TestDependencyBinding:
    """Round-6 P1: the reference's local execution dependencies are bound at
    verification; changed helper bytes defeat emission and strict entry, and
    an unprovable dependency closure refuses strict qualification."""

    def test_local_helper_dependency_recorded_in_receipt(self, tmp_path):
        res, source, dsl, helper = _helper_root(tmp_path, "helper_rt6rec")
        receipt = _emit(res, tmp_path, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=tmp_path / "definition.json")
        deps = read_receipt(receipt)["hashes"]["dependencies"]
        assert str(helper.resolve()) in deps
        assert len(deps[str(helper.resolve())]) == 64

    def test_helper_change_refused_at_emit(self, tmp_path):
        res, source, dsl, helper = _helper_root(tmp_path, "helper_rt6emit")
        helper.write_text(helper.read_text() + _ZERO_WRAP)
        with pytest.raises(ReceiptRefusal, match="dependency"):
            _emit(res, tmp_path, tmp_path / "r.json",
                  source_path=source, dsl_path=dsl,
                  definition_path=tmp_path / "definition.json")
        assert not (tmp_path / "r.json").exists()

    def test_helper_change_refused_at_strict_entry(self, tmp_path):
        from solar.translate import sol
        res, source, dsl, helper = _helper_root(tmp_path, "helper_rt6strict")
        receipt = _emit(res, tmp_path, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=tmp_path / "definition.json")
        helper.write_text(helper.read_text() + _ZERO_WRAP)
        with pytest.raises(ReceiptRefusal, match="dependency"):
            check_strict(receipt, source_path=source, dsl_path=dsl,
                         definition_path=tmp_path / "definition.json",
                         workload_uuid="definition-only:0")
        out_dir = tmp_path / "strict-sol"
        with pytest.raises(ReceiptRefusal):
            sol(dsl, tmp_path, out_dir, precision="fp16",
                fallback_adapter=source, receipt=receipt, strict=True,
                source_path=source)
        assert not (out_dir / "dsl_full").exists()

    def test_helper_deleted_refused_at_strict_entry(self, tmp_path):
        res, source, dsl, helper = _helper_root(tmp_path, "helper_rt6del")
        receipt = _emit(res, tmp_path, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=tmp_path / "definition.json")
        helper.unlink()
        with pytest.raises(ReceiptRefusal, match="dependency"):
            check_strict(receipt, source_path=source, dsl_path=dsl,
                         definition_path=tmp_path / "definition.json",
                         workload_uuid="definition-only:0")

    def test_outside_project_dependency_refuses_strict_qualification(
            self, tmp_path_factory):
        """A reference importing a module OUTSIDE the adapter directory and
        outside the environment/checkout cannot prove its closure: strict
        qualification is refused rather than presumed."""
        root = tmp_path_factory.mktemp("root")
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        res, source, dsl, _ = _helper_root(root, "helper_rt6ext",
                                           helper_dir=elsewhere)
        with pytest.raises(ReceiptRefusal, match="provenance"):
            _emit(res, root, root / "r.json",
                  source_path=source, dsl_path=dsl,
                  definition_path=root / "definition.json")


class TestCachedDependencyExecution:
    """Round-7 P1: repeated same-process verification must execute the
    CURRENT helper bytes, never cached old code bound to new hashes."""

    def test_repeated_verification_executes_current_helper_bytes(
            self, tmp_path):
        """The reviewer's exact reproduction: edit the helper, verify again
        in the SAME process — the run must fail like a fresh process does,
        not pass on cached old code while recording the new hash."""
        from solar.translate import verify
        res1, source, dsl, helper = _helper_root(tmp_path, "helper_rt7cache")
        assert res1.verified
        # simulate the import cache of a long-lived process: the helper
        # module object is still registered when the second verify starts
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "helper_rt7cache", helper)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["helper_rt7cache"] = mod
        try:
            helper.write_text(helper.read_text() + _ZERO_WRAP)
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt7cache", None)
        assert not res2.verified, (
            "second same-process verification passed on cached old helper "
            "code while the helper file had changed")
        assert res2.max_abs_error is not None and res2.max_abs_error > 1.0

    def test_repeated_verification_rebinds_fresh_benign_bytes(self, tmp_path):
        """Freshness, not blanket refusal: a benign helper edit re-executes
        and re-binds the new bytes, and the receipt carries the new hash."""
        import hashlib as _hashlib
        from solar.translate import verify
        res1, source, dsl, helper = _helper_root(tmp_path, "helper_rt7benign")
        helper.write_text("# benign comment\n" + helper.read_text())
        try:
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt7benign", None)
        assert res2.verified
        deps = dict(res2.execution.local_dependencies)
        assert deps[str(helper.resolve())] == _hashlib.sha256(
            helper.read_bytes()).hexdigest()
        receipt = _emit(res2, tmp_path, tmp_path / "r.json",
                        source_path=source, dsl_path=dsl,
                        definition_path=tmp_path / "definition.json")
        assert read_receipt(receipt)["verdicts"]["numerical"] == "pass"

    def test_eviction_recorded_in_execution_evidence(self, tmp_path):
        """A cached unproven module that was dropped before the run is
        visible in the evidence, so the snapshot behavior is auditable."""
        from solar.translate import verify
        res1, source, dsl, helper = _helper_root(tmp_path, "helper_rt7evict")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "helper_rt7evict", helper)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["helper_rt7evict"] = mod
        try:
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt7evict", None)
        assert res2.verified
        assert "helper_rt7evict" in [
            n.split(" ")[0] for n in res2.execution.evicted_cached_modules]


class TestStaleBytecodeExecution:
    """Round-8 P1: a timestamp-valid .pyc must not let stale bytecode
    execute while the changed source bytes get bound to the pass."""

    def test_timestamp_valid_stale_bytecode_does_not_mask_source_change(
            self, tmp_path):
        """The reviewer's exact reproduction: precompile the good helper,
        verify, then change the source at EQUAL byte length with preserved
        mtime (so the .pyc stays timestamp-valid) — the second verification
        must execute the new source and fail, like a cache-free process."""
        import os
        import py_compile
        from solar.translate import verify
        res1, source, dsl, helper = _helper_root(tmp_path, "helper_rt8pyc",
                                                 padding="# padding XXXX\n")
        py_compile.compile(str(helper), doraise=True)
        before = helper.stat()
        code = helper.read_text()
        changed = code.replace(
            "return y.to(hidden_states.dtype)",
            "return 0*y.to(hidden_states.dtype)").replace(
            "# padding XXXX", "# padding XX")
        assert len(changed.encode()) == len(code.encode())
        helper.write_text(changed)
        os.utime(helper, ns=(before.st_atime_ns, before.st_mtime_ns))
        try:
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt8pyc", None)
        assert not res2.verified, (
            "second verification passed on timestamp-valid stale bytecode "
            "while the helper source had changed")
        assert res2.max_abs_error is not None and res2.max_abs_error > 1.0

    def test_executed_bytes_are_what_the_receipt_binds(self, tmp_path):
        """The dependency hash must be of the bytes the controlled loader
        actually compiled and executed, which equal the on-disk source."""
        import hashlib as _hashlib
        import py_compile
        res, source, dsl, helper = _helper_root(tmp_path, "helper_rt8bind")
        # a pre-existing .pyc must not matter: the recorded hash is still
        # the source bytes that were compiled and executed
        py_compile.compile(str(helper), doraise=True)
        from solar.translate import verify
        try:
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt8bind", None)
        assert res2.verified
        deps = dict(res2.execution.local_dependencies)
        assert deps[str(helper.resolve())] == _hashlib.sha256(
            helper.read_bytes()).hexdigest()

    def test_sourceless_bytecode_dependency_refuses_strict(self, tmp_path):
        """A dependency available only as compiled bytecode cannot be tied
        to audited source bytes: strict qualification must refuse."""
        import py_compile
        from solar.translate import verify
        res1, source, dsl, helper = _helper_root(tmp_path, "helper_rt8src")
        # replace the .py helper with a legacy sourceless .pyc next to it
        pyc = tmp_path / "helper_rt8src.pyc"
        py_compile.compile(str(helper), cfile=str(pyc), doraise=True)
        helper.unlink()
        try:
            res2 = verify(source, dsl, source_type="cuda", quiet=True)
        finally:
            sys.modules.pop("helper_rt8src", None)
        assert res2.verified  # bytecode still runs the good code numerically
        with pytest.raises(ReceiptRefusal, match="provenance|tied"):
            _emit(res2, tmp_path, tmp_path / "r.json",
                  source_path=source, dsl_path=dsl,
                  definition_path=tmp_path / "definition.json")


def _adapter_root(tmp_path, padding="\n# padding XXXX\n"):
    """A fresh root with its own copy of the self-contained example adapter
    (padded so an equal-byte-length edit is possible) and DSL."""
    source = tmp_path / "adapter.py"
    source.write_text((EXAMPLE / "adapter.py").read_text() + padding)
    dsl = tmp_path / "solar_model_einsum_lib.py"
    dsl.write_bytes((EXAMPLE / "solar_model_einsum_lib.py").read_bytes())
    (tmp_path / "definition.json").write_text(json.dumps(DEFN))
    return source, dsl


def _stale_adapter_edit(source):
    """Precompile a .pyc, then zero the adapter's embedded reference at
    EQUAL byte length with preserved mtime — the stale bytecode stays
    timestamp-valid for the changed source."""
    import os
    import py_compile
    py_compile.compile(str(source), doraise=True)
    before = source.stat()
    code = source.read_text()
    changed = code.replace(
        "return y.to(hidden_states.dtype)",
        "return 0*y.to(hidden_states.dtype)").replace(
        "# padding XXXX", "# padding XX")
    assert changed != code and len(changed.encode()) == len(code.encode())
    source.write_text(changed)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))


class TestAdapterStaleBytecode:
    """Round-9 P1: the MAIN adapter was loaded via a direct
    spec_from_file_location(...).exec_module(), bypassing the audited
    meta-path finder — a timestamp-valid stale .pyc of the adapter itself
    executed old code while the changed source bytes were hashed."""

    def test_repeated_verify_executes_current_adapter_source(self, tmp_path):
        """The reviewer's exact reproduction: verify PASS, precompile,
        zero the embedded reference at equal byte length with preserved
        mtime — re-verification must execute the new adapter source and
        fail like a cache-free process."""
        from solar.translate import verify
        source, dsl = _adapter_root(tmp_path)
        res1 = verify(source, dsl, source_type="cuda", quiet=True)
        assert res1.verified
        _stale_adapter_edit(source)
        res2 = verify(source, dsl, source_type="cuda", quiet=True)
        assert not res2.verified, (
            "re-verification passed on the adapter's timestamp-valid stale "
            "bytecode while its source had changed")
        assert res2.max_abs_error is not None and res2.max_abs_error > 1.0

    def test_initial_verify_with_stale_pyc_executes_source(self, tmp_path):
        """Even the FIRST verification in a process must compile the current
        adapter source bytes, and the bound source hash must be of the bytes
        that executed."""
        import hashlib as _hashlib
        from solar.translate import verify
        source, dsl = _adapter_root(tmp_path)
        _stale_adapter_edit(source)
        res = verify(source, dsl, source_type="cuda", quiet=True)
        assert not res.verified
        assert res.max_abs_error is not None and res.max_abs_error > 1.0
        assert res.execution.source_sha256 == _hashlib.sha256(
            source.read_bytes()).hexdigest()
        assert res.execution.unsupported_dependencies == ()

    def test_fresh_process_with_stale_pyc_fails(self, tmp_path):
        """A fresh process must fail with the stale .pyc still present —
        the loader, not process state, is what defeats the cache."""
        source, dsl = _adapter_root(tmp_path)
        _stale_adapter_edit(source)
        code = (
            "import json, sys\n"
            "from solar.translate import verify\n"
            "r = verify(sys.argv[1], sys.argv[2], source_type='cuda', "
            "quiet=True)\n"
            "print('RES=' + json.dumps({'v': r.verified, "
            "'e': r.max_abs_error}))\n")
        proc = subprocess.run(
            [PY, "-c", code, str(source), str(dsl)],
            capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=600)
        assert proc.returncode == 0, proc.stderr
        out = json.loads(next(
            line[len("RES="):] for line in proc.stdout.splitlines()
            if line.startswith("RES=")))
        assert not out["v"] and out["e"] > 1.0

    def test_reference_rederivation_reads_current_source_bytes(
            self, tmp_path):
        """Receipt emission/strict entry re-derive the reference by
        importing the adapter: that import must compile the current source
        bytes, never a timestamp-valid stale .pyc."""
        import os
        import py_compile
        from solar.translate.receipts import _current_reference_text
        source, _ = _adapter_root(tmp_path)
        old_text = _current_reference_text(source, "cuda")
        py_compile.compile(str(source), doraise=True)
        before = source.stat()
        code = source.read_text()
        changed = code.replace("EPS = 1e-6", "EPS = 2e-6")  # equal length
        assert changed != code and len(changed.encode()) == len(code.encode())
        source.write_text(changed)
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert _current_reference_text(source, "cuda") != old_text, (
            "stale adapter bytecode resurrected the old reference text")

    def test_unaudited_adapter_load_refuses_strict(self, tmp_path,
                                                   monkeypatch):
        """Fail-closed backstop: a load path that skips the audited loader
        cannot tie the executed adapter code to the hashed source bytes, so
        strict qualification must refuse even on a numerical pass."""
        import importlib.util
        from solar.einsum_lib import cuda_frontend
        from solar.translate import verify

        def raw_load(file_path):
            path = Path(file_path).resolve()
            spec = importlib.util.spec_from_file_location(
                "_cuda_module", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        monkeypatch.setattr(cuda_frontend, "_load_module", raw_load)
        source, dsl = _adapter_root(tmp_path)
        res = verify(source, dsl, source_type="cuda", quiet=True)
        assert res.verified  # numerics are fine; the CLOSURE is unproven
        assert any("cannot be tied" in u
                   for u in res.execution.unsupported_dependencies)
        with pytest.raises(ReceiptRefusal, match="provenance"):
            _emit(res, tmp_path, tmp_path / "r.json",
                  source_path=source, dsl_path=dsl)

    @pytest.mark.parametrize("loader_name", [
        "cuda", "triton", "pytorch", "jax"])
    def test_supported_source_direct_loaders_bypass_bytecode(
            self, tmp_path, loader_name):
        """Every supported source type's direct module loader must compile
        the current source bytes, not a timestamp-valid stale .pyc."""
        import os
        import py_compile
        mod_file = tmp_path / f"m_rt9_{loader_name}.py"
        mod_file.write_text("VALUE = 1\n")
        py_compile.compile(str(mod_file), doraise=True)
        before = mod_file.stat()
        mod_file.write_text("VALUE = 2\n")  # equal byte length
        os.utime(mod_file, ns=(before.st_atime_ns, before.st_mtime_ns))
        if loader_name == "cuda":
            from solar.einsum_lib.cuda_frontend import _load_module as load
        elif loader_name == "triton":
            from solar.einsum_lib.triton_frontend import _load_module as load
        elif loader_name == "pytorch":
            from solar.einsum_lib.pipeline import load_pytorch_module as load
        else:
            pytest.importorskip("jax")
            from solar.einsum_lib.jax_frontend import load_jax_module as load
        try:
            mod = load(str(mod_file))
        finally:
            sys.modules.pop("_jax_model_module", None)
        assert mod.VALUE == 2, (
            f"{loader_name} loader executed stale bytecode")


class TestReceiptBinding:
    """Round-4 P1: hashes/tolerance/checker/options are captured INSIDE the
    verification execution; a passing VerifyResult can never be rebound to
    bytes other than the ones it actually verified."""

    def test_rebound_receipt_after_dsl_change_refused(self, tmp_path):
        """The reviewer's exact reproduction, in a fresh root: verify PASS,
        then change the DSL to return zeros — emitting a receipt against the
        changed bytes must refuse, not hash-and-bless them."""
        pytest.importorskip("torch")
        from solar.translate import verify
        source = tmp_path / "adapter.py"
        source.write_bytes((EXAMPLE / "adapter.py").read_bytes())
        dsl = tmp_path / "solar_model_einsum_lib.py"
        dsl.write_bytes((EXAMPLE / "solar_model_einsum_lib.py").read_bytes())
        (tmp_path / "definition.json").write_text(json.dumps(DEFN))
        res = verify(source, dsl, source_type="cuda", quiet=True)
        assert res.verified
        dsl.write_text(dsl.read_text().replace(
            'return {"output": out.data}',
            'return {"output": out.data * 0}'))
        with pytest.raises(ReceiptRefusal, match="changed since verification"):
            _emit(res, tmp_path, tmp_path / "r.json",
                  source_path=source, dsl_path=dsl,
                  definition_path=tmp_path / "definition.json")

    def test_source_swap_at_emit_refused(self, tmp_path, verified):
        res, problem = verified
        other = tmp_path / "other_source.py"
        other.write_text("# not the verified source\n")
        with pytest.raises(ReceiptRefusal, match="changed since verification"):
            _emit(res, problem, tmp_path / "r.json", source_path=other)

    def test_handcrafted_result_has_no_evidence_and_cannot_emit(
            self, tmp_path, verified):
        from solar.translate._runner import VerifyResult
        _, problem = verified
        forged = VerifyResult(verified=True, max_abs_error=0.0,
                              message="hand-typed")
        with pytest.raises(TypeError, match="execution evidence"):
            _emit(forged, problem, tmp_path / "r.json")

    def test_execution_evidence_not_constructible_by_callers(self):
        from solar.translate._runner import ExecutionEvidence
        with pytest.raises(TypeError):
            ExecutionEvidence()

    def test_tolerance_checker_options_recorded_from_execution(
            self, tmp_path, verified):
        res, problem = verified
        r = read_receipt(_emit(res, problem, tmp_path / "r.json"))
        assert r["tolerance"]["rtol"] == 1e-4
        assert r["tolerance"]["atol"] == 1e-5
        assert "min_tol" in r["tolerance"]
        assert r["checker"]["identity"]
        assert len(r["checker"]["sha256"]) == 64
        ex = r["execution"]
        assert ex["source_type"] == "cuda"
        assert ex["validation_shapes"]  # the shapes actually executed
        assert ex["verification_dtype"] in ("fp64", "fp32")

    def test_strict_sol_enforces_selected_precision(self, tmp_path, verified):
        from solar.translate import sol
        res, problem = verified
        receipt = _emit(res, problem, tmp_path / "r.json")  # dtype fp16
        with pytest.raises(ReceiptRefusal, match="dtype"):
            sol(EXAMPLE / "solar_model_einsum_lib.py", problem,
                tmp_path / "out", precision="fp8",
                fallback_adapter=EXAMPLE / "adapter.py",
                receipt=receipt, strict=True,
                source_path=EXAMPLE / "adapter.py")
        assert not (tmp_path / "out" / "dsl_full").exists()

    def test_pre_binding_schema_receipt_refused(self, tmp_path, verified):
        res, problem = verified
        path = _emit(res, problem, tmp_path / "r.json")
        doc = json.loads(path.read_text())
        doc["schema"] = "solar-receipt-v1"
        path.write_text(json.dumps(doc))
        with pytest.raises(ReceiptRefusal, match="schema"):
            read_receipt(path)

    def test_expected_tolerance_and_checker_identity_compared(
            self, tmp_path, verified):
        res, problem = verified
        receipt = _emit(res, problem, tmp_path / "r.json")
        kw = dict(source_path=EXAMPLE / "adapter.py",
                  dsl_path=EXAMPLE / "solar_model_einsum_lib.py",
                  definition_path=problem / "definition.json",
                  workload_uuid="definition-only:0")
        with pytest.raises(ReceiptRefusal, match="tolerance"):
            check_strict(receipt, expected_tolerance={"rtol": 1e-2}, **kw)
        with pytest.raises(ReceiptRefusal, match="checker"):
            check_strict(receipt, expected_checker="someone.else", **kw)


# ---------------------------------------------------------------------------
# stale mechanical artifact reuse
# ---------------------------------------------------------------------------

SIMPLE_MODEL = """
import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a @ b
def get_inputs():
    return [torch.randn(4, 8), torch.randn(8, 2)]
def get_init_inputs():
    return []
"""


def _stage1(tmp_path, src=SIMPLE_MODEL):
    model = tmp_path / "model.py"
    model.write_text(src)
    out = tmp_path / "graph"
    proc = subprocess.run(
        [PY, "-m", "solar.cli.process_model", "--model-file", str(model),
         "--output-dir", str(out)],
        capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
    return proc, out, model


class TestStaleArtifactClosure:
    def test_provenance_recorded_and_resume_honored(self, tmp_path):
        p1, out, _ = _stage1(tmp_path)
        assert p1.returncode == 0, p1.stdout[-800:]
        doc = yaml.safe_load((out / "pytorch_graph.yaml").read_text())
        prov = doc["provenance"]
        assert len(prov["source_sha256"]) == 64
        assert len(prov["capture_rev"]) == 64
        p2, _, _ = _stage1(tmp_path)
        assert "Skipping" in p2.stdout  # matching provenance -> resume

    def test_source_change_defeats_resume(self, tmp_path):
        p1, out, model = _stage1(tmp_path)
        assert p1.returncode == 0
        model.write_text(SIMPLE_MODEL.replace("randn(4, 8)", "randn(6, 8)"))
        p2 = subprocess.run(
            [PY, "-m", "solar.cli.process_model", "--model-file", str(model),
             "--output-dir", str(out)],
            capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
        assert "Skipping" not in p2.stdout
        doc = yaml.safe_load((out / "pytorch_graph.yaml").read_text())
        mm = next(v for v in doc["layers"].values()
                  if v.get("type") == "matmul")
        assert [6, 8] in mm["input_shapes"]  # genuinely reprocessed

    def test_prefix_artifact_without_provenance_is_reprocessed(self, tmp_path):
        """A stale pre-fix artifact (no provenance) cannot stand in for a
        fresh run — here the fresh run fails closed, so the stage must FAIL
        instead of exiting 0 on the stale bytes (round-2/3 masking)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "reg", Path(__file__).parent / "test_learned_param_matmul.py")
        reg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reg)
        model = tmp_path / "model.py"
        model.write_text(reg.UNKNOWN_SIGNATURE_INJECTED)
        out = tmp_path / "graph"
        out.mkdir()
        # plant the captured pre-fix defective artifact (provenance-free)
        (out / "pytorch_graph.yaml").write_text(
            (FIXTURES / "prefix_defect_matmul_graph.yaml").read_text())
        proc = subprocess.run(
            [PY, "-m", "solar.cli.process_model", "--model-file", str(model),
             "--output-dir", str(out)],
            capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
        assert "Skipping" not in proc.stdout
        assert proc.returncode != 0
        assert "UnsupportedFrontendOperand" in (proc.stdout + proc.stderr)
