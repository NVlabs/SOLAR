# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the campaign-audit repairs (round-10 continuation).

Three confirmed defect classes from the historical 235-problem B200 audit:

* Quant/001 + L1/007 — boundary STORAGE billed by a single per-graph byte
  scalar (BF16 public I/O at fp8's 1 byte; FP32 I/O at fp16's 2 bytes).
  Repaired: definition-declared storage dtypes are authoritative for
  boundary traffic; DSL tags are trace evidence for internal boundary
  tensors; unresolvable widths refuse. Compute throughput unchanged.
* L2/077 — floor-divided window extents were unresolvable, dropping kernels
  onto a standalone-index fallback that undercounted (2,304 vs ~6e11 MACs),
  and floor partitioning silently dropped the padded border.  On THIS branch:
  int-expression Dim sizes resolve under the reviewed input/parameter
  correspondence guard (the internal converter); an explicit Dim nest that does not
  resolve is REFUSED at conversion time (there is no UNRESOLVED reporting
  path, no ``loop_count_unresolved`` / ``total.unresolved_ops``: that older
  implementation was dropped for the fail-closed converter); the
  floor-partition reuse hazard is delivered by a separate change, on which
  ``test_floor_partition_is_a_reuse_hazard`` depends.
* L2/058 — capped module constants (``_G = 2`` vs definition n_groups=8)
  survived verbatim into the "full-shape" DSL.  PENDING on this branch
  (disclosed exclusion, not closure): ``einsum_dsl/constants_binding.py``
  is carried as a helper with no production caller and
  ``scripts/trace_saved_ir.py`` has no binding hook, so _AXIS_BINDINGS
  values are not checked, no provenance file is written and a baked capped
  constant does not refuse the trace.  The four ``TestConstantBindings``
  cases exercising that are strict xfails until a follow-up wires the
  checks; the three helper-level cases pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLAR_ROOT))

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer  # noqa: E402
from solar.einsum_dsl.constants_binding import (  # noqa: E402
    ConstantBindingError, verify_constant_bindings)
from solar.translate.reuse import guard_profile  # noqa: E402


# ---------------------------------------------------------------------------
# shared problem builders
# ---------------------------------------------------------------------------

SCALE_DSL = '''"""fp32-tagged elementwise scale (boundary-billing fixture)."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def scale(X: Tensor("B,H", dtype="fp32"),
          W: Tensor("H", role="weight", dtype="fp32")) -> Tensor("B,H"):
    Y = Tensor(B=X.shape[0], H=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            Y[b, h] = X[b, h] * W[h]
    return Y


def model(x, w):
    return scale(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "H", role="weight", dtype="fp32")
    return {"output": model(x, w).data}
'''

SCALE_ADAPTER = '''import numpy as np
def get_inputs():
    rng = np.random.RandomState(0)
    return {"x": rng.standard_normal((4, 16)).astype(np.float32),
            "w": rng.standard_normal(16).astype(np.float32)}
def kernel_fn(inputs):
    return {"output": inputs["x"] * inputs["w"]}
def get_source():
    return "elementwise scale fixture"
'''

B, H = 4, 16
IO_ELEMS = B * H + H + B * H  # x + w + output = 144


def _scale_problem(tmp_path, io_dtype):
    problem = tmp_path / "problem"
    problem.mkdir(exist_ok=True)
    (problem / "solar_model_einsum_lib.py").write_text(SCALE_DSL)
    (problem / "definition.json").write_text(json.dumps({
        "name": "boundary_fixture",
        "axes": {"B": {"type": "var"}, "H": {"type": "const", "value": H}},
        "inputs": {"x": {"shape": ["B", "H"], "dtype": io_dtype},
                   "w": {"shape": ["H"], "dtype": io_dtype}},
        "outputs": {"output": {"shape": ["B", "H"], "dtype": io_dtype}},
    }))
    (problem / "workload.jsonl").write_text(
        json.dumps({"uuid": "wl0", "axes": {"B": B}}) + "\n")
    (problem / "adapter.py").write_text(SCALE_ADAPTER)
    return problem


def _sol(problem, out, **kw):
    from solar.translate import sol
    return sol(problem / "solar_model_einsum_lib.py", problem, out,
               arch="B200", fallback_adapter=problem / "adapter.py", **kw)


def _perf(sr):
    return yaml.safe_load(Path(sr.perf_path).read_text())


def _mem_bytes(perf):
    return {k: perf[k]["memory_bytes"]
            for k in ("unfused", "fused", "fused_prefetched")}


class TestBoundaryStorageBilling:
    """Quant/001 + L1/007: declared storage widths, one shared layer."""

    def test_fp32_boundary_billed_4_bytes_under_fp16(self, tmp_path):
        """L1/007 class: definition declares float32 I/O; campaign default
        precision fp16 must not halve the boundary bytes."""
        problem = _scale_problem(tmp_path, "float32")
        perf = _perf(_sol(problem, tmp_path / "out", precision="fp16"))
        assert perf["workload"]["boundary_billing"] == \
            "declared_storage_dtypes"
        assert perf["workload"]["model_io_bytes_declared"] == IO_ELEMS * 4
        assert _mem_bytes(perf) == {k: IO_ELEMS * 4 for k in
                                    ("unfused", "fused", "fused_prefetched")}

    def test_declared_boundary_weight_input_is_billed_once(self, tmp_path):
        """Composition with an earlier decision: the definition's declared input `w` is typed `weight` by the
        DSL exporter, so the analyzer now reports it under weight_elements, not model_io_elements.  The
        declared raw total covers it (it is a boundary tensor); the perf re-pricing delta must be taken
        against the covered elements (`total.boundary_declared`), not against the model-I/O class —
        otherwise `w` is added a second time (observed +32 B unfused on this fixture)."""
        problem = _scale_problem(tmp_path, "float32")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        total = yaml.safe_load(Path(sr.analysis_path).read_text())["total"] if hasattr(sr, "analysis_path") else \
            yaml.safe_load((Path(sr.perf_path).parent.parent / "analysis" / "analysis.yaml").read_text())["total"]
        assert total["weight_elements"] == H and total["model_io_elements"] == IO_ELEMS - H
        bd = total["boundary_declared"]
        assert bd == {"elements_raw": IO_ELEMS, "elements_dedup": IO_ELEMS, "elements_raw_weight": H,
                      "bytes_raw_model_io": (IO_ELEMS - H) * 4, "bytes_raw_weight": H * 4,
                      "footprint_elements_weight": H, "bytes_footprint_weight": H * 4}   # w read once in full: footprint == reads
        perf = _perf(sr)
        assert _mem_bytes(perf) == {k: IO_ELEMS * 4 for k in ("unfused", "fused", "fused_prefetched")}
        mb = perf["memory_breakdown"]
        assert mb["model_io_bytes"] == (IO_ELEMS - H) * 4 and mb["weight_bytes"] == H * 4
        assert mb["model_io_bytes"] + mb["weight_bytes"] == perf["workload"]["model_io_bytes_declared"] == IO_ELEMS * 4

    def test_bf16_boundary_billed_2_bytes_under_fp8_override(self, tmp_path):
        """Quant/001 class: BF16 public I/O + graph-wide fp8 perf override.
        Boundary bills 2 bytes/elem; the DSL's fp32 tags must NOT double it
        to 4; fp8 MAC THROUGHPUT selection is untouched."""
        problem = _scale_problem(tmp_path, "bfloat16")
        perf = _perf(_sol(problem, tmp_path / "out", precision="fp16",
                          perf_precision="fp8"))
        assert perf["workload"]["model_io_bytes_declared"] == IO_ELEMS * 2
        assert _mem_bytes(perf) == {k: IO_ELEMS * 2 for k in
                                    ("unfused", "fused", "fused_prefetched")}
        # compute precision/scalar unchanged: workload bpe is still fp8's 1
        assert perf["workload"]["bytes_per_element"] == 1

    def test_bf16_positive_control_unchanged(self, tmp_path):
        """L1/010 class: BF16 I/O under fp16 billing was CORRECT (2 bytes);
        declared billing must reproduce the same bytes."""
        problem = _scale_problem(tmp_path, "bfloat16")
        perf = _perf(_sol(problem, tmp_path / "out", precision="fp16"))
        assert perf["workload"]["model_io_bytes_declared"] == IO_ELEMS * 2
        assert _mem_bytes(perf)["fused"] == IO_ELEMS * 2

    def test_boundary_provenance_recorded(self, tmp_path):
        problem = _scale_problem(tmp_path, "float32")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        analysis = yaml.safe_load(Path(sr.analysis_path).read_text())
        prov = analysis["metadata"]["boundary_storage"]
        assert prov["x"]["source"] == "definition"
        assert prov["x"]["bytes_per_element"] == 4
        # output tensor carries a DSL kernel name; bound by size match
        out_name = next(k for k in prov if k.startswith("scale"))
        assert "definition-output" in prov[out_name]["source"]
        assert prov[out_name]["bytes_per_element"] == 4

    def test_unknown_width_refuses(self, tmp_path):
        """No storage type is ever invented: an unknown declared dtype
        fails loudly."""
        problem = _scale_problem(tmp_path, "float128")
        with pytest.raises(ValueError, match="never silently invented"):
            _sol(problem, tmp_path / "out", precision="fp16")

    def test_ambiguous_output_widths_refuse(self, tmp_path):
        """Two size-matching declared outputs with DIFFERENT widths cannot
        be bound heuristically."""
        problem = _scale_problem(tmp_path, "float32")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        analyzer = EinsumGraphAnalyzer(debug=False)
        with pytest.raises(ValueError, match="ambiguous"):
            analyzer.analyze_graph(
                sr.einsum_path, tmp_path / "re-analysis", precision="fp16",
                copy_graph=False,
                boundary_storage_dtypes={"x": "float32", "w": "float32"},
                declared_output_storage=[
                    {"name": "o1", "dtype": "float32", "elements": B * H},
                    {"name": "o2", "dtype": "bfloat16", "elements": B * H},
                ])


def _renamed_graph(sr, tmp_path, aliases, origins=None):
    """Copy the einsum graph with only tensor names rewritten (shapes,
    equations and declarations untouched) — the C12-P1 reviewer probe.

    ``origins``: when given, REPLACES the graph's recorded
    ``tensor_origins`` map (None leaves whatever the exporter recorded,
    keyed by the ORIGINAL names — i.e. the renamed tensors lose their
    recorded provenance, exactly like the reviewer's alias probe)."""
    graph = yaml.safe_load(Path(sr.einsum_path).read_text())
    for layer in graph["layers"].values():
        for role, names in (layer.get("tensor_names") or {}).items():
            layer["tensor_names"][role] = [aliases.get(n, n) for n in names]
    if origins is not None:
        graph["tensor_origins"] = dict(origins)
    p = tmp_path / "renamed-graph.yaml"
    p.write_text(yaml.safe_dump(graph, sort_keys=False))
    return p


class TestBoundaryInputBinding:
    """C12-P1: public-input storage binding must survive aliases or refuse.

    Verification dtype is NOT public storage evidence merely because an
    aliased public input lacks a direct definition-name match."""

    BOUNDARY = {"x": "bfloat16", "w": "bfloat16", "output": "bfloat16"}
    OUTPUTS = [{"name": "output", "dtype": "bfloat16", "elements": B * H}]

    def _analyze(self, path, out, boundary=None, **kw):
        return EinsumGraphAnalyzer(debug=False).analyze_graph(
            path, out, precision="fp16", copy_graph=False,
            boundary_storage_dtypes=dict(boundary or self.BOUNDARY),
            declared_output_storage=list(self.OUTPUTS), **kw)

    @staticmethod
    def _mixed_problem(tmp_path):
        """BF16 x[4,16] / FP32 w[16] / BF16 output: mixed authoritative
        widths make name-collision binding precedence observable in the
        byte totals (the C19-P2 reviewer fixture; control = 320 bytes)."""
        problem = _scale_problem(tmp_path, "bfloat16")
        d = json.loads((problem / "definition.json").read_text())
        d["inputs"]["w"]["dtype"] = "float32"
        (problem / "definition.json").write_text(json.dumps(d))
        return problem, {"x": "bfloat16", "w": "float32",
                         "output": "bfloat16"}

    def test_renamed_public_inputs_refuse_without_binding(self, tmp_path):
        """The reviewer's positional-alias case: renaming x/w to x_0/x_1
        must NOT silently fall back to the DSL's fp32 trace tags."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        p = _renamed_graph(sr, tmp_path, {"x": "x_0", "w": "x_1"})
        with pytest.raises(ValueError,
                           match="public input.*never silently invented"):
            self._analyze(p, tmp_path / "alias-analysis")

    def test_explicit_binding_restores_declared_widths(self, tmp_path):
        """An explicit trace-to-definition input binding carries the
        authoritative BF16 declaration through the alias."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        baseline = self._analyze(sr.einsum_path, tmp_path / "base-analysis")
        p = _renamed_graph(sr, tmp_path, {"x": "x_0", "w": "x_1"})
        bound = self._analyze(
            p, tmp_path / "bound-analysis",
            boundary_input_binding={"x_0": "x", "x_1": "w"})
        assert (bound["total"]["model_io_bytes_declared"]
                == baseline["total"]["model_io_bytes_declared"]
                == IO_ELEMS * 2)
        prov = bound["metadata"]["boundary_storage"]
        assert prov["x_0"]["source"] == "definition(bound:x_0->x)"
        assert prov["x_0"]["bytes_per_element"] == 2

    def test_binding_to_undeclared_name_refuses(self, tmp_path):
        """A binding that names no declared definition input is dangling,
        not trace-fallback license."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        p = _renamed_graph(sr, tmp_path, {"x": "x_0"})
        with pytest.raises(ValueError, match="dangling"):
            self._analyze(p, tmp_path / "dangling-analysis",
                          boundary_input_binding={"x_0": "nonexistent"})

    def test_reserved_looking_public_alias_refuses(self, tmp_path):
        """C19-P1: ``__weight_0`` is a valid Python identifier a user can
        register as a PUBLIC input — a name prefix is not provenance.  A
        renamed public input whose recorded provenance is lost must refuse,
        never bill its fp32 trace tag."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        p = _renamed_graph(sr, tmp_path,
                           {"x": "__input_0", "w": "__weight_0"})
        with pytest.raises(ValueError,
                           match="public input.*never silently invented"):
            self._analyze(p, tmp_path / "reserved-alias-analysis")

    def test_recorded_internal_constant_keeps_trace_provenance(self, tmp_path):
        """A tensor the graph RECORDS as internally created (exporter
        ``tensor_origins``: internal-constant) keeps its trace dtype as
        admissible storage evidence, distinct from public inputs."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        p = _renamed_graph(sr, tmp_path, {"w": "__weight_0"},
                           origins={"__weight_0": "internal-constant"})
        a = self._analyze(p, tmp_path / "internal-analysis")
        prov = a["metadata"]["boundary_storage"]
        assert prov["__weight_0"]["source"] == "trace(internal-constant)"
        # trace tag is fp32 in SCALE_DSL — billed at its own width, 4 bytes
        assert prov["__weight_0"]["bytes_per_element"] == 4
        assert prov["x"]["source"] == "definition"

    def test_dangling_binding_refuses_even_for_internal_constant(self, tmp_path):
        """C19-P1: an explicit binding to an undeclared name refuses
        REGARDLESS of the tensor's name or recorded provenance — it never
        falls through to trace billing."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        p = _renamed_graph(sr, tmp_path, {"w": "__weight_0"},
                           origins={"__weight_0": "internal-constant"})
        with pytest.raises(ValueError, match="dangling"):
            self._analyze(p, tmp_path / "dangling-internal-analysis",
                          boundary_input_binding={"__weight_0": "nonexistent"})

    MIXED_BYTES = B * H * 2 + H * 4 + B * H * 2  # x bf16 + w fp32 + out bf16

    def test_explicit_binding_wins_over_direct_name_collision(self, tmp_path):
        """C19-P2: a supplied explicit binding is authoritative even when
        the graph alias COLLIDES with another declared input name.  Swapping
        only the graph names x<->w with the correct reverse mapping must
        restore the mixed-width control bytes, not silently resolve each
        alias by its (wrong) direct name."""
        problem, boundary = self._mixed_problem(tmp_path)
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        control = self._analyze(sr.einsum_path, tmp_path / "ctl-analysis",
                                boundary=boundary)
        assert (control["total"]["model_io_bytes_declared"]
                == self.MIXED_BYTES == 320)
        p = _renamed_graph(sr, tmp_path, {"x": "w", "w": "x"},
                           origins={"x": "registered-input",
                                    "w": "registered-input"})
        bound = self._analyze(p, tmp_path / "swap-analysis",
                              boundary=boundary,
                              boundary_input_binding={"w": "x", "x": "w"})
        assert bound["total"]["model_io_bytes_declared"] == self.MIXED_BYTES
        prov = bound["metadata"]["boundary_storage"]
        assert prov["w"]["source"] == "definition(bound:w->x)"
        assert prov["w"]["bytes_per_element"] == 2
        assert prov["x"]["source"] == "definition(bound:x->w)"
        assert prov["x"]["bytes_per_element"] == 4

    def test_direct_name_dangling_binding_refuses(self, tmp_path):
        """C19-P2: a binding to an undeclared name refuses even when the
        graph name ALSO matches a declared input directly — an invalid
        supplied binding is never silently ignored."""
        problem, boundary = self._mixed_problem(tmp_path)
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        with pytest.raises(ValueError, match="dangling"):
            self._analyze(sr.einsum_path, tmp_path / "direct-dangling",
                          boundary=boundary,
                          boundary_input_binding={"x": "nonexistent"})

    def test_no_binding_keeps_direct_name_resolution(self, tmp_path):
        """With no supplied binding the existing direct-name behavior is
        preserved (mixed widths billed per declaration)."""
        problem, boundary = self._mixed_problem(tmp_path)
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        a = self._analyze(sr.einsum_path, tmp_path / "plain-analysis",
                          boundary=boundary)
        prov = a["metadata"]["boundary_storage"]
        assert prov["x"]["source"] == prov["w"]["source"] == "definition"
        assert a["total"]["model_io_bytes_declared"] == self.MIXED_BYTES

    def test_exported_graph_records_registered_input_origins(self, tmp_path):
        """The einsum graph carries actual provenance from the trace: the
        registered public inputs appear in ``tensor_origins`` as
        registered-input (the analyzer's evidence, replacing any name-
        prefix inference)."""
        problem = _scale_problem(tmp_path, "bfloat16")
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        graph = yaml.safe_load(Path(sr.einsum_path).read_text())
        origins = graph.get("tensor_origins") or {}
        assert origins.get("x") == "registered-input"
        assert origins.get("w") == "registered-input"


class TestMetadataProvenance:
    """C19-P1: build_metadata records actual origin, never a name prefix."""

    def test_registered_reserved_looking_name_is_public(self):
        """A registered input NAMED __input_0 is still a public registered
        input — the __<role>_<n> spelling is not a reserved namespace."""
        from solar.einsum_lib.metadata import build_metadata
        md = build_metadata([
            {"type": "tensor", "id": 1, "dim_names": ["N"], "shape": [4],
             "dtype": "float32", "role": "input", "input_name": "__input_0"}])
        assert md["tensors"]["__input_0"]["origin"] == "registered-input"

    def test_auto_named_constant_records_internal_origin(self):
        from solar.einsum_lib.metadata import build_metadata
        md = build_metadata([
            {"type": "tensor", "id": 1, "dim_names": ["N"], "shape": [4],
             "dtype": "float32", "role": "input"}])
        assert md["tensors"]["__input_0"]["origin"] == "internal-constant"

    def test_auto_naming_never_clobbers_registered_tensor(self):
        """When a registered public input already owns __input_0, the
        auto-namer must pick a free name instead of overwriting it."""
        from solar.einsum_lib.metadata import build_metadata
        md = build_metadata([
            {"type": "tensor", "id": 1, "dim_names": ["N"], "shape": [4],
             "dtype": "float32", "role": "input", "input_name": "__input_0"},
            {"type": "tensor", "id": 2, "dim_names": ["M"], "shape": [8],
             "dtype": "float32", "role": "input"}])
        assert md["tensors"]["__input_0"]["origin"] == "registered-input"
        assert md["tensors"]["__input_0"]["dims"] == {"N": 4}
        auto = [n for n, t in md["tensors"].items()
                if t.get("origin") == "internal-constant"]
        assert len(auto) == 1 and auto[0] != "__input_0"
        assert md["tensors"][auto[0]]["dims"] == {"M": 8}


class TestDeclaredBoundaryRecoveredIndex:
    """M97-INTEG-D09-1 (composition with an earlier change): a gather index RECOVERED from the call attributes has
    no tensor name, so it never reached the named boundary bookkeeping and the declared raw/dedup
    totals (and perf's override) silently dropped its read (180 B where the undeclared traffic and the
    connected-index control say 204 B).  Every external read group now takes part in the declared
    totals; a recovered index is billed at the dtype the attributes recorded for it (the evidence that
    priced its undeclared read); an unrecorded dtype is billed at the trace width it already carries
    AND marked incomplete (layer -> total -> perf), never a silent smaller total."""

    DECL = {"start.Output": "float16", "Model.g.Output": "float32"}   # source fp16, output fp32: 30*2 + 3*8 + 30*4 = 204 B

    @staticmethod
    def _case(tmp_path, connected, render_dtype=True):
        from test_gather_traffic import _typed_getitem_case
        from solar.perf.perf_model import EinsumGraphPerfModel
        _, layer, a = _typed_getitem_case(tmp_path, connected=connected, render_dtype=render_dtype)
        decl = dict(TestDeclaredBoundaryRecoveredIndex.DECL)
        if connected:
            decl["start_1.Output"] = "int64"
        out = tmp_path / "declared"
        a2 = EinsumGraphAnalyzer().analyze_graph(tmp_path / "einsum_graph_renamed.yaml", out, precision="fp32",
                                                 copy_graph=False, boundary_storage_dtypes=decl)
        p = EinsumGraphPerfModel().predict(out / "analysis.yaml", out / "perf", arch_config="H100_PCIe",
                                           precision="fp32", copy_analysis=False)
        return a, a2, p

    def test_connected_index_control_is_204_bytes(self, tmp_path):
        a, a2, p = self._case(tmp_path, connected=True)
        assert (a["total"]["unfused_bytes"], a["total"]["fused_bytes"]) == (264, 264)           # fp32 everywhere, undeclared
        assert (a2["total"]["model_io_bytes_declared"], a2["total"]["external_io_bytes_declared_dedup"]) == (204, 204)
        assert (p["unfused"]["memory_bytes"], p["fused"]["memory_bytes"]) == (204, 204)
        assert p["workload"]["boundary_billing"] == "declared_storage_dtypes" and "status" not in p["workload"]

    def test_recovered_index_agrees_with_the_connected_control(self, tmp_path):
        a, a2, p = self._case(tmp_path, connected=False)
        assert (a["total"]["unfused_bytes"], a["total"]["fused_bytes"]) == (264, 264)
        assert (a2["total"]["model_io_bytes_declared"], a2["total"]["external_io_bytes_declared_dedup"]) == (204, 204), \
            "the 24-byte recovered index read must stay in the declared totals"
        assert (p["unfused"]["memory_bytes"], p["fused"]["memory_bytes"]) == (204, 204)
        bs = a2["metadata"]["boundary_storage"]["Model.g::recovered_index"]
        assert bs["dtype"] == "torch.int64" and bs["bytes_per_element"] == 8 and bs["elements_raw"] == 3 == bs["elements_dedup"]
        assert bs["source"].startswith("trace(recovered-index")
        assert "status" not in a2["layers"]["Model.g"] and "status" not in a2["total"] and "status" not in p["workload"]

    def test_recovered_index_of_unknown_width_is_incomplete_not_smaller(self, tmp_path):
        a, a2, p = self._case(tmp_path, connected=False, render_dtype=False)
        # undeclared: the index already carries the source's trace width (4 B) and an earlier fix's gather_index_dtype_unknown marker
        assert a["total"]["unfused_bytes"] == 30 * 4 + 3 * 4 + 30 * 4 == 252
        assert a["layers"]["Model.g"]["incomplete_reason"] == "gather_index_dtype_unknown"
        # declared: the index is billed at that trace width (never dropped), the layer keeps its first reason and
        # gains the boundary detail, the total and perf carry the incomplete status
        assert (a2["total"]["model_io_bytes_declared"], a2["total"]["external_io_bytes_declared_dedup"]) == (30 * 2 + 3 * 4 + 30 * 4, 30 * 2 + 3 * 4 + 30 * 4) == (192, 192)
        g = a2["layers"]["Model.g"]
        assert g["status"] == "incomplete" and g["boundary_width_unresolved"] == ["Model.g::recovered_index"]
        assert a2["total"]["status"] == "incomplete" and "Model.g" in a2["total"]["incomplete_layers"]
        bs = a2["metadata"]["boundary_storage"]["Model.g::recovered_index"]
        assert bs["bytes_per_element"] == 4 and "width unresolved" in bs["source"]
        assert p["workload"]["status"] == "incomplete" and p["workload"]["prediction_qualification"].startswith("unqualified")
        assert (p["unfused"]["memory_bytes"], p["fused"]["memory_bytes"]) == (192, 192)


class TestDeclaredWeightFootprint:
    """M97-WEIGHT-2 (the reviewer's round-2 footprint probes): perf's weight_bytes is the parameter
    FOOTPRINT (distinct parameter source per consuming op x full numel) at the declared width — never
    the raw read count.  A parameter in two slots is read twice; an indexed table is read partially;
    the breakdown must say 32 x width and 60 x width, with the traffic totals unchanged."""

    @staticmethod
    def _run(tmp_path, g, decl):
        import yaml as _yaml
        from solar.perf.perf_model import EinsumGraphPerfModel
        p = tmp_path / "input.yaml"; p.write_text(_yaml.safe_dump(g))
        a = EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "a", precision="fp32", copy_graph=False, boundary_storage_dtypes=decl)
        perf = EinsumGraphPerfModel().predict(tmp_path / "a" / "analysis.yaml", tmp_path / "perf", arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
        return a, perf

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_parameter_footprint_not_repeated_read_count(self, tmp_path, width):
        dtype = {1: "int8", 2: "float16", 4: "float32"}[width]
        g = {"model_name": "same_parameter_twice", "layers": {"e": {
            "type": "einsum", "einsum_equation": "AB,AB,AB->AB", "is_real_einsum": True,
            "tensor_names": {"inputs": ["x", "w", "w"], "outputs": ["y"]},
            "tensor_shapes": {"inputs": [[2, 16]] * 3, "outputs": [[2, 16]]},
            "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
            "tensor_dtypes": {"inputs": ["float32"] * 3, "outputs": ["float32"]},
            "parameter_sources": {"inputs": [None, "parameter-node", "parameter-node"]},
            "connections": {"inputs": [], "outputs": []}}}}
        a, p = self._run(tmp_path, g, {"x": "float32", "w": dtype, "y": "float32"})
        assert a["total"]["weight_elements"] == 32 and a["total"]["weight_read_elements"] == 64
        assert p["unfused"]["memory_bytes"] == 256 + 64 * width
        assert p["fused"]["memory_bytes"] == 256 + 32 * width
        assert p["memory_breakdown"]["weight_elements"] == 32
        assert p["memory_breakdown"]["weight_bytes"] == 32 * width, p["memory_breakdown"]
        bd = a["total"]["boundary_declared"]
        assert bd["footprint_elements_weight"] == 32 and bd["bytes_footprint_weight"] == 32 * width
        assert bd["elements_raw_weight"] == 64 and bd["bytes_raw_weight"] == 64 * width
        assert "status" not in a["total"] and "status" not in p["workload"]

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_indexed_parameter_footprint_not_subset_read_count(self, tmp_path, width):
        import yaml as _yaml
        from test_gather_traffic import _typed_getitem_case
        dtype = {1: "int8", 2: "float16", 4: "float32"}[width]
        _typed_getitem_case(tmp_path, connected=True)
        g = _yaml.safe_load((tmp_path / "einsum_graph_renamed.yaml").read_text())
        l = g["layers"]["Model.g"]; l["tensor_types"]["inputs"][0] = "weight"; l["parameter_sources"] = {"inputs": ["parameter-node", None]}
        a, p = self._run(tmp_path, g, {"start.Output": dtype, "start_1.Output": "int64", "Model.g.Output": "float32"})
        assert a["total"]["weight_elements"] == 60 and a["total"]["weight_read_elements"] == 30
        assert p["unfused"]["memory_bytes"] == p["fused"]["memory_bytes"] == 144 + 30 * width
        assert p["memory_breakdown"]["weight_elements"] == 60
        assert p["memory_breakdown"]["weight_bytes"] == 60 * width, p["memory_breakdown"]
        bd = a["total"]["boundary_declared"]
        assert bd["footprint_elements_weight"] == 60 and bd["elements_raw_weight"] == 30
        assert "status" not in a["total"] and "status" not in p["workload"]


class TestDeclaredOutputShapeExpressions:
    """C12-P2: valid definition shape expressions must resolve, not int()."""

    def test_expression_output_shape_matches_plain_axis(self, tmp_path):
        """[B, H*1] is semantically identical to [B, H]: same widths and
        declared byte counts."""
        problem = _scale_problem(tmp_path, "bfloat16")
        d = json.loads((problem / "definition.json").read_text())
        d["outputs"]["output"]["shape"] = ["B", "H*1"]
        (problem / "definition.json").write_text(json.dumps(d))
        perf = _perf(_sol(problem, tmp_path / "out", precision="fp16"))
        assert perf["workload"]["model_io_bytes_declared"] == IO_ELEMS * 2
        assert _mem_bytes(perf)["fused"] == IO_ELEMS * 2

    def test_unknown_axis_in_output_shape_still_errors(self, tmp_path):
        """A genuinely unknown axis name keeps a proper, loud error."""
        problem = _scale_problem(tmp_path, "bfloat16")
        d = json.loads((problem / "definition.json").read_text())
        d["outputs"]["output"]["shape"] = ["B", "H*Q"]
        (problem / "definition.json").write_text(json.dumps(d))
        with pytest.raises(ValueError, match="Unknown axis name 'Q'"):
            _sol(problem, tmp_path / "out", precision="fp16")


# ---------------------------------------------------------------------------
# L2/077: window counts + padding
# ---------------------------------------------------------------------------

WINDOW_DSL_FLOOR = '''"""Floored window sum (L2/077-class translation defect)."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_WS = 3
_FREE_CONSTANTS = ["_WS"]


@kernel
def win_sum(X: Tensor("B,H,W", dtype="fp32")) -> Tensor("B,P"):
    nwh = X.shape[1] // _WS
    nww = X.shape[2] // _WS
    Y = Tensor(B=X.shape[0], P=nwh * nww, dtype="fp32")
    for b in Dim("B", parallel=True):
        for wh in Dim("WH", size=nwh):
            for ww in Dim("WW", size=nww):
                for i in Dim("I", size=_WS):
                    for j in Dim("J", size=_WS):
                        Y[b, wh * nww + ww] += X[b, wh * _WS + i,
                                                 ww * _WS + j]
    return Y


def model(x):
    return win_sum(x)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,H,W", dtype="fp32")
    return {"output": model(x).data}
'''

PADDED_REF_ADAPTER = '''import numpy as np
B, H, W, WS = 2, 7, 7, 3

def get_inputs():
    rng = np.random.RandomState(0)
    return {"x": rng.standard_normal((B, H, W)).astype(np.float32)}

def kernel_fn(inputs):
    """GENUINE padded-window reference: pads H,W up to multiples of WS
    (SAM-style window partition), so a floored translation that drops the
    border MISMATCHES instead of being vacuously confirmed."""
    x = inputs["x"]
    hp = ((H + WS - 1) // WS) * WS
    wp = ((W + WS - 1) // WS) * WS
    xp = np.zeros((B, hp, wp), np.float32)
    xp[:, :H, :W] = x
    nwh, nww = hp // WS, wp // WS
    out = np.zeros((B, nwh * nww), np.float32)
    for wh in range(nwh):
        for ww in range(nww):
            out[:, wh * nww + ww] = xp[:, wh*WS:(wh+1)*WS,
                                       ww*WS:(ww+1)*WS].sum(axis=(1, 2))
    return {"output": out}

def get_source():
    return "padded window sum reference"
'''

WINDOW_DSL_PADDED = '''"""Padded window sum (corrected translation)."""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_WS = 3
_FREE_CONSTANTS = ["_WS"]


@kernel
def win_sum_pad(X: Tensor("B,H,W", dtype="fp32")) -> Tensor("B,P"):
    nwh = (X.shape[1] + _WS - 1) // _WS
    nww = (X.shape[2] + _WS - 1) // _WS
    Y = Tensor(B=X.shape[0], P=nwh * nww, dtype="fp32")
    for b in Dim("B", parallel=True):
        for wh in Dim("WH", size=nwh):
            for ww in Dim("WW", size=nww):
                for i in Dim("I", size=_WS):
                    for j in Dim("J", size=_WS):
                        if wh * _WS + i < X.shape[1]:
                            if ww * _WS + j < X.shape[2]:
                                Y[b, wh * nww + ww] += X[b, wh * _WS + i,
                                                         ww * _WS + j]
    return Y


def model(x):
    return win_sum_pad(x)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,H,W", dtype="fp32")
    return {"output": model(x).data}
'''


def _window_problem(tmp_path, dsl, n_windows):
    problem = tmp_path / "problem"
    problem.mkdir(exist_ok=True)
    (problem / "solar_model_einsum_lib.py").write_text(dsl)
    (problem / "definition.json").write_text(json.dumps({
        "name": "window_fixture",
        "axes": {"B": {"type": "var"},
                 "H": {"type": "const", "value": 7},
                 "W": {"type": "const", "value": 7},
                 "P": {"type": "const", "value": n_windows}},
        "inputs": {"x": {"shape": ["B", "H", "W"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["B", "P"], "dtype": "float32"}},
    }))
    (problem / "workload.jsonl").write_text(
        json.dumps({"uuid": "wl0", "axes": {"B": 2}}) + "\n")
    (problem / "adapter.py").write_text(PADDED_REF_ADAPTER)
    return problem


class TestWindowCounts:
    def test_compound_index_loop_count_closed_form(self, tmp_path):
        """Floor-divided extents now resolve: loop_iterations equals the
        closed-form B*nwh*nww*ws*ws product, not the standalone-index
        undercount (recorded 2 pre-repair)."""
        problem = _window_problem(tmp_path, WINDOW_DSL_FLOOR, 4)
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        graph = yaml.safe_load(Path(sr.einsum_path).read_text())
        kern = next(v for v in graph["layers"].values()
                    if v.get("type") == "kernel")
        assert kern["loop_iterations"] == 2 * 2 * 2 * 3 * 3  # 72

    def test_floored_translation_fails_against_genuine_padded_reference(
            self, tmp_path):
        """A reference that actually pads (not one that floors away the
        same border) catches the dropped domain numerically."""
        pytest.importorskip("torch")
        from solar.translate import verify
        problem = _window_problem(tmp_path, WINDOW_DSL_FLOOR, 9)
        res = verify(problem / "adapter.py",
                     problem / "solar_model_einsum_lib.py",
                     source_type="cuda", quiet=True)
        assert not res.verified  # shape/value mismatch vs padded reference

    def test_padded_translation_verifies_and_counts(self, tmp_path):
        """The CORRECTED padded translation verifies against the genuine
        padded reference on a nondivisible 7x7/ws=3 window, and its loop
        count covers the padded domain."""
        pytest.importorskip("torch")
        from solar.translate import verify
        problem = _window_problem(tmp_path, WINDOW_DSL_PADDED, 9)
        res = verify(problem / "adapter.py",
                     problem / "solar_model_einsum_lib.py",
                     source_type="cuda", quiet=True)
        assert res.verified, res.message
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        graph = yaml.safe_load(Path(sr.einsum_path).read_text())
        kern = next(v for v in graph["layers"].values()
                    if v.get("type") == "kernel")
        assert kern["loop_iterations"] == 2 * 3 * 3 * 3 * 3  # padded grid

    def test_unresolvable_nest_refuses_not_undercount(self, tmp_path):
        """A Dim nest whose extent cannot be established must never record
        a standalone-index undercount.  Reconciled contract (explicit-loop refusal + admission): an explicit Dim nest refuses outright
        rather than returning a soft ``unresolved`` marker that downstream
        stages could carry into numeric counts."""
        from solar.einsum_dsl.ast_parser import extract_kernel_info
        from solar.einsum_dsl import einsum_converter as ec
        src = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def k(X: Tensor("B,H", dtype="fp32"), ws: int) -> Tensor("B"):
    n = X.shape[1] // ws
    Y = Tensor(B=X.shape[0], dtype="fp32")
    for b in Dim("B", parallel=True):
        for w in Dim("WIN", size=n):
            for i in Dim("I", size=ws):
                Y[b] += X[b, w * ws + i]
    return Y
'''
        p = tmp_path / "m.py"
        p.write_text(src)
        info = extract_kernel_info(p)["k"]

        class _T:
            def __init__(self, shape, dims):
                self.shape = shape
                self.dims = dims

        class _In:
            def __init__(self, t):
                self.tensor = t

        class _K:
            id = "k"  # the retained converter names the kernel in its refusal (M97-P2-2)
            loop_expr = info["loop_expr"]
            loop_paths = info["loop_paths"]
            param_names = info["param_names"]
            inputs = [_In(_T([4, 16], {"B": 4, "H": 16}))]
            output = _T([4], {"B": 4})

        with pytest.raises(ValueError, match="WIN.*Expose its extent"):
            ec._kernel_loop_iterations(_K(), {})

    def test_floor_partition_is_a_reuse_hazard(self):
        prof = guard_profile(WINDOW_DSL_FLOOR)
        assert "floor_partition_extent" in prof.hazards
        assert prof.floor_partition_dims


# ---------------------------------------------------------------------------
# L2/058: constant bindings (real Mamba axes)
# ---------------------------------------------------------------------------

MAMBA_DSL = '''"""Grouped state contraction with REAL Mamba2 definition axes.

_G/_HD/_NST are SEMANTIC constants bound to definition axes; _CS is bound
too; _TILE is a legitimate independent tiling constant.
"""
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

_G = {g}
_HD = {hd}
_NST = {nst}
_CS = {cs}
_TILE = 4
_AXIS_BINDINGS = {{"_G": "n_groups", "_HD": "head_dim",
                   "_NST": "ssm_state_size", "_CS": "chunk_size"}}
_FREE_CONSTANTS = ["_TILE"]


@kernel
def group_state(X: Tensor("B,C,GN", dtype="fp32"),
                W: Tensor("GN,HD", role="weight", dtype="fp32")
                ) -> Tensor("B,C,GH"):
    Y = Tensor(B=X.shape[0], C=_CS, GH=_G * _HD, dtype="fp32")
    for b in Dim("B", parallel=True):
        for c in Dim("C", size=_CS):
            for g in Dim("G", size=_G):
                for h in Dim("HD", size=_HD):
                    for nt in Dim("NT", size=_NST // _TILE):
                        for t in Dim("T", size=_TILE):
                            Y[b, c, g * _HD + h] += (
                                X[b, c, g * _NST + nt * _TILE + t]
                                * W[g * _NST + nt * _TILE + t, h])
    return Y


def model(x, w):
    return group_state(x, w)


def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,C,GN", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "GN,HD", role="weight", dtype="fp32")
    return {{"output": model(x, w).data}}
'''

MAMBA_AXES = {"n_groups": 8, "head_dim": 64, "ssm_state_size": 256,
              "chunk_size": 128}


def _mamba_problem(tmp_path, g=8, hd=64, nst=256, cs=128, bindings=True):
    dsl = MAMBA_DSL.format(g=g, hd=hd, nst=nst, cs=cs)
    if not bindings:
        dsl = dsl.replace("_AXIS_BINDINGS", "_DISABLED_BINDINGS")
    problem = tmp_path / "problem"
    problem.mkdir(exist_ok=True)
    (problem / "solar_model_einsum_lib.py").write_text(dsl)
    (problem / "definition.json").write_text(json.dumps({
        "name": "mamba2_group_state_fixture",
        "axes": {"B": {"type": "var"},
                 **{k: {"type": "const", "value": v}
                    for k, v in MAMBA_AXES.items()},
                 "GN": {"type": "const", "value": 8 * 256},
                 "GH": {"type": "const", "value": 8 * 64}},
        "inputs": {"x": {"shape": ["B", "chunk_size", "GN"],
                         "dtype": "float32"},
                   "w": {"shape": ["GN", "head_dim"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["B", "chunk_size", "GH"],
                               "dtype": "float32"}},
    }))
    (problem / "workload.jsonl").write_text(
        json.dumps({"uuid": "wl0", "axes": {"B": 1}}) + "\n")
    (problem / "adapter.py").write_text('''import numpy as np
def get_inputs():
    rng = np.random.RandomState(0)
    return {"x": rng.standard_normal((1, 128, 2048)).astype(np.float32),
            "w": rng.standard_normal((2048, 64)).astype(np.float32)}
def get_source():
    return "mamba grouped state fixture"
''')
    return problem


class TestConstantBindings:
    @pytest.mark.xfail(strict=True, reason="PENDING (disclosed exclusion, M97-P2-1): constant-binding checks are not wired on this branch (constants_binding.py has no production caller; trace_saved_ir.py has no hook)")
    def test_correct_bindings_pass_with_full_shape_counts(self, tmp_path):
        """Real definition axes (G=8, HD=64, NST=256, CS=128): bound
        constants verify, provenance is recorded, and the loop count is
        the full-shape closed form — grouping is INTERNAL (GN/GH shapes
        alone cannot distinguish G=2 from G=8)."""
        problem = _mamba_problem(tmp_path)
        sr = _sol(problem, tmp_path / "out", precision="fp16")
        graph = yaml.safe_load(Path(sr.einsum_path).read_text())
        kern = next(v for v in graph["layers"].values()
                    if v.get("type") == "kernel")
        assert kern["loop_iterations"] == 1 * 128 * 8 * 64 * 256
        prov = json.loads(
            (Path(sr.dsl_dir) / "constant_provenance.json").read_text())
        assert prov["_G"] == {"value": 8, "binding": "n_groups",
                              "definition_value": 8,
                              "source": "_AXIS_BINDINGS"}
        assert prov["_TILE"] == {"value": 4, "binding": "free",
                                 "source": "_FREE_CONSTANTS"}

    @pytest.mark.xfail(strict=True, reason="PENDING (disclosed exclusion, M97-P2-1): constant-binding checks are not wired on this branch (constants_binding.py has no production caller; trace_saved_ir.py has no hook)")
    def test_baked_capped_constant_refuses(self, tmp_path):
        """The L2/058 case: _G=2 baked against definition n_groups=8 —
        the full-shape trace REFUSES instead of copying it verbatim."""
        problem = _mamba_problem(tmp_path, g=2)
        with pytest.raises(RuntimeError, match="capped value survived"):
            _sol(problem, tmp_path / "out", precision="fp16")

    @pytest.mark.xfail(strict=True, reason="PENDING (disclosed exclusion, M97-P2-1): constant-binding checks are not wired on this branch (constants_binding.py has no production caller; trace_saved_ir.py has no hook)")
    def test_missing_bindings_refuse(self, tmp_path):
        """Undeclared Dim-extent constants against declared const axes
        cannot be presumed independent."""
        problem = _mamba_problem(tmp_path, bindings=False)
        with pytest.raises(RuntimeError, match="no provenance"):
            _sol(problem, tmp_path / "out", precision="fp16")

    def test_similar_names_are_not_bindings(self):
        """A definition axis literally named _G must NOT bind heuristically:
        without _AXIS_BINDINGS the constant is still unbound."""
        code = ("_G = 2\n"
                "from solar.einsum_lib import Tensor, Dim, kernel\n"
                "@kernel\n"
                "def k(X: Tensor('A', dtype='fp32')) -> Tensor('A'):\n"
                "    Y = Tensor(A=X.shape[0], dtype='fp32')\n"
                "    for g in Dim('G', size=_G):\n"
                "        Y[g] = X[g]\n"
                "    return Y\n")
        defn = {"axes": {"_G": {"type": "const", "value": 2}}}
        with pytest.raises(ConstantBindingError, match="no provenance"):
            verify_constant_bindings(code, defn)

    def test_dangling_binding_refuses(self):
        code = ("_AXIS_BINDINGS = {'_G': 'n_groups'}\n")
        with pytest.raises(ConstantBindingError, match="not a module int"):
            verify_constant_bindings(code, {"axes": {
                "n_groups": {"type": "const", "value": 8}}})

    def test_free_constants_preserved_without_const_axes(self):
        """Tiling constants in a definition WITHOUT const axes stay free."""
        code = ("_T = 16\n"
                "from solar.einsum_lib import Tensor, Dim, kernel\n"
                "@kernel\n"
                "def k(X: Tensor('A', dtype='fp32')) -> Tensor('A'):\n"
                "    Y = Tensor(A=X.shape[0], dtype='fp32')\n"
                "    for t in Dim('T', size=_T):\n"
                "        Y[t] = X[t]\n"
                "    return Y\n")
        prov = verify_constant_bindings(code, {"axes": {
            "A": {"type": "var"}}})
        assert prov["_T"]["binding"] == "free"

    @pytest.mark.xfail(strict=True, reason="PENDING (disclosed exclusion, M97-P2-1): constant-binding checks are not wired on this branch (constants_binding.py has no production caller; trace_saved_ir.py has no hook)")
    def test_baked_module_const_dim_is_reuse_hazard(self):
        """reuse.py now classifies Dim(size=<module int const>) as a baked
        literal, not symbolic."""
        code = ("_G = 2\n"
                "from solar.einsum_lib import Tensor, Dim, kernel\n"
                "@kernel\n"
                "def k(X: Tensor('A', dtype='fp32')) -> Tensor('A'):\n"
                "    Y = Tensor(A=X.shape[0], dtype='fp32')\n"
                "    for g in Dim('G', size=_G):\n"
                "        Y[g] = X[g]\n"
                "    return Y\n")
        prof = guard_profile(code)
        assert any("_G=2" in s for s in prof.baked_dim_constants)
