# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TARGET-DSL-DIM-EXTENT-13 (the review, from the earlier change round-7 review; identical on plain an earlier revision).

``_kernel_loop_iterations`` sized a ``for i in Dim("I")`` loop by the WIDEST positional access that shared the
index letter: with ``X[N=16,H=4]``, ``Idx[I=2]``, output ``[I=2,H=4]`` and body ``Y[i,h] = X[i,h] * 2.0`` it
reported 64 complete iterations where the declared loop dimensions give 2 x 4 = 8.

A loop bound to a declared ``Dim("X")`` iterates the DECLARED dimension.  Its extent at a call is read from the
kernel's own Tensor declarations by axis position (parameter annotations, the return annotation, the output
allocation).  A positional access is only a fallback for a dim the kernel never declares, and a witness: a loop
variable indexing an axis NARROWER than its declared extent, or declaration sites that disagree, is an
inconsistency recorded as ``loop_dim_extent_mismatch`` (the count is incomplete, the analyzer keeps its equation
basis) -- never a silent max.

Round 2 (the review, annotation-binding-1 / TARGET-DSL-DIM-LOCAL-2): the binding of ``Dim("X")`` is the RUNTIME
registry's, not the parameter annotation's.  ``@kernel`` registers each argument's actual dim names (the caller's
``from_numpy`` names) and every ``Tensor(X=...)`` allocation in the body registers its names; annotations never rebind.
The converter mirrors exactly that: inputs by name, local allocations (sizes from this call's shapes), the returned
allocation; a name bound to different sizes, or only by an allocation it cannot size, is refused (incomplete).
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.einsum_dsl.einsum_converter import LoopDimExtentMismatch, _runtime_dim_bindings
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer, dsl_loop_count_evidence

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "loop_dim_extent"
H = "from solar.einsum_lib import Tensor, Dim, kernel\n"

GATHER = H + '''@kernel
def gather(X: Tensor("N,H", dtype="fp32"), Idx: Tensor("I", dtype="int64")) -> Tensor("I,H"):
    Y = Tensor(I=Idx.shape[0], H=X.shape[1], dtype="fp32")
    for i in Dim("I"):
        for h in Dim("H"):
            Y[i,h] = X[i,h] * 2.0
    return Y
'''
GEMM = H + '''@kernel
def gemm(A: Tensor("M,K"), B: Tensor("K,N")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[m,n] += A[m,k] * B[k,n]
    return Y
'''
PROJ = H + '''@kernel
def proj(X: Tensor("S,H"), W: Tensor("F,H")) -> Tensor("S,F"):
    Y = Tensor(S=X.shape[0], F=W.shape[0])
    for s in Dim("S"):
        for f in Dim("F"):
            for h in Dim("H"):
                Y[s,f] += X[s,h] * W[f,h]
    return Y
'''
NARROW = H + '''@kernel
def narrow(X: Tensor("T"), Z: Tensor("U")) -> Tensor("T"):
    Y = Tensor(T=X.shape[0])
    for t in Dim("T"):
        Y[t] = X[t] + Z[t]
    return Y
'''
UNDECLARED = H + '''@kernel
def half(X: Tensor("N")) -> Tensor("N"):
    Y = Tensor(N=X.shape[0])
    for p in Dim("P"):
        Y[p] = X[p]
    return Y
'''


def _t(dims, role="", dtype="fp32"):
    d = {"dims": dims, "dtype": dtype}
    if role:
        d["role"] = role
    return d


def _convert(tmp_path, py, tensors, graph):
    d = tmp_path / "dsl"
    d.mkdir()
    (d / "m_einsum_lib.py").write_text(py)
    (d / "m_einsum_lib.yaml").write_text(yaml.safe_dump({"tensors": tensors, "graph": graph}, sort_keys=False))
    p = translate_with_einsum(d, tmp_path / "einsum")
    return yaml.safe_load(p.read_text()), p


def _call(fn, inputs, output):
    # ``input_order: signature`` is what a real trace records (inputs in parameter order); arithmetic size specs need it
    return {"function": fn, "inputs": [{"tensor": t, "from": "input"} for t in inputs], "output": output, "input_order": "signature"}


class TestReview4Control:
    def test_declared_dims_size_the_loop(self, tmp_path):
        """the reviewer's case: I=2 x H=4 = 8 complete iterations, not 16 x 4 = 64."""
        raw, _ = _convert(tmp_path, GATHER,
                          {"x": _t({"N": 16, "H": 4}, "input"), "i": _t({"I": 2}, "input", "int64"), "out": _t({"I": 2, "H": 4})},
                          {"gather#0": _call("gather", ["x", "i"], "out")})
        L = raw["layers"]["gather#0"]
        assert L["loop_iterations"] == 8 and L["loop_iterations_complete"] is True
        assert "loop_dim_extent_mismatch" not in L and "loop_iterations_omitted" not in L

    def test_review4_file_verbatim(self, tmp_path):
        """the reviewer's ``test_loop_dimension.py`` byte-for-byte (sha256 in the fixture dir), run in a scratch copy
        together with the saved trace it loads from its own directory."""
        run = tmp_path / "run"
        shutil.copytree(FIX, run)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_loop_dimension.py"), "-q", "-p", "no:cacheprovider"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0 and "1 passed" in res.stdout, res.stdout[-2000:] + res.stderr[-1000:]

    def test_partial_access_of_a_wider_axis_is_consistent(self, tmp_path):
        """Reading rows 0..I-1 of a 16-row X is a partial traversal, not an inconsistency (this IS the reviewer's case);
        the wider axis never widens the loop."""
        raw, _ = _convert(tmp_path, GATHER,
                          {"x": _t({"N": 1024, "H": 4}, "input"), "i": _t({"I": 3}, "input", "int64"), "out": _t({"I": 3, "H": 4})},
                          {"gather#0": _call("gather", ["x", "i"], "out")})
        assert raw["layers"]["gather#0"]["loop_iterations"] == 12


class TestDeclaredExtentResolution:
    def test_parser_records_local_tensor_allocations(self, tmp_path):
        p = tmp_path / "k_einsum_lib.py"
        p.write_text(GATHER)
        info = extract_kernel_info(p)["gather"]
        [alloc] = info["local_tensor_dims"]
        assert alloc["dims"] == {"I": ("shape", "Idx", 0), "H": ("shape", "X", 1)} and alloc["conditional"] is False
        assert info["loop_path_lines"] == [[5, 6]] and alloc["line"] == 4 and alloc["target"] == "Y"   # each loop's entry line; the allocation at line 4 precedes both
        assert info["returned_names"] == ["Y"]
        assert "param_dims" not in info and "return_dims" not in info

    def test_consistent_reduction_counts_the_declared_product(self, tmp_path):
        raw, _ = _convert(tmp_path, GEMM,
                          {"a": _t({"M": 4, "K": 8}, "input"), "b": _t({"K": 8, "N": 3}, "weight"), "out": _t({"M": 4, "N": 3})},
                          {"gemm#0": _call("gemm", ["a", "b"], "out")})
        L = raw["layers"]["gemm#0"]
        assert L["loop_iterations"] == 96 and L["loop_iterations_complete"] is True and L["is_real_einsum"] is True

    def test_shared_kernel_is_sized_per_call_from_its_runtime_bindings(self, tmp_path):
        """q-proj with F=8 and k-proj with F=2 through ONE kernel: ``F`` is whatever the call's weight registers
        (by name, per call) -- 160 then 40 -- never a graph-wide max, never a positional guess."""
        raw, _ = _convert(tmp_path, PROJ,
                          {"x": _t({"S": 5, "H": 4}, "input"), "wq": _t({"F": 8, "H": 4}, "weight"), "wk": _t({"F": 2, "H": 4}, "weight"),
                           "q": _t({"S": 5, "F": 8}), "k": _t({"S": 5, "F": 2})},
                          {"proj#0": _call("proj", ["x", "wq"], "q"), "proj#1": _call("proj", ["x", "wk"], "k")})
        assert raw["layers"]["proj#0"]["loop_iterations"] == 5 * 8 * 4
        assert raw["layers"]["proj#1"]["loop_iterations"] == 5 * 2 * 4

    def test_annotation_names_are_not_bindings(self, tmp_path):
        """the reviewer's annotation-binding-1 shape: ``k(X: Tensor("T"), Z: Tensor("U"))`` called with x named N=8
        and z / w named T=2 / T=3: the loop over ``Dim("T")`` runs 2 then 3 times (the registry holds the callers'
        names), not 8 (X's axis 0 under the annotation)."""
        code = H + '''@kernel
def k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):
    Y = Tensor(O=1)
    for t in Dim("T"):
        Y[0] += X[0]
    return Y
'''
        raw, _ = _convert(tmp_path, code,
                          {"x": _t({"N": 8}, "input"), "z": _t({"T": 2}, "input"), "w": _t({"T": 3}, "input"), "a": _t({"O": 1}), "b": _t({"O": 1})},
                          {"k#0": _call("k", ["x", "z"], "a"), "k#1": _call("k", ["x", "w"], "b")})
        assert [raw["layers"][k]["loop_iterations"] for k in ("k#0", "k#1")] == [2, 3]
        assert all(raw["layers"][k]["loop_iterations_complete"] is True for k in ("k#0", "k#1"))

    def test_local_tensor_allocation_registers_its_dim(self, tmp_path):
        """TARGET-DSL-DIM-LOCAL-2 (baseline-owned, fixed here): ``Tmp = Tensor(R=3)`` registers R; the loop over
        ``Dim("R")`` runs 3 times although the positional access ``X[r]`` spans N=8."""
        code = H + '''@kernel
def k(X: Tensor("N")) -> Tensor("O"):
    Tmp = Tensor(R=3)
    Y = Tensor(O=1)
    for r in Dim("R"):
        Y[0] += X[r]
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"N": 8}, "input"), "out": _t({"O": 1})}, {"k#0": _call("k", ["x"], "out")})
        L = raw["layers"]["k#0"]
        assert L["loop_iterations"] == 3 and L["loop_iterations_complete"] is True

    def test_returned_allocation_with_a_non_static_size_uses_the_recorded_output(self, tmp_path):
        """``ni = G1.shape[2] // 2; Y = Tensor(..., I=ni)`` (row 229's swiglu): in a legacy trace without the
        signature-order marker the arithmetic size is not static, but the returned allocation's size IS recorded
        (output I=2048) -- the loop over ``Dim("I")`` counts 2048, not the 4096 of the positional access."""
        code = H + '''@kernel
def swiglu(G1: Tensor("T,N")) -> Tensor("T,I"):
    ni = G1.shape[1] // 2
    Y = Tensor(T=G1.shape[0], I=ni)
    for t in Dim("T"):
        for i in Dim("I"):
            Y[t, i] = G1[t, ni + i] * G1[t, i]
    return Y
'''
        graph = {"swiglu#0": _call("swiglu", ["g"], "out")}; graph["swiglu#0"].pop("input_order")      # legacy trace: no marker
        raw, _ = _convert(tmp_path, code, {"g": _t({"T": 7, "N": 4096}, "input"), "out": _t({"T": 7, "I": 2048})}, graph)
        L = raw["layers"]["swiglu#0"]
        assert L["loop_iterations"] == 7 * 2048 and L["loop_iterations_complete"] is True

    def test_local_allocation_with_unresolvable_size_is_refused(self, tmp_path):
        """A local ``Tensor(R=helper())`` binds R at run time to a size the converter cannot compute: no count,
        ``loop_dim_extent_unresolved``, incomplete -- never the positional fallback."""
        code = H + '''def helper():
    return 3
@kernel
def k(X: Tensor("N")) -> Tensor("O"):
    Tmp = Tensor(R=helper())
    Y = Tensor(O=1)
    for r in Dim("R"):
        Y[0] += X[r]
    return Y
'''
        raw, p = _convert(tmp_path, code, {"x": _t({"N": 8}, "input"), "out": _t({"O": 1})}, {"k#0": _call("k", ["x"], "out")})
        L = raw["layers"]["k#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
        assert L["loop_dim_extent_unresolved"].startswith("loop_dim_extent_unresolved: Dim('R')")
        assert dsl_loop_count_evidence(L).trusted is False

    def test_review3_independent_file_verbatim(self, tmp_path):
        """the reviewer's ``test_independent_dim_extent.py`` byte-for-byte (sha256 3cb2ee2a…): 7 passed (annotation
        binding 2/3, two-dims partial with and without annotations, shared kernel, local Dim alias declines,
        registry-only local Tensor, repository window crop 80)."""
        run = tmp_path / "run"; run.mkdir()
        (run / "test_independent_dim_extent.py").write_bytes((FIX / "test_independent_dim_extent.py").read_bytes())
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["REVIEW_SOURCE"] = str(ROOT); env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_independent_dim_extent.py"), "-q", "-p", "no:cacheprovider"],
                             cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "7 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]

    def test_undeclared_loop_dim_keeps_the_positional_fallback(self, tmp_path):
        raw, _ = _convert(tmp_path, UNDECLARED, {"x": _t({"N": 6}, "input"), "out": _t({"N": 6})},
                          {"half#0": _call("half", ["x"], "out")})
        assert raw["layers"]["half#0"]["loop_iterations"] == 6

    def test_explicit_size_still_wins(self, tmp_path):
        code = H + '''@kernel
def tail(X: Tensor("N")) -> Tensor("N"):
    Y = Tensor(N=X.shape[0])
    for n in Dim("N", size=3):
        Y[n] = X[n]
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"N": 6}, "input"), "out": _t({"N": 6})},
                          {"tail#0": _call("tail", ["x"], "out")})
        assert raw["layers"]["tail#0"]["loop_iterations"] == 3

    def test_runtime_bindings_helper(self):
        class _T:
            def __init__(self, dims):
                self.dims = dims
                self.shape = list(dims.values())

        class _I:
            def __init__(self, dims):
                self.tensor = _T(dims)

        class _K:
            id = "k#0"
            param_names = ["A", "B"]
            inputs = [_I({"M": 4, "K": 8}), _I({"K": 16, "N": 3})]
            output = _T({"M": 4, "N": 3})
            registration_order = None
            returned_names = ["Y"]
            local_tensor_dims = [{"line": 3, "dims": {"M": ("shape", "A", 0), "N": ("shape", "B", 1)}, "conditional": False, "target": "Y"},
                                 {"line": 4, "dims": {"R": 3, "Q": None}, "conditional": False, "target": "Tmp"},
                                 {"line": 7, "dims": {"K": 2}, "conditional": True, "target": "C"},
                                 {"line": 9, "dims": {"K": 2}, "conditional": False, "target": "D"}]

        shapes = {"A": [4, 8], "B": [16, 3]}
        # legacy-kworder-12 (round 4): WITHOUT a recorded registration order, A.K=8 and B.K=16 disagree and
        # which one the kernel read is unknowable from the trace -> K is refused with ``loop_dim_order_unknown``
        ext, conf, unres = _runtime_dim_bindings(_K(), shapes, None, before_line=6)     # a loop at line 6
        assert ext == {"M": 4, "N": 3, "R": 3} and conf == {} and set(unres) == {"Q", "K"}
        assert unres["K"].code == "loop_dim_order_unknown" and "A.K=8" in unres["K"] and "B.K=16" in unres["K"]
        k = _K(); k.registration_order = [0, 1]                                           # positional call: B registered last
        ext, conf, unres = _runtime_dim_bindings(k, shapes, None, before_line=6)
        assert ext == {"M": 4, "K": 16, "N": 3, "R": 3} and conf == {} and set(unres) == {"Q"}
        ext, conf, unres = _runtime_dim_bindings(k, shapes, None, before_line=8)        # after the CONDITIONAL K=2 at line 7
        assert "K" in unres and "K" not in ext                                            # not established at entry
        ext, conf, unres = _runtime_dim_bindings(k, shapes, None, before_line=12)       # the certain K=2 at line 9 re-establishes it
        assert ext["K"] == 2 and "K" not in unres
        k = _K(); k.registration_order = [1, 0]                                           # keyword call: B registered before A
        ext, conf, unres = _runtime_dim_bindings(k, shapes, None, before_line=6)
        assert ext["K"] == 8
        ext, conf, unres = _runtime_dim_bindings(_K(), shapes, None, before_line=None)  # legacy: no loop lines -> disagreement is a conflict
        assert "K" in conf

    def test_returned_allocation_is_identified_by_name_not_dim_set(self, tmp_path):
        """the reviewer's RETURN-IDENTITY-5: ``Y = Tensor(R=3); Tmp = Tensor(R=choose())`` returns Y.  The last allocation
        with dim set {R} is Tmp, but Y is what is returned; Tmp's unresolved size must not borrow Y's recorded 3 --
        the loop over R is refused."""
        code = H + '''def choose():
    return 5
@kernel
def k(X: Tensor("T")) -> Tensor("R"):
    Y = Tensor(R=3)
    Tmp = Tensor(R=choose())
    for r in Dim("R"):
        Y[0] += X[0]
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"T": 2}, "input"), "out": _t({"R": 3})}, {"k#0": _call("k", ["x"], "out")})
        L = raw["layers"]["k#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False and "loop_dim_extent_unresolved" in L

    def test_conditional_allocation_does_not_overwrite_the_registry(self, tmp_path):
        """the reviewer's CONDITIONAL-BINDING-4: an allocation in an untaken branch must not certify a count; with the
        established T=2 and a conditional T=5 the binding at loop entry is not certain -> refused."""
        code = H + '''@kernel
def k(X: Tensor("T")) -> Tensor("O"):
    Y = Tensor(O=1)
    if X[0] < 0:
        Tmp = Tensor(T=5)
    for t in Dim("T"):
        Y[0] += X[0]
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"T": 2}, "input"), "out": _t({"O": 1})}, {"k#0": _call("k", ["x"], "out")})
        L = raw["layers"]["k#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
        assert "conditional allocation" in L["loop_dim_extent_unresolved"]
        # a conditional allocation that rebinds the SAME extent changes nothing: counted, complete
        code2 = code.replace("Tensor(T=5)", "Tensor(T=2)")
        (tmp_path / "same").mkdir()
        raw, _ = _convert(tmp_path / "same", code2, {"x": _t({"T": 2}, "input"), "out": _t({"O": 1})}, {"k#0": _call("k", ["x"], "out")})
        assert raw["layers"]["k#0"]["loop_iterations"] == 2 and raw["layers"]["k#0"]["loop_iterations_complete"] is True

    def test_allocation_inside_the_outer_loop_sizes_the_inner_loop(self, tmp_path):
        """TARGET-DSL-DIM-INNER-6 (baseline, fixed here): ``for t in Dim("T"): Tmp = Tensor(U=3); for u in Dim("U")``
        with T=2, U=1 at entry -- the inner loop reads U=3 (re-registered every iteration): 6 inner assignments +
        2 allocation assignments = 8 statements."""
        code = H + '''@kernel
def k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):
    Y = Tensor(O=1)
    for t in Dim("T"):
        Tmp = Tensor(U=3)
        for u in Dim("U"):
            Y[0] += X[0]
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"T": 2}, "input"), "z": _t({"U": 1}, "input"), "out": _t({"O": 1})}, {"k#0": _call("k", ["x", "z"], "out")})
        L = raw["layers"]["k#0"]
        assert L["loop_iterations"] == 8 and L["loop_iterations_complete"] is True and L["loop_outer_statements"] == 2

    def test_keyword_call_registration_order_is_replayed(self, tmp_path):
        """TARGET-DSL-DIM-KWORDER-7 (baseline, fixed here): ``k(Z=z, X=x)`` registers Z then X; the trace now records
        that order and the mirror replays it (T = X's 2), while the recorded inputs stay in signature order."""
        code = H + '''@kernel
def k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):
    Y = Tensor(O=1)
    for t in Dim("T"):
        Y[0] += X[0]
    return Y
'''
        graph = {"k#0": _call("k", ["x", "z"], "out")}; graph["k#0"]["registration_order"] = [1, 0]
        raw, _ = _convert(tmp_path, code, {"x": _t({"T": 2}, "input"), "z": _t({"T": 5}, "input"), "out": _t({"O": 1})}, graph)
        assert raw["layers"]["k#0"]["loop_iterations"] == 2
        graph = {"k#0": _call("k", ["x", "z"], "out")}; graph["k#0"]["registration_order"] = [0, 1]   # positional call
        (tmp_path / "pos").mkdir()
        raw, _ = _convert(tmp_path / "pos", code, {"x": _t({"T": 2}, "input"), "z": _t({"T": 5}, "input"), "out": _t({"O": 1})}, graph)
        assert raw["layers"]["k#0"]["loop_iterations"] == 5
        # legacy-kworder-12 (round 4): a trace WITHOUT the order (saved before it was recorded) cannot say
        # which T the kernel read -- signature order was a guess that certified 5 for a keyword call that ran 2 -> refused
        graph = {"k#0": _call("k", ["x", "z"], "out")}
        (tmp_path / "legacy").mkdir()
        raw, _ = _convert(tmp_path / "legacy", code, {"x": _t({"T": 2}, "input"), "z": _t({"T": 5}, "input"), "out": _t({"O": 1})}, graph)
        L = raw["layers"]["k#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
        assert L["loop_dim_order_unknown"].startswith("loop_dim_order_unknown: Dim('T')") and "X.T=2" in L["loop_dim_order_unknown"] and "Z.T=5" in L["loop_dim_order_unknown"]
        # inputs that AGREE need no order
        graph = {"k#0": _call("k", ["x", "z"], "out")}
        (tmp_path / "agree").mkdir()
        raw, _ = _convert(tmp_path / "agree", code, {"x": _t({"T": 2}, "input"), "z": _t({"T": 2}, "input"), "out": _t({"O": 1})}, graph)
        assert raw["layers"]["k#0"]["loop_iterations"] == 2 and raw["layers"]["k#0"]["loop_iterations_complete"] is True

    def test_trace_records_the_registration_order(self):
        """A real traced call with keywords carries ``registration_order`` in the metadata; a positional call too."""
        from solar.einsum_lib import clear_dims, stop_trace, get_trace, build_metadata
        from solar.einsum_lib.verify import execute_translated
        code = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n" + '''@kernel
def k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):
    Y = Tensor(O=1)
    for t in Dim("T"):
        Y[0] += X[0]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "T")
    z = Tensor.from_numpy(inputs["z"], "T")
    return {"kw": k(Z=z, X=x).data, "pos": k(x, z).data}
'''
        import numpy as np
        clear_dims(); stop_trace()
        out, _ = execute_translated(code, {}, all_inputs={"x": np.ones(2), "z": np.ones(5)}, fast=False, trace=True, shape_only=False)
        assert out["kw"].tolist() == [2.0] and out["pos"].tolist() == [5.0]
        md = build_metadata(get_trace())
        assert md["graph"]["k#0"]["registration_order"] == [1, 0] and md["graph"]["k#0"]["input_order"] == "signature"
        assert md["graph"]["k#1"]["registration_order"] == [0, 1]

    def test_review4_registry_order_file_verbatim(self, tmp_path):
        """the reviewer's ``test_registry_order.py`` byte-for-byte (sha256 0eea40a5…): 9 passed."""
        run = tmp_path / "run"; run.mkdir()
        (run / "test_registry_order.py").write_bytes((FIX / "test_registry_order.py").read_bytes())
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_registry_order.py"), "-q", "-p", "no:cacheprovider"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "9 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]


class TestMismatchIsFlaggedNotMaxed:
    def test_last_registration_wins_like_the_runtime(self, tmp_path):
        """``X("B,N")`` then ``W("K,N")``: the registry's N is W's (last registration wins) and the allocation agrees,
        so a dense layer whose input is named with the same letter counts B x N x K complete -- the shape of the
        Flax VAE / MNIST matmuls in the saved examples."""
        code = H + '''@kernel
def dense(X: Tensor("B,K"), W: Tensor("K,N")) -> Tensor("B,N"):
    Y = Tensor(B=X.shape[0], N=W.shape[1])
    for b in Dim("B"):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[b, n] += X[b, k] * W[k, n]
    return Y
'''
        raw, _ = _convert(tmp_path, code,
                          {"x": _t({"B": 4, "N": 6}, "input"), "w": _t({"K": 6, "N": 3}, "weight"), "out": _t({"B": 4, "N": 3})},
                          {"dense#0": _call("dense", ["x", "w"], "out")})
        L = raw["layers"]["dense#0"]
        assert L["loop_iterations"] == 4 * 3 * 6 and L["loop_iterations_complete"] is True

    def test_allocation_before_the_loop_rebinds_the_name(self, tmp_path):
        """The registry docstring's own example: an input ``W=32`` and an output allocated ``W=64`` before the loop --
        the loop over ``Dim("W")`` runs 64 times (the allocation registered last)."""
        code = H + '''@kernel
def up(X: Tensor("W")) -> Tensor("W"):
    Y = Tensor(W=2 * X.shape[0])
    for w in Dim("W"):
        Y[w] = 1.0
    return Y
'''
        raw, _ = _convert(tmp_path, code, {"x": _t({"W": 32}, "input"), "out": _t({"W": 64})}, {"up#0": _call("up", ["x"], "out")})
        L = raw["layers"]["up#0"]
        assert L["loop_iterations"] == 64 and L["loop_iterations_complete"] is True

    def test_conflicting_runtime_bindings_with_a_narrower_access_are_refused(self, tmp_path):
        """K is registered as 8 by A and then 16 by B: the registry's K is 16, but the loop variable indexes A's
        8-element axis -- an inconsistency, refused: no count, ``loop_dim_extent_mismatch``, the analyzer gate
        refuses the count and the layer is ``loop_metadata_incomplete``."""
        call = _call("gemm", ["a", "b"], "out"); call["registration_order"] = [0, 1]    # the order is recorded (round 4)
        raw, p = _convert(tmp_path, GEMM,
                          {"a": _t({"M": 4, "K": 8}, "input"), "b": _t({"K": 16, "N": 3}, "weight"), "out": _t({"M": 4, "N": 3})},
                          {"gemm#0": call})
        L = raw["layers"]["gemm#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
        assert L["loop_dim_extent_mismatch"].startswith("loop_dim_extent_mismatch: Dim('K')") and "extent 16" in L["loop_dim_extent_mismatch"] and "only 8 elements" in L["loop_dim_extent_mismatch"]
        assert L["loop_iterations_omitted"] == [L["loop_dim_extent_mismatch"]]
        ev = dsl_loop_count_evidence(L)
        assert ev.present and not ev.trusted and ev.reasons == [L["loop_dim_extent_mismatch"]]
        a = EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        assert a["layers"]["gemm#0"]["incomplete_reason"] == "loop_metadata_incomplete"

    def test_loop_variable_indexing_a_narrower_axis(self, tmp_path):
        """T is declared 8 but ``Z[t]`` indexes a 4-element axis: an out-of-bounds (or guarded) traversal the source
        cannot size -- flagged, not sized by either extent."""
        raw, _ = _convert(tmp_path, NARROW,
                          {"x": _t({"T": 8}, "input"), "z": _t({"U": 4}, "input"), "out": _t({"T": 8})},
                          {"narrow#0": _call("narrow", ["x", "z"], "out")})
        L = raw["layers"]["narrow#0"]
        assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
        assert "bound to extent 8" in L["loop_dim_extent_mismatch"] and "only 4 elements" in L["loop_dim_extent_mismatch"]

    def test_mismatch_is_a_typed_value_error(self):
        assert issubclass(LoopDimExtentMismatch, ValueError)
