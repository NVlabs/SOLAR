# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""an earlier change round 4 (the reviewer's round-3 review of an earlier revision): the dim registry replayed as LOOP-CARRIED state, the
returned allocation found through EVERY binding of the returned name, early returns, legacy keyword-order traces,
tuple returns, and three KB fixtures from the review (GRU, MaxPool, Mamba2).

return-rebind-9   a non-allocation rebinding of the returned name (``Y = Existing; return Y``) was invisible,
                        so the recorded output dims were borrowed for the EARLIER ``Y = Tensor(R=choose())``
                        (certified 3, runtime 5).  The parser now records every binding; the converter resolves the
                        returned allocation as the last certain binding of the returned name through plain aliases.
loop-carried-10   per-loop entry snapshots filtered allocations by line, so a registration later in an outer
                        body did not reach the next outer iteration (runtime 12 / head 10; 6 / 4).  The nest is now a
                        tree replayed iteration by iteration: first + (n-1) x steady, or a refusal.
TARGET-DSL-EARLY-RETURN-11  a return that is not strictly after a loop (conditional or not) leaves the loop's trip
                        count unprovable: ``loop_entry_unresolved`` (the untaken control refuses too: the condition is
                        data-dependent; only a literal-constant test is decided statically).
legacy-kworder-12 a trace without ``registration_order`` whose inputs carry the same dim name with different
                        extents is refused (``loop_dim_order_unknown``) instead of being read in signature order; the
                        sealed round-2 keyword-order artifact (certified 5 for a run of 2) is the regression.
TARGET-DSL-TUPLE-13     a kernel returning a tuple is refused with ``unsupported_tuple_return`` (was KeyError 'k#0').
TARGET-KB-GRU-10 / Mamba2 tiny / MaxPool (the review KB accounting prep) / LSTM-CELL-EXTENT-1 (the review): the GRU update loop sized by the 768-wide gate
                        access (5 statements billed 15), the LSTM cell loop sized by the 4H gate axis (8 statements billed
                        32) and the Mamba2 chunk loop sized by the wider decay axis (4 MACs billed 6) are EXTENT defects, fixed by the registry binding; ``Dim("R", size=kernel_size)``
                        now resolves from the call's recorded int arguments (``scalar_args``); a trace saved without
                        them is refused naming the argument.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from solar.einsum_dsl.assembler import UnsupportedKernelReturn
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.einsum_dsl.einsum_converter import LoopDimExtentMismatch, _returned_allocation_index, _input_registrations
from solar.einsum_lib import clear_dims, stop_trace, get_trace, build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "loop_dim_extent"
PINS = {
    "round4/checks-00ce6d18/test_round3_edges.py": "b3172e8a4a1fe7850b5ba12926ff469a239990b14dba6c8a4f8dc1eb3d9573c2",
    "round4/checks-00ce6d18/test_legacy_trace.py": "0d412babbf0f395ffd0cc7abd01c78bd85031d94254d04a88d9069f4ad625ecd",
}
H = "from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n"
K = 'def choose():\n    return 5\n@kernel\ndef k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):\n'
RUN = '\ndef run_model(inputs):\n    clear_dims()\n    x = Tensor.from_numpy(inputs["x"], "T")\n    z = Tensor.from_numpy(inputs["z"], "U")\n    return {"out": k(x, z).data}\n'


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_fixture_pins():
    for rel, h in PINS.items():
        assert _sha(FIX / rel) == h, rel


def _trace(tmp_path, code, inputs, name="probe"):
    """Run ``run_model`` under the tracer (fresh registry), save the DSL pair, count executed in-loop assignments."""
    import ast, collections
    fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "k")
    weighted = {n.lineno for loop in ast.walk(fn) if isinstance(loop, ast.For)
                for n in ast.walk(loop) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))}
    counts = collections.Counter()

    def tracer(frame, event, arg):
        if frame.f_code.co_name == "k" and event == "line" and frame.f_lineno in weighted:
            counts[frame.f_lineno] += 1
        return tracer
    dsl = tmp_path / name / "dsl"
    dsl.mkdir(parents=True)
    (dsl / "probe_einsum_lib.py").write_text(code)
    clear_dims(); stop_trace(); sys.settrace(tracer)
    try:
        out, _ = execute_translated(code, {}, all_inputs=inputs, fast=False, trace=True, shape_only=False)
    finally:
        sys.settrace(None)
    md = build_metadata(get_trace())
    write_metadata_yaml(md, str(dsl / "probe_einsum_lib.yaml"))
    return dsl, sum(counts.values()), md, out


def _convert(dsl, out):
    p = translate_with_einsum(dsl, out)
    raw = yaml.safe_load(p.read_text())
    return raw, p


def _layer(raw):
    [(k, L)] = [(k, v) for k, v in raw["layers"].items() if k.startswith("k#")]
    return L


def _refused(L, code):
    assert "loop_iterations" not in L and L["loop_iterations_complete"] is False, L
    assert L[code].startswith(code + ":"), L
    assert L["loop_iterations_omitted"] == [L[code]]
    return L[code]


INPUTS = {"x": np.ones(2), "z": np.ones(1)}


class TestReview4Round3Edges:
    def test_review4_edges_file_verbatim(self, tmp_path):
        """the reviewer's ``test_round3_edges.py`` byte-for-byte (sha256 b3172e8a…): 15 of its 16 cases pass -- counts equal
        to the runtime, or a refusal with no count and an incomplete layer.  ``return_tuple`` is deselected: it asserts
        that conversion succeeds, while the manager's decision (TARGET-DSL-TUPLE-13) is a typed CONVERSION refusal
        (``unsupported_tuple_return``), covered by ``test_tuple_return_is_a_typed_refusal`` below."""
        run = tmp_path / "run"
        shutil.copytree(FIX / "round4" / "checks-00ce6d18", run)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "lane11"
        res = subprocess.run([sys.executable, "-m", "pytest", str(run / "test_round3_edges.py"), "-q", "-p", "no:cacheprovider",
                              "-k", "not return_tuple"], cwd=str(run), env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "15 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]
        # the records the file wrote: which cases counted and which refused, with the exact runtime figures
        got = {}
        for d in (run / "runs" / "edges-lane11").iterdir():
            r = json.loads((d / "record.json").read_text())
            L = next(v for k, v in r["graph"]["layers"].items() if k.startswith("k#"))
            got[d.name] = (r["runtime_weighted_assignments"], L.get("loop_iterations"), L.get("loop_iterations_complete"))
        assert got["between_inner_loops"] == (12, 12, True) and got["after_inner_loop"] == (6, 6, True)   # LOOP-CARRIED-10
        assert got["three_deep"] == (32, 32, True) and got["certain_reset"] == (3, 3, True)
        assert got["return_alias"] == (5, 5, True) and got["return_rebound_tensor"] == (3, 3, True)      # RETURN-REBIND-9
        for refused in ("return_rebound_nonallocation", "return_helper", "early_return_taken", "early_return_untaken",
                        "if_else_different", "try_except", "conditional_inside_outer"):
            assert got[refused][1] is None and got[refused][2] is False, (refused, got[refused])

    def test_review4_legacy_trace_file_verbatim(self, tmp_path):
        """the reviewer's ``test_legacy_trace.py`` (sha256 0d412bab…) against the SEALED round-2 keyword-order artifacts
        (their round-2 manifest hashes are checked by the file itself): 2 passed -- both legacy traces are refused."""
        run = tmp_path / "run"
        shutil.copytree(FIX / "round4", run)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"; env["REVIEW_ROLE"] = "lane11"
        res = subprocess.run([sys.executable, "-m", "pytest", "test_legacy_trace.py", "-q", "-p", "no:cacheprovider"],
                             cwd=str(run / "checks-00ce6d18"), env=env, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0 and "2 passed" in res.stdout, res.stdout[-3000:] + res.stderr[-1000:]
        for name in ("keyword_order", "last_wins_swapped"):
            r = json.loads((run / "checks-00ce6d18" / "runs" / "legacy-lane11" / name / "record.json").read_text())
            for k, L in r["graph"]["layers"].items():
                if k.startswith("k#"):
                    assert "loop_iterations" not in L and L["loop_iterations_complete"] is False
                    assert L["loop_dim_order_unknown"].startswith("loop_dim_order_unknown: Dim('T')"), L


class TestReturnRebinding:
    def test_non_allocation_rebinding_of_the_returned_name_is_not_borrowed(self, tmp_path):
        """RETURN-REBIND-9: ``Y = Existing; return Y`` -- the output IS Existing (R=3); the loop ran over the earlier
        ``Y = Tensor(R=choose())`` (5).  Borrowing R=3 certified 3; now the returned allocation is Existing's, the
        earlier allocation's size is unresolvable, and the loop is refused."""
        code = H + K + '    Existing = Tensor(R=3)\n    Y = Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0] += X[0]\n    Y = Existing\n    return Y\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        assert runtime == 5
        raw, _ = _convert(dsl, tmp_path / "einsum")
        msg = _refused(_layer(raw), "loop_dim_extent_unresolved")
        assert "line 7" in msg and "not statically resolvable" in msg
        info = extract_kernel_info(dsl / "probe_einsum_lib.py")["k"]
        kinds = [(b["name"], b["kind"], b.get("source")) for b in info["name_bindings"]]
        assert kinds == [("Existing", "alloc", None), ("Y", "alloc", None), ("r", "other", None), ("Y", "alias", "Existing")]
        assert info["returns"] == [{"seq": 5, "line": 11, "conditional": False, "name": "Y"}]

    def test_alias_of_the_allocation_is_followed(self, tmp_path):
        code = H + K + '    Y = Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0] += X[0]\n    Alias = Y\n    return Alias\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        assert _layer(raw)["loop_iterations"] == runtime == 5 and _layer(raw)["loop_iterations_complete"] is True

    def test_last_certain_allocation_wins_and_a_helper_return_borrows_nothing(self, tmp_path):
        code = H + K + '    Y = Tensor(R=choose())\n    Y = Tensor(R=3)\n    for r in Dim("R"):\n        Y[0] += X[0]\n    return Y\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        assert _layer(raw)["loop_iterations"] == runtime == 3
        code = H + 'def identity(x):\n    return x\n' + K + '    Y = Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0] += X[0]\n    return identity(Y)\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS, "helper")
        raw, _ = _convert(dsl, tmp_path / "einsum2")
        _refused(_layer(raw), "loop_dim_extent_unresolved")

    def test_returned_allocation_resolution_rules(self):
        class _K:
            local_tensor_dims = [{"line": 3, "dims": {"R": None}, "conditional": False, "target": "A"},
                                 {"line": 4, "dims": {"R": 3}, "conditional": False, "target": "B"},
                                 {"line": 6, "dims": {"R": 2}, "conditional": True, "target": "B"}]
            returned_names = ["B"]
            name_bindings = [{"seq": 1, "name": "A", "line": 3, "kind": "alloc", "source": None, "alloc": 0, "conditional": False},
                             {"seq": 2, "name": "B", "line": 4, "kind": "alloc", "source": None, "alloc": 1, "conditional": False},
                             {"seq": 3, "name": "B", "line": 6, "kind": "alloc", "source": None, "alloc": 2, "conditional": True},
                             {"seq": 4, "name": "C", "line": 7, "kind": "alias", "source": "A", "alloc": None, "conditional": False},
                             {"seq": 5, "name": "D", "line": 8, "kind": "other", "source": None, "alloc": None, "conditional": False}]
            returns = [{"seq": 6, "line": 9, "name": "B", "conditional": False}]
        k = _K()
        assert _returned_allocation_index(k) is None            # B's LAST binding is conditional: not certain
        k.returns = [{"seq": 6, "line": 9, "name": "C", "conditional": False}]
        assert _returned_allocation_index(k) == 0              # alias followed to A's allocation
        k.returns = [{"seq": 6, "line": 9, "name": "D", "conditional": False}]
        assert _returned_allocation_index(k) is None            # a non-allocation binding
        k.returns = [{"seq": 6, "line": 9, "name": "C", "conditional": False}, {"seq": 7, "line": 10, "name": "D", "conditional": False}]
        assert _returned_allocation_index(k) is None            # returns disagree
        k.returns = [{"seq": 6, "line": 9, "name": "X", "conditional": False}]
        assert _returned_allocation_index(k) is None            # a parameter
        k.name_bindings = None; k.returns = None                 # a graph parsed before bindings existed: round-3 rule
        assert _returned_allocation_index(k) is None            # B has a conditional allocation too


class TestLoopCarriedRegistry:
    def test_registration_after_the_inner_loop_reaches_the_next_outer_iteration(self, tmp_path):
        """LOOP-CARRIED-10: ``Tmp = Tensor(U=3)`` after the inner loop: iteration 1 of T runs the inner loop with the
        input's U=1, iteration 2 with U=3 -> 1 + 3 inner trips + 2 outer statements = 6 (head certified 4)."""
        code = H + K + '    Y = Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0] += X[0]\n        Tmp = Tensor(U=3)\n    return Y\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        L = _layer(raw)
        assert runtime == 6 and L["loop_iterations"] == 6 and L["loop_statements"] == 6 and L["loop_outer_statements"] == 2
        assert L["loop_iterations_complete"] is True
        info = extract_kernel_info(dsl / "probe_einsum_lib.py")["k"]
        [t] = info["loop_nest"]
        assert (t["dim"], t["line"], t["end"], t["level_assigns"]) == ("T", 7, 10, 1)
        assert [(c["dim"], c["line"], c["end"], c["weight"]) for c in t["children"]] == [("U", 8, 9, 1)]

    def test_two_inner_loops_around_the_registration(self, tmp_path):
        code = H + K + '    Y = Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0] += X[0]\n        Tmp = Tensor(U=3)\n        for u in Dim("U"):\n            Y[0] += X[0]\n    return Y\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        L = _layer(raw)
        assert runtime == 12 and L["loop_iterations"] == 12 and L["loop_iterations_complete"] is True   # 1+3 | 3+3 | +2

    def test_steady_state_with_three_levels_and_a_conditional_inside_the_outer_body(self, tmp_path):
        code = H + K + ('    Y = Tensor(O=1)\n    for t in Dim("T"):\n        Tmp = Tensor(U=3)\n        for u in Dim("U"):\n'
                        '            Deep = Tensor(V=4)\n            for v in Dim("V"):\n                Y[0] += X[0]\n    return Y\n') + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        assert runtime == 32 and _layer(raw)["loop_iterations"] == 32 and _layer(raw)["loop_iterations_complete"] is True
        code = H + K + '    Y = Tensor(O=1)\n    for t in Dim("T"):\n        if X[0] > 0:\n            Tmp = Tensor(U=3)\n        for u in Dim("U"):\n            Y[0] += X[0]\n    return Y\n' + RUN
        dsl, runtime, _, _ = _trace(tmp_path, code, INPUTS, "cond")
        raw, _ = _convert(dsl, tmp_path / "einsum2")
        msg = _refused(_layer(raw), "loop_dim_extent_unresolved")
        assert "conditional allocation at line 9" in msg

    def test_flat_paths_and_tree_agree_when_nothing_is_carried(self, tmp_path):
        """A nest without in-loop registrations counts exactly as the round-3 flat paths did."""
        code = H + '''@kernel
def gemm(A: Tensor("M,K"), B: Tensor("K,N")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1])
    for m in Dim("M"):
        for n in Dim("N"):
            acc = 0.0
            for k in Dim("K"):
                acc += A[m,k] * B[k,n]
            Y[m,n] = acc
    return Y

def run_model(inputs):
    clear_dims()
    return {"out": gemm(Tensor.from_numpy(inputs["a"], "M,K"), Tensor.from_numpy(inputs["b"], "K,N")).data}
'''
        dsl = tmp_path / "dsl"; dsl.mkdir()
        (dsl / "g_einsum_lib.py").write_text(code)
        clear_dims(); stop_trace()
        execute_translated(code, {}, all_inputs={"a": np.ones((4, 8)), "b": np.ones((8, 3))}, fast=False, trace=True, shape_only=False)
        write_metadata_yaml(build_metadata(get_trace()), str(dsl / "g_einsum_lib.yaml"))
        raw, _ = _convert(dsl, tmp_path / "einsum")
        L = raw["layers"]["gemm#0"]
        assert L["loop_iterations"] == 96 and L["loop_statements"] == 96 + 24 and L["loop_outer_statements"] == 24 and L["is_real_einsum"] is True


class TestEarlyReturn:
    @pytest.mark.parametrize("test, runtime", [("X[0] > 0", 0), ("X[0] < 0", 5)])
    def test_conditional_return_before_the_loop_is_refused(self, tmp_path, test, runtime):
        """EARLY-RETURN-11: taken (the loop never ran, head certified 5) or untaken -- the condition is data-dependent,
        so neither is counted: ``loop_entry_unresolved``, incomplete, no count."""
        code = H + K + f'    Y = Tensor(O=1)\n    if {test}:\n        return Y\n    Tmp = Tensor(T=5)\n    for t in Dim("T"):\n        Y[0] += X[0]\n    return Y\n' + RUN
        dsl, rt, _, _ = _trace(tmp_path, code, INPUTS)
        assert rt == runtime
        raw, p = _convert(dsl, tmp_path / "einsum")
        msg = _refused(_layer(raw), "loop_entry_unresolved")
        assert "is preceded by a conditional return at line 8 (the kernel may end before the loop runs)" in msg
        a = EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        assert a["layers"]["k#0"]["status"] == "incomplete" and a["layers"]["k#0"]["incomplete_reason"] == "loop_metadata_incomplete"

    def test_return_inside_the_loop_and_a_return_after_it(self, tmp_path):
        code = H + K + '    Y = Tensor(O=1)\n    for t in Dim("T"):\n        Y[0] += X[0]\n        if X[0] > 100:\n            return Y\n    return Y\n' + RUN
        dsl, rt, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        assert "encloses a conditional return at line 10 (the kernel may end mid-loop)" in _refused(_layer(raw), "loop_entry_unresolved")
        code = H + K + '    Y = Tensor(O=1)\n    for t in Dim("T"):\n        Y[0] += X[0]\n    if X[0] > 100:\n        return Y\n    return Y\n' + RUN
        dsl, rt, _, _ = _trace(tmp_path, code, INPUTS, "after")
        raw, _ = _convert(dsl, tmp_path / "einsum2")
        assert _layer(raw)["loop_iterations"] == rt == 2            # strictly after the loop: no effect

    def test_literal_constant_condition_is_decided_statically(self, tmp_path):
        code = H + K + '    Y = Tensor(O=1)\n    if False:\n        return Y\n    for t in Dim("T"):\n        Y[0] += X[0]\n    return Y\n' + RUN
        dsl, rt, _, _ = _trace(tmp_path, code, INPUTS)
        raw, _ = _convert(dsl, tmp_path / "einsum")
        assert _layer(raw)["loop_iterations"] == rt == 2 and _layer(raw)["loop_iterations_complete"] is True
        info = extract_kernel_info(dsl / "probe_einsum_lib.py")["k"]
        assert [r["line"] for r in info["returns"]] == [11]          # the dead branch is not walked


class TestLegacyKeywordOrder:
    def test_sealed_round2_keyword_order_artifact_is_refused(self, tmp_path):
        """The sealed round-2 artifact (k(Z=z, X=x), both named T, x T=2 / z T=5; no ``registration_order``) certified
        5 for an interpreter run of 2 -> refused."""
        src = FIX / "round4" / "checks-cf6251bc" / "runs" / "order-head" / "keyword_order"
        man = json.loads((FIX / "round4" / "checks-cf6251bc" / "SHA256SUMS.json").read_text())
        for f in (src / "dsl").iterdir():
            assert _sha(f) == man["runs/order-head/keyword_order/dsl/" + f.name]
        shutil.copytree(src / "dsl", tmp_path / "dsl")
        meta = yaml.safe_load((tmp_path / "dsl" / "probe_einsum_lib.yaml").read_text())
        assert "registration_order" not in meta["graph"]["k#0"] and meta["graph"]["k#0"]["input_order"] == "signature"
        raw, p = _convert(tmp_path / "dsl", tmp_path / "einsum")
        msg = _refused(raw["layers"]["k#0"], "loop_dim_order_unknown")
        assert "X.T=2" in msg and "Z.T=5" in msg and "did not record the registration order" in msg
        a = EinsumGraphAnalyzer().analyze_graph(str(p), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        assert a["layers"]["k#0"]["incomplete_reason"] == "loop_metadata_incomplete"
        assert json.loads((src / "record.json").read_text())["runtime_statement_counts"] == [2]

    def test_input_registration_rules(self):
        class _T:
            def __init__(self, dims): self.dims = dims; self.shape = list(dims.values())

        class _I:
            def __init__(self, dims): self.tensor = _T(dims)

        class _K:
            id = "k#0"; param_names = ["X", "Z"]; registration_order = None
            inputs = [_I({"T": 2}), _I({"T": 5})]
        ext, src, unknown = _input_registrations(_K())
        assert unknown == {"T": [("input X.T", 2), ("input Z.T", 5)]}
        k = _K(); k.registration_order = [1, 0]
        assert _input_registrations(k) == ({"T": 2}, {"T": [("input Z.T", 5), ("input X.T", 2)]}, {})
        k = _K(); k.inputs = [_I({"T": 2}), _I({"T": 2})]
        assert _input_registrations(k)[2] == {}                  # agreeing inputs need no order


class TestTupleReturn:
    def test_tuple_return_is_a_typed_refusal(self, tmp_path):
        code = H + K + '    Y = Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0] += X[0]\n    return Y, Z\n'
        code += '\ndef run_model(inputs):\n    clear_dims()\n    a, b = k(Tensor.from_numpy(inputs["x"], "T"), Tensor.from_numpy(inputs["z"], "U"))\n    return {"out": a.data, "other": b.data}\n'
        dsl = tmp_path / "dsl"; dsl.mkdir()
        (dsl / "probe_einsum_lib.py").write_text(code)
        clear_dims(); stop_trace()
        execute_translated(code, {}, all_inputs=INPUTS, fast=False, trace=True, shape_only=False)
        md = build_metadata(get_trace())
        assert md["graph"]["k#0"]["unsupported_return"] == {"kind": "tuple", "length": 2, "tensors": 2}
        write_metadata_yaml(md, str(dsl / "probe_einsum_lib.yaml"))
        with pytest.raises(UnsupportedKernelReturn) as ei:
            translate_with_einsum(dsl, tmp_path / "einsum")
        assert ei.value.code == "unsupported_tuple_return" and str(ei.value).startswith("unsupported_tuple_return: kernel 'k' (k#0) returned a tuple of 2 values")
        # a legacy trace (no return kind recorded, output missing): still a typed refusal, never KeyError
        meta = yaml.safe_load((dsl / "probe_einsum_lib.yaml").read_text())
        meta["graph"]["k#0"].pop("unsupported_return")
        (dsl / "probe_einsum_lib.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        with pytest.raises(UnsupportedKernelReturn) as ei:
            translate_with_einsum(dsl, tmp_path / "einsum2")
        assert ei.value.code == "unsupported_return"


class TestParserRound4:
    def test_if_branch_loops_do_not_misalign_the_path_lines(self, tmp_path):
        p = tmp_path / "m_einsum_lib.py"
        p.write_text(H + '''@kernel
def k(X: Tensor("T"), Z: Tensor("U")) -> Tensor("O"):
    Y = Tensor(O=1)
    if X[0] > 0:
        for u in Dim("U"):
            Y[0] += Z[0]
    for t in Dim("T"):
        for u in Dim("U"):
            Y[0] += X[0]
    return Y
''')
        info = extract_kernel_info(p)["k"]
        assert info["loop_paths"] == [([("T", None), ("U", None)], 1)] and info["loop_path_lines"] == [[8, 9]]
        assert [n["dim"] for n in info["loop_nest"]] == ["T"] and info["loop_paths_complete"] is False
        assert any("loop nest under a branch" in o for o in info["loop_paths_omitted"])

    def test_bindings_cover_every_form(self, tmp_path):
        p = tmp_path / "m_einsum_lib.py"
        p.write_text(H + '''import math
@kernel
def k(X: Tensor("T"), n: int) -> Tensor("O"):
    Y = Tensor(O=1)
    a, b = 1, 2
    c = (d := 3)
    for i in range(2):
        e = i
    with open("f") as fh:
        pass
    try:
        g = 1
    except ValueError as err:
        g = 2
    import os as o
    def inner():
        return 1
    Y[0] += a
    if True:
        h = 1
    else:
        dead = 1
    return Y
''')
        info = extract_kernel_info(p)["k"]
        names = [(b["name"], b["kind"], b["conditional"]) for b in info["name_bindings"]]
        assert names == [("Y", "alloc", False), ("a", "other", False), ("b", "other", False), ("d", "other", False), ("c", "other", False),
                         ("i", "other", True), ("e", "alias", True), ("fh", "other", True), ("g", "other", True), ("err", "other", True),
                         ("g", "other", True), ("o", "other", False), ("inner", "other", False), ("h", "other", False)]
        assert info["returns"] == [{"seq": len(info["name_bindings"]) + 1, "line": 24, "conditional": False, "name": "Y"}]
        assert info["loop_nest"] is None


class TestScalarArgRecording:
    """Control: the scalar-argument recorder keeps only plain Python scalars, never bools or tensors."""

    def test_scalar_args_are_not_recorded_for_bools_or_tensors(self):
        from solar.einsum_lib import Tensor, Dim, kernel
        from solar.einsum_lib import start_trace

        @kernel
        def k(X: Tensor("T"), n: int, flag: bool = False) -> Tensor("T"):
            Y = Tensor(T=X.shape[0])
            for t in Dim("T"):
                Y[t] = X[t] * n
            return Y
        clear_dims(); stop_trace(); start_trace()
        k(Tensor.from_numpy(np.ones(2), "T"), 3, flag=True)
        md = build_metadata(get_trace()); stop_trace()
        assert md["graph"]["k#0"]["scalar_args"] == {"n": 3}
