# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for torch.einsum MAC accounting (issue #32, an earlier decision).

Convention (see solar/einsum/ops/einsum_ops.py): an N-operand einsum is
costed as the left-to-right chain of pairwise contractions
``((op0 * op1) * op2) * ...`` -- torch's documented order when opt_einsum
is not used.  Each pairwise step costs the product of the extents of the
union of the two operands' index sets; the intermediate keeps the indices
that are still needed by the output or by a later operand.  For two
operands this is exactly the product of all unique indices, i.e. the
existing single-contraction formula (`_macs_from_equation`), so 2-operand
counts are bit-identical to that formula.

Every expected value below is written out as explicit hand arithmetic from
the equation and the extents; none of them calls the implementation.
"""

import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
try:  # the single-contraction formula lived in graph_analyzer until the release branch the lineage base
    from solar.analysis.graph_analyzer import _macs_from_equation
except ImportError:  # the lineage base+: the same product-of-unique-ranks rule is compute_cost_from_equation
    from solar.einsum.ops.base import compute_cost_from_equation as _compute_cost_from_equation

    def _macs_from_equation(equation, input_shapes):
        return _compute_cost_from_equation(equation, TensorShapes(inputs=list(input_shapes), outputs=[]))
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

try:  # absent on the base commit: the handler tests then fail (red), not skip
    from solar.einsum.ops.einsum_ops import TorchEinsumHandler, pairwise_chain_macs
except ImportError:  # pragma: no cover - base commit
    TorchEinsumHandler = None
    pairwise_chain_macs = None


def _ts(shapes, out):
    return TensorShapes(inputs=[list(s) for s in shapes], outputs=[list(out)])


# (id, equation, operand shapes, output shape, hand MACs, hand-derivation)
# Extents used: b=2, i=3, j=4, k=5, l=6, m=7 unless stated.
_CASES = [
    pytest.param(
        "2op-control-batched-matmul", "bij,bjk->bik",
        [[2, 3, 4], [2, 4, 5]], [2, 3, 5],
        2 * 3 * 4 * 5,
        id="2op-control"),
    pytest.param(
        "3op-batched", "bij,bjk,bkl->bil",
        [[2, 3, 4], [2, 4, 5], [2, 5, 6]], [2, 3, 6],
        # step1 (bij,bjk)->bik: b*i*j*k ; step2 (bik,bkl)->bil: b*i*k*l
        2 * 3 * 4 * 5 + 2 * 3 * 5 * 6,
        id="3op-batched"),
    pytest.param(
        "3op-no-batch", "ij,jk,kl->il",
        [[3, 4], [4, 5], [5, 6]], [3, 6],
        3 * 4 * 5 + 3 * 5 * 6,
        id="3op-no-batch"),
    pytest.param(
        "4op-no-batch", "ij,jk,kl,lm->im",
        [[3, 4], [4, 5], [5, 6], [6, 7]], [3, 7],
        3 * 4 * 5 + 3 * 5 * 6 + 3 * 6 * 7,
        id="4op-no-batch"),
    pytest.param(
        # SSD-shaped 4-operand with batch dims; b=2,c=2,l=3,h=2,n=4,s=3,p=5
        "4op-batched-ssd", "bclhn,bcshn,bhcls,bcshp->bclhp",
        [[2, 2, 3, 2, 4], [2, 2, 3, 2, 4], [2, 2, 2, 3, 3], [2, 2, 3, 2, 5]], [2, 2, 3, 2, 5],
        # step1 (bclhn,bcshn): b*c*l*h*n*s -> keeps bclhs (n summed: MACs)
        # step2 (bclhs,bhcls): b*c*l*h*s     -> keeps bclhs (nothing summed: 72 scalar multiplies, not MACs)
        # step3 (bclhs,bcshp): b*c*l*h*s*p   -> bclhp (s summed: MACs)
        (2 * 2 * 3 * 2 * 4 * 3) + (2 * 2 * 3 * 2 * 3 * 5),
        id="4op-batched-ssd"),
    pytest.param(
        # repeated index inside one operand (diagonal): ii has index set {i}
        "repeated-diag-3op", "ii,ij,jk->k",
        [[3, 3], [3, 4], [4, 5]], [5],
        # step1 (ii,ij): i*j -> keeps j ; step2 (j,jk): j*k
        3 * 4 + 4 * 5,
        id="repeated-index-3op"),
    pytest.param(
        "repeated-diag-2op", "bii,bij->bj",
        [[2, 3, 3], [2, 3, 4]], [2, 4],
        2 * 3 * 4,
        id="repeated-index-2op"),
    pytest.param(
        "3op-dead-index-summed", "ij,k,jl->il",
        [[3, 4], [5], [4, 6]], [3, 6],
        # step1 (ij,k): i*j*k -> keeps ij (k dead) ; step2 (ij,jl): i*j*l
        3 * 4 * 5 + 3 * 4 * 6,
        id="3op-dead-index"),
    pytest.param(
        "1op-transpose", "ij->ji",
        [[3, 4]], [4, 3],
        3 * 4,
        id="1op-control"),
]


class TestPairwiseChainCost:
    @pytest.mark.parametrize("name,eq,shapes,out,expected", _CASES)
    def test_handler_cost(self, name, eq, shapes, out, expected):
        assert TorchEinsumHandler is not None, "einsum handler missing (base commit)"
        ts = _ts(shapes, out)
        op = TorchEinsumHandler().generate_einsum("einsum", ts, equation=eq)
        assert op.get_compute_cost(ts) == expected
        assert op.is_real_einsum and op.elementwise_op == "mul" and op.reduction_op == "add"

    @pytest.mark.parametrize("name,eq,shapes,out,expected", _CASES)
    def test_analyzer_cost_with_equation(self, name, eq, shapes, out, expected):
        """The graph-analyzer path: get_compute_cost(op_type, ts, equation=...)."""
        ts = _ts(shapes, out)
        assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation=eq) == expected

    @pytest.mark.parametrize("name,eq,shapes,out,expected",
                             [c for c in _CASES if c.values[1].count(",") == 1
                              and "ii" not in c.values[1]])
    def test_two_operand_bit_identical_to_single_contraction_formula(self, name, eq, shapes, out, expected):
        """For two operands the chain is one step: product of all unique indices,
        exactly `_macs_from_equation` (the DSL-kernel formula)."""
        ts = _ts(shapes, out)
        assert _macs_from_equation(eq.upper(), shapes) == expected
        assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation=eq) == _macs_from_equation(eq.upper(), shapes)

    def test_two_operand_matches_matmul_handler(self):
        ts = _ts([[2, 3, 4], [2, 4, 5]], [2, 3, 5])
        assert EinsumAnalyzer().get_compute_cost("bmm", ts) == 2 * 3 * 4 * 5
        assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation="bij,bjk->bik") == 2 * 3 * 4 * 5

    def test_solar_format_tokens_with_digits(self):
        """Renamed equations carry tokens like S1/Q1/U0; cost must parse them."""
        ts = _ts([[2, 3, 4], [2, 4, 5], [2, 5, 6]], [2, 3, 6])
        eq = "B1I1J1,B1J1K1,B1K1L1->B1I1L1"
        assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation=eq) == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6

    def test_ellipsis_and_implicit_output(self):
        assert TorchEinsumHandler is not None
        ts = _ts([[2, 3, 4], [2, 4, 5]], [2, 3, 5])
        op = TorchEinsumHandler().generate_einsum("einsum", ts, equation="...ij,...jk->...ik")
        assert op.get_compute_cost(ts) == 2 * 3 * 4 * 5
        ts2 = _ts([[3, 4], [4, 5]], [3, 5])
        op2 = TorchEinsumHandler().generate_einsum("einsum", ts2, equation="ij,jk")
        assert op2.get_compute_cost(ts2) == 3 * 4 * 5

    def test_missing_operand_shapes_use_fallback(self):
        """When the graph carries fewer shapes than operands, the handler-known
        operand shapes resolve the missing extents."""
        assert TorchEinsumHandler is not None
        full = [[2, 3, 4], [2, 4, 5], [2, 5, 6]]
        truncated = _ts([[2, 4, 5], [2, 5, 6]], [2, 3, 6])  # op0 missing, but shapes not positional
        op = TorchEinsumHandler().generate_einsum(
            "einsum", _ts(full, [2, 3, 6]), equation="bij,bjk,bkl->bil")
        assert op.get_compute_cost(_ts(full, [2, 3, 6])) == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6
        # positional shapes of wrong rank/length fall back to the recorded operand shapes
        assert op.get_compute_cost(TensorShapes(inputs=[], outputs=[[2, 3, 6]])) == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6

    def test_no_equation_keeps_previous_fallback(self):
        """Without an equation nothing changes: elementwise copy, not a real einsum."""
        ts = _ts([[2, 3, 4], [2, 4, 5]], [2, 3, 5])
        op = EinsumAnalyzer().get_einsum_op("einsum", ts)
        assert op.equation == "ABC->ABC"
        assert op.is_real_einsum is False
        assert op.get_compute_cost(ts) == 2 * 3 * 4


class TestMamba2SSDHandExpectation:
    """KernelBench level3/48 Mamba2ReturnY: b=2048, c=2, l=s=64, h=8, n=16, p=64, z=c+1=3."""

    b, c, l, h, n, s, p, z = 2048, 2, 64, 8, 16, 64, 64, 3
    # operands in torch order, from raw_attributes of the saved graph
    E0 = ("bclhn,bcshn,bhcls,bcshp->bclhp",
          [[2048, 2, 64, 8, 16], [2048, 2, 64, 8, 16], [2048, 8, 2, 64, 64], [2048, 2, 64, 8, 64]],
          [2048, 2, 64, 8, 64])
    E1 = ("bclhn,bhcl,bclhp->bchpn",
          [[2048, 2, 64, 8, 16], [2048, 8, 2, 64], [2048, 2, 64, 8, 64]], [2048, 2, 8, 64, 16])
    E2 = ("bhzc,bchpn->bzhpn", [[2048, 8, 3, 3], [2048, 3, 8, 64, 16]], [2048, 3, 8, 64, 16])
    E3 = ("bclhn,bchpn,bhcl->bclhp",
          [[2048, 2, 64, 8, 16], [2048, 2, 8, 64, 16], [2048, 8, 2, 64]], [2048, 2, 64, 8, 64])

    def hand(self):
        """Reduction MACs per layer under the left-to-right schedule (a step counts as
        multiply-accumulate only when it sums an index away)."""
        b, c, l, h, n, s, p, z = self.b, self.c, self.l, self.h, self.n, self.s, self.p, self.z
        return {
            # E0: (bclhn,bcshn)->bclhs sums n ; (bclhs,bhcls)->bclhs sums nothing (scalar) ; (bclhs,bcshp)->bclhp sums s
            "E0": b * c * l * h * n * s + b * c * l * h * s * p,
            # E1: (bclhn,bhcl)->bclhn sums nothing (scalar) ; (bclhn,bclhp)->bchpn sums l
            "E1": b * c * l * h * n * p,
            # E2: single contraction over c: b*h*z*c*p*n  (z = c+1 chunks after cat)
            "E2": b * h * z * self.z * p * n,
            # E3: (bclhn,bchpn)->bclhp sums n ; (bclhp,bhcl)->bclhp sums nothing (scalar)
            "E3": b * c * l * h * n * p,
        }

    def hand_scalar(self):
        """Elementwise multiplies of the steps that sum nothing away (the mask / decay factors)."""
        b, c, l, h, n, s, p = self.b, self.c, self.l, self.h, self.n, self.s, self.p
        return {"E0": b * c * l * h * s, "E1": b * c * l * h * n, "E2": 0, "E3": b * c * l * h * p}

    def test_hand_values(self):
        hv, sv = self.hand(), self.hand_scalar()
        assert hv == {"E0": 10_737_418_240, "E1": 2_147_483_648, "E2": 150_994_944, "E3": 2_147_483_648}
        assert sv == {"E0": 134_217_728, "E1": 33_554_432, "E2": 0, "E3": 134_217_728}
        assert sum(hv.values()) == 15_183_380_480          # the SOL self-audit hand MAC value, same units
        assert sum(sv.values()) == 301_989_888
        assert sum(hv.values()) + sum(sv.values()) == 15_485_370_368   # the previously reported mixed count

    @pytest.mark.parametrize("key", ["E0", "E1", "E2", "E3"])
    def test_full_shapes(self, key):
        eq, shapes, out = getattr(self, key)
        ts = _ts(shapes, out)
        assert EinsumAnalyzer().get_compute_cost("einsum", ts, equation=eq) == self.hand()[key]
        assert TorchEinsumHandler().generate_einsum("einsum", ts, equation=eq).scalar_multiplies(ts) == self.hand_scalar()[key]

    def test_schedule_is_part_of_the_value(self):
        """The order [bclhn, bcshp, bcshn, bhcls] costs 283,467,841,536 under the same index-product rule:
        'any order gives the same' is false; the hand value is a declared left-to-right schedule."""
        from solar.einsum.ops.einsum_ops import pairwise_chain_cost
        ext = {"b": self.b, "c": self.c, "l": self.l, "h": self.h, "n": self.n, "s": self.s, "p": self.p}
        macs, scalar = pairwise_chain_cost([list("bclhn"), list("bcshp"), list("bcshn"), list("bhcls")], list("bclhp"), ext)
        assert macs + scalar == 283_467_841_536

    def test_saved_graph_values_were_first_operand_numel(self):
        """Documents the base defect: with no einsum handler the generic fallback
        counted numel(first captured operand)."""
        saved = {"E0": ([[2048, 8, 2, 64, 64], [2048, 2, 64, 8, 64]], 134_217_728),
                 "E1": ([[2048, 8, 2, 64], [2048, 2, 64, 8, 64]], 2_097_152),
                 "E2": ([[2048, 8, 3, 3], [2048, 3, 8, 64, 16]], 147_456),
                 "E3": ([[2048, 2, 8, 64, 16], [2048, 8, 2, 64]], 33_554_432)}
        for key, (shapes, macs) in saved.items():
            assert math.prod(shapes[0]) == macs


class TestConverterCompletesEinsumOperands:
    """pytorch_to_einsum: operands that torchview did not capture as inputs
    (here parameter-derived tensors) are restored from raw_attributes."""

    def _convert(self):
        import networkx as nx
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph()
        for n in ("Model.exp_1", "Model.reshape", "Model.einsum"):
            op_graph.add_node(n)
        op_graph.add_edge("Model.exp_1", "Model.einsum")
        op_graph.add_edge("Model.reshape", "Model.einsum")
        node_data = {
            "type": "einsum",
            "input_shapes": [[2, 2, 2, 3, 3], [2, 2, 3, 2, 5]],
            "output_shapes": [[2, 2, 3, 2, 5]],
            "input_types": ["input", "input"],
            "input_dtypes": ["torch.float32", "torch.float32"],
            "output_types": ["output"],
            "connections": {"inputs": ["Model.hidden-tensor_89", "Model.hidden-tensor_71"], "outputs": []},
            "module_args": {
                "function_name": "einsum",
                "raw_attributes": ("[['bclhn,bcshn,bhcls,bcshp->bclhp', "
                                   "Tensor(shape=(2, 2, 3, 2, 4), dtype=torch.float32), "
                                   "Tensor(shape=(2, 2, 3, 2, 4), dtype=torch.float32), "
                                   "Tensor(shape=(2, 2, 2, 3, 3), dtype=torch.float32), "
                                   "Tensor(shape=(2, 2, 3, 2, 5), dtype=torch.float32)], {}]"),
            },
        }
        return converter._convert_operation(
            node_id="Model.einsum", node_data=node_data, op_graph=op_graph,
            start_nodes_info=[], start_node_id_map={})

    def test_shapes_types_names(self):
        """Issue #46 / INTEG-2: the two operands without a producer are typed
        'unresolved' (never defaulted to weight), named, listed in the marker and the
        layer is incomplete; the connected operands keep their edges."""
        out = self._convert()
        assert out["einsum_equation"] == "BCLHN,BCSHN,BHCLS,BCSHP->BCLHP"
        assert out["tensor_shapes"]["inputs"] == [[2, 2, 3, 2, 4], [2, 2, 3, 2, 4], [2, 2, 2, 3, 3], [2, 2, 3, 2, 5]]
        assert out["tensor_types"]["inputs"] == ["unresolved", "unresolved", "input", "input"]
        assert out["tensor_names"]["inputs"] == [
            "Model.einsum.Unresolved", "Model.einsum.Unresolved_1", "Model.exp_1.Output", "Model.reshape.Output"]
        # activation connections are unchanged (unresolved operands are not graph layers)
        assert out["connections"]["inputs"] == ["Model.exp_1", "Model.reshape"]
        assert out["tensor_dtypes"]["inputs"] == ["torch.float32"] * 4
        assert out["status"] == "incomplete" and out["einsum_operands_unresolved"] == [0, 1]
        # every operand is a role of its own (INTEG-5): the AF builder maps roles to shapes positionally
        assert [r for r in out["operands"] if r != "Output"] == ["Input", "Input_1", "Input_2", "Input_3"]
        assert "weight" not in out["tensor_types"]["inputs"]

    def test_two_operand_einsum_unchanged(self):
        import networkx as nx
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph()
        op_graph.add_nodes_from(["Model.a", "Model.b", "Model.einsum_2"])
        op_graph.add_edge("Model.a", "Model.einsum_2"); op_graph.add_edge("Model.b", "Model.einsum_2")
        node_data = {
            "type": "einsum", "input_shapes": [[2, 8, 3, 3], [2, 3, 8, 4, 5]], "output_shapes": [[2, 3, 8, 4, 5]],
            "input_types": ["input", "input"], "output_types": ["output"],
            "connections": {"inputs": ["Model.h1", "Model.h2"], "outputs": []},
            "module_args": {"raw_attributes": "[['bhzc,bchpn->bzhpn', Tensor(shape=(2, 8, 3, 3), dtype=torch.float32), Tensor(shape=(2, 3, 8, 4, 5), dtype=torch.float32)], {}]"},
        }
        out = converter._convert_operation("Model.einsum_2", node_data, op_graph, [], {})
        assert out["tensor_shapes"]["inputs"] == [[2, 8, 3, 3], [2, 3, 8, 4, 5]]
        assert out["tensor_types"]["inputs"] == ["input", "input"]
        assert out["tensor_names"]["inputs"] == ["Model.a.Output", "Model.b.Output"]


class TestGraphAnalyzerEndToEnd:
    """analysis.yaml macs for a 3-operand einsum layer with complete shapes."""

    def test_graph_macs(self, tmp_path):
        layers = {
            "start": {"type": "start", "einsum_equation": "->ABC", "elementwise_op": "copy",
                      "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                      "tensor_names": {"inputs": [], "outputs": ["start.Output"]},
                      "tensor_types": {"inputs": [], "outputs": ["input"]},
                      "tensor_shapes": {"inputs": [], "outputs": [[2, 3, 4]]},
                      "connections": {"inputs": [], "outputs": ["Model.einsum"]}},
            "Model.einsum": {"type": "einsum", "einsum_equation": "B1I1J1,B1J1K1,B1K1L1->B1I1L1",
                             "elementwise_op": "mul", "reduction_op": "add", "is_real_einsum": True,
                             "is_einsum_supportable": True,
                             "tensor_names": {"inputs": ["start.Output", "Model.einsum.Weight", "Model.einsum.Weight_1"],
                                              "outputs": ["Model.einsum.Output"]},
                             "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
                             "tensor_shapes": {"inputs": [[2, 3, 4], [2, 4, 5], [2, 5, 6]], "outputs": [[2, 3, 6]]},
                             "connections": {"inputs": ["start"], "outputs": []}},
        }
        p = tmp_path / "einsum_graph_renamed.yaml"
        with open(p, "w") as f:
            yaml.safe_dump({"model_name": "t", "layers": layers}, f, sort_keys=False)
        res = EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "analysis", copy_graph=False)
        assert res is not None
        layer = res["layers"]["Model.einsum"]
        assert layer["macs"] == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6
        # traffic is shape-driven and independent of the cost path
        assert layer["unfused_elements"] == 2 * 3 * 4 + 2 * 4 * 5 + 2 * 5 * 6 + 2 * 3 * 6


_ATTR_REQUIREMENT = (
    "the capture must record call attributes for FunctionNodes: torchview with upstream #128 "
    "(collect_attributes propagated to input/hidden recorder tensors) or the torchview 0.2.7 "
    "release plus solar/patches/torchview-collect-attributes-inputs.patch as applied by "
    "validate-install; without them neither the torch.einsum equation nor its uncaptured "
    "operands can be recovered"
)


class TestEndToEndCaptureRequirement:
    """Environment requirement stated explicitly, and the operand restoration
    exercised through the real capture (tiny CPU trace, as in
    test_graph_analyzer_regression.py)."""

    SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(2, 4, 5))
            self.w2 = nn.Parameter(torch.randn(2, 5, 6))

        def forward(self, x):
            return torch.einsum("bij,bjk,bkl->bil", x, self.w1, self.w2)

    def get_inputs():
        return [torch.randn(2, 3, 4)]

    def get_init_inputs():
        return []
    """

    def _pipeline(self, tmp_path):
        from textwrap import dedent
        from solar.common.types import ProcessingConfig
        from solar.graph import PyTorchProcessor
        (tmp_path / "model.py").write_text(dedent(self.SRC))
        for sub in ("graph", "einsum", "analysis"):
            (tmp_path / sub).mkdir()
        assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)) \
            .process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
        assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
        analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"),
                                                       str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        graph = yaml.safe_load(open(tmp_path / "graph/pytorch_graph.yaml"))["layers"]
        einsum = yaml.safe_load(open(tmp_path / "einsum/einsum_graph.yaml"))["layers"]
        return graph, einsum, analysis

    def test_capture_records_call_attributes_and_einsum_is_costed(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("torchview")
        graph, einsum, analysis = self._pipeline(tmp_path)
        gnode = graph["Model.einsum"]
        assert gnode["module_args"].get("raw_attributes"), _ATTR_REQUIREMENT
        enode = einsum["Model.einsum"]
        assert "einsum_degraded" not in enode, _ATTR_REQUIREMENT
        assert enode["einsum_equation"] == "BIJ,BJK,BKL->BIL"
        assert enode["is_real_einsum"] is True
        assert enode["tensor_shapes"]["inputs"] == [[2, 3, 4], [2, 4, 5], [2, 5, 6]]
        layer = analysis["layers"]["Model.einsum"]
        assert layer["macs"] == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6      # left-to-right chain
        assert layer["other_ops"] == 0


def _assert_reads(reads, *, graph, layer, expected_preds, unresolved_positions):
    """AF reads of an einsum with unresolved operands, asserted by identity, order,
    provenance and coverage rather than by the external tensors' names (independent review
    P2-4): the integrated AF builder (an earlier change R7) names them ``<layer>_Unresolved[_k]``, the
    standalone one ``W<n>``; both are external tensors with no producer in the graph."""
    def san(n):
        return n.replace(".", "_").replace("-", "_")
    producers = {san(name) for name in graph["layers"]}
    n_slots = len(expected_preds) + len(unresolved_positions)
    assert len(reads) == n_slots, reads
    assert len(set(reads)) == n_slots, ("every operand is a distinct read", reads)
    pred_iter = iter(expected_preds)
    for i, r in enumerate(reads):
        if i in unresolved_positions:
            # an unresolved operand is an external tensor: read, but produced by no layer and never a weight relabel
            assert r not in producers, (i, r)
            assert not any(r.startswith(san(p)) and r != san(p) for p in expected_preds) or "_Unresolved" in r or r.startswith("W"), (i, r)
        else:
            expected = san(next(pred_iter))
            assert r == expected or r.startswith(expected + "_Output") or r.startswith(expected), (i, r, expected)   # multi-output slot suffixes allowed
    # complete predecessor coverage: every connected (non-weight) predecessor is read exactly once
    for p in expected_preds:
        assert sum(1 for r in reads if r == san(p) or r.startswith(san(p) + "_Output")) == 1, (p, reads)


class TestUnresolvedOperandProvenance:
    """Issue #46 (INTEG-2): an einsum operand's type comes from graph provenance only; an
    operand that cannot be resolved to a producer is 'unresolved' with a machine-readable
    marker, the layer incomplete through analysis and perf, and the MACs still the chain."""

    def test_reviewer_probe_middle_operand_unbound(self):
        """bij,bjk,bkl->bil with the first and third operands connected (independent review)."""
        import networkx as nx
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph(); op_graph.add_nodes_from(["Model.a", "Model.c", "Model.einsum"])
        op_graph.add_edge("Model.a", "Model.einsum"); op_graph.add_edge("Model.c", "Model.einsum")
        node_data = {"type": "einsum", "input_shapes": [[2, 3, 4], [2, 5, 6]], "output_shapes": [[2, 3, 6]],
                     "input_types": ["input", "input"], "input_dtypes": ["torch.float32"] * 2, "output_types": ["output"],
                     "connections": {"inputs": ["Model.a", "Model.c"], "outputs": []},
                     "module_args": {"function_name": "einsum", "raw_attributes": "[['bij,bjk,bkl->bil', Tensor(shape=(2, 3, 4), dtype=torch.float32), Tensor(shape=(2, 4, 5), dtype=torch.float32), Tensor(shape=(2, 5, 6), dtype=torch.float32)], {}]"}}
        out = converter._convert_operation("Model.einsum", node_data, op_graph, [], {})
        assert out["tensor_types"]["inputs"] == ["input", "unresolved", "input"]
        assert out["tensor_names"]["inputs"] == ["Model.a.Output", "Model.einsum.Unresolved", "Model.c.Output"]
        assert out["status"] == "incomplete" and out["einsum_operands_unresolved"] == [1]
        assert out["connections"]["inputs"] == ["Model.a", "Model.c"]

    def test_connected_operands_only_carry_no_marker(self):
        import networkx as nx
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph(); op_graph.add_nodes_from(["Model.a", "Model.b", "Model.c", "Model.einsum"])
        for n in ("Model.a", "Model.b", "Model.c"): op_graph.add_edge(n, "Model.einsum")
        node_data = {"type": "einsum", "input_shapes": [[2, 3, 4], [2, 4, 5], [2, 5, 6]], "output_shapes": [[2, 3, 6]],
                     "input_types": ["input", "input", "input"], "output_types": ["output"],
                     "connections": {"inputs": ["Model.a", "Model.b", "Model.c"], "outputs": []},
                     "module_args": {"raw_attributes": "[['bij,bjk,bkl->bil', Tensor(shape=(2, 3, 4), dtype=torch.float32), Tensor(shape=(2, 4, 5), dtype=torch.float32), Tensor(shape=(2, 5, 6), dtype=torch.float32)], {}]"}}
        out = converter._convert_operation("Model.einsum", node_data, op_graph, [], {})
        assert out["tensor_types"]["inputs"] == ["input"] * 3 and "status" not in out and "einsum_operands_unresolved" not in out

    @staticmethod
    def _mamba2_case():
        """The four-operand SSD einsum with two captured operands (integrated-dryrun-v2 INTEG-5 /
        first dry run INTEG-2), converted and embedded in a two-producer graph."""
        import networkx as nx
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph(); op_graph.add_nodes_from(["Model.exp_1", "Model.reshape", "Model.einsum"])
        op_graph.add_edge("Model.exp_1", "Model.einsum"); op_graph.add_edge("Model.reshape", "Model.einsum")
        shapes = [[2, 2, 3, 2, 4], [2, 2, 3, 2, 4], [2, 2, 2, 3, 3], [2, 2, 3, 2, 5]]
        node_data = {"type": "einsum", "input_shapes": [shapes[2], shapes[3]], "output_shapes": [[2, 2, 3, 2, 5]],
                     "input_types": ["input", "input"], "input_dtypes": ["torch.float32"] * 2, "output_types": ["output"],
                     "connections": {"inputs": ["Model.exp_1", "Model.reshape"], "outputs": []},
                     "module_args": {"function_name": "einsum", "raw_attributes": "[['bclhn,bcshn,bhcls,bcshp->bclhp', Tensor(shape=(2, 2, 3, 2, 4), dtype=torch.float32), Tensor(shape=(2, 2, 3, 2, 4), dtype=torch.float32), Tensor(shape=(2, 2, 2, 3, 3), dtype=torch.float32), Tensor(shape=(2, 2, 3, 2, 5), dtype=torch.float32)], {}]"}}
        layer = converter._convert_operation("Model.einsum", node_data, op_graph, [], {})
        def src(name, shape):
            return {"type": "start", "einsum_equation": "->ABCDE", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                    "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "tensor_shapes": {"inputs": [], "outputs": [shape]}, "operands": {"Output": list("ABCDE")}, "connections": {"inputs": [], "outputs": ["Model.einsum"]}}
        layers = {"Model.exp_1": src("Model.exp_1", shapes[2]), "Model.reshape": src("Model.reshape", shapes[3]),
                  "Model.einsum": {**layer, "connections": {"inputs": ["Model.exp_1", "Model.reshape"], "outputs": []}}}
        return {"model_name": "mamba2-case", "layers": layers}

    def test_mamba2_dry_run_case_is_incomplete_not_weighted(self, tmp_path):
        """Nothing is typed weight, the marker reaches analysis and perf, the MACs are the chain."""
        import yaml
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        from solar.perf.perf_model import EinsumGraphPerfModel
        graph = self._mamba2_case(); layer = graph["layers"]["Model.einsum"]
        assert layer["tensor_types"]["inputs"] == ["unresolved", "unresolved", "input", "input"]
        assert [r for r in layer["operands"] if r != "Output"] == ["Input", "Input_1", "Input_2", "Input_3"]
        path = tmp_path / "einsum_graph_renamed.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(graph, f, sort_keys=False)
        analysis = EinsumGraphAnalyzer().analyze_graph(str(path), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        a = analysis["layers"]["Model.einsum"]
        # left-to-right chain, hand arithmetic on these extents (b2 c2 l3 h2 n4 s3 p5):
        #   step 1 (bclhn,bcshn): unique b,c,l,h,n,s = 2*2*3*2*4*3 = 288, n summed away -> MACs ; keep bclhs
        #   step 2 (bclhs,bhcls): unique b,c,l,h,s = 2*2*3*2*3 = 72, nothing summed -> scalar multiplies
        #   step 3 (bclhs,bcshp): unique b,c,l,h,s,p = 2*2*3*2*3*5 = 360, s summed away -> MACs
        assert a["macs"] == 288 + 360
        assert a["scalar_multiplies"] == 72 and a["other_ops"] == 72
        assert a["status"] == "incomplete" and a["einsum_operands_unresolved"] == [0, 1]
        assert analysis["total"]["status"] == "incomplete" and analysis["total"]["incomplete_layers"] == {"Model.einsum": "einsum_operands_unresolved"}
        perf = EinsumGraphPerfModel().predict(analysis_path=str(tmp_path / "analysis" / "analysis.yaml"), output_dir=str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
        assert perf["workload"]["status"] == "incomplete" and perf["workload"]["incomplete_layers"] == {"Model.einsum": "einsum_operands_unresolved"}

    def test_mamba2_dry_run_case_through_af_builder(self):
        """Unresolved operands BEFORE the connected ones: the AF builder pairs the two real
        predecessors with the two connected roles (union and emit agree) and emits four reads
        (independent review P2-1; the earlier strict xfail misattributed this to the lineage)."""
        from solar.einsum.af_graph_builder import build_af_graph_from_dict
        af = build_af_graph_from_dict(self._mamba2_case())
        einsums = af["workload"]["einsums"] if isinstance(af.get("workload"), dict) else af["einsums"]
        e = next(e for e in einsums if e["name"].endswith("einsum"))
        reads = [ta["name"] for ta in e["tensor_accesses"] if not ta.get("output")]
        _assert_reads(reads, graph=self._mamba2_case(), layer="Model.einsum", expected_preds=["Model.exp_1", "Model.reshape"], unresolved_positions=[0, 1])

    def test_unresolved_operand_between_connected_operands_through_af_builder(self):
        """bij,bjk,bkl->bil with the middle operand unbound: reads a, W, c in that order."""
        import networkx as nx
        from solar.einsum.af_graph_builder import build_af_graph_from_dict
        converter = PyTorchToEinsum()
        op_graph = nx.DiGraph(); op_graph.add_nodes_from(["Model.a", "Model.c", "Model.einsum"])
        op_graph.add_edge("Model.a", "Model.einsum"); op_graph.add_edge("Model.c", "Model.einsum")
        node_data = {"type": "einsum", "input_shapes": [[2, 3, 4], [2, 5, 6]], "output_shapes": [[2, 3, 6]],
                     "input_types": ["input", "input"], "input_dtypes": ["torch.float32"] * 2, "output_types": ["output"],
                     "connections": {"inputs": ["Model.a", "Model.c"], "outputs": []},
                     "module_args": {"function_name": "einsum", "raw_attributes": "[['bij,bjk,bkl->bil', Tensor(shape=(2, 3, 4), dtype=torch.float32), Tensor(shape=(2, 4, 5), dtype=torch.float32), Tensor(shape=(2, 5, 6), dtype=torch.float32)], {}]"}}
        layer = converter._convert_operation("Model.einsum", node_data, op_graph, [], {})
        assert layer["tensor_types"]["inputs"] == ["input", "unresolved", "input"]
        def src(name, shape):
            return {"type": "start", "einsum_equation": "->ABC", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                    "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "tensor_shapes": {"inputs": [], "outputs": [shape]}, "operands": {"Output": list("ABC")}, "connections": {"inputs": [], "outputs": ["Model.einsum"]}}
        layers = {"Model.a": src("Model.a", [2, 3, 4]), "Model.c": src("Model.c", [2, 5, 6]), "Model.einsum": {**layer, "connections": {"inputs": ["Model.a", "Model.c"], "outputs": []}}}
        af = build_af_graph_from_dict({"model_name": "mid-unresolved", "layers": layers})
        einsums = af["workload"]["einsums"] if isinstance(af.get("workload"), dict) else af["einsums"]
        e = next(e for e in einsums if e["name"].endswith("einsum"))
        reads = [ta["name"] for ta in e["tensor_accesses"] if not ta.get("output")]
        _assert_reads(reads, graph={"model_name": "mid-unresolved", "layers": layers}, layer="Model.einsum", expected_preds=["Model.a", "Model.c"], unresolved_positions=[1])

    def test_untraced_view_operand_end_to_end(self, tmp_path):
        """A parameter VIEW (self.w.t()) is not traced by torchview: the operand has no producer,
        so it is 'unresolved' (not a weight) and the model is reported incomplete through perf,
        while the MACs are the chain and the conversion passes the AF coverage check."""
        from textwrap import dedent
        import yaml
        from solar.common.types import ProcessingConfig
        from solar.graph import PyTorchProcessor
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        from solar.perf.perf_model import EinsumGraphPerfModel
        (tmp_path / "model.py").write_text(dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = nn.Parameter(torch.randn(5, 4))
                    self.v = nn.Parameter(torch.randn(5, 6))

                def forward(self, x):
                    return torch.einsum('bij,jk,kl->bil', x, self.w.t(), self.v)

            def get_inputs():
                return [torch.randn(2, 3, 4)]

            def get_init_inputs():
                return []
            """))
        for n in ("graph", "einsum", "analysis"):
            (tmp_path / n).mkdir()
        assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
        assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
        e = yaml.safe_load((tmp_path / "einsum/einsum_graph.yaml").read_text())["layers"]
        k, layer = next((k, v) for k, v in e.items() if v["type"] == "einsum")
        types = layer["tensor_types"]["inputs"]
        assert len(types) == 3 and types[0] == "input" and "weight" in types or "unresolved" in types
        if "unresolved" in types:
            assert layer["status"] == "incomplete" and layer["einsum_operands_unresolved"] == [i for i, t in enumerate(types) if t == "unresolved"]
        analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        assert analysis["layers"][k]["macs"] == 2 * 3 * 4 * 5 + 2 * 3 * 5 * 6     # left-to-right chain
        if "unresolved" in types:
            assert analysis["total"]["status"] == "incomplete"
            perf = EinsumGraphPerfModel().predict(analysis_path=str(tmp_path / "analysis" / "analysis.yaml"), output_dir=str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
            assert perf["workload"]["status"] == "incomplete"
        else:
            assert "status" not in analysis["total"]


class TestChainUnits:
    """independent review P2-2: reduction MACs and scalar multiplies are separate units."""

    def test_elementwise_steps_are_scalar_multiplies_not_macs(self):
        from solar.einsum.ops.einsum_ops import pairwise_chain_cost
        # i,i->i: one step, nothing summed -> 3 scalar multiplies, 0 MACs
        assert pairwise_chain_cost([["i"], ["i"]], ["i"], {"i": 3}) == (0, 3)
        # bij,bjk->bik: j summed -> MACs only
        assert pairwise_chain_cost([list("bij"), list("bjk")], list("bik"), {"b": 2, "i": 3, "j": 4, "k": 5}) == (2 * 3 * 4 * 5, 0)
        # Mamba2 SSD chain at the KernelBench extents (b2048 c2 l64 h8 n16 s64 p64), left to right:
        ext = {"b": 2048, "c": 2, "l": 64, "h": 8, "n": 16, "s": 64, "p": 64}
        macs, scalar = pairwise_chain_cost([list("bclhn"), list("bcshn"), list("bhcls"), list("bcshp")], list("bclhp"), ext)
        assert (macs, scalar) == (2_147_483_648 + 8_589_934_592, 134_217_728)
        macs1, scalar1 = pairwise_chain_cost([list("bclhn"), list("bhcl"), list("bclhp")], list("bchpn"), ext)
        assert (macs1, scalar1) == (2_147_483_648, 33_554_432)
        macs3, scalar3 = pairwise_chain_cost([list("bclhn"), list("bchpn"), list("bhcl")], list("bclhp"), ext)
        assert (macs3, scalar3) == (2_147_483_648, 134_217_728)
        # the four-layer row: 15,183,380,480 reduction MACs + 301,989,888 scalar multiplies (= the hand MAC value)
        assert macs + macs1 + 150_994_944 + macs3 == 15_183_380_480 and scalar + scalar1 + scalar3 == 301_989_888

    def test_handler_reports_both_units(self):
        from solar.einsum.ops.einsum_ops import TorchEinsumHandler
        ts = _ts([[3], [3]], [3])
        op = TorchEinsumHandler().generate_einsum("einsum", ts, equation="i,i->i")
        assert op.get_compute_cost(ts) == 0 and op.scalar_multiplies(ts) == 3


class TestStoredGraphScalarBilling:
    """independent review P2-3: a stored einsum graph may carry an explicit einsum with
    is_real_einsum false (older graphs); its scalar work must still be billed, and perf must consume it."""

    @staticmethod
    def _graph(equation, in_shapes, out_shape, flag):
        def src(name, shape):
            labels = "ABCDE"[: len(shape)]
            return {"type": "start", "einsum_equation": f"->{labels}", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                    "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                    "tensor_shapes": {"inputs": [], "outputs": [shape]}, "operands": {"Output": list(labels)}, "connections": {"inputs": [], "outputs": ["Model.einsum"]}}
        names = [f"Model.in{i}" for i in range(len(in_shapes))]
        layers = {n: src(n, sh) for n, sh in zip(names, in_shapes)}
        lhs = equation.split("->")[0].split(",")
        operands = {"Input": list(lhs[0]), "Output": list(equation.split("->")[1])}
        if len(lhs) == 2:
            operands["Weight"] = list(lhs[1])
        else:
            for i, part in enumerate(lhs[1:], 1):
                operands[f"Input_{i}"] = list(part)
        layers["Model.einsum"] = {"type": "einsum", "einsum_equation": equation.upper(), "elementwise_op": "mul", "reduction_op": "add", "is_real_einsum": flag, "is_einsum_supportable": True,
                                  "tensor_names": {"inputs": [f"{n}.Output" for n in names], "outputs": ["Model.einsum.Output"]}, "tensor_types": {"inputs": ["input"] * len(names), "outputs": ["output"]},
                                  "tensor_shapes": {"inputs": in_shapes, "outputs": [out_shape]}, "tensor_dtypes": {"inputs": ["torch.float32"] * len(names), "outputs": ["torch.float32"]},
                                  "operands": {k: [c.upper() for c in v] for k, v in operands.items()}, "connections": {"inputs": names, "outputs": []}}
        return {"model_name": "stored", "layers": layers}

    @pytest.mark.parametrize("equation,in_shapes,out_shape,macs,scalar", [
        ("i,i->i", [[3], [3]], [3], 0, 3),                                   # elementwise: all work is scalar
        ("ij,jk,ik->ik", [[3, 4], [4, 5], [3, 5]], [3, 5], 60, 15),           # 60 reduction (j summed) + 15 elementwise
    ])
    @pytest.mark.parametrize("flag", [True, False], ids=["real", "stored-false-flag"])
    def test_scalar_work_is_billed_on_both_paths_and_reaches_perf(self, tmp_path, equation, in_shapes, out_shape, macs, scalar, flag):
        import yaml
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        from solar.perf.perf_model import EinsumGraphPerfModel
        path = tmp_path / "einsum_graph_renamed.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(self._graph(equation, in_shapes, out_shape, flag), f, sort_keys=False)
        analysis = EinsumGraphAnalyzer().analyze_graph(str(path), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        a = analysis["layers"]["Model.einsum"]
        if flag:
            assert (a["macs"], a["other_ops"]) == (macs, scalar)
        else:   # non-real: everything is CUDA-core work, reduction and elementwise steps alike
            assert (a["macs"], a["other_ops"]) == (0, macs + scalar)
        assert a.get("scalar_multiplies", 0) == scalar
        assert analysis["total"]["other_ops"] == a["other_ops"]
        perf = EinsumGraphPerfModel().predict(analysis_path=str(tmp_path / "analysis" / "analysis.yaml"), output_dir=str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
        assert perf["workload"]["total_other_ops"] == a["other_ops"]
