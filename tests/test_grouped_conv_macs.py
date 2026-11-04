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

"""Regression tests for grouped forward-convolution MAC accounting (issue #42).

For ``nn.Conv{1,2,3}d`` with ``groups = g`` the weight is ``[C_out, C_in/g, *k]``
and every output element costs ``(C_in / g) * prod(kernel)`` MACs:

    MACs = N * C_out * prod(output_spatial) * (C_in / g) * prod(kernel)

Before the fix the generic equation resolver took ``C`` from the *input*
tensor (total ``C_in``), so ``1 < g < C_in`` was over-counted by ``g``x and
a depthwise layer with a channel multiplier (``g == C_in``, ``C_out > C_in``)
by ``C_in``x.  Dense (``g == 1``, 1D/2D/3D) and plain depthwise in 1D/2D
(``g == C_in``, ``C_out == C_in``, dedicated depthwise equation) were already
exact and must stay bit-identical; plain 3D depthwise had no depthwise
equation and is a corrected case (``C_in``x on the base).  The equations are
unchanged, so traffic, renaming and exports do not move (that is not
numerical acceptance of the grouped lowering).  Expected values are hand
arithmetic; torch's flop counter (meta device, shapes only) is the
independent oracle.

``TestCapturedPipeline`` enforces the contract end to end (meta capture ->
converter -> renamed graph -> analyzer).  On the release branch the lineage base
the analyzer priced every real einsum from its stored equation before the
registered handler (equation-first dispatch), so the captured 3D grouped
cases were pinned ``xfail(strict=True)`` (grouped-conv-integration-1).  The
lineage-regression fix (an earlier change, handler precedence, item R2) merged into
the release branch as a lineage repair (target the target) restored handler
precedence; since this branch's reconciliation with that target all twelve
captured cases are required passes (INTEG-11).
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.conv_ops import Conv1dHandler, Conv2dHandler, Conv3dHandler

_H = {1: (Conv1dHandler, "conv1d"), 2: (Conv2dHandler, "conv2d"), 3: (Conv3dHandler, "conv3d")}
C_IN = 8


def _out(l, k, s, p, d):
    return (l + 2 * p - d * (k - 1) - 1) // s + 1


def _case(dims, groups, c_out=C_IN, n=2, spatial=(6, 5, 7), k=3, stride=1, padding=1, dilation=1):
    spatial = list(spatial[:dims])
    out_sp = [_out(l, k, stride, padding, dilation) for l in spatial]
    inp = [n, C_IN] + spatial
    w = [c_out, C_IN // groups] + [k] * dims
    out = [n, c_out] + out_sp
    ts = TensorShapes(inputs=[inp, w], outputs=[out])
    expected = n * c_out * math.prod(out_sp) * (C_IN // groups) * k ** dims
    return ts, expected


_CASES = [pytest.param(dims, g, dict(), id=f"{dims}d-groups{g}") for dims in (1, 2, 3) for g in (1, 2, C_IN // 2, C_IN)]
_CASES += [pytest.param(dims, 2, dict(stride=2, padding=0), id=f"{dims}d-groups2-stride2") for dims in (1, 2, 3)]
_CASES += [pytest.param(dims, C_IN, dict(c_out=2 * C_IN), id=f"{dims}d-depthwise-multiplier2") for dims in (1, 2, 3)]
# Controls that were already exact on the base: dense (groups == 1) in 1D/2D/3D and
# plain depthwise (groups == C_IN, C_out == C_IN) in 1D/2D, where the handlers emit
# the dedicated depthwise equation.  Conv3dHandler has no depthwise branch, so 3D
# depthwise was over-counted by C_IN on the base and is a fixed case, not a control.
_CONTROLS = [c for c in _CASES if not c.values[2].get("c_out")
             and (c.values[1] == 1 or (c.values[1] == C_IN and c.values[0] in (1, 2)))]


def _module_args(groups, kw):
    """What nn.Conv*d module_args carry (the lineage base handlers read groups/channels from them)."""
    return {"groups": groups, "in_channels": C_IN, "out_channels": kw.get("c_out", C_IN)}


class TestGroupedConvMacs:
    @pytest.mark.parametrize("dims,groups,kw", _CASES)
    def test_handler_cost(self, dims, groups, kw):
        cls, op = _H[dims]
        ts, expected = _case(dims, groups, **kw)
        assert cls().generate_einsum(op, ts, stride=[kw.get("stride", 1)] * dims,
                                     padding=[kw.get("padding", 1)] * dims).get_compute_cost(ts) == expected

    @pytest.mark.parametrize("dims,groups,kw", _CASES)
    def test_analyzer_cost(self, dims, groups, kw):
        """graph_analyzer path: EinsumAnalyzer.get_compute_cost(op_type, ts)."""
        _, op = _H[dims]
        ts, expected = _case(dims, groups, **kw)
        assert EinsumAnalyzer().get_compute_cost(op, ts) == expected

    @pytest.mark.parametrize("dims,groups,kw", _CONTROLS)
    def test_dense_and_depthwise_controls_equal_generic_resolver(self, dims, groups, kw):
        """groups == 1 and plain depthwise were exact before: the new cost must equal
        the generic equation product bit for bit.

        Since the release branch the lineage base the handlers select the depthwise
        equation from ``groups``/``in_channels``/``out_channels`` (nn.Conv*
        module_args or the functional call's raw_attributes) instead of the
        weight shape, so the call carries them as an nn.Conv module does."""
        from solar.einsum.ops.base import EinsumOp
        cls, op = _H[dims]
        ts, expected = _case(dims, groups, **kw)
        new_op = cls().generate_einsum(op, ts, module_args=_module_args(groups, kw))
        generic = EinsumOp(operands=new_op.operands, equation=new_op.equation, name=new_op.name)
        assert generic.get_compute_cost(ts) == expected
        assert new_op.get_compute_cost(ts) == generic.get_compute_cost(ts)

    @pytest.mark.parametrize("dims,groups,kw", _CASES)
    def test_equation_unchanged(self, dims, groups, kw):
        """ConvEinsumOp changes only the cost: the equation is whatever the handler
        emits for the call (the lineage base dense / depthwise / group-wise forms)."""
        cls, op = _H[dims]
        ts, _ = _case(dims, groups, **kw)
        eq = cls().generate_einsum(op, ts, module_args=_module_args(groups, kw)).equation
        depthwise = groups == C_IN and not kw.get("c_out")
        expected_eq = {1: ("BO(P+R),OCR->BOP", "BC(P+R),OCR->BOP", "BGI(P+R),GOIR->BGOP"),
                       2: ("BO(P+R)(Q+S),OCRS->BOPQ", "BC(P+R)(Q+S),OCRS->BOPQ", "BGI(P+R)(Q+S),GOIRS->BGOPQ"),
                       3: (None, "BC(P+T)(Q+R)(U+S),OCTRS->BOPQU", None)}[dims]
        if groups == 1:
            assert eq == expected_eq[1]
        elif depthwise and expected_eq[0]:
            assert eq == expected_eq[0]
        elif dims == 3:
            assert eq == expected_eq[1]          # Conv3dHandler still emits the dense form for every call
        else:
            assert eq == expected_eq[2]

    def test_groups_ratio_documents_defect(self):
        ts, expected = _case(2, 2)
        dense_c_formula = 2 * C_IN * 6 * 5 * C_IN * 9       # what the generic resolver computed
        assert dense_c_formula == 2 * expected
        assert Conv2dHandler().generate_einsum("conv2d", ts).get_compute_cost(ts) == expected

    def test_missing_output_shape_falls_back_to_analyzer_inference(self):
        """EinsumAnalyzer infers the conv output shape when it is absent."""
        ts, expected = _case(2, 2)
        ts_no_out = TensorShapes(inputs=list(ts.inputs), outputs=[])
        assert EinsumAnalyzer().get_compute_cost("conv2d", ts_no_out, stride=[1, 1], padding=[1, 1]) == expected


class TestTorchMetaOracle:
    @pytest.mark.parametrize("dims,groups,kw", _CASES)
    def test_matches_torch_flop_counter(self, dims, groups, kw):
        torch = pytest.importorskip("torch")
        from torch.utils.flop_counter import FlopCounterMode
        cls_t = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[dims]
        ts, expected = _case(dims, groups, **kw)
        with torch.device("meta"):
            m = cls_t(C_IN, kw.get("c_out", C_IN), 3, stride=kw.get("stride", 1), padding=kw.get("padding", 1),
                      groups=groups, bias=False)
            x = torch.empty(ts.inputs[0])
            with FlopCounterMode(display=False) as fc:
                y = m(x)
        assert list(y.shape) == ts.outputs[0] and list(m.weight.shape) == ts.inputs[1]
        assert fc.get_total_flops() == 2 * expected
        _, op = _H[dims]
        assert EinsumAnalyzer().get_compute_cost(op, ts) == expected


# grouped-conv-integration-1 (closed): on the release branch the lineage base EinsumAnalyzer.get_compute_cost
# returned compute_cost_from_equation for any stored equation before the registered handler, so
# ConvEinsumOp's grouped override was bypassed in graph analysis and the 3D grouped captured cases
# were strict xfails.  Closed by the lineage-regression fix R2 (handler precedence), merged into
# the release branch as a lineage repair (an earlier change; target the target): the marks are gone and the
# twelve cases below are all required passes (lane11 composition finding INTEG-11).


class TestCapturedPipeline:
    """independent review: the reviewer's twelve captured meta-pipeline controls (nn.Conv{1,2,3}d,
    C_in 6, kernel 2, groups {1, 2, 6} and depthwise multiplier 2), checked through
    meta capture -> converter -> renamed graph -> EinsumGraphAnalyzer, against the hand count and the
    native-shape handler."""

    @pytest.mark.parametrize("dims,groups,multiplier", [
        pytest.param(d, g, m, id=f"{d}d-groups{g}-mult{m}")
        for d in (1, 2, 3) for g, m in [(1, 1), (2, 1), (6, 1), (6, 2)]])
    def test_captured_grouped_conv_keeps_exact_macs(self, tmp_path, dims, groups, multiplier):
        import yaml
        from solar.common.types import ProcessingConfig
        from solar.graph import PyTorchProcessor
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
        cin = 6; cout = cin * multiplier
        ish = [2, cin] + [9, 8, 7][:dims]; osh = [2, cout] + [s - 1 for s in ish[2:]]; wsh = [cout, cin // groups] + [2] * dims
        expected = math.prod(osh) * (cin // groups) * (2 ** dims)
        (tmp_path / "model.py").write_text(f"""import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv{dims}d({cin}, {cout}, 2, groups={groups}, bias=False)
    def forward(self, x):
        return self.conv(x)
def get_inputs():
    return [torch.empty({tuple(ish)}, device='meta')]
def get_init_inputs():
    return []
""")
        for n in ("graph", "einsum", "analysis"):
            (tmp_path / n).mkdir()
        assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
        assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
        analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        assert analysis is not None
        # the native-shape handler path is exact on every case (ConvEinsumOp)
        assert EinsumAnalyzer().get_compute_cost(f"conv{dims}d", TensorShapes(inputs=[ish, wsh], outputs=[osh])) == expected
        e = yaml.safe_load((tmp_path / "einsum/einsum_graph.yaml").read_text())
        assert analysis["total"]["macs"] == expected, (expected, analysis["total"]["macs"], [(n, l["einsum_equation"]) for n, l in e["layers"].items() if l.get("is_real_einsum")])


class TestReshapedGroupwiseLayout:
    """INTEG-7: the lineage base handlers emit a group-wise equation over RESHAPED tensors
    (input [B, G, I, *], weight [G, O, I, *]) when module_args carry groups; ConvEinsumOp
    must not apply the torch-layout formula there — the generic product is exact."""

    @pytest.mark.parametrize("dims,G,I,O_pg,B,spatial,k", [(1, 4, 16, 16, 2, (128,), 3), (2, 4, 32, 64, 1, (32, 32), 3), (2, 128, 1, 2, 1, (32, 32), 3)])
    def test_groupwise_reshaped_shapes_cost_is_exact(self, dims, G, I, O_pg, B, spatial, k):
        from solar.einsum.ops.base import EinsumOp
        cls, op = _H[dims]
        out_sp = [s - k + 1 for s in spatial]
        ts = TensorShapes(inputs=[[B, G, I] + list(spatial), [G, O_pg, I] + [k] * dims], outputs=[[B, G, O_pg] + out_sp])
        new_op = cls().generate_einsum(op, ts, module_args={"groups": G, "in_channels": G * I, "out_channels": G * O_pg})
        assert "G" in new_op.equation
        expected = B * G * O_pg * math.prod(out_sp) * I * k ** dims
        assert new_op.get_compute_cost(ts) == expected
        assert EinsumOp(operands=new_op.operands, equation=new_op.equation, name=new_op.name).get_compute_cost(ts) == expected
