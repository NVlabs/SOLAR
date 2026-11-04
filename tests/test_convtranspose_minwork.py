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

"""Regression tests for the transposed-convolution minimum-work bound (issue #44).

Three quantities for a transposed convolution (per spatial dim: input i in
[0, in), tap r in [0, k), output index o = i*stride - padding + r*dilation):

* old output-based count  N * (C_out/groups) * prod(out) * C_in * prod(k)
* dense input-based count N * C_in * prod(in) * (C_out/groups) * prod(k)
  (the reviewed an earlier change convention, kept as ``macs``; = torch's flop counter)
* minimum work            N * C_in * (C_out/groups) * prod_dims #{(i, r): 0 <= o < out}
  (only taps that land inside the output), reported as the separate field
  ``min_work_macs``; ``min_work <= dense`` and ``min_work <= old`` always,
  and ``min_work == dense`` iff no tap is cropped (padding 0 in every dim).

The bound needs stride/padding/dilation, which the converter exports as
``additional_info.conv_args`` from the captured call attributes; without
them the layer reports ``min_work: not_representable``, never a number.
"""

import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar.common.types import TensorShapes
from solar.einsum.ops.conv_ops import ConvTranspose1dHandler, ConvTranspose3dHandler
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

try:  # absent on the base commit: those tests fail (red), they do not skip
    from solar.analysis.graph_analyzer import convtranspose_min_work_macs
    from solar.einsum.pytorch_to_einsum import parse_conv_transpose_args
except ImportError:  # pragma: no cover
    convtranspose_min_work_macs = None
    parse_conv_transpose_args = None


def _out(n, k, s, p, d, op):
    return (n - 1) * s - 2 * p + d * (k - 1) + op + 1


def _hand_min_work(N, Cin, Cout_pg, spatial, k, s, p, d, op):
    """Independent integer index arithmetic (the audit's minwork_count.py)."""
    dims = len(spatial)
    pairs = 1
    for ax in range(dims):
        o_len = _out(spatial[ax], k[ax], s[ax], p[ax], d[ax], op[ax])
        pairs *= sum(1 for i in range(spatial[ax]) for r in range(k[ax]) if 0 <= i * s[ax] - p[ax] + r * d[ax] < o_len)
    return N * Cin * Cout_pg * pairs


def _shapes(N, Cin, Cout, spatial, k, s, p, d, op, groups=1):
    dims = len(spatial)
    out = [_out(spatial[ax], k[ax], s[ax], p[ax], d[ax], op[ax]) for ax in range(dims)]
    return [N, Cin] + list(spatial), [Cin, Cout // groups] + list(k), [N, Cout] + out


def _args(s, p, d, op, groups=1):
    return {"stride": list(s), "padding": list(p), "output_padding": list(op), "dilation": list(d), "groups": groups}


L220_IN, L220_W, L220_OUT = [16, 32, 16, 32, 32], [32, 64, 3, 3, 3], [16, 64, 32, 64, 64]
L220_ARGS = _args((2, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1))


class TestMinWorkArithmetic:
    def test_l2_20(self):
        assert convtranspose_min_work_macs is not None, "min-work helper missing (base commit)"
        mw = convtranspose_min_work_macs(L220_IN, L220_W, L220_OUT, L220_ARGS)
        ts = TensorShapes(inputs=[L220_IN, L220_W, [64]], outputs=[L220_OUT])
        dense = ConvTranspose3dHandler().generate_einsum("convtranspose3d", ts).get_compute_cost(ts)
        assert dense == 14_495_514_624            # an earlier change convention, unchanged
        assert mw == 13_899_366_400               # 16*32*64 * 47*95*95
        assert mw * 442_368 == dense * 424_175    # ratio 424175/442368 by cross-multiplication
        assert mw == _hand_min_work(16, 32, 64, [16, 32, 32], (3,) * 3, (2,) * 3, (1,) * 3, (1,) * 3, (1,) * 3)

    def test_astra_counterexample(self):
        """N=1, Cin=Cout=4, in=7, k=3, s=1, p=2: old 240, dense 336, min-work 240."""
        i, w, o = _shapes(1, 4, 4, [7], (3,), (1,), (2,), (1,), (0,))
        assert o == [1, 4, 5]
        ts = TensorShapes(inputs=[i, w], outputs=[o])
        dense = ConvTranspose1dHandler().generate_einsum("convtranspose1d", ts).get_compute_cost(ts)
        old = 1 * 4 * 5 * 4 * 3
        assert (dense, old) == (336, 240)
        assert convtranspose_min_work_macs(i, w, o, _args((1,), (2,), (1,), (0,))) == 240

    def test_mr74_dilated_example(self):
        """in=7, k=4, s=2, p=1, d=2: dense 28*Cin*Cout, min-work 26*Cin*Cout (out 17)."""
        for cin, cout in ((1, 1), (3, 5)):
            i, w, o = _shapes(1, cin, cout, [7], (4,), (2,), (1,), (2,), (0,))
            assert o[2] == 17
            ts = TensorShapes(inputs=[i, w], outputs=[o])
            assert ConvTranspose1dHandler().generate_einsum("convtranspose1d", ts).get_compute_cost(ts) == 28 * cin * cout
            assert convtranspose_min_work_macs(i, w, o, _args((2,), (1,), (2,), (0,))) == 26 * cin * cout

    @pytest.mark.parametrize("s,d,op", [((1,), (1,), (0,)), ((2,), (1,), (1,)), ((3,), (2,), (2,)), ((2,), (3,), (0,))])
    def test_padding_zero_equals_dense(self, s, d, op):
        i, w, o = _shapes(2, 3, 4, [9], (4,), s, (0,), d, op)
        ts = TensorShapes(inputs=[i, w], outputs=[o])
        dense = ConvTranspose1dHandler().generate_einsum("convtranspose1d", ts).get_compute_cost(ts)
        assert convtranspose_min_work_macs(i, w, o, _args(s, (0,), d, op)) == dense == 2 * 3 * 9 * 4 * 4

    def test_bounds_hold_on_a_grid(self):
        """min_work <= dense and min_work <= old for every small configuration."""
        for n in (3, 5):
            for k in (2, 3):
                for s in (1, 2):
                    for p in range(0, k):
                        for d in (1, 2):
                            for op in range(0, s):
                                o_len = _out(n, k, s, p, d, op)
                                if o_len <= 0:
                                    continue
                                i, w, o = _shapes(1, 2, 2, [n], (k,), (s,), (p,), (d,), (op,))
                                mw = convtranspose_min_work_macs(i, w, o, _args((s,), (p,), (d,), (op,)))
                                dense = 1 * 2 * n * 2 * k
                                old = 1 * 2 * o_len * 2 * k
                                assert mw <= dense and mw <= old
                                assert (mw == dense) == (p == 0)

    def test_groups(self):
        i, w, o = _shapes(2, 4, 6, [5, 5], (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), groups=2)
        assert w == [4, 3, 3, 3]
        hand = _hand_min_work(2, 4, 3, [5, 5], (3, 3), (2, 2), (1, 1), (1, 1), (1, 1))
        assert convtranspose_min_work_macs(i, w, o, _args((2, 2), (1, 1), (1, 1), (1, 1), groups=2)) == hand

    def test_missing_shapes_not_representable(self):
        assert convtranspose_min_work_macs(L220_IN, None, L220_OUT, L220_ARGS) is None
        assert convtranspose_min_work_macs(L220_IN, L220_W, None, L220_ARGS) is None
        assert convtranspose_min_work_macs(L220_IN, L220_W, L220_OUT, None) is None


class TestTorchCrossCheck:
    def test_ones_conv_transpose3d_sums_to_surviving_taps(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        x = torch.ones(1, 1, 4, 4, 4); w = torch.ones(1, 1, 3, 3, 3)
        y = F.conv_transpose3d(x, w, stride=2, padding=1, output_padding=1)
        assert int(y.sum().item()) == 1331
        i, wsh, o = [1, 1, 4, 4, 4], [1, 1, 3, 3, 3], list(y.shape)
        assert convtranspose_min_work_macs(i, wsh, o, L220_ARGS) == 1331
        assert 1 * 1 * 64 * 1 * 27 == 1728                                           # dense
        xa = torch.ones(1, 4, 7); wa = torch.ones(4, 4, 3)
        ya = F.conv_transpose1d(xa, wa, stride=1, padding=2)
        assert int(ya.sum().item()) == 240 and ya.shape[-1] == 5                     # Astra's example


class TestConverterConvArgs:
    L220_RAW = ("[[Tensor(shape=(16, 32, 16, 32, 32), dtype=torch.float32), Tensor(shape=(32, 64, 3, 3, 3), dtype=torch.float32), "
                "Tensor(shape=(64,), dtype=torch.float32), [2, 2, 2], [1, 1, 1], [1, 1, 1], 1, [1, 1, 1]], {}]")

    def test_parse_positional(self):
        assert parse_conv_transpose_args is not None, "parser missing (base commit)"
        assert parse_conv_transpose_args(self.L220_RAW, 3) == L220_ARGS

    def test_parse_variants(self):
        # no bias, ints, keyword stride
        raw = "[[Tensor(shape=(1, 4, 7), dtype=torch.float32), Tensor(shape=(4, 4, 3), dtype=torch.float32), None, 1, 2], {}]"
        assert parse_conv_transpose_args(raw, 1) == _args((1,), (2,), (1,), (0,))
        raw = "[[Tensor(shape=(1, 4, 7), dtype=torch.float32), Tensor(shape=(4, 4, 3), dtype=torch.float32)], {stride: 2, padding: [1], output_padding: 1, groups: 2, dilation: 2}]"
        assert parse_conv_transpose_args(raw, 1) == _args((2,), (1,), (2,), (1,), groups=2)
        assert parse_conv_transpose_args("", 3) is None

    def _node(self, raw):
        node = {"type": "conv_transpose3d", "input_shapes": [L220_IN, L220_W, [64]], "output_shapes": [L220_OUT],
                "input_types": ["input", "weight", "weight"], "output_types": ["output"],
                "connections": {"inputs": ["Model.auxiliary-tensor", "Model.parameter-tensor", "Model.parameter-tensor_1"], "outputs": []},
                "module_args": {"function_name": "conv_transpose3d"}}
        if raw is not None:
            node["module_args"]["raw_attributes"] = raw
        return node

    def test_convert_operation_exports_conv_args(self):
        import networkx as nx
        g = nx.DiGraph(); g.add_node("Model.conv_transpose3d")
        out = PyTorchToEinsum()._convert_operation("Model.conv_transpose3d", self._node(self.L220_RAW), g, [], {})
        assert out["additional_info"]["conv_args"] == L220_ARGS
        assert out["einsum_equation"] == "BC(P-T)(Q-R)(U-S),CKTRS->BKPQU"          # unchanged
        assert out["tensor_shapes"]["inputs"] == [L220_IN, L220_W, [64]]            # unchanged

    def test_convert_operation_without_attributes_degrades_explicitly(self):
        import networkx as nx
        g = nx.DiGraph(); g.add_node("Model.conv_transpose3d")
        with pytest.warns(RuntimeWarning, match="call attributes"):
            out = PyTorchToEinsum()._convert_operation("Model.conv_transpose3d", self._node(None), g, [], {})
        assert "conv_args" not in out.get("additional_info", {})
        assert out["additional_info"]["conv_args_unavailable"] == "no_call_attributes"


def _layer_yaml(additional_info):
    layer = {"type": "convtranspose3d", "einsum_equation": "BC(P-T)(Q-R)(U-S),CKTRS->BKPQU", "elementwise_op": "mul",
             "reduction_op": "add", "is_real_einsum": True, "is_einsum_supportable": True,
             "tensor_names": {"inputs": ["start.Output", "Model.ct.Weight", "Model.ct.Weight_1"], "outputs": ["Model.ct.Output"]},
             "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
             "tensor_shapes": {"inputs": [L220_IN, L220_W, [64]], "outputs": [L220_OUT]},
             "connections": {"inputs": ["start"], "outputs": []}}
    if additional_info is not None:
        layer["additional_info"] = additional_info
    return {"model_name": "t", "layers": {
        "start": {"type": "start", "einsum_equation": "->ABCDE", "elementwise_op": "copy", "reduction_op": "none",
                  "is_real_einsum": False, "is_einsum_supportable": False,
                  "tensor_names": {"inputs": [], "outputs": ["start.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                  "tensor_shapes": {"inputs": [], "outputs": [L220_IN]}, "connections": {"inputs": [], "outputs": ["Model.ct"]}},
        "Model.ct": layer}}


# an earlier decision regressed at the lineage base (reconcile-the lineage base, the owning lane): the lineage prices a
# transposed convolution from its einsum equation over the OUTPUT extents and
# bypasses the reviewed an earlier change handler, so the dense ``macs`` is stride^3 = 8x the
# reviewed input-based convention here (115,964,116,992 vs 14,495,514,624 on
# L2/20; 13,824 vs 1,728 end to end).  The minimum-work bound is independent of
# that (shapes + conv_args), so the min-work assertions below stand.  The dense
# expectation (``TestDenseConventionPinned`` and the captured check below) was
# pinned as a strict xfail on the lineage base; the lineage repair R2 (an earlier change,
# handler precedence) closed an earlier decision and reached the target in `a lineage repair`
# (the release branch the target), so since the an earlier revision reconciliation both
# checks are ordinary active assertions.

class TestAnalyzerFields:
    def _run(self, tmp_path, additional_info):
        tmp_path.mkdir(parents=True, exist_ok=True)
        p = tmp_path / "einsum_graph_renamed.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(_layer_yaml(additional_info), f, sort_keys=False)
        return EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "analysis", copy_graph=False)

    def test_with_conv_args(self, tmp_path):
        a = self._run(tmp_path, {"conv_args": L220_ARGS})
        layer = a["layers"]["Model.ct"]
        assert layer["min_work_macs"] == 13_899_366_400
        assert "min_work" not in layer
        assert a["total"]["min_work_macs"] == 13_899_366_400

    def test_without_conv_args_not_representable(self, tmp_path):
        a = self._run(tmp_path, None)
        layer = a["layers"]["Model.ct"]
        assert layer["min_work"] == "not_representable"
        assert "min_work_macs" not in layer
        assert a["total"]["min_work"] == "not_representable"
        assert "min_work_macs" not in a["total"]

    def test_traffic_and_other_fields_identical_either_way(self, tmp_path):
        a = self._run(tmp_path / "a", {"conv_args": L220_ARGS}); b = self._run(tmp_path / "b", None)
        la, lb = a["layers"]["Model.ct"], b["layers"]["Model.ct"]
        for k in la:
            if k not in ("min_work_macs", "min_work"):
                assert la[k] == lb[k], k


_ATTR_REQUIREMENT = (
    "the capture must record call attributes for FunctionNodes: torchview with upstream #128 "
    "(collect_attributes propagated to input/hidden recorder tensors) or the torchview 0.2.7 "
    "release plus solar/patches/torchview-collect-attributes-inputs.patch as applied by "
    "validate-install; without them conv_args cannot be exported and min_work is not_representable"
)


class TestEndToEndCaptureRequirement:
    SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ct = nn.ConvTranspose3d(1, 1, 3, stride=2, padding=1, output_padding=1, bias=False)

        def forward(self, x):
            return self.ct(x)

    def get_inputs():
        return [torch.randn(1, 1, 4, 4, 4)]

    def get_init_inputs():
        return []
    """

    def _captured(self, tmp_path):
        pytest.importorskip("torch"); pytest.importorskip("torchview")
        from textwrap import dedent
        from solar.common.types import ProcessingConfig
        from solar.graph import PyTorchProcessor
        (tmp_path / "model.py").write_text(dedent(self.SRC))
        for sub in ("graph", "einsum", "analysis"):
            (tmp_path / sub).mkdir()
        assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)) \
            .process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
        graph = yaml.safe_load(open(tmp_path / "graph/pytorch_graph.yaml"))["layers"]
        gnode = [v for v in graph.values() if str(v["type"]).replace("_", "") == "convtranspose3d"][0]
        assert gnode["module_args"].get("raw_attributes"), _ATTR_REQUIREMENT
        assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
        einsum = yaml.safe_load(open(tmp_path / "einsum/einsum_graph.yaml"))["layers"]
        enode = [v for v in einsum.values() if v["type"] == "convtranspose3d"][0]
        assert enode["additional_info"]["conv_args"] == L220_ARGS, _ATTR_REQUIREMENT
        analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"),
                                                       str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
        layer = [v for v in analysis["layers"].values() if v["type"] == "convtranspose3d"][0]
        return analysis, layer

    def test_capture_records_call_attributes_and_min_work_is_reported(self, tmp_path):
        analysis, layer = self._captured(tmp_path)
        assert layer["min_work_macs"] == 1331                  # minimum-work bound: active check
        assert analysis["total"]["min_work_macs"] == 1331

    def test_captured_dense_macs_follow_the_mr74_convention(self, tmp_path):
        """Dense captured-route check (coverage-1).  Strict xfail on the lineage base (F2 / an earlier decision:
        13,824 instead of 1,728); active since the target carries lineage R2 (an earlier change, merged a lineage repair)."""
        analysis, layer = self._captured(tmp_path)
        assert layer["macs"] == 1728                            # dense, input-based an earlier change convention, captured route


class TestDenseConventionPinned:
    """Dense ``macs`` of the reviewed an earlier change convention (issue #31), regressed at the lineage base (an earlier decision; module note)."""

    def test_l2_20_dense_macs_follow_the_mr74_convention(self, tmp_path):
        """Strict xfail on the lineage base (F2 / an earlier decision: equation-priced transposed conv 8x the earlier convention);
        active since the target carries lineage R2 (an earlier change, merged a lineage repair / the target)."""
        a = TestAnalyzerFields()._run(tmp_path, {"conv_args": L220_ARGS})
        assert a["layers"]["Model.ct"]["macs"] == 14_495_514_624
        assert a["total"]["macs"] == 14_495_514_624



# --- Independent reviewer probes for the ConvTranspose min-work accounting (adapted) ----------

def _count_by_output(n, k, s, p, d, o):
    """Independent count: enumerate output/tap pairs and invert o = i*s - p + r*d (divisibility + input bounds)."""
    return sum(1 for out in range(o) for r in range(k) if (out + p - r * d) % s == 0 and 0 <= (out + p - r * d) // s < n)


def test_exhaustive_small_integer_geometry():
    import itertools
    cases = 0
    for n, k, s, p, d in itertools.product(range(1, 6), range(1, 5), range(1, 4), range(4), range(1, 4)):
        for op in range(max(s, d)):
            o = (n - 1) * s - 2 * p + d * (k - 1) + op + 1
            if o < 1:
                continue
            expected = 2 * 4 * 3 * _count_by_output(n, k, s, p, d, o)
            got = convtranspose_min_work_macs([2, 4, n], [4, 3, k], [2, 6, o], dict(stride=[s], padding=[p], dilation=[d], output_padding=[op], groups=2))
            assert got == expected, (n, k, s, p, d, op, got, expected)
            assert got <= 2 * 4 * 3 * n * k and got <= 2 * 4 * 3 * o * k
            assert (got == 2 * 4 * 3 * n * k) == (p == 0)
            cases += 1
    assert cases > 1000


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("style", ["native", "positional", "keyword"])
def test_capture_argument_routes_and_grouped_minwork(tmp_path, dims, style):
    import math
    pytest.importorskip("torchview")
    from solar.common.types import ProcessingConfig
    from solar.graph import PyTorchProcessor
    spatial = [3 + i for i in range(dims)]; kernel = [3] * dims; s = [2] * dims; p = [1] * dims; d = [1] * dims; op = [0] * dims
    ctor = (f"self.ct=nn.ConvTranspose{dims}d(4,6,{kernel},stride={s},padding={p},output_padding={op},groups=2,bias=False)"
            if style == "native" else f"self.w=nn.Parameter(torch.ones(4,3,*{kernel}))")
    call = ("self.ct(x)" if style == "native" else
            f"F.conv_transpose{dims}d(x,self.w,None,{s},{p},{op},2,{d})" if style == "positional" else
            f"F.conv_transpose{dims}d(input=x,weight=self.w,stride={s},padding={p},output_padding={op},groups=2,dilation={d})")
    src = f"""import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        {ctor}
    def forward(self,x):
        return {call}
def get_inputs():return [torch.ones(1,4,*{spatial})]
def get_init_inputs():return []
"""
    (tmp_path / "model.py").write_text(src)
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
    eg = yaml.safe_load((tmp_path / "einsum/einsum_graph_renamed.yaml").read_text())
    es = [v for v in eg["layers"].values() if str(v["type"]).replace("_", "") == f"convtranspose{dims}d"]; assert len(es) == 1
    assert es[0]["additional_info"].get("conv_args") == dict(stride=s, padding=p, dilation=d, output_padding=op, groups=2)
    a = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"), copy_graph=False)
    layers = [v for v in a["layers"].values() if str(v["type"]).replace("_", "") == f"convtranspose{dims}d"]; assert len(layers) == 1
    shape = [(n - 1) * 2 - 2 + 2 + 1 for n in spatial]
    expected = 4 * 3 * math.prod(_count_by_output(n, 3, 2, 1, 1, o) for n, o in zip(spatial, shape))
    assert layers[0]["min_work_macs"] == expected and a["total"]["min_work_macs"] == expected


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_grouped_small_cpu_crosscheck(dims):
    import torch
    import torch.nn.functional as F
    spatial = [3] * dims; k = [2] * dims; s = [1] * dims; p = [1] * dims; d = [2] * dims; op = [1] * dims
    x = torch.ones(2, 4, *spatial); w = torch.ones(4, 3, *k)
    y = getattr(F, f"conv_transpose{dims}d")(x, w, stride=s, padding=p, output_padding=op, groups=2, dilation=d)
    got = convtranspose_min_work_macs(list(x.shape), list(w.shape), list(y.shape), dict(stride=s, padding=p, dilation=d, output_padding=op, groups=2))
    assert got == int(y.sum())


@pytest.mark.parametrize("stride_text", ["opaque", "[s0]", "None", "0", "[2, 2]", "True", "-1"])
def test_unparsed_recorded_argument_does_not_become_a_default(stride_text):
    """args-1: an explicitly present but undecodable / invalid stride is not the omitted default."""
    raw = ("[[Tensor(shape=(1, 1, 7), dtype=torch.float32), Tensor(shape=(1, 1, 4), dtype=torch.float32)], "
           "{stride: " + stride_text + ", padding: 1, dilation: 2, output_padding: 0, groups: 1}]")
    info = PyTorchToEinsum()._build_additional_info({"type": "conv_transpose1d", "module_args": {"raw_attributes": raw}})
    assert parse_conv_transpose_args(raw, 1) is None
    assert info.get("conv_args_unavailable") == "unparsed_call_attributes" and "conv_args" not in info


def test_omitted_defaults_and_wholly_unparsed_control():
    raw = "[[Tensor(shape=(1, 1, 7), dtype=torch.float32), Tensor(shape=(1, 1, 4), dtype=torch.float32)], {}]"
    assert parse_conv_transpose_args(raw, 1) == dict(stride=[1], padding=[0], output_padding=[0], dilation=[1], groups=1)
    assert parse_conv_transpose_args("unparsed", 1) is None
    assert parse_conv_transpose_args("", 1) is None
    # positional ints and keyword overrides still bind; an undecodable keyword that is NOT a conv argument is ignored
    raw2 = "[[Tensor(shape=(1, 1, 7), dtype=torch.float32), Tensor(shape=(1, 1, 4), dtype=torch.float32), None, 2, 1], {dilation: [2], note: opaque}]"
    assert parse_conv_transpose_args(raw2, 1) == dict(stride=[2], padding=[1], output_padding=[0], dilation=[2], groups=1)
