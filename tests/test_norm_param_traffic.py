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

"""Regression tests for normalisation parameter traffic (issue #34, an earlier decision).

Convention: a normalisation layer's affine parameters (weight, bias) and,
for BatchNorm/InstanceNorm in eval mode, its running statistics are
constant per-layer tensors read once per layer.  They are billed as
``weight`` inputs of the norm node, exactly like nn.Linear's bias: their
elements enter the layer's unfused and fused (external) traffic.  The
elementwise scale/shift itself is already inside ``other_ops`` (one op per
output element); MACs stay 0 and ``other_ops`` is unchanged.

Hand formula per norm layer (elements):

    unfused_elements = 2 * numel(x) + sum(numel(p) for p in params)

torchview connects only traced parameters (e.g. conv weights passed
positionally) as ``parameter-tensor`` nodes; norm parameters are passed as
keyword arguments (and running stats are buffers), so they were missing
from the graph and from traffic.  Models are tiny and traced on CPU, as in
test_graph_analyzer_regression.py.
"""

import json
import math
import sys
import warnings
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchview")

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
import yaml


def _pipeline(tmp_path, model_source):
    (tmp_path / "model.py").write_text(dedent(model_source))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    ok = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)) \
        .process_model_file(str(tmp_path / "model.py"), str(graph_dir))
    assert ok
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir),
                                                   precision="fp32", copy_graph=False)
    assert analysis is not None
    graph = yaml.safe_load(open(graph_dir / "pytorch_graph.yaml"))["layers"]
    einsum = yaml.safe_load(open(einsum_dir / "einsum_graph.yaml"))["layers"]
    return graph, einsum, analysis


def _model(norm_ctor, input_shape):
    return f"""\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = {norm_ctor}

        def forward(self, x):
            return self.norm(x)

    def get_inputs():
        return [torch.randn{tuple(input_shape)}]

    def get_init_inputs():
        return []
    """


# (id, ctor, input shape, node type, expected parameter element counts in order, expected names)
X3 = (2, 8, 16)       # LayerNorm / RMSNorm over the last dim (16)
X4 = (2, 8, 5, 5)     # GroupNorm / BatchNorm over C = 8
_CASES = [
    pytest.param("layer_norm", "nn.LayerNorm(16)", X3, [16, 16], id="layernorm-affine"),
    pytest.param("layer_norm", "nn.LayerNorm(16, elementwise_affine=False)", X3, [], id="layernorm-no-affine-control"),
    pytest.param("rms_norm", "nn.RMSNorm(16)", X3, [16], id="rmsnorm-affine"),
    pytest.param("rms_norm", "nn.RMSNorm(16, elementwise_affine=False)", X3, [], id="rmsnorm-no-affine"),
    pytest.param("group_norm", "nn.GroupNorm(4, 8)", X4, [8, 8], id="groupnorm-affine"),
    pytest.param("group_norm", "nn.GroupNorm(4, 8, affine=False)", X4, [], id="groupnorm-no-affine"),
    # eval-mode BatchNorm: running_mean, running_var (buffers, read) + weight, bias
    pytest.param("batch_norm", "nn.BatchNorm2d(8)", X4, [8, 8, 8, 8], id="batchnorm-eval-affine"),
    pytest.param("batch_norm", "nn.BatchNorm2d(8, affine=False)", X4, [8, 8], id="batchnorm-eval-no-affine"),
]


def _norm_layer(layers, node_type):
    hits = [(k, v) for k, v in layers.items() if str(v.get("type")) == node_type]
    assert len(hits) == 1, f"expected one {node_type} node, got {[k for k, _ in hits]}"
    return hits[0]


_ATTR_REQUIREMENT = (
    "the capture must record call attributes for FunctionNodes: torchview with upstream #128 "
    "(collect_attributes propagated to input/hidden recorder tensors) or the torchview 0.2.7 "
    "release plus solar/patches/torchview-collect-attributes-inputs.patch as applied by "
    "validate-install; without them the norm parameters cannot be attached"
)


class TestNormParamTraffic:
    def test_capture_records_call_attributes(self, tmp_path):
        """Environment requirement stated explicitly (not inferred from a missing list)."""
        graph, _, _ = _pipeline(tmp_path, _model("nn.LayerNorm(16)", X3))
        _, gnode = _norm_layer(graph, "layer_norm")
        assert gnode["module_args"].get("raw_attributes"), _ATTR_REQUIREMENT
        assert "norm_params_unattached" not in gnode["module_args"], _ATTR_REQUIREMENT

    @pytest.mark.parametrize("node_type,ctor,x_shape,params", _CASES)
    def test_graph_einsum_and_traffic(self, tmp_path, node_type, ctor, x_shape, params):
        graph, einsum, analysis = _pipeline(tmp_path, _model(ctor, x_shape))
        n_x = math.prod(x_shape)

        # pytorch_graph.yaml: activation first, then each parameter as a weight input
        gid, gnode = _norm_layer(graph, node_type)
        if gnode["module_args"].get("norm_params_unattached"):
            pytest.fail(f"{gid}: parameters not attached ({gnode['module_args']['norm_params_unattached']}); " + _ATTR_REQUIREMENT)
        assert gnode["input_shapes"][0] == list(x_shape)
        assert [math.prod(s) for s in gnode["input_shapes"][1:]] == params
        assert gnode["input_types"] == ["input"] + ["weight"] * len(params)
        for pid in gnode["connections"]["inputs"][1:]:
            assert graph[pid]["type"] == "parameter-tensor"
            assert graph[pid]["connections"]["outputs"] == [gid]

        # einsum_graph.yaml: weights named like any other layer's weights
        eid, enode = _norm_layer(einsum, node_type)
        assert enode["tensor_types"]["inputs"] == ["input"] + ["weight"] * len(params)
        assert [math.prod(s) for s in enode["tensor_shapes"]["inputs"][1:]] == params
        expected_names = ([f"{eid}.Weight"] + [f"{eid}.Weight_{i}" for i in range(1, len(params))]) if params else []
        assert enode["tensor_names"]["inputs"][1:] == expected_names
        assert all("parameter" not in c for c in enode["connections"]["inputs"])   # weights are not graph layers
        assert enode["einsum_equation"].split("->")[0].count(",") == 0   # one activation operand

        # analysis.yaml: hand formula; MACs 0 and other_ops unchanged
        aid, anode = _norm_layer(analysis["layers"], node_type)
        assert anode["macs"] == 0
        assert anode["other_ops"] == n_x
        assert anode["unfused_elements"] == 2 * n_x + sum(params)
        # weights are external reads: they appear in fused (model-IO) traffic too
        assert anode["fused_elements"] == 2 * n_x + sum(params)
        assert analysis["total"]["macs"] == 0

    def test_non_affine_control_is_bit_identical_to_plain_elementwise(self, tmp_path):
        """Control: a non-affine LayerNorm has no parameters, so its traffic is
        exactly 2 * numel(x), as before the fix."""
        _, _, analysis = _pipeline(tmp_path, _model("nn.LayerNorm(16, elementwise_affine=False)", X3))
        _, anode = _norm_layer(analysis["layers"], "layer_norm")
        assert (anode["unfused_elements"], anode["fused_elements"], anode["macs"], anode["other_ops"]) == \
            (2 * math.prod(X3), 2 * math.prod(X3), 0, math.prod(X3))

    def test_linear_bias_convention_matches(self, tmp_path):
        """Reference point for the convention: nn.Linear's bias is a weight input
        read once per layer; LayerNorm's affine params are billed the same way."""
        src = """\
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return self.ln(self.fc(x))

        def get_inputs():
            return [torch.randn(2, 8, 16)]

        def get_init_inputs():
            return []
        """
        _, einsum, analysis = _pipeline(tmp_path, src)
        # SOLAR splits nn.Linear into the matmul (weight) and a bias_add layer
        # whose bias is a weight input; LayerNorm's weight and bias follow the
        # same rule on its own layer.
        lin = [v for v in einsum.values() if v["type"] == "linear"][0]
        bias_add = einsum["Model.linear.bias_add"]
        ln = [v for v in einsum.values() if v["type"] == "layer_norm"][0]
        assert lin["tensor_types"]["inputs"].count("weight") == 1
        assert bias_add["tensor_types"]["inputs"] == ["input", "weight"]
        assert bias_add["tensor_shapes"]["inputs"][1] == [16]
        assert ln["tensor_types"]["inputs"] == ["input", "weight", "weight"]
        assert ln["tensor_shapes"]["inputs"][1:] == [[16], [16]]
        a_ln = [v for v in analysis["layers"].values() if v["type"] == "layer_norm"][0]
        assert a_ln["unfused_elements"] == 2 * 2 * 8 * 16 + 2 * 16
        assert analysis["total"]["macs"] == 2 * 8 * 16 * 16              # the Linear only


# ── Round-2 probes (an earlier decision review: attach-1 / TRAIN-1 / CAP-1) ──

def _src(body, init="self.w = nn.Parameter(torch.ones(16)); self.b = nn.Parameter(torch.zeros(16))",
         inputs="torch.randn(2, 8, 16)", args="x"):
    return f"""\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            {init}

        def forward(self, {args}):
            return {body}

    def get_inputs():
        return [{inputs}]

    def get_init_inputs():
        return []
    """


def _observed(tmp_path, source, op="layer_norm"):
    graph, einsum, analysis = _pipeline(tmp_path, source)
    _, g = _norm_layer(graph, op); _, e = _norm_layer(einsum, op); _, a = _norm_layer(analysis["layers"], op)
    return g, e, a, analysis


def _keyword_activation_connected(g):
    """Did the capture connect the keyword ``input=`` activation as the first input?"""
    conns = g["connections"]["inputs"]
    return bool(conns) and g["input_shapes"][:1] == [[2, 8, 16]] and not conns[0].endswith("parameter-tensor")


def _incomplete(g, e, a, analysis, perf, reason=None):
    """The reviewer's end-to-end incompleteness contract (an earlier decision review)."""
    r = g["module_args"].get("norm_params_unattached")
    assert r and (reason is None or r == reason), (r, reason)
    assert e["norm_params_unattached"] == r and e["is_einsum_supportable"] is False
    assert a["status"] == "incomplete" and a["norm_params_unattached"] == r
    assert analysis["total"]["status"] == "incomplete" and r in analysis["total"]["incomplete_layers"].values()
    assert perf["status"] == "incomplete" and perf["incomplete_layers"] == analysis["total"]["incomplete_layers"]


def _stages(tmp_path, source, op="layer_norm"):
    """graph node, einsum node, analysis node, analysis, perf workload (all five stages)."""
    from solar.perf.perf_model import EinsumGraphPerfModel
    g, e, a, analysis = _observed(tmp_path, source, op)
    perf = EinsumGraphPerfModel().predict(analysis_path=str(tmp_path / "analysis" / "analysis.yaml"),
                                          output_dir=str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
    assert perf is not None
    return g, e, a, analysis, perf["workload"]


class TestArgumentIdentity:
    """attach-1: attach only what is missing, by role; never duplicate a connected input."""

    @pytest.mark.parametrize("body", [
        "F.layer_norm(x, (16,), self.w, bias=self.b)",
        "F.layer_norm(x, (16,), self.w, self.b)",
        "F.layer_norm(input=x, normalized_shape=(16,), weight=self.w, bias=self.b)",
        "torch.layer_norm(x, (16,), self.w, bias=self.b)",      # positional weight already connected: attach only bias
        "torch.layer_norm(x, (16,), self.w, self.b)",           # both connected: attach nothing
    ])
    def test_parameter_placement_gives_exact_traffic(self, tmp_path, body):
        g, e, a, _ = _observed(tmp_path, _src(body))
        assert g["input_shapes"] == [[2, 8, 16], [16], [16]], g["input_shapes"]
        assert g["input_types"] == ["input", "weight", "weight"]
        assert "norm_params_unattached" not in g["module_args"]
        assert (a["macs"], a["other_ops"], a["unfused_elements"]) == (0, 256, 544)

    @pytest.mark.parametrize("body", ["torch.layer_norm(x, (16,), None, b)", "F.layer_norm(x, (16,), weight=None, bias=b)"])
    def test_dynamic_bias_is_not_duplicated(self, tmp_path, body):
        g, e, a, _ = _observed(tmp_path, _src(body, init="pass", inputs="torch.randn(2, 8, 16), torch.randn(16)", args="x, b"))
        assert g["input_shapes"] == [[2, 8, 16], [16]], "an already-connected dynamic bias must not be appended again"
        assert a["unfused_elements"] == 512 + 16

    def test_keyword_activation_is_an_explicit_boundary(self, tmp_path):
        """torch.layer_norm(input=x, ...): when the capture records a producer edge for
        the keyword activation (the parameter-operand shim of the release branch
        the lineage base binds keyword tensors semantically) the call is handled like the
        positional spelling; when the edge is missing (the project's torchview patch
        orders positional tensors only) the dependency cannot be restored, the node is
        marked, nothing is guessed, and the operation is not supportable downstream."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g, e, a, analysis = _observed(tmp_path, _src("torch.layer_norm(input=x, normalized_shape=(16,), weight=self.w, bias=self.b)"))
        if _keyword_activation_connected(g):
            assert "norm_params_unattached" not in g["module_args"]
            assert g["input_shapes"] == [[2, 8, 16], [16], [16]] and g["input_types"] == ["input", "weight", "weight"]
            assert e["is_einsum_supportable"] is True and (a["macs"], a["other_ops"], a["unfused_elements"]) == (0, 256, 544)
            assert "status" not in analysis["total"]
        else:
            assert any("keyword input=" in str(w.message) for w in caught)
            assert g["module_args"]["norm_params_unattached"] == "keyword_activation"
            assert "weight" not in g["input_types"]                      # no guessed weights
            assert e["is_einsum_supportable"] is False and e["norm_params_unattached"] == "keyword_activation"
            assert a["status"] == "incomplete" and a["norm_params_unattached"] == "keyword_activation"
            assert analysis["total"]["status"] == "incomplete"

    @pytest.mark.parametrize("affine,track,expected", [(True, True, [8, 8, 8, 8]), (False, True, [8, 8]), (True, False, [8, 8]), (False, False, [])])
    def test_instance_norm_eval_read_only_parameters(self, tmp_path, affine, track, expected):
        g, e, a, _ = _observed(tmp_path, _model(f"nn.InstanceNorm2d(8, affine={affine}, track_running_stats={track})", (2, 8, 5, 5)), "instance_norm")
        assert [math.prod(s) for s in g["input_shapes"][1:]] == expected
        assert (a["macs"], a["other_ops"], a["unfused_elements"]) == (0, 400, 800 + sum(expected))


class TestTrainingModeExclusion:
    """train-1: modes that update running statistics are excluded explicitly."""

    @pytest.mark.parametrize("op,body", [
        ("batch_norm", "F.batch_norm(x, self.rm, self.rv, weight=None, bias=None, training=True)"),
        ("instance_norm", "F.instance_norm(x, self.rm, self.rv, weight=None, bias=None, use_input_stats=True)"),
    ])
    def test_updating_statistics_are_excluded_not_billed_as_weights(self, tmp_path, op, body):
        init = "self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))"
        with pytest.warns(RuntimeWarning, match="updates the running statistics"):
            g, e, a, analysis = _observed(tmp_path, _src(body, init=init, inputs="torch.randn(2, 8, 5, 5)"), op)
        assert g["module_args"]["norm_params_unattached"] == "running_stats_update"
        assert g["input_shapes"] == [[2, 8, 5, 5]]                  # statistics not attached as weights
        assert e["is_einsum_supportable"] is False
        assert a["status"] == "incomplete" and analysis["total"]["status"] == "incomplete"
        assert analysis["total"]["incomplete_layers"] == {next(k for k, v in analysis["layers"].items() if v["type"] == op): "running_stats_update"}

    def test_eval_mode_batch_norm_is_supported(self, tmp_path):
        g, e, a, analysis = _observed(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        assert "norm_params_unattached" not in g["module_args"] and e["is_einsum_supportable"] is True
        assert "status" not in analysis["total"]


class TestCapabilityMarkerPropagation:
    """cap-1: a missing-capability marker survives conversion, analysis and perf."""

    def test_synthetic_missing_attributes_warn_and_mark_graph(self):
        from solar.graph.torchview_processor import TorchviewProcessor
        from solar.common.types import NodeInfo
        p = TorchviewProcessor()
        n = NodeInfo(node_id="Model.layer_norm", type="layer_norm", node_class="FunctionNode", input_nodes=["input"], output_nodes=[],
                     input_shapes=[[2, 16]], output_shapes=[[2, 16]], input_dtypes=["torch.float32"], output_dtypes=["torch.float32"],
                     input_types=["input"], output_types=["output"], module_args={})
        with pytest.warns(RuntimeWarning, match="NOT billed"):
            p._attach_detached_parameters([n], {n.node_id: n})
        assert n.module_args["norm_params_unattached"] == "no_call_attributes"

    def test_marker_survives_converter_analyzer_and_perf(self, tmp_path):
        import networkx as nx
        from solar.perf.perf_model import EinsumGraphPerfModel
        node = {"type": "layer_norm", "input_shapes": [[2, 8, 16]], "output_shapes": [[2, 8, 16]], "input_types": ["input"],
                "input_dtypes": ["torch.float32"], "output_types": ["output"],
                "connections": {"inputs": ["Model.auxiliary-tensor"], "outputs": []},
                "module_args": {"function_name": "layer_norm", "norm_params_unattached": "no_call_attributes"}}
        g = nx.DiGraph(); g.add_node("Model.layer_norm")
        layer = PyTorchToEinsum()._convert_operation("Model.layer_norm", node, g, [], {})
        assert layer["is_einsum_supportable"] is False and layer["norm_params_unattached"] == "no_call_attributes"
        layers = {"start": {"type": "start", "einsum_equation": "->ABC", "elementwise_op": "copy", "reduction_op": "none",
                            "is_real_einsum": False, "is_einsum_supportable": False,
                            "tensor_names": {"inputs": [], "outputs": ["start.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                            "tensor_shapes": {"inputs": [], "outputs": [[2, 8, 16]]}, "connections": {"inputs": [], "outputs": ["Model.layer_norm"]}},
                  "Model.layer_norm": {**layer, "tensor_names": {"inputs": ["start.Output"], "outputs": ["Model.layer_norm.Output"]},
                                       "connections": {"inputs": ["start"], "outputs": []}}}
        p = tmp_path / "einsum_graph_renamed.yaml"
        with open(p, "w") as f:
            yaml.safe_dump({"model_name": "t", "layers": layers}, f, sort_keys=False)
        analysis = EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "analysis", precision="fp32", copy_graph=False)
        a = analysis["layers"]["Model.layer_norm"]
        assert a["status"] == "incomplete" and a["norm_params_unattached"] == "no_call_attributes"
        assert analysis["total"]["status"] == "incomplete" and analysis["total"]["incomplete_layers"] == {"Model.layer_norm": "no_call_attributes"}
        assert "norm_params_unattached" in json.dumps(analysis)
        perf = EinsumGraphPerfModel().predict(analysis_path=str(tmp_path / "analysis" / "analysis.yaml"),
                                              output_dir=str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
        assert perf["workload"].get("status") == "incomplete"
        assert perf["workload"].get("incomplete_layers") == {"Model.layer_norm": "no_call_attributes"}


class TestRound2Matrix:
    """Independent review of the norm-parameter traffic fix: the reviewer's 10-case matrix, adapted
    to the the release branch the lineage base capture (keyword tensors connected by the parameter-operand
    shim) while keeping the reviewer's contracts: identity by argument slot, never by shape;
    connected tensors keep their producer edge; updating calls excluded in both spellings; every
    boundary propagated end to end (graph -> einsum -> analysis -> perf)."""

    def test_keyword_activation_boundary_is_end_to_end(self, tmp_path):
        g, e, a, analysis, perf = _stages(tmp_path, _src("torch.layer_norm(input=x, normalized_shape=(16,), weight=self.w, bias=self.b)"))
        if _keyword_activation_connected(g):
            assert "norm_params_unattached" not in g["module_args"] and perf.get("status") is None
            assert g["input_types"] == ["input", "weight", "weight"] and a["unfused_elements"] == 544
        else:
            _incomplete(g, e, a, analysis, perf, "keyword_activation")
            assert "weight" not in g["input_types"]

    @pytest.mark.parametrize("op", ["batch_norm", "instance_norm"])
    @pytest.mark.parametrize("native", [False, True])
    def test_native_and_functional_updating_statistics_are_excluded(self, tmp_path, op, native):
        if native:
            body = f"torch.{op}(x, None, None, self.rm, self.rv, True, 0.1, 1e-5, False)"     # ATen order: weight, bias, running_mean, running_var
        elif op == "batch_norm":
            body = "F.batch_norm(x, self.rm, self.rv, weight=None, bias=None, training=True)"
        else:
            body = "F.instance_norm(x, self.rm, self.rv, weight=None, bias=None, use_input_stats=True)"
        init = "self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))"
        g, e, a, analysis, perf = _stages(tmp_path, _src(body, init=init, inputs="torch.randn(2, 8, 5, 5)"), op)
        _incomplete(g, e, a, analysis, perf, "running_stats_update")
        assert g["input_shapes"] == [[2, 8, 5, 5]]                  # the statistics are not billed as read-only weights

    @pytest.mark.parametrize("buffer_role", ["weight", "bias"])
    def test_same_shape_buffer_and_dynamic_operand_keep_actual_roles(self, tmp_path, buffer_role):
        """A missing buffer and a connected dynamic tensor of the same shape: the identity of
        the connected tensor is not established by its shape, so the call is excluded explicitly
        (argument_mapping_unresolved) rather than resolved by a guess."""
        if buffer_role == "weight":
            init, body, args = "self.register_buffer('w', torch.ones(16))", "torch.layer_norm(x, (16,), self.w, b)", "x, b"
        else:
            init, body, args = "self.register_buffer('b', torch.zeros(16))", "torch.layer_norm(x, (16,), w, self.b)", "x, w"
        g, e, a, analysis, perf = _stages(tmp_path, _src(body, init=init, args=args, inputs="torch.randn(2, 8, 16), torch.randn(16)"))
        if g["module_args"].get("norm_params_unattached"):
            _incomplete(g, e, a, analysis, perf, "argument_mapping_unresolved")
            assert g["input_shapes"] == [[2, 8, 16], [16]]          # nothing synthesized
            return
        attached = [k.rsplit(".", 1)[-1] for k in g["connections"]["inputs"] if k.startswith("Model.layer_norm.")]
        assert attached == [buffer_role], ("shape matching must not swap a detached buffer with a connected dynamic tensor", attached)
        assert g["input_types"] == ["input", "input", "weight"] and a["unfused_elements"] == 544

    @pytest.mark.parametrize("spelling,keyword", [("torch", False), ("torch", True), ("F", True)])
    def test_computed_parameter_edge_preserved_or_explicitly_excluded(self, tmp_path, spelling, keyword):
        call = f"{spelling}.layer_norm(x, (16,), " + ("weight=v, bias=None)" if keyword else "v, None)")
        src = _src(call, init="pass", args="x, w", inputs="torch.randn(2, 8, 16), torch.randn(16)").replace(
            f"return {call}", f"v = w * 2.0\n            return {call}")
        g, e, a, analysis, perf = _stages(tmp_path, src)
        if g["module_args"].get("norm_params_unattached"):
            _incomplete(g, e, a, analysis, perf); return
        mul = [k for k, v in analysis["layers"].items() if v["type"] == "mul"]
        assert len(mul) == 1
        assert mul[0] in e["connections"]["inputs"], "a computed parameter keeps its producer edge; it is never re-attached as a synthetic weight"
        assert "weight" not in g["input_types"][1:] or g["input_shapes"] == [[2, 8, 16], [16]]
        assert analysis["total"]["fused_elements"] == 528

    def test_negative_capability_is_end_to_end(self, tmp_path):
        """nn.LayerNorm module call: with call attributes the parameters are attached; on a
        capture without them (torchview before upstream #128 and without the project patch)
        the boundary no_call_attributes must reach perf."""
        g, e, a, analysis, perf = _stages(tmp_path, _src("self.norm(x)", init="self.norm = nn.LayerNorm(16)"))
        if g["module_args"].get("raw_attributes"):
            assert "norm_params_unattached" not in g["module_args"] and a["unfused_elements"] == 544
        else:
            _incomplete(g, e, a, analysis, perf, "no_call_attributes")


class TestRound3Identity:
    """Round-3 contracts beyond the matrix: per-spelling slots and module library identity."""

    @pytest.mark.parametrize("op", ["batch_norm", "instance_norm"])
    def test_native_eval_call_reads_statistics_and_affine_by_aten_order(self, tmp_path, op):
        """torch.<op>(x, w, b, rm, rv, False, ...): ATen order; w/b are connected Parameters, the
        buffers rm/rv are attached by their (native) roles."""
        init = ("self.w = nn.Parameter(torch.ones(8)); self.b = nn.Parameter(torch.zeros(8)); "
                "self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))")
        g, e, a, analysis, perf = _stages(tmp_path, _src(f"torch.{op}(x, self.w, self.b, self.rm, self.rv, False, 0.1, 1e-5, False)", init=init, inputs="torch.randn(2, 8, 5, 5)"), op)
        if g["module_args"].get("norm_params_unattached"):
            _incomplete(g, e, a, analysis, perf, "argument_mapping_unresolved"); return
        attached = [k.rsplit(".", 1)[-1] for k in g["connections"]["inputs"] if k.startswith(f"Model.{op}.")]
        assert attached == ["running_mean", "running_var"]
        assert (a["macs"], a["other_ops"], a["unfused_elements"]) == (0, 400, 832)

    def test_module_batch_norm_roles_are_bound_from_the_executed_call(self, tmp_path):
        """nn.BatchNorm2d: the recorded functional call binds the positional running statistics to
        detached buffers and the keyword weight/bias to the connected Parameters (operand_kinds);
        the statistics are attached by those roles — exactly four 8-element reads."""
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        assert "norm_params_unattached" not in g["module_args"]
        kinds = g["module_args"]["operand_kinds"]
        assert kinds[0] == ["activation", "detached", "detached"]
        assert {k: v for k, v in kinds[1].items() if isinstance(v, str)} == {"weight": "parameter", "bias": "parameter"}
        assert kinds[1]["training"] == {"scalar": "bool", "value": False}            # the scalar leaf is recorded structurally (round 11)
        attached = [k.rsplit(".", 1)[-1] for k in g["connections"]["inputs"] if k.startswith("Model.batch_norm.")]
        connected_params = [k for k in g["connections"]["inputs"] if "parameter-tensor" in k]
        assert attached == ["running_mean", "running_var"] and len(connected_params) == 2
        assert [math.prod(s) for s in g["input_shapes"][1:]] == [8, 8, 8, 8] and a["unfused_elements"] == 832

    def test_user_functional_batch_norm_with_same_shape_buffers_is_bound_by_slot(self, tmp_path):
        """F.batch_norm(x, rm, rv, w, b) written by hand: rm/rv (buffers) and w/b (Parameters) all
        have shape [8]; the slots the connected Parameters occupy are not inferred from the shapes
        but read from the executed call's binding (weight/bias keywords), so the two detached
        positional statistics are attached — no module class or forward object is consulted."""
        init = ("self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8)); "
                "self.w = nn.Parameter(torch.ones(8)); self.b = nn.Parameter(torch.zeros(8))")
        g, e, a, analysis, perf = _stages(tmp_path, _src("F.batch_norm(x, self.rm, self.rv, self.w, self.b, False)", init=init, inputs="torch.randn(2, 8, 5, 5)"), "batch_norm")
        assert "norm_params_unattached" not in g["module_args"]
        attached = [k.rsplit(".", 1)[-1] for k in g["connections"]["inputs"] if k.startswith("Model.batch_norm.")]
        assert attached == ["running_mean", "running_var"]
        assert g["input_shapes"] == [[2, 8, 5, 5], [8], [8], [8], [8]] and a["unfused_elements"] == 832 and perf.get("status") is None


_USER_NORM_SRC = """\
    import torch
    from torch import nn
    from torch.nn import functional as F

    class CLASSNAME(BASE):
        def __init__(self):
            super().__init__(INIT_ARGS)
            BODY_INIT

        def forward(self, x):
            return BODY_FORWARD

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = CLASSNAME()

        def forward(self, x):
            return self.norm(x)

    def get_inputs():
        return [torch.randn(2, 8, 5, 5)]

    def get_init_inputs():
        return []
    """


def _user_norm(classname, base="nn.Module", init_args="", body_init="pass", body_forward="F.batch_norm(x, self.rm, self.rv, self.w, self.b, False)"):
    return (_USER_NORM_SRC.replace("CLASSNAME", classname).replace("BASE", base).replace("INIT_ARGS", init_args)
            .replace("BODY_INIT", body_init).replace("BODY_FORWARD", body_forward))


# running statistics as (frozen) nn.Parameters, affine values as buffers: the
# reviewer's identity-3 reproducer; the connected tensors are the STATISTICS
_SWAPPED_ROLES_INIT = ("self.rm = nn.Parameter(torch.zeros(8), requires_grad=False); self.rv = nn.Parameter(torch.ones(8), requires_grad=False); "
                       "self.register_buffer('w', torch.ones(8)); self.register_buffer('b', torch.zeros(8))")


def _attached_roles(g, op="batch_norm"):
    return [k.rsplit(".", 1)[-1] for k in g["connections"]["inputs"] if k.startswith(f"Model.{op}.")]


class TestRound4Identity:
    """independent review rounds (identity-3/4/5): argument roles come ONLY from
    the executed recorded call's binding; a class name, a module class, a forward object or a model
    object never establishes (or denies) a role.  Round 6 removed the library-identity exception."""

    @pytest.mark.parametrize("classname", ["CustomNorm", "BatchNorm2d"])
    def test_user_class_name_is_irrelevant_roles_come_from_the_call(self, tmp_path, classname):
        """F.batch_norm(x, rm, rv, w, b) where rm/rv are (frozen) Parameters and w/b buffers: the
        connected Parameters ARE the statistics of that call; the detached weight/bias are attached."""
        g, e, a, analysis, perf = _stages(tmp_path, _user_norm(classname, body_init=_SWAPPED_ROLES_INIT), "batch_norm")
        assert "norm_params_unattached" not in g["module_args"] and "owner_module_class" not in g["module_args"]
        kinds = g["module_args"]["operand_kinds"]
        assert kinds[0] == ["activation", "parameter", "parameter"] and kinds[1]["weight"] == kinds[1]["bias"] == "detached"
        assert _attached_roles(g) == ["weight", "bias"]
        assert g["input_shapes"] == [[2, 8, 5, 5], [8], [8], [8], [8]] and a["unfused_elements"] == 832 and perf.get("status") is None

    def test_user_subclass_making_the_library_call_is_bound_like_the_library(self, tmp_path):
        """A user subclass whose forward makes the library's call (statistics positional, affine
        Parameters keyword-bound) gets the library's roles — from the call, not from the base class."""
        g, e, a, analysis, perf = _stages(tmp_path, _user_norm("MyBN", base="nn.BatchNorm2d", init_args="8", body_init="pass",
                                                               body_forward="F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False)"), "batch_norm")
        assert "norm_params_unattached" not in g["module_args"]
        assert _attached_roles(g) == ["running_mean", "running_var"] and a["unfused_elements"] == 832

    @pytest.mark.parametrize("ctor,op,expected", [("nn.BatchNorm2d(8)", "batch_norm", ["running_mean", "running_var"]),
                                                  ("nn.InstanceNorm2d(8, affine=True, track_running_stats=True)", "instance_norm", ["running_mean", "running_var"])])
    def test_torch_module_statistics_are_bound_positionally(self, tmp_path, ctor, op, expected):
        g, e, a, analysis, perf = _stages(tmp_path, _model(ctor, (2, 8, 5, 5)), op)
        assert "norm_params_unattached" not in g["module_args"] and "owner_module_class" not in g["module_args"]
        assert isinstance(g["module_args"].get("owner_module_id"), int)      # provenance stays a diagnostic
        assert _attached_roles(g, op) == expected and a["unfused_elements"] == 832 and perf.get("status") is None

    def _synthetic_norm(self, g, kinds, w_type="parameter-tensor", callee="keep", margs_override=None):
        """The nn.BatchNorm2d capture's norm node rebuilt without a model object, with the recorded
        binding replaced by ``kinds`` (None drops it), the first Parameter node typed ``w_type``, the
        recorded callee kept, dropped (None) or replaced, and ``margs_override`` applied last (a value of
        None drops the key)."""
        from solar.common.types import NodeInfo
        x_id, w_id, b_id = [c for c in g["connections"]["inputs"] if not c.startswith("Model.batch_norm.")]
        nodes = {x_id: NodeInfo(node_id=x_id, type="auxiliary-tensor", node_class="TensorNode", output_shapes=[[2, 8, 5, 5]]),
                 w_id: NodeInfo(node_id=w_id, type=w_type, node_class="TensorNode", output_shapes=[[8]]),
                 b_id: NodeInfo(node_id=b_id, type="parameter-tensor", node_class="TensorNode", output_shapes=[[8]])}
        margs = {k: v for k, v in g["module_args"].items() if k not in ("norm_params_unattached", "operand_kinds")}
        if kinds is not None:
            margs["operand_kinds"] = kinds
        if callee is None:
            margs.pop("callee", None)
        elif callee != "keep":
            margs["callee"] = callee
        for k, v in (margs_override or {}).items():
            if v is None:
                margs.pop(k, None)
            else:
                margs[k] = v
        n = NodeInfo(node_id="Model.batch_norm", type="batch_norm", node_class="FunctionNode", input_nodes=[x_id, w_id, b_id], output_nodes=[],
                     input_shapes=[[2, 8, 5, 5], [8], [8]], output_shapes=[[2, 8, 5, 5]], input_dtypes=["torch.float32"] * 3, output_dtypes=["torch.float32"],
                     input_types=["input", "weight", "weight"], output_types=["output"], module_args=margs)
        return n, nodes

    def test_without_a_recorded_binding_nothing_is_synthesized(self, tmp_path):
        """Same nn.BatchNorm2d capture, re-run on the graph nodes without the capture-time binding
        (and without any model object): the slots are not guessed from the four equal shapes."""
        from solar.graph.torchview_processor import TorchviewProcessor
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        n, nodes = self._synthetic_norm(g, None)
        with pytest.warns(RuntimeWarning, match="no operand binding"):
            TorchviewProcessor()._attach_detached_parameters([n] + list(nodes.values()), {n.node_id: n, **nodes})
        assert n.module_args["norm_params_unattached"] == "argument_mapping_unresolved" and len(n.input_shapes) == 3

    @pytest.mark.parametrize("kinds,w_type,match", [
        ([["activation", "parameter", "parameter"], {"weight": "parameter", "bias": "parameter"}], "parameter-tensor", "binds 4 tensor-valued"),
        ([["activation", "activation", "parameter"], {"weight": "detached", "bias": "detached"}], "parameter-tensor", "binds 'running_mean' to an activation"),
        ([["activation", "detached", "detached"], {"weight": "parameter", "bias": "parameter"}], "hidden-tensor", "binds 'weight' to an nn.Parameter"),
        ([["activation", "detached", "detached"], {"weight": "activation", "bias": "activation"}], "parameter-tensor", "binds 'weight' to an activation"),
    ])
    def test_a_binding_that_does_not_describe_the_connected_inputs_is_refused(self, tmp_path, kinds, w_type, match):
        from solar.graph.torchview_processor import TorchviewProcessor
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        n, nodes = self._synthetic_norm(g, kinds, w_type)
        with pytest.warns(RuntimeWarning, match=match):
            TorchviewProcessor()._attach_detached_parameters([n] + list(nodes.values()), {n.node_id: n, **nodes})
        assert n.module_args["norm_params_unattached"] == "argument_mapping_unresolved" and len(n.input_shapes) == 3


def _round4_model(kind, force_cpu=False):
    """independent review test_round4.py fixtures: native nn.BatchNorm2d, the same inside nn.Sequential,
    a user subclass, and a genuine instance whose forward is replaced (types.MethodType) by a call that
    passes the Parameters as the functional running statistics and the buffers as weight/bias."""
    subclass = "class UserBN(nn.BatchNorm2d):\n    pass\n" if kind == "subclass" else ""
    ctor = "UserBN(8)" if kind == "subclass" else "nn.Sequential(nn.Identity(), nn.BatchNorm2d(8), nn.Identity())" if kind == "sequential" else "nn.BatchNorm2d(8)"
    patch = ("        self.norm.weight.data.zero_()\n        self.norm.bias.data.fill_(1.)\n        self.norm.running_mean.fill_(2.)\n"
             "        self.norm.running_var.fill_(3.)\n        self.norm.forward = types.MethodType(swapped, self.norm)\n") if kind == "instance_override" else ""
    refusal = "        if x.device.type == 'meta':\n            raise RuntimeError('fixture requires the CPU fallback')\n" if force_cpu else ""
    return f"""import torch, types
from torch import nn
from torch.nn import functional as F
def swapped(self, x):
    return F.batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var, False)
{subclass}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = {ctor}
{patch}
    def forward(self, x):
{refusal}        return self.norm(x)
def get_inputs():
    return [torch.arange(400, dtype=torch.float32).reshape(2, 8, 5, 5) / 400]
def get_init_inputs():
    return []
"""


def _capture_spy(monkeypatch):
    """Spy on torchview.draw_graph: records the device and the module ids of each traced Model.

    The capability gate refuses to probe torchview while ``draw_graph`` is replaced in the
    process, so the capability is probed BEFORE the spy is installed (no-op on trees without the gate)."""
    import torchview
    try:
        from solar.graph.torchview_capability import probe_torchview_capability
        probe_torchview_capability()
    except Exception:
        pass
    calls = []; original = torchview.draw_graph
    def traced(*a, **kw):
        m = a[0] if a else kw.get("model")
        if type(m).__name__ == "Model":
            calls.append({"device": kw.get("device"), "root_id": id(m), "modules": {str(id(x)) for x in m.modules()}})
        return original(*a, **kw)
    monkeypatch.setattr(torchview, "draw_graph", traced)
    return calls


def _check_capture_paths(calls, args, force_cpu):
    assert calls and calls[0]["device"] == "meta"
    if force_cpu:
        assert [c["device"] for c in calls] == ["meta", "cpu"] and calls[0]["root_id"] != calls[1]["root_id"]
    else:
        assert len(calls) == 1
    assert str(args.get("owner_module_id")) in calls[-1]["modules"], "owner must resolve in the actually traced model"
    assert "owner_module_class" not in args


class TestRound4Provenance:
    """The reviewer's round-4 matrix (independent review) under the round-6 rule: roles from the
    executed recorded call only.  Native and nn.Sequential attach the positional statistics; a user
    subclass that inherits the library forward makes the library call and is bound the same way; the
    instance whose forward passes the Parameters as statistics gets weight/bias attached (the call's
    own detached slots).  Both capture paths (meta, CPU fallback)."""

    @pytest.mark.parametrize("kind", ["native", "sequential", "subclass", "instance_override"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_roles_follow_the_executed_call_on_both_capture_paths(self, tmp_path, monkeypatch, kind, force_cpu):
        calls = _capture_spy(monkeypatch)
        g, e, a, analysis, perf = _stages(tmp_path, _round4_model(kind, force_cpu), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert not args.get("norm_params_unattached")
        expected = ["weight", "bias"] if kind == "instance_override" else ["running_mean", "running_var"]
        assert _attached_roles(g) == expected
        assert a["unfused_elements"] == 832 and perf.get("status") is None


def _round5_model(kind, force_cpu=False):
    """independent review test_round5.py fixtures (identity-5): a genuine nn.BatchNorm2d whose
    executed call passes its Parameters as the running statistics and its buffers as weight/bias,
    installed as an instance functools.partial, a user subclass forward, or (by monkeypatch in the
    test) a replaced nn.BatchNorm2d.forward, a replaced _BatchNorm.forward, or a functools.wraps
    wrapper around F.batch_norm that reorders the arguments before delegating."""
    code = _round4_model("native", force_cpu)
    patch = ("self.norm = nn.BatchNorm2d(8)\n        self.norm.weight.data.zero_()\n        self.norm.bias.data.fill_(1.)\n"
             "        self.norm.running_mean.fill_(2.)\n        self.norm.running_var.fill_(3.)")
    code = code.replace("self.norm = nn.BatchNorm2d(8)", patch)
    if kind == "instance_partial":
        code = code.replace("import torch, types", "import torch, types, functools").replace(
            "self.norm.running_var.fill_(3.)", "self.norm.running_var.fill_(3.)\n        self.norm.forward = functools.partial(swapped, self.norm)")
    elif kind == "subclass_forward":
        code = code.replace("class Model(nn.Module):", "class UserBN(nn.BatchNorm2d):\n    def forward(self, x):\n        return swapped(self, x)\nclass Model(nn.Module):")
        code = code.replace("self.norm = nn.BatchNorm2d(8)", "self.norm = UserBN(8)")
    return code


class TestRound5Binding:
    """The reviewer's round-5 matrix (independent review, identity-5): whatever replaced or
    wrapped code path executes, the recorded F.batch_norm call carries the module's Parameters in
    the running_mean/running_var slots and detached buffers in weight/bias; the binding attaches
    weight/bias (never synthesizes statistics), on both capture paths."""

    KINDS = ["instance_partial", "subclass_forward", "child_class_forward", "base_class_forward", "functional_wrapper"]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_reordered_library_call_binds_parameters_as_statistics(self, tmp_path, monkeypatch, kind, force_cpu):
        import functools
        import torch.nn.functional as F
        from torch import nn
        calls = _capture_spy(monkeypatch)
        original_base_forward = nn.modules.batchnorm._BatchNorm.forward
        original_functional = F.batch_norm

        @functools.wraps(original_base_forward)
        def swapped(self, x):
            return original_functional(x, self.weight, self.bias, weight=self.running_mean, bias=self.running_var, training=False)

        if kind == "child_class_forward":
            monkeypatch.setattr(nn.BatchNorm2d, "forward", swapped)
        elif kind == "base_class_forward":
            monkeypatch.setattr(nn.modules.batchnorm._BatchNorm, "forward", swapped)
        elif kind == "functional_wrapper":
            depth = [0]

            @functools.wraps(original_functional)
            def wrapped(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
                if depth[0]:
                    return original_functional(input, running_mean, running_var, weight, bias, training, momentum, eps)
                depth[0] += 1
                try:
                    return original_functional(input, weight, bias, running_mean, running_var, training, momentum, eps)
                finally:
                    depth[0] -= 1
            monkeypatch.setattr(F, "batch_norm", wrapped)

        g, e, a, analysis, perf = _stages(tmp_path, _round5_model(kind, force_cpu), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert not args.get("norm_params_unattached")
        kinds = args["operand_kinds"]
        assert kinds[0] == ["activation", "parameter", "parameter"] and kinds[1]["weight"] == kinds[1]["bias"] == "detached"
        assert _attached_roles(g) == ["weight", "bias"], "the connected Parameters are the call's statistics; the detached affine slots are attached"
        assert g["input_shapes"] == [[2, 8, 5, 5], [8], [8], [8], [8]] and a["unfused_elements"] == 832 and perf.get("status") is None


def _round6_model(npos, force_cpu=False):
    """independent review test_round6.py fixtures (binding-6): native torch.batch_norm with the
    first ``npos`` of its nine ATen arguments positional and the rest by keyword; four distinct
    registered buffers (w, b, rm, rv), inference flags."""
    names = ["input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps", "cudnn_enabled"]
    values = ["x", "self.w", "self.b", "self.rm", "self.rv", "False", "0.1", "1e-5", "False"]
    args = values[:npos] + [f"{k}={v}" for k, v in zip(names[npos:], values[npos:])]
    refusal = "        if x.device.type == 'meta':\n            raise RuntimeError('fixture requires the CPU fallback')\n" if force_cpu else ""
    return f"""import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('w', torch.ones(8)); self.register_buffer('b', torch.zeros(8))
        self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))
    def forward(self, x):
{refusal}        return torch.batch_norm({", ".join(args)})
def get_inputs():
    return [torch.arange(400, dtype=torch.float32).reshape(2, 8, 5, 5) / 400]
def get_init_inputs():
    return []
"""


class TestRound6CalleeIdentity:
    """independent review binding-6: which signature table applies (functional wrapper vs native
    ATen entry point) comes from the RECORDED CALLEE identity, never from the positional-argument count;
    mixed positional/keyword native spellings bind every slot; an unrecorded or unknown callee is refused."""

    @pytest.mark.parametrize("npos", [1, 2, 3, 9])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_native_batchnorm_mixed_positional_keyword_roles(self, tmp_path, monkeypatch, npos, force_cpu):
        calls = _capture_spy(monkeypatch)
        g, e, a, analysis, perf = _stages(tmp_path, _round6_model(npos, force_cpu), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert args["callee"] == "torch._VariableFunctionsClass.batch_norm"
        assert not args.get("norm_params_unattached")
        assert sorted(_attached_roles(g)) == ["bias", "running_mean", "running_var", "weight"], _attached_roles(g)
        assert a["unfused_elements"] == 832 and perf.get("status") is None

    def test_functional_and_native_callees_are_recorded_distinctly(self, tmp_path):
        g_f, *_ = _stages(tmp_path / "f", _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm") if (tmp_path / "f").mkdir() is None else None
        assert g_f["module_args"]["callee"] == "torch.nn.functional.batch_norm"
        g_n, *_ = _stages(tmp_path / "n", _round6_model(9), "batch_norm") if (tmp_path / "n").mkdir() is None else None
        assert g_n["module_args"]["callee"] == "torch._VariableFunctionsClass.batch_norm"

    def test_callee_name_is_a_diagnostic_not_a_role_source(self, tmp_path):
        """independent review callee-7: the module-qualified callee name is recorded but never
        decides the roles; replacing it with a foreign or a native-looking string changes nothing."""
        from solar.graph.torchview_processor import TorchviewProcessor
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        assert g["module_args"]["binding_source"] == "signature" and g["module_args"]["callable_type"] == "function"
        for callee in (None, "torch._VariableFunctionsClass.batch_norm", "mypkg.batch_norm"):
            n, nodes = TestRound4Identity()._synthetic_norm(g, g["module_args"]["operand_kinds"], callee=callee)
            TorchviewProcessor()._attach_detached_parameters([n] + list(nodes.values()), {n.node_id: n, **nodes})
            assert "norm_params_unattached" not in n.module_args and len(n.input_shapes) == 5, callee

    @pytest.mark.parametrize("override,match", [
        ({"operand_bindings": None, "binding_source": None}, "no signature binding"),
        ({"callable_type": "function", "binding_source": "testing_override"}, "testing-override stub"),
        ({"operand_bindings": [["inp", "rm", "rv"], {"weight": "w", "bias": "b"}]}, "outside the batch_norm vocabulary"),
        ({"callable_type": "builtin_function_or_method", "binding_source": "testing_override",
          "binding_param_order": ["input", "running_mean", "running_var", "weight", "bias", "use_input_stats", "momentum", "eps", "cudnn_enabled"]},
         "disagrees with its ATen argument order"),
    ])
    def test_binding_evidence_boundaries_are_refused_not_guessed(self, tmp_path, override, match):
        """No binding, a binding that only exists through torch's testing-override stub for a wrapper, a
        binding outside the norm vocabulary, or a builtin whose stub order disagrees with its ATen order
        for a keyword-spelled call: explicit refusal with the incomplete marker, nothing synthesized."""
        from solar.graph.torchview_processor import TorchviewProcessor
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        n, nodes = TestRound4Identity()._synthetic_norm(g, g["module_args"]["operand_kinds"], margs_override=override)
        with pytest.warns(RuntimeWarning, match=match):
            TorchviewProcessor()._attach_detached_parameters([n] + list(nodes.values()), {n.node_id: n, **nodes})
        assert n.module_args["norm_params_unattached"] == "argument_mapping_unresolved" and len(n.input_shapes) == 3


def _adapter_model(npos, force_cpu=False, params_as_statistics=False):
    """independent review test_callee_identity.py fixture: the model calls F.batch_norm (= the
    installed native-order adapter) positionally in the ADAPTER's order (x, weight, bias, running_mean,
    running_var) with four distinct registered buffers; the second control passes frozen Parameters in
    the two statistics positions and buffers in the affine positions."""
    refusal = "        if x.device.type == 'meta':\n            raise RuntimeError('fixture requires the CPU fallback')\n" if force_cpu else ""
    if params_as_statistics:
        init = ("self.rm = nn.Parameter(torch.zeros(8), requires_grad=False); self.rv = nn.Parameter(torch.ones(8), requires_grad=False); "
                "self.register_buffer('w', torch.ones(8)); self.register_buffer('b', torch.zeros(8))")
    else:
        init = ("self.register_buffer('w', torch.ones(8)); self.register_buffer('b', torch.zeros(8)); "
                "self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))")
    return f"""import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        {init}
    def forward(self, x):
{refusal}        return F.batch_norm(x, self.w, self.b, self.rm, self.rv, False, 0.1, 1e-5)
def get_inputs():
    return [torch.ones(2, 8, 5, 5)]
def get_init_inputs():
    return []
"""


def _install_native_order_adapter(monkeypatch, npos):
    """The reviewer's adapter: a Python callable with functools.wraps(F.batch_norm) metadata, an accurate
    NATIVE-order ``__signature__`` of its own, dispatching through handle_torch_function with the first
    ``npos`` arguments positional, delegating to the real functional with all four correct tensors."""
    import functools, inspect
    import torch
    import torch.nn.functional as F
    original = F.batch_norm

    def adapter(input, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5):
        vals = [input, weight, bias, running_mean, running_var, training, momentum, eps]
        names = ["input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps"]
        if torch.overrides.has_torch_function_variadic(*vals[:5]):
            return torch.overrides.handle_torch_function(adapter, tuple(vals[:5]), *vals[:npos], **dict(zip(names[npos:], vals[npos:])))
        return original(input, running_mean, running_var, weight, bias, training, momentum, eps)

    actual_sig = inspect.signature(adapter)
    adapter = functools.wraps(original)(adapter)
    adapter.__signature__ = actual_sig
    monkeypatch.setattr(F, "batch_norm", adapter)
    return adapter


class TestRound8CalleeSignature:
    """independent review callee-7: roles come from the executed call bound to the callable's
    ACTUAL signature (the shim's binding), never from copied name metadata."""

    @pytest.mark.parametrize("npos", [0, 2, 3, 8])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_native_order_adapter_with_copied_metadata_binds_by_its_own_signature(self, tmp_path, monkeypatch, npos, force_cpu):
        adapter = _install_native_order_adapter(monkeypatch, npos)
        calls = _capture_spy(monkeypatch)
        g, e, a, analysis, perf = _stages(tmp_path, _adapter_model(npos, force_cpu), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert args["callee"] == "torch.nn.functional.batch_norm"          # the copied name: recorded, ignored
        assert args["callable_type"] == "function" and args["binding_source"] == "signature"
        assert args["binding_param_order"][:5] == ["input", "weight", "bias", "running_mean", "running_var"]
        assert not args.get("norm_params_unattached")
        assert sorted(_attached_roles(g)) == ["bias", "running_mean", "running_var", "weight"]
        assert a["unfused_elements"] == 832 and perf.get("status") is None

    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_adapter_with_parameters_in_the_statistics_slots_attaches_the_affine_buffers(self, tmp_path, monkeypatch, force_cpu):
        """The reviewer's second control: through the same adapter (npos=8) the model passes frozen
        Parameters in the adapter's running_mean/running_var positions and buffers in its weight/bias
        positions; the executed call binds the Parameters as the statistics, so the detached slots —
        and the attached reads — are weight/bias (the aggregate stays 832 either way)."""
        _install_native_order_adapter(monkeypatch, 8)
        calls = _capture_spy(monkeypatch)
        g, e, a, analysis, perf = _stages(tmp_path, _adapter_model(8, force_cpu, params_as_statistics=True), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert not args.get("norm_params_unattached")
        assert _attached_roles(g) == ["weight", "bias"]
        assert a["unfused_elements"] == 832 and perf.get("status") is None

    def test_builtin_type_is_recorded_for_native_calls(self, tmp_path):
        g, *_ = _stages(tmp_path, _round6_model(9), "batch_norm")
        assert g["module_args"]["callable_type"] == "builtin_function_or_method"
        assert g["module_args"]["binding_source"] == "testing_override"
        assert g["module_args"]["binding_param_order"][:5] == ["input", "weight", "bias", "running_mean", "running_var"]


class TestAttachedParameterProvenance:
    """Request from the change's owner (parameter_sources): an attached norm parameter must be a capture-style
    parameter-tensor node that the converter's parameter partition recognises (type ``parameter-tensor``,
    ``node_class TensorNode``), listed under the consumer's ``connections.inputs`` in a ``weight`` slot, and
    carry explicit provenance (``parameter_source`` on the node, ``attached_parameters`` on the consumer)."""

    def test_attached_statistics_carry_provenance_the_converter_can_bind(self, tmp_path):
        import yaml
        g, e, a, analysis, perf = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        graph = yaml.safe_load(open(tmp_path / "graph" / "pytorch_graph.yaml"))["layers"]
        recs = g["module_args"]["attached_parameters"]
        assert [r["role"] for r in recs] == ["running_mean", "running_var"]
        for r in recs:
            node = graph[r["node_id"]]
            assert node["type"] == "parameter-tensor" and node["node_class"] == "TensorNode"     # the converter's parameter partition
            assert node["module_args"]["parameter_source"] == r["source"] == f"recorded_call:Model.batch_norm:{r['role']}"
            assert node["module_args"]["parameter_name"] == r["role"] and node["module_args"]["parameter_of"] == "Model.batch_norm"
            assert g["connections"]["inputs"][r["slot"]] == r["node_id"] and g["input_types"][r["slot"]] == "weight"
            assert node["output_shapes"] == [r["shape"]] and node["output_dtypes"] == [r["dtype"]]
            assert node["connections"]["outputs"] == ["Model.batch_norm"] and node["connections"]["inputs"] == []
        # the converter keeps the slots typed weight; their traffic is the four 8-element reads (832 in total)
        assert e["tensor_types"]["inputs"] == ["input", "weight", "weight", "weight", "weight"] and a["unfused_elements"] == 832
        if "parameter_sources" in e:   # trees carrying an earlier change: the attached slots resolve to the attached nodes
            assert e["parameter_sources"]["inputs"][3:] == [r["node_id"] for r in recs]


class TestBindingDestinationSplit:
    """The shim's signature binding treats torch's ``out=`` destination like the operand-order binder
    does (INTEG-10 / an earlier fix's destination split): it is not an operand and is left out of the binding,
    so a call whose stub signature has no ``out`` parameter still binds its operands."""

    def test_out_destination_is_not_an_operand_for_the_binding(self):
        import torch
        from solar.graph._torchview_param_shim import _operand_bindings
        a, b, o = torch.randn(4, 6), torch.randn(6, 5), torch.empty(4, 5)
        with_out = _operand_bindings(torch.mm, (a, b), {"out": o})
        without = _operand_bindings(torch.mm, (a, b), {})
        assert with_out is not None and with_out[0] == ["input", "mat2"] and with_out[1] == {} and with_out[2] == "testing_override"
        assert with_out[:4] == without[:4]

    def test_mm_with_out_destination_still_captures_and_bills(self, tmp_path):
        """Runs where the capture accepts an out= destination (INTEG-10: an earlier fix's destination split in
        the operand-order binder); on a tree without it the fail-closed shim refuses the call at capture
        and this control is skipped, not faked."""
        from solar.graph import _torchview_param_shim as shim
        if not hasattr(shim, "_split_destination"):
            pytest.skip("INTEG-10: torch.mm(..., out=) captures only with an earlier fix's destination split (fix/alias-views-unfused-20261025)")
        src = """\
        import torch
        from torch import nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.w = nn.Parameter(torch.randn(6, 5))
            def forward(self, x):
                buf = torch.empty(x.shape[0], 5, dtype=x.dtype, device=x.device)
                return torch.mm(x, self.w, out=buf)
        def get_inputs():
            return [torch.randn(4, 6)]
        def get_init_inputs():
            return []
        """
        g, e, a = _pipeline(tmp_path, src)
        mm = [v for k, v in a["layers"].items() if v.get("type") in ("mm", "matmul")]
        assert len(mm) == 1 and mm[0]["macs"] == 4 * 6 * 5 and mm[0].get("status") is None


_FOUR_BUFFERS = ("self.register_buffer('w', torch.ones(8)); self.register_buffer('b', torch.zeros(8)); "
                 "self.register_buffer('rm', torch.zeros(8)); self.register_buffer('rv', torch.ones(8))")


def _body_model(body, init=_FOUR_BUFFERS, force_cpu=False):
    """A Model whose forward is ``body`` (a full statement list, 8-space indented), over [2, 8, 5, 5] ones."""
    refusal = "        if x.device.type == 'meta':\n            raise RuntimeError('fixture requires the CPU fallback')\n" if force_cpu else ""
    return f"""import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        {init}
    def forward(self, x):
{refusal}{body}
def get_inputs():
    return [torch.ones(2, 8, 5, 5)]
def get_init_inputs():
    return []
"""


class TestRound9UpdateModeDefaults:
    """independent review default-8: an update-mode flag the caller OMITTED takes the executed
    callable's own resolved default (recorded with the binding); default True is a statistics update and
    is refused as such, default False is eval, an explicit value overrides the default."""

    @pytest.mark.parametrize("default_training,explicit_training", [(True, None), (False, None), (True, False), (False, True)],
                             ids=["default-True-omitted", "default-False-omitted", "explicit-False-over-True", "explicit-True-over-False"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_omitted_default_is_the_executed_update_mode(self, tmp_path, monkeypatch, default_training, explicit_training, force_cpu):
        import functools, inspect
        import torch
        import torch.nn.functional as F
        calls = _capture_spy(monkeypatch); original = F.batch_norm; delegates = []

        def adapter(input, weight, bias, running_mean, running_var, training=default_training, momentum=0.1, eps=1e-5):
            vals = (input, weight, bias, running_mean, running_var)
            if torch.overrides.has_torch_function_variadic(*vals):
                kwargs = {} if explicit_training is None else {"training": training}      # the omitted default stays omitted
                return torch.overrides.handle_torch_function(adapter, vals, *vals, **kwargs)
            delegates.append(training)
            return original(input, running_mean, running_var, weight, bias, training, momentum, eps)
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, "batch_norm", adapter)
        explicit = "" if explicit_training is None else f", training={explicit_training}"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(f"        return F.batch_norm(x, self.w, self.b, self.rm, self.rv{explicit})", force_cpu=force_cpu), "batch_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        effective = default_training if explicit_training is None else explicit_training
        assert delegates and all(d == effective for d in delegates)
        if explicit_training is None:
            assert args["binding_defaults"]["training"] is default_training
        if effective:
            _incomplete(g, e, a, analysis, perf, "running_stats_update")
        else:
            assert not args.get("norm_params_unattached") and a["unfused_elements"] == 832 and perf.get("status") is None

    @pytest.mark.parametrize("default_flag,explicit_flag", [(True, None), (False, None), (True, False)], ids=["default-True-omitted", "default-False-omitted", "explicit-False-over-True"])
    def test_use_input_stats_default_is_handled_the_same_way(self, tmp_path, monkeypatch, default_flag, explicit_flag):
        import functools, inspect
        import torch
        import torch.nn.functional as F
        original = F.instance_norm

        def adapter(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=default_flag, momentum=0.1, eps=1e-5):
            vals = (input, running_mean, running_var, weight, bias)
            if torch.overrides.has_torch_function_variadic(*[v for v in vals if v is not None]):
                kwargs = {} if explicit_flag is None else {"use_input_stats": use_input_stats}
                return torch.overrides.handle_torch_function(adapter, tuple(v for v in vals if v is not None), *vals, **kwargs)
            return original(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, "instance_norm", adapter)
        explicit = "" if explicit_flag is None else f", use_input_stats={explicit_flag}"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(f"        return F.instance_norm(x, self.rm, self.rv, self.w, self.b{explicit})"), "instance_norm")
        effective = default_flag if explicit_flag is None else explicit_flag
        if effective:
            _incomplete(g, e, a, analysis, perf, "running_stats_update")
        else:
            assert not g["module_args"].get("norm_params_unattached") and a["unfused_elements"] == 832

    def test_unknown_update_mode_is_refused(self, tmp_path):
        """A recorded call with statistics but neither a supplied flag nor a recorded default: refused."""
        from solar.graph.torchview_processor import TorchviewProcessor
        g, *_ = _stages(tmp_path, _model("nn.BatchNorm2d(8)", (2, 8, 5, 5)), "batch_norm")
        kinds = [list(g["module_args"]["operand_kinds"][0]), {k: v for k, v in g["module_args"]["operand_kinds"][1].items() if k != "training"}]
        n, nodes = TestRound4Identity()._synthetic_norm(g, kinds, margs_override={"binding_defaults": {}})
        # drop the explicit training flag from the recorded call's STRUCTURED record (the module's own call
        # spells training=False by keyword); the text is stripped too but is no longer what the flag is read from
        n.module_args["raw_attributes"] = n.module_args["raw_attributes"].replace("training: False, ", "")
        with pytest.warns(RuntimeWarning, match="neither supplies 'training' nor records its default"):
            TorchviewProcessor()._attach_detached_parameters([n] + list(nodes.values()), {n.node_id: n, **nodes})
        assert n.module_args["norm_params_unattached"] == "argument_mapping_unresolved" and len(n.input_shapes) == 3

    @pytest.mark.parametrize("training", [False, True], ids=["eval", "updating"])
    @pytest.mark.parametrize("npos", [0, 2, 3])
    def test_no_signature_wrapper_keyword_order_is_refused_or_update_excluded(self, tmp_path, monkeypatch, training, npos):
        """The reviewer's round-8 v2 control: a functools.wraps wrapper WITHOUT __signature__ whose own def
        order differs from the wrapped one, spelling tensors by keyword: refused at capture (typed
        UnsupportedFrontendOperand) or, when it captures, excluded as a statistics update / unresolved."""
        import functools, inspect
        import torch
        import torch.nn.functional as F
        original = F.batch_norm

        def adapter(input, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5):
            vals = [input, weight, bias, running_mean, running_var, training, momentum, eps]
            names = ["input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps"]
            if torch.overrides.has_torch_function_variadic(*vals[:5]):
                return torch.overrides.handle_torch_function(adapter, tuple(vals[:5]), *vals[:npos], **dict(zip(names[npos:], vals[npos:])))
            return original(input, running_mean, running_var, weight, bias, training, momentum, eps)
        adapter = functools.wraps(original)(adapter)
        assert list(inspect.signature(adapter, follow_wrapped=False).parameters) != list(inspect.signature(adapter).parameters)
        monkeypatch.setattr(F, "batch_norm", adapter)
        try:
            g, e, a, analysis, perf = _stages(tmp_path, _body_model(f"        return F.batch_norm(x, self.w, self.b, self.rm, self.rv, {training}, 0.1, 1e-5)"), "batch_norm")
        except AssertionError:
            assert not (tmp_path / "graph" / "pytorch_graph.yaml").exists()       # the typed capture refusal left no artefact
            return
        _incomplete(g, e, a, analysis, perf, "running_stats_update" if training else "argument_mapping_unresolved")


class TestRound10NoneUpdateFlag:
    """independent review none-9: an explicit None update flag (keyword, positional, or a resolved
    default of None) does not establish the update mode — the same refusal as an omitted / unknown flag,
    never coerced to False; only a Boolean True / False establishes it. Reviewer file test_round9_v2.py."""

    @staticmethod
    def _install(monkeypatch, op, spelling, delegates, flag_value=None, truthy=False):
        import functools, inspect
        import torch
        import torch.nn.functional as F
        original = getattr(F, op)
        if op == "batch_norm":
            def adapter(input, weight, bias, running_mean, running_var, training=None, momentum=0.1, eps=1e-5):
                vals = (input, weight, bias, running_mean, running_var)
                if torch.overrides.has_torch_function_variadic(*vals):
                    pos = vals + (training,) if spelling.startswith("positional") else vals
                    kw = {"training": training} if spelling.startswith("keyword") else {}
                    return torch.overrides.handle_torch_function(adapter, vals, *pos, **kw)
                effective = bool(training) if truthy else (True if training is None else training)
                delegates.append({"supplied": training, "effective": effective})
                return original(input, running_mean, running_var, weight, bias, effective, momentum, eps)
        else:
            def adapter(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=None, momentum=0.1, eps=1e-5):
                vals = (input, running_mean, running_var, weight, bias)
                if torch.overrides.has_torch_function_variadic(*[v for v in vals if v is not None]):
                    pos = vals + (use_input_stats,) if spelling.startswith("positional") else vals
                    kw = {"use_input_stats": use_input_stats} if spelling.startswith("keyword") else {}
                    return torch.overrides.handle_torch_function(adapter, tuple(v for v in vals if v is not None), *pos, **kw)
                effective = bool(use_input_stats) if truthy else (True if use_input_stats is None else use_input_stats)
                delegates.append({"supplied": use_input_stats, "effective": effective})
                return original(input, running_mean, running_var, weight, bias, effective, momentum, eps)
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, op, adapter)
        return "F.batch_norm(x, self.w, self.b, self.rm, self.rv" if op == "batch_norm" else "F.instance_norm(x, self.rm, self.rv, self.w, self.b"

    @pytest.mark.parametrize("op", ["batch_norm", "instance_norm"])
    @pytest.mark.parametrize("spelling", ["omitted", "positional_none", "keyword_none"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_none_update_mode_cannot_be_assumed_eval(self, tmp_path, monkeypatch, op, spelling, force_cpu):
        """The reviewer's adapters: an optional update flag defaulting to None whose delegate reads None as
        automatic updating (calls the functional with True); omitted / positional None / keyword None all refuse."""
        calls = _capture_spy(monkeypatch); delegates = []
        call = self._install(monkeypatch, op, spelling, delegates)
        flag = "training" if op == "batch_norm" else "use_input_stats"
        body = "        return " + call + ("" if spelling == "omitted" else ", None" if spelling == "positional_none" else f", {flag}=None") + ")"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(body, force_cpu=force_cpu), op)
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert delegates and all(d["supplied"] is None and d["effective"] is True for d in delegates)
        assert flag not in (args.get("binding_defaults") or {}) or args["binding_defaults"][flag] is None
        _incomplete(g, e, a, analysis, perf, "argument_mapping_unresolved")
        assert args["norm_params_unattached"] == "argument_mapping_unresolved"

    @pytest.mark.parametrize("spelling,value", [("keyword", "0"), ("keyword", "1"), ("keyword", "'eval'"), ("positional", "0"), ("positional", "1")],
                             ids=["kw-int-0", "kw-int-1", "kw-str", "pos-int-0", "pos-int-1"])
    def test_non_boolean_flag_values_are_refused_not_coerced(self, tmp_path, monkeypatch, spelling, value):
        """Parser audit: an int or a string in the flag slot (keyword or positional) is recorded as unestablished
        and refused — whatever the adapter's delegate makes of it — never truth-tested into a Boolean flag."""
        delegates = []
        call = self._install(monkeypatch, "batch_norm", spelling, delegates, truthy=True)
        body = "        return " + call + (f", training={value})" if spelling == "keyword" else f", {value})")
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(body), "batch_norm")
        assert delegates and all(d["supplied"] is not None for d in delegates)
        _incomplete(g, e, a, analysis, perf, "argument_mapping_unresolved")
        assert g["module_args"]["norm_params_unattached"] == "argument_mapping_unresolved"

    @pytest.mark.parametrize("default_flag,explicit_flag", [(True, None), (False, None), (True, False), (False, True)],
                             ids=["default-True-omitted", "default-False-omitted", "explicit-False-over-True", "explicit-True-over-False"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_instance_norm_boolean_default_and_override_still_hold(self, tmp_path, monkeypatch, force_cpu, default_flag, explicit_flag):
        """The reviewer's v2 InstanceNorm matrix (both capture paths): Boolean defaults and overrides are unchanged."""
        import functools, inspect
        import torch
        import torch.nn.functional as F
        calls = _capture_spy(monkeypatch); original = F.instance_norm; delegates = []

        def adapter(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=default_flag, momentum=0.1, eps=1e-5):
            vals = (input, running_mean, running_var, weight, bias)
            if torch.overrides.has_torch_function_variadic(*[v for v in vals if v is not None]):
                kw = {} if explicit_flag is None else {"use_input_stats": use_input_stats}
                return torch.overrides.handle_torch_function(adapter, tuple(v for v in vals if v is not None), *vals, **kw)
            delegates.append(use_input_stats)
            return original(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, "instance_norm", adapter)
        extra = "" if explicit_flag is None else f", use_input_stats={explicit_flag}"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(f"        return F.instance_norm(x, self.rm, self.rv, self.w, self.b{extra})", force_cpu=force_cpu), "instance_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        effective = default_flag if explicit_flag is None else explicit_flag
        assert delegates and all(d == effective for d in delegates)
        if explicit_flag is None:
            assert args["binding_defaults"]["use_input_stats"] is default_flag
        if effective:
            _incomplete(g, e, a, analysis, perf, "running_stats_update")
        else:
            assert not args.get("norm_params_unattached") and a["unfused_elements"] == 832 and perf.get("status") is None


class TestRound11StructuredFlags:
    """independent review flagtext-10: the update-mode flag is bound from the shim's STRUCTURED record
    of the executed call's scalar argument (type + value), never from a substring of the stringified
    attributes: a Boolean-looking substring inside a mapping or a string establishes nothing. Reviewer
    file test_round10.py on the module helpers."""

    @staticmethod
    def _check(tmp_path, monkeypatch, op, spelling, force_cpu, value_expr):
        import functools, inspect
        import torch
        import torch.nn.functional as F
        calls = _capture_spy(monkeypatch); original = getattr(F, op); delegates = []
        flag = "training" if op == "batch_norm" else "use_input_stats"
        if op == "batch_norm":
            def adapter(input, weight, bias, running_mean, running_var, training=None, momentum=0.1, eps=1e-5):
                vals = (input, weight, bias, running_mean, running_var)
                if torch.overrides.has_torch_function_variadic(*vals):
                    return torch.overrides.handle_torch_function(adapter, vals, *(vals + (training,) if spelling == "positional" else vals),
                                                                 **({"training": training} if spelling == "keyword" else {}))
                effective = True if training is None else bool(training)
                delegates.append({"supplied": repr(training), "type": type(training).__name__, "effective": effective})
                return original(input, running_mean, running_var, weight, bias, effective, momentum, eps)
            call = "F.batch_norm(x, self.w, self.b, self.rm, self.rv"
        else:
            def adapter(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=None, momentum=0.1, eps=1e-5):
                vals = (input, running_mean, running_var, weight, bias)
                if torch.overrides.has_torch_function_variadic(*vals):
                    return torch.overrides.handle_torch_function(adapter, vals, *(vals + (use_input_stats,) if spelling == "positional" else vals),
                                                                 **({"use_input_stats": use_input_stats} if spelling == "keyword" else {}))
                effective = True if use_input_stats is None else bool(use_input_stats)
                delegates.append({"supplied": repr(use_input_stats), "type": type(use_input_stats).__name__, "effective": effective})
                return original(input, running_mean, running_var, weight, bias, effective, momentum, eps)
            call = "F.instance_norm(x, self.rm, self.rv, self.w, self.b"
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, op, adapter)
        body = "        return " + call + ", " + ("" if spelling == "positional" else flag + "=") + value_expr + ")"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(body, force_cpu=force_cpu), op)
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        assert delegates and all(d["type"] != "bool" for d in delegates)
        _incomplete(g, e, a, analysis, perf, "argument_mapping_unresolved")
        assert args["norm_params_unattached"] == "argument_mapping_unresolved"
        return args

    @pytest.mark.parametrize("value_expr", ["None", "0", "1", "0.0", "'True'", "'False'", "'eval'"])
    @pytest.mark.parametrize("op", ["batch_norm", "instance_norm"])
    @pytest.mark.parametrize("spelling", ["positional", "keyword"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_nonboolean_update_mode_refused(self, tmp_path, monkeypatch, op, spelling, force_cpu, value_expr):
        self._check(tmp_path, monkeypatch, op, spelling, force_cpu, value_expr)

    @pytest.mark.parametrize("container", ["mapping", "string"])
    @pytest.mark.parametrize("op", ["batch_norm", "instance_norm"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_keyword_nested_text_is_not_boolean(self, tmp_path, monkeypatch, op, force_cpu, container):
        """`training={'training': False}` / `training='training: False'`: the inner `training: False` text used to
        establish eval mode while the delegate truth-tested the non-empty value and ran with update mode True."""
        flag = "training" if op == "batch_norm" else "use_input_stats"
        value_expr = repr({flag: False}) if container == "mapping" else repr(flag + ": False")
        args = self._check(tmp_path, monkeypatch, op, "keyword", force_cpu, value_expr)
        recorded = args["operand_kinds"][1][flag]
        if container == "mapping":
            assert recorded == {flag: {"scalar": "bool", "value": False}}      # a container: its leaves are never a flag
        else:
            assert recorded == {"scalar": "opaque", "type": "str"}              # a string: opaque, no value read out of it
        assert f"{flag}: False" in str(args["raw_attributes"]), "the misleading substring IS in the text and is ignored"

    def test_structured_record_shapes(self, tmp_path, monkeypatch):
        """The shim's record for the executed call: Booleans / None / numbers carry type + value, strings and
        other objects are opaque, tensors keep their binding kind; the flag is read from it alone."""
        import functools, inspect
        import torch
        import torch.nn.functional as F
        original = F.batch_norm

        def adapter(input, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5, tag=None, mode=None):
            vals = (input, weight, bias, running_mean, running_var)
            if torch.overrides.has_torch_function_variadic(*vals):
                return torch.overrides.handle_torch_function(adapter, vals, *vals, training=training, momentum=momentum, eps=eps, tag=tag, mode=mode)
            return original(input, running_mean, running_var, weight, bias, training, momentum, eps)
        sig = inspect.signature(adapter); adapter = functools.wraps(original)(adapter); adapter.__signature__ = sig
        monkeypatch.setattr(F, "batch_norm", adapter)
        g, e, a, analysis, perf = _stages(tmp_path, _body_model("        return F.batch_norm(x, self.w, self.b, self.rm, self.rv, training=False, momentum=0.2, eps=1, tag='t', mode=None)"), "batch_norm")
        kinds = g["module_args"]["operand_kinds"]
        assert kinds[0] == ["activation", "detached", "detached", "detached", "detached"]
        assert kinds[1] == {"training": {"scalar": "bool", "value": False}, "momentum": {"scalar": "float", "value": 0.2}, "eps": {"scalar": "int", "value": 1},
                            "tag": {"scalar": "opaque", "type": "str"}, "mode": {"scalar": "NoneType", "value": None}}
        assert not g["module_args"].get("norm_params_unattached") and a["unfused_elements"] == 832 and perf.get("status") is None


class TestRound9BuiltinScalarKeywords:
    """independent review scalar-kw-8: scalar keywords (flags, momentum, eps) on a builtin whose
    stub order differs from ATen do not reorder the tensor operands; only tensor keywords are refused."""

    @pytest.mark.parametrize("scalar_keywords", [False, True], ids=["all-positional", "scalar-keywords"])
    @pytest.mark.parametrize("training", [False, True], ids=["eval", "updating"])
    @pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
    def test_scalar_keywords_do_not_reorder_tensor_operands(self, tmp_path, monkeypatch, scalar_keywords, training, force_cpu):
        calls = _capture_spy(monkeypatch)
        flags = f"use_input_stats={training}, momentum=0.1, eps=1e-5, cudnn_enabled=False" if scalar_keywords else f"{training}, 0.1, 1e-5, False"
        g, e, a, analysis, perf = _stages(tmp_path, _body_model(f"        return torch.instance_norm(x, self.w, self.b, self.rm, self.rv, {flags})", force_cpu=force_cpu), "instance_norm")
        args = g["module_args"]; _check_capture_paths(calls, args, force_cpu)
        if training:
            _incomplete(g, e, a, analysis, perf, "running_stats_update")
        else:
            # four registered buffers, all detached: every slot is attached by its ATen-position role
            assert not args.get("norm_params_unattached") and sorted(_attached_roles(g, "instance_norm")) == ["bias", "running_mean", "running_var", "weight"]
            assert a["unfused_elements"] == 832 and perf.get("status") is None
    # the refusal for a TENSOR keyword on the mismatched builtin stays pinned by
    # TestRound4Identity.test_binding_evidence_boundaries_are_refused_not_guessed (the builtin / testing_override case)
