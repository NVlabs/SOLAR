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

"""Regression tests for the weight / model-I/O element breakdown (issue #36, an earlier decision).

Defect: ``memory_breakdown.weight_elements`` was 0 for every case.  The graph
processor and converter already type parameter operands ``weight`` in
``tensor_types`` (hand-checked on all 19 saved KernelBench rows: 259 of 259
``parameter-tensor`` inputs typed ``weight``), but the analyzer folded every
external read into ``model_io_elements`` and never emitted ``weight_elements``,
so the perf model read ``total.weight_elements`` as 0.

Convention (EINSUM_GUIDE.md, "Tensor Types"): ``weight_elements`` is the
PARAMETER FOOTPRINT of a layer — the numel of each distinct parameter it
consumes, once per consuming op, established by provenance (the capture's
``parameter-tensor`` node recorded per slot in ``parameter_sources``), never by
a tensor's name or by the ``weight`` label alone; ``weight_read_elements`` is
the DRAM read of those parameters under the op's read model.  The traffic
identities, per layer (the totals are layer sums):

    weight_read_elements + model_io_elements + intermediate_elements == unfused_elements
    fused_elements (per layer) == weight_read_elements + model_io_elements

This is a relabel of the same element counts: unfused/fused traffic, MACs and
``other_ops`` are unchanged, which the hand totals below assert.  Footprint and
read differ for gathers (embedding rows: fewer or more elements than the
table), for a parameter occupying two slots of one call (read twice, one
footprint) and for ``weight``-typed slots of unknown provenance (excluded from
the footprint, read as model I/O, layer marked incomplete).

Models are tiny and traced on CPU, as in test_graph_analyzer_memory.py.
"""

import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchview")

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf.perf_model import EinsumGraphPerfModel
from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pipeline(tmp_path: Path, model_source: str):
    """Trace -> einsum -> analysis.  Returns (analysis dict, analysis.yaml path, pytorch_graph layers)."""
    (tmp_path / "model.py").write_text(dedent(model_source))
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    ok = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)
    ).process_model_file(str(tmp_path / "model.py"), str(graph_dir))
    assert ok, "PyTorchProcessor.process_model_file failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    einsum_graph = PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert einsum_graph is not None, "PyTorchToEinsum.convert failed"

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    analysis = EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir),
        precision="fp32", copy_graph=False,
    )
    assert analysis is not None, "EinsumGraphAnalyzer.analyze_graph failed"
    with open(graph_dir / "pytorch_graph.yaml") as f:
        pt_layers = (yaml.safe_load(f) or {}).get("layers") or {}
    return analysis, analysis_dir / "analysis.yaml", pt_layers


def _counted_weight_slots(layer: dict):
    """Indices of the weight-typed input slots whose provenance the analyzer accepted
    (every weight-typed slot minus the slot indices it reports in weight_provenance_unresolved)."""
    types = (layer.get("tensor_types") or {}).get("inputs", [])
    unresolved = set(layer.get("weight_provenance_unresolved") or [])
    return [i for i, t in enumerate(types) if t == "weight" and i not in unresolved]


def _weight_typed_read_elems(layer: dict) -> int:
    """Sum memory_elements.inputs[i] over the counted weight slots (the read of the parameters)."""
    reads = (layer.get("memory_elements") or {}).get("inputs", [])
    return sum(reads[i] for i in _counted_weight_slots(layer) if i < len(reads))


def _captured_parameter_elems(pt_layers: dict) -> int:
    """Elements of every parameter-tensor node the capture recorded."""
    total = 0
    for node in pt_layers.values():
        if str(node.get("type", "")).lower() == "parameter-tensor":
            for shp in node.get("output_shapes") or []:
                n = 1
                for d in shp:
                    n *= int(d)
                total += n
    return total


def _weight_typed_numel(layer: dict) -> int:
    """Parameter footprint recomputed from the layer record: numel of the counted weight slots,
    each distinct parameter_sources entry once (two slots fed by one parameter are one footprint)."""
    shapes = (layer.get("tensor_shapes") or {}).get("inputs", [])
    sources = (layer.get("parameter_sources") or {}).get("inputs", [])
    total, seen = 0, set()
    for i in _counted_weight_slots(layer):
        if i >= len(shapes):
            continue
        src = sources[i] if i < len(sources) else None
        if src is not None:
            if src in seen:
                continue
            seen.add(src)
        n = 1
        for d in shapes[i]:
            n *= int(d)
        total += n
    return total


def _assert_partition(analysis: dict) -> None:
    """Traffic: weight_read + model_io + intermediate == unfused and fused == weight_read + model_io,
    per layer and in total; weight_elements is the parameter footprint of the weight-typed inputs."""
    for lid, layer in analysis["layers"].items():
        assert layer["weight_read_elements"] + layer["model_io_elements"] + layer["intermediate_elements"] \
            == layer["unfused_elements"], lid
        assert layer["fused_elements"] == layer["weight_read_elements"] + layer["model_io_elements"], lid
        assert layer["weight_read_elements"] == _weight_typed_read_elems(layer), lid
        assert layer["weight_elements"] == _weight_typed_numel(layer), lid
    total = analysis["total"]
    layers = analysis["layers"].values()
    # Totals: weight_elements, weight_read_elements and model_io_elements are plain LAYER SUMS
    # (total.model_io_elements is not the deduplicated F7 footprint — that is total.fused_elements).
    # The unfused identity is a per-layer statement: total.unfused/fused also carry graph-level
    # model I/O that no layer owns (declared outputs, gather indices).
    assert total["weight_elements"] == sum(l["weight_elements"] for l in layers)
    assert total["weight_read_elements"] == sum(l["weight_read_elements"] for l in layers)
    assert total["model_io_elements"] == sum(l["model_io_elements"] for l in layers)
    # weight_elements_unresolved (when present) is the layer sum of the excluded numel
    unresolved_layers = [lid for lid, l in analysis["layers"].items() if l.get("weight_provenance_unresolved")]
    if unresolved_layers:
        assert total["weight_provenance_unresolved_layers"] == unresolved_layers
        assert total["weight_elements_unresolved"] == sum(
            analysis["layers"][lid]["weight_elements_unresolved"] for lid in unresolved_layers)
        assert total["status"] == "incomplete"
        assert all(analysis["layers"][lid]["status"] == "incomplete" for lid in unresolved_layers)
    else:
        assert "weight_elements_unresolved" not in total


def _perf(analysis_path: Path, tmp_path: Path) -> dict:
    perf = EinsumGraphPerfModel(debug=False).predict(
        str(analysis_path), str(tmp_path / "perf"),
        arch_config="H100_PCIe", precision="fp32", copy_analysis=False,
    )
    assert perf is not None, "EinsumGraphPerfModel.predict failed"
    return perf


# ---------------------------------------------------------------------------
# Analyzer unit tests on synthetic einsum graphs (no tracing)
# ---------------------------------------------------------------------------
def _analyze_synthetic(tmp_path: Path, layers: dict) -> dict:
    graph = {"model_name": "synthetic", "layers": {
        "start": {
            "type": "start", "einsum_equation": "", "is_real_einsum": False,
            "tensor_shapes": {"inputs": [], "outputs": [[2, 16]]},
            "tensor_types": {"inputs": [], "outputs": ["input"]},
            "tensor_names": {"inputs": [], "outputs": ["Model.Input"]},
            "connections": {"inputs": [], "outputs": [next(iter(layers))]},
        },
        **layers,
    }}
    graph_path = tmp_path / "einsum_graph.yaml"
    with open(graph_path, "w") as f:
        yaml.safe_dump(graph, f, sort_keys=False)
    analysis = EinsumGraphAnalyzer().analyze_graph(str(graph_path), str(tmp_path), copy_graph=False)
    assert analysis is not None
    return analysis


LINEAR_RELU_SYNTH = {
    # linear: x[2,16] (input) · W[16,8] (weight) + b[8] (weight) -> h[2,8] -> relu -> out
    "Model.linear": {
        "type": "linear", "einsum_equation": "AB,BC->AC", "is_real_einsum": True,
        "tensor_shapes": {"inputs": [[2, 16], [16, 8], [8]], "outputs": [[2, 8]]},
        "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
        "tensor_names": {"inputs": ["Model.Input", "Model.linear.Weight", "Model.linear.Weight_1"],
                         "outputs": ["Model.linear.Output"]},
        "connections": {"inputs": ["start"], "outputs": ["Model.relu"]},
    },
    "Model.relu": {
        "type": "relu", "einsum_equation": "AB->AB", "is_real_einsum": False,
        "tensor_shapes": {"inputs": [[2, 8]], "outputs": [[2, 8]]},
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "tensor_names": {"inputs": ["Model.linear.Output"], "outputs": ["Model.relu.Output"]},
        "connections": {"inputs": ["Model.linear"], "outputs": []},
    },
}


class TestAnalyzerSplitSynthetic:
    """The analyzer labels weight-typed external reads as weight_elements and nothing else moves."""

    def test_linear_relu_hand_counts(self, tmp_path):
        a = _analyze_synthetic(tmp_path, LINEAR_RELU_SYNTH)
        lin, relu = a["layers"]["Model.linear"], a["layers"]["Model.relu"]
        # linear: reads x 32 (model input) + W 128 + b 8 (weights), writes 16 (intermediate)
        assert lin["weight_elements"] == 128 + 8
        assert lin["model_io_elements"] == 32
        assert lin["intermediate_elements"] == 16
        assert lin["unfused_elements"] == 32 + 128 + 8 + 16
        assert lin["fused_elements"] == 32 + 128 + 8
        # relu: reads 16 (intermediate), writes 16 (model output)
        assert relu["weight_elements"] == 0
        assert relu["model_io_elements"] == 16
        assert relu["intermediate_elements"] == 16
        t = a["total"]
        assert t["weight_elements"] == 136
        assert t["model_io_elements"] == 48
        assert t["intermediate_elements"] == 32
        assert t["unfused_elements"] == 216
        assert t["fused_elements"] == 32 + 136 + 16  # dedup external I/O, unchanged
        _assert_partition(a)

    def test_totals_are_the_same_numbers_as_before(self, tmp_path):
        """MACs, other_ops, unfused and fused are untouched by the relabel (hand values)."""
        t = _analyze_synthetic(tmp_path, LINEAR_RELU_SYNTH)["total"]
        # The relabel leaves MACs alone: the total equals the analyzer's own count
        # for the synthetic layer (256 on the lineage, which prices the
        # saved equation AB,BC->AC; the pre-lineage handler gave 512).
        own = EinsumAnalyzer().get_compute_cost(
            "linear", TensorShapes(inputs=[[2, 16], [16, 8], [8]], outputs=[[2, 8]]), equation="AB,BC->AC")
        assert t["macs"] == own == 2 * 16 * 8
        assert t["other_ops"] == 16
        assert t["unfused_elements"] == 216
        assert t["fused_elements"] == 184
        assert t["intermediate_elements"] == 32

    def test_norm_params_typed_weight_are_counted(self, tmp_path):
        """The earlier decision attach step types a norm's weight/bias `weight`; they land in weight_elements."""
        d = 16
        layers = {
            "Model.layer_norm": {
                "type": "layer_norm", "einsum_equation": "AB->AB", "is_real_einsum": False,
                "tensor_shapes": {"inputs": [[2, d], [d], [d]], "outputs": [[2, d]]},
                "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.layer_norm.Weight", "Model.layer_norm.Weight_1"],
                                 "outputs": ["Model.layer_norm.Output"]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        ln = a["layers"]["Model.layer_norm"]
        assert ln["weight_elements"] == 2 * d
        assert ln["model_io_elements"] == 2 * d * 2  # x in + out
        assert a["total"]["weight_elements"] == 2 * d
        _assert_partition(a)

    def test_legacy_einsum_operands_typed_weight_are_provenance_unknown(self, tmp_path):
        """provenance-1: on a graph saved WITHOUT parameter_sources, a `weight`-typed operand
        of an explicit einsum is the one label a pre-#46 converter gave to untraced activations
        (restored from raw_attributes).  The label alone is not provenance: the numel is excluded
        from the footprint, the read is model I/O and the layer is marked incomplete."""
        layers = {
            "Model.einsum": {
                "type": "einsum", "einsum_equation": "AB,BC,CD->AD", "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[2, 16], [16, 4], [4, 8]], "outputs": [[2, 8]]},
                "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.einsum.Weight", "Model.einsum.Weight_1"],
                                 "outputs": ["Model.einsum.Output"]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        e = a["layers"]["Model.einsum"]
        assert e["weight_elements"] == 0 and e["weight_read_elements"] == 0
        assert e["weight_provenance_unresolved"] == [1, 2]
        assert e["weight_elements_unresolved"] == 16 * 4 + 4 * 8
        assert e["status"] == "incomplete" and e["incomplete_reason"] == "weight_provenance_unresolved"
        assert e["model_io_elements"] == 2 * 16 + (16 * 4 + 4 * 8) + 2 * 8   # x + the two reads + out
        assert a["total"]["weight_elements_unresolved"] == 16 * 4 + 4 * 8
        assert a["total"]["weight_provenance_unresolved_layers"] == ["Model.einsum"]
        _assert_partition(a)

    def test_legacy_non_einsum_weight_label_is_trusted(self, tmp_path):
        """Without parameter_sources the only other source of a `weight` label is a capture
        parameter-tensor input, so a legacy linear/conv/norm slot still counts (the reviewer's
        shared-parameter control: two consumers of one parameter sum per consumer)."""
        layers = {}
        for n in ("a", "b"):
            layers[n] = {
                "type": "linear", "einsum_equation": "AB,BC->AC", "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[2, 2], [2, 2]], "outputs": [[2, 2]]},
                "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "shared.W"], "outputs": [f"{n}.out"]},
                "connections": {"inputs": ["start"], "outputs": []},
            }
        a = _analyze_synthetic(tmp_path, layers)
        assert a["total"]["weight_elements"] == 8 == a["total"]["weight_read_elements"]  # 4 per consumer
        assert a["total"]["model_io_elements"] == 16 and a["total"]["fused_elements"] == 16
        assert "weight_elements_unresolved" not in a["total"]
        _assert_partition(a)

    def test_parameter_sources_dedup_one_parameter_in_two_slots(self, tmp_path):
        """parameter-identity-1: with parameter_sources recorded, two slots fed by the same
        capture parameter node are ONE footprint; the read still counts both occurrences."""
        layers = {
            "Model.einsum": {
                "type": "einsum", "einsum_equation": "AB,AB,AB->AB", "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[2, 16], [2, 16], [2, 16]], "outputs": [[2, 16]]},
                "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.einsum.Weight", "Model.einsum.Weight_1"],
                                 "outputs": ["Model.einsum.Output"]},
                "parameter_sources": {"inputs": [None, "Model.parameter-tensor", "Model.parameter-tensor"]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        e = a["layers"]["Model.einsum"]
        assert e["weight_elements"] == 32 and e["weight_read_elements"] == 64
        assert "weight_provenance_unresolved" not in e and e.get("status") != "incomplete"
        _assert_partition(a)

    def test_parameter_sources_distinct_parameters_count_each(self, tmp_path):
        """Two slots fed by two different capture parameter nodes are two footprints (einsum included:
        with sources recorded the op type no longer matters)."""
        layers = {
            "Model.einsum": {
                "type": "einsum", "einsum_equation": "AB,BC,CD->AD", "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[2, 16], [16, 4], [4, 8]], "outputs": [[2, 8]]},
                "tensor_types": {"inputs": ["input", "weight", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.einsum.Weight", "Model.einsum.Weight_1"],
                                 "outputs": ["Model.einsum.Output"]},
                "parameter_sources": {"inputs": [None, "Model.parameter-tensor", "Model.parameter-tensor_1"]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        e = a["layers"]["Model.einsum"]
        assert e["weight_elements"] == 16 * 4 + 4 * 8 == e["weight_read_elements"]
        assert e.get("status") != "incomplete"
        _assert_partition(a)

    def test_weight_slot_without_source_is_unknown_when_sources_are_recorded(self, tmp_path):
        """With parameter_sources recorded, a `weight` slot the converter could not trace to a
        parameter node is provenance unknown whatever the op type (a linear here)."""
        layers = {
            "Model.linear": {
                "type": "linear", "einsum_equation": "AB,BC->AC", "is_real_einsum": True,
                "tensor_shapes": {"inputs": [[2, 16], [16, 8]], "outputs": [[2, 8]]},
                "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.linear.Weight"], "outputs": ["Model.linear.Output"]},
                "parameter_sources": {"inputs": [None, None]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        l = a["layers"]["Model.linear"]
        assert l["weight_elements"] == 0 and l["weight_read_elements"] == 0
        assert l["weight_provenance_unresolved"] == [1] and l["weight_elements_unresolved"] == 128
        assert l["status"] == "incomplete" and l["incomplete_reason"] == "weight_provenance_unresolved"
        assert l["model_io_elements"] == 32 + 128 + 16
        _assert_partition(a)

    def test_untyped_extra_inputs_are_model_io_not_weights(self, tmp_path):
        """INTEG-3: an input beyond the tensor_types list has NO parameter provenance; it is
        external model I/O (the SDPA decomposition's sub-layers carry no tensor_types and their
        softmax/scale operands were reported as weights by the old weight default)."""
        layers = {
            "Model.add": {
                "type": "add", "einsum_equation": "AB,B->AB", "is_real_einsum": False,
                "tensor_shapes": {"inputs": [[2, 16], [16]], "outputs": [[2, 16]]},
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input", "Model.add.Weight"], "outputs": ["Model.add.Output"]},
                "connections": {"inputs": ["start"], "outputs": []},
            },
        }
        a = _analyze_synthetic(tmp_path, layers)
        add = a["layers"]["Model.add"]
        assert add["weight_elements"] == 0 and add["weight_read_elements"] == 0
        assert add["model_io_elements"] == 32 + 16 + 32
        _assert_partition(a)


# ---------------------------------------------------------------------------
# End-to-end CPU traces: linear, conv, embedding, norm
# ---------------------------------------------------------------------------
LINEAR_SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)          # W 8x16 = 128, b 8
        def forward(self, x):
            return self.fc(x)

    def get_inputs():
        return [torch.randn(2, 16)]
    def get_init_inputs():
        return []
    """

CONV_SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3)      # W 4x3x3x3 = 108, b 4
        def forward(self, x):
            return self.conv(x)

    def get_inputs():
        return [torch.randn(1, 3, 8, 8)]
    def get_init_inputs():
        return []
    """

EMBEDDING_SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 4)       # table 8x4 = 32
        def forward(self, idx):
            return self.emb(idx)

    def get_inputs():
        return [torch.randint(0, 8, (2, 10))]
    def get_init_inputs():
        return []
    """

NORM_SRC = """\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(16)          # weight 16, bias 16
            self.fc = nn.Linear(16, 8, bias=False)  # W 8x16 = 128
        def forward(self, x):
            return self.fc(self.ln(x))

    def get_inputs():
        return [torch.randn(2, 16)]
    def get_init_inputs():
        return []
    """


class TestLinear:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        return _pipeline(tmp_path_factory.mktemp("linear"), LINEAR_SRC)

    def test_weight_elements_equal_parameter_count(self, run):
        a, _, pt = run
        assert a["total"]["weight_elements"] == 128 + 8 == _captured_parameter_elems(pt)

    def test_split_and_totals(self, run):
        a, _, _ = run
        t = a["total"]
        # x 32 in, out 16, intermediate: matmul out 16 written + 16 read by bias_add
        assert t["model_io_elements"] == 32 + 16
        assert t["intermediate_elements"] == 32
        assert t["unfused_elements"] == (32 + 128 + 16) + (16 + 8 + 16)
        assert t["fused_elements"] == 32 + 128 + 8 + 16
        assert t["macs"] == 2 * 16 * 8
        _assert_partition(a)

    def test_perf_breakdown(self, run, tmp_path):
        a, path, _ = run
        mb = _perf(path, tmp_path)["memory_breakdown"]
        assert mb["weight_elements"] == 136
        assert mb["weight_bytes"] == 136 * 4
        assert mb["model_io_elements"] == 48
        assert mb["weight_elements"] + mb["model_io_elements"] + mb["intermediate_elements"] \
            == a["total"]["unfused_elements"]


class TestConv:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        return _pipeline(tmp_path_factory.mktemp("conv"), CONV_SRC)

    def test_weight_elements_equal_parameter_count(self, run):
        a, _, pt = run
        assert a["total"]["weight_elements"] == 108 + 4 == _captured_parameter_elems(pt)

    def test_split_and_totals(self, run):
        a, _, _ = run
        t = a["total"]
        x, out = 1 * 3 * 8 * 8, 1 * 4 * 6 * 6
        assert t["model_io_elements"] == x + out
        assert t["intermediate_elements"] == 0
        assert t["unfused_elements"] == x + 112 + out
        assert t["fused_elements"] == x + 112 + out
        assert t["macs"] == 4 * 6 * 6 * 3 * 3 * 3
        _assert_partition(a)

    def test_perf_breakdown(self, run, tmp_path):
        _, path, _ = run
        mb = _perf(path, tmp_path)["memory_breakdown"]
        assert mb["weight_elements"] == 112 and mb["weight_bytes"] == 112 * 4


class TestEmbedding:
    """INTEG-1 decision: weight_elements is a PARAMETER-COUNT metric — the embedding table's
    numel, independent of how many rows the gather reads (that read rule is an earlier decision's traffic field,
    an earlier change: numel(indices) x D). The read itself is reported as weight_read_elements."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        return _pipeline(tmp_path_factory.mktemp("emb"), EMBEDDING_SRC)

    def test_weight_elements_is_the_parameter_count_not_the_gather_read(self, run):
        a, _, pt = run
        emb = next(l for l in a["layers"].values() if l["type"] == "embedding")
        assert emb["tensor_types"]["inputs"][-1] == "weight"
        assert _captured_parameter_elems(pt) == 32
        assert emb["weight_elements"] == 32                       # table 8 x 4, whatever the read rule
        assert emb["weight_read_elements"] == emb["memory_elements"]["inputs"][-1] > 0   # the gather read (rule of the base: 52; an earlier decision/an earlier change: 80)
        assert a["total"]["weight_elements"] == 32
        _assert_partition(a)

    def test_perf_breakdown(self, run, tmp_path):
        a, path, _ = run
        mb = _perf(path, tmp_path)["memory_breakdown"]
        assert mb["weight_elements"] == a["total"]["weight_elements"] > 0


class TestNorm:
    """On a target without the earlier decision attach step the norm's affine parameters are not
    captured (no node, no type); the identity 'weight_elements == elements of the
    captured parameter-tensor nodes' holds either way, and with an earlier change merged the
    layer_norm layer gains 2*d weight-typed inputs that this accounting counts
    (see TestAnalyzerSplitSynthetic.test_norm_params_typed_weight_are_counted)."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        return _pipeline(tmp_path_factory.mktemp("norm"), NORM_SRC)

    def test_weight_elements_equal_captured_parameters(self, run):
        a, _, pt = run
        assert a["total"]["weight_elements"] == _captured_parameter_elems(pt) > 0
        ln = next(l for l in a["layers"].values() if l["type"] == "layer_norm")
        assert ln["weight_elements"] == _weight_typed_read_elems(ln)
        _assert_partition(a)

    def test_linear_after_norm_counts_its_weight(self, run):
        a, _, _ = run
        lin = next(l for l in a["layers"].values() if l["type"] == "linear")
        assert lin["weight_elements"] == 128
        assert lin["input_is_intermediate"] is True
        assert lin["model_io_elements"] == 16  # only the model output; x comes from the norm


# ---------------------------------------------------------------------------
# Saved-graph fixtures (INTEG-3 exact GPT-2 bound + SDPA sub-layers; INTEG-2 Mamba2 provenance)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parameter identity through conversion (parameter-identity-1)
# ---------------------------------------------------------------------------
REPEATED_PARAM_SRC = """\
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 2))

        def forward(self, x):
            return torch.einsum('ij,ij,ij->ij', x, self.w, self.w)

    def get_inputs():
        return [torch.ones(2, 2)]

    def get_init_inputs():
        return []
"""

SHARED_PARAM_SRC = """\
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 2))

        def forward(self, x, y):
            return torch.einsum('ij,jk->ik', x, self.w) + torch.einsum('ij,jk->ik', y, self.w)

    def get_inputs():
        return [torch.ones(2, 2), torch.ones(2, 2)]

    def get_init_inputs():
        return []
"""


def _renamed_graph(analysis_path: Path) -> dict:
    with open(analysis_path.parent.parent / "einsum" / "einsum_graph_renamed.yaml") as f:
        return yaml.safe_load(f)


class TestParameterIdentity:
    def test_one_parameter_in_two_slots_of_one_call_is_one_footprint(self, tmp_path):
        """The reviewer's case: einsum('ij,ij,ij->ij', x, w, w) has ONE parameter-tensor node
        (numel 4) in two slots; footprint 4, read 8."""
        a, path, pt = _pipeline(tmp_path, REPEATED_PARAM_SRC)
        assert _captured_parameter_elems(pt) == 4
        layer = _renamed_graph(path)["layers"]["Model.einsum"]
        assert layer["tensor_types"]["inputs"] == ["input", "weight", "weight"]
        src = layer["parameter_sources"]["inputs"]
        assert src[0] is None and src[1] == src[2] and pt[src[1]]["type"] == "parameter-tensor"
        e = a["layers"]["Model.einsum"]
        assert e["weight_elements"] == 4 and e["weight_read_elements"] == 8
        assert a["total"]["weight_elements"] == 4 == _captured_parameter_elems(pt)
        assert e["unfused_elements"] == 4 + 8 + 4 and e["fused_elements"] == 16
        assert e.get("status") != "incomplete"
        _assert_partition(a)
        assert _perf(path, tmp_path)["memory_breakdown"]["weight_elements"] == 4

    def test_one_parameter_in_two_consumers_counts_once_per_consumer(self, tmp_path):
        """Footprint is per consuming op: one nn.Parameter consumed by two einsums -> 4 + 4.
        The capture records a parameter-tensor node PER CONSUMING CALL (two nodes here, one
        consumer each), so cross-consumer identity is not something the capture proves; the
        unique parameter inventory (4) is a different quantity from the documented per-consumer
        sum (8), and the per-consumer sum equals the captured node sum."""
        a, path, pt = _pipeline(tmp_path, SHARED_PARAM_SRC)
        param_nodes = {k: v for k, v in pt.items() if v.get("type") == "parameter-tensor"}
        assert len(param_nodes) == 2 and _captured_parameter_elems(pt) == 8
        assert all(len(v["connections"]["outputs"]) == 1 for v in param_nodes.values())
        layers = _renamed_graph(path)["layers"]
        sources = {lid: [s for s in l.get("parameter_sources", {}).get("inputs", []) if s]
                   for lid, l in layers.items() if l["type"] == "einsum"}
        assert len(sources) == 2 and all(len(v) == 1 for v in sources.values())
        assert {v[0] for v in sources.values()} == set(param_nodes)   # one node per consumer
        assert a["total"]["weight_elements"] == 8 == a["total"]["weight_read_elements"]
        assert all(a["layers"][lid]["weight_elements"] == 4 for lid in sources)
        _assert_partition(a)

    def test_linear_split_records_weight_and_bias_sources(self, tmp_path):
        """The linear -> matmul + bias_add split carries the parameter node of each slot."""
        a, path, pt = _pipeline(tmp_path, LINEAR_SRC)
        layers = _renamed_graph(path)["layers"]
        with_sources = {lid: l["parameter_sources"]["inputs"] for lid, l in layers.items()
                        if any(l.get("parameter_sources", {}).get("inputs") or [])}
        nodes = [s for v in with_sources.values() for s in v if s]
        assert len(with_sources) == 2 and len(nodes) == 2 and len(set(nodes)) == 2
        assert all(pt[n]["type"] == "parameter-tensor" for n in nodes)
        assert a["total"]["weight_elements"] == _captured_parameter_elems(pt)
