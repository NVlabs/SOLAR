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

"""Regression tests for gather-style traffic and op accounting (issue #38, an earlier decision).

Convention (docs/EINSUM_GUIDE.md, "Gather, index_select and embedding"):
a gather reads every index tensor fully, reads only the gathered elements
of the source (``numel(output)`` elements, as written), writes the output
fully, and costs one ``other_op`` per output element for the address
computation.  MACs are zero.  This covers ``torch.gather``,
``index_select``, ``take``/``take_along_dim``, ``embedding`` and advanced
indexing (``x[idx]``, a ``__getitem__`` with tensor indices).  Basic
slicing (``x[:, 1:]``) stays a zero-copy view and dense ops are untouched.

Hand formula per gather layer (elements):

    unfused_elements = sum(numel(index_i)) + numel(output) + numel(output)
    other_ops        = numel(output)
    macs             = 0

Before the fix: ``index_select``/``gather`` read the whole source and
cost one op per *source* element, ``embedding`` billed only the gathered
rows (no index read, no output write, zero ops) and advanced indexing was
treated as a slice view (output-sized read, no write, zero ops).
"""

import math
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
    return graph, analysis


def _src(forward, inputs, init=""):
    return f"""\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            {init}

        def forward(self, x, idx):
            return {forward}

    def get_inputs():
        return [{inputs}]

    def get_init_inputs():
        return []
    """


def _layer(layers, node_type):
    hits = [(k, v) for k, v in layers.items() if str(v.get("type")) == node_type]
    assert len(hits) == 1, f"expected one {node_type} layer, got {[k for k, _ in hits]}"
    return hits[0]


# (id, forward, inputs, node type, numel(index tensors), numel(output))
_CASES = [
    pytest.param("torch.gather(x, 2, idx)", "torch.randn(4, 8, 16), torch.randint(0, 16, (4, 8, 5))",
                 "gather", 4 * 8 * 5, 4 * 8 * 5, id="gather-dim2"),
    pytest.param("torch.index_select(x, 0, idx)", "torch.randn(6, 10), torch.randint(0, 6, (4,))",
                 "index_select", 4, 4 * 10, id="index_select-dim0"),
    pytest.param("torch.index_select(x, 0, idx)", "torch.randn(3, 10), torch.randint(0, 3, (8,))",
                 "index_select", 8, 8 * 10, id="index_select-output-larger-than-source"),
    pytest.param("self.emb(idx)", "torch.randn(1), torch.randint(0, 50, (2, 7))",
                 "embedding", 2 * 7, 2 * 7 * 8, id="embedding"),
]


class TestGatherTraffic:
    @pytest.mark.parametrize("forward,inputs,node_type,n_idx,n_out", _CASES)
    def test_gather_like(self, tmp_path, forward, inputs, node_type, n_idx, n_out):
        init = "self.emb = nn.Embedding(50, 8)" if "emb" in forward else ""
        _, analysis = _pipeline(tmp_path, _src(forward, inputs, init))
        _, layer = _layer(analysis["layers"], node_type)
        assert layer["macs"] == 0
        assert layer["other_ops"] == n_out                       # one address computation per output element
        assert layer["unfused_elements"] == n_idx + n_out + n_out  # index + gathered source + output
        assert analysis["total"]["macs"] == 0

    def test_advanced_indexing_with_tensor_index(self, tmp_path):
        """x[idx] with a tensor index is a gather, classified from the recorded
        index argument (not from whether the index survives as an input shape)."""
        _, analysis = _pipeline(tmp_path, _src("x[idx]", "torch.randn(6, 10), torch.randint(0, 6, (4,))"))
        _, layer = _layer(analysis["layers"], "__getitem__")
        assert layer["index_kind"] == "tensor"
        assert layer["macs"] == 0
        assert layer["other_ops"] == 40
        assert layer["unfused_elements"] == 4 + 40 + 40

    def test_basic_slice_stays_a_view(self, tmp_path):
        """Control: basic slicing is a pointer offset (output-sized read, no write, zero ops)."""
        _, analysis = _pipeline(tmp_path, _src("x[:, 1:]", "torch.randn(6, 10), torch.randint(0, 6, (4,))"))
        _, layer = _layer(analysis["layers"], "__getitem__")
        assert (layer["macs"], layer["other_ops"], layer["unfused_elements"]) == (0, 0, 6 * 9)

    def test_dense_control_unchanged(self, tmp_path):
        """Control: a dense matmul keeps full-tensor traffic and its MACs."""
        _, analysis = _pipeline(tmp_path, _src("torch.matmul(x, x.transpose(0, 1))",
                                               "torch.randn(6, 10), torch.randint(0, 6, (4,))"))
        _, layer = _layer(analysis["layers"], "matmul")
        assert layer["macs"] == 6 * 10 * 6
        assert layer["other_ops"] == 0
        assert layer["unfused_elements"] == 60 + 60 + 36


# ── Reviewer boundary controls (an earlier decision review, index-1 / bag-1) ──

class TestIndexClassification:
    """Basic slicing stays a zero-copy view; list and tensor indices are gathers,
    decided from the recorded index argument."""

    @pytest.mark.parametrize("expr,nout", [("x[:, [1, 3, 1]]", 18), ("x[[1, 3, 1]]", 30)])
    def test_list_literal_index_is_a_gather(self, tmp_path, expr, nout):
        torch = pytest.importorskip("torch")
        x = torch.arange(60).reshape(6, 10)
        assert eval(expr).untyped_storage().data_ptr() != x.untyped_storage().data_ptr()  # a copy, not a view
        _, analysis = _pipeline(tmp_path, _src(expr, "torch.arange(60).reshape(6, 10), torch.tensor([1, 3, 1])"))
        _, l = _layer(analysis["layers"], "__getitem__")
        assert l["index_kind"] == "sequence"
        assert (l["macs"], l["other_ops"], l["output_elements"]) == (0, nout, nout)
        assert l["unfused_elements"] == 2 * nout            # no index tensor to read

    # RECON-83-1 (closed): on the release branch the lineage base af_graph_builder's AF coverage check
    # rejected this graph (0-dim index tensor, a model input, predecessor of __getitem__ without a
    # tensor access) and the case was a strict xfail.  The lineage-regression fix R6 waives SIGNED
    # integral scalar index arguments (an earlier change, merged into the release branch as a lineage repair;
    # target the target), so since this branch's reconciliation the case is a required pass
    # (lane11 composition finding INTEG-12).  uint8 and bool scalars stay OUTSIDE the waiver
    # (an earlier review round: a byte/Boolean scalar is a mask, advanced indexing with distinct storage);
    # see the two controls below.
    def test_scalar_long_tensor_index_is_a_basic_view(self, tmp_path):
        torch = pytest.importorskip("torch")
        x = torch.arange(60).reshape(6, 10)
        assert x[torch.tensor(1)].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        _, analysis = _pipeline(tmp_path, _src("x[idx]", "torch.arange(60).reshape(6, 10), torch.tensor(1)"))
        _, l = _layer(analysis["layers"], "__getitem__")
        assert l["index_kind"] == "basic"
        assert (l["other_ops"], l["output_elements"], l["unfused_elements"]) == (0, 0, 10)

    @pytest.mark.parametrize("index,dtype", [("torch.tensor(1, dtype=torch.uint8)", "uint8"), ("torch.tensor(True)", "bool")])
    def test_scalar_uint8_and_bool_indices_stay_refused(self, tmp_path, index, dtype):
        """Negative controls for the waiver above: a 0-dim uint8 or bool index is a MASK (torch:
        advanced indexing, output [1, 6, 10] with distinct storage), not a basic scalar view.  The
        lineage waiver (an earlier review round) excludes both dtypes, and the AF emit of this MR's materialising
        getitem (``AB->R0AB``) does not read the scalar mask, so the AF coverage check REFUSES the
        graph instead of letting it pass as a zero-access view (fail-closed; recorded here so that
        a future change that silently waives either dtype is caught)."""
        torch = pytest.importorskip("torch")
        import warnings
        x = torch.arange(60).reshape(6, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = x[eval(index)]
        assert tuple(y.shape) == (1, 6, 10) and y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        with pytest.raises(RuntimeError, match="AF graph coverage check failed") as info:
            _pipeline(tmp_path, _src("x[idx]", f"torch.arange(60).reshape(6, 10), {index}"))
        assert "Model.__getitem__" in str(info.value) and "missing from AF tensor_accesses" in str(info.value)

    @pytest.mark.parametrize("expr,nout", [("x[:, 1:]", 54), ("x[2, ::2]", 5), ("x[None, ..., 1:3]", 12)])
    def test_basic_slices_stay_views(self, tmp_path, expr, nout):
        _, analysis = _pipeline(tmp_path, _src(expr, "torch.arange(60.).reshape(6, 10), torch.tensor(1)"))
        _, l = _layer(analysis["layers"], "__getitem__")
        assert l["index_kind"] == "basic"
        assert (l["other_ops"], l["output_elements"], l["unfused_elements"]) == (0, 0, nout)

    def test_nested_sequence_index_is_a_gather(self, tmp_path):
        _, analysis = _pipeline(tmp_path, _src("x[:, [[1, 2]]]", "torch.arange(60.).reshape(6, 10), torch.tensor(1)"))
        _, l = _layer(analysis["layers"], "__getitem__")
        assert l["index_kind"] == "sequence"
        assert (l["other_ops"], l["output_elements"], l["unfused_elements"]) == (12, 12, 24)

    @pytest.mark.parametrize("expr,inputs,nidx,nout", [
        ("torch.take(x, idx)", "torch.arange(60).reshape(6, 10), torch.tensor([0, 13, 0])", 3, 3),
        ("torch.take_along_dim(x, idx, dim=1)", "torch.arange(60).reshape(6, 10), torch.tensor([[1, 3, 1]])", 3, 18),
    ])
    def test_other_named_gathers(self, tmp_path, expr, inputs, nidx, nout):
        _, analysis = _pipeline(tmp_path, _src(expr, inputs))
        _, l = _layer(analysis["layers"], "take_along_dim" if "along" in expr else "take")
        assert (l["macs"], l["other_ops"], l["unfused_elements"]) == (0, nout, nidx + 2 * nout)

    def test_tensor_index_absent_from_input_shapes(self, tmp_path):
        """BigBird's Model.__getitem___12 shape: the index tensor is recorded in
        the call attributes but not as an input shape (saved capture)."""
        from solar.analysis.graph_analyzer import classify_getitem_index
        raw = "[[Tensor(shape=(64, 64), dtype=torch.float32), Tensor(shape=(2232,), dtype=torch.int64)], {}]"
        assert classify_getitem_index(raw, [64, 64], [2232, 64]) == {"kind": "tensor", "index_elements": 2232}
        layers = {
            "start": {"type": "start", "einsum_equation": "->AB", "elementwise_op": "copy", "reduction_op": "none",
                      "is_real_einsum": False, "is_einsum_supportable": False,
                      "tensor_names": {"inputs": [], "outputs": ["start.Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]},
                      "tensor_shapes": {"inputs": [], "outputs": [[64, 64]]}, "connections": {"inputs": [], "outputs": ["Model.g"]}},
            "Model.g": {"type": "__getitem__", "einsum_equation": "AB->R0B", "elementwise_op": "copy", "reduction_op": "none",
                        "is_real_einsum": False, "is_einsum_supportable": True,
                        "tensor_names": {"inputs": ["start.Output"], "outputs": ["Model.g.Output"]},
                        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                        "tensor_shapes": {"inputs": [[64, 64]], "outputs": [[2232, 64]]},
                        "connections": {"inputs": ["start"], "outputs": []}, "raw_attributes": raw}}
        p = tmp_path / "einsum_graph_renamed.yaml"
        with open(p, "w") as f:
            yaml.safe_dump({"model_name": "t", "layers": layers}, f, sort_keys=False)
        res = EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "analysis", copy_graph=False)
        l = res["layers"]["Model.g"]
        assert l["index_kind"] == "tensor"
        # the recovered index is its own read slot (typing-2), after the source's gathered rows
        assert l["memory_elements"]["inputs"] == [142_848, 2_232]
        assert l["memory_elements"]["outputs"] == [142_848]
        assert (l["other_ops"], l["unfused_elements"]) == (142_848, 287_928)

    def test_classifier_cases(self):
        from solar.analysis.graph_analyzer import classify_getitem_index as c
        T = "Tensor(shape=(6, 10), dtype=torch.float32)"
        assert c(f"[[{T}, [slice(None, None, None), slice(1, None, None)]], {{}}]", [6, 10], [6, 9])["kind"] == "basic"
        assert c(f"[[{T}, [2, slice(None, None, 2)]], {{}}]", [6, 10], [5])["kind"] == "basic"
        assert c(f"[[{T}, [1, 3, 1]], {{}}]", [6, 10], [3, 10])["kind"] == "sequence"       # list index
        assert c(f"[[{T}, [1, 3]], {{}}]", [6, 10], [])["kind"] == "basic"                   # tuple of ints
        assert c(f"[[{T}, [slice(None, None, None), [..., ..., ...]]], {{}}]", [6, 10], [6, 3])["kind"] == "sequence"
        assert c(f"[[{T}, [slice(None, None, None), Tensor(shape=(4,), dtype=torch.int64)]], {{}}]", [6, 10], [6, 4]) == {"kind": "tensor", "index_elements": 4}
        assert c(f"[[{T}, Tensor(shape=(), dtype=torch.int64)], {{}}]", [6, 10], [10])["kind"] == "basic"
        assert c(f"[[{T}, Tensor(shape=(6, 10), dtype=torch.bool)], {{}}]", [6, 10], [29])["kind"] == "tensor"
        assert c(None, [6, 10], [6, 9])["kind"] == "unavailable"


class TestEmbeddingBag:
    def test_sum_two_bags(self, tmp_path):
        """EmbeddingBag(8, 4, mode='sum'), indices [[0,1,2],[3,4,5]]: 24 gathered
        elements reduced to 8 outputs: reads 6 + 24, writes 8, ops 24 (address)
        + 16 (additions) = 40."""
        _, analysis = _pipeline(tmp_path, _src("self.emb(idx)", "torch.randn(1), torch.tensor([[0, 1, 2], [3, 4, 5]])",
                                               "self.emb = nn.EmbeddingBag(8, 4, mode='sum')"))
        _, l = _layer(analysis["layers"], "embedding_bag")
        assert l["bag_mode"] == "sum" and l["gathered_elements"] == 24
        assert l["memory_elements"]["inputs"] == [6, 24]
        assert l["input_elements"] == 6 + 24
        assert l["output_elements"] == 8
        assert l["other_ops"] == 24 + 16 and l["other_ops"] >= 16
        assert l["macs"] == 0
        assert l["unfused_elements"] == 6 + 24 + 8

    def test_mean_with_offsets(self, tmp_path):
        """1-D indices with offsets [0, 2] -> bags of 2 and 4 rows; mean adds one
        division per output element; offsets are read."""
        _, analysis = _pipeline(tmp_path, _src("self.emb(idx, torch.tensor([0, 2]))", "torch.randn(1), torch.tensor([0, 1, 2, 3, 4, 5])",
                                               "self.emb = nn.EmbeddingBag(8, 4, mode='mean')"))
        _, l = _layer(analysis["layers"], "embedding_bag")
        assert l["bag_mode"] == "mean"
        assert l["memory_elements"]["inputs"] == [6 + 2, 24]
        assert l["output_elements"] == 8
        assert l["other_ops"] == 24 + (24 - 8) + 8


_ATTR_REQUIREMENT = (
    "the capture must record call attributes for FunctionNodes: torchview with upstream #128 "
    "(collect_attributes propagated to input/hidden recorder tensors) or the torchview 0.2.7 "
    "release plus solar/patches/torchview-collect-attributes-inputs.patch as applied by "
    "validate-install; without them __getitem__ indices and embedding_bag modes cannot be classified"
)


class TestCaptureRequirement:
    def test_capture_records_call_attributes(self, tmp_path):
        graph, analysis = _pipeline(tmp_path, _src("x[:, [1, 3, 1]]", "torch.arange(60.).reshape(6, 10), torch.tensor(1)"))
        _, g = _layer(graph, "__getitem__")
        assert g["module_args"].get("raw_attributes"), _ATTR_REQUIREMENT
        _, l = _layer(analysis["layers"], "__getitem__")
        assert l["index_kind"] != "unavailable", _ATTR_REQUIREMENT


def _typed_getitem_case(tmp_path, *, connected, scalar_bool=False, index_dtype="torch.int64", render_dtype=True):
    """independent review saved-graph fixture: float32 source [6,10], an index tensor (int64 [3] or a
    0-dim bool), float32 output; the index either connected as an input or recovered from attributes."""
    import yaml
    from solar.analysis.graph_analyzer import EinsumGraphAnalyzer, classify_getitem_index
    ins = [6, 10]; idx = [] if scalar_bool else [3]; outs = [1, 6, 10] if scalar_bool else [3, 10]
    dtype = "torch.bool" if scalar_bool else index_dtype
    idx_render = f"Tensor(shape={tuple(idx)!r}, dtype={dtype})" if render_dtype else f"Tensor(shape={tuple(idx)!r})"
    raw = f"[[Tensor(shape=(6, 10), dtype=torch.float32), {idx_render}], {{}}]"
    def start(name, shape, dt):
        return {"type": "start", "einsum_equation": "->AB" if len(shape) == 2 else "->A", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                "tensor_names": {"inputs": [], "outputs": [name + ".Output"]}, "tensor_types": {"inputs": [], "outputs": ["input"]}, "tensor_shapes": {"inputs": [], "outputs": [shape]},
                "tensor_dtypes": {"inputs": [], "outputs": [dt]}, "connections": {"inputs": [], "outputs": ["Model.g"]}}
    layers = {"start": start("start", ins, "torch.float32")}
    if connected:
        layers["start_1"] = start("start_1", idx, dtype)
    layers["Model.g"] = {"type": "__getitem__", "einsum_equation": ("AB->R0AB" if scalar_bool else "AB,I->IB" if connected else "AB->IB"), "elementwise_op": "copy", "reduction_op": "none",
                         "is_real_einsum": False, "is_einsum_supportable": True, "tensor_names": {"inputs": ["start.Output"] + (["start_1.Output"] if connected else []), "outputs": ["Model.g.Output"]},
                         "tensor_types": {"inputs": ["input"] * (2 if connected else 1), "outputs": ["output"]}, "tensor_shapes": {"inputs": [ins] + ([idx] if connected else []), "outputs": [outs]},
                         "tensor_dtypes": {"inputs": ["torch.float32"] + ([dtype] if connected else []), "outputs": ["torch.float32"]}, "connections": {"inputs": ["start"] + (["start_1"] if connected else []), "outputs": []},
                         "raw_attributes": raw}
    p = tmp_path / "einsum_graph_renamed.yaml"; p.write_text(yaml.safe_dump({"model_name": "review", "layers": layers}, sort_keys=False))
    result = EinsumGraphAnalyzer().analyze_graph(p, tmp_path / "analysis", copy_graph=False)
    assert result is not None
    return classify_getitem_index(raw, ins, outs), result["layers"]["Model.g"], result


class TestDeltaV2:
    """independent review typing-1 / index-2 (the reviewer's probes, verbatim contracts)."""

    @pytest.mark.parametrize("connected", [True, False], ids=["connected-index", "attribute-only-index"])
    def test_int64_index_retains_its_own_byte_width(self, tmp_path, connected):
        c, l, _ = _typed_getitem_case(tmp_path, connected=connected)
        assert c == {"kind": "tensor", "index_elements": 3}
        assert l["unfused_elements"] == 63
        assert l["unfused_bytes"] == 30 * 4 + 3 * 8 + 30 * 4, "Raw-attribute index dtype must not inherit the source float32 width"
        if not connected:
            assert l["recovered_index"] == {"elements": 3, "dtype": "torch.int64"} and "status" not in l

    def test_scalar_bool_index_materializes_output(self, tmp_path):
        torch = pytest.importorskip("torch")
        x = torch.arange(60, dtype=torch.float32).reshape(6, 10); y = x[torch.tensor(True)]
        assert tuple(y.shape) == (1, 6, 10)
        assert y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        c, l, _ = _typed_getitem_case(tmp_path, connected=True, scalar_bool=True)
        assert (c["kind"], l["output_elements"], l["other_ops"]) == ("tensor", 60, 60), "A scalar bool is advanced indexing; scalar integer view rule does not apply"

    def test_scalar_long_index_stays_a_view_in_the_classifier(self):
        from solar.analysis.graph_analyzer import classify_getitem_index
        raw = "[[Tensor(shape=(6, 10), dtype=torch.float32), Tensor(shape=(), dtype=torch.int64)], {}]"
        assert classify_getitem_index(raw, [6, 10], [10]) == {"kind": "basic", "index_elements": 0}
        assert classify_getitem_index(raw, [6, 10], [10], with_dtype=True)["index_dtype"] is None
        raw_bool = "[[Tensor(shape=(6, 10), dtype=torch.float32), Tensor(shape=(), dtype=torch.bool)], {}]"
        assert classify_getitem_index(raw_bool, [6, 10], [1, 6, 10], with_dtype=True) == {"kind": "tensor", "index_elements": 1, "index_dtype": "torch.bool"}

    @pytest.mark.parametrize("connected", [True, False], ids=["connected-control", "recovered-index"])
    def test_recovered_index_bytes_reach_external_and_fused_totals(self, tmp_path, connected):
        """independent review typing-2 (the reviewer's probe): float32 source [6,10], int64 index
        [3], float32 output [3,10].  The recovered index is a separately typed read: external input
        144 B (120 B source + 24 B index), unfused == fused == 264 B, identical to the connected control,
        and the dtype-aware perf model carries the same bytes."""
        from solar.perf.perf_model import EinsumGraphPerfModel
        _, layer, a = _typed_getitem_case(tmp_path, connected=connected)
        p = EinsumGraphPerfModel(dtype_bytes=True).predict(str(tmp_path / "analysis" / "analysis.yaml"), str(tmp_path / "perf"), arch_config="H100_PCIe", precision="fp32")
        assert layer["unfused_bytes"] == 264
        assert a["total"]["external_input_bytes"] == 30 * 4 + 3 * 8 == 144
        assert a["total"]["fused_bytes"] == a["total"]["unfused_bytes"] == 264
        assert "status" not in a["total"]
        assert p["unfused"]["memory_bytes"] == 264
        assert p["fused"]["memory_bytes"] == p["fused_prefetched"]["memory_bytes"] == 264
        if not connected:
            assert layer["memory_elements"]["inputs"] == [30, 3] and layer["bytes_per_element"]["inputs"] == [4.0, 8.0]
            assert layer["recovered_index"] == {"elements": 3, "dtype": "torch.int64"}

    def test_recovered_index_of_unknown_dtype_keeps_the_marker_and_source_width(self, tmp_path):
        """Without a recorded index dtype the separate slot is priced at the source width and the
        layer stays explicitly incomplete (gather_index_dtype_unknown)."""
        _, layer, a = _typed_getitem_case(tmp_path, connected=False, render_dtype=False)
        assert layer["status"] == "incomplete" and layer["incomplete_reason"] == "gather_index_dtype_unknown"
        assert layer["memory_elements"]["inputs"] == [30, 3] and layer["bytes_per_element"]["inputs"] == [4.0, 4.0]
        assert a["total"]["incomplete_layers"] == {"Model.g": "gather_index_dtype_unknown"}

    def test_recovered_index_without_dtype_is_marked_incomplete(self, tmp_path):
        """A rendering without a dtype cannot be priced at the index width: the layer and the
        analysis say so instead of silently using the source width."""
        c, l, result = _typed_getitem_case(tmp_path, connected=False, render_dtype=False)
        assert c == {"kind": "tensor", "index_elements": 3}
        assert l["recovered_index"] == {"elements": 3, "dtype": None}
        assert l["status"] == "incomplete" and l["incomplete_reason"] == "gather_index_dtype_unknown"
        assert result["total"]["status"] == "incomplete" and result["total"]["incomplete_layers"] == {"Model.g": "gather_index_dtype_unknown"}
