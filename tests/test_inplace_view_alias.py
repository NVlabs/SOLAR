# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place view ops (``unsqueeze_`` chains) convert with one shape per tensor name.

torchview records an in-place view op's ``input_shapes`` AFTER the reshape
(``x.unsqueeze_(2)`` on [1,12,64,64] records input [1,12,1,64,64]) while the
tensor node wired into it keeps the producer's shape.  The pre-AF consistency
check (lineage the lineage rewrite) then sees the producer's output name with two shapes
and refuses the graph: the saved BigBird capture failed ``toeinsum_model`` with
49 such conflicts (Finding 8).  The converter now re-anchors an in-place view
op's input to the connected tensor's recorded shape when the element count is
unchanged, so the op is modelled as its functional view; a slot whose element
count differs is still refused.
"""
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict

import pytest
import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor

PRE = [1, 12, 64, 64]          # what Model.view produced
POST1 = [1, 12, 1, 64, 64]     # after the first unsqueeze_(2)
POST2 = [1, 12, 1, 1, 64, 64]  # after the second unsqueeze_(2)


def _tensor(name: str, shape, producer, consumers, kind="hidden-tensor") -> Dict[str, Any]:
    return {
        "type": kind, "node_class": "TensorNode",
        "input_shapes": [list(shape)] if producer else [],
        "output_shapes": [list(shape)] if consumers or kind != "output-tensor" else [],
        "input_dtypes": ["torch.float32"], "output_dtypes": ["torch.float32"],
        "input_types": ["input"], "output_types": ["output"],
        "module_args": {"hierarchical_name": name},
        "connections": {"inputs": [producer] if producer else [], "outputs": list(consumers)},
    }


def _op(name: str, fn: str, in_shape, out_shape, producer, consumer, raw_shape=None) -> Dict[str, Any]:
    raw_shape = tuple(raw_shape if raw_shape is not None else in_shape)
    return {
        "type": fn, "node_class": "FunctionNode",
        "input_shapes": [list(in_shape)], "output_shapes": [list(out_shape)],
        "input_dtypes": ["torch.float32"], "output_dtypes": ["torch.float32"],
        "input_types": ["input"], "output_types": ["output"],
        "module_args": {"function_name": fn,
                        "raw_attributes": f"[[Tensor(shape={raw_shape}, dtype=torch.float32), 2], {{}}]",
                        "hierarchical_name": name},
        "connections": {"inputs": [producer], "outputs": [consumer]},
    }


def _bigbird_style_graph(first_unsqueeze_recorded_input=POST1) -> Dict[str, Any]:
    """The capture pattern verbatim: each ``unsqueeze_`` records its POST-op input shape."""
    layers = {
        "Model.auxiliary-tensor": _tensor("Model.auxiliary-tensor", [12, 64, 64], None, ["Model.view"], "auxiliary-tensor"),
        "Model.view": _op("Model.view", "view", [12, 64, 64], PRE, "Model.auxiliary-tensor", "Model.hidden-tensor"),
        "Model.hidden-tensor": _tensor("Model.hidden-tensor", PRE, "Model.view", ["Model.unsqueeze_"]),
        "Model.unsqueeze_": _op("Model.unsqueeze_", "unsqueeze_", first_unsqueeze_recorded_input, POST1,
                                "Model.hidden-tensor", "Model.hidden-tensor_1"),
        "Model.hidden-tensor_1": _tensor("Model.hidden-tensor_1", POST1, "Model.unsqueeze_", ["Model.unsqueeze__1"]),
        "Model.unsqueeze__1": _op("Model.unsqueeze__1", "unsqueeze_", POST2, POST2, "Model.hidden-tensor_1", "Model.hidden-tensor_2"),
        "Model.hidden-tensor_2": _tensor("Model.hidden-tensor_2", POST2, "Model.unsqueeze__1", ["Model.mul"]),
        "Model.mul": _op("Model.mul", "mul", POST2, POST2, "Model.hidden-tensor_2", "Model.output-tensor"),
        "Model.output-tensor": _tensor("Model.output-tensor", POST2, "Model.mul", [], "output-tensor"),
    }
    layers["Model.auxiliary-tensor"]["output_shapes"] = [[12, 64, 64]]
    return {"model_name": "inplace_unsqueeze_chain", "layers": layers}


def _convert(tmp_path: Path, graph: Dict[str, Any]) -> Dict[str, Any]:
    gpath = tmp_path / "pytorch_graph.yaml"
    gpath.write_text(yaml.safe_dump(graph, sort_keys=False))
    out = tmp_path / "einsum"
    result = PyTorchToEinsum().convert(str(gpath), str(out), copy_graph=False)
    assert result is not None
    return yaml.safe_load((out / "einsum_graph_renamed.yaml").read_text())


def _shapes_by_name(einsum_graph: Dict[str, Any]) -> Dict[str, set]:
    seen: Dict[str, set] = {}
    for layer in einsum_graph["layers"].values():
        for side in ("inputs", "outputs"):
            for name, shape in zip(layer["tensor_names"][side], layer["tensor_shapes"][side]):
                seen.setdefault(name, set()).add(tuple(shape))
    return seen


def test_inplace_unsqueeze_chain_converts_with_pre_op_input_shapes(tmp_path: Path):
    g = _convert(tmp_path, _bigbird_style_graph())
    layers = g["layers"]
    assert layers["Model.unsqueeze_"]["tensor_shapes"] == {"inputs": [PRE], "outputs": [POST1]}
    assert layers["Model.unsqueeze__1"]["tensor_shapes"] == {"inputs": [POST1], "outputs": [POST2]}
    assert layers["Model.unsqueeze_"]["tensor_names"]["inputs"] == ["Model.view.Output"]
    assert layers["Model.unsqueeze__1"]["tensor_names"]["inputs"] == ["Model.unsqueeze_.Output"]
    assert all(len(shapes) == 1 for shapes in _shapes_by_name(g).values()), _shapes_by_name(g)


def test_alias_pass_rewrites_only_in_place_view_slots_with_equal_numel():
    conv = PyTorchToEinsum()
    assert conv._inplace_view_base("unsqueeze_") == "unsqueeze"
    assert conv._inplace_view_base("transpose_") == "transpose"
    assert conv._inplace_view_base("t_") == "t"
    assert conv._inplace_view_base("unsqueeze") is None     # functional
    assert conv._inplace_view_base("add_") is None           # in-place arithmetic
    assert conv._inplace_view_base("__getitem__") is None    # dunder
    assert conv._inplace_view_base("resize_") is None        # may change numel
    layers = _bigbird_style_graph()["layers"]
    ops = [k for k, v in layers.items() if v["node_class"] == "FunctionNode"]
    tensors = [k for k, v in layers.items() if v["node_class"] == "TensorNode"]
    assert conv._alias_inplace_view_input_shapes(layers, ops, tensors) == 2
    assert layers["Model.unsqueeze_"]["input_shapes"] == [PRE]
    assert layers["Model.unsqueeze__1"]["input_shapes"] == [POST1]
    assert layers["Model.mul"]["input_shapes"] == [POST2]
    assert conv._alias_inplace_view_input_shapes(layers, ops, tensors) == 0  # idempotent


def test_genuine_shape_inconsistency_is_still_refused(tmp_path: Path):
    """A slot whose element count differs is not a view alias: the check stays."""
    graph = _bigbird_style_graph(first_unsqueeze_recorded_input=[2, 12, 1, 64, 64])
    with pytest.raises(ValueError, match="Pre-AF tensor-shape inconsistency"):
        _convert(tmp_path, graph)


INPLACE_CHAIN_MODEL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            y = x.view(1, 12, 64, 64)
            y.unsqueeze_(2)
            y.unsqueeze_(2)
            return y * 2.0

    def get_inputs():
        return [torch.randn(12, 64, 64)]

    def get_init_inputs():
        return []
"""


def test_cpu_trace_of_inplace_unsqueeze_chain_converts(tmp_path: Path):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(INPLACE_CHAIN_MODEL))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    capture = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())["layers"]
    inplace = {k: v for k, v in capture.items() if v.get("type") == "unsqueeze_"}
    assert len(inplace) == 2, sorted(capture)
    # the capture pattern this test is about: recorded input == post-op shape
    first = min(inplace, key=len)
    if capture[first]["input_shapes"] != [POST1]:
        pytest.skip(f"tracer no longer records the post-op input shape: {capture[first]['input_shapes']}")
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    g = yaml.safe_load((einsum_dir / "einsum_graph_renamed.yaml").read_text())
    assert g["layers"][first]["tensor_shapes"]["inputs"] == [PRE]
    assert g["layers"][first]["tensor_shapes"]["outputs"] == [POST1]
    assert all(len(shapes) == 1 for shapes in _shapes_by_name(g).values())
