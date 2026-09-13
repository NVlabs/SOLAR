# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import networkx as nx

from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def test_partition_nodes_treats_hidden_tensor_as_tensor_node():
    converter = PyTorchToEinsum()
    layers = {
        "Model.linear": {"type": "linear", "node_class": "FunctionNode"},
        "Model.hidden-tensor": {"type": "hidden-tensor", "node_class": "TensorNode"},
        "Model.auxiliary-tensor": {"type": "auxiliary-tensor", "node_class": "TensorNode"},
        "Model.parameter-tensor": {"type": "parameter-tensor", "node_class": "TensorNode"},
    }

    tensor_ids, op_ids, auxiliary_ids, parameter_ids = converter._partition_nodes(layers)

    assert "Model.linear" in op_ids
    assert "Model.hidden-tensor" in tensor_ids
    assert "Model.auxiliary-tensor" in auxiliary_ids
    assert "Model.parameter-tensor" in parameter_ids


def test_collect_start_node_info_preserves_given_order():
    converter = PyTorchToEinsum()
    layers = {
        "in_b": {
            "type": "auxiliary-tensor",
            "output_shapes": [[2, 3]],
            "connections": {"outputs": ["Model.op"]},
        },
        "in_a": {
            "type": "auxiliary-tensor",
            "output_shapes": [[4, 5]],
            "connections": {"outputs": ["Model.op"]},
        },
    }

    # Intentionally non-sorted order.
    info = converter._collect_start_node_info(layers, ["in_b", "in_a"], ["Model.op"])

    assert [x["original_id"] for x in info] == ["in_b", "in_a"]
    assert [x["index"] for x in info] == [0, 1]


def test_validate_input_types_alignment():
    """Test input_types / input_shapes alignment validation."""
    converter = PyTorchToEinsum()

    # Shorter input_types → padded with 'input'
    node_data = {
        "type": "linear",
        "input_shapes": [[2, 128, 256], [512, 256], [512]],
        "input_types": ["input"],
        "module_args": {"bias": True},
    }
    converter._validate_input_types_alignment("Model.linear", node_data)
    assert node_data["input_types"] == ["input", "input", "input"]

    # Longer input_types → raises
    node_data = {
        "type": "linear",
        "input_shapes": [[2, 128, 256]],
        "input_types": ["input", "weight", "bias"],
        "module_args": {},
    }
    with pytest.raises(ValueError):
        converter._validate_input_types_alignment("Model.linear", node_data)

    # Matching lengths → OK
    node_data = {
        "type": "linear",
        "input_shapes": [[2, 128, 256], [512, 256]],
        "input_types": ["input", "weight"],
        "module_args": {},
    }
    converter._validate_input_types_alignment("Model.linear", node_data)
    assert node_data["input_types"] == ["input", "weight"]


def test_convert_operation_remaps_hidden_tensor_input_to_predecessor_op():
    converter = PyTorchToEinsum()

    op_graph = nx.DiGraph()
    op_graph.add_node("Model.linear.bias_add")
    op_graph.add_node("Model.matmul")
    op_graph.add_edge("Model.linear.bias_add", "Model.matmul")

    node_data = {
        "type": "matmul",
        "input_shapes": [[2, 128, 512], [512, 64]],
        "output_shapes": [[2, 128, 64]],
        "input_types": ["input", "weight"],
        "output_types": ["output"],
        "connections": {
            # Raw graph still references hidden tensor.
            "inputs": ["Model.hidden-tensor", "Model.parameter-tensor_2"],
            "outputs": [],
        },
        "module_args": {},
    }

    out = converter._convert_operation(
        node_id="Model.matmul",
        node_data=node_data,
        op_graph=op_graph,
        start_nodes_info=[],
        start_node_id_map={"Model.parameter-tensor_2": "Model.parameter-tensor_2"},
    )

    assert out["tensor_names"]["inputs"][0] == "Model.linear.bias_add.Output"
    assert out["connections"]["inputs"] == ["Model.linear.bias_add"]


def test_multi_output_consumer_names_carry_output_slot(tmp_path):
    """Consumers of chunk/split outputs must reference the slot they read.

    Regression: every consumer of a multi-output op was named
    ``<producer>.Output``, collapsing all partition slices onto slot 0.
    This broke consumer attribution (Output_1/Output_2 appeared consumed
    by nobody) and made distinct chunk reads indistinguishable in the
    fused-model external-read deduplication.
    """
    import yaml
    from textwrap import dedent
    from solar.common.types import ProcessingConfig
    from solar.graph import PyTorchProcessor

    model_source = dedent(
        """\
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, x):
                a, b, c = x.chunk(3, dim=1)
                return a + b + c

        def get_inputs():
            return [torch.randn(8, 12)]

        def get_init_inputs():
            return []
        """
    )
    model_file = tmp_path / "model.py"
    model_file.write_text(model_source)
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)
    )
    assert processor.process_model_file(str(model_file), str(graph_dir))
    converter = PyTorchToEinsum()
    assert converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))

    with open(einsum_dir / "einsum_graph.yaml") as f:
        graph = yaml.safe_load(f)
    layers = graph["layers"]

    chunk_outputs = layers["Model.chunk"]["tensor_names"]["outputs"]
    assert chunk_outputs == [
        "Model.chunk.Output",
        "Model.chunk.Output_1",
        "Model.chunk.Output_2",
    ]
    # add(a, b) reads slots 0 and 1; add_1(<add>, c) reads slot 2.
    assert layers["Model.add"]["tensor_names"]["inputs"] == [
        "Model.chunk.Output",
        "Model.chunk.Output_1",
    ]
    assert layers["Model.add_1"]["tensor_names"]["inputs"] == [
        "Model.add.Output",
        "Model.chunk.Output_2",
    ]

    chunk = layers["Model.chunk"]
    chunk_access = chunk["access"]["outputs"]
    assert [entry["index"] for entry in chunk_access] == [0, 1, 2]
    assert [entry["base_tensor"] for entry in chunk_access] == [
        chunk["tensor_names"]["inputs"][0],
        chunk["tensor_names"]["inputs"][0],
        chunk["tensor_names"]["inputs"][0],
    ]
    assert [entry["base_shape"] for entry in chunk_access] == [
        [8, 12],
        [8, 12],
        [8, 12],
    ]
    assert [entry["boxes"] for entry in chunk_access] == [
        [[[0, 8], [0, 4]]],
        [[[0, 8], [4, 8]]],
        [[[0, 8], [8, 12]]],
    ]


def test_slice_access_metadata_is_emitted(tmp_path):
    """Slice nodes carry explicit region metadata for downstream consumers."""
    import yaml
    from textwrap import dedent
    from solar.common.types import ProcessingConfig
    from solar.graph import PyTorchProcessor

    model_source = dedent(
        """\
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, x):
                return x[0:10] + x[5:15]

        def get_inputs():
            return [torch.randn(16, 64)]

        def get_init_inputs():
            return []
        """
    )
    model_file = tmp_path / "model.py"
    model_file.write_text(model_source)
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)
    )
    assert processor.process_model_file(str(model_file), str(graph_dir))
    converter = PyTorchToEinsum()
    assert converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))

    with open(einsum_dir / "einsum_graph.yaml") as f:
        graph = yaml.safe_load(f)
    getitems = [layer for layer in graph["layers"].values() if layer["type"] == "__getitem__"]
    getitems = sorted(
        getitems,
        key=lambda layer: layer["access"]["inputs"][0]["boxes"][0][0][0],
    )

    assert [layer["access"]["inputs"][0]["boxes"] for layer in getitems] == [
        [[[0, 10], [0, 64]]],
        [[[5, 15], [0, 64]]],
    ]
    for layer in getitems:
        entry = layer["access"]["inputs"][0]
        assert entry["index"] == 0
        assert entry["kind"] == "region"
        assert entry["base_shape"] == [16, 64]
        assert layer["access"]["outputs"][0]["boxes"] == entry["boxes"]
