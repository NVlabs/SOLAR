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


def test_validate_input_types_alignment_raises_on_mismatch():
    converter = PyTorchToEinsum()
    node_data = {
        "type": "linear",
        "input_shapes": [[2, 128, 256], [512, 256], [512]],
        # input_types must match input_shapes length 1:1.
        "input_types": ["input"],
        "module_args": {"bias": True},
    }

    with pytest.raises(ValueError, match="mismatched input metadata"):
        converter._validate_input_types_alignment("Model.linear", node_data)


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
