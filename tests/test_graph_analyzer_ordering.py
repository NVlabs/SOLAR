import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


def test_consumer_mapping_is_order_independent(tmp_path):
    """Producer output must be marked intermediate regardless of layer order."""
    graph_path = tmp_path / "einsum_graph_renamed.yaml"
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()

    # Intentionally place consumer before producer to exercise ordering bug.
    graph = {
        "layers": {
            "Model.add": {
                "type": "add",
                "einsum_equation": "AB,AB->AB",
                "is_real_einsum": False,
                "tensor_shapes": {"inputs": [[2, 2], [2, 2]], "outputs": [[2, 2]]},
                "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.relu.Output", "Model.add.Bias"], "outputs": ["Model.add.Output"]},
                "connections": {"inputs": ["Model.relu", "start"], "outputs": []},
            },
            "Model.relu": {
                "type": "relu",
                "einsum_equation": "AB->AB",
                "is_real_einsum": False,
                "tensor_shapes": {"inputs": [[2, 2]], "outputs": [[2, 2]]},
                "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_names": {"inputs": ["Model.Input"], "outputs": ["Model.relu.Output"]},
                "connections": {"inputs": ["start"], "outputs": ["Model.add"]},
            },
        }
    }

    with open(graph_path, "w") as f:
        yaml.safe_dump(graph, f, sort_keys=False)

    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(graph_path, out_dir, copy_graph=False)
    assert analysis is not None

    relu = analysis["layers"]["Model.relu"]
    add = analysis["layers"]["Model.add"]

    # ReLU output is consumed by add -> intermediate output.
    assert relu["output_is_intermediate"] is True
    # Add consumes one graph-produced input and one external weight.
    assert add["input_is_intermediate"] is True
    assert add["model_io_elements"] == 8  # bias (4) + final output (4)
    assert add["fused_elements"] == 8
