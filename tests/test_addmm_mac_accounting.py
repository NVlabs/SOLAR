# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torch.addmm (Hugging Face Conv1D) must be billed like nn.Linear (issue #30, an earlier decision).

``torch.addmm(bias, x[M, K], W[K, N])`` reached the einsum graph as an unknown
op with equation ``I->I`` and zero MACs, so every GPT-2 block linear (HF
``Conv1D``) was uncounted.  These tests run the real mechanical route on tiny
synthetic modules (CPU capture, no model download) and pin:

* an addmm layer is billed M*K*N MACs, with x, W and bias read and the output
  written (fused bytes = x + W + b + y, the hand minimum);
* a Conv1D-style module (``addmm``) and an ``nn.Linear`` of the same shapes
  bill the same MACs, the same other_ops (the bias add) and the same fused
  bytes: no new convention, the existing linear-with-bias split is reused;
* the handler path (``EinsumAnalyzer.get_einsum_op("addmm")``) yields a real
  einsum whose compute cost is M*K*N.

Review of the first repair (its review) added:

* the split binds addmm's operands by their semantic position (bias, mat1,
  mat2) even when all three are runtime inputs with no parameter provenance;
* broadcast biases ([1], [1, N], [M, 1]) must not shrink the matrix work, and
  bias traffic follows the bias's actual storage extent;
* the direct handler refuses incomplete or non-2D operand sets instead of
  synthesising a missing matrix.
"""
import textwrap

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig, TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.pytorch_processor import PyTorchProcessor

M, K, N = 64, 48, 80

ADDMM_MODEL = textwrap.dedent(f'''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """Hugging Face Conv1D shape: y = bias + x @ W with W stored [in, out]."""
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn({K}, {N}))
            self.bias = nn.Parameter(torch.zeros({N}))

        def forward(self, x):
            return torch.addmm(self.bias, x, self.weight)

    def get_inputs():
        return [torch.randn({M}, {K})]

    def get_init_inputs():
        return []
''')

LINEAR_MODEL = textwrap.dedent(f'''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear({K}, {N}, bias=True)

        def forward(self, x):
            return self.fc(x)

    def get_inputs():
        return [torch.randn({M}, {K})]

    def get_init_inputs():
        return []
''')


def _analyze(tmp_path, name, source):
    model_file = tmp_path / f"{name}.py"
    model_file.write_text(source)
    graph_dir = tmp_path / f"{name}_graph"
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(graph_dir)) is True
    einsum_dir = tmp_path / f"{name}_einsum"
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir),
                                     copy_graph=False, enable_rename=True) is not None
    analysis = EinsumGraphAnalyzer(debug=False).analyze_graph(
        einsum_dir / "einsum_graph_renamed.yaml", tmp_path / f"{name}_analysis",
        precision="fp32", copy_graph=False)
    assert analysis is not None
    return analysis


def test_addmm_is_billed_mkn_macs_and_hand_minimum_bytes(tmp_path):
    analysis = _analyze(tmp_path, "addmm_model", ADDMM_MODEL)
    total = analysis["total"]
    assert total["macs"] == M * K * N
    # x + W + bias + y, each tensor once (the fused model excludes the matmul->bias_add intermediate)
    assert total["fused_elements"] == M * K + K * N + N + M * N
    # the bias add is billed as one elementwise op per output, like nn.Linear's bias_add layer
    assert total["other_ops"] == M * N
    kinds = {layer["type"] for layer in analysis["layers"].values()}
    assert "matmul" in kinds and "add" in kinds


def test_conv1d_style_addmm_equals_nn_linear(tmp_path):
    addmm = _analyze(tmp_path, "addmm_model", ADDMM_MODEL)["total"]
    linear = _analyze(tmp_path, "linear_model", LINEAR_MODEL)["total"]
    for key in ("macs", "other_ops", "fused_elements", "unfused_elements"):
        assert addmm[key] == linear[key], key
    assert linear["macs"] == M * K * N


def test_addmm_handler_yields_a_real_einsum_with_mkn_cost():
    analyzer = EinsumAnalyzer()
    ts = TensorShapes(inputs=[[N], [M, K], [K, N]], outputs=[[M, N]])
    op = analyzer.get_einsum_op("addmm", ts)
    assert op.is_real_einsum
    assert op.equation.count(",") == 2 and "->" in op.equation
    assert int(analyzer.get_compute_cost("addmm", ts)) == M * K * N


# --- review regressions -------------------------------------------------

RUNTIME_INPUT_ADDMM_MODEL = textwrap.dedent(f'''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """Every addmm operand is a runtime input: no nn.Parameter provenance to lean on."""
        def forward(self, bias, x, w):
            return torch.addmm(bias, x, w)

    def get_inputs():
        return [torch.zeros({N}), torch.randn({M}, {K}), torch.randn({K}, {N})]

    def get_init_inputs():
        return []
''')


def test_all_runtime_input_addmm_binds_operands_by_position(tmp_path):
    analysis = _analyze(tmp_path, "runtime_addmm", RUNTIME_INPUT_ADDMM_MODEL)
    total = analysis["total"]
    assert total["macs"] == M * K * N
    assert total["other_ops"] == M * N
    assert total["fused_elements"] == M * K + K * N + N + M * N
    matmul = [v for k, v in analysis["layers"].items() if v["type"] == "matmul" and not k.endswith(".bias_add")]
    assert len(matmul) == 1
    assert matmul[0]["tensor_shapes"]["inputs"] == [[M, K], [K, N]]   # mat1 then mat2, never the bias
    # MK,KN->MN up to the converter's label renaming: two matrices contracted
    # over their shared inner rank, never the bias as a vector operand.
    import re
    eq = re.fullmatch(r"(\w)(\w),(\w)(\w)->(\w)(\w)", matmul[0]["einsum_equation"])
    assert eq, matmul[0]["einsum_equation"]
    m1, k1, k2, n1, m_out, n_out = eq.groups()
    assert k1 == k2 and (m1, n1) == (m_out, n_out) and len({m1, k1, n1}) == 3
    # every operand the split wires up is an op layer or a canonical start
    # node (runtime inputs), never a tensor node the collapse dropped
    layers = analysis["layers"]
    bias_add = next(v for k, v in layers.items() if k.endswith(".bias_add"))
    for layer in (matmul[0], bias_add):
        for src in layer["connections"]["inputs"]:
            assert src in layers or re.fullmatch(r"start(_\d+)?", src), (src, sorted(layers))


def test_runtime_input_addmm_matches_parameter_addmm(tmp_path):
    runtime = _analyze(tmp_path, "runtime_addmm", RUNTIME_INPUT_ADDMM_MODEL)["total"]
    param = _analyze(tmp_path, "addmm_model", ADDMM_MODEL)["total"]
    for key in ("macs", "other_ops", "fused_elements", "unfused_elements"):
        assert runtime[key] == param[key], key


@pytest.mark.parametrize("bias", [[N], [M, N], [1], [1, N], [M, 1]])
def test_handler_broadcast_bias_keeps_matrix_work_and_bias_traffic(bias):
    analyzer = EinsumAnalyzer()
    ts = TensorShapes(inputs=[bias, [M, K], [K, N]], outputs=[[M, N]])
    op = analyzer.get_einsum_op("addmm", ts)
    assert op.is_real_einsum
    assert op.get_compute_cost(ts) == M * K * N, (bias, op.equation)
    bias_dims = op.operands[0].dims
    assert len(bias_dims) == len(bias)                       # traffic: the bias keeps its own storage extent
    # the bias operand never redefines M or N below their matrix extents
    labels = dict(zip(bias_dims, bias))
    for label, size in labels.items():
        if label in ("M", "N"):
            assert size == {"M": M, "N": N}[label]


@pytest.mark.parametrize("inputs", [
    [[K], [K, N]],            # bias missing, mat1 is a vector
    [[2, M, K], [K, N]],      # rank-3 mat1
    [[M, K]],                 # a single matrix: never synthesise the other
    [],                       # nothing recorded
    [[N], [M, K], [2, K, N]], # rank-3 mat2 with a bias
    [[N], [M, K], [K + 1, N]],  # inner dimensions disagree
    [[3], [M, K], [K, N]],    # bias that broadcasts to neither 1 nor N
])
def test_handler_refuses_incomplete_or_non_2d_operands(inputs):
    with pytest.raises(ValueError):
        EinsumAnalyzer().get_einsum_op("addmm", TensorShapes(inputs=inputs, outputs=[[M, N]]))


def test_handler_two_matrix_fallback_is_a_checked_2d_matmul():
    ts = TensorShapes(inputs=[[M, K], [K, N]], outputs=[[M, N]])
    op = EinsumAnalyzer().get_einsum_op("addmm", ts)
    assert op.equation == "MK,KN->MN"
    assert op.get_compute_cost(ts) == M * K * N


def test_ci_unit_lane_selects_this_module():
    import re
    ci = (pathlib_root() / ".gitlab-ci.yml").read_text()
    unit = ci[ci.index("unit-tests:"):ci.index("regression-tests:")]
    assert "tests/test_addmm_mac_accounting.py" in unit


def pathlib_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def test_split_refuses_addmm_whose_operands_cannot_be_bound_by_position(tmp_path):
    """Three shapes recorded but the connected operands do not line up with
    them (the bias has no tensor node): the positional slots cannot be bound,
    so the converter refuses instead of splitting on a guess."""
    import yaml
    from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    ids = ["Model.auxiliary-tensor_x", "Model.auxiliary-tensor_W"]
    shapes = [[N], [M, K], [K, N]]
    layers = {}
    for name, shape in zip(ids, shapes[1:]):
        layers[name] = {"type": "auxiliary-tensor", "node_class": "TensorNode", "input_shapes": [],
                        "output_shapes": [shape], "input_dtypes": [], "output_dtypes": ["torch.float32"],
                        "input_types": [], "output_types": ["output"], "module_args": {},
                        "connections": {"inputs": [], "outputs": ["op"]}}
    layers["op"] = {"type": "addmm", "node_class": "FunctionNode", "input_shapes": shapes,
                    "output_shapes": [[M, N]], "input_dtypes": ["torch.float32"] * 3,
                    "output_dtypes": ["torch.float32"], "input_types": ["input"] * 3, "output_types": ["output"],
                    "module_args": {}, "connections": {"inputs": ids, "outputs": ["output"]}}
    layers["output"] = {"type": "output-tensor", "node_class": "TensorNode", "input_shapes": [[M, N]],
                        "output_shapes": [], "input_dtypes": ["torch.float32"], "output_dtypes": [],
                        "connections": {"inputs": ["op"], "outputs": []}}
    raw = tmp_path / "raw.yaml"
    raw.write_text(yaml.safe_dump({"model_name": "metadata_only", "layers": layers}, sort_keys=False))
    with pytest.raises(ValueError, match="cannot bind \\(bias, mat1, mat2\\) by position"):
        PyTorchToEinsum(enable_agent=False).convert(raw, tmp_path / "einsum", copy_graph=False)


# --- review P2-5: the split's bias_add must keep a rank-two bias in its expression ----------

def _flat_labels(parsed_output):
    """parse_einsum_equation's output side as a flat list of labels (single-output equations)."""
    labels = list(parsed_output)
    return labels if all(isinstance(x, str) and len(x) == 1 for x in labels) else list(labels[0])


def _addmm_bias_model(bias_shape):
    return textwrap.dedent(f'''
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            """addmm with a rank-two (possibly broadcast) bias passed as a runtime input."""
            def forward(self, bias, x, w):
                return torch.addmm(bias, x, w)

        def get_inputs():
            return [torch.zeros(*{list(bias_shape)}), torch.randn({M}, {K}), torch.randn({K}, {N})]

        def get_init_inputs():
            return []
    ''')


@pytest.mark.parametrize("bias", [[M, N], [1, N], [M, 1], [1, 1], [1]])
def test_split_bias_add_keeps_a_broadcast_bias_operand(tmp_path, bias):
    import math
    from solar.common.utils import parse_einsum_equation
    analysis = _analyze(tmp_path, "addmm_bias_" + "x".join(map(str, bias)), _addmm_bias_model(bias))
    total = analysis["total"]
    assert total["macs"] == M * K * N
    assert total["other_ops"] == M * N
    (add,) = [v for k, v in analysis["layers"].items() if k.endswith(".bias_add")]
    operands, outputs = parse_einsum_equation(add["einsum_equation"])   # lists of label lists
    assert len(operands) == 2 == len(add["tensor_shapes"]["inputs"]), add["einsum_equation"]
    out_labels, bias_labels = list(operands[0]), list(operands[1])
    assert len(bias_labels) == len(bias)                      # the bias keeps its own rank
    assert len(out_labels) == 2 and _flat_labels(outputs) == out_labels
    # a full-extent bias axis shares the output label; a singleton axis gets its own label
    for label, extent, out_label in zip(bias_labels, bias, out_labels[len(out_labels) - len(bias):]):
        assert (label == out_label) if extent != 1 else (label not in out_labels), add["einsum_equation"]
    assert add["memory_elements"]["inputs"][1] == math.prod(bias)   # traffic: actual bias storage
    # the exported graph's TACO expression names the bias operand (the analysis copy drops TACO)
    graph = yaml.safe_load((tmp_path / ("addmm_bias_" + "x".join(map(str, bias)) + "_einsum") / "einsum_graph_renamed.yaml").read_text())
    (exported,) = [v for k, v in graph["layers"].items() if k.endswith(".bias_add")]
    assert "In1" in exported["taco_expression"], exported["taco_expression"]


def test_nn_linear_bias_add_equation_is_unchanged(tmp_path):
    from solar.common.utils import parse_einsum_equation
    analysis = _analyze(tmp_path, "linear_model", LINEAR_MODEL)
    (add,) = [v for k, v in analysis["layers"].items() if k.endswith(".bias_add")]
    operands, outputs = parse_einsum_equation(add["einsum_equation"])
    assert len(operands) == 2
    out_labels, bias_labels = list(operands[0]), list(operands[1])
    # nn.Linear's [N] bias: rank 1, sharing the output's last label (AB,B->AB up to renaming)
    assert len(out_labels) == 2 and bias_labels == [out_labels[-1]] and _flat_labels(outputs) == out_labels
    assert add["tensor_shapes"]["inputs"] == [[M, N], [N]]
