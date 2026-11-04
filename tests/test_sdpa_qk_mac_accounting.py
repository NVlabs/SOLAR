# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDPA qk_matmul must be billed B*H*Q*K*D MACs for any batch/heads layout (issue #37, an earlier decision).

``_expand_sdpa`` emits ``Q @ K^T`` as a ``matmul`` layer but recorded the key
operand in its captured layout ``[..., K_len, D]``.  The analyzer bills a
``matmul`` by regenerating the einsum from the operand shapes and assumes the
second operand is ``[..., K, N]``, so it read ``N`` as ``D`` and undercounted
by ``K_len / D`` (4x on GPT-2: 256 / 64).  The expansion also hard-coded four
dimensions.  These tests run the real mechanical route on tiny synthetic
modules (CPU capture, no download) with ``S != D`` so the defect cannot hide:

* 4-D SDPA ``[B, H, S, D]``: total MACs = 2 * B*H*S*S*D (qk + av);
* 3-D SDPA ``[B, S, D]`` (no heads axis): total MACs = 2 * B*S*S*D;
* control: an explicit ``q @ k.transpose(-1, -2)`` (K arrives transposed in
  the graph) is already right and must stay right;
* control: a plain ``[M, K] @ [K, N]`` matmul is unchanged.

Since the lineage base the expansion lives in :func:`solar.einsum.expansions.expand_sdpa`
and the analyzer prices every real einsum from its equation; the key is
recorded in its captured ``[..., K, D]`` layout and contracted on ``D``.
The lower-rank K / V tests pin review P2-3 on that code: each
operand is labelled with its own rank (right-aligned against Q's leading
axes; a differing extent gets a fresh letter), so the equation rank equals
the recorded operand rank and the equation-priced MACs stay B*H*Q*K*D.
"""
import textwrap

import pytest

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.pytorch_processor import PyTorchProcessor

B, H, S, D = 2, 3, 16, 8


def _module(forward_body, input_shapes, extra_init=""):
    shapes = ", ".join(f"torch.randn{tuple(s)}" for s in input_shapes)
    args = ", ".join(f"t{i}" for i in range(len(input_shapes)))
    return textwrap.dedent(f'''
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                {extra_init}

            def forward(self, {args}):
                {forward_body}

        def get_inputs():
            return [{shapes}]

        def get_init_inputs():
            return []
    ''')


SDPA_4D = _module("return F.scaled_dot_product_attention(t0, t1, t2)", [(B, H, S, D)] * 3)
SDPA_3D = _module("return F.scaled_dot_product_attention(t0, t1, t2)", [(B, S, D)] * 3)
EXPLICIT_QKT = _module("return torch.matmul(t0, t1.transpose(-1, -2))", [(B, H, S, D)] * 2)
PLAIN_MATMUL = _module("return torch.matmul(t0, t1)", [(S, D), (D, S)])


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


def _qk_and_av(analysis):
    qk = [v for k, v in analysis["layers"].items() if k.endswith(".qk_matmul")]
    av = [v for k, v in analysis["layers"].items() if k.endswith(".av_matmul")]
    assert len(qk) == 1 and len(av) == 1, sorted(analysis["layers"])
    return qk[0], av[0]


def test_sdpa_4d_qk_matmul_bills_bhssd(tmp_path):
    analysis = _analyze(tmp_path, "sdpa4d", SDPA_4D)
    qk, av = _qk_and_av(analysis)
    assert qk["macs"] == B * H * S * S * D          # was B*H*S*D*D (N read as D)
    assert av["macs"] == B * H * S * S * D
    assert analysis["total"]["macs"] == 2 * B * H * S * S * D
    # the recorded operands and the equation must describe the same contraction
    qd, kd = qk["tensor_shapes"]["inputs"]
    assert qd[-1] == kd[-1], (qd, kd)              # K recorded as captured [..., K, D]: Q's last dim contracts with K's last dim
    assert qk["tensor_shapes"]["outputs"][0] == [B, H, S, S]


def test_sdpa_3d_without_heads_axis_is_expanded_and_billed(tmp_path):
    analysis = _analyze(tmp_path, "sdpa3d", SDPA_3D)
    qk, av = _qk_and_av(analysis)
    assert qk["macs"] == B * S * S * D
    assert av["macs"] == B * S * S * D
    assert analysis["total"]["macs"] == 2 * B * S * S * D


def test_explicit_q_times_k_transposed_control_unchanged(tmp_path):
    analysis = _analyze(tmp_path, "explicit_qkt", EXPLICIT_QKT)
    assert analysis["total"]["macs"] == B * H * S * S * D


def test_plain_matmul_control_unchanged(tmp_path):
    analysis = _analyze(tmp_path, "plain_matmul", PLAIN_MATMUL)
    assert analysis["total"]["macs"] == S * D * S
    assert analysis["total"]["fused_elements"] == S * D + D * S + S * S


# --- review P2-1 / P2-2: K keeps its own leading extents and storage; the module runs in MR CI -----------------------

def _expand(q, k, v, output=None):
    """Run the SDPA expansion on shape metadata only (no tensors)."""
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_node("attention")
    data = {"type": "scaled_dot_product_attention", "input_shapes": [q, k, v],
            "output_shapes": [output] if output is not None else []}
    layers, final, mapping = PyTorchToEinsum()._expand_sdpa("attention", data, graph, [], {})
    assert final == "attention.av_matmul"
    return layers


def _analyze_layers(layers, tmp_path):
    import copy
    import yaml
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(yaml.safe_dump({"layers": copy.deepcopy(layers)}, sort_keys=False))
    result = EinsumGraphAnalyzer().analyze_graph(graph_path, tmp_path / "analysis", precision="fp32", copy_graph=False)
    assert result is not None
    return result


@pytest.mark.parametrize("qb,kb", [(2, 1), (1, 2), (2, 2)], ids=["broadcast-key", "broadcast-query", "matched"])
def test_key_keeps_its_own_leading_dims_and_storage(qb, kb, tmp_path):
    import math
    q, k, v = [qb, 3, 5, 7], [kb, 3, 11, 7], [kb, 3, 11, 13]
    layers = _expand(q, k, v, [2, 3, 5, 13])
    qk = layers["attention.qk_matmul"]
    # K is recorded in its captured layout; the leading (batch/head) extents are K's own, never Q's
    assert qk["tensor_shapes"]["inputs"][1] == k
    assert math.prod(qk["tensor_shapes"]["inputs"][1]) == math.prod(k)
    result = _analyze_layers(layers, tmp_path)
    assert result["layers"]["attention.qk_matmul"]["memory_elements"]["inputs"][1] == math.prod(k)
    assert result["layers"]["attention.qk_matmul"]["memory_elements"]["inputs"][0] == math.prod(q)


def test_matched_batch_cost_is_unchanged_by_the_storage_fix(tmp_path):
    layers = _expand([2, 3, 5, 7], [2, 3, 11, 7], [2, 3, 11, 13])
    result = _analyze_layers(layers, tmp_path)
    assert result["layers"]["attention.qk_matmul"]["macs"] == 2 * 3 * 5 * 7 * 11
    assert result["layers"]["attention.av_matmul"]["macs"] == 2 * 3 * 5 * 11 * 13


def test_ci_unit_lane_selects_this_module():
    from pathlib import Path
    ci = (Path(__file__).resolve().parents[1] / ".gitlab-ci.yml").read_text()
    unit = ci[ci.index("unit-tests:"):ci.index("regression-tests:")]
    assert "tests/test_sdpa_qk_mac_accounting.py" in unit


# --- review P2-3: equation labels follow each operand's actual rank (lower-rank K / V) ---------

def _rank_consistent(layers):
    from solar.common.utils import validate_einsum_ranks_match_shapes
    for name, layer in layers.items():
        ok, msg = validate_einsum_ranks_match_shapes(layer["einsum_equation"], layer["tensor_shapes"])
        assert ok, (name, msg)


@pytest.mark.parametrize("key,expected_k_labels", [([11, 7], "KD"), ([3, 11, 7], "HKD"), ([1, 3, 11, 7], "UHKD")],
                         ids=["key-rank2", "key-rank3", "key-rank4-broadcast"])
def test_lower_rank_key_gets_right_aligned_labels(key, expected_k_labels, tmp_path):
    import math, re
    from solar.einsum.einsum_to_taco import EinsumToTaco
    q, v = [2, 3, 5, 7], [2, 3, 11, 13]
    layers = _expand(q, key, v, [2, 3, 5, 13])
    qk = layers["attention.qk_matmul"]
    assert qk["tensor_shapes"]["inputs"][1] == key                            # K keeps its own captured shape
    assert qk["einsum_equation"] == f"BHQD,{expected_k_labels}->BHQK"        # a size-1 broadcast axis gets a fresh letter
    assert qk["tensor_shapes"]["outputs"] == [[2, 3, 5, 11]]
    _rank_consistent(layers)                      # (the BFS renamer is a tombstone at the lineage base; renamed == raw)
    access = re.search(r"In1\[([^\]]+)\]", EinsumToTaco().convert(qk["einsum_equation"]))
    assert access and len(access.group(1).split(",")) == len(key)            # TACO indexes K with its own rank
    result = _analyze_layers(layers, tmp_path)
    assert result["layers"]["attention.qk_matmul"]["memory_elements"]["inputs"][1] == math.prod(key)
    assert result["layers"]["attention.qk_matmul"]["macs"] == 2 * 3 * 5 * 7 * 11
    assert result["layers"]["attention.av_matmul"]["macs"] == 2 * 3 * 5 * 11 * 13


def test_lower_rank_value_gets_right_aligned_labels(tmp_path):
    import math
    layers = _expand([2, 3, 5, 7], [2, 3, 11, 7], [11, 13], [2, 3, 5, 13])
    av = layers["attention.av_matmul"]
    assert av["einsum_equation"] == "BHQK,KV->BHQV"
    assert av["tensor_shapes"]["inputs"][1] == [11, 13]
    _rank_consistent(layers)
    result = _analyze_layers(layers, tmp_path)
    assert result["layers"]["attention.av_matmul"]["memory_elements"]["inputs"][1] == 11 * 13
    assert result["layers"]["attention.av_matmul"]["macs"] == 2 * 3 * 5 * 11 * 13


def test_full_rank_operands_keep_the_existing_equations():
    layers = _expand([2, 3, 16, 8], [2, 3, 16, 8], [2, 3, 16, 8])
    assert layers["attention.qk_matmul"]["einsum_equation"] == "BHQD,BHKD->BHQK"
    assert layers["attention.scale"]["einsum_equation"] == "BHQK->BHQK"
    assert layers["attention.softmax"]["einsum_equation"] == "BHQK->BHQK"
    assert layers["attention.av_matmul"]["einsum_equation"] == "BHQK,BHKV->BHQV"
    _rank_consistent(layers)
