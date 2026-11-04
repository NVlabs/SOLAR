# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured grouped/depthwise/dense Conv1d/2d/3d through the whole graph path (grouped-conv-integration-1).

The reviewer's 12 captured controls (independent review of the grouped-conv fix,
test_integration.py): trace a one-layer nn.Conv{1,2,3}d on the meta device,
convert, analyze, and require total.macs == N * C_out * prod(out_spatial) *
(C_in / groups) * prod(kernel).  The equation-first dispatch of the lineage rewrite
bypassed every EinsumOp cost override (ConvTranspose, and an earlier fix's grouped
ConvEinsumOp); the handler-precedence repair covers every override.

Grouped 1D/2D layers are converted to the EXPANDED representation
([B, G, I, ...] x [G, O_pg, I, ...], equation BGI(P+R)...,GOI...->BGO...),
whose rank product is already exact; grouped 3D layers keep the dense
equation and need the handler's grouped override, which the lineage base does not
carry (it is an earlier fix's ConvEinsumOp): those three cases are strict xfails here
until an earlier change lands, and must pass on the composition.  The expanded 1D/2D
representations are pinned so a handler override cannot silently misread
them (the grouped-conv fix's conv_macs needs a rank guard for them).
"""
import math
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig, TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.base import EinsumOp
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor

CIN = 6
CASES = [(d, g, m) for d in (1, 2, 3) for g, m in [(1, 1), (2, 1), (6, 1), (6, 2)]]


def _handler_has_grouped_cost_override() -> bool:
    """True once the forward-conv handlers return an op whose cost is overridden (an earlier fix's ConvEinsumOp)."""
    ts = TensorShapes(inputs=[[2, CIN, 9, 8, 7], [CIN, CIN // 2, 2, 2, 2]], outputs=[[2, CIN, 8, 7, 6]])
    op = EinsumAnalyzer().get_einsum_op("conv3d", ts)
    return type(op).get_compute_cost is not EinsumOp.get_compute_cost


def _shapes(dims: int, groups: int, multiplier: int):
    cout = CIN * multiplier
    ish = [2, CIN] + [9, 8, 7][:dims]
    osh = [2, cout] + [s - 1 for s in ish[2:]]
    wsh = [cout, CIN // groups] + [2] * dims
    return ish, wsh, osh


def _expected(dims: int, groups: int, multiplier: int) -> int:
    ish, wsh, osh = _shapes(dims, groups, multiplier)
    return math.prod(osh) * (CIN // groups) * (2 ** dims)


def _run(tmp_path: Path, dims: int, groups: int, multiplier: int):
    ish, wsh, osh = _shapes(dims, groups, multiplier)
    cout = CIN * multiplier
    source = f'''
        import torch
        from torch import nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv{dims}d({CIN}, {cout}, 2, groups={groups}, bias=False)
            def forward(self, x):
                return self.conv(x)
        def get_inputs():
            return [torch.empty({tuple(ish)}, device="meta")]
        def get_init_inputs():
            return []
    '''
    (tmp_path / "model.py").write_text(dedent(source))
    for n in ("graph", "einsum", "analysis"):
        (tmp_path / n).mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
    analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"),
                                                   precision="fp32", copy_graph=False)
    assert analysis is not None
    einsum = yaml.safe_load((tmp_path / "einsum/einsum_graph.yaml").read_text())
    conv_layers = [(n, l) for n, l in einsum["layers"].items() if l.get("is_real_einsum")]
    assert len(conv_layers) == 1, conv_layers
    return analysis, conv_layers[0][1]


def _is_expanded_grouped(dims: int, groups: int, multiplier: int) -> bool:
    cout = CIN * multiplier
    return dims < 3 and groups > 1 and not (groups == CIN and groups == cout)


@pytest.mark.parametrize("dims,groups,multiplier", CASES)
def test_captured_conv_keeps_exact_macs_through_the_graph_path(tmp_path: Path, dims, groups, multiplier):
    if dims == 3 and groups > 1 and not _handler_has_grouped_cost_override():
        pytest.xfail("grouped Conv3d keeps the dense equation; the exact count needs an earlier fix's ConvEinsumOp override (not on this base)")
    analysis, layer = _run(tmp_path, dims, groups, multiplier)
    expected = _expected(dims, groups, multiplier)
    assert analysis["total"]["macs"] == expected, (expected, analysis["total"]["macs"], layer["einsum_equation"], layer["tensor_shapes"])


@pytest.mark.parametrize("dims,groups,multiplier", [c for c in CASES if _is_expanded_grouped(*c)])
def test_grouped_1d_2d_keep_the_expanded_representation(tmp_path: Path, dims, groups, multiplier):
    """The converter's expanded grouped form is pinned: [B, G, I, ...] x [G, O_pg, I, ...], equation BGI...,GOI...->BGO..."""
    _, layer = _run(tmp_path, dims, groups, multiplier)
    eq = layer["einsum_equation"]
    assert eq.startswith("BGI") and ",GOI" in eq and "->BGO" in eq, eq
    inputs = layer["tensor_shapes"]["inputs"]
    assert len(inputs[0]) == dims + 3 and inputs[0][1] == groups and inputs[0][2] == CIN // groups, inputs
    assert len(inputs[1]) == dims + 3 and inputs[1][0] == groups and inputs[1][2] == CIN // groups, inputs
    # the expanded equation's own rank product is the exact grouped count
    ts = TensorShapes(inputs=inputs, outputs=layer["tensor_shapes"]["outputs"])
    from solar.einsum.ops.base import compute_cost_from_equation
    assert compute_cost_from_equation(eq, ts) == _expected(dims, groups, multiplier)


def test_grouped_conv3d_direct_handler_state_is_recorded():
    """Documents the base: the dense conv3d equation over-counts grouped layers until an earlier fix's override."""
    ish, wsh, osh = _shapes(3, 2, 1)
    cost = EinsumAnalyzer().get_compute_cost("conv3d", TensorShapes(inputs=[ish, wsh], outputs=[osh]))
    if _handler_has_grouped_cost_override():
        assert cost == _expected(3, 2, 1)
    else:
        assert cost == math.prod(osh) * CIN * 8  # dense product: groups x too many
