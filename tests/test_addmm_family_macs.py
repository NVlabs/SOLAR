# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""addmv / baddbmm / addbmm / mv / dot must be billed as MACs (GitHub issue #13).

These ops had no einsum handler: they were refused as unrecognised (0 MACs,
``status: incomplete``) or, before the fail-closed route, billed as a copy.
Real models hit them (Bloom's ``alibi.baddbmm(...)`` attention scores).  They
are now billed like ``addmm``: the contraction as MACs, the bias add of the
bias-carrying ops as one elementwise op per output (other_ops), traffic at the
hand minimum.  Mechanical-route tests run on tiny synthetic CPU modules.
"""
import textwrap

import pytest

from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer

from test_addmm_mac_accounting import _analyze

# (name, forward args, expression, inputs, macs, other_ops, fused_elements)
CASES = {
    "addmm": ("bias, a, b", "torch.addmm(bias, a, b)",
              "[torch.zeros(6), torch.randn(4, 5), torch.randn(5, 6)]", 120, 24, 6 + 20 + 30 + 24),
    "baddbmm": ("bias, a, b", "torch.baddbmm(bias, a, b)",
                "[torch.zeros(3, 4, 6), torch.randn(3, 4, 5), torch.randn(3, 5, 6)]",
                360, 72, 72 + 60 + 90 + 72),
    "baddbmm_bcast_bias": ("bias, a, b", "torch.baddbmm(bias, a, b)",
                           "[torch.zeros(1, 4, 6), torch.randn(3, 4, 5), torch.randn(3, 5, 6)]",
                           360, 72, 24 + 60 + 90 + 72),
    "addbmm": ("bias, a, b", "torch.addbmm(bias, a, b)",
               "[torch.zeros(4, 6), torch.randn(3, 4, 5), torch.randn(3, 5, 6)]",
               360, 24, 24 + 60 + 90 + 24),
    "addmv": ("bias, a, b", "torch.addmv(bias, a, b)",
              "[torch.zeros(4), torch.randn(4, 5), torch.randn(5)]", 20, 4, 4 + 20 + 5 + 4),
    "mv": ("a, b", "torch.mv(a, b)", "[torch.randn(4, 5), torch.randn(5)]", 20, 0, 20 + 5 + 4),
    "dot": ("a, b", "torch.dot(a, b)", "[torch.randn(7), torch.randn(7)]", 7, 0, 7 + 7 + 1),
}


def _source(args, expr, inputs):
    return textwrap.dedent(f'''
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, {args}):
                return {expr}

        def get_inputs():
            return {inputs}

        def get_init_inputs():
            return []
    ''')


@pytest.mark.parametrize("case", list(CASES))
def test_pipeline_bills_macs_and_bias_add(tmp_path, case):
    args, expr, inputs, macs, other_ops, fused = CASES[case]
    analysis = _analyze(tmp_path, f"m_{case}", _source(args, expr, inputs))
    total = analysis["total"]
    assert total["macs"] == macs
    assert total["other_ops"] == other_ops
    assert total["fused_elements"] == fused
    # no longer refused as an unrecognised op
    assert total.get("status") != "incomplete"
    assert all(not layer.get("unrecognised_op") for layer in analysis["layers"].values())


def test_gpt2_style_conv1d_layer_is_49152_macs(tmp_path):
    # Hugging Face Conv1D: y = addmm(bias, x.view(-1, nx), weight[nx, nf]); 16 * 32 * 96
    src = textwrap.dedent('''
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(32, 96))
                self.bias = nn.Parameter(torch.zeros(96))

            def forward(self, x):
                return torch.addmm(self.bias, x.view(-1, 32), self.weight)

        def get_inputs():
            return [torch.randn(16, 32)]

        def get_init_inputs():
            return []
    ''')
    total = _analyze(tmp_path, "m_conv1d", src)["total"]
    assert total["macs"] == 16 * 32 * 96 == 49_152
    assert total["other_ops"] == 16 * 96


@pytest.mark.parametrize("name", ["mv", "mv.default", "aten.mv.default", "dot", "aten.dot.default"])
def test_overload_spellings_reach_the_handler(name):
    analyzer = EinsumAnalyzer()
    shapes = [[4, 5], [5]] if "mv" in name else [[7], [7]]
    ts = TensorShapes(inputs=shapes, outputs=[[4]] if "mv" in name else [[]])
    op = analyzer.get_einsum_op(name, ts)
    assert op.is_real_einsum
    assert int(analyzer.get_compute_cost(name, ts)) == (20 if "mv" in name else 7)


@pytest.mark.parametrize("name,inputs,macs", [
    ("addmv", [[4], [4, 5], [5]], 20),
    ("addmv", [[1], [4, 5], [5]], 20),
    ("baddbmm", [[3, 4, 6], [3, 4, 5], [3, 5, 6]], 360),
    ("baddbmm.default", [[4, 6], [3, 4, 5], [3, 5, 6]], 360),
    ("baddbmm", [[1, 1, 6], [3, 4, 5], [3, 5, 6]], 360),
    ("baddbmm", [[3, 4, 1], [3, 4, 5], [3, 5, 6]], 360),
    ("aten.addbmm.default", [[4, 6], [3, 4, 5], [3, 5, 6]], 360),
    ("addbmm", [[6], [3, 4, 5], [3, 5, 6]], 360),
    ("addbmm", [[1, 1], [3, 4, 5], [3, 5, 6]], 360),
    # the contraction alone (what the converter's split layer carries)
    ("addbmm", [[3, 4, 5], [3, 5, 6]], 360),
])
def test_handler_macs_are_the_contraction_ranks_even_with_broadcast_bias(name, inputs, macs):
    analyzer = EinsumAnalyzer()
    ts = TensorShapes(inputs=inputs, outputs=[[1]])
    assert int(analyzer.get_compute_cost(name, ts)) == macs


@pytest.mark.parametrize("name,inputs", [
    ("mv", [[4, 5], [6]]),               # inner mismatch
    ("mv", [[5], [5]]),                  # wrong rank
    ("dot", [[2, 3], [3]]),              # wrong rank
    ("dot", [[3], [4]]),                 # extent mismatch
    ("addmv", [[4], [4, 5], [6]]),
    ("addmv", [[3], [4, 5], [5]]),       # bias does not broadcast
    ("addmv", [[4], [4, 5]]),            # one operand missing -> not a vector
    ("baddbmm", [[3, 4, 6], [3, 4, 5], [2, 5, 6]]),   # batch mismatch
    ("baddbmm", [[3, 4, 7], [3, 4, 5], [3, 5, 6]]),   # bias does not broadcast
    ("baddbmm", [[2, 4, 6], [3, 4, 5], [3, 5, 6]]),
    ("baddbmm", [[4, 5], [5, 6]]),       # 2D operands
    ("addbmm", [[5, 6], [3, 4, 5], [3, 5, 6]]),
    ("addbmm", [[3, 4, 6], [3, 4, 5], [3, 5, 6]]),    # bias rank exceeds output
    ("addbmm", []),
])
def test_handlers_refuse_operands_that_do_not_bind(name, inputs):
    with pytest.raises(Exception):
        EinsumAnalyzer().get_einsum_op(name, TensorShapes(inputs=inputs, outputs=[[1]]))


def test_fused_einsum_puts_the_bias_first():
    op = EinsumAnalyzer().get_einsum_op(
        "baddbmm", TensorShapes(inputs=[[1, 4, 6], [3, 4, 5], [3, 5, 6]], outputs=[[3, 4, 6]]))
    assert op.equation == "PMN,BMK,BKN->BMN"
    assert [o.name for o in op.operands] == ["Bias", "Input", "Weight", "Output"]
