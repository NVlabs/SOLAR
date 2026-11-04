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

"""Regression tests for transposed-convolution MAC accounting (issue #31, an earlier decision).

A transposed convolution scatters every input element into all
``(C_out / groups) * prod(kernel)`` taps, so its exact dense MAC count is

    MACs = N * C_in * prod(input_spatial) * (C_out / groups) * prod(kernel)
         = numel(input) * numel(weight) / C_in

independent of stride, padding, output_padding and dilation (those only
decide where the taps land).  Counting from the *output* shape instead,
``N * (C_out / groups) * prod(output_spatial) * C_in * prod(kernel)``, is
too large by ``prod(output_spatial) / prod(input_spatial)``.  That ratio
is ``prod(stride)`` only when the output is exactly ``stride`` times the
input (KernelBench L2/20: 8x); the ``1d-stride2-dilation2`` case below
(input 7, output 17) has ratio 17/7, and at stride 1 the ratio is above 1
whenever the output is longer than the input.
``torch.utils.flop_counter.conv_flop_count(transposed=True)`` uses the
same input-based formula (times two for FLOPs).

The sliding-window equations ``BC(P-R),CKR->BKP`` etc. are deliberately
kept unchanged: byte/traffic accounting, rank renaming and the TACO /
Timeloop exports are derived from the equation and the tensor shapes, and
those were already exact.
"""

import math
import sys
from pathlib import Path

import pytest

# Add solar to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.conv_ops import (
    Conv1dHandler,
    Conv2dHandler,
    Conv3dHandler,
    ConvTranspose1dHandler,
    ConvTranspose2dHandler,
    ConvTranspose3dHandler,
)

_HANDLERS = {
    1: (ConvTranspose1dHandler, "convtranspose1d", Conv1dHandler, "conv1d"),
    2: (ConvTranspose2dHandler, "convtranspose2d", Conv2dHandler, "conv2d"),
    3: (ConvTranspose3dHandler, "convtranspose3d", Conv3dHandler, "conv3d"),
}

_EQUATIONS = {
    1: ("BC(P-R),CKR->BKP", [["B", "C", "P-R"], ["C", "K", "R"], ["B", "K", "P"]]),
    2: ("BC(P-R)(Q-S),CKRS->BKPQ",
        [["B", "C", "P-R", "Q-S"], ["C", "K", "R", "S"], ["B", "K", "P", "Q"]]),
    3: ("BC(P-T)(Q-R)(U-S),CKTRS->BKPQU",
        [["B", "C", "P-T", "Q-R", "U-S"], ["C", "K", "T", "R", "S"],
         ["B", "K", "P", "Q", "U"]]),
}


def _out_len(l_in, k, stride, padding, output_padding, dilation):
    """torch.nn.ConvTransposeNd output length."""
    return (l_in - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1


def _case(dims, n, c_in, c_out, spatial, k, stride=1, padding=0,
          output_padding=0, dilation=1, groups=1, bias=False):
    """Build (TensorShapes, expected_macs, output_based_macs) for one module."""
    spatial = list(spatial)
    out_spatial = [_out_len(l, k, stride, padding, output_padding, dilation)
                   for l in spatial]
    input_shape = [n, c_in] + spatial
    weight_shape = [c_in, c_out // groups] + [k] * dims   # torch layout
    output_shape = [n, c_out] + out_spatial
    inputs = [input_shape, weight_shape]
    if bias:
        inputs.append([c_out])
    ts = TensorShapes(inputs=inputs, outputs=[output_shape])
    expected = n * c_in * math.prod(spatial) * (c_out // groups) * k ** dims
    # What the generic resolver produced before the fix: K is resolved from
    # weight[1] (= C_out / groups) and P/Q/U from the *output* spatial dims.
    output_based = n * (c_out // groups) * math.prod(out_spatial) * c_in * k ** dims
    return ts, expected, output_based


# (dims, kwargs) -- one row per synthetic module.  `red_on_base` records which
# rows the previous output-based formula got wrong (stride > 1, or boundary
# taps at stride 1 with padding 0); it is informational only.
_CASES = [
    pytest.param(1, dict(n=2, c_in=4, c_out=6, spatial=[9], k=3, stride=1, padding=1),
                 id="1d-stride1-same"),
    pytest.param(1, dict(n=2, c_in=4, c_out=6, spatial=[9], k=3, stride=1, padding=0),
                 id="1d-stride1-pad0-boundary"),
    pytest.param(1, dict(n=2, c_in=4, c_out=6, spatial=[9], k=3, stride=2, padding=1,
                         output_padding=1), id="1d-stride2"),
    pytest.param(1, dict(n=2, c_in=4, c_out=6, spatial=[9], k=3, stride=2, padding=1,
                         output_padding=1, groups=2), id="1d-stride2-groups2"),
    pytest.param(1, dict(n=1, c_in=4, c_out=4, spatial=[7], k=4, stride=2, padding=1,
                         dilation=2), id="1d-stride2-dilation2"),
    pytest.param(2, dict(n=2, c_in=4, c_out=8, spatial=[5, 7], k=3, stride=1, padding=1),
                 id="2d-stride1-same"),
    pytest.param(2, dict(n=2, c_in=4, c_out=8, spatial=[5, 7], k=3, stride=2, padding=1,
                         output_padding=1), id="2d-stride2"),
    pytest.param(2, dict(n=2, c_in=4, c_out=8, spatial=[5, 7], k=3, stride=2, padding=1,
                         output_padding=1, groups=2), id="2d-stride2-groups2"),
    pytest.param(2, dict(n=1, c_in=6, c_out=6, spatial=[6, 6], k=2, stride=2),
                 id="2d-stride2-k2-upsample"),
    pytest.param(3, dict(n=2, c_in=4, c_out=8, spatial=[3, 4, 5], k=3, stride=1, padding=1),
                 id="3d-stride1-same"),
    pytest.param(3, dict(n=2, c_in=4, c_out=8, spatial=[3, 4, 5], k=3, stride=2, padding=1,
                         output_padding=1), id="3d-stride2"),
    pytest.param(3, dict(n=2, c_in=4, c_out=8, spatial=[3, 4, 5], k=3, stride=2, padding=1,
                         output_padding=1, groups=2), id="3d-stride2-groups2"),
    pytest.param(3, dict(n=1, c_in=4, c_out=8, spatial=[3, 4, 5], k=3, stride=2, padding=1,
                         output_padding=1, bias=True), id="3d-stride2-bias-input"),
]


class TestConvTransposeExactMacs:
    """Handler-level and analyzer-level MAC counts equal the exact dense count."""

    @pytest.mark.parametrize("dims,kw", _CASES)
    def test_handler_compute_cost(self, dims, kw):
        handler_cls, op_name, _, _ = _HANDLERS[dims]
        ts, expected, _ = _case(dims, **kw)
        op = handler_cls().generate_einsum(op_name, ts)
        assert op.get_compute_cost(ts) == expected

    @pytest.mark.parametrize("dims,kw", _CASES)
    def test_analyzer_compute_cost(self, dims, kw):
        """The graph analyzer path: EinsumAnalyzer.get_compute_cost(op_type, ts)."""
        _, op_name, _, _ = _HANDLERS[dims]
        ts, expected, _ = _case(dims, **kw)
        assert EinsumAnalyzer().get_compute_cost(op_name, ts) == expected

    @pytest.mark.parametrize("dims,kw", _CASES)
    def test_equation_and_operands_unchanged(self, dims, kw):
        """Equation/operands drive traffic, renaming and exports: must not move."""
        handler_cls, op_name, _, _ = _HANDLERS[dims]
        ts, _, _ = _case(dims, **kw)
        op = handler_cls().generate_einsum(op_name, ts)
        equation, dims_list = _EQUATIONS[dims]
        assert op.equation == equation
        assert [o.dims for o in op.operands] == dims_list
        assert [o.is_output for o in op.operands] == [False, False, True]
        assert op.name == op_name
        assert op.is_real_einsum and op.elementwise_op == "mul" and op.reduction_op == "add"

    @pytest.mark.parametrize("dims,kw", [c for c in _CASES if c.values[1].get("groups", 1) == 1])
    def test_identity_with_forward_conv_of_transposed_shapes(self, dims, kw):
        """ConvTranspose MACs == MACs of the forward conv whose output is the
        ConvTranspose input and whose input has the ConvTranspose output's
        channels (transposed-conv == data-gradient of that conv)."""
        t_cls, t_name, c_cls, c_name = _HANDLERS[dims]
        ts, expected, _ = _case(dims, **kw)
        input_shape, weight_shape = ts.inputs[0], ts.inputs[1]
        output_shape = ts.outputs[0]
        # forward conv: input [N, C_out, *out_spatial], weight [C_in, C_out, *k]
        # (torch conv layout [O, I, *k] with O = C_in_T, I = C_out_T),
        # output [N, C_in, *in_spatial]
        conv_ts = TensorShapes(
            inputs=[[output_shape[0], output_shape[1]] + list(output_shape[2:]),
                    list(weight_shape)],
            outputs=[list(input_shape)],
        )
        conv_cost = c_cls().generate_einsum(c_name, conv_ts).get_compute_cost(conv_ts)
        assert conv_cost == expected
        assert t_cls().generate_einsum(t_name, ts).get_compute_cost(ts) == expected

    def test_general_ratio_is_output_over_input_spatial(self):
        """old / new = prod(output_spatial) / prod(input_spatial); 17/7 here, not 2."""
        ts, expected, output_based = _case(
            1, n=1, c_in=4, c_out=4, spatial=[7], k=4, stride=2, padding=1, dilation=2)
        assert ts.outputs[0][2] == 17
        assert (output_based, expected) == (1_088, 448)
        assert output_based * 7 == expected * 17
        assert ConvTranspose1dHandler().generate_einsum("convtranspose1d", ts).get_compute_cost(ts) == expected

    def test_stride_ratio_is_stride_product(self):
        """Special case (L2/20 configuration, output = stride * input): the old
        count was exactly prod(stride) = 8x in 3D."""
        ts, expected, output_based = _case(
            3, n=2, c_in=4, c_out=8, spatial=[4, 4, 4], k=3, stride=2, padding=1,
            output_padding=1)
        assert output_based == 8 * expected
        assert ConvTranspose3dHandler().generate_einsum("convtranspose3d", ts).get_compute_cost(ts) == expected


class TestKernelBenchLevel2Case20:
    """KernelBench level2/20: ConvTranspose3d(32, 64, 3, stride=2, padding=1,
    output_padding=1), input [16, 32, 16, 32, 32] -> output [16, 64, 32, 64, 64]."""

    EXPECTED = 14_495_514_624
    OLD_OUTPUT_BASED = 115_964_116_992

    def _ts(self):
        # Same positional shapes as einsum_graph.yaml: input, weight, bias.
        return TensorShapes(
            inputs=[[16, 32, 16, 32, 32], [32, 64, 3, 3, 3], [64]],
            outputs=[[16, 64, 32, 64, 64]],
        )

    def test_expected_formula(self):
        assert self.EXPECTED == 16 * 32 * 16 * 32 * 32 * 64 * 27
        assert self.OLD_OUTPUT_BASED == 16 * 64 * 32 * 64 * 64 * 32 * 27
        assert self.OLD_OUTPUT_BASED == 8 * self.EXPECTED

    def test_handler(self):
        ts = self._ts()
        op = ConvTranspose3dHandler().generate_einsum("convtranspose3d", ts)
        assert op.get_compute_cost(ts) == self.EXPECTED

    def test_analyzer_graph_path(self):
        ts = self._ts()
        assert EinsumAnalyzer().get_compute_cost("convtranspose3d", ts) == self.EXPECTED
        assert EinsumAnalyzer().get_compute_cost("conv_transpose3d", ts) == self.EXPECTED


class TestTorchFlopCounterOracle:
    """torch's own transposed-conv formula (input-based) agrees, on tiny CPU shapes."""

    @pytest.mark.parametrize("dims,kw", _CASES)
    def test_matches_torch_flop_counter(self, dims, kw):
        torch = pytest.importorskip("torch")
        from torch.utils.flop_counter import FlopCounterMode

        nn_cls = {1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d,
                  3: torch.nn.ConvTranspose3d}[dims]
        mod = nn_cls(kw["c_in"], kw["c_out"], kw["k"], stride=kw.get("stride", 1),
                     padding=kw.get("padding", 0),
                     output_padding=kw.get("output_padding", 0),
                     dilation=kw.get("dilation", 1), groups=kw.get("groups", 1),
                     bias=kw.get("bias", False))
        x = torch.zeros([kw["n"], kw["c_in"]] + list(kw["spatial"]))
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            y = mod(x)
        ts, expected, _ = _case(dims, **kw)
        assert list(y.shape) == ts.outputs[0]
        assert list(mod.weight.shape) == ts.inputs[1]
        assert fc.get_total_flops() == 2 * expected
        handler_cls, op_name, _, _ = _HANDLERS[dims]
        assert handler_cls().generate_einsum(op_name, ts).get_compute_cost(ts) == expected
