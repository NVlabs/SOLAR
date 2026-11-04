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

"""Functional convs must honour ``groups`` (and stride/padding) from raw_attributes.

``F.conv1d(x, w, b, groups=C)`` has no nn.Module, so torchview records the
call kwargs only in ``raw_attributes`` (``{groups: 2048}``).  The conv
handlers read ``module_args["groups"]``, which exists only for nn.Conv*
modules, and fell back to ``groups=1``: a depthwise conv was priced as a
dense conv, inflating MACs by the channel count (SOL-ExecBench L1/005
causal conv 2x, L1/029 and L2/058 Mamba scans 4-5x).
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.ops.base import conv_call_kwargs
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


DEPTHWISE_CONV1D = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def forward(self, x, w, b):
            # x [B, C, L]; depthwise causal conv, kernel 4
            return F.conv1d(F.pad(x, (3, 0)), w, b, groups=x.shape[1])

    def get_inputs():
        return [torch.randn(2, 64, 128), torch.randn(64, 1, 4), torch.randn(64)]

    def get_init_inputs():
        return []
"""


def test_conv_call_kwargs_parses_raw_attributes():
    raw = ("[[Tensor(shape=(2, 2048, 4099), dtype=torch.bfloat16), "
           "Tensor(shape=(2048, 1, 4), dtype=torch.bfloat16), "
           "Tensor(shape=(2048,), dtype=torch.bfloat16)], {groups: 2048, padding: 1, stride: (2, 2)}]")
    kw = conv_call_kwargs({"raw_attributes": raw})
    assert kw["groups"] == 2048
    assert kw["padding"] == (1,)
    assert kw["stride"] == (2, 2)
    # positional form: conv1d(input, weight, bias, stride, padding, dilation, groups)
    raw_pos = ("[[Tensor(shape=(2, 64, 131), dtype=torch.float32), Tensor(shape=(64, 1, 4), dtype=torch.float32), "
               "Tensor(shape=(64,), dtype=torch.float32), 1, 0, 1, 64], {}]")
    kw = conv_call_kwargs({"raw_attributes": raw_pos})
    assert kw["groups"] == 64 and kw["stride"] == (1,) and kw["padding"] == (0,)
    # module args win when present
    assert conv_call_kwargs({"groups": 4, "raw_attributes": raw})["groups"] == 4


def test_depthwise_functional_conv1d_is_not_priced_as_dense(tmp_path):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(DEPTHWISE_CONV1D))
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    with open(einsum_dir / "einsum_graph.yaml") as f:
        layers = yaml.safe_load(f)["layers"]
    conv = next(v for v in layers.values() if v.get("type") == "conv1d")
    # depthwise form: output channel O appears in the input operand, no dense C contraction
    assert conv["einsum_equation"] == "BO(P+R),OCR->BOP", conv["einsum_equation"]

    from solar.analysis import EinsumGraphAnalyzer
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir))
    with open(analysis_dir / "analysis.yaml") as f:
        total = yaml.safe_load(f)["total"]["macs"]
    # depthwise MACs = B * C * L_out * K = 2 * 64 * 128 * 4; dense would be 64x more
    assert total == 2 * 64 * 128 * 4, total
