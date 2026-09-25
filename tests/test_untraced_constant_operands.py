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

"""Weights created inside ``forward`` (untraced constants) must still be modelled.

``F.conv2d(x, torch.tensor([...]))`` with a kernel built in ``forward`` has
no tensor node for the kernel: torchview records only the activation edge
and the kernel survives solely in ``raw_attributes``.  The conv handler then
saw one input, produced an empty einsum, and the AF builder dropped the
layer — its consumer read an orphan tensor (L2/071 Sobel edge filter).

The torchview-quirk repair now synthesizes an ``auxiliary-tensor`` source
for such operands so they are read from DRAM like any other model input.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


CONST_KERNEL_CONV = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def forward(self, mask):
            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(2, 3).contiguous()
            ex = F.conv2d(mask, sobel_x, padding=1)
            ey = F.conv2d(mask, sobel_y, padding=1)
            return torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)

    def get_inputs():
        return [torch.randn(1, 1, 64, 64)]

    def get_init_inputs():
        return []
"""


def _run(tmp_path: Path, src: str):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
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
    with open(einsum_dir / "af_einsum_graph.yaml") as f:
        af = yaml.safe_load(f)
    return layers, af


def test_conv_with_untraced_constant_kernel_is_modelled(tmp_path):
    layers, af = _run(tmp_path, CONST_KERNEL_CONV)
    convs = {k: v for k, v in layers.items() if v.get("type") == "conv2d"}
    assert convs, "no conv2d layers converted"
    for name, L in convs.items():
        assert L["einsum_equation"], f"{name} has an empty einsum equation"
        assert len(L["tensor_shapes"]["inputs"]) >= 2, f"{name} lost its kernel operand"
        assert [1, 1, 3, 3] in L["tensor_shapes"]["inputs"], L["tensor_shapes"]
    # The AF graph passed its own invariant checks (no orphan reads) and the
    # conv einsums are present.
    names = {e["name"] for e in af["workload"]["einsums"]}
    assert any(n.startswith("Model_conv2d") for n in names)
