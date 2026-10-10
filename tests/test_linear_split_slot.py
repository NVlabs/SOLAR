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

"""A linear-with-bias fed by slot k>0 of a split must name that slot.

``_split_linear_with_bias`` built its matmul input name as
``<producer>.Output`` regardless of which output slot of a multi-output
producer the activation came from.  Two linears reading the two halves of
one ``torch.split`` therefore both referenced ``Model.split.Output`` with
different shapes and the pre-AF shape check failed (L2/069 joint
transformer block: text/image streams split from one tensor).
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


SPLIT_INTO_TWO_LINEARS = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def forward(self, x, w_a, b_a, w_b, b_b):
            a, b = torch.split(x, [24, 8], dim=1)      # uneven split, two slots
            return F.linear(a, w_a, b_a).sum() + F.linear(b, w_b, b_b).sum()

    def get_inputs():
        return [torch.randn(1, 32, 16), torch.randn(12, 16), torch.randn(12),
                torch.randn(6, 16), torch.randn(6)]

    def get_init_inputs():
        return []
"""


def test_linear_reads_split_slot_one(tmp_path):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(SPLIT_INTO_TWO_LINEARS))
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
    linears = {k: v for k, v in layers.items()
               if v.get("type") in ("linear", "matmul") and "Model.split" in v["connections"]["inputs"]}
    assert len(linears) == 2, sorted(layers)
    slot_names = sorted(v["tensor_names"]["inputs"][0] for v in linears.values())
    assert slot_names == ["Model.split.Output", "Model.split.Output_1"], slot_names
