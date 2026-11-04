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

"""Producer-less tensors read by several consumers must share one rank tuple.

Tensors created inside ``forward`` by untraced ops (``torch.ones``,
``torch.arange``, dtype ``view`` ...) surface as torchview ``hidden-tensor``
placeholders with no producer layer.  The AF cross-layer union only unions
a consumer's input axes against its *producer's* output axes, so two
consumers of such a tensor (e.g. the ``[..., ::2]`` / ``[..., 1::2]``
strided slices used by FP4 packing code) ended up with independent ranks
for the same AF tensor name and the invariant check rejected the graph.
"""

from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


STRIDED_READS = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, y):
            packed = torch.ones(4, 8, 16, dtype=torch.float32)  # producer-less source
            a = packed[..., ::2] * x
            b = packed[..., 1::2] * y
            return a + b

    def get_inputs():
        return [torch.randn(4, 8, 8), torch.randn(4, 8, 8)]

    def get_init_inputs():
        return []
"""


def _run_pipeline(tmp_path: Path, src: str) -> dict:
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
    with open(einsum_dir / "af_einsum_graph.yaml") as f:
        return yaml.safe_load(f)


def test_two_strided_reads_of_producerless_tensor_share_ranks(tmp_path):
    af = _run_pipeline(tmp_path, STRIDED_READS)
    tuples = defaultdict(set)
    for e in af["workload"]["einsums"]:
        for ta in e["tensor_accesses"]:
            proj = ta["projection"]
            ranks = tuple(proj) if isinstance(proj, list) else tuple(proj.keys())
            tuples[ta["name"]].add(tuple(str(r).upper() for r in ranks))
    bad = {k: v for k, v in tuples.items() if len(v) > 1}
    assert not bad, f"tensors with more than one rank tuple: {bad}"
    # The producer-less source is read by both slices.
    src = [k for k in tuples if "hidden_tensor" in k]
    assert src, f"expected a producer-less hidden tensor read, got {sorted(tuples)}"
