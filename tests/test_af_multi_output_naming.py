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

"""AF tensor names for multi-output ops must not collide with sibling ops.

torchview names repeated ops ``Model.split``, ``Model.split_1``, ... and the
AF builder used to name output slot k of an op ``<op>_<k>``.  Slot 1 of
``Model.split`` therefore became ``Model_split_1`` — the same name as the
primary output of the sibling op ``Model.split_1`` — and the invariant
check rejected the graph ("inconsistent rank tuples").  Any model with two
or more multi-output ops of the same kind (split/chunk/unbind/topk/where)
hit this.
"""

from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from solar.common.types import ProcessingConfig
from solar.einsum.af_graph_builder import _af_pred_tensor_name, _af_slot_tensor_name
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


TWO_SPLITS = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, y):
            a, b = torch.split(x, [8, 24], dim=-1)   # Model.split   (2 slots)
            c, d = torch.split(y, [8, 24], dim=-1)   # Model.split_1 (2 slots)
            return torch.cat([b, d], dim=-1), a + c

    def get_inputs():
        return [torch.randn(4, 32), torch.randn(4, 32)]

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


def test_slot_name_cannot_collide_with_sibling_op():
    """Slot k of op X must never equal the sanitized name of any op X_<n>."""
    assert _af_slot_tensor_name("Model_split", 0) == "Model_split"
    assert _af_slot_tensor_name("Model_split", 1) != "Model_split_1"
    # Producer-side and consumer-side naming agree.
    assert _af_pred_tensor_name("Model.split", "Model.split.Output_1") == _af_slot_tensor_name("Model_split", 1)
    assert _af_pred_tensor_name("Model.split", "Model.split.Output") == "Model_split"


def test_two_sibling_splits_build_af_graph(tmp_path):
    af = _run_pipeline(tmp_path, TWO_SPLITS)
    einsums = af["workload"]["einsums"]

    # Every AF tensor is produced by at most one einsum.
    producers = defaultdict(list)
    for e in einsums:
        for ta in e["tensor_accesses"]:
            if ta.get("output"):
                producers[ta["name"]].append(e["name"])
    dups = {k: v for k, v in producers.items() if len(v) > 1}
    assert not dups, f"AF tensors produced by more than one einsum: {dups}"

    # Both slots of both splits are read by someone downstream.
    reads = {ta["name"] for e in einsums for ta in e["tensor_accesses"] if not ta.get("output")}
    for op in ("Model_split", "Model_split_1"):
        assert op in reads, f"{op} slot 0 never read"
        assert _af_slot_tensor_name(op, 1) in reads, f"{op} slot 1 never read"
