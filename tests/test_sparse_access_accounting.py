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

"""Sparse / untraced access patterns found on SOL-ExecBench.

* ``cache.to(fp32)[pages]``: the reference casts a whole paged KV cache
  before gathering a handful of pages. A kernel only touches the gathered
  pages, so a cast whose only consumers are gathers reads the gathered
  footprint, not the whole tensor (FlashInfer paged decode: 1.1 GB -> 44 KB).
* ``zeros_like(w)`` reads nothing from DRAM; it only borrows shape/dtype.
* A declared model output whose producer torchview did not trace (grads
  accumulated with ``grad[e] = ...``) is still written once, and reads of
  such an orphan accumulator are on-chip, not external input.
"""

from pathlib import Path
from textwrap import dedent

import yaml

from solar.analysis import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.common.utils import yaml_safe_load
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor


def _analyze(tmp_path: Path, src: str, precision: str = "fp32"):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    assert EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision=precision) is not None
    return (yaml_safe_load((einsum_dir / "einsum_graph_renamed.yaml").read_text()),
            yaml.safe_load((analysis_dir / "analysis.yaml").read_text()))


PAGED_GATHER = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, cache, pages):
            # cache: fp16 [4096, 64]; pages: int64 [8]
            c = cache.to(torch.float32)      # whole-cache cast in the reference
            return c[pages].sum(dim=0)       # but only 8 rows are ever used

    def get_inputs():
        return [torch.randn(4096, 64, dtype=torch.float16), torch.randint(0, 4096, (8,))]

    def get_init_inputs():
        return []
"""


def test_cast_feeding_only_gathers_reads_gathered_footprint(tmp_path):
    _, a = _analyze(tmp_path, PAGED_GATHER)
    total = a["total"]
    to = [l for l in a["layers"].values() if l["type"] == "to"][0]
    # The cast reads 8 rows of fp16 (gathered footprint), not 4096.
    assert to["fused_elements"] == 8 * 64, to
    # (the slice rule charges the gathered rows; the 8 int64 page ids are not counted)
    assert total["external_input_bytes"] == 8 * 64 * 2, total
    assert total["external_input_bytes"] < 4096 * 64 * 2 / 10


CREATION = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, w, x):
            # w: fp32 [512, 512] only lends its shape; x: fp32 [512, 512]
            return torch.zeros_like(w) + x

    def get_inputs():
        return [torch.randn(512, 512), torch.randn(512, 512)]

    def get_init_inputs():
        return []
"""


def test_creation_op_reads_nothing(tmp_path):
    _, a = _analyze(tmp_path, CREATION)
    n = 512 * 512
    zl = [l for l in a["layers"].values() if l["type"] == "zeros_like"][0]
    assert zl["fused_elements"] == 0, zl
    assert zl["memory_access"]["inputs"] == [0] if "memory_access" in zl else True
    total = a["total"]
    # External traffic: read x, write the sum. w is never read.
    assert total["external_input_bytes"] == n * 4, total
    assert total["external_output_bytes"] == n * 4, total


UNTRACED_ACCUMULATOR = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, w, x):
            # w: fp32 [8, 64, 32]; x: fp32 [64, 32]
            grad = torch.zeros_like(w)
            for e in range(8):
                grad[e] = x * float(e + 1)     # in-place writes torchview does not trace
            return grad

    def get_inputs():
        return [torch.randn(8, 64, 32), torch.randn(64, 32)]

    def get_init_inputs():
        return []
"""


def test_declared_output_with_untraced_producer_is_written_once(tmp_path):
    """torchview drops the returned accumulator entirely here; SOL-ExecBench
    traces keep an ``output-tensor`` node fed by a producer-less tensor
    (the accumulator after its in-place updates). Re-create that shape of
    graph by adding the two nodes torchview recorded on the real problems."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(UNTRACED_ACCUMULATOR))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    processor = PyTorchProcessor(
        ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(model_file), str(graph_dir))
    graph_path = graph_dir / "pytorch_graph.yaml"
    g = yaml_safe_load(graph_path.read_text())
    assert not any(l["type"] == "output-tensor" for l in g["layers"].values())
    common = {"node_class": "TensorNode", "input_types": ["input"], "output_types": ["output"],
              "module_args": {}}
    g["layers"]["Model.hidden-tensor_acc"] = dict(
        common, type="hidden-tensor", input_shapes=[[8, 64, 32]], output_shapes=[[8, 64, 32]],
        input_dtypes=["torch.float32"], output_dtypes=["torch.float32"],
        connections={"inputs": [], "outputs": ["Model.output-tensor"]})
    g["layers"]["Model.output-tensor"] = dict(
        common, type="output-tensor", input_shapes=[[8, 64, 32]], output_shapes=[],
        input_dtypes=["torch.float32"], output_dtypes=[],
        connections={"inputs": ["Model.hidden-tensor_acc"], "outputs": []})
    graph_path.write_text(yaml.safe_dump(g, sort_keys=False))

    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_path), str(einsum_dir)) is not None
    eg = yaml_safe_load((einsum_dir / "einsum_graph_renamed.yaml").read_text())
    assert eg["model_outputs"] == [{"op": None, "tensor": "Model.hidden-tensor_acc.Output",
                                    "shape": [8, 64, 32], "dtype": "torch.float32"}]
    analysis_dir = tmp_path / "analysis"; analysis_dir.mkdir()
    assert EinsumGraphAnalyzer().analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir), precision="fp32") is not None
    total = yaml.safe_load((analysis_dir / "analysis.yaml").read_text())["total"]
    # The accumulator is written once at its declared size; the eight
    # per-expert products are on-chip values (not eight extra outputs);
    # w only lends its shape and is never read.
    assert total["external_output_bytes"] == 8 * 64 * 32 * 4, total
    assert total["external_input_bytes"] == 64 * 32 * 4, total
