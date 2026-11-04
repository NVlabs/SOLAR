# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call attributes survive an in-place op on the lineage capture shim (R8 / INTEG-6).

A composed BigBird capture had 19,982 of 32,614 FunctionNodes without
``module_args.raw_attributes`` on the lineage base (0 with an earlier fix's code on the earlier baseline):
9,768 ``view`` + 9,768 ``__getitem__`` + bmm/unbind/softmax/einsum, all
downstream of the 48 in-place ``unsqueeze_`` / 36 ``add_`` nodes.  The lineage
shim's recorder did not carry torchview's ``collect_attributes`` flag onto the
result TensorNodes, and the processor's post-hoc wrapper re-read the flag from
the inputs AFTER the call — for an in-place op that is the op's own fresh
result node (flag False), which then propagated to every later call on that
tensor.  Both are fixed; this BigBird-shaped chain pins it.
"""
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

pytest.importorskip("torch")

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor

INPLACE_CHAIN = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(8)

        def forward(self, x, idx):
            y = self.ln(x)
            t = x.unsqueeze_(0)                 # in-place view: BigBird's unsqueeze_ pattern
            u = t.view(1, 2, 8)                 # view after the in-place op
            v = torch.bmm(u, x.reshape(1, 8, 2))
            w = v[idx]                          # __getitem__ after the in-place op
            w.add_(1.0)                         # in-place arithmetic (BigBird's add_)
            z = torch.softmax(w.view(-1), 0)
            return z, y

    def get_inputs():
        return [torch.randn(2, 8), torch.tensor([0, 0])]

    def get_init_inputs():
        return []
"""


def _capture(tmp_path: Path):
    (tmp_path / "model.py").write_text(dedent(INPLACE_CHAIN))
    (tmp_path / "graph").mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    return yaml.safe_load((tmp_path / "graph/pytorch_graph.yaml").read_text())["layers"]


def test_every_function_node_after_an_inplace_op_keeps_its_call_attributes(tmp_path: Path):
    layers = _capture(tmp_path)
    fn = {k: l for k, l in layers.items() if l.get("node_class") == "FunctionNode"}
    types = sorted(l["type"] for l in fn.values())
    assert "unsqueeze_" in types and "add_" in types and "bmm" in types and "__getitem__" in types, types
    missing = sorted((k, l["type"]) for k, l in fn.items() if not (l.get("module_args") or {}).get("raw_attributes"))
    assert missing == [], missing
    # the in-place op itself records the post-op operand shape and its argument
    unsq = next(l for l in fn.values() if l["type"] == "unsqueeze_")
    assert "Tensor(shape=" in unsq["module_args"]["raw_attributes"] and ", 0]" in unsq["module_args"]["raw_attributes"]
