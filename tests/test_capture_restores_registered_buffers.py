# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical capture must keep registered buffer VALUES across its to_empty() device loop.

``PyTorchProcessor._generate_torchview_graph`` moves the model with
``model.to_empty(device=...)`` before every torchview attempt (meta first, CPU
on fallback).  ``to_empty`` re-allocates parameters AND registered buffers
without values, so a forward that reads an index buffer (``token_type_ids``,
``position_ids`` in Hugging Face encoders) sees arbitrary integers on the CPU
fallback and fails with ``IndexError: index out of range in self`` (BigBird,
overnight KernelBench runnable smoke 2026-10-25).  These tests use a tiny
synthetic module (no Hugging Face model, no download): its forward forces the
meta attempt to fail the way the HF code does (``element in tensor`` needs
``.item()``) and then requires the buffer values to be intact on CPU.
"""
import textwrap

import torch
from torch import nn

from solar.common.types import ProcessingConfig
from solar.graph.pytorch_processor import PyTorchProcessor

SYNTHETIC = textwrap.dedent('''
    import torch
    import torch.nn as nn

    VOCAB, SEQ = 16, 8

    class Model(nn.Module):
        """Embedding lookup driven by two registered integer buffers, like an HF encoder."""
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 4)
            self.register_buffer("token_type_ids", torch.zeros(1, SEQ, dtype=torch.long))
            self.register_buffer("position_ids", torch.arange(SEQ).unsqueeze(0))

        def forward(self, x):
            n = x.shape[1]
            # `element in tensor` calls Tensor.item(): impossible on the meta device, exactly as
            # in the Hugging Face model, so capture must fall back to CPU.
            if 3 in self.position_ids:
                pass
            # Values must have survived the device move; a lost buffer is a wrong graph, not a
            # lucky lookup, so refuse loudly instead of relying on an out-of-range index.
            if not torch.equal(self.position_ids[0, :n], torch.arange(n, device=x.device)):
                raise RuntimeError("registered position_ids buffer lost its values")
            if int(self.token_type_ids.max()) != 0:
                raise RuntimeError("registered token_type_ids buffer lost its values")
            tt = self.token_type_ids[:, :n].expand(x.shape[0], n)
            pos = self.position_ids[:, :n].expand(x.shape[0], n)
            return self.emb(x) + self.emb(tt) + self.emb(pos)

    def get_inputs():
        return [torch.randint(0, VOCAB, (2, SEQ))]

    def get_init_inputs():
        return []
''')


def test_capture_survives_registered_integer_buffers(tmp_path):
    model_file = tmp_path / "buffer_model.py"
    model_file.write_text(SYNTHETIC)
    out = tmp_path / "graph"
    proc = PyTorchProcessor(ProcessingConfig(debug=False, force_rerun=True, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(out)) is True
    assert (out / "pytorch_graph.yaml").exists()


def test_buffer_snapshot_round_trips_through_to_empty():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("idx", torch.tensor([[5, 6, 7]]))
            self.register_buffer("flag", torch.tensor([True, False]))
            self.register_buffer("scale", torch.tensor([0.25]))

    m = M()
    saved = PyTorchProcessor._snapshot_buffers(m)
    assert set(saved) == {"idx", "flag", "scale"}
    m.to_empty(device="meta")           # values gone, as in the first capture attempt
    m.to_empty(device="cpu")            # CPU fallback: uninitialized memory
    PyTorchProcessor._restore_buffers(m, saved)
    assert torch.equal(m.idx, torch.tensor([[5, 6, 7]]))
    assert torch.equal(m.flag, torch.tensor([True, False]))
    assert torch.equal(m.scale, torch.tensor([0.25]))
    # Restoring onto a meta model is a no-op, never an error.
    m.to_empty(device="meta")
    PyTorchProcessor._restore_buffers(m, saved)
    assert m.idx.device.type == "meta"
