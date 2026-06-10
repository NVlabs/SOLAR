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

"""Tests for CPU fallback preserving non-persistent buffers.

HuggingFace models register buffers like position_ids with persistent=False,
which excludes them from state_dict(). When meta device tracing fails and
we fall back to CPU, these buffers must still contain valid data. Regression
test for the IndexError in BigBird embedding lookups.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

try:
    import torchview
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False

from solar.common.types import ProcessingConfig
from solar.graph.pytorch_processor import PyTorchProcessor


class EmbeddingWithNonPersistentBuffer(nn.Module):
    """Model that mimics HuggingFace's position_ids pattern.

    Registers position_ids as a non-persistent buffer (excluded from
    state_dict) and uses it to index into an embedding table during forward.
    On meta device, the embedding lookup fails with IndexError because meta
    tensors have no data. The CPU fallback must preserve the buffer.
    """

    def __init__(self, vocab_size=32, embed_dim=16, max_position=64):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_position, embed_dim)
        position_ids = torch.arange(max_position).unsqueeze(0)
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos_ids = self.position_ids[:, :seq_len]
        return self.word_embedding(input_ids) + self.position_embedding(pos_ids)


# Minimal module-like object that _generate_torchview_graph and
# _create_model_instance use for CPU fallback re-creation.
Model = EmbeddingWithNonPersistentBuffer


class _FakeModule:
    Model = EmbeddingWithNonPersistentBuffer

    @staticmethod
    def get_init_inputs():
        return ()

    @staticmethod
    def get_inputs():
        return [torch.randint(0, 32, (2, 8))]


@pytest.mark.skipif(not TORCHVIEW_AVAILABLE, reason="torchview not installed")
class TestNonPersistentBufferFallback:

    def test_nonpersistent_buffer_excluded_from_state_dict(self):
        """Verify the buffer IS excluded from state_dict — confirms the bug premise."""
        model = EmbeddingWithNonPersistentBuffer()
        assert "position_ids" not in model.state_dict()
        assert "position_ids" in dict(model.named_buffers())

    def test_to_empty_destroys_nonpersistent_buffer(self):
        """Verify to_empty destroys the buffer — confirms why state_dict restore fails."""
        model = EmbeddingWithNonPersistentBuffer()
        original_ids = model.position_ids.clone()
        model.to_empty(device="cpu")
        assert model.position_ids.shape == original_ids.shape
        assert not torch.equal(model.position_ids, original_ids)

    def test_generate_graph_with_nonpersistent_buffer(self):
        """CPU fallback must produce a graph when model has non-persistent buffers.

        This is the core regression test. Without the re-creation fix,
        to_empty(device='meta') destroys position_ids, and the CPU
        fallback gets an IndexError on the embedding lookup.
        """
        config = ProcessingConfig(
            debug=True,
            save_graph=False,
            force_rerun=True,
        )
        processor = PyTorchProcessor(config)

        model = EmbeddingWithNonPersistentBuffer()
        input_ids = torch.randint(0, 32, (2, 8))
        fake_module = _FakeModule()

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = processor._generate_torchview_graph(
                model, [input_ids], output_dir=tmpdir, module=fake_module
            )
            assert graph is not None, (
                "Graph generation failed — CPU fallback likely could not "
                "re-create model with valid non-persistent buffers"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
