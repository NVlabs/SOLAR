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

"""Integration test: standalone BERT-like transformer model.

This test creates a standalone model file (like kernelbench style),
processes it with `PyTorchProcessor`, and then analyzes the produced graph with
`PyTorchToEinsum`.
"""

from pathlib import Path

from solar.analysis import EinsumGraphAnalyzer
from solar.einsum import PyTorchToEinsum
from solar.perf import EinsumGraphPerfModel
from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor


def test_standalone_bert_like_model_process_and_convert(tmp_path: Path) -> None:
    model_py = tmp_path / "bert_like.py"
    model_py.write_text(
        """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int = 32, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        # x: [B, S, H]
        b, s, h = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, D]
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, heads, S, S]
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # [B, heads, S, D]
        ctx = ctx.transpose(1, 2).contiguous().view(b, s, h)  # [B, S, H]
        return self.out_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 32, num_heads: int = 4, intermediate_size: int = 64):
        super().__init__()
        self.attn = SelfAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, hidden_size * 2) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids):
        # input_ids: [B, S]
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        cls = x[:, 0, :]
        return self.classifier(cls)


def get_inputs():
    torch.manual_seed(0)
    batch = 2
    seq = 8
    vocab_size = 100
    input_ids = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
    return [input_ids]
"""
    )

    graph_out = tmp_path / "graph_out"
    cfg = ProcessingConfig(save_graph=False, force_rerun=True, debug=False)
    processor = PyTorchProcessor(cfg)
    assert processor.process_model_file(str(model_py), str(graph_out)) is True

    graph_path = graph_out / "pytorch_graph.yaml"
    assert graph_path.exists()

    out_dir = tmp_path / "out"

    # 1) PyTorch graph -> einsum graph (also generates einsum_graph_renamed.yaml).
    converter = PyTorchToEinsum(debug=False, enable_agent=False)
    einsum_graph = converter.convert_graph(graph_path, out_dir)
    assert einsum_graph is not None
    assert (out_dir / "einsum_graph.yaml").exists()
    assert (out_dir / "einsum_graph_renamed.yaml").exists()

    # 2) Einsum graph -> hardware-independent analysis (use renamed graph).
    analyzer = EinsumGraphAnalyzer(debug=False)
    analysis = analyzer.analyze_graph(out_dir / "einsum_graph_renamed.yaml", out_dir, precision="fp32")
    assert analysis is not None
    assert (out_dir / "analysis.yaml").exists()
    assert analysis["total"]["num_layers"] > 0
    assert analysis["total"]["macs"] > 0

    # 3) analysis.yaml + arch -> perf prediction.
    perf_model = EinsumGraphPerfModel(debug=False)
    perf = perf_model.predict(out_dir / "analysis.yaml", out_dir, arch_config="H100_PCIe", precision="fp32")
    assert perf is not None
    assert (out_dir / "perf_H100_PCIe.yaml").exists()


