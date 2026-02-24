"""Standalone BERT-like transformer model for Solar examples.

This file is intentionally self-contained and follows the kernelbench-style API:

- `class Model(nn.Module)` implements `forward`
- `get_inputs()` returns a list of inputs for `forward`
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """A minimal multi-head self-attention block."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """A minimal FFN block."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """A minimal transformer encoder layer (pre-norm omitted for simplicity)."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4, intermediate_size: int = 256):
        super().__init__()
        self.attn = SelfAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class Model(nn.Module):
    """Tiny BERT-like encoder with token+position embeddings and a classifier head."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, hidden_size * 4) for _ in range(num_layers)]
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S]
        b, s = input_ids.shape
        if s > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # [B, S, H]

        for layer in self.layers:
            x = layer(x)

        cls = x[:, 0, :]  # [B, H]
        return self.classifier(cls)  # [B, num_classes]


def get_inputs():
    """Return a small, deterministic input batch for graph extraction."""
    torch.manual_seed(0)
    batch = 2
    seq = 16
    vocab_size = 1000
    input_ids = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
    return [input_ids]


