"""Standalone Multi-Head Attention model for Solar examples.

This file demonstrates the attention mechanism that is the core of transformers.
It follows the kernelbench-style API:

- `class Model(nn.Module)` implements `forward`
- `get_inputs()` returns a list of inputs for `forward`

The attention computation follows the standard formula:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

With multi-head attention:
  1. Project Q, K, V with learned linear layers
  2. Split into multiple heads
  3. Compute scaled dot-product attention per head
  4. Concatenate heads and project output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    This implements the attention pattern:
      - Q: [B, S, H] -> [B, heads, S, D]
      - K: [B, S, H] -> [B, heads, S, D]  
      - V: [B, S, H] -> [B, heads, S, D]
      - QK: [B, heads, S, S] = Q @ K^T / sqrt(D)
      - Attn: [B, heads, S, D] = softmax(QK) @ V
      - Out: [B, S, H] = concat(heads) @ W_o
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [B, S_q, H]
            key: Key tensor [B, S_k, H]
            value: Value tensor [B, S_v, H] (usually S_v == S_k)
            mask: Optional attention mask [B, 1, S_q, S_k] or [B, heads, S_q, S_k]
            
        Returns:
            Output tensor [B, S_q, H]
        """
        b, s_q, h = query.shape
        s_k = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)  # [B, S_q, H]
        k = self.k_proj(key)    # [B, S_k, H]
        v = self.v_proj(value)  # [B, S_k, H]

        # Reshape to [B, heads, S, D]
        q = q.view(b, s_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: [B, heads, S_q, S_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [B, heads, S_q, D]
        context = torch.matmul(attn_weights, v)

        # Reshape back: [B, S_q, H]
        context = context.transpose(1, 2).contiguous().view(b, s_q, h)

        # Output projection
        output = self.out_proj(context)

        return output


class Model(nn.Module):
    """Wrapper model for multi-head attention demonstration.
    
    This model applies self-attention where Q=K=V=input.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        seq_len: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.attention = MultiHeadAttention(hidden_size, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention forward pass.
        
        Args:
            x: Input tensor [B, S, H]
            
        Returns:
            Output tensor [B, S, H]
        """
        # Self-attention: Q=K=V=x
        return self.attention(x, x, x)


def get_inputs():
    """Return a small, deterministic input batch for graph extraction."""
    torch.manual_seed(42)
    batch = 2
    seq = 32
    hidden = 64
    x = torch.randn(batch, seq, hidden)
    return [x]


if __name__ == "__main__":
    # Quick test
    model = Model()
    inputs = get_inputs()
    output = model(*inputs)
    print(f"Input shape: {inputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Attention model works!")

