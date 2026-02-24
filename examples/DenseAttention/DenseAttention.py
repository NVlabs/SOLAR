"""Dense Attention example model.

This implements standard dense attention which computes the full attention matrix:
    attention = softmax(Q @ K^T / sqrt(d_k)) @ V

Shape: (batch_size, num_heads, seq_len, head_dim) for Q, K, V
Output: (batch_size, num_heads, seq_len, head_dim)

Based on kernelbench/level4/22_attn_core_dense.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttention(nn.Module):
    """Standard dense attention - computes full attention matrix."""

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply dense attention.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, value)

        return output


class Model(nn.Module):
    """Wrapper model for dense attention."""

    def __init__(self):
        super().__init__()
        self.attention = DenseAttention(causal=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.attention(q, k, v)


def get_inputs():
    """Get sample inputs for the model."""
    B, H, L, D = 2, 4, 32, 64  # batch, heads, seq_len, head_dim
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    return [q, k, v]


def get_init_inputs():
    """Get initialization inputs (none needed)."""
    return []

