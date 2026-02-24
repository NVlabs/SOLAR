"""Sliding Window Attention example model.

This implements sliding window attention which only computes attention
within a local window around each position. This is more efficient than
dense attention for long sequences.

Each position i attends to positions [i - window_size, i + window_size].

Shape: (batch_size, num_heads, seq_len, head_dim) for Q, K, V
Output: (batch_size, num_heads, seq_len, head_dim)

Based on kernelbench/level4/23_attn_core_slidingwindows.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """Sliding window attention - only computes attention within a local window."""

    def __init__(self, window_size: int, causal: bool = False):
        super().__init__()
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        self.window_size = window_size
        self.causal = causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sliding window attention.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        output = torch.zeros_like(query)

        for i in range(seq_len):
            # Determine the window bounds for position i
            if self.causal:
                # Causal: can only attend to past and current positions
                start_pos = max(0, i - self.window_size)
                end_pos = i + 1
            else:
                # Non-causal: symmetric window around position i
                start_pos = max(0, i - self.window_size)
                end_pos = min(seq_len, i + self.window_size + 1)

            if start_pos >= end_pos:
                continue

            # Extract relevant keys and values for this query position
            q_i = query[:, :, i : i + 1, :]  # (batch, heads, 1, head_dim)
            k_window = key[:, :, start_pos:end_pos, :]  # (batch, heads, window_len, head_dim)
            v_window = value[:, :, start_pos:end_pos, :]  # (batch, heads, window_len, head_dim)

            # Compute attention scores for this window
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) * scale

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Compute output for this position
            output[:, :, i : i + 1, :] = torch.matmul(attn_weights, v_window)

        return output


class Model(nn.Module):
    """Wrapper model for sliding window attention."""

    def __init__(self):
        super().__init__()
        self.attention = SlidingWindowAttention(window_size=4, causal=False)

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

