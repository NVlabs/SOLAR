"""Random Attention example model.

This implements random attention which only computes attention for
randomly selected positions. This provides a sparse approximation
of full attention.

Shape: (batch_size, num_heads, seq_len, head_dim) for Q, K, V
Output: (batch_size, num_heads, seq_len, head_dim)

Based on kernelbench/level4/24_attn_core_random.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomAttention(nn.Module):
    """Random attention - only computes attention for randomly selected positions."""

    def __init__(self, sparsity_prob: float, causal: bool = False):
        super().__init__()
        if not (0.0 <= sparsity_prob <= 1.0):
            raise ValueError("Sparsity probability must be between 0 and 1")
        self.sparsity_prob = sparsity_prob
        self.causal = causal

    def _sample_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sample random positions for attention."""
        # Create random mask
        random_mask = torch.rand(seq_len, seq_len, device=device) < self.sparsity_prob

        if self.causal:
            # Apply causal constraint
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            random_mask = random_mask & causal_mask

        # Always allow self-attention (diagonal)
        random_mask.fill_diagonal_(True)

        return random_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random attention.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        # Sample random positions
        random_pattern = self._sample_positions(seq_len, query.device)

        output = torch.zeros_like(query)

        for i in range(seq_len):
            # Find positions this query can attend to
            attend_positions = torch.nonzero(random_pattern[i], as_tuple=False).squeeze(-1)

            if len(attend_positions) == 0:
                continue

            # Extract relevant keys and values
            q_i = query[:, :, i : i + 1, :]  # (batch, heads, 1, head_dim)
            k_selected = key[:, :, attend_positions, :]  # (batch, heads, num_selected, head_dim)
            v_selected = value[:, :, attend_positions, :]  # (batch, heads, num_selected, head_dim)

            # Compute attention scores only for selected positions
            scores = torch.matmul(q_i, k_selected.transpose(-2, -1)) * scale

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Compute output for this position
            output[:, :, i : i + 1, :] = torch.matmul(attn_weights, v_selected)

        return output


class Model(nn.Module):
    """Wrapper model for random attention."""

    def __init__(self):
        super().__init__()
        self.attention = RandomAttention(sparsity_prob=0.5, causal=False)

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

