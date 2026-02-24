"""Block Sparse Attention example model.

This implements block-sparse attention which divides the attention matrix
into blocks and only computes attention for selected blocks. This provides
a structured sparse approximation of full attention.

Shape: (batch_size, num_heads, seq_len, head_dim) for Q, K, V
Output: (batch_size, num_heads, seq_len, head_dim)

Based on kernelbench/level4/25_attn_core_blockrandom.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSparseAttention(nn.Module):
    """Block-sparse attention - divides attention into blocks and selects which to compute."""

    def __init__(
        self,
        block_size: int,
        sparsity_prob: float = 0.5,
        causal: bool = False,
    ):
        super().__init__()
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        if not (0.0 <= sparsity_prob <= 1.0):
            raise ValueError("Sparsity probability must be between 0 and 1")

        self.block_size = block_size
        self.sparsity_prob = sparsity_prob
        self.causal = causal

    def _get_block_pattern(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate block-sparse pattern."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Random block pattern
        block_mask = torch.rand(num_blocks, num_blocks, device=device) < self.sparsity_prob

        if self.causal:
            # Apply causal constraint to blocks
            causal_block_mask = torch.tril(
                torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool)
            )
            block_mask = block_mask & causal_block_mask

        # Always include diagonal blocks
        block_mask.fill_diagonal_(True)

        return block_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply block-sparse attention.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        # Get block pattern
        block_pattern = self._get_block_pattern(seq_len, query.device)
        num_blocks = block_pattern.shape[0]

        output = torch.zeros_like(query)

        # Process each block
        for i in range(num_blocks):
            for j in range(num_blocks):
                if not block_pattern[i, j]:
                    continue

                # Block boundaries
                q_start = i * self.block_size
                q_end = min((i + 1) * self.block_size, seq_len)
                k_start = j * self.block_size
                k_end = min((j + 1) * self.block_size, seq_len)

                # Extract block
                q_block = query[:, :, q_start:q_end, :]
                k_block = key[:, :, k_start:k_end, :]
                v_block = value[:, :, k_start:k_end, :]

                # Compute attention for this block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

                # Softmax
                attn_weights = F.softmax(scores, dim=-1)

                # Accumulate output
                output[:, :, q_start:q_end, :] += torch.matmul(attn_weights, v_block)

        return output


class Model(nn.Module):
    """Wrapper model for block-sparse attention."""

    def __init__(self):
        super().__init__()
        self.attention = BlockSparseAttention(
            block_size=8,
            sparsity_prob=0.5,
            causal=True,
        )

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

