"""Core Neural Network Blocks for E.D.I.T.H. Architecture."""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding with learnable blending and dropout."""

    def __init__(self, d_model: int, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module with scaled dot-product and residual projections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k, v: [batch_size, seq_len, d_model]
            mask: Optional [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        Returns:
            context: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = q.size(0)

        # 1) Linear projections and reshape to [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3) Weighted combination and output projection
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(context)

        return out, attn_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer with residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention sublayer with Pre-LN
        norm_x = self.ln1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward sublayer with Pre-LN
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout2(ffn_out)

        return x
