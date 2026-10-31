"""E.D.I.T.H. V13 Decoder-Only Transformer Architecture (EdithLM).

Pure PyTorch implementation written from scratch without external HuggingFace classes.

Architecture Highlights:
1. Rotary Position Embeddings (RoPE) for relative position awareness and length extrapolation.
2. RMSNorm for stable pre-layer normalization.
3. SwiGLU activation for enhanced representation density.
4. Grouped-Query Attention (GQA) for high throughput and reduced KV cache memory footprint.
5. Key-Value (KV) Cache for O(1) autoregressive generation steps.
6. Weight Tying between token embeddings and output projection head.
7. Pure PyTorch int8 dynamic weight quantization.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EdithLMConfig:
    vocab_size: int = 3051
    max_seq_len: int = 1024
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    intermediate_dim: int = 1376
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    @classmethod
    def tiny(cls, vocab_size: int = 3051) -> "EdithLMConfig":
        """~25M Parameter configuration (Smoke Test & Fast Reflex Reasoning)."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=1024,
            dim=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            intermediate_dim=1376,
        )

    @classmethod
    def base(cls, vocab_size: int = 3051) -> "EdithLMConfig":
        """~72M Parameter configuration (Target Production Engine)."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=1024,
            dim=768,
            n_layers=16,
            n_heads=12,
            n_kv_heads=4,
            intermediate_dim=2048,
        )

    @classmethod
    def large(cls, vocab_size: int = 3051) -> "EdithLMConfig":
        """~138M Parameter configuration (Stretch Heavy Reasoning)."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=1024,
            dim=1024,
            n_layers=20,
            n_heads=16,
            n_kv_heads=4,
            intermediate_dim=2752,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes complex rotary position frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies complex Rotary Position Embeddings to Query and Key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[: xq_.shape[1], : xq_.shape[-1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with optional KV-cache."""

    def __init__(self, config: EdithLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Autoregressive KV-cache handling
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose to (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        output = torch.matmul(probs, v)  # (B, H, S, D)

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(output), kv_cache


class SwiGLUMLP(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network."""

    def __init__(self, config: EdithLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single EdithLM Transformer Layer."""

    def __init__(self, config: EdithLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        h_attn, kv_cache = self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache)
        x = x + h_attn
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv_cache


class EdithLM(nn.Module):
    """Complete Decoder-Only Generative Language Model for E.D.I.T.H. V13."""

    def __init__(self, config: EdithLMConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs_cis(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Counts total trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and not self.config.tie_word_embeddings:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        B, S = input_ids.shape

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + S].to(h.device)

        # Causal triangular attention mask
        if S > 1:
            mask = torch.full((S, S), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, S, S)
        else:
            mask = None

        new_kv_caches = [] if kv_caches is not None else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            h, updated_cache = layer(h, freqs_cis, mask, layer_cache)
            if new_kv_caches is not None:
                new_kv_caches.append(updated_cache)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss, new_kv_caches

    # -------------------------------------------------------------------------
    # Own-Trained Int8 Dynamic Quantization
    # -------------------------------------------------------------------------

    def quantize_int8(self) -> "EdithLM":
        """Applies pure PyTorch int8 dynamic quantization to Linear weights."""
        return torch.ao.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8
        )
