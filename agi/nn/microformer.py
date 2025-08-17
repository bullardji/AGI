# agi/nn/microformer.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MicroConfig:
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10_000.0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # variance over last dim
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def build_rope_cache(head_dim: int,
                     max_seq_len: int,
                     device,
                     dtype,
                     base: float = 10_000.0):
    """
    Returns cos, sin with shape [1, 1, T, head_dim] for broadcasting onto q/k
    """
    # pairwise rotation requires even head_dim
    assert head_dim % 2 == 0, f"RoPE requires head_dim to be even, got {head_dim}"
    half = head_dim // 2

    t = torch.arange(max_seq_len, device=device, dtype=dtype)  # [T]
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))  # [half]
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, half]
    # duplicate for interleaving even/odd
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    cos = emb.cos()[None, None, :, :]  # [1,1,T,D]
    sin = emb.sin()[None, None, :, :]  # [1,1,T,D]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, T, D], cos/sin: [1,1,T,D]
    Applies pairwise rotation on even/odd dims.
    """
    # split into even/odd channels
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos_even = cos[..., ::2]
    cos_odd = cos[..., 1::2]
    sin_even = sin[..., ::2]
    sin_odd = sin[..., 1::2]

    # (a + jb) * e^{jθ} = (a cosθ - b sinθ) + j(a sinθ + b cosθ)
    x_rot_even = x_even * cos_even - x_odd * sin_even
    x_rot_odd = x_even * sin_odd + x_odd * cos_odd
    # interleave back
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


class MHA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int, rope_base: float):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} must be divisible by n_head {n_head}"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {self.head_dim}"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # rope cache filled lazily on first forward (device/dtype aware)
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

    def _maybe_build_rope(self, T: int, device, dtype):
        if (self._rope_cos is None or
            self._rope_cos.size(2) < T or
            self._rope_cos.device != device or
            self._rope_cos.dtype != dtype):
            cos, sin = build_rope_cache(self.head_dim, max(T, self.max_seq_len), device, dtype, self.rope_base)
            self._rope_cos = cos
            self._rope_sin = sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.shape
        H = self.n_head
        Hd = self.head_dim

        qkv = self.qkv(x)  # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B,T,D]

        # reshape to [B,H,T,Hd]
        def reshape_heads(t):
            return t.view(B, T, H, Hd).permute(0, 2, 1, 3).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # RoPE on q/k
        self._maybe_build_rope(T, x.device, x.dtype)
        cos = self._rope_cos[..., :T, :]  # [1,1,T,Dh]
        sin = self._rope_sin[..., :T, :]  # [1,1,T,Dh]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # SDPA expects [B,H,T,D]
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )  # [B,H,T,Hd]

        # merge heads back
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B,T,D]
        return self.out(attn_out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: MicroConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MHA(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len, cfg.rope_base)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, cfg: MicroConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [B, T] (token ids)
        returns: logits [B, T, V]
        """
        x = self.tok_emb(idx)              # [B,T,D]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
