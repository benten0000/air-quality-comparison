from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from air_quality_imputer.models.forecast_model_mixin import ForecastModelMixin


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 5000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_len).unsqueeze(1)
        angle = positions * theta.unsqueeze(0)
        self.register_buffer("cos_cached", torch.cos(angle))
        self.register_buffer("sin_cached", torch.sin(angle))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, seq_len, _ = x.size()
        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        cos = cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.n_head = int(config.n_head)
        self.dropout = float(config.dropout)

        if self.n_head <= 0:
            raise ValueError("n_head must be > 0")

        if config.d_k is None:
            if config.d_model % self.n_head != 0:
                raise ValueError("d_model must be divisible by n_head when d_k is not set")
            self.d_k = config.d_model // self.n_head
        else:
            self.d_k = int(config.d_k)

        if config.d_v is None:
            if config.d_model % self.n_head != 0:
                raise ValueError("d_model must be divisible by n_head when d_v is not set")
            self.d_v = config.d_model // self.n_head
        else:
            self.d_v = int(config.d_v)

        if self.d_k <= 0 or self.d_v <= 0:
            raise ValueError("d_k and d_v must be > 0")
        if self.d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary positional encoding")

        self.q_proj = nn.Linear(config.d_model, self.n_head * self.d_k, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, self.n_head * self.d_k, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, self.n_head * self.d_v, bias=config.bias)
        self.c_proj = nn.Linear(self.n_head * self.d_v, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryPositionalEncoding(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.d_v)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        hidden_dim = int(config.d_ffn) if config.d_ffn is not None else (4 * config.d_model)
        if hidden_dim <= 0:
            raise ValueError("d_ffn must be > 0")
        self.c_fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x) * F.silu(self.c_gate(x))
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ff = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    block_size: int = 24
    n_features: int = 10
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    d_ffn: int | None = None
    d_k: int | None = None
    d_v: int | None = None
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-6
    learning_rate: float = 1e-3
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False
    optimizer_fused: bool = True
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_ratio: float = 0.125
    scheduler_warmup_start_factor: float = 0.3
    scheduler_min_lr: float = 1e-7
    grad_clip_norm: float = 0.5


class TransformerForecaster(ForecastModelMixin, nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(config.n_features, config.d_model)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.n_features)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        station_ids: torch.Tensor | None = None,
        station_geo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask, station_ids, station_geo
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B,T,F), got {tuple(x.shape)}")
        if x.size(-1) != self.config.n_features:
            raise ValueError(
                f"Expected last dim F={self.config.n_features}, got {x.size(-1)}"
            )

        h = self.feature_proj(x)
        for block in self.transformer:
            h = block(h)
        h = self.ln_f(h)
        h_last = h[:, -1, :]
        return self.output_head(h_last)
