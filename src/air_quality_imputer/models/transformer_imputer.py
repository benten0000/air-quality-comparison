from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from air_quality_imputer.models.forecast_model_mixin import ForecastModelMixin


@dataclass
class TransformerConfig:
    block_size: int = 24
    n_features: int = 10
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    d_ffn: int | None = None
    diagonal_attention_mask: bool = True
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-6


class TransformerImputer(ForecastModelMixin, nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.missing_value = nn.Parameter(torch.zeros(config.n_features, dtype=torch.float32))

        in_dim = int(config.n_features) * 2  # values + observation mask
        d_ffn = int(config.d_ffn) if config.d_ffn is not None else 4 * int(config.d_model)

        self.in_proj = nn.Linear(in_dim, int(config.d_model), bias=bool(config.bias))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_head),
            dim_feedforward=d_ffn,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            bias=bool(config.bias),
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config.n_layer),
            norm=nn.LayerNorm(int(config.d_model), eps=float(config.norm_eps), bias=bool(config.bias)),
        )
        self.out_proj = nn.Linear(int(config.d_model), int(config.n_features), bias=bool(config.bias))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        station_ids: torch.Tensor | None = None,
        station_geo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del station_ids, station_geo
        mv = self.missing_value.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
        x_filled = x * mask + mv * (1.0 - mask)
        h = self.in_proj(torch.cat([x_filled, mask], dim=-1))

        attn_mask = None
        if bool(self.config.diagonal_attention_mask):
            seq_len = int(h.size(1))
            attn_mask = torch.eye(seq_len, device=h.device, dtype=torch.bool)

        h = self.encoder(h, mask=attn_mask)
        return self.out_proj(h)
