from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from air_quality_imputer.models.forecast_model_mixin import ForecastModelMixin


@dataclass
class RecurrentForecasterConfig:
    n_features: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bias: bool = True


@dataclass
class HybridLocLSTMConfig(RecurrentForecasterConfig):
    num_stations: int = 1
    embed_dim: int = 16
    geo_dim: int = 2
    station_feature_index: int = 0


class RecurrentForecasterBase(ForecastModelMixin, nn.Module):
    rnn_cls: type[nn.Module]

    def __init__(self, config: RecurrentForecasterConfig):
        super().__init__()
        self.config = config
        self.missing_value = nn.Parameter(torch.zeros(config.n_features, dtype=torch.float32))
        self.rnn = self.rnn_cls(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=float(config.dropout) if int(config.num_layers) > 1 else 0.0,
            bias=bool(config.bias),
        )
        self.output_head = nn.Linear(config.hidden_size, config.n_features, bias=bool(config.bias))

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
        out, _ = self.rnn(x_filled)
        return self.output_head(out)


class LSTMForecaster(RecurrentForecasterBase):
    rnn_cls = nn.LSTM


class GRUForecaster(RecurrentForecasterBase):
    rnn_cls = nn.GRU


class HybridLocLSTM(ForecastModelMixin, nn.Module):
    def __init__(self, config: HybridLocLSTMConfig):
        super().__init__()
        self.config = config
        self.missing_value = nn.Parameter(torch.zeros(config.n_features, dtype=torch.float32))
        self.station_emb = nn.Embedding(config.num_stations, config.embed_dim)
        self.geo_mlp = nn.Sequential(
            nn.Linear(config.geo_dim, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, config.embed_dim, bias=True),
            nn.ReLU(),
        )

        use_station_feature = 0 <= int(config.station_feature_index) < int(config.n_features)
        dyn_features = int(config.n_features) - 1 if use_station_feature else int(config.n_features)
        self.dyn_features = dyn_features
        self.use_station_feature = use_station_feature
        self.station_feature_index = int(config.station_feature_index)

        self.lstm = nn.LSTM(
            input_size=dyn_features + int(config.embed_dim),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            batch_first=True,
            dropout=float(config.dropout) if int(config.num_layers) > 1 else 0.0,
            bias=bool(config.bias),
        )
        self.output_head = nn.Linear(int(config.hidden_size), int(config.n_features), bias=bool(config.bias))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        station_ids: torch.Tensor | None = None,
        station_geo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mv = self.missing_value.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
        x_filled = x * mask + mv * (1.0 - mask)

        if self.use_station_feature:
            idx = self.station_feature_index
            dyn_seq = torch.cat([x_filled[..., :idx], x_filled[..., idx + 1 :]], dim=-1)
        else:
            dyn_seq = x_filled

        if station_ids is None:
            if self.use_station_feature:
                idx = self.station_feature_index
                station_ids = torch.round(x_filled[:, -1, idx]).long()
            else:
                station_ids = torch.zeros((x_filled.size(0),), dtype=torch.long, device=x_filled.device)
        station_ids = station_ids.to(device=x_filled.device, dtype=torch.long)
        station_ids = torch.clamp(station_ids, min=0, max=int(self.config.num_stations) - 1)

        if station_geo is None:
            station_geo = torch.zeros(
                (x_filled.size(0), int(self.config.geo_dim)),
                dtype=x_filled.dtype,
                device=x_filled.device,
            )
        station_geo = station_geo.to(device=x_filled.device, dtype=x_filled.dtype)

        e_id = self.station_emb(station_ids)
        e_geo = self.geo_mlp(station_geo)
        e = e_id + e_geo
        e_seq = e.unsqueeze(1).expand(-1, dyn_seq.size(1), -1)
        seq = torch.cat([dyn_seq, e_seq], dim=-1)
        out, _ = self.lstm(seq)
        return self.output_head(out)


HybridLocLSTMForecaster = HybridLocLSTM
