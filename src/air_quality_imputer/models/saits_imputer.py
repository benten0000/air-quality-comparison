from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SAITSConfig:
    block_size: int = 24
    n_features: int = 10
    n_layers: int = 3
    d_model: int = 128
    n_head: int = 4
    dropout: float = 0.1
    d_ffn: int | None = None
    ORT_weight: int = 1
    MIT_weight: int = 1


class SAITSImputer:
    def __init__(self, config: SAITSConfig):
        self.config = config
        self.model = None
        self._build_model(epochs=300, batch_size=128, patience=250, lr=1e-3)

    def _require_saits(self):
        try:
            from pypots.imputation import SAITS
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SAITS requires 'pypots'. Install with: pip install pypots"
            ) from exc
        return SAITS

    def _build_model(self, epochs: int, batch_size: int, patience: int, lr: float):
        SAITS = self._require_saits()
        from pypots.optim.adam import Adam

        d_k = self.config.d_model // self.config.n_head
        d_v = self.config.d_model // self.config.n_head
        self.model = SAITS(
            n_steps=self.config.block_size,
            n_features=self.config.n_features,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            n_heads=self.config.n_head,
            d_k=d_k,
            d_v=d_v,
            d_ffn=self.config.d_ffn or (4 * self.config.d_model),
            dropout=self.config.dropout,
            ORT_weight=self.config.ORT_weight,
            MIT_weight=self.config.MIT_weight,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            optimizer=Adam(lr=lr),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def fit(
        self,
        dataset: dict[str, np.ndarray],
        epochs: int = 300,
        batch_size: int = 128,
        initial_lr: float = 1e-3,
        patience: int = 250,
        min_delta: float = 0.0,
        validation_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        del min_delta
        self._build_model(epochs=epochs, batch_size=batch_size, patience=patience, lr=initial_lr)
        assert self.model is not None
        train_set = {"X": dataset["X"]}
        val_set = None
        if validation_data is not None and "X" in validation_data and "X_ori" in validation_data:
            val_set = {"X": validation_data["X"], "X_ori": validation_data["X_ori"]}
        if val_set is not None:
            self.model.fit(train_set, val_set)
        else:
            self.model.fit(train_set)

    def impute(self, dataset: dict[str, np.ndarray]) -> np.ndarray | None:
        if self.model is None:
            return None
        output = self.model.impute({"X": dataset["X"]})
        if isinstance(output, dict):
            if "imputation" in output:
                return output["imputation"]
            if "X_imputed" in output:
                return output["X_imputed"]
        if isinstance(output, np.ndarray):
            return output
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"serialized_model": self.model}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model = state_dict["serialized_model"]

    def to(self, device: torch.device):
        del device
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def number_of_params(self) -> int:
        if self.model is None:
            return 0
        torch_model = getattr(self.model, "model", None)
        if torch_model is None:
            return 0
        return int(sum(p.numel() for p in torch_model.parameters()))
