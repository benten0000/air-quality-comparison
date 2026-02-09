import os
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_len).unsqueeze(1)
        angle = positions * theta.unsqueeze(0)
        self.register_buffer("cos_cached", torch.cos(angle))
        self.register_buffer("sin_cached", torch.sin(angle))

    def forward(self, x):
        _, T, _ = x.size()
        cos = cast(torch.Tensor, self.cos_cached)[:T]
        sin = cast(torch.Tensor, self.sin_cached)[:T]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1).flatten(-2)
        return x_rot


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


_diagonal_mask_cache = {}


def get_diagonal_mask(T, device):
    cache_key = (T, str(device))
    if cache_key not in _diagonal_mask_cache:
        mask = torch.zeros((T, T), device=device)
        mask.fill_diagonal_(-1e9)
        _diagonal_mask_cache[cache_key] = mask
    return _diagonal_mask_cache[cache_key]


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.head_dim = config.d_model // config.n_head
        self.rotary = RotaryPositionalEncoding(config.d_model)
        self.qk_norm = config.qk_norm
        self.qk_norm_eps = config.qk_norm_eps

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.rotary(q)
        k = self.rotary(k)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)

        mask = get_diagonal_mask(T, x.device)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.d_model
        self.c_fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_val = self.c_fc(x)
        x_gate = F.silu(self.c_gate(x))
        x = x_val * x_gate
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    block_size: int = 24
    n_features: int = 10
    d_model: int = 128
    t_dim: int = 24
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-6
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    train_mask_rate: float = 0.25
    train_mask_rate_min: float = 0.10
    train_mask_rate_max: float = 0.90
    random_target_ratio: bool = True
    masked_huber_delta: float = 1.0
    masked_robust_weight: float = 1.0


class TransformerImputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(config.n_features, config.d_model)
        self.mask_proj = nn.Linear(config.n_features, config.d_model, bias=config.bias)
        self.combined_proj = nn.Linear(config.d_model, config.d_model)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1),
            )
            for _ in range(config.n_features)
        ])

    def forward(self, x, mask):
        x_embedded = self.feature_proj(x)
        mask_embedded = self.mask_proj(mask)
        combined = self.combined_proj(x_embedded + mask_embedded)
        for block in self.transformer:
            combined = block(combined)
        combined = self.ln_f(combined)
        return torch.cat([head(combined) for head in self.output_heads], dim=-1)

    def fit(
        self,
        dataset,
        epochs=300,
        batch_size=128,
        initial_lr=1e-3,
        patience=250,
        min_delta=0.0,
        validation_data=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X = dataset["X"]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            persistent_workers=False,
        )

        warmup_epochs = max(1, epochs // 8)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=initial_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.3,
            total_iters=warmup_epochs,
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        use_validation_for_early_stopping = validation_data is not None
        best_loss = float("inf")
        patience_counter = 0

        amp_enabled = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0

            for (batch_x_cpu,) in loader:
                batch_x = batch_x_cpu.to(device)
                original_mask = (~torch.isnan(batch_x)).float()
                if self.config.random_target_ratio:
                    target_rate = float(
                        torch.empty(1, device=device).uniform_(
                            self.config.train_mask_rate_min,
                            self.config.train_mask_rate_max,
                        )
                    )
                else:
                    target_rate = self.config.train_mask_rate
                mit_mask = (torch.rand_like(batch_x) < target_rate) & original_mask.bool()

                x_input = batch_x.clone()
                x_input[mit_mask] = 0.0
                input_mask = original_mask.clone()
                input_mask[mit_mask] = 0.0

                torch.nan_to_num_(x_input, nan=0.0)
                target = torch.nan_to_num(batch_x.clone(), nan=0.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    predictions = self.forward(x_input, input_mask)
                    if mit_mask.sum() > 0:
                        masked_pred = predictions[mit_mask]
                        masked_target = target[mit_mask]
                        mit_mae = torch.abs(masked_pred - masked_target).mean()
                        mit_huber = F.huber_loss(masked_pred, masked_target, delta=self.config.masked_huber_delta)
                        mit_loss = mit_mae + self.config.masked_robust_weight * mit_huber
                    else:
                        mit_loss = torch.tensor(0.0, device=device)
                    observed_mask = original_mask.bool() & ~mit_mask
                    ort_loss = (
                        torch.abs(predictions[observed_mask] - target[observed_mask]).mean()
                        if observed_mask.sum() > 0
                        else torch.tensor(0.0, device=device)
                    )
                    loss = mit_loss + ort_loss

                if loss > 0:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    num_batches += 1

            if num_batches == 0:
                continue

            avg_loss = total_loss / num_batches
            val_loss = self._validate_imputation(validation_data) if validation_data else None
            scheduler.step()

            if epoch % 10 == 0 or epoch < 10:
                current_lr = scheduler.get_last_lr()[0]
                if val_loss is not None and not np.isnan(val_loss):
                    print(f"Epoch {epoch:3d}, Train: {avg_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
                else:
                    print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

            current_loss = avg_loss
            if use_validation_for_early_stopping and val_loss is not None and not np.isnan(val_loss):
                current_loss = float(val_loss)

            if current_loss < best_loss - min_delta:
                best_loss, patience_counter = current_loss, 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Ucenje se je predcasno ustavilo pri epohi {epoch} (najnizja izguba: {best_loss:.6f})")
                break

        print(f"Ucenje zakljuceno. Koncna vrednost izgube: {best_loss:.6f}")

    def _validate_imputation(self, validation_data):
        try:
            X_val_masked = validation_data["X"]
            X_val_ori = validation_data["X_ori"]
            if X_val_masked is None or X_val_ori is None:
                return np.nan
            if len(X_val_masked) == 0 or len(X_val_ori) == 0:
                return np.nan

            missing_mask = np.isnan(X_val_masked) & ~np.isnan(X_val_ori)
            if not missing_mask.any():
                return np.nan

            dataset_masked = {"X": X_val_masked}
            self.eval()
            with torch.no_grad():
                X_imputed = self.impute(dataset_masked)
                if X_imputed is None:
                    return np.nan
                original_values = X_val_ori[missing_mask]
                imputed_values = X_imputed[missing_mask]
                if np.isnan(original_values).any() or np.isnan(imputed_values).any():
                    return np.nan
                mae = np.abs(imputed_values - original_values).mean()
                if np.isnan(mae) or np.isinf(mae):
                    return np.nan
                return mae
        except Exception as e:
            print(f"Validacija imputacije ni uspela: {e}")
            return np.nan
        finally:
            self.train()

    def impute(self, dataset):
        try:
            self.eval()
            device = next(self.parameters()).device
            X = dataset["X"]
            if X is None or len(X) == 0:
                return None

            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                missing_mask = torch.isnan(X_tensor)
                observation_mask = (~missing_mask).float()
                X_input = torch.nan_to_num(X_tensor, nan=0.0)
                predictions = self.forward(X_input, observation_mask)
                result = X_input.clone()
                result[missing_mask] = predictions[missing_mask]
                if torch.isnan(result).any():
                    print("NaN vrednosti v rezultatu zapolnjevanja")
                    return None
                return result.cpu().numpy()
        except Exception as e:
            print(f"Napaka pri zapolnjevanju manjkajocih vrednosti: {e}")
            return None

    def number_of_params(self):
        return sum(p.numel() for p in self.parameters())
