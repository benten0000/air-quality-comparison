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
    def __init__(self, head_dim, max_len=5000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_len).unsqueeze(1)
        angle = positions * theta.unsqueeze(0)
        self.register_buffer("cos_cached", torch.cos(angle))
        self.register_buffer("sin_cached", torch.sin(angle))

    def forward(self, x):
        # x: (B, n_head, T, head_dim)
        _, _, T, _ = x.size()
        cos = cast(torch.Tensor, self.cos_cached)[:T].unsqueeze(0).unsqueeze(0)
        sin = cast(torch.Tensor, self.sin_cached)[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ],
            dim=-1,
        ).flatten(-2)
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
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.head_dim = config.d_model // config.n_head
        self.rotary = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # RoPE applied per-head (difference vs original provided script).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.rotary(q)
        k = self.rotary(k)

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
        # SwiGLU (difference vs original provided script).
        x = self.c_fc(x) * F.silu(self.c_gate(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


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


class TransformerImputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(config.n_features, config.d_model)
        self.mask_proj = nn.Linear(config.n_features, config.d_model, bias=config.bias)
        self.combined_proj = nn.Linear(config.d_model, config.d_model)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.n_features)

    def forward(self, x, mask):
        x_embedded = self.feature_proj(x)
        mask_embedded = self.mask_proj(mask)
        combined = x_embedded + mask_embedded
        combined = self.combined_proj(combined)
        for block in self.transformer:
            combined = block(combined)
        combined = self.ln_f(combined)
        return self.output_head(combined)

    def fit(self, dataset, epochs=300, batch_size=128, initial_lr=1e-3, patience=250, min_delta=0.0, validation_data=None):
        device = torch.device("cuda")
        self.to(device)
        compiled_model = cast(nn.Module, torch.compile(self))
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        X = dataset["X"]
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            persistent_workers=False,
        )
        warmup_epochs = epochs // 8
        total_epochs = epochs
        eta_min = 1e-7
        optimizer = torch.optim.AdamW(
            compiled_model.parameters(),
            lr=initial_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=False,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.3,
            total_iters=warmup_epochs,
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        use_validation_for_early_stopping = validation_data is not None
        best_loss = float("inf")
        patience_counter = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            compiled_model.train()
            total_loss = 0.0
            num_batches = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device, non_blocking=False)
                original_mask = (~torch.isnan(batch_x)).float()
                mit_mask = (torch.rand_like(batch_x, device=device) < 0.25) & original_mask.bool()
                x_input = batch_x.clone()
                x_input[mit_mask] = 0
                input_mask = original_mask.clone()
                input_mask[mit_mask] = 0
                torch.nan_to_num_(x_input, 0)
                target = torch.nan_to_num(batch_x.clone(), 0)
                with torch.cuda.amp.autocast():
                    predictions = compiled_model(x_input, input_mask)
                    mit_loss = (
                        torch.abs(predictions[mit_mask] - target[mit_mask]).mean()
                        if mit_mask.sum() > 0
                        else torch.tensor(0.0, device=device)
                    )
                    observed_mask = original_mask.bool() & ~mit_mask
                    ort_loss = (
                        torch.abs(predictions[observed_mask] - target[observed_mask]).mean()
                        if observed_mask.sum() > 0
                        else torch.tensor(0.0, device=device)
                    )
                    loss = mit_loss + ort_loss
                if loss > 0:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    num_batches += 1
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                val_loss = self._validate_imputation(validation_data) if validation_data else None
                scheduler.step()
                if epoch % 10 == 0 or epoch < 10:
                    current_lr = scheduler.get_last_lr()[0]
                    if val_loss and not np.isnan(val_loss):
                        print(f"Epoch {epoch:3d}, Train: {avg_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
                    else:
                        print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                current_loss = val_loss if use_validation_for_early_stopping and val_loss and not np.isnan(val_loss) else avg_loss
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
            device = torch.device("cuda")
            X = dataset["X"]
            if X is None or len(X) == 0:
                return None
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                missing_mask = torch.isnan(X_tensor)
                observation_mask = (~missing_mask).float()
                X_input = torch.nan_to_num(X_tensor, 0)
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        predictions = self.forward(X_input, observation_mask)
                    predictions = predictions.float()
                else:
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
