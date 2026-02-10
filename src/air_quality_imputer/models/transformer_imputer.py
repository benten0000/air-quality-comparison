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


def get_diagonal_mask(T, device):
    mask = torch.zeros((T, T), device=device)
    mask.fill_diagonal_(-1e9)
    return mask


class MultiHeadSelfAttention(nn.Module):
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

SelfAttention = MultiHeadSelfAttention


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
        self.attn = MultiHeadSelfAttention(config)
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
    n_stations: int = 1
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-6
    train_mask_mode: str = "block_feature"
    train_missing_rate: float = 0.25
    train_block_min_len: int = 2
    train_block_max_len: int = 12
    train_block_missing_prob: float = 0.35
    train_feature_block_prob: float = 0.6
    train_block_no_overlap: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False
    optimizer_fused: bool = True
    preload_data_to_gpu: bool = True
    gpu_preload_max_mem_frac: float = 0.5
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True


class TransformerImputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(config.n_features, config.d_model)
        self.mask_proj = nn.Linear(config.n_features, config.d_model, bias=config.bias)
        self.station_emb = nn.Embedding(max(1, int(config.n_stations)), config.d_model)
        self.combined_proj = nn.Linear(3 * config.d_model, config.d_model)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.n_features)

    def forward(self, x, mask, station_ids: torch.Tensor | None = None):
        x_embedded = self.feature_proj(x)
        mask_embedded = self.mask_proj(mask)
        if station_ids is not None:
            station_ids = station_ids.long()
            if station_ids.dim() == 1:
                station_embedded = self.station_emb(station_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            elif station_ids.dim() == 2:
                station_embedded = self.station_emb(station_ids)
            else:
                raise ValueError(f"station_ids must be 1D or 2D, got shape {tuple(station_ids.shape)}")
        else:
            station_embedded = torch.zeros_like(x_embedded)
        combined = torch.cat([x_embedded, mask_embedded, station_embedded], dim=-1)
        combined = self.combined_proj(combined)
        for block in self.transformer:
            combined = block(combined)
        combined = self.ln_f(combined)
        return self.output_head(combined)

    def _sample_train_mask(self, observed_mask: torch.Tensor, never_mask_indices: list[int] | None = None) -> torch.Tensor:
        mode = getattr(self.config, "train_mask_mode", "block_feature")
        missing_rate = float(getattr(self.config, "train_missing_rate", 0.25))
        missing_rate = min(max(missing_rate, 0.0), 1.0)
        observed = observed_mask.bool()
        bsz, seq_len, n_feat = observed.shape
        allowed_feature_mask = torch.ones(n_feat, dtype=torch.bool, device=observed.device)
        if never_mask_indices:
            for idx in never_mask_indices:
                if 0 <= idx < n_feat:
                    allowed_feature_mask[idx] = False
        if not allowed_feature_mask.any():
            return torch.zeros_like(observed, dtype=torch.bool)
        maskable_observed = observed & allowed_feature_mask.view(1, 1, n_feat)

        if mode == "random":
            return (torch.rand_like(observed_mask) < missing_rate) & maskable_observed

        if mode != "block_feature":
            raise ValueError(f"Unsupported train_mask_mode: {mode}")

        min_len = int(getattr(self.config, "train_block_min_len", 2))
        max_len = int(getattr(self.config, "train_block_max_len", 12))
        if min_len <= 0 or max_len <= 0:
            raise ValueError("train_block_min_len and train_block_max_len must be positive")
        if min_len > max_len:
            raise ValueError("train_block_min_len must be <= train_block_max_len")
        p_block_missing = float(getattr(self.config, "train_block_missing_prob", missing_rate))
        p_block_missing = min(max(p_block_missing, 0.0), 1.0)
        p_feature_missing = float(getattr(self.config, "train_feature_block_prob", 0.6))
        p_feature_missing = min(max(p_feature_missing, 0.0), 1.0)
        no_overlap = bool(getattr(self.config, "train_block_no_overlap", True))

        if no_overlap:
            n_blocks = max(1, (seq_len + min_len - 1) // min_len)
            block_lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device)
            block_ends = torch.cumsum(block_lengths, dim=1)
            block_starts = torch.cat(
                [torch.zeros((bsz, 1), device=observed.device, dtype=block_ends.dtype), block_ends[:, :-1]],
                dim=1,
            )
            block_starts = torch.clamp(block_starts, max=seq_len).long()
            block_ends = torch.clamp(block_ends, max=seq_len).long()
        else:
            avg_block_len = max(1.0, (min_len + max_len) / 2.0)
            n_blocks = max(1, int(round(seq_len / avg_block_len)))
            block_lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device)
            block_lengths = torch.clamp(block_lengths, max=seq_len)
            start_max = (seq_len - block_lengths + 1).clamp(min=1).float()
            block_starts = torch.floor(torch.rand((bsz, n_blocks), device=observed.device) * start_max).long()
            block_ends = block_starts + block_lengths

        block_valid = block_starts < seq_len
        block_active = (torch.rand((bsz, n_blocks), device=observed.device) < p_block_missing) & block_valid
        feature_sel = (torch.rand((bsz, n_blocks, n_feat), device=observed.device) < p_feature_missing) & allowed_feature_mask.view(
            1, 1, n_feat
        )
        active_feat = feature_sel & block_active.unsqueeze(-1)

        mit_mask = torch.zeros_like(observed, dtype=torch.bool)
        time_idx = torch.arange(seq_len, device=observed.device).view(1, seq_len, 1)
        for block_idx in range(n_blocks):
            start = block_starts[:, block_idx].view(bsz, 1, 1)
            end = block_ends[:, block_idx].view(bsz, 1, 1)
            time_sel = (time_idx >= start) & (time_idx < end)
            mit_mask |= time_sel & active_feat[:, block_idx].view(bsz, 1, n_feat)

        mit_mask = mit_mask & maskable_observed

        # Keep effective missing ratio close to train_missing_rate on maskable entries.
        target_missing = int(round(missing_rate * maskable_observed.sum().item()))
        current_missing = int(mit_mask.sum().item())
        if current_missing > target_missing:
            active = torch.nonzero(mit_mask.view(-1), as_tuple=False).squeeze(1)
            if active.numel() > 0:
                keep = target_missing
                if keep <= 0:
                    mit_mask.zero_()
                elif keep < active.numel():
                    keep_idx = active[torch.randperm(active.numel(), device=observed.device)[:keep]]
                    new_mask = torch.zeros_like(mit_mask.view(-1), dtype=torch.bool)
                    new_mask[keep_idx] = True
                    mit_mask = new_mask.view_as(mit_mask)
        elif current_missing < target_missing:
            available = torch.nonzero((maskable_observed & ~mit_mask).view(-1), as_tuple=False).squeeze(1)
            need = target_missing - current_missing
            if need > 0 and available.numel() > 0:
                add_idx = available[torch.randperm(available.numel(), device=observed.device)[: min(need, available.numel())]]
                mit_mask.view(-1)[add_idx] = True

        return mit_mask

    def fit(self, dataset, epochs=300, batch_size=128, initial_lr=1e-3, patience=250, min_delta=0.0, validation_data=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        compiled_model: nn.Module = self
        use_torch_compile = bool(getattr(self.config, "use_torch_compile", True))
        if device.type == "cuda" and use_torch_compile:
            try:
                compile_mode = str(getattr(self.config, "compile_mode", "reduce-overhead"))
                compile_dynamic = bool(getattr(self.config, "compile_dynamic", False))
                compiled_model = cast(
                    nn.Module,
                    torch.compile(
                        self,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                    ),
                )
            except Exception as exc:
                print(f"torch.compile unavailable, fallback to eager mode: {exc}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        X = dataset["X"]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        station_ids = dataset.get("station_ids")
        S_tensor = None
        if station_ids is not None:
            S_tensor = torch.tensor(station_ids, dtype=torch.long)
            if len(S_tensor) != len(X_tensor):
                raise ValueError("station_ids length must match X length")

        amp_enabled = device.type == "cuda"
        use_gpu_preload = False
        if amp_enabled and bool(getattr(self.config, "preload_data_to_gpu", True)):
            bytes_needed = X_tensor.numel() * X_tensor.element_size()
            if S_tensor is not None:
                bytes_needed += S_tensor.numel() * S_tensor.element_size()
            free_mem, _ = torch.cuda.mem_get_info(device)
            max_frac = float(getattr(self.config, "gpu_preload_max_mem_frac", 0.5))
            max_frac = min(max(max_frac, 0.05), 0.95)
            if bytes_needed <= int(free_mem * max_frac):
                X_tensor = X_tensor.to(device, non_blocking=True)
                if S_tensor is not None:
                    S_tensor = S_tensor.to(device, non_blocking=True)
                use_gpu_preload = True
                print(f"[INFO] Preloaded training tensors to GPU ({bytes_needed / (1024**2):.1f} MiB).")
            else:
                print(
                    f"[INFO] Skipping GPU preload ({bytes_needed / (1024**2):.1f} MiB needed, "
                    f"{free_mem / (1024**2):.1f} MiB free)."
                )

        loader = None
        if not use_gpu_preload:
            pin_memory = bool(getattr(self.config, "dataloader_pin_memory", True)) and amp_enabled
            num_workers = max(0, int(getattr(self.config, "dataloader_num_workers", 0)))
            persistent_workers = num_workers > 0
            if S_tensor is not None:
                loader = DataLoader(
                    TensorDataset(X_tensor, S_tensor),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                )
            else:
                loader = DataLoader(
                    TensorDataset(X_tensor),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                )
        warmup_epochs = max(1, epochs // 8)
        total_epochs = epochs
        eta_min = 1e-7
        use_fused_optim = bool(getattr(self.config, "optimizer_fused", True)) and amp_enabled
        try:
            optimizer = torch.optim.AdamW(
                compiled_model.parameters(),
                lr=initial_lr,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=use_fused_optim,
            )
        except Exception:
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
            T_max=max(1, total_epochs - warmup_epochs),
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
        scaler = torch.GradScaler(device.type, enabled=amp_enabled)
        never_mask_indices = dataset.get("never_mask_feature_indices")
        use_cudagraph_step_mark = (
            device.type == "cuda"
            and use_torch_compile
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        )
        for epoch in range(epochs):
            compiled_model.train()
            total_loss = 0.0
            num_batches = 0
            if use_gpu_preload:
                n_samples = X_tensor.shape[0]
                perm = torch.randperm(n_samples, device=device)
                for start in range(0, n_samples, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_x = X_tensor.index_select(0, idx)
                    batch_station = S_tensor.index_select(0, idx) if S_tensor is not None else None
                    original_mask = (~torch.isnan(batch_x)).float()
                    mit_mask = self._sample_train_mask(original_mask, never_mask_indices=never_mask_indices)
                    if not mit_mask.any():
                        continue
                    x_input = batch_x.clone()
                    x_input[mit_mask] = 0
                    input_mask = original_mask.clone()
                    input_mask[mit_mask] = 0
                    torch.nan_to_num_(x_input, 0)
                    target = torch.nan_to_num(batch_x.clone(), 0)
                    if use_cudagraph_step_mark:
                        torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        predictions = compiled_model(x_input, input_mask, batch_station)
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
                    if torch.isfinite(loss).item() and loss.item() > 0:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                        total_loss += loss.item()
                        num_batches += 1
            else:
                if loader is None:
                    raise RuntimeError("DataLoader was not initialized.")
                for batch in loader:
                    if len(batch) == 2:
                        batch_x, batch_station = batch
                        batch_station = batch_station.to(device, non_blocking=amp_enabled)
                    else:
                        (batch_x,) = batch
                        batch_station = None
                    batch_x = batch_x.to(device, non_blocking=amp_enabled)
                    original_mask = (~torch.isnan(batch_x)).float()
                    mit_mask = self._sample_train_mask(original_mask, never_mask_indices=never_mask_indices)
                    if not mit_mask.any():
                        continue
                    x_input = batch_x.clone()
                    x_input[mit_mask] = 0
                    input_mask = original_mask.clone()
                    input_mask[mit_mask] = 0
                    torch.nan_to_num_(x_input, 0)
                    target = torch.nan_to_num(batch_x.clone(), 0)
                    if use_cudagraph_step_mark:
                        torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        predictions = compiled_model(x_input, input_mask, batch_station)
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
                    if torch.isfinite(loss).item() and loss.item() > 0:
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
                    if val_loss is not None and not np.isnan(val_loss):
                        print(f"Epoch {epoch:3d}, Train: {avg_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
                    else:
                        print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                current_loss = val_loss if use_validation_for_early_stopping and val_loss is not None and not np.isnan(val_loss) else avg_loss
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
            if validation_data.get("station_ids") is not None:
                dataset_masked["station_ids"] = validation_data["station_ids"]
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            X = dataset["X"]
            if X is None or len(X) == 0:
                return None
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            station_ids = dataset.get("station_ids")
            station_ids_tensor = None
            if station_ids is not None:
                station_ids_tensor = torch.tensor(station_ids, dtype=torch.long, device=device)
                if station_ids_tensor.shape[0] != X_tensor.shape[0]:
                    raise ValueError("station_ids length must match X length")
            amp_enabled = device.type == "cuda"
            with torch.no_grad():
                missing_mask = torch.isnan(X_tensor)
                observation_mask = (~missing_mask).float()
                X_input = torch.nan_to_num(X_tensor, 0)
                if amp_enabled:
                    with torch.autocast(device_type=device.type, enabled=True):
                        predictions = self.forward(X_input, observation_mask, station_ids_tensor)
                    predictions = predictions.float()
                else:
                    predictions = self.forward(X_input, observation_mask, station_ids_tensor)
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
