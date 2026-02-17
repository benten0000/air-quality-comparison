from __future__ import annotations

import os
from math import ceil
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def maybe_compile_model(model: nn.Module, config, device: torch.device) -> nn.Module:
    use_torch_compile = bool(config.use_torch_compile)
    if device.type != "cuda" or not use_torch_compile:
        return model
    try:
        compile_mode = str(config.compile_mode)
        compile_dynamic = bool(config.compile_dynamic)
        compiled = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
        return cast(nn.Module, compiled)
    except Exception as exc:
        print(f"torch.compile unavailable, fallback to eager mode: {exc}")
        return model


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _build_allowed_feature_mask(
    n_feat: int,
    device: torch.device,
    never_mask_indices: list[int] | None,
) -> torch.Tensor:
    allowed = torch.ones(n_feat, dtype=torch.bool, device=device)
    if not never_mask_indices:
        return allowed
    idx = torch.as_tensor(never_mask_indices, device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < n_feat)]
    if idx.numel() > 0:
        allowed[idx] = False
    return allowed


def sample_train_mask(
    observed_mask: torch.Tensor,
    config,
    never_mask_indices: list[int] | None = None,
) -> torch.Tensor:
    mode = str(config.train_mask_mode).lower()
    if mode in {"none", "off"}:
        return torch.zeros_like(observed_mask, dtype=torch.bool)
    if mode != "random":
        raise ValueError(f"Unsupported train_mask_mode: {mode}")
    missing_rate = _clamp01(float(config.train_missing_rate))

    observed = observed_mask.bool()
    _, _, n_feat = observed.shape
    allowed_feature_mask = _build_allowed_feature_mask(
        n_feat=n_feat,
        device=observed.device,
        never_mask_indices=never_mask_indices,
    )
    if not allowed_feature_mask.any():
        return torch.zeros_like(observed, dtype=torch.bool)
    maskable_observed = observed & allowed_feature_mask.view(1, 1, n_feat)
    return (torch.rand(observed_mask.shape, device=observed_mask.device) < missing_rate) & maskable_observed


def build_forecast_dataloader(
    *,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    extra_tensors: tuple[torch.Tensor, ...] = (),
    batch_size: int,
    amp_enabled: bool,
    num_workers: int = -1,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    if num_workers < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(8, ceil(cpu_count / 2)))
    num_workers = max(0, int(num_workers))
    pin_memory = bool(pin_memory) and amp_enabled
    persistent_workers = bool(persistent_workers) and num_workers > 0
    prefetch_factor = max(2, int(prefetch_factor))

    dataset = TensorDataset(x_tensor, y_tensor, *extra_tensors)
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
