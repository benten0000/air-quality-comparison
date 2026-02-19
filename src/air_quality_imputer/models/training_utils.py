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
    # Eval/predict loaders are often created repeatedly in loops. Keeping them
    # single-process avoids excessive worker churn and open file descriptors.
    if not shuffle:
        num_workers = 0
        persistent_workers = False

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
