from __future__ import annotations

import random
import time
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from air_quality_imputer.models.training_utils import (
    build_forecast_dataloader,
    configure_cuda_runtime,
    maybe_compile_model,
)


class ForecastModelMixin:
    @staticmethod
    def _assert_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains NaN/Inf; preprocess data before training/inference")

    @staticmethod
    def _assert_finite_array(array: np.ndarray, name: str) -> None:
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf; preprocess data before training/inference")

    @staticmethod
    def _mask_key(batch: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype, str]:
        return (tuple(batch.shape), batch.dtype, str(batch.device))

    @classmethod
    def _input_mask(
        cls,
        batch: torch.Tensor,
        cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor],
    ) -> torch.Tensor:
        key = cls._mask_key(batch)
        mask = cache.get(key)
        if mask is None:
            mask = torch.ones_like(batch)
            cache[key] = mask
        return mask

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _device_from_cfg(device_cfg: str) -> torch.device:
        if device_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_cfg)

    @staticmethod
    def _grad_scaler(amp: bool):
        amp_mod = getattr(torch, "amp", None)
        grad_scaler_cls = getattr(amp_mod, "GradScaler", None) if amp_mod is not None else None
        if grad_scaler_cls is not None:
            return grad_scaler_cls("cuda", enabled=amp)
        return torch.cuda.amp.GradScaler(enabled=amp)

    @staticmethod
    def _station_payload(
        n_samples: int,
        station_ids: np.ndarray | None,
        station_geo: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sid = (
            np.zeros((n_samples,), dtype=np.int64)
            if station_ids is None
            else np.asarray(station_ids, dtype=np.int64).reshape(-1)
        )
        geo = (
            np.zeros((n_samples, 2), dtype=np.float32)
            if station_geo is None
            else np.asarray(station_geo, dtype=np.float32)
        )
        if sid.shape[0] != n_samples:
            raise ValueError(f"station_ids length {sid.shape[0]} != n_samples {n_samples}")
        if geo.shape[0] != n_samples:
            raise ValueError(f"station_geo length {geo.shape[0]} != n_samples {n_samples}")
        return sid, geo

    @staticmethod
    def _build_loader(
        x: np.ndarray,
        y: np.ndarray,
        station_ids: np.ndarray,
        station_geo: np.ndarray,
        batch_size: int,
        shuffle: bool,
        amp: bool,
        runtime_cfg: DictConfig,
    ):
        return build_forecast_dataloader(
            x_tensor=torch.from_numpy(np.asarray(x, dtype=np.float32)),
            y_tensor=torch.from_numpy(np.asarray(y, dtype=np.float32)),
            extra_tensors=(
                torch.from_numpy(station_ids.astype(np.int64)),
                torch.from_numpy(station_geo.astype(np.float32)),
            ),
            batch_size=int(batch_size),
            amp_enabled=amp,
            num_workers=int(runtime_cfg.dataloader_num_workers),
            prefetch_factor=int(runtime_cfg.dataloader_prefetch_factor),
            persistent_workers=bool(runtime_cfg.dataloader_persistent_workers),
            pin_memory=bool(runtime_cfg.dataloader_pin_memory),
            shuffle=shuffle,
        )

    @staticmethod
    def _as_device_tensors(
        *,
        x: np.ndarray,
        y: np.ndarray,
        station_ids: np.ndarray,
        station_geo: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # One-time H2D transfer keeps the GPU fed; the server GPU has ample VRAM.
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device, non_blocking=False)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device, non_blocking=False)
        sid_t = torch.from_numpy(np.asarray(station_ids, dtype=np.int64)).to(device, non_blocking=False)
        geo_t = torch.from_numpy(np.asarray(station_geo, dtype=np.float32)).to(device, non_blocking=False)
        return x_t, y_t, sid_t, geo_t

    @staticmethod
    def _as_device_inputs(
        *,
        x: np.ndarray,
        station_ids: np.ndarray,
        station_geo: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device, non_blocking=False)
        sid_t = torch.from_numpy(np.asarray(station_ids, dtype=np.int64)).to(device, non_blocking=False)
        geo_t = torch.from_numpy(np.asarray(station_geo, dtype=np.float32)).to(device, non_blocking=False)
        return x_t, sid_t, geo_t

    @staticmethod
    def _eval_loss_tensors(
        model: nn.Module,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        sid: torch.Tensor,
        geo: torch.Tensor,
        target_idx_tensor: torch.Tensor,
        batch_size: int,
        mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor],
    ) -> float:
        criterion = nn.MSELoss()
        total = torch.zeros((), device=x.device, dtype=torch.float32)
        n_batches = 0
        model.eval()
        with torch.inference_mode():
            n = int(x.shape[0])
            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                bx = x[start:end]
                by = y[start:end]
                bsid = sid[start:end]
                bgeo = geo[start:end]
                input_mask = ForecastModelMixin._input_mask(bx, mask_cache)
                pred_all = model(bx, input_mask, station_ids=bsid, station_geo=bgeo)
                pred = pred_all.index_select(1, target_idx_tensor)
                total += criterion(pred, by).detach().float()
                n_batches += 1
        if n_batches == 0:
            return float("inf")
        return float((total / n_batches).item())

    @staticmethod
    def _predict_tensors(
        model: nn.Module,
        *,
        x: torch.Tensor,
        sid: torch.Tensor,
        geo: torch.Tensor,
        target_idx_tensor: torch.Tensor,
        batch_size: int,
        mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor],
    ) -> np.ndarray:
        out: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            n = int(x.shape[0])
            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                bx = x[start:end]
                bsid = sid[start:end]
                bgeo = geo[start:end]
                input_mask = ForecastModelMixin._input_mask(bx, mask_cache)
                pred_all = model(bx, input_mask, station_ids=bsid, station_geo=bgeo)
                pred = pred_all.index_select(1, target_idx_tensor)
                out.append(pred.detach().cpu().numpy())
        if not out:
            return np.empty((0, int(target_idx_tensor.numel())), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    @staticmethod
    def _predict_batches(
        model: nn.Module,
        data_loader,
        device: torch.device,
        target_idx_tensor: torch.Tensor,
        amp: bool,
    ) -> np.ndarray:
        out: list[np.ndarray] = []
        mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
        model.eval()
        with torch.inference_mode():
            for bx, _, sid, geo in data_loader:
                bx = bx.to(device, non_blocking=amp)
                sid = sid.to(device, non_blocking=amp)
                geo = geo.to(device, non_blocking=amp)
                input_mask = ForecastModelMixin._input_mask(bx, mask_cache)
                pred_all = model(bx, input_mask, station_ids=sid, station_geo=geo)
                pred = pred_all.index_select(1, target_idx_tensor)
                out.append(pred.detach().cpu().numpy())
        if not out:
            return np.empty((0, int(target_idx_tensor.numel())), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    @staticmethod
    def _eval_loss(
        model: nn.Module,
        data_loader,
        device: torch.device,
        target_idx_tensor: torch.Tensor,
        amp: bool,
    ) -> float:
        criterion = nn.MSELoss()
        total, n_batches = 0.0, 0
        mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
        model.eval()
        with torch.inference_mode():
            for bx, by, sid, geo in data_loader:
                bx = bx.to(device, non_blocking=amp)
                by = by.to(device, non_blocking=amp)
                sid = sid.to(device, non_blocking=amp)
                geo = geo.to(device, non_blocking=amp)
                input_mask = ForecastModelMixin._input_mask(bx, mask_cache)
                pred_all = model(bx, input_mask, station_ids=sid, station_geo=geo)
                pred = pred_all.index_select(1, target_idx_tensor)
                total += float(criterion(pred, by).item())
                n_batches += 1
        return total / max(n_batches, 1)

    def fit_forecaster(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        target_indices: list[int],
        training_cfg: DictConfig,
        runtime_cfg: DictConfig,
        seed: int,
        train_station_ids: np.ndarray | None = None,
        train_station_geo: np.ndarray | None = None,
        val_station_ids: np.ndarray | None = None,
        val_station_geo: np.ndarray | None = None,
    ) -> float:
        if len(x_train) == 0:
            raise ValueError("Training set is empty")
        self._assert_finite_array(x_train, "train.x")
        self._assert_finite_array(y_train, "train.y")
        if len(x_val) > 0:
            self._assert_finite_array(x_val, "val.x")
            self._assert_finite_array(y_val, "val.y")

        module_self = cast(nn.Module, self)
        self._set_seed(seed)
        device = self._device_from_cfg(str(training_cfg.device))
        configure_cuda_runtime(device)
        module_self.to(device)
        active_model = maybe_compile_model(module_self, runtime_cfg, device)
        amp = device.type == "cuda"
        scaler = self._grad_scaler(amp)

        use_fused = bool(runtime_cfg.optimizer_fused) and amp
        weight_decay = float(runtime_cfg.optimizer_weight_decay)
        lr = float(runtime_cfg.get("optimizer_lr", 1e-3))
        try:
            optimizer = torch.optim.Adam(
                active_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=use_fused,
            )
        except Exception:
            optimizer = torch.optim.Adam(
                active_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=False,
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(runtime_cfg.scheduler_reduce_factor),
            patience=int(runtime_cfg.scheduler_plateau_patience),
            min_lr=float(runtime_cfg.scheduler_min_lr),
        )

        sid_train_np, geo_train_np = self._station_payload(len(x_train), train_station_ids, train_station_geo)
        if len(x_val) > 0:
            sid_val_np, geo_val_np = self._station_payload(len(x_val), val_station_ids, val_station_geo)
        else:
            sid_val_np, geo_val_np = np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

        # Keep training/val tensors on GPU to reduce CPU/dataloader overhead.
        if device.type == "cuda":
            x_train_t, y_train_t, sid_train_t, geo_train_t = self._as_device_tensors(
                x=x_train, y=y_train, station_ids=sid_train_np, station_geo=geo_train_np, device=device
            )
            if len(x_val) > 0:
                x_val_t, y_val_t, sid_val_t, geo_val_t = self._as_device_tensors(
                    x=x_val, y=y_val, station_ids=sid_val_np, station_geo=geo_val_np, device=device
                )
            else:
                x_val_t = y_val_t = sid_val_t = geo_val_t = None
        else:
            train_loader = self._build_loader(
                x_train,
                y_train,
                sid_train_np,
                geo_train_np,
                int(training_cfg.batch_size),
                True,
                amp,
                runtime_cfg,
            )
            if len(x_val) > 0:
                val_loader = self._build_loader(
                    x_val,
                    y_val,
                    sid_val_np,
                    geo_val_np,
                    int(training_cfg.batch_size),
                    False,
                    amp,
                    runtime_cfg,
                )
            else:
                val_loader = None

        criterion = nn.MSELoss()
        target_idx_tensor = torch.as_tensor(target_indices, dtype=torch.long, device=device)
        best_metric = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None
        mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
        start = time.perf_counter()

        for epoch in range(1, int(training_cfg.epochs) + 1):
            active_model.train()
            running = torch.zeros((), device=device, dtype=torch.float32)
            n_batches = 0

            if device.type == "cuda":
                n = int(x_train_t.shape[0])
                bs = int(training_cfg.batch_size)
                if bs > n:
                    bs = n
                perm = torch.randperm(n, device=device)
                # Keep batch shapes stable for torch.compile (avoid last smaller batch).
                remainder = int(perm.numel()) % bs
                if remainder:
                    pad = bs - remainder
                    perm = torch.cat([perm, perm[:pad]], dim=0)
                for start_idx in range(0, int(perm.numel()), bs):
                    idx = perm[start_idx : start_idx + bs]
                    bx = x_train_t.index_select(0, idx)
                    by = y_train_t.index_select(0, idx)
                    sid = sid_train_t.index_select(0, idx)
                    geo = geo_train_t.index_select(0, idx)
                    input_mask = self._input_mask(bx, mask_cache)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        pred_all = active_model(bx, input_mask, station_ids=sid, station_geo=geo)
                        pred = pred_all.index_select(1, target_idx_tensor)
                        loss = criterion(pred, by)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=float(runtime_cfg.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    running += loss.detach().float()
                    n_batches += 1

                train_loss = float((running / max(n_batches, 1)).item())
                if x_val_t is not None:
                    metric = self._eval_loss_tensors(
                        active_model,
                        x=x_val_t,
                        y=y_val_t,
                        sid=sid_val_t,
                        geo=geo_val_t,
                        target_idx_tensor=target_idx_tensor,
                        batch_size=int(training_cfg.batch_size),
                        mask_cache=mask_cache,
                    )
                else:
                    metric = train_loss
            else:
                for bx, by, sid, geo in train_loader:
                    bx = bx.to(device, non_blocking=amp)
                    by = by.to(device, non_blocking=amp)
                    sid = sid.to(device, non_blocking=amp)
                    geo = geo.to(device, non_blocking=amp)

                    input_mask = self._input_mask(bx, mask_cache)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        pred_all = active_model(bx, input_mask, station_ids=sid, station_geo=geo)
                        pred = pred_all.index_select(1, target_idx_tensor)
                        loss = criterion(pred, by)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=float(runtime_cfg.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    running += loss.detach().float()
                    n_batches += 1

                train_loss = float((running / max(n_batches, 1)).item())
                metric = self._eval_loss(active_model, val_loader, device, target_idx_tensor, amp) if val_loader is not None else train_loss

            scheduler.step(metric)

            if epoch == 1 or epoch % int(training_cfg.log_every) == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                    f"val_loss={metric:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if metric < best_metric - float(training_cfg.min_delta):
                best_metric = metric
                best_epoch = epoch
                patience_counter = 0
                # Keep best weights on-device to avoid CPU transfers that stall the GPU.
                best_state = {k: v.detach().clone() for k, v in module_self.state_dict().items()}
            else:
                patience_counter += 1

            if bool(training_cfg.early_stopping) and patience_counter >= int(training_cfg.patience):
                print(f"Early stopping at epoch {epoch} | best_val={best_metric:.6f}")
                break

        if best_state is not None:
            module_self.load_state_dict(best_state)
            print(f"Restored best model from epoch {best_epoch} | best_val={best_metric:.6f}")
        return time.perf_counter() - start

    def predict_forecaster(
        self,
        *,
        x: np.ndarray,
        target_indices: list[int],
        batch_size: int,
        device: torch.device,
        runtime_cfg: DictConfig,
        station_ids: np.ndarray | None = None,
        station_geo: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(x) == 0:
            return np.empty((0, len(target_indices)), dtype=np.float32)
        self._assert_finite_array(x, "predict.x")
        module_self = cast(nn.Module, self)
        module_self.to(device)
        amp = device.type == "cuda"
        sid, geo = self._station_payload(len(x), station_ids, station_geo)
        target_idx_tensor = torch.as_tensor(target_indices, dtype=torch.long, device=device)
        if device.type == "cuda":
            x_t, sid_t, geo_t = self._as_device_inputs(
                x=x,
                station_ids=sid,
                station_geo=geo,
                device=device,
            )
            mask_cache: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
            return self._predict_tensors(
                module_self,
                x=x_t,
                sid=sid_t,
                geo=geo_t,
                target_idx_tensor=target_idx_tensor,
                batch_size=int(batch_size),
                mask_cache=mask_cache,
            )

        y_stub = np.zeros((len(x), len(target_indices)), dtype=np.float32)
        data_loader = self._build_loader(
            x,
            y_stub,
            sid,
            geo,
            batch_size,
            False,
            amp,
            runtime_cfg,
        )
        return self._predict_batches(module_self, data_loader, device, target_idx_tensor, amp)
