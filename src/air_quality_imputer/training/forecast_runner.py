from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from air_quality_imputer.models.training_utils import configure_cuda_runtime
from air_quality_imputer.models.recurrent_forecasters import (
    GRUForecaster,
    HybridLocLSTM,
    HybridLocLSTMConfig,
    LSTMForecaster,
    RecurrentForecasterConfig,
)
from air_quality_imputer.models.transformer_forecaster import TransformerConfig, TransformerForecaster
from air_quality_imputer.tracking import MLflowTracker

ModelKind = Literal["transformer", "lstm", "gru", "hybrid_loclstm"]

SUPPORTED_MODELS: tuple[str, ...] = ("transformer", "lstm", "gru", "hybrid_loclstm")
RUNTIME_DEFAULTS: dict[str, Any] = {
    "optimizer_lr": 1e-3,
    "bias": True,
    "use_torch_compile": True,
    "compile_mode": "reduce-overhead",
    "compile_dynamic": False,
    # AMP dtype selection is used inside ForecastModelMixin.
    # "auto": prefer bf16 on modern NVIDIA GPUs when supported.
    "amp_dtype": "auto",  # "auto" | "fp16" | "bf16"
    # When running on CUDA, optionally preload the full dataset to GPU tensors.
    # Per-station uses an adaptive check and may flip this off.
    "cuda_preload": True,
    "optimizer_fused": True,
    "optimizer_weight_decay": 0.01,
    "scheduler_reduce_factor": 0.5,
    "scheduler_plateau_patience": 5,
    "scheduler_min_lr": 1e-6,
    "grad_clip_norm": 0.5,
    "dataloader_num_workers": -1,
    "dataloader_prefetch_factor": 4,
    "dataloader_persistent_workers": True,
    "dataloader_pin_memory": True,
}


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return out.strip("_").lower() or "na"


def _lower_list(raw: Any, *, default: list[str]) -> list[str]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [raw]
    else:
        values = [str(v) for v in raw]
    out: list[str] = []
    for v in values:
        key = str(v).strip().lower()
        if key and key not in out:
            out.append(key)
    return out


def _cat(arrs: list[np.ndarray], *, empty: np.ndarray) -> np.ndarray:
    return np.concatenate(arrs, axis=0) if arrs else empty


class SplitSet(TypedDict):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class PerStationData(TypedDict):
    scaler: StandardScaler
    splits: SplitSet
    sid_train: np.ndarray
    geo_train: np.ndarray
    sid_val: np.ndarray
    geo_val: np.ndarray
    sid_test: np.ndarray
    geo_test: np.ndarray


class GlobalPrepared(TypedDict):
    stations: list[str]
    scaler: StandardScaler
    x_train: np.ndarray
    y_train: np.ndarray
    sid_train: np.ndarray
    geo_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    sid_val: np.ndarray
    geo_val: np.ndarray
    x_test: np.ndarray
    y_test_scaled: np.ndarray
    sid_test: np.ndarray
    geo_test: np.ndarray
    test_counts: list[int]


class PerStationCudaData(TypedDict):
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    sid_train: torch.Tensor
    geo_train: torch.Tensor
    sid_val: torch.Tensor
    geo_val: torch.Tensor
    x_test: torch.Tensor
    sid_test: torch.Tensor
    geo_test: torch.Tensor


class WindowIndex(TypedDict):
    start_idx: np.ndarray
    target_idx: np.ndarray
    start_ts: np.ndarray
    target_ts: np.ndarray


def to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        container = OmegaConf.to_container(value, resolve=True)
        if isinstance(container, Mapping):
            return {str(key): val for key, val in container.items()}
        return {}
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    return {}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def model_runtime_cfg(cfg: DictConfig, model_kind: ModelKind) -> DictConfig:
    runtime_raw = OmegaConf.select(cfg, f"models.{model_kind}.runtime", default={})
    runtime = {**RUNTIME_DEFAULTS, **to_plain_dict(runtime_raw)}
    return OmegaConf.create(runtime)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_cfg(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_eps: float = 1e-6,
    mape_min_abs_target: float = 0.0,
) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    err = y_pred - y_true
    mse = float(np.mean(err**2))

    abs_true = np.abs(y_true)
    valid_mape = abs_true > float(mape_eps)
    if bool(valid_mape.any()):
        denom_used = np.maximum(abs_true[valid_mape], float(max(mape_min_abs_target, 0.0)))
        mape = float(np.mean(np.abs(err[valid_mape]) / denom_used) * 100.0)
    else:
        mape = float("nan")

    denom_smape = abs_true + np.abs(y_pred)
    valid_smape = denom_smape > float(mape_eps)
    smape = (
        float(np.mean((2.0 * np.abs(err[valid_smape])) / denom_smape[valid_smape]) * 100.0)
        if bool(valid_smape.any())
        else float("nan")
    )

    wape_denom = float(np.sum(abs_true))
    wape = float(np.sum(np.abs(err)) / wape_denom * 100.0) if wape_denom > float(mape_eps) else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "evs": _explained_variance_uniform(y_true, y_pred),
        "r2": _r2_uniform(y_true, y_pred),
    }


def _as_2d_float64(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _explained_variance_uniform(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_2d_float64(y_true)
    yp = _as_2d_float64(y_pred)
    var_y = np.var(yt, axis=0)
    var_err = np.var(yt - yp, axis=0)
    out = np.empty_like(var_y, dtype=np.float64)
    nz = var_y > 0.0
    out[nz] = 1.0 - (var_err[nz] / var_y[nz])
    if bool((~nz).any()):
        out[~nz] = np.where(var_err[~nz] <= np.finfo(np.float64).eps, 1.0, 0.0)
    return float(np.mean(out))


def _r2_uniform(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_2d_float64(y_true)
    yp = _as_2d_float64(y_pred)
    diff = yt - yp
    ss_res = np.sum(diff * diff, axis=0)
    centered = yt - np.mean(yt, axis=0)
    ss_tot = np.sum(centered * centered, axis=0)
    out = np.empty_like(ss_tot, dtype=np.float64)
    nz = ss_tot > 0.0
    out[nz] = 1.0 - (ss_res[nz] / ss_tot[nz])
    if bool((~nz).any()):
        out[~nz] = np.where(ss_res[~nz] <= np.finfo(np.float64).eps, 1.0, 0.0)
    return float(np.mean(out))


def load_frames(
    data_dir: Path,
    sources: list[str],
    datetime_col: str,
    features: list[str],
) -> dict[str, pd.DataFrame]:
    source_names = [Path(str(name).strip()).stem for name in sources if str(name).strip()]
    if not source_names:
        raise ValueError("data.stations cannot be empty")

    non_station_features = [feature for feature in features if feature != "station"]
    required_base = [datetime_col] + non_station_features
    out: dict[str, pd.DataFrame] = {}
    read_cols = {datetime_col, *non_station_features, "station"}
    dtype_map: dict[str, Any] = {col: "float32" for col in non_station_features}

    for source_name in source_names:
        csv_path = data_dir / f"{source_name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing data source: {csv_path}")

        try:
            frame = pd.read_csv(
                csv_path,
                usecols=lambda col: col in read_cols,
                parse_dates=[datetime_col],
                dtype=cast(Any, dtype_map) if dtype_map else None,
                low_memory=False,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid schema or numeric values in {csv_path}: {exc}") from exc
        if frame.empty:
            continue
        frame = frame.copy()
        if "station" in frame.columns:
            frame["_station_label"] = frame["station"].astype(str).str.strip()
        else:
            frame["_station_label"] = source_name

        missing_cols = [column for column in required_base if column not in frame.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

        work = frame[[datetime_col] + non_station_features + ["_station_label"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(work[datetime_col].dtype):
            work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
        work = work.dropna(subset=[datetime_col]).copy()
        work["_station_label"] = work["_station_label"].astype(str).str.strip()
        work = work[work["_station_label"] != ""]
        if work.empty:
            continue

        if non_station_features:
            if work[non_station_features].isna().any().any():
                raise ValueError(f"Invalid numeric values in {csv_path}")

        for station_name, station_df in work.groupby("_station_label", sort=True):
            key = str(station_name)
            if key in out:
                continue
            cleaned = station_df[[datetime_col] + non_station_features].sort_values(datetime_col).reset_index(drop=True)
            if cleaned.empty:
                continue
            out[key] = cleaned

    if not out:
        raise ValueError(f"No usable rows found for data.stations={source_names}")

    if "station" in features:
        station_code_map = {name: idx for idx, name in enumerate(sorted(out))}
        for station_name in list(out.keys()):
            frame = out[station_name].copy()
            frame["station"] = np.float32(station_code_map[station_name])
            out[station_name] = frame[[datetime_col] + features]
    else:
        for station_name in list(out.keys()):
            out[station_name] = out[station_name][[datetime_col] + non_station_features]
    return out


def resolve_model_kinds(cfg: DictConfig) -> list[ModelKind]:
    out: list[ModelKind] = []
    for key in _lower_list(OmegaConf.select(cfg, "experiment.common.models", default=None), default=["transformer"]):
        if key not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported experiment model '{key}'. Supported: {list(SUPPORTED_MODELS)}")
        out.append(cast(ModelKind, key))
    if not out:
        raise ValueError("experiment.models resolved to an empty list")
    return out


def load_station_geo_vectors(cfg: DictConfig, stations: list[str]) -> dict[str, np.ndarray]:
    default = {st: np.zeros((2,), dtype=np.float32) for st in stations}
    geo_path_raw = OmegaConf.select(cfg, "data.station_geo_path", default=None)
    if geo_path_raw is None:
        return default
    geo_path = Path(str(geo_path_raw))
    if not geo_path.exists():
        print(f"[WARN] station_geo_path does not exist: {geo_path}. HybridLocLSTM will use zero geo vectors.")
        return default

    payload = json.loads(geo_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = list(payload.values())
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unsupported geo JSON format in {geo_path}")

    parsed: dict[str, np.ndarray] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = item.get("station") or item.get("serial") or item.get("code") or item.get("name")
        lat = item.get("lat", item.get("latitude"))
        lon = item.get("lon", item.get("longitude"))
        if key is None or lat is None or lon is None:
            continue
        key_str = str(key)
        parsed[key_str] = np.array([float(lat) / 90.0, float(lon) / 180.0], dtype=np.float32)

    missing = [st for st in stations if st not in parsed]
    if missing:
        print(f"[WARN] Missing geo entries for {len(missing)} stations in {geo_path}; using zeros for those stations.")
    return {st: parsed.get(st, default[st]) for st in stations}


def build_window_index(
    times: np.ndarray,
    history_length: int,
    horizon: int,
    step_size: int,
) -> WindowIndex:
    t = int(times.shape[0])
    if t < int(history_length) + int(horizon):
        return {
            "start_idx": np.empty((0,), dtype=np.int64),
            "target_idx": np.empty((0,), dtype=np.int64),
            "start_ts": np.empty((0,), dtype="datetime64[ns]"),
            "target_ts": np.empty((0,), dtype="datetime64[ns]"),
        }
    end_idx = np.arange(int(history_length), t - int(horizon) + 1, int(step_size), dtype=np.int64)
    start_idx = end_idx - int(history_length)
    target_idx = end_idx + int(horizon) - 1
    return {
        "start_idx": start_idx,
        "target_idx": target_idx,
        "start_ts": times[start_idx].astype("datetime64[ns]", copy=False),
        "target_ts": times[target_idx].astype("datetime64[ns]", copy=False),
    }


def _scale_float32(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    mean32, scale32 = _scaler_stats_float32(scaler)
    return (values - mean32) / scale32


def _scaler_stats_float32(scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    mean = scaler.mean_
    scale = scaler.scale_
    if mean is None or scale is None:
        raise ValueError("Scaler must be fitted before extracting stats")
    mean32 = getattr(scaler, "_aqi_mean32", None)
    scale32 = getattr(scaler, "_aqi_scale32", None)
    if isinstance(mean32, np.ndarray) and isinstance(scale32, np.ndarray):
        return mean32, scale32
    mean32 = np.asarray(mean, dtype=np.float32)
    scale32 = np.asarray(scale, dtype=np.float32)
    setattr(scaler, "_aqi_mean32", mean32)
    setattr(scaler, "_aqi_scale32", scale32)
    return mean32, scale32


def build_split(
    frame: pd.DataFrame,
    datetime_col: str,
    features: list[str],
    target_indices: list[int],
    scaler: StandardScaler,
    history_length: int,
    horizon: int,
    step_size: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    window_index: WindowIndex | None = None,
) -> dict[str, np.ndarray]:
    # load_frames already coerces datetime_col; avoid repeated pd.to_datetime overhead.
    times = frame[datetime_col].to_numpy(dtype="datetime64[ns]")
    scaled = _scale_float32(frame[features].to_numpy(dtype=np.float32), scaler)
    win = build_window_index(times, history_length, horizon, step_size) if window_index is None else window_index
    start_idx = win["start_idx"]
    target_idx = win["target_idx"]
    if len(start_idx) == 0:
        return {
            "x_train": np.empty((0, history_length, len(features)), dtype=np.float32),
            "y_train": np.empty((0, len(target_indices)), dtype=np.float32),
            "x_val": np.empty((0, history_length, len(features)), dtype=np.float32),
            "y_val": np.empty((0, len(target_indices)), dtype=np.float32),
            "x_test": np.empty((0, history_length, len(features)), dtype=np.float32),
            "y_test": np.empty((0, len(target_indices)), dtype=np.float32),
        }

    x_view = np.lib.stride_tricks.sliding_window_view(scaled, int(history_length), axis=0).swapaxes(1, 2)
    y_base = scaled[target_idx][:, target_indices].astype(np.float32, copy=False)

    train_start64 = np.datetime64(train_start)
    train_end64 = np.datetime64(train_end)
    val_end64 = np.datetime64(val_end)
    eval_start64 = np.datetime64(eval_start)
    train_mask = (win["start_ts"] >= train_start64) & (win["target_ts"] < train_end64)
    val_mask = (win["start_ts"] >= train_end64) & (win["target_ts"] < val_end64)
    eval_mask = win["target_ts"] >= eval_start64

    return {
        "x_train": x_view[start_idx[train_mask]].astype(np.float32, copy=False),
        "y_train": y_base[train_mask].astype(np.float32, copy=False),
        "x_val": x_view[start_idx[val_mask]].astype(np.float32, copy=False),
        "y_val": y_base[val_mask].astype(np.float32, copy=False),
        "x_test": x_view[start_idx[eval_mask]].astype(np.float32, copy=False),
        "y_test": y_base[eval_mask].astype(np.float32, copy=False),
    }


def prepare_global_data(
    *,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    cfg: DictConfig,
    station_to_id: dict[str, int],
    station_geo_map: dict[str, np.ndarray],
    window_cache: dict[str, WindowIndex] | None = None,
) -> GlobalPrepared:
    features = list(cfg.data.features)
    target_features = list(cfg.data.target_features)
    target_idx = [features.index(f) for f in target_features]
    history_length = int(cfg.experiment.common.history_length)
    datetime_col = str(cfg.data.datetime_col)
    horizon = int(cfg.experiment.common.forecast_horizon)
    step_size = int(cfg.experiment.common.step_size)
    stations = sorted(frames)

    scaler = fit_global_scaler(frames, datetime_col, features, train_starts, train_end)
    splits: dict[str, SplitSet] = {}
    for st in stations:
        splits[st] = cast(
            SplitSet,
            build_split(
                frames[st],
                datetime_col,
                features,
                target_idx,
                scaler,
                history_length,
                horizon,
                step_size,
                train_starts[st],
                train_end,
                val_end,
                eval_start,
                window_index=(window_cache.get(st) if window_cache is not None else None),
            ),
        )
        if len(splits[st]["x_train"]) == 0:
            raise ValueError(f"global: station {st} has no training samples")
        if len(splits[st]["x_test"]) == 0:
            raise ValueError(f"global: station {st} has no evaluation samples")

    def sid_geo(station: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((n,), station_to_id[station], dtype=np.int64),
            np.repeat(station_geo_map[station][None, :], n, axis=0).astype(np.float32, copy=False),
        )

    def payload_for(stations_in: list[str], key: str) -> tuple[np.ndarray, np.ndarray]:
        sid_parts: list[np.ndarray] = []
        geo_parts: list[np.ndarray] = []
        for st in stations_in:
            sid_arr, geo_arr = sid_geo(st, len(splits[st][key]))
            sid_parts.append(sid_arr)
            geo_parts.append(geo_arr)
        sid = _cat(sid_parts, empty=np.empty((0,), dtype=np.int64))
        geo = _cat(geo_parts, empty=np.empty((0, 2), dtype=np.float32))
        return sid, geo

    x_train = np.concatenate([splits[st]["x_train"] for st in stations], axis=0)
    y_train = np.concatenate([splits[st]["y_train"] for st in stations], axis=0)
    sid_train, geo_train = payload_for(stations, "x_train")

    val_stations = [st for st in stations if len(splits[st]["x_val"]) > 0]
    x_val = _cat(
        [splits[st]["x_val"] for st in val_stations],
        empty=np.empty((0, history_length, len(features)), dtype=np.float32),
    )
    y_val = _cat(
        [splits[st]["y_val"] for st in val_stations],
        empty=np.empty((0, len(target_features)), dtype=np.float32),
    )
    sid_val, geo_val = payload_for(val_stations, "x_val") if val_stations else (
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.float32),
    )

    x_test_all = _cat(
        [splits[st]["x_test"] for st in stations],
        empty=np.empty((0, history_length, len(features)), dtype=np.float32),
    )
    y_test_scaled_all = _cat(
        [splits[st]["y_test"] for st in stations],
        empty=np.empty((0, len(target_features)), dtype=np.float32),
    )
    sid_test, geo_test = payload_for(stations, "x_test")
    test_counts = [int(len(splits[st]["x_test"])) for st in stations]

    return {
        "stations": stations,
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "sid_train": sid_train,
        "geo_train": geo_train,
        "x_val": x_val,
        "y_val": y_val,
        "sid_val": sid_val,
        "geo_val": geo_val,
        "x_test": x_test_all,
        "y_test_scaled": y_test_scaled_all,
        "sid_test": sid_test,
        "geo_test": geo_test,
        "test_counts": test_counts,
    }


def prepare_per_station_data(
    *,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    cfg: DictConfig,
    station_geo_map: dict[str, np.ndarray],
    window_cache: dict[str, WindowIndex] | None = None,
) -> dict[str, PerStationData]:
    features = list(cfg.data.features)
    target_features = list(cfg.data.target_features)
    target_idx = [features.index(f) for f in target_features]
    history_length = int(cfg.experiment.common.history_length)
    datetime_col = str(cfg.data.datetime_col)
    horizon = int(cfg.experiment.common.forecast_horizon)
    step_size = int(cfg.experiment.common.step_size)
    stations = sorted(frames)

    out: dict[str, PerStationData] = {}
    for st in stations:
        frame = frames[st]
        train_mask = (frame[datetime_col] >= train_starts[st]) & (frame[datetime_col] < train_end)
        train_arr = frame.loc[train_mask, features].to_numpy(dtype=np.float32)
        if train_arr.size == 0:
            raise ValueError(f"per_station: station {st} has no training rows for scaler")
        scaler = StandardScaler()
        scaler.fit(train_arr)

        splits = cast(
            SplitSet,
            build_split(
                frame,
                datetime_col,
                features,
                target_idx,
                scaler,
                history_length,
                horizon,
                step_size,
                train_starts[st],
                train_end,
                val_end,
                eval_start,
                window_index=(window_cache.get(st) if window_cache is not None else None),
            ),
        )
        if len(splits["x_train"]) == 0:
            raise ValueError(f"per_station: station {st} has no training samples")
        if len(splits["x_test"]) == 0:
            raise ValueError(f"per_station: station {st} has no evaluation samples")

        sid_train = np.zeros((len(splits["x_train"]),), dtype=np.int64)
        geo_train = np.repeat(station_geo_map[st][None, :], len(splits["x_train"]), axis=0).astype(np.float32, copy=False)
        if len(splits["x_val"]) > 0:
            sid_val = np.zeros((len(splits["x_val"]),), dtype=np.int64)
            geo_val = np.repeat(station_geo_map[st][None, :], len(splits["x_val"]), axis=0).astype(np.float32, copy=False)
        else:
            sid_val = np.empty((0,), dtype=np.int64)
            geo_val = np.empty((0, 2), dtype=np.float32)

        sid_test = np.zeros((len(splits["x_test"]),), dtype=np.int64)
        geo_test = np.repeat(station_geo_map[st][None, :], len(splits["x_test"]), axis=0).astype(np.float32, copy=False)

        out[st] = {
            "scaler": scaler,
            "splits": splits,
            "sid_train": sid_train,
            "geo_train": geo_train,
            "sid_val": sid_val,
            "geo_val": geo_val,
            "sid_test": sid_test,
            "geo_test": geo_test,
        }
    return out


def inverse_scale_targets(y_scaled: np.ndarray, scaler: StandardScaler, target_indices: list[int]) -> np.ndarray:
    mean32, scale32 = _scaler_stats_float32(scaler)
    return y_scaled * scale32[target_indices] + mean32[target_indices]


def build_model(
    cfg: DictConfig,
    model_kind: ModelKind,
    runtime_cfg: DictConfig,
    n_features: int,
    n_stations: int,
    station_feature_index: int,
) -> nn.Module:
    transformer = cfg.models.transformer.params
    ff_mult = OmegaConf.select(transformer, "ff_multiplier", default=None)
    d_ffn_raw = OmegaConf.select(transformer, "d_ffn", default=None)
    d_k_raw = OmegaConf.select(transformer, "d_k", default=None)
    d_v_raw = OmegaConf.select(transformer, "d_v", default=None)
    d_model = int(transformer.d_model)
    d_ffn = int(d_ffn_raw) if d_ffn_raw is not None else (int(round(d_model * float(ff_mult))) if ff_mult is not None else None)

    if model_kind == "transformer":
        return TransformerForecaster(
            TransformerConfig(
                block_size=int(cfg.experiment.common.history_length),
                n_features=n_features,
                d_model=d_model,
                n_head=int(transformer.n_head),
                n_layer=int(transformer.n_layer),
                d_ffn=d_ffn,
                d_k=(int(d_k_raw) if d_k_raw is not None else None),
                d_v=(int(d_v_raw) if d_v_raw is not None else None),
                dropout=float(transformer.dropout),
                bias=bool(runtime_cfg.bias),
                norm_eps=float(transformer.norm_eps),
            )
        )

    rnn = cfg.models[model_kind].params
    rnn_cfg = RecurrentForecasterConfig(
        n_features=n_features,
        hidden_size=int(OmegaConf.select(rnn, "hidden_size", default=96)),
        num_layers=int(OmegaConf.select(rnn, "num_layers", default=2)),
        dropout=float(OmegaConf.select(rnn, "dropout", default=0.1)),
        bias=bool(runtime_cfg.bias),
    )
    if model_kind == "lstm":
        return LSTMForecaster(rnn_cfg)
    if model_kind == "gru":
        return GRUForecaster(rnn_cfg)
    if model_kind == "hybrid_loclstm":
        return HybridLocLSTM(
            HybridLocLSTMConfig(
                n_features=n_features,
                hidden_size=int(OmegaConf.select(rnn, "hidden_size", default=96)),
                num_layers=int(OmegaConf.select(rnn, "num_layers", default=2)),
                dropout=float(OmegaConf.select(rnn, "dropout", default=0.1)),
                bias=bool(runtime_cfg.bias),
                num_stations=max(int(n_stations), 1),
                embed_dim=int(OmegaConf.select(rnn, "embed_dim", default=16)),
                geo_dim=int(OmegaConf.select(rnn, "geo_dim", default=2)),
                station_feature_index=int(OmegaConf.select(rnn, "station_feature_index", default=station_feature_index)),
            )
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def fit_model(model: nn.Module, **kwargs: Any) -> float:
    fit_fn = getattr(model, "fit_forecaster", None)
    if fit_fn is None:
        raise TypeError(f"Model {type(model).__name__} does not implement fit_forecaster")
    return float(fit_fn(**kwargs))


def predict_model(model: nn.Module, **kwargs: Any) -> np.ndarray:
    predict_fn = getattr(model, "predict_forecaster", None)
    if predict_fn is None:
        raise TypeError(f"Model {type(model).__name__} does not implement predict_forecaster")
    return cast(np.ndarray, predict_fn(**kwargs))


def save_ckpt(
    path: Path,
    model: nn.Module,
    model_kind: ModelKind,
    features: list[str],
    target_features: list[str],
    scaler: StandardScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_obj = getattr(model, "config", None)
    model_cfg: dict[str, Any] | None = None
    if cfg_obj is not None and is_dataclass(cfg_obj) and not isinstance(cfg_obj, type):
        model_cfg = cast(dict[str, Any], asdict(cfg_obj))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": model_kind,
            "model_config": model_cfg,
            "features": features,
            "target_features": target_features,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        path,
    )


def fit_global_scaler(
    frames: dict[str, pd.DataFrame],
    datetime_col: str,
    features: list[str],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
) -> StandardScaler:
    sc = StandardScaler()
    fitted = False
    for st, frame in frames.items():
        m = (frame[datetime_col] >= train_starts[st]) & (frame[datetime_col] < train_end)
        arr = frame.loc[m, features].to_numpy(dtype=np.float32)
        if arr.size:
            sc.partial_fit(arr)
            fitted = True
    if not fitted:
        raise ValueError("No training rows available for global scaler")
    return sc


def run_approach(
    source_name: str,
    model_kind: ModelKind,
    approach: str,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    station_to_id: dict[str, int],
    station_geo_map: dict[str, np.ndarray],
    cfg: DictConfig,
    training_cfg: DictConfig,
    fold_id: int,
    tracker: MLflowTracker | None = None,
    model_artifact_path: str = "model",
    prepared: GlobalPrepared | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    paths = cfg.paths
    features = list(cfg.data.features)
    target_features = list(cfg.data.target_features)
    target_idx = [features.index(f) for f in target_features]
    station_feature_index = features.index("station") if "station" in features else -1
    history_length = int(cfg.experiment.common.history_length)
    mape_eps = float(OmegaConf.select(cfg, "metrics.mape_eps", default=1e-6))
    mape_min_abs_target = float(OmegaConf.select(cfg, "metrics.mape_min_abs_target", default=0.0))
    if prepared is None:
        prepared = prepare_global_data(
            frames=frames,
            train_starts=train_starts,
            train_end=train_end,
            val_end=val_end,
            eval_start=eval_start,
            cfg=cfg,
            station_to_id=station_to_id,
            station_geo_map=station_geo_map,
        )

    stations = prepared["stations"]
    scaler = prepared["scaler"]
    x_train = prepared["x_train"]
    y_train = prepared["y_train"]
    sid_train = prepared["sid_train"]
    geo_train = prepared["geo_train"]
    x_val = prepared["x_val"]
    y_val = prepared["y_val"]
    sid_val = prepared["sid_val"]
    geo_val = prepared["geo_val"]
    x_test_all = prepared["x_test"]
    y_test_scaled_all = prepared["y_test_scaled"]
    sid_test_all = prepared["sid_test"]
    geo_test_all = prepared["geo_test"]
    test_counts = prepared["test_counts"]

    dev = device_from_cfg(str(training_cfg.device))
    runtime_cfg = model_runtime_cfg(cfg, model_kind)
    # cuDNN RNN kernels are already highly optimized; torch.compile overhead often does not pay off.
    if model_kind in ("lstm", "gru", "hybrid_loclstm"):
        runtime_cfg.use_torch_compile = False

    model = build_model(
        cfg,
        model_kind,
        runtime_cfg,
        len(features),
        n_stations=len(stations),
        station_feature_index=station_feature_index,
    )
    total_time = fit_model(
        model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        target_indices=target_idx,
        training_cfg=training_cfg,
        runtime_cfg=runtime_cfg,
        seed=int(cfg.seed) + fold_id * 1000 + 1,
        train_station_ids=sid_train,
        train_station_geo=geo_train,
        val_station_ids=sid_val,
        val_station_geo=geo_val,
    )
    if tracker is not None:
        tracker.log_torch_model(model, artifact_path=model_artifact_path)

    station_rows: list[dict[str, float | int | str]] = []

    # Predict all stations in one go to reduce Python/DataLoader overhead and keep GPU utilization high.
    pred_scaled_all = predict_model(
        model,
        x=x_test_all,
        target_indices=target_idx,
        batch_size=int(training_cfg.batch_size),
        device=dev,
        runtime_cfg=runtime_cfg,
        station_ids=sid_test_all,
        station_geo=geo_test_all,
    )

    # Global approach uses a single scaler for all stations; inverse-scale in one pass.
    y_true_all = inverse_scale_targets(y_test_scaled_all, scaler, target_idx)
    y_pred_all = inverse_scale_targets(pred_scaled_all, scaler, target_idx)

    offset = 0
    for st, n in zip(stations, test_counts):
        y_true = y_true_all[offset : offset + n]
        y_pred = y_pred_all[offset : offset + n]
        offset += n
        station_rows.append(
            {
                "approach": approach,
                "model": model_kind,
                "fold": fold_id,
                "station": st,
                "n_eval_samples": int(len(y_true)),
                **metrics(y_true, y_pred, mape_eps=mape_eps, mape_min_abs_target=mape_min_abs_target),
            }
        )

    if bool(cfg.output.save_models):
        save_ckpt(
            Path(paths.models_dir) / source_name / f"fold_{fold_id:02d}" / f"{approach}.pt",
            model,
            model_kind,
            features,
            target_features,
            scaler,
        )

    all_true_arr = y_true_all
    all_pred_arr = y_pred_all
    return (
        {
            "approach": approach,
            "model": model_kind,
            "fold": fold_id,
            "n_stations": len(stations),
            "n_eval_samples": int(len(all_true_arr)),
            "train_time_sec": float(total_time),
            **metrics(
                all_true_arr,
                all_pred_arr,
                mape_eps=mape_eps,
                mape_min_abs_target=mape_min_abs_target,
            ),
        },
        station_rows,
    )


def run_approach_per_station(
    source_name: str,
    model_kind: ModelKind,
    approach: str,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    station_geo_map: dict[str, np.ndarray],
    cfg: DictConfig,
    training_cfg: DictConfig,
    fold_id: int,
    tracker: MLflowTracker | None = None,
    model_artifact_path_prefix: str = "model",
    prepared: dict[str, PerStationData] | None = None,
    log_station_models: bool = False,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    paths = cfg.paths
    features = list(cfg.data.features)
    target_features = list(cfg.data.target_features)
    target_idx = [features.index(f) for f in target_features]
    station_feature_index = features.index("station") if "station" in features else -1
    history_length = int(cfg.experiment.common.history_length)
    mape_eps = float(OmegaConf.select(cfg, "metrics.mape_eps", default=1e-6))
    mape_min_abs_target = float(OmegaConf.select(cfg, "metrics.mape_min_abs_target", default=0.0))
    stations = sorted(frames)
    dev = device_from_cfg(str(training_cfg.device))

    runtime_cfg = model_runtime_cfg(cfg, model_kind)
    # Per-station baseline runs many short trainings in sequence.
    # Force single-process dataloading to avoid intermittent multiprocessing/pin-memory crashes.
    runtime_cfg_per_station = OmegaConf.create(to_plain_dict(runtime_cfg))
    runtime_cfg_per_station.dataloader_num_workers = 0
    runtime_cfg_per_station.dataloader_persistent_workers = False
    runtime_cfg_per_station.dataloader_pin_memory = False
    # Speed-first: collapse per-station epochs to ~1 step/epoch when feasible.
    runtime_cfg_per_station.batch_mode = "full"
    # For many short per-station RNN fits, compile startup cost often dominates.
    if model_kind in ("lstm", "gru", "hybrid_loclstm"):
        runtime_cfg_per_station.use_torch_compile = False
    runtime_cfg_per_station.eval_every = int(OmegaConf.select(runtime_cfg_per_station, "eval_every", default=5))

    if prepared is None:
        prepared = prepare_per_station_data(
            frames=frames,
            train_starts=train_starts,
            train_end=train_end,
            val_end=val_end,
            eval_start=eval_start,
            cfg=cfg,
            station_geo_map=station_geo_map,
        )
    station_rows: list[dict[str, float | int | str]] = []
    all_true, all_pred = [], []
    total_time = 0.0

    per_station = prepared
    per_station_cuda: dict[str, PerStationCudaData] | None = None
    if dev.type == "cuda":
        # Adaptive VRAM check: if preloading all per-station tensors would consume too much
        # free VRAM, fall back to streaming via DataLoader + pinned memory.
        free_b, _total_b = torch.cuda.mem_get_info(dev)
        est_b = 0
        for st in stations:
            station_pack = per_station[st]
            splits = station_pack["splits"]
            est_b += int(splits["x_train"].nbytes)
            est_b += int(splits["y_train"].nbytes)
            est_b += int(splits["x_val"].nbytes)
            est_b += int(splits["y_val"].nbytes)
            est_b += int(splits["x_test"].nbytes)
            est_b += int(station_pack["sid_train"].nbytes)
            est_b += int(station_pack["geo_train"].nbytes)
            est_b += int(station_pack["sid_val"].nbytes)
            est_b += int(station_pack["geo_val"].nbytes)
            est_b += int(station_pack["sid_test"].nbytes)
            est_b += int(station_pack["geo_test"].nbytes)

        preload_ok = est_b <= int(float(free_b) * 0.60)
        if preload_ok and bool(OmegaConf.select(runtime_cfg_per_station, "cuda_preload", default=True)):
            per_station_cuda = {}
            for st in stations:
                station_pack = per_station[st]
                splits = station_pack["splits"]
                per_station_cuda[st] = {
                    "x_train": torch.from_numpy(splits["x_train"]).to(dev, non_blocking=False),
                    "y_train": torch.from_numpy(splits["y_train"]).to(dev, non_blocking=False),
                    "x_val": torch.from_numpy(splits["x_val"]).to(dev, non_blocking=False),
                    "y_val": torch.from_numpy(splits["y_val"]).to(dev, non_blocking=False),
                    "sid_train": torch.from_numpy(station_pack["sid_train"]).to(dev, non_blocking=False),
                    "geo_train": torch.from_numpy(station_pack["geo_train"]).to(dev, non_blocking=False),
                    "sid_val": torch.from_numpy(station_pack["sid_val"]).to(dev, non_blocking=False),
                    "geo_val": torch.from_numpy(station_pack["geo_val"]).to(dev, non_blocking=False),
                    "x_test": torch.from_numpy(splits["x_test"]).to(dev, non_blocking=False),
                    "sid_test": torch.from_numpy(station_pack["sid_test"]).to(dev, non_blocking=False),
                    "geo_test": torch.from_numpy(station_pack["geo_test"]).to(dev, non_blocking=False),
                }
        else:
            # Force streaming path in ForecastModelMixin on CUDA.
            runtime_cfg_per_station.cuda_preload = False
            runtime_cfg_per_station.dataloader_pin_memory = True

    # Reuse a single model instance across stations (reset weights) to avoid rebuild/compile overhead.
    model = build_model(
        cfg,
        model_kind,
        runtime_cfg_per_station,
        len(features),
        n_stations=1,
        station_feature_index=station_feature_index,
    )
    # Move once so resets are device-to-device copies (avoids repeated CPU->GPU transfers per station).
    model.to(dev)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for idx, st in enumerate(stations):
        model.load_state_dict(init_state)
        scaler = per_station[st]["scaler"]
        splits = per_station[st]["splits"]
        if per_station_cuda is not None:
            cuda_pack = per_station_cuda[st]
            total_time += fit_model(
                model,
                x_train=cuda_pack["x_train"],
                y_train=cuda_pack["y_train"],
                x_val=cuda_pack["x_val"],
                y_val=cuda_pack["y_val"],
                target_indices=target_idx,
                training_cfg=training_cfg,
                runtime_cfg=runtime_cfg_per_station,
                seed=int(cfg.seed) + fold_id * 1000 + idx + 1,
                train_station_ids=cuda_pack["sid_train"],
                train_station_geo=cuda_pack["geo_train"],
                val_station_ids=cuda_pack["sid_val"],
                val_station_geo=cuda_pack["geo_val"],
            )
        else:
            sid_train = per_station[st]["sid_train"]
            geo_train = per_station[st]["geo_train"]
            sid_val = per_station[st]["sid_val"]
            geo_val = per_station[st]["geo_val"]
            total_time += fit_model(
                model,
                x_train=splits["x_train"],
                y_train=splits["y_train"],
                x_val=splits["x_val"],
                y_val=splits["y_val"],
                target_indices=target_idx,
                training_cfg=training_cfg,
                runtime_cfg=runtime_cfg_per_station,
                seed=int(cfg.seed) + fold_id * 1000 + idx + 1,
                train_station_ids=sid_train,
                train_station_geo=geo_train,
                val_station_ids=sid_val,
                val_station_geo=geo_val,
            )
        if tracker is not None and log_station_models:
            tracker.log_torch_model(model, artifact_path=f"{model_artifact_path_prefix}__{_slug(st)}")

        if per_station_cuda is not None:
            cuda_pack = per_station_cuda[st]
            pred_scaled = predict_model(
                model,
                x=cuda_pack["x_test"],
                target_indices=target_idx,
                batch_size=int(training_cfg.batch_size),
                device=dev,
                runtime_cfg=runtime_cfg_per_station,
                station_ids=cuda_pack["sid_test"],
                station_geo=cuda_pack["geo_test"],
            )
        else:
            sid_test = per_station[st]["sid_test"]
            geo_test = per_station[st]["geo_test"]
            pred_scaled = predict_model(
                model,
                x=splits["x_test"],
                target_indices=target_idx,
                batch_size=int(training_cfg.batch_size),
                device=dev,
                runtime_cfg=runtime_cfg_per_station,
                station_ids=sid_test,
                station_geo=geo_test,
            )
        y_true = inverse_scale_targets(splits["y_test"], scaler, target_idx)
        y_pred = inverse_scale_targets(pred_scaled, scaler, target_idx)
        station_rows.append(
            {
                "approach": approach,
                "model": model_kind,
                "fold": fold_id,
                "station": st,
                "n_eval_samples": int(len(y_true)),
                **metrics(y_true, y_pred, mape_eps=mape_eps, mape_min_abs_target=mape_min_abs_target),
            }
        )
        all_true.append(y_true)
        all_pred.append(y_pred)

        if bool(cfg.output.save_models):
            save_ckpt(
                Path(paths.models_dir) / source_name / f"fold_{fold_id:02d}" / f"{approach}_{st}.pt",
                model,
                model_kind,
                features,
                target_features,
                scaler,
            )

    all_true_arr = np.concatenate(all_true, axis=0)
    all_pred_arr = np.concatenate(all_pred, axis=0)
    return (
        {
            "approach": approach,
            "model": model_kind,
            "fold": fold_id,
            "n_stations": len(stations),
            "n_eval_samples": int(len(all_true_arr)),
            "train_time_sec": float(total_time),
            **metrics(all_true_arr, all_pred_arr, mape_eps=mape_eps, mape_min_abs_target=mape_min_abs_target),
        },
        station_rows,
    )


def run_scenario(
    source_name: str,
    sources_used: list[str],
    scenario_name: str,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    reduced_stations: list[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    cfg: DictConfig,
    tracker: MLflowTracker,
    scenario_experiment: str,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    print(f"\n=== Source: {source_name} | Scenario: {scenario_name} | fold={fold_id} ===")
    if reduced_stations:
        print(f"Reduced-data stations ({len(reduced_stations)}): {', '.join(reduced_stations)}")
    training_cfg = cfg.training
    model_kinds = resolve_model_kinds(cfg)
    datetime_col = str(cfg.data.datetime_col)
    history_length = int(cfg.experiment.common.history_length)
    horizon = int(cfg.experiment.common.forecast_horizon)
    step_size = int(cfg.experiment.common.step_size)
    stations = sorted(frames)
    station_to_id = {st: idx for idx, st in enumerate(stations)}
    station_geo_map = load_station_geo_vectors(cfg, stations)
    print(
        f"[INFO] Training config ({scenario_name}): epochs={int(training_cfg.epochs)}, "
        f"batch_size={int(training_cfg.batch_size)}, lr=model-specific"
    )
    print(f"[INFO] Models: {', '.join(model_kinds)}")

    meta = {
        "data_source": source_name,
        "scenario": scenario_name,
        "reduced_station_count": len(reduced_stations),
        "reduced_stations": ",".join(reduced_stations),
    }

    stations_key = ",".join(stations).encode("utf-8")
    sources_key = ",".join(sources_used).encode("utf-8")
    reduced_key = ",".join(reduced_stations).encode("utf-8")
    missing_months = int(OmegaConf.select(cfg, "experiment.five_fold.missing_months", default=0))

    time_range_by_station = {
        st: {
            "min": str(frames[st][datetime_col].min()),
            "max": str(frames[st][datetime_col].max()),
        }
        for st in stations
    }

    dataset_name = f"{source_name}:{scenario_name}:fold{int(fold_id):02d}"
    dataset_manifest = {
        "data_source": source_name,
        "sources_used": list(sources_used),
        "scenario": scenario_name,
        "fold": int(fold_id),
        "datetime_col": datetime_col,
        "features": list(cfg.data.features),
        "target_features": list(cfg.data.target_features),
        "stations": list(stations),
        "time_range_by_station": time_range_by_station,
        "train_start_by_station": {st: str(train_starts[st]) for st in stations},
        "train_end": str(train_end),
        "val_end": str(val_end),
        "test_start": str(eval_start),
        "missing_months": missing_months,
        "reduced_stations": list(reduced_stations),
    }
    dataset_manifest_json = json.dumps(dataset_manifest, indent=2, sort_keys=True)
    dataset_manifest_sha1 = hashlib.sha1(dataset_manifest_json.encode("utf-8")).hexdigest()

    reduced_set = set(reduced_stations)
    dataset_rows: list[dict[str, Any]] = [
        {
            "station": str(st),
            "fold": int(fold_id),
            "source_name": str(source_name),
            "scenario": str(scenario_name),
            "is_reduced": st in reduced_set,
            "n_rows": int(len(frames[st])),
            "train_start": str(train_starts[st]),
            "train_end": str(train_end),
            "val_end": str(val_end),
            "test_start": str(eval_start),
            "time_min": str(frames[st][datetime_col].min()),
            "time_max": str(frames[st][datetime_col].max()),
        }
        for st in stations
    ]
    summary_rows: list[dict[str, float | int | str]] = []
    station_rows: list[dict[str, float | int | str]] = []
    window_cache = {
        st: build_window_index(
            frames[st][datetime_col].to_numpy(dtype="datetime64[ns]"),
            history_length,
            horizon,
            step_size,
        )
        for st in stations
    }

    # Prepare dataset splits once per scenario+fold and reuse across all model runs.
    # This removes repeated scaler fitting + windowing work between models.
    global_prepared = prepare_global_data(
        frames=frames,
        train_starts=train_starts,
        train_end=train_end,
        val_end=val_end,
        eval_start=eval_start,
        cfg=cfg,
        station_to_id=station_to_id,
        station_geo_map=station_geo_map,
        window_cache=window_cache,
    )
    per_station_prepared = prepare_per_station_data(
        frames=frames,
        train_starts=train_starts,
        train_end=train_end,
        val_end=val_end,
        eval_start=eval_start,
        cfg=cfg,
        station_geo_map=station_geo_map,
        window_cache=window_cache,
    )
    log_dataset_every_run = bool(OmegaConf.select(cfg, "tracking.log_dataset_every_run", default=False))
    log_per_station_torch_models = bool(OmegaConf.select(cfg, "tracking.log_per_station_torch_models", default=False))
    dataset_logged = False
    for model_kind in model_kinds:
        for approach_kind in ("global", "per_station"):
            approach = f"{model_kind}__{approach_kind}"
            model_display = (
                f"{model_kind} | {approach_kind} | {scenario_name} | "
                f"fold {int(fold_id):02d} | {source_name}"
            )
            model_artifact_path = (
                f"model__{_slug(source_name)}__{_slug(scenario_name)}__"
                f"fold{int(fold_id):02d}__{_slug(model_kind)}__{_slug(approach_kind)}"
            )
            run_name = f"forecast-{source_name}-{scenario_name}-fold{int(fold_id):02d}-{approach}"
            with tracker.start_run(
                run_name=run_name,
                tags={
                    "stage": "forecast_experiments",
                    "data_source": source_name,
                    "scenario": scenario_name,
                    "model": model_kind,
                    "approach": approach_kind,
                    "fold": int(fold_id),
                    "model_display": model_display,
                    "training_type": approach_kind,
                },
                experiment_name=scenario_experiment,
            ):
                tracker.log_params(to_plain_dict(cfg.experiment), prefix="experiment")
                tracker.log_params(to_plain_dict(cfg.training), prefix="training")
                tracker.log_params(to_plain_dict(model_runtime_cfg(cfg, model_kind)), prefix=f"models.{model_kind}.runtime")
                tracker.log_params(to_plain_dict(cfg.models[model_kind].params), prefix=f"models.{model_kind}")

                tracker.log_params(
                    {
                        "n_sources": len(sources_used),
                        "sources_sha1": hashlib.sha1(sources_key).hexdigest()[:12],
                        "n_stations": len(stations),
                        "stations_sha1": hashlib.sha1(stations_key).hexdigest()[:12],
                        "train_end": str(train_end),
                        "val_end": str(val_end),
                        "test_start": str(eval_start),
                        "missing_months": missing_months,
                        "reduced_station_count": len(reduced_stations),
                        "reduced_stations_sha1": hashlib.sha1(reduced_key).hexdigest()[:12],
                    },
                    prefix="dataset",
                )

                tracker.log_params({"manifest_sha1": dataset_manifest_sha1}, prefix="dataset")
                if log_dataset_every_run or not dataset_logged:
                    tracker.log_dict(dataset_manifest, artifact_file="dataset_manifest.json", artifact_path="dataset")
                    tracker.log_dataset_input(
                        name=dataset_name,
                        source=str(cfg.data.data_dir),
                        digest=dataset_manifest_sha1,
                        context="training",
                        rows=cast(list[Mapping[str, Any]], dataset_rows),
                        metadata={
                            "n_sources": len(sources_used),
                            "n_stations": len(stations),
                            "missing_months": missing_months,
                            "reduced_station_count": len(reduced_stations),
                        },
                    )
                    dataset_logged = True
                model_manifest: dict[str, Any] = {
                    "display_name": model_display,
                    "model_kind": str(model_kind),
                    "approach": str(approach),
                    "approach_kind": str(approach_kind),
                    "artifact_path": model_artifact_path,
                    "runtime": to_plain_dict(model_runtime_cfg(cfg, model_kind)),
                    "params": to_plain_dict(cfg.models[model_kind].params),
                    "features": list(cfg.data.features),
                    "target_features": list(cfg.data.target_features),
                    "save_models": bool(cfg.output.save_models),
                }
                tracker.log_dict(model_manifest, artifact_file="model_manifest.json", artifact_path="model")
                tracker.log_params(
                    {
                        "display_name": model_display,
                        "kind": str(model_kind),
                        "approach": str(approach_kind),
                        "training_type": str(approach_kind),
                        "artifact_path": model_artifact_path,
                        "artifacts_saved": bool(cfg.output.save_models),
                    },
                    prefix="model",
                )
                if approach_kind == "global":
                    summary, rows = run_approach(
                        source_name,
                        model_kind,
                        approach,
                        frames,
                        train_starts,
                        train_end,
                        val_end,
                        eval_start,
                        station_to_id,
                        station_geo_map,
                        cfg,
                        training_cfg,
                        fold_id,
                        tracker=tracker,
                        model_artifact_path=model_artifact_path,
                        prepared=global_prepared,
                    )
                else:
                    summary, rows = run_approach_per_station(
                        source_name,
                        model_kind,
                        approach,
                        frames,
                        train_starts,
                        train_end,
                        val_end,
                        eval_start,
                        station_geo_map,
                        cfg,
                        training_cfg,
                        fold_id,
                        tracker=tracker,
                        model_artifact_path_prefix=model_artifact_path,
                        prepared=per_station_prepared,
                        log_station_models=log_per_station_torch_models,
                    )

                if bool(cfg.output.save_models):
                    models_dir = Path(str(cfg.paths.models_dir)) / source_name / f"fold_{fold_id:02d}"
                    if approach_kind == "global":
                        tracker.log_artifact(models_dir / f"{approach}.pt", artifact_path="models")
                    else:
                        for st in stations:
                            tracker.log_artifact(models_dir / f"{approach}_{st}.pt", artifact_path="models")
                tracker.log_metrics(
                    {
                        f"summary.{k}": float(v)
                        for k, v in summary.items()
                        if isinstance(v, (int, float)) and np.isfinite(float(v))
                    }
                )
            summary_rows.append({**summary, **meta})
            station_rows.extend({**row, **meta} for row in rows)

    return summary_rows, station_rows


def validate_cfg(cfg: DictConfig) -> None:
    def req(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)

    features = list(cfg.data.features)
    target_features = list(cfg.data.target_features)
    req(bool(features), "cfg.data.features cannot be empty")
    req(bool(target_features), "cfg.data.target_features cannot be empty")
    req(all(t in features for t in target_features), "All target_features must exist in data.features")
    req(bool(str(cfg.data.data_dir).strip()), "cfg.data.data_dir cannot be empty")
    req(len(list(cfg.data.stations)) > 0, "cfg.data.stations cannot be empty")

    req(int(cfg.experiment.common.forecast_horizon) >= 1, "forecast_horizon must be >= 1")
    req(int(cfg.experiment.common.history_length) >= 1, "history_length must be >= 1")
    req(int(cfg.experiment.common.step_size) >= 1, "step_size must be >= 1")
    train_ratio_cfg = float(cfg.experiment.standard.train_ratio)
    val_ratio_cfg = float(cfg.experiment.standard.val_ratio)
    test_ratio_cfg = float(cfg.experiment.standard.test_ratio)
    req(0.0 < train_ratio_cfg < 1.0, "standard.train_ratio must be in (0, 1)")
    req(0.0 <= val_ratio_cfg < 1.0, "standard.val_ratio must be in [0, 1)")
    req(0.0 < test_ratio_cfg < 1.0, "standard.test_ratio must be in (0, 1)")
    req(
        abs((train_ratio_cfg + val_ratio_cfg + test_ratio_cfg) - 1.0) < 1e-9,
        "standard.train_ratio + standard.val_ratio + standard.test_ratio must equal 1.0",
    )
    missing_months_raw = OmegaConf.select(cfg, "experiment.five_fold.missing_months", default=None)
    if missing_months_raw is not None:
        req(int(missing_months_raw) >= 0, "five_fold.missing_months must be >= 0")
    req(
        (not bool(cfg.experiment.five_fold.run)) or int(cfg.experiment.five_fold.n_folds) >= 2,
        "n_folds must be >= 2 when five_fold.run=true",
    )
    req(int(cfg.training.epochs) >= 1, "training.epochs must be >= 1")
    req(int(cfg.training.batch_size) >= 1, "training.batch_size must be >= 1")

    resolve_model_kinds(cfg)


def run(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    # One-time CUDA runtime configuration (idempotent).
    configure_cuda_runtime(device_from_cfg(str(cfg.training.device)))
    paths = cfg.paths
    Path(paths.models_dir).mkdir(parents=True, exist_ok=True)
    tracking_cfg = to_plain_dict(OmegaConf.select(cfg, "tracking", default={}))
    tracking_cfg.setdefault("dataset_name", "-".join([str(st) for st in cfg.data.stations]))
    tracker = MLflowTracker(tracking_cfg)
    standard_experiment = str(
        OmegaConf.select(cfg, "tracking.experiment_standard", default="comperosion-normal")
    ).strip() or "comperosion-normal"
    five_fold_experiment = str(
        OmegaConf.select(cfg, "tracking.experiment_five_fold", default="comperosion-5fold")
    ).strip() or "comperosion-5fold"

    all_summary, all_station, fold_rows = [], [], []
    source_names = [Path(str(name).strip()).stem for name in cfg.data.stations if str(name).strip()]
    source_names = list(dict.fromkeys(source_names))

    # If the user lists multiple station CSVs (E403, E404, ...), treat them as one combined dataset
    # so we can do a station-level KFold as described in the mentor spec.
    if len(source_names) > 1 and "all_stations" not in source_names:
        source_groups: list[tuple[str, list[str]]] = [("yaml_stations", source_names)]
    else:
        source_groups = [(name, [name]) for name in source_names]

    for source_name, sources in source_groups:
        frames = load_frames(
            data_dir=Path(str(cfg.data.data_dir)),
            sources=sources,
            datetime_col=str(cfg.data.datetime_col),
            features=list(cfg.data.features),
        )
        datetime_col = str(cfg.data.datetime_col)
        stations = sorted(frames)
        min_ts = min(cast(pd.Timestamp, frame[datetime_col].min()) for frame in frames.values())
        max_ts = max(cast(pd.Timestamp, frame[datetime_col].max()) for frame in frames.values())
        total_seconds = (max_ts - min_ts).total_seconds()
        train_ratio = float(cfg.experiment.standard.train_ratio)
        val_ratio = float(cfg.experiment.standard.val_ratio)
        train_end = min_ts + pd.to_timedelta(total_seconds * train_ratio, unit="s")
        val_end = min_ts + pd.to_timedelta(total_seconds * (train_ratio + val_ratio), unit="s")
        eval_start = val_end
        missing_months = int(OmegaConf.select(cfg, "experiment.five_fold.missing_months", default=0))
        station_first_ts = {
            st: cast(pd.Timestamp, frames[st][datetime_col].min()) for st in stations
        }
        reduced_start_by_station = {
            st: station_first_ts[st] + pd.DateOffset(months=missing_months) for st in stations
        }

        print(f"\n### Data source: {source_name}")
        print(f"Stations: {len(stations)}")
        print(f"Train period cutoff: {train_end}")
        print(f"Validation period cutoff: {val_end}")
        print(f"Evaluation starts at: {eval_start}")
        if bool(cfg.experiment.five_fold.run):
            print(f"[INFO] five_fold reduced stations drop first {missing_months} months of training data (per-station)")

        if bool(cfg.experiment.standard.run):
            summary, rows = run_scenario(
                source_name,
                sources,
                "standard",
                0,
                frames,
                {st: min_ts for st in stations},
                [],
                train_end,
                val_end,
                eval_start,
                cfg,
                tracker,
                standard_experiment,
            )
            all_summary.extend(summary)
            all_station.extend(rows)

        run_five_fold = bool(cfg.experiment.five_fold.run)
        if run_five_fold:
            n_folds = int(cfg.experiment.five_fold.n_folds)
            if n_folds > len(stations):
                print(
                    f"[WARN] Skipping five_fold for source={source_name}: "
                    f"n_folds={n_folds} > number_of_stations={len(stations)}"
                )
                continue
            for fold, (_, reduced_idx) in enumerate(
                KFold(n_splits=n_folds, shuffle=True, random_state=int(cfg.seed)).split(stations),
                start=1,
            ):
                reduced = [stations[i] for i in reduced_idx]
                starts = {st: (reduced_start_by_station[st] if st in reduced else min_ts) for st in stations}
                fold_rows.extend(
                    {
                        "data_source": source_name,
                        "fold": fold,
                        "station": st,
                        "scenario": "five_fold",
                        "train_start": str(reduced_start_by_station[st]),
                    }
                    for st in reduced
                )
                summary, rows = run_scenario(
                    source_name,
                    sources,
                    "five_fold",
                    fold,
                    frames,
                    starts,
                    reduced,
                    train_end,
                    val_end,
                    eval_start,
                    cfg,
                    tracker,
                    five_fold_experiment,
                )
                all_summary.extend(summary)
                all_station.extend(rows)

    yaml_station_filter = [st for st in source_names if st != "all_stations"]
    avg_data_source = "all_stations" if ("all_stations" in source_names) else ("yaml_stations" if len(source_names) > 1 else None)

    if yaml_station_filter and avg_data_source is not None:
        station_df = pd.DataFrame(all_station)
        if not station_df.empty and {"scenario", "station", "model", "fold", "data_source"}.issubset(set(station_df.columns)):
            standard_subset = station_df[
                (station_df["scenario"].astype(str) == "standard")
                & (station_df["data_source"].astype(str) == str(avg_data_source))
                & (station_df["station"].astype(str).isin(yaml_station_filter))
            ].copy()
            if not standard_subset.empty:
                metric_mean_keys = ["mae", "mape", "smape", "wape", "mse", "rmse", "evs", "r2"]
                for (approach, model, fold), sub in standard_subset.groupby(["approach", "model", "fold"], dropna=True):
                    avg_row: dict[str, float | int | str] = {
                        "approach": str(approach),
                        "model": str(model),
                        "fold": int(cast(Any, fold)),
                        "n_stations": int(sub["station"].astype(str).nunique()),
                        "n_eval_samples": int(pd.to_numeric(sub["n_eval_samples"], errors="coerce").fillna(0).sum()),
                        "train_time_sec": float("nan"),
                        "scenario": "standard",
                        "data_source": "yaml_stations_avg",
                        "reduced_station_count": 0,
                        "reduced_stations": "",
                    }
                    for key in metric_mean_keys:
                        vals = pd.to_numeric(sub[key], errors="coerce").dropna() if key in sub.columns else pd.Series(dtype=float)
                        if len(vals):
                            avg_row[key] = float(vals.mean())
                    all_summary.append(avg_row)

    def dvc_metrics_payload(summary_rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
        df = pd.DataFrame(summary_rows)
        if df.empty or "scenario" not in df.columns or "approach" not in df.columns:
            return {}
        if "data_source" not in df.columns:
            df = df.assign(data_source="unknown")
        metric_keys = ["mae", "mape", "smape", "wape", "mse", "rmse", "evs", "r2", "train_time_sec", "n_eval_samples"]
        out: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
        for (scenario, data_source, approach), sub in df.groupby(["scenario", "data_source", "approach"], dropna=True):
            agg: dict[str, object] = {"n_folds": int(sub["fold"].nunique()) if "fold" in sub.columns else int(len(sub))}
            for key in metric_keys:
                if key not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[key], errors="coerce").dropna()
                if len(vals):
                    agg[key] = float(vals.mean())
            out.setdefault(str(scenario), {}).setdefault(str(data_source), {})[str(approach)] = agg
        return out

    reports_dir = Path(paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summary).to_csv(reports_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(all_station).to_csv(reports_dir / "station_metrics.csv", index=False)
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(reports_dir / "fold_assignments.csv", index=False)
    OmegaConf.save(cfg, reports_dir / "resolved_config.yaml")

    metrics_path = Path(paths.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(dvc_metrics_payload(all_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_targets: list[tuple[str, str, str]] = []
    if any(str(row.get("scenario", "")) == "standard" for row in all_summary):
        summary_targets.append(("standard", "forecast-experiments-summary-standard", f"{standard_experiment}-summary"))
    if any(str(row.get("scenario", "")) == "five_fold" for row in all_summary):
        summary_targets.append(("five_fold", "forecast-experiments-summary-five-fold", f"{five_fold_experiment}-summary"))

    for scenario_name, run_name, experiment_name in summary_targets:
        with tracker.start_run(
            run_name=run_name,
            tags={"stage": "forecast_experiments", "scenario": scenario_name},
            experiment_name=experiment_name,
        ):
            tracker.log_params(to_plain_dict(cfg), prefix="config")
            tracker.log_artifact(reports_dir / "summary_metrics.csv", artifact_path="reports")
            tracker.log_artifact(reports_dir / "station_metrics.csv", artifact_path="reports")
            if fold_rows:
                tracker.log_artifact(reports_dir / "fold_assignments.csv", artifact_path="reports")
            tracker.log_artifact(reports_dir / "resolved_config.yaml", artifact_path="reports")
            tracker.log_artifact(metrics_path, artifact_path="reports")

    print("\nSaved reports:")
    print(f"- {reports_dir / 'summary_metrics.csv'}")
    print(f"- {reports_dir / 'station_metrics.csv'}")
    if fold_rows:
        print(f"- {reports_dir / 'fold_assignments.csv'}")
    print(f"- {reports_dir / 'resolved_config.yaml'}")
    print(f"- {metrics_path}")


def run_from_params(params_path: Path) -> None:
    load_env_file(Path(".env"))
    cfg = OmegaConf.load(params_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {params_path}, got {type(cfg)!r}")
    validate_cfg(cfg)
    run(cfg)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run forecasting experiments and emit reports/metrics.")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("configs/pipeline/params.yaml"),
        help="Path to YAML params file.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_from_params(Path(args.params))


if __name__ == "__main__":
    main()
