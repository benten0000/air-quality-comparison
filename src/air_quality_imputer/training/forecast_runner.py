from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    "bias": True,
    "use_torch_compile": True,
    "compile_mode": "reduce-overhead",
    "compile_dynamic": False,
    "optimizer_fused": True,
    "optimizer_weight_decay": 0.01,
    "scheduler_warmup_ratio": 0.125,
    "scheduler_warmup_start_factor": 0.3,
    "scheduler_min_lr": 1e-7,
    "grad_clip_norm": 0.5,
    "dataloader_num_workers": -1,
    "dataloader_prefetch_factor": 4,
    "dataloader_persistent_workers": True,
    "dataloader_pin_memory": True,
}


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
    if np.any(valid_mape):
        denom_used = np.maximum(abs_true[valid_mape], float(max(mape_min_abs_target, 0.0)))
        mape = float(np.mean(np.abs(err[valid_mape]) / denom_used) * 100.0)
    else:
        mape = float("nan")

    denom_smape = abs_true + np.abs(y_pred)
    valid_smape = denom_smape > float(mape_eps)
    smape = (
        float(np.mean((2.0 * np.abs(err[valid_smape])) / denom_smape[valid_smape]) * 100.0)
        if np.any(valid_smape)
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
        "evs": float(explained_variance_score(y_true, y_pred, multioutput="uniform_average")),
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


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

    for source_name in source_names:
        csv_path = data_dir / f"{source_name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing data source: {csv_path}")

        frame = pd.read_csv(csv_path)
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
        work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
        work = work.dropna(subset=[datetime_col]).copy()
        work["_station_label"] = work["_station_label"].astype(str).str.strip()
        work = work[work["_station_label"] != ""]
        if work.empty:
            continue

        if non_station_features:
            work[non_station_features] = work[non_station_features].apply(pd.to_numeric, errors="coerce")
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


def make_samples(
    values: np.ndarray,
    times: np.ndarray,
    history_length: int,
    horizon: int,
    target_indices: list[int],
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = int(values.shape[0])
    if t < int(history_length) + int(horizon):
        return (
            np.empty((0, history_length, values.shape[1]), dtype=np.float32),
            np.empty((0, len(target_indices)), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    ends = np.arange(int(history_length), t - int(horizon) + 1, int(step_size))
    x_all = np.lib.stride_tricks.sliding_window_view(values, int(history_length), axis=0).swapaxes(1, 2)
    x = x_all[ends - int(history_length)].astype(np.float32, copy=False)
    t_idx = ends + int(horizon) - 1
    y = values[t_idx][:, target_indices].astype(np.float32, copy=False)
    ts = times[t_idx].astype("datetime64[ns]", copy=False)
    return x, y, ts


def split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        return x, y, np.empty((0, *x.shape[1:]), dtype=np.float32), np.empty((0, y.shape[1]), dtype=np.float32)
    cut = min(max(int(round(len(x) * (1.0 - val_ratio))), 1), len(x) - 1)
    return x[:cut], y[:cut], x[cut:], y[cut:]


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
    eval_start: pd.Timestamp,
    val_ratio: float,
) -> dict[str, np.ndarray]:
    times = pd.to_datetime(frame[datetime_col]).to_numpy(dtype="datetime64[ns]")
    scaled = scaler.transform(frame[features].to_numpy(dtype=np.float32)).astype(np.float32)

    mask = (times >= np.datetime64(train_start)) & (times < np.datetime64(train_end))
    x_train_all, y_train_all, _ = make_samples(
        scaled[mask],
        times[mask],
        history_length,
        horizon,
        target_indices,
        step_size,
    )
    x_train, y_train, x_val, y_val = split_train_val(x_train_all, y_train_all, val_ratio)

    x_all, y_all, t_all = make_samples(scaled, times, history_length, horizon, target_indices, step_size)
    eval_mask = t_all >= np.datetime64(eval_start)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_all[eval_mask],
        "y_test": y_all[eval_mask],
    }


def inverse_scale_targets(y_scaled: np.ndarray, scaler: StandardScaler, target_indices: list[int]) -> np.ndarray:
    mean = scaler.mean_
    scale = scaler.scale_
    if mean is None or scale is None:
        raise ValueError("Scaler must be fitted before inverse scaling targets")
    return y_scaled * scale[target_indices] + mean[target_indices]


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
    blocks = []
    for st, frame in frames.items():
        m = (frame[datetime_col] >= train_starts[st]) & (frame[datetime_col] < train_end)
        arr = frame.loc[m, features].to_numpy(dtype=np.float32)
        if arr.size:
            blocks.append(arr)
    if not blocks:
        raise ValueError("No training rows available for global scaler")
    sc = StandardScaler()
    sc.fit(np.concatenate(blocks, axis=0))
    return sc


def run_approach(
    source_name: str,
    model_kind: ModelKind,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    station_to_id: dict[str, int],
    station_geo_map: dict[str, np.ndarray],
    cfg: DictConfig,
    training_cfg: DictConfig,
    fold_id: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    approach = model_kind
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

    datetime_col = str(cfg.data.datetime_col)
    horizon = int(cfg.experiment.common.forecast_horizon)
    step_size = int(cfg.experiment.common.step_size)
    val_ratio = float(cfg.experiment.common.val_ratio)
    runtime_cfg = model_runtime_cfg(cfg, model_kind)

    scaler = fit_global_scaler(frames, datetime_col, features, train_starts, train_end)
    scalers = {st: scaler for st in stations}

    splits: dict[str, dict[str, np.ndarray]] = {}
    for st in stations:
        splits[st] = build_split(
            frames[st],
            datetime_col,
            features,
            target_idx,
            scalers[st],
            history_length,
            horizon,
            step_size,
            train_starts[st],
            train_end,
            eval_start,
            val_ratio,
        )

    for st in stations:
        sp = splits[st]
        if len(sp["x_train"]) == 0:
            raise ValueError(f"{approach}: station {st} has no training samples")
        if len(sp["x_test"]) == 0:
            raise ValueError(f"{approach}: station {st} has no evaluation samples")

    def sid_geo(station: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((n,), station_to_id[station], dtype=np.int64),
            np.repeat(station_geo_map[station][None, :], n, axis=0).astype(np.float32, copy=False),
        )

    def payload_for(stations_in: list[str], key: str) -> tuple[np.ndarray, np.ndarray]:
        sid = _cat(
            [sid_geo(st, len(splits[st][key]))[0] for st in stations_in],
            empty=np.empty((0,), dtype=np.int64),
        )
        geo = _cat(
            [sid_geo(st, len(splits[st][key]))[1] for st in stations_in],
            empty=np.empty((0, 2), dtype=np.float32),
        )
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

    station_rows: list[dict[str, float | int | str]] = []
    all_true, all_pred = [], []
    for st in stations:
        sp = splits[st]
        sid_test, geo_test = sid_geo(st, len(sp["x_test"]))
        pred_scaled = predict_model(
            model,
            x=sp["x_test"],
            target_indices=target_idx,
            batch_size=int(training_cfg.batch_size),
            device=dev,
            runtime_cfg=runtime_cfg,
            station_ids=sid_test,
            station_geo=geo_test,
        )
        y_true = inverse_scale_targets(sp["y_test"], scalers[st], target_idx)
        y_pred = inverse_scale_targets(pred_scaled, scalers[st], target_idx)
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
            Path(paths.models_dir) / source_name / f"fold_{fold_id:02d}" / f"{approach}.pt",
            model,
            model_kind,
            features,
            target_features,
            scalers[stations[0]],
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
            **metrics(
                all_true_arr,
                all_pred_arr,
                mape_eps=mape_eps,
                mape_min_abs_target=mape_min_abs_target,
            ),
        },
        station_rows,
    )


def run_scenario(
    source_name: str,
    scenario_name: str,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    train_starts: dict[str, pd.Timestamp],
    reduced_stations: list[str],
    train_end: pd.Timestamp,
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
    stations = sorted(frames)
    station_to_id = {st: idx for idx, st in enumerate(stations)}
    station_geo_map = load_station_geo_vectors(cfg, stations)
    print(
        f"[INFO] Training config ({scenario_name}): epochs={int(training_cfg.epochs)}, "
        f"batch_size={int(training_cfg.batch_size)}, lr={float(training_cfg.lr):.6g}"
    )
    print(f"[INFO] Models: {', '.join(model_kinds)}")

    meta = {
        "data_source": source_name,
        "scenario": scenario_name,
        "reduced_station_count": len(reduced_stations),
        "reduced_stations": ",".join(reduced_stations),
    }
    summary_rows: list[dict[str, float | int | str]] = []
    station_rows: list[dict[str, float | int | str]] = []
    for model_kind in model_kinds:
        run_name = f"forecast-{source_name}-{scenario_name}-fold{int(fold_id):02d}-{model_kind}"
        with tracker.start_run(
            run_name=run_name,
            tags={
                "stage": "forecast_experiments",
                "data_source": source_name,
                "scenario": scenario_name,
                "model": model_kind,
                "fold": int(fold_id),
            },
            experiment_name=scenario_experiment,
        ):
            tracker.log_params(to_plain_dict(cfg.experiment), prefix="experiment")
            tracker.log_params(to_plain_dict(cfg.training), prefix="training")
            tracker.log_params(to_plain_dict(model_runtime_cfg(cfg, model_kind)), prefix=f"models.{model_kind}.runtime")
            tracker.log_params(to_plain_dict(cfg.models[model_kind].params), prefix=f"models.{model_kind}")
            summary, rows = run_approach(
                source_name,
                model_kind,
                frames,
                train_starts,
                train_end,
                eval_start,
                station_to_id,
                station_geo_map,
                cfg,
                training_cfg,
                fold_id,
            )
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
    req(0.0 < float(cfg.experiment.standard.train_ratio) < 1.0, "standard.train_ratio must be in (0, 1)")
    req(0.0 < float(cfg.experiment.five_fold.reduced_train_ratio) <= 1.0, "five_fold.reduced_train_ratio must be in (0, 1]")
    req(0.0 <= float(cfg.experiment.common.val_ratio) < 1.0, "val_ratio must be in [0, 1)")
    req(
        float(cfg.experiment.five_fold.reduced_train_ratio) <= float(cfg.experiment.standard.train_ratio),
        "five_fold.reduced_train_ratio must be <= standard.train_ratio",
    )
    req(
        (not bool(cfg.experiment.five_fold.run)) or int(cfg.experiment.five_fold.n_folds) >= 2,
        "n_folds must be >= 2 when five_fold.run=true",
    )
    req(float(cfg.training.lr) > 0.0, "training.lr must be > 0")
    req(int(cfg.training.epochs) >= 1, "training.epochs must be >= 1")
    req(int(cfg.training.batch_size) >= 1, "training.batch_size must be >= 1")

    resolve_model_kinds(cfg)


def run(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
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

    for source_name in source_names:
        frames = load_frames(
            data_dir=Path(str(cfg.data.data_dir)),
            sources=[source_name],
            datetime_col=str(cfg.data.datetime_col),
            features=list(cfg.data.features),
        )
        stations = sorted(frames)
        all_times = pd.concat([frame[str(cfg.data.datetime_col)] for frame in frames.values()], ignore_index=True)
        min_ts = pd.to_datetime(all_times.min())
        max_ts = pd.to_datetime(all_times.max())
        total_seconds = (max_ts - min_ts).total_seconds()
        train_ratio = float(cfg.experiment.standard.train_ratio)
        train_end = min_ts + pd.to_timedelta(total_seconds * train_ratio, unit="s")
        eval_start = train_end
        reduced_ratio = float(cfg.experiment.five_fold.reduced_train_ratio)
        reduced_start = min_ts + pd.to_timedelta(total_seconds * (train_ratio - reduced_ratio), unit="s")

        print(f"\n### Data source: {source_name}")
        print(f"Stations: {len(stations)}")
        print(f"Train period cutoff: {train_end}")
        print(f"Evaluation starts at: {eval_start}")

        if bool(cfg.experiment.standard.run):
            summary, rows = run_scenario(
                source_name,
                "standard",
                0,
                frames,
                {st: min_ts for st in stations},
                [],
                train_end,
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
                starts = {st: (reduced_start if st in reduced else min_ts) for st in stations}
                fold_rows.extend(
                    {
                        "data_source": source_name,
                        "fold": fold,
                        "station": st,
                        "scenario": "five_fold",
                        "train_start": str(reduced_start),
                    }
                    for st in reduced
                )
                summary, rows = run_scenario(
                    source_name,
                    "five_fold",
                    fold,
                    frames,
                    starts,
                    reduced,
                    train_end,
                    eval_start,
                    cfg,
                    tracker,
                    five_fold_experiment,
                )
                all_summary.extend(summary)
                all_station.extend(rows)

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
