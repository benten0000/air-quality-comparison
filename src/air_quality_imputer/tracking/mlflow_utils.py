from __future__ import annotations

import contextlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

try:
    import dagshub
except Exception:  # pragma: no cover
    dagshub = None


logger = logging.getLogger("air_quality_imputer.tracking")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        nested = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, nested))
        else:
            out[nested] = value
    return out


class MLflowTracker:
    def __init__(self, tracking_cfg: Mapping[str, Any] | None):
        cfg = dict(tracking_cfg or {})
        self.enabled = bool(cfg.get("enabled", True)) and mlflow is not None
        self.dataset_name = str(cfg.get("dataset_name", "air_quality")).strip() or "air_quality"
        self.base_tags = {"git_commit": _git_commit(), "project": "air-quality-imputer", "dataset": self.dataset_name}
        if not self.enabled:
            return

        repo_owner = cfg.get("repo_owner")
        repo_name = cfg.get("repo_name")
        if repo_owner and repo_name:
            if dagshub is None:
                logger.warning("dagshub not installed; skipping dagshub.init")
            else:
                try:
                    dagshub.init(repo_owner=str(repo_owner), repo_name=str(repo_name), mlflow=True)
                except Exception as exc:
                    logger.warning("dagshub.init failed: %s", exc)

        try:
            mlflow.set_experiment(str(cfg.get("experiment", "air-quality-imputer")))
        except Exception as exc:
            logger.warning("mlflow.set_experiment failed: %s", exc)
            self.enabled = False

    @contextlib.contextmanager
    def start_run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled or mlflow is None:
            yield None
            return
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({k: str(v) for k, v in {**self.base_tags, **dict(tags or {})}.items() if v is not None})
            yield run

    def log_params(self, params: Mapping[str, Any], prefix: str | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        flat = _flatten(params)
        if prefix:
            flat = {f"{prefix}.{k}": v for k, v in flat.items()}
        payload = {k: str(v) for k, v in flat.items() if v is not None}
        if payload:
            mlflow.log_params(payload)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                number = float(value)
            except Exception:
                continue
            if number == number:
                payload[str(key)] = number
        if not payload:
            return
        mlflow.log_metrics(payload) if step is None else mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
