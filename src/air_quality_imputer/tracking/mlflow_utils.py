from __future__ import annotations

import contextlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
        tracking_uri = str(cfg.get("tracking_uri", "")).strip()
        if not tracking_uri and repo_owner and repo_name:
            tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

        if tracking_uri:
            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as exc:
                logger.warning("mlflow.set_tracking_uri failed: %s", exc)
                self.enabled = False
                return

        if repo_owner and repo_name:
            if dagshub is None:
                logger.warning("dagshub not installed; skipping dagshub.init")
            else:
                try:
                    dagshub.init(repo_owner=str(repo_owner), repo_name=str(repo_name), mlflow=True)
                except Exception as exc:
                    logger.warning("dagshub.init failed: %s", exc)

        self.default_experiment = str(cfg.get("experiment", "air-quality-imputer"))

    def _set_experiment(self, experiment_name: str | None) -> None:
        if not self.enabled or mlflow is None:
            return
        name = (experiment_name or self.default_experiment).strip() or self.default_experiment
        try:
            mlflow.set_experiment(name)
            return
        except Exception as exc:
            logger.warning("mlflow.set_experiment failed for '%s': %s", name, exc)

        # Some tracking backends (incl. DagsHub/MLflow) keep deleted experiments
        # in a recoverable state; restore and retry before disabling tracking.
        try:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(name)
            if exp is not None and str(getattr(exp, "lifecycle_stage", "active")) != "active":
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(name)
                logger.info("Restored deleted MLflow experiment '%s' (id=%s)", name, exp.experiment_id)
                return
        except Exception as restore_exc:
            logger.warning("Failed to restore deleted MLflow experiment '%s': %s", name, restore_exc)

        self.enabled = False

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: str,
        tags: Mapping[str, Any] | None = None,
        experiment_name: str | None = None,
    ) -> Iterator[Any]:
        if not self.enabled or mlflow is None:
            yield None
            return
        self._set_experiment(experiment_name)
        if not self.enabled:
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

    def log_dict(self, payload: Mapping[str, Any], artifact_file: str, artifact_path: str | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        try:
            rel_path = artifact_file.strip("/")
            if artifact_path:
                rel_path = f"{artifact_path.strip('/')}/{rel_path}"
            if hasattr(mlflow, "log_dict"):
                mlflow.log_dict(dict(payload), rel_path)
                return
            logger.warning("mlflow.log_dict unavailable; skipping dict artifact '%s'", rel_path)
        except Exception as exc:
            logger.warning("mlflow.log_dict failed for '%s': %s", artifact_file, exc)

    def log_dataset_input(
        self,
        *,
        name: str,
        source: str,
        digest: str,
        context: str = "training",
        rows: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled or mlflow is None:
            return
        try:
            data_module = getattr(mlflow, "data", None)
            from_pandas = getattr(data_module, "from_pandas", None) if data_module is not None else None
            log_input = getattr(mlflow, "log_input", None)
            if from_pandas is None or log_input is None:
                logger.warning("mlflow dataset input API unavailable; skipping dataset input logging")
                return

            import pandas as pd

            data_rows: list[dict[str, Any]]
            if rows:
                data_rows = [dict(r) for r in rows]
            else:
                data_rows = [{"name": str(name), "source": str(source), "digest": str(digest)}]

            for row in data_rows:
                row.setdefault("name", str(name))
                row.setdefault("source", str(source))
                row.setdefault("digest", str(digest))
                for key, value in dict(metadata or {}).items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        row.setdefault(str(key), value)

            ds = from_pandas(pd.DataFrame(data_rows), source=str(source), name=str(name))
            log_input(ds, context=str(context))
        except Exception as exc:
            logger.warning("mlflow.log_input failed for dataset '%s': %s", name, exc)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def log_torch_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self.enabled or mlflow is None:
            return
        try:
            mlflow_pytorch = getattr(mlflow, "pytorch", None)
            log_model = getattr(mlflow_pytorch, "log_model", None) if mlflow_pytorch is not None else None
            if log_model is None:
                logger.warning("mlflow.pytorch.log_model unavailable; skipping model logging")
                return
            log_model(model, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("mlflow.pytorch.log_model failed for '%s': %s", artifact_path, exc)
