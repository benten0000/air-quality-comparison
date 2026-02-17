from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from air_quality_imputer.training.forecast_runner import run, validate_cfg


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run forecasting experiments and emit DVC metrics.")
    p.add_argument(
        "--params",
        type=Path,
        default=Path("configs/pipeline/params.yaml"),
        help="Path to pipeline params.yaml used by DVC.",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = OmegaConf.load(Path(args.params))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {args.params}, got {type(cfg)!r}")
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
