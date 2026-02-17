from __future__ import annotations

import argparse
from pathlib import Path

from air_quality_imputer.training.forecast_runner import run_from_params


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
    run_from_params(Path(args.params))


if __name__ == "__main__":
    main()
