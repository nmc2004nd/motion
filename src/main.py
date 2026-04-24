"""CLI entry point cho pipeline batch: `python -m src.main`."""

from __future__ import annotations

import argparse
import logging

from .config import load_config
from .config.loader import ConfigError
from .pipelines.batch import TactileMarkerTrackingPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactile marker tracking pipeline")
    parser.add_argument(
        "--config-path", default="config/pipeline_config.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    try:
        config = load_config(args.config_path)
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")
    TactileMarkerTrackingPipeline(config=config).run()


if __name__ == "__main__":
    main()
