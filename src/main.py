"""Command Line Interface (CLI) entry point for the tactile marker tracking pipeline."""

import argparse
import logging

from .pipeline import TactileMarkerTrackingPipeline
from .utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tactile marker tracking pipeline")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml", help="Path to the configuration YAML file")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    config_parser = load_config(args.config_path)
    config = config_parser.config
    
    pipeline = TactileMarkerTrackingPipeline(config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
