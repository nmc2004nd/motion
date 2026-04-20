"""Diem vao CLI de chay pipeline theo doi marker tactile."""

import argparse

from .pipeline import PipelineConfig, TactileMarkerTrackingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactile marker tracking pipeline")
    parser.add_argument("--ref-image", default="data/ref/my_photo_1.jpg", help="Path to reference image")
    parser.add_argument("--def-image", default="data/img/my_photo_4.png", help="Path to deformed image")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=2.0,
        help="Scale factor for displacement arrows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        ref_image_path=args.ref_image,
        def_image_path=args.def_image,
        output_dir=args.output_dir,
        arrow_scale=args.arrow_scale,
    )
    pipeline = TactileMarkerTrackingPipeline(config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
