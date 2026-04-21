"""Command Line Interface (CLI) entry point for the tactile marker tracking pipeline."""

import argparse
import logging

from .pipeline import PipelineConfig, TactileMarkerTrackingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tactile marker tracking pipeline")
    parser.add_argument("--ref-image", default="data/ref/my_photo_1.jpg", help="Path to the reference image")
    parser.add_argument("--def-image", dest="deformed_image", default="data/img/my_photo_2.jpg", help="Path to the deformed image")
    parser.add_argument("--calib-file", default="config/calib_result.npz", help="Path to camera calibration file (.npz)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=2.0,
        help="Scale factor for displacement arrows",
    )
    parser.add_argument(
        "--tracking-method",
        type=str,
        choices=["H", "LK"],
        default="H",
        help="Tracking method to use: 'H' for Hungarian or 'LK' for PyrLK",
    )
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Apply camera calibration to undistort images before tracking",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    config = PipelineConfig(
        reference_image_path=args.ref_image,
        deformed_image_path=args.deformed_image,
        calib_file_path=args.calib_file,
        output_dir=args.output_dir,
        arrow_scale=args.arrow_scale,
        tracking_method=args.tracking_method,
        use_calibration=args.use_calibration,
    )
    pipeline = TactileMarkerTrackingPipeline(config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
