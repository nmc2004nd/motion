"""Pipeline realtime chỉ tracking (không slip). Hỗ trợ cProfile."""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats

import cv2
import numpy as np

from ..common.overlay import draw_text, draw_timings, draw_tracking_status
from ..common.perf import TimingAccumulator
from ..config import load_config, require
from ..config.loader import ConfigError
from ..core.tracking import track_markers_lk
from ..core.visualization import visualize_flow_arrows
from .base import BaseRealtimePipeline

logger = logging.getLogger(__name__)

_FLOW_WINDOW = "Realtime Flow Arrows"


class RealtimeTactileTracking(BaseRealtimePipeline):
    @property
    def window_name(self) -> str:
        return f"Raw WebCam (/dev/video{self.camera.device_id})"

    def process_frame(self, gray: np.ndarray, display: np.ndarray, fps: float) -> None:
        if len(self.reference_markers) == 0:  # type: ignore[arg-type]
            draw_tracking_status(display, "Tracking (LK) | 0 markers", self.config)
            return

        timings = TimingAccumulator()
        with timings.measure("track"):
            tracked, valid = track_markers_lk(
                self.reference_image, gray, self.reference_markers, config=self.config
            )
        with timings.measure("visualize"):
            if valid.any():
                vis = visualize_flow_arrows(
                    gray, self.reference_markers, tracked, valid,
                    config=self.config, save_path=None,
                )
                cv2.imshow(_FLOW_WINDOW, vis)

        self._draw_overlay(display, fps, timings.snapshot())

    def _draw_overlay(self, display: np.ndarray, fps: float, timings: dict[str, float]) -> None:
        n = len(self.reference_markers)  # type: ignore[arg-type]
        draw_tracking_status(display, f"Dang track (LK): {n} markers", self.config)

        # FPS màu theo "tracking" (khi đang có reference).
        pos_fps = tuple(require(self.config, "visualization.text.fps_pos"))
        scale = float(require(self.config, "visualization.text.scale_normal"))
        thickness = int(require(self.config, "visualization.text.thickness"))
        color = tuple(require(self.config, "visualization.text.color_tracking"))
        draw_text(display, f"FPS: {fps:.1f}", pos_fps, color, scale, thickness)

        draw_timings(display, timings, self.config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Realtime tactile tracking")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml",
                        help="Path to YAML configuration")
    parser.add_argument("--profile", action="store_true", help="Enable profiling with cProfile")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")

    pipeline = RealtimeTactileTracking(config=config)

    if args.profile:
        logger.info("Chạy ở chế độ Profile...")
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            pipeline.run()
        finally:
            profiler.disable()
            pstats.Stats(profiler).sort_stats("cumtime").dump_stats("profile_result.prof")
            logger.info("Đã lưu kết quả profiling tại 'profile_result.prof'. "
                        "Dùng snakeviz profile_result.prof để xem.")
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
