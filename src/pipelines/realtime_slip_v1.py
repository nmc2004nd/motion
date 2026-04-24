"""Pipeline realtime + SlipDetector V1 (MRVL)."""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

from ..common.overlay import draw_text, draw_tracking_status
from ..config import load_config, require
from ..config.loader import ConfigError
from ..core.tracking import track_markers_lk
from ..core.visualization import visualize_flow_arrows
from ..slip.v1 import SlipDetector
from .base import BaseRealtimePipeline

logger = logging.getLogger(__name__)

_FLOW_WINDOW = "Realtime Flow Arrows"


class RealtimeSlipTracking(BaseRealtimePipeline):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.slip_detector = SlipDetector(config=config)
        self.history_length = int(require(config, "slip_detection.history_buffer_length"))
        self.slip_threshold_display = float(require(config, "slip_detection.slip_threshold"))
        self.moving_count_thresh = int(require(config, "slip_detection.moving_count_thresh"))
        self.marker_history: list[np.ndarray] = []

    @property
    def window_name(self) -> str:
        return f"WebCam w/ Slip Detection (/dev/video{self.camera.device_id})"

    @property
    def startup_lines(self) -> list[str]:
        return [
            f"=== BẮT ĐẦU CAMERA {self.camera.device_id} TÍCH HỢP SLIP DETECTION ===",
            "- Nhấn 'r' để chụp frame hiện tại làm Reference (Trạng thái tĩnh).",
            "- Nhấn 'c' để xóa Reference frame.",
            "- Nhấn 'q' để thoát.",
        ]

    def on_reference_captured(self, gray: np.ndarray) -> None:
        # KHÔNG seed history bằng reference để tránh velocity spike ở frame đầu.
        self.marker_history = []

    def on_reference_cleared(self) -> None:
        self.marker_history = []
        self.slip_detector.reset()

    def process_frame(self, gray: np.ndarray, display: np.ndarray, fps: float) -> None:
        n = len(self.reference_markers)  # type: ignore[arg-type]
        draw_tracking_status(display, f"Tracking (LK) | {n} markers", self.config)

        if n == 0:
            return

        tracked, valid = track_markers_lk(
            self.reference_image, gray, self.reference_markers, config=self.config
        )

        slip_info = None
        if valid.any():
            self.marker_history.append(tracked.copy())
            if len(self.marker_history) > self.history_length:
                self.marker_history.pop(0)

            # Chỉ tính slip khi buffer có ≥ 2 frame thực (tránh velocity spike frame đầu).
            if len(self.marker_history) >= 2:
                past = self.marker_history[0]
                slip_info = self.slip_detector.calculate_slip_probability(
                    prev_markers=past,
                    current_markers=tracked,
                    valid_mask=valid,
                    ref_markers=self.reference_markers,
                )

            vis = visualize_flow_arrows(
                gray, self.reference_markers, tracked, valid,
                config=self.config, save_path=None,
            )
            cv2.imshow(_FLOW_WINDOW, vis)

        if slip_info is not None:
            self._draw_slip_text(display, slip_info)

    def _draw_slip_text(self, display: np.ndarray, slip_info: dict) -> None:
        r_val = slip_info["r_value"]
        moving = slip_info["moving_count"]
        slip_status = 1 if r_val > self.slip_threshold_display else 0

        pos = tuple(require(self.config, "visualization.text.slip_pos"))
        scale = float(require(self.config, "visualization.text.scale_normal"))
        thickness = int(require(self.config, "visualization.text.thickness"))
        c_slip = tuple(require(self.config, "visualization.text.color_slip"))
        c_moving = tuple(require(self.config, "visualization.text.color_moving"))
        c_track = tuple(require(self.config, "visualization.text.color_tracking"))

        if slip_status == 1:
            color = c_slip
        elif moving > self.moving_count_thresh:
            color = c_moving
        else:
            color = c_track

        text = f"Slip Prob: {r_val:.2f} | Slip: {slip_status} | Moves: {moving}"
        draw_text(display, text, pos, color, scale, thickness)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Realtime tactile tracking WITH SLIP DETECTION")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml",
                        help="Path to YAML configuration")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")

    RealtimeSlipTracking(config=config).run()


if __name__ == "__main__":
    main()
