"""Pipeline realtime + SlipDetectorV2 (translation / radial decomposition)."""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

from ..common.overlay import (
    draw_slip_hud_arrow,
    draw_slip_score_bar,
    draw_text,
)
from ..common.perf import TimingAccumulator
from ..config import load_config, require
from ..config.loader import ConfigError
from ..core.tracking import track_markers_lk
from ..core.visualization import visualize_flow_arrows
from ..slip.v2 import SlipDetectorV2
from .base import BaseRealtimePipeline

logger = logging.getLogger(__name__)

_FLOW_WINDOW = "Flow Arrows (v2)"


class RealtimeSlipV2Pipeline(BaseRealtimePipeline):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.slip_detector = SlipDetectorV2(config=config)

    @property
    def window_name(self) -> str:
        return f"Slip V2 (/dev/video{self.camera.device_id})"

    @property
    def startup_lines(self) -> list[str]:
        return [
            f"=== Slip V2 pipeline | camera {self.camera.device_id} ===",
            "r: reference | c: clear | q: quit",
        ]

    def on_reference_captured(self, gray: np.ndarray) -> None:
        self.slip_detector.reset()

    def on_reference_cleared(self) -> None:
        self.slip_detector.reset()

    def draw_idle(self, display: np.ndarray, fps: float) -> None:
        # Overlay idle riêng cho V2 (scale + màu khác với base).
        cfg = self.config
        main_pos = tuple(require(cfg, "visualization.overlay_v2.idle_main_pos"))
        fps_pos = tuple(require(cfg, "visualization.overlay_v2.idle_fps_pos"))
        scale_main = float(require(cfg, "visualization.overlay_v2.font_scale_idle"))
        scale_fps = float(require(cfg, "visualization.overlay_v2.font_scale_idle_fps"))
        thickness = int(require(cfg, "visualization.text.thickness"))
        c_main = tuple(require(cfg, "visualization.overlay_v2.idle_color"))
        c_fps = tuple(require(cfg, "visualization.overlay_v2.idle_fps_color"))

        draw_text(display, "Press 'r' to capture reference", main_pos, c_main, scale_main, thickness)
        draw_text(display, f"FPS {fps:.1f}", fps_pos, c_fps, scale_fps, thickness)

    def process_frame(self, gray: np.ndarray, display: np.ndarray, fps: float) -> None:
        n = len(self.reference_markers)  # type: ignore[arg-type]
        if n == 0:
            self.draw_idle(display, fps)
            return

        timings = TimingAccumulator()
        with timings.measure("track"):
            tracked, valid = track_markers_lk(
                self.reference_image, gray, self.reference_markers,
                config=self.config, apply_deadzone=False,
            )
        with timings.measure("detect"):
            info = self.slip_detector.update(
                ref_markers=self.reference_markers, curr_markers=tracked, valid_mask=valid,
            )

        if valid.any():
            vis = visualize_flow_arrows(
                gray, self.reference_markers, tracked, valid,
                config=self.config, save_path=None,
            )
            cv2.imshow(_FLOW_WINDOW, vis)

        ts = timings.snapshot()
        self._draw_overlay(display, info, fps, ts.get("track", 0.0), ts.get("detect", 0.0))

    def _draw_overlay(
        self,
        display: np.ndarray,
        info: dict,
        fps: float,
        track_ms: float,
        det_ms: float,
    ) -> None:
        cfg = self.config
        thickness = int(require(cfg, "visualization.text.thickness"))
        scale_main = float(require(cfg, "visualization.overlay_v2.font_scale_main"))
        scale_detail = float(require(cfg, "visualization.overlay_v2.font_scale_detail"))
        scale_perf = float(require(cfg, "visualization.overlay_v2.font_scale_perf"))
        main_pos = tuple(require(cfg, "visualization.overlay_v2.main_pos"))
        detail_pos = tuple(require(cfg, "visualization.overlay_v2.detail_pos"))
        perf_pos = tuple(require(cfg, "visualization.overlay_v2.perf_pos"))
        perf_color = tuple(require(cfg, "visualization.overlay_v2.perf_color"))

        color = self._pick_phase_color(info)

        tx, ty = info["translation"]
        score = info["slip_score"]
        raw = info["raw_score"]
        is_slip = info["is_slip"]
        moving = info["moving_count"]
        coh = info["coherence"]
        scale = info["scale"]
        rr = info.get("radial_rate", 0.0)
        phase = info["phase"]

        draw_text(
            display,
            f"SLIP={'YES' if is_slip else 'no'}  score={score:.2f}  raw={raw:.2f}  phase={phase}",
            main_pos, color, scale_main, thickness,
        )
        draw_text(
            display,
            f"|t|={np.hypot(tx, ty):.2f}px  dir=({tx:+.2f},{ty:+.2f})  "
            f"R={coh:.2f}  n={moving}  scale={scale}  rad={rr:+.3f}",
            detail_pos, color, scale_detail, thickness,
        )
        draw_text(
            display,
            f"FPS {fps:5.1f} | track {track_ms:4.1f}ms | detect {det_ms:4.1f}ms",
            perf_pos, perf_color, scale_perf, 1,
        )

        draw_slip_hud_arrow(display, (tx, ty), color, cfg)
        draw_slip_score_bar(
            display, score,
            self.slip_detector.score_on, self.slip_detector.score_off,
            color, cfg,
        )

    def _pick_phase_color(self, info: dict) -> tuple[int, int, int]:
        cfg = self.config
        key_base = "visualization.overlay_v2.phase_colors"
        if info["is_slip"]:
            return tuple(require(cfg, f"{key_base}.slip"))
        if info["phase"] == "pressing":
            return tuple(require(cfg, f"{key_base}.pressing"))
        if info["phase"] == "releasing":
            return tuple(require(cfg, f"{key_base}.releasing"))
        if info["slip_score"] > self.slip_detector.score_off:
            return tuple(require(cfg, f"{key_base}.alert"))
        return tuple(require(cfg, f"{key_base}.tracking"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Realtime tactile slip detection v2")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")

    RealtimeSlipV2Pipeline(config=config).run()


if __name__ == "__main__":
    main()
