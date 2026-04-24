"""Khung chung cho mọi pipeline realtime: camera loop + capture/clear reference.

Các pipeline cụ thể (tracking only, slip v1, slip v2) chỉ cần override
`process_frame` (và tuỳ chọn `draw_idle`, `window_name`, `startup_lines`,
`on_reference_captured`, `on_reference_cleared`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..common.camera import CameraSource
from ..common.keyboard import KeyboardController, KeyCommand
from ..common.overlay import draw_fps, draw_idle_prompt
from ..common.perf import FpsMeter
from ..core.detection import create_blob_detector, detect_markers
from ..core.preprocessing import make_clahe, preprocess

logger = logging.getLogger(__name__)


class BaseRealtimePipeline(ABC):
    """Abstract cho pipeline realtime. Quản lý vòng đời camera + reference."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.camera = CameraSource(config)
        self.keyboard = KeyboardController(config)
        self.fps_meter = FpsMeter()
        self._clahe = make_clahe(config)
        self._blob_detector = create_blob_detector(config)

        self.reference_image: np.ndarray | None = None
        self.reference_markers: np.ndarray | None = None

    # ---------- Subclass hooks ----------

    @property
    @abstractmethod
    def window_name(self) -> str:
        """Tiêu đề cửa sổ OpenCV chính."""

    @property
    def startup_lines(self) -> list[str]:
        """Các dòng log in khi bắt đầu chạy."""
        return [
            f"=== BẮT ĐẦU CAMERA {self.camera.device_id} ===",
            "- Nhấn 'r' để chụp frame hiện tại làm Reference (Trạng thái tĩnh).",
            "- Nhấn 'c' để xóa Reference frame.",
            "- Nhấn 'q' để thoát.",
        ]

    def on_reference_captured(self, gray: np.ndarray) -> None:
        """Hook gọi sau khi reference_image + reference_markers đã cập nhật."""

    def on_reference_cleared(self) -> None:
        """Hook gọi sau khi reference đã bị xoá."""

    @abstractmethod
    def process_frame(self, gray: np.ndarray, display: np.ndarray, fps: float) -> None:
        """Xử lý 1 frame khi đã có reference. Subclass chịu trách nhiệm vẽ lên `display`."""

    def draw_idle(self, display: np.ndarray, fps: float) -> None:
        """Vẽ overlay khi chưa có reference. Override nếu muốn kiểu khác."""
        draw_idle_prompt(display, self.config)
        draw_fps(display, fps, self.config, color_key="visualization.text.color_idle")

    # ---------- Main loop ----------

    def run(self) -> None:
        for line in self.startup_lines:
            logger.info(line)

        with self.camera as cam:
            for frame in cam.frames():
                cmd = self.keyboard.poll()
                if cmd is KeyCommand.QUIT:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
                fps = self.fps_meter.tick()

                if cmd is KeyCommand.CAPTURE:
                    self._capture_reference(gray)
                elif cmd is KeyCommand.CLEAR:
                    self._clear_reference()

                if self.reference_image is not None and self.reference_markers is not None:
                    self.process_frame(gray, display, fps)
                else:
                    self.draw_idle(display, fps)

                cv2.imshow(self.window_name, display)

    # ---------- Reference lifecycle ----------

    def _capture_reference(self, gray: np.ndarray) -> None:
        self.reference_image = gray.copy()
        ref_proc = preprocess(self.reference_image, config=self.config, _clahe=self._clahe)
        self.reference_markers, _ = detect_markers(
            ref_proc, config=self.config, _detector=self._blob_detector
        )
        logger.info("Đã chụp ảnh tham chiếu: Phát hiện %d markers.", len(self.reference_markers))
        self.on_reference_captured(gray)

    def _clear_reference(self) -> None:
        self.reference_image = None
        self.reference_markers = None
        self.on_reference_cleared()
        logger.info("Đã xóa ảnh tham chiếu.")
