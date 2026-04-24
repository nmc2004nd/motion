"""Context manager cho camera: mở, warmup, iterate frame, tự release."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from ..config import require


class CameraSource:
    """Wrapper `cv2.VideoCapture` dạng context manager.

    Usage:
        with CameraSource(config) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, config: dict) -> None:
        self.device_id = int(require(config, "camera.device_id"))
        self.warmup_frames = int(require(config, "camera.warmup_frames"))
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "CameraSource":
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở camera: {self.device_id}")
        for _ in range(self.warmup_frames):
            self._cap.read()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield từng frame BGR. Dừng khi camera không đọc được."""
        assert self._cap is not None, "CameraSource chưa được mở (dùng trong `with`)."
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame
