"""Phát hiện marker bằng OpenCV SimpleBlobDetector.

SimpleBlobDetector ổn định hơn ngưỡng + centroid vì tích hợp sẵn bộ lọc diện
tích, độ tròn, quán tính và độ lồi.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import require


@dataclass(frozen=True)
class BlobDetectorConfig:
    min_threshold: int
    max_threshold: int
    threshold_step: int
    min_area: float
    max_area: float
    min_circularity: float
    min_inertia_ratio: float
    min_convexity: float

    @classmethod
    def from_config(cls, config: dict) -> "BlobDetectorConfig":
        return cls(
            min_threshold=int(require(config, "detection.min_threshold")),
            max_threshold=int(require(config, "detection.max_threshold")),
            threshold_step=int(require(config, "detection.step")),
            min_area=float(require(config, "detection.min_area")),
            max_area=float(require(config, "detection.max_area")),
            min_circularity=float(require(config, "detection.min_circularity")),
            min_inertia_ratio=float(require(config, "detection.min_inertia")),
            min_convexity=float(require(config, "detection.min_convexity")),
        )


def create_blob_detector(config: dict) -> cv2.SimpleBlobDetector:
    """Tạo detector blob cho marker sáng trên nền tối."""
    cfg = BlobDetectorConfig.from_config(config)
    """
    detection:
    min_threshold: 50
    max_threshold: 220
    step: 10
    min_area: 30.0
    max_area: 500.0
    min_circularity: 0.5
    min_inertia: 0.3
    min_convexity: 0.7
    """

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = cfg.min_threshold
    params.maxThreshold = cfg.max_threshold
    params.thresholdStep = cfg.threshold_step

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = cfg.min_area
    params.maxArea = cfg.max_area

    params.filterByCircularity = True
    params.minCircularity = cfg.min_circularity

    params.filterByInertia = True
    params.minInertiaRatio = cfg.min_inertia_ratio

    params.filterByConvexity = True
    params.minConvexity = cfg.min_convexity

    return cv2.SimpleBlobDetector_create(params)


def detect_markers(
    img_processed: np.ndarray,
    config: dict,
    _detector: cv2.SimpleBlobDetector | None = None,
) -> tuple[np.ndarray, list[cv2.KeyPoint]]:
    """Trả về tâm marker (Nx2 float32) và danh sách keypoint OpenCV.

    Args:
        img_processed: ảnh xám đã tiền xử lý.
        config: cấu hình pipeline.
        _detector: detector tạo sẵn để tái sử dụng giữa các frame realtime.
    """
    detector = _detector if _detector is not None else create_blob_detector(config=config)
    keypoints = detector.detect(img_processed)
    centers = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return centers, keypoints
