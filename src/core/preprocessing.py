"""Tiền xử lý ảnh — khử chiếu sáng không đồng đều (vignetting).

Pipeline: background subtraction (box blur) → min-max normalize → CLAHE.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import as_int_tuple, require


def preprocess(
    img_gray: np.ndarray,
    config: dict,
    _clahe: cv2.CLAHE | None = None,
) -> np.ndarray:
    """Chuẩn hóa độ chiếu sáng và tăng tương phản cục bộ.

    Args:
        img_gray: ảnh xám đầu vào (H, W) uint8.
        config: cấu hình đã load qua `load_config`.
        _clahe: đối tượng CLAHE tạo sẵn để tái sử dụng (tránh tạo mới mỗi frame).

    Returns:
        Ảnh uint8 đã tăng cường, cùng kích thước đầu vào.
    """
    blur_kernel = as_int_tuple(require(config, "preprocessing.blur_kernel"), "blur_kernel")
    clip_limit = float(require(config, "preprocessing.clahe_clip_limit"))
    grid_size = as_int_tuple(require(config, "preprocessing.clahe_grid"), "clahe_grid")

    # Box filter (integral image, O(1)/pixel) nhanh hơn GaussianBlur 101×101 ~10–20x
    # mà không tạo grid artifacts.
    background = cv2.blur(img_gray, blur_kernel)
    normalized = cv2.subtract(img_gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    clahe = _clahe if _clahe is not None else cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(normalized.astype(np.uint8))


def make_clahe(config: dict) -> cv2.CLAHE:
    """Tạo CLAHE object theo config (dùng để cache ngoài vòng lặp realtime)."""
    clip_limit = float(require(config, "preprocessing.clahe_clip_limit"))
    grid_size = as_int_tuple(require(config, "preprocessing.clahe_grid"), "clahe_grid")
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
