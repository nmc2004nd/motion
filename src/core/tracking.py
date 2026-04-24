"""Theo dõi marker bằng Pyramid Lucas-Kanade + forward-backward check.

Hợp nhất hai biến thể từ codebase cũ (`pyr_lk.track_markers_lk` và
`slip_v2.tracking.track_lk_nodeadzone`) qua flag `apply_deadzone`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import as_int_tuple, require


def track_markers_lk(
    img_ref: np.ndarray,
    img_def: np.ndarray,
    ref_points: np.ndarray,
    config: dict,
    *,
    apply_deadzone: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Track marker từ ảnh tham chiếu sang ảnh biến dạng với FB check.

    Args:
        img_ref, img_def: ảnh xám (H, W) uint8 của ref / deformed.
        ref_points: toạ độ marker tham chiếu (N, 2) float32.
        config: cấu hình pipeline.
        apply_deadzone: nếu True, dịch chuyển < `tracking.min_displacement` sẽ bị
            reset về vị trí tham chiếu (triệt tiêu nhiễu đàn hồi). Slip V2 dùng
            False để giữ tín hiệu slip chậm tích luỹ.

    Returns:
        (pts_def (N, 2) float32, valid mask (N,) bool).
    """
    if ref_points.size == 0:
        return ref_points.copy(), np.zeros((0,), dtype=bool)

    win_size = as_int_tuple(require(config, "tracking.pyrlk.win_size"), "pyrlk.win_size")
    max_level = int(require(config, "tracking.pyrlk.max_level"))
    fb_threshold = float(require(config, "tracking.pyrlk.fb_threshold"))
    max_iter = int(require(config, "tracking.pyrlk.term_criteria.max_iter"))
    eps = float(require(config, "tracking.pyrlk.term_criteria.eps"))

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps),
    )

    pts_ref = ref_points.reshape(-1, 1, 2).astype(np.float32)
    pts_def, status_f, _ = cv2.calcOpticalFlowPyrLK(img_ref, img_def, pts_ref, None, **lk_params)
    pts_back, status_b, _ = cv2.calcOpticalFlowPyrLK(img_def, img_ref, pts_def, None, **lk_params)

    fb_error = np.linalg.norm(pts_back.reshape(-1, 2) - ref_points, axis=1)
    valid = (status_f.flatten() == 1) & (status_b.flatten() == 1) & (fb_error < fb_threshold)

    pts_def = pts_def.reshape(-1, 2)
    if apply_deadzone:
        min_disp = float(require(config, "tracking.min_displacement"))
        disps = np.linalg.norm(pts_def - ref_points, axis=1)
        pts_def[disps < min_disp] = ref_points[disps < min_disp]

    return pts_def, valid
