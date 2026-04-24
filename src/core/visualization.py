"""Vẽ trường flow dạng mũi tên thưa."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import require


def visualize_flow_arrows(
    img: np.ndarray,
    ref_pts: np.ndarray,
    def_pts: np.ndarray,
    valid: np.ndarray,
    config: dict,
    save_path: str | None = None,
) -> np.ndarray:
    """Vẽ mũi tên ref → deformed, có scale để dễ quan sát.

    Chi tiết màu / độ dày / ngưỡng ở `visualization.*` của config.
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    scale = float(require(config, "visualization.arrow_scale"))
    mot_thresh = float(require(config, "visualization.motion_magnitude_threshold"))
    c_non_moving = tuple(require(config, "visualization.arrow_non_moving_color"))
    c_start = tuple(require(config, "visualization.start_marker_color"))
    intensity_mult = int(require(config, "visualization.color_intensity_multiplier"))
    arr_thick = int(require(config, "visualization.arrow_thickness"))
    arr_tip = float(require(config, "visualization.arrow_tip_length"))

    for i in range(len(ref_pts)):
        if not valid[i]:
            continue

        p0 = ref_pts[i]
        p1 = def_pts[i]
        disp = p1 - p0
        mag = float(np.linalg.norm(disp))

        if mag < mot_thresh:
            cv2.circle(vis, tuple(p0.astype(int)), 2, c_non_moving, -1)
            continue

        start = tuple(p0.astype(int))
        end = tuple((p0 + disp * scale).astype(int))
        color_intensity = min(int(mag * intensity_mult), 255)
        color = (0, 255 - color_intensity, color_intensity)

        cv2.arrowedLine(vis, start, end, color, arr_thick, tipLength=arr_tip)
        cv2.circle(vis, start, 2, c_start, -1)

    if save_path:
        cv2.imwrite(save_path, vis)
    return vis
