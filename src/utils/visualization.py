"""Bước trực quan hóa với mũi tên thưa."""

import cv2
import numpy as np


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 4: TRỰC QUAN HÓA trường flow
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
def visualize_flow_arrows(
    img: np.ndarray,
    ref_pts: np.ndarray,
    def_pts: np.ndarray,
    valid: np.ndarray,
    config: dict = None,
    save_path: str | None = None,
) -> np.ndarray:
    """Vẽ mũi tên từ vị trí tham chiếu đến vị trí đã biến dạng (có scale)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    if config is None:
        config = {}
    
    viz_cfg = config.get("visualization", {})
    scale = viz_cfg.get("arrow_scale", 2.0)
    mot_thresh = viz_cfg.get("motion_magnitude_threshold", 0.3)
    c_non_moving = tuple(viz_cfg.get("arrow_non_moving_color", [180, 180, 180]))
    c_start_maker = tuple(viz_cfg.get("start_marker_color", [255, 255, 0]))
    intensity_multi = viz_cfg.get("color_intensity_multiplier", 20)
    arr_thick = viz_cfg.get("arrow_thickness", 1)
    arr_tip = viz_cfg.get("arrow_tip_length", 0.3)

    for i in range(len(ref_pts)):
        if not valid[i]:
            continue

        p0 = ref_pts[i]
        p1 = def_pts[i]

        # Scale độ dịch chuyển để dễ quan sát.
        disp = p1 - p0
        mag = np.linalg.norm(disp)

        # Bỏ qua chuyển động quá nhỏ.
        if mag < mot_thresh:
            cv2.circle(vis, tuple(p0.astype(int)), 2, c_non_moving, -1)
            continue

        start = p0.astype(int)
        end = (p0 + disp * scale).astype(int)
        color_intensity = min(int(mag * intensity_multi), 255)

        # Tô màu theo biên độ (đỏ hơn khi độ dịch chuyển lớn hơn).
        color = (0, 255 - color_intensity, color_intensity)

        cv2.arrowedLine(vis, tuple(start), tuple(end), color, arr_thick, tipLength=arr_tip)
        cv2.circle(vis, tuple(start), 2, c_start_maker, -1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis
