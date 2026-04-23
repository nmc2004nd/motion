"""Bước trực quan hóa với mũi tên thưa và trường HSV dày."""

import cv2
import numpy as np
from scipy.spatial import cKDTree


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


def build_hsv_interp_cache(
    ref_pts: np.ndarray,
    img_shape: tuple,
    config: dict = None,
) -> dict | None:
    """Pre-compute IDW interpolation weights trên grid thu nhỏ.

    Gọi 1 lần khi chụp reference. Kết quả được truyền vào ``visualize_flow_hsv``
    qua ``_interp_cache`` để bỏ qua bước KD-tree query tốn kém mỗi frame.

    Grid được tính ở 1/(interp_scale) resolution rồi upsample — giảm số điểm
    query từ H*W xuống (H/s)*(W/s), tức s² lần ít hơn (mặc định 16x).
    """
    if config is None:
        config = {}
    if len(ref_pts) == 0:
        return None

    h, w = img_shape[:2]
    hsv_cfg = config.get("visualization", {}).get("hsv", {})
    interp_neighbors = hsv_cfg.get("interpolation_neighbors", 6)
    scale = hsv_cfg.get("interp_scale", 4)  # downsample factor cho grid nội suy

    hs, ws = max(1, h // scale), max(1, w // scale)

    # Grid pixel ở resolution thấp (x, y)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # KD-tree dựa trên ref_pts đã scale xuống
    scaled_pts = ref_pts / scale
    k = max(1, min(interp_neighbors, len(ref_pts)))
    tree = cKDTree(scaled_pts)
    dists, idxs = tree.query(grid_pts, k=k)

    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)

    return {
        "idxs": idxs,
        "weights": weights.astype(np.float32),
        "hs": hs, "ws": ws,
        "h": h, "w": w,
        "n": len(ref_pts),
    }


def visualize_flow_hsv(
    ref_pts: np.ndarray,
    def_pts: np.ndarray,
    valid: np.ndarray,
    img_shape: tuple[int, int],
    config: dict = None,
    save_path: str | None = None,
    _interp_cache: dict | None = None,
) -> np.ndarray:
    """
    Trực quan hóa HSV giống ảnh mục tiêu.
    Hue biểu diễn hướng, Saturation biểu diễn biên độ.
    Tạo trường flow dày bằng nội suy từ marker thưa.

    Args:
        _interp_cache: output của ``build_hsv_interp_cache`` — bỏ qua KD-tree
            query tốn kém, chỉ tính phép nhân weighted sum mỗi frame.
    """
    h, w = img_shape[:2]

    if config is None:
        config = {}

    hsv_cfg = config.get("visualization", {}).get("hsv", {})
    interp_neighbors = hsv_cfg.get("interpolation_neighbors", 6)
    hue_divisor = hsv_cfg.get("hue_divisor", 2)
    sat_multi = hsv_cfg.get("saturation_multiplier", 25)
    hsv_value = hsv_cfg.get("value", 255)
    start_c = tuple(hsv_cfg.get("start_marker_color", [0, 0, 0]))
    start_r = hsv_cfg.get("start_marker_radius", 1)
    disp_multi = hsv_cfg.get("multiplier", 3)
    arr_thick = config.get("visualization", {}).get("arrow_thickness", 1)
    arr_tip = config.get("visualization", {}).get("arrow_tip_length", 0.3)

    valid_pts = ref_pts[valid]
    if valid_pts.shape[0] == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        if save_path:
            cv2.imwrite(save_path, empty)
        return empty

    # --- Tính trường displacement dày ---
    use_cache = _interp_cache is not None and _interp_cache.get("n") == len(ref_pts)

    if use_cache:
        # Fast path: dùng weights pre-computed, chỉ tính dot product.
        # Invalid markers được gán displacement = 0 (không có chuyển động đo được).
        idxs = _interp_cache["idxs"]
        weights = _interp_cache["weights"]
        hs, ws_small = _interp_cache["hs"], _interp_cache["ws"]

        disp_all = np.zeros_like(ref_pts)
        disp_all[valid] = def_pts[valid] - ref_pts[valid]

        dx_s = (disp_all[idxs, 0] * weights).sum(axis=1).reshape(hs, ws_small)
        dy_s = (disp_all[idxs, 1] * weights).sum(axis=1).reshape(hs, ws_small)

        dx_dense = cv2.resize(dx_s, (w, h), interpolation=cv2.INTER_LINEAR)
        dy_dense = cv2.resize(dy_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Slow path (batch hoặc lần đầu chưa có cache).
        disp = def_pts - ref_pts
        valid_disp = disp[valid]

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

        k = max(1, min(interp_neighbors, valid_pts.shape[0]))
        tree = cKDTree(valid_pts)
        dists, idxs = tree.query(grid_pts, k=k)

        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)

        dx_dense = (valid_disp[idxs, 0] * weights).sum(axis=1).reshape(h, w)
        dy_dense = (valid_disp[idxs, 1] * weights).sum(axis=1).reshape(h, w)

    # Chuyển sang HSV để trực quan hướng + độ lớn chuyển động.
    mag, ang = cv2.cartToPolar(dx_dense, dy_dense, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / hue_divisor).astype(np.uint8)
    hsv[..., 1] = np.clip(mag * sat_multi, 0, 255).astype(np.uint8)
    hsv[..., 2] = hsv_value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Chồng thêm mũi tên thưa ở các marker hợp lệ.
    for i in range(len(ref_pts)):
        if not valid[i]:
            continue
        p0 = ref_pts[i].astype(int)
        p1 = (ref_pts[i] + (def_pts[i] - ref_pts[i]) * disp_multi).astype(int)
        cv2.arrowedLine(bgr, tuple(p0), tuple(p1), start_c, arr_thick, tipLength=arr_tip)
        cv2.circle(bgr, tuple(p0), start_r, start_c, -1)

    if save_path:
        cv2.imwrite(save_path, bgr)

    return bgr
