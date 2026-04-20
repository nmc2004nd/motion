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
    scale: float = 2.0,
    save_path: str | None = None,
) -> np.ndarray:
    """Vẽ mũi tên từ vị trí tham chiếu đến vị trí đã biến dạng (có scale)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    for i in range(len(ref_pts)):
        if not valid[i]:
            continue

        p0 = ref_pts[i]
        p1 = def_pts[i]

        # Scale độ dịch chuyển để dễ quan sát.
        disp = p1 - p0
        mag = np.linalg.norm(disp)

        # Bỏ qua chuyển động quá nhỏ.
        if mag < 0.3:
            cv2.circle(vis, tuple(p0.astype(int)), 2, (180, 180, 180), -1)
            continue

        start = p0.astype(int)
        end = (p0 + disp * scale).astype(int)
        color_intensity = min(int(mag * 20), 255)

        # Tô màu theo biên độ (đỏ hơn khi độ dịch chuyển lớn hơn).
        color = (0, 255 - color_intensity, color_intensity)

        cv2.arrowedLine(vis, tuple(start), tuple(end), color, 1, tipLength=0.3)
        cv2.circle(vis, tuple(start), 2, (255, 255, 0), -1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


def visualize_flow_hsv(
    ref_pts: np.ndarray,
    def_pts: np.ndarray,
    valid: np.ndarray,
    img_shape: tuple[int, int],
    save_path: str | None = None,
    interpolation_neighbors: int = 6,
) -> np.ndarray:
    """
    Trực quan hóa HSV giống ảnh mục tiêu.
    Hue biểu diễn hướng, Saturation biểu diễn biên độ.
    Tạo trường flow dày bằng nội suy từ marker thưa.
    """
    h, w = img_shape[:2]
    disp = def_pts - ref_pts
    valid_pts = ref_pts[valid]
    valid_disp = disp[valid]

    if valid_pts.shape[0] == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        if save_path:
            cv2.imwrite(save_path, empty)
        return empty

    # Tạo lưới dày toàn ảnh.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Nội suy theo trọng số khoảng cách nghịch đảo từ các marker lân cận.
    k = max(1, min(interpolation_neighbors, valid_pts.shape[0]))
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
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(mag * 25, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Chồng thêm mũi tên thưa ở các marker hợp lệ.
    for i in range(len(ref_pts)):
        if not valid[i]:
            continue
        p0 = ref_pts[i].astype(int)
        p1 = (ref_pts[i] + (def_pts[i] - ref_pts[i]) * 3).astype(int)
        cv2.arrowedLine(bgr, tuple(p0), tuple(p1), (0, 0, 0), 1, tipLength=0.3)
        cv2.circle(bgr, tuple(p0), 1, (0, 0, 0), -1)

    if save_path:
        cv2.imwrite(save_path, bgr)

    return bgr
