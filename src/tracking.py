"""Bước thiết lập tương ứng marker bằng optical flow Lucas-Kanade."""

import cv2
import numpy as np
from src.preprocessing import preprocess
from src.detection import detect_markers


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 3: TẠO TƯƠNG ỨNG bằng optical flow
# Thay vì phát hiện lại ở frame 2 và hy vọng thứ tự trùng khớp,
# ta TRACK từng marker tham chiếu bằng Lucas-Kanade.
# Cách này đảm bảo tương ứng 1-1 theo từng marker.
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
def track_markers_lk(
    img_ref: np.ndarray,
    img_def: np.ndarray,
    ref_points: np.ndarray,
    fb_threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Track marker từ ảnh tham chiếu sang ảnh biến dạng với kiểm tra FB."""
    if ref_points.size == 0:
        return ref_points.copy(), np.zeros((0,), dtype=bool)

    lk_params = dict(
        # Cửa sổ đủ lớn để bắt được chuyển động marker.
        winSize=(21, 21),
        # Kim tự tháp đa mức giúp xử lý dịch chuyển lớn hơn.
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    pts_ref = ref_points.reshape(-1, 1, 2).astype(np.float32)
    pts_def, status, _ = cv2.calcOpticalFlowPyrLK(
        img_ref, img_def, pts_ref, None, **lk_params
    )

    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        img_def, img_ref, pts_def, None, **lk_params
    )

    # Forward-backward check để loại bỏ track không ổn định.
    fb_error = np.linalg.norm(pts_back.reshape(-1, 2) - ref_points, axis=1)
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < fb_threshold)
    return pts_def.reshape(-1, 2), valid

if __name__ == "__main__":
    # Test nhanh trên một cặp ảnh mẫu.
    img_ref = cv2.imread("/home/nmc/ManhCuong/motion/data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
    img_def = cv2.imread("/home/nmc/ManhCuong/motion/data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

    ref_proc = preprocess(img_ref)
    def_proc = preprocess(img_def)

    # dùng dectection không cần giả sử
    ref_markers, _ = detect_markers(ref_proc)

    tracked_pts, valid = track_markers_lk(ref_proc, def_proc, ref_markers)
    print(f"Tracked points: {len(tracked_pts)}\n ref_markers : {len(ref_markers)}\n")
    print(f"Tracked points:\n{tracked_pts}\nValid mask:\n{valid}")