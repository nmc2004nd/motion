"""Bước thiết lập tương ứng marker bằng Hungarian và ràng buộc vật lý."""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 3: TẠO TƯƠNG ỨNG bằng Hungarian Algorithm
# Thay vì dùng Optical Flow (LK) dễ bị "nhảy" marker lân cận khi độ biến dạng lớn hơn 
# một nửa khoảng cách grid (gây ra hiện tượng marker hướng ngược vào trong sai vật lý),
# ta dùng thuật toán Hungarian trên các marker đã detect với ràng buộc Center-Radial.
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
def match_markers_robust(
    ref_points: np.ndarray,
    def_points: np.ndarray,
    img_shape: tuple[int, int],
    config: dict = None,
    force_center: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tìm tương ứng marker từ ảnh tham chiếu sang ảnh biến dạng bằng Hungarian.
    Có thể truyền vào force_center (tâm lực nhấn) nếu có. Nếu không, thuật toán
    sẽ tự động xác định thông qua một lần chạy Hungarian thô (2-pass).
    """
    if ref_points.size == 0 or def_points.size == 0:
        return ref_points.copy(), np.zeros(len(ref_points), dtype=bool)

    if config is None:
        config = {
            "tracking": {
                "min_displacement": 1.5,
                "hungarian": {
                    "max_displacement": 60.0,
                    "top_points_ratio": 0.1,
                    "inward_penalty_threshold": -2.0,
                    "penalty_multiplier": 5.0
                }
            }
        }
        
    track_cfg = config.get("tracking", {})
    hg_cfg = track_cfg.get("hungarian", {})
    
    max_disp = hg_cfg.get("max_displacement", 60.0)
    min_disp = track_cfg.get("min_displacement", 1.5)
    top_points_ratio = hg_cfg.get("top_points_ratio", 0.1)
    inward_penalty_thresh = hg_cfg.get("inward_penalty_threshold", -2.0)
    penalty_multi = hg_cfg.get("penalty_multiplier", 5.0)

    # Ma trận khoảng cách Euclidean cơ bản
    D_base = distance_matrix(ref_points, def_points)

    if force_center is None:
        # Pass 1: Chạy Hungarian thô không có ràng buộc lực để tìm các vector dịch chuyển
        row_ind_raw, col_ind_raw = linear_sum_assignment(D_base)
        
        # Lấy các vector dịch chuyển
        disps_raw = def_points[col_ind_raw] - ref_points[row_ind_raw]
        disp_mags = np.linalg.norm(disps_raw, axis=1)
        
        # Tự động tìm tâm lực: lấy tọa độ trung bình của top x% các điểm dịch chuyển mạnh nhất
        # (Khi bị nhấn, vùng quanh tâm lực sẽ có chuyển động lớn nhất)
        n_top = max(1, int(len(disp_mags) * top_points_ratio))
        top_indices = np.argsort(disp_mags)[-n_top:]
        
        # Tâm lực tự động là trung bình của các điểm biến dạng mạnh nhất
        center = np.mean(ref_points[row_ind_raw[top_indices]], axis=0)
    else:
        center = np.array(force_center, dtype=np.float32)

    D = D_base.copy()

    # Ràng buộc vật lý: Khi bị nén/nhấn, marker tỏa ra từ tâm (Radial Expansion)
    # Ta thêm hình phạt (penalty) nặng cho các chuyển động hướng ngược vào trong.
    radial_dirs = ref_points - center
    norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
    norms[norms < 1e-5] = 1.0
    radial_dirs = radial_dirs / norms

    # Tính tích vô hướng tối ưu bằng phép nhân ma trận thay vì dùng mảng 3D
    # (N, 2) @ (2, M) -> (N, M)
    # def_points.T có shape (2, M)
    # Phép tính tương đương: (def_points - ref_points) @ radial_dirs
    dot_prods = radial_dirs @ def_points.T - np.sum(ref_points * radial_dirs, axis=1, keepdims=True)

    # Áp dụng penalty nếu dot_prod âm (chuyển động hướng vào tâm quá mức)
    inward_mask = dot_prods < inward_penalty_thresh  # Cho phép nhiễu dịch chuyển nhỏ ~2px
    D[inward_mask] += np.abs(dot_prods[inward_mask]) * penalty_multi

    # Tối ưu hóa phân công toàn cục (Global Assignment)
    row_ind, col_ind = linear_sum_assignment(D)

    tracked_points = ref_points.copy()
    valid = np.zeros(len(ref_points), dtype=bool)

    for r, c in zip(row_ind, col_ind):
        actual_dist = np.linalg.norm(def_points[c] - ref_points[r])
        # Loại bỏ các match quá xa (do thiếu marker)
        if actual_dist < max_disp:
            if actual_dist < min_disp:
                tracked_points[r] = ref_points[r]
            else:
                tracked_points[r] = def_points[c]
            valid[r] = True

    return tracked_points, valid

if __name__ == "__main__":
    import cv2
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.preprocessing import preprocess
    from utils.detection import detect_markers

    # Test nhanh trên một cặp ảnh mẫu.
    img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
    img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

    ref_proc = preprocess(img_ref)
    def_proc = preprocess(img_def)

    ref_markers, _ = detect_markers(ref_proc)
    def_markers, _ = detect_markers(def_proc)

    tracked_pts, valid = match_markers_robust(ref_markers, def_markers, img_ref.shape)
    print(f"Tracked points: {len(tracked_pts)}\n ref_markers : {len(ref_markers)}\n")
    print(f"Valid ratio: {valid.sum()}/{len(valid)}")
