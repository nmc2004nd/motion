import cv2
import numpy as np

def update_files():
    with open("src/tracking.py", "w") as f:
        f.write('''"""Bước thiết lập tương ứng marker bằng Hungarian và ràng buộc vật lý."""

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
    max_disp: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tìm tương ứng marker từ ảnh tham chiếu sang ảnh biến dạng bằng Hungarian."""
    if ref_points.size == 0 or def_points.size == 0:
        return ref_points.copy(), np.zeros(len(ref_points), dtype=bool)

    h, w = img_shape[:2]
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    # Ma trận khoảng cách Euclidean
    D = distance_matrix(ref_points, def_points)

    # Ràng buộc vật lý: Khi bị nén/nhấn, marker tỏa ra từ tâm (Radial Expansion)
    # Ta thêm hình phạt (penalty) nặng cho các chuyển động hướng ngược vào trong.
    radial_dirs = ref_points - center
    norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
    norms[norms < 1e-5] = 1.0
    radial_dirs = radial_dirs / norms

    # Vector dịch chuyển: (N, M, 2)
    disps = def_points[None, :, :] - ref_points[:, None, :]
    # Tính tích vô hướng để xem hướng di chuyển: (N, M)
    dot_prods = np.sum(disps * radial_dirs[:, None, :], axis=2)

    # Áp dụng penalty nếu dot_prod âm (chuyển động hướng vào tâm quá mức)
    inward_mask = dot_prods < -2.0  # Cho phép nhiễu dịch chuyển nhỏ ~2px
    D[inward_mask] += np.abs(dot_prods[inward_mask]) * 5.0

    # Tối ưu hóa phân công toàn cục (Global Assignment)
    row_ind, col_ind = linear_sum_assignment(D)

    tracked_points = ref_points.copy()
    valid = np.zeros(len(ref_points), dtype=bool)

    for r, c in zip(row_ind, col_ind):
        actual_dist = np.linalg.norm(def_points[c] - ref_points[r])
        # Loại bỏ các match quá xa (do thiếu marker)
        if actual_dist < max_disp:
            tracked_points[r] = def_points[c]
            valid[r] = True

    return tracked_points, valid
''')
update_files()
print("Updated tracking.py")
