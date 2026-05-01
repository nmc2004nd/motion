"""Scalar features từ displacement field cho polynomial regression.

Input: ref_pts (N, 2), disp (N, 2), valid (N,), image_w, image_h.
Output: feature vector (D,) float32 — đã scale-invariant theo image size.

Tất cả thống kê đều mask-aware (chỉ tính trên valid markers). Trả 0 cho frame
không có valid marker — model sẽ học bias term tương ứng.
"""

from __future__ import annotations

import numpy as np

# Tên features theo thứ tự cố định — dùng cho debug + log hệ số polynomial.
FEATURE_NAMES_V1: tuple[str, ...] = (
    "disp_mag_mean",        # mức biến dạng trung bình
    "disp_mag_max",         # peak biến dạng (cứng nhất hoặc gần điểm tải)
    "disp_mag_std",         # độ lệch giữa các marker
    "dx_mean",              # hướng tải trung bình theo x
    "dy_mean",              # hướng tải trung bình theo y
    "disp_mag_sum_norm",    # tổng biến dạng / N_valid (tránh phụ thuộc N)
    "radial_disp_mean",     # nén (-) / giãn (+) trung bình quanh tâm marker grid
    "tangential_disp_mean", # shear / xoắn trung bình
    "valid_ratio",          # độ phủ marker — mất tracking nhiều thì lực ước lượng yếu
)

FEATURE_DIM_V1 = len(FEATURE_NAMES_V1)


def feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "v1":
        return FEATURE_NAMES_V1
    raise ValueError(f"feature_set không hỗ trợ: {feature_set}")


def feature_dim(feature_set: str) -> int:
    return len(feature_names(feature_set))


def compute_features_v1(
    ref_pts: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    image_w: int,
) -> np.ndarray:
    """Tính feature vector cho 1 frame.

    Args:
        ref_pts: (N, 2) float — toạ độ marker tham chiếu.
        disp:    (N, 2) float — dịch chuyển (tracked - ref) theo px.
        valid:   (N,) bool — LK valid mask.
        image_w: bề rộng ảnh (px) — dùng cho cả 2 trục để dx, dy cùng đơn vị
            (radial/tangential mới hợp lệ).

    Returns:
        (FEATURE_DIM_V1,) float32 — scale-invariant với độ phân giải.
    """
    N = ref_pts.shape[0]
    out = np.zeros(FEATURE_DIM_V1, dtype=np.float32)
    if N == 0 or not valid.any():
        return out

    # Normalize displacement và toạ độ theo W (dùng cùng đơn vị cho x và y để
    # phép radial/tangential hợp lệ; vẫn không phụ thuộc vào pixel scale).
    scale = float(image_w)
    dx = disp[:, 0].astype(np.float32) / scale
    dy = disp[:, 1].astype(np.float32) / scale
    mag = np.sqrt(dx * dx + dy * dy)

    # Tâm marker grid — chỉ dùng valid points để không bị lệch khi nhiều marker mất.
    rx_all = ref_pts[:, 0].astype(np.float32) / scale
    ry_all = ref_pts[:, 1].astype(np.float32) / scale
    cx = float(rx_all[valid].mean())
    cy = float(ry_all[valid].mean())
    rx = rx_all - cx
    ry = ry_all - cy
    r_norm = np.sqrt(rx * rx + ry * ry) + 1e-8
    rx_hat = rx / r_norm
    ry_hat = ry / r_norm

    # Radial = dot(disp, r_hat); tangential = cross_z(r_hat, disp) = rx*dy - ry*dx.
    radial = dx * rx_hat + dy * ry_hat
    tangential = rx_hat * dy - ry_hat * dx

    m = valid
    n_valid = int(m.sum())

    out[0] = mag[m].mean()
    out[1] = mag[m].max()
    out[2] = mag[m].std()
    out[3] = dx[m].mean()
    out[4] = dy[m].mean()
    out[5] = mag[m].sum() / max(n_valid, 1)
    out[6] = radial[m].mean()
    out[7] = tangential[m].mean()
    out[8] = n_valid / N
    return out


def compute_features_trial(
    ref_pts: np.ndarray,
    disp_seq: np.ndarray,
    valid_seq: np.ndarray,
    image_w: int,
    feature_set: str = "v1",
) -> np.ndarray:
    """Tính features cho toàn bộ T frame của 1 trial.

    Returns:
        (T, D) float32.
    """
    if feature_set != "v1":
        raise ValueError(f"feature_set không hỗ trợ: {feature_set}")
    T = disp_seq.shape[0]
    out = np.zeros((T, FEATURE_DIM_V1), dtype=np.float32)
    for t in range(T):
        out[t] = compute_features_v1(
            ref_pts, disp_seq[t], valid_seq[t], image_w,
        )
    return out
