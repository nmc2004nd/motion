"""Bước tiền xử lý ảnh cho bài toán theo dõi marker tactile."""

import cv2
import numpy as np


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 1: TIỀN XỬ LÝ — khử chiếu sáng không đồng đều (vignetting)
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
def preprocess(img_gray: np.ndarray, config: dict = None, _clahe=None) -> np.ndarray:
    """
    Chuẩn hóa độ chiếu sáng bằng phép trừ nền.
    LED thường gây sáng mạnh ở giữa và tối dần về rìa.
    Ta ước lượng nền chiếu sáng rồi loại bỏ nó.

    Args:
        _clahe: CLAHE object được tạo sẵn để tái sử dụng (tránh tạo mới mỗi frame).
    """
    if config is None:
        config = {
            "preprocessing": {
                "blur_kernel": [101, 101],
                "clahe_clip_limit": 2.5,
                "clahe_grid": [8, 8]
            }
        }

    pre_cfg = config.get("preprocessing", {})
    blur_kernel = tuple(pre_cfg.get("blur_kernel", [101, 101]))
    clip_limit = pre_cfg.get("clahe_clip_limit", 2.5)
    grid_size = tuple(pre_cfg.get("clahe_grid", [8, 8]))

    # Ước lượng nền chiếu sáng trên ảnh thu nhỏ 4x để tăng tốc ~16x.
    # Vùng phủ không gian tương đương: sigma gốc ≈ 15.5px, sigma_small * 4 ≈ 16.4px.
    h, w = img_gray.shape[:2]
    ds = 4
    sw, sh = max(16, w // ds), max(16, h // ds)
    small = cv2.resize(img_gray, (sw, sh), interpolation=cv2.INTER_AREA)
    small_bk = max(3, (blur_kernel[0] // ds) | 1)  # kernel lẻ, tối thiểu 3
    bg_small = cv2.GaussianBlur(small, (small_bk, small_bk), 0)
    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Trừ nền và chuẩn hóa lại dải cường độ để marker nổi bật hơn.
    normalized = cv2.subtract(img_gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE tăng tương phản cục bộ. Dùng object được truyền vào nếu có.
    clahe = _clahe if _clahe is not None else cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(normalized.astype(np.uint8))
    return enhanced

# if __name__ == "__main__":
#     # Test nhanh trên một ảnh mẫu.
#     img = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
#     proc = preprocess(img)
#     cv2.imwrite("outputs/step_1/sample_ref_preprocessed.png", proc)
#     print("Preprocessing test done, output saved to outputs/step_1/sample_ref_preprocessed.png")
