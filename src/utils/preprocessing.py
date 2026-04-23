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

    # Ước lượng nền chiếu sáng bằng box filter (integral image → O(1)/pixel).
    # Nhanh hơn GaussianBlur(101×101) ~10-20x và không tạo grid artifacts
    # như cách downsample+upsample, giữ nguyên chất lượng phát hiện marker.
    background = cv2.blur(img_gray, blur_kernel)

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
