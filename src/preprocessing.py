"""Bước tiền xử lý ảnh cho bài toán theo dõi marker tactile."""

import cv2
import numpy as np


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 1: TIỀN XỬ LÝ — khử chiếu sáng không đồng đều (vignetting)
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
def preprocess(img_gray: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa độ chiếu sáng bằng phép trừ nền.
    LED thường gây sáng mạnh ở giữa và tối dần về rìa.
    Ta ước lượng nền chiếu sáng rồi loại bỏ nó.
    """
    # Gaussian blur kernel lớn để xấp xỉ nền chiếu sáng.
    background = cv2.GaussianBlur(img_gray, (101, 101), 0)

    # Trừ nền và chuẩn hóa lại dải cường độ để marker nổi bật hơn.
    normalized = cv2.subtract(img_gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE tăng tương phản cục bộ, hữu ích ở vùng có tương phản thấp.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized.astype(np.uint8))
    return enhanced

# if __name__ == "__main__":
#     # Test nhanh trên một ảnh mẫu.
#     img = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
#     proc = preprocess(img)
#     cv2.imwrite("outputs/step_1/sample_ref_preprocessed.png", proc)
#     print("Preprocessing test done, output saved to outputs/step_1/sample_ref_preprocessed.png")
