"""Bước phát hiện marker bằng OpenCV SimpleBlobDetector."""

from dataclasses import dataclass

import cv2
import numpy as np
from src.preprocessing import preprocess


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# BƯỚC 2: PHÁT HIỆN BLOB với SimpleBlobDetector
# SimpleBlobDetector ổn định hơn ngưỡng + centroid vì có các bộ lọc:
# diện tích, độ tròn, quán tính và độ lồi.
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
@dataclass
class BlobDetectorConfig:
    min_threshold: int = 50
    max_threshold: int = 220
    threshold_step: int = 10
    min_area: float = 30
    max_area: float = 500
    min_circularity: float = 0.5
    min_inertia_ratio: float = 0.3
    min_convexity: float = 0.7


def create_blob_detector(config: BlobDetectorConfig | None = None) -> cv2.SimpleBlobDetector:
    """Tạo detector blob được tinh chỉnh cho marker sáng, gần tròn."""
    cfg = config or BlobDetectorConfig()

    params = cv2.SimpleBlobDetector_Params()
    # Marker màu trắng trên nền tối.
    params.minThreshold = cfg.min_threshold
    params.maxThreshold = cfg.max_threshold
    params.thresholdStep = cfg.threshold_step

    # Phát hiện blob sáng.
    params.filterByColor = True
    params.blobColor = 255

    # Lọc theo diện tích (cần tinh chỉnh theo kích thước marker thực tế).
    params.filterByArea = True
    params.minArea = cfg.min_area
    params.maxArea = cfg.max_area

    # Lọc theo độ tròn (marker thường gần hình tròn).
    params.filterByCircularity = True
    params.minCircularity = cfg.min_circularity

    # Lọc theo quán tính (1.0 gần hình tròn, thấp hơn khi bị kéo dài).
    params.filterByInertia = True
    params.minInertiaRatio = cfg.min_inertia_ratio

    # Lọc theo độ lồi.
    params.filterByConvexity = True
    params.minConvexity = cfg.min_convexity

    return cv2.SimpleBlobDetector_create(params)


def detect_markers(
    img_processed: np.ndarray, config: BlobDetectorConfig | None = None
) -> tuple[np.ndarray, list[cv2.KeyPoint]]:
    """Trả về tâm marker dạng mảng float Nx2 và danh sách keypoint OpenCV."""
    detector = create_blob_detector(config=config)
    keypoints = detector.detect(img_processed)
    centers = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return centers, keypoints


# if __name__ == "__main__":
#     # Test nhanh trên một ảnh mẫu đã qua tiền xử lý.
#     img = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
#     proc = preprocess(img)
#     centers, keypoints = detect_markers(proc)
#     print(f"Detected: {len(centers)}\n markers: \n{centers}")
#     vis = cv2.drawKeypoints(
#         proc,
#         keypoints,
#         None,
#         (0, 255, 0),
#         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#     )
#     cv2.imwrite("outputs/step_2/sample_ref_markers.png", vis)
#     print("Marker detection test done, output saved to outputs/step_2/sample_ref_markers.png")