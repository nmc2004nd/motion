import logging
import os
import sys
import cv2
import numpy as np

# Thêm thư mục root của project vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.loader import load_config, require, as_int_tuple
from src.core.preprocessing import preprocess, make_clahe
from src.core.detection import create_blob_detector, detect_markers

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    config_path = os.path.join(project_root, "config", "pipeline_config.yaml")
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Lỗi đọc config: {e}")
        return

    # 1. Đường dẫn ảnh đầu vào
    ref_path = require(config, "paths.ref_image")
    def_path = require(config, "paths.deformed_image")
    
    if not os.path.isabs(ref_path): ref_path = os.path.join(project_root, ref_path)
    if not os.path.isabs(def_path): def_path = os.path.join(project_root, def_path)

    # Thư mục lưu ảnh đầu ra
    out_dir_base = require(config, "paths.output_dir")
    if not os.path.isabs(out_dir_base): 
        out_dir_base = os.path.join(project_root, out_dir_base)
    out_dir = os.path.join(out_dir_base, "detection_test")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Đọc ảnh
    ref_img = cv2.imread(ref_path)
    def_img = cv2.imread(def_path)
    
    if ref_img is None:
        logging.error(f"Không thể đọc ảnh Reference: {ref_path}")
        return
    if def_img is None:
        logging.error(f"Không thể đọc ảnh Deformed (trượt): {def_path}")
        return

    # Đổi sang ảnh xám
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    def_gray = cv2.cvtColor(def_img, cv2.COLOR_BGR2GRAY)

    # 3. Tiền xử lý
    logging.info("Đang tiền xử lý ảnh...")
    clahe = make_clahe(config)
    ref_proc = preprocess(ref_gray, config, _clahe=clahe)
    def_proc = preprocess(def_gray, config, _clahe=clahe)

    # 4. Phát hiện marker
    logging.info("Đang detect markers...")
    detector = create_blob_detector(config)
    ref_centers, ref_kps = detect_markers(ref_proc, config, _detector=detector)
    def_centers, def_kps = detect_markers(def_proc, config, _detector=detector)

    logging.info(f"Ảnh Reference: tìm thấy {len(ref_kps)} markers")
    logging.info(f"Ảnh Deformed: tìm thấy {len(def_kps)} markers")

    # 5. Vẽ marker
    # Lấy màu keypoint từ config
    kp_color = as_int_tuple(require(config, "visualization.keypoint_color"), "keypoint_color")
    color = (int(kp_color[0]), int(kp_color[1]), int(kp_color[2]))

    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS vẽ vòng tròn mô tả cả diện tích của marker
    ref_drawn = cv2.drawKeypoints(ref_img, ref_kps, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    def_drawn = cv2.drawKeypoints(def_img, def_kps, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 6. Lưu ảnh và hiển thị
    ref_save_path = os.path.join(out_dir, "ref_detected.jpg")
    def_save_path = os.path.join(out_dir, "def_detected.jpg")
    
    cv2.imwrite(ref_save_path, ref_drawn)
    cv2.imwrite(def_save_path, def_drawn)
    logging.info(f"Đã lưu ảnh Reference tại: {ref_save_path}")
    logging.info(f"Đã lưu ảnh Deformed tại: {def_save_path}")

    # Gõ scale để fit màn hình
    h, w = ref_gray.shape
    scale = min(800/w, 600/h, 1.0)
    if scale < 1.0:
        ref_disp = cv2.resize(ref_drawn, (0,0), fx=scale, fy=scale)
        def_disp = cv2.resize(def_drawn, (0,0), fx=scale, fy=scale)
    else:
        ref_disp = ref_drawn
        def_disp = def_drawn
        
    cv2.imshow("Reference Detections", ref_disp)
    cv2.imshow("Deformed Detections", def_disp)
    
    logging.info("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
