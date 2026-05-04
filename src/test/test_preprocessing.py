import logging
import os
import sys
import cv2
import numpy as np

# Thêm thư mục root của project vào sys.path để import file từ src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.loader import load_config, as_int_tuple, require

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    config_path = os.path.join(project_root, "config", "pipeline_config.yaml")
    
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return

    # Lấy đường dẫn ảnh tham chiếu từ config file
    img_path = require(config, "paths.ref_image")
    # Nếu là đường dẫn tương đối, ghép với project root, nếu là tuyệt đối thì giữ nguyên
    if not os.path.isabs(img_path):
        img_path = os.path.join(project_root, img_path)
        
    logging.info(f"Đọc ảnh từ: {img_path}")
    img_bgr = cv2.imread(img_path)
    
    # Nếu không đọc được ảnh tham chiếu, thử lấy ảnh từ camera
    if img_bgr is None:
        logging.warning("Không tìm thấy ảnh tham chiếu, thử lấy một frame từ camera...")
        cam_id = require(config, "camera.device_id")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            logging.error(f"Không thể mở camera {cam_id}")
            return
        ret, img_bgr = cap.read()
        cap.release()
        if not ret or img_bgr is None:
            logging.error("Không thể đọc frame từ camera.")
            return

    # Chuyển ảnh màu sang ảnh xám
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Lấy thông số từ cấu hình
    blur_kernel = as_int_tuple(require(config, "preprocessing.blur_kernel"), "blur_kernel")
    clip_limit = float(require(config, "preprocessing.clahe_clip_limit"))
    grid_size = as_int_tuple(require(config, "preprocessing.clahe_grid"), "clahe_grid")

    # Khởi tạo kích thước hiển thị cố định nếu ảnh quá to để có thể nhìn rõ trên màn hình
    h, w = img_gray.shape
    scale = min(800/w, 600/h, 1.0)
    
    # Chuẩn bị thư mục lưu ảnh
    out_dir_base = require(config, "paths.output_dir")
    if not os.path.isabs(out_dir_base):
        out_dir_base = os.path.join(project_root, out_dir_base)
    out_dir = os.path.join(out_dir_base, "preprocessing_steps")
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Kết quả sẽ được lưu vào: {out_dir}")

    def show_and_save(title, filename, img):
        # Hiển thị
        if scale < 1.0:
            display_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        else:
            display_img = img
        cv2.imshow(title, display_img)
        # Lưu
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, img)

    logging.info("Hiển thị và lưu 4 công đoạn tiền xử lý:")
    # Công đoạn 1: Ảnh xám gốc
    show_and_save("1. Gray Image", "1_gray.jpg", img_gray)
    
    # Công đoạn 2: Lọc nền (box blur)
    background = cv2.blur(img_gray, blur_kernel)
    show_and_save("2. Background (Blur)", "2_background.jpg", background)
    
    # Công đoạn 3: Trừ ảnh gốc cho nền và chuẩn hóa min-max
    normalized = cv2.subtract(img_gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    show_and_save("3. Normalized (Subtracted & MinMax)", "3_normalized.jpg", normalized)
    
    # Công đoạn 4: Tăng cường tương phản cục bộ bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    final_img = clahe.apply(normalized.astype(np.uint8))
    show_and_save("4. Final CLAHE output", "4_clahe_final.jpg", final_img)
    
    logging.info("Nhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
