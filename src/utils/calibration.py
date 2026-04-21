import glob
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def calibrate_camera(
    calib_dir: str = './data/calib', 
    checkerboard_size: Tuple[int, int] = (8, 6), 
    square_size: float = 25.0
) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
    """Hiệu chuẩn camera sử dụng các ảnh bàn cờ."""
    # Tiêu chí dừng (termination criteria) cho việc tinh chỉnh toạ độ góc đạt độ chính xác sub-pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chuẩn bị các điểm vật thể trong không gian 3D
    object_points_grid = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    object_points_grid[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    object_points_grid = object_points_grid * square_size

    object_points = []  # Các điểm 3D trong không gian thực
    image_points = []   # Các điểm 2D trên mặt phẳng ảnh

    images = glob.glob(os.path.join(calib_dir, '*.jpg'))
    
    if not images:
        logger.error(f"No images found in {calib_dir}")
        return None, None

    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Tìm các góc bên trong của bàn cờ
        success, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if success:
            object_points.append(object_points_grid)
            # Tinh chỉnh toạ độ các góc đạt mức sub-pixel (chính xác hơn)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(refined_corners)

    if not object_points:
        logger.error("Could not find checkerboard corners in any images.")
        return None, None

    # Thực hiện hiệu chuẩn camera
    success, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None
    )
    
    logger.info("Camera calibration successful!")
    logger.debug(f"Camera Matrix:\n{camera_matrix}")
    logger.debug(f"Distortion Coefficients:\n{distortion_coeffs}")
    
    os.makedirs('config', exist_ok=True)
    np.savez('config/calib_result.npz', mtx=camera_matrix, dist=distortion_coeffs, rvecs=rvecs, tvecs=tvecs)
    
    return camera_matrix, distortion_coeffs

def load_calibration(calib_file: str = 'config/calib_result.npz') -> Tuple[npt.NDArray, npt.NDArray]:
    """Tải kết quả hiệu chuẩn camera (ma trận và hệ số biến dạng)."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file '{calib_file}' not found. Please run calibration first.")
        
    data = np.load(calib_file)
    return data['mtx'], data['dist']


def undistort_image(
    img: npt.NDArray, 
    camera_matrix: npt.NDArray, 
    distortion_coeffs: npt.NDArray
) -> npt.NDArray:
    """Khử méo (undistort) ảnh sử dụng ma trận camera và hệ số biến dạng đã hiệu chuẩn."""
    if camera_matrix is None or distortion_coeffs is None:
        return img
    
    h, w = img.shape[:2]
    # Tính toán lại thông số ma trận camera tối ưu dựa trên vùng quan tâm (ROI)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    
    # Tạo bản đồ ánh xạ để khử méo
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, new_camera_matrix, (w, h), 5)
    # Áp dụng bản đồ lên bức ảnh
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    
    # Cắt (crop) loại bỏ vùng đen sau khi khử méo ảnh
    x, y, w, h = roi
    if roi != (0, 0, 0, 0): 
        dst = dst[y:y+h, x:x+w]
        
    return dst

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    calibrate_camera()
