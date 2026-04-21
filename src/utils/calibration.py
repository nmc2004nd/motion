import numpy as np
import cv2
import glob
import os

def calibrate_camera(calib_dir='./data/calib', checkerboard_size=(8, 6), square_size=25.0):
    # Tiêu chí dừng (termination criteria) cho việc tinh chỉnh toạ độ góc đạt độ chính xác sub-pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chuẩn bị các điểm vật thể (object points) trong không gian 3D, ví dụ: (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Các mảng dùng để lưu điểm 3D (thế giới thực) và điểm 2D (trên mặt phẳng ảnh) từ tất cả các ảnh
    objpoints = [] # Điểm 3D trong không gian thực
    imgpoints = [] # Điểm 2D trên mặt phẳng ảnh

    images = glob.glob(os.path.join(calib_dir, '*.jpg'))
    
    if not images:
        print(f"No images found in {calib_dir}")
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
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # Nếu tìm thấy, thêm điểm vật thể 3D và điểm trên ảnh 2D (sau khi tinh chỉnh)
        if ret == True:
            objpoints.append(objp)
            # Tinh chỉnh toạ độ các góc đạt mức sub-pixel (chính xác hơn)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Vẽ và hiển thị các góc (dành cho mục đích debug)
            # cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    if not objpoints:
        print("Không thể tìm thấy góc checkerboard trong bất kỳ ảnh nào.")
        return None, None

    # Thực hiện camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    print("Camera calibration successful!")
    print("\nMa trận Camera (Camera Matrix):\n", mtx)
    print("\nHệ số biến dạng (Distortion Coefficients):\n", dist)
    
    # Lưu lại kết quả calibration vào file
    np.savez('calib_result.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
    return mtx, dist

def load_calibration(calib_file='calib_result.npz'):
    """Tải kết quả hiệu chuẩn camera (matrix và distortion coefficients)."""
    try:
        data = np.load(calib_file)
        return data['mtx'], data['dist']
    except FileNotFoundError:
        print(f"Không tìm thấy file calibration {calib_file}. Đang tự động tiến hành chạy hiệu chuẩn...")
        mtx, dist = calibrate_camera()
        return mtx, dist

def undistort_image(img, mtx, dist):
    """Khử méo (undistort) một bức ảnh sử dụng ma trận camera và hệ số biến dạng đã thiết lập."""
    if mtx is None or dist is None:
        return img
    
    h, w = img.shape[:2]
    # Tính toán lại thông số ma trận camera tối ưu dựa trên ROI (Region of interest)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Tạo bản đồ ánh xạ để khử méo
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # Áp dụng bản đồ lên bức ảnh
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # Cắt (crop) loại bỏ vùng đen sau khi khử méo ảnh
    x, y, w, h = roi
    if roi != (0, 0, 0, 0): # Kiểm tra ROI hợp lệ trước khi cắt
        dst = dst[y:y+h, x:x+w]
        
    return dst

if __name__ == '__main__':
    # Bạn có thể cần điều chỉnh 'checkerboard_size' tuỳ thuộc vào kích thước bàn cờ calibration thực tế
    calibrate_camera()
