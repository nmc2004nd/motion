"""Hiệu chuẩn camera bằng ảnh bàn cờ + khử méo."""

from __future__ import annotations

import glob
import logging
import os

import cv2
import numpy as np
import numpy.typing as npt

from ..config import as_int_tuple, require

logger = logging.getLogger(__name__)


def calibrate_camera(config: dict) -> tuple[npt.NDArray | None, npt.NDArray | None]:
    """Hiệu chuẩn camera từ các ảnh bàn cờ trong `paths.calib_dir`.

    Lưu kết quả ra `paths.calib_file` (`.npz` với `mtx`, `dist`, `rvecs`, `tvecs`).

    Returns:
        (camera_matrix, distortion_coeffs) hoặc (None, None) nếu thất bại.
    """
    calib_dir = require(config, "paths.calib_dir")
    calib_file = require(config, "paths.calib_file")
    checkerboard_size = as_int_tuple(require(config, "calibration.checkerboard_size"), "checkerboard_size")
    square_size = float(require(config, "calibration.square_size"))
    max_iter = int(require(config, "calibration.criteria.max_iter"))
    eps = float(require(config, "calibration.criteria.eps"))
    subpix_window = as_int_tuple(require(config, "calibration.subpix_window"), "subpix_window")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    object_points_grid = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    object_points_grid[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    object_points_grid = object_points_grid * square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    images = glob.glob(os.path.join(calib_dir, "*.jpg"))
    if not images:
        logger.error("No images found in %s", calib_dir)
        return None, None

    img_shape: tuple[int, int] | None = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
        success, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if success:
            object_points.append(object_points_grid)
            refined_corners = cv2.cornerSubPix(gray, corners, subpix_window, (-1, -1), criteria)
            image_points.append(refined_corners)

    if not object_points:
        logger.error("Could not find checkerboard corners in any images.")
        return None, None

    _, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None
    )
    logger.info("Camera calibration successful!")
    logger.debug("Camera Matrix:\n%s", camera_matrix)
    logger.debug("Distortion Coefficients:\n%s", distortion_coeffs)

    calib_dir_out = os.path.dirname(calib_file)
    if calib_dir_out:
        os.makedirs(calib_dir_out, exist_ok=True)
    np.savez(calib_file, mtx=camera_matrix, dist=distortion_coeffs, rvecs=rvecs, tvecs=tvecs)
    return camera_matrix, distortion_coeffs


def load_calibration(config: dict) -> tuple[npt.NDArray, npt.NDArray]:
    """Tải kết quả hiệu chuẩn (ma trận camera + hệ số biến dạng)."""
    calib_file = require(config, "paths.calib_file")
    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"Calibration file '{calib_file}' not found. Please run calibration first."
        )
    data = np.load(calib_file)
    return data["mtx"], data["dist"]


def undistort_image(
    img: npt.NDArray,
    camera_matrix: npt.NDArray,
    distortion_coeffs: npt.NDArray,
) -> npt.NDArray:
    """Khử méo ảnh + crop vùng đen dư."""
    if camera_matrix is None or distortion_coeffs is None:
        return img

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeffs, (w, h), 1, (w, h)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, distortion_coeffs, None, new_camera_matrix, (w, h), 5
    )
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y : y + h, x : x + w]
    return dst
