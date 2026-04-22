import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from src.utils.detection import detect_markers
from src.utils.preprocessing import preprocess
from src.Hungarian.Hungarian import match_markers_robust
from src.PyrLK.PyrLK import track_markers_lk
from src.utils.visualization import visualize_flow_arrows, visualize_flow_hsv

logger = logging.getLogger(__name__)


class RealtimeTactileTracking:
    """Pipeline realtime để theo dõi các marker tactile từ luồng webcam."""

    def __init__(self, camera_id: int = 0, arrow_scale: float = 1.0, tracking_method: str = "H", min_disp: float = 1.5) -> None:
        self.camera_id = camera_id
        self.arrow_scale = arrow_scale
        self.tracking_method = tracking_method
        self.min_disp = min_disp
        self.reference_image: Optional[npt.NDArray] = None
        self.reference_markers: Optional[npt.NDArray] = None

    def _wait_for_camera_warmup(self, capture: cv2.VideoCapture, frames: int = 15) -> None:
        """Bỏ qua vài frame đầu tiên để cảm biến camera tự điều chỉnh."""
        for _ in range(frames):
            capture.read()

    def _handle_keyboard_events(self, key: int, gray_frame: npt.NDArray) -> bool:
        """Xử lý các sự kiện bàn phím OpenCV. Trả về True nếu người dùng muốn thoát."""
        if key == ord('q'):
            return True
        elif key == ord('r'):
            self.reference_image = gray_frame.copy()
            reference_proc = preprocess(self.reference_image)
            self.reference_markers, _ = detect_markers(reference_proc)
            logger.info(f"Đã chụp ảnh tham chiếu: Phát hiện {len(self.reference_markers)} markers.")
        elif key == ord('c'):
            self.reference_image = None
            self.reference_markers = None
            logger.info("Đã xóa ảnh tham chiếu.")
        return False

    def _process_tracking(self, gray_frame: npt.NDArray) -> None:
        """Xử lý phát hiện và theo dõi marker khi ảnh tham chiếu đã được thiết lập."""
        if self.reference_image is None or self.reference_markers is None:
            return

        deformed_proc = preprocess(gray_frame)
        deformed_markers_naive, _ = detect_markers(deformed_proc)

        if len(self.reference_markers) == 0:
            return

        if self.tracking_method == "LK":
            deformed_markers_tracked, valid = track_markers_lk(
                self.reference_image, gray_frame, self.reference_markers,
                min_disp=self.min_disp
            )
        else:
            if len(deformed_markers_naive) > 0:
                max_displacement = self.reference_image.shape[1] / 10.0
                deformed_markers_tracked, valid = match_markers_robust(
                    self.reference_markers, deformed_markers_naive, self.reference_image.shape,
                    max_disp=max_displacement, min_disp=self.min_disp
                )
            else:
                valid = np.zeros(len(self.reference_markers), dtype=bool)

        if valid.any():
            vis_arrows = visualize_flow_arrows(
                deformed_proc, self.reference_markers, deformed_markers_tracked, valid,
                scale=self.arrow_scale, save_path=None
            )
            cv2.imshow("Realtime Flow Arrows", vis_arrows)

            vis_hsv = visualize_flow_hsv(
                self.reference_markers, deformed_markers_tracked, valid,
                self.reference_image.shape, save_path=None
            )
            cv2.imshow("Realtime Flow HSV", vis_hsv)

    def run(self) -> None:
        """Vòng lặp sự kiện thực thi chính."""
        capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            raise RuntimeError(f"Không thể mở camera: {self.camera_id}")

        logger.info(f"=== BẮT ĐẦU CAMERA {self.camera_id} ===")
        logger.info("- Nhấn 'r' để chụp frame hiện tại làm Reference (Trạng thái tĩnh).")
        logger.info("- Nhấn 'c' để xóa Reference frame.")
        logger.info("- Nhấn 'q' để thoát.")

        self._wait_for_camera_warmup(capture)

        while True:
            success, frame = capture.read()
            if not success:
                logger.error("Không thể đọc frame từ camera.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            key = cv2.waitKey(1) & 0xFF
            if self._handle_keyboard_events(key, gray_frame):
                break

            if self.reference_image is not None and self.reference_markers is not None:
                self._process_tracking(gray_frame)
                status_text = f"Dang track ({self.tracking_method}): gioi han {len(self.reference_markers)} markers"
                cv2.putText(
                    display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    display_frame, "Nhan 'r' de chup tham chieu", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow(f"Raw WebCam (/dev/video{self.camera_id})", display_frame)

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Realtime tactile tracking")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--arrow-scale", type=float, default=1.0, help="Arrow scale factor")
    parser.add_argument("--min-disp", type=float, default=1.5, help="Deadzone threshold to filter out material hysteresis (pixels)")
    parser.add_argument("--tracking-method", type=str, choices=["H", "LK"], default="H", help="Tracking method (H: Hungarian, LK: PyrLK)")
    args = parser.parse_args()

    pipeline = RealtimeTactileTracking(
        camera_id=args.camera_id, 
        arrow_scale=args.arrow_scale, 
        min_disp=args.min_disp,
        tracking_method=args.tracking_method
    )
    pipeline.run()

    """
    python -m src.Real_time.realtime_pipeline --tracking-method H
    python -m src.Real_time.realtime_pipeline --tracking-method H --min-disp 2.0
    
    """