import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

# Tận dụng các module tính toán, hình ảnh từ base code thông qua absolute import
from src.utils.detection import detect_markers
from src.utils.preprocessing import preprocess
from src.Hungarian.Hungarian import match_markers_robust
from src.PyrLK.PyrLK import track_markers_lk
from src.utils.visualization import visualize_flow_arrows, visualize_flow_hsv

# Import SlipDetector để tính toán khả năng trượt
from src.slip_prob.slip_prob import SlipDetector

logger = logging.getLogger(__name__)


class RealtimeSlipTracking:
    """Pipeline realtime theo dõi markers và CẢNH BÁO TRƯỢT (Slip Detection)
    Dựa trên việc kế thừa logic của base code nhưng được viết hoàn chỉnh, tách rời."""

    def __init__(self, camera_id: int = 0, arrow_scale: float = 1.0, tracking_method: str = "H", slip_threshold: float = 0.8) -> None:
        self.camera_id = camera_id
        self.arrow_scale = arrow_scale
        self.tracking_method = tracking_method
        
        # State của hệ thống chụp
        self.reference_image: Optional[npt.NDArray] = None
        self.reference_markers: Optional[npt.NDArray] = None
        
        # Thêm State chuyên biệt cho Slip Detection
        self.marker_history = []
        self.history_length = 5  # Dùng bộ đệm 5 frames để loại bỏ giật nháy lúc trượt (Flickering)
        self.slip_detector = SlipDetector(min_motion_thresh=1, slip_threshold=slip_threshold)

    def _wait_for_camera_warmup(self, capture: cv2.VideoCapture, frames: int = 15) -> None:
        """Bỏ qua vài frame đầu tiên để cảm biến camera xử lý sáng."""
        for _ in range(frames):
            capture.read()

    def _handle_keyboard_events(self, key: int, gray_frame: npt.NDArray) -> bool:
        """Xử lý thao tác bàn phím, trả về True nếu chọn thoát (q)."""
        if key == ord('q'):
            return True
        elif key == ord('r'):
            # Chụp một frame tĩnh làm mốc (Reference)
            self.reference_image = gray_frame.copy()
            reference_proc = preprocess(self.reference_image)
            self.reference_markers, _ = detect_markers(reference_proc)
            
            # Reset lịch sử toạ độ marker
            self.marker_history = [self.reference_markers.copy()] 
            
            logger.info(f"Đã chụp ảnh tham chiếu: Phát hiện {len(self.reference_markers)} markers.")
            
        elif key == ord('c'):
            # Xóa Reference
            self.reference_image = None
            self.reference_markers = None
            self.marker_history = []
            self.slip_detector.reset()
            logger.info("Đã xóa ảnh tham chiếu.")
        return False

    def _process_tracking(self, gray_frame: npt.NDArray, display_frame: npt.NDArray) -> None:
        """Xử lý quá trình giải quyết tracking và tính toán trượt."""
        if self.reference_image is None or self.reference_markers is None:
            return

        deformed_proc = preprocess(gray_frame)
        deformed_markers_naive, _ = detect_markers(deformed_proc)

        if len(self.reference_markers) == 0:
            return

        # 1. THỰC HIỆN TRACKING 
        if self.tracking_method == "LK":
            deformed_markers_tracked, valid = track_markers_lk(
                self.reference_image, gray_frame, self.reference_markers
            )
        else:
            if len(deformed_markers_naive) > 0:
                max_displacement = self.reference_image.shape[1] / 10.0
                deformed_markers_tracked, valid = match_markers_robust(
                    self.reference_markers, deformed_markers_naive, self.reference_image.shape,
                    max_disp=max_displacement
                )
            else:
                valid = np.zeros(len(self.reference_markers), dtype=bool)

        # 2. KIỂM TRA TRƯỢT (SLIP DETECTION)
        slip_info = None
        if valid.any():
            # Thêm trạng thái hiện tại vào vùng đệm
            self.marker_history.append(deformed_markers_tracked.copy())
            
            # Loại bỏ các frame quá cũ khỏi vùng đệm
            if len(self.marker_history) > self.history_length:
                self.marker_history.pop(0)
            
            # Tính đạo hàm với frame quá khứ (cách đây history_length frames)
            past_markers = self.marker_history[0]
            slip_info = self.slip_detector.calculate_slip_probability(
                prev_markers=past_markers,
                current_markers=deformed_markers_tracked,
                valid_mask=valid,
                ref_markers=self.reference_markers   # CHỐNG NHIỄU ĐÀN HỒI
            )

        # 3. HIỂN THỊ VISUALIZATION BASE
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

        # 4. HIỂN THỊ TÌNH TRẠNG TRƯỢT LÊN WEBCAM DISPLAY
        if slip_info is not None:
            r_val = slip_info['r_value']
            moving = slip_info['moving_count']
            
            # Cập nhật ngưỡng 0.5 theo yêu cầu để bật cờ trượt 
            slip_status = 1 if r_val > 0.5 else 0
            
            # Chữ đỏ nếu Slip = 1, vàng nếu đang phân tán nhẹ, xanh nếu đứng im
            text_color = (0, 0, 255) if slip_status == 1 else ((0, 255, 255) if moving > 2 else (0, 255, 0))
            
            # Hiển thị gọn gàng, giữ nguyên xác suất và chuyển Slip thành 0/1
            display_text = f"Slip Prob: {r_val:.2f} | Slip: {slip_status} | Moves: {moving}"
            cv2.putText(display_frame, display_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)


    def run(self) -> None:
        """Vòng lặp sự kiện thực thi chính."""
        capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            raise RuntimeError(f"Không thể mở camera: {self.camera_id}")

        logger.info(f"=== BẮT ĐẦU CAMERA {self.camera_id} TÍCH HỢP SLIP DETECTION ===")
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
                # Gọi xử lý
                self._process_tracking(gray_frame, display_frame)
                
                status_text = f"Tracking ({self.tracking_method}) | {len(self.reference_markers)} markers"
                cv2.putText(
                    display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    display_frame, "Nhan 'r' de chup tham chieu", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow(f"WebCam w/ Slip Detection (/dev/video{self.camera_id})", display_frame)

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Realtime tactile tracking WITH SLIP DETECTION")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--arrow-scale", type=float, default=1.0, help="Arrow scale factor")
    parser.add_argument("--tracking-method", type=str, choices=["H", "LK"], default="H", help="Tracking method (H: Hungarian, LK: PyrLK)")
    parser.add_argument("--slip-threshold", type=float, default=0.6, help="Ngưỡng đồng nhất hướng R để cảnh báo trượt (Mặc định 0.8)")
    args = parser.parse_args()

    pipeline = RealtimeSlipTracking(
        camera_id=args.camera_id, 
        arrow_scale=args.arrow_scale, 
        tracking_method=args.tracking_method,
        slip_threshold=args.slip_threshold
    )
    pipeline.run()

"""
python -m src.slip_prob.realtime_slip_pipeline \
       --camera-id 0 \
       --tracking-method LK \
       --slip-threshold 0.6
       """
