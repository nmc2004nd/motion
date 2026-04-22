import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

# Tận dụng các module tính toán, hình ảnh từ base code thông qua absolute import
from src.utils.detection import detect_markers
from src.utils.preprocessing import preprocess
from src.hungarian.hungarian import match_markers_robust
from src.pyr_lk.pyr_lk import track_markers_lk
from src.utils.visualization import visualize_flow_arrows, visualize_flow_hsv
from src.utils.config_parser import load_config

# Import SlipDetector để tính toán khả năng trượt
from src.slip_prob.slip_prob import SlipDetector

logger = logging.getLogger(__name__)


class RealtimeSlipTracking:
    """Pipeline realtime theo dõi markers và CẢNH BÁO TRƯỢT (Slip Detection)
    Dựa trên việc kế thừa logic của base code nhưng được viết hoàn chỉnh, tách rời."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.camera_id = self.config.get("camera", {}).get("device_id", 0)
        self.tracking_method = self.config.get("tracking", {}).get("method", "H")
        
        # Thêm State chuyên biệt cho Slip Detection
        self.marker_history = []
        self.history_length = self.config.get("slip_detection", {}).get("history_buffer_length", 5)
        self.slip_detector = SlipDetector(config=self.config)
        self.slip_threshold_display = self.config.get("slip_detection", {}).get("slip_threshold", 0.8)
        self.moving_count_thresh = self.config.get("slip_detection", {}).get("moving_count_thresh", 2)
        
        # Load keyboard controls
        controls = self.config.get("controls", {})
        self.wait_key_time = controls.get("wait_key", 1)
        self.key_capture = ord(controls.get("key_capture", "r"))
        self.key_clear = ord(controls.get("key_clear", "c"))
        self.key_quit = ord(controls.get("key_quit", "q"))
        
        # State của hệ thống chụp
        self.reference_image: Optional[npt.NDArray] = None
        self.reference_markers: Optional[npt.NDArray] = None

    def _wait_for_camera_warmup(self, capture: cv2.VideoCapture) -> None:
        """Bỏ qua vài frame đầu tiên để cảm biến camera tự điều chỉnh."""
        frames = self.config.get("camera", {}).get("warmup_frames", 15)
        for _ in range(frames):
            capture.read()

    def _handle_keyboard_events(self, key: int, gray_frame: npt.NDArray) -> bool:
        """Xử lý thao tác bàn phím, trả về True nếu chọn thoát (q)."""
        if key == self.key_quit:
            return True
        elif key == self.key_capture:
            # Chụp một frame tĩnh làm mốc (Reference)
            self.reference_image = gray_frame.copy()
            reference_proc = preprocess(self.reference_image, config=self.config)
            self.reference_markers, _ = detect_markers(reference_proc, config=self.config)
            
            # Reset lịch sử toạ độ marker
            self.marker_history = [self.reference_markers.copy()] 
            
            logger.info(f"Đã chụp ảnh tham chiếu: Phát hiện {len(self.reference_markers)} markers.")
            
        elif key == self.key_clear:
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

        deformed_proc = preprocess(gray_frame, config=self.config)
        deformed_markers_naive, _ = detect_markers(deformed_proc, config=self.config)

        if len(self.reference_markers) == 0:
            return

        # 1. THỰC HIỆN TRACKING 
        if self.tracking_method == "LK":
            deformed_markers_tracked, valid = track_markers_lk(
                self.reference_image, gray_frame, self.reference_markers, config=self.config
            )
        else:
            if len(deformed_markers_naive) > 0:
                deformed_markers_tracked, valid = match_markers_robust(
                    self.reference_markers, deformed_markers_naive, self.reference_image.shape,
                    config=self.config
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
                config=self.config, save_path=None
            )
            cv2.imshow("Realtime Flow Arrows", vis_arrows)

            vis_hsv = visualize_flow_hsv(
                self.reference_markers, deformed_markers_tracked, valid,
                self.reference_image.shape, config=self.config, save_path=None
            )
            cv2.imshow("Realtime Flow HSV", vis_hsv)

        # 4. HIỂN THỊ TÌNH TRẠNG TRƯỢT LÊN WEBCAM DISPLAY
        if slip_info is not None:
            r_val = slip_info['r_value']
            moving = slip_info['moving_count']
            
            text_cfg = self.config.get("visualization", {}).get("text", {})
            pos_slip = tuple(text_cfg.get("slip_pos", [10, 70]))
            scale_normal = text_cfg.get("scale_normal", 0.8)
            thickness = text_cfg.get("thickness", 2)
            c_slip = tuple(text_cfg.get("color_slip", [0, 0, 255]))
            c_moving = tuple(text_cfg.get("color_moving", [0, 255, 255]))
            c_track = tuple(text_cfg.get("color_tracking", [0, 255, 0]))

            slip_status = 1 if r_val > self.slip_threshold_display else 0
            
            text_color = c_slip if slip_status == 1 else (c_moving if moving > self.moving_count_thresh else c_track)
            
            display_text = f"Slip Prob: {r_val:.2f} | Slip: {slip_status} | Moves: {moving}"
            cv2.putText(display_frame, display_text, pos_slip,
                        cv2.FONT_HERSHEY_SIMPLEX, scale_normal, text_color, thickness)


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

            key = cv2.waitKey(self.wait_key_time) & 0xFF
            if self._handle_keyboard_events(key, gray_frame):
                break
            
            text_cfg = self.config.get("visualization", {}).get("text", {})
            pos_status = tuple(text_cfg.get("status_pos", [10, 30]))
            scale_normal = text_cfg.get("scale_normal", 0.8)
            scale_idle = text_cfg.get("scale_idle", 0.8)
            thickness = text_cfg.get("thickness", 2)
            c_track = tuple(text_cfg.get("color_tracking", [0, 255, 0]))
            c_idle = tuple(text_cfg.get("color_idle", [0, 0, 255]))

            if self.reference_image is not None and self.reference_markers is not None:
                # Gọi xử lý
                self._process_tracking(gray_frame, display_frame)
                
                status_text = f"Tracking ({self.tracking_method}) | {len(self.reference_markers)} markers"
                cv2.putText(
                    display_frame, status_text, pos_status, 
                    cv2.FONT_HERSHEY_SIMPLEX, scale_normal, c_track, thickness
                )
            else:
                cv2.putText(
                    display_frame, "Nhan 'r' de chup tham chieu", 
                    pos_status, cv2.FONT_HERSHEY_SIMPLEX, scale_idle, c_idle, thickness
                )

            cv2.imshow(f"WebCam w/ Slip Detection (/dev/video{self.camera_id})", display_frame)

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Realtime tactile tracking WITH SLIP DETECTION")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    config_parser = load_config(args.config_path)
    pipeline = RealtimeSlipTracking(config=config_parser.config)
    pipeline.run()

"""
python -m src.slip_prob.realtime_slip_pipeline \
       --camera-id 0 \
       --tracking-method H \
       --slip-threshold 0.6
       """
