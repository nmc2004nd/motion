import argparse
import logging
import time
import cProfile
import pstats
from typing import Optional, Dict

import cv2
import numpy.typing as npt

from src.utils.detection import detect_markers, create_blob_detector
from src.utils.preprocessing import preprocess
from src.pyr_lk.pyr_lk import track_markers_lk
from src.utils.visualization import visualize_flow_arrows
from src.utils.config_parser import load_config

logger = logging.getLogger(__name__)


class RealtimeTactileTracking:
    """Pipeline realtime để theo dõi các marker tactile từ luồng webcam."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.camera_id = self.config.get("camera", {}).get("device_id", 0)

        controls = self.config.get("controls", {})
        self.wait_key_time = controls.get("wait_key", 1)
        self.key_capture = ord(controls.get("key_capture", "r"))
        self.key_clear = ord(controls.get("key_clear", "c"))
        self.key_quit = ord(controls.get("key_quit", "q"))

        self.reference_image: Optional[npt.NDArray] = None
        self.reference_markers: Optional[npt.NDArray] = None

        pre_cfg = config.get("preprocessing", {})
        self._clahe = cv2.createCLAHE(
            clipLimit=pre_cfg.get("clahe_clip_limit", 2.5),
            tileGridSize=tuple(pre_cfg.get("clahe_grid", [8, 8])),
        )
        self._blob_detector = create_blob_detector(config=config)

    def _wait_for_camera_warmup(self, capture: cv2.VideoCapture) -> None:
        frames = self.config.get("camera", {}).get("warmup_frames", 15)
        for _ in range(frames):
            capture.read()

    def _handle_keyboard_events(self, key: int, gray_frame: npt.NDArray) -> bool:
        """Xử lý các sự kiện bàn phím OpenCV. Trả về True nếu người dùng muốn thoát."""
        if key == self.key_quit:
            return True
        elif key == self.key_capture:
            self.reference_image = gray_frame.copy()
            reference_proc = preprocess(self.reference_image, config=self.config, _clahe=self._clahe)
            self.reference_markers, _ = detect_markers(
                reference_proc, config=self.config, _detector=self._blob_detector
            )
            logger.info(f"Đã chụp ảnh tham chiếu: Phát hiện {len(self.reference_markers)} markers.")
        elif key == self.key_clear:
            self.reference_image = None
            self.reference_markers = None
            logger.info("Đã xóa ảnh tham chiếu.")
        return False

    def _process_tracking(self, gray_frame: npt.NDArray) -> Dict[str, float]:
        """Track markers bằng LK từ reference sang frame hiện tại. Trả về latency (ms)."""
        timings = {}

        if self.reference_image is None or self.reference_markers is None:
            return timings
        if len(self.reference_markers) == 0:
            return timings

        t_track = time.perf_counter()
        deformed_markers, valid = track_markers_lk(
            self.reference_image, gray_frame, self.reference_markers, config=self.config
        )
        timings["track"] = (time.perf_counter() - t_track) * 1000

        t_vis = time.perf_counter()
        if valid.any():
            vis_arrows = visualize_flow_arrows(
                gray_frame, self.reference_markers, deformed_markers, valid,
                config=self.config, save_path=None
            )
            cv2.imshow("Realtime Flow Arrows", vis_arrows)
        timings["visualize"] = (time.perf_counter() - t_vis) * 1000

        return timings

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
        prev_time = time.perf_counter()

        while True:
            success, frame = capture.read()
            if not success:
                logger.error("Không thể đọc frame từ camera.")
                break

            current_time = time.perf_counter()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            key = cv2.waitKey(self.wait_key_time) & 0xFF
            if self._handle_keyboard_events(key, gray_frame):
                break

            text_cfg = self.config.get("visualization", {}).get("text", {})
            pos_status = tuple(text_cfg.get("status_pos", [10, 30]))
            pos_fps = (10, 60)
            scale_normal = text_cfg.get("scale_normal", 1.0)
            scale_idle = text_cfg.get("scale_idle", 0.6)
            thickness = text_cfg.get("thickness", 2)
            c_track = tuple(text_cfg.get("color_tracking", [0, 255, 0]))
            c_idle = tuple(text_cfg.get("color_idle", [0, 0, 255]))

            if self.reference_image is not None and self.reference_markers is not None:
                timings = self._process_tracking(gray_frame)
                status_text = f"Dang track (LK): {len(self.reference_markers)} markers"

                cv2.putText(display_frame, status_text, pos_status, cv2.FONT_HERSHEY_SIMPLEX, scale_normal, c_track, thickness)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", pos_fps, cv2.FONT_HERSHEY_SIMPLEX, scale_normal, c_track, thickness)

                y_offset = 90
                for step, t_ms in timings.items():
                    cv2.putText(display_frame, f"{step}: {t_ms:.1f}ms", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale_idle, (255, 255, 0), thickness)
                    y_offset += 25
            else:
                cv2.putText(
                    display_frame, "Nhan 'r' de chup tham chieu",
                    pos_status, cv2.FONT_HERSHEY_SIMPLEX, scale_idle, c_idle, thickness
                )
                cv2.putText(display_frame, f"FPS: {fps:.1f}", pos_fps, cv2.FONT_HERSHEY_SIMPLEX, scale_normal, c_idle, thickness)

            cv2.imshow(f"Raw WebCam (/dev/video{self.camera_id})", display_frame)

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Realtime tactile tracking")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--profile", action="store_true", help="Enable profiling with cProfile")
    args = parser.parse_args()

    config_parser = load_config(args.config_path)
    pipeline = RealtimeTactileTracking(config=config_parser.config)

    if args.profile:
        logger.info("Chạy ở chế độ Profile...")
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            pipeline.run()
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.dump_stats('profile_result.prof')
            logger.info("Đã lưu kết quả profiling tại 'profile_result.prof'. Dùng snakeviz profile_result.prof để xem.")
    else:
        pipeline.run()

    """
    python -m src.real_time.realtime_pipeline
    python -m src.real_time.realtime_pipeline --profile
    snakeviz profile_result.prof
    """
