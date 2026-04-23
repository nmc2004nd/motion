"""Điều phối pipeline cho việc theo dõi các marker tactile."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils.detection import detect_markers
from .utils.preprocessing import preprocess
from .utils.calibration import load_calibration, undistort_image
from .utils.config_parser import load_config
from .pyr_lk.pyr_lk import track_markers_lk
from .utils.visualization import visualize_flow_arrows

logger = logging.getLogger(__name__)


class TactileMarkerTrackingPipeline:
    """Pipeline theo dõi marker tactile từ đầu đến cuối với các bước được module hóa."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.tracking_method = "LK"
        output_dir_base = self.config.get("paths", {}).get("output_dir", "outputs")
        self.output_dir = Path(output_dir_base) / self.tracking_method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_calib = self.config.get("pipeline", {}).get("use_calibration", False)
        self.ref_image_path = self.config.get("paths", {}).get("ref_image", "data/ref/my_photo_1.jpg")
        self.def_image_path = self.config.get("paths", {}).get("deformed_image", "data/img/my_photo_2.jpg")
        self.calib_file_path = self.config.get("paths", {}).get("calib_file", "config/calib_result.npz")
        self.summary_figsize = tuple(self.config.get("visualization", {}).get("summary_figsize", [18, 10]))
        self.summary_dpi = self.config.get("visualization", {}).get("summary_dpi", 100)

    def _read_gray(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img

    def _save_summary_figure(
        self,
        img_reference: np.ndarray,
        img_deformed: np.ndarray,
        reference_proc: np.ndarray,
        vis_reference: np.ndarray,
        vis_arrows: np.ndarray,
        marker_count: int,
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=self.summary_figsize)
        axes[0, 0].imshow(img_reference, cmap="gray")
        axes[0, 0].set_title("1. Raw reference")
        axes[0, 1].imshow(img_deformed, cmap="gray")
        axes[0, 1].set_title("2. Raw deformed")
        axes[0, 2].imshow(reference_proc, cmap="gray")
        axes[0, 2].set_title("3. Preprocessed")

        axes[1, 0].imshow(cv2.cvtColor(vis_reference, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"4. Markers in reference: {marker_count}")
        axes[1, 1].imshow(cv2.cvtColor(vis_arrows, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5. Flow arrows")
        axes[1, 2].axis("off")

        for axis in axes.ravel():
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "06_summary.png", dpi=self.summary_dpi, bbox_inches="tight")
        plt.close(fig)

    def _load_and_undistort_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_reference_raw = self._read_gray(self.ref_image_path)
        img_deformed_raw = self._read_gray(self.def_image_path)

        if self.use_calib:
            logger.info(f"Dang tai hieu chuan camera tu {self.calib_file_path}...")
            camera_matrix, dist_coeffs = load_calibration(config=self.config)
            img_reference = undistort_image(img_reference_raw, camera_matrix, dist_coeffs)
            img_deformed = undistort_image(img_deformed_raw, camera_matrix, dist_coeffs)
            
            # Luu calc anh da khu meo nhu mot buoc debug
            cv2.imwrite(str(self.output_dir / "01_reference_undistorted.png"), img_reference)
            cv2.imwrite(str(self.output_dir / "01_deformed_undistorted.png"), img_deformed)
            logger.info("Da khu meo anh thanh cong.")
        else:
            img_reference = img_reference_raw
            img_deformed = img_deformed_raw
            
        return img_reference_raw, img_deformed_raw, img_reference, img_deformed

    def _preprocess_images(self, img_reference: np.ndarray, img_deformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference_proc = preprocess(img_reference, config=self.config)
        deformed_proc = preprocess(img_deformed, config=self.config)
        cv2.imwrite(str(self.output_dir / "02_reference_preprocessed.png"), reference_proc)
        cv2.imwrite(str(self.output_dir / "02_deformed_preprocessed.png"), deformed_proc)
        logger.info("Qua trinh tien xu ly hoan tat.")
        return reference_proc, deformed_proc

    def _detect_markers(self, reference_proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoint_color = tuple(self.config.get("visualization", {}).get("keypoint_color", [0, 255, 0]))
        reference_markers, reference_keypoints = detect_markers(reference_proc, config=self.config)
        logger.info(f"Da phat hien {len(reference_markers)} marker trong anh tham chieu.")
        vis_reference = cv2.drawKeypoints(
            reference_proc,
            reference_keypoints,
            None,
            keypoint_color,
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(str(self.output_dir / "03_reference_markers.png"), vis_reference)
        return reference_markers, vis_reference

    def _track_and_analyze(
        self,
        img_reference: np.ndarray,
        img_deformed: np.ndarray,
        reference_markers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        deformed_markers_tracked, valid = track_markers_lk(
            img_reference, img_deformed, reference_markers, config=self.config
        )
        logger.info(f"Da theo doi thanh cong {valid.sum()}/{len(reference_markers)} markers (LK).")

        if valid.any():
            displacements = deformed_markers_tracked - reference_markers
            magnitudes = np.linalg.norm(displacements[valid], axis=1)
            logger.info(
                f"Do dich chuyen: trung binh={magnitudes.mean():.2f}px, "
                f"max={magnitudes.max():.2f}px, min={magnitudes.min():.2f}px"
            )
        else:
            logger.warning("Do dich chuyen: khong tim thay marker hop le nao.")
            
        return deformed_markers_tracked, valid

    def _visualize_results(
        self, 
        deformed_proc: np.ndarray, 
        reference_markers: np.ndarray, 
        deformed_markers_tracked: np.ndarray, 
        valid: np.ndarray, 
        img_shape: tuple
    ) -> np.ndarray:
        vis_arrows = visualize_flow_arrows(
            deformed_proc,
            reference_markers,
            deformed_markers_tracked,
            valid,
            config=self.config,
            save_path=str(self.output_dir / "04_flow_arrows.png"),
        )
        logger.info("Da luu hinh anh truc quan hoa.")
        return vis_arrows

    def run(self) -> None:
        """Thuc thi tuan tu cac buoc cua pipeline."""
        # 1. Tai va Khu meo
        img_reference_raw, img_deformed_raw, img_reference, img_deformed = self._load_and_undistort_images()

        # 2. Tien xu ly
        reference_proc, deformed_proc = self._preprocess_images(img_reference, img_deformed)

        # 3. Phat hien marker
        reference_markers, vis_reference = self._detect_markers(reference_proc)

        # 4. Theo doi va Xac thuc
        deformed_markers_tracked, valid = self._track_and_analyze(
            img_reference, img_deformed, reference_markers
        )

        # 5. Truc quan hoa Flow
        vis_arrows = self._visualize_results(
            deformed_proc, reference_markers, deformed_markers_tracked, valid, img_reference.shape
        )

        # 6. Luu hinh tong hop
        self._save_summary_figure(
            img_reference_raw,
            img_deformed_raw,
            reference_proc,
            vis_reference,
            vis_arrows,
            len(reference_markers),
        )
        logger.info("Da luu hinh tong hop. Hoan thanh chay pipeline.")
