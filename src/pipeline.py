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
from .Hungarian.Hungarian import match_markers_robust
from .PyrLK.PyrLK import track_markers_lk
from .utils.visualization import visualize_flow_arrows, visualize_flow_hsv

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    reference_image_path: str = "data/ref/my_photo_1.jpg"
    deformed_image_path: str = "data/img/sample_0010.png"
    calib_file_path: str = "config/calib_result.npz"
    output_dir: str = "outputs"
    arrow_scale: float = 1.0
    tracking_method: str = "H"  # 'H' cho Hungarian, 'LK' cho PyrLK
    use_calibration: bool = False


class TactileMarkerTrackingPipeline:
    """Pipeline theo dõi marker tactile từ đầu đến cuối với các bước được module hóa."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir) / self.config.tracking_method
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        vis_hsv: np.ndarray,
        marker_count: int,
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
        axes[1, 2].imshow(cv2.cvtColor(vis_hsv, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("6. Dense HSV flow")

        for axis in axes.ravel():
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "06_summary.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _load_and_undistort_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_reference_raw = self._read_gray(self.config.reference_image_path)
        img_deformed_raw = self._read_gray(self.config.deformed_image_path)

        if self.config.use_calibration:
            logger.info(f"Dang tai hieu chuan camera tu {self.config.calib_file_path}...")
            camera_matrix, dist_coeffs = load_calibration(self.config.calib_file_path)
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
        reference_proc = preprocess(img_reference)
        deformed_proc = preprocess(img_deformed)
        cv2.imwrite(str(self.output_dir / "02_reference_preprocessed.png"), reference_proc)
        cv2.imwrite(str(self.output_dir / "02_deformed_preprocessed.png"), deformed_proc)
        logger.info("Qua trinh tien xu ly hoan tat.")
        return reference_proc, deformed_proc

    def _detect_markers(self, reference_proc: np.ndarray, deformed_proc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        reference_markers, reference_keypoints = detect_markers(reference_proc)
        logger.info(f"Da phat hien {len(reference_markers)} marker trong anh tham chieu.")
        vis_reference = cv2.drawKeypoints(
            reference_proc,
            reference_keypoints,
            None,
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(str(self.output_dir / "03_reference_markers.png"), vis_reference)

        # Phat hien "ngay tho" tren frame bien dang de the hien cac van de ve su tuong ung
        deformed_markers_naive, deformed_keypoints_naive = detect_markers(deformed_proc)
        logger.info(
            f"Phat hien ngay tho tren anh bien dang: {len(deformed_markers_naive)} markers "
            "(sai lech so luong = van de ve su tuong ung!)"
        )
        vis_deformed_naive = cv2.drawKeypoints(
            deformed_proc,
            deformed_keypoints_naive,
            None,
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(str(self.output_dir / "03_deformed_markers_naive.png"), vis_deformed_naive)
        
        return reference_markers, deformed_markers_naive, vis_reference

    def _track_and_analyze(
        self, 
        img_reference: np.ndarray, 
        img_deformed: np.ndarray, 
        reference_markers: np.ndarray, 
        deformed_markers_naive: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.tracking_method == "LK":
            deformed_markers_tracked, valid = track_markers_lk(img_reference, img_deformed, reference_markers)
        else:
            max_displacement = img_reference.shape[1] / 10.0
            deformed_markers_tracked, valid = match_markers_robust(
                reference_markers, deformed_markers_naive, img_reference.shape, max_disp=max_displacement
            )
            
        logger.info(f"Da theo doi thanh cong {valid.sum()}/{len(reference_markers)} markers (phuong phap: {self.config.tracking_method}).")

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
    ) -> tuple[np.ndarray, np.ndarray]:
        vis_arrows = visualize_flow_arrows(
            deformed_proc,
            reference_markers,
            deformed_markers_tracked,
            valid,
            scale=self.config.arrow_scale,
            save_path=str(self.output_dir / "04_flow_arrows.png"),
        )
        vis_hsv = visualize_flow_hsv(
            reference_markers,
            deformed_markers_tracked,
            valid,
            img_shape,
            save_path=str(self.output_dir / "05_flow_hsv.png"),
        )
        logger.info("Da luu hinh anh truc quan hoa.")
        return vis_arrows, vis_hsv

    def run(self) -> None:
        """Thuc thi tuan tu cac buoc cua pipeline."""
        # 1. Tai va Khu meo
        img_reference_raw, img_deformed_raw, img_reference, img_deformed = self._load_and_undistort_images()

        # 2. Tien xu ly
        reference_proc, deformed_proc = self._preprocess_images(img_reference, img_deformed)

        # 3. Phat hien marker
        reference_markers, deformed_markers_naive, vis_reference = self._detect_markers(reference_proc, deformed_proc)

        # 4. Theo doi va Xac thuc
        deformed_markers_tracked, valid = self._track_and_analyze(
            img_reference, img_deformed, reference_markers, deformed_markers_naive
        )

        # 5. Truc quan hoa Flow
        vis_arrows, vis_hsv = self._visualize_results(
            deformed_proc, reference_markers, deformed_markers_tracked, valid, img_reference.shape
        )

        # 6. Luu hinh tong hop
        self._save_summary_figure(
            img_reference_raw,
            img_deformed_raw,
            reference_proc,
            vis_reference,
            vis_arrows,
            vis_hsv,
            len(reference_markers),
        )
        logger.info("Da luu hinh tong hop. Hoan thanh chay pipeline.")
