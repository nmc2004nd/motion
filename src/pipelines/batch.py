"""Pipeline batch — xử lý một cặp ảnh (ref / deformed) tĩnh và xuất 6 ảnh tổng hợp."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..config import as_int_tuple, require
from ..core.calibration import load_calibration, undistort_image
from ..core.detection import detect_markers
from ..core.preprocessing import preprocess
from ..core.tracking import track_markers_lk
from ..core.visualization import visualize_flow_arrows

logger = logging.getLogger(__name__)


# Tên file output cố định theo thứ tự bước pipeline (không phụ thuộc tham số).
_FILE_REF_UNDIST = "01_reference_undistorted.png"
_FILE_DEF_UNDIST = "01_deformed_undistorted.png"
_FILE_REF_PROC = "02_reference_preprocessed.png"
_FILE_DEF_PROC = "02_deformed_preprocessed.png"
_FILE_REF_MARKERS = "03_reference_markers.png"
_FILE_FLOW_ARROWS = "04_flow_arrows.png"
_FILE_SUMMARY = "06_summary.png"


class TactileMarkerTrackingPipeline:
    """Pipeline theo dõi marker cho cặp ảnh tĩnh (batch mode)."""

    def __init__(self, config: dict) -> None:
        self.config = config
        base_dir = Path(require(config, "paths.output_dir"))
        self.output_dir = base_dir / str(require(config, "paths.batch_output_subdir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_calib = bool(require(config, "pipeline.use_calibration"))
        self.ref_image_path = str(require(config, "paths.ref_image"))
        self.def_image_path = str(require(config, "paths.deformed_image"))

        self.summary_figsize = as_int_tuple(
            require(config, "visualization.summary_figsize"), "summary_figsize"
        )
        self.summary_dpi = int(require(config, "visualization.summary_dpi"))

    def run(self) -> None:
        img_ref_raw, img_def_raw, img_ref, img_def = self._load_images()
        ref_proc, def_proc = self._preprocess(img_ref, img_def)
        ref_markers, vis_ref = self._detect(ref_proc)
        def_markers, valid = self._track(img_ref, img_def, ref_markers)
        vis_arrows = self._visualize(def_proc, ref_markers, def_markers, valid)
        self._save_summary(img_ref_raw, img_def_raw, ref_proc, vis_ref, vis_arrows, len(ref_markers))
        logger.info("Đã lưu hình tổng hợp. Hoàn thành chạy pipeline.")

    # ---------- Pipeline steps ----------

    def _load_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_ref_raw = _read_gray(self.ref_image_path)
        img_def_raw = _read_gray(self.def_image_path)

        if not self.use_calib:
            return img_ref_raw, img_def_raw, img_ref_raw, img_def_raw

        logger.info("Đang tải hiệu chuẩn camera từ %s...", require(self.config, "paths.calib_file"))
        camera_matrix, dist_coeffs = load_calibration(self.config)
        img_ref = undistort_image(img_ref_raw, camera_matrix, dist_coeffs)
        img_def = undistort_image(img_def_raw, camera_matrix, dist_coeffs)
        cv2.imwrite(str(self.output_dir / _FILE_REF_UNDIST), img_ref)
        cv2.imwrite(str(self.output_dir / _FILE_DEF_UNDIST), img_def)
        logger.info("Đã khử méo ảnh thành công.")
        return img_ref_raw, img_def_raw, img_ref, img_def

    def _preprocess(
        self, img_ref: np.ndarray, img_def: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_proc = preprocess(img_ref, config=self.config)
        def_proc = preprocess(img_def, config=self.config)
        cv2.imwrite(str(self.output_dir / _FILE_REF_PROC), ref_proc)
        cv2.imwrite(str(self.output_dir / _FILE_DEF_PROC), def_proc)
        logger.info("Quá trình tiền xử lý hoàn tất.")
        return ref_proc, def_proc

    def _detect(self, ref_proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoint_color = tuple(require(self.config, "visualization.keypoint_color"))
        centers, keypoints = detect_markers(ref_proc, config=self.config)
        logger.info("Đã phát hiện %d marker trong ảnh tham chiếu.", len(centers))
        vis_ref = cv2.drawKeypoints(
            ref_proc, keypoints, None, keypoint_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(str(self.output_dir / _FILE_REF_MARKERS), vis_ref)
        return centers, vis_ref

    def _track(
        self, img_ref: np.ndarray, img_def: np.ndarray, ref_markers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tracked, valid = track_markers_lk(img_ref, img_def, ref_markers, config=self.config)
        logger.info("Đã theo dõi thành công %d/%d markers (LK).", valid.sum(), len(ref_markers))

        if valid.any():
            disps = tracked - ref_markers
            mags = np.linalg.norm(disps[valid], axis=1)
            logger.info(
                "Độ dịch chuyển: trung bình=%.2fpx, max=%.2fpx, min=%.2fpx",
                mags.mean(), mags.max(), mags.min(),
            )
        else:
            logger.warning("Độ dịch chuyển: không tìm thấy marker hợp lệ nào.")
        return tracked, valid

    def _visualize(
        self,
        def_proc: np.ndarray,
        ref_markers: np.ndarray,
        def_markers: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        vis = visualize_flow_arrows(
            def_proc, ref_markers, def_markers, valid,
            config=self.config, save_path=str(self.output_dir / _FILE_FLOW_ARROWS),
        )
        logger.info("Đã lưu hình ảnh trực quan hóa.")
        return vis

    def _save_summary(
        self,
        img_ref_raw: np.ndarray,
        img_def_raw: np.ndarray,
        ref_proc: np.ndarray,
        vis_ref: np.ndarray,
        vis_arrows: np.ndarray,
        marker_count: int,
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=self.summary_figsize)
        axes[0, 0].imshow(img_ref_raw, cmap="gray")
        axes[0, 0].set_title("1. Raw reference")
        axes[0, 1].imshow(img_def_raw, cmap="gray")
        axes[0, 1].set_title("2. Raw deformed")
        axes[0, 2].imshow(ref_proc, cmap="gray")
        axes[0, 2].set_title("3. Preprocessed")

        axes[1, 0].imshow(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"4. Markers in reference: {marker_count}")
        axes[1, 1].imshow(cv2.cvtColor(vis_arrows, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5. Flow arrows")
        axes[1, 2].axis("off")

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / _FILE_SUMMARY, dpi=self.summary_dpi, bbox_inches="tight")
        plt.close(fig)


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
