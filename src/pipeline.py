"""Điều phối pipeline cho bài toán theo dõi marker tactile."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detection import detect_markers
from .preprocessing import preprocess
from .tracking import track_markers_lk
from .visualization import visualize_flow_arrows, visualize_flow_hsv


@dataclass
class PipelineConfig:
    ref_image_path: str = "data/ref/my_photo_1.jpg"
    def_image_path: str = "data/img/my_photo_2.jpg"
    output_dir: str = "outputs"
    arrow_scale: float = 3.0


class TactileMarkerTrackingPipeline:
    """Pipeline theo dõi marker tactile đầu-cuối với các step tách module."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_gray(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img

    def _save_summary_figure(
        self,
        img_ref: np.ndarray,
        img_def: np.ndarray,
        ref_proc: np.ndarray,
        vis_ref: np.ndarray,
        vis_arrows: np.ndarray,
        vis_hsv: np.ndarray,
        marker_count: int,
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes[0, 0].imshow(img_ref, cmap="gray")
        axes[0, 0].set_title("1. Raw reference")
        axes[0, 1].imshow(img_def, cmap="gray")
        axes[0, 1].set_title("2. Raw deformed")
        axes[0, 2].imshow(ref_proc, cmap="gray")
        axes[0, 2].set_title("3. Preprocessed")

        axes[1, 0].imshow(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"4. Markers in ref: {marker_count}")
        axes[1, 1].imshow(cv2.cvtColor(vis_arrows, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5. Flow arrows")
        axes[1, 2].imshow(cv2.cvtColor(vis_hsv, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("6. Dense HSV flow")

        for axis in axes.ravel():
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "06_summary.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def run(self) -> None:
        img_ref = self._read_gray(self.config.ref_image_path)
        img_def = self._read_gray(self.config.def_image_path)

        # Bước 1: Tiền xử lý
        ref_proc = preprocess(img_ref)
        def_proc = preprocess(img_def)
        cv2.imwrite(str(self.output_dir / "02_ref_preprocessed.png"), ref_proc)
        cv2.imwrite(str(self.output_dir / "02_def_preprocessed.png"), def_proc)
        print("[1] Preprocessing done")

        # Bước 2: Phát hiện marker trong ảnh tham chiếu
        ref_markers, ref_keypoints = detect_markers(ref_proc)
        print(f"[2] Detected {len(ref_markers)} markers in reference")
        vis_ref = cv2.drawKeypoints(
            ref_proc,
            ref_keypoints,
            None,
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(str(self.output_dir / "03_ref_markers.png"), vis_ref)

        # Đồng thời quan sát kết quả phát hiện ngây thơ ở frame biến dạng
        def_markers_naive, def_keypoints_naive = detect_markers(def_proc)
        print(
            f"    Naive detection on deformed: {len(def_markers_naive)} markers "
            "(count mismatch = correspondence issue!)"
        )
        vis_def_naive = cv2.drawKeypoints(
            def_proc,
            def_keypoints_naive,
            None,
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(str(self.output_dir / "03_def_markers_naive.png"), vis_def_naive)

        # Bước 3: Track marker tham chiếu sang frame biến dạng (đảm bảo tương ứng)
        def_markers_tracked, valid = track_markers_lk(ref_proc, def_proc, ref_markers)
        print(f"[3] Tracked {valid.sum()}/{len(ref_markers)} markers successfully")

        # Bước 4: Thống kê độ dịch chuyển
        if valid.any():
            displacements = def_markers_tracked - ref_markers
            mags = np.linalg.norm(displacements[valid], axis=1)
            print(
                f"[4] Displacement: mean={mags.mean():.2f}px, "
                f"max={mags.max():.2f}px, min={mags.min():.2f}px"
            )
        else:
            print("[4] Displacement: no valid tracked markers")

        # Bước 5: Trực quan hóa
        vis_arrows = visualize_flow_arrows(
            def_proc,
            ref_markers,
            def_markers_tracked,
            valid,
            scale=self.config.arrow_scale,
            save_path=str(self.output_dir / "04_flow_arrows.png"),
        )
        vis_hsv = visualize_flow_hsv(
            ref_markers,
            def_markers_tracked,
            valid,
            img_ref.shape,
            save_path=str(self.output_dir / "05_flow_hsv.png"),
        )
        print("[5] Visualizations saved")

        # Hình tổng hợp các bước
        self._save_summary_figure(
            img_ref,
            img_def,
            ref_proc,
            vis_ref,
            vis_arrows,
            vis_hsv,
            len(ref_markers),
        )
        print("[6] Summary figure saved")
