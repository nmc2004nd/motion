"""Single-pair inference: (ref, frame) → predicted force (N).

Reuse pipeline detect + LK tracking từ src/core để compute features tại runtime.

Usage:
    python -m src.force_poly.infer --frame data/img/my_photo_2.jpg

    python -m src.force_poly.infer \
        --frame data/img/my_photo_2.jpg \
        --ref data/ref/my_photo_1.jpg \
        --ckpt outputs/force_poly/checkpoints/best.pt \
        --pipeline-config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ..config import load_config
from ..core.detection import create_blob_detector, detect_markers
from ..core.preprocessing import make_clahe, preprocess
from ..core.tracking import track_markers_lk
from .features import compute_features_v1, feature_dim
from .model import PolynomialRegressor

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def predict_force(
    ckpt_path: Path,
    ref_path: Path,
    frame_path: Path,
    config: dict,
    pipeline_config: dict,
    device: torch.device,
) -> tuple[float, dict]:
    """Trả (force_pred, debug_info)."""
    ref_bgr = cv2.imread(str(ref_path))
    if ref_bgr is None:
        raise RuntimeError(f"Không đọc được reference: {ref_path}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise RuntimeError(f"Không đọc được frame: {frame_path}")
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Detect marker trên reference (đã preprocess).
    clahe = make_clahe(pipeline_config)
    detector = create_blob_detector(pipeline_config)
    ref_proc = preprocess(ref_gray, config=pipeline_config, _clahe=clahe)
    ref_pts, _ = detect_markers(ref_proc, config=pipeline_config, _detector=detector)
    if ref_pts.shape[0] == 0:
        raise RuntimeError(
            f"Không detect được marker trong reference {ref_path.name} — "
            f"kiểm tra detection config."
        )

    tracked, valid = track_markers_lk(
        ref_gray, frame_gray, ref_pts, config=pipeline_config, apply_deadzone=False,
    )
    disp = (tracked - ref_pts).astype(np.float32)
    image_w = int(ref_gray.shape[1])

    feature_set = str(config["model"]["feature_set"])
    feat = compute_features_v1(ref_pts, disp, valid, image_w)

    model = PolynomialRegressor(
        n_features=feature_dim(feature_set),
        degree=int(config["model"]["degree"]),
        head=str(config["model"]["head"]),
        hidden_head=int(config["model"]["hidden_head"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(feat[None, :]).to(device)
        force = float(model(x).item())

    return force, {
        "n_markers": int(ref_pts.shape[0]),
        "n_valid": int(valid.sum()),
        "feature": feat.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Inference (ref, frame) → predicted force (N).",
    )
    parser.add_argument("--config", default="src/force_poly/config.yaml")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yaml",
                        help="Pipeline config cho detection + tracking params")
    parser.add_argument("--frame", required=True, help="Đường dẫn ảnh deformed")
    parser.add_argument("--ref", required=True, help="Reference image (trạng thái tĩnh)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path; default từ config.train.checkpoint_dir/best.pt")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    pipeline_cfg = load_config(args.pipeline_config)

    ref_path = Path(args.ref)
    if not ref_path.exists():
        raise SystemExit(f"Reference không tồn tại: {ref_path}")
    frame_path = Path(args.frame)
    if not frame_path.exists():
        raise SystemExit(f"Frame không tồn tại: {frame_path}")

    ckpt_path = Path(args.ckpt or (Path(cfg["train"]["checkpoint_dir"]) / "best.pt"))
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint không tồn tại: {ckpt_path}. Train trước hoặc --ckpt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force, info = predict_force(
        ckpt_path, ref_path, frame_path, cfg, pipeline_cfg, device,
    )
    print(f"Predicted force: {force:.4f} N")
    logger.info(
        "ref=%s frame=%s n_markers=%d n_valid=%d device=%s",
        ref_path, frame_path, info["n_markers"], info["n_valid"], device,
    )


if __name__ == "__main__":
    main()
