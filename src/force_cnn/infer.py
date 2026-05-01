"""Single-image inference cho ForceCNN: (ref, frame) → scalar force.

Usage:
    # Dùng default ref (data/ref/my_photo_1.jpg) + checkpoint từ config:
    python -m src.force_cnn.infer --frame data/img/my_photo_2.jpg

    # Custom ref + checkpoint:
    python -m src.force_cnn.infer \
        --frame data/img/my_photo_2.jpg \
        --ref data/ref/my_photo_1.jpg \
        --ckpt outputs/force_cnn/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import load_gray_normalized
from .model import ForceCNN

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def predict_force(
    ckpt_path: Path,
    ref_path: Path,
    frame_path: Path,
    config: dict,
    device: torch.device,
) -> float:
    image_size = tuple(config["data"]["image_size"])
    ref = load_gray_normalized(ref_path, image_size)
    frame = load_gray_normalized(frame_path, image_size)
    inp = np.stack([ref, frame], axis=0)[None, ...]  # (1, 2, H, W)
    inp_t = torch.from_numpy(inp.astype(np.float32)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ForceCNN(
        in_channels=2,
        backbone=config["model"]["backbone"],
        pretrained=False,  # luôn False khi load checkpoint, weights đến từ ckpt
        hidden_head=int(config["model"]["hidden_head"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(inp_t)
    return float(pred.item())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Inference (ref, frame) → predicted force (N)."
    )
    parser.add_argument("--config", default="src/force_cnn/config.yaml")
    parser.add_argument("--frame", required=True, help="Đường dẫn ảnh deformed")
    parser.add_argument("--ref", default=None,
                        help="Reference image; default từ config.infer.default_ref")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path; default từ config.train.checkpoint_dir/best.pt")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    ref_path = Path(args.ref or cfg["infer"]["default_ref"])
    if not ref_path.exists():
        raise SystemExit(f"Reference không tồn tại: {ref_path}")
    frame_path = Path(args.frame)
    if not frame_path.exists():
        raise SystemExit(f"Frame không tồn tại: {frame_path}")

    ckpt_path = Path(
        args.ckpt or (Path(cfg["train"]["checkpoint_dir"]) / "best.pt")
    )
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint không tồn tại: {ckpt_path}. "
            f"Train trước hoặc chỉ định --ckpt."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force = predict_force(ckpt_path, ref_path, frame_path, cfg, device)
    print(f"Predicted force: {force:.4f} N")
    logger.info("ref=%s, frame=%s, ckpt=%s, device=%s",
                ref_path, frame_path, ckpt_path, device)


if __name__ == "__main__":
    main()
