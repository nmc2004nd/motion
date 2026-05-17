"""Single-image inference for the shape classifier.

Usage:
    python -m src.shape_model.infer \
        --image data/sessions/c1/trials/trial_001/frames/frame_8920.000000.jpg
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import torch
import yaml

from .dataset import (
    TrialIndex,
    discover_trials_from_roots,
    load_image_normalized,
    normalize_shape_label,
    split_trials,
)
from .model import ShapeClassifier

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _max_frames(value: object) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    return int(value)


def _data_roots_from_config(cfg: dict, override: str | None = None) -> list[Path]:
    if override:
        return [Path(p.strip()) for p in override.split(",") if p.strip()]
    roots = cfg["data"].get("data_roots")
    if roots:
        return [Path(p) for p in roots]
    return [Path(cfg["data"]["data_root"])]


def _fallback_shape_by_session(cfg: dict) -> dict[str, str]:
    raw = cfg["data"].get("fallback_shape_by_session", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): normalize_shape_label(v) for k, v in raw.items()}


def _idx_to_label_from_ckpt(ckpt: dict) -> dict[int, str]:
    if "idx_to_label" in ckpt:
        return {int(k): str(v) for k, v in ckpt["idx_to_label"].items()}
    label_to_idx = ckpt.get("label_to_idx", {})
    return {int(v): str(k) for k, v in label_to_idx.items()}


def _trials_for_split(
    trials: list[TrialIndex],
    cfg: dict,
    split_name: str,
) -> list[TrialIndex]:
    if split_name == "all":
        return trials
    split = split_trials(
        trials,
        ratios=cfg["data"]["split"],
        seed=int(cfg["data"]["split_seed"]),
    )
    return {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }[split_name]


def choose_random_frame(
    cfg: dict,
    *,
    data_root_override: str | None,
    split_name: str,
    seed: int | None,
) -> tuple[Path, TrialIndex]:
    trials = discover_trials_from_roots(
        _data_roots_from_config(cfg, data_root_override),
        max_frames_per_trial=_max_frames(cfg["data"].get("max_frames_per_trial")),
        fallback_shape_by_session=_fallback_shape_by_session(cfg),
    )
    trials = _trials_for_split(trials, cfg, split_name)
    if not trials:
        raise RuntimeError(f"Không tìm thấy trial nào cho split '{split_name}'.")

    rng = random.Random(seed)
    trial = rng.choice(trials)
    image_path = rng.choice(trial.frames)
    return image_path, trial


@torch.no_grad()
def predict_shape(
    ckpt_path: Path,
    image_path: Path,
    config: dict,
    device: torch.device,
) -> tuple[str, float, list[tuple[str, float]]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    idx_to_label = _idx_to_label_from_ckpt(ckpt)
    if not idx_to_label:
        raise RuntimeError(f"Checkpoint thiếu label mapping: {ckpt_path}")

    image_size = tuple(int(x) for x in config["data"]["image_size"])
    img = load_image_normalized(image_path, image_size)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    model = ShapeClassifier(
        num_classes=len(idx_to_label),
        backbone=str(config["model"]["backbone"]),
        pretrained=False,
        hidden_head=int(config["model"]["hidden_head"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()
    top = torch.argsort(probs, descending=True)
    ranked = [(idx_to_label[int(i)], float(probs[int(i)])) for i in top]
    label, confidence = ranked[0]
    return label, confidence, ranked


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Predict object shape from one image.")
    parser.add_argument("--config", default="src/shape_model/config.yaml")
    parser.add_argument("--image", default=None, help="Frame image path")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Chọn ngẫu nhiên một frame từ data roots trong config.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Split dùng khi --random.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data roots khi --random; dùng dấu phẩy nếu muốn truyền nhiều root.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed chọn ảnh random.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path")
    parser.add_argument("--device", default=None, help="Override train.device")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    trial: TrialIndex | None = None
    if args.random:
        try:
            image_path, trial = choose_random_frame(
                cfg,
                data_root_override=args.data_root,
                split_name=args.split,
                seed=args.seed,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.image:
        image_path = Path(args.image)
    else:
        raise SystemExit("Cần truyền --image hoặc dùng --random.")

    if not image_path.exists():
        raise SystemExit(f"Image không tồn tại: {image_path}")

    ckpt_path = Path(args.ckpt or cfg["infer"]["checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint không tồn tại: {ckpt_path}")

    device = resolve_device(str(args.device or cfg["train"]["device"]))
    label, confidence, ranked = predict_shape(ckpt_path, image_path, cfg, device)
    print(f"Image: {image_path}")
    if trial is not None:
        print(f"Trial: {trial.session_id}/{trial.trial_id}")
        print(f"True shape: {trial.shape}")
    print(f"Predicted shape: {label} ({confidence:.3f})")
    print("Top probabilities:")
    for shape, prob in ranked:
        print(f"  {shape}: {prob:.3f}")
    logger.info("image=%s, ckpt=%s, device=%s", image_path, ckpt_path, device)


if __name__ == "__main__":
    main()
