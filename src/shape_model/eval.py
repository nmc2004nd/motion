"""Evaluate a trained shape classifier on a trial-level split.

Usage:
    python -m src.shape_model.eval --config src/shape_model/config.yaml --split test
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    AugmentConfig,
    ShapeImageDataset,
    discover_trials_from_roots,
    normalize_shape_label,
    split_trials,
)
from .model import ShapeClassifier
from .train import compute_metrics

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: ShapeImageDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    model.eval()
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds_all.append(torch.argmax(probs, dim=1).cpu().numpy())
        targets_all.append(targets.numpy())
        probs_all.append(probs.cpu().numpy())

    preds = np.concatenate(preds_all) if preds_all else np.zeros(0, dtype=np.int64)
    targets_np = (
        np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.int64)
    )
    probs_np = (
        np.concatenate(probs_all) if probs_all else np.zeros((0, 0), dtype=np.float32)
    )
    return preds, targets_np, probs_np


def confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds, strict=False):
        mat[int(target), int(pred)] += 1
    return mat


def per_trial_rows(
    preds: np.ndarray,
    targets: np.ndarray,
    index: list[tuple[int, int]],
    dataset: ShapeImageDataset,
    idx_to_label: dict[int, str],
) -> list[dict[str, object]]:
    by_trial: dict[int, list[int]] = {}
    for sample_idx, (trial_idx, _frame_idx) in enumerate(index):
        by_trial.setdefault(trial_idx, []).append(sample_idx)

    rows: list[dict[str, object]] = []
    for trial_idx, sample_indices in sorted(by_trial.items()):
        ks = np.array(sample_indices)
        trial = dataset.trials[trial_idx]
        trial_preds = preds[ks]
        trial_targets = targets[ks]
        values, counts = np.unique(trial_preds, return_counts=True)
        majority_idx = int(values[int(np.argmax(counts))])
        acc = float(np.mean(trial_preds == trial_targets))
        rows.append(
            {
                "trial": f"{trial.session_id}/{trial.trial_id}",
                "shape_true": trial.shape,
                "shape_majority_pred": idx_to_label[majority_idx],
                "frame_accuracy": acc,
                "n_frames": len(ks),
            }
        )
    return rows


def write_confusion_csv(mat: np.ndarray, idx_to_label: dict[int, str], path: Path) -> None:
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *mat[i].tolist()])


def plot_confusion_matrix(
    mat: np.ndarray,
    idx_to_label: dict[int, str],
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=np.float64),
        where=row_sums > 0,
    )

    fig_size = max(6.0, 1.15 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized ratio")

    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted shape")
    ax.set_ylabel("True shape")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pct = norm[i, j] * 100.0
            text_color = "white" if norm[i, j] >= 0.55 else "#1f2937"
            ax.text(
                j,
                i,
                f"{mat[i, j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_trial_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Evaluate shape classifier.")
    parser.add_argument("--config", default="src/shape_model/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data roots; dùng dấu phẩy nếu muốn truyền nhiều root.",
    )
    parser.add_argument("--ckpt", default=None, help="Checkpoint path")
    parser.add_argument("--out", default="outputs/shape_model/eval")
    parser.add_argument("--no-plots", action="store_true", help="Không xuất ảnh PNG.")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    ckpt_path = Path(args.ckpt or cfg["infer"]["checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint không tồn tại: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    idx_to_label = _idx_to_label_from_ckpt(ckpt)
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}
    if not label_to_idx:
        raise SystemExit(f"Checkpoint thiếu label mapping: {ckpt_path}")

    data_roots = _data_roots_from_config(cfg, args.data_root)
    trials = discover_trials_from_roots(
        data_roots,
        max_frames_per_trial=_max_frames(cfg["data"].get("max_frames_per_trial")),
        fallback_shape_by_session=_fallback_shape_by_session(cfg),
    )
    split = split_trials(
        trials,
        ratios=cfg["data"]["split"],
        seed=int(cfg["data"]["split_seed"]),
    )
    chosen = {"train": split.train, "val": split.val, "test": split.test}[args.split]
    if not chosen:
        raise SystemExit(f"Split '{args.split}' rỗng.")

    image_size = tuple(int(x) for x in cfg["data"]["image_size"])
    dataset = ShapeImageDataset(
        chosen,
        label_to_idx,
        image_size=image_size,
        augment=AugmentConfig(enabled=False),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeClassifier(
        num_classes=len(idx_to_label),
        backbone=str(cfg["model"]["backbone"]),
        pretrained=False,
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets, probs = predict_dataset(
        model,
        dataset,
        device,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )
    metrics = compute_metrics(preds, targets, len(idx_to_label))
    mat = confusion_matrix(preds, targets, len(idx_to_label))

    logger.info(
        "%s split: %d frames, accuracy=%.3f, macro_accuracy=%.3f",
        args.split,
        len(preds),
        metrics["accuracy"],
        metrics["macro_accuracy"],
    )
    logger.info("Confusion matrix rows=true, cols=pred:\n%s", mat)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.split}_predictions.npz",
        preds=preds,
        targets=targets,
        probs=probs,
        **metrics,
    )
    write_confusion_csv(mat, idx_to_label, out_dir / f"{args.split}_confusion.csv")
    if not args.no_plots and len(preds) > 0:
        try:
            plot_confusion_matrix(
                mat,
                idx_to_label,
                out_dir / f"{args.split}_confusion.png",
                title=(
                    f"{args.split} confusion matrix "
                    f"(acc={metrics['accuracy']:.3f}, macro={metrics['macro_accuracy']:.3f})"
                ),
            )
            logger.info("Confusion matrix image: %s", out_dir / f"{args.split}_confusion.png")
        except ImportError:
            logger.warning("matplotlib chưa cài — bỏ qua ảnh confusion matrix.")
    write_trial_csv(
        per_trial_rows(preds, targets, list(dataset._index), dataset, idx_to_label),
        out_dir / f"{args.split}_per_trial.csv",
    )


if __name__ == "__main__":
    main()
