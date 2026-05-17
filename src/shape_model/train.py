"""Train a frame-based object-shape classifier.

Usage:
    python -m src.shape_model.train --config src/shape_model/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import (
    AugmentConfig,
    build_datasets,
    build_label_map,
    discover_trials_from_roots,
    frame_counts_by_shape,
    normalize_shape_label,
    split_trials,
    trial_counts_by_shape,
)
from .model import ShapeClassifier, count_parameters

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


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    if targets.size == 0:
        return {"accuracy": float("nan"), "macro_accuracy": float("nan")}
    accuracy = float(np.mean(preds == targets))
    per_class_acc: list[float] = []
    for cls_idx in range(num_classes):
        mask = targets == cls_idx
        if mask.any():
            per_class_acc.append(float(np.mean(preds[mask] == targets[mask])))
    macro_accuracy = float(np.mean(per_class_acc)) if per_class_acc else float("nan")
    return {"accuracy": accuracy, "macro_accuracy": macro_accuracy}


def class_weights_from_dataset(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(targets, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def dataset_targets(dataset) -> list[int]:
    targets: list[int] = []
    for trial_idx, _frame_idx in dataset._index:
        shape = dataset.trials[trial_idx].shape
        targets.append(dataset.label_to_idx[shape])
    return targets


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            preds_all.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    preds = np.concatenate(preds_all) if preds_all else np.zeros(0, dtype=np.int64)
    targets_np = (
        np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.int64)
    )
    return avg_loss, preds, targets_np


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train object shape classifier.")
    parser.add_argument("--config", default="src/shape_model/config.yaml")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data roots; dùng dấu phẩy nếu muốn truyền nhiều root.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.num_epochs")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    data_roots = _data_roots_from_config(cfg, args.data_root)
    trials = discover_trials_from_roots(
        data_roots,
        max_frames_per_trial=_max_frames(cfg["data"].get("max_frames_per_trial")),
        fallback_shape_by_session=_fallback_shape_by_session(cfg),
    )
    if not trials:
        raise SystemExit(f"Không tìm thấy trial có label shape trong {data_roots}")

    label_to_idx = build_label_map(trials, classes=cfg["data"].get("classes"))
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    logger.info("Discovered %d trials, %d frames", len(trials), sum(t.n_frames for t in trials))
    logger.info("Trial counts by shape: %s", trial_counts_by_shape(trials))
    logger.info("Frame counts by shape: %s", frame_counts_by_shape(trials))
    logger.info("Label map: %s", label_to_idx)
    missing_classes = sorted(set(label_to_idx) - {t.shape for t in trials})
    if missing_classes:
        logger.warning("Các class chưa có sample trong data hiện tại: %s", missing_classes)
    if len({t.shape for t in trials}) == 1:
        logger.warning(
            "Chỉ có 1 shape label. Model vẫn chạy, nhưng chưa học phân biệt shape; "
            "hãy thêm dữ liệu cho các shape khác để đánh giá thật."
        )

    split = split_trials(
        trials,
        ratios=cfg["data"]["split"],
        seed=int(cfg["data"]["split_seed"]),
    )
    if not split.train or not split.val:
        raise SystemExit(
            f"Cần ít nhất 1 trial train và 1 trial val "
            f"(train={len(split.train)}, val={len(split.val)})."
        )

    image_size = tuple(int(x) for x in cfg["data"]["image_size"])
    augment_cfg = AugmentConfig(**cfg["augment"])
    train_ds, val_ds, test_ds = build_datasets(
        split,
        label_to_idx,
        image_size,
        augment_cfg,
    )
    logger.info(
        "Split trials: train=%d, val=%d, test=%d",
        len(split.train), len(split.val), len(split.test),
    )
    logger.info(
        "Split frames: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if len(test_ds) > 0
        else None
    )

    device = resolve_device(str(cfg["train"]["device"]))
    model = ShapeClassifier(
        num_classes=len(label_to_idx),
        backbone=str(cfg["model"]["backbone"]),
        pretrained=bool(cfg["model"].get("pretrained", False)),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    logger.info(
        "Model: %s, classes=%d, params=%d, device=%s",
        cfg["model"]["backbone"],
        len(label_to_idx),
        count_parameters(model),
        device,
    )

    weights = class_weights_from_dataset(
        dataset_targets(train_ds),
        num_classes=len(label_to_idx),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    n_epochs = int(args.epochs or cfg["train"]["num_epochs"])
    scheduler = None
    if str(cfg["train"].get("scheduler", "none")) == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    log_dir = Path(cfg["train"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_best = 0
    patience = int(cfg["train"]["early_stop_patience"])
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_preds, train_targets = run_epoch(
            model,
            train_loader,
            device,
            criterion=criterion,
            optimizer=optimizer,
        )
        val_loss, val_preds, val_targets = run_epoch(
            model,
            val_loader,
            device,
            criterion=criterion,
            optimizer=None,
        )
        if scheduler is not None:
            scheduler.step()

        train_metrics = compute_metrics(train_preds, train_targets, len(label_to_idx))
        val_metrics = compute_metrics(val_preds, val_targets, len(label_to_idx))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_accuracy": train_metrics["macro_accuracy"],
            "val_macro_accuracy": val_metrics["macro_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        logger.info(
            "Ep %3d | train loss %.4f acc %.3f | val loss %.4f acc %.3f macro %.3f",
            epoch,
            train_loss,
            train_metrics["accuracy"],
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_accuracy"],
        )

        if val_metrics["accuracy"] > best_val_acc + 1e-9:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_accuracy": best_val_acc,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                logger.info("Early stop sau %d epoch không cải thiện.", patience)
                break

    elapsed = time.time() - t0
    logger.info(
        "Train xong sau %.1fs. Best val accuracy = %.3f tại epoch %d",
        elapsed,
        best_val_acc,
        best_epoch,
    )

    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(log_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    if test_loader is not None and (ckpt_dir / "best.pt").exists():
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        _, test_preds, test_targets = run_epoch(
            model,
            test_loader,
            device,
            criterion=criterion,
            optimizer=None,
        )
        test_metrics = compute_metrics(test_preds, test_targets, len(label_to_idx))
        logger.info(
            "TEST (best ckpt): acc=%.3f, macro_acc=%.3f",
            test_metrics["accuracy"],
            test_metrics["macro_accuracy"],
        )
        np.savez(
            log_dir / "test_predictions.npz",
            preds=test_preds,
            targets=test_targets,
            **test_metrics,
        )
    else:
        logger.warning("Không có test split hoặc chưa có checkpoint best.pt.")


if __name__ == "__main__":
    main()
