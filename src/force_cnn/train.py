"""Training loop cho ForceCNN: trial-split, Huber, AdamW, cosine LR, early stop.

Usage:
    python -m src.force_cnn.train --config src/force_cnn/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    discover_trials,
    split_trials,
)
from .model import ForceCNN, count_parameters

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    err = preds - targets
    if err.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Trả (avg_loss, preds, targets). Optimizer=None → eval mode."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for inp, force in loader:
            inp = inp.to(device, non_blocking=True)
            force = force.to(device, non_blocking=True)

            pred = model(inp)
            loss = criterion(pred, force)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(force.detach().cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    preds = np.concatenate(preds_all) if preds_all else np.zeros(0)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0)
    return avg_loss, preds, targets


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train ForceCNN end-to-end.")
    parser.add_argument("--config", default="src/force_cnn/config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train.num_epochs")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    sessions_root = Path(cfg["data"]["sessions_root"])
    trials = discover_trials(
        sessions_root,
        force_sync_tolerance_s=float(cfg["data"]["force_sync_tolerance_s"]),
    )
    if not trials:
        raise SystemExit(
            f"Không tìm thấy trial nào trong {sessions_root}. "
            f"Thu thập data với src/collection trước."
        )
    logger.info("Discovered %d trials, %d total frames",
                len(trials), sum(t.n_frames for t in trials))

    split = split_trials(
        trials, ratios=cfg["data"]["split"], seed=cfg["data"]["split_seed"],
    )
    if not split.train or not split.val:
        raise SystemExit(
            f"Cần >=1 trial cho train và >=1 cho val (có {len(split.train)} train, "
            f"{len(split.val)} val). Thu thêm trial."
        )
    logger.info(
        "Split: train=%d, val=%d, test=%d trials",
        len(split.train), len(split.val), len(split.test),
    )

    image_size = tuple(cfg["data"]["image_size"])
    aug_cfg = AugmentConfig(**cfg["augment"])
    train_ds, val_ds, test_ds = build_datasets(split, image_size, aug_cfg)
    logger.info(
        "Frames: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
        if len(test_ds) > 0 else None
    )

    device = resolve_device(cfg["train"]["device"])
    logger.info("Device: %s", device)

    model = ForceCNN(
        in_channels=2,
        backbone=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"].get("pretrained", False)),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    logger.info(
        "Model: %s (pretrained=%s), params: %d",
        cfg["model"]["backbone"], cfg["model"].get("pretrained", False),
        count_parameters(model),
    )

    criterion = nn.HuberLoss(delta=float(cfg["train"]["huber_delta"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    n_epochs = int(args.epochs or cfg["train"]["num_epochs"])
    sched: CosineAnnealingLR | None = None
    if cfg["train"]["scheduler"] == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=n_epochs)

    patience = int(cfg["train"]["early_stop_patience"])
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["train"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss, tr_preds, tr_targets = run_epoch(
            model, train_loader, device, optimizer=optimizer, criterion=criterion,
        )
        train_metrics = compute_metrics(tr_preds, tr_targets)

        val_loss, va_preds, va_targets = run_epoch(
            model, val_loader, device, optimizer=None, criterion=criterion,
        )
        val_metrics = compute_metrics(va_preds, va_targets)

        if sched is not None:
            sched.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        logger.info(
            "Ep %3d | train loss %.4f mae %.4f r2 %+.3f | val loss %.4f mae %.4f r2 %+.3f",
            epoch, train_loss, train_metrics["mae"], train_metrics["r2"],
            val_loss, val_metrics["mae"], val_metrics["r2"],
        )

        if val_metrics["mae"] < best_val_mae - 1e-6:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_mae": best_val_mae,
                "config": cfg,
            }, ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                logger.info("Early stop sau %d epoch không cải thiện.", patience)
                break

    elapsed = time.time() - t0
    logger.info(
        "Train xong sau %.1fs. Best val MAE = %.4f tại epoch %d",
        elapsed, best_val_mae, best_epoch,
    )

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if test_loader is not None:
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        _, te_preds, te_targets = run_epoch(
            model, test_loader, device, optimizer=None, criterion=criterion,
        )
        test_metrics = compute_metrics(te_preds, te_targets)
        logger.info(
            "TEST (best ckpt): MAE=%.4f, RMSE=%.4f, R²=%+.3f",
            test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )
        np.savez(
            log_dir / "test_predictions.npz",
            preds=te_preds, targets=te_targets, **test_metrics,
        )
    else:
        logger.warning("Không có test split (cần >=3 trials).")


if __name__ == "__main__":
    main()
