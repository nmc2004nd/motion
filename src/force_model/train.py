"""Training loop: trial-level split, Huber loss, Adam, early stop, save best.

Usage:
    python -m src.force_model.train --config src/force_model/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from . import dataset as ds_mod
from .dataset import AugmentConfig, build_datasets, compute_n_max, discover_trials, split_trials
from .model import ForceNet, count_parameters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
#  Config loader (đơn giản, không reuse src.config schema)                       #
# ---------------------------------------------------------------------------- #


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------- #
#  Metrics                                                                       #
# ---------------------------------------------------------------------------- #


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------- #
#  Train / eval epoch                                                            #
# ---------------------------------------------------------------------------- #


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
        for feat, mask, force in loader:
            feat = feat.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            force = force.to(device, non_blocking=True)

            pred = model(feat, mask)
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


# ---------------------------------------------------------------------------- #
#  Main                                                                          #
# ---------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train ForceNet on cached trials.")
    parser.add_argument("--config", default="src/force_model/config.yaml")
    parser.add_argument("--cache-dir", default=None,
                        help="Override data.cache_dir trong config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train.num_epochs")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    cache_dir = Path(args.cache_dir or cfg["data"]["cache_dir"])
    if not cache_dir.is_dir():
        raise SystemExit(f"cache_dir không tồn tại: {cache_dir}. Chạy prepare.py trước.")

    cache_paths = discover_trials(cache_dir)
    if not cache_paths:
        raise SystemExit(f"Không tìm thấy .npz trong {cache_dir}")

    logger.info("Discovered %d trial caches", len(cache_paths))

    split = split_trials(cache_paths, ratios=cfg["data"]["split"], seed=cfg["data"]["split_seed"])
    logger.info(
        "Split: train=%d, val=%d, test=%d trials",
        len(split.train), len(split.val), len(split.test),
    )
    if not split.train or not split.val:
        raise SystemExit(
            f"Cần ít nhất 1 trial cho train và 1 cho val (có {len(split.train)} train, "
            f"{len(split.val)} val). Thu thêm trial."
        )

    n_max = compute_n_max(cache_paths)
    logger.info("N_max markers (across trials) = %d", n_max)

    aug_cfg = AugmentConfig(**cfg["augment"])
    train_ds, val_ds, test_ds = build_datasets(split, n_max=n_max, augment_cfg=aug_cfg)
    logger.info(
        "Dataset frames: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw) if len(test_ds) > 0 else None

    device = resolve_device(cfg["train"]["device"])
    logger.info("Device: %s", device)

    model = ForceNet(
        in_dim=4,
        hidden_per_point=tuple(cfg["model"]["hidden_per_point"]),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    logger.info("Model params: %d", count_parameters(model))

    criterion = nn.HuberLoss(delta=float(cfg["train"]["huber_delta"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    n_epochs = int(args.epochs or cfg["train"]["num_epochs"])
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
            model, train_loader, device,
            optimizer=optimizer, criterion=criterion,
        )
        train_metrics = compute_metrics(tr_preds, tr_targets)

        val_loss, va_preds, va_targets = run_epoch(
            model, val_loader, device,
            optimizer=None, criterion=criterion,
        )
        val_metrics = compute_metrics(va_preds, va_targets)

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
                "n_max": n_max,
                "config": cfg,
            }, ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                logger.info("Early stop sau %d epoch không cải thiện.", patience)
                break

    elapsed = time.time() - t0
    logger.info("Train xong sau %.1fs. Best val MAE = %.4f tại epoch %d", elapsed, best_val_mae, best_epoch)

    # Save history
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Test eval với best checkpoint
    if test_loader is not None:
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        _, te_preds, te_targets = run_epoch(
            model, test_loader, device,
            optimizer=None, criterion=criterion,
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
