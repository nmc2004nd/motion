"""Evaluate best checkpoint trên một split: scatter, per-trial breakdown.

Usage:
    python -m src.force_cnn.eval --config src/force_cnn/config.yaml
                                  --split test
                                  --out outputs/force_cnn/eval
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    AugmentConfig,
    ForceImageDataset,
    TrialIndex,
    discover_trials,
    split_trials,
)
from .model import ForceCNN

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    err = preds - targets
    if err.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: ForceImageDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    model.eval()
    for inp, force in loader:
        inp = inp.to(device)
        pred = model(inp)
        preds_all.append(pred.cpu().numpy())
        targets_all.append(force.numpy())
    preds = np.concatenate(preds_all) if preds_all else np.zeros(0)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0)
    return preds, targets, list(dataset._index)


def per_trial_breakdown(
    preds: np.ndarray,
    targets: np.ndarray,
    index: list[tuple[int, int]],
    trials: list[TrialIndex],
) -> list[dict]:
    by_trial: dict[int, list[int]] = {}
    for k, (ti, _fi) in enumerate(index):
        by_trial.setdefault(ti, []).append(k)
    rows: list[dict] = []
    for ti, ks in sorted(by_trial.items()):
        ks_arr = np.array(ks)
        m = _metrics(preds[ks_arr], targets[ks_arr])
        t = trials[ti]
        rows.append({
            "trial": f"{t.session_id}/{t.trial_id}",
            "n_frames": len(ks_arr),
            **m,
        })
    return rows


def _plot_scatter(preds: np.ndarray, targets: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, s=8, alpha=0.5, c="#0d6efd")
    lo = float(min(targets.min(), preds.min()))
    hi = float(max(targets.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Force ground truth (N)")
    ax.set_ylabel("Force predicted (N)")
    m = _metrics(preds, targets)
    ax.set_title(
        f"Predicted vs Actual  |  MAE={m['mae']:.3f}  "
        f"RMSE={m['rmse']:.3f}  R²={m['r2']:+.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_timeseries_first_trial(
    preds: np.ndarray,
    targets: np.ndarray,
    index: list[tuple[int, int]],
    trials: list[TrialIndex],
    out_path: Path,
) -> None:
    """Plot trial đầu tiên trong split: predicted vs ground truth theo frame index."""
    import matplotlib.pyplot as plt
    if not trials:
        return
    target_ti = 0
    ks = [k for k, (ti, _fi) in enumerate(index) if ti == target_ti]
    if not ks:
        return
    ks_arr = np.array(ks)
    pr = preds[ks_arr]
    tg = targets[ks_arr]
    fis = np.array([fi for k in ks for (_ti, fi) in [index[k]]])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fis, tg, "k-", lw=1.2, label="ground truth")
    ax.plot(fis, pr, "r-", lw=1.0, alpha=0.8, label="predicted")
    ax.set_xlabel("frame index trong trial")
    ax.set_ylabel("force (N)")
    m = _metrics(pr, tg)
    t = trials[target_ti]
    ax.set_title(
        f"{t.session_id}/{t.trial_id} | MAE={m['mae']:.3f} N  R²={m['r2']:+.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Eval ForceCNN best checkpoint.")
    parser.add_argument("--config", default="src/force_cnn/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", default="outputs/force_cnn/eval")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    sessions_root = Path(cfg["data"]["sessions_root"])
    trials = discover_trials(
        sessions_root,
        force_sync_tolerance_s=float(cfg["data"]["force_sync_tolerance_s"]),
    )
    if not trials:
        raise SystemExit(f"Không có trial nào trong {sessions_root}")

    split = split_trials(
        trials, ratios=cfg["data"]["split"], seed=cfg["data"]["split_seed"],
    )
    chosen = {"train": split.train, "val": split.val, "test": split.test}[args.split]
    if not chosen:
        raise SystemExit(f"Split '{args.split}' rỗng.")

    image_size = tuple(cfg["data"]["image_size"])
    ds = ForceImageDataset(
        chosen, image_size=image_size, augment=AugmentConfig(enabled=False),
    )

    ckpt_path = Path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Không tìm thấy checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ForceCNN(
        in_channels=2,
        backbone=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"].get("pretrained", False)),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets, index = predict_dataset(
        model, ds, device, batch_size=int(cfg["train"]["batch_size"]),
    )
    overall = _metrics(preds, targets)
    logger.info(
        "%s split: %d frames, MAE=%.4f, RMSE=%.4f, R²=%+.3f",
        args.split, len(preds), overall["mae"], overall["rmse"], overall["r2"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = per_trial_breakdown(preds, targets, index, chosen)
    print()
    print(f"=== Per-trial ({args.split}) ===")
    print(f"{'trial':<60} {'n':>5} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    for r in rows:
        print(
            f"{r['trial']:<60} {r['n_frames']:>5} "
            f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>+8.3f}"
        )

    np.savez(
        out_dir / f"{args.split}_predictions.npz",
        preds=preds, targets=targets, **overall,
    )
    with open(out_dir / f"{args.split}_breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial", "n_frames", "mae", "rmse", "r2"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if not args.no_plots and len(preds) > 0:
        try:
            _plot_scatter(preds, targets, out_dir / f"{args.split}_scatter.png")
            _plot_timeseries_first_trial(
                preds, targets, index, chosen,
                out_dir / f"{args.split}_timeseries_trial0.png",
            )
            logger.info("Plots → %s", out_dir)
        except ImportError:
            logger.warning("matplotlib chưa cài — bỏ qua plots.")


if __name__ == "__main__":
    main()
