"""Evaluate best checkpoint: predicted vs actual scatter, time-series overlay,
per-trial breakdown.

Usage:
    python -m src.force_model.eval [--config src/force_model/config.yaml]
                                    [--split test]
                                    [--out outputs/force_model/eval]
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    AugmentConfig,
    TrialDataset,
    compute_n_max,
    discover_trials,
    split_trials,
)
from .model import ForceNet

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
    dataset: TrialDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Trả (preds, targets, index) — index khớp dataset._index để map về trial."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    model.eval()
    for feat, mask, force in loader:
        feat = feat.to(device)
        mask = mask.to(device)
        pred = model(feat, mask)
        preds_all.append(pred.cpu().numpy())
        targets_all.append(force.numpy())
    preds = np.concatenate(preds_all) if preds_all else np.zeros(0)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0)
    return preds, targets, list(dataset._index)


def per_trial_breakdown(
    preds: np.ndarray,
    targets: np.ndarray,
    index: list[tuple[int, int]],
    cache_paths: list[Path],
) -> list[dict]:
    """Group preds/targets theo trial → tính metrics riêng."""
    by_trial: dict[int, list[int]] = {}
    for k, (ci, _fi) in enumerate(index):
        by_trial.setdefault(ci, []).append(k)

    rows: list[dict] = []
    for ci, ks in sorted(by_trial.items()):
        ks_arr = np.array(ks)
        m = _metrics(preds[ks_arr], targets[ks_arr])
        rows.append({
            "trial_path": str(cache_paths[ci]),
            "n_frames": len(ks_arr),
            **m,
        })
    return rows


def _plot_scatter(preds: np.ndarray, targets: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, s=8, alpha=0.5, c="#0d6efd")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Force ground truth (N)")
    ax.set_ylabel("Force predicted (N)")
    m = _metrics(preds, targets)
    ax.set_title(f"Predicted vs Actual  |  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:+.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_timeseries(
    preds: np.ndarray,
    targets: np.ndarray,
    index: list[tuple[int, int]],
    cache_paths: list[Path],
    out_path: Path,
    trial_ci: int,
) -> None:
    """Plot 1 trial: GT force và predicted theo thời gian (frame index)."""
    import matplotlib.pyplot as plt
    ks = [k for k, (ci, _fi) in enumerate(index) if ci == trial_ci]
    if not ks:
        return
    ks_arr = np.array(ks)
    pr = preds[ks_arr]
    tg = targets[ks_arr]

    # Lấy ts_mono từ cache để plot theo thời gian thực thay vì frame index.
    with np.load(cache_paths[trial_ci], allow_pickle=False) as data:
        ts = data["ts_mono"]
        trial_id = str(data["trial_id"])
    # ts đã align với mọi frame của trial (cùng số T) — index trong dataset cũng vậy.
    fis = np.array([fi for (_ci, fi) in [index[k] for k in ks]])
    t_rel = ts[fis] - ts[fis].min()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_rel, tg, "k-", lw=1.2, label="ground truth")
    ax.plot(t_rel, pr, "r-", lw=1.0, alpha=0.8, label="predicted")
    ax.set_xlabel("time since trial start (s)")
    ax.set_ylabel("force (N)")
    m = _metrics(pr, tg)
    ax.set_title(f"{trial_id} | MAE={m['mae']:.3f} N  R²={m['r2']:+.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Eval ForceNet.")
    parser.add_argument("--config", default="src/force_model/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", default="outputs/force_model/eval",
                        help="Output dir for plots + breakdown")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_paths = discover_trials(cache_dir)
    if not cache_paths:
        raise SystemExit(f"Không có .npz trong {cache_dir}")

    split = split_trials(cache_paths, ratios=cfg["data"]["split"], seed=cfg["data"]["split_seed"])
    chosen = {"train": split.train, "val": split.val, "test": split.test}[args.split]
    if not chosen:
        raise SystemExit(f"Split '{args.split}' rỗng.")

    n_max = compute_n_max(cache_paths)
    ds = TrialDataset(chosen, n_max=n_max, augment=AugmentConfig(enabled=False))

    ckpt_path = Path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Không tìm thấy checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ForceNet(
        in_dim=4,
        hidden_per_point=tuple(cfg["model"]["hidden_per_point"]),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets, index = predict_dataset(model, ds, device,
                                            batch_size=int(cfg["train"]["batch_size"]))
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
        name = Path(r["trial_path"]).name
        print(f"{name:<60} {r['n_frames']:>5} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>+8.3f}")

    # Save raw predictions
    np.savez(
        out_dir / f"{args.split}_predictions.npz",
        preds=preds, targets=targets, **overall,
    )

    # Save breakdown CSV
    import csv
    with open(out_dir / f"{args.split}_breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_path", "n_frames", "mae", "rmse", "r2"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if not args.no_plots and len(preds) > 0:
        try:
            _plot_scatter(preds, targets, out_dir / f"{args.split}_scatter.png")
            # First trial of split for time-series demo
            _plot_timeseries(preds, targets, index, chosen,
                             out_dir / f"{args.split}_timeseries_trial0.png", trial_ci=0)
            logger.info("Plots → %s", out_dir)
        except ImportError:
            logger.warning("matplotlib chưa cài — bỏ qua plots.")


if __name__ == "__main__":
    main()
