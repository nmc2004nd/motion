"""Evaluate best force_cnn checkpoint on one recorded session.

Usage:
    python -m src.force_cnn.eval_session \
        --config src/force_cnn/config.yaml \
        --session /home/nmc/ManhCuong/motion/data/sessions/3

This does not use train/val/test split. It indexes the raw session directory
directly and evaluates every finalized trial with synced force samples.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import torch

from .dataset import AugmentConfig, ForceImageDataset, TrialIndex, index_trial
from .eval import (
    _metrics,
    _plot_scatter,
    _plot_timeseries_first_trial,
    load_yaml,
    per_trial_breakdown,
    predict_dataset,
)
from .model import ForceCNN

logger = logging.getLogger(__name__)


def _safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return label or "session"


def _resolve_session_trials(session_path: Path, force_sync_tolerance_s: float) -> list[TrialIndex]:
    if not session_path.exists():
        raise SystemExit(f"Session path does not exist: {session_path}")
    if not session_path.is_dir():
        raise SystemExit(f"Session path must be a directory: {session_path}")

    if (session_path / "trial.yaml").exists():
        session_dir = session_path.parent.parent
        trial = index_trial(
            session_path,
            session_dir,
            force_sync_tolerance_s=force_sync_tolerance_s,
        )
        return [trial] if trial is not None else []

    trials_dir = session_path / "trials"
    if not trials_dir.is_dir():
        raise SystemExit(f"No trials directory found in: {session_path}")

    trials: list[TrialIndex] = []
    for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
        trial = index_trial(
            trial_dir,
            session_path,
            force_sync_tolerance_s=force_sync_tolerance_s,
        )
        if trial is not None:
            trials.append(trial)
    return trials


def _build_model(cfg: dict, ckpt_path: Path, device: torch.device) -> ForceCNN:
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ForceCNN(
        in_channels=2,
        backbone=cfg["model"]["backbone"],
        # The checkpoint supplies trained weights; avoid any torchvision weight
        # download during offline evaluation.
        pretrained=False,
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Eval ForceCNN on all finalized trials from one session.",
    )
    parser.add_argument("--config", default="src/force_cnn/config.yaml")
    parser.add_argument("--session", required=True, help="Raw session dir or one trial dir")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path; default config checkpoint_dir/best.pt")
    parser.add_argument("--out", default="outputs/force_cnn/eval_session")
    parser.add_argument("--label", default=None, help="Output filename label; default session directory name")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    session_path = Path(args.session).expanduser().resolve()
    trials = _resolve_session_trials(
        session_path,
        force_sync_tolerance_s=float(cfg["data"]["force_sync_tolerance_s"]),
    )
    if not trials:
        raise SystemExit(f"No finalized synced trials found for: {session_path}")

    image_size = tuple(cfg["data"]["image_size"])
    dataset = ForceImageDataset(
        trials,
        image_size=image_size,
        augment=AugmentConfig(enabled=False),
    )

    ckpt_path = Path(args.ckpt or (Path(cfg["train"]["checkpoint_dir"]) / "best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, ckpt_path, device)

    preds, targets, index = predict_dataset(
        model,
        dataset,
        device,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=0,
    )
    overall = _metrics(preds, targets)

    logger.info(
        "Session %s: %d trials, %d frames, MAE=%.4f, RMSE=%.4f, R2=%+.3f",
        session_path,
        len(trials),
        len(preds),
        overall["mae"],
        overall["rmse"],
        overall["r2"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = _safe_label(args.label or session_path.name)
    prefix = f"session_{label}"

    rows = per_trial_breakdown(preds, targets, index, trials)
    print()
    print(f"=== Per-trial ({session_path}) ===")
    print(f"{'trial':<60} {'n':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    for row in rows:
        print(
            f"{row['trial']:<60} {row['n_frames']:>5} "
            f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>+8.3f}"
        )

    np.savez(
        out_dir / f"{prefix}_predictions.npz",
        preds=preds,
        targets=targets,
        trials=np.array([f"{t.session_id}/{t.trial_id}" for t in trials]),
        session_path=str(session_path),
        **overall,
    )
    with open(out_dir / f"{prefix}_breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "n_frames", "mae", "rmse", "r2"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if not args.no_plots and len(preds) > 0:
        try:
            _plot_scatter(preds, targets, out_dir / f"{prefix}_scatter.png")
            _plot_timeseries_first_trial(
                preds,
                targets,
                index,
                trials,
                out_dir / f"{prefix}_timeseries_trial0.png",
            )
            logger.info("Outputs written to %s", out_dir)
        except ImportError:
            logger.warning("matplotlib is not installed; skipping plots.")


if __name__ == "__main__":
    main()
