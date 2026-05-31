"""Evaluate best force_model checkpoint on one recorded session.

Usage:
    python -m src.force_model.eval_session \
        --config src/force_model/config.yaml \
        --session /home/nmc/ManhCuong/motion/data/sessions/3

This evaluates every cached trial that belongs to the given session directory.
Raw sessions must already have matching cache files from ``src.force_model.prepare``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import torch

from .dataset import AugmentConfig, TrialDataset, compute_n_max, discover_trials
from .eval import (
    _metrics,
    _plot_scatter,
    _plot_timeseries,
    load_yaml,
    per_trial_breakdown,
    predict_dataset,
)
from .model import ForceNet

logger = logging.getLogger(__name__)


def _safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return label or "session"


def _resolve_session_cache_paths(session_path: Path, cache_dir: Path) -> list[Path]:
    if not session_path.exists():
        raise SystemExit(f"Session path does not exist: {session_path}")

    if session_path.is_file():
        if session_path.suffix != ".npz":
            raise SystemExit(f"Expected a .npz cache file, got: {session_path}")
        return [session_path]

    direct_caches = sorted(session_path.glob("*.npz"))
    if direct_caches:
        return direct_caches

    trials_dir = session_path / "trials"
    if trials_dir.is_dir():
        trial_dirs = sorted(p for p in trials_dir.iterdir() if p.is_dir())
        if not trial_dirs:
            raise SystemExit(f"No trial directories found in: {trials_dir}")

        session_cache_dir = cache_dir / session_path.name
        expected = [(p.name, session_cache_dir / f"{p.name}.npz") for p in trial_dirs]
        missing = [p for _name, p in expected if not p.exists()]
        if missing:
            missing_text = "\n  ".join(str(p) for p in missing)
            raise SystemExit(
                "Missing cache for one or more trials:\n"
                f"  {missing_text}\n\n"
                "Create cache first, for example:\n"
                "  python -m src.force_model.prepare "
                f"--sessions {session_path.parent} --output {cache_dir} "
                "--config config/pipeline_config.yaml --skip-existing"
            )
        return [p for _name, p in expected]

    nested_caches = sorted(session_path.rglob("*.npz"))
    if nested_caches:
        return nested_caches

    raise SystemExit(
        f"Could not resolve cached trials from {session_path}. "
        "Pass a raw data/sessions/<id> directory, a cache directory, or a .npz file."
    )


def _build_model(cfg: dict, ckpt_path: Path, device: torch.device) -> ForceNet:
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ForceNet(
        in_dim=4,
        hidden_per_point=tuple(cfg["model"]["hidden_per_point"]),
        hidden_head=int(cfg["model"]["hidden_head"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Eval ForceNet on all cached trials from one session.",
    )
    parser.add_argument("--config", default="src/force_model/config.yaml")
    parser.add_argument("--session", required=True, help="Raw session dir, cache dir, or .npz file")
    parser.add_argument("--cache-dir", default=None, help="Override data.cache_dir from config")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path; default config checkpoint_dir/best.pt")
    parser.add_argument("--out", default="outputs/force_model/eval_session")
    parser.add_argument("--label", default=None, help="Output filename label; default session directory name")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    session_path = Path(args.session).expanduser().resolve()
    cache_dir = Path(args.cache_dir or cfg["data"]["cache_dir"])
    cache_paths = _resolve_session_cache_paths(session_path, cache_dir)
    if not cache_paths:
        raise SystemExit(f"No cached trials found for: {session_path}")

    all_cache_paths = discover_trials(cache_dir)
    n_max = compute_n_max(all_cache_paths or cache_paths)
    dataset = TrialDataset(cache_paths, n_max=n_max, augment=AugmentConfig(enabled=False))

    ckpt_path = Path(args.ckpt or (Path(cfg["train"]["checkpoint_dir"]) / "best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, ckpt_path, device)

    preds, targets, index = predict_dataset(
        model,
        dataset,
        device,
        batch_size=int(cfg["train"]["batch_size"]),
    )
    overall = _metrics(preds, targets)

    logger.info(
        "Session %s: %d trials, %d frames, MAE=%.4f, RMSE=%.4f, R2=%+.3f",
        session_path,
        len(cache_paths),
        len(preds),
        overall["mae"],
        overall["rmse"],
        overall["r2"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = _safe_label(args.label or session_path.name)
    prefix = f"session_{label}"

    rows = per_trial_breakdown(preds, targets, index, cache_paths)
    print()
    print(f"=== Per-trial ({session_path}) ===")
    print(f"{'trial':<60} {'n':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    for row in rows:
        name = Path(row["trial_path"]).name
        print(
            f"{name:<60} {row['n_frames']:>5} "
            f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>+8.3f}"
        )

    np.savez(
        out_dir / f"{prefix}_predictions.npz",
        preds=preds,
        targets=targets,
        cache_paths=np.array([str(p) for p in cache_paths]),
        session_path=str(session_path),
        **overall,
    )
    with open(out_dir / f"{prefix}_breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_path", "n_frames", "mae", "rmse", "r2"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if not args.no_plots and len(preds) > 0:
        try:
            _plot_scatter(preds, targets, out_dir / f"{prefix}_scatter.png")
            _plot_timeseries(
                preds,
                targets,
                index,
                cache_paths,
                out_dir / f"{prefix}_timeseries_trial0.png",
                trial_ci=0,
            )
            logger.info("Outputs written to %s", out_dir)
        except ImportError:
            logger.warning("matplotlib is not installed; skipping plots.")


if __name__ == "__main__":
    main()
