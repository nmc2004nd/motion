"""CLI: in stats một trial + optional plot force-vs-time với motor events.

Usage:
    python -m src.collection.inspect_trial <path/to/trial_NNN>
    python -m src.collection.inspect_trial <path/to/trial_NNN> --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from src.collection import dataset


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def inspect(trial_dir: Path, plot: bool = False) -> None:
    if not trial_dir.is_dir():
        print(f"ERROR: not a directory: {trial_dir}", file=sys.stderr)
        sys.exit(1)

    yaml_path = trial_dir / "trial.yaml"
    meta = dataset.read_yaml(yaml_path) if yaml_path.exists() else {}

    frames = _read_csv(trial_dir / "frames.csv")
    forces = _read_csv(trial_dir / "force_log.csv")
    motors = _read_csv(trial_dir / "motor_log.csv")

    print(f"=== {trial_dir.name} ===")
    if meta:
        print(f"  tags:                 {meta.get('tags', [])}")
        print(f"  motor_state_at_start: {meta.get('motor_state_at_start', '?')}")
        print(f"  duration_s (yaml):    {meta.get('duration_s', '?')}")
        print(f"  n_dropped (yaml):     {meta.get('n_dropped', '?')}")
    else:
        print("  (no trial.yaml — trial may be incomplete)")

    print()
    print(f"  frames.csv:    {len(frames)} rows")
    print(f"  force_log.csv: {len(forces)} rows")
    print(f"  motor_log.csv: {len(motors)} rows")

    # Inferred duration & FPS từ frames.csv
    if len(frames) >= 2:
        t0 = _safe_float(frames[0]["ts_mono"])
        t1 = _safe_float(frames[-1]["ts_mono"])
        if not (math.isnan(t0) or math.isnan(t1)) and t1 > t0:
            dur = t1 - t0
            fps = (len(frames) - 1) / dur
            print(f"  frame duration: {dur:.3f} s  ({fps:.1f} fps actual)")

    # Force stats
    if forces:
        vals = [v for v in (_safe_float(r["force_n"]) for r in forces) if not math.isnan(v)]
        if vals:
            print()
            print("  force (N):")
            print(f"    min:  {min(vals):+.4f}")
            print(f"    max:  {max(vals):+.4f}")
            print(f"    mean: {sum(vals)/len(vals):+.4f}")
            print(f"    n_valid: {len(vals)} / {len(forces)}")

    # Motor events
    if motors:
        print()
        print("  motor events:")
        for row in motors[:20]:
            ts = row["ts_mono"]
            print(f"    [{ts}] {row['direction']:<4} {row['payload']}")
        if len(motors) > 20:
            print(f"    … (+{len(motors) - 20} more)")

    # Frames count cross-check vs files trên đĩa
    frames_dir = trial_dir / "frames"
    if frames_dir.exists():
        n_files = sum(1 for _ in frames_dir.iterdir())
        if n_files != len(frames):
            print()
            print(
                f"  ⚠ frames/ has {n_files} files but frames.csv has {len(frames)} rows"
            )

    if plot:
        _plot(forces, motors, trial_dir.name)


def _plot(forces: list[dict[str, str]], motors: list[dict[str, str]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — pip install matplotlib", file=sys.stderr)
        return
    if not forces:
        print("No force data to plot.", file=sys.stderr)
        return

    ts = [_safe_float(r["ts_mono"]) for r in forces]
    fn = [_safe_float(r["force_n"]) for r in forces]
    t0 = next((t for t in ts if not math.isnan(t)), 0.0)
    rel_t = [t - t0 for t in ts]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rel_t, fn, color="#0d6efd", lw=1.2, label="force (N)")
    ax.set_xlabel("time since trial start (s)")
    ax.set_ylabel("force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    for row in motors:
        t_event = _safe_float(row["ts_mono"]) - t0
        if math.isnan(t_event):
            continue
        color = "#28a745" if row["direction"] == "cmd" else "#ffc107"
        ax.axvline(t_event, color=color, lw=0.8, alpha=0.7)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect a recorded trial.")
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--plot", action="store_true",
                        help="Plot force vs time with motor event markers")
    args = parser.parse_args(argv)
    inspect(args.trial_dir, plot=args.plot)


if __name__ == "__main__":
    main()
