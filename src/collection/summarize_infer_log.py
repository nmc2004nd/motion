"""Summarize realtime force inference diagnostic logs.

Usage:
    python -m src.collection.summarize_infer_log
    python -m src.collection.summarize_infer_log --log-id 20260531_120000
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_SUMMARY = Path("data/infer_delta_summary_log_v4.csv")
DEFAULT_SAMPLES = Path("data/infer_samples_log_v4.csv")
FALLBACK_SUMMARIES = [
    Path("data/infer_delta_summary_log_v3.csv"),
    Path("data/infer_delta_summary_log_v2.csv"),
    Path("data/infer_delta_summary_log.csv"),
]
FALLBACK_SAMPLES = [
    Path("data/infer_samples_log_v3.csv"),
    Path("data/infer_samples_log_v2.csv"),
    Path("data/infer_samples_log.csv"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return next(csv.reader(f), [])


def _resolve_path(primary: Path, fallbacks: list[Path]) -> Path:
    if primary.exists():
        return primary
    return next((p for p in fallbacks if p.exists()), primary)


def _warn_if_old_schema(path: Path) -> None:
    required = {
        "sample_ts_mono",
        "infer_ms",
        "frame_force_dt_s",
        "frame_latest_force_dt_s",
        "imada_raw_line",
        "imada_latest_corrected_n",
        "abs_delta_n",
        "is_spike",
        "motor_state",
    }
    header = set(_header(path))
    missing = sorted(required - header)
    if missing:
        print(
            "WARNING: sample log schema is old or mixed; FIELD STATS may be "
            f"misaligned. Missing columns: {', '.join(missing)}"
        )
        print("Run realtime_infer again to create data/infer_samples_log_v4.csv.")


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if not math.isnan(out) else math.nan


def _finite(rows: list[dict[str, str]], field: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = _to_float(row.get(field))
        if not math.isnan(value):
            vals.append(value)
    return vals


def _stats(rows: list[dict[str, str]], field: str) -> tuple[int, float, float, float, float]:
    vals = _finite(rows, field)
    if not vals:
        return 0, math.nan, math.nan, math.nan, math.nan
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return len(vals), statistics.mean(vals), std, min(vals), max(vals)


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


def _latest_log_id(
    summary_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> str | None:
    for row in reversed(summary_rows):
        log_id = row.get("log_id")
        if log_id:
            return log_id
    for row in reversed(sample_rows):
        log_id = row.get("log_id")
        if log_id:
            return log_id
    return None


def _print_summary_row(rows: list[dict[str, str]], log_id: str) -> None:
    row = next((r for r in reversed(rows) if r.get("log_id") == log_id), None)
    if row is None:
        return
    keys = [
        "timestamp",
        "command",
        "duration_s",
        "n_samples",
        "n_delta_samples",
        "spike_threshold_n",
        "n_spikes",
        "max_abs_delta_n",
        "first_spike_t_rel_s",
        "delta_mean_n",
        "delta_std_n",
        "model_raw_mean_n",
        "model_zero_mean_n",
        "model_tared_mean_n",
        "model_display_mean_n",
        "imada_raw_mean_n",
        "imada_zero_mean_n",
        "imada_corrected_mean_n",
        "imada_latest_corrected_mean_n",
        "n_valid_mean",
        "disp_mean_px",
        "disp_max_px",
        "infer_ms_mean",
        "infer_ms_max",
        "frame_force_dt_mean_s",
        "frame_latest_force_dt_mean_s",
        "force_age_mean_s",
        "force_latest_age_mean_s",
    ]
    print("SUMMARY")
    for key in keys:
        if key in row:
            print(f"  {key}: {row.get(key, '')}")


def _print_field_stats(rows: list[dict[str, str]]) -> None:
    fields = [
        ("pred_raw_n", 4),
        ("model_zero_n", 4),
        ("pred_tared_n", 4),
        ("pred_display_n", 4),
        ("imada_raw_n", 4),
        ("imada_corrected_n", 4),
        ("imada_latest_corrected_n", 4),
        ("delta_n", 4),
        ("abs_delta_n", 4),
        ("n_valid", 2),
        ("valid_ratio", 4),
        ("disp_mean_px", 4),
        ("disp_max_px", 4),
        ("dx_mean_px", 4),
        ("dy_mean_px", 4),
        ("infer_ms", 2),
        ("frame_force_dt_s", 4),
        ("frame_latest_force_dt_s", 4),
        ("force_age_s", 4),
        ("force_latest_age_s", 4),
    ]
    print("\nFIELD STATS")
    print(f"{'field':<22} {'n':>6} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    for field, digits in fields:
        n, mean, std, min_v, max_v = _stats(rows, field)
        print(
            f"{field:<22} {n:>6} "
            f"{_fmt(mean, digits):>12} {_fmt(std, digits):>12} "
            f"{_fmt(min_v, digits):>12} {_fmt(max_v, digits):>12}"
        )


def _print_imada_raw_lines(rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    for row in rows:
        raw = row.get("imada_raw_line", "")
        if raw and raw not in lines:
            lines.append(raw)
        if len(lines) >= 8:
            break
    if lines:
        print("\nIMADA RAW EXAMPLES")
        for raw in lines:
            print(f"  {raw}")


def _print_spikes(rows: list[dict[str, str]], threshold: float, context: int) -> None:
    spikes = [
        (i, row) for i, row in enumerate(rows)
        if abs(_to_float(row.get("delta_n"))) >= threshold
    ]
    max_row = max(rows, key=lambda r: abs(_to_float(r.get("delta_n"))), default=None)
    max_abs = abs(_to_float(max_row.get("delta_n"))) if max_row else math.nan

    print("\nSPIKE CHECK")
    print(f"  threshold_n: {threshold:.3f}")
    print(f"  n_spikes: {len(spikes)}")
    print(f"  max_abs_delta_n: {_fmt(max_abs, 4)}")
    if not spikes:
        print("  No sample exceeded threshold in this log.")
        return

    fields = [
        "t_rel_s",
        "motor_state",
        "pred_raw_n",
        "pred_tared_n",
        "pred_display_n",
        "imada_corrected_n",
        "imada_latest_corrected_n",
        "delta_n",
        "n_valid",
        "valid_ratio",
        "disp_mean_px",
        "disp_max_px",
        "frame_force_dt_s",
        "frame_latest_force_dt_s",
        "force_age_s",
    ]
    print("\nSPIKE ROWS")
    print(",".join(fields))
    for _, row in spikes[:12]:
        print(",".join(row.get(field, "") for field in fields))

    first_idx, _ = spikes[0]
    lo = max(0, first_idx - context)
    hi = min(len(rows), first_idx + context + 1)
    print(f"\nFIRST SPIKE CONTEXT ({lo}:{hi})")
    print(",".join(fields))
    for row in rows[lo:hi]:
        print(",".join(row.get(field, "") for field in fields))

    spike_rows = [row for _, row in spikes]
    dt = [abs(v) for v in _finite(spike_rows, "frame_force_dt_s")]
    latest_dt = [abs(v) for v in _finite(spike_rows, "frame_latest_force_dt_s")]
    valid_ratio = _finite(spike_rows, "valid_ratio")
    imada = [abs(v) for v in _finite(spike_rows, "imada_corrected_n")]
    pred = [abs(v) for v in _finite(spike_rows, "pred_display_n")]
    disp = _finite(spike_rows, "disp_mean_px")
    first_t = _to_float(spike_rows[0].get("t_rel_s")) if spike_rows else math.nan

    print("\nLIKELY CAUSE HINTS")
    if (
        not math.isnan(first_t)
        and first_t < 0.3
        and imada
        and pred
        and statistics.mean(imada) > statistics.mean(pred) + threshold * 0.5
    ):
        print(
            "  - Spike is at command start and Imada is much higher than model; "
            "likely unloading/preload transient not learned by the frame-only Poly model."
        )
    if dt and max(dt) > 0.08:
        print("  - Force/frame alignment is large during spike; inspect time sync or Imada rate.")
    elif latest_dt and max(latest_dt) > 0.08:
        print("  - Latest Imada sample is far from frame time; aligned-force logging should be used.")
    if valid_ratio and min(valid_ratio) < 0.9:
        print("  - Marker tracking drops during spike; inspect camera/lighting/LK tracking.")
    if imada and disp and statistics.mean(imada) < 0.05 and statistics.mean(disp) > 0.8:
        print("  - Markers move while Imada is near zero; likely rigid shift, backlash, or no-load motion.")
    if not (
        (
            not math.isnan(first_t)
            and first_t < 0.3
            and imada
            and pred
            and statistics.mean(imada) > statistics.mean(pred) + threshold * 0.5
        )
        or
        (dt and max(dt) > 0.08)
        or (latest_dt and max(latest_dt) > 0.08)
        or (valid_ratio and min(valid_ratio) < 0.9)
        or (imada and disp and statistics.mean(imada) < 0.05 and statistics.mean(disp) > 0.8)
    ):
        print("  - No obvious timing/tracking/no-load clue; compare spike shape with train distribution.")


def _print_tail(rows: list[dict[str, str]], n: int) -> None:
    fields = [
        "t_rel_s",
        "motor_state",
        "pred_raw_n",
        "model_zero_n",
        "pred_tared_n",
        "pred_display_n",
        "imada_corrected_n",
        "imada_latest_corrected_n",
        "delta_n",
        "abs_delta_n",
        "is_spike",
        "n_valid",
        "disp_mean_px",
        "frame_force_dt_s",
        "frame_latest_force_dt_s",
        "force_age_s",
    ]
    print(f"\nLAST {min(n, len(rows))} SAMPLES")
    print(",".join(fields))
    for row in rows[-n:]:
        print(",".join(row.get(field, "") for field in fields))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--log-id", default=None)
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--context", type=int, default=4)
    args = parser.parse_args(argv)

    summary_path = _resolve_path(Path(args.summary), FALLBACK_SUMMARIES)
    sample_path = _resolve_path(Path(args.samples), FALLBACK_SAMPLES)
    print(f"summary_log: {summary_path}")
    print(f"samples_log: {sample_path}")
    _warn_if_old_schema(sample_path)

    summary_rows = _read_csv(summary_path)
    sample_rows = _read_csv(sample_path)
    log_id = args.log_id or _latest_log_id(summary_rows, sample_rows)
    if not log_id:
        raise SystemExit("No log_id found. Run realtime_infer logging first.")

    rows = [row for row in sample_rows if row.get("log_id") == log_id]
    if not rows:
        raise SystemExit(f"No sample rows found for log_id={log_id}")

    print(f"log_id: {log_id}")
    _print_summary_row(summary_rows, log_id)
    _print_field_stats(rows)
    _print_spikes(rows, threshold=args.threshold, context=args.context)
    _print_imada_raw_lines(rows)
    _print_tail(rows, args.tail)


if __name__ == "__main__":
    main()
