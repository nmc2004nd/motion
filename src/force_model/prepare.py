"""Offline preprocessing: scan trials, track LK, sync force, cache .npz.

Output 1 file `.npz` mỗi trial chứa:
    ref_pts        (N, 2) float32      — toạ độ marker tham chiếu
    disp           (T, N, 2) float32   — dịch chuyển từng frame (tracked - ref)
    valid          (T, N) bool         — LK valid mask
    force          (T,) float32        — force đã trừ zero offset (N)
    force_raw      (T,) float32        — force gốc trước khi trừ offset
    ts_mono        (T,) float64        — monotonic timestamp của frame
    image_w, image_h    int            — kích thước ảnh
    n_markers      int                 — số marker (= N)
    trial_id, session_id  str
    force_zero_offset_n   float

Usage:
    python -m src.force_model.prepare \
        --sessions data/sessions \
        --output data/cache/force_model \
        [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import load_config
from ..config.loader import ConfigError
from ..core.detection import create_blob_detector, detect_markers
from ..core.preprocessing import make_clahe, preprocess
from ..core.tracking import track_markers_lk
from ..collection import dataset as dscol

logger = logging.getLogger(__name__)


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


def _find_session_reference(session_dir: Path) -> Path | None:
    """Lấy reference image mới nhất từ session/reference/. None nếu không có."""
    ref_dir = session_dir / "reference"
    if not ref_dir.is_dir():
        return None
    refs = sorted(ref_dir.glob("ref_*.jpg"))
    return refs[-1] if refs else None


def _nearest_force(
    ts_target: float,
    force_ts: np.ndarray,
    force_vals: np.ndarray,
    tolerance_s: float,
) -> float:
    """Trả force_n tại ts_target bằng nearest-neighbor; nan nếu ngoài tolerance."""
    if force_ts.size == 0:
        return math.nan
    idx = int(np.argmin(np.abs(force_ts - ts_target)))
    if abs(force_ts[idx] - ts_target) > tolerance_s:
        return math.nan
    return float(force_vals[idx])


def _detect_reference_markers(
    ref_gray: np.ndarray, config: dict
) -> np.ndarray:
    clahe = make_clahe(config)
    detector = create_blob_detector(config)
    ref_proc = preprocess(ref_gray, config=config, _clahe=clahe)
    ref_pts, _ = detect_markers(ref_proc, config=config, _detector=detector)
    return ref_pts


def prepare_trial(
    trial_dir: Path,
    session_dir: Path,
    config: dict,
    output_path: Path,
    *,
    force_sync_tolerance_s: float = 0.1,
) -> dict:
    """Chuẩn bị 1 trial → ghi .npz vào `output_path`. Trả stats dict."""
    trial_id = trial_dir.name
    session_id = session_dir.name

    # Reference: ưu tiên session/reference/ref_*.jpg, fallback frame đầu tiên của trial.
    ref_path = _find_session_reference(session_dir)
    used_first_frame_as_ref = False
    if ref_path is None:
        # Fallback
        frames_dir = trial_dir / "frames"
        first = sorted(frames_dir.glob("frame_*.jpg"))
        if not first:
            raise FileNotFoundError(
                f"{trial_id}: không có session reference và không có frames để fallback"
            )
        ref_path = first[0]
        used_first_frame_as_ref = True

    ref_bgr = cv2.imread(str(ref_path))
    if ref_bgr is None:
        raise RuntimeError(f"Không đọc được reference: {ref_path}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    image_h, image_w = ref_gray.shape

    ref_pts = _detect_reference_markers(ref_gray, config=config)
    n_markers = ref_pts.shape[0]
    if n_markers == 0:
        raise RuntimeError(
            f"{trial_id}: detect 0 markers ở reference {ref_path.name} — "
            f"kiểm tra detection config hoặc thay reference"
        )

    # Force log
    force_rows = _read_csv(trial_dir / "force_log.csv")
    if force_rows:
        force_ts = np.array([_safe_float(r["ts_mono"]) for r in force_rows], dtype=np.float64)
        force_vals = np.array([_safe_float(r["force_n"]) for r in force_rows], dtype=np.float32)
        # Loại nan trong force_log
        keep = ~(np.isnan(force_ts) | np.isnan(force_vals))
        force_ts = force_ts[keep]
        force_vals = force_vals[keep]
    else:
        force_ts = np.zeros(0, dtype=np.float64)
        force_vals = np.zeros(0, dtype=np.float32)

    # Session metadata cho zero offset
    session_yaml_path = session_dir / "session.yaml"
    session_meta = (
        dscol.read_yaml(session_yaml_path) if session_yaml_path.exists() else {}
    )
    zero_offset = float(
        session_meta.get("force", {}).get("zero_offset_n", 0.0)
        if isinstance(session_meta.get("force"), dict)
        else 0.0
    )

    # Frames
    frame_rows = _read_csv(trial_dir / "frames.csv")
    if not frame_rows:
        raise RuntimeError(f"{trial_id}: frames.csv rỗng")

    disp_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    force_raw_list: list[float] = []
    ts_list: list[float] = []
    n_skipped = 0

    for row in frame_rows:
        ts_mono = _safe_float(row["ts_mono"])
        if math.isnan(ts_mono):
            n_skipped += 1
            continue

        f_raw = _nearest_force(ts_mono, force_ts, force_vals, force_sync_tolerance_s)
        if math.isnan(f_raw):
            n_skipped += 1
            continue

        img_path = trial_dir / "frames" / row["image_name"]
        if not img_path.exists():
            n_skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            n_skipped += 1
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        tracked, valid = track_markers_lk(
            ref_gray, gray, ref_pts, config=config, apply_deadzone=False
        )
        disp = (tracked - ref_pts).astype(np.float32)

        disp_list.append(disp)
        valid_list.append(valid)
        force_raw_list.append(f_raw)
        ts_list.append(ts_mono)

    if not disp_list:
        raise RuntimeError(
            f"{trial_id}: không sync được frame nào với force log "
            f"(skipped {n_skipped}/{len(frame_rows)})"
        )

    disp_arr = np.stack(disp_list, axis=0)              # (T, N, 2)
    valid_arr = np.stack(valid_list, axis=0)            # (T, N)
    force_raw_arr = np.array(force_raw_list, dtype=np.float32)
    force_arr = (force_raw_arr - zero_offset).astype(np.float32)
    ts_arr = np.array(ts_list, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ref_pts=ref_pts.astype(np.float32),
        disp=disp_arr,
        valid=valid_arr,
        force=force_arr,
        force_raw=force_raw_arr,
        ts_mono=ts_arr,
        image_w=np.int32(image_w),
        image_h=np.int32(image_h),
        n_markers=np.int32(n_markers),
        trial_id=trial_id,
        session_id=session_id,
        force_zero_offset_n=np.float32(zero_offset),
        used_first_frame_as_ref=np.bool_(used_first_frame_as_ref),
    )

    return {
        "trial_id": trial_id,
        "session_id": session_id,
        "n_markers": n_markers,
        "n_frames_kept": len(disp_list),
        "n_frames_total": len(frame_rows),
        "n_skipped": n_skipped,
        "force_min": float(force_arr.min()),
        "force_max": float(force_arr.max()),
        "valid_coverage": float(valid_arr.mean()),
        "output": str(output_path),
        "used_first_frame_as_ref": used_first_frame_as_ref,
    }


def prepare_all(
    sessions_root: Path,
    output_root: Path,
    config: dict,
    *,
    force_sync_tolerance_s: float = 0.1,
    skip_existing: bool = False,
) -> list[dict]:
    """Quét toàn bộ data/sessions → cache .npz cho từng trial."""
    if not sessions_root.is_dir():
        raise FileNotFoundError(f"sessions_root không tồn tại: {sessions_root}")

    stats: list[dict] = []
    for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        trials_dir = session_dir / "trials"
        if not trials_dir.is_dir():
            continue
        for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
            if not (trial_dir / "trial.yaml").exists():
                logger.warning("Bỏ qua %s (chưa finalize — không có trial.yaml)",
                               trial_dir)
                continue

            out_path = output_root / session_dir.name / f"{trial_dir.name}.npz"
            if skip_existing and out_path.exists():
                logger.info("Skip (đã có cache): %s", out_path)
                continue

            try:
                s = prepare_trial(
                    trial_dir,
                    session_dir,
                    config=config,
                    output_path=out_path,
                    force_sync_tolerance_s=force_sync_tolerance_s,
                )
                stats.append(s)
                logger.info(
                    "OK %s/%s: %d markers, %d/%d frames kept (skipped %d), "
                    "force [%.3f, %.3f], valid %.1f%%",
                    s["session_id"], s["trial_id"],
                    s["n_markers"], s["n_frames_kept"], s["n_frames_total"],
                    s["n_skipped"], s["force_min"], s["force_max"],
                    100.0 * s["valid_coverage"],
                )
            except Exception as e:
                logger.error("FAIL %s: %s", trial_dir, e)

    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Prepare force_model cache from trials.")
    parser.add_argument("--sessions", type=Path, required=True,
                        help="Root data/sessions directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output cache root, e.g. data/cache/force_model")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Pipeline config YAML (cho detection + tracking params)")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Force-sync tolerance (s)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Bỏ qua trial đã có .npz cache")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"[config] {e}", file=sys.stderr)
        sys.exit(1)

    stats = prepare_all(
        args.sessions,
        args.output,
        config=config,
        force_sync_tolerance_s=args.tolerance,
        skip_existing=args.skip_existing,
    )

    if not stats:
        print("Không xử lý được trial nào.", file=sys.stderr)
        sys.exit(1)

    # Summary
    n_trials = len(stats)
    n_frames_total = sum(s["n_frames_kept"] for s in stats)
    avg_valid = sum(s["valid_coverage"] for s in stats) / n_trials
    print()
    print(f"=== Summary: {n_trials} trials, {n_frames_total} frames cached ===")
    print(f"  avg LK valid coverage: {100.0*avg_valid:.1f}%")


if __name__ == "__main__":
    main()
