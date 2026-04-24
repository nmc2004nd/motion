"""Phase 1 spike: validate RAFT-Small on tactile images.

Mục tiêu:
  1. Load RAFT-Small (torchvision pretrained), đẩy lên RTX 5060 Ti (half).
  2. Chạy trên 1 cặp ref/deformed từ data/, đo latency warmup + steady state.
  3. Visualize flow field (HSV color wheel) → ta cần nhìn xem RAFT có hiểu
     ảnh tactile không (vùng flat giữa marker sẽ noisy, nhưng ở marker
     phải có hướng hợp lý).
  4. Sample flow tại marker positions (từ blob detector), so với LK baseline.

Chạy:
    python -m src.flow_raft.spike \\
        --ref data/ref/my_photo_1.jpg \\
        --def data/img/my_photo_3.jpg \\
        --out outputs/flow_raft_spike
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from src.config import load_config
from src.core.preprocessing import preprocess
from src.core.detection import create_blob_detector, detect_markers
from src.core.tracking import track_markers_lk


def _to_raft_tensor(img_gray: np.ndarray, device: str) -> torch.Tensor:
    """gray uint8 (H,W) → (1,3,H,W) float32 tensor đã padding H,W bội số 8."""
    h, w = img_gray.shape
    h8 = ((h + 7) // 8) * 8
    w8 = ((w + 7) // 8) * 8
    padded = np.zeros((h8, w8), dtype=np.uint8)
    padded[:h, :w] = img_gray
    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().unsqueeze(0)
    t = t.to(device=device, dtype=torch.float32) / 255.0
    t = t * 2.0 - 1.0  # RAFT expects [-1, 1]
    return t, (h, w)


def _flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """(H,W,2) → BGR HSV visualization (hue=angle, value=magnitude)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv[..., 0] = (ang * 180.0 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 32.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _sample_bilinear(flow: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sample (H,W,2) flow tại pts (N,2) dùng cv2.remap bilinear."""
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    map_x = pts[:, 0].reshape(1, -1).astype(np.float32)
    map_y = pts[:, 1].reshape(1, -1).astype(np.float32)
    fx = cv2.remap(flow[..., 0], map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).reshape(-1)
    fy = cv2.remap(flow[..., 1], map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).reshape(-1)
    return np.stack([fx, fy], axis=1)


def run_spike(
    ref_path: str,
    def_path: str,
    out_dir: str,
    config_path: str,
    num_iters: int = 6,
    n_timing_runs: int = 20,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(config_path)

    # -------- 1. Load images + preprocess + detect markers --------
    img_ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    img_def_raw = cv2.imread(def_path, cv2.IMREAD_GRAYSCALE)
    if img_ref_raw is None or img_def_raw is None:
        raise FileNotFoundError(f"ref={ref_path} def={def_path}")

    # Pad/resize to same size
    if img_ref_raw.shape != img_def_raw.shape:
        img_def_raw = cv2.resize(img_def_raw, (img_ref_raw.shape[1], img_ref_raw.shape[0]))

    img_ref = preprocess(img_ref_raw, config=config)
    img_def = preprocess(img_def_raw, config=config)
    H, W = img_ref.shape

    detector = create_blob_detector(config=config)
    ref_markers, _ = detect_markers(img_ref, config=config, _detector=detector)
    print(f"[data] shape={H}x{W}  ref markers detected={len(ref_markers)}")

    # -------- 2. Load RAFT-Small --------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — reinstall torch with cu128")
    device = "cuda"
    print(f"[torch] {torch.__version__}  cuda={torch.version.cuda}  dev={torch.cuda.get_device_name(0)}")

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).to(device).eval()

    t_ref, (oh, ow) = _to_raft_tensor(img_ref, device)
    t_def, _ = _to_raft_tensor(img_def, device)
    print(f"[raft] input tensor shape={tuple(t_ref.shape)}  padded=({t_ref.shape[2]},{t_ref.shape[3]})  orig=({oh},{ow})")

    # -------- 3. Warmup + timing (FP32 + autocast FP16) --------
    def _run():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(t_ref, t_def, num_flow_updates=num_iters)

    with torch.inference_mode():
        for _ in range(3):
            _ = _run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_timing_runs):
            flows = _run()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_timing_runs
    ms = dt * 1000.0
    fps = 1.0 / dt
    print(f"[timing] RAFT-Small iters={num_iters}  mean={ms:.2f}ms  fps={fps:.1f}")

    # -------- 4. Take final flow, unpad, to np --------
    flow_t = flows[-1][0].float().cpu().numpy()  # (2, Hp, Wp)
    flow = flow_t.transpose(1, 2, 0)[:oh, :ow]   # (H, W, 2)
    print(f"[flow] stats  min={flow.min():.3f}  max={flow.max():.3f}  mean_mag={np.linalg.norm(flow, axis=-1).mean():.3f}px")

    # -------- 5. Visualize HSV flow field --------
    vis_hsv = _flow_to_hsv(flow)
    cv2.imwrite(os.path.join(out_dir, "flow_hsv.png"), vis_hsv)

    # -------- 6. Sample flow at marker positions + draw arrows --------
    if len(ref_markers) > 0:
        flow_at_markers = _sample_bilinear(flow, ref_markers)
        mags = np.linalg.norm(flow_at_markers, axis=1)
        print(f"[sample] per-marker flow  mean_mag={mags.mean():.3f}  max={mags.max():.3f}")
        for i in range(min(5, len(ref_markers))):
            print(f"  marker[{i}] ref=({ref_markers[i,0]:.1f},{ref_markers[i,1]:.1f})  flow=({flow_at_markers[i,0]:+.2f},{flow_at_markers[i,1]:+.2f})")

        # Overlay arrows on def image
        overlay = cv2.cvtColor(img_def, cv2.COLOR_GRAY2BGR)
        for (rx, ry), (fx, fy) in zip(ref_markers, flow_at_markers):
            p0 = (int(rx), int(ry))
            p1 = (int(rx + fx * 3), int(ry + fy * 3))  # arrow scale 3x for visibility
            cv2.arrowedLine(overlay, p0, p1, (0, 180, 255), 1, tipLength=0.3)
            cv2.circle(overlay, p0, 2, (80, 80, 255), -1)
        cv2.imwrite(os.path.join(out_dir, "raft_arrows_on_def.png"), overlay)
    else:
        flow_at_markers = np.zeros((0, 2), dtype=np.float32)

    # -------- 7. LK baseline for comparison --------
    tracked_lk, valid_lk = track_markers_lk(img_ref, img_def, ref_markers, config=config)
    lk_flow = tracked_lk - ref_markers
    if valid_lk.any():
        lk_mags = np.linalg.norm(lk_flow[valid_lk], axis=1)
        print(f"[LK   ] valid={int(valid_lk.sum())}/{len(ref_markers)}  mean_mag={lk_mags.mean():.3f}px")
        # Compare RAFT vs LK on mutually valid markers
        if len(flow_at_markers) == len(lk_flow):
            diff = np.linalg.norm(flow_at_markers[valid_lk] - lk_flow[valid_lk], axis=1)
            print(f"[cmp  ] |RAFT - LK| per marker  mean={diff.mean():.3f}px  max={diff.max():.3f}px  p90={np.percentile(diff, 90):.3f}px")

    # -------- 8. Side-by-side summary --------
    ref_bgr = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    def_bgr = cv2.cvtColor(img_def, cv2.COLOR_GRAY2BGR)
    row1 = np.hstack([ref_bgr, def_bgr])
    row2 = np.hstack([vis_hsv, cv2.imread(os.path.join(out_dir, "raft_arrows_on_def.png"))
                      if os.path.exists(os.path.join(out_dir, "raft_arrows_on_def.png")) else vis_hsv])
    summary = np.vstack([row1, row2])
    cv2.imwrite(os.path.join(out_dir, "summary.png"), summary)
    print(f"[done] outputs in {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="data/ref/my_photo_1.jpg")
    parser.add_argument("--def", dest="def_path", default="data/img/my_photo_2.jpg")
    parser.add_argument("--out", default="outputs/flow_raft_spike")
    parser.add_argument("--config-path", default="config/pipeline_config.yaml")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    run_spike(
        ref_path=args.ref,
        def_path=args.def_path,
        out_dir=args.out,
        config_path=args.config_path,
        num_iters=args.iters,
        n_timing_runs=args.runs,
    )


if __name__ == "__main__":
    main()
