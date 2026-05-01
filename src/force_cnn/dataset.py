"""Dataset cho end-to-end CNN force regression.

Mỗi sample là 1 frame:
    inp:    (2, H, W) float32  — [ref_gray, deformed_gray] đã normalize [0, 1]
    force:  scalar    float32  — force_n (đã trừ zero_offset)

Trial-level split tránh leakage giữa các frame liền kề. Lazy-load ảnh on-the-fly
(không cache pixel ra .npz) để tránh phình disk.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------- #
#  IO helpers                                                                   #
# ---------------------------------------------------------------------------- #


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


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_session_reference(session_dir: Path) -> Path | None:
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
    if force_ts.size == 0:
        return math.nan
    idx = int(np.argmin(np.abs(force_ts - ts_target)))
    if abs(force_ts[idx] - ts_target) > tolerance_s:
        return math.nan
    return float(force_vals[idx])


# ---------------------------------------------------------------------------- #
#  Trial discovery + indexing                                                   #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialIndex:
    session_id: str
    trial_id: str
    ref_path: Path
    samples: list[tuple[Path, float]]  # (frame_path, force_n)

    @property
    def n_frames(self) -> int:
        return len(self.samples)


def index_trial(
    trial_dir: Path,
    session_dir: Path,
    *,
    force_sync_tolerance_s: float,
) -> TrialIndex | None:
    """Index một trial → danh sách (frame_path, force_n) đã sync. None nếu skip."""
    if not (trial_dir / "trial.yaml").exists():
        return None  # chưa finalize

    ref_path = _find_session_reference(session_dir)
    if ref_path is None:
        first = sorted((trial_dir / "frames").glob("frame_*.jpg"))
        if not first:
            return None
        ref_path = first[0]

    force_rows = _read_csv(trial_dir / "force_log.csv")
    if not force_rows:
        return None
    force_ts = np.array([_safe_float(r["ts_mono"]) for r in force_rows], dtype=np.float64)
    force_vals = np.array([_safe_float(r["force_n"]) for r in force_rows], dtype=np.float32)
    keep = ~(np.isnan(force_ts) | np.isnan(force_vals))
    force_ts = force_ts[keep]
    force_vals = force_vals[keep]

    session_meta = _load_yaml(session_dir / "session.yaml")
    zero_offset = 0.0
    f_meta = session_meta.get("force") if isinstance(session_meta, dict) else None
    if isinstance(f_meta, dict):
        zero_offset = float(f_meta.get("zero_offset_n", 0.0))

    frame_rows = _read_csv(trial_dir / "frames.csv")
    if not frame_rows:
        return None

    samples: list[tuple[Path, float]] = []
    for row in frame_rows:
        ts_mono = _safe_float(row["ts_mono"])
        if math.isnan(ts_mono):
            continue
        f_raw = _nearest_force(ts_mono, force_ts, force_vals, force_sync_tolerance_s)
        if math.isnan(f_raw):
            continue
        img_path = trial_dir / "frames" / row["image_name"]
        if not img_path.exists():
            continue
        samples.append((img_path, float(f_raw - zero_offset)))

    if not samples:
        return None

    return TrialIndex(
        session_id=session_dir.name,
        trial_id=trial_dir.name,
        ref_path=ref_path,
        samples=samples,
    )


def discover_trials(
    sessions_root: Path,
    *,
    force_sync_tolerance_s: float = 0.1,
) -> list[TrialIndex]:
    """Quét data/sessions → list TrialIndex (chỉ những trial finalize + sync được force)."""
    if not sessions_root.is_dir():
        return []
    out: list[TrialIndex] = []
    for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        trials_dir = session_dir / "trials"
        if not trials_dir.is_dir():
            continue
        for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
            ti = index_trial(
                trial_dir, session_dir,
                force_sync_tolerance_s=force_sync_tolerance_s,
            )
            if ti is not None:
                out.append(ti)
    return out


# ---------------------------------------------------------------------------- #
#  Trial-level split                                                            #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitResult:
    train: list[TrialIndex]
    val: list[TrialIndex]
    test: list[TrialIndex]


def split_trials(
    trials: list[TrialIndex],
    ratios: dict[str, float],
    seed: int,
) -> SplitResult:
    n = len(trials)
    if n == 0:
        return SplitResult([], [], [])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    n_train = max(n_train, 1) if n >= 3 else n_train
    n_val = max(n_val, 1) if n - n_train >= 2 else n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return SplitResult(
        train=[trials[i] for i in train_idx],
        val=[trials[i] for i in val_idx],
        test=[trials[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------- #
#  Augmentation                                                                 #
# ---------------------------------------------------------------------------- #


@dataclass
class AugmentConfig:
    enabled: bool = False
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5
    rotation_max_deg: float = 5.0
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1


def _apply_paired_augment(
    ref: np.ndarray,
    frame: np.ndarray,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment (ref, frame) đồng bộ về geometry và photometric.

    Brightness/contrast được áp giống hệt nhau cho cả ref và frame để mô phỏng
    exposure drift của camera, không phá quan hệ vật lý giữa hai ảnh.
    """
    H, W = ref.shape
    if np.random.random() < cfg.flip_h_prob:
        ref = ref[:, ::-1].copy()
        frame = frame[:, ::-1].copy()
    if np.random.random() < cfg.flip_v_prob:
        ref = ref[::-1, :].copy()
        frame = frame[::-1, :].copy()
    if cfg.rotation_max_deg > 0:
        angle = float(np.random.uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg))
        M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), angle, 1.0)
        ref = cv2.warpAffine(ref, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if cfg.brightness_jitter > 0 or cfg.contrast_jitter > 0:
        b = float(np.random.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
        c = 1.0 + float(np.random.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
        ref = np.clip((ref - 0.5) * c + 0.5 + b, 0.0, 1.0)
        frame = np.clip((frame - 0.5) * c + 0.5 + b, 0.0, 1.0)
    return ref.astype(np.float32), frame.astype(np.float32)


# ---------------------------------------------------------------------------- #
#  Image loading                                                                #
# ---------------------------------------------------------------------------- #


def load_gray_normalized(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Đọc ảnh, convert grayscale, resize → (H, W) float32 trong [0, 1]."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Không đọc được ảnh: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = image_size
    if gray.shape != (H, W):
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------- #
#  Dataset                                                                      #
# ---------------------------------------------------------------------------- #


class ForceImageDataset(Dataset):
    """Lazy-load (ref, frame) pair từ list TrialIndex.

    Reference mỗi trial được cache trong RAM sau lần đọc đầu (chỉ vài chục MB
    cho hàng trăm trial). Frame thì luôn đọc từ disk — DataLoader num_workers
    sẽ pipeline IO.
    """

    def __init__(
        self,
        trials: list[TrialIndex],
        image_size: tuple[int, int],
        augment: AugmentConfig | None = None,
    ) -> None:
        self.trials = list(trials)
        self.image_size = image_size
        self.augment = augment or AugmentConfig(enabled=False)
        self._index: list[tuple[int, int]] = []
        for ti, t in enumerate(self.trials):
            for fi in range(t.n_frames):
                self._index.append((ti, fi))
        self._ref_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _get_ref(self, trial_idx: int) -> np.ndarray:
        ref = self._ref_cache.get(trial_idx)
        if ref is None:
            ref = load_gray_normalized(
                self.trials[trial_idx].ref_path, self.image_size
            )
            self._ref_cache[trial_idx] = ref
        return ref

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ti, fi = self._index[idx]
        trial = self.trials[ti]
        frame_path, force_n = trial.samples[fi]
        ref = self._get_ref(ti).copy()
        frame = load_gray_normalized(frame_path, self.image_size)

        if self.augment.enabled:
            ref, frame = _apply_paired_augment(ref, frame, self.augment)

        inp = np.stack([ref, frame], axis=0)  # (2, H, W)
        return (
            torch.from_numpy(inp.astype(np.float32)),
            torch.tensor(force_n, dtype=torch.float32),
        )


def build_datasets(
    split: SplitResult,
    image_size: tuple[int, int],
    augment_cfg: AugmentConfig,
) -> tuple[ForceImageDataset, ForceImageDataset, ForceImageDataset]:
    """Build (train, val, test) datasets. Augmentation chỉ áp cho train."""
    no_aug = AugmentConfig(enabled=False)
    train_ds = ForceImageDataset(split.train, image_size, augment_cfg)
    val_ds = ForceImageDataset(split.val, image_size, no_aug)
    test_ds = ForceImageDataset(split.test, image_size, no_aug)
    return train_ds, val_ds, test_ds
