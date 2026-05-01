"""PyTorch Dataset cho force regression từ displacement field cache.

Mỗi sample là 1 frame:
    feat:  (N_max, 4) float32  — [x_ref_norm, y_ref_norm, dx_norm, dy_norm], pad 0
    mask:  (N_max,)   bool      — True cho marker hợp lệ (LK valid + chưa pad)
    force: scalar     float32   — force_n (đã trừ zero_offset)

Trial-level split: split_trials() chia danh sách .npz theo seed cố định.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------- #
#  Discovery / split                                                            #
# ---------------------------------------------------------------------------- #


def discover_trials(cache_dir: Path) -> list[Path]:
    """Trả về tất cả .npz trong cache_dir/<session>/<trial>.npz, sorted."""
    return sorted(cache_dir.rglob("*.npz"))


@dataclass(frozen=True)
class SplitResult:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def split_trials(
    cache_paths: list[Path],
    ratios: dict[str, float],
    seed: int,
) -> SplitResult:
    """Chia trials theo ratio. Sum của ratios không cần đúng 1.0 — phần dư về test."""
    n = len(cache_paths)
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
        train=[cache_paths[i] for i in train_idx],
        val=[cache_paths[i] for i in val_idx],
        test=[cache_paths[i] for i in test_idx],
    )


def compute_n_max(cache_paths: list[Path]) -> int:
    """Lấy max(n_markers) qua tất cả trial — cho padding."""
    n_max = 0
    for p in cache_paths:
        with np.load(p, allow_pickle=False) as data:
            n_max = max(n_max, int(data["n_markers"]))
    return n_max


# ---------------------------------------------------------------------------- #
#  Dataset                                                                      #
# ---------------------------------------------------------------------------- #


@dataclass
class AugmentConfig:
    enabled: bool = False
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5
    rotation_max_deg: float = 5.0
    noise_std_px: float = 0.5


class TrialDataset(Dataset):
    """Load tất cả frames từ list .npz trial cache vào RAM (gọn vì marker-level)."""

    def __init__(
        self,
        cache_paths: list[Path],
        n_max: int,
        augment: AugmentConfig | None = None,
    ) -> None:
        self.cache_paths = list(cache_paths)
        self.n_max = n_max
        self.augment = augment or AugmentConfig(enabled=False)

        self._caches: list[dict] = []
        self._index: list[tuple[int, int]] = []  # (cache_idx, frame_idx)

        for ci, p in enumerate(self.cache_paths):
            with np.load(p, allow_pickle=False) as data:
                cache = {
                    "ref_pts": data["ref_pts"].copy(),         # (N, 2)
                    "disp": data["disp"].copy(),               # (T, N, 2)
                    "valid": data["valid"].copy(),             # (T, N)
                    "force": data["force"].copy(),             # (T,)
                    "image_w": int(data["image_w"]),
                    "image_h": int(data["image_h"]),
                    "trial_id": str(data["trial_id"]),
                    "session_id": str(data["session_id"]),
                }
            self._caches.append(cache)
            T = cache["disp"].shape[0]
            for fi in range(T):
                self._index.append((ci, fi))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ci, fi = self._index[idx]
        c = self._caches[ci]
        ref = c["ref_pts"]              # (N, 2)
        disp = c["disp"][fi]            # (N, 2)
        valid = c["valid"][fi].astype(bool)  # (N,)
        force = float(c["force"][fi])
        W, H = c["image_w"], c["image_h"]

        x = ref[:, 0].astype(np.float32)
        y = ref[:, 1].astype(np.float32)
        dx = disp[:, 0].astype(np.float32)
        dy = disp[:, 1].astype(np.float32)

        if self.augment.enabled:
            x, y, dx, dy = self._apply_augment(x, y, dx, dy, W, H)

        # Normalize sang [0, 1] cho position; disp normalize theo W (giữ tỉ lệ).
        x_n = x / W
        y_n = y / H
        dx_n = dx / W
        dy_n = dy / H

        feat = np.stack([x_n, y_n, dx_n, dy_n], axis=1)  # (N, 4)
        N = feat.shape[0]

        feat_pad = np.zeros((self.n_max, 4), dtype=np.float32)
        mask_pad = np.zeros((self.n_max,), dtype=bool)
        feat_pad[:N] = feat
        mask_pad[:N] = valid

        return (
            torch.from_numpy(feat_pad),
            torch.from_numpy(mask_pad),
            torch.tensor(force, dtype=torch.float32),
        )

    # ---- Augmentation ----

    def _apply_augment(
        self,
        x: np.ndarray, y: np.ndarray,
        dx: np.ndarray, dy: np.ndarray,
        W: int, H: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = self.augment
        rng = np.random

        if rng.random() < a.flip_h_prob:
            x = (W - 1) - x
            dx = -dx
        if rng.random() < a.flip_v_prob:
            y = (H - 1) - y
            dy = -dy

        if a.rotation_max_deg > 0:
            theta = math.radians(rng.uniform(-a.rotation_max_deg, a.rotation_max_deg))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            # Rotate position around image center
            xr = cos_t * (x - cx) - sin_t * (y - cy) + cx
            yr = sin_t * (x - cx) + cos_t * (y - cy) + cy
            # Rotate displacement vector (no translation, since disp is a direction)
            dxr = cos_t * dx - sin_t * dy
            dyr = sin_t * dx + cos_t * dy
            x, y, dx, dy = xr, yr, dxr, dyr

        if a.noise_std_px > 0:
            dx = dx + rng.normal(0.0, a.noise_std_px, size=dx.shape).astype(np.float32)
            dy = dy + rng.normal(0.0, a.noise_std_px, size=dy.shape).astype(np.float32)

        return x.astype(np.float32), y.astype(np.float32), dx.astype(np.float32), dy.astype(np.float32)


# ---------------------------------------------------------------------------- #
#  Helper: bundle from list of paths with shared n_max                           #
# ---------------------------------------------------------------------------- #


def build_datasets(
    split: SplitResult,
    n_max: int,
    augment_cfg: AugmentConfig,
) -> tuple[TrialDataset, TrialDataset, TrialDataset]:
    """Build (train, val, test) datasets. Augmentation chỉ áp cho train."""
    no_aug = AugmentConfig(enabled=False)
    train_ds = TrialDataset(split.train, n_max=n_max, augment=augment_cfg)
    val_ds = TrialDataset(split.val, n_max=n_max, augment=no_aug)
    test_ds = TrialDataset(split.test, n_max=n_max, augment=no_aug)
    return train_ds, val_ds, test_ds
