"""PyTorch Dataset cho polynomial force regression.

Tái sử dụng cache .npz từ src/force_model/prepare.py — không cần rerun pipeline.
Trial-level split tránh leakage. Mỗi sample là 1 frame:
    feat:  (D,) float32  — scalar features (compute_features_v1)
    force: scalar float32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import compute_features_trial


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
    n = len(cache_paths)
    if n == 0:
        return SplitResult([], [], [])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = max(1, int(round(n * ratios["train"])))
    n_val = max(1, int(round(n * ratios["val"])))
    if n_train + n_val > n:
        n_train = n - 1
        n_val = 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return SplitResult(
        train=[cache_paths[i] for i in train_idx],
        val=[cache_paths[i] for i in val_idx],
        test=[cache_paths[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------- #
#  Dataset                                                                      #
# ---------------------------------------------------------------------------- #


class PolyFeatureDataset(Dataset):
    """Pre-compute features cho mọi frame trong RAM (rất gọn — vài KB/trial)."""

    def __init__(
        self,
        cache_paths: list[Path],
        feature_set: str = "v1",
    ) -> None:
        self.cache_paths = list(cache_paths)
        self.feature_set = feature_set

        feats_list: list[np.ndarray] = []
        forces_list: list[np.ndarray] = []
        # Mỗi entry: (cache_idx, frame_idx_within_trial) — để eval per-trial.
        index: list[tuple[int, int]] = []

        for ci, p in enumerate(self.cache_paths):
            with np.load(p, allow_pickle=False) as data:
                ref_pts = data["ref_pts"].astype(np.float32)
                disp = data["disp"].astype(np.float32)
                valid = data["valid"].astype(bool)
                force = data["force"].astype(np.float32)
                W = int(data["image_w"])
            feats = compute_features_trial(
                ref_pts, disp, valid, W, feature_set=feature_set,
            )
            feats_list.append(feats)
            forces_list.append(force)
            for fi in range(feats.shape[0]):
                index.append((ci, fi))

        self.features = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 0))
        self.forces = np.concatenate(forces_list, axis=0) if forces_list else np.zeros((0,))
        self._index = index

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.forces[idx], dtype=torch.float32),
        )

    def feature_matrix(self) -> np.ndarray:
        """Toàn bộ feature matrix (N_frames, D) — dùng cho fit_scaler."""
        return self.features


def build_datasets(
    split: SplitResult,
    feature_set: str = "v1",
) -> tuple[PolyFeatureDataset, PolyFeatureDataset, PolyFeatureDataset]:
    train_ds = PolyFeatureDataset(split.train, feature_set=feature_set)
    val_ds = PolyFeatureDataset(split.val, feature_set=feature_set)
    test_ds = PolyFeatureDataset(split.test, feature_set=feature_set)
    return train_ds, val_ds, test_ds
