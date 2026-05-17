"""Dataset utilities for object-shape classification.

The collector stores the shape label at trial level in ``trial.yaml``:

    shape: circle

Each image frame from that trial becomes one classification sample. Splitting is
done by trial, not by frame, so adjacent frames from the same capture do not leak
between train/validation/test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


CANONICAL_SHAPES = ("circle", "square", "triangle", "unknown")
SHAPE_ALIASES = {
    "c": "circle",
    "circle": "circle",
    "round": "circle",
    "tron": "circle",
    "tròn": "circle",
    "s": "square",
    "square": "square",
    "vuong": "square",
    "vuông": "square",
    "t": "triangle",
    "triangle": "triangle",
    "tam giac": "triangle",
    "tam giác": "triangle",
    "unknown": "unknown",
    "unknow": "unknown",
    "other": "unknown",
    "none": "unknown",
    "1": "unknown",
}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_shape_label(value: object, *, default: str = "unknown") -> str:
    """Normalize labels from metadata/config to the 4 canonical classes."""
    raw = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    raw = " ".join(raw.split())
    if not raw:
        return default
    return SHAPE_ALIASES.get(raw, default)


@dataclass(frozen=True)
class TrialIndex:
    session_id: str
    trial_id: str
    shape: str
    frames: list[Path]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class SplitResult:
    train: list[TrialIndex]
    val: list[TrialIndex]
    test: list[TrialIndex]


@dataclass
class AugmentConfig:
    enabled: bool = False
    flip_h_prob: float = 0.5
    rotation_max_deg: float = 8.0
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12


def index_trial(
    trial_dir: Path,
    *,
    max_frames_per_trial: int | None = None,
    fallback_shape_by_session: dict[str, str] | None = None,
) -> TrialIndex | None:
    """Read one finalized trial and return frame paths plus its shape label."""
    meta = _load_yaml(trial_dir / "trial.yaml")
    session_id = trial_dir.parent.parent.name if trial_dir.parent.name == "trials" else ""
    fallback = (fallback_shape_by_session or {}).get(session_id, "unknown")
    shape = normalize_shape_label(meta.get("shape"), default=fallback)

    frames_dir = trial_dir / "frames"
    if not frames_dir.is_dir():
        return None

    frame_rows = _read_csv(trial_dir / "frames.csv")
    if frame_rows:
        frames = [
            frames_dir / row["image_name"]
            for row in frame_rows
            if row.get("image_name") and (frames_dir / row["image_name"]).exists()
        ]
    else:
        frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return None

    max_frames = max_frames_per_trial or 0
    if max_frames > 0 and len(frames) > max_frames:
        # Deterministic uniform subsampling preserves trial coverage without
        # making a long capture dominate the class distribution.
        idx = np.linspace(0, len(frames) - 1, num=max_frames)
        frames = [frames[int(round(i))] for i in idx]

    trial_id = str(meta.get("trial_id") or trial_dir.name)
    return TrialIndex(
        session_id=session_id,
        trial_id=trial_id,
        shape=shape,
        frames=frames,
    )


def _session_dirs_from_root(data_root: Path) -> list[Path]:
    if (data_root / "trial.yaml").exists():
        return []
    if (data_root / "trials").is_dir():
        return [data_root]
    if data_root.is_dir():
        return sorted(p for p in data_root.iterdir() if (p / "trials").is_dir())
    return []


def discover_trials(
    data_root: Path,
    *,
    max_frames_per_trial: int | None = None,
    fallback_shape_by_session: dict[str, str] | None = None,
) -> list[TrialIndex]:
    """Discover shape-labeled trials.

    ``data_root`` may be:
    - a single trial directory containing ``trial.yaml``;
    - a session directory such as ``data/sessions/c1``;
    - a sessions root such as ``data/sessions``.
    """
    data_root = data_root.expanduser()
    if (data_root / "trial.yaml").exists():
        trial = index_trial(
            data_root,
            max_frames_per_trial=max_frames_per_trial,
            fallback_shape_by_session=fallback_shape_by_session,
        )
        return [trial] if trial is not None else []

    out: list[TrialIndex] = []
    for session_dir in _session_dirs_from_root(data_root):
        trials_dir = session_dir / "trials"
        for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
            trial = index_trial(
                trial_dir,
                max_frames_per_trial=max_frames_per_trial,
                fallback_shape_by_session=fallback_shape_by_session,
            )
            if trial is not None:
                out.append(trial)
    return out


def discover_trials_from_roots(
    data_roots: list[Path],
    *,
    max_frames_per_trial: int | None = None,
    fallback_shape_by_session: dict[str, str] | None = None,
) -> list[TrialIndex]:
    out: list[TrialIndex] = []
    seen: set[Path] = set()
    for root in data_roots:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.extend(
            discover_trials(
                root,
                max_frames_per_trial=max_frames_per_trial,
                fallback_shape_by_session=fallback_shape_by_session,
            )
        )
    return out


def build_label_map(
    trials: list[TrialIndex],
    classes: list[str] | None = None,
) -> dict[str, int]:
    if classes:
        labels = [normalize_shape_label(label) for label in classes]
        labels = list(dict.fromkeys(labels))
    else:
        labels = sorted({t.shape for t in trials})
    return {label: i for i, label in enumerate(labels)}


def split_trials(
    trials: list[TrialIndex],
    ratios: dict[str, float],
    seed: int,
) -> SplitResult:
    """Trial-level split with lightweight class grouping when possible."""
    if not trials:
        return SplitResult([], [], [])

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[TrialIndex]] = {}
    for trial in trials:
        by_label.setdefault(trial.shape, []).append(trial)

    train: list[TrialIndex] = []
    val: list[TrialIndex] = []
    test: list[TrialIndex] = []

    for label_trials in by_label.values():
        label_trials = list(label_trials)
        order = rng.permutation(len(label_trials))
        shuffled = [label_trials[i] for i in order]
        n = len(shuffled)
        if n == 1:
            train.extend(shuffled)
            continue

        n_train = int(round(n * float(ratios["train"])))
        n_val = int(round(n * float(ratios["val"])))
        n_train = min(max(n_train, 1), n - 1)
        remaining = n - n_train
        n_val = min(max(n_val, 1 if remaining >= 2 else 0), remaining)

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    train = [train[i] for i in rng.permutation(len(train))] if train else []
    val = [val[i] for i in rng.permutation(len(val))] if val else []
    test = [test[i] for i in rng.permutation(len(test))] if test else []
    return SplitResult(train=train, val=val, test=test)


def load_image_normalized(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load BGR image, convert to RGB, resize, normalize to [0, 1]."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Không đọc được ảnh: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = image_size
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def _apply_augment(img: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    h, w = img.shape[:2]
    if np.random.random() < cfg.flip_h_prob:
        img = img[:, ::-1, :].copy()

    if cfg.rotation_max_deg > 0:
        angle = float(np.random.uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg))
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if cfg.brightness_jitter > 0 or cfg.contrast_jitter > 0:
        b = float(np.random.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
        c = 1.0 + float(np.random.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
        img = np.clip((img - 0.5) * c + 0.5 + b, 0.0, 1.0)

    return img.astype(np.float32)


class ShapeImageDataset(Dataset):
    def __init__(
        self,
        trials: list[TrialIndex],
        label_to_idx: dict[str, int],
        image_size: tuple[int, int],
        augment: AugmentConfig | None = None,
    ) -> None:
        self.trials = list(trials)
        self.label_to_idx = dict(label_to_idx)
        self.image_size = image_size
        self.augment = augment or AugmentConfig(enabled=False)
        self._index: list[tuple[int, int]] = []
        for trial_idx, trial in enumerate(self.trials):
            for frame_idx in range(trial.n_frames):
                self._index.append((trial_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trial_idx, frame_idx = self._index[idx]
        trial = self.trials[trial_idx]
        img = load_image_normalized(trial.frames[frame_idx], self.image_size)
        if self.augment.enabled:
            img = _apply_augment(img, self.augment)
        # HWC RGB -> CHW RGB
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32))
        y = torch.tensor(_safe_int(self.label_to_idx.get(trial.shape)), dtype=torch.long)
        return x, y


def build_datasets(
    split: SplitResult,
    label_to_idx: dict[str, int],
    image_size: tuple[int, int],
    augment_cfg: AugmentConfig,
) -> tuple[ShapeImageDataset, ShapeImageDataset, ShapeImageDataset]:
    no_aug = AugmentConfig(enabled=False)
    return (
        ShapeImageDataset(split.train, label_to_idx, image_size, augment_cfg),
        ShapeImageDataset(split.val, label_to_idx, image_size, no_aug),
        ShapeImageDataset(split.test, label_to_idx, image_size, no_aug),
    )


def trial_counts_by_shape(trials: list[TrialIndex]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial.shape] = counts.get(trial.shape, 0) + 1
    return dict(sorted(counts.items()))


def frame_counts_by_shape(trials: list[TrialIndex]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial.shape] = counts.get(trial.shape, 0) + trial.n_frames
    return dict(sorted(counts.items()))
