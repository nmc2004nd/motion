"""Dataclasses + path helpers cho data collection theo session/trial.

Layout trên đĩa:

    data/sessions/<session_id>/
      session.yaml
      reference/ref_<ts>.jpg
      trials/trial_<NNN>/
        trial.yaml
        frames/frame_<ts>.jpg
        frames.csv      # ts_mono, ts_wall, image_name
        force_log.csv   # ts_mono, force_n
        motor_log.csv   # ts_mono, direction, payload
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data" / "sessions"


# ---------------------------------------------------------------------------- #
#  Paths                                                                        #
# ---------------------------------------------------------------------------- #


def data_root() -> Path:
    return _DATA_ROOT


def session_dir(session_id: str) -> Path:
    return _DATA_ROOT / session_id


@dataclass
class Session:
    session_id: str
    root: Path

    def trials_dir(self) -> Path:
        return self.root / "trials"

    def reference_dir(self) -> Path:
        return self.root / "reference"

    def yaml_path(self) -> Path:
        return self.root / "session.yaml"


@dataclass
class Trial:
    trial_id: str
    root: Path
    start_ts_mono: float = 0.0
    start_ts_wall: float = 0.0
    n_dropped: int = 0
    tags: list[str] = field(default_factory=list)
    motor_state_at_start: str = "unknown"

    def frames_dir(self) -> Path:
        return self.root / "frames"

    def frames_csv(self) -> Path:
        return self.root / "frames.csv"

    def force_csv(self) -> Path:
        return self.root / "force_log.csv"

    def motor_csv(self) -> Path:
        return self.root / "motor_log.csv"

    def yaml_path(self) -> Path:
        return self.root / "trial.yaml"


# ---------------------------------------------------------------------------- #
#  Session / Trial creation                                                     #
# ---------------------------------------------------------------------------- #


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def default_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"session_id phải khớp [A-Za-z0-9_-], nhận: {session_id!r}"
        )


def new_session(session_id: str, metadata: dict[str, Any]) -> Session:
    """Tạo session mới. Lỗi nếu folder đã tồn tại."""
    validate_session_id(session_id)
    root = session_dir(session_id)
    if root.exists():
        raise FileExistsError(f"Session đã tồn tại: {root}")
    root.mkdir(parents=True)
    (root / "trials").mkdir()
    (root / "reference").mkdir()

    payload = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    write_yaml(root / "session.yaml", payload)
    return Session(session_id=session_id, root=root)


def open_session(session_id: str) -> Session:
    validate_session_id(session_id)
    root = session_dir(session_id)
    if not (root / "session.yaml").exists():
        raise FileNotFoundError(f"Không tìm thấy session.yaml trong {root}")
    return Session(session_id=session_id, root=root)


def next_trial_id(session: Session) -> str:
    trials = session.trials_dir()
    trials.mkdir(exist_ok=True)
    existing = sorted(p.name for p in trials.iterdir() if p.is_dir())
    used = set()
    for name in existing:
        m = re.match(r"^trial_(\d+)$", name)
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"trial_{n:03d}"


def create_trial(session: Session) -> Trial:
    trial_id = next_trial_id(session)
    root = session.trials_dir() / trial_id
    root.mkdir(parents=True)
    (root / "frames").mkdir()
    return Trial(trial_id=trial_id, root=root)


def finalize_trial(
    trial: Trial,
    *,
    end_ts_mono: float,
    n_frames: int,
    n_force_samples: int,
    n_motor_events: int,
) -> None:
    """Ghi trial.yaml lúc stop recording."""
    payload = {
        "trial_id": trial.trial_id,
        "start_ts_wall": trial.start_ts_wall,
        "start_ts_mono": trial.start_ts_mono,
        "end_ts_mono": end_ts_mono,
        "duration_s": round(end_ts_mono - trial.start_ts_mono, 4),
        "n_frames": n_frames,
        "n_force_samples": n_force_samples,
        "n_motor_events": n_motor_events,
        "n_dropped": trial.n_dropped,
        "motor_state_at_start": trial.motor_state_at_start,
        "tags": list(trial.tags),
    }
    write_yaml(trial.yaml_path(), payload)


# ---------------------------------------------------------------------------- #
#  YAML I/O                                                                     #
# ---------------------------------------------------------------------------- #


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------- #
#  Repro helpers                                                                #
# ---------------------------------------------------------------------------- #


def git_sha(repo_root: Path = _REPO_ROOT) -> str | None:
    """Trả về git SHA full (40 hex). None nếu repo không hợp lệ."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def relevant_pip_versions() -> dict[str, str]:
    """Snapshot version các package thực sự ảnh hưởng dataset."""
    out: dict[str, str] = {}
    for name, mod in (
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pyserial", "serial"),
        ("customtkinter", "customtkinter"),
        ("pyyaml", "yaml"),
    ):
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[name] = "not_installed"
    return out


def parse_tags(raw: str) -> list[str]:
    """Comma-separated → list, strip + bỏ rỗng."""
    return [t.strip() for t in raw.split(",") if t.strip()]


# Backwards-compat: in trường hợp người dùng cũ cần data folder gốc, expose ra.
def legacy_data_root() -> Path:
    return _REPO_ROOT / "data"


def safe_session_dir_size(session: Session) -> int:
    """Kích thước folder session (bytes), dùng cho UI status."""
    if not session.root.exists():
        return 0
    total = 0
    for p in session.root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def disk_free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def repo_root() -> Path:
    return _REPO_ROOT
