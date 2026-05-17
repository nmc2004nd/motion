"""PySide realtime studio for tracking, slip, force, and shape tests.

Run:
    python -m src.collection.test_studio

The UI intentionally avoids Imada, Arduino, and motor stepping. The operator
chooses the mode and model(s), starts the camera, captures a reference when the
selected mode needs marker tracking, then tests directly from the live stream.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import PySide6

    _qt_plugin_path = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins" / "platforms"
    if "cv2" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    if _qt_plugin_path.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_qt_plugin_path))
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QCloseEvent, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QRadioButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: PySide6. Install it with: python -m pip install PySide6") from exc

import cv2
import numpy as np
import yaml

if _qt_plugin_path.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_plugin_path)

from src.config import load_config, require
from src.config.loader import ConfigError
from src.core.detection import create_blob_detector, detect_markers
from src.core.preprocessing import make_clahe, preprocess
from src.core.tracking import track_markers_lk
from src.core.visualization import visualize_flow_arrows
from src.force_poly.features import compute_features_v1, feature_dim
from src.slip.v1 import SlipDetector


PIPELINE_CONFIG_PATH = "config/pipeline_config.yaml"
SHAPE_CONFIG_PATH = "src/shape_model/config.yaml"

MODE_LABELS = {
    "tracking": "Tracking",
    "slip_v1": "Slip V1",
    "force": "Force",
    "shape": "Shape",
    "combined": "Force + Slip + Tracking",
    "all": "All Features",
}

MODE_HELP = {
    "tracking": "LK marker tracking using realtime_tracking.py logic.",
    "slip_v1": "LK tracking plus SlipDetector V1 MRVL probability.",
    "force": "Force model inference from selected checkpoint.",
    "shape": "Shape classifier from selected shape checkpoint.",
    "combined": "Force, slip V1, and marker tracking together.",
    "all": "Tracking, slip V1, force inference, and shape classification.",
}


@dataclass(frozen=True)
class ModelEntry:
    label: str
    ckpt_path: Path
    model_type: str
    val_metric: float
    epoch: int

    def menu_label(self) -> str:
        metric = f"{self.val_metric:.4f}" if not math.isnan(self.val_metric) else "?"
        return f"{self.label} | {self.model_type} | metric={metric} | epoch={self.epoch}"


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Torch unavailable: {exc}") from exc
    return torch


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _idx_to_label_from_ckpt(ckpt: dict) -> dict[int, str]:
    if "idx_to_label" in ckpt:
        return {int(k): str(v) for k, v in ckpt["idx_to_label"].items()}
    label_to_idx = ckpt.get("label_to_idx", {})
    return {int(v): str(k) for k, v in label_to_idx.items()}


def discover_force_models() -> tuple[list[ModelEntry], str | None]:
    try:
        torch = _torch()
    except RuntimeError as exc:
        return [], str(exc)

    entries: list[ModelEntry] = []
    for ckpt_path in sorted(Path("outputs").glob("*/checkpoints/best.pt")):
        if "shape_model" in ckpt_path.parts:
            continue
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            val_mae = float(ckpt.get("val_mae", math.nan))
            epoch = int(ckpt.get("epoch", -1))
            folder = ckpt_path.parent.parent.name
            model_cfg = ckpt.get("config", {}).get("model", {})
            if "n_max" in ckpt:
                entries.append(ModelEntry("ForceNet", ckpt_path, "forcenet", val_mae, epoch))
            elif "feature_set" in ckpt:
                entries.append(ModelEntry("PolyReg", ckpt_path, "polyregressor", val_mae, epoch))
            elif "force_cnn" in folder or "backbone" in model_cfg:
                entries.append(ModelEntry("ForceCNN", ckpt_path, "forcecnn", val_mae, epoch))
        except Exception as exc:
            print(f"Skip force checkpoint {ckpt_path}: {exc}")
    return entries, None


def discover_shape_models() -> tuple[list[ModelEntry], str | None]:
    try:
        torch = _torch()
    except RuntimeError as exc:
        return [], str(exc)

    candidates = sorted(Path("outputs/shape_model/checkpoints").glob("*.pt"))
    if not candidates and Path("outputs/shape_model/checkpoints/best.pt").exists():
        candidates = [Path("outputs/shape_model/checkpoints/best.pt")]

    entries: list[ModelEntry] = []
    for ckpt_path in candidates:
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            metric = float(ckpt.get("val_acc", ckpt.get("val_loss", math.nan)))
            epoch = int(ckpt.get("epoch", -1))
            entries.append(ModelEntry("ShapeClassifier", ckpt_path, "shape", metric, epoch))
        except Exception as exc:
            print(f"Skip shape checkpoint {ckpt_path}: {exc}")
    return entries, None


class ForceModelRunner:
    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.torch = _torch()
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model, self.n_max, self.image_size = self._build(entry)

    def _build(self, entry: ModelEntry) -> tuple[Any, int | None, tuple[int, int] | None]:
        ckpt = self.torch.load(str(entry.ckpt_path), map_location=self.device, weights_only=False)
        cfg = ckpt["config"]["model"]

        if entry.model_type == "forcenet":
            from src.force_model.model import ForceNet

            model = ForceNet(
                in_dim=4,
                hidden_per_point=tuple(cfg["hidden_per_point"]),
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg.get("dropout", 0.1)),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return model, int(ckpt.get("n_max", 150)), None

        if entry.model_type == "polyregressor":
            from src.force_poly.model import PolynomialRegressor

            model = PolynomialRegressor(
                n_features=feature_dim(str(cfg["feature_set"])),
                degree=int(cfg["degree"]),
                head=str(cfg["head"]),
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg["dropout"]),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return model, None, None

        if entry.model_type == "forcecnn":
            from src.force_cnn.model import ForceCNN

            model = ForceCNN(
                in_channels=2,
                backbone=str(cfg["backbone"]),
                pretrained=False,
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg["dropout"]),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            image_size = tuple(int(x) for x in ckpt["config"]["data"]["image_size"])
            return model, None, (image_size[0], image_size[1])

        raise ValueError(f"Unsupported force model type: {entry.model_type}")

    def predict(
        self,
        ref_gray: np.ndarray,
        gray: np.ndarray,
        ref_pts: np.ndarray | None,
        tracked: np.ndarray | None,
        valid: np.ndarray | None,
    ) -> float:
        if self.entry.model_type == "forcecnn":
            return self._predict_forcecnn(ref_gray, gray)
        if ref_pts is None or tracked is None or valid is None:
            return math.nan
        h, w = gray.shape[:2]
        if self.entry.model_type == "forcenet":
            return self._predict_forcenet(ref_pts, tracked, valid, w, h)
        if self.entry.model_type == "polyregressor":
            return self._predict_poly(ref_pts, tracked, valid, w)
        return math.nan

    def _predict_forcenet(self, ref_pts: np.ndarray, tracked: np.ndarray, valid: np.ndarray, w: int, h: int) -> float:
        assert self.n_max is not None
        n = min(len(ref_pts), self.n_max)
        feat = np.zeros((1, self.n_max, 4), dtype=np.float32)
        mask = np.zeros((1, self.n_max), dtype=bool)
        disp = tracked[:n] - ref_pts[:n]
        feat[0, :n, 0] = ref_pts[:n, 0] / w
        feat[0, :n, 1] = ref_pts[:n, 1] / h
        feat[0, :n, 2] = disp[:, 0] / w
        feat[0, :n, 3] = disp[:, 1] / h
        mask[0, :n] = valid[:n]
        with self.torch.no_grad():
            pred = self.model(
                self.torch.from_numpy(feat).to(self.device),
                self.torch.from_numpy(mask).to(self.device),
            )
        return float(pred.item())

    def _predict_poly(self, ref_pts: np.ndarray, tracked: np.ndarray, valid: np.ndarray, w: int) -> float:
        disp = (tracked - ref_pts).astype(np.float32)
        feat = compute_features_v1(ref_pts, disp, valid, w)
        with self.torch.no_grad():
            pred = self.model(self.torch.from_numpy(feat[None, :]).to(self.device))
        return float(pred.item())

    def _predict_forcecnn(self, ref_gray: np.ndarray, gray: np.ndarray) -> float:
        assert self.image_size is not None
        h, w = self.image_size
        ref = cv2.resize(ref_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        frame = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        inp = np.stack([ref, frame], axis=0)[None, ...].astype(np.float32)
        with self.torch.no_grad():
            pred = self.model(self.torch.from_numpy(inp).to(self.device))
        return float(pred.item())


class ShapeModelRunner:
    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.torch = _torch()
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.config = _load_yaml(Path(SHAPE_CONFIG_PATH))
        self.model, self.idx_to_label, self.image_size = self._build(entry)

    def _build(self, entry: ModelEntry) -> tuple[Any, dict[int, str], tuple[int, int]]:
        from src.shape_model.model import ShapeClassifier

        ckpt = self.torch.load(str(entry.ckpt_path), map_location=self.device, weights_only=False)
        idx_to_label = _idx_to_label_from_ckpt(ckpt)
        if not idx_to_label:
            raise RuntimeError(f"Checkpoint lacks label mapping: {entry.ckpt_path}")

        cfg = self.config["model"]
        model = ShapeClassifier(
            num_classes=len(idx_to_label),
            backbone=str(cfg["backbone"]),
            pretrained=False,
            hidden_head=int(cfg["hidden_head"]),
            dropout=float(cfg["dropout"]),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        image_size = tuple(int(x) for x in self.config["data"]["image_size"])
        return model, idx_to_label, (image_size[0], image_size[1])

    def predict(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        h, w = self.image_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x).float().to(self.device))
            probs = self.torch.softmax(logits, dim=1).squeeze(0).cpu()
        idx = int(self.torch.argmax(probs).item())
        return self.idx_to_label[idx], float(probs[idx])


class MetricCard(QFrame):
    def __init__(self, label: str, value: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        title = QLabel(label.upper())
        title.setObjectName("metricTitle")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.value.setStyleSheet(f"color: {accent};")
        layout.addWidget(title)
        layout.addWidget(self.value)


class MotionStudioWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.config = load_config(PIPELINE_CONFIG_PATH)
        except ConfigError as exc:
            raise SystemExit(f"[config] {exc}") from exc

        self.setWindowTitle("Motion Realtime Test Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1120, 740)

        self.cap: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_camera_tick)
        self.latest_gray: np.ndarray | None = None
        self.latest_bgr: np.ndarray | None = None
        self.reference_gray: np.ndarray | None = None
        self.reference_markers: np.ndarray | None = None
        self.marker_history: list[np.ndarray] = []
        self.last_tick = time.monotonic()
        self.fps = 0.0
        self.last_shape_ts = 0.0
        self.shape_label = "---"
        self.shape_conf = math.nan

        self.clahe = make_clahe(self.config)
        self.blob_detector = create_blob_detector(self.config)
        self.slip_detector = SlipDetector(self.config)
        self.slip_history_len = int(require(self.config, "slip_detection.history_buffer_length"))
        self.slip_threshold = float(require(self.config, "slip_detection.slip_threshold"))

        self.force_entries, self.force_error = discover_force_models()
        self.shape_entries, self.shape_error = discover_shape_models()
        self.force_by_label = {entry.menu_label(): entry for entry in self.force_entries}
        self.shape_by_label = {entry.menu_label(): entry for entry in self.shape_entries}
        self.force_runner: ForceModelRunner | None = None
        self.shape_runner: ShapeModelRunner | None = None

        self._build_ui()
        self._apply_styles()
        self._set_status("Ready. Start camera, choose mode/model, then capture reference if needed.")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(340)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(22, 24, 22, 22)
        side_layout.setSpacing(14)
        outer.addWidget(side)

        brand = QLabel("Motion Test")
        brand.setObjectName("brand")
        subtitle = QLabel("tracking / slip / force / shape")
        subtitle.setObjectName("subtitle")
        side_layout.addWidget(brand)
        side_layout.addWidget(subtitle)

        side_layout.addWidget(self._section("Test Mode"))
        self.mode_group = QButtonGroup(self)
        for idx, (key, label) in enumerate(MODE_LABELS.items()):
            radio = QRadioButton(label)
            radio.setProperty("mode_key", key)
            self.mode_group.addButton(radio, idx)
            side_layout.addWidget(radio)
            if key == "all":
                radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_change)

        side_layout.addWidget(self._section("Feature Toggles"))
        self.tracking_toggle = QCheckBox("Tracking overlay")
        self.slip_toggle = QCheckBox("Slip V1")
        self.force_toggle = QCheckBox("Force inference")
        self.shape_toggle = QCheckBox("Shape classifier")
        for toggle in (self.tracking_toggle, self.slip_toggle, self.force_toggle, self.shape_toggle):
            toggle.toggled.connect(self._on_feature_toggle)
            side_layout.addWidget(toggle)

        side_layout.addWidget(self._section("Camera"))
        camera_row = QHBoxLayout()
        self.camera_input = QLineEdit(str(require(self.config, "camera.device_id")))
        self.camera_input.setFixedWidth(74)
        self.camera_button = QPushButton("Start")
        self.camera_button.clicked.connect(self._toggle_camera)
        camera_row.addWidget(self.camera_input)
        camera_row.addWidget(self.camera_button, 1)
        side_layout.addLayout(camera_row)

        side_layout.addWidget(self._section("Force Model"))
        self.force_combo = QComboBox()
        self.force_combo.addItems(list(self.force_by_label) or ["No supported force model"])
        self.force_load = QPushButton("Load Force Model")
        self.force_load.clicked.connect(self._load_force_model)
        self.force_state = QLabel(self.force_error or "Not loaded")
        self.force_state.setWordWrap(True)
        side_layout.addWidget(self.force_combo)
        side_layout.addWidget(self.force_load)
        side_layout.addWidget(self.force_state)

        side_layout.addWidget(self._section("Shape Model"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(self.shape_by_label) or ["No supported shape model"])
        self.shape_load = QPushButton("Load Shape Model")
        self.shape_load.clicked.connect(self._load_shape_model)
        self.shape_state = QLabel(self.shape_error or "Not loaded")
        self.shape_state.setWordWrap(True)
        side_layout.addWidget(self.shape_combo)
        side_layout.addWidget(self.shape_load)
        side_layout.addWidget(self.shape_state)

        action_row = QHBoxLayout()
        self.capture_button = QPushButton("Capture Ref")
        self.capture_button.clicked.connect(self._capture_reference)
        clear_button = QPushButton("Clear Ref")
        clear_button.clicked.connect(self._clear_reference)
        action_row.addWidget(self.capture_button)
        action_row.addWidget(clear_button)
        side_layout.addLayout(action_row)

        side_layout.addStretch(1)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        side_layout.addWidget(self.status)

        main = QFrame()
        main.setObjectName("main")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(24, 22, 24, 22)
        main_layout.setSpacing(14)
        outer.addWidget(main, 1)

        header = QHBoxLayout()
        self.title = QLabel("All Features")
        self.title.setObjectName("title")
        self.reference_label = QLabel("No reference")
        self.reference_label.setObjectName("referenceLabel")
        header.addWidget(self.title, 1)
        header.addWidget(self.reference_label)
        main_layout.addLayout(header)

        self.help_label = QLabel(MODE_HELP["all"])
        self.help_label.setObjectName("help")
        main_layout.addWidget(self.help_label)

        self.video = QLabel("Camera stopped")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setMinimumHeight(420)
        main_layout.addWidget(self.video, 1)

        metrics = QGridLayout()
        metrics.setSpacing(12)
        self.force_card = MetricCard("Force", "--- N", "#38bdf8")
        self.slip_card = MetricCard("Slip V1", "---", "#f59e0b")
        self.marker_card = MetricCard("Markers", "---", "#a78bfa")
        self.shape_card = MetricCard("Shape", "---", "#2dd4bf")
        self.fps_card = MetricCard("FPS", "---", "#22c55e")
        for col, card in enumerate((self.force_card, self.slip_card, self.marker_card, self.shape_card, self.fps_card)):
            metrics.addWidget(card, 0, col)
        main_layout.addLayout(metrics)

        self._on_mode_change()

    def _section(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("section")
        return label

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #101214; color: #e5edf5; font-family: Inter, Segoe UI, Arial; }
            #sidebar { background: #15171b; border-right: 1px solid #252a32; }
            #main { background: #101214; }
            #brand { font-size: 30px; font-weight: 800; color: #f7fafc; }
            #subtitle, #help, #referenceLabel, #status { color: #8c96a3; }
            #title { font-size: 30px; font-weight: 800; color: #f7fafc; }
            #section { margin-top: 8px; color: #69717c; font-size: 11px; font-weight: 800; letter-spacing: 0; }
            QRadioButton { spacing: 10px; color: #dbe2ea; padding: 5px 0; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QLineEdit, QComboBox {
                background: #1a1d22; border: 1px solid #2b313a; border-radius: 7px;
                padding: 8px 10px; color: #f7fafc;
            }
            QPushButton {
                background: #242a32; border: 1px solid #303743; border-radius: 7px;
                padding: 9px 12px; color: #f7fafc; font-weight: 700;
            }
            QPushButton:hover { background: #303743; }
            QPushButton:disabled { color: #69717c; background: #1a1d22; }
            #video {
                background: #181b20; border: 1px solid #252a32; border-radius: 8px;
                color: #69717c; font-size: 18px;
            }
            #metricCard { background: #181b20; border: 1px solid #252a32; border-radius: 8px; }
            #metricTitle { color: #69717c; font-size: 11px; font-weight: 800; }
            #metricValue { font-size: 25px; font-weight: 800; }
            """
        )

    def _mode_key(self) -> str:
        button = self.mode_group.checkedButton()
        return str(button.property("mode_key")) if button is not None else "all"

    def _mode_needs_reference(self) -> bool:
        return self._tracking_enabled() or self._force_enabled()

    def _mode_uses_force(self) -> bool:
        return self._force_enabled()

    def _mode_uses_shape(self) -> bool:
        return self._shape_enabled()

    def _mode_uses_slip(self) -> bool:
        return self._slip_enabled()

    def _on_mode_change(self) -> None:
        key = self._mode_key()
        self.title.setText(MODE_LABELS[key])
        self.help_label.setText(MODE_HELP[key])
        self._apply_mode_preset(key)
        self._refresh_feature_controls()

    def _apply_mode_preset(self, key: str) -> None:
        presets = {
            "tracking": (True, False, False, False),
            "slip_v1": (True, True, False, False),
            "force": (True, False, True, False),
            "shape": (False, False, False, True),
            "combined": (True, True, True, False),
            "all": (True, True, True, True),
        }
        states = presets.get(key, presets["all"])
        for toggle, checked in zip(
            (self.tracking_toggle, self.slip_toggle, self.force_toggle, self.shape_toggle),
            states,
        ):
            toggle.blockSignals(True)
            toggle.setChecked(checked)
            toggle.blockSignals(False)

    def _on_feature_toggle(self) -> None:
        self._refresh_feature_controls()

    def _refresh_feature_controls(self) -> None:
        self.marker_history = []
        self.slip_detector.reset()
        force_active = self._force_enabled() and bool(self.force_by_label)
        shape_active = self._shape_enabled() and bool(self.shape_by_label)
        self.force_combo.setEnabled(force_active)
        self.force_load.setEnabled(force_active)
        self.shape_combo.setEnabled(shape_active)
        self.shape_load.setEnabled(shape_active)
        active = []
        if self._tracking_enabled():
            active.append("tracking")
        if self._slip_enabled():
            active.append("slip")
        if self._force_enabled():
            active.append("force")
        if self._shape_enabled():
            active.append("shape")
        self._set_status("Active features: " + (", ".join(active) if active else "none"))

    def _tracking_enabled(self) -> bool:
        return self.tracking_toggle.isChecked() or self.slip_toggle.isChecked() or self.force_toggle.isChecked()

    def _slip_enabled(self) -> bool:
        return self.slip_toggle.isChecked()

    def _force_enabled(self) -> bool:
        return self.force_toggle.isChecked()

    def _shape_enabled(self) -> bool:
        return self.shape_toggle.isChecked()

    def _toggle_camera(self) -> None:
        if self.cap is not None:
            self._stop_camera()
            return
        try:
            camera_id = int(self.camera_input.text().strip())
        except ValueError:
            self._set_status("Camera id must be an integer.")
            return
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self._set_status(f"Cannot open camera {camera_id}.")
            return
        for _ in range(int(require(self.config, "camera.warmup_frames"))):
            cap.read()
        self.cap = cap
        self.last_tick = time.monotonic()
        self.timer.start(15)
        self.camera_button.setText("Stop")
        self._set_status(f"Camera {camera_id} running.")

    def _stop_camera(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video.setText("Camera stopped")
        self.video.setPixmap(QPixmap())
        self.camera_button.setText("Start")
        self._set_status("Camera stopped.")

    def _load_force_model(self) -> None:
        entry = self.force_by_label.get(self.force_combo.currentText())
        if entry is None:
            return
        try:
            self.force_runner = ForceModelRunner(entry)
        except Exception as exc:
            self.force_runner = None
            self.force_state.setText(f"Load failed: {exc}")
            self._set_status("Force model load failed.")
            return
        self.force_state.setText(f"Loaded on {self.force_runner.device}: {entry.ckpt_path}")
        self._set_status("Force model loaded.")

    def _load_shape_model(self) -> None:
        entry = self.shape_by_label.get(self.shape_combo.currentText())
        if entry is None:
            return
        try:
            self.shape_runner = ShapeModelRunner(entry)
        except Exception as exc:
            self.shape_runner = None
            self.shape_state.setText(f"Load failed: {exc}")
            self._set_status("Shape model load failed.")
            return
        self.shape_state.setText(f"Loaded on {self.shape_runner.device}: {entry.ckpt_path}")
        self._set_status("Shape model loaded.")

    def _capture_reference(self) -> None:
        if self.latest_gray is None:
            self._set_status("No live frame. Start camera first.")
            return
        proc = preprocess(self.latest_gray, config=self.config, _clahe=self.clahe)
        markers, _ = detect_markers(proc, config=self.config, _detector=self.blob_detector)
        self.reference_gray = self.latest_gray.copy()
        self.reference_markers = markers
        self.marker_history = []
        self.slip_detector.reset()
        self.reference_label.setText(f"Reference: {len(markers)} markers")
        self._set_status(f"Reference captured with {len(markers)} markers.")

    def _clear_reference(self) -> None:
        self.reference_gray = None
        self.reference_markers = None
        self.marker_history = []
        self.slip_detector.reset()
        self.reference_label.setText("No reference")
        self._set_status("Reference cleared.")

    def _on_camera_tick(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._set_status("Camera frame read failed.")
            self._stop_camera()
            return

        now = time.monotonic()
        dt = max(now - self.last_tick, 1e-6)
        self.last_tick = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.latest_gray = gray.copy()
        self.latest_bgr = frame.copy()

        display, metrics = self._process_frame(gray, frame)
        self._show_frame(display)
        self.force_card.value.setText(metrics["force"])
        self.slip_card.value.setText(metrics["slip"])
        self.marker_card.value.setText(metrics["markers"])
        self.shape_card.value.setText(metrics["shape"])
        self.fps_card.value.setText(metrics["fps"])

    def _process_frame(self, gray: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
        key = self._mode_key()
        metrics = {
            "force": "--- N",
            "slip": "---",
            "markers": "---",
            "shape": self._shape_text(),
            "fps": f"{self.fps:.1f}",
        }

        ref_gray = self.reference_gray
        ref_pts = self.reference_markers
        tracked = valid = None
        display = frame.copy()

        if self._mode_needs_reference():
            if ref_gray is None or ref_pts is None or len(ref_pts) == 0:
                self._draw_text(display, "Capture reference to start marker-based tests", (18, 34), "#f59e0b")
                if self._mode_uses_shape():
                    self._maybe_predict_shape(frame, metrics)
                return display, metrics

            tracked, valid = track_markers_lk(ref_gray, gray, ref_pts, config=self.config)
            metrics["markers"] = f"{int(valid.sum())}/{len(ref_pts)}"
            display = visualize_flow_arrows(gray, ref_pts, tracked, valid, config=self.config, save_path=None)

        if self._mode_uses_slip() and tracked is not None and valid is not None and valid.any():
            self.marker_history.append(tracked.copy())
            if len(self.marker_history) > self.slip_history_len:
                self.marker_history.pop(0)
            if len(self.marker_history) >= 2:
                slip_info = self.slip_detector.calculate_slip_probability(
                    prev_markers=self.marker_history[0],
                    current_markers=tracked,
                    valid_mask=valid,
                    ref_markers=ref_pts,
                )
                r_val = float(slip_info["r_value"])
                moving = int(slip_info["moving_count"])
                state = "SLIP" if r_val > self.slip_threshold else "stable"
                metrics["slip"] = f"{r_val:.2f} {state}"
                self._draw_text(display, f"Slip V1: {r_val:.2f} | moving={moving} | {state}", (18, 72), "#f59e0b")

        if self._mode_uses_force():
            if self.force_runner is None:
                metrics["force"] = "no model"
            else:
                try:
                    pred = self.force_runner.predict(ref_gray, gray, ref_pts, tracked, valid)  # type: ignore[arg-type]
                    metrics["force"] = f"{pred:+.3f} N" if not math.isnan(pred) else "--- N"
                except Exception as exc:
                    metrics["force"] = "error"
                    self.force_state.setText(f"Inference error: {exc}")

        if self._mode_uses_shape():
            self._maybe_predict_shape(frame, metrics)

        self._draw_text(display, MODE_LABELS[key], (18, 34), "#22c55e")
        if metrics["force"] not in {"--- N", "no model"}:
            self._draw_text(display, f"Force: {metrics['force']}", (18, 110), "#38bdf8")
        if metrics["shape"] != "---":
            self._draw_text(display, f"Shape: {metrics['shape']}", (18, 148), "#2dd4bf")
        return display, metrics

    def _maybe_predict_shape(self, frame: np.ndarray, metrics: dict[str, str]) -> None:
        if self.shape_runner is None:
            metrics["shape"] = "no model"
            return
        now = time.monotonic()
        if now - self.last_shape_ts < 0.5:
            metrics["shape"] = self._shape_text()
            return
        self.last_shape_ts = now
        try:
            label, conf = self.shape_runner.predict(frame)
        except Exception as exc:
            self.shape_state.setText(f"Inference error: {exc}")
            metrics["shape"] = "error"
            return
        self.shape_label = label
        self.shape_conf = conf
        metrics["shape"] = self._shape_text()

    def _shape_text(self) -> str:
        if self.shape_label == "---" or math.isnan(self.shape_conf):
            return "---"
        return f"{self.shape_label} {self.shape_conf:.2f}"

    def _show_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pixmap)

    def _draw_text(self, img: np.ndarray, text: str, pos: tuple[int, int], color_hex: str) -> None:
        color = tuple(int(color_hex[i : i + 2], 16) for i in (5, 3, 1))
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_camera()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MotionStudioWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
