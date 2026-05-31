"""GUI inference station: camera + Imada force gauge + model.

So sánh lực đo thực tế (Imada) với lực ước tính từ model theo thời gian thực.
Hỗ trợ chọn model từ tất cả checkpoint tìm thấy trong outputs/*/checkpoints/best.pt.

Chạy:
    python -u src/collection/realtime_infer.py
"""

from __future__ import annotations

import csv
import math
import queue
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
import serial
import torch
from PIL import Image

from src.config import load_config
from src.core.detection import create_blob_detector, detect_markers
from src.core.preprocessing import make_clahe, preprocess
from src.core.tracking import track_markers_lk
from src.core.visualization import visualize_flow_arrows
from src.force_cnn.model import ForceCNN
from src.force_model.model import ForceNet
from src.force_poly.features import compute_features_v1, feature_dim
from src.force_poly.model import PolynomialRegressor

# ---------------------------------------------------------------------------- #
#  Constants                                                                    #
# ---------------------------------------------------------------------------- #

PIPELINE_CONFIG_PATH = "config/pipeline_config.yaml"

ARDUINO_PORT  = "/dev/ttyACM0"
IMADA_PORT    = "/dev/ttyACM1"
ARDUINO_BAUD  = 115200
IMADA_BAUD    = 19200
CAMERA_ID     = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 1080

MODEL_TARE_SAMPLES = 20
MODEL_EMA_ALPHA    = 0.35
DIAG_LOG_SECONDS   = 10
FORCE_ALIGN_TOLERANCE_S = 0.12
DELTA_SPIKE_N = 0.10

LEGACY_DELTA_LOG_PATH = Path("data/infer_delta_log.csv")
SUMMARY_LOG_PATH      = Path("data/infer_delta_summary_log_v4.csv")
SAMPLES_LOG_PATH      = Path("data/infer_samples_log_v4.csv")

_IMADA_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_imada(raw: str) -> float:
    raw = raw.strip()
    if not raw:
        return math.nan
    if raw[0] in ("r", "R"):
        raw = raw[1:]
    m = _IMADA_NUM_RE.match(raw)
    return float(m.group(0)) if m else math.nan


def _finite(values) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            out.append(f)
    return out


def _mean_or_nan(values) -> float:
    vals = _finite(values)
    return statistics.mean(vals) if vals else math.nan


def _stdev_or_zero(values) -> float:
    vals = _finite(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _round_or_nan(value: float, ndigits: int = 4) -> float:
    return math.nan if math.isnan(value) else round(float(value), ndigits)


def _disp_debug_stats(
    ref_pts: np.ndarray | None,
    tracked: np.ndarray | None,
    valid: np.ndarray | None,
) -> dict[str, float]:
    stats = {
        "valid_ratio": math.nan,
        "disp_mean_px": math.nan,
        "disp_max_px": math.nan,
        "dx_mean_px": math.nan,
        "dy_mean_px": math.nan,
    }
    if ref_pts is None or tracked is None or valid is None or len(ref_pts) == 0:
        return stats
    stats["valid_ratio"] = float(valid.sum()) / float(len(ref_pts))
    if not valid.any():
        return stats

    disp = (tracked - ref_pts).astype(np.float32)
    d = disp[valid]
    mag = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
    stats.update({
        "disp_mean_px": float(mag.mean()),
        "disp_max_px": float(mag.max()),
        "dx_mean_px": float(d[:, 0].mean()),
        "dy_mean_px": float(d[:, 1].mean()),
    })
    return stats


# ---------------------------------------------------------------------------- #
#  Model registry                                                               #
# ---------------------------------------------------------------------------- #


@dataclass
class ModelEntry:
    label: str          # tên ngắn để hiển thị
    ckpt_path: Path
    model_type: str     # "forcenet" | "polyregressor" | "forcecnn" | "unknown"
    val_mae: float
    epoch: int

    def menu_label(self) -> str:
        mae_str = f"{self.val_mae:.4f} N" if not math.isnan(self.val_mae) else "?"
        return f"{self.label}  (MAE={mae_str},  epoch={self.epoch})"


def _classify_checkpoint(ckpt_path: Path) -> ModelEntry:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    val_mae = float(ckpt.get("val_mae", math.nan))
    epoch   = int(ckpt.get("epoch", -1))
    folder = ckpt_path.parent.parent.name
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    if "n_max" in ckpt:
        return ModelEntry("ForceNet", ckpt_path, "forcenet", val_mae, epoch)
    if "feature_set" in ckpt:
        return ModelEntry("PolyReg", ckpt_path, "polyregressor", val_mae, epoch)
    if (
        "val_mae" in ckpt
        and "model_state" in ckpt
        and "backbone" in model_cfg
        and "image_size" in data_cfg
        and ("force_cnn" in folder or "force_cnn" in str(ckpt_path))
    ):
        return ModelEntry("ForceCNN", ckpt_path, "forcecnn", val_mae, epoch)
    return ModelEntry(folder, ckpt_path, "unknown", val_mae, epoch)


def discover_models() -> list[ModelEntry]:
    """Quét outputs/*/checkpoints/best.pt và phân loại từng checkpoint."""
    outputs = Path("outputs")
    if not outputs.exists():
        return []
    entries: list[ModelEntry] = []
    for ckpt_path in sorted(outputs.glob("*/checkpoints/best.pt")):
        try:
            entries.append(_classify_checkpoint(ckpt_path))
        except Exception as e:
            print(f"Bỏ qua {ckpt_path}: {e}")
    return entries


# ---------------------------------------------------------------------------- #
#  Model runner (uniform inference API)                                         #
# ---------------------------------------------------------------------------- #


class ModelRunner:
    """Wrapper inference cho ForceNet, PolynomialRegressor và ForceCNN."""

    def __init__(self, entry: ModelEntry, device: torch.device) -> None:
        self.entry = entry
        self._device = device
        self._model, self._n_max, self._image_size = self._build(entry, device)

    def _build(self, entry: ModelEntry, device: torch.device):
        ckpt = torch.load(str(entry.ckpt_path), map_location=device, weights_only=False)
        cfg  = ckpt["config"]["model"]

        if entry.model_type == "forcenet":
            model = ForceNet(
                in_dim=4,
                hidden_per_point=tuple(cfg["hidden_per_point"]),
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg.get("dropout", 0.1)),
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return model, int(ckpt.get("n_max", 150)), None

        if entry.model_type == "polyregressor":
            model = PolynomialRegressor(
                n_features=feature_dim(str(cfg["feature_set"])),
                degree=int(cfg["degree"]),
                head=str(cfg["head"]),
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg["dropout"]),
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return model, None, None

        if entry.model_type == "forcecnn":
            model = ForceCNN(
                in_channels=2,
                backbone=str(cfg["backbone"]),
                pretrained=False,
                hidden_head=int(cfg["hidden_head"]),
                dropout=float(cfg["dropout"]),
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            image_size = tuple(int(x) for x in ckpt["config"]["data"]["image_size"])
            return model, None, (image_size[0], image_size[1])

        raise ValueError(f"Không hỗ trợ model_type={entry.model_type!r}")

    def predict(
        self,
        ref_pts: np.ndarray | None,
        tracked: np.ndarray | None,
        valid: np.ndarray | None,
        W: int,
        H: int,
        ref_gray: np.ndarray | None = None,
        gray: np.ndarray | None = None,
    ) -> float:
        if self.entry.model_type == "forcecnn":
            if ref_gray is None or gray is None:
                return math.nan
            return self._predict_forcecnn(ref_gray, gray)
        if ref_pts is None or tracked is None or valid is None:
            return math.nan
        if self.entry.model_type == "forcenet":
            return self._predict_forcenet(ref_pts, tracked, valid, W, H)
        if self.entry.model_type == "polyregressor":
            return self._predict_poly(ref_pts, tracked, valid, W)
        return math.nan

    def _predict_forcenet(self, ref_pts, tracked, valid, W, H) -> float:
        n = min(len(ref_pts), self._n_max)
        feat = np.zeros((1, self._n_max, 4), dtype=np.float32)
        mask = np.zeros((1, self._n_max), dtype=bool)
        disp = tracked[:n] - ref_pts[:n]
        feat[0, :n, 0] = ref_pts[:n, 0] / W
        feat[0, :n, 1] = ref_pts[:n, 1] / H
        feat[0, :n, 2] = disp[:, 0] / W
        feat[0, :n, 3] = disp[:, 1] / H
        mask[0, :n] = valid[:n]
        with torch.no_grad():
            return float(
                self._model(
                    torch.from_numpy(feat).to(self._device),
                    torch.from_numpy(mask).to(self._device),
                ).item()
            )

    def _predict_poly(self, ref_pts, tracked, valid, W) -> float:
        disp = (tracked - ref_pts).astype(np.float32)
        feat = compute_features_v1(ref_pts, disp, valid, W)
        with torch.no_grad():
            return float(
                self._model(torch.from_numpy(feat[None, :]).to(self._device)).item()
            )

    def _predict_forcecnn(self, ref_gray: np.ndarray, gray: np.ndarray) -> float:
        if self._image_size is None:
            return math.nan
        h, w = self._image_size
        ref = cv2.resize(ref_gray, (w, h), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        inp = np.stack(
            [ref.astype(np.float32) / 255.0, frame.astype(np.float32) / 255.0],
            axis=0,
        )[None, ...].astype(np.float32)
        with torch.no_grad():
            return float(self._model(torch.from_numpy(inp).to(self._device)).item())


# ---------------------------------------------------------------------------- #
#  ForceInferStation                                                            #
# ---------------------------------------------------------------------------- #


class ForceInferStation:
    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._ref_lock = threading.Lock()
        self._runner_lock = threading.Lock()

        self._latest_vis: np.ndarray | None = None
        self._latest_force_n: float = math.nan
        self._latest_force_ts_mono: float = math.nan
        self._latest_force_raw_line: str = ""
        self._force_history: deque[tuple[float, float, str]] = deque(maxlen=500)
        self._latest_pred_raw_force: float = math.nan
        self._latest_pred_tared_force: float = math.nan
        self._latest_pred_force: float = math.nan
        self._latest_frame_ts_mono: float = math.nan
        self._latest_infer_ms: float = math.nan
        self._latest_n_valid: int = 0
        self._force_zero_n: float = math.nan
        self._model_zero_n: float = math.nan
        self._model_tare_pending = False
        self._model_tare_samples: list[float] = []
        self._model_ema_n: float = math.nan
        self._latest_debug: dict[str, float] = _disp_debug_stats(None, None, None)
        self._latest_model_label = ""
        self._latest_model_type = ""

        self._ref_image: np.ndarray | None = None
        self._ref_markers: np.ndarray | None = None

        self._runner: ModelRunner | None = None

        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._motor_state = "unknown"
        self._toast_widget = None
        self._toast_after_id = None

        # Delta logging state (only accessed from main tkinter thread)
        self._logging_active  = False
        self._delta_samples:  list[dict] = []
        self._log_start_time  = 0.0
        self._log_id          = ""
        self._log_command     = ""
        self._log_distance    = 0.0
        self._diag_after_id   = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline_config = load_config(PIPELINE_CONFIG_PATH)
        self._clahe        = make_clahe(self._pipeline_config)
        self._blob_detector = create_blob_detector(self._pipeline_config)

        self._model_entries = discover_models()

        self._init_hardware()
        self._init_gui()

        # Load model đầu tiên ngay sau khi GUI sẵn sàng
        if self._model_entries:
            self._load_runner(self._model_entries[0])

        self._start_threads()

    # ------------------------------------------------------------------ #
    #  Hardware                                                            #
    # ------------------------------------------------------------------ #

    def _init_hardware(self) -> None:
        try:
            self.arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.05)
            self.arduino_connected = True
        except Exception as e:
            print(f"Arduino: {e}")
            self.arduino_ser = None
            self.arduino_connected = False

        try:
            self.imada_ser = serial.Serial(
                port=IMADA_PORT, baudrate=IMADA_BAUD,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=0.1,
            )
            self.imada_connected = True
        except Exception as e:
            print(f"Imada: {e}")
            self.imada_ser = None
            self.imada_connected = False

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  GUI                                                                 #
    # ------------------------------------------------------------------ #

    def _init_gui(self) -> None:
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Force Inference Station")
        self.app.geometry("1150x800")

        # === Top bar =====================================================
        top = ctk.CTkFrame(self.app)
        top.pack(side="top", fill="x", padx=10, pady=(10, 4))

        self.btn_capture_ref = ctk.CTkButton(
            top, text="📷  Capture Reference", width=165,
            command=self._on_capture_reference,
            fg_color="#6610f2", hover_color="#520dc2",
        )
        self.btn_capture_ref.pack(side="left", padx=(10, 6))

        self.btn_clear_ref = ctk.CTkButton(
            top, text="✕  Clear Reference", width=140,
            command=self._on_clear_reference,
            fg_color="#495057", hover_color="#343a40",
            state="disabled",
        )
        self.btn_clear_ref.pack(side="left", padx=(0, 14))

        # Model selector
        ctk.CTkLabel(
            top, text="Model:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 6))

        menu_labels = [e.menu_label() for e in self._model_entries] or ["(không tìm thấy checkpoint)"]
        self._model_var = ctk.StringVar(value=menu_labels[0])
        self._model_menu = ctk.CTkOptionMenu(
            top,
            variable=self._model_var,
            values=menu_labels,
            width=380,
            command=self._on_model_selected,
            fg_color="#1a1a2e",
            button_color="#0d6efd",
            button_hover_color="#0b5ed7",
        )
        self._model_menu.pack(side="left", padx=4)

        # Model info (val MAE, type)
        self.lbl_model_info = ctk.CTkLabel(
            top, text="",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#adb5bd",
        )
        self.lbl_model_info.pack(side="left", padx=10)

        # === Main body ===================================================
        body = ctk.CTkFrame(self.app, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=4)

        # --- Left: camera + force ---
        left = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ctk.CTkLabel(
            left, text="CAMERA FEED",
            font=ctk.CTkFont(weight="bold", size=15),
        ).pack(pady=(8, 4))

        self.video_label = ctk.CTkLabel(left, text="Loading camera…")
        self.video_label.pack(pady=4, expand=True)

        # Two force boxes side by side
        force_row = ctk.CTkFrame(left, fg_color="transparent")
        force_row.pack(fill="x", padx=10, pady=(6, 4))
        force_row.columnconfigure(0, weight=1)
        force_row.columnconfigure(1, weight=1)

        imada_box = ctk.CTkFrame(force_row, corner_radius=10, fg_color=("#dee2e6", "#1e2329"))
        imada_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(
            imada_box, text="IMADA  (đo thực tế)",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#6c757d",
        ).pack(pady=(8, 0))
        self.lbl_imada = ctk.CTkLabel(
            imada_box, text="--- N",
            font=ctk.CTkFont(size=32, weight="bold"), text_color="#17a2b8",
        )
        self.lbl_imada.pack(pady=(2, 10))

        model_box = ctk.CTkFrame(force_row, corner_radius=10, fg_color=("#dee2e6", "#1e2329"))
        model_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.lbl_model_type = ctk.CTkLabel(
            model_box, text="MODEL  (ước tính)",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#6c757d",
        )
        self.lbl_model_type.pack(pady=(8, 0))
        self.lbl_model_force = ctk.CTkLabel(
            model_box, text="--- N",
            font=ctk.CTkFont(size=32, weight="bold"), text_color="#fd7e14",
        )
        self.lbl_model_force.pack(pady=(2, 10))

        # Delta + marker count
        meta_row = ctk.CTkFrame(left, fg_color="transparent")
        meta_row.pack(fill="x", padx=10, pady=(0, 8))
        self.lbl_delta = ctk.CTkLabel(
            meta_row, text="Δ = ---",
            font=ctk.CTkFont(size=13), text_color="#adb5bd",
        )
        self.lbl_delta.pack(side="left", padx=4)
        self.lbl_n_markers = ctk.CTkLabel(
            meta_row, text="",
            font=ctk.CTkFont(size=12), text_color="#6c757d",
        )
        self.lbl_n_markers.pack(side="right", padx=4)

        # --- Right: motor control ---
        right = ctk.CTkFrame(body, width=280)
        right.pack(side="right", fill="y", padx=(6, 0))

        ctk.CTkLabel(
            right, text="MOTOR CONTROL",
            font=ctk.CTkFont(weight="bold", size=15),
        ).pack(pady=10)

        self.distance_var = ctk.StringVar(value="1")
        inp = ctk.CTkFrame(right, fg_color="transparent")
        inp.pack(pady=8)
        ctk.CTkLabel(inp, text="Distance (mm):").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(inp, textvariable=self.distance_var, width=80).grid(row=0, column=1)

        ctk.CTkButton(
            right, text="FORWARD 🔼", command=self._forward,
            fg_color="#28a745", hover_color="#218838",
        ).pack(pady=8, padx=20, fill="x")
        ctk.CTkButton(
            right, text="BACKWARD 🔽", command=self._backward,
            fg_color="#007bff", hover_color="#0069d9",
        ).pack(pady=8, padx=20, fill="x")
        ctk.CTkButton(
            right, text="STOP 🛑", command=self._stop,
            fg_color="#dc3545", hover_color="#c82333", height=40,
        ).pack(pady=16, padx=20, fill="x")

        self.lbl_motor_state = ctk.CTkLabel(
            right, text="Motor: unknown",
            font=ctk.CTkFont(size=13), text_color="#adb5bd",
        )
        self.lbl_motor_state.pack(pady=6)

        # Delta log panel
        ctk.CTkFrame(right, height=1, fg_color="#343a40").pack(
            fill="x", padx=12, pady=(10, 6),
        )
        ctk.CTkLabel(
            right, text="DELTA LOG",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#6c757d",
        ).pack()
        self.lbl_log_status = ctk.CTkLabel(
            right, text="Chưa có log",
            font=ctk.CTkFont(size=11), text_color="#6c757d",
            wraplength=240, justify="left",
        )
        self.lbl_log_status.pack(padx=12, pady=(2, 8), anchor="w")
        ctk.CTkButton(
            right, text="DIAG 10s", command=self._diag_log_10s,
            fg_color="#6f42c1", hover_color="#5a32a3",
        ).pack(pady=(0, 8), padx=20, fill="x")

        conn_text = "Connected" if self.arduino_connected else "Disconnected"
        ctk.CTkLabel(
            right, text=f"Arduino: {conn_text}",
            font=ctk.CTkFont(slant="italic"), text_color="gray",
        ).pack(side="bottom", pady=16)

        # === Status bar ==================================================
        self.status_var = ctk.StringVar(value="Nhấn 'Capture Reference' để bắt đầu.")
        ctk.CTkLabel(
            self.app, textvariable=self.status_var,
            font=ctk.CTkFont(size=12), text_color="#adb5bd", anchor="w",
        ).pack(side="bottom", fill="x", padx=14, pady=(0, 8))

    # ------------------------------------------------------------------ #
    #  Model loading / switching                                           #
    # ------------------------------------------------------------------ #

    def _load_runner(self, entry: ModelEntry) -> None:
        try:
            runner = ModelRunner(entry, self._device)
            with self._runner_lock:
                self._runner = runner
            with self._lock:
                self._latest_model_label = entry.label
                self._latest_model_type = entry.model_type
            self.lbl_model_info.configure(
                text=f"{entry.model_type}  |  {entry.ckpt_path}",
                text_color="#28a745",
            )
            self.lbl_model_type.configure(text=f"MODEL  ({entry.label})")
        except Exception as e:
            with self._runner_lock:
                self._runner = None
            with self._lock:
                self._latest_pred_raw_force = math.nan
                self._latest_pred_tared_force = math.nan
                self._latest_pred_force = math.nan
                self._latest_frame_ts_mono = math.nan
                self._latest_infer_ms = math.nan
                self._model_zero_n = math.nan
                self._model_tare_pending = False
                self._model_tare_samples = []
                self._model_ema_n = math.nan
                self._latest_model_label = ""
                self._latest_model_type = ""
            self.lbl_model_info.configure(
                text=f"Load thất bại: {e}", text_color="#dc3545",
            )

    def _begin_model_tare(self) -> None:
        with self._lock:
            self._latest_pred_raw_force = math.nan
            self._latest_pred_tared_force = math.nan
            self._latest_pred_force = math.nan
            self._latest_frame_ts_mono = math.nan
            self._latest_infer_ms = math.nan
            self._model_zero_n = math.nan
            self._model_tare_pending = True
            self._model_tare_samples = []
            self._model_ema_n = math.nan

    def _on_model_selected(self, choice: str) -> None:
        entry = next((e for e in self._model_entries if e.menu_label() == choice), None)
        if entry is None:
            return
        self._set_status(f"Đang load {entry.label}…")
        self._load_runner(entry)
        with self._ref_lock:
            has_ref = self._ref_image is not None
        if has_ref:
            self._begin_model_tare()
        self._show_toast(f"Đã chuyển sang {entry.label}", color="#0d6efd")
        tare_msg = " | đang lấy model zero" if has_ref else ""
        self._set_status(
            f"Model active: {entry.label}  |  MAE={entry.val_mae:.4f} N  "
            f"|  epoch={entry.epoch}{tare_msg}"
        )

    # ------------------------------------------------------------------ #
    #  Reference lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def _on_capture_reference(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            self._set_status("Không đọc được frame từ camera.", error=True)
            return

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_proc = preprocess(gray, config=self._pipeline_config, _clahe=self._clahe)
        markers, _ = detect_markers(
            ref_proc, config=self._pipeline_config, _detector=self._blob_detector,
        )
        zero = self._sample_force_zero(n=20)

        with self._ref_lock:
            self._ref_image   = gray.copy()
            self._ref_markers = markers
        with self._lock:
            self._force_zero_n = zero
        self._begin_model_tare()

        self.btn_clear_ref.configure(state="normal")
        n = len(markers)
        self._show_toast(f"Reference đã capture: {n} markers", color="#6610f2")
        self._set_status(
            f"Reference: {n} markers  |  force zero={zero:.4f} N  "
            f"|  đang lấy model zero ({MODEL_TARE_SAMPLES} samples)."
        )

    def _on_clear_reference(self) -> None:
        with self._ref_lock:
            self._ref_image   = None
            self._ref_markers = None
        with self._lock:
            self._latest_pred_raw_force = math.nan
            self._latest_pred_tared_force = math.nan
            self._latest_pred_force = math.nan
            self._latest_frame_ts_mono = math.nan
            self._latest_infer_ms = math.nan
            self._model_zero_n = math.nan
            self._model_tare_pending = False
            self._model_tare_samples = []
            self._model_ema_n = math.nan
            self._latest_debug = _disp_debug_stats(None, None, None)
        self.btn_clear_ref.configure(state="disabled")
        self._show_toast("Reference đã xoá", color="#495057")
        self._set_status("Reference đã xoá — capture lại để tiếp tục.")

    def _sample_force_zero(self, n: int = 20) -> float:
        if not self.imada_connected:
            return math.nan
        samples: list[float] = []
        for _ in range(n):
            with self._lock:
                v = self._latest_force_n
            if not math.isnan(v):
                samples.append(v)
            time.sleep(0.05)
        return statistics.median(samples) if samples else math.nan

    def _apply_model_tare_locked(self, pred_raw: float) -> tuple[float, float]:
        """Trừ baseline model tại reference và lọc EMA nhẹ. Gọi khi đã giữ _lock."""
        if math.isnan(pred_raw):
            return math.nan, math.nan

        if self._model_tare_pending:
            self._model_tare_samples.append(pred_raw)
            if len(self._model_tare_samples) < MODEL_TARE_SAMPLES:
                return math.nan, math.nan
            self._model_zero_n = statistics.median(self._model_tare_samples)
            self._model_tare_pending = False
            self._model_tare_samples = []
            self._model_ema_n = math.nan

        zero = 0.0 if math.isnan(self._model_zero_n) else self._model_zero_n
        pred_tared = pred_raw - zero
        if math.isnan(self._model_ema_n):
            self._model_ema_n = pred_tared
        else:
            self._model_ema_n = (
                MODEL_EMA_ALPHA * pred_tared
                + (1.0 - MODEL_EMA_ALPHA) * self._model_ema_n
            )
        return pred_tared, self._model_ema_n

    def _nearest_force_locked(
        self,
        ts_mono: float,
        tolerance_s: float = FORCE_ALIGN_TOLERANCE_S,
    ) -> tuple[float, float, str, float]:
        """Trả force sample gần frame timestamp nhất. Gọi khi đã giữ _lock."""
        if math.isnan(ts_mono) or not self._force_history:
            return math.nan, math.nan, "", math.nan

        best_ts, best_force, best_raw = min(
            self._force_history, key=lambda item: abs(item[0] - ts_mono),
        )
        dt = ts_mono - best_ts
        if abs(dt) > tolerance_s:
            return math.nan, math.nan, "", math.nan
        return best_force, best_ts, best_raw, dt

    # ------------------------------------------------------------------ #
    #  Motor commands                                                      #
    # ------------------------------------------------------------------ #

    def _model_tare_ready(self) -> bool:
        with self._lock:
            pending = self._model_tare_pending
            count = len(self._model_tare_samples)
        if pending:
            self._set_status(
                f"Đợi model tare xong ({count}/{MODEL_TARE_SAMPLES}) rồi hãy chạy motor.",
                error=True,
            )
            return False
        return True

    def _forward(self) -> None:
        try:
            d = float(self.distance_var.get())
        except ValueError:
            return
        if not self._model_tare_ready():
            return
        self._cmd_queue.put(f"f {d}")
        self._start_log("forward", d)

    def _backward(self) -> None:
        try:
            d = float(self.distance_var.get())
        except ValueError:
            return
        if not self._model_tare_ready():
            return
        self._cmd_queue.put(f"b {d}")
        self._start_log("backward", d)

    def _stop(self) -> None:
        self._cmd_queue.put("s")
        if self._logging_active:
            self._finalize_log()

    # ------------------------------------------------------------------ #
    #  Worker threads                                                      #
    # ------------------------------------------------------------------ #

    def _worker_camera(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_ts_mono = time.monotonic()
            infer_start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis  = frame.copy()
            pred_raw = math.nan
            n_valid = 0
            tracked = None
            valid = None
            debug_stats = _disp_debug_stats(None, None, None)
            model_label = ""
            model_type = ""

            with self._ref_lock:
                ref_img = self._ref_image
                ref_pts = self._ref_markers

            if ref_img is not None:
                if ref_pts is not None and len(ref_pts) > 0:
                    tracked, valid = track_markers_lk(
                        ref_img, gray, ref_pts,
                        self._pipeline_config, apply_deadzone=False,
                    )
                    vis = visualize_flow_arrows(
                        gray, ref_pts, tracked, valid,
                        config=self._pipeline_config, save_path=None,
                    )
                    n_valid = int(valid.sum())
                    debug_stats = _disp_debug_stats(ref_pts, tracked, valid)
                with self._runner_lock:
                    runner = self._runner
                if runner is not None:
                    model_label = runner.entry.label
                    model_type = runner.entry.model_type
                    H, W = gray.shape
                    try:
                        if runner.entry.model_type == "forcecnn":
                            pred_raw = runner.predict(
                                ref_pts, tracked, valid, W, H,
                                ref_gray=ref_img, gray=gray,
                            )
                        elif (
                            ref_pts is not None
                            and tracked is not None
                            and valid is not None
                            and n_valid > 0
                        ):
                            pred_raw = runner.predict(ref_pts, tracked, valid, W, H)
                    except Exception as e:
                        print(f"Inference error: {e}")

            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            with self._lock:
                pred_tared, pred_display = self._apply_model_tare_locked(pred_raw)
                self._latest_vis        = vis
                self._latest_pred_raw_force = pred_raw
                self._latest_pred_tared_force = pred_tared
                self._latest_pred_force = pred_display
                self._latest_frame_ts_mono = frame_ts_mono
                self._latest_infer_ms = infer_ms
                self._latest_n_valid    = n_valid
                self._latest_debug      = debug_stats
                self._latest_model_label = model_label
                self._latest_model_type = model_type

    def _worker_imada(self) -> None:
        while True:
            if self.imada_connected and self.imada_ser.is_open:
                try:
                    self.imada_ser.write(b"D\r")
                    raw = self.imada_ser.read_until(b"\r").decode(
                        "utf-8", errors="ignore",
                    ).strip()
                    if raw:
                        force_ts_mono = time.monotonic()
                        force_n = _parse_imada(raw)
                        with self._lock:
                            self._latest_force_n = force_n
                            self._latest_force_ts_mono = force_ts_mono
                            self._latest_force_raw_line = raw
                            if not math.isnan(force_n):
                                self._force_history.append((force_ts_mono, force_n, raw))
                except Exception as e:
                    print(f"Imada read error: {e}")
            time.sleep(0.05)

    def _worker_arduino(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
                if self.arduino_connected and self.arduino_ser.is_open:
                    self.arduino_ser.write((cmd + "\n").encode())
                    if cmd.startswith(("f ", "b ")):
                        self._set_motor_state("moving")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Arduino write error: {e}")

            if self.arduino_connected and self.arduino_ser.is_open:
                try:
                    if self.arduino_ser.in_waiting:
                        line = self.arduino_ser.readline().decode(
                            "utf-8", errors="ignore",
                        ).strip()
                        if line:
                            self._update_motor_state_from_resp(line)
                except Exception as e:
                    print(f"Arduino read error: {e}")
            time.sleep(0.005)

    def _update_motor_state_from_resp(self, resp: str) -> None:
        if resp.startswith("M: Stopped"):
            self._set_motor_state("idle")
        elif resp.startswith(("M: Forward", "M: Backward")):
            self._set_motor_state("moving")

    def _set_motor_state(self, state: str) -> None:
        self._motor_state = state
        color = {"idle": "#28a745", "moving": "#ffc107", "unknown": "#adb5bd"}.get(
            state, "#adb5bd",
        )
        self.app.after(0, lambda s=state, c=color: self._apply_motor_state(s, c))

    def _apply_motor_state(self, state: str, color: str) -> None:
        """Chạy trên main thread — cập nhật UI và finalize log nếu motor vừa dừng."""
        self.lbl_motor_state.configure(text=f"Motor: {state}", text_color=color)
        if state == "idle" and self._logging_active:
            self._finalize_log()

    # ------------------------------------------------------------------ #
    #  Delta logging                                                       #
    # ------------------------------------------------------------------ #

    def _start_log(self, direction: str, distance: float) -> None:
        """Bắt đầu thu thập delta samples cho lệnh điều khiển mới."""
        if self._diag_after_id is not None:
            self.app.after_cancel(self._diag_after_id)
            self._diag_after_id = None
        if self._logging_active:
            self._finalize_log()          # đóng log cũ nếu còn dang dở
        self._logging_active  = True
        self._delta_samples   = []
        self._log_start_time  = time.time()
        self._log_id          = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(self._log_start_time),
        )
        self._log_command     = (
            direction if math.isnan(distance) else f"{direction} {distance:.0f}mm"
        )
        self._log_distance    = distance
        self.lbl_log_status.configure(
            text=f"● {self._log_command}  (0 samples)", text_color="#ffc107",
        )
        self._set_status(f"Logging delta: {self._log_command}…")

    def _diag_log_10s(self) -> None:
        if self._logging_active:
            self._finalize_log()
            return
        self._start_log(f"diag_rest_{DIAG_LOG_SECONDS}s", math.nan)

        def _timeout():
            self._diag_after_id = None
            self._finalize_log()

        self._diag_after_id = self.app.after(DIAG_LOG_SECONDS * 1000, _timeout)

    def _append_csv_rows(
        self,
        path: Path,
        rows: list[dict],
        fieldnames: list[str],
    ) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        if path.exists():
            with open(path, newline="") as f:
                existing = next(csv.reader(f), [])
            if existing != fieldnames:
                raise RuntimeError(
                    f"CSV schema mismatch for {path}: expected {fieldnames}, "
                    f"found {existing}. Use a new *_vN.csv log path."
                )
            write_header = False
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _finalize_log(self) -> None:
        """Tính trung bình delta và ghi CSV. Chạy trên main thread."""
        if not self._logging_active:
            return
        if self._diag_after_id is not None:
            self.app.after_cancel(self._diag_after_id)
            self._diag_after_id = None
        self._logging_active = False

        samples  = self._delta_samples
        self._delta_samples = []
        duration = time.time() - self._log_start_time

        if not samples:
            self.lbl_log_status.configure(
                text=f"{self._log_command}: no samples", text_color="#6c757d",
            )
            return

        deltas = _finite(s["delta_n"] for s in samples)
        abs_deltas = [abs(v) for v in deltas]
        spike_samples = [
            s for s in samples
            if not math.isnan(float(s["abs_delta_n"]))
            and float(s["abs_delta_n"]) >= DELTA_SPIKE_N
        ]
        delta_mean = _mean_or_nan(deltas)
        delta_std  = _stdev_or_zero(deltas)
        max_abs_delta = max(abs_deltas, default=math.nan)
        first_spike_t = (
            float(spike_samples[0]["t_rel_s"]) if spike_samples else math.nan
        )

        row = {
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(self._log_start_time)),
            "log_id":       self._log_id,
            "command":      self._log_command,
            "distance_mm":  self._log_distance,
            "duration_s":   round(duration, 3),
            "n_samples":    len(samples),
            "n_delta_samples": len(deltas),
            "spike_threshold_n": DELTA_SPIKE_N,
            "n_spikes": len(spike_samples),
            "max_abs_delta_n": _round_or_nan(max_abs_delta),
            "first_spike_t_rel_s": _round_or_nan(first_spike_t, ndigits=3),
            "delta_mean_n": _round_or_nan(delta_mean),
            "delta_std_n":  _round_or_nan(delta_std),
            "model_raw_mean_n": _round_or_nan(
                _mean_or_nan(s["pred_raw_n"] for s in samples),
            ),
            "model_zero_mean_n": _round_or_nan(
                _mean_or_nan(s["model_zero_n"] for s in samples),
            ),
            "model_tared_mean_n": _round_or_nan(
                _mean_or_nan(s["pred_tared_n"] for s in samples),
            ),
            "model_display_mean_n": _round_or_nan(
                _mean_or_nan(s["pred_display_n"] for s in samples),
            ),
            "imada_raw_mean_n": _round_or_nan(
                _mean_or_nan(s["imada_raw_n"] for s in samples),
            ),
            "imada_zero_mean_n": _round_or_nan(
                _mean_or_nan(s["imada_zero_n"] for s in samples),
            ),
            "imada_corrected_mean_n": _round_or_nan(
                _mean_or_nan(s["imada_corrected_n"] for s in samples),
            ),
            "imada_latest_corrected_mean_n": _round_or_nan(
                _mean_or_nan(s["imada_latest_corrected_n"] for s in samples),
            ),
            "n_valid_mean": _round_or_nan(
                _mean_or_nan(s["n_valid"] for s in samples), ndigits=2,
            ),
            "disp_mean_px": _round_or_nan(
                _mean_or_nan(s["disp_mean_px"] for s in samples),
            ),
            "disp_max_px": _round_or_nan(
                _mean_or_nan(s["disp_max_px"] for s in samples),
            ),
            "infer_ms_mean": _round_or_nan(
                _mean_or_nan(s["infer_ms"] for s in samples), ndigits=2,
            ),
            "infer_ms_max": _round_or_nan(
                max(_finite(s["infer_ms"] for s in samples), default=math.nan),
                ndigits=2,
            ),
            "frame_force_dt_mean_s": _round_or_nan(
                _mean_or_nan(s["frame_force_dt_s"] for s in samples),
            ),
            "frame_latest_force_dt_mean_s": _round_or_nan(
                _mean_or_nan(s["frame_latest_force_dt_s"] for s in samples),
            ),
            "force_age_mean_s": _round_or_nan(
                _mean_or_nan(s["force_age_s"] for s in samples),
            ),
            "force_latest_age_mean_s": _round_or_nan(
                _mean_or_nan(s["force_latest_age_s"] for s in samples),
            ),
        }

        legacy_row = {
            "timestamp": row["timestamp"],
            "command": row["command"],
            "distance_mm": row["distance_mm"],
            "duration_s": row["duration_s"],
            "n_samples": row["n_samples"],
            "delta_mean_n": row["delta_mean_n"],
            "delta_std_n": row["delta_std_n"],
            "model_mean_n": row["model_display_mean_n"],
            "imada_mean_n": row["imada_corrected_mean_n"],
        }
        self._append_csv_rows(
            LEGACY_DELTA_LOG_PATH, [legacy_row], list(legacy_row.keys()),
        )
        self._append_csv_rows(
            SUMMARY_LOG_PATH, [row], list(row.keys()),
        )

        sample_fields = [
            "timestamp",
            "log_id",
            "t_rel_s",
            "sample_ts_mono",
            "frame_ts_mono",
            "force_ts_mono",
            "force_latest_ts_mono",
            "frame_age_s",
            "force_age_s",
            "force_latest_age_s",
            "frame_force_dt_s",
            "frame_latest_force_dt_s",
            "infer_ms",
            "command",
            "distance_mm",
            "motor_state",
            "model_label",
            "model_type",
            "pred_raw_n",
            "model_zero_n",
            "pred_tared_n",
            "pred_display_n",
            "imada_raw_n",
            "imada_raw_line",
            "imada_zero_n",
            "imada_corrected_n",
            "imada_latest_raw_n",
            "imada_latest_raw_line",
            "imada_latest_corrected_n",
            "delta_n",
            "abs_delta_n",
            "is_spike",
            "spike_threshold_n",
            "n_ref_markers",
            "n_valid",
            "valid_ratio",
            "disp_mean_px",
            "disp_max_px",
            "dx_mean_px",
            "dy_mean_px",
        ]
        self._append_csv_rows(
            SAMPLES_LOG_PATH, samples, sample_fields,
        )

        delta_text = "nan" if math.isnan(delta_mean) else f"{delta_mean:+.4f} N"
        summary = (
            f"{self._log_command}\n"
            f"Δ_mean={delta_text}  ±{delta_std:.4f}\n"
            f"{len(deltas)}/{len(samples)} valid samples  |  {duration:.1f}s"
        )
        self.lbl_log_status.configure(text=summary, text_color="#adb5bd")
        self._show_toast(f"Log saved: {self._log_command}", color="#198754")
        self._set_status(
            f"Log saved: {self._log_command}  Δ_mean={delta_text}  "
            f"±{delta_std:.4f}  ({len(deltas)}/{len(samples)} valid samples,  "
            f"{duration:.1f}s)"
        )

    # ------------------------------------------------------------------ #
    #  UI loop                                                             #
    # ------------------------------------------------------------------ #

    def _update_ui(self) -> None:
        with self._lock:
            vis        = self._latest_vis
            force_latest_raw = self._latest_force_n
            force_latest_ts_mono = self._latest_force_ts_mono
            force_latest_raw_line = self._latest_force_raw_line
            pred_raw   = self._latest_pred_raw_force
            pred_tared = self._latest_pred_tared_force
            pred       = self._latest_pred_force
            frame_ts_mono = self._latest_frame_ts_mono
            infer_ms = self._latest_infer_ms
            zero       = self._force_zero_n
            n_valid    = self._latest_n_valid
            model_zero = self._model_zero_n
            tare_pending = self._model_tare_pending
            tare_count = len(self._model_tare_samples)
            debug_stats = dict(self._latest_debug)
            model_label = self._latest_model_label
            model_type = self._latest_model_type
            (
                force_aligned_raw,
                force_aligned_ts_mono,
                force_aligned_raw_line,
                frame_force_dt_s,
            ) = self._nearest_force_locked(frame_ts_mono)

        with self._ref_lock:
            has_ref = self._ref_image is not None
            n_ref_markers = (
                0 if self._ref_markers is None else int(len(self._ref_markers))
            )

        # Camera
        if vis is not None:
            rgb  = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            rgb  = cv2.resize(rgb, (480, 360))
            pil  = Image.fromarray(rgb)
            cimg = ctk.CTkImage(light_image=pil, dark_image=pil, size=(480, 360))
            self.video_label.configure(image=cimg, text="")

        # Imada latest để hiển thị; delta/log dùng sample đã align với frame.
        force_latest_corrected = (
            force_latest_raw - zero
            if not (math.isnan(force_latest_raw) or math.isnan(zero))
            else force_latest_raw
        )
        force_corrected = (
            force_aligned_raw - zero
            if not (math.isnan(force_aligned_raw) or math.isnan(zero))
            else force_aligned_raw
        )
        self.lbl_imada.configure(
            text=(
                f"{force_latest_corrected:+.3f} N"
                if not math.isnan(force_latest_corrected)
                else "--- N"
            )
        )

        # Model
        model_text = (
            f"tare {tare_count}/{MODEL_TARE_SAMPLES}"
            if tare_pending
            else f"{pred:+.3f} N" if not math.isnan(pred)
            else "--- N"
        )
        self.lbl_model_force.configure(
            text=model_text
        )

        # Delta
        delta = math.nan
        if not (math.isnan(pred) or math.isnan(force_corrected)):
            delta = pred - force_corrected
            txt   = f"Δ (model − imada) = {delta:+.3f} N"
            color = (
                "#28a745" if abs(delta) < 0.1
                else "#ffc107" if abs(delta) < 0.3
                else "#dc3545"
            )
        elif tare_pending:
            txt = f"model tare: {tare_count}/{MODEL_TARE_SAMPLES}"
            color = "#ffc107"
        else:
            txt, color = "Δ = ---", "#adb5bd"
        self.lbl_delta.configure(text=txt, text_color=color)

        # Collect sample nếu đang logging. Ghi cả raw/tared để debug bias.
        if self._logging_active:
            now = time.time()
            now_mono = time.monotonic()
            abs_delta = abs(delta) if not math.isnan(delta) else math.nan
            frame_age_s = (
                now_mono - frame_ts_mono if not math.isnan(frame_ts_mono) else math.nan
            )
            force_age_s = (
                now_mono - force_aligned_ts_mono
                if not math.isnan(force_aligned_ts_mono)
                else math.nan
            )
            force_latest_age_s = (
                now_mono - force_latest_ts_mono
                if not math.isnan(force_latest_ts_mono)
                else math.nan
            )
            frame_latest_force_dt_s = (
                frame_ts_mono - force_latest_ts_mono
                if not (math.isnan(frame_ts_mono) or math.isnan(force_latest_ts_mono))
                else math.nan
            )
            self._delta_samples.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "log_id": self._log_id,
                "t_rel_s": round(now - self._log_start_time, 3),
                "sample_ts_mono": round(now_mono, 6),
                "frame_ts_mono": frame_ts_mono,
                "force_ts_mono": force_aligned_ts_mono,
                "force_latest_ts_mono": force_latest_ts_mono,
                "frame_age_s": frame_age_s,
                "force_age_s": force_age_s,
                "force_latest_age_s": force_latest_age_s,
                "frame_force_dt_s": frame_force_dt_s,
                "frame_latest_force_dt_s": frame_latest_force_dt_s,
                "infer_ms": infer_ms,
                "command": self._log_command,
                "distance_mm": self._log_distance,
                "motor_state": self._motor_state,
                "model_label": model_label,
                "model_type": model_type,
                "pred_raw_n": pred_raw,
                "model_zero_n": model_zero,
                "pred_tared_n": pred_tared,
                "pred_display_n": pred,
                "imada_raw_n": force_aligned_raw,
                "imada_raw_line": force_aligned_raw_line,
                "imada_zero_n": zero,
                "imada_corrected_n": force_corrected,
                "imada_latest_raw_n": force_latest_raw,
                "imada_latest_raw_line": force_latest_raw_line,
                "imada_latest_corrected_n": force_latest_corrected,
                "delta_n": delta,
                "abs_delta_n": abs_delta,
                "is_spike": int(not math.isnan(abs_delta) and abs_delta >= DELTA_SPIKE_N),
                "spike_threshold_n": DELTA_SPIKE_N,
                "n_ref_markers": n_ref_markers,
                "n_valid": n_valid,
                "valid_ratio": debug_stats["valid_ratio"],
                "disp_mean_px": debug_stats["disp_mean_px"],
                "disp_max_px": debug_stats["disp_max_px"],
                "dx_mean_px": debug_stats["dx_mean_px"],
                "dy_mean_px": debug_stats["dy_mean_px"],
            })
            self.lbl_log_status.configure(
                text=f"● {self._log_command}  ({len(self._delta_samples)} samples)",
                text_color="#ffc107",
            )

        # Marker count
        marker_text = ""
        if has_ref:
            marker_text = f"valid markers: {n_valid}/{n_ref_markers}"
            if tare_pending:
                marker_text += f" | tare {tare_count}/{MODEL_TARE_SAMPLES}"
            elif not math.isnan(model_zero):
                marker_text += f" | model zero: {model_zero:+.3f}N"
        self.lbl_n_markers.configure(
            text=marker_text
        )

        self.app.after(50, self._update_ui)

    # ------------------------------------------------------------------ #
    #  Toast / status                                                      #
    # ------------------------------------------------------------------ #

    def _show_toast(self, msg: str, *, color: str = "#28a745") -> None:
        if self._toast_widget is not None:
            try:
                self._toast_widget.destroy()
            except Exception:
                pass
        if self._toast_after_id is not None:
            self.app.after_cancel(self._toast_after_id)

        toast = ctk.CTkFrame(self.app, fg_color=color, corner_radius=8)
        ctk.CTkLabel(
            toast, text=f"✓  {msg}",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="white",
        ).pack(padx=18, pady=8)
        toast.place(relx=0.5, rely=0.0, anchor="n", y=10)
        self._toast_widget = toast

        def _dismiss():
            try:
                toast.destroy()
            except Exception:
                pass
            self._toast_widget = None
            self._toast_after_id = None

        self._toast_after_id = self.app.after(2500, _dismiss)

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        prefix = "ERROR: " if error else ""
        self.status_var.set(prefix + msg)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _start_threads(self) -> None:
        for target in (self._worker_camera, self._worker_imada, self._worker_arduino):
            threading.Thread(target=target, daemon=True).start()
        self._update_ui()

    def run(self) -> None:
        try:
            self.app.mainloop()
        finally:
            if self.cap.isOpened():
                self.cap.release()
            if self.arduino_ser:
                self.arduino_ser.close()
            if self.imada_ser:
                self.imada_ser.close()


if __name__ == "__main__":
    ForceInferStation().run()
