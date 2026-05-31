"""GUI tích hợp camera + Imada force gauge + Arduino motor.

Ghi dữ liệu theo cấu trúc session/trial vào ``data/sessions/<sid>/``.
Mỗi loại sensor có log riêng với timestamp riêng (monotonic), không ép join
ở write-time — downstream training script tự nearest-neighbor.
"""

from __future__ import annotations

import csv
import math
import queue
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import customtkinter as ctk
import serial
from PIL import Image

from src.collection import dataset

# ---------------------------------------------------------------------------- #
#  Hardware constants                                                           #
# ---------------------------------------------------------------------------- #

ARDUINO_PORT = "/dev/ttyACM0"
IMADA_PORT   = "/dev/ttyACM1"
ARDUINO_BAUD = 115200
IMADA_BAUD   = 19200
CAMERA_ID    = 0

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 1080

JPEG_QUALITY  = 95
IMWRITE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

FIRMWARE_INO = Path(__file__).parent / "Pyserial" / "Pyserial.ino"


# ---------------------------------------------------------------------------- #
#  Data structures                                                              #
# ---------------------------------------------------------------------------- #


@dataclass
class FrameSnap:
    ts_mono: float
    ts_wall: float
    frame: object        # np.ndarray


@dataclass
class ForceSnap:
    ts_mono: float
    force_n: float       # NaN nếu parse lỗi


@dataclass
class MotorEvent:
    ts_mono: float
    direction: str       # "cmd" | "resp"
    payload: str


# ---------------------------------------------------------------------------- #
#  Imada parsing                                                                #
# ---------------------------------------------------------------------------- #


_IMADA_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_imada(raw: str) -> float:
    """Parse phản hồi Imada ZTA → giá trị float (Newton).

    Imada gắn unit + status code dính liền giá trị, ví dụ ``-0.001NTO`` hay
    ``+0.469N`` hay ``r+0.469 N``. Bỏ tiền tố ``r``/``R`` rồi regex match
    số ở đầu. Trả ``NaN`` khi không parse được.
    """
    raw = raw.strip()
    if not raw:
        return math.nan
    if raw[0] in ("r", "R"):
        raw = raw[1:]
    m = _IMADA_NUM_RE.match(raw)
    if m is None:
        return math.nan
    try:
        return float(m.group(0))
    except ValueError:
        return math.nan


# ---------------------------------------------------------------------------- #
#  Station                                                                      #
# ---------------------------------------------------------------------------- #


class DataCollectionStation:
    def __init__(self):
        # Synchronization
        self._lock = threading.Lock()
        self._latest_force_n: float = math.nan
        self._latest_frame = None

        # Recording event — writer thread block khi không record
        self._recording_event = threading.Event()

        # Per-trial queues (3 sensor độc lập). maxsize giới hạn để tránh
        # bloat RAM khi disk chậm.
        self._frame_queue: queue.Queue[FrameSnap] = queue.Queue(maxsize=200)
        self._force_queue: queue.Queue[ForceSnap] = queue.Queue(maxsize=500)
        self._motor_queue: queue.Queue[MotorEvent] = queue.Queue(maxsize=500)

        # Motor command queue (GUI → arduino worker)
        self._cmd_queue: queue.Queue[str] = queue.Queue()

        # Active session/trial state
        self._session: dataset.Session | None = None
        self._trial: dataset.Trial | None = None
        self._trial_lock = threading.Lock()

        # Counters cho trial active
        self._n_frames = 0
        self._n_force = 0
        self._n_motor = 0
        self._n_dropped = 0

        # Motor state tracker (cập nhật từ Arduino response)
        self._motor_state = "unknown"  # idle | moving | unknown

        # Toast notification state
        self._toast_widget = None
        self._toast_after_id = None

        self._init_hardware()
        self._init_gui()
        self._start_threads()

    # ------------------------------------------------------------------ #
    #  Hardware                                                            #
    # ------------------------------------------------------------------ #

    def _init_hardware(self):
        try:
            self.arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.05)
            self.arduino_connected = True
        except Exception as e:
            print(f"Arduino port error: {e}")
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
            print(f"Imada port error: {e}")
            self.imada_ser = None
            self.imada_connected = False

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # Lock manual exposure / wb để tránh drift brightness giữa các trial.
        # Giá trị chính xác phụ thuộc backend (V4L2: 1=manual, 3=auto). Cố
        # gắng set, nếu fail thì giữ default.
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass

    def _camera_metadata(self) -> dict:
        """Đọc giá trị thực tế camera đang dùng (sau khi set)."""
        return {
            "id": CAMERA_ID,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "exposure": float(self.cap.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(self.cap.get(cv2.CAP_PROP_GAIN)),
            "auto_exposure": float(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "auto_wb": float(self.cap.get(cv2.CAP_PROP_AUTO_WB)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
        }

    # ------------------------------------------------------------------ #
    #  GUI                                                                 #
    # ------------------------------------------------------------------ #

    def _init_gui(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Realtime Data Collection Station")
        self.app.geometry("1100x780")

        # === Session bar (top) ===========================================
        sess_bar = ctk.CTkFrame(self.app)
        sess_bar.pack(side="top", fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(sess_bar, text="SESSION",
                     font=ctk.CTkFont(weight="bold", size=14)).pack(
                         side="left", padx=(10, 8))

        self.session_id_var = ctk.StringVar(value=dataset.default_session_id())
        ctk.CTkEntry(sess_bar, textvariable=self.session_id_var,
                     width=200).pack(side="left", padx=4)

        self.notes_var = ctk.StringVar(value="")
        ctk.CTkEntry(sess_bar, textvariable=self.notes_var,
                     placeholder_text="notes (gel, lighting, indenter…)",
                     width=320).pack(side="left", padx=4, fill="x", expand=True)

        self.btn_new_session = ctk.CTkButton(
            sess_bar, text="New session", width=110,
            command=self._on_new_session,
            fg_color="#0d6efd", hover_color="#0b5ed7",
        )
        self.btn_new_session.pack(side="left", padx=4)

        self.btn_capture_ref = ctk.CTkButton(
            sess_bar, text="Capture reference", width=140,
            command=self._on_capture_reference,
            fg_color="#6610f2", hover_color="#520dc2",
            state="disabled",
        )
        self.btn_capture_ref.pack(side="left", padx=4)

        # === Session / Trial info bar =====================================
        info_bar = ctk.CTkFrame(self.app, fg_color=("#e9ecef", "#2b2d30"), height=28)
        info_bar.pack(side="top", fill="x", padx=10, pady=(0, 3))
        info_bar.pack_propagate(False)

        self.lbl_sessions_count = ctk.CTkLabel(
            info_bar, text="Sessions: —",
            font=ctk.CTkFont(size=12), text_color="#adb5bd", anchor="w",
        )
        self.lbl_sessions_count.pack(side="left", padx=(14, 0))

        ctk.CTkLabel(info_bar, text="|", font=ctk.CTkFont(size=12),
                     text_color="#495057").pack(side="left", padx=6)

        self.lbl_current_session = ctk.CTkLabel(
            info_bar, text="Session: —",
            font=ctk.CTkFont(size=12), text_color="#adb5bd", anchor="w",
        )
        self.lbl_current_session.pack(side="left", padx=(0, 0))

        ctk.CTkLabel(info_bar, text="|", font=ctk.CTkFont(size=12),
                     text_color="#495057").pack(side="left", padx=6)

        self.lbl_trials_count = ctk.CTkLabel(
            info_bar, text="Trials: —",
            font=ctk.CTkFont(size=12), text_color="#adb5bd", anchor="w",
        )
        self.lbl_trials_count.pack(side="left", padx=(0, 0))

        ctk.CTkLabel(info_bar, text="|", font=ctk.CTkFont(size=12),
                     text_color="#495057").pack(side="left", padx=6)

        self.lbl_current_trial = ctk.CTkLabel(
            info_bar, text="Trial: —",
            font=ctk.CTkFont(size=12), text_color="#adb5bd", anchor="w",
        )
        self.lbl_current_trial.pack(side="left", padx=(0, 0))

        self._refresh_session_info()

        # === Main body ===================================================
        body = ctk.CTkFrame(self.app, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        left = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ctk.CTkLabel(left, text="CAMERA FEED",
                     font=ctk.CTkFont(weight="bold", size=16)).pack(pady=5)
        self.video_label = ctk.CTkLabel(left, text="Loading camera...")
        self.video_label.pack(pady=5, expand=True)

        self.lbl_force = ctk.CTkLabel(
            left, text="Force: --- N",
            font=ctk.CTkFont(size=20, weight="bold"), text_color="#17a2b8",
        )
        self.lbl_force.pack(pady=10)

        # Trial bar (giữa, dưới camera)
        trial_bar = ctk.CTkFrame(left)
        trial_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        ctk.CTkLabel(trial_bar, text="Tags:",
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(8, 4))
        self.tags_var = ctk.StringVar(value="")
        ctk.CTkEntry(trial_bar, textvariable=self.tags_var,
                     placeholder_text="push, slip, release_slow",
                     width=280).pack(side="left", padx=4, fill="x", expand=True)

        self.btn_record = ctk.CTkButton(
            trial_bar, text="🔴 START RECORDING", command=self._toggle_record,
            fg_color="#dc3545", hover_color="#c82333",
            height=40, width=200, font=ctk.CTkFont(weight="bold", size=13),
            state="disabled",
        )
        self.btn_record.pack(side="left", padx=8)

        # === Right: motor ================================================
        right = ctk.CTkFrame(body, width=300)
        right.pack(side="right", fill="y", padx=(5, 0))

        ctk.CTkLabel(right, text="MOTOR CONTROL",
                     font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)

        self.distance_var = ctk.StringVar(value="1")
        inp = ctk.CTkFrame(right, fg_color="transparent")
        inp.pack(pady=10)
        ctk.CTkLabel(inp, text="Distance (mm):").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(inp, textvariable=self.distance_var, width=80).grid(
            row=0, column=1)

        ctk.CTkButton(right, text="FORWARD 🔼", command=self._forward,
                      fg_color="#28a745", hover_color="#218838").pack(
                          pady=10, padx=20, fill="x")
        ctk.CTkButton(right, text="BACKWARD 🔽", command=self._backward,
                      fg_color="#007bff", hover_color="#0069d9").pack(
                          pady=10, padx=20, fill="x")
        ctk.CTkButton(right, text="STOP 🛑", command=self._stop,
                      fg_color="#dc3545", hover_color="#c82333",
                      height=40).pack(pady=20, padx=20, fill="x")

        self.lbl_motor_state = ctk.CTkLabel(
            right, text="Motor: unknown",
            font=ctk.CTkFont(size=13), text_color="#adb5bd",
        )
        self.lbl_motor_state.pack(pady=8)

        conn_text = "Connected" if self.arduino_connected else "Disconnected"
        ctk.CTkLabel(
            right, text=f"Arduino: {conn_text}",
            font=ctk.CTkFont(slant="italic"), text_color="gray",
        ).pack(side="bottom", pady=20)

        # === Status bar (bottom) =========================================
        self.status_var = ctk.StringVar(value="No session — click 'New session' to start.")
        ctk.CTkLabel(
            self.app, textvariable=self.status_var,
            font=ctk.CTkFont(size=12), text_color="#adb5bd",
            anchor="w",
        ).pack(side="bottom", fill="x", padx=14, pady=(0, 8))

    # ------------------------------------------------------------------ #
    #  Session lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def _on_new_session(self):
        if self._recording_event.is_set():
            self._set_status("Stop recording trước khi tạo session mới.", error=True)
            return
        sid = self.session_id_var.get().strip() or dataset.default_session_id()
        try:
            dataset.validate_session_id(sid)
        except ValueError as e:
            self._set_status(str(e), error=True)
            return

        # Force zero offset: lấy median 20 sample lúc bắt đầu (không tiếp xúc).
        force_zero = self._sample_force_zero(n=20)

        firmware_hash = (
            dataset.file_sha256(FIRMWARE_INO) if FIRMWARE_INO.exists() else None
        )

        metadata = {
            "hardware": {
                "arduino_port": ARDUINO_PORT,
                "imada_port": IMADA_PORT,
                "arduino_connected": self.arduino_connected,
                "imada_connected": self.imada_connected,
                "firmware_sha256": firmware_hash,
            },
            "camera": self._camera_metadata(),
            "force": {"zero_offset_n": force_zero},
            "notes": self.notes_var.get().strip(),
            "repro": {
                "git_sha": dataset.git_sha(),
                "pip_versions": dataset.relevant_pip_versions(),
            },
        }

        try:
            self._session = dataset.new_session(sid, metadata)
        except FileExistsError as e:
            self._set_status(str(e), error=True)
            return

        self.btn_capture_ref.configure(state="normal")
        self.btn_record.configure(state="normal")
        self._refresh_session_info()
        self._show_toast(f"Session '{sid}' đã tạo thành công")
        self._set_status(f"Session active: {sid} (force zero={force_zero:.4f}N)")

    def _sample_force_zero(self, n: int = 20) -> float:
        """Đọc n sample force, trả median. NaN nếu Imada không kết nối."""
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

    def _on_capture_reference(self):
        if self._session is None:
            self._set_status("Tạo session trước khi capture reference.", error=True)
            return
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            self._set_status("Chưa có frame từ camera.", error=True)
            return
        ts = time.monotonic()
        path = self._session.reference_dir() / f"ref_{ts:.6f}.jpg"
        cv2.imwrite(str(path), frame, IMWRITE_PARAMS)
        self._show_toast(f"Reference đã lưu: {path.name}", color="#6610f2")
        self._set_status(f"Reference saved: {path.name}")

    # ------------------------------------------------------------------ #
    #  Recording lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def _toggle_record(self):
        if not self._recording_event.is_set():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self._session is None:
            self._set_status("Tạo session trước khi recording.", error=True)
            return

        # Cảnh báo nếu motor đang chạy.
        if self._motor_state == "moving":
            self._set_status(
                "WARNING: motor đang moving — trial sẽ ghi motor_state_at_start=moving."
            )

        trial = dataset.create_trial(self._session)
        trial.start_ts_mono = time.monotonic()
        trial.start_ts_wall = time.time()
        trial.tags = dataset.parse_tags(self.tags_var.get())
        trial.motor_state_at_start = self._motor_state

        # Header CSV + chuẩn bị counters
        with open(trial.frames_csv(), "w", newline="") as f:
            csv.writer(f).writerow(["ts_mono", "ts_wall", "image_name"])
        with open(trial.force_csv(), "w", newline="") as f:
            csv.writer(f).writerow(["ts_mono", "force_n"])
        with open(trial.motor_csv(), "w", newline="") as f:
            csv.writer(f).writerow(["ts_mono", "direction", "payload"])

        with self._trial_lock:
            self._trial = trial
            self._n_frames = 0
            self._n_force = 0
            self._n_motor = 0
            self._n_dropped = 0

        # Drain stale events trong queues từ idle period
        self._drain_queues()

        self._recording_event.set()
        self.btn_record.configure(
            text="⏹ STOP RECORDING", fg_color="#6c757d", hover_color="#5a6268",
        )
        self._refresh_session_info()
        self._show_toast(f"Bắt đầu ghi: {trial.trial_id}", color="#0d6efd")
        self._set_status(f"Recording {trial.trial_id} (tags={trial.tags})…")

    def _stop_recording(self):
        self._recording_event.clear()
        # Cho writer flush các item còn trong queue. Đợi tối đa 2s.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (
            self._frame_queue.empty()
            and self._force_queue.empty()
            and self._motor_queue.empty()
        ):
            time.sleep(0.05)

        with self._trial_lock:
            trial = self._trial
            n_frames = self._n_frames
            n_force = self._n_force
            n_motor = self._n_motor
            trial.n_dropped = self._n_dropped if trial else 0
            self._trial = None  # block writer truy cập sau khi finalize

        if trial is not None:
            dataset.finalize_trial(
                trial,
                end_ts_mono=time.monotonic(),
                n_frames=n_frames,
                n_force_samples=n_force,
                n_motor_events=n_motor,
            )
            self._set_status(
                f"Saved {trial.trial_id}: frames={n_frames} force={n_force} "
                f"motor={n_motor} dropped={trial.n_dropped}"
            )

        self.btn_record.configure(
            text="🔴 START RECORDING", fg_color="#dc3545", hover_color="#c82333",
        )
        self._refresh_session_info()

    def _drain_queues(self):
        for q in (self._frame_queue, self._force_queue, self._motor_queue):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    # ------------------------------------------------------------------ #
    #  Motor commands                                                      #
    # ------------------------------------------------------------------ #

    def _forward(self):
        try:
            d = float(self.distance_var.get())
        except ValueError:
            return
        self._cmd_queue.put(f"f {d}")

    def _backward(self):
        try:
            d = float(self.distance_var.get())
        except ValueError:
            return
        self._cmd_queue.put(f"b {d}")

    def _stop(self):
        self._cmd_queue.put("s")

    # ------------------------------------------------------------------ #
    #  Worker threads                                                      #
    # ------------------------------------------------------------------ #

    def _worker_camera(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts_mono = time.monotonic()

            with self._lock:
                self._latest_frame = frame

            if self._recording_event.is_set():
                snap = FrameSnap(
                    ts_mono=ts_mono,
                    ts_wall=time.time(),
                    frame=frame.copy(),
                )
                try:
                    self._frame_queue.put_nowait(snap)
                except queue.Full:
                    with self._trial_lock:
                        self._n_dropped += 1

    def _worker_imada(self):
        while True:
            if self.imada_connected and self.imada_ser.is_open:
                try:
                    self.imada_ser.write(b"D\r")
                    raw = self.imada_ser.read_until(b"\r").decode(
                        "utf-8", errors="ignore",
                    ).strip()
                    if raw:
                        ts_mono = time.monotonic()
                        force_n = parse_imada(raw)
                        with self._lock:
                            self._latest_force_n = force_n
                        # UI update — capture giá trị qua default arg
                        self.app.after(
                            0,
                            lambda v=force_n: self.lbl_force.configure(
                                text=(
                                    f"Force: {v:+.3f} N"
                                    if not math.isnan(v) else "Force: --- N"
                                )
                            ),
                        )
                        if self._recording_event.is_set():
                            try:
                                self._force_queue.put_nowait(
                                    ForceSnap(ts_mono=ts_mono, force_n=force_n)
                                )
                            except queue.Full:
                                pass
                except Exception as e:
                    print(f"Imada read error: {e}")
            time.sleep(0.05)

    def _worker_arduino(self):
        while True:
            # Send queued commands
            try:
                cmd = self._cmd_queue.get_nowait()
                if self.arduino_connected and self.arduino_ser.is_open:
                    self.arduino_ser.write((cmd + "\n").encode())
                    self._record_motor("cmd", cmd)
                    # Optimistic state update — confirm khi response đến
                    if cmd.startswith(("f ", "b ")):
                        self._set_motor_state("moving")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Arduino write error: {e}")

            # Read responses
            if self.arduino_connected and self.arduino_ser.is_open:
                try:
                    if self.arduino_ser.in_waiting:
                        line = self.arduino_ser.readline().decode(
                            "utf-8", errors="ignore",
                        ).strip()
                        if line:
                            self._record_motor("resp", line)
                            self._update_motor_state_from_resp(line)
                except Exception as e:
                    print(f"Arduino read error: {e}")

            time.sleep(0.005)

    def _record_motor(self, direction: str, payload: str):
        ts_mono = time.monotonic()
        if self._recording_event.is_set():
            try:
                self._motor_queue.put_nowait(
                    MotorEvent(ts_mono=ts_mono, direction=direction, payload=payload)
                )
            except queue.Full:
                pass

    def _update_motor_state_from_resp(self, resp: str):
        if resp.startswith("M: Stopped"):
            self._set_motor_state("idle")
        elif resp.startswith("M: Forward") or resp.startswith("M: Backward"):
            # Firmware echo "M: Forward 50.0 mm" ngay khi nhận lệnh, motor sẽ
            # chạy hết quãng đường. Giữ "moving" — chỉ "Stopped" mới về idle.
            self._set_motor_state("moving")

    def _set_motor_state(self, state: str):
        self._motor_state = state
        color = {"idle": "#28a745", "moving": "#ffc107", "unknown": "#adb5bd"}.get(
            state, "#adb5bd"
        )
        self.app.after(
            0,
            lambda s=state, c=color: self.lbl_motor_state.configure(
                text=f"Motor: {s}", text_color=c
            ),
        )

    def _worker_writer(self):
        """Block khi không recording. Ghi từ 3 queue song song."""
        while True:
            self._recording_event.wait()
            self._drain_one_iteration()

    def _drain_one_iteration(self):
        # Process từng queue với timeout ngắn để cycle đều giữa 3 sensor.
        with self._trial_lock:
            trial = self._trial
        if trial is None:
            time.sleep(0.05)
            return

        # Frame
        try:
            snap: FrameSnap = self._frame_queue.get(timeout=0.05)
            self._write_frame(trial, snap)
        except queue.Empty:
            pass

        # Force
        try:
            f: ForceSnap = self._force_queue.get_nowait()
            self._write_force(trial, f)
        except queue.Empty:
            pass

        # Motor
        try:
            m: MotorEvent = self._motor_queue.get_nowait()
            self._write_motor(trial, m)
        except queue.Empty:
            pass

    def _write_frame(self, trial: dataset.Trial, snap: FrameSnap):
        img_name = f"frame_{snap.ts_mono:.6f}.jpg"
        try:
            cv2.imwrite(
                str(trial.frames_dir() / img_name), snap.frame, IMWRITE_PARAMS
            )
            with open(trial.frames_csv(), "a", newline="") as f:
                csv.writer(f).writerow(
                    [f"{snap.ts_mono:.6f}", f"{snap.ts_wall:.6f}", img_name]
                )
            with self._trial_lock:
                self._n_frames += 1
        except Exception as e:
            print(f"Frame write error: {e}")

    def _write_force(self, trial: dataset.Trial, f: ForceSnap):
        try:
            with open(trial.force_csv(), "a", newline="") as fp:
                val = "nan" if math.isnan(f.force_n) else f"{f.force_n:.6f}"
                csv.writer(fp).writerow([f"{f.ts_mono:.6f}", val])
            with self._trial_lock:
                self._n_force += 1
        except Exception as e:
            print(f"Force write error: {e}")

    def _write_motor(self, trial: dataset.Trial, m: MotorEvent):
        try:
            with open(trial.motor_csv(), "a", newline="") as fp:
                csv.writer(fp).writerow(
                    [f"{m.ts_mono:.6f}", m.direction, m.payload]
                )
            with self._trial_lock:
                self._n_motor += 1
        except Exception as e:
            print(f"Motor write error: {e}")

    # ------------------------------------------------------------------ #
    #  UI loop                                                             #
    # ------------------------------------------------------------------ #

    def _update_ui(self):
        with self._lock:
            frame = self._latest_frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (480, 360))
            pil = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(480, 360))
            self.video_label.configure(image=ctk_img, text="")

        # Refresh status nếu đang record
        if self._recording_event.is_set():
            with self._trial_lock:
                if self._trial is not None:
                    self._set_status(
                        f"Recording {self._trial.trial_id}: "
                        f"frames={self._n_frames} force={self._n_force} "
                        f"motor={self._n_motor} dropped={self._n_dropped}"
                    )

        self.app.after(50, self._update_ui)

    def _refresh_session_info(self):
        """Cập nhật info bar: tổng session, session hiện tại, số trial, trial đang active."""
        root = dataset.data_root()
        if root.exists():
            n_sessions = sum(
                1 for p in root.iterdir()
                if p.is_dir() and (p / "session.yaml").exists()
            )
        else:
            n_sessions = 0
        self.lbl_sessions_count.configure(text=f"Sessions: {n_sessions}")

        if self._session is None:
            self.lbl_current_session.configure(text="Session: —", text_color="#adb5bd")
            self.lbl_trials_count.configure(text="Trials: —", text_color="#adb5bd")
            self.lbl_current_trial.configure(text="Trial: —", text_color="#adb5bd")
            return

        self.lbl_current_session.configure(
            text=f"Session: {self._session.session_id}", text_color="#17a2b8",
        )

        trials_dir = self._session.trials_dir()
        n_trials = (
            sum(1 for p in trials_dir.iterdir() if p.is_dir())
            if trials_dir.exists() else 0
        )
        self.lbl_trials_count.configure(
            text=f"Trials: {n_trials}", text_color="#adb5bd",
        )

        with self._trial_lock:
            trial = self._trial
        if trial is not None:
            self.lbl_current_trial.configure(
                text=f"Trial: {trial.trial_id} ●", text_color="#dc3545",
            )
        else:
            next_id = dataset.next_trial_id(self._session)
            self.lbl_current_trial.configure(
                text=f"Trial: — (next: {next_id})", text_color="#adb5bd",
            )

    def _show_toast(self, msg: str, *, color: str = "#28a745"):
        """Hiện toast thành công ở trên cùng cửa sổ, tự ẩn sau 2.5s."""
        if self._toast_widget is not None:
            try:
                self._toast_widget.destroy()
            except Exception:
                pass
        if self._toast_after_id is not None:
            self.app.after_cancel(self._toast_after_id)

        toast = ctk.CTkFrame(self.app, fg_color=color, corner_radius=8)
        ctk.CTkLabel(
            toast,
            text=f"✓  {msg}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="white",
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

    def _set_status(self, msg: str, *, error: bool = False):
        prefix = "ERROR: " if error else ""
        self.status_var.set(prefix + msg)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _start_threads(self):
        for target in (
            self._worker_camera,
            self._worker_imada,
            self._worker_arduino,
            self._worker_writer,
        ):
            threading.Thread(target=target, daemon=True).start()
        self._update_ui()

    def run(self):
        try:
            self.app.mainloop()
        finally:
            if self._recording_event.is_set():
                self._stop_recording()
            if self.cap.isOpened():
                self.cap.release()
            if self.arduino_ser:
                self.arduino_ser.close()
            if self.imada_ser:
                self.imada_ser.close()


if __name__ == "__main__":
    DataCollectionStation().run()
