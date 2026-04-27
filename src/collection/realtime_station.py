import os
import time
import csv
import threading
import queue
from dataclasses import dataclass

import cv2
import serial
import customtkinter as ctk
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR  = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'record.csv')

ARDUINO_PORT = "/dev/ttyACM0"
IMADA_PORT   = "/dev/ttyACM1"
ARDUINO_BAUD = 115200
IMADA_BAUD   = 19200
CAMERA_ID    = 0


@dataclass
class SyncSnapshot:
    """Frame và force được snapshot cùng một thời điểm dưới lock."""
    frame: object      # np.ndarray
    force: str
    timestamp: float


def parse_imada(raw: str) -> str:
    """Trích giá trị số từ phản hồi Imada, ví dụ 'r+0.469 N' → '+0.469'."""
    raw = raw.strip()
    if raw and raw[0] in ('r', 'R'):
        token = raw[1:].split()
        return token[0] if token else raw
    return raw


class DataCollectionStation:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_force = "0.0"
        self._latest_frame = None

        # Writer chỉ hoạt động khi event được set
        self._recording_event = threading.Event()

        # Queue có giới hạn để tránh tích lũy frame khi disk chậm (~7s ở 30fps)
        self._write_queue: queue.Queue[SyncSnapshot] = queue.Queue(maxsize=200)

        self.cmd_queue: queue.Queue[str] = queue.Queue()

        self._init_hardware()
        self._init_gui()
        self._start_threads()

    # ------------------------------------------------------------------ #
    #  Hardware                                                            #
    # ------------------------------------------------------------------ #

    def _init_hardware(self):
        try:
            self.arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.01)
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

    # ------------------------------------------------------------------ #
    #  GUI                                                                 #
    # ------------------------------------------------------------------ #

    def _init_gui(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Realtime Data Collection Station")
        self.app.geometry("800x600")

        # --- Left: camera + force + record ---
        left = ctk.CTkFrame(self.app)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(left, text="CAMERA FEED",
                     font=ctk.CTkFont(weight="bold", size=16)).pack(pady=5)

        self.video_label = ctk.CTkLabel(left, text="Loading camera...")
        self.video_label.pack(pady=5, expand=True)

        self.lbl_force = ctk.CTkLabel(
            left, text="Force: 0.0 N",
            font=ctk.CTkFont(size=20, weight="bold"), text_color="#17a2b8",
        )
        self.lbl_force.pack(pady=10)

        self.btn_record = ctk.CTkButton(
            left, text="🔴 START RECORDING", command=self._toggle_record,
            fg_color="#dc3545", hover_color="#c82333",
            height=50, font=ctk.CTkFont(weight="bold", size=14),
        )
        self.btn_record.pack(pady=10, fill="x", padx=20)

        # --- Right: motor ---
        right = ctk.CTkFrame(self.app, width=300)
        right.pack(side="right", fill="y", padx=10, pady=10)

        ctk.CTkLabel(right, text="MOTOR CONTROL",
                     font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)

        self.distance_var = ctk.StringVar(value="150")
        inp = ctk.CTkFrame(right, fg_color="transparent")
        inp.pack(pady=10)
        ctk.CTkLabel(inp, text="Distance (mm):").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(inp, textvariable=self.distance_var, width=80).grid(row=0, column=1)

        ctk.CTkButton(right, text="FORWARD 🔼", command=self._forward,
                      fg_color="#28a745", hover_color="#218838").pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(right, text="BACKWARD 🔽", command=self._backward,
                      fg_color="#007bff", hover_color="#0069d9").pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(right, text="STOP 🛑", command=self._stop,
                      fg_color="#dc3545", hover_color="#c82333",
                      height=40).pack(pady=30, padx=20, fill="x")

        conn_text = "Connected" if self.arduino_connected else "Disconnected"
        ctk.CTkLabel(
            right, text=f"Arduino: {conn_text}",
            font=ctk.CTkFont(slant="italic"), text_color="gray",
        ).pack(side="bottom", pady=20)

    # ------------------------------------------------------------------ #
    #  Recording toggle                                                    #
    # ------------------------------------------------------------------ #

    def _toggle_record(self):
        if not self._recording_event.is_set():
            os.makedirs(IMG_DIR, exist_ok=True)
            file_exists = os.path.isfile(CSV_PATH)
            with open(CSV_PATH, 'a', newline='') as f:
                if not file_exists:
                    csv.writer(f).writerow(['timestamp', 'image_name', 'force'])
            self._recording_event.set()
            self.btn_record.configure(
                text="⏹ STOP RECORDING", fg_color="#6c757d", hover_color="#5a6268",
            )
        else:
            self._recording_event.clear()
            self.btn_record.configure(
                text="🔴 START RECORDING", fg_color="#dc3545", hover_color="#c82333",
            )

    # ------------------------------------------------------------------ #
    #  Motor commands                                                      #
    # ------------------------------------------------------------------ #

    def _forward(self):
        try:
            self.cmd_queue.put(f"f {float(self.distance_var.get())}")
        except ValueError:
            pass

    def _backward(self):
        try:
            self.cmd_queue.put(f"b {float(self.distance_var.get())}")
        except ValueError:
            pass

    def _stop(self):
        self.cmd_queue.put("s")

    # ------------------------------------------------------------------ #
    #  Worker threads                                                      #
    # ------------------------------------------------------------------ #

    def _worker_camera(self):
        """
        Đọc frame → acquire lock → snapshot (frame, force) cùng lúc.
        Đảm bảo mỗi bản ghi có frame và giá trị lực tương ứng cùng thời điểm.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame
                force_snapshot = self._latest_force  # atomic với frame

            if self._recording_event.is_set():
                snapshot = SyncSnapshot(
                    frame=frame.copy(),      # deep copy trước khi enqueue
                    force=force_snapshot,
                    timestamp=time.time(),
                )
                try:
                    self._write_queue.put_nowait(snapshot)
                except queue.Full:
                    pass  # drop frame nếu writer không kịp xử lý

    def _worker_imada(self):
        while True:
            if self.imada_connected and self.imada_ser.is_open:
                try:
                    self.imada_ser.write(b"D\r")
                    raw = self.imada_ser.readline().decode('utf-8', errors='ignore').strip()
                    if raw:
                        parsed = parse_imada(raw)
                        with self._lock:
                            self._latest_force = parsed
                        # Dùng default argument để capture giá trị hiện tại của parsed
                        self.app.after(0, lambda v=parsed: self.lbl_force.configure(
                            text=f"Force: {v} N"
                        ))
                except Exception as e:
                    print(f"Imada read error: {e}")
            time.sleep(0.05)

    def _worker_arduino(self):
        while True:
            try:
                cmd = self.cmd_queue.get(timeout=0.1)
                if self.arduino_connected and self.arduino_ser.is_open:
                    self.arduino_ser.write((cmd + "\n").encode())
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Arduino write error: {e}")

    def _worker_writer(self):
        """Block khi không recording, không busy-loop."""
        while True:
            self._recording_event.wait()  # ngủ cho đến khi record bắt đầu
            try:
                snapshot = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ts = f"{snapshot.timestamp:.4f}"
            img_name = f"frame_{ts}.jpg"
            try:
                cv2.imwrite(os.path.join(IMG_DIR, img_name), snapshot.frame)
                with open(CSV_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow([ts, img_name, snapshot.force])
            except Exception as e:
                print(f"Write error: {e}")

    # ------------------------------------------------------------------ #
    #  UI update loop                                                      #
    # ------------------------------------------------------------------ #

    def _update_ui(self):
        with self._lock:
            frame = self._latest_frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (400, 300))
            pil = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(400, 300))
            self.video_label.configure(image=ctk_img, text="")
        self.app.after(30, self._update_ui)

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
        self.app.mainloop()
        if self.cap.isOpened():
            self.cap.release()
        if self.arduino_ser:
            self.arduino_ser.close()
        if self.imada_ser:
            self.imada_ser.close()


if __name__ == "__main__":
    DataCollectionStation().run()
