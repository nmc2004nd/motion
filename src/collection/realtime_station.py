import os
import time
import csv
import threading
import queue
import cv2
import serial
import customtkinter as ctk
from PIL import Image, ImageTk

# ====== CONFIG PATHS ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'record.csv')

# Đảm bảo thư mục tồn tại
os.makedirs(IMG_DIR, exist_ok=True)

# ====== HARDWARE CONFIG ======
# Tuỳ chỉnh lại port tùy theo kết nối thực tế ở Linux
ARDUINO_PORT = "/dev/ttyACM0"  
IMADA_PORT = "/dev/ttyACM1" # Giả sử Imada cắm ở cổng khác

ARDUINO_BAUD = 115200
IMADA_BAUD = 19200
CAMERA_ID = 0

class DataCollectionStation:
    def __init__(self):
        # State
        self.is_recording = False
        self.latest_frame = None
        self.latest_force = "0.0"
        self.latest_frame_time = 0
        
        # Arduino Queue
        self.cmd_queue = queue.Queue()
        
        # Connect hardware
        self.init_hardware()
        
        # Setup GUI
        self.init_gui()
        
        # Start Threads
        self.start_threads()

    def init_hardware(self):
        # 1. Arduino
        try:
            self.arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.01)
            self.arduino_connected = True
        except Exception as e:
            print(f"Lỗi cổng Arduino: {e}")
            self.arduino_ser = None
            self.arduino_connected = False

        # 2. Imada Force Gauge
        try:
            self.imada_ser = serial.Serial(
                port=IMADA_PORT, baudrate=IMADA_BAUD,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=0.1
            )
            self.imada_connected = True
        except Exception as e:
            print(f"Lỗi cổng Imada: {e}")
            self.imada_ser = None
            self.imada_connected = False

        # 3. Camera
        self.cap = cv2.VideoCapture(CAMERA_ID)
        # Giảm buffer size của camera để lấy realtime nhất
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def init_gui(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Realtime Data Collection Station")
        self.app.geometry("800x600")

        # ====== LEFT PANEL (CAMERA & RECORDING) ======
        self.left_frame = ctk.CTkFrame(self.app)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.left_frame, text="CAMERA FEED", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=5)
        
        # Image display label
        self.video_label = ctk.CTkLabel(self.left_frame, text="Loading camera...")
        self.video_label.pack(pady=5, expand=True)

        self.lbl_imada = ctk.CTkLabel(self.left_frame, text="Force: 0.0 N", font=ctk.CTkFont(size=20, weight="bold"), text_color="#17a2b8")
        self.lbl_imada.pack(pady=10)

        self.btn_record = ctk.CTkButton(self.left_frame, text="🔴 START RECORDING", command=self.toggle_record, 
                                        fg_color="#dc3545", hover_color="#c82333", height=50, font=ctk.CTkFont(weight="bold", size=14))
        self.btn_record.pack(pady=10, fill="x", padx=20)

        # ====== RIGHT PANEL (ARDUINO CONTROLS) ======
        self.right_frame = ctk.CTkFrame(self.app, width=300)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.right_frame, text="MOTOR CONTROL", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)

        self.distance_var = ctk.StringVar(value="150")
        
        input_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        input_frame.pack(pady=10)
        ctk.CTkLabel(input_frame, text="Distance (mm):").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(input_frame, textvariable=self.distance_var, width=80).grid(row=0, column=1)

        ctk.CTkButton(self.right_frame, text="FORWARD 🔼", command=self.forward, fg_color="#28a745", hover_color="#218838").pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.right_frame, text="BACKWARD 🔽", command=self.backward, fg_color="#007bff", hover_color="#0069d9").pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.right_frame, text="STOP 🛑", command=self.stop, fg_color="#dc3545", hover_color="#c82333", height=40).pack(pady=30, padx=20, fill="x")

        self.status_var = ctk.StringVar(value="Arduino: " + ("Connected" if self.arduino_connected else "Disconnected"))
        ctk.CTkLabel(self.right_frame, textvariable=self.status_var, font=ctk.CTkFont(slant="italic"), text_color="gray").pack(side="bottom", pady=20)


    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.configure(text="⏹ STOP RECORDING", fg_color="#6c757d", hover_color="#5a6268")
            
            # Setup CSV nếu là lần ghi đầu tiên chưa có header
            file_exists = os.path.isfile(CSV_PATH)
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'image_name', 'force'])
                    
        else:
            self.is_recording = False
            self.btn_record.configure(text="🔴 START RECORDING", fg_color="#dc3545", hover_color="#c82333")

    # --- ARDUINO COMMANDS ---
    def forward(self):
        try:
            d = float(self.distance_var.get())
            self.cmd_queue.put(f"f {d}")
        except: pass

    def backward(self):
        try:
            d = float(self.distance_var.get())
            self.cmd_queue.put(f"b {d}")
        except: pass

    def stop(self):
        self.cmd_queue.put("s")

    # ====== WORKERS ======
    def worker_camera(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.latest_frame_time = time.perf_counter()
            time.sleep(0.01)

    def worker_imada(self):
        while True:
            if self.imada_connected and self.imada_ser.is_open:
                try:
                    self.imada_ser.write(b"D\r")
                    line = self.imada_ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.latest_force = line
                        # Update UI Force
                        self.app.after(0, lambda: self.lbl_imada.configure(text=f"Force: {self.latest_force} N"))
                except: pass
            time.sleep(0.05) 

    def worker_arduino(self):
        while True:
            if not self.cmd_queue.empty():
                cmd = self.cmd_queue.get()
                if self.arduino_connected and self.arduino_ser.is_open:
                    try:
                        self.arduino_ser.write((cmd + "\n").encode())
                    except: pass
            time.sleep(0.01)

    def worker_writer(self):
        last_saved_time = 0
        while True:
            if self.is_recording:
                # Đảm bảo chỉ ghi 1 ảnh 1 lần bằng cách so sánh timestamp frame
                if self.latest_frame is not None and self.latest_frame_time != last_saved_time:
                    curr_time = self.latest_frame_time
                    last_saved_time = curr_time
                    force_val = self.latest_force
                    
                    # Chuẩn bị dữ liệu
                    timestamp_str = f"{time.time():.4f}" # Dùng epoch time thực tế để lưu file
                    img_name = f"frame_{timestamp_str}.jpg"
                    img_path = os.path.join(IMG_DIR, img_name)
                    
                    # Ghi ảnh
                    cv2.imwrite(img_path, self.latest_frame)
                    
                    # Ghi CSV
                    with open(CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, img_name, force_val])
            
            time.sleep(0.005) # Loop cực nhanh chờ frame mới
            
    def update_ui(self):
        # Update camera image on UI
        if self.latest_frame is not None:
            # Chuyển OpenCV BGR sang RGB cho Tkinter
            cv_img = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            # Resize hiển thị cho nhẹ, không ảnh hưởng frame lưu
            cv_img = cv2.resize(cv_img, (400, 300))
            pil_img = Image.fromarray(cv_img)
            
            # Ctk Image
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(400, 300))
            self.video_label.configure(image=ctk_img, text="")
            
        self.app.after(30, self.update_ui) 

    def start_threads(self):
        threading.Thread(target=self.worker_camera, daemon=True).start()
        threading.Thread(target=self.worker_imada, daemon=True).start()
        threading.Thread(target=self.worker_arduino, daemon=True).start()
        threading.Thread(target=self.worker_writer, daemon=True).start()
        
        # Start chu kỳ update GUI
        self.update_ui()

    def run(self):
        self.app.mainloop()
        # Cleanup khi tắt UI
        if self.cap.isOpened():
            self.cap.release()
        if self.arduino_ser:
            self.arduino_ser.close()
        if self.imada_ser:
            self.imada_ser.close()

if __name__ == "__main__":
    app = DataCollectionStation()
    app.run()
