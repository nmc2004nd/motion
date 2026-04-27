import customtkinter as ctk
import serial
import threading
import queue
import time

# ====== CONFIG ======
SERIAL_PORT = "/dev/ttyACM0"   # đổi thành /dev/ttyUSB0 nếu Linux
BAUDRATE = 115200

# Khởi tạo serial nhưng không crash nếu chưa cắm mạch
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    connected = True
except Exception as e:
    print(f"Lỗi cổng Serial: {e}")
    ser = None
    connected = False

# Hàng đợi để gửi lệnh Thread-safe
cmd_queue = queue.Queue()

# ====== BACKGROUND SERIAL THREAD ======
def serial_worker():
    while True:
        # Xử lý gửi lệnh
        if not cmd_queue.empty():
            cmd = cmd_queue.get()
            if ser and ser.is_open:
                try:
                    ser.write((cmd + "\n").encode())
                except:
                    pass
        
        # Xử lý đọc trạng thái real-time từ Arduino
        if ser and ser.is_open:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # Cập nhật GUI an toàn từ main thread
                        app.after(0, update_status, line)
            except:
                pass
        
        time.sleep(0.005) # Delay cực nhỏ để không quá tải CPU nhưng vẫn realtime

threading.Thread(target=serial_worker, daemon=True).start()

# ====== GUI HANDLERS ======
def update_status(msg):
    status_var.set(msg)

def is_valid_input(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def forward():
    dist = distance_var.get()
    if is_valid_input(dist):
        cmd_queue.put(f"f {dist}")
        status_var.set(f"Req: Forward {dist} mm")
    else:
        status_var.set("Lỗi: Khoảng cách phải là số thực hợp lệ!")

def backward():
    dist = distance_var.get()
    if is_valid_input(dist):
        cmd_queue.put(f"b {dist}")
        status_var.set(f"Req: Backward {dist} mm")
    else:
        status_var.set("Lỗi: Khoảng cách phải là số thực hợp lệ!")

def stop():
    cmd_queue.put("s")
    status_var.set("Req: STOP")

# ====== MODERN GUI SETUP ======
ctk.set_appearance_mode("Dark")       # Dark, Light, System
ctk.set_default_color_theme("blue")   # blue, dark-blue, green

app = ctk.CTk()
app.title("Linear Controller via Arduino")
app.geometry("450x300")
app.resizable(False, False)

# Variables
status_var = ctk.StringVar(value="Đã kết nối" if connected else "Chưa cắm mạch (Disconnected)")
distance_var = ctk.StringVar(value="150") # Mặc định 150 mm

# Tiêu đề
title = ctk.CTkLabel(app, text="LINEAR DISTANCE CONTROLLER", font=ctk.CTkFont(size=20, weight="bold"))
title.pack(pady=(20, 10))

# Khung nhập liệu
input_frame = ctk.CTkFrame(app, fg_color="transparent")
input_frame.pack(pady=10)

ctk.CTkLabel(input_frame, text="Khoảng cách (mm):", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(0, 10))
step_entry = ctk.CTkEntry(input_frame, textvariable=distance_var, width=120, font=ctk.CTkFont(size=14))
step_entry.grid(row=0, column=1)

# Khung nút bấm
btn_frame = ctk.CTkFrame(app, fg_color="transparent")
btn_frame.pack(pady=15)

btn_fwd = ctk.CTkButton(btn_frame, text="FORWARD 🔼", command=forward, fg_color="#28a745", hover_color="#218838", font=ctk.CTkFont(weight="bold"))
btn_fwd.grid(row=0, column=0, padx=10)

btn_bwd = ctk.CTkButton(btn_frame, text="BACKWARD 🔽", command=backward, fg_color="#007bff", hover_color="#0069d9", font=ctk.CTkFont(weight="bold"))
btn_bwd.grid(row=0, column=1, padx=10)

btn_stop = ctk.CTkButton(app, text="S T O P 🛑", command=stop, fg_color="#dc3545", hover_color="#c82333", width=250, font=ctk.CTkFont(weight="bold"))
btn_stop.pack(pady=5)

# Trạng thái
status_label = ctk.CTkLabel(app, textvariable=status_var, font=ctk.CTkFont(size=13, slant="italic"), text_color="gray")
status_label.pack(pady=10)

app.mainloop()