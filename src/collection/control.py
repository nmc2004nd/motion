import threading
import queue
import time

import serial
import customtkinter as ctk

SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE    = 115200


class MotorControlApp:
    def __init__(self):
        self.cmd_queue: queue.Queue[str] = queue.Queue()
        self._init_serial()
        self._init_gui()
        threading.Thread(target=self._serial_worker, daemon=True).start()

    def _init_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
            self.connected = True
        except Exception as e:
            print(f"Serial port error: {e}")
            self.ser = None
            self.connected = False

    def _init_gui(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Linear Controller via Arduino")
        self.app.geometry("450x300")
        self.app.resizable(False, False)

        self.status_var  = ctk.StringVar(
            value="Đã kết nối" if self.connected else "Chưa cắm mạch (Disconnected)"
        )
        self.distance_var = ctk.StringVar(value="150")

        ctk.CTkLabel(
            self.app, text="LINEAR DISTANCE CONTROLLER",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(20, 10))

        inp = ctk.CTkFrame(self.app, fg_color="transparent")
        inp.pack(pady=10)
        ctk.CTkLabel(inp, text="Khoảng cách (mm):",
                     font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkEntry(inp, textvariable=self.distance_var, width=120,
                     font=ctk.CTkFont(size=14)).grid(row=0, column=1)

        btns = ctk.CTkFrame(self.app, fg_color="transparent")
        btns.pack(pady=15)
        ctk.CTkButton(btns, text="FORWARD 🔼", command=self._forward,
                      fg_color="#28a745", hover_color="#218838",
                      font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10)
        ctk.CTkButton(btns, text="BACKWARD 🔽", command=self._backward,
                      fg_color="#007bff", hover_color="#0069d9",
                      font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            self.app, text="S T O P 🛑", command=self._stop,
            fg_color="#dc3545", hover_color="#c82333", width=250,
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=5)

        ctk.CTkLabel(
            self.app, textvariable=self.status_var,
            font=ctk.CTkFont(size=13, slant="italic"), text_color="gray",
        ).pack(pady=10)

    # ------------------------------------------------------------------ #
    #  Commands                                                            #
    # ------------------------------------------------------------------ #

    def _get_distance(self) -> float | None:
        try:
            return float(self.distance_var.get())
        except ValueError:
            self.status_var.set("Lỗi: Khoảng cách phải là số thực hợp lệ!")
            return None

    def _forward(self):
        d = self._get_distance()
        if d is not None:
            self.cmd_queue.put(f"f {d}")
            self.status_var.set(f"Req: Forward {d} mm")

    def _backward(self):
        d = self._get_distance()
        if d is not None:
            self.cmd_queue.put(f"b {d}")
            self.status_var.set(f"Req: Backward {d} mm")

    def _stop(self):
        self.cmd_queue.put("s")
        self.status_var.set("Req: STOP")

    # ------------------------------------------------------------------ #
    #  Serial worker                                                       #
    # ------------------------------------------------------------------ #

    def _serial_worker(self):
        while True:
            # Gửi lệnh nếu có
            try:
                cmd = self.cmd_queue.get_nowait()
                if self.ser and self.ser.is_open:
                    self.ser.write((cmd + "\n").encode())
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Serial write error: {e}")

            # Đọc phản hồi từ Arduino
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.app.after(0, self.status_var.set, line)
                except Exception as e:
                    print(f"Serial read error: {e}")

            time.sleep(0.005)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        self.app.mainloop()
        if self.ser:
            self.ser.close()


if __name__ == "__main__":
    MotorControlApp().run()
