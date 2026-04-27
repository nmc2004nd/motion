import sys
import time

import serial

PORT = '/dev/ttyACM0'
BAUD = 19200


def parse_imada(raw: str) -> float | None:
    """Parse phản hồi Imada ZTA, ví dụ 'r+0.469 N' → 0.469."""
    raw = raw.strip()
    if not raw:
        return None
    token = raw[1:] if raw[0] in ('r', 'R') else raw
    try:
        return float(token.split()[0])
    except (ValueError, IndexError):
        return None


def main():
    try:
        ser = serial.Serial(
            port=PORT, baudrate=BAUD,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=0.5,
        )
    except serial.SerialException as e:
        print(f"Cannot open port {PORT}: {e}")
        sys.exit(1)

    print("--- Đang tự động yêu cầu dữ liệu từ Imada ZTA ---")
    try:
        while True:
            ser.write(b"D\r")
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw:
                force = parse_imada(raw)
                if force is not None:
                    print(f"Lực đo được: {force:.3f} N")
                else:
                    print(f"[raw] {raw}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDừng thu thập.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
