import re
import sys
import time

import serial

PORT = '/dev/ttyACM1'
BAUD = 19200

_IMADA_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_imada(raw: str) -> float | None:
    """Parse phản hồi Imada ZTA → float Newton.

    Format gặp thực tế: ``-0.001NTO``, ``+0.469N``, ``r+0.469 N``. Unit/status
    code dính liền giá trị nên dùng regex bắt số ở đầu thay vì split.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw[0] in ('r', 'R'):
        raw = raw[1:]
    m = _IMADA_NUM_RE.match(raw)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
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
