import serial
import time

PORT = '/dev/ttyACM0'
BAUD = 19200

try:
    # Mở cổng với cấu hình theo đúng bảng RS232C Condition trong ảnh
    ser = serial.Serial(
        port=PORT, 
        baudrate=BAUD, 
        bytesize=serial.EIGHTBITS, 
        parity=serial.PARITY_NONE, 
        stopbits=serial.STOPBITS_ONE, 
        timeout=0.5
    )
    
    print("--- Đang tự động yêu cầu dữ liệu từ Imada ZTA ---")
    
    while True:
        # Gửi lệnh 'D' kèm ký tự xuống dòng [CR] theo đúng Manual
        ser.write(b"D\r") 
        
        # Đọc phản hồi
        line = ser.readline().decode('utf-8').strip()
        
        if line:
            # Lọc dữ liệu số (ví dụ: r+0.469...)
            print(f"Lực đo được: {line}")
            
        # Nghỉ 0.1s để lấy mẫu với tốc độ 10Hz
        time.sleep(0.1)

except Exception as e:
    print(f"Lỗi: {e}")
finally:
    if 'ser' in locals():
        ser.close()