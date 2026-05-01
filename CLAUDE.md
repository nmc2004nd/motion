# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Giới thiệu dự án

Đây là pipeline theo dõi marker xúc giác (Tactile Marker Tracking) và phát hiện trượt (Slip Detection). Hệ thống xử lý ảnh từ cảm biến xúc giác (marker trắng trên nền tối) để:
- Phát hiện và theo dõi độ dịch chuyển của các marker
- Phát hiện sự kiện trượt (slip) trong tiếp xúc xúc giác
- Hỗ trợ cả xử lý batch (ảnh tĩnh) và theo dõi thời gian thực qua webcam

## Cài đặt

```bash
pip install opencv-python numpy scipy matplotlib pyyaml
```

## Lệnh chạy chương trình

### Xử lý batch (ảnh tĩnh)
```bash
python -m src.main --config-path config/pipeline_config.yaml
```
Kết quả xuất ra `outputs/` dưới dạng 6 ảnh PNG.

### Theo dõi thời gian thực (không có slip detection)
```bash
python -m src.real_time.realtime_pipeline --config-path config/pipeline_config.yaml

# Với profiling (kết quả lưu vào profile_result.prof)
python -m src.real_time.realtime_pipeline --config-path config/pipeline_config.yaml --profile
```

### Theo dõi thời gian thực + phát hiện trượt
```bash
python -m src.slip_prob.realtime_slip_pipeline --config-path config/pipeline_config.yaml
```

### Hiệu chỉnh camera
```python
from src.core.calibration import calibrate_camera
from src.config import load_config
config = load_config("config/pipeline_config.yaml")
calibrate_camera(config=config)
# Kết quả: config/calib_result.npz
```

### Xem profile
```bash
pip install snakeviz
snakeviz profile_result.prof
```

### Điều khiển bàn phím (chế độ thời gian thực)
- `r` — Chụp frame hiện tại làm ảnh tham chiếu
- `c` — Xóa ảnh tham chiếu (dừng theo dõi)
- `q` — Thoát

## Kiến trúc tổng thể

Pipeline theo dõi 6 bước:

```
Tiền xử lý → Phát hiện → Theo dõi → Trực quan hóa → Phân tích → Tổng hợp
```

### Layout thư mục

```
src/
├── config/       # load_config + schema validation (không fallback im lặng)
├── core/         # thuật toán lõi: preprocessing, detection, tracking, calibration, visualization
├── common/       # tiện ích realtime: camera, keyboard, overlay, perf
├── slip/         # detector slip v1 (MRVL) + v2 (translation/radial decomp)
├── pipelines/    # BaseRealtimePipeline + batch + 3 realtime pipelines
├── real_time/    # shim giữ entry point cũ
├── slip_prob/    # shim giữ entry point cũ
├── slip_v2/      # shim giữ entry point cũ
├── flow_raft/    # spike RAFT-Small (thử nghiệm)
└── collection/   # thu thập dữ liệu thực nghiệm (camera + force gauge + motor)
```

Pipeline được điều phối bởi:
- `TactileMarkerTrackingPipeline` (`src/pipelines/batch.py`) — xử lý batch
- `RealtimeTactileTracking` (`src/pipelines/realtime_tracking.py`) — thời gian thực
- `RealtimeSlipTracking` (`src/pipelines/realtime_slip_v1.py`) — realtime + slip v1
- `RealtimeSlipV2Pipeline` (`src/pipelines/realtime_slip_v2.py`) — realtime + slip v2

Cả 3 pipeline realtime đều kế thừa `BaseRealtimePipeline` (`src/pipelines/base.py`),
chỉ override `process_frame()` và (tuỳ chọn) `draw_idle()` / `window_name` /
`on_reference_captured()` / `on_reference_cleared()`. Vòng lặp camera + keyboard
+ lifecycle của reference do base xử lý.

## Luồng dữ liệu

**Chế độ batch:**
```
reference.jpg → preprocess → detect (N markers) ─┐
                                                   ├→ LK track → visualize → outputs/
deformed.jpg  → preprocess                        ─┘
```

**Chế độ thời gian thực (slip detection):**
```
webcam → grayscale → preprocess → detect
                                     ↓
         [nhấn 'r']               LK track ← reference đã lưu
         reference → detect          ↓
                              validity mask
                                     ↓
                  slip_detector ← history buffer (5 frames)
                                     ↓
                  hiển thị (arrows  + slip status)
```

## Thuật toán theo dõi

Hệ thống chỉ sử dụng **Pyramid Lucas-Kanade** (`src/core/tracking.py`):
- Forward tracking → backward tracking → so sánh để loại điểm trôi (forward-backward check)
- Ngưỡng FB error: 2.0 px
- **Deadzone**: dịch chuyển < `tracking.min_displacement` px bị reset về vị trí tham chiếu
  (lọc nhiễu đàn hồi vật liệu). Bật/tắt qua flag `apply_deadzone` — slip V2 dùng `False`
  để giữ tín hiệu slip chậm tích luỹ.
- Tham số: `win_size=(21,21)`, `max_level=3` (pyramid depth), `fb_threshold=2.0`

> Lưu ý: Thuật toán Hungarian (`src/hungarian/`) đã bị xóa khỏi codebase.

## Phát hiện trượt

`SlipDetector` (`src/slip/v1.py`) dùng thống kê vòng tròn:
1. Lọc marker không hợp lệ và dịch chuyển < `min_motion_thresh` (1.0 px)
2. **Rebound filtering**: loại marker đang hồi phục đàn hồi (dot product vận tốc · biến dạng < ngưỡng)
3. Kiểm tra đủ số marker đang di chuyển (≥ `min_moving_markers = 5`)
4. Tính **Mean Resultant Vector Length (R)** từ góc dịch chuyển, có trọng số theo độ lớn
5. Áp dụng EMA: `R_smooth = α * R_raw + (1-α) * R_prev` (α = 0.3)
6. Phân loại: `is_slip = (R_smooth > slip_threshold)`, mặc định `slip_threshold = 0.8`

**Hiển thị slip status:**
- Xanh lá: đang theo dõi ổn định
- Vàng: marker đang di chuyển nhưng chưa đủ điều kiện slip
- Đỏ: slip được phát hiện

## Tối ưu hiệu năng thời gian thực

Các đối tượng được tạo một lần và tái sử dụng qua các frame (do `BaseRealtimePipeline` quản lý):
- `_clahe`: đối tượng CLAHE (tránh khởi tạo lại mỗi frame)
- `_blob_detector`: SimpleBlobDetector

## Cấu hình

Tất cả tham số trong `config/pipeline_config.yaml`. Dùng `load_config("...")` từ
`src.config` — hàm validate toàn bộ required keys ngay khi load và raise
`ConfigError` với danh sách key thiếu. **Code không có fallback mặc định**: các
hàm xử lý dùng `require(config, "dot.path")` — thiếu key là lỗi cấu hình, không
âm thầm chạy với giá trị lạ. Thêm key mới nhớ khai báo ở `src/config/schema.py`.

Các nhóm cấu hình chính: `paths`, `tracking`, `preprocessing`, `detection`,
`slip_detection`, `slip_v2`, `visualization` (có `text` + `overlay_v2`),
`camera`, `calibration`, `controls`, `pipeline`.

Tham số quan trọng:

| Nhóm | Tham số | Mặc định | Ý nghĩa |
|------|---------|----------|---------|
| `tracking.pyrlk` | `win_size` | [21, 21] | Cửa sổ tìm kiếm LK |
| | `max_level` | 3 | Số tầng pyramid |
| | `fb_threshold` | 2.0 | Ngưỡng lỗi forward-backward (px) |
| | `min_displacement` | 2 | Deadzone (px) |
| `preprocessing` | `blur_kernel` | [101, 101] | Kernel ước lượng nền |
| | `clahe_clip_limit` | 2.5 | Cường độ tăng cường độ tương phản |
| `detection` | `min_area` | 30 | Diện tích blob tối thiểu (px²) |
| | `max_area` | 500 | Diện tích blob tối đa |
| | `min_circularity` | 0.5 | Độ tròn (0–1) |
| `slip_detection` | `slip_threshold` | 0.8 | Ngưỡng R để phân loại slip |
| | `min_motion_thresh` | 1.0 | Dịch chuyển tối thiểu để tính (px) |
| | `min_moving_markers` | 5 | Số marker tối thiểu để detect slip |
| | `alpha` | 0.3 | Hệ số EMA |
| | `history_buffer_length` | 5 | Số frame lưu lịch sử |

## Tiền xử lý ảnh (`src/core/preprocessing.py`)

1. **Ước lượng nền**: `cv2.blur` với kernel 101×101 (loại vignetting LED)
2. **Trừ nền**: `cv2.subtract(img, background)` → marker nổi rõ trên nền đồng đều
3. **Chuẩn hóa**: stretch về [0, 255]
4. **CLAHE**: tăng cường độ tương phản cục bộ (`clip_limit=2.5`, `grid=(8,8)`)

## Trực quan hóa (`src/core/visualization.py`)

**Arrows** (`visualize_flow_arrows`):
- Mũi tên từ vị trí tham chiếu → vị trí biến dạng, scale 3×
- Xám: marker không di chuyển; vàng: vị trí gốc; đỏ–vàng: mũi tên (độ sáng ∝ độ lớn dịch chuyển)

## Thu thập dữ liệu (`src/collection/`)

Module thu thập dữ liệu thực nghiệm đồng bộ: ảnh camera + lực từ force gauge Imada ZTA + điều khiển motor tuyến tính qua Arduino.

### Các file

| File | Vai trò |
|---|---|
| `realtime_station.py` | GUI tích hợp: camera + Imada + motor, ghi dữ liệu đồng bộ |
| `control.py` | GUI điều khiển motor độc lập (standalone) |
| `collect.py` | Script đọc force gauge Imada đơn giản, không GUI |
| `Pyserial/Pyserial.ino` | Firmware Arduino điều khiển stepper motor |

### Chạy station tích hợp

```bash
python -u src/collection/realtime_station.py
```

### Phần cứng

| Thiết bị | Port mặc định | Baud |
|---|---|---|
| Arduino (stepper) | `/dev/ttyACM0` | 115200 |
| Imada ZTA (force gauge) | `/dev/ttyACM1` | 19200 |
| Camera | ID `0` | — |

Thay đổi port trong các hằng số đầu file nếu cần.

### Thiết kế đồng bộ dữ liệu

```
Imada thread  → cập nhật _latest_force dưới _lock (20 Hz)
Camera thread → acquire _lock → snapshot (frame, force) cùng lúc → SyncSnapshot → write_queue
Writer thread → wait(recording_event) → dequeue → ghi ảnh + CSV
```

`SyncSnapshot` đảm bảo mỗi bản ghi CSV có `(timestamp, image_name, force)` với frame và lực được lấy tại cùng một thời điểm. Writer thread block hoàn toàn khi không recording — không busy-loop.

### Output

```
data/images/frame_<epoch>.jpg   # ảnh camera
data/record.csv                 # timestamp, image_name, force (N)
```

### Giao thức lệnh Arduino

```
f <mm>   → tiến (ví dụ: "f 150")
b <mm>   → lùi
s        → dừng mềm (deceleration)
```
Arduino phản hồi `"M: Forward X mm"` / `"ERR: invalid distance"` / `"ERR: unknown cmd"`.

### Cài đặt thêm

```bash
pip install pyserial customtkinter pillow
```

## Dữ liệu

```
data/ref/       # Ảnh tham chiếu (trạng thái tĩnh)
data/img/       # Ảnh biến dạng (trạng thái tiếp xúc)
data/calib/     # Ảnh hiệu chỉnh (checkerboard)
data/images/    # Ảnh thu thập từ collection station
data/record.csv # Log lực + ảnh từ collection station
```
