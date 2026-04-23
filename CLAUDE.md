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
Kết quả xuất ra `outputs/{H|LK}/` dưới dạng 6 ảnh PNG.

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
from src.utils.calibration import calibrate_camera
from src.utils.config_parser import load_config
config = load_config("config/pipeline_config.yaml").config
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

Mỗi bước được tách thành module độc lập trong `src/utils/`. Pipeline được điều phối bởi:
- `TactileMarkerTrackingPipeline` (`src/pipeline.py`) — xử lý batch
- `RealtimeTactileTracking` (`src/real_time/realtime_pipeline.py`) — thời gian thực
- `RealtimeSlipTracking` (`src/slip_prob/realtime_slip_pipeline.py`) — thời gian thực với slip detection

## Luồng dữ liệu

**Chế độ batch:**
```
reference.jpg → preprocess → detect (N markers) ─┐
                                                   ├→ track/match → visualize → outputs/
deformed.jpg  → preprocess → detect (M markers) ─┘
```

**Chế độ thời gian thực (slip detection):**
```
webcam → grayscale → preprocess → detect
                                     ↓
         [nhấn 'r']               match/track ← reference đã lưu
         reference → detect          ↓
                              validity mask
                                     ↓
                  slip_detector ← history buffer (5 frames)
                                     ↓
                  hiển thị (arrows + HSV + slip status)
```

## Thuật toán theo dõi

Chọn thuật toán qua `tracking.method` trong config (`"LK"` hoặc `"H"`):

**Lucas-Kanade (`src/pyr_lk/pyr_lk.py`)**
- Pyramid LK với forward-backward consistency check
- Ngưỡng FB error: 2.0 px
- Deadzone dịch chuyển tối thiểu: 1.5 px (lọc nhiễu đàn hồi vật liệu)

**Hungarian (`src/hungarian/hungarian.py`)**
- Phát hiện marker ở cả 2 frame, ghép cặp qua Hungarian algorithm
- Áp dụng **radial expansion penalty**: phạt chuyển động hướng vào tâm (ràng buộc vật lý xúc giác)
- Ngưỡng dịch chuyển tối đa: 60 px
- Phù hợp hơn LK khi biến dạng lớn

## Phát hiện trượt

`SlipDetector` (`src/slip_prob/slip_prob.py`) dùng thống kê vòng tròn:
- Tính **Mean Resultant Vector Length (R)** từ góc dịch chuyển
- R > 0.8 (mặc định) → slip
- Áp dụng exponential moving average (α=0.3) để làm mượt theo thời gian

## Cấu hình

Tất cả tham số trong `config/pipeline_config.yaml`. Dùng `ConfigParser` (`src/utils/config_parser.py`) để truy cập bằng dot-notation (ví dụ: `"tracking.pyrlk.win_size"`).

Các nhóm cấu hình chính: `paths`, `tracking`, `preprocessing`, `detection`, `slip_detection`, `visualization`, `camera`, `calibration`, `controls`.

## Dữ liệu

```
data/ref/    # Ảnh tham chiếu (trạng thái tĩnh)
data/img/    # Ảnh biến dạng (trạng thái tiếp xúc)
data/calib/  # Ảnh hiệu chỉnh (checkerboard)
```
