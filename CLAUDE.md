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

Hệ thống chỉ sử dụng **Pyramid Lucas-Kanade** (`src/pyr_lk/pyr_lk.py`):
- Forward tracking → backward tracking → so sánh để loại điểm trôi (forward-backward check)
- Ngưỡng FB error: 2.0 px
- **Deadzone**: dịch chuyển < 2 px bị reset về vị trí tham chiếu (lọc nhiễu đàn hồi vật liệu)
- Tham số: `win_size=(21,21)`, `max_level=3` (pyramid depth), `fb_threshold=2.0`

> Lưu ý: Thuật toán Hungarian (`src/hungarian/`) đã bị xóa khỏi codebase.

## Phát hiện trượt

`SlipDetector` (`src/slip_prob/slip_prob.py`) dùng thống kê vòng tròn:
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

Các đối tượng được tạo một lần và tái sử dụng qua các frame:
- `_clahe`: đối tượng CLAHE (tránh khởi tạo lại mỗi frame)
- `_blob_detector`: SimpleBlobDetector

## Cấu hình

Tất cả tham số trong `config/pipeline_config.yaml`. Dùng `ConfigParser` (`src/utils/config_parser.py`) để truy cập bằng dot-notation (ví dụ: `"tracking.pyrlk.win_size"`).

Các nhóm cấu hình chính: `paths`, `tracking`, `preprocessing`, `detection`, `slip_detection`, `visualization`, `camera`, `calibration`, `controls`.

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

## Tiền xử lý ảnh (`src/utils/preprocessing.py`)

1. **Ước lượng nền**: `cv2.blur` với kernel 101×101 (loại vignetting LED)
2. **Trừ nền**: `cv2.subtract(img, background)` → marker nổi rõ trên nền đồng đều
3. **Chuẩn hóa**: stretch về [0, 255]
4. **CLAHE**: tăng cường độ tương phản cục bộ (`clip_limit=2.5`, `grid=(8,8)`)

## Trực quan hóa (`src/utils/visualization.py`)

**Arrows** (`visualize_flow_arrows`):
- Mũi tên từ vị trí tham chiếu → vị trí biến dạng, scale 3×
- Xám: marker không di chuyển; vàng: vị trí gốc; đỏ–vàng: mũi tên (độ sáng ∝ độ lớn dịch chuyển)

## Dữ liệu

```
data/ref/    # Ảnh tham chiếu (trạng thái tĩnh)
data/img/    # Ảnh biến dạng (trạng thái tiếp xúc)
data/calib/  # Ảnh hiệu chỉnh (checkerboard)
```
