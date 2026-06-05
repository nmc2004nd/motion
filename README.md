# Motion

Motion là project xử lý ảnh tactile marker cho cảm biến mềm: phát hiện marker, theo dõi dịch chuyển bằng Lucas-Kanade, phát hiện slip realtime và huấn luyện một số model học máy để dự đoán force/shape từ dữ liệu thu thập.

## Chức năng chính

- Batch pipeline cho một cặp ảnh reference/deformed.
- Realtime tracking marker qua camera OpenCV.
- Slip detection dựa trên MRVL decomposition.
- Thu thập dữ liệu session/trial kèm force log.
- Huấn luyện và đánh giá force regression bằng PointNet-like model, polynomial regressor và CNN.
- Huấn luyện shape classifier từ frame ảnh.
- Tài liệu thiết kế và trang review tĩnh trong `docs/` và `review/`.

## Cấu trúc thư mục

```text
config/                 Cấu hình pipeline tactile marker/slip
data/                   Dữ liệu ảnh, session, cache mẫu nếu được track trong repo
docs/                   Ghi chú kiến trúc, thuật toán, kết quả
outputs/                Kết quả chạy, checkpoint, plot, video
review/                 Trang HTML review tài liệu/thuật toán
src/
  collection/           Thu thập dữ liệu và công cụ inspect/log
  common/               Camera, keyboard, overlay, FPS/timing helpers
  config/               Load và validate YAML config
  core/                 Calibration, preprocessing, detection, tracking, visualization
  force_model/          Force regressor trên displacement marker
  force_poly/           Polynomial force regressor
  force_cnn/            CNN force regressor end-to-end từ ảnh
  pipelines/            Batch và realtime pipelines
  shape_model/          Shape classifier
  slip/                 Slip detectors
  test/                 Script test/visualize thủ công
```

## Yêu cầu môi trường

- Khuyến dùng Python 3.10+.
- Camera USB/OpenCV nếu chạy realtime.
- Arduino/load-cell serial nếu dùng phần thu thập force trong `src/collection`.
- GPU là tùy chọn cho các model PyTorch.

`requirements.txt` hiện chưa cài `torch` và `torchvision` vì hai dòng này đang bị comment. Nếu train/eval các model học máy, cài PyTorch riêng theo CUDA/CPU của máy:

```bash
pip install torch torchvision
```

## Cài đặt

```bash
git clone <repo-url>
cd motion

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu môi trường bị giới hạn quyền ghi cache Matplotlib, đặt thêm:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## Cấu hình

Pipeline chính dùng file:

```text
config/pipeline_config.yaml
```

File này là single source of truth và được validate nghiêm ngặt trong `src/config/schema.py`. Thiếu key bắt buộc sẽ dừng chương trình ngay, thay vì dùng giá trị mặc định ngầm.

Một số path quan trọng:

- `paths.ref_image`: ảnh tactile ở trạng thái reference.
- `paths.deformed_image`: ảnh tactile sau biến dạng.
- `paths.calib_file`: file calibration `.npz` nếu bật `pipeline.use_calibration: true`.
- `paths.output_dir`: thư mục ghi kết quả.
- `camera.device_id`: id camera OpenCV, thường là `0`.

Các model có config riêng:

- `src/force_model/config.yaml`
- `src/force_poly/config.yaml`
- `src/force_cnn/config.yaml`
- `src/shape_model/config.yaml`

## Chạy batch pipeline

Chạy xử lý một cặp ảnh reference/deformed theo config:

```bash
python -m src.main --config-path config/pipeline_config.yaml
```

Output mặc định được ghi vào:

```text
outputs/LK/
```

Nếu `pipeline.use_calibration: true`, cần có file calibration tại `paths.calib_file`.

## Chạy realtime

Tracking marker realtime:

```bash
python -m src.pipelines.realtime_tracking --config-path config/pipeline_config.yaml
```

Slip detection:

```bash
python -m src.pipelines.realtime_slip_v1 --config-path config/pipeline_config.yaml
```


Phím điều khiển mặc định:

- `r`: chụp frame hiện tại làm reference.
- `c`: xóa reference.
- `q`: thoát.

## Chuẩn bị dữ liệu force model

Tạo cache `.npz` từ các session đã thu thập:

```bash
python -m src.force_model.prepare \
  --sessions data/sessions \
  --output data/cache/force_model \
  --config config/pipeline_config.yaml \
  --tolerance 0.1 \
  --skip-existing
```

Một session/trial cần có cấu trúc dữ liệu gồm `reference/`, `trials/*/frames/`, `frames.csv` và `force_log.csv`.

## Huấn luyện và đánh giá model force

Force model trên marker displacement:

```bash
python -m src.force_model.train --config src/force_model/config.yaml
python -m src.force_model.eval --config src/force_model/config.yaml --split test
python -m src.force_model.eval_session --session data/cache/force_model/1
```

Polynomial force regressor:

```bash
python -m src.force_poly.train --config src/force_poly/config.yaml
python -m src.force_poly.eval --config src/force_poly/config.yaml --split test
python -m src.force_poly.infer --frame data/img/my_photo_2.jpg --ref data/ref/my_photo_1.jpg
```

CNN force regressor:

```bash
python -m src.force_cnn.train --config src/force_cnn/config.yaml
python -m src.force_cnn.eval --config src/force_cnn/config.yaml --split test
python -m src.force_cnn.infer --frame data/img/my_photo_2.jpg --ref data/ref/my_photo_1.jpg
```

## Huấn luyện và đánh giá shape classifier

```bash
python -m src.shape_model.train --config src/shape_model/config.yaml
python -m src.shape_model.eval --config src/shape_model/config.yaml --split test
python -m src.shape_model.infer --image <path-to-frame.jpg>
```

Commit và push:

```bash
git add README.md
git commit -m "Add project README"
git push
```
