# Shape Model

Module độc lập để train classifier dự đoán `shape` của object từ frame ảnh.
Code chỉ đọc `data/sessions/...` và ghi output dưới `outputs/shape_model`.

Label nội bộ dùng 4 class cố định:

- `circle`: tròn
- `square`: vuông
- `triangle`: tam giác
- `unknown`: unknow/unknown

## Data format

Mỗi trial cần có:

- `trial.yaml` chứa field `shape`, ví dụ `shape: circle`
- `frames.csv` chứa `image_name`
- thư mục `frames/` chứa ảnh tương ứng

`data.data_roots` trong `config.yaml` đang gom 4 nhóm:

- `data/sessions/c1`: `circle`
- `data/sessions/s1`: `square`
- `data/sessions/t1`: `triangle`
- `data/sessions/1`: `unknown` qua `fallback_shape_by_session`, vì các trial này chưa có field `shape`

Mỗi entry trong `data.data_roots` có thể trỏ vào:

- `data/sessions`
- `data/sessions/c1`
- `data/sessions/c1/trials/trial_001`

## Train

```bash
python -m src.shape_model.train --config src/shape_model/config.yaml
```

Override data root:

```bash
python -m src.shape_model.train \
  --config src/shape_model/config.yaml \
  --data-root /home/nmc/ManhCuong/motion/data/sessions/c1,/home/nmc/ManhCuong/motion/data/sessions/s1
```

Checkpoint tốt nhất được lưu ở:

```text
outputs/shape_model/checkpoints/best.pt
```

## Eval

```bash
python -m src.shape_model.eval --config src/shape_model/config.yaml --split val
```

Kết quả CSV/NPZ nằm trong:

```text
outputs/shape_model/eval
```

Eval cũng lưu ảnh bảng confusion matrix để quan sát nhanh:

```text
outputs/shape_model/eval/val_confusion.png
outputs/shape_model/eval/test_confusion.png
```

Nếu chỉ muốn lưu CSV/NPZ và bỏ qua ảnh:

```bash
python -m src.shape_model.eval --config src/shape_model/config.yaml --split test --no-plots
```

## Infer

```bash
python -m src.shape_model.infer \
  --config src/shape_model/config.yaml \
  --image data/sessions/c1/trials/trial_001/frames/frame_8935.000000.jpg
```

Infer một ảnh ngẫu nhiên từ `data.data_roots` trong config:

```bash
python -m src.shape_model.infer --config src/shape_model/config.yaml --random
```

Chọn ngẫu nhiên trong test split:

```bash
python -m src.shape_model.infer --config src/shape_model/config.yaml --random --split test
```

## Lưu ý

Nếu muốn thêm nhãn tiếng Việt trực tiếp trong `trial.yaml`, các giá trị `tròn`,
`vuông`, `tam giác`, `unknow` đều được normalize về 4 class nội bộ ở trên.
