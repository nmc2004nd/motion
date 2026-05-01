# Hướng dẫn chạy và train Force Polynomial Pipeline

Pipeline ước lượng lực tiếp xúc (N) từ displacement field marker xúc giác,
dùng polynomial regression trên 9 scalar feature.

---

## Mục lục

1. [Quy trình tổng thể](#1-quy-trình-tổng-thể)
2. [Bước 1 — Thu thập dữ liệu](#2-bước-1--thu-thập-dữ-liệu)
3. [Bước 2 — Kiểm tra trial](#3-bước-2--kiểm-tra-trial)
4. [Bước 3 — Prepare cache](#4-bước-3--prepare-cache)
5. [Bước 4 — Train](#5-bước-4--train)
6. [Bước 5 — Evaluate và Inference](#6-bước-5--evaluate-và-inference)
7. [Ý nghĩa tham số config](#7-ý-nghĩa-tham-số-config)
8. [Ý nghĩa chỉ số đánh giá](#8-ý-nghĩa-chỉ-số-đánh-giá)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

---

## 1. Quy trình tổng thể

```
Thu thập  →  Kiểm tra  →  Prepare cache  →  Train  →  Eval / Infer
(station)   (inspect)    (LK + sync)      (poly)    (metrics + ảnh)
```

Dữ liệu đi qua 2 stage chính:
- **Prepare**: đọc ảnh từ `data/sessions/`, chạy LK tracking, sync force theo
  timestamp, lưu `.npz` vào `data/cache/force_model/`.
- **Train**: đọc `.npz`, tính 9 scalar feature, expand polynomial, học hệ số
  bằng AdamW + Huber loss, lưu checkpoint tốt nhất vào
  `outputs/force_poly/checkpoints/best.pt`.

---

## 2. Bước 1 — Thu thập dữ liệu

### Cài đặt

```bash
pip install opencv-python numpy torch pyserial customtkinter pillow pyyaml
```

### Kiểm tra phần cứng

Mở `src/collection/realtime_station.py` và kiểm tra:

```python
ARDUINO_PORT = "/dev/ttyACM0"   # Arduino điều khiển stepper motor
IMADA_PORT   = "/dev/ttyACM1"   # Force gauge Imada ZTA (19200 baud)
CAMERA_ID    = 2                # Camera (thử 0, 1, 2 nếu không thấy hình)
```

Kiểm tra force gauge hoạt động độc lập trước:

```bash
python -u src/collection/collect.py
# Phải in: Lực đo được: X.XXX N
# Nếu in [raw] ...: thiết bị đang ở mode lạ, tắt/bật lại Imada
```

### Chạy station

```bash
python -u src/collection/realtime_station.py
```

### Quy trình 1 session

| # | Hành động | Ghi chú |
|---|-----------|---------|
| 1 | Nhập Session ID + Notes → **New session** | Tạo `data/sessions/<sid>/` |
| 2 | Đặt cảm biến **không tiếp xúc** → **Capture reference** | Ảnh baseline để tính displacement |
| 3 | **START RECORDING** | Bắt đầu ghi 1 trial |
| 4 | Điều khiển motor Forward / Backward | Thay đổi lực tiếp xúc |
| 5 | **STOP RECORDING** | Kết thúc trial, ghi `trial.yaml` |
| 6 | Lặp bước 3–5 nhiều lần | Mỗi lần là 1 trial độc lập |

> **Cần ít nhất 6 trial** để có split train/val/test đủ ý nghĩa thống kê.
> Khuyến nghị: 10–20 trial, vary dải lực (nhẹ → mạnh) và tốc độ motor.

---

## 3. Bước 2 — Kiểm tra trial

```bash
python -m src.collection.inspect_trial data/sessions/<sid>/trials/trial_001

# Kèm plot force + motor events theo thời gian
python -m src.collection.inspect_trial data/sessions/<sid>/trials/trial_001 --plot
```

**Kiểm tra các chỉ số:**

| Chỉ số | Giá trị tốt | Vấn đề nếu... |
|--------|-------------|--------------|
| `n_valid` trong force | = tổng số row | < tổng → force có NaN, kiểm tra Imada |
| `force min/max` | dải rộng (0–3N) | Toàn ~0 → motor không di chuyển |
| `n_dropped` (yaml) | 0 | > 0 → queue đầy, disk chậm |
| `frame duration` | ≈ `duration_s` yaml | Chênh lớn → camera bị drop frame |

---

## 4. Bước 3 — Prepare cache

Chạy LK tracking offline, sync force, lưu `.npz` mỗi trial.

```bash
python -m src.force_model.prepare \
    --sessions data/sessions \
    --output   data/cache/force_model \
    --config   config/pipeline_config.yaml
```

**Options:**

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--sessions` | (bắt buộc) | Root chứa tất cả session |
| `--output` | (bắt buộc) | Thư mục lưu cache `.npz` |
| `--config` | `config/pipeline_config.yaml` | Config detection + LK tracking |
| `--tolerance` | `0.1` | Khoảng thời gian tối đa (giây) để ghép 1 frame với 1 force sample gần nhất |
| `--skip-existing` | off | Bỏ qua trial đã có `.npz` (tăng tốc khi thêm trial mới) |

**Log mong đợi:**
```
OK <sid>/trial_001: 150 markers, 407/407 frames kept (skipped 0), force [-0.02, 2.74], valid 95.7%
OK <sid>/trial_002: 150 markers, 327/327 frames kept (skipped 0), ...
...
=== Summary: 6 trials, 2368 frames cached ===
  avg LK valid coverage: 95.6%
```

`skipped > 0` nghĩa là frame không tìm được force sample trong vòng `--tolerance` giây —
thường do force sampling quá chậm. Nếu skipped nhiều, giảm `--tolerance` hoặc
tăng tốc độ đọc Imada.

---

## 5. Bước 4 — Train

### Config train

File: `src/force_poly/config.yaml` — xem chi tiết ý nghĩa ở [Mục 7](#7-ý-nghĩa-tham-số-config).

### Chạy train

```bash
python -m src.force_poly.train --config src/force_poly/config.yaml
```

Override không cần sửa file:

```bash
# Đổi số epoch
python -m src.force_poly.train --config src/force_poly/config.yaml --epochs 100

# Dùng cache dir khác
python -m src.force_poly.train --config src/force_poly/config.yaml \
    --cache-dir data/cache/force_model_v2
```

### Đọc log trong khi train

```
Ep  64 | train loss 0.0029 mae 0.0563 r2 +0.992 | val loss 0.0051 mae 0.0856 r2 +0.988
```

- **train loss / val loss**: Huber loss — dùng để so sánh xu hướng, không phải đơn vị Newton
- **mae**: Mean Absolute Error (Newton) — đơn vị thực, dễ hiểu nhất
- **r2**: R² coefficient — càng gần 1.0 càng tốt

Model tốt khi: `val mae` giảm cùng chiều `train mae`, không diverge.

### Output sau train

```
outputs/force_poly/
├── checkpoints/
│   └── best.pt          # checkpoint epoch có val MAE thấp nhất
└── logs/
    ├── history.json     # toàn bộ loss/metrics từng epoch
    └── test_predictions.npz  # preds + targets trên test split (nếu có)
```

---

## 6. Bước 5 — Evaluate và Inference

### Evaluate trên split

```bash
# Đánh giá trên val split
python -m src.force_poly.eval --config src/force_poly/config.yaml --split val

# Đánh giá trên test split (cần >= 4 trial)
python -m src.force_poly.eval --config src/force_poly/config.yaml --split test

# Không vẽ plot
python -m src.force_poly.eval --config src/force_poly/config.yaml --split val --no-plots

# Lưu kết quả sang thư mục khác
python -m src.force_poly.eval --config src/force_poly/config.yaml \
    --split val --out outputs/force_poly/eval_run2
```

Output: scatter plot (pred vs ground truth), time-series plot, `eval_metrics.csv`.

### Inference ảnh đơn

```bash
python -m src.force_poly.infer \
    --ref   data/sessions/<sid>/reference/ref_<ts>.jpg \
    --frame data/sessions/<sid>/trials/trial_001/frames/frame_<ts>.jpg
```

Full options:

```bash
python -m src.force_poly.infer \
    --config          src/force_poly/config.yaml \
    --pipeline-config config/pipeline_config.yaml \
    --ref             data/sessions/<sid>/reference/ref_<ts>.jpg \
    --frame           data/sessions/<sid>/trials/trial_001/frames/frame_<ts>.jpg \
    --ckpt            outputs/force_poly/checkpoints/best.pt
```

Output: `Predicted force: X.XXXX N`

---

## 7. Ý nghĩa tham số config

File: `src/force_poly/config.yaml`

```yaml
data:
  cache_dir: "data/cache/force_model"
  split:
    train: 0.7
    val:   0.15
    test:  0.15
  split_seed: 42
```

| Tham số | Ý nghĩa | Ghi chú |
|---------|---------|---------|
| `cache_dir` | Thư mục chứa `.npz` cache từ `prepare` | Phải chạy prepare trước |
| `split.train` | Tỉ lệ số trial dùng để train | Split theo **trial**, không phải frame — tránh data leakage |
| `split.val` | Tỉ lệ trial dùng để chọn checkpoint tốt nhất | Early stop dựa vào val MAE |
| `split.test` | Tỉ lệ trial dùng để báo cáo kết quả cuối | Không bao giờ dùng để chọn model |
| `split_seed` | Seed random cho việc shuffle trial trước khi split | Đổi seed để kiểm tra stability |

> **Lưu ý split**: 6 trial → train=4, val=1, test=1. Kết quả phụ thuộc vào
> trial nào vào val/test. Nếu kết quả biến động nhiều khi đổi `split_seed`,
> cần thu thêm trial để dataset ổn định hơn.

---

```yaml
model:
  feature_set: "v1"
  degree: 2
  head: "linear"
  hidden_head: 16
  dropout: 0.0
```

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `feature_set` | Bộ scalar feature trích xuất từ displacement field. `"v1"` gồm 9 feature: mean/max/std biến dạng, hướng trung bình (dx, dy), tổng biến dạng, radial/tangential component, tỉ lệ marker valid | Hiện chỉ có `"v1"` |
| `degree` | Bậc đa thức. Số term = C(9+d, d): d=1→10, d=2→55, d=3→220 | **Dùng `2`**. Degree 1 quá đơn giản (bỏ lỡ tương tác feature). Degree 3 dễ overfit với dataset nhỏ |
| `head` | Kiến trúc sau polynomial expansion. `"linear"`: một lớp linear thuần (= polynomial regression cổ điển). `"mlp_small"`: MLP nhỏ sau expansion — phức tạp hơn, cần nhiều data hơn | **Dùng `"linear"`** trước. Thử `"mlp_small"` khi có ≥ 20 trial |
| `hidden_head` | Số neuron ẩn của MLP (chỉ dùng khi `head: "mlp_small"`) | 16–64 |
| `dropout` | Dropout rate cho MLP head (chỉ dùng khi `head: "mlp_small"`) | 0.0–0.3 |

**Số term (tham số) theo degree:**

| degree | n_terms | Cần tối thiểu |
|--------|---------|---------------|
| 1 | 10 | ~50 frame train |
| 2 | 55 | ~300 frame train ← **khuyến nghị** |
| 3 | 220 | ~1500 frame train |

---

```yaml
train:
  batch_size: 64
  num_epochs: 300
  lr: 5.0e-3
  weight_decay: 1.0e-2
  huber_delta: 1.0
  early_stop_patience: 40
  num_workers: 0
  device: "auto"
  checkpoint_dir: "outputs/force_poly/checkpoints"
  log_dir: "outputs/force_poly/logs"
```

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `batch_size` | Số frame mỗi batch gradient. Nhỏ hơn → gradient noisier nhưng regularize tự nhiên hơn. Lớn hơn → training ổn định hơn nhưng cần nhiều RAM | 32–128 với dataset nhỏ (<2000 frame). 256+ khi có nhiều data |
| `num_epochs` | Số epoch tối đa. Thường dừng sớm vì `early_stop_patience` | 200–500 đủ với `early_stop_patience` = 40 |
| `lr` (learning rate) | Tốc độ học. Quá lớn → loss dao động không hội tụ. Quá nhỏ → hội tụ chậm | `5e-3` cho degree=2. Giảm xuống `1e-3` nếu loss dao động |
| `weight_decay` | L2 regularization — phạt hệ số lớn để tránh overfit. Đặc biệt quan trọng với polynomial bậc cao vì term bậc cao dễ blow-up | `1e-2` cho degree=2. Tăng lên `5e-2` nếu vẫn overfit |
| `huber_delta` | Ngưỡng chuyển đổi của Huber loss (Newton). Với error < delta: dùng MSE (phạt nặng). Với error > delta: dùng MAE (ít phạt hơn, robust với outlier) | `1.0` N phù hợp với dải lực 0–3N. Giảm xuống `0.5` nếu có nhiều outlier |
| `early_stop_patience` | Dừng train sau bao nhiêu epoch val MAE không cải thiện | 30–50. Tăng nếu muốn train lâu hơn để tìm minimum |
| `num_workers` | Số worker DataLoader. Dataset nhỏ đã load hết vào RAM nên để `0` | Luôn để `0` |
| `device` | `"auto"`: tự chọn CUDA nếu có, ngược lại CPU. Hoặc chỉ định `"cpu"`, `"cuda"`, `"cuda:0"` | Để `"auto"` |
| `checkpoint_dir` | Thư mục lưu `best.pt` | — |
| `log_dir` | Thư mục lưu `history.json` và `test_predictions.npz` | — |

---

## 8. Ý nghĩa chỉ số đánh giá

### MAE — Mean Absolute Error (đơn vị: Newton)

```
MAE = mean(|pred - target|)
```

**Ý nghĩa**: sai số trung bình tuyệt đối giữa lực dự đoán và lực thực đo,
tính bằng Newton. Đây là chỉ số trực quan nhất — **đọc ngay được sai số
thực tế**.

| MAE | Đánh giá (với dải lực 0–3N) |
|-----|---------------------------|
| < 0.1 N | Rất tốt (~3% sai số tương đối) |
| 0.1–0.2 N | Tốt |
| 0.2–0.5 N | Chấp nhận được |
| > 0.5 N | Cần cải thiện |

MAE robust hơn RMSE với outlier — không bị 1 frame tệ làm kéo số lên cao.

---

### RMSE — Root Mean Square Error (đơn vị: Newton)

```
RMSE = sqrt(mean((pred - target)²))
```

**Ý nghĩa**: tương tự MAE nhưng phạt nặng hơn các sai số lớn (do bình
phương). Nếu `RMSE >> MAE` tức là có một số frame bị sai lớn (outlier).

| Quan hệ | Diễn giải |
|---------|----------|
| RMSE ≈ MAE | Sai số đồng đều, không có outlier |
| RMSE > 1.5×MAE | Có outlier — kiểm tra frame bị tracking kém |

---

### R² — Coefficient of Determination (không đơn vị, từ −∞ đến 1.0)

```
R² = 1 - SS_res / SS_tot
   = 1 - sum((pred - target)²) / sum((target - mean_target)²)
```

**Ý nghĩa**: phần trăm variance của lực mà model giải thích được.

| R² | Diễn giải |
|----|----------|
| 1.0 | Dự đoán hoàn hảo |
| 0.99 | Model giải thích 99% variance — rất tốt |
| 0.95 | Tốt |
| 0.80 | Chấp nhận được |
| 0.0 | Model chỉ dự đoán bằng mean — vô nghĩa |
| < 0 | Model tệ hơn cả dự đoán mean — overfit hoặc distribution mismatch |

> **Lưu ý**: R² cao trên val chưa chắc nghĩa là tốt nếu val chỉ có 1 trial.
> Tin vào **test R²** hơn vì test trial chưa được dùng trong quá trình chọn model.

---

### Loss (Huber Loss) — chỉ dùng để theo dõi xu hướng

```
Huber(e) = 0.5 * e²             nếu |e| ≤ delta
         = delta * (|e| - 0.5 * delta)  nếu |e| > delta
```

**Ý nghĩa**: không có đơn vị trực tiếp. Chỉ dùng để xem training có hội
tụ không — phải giảm đều qua các epoch. Không so sánh loss giữa các config
khác nhau.

---

### Đọc log training — dấu hiệu tốt và xấu

**Tốt:**
```
Ep  10 | train mae 0.12 r2 +0.96 | val mae 0.14 r2 +0.95   ← val gần train
Ep  50 | train mae 0.07 r2 +0.99 | val mae 0.08 r2 +0.98   ← cả hai đều tốt
```

**Overfit** (degree quá cao, weight_decay quá nhỏ):
```
Ep  10 | train mae 0.05 r2 +0.99 | val mae 0.80 r2 -5.2    ← val diverge
Ep  20 | train mae 0.02 r2 +1.00 | val mae 1.50 r2 -20.0
```
→ Giảm `degree`, tăng `weight_decay`.

**Underfitting** (degree quá thấp, lr quá nhỏ):
```
Ep 200 | train mae 0.50 r2 +0.60 | val mae 0.52 r2 +0.58   ← cả hai đều kém
```
→ Tăng `degree` hoặc tăng `lr`.

**Learning rate quá cao:**
```
Ep   1 | train loss 1.23 | val loss 0.45
Ep   2 | train loss 3.45 | val loss 5.67   ← loss tăng vọt
Ep   3 | train loss nan  | ...
```
→ Giảm `lr` xuống 10×.

---

## 9. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách fix |
|-----|-------------|----------|
| `cache_dir không tồn tại` | Chưa chạy prepare | Chạy `force_model.prepare` trước |
| `Cần >=1 trial cho train và >=1 cho val` | Quá ít trial | Thu thêm (cần ≥ 2, tốt nhất ≥ 6) |
| `detect 0 markers ở reference` | Ảnh tối / sai config | Chụp lại reference, kiểm tra `detection` config |
| `không sync được frame nào với force log` | Force toàn NaN | Chạy `collect.py` debug; kiểm tra Imada |
| `Không có test split` | Ít hơn 4 trial | Chỉ là cảnh báo — bình thường với ≤ 3 trial |
| Force GUI hiện `---` | Imada mất kết nối | Tắt/bật lại Imada, kiểm tra port |
| Val MAE tốt nhưng Test MAE tệ | Quá ít trial, distribution lệch | Thu thêm trial đa dạng lực |
| Loss = NaN từ epoch đầu | `lr` quá lớn | Giảm `lr` xuống `1e-3` |
| skipped nhiều frame trong prepare | Force sampling chậm | Đã fix: `read_until(b"\r")` + timeout 0.1s |

---

## Cấu trúc thư mục sau khi chạy đầy đủ

```
data/
├── sessions/
│   └── <sid>/
│       ├── session.yaml           # metadata session (port, camera, zero offset)
│       ├── reference/ref_<ts>.jpg # ảnh baseline
│       └── trials/
│           ├── trial_001/
│           │   ├── frames/        # ảnh JPEG từng frame
│           │   ├── frames.csv     # ts_mono, ts_wall, image_name
│           │   ├── force_log.csv  # ts_mono, force_n (Newton)
│           │   ├── motor_log.csv  # ts_mono, direction, payload
│           │   └── trial.yaml     # metadata trial (duration, n_frames, tags)
│           └── trial_002/ ...
└── cache/
    └── force_model/
        └── <sid>/
            └── trial_001.npz      # ref_pts, disp, valid, force, ts_mono

outputs/
└── force_poly/
    ├── checkpoints/best.pt        # model tốt nhất theo val MAE
    └── logs/
        ├── history.json           # loss + metrics mỗi epoch
        └── test_predictions.npz  # preds, targets trên test split
```
