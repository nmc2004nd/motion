# Kiến trúc 3 Model Ước Tính Lực (Force Estimation)

Tài liệu này mô tả kiến trúc, luồng dữ liệu và so sánh 3 model ước tính lực từ cảm biến xúc giác.

---

## Tổng quan

| | `force_poly` | `force_model` | `force_cnn` |
|---|---|---|---|
| **Loại model** | Polynomial Regression | PointNet | CNN (ResNet18) |
| **Đầu vào** | 9 scalar features | (N, 4) marker features | 2-channel image |
| **Số tham số** | ~50–220 | ~10K–100K | ~110K / ~11M |
| **Tốc độ huấn luyện** | Rất nhanh | Trung bình | Chậm |
| **Interpretability** | Cao (đọc hệ số) | Trung bình | Thấp |
| **Yêu cầu tiền xử lý** | Cache `.npz` | Cache `.npz` | Ảnh gốc |

---

## 1. `force_poly` — Polynomial Regression

### Mô tả
Baseline đơn giản nhất. Trích xuất 9 scalar features từ trường dịch chuyển marker, mở rộng đa thức, rồi hồi quy tuyến tính ra lực. Dễ giải thích nhờ có thể đọc trực tiếp hệ số của từng đơn thức.

### Pipeline

```
Ảnh tham chiếu ──┐
                  ├─→ [Blob Detector] → ref_pts (N, 2)
                  │
Ảnh biến dạng ───┘   [Lucas-Kanade] → disp (N, 2), valid (N,)
                                           │
                                    [Feature Eng.]
                                           │
                              ┌────────────▼────────────┐
                              │  9D Feature Vector (D,)  │
                              └────────────┬────────────┘
                                           │  Standardize (mean/std)
                                           │  Polynomial Expand degree d
                                           │  → P monomials  C(9+d, d)
                                           │
                              ┌────────────▼────────────┐
                              │  Linear / MLP-Small head │
                              └────────────┬────────────┘
                                           │
                                    Force (scalar)
```

### Feature Engineering (`features.py`)

9 scalar features được tính từ displacement field của một frame:

| # | Tên | Ý nghĩa |
|---|-----|---------|
| 1 | `disp_mag_mean` | Trung bình độ lớn dịch chuyển — mức độ biến dạng tổng thể |
| 2 | `disp_mag_max` | Dịch chuyển cực đại — vùng chịu tải lớn nhất |
| 3 | `disp_mag_std` | Độ lệch chuẩn — mức độ không đồng đều |
| 4 | `dx_mean` | Dịch chuyển trung bình theo x — hướng lực chính |
| 5 | `dy_mean` | Dịch chuyển trung bình theo y — vuông góc |
| 6 | `disp_mag_sum_norm` | Tổng dịch chuyển / N_valid |
| 7 | `radial_disp_mean` | Thành phần hướng tâm: nén (−) vs giãn (+) |
| 8 | `tangential_disp_mean` | Thành phần tiếp tuyến: xoắn / trượt |
| 9 | `valid_ratio` | Tỉ lệ marker được track thành công |

Tất cả features được chuẩn hóa theo chiều rộng ảnh để bất biến tỷ lệ. Chỉ tính trên marker hợp lệ (mask-aware).

### Model (`model.py`)

```
Input (B, D=9)
    │
    ├─ Standardize: (x - feat_mean) / feat_std   [frozen after fit_scaler()]
    │
    ├─ Polynomial Expand (degree d):
    │     degree 0: [1]               (bias)
    │     degree 1: [x1, x2, ..., x9]
    │     degree 2: [x1², x1·x2, ...] → C(9+d, d) terms tổng
    │
    └─ Head:
         "linear"   → Linear(P, 1)
         "mlp_small" → Linear(P, hidden) → ReLU → Linear(hidden, 1)
                                            │
                                     Force (B,)
```

### Training

- Loss: Huber (delta cấu hình được)
- Optimizer: AdamW
- Scheduler: Cosine Annealing
- Early stopping: val MAE
- **Bước quan trọng:** `fit_scaler()` trên training set trước khi huấn luyện
- Output eval: `coefficients.csv` — hệ số từng đơn thức, sorted theo |weight|

---

## 2. `force_model` — PointNet on Displacement Field

### Mô tả
Model học sâu đầu tiên. Xử lý trực tiếp tập hợp N điểm marker (position + displacement) không có thứ tự bằng kiến trúc PointNet: per-point MLP → masked pooling → regression head.

### Pipeline

```
Cache .npz (từ prepare.py)
    │
    ├─ ref_pts  (N, 2) ─────────────────────┐
    ├─ disp     (T, N, 2)  → frame t ───────┤
    └─ valid    (T, N)     → frame t ───────┤
                                             │
                              ┌──────────────▼──────────────┐
                              │  Features per point (N, 4):  │
                              │  [x_norm, y_norm, dx_norm,   │
                              │         dy_norm]              │
                              │  Zero-pad → (N_max, 4)       │
                              └──────────────┬──────────────┘
                                             │
                         ┌───────────────────▼───────────────────┐
                         │           ForceNet                      │
                         │                                         │
                         │  (B, N_max, 4) ──Transpose──▶          │
                         │  (B, 4, N_max)                          │
                         │         │                               │
                         │  ┌──────▼──────┐                       │
                         │  │ SharedMLP   │  Conv1d layers        │
                         │  │ 4→64→128→256│  per-point, shared    │
                         │  └──────┬──────┘                       │
                         │         │ (B, 256, N_max)              │
                         │         │                               │
                         │  ┌──────▼────────────────────┐         │
                         │  │ Masked Max + Mean Pooling  │         │
                         │  │ invalid → −∞ before max   │         │
                         │  │ mean ÷ count(valid)        │         │
                         │  └──────┬────────────────────┘         │
                         │         │ (B, 512)  [cat max‖mean]     │
                         │         │                               │
                         │  ┌──────▼──────┐                       │
                         │  │  Head MLP   │  Linear→ReLU→Drop→Lin │
                         │  └──────┬──────┘                       │
                         └─────────┼───────────────────────────────┘
                                   │
                            Force (B,)
```

### Tiền xử lý: `prepare.py`

Script offline, chạy một lần trước khi training. Đọc session thô → output `.npz` cache:

```
data/sessions/<session_id>/<trial_id>/
    ├── frames/          # ảnh JPG
    ├── force_log.csv    # timestamp, force_n
    └── ref.jpg          # ảnh tham chiếu

    ──[prepare.py]──▶

data/cache/force_model/<session_id>/<trial_id>.npz
    ├── ref_pts    (N, 2)       # vị trí marker tham chiếu
    ├── disp       (T, N, 2)    # dịch chuyển tracking
    ├── valid      (T, N)       # mask LK thành công
    ├── force      (T,)         # lực sau trừ zero-offset
    ├── force_raw  (T,)         # lực thô
    ├── ts_mono    (T,)         # timestamps
    └── image_w, image_h, n_markers, trial_id, session_id
```

Bước đồng bộ: mỗi frame được ghép với bản ghi lực gần nhất trong `force_sync_tolerance_s` (mặc định 0.1s).

### Model: `ForceNet` (`model.py`)

```python
class ForceNet(nn.Module):
    # Input:  feat  (B, N_max, 4)  — [x_norm, y_norm, dx_norm, dy_norm]
    #         mask  (B, N_max)     — True = valid marker
    # Output: force (B,)

    SharedMLP:  Conv1d(4→64), Conv1d(64→128), Conv1d(128→256)
                  each: Conv1d + BatchNorm1d + ReLU

    Pooling: masked_max_mean_pool → concat → (B, 512)

    Head:    Linear(512, 256) → ReLU → Dropout → Linear(256, 1)
```

**Masked pooling:**
- Max-pool: set điểm không hợp lệ = −∞ trước khi `torch.max`
- Mean-pool: cộng features / `count(valid)` — tránh bias khi thiếu marker

---

## 3. `force_cnn` — End-to-End CNN

### Mô tả
Model end-to-end: học trực tiếp từ cặp ảnh (tham chiếu, biến dạng) mà không cần bước detection/tracking tường minh. Backbone CNN trích xuất đặc trưng ảnh, head hồi quy ra lực. Cho phép dùng pretrained ImageNet weights.

### Pipeline

```
ref.jpg  ──→ grayscale → resize (H,W) → normalize [0,1] ──┐
                                                             ├─ stack → (2, H, W)
frame.jpg ─→ grayscale → resize (H,W) → normalize [0,1] ──┘
                                             │
                           ┌─────────────────▼─────────────────┐
                           │              ForceCNN               │
                           │                                     │
                           │  Input: (B, 2, H, W)                │
                           │         │                           │
                           │  ┌──────▼──────────────────┐       │
                           │  │      Backbone           │       │
                           │  │                          │       │
                           │  │  "resnet18":             │       │
                           │  │   conv1 inflated 1ch→2ch │       │
                           │  │   (avg pretrained wts)   │       │
                           │  │   → AdaptiveAvgPool2d    │       │
                           │  │   → (B, 512)             │       │
                           │  │                          │       │
                           │  │  "small_cnn":            │       │
                           │  │   4 ConvBlocks           │       │
                           │  │   → AdaptiveAvgPool2d    │       │
                           │  │   → (B, 128)             │       │
                           │  └──────┬───────────────────┘       │
                           │         │                           │
                           │  ┌──────▼──────┐                   │
                           │  │  Head MLP   │                   │
                           │  │ Linear(out_dim)→ReLU→Drop→Lin(1)│
                           │  └──────┬──────┘                   │
                           └─────────┼───────────────────────────┘
                                     │
                              Force (B,)
```

### Backbone Options

#### `SmallCNN` (~110K params)
```
Input (B, 2, H, W)
  ConvBlock 1: Conv2d(2, 16, 3, pad=1) → BN → ReLU → MaxPool(2)
  ConvBlock 2: Conv2d(16, 32, 3, pad=1) → BN → ReLU → MaxPool(2)
  ConvBlock 3: Conv2d(32, 64, 3, pad=1) → BN → ReLU → MaxPool(2)
  ConvBlock 4: Conv2d(64, 128, 3, pad=1) → BN → ReLU
  AdaptiveAvgPool2d(1) → Flatten → (B, 128)
```

#### `ResNet18Backbone` (~11M params)
```
Input (B, 2, H, W)
  conv1: Conv2d(2, 64, 7, stride=2, pad=3)
         [pretrained weights: avg over channel dim để khởi tạo từ ImageNet]
  bn1 → ReLU → MaxPool
  layer1, layer2, layer3, layer4   [ResNet blocks chuẩn]
  AdaptiveAvgPool2d(1) → Flatten → (B, 512)
```

> Kỹ thuật khởi tạo 2-channel: lấy trọng số conv1 pretrained (64, 3, 7, 7), tính mean theo chiều channel → (64, 1, 7, 7), tile thành (64, 2, 7, 7). Giữ được feature detector từ ImageNet dù ảnh đầu vào là grayscale pair.

### Dataset (`dataset.py`)

```
data/sessions/<session_id>/<trial_id>/
    ├── frames/        # ảnh JPG
    ├── force_log.csv  # timestamp, force_n
    └── ref.jpg        # ảnh tham chiếu

    ──[ForceImageDataset]──▶

    (ref_gray, frame_gray) stacked → (2, H, W) tensor
    force_n - zero_offset          → scalar tensor
```

**Trial-level split**: toàn bộ frame của một trial nằm cùng một split (train/val/test), tránh data leakage từ các frame liên tiếp.

**Augmentation** (`AugmentConfig`):
- Geometric: flip ngang/dọc (đồng bộ ref + frame), rotation ±angle
- Photometric: brightness/contrast jitter (chỉ trên frame, không áp lên ref)
- Chỉ áp dụng cho training set

---

## So sánh chi tiết

### Luồng dữ liệu

```
Raw session data
       │
       ├──────────────────────────────────────────────────┐
       │                                                   │
       │  force_model/prepare.py                          │  force_cnn
       │  (offline, chạy 1 lần)                           │  (end-to-end)
       ▼                                                   │
  .npz cache                                              │
  [ref_pts, disp, valid, force]                           │
       │                                                   │
       ├──────────────────┐                                │
       │                  │                                │
  force_model         force_poly                    force_cnn
  (PointNet)         (Poly Reg)                    (ResNet18)
  Input: (N,4)      Input: (9,)                   Input: (2,H,W)
       │                  │                                │
       └──────────────────┴────────────────────────────────┘
                          │
                    Force scalar (N)
```

### Trade-offs

| Tiêu chí | `force_poly` | `force_model` | `force_cnn` |
|----------|:---:|:---:|:---:|
| **Yêu cầu GPU** | Không | Khuyến nghị | Bắt buộc (ResNet18) |
| **Overfitting risk** | Thấp | Trung bình | Cao |
| **Cần nhiều dữ liệu** | Ít (~100 trial) | Vừa | Nhiều (~1000+ trial) |
| **Debug / giải thích** | `coefficients.csv` | Feature importance khó | Hộp đen |
| **Thêm sensor dễ không** | Chỉ thêm feature | Thêm cột vào (N, D) | Phải thiết kế lại |
| **Inference không cần tracking** | Không | Không | **Có** |

### Checkpoint nội dung

| | `force_poly` | `force_model` | `force_cnn` |
|---|---|---|---|
| `model_state` | ✓ | ✓ | ✓ |
| `feature_set` | ✓ (feature names) | — | — |
| `n_max` | — | ✓ (padding size) | — |
| `config` | ✓ | ✓ | ✓ |
| `scaler (mean/std)` | ✓ (baked in model) | — | — |

---

## Hướng dẫn chạy

### Chuẩn bị cache (cần cho `force_model` và `force_poly`)
```bash
python -m src.force_model.prepare \
    --sessions-dir data/sessions \
    --cache-dir data/cache/force_model
```

### Huấn luyện
```bash
# Polynomial Regression
python -m src.force_poly.train --config src/force_poly/config.yaml

# PointNet
python -m src.force_model.train --config src/force_model/config.yaml

# CNN
python -m src.force_cnn.train --config src/force_cnn/config.yaml
```

### Đánh giá
```bash
python -m src.force_poly.eval  --config src/force_poly/config.yaml
python -m src.force_model.eval --config src/force_model/config.yaml
python -m src.force_cnn.eval   --config src/force_cnn/config.yaml
```

### Inference đơn ảnh
```bash
# Polynomial (cần detection + tracking)
python -m src.force_poly.infer \
    --ref data/ref/ref.jpg \
    --frame data/img/frame.jpg \
    --ckpt checkpoints/force_poly/best.pt

# CNN (không cần tracking)
python -m src.force_cnn.infer \
    --ref data/ref/ref.jpg \
    --frame data/img/frame.jpg \
    --ckpt checkpoints/force_cnn/best.pt
```
