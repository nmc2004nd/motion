# Tài liệu kỹ thuật: `src/force_cnn` — End-to-end CNN Force Regressor

## Tổng quan và lý do chọn kiến trúc CNN

`force_cnn` là module ước lượng lực theo hướng **end-to-end**: đầu vào là cặp ảnh thô `(ref, frame)`, đầu ra là giá trị lực tiếp xúc dạng scalar (Newton). Mạng CNN học trực tiếp từ pixel — không cần bước trung gian phát hiện marker hay tính displacement field.

### So sánh triết lý thiết kế với hai module khác

| Module | Đầu vào model | Bước trung gian | Tham số | Ưu điểm chính |
|---|---|---|---|---|
| `force_poly` | 6 hand-crafted features | detect → LK track → extract | ~vài K | Giải thích được, nhanh |
| `force_model` (ForceNet) | `(N, 4)` displacement field | prepare.py → .npz cache | ~25 K | Dùng toàn bộ marker geometry |
| **`force_cnn`** (ForceCNN) | `(2, H, W)` ảnh thô | Không có | ~11 M (ResNet18) | Không phụ thuộc chất lượng marker detection |

**Lý do chọn CNN end-to-end:**

1. **Loại bỏ phụ thuộc vào LK tracking** — LK optical flow thất bại khi marker bị che, nhòe, hoặc độ tương phản thấp. CNN học trực tiếp từ texture/gradient của ảnh, không quan tâm đến marker cụ thể nào.

2. **Học đặc trưng tiềm ẩn** — Có thể tồn tại các pattern biến dạng không được nắm bắt bởi displacement field dạng điểm (ví dụ: biến dạng phân tán, shadow gradient). CNN có khả năng phát hiện những pattern này.

3. **Tận dụng pretrained ImageNet weights** — ResNet18 đã học bộ lọc edge/blob/texture phổ quát có thể transfer sang ảnh xúc giác thông qua kỹ thuật conv1 inflate.

4. **Kiến trúc modular** — Dễ thay backbone (SmallCNN cho debug, ResNet18 cho production), không thay đổi dataset hay training loop.

---

## Kiến trúc pipeline tổng thể

```
data/sessions/
└── <session>/
    ├── session.yaml          (zero_offset_n)
    ├── reference/ref_*.jpg   (ảnh tham chiếu session)
    └── trials/<trial>/
        ├── trial.yaml        (marker finalize)
        ├── frames.csv        (ts_mono, image_name)
        ├── force_log.csv     (ts_mono, force_n)
        └── frames/frame_*.jpg

         │
         ▼ discover_trials() + index_trial()
         │ nearest-neighbor force sync
         │ zero offset subtraction
         ▼
    list[TrialIndex] → split_trials() → train/val/test
         │
         ▼ ForceImageDataset.__getitem__()
         │ lazy-load from disk
         │ paired augmentation
         ▼
    inp: (2, H, W) + force: scalar
         │
         ▼ ForceCNN.forward()
         │ backbone (SmallCNN | ResNet18)
         │ head MLP
         ▼
    pred: scalar (N)
         │
         ▼ HuberLoss + AdamW + CosineAnnealingLR
         ▼
    best.pt  →  eval.py (scatter, per-trial breakdown)
                infer.py (single image pair → force)
```

---

## 1. `dataset.py` — Đọc dữ liệu và đồng bộ lực

### 1.1 Sơ đồ cấu trúc dữ liệu (khác với force_model)

`force_cnn` **không có** bước `prepare.py` sinh `.npz` trung gian. Toàn bộ discovery và indexing xảy ra on-the-fly trong `dataset.py`:

```
index_trial(trial_dir, session_dir)
├── Đọc session.yaml → zero_offset_n
├── Đọc force_log.csv → (force_ts[], force_vals[])
├── Đọc frames.csv → list[(ts_mono, image_name)]
├── Tìm session reference (reference/ref_*.jpg)
│   └── Fallback: frames/frame_0.jpg đầu tiên của trial
└── nearest-neighbor force sync mỗi frame
    → samples: list[(frame_path, force_n)]
```

**Tại sao lazy-load thay vì pre-cache như force_model?**
- Mỗi ảnh 240×320 float32 ≈ 300 KB vs displacement field ≈ vài KB
- Cache toàn bộ pixel vào RAM/disk cho hàng nghìn frame là không thực tế
- DataLoader với `num_workers=4` pipeline I/O song song giảm thiểu bottleneck
- Reference images (chỉ ~vài chục ảnh/session) được cache trong RAM (`_ref_cache`)

### 1.2 Force synchronization (giống force_model)

```python
def _nearest_force(ts_target, force_ts, force_vals, tolerance_s=0.1):
    idx = argmin(|force_ts - ts_target|)
    if |force_ts[idx] - ts_target| > tolerance_s: return NaN   # bỏ frame này
    return force_vals[idx]
```

Camera (~30 fps) và force gauge Imada ZTA (~20 Hz) chạy trên luồng riêng, timestamp không đồng bộ. Nearest-neighbor join với cửa sổ ±100ms.

**Ảnh hưởng `force_sync_tolerance_s`:**
- Nhỏ (<0.05s): nhiều frame bị skip → dataset nhỏ hơn nhưng label chính xác hơn
- Lớn (>0.2s): giữ được nhiều frame nhưng label noise tăng (lực thay đổi 2 N/s → sai số 0.4 N)
- 0.1s là điểm cân bằng: tối đa 0.1 N label noise với lực thay đổi ≤ 1 N/s

### 1.3 `TrialIndex` — Structure dữ liệu của một trial

```python
@dataclass(frozen=True)
class TrialIndex:
    session_id: str
    trial_id: str
    ref_path: Path              # Ảnh reference của session/trial này
    samples: list[tuple[Path, float]]  # [(frame_path, force_n), ...]
```

`TrialIndex` là metadata-only (không chứa pixel), giúp split trial-level thuần túy.

### 1.4 `ForceImageDataset.__getitem__()` — Đầu vào model

```python
def __getitem__(self, idx):
    ti, fi = self._index[idx]
    ref   = load_gray_normalized(trial.ref_path, image_size)   # (H, W) ∈ [0,1]
    frame = load_gray_normalized(frame_path, image_size)       # (H, W) ∈ [0,1]

    if augment.enabled:
        ref, frame = _apply_paired_augment(ref, frame, augment)

    inp = np.stack([ref, frame], axis=0)  # (2, H, W) float32
    return inp, force_n                    # tensor (2,H,W), scalar
```

**Đầu vào model: 2-channel image pair**
- Channel 0: `ref` — ảnh grayscale trạng thái không tiếp xúc, normalize `[0, 1]`
- Channel 1: `frame` — ảnh grayscale trạng thái đang chịu lực, normalize `[0, 1]`
- `image_size = (H, W) = (240, 320)` — resize về kích thước cố định bằng `INTER_AREA`

**Tại sao 2 channel thay vì 1 (chỉ frame)?**
Lực tương quan với **độ biến dạng** (= frame − ref), không phải với texture tuyệt đối của frame. Cung cấp cả ref và frame cho phép mạng học residual deformation.

**Tại sao grayscale, không RGB?**
Marker trắng trên nền đen — màu sắc không mang thông tin lực. Grayscale giảm input size 3× và đơn giản hóa augmentation.

### 1.5 Paired augmentation — Nguyên tắc bảo toàn vật lý

```python
def _apply_paired_augment(ref, frame, cfg):
    # 1. Flip H/V: CÙNG transform cho cả ref và frame
    if random() < cfg.flip_h_prob:
        ref  = ref[:, ::-1]
        frame = frame[:, ::-1]

    # 2. Rotation: CÙNG matrix cho cả hai
    angle = uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg)
    M = getRotationMatrix2D(center, angle, 1.0)
    ref   = warpAffine(ref, M, ...)
    frame = warpAffine(frame, M, ...)

    # 3. Brightness/contrast: CÙNG hệ số b, c cho cả hai
    b = uniform(-brightness_jitter, brightness_jitter)
    c = 1 + uniform(-contrast_jitter, contrast_jitter)
    ref   = clip((ref   - 0.5) * c + 0.5 + b, 0, 1)
    frame = clip((frame - 0.5) * c + 0.5 + b, 0, 1)
```

**Tại sao phải dùng CÙNG transform cho cả ref và frame?**

| Augmentation | Cùng transform | Khác transform |
|---|---|---|
| Geometric (flip, rotate) | ✓ Quan hệ deformation không đổi | ✗ Vị trí marker lệch nhau → giả tạo displacement |
| Brightness/contrast | ✓ Mô phỏng exposure drift camera | ✗ Tạo ra "lực giả" từ chênh lệch brightness |

**Ý nghĩa vật lý của từng augmentation:**
- **Flip H/V**: cảm biến xúc giác có đối xứng hình học — lật ngang/dọc không thay đổi lực tiếp xúc (lực là scalar)
- **Rotation ±5°**: mô phỏng sai số gắn cảm biến (mount misalignment)
- **Brightness/contrast jitter**: mô phỏng drift phơi sáng giữa ref và frame (ref chụp ban đầu, frame chụp sau vài phút — ánh sáng môi trường có thể thay đổi nhẹ)

**Tham số augmentation và ảnh hưởng:**

| Tham số | Mặc định | Ảnh hưởng khi tăng |
|---|---|---|
| `flip_h_prob` | 0.5 | Tăng đa dạng dữ liệu, không ảnh hưởng vật lý |
| `flip_v_prob` | 0.5 | Tương tự flip_h |
| `rotation_max_deg` | 5.0° | >10°: biến dạng bị méo quá nhiều, mất tính thực tế |
| `brightness_jitter` | 0.1 | >0.3: pixel clamp gây mất thông tin đầu vào |
| `contrast_jitter` | 0.1 | >0.3: tương tự brightness, làm phẳng gradient |

---

## 2. `model.py` — Kiến trúc ForceCNN

### 2.1 Cấu trúc tổng thể

```
Input: (B, 2, H, W)
         │
         ▼ backbone
         │  ├── SmallCNN → (B, 128)
         │  └── ResNet18 → (B, 512)
         │
         ▼ head MLP
         Linear(backbone.out_dim → hidden_head=128)
         ReLU
         Dropout(p=0.2)
         Linear(128 → 1)
         │
         ▼ squeeze(-1)
Output: (B,)   scalar force per sample
```

### 2.2 SmallCNN — Backbone nhẹ (~110K params)

```python
class SmallCNN(nn.Module):
    out_dim = 128
    def __init__(self, in_channels=2):
        self.features = nn.Sequential(
            _ConvBlock(2,  16, kernel=5, pool=True),   # (B,16, H/2, W/2)
            _ConvBlock(16, 32, kernel=3, pool=True),   # (B,32, H/4, W/4)
            _ConvBlock(32, 64, kernel=3, pool=True),   # (B,64, H/8, W/8)
            _ConvBlock(64,128, kernel=3, pool=False),  # (B,128,H/8, W/8)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)             # (B, 128, 1, 1)

    def forward(self, x):
        return self.gap(self.features(x)).flatten(1)   # (B, 128)
```

Mỗi `_ConvBlock` = `Conv2d(same padding) → BatchNorm2d → ReLU → [MaxPool2d(2)]`

**Global Average Pooling (GAP):** Thay vì Flatten → FC (phụ thuộc kích thước ảnh), GAP lấy mean không gian → feature vector kích thước cố định 128 bất kể H, W. Ít tham số hơn, ít overfit hơn.

**Tại sao kernel=5 ở block đầu?**
Block đầu tiên xử lý pixel thô — receptive field lớn hơn (5×5) giúp nắm bắt gradient blob marker rộng hơn so với 3×3.

**Thông số kỹ thuật SmallCNN (input 240×320):**

| Layer | Output shape | Params |
|---|---|---|
| Input | (B,2,240,320) | — |
| ConvBlock(2→16,k=5,pool) | (B,16,120,160) | 816 |
| ConvBlock(16→32,k=3,pool) | (B,32,60,80) | 4,704 |
| ConvBlock(32→64,k=3,pool) | (B,64,30,40) | 18,560 |
| ConvBlock(64→128,k=3,npool) | (B,128,30,40) | 74,048 |
| GAP | (B,128) | 0 |
| **Tổng backbone** | | **~98K** |

### 2.3 ResNet18 Backbone (~11M params)

```python
class _ResNet18Backbone(nn.Module):
    out_dim = 512
    def __init__(self, in_channels=2, pretrained=True):
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Thay conv1 từ 3-channel sang in_channels-channel
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        if pretrained:
            net.conv1.weight = _inflate_conv1_from_pretrained(old_w, in_channels)
        net.fc = nn.Identity()  # Bỏ classifier head cuối
        self.net = net

    def forward(self, x):
        return self.net(x)  # (B, 512) — sau AdaptiveAvgPool2d(1) nội bộ ResNet18
```

**ResNet18 architecture** (4 residual stages):
```
conv1 (2→64, 7×7, stride=2) → BN → ReLU → MaxPool
layer1: 2 × ResBlock(64→64)
layer2: 2 × ResBlock(64→128, stride=2)
layer3: 2 × ResBlock(128→256, stride=2)
layer4: 2 × ResBlock(256→512, stride=2)
AdaptiveAvgPool2d(1) → flatten → (B, 512)
[fc = Identity → bỏ qua]
```

**Skip connections** trong ResBlock: `out = F(x) + x` — tránh vanishing gradient trong mạng sâu.

### 2.4 Kỹ thuật conv1 inflate từ ImageNet pretrained

```python
def _inflate_conv1_from_pretrained(pretrained_conv1_weight, in_channels):
    # pretrained_conv1_weight shape: (64, 3, 7, 7)  — 64 filters, 3 RGB channels
    mean_w = pretrained_conv1_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    return mean_w.repeat(1, in_channels, 1, 1)                  # (64, 2, 7, 7)
```

**Cơ chế:**
1. ResNet18 ImageNet có conv1 weights shape `(64, 3, 7, 7)` — 64 bộ lọc, 3 kênh RGB
2. Lấy mean qua chiều RGB → `(64, 1, 7, 7)` — bộ lọc "trung bình RGB"
3. Tile ra `in_channels=2` lần → `(64, 2, 7, 7)`

**Lý do kỹ thuật này hoạt động:**
- Các bộ lọc conv1 trong ResNet18 ImageNet học edge, blob, gradient — đây là các đặc trưng **phổ quát** cho mọi loại ảnh
- Ảnh xúc giác cũng có edge (viền marker) và blob (vùng marker tròn) → các filter này directly áp dụng được
- Mean qua RGB là approximation hợp lý: trong natural images, RGB channels tương quan cao, mean không mất nhiều thông tin

**Giới hạn của kỹ thuật:**
- Trong ImageNet: 3 channels = R, G, B có cùng semantic (màu sắc)
- Trong force_cnn: channel 0 = reference (no-contact), channel 1 = deformed — hai channel có **semantic hoàn toàn khác nhau**
- Tile cùng mean_w cho cả 2 channel giả định cả hai cần cùng loại filter — đây là approximation, không phải lý tưởng

### 2.5 Head MLP

```python
self.head = nn.Sequential(
    nn.Linear(backbone.out_dim, hidden_head=128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(128, 1),
)
```

- **Linear(512→128)**: giảm chiều từ backbone feature xuống compact representation
- **ReLU**: non-linearity sau compression
- **Dropout(0.2)**: regularization trước linear cuối, giảm co-adaptation giữa neurons
- **Linear(128→1)**: scalar force output

### 2.6 So sánh hai backbone

| | SmallCNN | ResNet18 |
|---|---|---|
| Params | ~110K | ~11M |
| out_dim | 128 | 512 |
| Pretrained | Không (random init) | ImageNet (conv1 inflate) |
| Thời gian inference | ~5ms CPU | ~30ms CPU / ~2ms GPU |
| Khi nào dùng | Debug/laptop không GPU | Training production trên GPU |
| Receptive field | 46px (4 layers) | 483px (7-layer depth) |
| Skip connections | Không | Có (ResBlock) |

---

## 3. `train.py` — Vòng lặp training

### 3.1 Optimizer: AdamW (khác force_model dùng Adam)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)
```

**AdamW vs Adam:**
- Adam: weight decay áp vào gradient (không hoàn toàn đúng về mặt lý thuyết)
- AdamW: weight decay tách biệt khỏi gradient update (decoupled regularization)
- Với ResNet18 ~11M params, AdamW quan trọng hơn để tránh overfit

**`lr=3e-4` (thấp hơn force_model `1e-3`):**
- ResNet18 với pretrained weights → cần LR nhỏ để fine-tune không phá vỡ pretrained representations
- Với SmallCNN random init có thể dùng `1e-3`, nhưng config thống nhất 1 giá trị

### 3.2 CosineAnnealingLR (khác force_model không có scheduler)

```python
if cfg["train"]["scheduler"] == "cosine":
    sched = CosineAnnealingLR(optimizer, T_max=n_epochs)
```

LR tại epoch t: `lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))`

Với `lr_min=0` (default), `lr_max=3e-4`, `T_max=60`:
- Epoch 1: lr = 3e-4 (đầy)
- Epoch 30: lr ≈ 1.5e-4 (giảm một nửa)
- Epoch 60: lr ≈ 0 (gần 0)

**Tại sao force_cnn có scheduler còn force_model không?**
- force_cnn ResNet18 ~11M params cần fine-tune cẩn thận: LR cao ban đầu khám phá, LR thấp cuối hội tụ mịn
- force_model ~25K params nhỏ, early stop (patience=10) dừng trước khi cần schedule

**Ảnh hưởng `scheduler: "cosine"` vs `"none"`:**
- `"cosine"`: LR giảm mượt, thường hội tụ tốt hơn cho CNN lớn
- `"none"`: LR không đổi, val MAE plateau sớm hơn, cần patience nhỏ hơn

### 3.3 HuberLoss (giống các module khác)

```python
criterion = nn.HuberLoss(delta=1.0)  # đơn vị Newton
```

Hàm loss: `L(ŷ, y) = 0.5*(ŷ-y)² nếu |ŷ-y| ≤ δ; δ*(|ŷ-y| - δ/2) nếu |ŷ-y| > δ`

- Dưới 1N: MSE-like → gradient nhỏ, training ổn định ở vùng lực nhỏ
- Trên 1N: MAE-like → giảm ảnh hưởng của force spike outlier do label noise

### 3.4 Checkpoint lưu gì?

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "val_mae": best_val_mae,
    "config": cfg,
}, ckpt_dir / "best.pt")
```

**Lưu ý quan trọng:** `force_cnn` checkpoint **không lưu `n_max`** (vì không có padding). Nhưng cần lưu ý `image_size` và `backbone` trong config — mô hình chỉ nhận ảnh đúng kích thước đã train.

### 3.5 Tóm tắt hyperparameter và ảnh hưởng

| Tham số | Giá trị | Ảnh hưởng khi thay đổi |
|---|---|---|
| `backbone` | resnet18 | small_cnn: nhanh nhưng yếu hơn; resnet18: mạnh, cần GPU |
| `pretrained` | true | false: cần nhiều epoch hơn để học từ đầu; true: converge nhanh hơn |
| `hidden_head` | 128 | Tăng: thêm capacity nhưng overfit nếu ít data; Giảm: cổ chai |
| `dropout` | 0.2 | Tăng: regularize mạnh hơn, giảm overfit; Giảm: fit nhanh hơn nhưng dễ overfit |
| `lr` | 3e-4 | Tăng: diverge với pretrained; Giảm: slow converge |
| `weight_decay` | 1e-4 | Tăng (1e-2): regularize mạnh cho dataset nhỏ; Giảm: tự do hơn |
| `batch_size` | 32 | Tăng (64): cần RAM GPU lớn hơn (300KB×64=19MB/batch); Giảm: noisy gradient |
| `num_epochs` | 60 | + early_stop_patience=10 → thực tế dừng ở epoch ~20-40 |
| `huber_delta` | 1.0N | Tăng (2.0): tolerant hơn với outlier; Giảm (0.5): nhạy với lỗi nhỏ |
| `image_size` | [240,320] | Tăng (480,640): chi tiết hơn nhưng batch_size phải giảm; Giảm: nhanh hơn nhưng mất chi tiết |
| `num_workers` | 4 | Tăng (8): I/O nhanh hơn nếu SSD; Giảm (1): debug không multiprocessing |

---

## 4. `eval.py` — Đánh giá và trực quan hóa

### 4.1 Điểm khác biệt với force_model eval

- **Timeseries theo frame index**, không theo `ts_mono` (vì force_cnn không cache timestamp vào TrialIndex)
- Scatter plot và per-trial breakdown giống nhau

### 4.2 Các chỉ số đánh giá

| Chỉ số | Công thức | Ý nghĩa với bài toán lực |
|---|---|---|
| MAE (N) | mean(|ŷ - y|) | Sai số trung bình tuyệt đối; đơn vị Newton dễ hiểu |
| RMSE (N) | sqrt(mean((ŷ-y)²)) | Phạt nặng outlier; RMSE >> MAE → có spike lớn |
| R² | 1 - SS_res/SS_tot | 1.0 = hoàn hảo; 0 = model bằng mean; < 0 = tệ hơn mean |

---

## 5. `infer.py` — Inference đơn ảnh (unique to force_cnn)

Module `infer.py` là điểm triển khai thực tế — không có tương đương trong `force_model` hay `force_poly`:

```python
# Usage:
python -m src.force_cnn.infer --frame data/img/my_photo_2.jpg
# → "Predicted force: 3.2471 N"
```

```python
def predict_force(ckpt_path, ref_path, frame_path, config, device):
    ref   = load_gray_normalized(ref_path, image_size)   # (H, W)
    frame = load_gray_normalized(frame_path, image_size) # (H, W)
    inp   = stack([ref, frame], axis=0)[None, ...]        # (1, 2, H, W)

    model = ForceCNN(pretrained=False, ...)  # pretrained=False khi load ckpt
    model.load_state_dict(ckpt["model_state"])
    pred = model(inp)                         # scalar
```

**Lưu ý `pretrained=False` khi load checkpoint:** Weights đến từ checkpoint, không phải ImageNet. Đặt `pretrained=True` khi load ckpt sẽ tốn thêm bandwidth download ImageNet weights mà không dùng.

---

## 6. So sánh ForceCNN với ForceNet (force_model)

| Khía cạnh | ForceCNN | ForceNet (force_model) |
|---|---|---|
| **Đầu vào** | `(2,H,W)` pixel thô | `(N,4)` displacement field |
| **Biểu diễn** | Implicit (CNN học) | Explicit (marker displacement) |
| **Phụ thuộc** | Không cần LK tracking | Cần LK tracking + prepare.py |
| **Tham số** | ~11.4M (ResNet18) | ~25K (PointNet) |
| **Data loading** | Lazy disk I/O | Pre-cached RAM |
| **Pretrained** | ImageNet (conv1 inflate) | Random init |
| **Optimizer** | AdamW | Adam |
| **LR scheduler** | CosineAnnealingLR | None |
| **LR** | 3e-4 | 1e-3 |
| **Backbone** | Convolutional (spatial) | Set-based (permutation invariant) |
| **Variable N** | Không (fixed H×W) | Có (padding + mask) |
| **Infer module** | Có (infer.py) | Không |
| **Giải thích được** | Thấp (black box) | Cao (displacement field visible) |

---

## 7. Giới hạn đã biết (Known Limitations)

| Vấn đề | Nguyên nhân | Hậu quả |
|---|---|---|
| **Overfit trên dataset nhỏ** | ResNet18 11M params >> 25K samples điển hình | Cần data regularization mạnh (dropout, wd, augment) |
| **Conv1 inflate assumption** | Ref/frame có semantic khác nhau (no-contact vs contact) nhưng tile cùng filter | Pretrained weight không optimal cho 2-channel tactile input |
| **I/O bottleneck không SSD** | Lazy-load từ disk, không pre-cache | Training chậm nếu HDD; num_workers=4 giảm nhẹ |
| **Image_size downscale** | 240×320 vs full resolution | Subtle marker deformation bị mất khi resize (INTER_AREA) |
| **Label noise từ force sync** | Nearest-neighbor join ±100ms | Sai số nhãn ≤0.1N cho lực thay đổi ≤1N/s |
| **Spurious correlation** | CNN có thể học artifact (lighting, timestamp patterns) | Fail silently trên data distribution mới |
| **Không lưu image_size trong checkpoint** | Config riêng, không embed vào best.pt | Deploy cần đảm bảo config khớp với checkpoint |
| **Reference drift** | Session reference chụp đầu session, frame cuối session | Ánh sáng, nhiệt độ thay đổi → baseline shift nhẹ |

---

## 8. Câu hỏi phản biện hội đồng

### Q1: Tại sao chọn CNN end-to-end thay vì dùng displacement field như ForceNet?

**Trả lời:** CNN end-to-end giải quyết trường hợp LK tracking thất bại — khi marker bị che khuất, nhòe, hoặc độ tương phản thấp, ForceNet không có đầu vào hợp lệ nhưng ForceCNN vẫn có thể ước lượng từ pixel. Ngoài ra, CNN có thể học các pattern biến dạng tinh tế mà displacement field dạng điểm không nắm bắt được, ví dụ biến dạng phân tán giữa các marker. Đánh đổi là model lớn hơn 440 lần (~11M vs ~25K params), cần GPU, và ít interpretable hơn.

### Q2: Kỹ thuật conv1 inflate là gì và tại sao cần thiết?

**Trả lời:** ResNet18 pretrained ImageNet có conv1 nhận 3 channel RGB, nhưng input của chúng tôi là 2 channel grayscale (ref, frame). Inflate trick lấy mean filter weights qua chiều RGB từ `(64,3,7,7)` → `(64,1,7,7)` rồi tile thành `(64,2,7,7)`. Điều này transfer các bộ lọc edge/blob/gradient mà ImageNet đã học — các đặc trưng cấp thấp này phổ quát cho mọi loại ảnh, kể cả ảnh xúc giác. Thay vì random init và phải học lại từ đầu, conv1 inflate cho phép fine-tuning từ điểm khởi đầu tốt, giảm số epoch cần thiết.

### Q3: Tại sao brightness/contrast jitter phải áp CÙNG hệ số cho cả ref và frame?

**Trả lời:** Nếu áp brightness khác nhau, model sẽ thấy sự chênh lệch brightness giữa hai channel như một "tín hiệu lực" — học một quan hệ sai lầm. Trong thực tế, exposure drift của camera ảnh hưởng cả ref và frame như nhau (camera settings không thay đổi giữa 2 ảnh). Augment cùng hệ số mô phỏng đúng hiện tượng vật lý này, còn geometric augment (flip, rotate) phải đồng bộ để không tạo ra displacement giả giữa hai channel.

### Q4: Tại sao batch_size=32 thay vì 64 như ForceNet?

**Trả lời:** Mỗi sample của ForceCNN là tensor `(2, 240, 320)` float32 ≈ 600KB. Với batch_size=32: 19.2MB RAM GPU chỉ cho input tensors. Với ResNet18 feature maps và gradients, tổng GPU memory per batch ≈ 1-2GB. Trên GPU 8GB (RTX 5060 Ti), batch_size=64 có thể gây OOM khi backward pass. ForceNet dùng batch_size=64 vì `(N=50, 4)` float32 chỉ ≈ 800B — nhỏ hơn 750 lần.

### Q5: Tại sao force_cnn không có file `prepare.py` như force_model?

**Trả lời:** `force_model/prepare.py` tạo cache `.npz` để lưu displacement field đã tính toán — bước này tốn CPU nhưng chỉ chạy một lần. Với force_cnn, "preprocessing" chỉ là load ảnh và normalize — đủ nhanh để làm on-the-fly trong DataLoader. Tradeoff: không cần bước cache nhưng training phụ thuộc disk I/O speed (cần SSD hoặc num_workers cao). Nếu training chậm do I/O, có thể pre-cache ảnh resize vào RAM trước.

### Q6: Tại sao dùng AdamW thay vì Adam như ForceNet?

**Trả lời:** ResNet18 có 11M tham số, nguy cơ overfit cao trên dataset xúc giác nhỏ. AdamW tách weight decay ra khỏi adaptive gradient update (decoupled regularization), trong khi Adam lẫn lộn hai mechanism này, dẫn đến weight decay không đủ hiệu quả với adaptive LR. Với model nhỏ như ForceNet (~25K params), sự khác biệt này ít quan trọng hơn.

### Q7: ResNet18 được thiết kế cho ảnh 224×224 RGB, dùng cho ảnh 240×320 grayscale có vấn đề gì không?

**Trả lời:** Không có vấn đề kỹ thuật — ResNet18 sử dụng `AdaptiveAvgPool2d(1)` trước FC layer nên nhận bất kỳ H×W nào. Conv1 stride=2, MaxPool stride=2 làm giảm xuống 60×80 trước layer1, phù hợp về mặt feature map size. Tuy nhiên, ResNet18 được optimized cho ảnh tự nhiên 3-channel với texture phong phú — ảnh xúc giác có texture đơn giản hơn (marker tròn trên nền đen). SmallCNN nhỏ hơn có thể đủ capacity cho ảnh xúc giác; ResNet18 là "overkill" nhưng đảm bảo upper bound.

### Q8: Làm thế nào model học được lực từ 2 ảnh grayscale?

**Trả lời:** Lực tiếp xúc gây ra biến dạng vật liệu sensor → marker dịch chuyển. Sự khác biệt spatial giữa channel 0 (ref, no-contact) và channel 1 (deformed) chính là tín hiệu mà model học. CNN đầu tiên học các filter phát hiện local difference patterns (giống optical flow detection), sau đó aggregate qua không gian để ước lượng lực tổng thể. Về bản chất, model implicit học displacement field trong không gian latent của CNN thay vì explicit như ForceNet.

### Q9: Nếu reference image bị lỗi (ví dụ: chụp trong khi đang tiếp xúc), model ảnh hưởng thế nào?

**Trả lời:** Nếu reference được chụp khi sensor đang chịu lực F_baseline, model sẽ học quan hệ giữa ảnh pair và `lực - F_baseline`. Dự đoán lúc runtime sẽ bị offset cộng thêm F_baseline. Điều này giống như zero offset calibration sai trong force_model. Chiến lược reference của force_cnn: ưu tiên `session/reference/ref_*.jpg` (chụp ở trạng thái không tiếp xúc), fallback về frame đầu tiên của trial — frame đầu có thể đã có lực nhỏ nếu trial bắt đầu từ trạng thái pre-contact.

### Q10: So sánh expressiveness: ForceCNN có thực sự tốt hơn ForceNet không?

**Trả lời:** Không nhất thiết. Với dataset lớn và đa dạng, ForceCNN (11M params, end-to-end) có thể học được biểu diễn tốt hơn ForceNet (25K params, structured). Nhưng với dataset xúc giác nhỏ (~vài chục trial, vài nghìn frame), ForceNet có lợi thế vì: (1) inductive bias của PointNet phù hợp với cấu trúc marker, (2) displacement field là đặc trưng discriminative trực tiếp, (3) ít tham số → ít overfit. ForceCNN là phương án thực nghiệm để so sánh — nếu đạt MAE tương đương trên cùng dataset, điều đó xác nhận rằng displacement field là đặc trưng đủ. Nếu ForceCNN tốt hơn đáng kể, có thể có signal trong ảnh mà displacement field bỏ qua.

---

## Phụ lục: Bảng tóm tắt tất cả tham số config.yaml

```yaml
data:
  sessions_root: "data/sessions"        # Thư mục gốc chứa tất cả sessions
  force_sync_tolerance_s: 0.1           # Tolerance join timestamp (s)
  image_size: [240, 320]               # (H, W) resize target
  split:
    train: 0.7
    val: 0.15
    test: 0.15
  split_seed: 42                        # Seed cho trial-level permutation

model:
  backbone: "resnet18"                  # "resnet18" | "small_cnn"
  pretrained: true                      # ImageNet weights (chỉ resnet18)
  hidden_head: 128                      # Neurons của head Linear layer đầu
  dropout: 0.2                          # Dropout rate trước output

train:
  batch_size: 32                        # Giới hạn bởi GPU memory (ảnh lớn)
  num_epochs: 60                        # Max epochs (early stop thường dừng trước)
  lr: 3.0e-4                           # LR ban đầu (thấp hơn vì pretrained)
  weight_decay: 1.0e-4                 # L2 regularization (AdamW decoupled)
  huber_delta: 1.0                     # Transition MSE↔MAE tại 1 Newton
  early_stop_patience: 10              # Stop nếu val MAE không cải thiện
  scheduler: "cosine"                  # "cosine" | "none"
  num_workers: 4                        # Parallel disk I/O workers

augment:
  enabled: true
  flip_h_prob: 0.5                      # Xác suất lật ngang
  flip_v_prob: 0.5                      # Xác suất lật dọc
  rotation_max_deg: 5.0                # Góc quay tối đa (±5°)
  brightness_jitter: 0.1               # ±10% brightness shift (cùng cho ref+frame)
  contrast_jitter: 0.1                 # ±10% contrast scale (cùng cho ref+frame)

infer:
  default_ref: "data/ref/my_photo_1.jpg"  # Reference mặc định cho infer.py
```
