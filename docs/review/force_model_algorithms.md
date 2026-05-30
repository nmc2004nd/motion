# Tài liệu kỹ thuật: `src/force_model`

> **Phạm vi**: `prepare.py`, `model.py`, `dataset.py`, `train.py`, `eval.py`, `config.yaml`
> **Mục tiêu**: Dự đoán lực tiếp xúc (Newton) từ trường dịch chuyển marker xúc giác dùng kiến trúc PointNet.

---

## 1. Tổng quan kiến trúc — Pipeline đầu đến cuối

```
┌─────────────────────── OFFLINE PREPARE (prepare.py) ──────────────────────────┐
│                                                                                 │
│  data/sessions/                                                                 │
│    └── <session>/                                                               │
│         ├── reference/ref_*.jpg   ←── ảnh tham chiếu (không có lực)           │
│         ├── session.yaml          ←── zero_offset_n (calibrate lực 0 N)       │
│         └── trials/<trial>/                                                     │
│              ├── frames/frame_*.jpg  ←── ảnh dưới lực                         │
│              ├── frames.csv          ←── (ts_mono, image_name)                 │
│              └── force_log.csv       ←── (ts_mono, force_n) @ 20Hz            │
│                                                                                 │
│  Xử lý: detect markers (ref) → LK track (apply_deadzone=False)                │
│          → nearest-neighbor sync force → trừ zero_offset                       │
│          → lưu data/cache/force_model/<session>/<trial>.npz                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │ .npz cache
                                   ▼
┌─────────────────────── TRAINING (train.py + dataset.py) ──────────────────────┐
│                                                                                 │
│  discover_trials → split_trials (trial-level, seed=42)                        │
│  compute_n_max (max markers across ALL trials → padding target)                │
│                                                                                 │
│  TrialDataset: load .npz vào RAM                                               │
│    per frame: (N,4) = [x/W, y/H, dx/W, dy/H] → pad to (N_max, 4)             │
│               mask (N_max,) bool = valid LK markers                            │
│               force: scalar float32                                             │
│                                                                                 │
│  Augmentation (train only): flip H/V · rotation ±5° · noise ~N(0,0.5px)       │
│                                                                                 │
│  ForceNet forward:                                                              │
│    (B, N_max, 4)  → transpose → (B, 4, N_max)                                 │
│    → SharedMLP [4→64→128] via Conv1d(kernel=1) + BN + ReLU                    │
│    → (B, 128, N_max)                                                           │
│    → masked_max_mean_pool → (B, 256)                                           │
│    → Head MLP [256→64→ReLU→Dropout(0.1)→1] → force (B,)                      │
│                                                                                 │
│  Loss: HuberLoss(δ=1.0N)                                                       │
│  Optimizer: Adam(lr=1e-3, wd=1e-4), no LR scheduler                           │
│  Early stop: val MAE, patience=10                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │ best.pt
                                   ▼
┌─────────────────────── EVALUATION (eval.py) ───────────────────────────────────┐
│  scatter plot (GT vs predicted) · time-series overlay · per-trial breakdown    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. `prepare.py` — Thu thập và đồng bộ dữ liệu

### 2.1 Vấn đề đồng bộ hai luồng dữ liệu bất đồng bộ

Camera ghi ảnh ở ~20–30 fps với timestamp `ts_mono` riêng. Force gauge Imada ZTA cũng ghi ở ~20 Hz nhưng trên thread riêng biệt. Hai luồng **không đồng bộ** về mặt đồng hồ — cần join bằng nearest-neighbor theo timestamp.

```
Frame timeline:    |---f0---|---f1---|---f2---|---f3---|
Force timeline: |--F0--|--F1--|--F2--|--F3--|--F4--|--F5--|

f0 ← nearest F: argmin|ts_frame - ts_force|
```

**`_nearest_force(ts_target, force_ts, force_vals, tolerance_s=0.1)`**:
- Tìm `idx = argmin |force_ts - ts_target|`
- Nếu khoảng cách > `tolerance_s` → trả `nan` → frame bị **skip**
- Lý do: nếu force gauge mất kết nối trong ~100ms, frame tương ứng không có nhãn tin cậy → loại bỏ thay vì nội suy (có thể gây nhiễu nhãn)

### 2.2 Reference image strategy

```python
ref_path = _find_session_reference(session_dir)  # session/reference/ref_*.jpg
if ref_path is None:
    ref_path = first_frame_of_trial               # fallback
```

**Tại sao ưu tiên session reference?**
- Reference là ảnh chụp khi cảm biến **không tiếp xúc** (lực = 0 N)
- Dùng frame đầu tiên của trial làm reference là fallback vì frame đó có thể đã có lực nhỏ
- Một session reference dùng được cho **tất cả** trials trong session → tiết kiệm lưu trữ và đảm bảo consistency

### 2.3 `apply_deadzone=False` — Quyết định thiết kế quan trọng

```python
tracked, valid = track_markers_lk(
    ref_gray, gray, ref_pts, config=config, apply_deadzone=False
)
```

**Tại sao `apply_deadzone=False` ở bước prepare?**

Deadzone reset displacement < `min_displacement` về 0. Điều này phù hợp với tracking realtime để loại nhiễu đàn hồi, nhưng **không phù hợp** cho training data vì:

1. **Lực nhỏ cần dữ liệu chính xác**: ở lực 0.2–0.5 N, dịch chuyển marker chỉ ~0.5–1.5 px — nhỏ hơn deadzone 2.0 px. Nếu áp deadzone, mọi frame lực thấp đều có displacement = 0 → model học sai hoàn toàn vùng lực thấp
2. **Thông tin vị trí tương đối**: displacement nhỏ nhưng **nhất quán** (cùng hướng, tăng dần theo lực) là tín hiệu quan trọng để model học tương quan lực-biến dạng

### 2.4 Zero offset subtraction

```python
zero_offset = session_meta["force"]["zero_offset_n"]  # đọc từ session.yaml
force_arr = (force_raw_arr - zero_offset).astype(np.float32)
```

Force gauge Imada bị **trôi baseline** (drift): khi không tiếp xúc, đồng hồ đọc 0.1–0.3 N thay vì 0 N do nhiệt độ và trọng lực của đầu đo. `zero_offset` được đo khi không tiếp xúc và trừ ra khỏi mọi reading → đảm bảo target label = 0 N khi không tiếp xúc.

### 2.5 Cấu trúc NPZ cache

| Array | Shape | Dtype | Ý nghĩa |
|-------|-------|-------|---------|
| `ref_pts` | (N, 2) | float32 | Tọa độ pixel marker tham chiếu |
| `disp` | (T, N, 2) | float32 | Displacement = tracked - ref (px) |
| `valid` | (T, N) | bool | LK tracking valid mask |
| `force` | (T,) | float32 | Lực đã trừ zero_offset (N) |
| `force_raw` | (T,) | float32 | Lực gốc từ gauge |
| `ts_mono` | (T,) | float64 | Monotonic timestamp (giây) |
| `image_w`, `image_h` | scalar | int32 | Kích thước ảnh |
| `n_markers` | scalar | int32 | N = số marker phát hiện ở reference |
| `force_zero_offset_n` | scalar | float32 | Offset đã trừ |

---

## 3. `model.py` — ForceNet: Kiến trúc PointNet

### 3.1 Lý do chọn PointNet cho bài toán này

Bài toán dự đoán lực từ displacement field có đặc thù:
- **N marker thay đổi** giữa các trial (số marker phát hiện được khác nhau)
- **Marker bị mất** (LK tracking fail) trong từng frame — cần mask-aware
- **Permutation invariant**: thứ tự đánh số marker không có ý nghĩa vật lý
- **Tương quan không gian cục bộ**: marker gần nhau có displacement tương quan

PointNet giải quyết cả 4 đặc thù trên: xử lý từng điểm độc lập → pool → không phụ thuộc thứ tự hay số lượng cố định.

### 3.2 Per-point feature engineering

Mỗi marker được biểu diễn bởi vector 4 chiều:

```
[x_ref/W,  y_ref/H,  dx/W,  dy/H]
```

| Feature | Ý nghĩa vật lý |
|---------|---------------|
| `x_ref/W` | Vị trí ngang tham chiếu (chuẩn hóa [0,1]) |
| `y_ref/H` | Vị trí dọc tham chiếu (chuẩn hóa [0,1]) |
| `dx/W` | Độ dịch chuyển ngang (chuẩn hóa theo chiều rộng ảnh) |
| `dy/H` | Độ dịch chuyển dọc (chuẩn hóa theo chiều cao ảnh) |

**Tại sao cần vị trí `(x_ref, y_ref)`?**

Displacement của marker ở trung tâm khác với marker ở rìa khi cùng một lực tác dụng:
- Marker trung tâm: chủ yếu dịch chuyển pháp tuyến (normal press)
- Marker rìa: dịch chuyển tiếp tuyến + nghiêng

Model cần biết **vị trí** để giải thích đúng displacement — đây là thông tin context về hình học.

**Tại sao normalize theo W, H?**

`dx/W` đảm bảo displacement scale-invariant khi ảnh có kích thước khác nhau (deploy trên camera khác). Với ảnh 640×480, `dx=10px → dx/W=0.0156`, không bị scale khi dùng ảnh 1280×960.

### 3.3 `_SharedMLP` — Weight sharing via Conv1d

```python
class _SharedMLP(nn.Module):
    # Conv1d(in, out, kernel_size=1) là fully-connected áp lên mỗi điểm
    layers = [Conv1d(4, 64, 1), BN(64), ReLU, Conv1d(64, 128, 1), BN(128), ReLU]
```

**Conv1d với kernel=1 là gì?**

Input shape: `(B, C_in, N)` — B batch, C_in features, N points

`Conv1d(C_in, C_out, kernel=1)` áp dụng **cùng một Linear layer** cho mỗi trong N points:

```
[B, 4, N_max] → Conv1d(4→64, k=1) → [B, 64, N_max]
              → BN(64)             → normalize across B×N
              → ReLU               → nonlinear
              → Conv1d(64→128, k=1) → [B, 128, N_max]
```

Đây tương đương với `nn.Linear(4, 64)` áp lên mỗi điểm với cùng trọng số — **share weights** qua tất cả markers. Cách implement bằng Conv1d hiệu quả hơn loop qua từng điểm.

**BatchNorm trong per-point MLP — tại sao cần thiết?**

- BN normalize theo chiều batch: `(B, C, N)` → normalize trên `B×N` samples cho channel C
- Lợi ích: ổn định gradient khi N_max lớn (~100 markers × B=64 = 6400 samples per batch)
- Không dùng LayerNorm vì số marker N thay đổi giữa samples trong batch (padding zeros)

### 3.4 `_masked_max_mean_pool` — Pooling có nhận thức mask

```python
def _masked_max_mean_pool(feats, mask):
    # feats: (B, C, N)  mask: (B, N) bool

    # MAX POOL: invalid markers → -inf trước khi max
    feats_for_max = feats.masked_fill(~mask.unsqueeze(1), very_neg)
    max_pool = feats_for_max.max(dim=2).values   # (B, C)

    # MEAN POOL: sum valid / count valid (clamp >=1)
    sum_pool = (feats * mask.unsqueeze(1)).sum(dim=2)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean_pool = sum_pool / count

    return torch.cat([max_pool, mean_pool], dim=1)  # (B, 2C)
```

**Tại sao cần cả max VÀ mean?**

| Pooling | Bắt được gì |
|---------|-------------|
| Max pool | Marker có displacement **lớn nhất** — phát hiện điểm tiếp xúc cực đại |
| Mean pool | Displacement **trung bình** — đặc trưng cho phân phối toàn diện |

Chỉ dùng max: bỏ qua thông tin phân phối → nhạy với outlier.
Chỉ dùng mean: bỏ qua điểm cực trị → bỏ mất thông tin về vùng chịu lực cục bộ.

Concatenate cả hai (`2C = 256` channels) cho phép head MLP kết hợp cả hai nguồn thông tin.

**Xử lý sample không có valid marker nào:**

```python
no_valid = (mask.sum(dim=1) == 0).unsqueeze(1)
max_pool = max_pool.masked_fill(no_valid, 0.0)
```

Nếu toàn bộ LK tracking fail (mask all-False), max_pool sẽ = `very_neg` (-3.4e38) — gây exploding gradient. Reset về 0 → model dự đoán lực = 0 (an toàn: không có thông tin → không dự đoán được).

### 3.5 Head MLP

```python
self.head = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
)
```

- Input: global feature vector 256-d từ max+mean pool
- Output: scalar lực (N), không activation (cho phép dự đoán âm để bắt offset nhỏ)
- Dropout(0.1): regularization nhẹ ở tầng cuối — không quá mạnh để giữ khả năng fit

**Tại sao không có activation cuối?**

Lực có thể là 0 hoặc có thể âm nhẹ do zero_offset không hoàn hảo. Dùng ReLU ở output sẽ cắt gradient khi predict âm → gây bias ở vùng lực thấp.

### 3.6 Đếm tham số (mặc định config)

| Component | Shape | Params |
|-----------|-------|--------|
| SharedMLP Conv1d 4→64 | weight + bias | 4×64 + 64 = 320 |
| BN(64) | scale + shift | 128 |
| SharedMLP Conv1d 64→128 | | 64×128 + 128 = 8,320 |
| BN(128) | | 256 |
| Head Linear 256→64 | | 256×64 + 64 = 16,448 |
| Head Linear 64→1 | | 64 + 1 = 65 |
| **Tổng** | | **~25,537** |

Model rất nhẹ (~25K params) — phù hợp với dataset nhỏ (~vài nghìn frames).

---

## 4. `dataset.py` — TrialDataset

### 4.1 N_max padding — Giải quyết variable-length input

```python
n_max = compute_n_max(cache_paths)  # max(n_markers) across ALL trials

feat_pad = np.zeros((n_max, 4), dtype=np.float32)  # pad bằng zeros
mask_pad = np.zeros((n_max,), dtype=bool)           # pad = invalid
feat_pad[:N] = feat
mask_pad[:N] = valid
```

**Tại sao không dùng padding âm?**

Padding zeros: các "marker ảo" có vị trí (0,0) và displacement (0,0) — không phải vị trí vật lý trên cảm biến. Mask = False đảm bảo chúng bị loại trong pooling step, nên giá trị zero không ảnh hưởng đến kết quả.

**Edge case: N thực < n_max**

Ví dụ trial A có 90 markers, trial B có 105 markers → n_max=105. Mỗi sample của trial A có 15 rows zeros ở cuối + mask=False. SharedMLP vẫn xử lý 105 points nhưng pooling bỏ qua 15 points ảo.

### 4.2 Data augmentation

```python
class AugmentConfig:
    enabled: bool = True
    flip_h_prob: float = 0.5       # lật ngang
    flip_v_prob: float = 0.5       # lật dọc
    rotation_max_deg: float = 5.0  # xoay ±5°
    noise_std_px: float = 0.5      # nhiễu displacement
```

**Flip H/V — tại sao hợp lệ cho cảm biến xúc giác?**

Cảm biến xúc giác hình vuông/tròn có tính đối xứng cao — lực ép thẳng không phân biệt trái/phải hay trên/dưới. Flip H lật tất cả:
- Vị trí: `x → (W-1) - x` (gương ngang)
- Displacement: `dx → -dx` (đảo chiều ngang, vật lý đúng)

Flip V tương tự theo chiều dọc.

**Rotation ±5° — tại sao giới hạn nhỏ?**

Cảm biến thường được mount không hoàn toàn thẳng (sai số ±3–5° do cơ học). Augmentation rotation ±5° mô phỏng misalignment này → model robust hơn với mount thực tế. Quá 5° là không thực tế và có thể làm model học pattern sai.

**Displacement noise `N(0, 0.5px)` — mục đích**

LK tracking có sai số nhỏ ~0.5–1.0 px ngay cả khi tracking thành công. Thêm noise mô phỏng sai số LK → model không bị overfit vào displacement chính xác.

**Chú ý quan trọng**: Augmentation chỉ áp cho train split:

```python
train_ds = TrialDataset(split.train, n_max=n_max, augment=augment_cfg)
val_ds   = TrialDataset(split.val,   n_max=n_max, augment=no_aug)
test_ds  = TrialDataset(split.test,  n_max=n_max, augment=no_aug)
```

Val/test cần deterministic để so sánh công bằng.

### 4.3 Trial-level split — tránh data leakage

```
Frame-level split (SAI):        Trial-level split (ĐÚNG):
trial_1: f0 f1 f2 f3 f4        trial_1 → train
         ↑train ↑val ↑train     trial_2 → train
                                trial_3 → val
```

Các frames trong 1 trial có **tương quan thời gian mạnh**: f0 và f1 gần như giống nhau (cùng lực, cùng vùng tiếp xúc). Frame-level split vô tình đưa frame từ cùng khoảnh khắc tiếp xúc vào cả train và val → R² ảo cao, model thực tế không generalise.

---

## 5. `train.py` — Vòng lặp huấn luyện

### 5.1 Loss function: HuberLoss(δ=1.0 N)

```
L_huber(e) = {
    0.5 * e²           nếu |e| ≤ δ (MSE regime)
    δ * (|e| - δ/2)    nếu |e| > δ (MAE regime)
}
```

**Tại sao Huber thay vì MSE hay MAE?**

| Loss | Vấn đề |
|------|--------|
| MSE | Force spike từ gauge (nhiễu cơ học) → outlier → gradient lớn bất thường → training mất ổn định |
| MAE | Gradient constant = ±1 khi loss nhỏ → convergence chậm ở vùng gần hội tụ |
| Huber | Vùng nhỏ (|e|<1N): gradient ~ MSE → hội tụ mượt. Vùng lớn (|e|>1N): gradient giới hạn → outlier-robust |

δ=1.0 N tương ứng với ngưỡng chuyển đổi. Force gauge Imada có độ phân giải ~0.01 N và sai số spike ~0.5–2 N → δ=1 N là ngưỡng hợp lý: sai số < 1N được coi là gradient bình thường, > 1N là spike cần giảm ảnh hưởng.

### 5.2 Optimizer: Adam (không phải AdamW)

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-4
)
```

**So sánh với `force_poly` dùng AdamW:**

| | `force_model` | `force_poly` |
|--|---|---|
| Optimizer | Adam | AdamW |
| weight_decay | 1e-4 | 1e-2 |

Adam với weight_decay thực hiện **L2 regularization** nhúng vào gradient update (cách cổ điển). AdamW tách weight decay ra khỏi gradient → hành xử đúng hơn về lý thuyết cho adaptive optimizer, nhưng hiệu quả thực tế tương đương khi weight_decay nhỏ (1e-4).

Lý do `force_poly` cần AdamW với wd=1e-2 lớn hơn: polynomial expansion tạo ~55–220 parameters có khả năng collinear → cần regularization mạnh hơn. ForceNet dùng network biểu diễn ngầm, ít collinearity hơn → wd=1e-4 đủ.

### 5.3 Không có LR scheduler

`force_poly` dùng CosineAnnealingLR (300 epochs). `force_model` **không có** scheduler.

Lý do:
- `force_model` train 50 epochs (ngắn hơn nhiều)
- Dataset nhỏ → risk overfit nhanh → early stop thường kích hoạt trước khi cần anneal
- Model đơn giản hơn (25K params) → converge nhanh → không cần warm-down LR

### 5.4 Early stopping

```python
patience = 10  # epochs
if val_metrics["mae"] < best_val_mae - 1e-6:
    best_val_mae = val_metrics["mae"]
    epochs_since_best = 0
    torch.save(...)   # lưu best.pt
else:
    epochs_since_best += 1
    if epochs_since_best >= patience:
        break
```

Ngưỡng cải thiện `1e-6` N (gần như 0) đảm bảo bất kỳ cải thiện thực sự nào đều được tính. Patience=10 với 50 epochs tối đa: nếu sau 10 epochs liên tiếp val MAE không giảm → dừng sớm.

**Tại sao monitor val MAE (không phải val loss)?**

Huber loss không có đơn vị trực tiếp. MAE (Newton) có ý nghĩa vật lý — dễ đánh giá "sai số 0.2N có chấp nhận được không?".

### 5.5 Checkpoint lưu gì?

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "val_mae": best_val_mae,
    "n_max": n_max,
    "config": cfg,
}, ckpt_dir / "best.pt")
```

`n_max` được lưu cùng model — quan trọng cho inference: khi deploy phải dùng cùng `n_max` để padding đúng. Nếu cảm biến mới có nhiều marker hơn n_max_train → padding không đủ → kết quả sai.

---

## 6. `eval.py` — Đánh giá

### 6.1 Ba metric cốt lõi

| Metric | Công thức | Đơn vị | Ý nghĩa |
|--------|-----------|--------|---------|
| MAE | `mean(|ŷ - y|)` | Newton | Sai số tuyệt đối trung bình — dễ hiểu |
| RMSE | `sqrt(mean((ŷ-y)²))` | Newton | Nhạy với outlier hơn MAE |
| R² | `1 - SS_res/SS_tot` | dimensionless | Phần phương sai được giải thích |

R² = 1.0: hoàn hảo. R² = 0: model chỉ đoán mean. R² < 0: tệ hơn đoán mean.

### 6.2 Per-trial breakdown

```python
rows = per_trial_breakdown(preds, targets, index, chosen)
# Output: CSV với trial_path, n_frames, mae, rmse, r2
```

Breakdowns quan trọng vì overall R² có thể ổn trong khi một trial cụ thể rất tệ — ví dụ trial với lực range hẹp sẽ có R² thấp dù MAE nhỏ.

### 6.3 Time-series plot

```python
_plot_timeseries(preds, targets, index, cache_paths, out_path, trial_ci=0)
```

Plot ground truth (đen) và predicted (đỏ) theo thời gian thực (giây) trong 1 trial. Cho thấy:
- Model có theo được **shape** của tín hiệu lực không
- Có **lag** (trễ dự đoán) không
- **Phase mismatch** trong lúc lực tăng/giảm nhanh không

---

## 7. `config.yaml` — Phân tích tham số

```yaml
data:
  force_sync_tolerance_s: 0.1   # 100ms cửa sổ sync
  split: {train: 0.7, val: 0.15, test: 0.15}
  split_seed: 42

model:
  hidden_per_point: [64, 128]
  hidden_head: 64
  dropout: 0.1

train:
  batch_size: 64
  num_epochs: 50
  lr: 1.0e-3
  weight_decay: 1.0e-4
  huber_delta: 1.0
  early_stop_patience: 10

augment:
  enabled: true
  flip_h_prob: 0.5
  flip_v_prob: 0.5
  rotation_max_deg: 5.0
  noise_std_px: 0.5
```

### Ảnh hưởng của từng tham số

**`force_sync_tolerance_s: 0.1`**
- Quá nhỏ (0.01s): nhiều frames bị skip nếu force gauge có jitter timestamp → dataset nhỏ hơn nhiều
- Quá lớn (0.5s): frames bị gán force từ thời điểm khác → label noise, force thực tế ≠ force trong ảnh
- 0.1s = khoảng cách tối đa 2 samples force (ở 20Hz = 0.05s/sample) → chấp nhận được

**`hidden_per_point: [64, 128]`**
- Tăng (128, 256): nhiều tham số hơn → fit tốt hơn nếu dataset lớn, nhưng overfit nếu dataset nhỏ
- Giảm (32, 64): underfitting nếu quan hệ lực-displacement phi tuyến mạnh
- [64, 128] = ~25K params phù hợp với dataset ~vài nghìn frames

**`hidden_head: 64`**
- Head nhỏ hơn (32): có thể underfitting khi combine max+mean pool
- Head lớn hơn (128): tăng risk overfit sau global pooling

**`dropout: 0.1`**
- Dropout chỉ ở head — nhẹ, mục đích chính là tránh head overfit sau khi feature extraction đã học
- Không dùng dropout trong SharedMLP: nếu drop node giữa per-point MLP thì pooled feature bị nhiễu không ổn định

**`lr: 1.0e-3`**
- 10× lớn hơn `force_poly` (5e-3) nhưng số epoch ít hơn (50 vs 300)
- Adam + lr=1e-3 là default phổ biến, thường converge ổn trong 20–50 epochs

**`batch_size: 64`**
- Nhỏ hơn → gradient noisy hơn → regularization ngầm
- Lớn hơn → gradient ổn định nhưng cần nhiều memory và update ít hơn per epoch
- 64 frame/batch × 105 markers × 4 features = ~26K floats/batch → hoàn toàn trong RAM

**`rotation_max_deg: 5.0`**
- Quá lớn (15°): pattern vị trí marker bị distort → model học placement không đúng
- Quá nhỏ (1°): augmentation không hiệu quả vì không thêm đủ diversity
- 5° ≈ sai số mount cơ học thực tế của cảm biến

**`noise_std_px: 0.5`**
- ~ 1/2 pixel = sai số LK tracking thực tế
- Quá lớn (2px): displacement bị mask bởi noise → model khó học tương quan lực-displacement

---

## 8. So sánh ForceNet vs PolynomialRegressor (force_poly)

| Tiêu chí | ForceNet | PolynomialRegressor |
|----------|----------|---------------------|
| **Input** | (N_max, 4) per-marker raw features | 9 aggregate scalar features |
| **Kiến trúc** | PointNet shared MLP + masked pooling | Poly expansion (55 terms) + MLP |
| **Xử lý missing** | mask-aware pooling | Aggregate trước rồi model → mất thông tin vị trí |
| **Permutation invariance** | Có (by design) | Không cần (đã aggregate) |
| **Augmentation** | Có (flip, rotation, noise) | Không |
| **Số tham số** | ~25K | ~1K (linear) / ~2K (mlp_small) |
| **Khả năng giải thích** | Thấp (deep features) | Cao (với linear head: mỗi poly term có weight) |
| **LR scheduler** | Không | Cosine Annealing |
| **Optimizer** | Adam | AdamW |
| **Lý thuyết nền** | Deep learning, PointNet | Hertz contact (F~δ^1.5), polynomial regression |

---

## 9. Câu hỏi phản biện và trả lời

**Q1: Tại sao dùng PointNet cho bài toán này thay vì CNN hoặc MLP đơn giản?**

> CNN yêu cầu input dạng grid (ảnh) — marker tracking cho ra point cloud không đều. MLP đơn giản yêu cầu kích thước input cố định nhưng N marker thay đổi giữa trial (do phát hiện blob). PointNet giải quyết cả hai: xử lý bất kỳ N nào, không cần fixed size. Hơn nữa, permutation invariance phù hợp vì đánh số marker là artifact của thuật toán, không phải vật lý.

**Q2: Tại sao cần vị trí `(x_ref, y_ref)` trong input? Chỉ displacement `(dx, dy)` không đủ sao?**

> Không đủ. Cùng một displacement `(3px, 0px)` mang ý nghĩa khác nhau tùy vị trí: marker ở tâm cảm biến dịch chuyển 3px → lực pháp tuyến trực tiếp; marker ở rìa dịch chuyển 3px theo hướng hướng tâm → cũng pháp tuyến nhưng khác distribution lực. Model cần biết context spatial để map displacement → force đúng.

**Q3: Mask-aware pooling — nếu 50% marker bị LK lost thì model dự đoán thế nào?**

> Max pool: -inf của markers ảo bị loại. Mean pool: chỉ tính trung bình trên valid markers (count được clamp ≥ 1). Kết quả: global feature vector chỉ phản ánh thông tin từ markers hợp lệ. Nếu pattern spatial của markers còn lại vẫn đủ (ví dụ markers trung tâm còn valid), model vẫn dự đoán được. Khi 0 markers valid → global feature = 0 → model predict force ≈ bias ở head cuối.

**Q4: Tại sao không normalize displacement bằng StandardScaler như PolynomialRegressor?**

> Ở force_poly, polynomial expansion khuếch đại scale `d^2`: nếu `d=10px` thì `d²=100` nhưng nếu `d=100px` thì `d²=10000` — scale blow-up → cần standardize trước. ForceNet dùng neural network: BatchNorm trong shared MLP tự normalize activation ở mỗi layer → không cần pre-standardize thủ công. Tuy nhiên, normalization `dx/W` vẫn cần để các features có cùng scale trước khi vào Conv1d đầu tiên.

**Q5: Force sync tolerance 0.1s — nếu lực thay đổi nhanh thì sao?**

> Đây là tradeoff. Nếu lực thay đổi gradient `dF/dt = 5 N/s` và error sync là 0.05s → label error = 0.25 N. Với force gauge accuracy ~0.1N thì label noise này đáng kể. Giải pháp tốt hơn là dùng hardware sync (trigger), hoặc giảm tolerance và chấp nhận mất frame. Trong thiết kế hiện tại, các trial được thu thập với lực thay đổi chậm (≤ 2 N/s) → 0.1s × 2 N/s = 0.2N error — chấp nhận được.

**Q6: Tại sao augmentation flip H/V không cần điều chỉnh lực (force label vẫn giữ nguyên)?**

> Lực đo được là **scalar** (magnitude) không có hướng trong không gian ảnh — force gauge Imada đo lực dọc trục thẳng đứng (pháp tuyến bề mặt). Lật ảnh ngang hay dọc không thay đổi độ lớn của lực tác dụng. Displacement vector được transform đúng (flip dấu tương ứng), nên cặp `(displacement_augmented, force)` vẫn nhất quán vật lý.

**Q7: BatchNorm trong per-point MLP có vấn đề gì với padding zeros không?**

> Có vấn đề tinh tế: BN tính mean/var trên toàn bộ `B × N_max` entries, bao gồm cả padding zeros. Nếu một trial có nhiều padding (N_actual = 60, n_max = 105 → 45 zeros), zeros này kéo mean về phía 0 và làm nhỏ var → activations của markers thực bị scale/shift không mong muốn. Giải pháp tốt hơn là InstanceNorm hoặc masked BN, nhưng trong thực tế tỉ lệ padding thường <20% → ảnh hưởng nhỏ.

**Q8: Tại sao train.py dùng Adam nhưng force_poly/train.py dùng AdamW?**

> Sự khác nhau chính là bản chất regularization: AdamW tách weight_decay khỏi gradient update (decoupled decay), Adam truyền thống gộp chung. Với polynomial regression (force_poly), weight_decay lớn (1e-2) cần thiết để regularize 55+ polynomial coefficients collinear → AdamW decoupled đúng hơn về lý thuyết. ForceNet dùng network biểu diễn ẩn, weight_decay nhỏ (1e-4), và sự khác biệt Adam vs AdamW ở weight_decay nhỏ là không đáng kể trong thực hành.

**Q9: Nếu deployment dùng cảm biến có nhiều marker hơn n_max_train thì sao?**

> Đây là **điểm yếu quan trọng** của kiến trúc padding-based. Nếu `N_deploy > n_max_train`, input shape không khớp với model (Conv1d weights được train với n_max_train points). Giải pháp: (1) retrain với n_max mới; (2) truncate về n_max_train bằng cách chọn N markers có displacement lớn nhất; (3) dùng attention-based architecture (Transformer) không có giới hạn N. PointNet thuần (không padding) tránh vấn đề này nhưng yêu cầu collation trong DataLoader phức tạp hơn.

**Q10: Model có thể học pattern của lực lớn gây ra nhiều LK failures không?**

> Đây là challenge thú vị: khi lực rất lớn, marker dịch chuyển > tracking range → LK fail → mask = False → mean pool nhỏ hơn. Nếu pattern "nhiều markers bị mask = high force" đủ consistent trong training data, model có thể học ngầm từ `count_valid/n_max` ratio. Tuy nhiên, đây là thông tin ngầm — nếu muốn explicit, nên thêm feature `valid_ratio = mask.sum() / n_max` vào input.

---

## 10. Hạn chế đã biết

| Hạn chế | Mô tả | Giải pháp tiềm năng |
|---------|-------|-------------------|
| n_max cứng | Không inference được khi N_deploy > n_max_train | Dynamic collation / Transformer |
| BatchNorm + padding | BN bị bias bởi padding zeros | Masked BN / InstanceNorm |
| Không có LR schedule | Convergence có thể unstable ở cuối training | CosineAnnealing như force_poly |
| Label noise từ sync | ±50ms × dF/dt lực nhanh → error label | Hardware sync trigger |
| Augmentation rng global | `np.random` (global state) không reproducible theo seed | Dùng `np.random.default_rng` per-sample |

---

## 11. Luồng chạy end-to-end

```bash
# Bước 1: Chuẩn bị cache từ raw session data
python -m src.force_model.prepare \
    --sessions data/sessions \
    --output data/cache/force_model \
    --config config/pipeline_config.yaml \
    --tolerance 0.1 \
    --skip-existing

# Bước 2: Train
python -m src.force_model.train --config src/force_model/config.yaml

# Bước 3: Evaluate
python -m src.force_model.eval \
    --config src/force_model/config.yaml \
    --split test \
    --out outputs/force_model/eval
```

Checkpoint tốt nhất lưu tại `outputs/force_model/checkpoints/best.pt`. Có thể load cho inference bằng:

```python
ckpt = torch.load("outputs/force_model/checkpoints/best.pt", map_location="cpu")
model = ForceNet(**model_kwargs)
model.load_state_dict(ckpt["model_state"])
n_max = ckpt["n_max"]  # Quan trọng: dùng n_max này để padding input inference
```
