# Kết Quả Huấn Luyện — Force Estimation Models

Ngày: 2026-05-05  
Dataset: 26 trials, split theo trial (seed=42): **train=18 / val=4 / test=4**  
Frames: train=8 642, val=1 658, test=1 728  
Thiết bị: CPU

---

## Tóm tắt so sánh

| Chỉ số | `force_poly` | `force_model` | Chênh lệch |
|---|---|---|---|
| Kiến trúc | Poly(d=2) + MLP head | PointNet (ForceNet) | — |
| Số tham số | **913** | 25 537 | ×28 |
| Thời gian train | **14.7 s** | 148.8 s | ×10 |
| Best val MAE (epoch) | **0.0361** (ep 58) | 0.0543 (ep 39) | +50% |
| **Test MAE** | **0.0472 N** | 0.0675 N | +43% |
| **Test RMSE** | **0.0680 N** | 0.0966 N | +42% |
| **Test R²** | **+0.996** | +0.993 | — |
| Epochs chạy | 98 (patience=40) | 49 (patience=10) | — |

> **Kết luận:** `force_poly` vượt trội `force_model` trên mọi chỉ số với ít tham số hơn ×28 và huấn luyện nhanh hơn ×10. Nguyên nhân có thể do 9 scalar features v1 đã encode đủ thông tin cần thiết, khiến kiến trúc PointNet dư thừa ở tập dữ liệu hiện tại.

---

## 1. `force_poly` — Polynomial Regression + MLP Head

### Cấu hình model

| Tham số | Giá trị |
|---|---|
| Feature set | v1 (9 scalar features) |
| Polynomial degree | 2 |
| Số monomial (n_terms) | 55 → C(9+2, 2) = 55 |
| Head | `mlp_small` (Linear→ReLU→Linear) |
| Hidden head | 16 |
| Dropout | 0.0 |
| Tổng params | **913** |
| Optimizer | AdamW, lr=5e-3, wd=1e-2 |
| Loss | Huber (δ=1.0) |
| Early stop patience | 40 |

### Đặc trưng đầu vào (v1, 9 features)

| # | Tên | Ý nghĩa |
|---|---|---|
| 0 | `disp_mag_mean` | Biến dạng trung bình trên các marker hợp lệ |
| 1 | `disp_mag_max` | Peak biến dạng — marker chịu lực lớn nhất |
| 2 | `disp_mag_std` | Độ phân tán biến dạng giữa các marker |
| 3 | `dx_mean` | Hướng tải trung bình theo trục x |
| 4 | `dy_mean` | Hướng tải trung bình theo trục y |
| 5 | `disp_mag_sum_norm` | Tổng biến dạng / N_valid |
| 6 | `radial_disp_mean` | Thành phần nén/giãn hướng kính từ tâm |
| 7 | `tangential_disp_mean` | Thành phần shear/xoắn tiếp tuyến |
| 8 | `valid_ratio` | Tỷ lệ marker còn tracking (độ phủ) |

Tất cả features được normalize theo `image_w` để scale-invariant với độ phân giải.

### Quá trình training

Convergence nhanh và ổn định. Epoch đầu đã đạt val R²=+0.944, từ epoch 6 trở đi
ổn định ở vùng val R²≥0.995. Một vài spike val loss nhỏ (ep 18, 47, 84) không
ảnh hưởng trajectory chung. Early stop kích hoạt tại epoch 98 (40 epoch không
cải thiện từ best tại ep 58).

```
Ep   1 | train R² +0.897 | val R² +0.944
Ep   6 | train R² +0.995 | val R² +0.996
Ep  29 | train R² +0.997 | val R² +0.998   ← val MAE 0.0389
Ep  58 | train R² +0.997 | val R² +0.998   ← BEST val MAE 0.0361  ✓
Ep  98 | early stop
```

### Kết quả test

**Tổng thể:** MAE=0.0472 N, RMSE=0.0680 N, R²=+0.996

| Trial | Frames | MAE (N) | RMSE (N) | R² |
|---|---|---|---|---|
| trial_018 | 595 | 0.0349 | 0.0486 | +0.998 |
| trial_008 | 470 | 0.0361 | 0.0529 | +0.998 |
| trial_002 | 327 | 0.0628 | 0.0791 | +0.993 |
| trial_003 | 336 | 0.0690 | 0.0981 | +0.994 |

Trial 018 và 008 đạt MAE <0.037 N — rất chính xác. Trial 002 và 003 kém hơn
~2× nhưng vẫn R²>0.993, cho thấy hai trial này có phân bố lực hoặc điều kiện
tiếp xúc khác biệt so với train set.

---

## 2. `force_model` — ForceNet (PointNet-style)

### Cấu hình model

| Tham số | Giá trị |
|---|---|
| Đầu vào mỗi marker | 4 features: `[x_ref_norm, y_ref_norm, dx_norm, dy_norm]` |
| N_max markers | 150 |
| Shared MLP | Conv1d: 4 → 64 → 128 (+ BN + ReLU) |
| Pooling | Mask-aware max-pool + mean-pool → concat (256-dim) |
| Head | Linear(256→64) → ReLU → Dropout(0.1) → Linear(64→1) |
| Tổng params | **25 537** |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Loss | Huber (δ=1.0) |
| Early stop patience | 10 |
| Augmentation | Flip H/V (p=0.5), rotate ±5°, noise 0.5 px |

### Quá trình training

Hội tụ chậm hơn và không ổn định hơn so với force_poly. Từ epoch 1 đến 9 loss
giảm nhanh (R² từ +0.666 lên +0.983), sau đó dao động mạnh giữa các epoch
(ep 18: val R²=+0.926; ep 29: val R²=+0.992). Early stop kích hoạt tại epoch 49
(10 epoch không cải thiện từ best tại ep 39).

```
Ep   1 | train R² +0.666 | val R² +0.633
Ep   9 | train R² +0.952 | val R² +0.983
Ep  29 | train R² +0.973 | val R² +0.992   ← val MAE 0.0543 ✓
Ep  39 | train R² +0.978 | val R² +0.995   ← BEST val MAE 0.0543 ✓
Ep  49 | early stop
```

*(val MAE best epoch 29 và 39 trùng giá trị 0.0543 — epoch 39 được chọn vì checkpoint mới hơn)*

### Kết quả test

**Tổng thể:** MAE=0.0675 N, RMSE=0.0966 N, R²=+0.993

| Trial | Frames | MAE (N) | RMSE (N) | R² |
|---|---|---|---|---|
| trial_018 | 595 | 0.0549 | 0.0735 | +0.996 |
| trial_002 | 327 | 0.0630 | 0.0761 | +0.993 |
| trial_008 | 470 | 0.0558 | 0.0864 | +0.994 |
| trial_003 | 336 | **0.1108** | **0.1494** | +0.987 |

Trial 003 là điểm yếu rõ rệt — MAE gần gấp đôi các trial còn lại. Trong khi
force_poly xử lý trial này với MAE=0.069 N (R²=+0.994), ForceNet bị degraded
đáng kể, gợi ý model chưa generalize tốt trên điều kiện tiếp xúc đặc thù của
trial này.

---

## 3. Phân tích so sánh

### Tại sao force_poly thắng?

1. **Feature engineering đủ mạnh.** 9 scalar features v1 (đặc biệt `disp_mag_mean`,
   `radial_disp_mean`, `valid_ratio`) đã capture hầu hết phương sai của lực.
   ForceNet phải học các aggregation tương đương từ raw per-marker data — khó
   hơn với dataset ~8K frame.

2. **Regularization phù hợp.** Polynomial degree=2 là inductive bias tốt cho
   quan hệ lực–biến dạng (gần tuyến tính ở vùng đàn hồi). L2 weight decay=1e-2
   kiểm soát bậc cao hiệu quả.

3. **Dataset nhỏ bất lợi cho PointNet.** ForceNet có 25 537 tham số trên 8 642
   frame train (~3 frame/param), dễ bị overfit. force_poly có tỷ lệ ~9.5 frame/param
   — thoải mái hơn.

4. **Augmentation không đủ.** ForceNet dùng flip + rotate + noise, nhưng val loss
   vẫn spike mạnh (ep 18, 41, 47, 49), cho thấy variance cao giữa các epoch.

### Độ nhạy theo trial

```
Per-trial MAE so sánh:
                 force_poly   force_model   delta
trial_018           0.035        0.055      +57%
trial_008           0.036        0.056      +55%
trial_002           0.063        0.063       +0%  ← tương đương
trial_003           0.069        0.111      +61%  ← gap lớn nhất
```

Trial 002 là trường hợp duy nhất hai model ngang nhau — gợi ý trial này chứa
pattern mà force_poly cũng khó (không được mô tả bởi 9 scalar features).

---

## 4. Hướng cải thiện

### Cho `force_poly`
- Thử `degree=3` (220 monomial) với dropout cao hơn để kiểm tra xem polynomial
  bậc cao có capture thêm phi tuyến không.
- Feature thêm: `disp_mag_p75` (percentile), `n_valid` absolute (thay vì ratio)
  để phân biệt sensor bị che một phần.

### Cho `force_model`
- Tăng `early_stop_patience` lên 20–30 để tránh stop quá sớm khi val loss dao động.
- Giảm learning rate xuống 5e-4 + warmup để ổn định hội tụ.
- Thêm input feature: tọa độ tuyệt đối của centroid tiếp xúc (vị trí điểm đặt lực).

### Chung
- Cross-validate trên nhiều seed split để đánh giá variance thực sự.
- Kiểm tra trial_003 bằng tay — nhiều khả năng trial này có điều kiện
  lực/tiếp xúc bất thường (baseline cao, load nhanh, v.v.).

---

## 5. Artifacts

| File | Mô tả |
|---|---|
| `outputs/force_poly/checkpoints/` | Checkpoint best val của force_poly |
| `outputs/force_poly/eval/test_scatter.png` | Scatter plot predicted vs actual (test) |
| `outputs/force_poly/eval/test_timeseries_trial0.png` | Time-series trial đầu tiên |
| `outputs/force_poly/eval/test_breakdown.csv` | Per-trial MAE/RMSE/R² (test) |
| `outputs/force_poly/eval/coefficients.csv` | Hệ số từng monomial (interpretable) |
| `outputs/force_model/checkpoints/` | Checkpoint best val của force_model |
| `outputs/force_model/eval/test_scatter.png` | Scatter plot (test) |
| `outputs/force_model/eval/test_breakdown.csv` | Per-trial breakdown (test) |
