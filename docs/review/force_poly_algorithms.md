# Tài liệu Thuật toán — `src/force_poly`

> Ước lượng lực tiếp xúc (Newton) từ trường dịch chuyển marker xúc giác bằng Polynomial Regression.

---

## 1. Tổng quan kiến trúc

### 1.1 Vị trí trong hệ thống

`force_poly` là module **học máy hóa trường vật lý** — chuyển đổi thông tin hình học (dịch chuyển marker theo pixel) thành đại lượng vật lý (lực N). Nó **phụ thuộc hoàn toàn** vào pipeline tracking ở `src/core`.

```
Ảnh ref + Ảnh frame
        ↓ preprocess (src/core/preprocessing)
        ↓ detect_markers (src/core/detection)
        ↓ track_markers_lk — apply_deadzone=False
        ↓
Displacement field (N_markers × 2) + valid mask
        ↓
compute_features_v1 → vector (9,) float32
        ↓
standardize → polynomial_expand(degree=2) → (55,) float32
        ↓
Linear head / MLP head (16 units)
        ↓
Force prediction (N)
```

### 1.2 Lý do chọn Polynomial Regression

**Cơ sở vật lý (Hertz Contact Theory):**

Tiếp xúc giữa vật cứng và vật liệu đàn hồi tuân theo:

```
F = (4/3) × E* × √R × δ^(3/2)
```

Trong đó:
- `F`: lực tiếp xúc (N)
- `E*`: modulus hiệu dụng của vật liệu
- `R`: bán kính hiệu dụng của đầu tiếp xúc
- `δ`: độ biến dạng (displacement trung bình của marker)

→ Quan hệ `F ~ δ^1.5` là **phi tuyến dạng lũy thừa** — polynomial bậc 2–3 của các feature displacement là xấp xỉ tự nhiên trong vùng hoạt động thực tế.

**Lý do không dùng neural network lớn:**

| Tiêu chí | Polynomial Reg | Deep NN |
|---------|---------------|---------|
| Số tham số (D=9, d=2) | 55 | Hàng nghìn+ |
| Dữ liệu cần | ~100–500 trial | ~10k+ trial |
| Interpretability | Cao (xem hệ số) | Thấp |
| Inference time | <0.1ms | ~1–5ms |
| Generalization trên hardware mới | Tốt hơn | Cần retrain |

---

## 2. Feature Engineering (`features.py`)

### 2.1 Tiêu chí thiết kế feature

Mỗi feature phải thỏa mãn:
1. **Scale-invariant**: không phụ thuộc độ phân giải camera
2. **Mask-aware**: chỉ tính trên marker hợp lệ (LK valid)
3. **Vật lý có ý nghĩa**: phản ánh một khía cạnh của trường biến dạng
4. **Robust với thiếu marker**: trả 0 khi không có valid marker

### 2.2 Chuẩn hóa scale

```python
scale = float(image_w)        # bề rộng ảnh (px)
dx = disp[:, 0] / scale       # đơn vị: phân số bề rộng ảnh
dy = disp[:, 1] / scale
mag = sqrt(dx² + dy²)
```

**Quan trọng:** Dùng cùng `image_w` cho cả trục x lẫn y (không dùng `image_h` riêng), đảm bảo `dx` và `dy` cùng đơn vị. Điều này cần thiết để phép phân tích radial/tangential có ý nghĩa hình học (nếu dùng đơn vị khác nhau cho x/y, vector radial sẽ bị méo).

### 2.3 9 Features và ý nghĩa vật lý

#### Group 1 — Thống kê độ lớn dịch chuyển

| # | Tên | Công thức | Ý nghĩa vật lý |
|---|-----|-----------|----------------|
| 0 | `disp_mag_mean` | `mean(‖disp‖)` over valid | Mức biến dạng trung bình toàn bề mặt |
| 1 | `disp_mag_max` | `max(‖disp‖)` over valid | Đỉnh biến dạng — gần điểm tải nhất |
| 2 | `disp_mag_std` | `std(‖disp‖)` over valid | Độ không đồng đều biến dạng |
| 5 | `disp_mag_sum_norm` | `sum(‖disp‖) / N_valid` | Tổng biến dạng chuẩn hóa |

> **Lưu ý kỹ thuật:** `disp_mag_mean` (feature 0) và `disp_mag_sum_norm` (feature 5) có công thức tương đương về mặt toán học (`mean = sum/N`). Cả hai đều trả giá trị bằng nhau. Đây là redundancy trong feature set — polynomial expansion sẽ sinh ra monomial trùng lặp cho hai feature này. Model vẫn học được nhưng có thể gây multicollinearity cho bậc cao. Hội đồng có thể hỏi điểm này.

#### Group 2 — Hướng tải

| # | Tên | Công thức | Ý nghĩa vật lý |
|---|-----|-----------|----------------|
| 3 | `dx_mean` | `mean(dx)` over valid | Thành phần ngang trung bình (shear x) |
| 4 | `dy_mean` | `mean(dy)` over valid | Thành phần dọc trung bình (shear y hoặc press) |

#### Group 3 — Phân tích Radial/Tangential

**Tính tâm marker grid:**
```python
cx = mean(ref_pts[valid, 0]) / W
cy = mean(ref_pts[valid, 1]) / W
```

Từ mỗi marker đến tâm, tính vector đơn vị hướng tâm `r̂`:
```python
rx_hat = (rx - cx) / ||r||
ry_hat = (ry - cy) / ||r||  (+ 1e-8 để tránh chia 0 tại tâm)
```

**Phân tích:**
```python
radial     = dot(disp, r̂) = dx*rx_hat + dy*ry_hat
tangential = cross_z(r̂, disp) = rx_hat*dy - ry_hat*dx
```

| # | Tên | Giá trị | Ý nghĩa vật lý |
|---|-----|---------|----------------|
| 6 | `radial_disp_mean` | > 0: giãn, < 0: nén | Phân tích nén/giãn tâm — phân biệt press thẳng vs shear |
| 7 | `tangential_disp_mean` | ≠ 0: xoắn/shear | Thành phần xoắn — phân biệt load đối xứng vs lệch tâm |

**Tại sao phân tích radial/tangential quan trọng?**

Khi nhấn thẳng (normal force): marker dịch chuyển hướng **ra ngoài** từ điểm tiếp xúc → `radial_disp_mean < 0` (hướng vào tâm), `tangential_disp_mean ≈ 0`.

Khi kéo trượt (shear force): marker có thành phần tangential lớn, pattern không đối xứng → `|tangential_disp_mean| > 0`.

Hai feature này giúp model **phân biệt normal press vs shear** — điều không làm được chỉ với `dx_mean`, `dy_mean`.

#### Group 4 — Chất lượng tracking

| # | Tên | Công thức | Ý nghĩa |
|---|-----|-----------|---------|
| 8 | `valid_ratio` | `N_valid / N_total` | Nếu nhiều marker mất track → lực ước lượng kém tin cậy |

---

## 3. Model — PolynomialRegressor (`model.py`)

### 3.1 Kiến trúc đầy đủ

```
Input x (B, 9)
  │
  ▼ Standardize (frozen)
  z = (x - feat_mean) / feat_std    → (B, 9)
  │
  ▼ Polynomial Expand
  p = polynomial_expand(z, degree)  → (B, P)
  │
  ▼ Head
  [linear]:    P_in → 1             → (B,)     [pure polynomial]
  [mlp_small]: P_in → 16 → ReLU → Dropout → 1 → (B,)
```

### 3.2 Polynomial Expansion chi tiết

Với D=9 features, bậc d=2, tất cả monomial bậc ≤ 2:

```
Bậc 0: {1}                                        → 1 term  (bias)
Bậc 1: {f0, f1, ..., f8}                          → 9 terms
Bậc 2: {f0², f0·f1, f0·f2, ..., f8²}              → C(9+1,2) = 45 terms
                                          Tổng: 55 terms
```

**Công thức tổng quát:** Số monomial = C(D+d, d)

| Degree | D=9 terms | Ý nghĩa |
|--------|-----------|---------|
| d=1 | 10 | Linear regression thuần |
| d=2 | 55 | Có tương tác bậc 2 (current config) |
| d=3 | 220 | Bổ sung cubic — phù hợp Hertz hơn |

**Cài đặt kỹ thuật:** Monomial indices được flatten và lưu vào `register_buffer` để serialize cùng `state_dict`. Khi load checkpoint, indices được khôi phục chính xác mà không cần code riêng.

```python
# Mỗi combo = tuple index: e.g., (2, 5) → f2 * f5
for combo in combinations_with_replacement(range(D), d):
    term = x[:, combo[0]]
    for i in combo[1:]:
        term = term * x[:, i]
```

### 3.3 Standardize — tại sao đặt TRƯỚC polynomial expand?

Giả sử feature `disp_mag_mean` có giá trị ~0.02 (2% bề rộng ảnh), còn `valid_ratio` ~0.9.

**Không standardize:**
- Bậc 2 của `disp_mag_mean` ≈ 0.0004
- Bậc 2 của `valid_ratio` ≈ 0.81
- Tỉ lệ ~2000× → gradient vanishing cho các monomial của feature nhỏ

**Có standardize:**
- Cả hai feature được đưa về mean=0, std=1
- Các monomial có cùng order of magnitude → gradient ổn định
- Weight decay tác động đều lên các hệ số

### 3.4 Head options

**`head="linear"` — Polynomial Regression thuần:**

```
params = 55  (d=2)  hoặc  220  (d=3)
```

Ưu điểm: hệ số giải thích được. Mỗi hệ số tương ứng 1 monomial → có thể xuất CSV coefficients (eval.py `_dump_linear_coefficients`).

**`head="mlp_small"` — Poly + MLP (hiện tại config):**

```
params = P_in × 16 + 16 + 16 × 1 + 1
       = 55 × 16 + 17
       = 897 params  (d=2)
```

Thêm một lớp phi tuyến ReLU → bắt được interaction bậc cao hơn mà không cần tăng `degree`. Đánh đổi: mất interpretability hoàn toàn.

---

## 4. Dataset và Data Split (`dataset.py`)

### 4.1 Trial-level split — tránh data leakage

```python
# SAI (frame-level split): frame t và t+1 đều trong train, test "thấy" hàng xóm
shuffle(all_frames) → split 70/15/15

# ĐÚNG (trial-level split): toàn bộ trial vào 1 split duy nhất
shuffle(trials) → split 70/15/15 trials
```

**Lý do critical:** Trong một trial, force profile là đường cong liên tục (tăng dần → đỉnh → giảm dần). Frame t và t+1 có force gần như bằng nhau. Nếu split theo frame: model "nhớ" giá trị lân cận → R² ảo ~0.99 trong khi không generalize sang trial mới.

Trial-level split = kiểm tra generalization thực sự: model có học được **quan hệ vật lý** hay chỉ nhớ trajectory cụ thể?

### 4.2 Cache NPZ format

Dataset đọc từ cache `.npz` được chuẩn bị bởi `src/force_model/prepare.py`. Mỗi file lưu:

```
ref_pts:  (N_markers, 2)  — toạ độ marker tham chiếu
disp:     (T, N, 2)       — dịch chuyển theo frame
valid:    (T, N)           — LK valid mask
force:    (T,)             — lực ground truth (N) từ Imada gauge
image_w:  scalar           — bề rộng ảnh
ts_mono:  (T,)             — timestamp monotonic (giây)
trial_id: string           — nhận dạng trial
```

### 4.3 Pre-compute vào RAM

```python
for p in cache_paths:
    feats = compute_features_trial(ref_pts, disp, valid, W)  # (T, 9)
    # Concat tất cả vào self.features = (N_total_frames, 9)
```

Với dữ liệu xúc giác điển hình: 100 trial × 300 frame × 9 feature × 4 bytes = ~1.1 MB. Hoàn toàn phù hợp RAM — không cần num_workers, không I/O khi train.

---

## 5. Training Loop (`train.py`)

### 5.1 Loss function — Huber Loss

```
L_Huber(e) = {  ½ e²           nếu |e| ≤ δ
              {  δ(|e| - ½δ)   nếu |e| > δ

δ = 1.0 N  (1 Newton threshold)
```

**Tại sao Huber chứ không phải MSE?**

Force gauge Imada có thể có spike nhiễu (impact khi tiếp xúc đột ngột, rung motor). Với MSE, spike 5N tạo loss = 25 — áp đảo 10 frame bình thường. Với Huber (δ=1N): spike 5N → loss = 1×(5-0.5) = 4.5 — giảm ảnh hưởng nhưng vẫn học.

**Lựa chọn δ=1.0 N:** Sai số cơ học thực tế của force gauge ≈ 0.1–0.5N. Đặt δ=1N có nghĩa: model bị "train nghiêm" cho sai số 0–1N (zone MSE), robust với nhiễu >1N.

### 5.2 Optimizer — AdamW

```python
AdamW(lr=5e-3, weight_decay=1e-2)
```

**Weight decay 1e-2** = L2 regularization trên hệ số polynomial. Với 55 monomial, nhiều term bậc cao sẽ overfit nếu không regularize. Hiệu ứng: các monomial ít đóng góp bị shrink về 0 → sparse coefficients tự nhiên.

**lr=5e-3:** Cao cho AdamW — phù hợp dataset nhỏ (~30k frames), hội tụ nhanh. Scheduler cosine annealing giảm về 5e-5 (1% lr ban đầu) để fine-tune cuối.

### 5.3 Learning Rate Schedule — Cosine Annealing

```
lr(epoch) = lr_min + ½(lr_max - lr_min)(1 + cos(πt/T_max))
```

```
lr_max = 5e-3, lr_min = 5e-5, T_max = 300
```

Không restart (CosineAnnealingLR đơn giản, không CosineAnnealingWarmRestarts). Giảm smooth từ 5e-3 → 5e-5 qua 300 epoch. Tránh oscillation ở cuối training như fixed lr gây ra.

### 5.4 Early Stopping

```python
patience = 40 epochs  (không cải thiện val MAE)
```

Với 300 epoch tối đa và patience=40: nếu model converge ở epoch 150, training dừng ở 190. Checkpoint tốt nhất (theo val MAE) được lưu riêng — không dùng model ở epoch cuối.

### 5.5 Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| MAE | `mean(|pred - target|)` | Sai số tuyệt đối trung bình (N) — dễ hiểu vật lý |
| RMSE | `sqrt(mean((pred-target)²))` | Nhạy outlier hơn MAE |
| R² | `1 - SS_res/SS_tot` | 1.0 = hoàn hảo; 0 = không hơn mean baseline |

**MAE được dùng để chọn best checkpoint** (không phải loss) vì loss là Huber — không trực tiếp tương đương sai số Newton.

---

## 6. Inference (`infer.py`) — Điểm quan trọng

```python
tracked, valid = track_markers_lk(
    ref_gray, frame_gray, ref_pts, config, apply_deadzone=False
)
```

**`apply_deadzone=False` trong inference** — khác với default pipeline tracking (deadzone=1px). Lý do: lực nhỏ (0.1–0.5N) tương ứng biến dạng nhỏ (~0.5–1px). Nếu dùng deadzone, những biến dạng nhỏ bị zero-out → feature vector =0 → model dự đoán ~0N ngay cả khi có lực thực. Tắt deadzone cho force estimation là đúng về mặt vật lý.

---

## 7. Luồng dữ liệu đầy đủ

```
[Thu thập]
Camera + Imada gauge → SyncSnapshot → data/images/ + data/record.csv

[Chuẩn bị cache]
src/force_model/prepare.py:
  read record.csv + images → detect markers → LK tracking
  → cache .npz (ref_pts, disp, valid, force, ts_mono)
  → data/cache/force_model/session/trial.npz

[Train]
discover_trials → split (trial-level 70/15/15) → PolyFeatureDataset
  → fit_scaler (từ train features) → PolynomialRegressor
  → AdamW + Huber + CosineAnnealing + EarlyStop
  → outputs/force_poly/checkpoints/best.pt

[Eval]
best.pt → PolyFeatureDataset(test) → MAE/RMSE/R² per trial
  → scatter.png + timeseries.png + coefficients.csv

[Inference]
(ref_img, frame_img) → preprocess → detect → track (no deadzone)
  → compute_features_v1 → best.pt → force (N)
```

---

## 8. Bảng tham số và hướng dẫn điều chỉnh

| Module | Tham số | Giá trị hiện tại | Tăng | Giảm |
|--------|---------|-----------------|------|------|
| Model | `degree` | 2 (55 terms) | Fit tốt hơn nếu đủ data | Underfitting |
| Model | `head` | `mlp_small` | — | `linear` để xem hệ số |
| Model | `hidden_head` | 16 | Capacity cao hơn | Faster, ít overfit |
| Train | `weight_decay` | 1e-2 | Regularize mạnh hơn, sparse coeff | Overfit nếu ít data |
| Train | `lr` | 5e-3 | Diverge nếu quá cao | Hội tụ chậm |
| Train | `huber_delta` | 1.0 N | Robust hơn với outlier nhỏ | MSE-like |
| Train | `early_stop_patience` | 40 | Tránh early stop sớm | Dừng sớm hơn |
| Data | `split_seed` | 42 | Đổi để kiểm tra variance | — |

---

## 9. Câu hỏi phản biện và cách trả lời

**Q: Tại sao chọn polynomial regression thay vì random forest hoặc SVR?**
> A: (1) Physics prior: quan hệ lực-biến dạng là lũy thừa → polynomial là xấp xỉ tự nhiên. (2) Dữ liệu hạn chế: RF cần nhiều sample hơn; với ~100 trial, polynomial ít overfit hơn. (3) Interpretability: hệ số tuyến tính cho phép kiểm tra physics của model (feature nào quan trọng?). (4) Inference <0.1ms — quan trọng cho realtime.

**Q: Feature `disp_mag_mean` (0) và `disp_mag_sum_norm` (5) có giá trị bằng nhau không?**
> A: Đúng, `sum/N_valid = mean` — đây là redundancy. Trong linear head, hai cột giống nhau → ma trận feature rank-deficient → hệ số không unique. Với polynomial expansion bậc 2, tạo thêm monomial `f0*f5` = `f0²` — tức là một monomial trùng lặp. Model vẫn học được (optimizer tìm một trong vô số nghiệm), nhưng regularization sẽ phân bố trọng số tùy tiện giữa chúng. Giải pháp: xóa feature 5 và thay bằng feature vật lý khác (ví dụ kurtosis biến dạng để bắt loading point).

**Q: Tại sao tâm radial/tangential là mean của valid ref_pts thay vì centroid cố định?**
> A: Lựa chọn hiện tại có nhược điểm: nếu nhóm marker bên trái mất track, tâm bị lệch phải → feature radial/tangential bị bias theo thời gian trong một trial. Tốt hơn nên dùng centroid của **tất cả** ref_pts (không phụ thuộc valid mask). Lý do dùng valid mean: đơn giản và tránh trường hợp marker gốc không đều grid. Trong thực tế, valid_ratio > 0.7 trong đa số frame nên sai số nhỏ.

**Q: degree=2 có đủ để mô tả Hertz contact (F ~ δ^1.5) không?**
> A: Taylor expansion của δ^1.5 quanh điểm hoạt động δ₀: `δ^1.5 ≈ a + b·δ + c·δ²` — bậc 2 đủ **trong vùng làm việc hẹp** (ví dụ 0–5N). Với range rộng hơn (0–20N), có thể cần degree=3. Config hiện tại là compromise: d=2 ít overfit hơn với dữ liệu hạn chế.

**Q: Tại sao không dùng time-series model (LSTM, TCN) thay vì per-frame regression?**
> A: (1) Force tức thời chủ yếu phụ thuộc biến dạng hiện tại, không cần lịch sử dài. (2) Per-frame model dễ đánh giá: MAE theo Newton rõ ràng, không cần warm-up. (3) Latency: LSTM tốn ~5–10ms vs <0.1ms cho polynomial. (4) Temporal smoothing nếu cần có thể thêm EMA ở post-processing mà không ảnh hưởng model.

**Q: Trial-level split có đảm bảo không có leakage nếu cùng hardware setup?**
> A: Trial-level split loại bỏ temporal leakage nhưng không loại bỏ **distribution leakage** nếu tất cả trials có cùng force range. Nên kiểm tra distribution của force trong train vs test split (histogram). Nếu force range trong test không có trong train → extrapolation, R² sẽ giảm mạnh. Giải pháp: đảm bảo thu thập trial đủ phủ range force quan tâm.

**Q: Tại sao dùng AdamW thay vì Adam hoặc SGD?**
> A: Adam + L2 penalty trong loss ≠ true weight decay (Adam scale gradient → weight decay không uniform). AdamW tách weight decay khỏi gradient step → regularization đúng về mặt lý thuyết. Với polynomial có nhiều hệ số nhỏ cần shrink về 0, AdamW regularize tốt hơn Adam.

**Q: Model có thể dự đoán lực âm không? Làm thế nào?**
> A: Polynomial head không giới hạn output → có thể dự đoán giá trị âm với input ngoài vùng train (ví dụ không tiếp xúc). Giải pháp: clip output tại inference (`max(0, pred)`) hoặc thêm output activation `Softplus`. Hiện tại code không có clip — cần xử lý ở ứng dụng.

---

## 10. Hạn chế đã biết

| Hạn chế | Nguyên nhân | Giải pháp đề xuất |
|---------|-------------|-------------------|
| feature 0 = feature 5 (redundancy) | `sum/N = mean` | Thay feature 5 bằng kurtosis hoặc weighted sum by distance |
| Tâm radial phụ thuộc valid mask | Tránh state cố định | Dùng centroid tất cả ref_pts (không filter valid) |
| Không giới hạn output dương | Polynomial head | Clip `max(0, pred)` tại inference |
| degree=2 xấp xỉ kém ở force lớn | Hertz ~ δ^1.5 nonlinear | Thử degree=3 khi có đủ data |
| Per-frame, không có temporal smoothing | Thiết kế đơn giản | EMA post-processing hoặc chuyển sang causal RNN |
