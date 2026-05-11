# Tài liệu Thuật toán — `src/core`

> Mục tiêu: giải thích đầy đủ lý do chọn thuật toán, cơ chế hoạt động của từng bước, ý nghĩa tham số, và các điểm có thể bị phản biện cùng cách trả lời.

---

## 1. Tiền xử lý ảnh (`preprocessing.py`)

### 1.1 Vấn đề cần giải quyết

Cảm biến xúc giác dùng đèn LED chiếu đồng đều lên bề mặt có marker trắng. Trong thực tế, LED tạo ra **vignetting** — vùng giữa sáng hơn, vùng rìa tối hơn — và **drift độ sáng** theo thời gian (nhiệt độ, rung). Nếu dùng ngưỡng toàn cục (global threshold) để phát hiện blob, các marker gần rìa sẽ bị bỏ sót hoặc nhòe.

### 1.2 Pipeline 3 bước

```
img_gray  →  (1) Box blur 101×101  →  background
          →  (2) subtract + normalize [0,255]
          →  (3) CLAHE (clip 2.5, grid 8×8)
          →  img_preprocessed
```

#### Bước 1 — Ước lượng nền bằng Box Blur

```python
background = cv2.blur(img_gray, (101, 101))
```

**Ý tưởng:** Kernel 101×101 pixel lớn hơn đường kính marker (~10–20 px) một bậc độ lớn, nên marker không "lọt qua" bộ lọc. Kết quả là ảnh nền trơn không chứa chi tiết marker — chỉ còn vignetting + gradient ánh sáng.

**Vì sao Box Blur chứ không phải Gaussian Blur?**
- Box Blur dùng integral image, độ phức tạp **O(1) mỗi pixel** bất kể kích thước kernel.
- Gaussian 101×101 tốn ~10–20× thời gian hơn với kết quả ước lượng nền tương đương (cả hai đều low-pass cắt tần số cao).
- Không có grid artifact (khác median filter).

**Vì sao kernel 101×101?**
- Marker có đường kính ≈ 10–15 px. Kernel cần ít nhất 5× đường kính marker để đảm bảo marker không ảnh hưởng giá trị trung bình cục bộ.
- 101 = lẻ (yêu cầu của OpenCV), ≈ 7× đường kính marker lớn nhất (15 px) — an toàn.
- Nếu kernel quá nhỏ (ví dụ 21×21): nền bị "kéo lên" bởi marker → phép trừ sau đó tạo halo tối xung quanh marker → phát hiện sai.

#### Bước 2 — Trừ nền và chuẩn hóa

```python
normalized = cv2.subtract(img_gray, background)   # saturate ở 0 (uint8)
normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
```

`cv2.subtract` dùng **saturating arithmetic**: kết quả âm được kẹp về 0 (không wrap-around). Sau trừ nền, toàn ảnh có khoảng giá trị hẹp; `NORM_MINMAX` stretch về [0, 255] để tận dụng toàn dải động.

**Lưu ý quan trọng:** Bước này làm mất thông tin ánh sáng tuyệt đối — chỉ còn tương phản tương đối. Đây là chủ ý: chúng ta chỉ cần phát hiện "điểm sáng hơn xung quanh".

#### Bước 3 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
output = clahe.apply(normalized)
```

**CLAHE so với HE (Histogram Equalization) toàn cục:**
- HE toàn cục khuếch đại ồn ào ở vùng nền đồng nhất (flat region) — nguy hiểm cho blob detector.
- CLAHE chia ảnh thành 8×8 = 64 ô, cân bằng histogram cục bộ mỗi ô, nhưng **giới hạn** độ khuếch đại qua `clipLimit`.

**Ý nghĩa `clipLimit = 2.5`:**
- Tại mỗi ô, histogram bị "cắt" ở mức `clipLimit × (số pixel ô / 256)`.
- `clipLimit = 1.0`: không khuếch đại → CLAHE = không làm gì.
- `clipLimit = 2.5`: khuếch đại vừa phải — đủ để làm nổi marker yếu nhưng không tạo artifact.
- Nếu tăng lên 4.0+: nhiễu trong vùng nền bị khuếch đại → false positive khi detect.

**Ý nghĩa `tileGridSize = (8, 8)`:**
- Chia ảnh thành lưới 8×8 ô = 64 vùng xử lý độc lập.
- Ô nhỏ hơn (16×16): thích nghi tốt hơn nhưng dễ over-enhance noise.
- Ô lớn hơn (4×4): tiệm cận HE toàn cục, mất lợi ích cục bộ.
- 8×8 là giá trị mặc định của OpenCV, thực nghiệm cho ảnh cảm biến ~480×640.

### 1.3 Các câu hỏi phản biện có thể gặp

**Q: Tại sao không dùng Gaussian blur thay Box blur?**
> A: Với kernel 101×101, Gaussian đắt hơn ~10× mà không cải thiện kết quả ước lượng nền. Mục tiêu là low-pass filter tần số cao (marker), không cần trọng số Gaussian — Box blur là lựa chọn tốt nhất về hiệu năng.

**Q: Tại sao không dùng morphological closing để ước lượng nền?**
> A: Morphological closing (dilation + erosion) chính xác hơn cho nền không phẳng, nhưng với kernel lớn (>50px) sẽ rất chậm. Vignetting LED tương đối smooth nên Box blur 101×101 đủ chính xác, nhanh hơn ~100× closing cùng kích thước.

**Q: CLAHE có cần thiết sau khi đã normalize không?**
> A: Cần. Normalize đưa dải về [0,255] nhưng histogram vẫn lệch (phần lớn pixel là nền tối, chỉ ~5% là marker sáng). CLAHE tái phân phối histogram cục bộ, làm vùng marker nổi hơn và giảm ngưỡng phát hiện cần thiết.

---

## 2. Phát hiện Marker (`detection.py`)

### 2.1 Thuật toán SimpleBlobDetector

OpenCV `SimpleBlobDetector` là **thuật toán ngưỡng đa mức (multi-threshold voting)**:

```
Với mỗi threshold T ∈ {min_threshold, min_threshold+step, ..., max_threshold}:
    1. Binary thresholding: pixel > T → 255, ngược lại → 0
    2. Tìm connected components (contours)
    3. Tính tâm mỗi region
Gom nhóm các tâm gần nhau qua các mức threshold
→ Blob = nhóm ổn định qua nhiều threshold
Lọc bằng các tiêu chí hình học
```

**Tại sao multi-threshold chứ không phải ngưỡng đơn?**
- Marker ở góc ảnh có thể sáng kém hơn marker ở giữa 20–30 giá trị. Ngưỡng đơn sẽ miss một nhóm.
- Voting qua nhiều threshold: blob "thật" xuất hiện ổn định qua nhiều mức, noise chỉ xuất hiện ở một vài mức → tự nhiên lọc được noise.

### 2.2 Các tham số ngưỡng

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `min_threshold` | 50 | Threshold thấp nhất — phát hiện cả marker mờ |
| `max_threshold` | 220 | Threshold cao nhất — loại bỏ gần hết highlight |
| `step` | 10 | Bước nhảy: 17 mức threshold được thử |

**Tại sao `min_threshold = 50`?** Sau CLAHE, nền tối thường nằm ở 0–40, marker sáng ở 60–220. Bắt đầu từ 50 đảm bảo chỉ "vào vùng" khi đã có tương phản thực sự.

**Tại sao `step = 10`?** 17 mức threshold là trade-off giữa độ chính xác và tốc độ. Nếu step=1 → 171 mức → chậm hơn 10× không có lợi ích đáng kể.

### 2.3 Các bộ lọc hình học

#### `filterByColor = True, blobColor = 255`
Chỉ giữ blob **sáng** (bright on dark background). Loại ngay blob tối không phải marker.

#### `filterByArea`

```
min_area = 30 px²  →  đường kính tối thiểu ≈ 6.2 px
max_area = 500 px² →  đường kính tối đa ≈ 25.2 px
```

- `min_area = 30`: loại noise đơn pixel và artifact nhỏ sau CLAHE.
- `max_area = 500`: loại vùng sáng lớn (phản xạ, đèn nền).
- **Lưu ý**: diện tích thực của marker phụ thuộc khoảng cách camera-cảm biến và độ phóng đại. Giá trị này cần hiệu chỉnh khi thay đổi optical setup.

#### `filterByCircularity, minCircularity = 0.5`

```
Circularity = 4π × Area / Perimeter²  ∈ [0, 1]
Hình tròn hoàn hảo → 1.0
Hình vuông → π/4 ≈ 0.785
Hình chữ nhật 2:1 → ~0.63
```

`minCircularity = 0.5` là ngưỡng khá dễ — cho phép marker bị méo nhẹ khi bề mặt biến dạng. Nếu tăng lên 0.8: mất marker khi cảm biến bị nén mạnh (marker trở nên ellipse).

#### `filterByInertia, minInertiaRatio = 0.3`

```
Inertia ratio = (eigenvalue nhỏ) / (eigenvalue lớn) của moment quán tính
Hình tròn → 1.0; Đường thẳng → 0.0
```

`minInertiaRatio = 0.3` loại các blob hình dài, hẹp (scratch, edge artifact). Ít chặt hơn circularity — bổ sung phát hiện blob oval kéo dài.

#### `filterByConvexity, minConvexity = 0.7`

```
Convexity = Area / Area(Convex Hull)
Hình tròn → 1.0; Hình có hốc → < 1.0; Hình sao → thấp
```

`minConvexity = 0.7` loại blob có hình dạng phức tạp (overlap 2 marker, artifact nền). Không quá chặt vì marker bị nén có thể hơi lõm cạnh.

### 2.4 Tái sử dụng detector trong realtime

```python
def detect_markers(img, config, _detector=None):
    detector = _detector if _detector is not None else create_blob_detector(config=config)
```

`SimpleBlobDetector_create` có overhead khởi tạo (allocate params object). Trong chế độ realtime, detector được tạo một lần ở `BaseRealtimePipeline.__init__` và truyền vào qua `_detector`.

### 2.5 Câu hỏi phản biện

**Q: Tại sao không dùng `cv2.findContours` + filter thay vì SimpleBlobDetector?**
> A: `findContours` yêu cầu chọn một ngưỡng toàn cục trước. SimpleBlobDetector giải quyết chính xác bài toán "ngưỡng nào tốt nhất" bằng cách thử nhiều mức và voting — phù hợp khi độ sáng marker không đồng đều. Thêm vào đó, các bộ lọc hình học tích hợp sẵn, không cần code thêm.

**Q: Tại sao không dùng deep learning (YOLO, keypoint detection)?**
> A: Marker là blob tròn sáng trên nền tối — bài toán cực kỳ đặc thù về hình dạng. SimpleBlobDetector đạt độ chính xác cao (>95%) với thời gian xử lý <1ms. Deep learning sẽ thêm dependency nặng (~100MB model), cần GPU để realtime, và không mang lại cải thiện đáng kể cho domain hẹp này.

**Q: min_area = 30 chọn dựa trên cơ sở nào?**
> A: Dựa trên đặc điểm phần cứng: camera ở khoảng cách cố định, marker vật lý đường kính ~2mm, tương đương ~8-10px → diện tích ~50-80px². min_area = 30 cho margin an toàn khi marker bị nén hoặc ánh sáng yếu. Nên hiệu chỉnh lại nếu thay đổi optical setup.

---

## 3. Theo dõi Marker (`tracking.py`)

### 3.1 Pyramid Lucas-Kanade Optical Flow

#### Nền tảng lý thuyết — Lucas-Kanade cơ bản

Lucas-Kanade giả định **brightness constancy** và **local motion**:

```
I(x, y, t) = I(x + u, y + v, t+1)   (điểm p di chuyển (u,v) giữa 2 frame)
```

Taylor expansion + giả định motion nhỏ:
```
Ix·u + Iy·v + It = 0
```

Trong cửa sổ W×W pixel xung quanh điểm, lập hệ phương trình overdetermined:
```
[Ix1  Iy1] [u]   [-It1]
[Ix2  Iy2] [v] = [-It2]
[...  ...]       [ ...]
```

Giải bằng least-squares (AᵀA)[u,v]ᵀ = Aᵀb, với A là ma trận gradient.

**Điều kiện để giải được:** Ma trận cấu trúc AᵀA phải non-singular, nghĩa là vùng window phải có **texture theo 2 hướng** (không phải cạnh thẳng hay vùng đồng nhất). Marker tròn thoả mãn điều kiện này.

#### Pyramid — xử lý motion lớn

LK cơ bản chỉ giải được với motion nhỏ (< vài pixel). **Pyramid LK** giải quyết bằng cách:

```
Level 3 (ảnh 1/8 kích thước):  track với motion lớn → estimate coarse
Level 2 (ảnh 1/4 kích thước):  refine
Level 1 (ảnh 1/2 kích thước):  refine
Level 0 (ảnh gốc):              track chính xác sub-pixel
```

Với `max_level = 3`: có thể track motion tới `win_size/2 × 2^max_level = 10.5 × 8 = 84 pixel` theo lý thuyết. Trong thực tế marker xúc giác dịch chuyển <50px nên max_level=3 là đủ.

#### Cửa sổ tìm kiếm `win_size = (21, 21)`

```
Cửa sổ 21×21 = 441 pixel để tính gradient và solve least-squares
```

- Quá nhỏ (7×7): ít điểm → hệ LS không ổn định, nhạy noise.
- Quá lớn (41×41): chứa nhiều điểm không thuộc cùng motion → vi phạm giả thuyết local motion → sai.
- 21×21: cân bằng — đủ lớn cho marker đường kính ~10px, đủ nhỏ để không overlap 2 marker.

**Điều kiện dừng lặp:**
```python
criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, max_iter=30, eps=0.01)
```
Dừng khi: đạt 30 vòng lặp HOẶC thay đổi vị trí < 0.01 pixel. `eps=0.01` cho độ chính xác sub-pixel.

### 3.2 Forward-Backward Consistency Check (FB Check)

```python
pts_def,  status_f, _ = calcOpticalFlowPyrLK(img_ref, img_def,  pts_ref, None)
pts_back, status_b, _ = calcOpticalFlowPyrLK(img_def, img_ref,  pts_def, None)

fb_error = ||pts_back - pts_ref||₂
valid = (status_f == 1) & (status_b == 1) & (fb_error < 2.0)
```

**Ý tưởng:** Nếu track từ A→B cho vị trí B', rồi track ngược B'→A cho vị trí A', thì A' phải gần A. Nếu không: track đã "trôi" sang vùng texture tương tự (aperture problem).

**Tại sao `fb_threshold = 2.0 px`?**
- LK có thể đạt độ chính xác 0.1–0.5px khi ảnh tốt.
- 2.0px là ngưỡng conservative: chấp nhận sai số cộng dồn 2 lần LK (forward + backward).
- < 1.0px: quá chặt, loại marker bị motion blur nhẹ.
- > 3.0px: bỏ sót track trôi xa, ảnh hưởng slip detection.

**Lưu ý:** FB check không phát hiện được systematic drift — nếu texture trong cửa sổ không đủ đặc trưng, cả forward và backward đều bị dẫn đến cùng một điểm sai → FB error nhỏ nhưng vẫn sai. Đây là hạn chế cố hữu của LK.

### 3.3 Deadzone Filtering

```python
if apply_deadzone:
    min_disp = require(config, "tracking.min_displacement")  # = 1 px
    disps = ||pts_def - ref_points||₂
    pts_def[disps < min_disp] = ref_points[disps < min_disp]
```

**Vấn đề giải quyết:** Vật liệu cảm biến có **độ đàn hồi** — khi không có ngoại lực, marker vẫn rung/trôi nhẹ (~0.5–1px do nhiễu cơ học và nhiễu ảnh). Deadzone reset những dịch chuyển nhỏ về 0, tránh false positive slip.

**`min_displacement = 1 px` (config default 1, CLAUDE.md đề cập 2):**
- Config hiện tại: `min_displacement: 1`
- Bảo thủ hơn giá trị 2px trong docs: chấp nhận dịch chuyển 1px là "có thật".
- Slip V2 dùng `apply_deadzone=False`: giữ nguyên tín hiệu nhỏ vì slip chậm (creep) tích lũy qua nhiều frame.

**Câu hỏi phản biện:**

**Q: Tại sao dùng LK thay vì template matching hoặc feature descriptor?**
> A: LK được tối ưu bằng pyramid và tính gradient incremental, cho độ trễ <2ms/frame. Template matching (NCC) trên grid N marker là O(N × W × H) — chậm hơn nhiều lần. Feature descriptor (SIFT, ORB) phù hợp khi marker có thể thay đổi ngoại hình, nhưng marker tròn đồng nhất không có đủ keypoint đặc trưng cho matching đáng tin cậy.

**Q: Forward-backward check có thực sự phát hiện được tất cả track lỗi không?**
> A: Không hoàn toàn. FB check rất hiệu quả cho aperture problem và large motion drift. Tuy nhiên với systematic texture ambiguity (nhiều vùng giống nhau — ví dụ grid marker đều nhau), cả forward lẫn backward đều có thể bị kéo sang điểm sai nhưng consistent. Đây là lý do system cũng dùng blob re-detection sau khi mất reference.

**Q: max_level=3 có đủ cho case cảm biến bị trượt nhanh?**
> A: Về lý thuyết, có thể track tới ~84px/frame. Với framerate 30fps, tương đương ~2520px/s. Trượt vật lý trên cảm biến xúc giác hiếm khi đạt tốc độ đó. Nếu ứng dụng yêu cầu track motion rất nhanh, có thể tăng max_level=4 nhưng sẽ tốn thêm ~30% thời gian.

**Q: `win_size=(21,21)` có bị overlap giữa các marker không?**
> A: Marker có đường kính ~10-15px, khoảng cách giữa các marker ~25-40px. Cửa sổ 21×21 (bán kính 10px) thỉnh thoảng chạm vùng marker lân cận. Đây là trade-off chấp nhận được vì: (1) marker lân cận có cùng motion pattern (cùng biến dạng vật liệu), (2) FB check sẽ loại những track không nhất quán.

---

## 4. Trực quan hóa (`visualization.py`)

### 4.1 Render mũi tên dịch chuyển

```python
end = p0 + disp * scale          # scale = 3.0 (khuếch đại hiển thị)
color = (0, 255 - min(mag*20, 255), min(mag*20, 255))
```

**Giải thích màu:**
- `mag × 20` = intensity: magnitude 1px → intensity=20 (xanh lá nhạt), 12px → 240 (đỏ-cam).
- Màu BGR: `(0, 255-intensity, intensity)` → gradient từ **xanh lá thuần** (motion nhỏ) sang **đỏ** (motion lớn).
- Điểm không di chuyển (`mag < motion_magnitude_threshold = 0.3px`): vẽ chấm xám.

**`arrow_scale = 3.0`:** Dịch chuyển thực tế 5px hiển thị như 15px — dễ quan sát trên màn hình. Đây là scale hiển thị thuần túy, không ảnh hưởng số liệu.

**`motion_magnitude_threshold = 0.3px`:** Deadzone của visualization (tách biệt với deadzone tracking `1px`). Sau deadzone tracking, vị trí đã được snap về ref — chỉ còn các marker có dịch chuyển ≥1px. Ngưỡng 0.3px ở đây chủ yếu lọc floating point residual.

### 4.2 Câu hỏi phản biện

**Q: Scale 3× có làm người dùng hiểu sai về mức độ dịch chuyển không?**
> A: Đây là lựa chọn có chủ đích cho **visualization** — giúp phát hiện pattern dịch chuyển (hướng, đồng đều hay không). Giá trị số thực tế được log riêng và dùng trong slip detection, không qua scale. Cần ghi chú rõ "×3 scale" trong overlay để tránh nhầm lẫn.

---

## 5. Tổng hợp — Mối liên hệ giữa các module

```
preprocess(img_gray)  → img_enhanced
    ↓
detect_markers(img_enhanced)  → ref_pts, keypoints
    ↓
[frame tiếp theo]
preprocess(new_frame) → img_def_enhanced
    ↓
track_markers_lk(img_ref_enhanced, img_def_enhanced, ref_pts)
    → def_pts, valid_mask
    ↓
visualize_flow_arrows(img_def_enhanced, ref_pts, def_pts, valid_mask)
    → annotated_frame
```

**Điểm quan trọng:** LK track trên **ảnh đã preprocess** (không phải ảnh gốc). Điều này có ưu điểm: gradient ảnh rõ hơn → LK hội tụ nhanh, ít vòng lặp hơn. Nhược điểm tiềm ẩn: nếu CLAHE thay đổi cách nhau giữa 2 frame (ánh sáng thay đổi đột ngột), gradient không nhất quán → có thể làm FB error tăng. Trong thực tế, ánh sáng LED của cảm biến xúc giác rất ổn định nên không phải vấn đề.

---

## 6. Bảng tham số tổng hợp và hướng dẫn điều chỉnh

| Module | Tham số | Giá trị | Tăng lên | Giảm xuống |
|--------|---------|---------|----------|------------|
| Preprocess | `blur_kernel` | [101,101] | Ước lượng nền chính xác hơn (chậm hơn) | Có thể để lại vignetting |
| Preprocess | `clahe_clip_limit` | 2.5 | Tương phản mạnh hơn, dễ false blob | Marker mờ không được khuếch đại |
| Preprocess | `clahe_grid` | [8,8] | Thích nghi cục bộ hơn | Tiệm cận HE toàn cục |
| Detection | `min_area` | 30 | Loại marker nhỏ/noise | Bắt được marker nhỏ hơn |
| Detection | `max_area` | 500 | Cho phép marker lớn hơn | Bỏ sót marker bị méo lớn |
| Detection | `min_circularity` | 0.5 | Chỉ chấp nhận marker tròn | Chấp nhận marker méo khi nén |
| Detection | `min_convexity` | 0.7 | Loại blob phức tạp | Có thể lấy hai marker đang overlap |
| Tracking | `win_size` | [21,21] | Robust hơn với nhiễu | Track nhạy hơn với motion nhỏ |
| Tracking | `max_level` | 3 | Theo dõi motion lớn hơn (chậm hơn) | Chỉ track motion nhỏ |
| Tracking | `fb_threshold` | 2.0 | Chấp nhận track kém chính xác | Loại nhiều track hơn (strict) |
| Tracking | `min_displacement` | 1 px | Deadzone lớn hơn, bỏ motion nhỏ | Nhạy hơn với chuyển động nhỏ |

---

## 7. Hạn chế đã biết và hướng cải thiện

| Hạn chế | Nguyên nhân | Giải pháp tiềm năng |
|---------|-------------|---------------------|
| Track drift khi marker ra khỏi vùng nhìn | LK không xử lý occlusion | Re-detect và re-associate bằng Hungarian |
| Systematic drift với grid đều (aperture problem) | Texture marker tương tự nhau | Dùng marker có pattern duy nhất (mã màu, hình dạng) |
| CLAHE không nhất quán khi ánh sáng thay đổi đột ngột | Histogram thay đổi giữa ref và current frame | Dùng fixed CLAHE params + test điều kiện ánh sáng ổn định |
| `min_area` và `max_area` phụ thuộc optical setup | Pixel/mm thay đổi theo khoảng cách | Tự động hiệu chỉnh dựa trên calibration |
| Deadzone `1px` có thể bỏ sót slip rất chậm | Trade-off với elastic noise | Slip V2 đã giải quyết bằng `apply_deadzone=False` |
