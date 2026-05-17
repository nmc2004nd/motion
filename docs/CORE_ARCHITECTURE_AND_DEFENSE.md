# Kiến Trúc `src/core` Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code trong `src/core`, bỏ qua phần `calibration.py` theo yêu cầu. `src/core` không phải là một mô hình học sâu, mà là pipeline xử lý ảnh nền tảng để phát hiện marker, theo dõi chuyển động marker và trực quan hóa trường dịch chuyển. Các module này được dùng bởi các phần phía sau như `force_poly`, `force_model`, pipeline thu thập dữ liệu và các thuật toán slip.

## 1. Vai trò của `src/core`

`src/core` xử lý ảnh tactile marker theo luồng:

```text
Ảnh grayscale từ camera
        |
        v
preprocessing.py
  khử chiếu sáng không đều
  tăng tương phản cục bộ
        |
        v
detection.py
  phát hiện tâm marker trên ảnh reference
        |
        v
tracking.py
  Lucas-Kanade tracking từ reference sang frame biến dạng
  forward-backward check
  deadzone tùy chọn
        |
        v
visualization.py
  vẽ mũi tên displacement ref -> deformed
```

Các module chính:

| File | Vai trò |
|---|---|
| `preprocessing.py` | Tiền xử lý ảnh grayscale: background subtraction, normalize, CLAHE |
| `detection.py` | Phát hiện marker sáng bằng OpenCV `SimpleBlobDetector` |
| `tracking.py` | Theo dõi marker bằng Pyramid Lucas-Kanade và forward-backward check |
| `visualization.py` | Vẽ trường displacement dạng mũi tên |
| `calibration.py` | Ngoài phạm vi tài liệu này |

Config liên quan nằm trong `config/pipeline_config.yaml`:

| Nhóm config | Một số tham số chính |
|---|---|
| `preprocessing` | `blur_kernel`, `clahe_clip_limit`, `clahe_grid` |
| `detection` | threshold range, area, circularity, inertia, convexity |
| `tracking` | LK window, pyramid level, FB threshold, min displacement |
| `visualization` | arrow scale, màu, ngưỡng chuyển động, độ dày mũi tên |

## 2. Tiền xử lý ảnh trong `preprocessing.py`

Ảnh đầu vào là grayscale `uint8`. Pipeline tiền xử lý:

```text
img_gray
   |
   v
cv2.blur(img_gray, blur_kernel)
   |
   v
background estimate
   |
   v
cv2.subtract(img_gray, background)
   |
   v
cv2.normalize(..., 0, 255)
   |
   v
CLAHE
   |
   v
img_processed
```

Config hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `blur_kernel` | `[101, 101]` | Kernel ước lượng nền chậm biến thiên |
| `clahe_clip_limit` | 2.5 | Giới hạn khuếch đại tương phản |
| `clahe_grid` | `[8, 8]` | Lưới CLAHE |

### 2.1. Vì sao cần tiền xử lý?

Trong cảm biến marker quang học, ảnh thường bị:

- Vignetting: vùng giữa và vùng rìa có độ sáng khác nhau.
- Drift ánh sáng theo thời gian.
- Nền không đồng đều do LED/camera.
- Marker có độ sáng không đều giữa các vùng ảnh.

Nếu đưa ảnh gốc vào detector, một ngưỡng sáng cố định có thể phát hiện tốt marker ở giữa ảnh nhưng bỏ sót marker ở rìa. Tiền xử lý làm nền đồng đều hơn và tăng tương phản marker-nền.

### 2.2. Vì sao dùng box blur để ước lượng nền?

`cv2.blur` với kernel lớn đóng vai trò low-pass filter. Marker là chi tiết nhỏ/tần số cao, còn nền ánh sáng là thành phần biến thiên chậm/tần số thấp. Kernel `[101, 101]` lớn hơn nhiều so với kích thước marker nên ảnh blur chủ yếu giữ lại nền, không giữ chi tiết marker.

Sau đó:

```text
normalized = img_gray - background
```

Trong code dùng `cv2.subtract`, với ảnh `uint8` phép trừ bị clamp tại 0. Điều này phù hợp vì marker sáng hơn nền sẽ được giữ lại, còn vùng tối hơn nền bị đưa về 0.

### 2.3. Vì sao dùng normalize sau background subtraction?

Sau khi trừ nền, dải giá trị ảnh có thể hẹp hoặc khác nhau giữa các frame. `cv2.normalize(..., 0, 255)` đưa ảnh về cùng dải `[0, 255]`, giúp các tham số detector ổn định hơn giữa các frame/session.

### 2.4. Vì sao dùng CLAHE?

CLAHE là Contrast Limited Adaptive Histogram Equalization. Nó tăng tương phản cục bộ nhưng giới hạn mức khuếch đại bằng `clipLimit`. Lý do dùng CLAHE thay vì histogram equalization toàn cục:

- Marker ở các vùng khác nhau cần tăng tương phản khác nhau.
- Histogram equalization toàn cục dễ khuếch đại nhiễu ở vùng nền phẳng.
- CLAHE làm cục bộ nên marker ở rìa ảnh vẫn nổi lên.
- `clipLimit=2.5` hạn chế việc nhiễu bị khuếch đại quá mức.

### 2.5. Vì sao có `make_clahe`?

`make_clahe(config)` tạo sẵn object CLAHE để tái sử dụng. Trong realtime, nếu tạo CLAHE mới ở mỗi frame sẽ tốn overhead không cần thiết. Hàm `preprocess` nhận `_clahe` tùy chọn để dùng lại object này.

## 3. Phát hiện marker trong `detection.py`

Module `detection.py` dùng OpenCV `SimpleBlobDetector` để phát hiện marker sáng trên nền tối.

Luồng:

```text
img_processed
        |
        v
SimpleBlobDetector.detect
        |
        v
keypoints
        |
        v
centers = [[kp.pt[0], kp.pt[1]], ...]
```

Config hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `min_threshold` | 50 | Ngưỡng thấp nhất khi quét threshold |
| `max_threshold` | 220 | Ngưỡng cao nhất khi quét threshold |
| `step` | 10 | Bước quét threshold |
| `min_area` | 30.0 | Diện tích blob nhỏ nhất |
| `max_area` | 500.0 | Diện tích blob lớn nhất |
| `min_circularity` | 0.5 | Độ tròn tối thiểu |
| `min_inertia` | 0.3 | Tỷ lệ quán tính tối thiểu |
| `min_convexity` | 0.7 | Độ lồi tối thiểu |

### 3.1. Vì sao dùng `SimpleBlobDetector`?

Marker là các blob sáng có hình dạng tương đối tròn. `SimpleBlobDetector` phù hợp vì tích hợp nhiều bước trong một detector:

- Quét nhiều ngưỡng thay vì chỉ dùng một threshold cố định.
- Lọc theo màu blob sáng.
- Lọc theo diện tích.
- Lọc theo độ tròn.
- Lọc theo inertia ratio.
- Lọc theo convexity.

Nếu tự dùng `threshold + findContours`, cần chọn một ngưỡng duy nhất và tự viết nhiều bộ lọc hình học. Trong điều kiện ánh sáng không đều, multi-threshold của `SimpleBlobDetector` ổn định hơn.

### 3.2. Vì sao lọc `blobColor=255`?

Sau tiền xử lý, marker cần phát hiện là vùng sáng trên nền tối. Lọc `blobColor=255` giúp bỏ qua các blob tối hoặc artifact nền không phải marker.

### 3.3. Vì sao lọc diện tích?

Diện tích giúp loại:

- Nhiễu nhỏ.
- Điểm sáng không phải marker.
- Vùng sáng lớn do phản xạ hoặc marker bị dính.

`min_area=30` giữ marker nhỏ hoặc hơi mờ, còn `max_area=500` loại vùng sáng quá lớn. Các giá trị này phụ thuộc setup camera/marker, nên nếu thay đổi khoảng cách camera hoặc kích thước marker cần hiệu chỉnh lại.

### 3.4. Vì sao dùng circularity, inertia, convexity?

Marker lý tưởng gần tròn. Nhưng khi gel biến dạng hoặc ảnh nhiễu, marker có thể hơi méo. Vì vậy ngưỡng không đặt quá chặt:

| Bộ lọc | Lý do |
|---|---|
| Circularity | Giữ blob gần tròn, loại vệt dài hoặc artifact |
| Inertia ratio | Loại blob quá dẹt |
| Convexity | Loại hình dạng lõm/phức tạp, ví dụ hai blob dính nhau |

Các ngưỡng hiện tại tương đối mềm để không loại nhầm marker bị méo nhẹ.

### 3.5. Vì sao detector có thể được truyền qua `_detector`?

`detect_markers` nhận `_detector` tùy chọn. Trong realtime, tạo `SimpleBlobDetector` liên tục sẽ tốn overhead. Tạo một lần rồi truyền vào giúp pipeline nhanh và ổn định hơn.

## 4. Tracking marker trong `tracking.py`

`tracking.py` theo dõi marker từ ảnh reference sang ảnh biến dạng bằng Pyramid Lucas-Kanade optical flow.

Input:

```text
img_ref:    ảnh reference grayscale
img_def:    ảnh deformed grayscale
ref_points: tâm marker ở reference, shape (N, 2)
```

Output:

```text
pts_def: tọa độ marker sau tracking, shape (N, 2)
valid:   mask marker tracking hợp lệ, shape (N,)
```

Luồng:

```text
ref_points
    |
    v
LK forward: img_ref -> img_def
    |
    v
pts_def
    |
    v
LK backward: img_def -> img_ref
    |
    v
pts_back
    |
    v
forward-backward error
    |
    v
valid mask
    |
    v
optional deadzone
```

Config hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `tracking.pyrlk.win_size` | `[21, 21]` | Cửa sổ LK quanh mỗi marker |
| `tracking.pyrlk.max_level` | 3 | Số level pyramid |
| `tracking.pyrlk.fb_threshold` | 2.0 | Ngưỡng lỗi forward-backward |
| `tracking.pyrlk.term_criteria.max_iter` | 30 | Số vòng lặp LK tối đa |
| `tracking.pyrlk.term_criteria.eps` | 0.01 | Điều kiện dừng theo hội tụ |
| `tracking.min_displacement` | 1 | Deadzone cho dịch chuyển nhỏ |

### 4.1. Vì sao dùng Lucas-Kanade?

Lucas-Kanade phù hợp vì:

- Marker đã được phát hiện ở reference, nên tracking chỉ cần theo dõi điểm đã biết.
- Chuyển động giữa hai frame thường nhỏ hoặc vừa phải.
- LK nhanh, có sẵn trong OpenCV và hỗ trợ sub-pixel.
- Pyramid LK xử lý được chuyển động lớn hơn LK cơ bản.

So với template matching, LK nhanh hơn vì không phải quét toàn ảnh cho từng marker. So với deep learning keypoint tracking, LK đơn giản hơn, không cần dữ liệu gán nhãn và đủ tốt cho marker tròn trong setup cố định.

### 4.2. Vì sao dùng pyramid LK?

LK cơ bản dựa trên xấp xỉ tuyến tính nên chỉ tốt với dịch chuyển nhỏ. Pyramid LK xây ảnh nhiều mức:

```text
coarse level -> estimate motion lớn
fine level   -> tinh chỉnh sub-pixel
```

Với `max_level=3`, thuật toán có thể bắt được motion lớn hơn so với chỉ chạy trên ảnh gốc. Điều này hữu ích khi marker dịch chuyển nhiều do lực tác động mạnh.

### 4.3. Vì sao `win_size=[21,21]`?

Cửa sổ LK phải đủ lớn để chứa cấu trúc marker và gradient xung quanh, nhưng không quá lớn để lẫn nhiều marker lân cận. `21x21` là cân bằng thực dụng:

- Đủ chứa marker đường kính khoảng 10-15 px.
- Có thêm vùng nền quanh marker để LK có gradient.
- Không quá lớn để tăng chi phí hoặc lẫn quá nhiều cấu trúc khác.

### 4.4. Forward-backward check là gì?

Code chạy LK hai chiều:

```text
Forward:  ref -> def
Backward: def -> ref
```

Nếu tracking đúng, điểm quay lại `pts_back` phải gần điểm ban đầu `ref_points`. Sai số:

```text
fb_error = ||pts_back - ref_points||
```

Marker được coi là hợp lệ khi:

```text
status_forward == 1
status_backward == 1
fb_error < fb_threshold
```

Với `fb_threshold=2.0`, marker có lỗi đi-về dưới 2 pixel được giữ.

### 4.5. Vì sao cần forward-backward check?

LK có thể trôi sang vùng tương tự, nhất là khi marker giống nhau hoặc ảnh bị motion blur. Forward-backward check loại nhiều trường hợp tracking không nhất quán. Đây là cách kiểm soát chất lượng tracking mà không cần ground truth.

Hạn chế: nếu tracking sai nhưng sai một cách nhất quán cả hai chiều, FB check vẫn có thể không phát hiện được. Đây là hạn chế cố hữu của optical flow.

### 4.6. Deadzone là gì và vì sao cần?

Nếu `apply_deadzone=True`, các dịch chuyển nhỏ hơn `tracking.min_displacement` sẽ bị reset về vị trí reference:

```text
if ||pts_def - ref_points|| < min_displacement:
    pts_def = ref_points
```

Lý do:

- Ảnh có nhiễu sub-pixel.
- Marker có thể rung nhẹ dù không có lực đáng kể.
- Gel đàn hồi có thể có dao động nhỏ.

Deadzone giảm false positive khi chỉ muốn quan sát chuyển động đáng kể. Tuy nhiên, một số thuật toán cần tín hiệu trượt chậm tích lũy sẽ gọi `track_markers_lk(..., apply_deadzone=False)`.

## 5. Visualization trong `visualization.py`

`visualize_flow_arrows` vẽ mũi tên từ vị trí reference sang vị trí deformed.

Input:

```text
img
ref_pts
def_pts
valid
config
```

Luồng:

```text
for each marker:
    if not valid: skip
    disp = def_pt - ref_pt
    mag = norm(disp)
    if mag < visualization.motion_magnitude_threshold:
        draw small non-moving circle
    else:
        draw arrow ref -> ref + disp * arrow_scale
```

Config hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `arrow_scale` | 3.0 | Phóng đại mũi tên để dễ nhìn |
| `motion_magnitude_threshold` | 0.3 | Ngưỡng vẽ marker đứng yên |
| `arrow_thickness` | 1 | Độ dày mũi tên |
| `arrow_tip_length` | 0.3 | Độ dài đầu mũi tên |
| `color_intensity_multiplier` | 20 | Đổi màu theo độ lớn displacement |

### Vì sao mũi tên được scale?

Displacement marker có thể chỉ vài pixel, khó nhìn trên ảnh. `arrow_scale=3.0` chỉ dùng cho trực quan hóa, không thay đổi dữ liệu thật. Khi đưa vào báo cáo cần ghi chú mũi tên đã được phóng đại.

### Vì sao màu mũi tên phụ thuộc magnitude?

Code tính:

```text
color_intensity = min(int(mag * multiplier), 255)
color = (0, 255 - color_intensity, color_intensity)
```

Marker dịch chuyển nhỏ có màu gần xanh, marker dịch chuyển lớn chuyển dần sang đỏ. Điều này giúp nhìn nhanh vùng biến dạng mạnh.

## 6. Kiến trúc tổng thể khi dùng thực tế

Trong một pipeline xử lý marker điển hình:

```text
Reference image
    |
    v
preprocess(reference)
    |
    v
detect_markers(reference_processed)
    |
    v
ref_pts

New frame
    |
    v
preprocess(frame)
    |
    v
track_markers_lk(reference_processed, frame_processed, ref_pts)
    |
    v
def_pts, valid
    |
    v
disp = def_pts - ref_pts
    |
    v
force/slip/visualization modules
```

`src/core` vì vậy đóng vai trò chuyển ảnh camera thành trường dịch chuyển marker. Trường dịch chuyển này là đầu vào trực tiếp hoặc gián tiếp cho:

- `force_poly`: rút gọn thành 9 feature.
- `force_model`: dùng vị trí marker và displacement.
- Slip detection: phân tích hướng và vận tốc marker.
- Visualization: vẽ flow arrows.

## 7. Vì sao thiết kế này hợp lý?

### 7.1. Tận dụng đặc tính phần cứng

Hệ tactile marker có cấu trúc rất rõ: marker sáng, gần tròn, nằm trên nền tối, camera cố định. Vì vậy các thuật toán cổ điển như background subtraction, blob detection và LK tracking đủ mạnh, nhanh và dễ kiểm soát.

### 7.2. Tách reference và frame biến dạng

Marker được detect chủ yếu trên ảnh reference, sau đó tracking sang frame mới. Cách này tránh phải detect lại marker ở mọi frame, giảm nhiễu do detector thay đổi thứ tự hoặc bỏ sót marker.

### 7.3. Có cơ chế kiểm soát chất lượng

Pipeline không chỉ trả vị trí marker mà còn trả `valid mask`. Mask này rất quan trọng vì các mô hình phía sau có thể bỏ marker tracking lỗi thay vì dùng dữ liệu sai.

### 7.4. Phù hợp realtime

Các lựa chọn đều thiên về tốc độ:

- Box blur thay vì Gaussian lớn.
- Cache CLAHE object.
- Cache SimpleBlobDetector.
- LK thay vì matching toàn cục.
- Visualization đơn giản bằng OpenCV.

## 8. Hạn chế của `src/core`

| Hạn chế | Nguyên nhân | Hướng xử lý |
|---|---|---|
| Phụ thuộc ánh sáng | Detector dựa vào blob sáng | Kiểm soát LED, tăng tiền xử lý, augment dữ liệu |
| Phụ thuộc kích thước marker pixel | Area/circularity threshold cố định | Hiệu chỉnh config khi đổi camera/setup |
| LK có thể drift | Marker giống nhau, motion blur, texture nghèo | Forward-backward check, re-detect định kỳ, marker có pattern phân biệt |
| Không xử lý tốt occlusion nặng | Marker bị che mất | Dùng valid mask, bỏ frame kém, thêm detector/reassociation |
| Deadzone có thể làm mất tín hiệu nhỏ | Dịch chuyển nhỏ bị reset | Tắt `apply_deadzone` cho thuật toán cần tín hiệu tích lũy |
| Visualization có scale giả | Mũi tên được phóng đại | Ghi chú rõ scale khi báo cáo |

## 9. Câu hỏi có thể gặp khi phản biện

### Q1. `src/core` có phải là mô hình học máy không?

Không. `src/core` là pipeline xử lý ảnh cổ điển gồm tiền xử lý, phát hiện blob, optical flow Lucas-Kanade và visualization. Nó tạo ra trường dịch chuyển marker để các mô hình force hoặc thuật toán slip sử dụng.

### Q2. Vì sao cần tiền xử lý ảnh trước khi detect marker?

Ảnh tactile thường bị chiếu sáng không đều và vignetting. Nếu detect trực tiếp trên ảnh gốc, marker ở vùng tối dễ bị bỏ sót. Tiền xử lý khử nền chậm biến thiên và tăng tương phản cục bộ, giúp detector ổn định hơn.

### Q3. Vì sao dùng box blur thay vì Gaussian blur?

Mục tiêu là ước lượng nền ánh sáng chậm biến thiên, không cần trọng số Gaussian chính xác. Box blur nhanh hơn nhiều với kernel lớn và đủ tốt để loại thành phần nền. Marker là chi tiết nhỏ nên bị làm mờ khỏi ảnh nền.

### Q4. Vì sao kernel blur là 101x101?

Kernel phải lớn hơn marker nhiều lần để marker không ảnh hưởng đáng kể đến ảnh nền ước lượng. Nếu kernel quá nhỏ, marker sẽ bị đưa vào nền và phép trừ tạo artifact. Nếu quá lớn, chi phí tăng và nền có thể bị làm mượt quá mức. 101x101 là lựa chọn thực nghiệm cân bằng với kích thước marker hiện tại.

### Q5. Vì sao dùng CLAHE?

CLAHE tăng tương phản cục bộ và giới hạn khuếch đại nhiễu. Điều này phù hợp với ảnh có độ sáng không đồng đều: marker ở vùng tối vẫn được tăng tương phản mà không làm toàn ảnh bị nhiễu như histogram equalization toàn cục.

### Q6. Vì sao không dùng threshold cố định để detect marker?

Threshold cố định nhạy với ánh sáng và vị trí ảnh. `SimpleBlobDetector` quét nhiều threshold và lọc hình học, nên ổn định hơn khi marker có độ sáng khác nhau giữa các vùng ảnh.

### Q7. Vì sao dùng `SimpleBlobDetector` thay vì YOLO/keypoint network?

Marker có hình dạng đơn giản, setup camera cố định và không cần semantic phức tạp. `SimpleBlobDetector` nhanh, không cần dữ liệu gán nhãn, dễ chỉnh tham số và đủ tốt cho bài toán này. Deep detector có thể mạnh hơn nhưng tăng chi phí và cần dataset nhãn.

### Q8. Vì sao phải lọc diện tích blob?

Diện tích giúp loại nhiễu nhỏ và vùng sáng lớn không phải marker. Marker vật lý có kích thước pixel nằm trong một khoảng dự kiến, nên area filtering là ràng buộc hình học tự nhiên.

### Q9. Vì sao ngưỡng circularity không đặt quá cao?

Marker lý tưởng tròn, nhưng khi gel biến dạng hoặc ảnh mờ, marker có thể hơi méo. Nếu circularity quá cao, detector sẽ bỏ sót marker thật. Ngưỡng 0.5 tương đối mềm để chấp nhận biến dạng nhẹ.

### Q10. Vì sao detect marker trên reference rồi track, thay vì detect lại mỗi frame?

Detect lại mỗi frame có thể làm thay đổi thứ tự marker, bỏ sót marker hoặc tạo marker giả. Tracking từ reference giữ được identity của từng marker theo thời gian, cần thiết để tính displacement chính xác.

### Q11. Vì sao dùng Lucas-Kanade?

Lucas-Kanade nhanh, có sẵn trong OpenCV, hỗ trợ sub-pixel và phù hợp khi đã biết vị trí điểm ở reference. Với marker tactile giữa các frame liên tiếp, chuyển động thường đủ nhỏ để LK hoạt động tốt, đặc biệt khi dùng pyramid.

### Q12. Vì sao cần pyramid trong LK?

LK cơ bản chỉ ổn với chuyển động nhỏ. Pyramid xử lý ảnh ở nhiều độ phân giải, bắt chuyển động lớn ở mức coarse rồi tinh chỉnh ở mức fine. Điều này giúp tracking ổn định hơn khi lực làm marker dịch chuyển nhiều.

### Q13. Vì sao dùng forward-backward check?

Forward-backward check kiểm tra tính nhất quán của tracking. Nếu track từ reference sang frame rồi track ngược lại không về gần điểm ban đầu, marker đó nhiều khả năng bị tracking lỗi và được đánh invalid.

### Q14. Forward-backward check có phát hiện mọi lỗi tracking không?

Không. Nếu tracking sai nhưng nhất quán cả hai chiều, FB error vẫn có thể nhỏ. Tuy nhiên nó loại được nhiều lỗi drift phổ biến và cung cấp valid mask hữu ích cho các bước sau.

### Q15. Vì sao `fb_threshold=2.0` pixel?

Ngưỡng này cho phép sai số sub-pixel và nhiễu nhỏ nhưng vẫn loại các track trôi xa. Nếu đặt quá nhỏ, nhiều marker hợp lệ bị loại; nếu quá lớn, marker sai vẫn được giữ. 2 pixel là mức cân bằng thực nghiệm.

### Q16. Vì sao cần `valid mask`?

Không phải marker nào cũng tracking đúng. `valid mask` cho phép các module sau bỏ qua marker lỗi. Ví dụ `force_model` dùng mask-aware pooling, `force_poly` chỉ tính feature trên marker valid.

### Q17. Deadzone có tác dụng gì?

Deadzone reset các dịch chuyển rất nhỏ về 0 để loại rung/nhiễu khi không có chuyển động đáng kể. Nó giúp visualization và một số pipeline force ổn định hơn.

### Q18. Khi nào không nên dùng deadzone?

Không nên dùng deadzone khi cần giữ tín hiệu nhỏ tích lũy theo thời gian, ví dụ slip chậm. Vì vậy `track_markers_lk` có tham số `apply_deadzone=False` cho các thuật toán cần độ nhạy cao.

### Q19. Nếu marker bị mất tracking nhiều thì sao?

Pipeline vẫn trả các marker còn valid. Các bước sau cần kiểm tra tỷ lệ valid marker. Nếu valid quá thấp, dự đoán lực hoặc slip có thể kém tin cậy và nên bỏ frame hoặc báo lỗi.

### Q20. Visualization có làm thay đổi dữ liệu không?

Không. `visualize_flow_arrows` chỉ vẽ mũi tên lên ảnh copy để quan sát. `arrow_scale` chỉ phóng đại mũi tên hiển thị, không thay đổi displacement dùng cho tính toán.

### Q21. Vì sao màu mũi tên thay đổi theo độ lớn displacement?

Màu giúp người xem nhận ra nhanh vùng biến dạng mạnh. Marker dịch chuyển lớn chuyển dần sang đỏ, marker ít dịch chuyển giữ màu nhạt/xanh hơn.

### Q22. Pipeline có chạy realtime được không?

Có, thiết kế hướng đến realtime: detector và CLAHE có thể cache, LK nhanh hơn matching toàn cục, và visualization dùng OpenCV trực tiếp. Tốc độ thực tế còn phụ thuộc số marker, độ phân giải và CPU/GPU.

### Q23. Pipeline có phụ thuộc setup camera không?

Có. Các tham số area, threshold, blur kernel và LK window phụ thuộc kích thước marker theo pixel, ánh sáng và độ phân giải. Khi thay đổi camera/lens/khoảng cách, cần kiểm tra và chỉnh lại config.

### Q24. Vì sao bỏ qua calibration trong tài liệu này có ổn không?

Calibration liên quan đến hiệu chỉnh méo camera và ánh xạ hình học, nhưng pipeline lõi ở đây vẫn hoạt động trong hệ tọa độ pixel. Theo yêu cầu, tài liệu tập trung vào xử lý ảnh, detection, tracking và visualization, không phân tích `calibration.py`.

### Q25. Điểm mạnh chính của `src/core` là gì?

Điểm mạnh là đơn giản, nhanh, dễ debug và tận dụng tốt cấu trúc marker vật lý. Pipeline tạo được displacement field có valid mask, đủ để phục vụ cả mô hình force và thuật toán slip.

### Q26. Điểm yếu chính của `src/core` là gì?

Điểm yếu là phụ thuộc nhiều vào chất lượng ảnh và tham số thủ công. Khi ánh sáng, camera, kích thước marker hoặc điều kiện tiếp xúc thay đổi mạnh, cần hiệu chỉnh lại config hoặc bổ sung phương pháp robust hơn.

### Q27. Có thể cải tiến pipeline thế nào?

Có thể thêm re-detection định kỳ, re-association marker bằng Hungarian matching, lọc outlier theo lân cận không gian, dùng marker có pattern phân biệt, thêm confidence score cho tracking hoặc kết hợp optical flow dense/CNN khi ảnh khó.

## 10. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn như sau:

> Module `src/core` đảm nhiệm chuỗi xử lý ảnh nền tảng cho cảm biến tactile marker. Ảnh grayscale đầu vào được tiền xử lý bằng trừ nền dựa trên box blur, chuẩn hóa dải sáng và CLAHE để giảm ảnh hưởng chiếu sáng không đồng đều. Marker trên ảnh reference được phát hiện bằng `SimpleBlobDetector` với các bộ lọc ngưỡng, diện tích và hình học. Từ các tâm marker reference, hệ thống dùng Pyramid Lucas-Kanade để tracking marker sang frame biến dạng, đồng thời dùng forward-backward check để loại các track không ổn định. Kết quả là tọa độ marker biến dạng và mask hợp lệ, từ đó tính được trường dịch chuyển marker phục vụ ước lượng lực, phát hiện trượt và trực quan hóa.

Sơ đồ nên vẽ:

```text
Raw grayscale image
        |
        v
Background subtraction + normalize + CLAHE
        |
        v
Processed reference image
        |
        v
SimpleBlobDetector
        |
        v
Reference marker centers
        |
        v
Pyramid Lucas-Kanade tracking
        |
        v
Forward-backward validation
        |
        v
Deformed marker centers + valid mask
        |
        v
Displacement field / visualization / force models
```

