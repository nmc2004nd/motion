# Kiến Trúc Slip V1 Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code `src/slip/v1.py` và pipeline gọi nó trong `src/pipelines/realtime_slip_v1.py`. Trong repo hiện tại, module V1 nằm ở file `src/slip/v1.py`, không phải thư mục `src/slip/v1`.

Slip V1 là detector phát hiện trượt dựa trên trường chuyển động marker. Nó không phải mô hình học máy, mà là thuật toán phân tích hình học/thống kê trên vector chuyển động của marker.

## 1. Vai trò của Slip V1

Slip V1 phát hiện trượt bằng giả thuyết vật lý:

```text
Khi trượt xảy ra, nhiều marker có xu hướng chuyển động cùng hướng.
Khi chỉ nhấn/thả hoặc biến dạng đàn hồi, hướng chuyển động thường phân tán hoặc mang dạng xuyên tâm.
```

Vì vậy, detector đo mức độ đồng hướng của các vector chuyển động marker bằng MRVL:

```text
MRVL = Mean Resultant Vector Length
```

Luồng tổng quát:

```text
Reference markers
        |
        v
Core tracking LK
  tracked markers + valid mask
        |
        v
RealtimeSlipTracking
  lưu marker_history
  lấy past frame và current frame
        |
        v
SlipDetector.calculate_slip_probability
  lọc marker valid
  lọc chuyển động nhỏ
  lọc rebound
  khử radial press/release
  tính weighted MRVL
  boost bằng translation/motion/participation
  EMA smoothing
        |
        v
is_slip, r_value, phase, direction, moving_count
```

## 2. File và thành phần chính

| File | Vai trò |
|---|---|
| `src/slip/v1.py` | Định nghĩa lớp `SlipDetector`, tính xác suất trượt |
| `src/pipelines/realtime_slip_v1.py` | Pipeline realtime, giữ history buffer và gọi detector |
| `config/pipeline_config.yaml` | Chứa các tham số `slip_detection.*` |
| `src/core/tracking.py` | Tracking marker bằng LK, cung cấp tọa độ và valid mask |

Config hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `slip_threshold` | 0.8 | Ngưỡng `r_value` để kết luận slip |
| `min_motion_thresh` | 2.0 px | Marker phải dịch chuyển đủ lớn mới được xét |
| `min_moving_markers` | 5 | Số marker chuyển động tối thiểu |
| `alpha` | 0.3 | Hệ số EMA cơ bản khi score tăng |
| `alpha_decay` | 0.6 | Hệ số decay khi score giảm hoặc thiếu evidence |
| `rebound_dot_prod_threshold` | -0.1 | Ngưỡng lọc chuyển động hồi đàn hồi |
| `press_rate_threshold` | 0.5 px/frame | Ngưỡng phát hiện đang nhấn |
| `history_buffer_length` | 5 frame | Số frame history do pipeline realtime giữ |
| `moving_count_thresh` | 2 | Ngưỡng hiển thị trạng thái moving trong overlay |

Lưu ý: `history_buffer_length` và `moving_count_thresh` không được dùng trực tiếp trong `SlipDetector`. `history_buffer_length` được dùng trong `RealtimeSlipTracking` để chọn frame quá khứ; `moving_count_thresh` dùng cho màu hiển thị overlay.

## 3. Input và output của `SlipDetector`

Hàm chính:

```python
calculate_slip_probability(
    prev_markers,
    current_markers,
    valid_mask,
    ref_markers=None,
)
```

Input:

| Input | Shape | Ý nghĩa |
|---|---:|---|
| `prev_markers` | `(N, 2)` | Tọa độ marker ở frame quá khứ |
| `current_markers` | `(N, 2)` | Tọa độ marker ở frame hiện tại |
| `valid_mask` | `(N,)` | Marker tracking hợp lệ |
| `ref_markers` | `(N, 2)` hoặc `None` | Tọa độ marker trạng thái tĩnh ban đầu |

Output là dict:

| Key | Ý nghĩa |
|---|---|
| `is_slip` | True nếu `r_value > slip_threshold` |
| `r_value` | Score đã làm mượt bằng EMA |
| `raw_r_value` | MRVL thô trước boost và smoothing |
| `raw_score` | Score sau khi boost bằng translation/motion/participation |
| `translation` | Vector tịnh tiến trung bình |
| `translation_mag` | Độ lớn vector tịnh tiến |
| `mean_motion` | Độ lớn chuyển động trung bình |
| `pressing` | Có đang ở pha nhấn không |
| `mean_direction` | Hướng trượt trung bình nếu đang slip |
| `moving_count` | Số marker chuyển động đủ lớn |
| `phase` | `"slip"`, `"tracking"`, `"pressing"`, `"no_markers"`, `"insufficient_motion"` |

## 4. Vai trò của history buffer trong pipeline

`SlipDetector` không tự lưu chuỗi tọa độ marker. Pipeline `RealtimeSlipTracking` giữ:

```python
self.marker_history: list[np.ndarray]
```

Ở mỗi frame:

1. Tracking marker hiện tại bằng `track_markers_lk`.
2. Append `tracked` vào `marker_history`.
3. Nếu history dài hơn `history_buffer_length`, bỏ frame cũ nhất.
4. Khi có ít nhất 2 frame, lấy:

```text
past = marker_history[0]
current = tracked
```

và gọi detector.

Với `history_buffer_length = 5`, detector so sánh frame hiện tại với frame cách đó tối đa khoảng 4 frame. Cách này làm chuyển động slip rõ hơn so với so sánh frame liền kề, vì slip nhỏ có thể tích lũy qua vài frame.

Pipeline không seed history bằng reference khi vừa capture reference để tránh velocity spike ở frame đầu.

## 5. Các bước xử lý trong Slip V1

### 5.1. Lọc marker invalid

Nếu không có marker hợp lệ:

```text
if not valid_mask.any():
    decay score
    return phase = "no_markers"
```

Lý do: marker invalid là marker tracking lỗi. Nếu dùng chúng, hướng chuyển động có thể sai và tạo false slip.

### 5.2. Phát hiện pha nhấn bằng mean deformation

Nếu có `ref_markers`, detector tính biến dạng so với reference:

```text
deformation = current_markers - ref_markers
mean_def_mag = mean(||deformation_i||)
delta_def = mean_def_mag - prev_mean_deformation
pressing = delta_def > press_rate_threshold
```

Ý nghĩa:

- Khi đang nhấn, độ biến dạng trung bình tăng nhanh.
- Nếu tăng quá `0.5 px/frame`, detector xem đây là pha pressing.
- Pressing có thể tạo nhiều marker chuyển động, nhưng không nhất thiết là slip.

Detector chỉ suppress pressing nếu chuyển động tịnh tiến yếu. Nếu vừa nhấn vừa trượt rõ, tín hiệu translation vẫn được cho qua.

### 5.3. Tính displacement giữa hai frame

Vector chuyển động marker:

```text
displacements = current_markers - prev_markers
magnitudes = ||displacements||
```

Chỉ giữ marker có:

```text
magnitude > min_motion_thresh
```

Với config hiện tại:

```text
min_motion_thresh = 2.0 px
```

Lý do: chuyển động nhỏ có thể là nhiễu tracking, rung nhẹ hoặc dao động đàn hồi. Nếu đưa vào MRVL, các vector nhỏ có hướng không ổn định sẽ làm score nhiễu.

### 5.4. Rebound filter

Khi có `ref_markers`, detector lọc marker đang hồi đàn hồi:

```text
deformation = current - ref
dot = displacement · deformation
rebound nếu dot < rebound_dot_prod_threshold
```

Nếu `dot` âm, vector chuyển động đang đi ngược hướng biến dạng hiện tại. Đây thường là dấu hiệu gel/marker đang hồi về vị trí ban đầu sau khi bị biến dạng, không phải slip chủ động.

Config hiện tại:

```text
rebound_dot_prod_threshold = -0.1
```

Ngưỡng âm nhỏ giúp loại chuyển động hồi rõ rệt nhưng không quá chặt với nhiễu nhỏ.

### 5.5. Kiểm tra số marker chuyển động

Sau các bước lọc, nếu:

```text
moving_count < min_moving_markers
```

detector decay score và trả:

```text
phase = "insufficient_motion"
```

Với `min_moving_markers = 5`, thuật toán yêu cầu ít nhất 5 marker cùng có chuyển động đáng kể. Lý do: nếu chỉ vài marker chuyển động, đó có thể là nhiễu tracking hoặc tiếp xúc cục bộ không đủ tin cậy.

### 5.6. Khử thành phần radial press/release

Nếu có `ref_markers`, detector gọi:

```python
_remove_radial_component(significant_disp, ref_moving)
```

Ý tưởng:

```text
displacement ≈ translation + radial_scale * centered_ref
```

Trong đó:

- `translation` là chuyển động tịnh tiến trung bình, liên quan tới slip.
- `radial_scale * centered_ref` là thành phần xuyên tâm do press/release.

Code ước lượng `radial_scale` bằng least-squares đơn giản rồi trừ thành phần radial:

```text
clean_disp = displacement - radial_scale * centered_ref
```

Lý do: khi nhấn/thả, marker có thể di chuyển theo dạng nở/co lại quanh tâm. Dạng này không phải slip nhưng có thể tạo nhiều vector có cấu trúc. Khử radial giúp MRVL tập trung hơn vào thành phần trượt tịnh tiến.

### 5.7. Tính Weighted MRVL

Với vector sau lọc `clean_disp`, detector tính:

```text
angle_i = atan2(dy_i, dx_i)
weight_i = ||disp_i|| / sum_j ||disp_j||
R = sqrt((sum_i weight_i*cos(angle_i))^2 + (sum_i weight_i*sin(angle_i))^2)
```

Ý nghĩa:

| Giá trị R | Diễn giải |
|---|---|
| Gần 1 | Các vector chuyển động gần cùng hướng, giống slip tịnh tiến |
| Gần 0 | Hướng chuyển động phân tán, giống press/release/nhiễu |

Vì dùng trọng số theo độ lớn chuyển động, marker dịch chuyển mạnh đóng góp nhiều hơn marker dịch chuyển yếu. Điều này hợp lý vì slip thật thường tạo displacement rõ trên một vùng marker.

### 5.8. Boost score bằng translation, participation và motion

MRVL chỉ đo đồng hướng, nhưng chưa biết chuyển động có đủ mạnh và đủ nhiều marker hay không. Code tạo thêm 3 gain:

```text
translation_gain = saturate(translation_mag / min_motion_thresh)
participation_gain = saturate(moving_count / (2 * min_moving_markers))
motion_gain = saturate(mean_motion / min_motion_thresh)
```

Sau đó:

```text
raw_score = raw_r_value
raw_score *= 0.55 + 0.45 * translation_gain
raw_score *= 0.90 + 0.10 * participation_gain
raw_score *= 0.85 + 0.15 * motion_gain
raw_score = min(raw_score, 1.0)
```

Lý do:

- `translation_gain`: slip thật thường có thành phần tịnh tiến rõ.
- `participation_gain`: càng nhiều marker tham gia thì càng đáng tin.
- `motion_gain`: chuyển động quá yếu thì không nên tạo score cao.

Các hệ số được đặt để MRVL vẫn là thành phần chính, nhưng score nhạy hơn với slip thật và ít bị nhiễu bởi vài marker lẻ.

### 5.9. Pressing gate

Nếu đang pressing và translation yếu:

```text
if pressing and translation_gain < 0.35:
    decay score
    return phase = "pressing"
```

Lý do: nhấn xuống có thể làm nhiều marker chuyển động nhưng không phải trượt. Tuy nhiên code không triệt tiêu mọi trường hợp pressing. Nếu đang nhấn nhưng translation rõ, detector vẫn cho qua vì có thể là vừa nhấn vừa trượt.

### 5.10. EMA smoothing và adaptive alpha

Detector giữ state:

```text
_smoothed_r
_prev_mean_deformation
```

Nếu `raw_score` tăng:

```text
alpha_up = adaptive_alpha(raw_score, raw_r_value, translation_gain)
smoothed = alpha_up * raw_score + (1 - alpha_up) * smoothed
```

Nếu `raw_score` giảm:

```text
smoothed = alpha_decay * raw_score + (1 - alpha_decay) * smoothed
```

Adaptive alpha:

```text
evidence = saturate((raw_score + coherence + translation_gain) / 3)
alpha_up = saturate(alpha + (1 - alpha) * 0.65 * evidence)
```

Lý do:

- Smoothing tránh trạng thái slip bật/tắt liên tục vì nhiễu.
- Khi evidence mạnh, alpha tăng để detector phản ứng nhanh hơn.
- Khi thiếu evidence, score decay dần thay vì rơi đột ngột.

Kết luận slip:

```text
is_slip = smoothed_r > slip_threshold
```

Với config hiện tại:

```text
slip_threshold = 0.8
```

## 6. Vì sao kiến trúc này hợp lý?

### 6.1. Dựa trên đặc trưng vật lý của slip

Slip thường tạo chuyển động tịnh tiến đồng hướng trên nhiều marker. MRVL là cách đo trực tiếp độ đồng hướng này. Nó đơn giản, nhanh và không cần dữ liệu nhãn.

### 6.2. Kết hợp lọc nhiễu và logic vật lý

Detector không chỉ tính MRVL thô. Nó có thêm:

- `valid_mask` để bỏ marker tracking lỗi.
- `min_motion_thresh` để bỏ chuyển động nhỏ.
- `rebound filter` để bỏ hồi đàn hồi.
- `radial removal` để giảm nhầm press/release.
- `pressing gate` để suppress nhấn xuống không trượt.
- EMA để giảm flicker.

Các bước này làm detector thực dụng hơn trong dữ liệu tactile thật.

### 6.3. Rất nhẹ và phù hợp realtime

Thuật toán chỉ dùng vector numpy trên marker, không dùng mạng neural. Chi phí tính toán nhỏ, dễ chạy realtime sau khi đã có marker tracking.

## 7. Hạn chế của Slip V1

| Hạn chế | Nguyên nhân | Hướng xử lý |
|---|---|---|
| Dựa vào tracking marker | Input đến từ LK tracking | Cải thiện tracking, lọc outlier, kiểm tra valid ratio |
| Một thang thời gian | So sánh current với một frame past trong history | Dùng multi-scale như Slip V2 |
| Có thể bỏ sót slow slip | Chuyển động nhỏ dưới `min_motion_thresh` | Giảm ngưỡng hoặc dùng tích lũy đa thang |
| Pressing có thể che slip yếu | Pressing gate suppress khi translation thấp | Dùng phân rã translation/radial rõ hơn như V2 |
| Không học từ dữ liệu | Heuristic threshold thủ công | Thu thập nhãn slip và train model học máy |
| Nhạy với tham số | Threshold phụ thuộc camera/marker/framerate | Tuning theo setup và validation thực nghiệm |
| Không ước lượng mức độ slip vật lý tuyệt đối | Score là xác suất/độ tin cậy heuristic | Cần calibration hoặc nhãn slip severity |

## 8. So sánh nhanh V1 với V2

| Tiêu chí | Slip V1 | Slip V2 |
|---|---|---|
| Ý tưởng chính | MRVL trên hướng chuyển động marker | Phân rã translation/radial đa thang |
| Số thang thời gian | Một history window | Nhiều scale `[1, 3, 9]` |
| Xử lý press | Press-rate gate + radial removal đơn giản | Least-squares decomposition rõ ràng hơn |
| Độ phức tạp | Thấp | Cao hơn |
| Realtime | Rất nhẹ | Vẫn realtime nhưng nhiều tính toán hơn |
| Hạn chế chính | Dễ bỏ sót slow slip hoặc slip khi press | Nhiều tham số hơn |

## 9. Câu hỏi có thể gặp khi phản biện

### Q1. Slip V1 có phải mô hình học máy không?

Không. Slip V1 là thuật toán heuristic/thống kê dựa trên vector chuyển động marker. Nó không cần dữ liệu train, không có tham số học được, chỉ có threshold và state EMA.

### Q2. Ý tưởng chính của Slip V1 là gì?

Khi trượt xảy ra, nhiều marker dịch chuyển theo cùng một hướng tịnh tiến. Slip V1 đo mức độ đồng hướng của các vector chuyển động bằng weighted MRVL. Nếu score sau smoothing vượt ngưỡng, detector kết luận slip.

### Q3. MRVL là gì?

MRVL là độ dài vector tổng trung bình trên vòng tròn. Mỗi vector chuyển động được chuyển thành một góc. Nếu các góc cùng hướng, tổng vector có độ dài gần 1. Nếu hướng phân tán, tổng triệt tiêu nhau và giá trị gần 0.

### Q4. Vì sao dùng weighted MRVL?

Không phải mọi marker đều đáng tin như nhau. Marker dịch chuyển mạnh thường mang nhiều thông tin hơn marker dịch chuyển rất nhỏ. Weighted MRVL cho marker có độ lớn displacement lớn đóng góp nhiều hơn.

### Q5. Vì sao cần `min_motion_thresh`?

Vector chuyển động quá nhỏ thường bị chi phối bởi nhiễu tracking hoặc rung nhẹ. Nếu đưa vào tính hướng, góc của chúng không ổn định và có thể làm sai MRVL. `min_motion_thresh=2px` lọc các chuyển động nhỏ này.

### Q6. Vì sao cần ít nhất 5 marker chuyển động?

Nếu chỉ một vài marker chuyển động, rất khó phân biệt slip thật với nhiễu hoặc tracking lỗi cục bộ. Yêu cầu ít nhất 5 marker giúp bằng chứng slip đáng tin hơn.

### Q7. Vì sao cần valid mask?

Valid mask đến từ tracking LK. Marker invalid có thể bị tracking sai, nếu dùng vào slip detector sẽ tạo hướng chuyển động giả. Slip V1 chỉ dùng marker valid để tránh nhiễu.

### Q8. Rebound filter dùng để làm gì?

Khi gel hồi đàn hồi, marker có thể di chuyển ngược về vị trí reference. Chuyển động này không phải slip. Rebound filter dùng tích vô hướng giữa displacement hiện tại và deformation so với reference để loại các marker đang hồi về.

### Q9. Vì sao dot product âm lại gợi ý rebound?

Nếu displacement trong frame hiện tại ngược hướng với deformation hiện tại, marker đang đi về phía reference. Đó là dấu hiệu hồi đàn hồi/release, không phải trượt tịnh tiến mới.

### Q10. Vì sao cần khử radial component?

Press/release thường tạo biến dạng xuyên tâm quanh vùng tiếp xúc. Thành phần này có thể làm nhiều marker chuyển động nhưng không phải slip. Khử radial component giúp MRVL tập trung vào thành phần tịnh tiến liên quan tới slip.

### Q11. Radial removal trong V1 có phải phân rã đầy đủ không?

Không. V1 dùng xấp xỉ đơn giản: trừ thành phần `radial_scale * centered_ref`. Nó nhẹ và nhanh nhưng không đầy đủ như phân rã least-squares đa thang của V2.

### Q12. Vì sao cần translation gain nếu đã có MRVL?

MRVL chỉ đo hướng có đồng nhất không. Một số nhiễu có thể tạo hướng khá đồng nhất nhưng chuyển động tịnh tiến yếu. Translation gain giúp score cao hơn khi có chuyển động tịnh tiến thật sự rõ.

### Q13. Vì sao cần participation gain?

Slip thật thường ảnh hưởng nhiều marker. Nếu chỉ vài marker tạo MRVL cao, đó có thể là nhiễu. Participation gain tăng độ tin khi nhiều marker cùng tham gia chuyển động.

### Q14. Vì sao cần motion gain?

Nếu chuyển động trung bình quá nhỏ, kết luận slip không đáng tin. Motion gain giảm score khi chuyển động yếu và tăng score khi displacement đủ lớn.

### Q15. Vì sao có pressing gate?

Khi đang nhấn xuống, marker có thể chuyển động nhiều nhưng không phải trượt. Pressing gate suppress score trong trường hợp biến dạng tăng nhanh mà translation yếu. Điều này giảm false positive do press.

### Q16. Nếu vừa nhấn vừa trượt thì V1 có phát hiện được không?

Có thể, nếu translation đủ rõ. Code chỉ suppress pressing khi `translation_gain < 0.35`. Nếu đang nhấn nhưng có thành phần tịnh tiến mạnh, tín hiệu vẫn được cho qua.

### Q17. Vì sao dùng EMA smoothing?

Slip score theo từng frame có thể nhiễu. EMA làm mượt score để tránh trạng thái slip nhấp nháy. Adaptive alpha giúp detector vẫn phản ứng nhanh khi evidence slip mạnh.

### Q18. Vì sao ngưỡng slip là 0.8?

`r_value` nằm trong khoảng 0-1. Ngưỡng 0.8 yêu cầu mức đồng hướng/evidence cao trước khi báo slip, giúp giảm false positive. Ngưỡng này là tham số thực nghiệm và cần tuning nếu thay đổi setup.

### Q19. `history_buffer_length=5` ảnh hưởng thế nào?

Pipeline so sánh frame hiện tại với frame cũ nhất trong buffer. Buffer dài hơn làm displacement tích lũy lớn hơn, dễ bắt slow slip hơn nhưng tăng độ trễ. Buffer ngắn hơn phản ứng nhanh hơn nhưng nhạy với nhiễu.

### Q20. Vì sao không so sánh luôn với reference?

So sánh với reference cho deformation tích lũy, nhưng slip là chuyển động tương đối theo thời gian. Nếu so với reference, press/release có thể lẫn vào slip. So sánh frame quá khứ với frame hiện tại giúp đo vận tốc/chuyển động gần thời điểm hiện tại.

### Q21. Slip V1 có phát hiện slow slip tốt không?

Không phải luôn tốt. Nếu chuyển động giữa các frame quá nhỏ và dưới `min_motion_thresh`, V1 có thể bỏ sót. Dùng history buffer giúp phần nào, nhưng V2 multi-scale phù hợp hơn cho slow slip.

### Q22. Slip V1 có phụ thuộc framerate không?

Có. Với cùng vận tốc trượt, framerate cao làm displacement giữa frame nhỏ hơn; framerate thấp làm displacement lớn hơn. Do đó các ngưỡng pixel/frame như `min_motion_thresh` và `press_rate_threshold` cần phù hợp với framerate.

### Q23. Slip V1 có phụ thuộc số marker không?

Có. Nếu số marker quá ít hoặc valid mask thấp, `moving_count` không đủ và detector không báo slip. Với cảm biến nhiều marker, V1 đáng tin hơn vì có nhiều vector để thống kê hướng.

### Q24. Vì sao phase có thể là `"tracking"` dù đang có chuyển động?

Nếu có chuyển động nhưng score sau smoothing chưa vượt `slip_threshold`, detector trả phase `"tracking"`. Tức là hệ thống vẫn đang theo dõi marker nhưng chưa đủ bằng chứng slip.

### Q25. Khi nào phase là `"insufficient_motion"`?

Khi số marker có displacement lớn hơn `min_motion_thresh` nhỏ hơn `min_moving_markers`. Detector xem evidence không đủ và decay score.

### Q26. Điểm mạnh chính của Slip V1 là gì?

Nó nhẹ, dễ hiểu, không cần train, chạy realtime và bám sát trực giác vật lý: slip tạo chuyển động đồng hướng. Đây là baseline tốt cho phát hiện gross slip.

### Q27. Điểm yếu chính của Slip V1 là gì?

Nó dựa vào threshold thủ công và một thang thời gian, nên có thể bỏ sót slip chậm hoặc các trường hợp slip xảy ra đồng thời với press/release phức tạp. Nó cũng phụ thuộc mạnh vào chất lượng tracking marker.

### Q28. Vì sao cần Slip V2 nếu đã có Slip V1?

Slip V2 giải quyết một số hạn chế của V1 bằng phân rã translation/radial trên nhiều thang thời gian. V2 phát hiện tốt hơn slow slip và tách press/release rõ hơn, nhưng phức tạp hơn.

### Q29. Có thể cải tiến Slip V1 thế nào?

Có thể dùng nhiều history scale, tự động tuning threshold theo framerate, thêm confidence từ LK, dùng local neighborhood để lọc outlier, hoặc học threshold/score bằng dữ liệu gán nhãn slip.

### Q30. Slip V1 có cần ground truth slip để hoạt động không?

Không. Nó hoạt động không cần nhãn slip. Tuy nhiên, để chọn threshold tốt và đánh giá độ chính xác, vẫn nên có dữ liệu thực nghiệm hoặc video được gán nhãn slip.

## 10. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn như sau:

> Slip V1 là thuật toán phát hiện trượt dựa trên độ đồng hướng của chuyển động marker. Từ các vị trí marker đã tracking, hệ thống so sánh tọa độ marker hiện tại với một frame quá khứ trong buffer để lấy vector chuyển động. Các marker invalid, chuyển động quá nhỏ và chuyển động hồi đàn hồi được loại bỏ. Sau đó thuật toán khử thành phần biến dạng xuyên tâm do press/release và tính weighted MRVL trên các vector còn lại. Score MRVL được điều chỉnh thêm bằng độ lớn tịnh tiến, số marker tham gia và độ lớn chuyển động trung bình, rồi làm mượt bằng EMA. Khi score đã làm mượt vượt ngưỡng 0.8, hệ thống kết luận có slip.

Sơ đồ nên vẽ:

```text
Tracked markers over time
        |
        v
History buffer
  past markers + current markers
        |
        v
Valid mask filtering
        |
        v
Motion threshold filtering
        |
        v
Rebound filtering
        |
        v
Radial press/release removal
        |
        v
Weighted MRVL
        |
        v
Translation + participation + motion gains
        |
        v
EMA smoothing
        |
        v
Slip decision
```
