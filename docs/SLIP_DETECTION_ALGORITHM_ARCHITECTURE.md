# Kiến Trúc Thuật Toán Phát Hiện Trượt

Tài liệu này mô tả thuật toán phát hiện trượt đang được triển khai trong `src/slip/v1.py`. Mục tiêu là giải thích thuật toán ở mức có thể đưa vào quyển đồ án: bài toán cần giải quyết, dữ liệu đầu vào, các bước xử lý, công thức tính điểm trượt, cơ chế chống nhiễu và cách hệ thống đưa ra quyết định cuối cùng.

Trong phiên bản này, phát hiện trượt không dùng mô hình học sâu. Thuật toán dựa trên chuyển động của các marker trên bề mặt cảm biến xúc giác. Khi vật tiếp xúc bắt đầu trượt trên bề mặt cảm biến, nhiều marker thường dịch chuyển theo một hướng tương đối đồng nhất. Ngược lại, khi chỉ nhấn xuống hoặc nhả ra, trường chuyển động thường có dạng phân tán hoặc xuyên tâm. Thuật toán khai thác khác biệt này để phân biệt trượt với biến dạng đàn hồi thông thường.

## 1. Mục tiêu của bài toán phát hiện trượt

Trong hệ cảm biến xúc giác có marker, camera quan sát sự thay đổi vị trí marker theo thời gian. Sau bước phát hiện và tracking marker, mỗi frame cung cấp một tập tọa độ:

```text
p_i(t) = (x_i(t), y_i(t)), i = 1..N
```

Trong đó `N` là số marker được phát hiện ở trạng thái tham chiếu. Bài toán phát hiện trượt cần trả lời câu hỏi:

```text
Tại thời điểm t, vật có đang trượt tương đối trên bề mặt cảm biến hay không?
```

Thuật toán trong `SlipDetector` nhận vào tọa độ marker ở một frame quá khứ, tọa độ marker hiện tại, mask marker tracking hợp lệ và tùy chọn tọa độ reference. Kết quả đầu ra là một điểm tin cậy trượt trong khoảng `[0, 1]` và cờ `is_slip`.

## 2. Giả thuyết vật lý

Thuật toán dựa trên ba quan sát chính:

1. Khi có trượt, phần lớn marker bị kéo theo hướng tịnh tiến tương đối giống nhau.
2. Khi chỉ nhấn hoặc nhả, marker thường di chuyển theo dạng xuyên tâm, nở ra hoặc co lại quanh vùng tiếp xúc.
3. Marker tracking có nhiễu, vì vậy chuyển động nhỏ hoặc marker invalid không nên được dùng trực tiếp để quyết định trượt.

Từ đó, thuật toán không chỉ xét độ lớn chuyển động, mà xét cấu trúc hướng của trường chuyển động. Nếu nhiều vector chuyển động có cùng hướng và đủ mạnh, xác suất trượt tăng. Nếu hướng phân tán, chuyển động quá nhỏ hoặc chỉ có vài marker tham gia, xác suất trượt giảm.

## 3. Vị trí trong pipeline realtime

Trong hệ thống realtime, phát hiện trượt nằm sau khối tracking marker:

```text
Camera frame
    |
    v
Preprocessing + reference marker detection
    |
    v
Lucas-Kanade marker tracking
    |
    v
History buffer marker positions
    |
    v
SlipDetector.calculate_slip_probability()
    |
    v
Slip score + slip flag + visualization
```

Các file liên quan:

| File | Vai trò |
|---|---|
| `src/slip/v1.py` | Cài đặt thuật toán phát hiện trượt V1 |
| `src/pipelines/realtime_slip_v1.py` | Pipeline realtime gọi `SlipDetector` |
| `src/core/tracking.py` | Tracking marker bằng Lucas-Kanade |
| `config/pipeline_config.yaml` | Cấu hình ngưỡng và tham số thuật toán |

Pipeline `RealtimeSlipTracking` giữ một `marker_history` gồm các tọa độ marker đã tracking trong vài frame gần nhất. Khi đủ ít nhất hai frame, pipeline lấy frame cũ nhất trong buffer làm `prev_markers` và frame hiện tại làm `current_markers`.

Với cấu hình hiện tại:

```text
history_buffer_length = 5
```

Nghĩa là thuật toán so sánh trạng thái hiện tại với một frame cách tối đa khoảng 4 frame. Cách này giúp chuyển động nhỏ tích lũy rõ hơn so với chỉ so sánh hai frame liên tiếp.

## 4. Đầu vào và đầu ra của thuật toán

Hàm chính trong `src/slip/v1.py`:

```python
calculate_slip_probability(
    prev_markers,
    current_markers,
    valid_mask,
    ref_markers=None,
)
```

### 4.1. Đầu vào

| Biến | Kích thước | Ý nghĩa |
|---|---:|---|
| `prev_markers` | `(N, 2)` | Tọa độ marker ở frame quá khứ |
| `current_markers` | `(N, 2)` | Tọa độ marker ở frame hiện tại |
| `valid_mask` | `(N,)` | Đánh dấu marker tracking hợp lệ |
| `ref_markers` | `(N, 2)` hoặc `None` | Tọa độ marker ở trạng thái reference |

`valid_mask` rất quan trọng vì không phải marker nào cũng tracking thành công. Marker invalid có thể tạo vector chuyển động sai, làm nhiễu quyết định trượt.

### 4.2. Đầu ra

Thuật toán trả về một dictionary:

| Key | Ý nghĩa |
|---|---|
| `is_slip` | `True` nếu score sau làm mượt vượt ngưỡng |
| `r_value` | Điểm trượt đã làm mượt bằng EMA |
| `raw_r_value` | Giá trị MRVL thô trước khi boost và smoothing |
| `raw_score` | Score sau khi kết hợp MRVL với các gain |
| `translation` | Vector tịnh tiến trung bình của trường chuyển động |
| `translation_mag` | Độ lớn vector tịnh tiến trung bình |
| `mean_motion` | Độ lớn chuyển động trung bình |
| `pressing` | Cờ nhận biết đang ở pha nhấn |
| `mean_direction` | Hướng trượt trung bình nếu đang slip |
| `moving_count` | Số marker có chuyển động đủ lớn |
| `phase` | Trạng thái hiện tại: `slip`, `tracking`, `pressing`, `no_markers`, `insufficient_motion` |

## 5. Cấu hình thuật toán

Các tham số chính nằm trong `config/pipeline_config.yaml`:

| Tham số | Giá trị hiện tại | Vai trò |
|---|---:|---|
| `slip_threshold` | `0.8` | Ngưỡng quyết định trượt |
| `min_motion_thresh` | `2.0 px` | Ngưỡng lọc chuyển động nhỏ |
| `min_moving_markers` | `5` | Số marker chuyển động tối thiểu |
| `alpha` | `0.3` | Hệ số EMA cơ bản khi score tăng |
| `alpha_decay` | `0.6` | Hệ số EMA khi score giảm hoặc thiếu bằng chứng |
| `rebound_dot_prod_threshold` | `-0.1` | Ngưỡng lọc chuyển động hồi đàn hồi |
| `press_rate_threshold` | `0.5 px/frame` | Ngưỡng phát hiện pha nhấn |
| `history_buffer_length` | `5 frame` | Độ dài buffer trong pipeline realtime |

Các tham số này được chọn theo đơn vị pixel và frame, vì thuật toán hoạt động trực tiếp trên tọa độ ảnh của marker.

## 6. Quy trình xử lý tổng quát

Thuật toán có thể tóm tắt bằng các bước:

```text
prev_markers, current_markers, valid_mask, ref_markers
    |
    v
Lọc marker invalid
    |
    v
Tính displacement giữa frame hiện tại và frame quá khứ
    |
    v
Lọc chuyển động nhỏ
    |
    v
Lọc rebound nếu có reference
    |
    v
Kiểm tra số marker chuyển động tối thiểu
    |
    v
Khử thành phần radial do press/release
    |
    v
Tính weighted MRVL
    |
    v
Boost score bằng translation, participation, motion
    |
    v
Pressing gate
    |
    v
EMA smoothing
    |
    v
So sánh với slip_threshold
```

Các phần tiếp theo trình bày chi tiết từng bước.

## 7. Lọc marker invalid

Đầu tiên, thuật toán chỉ giữ các marker có tracking hợp lệ:

```text
valid_prev = prev_markers[valid_mask]
valid_curr = current_markers[valid_mask]
```

Nếu không có marker hợp lệ, thuật toán không cố gắng tính slip. Thay vào đó, score cũ được giảm dần:

```text
smoothed_score = smoothed_score * (1 - alpha_decay)
phase = "no_markers"
```

Lý do là khi tracking mất hết marker, mọi kết luận về hướng chuyển động đều không đáng tin cậy. Giảm dần score giúp trạng thái slip không bị giữ mãi sau khi mất tracking.

## 8. Phát hiện pha nhấn

Nếu có `ref_markers`, thuật toán tính biến dạng so với trạng thái reference:

```text
deformation_i(t) = p_i(t) - p_i(ref)
```

Sau đó tính độ biến dạng trung bình:

```text
mean_def_mag(t) = mean_i ||deformation_i(t)||
```

Tốc độ tăng biến dạng:

```text
delta_def = mean_def_mag(t) - mean_def_mag(t - 1)
```

Nếu:

```text
delta_def > press_rate_threshold
```

thì thuật toán xem hệ thống đang trong pha nhấn (`pressing = True`). Pha nhấn có thể làm nhiều marker chuyển động, nhưng chuyển động này không nhất thiết là trượt. Vì vậy thông tin pressing được dùng để giảm false positive ở các bước sau.

## 9. Tính vector chuyển động marker

Vector chuyển động giữa frame quá khứ và frame hiện tại:

```text
d_i = p_i(t) - p_i(t - h)
```

Trong đó `h` phụ thuộc vào độ dài history buffer. Độ lớn chuyển động:

```text
m_i = ||d_i||
```

Chỉ những marker có chuyển động đủ lớn mới được giữ:

```text
m_i > min_motion_thresh
```

Với cấu hình hiện tại:

```text
min_motion_thresh = 2.0 px
```

Mục đích của bước này là loại các vector quá nhỏ. Khi vector rất nhỏ, hướng của nó dễ bị chi phối bởi nhiễu tracking, rung camera hoặc dao động đàn hồi nhỏ.

## 10. Lọc rebound

Khi có reference, thuật toán loại các marker đang hồi đàn hồi về vị trí ban đầu. Với mỗi marker, xét tích vô hướng giữa displacement hiện tại và deformation so với reference:

```text
dot_i = d_i · deformation_i(t)
```

Nếu:

```text
dot_i < rebound_dot_prod_threshold
```

marker đó được xem là rebound và bị loại khỏi tập marker chuyển động.

Diễn giải:

- Nếu `dot_i > 0`, marker đang đi cùng hướng với biến dạng hiện tại.
- Nếu `dot_i < 0`, marker đang đi ngược hướng biến dạng, tức có xu hướng trở về trạng thái reference.

Chuyển động hồi đàn hồi thường xảy ra khi nhả lực hoặc vật liệu gel trở lại hình dạng ban đầu. Đây không phải là trượt chủ động, nên cần loại bỏ để tránh báo trượt sai.

## 11. Điều kiện số marker chuyển động tối thiểu

Sau khi lọc chuyển động nhỏ và rebound, thuật toán đếm:

```text
moving_count = số marker còn lại
```

Nếu:

```text
moving_count < min_moving_markers
```

thì thuật toán trả về `phase = "insufficient_motion"` và giảm dần score cũ.

Với cấu hình hiện tại:

```text
min_moving_markers = 5
```

Điều kiện này giúp loại các tình huống chỉ một vài marker chuyển động do nhiễu, lỗi tracking hoặc tiếp xúc cục bộ nhỏ. Slip đáng tin cậy thường tạo chuyển động trên nhiều marker.

## 12. Khử thành phần xuyên tâm do press/release

Nhấn hoặc nhả thường tạo chuyển động marker theo dạng xuyên tâm quanh vùng tiếp xúc. Dạng chuyển động này có thể có cấu trúc rõ nhưng không phải slip. Vì vậy trước khi đo độ đồng hướng, thuật toán khử một thành phần radial xấp xỉ.

Gọi tọa độ reference đã căn tâm:

```text
r_i = p_i(ref) - mean_j p_j(ref)
```

Thuật toán mô hình hóa displacement như:

```text
d_i ≈ t + s r_i
```

Trong đó:

- `t` là thành phần tịnh tiến trung bình.
- `s r_i` là thành phần xuyên tâm do press/release.
- `s` là hệ số radial scale.

Trong code, `translation` được tính:

```text
t = mean_i d_i
```

Hệ số radial được ước lượng bằng:

```text
s = sum_i r_i · (d_i - t) / sum_i ||r_i||^2
```

Sau đó loại thành phần radial:

```text
clean_d_i = d_i - s r_i
```

Kết quả `clean_d_i` được dùng cho bước tính MRVL. Nhờ đó, score tập trung hơn vào chuyển động tịnh tiến liên quan đến trượt.

## 13. Weighted MRVL

MRVL là viết tắt của Mean Resultant Vector Length. Đây là thước đo mức độ đồng nhất hướng của một tập vector.

Với mỗi vector `clean_d_i`, tính góc:

```text
theta_i = atan2(clean_d_i_y, clean_d_i_x)
```

Trọng số theo độ lớn vector:

```text
w_i = ||clean_d_i|| / sum_j ||clean_d_j||
```

Sau đó tính:

```text
C = sum_i w_i cos(theta_i)
S = sum_i w_i sin(theta_i)
R = sqrt(C^2 + S^2)
```

Trong code, `R` chính là `raw_r_value`.

Ý nghĩa:

| Giá trị `R` | Diễn giải |
|---|---|
| Gần `1` | Các marker chuyển động gần cùng hướng |
| Gần `0` | Hướng chuyển động phân tán hoặc triệt tiêu nhau |

Vì dùng trọng số theo độ lớn displacement, marker chuyển động mạnh đóng góp nhiều hơn marker chuyển động yếu. Điều này phù hợp với trực giác vật lý: slip thật thường tạo displacement rõ hơn nhiễu nhỏ.

Hướng trượt trung bình được tính bằng:

```text
mean_direction = atan2(S, C)
```

## 14. Kết hợp thêm bằng chứng chuyển động

MRVL chỉ cho biết các vector có cùng hướng hay không. Tuy nhiên một tập vector nhỏ, yếu nhưng tình cờ cùng hướng cũng có thể tạo `R` cao. Vì vậy thuật toán kết hợp thêm ba đại lượng:

### 14.1. Translation gain

Vector tịnh tiến trung bình:

```text
translation = mean_i clean_d_i
translation_mag = ||translation||
```

Gain:

```text
translation_gain = saturate(translation_mag / min_motion_thresh)
```

Nếu trường chuyển động có thành phần tịnh tiến rõ, gain tăng. Điều này phù hợp với slip vì trượt thường tạo chuyển động cùng hướng trên nhiều marker.

### 14.2. Participation gain

```text
participation_gain = saturate(moving_count / (2 * min_moving_markers))
```

Gain này tăng khi có nhiều marker tham gia. Nếu chỉ vừa đủ vài marker, score không được tăng quá mạnh.

### 14.3. Motion gain

```text
mean_motion = mean_i ||clean_d_i||
motion_gain = saturate(mean_motion / min_motion_thresh)
```

Gain này giảm ảnh hưởng của các chuyển động quá yếu.

Hàm `saturate` giới hạn giá trị trong đoạn `[0, 1]`.

## 15. Công thức score thô

Score thô sau khi kết hợp MRVL và các gain:

```text
raw_score = R
raw_score = raw_score * (0.55 + 0.45 * translation_gain)
raw_score = raw_score * (0.90 + 0.10 * participation_gain)
raw_score = raw_score * (0.85 + 0.15 * motion_gain)
raw_score = min(raw_score, 1.0)
```

Các hệ số được thiết kế để MRVL vẫn là thành phần chính. Translation, số marker tham gia và độ lớn chuyển động chỉ đóng vai trò điều chỉnh. Nhờ đó, thuật toán vừa giữ được ý nghĩa chính là độ đồng hướng, vừa tránh báo slip khi chuyển động quá yếu hoặc chỉ có ít marker.

## 16. Pressing gate

Nếu hệ thống đang trong pha nhấn nhưng thành phần tịnh tiến yếu:

```text
if pressing and translation_gain < 0.35:
    decay score
    phase = "pressing"
```

Khi nhấn xuống, marker có thể dịch chuyển mạnh nhưng chủ yếu theo dạng biến dạng đàn hồi, không phải trượt. Pressing gate giúp giảm báo sai trong trường hợp này.

Tuy nhiên thuật toán không chặn mọi trường hợp pressing. Nếu vừa nhấn vừa có trượt rõ, `translation_gain` sẽ đủ lớn và tín hiệu vẫn được cho qua. Đây là điểm quan trọng vì trong thực tế slip có thể xảy ra đồng thời với thao tác nhấn.

## 17. Làm mượt bằng EMA

Score theo từng frame có thể dao động do nhiễu tracking, rung camera hoặc thay đổi tiếp xúc. Vì vậy thuật toán dùng Exponential Moving Average để làm mượt:

```text
smoothed_r(t) = alpha * raw_score(t) + (1 - alpha) * smoothed_r(t - 1)
```

Trong code, EMA là bất đối xứng:

- Khi score tăng, dùng `alpha_up` thích nghi theo độ mạnh của bằng chứng.
- Khi score giảm, dùng `alpha_decay`.

Hệ số tăng thích nghi:

```text
evidence = saturate((raw_score + R + translation_gain) / 3)
alpha_up = saturate(alpha + (1 - alpha) * 0.65 * evidence)
```

Nếu bằng chứng trượt mạnh, `alpha_up` lớn hơn, detector phản ứng nhanh hơn. Nếu bằng chứng yếu, score tăng chậm hơn, giúp tránh false positive.

Khi thiếu marker hoặc không đủ chuyển động, thuật toán cũng decay score:

```text
smoothed_r = smoothed_r * (1 - alpha_decay)
```

Cơ chế này giúp trạng thái slip tự tắt khi không còn bằng chứng mới.

## 18. Quyết định trượt

Quyết định cuối cùng:

```text
is_slip = smoothed_r > slip_threshold
```

Với cấu hình hiện tại:

```text
slip_threshold = 0.8
```

Nếu `is_slip = True`, thuật toán trả về `phase = "slip"` và cung cấp thêm `mean_direction` để biểu diễn hướng trượt. Nếu score chưa vượt ngưỡng, trạng thái là `tracking`, `pressing`, `no_markers` hoặc `insufficient_motion` tùy nguyên nhân.

## 19. Pseudocode thuật toán

```text
Input:
    prev_markers, current_markers, valid_mask, ref_markers

State:
    smoothed_r
    prev_mean_deformation

Algorithm:
    if no valid marker:
        decay smoothed_r
        return no_markers

    valid_prev = prev_markers[valid_mask]
    valid_curr = current_markers[valid_mask]

    if ref_markers exists:
        deformation = valid_curr - valid_ref
        mean_def = mean(norm(deformation))
        pressing = (mean_def - prev_mean_deformation) > press_rate_threshold
        prev_mean_deformation = mean_def

    displacement = valid_curr - valid_prev
    magnitude = norm(displacement)
    motion_mask = magnitude > min_motion_thresh

    if ref_markers exists:
        dot = displacement dot deformation
        rebound = dot < rebound_dot_prod_threshold
        motion_mask = motion_mask and not rebound

    significant_disp = displacement[motion_mask]

    if count(significant_disp) < min_moving_markers:
        decay smoothed_r
        return insufficient_motion

    if ref_markers exists:
        clean_disp = remove_radial_component(significant_disp, ref_points)
    else:
        clean_disp = significant_disp

    R, mean_direction = weighted_mrvl(clean_disp)

    translation = mean(clean_disp)
    translation_gain = saturate(norm(translation) / min_motion_thresh)
    participation_gain = saturate(moving_count / (2 * min_moving_markers))
    motion_gain = saturate(mean(norm(clean_disp)) / min_motion_thresh)

    raw_score = R
    raw_score *= 0.55 + 0.45 * translation_gain
    raw_score *= 0.90 + 0.10 * participation_gain
    raw_score *= 0.85 + 0.15 * motion_gain
    raw_score = min(raw_score, 1.0)

    if pressing and translation_gain < 0.35:
        decay smoothed_r
        return pressing

    if raw_score >= smoothed_r:
        alpha_up = adaptive_alpha(raw_score, R, translation_gain)
        smoothed_r = alpha_up * raw_score + (1 - alpha_up) * smoothed_r
    else:
        smoothed_r = alpha_decay * raw_score + (1 - alpha_decay) * smoothed_r

    is_slip = smoothed_r > slip_threshold
    return slip information
```

## 20. Ưu điểm của kiến trúc thuật toán

### 20.1. Không cần dữ liệu huấn luyện

Thuật toán hoạt động dựa trên quy luật hình học và thống kê hướng. Vì vậy nó có thể chạy ngay cả khi chưa có bộ dữ liệu gán nhãn slip đầy đủ.

### 20.2. Dễ giải thích

Mỗi bước xử lý có ý nghĩa vật lý rõ ràng:

| Bước | Ý nghĩa |
|---|---|
| Lọc invalid marker | Tránh lỗi tracking |
| Lọc chuyển động nhỏ | Giảm nhiễu pixel và rung nhẹ |
| Rebound filter | Loại chuyển động hồi đàn hồi |
| Radial removal | Giảm nhầm lẫn press/release với slip |
| Weighted MRVL | Đo độ đồng hướng của chuyển động |
| Translation gain | Tăng độ tin khi có tịnh tiến rõ |
| EMA | Giảm nhấp nháy theo frame |

### 20.3. Phù hợp realtime

Thuật toán chỉ dùng các phép tính vector trên mảng marker, không cần mạng neural hoặc tối ưu phức tạp. Độ phức tạp xấp xỉ tuyến tính theo số marker:

```text
O(N)
```

Điều này phù hợp với pipeline realtime sau khi marker đã được tracking.

### 20.4. Có cơ chế chống false positive

Các nguồn false positive phổ biến như tracking lỗi, chuyển động rất nhỏ, rebound và nhấn xuống đều được xử lý bằng các bước lọc hoặc gate riêng.

## 21. Hạn chế

| Hạn chế | Nguyên nhân | Hướng cải thiện |
|---|---|---|
| Phụ thuộc chất lượng tracking | Đầu vào là tọa độ marker từ LK | Tăng chất lượng detect/track marker, lọc outlier |
| Nhạy với tham số pixel/frame | Ngưỡng phụ thuộc camera và framerate | Tuning theo setup hoặc chuẩn hóa theo thời gian |
| Có thể bỏ sót slip chậm | `min_motion_thresh` lọc chuyển động quá nhỏ | Dùng multi-scale hoặc giảm ngưỡng có kiểm soát |
| Pressing có thể che slip yếu | Pressing gate suppress khi translation thấp | Dùng phân rã translation/radial rõ hơn |
| Không học theo dữ liệu | Threshold thủ công | Thu thập nhãn slip và học score/threshold |

Các hạn chế này là lý do repo có thêm `SlipDetectorV2`, dùng phân rã translation/radial đa thang để phát hiện tốt hơn slow slip và tách press/release rõ hơn.

## 22. So sánh với hướng học máy

Slip V1 không phải mô hình phân loại học máy. Nó không tối ưu loss và không có trọng số được học. So với mô hình học máy, cách này có các đặc điểm:

| Tiêu chí | Slip V1 | Mô hình học máy |
|---|---|---|
| Dữ liệu nhãn | Không bắt buộc | Cần dữ liệu gán nhãn |
| Giải thích | Rất rõ | Phụ thuộc mô hình |
| Tuning | Threshold thủ công | Huấn luyện và validation |
| Realtime | Nhẹ | Tùy kích thước mô hình |
| Khả năng thích nghi | Hạn chế | Tốt hơn nếu có dữ liệu đủ đa dạng |

Trong đồ án, Slip V1 có thể được trình bày như một thuật toán baseline có tính giải thích cao, phù hợp để chứng minh mối liên hệ giữa trường chuyển động marker và hiện tượng trượt.

## 23. Cách trình bày ngắn trong quyển đồ án

Có thể viết trong phần thuyết minh như sau:

> Hệ thống phát hiện trượt dựa trên trường chuyển động của các marker xúc giác. Sau khi tracking marker bằng Lucas-Kanade, thuật toán so sánh tọa độ marker hiện tại với tọa độ ở một frame quá khứ trong history buffer để thu được các vector chuyển động. Các marker tracking không hợp lệ, chuyển động nhỏ và chuyển động hồi đàn hồi được loại bỏ. Sau đó thuật toán khử xấp xỉ thành phần chuyển động xuyên tâm do press/release, rồi tính Weighted Mean Resultant Vector Length để đo mức độ đồng hướng của các vector còn lại. Score MRVL được điều chỉnh thêm bằng độ lớn tịnh tiến, số marker tham gia và độ lớn chuyển động trung bình. Cuối cùng score được làm mượt bằng EMA và so sánh với ngưỡng `0.8` để quyết định có trượt hay không.

## 24. Gợi ý hình vẽ trong đồ án

Nên vẽ một sơ đồ khối gồm các thành phần:

```text
Ảnh camera
    |
    v
Tracking marker LK
    |
    v
Tọa độ marker hiện tại + valid mask
    |
    v
History buffer
    |
    v
Vector chuyển động marker
    |
    v
Lọc invalid / small motion / rebound
    |
    v
Khử radial press-release
    |
    v
Weighted MRVL
    |
    v
Score boost + EMA
    |
    v
Slip / No slip
```

Nếu cần minh họa trực quan, có thể vẽ hai trường hợp:

| Trạng thái | Hình vector marker |
|---|---|
| Không trượt, chỉ nhấn | Vector tỏa ra hoặc co vào quanh tâm |
| Trượt | Nhiều vector gần song song cùng hướng |

## 25. Các điểm nên nhấn mạnh khi bảo vệ

1. Thuật toán khai thác trực tiếp đặc trưng vật lý của slip: chuyển động đồng hướng trên nhiều marker.
2. MRVL là thước đo phù hợp vì nó đo độ tập trung hướng của vector trên mặt phẳng.
3. Weighted MRVL giúp marker chuyển động rõ đóng góp nhiều hơn marker nhiễu nhỏ.
4. Rebound filter và radial removal giúp giảm nhầm lẫn giữa slip với biến dạng đàn hồi.
5. EMA giúp quyết định ổn định theo thời gian, tránh trạng thái bật/tắt liên tục.
6. Thuật toán nhẹ, dễ giải thích và chạy realtime, nhưng phụ thuộc vào chất lượng tracking và cần tuning tham số theo setup.

