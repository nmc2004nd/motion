# Kiến Trúc `force_cnn` Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code trong `src/force_cnn`. Mục tiêu là giải thích kiến trúc CNN end-to-end dùng để ước lượng lực, lý do lựa chọn thiết kế, ưu nhược điểm so với `force_poly` và `force_model`, đồng thời chuẩn bị các câu hỏi có thể gặp khi phản biện đồ án.

## 1. Vai trò của `force_cnn`

`force_cnn` là mô hình ước lượng lực trực tiếp từ ảnh. Khác với `force_poly` và `force_model`, mô hình này không cần detect marker và không cần Lucas-Kanade tracking tại thời điểm inference. Đầu vào của mô hình là một cặp ảnh:

```text
[reference_gray, deformed_gray]
```

Hai ảnh được xếp thành tensor 2 kênh:

```text
input: (2, H, W)
```

Trong config hiện tại, ảnh được resize về:

```text
H = 240, W = 320
```

Luồng xử lý tổng quát:

```text
data/sessions
  trial frames + force_log + reference
        |
        v
dataset.py
  sync frame với force
  load reference + deformed frame
  resize, grayscale, normalize [0, 1]
        |
        v
input tensor (2, 240, 320)
        |
        v
ForceCNN
  ResNet18 backbone hoặc SmallCNN
        |
        v
regression head
        |
        v
predicted force
```

Trong config hiện tại:

| Thành phần | Giá trị |
|---|---:|
| Input | Cặp ảnh grayscale 2 kênh |
| Kích thước ảnh | 240 x 320 |
| Backbone | ResNet18 |
| Pretrained | Có, dùng ImageNet weights |
| Head | 512 -> 128 -> 1 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Loss | Huber loss, delta = 1.0 |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Số tham số | 11 239 169 |

Code chính:

| File | Vai trò |
|---|---|
| `src/force_cnn/dataset.py` | Index trial, load cặp ảnh, đồng bộ force, augmentation |
| `src/force_cnn/model.py` | Định nghĩa `ForceCNN`, `ResNet18Backbone`, `SmallCNN` |
| `src/force_cnn/train.py` | Huấn luyện, scheduler, early stopping, checkpoint |
| `src/force_cnn/eval.py` | Đánh giá tổng thể, theo trial, xuất plot/CSV |
| `src/force_cnn/infer.py` | Inference từ một cặp ảnh reference-frame |
| `src/force_cnn/config.yaml` | Cấu hình dữ liệu, model, train, augment, infer |

## 2. Dữ liệu đầu vào trong `dataset.py`

`force_cnn` không dùng cache marker `.npz`. Thay vào đó, dataset đọc trực tiếp từ `data/sessions`. Mỗi trial được index thành một `TrialIndex` gồm:

| Trường | Ý nghĩa |
|---|---|
| `session_id` | Tên session |
| `trial_id` | Tên trial |
| `ref_path` | Đường dẫn ảnh reference |
| `samples` | Danh sách `(frame_path, force_n)` sau khi sync |

### 2.1. Đồng bộ frame với force

Mỗi frame có timestamp `ts_mono`. Force log cũng có timestamp. Code dùng nearest-neighbor để tìm giá trị force gần timestamp của frame nhất:

```text
force(frame_t) = force_log[argmin(|force_ts - frame_t|)]
```

Nếu chênh lệch lớn hơn `force_sync_tolerance_s = 0.1 s`, frame bị bỏ qua. Điều này tránh gán nhãn force sai thời điểm.

Sau khi lấy force, code trừ `zero_offset_n` trong `session.yaml`:

```text
force_n = force_raw - zero_offset
```

Lý do: cảm biến lực có thể có offset ban đầu. Trừ offset giúp nhãn phản ánh lực tương đối do tác động, không bị lệch bởi bias của cảm biến.

### 2.2. Cách tạo input ảnh

Trong `ForceImageDataset`, mỗi sample được tạo như sau:

1. Load ảnh reference của trial.
2. Load frame biến dạng.
3. Convert sang grayscale.
4. Resize về `image_size`.
5. Normalize pixel về `[0, 1]`.
6. Stack thành tensor 2 kênh:

```text
inp = stack([ref, frame], axis=0)
```

Kết quả:

```text
inp:   (2, H, W) float32
force: scalar float32
```

### 2.3. Vì sao dùng cặp ảnh reference và deformed?

Nếu chỉ đưa ảnh deformed vào mô hình, CNN phải tự suy ra trạng thái ban đầu của cảm biến. Điều này khó hơn vì mỗi session có thể có ánh sáng, nền, marker, vị trí camera hoặc trạng thái ban đầu khác nhau.

Đưa cả ảnh reference và ảnh deformed giúp mô hình học sự khác biệt giữa trạng thái không tải và trạng thái chịu lực. Nói cách khác, reference đóng vai trò mốc so sánh. Đây là cách thay thế cho việc tính displacement thủ công trong `force_poly` và `force_model`.

### 2.4. Vì sao dùng grayscale?

Bài toán chủ yếu phụ thuộc vào hình dạng, vị trí và biến dạng của marker. Thông tin màu RGB không thật sự cần thiết nếu ảnh marker là ảnh đơn sắc hoặc độ tương phản là tín hiệu chính. Grayscale giúp:

- Giảm số kênh input từ 6 nếu dùng cặp RGB xuống 2.
- Giảm số tham số ở conv đầu.
- Giảm chi phí tính toán.
- Tập trung vào cấu trúc biến dạng thay vì màu sắc.

### 2.5. Vì sao lazy-load ảnh thay vì cache pixel?

Dataset đọc ảnh frame từ disk khi cần, còn reference được cache trong RAM sau lần đọc đầu. Lý do:

- Tránh tạo cache pixel rất lớn.
- Dữ liệu ảnh có thể chiếm nhiều dung lượng hơn marker displacement.
- DataLoader với `num_workers` có thể pipeline I/O trong lúc GPU train.
- Reference dùng lại nhiều lần trong một trial nên cache reference là hợp lý.

## 3. Augmentation ảnh

Augmentation chỉ áp dụng cho train set. Validation/test không augment để metric phản ánh dữ liệu thật.

Config hiện tại:

| Augmentation | Giá trị | Ý nghĩa |
|---|---:|---|
| Flip ngang | 0.5 | Lật cả reference và frame theo chiều ngang |
| Flip dọc | 0.5 | Lật cả reference và frame theo chiều dọc |
| Rotation | +/- 5 độ | Xoay đồng bộ cả hai ảnh |
| Brightness jitter | 0.1 | Dịch độ sáng cả hai ảnh |
| Contrast jitter | 0.1 | Thay đổi tương phản cả hai ảnh |

Điểm quan trọng là augmentation được áp dụng đồng bộ cho cả reference và frame. Nếu chỉ augment một ảnh, quan hệ biến dạng giữa hai ảnh sẽ bị phá vỡ. Code dùng `_apply_paired_augment` để đảm bảo hai ảnh được biến đổi giống nhau về hình học và photometric.

### Vì sao brightness/contrast cũng áp giống nhau cho cả hai ảnh?

Mục tiêu là mô phỏng thay đổi điều kiện camera hoặc exposure mà không phá quan hệ vật lý giữa reference và frame. Nếu chỉ thay brightness của frame mà không thay reference, mô hình có thể học tín hiệu giả không tồn tại trong thực tế. Áp giống nhau giúp mô hình robust hơn với thay đổi ánh sáng chung.

## 4. Kiến trúc trong `model.py`

`ForceCNN` hỗ trợ hai backbone:

| Backbone | Vai trò |
|---|---|
| `resnet18` | Mô hình chính, dùng pretrained ImageNet |
| `small_cnn` | Mô hình nhẹ để debug nhanh trên CPU/laptop |

Config hiện tại dùng:

```yaml
backbone: "resnet18"
pretrained: true
hidden_head: 128
dropout: 0.2
```

Kiến trúc tổng quát:

```text
Input (B, 2, 240, 320)
        |
        v
ResNet18 backbone
  conv1 được đổi từ 3 kênh sang 2 kênh
  fc cuối được thay bằng Identity
        |
        v
Feature vector (B, 512)
        |
        v
Regression head
  Linear(512 -> 128)
  ReLU
  Dropout(0.2)
  Linear(128 -> 1)
        |
        v
Predicted force (B,)
```

## 5. ResNet18 backbone

ResNet18 ban đầu được thiết kế cho ảnh RGB 3 kênh và phân loại ImageNet 1000 lớp. Trong `force_cnn`, ResNet18 được sửa theo hai điểm:

1. Thay `conv1` để nhận input 2 kênh thay vì 3 kênh.
2. Thay `fc` cuối bằng `Identity` để lấy feature 512 chiều thay vì logits phân loại.

### 5.1. Vì sao dùng ResNet18?

ResNet18 là backbone cân bằng giữa độ mạnh và chi phí tính toán:

- Đủ sâu để học đặc trưng thị giác phức tạp.
- Nhẹ hơn ResNet34/50.
- Có pretrained ImageNet sẵn.
- Residual connection giúp training ổn định hơn mạng CNN sâu thông thường.

Với dữ liệu không quá lớn, dùng backbone pretrained thường tốt hơn train CNN lớn từ đầu.

### 5.2. Vì sao dùng pretrained ImageNet cho bài toán grayscale tactile?

Dù ImageNet là ảnh tự nhiên RGB, các lớp đầu của CNN thường học filter tổng quát như cạnh, blob, texture và gradient. Những filter này vẫn hữu ích cho ảnh marker grayscale. Pretrained giúp mô hình bắt đầu từ đặc trưng thị giác tốt thay vì học từ ngẫu nhiên.

Trong code, khi `pretrained=True`, trọng số `conv1` RGB của ResNet18 được chuyển sang input 2 kênh bằng hàm `_inflate_conv1_from_pretrained`.

### 5.3. Conv1 3 kênh được chuyển sang 2 kênh như thế nào?

ResNet18 pretrained có `conv1.weight` shape:

```text
(64, 3, 7, 7)
```

Input của `force_cnn` là 2 kênh, nên conv1 mới cần shape:

```text
(64, 2, 7, 7)
```

Code làm:

```text
mean_w = mean(weight_rgb, dim=channel)
new_weight = repeat(mean_w, in_channels=2)
```

Ý nghĩa: lấy trung bình filter qua RGB để tạo filter grayscale, sau đó nhân bản cho hai kênh reference và frame. Cách này giữ lại filter cạnh/blob đã học từ ImageNet, đồng thời thích ứng với input 2 kênh.

### 5.4. Vì sao không dùng 3 kênh bằng cách thêm ảnh difference?

Có thể thiết kế input 3 kênh như `[ref, frame, frame-ref]`. Tuy nhiên code hiện tại dùng 2 kênh để giữ pipeline đơn giản và để CNN tự học phép so sánh giữa reference và frame. Với ResNet18, conv đầu có thể học filter kết hợp hai kênh để phát hiện vùng khác biệt.

Thêm difference channel có thể là hướng cải tiến, nhưng không bắt buộc.

## 6. SmallCNN fallback

Ngoài ResNet18, code có `SmallCNN`:

```text
ConvBlock(2 -> 16, kernel=5) + MaxPool
ConvBlock(16 -> 32) + MaxPool
ConvBlock(32 -> 64) + MaxPool
ConvBlock(64 -> 128, no pool)
AdaptiveAvgPool2d(1)
feature vector 128
```

`SmallCNN` dùng để debug nhanh hoặc chạy trên máy yếu. Nó ít tham số hơn nhiều nhưng thường kém chính xác hơn ResNet18 vì không có pretrained và năng lực biểu diễn thấp hơn.

## 7. Regression head

Sau backbone, mô hình dùng head:

```text
Linear(backbone_out_dim -> hidden_head)
ReLU
Dropout
Linear(hidden_head -> 1)
```

Với ResNet18:

```text
backbone_out_dim = 512
hidden_head = 128
```

Số tham số head:

```text
Linear(512, 128): 512*128 + 128 = 65 664
Linear(128, 1):   128*1 + 1     = 129
Tổng head: 65 793
```

Lý do dùng head nhỏ:

- Chuyển feature thị giác 512 chiều thành scalar force.
- ReLU thêm phi tuyến cho bài toán regression.
- Dropout giảm overfit.
- Head nhỏ đủ vì backbone đã làm phần trích đặc trưng chính.

## 8. Số tham số

Với ResNet18 pretrained, input 2 kênh và head 512-128-1, tổng số tham số trainable hiện tại là:

```text
11 239 169
```

Xấp xỉ:

| Thành phần | Tham số |
|---|---:|
| ResNet18 backbone bỏ fc, conv1 đổi 2 kênh | 11 173 376 |
| Regression head | 65 793 |
| Tổng | 11 239 169 |

So với hai mô hình còn lại:

| Mô hình | Số tham số |
|---|---:|
| `force_poly` | 913 |
| `force_model` | 25 537 |
| `force_cnn` | 11 239 169 |

`force_cnn` lớn hơn nhiều, nhưng đổi lại có khả năng học trực tiếp từ ảnh và đạt kết quả test tốt nhất trong thí nghiệm hiện tại.

## 9. Huấn luyện trong `train.py`

Training dùng:

| Thành phần | Lý do |
|---|---|
| Huber loss | Giảm ảnh hưởng của outlier do force sync, nhiễu ảnh hoặc nhãn sai |
| AdamW | Tối ưu ổn định, weight decay tách khỏi gradient |
| Weight decay 1e-4 | Regularization nhẹ cho mô hình lớn |
| CosineAnnealingLR | Giảm learning rate dần, giúp fine-tune ổn định |
| Early stopping theo validation MAE | Chọn checkpoint có sai số validation tốt nhất |
| Pin memory khi có CUDA | Tăng tốc copy batch CPU -> GPU |

Checkpoint `best.pt` lưu:

| Trường | Ý nghĩa |
|---|---|
| `epoch` | Epoch có validation MAE tốt nhất |
| `model_state` | Trọng số mô hình |
| `val_mae` | Validation MAE tốt nhất |
| `config` | Cấu hình huấn luyện |

Kết quả huấn luyện hiện tại:

| Metric test | Giá trị |
|---|---:|
| MAE | 0.0307 |
| RMSE | 0.0405 |
| R² | 0.999 |

Đây là kết quả tốt nhất trong ba mô hình, cho thấy ảnh raw chứa thêm thông tin hữu ích mà các pipeline marker-level có thể bỏ mất.

## 10. Đánh giá trong `eval.py`

`eval.py` load checkpoint tốt nhất, dựng lại split theo seed và đánh giá trên train/val/test. Output gồm:

| Output | Ý nghĩa |
|---|---|
| `test_predictions.npz` | Dự đoán, nhãn thật và metric tổng thể |
| `test_breakdown.csv` | MAE/RMSE/R² theo từng trial |
| `test_scatter.png` | Scatter predicted vs actual |
| `test_timeseries_trial0.png` | Dự đoán theo frame index của trial đầu tiên |

Đánh giá theo trial quan trọng vì CNN có thể rất tốt về trung bình nhưng vẫn yếu ở một session hoặc điều kiện ánh sáng cụ thể.

## 11. Inference trong `infer.py`

Inference thực tế chỉ cần:

```text
reference image
deformed frame
checkpoint
```

Luồng:

```text
load ref + frame
grayscale + resize + normalize
stack thành (1, 2, H, W)
load ForceCNN checkpoint
forward
predicted force
```

Điểm khác biệt lớn so với `force_poly` và `force_model`: inference không cần detect marker, không cần optical flow, không cần valid mask. Điều này làm triển khai đơn giản hơn, nhưng mô hình phụ thuộc mạnh hơn vào phân phối ảnh đã train.

## 12. Vì sao kiến trúc này hợp lý?

### 12.1. Học trực tiếp từ tín hiệu ảnh

Marker tracking là một pipeline rời rạc: detect, track, lọc valid, tính displacement. Mỗi bước có thể làm mất thông tin. CNN bỏ qua bước thủ công đó và học trực tiếp từ ảnh reference-frame, nên có thể tận dụng:

- Hình dạng marker.
- Biến dạng cục bộ.
- Texture và độ tương phản.
- Vùng ảnh ngoài marker nếu có liên quan.
- Sai khác tinh vi giữa reference và frame.

### 12.2. Phù hợp khi độ chính xác là ưu tiên chính

Kết quả test hiện tại cho thấy `force_cnn` đạt MAE thấp nhất. Điều này hợp lý vì CNN nhận nhiều thông tin nhất: toàn bộ ảnh thay vì chỉ 9 feature hoặc tọa độ marker.

### 12.3. Pretrained giúp giảm nhu cầu dữ liệu

Train ResNet18 từ đầu với số trial hạn chế dễ overfit. Dùng pretrained giúp backbone đã có khả năng nhận diện cạnh, blob và texture. Mô hình chỉ cần fine-tune để ánh xạ đặc trưng ảnh sang lực.

### 12.4. Cặp ảnh giúp học biến dạng tương đối

Reference cung cấp trạng thái không tải. Frame cung cấp trạng thái hiện tại. CNN có thể học phép so sánh giữa hai kênh để suy ra mức biến dạng. Đây là analog học được của displacement field trong các mô hình marker-level.

## 13. Hạn chế của `force_cnn`

| Hạn chế | Nguyên nhân | Hướng xử lý |
|---|---|---|
| Nhiều tham số, train lâu | ResNet18 ~11.2M params | Dùng `small_cnn`, freeze backbone, giảm ảnh size |
| Khó giải thích | CNN học feature ẩn từ pixel | Dùng Grad-CAM, saliency map, ablation vùng ảnh |
| Phụ thuộc phân phối ảnh | Học trực tiếp từ pixel, nhạy ánh sáng/camera | Augmentation, calibration, train nhiều session |
| Có thể học shortcut | Nền, ánh sáng hoặc artefact có thể tương quan với force | Kiểm soát dataset, randomize điều kiện, kiểm tra saliency |
| Không đảm bảo output không âm | Head cuối tuyến tính | Clip output hoặc dùng Softplus |
| Resize có thể mất chi tiết | Ảnh resize 240 x 320 | Tăng image size nếu GPU đủ mạnh |
| Cần reference phù hợp | Dự đoán dựa trên cặp ref-frame | Dùng đúng reference của session hoặc cơ chế chọn ref tự động |

## 14. Câu hỏi có thể gặp khi phản biện

### Q1. Tại sao dùng CNN trong khi đã có marker tracking?

CNN học trực tiếp từ ảnh nên không phụ thuộc vào lỗi detect/tracking marker. Pipeline marker-level có thể mất thông tin khi marker bị che, tracking sai hoặc khi feature thủ công không mô tả hết biến dạng. CNN có thể khai thác toàn bộ tín hiệu thị giác và trong kết quả hiện tại cho sai số thấp nhất.

### Q2. Vì sao input là 2 kênh `[reference, frame]`?

Reference là trạng thái không tải, frame là trạng thái biến dạng. Cặp này cho phép mô hình học sự thay đổi tương đối do lực gây ra. Nếu chỉ dùng frame, mô hình phải tự suy ra baseline, dễ bị ảnh hưởng bởi session, ánh sáng và cấu hình camera.

### Q3. Vì sao không dùng ảnh RGB?

Bài toán chủ yếu dựa vào hình dạng và độ tương phản marker, không cần đầy đủ màu sắc. Grayscale giảm số kênh, giảm chi phí tính toán và tập trung vào cấu trúc biến dạng. Nếu dữ liệu màu chứa thông tin hữu ích, có thể thử input RGB hoặc 6 kênh `[ref_rgb, frame_rgb]` như một thí nghiệm mở rộng.

### Q4. Vì sao dùng ResNet18?

ResNet18 đủ mạnh để học đặc trưng ảnh, có residual connection giúp training ổn định, có pretrained ImageNet và vẫn nhẹ hơn các backbone sâu hơn như ResNet50. Với dataset không quá lớn, ResNet18 là lựa chọn cân bằng giữa độ chính xác và chi phí.

### Q5. Pretrained ImageNet có phù hợp với ảnh tactile grayscale không?

Các lớp đầu của CNN pretrained thường học filter tổng quát như cạnh, blob, gradient và texture. Những đặc trưng này vẫn hữu ích cho ảnh marker grayscale. Pretrained không phải học lại từ đầu, giúp hội tụ nhanh hơn và giảm overfit.

### Q6. Conv1 pretrained RGB được chuyển sang 2 kênh như thế nào?

Code lấy trung bình trọng số conv1 qua 3 kênh RGB để tạo filter grayscale, sau đó lặp filter này cho 2 kênh input. Nhờ vậy mô hình giữ được filter thị giác đã học từ ImageNet nhưng thích ứng với input `[ref, frame]`.

### Q7. Vì sao không dùng channel thứ ba là ảnh difference?

CNN có thể tự học phép so sánh giữa reference và frame qua conv đầu. Thêm difference channel có thể giúp mô hình học nhanh hơn nhưng cũng tăng giả định thiết kế. Đây là hướng cải tiến hợp lý để thử nghiệm, nhưng thiết kế 2 kênh hiện tại đơn giản và đã cho kết quả tốt.

### Q8. Vì sao resize ảnh về 240 x 320?

Resize giảm bộ nhớ GPU và thời gian train. Kích thước 240 x 320 vẫn giữ đủ cấu trúc marker để học biến dạng. Nếu dùng ảnh lớn hơn, mô hình có thể giữ thêm chi tiết nhưng chi phí tính toán tăng.

### Q9. Resize có làm mất thông tin biến dạng nhỏ không?

Có thể có. Đây là đánh đổi giữa độ phân giải và chi phí. Kết quả hiện tại cho thấy 240 x 320 vẫn đủ để đạt MAE thấp. Nếu muốn kiểm chứng, có thể train lại với 480 x 640 và so sánh MAE/RMSE.

### Q10. Vì sao augmentation phải áp dụng đồng bộ cho reference và frame?

Vì lực được suy ra từ quan hệ giữa hai ảnh. Nếu chỉ xoay/lật/thay sáng một ảnh, mô hình sẽ thấy biến dạng giả. Áp dụng đồng bộ giữ nguyên quan hệ vật lý giữa reference và frame.

### Q11. Brightness/contrast jitter có phá dữ liệu không?

Không nếu áp giống nhau cho cả reference và frame. Nó mô phỏng thay đổi ánh sáng/camera chung. Nếu áp khác nhau, mô hình có thể học tín hiệu giả do chênh lệch exposure thay vì biến dạng thật.

### Q12. Vì sao chia theo trial chứ không chia theo frame?

Frame trong cùng trial liên tiếp và rất giống nhau. Nếu chia frame ngẫu nhiên, test có thể chứa frame gần giống train, làm kết quả quá lạc quan. Chia theo trial đánh giá khả năng tổng quát hóa trên lần đo mới.

### Q13. Vì sao không dùng cache `.npz` như `force_model`?

`force_cnn` cần ảnh raw, nếu cache toàn bộ pixel có thể tốn dung lượng lớn. Dataset lazy-load frame từ disk và chỉ cache reference trong RAM. Cách này tiết kiệm disk, đồng thời DataLoader có thể đọc song song bằng nhiều worker.

### Q14. Vì sao dùng Huber loss?

Huber loss ít nhạy với outlier hơn MSE và mượt hơn MAE. Dữ liệu có thể có sai số đồng bộ force, frame nhiễu hoặc biến dạng bất thường, nên Huber giúp training ổn định hơn.

### Q15. Vì sao dùng AdamW?

AdamW phù hợp để fine-tune CNN pretrained vì tối ưu ổn định và weight decay được tách khỏi gradient update. Weight decay giúp regularize mô hình lớn, giảm overfit.

### Q16. Vì sao dùng cosine learning rate scheduler?

Cosine scheduler giảm learning rate dần trong quá trình train, giúp mô hình học nhanh ở đầu và tinh chỉnh ổn định ở cuối. Điều này thường phù hợp khi fine-tune backbone pretrained.

### Q17. Vì sao có dropout ở regression head?

Dropout giúp giảm overfit ở head, tránh mô hình phụ thuộc quá mạnh vào một số feature ẩn của backbone. Vì backbone lớn, regularization ở head là cần thiết.

### Q18. Mô hình có đảm bảo lực dự đoán không âm không?

Không. Lớp cuối là Linear nên output có thể âm. Nếu yêu cầu vật lý bắt buộc lực không âm, có thể clip `max(0, pred)` ở inference hoặc thay output bằng Softplus.

### Q19. CNN có thể học nhầm background hoặc ánh sáng thay vì biến dạng không?

Có thể. Đây là hạn chế của mô hình học trực tiếp từ pixel. Cần kiểm soát dataset, dùng augmentation, kiểm tra theo nhiều session và dùng saliency/Grad-CAM để xem mô hình tập trung vào vùng marker hay không.

### Q20. Vì sao `force_cnn` tốt hơn `force_poly` và `force_model` trong kết quả?

Vì CNN nhận nhiều thông tin nhất: toàn bộ ảnh reference-frame. `force_poly` nén frame thành 9 đặc trưng, `force_model` dùng marker displacement, còn CNN có thể học cả biến dạng marker, texture, tương phản và các tín hiệu thị giác khác. Đổi lại, CNN nhiều tham số và train lâu hơn.

### Q21. Nếu reference không đúng session thì sao?

Dự đoán có thể sai vì mô hình học sự khác biệt giữa reference và frame. Reference sai làm baseline sai, tạo ra biến dạng giả. Khi triển khai cần dùng đúng reference tương ứng với trạng thái không tải của cùng setup/session.

### Q22. Inference của `force_cnn` có cần detect marker không?

Không. Inference chỉ cần load reference và frame, resize/normalize rồi forward qua CNN. Đây là ưu điểm triển khai so với các mô hình cần marker detection/tracking.

### Q23. Vì sao `infer.py` đặt `pretrained=False` khi load checkpoint?

Khi inference, trọng số đã nằm trong checkpoint `best.pt`. Không cần tải lại ImageNet weights. `pretrained=False` chỉ tránh load pretrained dư thừa; sau đó `model.load_state_dict` nạp đúng trọng số đã train.

### Q24. Có nên freeze backbone không?

Code hiện tại fine-tune toàn bộ mô hình. Freeze backbone có thể giảm overfit và train nhanh hơn nếu dữ liệu ít, nhưng có thể giảm độ chính xác vì ảnh tactile khác ImageNet. Đây là một thí nghiệm ablation hợp lý.

### Q25. Vì sao dùng validation MAE để chọn checkpoint?

MAE có cùng đơn vị với lực, dễ giải thích và ít bị outlier chi phối hơn RMSE. Chọn checkpoint theo validation MAE giúp mô hình tối ưu sai số trung bình tuyệt đối trên dữ liệu chưa thấy.

### Q26. Làm sao chứng minh mô hình không overfit?

Cần so sánh train/val/test metric, dùng split theo trial, theo dõi early stopping và đánh giá per-trial. Để chắc hơn, nên chạy nhiều seed split hoặc cross-validation theo trial/session.

### Q27. Hạn chế lớn nhất của `force_cnn` là gì?

Hạn chế lớn nhất là chi phí tính toán và khả năng giải thích thấp. Mô hình có hơn 11 triệu tham số, train lâu và có nguy cơ học shortcut từ ảnh. Cần kiểm soát dữ liệu và dùng công cụ giải thích như Grad-CAM nếu muốn chứng minh mô hình tập trung vào vùng biến dạng.

### Q28. Hướng cải tiến kiến trúc là gì?

Có thể thử input 3 kênh `[ref, frame, frame-ref]`, tăng image size, freeze hoặc partial-freeze backbone, dùng backbone nhẹ hơn như MobileNet/EfficientNet, thêm uncertainty estimation, hoặc dùng Siamese network để trích đặc trưng reference và frame rồi so sánh rõ ràng hơn.

## 15. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn như sau:

> Mô hình `force_cnn` ước lượng lực trực tiếp từ cặp ảnh reference và ảnh biến dạng. Hai ảnh grayscale được resize về 240 x 320, chuẩn hóa về [0, 1] và xếp thành input 2 kênh. Mô hình sử dụng ResNet18 pretrained làm backbone, trong đó lớp conv đầu được điều chỉnh từ 3 kênh RGB sang 2 kênh bằng cách lấy trung bình trọng số pretrained theo kênh màu và lặp lại cho hai kênh input. Lớp phân loại cuối của ResNet18 được thay bằng Identity để lấy vector đặc trưng 512 chiều. Vector này đi qua head hồi quy 512-128-1 để dự đoán lực. Thiết kế này cho phép mô hình học trực tiếp các biến dạng thị giác mà không cần bước phát hiện và tracking marker tại inference.

Sơ đồ kiến trúc nên vẽ:

```text
Reference image         Deformed frame
      |                       |
      v                       v
 grayscale + resize + normalize [0, 1]
      |                       |
      +---------- stack 2 channels ----------+
                         |
                         v
              Input tensor (2, 240, 320)
                         |
                         v
                 ResNet18 backbone
             conv1: 2 channels, fc: Identity
                         |
                         v
                  Feature vector 512
                         |
                         v
             Regression head 512 -> 128 -> 1
                         |
                         v
                  Predicted force
```

