# Kiến Trúc `shape_model` Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code trong `src/shape_model`. Mục tiêu là giải thích kiến trúc mô hình phân loại shape, cách dữ liệu được đưa vào model, lý do lựa chọn thiết kế, các hạn chế hiện tại và các câu hỏi có thể gặp khi phản biện đồ án.

## 1. Vai trò của `shape_model`

`shape_model` là module độc lập để dự đoán hình dạng object từ từng frame ảnh. Bài toán được xây dựng như một bài toán classification 4 lớp:

| Class nội bộ | Ý nghĩa |
|---|---|
| `circle` | Vật hình tròn |
| `square` | Vật hình vuông |
| `triangle` | Vật hình tam giác |
| `unknown` | Không rõ/khác/chưa gán đúng shape |

Khác với các module ước lượng lực, `shape_model` không dự đoán giá trị liên tục. Output của mô hình là logits cho từng class, sau đó dùng softmax để lấy xác suất.

Luồng xử lý tổng quát:

```text
data/sessions
  trial.yaml + frames.csv + frames/*.jpg
        |
        v
dataset.py
  đọc label shape ở trial level
  index từng frame thành một sample
  resize + normalize ảnh RGB
        |
        v
input tensor (3, 224, 224)
        |
        v
ShapeClassifier
  SmallCNN hoặc ResNet18 backbone
        |
        v
classification head
        |
        v
logits 4 lớp
        |
        v
softmax -> predicted shape
```

Trong config hiện tại:

| Thành phần | Giá trị |
|---|---:|
| Input | Ảnh RGB một frame |
| Kích thước ảnh | 224 x 224 |
| Số class | 4 |
| Backbone | `small_cnn` |
| Pretrained | `false` |
| Head | 256 -> 128 -> 4 |
| Dropout | 0.2 |
| Loss | Weighted CrossEntropyLoss |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Early stopping | 8 epoch không cải thiện validation accuracy |

Code chính:

| File | Vai trò |
|---|---|
| `src/shape_model/dataset.py` | Index trial, normalize label, load ảnh, augmentation, split theo trial |
| `src/shape_model/model.py` | Định nghĩa `ShapeClassifier`, `SmallCNN`, `_ResNet18Backbone` |
| `src/shape_model/train.py` | Huấn luyện, class weights, metrics, checkpoint |
| `src/shape_model/eval.py` | Đánh giá train/val/test, confusion matrix, per-trial CSV |
| `src/shape_model/infer.py` | Inference một ảnh hoặc ảnh random từ dataset |
| `src/shape_model/config.yaml` | Cấu hình dữ liệu, model, train, augment, infer |

## 2. Dữ liệu đầu vào trong `dataset.py`

Mỗi trial có thể chứa:

```text
trial.yaml
frames.csv
frames/frame_*.jpg
```

Label shape nằm ở mức trial:

```yaml
shape: circle
```

Mỗi frame trong trial đó trở thành một sample classification. Nếu trial có 80 frame và label là `circle`, thì cả 80 frame đều có target `circle`.

### 2.1. `TrialIndex`

`dataset.py` gom thông tin một trial vào `TrialIndex`:

| Trường | Ý nghĩa |
|---|---|
| `session_id` | Tên session, ví dụ `c1`, `s1`, `t1` |
| `trial_id` | Tên trial |
| `shape` | Label đã normalize về class nội bộ |
| `frames` | Danh sách đường dẫn ảnh frame |

### 2.2. Normalize label

Code dùng `normalize_shape_label` để đưa nhiều cách ghi nhãn về 4 class chuẩn. Ví dụ:

| Input trong metadata | Class nội bộ |
|---|---|
| `circle`, `round`, `tron`, `tròn`, `c` | `circle` |
| `square`, `vuong`, `vuông`, `s` | `square` |
| `triangle`, `tam giac`, `tam giác`, `t` | `triangle` |
| `unknown`, `unknow`, `other`, `none`, `1` | `unknown` |

Lý do cần bước này là dữ liệu có thể được nhập bằng tiếng Anh, tiếng Việt không dấu, tiếng Việt có dấu hoặc viết tắt. Normalize giúp model không bị tạo ra nhiều class giả chỉ vì khác cách viết.

### 2.3. Fallback shape theo session

Trong config hiện tại:

```yaml
fallback_shape_by_session:
  "1": "unknown"
```

Điều này nghĩa là nếu các trial trong session `data/sessions/1` chưa có field `shape`, code vẫn gán label fallback là `unknown`.

Cách này giúp dùng lại dữ liệu cũ chưa có metadata shape, nhưng cần được hiểu rõ: fallback không phải ground truth mạnh như label ghi trực tiếp trong `trial.yaml`. Khi dữ liệu hoàn chỉnh hơn, nên ghi `shape` trực tiếp trong từng `trial.yaml`.

### 2.4. Sampling frame trong trial

Config hiện tại:

```yaml
max_frames_per_trial: 80
```

Nếu trial có nhiều hơn 80 frame, code dùng uniform subsampling cố định để lấy tối đa 80 frame. Mục đích:

- Tránh trial dài thống trị loss và metric.
- Giữ coverage theo thời gian trong trial.
- Giảm thời gian train.
- Giữ kết quả deterministic, vì không random sampling mỗi lần chạy.

### 2.5. Chia train/val/test theo trial

Split được thực hiện theo trial, không chia ngẫu nhiên theo frame:

```yaml
split:
  train: 0.7
  val: 0.15
  test: 0.15
split_seed: 42
```

Lý do rất quan trọng: các frame trong cùng trial thường liên tiếp và rất giống nhau. Nếu chia theo frame, train có thể chứa frame gần như trùng với frame trong test, làm accuracy quá lạc quan. Chia theo trial giúp đánh giá khả năng tổng quát hóa trên lần chụp mới.

Code cũng group theo label trước khi split để mỗi class có cơ hội xuất hiện trong train/val/test khi số trial đủ lớn.

## 3. Cách tạo input ảnh

Mỗi frame được xử lý bởi `load_image_normalized`:

1. Đọc ảnh bằng OpenCV.
2. Convert BGR sang RGB.
3. Resize về `image_size`.
4. Normalize pixel về `[0, 1]`.
5. Convert HWC sang CHW để đưa vào PyTorch.

Kết quả:

```text
image:  (3, 224, 224) float32
target: scalar long, index của class
```

### Vì sao dùng RGB thay vì grayscale?

Bài toán shape chủ yếu dựa vào contour và cấu trúc vật thể, nên grayscale có thể đủ. Tuy nhiên code hiện tại dùng RGB vì:

- Tương thích trực tiếp với ResNet18 mặc định của torchvision, vốn nhận 3 kênh.
- Giữ lại thông tin màu nếu điều kiện ảnh hoặc vật thể có tín hiệu màu hữu ích.
- Không cần sửa conv đầu của ResNet18 khi dùng pretrained.
- Với `small_cnn`, chi phí thêm từ 1 kênh lên 3 kênh vẫn nhỏ.

Nếu sau này muốn tối ưu tốc độ hoặc giảm phụ thuộc màu, có thể thử grayscale 1 kênh như một ablation.

## 4. Augmentation ảnh

Augmentation chỉ áp dụng cho train set. Validation/test không augment để metric phản ánh dữ liệu thật.

Config hiện tại:

| Augmentation | Giá trị | Ý nghĩa |
|---|---:|---|
| Flip ngang | 0.5 | Lật ảnh theo chiều ngang |
| Rotation | +/- 8 độ | Xoay nhẹ ảnh quanh tâm |
| Brightness jitter | 0.12 | Dịch độ sáng |
| Contrast jitter | 0.12 | Thay đổi tương phản |

### Vì sao cần augmentation?

Dữ liệu shape có thể bị ảnh hưởng bởi:

- Vị trí object trong ảnh.
- Góc đặt object.
- Độ sáng camera.
- Độ tương phản giữa object và nền.
- Nhiễu hoặc khác biệt giữa các session.

Augmentation giúp model học đặc trưng shape ổn định hơn thay vì ghi nhớ điều kiện chụp cụ thể.

### Vì sao rotation chỉ nhỏ?

Rotation nhỏ mô phỏng sai lệch nhẹ khi đặt vật thể/camera. Nếu xoay quá lớn, một số hình có thể thay đổi cách xuất hiện hoặc bị crop mạnh sau resize/warp. Với phân loại hình cơ bản, rotation lớn có thể vẫn hợp lý, nhưng config hiện tại chọn mức bảo thủ `8 độ` để giảm nguy cơ tạo sample không tự nhiên.

## 5. Kiến trúc trong `model.py`

`ShapeClassifier` gồm hai phần:

```text
backbone -> feature vector -> classification head -> logits
```

Backbone hỗ trợ:

| Backbone | Vai trò |
|---|---|
| `small_cnn` | Mô hình nhẹ, chạy offline/CPU, phù hợp dataset nhỏ |
| `resnet18` | Backbone mạnh hơn, có thể dùng pretrained ImageNet |

Config hiện tại dùng:

```yaml
backbone: "small_cnn"
pretrained: false
hidden_head: 128
dropout: 0.2
```

Luồng với config hiện tại:

```text
Input (B, 3, 224, 224)
        |
        v
SmallCNN
  ConvBlock(3 -> 32, kernel=5) + MaxPool
  ConvBlock(32 -> 64) + MaxPool
  ConvBlock(64 -> 128) + MaxPool
  ConvBlock(128 -> 256, no pool)
  AdaptiveAvgPool2d(1)
        |
        v
Feature vector (B, 256)
        |
        v
Classification head
  Linear(256 -> 128)
  ReLU
  Dropout(0.2)
  Linear(128 -> 4)
        |
        v
Logits (B, 4)
```

## 6. `SmallCNN`

`SmallCNN` là backbone mặc định trong config. Mỗi `_ConvBlock` gồm:

```text
Conv2d
BatchNorm2d
ReLU
optional MaxPool2d(2)
```

Kiến trúc chi tiết:

| Block | Input -> Output | Kernel | Pool |
|---|---|---:|---|
| 1 | 3 -> 32 | 5x5 | Có |
| 2 | 32 -> 64 | 3x3 | Có |
| 3 | 64 -> 128 | 3x3 | Có |
| 4 | 128 -> 256 | 3x3 | Không |
| GAP | 256 feature maps -> vector 256 | - | AdaptiveAvgPool2d(1) |

Với ảnh 224 x 224, kích thước không gian đi qua các block xấp xỉ:

```text
224 x 224
  -> MaxPool -> 112 x 112
  -> MaxPool -> 56 x 56
  -> MaxPool -> 28 x 28
  -> no pool -> 28 x 28
  -> AdaptiveAvgPool2d(1) -> 1 x 1
```

### Vì sao dùng BatchNorm?

BatchNorm ổn định phân phối activation trong quá trình train, giúp model hội tụ dễ hơn và giảm nhạy với scale pixel. Với CNN nhỏ, BatchNorm là regularization nhẹ và thường cải thiện tính ổn định.

### Vì sao dùng AdaptiveAvgPool2d?

Adaptive average pooling biến feature map kích thước bất kỳ thành vector cố định. Điều này giúp head không phụ thuộc trực tiếp vào kích thước ảnh cuối cùng. Nếu sau này đổi `image_size`, backbone vẫn có thể tạo feature vector 256 chiều mà không cần sửa linear layer đầu của head.

### Vì sao `SmallCNN` hợp lý cho module này?

Shape classification 4 lớp là bài toán nhỏ hơn force regression từ ảnh. Với dữ liệu chưa quá lớn, một CNN nhỏ có các ưu điểm:

- Ít tham số hơn ResNet18.
- Train nhanh hơn.
- Chạy được trên CPU/offline.
- Ít nguy cơ overfit hơn backbone lớn.
- Không phụ thuộc tải pretrained weights qua mạng.

## 7. ResNet18 backbone

Code vẫn hỗ trợ `resnet18`:

```yaml
backbone: "resnet18"
pretrained: true
```

Khi dùng ResNet18:

```text
Input (B, 3, 224, 224)
        |
        v
ResNet18
  fc cuối được thay bằng Identity
        |
        v
Feature vector (B, 512)
        |
        v
Classification head
  Linear(512 -> hidden_head)
  ReLU
  Dropout
  Linear(hidden_head -> num_classes)
```

Khác với `force_cnn`, `shape_model` dùng ảnh RGB 3 kênh, nên không cần sửa `conv1` của ResNet18. Nếu `pretrained=True`, ResNet18 có thể dùng trực tiếp weight ImageNet.

### Khi nào nên dùng ResNet18?

Nên cân nhắc ResNet18 khi:

- Dataset có nhiều session và nhiều biến thể ánh sáng/góc nhìn.
- `small_cnn` underfit hoặc accuracy thấp.
- Có GPU hoặc môi trường đủ nhanh.
- Muốn tận dụng pretrained ImageNet.

Không nhất thiết dùng ResNet18 ngay từ đầu vì bài toán shape 4 lớp có thể đơn giản hơn nhiều so với ImageNet.

## 8. Classification head

Head hiện tại:

```text
Linear(backbone_out_dim -> hidden_head)
ReLU
Dropout(dropout)
Linear(hidden_head -> num_classes)
```

Với config hiện tại:

```text
backbone_out_dim = 256
hidden_head = 128
num_classes = 4
```

Số tham số head:

```text
Linear(256, 128): 256*128 + 128 = 32 896
Linear(128, 4):   128*4 + 4     = 516
Tổng head: 33 412
```

Lý do dùng head nhỏ:

- Backbone đã trích đặc trưng ảnh chính.
- Head chỉ cần ánh xạ feature vector sang logits class.
- Dropout giảm overfit.
- Head nhỏ giúp train nhanh và dễ ổn định.

## 9. Số tham số

Với `small_cnn`, 3 kênh input và head 256-128-4, số tham số trainable xấp xỉ:

| Thành phần | Tham số |
|---|---:|
| SmallCNN backbone | 390 432 |
| Classification head | 33 412 |
| Tổng | 423 844 |

Đây là quy mô vừa phải cho bài toán 4 lớp. So với ResNet18 khoảng 11 triệu tham số, `small_cnn` nhẹ hơn nhiều và phù hợp cho bước baseline.

## 10. Huấn luyện trong `train.py`

Training dùng:

| Thành phần | Lý do |
|---|---|
| Weighted CrossEntropyLoss | Phù hợp classification nhiều lớp, có cân bằng class |
| Class weights từ train set | Giảm bias về class có nhiều frame hơn |
| AdamW | Tối ưu ổn định, weight decay tách khỏi gradient |
| Weight decay 1e-4 | Regularization nhẹ |
| CosineAnnealingLR | Giảm learning rate dần qua epoch |
| Early stopping theo validation accuracy | Giữ checkpoint có validation accuracy tốt nhất |
| Split theo trial | Tránh leakage giữa train và test |

### 10.1. Vì sao dùng class weights?

Số frame mỗi class có thể không cân bằng. Nếu class `circle` có nhiều frame hơn `triangle`, model có thể ưu tiên dự đoán `circle` để đạt accuracy cao. Code tính weight theo công thức:

```text
weight[class] = total_samples / (num_classes * count[class])
```

Class ít sample sẽ có weight lớn hơn trong loss, giúp model chú ý hơn đến các class hiếm.

### 10.2. Metric khi train

`compute_metrics` trả:

| Metric | Ý nghĩa |
|---|---|
| `accuracy` | Tỷ lệ frame dự đoán đúng |
| `macro_accuracy` | Trung bình accuracy theo từng class có sample |

`macro_accuracy` hữu ích khi class imbalance. Nếu model chỉ đúng class nhiều mẫu nhưng sai class ít mẫu, accuracy tổng có thể vẫn cao, còn macro accuracy sẽ giảm.

### 10.3. Checkpoint

Checkpoint `best.pt` lưu:

| Trường | Ý nghĩa |
|---|---|
| `epoch` | Epoch có validation accuracy tốt nhất |
| `model_state` | Trọng số model |
| `val_accuracy` | Validation accuracy tốt nhất |
| `label_to_idx` | Mapping label -> index |
| `idx_to_label` | Mapping index -> label |
| `config` | Config huấn luyện |

Lưu mapping label là cần thiết vì inference phải biết logit index nào tương ứng với class nào.

## 11. Đánh giá trong `eval.py`

`eval.py` load checkpoint, dựng lại split theo seed trong config, rồi đánh giá trên `train`, `val` hoặc `test`.

Output gồm:

| Output | Ý nghĩa |
|---|---|
| `{split}_predictions.npz` | `preds`, `targets`, `probs`, accuracy, macro accuracy |
| `{split}_confusion.csv` | Confusion matrix dạng bảng CSV |
| `{split}_confusion.png` | Ảnh heatmap confusion matrix |
| `{split}_per_trial.csv` | Majority prediction và frame accuracy theo từng trial |

Confusion matrix có dạng:

```text
rows = true label
cols = predicted label
```

Nó cho biết model hay nhầm class nào với class nào. Ví dụ nếu hàng `triangle`, cột `circle` cao, nghĩa là nhiều ảnh tam giác bị dự đoán thành tròn.

### Vì sao cần per-trial evaluation?

Frame-level accuracy có thể cao nếu một vài trial dài được dự đoán đúng. Per-trial CSV giúp xem từng trial có ổn không:

- Trial nào bị sai nhiều.
- Majority prediction của trial có đúng không.
- Class nào dễ nhầm ở cấp trial.

Với dữ liệu video/frame liên tiếp, đánh giá theo trial thường đáng tin hơn chỉ nhìn metric frame-level.

## 12. Inference trong `infer.py`

Inference một ảnh cụ thể:

```bash
python -m src.shape_model.infer \
  --config src/shape_model/config.yaml \
  --image data/sessions/c1/trials/trial_001/frames/frame_8935.000000.jpg
```

Inference ảnh random từ dataset:

```bash
python -m src.shape_model.infer --config src/shape_model/config.yaml --random
```

Inference random trong test split:

```bash
python -m src.shape_model.infer --config src/shape_model/config.yaml --random --split test
```

Luồng inference:

```text
load image
RGB + resize + normalize
tensor (1, 3, 224, 224)
load ShapeClassifier
load checkpoint
forward -> logits
softmax -> probabilities
argmax -> predicted shape
```

Trong inference, `pretrained=False` khi dựng model là hợp lý vì checkpoint đã chứa trọng số đã train. Không cần tải lại ImageNet weights trước khi `load_state_dict`.

## 13. Vì sao kiến trúc này hợp lý?

### 13.1. Bài toán shape phù hợp với CNN

Shape là đặc trưng thị giác không gian: cạnh, góc, contour, vùng nền-vật thể. CNN phù hợp vì convolution học filter cục bộ và kết hợp chúng qua nhiều tầng để nhận diện cấu trúc lớn hơn.

### 13.2. Không cần marker tracking

Model dự đoán trực tiếp từ ảnh frame, không cần detect marker, optical flow hay feature thủ công. Điều này làm pipeline đơn giản và ít phụ thuộc vào các bước xử lý ảnh rời rạc.

### 13.3. Trial-level split giảm đánh giá ảo

Vì frame trong cùng trial rất giống nhau, split theo trial là lựa chọn quan trọng để metric phản ánh khả năng tổng quát hóa tốt hơn. Đây là điểm cần nhấn mạnh khi bảo vệ kết quả.

### 13.4. SmallCNN là baseline hợp lý

Với 4 class shape cơ bản, một CNN nhỏ có thể đủ mạnh. Bắt đầu bằng `small_cnn` giúp có baseline nhanh, ít phụ thuộc môi trường và dễ debug. ResNet18 vẫn có sẵn nếu cần tăng năng lực mô hình.

### 13.5. Weighted loss xử lý mất cân bằng dữ liệu

Nếu một số class có nhiều frame/trial hơn, weighted CrossEntropy giúp giảm thiên lệch về class lớn. Điều này quan trọng vì dữ liệu thu thập thực tế thường không cân bằng hoàn hảo.

## 14. Hạn chế của `shape_model`

| Hạn chế | Nguyên nhân | Hướng xử lý |
|---|---|---|
| Dự đoán theo frame có thể nhiễu | Một frame đơn lẻ có thể mờ, lệch sáng hoặc crop chưa tốt | Majority vote theo trial hoặc smoothing theo thời gian |
| Có thể học shortcut từ session | Mỗi shape hiện gắn với root/session cụ thể | Thu nhiều shape trong nhiều session, split theo session để kiểm tra |
| Label fallback `unknown` yếu hơn label thật | Một số trial cũ chưa có `shape` trong metadata | Bổ sung `shape` trực tiếp vào `trial.yaml` |
| Accuracy frame-level có thể quá lạc quan | Nhiều frame gần giống nhau trong một trial | Báo cáo thêm per-trial accuracy và confusion matrix |
| SmallCNN có thể underfit khi dữ liệu đa dạng | Năng lực biểu diễn thấp hơn ResNet18 | Chuyển sang ResNet18 hoặc backbone pretrained nhẹ |
| ResNet18 có thể overfit nếu dữ liệu ít | Nhiều tham số hơn nhiều so với SmallCNN | Freeze backbone, augmentation, thêm dữ liệu |
| Không giải thích trực tiếp vùng ảnh model dùng | CNN học feature ẩn | Dùng Grad-CAM/saliency hoặc ablation vùng ảnh |
| Augmentation còn đơn giản | Chưa mô phỏng blur, crop, noise mạnh | Thêm random crop, blur nhẹ, color jitter có kiểm soát |

## 15. Câu hỏi có thể gặp khi phản biện

### Q1. `shape_model` giải quyết bài toán gì?

Nó phân loại hình dạng object từ ảnh frame thành 4 class: `circle`, `square`, `triangle`, `unknown`. Đây là bài toán classification nhiều lớp, output là xác suất từng class sau softmax.

### Q2. Vì sao label đặt ở trial level nhưng sample lại là frame?

Trong một trial, object shape không đổi theo thời gian. Vì vậy mọi frame của trial có cùng label shape. Cách này tận dụng nhiều frame để train, nhưng vẫn phải split theo trial để tránh leakage.

### Q3. Vì sao phải split theo trial thay vì theo frame?

Frame trong cùng trial rất giống nhau. Nếu chia theo frame, train và test có thể chứa ảnh gần như trùng nhau, làm metric quá cao nhưng không phản ánh khả năng tổng quát hóa. Split theo trial đánh giá công bằng hơn.

### Q4. Vì sao dùng CNN cho shape classification?

CNN học tốt đặc trưng không gian như cạnh, góc, contour và texture. Shape là tín hiệu thị giác cục bộ-kết hợp-toàn cục, nên phù hợp với convolution hơn so với feature thủ công đơn giản.

### Q5. Vì sao dùng `small_cnn` thay vì ResNet18 trong config hiện tại?

Bài toán hiện chỉ có 4 class cơ bản và dataset chưa quá lớn. `small_cnn` train nhanh, chạy được CPU/offline, ít tham số và ít phụ thuộc pretrained weights. ResNet18 vẫn có thể dùng nếu cần độ chính xác cao hơn.

### Q6. Khi nào nên chuyển sang ResNet18?

Khi dữ liệu đa dạng hơn, nhiều session hơn, background/ánh sáng phức tạp hơn hoặc `small_cnn` không đạt accuracy mong muốn. ResNet18 có năng lực biểu diễn mạnh hơn và có thể dùng pretrained ImageNet.

### Q7. Vì sao dùng ảnh RGB 3 kênh?

RGB tương thích trực tiếp với ResNet18 pretrained và giữ lại mọi thông tin màu có thể hữu ích. Với `small_cnn`, chi phí thêm không lớn. Nếu muốn kiểm chứng màu có cần thiết không, có thể train thêm bản grayscale để so sánh.

### Q8. Vì sao resize về 224 x 224?

224 x 224 là kích thước chuẩn cho nhiều backbone CNN, đặc biệt ResNet. Nó giảm chi phí tính toán so với ảnh gốc nhưng vẫn giữ đủ thông tin shape cho bài toán phân loại hình cơ bản.

### Q9. Resize có làm mất chi tiết không?

Có thể, nhưng shape cơ bản thường phụ thuộc contour lớn hơn là chi tiết rất nhỏ. Nếu object nhỏ trong ảnh hoặc shape khó phân biệt, có thể tăng image size hoặc crop vùng object trước khi đưa vào model.

### Q10. Vì sao dùng Weighted CrossEntropyLoss?

Vì số frame mỗi class có thể không cân bằng. Weighted loss tăng ảnh hưởng của class ít mẫu, giảm nguy cơ model thiên về class nhiều mẫu.

### Q11. `macro_accuracy` khác gì `accuracy`?

`accuracy` tính đúng/sai trên toàn bộ frame. `macro_accuracy` tính accuracy riêng từng class rồi lấy trung bình. Macro accuracy phản ánh tốt hơn khi dữ liệu mất cân bằng.

### Q12. Vì sao cần confusion matrix?

Confusion matrix cho biết model nhầm class nào với class nào. Accuracy chỉ nói tổng thể đúng bao nhiêu, còn confusion matrix giải thích lỗi cụ thể, ví dụ `triangle` hay bị nhầm thành `circle`.

### Q13. Vì sao cần đánh giá per-trial?

Vì một trial có nhiều frame tương tự nhau. Per-trial evaluation giúp biết cả trial đó có được nhận diện đúng không, thay vì chỉ nhìn từng frame rời rạc.

### Q14. Nếu một frame bị dự đoán sai nhưng đa số frame trong trial đúng thì sao?

Trong ứng dụng thực tế có thể dùng majority vote theo trial hoặc theo cửa sổ thời gian. Vì shape không đổi nhanh theo frame, majority vote thường ổn định hơn dự đoán từng frame đơn lẻ.

### Q15. Model có thể học nhầm session thay vì shape không?

Có thể, nếu mỗi shape chỉ xuất hiện trong một session/root riêng và background/ánh sáng của session tương quan mạnh với shape. Cần thu nhiều shape ở nhiều session khác nhau hoặc kiểm tra split theo session để giảm rủi ro shortcut.

### Q16. Fallback `unknown` có rủi ro gì?

Fallback giúp dùng dữ liệu cũ thiếu metadata, nhưng label này có thể kém chắc chắn hơn label được ghi trực tiếp trong `trial.yaml`. Khi làm báo cáo nghiêm túc, nên hoàn thiện metadata để label rõ ràng.

### Q17. Vì sao dùng dropout ở classification head?

Dropout giảm overfit ở head, nhất là khi số frame nhiều nhưng thực chất số trial độc lập không quá lớn. Nó buộc model không phụ thuộc quá mạnh vào một vài feature.

### Q18. Vì sao dùng AdamW?

AdamW là optimizer ổn định cho neural network và xử lý weight decay rõ ràng hơn Adam thường. Weight decay đóng vai trò regularization, giúp giảm overfit.

### Q19. Vì sao dùng cosine scheduler?

Cosine scheduler cho learning rate cao hơn ở đầu để học nhanh, sau đó giảm dần để tinh chỉnh trọng số ổn định hơn ở cuối training.

### Q20. Checkpoint lưu `label_to_idx` và `idx_to_label` để làm gì?

Logits của model chỉ là vector số. Mapping label giúp biết index 0/1/2/3 tương ứng với class nào. Nếu không lưu mapping, inference có thể diễn giải sai output.

### Q21. Vì sao inference đặt `pretrained=False`?

Khi inference, model sẽ nạp trọng số từ checkpoint `best.pt`. Không cần tải pretrained weights nữa. `pretrained=False` tránh tải dư thừa, sau đó `load_state_dict` đưa model về đúng trạng thái đã train.

### Q22. Làm sao chứng minh model không overfit?

Cần so sánh train/val/test accuracy, macro accuracy, confusion matrix và per-trial accuracy. Nếu train rất cao nhưng val/test thấp, đó là dấu hiệu overfit. Có thể chạy thêm nhiều split seed hoặc cross-validation theo trial/session.

### Q23. Nếu class bị mất cân bằng mạnh thì nên báo cáo metric nào?

Nên báo cáo `accuracy`, `macro_accuracy`, confusion matrix và nếu bổ sung thêm thì precision/recall/F1 theo class. Accuracy đơn lẻ không đủ khi class imbalance.

### Q24. Hạn chế lớn nhất của thiết kế hiện tại là gì?

Rủi ro lớn nhất là shortcut theo session hoặc background nếu dữ liệu mỗi shape được thu trong điều kiện khác nhau. Về mặt model, `small_cnn` là baseline tốt nhưng cần kiểm chứng trên test split đủ độc lập.

### Q25. Hướng cải tiến kiến trúc là gì?

Có thể thử ResNet18 pretrained, MobileNet/EfficientNet nhẹ, grayscale input, crop vùng object, majority vote theo trial, thêm F1-score, hoặc dùng Grad-CAM để kiểm tra model tập trung vào object thay vì background.

## 16. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn như sau:

> Mô hình `shape_model` là một CNN phân loại hình dạng object từ ảnh frame RGB. Mỗi trial có một label shape trong metadata, và các frame của trial được dùng làm sample có cùng label. Ảnh được resize về 224 x 224, chuẩn hóa về [0, 1] và đưa vào `ShapeClassifier`. Với cấu hình hiện tại, backbone là `SmallCNN` gồm 4 khối convolution, BatchNorm, ReLU và MaxPool, sau đó dùng Adaptive Average Pooling để tạo vector đặc trưng 256 chiều. Vector này đi qua head phân loại 256-128-4 để tạo logits cho 4 class `circle`, `square`, `triangle`, `unknown`. Model được train bằng weighted CrossEntropyLoss để giảm ảnh hưởng mất cân bằng class, split dữ liệu theo trial để tránh leakage, và đánh giá bằng accuracy, macro accuracy, confusion matrix và per-trial accuracy.

Sơ đồ kiến trúc nên vẽ:

```text
Frame RGB image
      |
      v
resize 224 x 224 + normalize [0, 1]
      |
      v
Input tensor (3, 224, 224)
      |
      v
SmallCNN backbone
  ConvBlock 3 -> 32 + MaxPool
  ConvBlock 32 -> 64 + MaxPool
  ConvBlock 64 -> 128 + MaxPool
  ConvBlock 128 -> 256
  AdaptiveAvgPool2d(1)
      |
      v
Feature vector 256
      |
      v
Classification head
  Linear 256 -> 128
  ReLU
  Dropout 0.2
  Linear 128 -> 4
      |
      v
Logits 4 classes
      |
      v
Softmax probabilities
      |
      v
Predicted shape
```
