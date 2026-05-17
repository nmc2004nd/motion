# Kiến Trúc `force_model` Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code trong `src/force_model`. Mục tiêu là giải thích rõ kiến trúc `force_model`, vì sao thiết kế theo hướng PointNet-style, dữ liệu đi qua mô hình như thế nào, ưu nhược điểm ra sao và các câu hỏi có thể gặp khi phản biện đồ án.

## 1. Vai trò của `force_model`

`force_model` là mô hình học sâu ước lượng lực từ tập marker đã tracking. Khác với `force_poly`, mô hình này không rút gọn toàn bộ trường dịch chuyển thành 9 đặc trưng thủ công. Thay vào đó, mỗi marker được giữ như một điểm dữ liệu riêng, gồm vị trí ban đầu và vector dịch chuyển.

Đầu vào của một marker:

```text
[x_ref_norm, y_ref_norm, dx_norm, dy_norm]
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `x_ref_norm` | Tọa độ x của marker trong ảnh reference, chuẩn hóa theo chiều rộng ảnh |
| `y_ref_norm` | Tọa độ y của marker trong ảnh reference, chuẩn hóa theo chiều cao ảnh |
| `dx_norm` | Dịch chuyển theo x, chuẩn hóa theo chiều rộng ảnh |
| `dy_norm` | Dịch chuyển theo y, chuẩn hóa theo chiều cao ảnh |

Luồng tổng quát:

```text
trial images + force log
        |
        v
prepare.py
  detect marker reference
  track marker bằng Lucas-Kanade
  sync frame với force log
        |
        v
cache .npz
  ref_pts, disp, valid, force
        |
        v
TrialDataset
  mỗi frame -> (N_max, 4) + mask
        |
        v
ForceNet
  shared MLP per marker
  mask-aware max pool + mean pool
  regression head
        |
        v
predicted force
```

Trong config hiện tại:

| Thành phần | Giá trị |
|---|---:|
| Input mỗi marker | 4 feature |
| `N_max` | Tự lấy theo số marker lớn nhất trong toàn bộ cache |
| Shared MLP | 4 -> 64 -> 128 |
| Pooling | Mask-aware max pool + mean pool |
| Vector sau pooling | 256 chiều |
| Head | 256 -> 64 -> 1 |
| Dropout | 0.1 |
| Tổng số tham số | 25 537 |
| Loss | Huber loss, delta = 1.0 |
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Early stopping | 10 epoch không cải thiện validation MAE |

Code chính:

| File | Vai trò |
|---|---|
| `src/force_model/prepare.py` | Tạo cache `.npz` từ trial ảnh và force log |
| `src/force_model/dataset.py` | Đọc cache, tạo input `(N_max, 4)`, mask và augmentation |
| `src/force_model/model.py` | Định nghĩa ForceNet: shared MLP, pooling, head |
| `src/force_model/train.py` | Huấn luyện, early stopping, lưu checkpoint tốt nhất |
| `src/force_model/eval.py` | Đánh giá tổng thể, theo trial, xuất plot và CSV |
| `src/force_model/config.yaml` | Cấu hình dữ liệu, mô hình, train và augment |

## 2. Chuẩn bị dữ liệu trong `prepare.py`

`prepare.py` là bước offline preprocessing. Mỗi trial được chuyển thành một file `.npz` chứa dữ liệu marker-level đã sẵn sàng cho training.

Các bước chính:

1. Tìm ảnh reference của session. Nếu không có reference, dùng frame đầu tiên của trial làm fallback.
2. Tiền xử lý ảnh reference và detect marker.
3. Đọc force log của trial.
4. Đồng bộ timestamp frame với force bằng nearest-neighbor trong cửa sổ `force_sync_tolerance_s`.
5. Tracking marker từ reference sang từng frame bằng Lucas-Kanade.
6. Tính displacement:

```text
disp = tracked_pts - ref_pts
```

7. Trừ zero offset của force trong `session.yaml`.
8. Lưu cache `.npz`.

Mỗi cache chứa:

| Trường | Shape | Ý nghĩa |
|---|---:|---|
| `ref_pts` | `(N, 2)` | Tọa độ marker reference |
| `disp` | `(T, N, 2)` | Dịch chuyển marker theo từng frame |
| `valid` | `(T, N)` | Marker tracking hợp lệ hay không |
| `force` | `(T,)` | Lực đã trừ zero offset |
| `force_raw` | `(T,)` | Lực gốc trước khi trừ offset |
| `ts_mono` | `(T,)` | Timestamp frame |
| `image_w`, `image_h` | scalar | Kích thước ảnh |
| `n_markers` | scalar | Số marker trong reference |
| `trial_id`, `session_id` | string | Định danh trial/session |

### Vì sao cần cache `.npz`?

Detection marker, Lucas-Kanade tracking và đồng bộ force là các bước tốn thời gian và có nhiều I/O. Nếu thực hiện lại mỗi epoch, training sẽ chậm và khó tái lập. Cache giúp:

- Tách riêng preprocessing và training.
- Training nhanh hơn vì chỉ đọc dữ liệu marker-level.
- Dễ dùng cùng cache cho nhiều mô hình như `force_model` và `force_poly`.
- Giữ split/eval ổn định giữa các lần chạy.

### Vì sao đồng bộ force bằng nearest-neighbor?

Frame camera và force log thường không có timestamp trùng tuyệt đối. Code chọn mẫu force gần nhất với timestamp frame, miễn là chênh lệch nhỏ hơn `force_sync_tolerance_s` mặc định 0.1 giây. Cách này đơn giản, ổn định và phù hợp nếu tần số đo force đủ cao so với camera.

Nếu chênh lệch timestamp lớn hơn tolerance, frame bị bỏ qua. Điều này tránh gán nhãn force sai thời điểm.

## 3. Dataset và biểu diễn input trong `dataset.py`

`TrialDataset` đọc toàn bộ cache vào RAM. Mỗi sample là một frame:

```text
feat:  (N_max, 4)
mask:  (N_max,)
force: scalar
```

Với mỗi marker thật:

```text
feat_i = [x/W, y/H, dx/W, dy/H]
```

Nếu trial có ít marker hơn `N_max`, phần còn lại được pad bằng 0. `mask` đánh dấu marker thật và hợp lệ:

```text
mask_i = True  nếu marker i tracking hợp lệ
mask_i = False nếu marker i bị mất tracking hoặc là padding
```

### Vì sao cần `N_max` và padding?

Các trial có thể có số marker khác nhau. PyTorch batch cần tensor cùng shape, nên dataset lấy `N_max` là số marker lớn nhất trong toàn bộ cache và pad các trial nhỏ hơn lên cùng kích thước.

Padding giúp batch hóa đơn giản:

```text
Batch input: (B, N_max, 4)
Batch mask:  (B, N_max)
```

Nếu không có mask, mô hình sẽ tưởng các điểm pad 0 cũng là marker thật. Vì vậy mask là thành phần bắt buộc.

### Vì sao chia dữ liệu theo trial?

`split_trials` chia danh sách `.npz` theo trial, không chia theo frame. Lý do là các frame trong cùng trial có tương quan thời gian cao. Nếu chia frame ngẫu nhiên, model có thể train trên frame gần giống test, làm kết quả đánh giá quá lạc quan. Chia theo trial giúp kiểm tra khả năng tổng quát hóa trên lần đo mới.

## 4. Augmentation trong dataset

Augmentation chỉ áp dụng cho train set, không áp dụng cho validation/test. Config hiện tại:

| Augmentation | Giá trị | Ý nghĩa |
|---|---:|---|
| Flip ngang | 0.5 | Đảo tọa độ x và đổi dấu dx |
| Flip dọc | 0.5 | Đảo tọa độ y và đổi dấu dy |
| Rotation | +/- 5 độ | Xoay vị trí marker và vector dịch chuyển |
| Noise displacement | 0.5 px | Thêm nhiễu Gaussian vào dx, dy |

Lý do dùng augmentation:

- Tăng độ đa dạng dữ liệu.
- Giúp mô hình ít phụ thuộc vào hướng đặt cảm biến/camera.
- Mô phỏng nhiễu tracking nhỏ trong Lucas-Kanade.
- Giảm overfit khi số trial chưa quá lớn.

Điểm cần lưu ý: augmentation hình học phải biến đổi cả vị trí marker và vector dịch chuyển một cách nhất quán. Code đã xử lý điều này: khi flip thì đổi dấu displacement tương ứng; khi rotate thì xoay cả tọa độ và vector displacement.

## 5. Kiến trúc ForceNet trong `model.py`

ForceNet có ba khối chính:

```text
Input (B, N, 4)
        |
        v
Shared MLP per marker
  Conv1d(4 -> 64, kernel=1) + BatchNorm + ReLU
  Conv1d(64 -> 128, kernel=1) + BatchNorm + ReLU
        |
        v
Mask-aware pooling
  max pool + mean pool
        |
        v
Regression head
  Linear(256 -> 64) + ReLU + Dropout
  Linear(64 -> 1)
        |
        v
Force scalar
```

### 5.1. Vì sao gọi là PointNet-style?

PointNet là kiến trúc xử lý tập điểm bằng cách:

1. Dùng cùng một MLP cho từng điểm.
2. Tổng hợp thông tin toàn cục bằng pooling bất biến với thứ tự điểm.
3. Dùng vector toàn cục để phân loại hoặc hồi quy.

`force_model` dùng đúng tư tưởng này. Mỗi marker là một điểm có 4 feature. Shared MLP biến mỗi marker thành embedding 128 chiều. Sau đó pooling gom toàn bộ marker thành một vector global để dự đoán lực.

### 5.2. Vì sao dùng Conv1d kernel=1 thay vì Linear?

`Conv1d(kernel_size=1)` trên tensor `(B, C, N)` tương đương với một Linear layer áp dụng độc lập cho từng marker và dùng chung trọng số cho mọi marker. Cách viết này tiện cho batch marker set:

```text
(B, 4, N) -> Conv1d(4, 64, 1) -> (B, 64, N)
```

Ưu điểm:

- Trọng số được share trên mọi marker.
- Không phụ thuộc vào số lượng marker.
- Tính toán hiệu quả trên GPU.
- Giữ đúng tính chất "mỗi marker xử lý cùng một hàm".

### 5.3. Vì sao dùng BatchNorm1d?

BatchNorm giúp ổn định phân phối activation sau mỗi lớp Conv1d, từ đó training ổn định hơn. Input đã được chuẩn hóa theo kích thước ảnh, nhưng activation bên trong mạng vẫn có thể lệch scale. BatchNorm giúp giảm vấn đề này và cho phép dùng learning rate ổn định hơn.

### 5.4. Vì sao dùng max pool và mean pool cùng lúc?

Sau shared MLP, tensor có shape:

```text
feats: (B, 128, N)
```

Pooling tạo vector global:

```text
max_pool:  (B, 128)
mean_pool: (B, 128)
concat:    (B, 256)
```

Max pool và mean pool mang ý nghĩa khác nhau:

| Pooling | Ý nghĩa |
|---|---|
| Max pool | Bắt các marker có đáp ứng mạnh nhất, gần vùng chịu lực lớn |
| Mean pool | Bắt xu hướng biến dạng trung bình trên toàn bộ cảm biến |

Nếu chỉ dùng mean pool, mô hình có thể bỏ lỡ điểm chịu tải cục bộ mạnh. Nếu chỉ dùng max pool, mô hình có thể quá nhạy với nhiễu hoặc một marker bất thường. Kết hợp cả hai giúp mô hình có thông tin về cả cực trị và phân bố tổng thể.

### 5.5. Vì sao cần mask-aware pooling?

Không phải marker nào cũng tracking thành công. Ngoài ra, dữ liệu còn có padding tới `N_max`. Nếu pooling trực tiếp trên tất cả điểm, các marker invalid và padding sẽ ảnh hưởng đến kết quả.

Trong code:

- Max pool: marker invalid được thay bằng giá trị rất âm trước khi lấy max.
- Mean pool: chỉ cộng marker valid và chia cho số marker valid.
- Nếu frame không có marker valid, max pool được set về 0 để tránh giá trị vô nghĩa.

Nhờ đó mô hình chỉ học từ marker hợp lệ.

## 6. Số tham số của mô hình

Với config hiện tại:

```text
Input dim = 4
Shared MLP = [64, 128]
Head hidden = 64
```

Số tham số:

| Layer | Tham số |
|---|---:|
| Conv1d 4 -> 64 | 4*64 + 64 = 320 |
| BatchNorm 64 | 64 gamma + 64 beta = 128 |
| Conv1d 64 -> 128 | 64*128 + 128 = 8 320 |
| BatchNorm 128 | 128 gamma + 128 beta = 256 |
| Linear 256 -> 64 | 256*64 + 64 = 16 448 |
| Linear 64 -> 1 | 64*1 + 1 = 65 |
| Tổng | 25 537 |

Mô hình lớn hơn `force_poly` nhưng nhỏ hơn rất nhiều so với `force_cnn`. Đây là hướng trung gian: giữ nhiều thông tin không gian hơn `force_poly`, nhưng vẫn nhẹ hơn mô hình ảnh end-to-end.

## 7. Huấn luyện trong `train.py`

Training dùng:

| Thành phần | Lý do |
|---|---|
| Huber loss | Giảm ảnh hưởng của outlier do tracking hoặc force sync lỗi |
| Adam | Tối ưu ổn định cho mạng nhỏ |
| Weight decay 1e-4 | Regularization nhẹ để giảm overfit |
| Early stopping theo validation MAE | Chọn checkpoint có sai số validation tốt nhất |
| Batch size 64 | Cân bằng giữa ổn định gradient và tốc độ |
| Augmentation train-only | Tăng tổng quát hóa nhưng giữ validation/test trung thực |

Checkpoint tốt nhất được chọn theo `val_mae`. File `best.pt` lưu:

| Trường | Ý nghĩa |
|---|---|
| `epoch` | Epoch tốt nhất |
| `model_state` | Trọng số mô hình |
| `val_mae` | Validation MAE tốt nhất |
| `n_max` | Số marker tối đa dùng khi padding |
| `config` | Cấu hình huấn luyện |

Kết quả training hiện tại:

| Metric test | Giá trị |
|---|---:|
| MAE | 0.0564 |
| RMSE | 0.0798 |
| R² | 0.995 |

Mô hình đạt R² cao, cho thấy biểu diễn marker-level có khả năng học tốt quan hệ giữa biến dạng và lực.

## 8. Đánh giá trong `eval.py`

`eval.py` load checkpoint tốt nhất, dựng lại dataset theo cùng split seed và tính:

| Output | Ý nghĩa |
|---|---|
| `test_predictions.npz` | Dự đoán, nhãn thật và metric tổng thể |
| `test_breakdown.csv` | Metric từng trial |
| `test_scatter.png` | Scatter predicted vs actual |
| `test_timeseries_trial0.png` | Dự đoán theo thời gian trên trial đầu tiên |

Đánh giá theo từng trial rất quan trọng vì nó cho biết mô hình có bị yếu ở một số điều kiện tiếp xúc cụ thể hay không. Một MAE trung bình tốt có thể che giấu việc mô hình sai nhiều trên một trial khó.

## 9. Vì sao kiến trúc này hợp lý?

### 9.1. Marker là một tập điểm, không phải vector cố định tự nhiên

Thứ tự marker trong cache không nên quyết định kết quả dự đoán. Điều quan trọng là phân bố vị trí và dịch chuyển của marker. Kiến trúc PointNet-style phù hợp vì shared MLP xử lý từng điểm giống nhau, còn pooling làm kết quả gần như bất biến với thứ tự marker.

### 9.2. Giữ nhiều thông tin hơn `force_poly`

`force_poly` nén toàn bộ biến dạng thành 9 số. Cách đó nhẹ và dễ giải thích nhưng mất thông tin không gian chi tiết. `force_model` giữ từng marker, nên có thể học các mẫu biến dạng theo vị trí:

- Vùng nào biến dạng mạnh.
- Marker gần tâm hay rìa cảm biến dịch chuyển nhiều.
- Phân bố vector dịch chuyển có tập trung hay lan rộng.
- Các điểm cực trị và trung bình có quan hệ thế nào với lực.

### 9.3. Nhẹ hơn CNN end-to-end

CNN học trực tiếp từ ảnh nên có thể chính xác hơn, nhưng cần nhiều tham số và thời gian train lớn. `force_model` dùng dữ liệu marker-level đã được rút trích, nên bỏ qua phần lớn thông tin ảnh không cần thiết và giảm chi phí tính toán.

### 9.4. Robust với marker bị mất

Nhờ mask-aware pooling, mô hình có thể xử lý frame có một phần marker tracking lỗi. Đây là điểm quan trọng trong dữ liệu xúc giác thực tế vì marker có thể bị che, nhiễu hoặc biến dạng mạnh.

## 10. Hạn chế của `force_model`

| Hạn chế | Nguyên nhân | Hướng xử lý |
|---|---|---|
| Phụ thuộc vào detection và LK tracking | Input đến từ marker displacement | Cải thiện tracking, lọc outlier, kiểm tra valid mask |
| Không dùng texture ảnh raw | Chỉ nhận marker tọa độ và displacement | Dùng `force_cnn` nếu cần khai thác ảnh đầy đủ |
| Pooling làm mất một phần cấu trúc không gian | Max/mean gom toàn bộ marker thành vector global | Thêm attention, local neighborhood hoặc graph network |
| Không đảm bảo output không âm | Head cuối là linear | Clip output hoặc dùng Softplus nếu cần |
| BatchNorm phụ thuộc batch statistics | Batch nhỏ hoặc phân phối khác có thể làm lệch activation | Tăng batch size, dùng LayerNorm/GroupNorm nếu cần |
| Augmentation có thể không đúng với mọi setup vật lý | Flip/rotation giả định tính đối xứng tương đối | Tắt hoặc giới hạn augmentation nếu cảm biến không đối xứng |

## 11. Câu hỏi có thể gặp khi phản biện

### Q1. Tại sao chọn kiến trúc PointNet-style cho `force_model`?

Vì dữ liệu đầu vào là tập marker, mỗi marker có vị trí và vector dịch chuyển. Thứ tự marker không nên ảnh hưởng đến lực dự đoán. PointNet-style phù hợp vì xử lý từng marker bằng cùng một MLP rồi dùng pooling để tạo biểu diễn toàn cục không phụ thuộc nhiều vào thứ tự điểm.

### Q2. Vì sao không dùng MLP phẳng trên toàn bộ vector marker?

Nếu flatten toàn bộ marker thành một vector, mô hình sẽ phụ thuộc vào thứ tự marker và số lượng marker cố định. Khi số marker thay đổi hoặc có marker mất tracking, flatten khó xử lý. Shared MLP + mask-aware pooling linh hoạt hơn và phù hợp với dữ liệu dạng tập.

### Q3. Vì sao input mỗi marker chỉ gồm 4 feature?

4 feature này chứa thông tin tối thiểu cần thiết: vị trí ban đầu của marker và vector dịch chuyển của nó. Vị trí cho biết marker nằm ở đâu trên cảm biến, displacement cho biết biến dạng tại vị trí đó. Từ tập các marker này, mô hình có thể học phân bố biến dạng để suy ra lực.

### Q4. Vì sao cần cả vị trí reference, không chỉ cần displacement?

Cùng một vector dịch chuyển có ý nghĩa khác nhau tùy vị trí trên cảm biến. Marker ở tâm, rìa hoặc gần vùng tiếp xúc có thể phản ánh lực khác nhau. Vị trí reference giúp mô hình học mối quan hệ không gian giữa biến dạng và lực.

### Q5. Vì sao chuẩn hóa `x, y, dx, dy` theo kích thước ảnh?

Chuẩn hóa giúp input không phụ thuộc trực tiếp vào đơn vị pixel và độ phân giải ảnh. Nó cũng đưa các giá trị về scale ổn định hơn để mạng học dễ hơn. Nếu dùng pixel gốc, cùng một biến dạng vật lý nhưng camera/độ phân giải khác có thể tạo giá trị khác.

### Q6. Vì sao `dx` chia cho W và `dy` chia cho H?

Vì `dx` nằm trên trục ngang của ảnh nên chuẩn hóa theo chiều rộng, còn `dy` nằm trên trục dọc nên chuẩn hóa theo chiều cao. Cách này đưa displacement về tỷ lệ tương đối so với kích thước ảnh theo từng trục.

### Q7. Vì sao dùng Conv1d kernel=1?

Conv1d kernel=1 tương đương với một Linear layer áp dụng cho từng marker và share trọng số trên mọi marker. Nó giúp xử lý tensor dạng `(B, C, N)` hiệu quả, không phụ thuộc trực tiếp vào số marker và giữ đúng tư tưởng per-point MLP.

### Q8. Vì sao dùng BatchNorm trong shared MLP?

BatchNorm ổn định phân phối activation sau mỗi lớp, giúp training nhanh và ít dao động hơn. Dù input đã được normalize theo ảnh, activation bên trong mạng vẫn cần được kiểm soát scale.

### Q9. Vì sao pooling phải mask-aware?

Vì input có marker invalid do Lucas-Kanade tracking thất bại và có padding để đủ `N_max`. Nếu không mask, các điểm invalid/padding sẽ bị đưa vào max/mean pool và làm sai vector biểu diễn toàn cục.

### Q10. Vì sao dùng cả max pool và mean pool?

Max pool bắt tín hiệu mạnh nhất, thường liên quan tới vùng chịu tải cục bộ. Mean pool bắt xu hướng biến dạng tổng thể. Kết hợp cả hai giúp mô hình vừa nhạy với điểm cực trị vừa ổn định với phân bố toàn cục.

### Q11. Nếu thứ tự marker thay đổi thì mô hình có đổi kết quả không?

Về nguyên lý, shared MLP cộng với max/mean pooling làm mô hình bất biến với hoán vị marker hợp lệ. Nếu hoán vị cả feature và mask cùng nhau, kết quả pooling không đổi. Đây là lý do kiến trúc phù hợp với dữ liệu dạng tập điểm.

### Q12. Vì sao cần padding tới `N_max`?

Batch training cần các sample trong batch có cùng shape. Vì số marker giữa các trial có thể khác nhau, padding lên `N_max` giúp tạo tensor đồng nhất. Mask đảm bảo padding không ảnh hưởng đến pooling.

### Q13. Vì sao không dùng attention thay cho pooling?

Attention có thể học quan hệ giữa marker tốt hơn, nhưng phức tạp hơn và cần nhiều dữ liệu hơn. Với dataset hiện tại, max+mean pooling là lựa chọn đơn giản, ổn định và ít tham số. Attention có thể là hướng cải tiến sau khi baseline PointNet-style đã được kiểm chứng.

### Q14. Vì sao dùng Huber loss?

Huber loss ít nhạy với outlier hơn MSE và mượt hơn MAE. Dữ liệu cảm biến có thể có frame tracking sai, force sync lệch hoặc nhiễu đo lực, nên Huber loss phù hợp để giảm ảnh hưởng của các mẫu bất thường.

### Q15. Vì sao dùng Adam thay vì SGD?

Adam thích hợp cho mô hình nhỏ và dữ liệu có scale/gradient thay đổi giữa các tham số. Nó thường hội tụ nhanh hơn SGD trong các bài toán regression thực nghiệm. Với dataset hiện tại, Adam giúp training nhanh và ổn định.

### Q16. Vì sao weight decay của `force_model` nhỏ hơn `force_poly`?

`force_poly` có nhiều polynomial terms tương quan trực tiếp nên cần weight decay mạnh để tránh overfit hệ số bậc cao. `force_model` dùng mạng neural với BatchNorm và dropout, nên weight decay 1e-4 là mức regularization nhẹ hợp lý hơn.

### Q17. Vì sao có dropout ở head?

Dropout giảm phụ thuộc quá mức vào một số chiều feature sau pooling, giúp giảm overfit. Nó được đặt ở head thay vì shared MLP để regularize biểu diễn global trước khi dự đoán lực.

### Q18. Augmentation flip/rotation có làm sai ý nghĩa lực không?

Nếu cảm biến và tác động có tính đối xứng tương đối, flip/rotation nhỏ giúp mô hình học tính bất biến hình học và giảm overfit. Code biến đổi cả vị trí và displacement nhất quán, nên quan hệ hình học được giữ. Tuy nhiên, nếu setup vật lý không đối xứng, cần giảm hoặc tắt augmentation.

### Q19. Tại sao không dùng ảnh trực tiếp như CNN?

Mục tiêu của `force_model` là khai thác thông tin marker-level đã tracking, giảm chi phí so với CNN và dễ kiểm soát hơn ảnh raw. CNN có thể chính xác hơn nhưng cần nhiều tham số, thời gian train lâu hơn và khó giải thích hơn.

### Q20. Mô hình có đảm bảo lực dự đoán không âm không?

Không. Lớp cuối là tuyến tính nên có thể dự đoán lực âm khi input ngoài phân phối train hoặc không tiếp xúc. Khi triển khai thực tế, có thể clip về 0 hoặc thay output activation bằng Softplus nếu cần ràng buộc vật lý.

### Q21. Làm sao biết mô hình không chỉ học thuộc trial?

Dữ liệu được chia theo trial, không chia theo frame. Mô hình được chọn theo validation trial và đánh giá trên test trial chưa thấy trong train. Đây là cách kiểm tra tổng quát hóa tốt hơn so với split frame ngẫu nhiên. Để chắc hơn, có thể chạy nhiều seed hoặc cross-validation theo trial.

### Q22. Vì sao validation/test không dùng augmentation?

Validation/test phải phản ánh dữ liệu thật, không phải dữ liệu biến đổi nhân tạo. Augmentation chỉ dùng để train nhằm tăng khả năng tổng quát hóa. Nếu augment validation/test, metric sẽ không còn phản ánh đúng hiệu năng trên dữ liệu thực.

### Q23. Nếu một frame không có marker valid thì sao?

Trong pooling, mean pool chia cho count đã clamp tối thiểu 1, còn max pool được set về 0 nếu không có marker valid. Tuy nhiên, dự đoán trong trường hợp này sẽ kém tin cậy vì mô hình không có thông tin biến dạng thật. Khi triển khai có thể loại bỏ frame như vậy hoặc báo độ tin cậy thấp.

### Q24. Điểm khác nhau chính giữa `force_model` và `force_poly` là gì?

`force_poly` rút gọn toàn bộ frame thành 9 feature thủ công rồi dùng polynomial regression/MLP nhỏ. `force_model` giữ từng marker như một điểm riêng và học cách tổng hợp bằng shared MLP + pooling. Vì vậy `force_model` giữ nhiều thông tin không gian hơn nhưng khó diễn giải hơn.

### Q25. Điểm khác nhau chính giữa `force_model` và `force_cnn` là gì?

`force_cnn` học trực tiếp từ ảnh, nên có thể khai thác cả texture, biến dạng marker và tín hiệu thị giác khác. `force_model` chỉ dùng marker đã tracking, nên nhẹ hơn và tập trung hơn, nhưng phụ thuộc vào chất lượng detection/tracking.

### Q26. Hướng cải tiến kiến trúc là gì?

Có thể cải tiến bằng cách thêm feature mỗi marker như độ tin cậy LK, khoảng cách tới tâm, độ lớn displacement; thay pooling bằng attention; dùng graph neural network để học quan hệ lân cận giữa marker; hoặc dự đoán thêm uncertainty để biết khi nào model không chắc chắn.

## 12. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn như sau:

> Mô hình `force_model` sử dụng kiến trúc PointNet-style để ước lượng lực từ tập marker đã tracking. Mỗi marker được biểu diễn bởi vị trí chuẩn hóa trong ảnh reference và vector dịch chuyển chuẩn hóa. Các marker được xử lý độc lập bởi shared MLP 4-64-128, sau đó tổng hợp bằng mask-aware max pooling và mean pooling để tạo vector đặc trưng toàn cục 256 chiều. Vector này đi qua MLP head 256-64-1 để dự đoán lực. Thiết kế này phù hợp với dữ liệu dạng tập điểm, không phụ thuộc vào thứ tự marker, xử lý được marker mất tracking nhờ mask và giữ nhiều thông tin không gian hơn mô hình đặc trưng thủ công.

Sơ đồ kiến trúc nên vẽ:

```text
Per-frame marker set
  (N_max, 4) + valid mask
        |
        v
Shared MLP per marker
  Conv1d 4->64->128
        |
        v
Mask-aware pooling
  max pool + mean pool
        |
        v
Global feature vector
  256 dims
        |
        v
Regression head
  Linear 256->64->1
        |
        v
Predicted force
```

