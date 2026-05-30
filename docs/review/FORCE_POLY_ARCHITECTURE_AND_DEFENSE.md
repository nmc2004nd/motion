# Kiến Trúc `force_poly` Và Câu Hỏi Phản Biện

Tài liệu này tổng hợp từ code trong `src/force_poly`. Mục tiêu là giải thích mô hình `force_poly` theo hướng có thể đưa vào đồ án: mô hình làm gì, kiến trúc gồm những khối nào, tại sao thiết kế như vậy, ưu nhược điểm là gì và những câu hỏi có thể gặp khi phản biện.

## 1. Vai trò của `force_poly`

`force_poly` là mô hình ước lượng lực từ trường dịch chuyển của các marker xúc giác. Khác với `force_cnn`, mô hình này không học trực tiếp từ ảnh. Khác với `force_model`, mô hình này cũng không đưa toàn bộ tập điểm marker vào mạng. Thay vào đó, pipeline rút gọn trường dịch chuyển marker thành 9 đặc trưng vô hướng, sau đó dùng khai triển đa thức và một head nhỏ để hồi quy ra lực.

Luồng xử lý tổng quát:

```text
cache trial .npz
  gồm ref_pts, disp, valid, force, image_w
        |
        v
compute_features_v1
  displacement field -> 9 scalar features
        |
        v
standardize theo train set
        |
        v
polynomial expansion bậc d
  với d=2, 9 features -> 55 monomial terms
        |
        v
head: linear hoặc mlp_small
        |
        v
force prediction
```

Trong config hiện tại, mô hình dùng:

| Thành phần | Giá trị hiện tại |
|---|---:|
| Feature set | `v1` |
| Số feature đầu vào | 9 |
| Bậc đa thức | 2 |
| Số monomial sau expansion | 55 |
| Head | `mlp_small` |
| Hidden size | 16 |
| Dropout | 0.0 |
| Tổng số tham số trainable | 913 |
| Loss | Huber loss, delta = 1.0 |
| Optimizer | AdamW |
| Weight decay | 1e-2 |
| Early stopping | 40 epoch không cải thiện validation MAE |

Code chính:

| File | Vai trò |
|---|---|
| `src/force_poly/features.py` | Trích xuất 9 đặc trưng từ marker displacement |
| `src/force_poly/dataset.py` | Đọc cache `.npz`, chia trial, tạo dataset |
| `src/force_poly/model.py` | Chuẩn hóa feature, khai triển đa thức, định nghĩa head |
| `src/force_poly/train.py` | Huấn luyện, chọn checkpoint tốt nhất theo validation MAE |
| `src/force_poly/eval.py` | Đánh giá tổng thể và theo từng trial |
| `src/force_poly/infer.py` | Chạy inference từ cặp ảnh reference và ảnh biến dạng |

## 2. Dữ liệu đầu vào và dataset

`force_poly` tái sử dụng cache `.npz` được tạo bởi pipeline chuẩn bị dữ liệu của `force_model`. Mỗi trial cache chứa:

| Trường | Ý nghĩa |
|---|---|
| `ref_pts` | Tọa độ marker ở ảnh reference, shape `(N, 2)` |
| `disp` | Chuỗi vector dịch chuyển marker theo thời gian, shape `(T, N, 2)` |
| `valid` | Mask marker tracking hợp lệ, shape `(T, N)` |
| `force` | Nhãn lực ground truth theo từng frame, shape `(T,)` |
| `image_w` | Chiều rộng ảnh, dùng để chuẩn hóa tọa độ và dịch chuyển |

Trong `dataset.py`, dữ liệu được chia theo trial bằng `split_trials`, không chia ngẫu nhiên theo frame. Đây là lựa chọn quan trọng vì các frame trong cùng một trial thường có tương quan thời gian rất mạnh. Nếu chia frame ngẫu nhiên, frame gần nhau có thể xuất hiện cả ở train và test, làm kết quả đánh giá bị lạc quan. Chia theo trial giúp test set giống tình huống thực hơn: mô hình phải dự đoán trên những lần đo chưa từng thấy.

`PolyFeatureDataset` đọc từng trial, tính feature cho toàn bộ frame bằng `compute_features_trial`, rồi nối tất cả frame thành ma trận:

```text
features: (N_total_frames, 9)
forces:   (N_total_frames,)
```

Vì sau khi trích feature dữ liệu rất nhỏ, toàn bộ feature được pre-compute và giữ trong RAM. Điều này làm training rất nhanh và không cần `num_workers` lớn.

## 3. Trích xuất đặc trưng trong `features.py`

Mô hình không dùng trực tiếp toàn bộ marker, mà dùng 9 đặc trưng tổng hợp từ trường dịch chuyển:

| # | Feature | Ý nghĩa |
|---:|---|---|
| 1 | `disp_mag_mean` | Độ lớn dịch chuyển trung bình của các marker hợp lệ |
| 2 | `disp_mag_max` | Dịch chuyển lớn nhất, thường gần vùng chịu tải mạnh |
| 3 | `disp_mag_std` | Độ phân tán dịch chuyển giữa các marker |
| 4 | `dx_mean` | Thành phần dịch chuyển trung bình theo trục x |
| 5 | `dy_mean` | Thành phần dịch chuyển trung bình theo trục y |
| 6 | `disp_mag_sum_norm` | Tổng độ lớn dịch chuyển chia cho số marker hợp lệ |
| 7 | `radial_disp_mean` | Thành phần dịch chuyển hướng kính quanh tâm marker |
| 8 | `tangential_disp_mean` | Thành phần dịch chuyển tiếp tuyến, biểu diễn xu hướng shear/xoắn |
| 9 | `valid_ratio` | Tỷ lệ marker tracking hợp lệ |

Các feature được tính mask-aware, tức là chỉ dùng marker có `valid=True`. Nếu một frame không có marker hợp lệ, hàm trả về vector 0. Khi đó bias term trong mô hình có thể học mức dự đoán nền.

### 3.1. Vì sao chuẩn hóa theo `image_w`?

Trong `compute_features_v1`, cả `dx`, `dy` và tọa độ marker đều được chia cho `image_w`. Lý do:

1. Giảm phụ thuộc vào độ phân giải ảnh. Cùng một biến dạng vật lý nhưng ảnh có kích thước khác nhau sẽ cho số pixel khác nhau.
2. Đưa các feature về scale nhỏ và ổn định hơn trước khi tính thống kê.
3. Dùng cùng một thang đo cho x và y để phép chiếu radial/tangential có ý nghĩa nhất quán.

Điểm cần lưu ý khi phản biện: nếu ảnh có tỷ lệ khung hình thay đổi lớn, việc dùng `image_w` cho cả hai trục có thể gây sai lệch nhẹ cho trục y. Trong dữ liệu hiện tại, ảnh có cùng cấu hình nên cách làm này hợp lý và đơn giản.

### 3.2. Vì sao cần các feature radial và tangential?

Các thống kê như mean, max, std chỉ mô tả độ lớn dịch chuyển, nhưng không mô tả hướng và cấu trúc không gian của biến dạng. Hai feature radial/tangential bổ sung thông tin này:

```text
radial = dot(displacement, radial_unit_vector)
tangential = cross_z(radial_unit_vector, displacement)
```

`radial_disp_mean` phản ánh xu hướng nén/giãn quanh tâm lưới marker. `tangential_disp_mean` phản ánh xu hướng trượt hoặc xoắn. Với cảm biến xúc giác, lực không chỉ làm marker dịch chuyển mạnh hơn mà còn tạo mẫu biến dạng có hướng, nên hai feature này giúp mô hình phân biệt các kiểu tiếp xúc khác nhau.

### 3.3. Vì sao có `valid_ratio`?

Tracking marker có thể bị mất do che khuất, nhiễu ảnh hoặc biến dạng quá mạnh. Nếu chỉ dùng trung bình trên các marker còn hợp lệ, mô hình không biết frame đó có bao nhiêu marker bị mất. `valid_ratio` cung cấp thông tin về độ tin cậy của quan sát. Đây là một feature chất lượng dữ liệu, không chỉ là feature vật lý.

## 4. Polynomial expansion trong `model.py`

Sau khi có vector feature:

```text
x = [x1, x2, ..., x9]
```

mô hình chuẩn hóa theo mean/std của train set:

```text
z_i = (x_i - mean_i) / std_i
```

Sau đó sinh tất cả monomial có bậc nhỏ hơn hoặc bằng `degree`. Với `degree=2`, vector sau expansion gồm:

```text
1
z1, z2, ..., z9
z1^2, z1*z2, ..., z9^2
```

Số chiều sau expansion là:

```text
P = C(D + d, d)
```

Với `D=9`, `d=2`:

```text
P = C(11, 2) = 55
```

Trong code, `_monomial_indices` dùng `combinations_with_replacement` để sinh các tổ hợp chỉ số. Bậc 0 là tuple rỗng, tương ứng với hằng số 1. Điều này đóng vai trò bias. Vì vậy, nếu dùng `head="linear"`, layer linear được đặt `bias=False` để không có hai bias trùng nhau.

### 4.1. Vì sao phải standardize trước polynomial expansion?

Nếu không chuẩn hóa, các term bậc cao có thể phóng đại scale. Ví dụ một feature có giá trị lớn hơn feature khác 10 lần thì bình phương của nó lớn hơn 100 lần. Khi đưa vào polynomial regression, điều này làm optimizer khó học, hệ số dễ bị lệch về các feature có scale lớn và term bậc cao dễ gây overfit.

Trong `PolynomialRegressor`, `fit_scaler` tính mean/std chỉ từ train set, lưu vào buffer `feat_mean` và `feat_std`, sau đó freeze trong checkpoint. Cách làm này tránh rò rỉ thông tin từ validation/test vào quá trình huấn luyện.

### 4.2. Vì sao chọn đa thức bậc 2?

Bậc 1 chỉ học quan hệ tuyến tính giữa từng feature và lực:

```text
F = a0 + a1*x1 + ... + a9*x9
```

Trong thực tế, quan hệ giữa biến dạng và lực thường có tương tác phi tuyến: độ lớn dịch chuyển trung bình có thể phụ thuộc vào độ phân tán, hướng dịch chuyển, hoặc tỷ lệ marker hợp lệ. Bậc 2 cho phép học các tương tác như:

```text
disp_mag_mean * disp_mag_std
disp_mag_max * radial_disp_mean
dx_mean * dy_mean
valid_ratio * disp_mag_mean
```

Như vậy, bậc 2 là điểm cân bằng giữa khả năng biểu diễn và nguy cơ overfit. Với 9 feature, bậc 2 tạo 55 term, vẫn rất nhỏ. Nếu tăng lên bậc 3, số term thành 220, dễ cần nhiều dữ liệu hơn và regularization mạnh hơn.

## 5. Head của mô hình: `linear` và `mlp_small`

Code hỗ trợ hai loại head:

| Head | Công thức | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `linear` | `Linear(55 -> 1, bias=False)` | Dễ diễn giải, mỗi monomial có một hệ số | Khả năng biểu diễn thấp hơn |
| `mlp_small` | `Linear(55 -> 16) -> ReLU -> Dropout -> Linear(16 -> 1)` | Biểu diễn phi tuyến hơn, vẫn rất nhỏ | Khó diễn giải trực tiếp hệ số |

Config hiện tại dùng `mlp_small`. Với `degree=2`, `n_terms=55`, `hidden_head=16`, số tham số là:

```text
Linear(55, 16): 55*16 + 16 = 896
Linear(16, 1):  16*1 + 1  = 17
Tổng: 913 tham số
```

Lý do dùng `mlp_small` thay vì `linear` là để tăng nhẹ khả năng mô hình hóa phi tuyến mà không làm mô hình quá lớn. Sau polynomial expansion bậc 2, ReLU trong MLP có thể học các tổ hợp phi tuyến mềm hơn so với hồi quy đa thức tuyến tính thuần.

Điểm cần nói rõ khi bảo vệ: với `mlp_small`, mô hình không còn diễn giải trực tiếp bằng một hệ số cho từng monomial như `linear`. Nếu mục tiêu chính là giải thích vật lý qua hệ số, nên chạy thêm bản `head="linear"` để xuất `coefficients.csv` rõ ràng hơn.

## 6. Hàm mất mát và quá trình huấn luyện

Training trong `train.py` dùng:

| Thành phần | Lý do |
|---|---|
| Huber loss | Ít nhạy với outlier hơn MSE, nhưng vẫn mượt hơn MAE |
| AdamW | Tối ưu ổn định, weight decay tách khỏi gradient update |
| Weight decay = 1e-2 | Regularize các hệ số polynomial và giảm overfit |
| CosineAnnealingLR | Giảm learning rate dần, giúp hội tụ mượt hơn ở cuối training |
| Early stopping theo validation MAE | Chọn mô hình có sai số validation thấp nhất, tránh train quá lâu |

Validation metric chính là MAE. Đây là lựa chọn phù hợp vì MAE có đơn vị cùng với lực, dễ diễn giải trong báo cáo. RMSE vẫn được tính để phản ánh lỗi lớn; R² dùng để đánh giá mức độ giải thích phương sai của dữ liệu.

Checkpoint `best.pt` lưu:

| Trường | Ý nghĩa |
|---|---|
| `model_state` | Trọng số mô hình và scaler buffer |
| `val_mae` | MAE validation tốt nhất |
| `feature_set` | Bộ feature đang dùng |
| `feature_names` | Tên feature để debug |
| `config` | Cấu hình huấn luyện |

## 7. Đánh giá và inference

Trong `eval.py`, mô hình load checkpoint tốt nhất, dự đoán trên split được chọn và xuất:

| Output | Ý nghĩa |
|---|---|
| `test_predictions.npz` | Dự đoán, nhãn thật và metric tổng thể |
| `test_breakdown.csv` | MAE/RMSE/R² theo từng trial |
| `test_scatter.png` | Biểu đồ predicted vs actual |
| `test_timeseries_trial0.png` | Dự đoán theo thời gian cho trial đầu tiên |
| `coefficients.csv` | Chỉ xuất khi `head="linear"` |

Trong `infer.py`, inference thực tế bắt đầu từ ảnh reference và ảnh biến dạng:

```text
reference image -> detect marker
reference + frame -> Lucas-Kanade tracking
tracked - ref_pts -> displacement
compute_features_v1 -> PolynomialRegressor -> predicted force
```

Điều này cho thấy `force_poly` phụ thuộc trực tiếp vào chất lượng phát hiện và tracking marker. Nếu detector hoặc optical flow sai, feature đầu vào sai và dự đoán lực cũng sai.

## 8. Vì sao kiến trúc này hợp lý?

### 8.1. Phù hợp với bài toán vật lý có dữ liệu vừa phải

Bài toán ước lượng lực từ biến dạng thường có quan hệ tương đối có cấu trúc: lực lớn làm biến dạng lớn, hướng lực ảnh hưởng đến hướng dịch chuyển, và vùng tiếp xúc tạo phân bố biến dạng đặc trưng. Thay vì để mạng lớn tự học từ ảnh, `force_poly` đưa trước các đại lượng có ý nghĩa vật lý vào mô hình.

Ưu điểm là giảm số tham số, giảm nhu cầu dữ liệu và tăng tốc huấn luyện. Với kết quả hiện tại, mô hình chỉ có 913 tham số nhưng vẫn đạt R² test rất cao.

### 8.2. Cân bằng giữa tuyến tính và phi tuyến

Polynomial expansion bậc 2 cho phép mô hình học tương tác giữa các feature mà không cần mạng sâu. `mlp_small` bổ sung thêm một lớp phi tuyến nhẹ. Đây là thiết kế trung gian giữa hồi quy tuyến tính đơn giản và deep learning lớn.

### 8.3. Dễ kiểm soát và debug

Vì input chỉ có 9 feature, có thể kiểm tra từng feature, xem phân phối train/val/test, phát hiện frame tracking lỗi, và so sánh trial khó. Đây là lợi thế lớn so với CNN end-to-end, nơi lỗi khó truy vết hơn.

## 9. Hạn chế của `force_poly`

| Hạn chế | Nguyên nhân | Cách khắc phục |
|---|---|---|
| Mất thông tin không gian chi tiết | Nhiều marker được rút gọn thành 9 số | Dùng thêm feature percentile, vùng tiếp xúc, moment không gian hoặc dùng `force_model` |
| Phụ thuộc vào tracking marker | Feature tính từ `disp` và `valid` | Cải thiện detection/LK, lọc outlier, kiểm tra valid mask |
| Khó diễn giải khi dùng `mlp_small` | Có ReLU và hidden layer | Train thêm bản `head="linear"` để phân tích hệ số |
| Có thể dự đoán lực âm | Output không bị ràng buộc dương | Clip tại inference hoặc dùng activation như Softplus |
| Bậc 2 có thể thiếu ở vùng lực lớn | Quan hệ vật liệu/tiếp xúc có thể phi tuyến mạnh hơn | Thử degree=3, thêm regularization và cross-validation |
| Chỉ hỗ trợ `feature_set="v1"` | Code hiện tại chưa có feature set khác | Thêm `v2` trong `features.py` và cập nhật config |

## 10. Câu hỏi có thể gặp khi phản biện

### Q1. Tại sao chọn `force_poly` trong khi CNN cho kết quả tốt hơn?

CNN cho độ chính xác cao hơn, nhưng `force_poly` có vai trò khác: mô hình nhẹ, huấn luyện nhanh, dễ phân tích và dùng làm baseline có ý nghĩa vật lý. Trong đồ án, `force_poly` giúp chứng minh rằng các đặc trưng biến dạng cơ bản đã chứa thông tin lực đáng kể. Nó cũng hữu ích cho hệ thống tài nguyên thấp hoặc cần dự đoán nhanh.

### Q2. Vì sao không đưa toàn bộ marker vào mô hình mà lại rút gọn thành 9 feature?

Rút gọn giúp giảm độ phức tạp và giảm nguy cơ overfit khi dữ liệu chưa quá lớn. Các feature được chọn để đại diện cho biên độ, hướng, độ phân tán và chất lượng tracking. Tuy nhiên, đánh đổi là mất một phần thông tin không gian. Vì vậy đồ án có thêm `force_model` và `force_cnn` để so sánh với hướng giữ nhiều thông tin hơn.

### Q3. Vì sao chia dữ liệu theo trial thay vì chia ngẫu nhiên theo frame?

Các frame trong cùng một trial liên tiếp nhau theo thời gian và rất giống nhau. Nếu chia ngẫu nhiên theo frame, mô hình có thể thấy các frame gần giống test trong train, làm kết quả test không phản ánh khả năng tổng quát hóa. Chia theo trial nghiêm ngặt hơn và phù hợp với tình huống triển khai thực tế.

### Q4. Vì sao dùng polynomial degree = 2?

Bậc 1 chỉ học quan hệ tuyến tính, dễ thiếu tương tác giữa các feature. Bậc 2 cho phép học cả bình phương và tích chéo giữa các feature, đủ để mô tả nhiều quan hệ phi tuyến vừa phải giữa biến dạng và lực. Bậc 3 tạo 220 term với 9 feature, dễ tăng overfit và cần nhiều dữ liệu hơn.

### Q5. Công thức số term 55 đến từ đâu?

Số monomial bậc nhỏ hơn hoặc bằng `d` với `D` feature là:

```text
C(D + d, d)
```

Với `D=9`, `d=2`:

```text
C(11, 2) = 55
```

Bao gồm 1 bias term, 9 term bậc 1 và 45 term bậc 2.

### Q6. Vì sao phải standardize trước khi khai triển đa thức?

Polynomial expansion tạo các term như `x_i^2` và `x_i*x_j`. Nếu feature không cùng scale, các term bậc cao có thể rất lớn và làm mô hình học lệch. Standardize giúp mỗi feature có trung bình 0, độ lệch chuẩn 1 trên train set, làm optimization ổn định hơn và giảm hiện tượng hệ số bị chi phối bởi scale.

### Q7. Vì sao dùng Huber loss thay vì MSE hoặc MAE?

MSE phạt lỗi lớn rất mạnh nên nhạy với outlier. MAE ít nhạy với outlier hơn nhưng gradient không mượt tại 0. Huber loss kết hợp hai ưu điểm: gần 0 hoạt động giống MSE, khi lỗi lớn hoạt động giống MAE. Điều này phù hợp với dữ liệu cảm biến có thể có frame nhiễu hoặc tracking sai.

### Q8. Vì sao dùng AdamW và weight decay lớn 1e-2?

Polynomial features có nhiều term tương quan với nhau, đặc biệt là các term bậc 2. Weight decay giúp phạt trọng số lớn, giảm overfit và làm mô hình ổn định hơn. AdamW tách weight decay khỏi gradient update, thường phù hợp hơn khi muốn regularization rõ ràng.

### Q9. `mlp_small` có còn là polynomial regression không?

Nếu dùng `head="linear"`, đó là polynomial regression thuần: output là tổ hợp tuyến tính của các monomial. Với `head="mlp_small"`, mô hình là polynomial feature expansion kết hợp MLP nhỏ. Nó vẫn dựa trên đặc trưng đa thức, nhưng không còn là hồi quy đa thức tuyến tính thuần. Cách gọi chính xác hơn là "polynomial features + small MLP regressor".

### Q10. Nếu dùng `mlp_small`, còn giải thích được hệ số không?

Không trực tiếp như `linear`. Với `linear`, mỗi monomial có một hệ số rõ ràng và có thể xuất `coefficients.csv`. Với `mlp_small`, có hidden layer và ReLU nên ảnh hưởng của từng term phụ thuộc vào trạng thái activation. Có thể phân tích gián tiếp bằng ablation feature, permutation importance hoặc train thêm bản linear để giải thích.

### Q11. Tại sao có feature `disp_mag_sum_norm` trong khi nó giống `disp_mag_mean`?

Trong code hiện tại, `disp_mag_sum_norm = sum(mag_valid) / n_valid`, về mặt toán học tương đương với mean của `mag_valid`. Vì vậy feature này có thể dư thừa so với `disp_mag_mean`. Tuy nhiên, khi giữ trong mô hình, weight decay và head có thể học cách bỏ qua feature dư thừa. Đây cũng là điểm có thể cải thiện: thay feature này bằng percentile, median hoặc tổng chuẩn hóa theo tổng marker ban đầu để mang thêm thông tin mới.

### Q12. Vì sao `valid_ratio` lại liên quan đến lực?

`valid_ratio` không trực tiếp là lực, nhưng phản ánh chất lượng quan sát. Khi biến dạng mạnh hoặc ảnh nhiễu, tracking có thể mất nhiều marker. Nếu không có `valid_ratio`, mô hình chỉ thấy thống kê trên các marker còn lại và không biết độ tin cậy của frame. Feature này giúp mô hình điều chỉnh dự đoán khi dữ liệu đầu vào kém tin cậy.

### Q13. Vì sao radial/tangential được tính quanh tâm các marker hợp lệ?

Nếu dùng tâm của tất cả marker, khi nhiều marker bị mất hoặc không hợp lệ, tâm có thể không phản ánh vùng quan sát thực tế. Code dùng trung bình tọa độ của marker hợp lệ để giảm lệch do missing marker. Sau đó, radial/tangential mô tả hướng dịch chuyển tương đối quanh tâm quan sát hiện tại.

### Q14. Mô hình có bị phụ thuộc vào camera hoặc độ phân giải không?

Có giảm phụ thuộc nhờ chuẩn hóa dịch chuyển và tọa độ theo `image_w`, nhưng không loại bỏ hoàn toàn phụ thuộc vào camera. Nếu thay đổi camera, ánh sáng, độ phân giải, lens hoặc cấu hình marker, cần kiểm tra lại calibration và có thể cần train/fine-tune lại.

### Q15. Vì sao không dùng trực tiếp pixel displacement mà lại chia cho `image_w`?

Pixel displacement phụ thuộc độ phân giải ảnh. Ví dụ cùng một dịch chuyển vật lý nhưng ảnh có độ phân giải gấp đôi sẽ tạo số pixel gần gấp đôi. Chia cho `image_w` giúp feature gần bất biến hơn với scale ảnh và làm giá trị nhỏ, ổn định cho polynomial expansion.

### Q16. Mô hình có đảm bảo lực dự đoán không âm không?

Không. Head cuối là tuyến tính nên output có thể âm, đặc biệt với input ngoài phân phối train hoặc frame không tiếp xúc. Khi triển khai, có thể clip `max(0, pred)` hoặc thay output bằng Softplus nếu yêu cầu vật lý bắt buộc lực không âm.

### Q17. Vì sao không thêm nhiều feature hơn để tăng độ chính xác?

Thêm feature có thể tăng độ chính xác nhưng cũng tăng nguy cơ overfit và làm mô hình khó phân tích hơn. Với mục tiêu baseline nhẹ, 9 feature là lựa chọn vừa đủ để mô tả biên độ, hướng, phân tán và chất lượng tracking. Nếu muốn cải thiện, nên thêm feature có ý nghĩa rõ ràng như percentile, median, moment không gian hoặc centroid vùng biến dạng.

### Q18. Làm sao biết mô hình không overfit?

Có thể xem khoảng cách train/validation/test MAE, dùng early stopping theo validation MAE và đánh giá trên test split theo trial. Kết quả test R² cao cho thấy mô hình tổng quát tốt trên split hiện tại. Tuy nhiên, để kết luận chắc hơn, nên chạy thêm nhiều seed split hoặc cross-validation theo trial.

### Q19. Nếu tracking marker sai thì mô hình xử lý thế nào?

Mô hình chỉ thấy feature sau tracking, nên không thể sửa lỗi tracking gốc. `valid_ratio` giúp mô hình biết phần nào độ tin cậy của frame, nhưng nếu displacement sai mà vẫn được đánh dấu valid, dự đoán vẫn có thể sai. Vì vậy chất lượng detection/LK là tiền đề quan trọng của `force_poly`.

### Q20. Vì sao chọn MAE để chọn checkpoint tốt nhất?

MAE có cùng đơn vị với lực, dễ diễn giải trực tiếp trong đồ án và ít nhạy với outlier hơn RMSE. Dùng validation MAE để early stopping giúp chọn mô hình có sai số trung bình tuyệt đối thấp nhất trên dữ liệu chưa dùng để train.

### Q21. Điểm mới hoặc đóng góp của mô hình này là gì?

Đóng góp chính không nằm ở việc phát minh một kiến trúc deep learning mới, mà ở thiết kế pipeline có bias vật lý: từ tracking marker, trích các đặc trưng biến dạng có ý nghĩa, mở rộng bằng polynomial interaction và huấn luyện mô hình nhẹ. Cách này tạo baseline mạnh, nhanh và dễ giải thích để so sánh với các mô hình học sâu hơn.

### Q22. Khi nào nên dùng `force_poly` thay vì `force_model` hoặc `force_cnn`?

Nên dùng `force_poly` khi cần mô hình nhẹ, tốc độ train/inference nhanh, dễ debug hoặc chạy trên thiết bị hạn chế. Nếu ưu tiên độ chính xác cao nhất và có đủ tài nguyên, `force_cnn` phù hợp hơn. Nếu muốn giữ thông tin marker nhiều hơn nhưng vẫn không dùng ảnh raw, `force_model` là lựa chọn trung gian.

## 11. Gợi ý trình bày trong đồ án

Có thể mô tả ngắn gọn kiến trúc trong đồ án như sau:

> Mô hình `force_poly` ước lượng lực bằng cách chuyển trường dịch chuyển marker thành 9 đặc trưng vô hướng có ý nghĩa vật lý, gồm độ lớn dịch chuyển, hướng dịch chuyển, độ phân tán, thành phần radial/tangential và tỷ lệ marker hợp lệ. Các đặc trưng được chuẩn hóa theo thống kê train set, sau đó khai triển đa thức bậc 2 để bổ sung các tương tác phi tuyến giữa đặc trưng. Vector 55 chiều sau khai triển được đưa vào một MLP nhỏ 55-16-1 để dự đoán lực. Thiết kế này giúp mô hình có số tham số thấp, huấn luyện nhanh và đóng vai trò baseline dễ kiểm soát cho bài toán ước lượng lực từ biến dạng xúc giác.

Khi trình bày hình kiến trúc, nên vẽ theo 5 khối:

```text
Marker tracking output
        |
        v
9 handcrafted deformation features
        |
        v
Standardization using train mean/std
        |
        v
Polynomial expansion degree 2
        |
        v
Small regression head
        |
        v
Predicted force
```

