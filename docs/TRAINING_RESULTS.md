# Kết Quả Thực Nghiệm Ước Lượng Lực

## 1. Thiết lập thực nghiệm

Thực nghiệm được thực hiện trên tập dữ liệu gồm 56 trial đã được tiền xử lý và lưu dưới dạng cache. Dữ liệu được chia theo trial, thay vì chia ngẫu nhiên theo từng frame, để đánh giá khả năng tổng quát hóa của mô hình trên các lần đo chưa xuất hiện trong quá trình huấn luyện.

| Thành phần | Giá trị |
|---|---:|
| Tổng số trial cache | 56 |
| Số trial train / validation / test | 39 / 8 / 9 |
| Số frame train | 11 823 |
| Số frame validation | 3 014 |
| Số frame test | 2 660 |
| Thiết bị huấn luyện | CUDA |
| Chiến lược đánh giá | Chọn checkpoint có MAE validation tốt nhất, sau đó đánh giá trên test split |

Ba hướng mô hình được so sánh:

| Ký hiệu | Mô hình | Đầu vào chính | Ý tưởng |
|---|---|---|---|
| `force_poly` | Polynomial Regression + MLP nhỏ | 9 đặc trưng thủ công từ trường dịch chuyển marker | Mô hình nhẹ, có khả năng diễn giải |
| `force_model` | PointNet-style ForceNet | Tập marker sau tracking, tối đa 150 marker | Học trực tiếp từ phân bố điểm và vector dịch chuyển |
| `force_cnn` | ResNet18 pretrained | Ảnh reference và ảnh biến dạng | Học end-to-end từ ảnh |

Các chỉ số sử dụng gồm MAE, RMSE và hệ số xác định R². MAE và RMSE càng nhỏ càng tốt; R² càng gần 1 càng tốt.

## 2. Kết quả tổng quan

| Mô hình | Số tham số | Thời gian train | Best val MAE | Epoch tốt nhất | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `force_poly` | 913 | 31.8 s | 0.0596 | 69 | 0.0680 | 0.1014 | 0.992 |
| `force_model` | 25 537 | 28.6 s | 0.0517 | 34 | 0.0564 | 0.0798 | 0.995 |
| `force_cnn` | 11 239 169 | 1 529.7 s | 0.0310 | 57 | 0.0307 | 0.0405 | 0.999 |

Kết quả cho thấy cả ba mô hình đều học được quan hệ giữa biến dạng bề mặt và lực tác dụng, với R² trên tập test đều lớn hơn 0.99. Trong đó, mô hình CNN đạt kết quả tốt nhất với MAE = 0.0307 và RMSE = 0.0405. So với mô hình `force_model`, CNN giảm MAE khoảng 45.6% và giảm RMSE khoảng 49.2%. So với `force_poly`, CNN giảm MAE khoảng 54.9% và giảm RMSE khoảng 60.1%.

Tuy nhiên, sự cải thiện này đi kèm chi phí tính toán lớn hơn đáng kể. `force_cnn` có hơn 11.2 triệu tham số và cần khoảng 25.5 phút để huấn luyện, trong khi `force_poly` chỉ có 913 tham số và hoàn thành trong 31.8 giây. Vì vậy, lựa chọn mô hình phụ thuộc vào yêu cầu của hệ thống: nếu ưu tiên độ chính xác cao nhất, CNN là lựa chọn tốt nhất; nếu ưu tiên mô hình nhẹ, dễ triển khai và dễ phân tích, `force_poly` vẫn là phương án có giá trị.

## 3. Kết quả từng mô hình

### 3.1. Polynomial Regression với đặc trưng thủ công

Mô hình `force_poly` sử dụng 9 đặc trưng v1 được trích xuất từ dịch chuyển marker, sau đó mở rộng đa thức bậc 2. Với 9 đặc trưng đầu vào, số hạng đa thức là 55. Phần head sử dụng MLP nhỏ, tổng cộng 913 tham số.

| Thuộc tính | Giá trị |
|---|---:|
| Feature set | v1, 9 đặc trưng |
| Bậc đa thức | 2 |
| Số hạng đa thức | 55 |
| Head | `mlp_small` |
| Số tham số | 913 |
| Early stopping | Sau 40 epoch không cải thiện |
| Best validation MAE | 0.0596 tại epoch 69 |
| Test MAE / RMSE / R² | 0.0680 / 0.1014 / 0.992 |

Trong quá trình huấn luyện, mô hình hội tụ rất nhanh. Ngay từ epoch đầu tiên, validation R² đã đạt 0.963. Sau khoảng 10 epoch, R² validation dao động quanh 0.988-0.990 và tiếp tục cải thiện chậm ở các epoch sau. Checkpoint tốt nhất xuất hiện tại epoch 69 với validation MAE = 0.0596.

Kết quả theo từng trial trên tập test:

| Trial | Số frame | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| s1/trial_008 | 159 | 0.0512 | 0.0840 | 0.996 |
| 2/trial_009 | 420 | 0.0410 | 0.0547 | 0.998 |
| t1/trial_009 | 142 | 0.0755 | 0.1077 | 0.993 |
| 1/trial_003 | 361 | 0.0589 | 0.0707 | 0.992 |
| s1/trial_001 | 334 | 0.1020 | 0.1556 | 0.981 |
| c1/trial_008 | 140 | 0.0720 | 0.1031 | 0.992 |
| 1/trial_002 | 316 | 0.1335 | 0.1584 | 0.971 |
| 2/trial_008 | 462 | 0.0353 | 0.0543 | 0.998 |
| 2/trial_003 | 326 | 0.0639 | 0.0905 | 0.995 |

Mô hình đạt kết quả tốt trên phần lớn trial, nhưng sai số tăng rõ ở `1/trial_002` và `s1/trial_001`. Điều này cho thấy các đặc trưng tổng hợp vẫn có giới hạn khi điều kiện tiếp xúc hoặc phân bố biến dạng khác biệt so với phần lớn dữ liệu huấn luyện.

Một ưu điểm quan trọng của `force_poly` là khả năng diễn giải. Các hệ số lớn nhất trong mô hình liên quan đến độ lớn dịch chuyển cực đại, độ lệch chuẩn dịch chuyển, dịch chuyển trung bình theo trục x và thành phần dịch chuyển hướng kính. Điều này phù hợp với trực giác vật lý: lực tác dụng lớn thường làm tăng biên độ biến dạng, đồng thời làm thay đổi phân bố biến dạng theo không gian.

### 3.2. ForceNet dạng PointNet

Mô hình `force_model` nhận trực tiếp tập marker đã tracking. Mỗi marker mang thông tin vị trí chuẩn hóa và vector dịch chuyển. Kiến trúc dùng shared MLP trên từng marker, sau đó tổng hợp đặc trưng bằng pooling để dự đoán lực.

| Thuộc tính | Giá trị |
|---|---:|
| Số marker tối đa | 150 |
| Shared MLP | 64, 128 |
| Hidden head | 64 |
| Dropout | 0.1 |
| Số tham số | 25 537 |
| Early stopping | Sau 10 epoch không cải thiện |
| Best validation MAE | 0.0517 tại epoch 34 |
| Test MAE / RMSE / R² | 0.0564 / 0.0798 / 0.995 |

So với `force_poly`, mô hình PointNet-style cải thiện rõ trên tập test. MAE giảm từ 0.0680 xuống 0.0564, tương đương giảm khoảng 17.1%; RMSE giảm từ 0.1014 xuống 0.0798, tương đương giảm khoảng 21.3%. Điều này cho thấy việc học trực tiếp từ phân bố marker giúp mô hình giữ lại nhiều thông tin không gian hơn so với các đặc trưng tổng hợp thủ công.

Kết quả theo từng trial trên tập test:

| Trial | Số frame | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| s1/trial_008 | 159 | 0.0678 | 0.0931 | 0.995 |
| 2/trial_009 | 420 | 0.0463 | 0.0691 | 0.996 |
| t1/trial_009 | 142 | 0.0764 | 0.0979 | 0.994 |
| 1/trial_003 | 361 | 0.0491 | 0.0685 | 0.992 |
| s1/trial_001 | 334 | 0.0691 | 0.0953 | 0.993 |
| c1/trial_008 | 140 | 0.0532 | 0.0716 | 0.996 |
| 1/trial_002 | 316 | 0.0694 | 0.0910 | 0.990 |
| 2/trial_008 | 462 | 0.0442 | 0.0641 | 0.997 |
| 2/trial_003 | 326 | 0.0564 | 0.0827 | 0.996 |

Đặc biệt, trên các trial khó đối với `force_poly` như `1/trial_002` và `s1/trial_001`, `force_model` giảm sai số đáng kể. Với `1/trial_002`, MAE giảm từ 0.1335 xuống 0.0694. Với `s1/trial_001`, MAE giảm từ 0.1020 xuống 0.0691. Đây là bằng chứng cho thấy biểu diễn theo tập marker giúp mô hình xử lý tốt hơn các trường hợp phân bố biến dạng phức tạp.

### 3.3. CNN end-to-end từ ảnh

Mô hình `force_cnn` dùng ResNet18 pretrained làm backbone. Đầu vào là ảnh đã resize về 240 x 320, cho phép mô hình học trực tiếp từ thông tin thị giác thay vì phụ thuộc hoàn toàn vào kết quả tracking marker.

| Thuộc tính | Giá trị |
|---|---:|
| Backbone | ResNet18 |
| Pretrained | Có |
| Kích thước ảnh | 240 x 320 |
| Hidden head | 128 |
| Dropout | 0.2 |
| Số tham số | 11 239 169 |
| Số epoch huấn luyện | 60 |
| Best validation MAE | 0.0310 tại epoch 57 |
| Test MAE / RMSE / R² | 0.0307 / 0.0405 / 0.999 |

Đường huấn luyện cho thấy CNN cải thiện đều theo thời gian. Validation MAE giảm từ 0.1778 ở epoch 1 xuống 0.0310 tại epoch 57. Trên tập test, mô hình đạt MAE = 0.0307 và R² = 0.999, là kết quả tốt nhất trong ba mô hình.

Kết quả theo từng trial trên tập test:

| Trial | Số frame | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| s1/trial_008 | 159 | 0.0374 | 0.0526 | 0.998 |
| 2/trial_009 | 420 | 0.0249 | 0.0332 | 0.999 |
| t1/trial_009 | 142 | 0.0369 | 0.0532 | 0.998 |
| 1/trial_003 | 361 | 0.0245 | 0.0333 | 0.998 |
| s1/trial_001 | 334 | 0.0333 | 0.0391 | 0.999 |
| c1/trial_008 | 140 | 0.0431 | 0.0595 | 0.997 |
| 1/trial_002 | 316 | 0.0354 | 0.0428 | 0.998 |
| 2/trial_008 | 462 | 0.0254 | 0.0336 | 0.999 |
| 2/trial_003 | 326 | 0.0339 | 0.0412 | 0.999 |

Sai số của CNN không chỉ thấp hơn về trung bình mà còn ổn định hơn giữa các trial. Trial có MAE cao nhất của CNN là `c1/trial_008` với MAE = 0.0431, vẫn thấp hơn đáng kể so với MAE trung bình của hai mô hình còn lại. Điều này cho thấy mô hình ảnh khai thác được các tín hiệu biến dạng mà pipeline marker có thể làm mất trong quá trình phát hiện, tracking và rút gọn đặc trưng.

## 4. So sánh theo từng trial

Bảng dưới đây tổng hợp MAE của ba mô hình trên cùng 9 trial test:

| Trial | `force_poly` | `force_model` | `force_cnn` | Mô hình tốt nhất |
|---|---:|---:|---:|---|
| s1/trial_008 | 0.0512 | 0.0678 | 0.0374 | `force_cnn` |
| 2/trial_009 | 0.0410 | 0.0463 | 0.0249 | `force_cnn` |
| t1/trial_009 | 0.0755 | 0.0764 | 0.0369 | `force_cnn` |
| 1/trial_003 | 0.0589 | 0.0491 | 0.0245 | `force_cnn` |
| s1/trial_001 | 0.1020 | 0.0691 | 0.0333 | `force_cnn` |
| c1/trial_008 | 0.0720 | 0.0532 | 0.0431 | `force_cnn` |
| 1/trial_002 | 0.1335 | 0.0694 | 0.0354 | `force_cnn` |
| 2/trial_008 | 0.0353 | 0.0442 | 0.0254 | `force_cnn` |
| 2/trial_003 | 0.0639 | 0.0564 | 0.0339 | `force_cnn` |

CNN là mô hình tốt nhất trên cả 9 trial test. `force_model` đứng thứ hai trên 6/9 trial, đặc biệt cải thiện rõ ở các trial mà `force_poly` có sai số lớn. `force_poly` vẫn cạnh tranh tốt ở một số trial như `2/trial_008` và `2/trial_009`, nhưng kém ổn định hơn khi gặp các trial có phân bố biến dạng khác biệt.

## 5. Phân tích và nhận xét

Kết quả thực nghiệm cho thấy mức độ thông tin đầu vào ảnh hưởng trực tiếp đến chất lượng dự đoán lực. `force_poly` sử dụng đặc trưng tổng hợp nên rất nhẹ và huấn luyện nhanh, nhưng quá trình tổng hợp làm mất một phần thông tin không gian. Vì vậy, mô hình này phù hợp với hệ thống yêu cầu tốc độ cao, tài nguyên thấp hoặc cần khả năng giải thích.

`force_model` giữ lại cấu trúc theo từng marker, nhờ đó biểu diễn tốt hơn sự phân bố biến dạng trên bề mặt cảm biến. Mô hình này cải thiện đáng kể so với `force_poly` nhưng vẫn phụ thuộc vào chất lượng phát hiện và tracking marker. Nếu marker bị mất, nhiễu hoặc phân bố không đều, sai số có thể tăng.

`force_cnn` đạt kết quả tốt nhất vì học trực tiếp từ ảnh, không cần rút gọn biến dạng thành vài đặc trưng hoặc phụ thuộc hoàn toàn vào tọa độ marker đã tracking. Backbone ResNet18 pretrained có khả năng trích xuất đặc trưng thị giác mạnh, giúp mô hình nhận biết các thay đổi nhỏ trong hình ảnh biến dạng. Đổi lại, mô hình có số tham số lớn hơn nhiều và thời gian huấn luyện cao hơn khoảng 48 lần so với `force_poly`.

Về mặt triển khai, có thể xem ba mô hình như ba mức đánh đổi:

| Mục tiêu triển khai | Mô hình phù hợp |
|---|---|
| Nhẹ, nhanh, dễ giải thích | `force_poly` |
| Cân bằng giữa độ chính xác và chi phí tính toán | `force_model` |
| Ưu tiên độ chính xác cao nhất | `force_cnn` |

## 6. Kết luận

Từ các kết quả trên tập test, mô hình CNN end-to-end là phương án cho độ chính xác cao nhất trong bài toán ước lượng lực từ biến dạng ảnh. Mô hình đạt MAE = 0.0307, RMSE = 0.0405 và R² = 0.999 trên 2 660 frame test, đồng thời cho sai số thấp và ổn định trên toàn bộ 9 trial test.

Mô hình PointNet-style `force_model` là phương án trung gian hiệu quả, đạt MAE = 0.0564 và R² = 0.995, tốt hơn mô hình đa thức nhưng nhẹ hơn rất nhiều so với CNN. Trong khi đó, `force_poly` tuy có sai số cao hơn, nhưng chỉ sử dụng 913 tham số, huấn luyện rất nhanh và có khả năng diễn giải tốt thông qua các đặc trưng biến dạng.

Do đó, nếu mục tiêu của đồ án là chứng minh khả năng ước lượng lực chính xác nhất từ dữ liệu thị giác, `force_cnn` nên được chọn làm mô hình chính. Nếu cần một mô hình tham chiếu đơn giản, dễ phân tích và phù hợp với thiết bị hạn chế tài nguyên, `force_poly` là baseline hợp lý; còn `force_model` đóng vai trò cầu nối giữa hướng đặc trưng thủ công và hướng học sâu end-to-end.

## 7. Hình ảnh và file kết quả

Các hình và bảng kết quả đã được lưu trong thư mục `outputs`:

| Đường dẫn | Nội dung |
|---|---|
| `outputs/force_poly/eval/test_scatter.png` | Scatter plot dự đoán - nhãn thật của `force_poly` |
| `outputs/force_poly/eval/test_timeseries_trial0.png` | Chuỗi thời gian dự đoán của `force_poly` trên trial đầu tiên |
| `outputs/force_poly/eval/test_breakdown.csv` | Bảng kết quả từng trial của `force_poly` |
| `outputs/force_poly/eval/coefficients.csv` | Hệ số các đặc trưng đa thức của `force_poly` |
| `outputs/force_model/eval/test_scatter.png` | Scatter plot dự đoán - nhãn thật của `force_model` |
| `outputs/force_model/eval/test_timeseries_trial0.png` | Chuỗi thời gian dự đoán của `force_model` trên trial đầu tiên |
| `outputs/force_model/eval/test_breakdown.csv` | Bảng kết quả từng trial của `force_model` |
| `outputs/force_cnn/eval/test_scatter.png` | Scatter plot dự đoán - nhãn thật của `force_cnn` |
| `outputs/force_cnn/eval/test_timeseries_trial0.png` | Chuỗi thời gian dự đoán của `force_cnn` trên trial đầu tiên |
| `outputs/force_cnn/eval/test_breakdown.csv` | Bảng kết quả từng trial của `force_cnn` |

Có thể chèn các hình scatter plot và time-series vào báo cáo để minh họa trực quan. Scatter plot dùng để đánh giá độ khớp tổng thể giữa dự đoán và giá trị thật; time-series plot dùng để quan sát khả năng bám theo biến thiên lực theo thời gian.

Ghi chú kỹ thuật: trong log đánh giá có xuất hiện dòng `--split: command not found` và `--out: command not found` do lệnh shell bị xuống dòng mà không có ký tự nối dòng. Các chỉ số đánh giá đã được in ra trước đó vẫn hợp lệ; để chạy lại đúng cú pháp, cần viết lệnh trên một dòng hoặc thêm dấu `\` ở cuối dòng.
