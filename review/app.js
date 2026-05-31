const modules = [
  {
    id: "preprocessing",
    icon: "01",
    title: "Tiền xử lý",
    file: "preprocessing.py",
    short: "Khử nền, normalize, CLAHE",
    thesis:
      "Mục tiêu là biến ảnh marker có chiếu sáng không đều thành ảnh tương phản ổn định để detector dùng chung một dải tham số.",
    flow: ["Box blur 101x101", "Subtract + normalize", "CLAHE clip 2.5, grid 8x8"],
    keys: [
      ["Box blur", "Ước lượng nền chậm biến thiên; kernel lớn hơn marker nhiều lần nên marker bị loại khỏi ảnh nền."],
      ["cv2.subtract", "Dùng saturating arithmetic trên uint8, vùng âm bị clamp về 0 thay vì wrap-around."],
      ["Normalize", "Kéo dải sau trừ nền về [0,255], giúp threshold và detector ổn định hơn giữa frame/session."],
      ["CLAHE", "Tăng tương phản cục bộ nhưng giới hạn khuếch đại nhiễu bằng clipLimit."]
    ],
    params: [
      ["blur_kernel", "[101,101]", "Tăng: nền mượt hơn nhưng chậm hơn. Giảm: còn vignetting hoặc marker lọt vào nền."],
      ["clahe_clip_limit", "2.5", "Tăng: marker mờ nổi hơn, dễ false positive. Giảm: ít khuếch đại tương phản."],
      ["clahe_grid", "[8,8]", "Tăng số ô: thích nghi cục bộ hơn nhưng dễ over-enhance noise."]
    ],
    traps: [
      "Đừng nói normalize giữ ánh sáng tuyệt đối. Bước này cố ý bỏ thông tin ánh sáng tuyệt đối.",
      "Đừng chọn Gaussian chỉ vì quen thuộc; với kernel lớn, box blur là lựa chọn hiệu năng hợp lý."
    ]
  },
  {
    id: "detection",
    icon: "02",
    title: "Phát hiện marker",
    file: "detection.py",
    short: "SimpleBlobDetector đa ngưỡng",
    thesis:
      "Marker là blob sáng gần tròn trên nền tối, nên detector cổ điển với multi-threshold và filter hình học phù hợp hơn mô hình nặng.",
    flow: ["Threshold sweep", "Connected components", "Group centers", "Geometry filters"],
    keys: [
      ["Multi-threshold", "Blob thật xuất hiện ổn định qua nhiều threshold; noise thường chỉ xuất hiện ở vài mức."],
      ["blobColor=255", "Chỉ giữ vùng sáng sau tiền xử lý, loại blob tối và artifact nền."],
      ["Area filter", "Giữ marker trong khoảng kích thước pixel dự kiến, loại nhiễu nhỏ và vùng phản xạ lớn."],
      ["Shape filters", "Circularity, inertia, convexity giữ blob gần tròn nhưng vẫn chấp nhận méo nhẹ khi gel biến dạng."]
    ],
    params: [
      ["min_threshold / max_threshold / step", "50 / 220 / 10", "17 mức threshold, cân bằng độ ổn định và tốc độ."],
      ["min_area / max_area", "30 / 500 px2", "Phụ thuộc optical setup, cần chỉnh khi đổi camera/khoảng cách/marker."],
      ["min_circularity", "0.5", "Ngưỡng mềm để không bỏ marker bị nén thành ellipse."],
      ["min_inertia / min_convexity", "0.3 / 0.7", "Loại vệt dài, blob lõm, blob dính nhau hoặc artifact phức tạp."]
    ],
    traps: [
      "Nếu dùng threshold + findContours phải chọn một ngưỡng duy nhất, yếu hơn khi marker sáng không đều.",
      "YOLO/keypoint network thêm dataset, dependency và latency cho bài toán hình học rất hẹp."
    ]
  },
  {
    id: "tracking",
    icon: "03",
    title: "Tracking",
    file: "tracking.py",
    short: "Pyramid LK + FB check",
    thesis:
      "Sau khi có marker reference, Pyramid Lucas-Kanade theo dõi vị trí với sub-pixel accuracy, rồi forward-backward check tạo valid mask.",
    flow: ["LK forward", "LK backward", "FB error", "Valid mask", "Deadzone tùy chọn"],
    keys: [
      ["Brightness constancy", "LK giả định cùng điểm ảnh giữ độ sáng khi di chuyển giữa hai frame."],
      ["Pyramid", "Bắt motion lớn ở level coarse rồi refine ở ảnh gốc để có độ chính xác sub-pixel."],
      ["win_size 21x21", "Đủ chứa marker 10-15px và vùng nền quanh nó, nhưng chưa quá lớn để lẫn marker khác."],
      ["Forward-backward", "Track ref->def rồi def->ref; nếu không quay gần điểm gốc thì marker bị invalid."]
    ],
    params: [
      ["win_size", "[21,21]", "Tăng: robust hơn với nhiễu nhưng dễ vi phạm local motion. Giảm: nhạy hơn nhưng kém ổn định."],
      ["max_level", "3", "Theo lý thuyết bắt motion tới khoảng 84px/frame với win 21."],
      ["fb_threshold", "2.0 px", "Dưới 1px quá chặt; trên 3px dễ giữ track trôi."],
      ["min_displacement", "1 px", "Deadzone giảm rung/nhiễu; tắt khi cần giữ tín hiệu slip chậm."]
    ],
    traps: [
      "FB check không bắt được mọi lỗi; tracking sai nhưng nhất quán hai chiều vẫn có thể qua.",
      "Deadzone tốt cho visualization/force ổn định, nhưng có thể làm mất creep slip nhỏ."
    ]
  },
  {
    id: "visualization",
    icon: "04",
    title: "Visualization",
    file: "visualization.py",
    short: "Mũi tên displacement",
    thesis:
      "Visualization chỉ giúp quan sát hướng và độ lớn tương đối của displacement; nó không thay đổi dữ liệu thật.",
    flow: ["disp = def - ref", "magnitude", "scale arrow", "color by magnitude"],
    keys: [
      ["arrow_scale=3", "Phóng đại mũi tên để displacement vài pixel nhìn rõ trên màn hình."],
      ["Color mapping", "Magnitude nhỏ gần xanh, magnitude lớn chuyển đỏ theo multiplier."],
      ["motion threshold", "Ngưỡng 0.3px chỉ lọc residual khi vẽ, tách biệt với deadzone tracking."],
      ["Data integrity", "Ảnh vẽ là output trực quan; displacement số dùng cho model không bị scale."]
    ],
    params: [
      ["arrow_scale", "3.0", "Tăng để dễ nhìn pattern, nhưng phải ghi chú khi báo cáo."],
      ["motion_magnitude_threshold", "0.3 px", "Lọc marker gần như đứng yên khi vẽ."],
      ["color_intensity_multiplier", "20", "Điều chỉnh tốc độ chuyển màu theo magnitude."]
    ],
    traps: [
      "Không được trình bày mũi tên phóng đại như displacement thật.",
      "Visualization nên dùng valid mask, không vẽ marker tracking lỗi."
    ]
  },
  {
    id: "architecture",
    icon: "05",
    title: "Kiến trúc",
    file: "src/core",
    short: "Ảnh -> field -> force/slip",
    thesis:
      "src/core không phải mô hình học sâu; nó là pipeline xử lý ảnh cổ điển tạo displacement field và valid mask cho force, slip và visualization.",
    flow: ["preprocess", "detect reference", "track frame", "validate", "consume downstream"],
    keys: [
      ["Reference-first", "Detect marker trên reference rồi track để giữ identity và tránh detector nhảy thứ tự mỗi frame."],
      ["Quality control", "valid mask cho phép module sau bỏ marker lỗi thay vì dùng dữ liệu sai."],
      ["Realtime", "Cache CLAHE, cache detector, LK nhanh hơn matching toàn cục."],
      ["Setup-dependent", "Threshold, area, blur kernel, LK window phụ thuộc camera, ánh sáng, kích thước marker pixel."]
    ],
    params: [
      ["force_poly", "9 features", "Dùng field marker đã lọc để rút gọn đặc trưng."],
      ["force_model", "mask-aware", "Có thể dùng valid mask khi pooling hoặc chọn marker."],
      ["slip detection", "velocity/direction", "Cần tín hiệu displacement theo thời gian, đôi khi tắt deadzone."]
    ],
    traps: [
      "Không gọi src/core là model học máy.",
      "Calibration bị bỏ qua trong tài liệu này nhưng pipeline vẫn làm việc trong hệ tọa độ pixel."
    ]
  },
  {
    id: "force_poly_overview",
    icon: "06",
    title: "Force Poly",
    file: "src/force_poly",
    short: "9 features -> force N",
    thesis:
      "force_poly là baseline học máy nhẹ: không học từ ảnh raw, mà chuyển displacement field thành 9 đặc trưng vật lý rồi hồi quy lực.",
    flow: ["cache npz", "compute 9 features", "standardize train", "poly degree 2", "regression head"],
    keys: [
      ["Vai trò", "Dùng output tracking của src/core gồm ref_pts, disp, valid và force label để học quan hệ biến dạng-lực."],
      ["Physics prior", "Hertz contact cho quan hệ lực-biến dạng phi tuyến dạng lũy thừa, nên polynomial là xấp xỉ tự nhiên trong vùng làm việc."],
      ["Baseline nhẹ", "Config hiện tại chỉ có 9 input feature, 55 monomial bậc 2 và 913 tham số với mlp_small."],
      ["Trade-off", "Nhanh, dễ debug, ít dữ liệu hơn CNN; đổi lại mất thông tin không gian chi tiết của toàn bộ marker field."]
    ],
    params: [
      ["feature_set", "v1", "Bộ 9 feature thủ công hiện tại, mask-aware và scale-normalized."],
      ["degree", "2", "9 feature thành C(11,2)=55 monomial terms."],
      ["head", "mlp_small", "Linear 55->16, ReLU, Dropout, Linear 16->1; tổng 913 tham số."],
      ["input source", "cache .npz", "Tái dùng cache từ force_model prepare: ref_pts, disp, valid, force, image_w."]
    ],
    traps: [
      "Không nói force_poly học trực tiếp từ ảnh; nó phụ thuộc hoàn toàn vào detection/tracking của src/core.",
      "Nếu dùng mlp_small, gọi chính xác hơn là polynomial features + small MLP regressor, không phải polynomial regression thuần."
    ]
  },
  {
    id: "force_poly_features",
    icon: "07",
    title: "Feature Engineering",
    file: "features.py",
    short: "9 feature vật lý",
    thesis:
      "9 feature tóm tắt biên độ, hướng, độ phân tán, cấu trúc radial/tangential và chất lượng tracking của displacement field.",
    flow: ["scale by image_w", "mask valid markers", "magnitude stats", "radial tangential", "valid ratio"],
    keys: [
      ["Scale-normalized", "dx, dy và tọa độ marker được chia cho image_w để giảm phụ thuộc độ phân giải và giữ cùng thang đo cho hai trục."],
      ["Mask-aware", "Chỉ marker valid tham gia thống kê; frame không có marker hợp lệ trả vector 0 để bias/head học mức nền."],
      ["Radial/tangential", "Bổ sung hướng biến dạng quanh tâm marker, giúp phân biệt press đối xứng với shear hoặc xoắn."],
      ["valid_ratio", "Không trực tiếp là lực, nhưng báo cho model chất lượng quan sát khi tracking mất nhiều marker."]
    ],
    params: [
      ["disp_mag_mean/max/std", "features 0..2", "Mô tả mức biến dạng trung bình, đỉnh biến dạng và độ không đồng đều."],
      ["dx_mean/dy_mean", "features 3..4", "Thành phần dịch chuyển trung bình theo hai trục, liên quan shear/hướng tải."],
      ["disp_mag_sum_norm", "feature 5", "Hiện tương đương disp_mag_mean vì sum/N_valid = mean; là redundancy cần nêu được."],
      ["radial/tangential/valid", "features 6..8", "Mô tả nén/giãn, xoắn/shear và độ tin cậy tracking."]
    ],
    traps: [
      "Feature 0 và feature 5 đang trùng công thức, gây multicollinearity; model vẫn học được nhờ regularization nhưng đây là điểm cải thiện rõ.",
      "Tâm radial hiện tính theo marker valid; nếu marker mất lệch một phía, tâm có thể bias. Có thể cải thiện bằng centroid cố định của tất cả ref_pts."
    ]
  },
  {
    id: "force_poly_model",
    icon: "08",
    title: "Poly Model",
    file: "model.py",
    short: "standardize + monomial",
    thesis:
      "Mô hình chuẩn hóa feature theo train set trước khi sinh monomial để giữ scale ổn định và tránh rò rỉ validation/test.",
    flow: ["fit scaler train", "z-score features", "monomial indices", "55 terms", "linear or MLP"],
    keys: [
      ["Standardize trước expand", "Nếu không chuẩn hóa, term bình phương/tích chéo của feature lớn sẽ áp đảo gradient và weight decay."],
      ["Số term", "Số monomial bậc <= d với D feature là C(D+d,d); D=9, d=2 cho 55 term."],
      ["Bias term", "Tuple rỗng tạo hằng số 1; vì vậy linear head dùng bias=False để tránh hai bias trùng nhau."],
      ["Head choice", "linear dễ giải thích hệ số; mlp_small biểu diễn tốt hơn nhưng mất interpretability trực tiếp."]
    ],
    params: [
      ["degree=1", "10 terms", "Linear regression thuần, dễ underfit nếu quan hệ phi tuyến."],
      ["degree=2", "55 terms", "Current config, có bình phương và interaction giữa feature."],
      ["degree=3", "220 terms", "Có thể hợp hơn vùng lực rộng nhưng dễ overfit, cần thêm dữ liệu/regularization."],
      ["hidden_head", "16", "Capacity nhỏ, giữ model rất nhẹ."]
    ],
    traps: [
      "Scaler phải fit từ train set, không dùng toàn bộ data, nếu không sẽ leakage.",
      "mlp_small không xuất được coefficients.csv có ý nghĩa trực tiếp như linear head."
    ]
  },
  {
    id: "force_poly_training",
    icon: "09",
    title: "Train/Eval",
    file: "train.py + eval.py",
    short: "trial split + Huber",
    thesis:
      "Training được thiết kế để tránh temporal leakage, robust với outlier force/tracking và chọn checkpoint bằng validation MAE.",
    flow: ["trial-level split", "precompute RAM", "Huber loss", "AdamW schedule", "eval per trial"],
    keys: [
      ["Trial split", "Không chia random theo frame vì frame liền kề trong cùng trial rất giống nhau, dễ làm test lạc quan."],
      ["Huber loss", "Gần 0 giống MSE để học mượt, lỗi lớn giống MAE để giảm ảnh hưởng spike/outlier."],
      ["AdamW + weight decay", "Regularize các monomial tương quan và shrink hệ số bậc cao để giảm overfit."],
      ["Inference no deadzone", "force estimation tắt apply_deadzone vì lực nhỏ có thể chỉ tạo displacement 0.5-1px."]
    ],
    params: [
      ["huber_delta", "1.0 N", "Train nghiêm với sai số dưới 1N, robust hơn với spike lớn."],
      ["lr", "5e-3 -> 5e-5", "CosineAnnealingLR giảm mượt learning rate tới cuối training."],
      ["weight_decay", "1e-2", "Regularization quan trọng vì polynomial terms tương quan mạnh."],
      ["early_stop_patience", "40 epoch", "Lưu best.pt theo validation MAE, không dùng epoch cuối nếu đã overfit."]
    ],
    traps: [
      "Trial-level split giảm temporal leakage nhưng vẫn cần kiểm tra distribution force giữa train/test.",
      "Output head tuyến tính không đảm bảo lực không âm; inference có thể cần clip max(0,pred) hoặc Softplus."
    ]
  },
  {
    id: "force_model_prepare",
    icon: "10",
    title: "FM Prepare",
    file: "prepare.py",
    short: "raw trial -> npz cache",
    thesis:
      "force_model bắt đầu bằng bước offline prepare: đồng bộ ảnh với force log, detect/track marker không deadzone, trừ zero offset và lưu cache marker-level.",
    flow: ["session reference", "nearest force sync", "LK no deadzone", "zero offset", "npz cache"],
    keys: [
      ["Cache .npz", "Tách preprocessing tốn I/O khỏi training, giúp train nhanh và tái lập được cùng dữ liệu marker-level."],
      ["Nearest-neighbor sync", "Camera và force gauge không trùng timestamp; chọn mẫu force gần nhất trong tolerance 0.1s, quá xa thì bỏ frame."],
      ["Reference strategy", "Ưu tiên session reference lúc không tiếp xúc; fallback frame đầu trial có rủi ro đã có lực nhỏ."],
      ["Zero offset", "Trừ baseline từ session.yaml để force target trở về 0N khi không tiếp xúc."]
    ],
    params: [
      ["force_sync_tolerance_s", "0.1s", "Quá nhỏ bỏ nhiều frame; quá lớn gán nhãn force sai thời điểm."],
      ["apply_deadzone", "False", "Giữ displacement nhỏ cho vùng lực thấp, tránh zero-out dữ liệu training."],
      ["cache fields", "ref_pts, disp, valid, force", "Lưu thêm force_raw, ts_mono, image_w/h, n_markers, trial/session id."],
      ["force_zero_offset_n", "session.yaml", "Bù drift force gauge do nhiệt độ/trọng lực đầu đo."]
    ],
    traps: [
      "Nếu force thay đổi nhanh, tolerance 0.1s có thể tạo label noise đáng kể; hardware sync là hướng tốt hơn.",
      "Prepare phụ thuộc chất lượng src/core tracking; cache sai sẽ làm model học từ displacement sai."
    ]
  },
  {
    id: "force_model_dataset",
    icon: "11",
    title: "FM Dataset",
    file: "dataset.py",
    short: "N_max, mask, augment",
    thesis:
      "TrialDataset biến mỗi frame thành tập marker `(N_max,4)` kèm mask, cho phép batch hóa trial có số marker khác nhau và bỏ qua marker invalid/padding.",
    flow: ["per marker 4D", "pad to N_max", "valid mask", "train augment", "trial split"],
    keys: [
      ["Per-marker input", "Mỗi marker là [x_ref/W, y_ref/H, dx/W, dy/H], giữ cả vị trí và displacement."],
      ["N_max padding", "PyTorch batch cần cùng shape; trial ít marker được pad zeros và mask=False để pooling bỏ qua."],
      ["Trial-level split", "Giống force_poly, split theo trial để tránh frame lân cận của cùng trial rơi vào cả train và test."],
      ["Augmentation nhất quán", "Flip/rotation phải biến đổi cả vị trí marker và vector displacement, chỉ dùng cho train set."]
    ],
    params: [
      ["N_max", "max n_markers", "Lưu trong checkpoint; inference phải padding cùng N_max đã train."],
      ["flip_h/v_prob", "0.5 / 0.5", "Lật vị trí và đổi dấu dx/dy tương ứng; hợp lý nếu setup đối xứng."],
      ["rotation_max_deg", "+/-5", "Mô phỏng sai số mount nhỏ; quá lớn dễ tạo pattern không thực tế."],
      ["noise_std_px", "0.5px", "Mô phỏng sai số LK nhỏ, tránh overfit displacement quá sạch."]
    ],
    traps: [
      "Nếu deploy có nhiều marker hơn N_max train, pipeline padding-based không khớp; cần retrain, truncate hoặc dynamic architecture.",
      "Validation/test không được augment, nếu không metric không còn phản ánh dữ liệu thật."
    ]
  },
  {
    id: "force_model_model",
    icon: "12",
    title: "ForceNet",
    file: "model.py",
    short: "PointNet-style",
    thesis:
      "ForceNet dùng shared MLP trên từng marker rồi mask-aware max+mean pooling để tạo vector global 256 chiều dự đoán lực.",
    flow: ["Conv1d k1 shared", "BatchNorm ReLU", "masked max pool", "masked mean pool", "head 256-64-1"],
    keys: [
      ["PointNet-style", "Marker là tập điểm, thứ tự marker không có ý nghĩa vật lý; shared MLP + pooling tạo gần permutation-invariant representation."],
      ["Conv1d kernel=1", "Tương đương Linear áp độc lập cho từng marker với cùng trọng số, hiệu quả trên tensor (B,C,N)."],
      ["Max + mean pool", "Max bắt điểm biến dạng mạnh nhất; mean bắt xu hướng toàn cảm biến. Kết hợp giảm nhược điểm của từng loại."],
      ["Mask-aware", "Invalid marker và padding bị loại khỏi max/mean; frame không có marker valid reset max_pool về 0."]
    ],
    params: [
      ["input dim", "4", "x_ref_norm, y_ref_norm, dx_norm, dy_norm."],
      ["shared MLP", "4->64->128", "Conv1d k=1 + BatchNorm + ReLU cho từng marker."],
      ["pooled vector", "256", "Concat max_pool 128 và mean_pool 128."],
      ["params", "25,537", "Lớn hơn force_poly nhưng nhỏ hơn nhiều so với CNN end-to-end."]
    ],
    traps: [
      "BatchNorm tính trên B x N_max có thể bị padding zeros bias nếu số padding lớn; masked BN/InstanceNorm là hướng cải thiện.",
      "Pooling global làm mất một phần cấu trúc không gian cục bộ; attention hoặc graph network có thể giữ quan hệ lân cận tốt hơn."
    ]
  },
  {
    id: "force_model_training",
    icon: "13",
    title: "FM Train/Eval",
    file: "train.py + eval.py",
    short: "Huber, Adam, per-trial",
    thesis:
      "force_model train bằng Huber + Adam với regularization nhẹ, chọn best checkpoint theo validation MAE và đánh giá cả metric tổng thể lẫn từng trial.",
    flow: ["Huber delta 1", "Adam lr 1e-3", "wd 1e-4", "early stop 10", "trial breakdown"],
    keys: [
      ["Huber loss", "Robust với tracking lỗi, force sync lệch hoặc spike force gauge, nhưng vẫn mượt gần hội tụ."],
      ["Adam not AdamW", "Weight decay nhỏ 1e-4 nên khác biệt Adam/AdamW ít hơn; force_poly cần AdamW mạnh hơn vì polynomial collinearity."],
      ["Không LR scheduler", "Train ngắn 50 epoch, early stop thường đủ; khác force_poly dùng cosine schedule dài hơn."],
      ["Eval per trial", "Overall MAE/R2 có thể che trial khó; breakdown cho biết lỗi theo điều kiện tiếp xúc cụ thể."]
    ],
    params: [
      ["batch_size", "64", "Cân bằng gradient ổn định và memory, dữ liệu marker-level rất nhỏ."],
      ["num_epochs", "50", "Ngắn hơn force_poly; early stop theo val MAE."],
      ["early_stop_patience", "10", "Dừng nếu 10 epoch không cải thiện validation MAE."],
      ["test metrics", "MAE 0.0564, RMSE 0.0798, R2 0.995", "Kết quả hiện tại trong tài liệu, cho thấy marker-level representation học tốt."]
    ],
    traps: [
      "Output linear không đảm bảo lực không âm; có thể clip hoặc dùng Softplus nếu yêu cầu vật lý.",
      "R2 cao trên split hiện tại chưa đủ kết luận tuyệt đối; nên chạy nhiều seed hoặc cross-validation theo trial nếu cần chắc hơn."
    ]
  },
  {
    id: "force_cnn_dataset",
    icon: "14",
    title: "CNN Dataset",
    file: "src/force_cnn/dataset.py",
    short: "ảnh ref+frame 2 kênh",
    thesis:
      "force_cnn không tạo cache marker; dataset index trial trực tiếp từ data/sessions, lazy-load cặp ảnh reference-frame và đồng bộ force theo timestamp.",
    flow: ["index trial", "sync force cnn", "load ref frame", "resize normalize", "stack 2 channels"],
    keys: [
      ["End-to-end input", "Model nhận tensor (2,H,W): channel 0 là reference không tải, channel 1 là frame biến dạng."],
      ["Không cần LK", "Inference không cần detect marker, optical flow hay valid mask; CNN học trực tiếp từ pixel."],
      ["Lazy-load ảnh", "Ảnh raw lớn hơn displacement field nhiều, nên đọc từ disk theo batch và cache reference trong RAM."],
      ["Reference là baseline", "Cặp ảnh giúp model học biến dạng tương đối; nếu chỉ dùng frame, model phải tự suy ra trạng thái không tải."]
    ],
    params: [
      ["image_size", "240x320", "Giảm memory/GPU time nhưng có thể mất biến dạng marker rất nhỏ."],
      ["input channels", "2", "Grayscale ref + grayscale frame; không dùng RGB vì màu không mang nhiều thông tin lực."],
      ["force_sync_tolerance_s", "0.1s", "Giống force_model: quá nhỏ skip frame, quá lớn label noise."],
      ["zero_offset_n", "session.yaml", "Trừ baseline force gauge trước khi dùng làm label."]
    ],
    traps: [
      "Reference sai session hoặc chụp khi đang tiếp xúc sẽ tạo baseline sai và dự đoán lệch.",
      "Lazy disk I/O có thể thành bottleneck nếu dùng HDD; num_workers hoặc cache ảnh resize là hướng cải thiện."
    ]
  },
  {
    id: "force_cnn_augment",
    icon: "15",
    title: "CNN Augment",
    file: "dataset.py",
    short: "paired image transforms",
    thesis:
      "Augmentation của force_cnn phải áp dụng đồng bộ cho cả reference và frame để không tạo ra biến dạng giả giữa hai kênh.",
    flow: ["paired flip", "paired rotate", "paired brightness", "paired contrast", "train only"],
    keys: [
      ["Paired geometry", "Flip/rotate cùng transform cho ref và frame để quan hệ deformation không đổi."],
      ["Paired photometric", "Brightness/contrast dùng cùng hệ số cho cả hai ảnh để mô phỏng exposure chung, không tạo lực giả."],
      ["Train-only", "Validation/test không augment để metric phản ánh dữ liệu thật và deterministic."],
      ["Scalar force", "Flip H/V không đổi force label vì force gauge đo magnitude, không phải vector hướng ảnh."]
    ],
    params: [
      ["flip_h/v_prob", "0.5 / 0.5", "Tăng đa dạng nếu cảm biến có đối xứng tương đối."],
      ["rotation_max_deg", "+/-5", "Mô phỏng mount misalignment; >10 độ dễ thiếu thực tế."],
      ["brightness_jitter", "0.1", ">0.3 dễ clamp pixel và mất thông tin."],
      ["contrast_jitter", "0.1", "Tăng robust ánh sáng nhưng quá cao làm méo gradient marker."]
    ],
    traps: [
      "Nếu augment ref và frame khác nhau, model có thể học chênh lệch nhân tạo như tín hiệu lực.",
      "Nếu setup vật lý không đối xứng, flip/rotation mạnh có thể làm label không còn đúng."
    ]
  },
  {
    id: "force_cnn_model",
    icon: "16",
    title: "ForceCNN",
    file: "model.py",
    short: "ResNet18 end-to-end",
    thesis:
      "ForceCNN dùng backbone CNN, mặc định ResNet18 pretrained, để học đặc trưng thị giác trực tiếp từ cặp ảnh và hồi quy ra lực.",
    flow: ["input 2HW", "resnet conv1 inflate", "fc identity", "feature 512", "head 512-128-1"],
    keys: [
      ["ResNet18", "Đủ mạnh, có residual connection, nhẹ hơn ResNet50 và có pretrained ImageNet."],
      ["Conv1 inflate", "Trọng số conv1 RGB (64,3,7,7) được mean qua RGB rồi lặp thành 2 kênh ref/frame."],
      ["Pretrained transfer", "Filter edge/blob/texture cấp thấp từ ImageNet vẫn hữu ích cho marker grayscale."],
      ["SmallCNN fallback", "Backbone nhẹ khoảng 110K params để debug CPU/laptop, nhưng thường kém ResNet18."]
    ],
    params: [
      ["backbone", "resnet18", "Production/default, khoảng 11.24M tham số."],
      ["pretrained", "true", "Hội tụ nhanh hơn và giảm nhu cầu dữ liệu so với random init."],
      ["hidden_head", "128", "Head hồi quy 512->128->1, thêm ReLU và Dropout."],
      ["dropout", "0.2", "Regularization ở head để giảm overfit của backbone lớn."]
    ],
    traps: [
      "Conv1 inflate là approximation vì ref và frame có semantic khác RGB; có thể thử difference channel hoặc Siamese design.",
      "CNN dễ học shortcut từ ánh sáng/nền/timestamp nếu dataset không kiểm soát tốt; cần saliency/Grad-CAM hoặc ablation để kiểm tra."
    ]
  },
  {
    id: "force_cnn_training",
    icon: "17",
    title: "CNN Train/Eval",
    file: "train.py + eval.py + infer.py",
    short: "AdamW, cosine, infer ảnh",
    thesis:
      "force_cnn fine-tune mô hình lớn bằng AdamW, cosine scheduler, Huber loss và đánh giá theo trial; inference chỉ cần ref, frame và checkpoint.",
    flow: ["Huber cnn", "AdamW lr 3e-4", "cosine lr", "best val MAE", "infer image pair"],
    keys: [
      ["AdamW", "ResNet18 có ~11M params nên decoupled weight decay quan trọng hơn force_model nhỏ."],
      ["LR thấp", "3e-4 phù hợp fine-tune pretrained, tránh phá representation đã học."],
      ["CosineAnnealingLR", "LR giảm mượt: học nhanh ban đầu, fine-tune ổn định cuối training."],
      ["Infer.py", "Không cần marker pipeline; load ref+frame, resize/normalize, stack (1,2,H,W), forward."]
    ],
    params: [
      ["batch_size", "32", "Ảnh lớn và ResNet feature maps tốn GPU memory hơn ForceNet rất nhiều."],
      ["num_epochs", "60", "Early stop thường dừng trước nếu val MAE không cải thiện."],
      ["weight_decay", "1e-4", "Regularization nhẹ cho fine-tune CNN lớn."],
      ["test metrics", "MAE 0.0307, RMSE 0.0405, R2 0.999", "Theo tài liệu, tốt nhất trong ba mô hình hiện tại."]
    ],
    traps: [
      "Checkpoint không cần n_max nhưng deploy phải dùng config image_size/backbone khớp checkpoint.",
      "Khi load checkpoint để inference, đặt pretrained=False vì weights đã nằm trong best.pt."
    ]
  },
  {
    id: "slip_v1_overview",
    icon: "18",
    title: "Slip V1",
    file: "src/slip/v1.py",
    short: "MRVL slip detector",
    thesis:
      "Slip V1 không phải mô hình học máy; nó là detector heuristic dùng độ đồng hướng chuyển động marker để phát hiện trượt realtime.",
    flow: ["marker history", "past current", "valid motion", "weighted MRVL", "slip phase"],
    keys: [
      ["Giả thuyết vật lý", "Slip tạo nhiều vector marker cùng hướng tịnh tiến; press/release thường phân tán hoặc xuyên tâm."],
      ["MRVL", "Mean Resultant Vector Length đo độ đồng hướng của các góc chuyển động, gần 1 là cùng hướng, gần 0 là phân tán."],
      ["Realtime pipeline", "RealtimeSlipTracking giữ marker_history rồi so sánh current với frame cũ nhất trong buffer."],
      ["Không cần nhãn", "Detector không train, không có trọng số học được; chỉ có threshold, filter và state EMA."]
    ],
    params: [
      ["slip_threshold", "0.8", "Ngưỡng r_value đã smooth để kết luận slip."],
      ["history_buffer_length", "5 frame", "So sánh current với past cách tối đa 4 frame để làm motion rõ hơn."],
      ["output", "dict", "is_slip, r_value, raw_score, translation, direction, moving_count, phase."],
      ["phase", "slip/tracking/pressing/no_markers/insufficient_motion", "Cho biết lý do trạng thái hiện tại."]
    ],
    traps: [
      "Slip V1 phụ thuộc vào LK tracking và valid mask; tracking sai thì hướng chuyển động sai.",
      "history_buffer_length nằm trong realtime pipeline, không phải logic nội bộ của SlipDetector."
    ]
  },
  {
    id: "slip_v1_filters",
    icon: "19",
    title: "Slip Filters",
    file: "v1.py filtering",
    short: "valid, motion, rebound, radial",
    thesis:
      "Trước khi tính MRVL, Slip V1 lọc marker không đáng tin và loại các chuyển động giống nhiễu, hồi đàn hồi hoặc press/release xuyên tâm.",
    flow: ["valid filter", "press detect", "motion threshold", "rebound filter", "radial removal"],
    keys: [
      ["Valid filter", "Marker invalid từ LK bị bỏ để tránh hướng chuyển động giả."],
      ["Motion threshold", "Chỉ marker có magnitude > min_motion_thresh mới tham gia, vì vector nhỏ có góc nhiễu."],
      ["Rebound filter", "Nếu displacement đi ngược deformation so với reference, marker có thể đang hồi đàn hồi về gốc."],
      ["Radial removal", "Trừ thành phần radial_scale * centered_ref để giảm nhầm press/release với slip."]
    ],
    params: [
      ["min_motion_thresh", "2.0 px", "Lọc rung/nhiễu nhỏ; quá cao có thể bỏ slow slip."],
      ["min_moving_markers", "5", "Cần đủ marker chuyển động để evidence đáng tin."],
      ["rebound_dot_prod_threshold", "-0.1", "Dot âm rõ thì xem là rebound/release."],
      ["press_rate_threshold", "0.5 px/frame", "Mean deformation tăng nhanh được xem là pressing."]
    ],
    traps: [
      "Radial removal của V1 là xấp xỉ đơn giản, không phải decomposition đầy đủ như V2.",
      "Các ngưỡng pixel/frame phụ thuộc framerate, camera scale và mật độ marker."
    ]
  },
  {
    id: "slip_v1_scoring",
    icon: "20",
    title: "Slip Score",
    file: "v1.py scoring",
    short: "gains + EMA",
    thesis:
      "Slip V1 lấy weighted MRVL làm score gốc, boost bằng translation/participation/motion rồi làm mượt bằng EMA adaptive trước khi quyết định slip.",
    flow: ["angle weights", "raw R", "score gains", "pressing gate", "EMA smooth"],
    keys: [
      ["Weighted MRVL", "Mỗi vector đóng góp theo magnitude/sum_magnitude, nên marker chuyển động mạnh ảnh hưởng nhiều hơn."],
      ["Translation gain", "Slip thật thường có tịnh tiến rõ; MRVL cao nhưng translation yếu không nên quá tự tin."],
      ["Participation/motion gain", "Nhiều marker tham gia và motion trung bình đủ lớn làm score đáng tin hơn."],
      ["EMA adaptive", "Giảm flicker, nhưng tăng alpha khi evidence mạnh để detector phản ứng nhanh."]
    ],
    params: [
      ["alpha", "0.3", "Hệ số EMA cơ bản khi score tăng."],
      ["alpha_decay", "0.6", "Hệ số decay khi score giảm hoặc thiếu evidence."],
      ["pressing gate", "translation_gain < 0.35", "Nếu đang pressing và translation yếu thì suppress score."],
      ["moving_count_thresh", "2", "Dùng cho overlay hiển thị moving, không trực tiếp trong SlipDetector."]
    ],
    traps: [
      "MRVL chỉ đo đồng hướng; vì vậy cần thêm gain và gate để tránh false positive.",
      "Slip V1 một thang thời gian nên có thể bỏ sót slow slip; V2 multi-scale xử lý tốt hơn."
    ]
  }
];

const pipeline = [
  ["Raw grayscale", "Ảnh camera đầu vào"],
  ["Core or image", "Tracking marker hoặc cặp ảnh"],
  ["Force/slip inputs", "Feature, marker set, image pair, history"],
  ["Model/detector", "Poly, ForceNet, CNN, Slip V1"],
  ["Output", "Force N hoặc slip phase"]
];

const flashcards = [
  ["Tiền xử lý", "Vì sao dùng box blur 101x101 để ước lượng nền?", "Kernel lớn hơn marker nhiều lần nên marker bị low-pass ra khỏi nền; nền LED/vignetting biến thiên chậm vẫn được giữ. Box blur nhanh hơn Gaussian lớn và đủ chính xác cho mục tiêu ước lượng nền."],
  ["Tiền xử lý", "CLAHE khác gì histogram equalization toàn cục?", "CLAHE cân bằng histogram theo ô cục bộ và giới hạn khuếch đại bằng clipLimit, nên marker vùng tối nổi lên nhưng ít khuếch đại nhiễu nền hơn HE toàn cục."],
  ["Detection", "Vì sao SimpleBlobDetector ổn hơn threshold cố định?", "Nó quét nhiều threshold và gom blob ổn định qua các mức. Marker thật xuất hiện bền hơn noise, còn threshold cố định dễ miss marker rìa tối."],
  ["Detection", "min_area=30 và max_area=500 bảo vệ khỏi lỗi gì?", "min_area loại noise/artifact nhỏ; max_area loại highlight hoặc blob dính lớn. Hai giá trị này phụ thuộc optical setup."],
  ["Tracking", "Forward-backward check hoạt động thế nào?", "Track điểm ref sang frame biến dạng, rồi track ngược về ref. Nếu điểm quay lại lệch quá fb_threshold hoặc status lỗi, marker bị đánh invalid."],
  ["Tracking", "Vì sao Pyramid LK cần max_level=3?", "LK cơ bản chỉ tốt với motion nhỏ. Pyramid bắt motion lớn ở ảnh downsample rồi refine ở level cao hơn, với win 21 có thể chịu motion lớn hơn nhiều."],
  ["Tracking", "Deadzone nên tắt trong trường hợp nào?", "Khi cần giữ tín hiệu nhỏ tích lũy như slip chậm/creep. Deadzone hữu ích để lọc rung nhưng có thể xoá motion nhỏ có thật."],
  ["Visualization", "arrow_scale=3 có làm sai dữ liệu không?", "Không. Nó chỉ scale mũi tên trên ảnh hiển thị. Dữ liệu displacement dùng cho tính toán vẫn là giá trị thật."],
  ["Kiến trúc", "Vì sao detect reference rồi track thay vì detect lại mọi frame?", "Tracking giữ identity marker ổn định. Detect lại mỗi frame có thể đổi thứ tự, bỏ sót hoặc tạo marker giả, làm displacement sai."],
  ["Kiến trúc", "src/core có phải deep learning không?", "Không. Nó là pipeline xử lý ảnh cổ điển: preprocessing, blob detection, optical flow LK, validation và visualization."],
  ["Force Poly", "force_poly khác force_cnn ở điểm nào?", "force_poly không học từ ảnh raw. Nó dùng displacement field từ tracking, rút thành 9 feature vật lý rồi hồi quy lực; vì vậy nhẹ, nhanh, dễ debug hơn nhưng mất thông tin không gian chi tiết."],
  ["Force Poly", "Vì sao polynomial regression hợp với lực tiếp xúc?", "Hertz contact gợi ý quan hệ lực-biến dạng phi tuyến dạng lũy thừa. Polynomial bậc 2-3 là xấp xỉ thực dụng trong vùng hoạt động hữu hạn."],
  ["Feature Engineering", "9 feature của force_poly gồm những nhóm nào?", "Thống kê magnitude, hướng dx/dy, tổng magnitude chuẩn hóa, radial/tangential quanh tâm và valid_ratio."],
  ["Feature Engineering", "Vì sao chia dx, dy và tọa độ cho image_w?", "Để giảm phụ thuộc độ phân giải và dùng cùng một thang đo cho x/y, giúp radial/tangential có ý nghĩa hình học nhất quán."],
  ["Feature Engineering", "Feature nào đang dư thừa trong v1?", "disp_mag_sum_norm = sum(mag_valid)/N_valid, về toán học bằng disp_mag_mean. Đây là redundancy có thể thay bằng percentile, median hoặc moment không gian."],
  ["Poly Model", "Vì sao standardize trước polynomial expansion?", "Nếu không chuẩn hóa, term bình phương/tích chéo của feature scale lớn áp đảo gradient và weight decay. Standardize đưa feature về mean 0, std 1 theo train set."],
  ["Poly Model", "Công thức 55 monomial đến từ đâu?", "Số monomial bậc <= d với D feature là C(D+d,d). Với D=9, d=2: C(11,2)=55, gồm bias, 9 term bậc 1 và 45 term bậc 2."],
  ["Poly Model", "linear head và mlp_small khác nhau thế nào?", "linear là polynomial regression thuần và giải thích được hệ số. mlp_small thêm hidden 16 + ReLU, biểu diễn mạnh hơn nhưng không còn hệ số monomial trực tiếp."],
  ["Train/Eval", "Vì sao chia dữ liệu theo trial?", "Frame cùng trial có tương quan thời gian mạnh. Chia random theo frame gây temporal leakage và làm test quá lạc quan; split theo trial kiểm tra generalization thật hơn."],
  ["Train/Eval", "Vì sao inference force_poly dùng apply_deadzone=False?", "Lực nhỏ có thể tạo displacement chỉ 0.5-1px. Deadzone sẽ zero-out tín hiệu này, làm feature về gần 0 và dự đoán lực nhỏ sai."],
  ["Force Model", "force_model khác force_poly ở điểm nào?", "force_model giữ từng marker như một điểm 4D và học tổng hợp bằng shared MLP + pooling. force_poly nén cả frame thành 9 feature thủ công trước khi hồi quy."],
  ["Prepare", "Vì sao cần cache .npz trong force_model?", "Detection, tracking và force sync tốn I/O và thời gian. Cache marker-level giúp train nhanh, tái lập và dùng lại cho nhiều mô hình."],
  ["Prepare", "Nearest-neighbor force sync hoạt động thế nào?", "Với mỗi timestamp frame, chọn mẫu force gần nhất nếu lệch dưới tolerance. Nếu lệch quá tolerance thì bỏ frame để tránh label sai thời điểm."],
  ["Prepare", "Vì sao prepare dùng apply_deadzone=False?", "Training lực cần giữ displacement nhỏ ở vùng 0.2-0.5N. Deadzone sẽ xoá tín hiệu lực thấp và làm model học sai vùng gần 0N."],
  ["Dataset", "Input 4 chiều của mỗi marker là gì?", "[x_ref/W, y_ref/H, dx/W, dy/H]. Vị trí cho context không gian, displacement cho biến dạng tại marker đó."],
  ["Dataset", "Vì sao cần N_max và padding?", "Số marker khác nhau giữa trial nhưng PyTorch batch cần cùng shape. Pad zeros lên N_max và dùng mask để loại padding khỏi pooling."],
  ["Dataset", "Augmentation force_model cần lưu ý gì?", "Flip/rotation phải biến đổi nhất quán cả vị trí marker và vector displacement; chỉ áp dụng train, không áp dụng validation/test."],
  ["ForceNet", "Vì sao gọi là PointNet-style?", "Nó xử lý từng marker bằng cùng một MLP, rồi pooling để tạo vector global gần bất biến với thứ tự marker."],
  ["ForceNet", "Conv1d kernel=1 tương đương gì?", "Tương đương Linear layer áp độc lập cho từng marker với cùng trọng số, nhưng thuận tiện và hiệu quả trên tensor (B,C,N)."],
  ["ForceNet", "Vì sao dùng cả max pool và mean pool?", "Max bắt marker đáp ứng mạnh gần vùng chịu tải; mean bắt xu hướng biến dạng toàn bề mặt. Kết hợp vừa nhạy cực trị vừa ổn định hơn."],
  ["Train/Eval", "Vì sao force_model dùng Adam, wd=1e-4 còn force_poly dùng AdamW, wd=1e-2?", "force_poly có polynomial terms collinear nên cần regularization mạnh và decoupled decay. ForceNet dùng network nhỏ với BN/dropout, weight decay nhẹ là đủ trong config hiện tại."],
  ["Train/Eval", "Nếu deploy có nhiều marker hơn n_max_train thì sao?", "Đây là điểm yếu của padding-based setup. Cần retrain với n_max mới, truncate marker, hoặc chuyển sang dynamic collation/attention architecture."],
  ["Force CNN", "force_cnn khác force_model ở điểm nào?", "force_cnn học trực tiếp từ cặp ảnh ref-frame và không cần LK tracking tại inference. force_model dùng displacement field marker đã tracking."],
  ["CNN Dataset", "Vì sao input là 2 kênh [reference, frame]?", "Lực tương quan với biến dạng tương đối. Reference cung cấp baseline không tải, frame cung cấp trạng thái biến dạng để CNN học so sánh."],
  ["CNN Dataset", "Vì sao force_cnn lazy-load ảnh thay vì cache npz?", "Ảnh pixel lớn hơn displacement field rất nhiều. Lazy-load tiết kiệm RAM/disk; reference được cache vì dùng lại nhiều lần."],
  ["CNN Augment", "Vì sao augmentation phải paired?", "Nếu ref và frame bị transform khác nhau, mạng thấy displacement hoặc brightness difference giả. Paired transform giữ quan hệ vật lý giữa hai ảnh."],
  ["CNN Augment", "Brightness/contrast jitter áp như thế nào?", "Dùng cùng hệ số cho cả ref và frame để mô phỏng exposure drift chung; khác hệ số sẽ tạo tín hiệu lực giả."],
  ["ForceCNN", "Conv1 inflate từ ImageNet là gì?", "Lấy mean trọng số conv1 RGB qua 3 kênh để tạo filter grayscale, rồi lặp thành 2 kênh cho [ref, frame]."],
  ["ForceCNN", "Vì sao pretrained ImageNet vẫn hữu ích?", "Các lớp đầu học edge, blob, gradient và texture phổ quát. Marker grayscale cũng chứa cạnh và blob, nên transfer giúp hội tụ nhanh hơn."],
  ["ForceCNN", "SmallCNN dùng khi nào?", "Dùng để debug nhanh hoặc chạy máy yếu. Nó ít tham số hơn nhiều nhưng thường kém chính xác hơn ResNet18 pretrained."],
  ["CNN Train/Eval", "Vì sao force_cnn dùng AdamW và cosine scheduler?", "ResNet18 lớn cần regularization tốt và fine-tune cẩn thận. AdamW tách weight decay, cosine giảm LR mượt về cuối training."],
  ["CNN Infer", "Vì sao infer.py đặt pretrained=False khi load checkpoint?", "Checkpoint đã chứa weights đã train. pretrained=False tránh tải ImageNet weights thừa trước khi load state_dict."],
  ["Slip V1", "Slip V1 có phải mô hình học máy không?", "Không. Nó là thuật toán heuristic/thống kê trên vector chuyển động marker, không train và không có trọng số học được."],
  ["Slip V1", "Ý tưởng vật lý chính của Slip V1 là gì?", "Khi trượt, nhiều marker có xu hướng chuyển động cùng hướng tịnh tiến. Khi press/release, hướng thường phân tán hoặc mang dạng xuyên tâm."],
  ["Slip V1", "MRVL đo điều gì?", "MRVL đo độ đồng hướng của các vector chuyển động. R gần 1 nghĩa là cùng hướng, R gần 0 nghĩa là hướng triệt tiêu/phân tán."],
  ["Slip V1", "Vì sao dùng weighted MRVL?", "Marker dịch chuyển mạnh thường mang evidence slip rõ hơn, nên được weight cao hơn marker chuyển động yếu."],
  ["Slip Filters", "Rebound filter dùng dot product thế nào?", "Nếu displacement frame hiện tại ngược hướng deformation so với reference, dot âm, marker có thể đang hồi đàn hồi nên bị loại."],
  ["Slip Filters", "Vì sao cần khử radial component?", "Press/release tạo chuyển động nở/co quanh tâm, không phải slip tịnh tiến. Trừ radial giúp MRVL tập trung vào thành phần trượt."],
  ["Slip Score", "Vì sao cần translation gain nếu đã có MRVL?", "MRVL chỉ biết hướng có đồng nhất không. Translation gain kiểm tra có thành phần tịnh tiến đủ rõ để giống slip thật."],
  ["Slip Score", "EMA smoothing có tác dụng gì?", "Làm mượt score để tránh slip bật/tắt vì nhiễu, nhưng adaptive alpha giúp phản ứng nhanh khi evidence mạnh."],
  ["Slip V1", "history_buffer_length=5 ảnh hưởng thế nào?", "So sánh current với frame quá khứ giúp motion tích lũy rõ hơn, bắt slip nhỏ tốt hơn nhưng tăng độ trễ."],
  ["Slip V1", "Điểm yếu chính của Slip V1 là gì?", "Dựa vào threshold thủ công và một history scale, dễ bỏ sót slow slip hoặc slip lẫn pressing phức tạp; phụ thuộc LK tracking."]
];

const quiz = [
  {
    q: "Bước nào cố ý làm mất thông tin ánh sáng tuyệt đối?",
    options: ["Subtract + normalize", "Forward-backward check", "Convexity filter", "Arrow scaling"],
    answer: 0,
    why: "Sau trừ nền và normalize, hệ chỉ giữ tương phản tương đối để phát hiện marker sáng hơn xung quanh."
  },
  {
    q: "Lý do chính không dùng Gaussian blur 101x101 là gì?",
    options: ["Không chạy được với OpenCV", "Đắt hơn nhiều mà không cải thiện đáng kể nền low-pass", "Làm marker sáng hơn", "Không hỗ trợ ảnh grayscale"],
    answer: 1,
    why: "Mục tiêu chỉ là ước lượng nền chậm biến thiên, box blur đủ tốt và nhanh hơn với kernel lớn."
  },
  {
    q: "SimpleBlobDetector giải quyết điểm yếu nào của threshold + findContours?",
    options: ["Không cần ảnh đầu vào", "Tự tạo dữ liệu nhãn", "Quét nhiều threshold và vote blob ổn định", "Tự sửa lens distortion"],
    answer: 2,
    why: "Multi-threshold làm detector bền hơn khi marker có độ sáng không đều."
  },
  {
    q: "fb_threshold=2.0 dùng để làm gì?",
    options: ["Ngưỡng màu của marker", "Ngưỡng lỗi đi-về của LK", "Kích thước kernel CLAHE", "Scale mũi tên"],
    answer: 1,
    why: "Marker hợp lệ khi forward và backward status tốt, đồng thời điểm quay lại gần reference dưới 2px."
  },
  {
    q: "Rủi ro nếu min_circularity đặt quá cao là gì?",
    options: ["Bỏ sót marker thật bị méo khi gel biến dạng", "Ảnh bị tối hơn", "LK chạy chậm hơn 10 lần", "Không tạo được valid mask"],
    answer: 0,
    why: "Marker lý tưởng tròn nhưng trong thực tế có thể thành ellipse hoặc méo nhẹ."
  },
  {
    q: "valid mask quan trọng vì sao?",
    options: ["Để đổi màu UI", "Để module sau bỏ marker tracking lỗi", "Để thay threshold detector", "Để tăng CLAHE clipLimit"],
    answer: 1,
    why: "Force/slip/visualization cần tránh dùng marker không đáng tin cậy."
  },
  {
    q: "Khi đổi camera hoặc khoảng cách lens, tham số nào dễ phải chỉnh nhất?",
    options: ["Area threshold và blur/LK scale", "Tên file Python", "Màu mũi tên duy nhất", "Số câu hỏi phản biện"],
    answer: 0,
    why: "Kích thước marker theo pixel và ánh sáng thay đổi theo optical setup."
  },
  {
    q: "Visualization arrow_scale ảnh hưởng gì?",
    options: ["Chỉ ảnh hiển thị", "Dữ liệu displacement thật", "Kết quả SimpleBlobDetector", "CLAHE object cache"],
    answer: 0,
    why: "Scale chỉ phóng đại mũi tên để nhìn pattern, không thay đổi số liệu tính toán."
  },
  {
    q: "force_poly nhận input trực tiếp từ đâu?",
    options: ["Ảnh RGB raw", "Displacement field và valid mask từ tracking", "File âm thanh force gauge", "Bounding box YOLO"],
    answer: 1,
    why: "force_poly phụ thuộc vào src/core/force_model cache: ref_pts, disp, valid, force và image_w."
  },
  {
    q: "Với 9 feature và polynomial degree=2, số monomial là bao nhiêu?",
    options: ["18", "45", "55", "913"],
    answer: 2,
    why: "C(D+d,d)=C(11,2)=55, gồm cả bias term."
  },
  {
    q: "Lý do quan trọng nhất để split theo trial là gì?",
    options: ["Tăng kích thước file cache", "Tránh temporal leakage giữa frame gần nhau", "Để bỏ validation set", "Để tạo thêm marker"],
    answer: 1,
    why: "Frame liền kề trong cùng trial rất giống nhau; random frame split làm test thấy dữ liệu gần như train."
  },
  {
    q: "Feature nào trong v1 bị dư thừa về mặt toán học?",
    options: ["valid_ratio", "radial_disp_mean", "disp_mag_sum_norm", "dy_mean"],
    answer: 2,
    why: "disp_mag_sum_norm = sum(mag)/N_valid, tương đương mean(mag)."
  },
  {
    q: "Vì sao Huber loss phù hợp hơn MSE trong training force_poly?",
    options: ["Không cần label lực", "Robust hơn với spike/outlier nhưng vẫn mượt gần 0", "Tự sinh polynomial terms", "Đảm bảo output không âm"],
    answer: 1,
    why: "MSE phạt spike quá mạnh; Huber chuyển sang MAE-like khi lỗi lớn."
  },
  {
    q: "Khi dùng mlp_small, phát biểu nào đúng?",
    options: ["Vẫn giải thích được từng hệ số monomial trực tiếp", "Không cần standardize", "Biểu diễn mạnh hơn linear nhưng kém interpretability", "Không có tham số trainable"],
    answer: 2,
    why: "Hidden layer và ReLU làm ảnh hưởng từng monomial phụ thuộc activation, không còn một hệ số trực tiếp."
  },
  {
    q: "Vì sao force_poly inference tắt deadzone?",
    options: ["Để tăng FPS của camera", "Để giữ displacement nhỏ tương ứng lực nhỏ", "Để bỏ valid mask", "Để đổi đơn vị Newton sang pixel"],
    answer: 1,
    why: "Deadzone có thể xoá motion 0.5-1px, trong khi đó là tín hiệu vật lý quan trọng cho lực nhỏ."
  },
  {
    q: "force_model biểu diễn mỗi marker bằng vector nào?",
    options: ["[x/W, y/H, dx/W, dy/H]", "[force, time, image_w, image_h]", "[mean, max, std, valid_ratio]", "[R, G, B, alpha]"],
    answer: 0,
    why: "force_model giữ vị trí reference và displacement của từng marker, chuẩn hóa theo kích thước ảnh."
  },
  {
    q: "Vì sao ForceNet dùng shared MLP + pooling?",
    options: ["Để phụ thuộc mạnh vào thứ tự marker", "Để xử lý tập marker và gần bất biến với hoán vị", "Để bỏ mask valid", "Để train trực tiếp từ ảnh raw"],
    answer: 1,
    why: "Marker là tập điểm; thứ tự marker không có ý nghĩa vật lý, nên PointNet-style phù hợp."
  },
  {
    q: "Trong mask-aware max pool, marker invalid được xử lý thế nào?",
    options: ["Đặt rất âm trước khi lấy max", "Nhân đôi giá trị", "Đặt force bằng 1N", "Chuyển thành ảnh RGB"],
    answer: 0,
    why: "Invalid/padding phải bị loại khỏi max; nếu không chúng có thể ảnh hưởng global feature."
  },
  {
    q: "Lý do dùng cả max pool và mean pool là gì?",
    options: ["Để tăng số epoch", "Max bắt cực trị, mean bắt xu hướng toàn cục", "Để thay thế valid mask", "Để chuẩn hóa timestamp"],
    answer: 1,
    why: "Max nhạy với điểm chịu tải mạnh; mean ổn định với phân bố biến dạng chung."
  },
  {
    q: "Augmentation nào chỉ được dùng cho train set?",
    options: ["Flip, rotation, displacement noise", "Đổi force label ngẫu nhiên", "Bỏ valid mask", "Dùng test prediction làm label"],
    answer: 0,
    why: "Validation/test phải phản ánh dữ liệu thật và deterministic."
  },
  {
    q: "Checkpoint best.pt của force_model cần lưu n_max vì sao?",
    options: ["Để padding input inference đúng shape đã train", "Để đổi màu plot", "Để tính CLAHE", "Để bỏ batch size"],
    answer: 0,
    why: "n_max xác định chiều N của tensor input; deploy phải pad/truncate nhất quán."
  },
  {
    q: "Rủi ro của BatchNorm với padding zeros là gì?",
    options: ["Padding zeros có thể kéo mean/var activation", "BatchNorm làm mất file cache", "BatchNorm tự đổi force label", "Padding zeros làm Conv1d không chạy"],
    answer: 0,
    why: "BN tính statistics trên B x N_max, có thể gồm nhiều điểm padding nếu mask không tham gia BN."
  },
  {
    q: "force_model hiện tại dùng optimizer nào?",
    options: ["Adam lr=1e-3 wd=1e-4", "AdamW lr=5e-3 wd=1e-2", "SGD không weight decay", "RMSProp với scheduler cosine"],
    answer: 0,
    why: "Tài liệu ghi force_model dùng Adam, không scheduler, regularization nhẹ hơn force_poly."
  },
  {
    q: "force_cnn nhận input model dạng nào?",
    options: ["(2,H,W) gồm reference và frame grayscale", "(N,4) marker features", "(9,) scalar features", "(3,) RGB force vector"],
    answer: 0,
    why: "Hai ảnh grayscale được stack thành 2 channel: ref không tải và frame biến dạng."
  },
  {
    q: "Ưu điểm triển khai lớn của force_cnn là gì?",
    options: ["Không cần detect marker/LK tracking tại inference", "Không cần reference image", "Không cần label force", "Luôn giải thích được hệ số"],
    answer: 0,
    why: "Inference chỉ cần ref, frame và checkpoint; CNN học trực tiếp từ pixel."
  },
  {
    q: "Conv1 inflate trong ResNet18 làm gì?",
    options: ["Chuyển conv1 RGB 3 kênh thành conv1 2 kênh", "Tăng image_size lên 640x480", "Bỏ toàn bộ pretrained weights", "Tạo valid mask"],
    answer: 0,
    why: "Mean RGB weights thành filter grayscale rồi repeat cho hai kênh ref/frame."
  },
  {
    q: "Vì sao augmentation của force_cnn phải paired?",
    options: ["Để không tạo biến dạng giả giữa reference và frame", "Để tăng force label", "Để thay ResNet bằng PointNet", "Để bỏ trial split"],
    answer: 0,
    why: "Transform khác nhau giữa hai ảnh sẽ phá quan hệ vật lý mà model cần học."
  },
  {
    q: "Backbone mặc định của force_cnn là gì?",
    options: ["ResNet18 pretrained", "YOLOv8", "ForceNet Conv1d", "PolynomialRegressor"],
    answer: 0,
    why: "Config hiện tại dùng ResNet18 với ImageNet weights và regression head 512->128->1."
  },
  {
    q: "Tại sao batch_size force_cnn nhỏ hơn force_model?",
    options: ["Ảnh và ResNet feature maps tốn memory hơn marker tensor", "force_cnn không dùng GPU", "force_model không có batch", "CNN không cần gradient"],
    answer: 0,
    why: "Input ảnh (2,240,320) và activation ResNet lớn hơn rất nhiều so với (N,4)."
  },
  {
    q: "Rủi ro chính của CNN end-to-end là gì?",
    options: ["Học shortcut từ ánh sáng/nền thay vì biến dạng", "Không đọc được pixel", "Không thể dùng Huber loss", "Không thể resize ảnh"],
    answer: 0,
    why: "CNN black-box có thể bám vào artifact tương quan với force nếu dataset không được kiểm soát."
  },
  {
    q: "Khi load checkpoint force_cnn để inference nên đặt pretrained thế nào?",
    options: ["pretrained=False", "pretrained=True bắt buộc", "Không load state_dict", "Chỉ dùng ImageNet weights"],
    answer: 0,
    why: "Weights đã nằm trong checkpoint best.pt; không cần tải ImageNet weights."
  },
  {
    q: "Slip V1 dựa trên đại lượng chính nào?",
    options: ["Weighted MRVL", "Huber loss", "Cosine scheduler", "Conv1 inflate"],
    answer: 0,
    why: "MRVL đo mức độ đồng hướng của vector chuyển động marker, là tín hiệu chính để phát hiện slip."
  },
  {
    q: "Slip V1 có phải học máy không?",
    options: ["Không, là heuristic/thống kê", "Có, là ResNet18", "Có, là PointNet", "Có, là polynomial regression"],
    answer: 0,
    why: "Slip V1 không train, không có parameter học được; chỉ có threshold, filter và EMA state."
  },
  {
    q: "MRVL gần 1 nghĩa là gì?",
    options: ["Các vector chuyển động gần cùng hướng", "Không có marker valid", "Force label bị lệch", "Ảnh reference sai"],
    answer: 0,
    why: "Các góc vector cùng hướng thì tổng vector đơn vị có độ dài lớn."
  },
  {
    q: "min_motion_thresh=2px dùng để làm gì?",
    options: ["Bỏ chuyển động nhỏ nhiều nhiễu", "Chọn backbone CNN", "Tính force zero offset", "Resize ảnh"],
    answer: 0,
    why: "Vector nhỏ có hướng không ổn định do LK noise/rung, nên không nên đưa vào MRVL."
  },
  {
    q: "Rebound filter loại marker khi nào?",
    options: ["Khi displacement đi ngược deformation hiện tại", "Khi marker có force lớn", "Khi ảnh là RGB", "Khi batch size nhỏ"],
    answer: 0,
    why: "Dot product âm gợi ý marker đang hồi về reference, không phải slip chủ động."
  },
  {
    q: "Pressing gate suppress trong trường hợp nào?",
    options: ["Đang pressing và translation yếu", "MRVL bằng 1", "Có nhiều marker valid", "Backbone là SmallCNN"],
    answer: 0,
    why: "Nhấn xuống có thể tạo nhiều chuyển động nhưng không phải slip; nếu translation rõ thì vẫn cho qua."
  },
  {
    q: "history_buffer_length dài hơn có trade-off gì?",
    options: ["Bắt slow slip tốt hơn nhưng tăng độ trễ", "Luôn giảm false negative về 0", "Bỏ valid mask", "Tăng tham số ResNet"],
    answer: 0,
    why: "So sánh với frame xa hơn làm displacement tích lũy lớn hơn nhưng phản ứng chậm hơn."
  },
  {
    q: "Vì sao Slip V1 có thể bỏ sót slow slip?",
    options: ["Motion giữa frame dưới min_motion_thresh", "MRVL không dùng angle", "Không có reference image", "Do dùng RGB"],
    answer: 0,
    why: "Nếu chuyển động mỗi window quá nhỏ, marker không qua motion filter nên evidence không đủ."
  }
];

const defense = [
  ["Vì sao cần tiền xử lý ảnh trước detect marker?", "Ảnh tactile bị vignetting, drift ánh sáng và nền không đồng đều. Tiền xử lý khử thành phần nền chậm biến thiên, normalize dải sáng và tăng tương phản cục bộ để marker ở rìa vẫn được phát hiện ổn định."],
  ["Vì sao dùng box blur thay vì Gaussian blur?", "Với kernel lớn, mục tiêu là low-pass nền chậm biến thiên chứ không cần trọng số Gaussian. Box blur nhanh hơn đáng kể và đủ tốt cho nền LED/vignetting."],
  ["Vì sao kernel blur là 101x101?", "Kernel cần lớn hơn marker nhiều lần để marker không đi vào nền ước lượng. 101x101 khoảng nhiều lần đường kính marker hiện tại, là cân bằng thực nghiệm giữa độ mượt nền và chi phí."],
  ["Vì sao dùng CLAHE?", "CLAHE tăng tương phản cục bộ và giới hạn khuếch đại nhiễu. Nó phù hợp với ảnh marker có độ sáng không đều giữa trung tâm và rìa."],
  ["Vì sao không dùng threshold cố định?", "Threshold cố định nhạy với ánh sáng và vị trí trong ảnh. SimpleBlobDetector quét nhiều threshold nên bền hơn khi marker sáng không đồng đều."],
  ["Vì sao dùng SimpleBlobDetector thay vì YOLO?", "Marker là blob sáng gần tròn trong setup cố định. Detector cổ điển nhanh, dễ debug, không cần dataset nhãn hoặc model nặng; deep detector tăng chi phí mà lợi ích thấp cho domain này."],
  ["Vì sao detect reference rồi track?", "Cách này giữ identity marker theo thời gian. Detect lại mọi frame có thể đổi thứ tự, miss marker hoặc tạo false marker, làm displacement sai."],
  ["Vì sao dùng Lucas-Kanade?", "LK nhanh, có sẵn trong OpenCV, hỗ trợ sub-pixel và phù hợp khi đã biết điểm reference. Pyramid LK xử lý được motion lớn hơn LK cơ bản."],
  ["Forward-backward check có bắt mọi lỗi không?", "Không. Nó bắt nhiều lỗi drift không nhất quán, nhưng nếu sai một cách nhất quán hai chiều thì FB error vẫn nhỏ. Vì vậy valid mask là kiểm soát chất lượng, không phải bảo đảm tuyệt đối."],
  ["Deadzone có tác dụng gì?", "Deadzone reset displacement rất nhỏ về 0 để loại rung cơ học và nhiễu ảnh. Không nên dùng khi cần tín hiệu nhỏ tích lũy như slip chậm."],
  ["Pipeline có chạy realtime được không?", "Có, vì các lựa chọn ưu tiên tốc độ: box blur, cache CLAHE/detector, LK thay vì matching toàn cục, visualization OpenCV đơn giản. Tốc độ thực tế phụ thuộc số marker, độ phân giải và CPU."],
  ["Điểm yếu chính của src/core là gì?", "Phụ thuộc chất lượng ảnh và tham số thủ công. Khi thay camera, lens, ánh sáng hoặc marker size, cần hiệu chỉnh threshold, area, blur kernel và LK window."],
  ["Tại sao chọn force_poly trong khi CNN có thể chính xác hơn?", "force_poly nhẹ, train nhanh, inference rất nhanh và dễ phân tích. Nó là baseline có ý nghĩa vật lý để chứng minh displacement field đã chứa thông tin lực đáng kể. CNN có thể chính xác hơn nhưng cần nhiều dữ liệu/tài nguyên hơn và khó debug hơn."],
  ["Vì sao không đưa toàn bộ marker vào force_poly?", "Rút gọn thành 9 feature giảm số tham số và nguy cơ overfit khi dữ liệu vừa phải. Đánh đổi là mất chi tiết không gian, nên có thể so sánh thêm với force_model hoặc force_cnn."],
  ["Vì sao dùng polynomial degree=2?", "Bậc 1 thiếu tương tác giữa feature; bậc 2 thêm bình phương và tích chéo với chỉ 55 terms. Bậc 3 thành 220 terms, có thể hợp lực lớn hơn nhưng dễ cần nhiều data và regularization hơn."],
  ["Số 55 monomial đến từ đâu?", "Số monomial bậc nhỏ hơn hoặc bằng d với D feature là C(D+d,d). Với D=9, d=2 thì C(11,2)=55, gồm 1 bias, 9 term bậc 1 và 45 term bậc 2."],
  ["Vì sao phải standardize trước polynomial expansion?", "Polynomial sinh x_i^2 và x_i*x_j. Nếu feature khác scale, term bậc cao của feature lớn áp đảo optimization. Standardize theo train mean/std giúp gradient và weight decay ổn định hơn."],
  ["Vì sao dùng Huber loss?", "Huber gần 0 giống MSE nên học mượt, nhưng lỗi lớn giống MAE nên ít bị spike force gauge hoặc frame tracking lỗi chi phối hơn MSE."],
  ["Vì sao dùng AdamW và weight_decay=1e-2?", "Polynomial terms thường tương quan mạnh. Weight decay giúp shrink trọng số lớn và giảm overfit; AdamW tách weight decay khỏi gradient update rõ ràng hơn Adam + L2 thường."],
  ["mlp_small có còn là polynomial regression không?", "Không phải polynomial regression thuần như linear head. Cách gọi chính xác hơn là polynomial feature expansion kết hợp small MLP regressor."],
  ["Nếu dùng mlp_small còn giải thích hệ số được không?", "Không trực tiếp. Có thể phân tích gián tiếp bằng ablation/permutation importance hoặc train thêm head=linear để xuất coefficients.csv."],
  ["Feature disp_mag_sum_norm có vấn đề gì?", "Nó bằng disp_mag_mean vì sum/N_valid = mean. Đây là feature dư thừa, gây multicollinearity; có thể thay bằng percentile, median, kurtosis hoặc moment không gian."],
  ["Vì sao valid_ratio liên quan đến lực?", "Nó không trực tiếp là lực, nhưng phản ánh chất lượng quan sát. Khi mất nhiều marker, thống kê trên marker còn lại kém tin cậy; valid_ratio giúp model biết tình trạng đó."],
  ["Mô hình có đảm bảo lực không âm không?", "Không. Head cuối tuyến tính có thể cho output âm với input ngoài phân phối. Khi triển khai có thể clip max(0,pred) hoặc dùng Softplus nếu cần ràng buộc vật lý."],
  ["Trial-level split có loại bỏ mọi leakage không?", "Nó loại bỏ temporal leakage giữa frame liền kề, nhưng vẫn cần kiểm tra phân phối force giữa train/test. Nếu test có force range chưa có trong train, model phải extrapolate và R2 có thể giảm."],
  ["Khi nào nên dùng force_poly?", "Khi cần baseline nhẹ, nhanh, dễ debug hoặc chạy trên thiết bị hạn chế. Nếu ưu tiên độ chính xác cao nhất và có đủ dữ liệu/tài nguyên, force_cnn hoặc force_model có thể phù hợp hơn."],
  ["Tại sao chọn PointNet-style cho force_model?", "Vì dữ liệu là tập marker có số lượng thay đổi, có marker mất tracking và thứ tự marker không có ý nghĩa vật lý. Shared MLP xử lý từng marker giống nhau, pooling tạo biểu diễn global gần bất biến với hoán vị."],
  ["Vì sao không dùng MLP phẳng trên toàn bộ marker?", "Flatten phụ thuộc vào thứ tự marker và yêu cầu số marker cố định. Khi marker thiếu hoặc trial có N khác nhau, flatten khó xử lý. Shared MLP + mask-aware pooling linh hoạt hơn."],
  ["Vì sao cần vị trí reference trong input?", "Cùng một displacement có ý nghĩa khác nhau tùy marker ở tâm, rìa hoặc gần vùng tiếp xúc. Vị trí reference cung cấp context không gian để map biến dạng sang lực."],
  ["Vì sao x, y, dx, dy được normalize theo W/H?", "Chuẩn hóa giảm phụ thuộc pixel và độ phân giải. dx dùng W, dy dùng H vì chúng nằm trên hai trục ảnh khác nhau."],
  ["Vì sao dùng Conv1d kernel=1?", "Conv1d k=1 tương đương Linear áp lên từng marker với cùng trọng số. Cách này hiệu quả cho tensor (B,C,N) và giữ đúng ý tưởng per-point shared MLP."],
  ["Vì sao pooling phải mask-aware?", "Input có marker invalid từ LK và padding tới N_max. Nếu không mask, các điểm không hợp lệ sẽ tham gia max/mean pool và làm sai vector global."],
  ["Nếu thứ tự marker thay đổi thì output có đổi không?", "Về nguyên lý, nếu hoán vị cả feature và mask cùng nhau, shared MLP + max/mean pooling cho kết quả không đổi. Đây là tính chất phù hợp với dữ liệu dạng tập."],
  ["Vì sao cần padding tới N_max?", "Batch training cần tensor cùng shape. N_max là số marker lớn nhất trong cache; trial ít marker được pad zeros và mask=False để pooling bỏ qua."],
  ["Vì sao không dùng attention thay pooling?", "Attention có thể mạnh hơn nhưng phức tạp hơn và cần nhiều dữ liệu hơn. Max+mean pooling là baseline đơn giản, ổn định và ít tham số."],
  ["Augmentation flip/rotation có làm sai force label không?", "Không nếu force label là scalar magnitude và transform được áp dụng nhất quán cho vị trí/displacement. Tuy nhiên nếu setup không đối xứng, nên giảm hoặc tắt augmentation."],
  ["BatchNorm với padding zeros có vấn đề gì?", "BN tính mean/var trên B x N_max, bao gồm padding zeros trước khi mask pooling. Nếu padding nhiều, statistics có thể lệch; masked BN, InstanceNorm hoặc GroupNorm là hướng cải thiện."],
  ["Vì sao force_model dùng Adam thay vì AdamW?", "Weight decay của force_model nhỏ 1e-4, mạng có BN/dropout, nên khác biệt Adam và AdamW ít hơn trong thực hành. force_poly dùng AdamW vì polynomial coefficients collinear và weight decay lớn hơn."],
  ["Nếu deployment có nhiều marker hơn n_max_train thì sao?", "Input shape không còn khớp với padding target đã train. Cần retrain với n_max mới, truncate marker theo tiêu chí hợp lý hoặc dùng kiến trúc dynamic hơn."],
  ["Model có thể học việc lực lớn làm mất marker không?", "Có thể học ngầm qua pattern mask/pooling nếu consistent, nhưng chưa explicit. Cải thiện tốt hơn là thêm valid_ratio hoặc feature confidence vào input/global feature."],
  ["Điểm khác nhau giữa force_model và force_cnn là gì?", "force_cnn học trực tiếp từ ảnh raw nên có thể khai thác texture đầy đủ nhưng nặng và khó giải thích hơn. force_model chỉ dùng marker đã tracking, nhẹ hơn nhưng phụ thuộc detection/LK."],
  ["Tại sao dùng CNN trong khi đã có marker tracking?", "CNN không phụ thuộc vào detect/LK tracking, nên vẫn có thể học khi marker bị che, nhòe hoặc tracking fail. Nó cũng có thể khai thác texture, gradient và signal thị giác ngoài displacement field. Đổi lại là nhiều tham số, cần GPU và khó giải thích hơn."],
  ["Vì sao input force_cnn là 2 kênh reference-frame?", "Reference là trạng thái không tải, frame là trạng thái chịu lực. Cặp ảnh cho phép model học thay đổi tương đối do lực gây ra, thay vì phải tự suy ra baseline từ một ảnh đơn."],
  ["Vì sao không dùng ảnh RGB?", "Tín hiệu chính là hình dạng, marker, độ tương phản và biến dạng. Grayscale giảm số kênh, giảm chi phí và tránh học màu không cần thiết. Nếu màu thật sự có thông tin, RGB hoặc 6 kênh là thí nghiệm mở rộng."],
  ["Vì sao dùng ResNet18?", "ResNet18 cân bằng giữa độ mạnh và chi phí, có residual connection và pretrained ImageNet. Nó nhẹ hơn ResNet50 nhưng mạnh hơn SmallCNN debug."],
  ["Pretrained ImageNet có phù hợp ảnh tactile grayscale không?", "Có phần phù hợp vì lớp đầu học edge/blob/gradient phổ quát. Tuy nhiên domain khác ảnh tự nhiên, nên pretrained là điểm khởi đầu tốt chứ không phải đảm bảo tối ưu."],
  ["Conv1 RGB chuyển sang 2 kênh thế nào?", "Lấy trung bình trọng số conv1 qua 3 kênh RGB thành filter grayscale rồi lặp filter đó cho 2 kênh reference và frame."],
  ["Vì sao không thêm channel thứ ba frame-ref?", "CNN 2 kênh có thể tự học phép so sánh giữa ref và frame. Difference channel có thể giúp học nhanh hơn nhưng thêm giả định thiết kế; đây là hướng ablation hợp lý."],
  ["Resize 240x320 có làm mất thông tin không?", "Có thể mất biến dạng rất nhỏ. Đây là trade-off giữa chi tiết ảnh và GPU memory/tốc độ. Có thể kiểm chứng bằng train lại ở 480x640 nếu tài nguyên cho phép."],
  ["Vì sao augmentation force_cnn phải đồng bộ?", "Lực nằm trong quan hệ giữa hai ảnh. Transform khác nhau giữa ref và frame tạo deformation giả hoặc brightness difference giả, làm model học sai."],
  ["Brightness/contrast jitter có phá dữ liệu không?", "Không nếu dùng cùng hệ số cho cả ref và frame. Nó mô phỏng drift ánh sáng chung; nếu dùng khác hệ số, model có thể học chênh lệch exposure như lực giả."],
  ["Vì sao force_cnn không có prepare.py như force_model?", "force_model cần cache displacement field sau detect/LK. force_cnn chỉ cần load ảnh, resize, normalize, nên làm on-the-fly trong Dataset. Trade-off là phụ thuộc tốc độ disk I/O."],
  ["Vì sao dùng AdamW cho force_cnn?", "ResNet18 có hơn 11 triệu tham số, nguy cơ overfit lớn. AdamW decoupled weight decay rõ ràng hơn Adam khi fine-tune mô hình lớn."],
  ["Vì sao dùng cosine scheduler?", "Fine-tune CNN pretrained cần LR cao vừa phải ở đầu và thấp dần ở cuối để hội tụ mịn. CosineAnnealingLR giảm LR mượt qua epoch."],
  ["CNN có thể học nhầm background/ánh sáng không?", "Có. Đây là rủi ro của end-to-end pixel model. Cần kiểm soát dataset, augmentation, đánh giá nhiều session và dùng saliency/Grad-CAM để kiểm tra vùng model chú ý."],
  ["Nếu reference không đúng session thì sao?", "Baseline ảnh sai tạo deformation giả, làm dự đoán lệch. Deploy phải dùng reference không tải của cùng setup/session hoặc cơ chế chọn reference đáng tin cậy."],
  ["Inference của force_cnn có cần detect marker không?", "Không. Inference chỉ load ref và frame, grayscale/resize/normalize, stack thành (1,2,H,W), load checkpoint và forward."],
  ["Có nên freeze backbone không?", "Có thể nếu dữ liệu ít để giảm overfit và train nhanh hơn. Nhưng ảnh tactile khác ImageNet, freeze quá nhiều có thể giảm độ chính xác. Đây là ablation nên thử."],
  ["Hạn chế lớn nhất của force_cnn là gì?", "Chi phí tính toán và khả năng giải thích thấp. Model lớn, train lâu, dễ học shortcut nếu dataset không kiểm soát; cần công cụ giải thích như Grad-CAM hoặc saliency."],
  ["Slip V1 có phải mô hình học máy không?", "Không. Slip V1 là thuật toán heuristic/thống kê dựa trên vector chuyển động marker, không cần dữ liệu train và không có trọng số học được."],
  ["Ý tưởng chính của Slip V1 là gì?", "Slip tạo chuyển động marker cùng hướng tịnh tiến. V1 đo độ đồng hướng bằng weighted MRVL, sau đó boost và smooth score trước khi so với threshold."],
  ["MRVL là gì?", "MRVL là độ dài vector tổng trung bình của các hướng chuyển động. Góc cùng hướng cho R gần 1, góc phân tán triệt tiêu nhau cho R gần 0."],
  ["Vì sao dùng weighted MRVL?", "Marker chuyển động mạnh thường đáng tin hơn marker chuyển động yếu, nên V1 weight theo magnitude để vector có evidence rõ đóng góp nhiều hơn."],
  ["Vì sao cần min_motion_thresh?", "Chuyển động quá nhỏ dễ là nhiễu tracking hoặc rung. Góc của vector nhỏ không ổn định, đưa vào MRVL sẽ làm score nhiễu."],
  ["Vì sao cần ít nhất 5 marker moving?", "Một vài marker chuyển động có thể là tracking lỗi hoặc nhiễu cục bộ. min_moving_markers yêu cầu evidence từ nhiều marker để đáng tin hơn."],
  ["Rebound filter làm gì?", "Nó loại marker đang hồi đàn hồi về reference. Nếu displacement hiện tại ngược hướng deformation hiện tại, dot product âm và marker bị xem là rebound."],
  ["Vì sao cần khử radial component?", "Press/release có thể tạo nở/co xuyên tâm quanh tâm marker. Thành phần này không phải slip tịnh tiến, nên V1 trừ xấp xỉ radial_scale * centered_ref trước khi tính MRVL."],
  ["Radial removal V1 có đầy đủ không?", "Không. Nó là xấp xỉ nhẹ để realtime. V2 phân rã translation/radial rõ hơn và đa thang thời gian."],
  ["Vì sao cần translation gain?", "MRVL chỉ đo đồng hướng. Translation gain giúp score cao hơn khi có chuyển động tịnh tiến thật sự, giảm false positive từ pattern đồng hướng yếu."],
  ["Vì sao cần participation và motion gain?", "Participation gain tăng độ tin khi nhiều marker tham gia; motion gain giảm score nếu chuyển động trung bình quá nhỏ."],
  ["Pressing gate hoạt động thế nào?", "Nếu mean deformation tăng nhanh hơn press_rate_threshold và translation_gain thấp, V1 decay score và trả phase pressing. Nếu vừa nhấn vừa trượt rõ, translation mạnh vẫn được cho qua."],
  ["Vì sao dùng EMA smoothing?", "Score từng frame có thể nhiễu. EMA giảm flicker, còn adaptive alpha tăng phản ứng khi raw_score, coherence và translation evidence mạnh."],
  ["Vì sao slip_threshold là 0.8?", "r_value nằm 0-1. Ngưỡng 0.8 yêu cầu evidence cao để giảm false positive. Đây là tham số thực nghiệm cần tuning theo setup."],
  ["history_buffer_length=5 ảnh hưởng thế nào?", "Buffer dài hơn làm displacement tích lũy rõ hơn và giúp bắt slow slip, nhưng tăng độ trễ. Buffer ngắn hơn phản ứng nhanh nhưng nhạy nhiễu."],
  ["Vì sao không so sánh luôn với reference?", "Slip là chuyển động theo thời gian gần hiện tại. So với reference đo deformation tích lũy, dễ lẫn press/release; so current với past đo motion gần thời điểm hiện tại."],
  ["Slip V1 có phụ thuộc framerate không?", "Có. Cùng vận tốc slip nhưng framerate cao làm pixel/frame nhỏ hơn. min_motion_thresh và press_rate_threshold cần phù hợp framerate."],
  ["Khi nào phase là insufficient_motion?", "Khi số marker có displacement lớn hơn min_motion_thresh nhỏ hơn min_moving_markers. Detector xem evidence chưa đủ và decay score."],
  ["Vì sao cần Slip V2 nếu có V1?", "V2 xử lý một số điểm yếu của V1 bằng phân rã translation/radial đa thang, giúp bắt slow slip và tách press/release tốt hơn nhưng phức tạp hơn."]
];

const tuning = [
  {
    name: "blur_kernel",
    min: 21,
    max: 151,
    step: 10,
    value: 101,
    unit: "px",
    low: "Nền có thể còn vignetting; marker dễ lọt vào nền gây halo sau subtract.",
    high: "Nền mượt hơn nhưng chậm hơn; có thể làm mất biến thiên nền cục bộ.",
    sweet: "Khoảng 5-7 lần đường kính marker là vùng hợp lý."
  },
  {
    name: "clahe_clip_limit",
    min: 1,
    max: 5,
    step: 0.1,
    value: 2.5,
    unit: "",
    low: "Ít khuếch đại; marker mờ có thể không nổi.",
    high: "Tương phản mạnh hơn nhưng nhiễu nền dễ thành false blob.",
    sweet: "2.5 là mức vừa đủ cho marker yếu mà vẫn hạn chế artifact."
  },
  {
    name: "min_area",
    min: 5,
    max: 120,
    step: 5,
    value: 30,
    unit: "px2",
    low: "Bắt được marker nhỏ nhưng nhận thêm noise/artifact.",
    high: "Loại noise tốt hơn nhưng có thể bỏ marker mờ hoặc bị nén.",
    sweet: "Nên thấp hơn diện tích marker thật với margin an toàn."
  },
  {
    name: "fb_threshold",
    min: 0.5,
    max: 5,
    step: 0.1,
    value: 2,
    unit: "px",
    low: "Strict hơn, dễ loại marker hợp lệ khi motion blur nhẹ.",
    high: "Dễ giữ track trôi, ảnh hưởng force/slip phía sau.",
    sweet: "2px cân bằng giữa sub-pixel noise và drift lớn."
  },
  {
    name: "win_size",
    min: 7,
    max: 41,
    step: 2,
    value: 21,
    unit: "px",
    low: "Ít pixel để solve gradient, nhạy noise.",
    high: "Ổn định hơn nhưng có thể lẫn marker lân cận, vi phạm local motion.",
    sweet: "21x21 đủ chứa marker 10-15px và nền quanh marker."
  },
  {
    name: "arrow_scale",
    min: 1,
    max: 6,
    step: 0.5,
    value: 3,
    unit: "x",
    low: "Mũi tên sát displacement thật nhưng khó nhìn với motion nhỏ.",
    high: "Pattern rõ hơn nhưng dễ gây hiểu nhầm nếu không ghi chú scale.",
    sweet: "3x giúp quan sát nhanh mà vẫn còn trực quan."
  },
  {
    name: "poly_degree",
    min: 1,
    max: 4,
    step: 1,
    value: 2,
    unit: "",
    low: "Bậc thấp dễ underfit vì chỉ học tuyến tính hoặc ít interaction.",
    high: "Nhiều monomial hơn, fit mạnh hơn nhưng tăng overfit và cần nhiều dữ liệu.",
    sweet: "Degree 2 cho 55 terms với 9 feature, là cân bằng hiện tại."
  },
  {
    name: "hidden_head",
    min: 4,
    max: 64,
    step: 4,
    value: 16,
    unit: "",
    low: "Head nhỏ hơn nhanh và ít overfit hơn nhưng capacity thấp.",
    high: "Capacity cao hơn nhưng mất thêm interpretability và dễ overfit.",
    sweet: "16 hidden units giữ model khoảng 913 tham số với degree 2."
  },
  {
    name: "weight_decay",
    min: 0,
    max: 0.05,
    step: 0.005,
    value: 0.01,
    unit: "",
    low: "Regularization yếu, polynomial terms tương quan dễ overfit.",
    high: "Shrink trọng số mạnh hơn, có thể underfit nếu quá lớn.",
    sweet: "1e-2 giúp kiểm soát hệ số polynomial và bậc cao."
  },
  {
    name: "huber_delta",
    min: 0.1,
    max: 3,
    step: 0.1,
    value: 1,
    unit: "N",
    low: "MAE-like sớm hơn, robust hơn nhưng gradient ít mượt.",
    high: "MSE-like rộng hơn, học nghiêm lỗi vừa nhưng nhạy outlier hơn.",
    sweet: "1N phù hợp khi lỗi gauge/tracking lớn hơn mức này nên được giảm ảnh hưởng."
  },
  {
    name: "early_stop_patience",
    min: 5,
    max: 80,
    step: 5,
    value: 40,
    unit: "epoch",
    low: "Dừng sớm hơn, có thể chưa kịp hội tụ.",
    high: "Cho model thêm thời gian nhưng tốn train và dễ train quá lâu sau plateau.",
    sweet: "40 epoch không cải thiện val MAE là mức bảo thủ cho tối đa 300 epoch."
  },
  {
    name: "force_sync_tolerance_s",
    min: 0.01,
    max: 0.5,
    step: 0.01,
    value: 0.1,
    unit: "s",
    low: "Nhiều frame bị skip nếu force gauge có timestamp jitter.",
    high: "Dễ gán force sai thời điểm, tạo label noise khi lực thay đổi nhanh.",
    sweet: "0.1s là cửa sổ thực dụng nếu force log khoảng 20Hz và trial thay đổi lực chậm."
  },
  {
    name: "fm_hidden_per_point",
    min: 32,
    max: 256,
    step: 32,
    value: 128,
    unit: "",
    low: "Shared MLP nhỏ hơn, nhanh hơn nhưng dễ underfit pattern marker phức tạp.",
    high: "Nhiều capacity hơn nhưng tăng tham số và overfit nếu ít trial.",
    sweet: "4->64->128 cho khoảng 25K tham số tổng thể, hợp với dataset vừa."
  },
  {
    name: "fm_hidden_head",
    min: 16,
    max: 128,
    step: 16,
    value: 64,
    unit: "",
    low: "Head yếu hơn khi combine max+mean feature 256 chiều.",
    high: "Head mạnh hơn nhưng dễ overfit sau global pooling.",
    sweet: "64 là default cân bằng cho vector pooled 256 chiều."
  },
  {
    name: "fm_dropout",
    min: 0,
    max: 0.5,
    step: 0.05,
    value: 0.1,
    unit: "",
    low: "Regularization yếu, head có thể phụ thuộc quá mức vào vài chiều pooled feature.",
    high: "Regularization mạnh, có thể làm regression kém ổn định.",
    sweet: "0.1 ở head là nhẹ; không đặt dropout trong shared MLP để tránh nhiễu per-point feature."
  },
  {
    name: "fm_rotation_max_deg",
    min: 0,
    max: 20,
    step: 1,
    value: 5,
    unit: "deg",
    low: "Ít tăng diversity, không mô phỏng đủ sai số mount.",
    high: "Pattern hình học có thể không thực tế nếu cảm biến không xoay nhiều như vậy.",
    sweet: "5 độ tương ứng sai số cơ khí nhỏ và hợp lý cho augmentation."
  },
  {
    name: "fm_noise_std_px",
    min: 0,
    max: 2,
    step: 0.1,
    value: 0.5,
    unit: "px",
    low: "Model dễ overfit vào tracking quá sạch.",
    high: "Noise che mất tương quan lực-displacement thật.",
    sweet: "0.5px gần sai số LK nhỏ trong thực tế."
  },
  {
    name: "cnn_image_width",
    min: 160,
    max: 640,
    step: 80,
    value: 320,
    unit: "px",
    low: "Train/infer nhanh hơn nhưng mất chi tiết biến dạng nhỏ.",
    high: "Giữ nhiều chi tiết marker hơn nhưng tốn GPU memory và thời gian.",
    sweet: "320x240 là cân bằng hiện tại giữa chi tiết và chi phí."
  },
  {
    name: "cnn_hidden_head",
    min: 32,
    max: 512,
    step: 32,
    value: 128,
    unit: "",
    low: "Head quá nhỏ có thể thành cổ chai sau feature 512 chiều.",
    high: "Thêm capacity nhưng dễ overfit nếu dataset ít.",
    sweet: "128 đủ gọn vì ResNet backbone đã trích feature chính."
  },
  {
    name: "cnn_dropout",
    min: 0,
    max: 0.6,
    step: 0.05,
    value: 0.2,
    unit: "",
    low: "Fit nhanh hơn nhưng dễ overfit head.",
    high: "Regularization mạnh hơn nhưng có thể làm regression dao động.",
    sweet: "0.2 là regularization vừa phải cho head 512->128->1."
  },
  {
    name: "cnn_lr",
    min: 0.00005,
    max: 0.001,
    step: 0.00005,
    value: 0.0003,
    unit: "",
    low: "Fine-tune chậm, cần nhiều epoch hơn.",
    high: "Dễ phá pretrained representation hoặc làm val MAE dao động.",
    sweet: "3e-4 thấp hơn ForceNet vì đang fine-tune ResNet18 pretrained."
  },
  {
    name: "cnn_batch_size",
    min: 8,
    max: 64,
    step: 8,
    value: 32,
    unit: "",
    low: "Gradient noisy hơn nhưng ít tốn GPU memory.",
    high: "Gradient ổn định hơn nhưng dễ OOM với ResNet feature maps.",
    sweet: "32 là mức thực dụng cho input 2x240x320 và ResNet18."
  },
  {
    name: "cnn_brightness_jitter",
    min: 0,
    max: 0.4,
    step: 0.05,
    value: 0.1,
    unit: "",
    low: "Ít robust với drift ánh sáng.",
    high: "Dễ clamp pixel và làm mất gradient/texture marker.",
    sweet: "0.1 mô phỏng drift nhẹ mà không phá tín hiệu ảnh."
  },
  {
    name: "slip_threshold",
    min: 0.4,
    max: 0.98,
    step: 0.02,
    value: 0.8,
    unit: "",
    low: "Nhạy hơn, bắt slip sớm hơn nhưng dễ false positive.",
    high: "Bảo thủ hơn, giảm false positive nhưng có thể bỏ sót slip yếu.",
    sweet: "0.8 yêu cầu evidence đồng hướng cao trước khi báo slip."
  },
  {
    name: "slip_min_motion_thresh",
    min: 0.5,
    max: 5,
    step: 0.25,
    value: 2,
    unit: "px",
    low: "Bắt motion nhỏ hơn nhưng góc vector dễ nhiễu.",
    high: "Lọc nhiễu tốt hơn nhưng có thể bỏ slow slip.",
    sweet: "2px là ngưỡng hiện tại để bỏ rung/LK noise."
  },
  {
    name: "slip_min_moving_markers",
    min: 1,
    max: 15,
    step: 1,
    value: 5,
    unit: "",
    low: "Nhạy với ít marker nhưng dễ bị tracking lỗi cục bộ.",
    high: "Evidence chắc hơn nhưng bỏ sót khi valid marker ít.",
    sweet: "5 marker là yêu cầu tối thiểu để evidence đáng tin hơn."
  },
  {
    name: "slip_history_buffer",
    min: 2,
    max: 15,
    step: 1,
    value: 5,
    unit: "frame",
    low: "Phản ứng nhanh hơn nhưng displacement nhỏ và nhiễu hơn.",
    high: "Bắt slow slip tốt hơn nhưng tăng độ trễ.",
    sweet: "5 frame so current với past cách tối đa khoảng 4 frame."
  },
  {
    name: "slip_alpha",
    min: 0.05,
    max: 0.9,
    step: 0.05,
    value: 0.3,
    unit: "",
    low: "Smooth hơn, ít flicker nhưng phản ứng chậm.",
    high: "Phản ứng nhanh hơn nhưng score dễ dao động.",
    sweet: "0.3 là alpha nền, còn adaptive alpha tăng khi evidence mạnh."
  },
  {
    name: "slip_press_rate_threshold",
    min: 0.1,
    max: 2,
    step: 0.1,
    value: 0.5,
    unit: "px/frame",
    low: "Dễ coi motion là pressing và suppress slip yếu.",
    high: "Ít suppress pressing hơn nhưng false positive do nhấn có thể tăng.",
    sweet: "0.5px/frame nhận diện mean deformation tăng nhanh khi nhấn."
  }
];

const state = {
  module: modules[0].id,
  mode: "study",
  card: Number(localStorage.getItem("review.card") || 0),
  known: new Set(JSON.parse(localStorage.getItem("review.knownCards") || "[]")),
  quizIndex: 0,
  selectedOption: null,
  quizCorrect: 0,
  answered: false
};

const $ = (id) => document.getElementById(id);

function saveProgress() {
  localStorage.setItem("review.knownCards", JSON.stringify([...state.known]));
  localStorage.setItem("review.card", String(state.card));
  updateProgress();
}

function updateProgress() {
  const pct = Math.round((state.known.size / flashcards.length) * 100);
  $("progressText").textContent = `${pct}%`;
  $("progressBar").style.width = `${pct}%`;
}

function renderNavigation() {
  $("moduleNav").innerHTML = modules
    .map(
      (mod) => `
        <button class="module-btn ${mod.id === state.module ? "active" : ""}" data-module="${mod.id}" type="button">
          <span class="module-icon">${mod.icon}</span>
          <span><strong>${mod.title}</strong><span>${mod.short}</span></span>
        </button>
      `
    )
    .join("");

  document.querySelectorAll("[data-module]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.module = btn.dataset.module;
      renderNavigation();
      renderStudy();
    });
  });
}

function renderPipeline() {
  $("pipelineSteps").innerHTML = pipeline
    .map((step, index) => `<div class="step-chip"><strong>${index + 1}. ${step[0]}</strong><span>${step[1]}</span></div>`)
    .join("");
}

function activeModule() {
  return modules.find((mod) => mod.id === state.module) || modules[0];
}

function renderStudy() {
  const mod = activeModule();
  $("studyView").innerHTML = `
    <article class="card module-summary">
      <p class="eyebrow">${mod.file}</p>
      <h3>${mod.title}</h3>
      <p>${mod.thesis}</p>
      <ul class="key-list">
        ${mod.flow.map((item) => `<li><strong>${item}</strong><span>${flowHint(item)}</span></li>`).join("")}
      </ul>
    </article>
    <article class="card module-details">
      <p class="eyebrow">Ý chính cần nhớ</p>
      <h3>Luận điểm phòng thủ</h3>
      <ul class="key-list">
        ${mod.keys.map(([title, body]) => `<li><strong>${title}</strong><span>${body}</span></li>`).join("")}
      </ul>
    </article>
    <article class="card full-card">
      <p class="eyebrow">Tham số và bẫy phản biện</p>
      <h3>Những câu dễ bị hỏi sâu</h3>
      <ul class="param-list">
        ${mod.params.map(([name, value, body]) => `<li><strong><code>${name}</code> = ${value}</strong><span>${body}</span></li>`).join("")}
      </ul>
      <ul class="qa-list">
        ${mod.traps.map((trap) => `<li><strong>Lưu ý</strong><span>${trap}</span></li>`).join("")}
      </ul>
    </article>
  `;
}

function flowHint(item) {
  const hints = {
    "Box blur 101x101": "Ước lượng nền LED/vignetting",
    "Subtract + normalize": "Giữ tương phản marker-nền",
    "CLAHE clip 2.5, grid 8x8": "Tăng tương phản cục bộ",
    "Threshold sweep": "Quét 50..220, step 10",
    "Connected components": "Tìm region sáng",
    "Group centers": "Gom tâm ổn định qua threshold",
    "Geometry filters": "Area, circularity, inertia, convexity",
    "LK forward": "Reference sang deformed",
    "LK backward": "Deformed quay về reference",
    "FB error": "Khoảng cách điểm quay lại",
    "Valid mask": "Giữ marker đáng tin cậy",
    "Deadzone tùy chọn": "Snap motion nhỏ về 0",
    "disp = def - ref": "Vector dịch chuyển thật",
    magnitude: "Độ lớn displacement",
    "scale arrow": "Chỉ phóng đại hiển thị",
    "color by magnitude": "Xanh nhỏ, đỏ lớn",
    preprocess: "Ảnh ổn định hơn",
    "detect reference": "Lấy identity ban đầu",
    "track frame": "Theo frame biến dạng",
    validate: "FB check + mask",
    "consume downstream": "force, slip, visualization",
    "cache npz": "ref_pts, disp, valid, force",
    "compute 9 features": "Tóm tắt displacement field",
    "standardize train": "Dùng mean/std từ train set",
    "poly degree 2": "Sinh 55 monomial terms",
    "regression head": "linear hoặc mlp_small",
    "scale by image_w": "Giảm phụ thuộc độ phân giải",
    "mask valid markers": "Chỉ thống kê marker hợp lệ",
    "magnitude stats": "mean, max, std, sum_norm",
    "radial tangential": "Nén/giãn và xoắn/shear",
    "valid ratio": "Độ tin cậy tracking",
    "fit scaler train": "Tránh leakage val/test",
    "z-score features": "Đưa feature về cùng scale",
    "monomial indices": "combinations_with_replacement",
    "55 terms": "1 bias + 9 linear + 45 bậc 2",
    "linear or MLP": "Interpretability hoặc capacity",
    "trial-level split": "Tránh temporal leakage",
    "precompute RAM": "Feature nhỏ, train nhanh",
    "Huber loss": "Robust với outlier",
    "AdamW schedule": "Weight decay + cosine LR",
    "eval per trial": "Xem trial khó và metric tổng thể",
    "session reference": "Ảnh không tiếp xúc cho cả session",
    "nearest force sync": "Join frame-force bằng timestamp gần nhất",
    "LK no deadzone": "Giữ displacement nhỏ cho lực thấp",
    "zero offset": "Trừ baseline force gauge",
    "npz cache": "Marker-level data để train nhanh",
    "per marker 4D": "x/W, y/H, dx/W, dy/H",
    "pad to N_max": "Cùng shape cho batch",
    "valid mask": "Loại LK fail và padding",
    "train augment": "Flip, rotation, noise chỉ train",
    "trial split": "Không leak frame cùng trial",
    "Conv1d k1 shared": "Linear shared cho từng marker",
    "BatchNorm ReLU": "Ổn định activation trong MLP",
    "masked max pool": "Bắt marker đáp ứng mạnh nhất",
    "masked mean pool": "Bắt xu hướng toàn bề mặt",
    "head 256-64-1": "Global vector sang scalar force",
    "Huber delta 1": "Robust với outlier force/tracking",
    "Adam lr 1e-3": "Optimizer hiện tại của ForceNet",
    "wd 1e-4": "Regularization nhẹ",
    "early stop 10": "Dừng theo validation MAE",
    "trial breakdown": "Metric riêng từng trial",
    "index trial": "Metadata frame-force từ data/sessions",
    "sync force cnn": "Nearest-neighbor timestamp",
    "load ref frame": "Lazy-load ảnh reference và frame",
    "resize normalize": "240x320, pixel [0,1]",
    "stack 2 channels": "Tensor ref+frame",
    "paired flip": "Lật đồng bộ hai ảnh",
    "paired rotate": "Xoay cùng ma trận",
    "paired brightness": "Cùng brightness shift",
    "paired contrast": "Cùng contrast scale",
    "train only": "Không augment val/test",
    "input 2HW": "Batch ảnh 2 kênh",
    "resnet conv1 inflate": "RGB pretrained sang 2 kênh",
    "fc identity": "Bỏ classifier ImageNet",
    "feature 512": "Vector ResNet trước head",
    "head 512-128-1": "Regression scalar force",
    "Huber cnn": "Robust label/image outlier",
    "AdamW lr 3e-4": "Fine-tune pretrained CNN",
    "cosine lr": "Giảm LR mượt",
    "best val MAE": "Checkpoint theo Newton MAE",
    "infer image pair": "Ref+frame trực tiếp",
    "marker history": "Buffer tọa độ marker realtime",
    "past current": "So frame cũ với frame hiện tại",
    "valid motion": "Marker valid và đủ displacement",
    "weighted MRVL": "Độ đồng hướng có weight magnitude",
    "slip phase": "slip/tracking/pressing/no_markers",
    "valid filter": "Bỏ marker LK invalid",
    "press detect": "Mean deformation tăng nhanh",
    "motion threshold": "Bỏ vector nhỏ nhiễu",
    "rebound filter": "Loại hồi đàn hồi về reference",
    "radial removal": "Trừ thành phần press/release xuyên tâm",
    "angle weights": "atan2 và magnitude weights",
    "raw R": "MRVL trước boost/smooth",
    "score gains": "Translation, participation, motion",
    "pressing gate": "Suppress pressing không translation",
    "EMA smooth": "Giảm flicker slip score"
  };
  return hints[item] || "Mắt xích trong pipeline";
}

function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-tab").forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  ["studyView", "cardsView", "quizView", "tuningView", "defenseView"].forEach((id) => $(id).classList.add("hidden"));
  $(`${mode}View`).classList.remove("hidden");

  if (mode === "cards") renderCard();
  if (mode === "quiz") renderQuiz();
  if (mode === "tuning") renderTuning();
  if (mode === "defense") renderDefense();
}

function renderCard() {
  state.card = (state.card + flashcards.length) % flashcards.length;
  const [module, question, answer] = flashcards[state.card];
  $("cardCounter").textContent = `${state.card + 1} / ${flashcards.length}`;
  $("cardModule").textContent = module;
  $("cardQuestion").textContent = question;
  $("cardAnswer").textContent = answer;
  $("cardAnswer").classList.add("hidden");
  $("revealCardBtn").textContent = "Hiện đáp án";
}

function renderQuiz() {
  const item = quiz[state.quizIndex];
  state.selectedOption = null;
  state.answered = false;
  $("quizTitle").textContent = `Câu hỏi ${state.quizIndex + 1} / ${quiz.length}`;
  $("quizQuestion").textContent = item.q;
  $("quizFeedback").textContent = "";
  $("quizScore").textContent = `${state.quizCorrect} đúng`;
  $("quizOptions").innerHTML = item.options
    .map((option, index) => `<button class="option-btn" data-option="${index}" type="button">${option}</button>`)
    .join("");

  document.querySelectorAll("[data-option]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (state.answered) return;
      state.selectedOption = Number(btn.dataset.option);
      document.querySelectorAll("[data-option]").forEach((itemBtn) => itemBtn.classList.remove("selected"));
      btn.classList.add("selected");
    });
  });
}

function submitQuiz() {
  if (state.selectedOption === null || state.answered) return;
  const item = quiz[state.quizIndex];
  state.answered = true;
  const correct = state.selectedOption === item.answer;
  if (correct) state.quizCorrect += 1;

  document.querySelectorAll("[data-option]").forEach((btn) => {
    const idx = Number(btn.dataset.option);
    if (idx === item.answer) btn.classList.add("correct");
    if (idx === state.selectedOption && !correct) btn.classList.add("wrong");
  });

  $("quizFeedback").textContent = `${correct ? "Đúng." : "Sai."} ${item.why}`;
  $("quizScore").textContent = `${state.quizCorrect} đúng`;
}

function nextQuiz() {
  state.quizIndex = (state.quizIndex + 1) % quiz.length;
  if (state.quizIndex === 0) state.quizCorrect = 0;
  renderQuiz();
}

function renderTuning() {
  $("tuningView").innerHTML = tuning
    .map(
      (item, index) => `
        <article class="card tuning-card">
          <p class="eyebrow">Config</p>
          <h3>${item.name}</h3>
          <div class="slider-row">
            <label class="slider-label" for="tune-${index}">
              <span>Giá trị</span>
              <strong id="tune-value-${index}">${item.value}${item.unit}</strong>
            </label>
            <input id="tune-${index}" data-tune="${index}" type="range" min="${item.min}" max="${item.max}" step="${item.step}" value="${item.value}" />
          </div>
          <div class="effect-box" id="tune-effect-${index}"></div>
        </article>
      `
    )
    .join("");

  document.querySelectorAll("[data-tune]").forEach((input) => {
    input.addEventListener("input", () => updateTune(Number(input.dataset.tune), Number(input.value)));
    updateTune(Number(input.dataset.tune), Number(input.value));
  });
}

function updateTune(index, value) {
  const item = tuning[index];
  const midpoint = (item.min + item.max) / 2;
  const defaultDistance = Math.abs(value - item.value);
  const level = defaultDistance < (item.max - item.min) * 0.08 ? "Vùng mặc định" : value < midpoint ? "Thiên về thấp" : "Thiên về cao";
  $("tune-value-" + index).textContent = `${value}${item.unit}`;
  $("tune-effect-" + index).innerHTML = `
    <span class="risk-pill">${level}</span>
    <p><strong>Giảm:</strong> ${item.low}</p>
    <p><strong>Tăng:</strong> ${item.high}</p>
    <p><strong>Ghi nhớ:</strong> ${item.sweet}</p>
  `;
}

function renderDefense() {
  const keyword = $("defenseSearch").value.trim().toLowerCase();
  const rows = defense.filter(([q, a]) => `${q} ${a}`.toLowerCase().includes(keyword));
  $("defenseList").innerHTML = rows
    .map(
      ([q, a], index) => `
        <details class="qa-item" ${index === 0 ? "open" : ""}>
          <summary>${q}</summary>
          <p>${a}</p>
        </details>
      `
    )
    .join("");
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((btn) => btn.addEventListener("click", () => switchMode(btn.dataset.mode)));
  $("prevCardBtn").addEventListener("click", () => {
    state.card -= 1;
    saveProgress();
    renderCard();
  });
  $("nextCardBtn").addEventListener("click", () => {
    state.card += 1;
    saveProgress();
    renderCard();
  });
  $("flashcard").addEventListener("click", () => $("cardAnswer").classList.toggle("hidden"));
  $("revealCardBtn").addEventListener("click", () => {
    $("cardAnswer").classList.toggle("hidden");
    $("revealCardBtn").textContent = $("cardAnswer").classList.contains("hidden") ? "Hiện đáp án" : "Ẩn đáp án";
  });
  $("knownCardBtn").addEventListener("click", () => {
    state.known.add(state.card);
    state.card += 1;
    saveProgress();
    renderCard();
  });
  $("reviewCardBtn").addEventListener("click", () => {
    state.known.delete(state.card);
    saveProgress();
  });
  $("submitQuizBtn").addEventListener("click", submitQuiz);
  $("nextQuizBtn").addEventListener("click", nextQuiz);
  $("defenseSearch").addEventListener("input", renderDefense);
  $("resetProgressBtn").addEventListener("click", () => {
    state.known.clear();
    state.card = 0;
    state.quizIndex = 0;
    state.quizCorrect = 0;
    saveProgress();
    renderCard();
    renderQuiz();
  });
}

renderPipeline();
renderNavigation();
renderStudy();
renderCard();
renderQuiz();
renderTuning();
renderDefense();
bindEvents();
updateProgress();
