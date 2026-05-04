# Chương 1 — Giới thiệu

## 1.1 Bối cảnh

Khả năng cảm nhận xúc giác (tactile sensing) là một trong những giác quan
quan trọng nhất giúp con người thực hiện các thao tác cầm nắm tinh tế: khi
ngón tay tiếp xúc với một vật thể, hàng nghìn cơ quan thụ cảm dưới da đồng
thời truyền về não thông tin về **lực tiếp xúc**, **sự phân bố ứng suất**,
**rung động**, và đặc biệt là tín hiệu **trượt sớm (incipient slip)** — cho
phép con người vô thức siết chặt tay khi vật bắt đầu tuột. Trong robot học,
tái tạo được kênh cảm thụ này là điều kiện tiên quyết để tay máy chuyển từ
nhóm tác vụ pick-and-place đơn giản sang những thao tác đòi hỏi khéo léo:
cầm cốc nước có trọng lượng thay đổi, lắp ghép linh kiện cơ khí, sử dụng dụng
cụ y tế, hoặc tương tác an toàn với con người.

Suốt hai thập kỷ qua, hàng loạt công nghệ cảm biến xúc giác đã được phát
triển: cảm biến áp điện (piezoelectric), cảm biến điện dung (capacitive), cảm
biến từ trường (magnetic), cảm biến quang học, và gần đây là cảm biến dựa
trên camera (vision-based / camera-based tactile sensors). Trong nhóm cuối
cùng, một mảng cảm biến nhỏ gồm camera + lớp gel đàn hồi đã trở thành chuẩn
mực mới nhờ ba ưu thế: (i) **mật độ thông tin cao** — mỗi frame chứa hàng
trăm ngàn pixel ⇒ độ phân giải không gian vượt trội so với mảng tactel rời
rạc; (ii) **chi phí thấp** — chỉ cần một camera ~10 USD và một khối silicone
đúc khuôn; (iii) **tận dụng thuật toán thị giác máy tính** đã chín muồi để
chuyển ảnh thô thành các đại lượng vật lý có ý nghĩa (lực, biến dạng, hình
học bề mặt vật).

Mảng cảm biến mở đầu là họ **GelSight** [Yuan và cộng sự, 2017], trong đó
lớp gel được phủ một lớp pigment phản xạ Lambertian và chiếu sáng bằng LED
RGB nhiều phương; thuật toán photometric stereo cho phép tái tạo hình học
3D bề mặt vật. Để đo trường lực + slip, người ta in/đúc thêm một mảng marker
(chấm đen hoặc trắng) trên mặt gel — biến dạng trong mặt phẳng được suy ra
trực tiếp từ độ dịch chuyển của các marker này. Hướng nghiên cứu *marker
motion* tỏ ra đặc biệt hiệu quả cho bài toán phát hiện trượt vì **trường
vector dịch chuyển** mang ngay đặc trưng vật lý: trượt thuần ⇒ vector đồng
nhất theo một hướng; nén / nhả ⇒ vector xuyên tâm; bám dính ⇒ vector gần
không. Đây cũng là cơ sở lý thuyết của hệ thống được trình bày trong luận
văn này.

## 1.2 Các nghiên cứu liên quan

### 1.2.1 Họ cảm biến xúc giác kiểu camera

**GelSight (MIT, 2009 – nay).** Là cảm biến tiên phong sử dụng lớp gel phủ
pigment + LED màu để đo hình học bề mặt. Phiên bản cải tiến năm 2017 [Dong
và cộng sự, ICRA 2017] thêm marker đen vào gel và chứng minh khả năng đo
*shear force* và phát hiện slip.

**GelSlim (MIT, 2018 – 2022).** Tái thiết kế hoàn toàn đường quang học để
dồn cảm biến vào dạng *finger* mỏng gắn được lên đầu kẹp robot. GelSlim 3.0
[Taylor và cộng sự, 2022] tích hợp phép đo hình dạng + lực + slip trên một
ngón cảm biến duy nhất, mở đường cho ứng dụng *in-hand manipulation*.

**DIGIT (Meta AI, 2020).** [Lambeta và cộng sự, RAL 2020] giới thiệu thiết
kế *low-cost, compact, high-resolution* — kích thước chỉ 20 × 27 × 18 mm,
khối lượng ≈ 20 g, và đặc biệt là toàn bộ thiết kế cơ khí + CAD + firmware
được mở mã nguồn. DIGIT đã trở thành de-facto standard cho cộng đồng nghiên
cứu khi cần một cảm biến chuẩn để so sánh thuật toán.

**TacTip (Bristol, 2009 – nay).** [Lepora và cộng sự] thay vì photometric
stereo + gel phẳng, TacTip đúc các *gai mềm sinh học* trên mặt gel, đầu
gai sơn marker. Khi tiếp xúc, gai bị uốn cong; chuyển động marker ở đầu
gai tương đương đáp ứng của các thụ cảm cơ học dưới da. Đây là một dòng
*marker-only* — không cần photometric stereo.

**DigiTac (Bristol – Meta, 2022).** Lai DIGIT + TacTip: giữ phần điện tử
và quang học của DIGIT nhưng thay nắp gel bằng đầu TacTip 3D-print. Cho
phép so sánh trực tiếp hai trường phái *image-based* và *marker-based*
trên cùng một nền cứng.

**9DTact (Tsinghua, 2024).** [Lin và cộng sự] một thiết kế **compact**
(32.5 × 25.5 × 25.5 mm), giá rẻ, in 3D dễ dàng, đo được đồng thời *9-DoF*:
3D shape + 6D force/torque. Toàn bộ phần cứng và phần mềm đều mở.

**CrystalTac (2024).** Đề xuất quy trình *rapid monolithic manufacturing*
— đúc toàn bộ cảm biến trong một lượt in 3D đa vật liệu. Đơn giản hoá quy
trình lắp ráp, tăng độ đồng nhất giữa các bản sao, phù hợp với production
scale.

**Minsight (Max-Planck, 2023).** Cảm biến haptic cỡ ngón tay, mở mã nguồn,
tập trung vào ước lượng force field độ phân giải cao bằng deep learning.

> Xu hướng chung của ngành: **gọn hơn, rẻ hơn, mở mã nguồn hơn** — và tiêu
> chí *thẩm mỹ* (tích hợp được vào sản phẩm dạng tay người máy) bắt đầu
> được quan tâm song song với tiêu chí kỹ thuật.

### 1.2.2 Phát hiện trượt (slip detection)

Có ba dòng tiếp cận chính trong tài liệu:

**(a) Phương pháp dựa trên rung động / IMU.** Khi vật bắt đầu trượt, ma
sát stick-slip sinh ra rung động tần số cao ở vùng tiếp xúc. Đặt
microphone hoặc IMU lên ngón gắp + bộ lọc thông cao + threshold = phát
hiện slip. Ưu điểm: phản ứng cực nhanh (vài ms). Nhược điểm: nhạy với
nhiễu cơ học, không phân biệt được hướng trượt, không thấy *incipient
slip* — slip đã hoàn toàn xảy ra mới phát hiện được [Howe & Cutkosky 1989,
James và cộng sự RAL 2018].

**(b) Phương pháp dựa trên trường lực.** Cảm biến đo lực 3D đặt ở đầu kẹp;
khi $|F_t|/(\mu F_n)$ vượt 1, khối tiếp xúc đang trượt theo định luật
Coulomb. Yêu cầu calibrate hệ số ma sát $\mu$ — mà $\mu$ phụ thuộc cả vật
liệu vật lẫn trạng thái bề mặt gel ⇒ khó tổng quát hoá.

**(c) Phương pháp dựa trên marker motion (camera-based).** Đây là dòng
được sử dụng trong luận văn này. Ý tưởng: khi vật trượt, các marker nằm
trong vùng bám sẽ có vector dịch chuyển *gần như đồng nhất* theo hướng
trượt; các marker rìa có thể vẫn dính. Bằng cách phân tích **độ đồng nhất
hướng** (ví dụ entropy, mean resultant length, radial-tangential
decomposition), ta phát hiện được cả *incipient slip*.

Một số mốc nghiên cứu đại diện:

- **Yuan và cộng sự, ICRA 2015** — đo shear và slip với GelSight bằng
  marker đen, dùng PCA trên vector dịch chuyển để phát hiện hướng trượt.
- **Dong và cộng sự, IROS 2017** — cải tiến GelSight cho slip detection,
  mở đầu hướng "marker dot tracking" + threshold.
- **Yuan và cộng sự, 2017** — survey "GelSight: high-resolution tactile
  sensors for estimating geometry and force".
- **Li và cộng sự, 2018** — Slip detection with combined tactile and
  visual information, kết hợp camera ngoài + GelSight.
- **Zhang và cộng sự, 2023 (arXiv 2303.00935)** — phát hiện slip bằng
  *entropy* trường lực tiếp xúc (Shannon entropy của trường marker
  displacement) — mô hình deep learning đạt > 90% accuracy.
- **Khả năng học có giám sát** [Roberge và cộng sự, IEEE 2024] — *Robust
  learning-based incipient slip detection* trên cảm biến PapillArray với
  CNN-LSTM hybrid; cho thấy DL có thể vượt baseline thủ công nếu có đủ
  data ground truth slip.
- **Xu và cộng sự, 2025** — *Slip detection and stable grasping with
  multi-fingered robotic hand using deep learning*, dùng CNN-LSTM kết
  hợp tactile và force feedback.
- **Chen và cộng sự, 2025** — *Universal slip detection of robotic hand
  with tactile sensing*, mạng CNN nhẹ phân loại 3 trạng thái (no-touch /
  slipping / stable grasp), độ chính xác > 97%.
- **Weiß và cộng sự, npj Robotics 2025** — protocol "DENSE" để thu
  data slip dày đặc và đa dạng cho training.
- **Cui và cộng sự, arXiv 2411.07442 (2024)** — *Learned slip-detection
  severity framework using tactile deformation field feedback*; dùng
  GelSight Mini với 63 marker đen, đề xuất hai mạng song song: một cho
  phát hiện slip, một cho **đánh giá độ nghiêm trọng** (slip velocity).

Ngoài ra, các phương pháp **không học** vẫn được ưa chuộng vì khả năng
diễn giải và không cần training data: phương pháp dùng *Mean Resultant
Vector Length* (thống kê vòng tròn) để đo độ đồng nhất hướng [Fisher
1995], phương pháp *radial-translation decomposition* (phân rã trường
biến dạng thành thành phần xuyên tâm + tịnh tiến) — chính là hai detector
được hiện thực hoá trong luận văn này.

### 1.2.3 Ước lượng lực tiếp xúc

Lực tiếp xúc có thể là *scalar* (lực pháp tuyến) hoặc *vector* (3-DoF: 1
pháp + 2 tiếp tuyến) hoặc *trường* (force distribution map). Có ba lớp
phương pháp:

**(a) Phương pháp giải tích — đàn hồi tuyến tính.** Ở biến dạng nhỏ, gel
silicone tuân theo định luật Hooke; trường biến dạng ở mặt cảm biến tỉ lệ
tuyến tính với phân bố ứng suất tại điểm tiếp xúc. Inverse FEM cho phép
giải ra force distribution từ đo đạc displacement. Phương pháp này yêu
cầu calibration vật liệu kỹ + có thể chậm khi giải nghịch FEM realtime
[Ma và cộng sự, ICRA 2019 — *Dense Tactile Force Estimation using
GelSlim and Inverse FEM*].

**(b) Feature-engineered shallow regression.** Trích xuất một vài đặc
trưng vô hướng (lực bình phương trung bình của marker, độ phủ valid,
xuyên tâm trung bình, tangent trung bình, …) sau đó hồi quy bằng
linear / polynomial / random forest. Ưu thế: *diễn giải được*, training
nhanh, ít data; nhưng khả năng tổng quát hoá phụ thuộc vào chất lượng
feature thủ công.

**(c) Deep learning end-to-end.** Mạng CNN nhận trực tiếp ảnh tactile
(hoặc difference image so với reference) và xuất scalar / vector lực.
Một số kiến trúc tiêu biểu:
- **VGG-16 / GoogLeNet/ ResNet biến đổi**: thay FC cuối thành 1–4 neuron
  ứng với (Fx, Fy, Fz, Tz), dùng MSE loss.
- **U-Net** [Funk và cộng sự, arXiv 2411.03315 (2024)] — học mapping ảnh
  → trường lực 2D, dùng FEA làm ground truth.
- **PointNet-style** [Qi và cộng sự, CVPR 2017] — coi mảng marker là
  point cloud không có thứ tự, dùng shared MLP + symmetric pooling. Phù
  hợp khi số marker biến đổi và cần permutation invariance.
- **FeelAnyForce** [Sun và cộng sự] — phương pháp ước lượng lực không
  phụ thuộc hình dạng vật tiếp xúc.
- **3D force identification** [Wang và cộng sự, Mechatronics 2024] —
  CNN với GelSight-structured sensor cho 3D force.

Survey của *Fang và cộng sự, Advanced Intelligent Systems 2025 — Force
Measurement Technology of Vision-Based Tactile Sensor* và bài tổng quan
*Classification of Vision-Based Tactile Sensors* (arXiv 2509.02478, 2025)
hệ thống hoá bốn lớp phương pháp: marker displacement, RGB variations,
indentation area, optical flow vectors.

### 1.2.4 Khoảng trống

Tổng kết tài liệu cho thấy bốn khoảng trống:

1. **Phần lớn cảm biến mở mã nguồn vẫn ở dạng prototype hộp vuông**, gắn
   được lên kẹp robot nhưng chưa hài hoà về thẩm mỹ với một bàn tay
   robot dạng người (humanoid hand) — yếu tố quan trọng nếu muốn ứng
   dụng vào sản phẩm tiêu dùng hoặc robot dịch vụ.
2. **Slip detector dạng deep learning** đạt độ chính xác cao nhưng cần
   nhiều data có nhãn slip — khó thu, và mô hình khó *diễn giải* khi
   triển khai trên robot thực tế.
3. **Slip detector dạng analytic một thang thời gian** (như MRVL trên
   velocity) không phát hiện được *slow slip* tích luỹ và bị che bởi
   tín hiệu nhấn cùng pha.
4. **Phương pháp ước lượng lực** thường được công bố riêng lẻ — chưa
   có nghiên cứu nào *so sánh fairly* trên cùng dataset giữa shallow
   poly regression, point-cloud network, và CNN end-to-end.

Luận văn này được thiết kế để lấp đầy **bốn khoảng trống** đó.

## 1.3 Động lực

Động lực của đề tài đến từ ba quan sát thực tiễn và một mục tiêu kỹ thuật:

1. **Sự dịch chuyển từ nghiên cứu sang sản phẩm**: cộng đồng tactile
   sensing đang đi từ "demo trên bàn lab" sang "tích hợp vào tay máy
   thật". Bước chuyển này đòi hỏi cảm biến *gọn, nhẹ, thẩm mỹ, lắp đặt
   được vào ngón tay robot*. Các thiết kế hiện tại (DIGIT, GelSlim) là
   bước đầu nhưng vẫn chưa tối ưu cho tay máy dạng người: form factor
   vuông, chiều cao đầu cảm biến lớn, dây cáp lộ ra ngoài.

2. **Vai trò của slip detection trong điều khiển grasp realtime**: Nếu
   slip detector phản ứng đúng trong < 50 ms, robot có thể tự động siết
   thêm trước khi vật rơi — làm tăng độ thành công của thao tác trên
   các vật khó (vật dầu mỡ, vật nhẹ và trơn, vật đang rung). Điều này
   đặt yêu cầu cho slip detector *vừa nhanh vừa diễn giải được* — nghĩa
   là không thể chỉ dựa vào một mạng deep học từ data, mà cần thuật
   toán có nguyên lý vật lý rõ ràng để debug được trên robot.

3. **Khả năng tái sử dụng dữ liệu cho nhiều bài toán**: Một dataset
   tactile được thu công phu (đồng bộ camera + force gauge + motor) có
   thể được dùng để (a) đánh giá slip detector, (b) huấn luyện
   regression lực, (c) làm benchmark cho nghiên cứu sau. Vì thế hệ
   thống thu thập cần thiết kế *theo chuẩn* (session/trial, metadata
   đầy đủ, repro snapshot) ngay từ đầu — không thể coi là thứ yếu.

4. **Mục tiêu giáo dục**: Cuối cùng, đề tài còn có ý nghĩa đào tạo —
   việc tự thiết kế cảm biến từ phần cứng đến thuật toán đến mô hình
   học máy cho phép sinh viên *làm chủ toàn bộ stack*, một năng lực
   then chốt khi gia nhập thị trường robot.

## 1.4 Mục tiêu của luận văn

Luận văn đặt ra **bốn mục tiêu** chính, sắp xếp theo chiều từ phần cứng
ra thuật toán:

### Mục tiêu 1 — Hoàn thiện thiết kế cảm biến gọn nhẹ, thẩm mỹ

- Thu nhỏ kích thước tổng thể của cảm biến, đảm bảo ráp được vào ngón
  tay của tay máy dạng người mà không phá vỡ hình thái.
- Thiết kế lại đường quang học (LED + camera + lớp gel) để giảm chiều
  cao và che kín toàn bộ điện tử, tránh "ruột gan" lộ ra ngoài.
- In 3D vỏ + khuôn đúc gel theo quy trình **monolithic** (theo tinh
  thần CrystalTac) để tái lập nhanh và đồng nhất giữa các bản sao.
- Tiêu chí đánh giá: kích thước < 30 mm × 30 mm × 30 mm, khối lượng
  < 25 g, chỉ một dây cáp duy nhất ra ngoài.

### Mục tiêu 2 — Phát hiện trượt qua chuyển động marker

Cài đặt và so sánh hai phương pháp **không học** dựa trên marker motion,
đảm bảo *diễn giải được* và *triển khai realtime* trên CPU:

- **V1 — Mean Resultant Vector Length**: thống kê vòng tròn có trọng số
  trên trường vận tốc marker, kèm bộ lọc *press-rate* và *rebound*, EMA
  bất đối xứng. Dùng làm baseline diễn giải.
- **V2 — Phân rã đa thang Translation/Radial**: phân tách trường biến
  dạng tích lũy thành thành phần tịnh tiến (slip) + xuyên tâm
  (press/release) bằng least-squares trên nhiều thang thời gian. Khắc
  phục hai hạn chế chính của V1: không bắt được slip chậm và bị che
  bởi press cùng pha.
- Chứng minh trên thực nghiệm rằng V2 phát hiện được slip ở vận tốc
  thấp (< 1 px/frame) ngay cả khi đang nhấn xuống — điều mà V1 không
  làm được.

### Mục tiêu 3 — Ước lượng lực bằng nhiều phương pháp

Thay vì chỉ chọn một mô hình, luận văn cài đặt **ba kiến trúc** trên cùng
dataset và so sánh fair:

- **PolynomialRegressor** (`force_poly`): 9 đặc trưng vô hướng từ
  trường biến dạng (mean, max, std, radial, tangential, …) → polynomial
  expansion bậc 2 → linear head. ~55 tham số. Vai trò: baseline diễn
  giải được; hệ số mỗi monomial mang ý nghĩa vật lý (tuyến tính / nén
  bậc cao).
- **ForceNet PointNet-style** (`force_model`): coi mảng marker là
  point set không có thứ tự, shared MLP per-point + masked max+mean
  pool → head MLP. ~25K tham số. Vai trò: kiểm chứng hiệu quả của
  permutation-invariance và pooling.
- **ForceCNN end-to-end** (`force_cnn`): nhận 2 channel ảnh
  $[I_{\text{ref}}, I_{\text{def}}]$, backbone ResNet-18 (pretrained
  ImageNet, conv1 inflate 3→2 channel) hoặc SmallCNN. ~11 M / 110 K
  tham số. Vai trò: *upper bound* khi data đủ phong phú, không cần
  detect/track.
- Mục tiêu so sánh: MAE, RMSE, R², thời gian inference, dung lượng
  checkpoint. Trả lời câu hỏi: *"khi nào feature engineering đủ tốt,
  khi nào cần học từ pixel?"*

### Mục tiêu 4 — Hệ thống thu thập dữ liệu chuẩn hoá

Để hỗ trợ Mục tiêu 2 + 3 và phục vụ nghiên cứu tương lai, xây dựng một
**trạm thu thập dữ liệu** thực nghiệm:

- Đồng bộ ba nguồn dữ liệu: camera tactile, force gauge Imada ZTA, motor
  tuyến tính do Arduino điều khiển.
- Kiến trúc *producer-consumer* đa luồng: 3 thread sensor độc lập + 1
  thread writer; mỗi thread gắn timestamp `monotonic` ngay khi sample
  đến, không *join* ở write-time.
- Tổ chức dữ liệu theo schema *session/trial* + metadata
  (`session.yaml`, `trial.yaml`) đính kèm git SHA, firmware SHA-256, pip
  versions để đảm bảo *reproducibility*.
- Tự động lấy *force zero offset* (median 20 sample lúc tạo session).
- Tỉ lệ drop frame < 1% ở 30 FPS.

## 1.5 Đóng góp của luận văn

Sau khi hoàn thành, luận văn dự kiến mang lại bốn đóng góp cụ thể:

1. **Một thiết kế phần cứng cảm biến tactile mới** — gọn, thẩm mỹ, in 3D
   theo quy trình monolithic, kèm CAD và BOM mở mã nguồn.
2. **Slip detector V2** — đề xuất phương pháp *Translation/Radial
   decomposition đa thang* mới, vượt trội so với MRVL truyền thống ở khả
   năng phát hiện slow slip + tách biệt slip / press. Phân tích lý
   thuyết (least-squares decoupling khi centroid được lấy theo ref) +
   thực nghiệm.
3. **So sánh thống nhất ba phương pháp ước lượng lực** trên cùng dataset
   tự thu — tài liệu cho cộng đồng *tradeoff giữa diễn giải và độ
   chính xác* trong tactile force regression.
4. **Một bộ pipeline phần mềm mở mã nguồn**, đầy đủ các module: hiệu
   chuẩn camera, tiền xử lý, detection, tracking, slip V1/V2, training
   3 mô hình lực, và trạm thu thập dữ liệu — tất cả tham số tách biệt
   trong YAML, dễ tái sử dụng cho nghiên cứu tương lai.

## 1.6 Phạm vi và giới hạn

Luận văn tập trung vào *cảm biến marker-based* với marker sáng trên nền
tối; không đề cập đến photometric stereo cho 3D shape reconstruction
(đã được nghiên cứu kỹ ở GelSight và biến thể). Lực ước lượng giới hạn
ở scalar (lực pháp tuyến); mở rộng sang vector 3-DoF hoặc trường lực 2D
yêu cầu ground truth 6-DoF F/T sensor — nằm ngoài phạm vi đề tài. Slip
detector chỉ chạy trên CPU thông thường, không yêu cầu GPU realtime —
phù hợp cho robot có hardware hạn chế nhưng không tối ưu cho tốc độ
hơn 60 FPS.

## 1.7 Cấu trúc luận văn

Phần còn lại của luận văn được tổ chức như sau:

- **Chương 2 — Cơ sở lý thuyết.** Trình bày toàn bộ lý thuyết các thuật
  toán: hiệu chuẩn camera (Zhang), tiền xử lý (background subtraction,
  CLAHE), phát hiện blob, theo dõi Lucas-Kanade, hai detector slip,
  ba mô hình ước lượng lực. Tham chiếu chéo tới `THEORY.md`.
- **Chương 3 — Thiết kế phần cứng và hệ thống thu thập dữ liệu.** Mô
  tả quá trình thiết kế mới cảm biến, đường quang học, in 3D, lắp ráp;
  kèm thiết kế trạm thu thập đồng bộ và schema dataset.
- **Chương 4 — Cài đặt thuật toán.** Mô tả triển khai chi tiết các
  pipeline batch / realtime, cấu trúc code, cấu hình YAML, profiling
  hiệu năng.
- **Chương 5 — Thực nghiệm và đánh giá.** Trình bày các kịch bản đánh
  giá: (i) chất lượng detection + tracking trên cặp ảnh tĩnh, (ii) độ
  chính xác và độ trễ slip detector V1 vs V2, (iii) sai số ba mô hình
  ước lượng lực, (iv) đánh giá thẩm mỹ + độ gọn của thiết kế phần cứng.
- **Chương 6 — Kết luận và hướng phát triển.** Tóm tắt đóng góp, thảo
  luận hạn chế (đàn hồi tuyến tính, perspective projection, drift
  nhiệt), và đề xuất hướng mở rộng: thay LK bằng RAFT cho dense flow,
  sang vector lực 3-DoF, kết hợp slip detector phân tích với
  CNN-LSTM…

---

## Tài liệu tham khảo (chương 1)

### Cảm biến xúc giác kiểu camera

1. W. Yuan, S. Dong, E. H. Adelson. *GelSight: High-Resolution Robot
   Tactile Sensors for Estimating Geometry and Force*. Sensors 17(12),
   2017. <https://www.mdpi.com/1424-8220/17/12/2762>
2. M. Lambeta và cộng sự. *DIGIT: A Novel Design for a Low-Cost Compact
   High-Resolution Tactile Sensor with Application to In-Hand
   Manipulation*. IEEE RAL 2020.
   <https://www.semanticscholar.org/paper/DIGIT:-A-Novel-Design-for-a-Low-Cost-Compact-Sensor-Lambeta-Chou/643cc2cf95dda915c78eed8e263e45eed066bb86>
3. I. Taylor, S. Dong, A. Rodriguez. *GelSlim 3.0: High-Resolution
   Measurement of Shape, Force and Slip in a Compact Tactile-Sensing
   Finger*. ICRA 2022.
   <https://scite.ai/reports/gelslim-3-0-high-resolution-measurement-of-wmGlr6Mz>
4. D. F. Gomes và cộng sự. *DigiTac: A DIGIT-TacTip Hybrid Tactile
   Sensor for Comparing Low-Cost High-Resolution Robot Touch*. RAL 2022.
   <https://www.researchgate.net/publication/361605849_DigiTac_A_DIGIT-TacTip_Hybrid_Tactile_Sensor_for_Comparing_Low-Cost_High-Resolution_Robot_Touch>
5. C. Lin và cộng sự. *9DTact: A Compact Vision-Based Tactile Sensor
   for Accurate 3D Shape Reconstruction and Generalizable 6D Force
   Estimation*. arXiv 2308.14277, 2024.
   <https://arxiv.org/html/2308.14277v2>
6. *CrystalTac: Vision-Based Tactile Sensor Family Fabricated via Rapid
   Monolithic Manufacturing*. PMC, 2024.
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC11982672/>
7. Awesome-Touch — repo tổng hợp mã nguồn mở cảm biến xúc giác.
   <https://github.com/linchangyi1/Awesome-Touch>
8. *Vision-based tactile sensor design using physically based
   rendering*. Communications Engineering, 2025.
   <https://www.nature.com/articles/s44172-025-00350-4>
9. Minsight — vision-based haptic sensor (Max-Planck).
   <https://github.com/martius-lab/Minsight-sensor>
10. *Implementing Monocular Visual-Tactile Sensors for Robust
    Manipulation*. PMC.
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC9494691/>

### Phát hiện trượt

11. R. D. Howe, M. R. Cutkosky. *Sensing skin acceleration for slip and
    texture perception*. ICRA 1989.
12. W. Yuan và cộng sự. *Measurement of Shear and Slip with a GelSight
    Tactile Sensor*. ICRA 2015.
    <https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf>
13. S. Dong và cộng sự. *Improved GelSight Tactile Sensor for Measuring
    Geometry and Slip*. IROS 2017.
    <https://ieeexplore.ieee.org/document/8202149/>
14. J. Li và cộng sự. *Slip Detection with Combined Tactile and Visual
    Information*. ICRA 2018. <https://arxiv.org/pdf/1802.10153>
15. *Learning to Detect Slip through Tactile Estimation of the Contact
    Force Field and its Entropy*. arXiv 2303.00935.
    <https://arxiv.org/html/2303.00935v4>
16. *Robust Learning-Based Incipient Slip Detection Using the
    PapillArray Optical Tactile Sensor*. IEEE 2024.
    <https://ieeexplore.ieee.org/document/10374211/>
17. *Universal slip detection of robotic hand with tactile sensing*.
    Frontiers Neurorobotics 2025.
    <https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1478758/full>
18. *Slip Detection and Stable Grasping With Multi-Fingered Robotic
    Hand Using Deep Learning Approach*. IET Cyber-Systems & Robotics
    2025. <https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/csy2.70036>
19. *Let's DENSE: a novel protocol for efficiently collecting dense and
    diverse data for tactile slip detection*. npj Robotics 2025.
    <https://www.nature.com/articles/s44182-025-00055-y>
20. *Slip detection for compliant robotic hands using inertial signals
    and deep learning*. Frontiers Robotics & AI 2025.
    <https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1698591/full>
21. *Learned Slip-Detection-Severity Framework using Tactile
    Deformation Field Feedback for Robotic Manipulation*. arXiv
    2411.07442, 2024. <https://arxiv.org/html/2411.07442v1>

### Ước lượng lực từ cảm biến tactile

22. *Learning Force Distribution Estimation for the GelSight Mini
    Optical Tactile Sensor Based on Finite Element Analysis*. arXiv
    2411.03315, 2024. <https://arxiv.org/html/2411.03315v1>
23. *Object Recognition and Force Estimation with the GelSight Baby
    Fin Ray*. arXiv 2509.14510, 2025.
    <https://arxiv.org/html/2509.14510>
24. *Estimating Contact Force Feedback from Tactile Sensation —
    FeelAnyForce*. UMD PRG.
    <http://prg.cs.umd.edu/research/FeelAnyForce_files/FeelAnyForce.pdf>
25. *3D force identification and prediction using deep learning based
    on a GelSight-structured sensor*. Mechatronics 2024.
    <https://www.sciencedirect.com/science/article/abs/pii/S0924424724000293>
26. *Force Measurement Technology of Vision-Based Tactile Sensor*.
    Advanced Intelligent Systems 2025.
    <https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400290>
27. *Classification of Vision-Based Tactile Sensors: A Review*. arXiv
    2509.02478, 2025. <https://arxiv.org/pdf/2509.02478>
28. *Marker or Markerless? Mode-Switchable Optical Tactile Sensing for
    Diverse Robot Tasks*. arXiv 2408.08276, 2024.
    <https://arxiv.org/html/2408.08276v1>
29. *Dense Tactile Force Estimation using GelSlim and inverse FEM*.
    MIT, 2020. <https://dspace.mit.edu/bitstream/handle/1721.1/130473/1810.04621.pdf>
30. C. R. Qi, H. Su, K. Mo, L. J. Guibas. *PointNet: Deep Learning on
    Point Sets for 3D Classification and Segmentation*. CVPR 2017.

### Cơ sở thuật toán

31. B. D. Lucas, T. Kanade. *An iterative image registration technique
    with an application to stereo vision*. IJCAI 1981.
32. J.-Y. Bouguet. *Pyramidal implementation of the Lucas–Kanade feature
    tracker*. Intel Corp., 2001.
33. Z. Kalal, K. Mikolajczyk, J. Matas. *Forward-Backward Error:
    Automatic Detection of Tracking Failures*. ICPR 2010.
34. K. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization*.
    Graphics Gems IV, 1994.
35. Z. Zhang. *A flexible new technique for camera calibration*. IEEE
    TPAMI 22(11), 2000.
36. N. I. Fisher. *Statistical Analysis of Circular Data*. Cambridge
    University Press, 1995.
37. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image
    Recognition*. CVPR 2016.
38. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization*.
    ICLR 2019.
