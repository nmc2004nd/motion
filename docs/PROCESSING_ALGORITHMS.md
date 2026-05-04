# Chương 2 — Cơ sở lý thuyết các thuật toán xử lý

> Tài liệu này trình bày **chi tiết toán học** và **mô hình lý thuyết** cho toàn
> bộ chuỗi thuật toán xử lý của hệ thống cảm biến xúc giác kiểu marker quang
> học. Năm phần chính, mỗi phần đi từ *động cơ vật lý → mô hình toán học →
> công thức nghiệm → tham số hiện thực → giới hạn / tinh chỉnh*. Tham chiếu
> file:line trỏ về vị trí cài đặt trong `src/` để dễ truy ngược.
>
> **Mục lục:**
>
> 1. [Mô hình toán học của cảm biến (mô hình quang học)](#1-mô-hình-toán-học-của-cảm-biến-mô-hình-quang-học)
> 2. [Cơ sở lý thuyết thuật toán theo dõi (tracking)](#2-cơ-sở-lý-thuyết-thuật-toán-theo-dõi-tracking)
> 3. [Phát hiện trượt (slip detection)](#3-phát-hiện-trượt-slip-detection)
> 4. [Thiết kế hệ thống thu thập dữ liệu](#4-thiết-kế-hệ-thống-thu-thập-dữ-liệu)
> 5. [Thuật toán ước lượng lực tiếp xúc](#5-thuật-toán-ước-lượng-lực-tiếp-xúc)

---

## 1. Mô hình toán học của cảm biến (mô hình quang học)

Cảm biến xúc giác camera-based bao gồm bốn thành phần vật lý: **lớp gel đàn
hồi** (silicone trong/bán-trong suốt), **mảng marker** in trên mặt gel, **hệ
chiếu sáng LED** ven cảm biến, và **camera** đặt ở phía sau gel quan sát mặt
trong. Khi vật ngoài tiếp xúc gel, gel biến dạng → marker dịch chuyển trong
mặt phẳng ảnh → camera ghi lại trường biến dạng. Toàn bộ pipeline xây dựng
trên trường dịch chuyển 2D này.

Phần này thiết lập mô hình toán cho quan hệ **biến dạng vật lý 3D ↔ toạ độ
pixel**, giải thích vì sao có thể làm việc trực tiếp trên pixel mà không cần
quy đổi sang đơn vị mét, và cuối cùng là quy trình hiệu chuẩn camera (Zhang
2000) hiện thực ở `src/core/calibration.py`.

### 1.1 Hệ trục và biến cơ sở

Đặt:

- $\mathbf{X}_i = (X_i, Y_i, Z_i)^\top \in \mathbb{R}^3$ — vị trí 3D của
  marker $i$ trên mặt gel ở **trạng thái nghỉ** (no-load), trong hệ toạ độ
  camera.
- $\mathbf{X}_i + \mathbf{d}_i$ — vị trí 3D ở **trạng thái biến dạng**, với
  $\mathbf{d}_i = (d_{i,x}, d_{i,y}, d_{i,z})^\top$ là vector dịch chuyển
  của marker thứ $i$.
- $\mathbf{p}_i = (u_i, v_i)^\top$, $\mathbf{p}_i' = (u_i', v_i')^\top$ —
  toạ độ pixel của marker ở hai trạng thái.
- $\mathbf{u}_i \equiv \mathbf{p}_i' - \mathbf{p}_i \in \mathbb{R}^2$ —
  vector dịch chuyển *quan sát được* trên ảnh. Đây là đầu ra của khối
  tracking và đầu vào của các khối tiếp theo (slip detection, force).

### 1.2 Mô hình camera lỗ kim (pinhole)

Phép chiếu của camera lý tưởng (không méo) ánh xạ điểm 3D sang pixel qua:

$$
s\,\tilde{\mathbf{p}} \;=\; \mathbf{K}\bigl[\mathbf{R}\,\big|\,\mathbf{t}\bigr]\,\tilde{\mathbf{X}}^{w},
\qquad
\mathbf{K}=\begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix},
$$

với $\tilde{\mathbf{p}} = (u,v,1)^\top$, $\tilde{\mathbf{X}}^w = (X^w,Y^w,Z^w,1)^\top$,
$s$ là yếu tố scale homogeneous. $\mathbf{K}$ là **ma trận nội tại**
(intrinsic): $f_x, f_y$ tiêu cự theo pixel, $(c_x, c_y)$ là điểm chính
(principal point). $[\mathbf{R}|\mathbf{t}]$ là **ngoại tại** đưa hệ thế
giới về hệ camera.

Camera thực có biến dạng quang học, mô tả bằng mô hình **Brown–Conrady**:

$$
\begin{aligned}
\hat{x} &= x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2),\\
\hat{y} &= y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y,
\end{aligned}
$$

với $(x,y) = (X/Z, Y/Z)$ là toạ độ chuẩn hoá, $r^2 = x^2 + y^2$, $(k_1, k_2, k_3)$
là hệ số méo radial, $(p_1, p_2)$ là méo tangential. Sau đó áp $\mathbf{K}$ để
ra pixel. Vector tham số chuẩn xuất bởi OpenCV: $(k_1, k_2, p_1, p_2, k_3)$.

### 1.3 Xấp xỉ "phẳng song song" cho cảm biến tactile

Đối với cảm biến tactile, mặt gel và mặt phẳng cảm biến gần như **song song**
và cách nhau khoảng $Z_0$ cố định nhỏ (1–3 mm). Trên một marker bất kỳ:

$$
\mathbf{p}_i = \pi(\mathbf{K}, \mathbf{X}_i),
\qquad
\mathbf{p}_i' = \pi(\mathbf{K}, \mathbf{X}_i + \mathbf{d}_i).
$$

Khai triển Taylor bậc 1 quanh $\mathbf{X}_i$ với $|\mathbf{d}_i| \ll Z_0$:

$$
\mathbf{u}_i = \mathbf{p}_i' - \mathbf{p}_i
\;\approx\; J_\pi(\mathbf{X}_i)\,\mathbf{d}_i
= \frac{1}{Z_i}
\begin{pmatrix}
f_x & 0 & -f_x X_i / Z_i\\
0 & f_y & -f_y Y_i / Z_i
\end{pmatrix}
\mathbf{d}_i.
$$

Vì $Z_i \approx Z_0 = $ const và $X_i, Y_i \ll Z_0$ (camera nhìn gần như
chính diện), số hạng méo perspective $X_i/Z_i, Y_i/Z_i$ là $O(10^{-2})$, có
thể bỏ. Còn lại quan hệ tuyến tính:

$$
\boxed{\;\mathbf{u}_i \;\approx\; \alpha\,(\mathbf{d}_{i,x},\, \mathbf{d}_{i,y})^\top\;}
\qquad
\alpha = \frac{f_x}{Z_0} \approx \frac{f_y}{Z_0}.
$$

**Hệ quả** (rất quan trọng cho hệ thống):

1. **Thành phần out-of-plane $d_{i,z}$ không xuất hiện** trong $\mathbf{u}_i$
   ở xấp xỉ bậc 1. Hệ thống không quan sát trực tiếp được pháp tuyến.
2. **Hệ số $\alpha$ là vô hướng đồng nhất** trên toàn cảm biến (vì $Z_0$ const).
   Vậy nên *tỉ lệ* giữa các vector $\mathbf{u}_i$ phản ánh *tỉ lệ* các
   $(\mathbf{d}_x, \mathbf{d}_y)$ — pipeline có thể làm việc trực tiếp trên
   pixel mà **không cần** biết $\alpha$, ngoại trừ bước ước lượng lực tuyệt
   đối (xem §5).
3. **Tracking output là bất biến với scale $\alpha$**. Mọi đại lượng dùng
   xuống dưới (hướng $\theta_i$, MRVL, slip score) đều là đặc trưng *hình
   dáng* của trường vector — không bị ảnh hưởng bởi tiêu cự/khoảng cách camera.

### 1.4 Mô hình đàn hồi của lớp gel (Hooke tuyến tính)

Ở chế độ biến dạng nhỏ, gel silicone xấp xỉ là môi trường **đàn hồi tuyến
tính, đẳng hướng**. Quan hệ ứng suất–biến dạng (Hooke):

$$
\boldsymbol{\sigma} = \mathbf{C}\,\boldsymbol{\varepsilon},
\quad
\boldsymbol{\sigma}_{kl} = \lambda\,\delta_{kl}\,\varepsilon_{mm} + 2\mu\,\varepsilon_{kl},
$$

với $\lambda, \mu$ là hệ số Lamé, $\boldsymbol{\varepsilon} = \tfrac{1}{2}(\nabla \mathbf{w} + \nabla\mathbf{w}^\top)$
là tensor biến dạng nhỏ tại điểm $\mathbf{x}$ trong gel, $\mathbf{w}(\mathbf{x})$
là trường dịch chuyển 3D.

Trên **mặt cảm biến** (mặt sau gel, nơi marker nằm), trường $\mathbf{w}|_{\text{mặt}}$
phụ thuộc tuyến tính vào lực tác dụng trên **mặt tiếp xúc** (mặt trước gel)
qua một toán tử Green tích phân:

$$
\mathbf{w}(\mathbf{x}_s) = \int_{\Omega_c} \mathbf{G}(\mathbf{x}_s - \mathbf{x}_c)\,\mathbf{f}(\mathbf{x}_c)\,d\Omega_c,
$$

với $\Omega_c$ là vùng tiếp xúc, $\mathbf{f}$ là phân bố lực mặt, $\mathbf{G}$ là
tensor Green của bài toán Boussinesq–Cerruti cho nửa không gian đàn hồi. Đây
là cơ sở của các phương pháp **inverse FEM** (Ma et al. 2019).

Trong khuôn khổ luận văn, ta không giải nghịch $\mathbf{G}$ tường minh, mà
**học** quan hệ $\mathbf{u}_i \mapsto F$ bằng các mô hình ML (xem §5). Tuyến
tính Hooke chỉ xuất hiện gián tiếp dưới các tiên đề định tính sau, làm cơ sở
cho slip detection (§3):

| Trạng thái tiếp xúc | Trường $\mathbf{u}_i$ trên mặt cảm biến |
|---|---|
| **Stick (dính)** | Đồng pha với chuyển động vật, vector phẳng và tĩnh khi vật đứng. |
| **Press (nhấn)** | Xuyên tâm hướng vào tâm tiếp xúc (compress) hoặc ra (release). |
| **Slip (trượt)** | Tịnh tiến đồng nhất theo hướng trượt cho mọi marker dưới vùng tiếp xúc. |

Quan sát **tịnh tiến đồng nhất ⇔ slip** và **xuyên tâm ⇔ press/release** là
nguồn gốc vật lý của hai detector slip ở §3.

### 1.5 Hiệu chuẩn camera (Zhang 2000)

Hiệu chuẩn xác định $\mathbf{K}$ và $(k_1,k_2,p_1,p_2,k_3)$. Code:
`src/core/calibration.py`.

**Phương pháp Zhang.** Sử dụng bàn cờ phẳng có toạ độ thế giới đã biết. Với
$M$ ảnh chụp bàn cờ ở các vị thế khác nhau:

1. Cho mỗi ảnh, ràng buộc $\mathbf{p}_j = \mathbf{H}\,\mathbf{X}_j^w$ (vì mặt
   phẳng bàn cờ → $Z^w=0$) cho ta một homography $\mathbf{H} = \mathbf{K}\,[\mathbf{r}_1\,\mathbf{r}_2\,\mathbf{t}]$.
   Ước lượng $\mathbf{H}$ bằng SVD trên tập $\ge 4$ điểm.
2. Tính chất trực giao và đơn vị của $\mathbf{r}_1, \mathbf{r}_2$ cho hai
   phương trình tuyến tính trên các phần tử của
   $\mathbf{B} = \mathbf{K}^{-\top}\mathbf{K}^{-1}$. Với $M\ge 3$, hệ đủ ràng
   buộc; giải $\mathbf{B}$ rồi Cholesky → $\mathbf{K}$.
3. **Tinh chỉnh phi tuyến** Levenberg–Marquardt với cost reprojection:
   $$
   \min_{\mathbf{K},\,\boldsymbol{k},\,\{\mathbf{R}_m,\mathbf{t}_m\}}
   \sum_{m=1}^M\sum_j \big\|\mathbf{p}_{m,j} - \pi(\mathbf{K},\boldsymbol{k},\mathbf{R}_m,\mathbf{t}_m,\mathbf{X}_j)\big\|^2.
   $$

**Cài đặt trong codebase.** Bàn cờ $8\times 6$ corner trong, cạnh ô 25 mm
(`config/pipeline_config.yaml`). Pipeline `calibrate_camera`
(`calibration.py:18-77`):

1. `cv2.findChessboardCorners` định vị thô các góc.
2. `cv2.cornerSubPix` tinh chỉnh sub-pixel với cửa sổ 11×11, criteria
   `(EPS=0.001, MAX_ITER=30)`.
3. `cv2.calibrateCamera` chạy toàn bộ Zhang + LM refine, trả
   `(camera_matrix, dist_coeffs, rvecs, tvecs)`. Lưu ra `config/calib_result.npz`.

**Khử méo** (`undistort_image`, `calibration.py:91-112`):

1. `cv2.getOptimalNewCameraMatrix(α=1)` chọn ma trận mới giữ nguyên FOV.
2. `cv2.initUndistortRectifyMap` dựng bản đồ pixel→pixel.
3. `cv2.remap` với nội suy bilinear, sau đó crop theo `roi`.

**Khi nào nên bật calibration.** Trong pipeline chính, cờ `pipeline.use_calibration`
**mặc định false**. Lý do: với cảm biến tactile nhìn gần như chính diện, méo
radial là $O(0,1\%)$ trên vài chục pixel biến dạng → không ảnh hưởng tracking,
mà thêm một bước remap. Calibration chỉ cần thiết khi:

- Cần ánh xạ pixel ↔ mm cho ước lượng lực **tuyệt đối** theo Boussinesq.
- So sánh chéo dữ liệu giữa nhiều cảm biến hoặc nhiều cấu hình camera.

---

## 2. Cơ sở lý thuyết thuật toán theo dõi (tracking)

Module tracking nhận vào hai ảnh xám $I^{(0)}$ (reference, gel ở trạng thái
nghỉ) và $I^{(t)}$ (deformed) cùng tập tâm marker
$\{\mathbf{p}_i^{(0)}\}_{i=1}^N$ trên ảnh ref, và xuất ra:
$\{\mathbf{p}_i^{(t)}\}_{i=1}^N$ ở ảnh def cùng mask `valid` $\in \{0,1\}^N$.

Pipeline gồm bốn khâu nối tiếp: **tiền xử lý → phát hiện marker (chỉ trên
ref) → Pyramid Lucas–Kanade → forward–backward check + deadzone**.

### 2.1 Tiền xử lý ảnh — `src/core/preprocessing.py`

#### 2.1.1 Vấn đề: chiếu sáng không đồng đều

LED gắn quanh viền cảm biến gây *vignetting*: vùng giữa ảnh sáng hơn rìa,
chênh 30–40% intensity. Một ngưỡng tuyệt đối không thể đúng đồng thời cho
trung tâm và rìa ảnh.

#### 2.1.2 Trừ nền bằng box blur

Ý tưởng: ước lượng nền bằng bộ lọc trung bình cửa sổ $w \times w$ rất lớn so
với kích thước marker, sau đó trừ:

$$
B(\mathbf{x}) = \frac{1}{|N(\mathbf{x})|}\sum_{\mathbf{y}\in N(\mathbf{x})} I(\mathbf{y}),
\qquad
\tilde{I}(\mathbf{x}) = \max\bigl(0,\, I(\mathbf{x}) - B(\mathbf{x})\bigr).
$$

Codebase chọn $w = 101$ pixel (`preprocessing.blur_kernel = [101, 101]`).
$|N| \approx 10^4$ pixel ⇒ $B$ là ước lượng "DC" (tần số thấp) — phổ này
chính là vignetting LED. Marker (đối tượng nhỏ, tần số cao) gần như không bị
ảnh hưởng.

**Lý do dùng box blur thay vì Gaussian.** OpenCV `cv2.blur` dùng *integral
image* (summed-area table) → độ phức tạp $O(1)$ mỗi pixel, không phụ thuộc
$w$. Gaussian 101×101 separable mất gấp 10–20× thời gian. Trade-off ghi rõ ở
`preprocessing.py:33-34`: box artifact tần số cao là chấp nhận được vì $B$
chỉ dùng trừ nền.

#### 2.1.3 Min–max normalize

`cv2.normalize(..., NORM_MINMAX)`:

$$
\tilde{I}_{\text{norm}}(\mathbf{x}) = 255 \cdot \frac{\tilde{I}(\mathbf{x}) - \min \tilde{I}}{\max \tilde{I} - \min \tilde{I}}.
$$

Khôi phục dynamic range sau khi trừ nền (vì $\tilde I$ có giá trị nhỏ hơn $I$).

#### 2.1.4 CLAHE — Contrast Limited Adaptive Histogram Equalization

CLAHE (Zuiderveld 1994) chia ảnh thành lưới $G \times G$ ô (ở đây $G = 8$):

1. **Local histogram**: trong mỗi ô, đếm tần số $h[k]$ cho 256 bin.
2. **Clip**: $h[k] \leftarrow \min(h[k], c \cdot N_{\text{px}})$ với
   $c = 2{,}5$. Phần dư cộng vào *toàn bộ* các bin → giới hạn khuếch đại
   noise ở vùng đồng nhất.
3. **CDF mapping**: $T_{\text{ô}}[k] = 255 \cdot (\sum_{j \le k} h[j]) / N_{\text{px}}$.
4. **Bilinear interpolation across tiles**: pixel ở giao 4 ô được nội suy
   tuyến tính giữa 4 mapping → tránh "tile boundary" artifact.

Kết quả: marker tròn đều, nền đồng nhất, sẵn sàng cho blob detector.

#### 2.1.5 Tái sử dụng đối tượng CLAHE trong realtime

`make_clahe(config)` (`preprocessing.py:43`) khởi tạo `cv2.CLAHE` một lần;
`BaseRealtimePipeline` truyền vào tham số `_clahe` của `preprocess` để tránh
tạo lại mỗi frame — tránh chi phí Python overhead khoảng 0,3 ms/frame ở
`fps = 60`.

### 2.2 Phát hiện marker — `src/core/detection.py`

Chỉ chạy **một lần** trên ảnh reference ngay khi người dùng nhấn `r` (hoặc
khi load ảnh ở batch mode). Các frame tiếp theo *không* detect lại — LK sẽ
track từ vị trí ref sang def.

#### 2.2.1 SimpleBlobDetector

OpenCV `cv2.SimpleBlobDetector` thực hiện 5 bước:

**Bước 1 — Multi-threshold binarization.** Quét ngưỡng $t \in
[\text{minThreshold},\text{maxThreshold}]$ (= $[50, 220]$) với bước $\Delta t = 10$
. Mỗi $t$ tạo một mặt nạ nhị phân $B_t = \mathbb{1}[\tilde I_{\text{norm}} > t]$.

**Bước 2 — Connected components.** Trong mỗi $B_t$, tìm vùng liên thông; tính
centroid $(\bar u, \bar v)$ và các đặc trưng hình học.

**Bước 3 — Lọc theo property** (tổ hợp 4 bất phương trình):

| Filter | Định nghĩa | Ngưỡng | Vai trò |
|---|---|---|---|
| **Color** | Giá trị trung tâm = `blobColor=255` | bật | Giữ blob *sáng* trên nền tối |
| **Area** | $A \in [A_{\min}, A_{\max}]$ | $[30, 500]$ px² | Loại nhiễu nhỏ + cụm đơn |
| **Circularity** | $\mathcal{C} = 4\pi A/P^2$ | $\ge 0{,}5$ | Loại blob méo dài |
| **Inertia ratio** | $\lambda_{\min}/\lambda_{\max}$ ma trận quán tính | $\ge 0{,}3$ | Loại vạch dài |
| **Convexity** | $A / A_{\text{convex hull}}$ | $\ge 0{,}7$ | Loại blob lồi lõm |

Với $\mathcal{C}$: hình tròn lý tưởng $\mathcal{C} = 1$, hình vuông $\pi/4
\approx 0{,}785$. Inertia ratio $= 1$ ⇔ tròn, $\to 0$ ⇔ vạch.

**Bước 4 — Cluster across thresholds.** Một blob xuất hiện ở nhiều $t$ liên
tiếp được merge (theo `minDistBetweenBlobs`) → robust với noise.

**Bước 5 — Output.** Trả `list[cv2.KeyPoint]`, codebase chuyển thành
$(N, 2)$ float32 (`detection.py:84`).

#### 2.2.2 Vì sao SimpleBlobDetector ổn hơn threshold thuần

Một detector dựa trên `threshold + cv2.connectedComponents + filter area`
chỉ kiểm soát kích thước. SimpleBlobDetector kết hợp **4 prior hình học**
(color, area, circularity, inertia, convexity) trong một detector duy nhất,
và multi-threshold làm robust với chiếu sáng thay đổi → marker mới được
detect đáng tin cậy ngay cả khi vị trí khác nhau giữa các session.

**Hạn chế.** Không sub-pixel localization. Bù lại bằng LK ở §2.3 — tracking
LK vốn tự nội suy sub-pixel.

### 2.3 Lucas–Kanade pyramid — `src/core/tracking.py`

#### 2.3.1 Phương trình quang lưu (optical flow constraint)

Giả định cường độ ảnh không đổi giữa hai frame:

$$
I(\mathbf{x}, t) = I(\mathbf{x}+\mathbf{u}, t+1).
$$

Khai triển Taylor bậc 1:

$$
I_x u + I_y v + I_t = 0,
\qquad
\nabla I^\top \mathbf{u} + I_t = 0.
$$

Một phương trình hai ẩn — gọi là *aperture problem*: chỉ ràng buộc thành phần
$\mathbf{u}$ song song gradient. Cần thêm giả thiết để khép hệ.

#### 2.3.2 Lucas–Kanade (1981)

Giả định $\mathbf{u}$ **hằng số** trên cửa sổ $\Omega$ kích thước $w \times w$
quanh marker. Tích lũy ràng buộc trên mọi pixel $\mathbf{y}\in\Omega$:

$$
\mathbf{u}^* = \arg\min_{\mathbf{u}}
\sum_{\mathbf{y}\in\Omega} \bigl(\nabla I(\mathbf{y})^\top \mathbf{u} + I_t(\mathbf{y})\bigr)^2.
$$

Đạo hàm theo $\mathbf{u}$, đặt = 0 ⇒ nghiệm bình phương tối thiểu:

$$
\boxed{\;\mathbf{u}^* = -\,\mathbf{A}^{-1}\,\mathbf{b}\;}
\qquad
\mathbf{A} = \sum_\Omega \nabla I\,\nabla I^\top,
\quad
\mathbf{b} = \sum_\Omega I_t\,\nabla I.
$$

**Điều kiện đảo $\mathbf{A}$:** ma trận $2\times 2$ với hai trị riêng đều
lớn ⇒ pixel "góc" (corner). Đó là lý do LK + Shi–Tomasi/Harris hay đi cùng
nhau. Trong codebase, marker do `SimpleBlobDetector` cung cấp — mỗi marker
**đã có** gradient đa hướng quanh viền (vì là blob tròn sáng) → tính chất
"corner" tự nhiên thoả mãn, không cần Harris pre-filter.

#### 2.3.3 Pyramid (Bouguet 2001)

LK chỉ đúng cho dịch chuyển nhỏ (vài pixel) — vì khai triển Taylor bậc 1.
Để xử lý dịch lớn, dùng **tháp Gaussian** $L+1$ tầng:

- $I_0 = I$ (gốc), $I_\ell$ = downsample $2^\ell$ lần (smooth + decimate).
- Khởi tạo $\mathbf{u}^{(L)} = \mathbf{0}$ ở tầng cao nhất.
- Ở mỗi tầng $\ell$: warp $I_\ell^{(t+1)}$ theo $2\,\mathbf{u}^{(\ell+1)}$,
  rồi LK tinh chỉnh $\delta \mathbf{u}^{(\ell)}$:
  $\mathbf{u}^{(\ell)} = 2\mathbf{u}^{(\ell+1)} + \delta\mathbf{u}^{(\ell)}$.
- Dừng ở tầng 0 → nghiệm $\mathbf{u}^*$.

Phạm vi dịch chuyển bám được $\approx (w/2)\cdot 2^L$. Codebase: $w=21$,
$L=3$ → $\approx 84$ pixel — quá đủ cho biến dạng tactile (thường $< 30$ px).

**Tham số (`tracking.pyrlk` trong YAML):**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `win_size` | $[21, 21]$ | Cửa sổ $\Omega$ |
| `max_level` | $3$ | Số tầng pyramid (4 tầng: $1, \tfrac12, \tfrac14, \tfrac18$) |
| `term_criteria.max_iter` | $30$ | Số iter tối đa mỗi tầng |
| `term_criteria.eps` | $0{,}01$ | Ngưỡng dừng |

#### 2.3.4 Forward–Backward error (Kalal et al. 2010)

Một marker có thể "trôi" sang vùng khác nếu LK rơi vào local minimum (vd.
hai marker dính sát). **Forward–backward check** loại các tracking failure
mà không cần ground truth:

$$
\begin{aligned}
\text{Forward:}  &\quad \mathbf{p}_i^{(0)} \;\xrightarrow{\text{LK}}\; \tilde{\mathbf{p}}_i^{(1)}, \\
\text{Backward:} &\quad \tilde{\mathbf{p}}_i^{(1)} \;\xrightarrow{\text{LK}}\; \tilde{\mathbf{p}}_i^{(0)}, \\
\text{FB error:} &\quad \varepsilon_i = \big\| \tilde{\mathbf{p}}_i^{(0)} - \mathbf{p}_i^{(0)} \big\|_2.
\end{aligned}
$$

Một marker tracking ổn định phải đi-về cùng một điểm; nếu $\varepsilon_i \ge
\tau_{\text{FB}}$, marker bị đánh **invalid**. Codebase: $\tau_{\text{FB}} = 2{,}0$
px (`tracking.pyrlk.fb_threshold`). Mask cuối:

$$
\text{valid}_i = (\text{status}^f_i = 1) \wedge (\text{status}^b_i = 1) \wedge (\varepsilon_i < \tau_{\text{FB}}).
$$

(`tracking.py:55-56`).

#### 2.3.5 Deadzone — chống dao động đàn hồi vi mô

Vật liệu silicone có dao động nhiệt-đàn hồi cỡ $\pm 1$ px ngay cả khi không
tải. Để giữ tín hiệu sạch, áp **deadzone** trên dịch chuyển:

$$
\hat{\mathbf{p}}_i^{(t)} =
\begin{cases}
\mathbf{p}_i^{(0)} & \text{nếu } \big\|\mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}\big\| < d_{\min}, \\[4pt]
\mathbf{p}_i^{(t)} & \text{ngược lại}.
\end{cases}
$$

Codebase: $d_{\min} = 1$ px (`tracking.min_displacement`). Cờ
`apply_deadzone` (`tracking.py:21`) bật/tắt cơ chế:

- **Batch + slip V1**: `True` — tránh nhiễu.
- **Slip V2 + force preprocessing**: `False` — cần tín hiệu trượt chậm tích
  luỹ qua nhiều frame; deadzone sẽ "ngắt" tín hiệu này.

#### 2.3.6 Mapping output → pipeline xuống dưới

Sau bước này, đầu ra cho mỗi frame là:

- $\{\mathbf{p}_i^{(t)}\}_{i=1}^N$ — toạ độ marker biến dạng (sub-pixel float32).
- $\{\text{valid}_i\}_{i=1}^N$ — mask boolean.

Hai đại lượng phái sinh dùng xuyên suốt §3 và §5:

- **Velocity vector** $\Delta \mathbf{p}_i = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t-h)}$
  (giữa hai frame cách $h$ frame).
- **Cumulative deformation** $\mathbf{D}_i(t) = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}$
  (so với reference cố định).

---

## 3. Phát hiện trượt (slip detection)

Codebase chứa hai detector slip độc lập:

- **V1 — Mean Resultant Vector Length** (`src/slip/v1.py`): thống kê vòng
  tròn có trọng số trên trường vận tốc, kèm bộ lọc rebound + press-rate gate,
  EMA bất đối xứng.
- **V2 — Phân rã đa thang Translation/Radial** (`src/slip/v2.py`): least-squares
  decomposition của trường biến dạng tích lũy thành thành phần tịnh tiến (slip)
  + xuyên tâm (press/release) + dư, trên nhiều thang thời gian.

V2 là đóng góp gốc của luận văn, khắc phục hai hạn chế chính của V1: không
phát hiện được *slow slip* và bị **press-rate gate** *che lấp* slip xảy ra
đồng thời với press.

### 3.1 Slip V1 — Mean Resultant Vector Length

#### 3.1.1 Quan sát hiện tượng

Khi gel trượt khỏi vật, **toàn bộ** marker trong vùng tiếp xúc dịch theo
cùng một hướng (vector tịnh tiến). Khi gel chỉ bị nhấn xuống, marker dịch
theo nhiều hướng (xuyên tâm). Khi đứng yên, dao động đàn hồi sinh các vector
ngẫu nhiên nhỏ. Dấu hiệu của slip vì thế là **độ đồng nhất hướng** của
trường vận tốc.

#### 3.1.2 Định nghĩa toán — thống kê vòng tròn

Cho mỗi marker valid $i$, vector vận tốc giữa hai frame:

$$
\Delta\mathbf{p}_i = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t-h)}, \quad h = \text{`history\_buffer\_length'} = 5.
$$

Ký hiệu độ lớn $m_i = \|\Delta\mathbf{p}_i\|$ và góc
$\theta_i = \arctan2(\Delta y_i, \Delta x_i) \in (-\pi, \pi]$.

**Mean Resultant Vector Length (MRVL)** có trọng số (Fisher 1995):

$$
\boxed{\;R \;=\; \biggl|\, \sum_{i \in V_{\text{moving}}} w_i\, e^{j\theta_i} \biggr|\;}
\qquad
w_i = \frac{m_i}{\sum_j m_j}.
$$

Tương đương:

$$
R = \sqrt{\Bigl(\sum_i w_i \cos\theta_i\Bigr)^2 + \Bigl(\sum_i w_i \sin\theta_i\Bigr)^2}, \qquad
R \in [0, 1].
$$

Ý nghĩa hình học:

- $R \to 1$ ⇔ tất cả $\theta_i$ trùng nhau ⇒ slip thuần (đồng pha).
- $R \to 0$ ⇔ $\theta_i$ phân tán đều ⇒ press / nhiễu / shear hỗn loạn.
- Trọng số $w_i \propto m_i$ ưu tiên marker dịch chuyển mạnh — chuẩn hoá
  trên nền nhiễu nhỏ.

Cài đặt ở `v1.py:91-101`:

```python
angles  = np.arctan2(significant_disp[:, 1], significant_disp[:, 0])
weights = valid_mags / np.sum(valid_mags)
sum_cos = np.sum(weights * np.cos(angles))
sum_sin = np.sum(weights * np.sin(angles))
raw_r_value    = np.sqrt(sum_cos**2 + sum_sin**2)
mean_direction = np.arctan2(sum_sin, sum_cos)
```

#### 3.1.3 Ba bộ lọc tiền xử lý $R$

Trước khi tính $R$, ba bộ lọc loại các marker không hữu ích:

**(a) Motion gate.** $m_i > m_{\min} = $ `min_motion_thresh = 2{,}0` px
(`v1.py:75`). Marker đứng yên đóng góp $\theta_i$ ngẫu nhiên do làm tròn
sub-pixel ⇒ giảm $R$ "giả". Cần ngưỡng > deadzone của tracking.

**(b) Press-rate gate.** Khi gel đang bị nén nhanh, độ lớn trung bình của
trường biến dạng (so với ref) tăng:

$$
\bar m_t = \frac{1}{|V|} \sum_{i\in V} \big\| \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)} \big\|,
\qquad
\Delta \bar m = \bar m_t - \bar m_{t-1}.
$$

Nếu $\Delta \bar m > $ `press_rate_threshold = 0{,}5` px/frame, *ép*
$R \to 0$ và đặt `phase = "pressing"` (`v1.py:64-71`). Cơ chế zero-th order
chống false positive khi siết tay gắp. **Hạn chế chính**: nếu slip xảy ra
*đồng thời* với press, gate này che lấp slip.

**(c) Rebound gate.** Marker đang "hồi phục đàn hồi" sau khi vật bị nhấc có
vận tốc ngược chiều biến dạng. Loại nếu

$$
\langle \Delta\mathbf{p}_i,\, \mathbf{D}_i \rangle < \tau_{\text{rebound}} = -0{,}1
$$

(`v1.py:78-83`), với $\mathbf{D}_i = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}$
là biến dạng tích luỹ. Lý giải vật lý: khi gel nhả, marker đàn hồi về tâm —
đó *không* phải slip.

#### 3.1.4 Kiểm tra số marker tối thiểu

Sau lọc, đếm $|V_{\text{moving}}|$. Nếu $< $ `min_moving_markers = 5`
(`v1.py:87`), phase `"insufficient_motion"` — không đủ ràng buộc tin cậy
cho $R$.

#### 3.1.5 Làm trơn EMA bất đối xứng

$R$ thô dao động giữa các frame. Áp filter EMA hai hằng số:

$$
\hat R_t =
\begin{cases}
\alpha\, R_t + (1-\alpha)\,\hat R_{t-1} & \text{nếu } R_t \ge \hat R_{t-1}, \\[4pt]
\alpha_d\, R_t + (1-\alpha_d)\,\hat R_{t-1} & \text{nếu } R_t < \hat R_{t-1},
\end{cases}
$$

với $\alpha = 0{,}3$, $\alpha_d = 0{,}6$. Triết lý **tăng chậm, giảm nhanh**:

- $\alpha$ nhỏ ⇒ cần nhiều frame xác nhận trước khi $\hat R$ vượt ngưỡng → chống false positive.
- $\alpha_d$ lớn ⇒ khi slip dừng, $\hat R$ rơi nhanh → chống "stuck-in-slip".

**Phân loại cuối:**

$$
\text{is\_slip}_t = \big[\hat R_t > \tau_{\text{slip}}\big],
\qquad \tau_{\text{slip}} = 0{,}8.
$$

**Decay khi gate sớm:** ngay cả khi gate (a)–(c) trả về sớm,
$\hat R \leftarrow \hat R \cdot (1 - \alpha_d)$ vẫn chạy (`v1.py:115-116`)
— nếu trước đó đang slip nhưng bị `pressing` chặn, $\hat R$ vẫn rơi.

### 3.2 Slip V2 — Phân rã đa thang Translation/Radial

V2 thay đổi *hai* nguyên lý cơ bản so với V1:

1. **Tín hiệu nguồn** từ velocity $\Delta\mathbf{p}$ → cumulative deformation
   $\mathbf{D}(t) = \mathbf{p}^{(t)} - \mathbf{p}^{(0)}$. Slow slip
   ($\| \Delta\mathbf{p} \| \ll 1$ px/frame) tích luỹ trên $\mathbf{D}$ qua
   nhiều frame, không bị deadzone triệt.
2. **Mô hình hoá tường minh** thành phần tịnh tiến (slip) và xuyên tâm
   (press/release) bằng bình phương tối thiểu — hai thành phần *trực giao*
   (vì centroid lấy theo ref) ⇒ phát hiện slip ngay khi đang nhấn.

Pipeline dùng `apply_deadzone=False` cho LK (xem `realtime_slip_v2.py:75`).

#### 3.2.1 Đa thang thời gian

Xét sai phân $\Delta\mathbf{D}$ qua nhiều khoảng:

$$
\Delta\mathbf{D}_i^{(k)}(t) = \mathbf{D}_i(t) - \mathbf{D}_i(t-k),
\qquad
k \in \mathcal{S} = \{1, 3, 9\}.
$$

(`slip_v2.scales`). Scale $k=1$ phản ứng nhanh; $k=9$ tích luỹ slip chậm
dưới ngưỡng noise. Buffer history dài $\max(\mathcal{S}) + 2$ frame
(`v2.py:38`).

#### 3.2.2 Phân rã least-squares

Với mỗi scale $k$, tập marker valid ở **cả** $t$ và $t-k$ là $V_k$. Định nghĩa:

- **Centroid** $\mathbf{c} = \tfrac{1}{|V_k|}\sum_{i\in V_k} \mathbf{p}_i^{(0)}$
  — luôn lấy theo trạng thái nghỉ (bất biến qua thời gian) — `v2.py:120-122`.
- **Vector xuyên tâm** $\mathbf{r}_i = \mathbf{p}_i^{(0)} - \mathbf{c}$.

Xấp xỉ trường $\Delta\mathbf{D}^{(k)}$ trên $V_k$ bằng mô hình hai tham số:

$$
\boxed{\;\Delta\mathbf{D}_i \;=\; \mathbf{t} \;+\; s\,\mathbf{r}_i \;+\; \boldsymbol{\rho}_i\;}
$$

- $\mathbf{t}\in\mathbb{R}^2$: thành phần **tịnh tiến đồng nhất** = slip vector.
- $s\in\mathbb{R}$: hệ số **xuyên tâm** đồng nhất; $s>0$ ⇔ giãn (release),
  $s<0$ ⇔ nén (press).
- $\boldsymbol{\rho}_i$: dư — shear, xoay, nhiễu cục bộ.

**Bài toán bình phương tối thiểu:**

$$
\min_{\mathbf{t}, s} \sum_{i\in V_k} \big\| \Delta\mathbf{D}_i - \mathbf{t} - s\,\mathbf{r}_i \big\|^2.
$$

Vì centroid lấy theo ref ⇒ $\sum_i \mathbf{r}_i = \mathbf{0}$, nên hai biến
$(\mathbf{t}, s)$ **trực giao**, nghiệm tách rời:

$$
\boxed{\;
\mathbf{t} = \frac{1}{|V_k|} \sum_{i\in V_k} \Delta\mathbf{D}_i,
\qquad
s = \frac{\sum_i \langle \mathbf{r}_i,\, \Delta\mathbf{D}_i - \mathbf{t} \rangle}{\sum_i \|\mathbf{r}_i\|^2}
= \frac{\sum_i \langle \mathbf{r}_i,\, \Delta\mathbf{D}_i \rangle}{\sum_i \|\mathbf{r}_i\|^2}.
\;}
$$

(Đẳng thức cuối vì $\sum_i \mathbf{r}_i = \mathbf{0}$ ⇒ $\sum_i \langle \mathbf{r}_i, \mathbf{t} \rangle = 0$.)
Cài đặt ở `v2.py:62-89` (hàm `_decompose`).

#### 3.2.3 Năng lượng mỗi thành phần

Năng lượng (Frobenius² trên $|V_k|$ marker):

$$
E_{\text{slip}} = |V_k|\cdot \|\mathbf{t}\|^2,
\quad
E_{\text{press}} = s^2 \sum_{i\in V_k} \|\mathbf{r}_i\|^2,
\quad
E_{\text{resid}} = \sum_{i\in V_k} \|\boldsymbol{\rho}_i\|^2.
$$

Vì hai biến trực giao, **decomposition Pythagore**:

$$
\sum_i \|\Delta\mathbf{D}_i\|^2 = E_{\text{slip}} + E_{\text{press}} + E_{\text{resid}}.
$$

**Slip fraction:**

$$
\text{slip\_fraction} = \frac{E_{\text{slip}}}{E_{\text{slip}} + E_{\text{resid}} + \epsilon} \in (0, 1].
$$

Phép chia *loại* $E_{\text{press}}$ ⇒ slip_fraction phản ánh phần trăm năng
lượng **không-press** thực sự là tịnh tiến đồng nhất.

#### 3.2.4 MRVL trên trường đã khử radial

Sau khi loại thành phần xuyên tâm:

$$
\widetilde{\Delta\mathbf{D}}_i = \Delta\mathbf{D}_i - s\,\mathbf{r}_i.
$$

Áp lại MRVL có trọng số (như §3.1.2) trên $\widetilde{\Delta\mathbf{D}}$ →
coherence $R \in [0, 1]$.

**Ưu điểm.** Ngay cả khi đang nhấn ($|s|$ lớn), nếu có thêm slip nhỏ cộng vào,
$\widetilde{\Delta\mathbf{D}}$ vẫn cho coherence cao — V2 *không* bị che bởi
press như V1.

#### 3.2.5 Score per-scale

Với mỗi scale $k$ vượt qua ba điều kiện cứng:

- $|V_k| \ge $ `min_valid_markers = 6`,
- $\|\mathbf{t}\| \ge $ `translation_min = 0{,}10` px,
- $R \ge $ `coherence_min = 0{,}45`,

tính:

$$
\boxed{\;
\text{score}^{(k)} = \min\!\Bigl(\frac{\|\mathbf{t}\|}{T_0},\,1\Bigr) \cdot \text{slip\_fraction} \cdot R,
\quad T_0 = \text{`translation\_norm'} = 2{,}0 \text{ px}.
\;}
$$

Ba thừa số độc lập về ý nghĩa:

1. **Cường độ tịnh tiến** $\min(\|\mathbf t\|/T_0, 1)$: saturate sau $T_0$ —
   slip mạnh hơn không thưởng thêm.
2. **Tỉ lệ thuộc về slip**: bao nhiêu phần năng lượng không phải nhiễu shear/xoay.
3. **Tính đồng nhất hướng**: trường tịnh tiến phải coherent.

Tích chỉ cao khi cả ba lớn ⇒ false positive thấp.

#### 3.2.6 Short-scale gate — chống stuck-in-slip

**Vấn đề:** sau khi slip kết thúc, $\Delta\mathbf{D}^{(9)}$ vẫn còn năng lượng
tịnh tiến (do tích luỹ 9 frame trước), khiến score scale dài giảm chậm.

**Giải pháp:** dùng scale ngắn (phản ứng nhanh) làm "gate" cho scale dài:

$$
g = \max\!\biggl(\text{softmin},\; \max_{k \le K_g} \min\!\Bigl(\frac{\|\mathbf{t}^{(k)}\|}{N_g \sqrt{k}},\,1\Bigr)\biggr),
$$

- $K_g = $ `short_gate_max_scale = 3`,
- $N_g = $ `short_gate_norm = 0{,}25` px,
- `softmin = 0{,}15`.

Áp dụng: score tại scale $k > K_g$ bị nhân với $g$. Khi slip dừng, scale $\le K_g$
sụp nhanh ⇒ $g \to $ softmin ⇒ score scale dài bị **kéo xuống** đồng thời.
Cài đặt: `v2.py:163-184`.

#### 3.2.7 Phân loại phase từ radial rate

Chuẩn hoá hệ số xuyên tâm theo scale (vì $s$ tỉ lệ với độ lớn $\Delta\mathbf{D}$,
mà $\Delta\mathbf{D}$ tỉ lệ với scale ở slip đều):

$$
\dot s^{(k)} = s^{(k)} / \max(k, 1).
$$

Phase lấy từ scale ngắn nhất có dữ liệu (`v2.py:220-232`):

| Điều kiện | Phase |
|---|---|
| $\dot s > $ `radial_rate_min = 0{,}004` | `"pressing"` |
| $\dot s < -$ `radial_rate_min` | `"releasing"` |
| Còn lại | `"stick"` |
| Không scale nào thoả | `"idle"` |

**Press/release gate:** khi phase là pressing/releasing, V2 *chỉ giữ* slip nếu

$$
\frac{E_{\text{slip}}}{E_{\text{press}} + \epsilon} > \text{`press\_slip\_ratio'} = 0{,}35
$$

(`v2.py:189-192`). Trong lúc nhấn, slip phải đủ mạnh so với press mới được
công nhận — tránh false positive khi siết tay gắp ổn định.

#### 3.2.8 Hysteresis Schmitt + EMA bất đối xứng

Score được EMA với hai hằng số $\alpha_\uparrow = 0{,}5$, $\alpha_\downarrow = 0{,}7$.
Cờ slip dùng hysteresis Schmitt (hai ngưỡng):

$$
\text{is\_slip}_t =
\begin{cases}
\text{True}  & \text{nếu False và } \hat S_t > \tau_{\text{on}} = 0{,}55, \\
\text{False} & \text{nếu True và } \hat S_t < \tau_{\text{off}} = 0{,}35,
\end{cases}
$$

(`v2.py:197-203`). Hai ngưỡng tách biệt loại jitter quanh điểm chuyển trạng
thái — cốt yếu khi dùng làm trigger điều khiển grasp.

### 3.3 So sánh V1 và V2

| Khía cạnh | V1 | V2 |
|---|---|---|
| Tín hiệu nguồn | Velocity $\Delta\mathbf{p}$ giữa 2 frame | Cumulative deformation $\mathbf{D}(t)$ vs ref |
| Thang thời gian | 1 (history_buffer) | Nhiều ($\mathcal{S}$) |
| Tách press / slip | Press-rate gate (fail khi cùng xảy ra) | Phân rã LS hoàn toàn |
| Slow slip | Bị deadzone & motion-gate triệt | Bắt được nhờ scale dài |
| Latency phản ứng tắt | 1 hằng số EMA | EMA + short-scale gate |
| Phân loại phase | Binary slip / non-slip | slipping / pressing / releasing / stick |
| Tham số cấu hình | 9 | 14 |
| Diễn giải | Cao (scalar $R$) | Cao (4 đại lượng vật lý $\mathbf{t}, s, R, $ slip_frac) |

Hai detector dùng đồng pipeline tracking và tham số tracking; chỉ khác ở
`apply_deadzone`. Việc giữ cả hai cho phép so sánh A/B trên cùng hardware
+ cùng dữ liệu.

---

## 4. Thiết kế hệ thống thu thập dữ liệu

Để huấn luyện mô hình ước lượng lực (§5) và đánh giá slip detector trên
ground truth, cần dữ liệu **đồng bộ thời gian** giữa ba nguồn: camera tactile,
force gauge tham chiếu (Imada ZTA), và vị trí indenter (motor tuyến tính
Arduino). Hệ thống được hiện thực ở `src/collection/`.

### 4.1 Yêu cầu thiết kế

| Yêu cầu | Tại sao | Cách đáp ứng |
|---|---|---|
| **Đồng bộ ms-precision** | Force regression cần (frame, force) cùng moment | Mỗi sensor tự gắn `time.monotonic()` ngay khi sample đến — không join ở write-time |
| **Throughput 30 FPS không drop** | Tránh aliasing chuyển động nhanh | Producer-consumer + bounded queue, writer dedicated thread |
| **Trial-level metadata** | Reproducibility + fair model split | YAML schema session/trial + tags |
| **Force zero offset** | Imada có DC bias | Median 20 sample lúc tạo session |
| **Repro snapshot** | Truy vết version code/firmware | Lưu git SHA, pip versions, firmware SHA-256 vào session.yaml |

### 4.2 Kiến trúc Producer–Consumer

`src/collection/realtime_station.py:513-664`. Bốn thread chạy song song trong
một process Python (dùng `threading`, không phải `multiprocessing` vì tất cả
đều I/O-bound, không bị GIL):

```
┌────────────────┐        ┌───────────────┐
│ Camera thread  │ ──────►│ frame_queue   │ (maxsize=200)
│ (poll cv2)     │        └───────────────┘
└────────────────┘                │
                                  │
┌────────────────┐        ┌───────────────┐
│ Imada thread   │ ──────►│ force_queue   │ (maxsize=500)
│ (serial 19200) │        └───────────────┘
└────────────────┘                │
                                  │       ┌──────────────┐
┌────────────────┐        ┌──────────────┐│ Writer thread│
│ Arduino thread │ ──────►│ motor_queue  ││ (block tới   │
│ (serial 115200)│        └──────────────┘│ recording_  │
└────────────────┘                │       │ event)       │
                                  └──────►└──────────────┘
                                              │
                                              ▼
                                  ghi 1 jpg + 3 csv
```

#### 4.2.1 Camera thread (`_worker_camera`, `realtime_station.py:513-534`)

```python
while True:
    ret, frame = self.cap.read()
    if not ret: time.sleep(0.01); continue
    ts_mono = time.monotonic()
    with self._lock:                       # snapshot atomically
        self._latest_frame = frame
    if self._recording_event.is_set():
        snap = FrameSnap(ts_mono, ts_wall, frame.copy())
        self._frame_queue.put_nowait(snap) # drop nếu queue full
```

Camera đọc liên tục với buffer 1 (`CAP_PROP_BUFFERSIZE = 1`) để frame mới
nhất luôn sẵn sàng. Auto-exposure và auto-WB **bị tắt** (`CAP_PROP_AUTO_EXPOSURE = 1`,
`CAP_PROP_AUTO_WB = 0`) để tránh drift brightness giữa các trial trong cùng
session.

#### 4.2.2 Imada thread (`_worker_imada`, `realtime_station.py:536-568`)

Imada ZTA force gauge giao tiếp serial 19200 baud. Protocol:

- Master gửi `D\r` (request data).
- Slave trả `[r]±X.XXNUNIT[STATUS]\r`, ví dụ `+0.469N` hoặc `-0.001NTO`.

Parser `parse_imada` (`realtime_station.py:79-97`):

1. Strip, bỏ tiền tố `r`/`R` (read mode).
2. Regex `[-+]?\d*\.?\d+` match số ở đầu chuỗi.
3. Trả `NaN` nếu fail — sample sẽ bị filter ở downstream.

Polling rate ~20 Hz (`time.sleep(0.05)` giữa các request).

#### 4.2.3 Arduino thread (`_worker_arduino`, `realtime_station.py:570-599`)

Bidirectional serial 115200 baud:

- **Send**: pull từ `_cmd_queue` (GUI → arduino), gửi `f <mm>\n`/`b <mm>\n`/`s\n`.
- **Read**: parse response `M: Forward X mm` / `M: Backward X mm` /
  `M: Stopped` / `ERR: ...`. Mỗi response gắn ts_mono → motor_queue.
- **State tracking**: `_motor_state ∈ {idle, moving, unknown}` cập nhật từ
  response — `M: Stopped` ⇒ idle, `M: Forward/Backward` ⇒ moving.

#### 4.2.4 Writer thread (`_worker_writer`, `realtime_station.py:631-664`)

```python
while True:
    self._recording_event.wait()        # block hoàn toàn khi không record
    self._drain_one_iteration()         # pull 3 queue, ghi 3 đích
```

`recording_event.wait()` block hoàn toàn (không busy-loop) khi không record
— tiết kiệm CPU. Mỗi iter của `_drain_one_iteration` cycle qua 3 queue với
timeout ngắn (50 ms cho frame, 0 ms cho force/motor) để đảm bảo cả 3 sensor
được flush đều.

#### 4.2.5 Drop frame counting

Khi `frame_queue.put_nowait` raise `queue.Full` (writer chậm hơn camera),
counter `_n_dropped` tăng. Cuối trial, `n_dropped` được ghi vào `trial.yaml`
để đánh giá chất lượng dữ liệu — ngưỡng chấp nhận: $< 1\%$ ở 30 FPS.

### 4.3 Đồng bộ thời gian — `time.monotonic()`

`time.monotonic()` là đồng hồ không bị NTP/system-time điều chỉnh, đảm bảo
*đơn điệu tăng* — bắt buộc cho timestamp dữ liệu khoa học. Pipeline chỉ
dùng `monotonic` để join giữa các stream; `time.time()` (wall clock) chỉ
ghi cho hiển thị, không dùng cho join.

#### 4.3.1 Sync ở phía downstream

Vì 3 stream có sample rate khác nhau (camera 30 Hz, force 20 Hz, motor
event-driven), join chỉ thực hiện *offline* khi prepare cache. Hàm
`_nearest_force` trong `force_model/prepare.py:67-79`:

```python
idx = int(np.argmin(np.abs(force_ts - ts_target)))
if abs(force_ts[idx] - ts_target) > tolerance_s:  # tolerance_s = 0.1
    return math.nan
return float(force_vals[idx])
```

Mỗi frame được gắn force qua nearest-neighbor với tolerance $0{,}1$ s. Frame
ngoài tolerance bị skip (`n_skipped`). Tolerance này phải:

- Đủ lớn để chấp nhận lệch pha 30 Hz vs 20 Hz (chu kỳ $\le 50$ ms).
- Đủ nhỏ để không join nhầm hai event lực khác nhau (lực thay đổi nhanh khi
  contact).

### 4.4 Schema dataset

Định nghĩa ở `src/collection/dataset.py`. Layout thư mục:

```
data/sessions/<session_id>/
  session.yaml                  # metadata + repro snapshot
  reference/
    ref_<ts>.jpg                # reference image — chung cho cả session
  trials/
    trial_001/
      trial.yaml                # tags, motor_state_at_start, n_frames…
      frames/
        frame_<ts>.jpg          # JPEG quality 95
      frames.csv                # ts_mono, ts_wall, image_name
      force_log.csv             # ts_mono, force_n
      motor_log.csv             # ts_mono, direction, payload
    trial_002/...
```

#### 4.4.1 `session.yaml` — metadata + repro

```yaml
session_id: 20260502_143027
created_at: 2026-05-02T14:30:27
hardware:
  arduino_port: /dev/ttyACM0
  imada_port:   /dev/ttyACM1
  arduino_connected: true
  imada_connected:   true
  firmware_sha256: 8a9c... (SHA-256 của Pyserial.ino đang nạp)
camera:
  id: 2
  width:  1280
  height: 1080
  exposure: -6.0
  gain: 0.0
  auto_exposure: 1.0     # 1 = manual, 3 = auto
  auto_wb: 0.0
  fps: 30.0
force:
  zero_offset_n: -0.0034    # median 20 sample lúc tạo session
notes: "gel v3, LED 80%, indenter cylinder 8mm"
repro:
  git_sha: e9cd54d... (40-hex)
  pip_versions:
    opencv-python: 4.10.0.84
    numpy:         1.26.4
    pyserial:      3.5
    customtkinter: 5.2.2
    pyyaml:        6.0.1
```

Bốn nhóm metadata phục vụ FAIR (Findable–Accessible–Interoperable–Reusable):

1. **hardware** — port, connect status, firmware hash. Phát hiện ngay khi
   firmware bị thay đổi giữa session.
2. **camera** — toàn bộ thông số V4L2 *sau khi* set, bao gồm exposure thực
   tế (≠ giá trị set vì backend round). Cho phép tái lập điều kiện chiếu sáng.
3. **force.zero_offset_n** — bias DC trừ ở downstream training.
4. **repro** — git SHA cho code, pip versions cho thư viện.

#### 4.4.2 `trial.yaml`

```yaml
trial_id: trial_001
start_ts_wall: 1746189027.123
start_ts_mono: 12345.6789
end_ts_mono:   12350.9876
duration_s:    5.3087
n_frames:        159
n_force_samples: 106
n_motor_events:    4
n_dropped:         0
motor_state_at_start: idle
tags: ["push", "slip", "release_slow"]
```

Tags là free-form, parse từ GUI theo comma. Model split ở §5.1 dùng *trial-level*
(không frame-level) để tránh leakage giữa các frame liền kề trong cùng trial.

### 4.5 Giao thức Arduino + tính steps_per_mm

Firmware `src/collection/Pyserial/Pyserial.ino` dùng thư viện
`AccelStepper`. Tham số mechanical:

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `MICRO_STEP` | 16 | Microstepping mode driver |
| `ANGLE_STEP` | 1.8° | Bước đầy đủ stepper NEMA-17 |
| `MM_PER_REV` | 8 mm | Trục vít M8, bước 1.25 mm × 8 ren = lead 8 |

Quy đổi:

$$
\text{STEPS\_PER\_MM} = \frac{\text{MICRO\_STEP} \cdot (360^\circ / \text{ANGLE\_STEP})}{\text{MM\_PER\_REV}}
= \frac{16 \cdot 200}{8} = 400.
$$

**Lệnh:**

| ASCII | Nghĩa | Phản hồi |
|---|---|---|
| `f <mm>` | tiến `<mm>` (steps = round(mm × 400)) | `M: Forward X mm` |
| `b <mm>` | lùi | `M: Backward X mm` |
| `s` | dừng mềm (deceleration) | `M: Stopped` |
| (không hợp lệ) | – | `ERR: invalid distance` / `ERR: unknown cmd` |

Pipeline Python parse các phản hồi này để cập nhật `_motor_state` ∈ {idle, moving, unknown}
(`realtime_station.py:611-617`).

### 4.6 Đảm bảo độ tin cậy

Bốn cơ chế bảo vệ chất lượng dữ liệu:

1. **Bounded queue** (200/500/500): writer chậm thì drop frame *thay vì* OOM.
   Số drop ghi vào `trial.yaml`.
2. **Drain stale events** trước mỗi recording (`_drain_queues`): xoá sample
   từ idle period để tránh trộn vào trial mới.
3. **Force zero offset** lấy median 20 sample (trung vị robust với outlier
   của Imada — nhiều khi parse ra giá trị bất thường vì status code dính
   liền số).
4. **Manual exposure/WB**: tắt auto để brightness của reference khớp
   brightness của các frame trong trial — nếu auto, exposure điều chỉnh khi
   gel nén ⇒ reference không còn đại diện đúng cho "no-load".

---

## 5. Thuật toán ước lượng lực tiếp xúc

Codebase cung cấp **ba** đường đi từ dữ liệu thu thập sang lực scalar
(Newton). Mỗi mô hình tiếp cận ở mức độ trừu tượng khác nhau, cho phép so
sánh fair *"khi nào feature engineering đủ tốt, khi nào cần học từ pixel?"*.

| Module | Input | Mô hình | Số tham số (mặc định) |
|---|---|---|---|
| `force_poly` | 9 scalar feature từ trường disp | Polynomial regression bậc ≤ 2 | $\sim 55$ |
| `force_model` | $(N, 4)$ feature mỗi marker | PointNet (shared MLP + masked pool) | $\sim 25\text{K}$ |
| `force_cnn` | 2-channel image $[I_{\text{ref}}, I_{\text{def}}]$ | ResNet-18 hoặc SmallCNN | $\sim 11\text{M}$ / $\sim 110\text{K}$ |

Cả ba dùng **cùng dataset** đã prepare offline để đảm bảo so sánh fair.

### 5.1 Tiền xử lý chung — `force_model/prepare.py`

Trước khi train, `prepare.py` quét toàn bộ `data/sessions/<>/trials/<>` và
cache kết quả ra `.npz` cho mỗi trial:

1. Detect marker trên reference của session (chung cho mọi trial trong session).
2. Track LK với `apply_deadzone=False` (cần signal slow slip cho slip downstream)
   cho mọi frame của trial → `disp` $(T, N, 2)$ + `valid` $(T, N)$.
3. Sync force theo nearest-neighbor (tolerance $0{,}1$ s).
4. Trừ zero offset: `force = force_raw - zero_offset_n`.
5. Lưu compressed `.npz`:

```text
ref_pts (N, 2)        — toạ độ marker tham chiếu
disp    (T, N, 2)     — dịch chuyển từng frame
valid   (T, N) bool   — LK valid mask
force   (T,)          — force đã trừ offset (Newton)
ts_mono (T,)          — timestamp monotonic
image_w, image_h, n_markers, trial_id, session_id, force_zero_offset_n
```

**Trial-level split** (`src/force_model/dataset.py:39-65` và
`src/force_cnn/dataset.py:185-206`):

- 70/15/15 train/val/test theo *trial id*, seed cố định (= 42).
- Frame trong cùng trial chỉ thuộc một split — tránh leakage.

### 5.2 PolynomialRegressor (`src/force_poly/`)

#### 5.2.1 Feature engineering — 9 đặc trưng vô hướng

Định nghĩa ở `force_poly/features.py:15-25` (`FEATURE_NAMES_V1`):

| Tên | Công thức | Ý nghĩa vật lý |
|---|---|---|
| `disp_mag_mean` | $\bar m = \tfrac{1}{|V|}\sum_{i\in V} \|\mathbf u_i\|$ | Mức biến dạng trung bình |
| `disp_mag_max` | $\max_i \|\mathbf u_i\|$ | Peak biến dạng (chỉ điểm tải) |
| `disp_mag_std` | $\text{std}_i\|\mathbf u_i\|$ | Tính không đồng nhất |
| `dx_mean`, `dy_mean` | $\bar u_x, \bar u_y$ | Hướng tải tổng hợp |
| `disp_mag_sum_norm` | $\tfrac{1}{|V|}\sum \|\mathbf u_i\|$ | Tổng biến dạng / N (tránh phụ thuộc N) |
| `radial_disp_mean` | $\bar r$ — xem dưới | Nén ($-$) / giãn ($+$) quanh tâm grid |
| `tangential_disp_mean` | $\bar \tau$ — xem dưới | Shear / xoắn |
| `valid_ratio` | $|V| / N$ | Độ phủ marker — mất tracking nhiều ⇒ lực ước lượng yếu |

**Radial / tangential** — vector đơn vị xuyên tâm
$\hat{\mathbf r}_i = \mathbf r_i / \|\mathbf r_i\|$ với
$\mathbf r_i = \mathbf p_i^{(0)} - \mathbf c$, $\mathbf c$ là tâm marker grid:

$$
\text{radial}_i = \langle \mathbf u_i, \hat{\mathbf r}_i \rangle,
\qquad
\text{tangential}_i = \hat r_{i,x} \cdot u_{i,y} - \hat r_{i,y} \cdot u_{i,x}.
$$

(component sau là cross product 2D — tương đương xoắn quanh tâm). Cài đặt
`features.py:81-83`.

**Scale invariance.** Toạ độ và disp đều normalize theo $W$ (image width):

$$
\tilde u_x = u_x / W, \quad \tilde u_y = u_y / W, \quad \tilde x = x / W, \quad \tilde y = y / W.
$$

Dùng *cùng* $W$ cho cả 2 trục để radial/tangential hợp lệ (cùng đơn vị).
Nhờ đó feature *scale-invariant* với độ phân giải camera; tâm grid lấy theo
các marker valid (tránh lệch khi mất tracking).

#### 5.2.2 Standardize trước, expand sau

Trước khi expand polynomial, standardize 9 feature theo train statistics:

$$
\tilde z_d = \frac{z_d - \mu_d}{\sigma_d + \epsilon}.
$$

`fit_scaler` (`model.py:122-135`) học $(\mu_d, \sigma_d)$ một lần từ train
data, lưu vào `register_buffer` ⇒ serialize cùng checkpoint.

**Vì sao standardize trước expand?** Nếu không standardize, hệ số bậc cao
$z_d^d$ phụ thuộc bậc $d$ của scale gốc ⇒ optimizer (AdamW) sẽ điều chỉnh hệ
số chậm (gradient ratio giữa term bậc 1 và term bậc 2 quá chênh) và dễ
diverge.

#### 5.2.3 Polynomial expansion

Cho $D = 9$ feature đầu vào, expand thành mọi monomial bậc $\le d$:

$$
\Phi(\tilde{\mathbf z}) = \bigl(\,1,\, \tilde z_1,\, \tilde z_2,\, \dots,\, \tilde z_1^2,\, \tilde z_1 \tilde z_2,\, \dots,\, \tilde z_D^d\,\bigr)^\top \in \mathbb{R}^P,
$$

với $P = \binom{D + d}{d}$. Default $D=9, d=2 \Rightarrow P = 55$. Nếu
$d=3 \Rightarrow P = 220$.

Cài đặt qua `itertools.combinations_with_replacement` (`model.py:25-35`):

```python
def _monomial_indices(n_features, degree):
    out = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            out.append(combo)
    return out
```

Indices được flatten + offset rồi đưa vào `register_buffer` — serialize cùng
checkpoint, không cần lưu state riêng (`model.py:97-103`).

#### 5.2.4 Head

| Head | Cấu trúc | Số param thêm |
|---|---|---|
| `linear` | $\mathbf{w}^\top \Phi$, no bias (bias đã có ở term bậc 0) | $P$ |
| `mlp_small` | $\Phi \to \text{Linear}(P, 16) \to \text{ReLU} \to \text{Dropout} \to \text{Linear}(16, 1)$ | $\sim 16 P + 17$ |

`linear` là polynomial regression thuần — *diễn giải trực tiếp được*: hệ số
mỗi monomial mang ý nghĩa vật lý (tuyến tính = Hooke; bậc 2 = nén bậc cao).

#### 5.2.5 Loss + optimizer + early stop

```yaml
loss:        HuberLoss(delta=1.0)        # robust với outlier
optimizer:   AdamW(lr=5e-3, wd=1e-2)
scheduler:   CosineAnnealingLR(T_max=num_epochs, eta_min=lr*0.01)
early_stop:  patience=40 (epoch không cải thiện val MAE)
batch_size:  256
seed:        42 (cho split + init torch)
```

**Huber loss:**

$$
\mathcal{L}_\delta(e) =
\begin{cases}
\tfrac{1}{2} e^2 & |e| \le \delta, \\
\delta |e| - \tfrac{1}{2}\delta^2 & |e| > \delta.
\end{cases}
$$

Bình phương ở vùng nhỏ (smooth gradient gần optimum), tuyến tính ở vùng lớn
(robust với outlier — Imada parse fail thành giá trị lạ thỉnh thoảng đi qua
filter NaN).

**Cosine annealing:**

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\bigl(1 + \cos(\pi t / T_{\max})\bigr).
$$

LR giảm dần từ $\eta_0$ về $\eta_{\min} = \eta_0 / 100$ — phù hợp với polynomial
regression (loss surface khá smooth).

#### 5.2.6 Vai trò trong luận văn

`force_poly` đóng vai trò **baseline diễn giải được**. Sau khi train, hệ số
polynomial có thể đọc trực tiếp ý nghĩa vật lý:

- Hệ số term bậc 1 trên `disp_mag_mean` ≈ độ cứng tuyến tính (Hooke).
- Hệ số bậc 2 trên `disp_mag_mean` × `radial_disp_mean` ≈ phi tuyến nén.
- Hệ số bậc 1 trên `valid_ratio` ≈ correction cho mất tracking.

Đây là điểm tham chiếu để phán xét mô hình deep learning có *đáng* hay không.

### 5.3 ForceNet — PointNet-style (`src/force_model/`)

#### 5.3.1 Đặc trưng đầu vào

Mỗi marker → vector 4 chiều:

$$
\mathbf{f}_i = \bigl(x_i^{(0)}/W,\; y_i^{(0)}/H,\; u_{i,x}/W,\; u_{i,y}/W\bigr) \in \mathbb{R}^4.
$$

Padding tới $N_{\max} = \max_t N(t)$ (qua tất cả trial,
`dataset.py:68-74`), kèm boolean mask đánh dấu marker thật (`mask=True`) so
với pad (`mask=False`).

**Augmentation** (`dataset.py:166-198`):

- **Flip ngang/dọc** (prob 0.5): $x \to W-1-x$, $\mathbf{u}_x \to -\mathbf{u}_x$.
- **Xoay** $\le 5°$: rotate cả $(x, y)$ quanh tâm ảnh và rotate vector $\mathbf{u}$.
- **Noise** $\sigma = 0{,}5$ px lên $\mathbf{u}$.

#### 5.3.2 Kiến trúc PointNet (Qi et al. 2017)

Trường marker là **set không có thứ tự** — model phải bất biến hoán vị.
PointNet áp dụng hai nguyên lý:

**1. Shared per-point MLP $\phi: \mathbb{R}^4 \to \mathbb{R}^{128}$.**
Hiện thực qua `Conv1d(kernel=1)` (`model.py:13-23`):

$$
\phi(\mathbf{f}_i) = \text{ReLU}(\text{BN}(\mathbf W_2 \cdot \text{ReLU}(\text{BN}(\mathbf W_1 \cdot \mathbf{f}_i)))).
$$

Codebase: hidden dims $[64, 128]$. `Conv1d(kernel=1)` trên tensor $(B, 4, N)$
≡ áp cùng MLP cho từng marker mà giữ batch dim.

**2. Symmetric pooling.** Hàm pooling phải là *đối xứng* để output bất biến
với hoán vị marker. Codebase concat **max** và **mean**:

$$
g = \bigl[\,\max_{i\in V} \phi(\mathbf{f}_i)\,\big\|\,\tfrac{1}{|V|} \sum_{i\in V} \phi(\mathbf{f}_i)\,\bigr] \in \mathbb{R}^{2 \cdot 128}.
$$

(`model.py:31-60`). Max-pool capture *peak feature* (marker dịch nhiều
nhất); mean-pool capture *trung bình toàn cục*. Concat giữ permutation-invariance
mà tăng biểu diễn.

**3. Mask-aware pooling** — chi tiết quan trọng:

```python
# Max: set invalid to very negative (không lọt vào max)
feats_for_max = feats.masked_fill(~mask, very_neg)
max_pool = feats_for_max.max(dim=2).values

# Mean: chỉ tính trên valid; clamp count ≥ 1 tránh div 0
sum_pool = (feats * mask).sum(dim=2)
count    = mask.sum(dim=2).clamp(min=1.0)
mean_pool = sum_pool / count

# Edge case: batch không có valid marker nào → set max=0
no_valid = (mask.sum(dim=1) == 0)
max_pool = max_pool.masked_fill(no_valid, 0.0)
```

Tránh gradient leak qua điểm pad — nếu không mask, max-pool sẽ chọn $\phi(\mathbf 0)$
cho marker pad, làm noise vào output.

**4. Head MLP** $\psi: \mathbb{R}^{256} \to \mathbb{R}^{64} \to \mathbb{R}$:

$$
\hat F = \text{Linear}(\text{Dropout}(\text{ReLU}(\text{Linear}(g)))).
$$

#### 5.3.3 Tính chất quan trọng

- **Permutation-invariance**: shared MLP + symmetric pooling ⇒ output không
  phụ thuộc thứ tự marker. Quan trọng vì detector trả keypoints theo thứ tự
  raster của OpenCV — thứ tự thay đổi khi marker pad/lost.
- **Compact**: $\sim 25$K params — chạy realtime trên CPU.
- **Mask-aware**: invalid marker được "vô hiệu hoá" trong cả max và mean,
  không leak gradient.

### 5.4 ForceCNN — End-to-end (`src/force_cnn/`)

#### 5.4.1 Input

2-channel image $[I_{\text{ref}}, I_{\text{def}}]$ resize về $240\times 320$,
normalize $[0, 1]$. **Không cần** detect/track — model học trực tiếp từ pixel.

**Augmentation đồng bộ** giữa hai channel (`dataset.py:224-251`):

- Flip ngang/dọc (cả ref và def cùng flip).
- Xoay $\le 5°$ (cả hai cùng góc).
- Brightness $\Delta \in \pm 0{,}1$ và contrast $\times (1 \pm 0{,}1)$ — *cùng*
  cho ref và def, mô phỏng exposure drift của camera mà *không* phá quan hệ
  vật lý giữa hai ảnh.

#### 5.4.2 Backbone — hai lựa chọn

**(a) ResNet-18 (mặc định, pretrained ImageNet)** — `~11M` params.

ImageNet `conv1` ban đầu là $W^{\text{old}} \in \mathbb{R}^{64 \times 3 \times 7 \times 7}$
cho 3-channel RGB. Codebase **inflate** sang 2-channel:

$$
W^{\text{new}}_{:,c,:,:} = \frac{1}{3} \sum_{c'=1}^{3} W^{\text{old}}_{:,c',:,:},
\qquad c \in \{0, 1\}.
$$

(`model.py:59-68`):

```python
def _inflate_conv1_from_pretrained(W_old, in_channels):
    mean_w = W_old.mean(dim=1, keepdim=True)   # (64, 1, 7, 7)
    return mean_w.repeat(1, in_channels, 1, 1) # (64, 2, 7, 7)
```

Trick này giữ nguyên các filter edge/blob mà ImageNet đã học, ngay cả khi
domain mới không phải RGB — quan trọng vì dữ liệu tactile khan hiếm
(vài ngàn frame) so với ImageNet (1.4M).

**(b) SmallCNN** — `~110K` params. 4 block conv-bn-relu-pool, GAP cuối:

$$
[2, 240, 320] \xrightarrow{\text{conv5×5}} [16, 120, 160] \xrightarrow{\text{conv3×3}} [32, 60, 80] \xrightarrow{} [64, 30, 40] \xrightarrow{\text{conv3×3 (no pool)}} [128, 30, 40] \xrightarrow{\text{GAP}} [128].
$$

Dùng để debug nhanh trên CPU/laptop khi không có GPU.

#### 5.4.3 Head

```
Linear(out_dim, 128) → ReLU → Dropout(0.2) → Linear(128, 1)
```

`out_dim = 512` cho ResNet-18 (sau khi `fc → Identity`), `128` cho SmallCNN.

#### 5.4.4 Vai trò luận văn

`force_cnn` là *upper-bound* khi data phong phú: học cả feature lẫn regression
end-to-end. Nhược điểm: cần nhiều dữ liệu, khó diễn giải, latency lớn (~200 MB
weights cho ResNet-18 fp32).

### 5.5 So sánh ba mô hình

| Tiêu chí | `force_poly` | `force_model` (PointNet) | `force_cnn` |
|---|---|---|---|
| Input | 9 scalar | $(N, 4)$ + mask | 2-channel image |
| Cần preprocess (detect+track) | Có | Có | **Không** |
| Số param | $\sim 55$ | $\sim 25$K | $\sim 11$M / $\sim 110$K |
| Diễn giải | **Cao** (đọc hệ số) | Trung bình | Thấp (black-box) |
| Cần GPU | Không | Tuỳ | Có (cho training thực dụng) |
| Augmentation hữu hiệu | ít | rotate/noise | flip/rotate/photometric |
| Robustness với mất tracking | trực tiếp qua `valid_ratio` | qua mask trong pool | gián tiếp qua image (LK error không chuyển vào input) |
| Latency inference (CPU) | $< 1$ ms | $\sim 5$ ms | $\sim 30$–$200$ ms tuỳ backbone |

Câu hỏi nghiên cứu: *"khi nào feature engineering đủ tốt, khi nào cần học
từ pixel?"* Dự đoán định tính:

- Với data **ít** (vài trăm trial): `force_poly` có thể tốt nhất (variance
  của model thấp, bias chấp nhận được).
- Với data **trung bình** (vài ngàn trial): `force_model` thắng vì giữ được
  cấu trúc spatial của mảng marker mà không cần quá nhiều param.
- Với data **rất nhiều** (>10K trial): `force_cnn` mới phát huy — ImageNet
  pretrain giúp khởi đầu, fine-tune trên đủ data sẽ học được feature *vượt
  hơn* cả detect+track + 4 scalar feature.

### 5.6 Loss + optimizer chung cho cả 3 model

Cả 3 module dùng **cùng** training recipe (chỉ khác model + dataset):

```yaml
loss:        HuberLoss(delta=1.0)
optimizer:   AdamW(lr=5e-3 hoặc 1e-3, weight_decay=1e-2)
scheduler:   CosineAnnealingLR(T_max=num_epochs, eta_min=lr*0.01)
early_stop:  patience=40 epoch (val MAE)
metrics:     MAE, RMSE, R²
split_seed:  42
ratios:      train/val/test = 70/15/15 trial-level
```

Để fair comparison, các tham số này được lock — chỉ tinh chỉnh hyperparam
*nội tại* model (degree cho poly, hidden_per_point cho PointNet, backbone
cho CNN).

---

## Phụ lục A — Mapping nhanh "khái niệm ↔ code"

| Khái niệm lý thuyết | File:line |
|---|---|
| Mô hình lỗ kim + Brown–Conrady | `src/core/calibration.py:18-77` |
| Khử méo + remap | `src/core/calibration.py:91-112` |
| Trừ nền box-blur | `src/core/preprocessing.py:35` |
| CLAHE | `src/core/preprocessing.py:39` |
| 4 filter blob | `src/core/detection.py:42-67` |
| Pyramid LK + tiêu chí dừng | `src/core/tracking.py:45-48` |
| Forward–backward check | `src/core/tracking.py:53-56` |
| Deadzone (apply_deadzone flag) | `src/core/tracking.py:59-62` |
| MRVL có trọng số | `src/slip/v1.py:91-101` |
| Press-rate gate | `src/slip/v1.py:64-71` |
| Rebound filter | `src/slip/v1.py:78-83` |
| EMA bất đối xứng V1 | `src/slip/v1.py:101-104` |
| Translation/Radial decomposition | `src/slip/v2.py:62-89` |
| Multi-scale loop | `src/slip/v2.py:124-161` |
| Short-scale gate | `src/slip/v2.py:163-184` |
| Phase classification | `src/slip/v2.py:220-232` |
| Press-slip ratio gate | `src/slip/v2.py:189-192` |
| Hysteresis Schmitt V2 | `src/slip/v2.py:197-203` |
| Producer-consumer 4-thread | `src/collection/realtime_station.py:513-664` |
| Synchronized snapshot lock | `src/collection/realtime_station.py:521-535` |
| Imada parser | `src/collection/realtime_station.py:79-97` |
| Force zero offset (median 20) | `src/collection/realtime_station.py:366-377` |
| Schema session/trial | `src/collection/dataset.py:47-86` |
| Repro snapshot | `src/collection/dataset.py:203-239` |
| `STEPS_PER_MM = 400` | `src/collection/Pyserial/Pyserial.ino:8-11` |
| Nearest-neighbor force sync | `src/force_model/prepare.py:67-79` |
| Detect-track-cache pipeline | `src/force_model/prepare.py:92-227` |
| Radial/tangential feature | `src/force_poly/features.py:81-83` |
| Polynomial expansion | `src/force_poly/model.py:38-56` |
| Standardize trước expand | `src/force_poly/model.py:137-141` |
| Shared MLP qua Conv1d | `src/force_model/model.py:13-23` |
| Masked max+mean pool | `src/force_model/model.py:31-60` |
| Conv1 inflate ImageNet 3→2 | `src/force_cnn/model.py:59-68` |
| Trial-level split | `src/force_model/dataset.py:39-65` |

## Phụ lục B — Tham số hệ thống (tổng hợp)

Tất cả tham số trong `config/pipeline_config.yaml`. Schema validate ở
`src/config/schema.py` — thiếu key gây `ConfigError` ngay lúc load.

### Tracking + preprocessing + detection

| Tham số | Default | Ý nghĩa |
|---|---|---|
| `preprocessing.blur_kernel` | $[101, 101]$ | Kernel ước lượng nền |
| `preprocessing.clahe_clip_limit` | $2{,}5$ | Cường độ CLAHE |
| `preprocessing.clahe_grid` | $[8, 8]$ | Lưới CLAHE |
| `detection.min_threshold` / `.max_threshold` / `.step` | $50/220/10$ | Quét binarization |
| `detection.min_area` / `.max_area` | $30/500$ | Diện tích blob (px²) |
| `detection.min_circularity` / `.min_inertia` / `.min_convexity` | $0{,}5/0{,}3/0{,}7$ | Hình học blob |
| `tracking.min_displacement` | $1$ | Deadzone (px) |
| `tracking.pyrlk.win_size` | $[21, 21]$ | LK window |
| `tracking.pyrlk.max_level` | $3$ | Pyramid depth |
| `tracking.pyrlk.fb_threshold` | $2{,}0$ | FB error threshold (px) |

### Slip V1

| Tham số | Default | Ý nghĩa |
|---|---|---|
| `slip_detection.slip_threshold` | $0{,}8$ | Ngưỡng $\hat R$ |
| `slip_detection.min_motion_thresh` | $2{,}0$ | $m_{\min}$ (px) |
| `slip_detection.min_moving_markers` | $5$ | Marker tối thiểu |
| `slip_detection.alpha` / `.alpha_decay` | $0{,}3 / 0{,}6$ | EMA up/down |
| `slip_detection.rebound_dot_prod_threshold` | $-0{,}1$ | Loại rebound |
| `slip_detection.press_rate_threshold` | $0{,}5$ | Gate press |
| `slip_detection.history_buffer_length` | $5$ | Buffer (frame) |

### Slip V2

| Tham số | Default | Ý nghĩa |
|---|---|---|
| `slip_v2.scales` | $[1, 3, 9]$ | $\mathcal S$ |
| `slip_v2.min_valid_markers` | $6$ | $|V_k|$ tối thiểu |
| `slip_v2.translation_norm` ($T_0$) | $2{,}0$ | px |
| `slip_v2.translation_min` | $0{,}10$ | px |
| `slip_v2.short_gate_norm` ($N_g$) | $0{,}25$ | px |
| `slip_v2.short_gate_softmin` | $0{,}15$ | – |
| `slip_v2.short_gate_max_scale` ($K_g$) | $3$ | frame |
| `slip_v2.coherence_min` | $0{,}45$ | – |
| `slip_v2.score_on` / `.score_off` | $0{,}55 / 0{,}35$ | Hysteresis |
| `slip_v2.alpha_up` / `.alpha_down` | $0{,}5 / 0{,}7$ | EMA |
| `slip_v2.radial_rate_min` | $0{,}004$ | px/frame |
| `slip_v2.press_slip_ratio` | $0{,}35$ | – |

## Phụ lục C — Tài liệu tham khảo gốc (gợi ý cite)

| Khái niệm | Nguồn |
|---|---|
| Camera calibration (Zhang) | Z. Zhang, *A flexible new technique for camera calibration*, IEEE TPAMI 22(11), 2000. |
| Distortion model | D. C. Brown, *Decentering distortion of lenses*, Photogrammetric Eng. 32(3), 1966. |
| CLAHE | K. Zuiderveld, *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV, 1994. |
| Lucas–Kanade | B. D. Lucas & T. Kanade, *An iterative image registration technique with an application to stereo vision*, IJCAI 1981. |
| Pyramid LK | J.-Y. Bouguet, *Pyramidal implementation of the Lucas–Kanade feature tracker*, Intel Corp., 2001. |
| Forward–Backward error | Z. Kalal, K. Mikolajczyk, J. Matas, *Forward-Backward Error: Automatic Detection of Tracking Failures*, ICPR 2010. |
| Mean resultant length | N. I. Fisher, *Statistical Analysis of Circular Data*, Cambridge UP, 1995. |
| Multi-scale T/R decomposition | (đóng góp của luận văn) |
| Boussinesq Green tensor | K. L. Johnson, *Contact Mechanics*, Cambridge UP, 1985. |
| Inverse FEM tactile | D. Ma et al., *Dense Tactile Force Estimation using GelSlim and Inverse FEM*, ICRA 2019. |
| PointNet | C. R. Qi et al., *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*, CVPR 2017. |
| ResNet | K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. |
| Huber loss | P. J. Huber, *Robust estimation of a location parameter*, Annals Math. Stat. 35(1), 1964. |
| AdamW | I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. |
| Cosine annealing | I. Loshchilov & F. Hutter, *SGDR*, ICLR 2017. |
| GelSight family | W. Yuan, S. Dong, E. H. Adelson, *GelSight: High-Resolution Robot Tactile Sensors*, Sensors 17(12), 2017. |

---

*Tài liệu được biên soạn từ codebase tại nhánh `collection`, commit `e9cd54d`.
Mọi tham số trong YAML đều có thể override qua command-line khi chạy
pipeline. Mọi sửa đổi cấu trúc thuật toán cần cập nhật cả file này lẫn
`src/config/schema.py` để đảm bảo nhất quán.*
