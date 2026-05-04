# Cơ sở lý thuyết hệ thống theo dõi marker xúc giác và phát hiện trượt

> Tài liệu này tổng hợp toàn bộ cơ sở lý thuyết của codebase, phục vụ phần
> "Cơ sở lý thuyết" trong luận văn tốt nghiệp. Mỗi mục giải thích một khối
> chức năng theo trình tự: **động cơ vật lý → mô hình toán học → công thức cụ thể
> → tham số hiện thực trong code → giới hạn / tinh chỉnh thực dụng**.
>
> Ký hiệu file:line dùng để truy ngược về nơi cài đặt tương ứng.

---

## Mục lục

1. [Bối cảnh bài toán và đóng góp của hệ thống](#1-bối-cảnh-bài-toán-và-đóng-góp-của-hệ-thống)
2. [Cảm biến xúc giác kiểu marker quang học — mô hình vật lý](#2-cảm-biến-xúc-giác-kiểu-marker-quang-học--mô-hình-vật-lý)
3. [Hiệu chuẩn camera (Zhang 2000)](#3-hiệu-chuẩn-camera-zhang-2000)
4. [Tiền xử lý ảnh: trừ nền + CLAHE](#4-tiền-xử-lý-ảnh-trừ-nền--clahe)
5. [Phát hiện marker bằng SimpleBlobDetector](#5-phát-hiện-marker-bằng-simpleblobdetector)
6. [Theo dõi quang học bằng Pyramid Lucas–Kanade + Forward–Backward check](#6-theo-dõi-quang-học-bằng-pyramid-lucaskanade--forwardbackward-check)
7. [Phát hiện trượt V1 — Mean Resultant Vector Length](#7-phát-hiện-trượt-v1--mean-resultant-vector-length)
8. [Phát hiện trượt V2 — Phân rã đa thang Translation/Radial](#8-phát-hiện-trượt-v2--phân-rã-đa-thang-translationradial)
9. [Trực quan hóa trường biến dạng](#9-trực-quan-hóa-trường-biến-dạng)
10. [Hệ thống thu thập dữ liệu thực nghiệm](#10-hệ-thống-thu-thập-dữ-liệu-thực-nghiệm)
11. [Ước lượng lực tiếp xúc bằng học máy](#11-ước-lượng-lực-tiếp-xúc-bằng-học-máy)
12. [Tổng hợp tham số toàn hệ thống](#12-tổng-hợp-tham-số-toàn-hệ-thống)
13. [Sơ đồ luồng dữ liệu end-to-end](#13-sơ-đồ-luồng-dữ-liệu-end-to-end)
14. [Tài liệu tham khảo gốc và phương pháp ngụ ý](#14-tài-liệu-tham-khảo-gốc-và-phương-pháp-ngụ-ý)

---

## 1. Bối cảnh bài toán và đóng góp của hệ thống

### 1.1 Phát biểu bài toán

Cảm biến xúc giác (tactile sensor) kiểu camera-based — như **GelSight**, **GelSlim**,
**TacTip**, **DIGIT** — đặt một lớp gel đàn hồi (silicone/elastomer) phía trước
một camera nội bộ. Trên bề mặt gel có một mảng marker (chấm, lưới hoặc gai) được
chiếu sáng bằng LED. Khi gel tiếp xúc với vật thể bên ngoài, lớp gel bị biến
dạng, các marker dịch chuyển trong mặt phẳng ảnh; hệ thống thị giác máy tính có
nhiệm vụ:

- **Bước 1 — Tracking**: ước lượng trường dịch chuyển 2D
  $\mathbf{u}_i = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}$ cho từng marker
  giữa trạng thái tham chiếu (gel ở trạng thái nghỉ) và trạng thái biến dạng.
- **Bước 2 — Slip detection**: phát hiện sự kiện trượt (gross slip / incipient
  slip) dựa trên trường $\{\mathbf{u}_i\}_i$ — đây là biến điều khiển then chốt
  cho grasping ổn định ở robot manipulation.
- **Bước 3 (mở rộng) — Force regression**: ước lượng lực pháp tuyến (và tiếp
  tuyến) tác dụng lên gel dưới dạng scalar/vector — cần để feedback servo control.

### 1.2 Phạm vi cài đặt trong codebase

| Thành phần | File | Vai trò |
|---|---|---|
| Pipeline batch (cặp ảnh tĩnh) | `src/pipelines/batch.py` | Chứng minh thuật toán trên ảnh đã chụp. |
| Pipeline realtime, không slip | `src/pipelines/realtime_tracking.py` | Theo dõi từ webcam, hỗ trợ profiling. |
| Pipeline realtime, slip V1 | `src/pipelines/realtime_slip_v1.py` | Tracking + MRVL slip detector. |
| Pipeline realtime, slip V2 | `src/pipelines/realtime_slip_v2.py` | Tracking + multi-scale slip detector. |
| Trạm thu thập dữ liệu | `src/collection/realtime_station.py` | Đồng bộ camera + force gauge + motor. |
| Mô hình lực | `src/force_poly`, `src/force_model`, `src/force_cnn` | 3 đường đi từ tracking → lực. |

### 1.3 Đóng góp lý thuyết của hệ thống

1. **Tách biệt strict tham số khỏi code** (`src/config/loader.py`,
   `src/config/schema.py`): mọi giá trị thuật toán phải khai báo ở YAML; thiếu
   key → `ConfigError` ngay tại load-time. Nhờ vậy luận văn có thể tổng kết toàn
   bộ tham số trong một bảng (xem §12) mà không cần grep code.
2. **Hai phiên bản slip detector** với hai hệ tiên đề khác nhau: V1 dựa trên
   thống kê vòng tròn của vector vận tốc; V2 phân rã trường biến dạng tích lũy
   theo thành phần tịnh tiến + xuyên tâm trên nhiều thang thời gian — minh hoạ
   trade-off giữa độ trễ phản ứng và khả năng phân biệt nhấn / trượt.
3. **Ba lớp mô hình ước lượng lực** (polynomial regression — PointNet — CNN
   end-to-end) trên cùng một tập dữ liệu, cho phép so sánh fairly giữa
   feature-engineered shallow model và data-driven deep model.

---

## 2. Cảm biến xúc giác kiểu marker quang học — mô hình vật lý

### 2.1 Cấu trúc cảm biến

Một cảm biến marker-based điển hình có 3 lớp: (i) **lớp gel đàn hồi** trong suốt
hoặc bán trong suốt, (ii) **lớp marker** in/đúc trên mặt gel, (iii) **camera +
LED** quan sát từ phía sau gel. Các marker được thiết kế để có độ tương phản
cao trên nền gel: trong codebase này, marker là chấm sáng trên nền tối (xem
`detection.py:52`: `params.blobColor = 255`).

### 2.2 Quan hệ biến dạng – ảnh

Gọi $\mathbf{X}_i \in \mathbb{R}^3$ là vị trí 3D của marker $i$ trên mặt gel ở
trạng thái nghỉ và $\mathbf{X}_i + \mathbf{d}_i$ ở trạng thái biến dạng. Phép
chiếu của camera (đã hiệu chuẩn — xem §3) cho ta điểm ảnh:
$$
\mathbf{p}_i = \pi(\mathbf{K},\mathbf{X}_i),
\qquad
\mathbf{p}_i' = \pi(\mathbf{K},\mathbf{X}_i + \mathbf{d}_i)
$$

Vì gel mỏng (độ dày nhỏ so với khoảng cách camera) và biến dạng diễn ra
chủ yếu trong mặt phẳng song song với cảm biến, ta xấp xỉ:
$$
\mathbf{u}_i \;\equiv\; \mathbf{p}_i' - \mathbf{p}_i \;\approx\; \alpha\,(\mathbf{d}_i)_{xy}
\quad\text{(scale } \alpha \text{ phụ thuộc tiêu cự + khoảng cách)}.
$$

Toàn bộ pipeline làm việc trực tiếp trên trường 2D
$\mathbf{u}_i$ (đơn vị pixel) **không cần** quy đổi sang đơn vị vật lý —
trừ khi muốn ra lực Newton (§11).

### 2.3 Mô hình đàn hồi tuyến tính (rất thô)

Ở chế độ biến dạng nhỏ, gel có thể được xấp xỉ bằng môi trường đàn hồi tuyến
tính đẳng hướng (Hooke). Khi đó trường vận tốc bề mặt $\mathbf{u}(\mathbf{x})$
phụ thuộc vào trạng thái tiếp xúc:

- **Stick (dính)**: marker bị "khóa" cùng vật tiếp xúc, $\mathbf{u}$ phẳng và đồng pha.
- **Press (nhấn)**: trường $\mathbf{u}$ có tính xuyên tâm hướng vào (compress)
  hoặc hướng ra (release), pha với $\mathbf{r}_i$.
- **Slip (trượt)**: $\mathbf{u}$ đồng nhất theo một hướng tịnh tiến — tất cả
  marker dưới vùng tiếp xúc dịch theo cùng vector.

Quan sát này là **nền tảng vật lý** của cả hai detector trong §7–8: hướng
*đồng nhất* ⇒ slip; hướng *xuyên tâm* ⇒ press/release. Hệ thức toán học
được hiện thực hoá qua thống kê vòng tròn (V1) và phân rã least-squares (V2).

---

## 3. Hiệu chuẩn camera (Zhang 2000)

**File**: `src/core/calibration.py` (hàm `calibrate_camera` line 18, `load_calibration` line 80, `undistort_image` line 91).

### 3.1 Mô hình camera lỗ kim

Camera lỗ kim ánh xạ điểm 3D $\mathbf{X} = (X, Y, Z)^\top$ trong hệ camera sang
điểm ảnh $\mathbf{p}$ qua:
$$
s\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \mathbf{K}\,\big[\mathbf{R} \,|\, \mathbf{t}\big] \begin{pmatrix} X^w \\ Y^w \\ Z^w \\ 1 \end{pmatrix},
\qquad
\mathbf{K}=\begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}.
$$
Trong đó $\mathbf{K}$ là ma trận nội tại (intrinsic), $[\mathbf{R}|\mathbf{t}]$
là ngoại tại (extrinsic). Camera thực tế còn có méo radial + tangential, mô tả
bằng vector $(k_1, k_2, p_1, p_2, k_3)$ — Brown–Conrady.

### 3.2 Phương pháp Zhang

Zhang [2000] khai thác bàn cờ phẳng có toạ độ thế giới đã biết: với $M$ ảnh, ta
có $M$ ràng buộc đồng nhất (homography) $\mathbf{H}_m = \mathbf{K}[\mathbf{r}_1\, \mathbf{r}_2\, \mathbf{t}]$.
Tính chất trực giao của $\mathbf{r}_1, \mathbf{r}_2$ cho hai phương trình tuyến
tính trên các phần tử của $\mathbf{B}=\mathbf{K}^{-\top}\mathbf{K}^{-1}$. Với
$M\!\ge\!3$ giải $\mathbf{B}$, decompose Cholesky thành $\mathbf{K}$, rồi
nonlinear refine theo Levenberg–Marquardt với cost reprojection.

### 3.3 Cài đặt trong codebase

```text
checkerboard_size: [8, 6]   # số góc trong (không tính cạnh ngoài)
square_size:       25.0     # mm
criteria:          max_iter=30, eps=0.001
subpix_window:     [11, 11]
```
(`config/pipeline_config.yaml:18`; tham chiếu schema `src/config/schema.py:18`).

`cv2.findChessboardCorners` định vị góc thô; `cv2.cornerSubPix` tinh chỉnh
sub-pixel với cửa sổ 11×11 (`calibration.py:59`). `cv2.calibrateCamera` thực
hiện toàn bộ pipeline Zhang và xuất `mtx, dist, rvecs, tvecs` lưu vào
`config/calib_result.npz` (`calibration.py:66–76`).

`undistort_image` (`calibration.py:91–112`) khử méo bằng:
1. `getOptimalNewCameraMatrix` chọn ma trận mới cân bằng giữa giữ nguyên FOV
   và bỏ vùng đen ngoài rìa, kèm `alpha=1`.
2. `initUndistortRectifyMap` dựng bản đồ pixel → pixel.
3. `cv2.remap` với nội suy bilinear; cuối cùng crop theo `roi`.

### 3.4 Khi nào nên bật calibration

Trong pipeline chính, calibration được bật/tắt qua flag boolean
`pipeline.use_calibration` (`config/pipeline_config.yaml:147`, mặc định
**false**). Lý do: với cảm biến xúc giác, biến dạng quan sát chỉ vài chục pixel
trên cảm biến gần như chính diện — sai số méo radial là $O(0{,}1\%)$ ⇒ tracking
quality không thay đổi đáng kể, nhưng bật calibration sẽ chậm thêm một bước
remap. Calibration vẫn cần thiết khi: (i) ánh xạ pixel ↔ mm cho ước lượng lực
tuyệt đối, (ii) so sánh chéo giữa nhiều cảm biến.

---

## 4. Tiền xử lý ảnh: trừ nền + CLAHE

**File**: `src/core/preprocessing.py` (hàm `preprocess` line 14, `make_clahe` line 43).

### 4.1 Vấn đề: chiếu sáng không đồng đều

LED trong cảm biến tactile thường gây vignetting — vùng giữa sáng hơn rìa, có
khi chênh 30–40% intensity. Nếu áp threshold tuyệt đối, ngưỡng đúng cho vùng
giữa sẽ bỏ sót marker ở rìa và ngược lại.

### 4.2 Trừ nền bằng box blur

Ý tưởng: ước lượng nền bằng một bộ lọc trung bình (box) cửa sổ rất lớn so với
kích thước marker, sau đó trừ:
$$
B(\mathbf{x}) = \frac{1}{|N(\mathbf{x})|}\sum_{\mathbf{y}\in N(\mathbf{x})}I(\mathbf{y}),
\qquad
\tilde{I}(\mathbf{x}) = \max\big(0,\, I(\mathbf{x}) - B(\mathbf{x})\big).
$$
Với $|N| \approx 101 \times 101 \approx 10^4$ pixel, $B$ là ước lượng "DC" của
ảnh; phổ tần số thấp này tương ứng với vignetting LED, còn marker (đối tượng
nhỏ, tần số cao) gần như không bị ảnh hưởng.

**Lý do dùng box blur thay vì Gaussian**: với box, OpenCV dùng *integral image*
(SAT) — độ phức tạp $O(1)$ mỗi pixel, không phụ thuộc kích thước kernel. Gaussian
101×101 separable sẽ mất gấp 10–20× thời gian. Comment trong code
(`preprocessing.py:33-34`) ghi rõ trade-off này. Nhược điểm là box blur có
"grid artifact" tần số cao, nhưng vì kết quả chỉ dùng làm nền nên không ảnh hưởng.

### 4.3 Min–max normalize

`cv2.normalize(..., NORM_MINMAX)` co giãn về `[0, 255]`:
$$
\tilde{I}_{norm}(\mathbf{x}) = 255 \cdot \frac{\tilde{I}(\mathbf{x}) - \min \tilde{I}}{\max \tilde{I} - \min \tilde{I}}.
$$
Bước này khôi phục dynamic range sau khi trừ nền.

### 4.4 CLAHE — Contrast Limited Adaptive Histogram Equalization

CLAHE [Zuiderveld 1994] chia ảnh thành lưới $G \times G$ ô (ở đây
$8 \times 8$, `clahe_grid: [8, 8]`). Trong mỗi ô:
1. Xây histogram local $h[k]$.
2. **Clip** mọi bin $h[k] > c \cdot N_{\text{px}}$ ($c$ = `clip_limit = 2.5`),
   redistribute phần dư đều cho mọi bin → tránh khuếch đại noise ở vùng đồng nhất.
3. Tính CDF $T[k]$ trên histogram đã clip.
4. Áp $T$ tại mỗi pixel trung tâm ô; ở pixel khác dùng nội suy song tuyến giữa 4
   ô lân cận để tránh "tile boundary" artifact.

Kết quả: marker tròn rõ, nền đồng đều, sẵn sàng cho `SimpleBlobDetector`. Trong
realtime pipeline, đối tượng `cv2.CLAHE` được tạo một lần ở `BaseRealtimePipeline.__init__`
(`base.py:34`) và truyền lại qua tham số `_clahe` để tránh re-init mỗi frame
(`preprocessing.py:39`).

### 4.5 Tham số

| Tham số | Giá trị | File:line |
|---|---|---|
| `preprocessing.blur_kernel` | `[101, 101]` | `pipeline_config.yaml:26` |
| `preprocessing.clahe_clip_limit` | `2.5` | `pipeline_config.yaml:27` |
| `preprocessing.clahe_grid` | `[8, 8]` | `pipeline_config.yaml:28` |

---

## 5. Phát hiện marker bằng SimpleBlobDetector

**File**: `src/core/detection.py` (hàm `create_blob_detector` line 42, `detect_markers` line 70).

### 5.1 Định nghĩa "blob"

Trong thị giác máy tính, *blob* là một vùng liên thông có cường độ khác biệt với
môi trường xung quanh. Với marker tactile, blob là vùng sáng đơn liên thông trên
nền tối.

### 5.2 Thuật toán SimpleBlobDetector của OpenCV

`cv2.SimpleBlobDetector` thực hiện:

**Bước 1 — Multi-threshold binarization.** Quét ngưỡng $t$ trong
$[\text{minThreshold}, \text{maxThreshold}]$ (= `[50, 220]`) với bước
$\Delta t = \text{thresholdStep}$ (= `10`). Mỗi giá trị $t$ tạo một mặt nạ nhị
phân $B_t$.

**Bước 2 — Connected component extraction.** Mỗi mặt nạ $B_t$, tìm các vùng
liên thông; tính **centroid** và các đặc trưng hình học.

**Bước 3 — Lọc theo property** (mỗi filter là một bất phương trình):

- **Color**: nếu `filterByColor`, giữ blob có giá trị trung tâm = `blobColor`
  (255 cho marker sáng).
- **Area**: $A \in [A_{\min}, A_{\max}]$. Codebase: `[30, 500]` px²
  (`pipeline_config.yaml:34`).
- **Circularity**:
  $$\mathcal{C} = \frac{4\pi A}{P^2}$$
  ($A$ = diện tích, $P$ = chu vi). Hình tròn lý tưởng $\mathcal{C}=1$, hình
  vuông $\mathcal{C}=\pi/4 \approx 0{,}785$. Ngưỡng: `0.5` (`pipeline_config.yaml:36`).
- **Inertia ratio**: tỉ số $\lambda_{\min}/\lambda_{\max}$ giữa hai trị riêng
  của ma trận quán tính $\mathbf{I}$ của vùng. Hình tròn: `1.0`; vạch dài: `~0`.
  Ngưỡng: `0.3` (`pipeline_config.yaml:37`) — chấp nhận chút biến dạng do
  perspective.
- **Convexity**: $A / A_{\text{convex hull}}$. Với marker tròn, gần `1.0`. Ngưỡng:
  `0.7` (`pipeline_config.yaml:38`).

**Bước 4 — Cluster across thresholds.** Các blob xuất hiện ở nhiều $t$ liên
tiếp được merge dựa trên `minDistBetweenBlobs` (mặc định OpenCV).

**Bước 5 — Output**: list `cv2.KeyPoint`; codebase chuyển sang `(N, 2) float32`
(`detection.py:84`).

### 5.3 Nhận xét

- SimpleBlobDetector kết hợp 4 prior hình học vào một detector duy nhất, tốt
  hơn so với threshold + centroid thuần (vốn không kiểm soát được hình dạng).
- Multi-threshold làm detector robust với chiếu sáng dao động — nhờ vậy
  preprocessing không cần threshold tuyệt đối.
- Limitation: không có sub-pixel localization. Ở §6, Lucas–Kanade sẽ tự động
  tinh chỉnh sub-pixel.

### 5.4 Tham số

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `detection.min_threshold` | `50` | Ngưỡng dưới quét binarization |
| `detection.max_threshold` | `220` | Ngưỡng trên |
| `detection.step` | `10` | Khoảng cách hai ngưỡng |
| `detection.min_area` | `30.0` | Diện tích tối thiểu (px²) |
| `detection.max_area` | `500.0` | Diện tích tối đa (px²) |
| `detection.min_circularity` | `0.5` | Loại blob méo |
| `detection.min_inertia` | `0.3` | Loại blob dài |
| `detection.min_convexity` | `0.7` | Loại blob lồi lõm |

---

## 6. Theo dõi quang học bằng Pyramid Lucas–Kanade + Forward–Backward check

**File**: `src/core/tracking.py` (hàm `track_markers_lk` line 15).

### 6.1 Phương trình quang lưu (optical flow constraint)

Giả sử cường độ ảnh không đổi giữa hai frame $I(\mathbf{x},t) =
I(\mathbf{x}+\mathbf{u}, t+1)$. Khai triển Taylor bậc 1:
$$
I_x u + I_y v + I_t = 0,
\qquad
\nabla I^\top \mathbf{u} + I_t = 0.
$$
Một phương trình hai ẩn — *aperture problem*: chỉ ràng buộc thành phần $\mathbf{u}$
song song gradient.

### 6.2 Lucas–Kanade (LK)

Giả định $\mathbf{u}$ hằng số trên cửa sổ $\Omega$ kích thước $w \times w$.
Tích lũy ràng buộc trên mọi pixel $\mathbf{y}\in\Omega$ và giải bình phương tối
thiểu:
$$
\mathbf{u}^* = \arg\min_{\mathbf{u}} \sum_{\mathbf{y}\in\Omega}
\big(\nabla I(\mathbf{y})^\top \mathbf{u} + I_t(\mathbf{y})\big)^2,
$$
với nghiệm đóng:
$$
\mathbf{u}^* = -\,\mathbf{A}^{-1} \mathbf{b},
\quad
\mathbf{A} = \sum \nabla I \nabla I^\top,
\quad
\mathbf{b} = \sum I_t \,\nabla I.
$$
Điều kiện đảo $\mathbf{A}$: hai trị riêng đều lớn — pixel "góc" (corner). Đó là
lý do LK + Shi–Tomasi/Harris hay đi cùng nhau; ở đây ta thay thế bằng marker do
SimpleBlobDetector cung cấp — mỗi marker bản thân là một blob nhỏ có gradient
đa hướng quanh viền.

### 6.3 Pyramid LK

LK chỉ đúng cho dịch chuyển nhỏ (vài pixel). Để xử lý dịch lớn, dùng tháp ảnh
Gaussian: pyramid có $L+1$ tầng từ $I_0$ (gốc) đến $I_L$ (downsample $2^L$):
1. Khởi tạo $\mathbf{u}^{(L)} = \mathbf{0}$ ở tầng cao nhất.
2. Ở tầng $\ell$, dùng LK tinh chỉnh: $\mathbf{u}^{(\ell)} = 2\mathbf{u}^{(\ell+1)} + \delta\mathbf{u}^{(\ell)}$.
3. Tiếp tục đến tầng $\ell=0$ → $\mathbf{u}^*$.

OpenCV cài đặt thuật toán Bouguet (`cv2.calcOpticalFlowPyrLK`) với
- `winSize = (21, 21)` (`pipeline_config.yaml:43`)
- `maxLevel = 3` (4 tầng: $1, \tfrac12, \tfrac14, \tfrac18$) (`pipeline_config.yaml:44`)
- Tiêu chí dừng iter mỗi tầng: `(EPS=0.01, MAX_ITER=30)` (`tracking.py:48`).

Kích thước $21\!\times\!21$ pixel × hiệu lực $2^3$-fold pyramid ⇒ có thể bám
dịch chuyển ~$84$ pixel, đủ cho biến dạng tactile.

### 6.4 Forward–Backward check (Kalal et al. 2010)

Một marker có thể "trôi" sang vùng khác nếu LK rơi vào local minimum. Giải pháp:
1. Track tiến: $\mathbf{p}^{(0)} \mapsto \tilde{\mathbf{p}}^{(1)}$ (ảnh ref → def).
2. Track lùi: $\tilde{\mathbf{p}}^{(1)} \mapsto \tilde{\mathbf{p}}^{(0)}$ (def → ref).
3. Lỗi FB: $\varepsilon_i = \big\|\tilde{\mathbf{p}}_i^{(0)} - \mathbf{p}_i^{(0)}\big\|_2$.
4. Loại marker có $\varepsilon_i \ge \tau_{\text{FB}}$ (= `2.0` px,
   `tracking.py:56`).

Một marker tracking ổn định phải đi-về cùng một điểm; nếu không, status đặt
**invalid**. Status mask cuối:
$$
\text{valid}_i = \big(\text{status}_f^{(i)}=1\big) \wedge \big(\text{status}_b^{(i)}=1\big) \wedge \big(\varepsilon_i < \tau_{\text{FB}}\big).
$$
(`tracking.py:55–56`).

### 6.5 Deadzone (tracking.min_displacement)

Vật liệu silicone có dao động đàn hồi vi mô — hậu quả là marker "rung" $\pm 1$
pixel ngay cả khi không có tải. Để khử nhiễu này:
$$
\hat{\mathbf{p}}_i^{(1)} =
\begin{cases}
\mathbf{p}_i^{(0)} & \text{nếu } \big\|\mathbf{p}_i^{(1)} - \mathbf{p}_i^{(0)}\big\| < d_{\min}, \\
\mathbf{p}_i^{(1)} & \text{ngược lại}.
\end{cases}
$$
với $d_{\min}=1$ px (`pipeline_config.yaml:41`). Cờ `apply_deadzone`
(`tracking.py:21`) bật/tắt cơ chế này — **batch + slip V1** dùng `True`,
**slip V2** dùng `False` vì cần tín hiệu trượt chậm tích luỹ qua nhiều frame
(xem §8.2).

---

## 7. Phát hiện trượt V1 — Mean Resultant Vector Length

**File**: `src/slip/v1.py` (lớp `SlipDetector`).

### 7.1 Quan sát hiện tượng

Khi gel trượt khỏi vật, **toàn bộ** marker trong vùng tiếp xúc dịch theo cùng
một hướng (vector tịnh tiến). Khi gel chỉ bị nhấn xuống, marker dịch theo
nhiều hướng (xuyên tâm) hoặc dao động không nhất quán. Dấu hiệu của slip vì thế
là **độ đồng nhất hướng** của trường vận tốc.

### 7.2 Định nghĩa toán học (thống kê vòng tròn)

Cho mỗi marker valid $i$, vector vận tốc giữa hai frame là
$\Delta\mathbf{p}_i = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t-h)}$ với $h$ là độ
trễ history (`history_buffer_length = 5`). Ký hiệu độ lớn $m_i = \|\Delta\mathbf{p}_i\|$
và góc $\theta_i = \mathrm{atan2}(\Delta y_i, \Delta x_i)$.

**Mean Resultant Vector Length** (MRVL) có trọng số:
$$
R \;=\; \left|\, \sum_i w_i\, e^{j\theta_i} \right|,
\qquad
w_i = \frac{m_i}{\sum_j m_j}.
$$
Tương đương:
$$
R = \sqrt{\Big(\sum w_i \cos\theta_i\Big)^2 + \Big(\sum w_i \sin\theta_i\Big)^2}.
$$
Ý nghĩa:
- $R \to 1$ ⇔ tất cả $\theta_i$ trùng nhau ⇒ slip thuần.
- $R \to 0$ ⇔ $\theta_i$ phân tán đều ⇒ press / nhiễu.
- Trọng số $w_i \propto m_i$ ưu tiên marker dịch chuyển mạnh — chuẩn hoá so với
  nền nhiễu.

Hiện thực ở `v1.py:91-101`:
```python
angles = np.arctan2(significant_disp[:, 1], significant_disp[:, 0])
weights = valid_mags / np.sum(valid_mags)
sum_cos = np.sum(weights * np.cos(angles))
sum_sin = np.sum(weights * np.sin(angles))
raw_r_value = np.sqrt(sum_cos**2 + sum_sin**2)
mean_direction = np.arctan2(sum_sin, sum_cos)
```

### 7.3 Bộ lọc nhiễu phía trước MRVL

Trước khi tính $R$, ta loại 3 nhóm marker không hữu ích:

**(a) Motion gate**: $m_i > m_{\min}$ với $m_{\min}=$ `min_motion_thresh = 2.0`
px (`v1.py:74-75`). Marker đứng yên đóng góp $\theta_i$ ngẫu nhiên do làm
tròn — có thể giảm $R$ "giả".

**(b) Press-rate gate**: nếu lực nhấn đang tăng nhanh (gel đang bị nén), độ lớn
trung bình của trường biến dạng $\bar{m}_t = \frac{1}{|V|}\sum_{i\in V}\|\mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}\|$
sẽ tăng $\Delta\bar{m} = \bar{m}_t - \bar{m}_{t-1}$. Nếu $\Delta\bar{m} >$
`press_rate_threshold = 0.5` px/frame, ép $R \to 0$ và phase = `"pressing"`
(`v1.py:64-71`). Đây là cơ chế phát hiện cấp 0 (zero-th order) chống false
positive khi siết tay gắp.

**(c) Rebound gate**: marker "hồi phục đàn hồi" (đang quay về vị trí cân bằng
sau khi vật bị nhấc) có vận tốc $\Delta\mathbf{p}_i$ ngược chiều biến dạng
$\mathbf{D}_i = \mathbf{p}_i^{(t)}-\mathbf{p}_i^{(0)}$. Loại nếu
$\langle \Delta\mathbf{p}_i, \mathbf{D}_i \rangle < $
`rebound_dot_prod_threshold = -0.1` (`v1.py:78-83`). Lý giải vật lý: khi gel
nhả ra, các marker "đàn hồi" về tâm — đó không phải slip.

### 7.4 Số marker tối thiểu

Sau các lọc trên, đếm $|V_{\text{moving}}|$. Nếu $< $
`min_moving_markers = 5` (`v1.py:87`), trả về phase `"insufficient_motion"` —
không có đủ ràng buộc để tin cậy $R$.

### 7.5 Làm trơn EMA bất đối xứng

$R$ thô dao động từng frame. Ta áp filter EMA hai hằng số:
$$
\hat{R}_t = \begin{cases}
\alpha\, R_t + (1-\alpha)\,\hat{R}_{t-1} & \text{nếu } R_t \ge \hat{R}_{t-1} \\
\alpha_d\, R_t + (1-\alpha_d)\,\hat{R}_{t-1} & \text{nếu } R_t < \hat{R}_{t-1}
\end{cases}
$$
với $\alpha = $ `0.3`, $\alpha_d = $ `0.6` (`v1.py:101-104`).

Triết lý: **tăng chậm, giảm nhanh** —
- $\alpha$ nhỏ ⇒ cần nhiều frame xác nhận trước khi $\hat R$ vượt ngưỡng (chống
  false positive).
- $\alpha_d$ lớn ⇒ khi slip dừng, $\hat R$ rơi nhanh (chống "stuck-in-slip" sau
  khi sự kiện kết thúc).

Phân loại cuối:
$$
\text{is\_slip}_t = \big[\hat{R}_t > \tau_{\text{slip}}\big],
\qquad \tau_{\text{slip}} = 0{,}8.
$$

### 7.6 Decay khi không đủ điều kiện

Khi gate (a)–(c) trả về sớm, $\hat R$ vẫn được cập nhật:
$$
\hat R_t = \hat R_{t-1} \cdot (1 - \alpha_d)
$$
(`v1.py:115-116`). Nhờ đó nếu trước đó đang slip nhưng hiện tại bị "press"
gate chặn, $\hat R$ vẫn rơi xuống dưới ngưỡng và slip-flag được tắt.

### 7.7 Hạn chế của V1

- Sử dụng **velocity giữa hai frame** — slip rất chậm (≪ 1 px/frame) bị deadzone
  và motion-gate bóp về 0 trước khi tích luỹ.
- Chỉ một thang thời gian (= history buffer length).
- Press-rate gate có thể che lấp slip xảy ra **đồng thời** với press (slip
  trong khi đang siết).

V2 ở §8 giải quyết cả ba.

### 7.8 Tham số

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `slip_detection.slip_threshold` | `0.8` | Ngưỡng $\hat R$ phân loại slip |
| `slip_detection.min_motion_thresh` | `2.0` | $m_{\min}$ (px) |
| `slip_detection.min_moving_markers` | `5` | Số marker tối thiểu |
| `slip_detection.alpha` | `0.3` | EMA tăng |
| `slip_detection.alpha_decay` | `0.6` | EMA giảm |
| `slip_detection.rebound_dot_prod_threshold` | `-0.1` | Loại rebound |
| `slip_detection.press_rate_threshold` | `0.5` | Gate press |
| `slip_detection.history_buffer_length` | `5` | Số frame buffer |

---

## 8. Phát hiện trượt V2 — Phân rã đa thang Translation/Radial

**File**: `src/slip/v2.py` (lớp `SlipDetectorV2`).

### 8.1 Đặt vấn đề

V1 dùng **velocity** $\Delta\mathbf{p}_i$ giữa frame $t$ và $t-h$. Slow slip
sinh velocity xấp xỉ 0 mỗi frame; tích phân theo thời gian cũng tan biến vì
deadzone. V2 thay đổi mô hình: thao tác trên trường **biến dạng tích lũy**
$$
\mathbf{D}_i(t) = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(0)}
$$
(không có deadzone — pipeline V2 gọi `track_markers_lk(..., apply_deadzone=False)`,
xem `realtime_slip_v2.py:75`).

### 8.2 Đa thang thời gian (multi-scale)

Ta xét sai phân $\Delta\mathbf{D}$ qua nhiều khoảng $k$:
$$
\Delta\mathbf{D}_i^{(k)}(t) = \mathbf{D}_i(t) - \mathbf{D}_i(t-k),
\qquad k \in \mathcal{S} = \{1, 3, 9\}.
$$
($\mathcal{S}$ = `slip_v2.scales`, `pipeline_config.yaml:62`).
Scale $k=1$ phản ứng nhanh; $k=9$ đủ dài để tích luỹ slip chậm dưới
ngưỡng noise. History buffer dài $\max(\mathcal{S}) + 2$ frame
(`v2.py:38`).

### 8.3 Phân rã least-squares (translation + radial)

Với mỗi scale $k$, ký hiệu tập marker hợp lệ ở cả hai mốc thời gian là $V_k$.
Định nghĩa:
- **Centroid** $\mathbf{c} = \frac{1}{|V_k|}\sum_{i\in V_k}\mathbf{p}_i^{(0)}$
  — luôn lấy theo trạng thái nghỉ (bất biến qua thời gian) (`v2.py:120-122`).
- **Vector xuyên tâm** $\mathbf{r}_i = \mathbf{p}_i^{(0)} - \mathbf{c}$.

Chúng ta xấp xỉ trường $\Delta\mathbf{D}^{(k)}$ trên $V_k$ bằng:
$$
\boxed{\;\Delta\mathbf{D}_i \;=\; \mathbf{t} \;+\; s\cdot \mathbf{r}_i \;+\; \boldsymbol{\rho}_i\;}
$$
- $\mathbf{t}\in\mathbb{R}^2$: thành phần **tịnh tiến đồng nhất** = slip vector.
- $s\in\mathbb{R}$: hệ số **xuyên tâm** đồng nhất; $s>0$ ⇔ giãn (release), $s<0$ ⇔ nén (press).
- $\boldsymbol{\rho}_i$: dư = thành phần shear / xoay / nhiễu cục bộ.

**Nghiệm bình phương tối thiểu** (decoupled vì $\sum \mathbf{r}_i = 0$ nhờ định nghĩa qua centroid):
$$
\mathbf{t} = \frac{1}{|V_k|}\sum_{i\in V_k}\Delta\mathbf{D}_i,
\qquad
s = \frac{\sum_i \langle \mathbf{r}_i,\, \Delta\mathbf{D}_i - \mathbf{t}\rangle}{\sum_i \|\mathbf{r}_i\|^2}.
$$
Cài đặt ở `v2.py:62-89` (hàm `_decompose`).

### 8.4 Năng lượng từng thành phần

Định nghĩa năng lượng (Frobenius² qua tập $|V_k|$ marker):
$$
E_{\text{slip}} = |V_k|\cdot \|\mathbf{t}\|^2,\quad
E_{\text{press}} = s^2 \sum_i \|\mathbf{r}_i\|^2,\quad
E_{\text{resid}} = \sum_i \|\boldsymbol{\rho}_i\|^2.
$$

**Slip fraction**:
$$
\text{slip\_fraction} = \frac{E_{\text{slip}}}{E_{\text{slip}} + E_{\text{resid}} + \epsilon}
\in (0, 1].
$$
Phép chia loại $E_{\text{press}}$ ⇒ slip fraction phản ánh mức độ phần trăm năng
lượng "không-press" thực sự là tịnh tiến đồng nhất, không phải nhiễu cục bộ.

### 8.5 MRVL trên trường đã khử radial

Sau khi loại thành phần xuyên tâm, ta có
$\widetilde{\Delta\mathbf{D}}_i = \Delta\mathbf{D}_i - s\mathbf{r}_i$. Áp lại
MRVL có trọng số (như §7) trên $\widetilde{\Delta\mathbf{D}}$ → coherence $R\in[0,1]$.

Ưu điểm: ngay cả khi đang nhấn xuống (s lớn), nếu có thêm slip nhỏ cộng vào,
$\widetilde{\Delta\mathbf{D}}$ vẫn cho coherence cao — V2 không bị che bởi press
như V1.

### 8.6 Score per-scale

Với mỗi scale $k$ vượt qua các điều kiện cứng:
- $|V_k| \ge $ `min_valid_markers = 6`,
- $\|\mathbf{t}\| \ge $ `translation_min = 0.10` px,
- $R \ge$ `coherence_min = 0.45`,

tính:
$$
\text{score}^{(k)} = \min\!\Big(\frac{\|\mathbf{t}\|}{T_0},\,1\Big) \cdot \text{slip\_fraction} \cdot R,
\qquad T_0 = \text{`translation\_norm` = 2.0 px}.
$$
Ba thừa số là độc lập về ý nghĩa: *cường độ tịnh tiến → tỉ lệ thuộc về slip → tính
đồng nhất hướng*. Tích chỉ cao khi cả ba lớn.

### 8.7 Short-scale gate (chống stuck-in-slip)

Vấn đề thực tế: sau khi slip kết thúc, $\Delta\mathbf{D}^{(9)}$ vẫn còn năng
lượng tịnh tiến (do tích lũy 9 frame trước đó), khiến score scale dài giảm
chậm. Để score "rơi nhanh" theo đúng signal phía trước, V2 dùng **short-scale
gate**:
$$
g = \max\!\left(\;\text{softmin},\; \max_{k\le K_g} \min\!\Big(\frac{\|\mathbf{t}^{(k)}\|}{N_g\sqrt{k}},\,1\Big)\;\right),
$$
- $K_g = $ `short_gate_max_scale = 3`,
- $N_g = $ `short_gate_norm = 0.25` px,
- `softmin = short_gate_softmin = 0.15`.

Áp dụng: score tại scale $k > K_g$ bị nhân với $g$. Khi slip dừng, scale $\le K_g$
phản ứng nhanh ⇒ $g \to $ softmin ⇒ score scale dài bị "kéo xuống" đồng thời.
Cài đặt: `v2.py:163-184`.

### 8.8 Phân loại phase từ radial rate

Chuẩn hoá hệ số xuyên tâm theo scale:
$$
\dot s^{(k)} = s^{(k)} / \max(k, 1).
$$
Phase được lấy từ scale ngắn nhất có dữ liệu (`v2.py:220-232`):
- $\dot s > $ `radial_rate_min = 0.004` ⇒ `"pressing"`,
- $\dot s < -$ `radial_rate_min` ⇒ `"releasing"`,
- |else| ⇒ `"stick"`,
- không có scale nào thoả ⇒ `"idle"`.

Khi phase là pressing/releasing, V2 gate kết quả: **chỉ giữ slip nếu**
$E_{\text{slip}}/E_{\text{press}} > $ `press_slip_ratio = 0.35` (`v2.py:189-192`).
Cách đọc: trong lúc nhấn, slip phải đủ mạnh so với press mới được công nhận.

### 8.9 Hysteresis + EMA bất đối xứng

Score được EMA với hai hằng số:
- $\alpha_{\uparrow} = $ `alpha_up = 0.5` (tăng vừa)
- $\alpha_{\downarrow} = $ `alpha_down = 0.7` (giảm nhanh hơn)

Cờ slip dùng hysteresis Schmitt:
$$
\text{is\_slip} =
\begin{cases}
\text{True} & \text{nếu False và } \hat S > \tau_{\text{on}} = 0.55 \\
\text{False} & \text{nếu True và } \hat S < \tau_{\text{off}} = 0.35
\end{cases}
$$
(`v2.py:197-203`). Hai ngưỡng tách biệt loại jitter quanh điểm chuyển trạng
thái — cốt yếu khi dùng làm trigger điều khiển grasp.

### 8.10 So sánh V1 vs V2

| Khía cạnh | V1 | V2 |
|---|---|---|
| Tín hiệu nguồn | velocity giữa 2 frame | biến dạng tích luỹ vs ref |
| Thang thời gian | 1 (history_buffer) | nhiều (`scales`) |
| Tách press / slip | press-rate gate (fail khi cùng xảy ra) | phân rã LS hoàn toàn |
| Slow slip | bị deadzone & motion-gate triệt | bắt được nhờ scale dài |
| Latency phản ứng tắt | 1 hằng số EMA | EMA + short-scale gate |
| Phân loại phase | binary slip/non-slip | slipping/pressing/releasing/stick |
| Tham số cấu hình | 9 | 14 |

---

## 9. Trực quan hóa trường biến dạng

**File**: `src/core/visualization.py` (`visualize_flow_arrows` line 11);
`src/common/overlay.py` (HUD slip V2).

### 9.1 Mũi tên dịch chuyển

Cho mỗi marker valid, vẽ một mũi tên có:
- Điểm gốc: $\mathbf{p}_i^{(0)}$ (vị trí tham chiếu).
- Điểm ngọn: $\mathbf{p}_i^{(0)} + \alpha \cdot (\mathbf{p}_i^{(t)}-\mathbf{p}_i^{(0)})$,
  $\alpha = $ `arrow_scale = 3.0` (phóng đại để dễ quan sát).
- Màu sắc: gradient đỏ→vàng theo độ lớn $m_i$:
  $\text{color\_intensity} = \min(m_i \cdot k_c, 255)$, $k_c = $
  `color_intensity_multiplier = 20`. Mũi tên = `(0, 255-c, c)` BGR ⇒ vector
  dài là đỏ rực, vector ngắn là vàng nhạt. Logic ở `visualization.py:40-51`.
- Marker không di chuyển ($m_i < $ `motion_magnitude_threshold = 0.3`) chỉ vẽ
  chấm xám tránh rối hình.

Mục đích: làm cho người vận hành nhìn thấy **trường biến dạng dạng vector**
tương đương như particle image velocimetry (PIV) — debug nhanh stick/slip/press
mà không cần đọc số.

### 9.2 HUD slip V2

Slip V2 có hai widget tổng quát phản ánh trực tiếp các đại lượng mô hình:

- **HUD arrow** (`overlay.py:68-92`): mũi tên ở góc dưới-phải vẽ vector $\mathbf{t}$
  đã ước lượng (translation), scale = `hud_arrow.scale = 20`. Khi
  $\|\mathbf{t}\|$ nhỏ hơn `min_magnitude = 0.05`, không vẽ. Màu mũi tên = màu
  phase (xanh khi tracking, vàng alert, đỏ slip…).

- **Score bar** (`overlay.py:95-121`): thanh ngang biểu diễn $\hat S$, kèm hai
  vạch đứng tại $\tau_{\text{on}}, \tau_{\text{off}}$ — tương ứng với
  hysteresis trong §8.9. Người dùng có thể tinh chỉnh ngưỡng trên thực địa.

---

## 10. Hệ thống thu thập dữ liệu thực nghiệm

**Files**:
- `src/collection/realtime_station.py` — GUI + đa luồng (camera, force, motor, writer).
- `src/collection/dataset.py` — schema folder data (session/trial), helper repro.
- `src/collection/Pyserial/Pyserial.ino` — firmware Arduino điều khiển stepper.

### 10.1 Mục tiêu

Để huấn luyện mô hình ước lượng lực (§11) hoặc đánh giá slip detector trên
ground truth, cần dữ liệu đồng bộ giữa **camera tactile**, **lực tham chiếu**
(Imada ZTA) và **vị trí indenter** (motor tuyến tính). Hệ thống xây dựng theo
mô hình **producer-consumer** đa luồng, với 3 producer (camera, force, motor)
và 1 consumer ghi đĩa.

### 10.2 Đồng bộ thời gian

Mỗi sensor có thread riêng và gắn timestamp $t_{\text{mono}} = $
`time.monotonic()` ngay khi sample đến. Không có "join" ở write-time:
- **Camera**: đọc liên tục → snap $(t, \text{frame})$ → push queue khi recording.
- **Imada**: poll 20 Hz qua serial 19200 bps → parse số float → push queue.
- **Motor**: cmd/resp Arduino qua serial 115200 → push queue.
- **Writer**: block trên `recording_event`; mỗi iter pull từ 3 queue, ghi
  ảnh + 3 CSV (frames.csv / force_log.csv / motor_log.csv) (`realtime_station.py:631-700`).

Downstream training (`force_model/prepare.py:67-79`) tự **nearest-neighbor
join** force vào frame timestamp với tolerance $0.1$ s, frame nào ngoài tolerance bị skip.

### 10.3 Cấu trúc lưu trữ

```
data/sessions/<session_id>/
  session.yaml                         # metadata camera + force_zero_offset + git_sha
  reference/ref_<ts>.jpg               # ảnh tham chiếu chung cho cả session
  trials/trial_NNN/
    trial.yaml                         # tags, motor_state_at_start, n_frames…
    frames/frame_<ts>.jpg              # JPEG quality 95
    frames.csv                         # ts_mono, ts_wall, image_name
    force_log.csv                      # ts_mono, force_n
    motor_log.csv                      # ts_mono, direction, payload
```
Định nghĩa schema ở `dataset.py:47-86`. Mỗi session có **force zero offset**
được lấy median 20 sample lúc tạo session (`realtime_station.py:366-377`) để
trừ bias DC của Imada.

### 10.4 Repro snapshot

`session.yaml` lưu kèm:
- `repro.git_sha` từ `git rev-parse HEAD` (`dataset.py:203-213`).
- `repro.pip_versions` cho `opencv-python`, `numpy`, `pyserial`, `customtkinter`,
  `pyyaml` (`dataset.py:224-239`).
- `hardware.firmware_sha256` của file `Pyserial.ino` đang dùng — phát hiện ngay
  khi firmware bị thay đổi giữa các session.

Đảm bảo dữ liệu **truy vết được** — tiêu chuẩn FAIR cho dataset khoa học.

### 10.5 Giao thức Arduino

Firmware (`Pyserial.ino`) dùng `AccelStepper`. Quy đổi:
$$
\text{steps\_per\_mm} = \frac{\text{micro\_step} \times (360^\circ/\theta_{\text{step}})}{\text{mm\_per\_rev}}
= \frac{16 \cdot 200}{8} = 400.
$$

Lệnh:
- `f <mm>` → `stepper.move(+steps)`.
- `b <mm>` → `stepper.move(-steps)`.
- `s` → `stepper.stop()` (decel mềm).

Phản hồi: `M: Forward X mm` / `M: Backward X mm` / `M: Stopped` /
`ERR: invalid distance` / `ERR: unknown cmd`. Pipeline trên Python parse các
phản hồi này để cập nhật `motor_state` ∈ {idle, moving, unknown}
(`realtime_station.py:611-617`).

---

## 11. Ước lượng lực tiếp xúc bằng học máy

Codebase cung cấp **ba** đường đi từ dữ liệu thu thập sang ước lượng lực
scalar (Newton). Mỗi mô hình lấy input ở một mức độ trừu tượng khác nhau:

| Module | Input | Mô hình | Số tham số (mặc định) |
|---|---|---|---|
| `force_poly` | 9 scalar feature từ trường disp | Polynomial regression bậc 2 | 55 |
| `force_model` | $(N, 4)$ feature mỗi marker | PointNet (shared MLP + masked pool) | ~25K |
| `force_cnn` | 2-channel image $[\text{ref}, \text{def}]$ | ResNet-18 (hoặc small CNN) | ~11M / 110K |

Tất cả huấn luyện dưới regression scalar với mất mát Huber (delta=1) và
optimizer AdamW + cosine annealing.

### 11.1 Tiền xử lý chung — `force_model/prepare.py`

Trước khi train bất cứ mô hình nào, `prepare.py` quét toàn bộ
`data/sessions/<>/trials/<>` và:
1. Detect marker trên reference của session.
2. Track LK (apply_deadzone=False) cho mọi frame của trial → $(T, N, 2)$ disp + valid mask.
3. Sync force theo nearest-neighbor (tolerance $0.1$ s).
4. Lưu compressed `.npz` cho mỗi trial (`prepare.py:200-227`).

Output:
```text
ref_pts (N,2), disp (T,N,2), valid (T,N), force (T,), force_raw (T,),
ts_mono (T,), image_w, image_h, n_markers, trial_id, session_id, force_zero_offset_n
```
Bằng cách cache offline, mọi mô hình share cùng phép detect+track ⇒ so sánh fair.

### 11.2 PolynomialRegressor (`force_poly`)

#### 11.2.1 Feature engineering

`features.py` định nghĩa 9 scalar đặc trưng (`FEATURE_NAMES_V1`):

| Tên | Ý nghĩa vật lý |
|---|---|
| `disp_mag_mean` | mức biến dạng trung bình |
| `disp_mag_max` | peak biến dạng (chỉ điểm tải) |
| `disp_mag_std` | tính không đồng nhất |
| `dx_mean`, `dy_mean` | hướng tải tổng hợp |
| `disp_mag_sum_norm` | tổng biến dạng / N_valid |
| `radial_disp_mean` | nén/giãn quanh tâm grid |
| `tangential_disp_mean` | shear/xoắn |
| `valid_ratio` | độ phủ marker |

Toạ độ và disp đều normalize theo image width $W$ ⇒ feature scale-invariant
trên độ phân giải; tâm grid lấy theo các marker valid tránh lệch khi mất
tracking (`features.py:66-83`).

Radial / tangential dùng vector đơn vị xuyên tâm $\hat{\mathbf r}_i = \mathbf{r}_i/\|\mathbf{r}_i\|$:
$$
\text{radial}_i = \langle \mathbf{u}_i, \hat{\mathbf r}_i\rangle,
\qquad
\text{tangential}_i = \hat{r}_{i,x}\cdot u_{i,y} - \hat{r}_{i,y}\cdot u_{i,x}
$$
(component thứ hai là cross product 2D).

#### 11.2.2 Polynomial expansion

Cho $D$ feature đầu vào, expand thành mọi monomial bậc $\le d$:
$$
\Phi(\mathbf{z}) = \big(\,1,\, z_1,\, z_2,\, \dots,\, z_1^2,\, z_1z_2,\, \dots,\, z_D^d\,\big)^\top \in \mathbb{R}^{P}
$$
với $P = \binom{D+d}{d}$. Cài đặt tổ hợp ở `model.py:25-35` qua
`itertools.combinations_with_replacement`. Default $D=9, d=2 \Rightarrow P=55$.

#### 11.2.3 Standardize trước, expand sau

Mean–std học từ dữ liệu train (`fit_scaler`, `model.py:122-135`):
$$
\tilde z_d = \frac{z_d - \mu_d}{\sigma_d + \epsilon}.
$$
Nếu **không** standardize, hệ số bậc cao bị blow-up vì
$z_d^d$ phụ thuộc bậc $d$ của scale ⇒ optimizer không hội tụ. Đây là lý do
*scaler được lưu vào state\_dict* qua `register_buffer` để serialize cùng
checkpoint.

#### 11.2.4 Head

- **Linear head** (`head="linear"`): $\hat F = \mathbf{w}^\top \Phi(\tilde{\mathbf z})$
  — polynomial regression thuần, chỉ $P$ tham số (không bias vì bias đã có ở
  monomial bậc 0).
- **Mlp_small head**: $\Phi \to \text{Linear}(P, 16)\to\text{ReLU}\to\text{Dropout}\to\text{Linear}(16, 1)$
  — vẫn coi $\Phi$ là feature, nhưng có thêm khả năng phi tuyến trên không gian
  monomial.

#### 11.2.5 Loss + optimizer

- $\mathcal{L} = $ HuberLoss($\delta = 1.0$): bình phương ở $|e|<\delta$, tuyến tính
  ngoài → robust với outlier (`train.py:160`).
- AdamW (`lr = 5e-3`, `weight_decay = 1e-2`) + Cosine annealing.
- Early stop: 40 epoch không cải thiện val MAE.
- Trial-level split (70/15/15, seed=42) tránh leakage frame liền kề.

#### 11.2.6 Vai trò trong luận văn

`force_poly` là **baseline diễn giải được**. Hệ số polynomial sau khi train có
thể đọc trực tiếp ý nghĩa vật lý: bậc 1 tương ứng tuyến tính lực-biến dạng (định
luật Hooke), bậc 2 cho hiệu ứng nén bậc cao. Nó là điểm tham chiếu để phán xét
xem mô hình deep learning có *đáng* hay không.

### 11.3 ForceNet — PointNet-style (`force_model`)

#### 11.3.1 Đặc trưng đầu vào

Mỗi marker → vector 4 chiều
$\mathbf{f}_i = (x_i/W,\, y_i/H,\, \Delta x_i/W,\, \Delta y_i/W)$. Padding tới
$N_{\max} = \max_t N(t)$ qua tất cả trial (`dataset.py:68-74`), kèm boolean
mask đánh dấu marker thật. Augmentation (`dataset.py:166-198`): flip ngang/dọc,
xoay $\le 5^\circ$, noise $0{,}5$ px lên disp.

#### 11.3.2 Kiến trúc PointNet

Trường marker là **set không có thứ tự** — model phải bất biến hoán vị. Theo
PointNet [Qi et al. 2017]:

1. **Shared MLP** $\phi: \mathbb{R}^4 \to \mathbb{R}^{128}$, áp riêng cho từng
   marker (cài đặt qua `Conv1d(kernel=1)`, `model.py:13-23`).
2. **Masked symmetric pooling**: với $f_i = \phi(\mathbf{f}_i)$, ta lấy đồng
   thời max và mean qua các marker hợp lệ:
   $$
   g = \big[\max_{i\in V} f_i \,\|\, \tfrac{1}{|V|}\sum_{i\in V} f_i\big]
   \in \mathbb{R}^{2\cdot 128}
   $$
   (`model.py:31-60`). Max-pool capture *peak feature* (marker dịch nhiều nhất);
   mean capture *trung bình toàn cục*. Concat hai phép pool tăng biểu diễn
   trong khi vẫn permutation-invariant.
3. **Head MLP** $\psi: \mathbb{R}^{256}\to\mathbb{R}^{64}\to\mathbb{R}\to\hat F$.

#### 11.3.3 Tính chất quan trọng

- **Mask-aware**: invalid marker bị set $-\infty$ trước max và bỏ khỏi sum/count
  trước mean → tránh gradient leak sang điểm pad.
- **Permutation-invariant**: shared MLP + symmetric pooling ⇒ kết quả không phụ
  thuộc thứ tự marker, phù hợp khi detector trả keypoints theo thứ tự ngẫu
  nhiên qua các session.
- **Compact**: ~25K params — chạy được trên CPU realtime.

### 11.4 ForceCNN — End-to-end (`force_cnn`)

#### 11.4.1 Input

2-channel image $\big[I_{\text{ref}}, I_{\text{def}}\big]$ resize về $240\!\times\!320$,
normalize $[0,1]$. Không cần detect/track — model học trực tiếp từ pixel.
Augmentation đồng bộ giữa hai channel (cùng flip / rotate / brightness) để giữ
quan hệ vật lý (`dataset.py:224-251`).

#### 11.4.2 Kiến trúc backbone

Hai lựa chọn:

**(a) ResNet-18** (mặc định), nhưng `conv1` ban đầu của ImageNet là $3\times 64\times 7\times 7$.
Codebase **inflate** sang 2 channel bằng cách lấy mean qua chiều RGB rồi tile:
$$
W^{(\text{new})}_{:,c,:,:} = \frac{1}{3}\sum_{c'=1}^{3} W^{(\text{old})}_{:,c',:,:}
\quad\forall c\in\{0,1\}.
$$
(`model.py:59-68`). Trick này giữ nguyên các filter edge/blob mà ImageNet đã
học, ngay cả khi domain mới không phải RGB — quan trọng vì dữ liệu tactile
khan hiếm.

**(b) SmallCNN** (~110K params): 4 block conv-bn-relu-pool, GAP cuối cùng. Dùng
debug nhanh trên CPU.

Head: `Linear(out_dim, 128) → ReLU → Dropout(0.2) → Linear(128, 1)`.

#### 11.4.3 Vai trò luận văn

`force_cnn` là *upper-bound* nếu data phong phú: học cả feature lẫn regression
end-to-end. Nhược điểm: cần nhiều dữ liệu, khó diễn giải, latency lớn (~200 MB
weights).

### 11.5 So sánh ba mô hình

| Tiêu chí | `force_poly` | `force_model` | `force_cnn` |
|---|---|---|---|
| Input | 9 scalar | $(N, 4)$ + mask | 2-channel image |
| Số param | 55 | ~25K | ~11M |
| Cần preprocess (detect+track) | Có | Có | Không |
| Diễn giải | Cao | Trung bình | Thấp |
| Cần GPU | Không | Tuỳ | Có |
| Augmentation hữu hiệu | ít | rotate/noise | flip/rotate/photometric |

Trong luận văn, có thể tổ chức chương ước lượng lực thành 3 mục con và đặt câu
hỏi nghiên cứu: *"khi nào feature engineering đủ tốt, khi nào cần học từ pixel?"*.

---

## 12. Tổng hợp tham số toàn hệ thống

| Module | Tham số (đường dẫn YAML) | Default | Đơn vị / kiểu |
|---|---|---|---|
| Camera | `camera.device_id` | `2` | int |
| | `camera.warmup_frames` | `15` | frame |
| Calibration | `calibration.checkerboard_size` | `[8, 6]` | góc |
| | `calibration.square_size` | `25.0` | mm |
| | `calibration.criteria.max_iter` | `30` | iter |
| | `calibration.criteria.eps` | `0.001` | – |
| | `calibration.subpix_window` | `[11, 11]` | px |
| Preprocess | `preprocessing.blur_kernel` | `[101, 101]` | px |
| | `preprocessing.clahe_clip_limit` | `2.5` | – |
| | `preprocessing.clahe_grid` | `[8, 8]` | ô |
| Detection | `detection.min_threshold` | `50` | gray |
| | `detection.max_threshold` | `220` | gray |
| | `detection.step` | `10` | gray |
| | `detection.min_area` | `30` | px² |
| | `detection.max_area` | `500` | px² |
| | `detection.min_circularity` | `0.5` | – |
| | `detection.min_inertia` | `0.3` | – |
| | `detection.min_convexity` | `0.7` | – |
| Tracking | `tracking.min_displacement` | `1` | px |
| | `tracking.pyrlk.win_size` | `[21, 21]` | px |
| | `tracking.pyrlk.max_level` | `3` | tầng |
| | `tracking.pyrlk.fb_threshold` | `2.0` | px |
| | `tracking.pyrlk.term_criteria.max_iter` | `30` | iter |
| | `tracking.pyrlk.term_criteria.eps` | `0.01` | – |
| Slip V1 | `slip_detection.slip_threshold` | `0.8` | – |
| | `slip_detection.min_motion_thresh` | `2.0` | px |
| | `slip_detection.min_moving_markers` | `5` | marker |
| | `slip_detection.alpha` | `0.3` | – |
| | `slip_detection.alpha_decay` | `0.6` | – |
| | `slip_detection.rebound_dot_prod_threshold` | `-0.1` | px² |
| | `slip_detection.press_rate_threshold` | `0.5` | px/frame |
| | `slip_detection.history_buffer_length` | `5` | frame |
| | `slip_detection.moving_count_thresh` | `2` | marker |
| Slip V2 | `slip_v2.scales` | `[1, 3, 9]` | frame |
| | `slip_v2.min_valid_markers` | `6` | marker |
| | `slip_v2.translation_norm` ($T_0$) | `2.0` | px |
| | `slip_v2.translation_min` | `0.10` | px |
| | `slip_v2.short_gate_norm` ($N_g$) | `0.25` | px |
| | `slip_v2.short_gate_softmin` | `0.15` | – |
| | `slip_v2.short_gate_max_scale` ($K_g$) | `3` | frame |
| | `slip_v2.coherence_min` | `0.45` | – |
| | `slip_v2.score_on` | `0.55` | – |
| | `slip_v2.score_off` | `0.35` | – |
| | `slip_v2.alpha_up` | `0.5` | – |
| | `slip_v2.alpha_down` | `0.7` | – |
| | `slip_v2.radial_rate_min` | `0.004` | – |
| | `slip_v2.press_slip_ratio` | `0.35` | – |
| Pipeline | `pipeline.use_calibration` | `false` | bool |

Tham số visualization (text positions, colors, slip-bar size…) đã được liệt kê
chi tiết trong `src/config/schema.py:69-126`; chúng không ảnh hưởng đến thuật
toán nên không lặp lại ở đây.

---

## 13. Sơ đồ luồng dữ liệu end-to-end

### 13.1 Batch mode (`python -m src.main`)

```
ref.jpg ──► Read gray ──► (Optional) undistort ──┐
                                                   ├─► preprocess (blur sub + CLAHE) ──┐
def.jpg ──► Read gray ──► (Optional) undistort ──┘                                     │
                                                                                       ▼
                                                                            detect blobs (ref)
                                                                                       │
                                              ref_pts ◄──────────────────────────────┘
                                                  │
                                                  ▼
                                       track LK (ref → def, FB check, deadzone)
                                                  │
                                          (def_pts, valid)
                                                  │
                                                  ▼
                                     visualize_flow_arrows (PNG)
                                                  │
                                                  ▼
                                    save 6 PNG ➜ outputs/LK/
```

### 13.2 Realtime tracking mode (`python -m src.real_time.realtime_pipeline`)

```
webcam ─► gray ─► (preprocess + detect on press 'r') ─► reference_markers
   │
   └──► loop frame:
            │
            ├─ if reference set:
            │    track_markers_lk(ref_image, gray, ref_markers)
            │    visualize arrows ──► window
            │    overlay text + FPS + timing
            │
            └─ else: idle prompt + FPS
       quit on 'q', clear on 'c'
```

### 13.3 Realtime slip V1

```
webcam ─► gray ─► loop:
              ├─► track LK (deadzone=True)
              ├─► append tracked vào history (max 5 frame)
              ├─► slip_detector.calculate_slip_probability(
              │      prev=history[0], curr=tracked, valid, ref)
              │       │ press-rate gate
              │       │ rebound filter
              │       │ MRVL → R_smooth (EMA bất đối xứng)
              │       └─► is_slip = (R_smooth > τ)
              ├─► visualize arrows
              └─► overlay slip prob | slip flag | moving count
```

### 13.4 Realtime slip V2

```
webcam ─► gray ─► loop:
              ├─► track LK (deadzone=False)
              ├─► slip_detector_v2.update(ref, curr, valid)
              │      ├─ append D(t) = curr - ref vào hist
              │      ├─ for k in scales (=1,3,9):
              │      │    ΔD = D(t) - D(t-k); centroid + r_i
              │      │    decompose: t, s, slip_fraction, dD_clean
              │      │    R, dir = MRVL(dD_clean)
              │      │    score_k = saturate(|t|/T0) * slip_fraction * R
              │      ├─ short_gate g từ scale ≤ 3
              │      ├─ best score = max(score_k * gate)
              │      ├─ classify phase từ s/k của scale ngắn nhất
              │      ├─ EMA score (alpha_up/alpha_down)
              │      └─ hysteresis: is_slip = Schmitt(score, 0.55, 0.35)
              ├─► visualize arrows
              └─► overlay: slip flag, score bar, HUD arrow, phase color
```

### 13.5 Đường đi data → force model

```
Imada ─┐
Camera ├─► realtime_station ─► sessions/<sid>/trials/<tid>/{frames, *.csv}
Motor ─┘
                                          │
                                          ▼
                            python -m src.force_model.prepare
                                          │
                                          ▼
                              data/cache/force_model/<sid>/<tid>.npz
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                        force_poly   force_model   force_cnn
                        (PolyReg)    (PointNet)    (ResNet/SmallCNN)
                              │           │           │
                              └─────┐     │     ┌─────┘
                                    ▼     ▼     ▼
                             outputs/<model>/{checkpoints, logs}
```

---

## 14. Tài liệu tham khảo gốc và phương pháp ngụ ý

Khi viết luận văn, gợi ý cite các tài liệu sau (không phải tất cả đều hiện ra
trực tiếp trong code, nhưng tương ứng với phương pháp đang dùng):

| § | Phương pháp | Nguồn nguyên bản (gợi ý cite) |
|---|---|---|
| 3 | Camera calibration (Zhang) | Z. Zhang, *A flexible new technique for camera calibration*, IEEE TPAMI 22(11), 2000. |
| 3 | Distortion model | D. C. Brown, *Decentering distortion of lenses*, Photogrammetric Eng. 32(3), 1966. |
| 4 | CLAHE | K. Zuiderveld, *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV, 1994. |
| 5 | SimpleBlobDetector | OpenCV implementation; theory: Lindeberg, *Feature detection with automatic scale selection*, IJCV 30(2), 1998. |
| 6 | Lucas–Kanade | B. D. Lucas & T. Kanade, *An iterative image registration technique with an application to stereo vision*, IJCAI 1981. |
| 6 | Pyramid LK | J.-Y. Bouguet, *Pyramidal implementation of the Lucas-Kanade feature tracker*, Intel Corp., 2001. |
| 6 | Forward–Backward error | Z. Kalal, K. Mikolajczyk, J. Matas, *Forward-Backward Error: Automatic Detection of Tracking Failures*, ICPR 2010. |
| 7 | Mean resultant length / circular statistics | N. I. Fisher, *Statistical Analysis of Circular Data*, Cambridge UP, 1995. |
| 8 | Decomposition + multi-scale slip | (đóng góp riêng — có thể đặt làm contribution của luận văn) |
| 11 | PointNet | C. R. Qi et al., *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*, CVPR 2017. |
| 11 | ResNet | K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. |
| 11 | Huber loss | P. J. Huber, *Robust estimation of a location parameter*, Annals Math. Stat. 35(1), 1964. |
| 11 | AdamW | I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. |
| 11 | Cosine annealing | I. Loshchilov & F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017. |
| 10 | Producer–consumer pattern | E. Dijkstra, *The structure of the THE-multiprogramming system*, CACM 11(5), 1968. |
| Tactile sensors | GelSight | W. Yuan, S. Dong, E. H. Adelson, *GelSight: High-Resolution Robot Tactile Sensors*, Sensors 17(12), 2017. |
| Slip detection background | M. R. Cutkosky | R. D. Howe, M. R. Cutkosky, *Sensing skin acceleration for slip and texture perception*, ICRA 1989; James et al., *Slip Detection With a Biomimetic Tactile Sensor*, RAL 2018. |

Ngoài ra, ngữ cảnh đối chiếu (phương pháp **không** dùng trong code — nên ghi
chú trong luận văn nếu nói đến lý do bỏ): RAFT (`src/test/flow_raft/spike.py`)
là spike thử nghiệm dense optical flow neural network, chưa tích hợp pipeline;
thuật toán Hungarian (đã loại khỏi codebase, theo CLAUDE.md) từng dùng cho data
association giữa các marker — bị thay thế bằng LK do LK đã giữ được thứ tự
marker giữa hai frame mà không cần matching tường minh.

---

## Phụ lục A — Mapping nhanh "khái niệm ↔ code"

| Khái niệm lý thuyết | File:line |
|---|---|
| Phép trừ nền box-blur | `src/core/preprocessing.py:35` |
| CLAHE | `src/core/preprocessing.py:39` |
| 4 filter blob | `src/core/detection.py:54-65` |
| Pyramid LK + tiêu chí dừng | `src/core/tracking.py:45-48` |
| Forward–backward check | `src/core/tracking.py:53-56` |
| Deadzone | `src/core/tracking.py:59-62` |
| Press-rate gate | `src/slip/v1.py:64-71` |
| Rebound filter | `src/slip/v1.py:78-83` |
| MRVL có trọng số | `src/slip/v1.py:91-101` |
| EMA bất đối xứng V1 | `src/slip/v1.py:101-104` |
| Decay V1 | `src/slip/v1.py:115-116` |
| Translation/Radial decomposition | `src/slip/v2.py:62-89` |
| Multi-scale loop | `src/slip/v2.py:124-161` |
| Short-scale gate | `src/slip/v2.py:163-184` |
| Phase classification | `src/slip/v2.py:220-232` |
| Press-slip ratio gate | `src/slip/v2.py:189-192` |
| Hysteresis Schmitt V2 | `src/slip/v2.py:197-203` |
| Producer-consumer thread | `src/collection/realtime_station.py:513-664` |
| Synchronized snapshot lock | `src/collection/realtime_station.py:521-535` |
| Steps_per_mm | `src/collection/Pyserial/Pyserial.ino:8-11` |
| Polynomial expansion | `src/force_poly/model.py:38-56` |
| Standardize trước expand | `src/force_poly/model.py:137-141` |
| Radial/tangential feature | `src/force_poly/features.py:81-83` |
| Shared MLP qua Conv1d | `src/force_model/model.py:13-23` |
| Masked max+mean pool | `src/force_model/model.py:31-60` |
| Conv1 inflate ImageNet | `src/force_cnn/model.py:59-68` |
| Trial-level split | `src/force_model/dataset.py:39-65` |
| Nearest-neighbor force sync | `src/force_model/prepare.py:67-79` |

---

## Phụ lục B — Giả định và giới hạn của hệ thống

Khi viết phần "thảo luận" trong luận văn, có thể đề cập:

1. **Phép xấp xỉ đàn hồi tuyến tính** (§2.3) chỉ đúng ở biến dạng nhỏ. Khi gel
   bị nén mạnh, đáp ứng phi tuyến — pipeline vẫn track được nhưng quan hệ
   slip ↔ MRVL có thể sai pha.

2. **Phép chiếu camera giả phẳng**: Bỏ qua component dịch chuyển out-of-plane
   ($\mathbf{d}_z$) khi biểu diễn $\mathbf{u}_i$. Với cảm biến mỏng, sai số
   này nhỏ; nhưng khi gel bị "đẩy lên" đáng kể (hard contact), $u_i$ trên ảnh
   không còn tỉ lệ tuyến tính với biến dạng.

3. **Detector blob không phân biệt được marker nếu chúng dính sát**: trong
   trạng thái nén mạnh, blob có thể hợp nhất → giảm $N$ → mất valid marker.
   `prepare.py` xử lý bằng cách *cố định* `ref_pts` từ frame nghỉ và dùng LK
   để track riêng biệt.

4. **Slip V1 không phát hiện slip xảy ra cùng pha với press**: do `press_rate_threshold`
   ép $R \to 0$ khi đang nhấn. V2 giải quyết bằng phân rã LS.

5. **Slip V2 cần buffer history dài** (tối đa scale + 2 frame). Latency ổn
   định ~ 9 frame ≈ 300 ms ở 30 FPS — chấp nhận được cho slip control,
   nhưng không phù hợp cho phản xạ siêu nhanh (< 100 ms).

6. **Ước lượng lực**: cả ba mô hình đều regression scalar (lực pháp tuyến).
   Mở rộng sang vector lực 3D đòi hỏi ground truth 6-DoF F/T sensor (chưa có
   trong codebase).

7. **Ảnh hưởng nhiệt và lão hoá gel**: trường biến dạng tham chiếu phụ thuộc
   trạng thái nghỉ — nếu gel chảy lệch dần do nhiệt, ref cần chụp lại định kỳ.
   Hệ thống thu thập nhắc người dùng chụp reference đầu mỗi session
   (`realtime_station.py:228-235`), nhưng không tự phát hiện drift.

---

*Tài liệu được biên soạn từ trạng thái codebase tại nhánh `collection`,
commit gần nhất `e9cd54d`. Mọi tham số trong YAML đều có thể được override khi
chạy mà không cần đổi code.*
