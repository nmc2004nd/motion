import cv2
import numpy as np

# Import các module đã có sẵn từ thư mục src
from src.detection import detect_markers
from src.preprocessing import preprocess
from src.tracking import match_markers_robust
from src.visualization import visualize_flow_arrows, visualize_flow_hsv

class RealtimeTactileTracking:
    def __init__(self, camera_id=0, arrow_scale=1.0):
        self.camera_id = camera_id
        self.arrow_scale = arrow_scale
        self.ref_img = None
        self.ref_markers = None
        
    def run(self):
        # Mở luồng video /dev/video0 (camera_id=0)
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera_id}")

        print(f"=== BẮT ĐẦU CAMERA {self.camera_id} ===")
        print("- Nhấn phím 'r' để lấy frame hiện tại làm Reference (Trạng thái tĩnh/chưa biến dạng).")
        print("- Nhấn phím 'c' để xóa Reference hiện tại và quay lại ban đầu.")
        print("- Nhấn phím 'q' để thoát.")

        # Bỏ qua một số frame đầu để camera ổn định ánh sáng
        for _ in range(15):
            cap.read()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi: Không thể lấy frame từ camera.")
                break

            # Chuyển ảnh sang dạng xám
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            # Bắt sự kiện phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Lưu mốc tham chiếu
                self.ref_img = gray_frame.copy()
                ref_proc = preprocess(self.ref_img)
                self.ref_markers, _ = detect_markers(ref_proc)
                print(f"Đã lưu ảnh Reference: Phát hiện {len(self.ref_markers)} markers.")
            elif key == ord('c'):
                # Xoá refernce
                self.ref_img = None
                self.ref_markers = None
                print("Đã xoá ảnh Reference.")

            # Nếu đã có frame tham chiếu, tiến hành tính toán tracking realtime
            if self.ref_img is not None and self.ref_markers is not None:
                def_proc = preprocess(gray_frame)
                def_markers_naive, _ = detect_markers(def_proc)
                
                # Gọi tính năng Tracking Match (cân nhắc giảm max_disp hoặc scale thủ công nếu chạy quá chậm)
                if len(def_markers_naive) > 0 and len(self.ref_markers) > 0:
                    def_markers_tracked, valid = match_markers_robust(
                        self.ref_markers, def_markers_naive, self.ref_img.shape, 
                        max_disp=self.ref_img.shape[1]/10.0
                    )

                    if valid.any():
                        # Trực quan hoá luồng Vector/Arrows (không lưu file, hiển thị thẳng lên màn hình)
                        vis_arrows = visualize_flow_arrows(
                            def_proc, self.ref_markers, def_markers_tracked, valid, 
                            scale=self.arrow_scale, save_path=None
                        )
                        cv2.imshow("Realtime Flow Arrows", vis_arrows)

                        # Trực quan hoá luồng Màu HSV (không lưu file)
                        vis_hsv = visualize_flow_hsv(
                            self.ref_markers, def_markers_tracked, valid, 
                            self.ref_img.shape, save_path=None
                        )
                        cv2.imshow("Realtime Flow HSV", vis_hsv)

                # Hiển thị text trạng thái đang Tracking
                cv2.putText(display_frame, f"Tracking: {len(self.ref_markers)} markers limit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Chưa có Reference, nhắc người dùng
                cv2.putText(display_frame, "Press 'r' to capture reference", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Hiển thị ảnh Raw từ Camera liên tục
            cv2.imshow("Raw WebCam (/dev/video0)", display_frame)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # camera_id = 0 thông thường là /dev/video0. 
    # Nếu hệ thống của bạn ở thiết bị khác, truyền chuỗi string camera_id="/dev/video0"
    pipeline = RealtimeTactileTracking(camera_id=0, arrow_scale=1.0)
    pipeline.run()
