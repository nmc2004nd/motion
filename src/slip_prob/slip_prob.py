import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Tuple, Optional

class SlipDetector:
    """
    Module hỗ trợ phát hiện hiện tượng trượt (Gross Slip) dựa trên sự đồng nhất
    hướng chuyển động của các marker (Circular statistics - Mean Resultant Vector Length).
    
    Thuật toán không phụ thuộc vào trạng thái hệ thống, chỉ yêu cầu 2 tập điểm toạ độ
    đầu vào (prev_markers và current_markers) cùng mảng trạng thái hợp lệ (valid_mask).
    """
    
    def __init__(self, min_motion_thresh: float = 0.5, slip_threshold: float = 0.8, min_moving_markers: int = 5):
        """
        Khởi tạo SlipDetector.
        
        Args:
            min_motion_thresh (float): Ngưỡng di chuyển tối thiểu (pixel) để loại bỏ nhiễu rung camera.
            slip_threshold (float): Ngưỡng hệ số đồng hướng R (0 -> 1) để xác định trượt. 
                                    R càng gần 1 tức là các marker càng di chuyển song song.
            min_moving_markers (int): Số lượng marker tối thiểu đang chuyển động để có thể kết luận trượt.
        """
        self.min_motion_thresh = min_motion_thresh
        self.slip_threshold = slip_threshold
        self.min_moving_markers = min_moving_markers

    def calculate_slip_probability(
        self, 
        prev_markers: npt.NDArray, 
        current_markers: npt.NDArray, 
        valid_mask: npt.NDArray,
        ref_markers: Optional[npt.NDArray] = None
    ) -> Dict[str, Any]:
        """
        Tính toán xác suất trượt dựa vào độ đồng nhất của hướng vector chuyển động, bao gồm khử nhiễu hồi tiếp.
        
        Args:
            prev_markers: Toạ độ marker ở trạng thái trước đó thời gian ngắn (N, 2)
            current_markers: Toạ độ marker ở trạng thái hiện tại (N, 2)
            valid_mask: Mảng boolean (N,) đánh dấu các marker track thành công.
            ref_markers: (Optional) Toạ độ trạng thái tĩnh ban đầu. Dùng để xem xét chiều chuyển động (Tránh nhận diện nhầm sự đàn hồi lúc thả tay thành trượt).
            
        Returns:
            Dict chứa các thông tin:
                - is_slip (bool): Trạng thái có đang bị trượt hay không.
                - r_value (float): Hệ số đồng hướng R.
                - mean_direction (float): Góc trượt trung bình (radian), None nếu không trượt.
                - moving_count (int): Số lượng marker thực sự chuyển động qua ngưỡng (không phải nảy ngược).
        """
        # Trả về giá trị mặc định nếu không có marker nào hợp lệ
        if not valid_mask.any():
            return {
                "is_slip": False,
                "r_value": 0.0,
                "mean_direction": None,
                "moving_count": 0
            }

        # Chỉ lấy những marker hợp lệ
        valid_prev = prev_markers[valid_mask]
        valid_curr = current_markers[valid_mask]

        # Tính vector dịch chuyển
        displacements = valid_curr - valid_prev
        magnitudes = np.linalg.norm(displacements, axis=1)

        # Lọc bỏ các chuyển động nhỏ (nhiễu rùng mình hoặc marker đứng im)
        motion_mask = magnitudes > self.min_motion_thresh
        
        # KIỂM TRA ĐÀN HỒI (REBOUND) LÚC THẢ TAY RA
        if ref_markers is not None:
            valid_ref = ref_markers[valid_mask]
            deformation = valid_curr - valid_ref
            
            # Tính hướng tương đối giữa (chuyển động của marker) và (sự móp méo chung)
            # Dot_product < 0 nghĩa là vector vận tốc đang kéo marker ngược thẳng về vị trí nghỉ lúc đầu!
            dot_prods = np.sum(displacements * deformation, axis=1)
            rebound_mask = dot_prods < -0.1
            
            # Nếu marker đang nảy ngược đàn hồi, loại bỏ nó khỏi bộ lọc trượt
            motion_mask = motion_mask & (~rebound_mask)

        significant_disp = displacements[motion_mask]
        
        moving_count = len(significant_disp)

        # Không đủ marker chuyển động -> không thể kết luận trượt (đang đứng yên/nhiễu)
        if moving_count < self.min_moving_markers:
            return {
                "is_slip": False,
                "r_value": 0.0,
                "mean_direction": None,
                "moving_count": moving_count
            }

        # Tính góc phi (theta) của mỗi điểm chuyển động (radian, từ -pi đến pi)
        angles = np.arctan2(significant_disp[:, 1], significant_disp[:, 0])

        # Tính Mean Resultant Vector Length (R) theo circular statistics
        sum_cos = np.sum(np.cos(angles))
        sum_sin = np.sum(np.sin(angles))
        
        r_value = np.sqrt(sum_cos**2 + sum_sin**2) / moving_count
        mean_direction = np.arctan2(sum_sin, sum_cos)

        is_slip = bool(r_value > self.slip_threshold)

        return {
            "is_slip": is_slip,
            "r_value": float(r_value),
            "mean_direction": float(mean_direction) if is_slip else None,
            "moving_count": int(moving_count)
        }

