"""Slip Detector V1 — MRVL (Mean Resultant Vector Length).

Phát hiện gross slip dựa trên độ đồng nhất hướng vector chuyển động của marker
(thống kê vòng tròn có trọng số). EMA bất đối xứng: tăng chậm, giảm nhanh.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ..config import require


class SlipDetector:
    """Detector slip dựa trên MRVL. Không giữ state ngoài hai scalar EMA."""

    def __init__(self, config: dict) -> None:
        self.min_motion_thresh = float(require(config, "slip_detection.min_motion_thresh"))
        self.slip_threshold = float(require(config, "slip_detection.slip_threshold"))
        self.min_moving_markers = int(require(config, "slip_detection.min_moving_markers"))
        self.alpha = float(require(config, "slip_detection.alpha"))
        self.alpha_decay = float(require(config, "slip_detection.alpha_decay"))
        self.rebound_dot_prod_threshold = float(
            require(config, "slip_detection.rebound_dot_prod_threshold")
        )
        self.press_rate_threshold = float(require(config, "slip_detection.press_rate_threshold"))

        self._smoothed_r = 0.0
        self._prev_mean_deformation = 0.0

    def reset(self) -> None:
        self._smoothed_r = 0.0
        self._prev_mean_deformation = 0.0

    def calculate_slip_probability(
        self,
        prev_markers: npt.NDArray,
        current_markers: npt.NDArray,
        valid_mask: npt.NDArray,
        ref_markers: Optional[npt.NDArray] = None,
    ) -> dict[str, Any]:
        """Tính xác suất trượt dựa trên độ đồng nhất hướng vector chuyển động.

        Args:
            prev_markers: toạ độ ở trạng thái trước (N, 2).
            current_markers: toạ độ ở trạng thái hiện tại (N, 2).
            valid_mask: boolean (N,) đánh dấu marker track thành công.
            ref_markers: (tuỳ chọn) toạ độ trạng thái tĩnh ban đầu, dùng cho rebound
                filter + press-phase detection.

        Returns:
            dict: {is_slip, r_value, mean_direction, moving_count, phase}.
        """
        if not valid_mask.any():
            return self._decay_and_return("no_markers")

        valid_prev = prev_markers[valid_mask]
        valid_curr = current_markers[valid_mask]

        # Press phase: nếu deformation đang tăng nhanh (đang nhấn xuống), suppress R.
        if ref_markers is not None:
            valid_ref = ref_markers[valid_mask]
            deformation = valid_curr - valid_ref
            mean_def_mag = float(np.mean(np.linalg.norm(deformation, axis=1)))
            delta_def = mean_def_mag - self._prev_mean_deformation
            self._prev_mean_deformation = mean_def_mag
            if delta_def > self.press_rate_threshold:
                return self._decay_and_return("pressing")

        displacements = valid_curr - valid_prev
        magnitudes = np.linalg.norm(displacements, axis=1)
        motion_mask = magnitudes > self.min_motion_thresh

        # Rebound filter: loại marker đang hồi phục đàn hồi.
        if ref_markers is not None:
            valid_ref = ref_markers[valid_mask]
            deformation = valid_curr - valid_ref
            dot_prods = np.sum(displacements * deformation, axis=1)
            rebound_mask = dot_prods < self.rebound_dot_prod_threshold
            motion_mask = motion_mask & (~rebound_mask)

        significant_disp = displacements[motion_mask]
        moving_count = len(significant_disp)
        if moving_count < self.min_moving_markers:
            return self._decay_and_return("insufficient_motion")

        # Mean Resultant Vector Length có trọng số theo độ lớn dịch chuyển.
        angles = np.arctan2(significant_disp[:, 1], significant_disp[:, 0])
        valid_mags = magnitudes[motion_mask]
        weights = valid_mags / np.sum(valid_mags)

        sum_cos = float(np.sum(weights * np.cos(angles)))
        sum_sin = float(np.sum(weights * np.sin(angles)))
        raw_r_value = float(np.sqrt(sum_cos**2 + sum_sin**2))
        mean_direction = float(np.arctan2(sum_sin, sum_cos))

        # EMA bất đối xứng: tăng chậm (confirm trước khi trigger), giảm nhanh.
        if raw_r_value >= self._smoothed_r:
            self._smoothed_r = self.alpha * raw_r_value + (1 - self.alpha) * self._smoothed_r
        else:
            self._smoothed_r = self.alpha_decay * raw_r_value + (1 - self.alpha_decay) * self._smoothed_r

        is_slip = bool(self._smoothed_r > self.slip_threshold)
        return {
            "is_slip": is_slip,
            "r_value": float(self._smoothed_r),
            "mean_direction": mean_direction if is_slip else None,
            "moving_count": int(moving_count),
            "phase": "slip" if is_slip else "tracking",
        }

    def _decay_and_return(self, phase: str) -> dict[str, Any]:
        self._smoothed_r *= 1 - self.alpha_decay
        return {
            "is_slip": False,
            "r_value": float(self._smoothed_r),
            "mean_direction": None,
            "moving_count": 0,
            "phase": phase,
        }
