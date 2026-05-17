"""Slip Detector V1 — MRVL (Mean Resultant Vector Length).

Phát hiện gross slip dựa trên độ đồng nhất hướng vector chuyển động của marker
(thống kê vòng tròn có trọng số). Score được boost bởi chuyển động tịnh tiến
nhất quán để slip thật không bị ì bởi EMA ở vài frame đầu.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ..config import require


_EPS = 1e-9


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

        pressing = False
        mean_def_mag = 0.0
        if ref_markers is not None:
            valid_ref = ref_markers[valid_mask]
            deformation = valid_curr - valid_ref
            mean_def_mag = self._mean_norm(deformation)
            delta_def = mean_def_mag - self._prev_mean_deformation
            self._prev_mean_deformation = mean_def_mag
            pressing = delta_def > self.press_rate_threshold

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

        # Khử radial press/release trước khi đo đồng hướng. Khi ref không có sẵn
        # thì detector trở về MRVL nguyên bản trên displacement thô.
        clean_disp = significant_disp
        if ref_markers is not None:
            ref_moving = ref_markers[valid_mask][motion_mask]
            clean_disp = self._remove_radial_component(significant_disp, ref_moving)

        # Mean Resultant Vector Length có trọng số theo độ lớn dịch chuyển.
        raw_r_value, mean_direction = self._weighted_mrvl(clean_disp)
        mean_translation = np.mean(clean_disp, axis=0)
        translation_mag = float(np.linalg.norm(mean_translation))
        mean_motion = self._mean_norm(clean_disp)

        # MRVL chỉ biết "cùng hướng" chứ không biết trượt mạnh hay yếu. Nhân thêm
        # evidence tịnh tiến và tỉ lệ marker đang chuyển động để score nhạy hơn
        # với slip thật, nhưng vẫn thấp khi chỉ có vài marker nhiễu.
        translation_gain = self._saturate(
            translation_mag / max(self.min_motion_thresh, _EPS)
        )
        participation_gain = self._saturate(
            moving_count / max(self.min_moving_markers * 2.0, 1.0)
        )
        motion_gain = self._saturate(mean_motion / max(self.min_motion_thresh, _EPS))
        raw_score = raw_r_value * (0.55 + 0.45 * translation_gain)
        raw_score *= 0.90 + 0.10 * participation_gain
        raw_score *= 0.85 + 0.15 * motion_gain
        raw_score = min(raw_score, 1.0)

        # Đang press mà gần như không có translation thì vẫn suppress; nếu vừa
        # press vừa trượt, cho tín hiệu tịnh tiến đi qua thay vì triệt tiêu sớm.
        if pressing and translation_gain < 0.35:
            return self._decay_and_return("pressing")

        # EMA bất đối xứng, nhưng tăng nhanh hơn khi evidence slip đã rõ.
        if raw_score >= self._smoothed_r:
            alpha_up = self._adaptive_alpha(raw_score, raw_r_value, translation_gain)
            self._smoothed_r = alpha_up * raw_score + (1 - alpha_up) * self._smoothed_r
        else:
            self._smoothed_r = (
                self.alpha_decay * raw_score + (1 - self.alpha_decay) * self._smoothed_r
            )

        is_slip = bool(self._smoothed_r > self.slip_threshold)
        return {
            "is_slip": is_slip,
            "r_value": float(self._smoothed_r),
            "raw_r_value": float(raw_r_value),
            "raw_score": float(raw_score),
            "translation": (float(mean_translation[0]), float(mean_translation[1])),
            "translation_mag": float(translation_mag),
            "mean_motion": float(mean_motion),
            "pressing": bool(pressing),
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

    @staticmethod
    def _mean_norm(vectors: npt.NDArray) -> float:
        if len(vectors) == 0:
            return 0.0
        return float(np.mean(np.linalg.norm(vectors, axis=1)))

    @staticmethod
    def _saturate(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _weighted_mrvl(vectors: npt.NDArray) -> tuple[float, float]:
        mags = np.linalg.norm(vectors, axis=1)
        total = float(np.sum(mags))
        if total < _EPS:
            return 0.0, 0.0

        weights = mags / total
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sum_cos = float(np.sum(weights * np.cos(angles)))
        sum_sin = float(np.sum(weights * np.sin(angles)))
        return (
            float(np.sqrt(sum_cos**2 + sum_sin**2)),
            float(np.arctan2(sum_sin, sum_cos)),
        )

    @staticmethod
    def _remove_radial_component(
        displacements: npt.NDArray,
        ref_points: npt.NDArray,
    ) -> npt.NDArray:
        if len(displacements) < 2:
            return displacements

        centered_ref = ref_points - np.mean(ref_points, axis=0)
        ref_energy = float(np.sum(centered_ref * centered_ref))
        if ref_energy < _EPS:
            return displacements

        translation = np.mean(displacements, axis=0)
        radial_scale = float(np.sum(centered_ref * (displacements - translation))) / ref_energy
        return displacements - radial_scale * centered_ref

    def _adaptive_alpha(
        self,
        raw_score: float,
        coherence: float,
        translation_gain: float,
    ) -> float:
        evidence = self._saturate((raw_score + coherence + translation_gain) / 3.0)
        return self._saturate(self.alpha + (1.0 - self.alpha) * 0.65 * evidence)
