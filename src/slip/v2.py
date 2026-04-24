"""Slip Detector V2 — multi-scale, translation + radial decomposition.

Khác V1:
  - Làm việc trên trường biến dạng D(t) = curr - ref thay vì velocity thô.
    Slow slip (< vài px/frame) không bị deadzone triệt tiêu vì D(t) tích luỹ.
  - Đa thang thời gian: ΔD = D(t) - D(t-Δ) với Δ ∈ config `slip_v2.scales`.
  - Phân rã least-squares: ΔD_i = t (slip tịnh tiến) + s·r_i (press/release radial)
    + residual. Nhờ đó slip và press tách biệt — phát hiện slip ngay khi đang nhấn.
  - Score = saturate(|t|/T₀) · slip_fraction · R_clean (MRVL của ΔD đã khử radial).
  - Phase dựa trên s; gate press/release chỉ khi tỉ lệ slip/press nhỏ.
  - Short-scale gate: khi slip dừng, scale ngắn sụp nhanh → kéo score dài xuống.
  - Hysteresis 2 ngưỡng + EMA bất đối xứng.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import require


_EPS_R = 1e-6   # ngưỡng trivia cho ||r||² (tránh chia 0 khi N<2)
_EPS_FRAC = 1e-9  # smoothing cho slip_fraction


class SlipDetectorV2:
    def __init__(self, config: dict) -> None:
        scales_raw = require(config, "slip_v2.scales")
        scales = tuple(int(s) for s in scales_raw if int(s) >= 1)
        if not scales:
            raise ValueError("slip_v2.scales must contain at least one scale >= 1")
        self.scales: tuple[int, ...] = scales
        self.history_length = max(self.scales) + 2

        self.min_valid_markers = int(require(config, "slip_v2.min_valid_markers"))
        self.translation_norm = float(require(config, "slip_v2.translation_norm"))
        self.translation_min = float(require(config, "slip_v2.translation_min"))
        self.short_gate_norm = float(require(config, "slip_v2.short_gate_norm"))
        self.short_gate_softmin = float(require(config, "slip_v2.short_gate_softmin"))
        self.short_gate_max_scale = int(require(config, "slip_v2.short_gate_max_scale"))
        self.coherence_min = float(require(config, "slip_v2.coherence_min"))
        self.score_on = float(require(config, "slip_v2.score_on"))
        self.score_off = float(require(config, "slip_v2.score_off"))
        self.alpha_up = float(require(config, "slip_v2.alpha_up"))
        self.alpha_down = float(require(config, "slip_v2.alpha_down"))
        self.radial_rate_min = float(require(config, "slip_v2.radial_rate_min"))
        self.press_slip_ratio = float(require(config, "slip_v2.press_slip_ratio"))

        self.reset()

    def reset(self) -> None:
        self._D_hist: deque = deque(maxlen=self.history_length)
        self._valid_hist: deque = deque(maxlen=self.history_length)
        self._smoothed_score: float = 0.0
        self._is_slip: bool = False

    @staticmethod
    def _decompose(dD: npt.NDArray, r_centered: npt.NDArray) -> dict[str, Any]:
        """Phân rã ΔD_i = t + s·r_i + resid bằng least-squares."""
        M = dD.shape[0]
        t = np.mean(dD, axis=0)
        r_sq = float(np.sum(r_centered * r_centered))
        if r_sq < _EPS_R:
            s = 0.0
            resid = dD - t
            dD_clean = dD.copy()
        else:
            s = float(np.sum(r_centered * (dD - t))) / r_sq
            radial = s * r_centered
            resid = dD - t - radial
            dD_clean = dD - radial

        slip_energy = M * float(np.sum(t * t))
        press_energy = (s * s) * r_sq
        resid_energy = float(np.sum(resid * resid))
        slip_fraction = slip_energy / (slip_energy + resid_energy + _EPS_FRAC)
        return {
            "t": t,
            "s": s,
            "slip_energy": slip_energy,
            "press_energy": press_energy,
            "resid_energy": resid_energy,
            "slip_fraction": slip_fraction,
            "dD_clean": dD_clean,
        }

    @staticmethod
    def _weighted_mrvl(v: npt.NDArray) -> tuple[float, float]:
        mags = np.linalg.norm(v, axis=1)
        total = float(np.sum(mags))
        if total < _EPS_R:
            return 0.0, 0.0
        w = mags / total
        angles = np.arctan2(v[:, 1], v[:, 0])
        cs = float(np.sum(w * np.cos(angles)))
        sn = float(np.sum(w * np.sin(angles)))
        return float(np.sqrt(cs * cs + sn * sn)), float(np.arctan2(sn, cs))

    def update(
        self,
        ref_markers: npt.NDArray,
        curr_markers: npt.NDArray,
        valid_mask: npt.NDArray,
    ) -> dict[str, Any]:
        if ref_markers.shape[0] == 0 or not valid_mask.any():
            return self._decay("no_markers")

        D_t = curr_markers - ref_markers
        self._D_hist.append(D_t.copy())
        self._valid_hist.append(valid_mask.copy())

        if len(self._D_hist) < 2:
            return self._decay("warmup", phase="idle")

        # Centroid lấy từ ref (bất biến) theo tập valid hiện tại.
        ref_v = ref_markers[valid_mask]
        centroid = np.mean(ref_v, axis=0)
        r_full = ref_markers - centroid

        per_scale: dict[int, dict[str, Any]] = {}
        for k in self.scales:
            if len(self._D_hist) <= k:
                continue
            D_past = self._D_hist[-1 - k]
            valid_past = self._valid_hist[-1 - k]
            both = valid_mask & valid_past
            if int(both.sum()) < self.min_valid_markers:
                continue

            dD = (D_t - D_past)[both]
            r_both = r_full[both]

            dec = self._decompose(dD, r_both)
            t_mag = float(np.linalg.norm(dec["t"]))
            if t_mag < self.translation_min:
                continue

            R, mean_dir = self._weighted_mrvl(dec["dD_clean"])
            if R < self.coherence_min:
                continue

            mag_term = min(t_mag / self.translation_norm, 1.0)
            score_k = mag_term * dec["slip_fraction"] * R
            radial_rate = dec["s"] / max(k, 1)

            per_scale[k] = {
                "score": score_k,
                "t": dec["t"],
                "t_mag": t_mag,
                "R": R,
                "dir": mean_dir,
                "moving": int(dD.shape[0]),
                "slip_energy": dec["slip_energy"],
                "press_energy": dec["press_energy"],
                "slip_fraction": dec["slip_fraction"],
                "radial_rate": radial_rate,
            }

        # Short-scale gate: dùng scale <= `short_gate_max_scale`, chuẩn hoá √k.
        gate_scales = [k for k in self.scales if k <= self.short_gate_max_scale]
        short_gate = 0.0
        for gk in gate_scales:
            ginfo = per_scale.get(gk)
            if ginfo is None:
                continue
            gk_val = min(ginfo["t_mag"] / (self.short_gate_norm * np.sqrt(gk)), 1.0)
            if gk_val > short_gate:
                short_gate = gk_val
        short_gate = max(short_gate, self.short_gate_softmin)
        short_k = gate_scales[0] if gate_scales else self.scales[0]

        best_score = 0.0
        best: dict[str, Any] | None = None
        best_scale: int | None = None
        for k, info in per_scale.items():
            effective = info["score"] if k <= short_k else info["score"] * short_gate
            if effective > best_score:
                best_score = effective
                best = info
                best_scale = k

        phase = self._classify_phase(per_scale)

        # Gate press/release chỉ khi slip energy nhỏ tương đối so press energy.
        if phase in ("pressing", "releasing") and best is not None:
            ratio = best["slip_energy"] / (best["press_energy"] + _EPS_FRAC)
            if ratio < self.press_slip_ratio:
                return self._decay(phase, phase=phase)

        a = self.alpha_up if best_score >= self._smoothed_score else self.alpha_down
        self._smoothed_score = a * best_score + (1 - a) * self._smoothed_score

        if self._is_slip:
            if self._smoothed_score < self.score_off:
                self._is_slip = False
        else:
            if self._smoothed_score > self.score_on:
                self._is_slip = True

        out_phase = "slipping" if self._is_slip else phase
        if best is None:
            return self._empty_frame(phase=out_phase)
        return {
            "is_slip": bool(self._is_slip),
            "slip_score": float(self._smoothed_score),
            "raw_score": float(best_score),
            "translation": (float(best["t"][0]), float(best["t"][1])),
            "coherence": float(best["R"]),
            "mean_direction": best["dir"] if self._is_slip else None,
            "moving_count": int(best["moving"]),
            "scale": int(best_scale) if best_scale is not None else 0,
            "radial_rate": float(best["radial_rate"]),
            "phase": out_phase,
        }

    def _classify_phase(self, per_scale: dict[int, dict[str, Any]]) -> str:
        """Phase từ radial rate của scale ngắn nhất có dữ liệu."""
        for k in self.scales:
            info = per_scale.get(k)
            if info is None:
                continue
            rr = info["radial_rate"]
            if rr > self.radial_rate_min:
                return "pressing"
            if rr < -self.radial_rate_min:
                return "releasing"
            return "stick"
        return "idle"

    def _decay(self, reason: str, phase: str | None = None) -> dict[str, Any]:
        self._smoothed_score *= max(0.0, 1.0 - self.alpha_down)
        if self._smoothed_score < self.score_off:
            self._is_slip = False
        return self._empty_frame(phase=phase or reason)

    def _empty_frame(self, phase: str) -> dict[str, Any]:
        return {
            "is_slip": bool(self._is_slip),
            "slip_score": float(self._smoothed_score),
            "raw_score": 0.0,
            "translation": (0.0, 0.0),
            "coherence": 0.0,
            "mean_direction": None,
            "moving_count": 0,
            "scale": 0,
            "radial_rate": 0.0,
            "phase": phase,
        }
