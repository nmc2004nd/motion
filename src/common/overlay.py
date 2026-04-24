"""Các hàm vẽ overlay text / thanh trượt lên frame BGR."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from ..config import require

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tuple(cfg: dict, key: str) -> tuple:
    return tuple(require(cfg, key))


def draw_text(
    img: npt.NDArray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(img, text, pos, _FONT, scale, color, thickness)


def draw_idle_prompt(img: npt.NDArray, config: dict) -> None:
    """Vẽ gợi ý 'Nhan r de chup tham chieu' + FPS khi chưa có reference."""
    pos = _tuple(config, "visualization.text.status_pos")
    scale = float(require(config, "visualization.text.scale_idle"))
    thickness = int(require(config, "visualization.text.thickness"))
    color = _tuple(config, "visualization.text.color_idle")
    draw_text(img, "Nhan 'r' de chup tham chieu", pos, color, scale, thickness)


def draw_fps(img: npt.NDArray, fps: float, config: dict, color_key: str = "visualization.text.color_tracking") -> None:
    pos = _tuple(config, "visualization.text.fps_pos")
    scale = float(require(config, "visualization.text.scale_normal"))
    thickness = int(require(config, "visualization.text.thickness"))
    color = tuple(require(config, color_key))
    draw_text(img, f"FPS: {fps:.1f}", pos, color, scale, thickness)


def draw_tracking_status(img: npt.NDArray, text: str, config: dict) -> None:
    pos = _tuple(config, "visualization.text.status_pos")
    scale = float(require(config, "visualization.text.scale_normal"))
    thickness = int(require(config, "visualization.text.thickness"))
    color = _tuple(config, "visualization.text.color_tracking")
    draw_text(img, text, pos, color, scale, thickness)


def draw_timings(img: npt.NDArray, timings: dict[str, float], config: dict) -> None:
    """Vẽ từng cặp `stage: X.Xms` theo cột, bắt đầu từ `timing_start_y`."""
    start_y = int(require(config, "visualization.text.timing_start_y"))
    step_y = int(require(config, "visualization.text.timing_step_y"))
    scale = float(require(config, "visualization.text.scale_idle"))
    thickness = int(require(config, "visualization.text.thickness"))
    color = _tuple(config, "visualization.text.color_timing")
    x = int(require(config, "visualization.text.fps_pos")[0])
    y = start_y
    for name, ms in timings.items():
        draw_text(img, f"{name}: {ms:.1f}ms", (x, y), color, scale, thickness)
        y += step_y


def draw_slip_hud_arrow(
    img: npt.NDArray,
    translation: tuple[float, float],
    color: tuple[int, int, int],
    config: dict,
) -> None:
    """Vẽ mũi tên chỉ hướng trượt ở góc phải dưới (cho slip V2 overlay)."""
    cfg_key = "visualization.overlay_v2.hud_arrow"
    offset_x = int(require(config, f"{cfg_key}.offset_x_from_right"))
    offset_y = int(require(config, f"{cfg_key}.offset_y_from_bottom"))
    scale = float(require(config, f"{cfg_key}.scale"))
    radius = int(require(config, f"{cfg_key}.center_radius"))
    center_color = _tuple(config, f"{cfg_key}.center_color")
    min_mag = float(require(config, f"{cfg_key}.min_magnitude"))
    thickness = int(require(config, f"{cfg_key}.thickness"))
    tip_length = float(require(config, f"{cfg_key}.tip_length"))

    h, w = img.shape[:2]
    cx, cy = w - offset_x, h - offset_y
    cv2.circle(img, (cx, cy), radius, center_color, -1)
    tx, ty = translation
    if float(np.hypot(tx, ty)) > min_mag:
        ex = int(cx + tx * scale)
        ey = int(cy + ty * scale)
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, thickness, tipLength=tip_length)


def draw_slip_score_bar(
    img: npt.NDArray,
    score: float,
    score_on: float,
    score_off: float,
    fill_color: tuple[int, int, int],
    config: dict,
) -> None:
    """Vẽ thanh trượt score + hai vạch ngưỡng on/off (slip V2 overlay)."""
    cfg_key = "visualization.overlay_v2.slip_bar"
    margin_x = int(require(config, f"{cfg_key}.margin_x"))
    margin_y = int(require(config, f"{cfg_key}.margin_y_from_bottom"))
    bar_w = int(require(config, f"{cfg_key}.width"))
    bar_h = int(require(config, f"{cfg_key}.height"))
    bg_color = _tuple(config, f"{cfg_key}.bg_color")
    on_color = _tuple(config, f"{cfg_key}.threshold_on_color")
    off_color = _tuple(config, f"{cfg_key}.threshold_off_color")

    h = img.shape[0]
    x, y = margin_x, h - margin_y
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), bg_color, -1)
    fill = int(bar_w * min(max(score, 0.0), 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + bar_h), fill_color, -1)
    on_px = x + int(bar_w * score_on)
    off_px = x + int(bar_w * score_off)
    cv2.line(img, (on_px, y), (on_px, y + bar_h), on_color, 1)
    cv2.line(img, (off_px, y), (off_px, y + bar_h), off_color, 1)
