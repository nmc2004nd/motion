"""Xử lý phím tắt realtime (capture reference / clear / quit)."""

from __future__ import annotations

from enum import Enum, auto

import cv2

from ..config import require


class KeyCommand(Enum):
    NONE = auto()
    CAPTURE = auto()
    CLEAR = auto()
    QUIT = auto()


class KeyboardController:
    """Đọc phím nhấn và dịch sang `KeyCommand`.

    Gắn 3 phím (capture / clear / quit) + `wait_key_ms` cấu hình trong `controls.*`.
    """

    def __init__(self, config: dict) -> None:
        self.wait_key_ms = int(require(config, "controls.wait_key"))
        self._key_capture = ord(str(require(config, "controls.key_capture")))
        self._key_clear = ord(str(require(config, "controls.key_clear")))
        self._key_quit = ord(str(require(config, "controls.key_quit")))

    def poll(self) -> KeyCommand:
        key = cv2.waitKey(self.wait_key_ms) & 0xFF
        if key == self._key_quit:
            return KeyCommand.QUIT
        if key == self._key_capture:
            return KeyCommand.CAPTURE
        if key == self._key_clear:
            return KeyCommand.CLEAR
        return KeyCommand.NONE
