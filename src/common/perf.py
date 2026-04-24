"""FPS meter + timing helper cho vòng lặp realtime."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class FpsMeter:
    """Tính FPS tức thời từ chênh lệch `perf_counter` giữa các frame."""

    def __init__(self) -> None:
        self._prev: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._prev is None:
            self._prev = now
            return 0.0
        dt = now - self._prev
        self._prev = now
        return 1.0 / dt if dt > 0 else 0.0


class TimingAccumulator:
    """Gom thời gian của từng stage trong 1 frame, đơn vị ms."""

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = (time.perf_counter() - t0) * 1000.0

    def snapshot(self) -> dict[str, float]:
        return dict(self._timings)

    def reset(self) -> None:
        self._timings.clear()
