"""Shim giữ entry point `python -m src.real_time.realtime_pipeline`.

Logic đã chuyển sang `src.pipelines.realtime_tracking`.
"""

from src.pipelines.realtime_tracking import RealtimeTactileTracking, main

__all__ = ["RealtimeTactileTracking", "main"]


if __name__ == "__main__":
    main()
