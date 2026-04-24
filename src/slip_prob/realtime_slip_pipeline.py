"""Shim giữ entry point `python -m src.slip_prob.realtime_slip_pipeline`.

Logic đã chuyển sang `src.pipelines.realtime_slip_v1`.
"""

from src.pipelines.realtime_slip_v1 import RealtimeSlipTracking, main

__all__ = ["RealtimeSlipTracking", "main"]


if __name__ == "__main__":
    main()
