"""Shim giữ entry point `python -m src.slip_v2.realtime_pipeline_v2`.

Logic đã chuyển sang `src.pipelines.realtime_slip_v2`.
"""

from src.pipelines.realtime_slip_v2 import RealtimeSlipV2Pipeline, main

__all__ = ["RealtimeSlipV2Pipeline", "main"]


if __name__ == "__main__":
    main()
