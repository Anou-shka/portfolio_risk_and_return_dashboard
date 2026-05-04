# src/auto_updater.py
from __future__ import annotations

import datetime as dt
import time
from zoneinfo import ZoneInfo

from src.data_fetch import update_parquet_daily


def _next_run_delay() -> float:
    """Seconds until ~16:10 NY time, when final EOD bars are reliably posted."""
    ny = ZoneInfo("America/New_York")
    now = dt.datetime.now(ny)
    target = now.replace(hour=16, minute=10, second=0, microsecond=0)
    if now >= target:
        target += dt.timedelta(days=1)
    return max(60, (target - now).total_seconds())


def main() -> None:
    print("[scheduler] EOD loop started", flush=True)
    while True:
        time.sleep(_next_run_delay())
        try:
            result = update_parquet_daily(force=False)
            print(f"[scheduler] EOD: {result}", flush=True)
        except Exception as e:
            print(f"[scheduler] EOD error: {e}", flush=True)
        time.sleep(600)  # guard against multiple runs in the same 16:10 window


if __name__ == "__main__":
    main()
