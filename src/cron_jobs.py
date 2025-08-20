# src/cron_jobs.py
from __future__ import annotations
import argparse, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import TICKERS, DEFAULT_BENCH
from src.live import market_is_open, record_tick_to_raw, finalize_today_and_cleanup

ROOT = Path(__file__).resolve().parents[1]
DIR_CACHE = ROOT / "data" / "cache"
DIR_CACHE.mkdir(parents=True, exist_ok=True)

def _flag(day: dt.date) -> Path:
    return DIR_CACHE / f"finalized_{day.isoformat()}.flag"

def job_tick():
    symbols = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
    if market_is_open():
        record_tick_to_raw(symbols)

def job_finalize():
    symbols = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
    today = dt.datetime.now(dt.timezone.utc).date()
    if _flag(today).exists():
        return
    # Only finalize if we're past 4pm New York
    ny = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if ny.hour >= 16:
        try:
            finalize_today_and_cleanup(symbols)
            _flag(today).write_text("ok", encoding="utf-8")
        except Exception:
            # Try again next run; bar may not be posted yet
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tick", "finalize", "auto"])
    args = ap.parse_args()
    if args.mode == "tick":
        job_tick()
    elif args.mode == "finalize":
        job_finalize()
    else:
        if market_is_open():
            job_tick()
        else:
            job_finalize()

if __name__ == "__main__":
    main()
