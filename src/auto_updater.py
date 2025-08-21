# src/auto_updater.py
from __future__ import annotations
import threading, time, logging, datetime as dt
from pathlib import Path

from src.config import REFRESH_SEC_MARKET, REFRESH_SEC_CLOSED
from src.data_fetch import ensure_history, record_tick, build_prices_overlay, finalize_eod
from src.live import market_is_open  # your existing market status helper

ROOT = Path(__file__).resolve().parents[1]
DIR_CACHE = ROOT / "data" / "cache"
DIR_CACHE.mkdir(parents=True, exist_ok=True)

def _final_flag_path(day: dt.date) -> Path:
    return DIR_CACHE / f"finalized_{day.isoformat()}.flag"

def _mark_finalized(day: dt.date) -> None:
    try:
        _final_flag_path(day).write_text("ok", encoding="utf-8")
    except Exception:
        logging.exception("[auto] could not write finalized flag")

def _is_finalized(day: dt.date) -> bool:
    return _final_flag_path(day).exists()

def _auto_loop(stop_event: threading.Event) -> None:
    logging.info("[auto] starting background updater loop")
    # Make sure we have history once, safe to call repeatedly
    try:
        ensure_history()
    except Exception:
        logging.exception("[auto] ensure_history failed (continuing)")

    while not stop_event.is_set():
        try:
            today = dt.date.today()
            if market_is_open():
                # Intraday: capture a tick and build the overlay for UI
                try:
                    record_tick()
                    build_prices_overlay(save_to_cache=True)
                except Exception:
                    logging.exception("[auto] intraday tick/overlay failed")
                # sleep short during open market
                time.sleep(max(5, int(REFRESH_SEC_MARKET)))
            else:
                # If we haven't finalized today's bar yet, try once
                if not _is_finalized(today):
                    try:
                        finalize_eod(today)
                        _mark_finalized(today)
                        logging.info(f"[auto] finalized EOD for {today}")
                    except Exception:
                        logging.exception("[auto] finalize_eod failed")
                # sleep longer when market is closed
                time.sleep(max(30, int(REFRESH_SEC_CLOSED)))
        except Exception:
            logging.exception("[auto] unexpected error in loop")
            time.sleep(30)  # backoff to avoid tight error loops

def start_auto_updater() -> None:
    """
    Start a single background thread for this server process.
    Safe to call multiple times (no duplicate threads).
    """
    if getattr(start_auto_updater, "_started", False):
        return
    stop_event = threading.Event()
    t = threading.Thread(target=_auto_loop, args=(stop_event,), daemon=True, name="auto-updater")
    t.start()
    start_auto_updater._started = True  # type: ignore[attr-defined]

# src/auto_updater.py
import time, datetime as dt
from zoneinfo import ZoneInfo
from src.data_fetch import update_parquet_daily

def _sleeps():
    ny = ZoneInfo("America/New_York")
    while True:
        now = dt.datetime.now(ny)
        # run ~16:10 NY time to allow final bars to post
        target = now.replace(hour=16, minute=10, second=0, microsecond=0)
        if now >= target:
            target += dt.timedelta(days=1)
        yield max(60, (target - now).total_seconds())

def main():
    print("[scheduler] EOD loop started", flush=True)
    for s in _sleeps():
        time.sleep(s)
        try:
            print("[scheduler] EOD:", update_parquet_daily(force=False), flush=True)
        except Exception as e:
            print("[scheduler] EOD error:", e, flush=True)
        time.sleep(600)  # avoid multiple runs in the same minute

if __name__ == "__main__":
    main()

