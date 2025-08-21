# # src/live.py

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

# ---- robust config import ----
try:
    from src.config import TICKERS, DEFAULT_BENCH
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.config import TICKERS, DEFAULT_BENCH

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIR_RAW_INTRADAY = DATA / "raw" / "intraday"
DIR_RAW_INTRADAY.mkdir(parents=True, exist_ok=True)

RAW_FILE = DIR_RAW_INTRADAY / "latest_data_tickers.parquet"  # single intraday cache

# ---- settings ----
DEFAULT_PERIOD = "1d"     # fetch today's history
DEFAULT_INTERVAL = "1m"   # 1-minute bars
FIELDS_DESIRED = ["Open", "High", "Low", "Close", "Adj Close"]  # intraday may miss Adj Close

# ---- helpers ----

def _index_dates_ny(idx) -> pd.Series:
    """Return a vector of NY-local dates for a DatetimeIndex that may be naive or tz-aware."""
    idx = pd.to_datetime(idx)
    # If naive (no tz), assume UTC (what yfinance often uses for intraday)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ZoneInfo("America/New_York")).date


def _symbols() -> List[str]:
    # stable order: config tickers + benchmark last
    return list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))

def _now_utc() -> dt.datetime:
    return dt.datetime.now(timezone.utc)

def _now_ny() -> dt.datetime:
    return _now_utc().astimezone(ZoneInfo("America/New_York"))

def market_is_open() -> bool:
    """
    Simple NYSE gate: Mon-Fri, 09:30–16:00 America/New_York.
    (Good enough for automation; holidays still return False because no data will arrive.)
    """
    now = _now_ny()
    if now.weekday() >= 5:
        return False
    after_open = (now.hour > 9) or (now.hour == 9 and now.minute >= 30)
    before_close = now.hour < 16
    return after_open and before_close

def _ensure_intraday_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure MultiIndex columns are (ticker, field) and include 'Adj Close' (copy of 'Close' if missing).
    Keep columns ordered as [symbols x FIELDS_DESIRED].
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # single ticker shape -> normalize
        df = pd.concat({ _symbols()[0]: df }, axis=1)

    # If yfinance gave (field, ticker), flip to (ticker, field)
    level0 = set(df.columns.get_level_values(0))
    if "Open" in level0 or "Close" in level0 or "Adj Close" in level0:
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Add Adj Close if missing (intraday often lacks it)
    # Build columns for each ticker
    tickers = df.columns.get_level_values(0).unique().tolist()
    for t in tickers:
        if (t, "Adj Close") not in df.columns and (t, "Close") in df.columns:
            df[(t, "Adj Close")] = df[(t, "Close")]

    # Reorder strictly: [symbols x FIELDS_DESIRED]
    desired = pd.MultiIndex.from_product([_symbols(), FIELDS_DESIRED])
    keep = [c for c in desired if c in df.columns]
    df = df.loc[:, keep]
    return df

def fetch_latest_data(tickers: List[str]) -> pd.DataFrame:
    """
    Download today's 1-minute bars for all tickers, normalized to (ticker, field) with desired fields.
    """
    df = yf.download(
        tickers,
        period=DEFAULT_PERIOD,
        interval=DEFAULT_INTERVAL,
        group_by="ticker",   # ticker-first columns
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("No intraday data returned (market closed or network issue).")
    df.index = pd.to_datetime(df.index)
    df = _ensure_intraday_columns(df)
    # drop exact duplicate index stamps (rare)
    df = df[~df.index.duplicated(keep="last")]
    return df

def _append_parquet_atomic(path: Path, new_df: pd.DataFrame) -> None:
    """
    Append by read→concat→dedupe→sort, write atomically via temp file.
    """
    if path.exists():
        old = pd.read_parquet(path)
        # ensure both have same column order
        old = _ensure_intraday_columns(old)
        combined = pd.concat([old, new_df], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df.sort_index()

    tmp = path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=True, compression="gzip")
    tmp.replace(path)

def _reset_if_new_trading_day(path: Path) -> None:
    """
    If market is open and the parquet contains rows from a prior NY date,
    keep only rows whose NY-local date == today.
    """
    if not path.exists() or not market_is_open():
        return

    df = pd.read_parquet(path)
    if df.empty:
        return

    today_ny = _now_ny().date()
    dates_ny = _index_dates_ny(df.index)
    last_date = dates_ny[-1]
    if last_date == today_ny:
        return  # already today's file

    df_today = df[dates_ny == today_ny]
    tmp = path.with_suffix(".tmp.parquet")
    df_today.to_parquet(tmp, index=True, compression="gzip")
    tmp.replace(path)


def tick_once() -> str:
    """
    One cron-safe cycle.
    Returns:
      - "market_closed" when outside hours,
      - "no_data" when open but data unavailable (holiday/early close/API issue),
      - or a success message.
    """
    if not market_is_open():
        return "market_closed"

    _reset_if_new_trading_day(RAW_FILE)
    symbols = _symbols()
    try:
        df = fetch_latest_data(symbols)
    except RuntimeError:
        return "no_data"

    if df.empty:
        return "no_data"

    _append_parquet_atomic(RAW_FILE, df)
    shape = pd.read_parquet(RAW_FILE).shape
    return f"appended rows; file shape={shape}"


# -------- CLI --------
def _parse_args():
    ap = argparse.ArgumentParser(description="Intraday cache writer (1m).")
    ap.add_argument("cmd", choices=["tick", "daemon", "reset"], help="tick=one pass; daemon=loop; reset=clear to today's data")
    ap.add_argument("--interval", type=int, default=60, help="daemon sleep seconds (default 60)")
    return ap.parse_args()

def _daemon_loop(sleep_sec: int = 60):
    while True:
        try:
            status = tick_once()
            print(f"[{_now_utc().isoformat()}] {status}")
        except Exception as e:
            print(f"[{_now_utc().isoformat()}] error: {e}")
        finally:
            # sleep irrespective of success
            import time as _time
            _time.sleep(max(10, sleep_sec))

def _reset_today():
    """Clear the parquet to only today's NY-local rows (if any)."""
    today_ny = _now_ny().date()
    if RAW_FILE.exists():
        df = pd.read_parquet(RAW_FILE)
        if not df.empty:
            dates_ny = _index_dates_ny(df.index)
            df_today = df[dates_ny == today_ny]
        else:
            df_today = df
        tmp = RAW_FILE.with_suffix(".tmp.parquet")
        df_today.to_parquet(tmp, index=True, compression="gzip")
        tmp.replace(RAW_FILE)
    print("reset_done")

if __name__ == "__main__":
    args = _parse_args()
    if args.cmd == "tick":
        print(tick_once())
    elif args.cmd == "daemon":
        _daemon_loop(sleep_sec=args.interval)
    else:
        _reset_today()
