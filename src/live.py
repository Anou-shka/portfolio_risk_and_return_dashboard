# # src/live.py
# from __future__ import annotations

# from pathlib import Path
# from typing import List
# import pandas as pd
# import yfinance as yf
# import datetime as dt
# from datetime import timezone  # for tz-aware UTC
# import time

# # --- paths ---
# ROOT = Path(__file__).resolve().parents[1]
# DATA = ROOT / "data"
# DIR_RAW_INTRADAY = DATA / "raw" / "intraday"
# DIR_RAW_INTRADAY.mkdir(parents=True, exist_ok=True)

# # your config
# from config import TICKERS, DEFAULT_BENCH

# DEFAULT_PERIOD = "1d"   # fetch today's intraday
# DEFAULT_INTERVAL = "1m" # 1-minute bars
# RAW_FILE = DIR_RAW_INTRADAY / "latest_data_tickers.parquet"  # single file for all symbols

# # ---------- market gate (simple ET) ----------
# def market_is_open() -> bool:
#     """
#     Coarse NYSE hours (Mon-Fri, 09:30–16:00 ET).
#     Uses tz-aware UTC clock; ET approx = UTC-4 (simple heuristic).
#     """
#     now_utc = dt.datetime.now(timezone.utc)
#     et = now_utc - dt.timedelta(hours=4)  # rough EDT; fine for an open/closed gate
#     if et.weekday() >= 5:
#         return False
#     after_open = (et.hour > 9) or (et.hour == 9 and et.minute >= 30)
#     before_close = et.hour < 16
#     return after_open and before_close

# # ---------- fetch intraday (today) ----------
# def fetch_latest_data(tickers: List[str]) -> pd.DataFrame:
#     """
#     Download today's 1-minute bars for the given tickers.
#     Returns a DataFrame with MultiIndex columns:
#       level-0 = ticker (because group_by='ticker'), level-1 = field (Open/High/Low/Close/Adj Close/Volume)
#     """
#     df = yf.download(
#         tickers,
#         period=DEFAULT_PERIOD,
#         interval=DEFAULT_INTERVAL,
#         group_by="ticker",   # <-- singular
#         auto_adjust=False,
#         progress=False,
#         threads=True,
#     )
#     if df.empty:
#         raise RuntimeError("No intraday data returned (market closed or network issue).")
#     df.index = pd.to_datetime(df.index)
#     return df

# # ---------- safe append (no append kw in pandas/pyarrow) ----------
# def append_parquet_atomic(path: Path, new_df: pd.DataFrame) -> None:
#     """
#     Read existing parquet (if any), concat with new_df, drop duplicate timestamps,
#     sort by index, and write atomically via a temp file.
#     """
#     if path.exists():
#         old = pd.read_parquet(path)
#         # concat and keep the last occurrence for duplicate timestamps
#         combined = pd.concat([old, new_df], axis=0)
#         combined = combined[~combined.index.duplicated(keep="last")]
#         combined = combined.sort_index()
#     else:
#         combined = new_df.sort_index()

#     tmp = path.with_suffix(".tmp.parquet")
#     combined.to_parquet(tmp, index=True, compression="gzip")
#     tmp.replace(path)

# # ---------- one cycle you can run ----------
# def run_once() -> None:
#     """
#     If market is open: fetch today's intraday (1m) and append to raw parquet.
#     If market is closed: no-op here (do your EOD finalize in a separate function).
#     """
#     symbols = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
#     if not market_is_open():
#         print("Market closed → skipping intraday fetch.")
#         return
#     df = fetch_latest_data(symbols)
#     append_parquet_atomic(RAW_FILE, df)
#     print(f"Wrote/updated {RAW_FILE} → shape {pd.read_parquet(RAW_FILE).shape}")

# # # ---------- clear previous day's data ----------
# # def clear_previous_day_data(path: Path) -> None:
# #         """
# #         Remove rows from the parquet file that correspond to the previous day's date.
# #         """
# #         if path.exists():
# #             df = pd.read_parquet(path)
# #             today = dt.datetime.now(timezone.utc).date()
# #             previous_day = today - dt.timedelta(days=1)
# #             df = df[df.index.date != previous_day]
# #             df.to_parquet(path, index=True, compression="gzip")

# #     # Call the function before fetching new data
# #     clear_previous_day_data(RAW_FILE)


# # ---------- script entry ----------
# if __name__ == "__main__":
#     while True: 
#         run_once()
#         time.sleep(60)  # wait for 1 minute before the next fetch

# src/live.py
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
    At the first minute of a new trading day *while market is open*, clear yesterday’s rows.
    Implementation: keep only rows whose index.date == today (NY).
    If file doesn’t exist, no-op.
    """
    if not path.exists():
        return
    if not market_is_open():
        return

    today_ny = _now_ny().date()
    df = pd.read_parquet(path)
    if df.empty:
        return

    # If the latest row already belongs to today (NY date), do nothing.
    last_date = pd.to_datetime(df.index).date[-1]
    if last_date == today_ny:
        return

    # Otherwise, keep only today's rows (likely none at the first minute)
    df_today = df[pd.to_datetime(df.index).date == today_ny]
    tmp = path.with_suffix(".tmp.parquet")
    df_today.to_parquet(tmp, index=True, compression="gzip")
    tmp.replace(path)

def tick_once() -> str:
    """
    One cron-safe cycle:
      - If market open: reset file at start-of-day, fetch 1m history for today, append, done.
      - If market closed: no-op.
    Returns a short status string for logs.
    """
    if not market_is_open():
        return "market_closed"

    _reset_if_new_trading_day(RAW_FILE)
    symbols = _symbols()
    df = fetch_latest_data(symbols)
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
    # force reset based on today's NY date
    today_ny = _now_ny().date()
    if RAW_FILE.exists():
        df = pd.read_parquet(RAW_FILE)
        df_today = df[pd.to_datetime(df.index).date == today_ny]
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
