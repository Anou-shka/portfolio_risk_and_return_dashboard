# src/data_fetch.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd
import yfinance as yf
import datetime as dt
from zoneinfo import ZoneInfo
import argparse

# import config robustly whether run as module or script
try:
    from src.config import TICKERS, DEFAULT_BENCH
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.config import TICKERS, DEFAULT_BENCH

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIR_PROCESSED = DATA / "processed"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---- settings ----
CUTOFF_DATE = pd.Timestamp("2025-08-20")                  # inclusive
START_DATE  = (CUTOFF_DATE - pd.DateOffset(years=3))
END_DATE    = (CUTOFF_DATE + pd.Timedelta(days=1))        # yfinance end is exclusive
FIELDS      = ["Open", "High", "Low", "Close", "Adj Close"]

COL_PATH = DIR_PROCESSED / f"historical_3y_to_{CUTOFF_DATE.date()}_column.parquet"  # (field, ticker)
# TIC_PATH = DIR_PROCESSED / f"historical_3y_to_{CUTOFF_DATE.date()}_ticker.parquet"  # (ticker, field)


# ------------------------
# Helpers
# ------------------------
def _symbols() -> List[str]:
    return list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))  # stable order


def _download_range(symbols: List[str], group_by: str) -> pd.DataFrame:
    """Download [START_DATE, CUTOFF_DATE] OHLC + Adj Close; drop Volume; enforce order."""
    df = yf.download(
        symbols,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
        auto_adjust=False,
        group_by=group_by,     # 'column' or 'ticker'
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Check tickers/network.")
    df.index = pd.to_datetime(df.index)
    df = df[df.index <= CUTOFF_DATE]

    if group_by == "column":
        df = df[FIELDS]
        desired = pd.MultiIndex.from_product([FIELDS, _symbols()])
        keep = [c for c in desired if c in df.columns]
        df = df.loc[:, keep]
    elif group_by == "ticker":
        df = df.drop(columns="Volume", level=1, errors="ignore")
        desired = pd.MultiIndex.from_product([_symbols(), FIELDS])
        keep = [c for c in desired if c in df.columns]
        df = df.loc[:, keep]
    else:
        raise ValueError("group_by must be 'column' or 'ticker'")
    return df


def _download_day(symbols: List[str], date: dt.date) -> pd.DataFrame:
    """
    Official daily OHLC + Adj Close for one date.
    Returns columns as (field, ticker) to match COL_PATH orientation.
    """
    start = pd.Timestamp(date)
    end = start + pd.Timedelta(days=1)
    df = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError(f"No daily bar available for {date}.")
    df = df[FIELDS]
    df.index = pd.to_datetime(df.index)
    df = df[df.index.normalize() == pd.Timestamp(date)]
    if df.empty:
        raise RuntimeError(f"Official bar for {date} not posted yet.")
    # keep exactly one row
    df = df.groupby(df.index.normalize()).tail(1)
    # enforce symbol order
    desired = pd.MultiIndex.from_product([FIELDS, _symbols()])
    keep = [c for c in desired if c in df.columns]
    return df.loc[:, keep]


def _append_or_replace_date(base_path: Path, add_row: pd.DataFrame, date: dt.date) -> None:
    """
    Append/replace a single date row, then write atomically.
    Works for either orientation (we pass the right shape in).
    """
    if base_path.exists():
        base = pd.read_parquet(base_path)
        # align columns (union), drop same date, concat
        base, add_row = base.align(add_row, join="outer", axis=1)
        base = base[base.index.normalize() != pd.Timestamp(date)]
        out = pd.concat([base, add_row], axis=0).sort_index()
    else:
        out = add_row.sort_index()

    tmp = base_path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=True, compression="gzip")
    tmp.replace(base_path)


def _is_past_close_ny(now_utc: Optional[dt.datetime] = None) -> bool:
    """Past 4:00pm America/New_York (handles DST)."""
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    close = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return ny >= close


# ------------------------
# Public entry points
# ------------------------
def ensure_history_to_cutoff() -> None:
    """Write both processed parquet files up to CUTOFF_DATE (idempotent)."""
    if COL_PATH.exists():
        print("[init] processed files already exist — nothing to do.")
        return

    symbols = _symbols()
    df_col = _download_range(symbols, group_by="column")   # (field, ticker)
    df_tic = _download_range(symbols, group_by="ticker")   # (ticker, field)

    df_col.to_parquet(COL_PATH, index=True, compression="gzip")
    # df_tic.to_parquet(TIC_PATH, index=True, compression="gzip")

    print(f"[init] wrote {COL_PATH}  shape={df_col.shape}")
    # print(f"[init] wrote {TIC_PATH}  shape={df_tic.shape}")


def update_parquet_daily(force: bool = False) -> str:
    """
    Fetch the latest official daily bar and append it to both parquet files
    **only when** the US market is closed and today isn't already saved.
    Use `force=True` to try regardless of time.
    """
    # ensure base history exists
    if not (COL_PATH.exists()):
        ensure_history_to_cutoff()

    today = dt.datetime.now(dt.timezone.utc).date()
    # Check if today already saved (look at column-oriented file)
    last_saved = pd.read_parquet(COL_PATH).index.max()
    if pd.notna(last_saved) and last_saved.normalize().date() == today and not force:
        return "Up-to-date: today's row already saved."

    # Gate on market close unless forced
    if not force and not _is_past_close_ny():
        return "Skipped: market not past NY close yet."

    symbols = _symbols()

    # Try to fetch today's official bar
    try:
        day_col = _download_day(symbols, today)  # (field, ticker)
    except Exception as e:
        # If not yet posted, try yesterday (e.g., if running right after midnight UTC)
        yday = today - dt.timedelta(days=1)
        try:
            day_col = _download_day(symbols, yday)
            today = yday  # append for yesterday
        except Exception as e2:
            return f"No bar available yet: {e2}"

    # Build ticker-oriented version
    day_tic = day_col.swaplevel(0, 1, axis=1).sort_index(axis=1)

    _append_or_replace_date(COL_PATH, day_col, today)
    # _append_or_replace_date(TIC_PATH, day_tic, today)

    return f"Appended official bar for {today}."


# ------------------------
# CLI
# ------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Init history up to cutoff and/or append daily EOD.")
    ap.add_argument("cmd", choices=["init", "update"], nargs="?", default="init")
    ap.add_argument("--force", action="store_true", help="Ignore market-close gate for update.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.cmd == "init":
        ensure_history_to_cutoff()
    else:
        print(update_parquet_daily(force=args.force))
