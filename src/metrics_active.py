# src/metrics_active.py
"""Latest-session ("today") returns from daily and intraday price data.

Each asset is measured against the close of the session *before its own latest
session*. That matters because assets keep different calendars: on a Saturday
a stock's latest session is Friday while BTC-USD's is Saturday, and on a
weekday morning before 09:30 ET a stock's latest session is still yesterday.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.metrics_core import is_crypto

NY_TZ = "America/New_York"


def session_tz(symbol: str) -> str:
    """Timezone whose calendar date defines a trading session for `symbol`.
    Yahoo stamps crypto daily bars at 00:00 UTC; US listings use New York."""
    return "UTC" if is_crypto(symbol) else NY_TZ


def latest_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol return of the latest daily bar vs the previous one.

    `prices` is a raw (not calendar-aligned) daily close matrix. Returns a frame
    indexed by symbol with columns: ret, last, prev, date, prev_date.
    """
    rows = {}
    for sym in prices.columns:
        s = prices[sym].dropna()
        if len(s) < 2 or s.iloc[-2] <= 0:
            continue
        rows[sym] = {
            "ret": float(s.iloc[-1] / s.iloc[-2] - 1.0),
            "last": float(s.iloc[-1]),
            "prev": float(s.iloc[-2]),
            "date": s.index[-1],
            "prev_date": s.index[-2],
        }
    cols = ["ret", "last", "prev", "date", "prev_date"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def _to_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def intraday_session_returns(
    intra_close: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    """Minute-by-minute return of each symbol's latest session vs its prior close.

    intra_close  : intraday close matrix (columns = symbols), any tz.
    daily_prices : raw daily close matrix with tz-naive date index.

    Returns (returns matrix on a UTC index, {symbol: session date}). Each
    column is forward-filled, and the matrix starts once every symbol's latest
    session has begun, so every row has a value for every symbol.
    """
    if intra_close.empty:
        return pd.DataFrame(), {}
    intra_close = intra_close.copy()
    intra_close.index = _to_utc_index(intra_close.index)

    series: Dict[str, pd.Series] = {}
    sessions: Dict[str, pd.Timestamp] = {}
    for sym in intra_close.columns:
        s = intra_close[sym].dropna()
        if s.empty or sym not in daily_prices.columns:
            continue
        local_dates = s.index.tz_convert(session_tz(sym)).normalize().tz_localize(None)
        session = local_dates[-1]
        s = s[local_dates == session]
        daily = daily_prices[sym].dropna()
        prior = daily[daily.index.normalize() < session]
        if prior.empty or prior.iloc[-1] <= 0:
            continue
        series[sym] = s / float(prior.iloc[-1]) - 1.0
        sessions[sym] = session

    if not series:
        return pd.DataFrame(), {}
    start = max(s.index[0] for s in series.values())
    m = pd.DataFrame(series).sort_index().ffill()
    m = m.loc[m.index >= start]
    return m, sessions


def weighted_return(returns: pd.DataFrame | pd.Series, weights: pd.Series) -> pd.Series | float:
    """Weighted portfolio return using only the symbols present in `returns`,
    renormalising the weights over them."""
    cols = returns.index if isinstance(returns, pd.Series) else returns.columns
    w = weights.reindex([c for c in weights.index if c in cols]).fillna(0.0)
    if w.empty or not w.sum():
        return np.nan if isinstance(returns, pd.Series) else pd.Series(np.nan, index=returns.index)
    w = w / w.sum()
    if isinstance(returns, pd.Series):
        return float((returns.reindex(w.index) * w).sum())
    return returns[w.index].dot(w)
