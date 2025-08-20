# src/metrics_active.py
from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from src.metrics_core import mc_var_es

# paths
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw" / "intraday"

# defaults/filenames (adjust if you changed them)
PROCESSED_COLUMN = max((PROC.glob("historical_3y_to_*_column.parquet")), key=lambda p: p.stat().st_mtime)
INTRADAY_FILE   = RAW / "latest_data_tickers.parquet"

FIELDS = ["Open","High","Low","Close","Adj Close"]

def _extract_prices_adj_from_column(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    # df columns: (field, ticker)
    px = df.xs("Adj Close", level=0, axis=1)
    cols = [s for s in symbols if s in px.columns]
    return px.loc[:, cols].sort_index()

def load_prices_adj(symbols: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_COLUMN)
    return _extract_prices_adj_from_column(df, symbols)

def latest_intraday_last(intraday_path: Path, symbols: List[str]) -> pd.Series:
    """
    Return the latest 'last price' per symbol from today’s intraday file.
    File columns are MultiIndex: (ticker, field). We use 'Close' (or Adj Close).
    """
    if not intraday_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(intraday_path)
    if df.empty: return pd.Series(dtype=float)
    # columns: (ticker, field)
    if not isinstance(df.columns, pd.MultiIndex):
        # single ticker fallback
        sym = symbols[0]
        last = df["Close"].ffill().iloc[-1]
        return pd.Series({sym: float(last)})
    out = {}
    for s in symbols:
        if (s, "Close") in df.columns:
            out[s] = float(df[(s, "Close")].ffill().iloc[-1])
        elif (s, "Adj Close") in df.columns:
            out[s] = float(df[(s, "Adj Close")].ffill().iloc[-1])
    return pd.Series(out, dtype=float)

def prev_close_map(prices_adj: pd.DataFrame) -> Dict[str, float]:
    last_row = prices_adj.iloc[-1]
    return {c: float(last_row[c]) for c in prices_adj.columns}

def today_returns(last: pd.Series, prev_map: Dict[str,float]) -> pd.Series:
    ret = {}
    for s, p in last.items():
        pc = prev_map.get(s, np.nan)
        if pc and not np.isnan(pc) and pc>0:
            ret[s] = p/pc - 1.0
        else:
            ret[s] = np.nan
    return pd.Series(ret, dtype=float).sort_index()

def overlay_prices(prices_adj: pd.DataFrame, last: pd.Series, prev_map: Dict[str,float]) -> pd.DataFrame:
    """
    prices_adj (official history) + synthetic 'today' row using last/prev_close ratio.
    """
    if last.empty: return prices_adj
    yday = prices_adj.iloc[-1]
    new_vals = {}
    for s in prices_adj.columns:
        pc = prev_map.get(s, np.nan)
        if s in last and pc and not np.isnan(pc) and pc>0:
            new_vals[s] = float(yday[s]) * (float(last[s]) / pc)
        else:
            new_vals[s] = float(yday[s])
    today_idx = pd.Timestamp.today().normalize()
    out = prices_adj.copy()
    if today_idx in out.index:
        out.loc[today_idx, list(new_vals.keys())] = pd.Series(new_vals)
    else:
        out = pd.concat([out, pd.DataFrame([new_vals], index=[today_idx])]).sort_index()
    return out

def live_portfolio_metrics(
    prices_adj: pd.DataFrame,
    last: pd.Series,
    weights: Dict[str,float],
    bench_symbol: str,
    daily_returns: pd.DataFrame,   # historical daily returns (from prices_adj)
    rf_annual: float = 0.0,
) -> Dict[str, object]:
    prev_map = prev_close_map(prices_adj)
    trets = today_returns(last, prev_map)  # per symbol
    # today portfolio & bench
    cols = [c for c in prices_adj.columns if c in trets.index]
    w = pd.Series({k: weights.get(k, 0.0) for k in cols}, dtype=float)
    if w.sum(): w = w / w.sum()
    port_today = float((trets[cols] * w).sum())
    bench_today = float(trets.get(bench_symbol, np.nan))
    rel_today = port_today - bench_today if not np.isnan(bench_today) else np.nan

    # drifted weights (based on today moves)
    rel_moves = (trets[cols].fillna(0) + 1.0)
    drifted = (w * rel_moves)
    if drifted.sum(): drifted = drifted / drifted.sum()

    # VaR nowcast (parametric on historical cov)
    var1d, es1d = mc_var_es(daily_returns[cols], drifted, alpha=0.95, horizon_days=1, n_sims=10_000, seed=42)

    return {
        "today_returns_by_symbol": trets,
        "portfolio_today": port_today,
        "benchmark_today": bench_today,
        "relative_today": rel_today,
        "drifted_weights": drifted.sort_values(ascending=False),
        "var_1d": var1d,
        "es_1d": es1d,
    }


# --- add to src/metrics_active.py ---
from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path

def intraday_close_matrix(intraday_path: Path, symbols: List[str]) -> pd.DataFrame:
    """
    Return minute-by-minute Close for the selected symbols from the intraday parquet.
    Columns: symbols, Index: minute timestamps (tz-naive or UTC from parquet).
    """
    if not intraday_path.exists():
        return pd.DataFrame(columns=symbols)
    df = pd.read_parquet(intraday_path)
    if df.empty:
        return pd.DataFrame(columns=symbols)
    # Normalize to (ticker, field)
    if not isinstance(df.columns, pd.MultiIndex):
        # single ticker fallback
        return pd.DataFrame({symbols[0]: df["Close"]}, index=df.index)

    out = {}
    for s in symbols:
        if (s, "Close") in df.columns:
            out[s] = df[(s, "Close")].ffill()
        elif (s, "Adj Close") in df.columns:
            out[s] = df[(s, "Adj Close")].ffill()
    if not out:
        return pd.DataFrame(columns=symbols)
    m = pd.DataFrame(out).sort_index()
    # drop duplicated stamps just in case
    m = m[~m.index.duplicated(keep="last")]
    return m

def intraday_return_matrix(intra_close: pd.DataFrame, prev_close: Dict[str, float]) -> pd.DataFrame:
    """
    Minute-by-minute return since prior official close: Close_t / prev_close - 1.
    """
    if intra_close.empty:
        return intra_close
    prev = pd.Series(prev_close)
    # keep only symbols present in intra_close
    prev = prev.reindex(intra_close.columns).astype(float)
    # avoid division by zero
    prev[~np.isfinite(prev) | (prev <= 0)] = np.nan
    return intra_close.divide(prev, axis=1) - 1.0
