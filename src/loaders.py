# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import os
import pandas as pd

try:
    from src.config import TICKERS, DEFAULT_BENCH, RF_ANN_PCT, METADATA
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.config import TICKERS, DEFAULT_BENCH, RF_ANN_PCT, METADATA

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw" / "intraday"

def symbols(include_bench=True) -> List[str]:
    s = list(dict.fromkeys(TICKERS + ([DEFAULT_BENCH] if include_bench else [])))
    return s

def processed_column_path() -> Path:
    # pick latest processed 'column' file
    cands = list(PROC.glob("historical_3y_to_*_column.parquet"))
    if not cands:
        raise FileNotFoundError("No processed 'column' parquet found.")
    return max(cands, key=lambda p: p.stat().st_mtime)

def intraday_file_path() -> Path:
    return RAW / "latest_data_tickers.parquet"

def load_prices_adj() -> pd.DataFrame:
    df = pd.read_parquet(processed_column_path())
    # columns: (field, ticker) -> Adj Close matrix
    px = df.xs("Adj Close", level=0, axis=1)
    cols = [s for s in symbols() if s in px.columns]
    return px.loc[:, cols].sort_index()

def file_mtime(path: Path) -> float:
    return path.stat().st_mtime

def defaults() -> Dict[str, object]:
    return {"rf_annual": float(RF_ANN_PCT), "metadata": METADATA}
