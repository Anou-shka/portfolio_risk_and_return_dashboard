# app/app_explorer.py
from __future__ import annotations

# --- path shim so "import src" works from anywhere ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- project config & metrics ---
from src.config import TICKERS, DEFAULT_BENCH, RF_ANN_PCT, DEFAULT_WEIGHTS, METADATA
from src.metrics_core import (
    price_to_returns, portfolio_returns, nav_from_returns, ann_return, ann_vol, sharpe,
    beta_alpha, tracking_error, information_ratio, max_drawdown, cov_matrix,
    risk_contributions, optimize_min_variance, optimize_max_sharpe,
    trace_efficient_frontier, mc_var_es, relative_return_series
)

# ---------------------------------------
# Minimal file helpers (no src.loaders dep)
# ---------------------------------------
PROCDIR = Path(ROOT) / "data" / "processed"

def processed_column_path() -> Path:
    cands = sorted(PROCDIR.glob("historical_3y_to_*_column.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        st.error("No processed parquet found. Run: `python -m src.data_fetch init` first.")
        st.stop()
    return cands[0]

def load_prices_adj_from_column(p: Path, symbols: list[str]) -> pd.DataFrame:
    """Read column-oriented parquet; extract Adj Close for symbols in desired order."""
    df = pd.read_parquet(p)        # columns: (field, ticker)
    px = df.xs("Adj Close", level=0, axis=1)
    cols = [s for s in symbols if s in px.columns]
    return px.loc[:, cols].sort_index()

# ------------
# UI defaults
# ------------
ALL_SYMS = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
PROC = processed_column_path()

# ------------
# Streamlit UI
# ------------
st.set_page_config(page_title="Portfolio Explorer", layout="wide")
st.title("📈 Portfolio Explorer")

with st.sidebar:
    st.subheader("Controls")

    chosen = st.multiselect(
        "Assets to include",
        options=ALL_SYMS,
        default=[s for s in TICKERS if s in ALL_SYMS],
        help="Benchmark will be excluded from portfolio weights automatically."
    )
    if not chosen:
        st.stop()

    bench = st.selectbox("Benchmark", options=[DEFAULT_BENCH] + [s for s in chosen if not s.startswith("^")], index=0)

    range_choice = st.radio("Date range", ["Today","1W","1M","6M","1Y","3Y","MAX","Custom"], index=5, horizontal=True)
    custom = None
    if range_choice == "Custom":
        custom = st.date_input("Pick start/end", value=[])

    weight_mode = st.radio("Weights", ["Equal (ex-bench)", "YAML defaults", "Custom"], index=0)
    rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=0.2, value=float(RF_ANN_PCT), step=0.005)

# -------------------
# Load core time series
# -------------------
@st.cache_data(show_spinner=False)
def load_core(px_path: str, syms: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = Path(px_path)
    prices = load_prices_adj_from_column(p, list(syms))
    rets = price_to_returns(prices)
    return prices, rets

prices, rets = load_core(str(PROC), tuple(ALL_SYMS))

# guard: only keep relevant columns
VIEW_SYMS = list(dict.fromkeys(chosen + [bench]))
prices = prices.loc[:, [c for c in VIEW_SYMS if c in prices.columns]]
rets   = rets.loc[:,   [c for c in VIEW_SYMS if c in rets.columns]]

# -------------------
# Date slicing helper
# -------------------
today = prices.index.max()

def cutoff_ts(choice: str):
    if choice == "Today": return today, True
    if choice == "1W":    return today - pd.Timedelta(days=7), False
    if choice == "1M":    return today - pd.Timedelta(days=30), False
    if choice == "6M":    return today - pd.Timedelta(days=182), False
    if choice == "1Y":    return today - pd.Timedelta(days=365), False
    if choice == "3Y":    return today - pd.Timedelta(days=365*3), False
    if choice == "MAX":   return prices.index.min(), False
    # Custom
    if custom and len(custom) == 2:
        return pd.Timestamp(custom[0]), False
    return prices.index.min(), False

cut, is_today = cutoff_ts(range_choice)
if is_today:
    st.info("‘Today’ uses yesterday’s official close only (intraday lives in your other app). Choose ≥1W for historical stats.")
hist_px = prices.loc[prices.index >= cut]
hist_rt = rets.loc[rets.index >= cut]

# -------------------
# Build weights
# -------------------
def normalize_ex_bench(d: dict[str, float], bench_symbol: str) -> pd.Series:
    d = {k: float(v) for k,v in d.items()}
    d.pop(bench_symbol, None)
    s = pd.Series(d, dtype=float)
    return s / s.sum() if s.sum() else s

if weight_mode == "Equal (ex-bench)":
    w = {s: 0.0 if s == bench else 1.0 for s in chosen}
    w = normalize_ex_bench(w, bench)
elif weight_mode == "YAML defaults":
    base = {s: DEFAULT_WEIGHTS.get(s, 0.0) for s in chosen}
    w = normalize_ex_bench(base, bench)
else:
    st.write("Set custom weights (they will be renormalized):")
    cols = [s for s in chosen if s != bench]
    sliders = {}
    for c in cols:
        sliders[c] = st.slider(c, min_value=0.0, max_value=1.0, value=float(DEFAULT_WEIGHTS.get(c, 0.0)), step=0.01)
    w = normalize_ex_bench(sliders, bench)

# -------------------
# Compute metrics
# -------------------
port = portfolio_returns(hist_rt.drop(columns=[bench], errors="ignore"), w)
bench_s = hist_rt.get(bench)

tiles = {}
tiles["Ann. Return"]   = f"{ann_return(port):.2%}"
tiles["Ann. Vol"]      = f"{ann_vol(port):.2%}"
tiles["Sharpe"]        = f"{sharpe(port, rf_annual=rf):.2f}"
b, a = beta_alpha(port, bench_s, rf_annual=rf)
tiles["Beta"]          = f"{b:.2f}"
tiles["Tracking Error"]= f"{tracking_error(port, bench_s):.2%}"
tiles["Info Ratio"]    = f"{information_ratio(port, bench_s):.2f}"
dd = max_drawdown(port)
tiles["Max Drawdown"]  = f"{dd.max_drawdown:.2%}"

c1,c2,c3,c4 = st.columns(4)
c5,c6,c7    = st.columns(3)
for (k,v), col in zip(list(tiles.items())[:4], [c1,c2,c3,c4]): col.metric(k, v)
for (k,v), col in zip(list(tiles.items())[4:], [c5,c6,c7]): col.metric(k, v)

# -------------------
# Per-ticker comparisons
# -------------------
def per_ticker_metrics(rt: pd.DataFrame, bench_s: pd.Series, rf_annual: float) -> pd.DataFrame:
    rows = []
    for col in [c for c in rt.columns if c != bench]:
        s = rt[col].dropna()
        if s.empty: 
            continue
        beta,_ = beta_alpha(s, bench_s, rf_annual=rf_annual)
        rows.append({
            "Ticker": col,
            "AnnRet": ann_return(s),
            "AnnVol": ann_vol(s),
            "Sharpe": sharpe(s, rf_annual=rf_annual),
            "Beta": beta,
            "TE": tracking_error(s, bench_s),
            "IR": information_ratio(s, bench_s),
            "MaxDD": max_drawdown(s).max_drawdown
        })
    df = pd.DataFrame(rows).set_index("Ticker")
    return df

st.subheader("By Asset (same window & benchmark)")
by_asset = per_ticker_metrics(hist_rt, bench_s, rf)
st.dataframe(
    (by_asset * pd.Series({"AnnRet":100,"AnnVol":100,"TE":100,"MaxDD":100})).rename(
        columns={"AnnRet":"AnnRet %", "AnnVol":"AnnVol %","TE":"TE %","MaxDD":"MaxDD %"}
    ).fillna(by_asset),
    use_container_width=True
)

# -------------------
# Charts
# -------------------
st.subheader("Cumulative NAV")
nav_df = pd.DataFrame({
    "Portfolio": nav_from_returns(port),
    bench: nav_from_returns(bench_s) if bench_s is not None else pd.Series(dtype=float)
}).dropna()
for t in [s for s in chosen if s in hist_rt.columns and s != bench]:
    nav_df[t] = nav_from_returns(hist_rt.loc[nav_df.index, t])
st.plotly_chart(px.line(nav_df, title="Cumulative NAV (selected window)"), use_container_width=True)

st.subheader("Relative Outperformance vs Benchmark")
rel = relative_return_series(port, bench_s)
st.plotly_chart(px.area(rel.rename("Relative vs Benchmark")), use_container_width=True)

# st.subheader("Risk Contributions (current window)")
# cols = [c for c in hist_rt.columns if c in chosen and c != bench]
# R = hist_rt[cols]
# if not R.empty and len(cols) >= 2:
#     Sigma = cov_matrix(R)
#     w_now = w.reindex(cols).fillna(0.0)
#     if w_now.sum(): w_now = w_now / w_now.sum()
#     rc = risk_contributions(w_now, Sigma).sort_values(ascending=False)
#     st.plotly_chart(px.bar(rc, title="Risk Contribution (%)"), use_container_width=True)
# else:
#     st.info("Select at least two assets (excluding benchmark) to view risk contributions.")

# -------------------
# Risk Contributions (current window)
# -------------------
st.subheader("Risk & Contributions")

# portfolio & benchmark total risk (always useful to show)
port_vol  = ann_vol(port)
bench_vol = ann_vol(bench_s)

cols = [c for c in hist_rt.columns if c in chosen and c != bench]
R = hist_rt[cols]

# Layout: contributions on the left, total risk on the right
c_left, c_right = st.columns([2, 1])

with c_left:
    if not R.empty and len(cols) >= 2:
        Sigma = cov_matrix(R)
        w_now = w.reindex(cols).fillna(0.0)
        if w_now.sum(): 
            w_now = w_now / w_now.sum()
        rc = risk_contributions(w_now, Sigma).sort_values(ascending=False)  # sums to 1

        fig_rc = px.bar(
            rc.rename("Contribution"),
            title="Risk Contribution (weights × covariance)",
            labels={"index": "Ticker", "value": "Contribution"},
        )
        fig_rc.update_yaxes(tickformat=".0%")  # show as %
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("Select at least two assets (excluding benchmark) to view contributions.")

with c_right:
    vol_series = pd.Series(
        {"Portfolio": port_vol, bench: bench_vol}, 
        dtype=float, name="Ann. Vol"
    )
    fig_vol = px.bar(
        vol_series,
        title="Total Risk (Ann. Vol)",
        labels={"index": "Series", "value": "Ann. Vol"},
    )
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)


st.subheader("Efficient Frontier (optional)")
try:
    ef = trace_efficient_frontier(R, n_points=30)
    if ef.empty:
        st.caption("Install `scipy` to enable the frontier.")
    else:
        st.plotly_chart(px.scatter(ef, x="vol_ann", y="ret_ann", title="Efficient Frontier"), use_container_width=True)
except Exception as e:
    st.caption(f"Frontier unavailable: {e}")

# Optional: show current weights and metadata grouping
st.subheader("Current Weights")
st.dataframe(w.rename("weight").to_frame().style.format({"weight":"{:.2%}"}), use_container_width=True)

if METADATA:
    st.subheader("Diversification snapshot")
    lab_sec = {k: METADATA.get(k,{}).get("sector","Unknown") for k in w.index}
    lab_reg = {k: METADATA.get(k,{}).get("region","Unknown") for k in w.index}
    sec_w = pd.Series(w).groupby(pd.Series(lab_sec)).sum().sort_values(ascending=False)
    reg_w = pd.Series(w).groupby(pd.Series(lab_reg)).sum().sort_values(ascending=False)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(sec_w, title="By Sector"), use_container_width=True)
    c2.plotly_chart(px.bar(reg_w, title="By Region"), use_container_width=True)
