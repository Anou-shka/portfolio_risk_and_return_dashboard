# app/app_sanity.py
from __future__ import annotations

# --- path shim so "import src" works when running from repo root or app/ ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---- config & metrics ----
from src.config import TICKERS, DEFAULT_BENCH, RF_ANN_PCT, DEFAULT_WEIGHTS, METADATA
from src.metrics_core import (
    price_to_returns, portfolio_returns, nav_from_returns, ann_return, ann_vol, sharpe,
    beta_alpha, tracking_error, information_ratio, max_drawdown, cov_matrix,
    risk_contributions, optimize_min_variance, optimize_max_sharpe,
    trace_efficient_frontier, mc_var_es, relative_return_series
)
from src.metrics_active import (
    latest_intraday_last, intraday_close_matrix, intraday_return_matrix, overlay_prices
)

# ---- file helpers (no dependency on src.loaders) ----
PROCDIR = Path(ROOT) / "data" / "processed"
INTRADIR = Path(ROOT) / "data" / "raw" / "intraday"

def processed_column_path() -> Path:
    candidates = sorted(PROCDIR.glob("historical_3y_to_*_column.parquet"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        st.error("No processed parquet found. Run: `python -m src.data_fetch init` first.")
        st.stop()
    return candidates[0]

def intraday_file_path() -> Path:
    return INTRADIR / "latest_data_tickers.parquet"

def load_prices_adj_from_column(p: Path, symbols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(p)  # columns: (field, ticker)
    px = df.xs("Adj Close", level=0, axis=1)
    cols = [s for s in symbols if s in px.columns]
    return px.loc[:, cols].sort_index()

# ---- cache helpers ----
@st.cache_data(show_spinner=False)
def load_core(px_path: str):
    p = Path(px_path)
    # keep benchmark last for deterministic column order
    syms = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
    prices = load_prices_adj_from_column(p, syms)
    rets = price_to_returns(prices)
    return prices, rets

@st.cache_data(ttl=60, show_spinner=False)
def load_intraday(intra_path: str, symbols_all: list[str]):
    p = Path(intra_path)
    last = latest_intraday_last(p, symbols_all)              # per-symbol last price (today)
    close_m = intraday_close_matrix(p, symbols_all)          # minute × symbols
    return last, close_m

# ---- UI ----
st.set_page_config(page_title="Portfolio — Sanity Dashboard", layout="wide")
st.title("✅ Sanity Check: Metrics & Visuals")

SYMS_ALL = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
BENCH = DEFAULT_BENCH
PROC = processed_column_path()
INTRA = intraday_file_path()

with st.sidebar:
    st.subheader("Filters")
    sel = st.multiselect("Assets", options=SYMS_ALL, default=[s for s in TICKERS if s in SYMS_ALL])
    if not sel:
        st.stop()
    bench = st.selectbox("Benchmark", options=[BENCH] + [s for s in sel if not s.startswith("^")], index=0)
    date_range = st.radio("Date range", ["Today","1W","1M","6M","1Y","3Y","MAX"], index=3)
    equal = st.toggle("Equal weights (exclude benchmark)", value=True)
    rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=0.2, value=float(RF_ANN_PCT), step=0.005)
    st.caption("Intraday view auto-refreshes ~every 60s.")

# ---- load data ----
prices, rets = load_core(str(PROC))
last, intra_close = load_intraday(str(INTRA), SYMS_ALL)

# ---- choose weights ----
if equal:
    base_w = {s: 0.0 if s == bench else 1.0 for s in sel}
else:
    # use config defaults if available; else equal
    w_raw = {s: DEFAULT_WEIGHTS.get(s, 0.0) for s in sel}
    if bench in w_raw: w_raw[bench] = 0.0
    if sum(w_raw.values()) == 0:
        base_w = {s: 0.0 if s == bench else 1.0 for s in sel}
    else:
        base_w = w_raw

# normalize
w = pd.Series({k: v for k, v in base_w.items() if k != bench}, dtype=float)
if w.sum(): w = w / w.sum()

# ---- date slicing ----
today = prices.index.max()
def cutoff(label: str) -> pd.Timestamp:
    if label == "Today": return today
    if label == "1W": return today - pd.Timedelta(days=7)
    if label == "1M": return today - pd.Timedelta(days=30)
    if label == "6M": return today - pd.Timedelta(days=182)
    if label == "1Y": return today - pd.Timedelta(days=365)
    if label == "3Y": return today - pd.Timedelta(days=365*3)
    return prices.index.min()

view_syms = list(dict.fromkeys(sel + [bench]))
px_view = prices.loc[:, [c for c in view_syms if c in prices.columns]]

# =======================
# TABS
# =======================
tab_perf, tab_live, tab_risk, tab_div = st.tabs(["Performance (daily)", "Live (today)", "Risk & Frontier", "Diversification"])

# -----------------------
# Performance (daily)
# -----------------------
with tab_perf:
    if date_range == "Today":
        st.info("Switch to 1W/1M/… to see historical metrics.")
    else:
        cut = cutoff(date_range)
        prices_hist = px_view.loc[px_view.index >= cut]    # <-- rename
        rets_view = price_to_returns(prices_hist)
        port = portfolio_returns(rets_view.drop(columns=[bench], errors="ignore"), w)
        bench_s = rets_view.get(bench)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ann. Return", f"{ann_return(port):.2%}")
        c2.metric("Ann. Vol", f"{ann_vol(port):.2%}")
        c3.metric("Sharpe", f"{sharpe(port, rf_annual=rf):.2f}")
        b, a = beta_alpha(port, bench_s, rf_annual=rf)
        c4.metric("Beta", f"{b:.2f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Tracking Error", f"{tracking_error(port, bench_s):.2%}")
        c6.metric("Info Ratio", f"{information_ratio(port, bench_s):.2f}")
        dd = max_drawdown(port)
        c7.metric("Max Drawdown", f"{dd.max_drawdown:.2%}")

        nav_df = pd.DataFrame({
            "Portfolio": nav_from_returns(port),
            bench: nav_from_returns(bench_s) if bench_s is not None else pd.Series(dtype=float)
        }).dropna()
        for t in [s for s in sel if s in rets.columns and s != bench]:
            nav_df[t] = nav_from_returns(rets.loc[nav_df.index, t])
        st.plotly_chart(px.line(nav_df, title="Cumulative NAV"), use_container_width=True)  # px is Plotly again

        rel = relative_return_series(port, bench_s)
        st.plotly_chart(px.area(rel.rename("Relative vs Benchmark"), title=f"Relative Outperformance vs {bench}"),
                use_container_width=True)

        # Rolling 60D vol & beta
        df_rb = pd.concat([port.rename("p"), bench_s.rename("b")], axis=1).dropna()
        if not df_rb.empty:
            roll_vol = df_rb["p"].rolling(60).std(ddof=1) * np.sqrt(252)
            st.plotly_chart(px.line(roll_vol, title="Rolling 60D Annualized Vol (Portfolio)"), use_container_width=True)
            cov_pb = df_rb["p"].rolling(60).cov(df_rb["b"])
            var_b  = df_rb["b"].rolling(60).var()
            beta_roll = (cov_pb / var_b).rename("beta")
            st.plotly_chart(px.line(beta_roll, title="Rolling 60D Beta vs Benchmark"), use_container_width=True)

# -----------------------
# Live (today, intraday)
# -----------------------
with tab_live:
    st.caption("Auto-updates every ~60s (cache TTL).")
    if not INTRA.exists():
        st.info("No intraday file yet (run `python -m src.live daemon` during market hours).")
    elif intra_close.empty:
        st.info("Intraday file present but no data yet for selected symbols.")
    else:
        # Today minute returns since prior close
        prev_map = {c: float(prices.iloc[-1][c]) for c in prices.columns}

        cols = [c for c in view_syms if c in intra_close.columns]
        if not cols:
            st.info("No intraday columns for the selected symbols.")
        else:
            intra_close_view = intra_close.loc[:, cols]
            rets_min = intraday_return_matrix(intra_close_view, prev_map)

            if rets_min.empty:
                st.info("No intraday returns yet for the selected symbols.")
            else:
                port_today = (rets_min.drop(columns=[bench], errors="ignore").dot(w)).rename("Portfolio (today)")
                chart_df = pd.concat([port_today, rets_min.get(bench)], axis=1).dropna()
                st.plotly_chart(
                    px.line(chart_df, title="Intraday Return (since prior close)"),
                    use_container_width=True
                )

                # Latest “today” cards
                if len(chart_df):
                    p_now = chart_df["Portfolio (today)"].iloc[-1]
                    b_now = chart_df[bench].iloc[-1] if bench in chart_df.columns else np.nan
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Today (Portfolio)", f"{p_now:.2%}")
                    c2.metric(f"Today ({bench})", f"{(b_now if pd.notna(b_now) else 0):.2%}")
                    c3.metric("Today Relative", f"{(p_now - b_now if pd.notna(b_now) else np.nan):.2%}")

                st.write("Today returns by symbol (latest):")
                last_ret = rets_min.tail(1).T.rename(columns=lambda _: "today %")
                st.dataframe(last_ret.reindex(view_syms), use_container_width=True)

                # Overlay price (Adj Close + synthetic today)
                ov = overlay_prices(prices, latest_intraday_last(INTRA, view_syms), prev_map)
                st.plotly_chart(
                    px.line(
                        ov.loc[ov.index >= ov.index.max() - pd.Timedelta(days=7)],
                        title="Overlay Adj Close (last 7 days incl. synthetic today)"
                    ),
                    use_container_width=True
                )


# -----------------------
# Risk & Frontier (daily)
# -----------------------
with tab_risk:
    cut = cutoff("6M") if date_range == "Today" else cutoff(date_range)
    rets_view = price_to_returns(px_view.loc[px_view.index >= cut])
    cols = [c for c in rets_view.columns if c in sel and c != bench]
    R = rets_view[cols]
    if R.empty:
        st.info("Select at least one asset (ex-benchmark) with enough history.")
    else:
        Sigma = cov_matrix(R)
        w_now = w.reindex(cols).fillna(0.0)
        if w_now.sum(): w_now = w_now / w_now.sum()
        rc = risk_contributions(w_now, Sigma).sort_values(ascending=False)
        st.subheader("Risk Contributions")
        st.dataframe(rc.to_frame("risk %"), use_container_width=True)

        st.subheader("Efficient Frontier")
        try:
            ef = trace_efficient_frontier(R, n_points=25)
            if ef.empty:
                st.info("Install `scipy` to compute the frontier.")
            else:
                st.plotly_chart(px.scatter(ef, x="vol_ann", y="ret_ann", title="Efficient Frontier"),
                                use_container_width=True)
        except Exception as e:
            st.info(f"Frontier unavailable: {e}")

        var1d, es1d = mc_var_es(R, w_now, alpha=0.95, horizon_days=1, n_sims=10000, seed=42)
        st.caption(f"Parametric 1d VaR(95%): {var1d:.2%} | ES: {es1d:.2%}")

# -----------------------
# Diversification (by metadata)
# -----------------------
with tab_div:
    md = METADATA or {}
    # aggregate current weights by sector & region
    sec = {}
    reg = {}
    for k, val in w.items():
        lab = md.get(k, {})
        sec[k] = lab.get("sector", "Unknown") or "Unknown"
        reg[k] = lab.get("region", "Unknown") or "Unknown"
    sec_w = pd.Series(w).groupby(pd.Series(sec)).sum().sort_values(ascending=False)
    reg_w = pd.Series(w).groupby(pd.Series(reg)).sum().sort_values(ascending=False)

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(sec_w, title="Weights by Sector"), use_container_width=True)
    c2.plotly_chart(px.bar(reg_w, title="Weights by Region"), use_container_width=True)

st.caption("Tip: for live data, run the intraday writer during market hours: `python -m src.live daemon`.")
