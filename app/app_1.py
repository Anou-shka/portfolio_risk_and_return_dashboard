# app/app_showcase.py
from __future__ import annotations

# ---- path shim so "import src" works no matter where you run this ----
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---- project imports ----
from src.config import TICKERS, DEFAULT_BENCH, RF_ANN_PCT, DEFAULT_WEIGHTS, METADATA
from src.metrics_core import (
    price_to_returns, portfolio_returns, nav_from_returns, ann_return, ann_vol, sharpe,
    beta_alpha, tracking_error, information_ratio, max_drawdown, cov_matrix,
    risk_contributions, optimize_min_variance, optimize_max_sharpe,
    trace_efficient_frontier, mc_var_es, relative_return_series,
)
from src.metrics_active import (
    latest_intraday_last, intraday_close_matrix, intraday_return_matrix, overlay_prices,
)

# ---------------- helpers (file paths, loaders) ----------------
PROCDIR = Path(ROOT) / "data" / "processed"
INTRADIR = Path(ROOT) / "data" / "raw" / "intraday"

def processed_column_path() -> Path:
    cands = sorted(PROCDIR.glob("historical_3y_to_*_column.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        st.error("No processed parquet found. Run `python -m src.data_fetch init` first.")
        st.stop()
    return cands[0]

def intraday_file_path() -> Path:
    return INTRADIR / "latest_data_tickers.parquet"

def load_prices_adj_from_column(p: Path, symbols: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(p)                 # columns: (field, ticker)
    px_adj = df.xs("Adj Close", level=0, axis=1)
    cols = [s for s in symbols if s in px_adj.columns]
    return px_adj.loc[:, cols].sort_index()

@st.cache_data(show_spinner=False)
def load_core(_path: str, syms: tuple[str, ...]):
    p = Path(_path)
    prices = load_prices_adj_from_column(p, list(syms))
    rets = price_to_returns(prices)
    return prices, rets

@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_cached(_path: str, symbols: tuple[str, ...]):
    p = Path(_path)
    last = latest_intraday_last(p, list(symbols))
    close_m = intraday_close_matrix(p, list(symbols))
    return last, close_m

def normalize_ex_bench(d: Dict[str, float], bench: str) -> pd.Series:
    d = {k: float(v) for k,v in d.items()}
    d.pop(bench, None)
    s = pd.Series(d, dtype=float)
    return s / s.sum() if s.sum() else s

# -------------------- page setup & css --------------------
st.set_page_config(page_title="Portfolio Showcase", layout="wide")
st.markdown("""
<style>
/* compact top padding */
.block-container { padding-top: 1.2rem; }

/* metric cards */
.metric-card {
  border-radius: 16px;
  padding: 18px 20px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.metric-title { font-size: 0.95rem; opacity: 0.8; margin-bottom: 6px; }
.metric-value { font-size: 2.0rem; font-weight: 700; }
.small-note   { font-size: 0.8rem; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

st.title("💼 Portfolio Optimization & Performance — Showcase")

# -------------------- sidebar controls --------------------
ALL_SYMS = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
PROC = processed_column_path()
INTRA = intraday_file_path()

with st.sidebar:
    st.header("Controls")
    chosen = st.multiselect(
        "Assets",
        options=ALL_SYMS,
        default=[s for s in TICKERS if s in ALL_SYMS],
        help="Benchmark is used for comparison but excluded from portfolio weights."
    )
    if not chosen:
        st.stop()

    bench = st.selectbox("Benchmark", options=[DEFAULT_BENCH] + [s for s in chosen if not s.startswith("^")], index=0)

    range_choice = st.radio("Window", ["Today","1W","1M","6M","1Y","3Y","MAX","Custom"], index=5)
    custom = None
    if range_choice == "Custom":
        custom = st.date_input("Pick start/end", value=[])

    st.divider()
    weight_mode = st.radio("Weights", ["Equal (ex-bench)","YAML defaults","Custom","Max Sharpe (opt)","Min Variance (opt)"], index=0)
    rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=0.2, value=float(RF_ANN_PCT), step=0.005)
    show_assets_in_nav = st.toggle("Show individual assets in NAV chart", value=True)

# -------------------- load data --------------------
prices_all, rets_all = load_core(str(PROC), tuple(ALL_SYMS))
VIEW_SYMS = list(dict.fromkeys(chosen + [bench]))
prices = prices_all.loc[:, [c for c in VIEW_SYMS if c in prices_all.columns]]
rets   = rets_all.loc[:,   [c for c in VIEW_SYMS if c in rets_all.columns]]

# -------------------- date slicing --------------------
today = prices.index.max()
def cutoff_ts(choice: str):
    if choice == "Today": return today, True
    if choice == "1W":    return today - pd.Timedelta(days=7), False
    if choice == "1M":    return today - pd.Timedelta(days=30), False
    if choice == "6M":    return today - pd.Timedelta(days=182), False
    if choice == "1Y":    return today - pd.Timedelta(days=365), False
    if choice == "3Y":    return today - pd.Timedelta(days=365*3), False
    if choice == "MAX":   return prices.index.min(), False
    if custom and len(custom) == 2:
        return pd.Timestamp(custom[0]), False
    return prices.index.min(), False

cut, is_today = cutoff_ts(range_choice)
hist_px = prices.loc[prices.index >= cut]
hist_rt = rets.loc[rets.index >= cut]

# -------------------- build weights --------------------
if weight_mode == "Equal (excluding benchmark)":
    w = normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)
elif weight_mode == "Defaults":
    w = normalize_ex_bench({s: DEFAULT_WEIGHTS.get(s, 0.0) for s in chosen}, bench)
elif weight_mode == "Custom":
    st.sidebar.write("Custom weights (renormalized):")
    sliders = {}
    for c in [s for s in chosen if s != bench]:
        sliders[c] = st.sidebar.slider(c, 0.0, 1.0, float(DEFAULT_WEIGHTS.get(c, 0.0)), 0.01)
    w = normalize_ex_bench(sliders, bench)
else:
    # optimization choices (uses historical window you selected)
    cols = [c for c in hist_rt.columns if c in chosen and c != bench]
    R = hist_rt[cols]
    if weight_mode == "Max Sharpe (opt)":
        wopt = optimize_max_sharpe(R, rf_annual=rf) if len(cols) >= 2 else pd.Series(np.nan, index=cols)
    else:
        wopt = optimize_min_variance(R) if len(cols) >= 2 else pd.Series(np.nan, index=cols)
    wopt = wopt.dropna()
    w = wopt if wopt.sum() else normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)

# -------------------- portfolio & benchmark series --------------------
port = portfolio_returns(hist_rt.drop(columns=[bench], errors="ignore"), w)
bench_s = hist_rt.get(bench)

# -------------------- KPI row --------------------
def metric_card(title: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k5,k6,k7 = st.columns(3)

metric_card("Ann. Return",   f"{ann_return(port):.2%}");      k1.empty()
metric_card("Ann. Vol",      f"{ann_vol(port):.2%}");         k2.empty()
metric_card("Sharpe",        f"{sharpe(port, rf_annual=rf):.2f}"); k3.empty()
b,_ = beta_alpha(port, bench_s, rf_annual=rf)
metric_card("Beta",          f"{b:.2f}");                     k4.empty()
metric_card("Tracking Error",f"{tracking_error(port, bench_s):.2%}"); k5.empty()
metric_card("Info Ratio",    f"{information_ratio(port, bench_s):.2f}"); k6.empty()
dd = max_drawdown(port)
metric_card("Max Drawdown",  f"{dd.max_drawdown:.2%}");       k7.empty()

st.caption(f"Window: **{cut.date()} → {hist_px.index.max().date()}** · Benchmark: **{bench}**")

# -------------------- NAV & relative charts --------------------
st.subheader("Cumulative NAV")
nav_df = pd.DataFrame({
    "Portfolio": nav_from_returns(port),
    bench: nav_from_returns(bench_s) if bench_s is not None else pd.Series(dtype=float),
}).dropna()
if show_assets_in_nav:
    for s in [c for c in chosen if c in hist_rt.columns and c != bench]:
        nav_df[s] = nav_from_returns(hist_rt.loc[nav_df.index, s])
st.plotly_chart(px.line(nav_df, title="Cumulative NAV (selected window)"), use_container_width=True)

st.subheader("Relative Outperformance vs Benchmark")
rel = relative_return_series(port, bench_s)
st.plotly_chart(px.area(rel.rename("Relative vs Benchmark")), use_container_width=True)

# -------------------- Rolling analytics --------------------
st.subheader("Rolling analytics")
roll_win = st.slider("Rolling window (days)", 20, 120, 60, 5)
df_rb = pd.concat([port.rename("p"), bench_s.rename("b")], axis=1).dropna()
if len(df_rb) >= roll_win + 2:
    roll_vol = df_rb["p"].rolling(roll_win).std(ddof=1) * np.sqrt(252)
    cov_pb = df_rb["p"].rolling(roll_win).cov(df_rb["b"])
    var_b  = df_rb["b"].rolling(roll_win).var()
    beta_roll = (cov_pb / var_b).rename("Rolling Beta")
    st.plotly_chart(px.line(roll_vol.rename("Rolling Vol (ann)")), use_container_width=True)
    st.plotly_chart(px.line(beta_roll), use_container_width=True)
else:
    st.info("Increase the window length or select a longer date range.")

# -------------------- Risk & contributions + total risk --------------------
st.subheader("Risk & Contributions")
cols = [c for c in hist_rt.columns if c in chosen and c != bench]
R = hist_rt[cols]
left, right = st.columns([2,1])

with left:
    if not R.empty and len(cols) >= 2:
        Sigma = cov_matrix(R)
        w_now = w.reindex(cols).fillna(0.0)
        if w_now.sum(): w_now = w_now / w_now.sum()
        rc = risk_contributions(w_now, Sigma)  # sums to 1
        fig_rc = px.bar(rc.sort_values(ascending=False).rename("Contribution"),
                        title="Risk Contribution (weights × covariance)",
                        labels={"index": "Ticker", "value": "Contribution"})
        fig_rc.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("Select at least two assets (excluding benchmark) to view contributions.")

with right:
    pv, bv = ann_vol(port), ann_vol(bench_s)
    vol_series = pd.Series({"Portfolio": pv, bench: bv}, name="Ann. Vol")
    fig_vol = px.bar(vol_series, title="Total Risk (Ann. Vol)",
                     labels={"index":"Series","value":"Ann. Vol"})
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

# -------------------- Efficient frontier & VaR --------------------
st.subheader("Frontier & VaR")
cA, cB = st.columns([2,1])

with cA:
    try:
        ef = trace_efficient_frontier(R, n_points=30)
        if ef.empty:
            st.caption("Install `scipy` to enable the efficient frontier.")
        else:
            st.plotly_chart(px.scatter(ef, x="vol_ann", y="ret_ann",
                                       title="Efficient Frontier",
                                       labels={"vol_ann":"Vol (ann)","ret_ann":"Return (ann)"}),
                            use_container_width=True)
    except Exception as e:
        st.caption(f"Frontier unavailable: {e}")

with cB:
    if not R.empty:
        var1d, es1d = mc_var_es(R, w.reindex(R.columns).fillna(0.0), alpha=0.95, horizon_days=1, n_sims=10000, seed=42)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Monte Carlo (1-day, 95%)</div>
          <div class="metric-value">VaR: {var1d:.2%}</div>
          <div class="metric-title small-note">ES: {es1d:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Diversification --------------------
st.subheader("Diversification")
if METADATA:
    lab_sec = {k: METADATA.get(k,{}).get("sector","Unknown") for k in w.index}
    lab_reg = {k: METADATA.get(k,{}).get("region","Unknown") for k in w.index}
    sec_w = pd.Series(w).groupby(pd.Series(lab_sec)).sum().sort_values(ascending=False)
    reg_w = pd.Series(w).groupby(pd.Series(lab_reg)).sum().sort_values(ascending=False)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(sec_w, values=sec_w.values, names=sec_w.index, title="By Sector"), use_container_width=True)
    c2.plotly_chart(px.pie(reg_w, values=reg_w.values, names=reg_w.index, title="By Region"), use_container_width=True)
else:
    st.info("No metadata in YAML to aggregate diversification.")

# -------------------- Per-asset table & downloads --------------------
st.subheader("By Asset (same window & benchmark)")
def per_ticker_metrics(rt: pd.DataFrame, bench_s: pd.Series, rf_annual: float) -> pd.DataFrame:
    rows=[]
    for col in [c for c in rt.columns if c != bench]:
        s = rt[col].dropna()
        if s.empty: continue
        beta,_ = beta_alpha(s, bench_s, rf_annual=rf_annual)
        rows.append({
            "Ticker": col,
            "AnnRet": ann_return(s),
            "AnnVol": ann_vol(s),
            "Sharpe": sharpe(s, rf_annual=rf_annual),
            "Beta": beta,
            "TE": tracking_error(s, bench_s),
            "IR": information_ratio(s, bench_s),
            "MaxDD": max_drawdown(s).max_drawdown,
        })
    return (pd.DataFrame(rows).set_index("Ticker")).sort_index()

by_asset = per_ticker_metrics(hist_rt, bench_s, rf)
disp = by_asset.copy()
for c in ["AnnRet","AnnVol","TE","MaxDD"]:
    disp[c] = (disp[c] * 100).map("{:.2f}%".format)
disp["Sharpe"] = disp["Sharpe"].map("{:.2f}".format)
disp["Beta"]   = disp["Beta"].map("{:.2f}".format)
disp["IR"]     = disp["IR"].map("{:.2f}".format)
st.dataframe(disp, use_container_width=True)

# downloads
col_dl1, col_dl2, col_dl3 = st.columns(3)
col_dl1.download_button(
    "Download portfolio NAV (CSV)",
    nav_df.to_csv(index=True).encode("utf-8"),
    file_name="portfolio_nav.csv", mime="text/csv"
)
metrics_csv = pd.DataFrame({"metric": list(disp.columns), **{t: by_asset.loc[t].values for t in by_asset.index}})
col_dl2.download_button(
    "Download per-asset metrics (CSV)",
    by_asset.to_csv(index=True).encode("utf-8"),
    file_name="by_asset_metrics.csv", mime="text/csv"
)
col_dl3.download_button(
    "Download weights (CSV)",
    w.rename("weight").to_csv(index=True).encode("utf-8"),
    file_name="weights.csv", mime="text/csv"
)

# -------------------- Intraday “today” snapshot (optional) --------------------
st.subheader("Live (today) — optional")
if INTRA.exists():
    last, intra_close = load_intraday_cached(str(INTRA), tuple(ALL_SYMS))
    if not intra_close.empty:
        prev_map = {c: float(prices_all.iloc[-1][c]) for c in prices_all.columns}
        cols = [c for c in VIEW_SYMS if c in intra_close.columns]
        if cols:
            rets_min = intraday_return_matrix(intra_close.loc[:, cols], prev_map)
            if not rets_min.empty:
                port_today = (rets_min.drop(columns=[bench], errors="ignore").dot(w)).rename("Portfolio (today)")
                chart_df = pd.concat([port_today, rets_min.get(bench)], axis=1).dropna()
                st.plotly_chart(px.line(chart_df, title="Intraday Return (since prior close)"), use_container_width=True)
                # tiles
                if len(chart_df):
                    p_now = chart_df["Portfolio (today)"].iloc[-1]
                    b_now = chart_df[bench].iloc[-1] if bench in chart_df.columns else np.nan
                    t1, t2, t3 = st.columns(3)
                    for label, val, col in [
                        ("Today (Portfolio)", p_now, t1),
                        (f"Today ({bench})", b_now, t2),
                        ("Today Relative", p_now - b_now if pd.notna(b_now) else np.nan, t3),
                    ]:
                        col.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-title">{label}</div>
                          <div class="metric-value">{(val if pd.notna(val) else np.nan):.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                # overlay “today” price
                ov = overlay_prices(prices_all, last, prev_map)
                st.plotly_chart(
                    px.line(ov.loc[ov.index >= ov.index.max() - pd.Timedelta(days=7)], 
                            title="Overlay Adj Close (last 7 days incl. synthetic today)"),
                    use_container_width=True
                )
    else:
        st.info("Intraday cache exists but is empty for selected symbols.")
else:
    st.caption("No intraday file yet — run `python -m src.live daemon` during market hours.")
