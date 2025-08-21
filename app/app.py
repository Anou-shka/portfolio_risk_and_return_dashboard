# app/app_2.py — Beautified layout
from __future__ import annotations

import os, streamlit as st

ACCESS_CODE = os.getenv("APP_ACCESS_CODE", "")

def require_access_code():
    if not ACCESS_CODE:
        return
    if not st.session_state.get("auth_ok"):
        st.set_page_config(page_title="Portfolio Pulse", layout="wide", initial_sidebar_state="expanded")
        st.title("🔒 Portfolio Pulse")
        pwd = st.text_input("Enter access code", type="password")
        if st.button("Enter"):
            if pwd == ACCESS_CODE:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Incorrect code")
        st.stop()

require_access_code()


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
import plotly.io as pio
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

# -------------------- THEME & CSS --------------------
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"
px.defaults.width = None
px.defaults.height = 420

DEFAULT_CSS = r"""
/* ---------- Layout ---------- */
:root{
  --bg: #0d1224;          /* page background */
  --card: #121a33;        /* card background */
  --muted: #93a0b6;       /* secondary text */
  --text: #e6e9f2;        /* primary text */
  --accent: #4cc9f0;      /* accents */
  --good:#22c55e;         /* positive */
  --bad:#ef4444;          /* negative */
  --warn:#f59e0b;         /* warning */
}

html, body, [data-testid="stAppViewContainer"]{ background: radial-gradient(1200px 600px at 10% -10%, #1a2456, transparent),
                                                  radial-gradient(800px 400px at 90% 10%, #1b3d5e, transparent), var(--bg) !important; }

/* tighten & widen content */
.main .block-container{ max-width: 1400px; padding-top: 0.5rem; }

/* page title */
.pp-title{ text-align:center; font-size: 2.1rem; font-weight: 700; letter-spacing: .5px; color: var(--text); margin: .4rem 0 1rem; }
.pp-sub{ text-align:center; color: var(--muted); margin-top:-.4rem; }

/* pills */
.pp-pills{ display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin: .3rem 0 1rem; }
.pp-pill{ background: linear-gradient(145deg, #182142, #0e1733); color: var(--text); border: 1px solid rgba(255,255,255,.06); padding:.25rem .6rem; border-radius: 999px; font-size:.85rem; }

/* group card */
.card{ background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
       border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: .8rem 1rem; box-shadow: 0 6px 18px rgba(0,0,0,.25); }

/* metric tiles */
.metric-row{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.75rem; }
.metric-card{ background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.01)); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:.75rem 1rem; }
.metric-title{ color: var(--muted); font-size:.8rem; margin-bottom:.2rem; }
.metric-value{ font-weight:700; font-size:1.35rem; color: var(--text); }
.small-note{ color: var(--muted); font-size:.9rem; margin-top:.25rem; }

/* tables */
[data-testid="stDataFrame"] div{ color: var(--text) !important; }
[data-testid="stDataFrame"] thead th{ background: rgba(255,255,255,.03) !important; }

/* sidebar */
section[data-testid="stSidebar"]{ background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(0,0,0,.15)); border-right:1px solid rgba(255,255,255,.08); }

/* buttons */
.stDownloadButton, .stButton>button{ border-radius: 12px; border: 1px solid rgba(255,255,255,.12); background: linear-gradient(180deg, #1d2a52, #0f1734); }

/* plots */
.js-plotly-plot .plotly .modebar{ background: transparent !important; }
"""


def inject_css():
    # If you prefer external CSS, drop a file at app/assets/theme.css
    css_external = Path(ROOT) / "app" / "assets" / "theme.css"
    if css_external.exists():
        st.markdown(f"<style>{css_external.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)


def nice_fig(fig, title: str | None = None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        margin=dict(t=60, r=20, b=40, l=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------- loaders ----------------

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
    d = {k: float(v) for k, v in d.items()}
    d.pop(bench, None)
    s = pd.Series(d, dtype=float)
    return s / s.sum() if s.sum() else s


# -------------------- page setup & css --------------------
st.set_page_config(page_title="Portfolio Pulse", layout="wide", initial_sidebar_state="expanded")
inject_css()

# --- Make the top black header blend with the page & add an app title bar ---
st.markdown("""
<style>
/* Streamlit header strip -> transparent so the page gradient shows */
header[data-testid="stHeader"]{background:transparent !important;}
header[data-testid="stHeader"] > *{background:transparent !important; box-shadow:none !important;}
/* Optional: hide the header buttons completely (uncomment next line) */
/* div[data-testid="stToolbar"]{display:none !important;} */

/* Sticky top bar we control */
.pp-topbar{position:sticky; top:0; z-index:1000; width:100%;
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border-bottom:1px solid rgba(255,255,255,.08); backdrop-filter: blur(8px);
  padding:.6rem 1rem .7rem; display:flex; flex-direction:column; align-items:center;}
.pp-topbar__title{font-size:1.1rem; font-weight:700; letter-spacing:.3px; color:var(--text);} 
.pp-topbar__sub{font-size:.85rem; color:var(--muted); margin-top:.1rem;}
@media (max-width:680px){ .pp-topbar__sub{display:none;} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-title">Portfolio Pulse</div>
<div class="pp-sub">Clean visuals • Quick insights • Actionable risk</div>
""", unsafe_allow_html=True)

# -------------------- sidebar controls --------------------
ALL_SYMS = list(dict.fromkeys(TICKERS + [DEFAULT_BENCH]))
PROC = processed_column_path()
INTRA = intraday_file_path()

with st.sidebar:
    st.header("Filters")
    chosen = st.multiselect(
        "Assets",
        options=ALL_SYMS,
        default=[s for s in TICKERS if s in ALL_SYMS],
        help="Choose the assets you want to analyze.",
    )
    if not chosen:
        st.stop()

    bench = st.selectbox(
        "Benchmark",
        options=[DEFAULT_BENCH],
        help="Benchmark used for comparison.",
        index=0,
    )

    range_choice = st.radio("Time Period", ["Today", "1W", "1M", "6M", "1Y", "3Y", "Custom"], index=5, help="Select the time period for analysis.", horizontal=True)
    custom = None
    if range_choice == "Custom":
        custom = st.date_input("Pick start/end", value=[])

    st.divider()
    weight_mode = st.radio("Weights", ["Equal-weighted (excluding benchmark)", "Custom Weights", "Max Sharpe", "Min Variance"], index=0, help="How to assign weights to assets.")
    if weight_mode == "Custom Weights":
        st.caption("Custom weights (renormalized):")
        sliders = {}
        for c in [s for s in chosen if s != bench]:
            sliders[c] = st.slider(c, 0.0, 1.0, float(DEFAULT_WEIGHTS.get(c, 0.0)), 0.01)
    rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=0.2, value=float(RF_ANN_PCT), step=0.005, help="Annual risk-free rate.")
    st.caption("Intraday view auto-refreshes ~every 60s.")
    st.divider()

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
    if custom and len(custom) == 2:
        return pd.Timestamp(custom[0]), False
    return prices.index.min(), False

cut, is_today = cutoff_ts(range_choice)
hist_px = prices.loc[prices.index >= cut]
hist_rt = rets.loc[rets.index >= cut]

# -------------------- build weights --------------------
if weight_mode == "Equal-weighted (excluding benchmark)":
    w = normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)
elif weight_mode == "Custom Weights":
    # `sliders` was created in the sidebar above when this mode is selected
    w = normalize_ex_bench(sliders, bench)
else:
    # optimization choices (uses historical window you selected)
    cols = [c for c in hist_rt.columns if c in chosen and c != bench]
    R = hist_rt[cols]
    if weight_mode == "Max Sharpe":
        wopt = optimize_max_sharpe(R, rf_annual=rf) if len(cols) >= 2 else pd.Series(np.nan, index=cols)
    else:
        wopt = optimize_min_variance(R) if len(cols) >= 2 else pd.Series(np.nan, index=cols)
    wopt = wopt.dropna()
    w = wopt if wopt.sum() else normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)

# -------------------- Summary pills --------------------
asset_ct = len([s for s in chosen if s != bench])
st.markdown(
    f"""
    <div class='pp-pills'>
      <span class='pp-pill'>Assets: <b>{asset_ct}</b></span>
      <span class='pp-pill'>Benchmark: <b>{bench}</b></span>
      <span class='pp-pill'>Weights: <b>{weight_mode.replace(' (excluding benchmark)','')}</b></span>
      <span class='pp-pill'>Window: <b>{'Today' if is_today else cut.date()} → {today.date()}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- portfolio & benchmark series --------------------
port = portfolio_returns(hist_rt.drop(columns=[bench], errors="ignore"), w)
bench_s = hist_rt.get(bench)

# =======================
# TABS
# =======================
tab_perf, tab_live, tab_vol, tab_risk, tab_div = st.tabs(["Performance", "Live", "Volatility", "Risk & Frontier", "Diversification"])

# -------------------- NAV & relative charts --------------------
with tab_perf:
    if range_choice == "Today":
        st.info("Switch to 1W/1M/… to see historical metrics.")
    else:
        rets_view = hist_rt.copy()
        port    = portfolio_returns(rets_view.drop(columns=[bench], errors="ignore"), w)
        bench_s = rets_view.get(bench)

        # --- KPI row(s)
        c1, c2, c3, c4 = st.columns(4)
        # c1.metric("Ann. Return", f"{ann_return(port):.2%}")
        c1.metric(
            "Ann. Return",
            f"{ann_return(port):.2%}",
            help="The portfolio’s compounded return per year over the selected window. Higher is better."
        )
        c2.metric("Ann. Vol",    f"{ann_vol(port):.2%}", help="Variability of returns (standard deviation), annualized. A proxy for risk. Lower is calmer.")
        c3.metric("Sharpe",      f"{sharpe(port, rf_annual=rf):.2f}", help="Risk-adjusted return = (Ann. Return − Risk-free) / Ann. Vol. Higher is better.")
        beta, alpha = beta_alpha(port, bench_s, rf_annual=rf)
        c4.metric("Beta vs Bench", f"{beta:.2f}", help="Sensitivity to the benchmark. 1 ≈ moves with it; >1 amplifies; <1 is more defensive.")

        c5, c6, c7 = st.columns(3)
        c5.metric("Tracking Error", f"{tracking_error(port, bench_s):.2%}", help="Volatility of active returns (Portfolio − Benchmark). Higher = more active risk.")
        c6.metric("Info Ratio",     f"{information_ratio(port, bench_s):.2f}", help="Average active return divided by tracking error. >0.5 is decent, >1 is strong.")
        dd_stats = max_drawdown(port)
        c7.metric("Max Drawdown",    f"{dd_stats.max_drawdown:.2%}", help="Largest peak-to-trough loss within the selected period.")

        # --- Cumulative NAV (portfolio, benchmark, optional assets)
        if "nav_assets_toggle" not in st.session_state:
            st.session_state.nav_assets_toggle = False
        show_assets_in_nav = st.session_state.nav_assets_toggle

        series = {"Portfolio": nav_from_returns(port)}
        if bench_s is not None and not bench_s.dropna().empty:
            series[bench] = nav_from_returns(bench_s)

        if show_assets_in_nav:
            asset_cols = [s for s in chosen if s in rets_view.columns and s != bench]
            for t in asset_cols:
                series[t] = nav_from_returns(rets_view.loc[series["Portfolio"].index, t])

        nav_df = pd.DataFrame(series).dropna(how="any")

        st.plotly_chart(
            px.line(nav_df, title="Cumulative Net Asset Value"),
            use_container_width=True,
            help="Cumulative NAV of the portfolio and benchmark, with optional individual assets."
        )

        st.toggle("Show individual assets in NAV chart", key="nav_assets_toggle", help="Include each selected asset as its own line in the NAV chart.")
        
        # --- Relative outperformance vs benchmark
        if bench_s is not None and not bench_s.dropna().empty:
            rel = relative_return_series(port, bench_s)
            st.plotly_chart(
            px.area(rel.rename(f"Relative vs {bench}"), title=f"Relative Outperformance vs {bench}"),
            use_container_width=True,
            help="Cumulative outperformance = Portfolio return − Benchmark. Above 0 means you’re ahead; rising area = gaining ground."
        )

# -------------------- rolling risk volatility charts --------------------
with tab_vol:
    if range_choice == "Today":
        st.info("Switch to 1W/1M/… to see historical metrics.")
    else:
        df_rb = pd.concat([port.rename("p"), bench_s.rename("b")], axis=1).dropna()
        if df_rb.empty:
            st.info("No overlapping data for portfolio & benchmark in this range.")
        else:
            max_win = int(min(252, len(df_rb)))
            if max_win < 10:
                st.info("Not enough data for rolling stats; choose a longer window.")
            else:
                win = st.slider("Rolling window (trading days)", min_value=10, max_value=max_win, value=min(60, max_win), step=5, help="Window length used for rolling volatility and beta.")
                if len(df_rb) < win:
                    st.info(f"Not enough data for a {win}-day window; reduce the window or extend the date range.")
                else:
                    roll_vol = df_rb["p"].rolling(win).std(ddof=1) * np.sqrt(252)
                    cov_pb   = df_rb["p"].rolling(win).cov(df_rb["b"])
                    var_b    = df_rb["b"].rolling(win).var()
                    beta_roll = (cov_pb / var_b).replace([np.inf, -np.inf], np.nan)

                    last_idx = df_rb.index[-win:]
                    roll_vol_view  = roll_vol.loc[last_idx].dropna()
                    beta_roll_view = beta_roll.loc[last_idx].dropna()

                    c1, c2 = st.columns(2)
                    with c1:
                        fig_vol = px.line(roll_vol_view.rename(f"Rolling {win}D Ann. Vol"))
                        nice_fig(fig_vol, title=f"Rolling {win}D Annualized Volatility (last {len(last_idx)} days)")
                        st.plotly_chart(fig_vol, use_container_width=True)
                    with c2:
                        fig_beta = px.line(beta_roll_view.rename(f"Rolling {win}D Beta"))
                        nice_fig(fig_beta, title=f"Rolling {win}D Beta vs {bench} (last {len(last_idx)} days)")
                        st.plotly_chart(fig_beta, use_container_width=True)


# -------------------- Risk & Frontier charts --------------------
with tab_risk:
    if range_choice == "Today":
        st.info("Switch to 1W/1M/… to analyze risk with daily history.")
    else:
        st.subheader("Risk & Contributions")

        cols_sel = [c for c in hist_rt.columns if c in chosen and c != bench]
        R = hist_rt[cols_sel].dropna(how="all")
        if R.empty or len(cols_sel) < 1:
            st.info("Select at least one asset (excluding the benchmark).")
            st.stop()

        shrink = st.slider("Covariance shrinkage λ", 0.0, 0.9, 0.0, 0.05, help="λ=0 uses sample cov; λ>0 shrinks toward diagonal.")

        left, right = st.columns([2, 1])
        with left:
            if len(cols_sel) >= 2:
                Sigma = cov_matrix(R, shrink=shrink)
                w_now = w.reindex(cols_sel).fillna(0.0)
                if w_now.sum():
                    w_now = w_now / w_now.sum()
                rc = risk_contributions(w_now, Sigma)  # sums to 1
                fig_rc = px.bar(rc.sort_values(ascending=False).rename("Contribution"), labels={"index": "Ticker", "value": "Contribution"})
                fig_rc.update_yaxes(tickformat=".0%")
                nice_fig(fig_rc, title="Risk Contribution (weights × covariance)")
                st.plotly_chart(fig_rc, use_container_width=True)
            else:
                st.info("Select at least two assets to view risk contributions.")

        with right:
            pv, bv = ann_vol(port), ann_vol(bench_s)
            vol_series = pd.Series({"Portfolio": pv, bench: bv}, name="Ann. Vol")
            fig_volb = px.bar(vol_series, labels={"index": "Series", "value": "Ann. Vol"})
            fig_volb.update_yaxes(tickformat=".0%")
            nice_fig(fig_volb, title="Total Risk (Ann. Vol)")
            st.plotly_chart(fig_volb, use_container_width=True)

        # -------------------- Efficient frontier & VaR --------------------
        st.subheader("Frontier & VaR")

        ctl1, ctl2, ctl3, ctl4 = st.columns([1.2, 1, 1, 1.2])
        with ctl1:
            n_points = st.slider(
                "Frontier points", 10, 60, 30, 5,
                help="How many portfolios to sample along the efficient frontier (more = smoother curve)."
            )
        with ctl2:
            bounds_min, bounds_max = st.slider(
                "Weight bounds", 0.0, 1.0, (0.0, 1.0), 0.05,
                help="Per-asset min/max weights used by the optimizer. Tight bounds reduce concentration."
            )
        with ctl3:
            alpha = st.select_slider(
                "VaR confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
                help="Confidence level for Value-at-Risk. 0.95 means a 5% worst-case tail."
            )
        with ctl4:
            horizon = st.slider(
                "VaR horizon (days)", 1, 10, 1,
                help="Loss horizon for VaR/ES simulation, in trading days."
            )


        cA, cB = st.columns([3, 1])

        with cA:
            try:
                if len(cols_sel) >= 2:
                    ef = trace_efficient_frontier(R, n_points=n_points, bounds=(bounds_min, bounds_max))
                else:
                    ef = pd.DataFrame()

                if ef.empty:
                    st.caption("Efficient frontier unavailable (need ≥ 2 assets and `scipy`).")
                else:
                    # Current portfolio / reference points
                    ret_p, vol_p = ann_return(port), ann_vol(port)

                    # Equal-weight reference (ex-bench)
                    ew = pd.Series(1.0/len(cols_sel), index=cols_sel)
                    ret_ew = ann_return(R.dot(ew))
                    vol_ew = ann_vol(R.dot(ew))

                    # Optional optimized references
                    try:
                        w_ms = optimize_max_sharpe(R, rf_annual=rf, bounds=(bounds_min, bounds_max))
                        w_mv = optimize_min_variance(R, bounds=(bounds_min, bounds_max))
                    except Exception:
                        w_ms = pd.Series(dtype=float)
                        w_mv = pd.Series(dtype=float)

                    pts = pd.DataFrame({
                        "name": ["Portfolio", "Equal-Weight"],
                        "vol_ann": [vol_p, vol_ew],
                        "ret_ann": [ret_p, ret_ew],
                    })

                    if not w_ms.dropna().empty:
                        pts = pd.concat([pts, pd.DataFrame({
                            "name": ["Max Sharpe"],
                            "vol_ann": [ann_vol(R.dot(w_ms))],
                            "ret_ann": [ann_return(R.dot(w_ms))],
                        })], ignore_index=True)

                    if not w_mv.dropna().empty:
                        pts = pd.concat([pts, pd.DataFrame({
                            "name": ["Min Variance"],
                            "vol_ann": [ann_vol(R.dot(w_mv))],
                            "ret_ann": [ann_return(R.dot(w_mv))],
                        })], ignore_index=True)

                    fig = px.scatter(ef, x="vol_ann", y="ret_ann", labels={"vol_ann": "Volatility", "ret_ann": "Return"})
                    fig.update_traces(mode="lines+markers")
                    fig.add_scatter(x=pts["vol_ann"], y=pts["ret_ann"], mode="markers+text", text=pts["name"], textposition="top center", marker=dict(size=10), name="Reference")
                    nice_fig(fig, title="Efficient Frontier (annualized)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Frontier unavailable: {e}")

        with cB:
            if not R.empty:
                var1d, es1d = mc_var_es(R, w.reindex(R.columns).fillna(0.0), alpha=float(alpha), horizon_days=int(horizon), n_sims=10_000, seed=42)
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-title">Monte Carlo (h={int(horizon)}d, {int(alpha*100)}%)</div>
                  <div class="metric-value">VaR: {var1d:.2%}</div>
                  <div class="metric-title small-note">ES: {es1d:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Not enough data for VaR.")

# -------------------- Diversification Charts ---------------
with tab_div:
    st.subheader("Diversification")

    # weights as Series (exclude benchmark for diversification)
    w_s = pd.Series(w, dtype=float).dropna()
    w_s = w_s[w_s.index != bench]

    if METADATA and not w_s.empty:
        meta = pd.DataFrame(index=w_s.index)
        meta["sector"] = [METADATA.get(t, {}).get("sector", "Unknown") or "Unknown" for t in w_s.index]
        meta["region"] = [METADATA.get(t, {}).get("region", "Unknown") or "Unknown" for t in w_s.index]

        sec_w = w_s.groupby(meta["sector"]).sum().sort_values(ascending=False)
        reg_w = w_s.groupby(meta["region"]).sum().sort_values(ascending=False)

        c1, c2 = st.columns(2)
        fig_sec = px.pie(values=sec_w.values, names=sec_w.index, hole=0.5)
        fig_sec.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:.2%}<extra></extra>")
        nice_fig(fig_sec, title="By Sector")
        c1.plotly_chart(fig_sec, use_container_width=True)

        fig_reg = px.pie(values=reg_w.values, names=reg_w.index, hole=0.5)
        fig_reg.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:.2%}<extra></extra>")
        nice_fig(fig_reg, title="By Region")
        c2.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("No metadata available or no non-benchmark assets to aggregate.")

    # -------------------- Per-asset table & downloads --------------------
    st.subheader("By Asset")

    def auto_height(df, row_px=36, header_px=38, pad_px=16, max_px=800):
        n = max(1, len(df))
        return min(max_px, header_px + pad_px + n * row_px)

    def per_ticker_metrics(rt: pd.DataFrame, bench_s: pd.Series, rf_annual: float) -> pd.DataFrame:
        rows = []
        for col in [c for c in rt.columns if c != bench]:
            s = rt[col].dropna()
            if s.empty:
                continue
            b, _ = beta_alpha(s, bench_s, rf_annual=rf_annual)
            rows.append({
                "Ticker": col,
                "AnnRet": ann_return(s),
                "AnnVol": ann_vol(s),
                "Sharpe": sharpe(s, rf_annual=rf_annual),
                "Beta": b,
                "TE": tracking_error(s, bench_s),
                "IR": information_ratio(s, bench_s),
                "MaxDD": max_drawdown(s).max_drawdown,
            })
        if not rows:
            return pd.DataFrame(columns=["AnnRet", "AnnVol", "Sharpe", "Beta", "TE", "IR", "MaxDD"])
        return pd.DataFrame(rows).set_index("Ticker").sort_index()

    by_asset = per_ticker_metrics(hist_rt, bench_s, rf)
    order = [c for c in chosen if c in by_asset.index]
    by_asset = by_asset.reindex(order).dropna(how="all")

    disp = by_asset.copy()
    for c in ["AnnRet", "AnnVol", "TE", "MaxDD"]:
        disp[c] = (disp[c] * 100).map("{:.2f}%".format)
    disp["Sharpe"] = disp["Sharpe"].map("{:.2f}".format)
    disp["Beta"]   = disp["Beta"].map("{:.2f}".format)
    disp["IR"]     = disp["IR"].map("{:.2f}".format)

    h = auto_height(disp)
    st.dataframe(disp, use_container_width=True, height=h)

    # --- build nav_df locally (scope-safe)
    nav_df = pd.DataFrame({
        "Portfolio": nav_from_returns(port),
        bench: nav_from_returns(bench_s) if bench_s is not None else pd.Series(dtype=float)
    }).dropna()

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    col_dl1.download_button("Download portfolio NAV (CSV)", nav_df.to_csv(index=True).encode("utf-8"), file_name="portfolio_nav.csv", mime="text/csv")
    col_dl2.download_button("Download per-asset metrics (CSV)", by_asset.to_csv(index=True).encode("utf-8"), file_name="by_asset_metrics.csv", mime="text/csv")
    col_dl3.download_button("Download weights (CSV)", w.rename("weight").to_csv(index=True).encode("utf-8"), file_name="weights.csv", mime="text/csv")

# ------------- Live Intraday Charts ---------------------
with tab_live:
    st.subheader("Live (today)")
    win_opt = st.select_slider("Show last…", options=["30m", "1h", "2h", "3h", "6h", "Full day"], value="3h", help="Slice the intraday cache to the recent window you care about.")
    _mins = {"30m": 30, "1h": 60, "2h": 120, "3h": 180, "6h": 360}

    if (INTRA := intraday_file_path()).exists():
        last, intra_close = load_intraday_cached(str(INTRA), tuple(ALL_SYMS))
        if not intra_close.empty:
            if win_opt != "Full day":
                start_ts = intra_close.index.max() - pd.Timedelta(minutes=_mins[win_opt])
                intra_close = intra_close.loc[intra_close.index >= start_ts]

            prev_map = {c: float(prices_all.iloc[-1][c]) for c in prices_all.columns}
            cols = [s for s in VIEW_SYMS if s in intra_close.columns]
            if cols:
                rets_min = intraday_return_matrix(intra_close.loc[:, cols], prev_map)
                if not rets_min.empty:
                    w_cols = w.reindex([c for c in cols if c != bench]).fillna(0.0)
                    if w_cols.sum():
                        w_cols = w_cols / w_cols.sum()
                    port_today  = rets_min[w_cols.index].dot(w_cols).rename("Portfolio (today)")
                    bench_today = rets_min.get(bench)
                    chart_df = pd.concat([port_today, bench_today], axis=1).dropna(how="all")
                    npts = len(chart_df)
                    fig_live = px.line(chart_df)
                    nice_fig(fig_live, title=f"Intraday Return (since prior close) — {npts:,} points")
                    st.plotly_chart(fig_live, use_container_width=True)

                    if npts:
                        p_now = float(port_today.dropna().iloc[-1]) if len(port_today.dropna()) else float("nan")
                        b_now = float(bench_today.dropna().iloc[-1]) if bench_today is not None and len(bench_today.dropna()) else float("nan")
                        t1, t2, t3 = st.columns(3)
                        tooltips_today = {
                            "Today (Portfolio)": "Intraday return since the previous official close for your weighted portfolio.",
                            f"Today ({bench})": "Intraday return since the previous official close for the benchmark.",
                            "Today Relative": "Portfolio intraday return minus benchmark intraday return (your edge today).",
                        }

                        for label, val, col in [
                            ("Today (Portfolio)", p_now, t1),
                            (f"Today ({bench})", b_now, t2),
                            ("Today Relative", p_now - b_now if np.isfinite(b_now) else np.nan, t3),
                        ]:
                            tip = tooltips_today.get(label, "")
                            col.markdown(f"""
                            <div class="metric-card" title="{tip}">
                            <div class="metric-title">{label}</div>
                            <div class="metric-value">{(val if pd.notna(val) else np.nan):.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)


                    show_overlay = st.toggle("Show overlay: last 7 days Adj Close (incl. synthetic today)", value=False)
                    if show_overlay and not last.empty:
                        ov = overlay_prices(prices_all, last, prev_map)
                        fig_ov = px.line(ov.loc[ov.index >= ov.index.max() - pd.Timedelta(days=7)])
                        nice_fig(fig_ov, title="Overlay Adj Close — last 7 days")
                        st.plotly_chart(fig_ov, use_container_width=True)

            mtime_ny = pd.Timestamp(INTRA.stat().st_mtime, unit="s", tz="UTC").tz_convert("America/New_York")
            st.caption(f"Intraday cache updated {mtime_ny:%Y-%m-%d %H:%M ET} • rows loaded: {len(intra_close):,}")
        else:
            st.info("Intraday cache exists but is empty for selected symbols.")
    else:
        st.caption("No intraday file yet — run `python -m src.live daemon` during market hours.")
