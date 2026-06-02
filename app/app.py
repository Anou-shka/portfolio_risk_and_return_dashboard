# app/app.py
from __future__ import annotations

import logging
import os
import re
import sys
import datetime as dt
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

# ── path shim so "import src" works regardless of cwd ─────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DEFAULT_BENCH, RF_ANN_PCT
from src.metrics_core import (
    ann_return, ann_vol, beta_alpha, calmar, cov_matrix, hist_var_cvar,
    information_ratio, max_drawdown, mc_var_es, nav_from_returns,
    optimize_max_sharpe, optimize_min_variance, portfolio_returns,
    price_to_returns, relative_return_series, risk_contributions, sharpe,
    sortino, tracking_error, trace_efficient_frontier, treynor,
    validate_all_metrics, TRADING_DAYS,
)
from src.metrics_active import (
    intraday_close_matrix, intraday_return_matrix, latest_intraday_last,
    overlay_prices,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── paths ─────────────────────────────────────────────────────────────────
INTRADIR = Path(ROOT) / "data" / "raw" / "intraday"

# ── plotly theme ──────────────────────────────────────────────────────────
pio.templates.default = "plotly_dark"
px.defaults.template  = "plotly_dark"
px.defaults.height    = 420

# ══════════════════════════════════════════════════════════════════════════
#  TICKER CATALOG  (80+ options shown in sidebar dropdown)
# ══════════════════════════════════════════════════════════════════════════
TICKER_CATALOG: List[str] = [
    # ── Mega Cap Tech ──────────────────────────────────────────────────
    "Apple (AAPL)", "Microsoft (MSFT)", "NVIDIA (NVDA)", "Amazon (AMZN)",
    "Alphabet (GOOGL)", "Meta (META)", "Tesla (TSLA)", "Netflix (NFLX)",
    "Adobe (ADBE)", "Salesforce (CRM)", "Broadcom (AVGO)", "Intel (INTC)",
    "AMD (AMD)", "Qualcomm (QCOM)", "Texas Instruments (TXN)",
    "CrowdStrike (CRWD)", "Palo Alto Networks (PANW)", "Datadog (DDOG)",
    "Snowflake (SNOW)", "Palantir (PLTR)", "Shopify (SHOP)",
    "Airbnb (ABNB)", "Uber (UBER)", "Zoom (ZM)",
    # ── Financials ─────────────────────────────────────────────────────
    "Berkshire Hathaway (BRK-B)", "JP Morgan (JPM)", "Bank of America (BAC)",
    "Wells Fargo (WFC)", "Goldman Sachs (GS)", "Morgan Stanley (MS)",
    "Visa (V)", "Mastercard (MA)", "PayPal (PYPL)", "Coinbase (COIN)",
    "Charles Schwab (SCHW)", "BlackRock (BLK)",
    # ── Healthcare ─────────────────────────────────────────────────────
    "Johnson & Johnson (JNJ)", "UnitedHealth (UNH)", "Pfizer (PFE)",
    "Eli Lilly (LLY)", "Moderna (MRNA)", "Abbott Labs (ABT)",
    "Merck (MRK)", "Bristol-Myers Squibb (BMY)",
    # ── Consumer ───────────────────────────────────────────────────────
    "Walt Disney (DIS)", "Coca-Cola (KO)", "McDonald's (MCD)",
    "Nike (NKE)", "Walmart (WMT)", "Home Depot (HD)",
    "Procter & Gamble (PG)", "Starbucks (SBUX)", "Target (TGT)",
    "Costco (COST)", "Ford (F)", "General Motors (GM)",
    # ── Energy & Materials ─────────────────────────────────────────────
    "ExxonMobil (XOM)", "Chevron (CVX)", "ConocoPhillips (COP)",
    "Occidental Petroleum (OXY)", "Freeport-McMoRan (FCX)",
    # ── Industrials & Telecom ──────────────────────────────────────────
    "Boeing (BA)", "Caterpillar (CAT)", "3M (MMM)",
    "AT&T (T)", "Verizon (VZ)",
    # ── Broad Market ETFs ──────────────────────────────────────────────
    "S&P 500 ETF (SPY)", "S&P 500 Vanguard (VOO)", "Nasdaq ETF (QQQ)",
    "Total Market ETF (VTI)", "International ETF (VXUS)",
    "Russell 2000 ETF (IWM)", "Mid-Cap ETF (VO)", "Dividend ETF (VYM)",
    "ARK Innovation ETF (ARKK)",
    # ── Sector ETFs ────────────────────────────────────────────────────
    "Energy ETF (XLE)", "Financial ETF (XLF)", "Tech ETF (XLK)",
    "Healthcare ETF (XLV)", "Consumer Disc ETF (XLY)",
    "Utilities ETF (XLU)", "Materials ETF (XLB)",
    # ── Alternative & Bond ETFs ────────────────────────────────────────
    "Real Estate ETF (VNQ)", "Emerging Markets ETF (EEM)",
    "Gold (GLD)", "Silver (SLV)", "Oil ETF (USO)",
    "Treasury Bond ETF (TLT)", "Short-Term Treasury (SHY)",
    "Aggregate Bond ETF (AGG)", "High Yield ETF (HYG)",
    # ── Crypto ─────────────────────────────────────────────────────────
    "Bitcoin (BTC-USD)", "Ethereum (ETH-USD)", "Solana (SOL-USD)",
    "Cardano (ADA-USD)", "Ripple (XRP-USD)", "Dogecoin (DOGE-USD)",
    "Binance Coin (BNB-USD)", "Polygon (MATIC-USD)", "Chainlink (LINK-USD)",
    "Avalanche (AVAX-USD)", "Litecoin (LTC-USD)", "Polkadot (DOT-USD)",
    "Uniswap (UNI-USD)", "Cosmos (ATOM-USD)", "Shiba Inu (SHIB-USD)",
    "Near Protocol (NEAR-USD)", "Stellar (XLM-USD)", "VeChain (VET-USD)",
]

_CATALOG_PLACEHOLDER = "— choose from catalog —"


def extract_ticker(catalog_entry: str) -> str:
    """Extract 'AAPL' from 'Apple (AAPL)' format."""
    m = re.search(r"\(([^)]+)\)$", catalog_entry.strip())
    return m.group(1).upper() if m else catalog_entry.strip().upper()


# ══════════════════════════════════════════════════════════════════════════
#  CSS / THEME
# ══════════════════════════════════════════════════════════════════════════
_THEME_CSS = r"""
:root {
  --bg:      #0d1224;
  --card:    #121a33;
  --muted:   #93a0b6;
  --text:    #e6e9f2;
  --accent:  #f5b731;   /* gold */
  --good:    #22c55e;
  --bad:     #ef4444;
  --warn:    #f59e0b;
  --border:  rgba(255,255,255,0.08);
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, #1a2456, transparent),
              radial-gradient(800px 400px at 90% 10%, #1b3d5e, transparent),
              var(--bg) !important;
}

.main .block-container { max-width: 1400px; padding-top: 0.5rem; }

.pp-title { text-align:center; font-size:2.1rem; font-weight:700;
            letter-spacing:.5px; color:var(--text); margin:.4rem 0 .3rem; }
.pp-sub   { text-align:center; color:var(--muted); margin-top:-.4rem; font-size:.95rem; }
.pp-date  { text-align:center; color:var(--accent); font-size:.85rem;
            font-weight:600; margin-bottom:.8rem; }

.pp-pills { display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin:.3rem 0 1rem; }
.pp-pill  { background:linear-gradient(145deg,#182142,#0e1733); color:var(--text);
            border:1px solid var(--border); padding:.25rem .6rem;
            border-radius:999px; font-size:.85rem; }

.metric-card  { background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
                border:1px solid var(--border); border-radius:16px;
                padding:.75rem 1rem; height:100%; }
.metric-title { color:var(--muted); font-size:.8rem; margin-bottom:.2rem; }
.metric-value { font-weight:700; font-size:1.35rem; color:var(--text); }
.metric-delta { font-size:.8rem; margin-top:.15rem; }
.metric-note  { color:var(--muted); font-size:.75rem; margin-top:.25rem; }

.card { background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02));
        border:1px solid var(--border); border-radius:16px;
        padding:.8rem 1rem; box-shadow:0 6px 18px rgba(0,0,0,.25); }

.section-title { font-weight:700; font-size:1.05rem; color:var(--accent);
                  margin:1rem 0 .5rem; border-bottom:1px solid var(--border);
                  padding-bottom:.3rem; }

.pp-chip { display:inline-flex; align-items:center; background:rgba(245,183,49,.12);
           border:1px solid rgba(245,183,49,.3); color:var(--text);
           border-radius:8px; padding:.2rem .5rem; font-size:.82rem;
           margin:.15rem 0; }

footer-bar { display:block; text-align:center; color:var(--muted);
             font-size:.78rem; padding:1.5rem 0 .8rem;
             border-top:1px solid var(--border); margin-top:2rem; }

section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(255,255,255,.02),rgba(0,0,0,.15));
  border-right:1px solid var(--border);
}

.stDownloadButton button, .stButton>button {
  border-radius:12px; border:1px solid var(--border);
  background:linear-gradient(180deg,#1d2a52,#0f1734);
}

[data-testid="stDataFrame"] div   { color:var(--text) !important; }
[data-testid="stDataFrame"] thead th { background:rgba(255,255,255,.03) !important; }
"""

_TOPBAR_CSS = r"""
<style>
header[data-testid="stHeader"] { background:transparent !important; }
header[data-testid="stHeader"] > * {
  background:transparent !important; box-shadow:none !important;
}
</style>
"""


def inject_css() -> None:
    css_path = Path(ROOT) / "app" / "assets" / "theme.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else _THEME_CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(_TOPBAR_CSS, unsafe_allow_html=True)


def nice_fig(fig, title: str | None = None, xlab: str = "", ylab: str = ""):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    fig.update_layout(
        margin=dict(t=55, r=20, b=40, l=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_treasury_rate() -> float:
    """Live US 10Y Treasury yield from ^TNX, divided by 100. Falls back to 4.5%."""
    try:
        hist = yf.Ticker("^TNX").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


@st.cache_data(ttl=300, show_spinner="Fetching price history…")
def fetch_history(syms: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download 3 years of daily adjusted close, always ending today. Cached 5 min."""
    end   = dt.date.today() + dt.timedelta(days=1)           # yfinance end is exclusive
    start = dt.date.today() - dt.timedelta(days=365 * 3 + 5) # +5 for weekend buffer
    df = yf.download(
        list(syms),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).unique()
        price_col = "Close" if "Close" in lvl0 else lvl0[0]
        prices = df[price_col].copy()
    else:
        col = "Close" if "Close" in df.columns else df.columns[0]
        raw = df[col]
        prices = raw.to_frame(name=syms[0]) if isinstance(raw, pd.Series) else raw

    prices = prices.sort_index().dropna(how="all")
    rets   = price_to_returns(prices)
    logging.info(
        "fetch_history: %d ticker(s) | %s → %s",
        len(syms),
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices, rets


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_meta(symbol: str) -> dict:
    """Fetch display name, sector, and region from Yahoo Finance info dict."""
    meta = {"name": symbol, "sector": "Unknown", "region": "Unknown"}
    if symbol.upper().endswith("-USD"):
        meta["sector"] = "Cryptocurrency"
        meta["region"] = "Global"
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or symbol
        meta["name"] = str(name)
        sector = info.get("sector", "")
        if sector:
            meta["sector"] = sector
        country = info.get("country", "")
        if country:
            meta["region"] = country
    except Exception:
        pass
    return meta


@st.cache_data(ttl=300, show_spinner=False)
def validate_and_get_info(symbol: str) -> tuple[bool, dict]:
    """Validate a ticker and return metadata. Cached 5 min."""
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        if hist.empty:
            return False, {}
        meta = fetch_ticker_meta(symbol)
        return True, meta
    except Exception:
        return False, {}


@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_cached(_path: str, symbols: tuple[str, ...]):
    p = Path(_path)
    last      = latest_intraday_last(p, list(symbols))
    close_m   = intraday_close_matrix(p, list(symbols))
    return last, close_m


def intraday_file_path() -> Path:
    return INTRADIR / "latest_data_tickers.parquet"


def normalize_ex_bench(d: Dict[str, float], bench: str) -> pd.Series:
    d = {k: float(v) for k, v in d.items()}
    d.pop(bench, None)
    s = pd.Series(d, dtype=float)
    return s / s.sum() if s.sum() else s


def per_ticker_metrics(
    rt: pd.DataFrame,
    bench_s: pd.Series,
    rf_annual: float,
    bench_sym: str,
) -> pd.DataFrame:
    rows = []
    for col in [c for c in rt.columns if c != bench_sym]:
        s = rt[col].dropna()
        if s.empty:
            continue
        b, _ = beta_alpha(s, bench_s, rf_annual=rf_annual)
        var_h, cvar_h = hist_var_cvar(s, alpha=0.95)
        rows.append({
            "Ticker":  col,
            "AnnRet":  ann_return(s),
            "AnnVol":  ann_vol(s),
            "Sharpe":  sharpe(s, rf_annual=rf_annual),
            "Sortino": sortino(s, rf_annual=rf_annual),
            "Calmar":  calmar(s),
            "Beta":    b,
            "TE":      tracking_error(s, bench_s),
            "IR":      information_ratio(s, bench_s),
            "MaxDD":   max_drawdown(s).max_drawdown,
            "VaR95":   var_h,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Ticker").sort_index()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Pulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]

if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = list(_DEFAULT_TICKERS)
if "ticker_meta" not in st.session_state:
    st.session_state.ticker_meta = {}
if "last_add_status" not in st.session_state:
    st.session_state.last_add_status = None   # ("ok"|"err", msg)
if "metrics_validated" not in st.session_state:
    st.session_state.metrics_validated = False

# ══════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="pp-title">📈 Portfolio Pulse</div>
<div class="pp-sub">Clean visuals · Quick insights · Actionable risk</div>
<div class="pp-date">{dt.date.today():%B %d, %Y}</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
INTRA = intraday_file_path()

with st.sidebar:
    st.markdown("## 📈 Portfolio Pulse")
    st.caption(f"Today: {dt.date.today():%B %d, %Y}")

    # ── 📊 Portfolio ──────────────────────────────────────────────────────
    with st.expander("📊 Portfolio", expanded=True):
        # Catalog dropdown
        selected_catalog = st.selectbox(
            "Select from catalog",
            options=[_CATALOG_PLACEHOLDER] + TICKER_CATALOG,
            index=0,
            key="catalog_selectbox",
            help="80+ stocks, ETFs, and crypto to choose from.",
        )
        # Manual entry
        custom_input = st.text_input(
            "Or type any ticker",
            placeholder="e.g. TSLA, SOL-USD",
            key="custom_ticker_input",
            help="Enter any valid Yahoo Finance symbol.",
        )

        add_clicked = st.button(
            "＋ Add to Portfolio",
            use_container_width=True,
            type="primary",
            key="btn_add_ticker",
        )

        # Determine what to add (custom takes priority if filled)
        ticker_to_add: str | None = None
        if custom_input.strip():
            ticker_to_add = custom_input.strip().upper()
        elif selected_catalog != _CATALOG_PLACEHOLDER:
            ticker_to_add = extract_ticker(selected_catalog)

        if add_clicked:
            if not ticker_to_add:
                st.warning("Select a ticker from the catalog or type one above.")
            elif ticker_to_add in st.session_state.portfolio_tickers:
                st.warning(f"{ticker_to_add} is already in your portfolio.")
            else:
                with st.spinner(f"Validating {ticker_to_add}…"):
                    valid, meta = validate_and_get_info(ticker_to_add)
                if valid:
                    st.session_state.portfolio_tickers.append(ticker_to_add)
                    st.session_state.ticker_meta[ticker_to_add] = meta
                    name = meta.get("name", ticker_to_add)
                    st.session_state.last_add_status = ("ok", f"✅ {name} ({ticker_to_add}) added!")
                    st.rerun()
                else:
                    st.error(f"❌ '{ticker_to_add}' not found on Yahoo Finance.")

        # Show result from previous rerun
        if st.session_state.last_add_status:
            kind, msg = st.session_state.last_add_status
            if kind == "ok":
                st.success(msg)
            else:
                st.error(msg)
            st.session_state.last_add_status = None

        # Current portfolio chips
        if st.session_state.portfolio_tickers:
            st.caption("Current portfolio — ✕ to remove:")
            to_remove = None
            for sym in list(st.session_state.portfolio_tickers):
                meta = st.session_state.ticker_meta.get(sym, {})
                name = meta.get("name", "")
                label = f"**{sym}**" + (
                    f"  ·  {name[:18]}" if name and name != sym else ""
                )
                c1, c2 = st.columns([5, 1])
                c1.markdown(label)
                if c2.button("✕", key=f"rm_{sym}", help=f"Remove {sym}"):
                    to_remove = sym
            if to_remove:
                st.session_state.portfolio_tickers.remove(to_remove)
                st.session_state.ticker_meta.pop(to_remove, None)
                st.rerun()
        else:
            st.info("No tickers yet. Add one above.")
            st.stop()

    # ── ⚙️ Settings ───────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=True):
        ALL_SYMS = list(dict.fromkeys(st.session_state.portfolio_tickers + [DEFAULT_BENCH]))

        chosen = st.multiselect(
            "Assets to analyze",
            options=ALL_SYMS,
            default=ALL_SYMS,
            help="Choose which assets to include in the analysis.",
        )
        if not chosen:
            st.stop()

        bench = st.selectbox(
            "Benchmark",
            options=[DEFAULT_BENCH] + [s for s in chosen if not s.startswith("^")],
            index=0,
            help="Benchmark used for all relative-performance metrics.",
        )

        range_choice = st.radio(
            "Time Period",
            ["Today", "1W", "1M", "6M", "1Y", "3Y", "Custom"],
            index=5,
            horizontal=True,
            help="Select the analysis window.",
        )
        custom_dates = None
        if range_choice == "Custom":
            custom_dates = st.date_input("Pick start/end", value=[])

        st.divider()

        weight_mode = st.radio(
            "Portfolio Weights",
            [
                "Equal-weighted (excl. benchmark)",
                "Custom Weights",
                "Max Sharpe",
                "Min Variance",
            ],
            index=0,
        )
        sliders: Dict[str, float] = {}
        if weight_mode == "Custom Weights":
            st.caption("Set weights (renormalized to 100%):")
            for c in [s for s in chosen if s != bench]:
                sliders[c] = st.slider(c, 0.0, 1.0, 0.0, 0.01, key=f"w_{c}")

        st.divider()

        # ── Risk-Free Rate ─────────────────────────────────────────────
        st.markdown("**🏦 Risk-Free Rate**", help="Annual risk-free rate used in all ratio calculations.")
        live_rf = fetch_treasury_rate()
        st.caption(f"🏦 Current 10Y Treasury: **{live_rf * 100:.2f}%**")
        rf_slider = st.slider(
            "Annual RF rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(live_rf * 100),
            step=0.05,
            format="%.2f%%",
            help=(
                "The risk-free rate is the return of a risk-free investment (typically "
                "the 10-year US Treasury). It is subtracted from portfolio returns when "
                "computing risk-adjusted metrics like Sharpe and Sortino."
            ),
        )
        rf_num = st.number_input(
            "Or enter precise value (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(rf_slider),
            step=0.01,
            format="%.2f",
            key="rf_number_input",
        )
        rf = float(rf_num) / 100.0   # decimal form for all calculations

    # ── 🔄 Data ───────────────────────────────────────────────────────────
    with st.expander("🔄 Data", expanded=False):
        st.caption("Price data is auto-refreshed every 5 minutes from Yahoo Finance.")
        st.caption(f"Session started: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
        if st.button("🔄 Refresh Data Now", use_container_width=True, key="btn_refresh"):
            st.cache_data.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  LOAD & SLICE DATA
# ══════════════════════════════════════════════════════════════════════════
prices_all, rets_all = fetch_history(tuple(ALL_SYMS))

if prices_all.empty:
    missing = [s for s in ALL_SYMS if s not in (prices_all.columns if not prices_all.empty else [])]
    st.error(
        "No price data could be loaded. "
        "Check symbols and your network connection. "
        f"Symbols attempted: {', '.join(ALL_SYMS)}"
    )
    st.stop()

# Show last updated in sidebar Data section
with st.sidebar:
    with st.expander("🔄 Data", expanded=False):
        last_date = prices_all.index[-1].date()
        st.success(f"**Last updated:** {dt.datetime.now():%H:%M:%S}")
        st.caption(f"Most recent bar: **{last_date}**")

# Auto-populate metadata for any portfolio ticker missing it
for sym in list(st.session_state.portfolio_tickers):
    if sym not in st.session_state.ticker_meta:
        meta = fetch_ticker_meta(sym)
        st.session_state.ticker_meta[sym] = meta

VIEW_SYMS = list(dict.fromkeys(chosen + [bench]))
prices = prices_all.loc[:, [c for c in VIEW_SYMS if c in prices_all.columns]]
rets   = rets_all.loc[:,   [c for c in VIEW_SYMS if c in rets_all.columns]]

# Error if any chosen ticker returned no data
for sym in chosen:
    if sym not in prices.columns or prices[sym].dropna().empty:
        st.error(f"⚠️ No data returned for **{sym}**. It may be delisted or invalid.")

today_ts = prices.index.max()


def cutoff_ts(choice: str):
    if choice == "Today": return today_ts, True
    if choice == "1W":    return today_ts - pd.Timedelta(days=7), False
    if choice == "1M":    return today_ts - pd.Timedelta(days=30), False
    if choice == "6M":    return today_ts - pd.Timedelta(days=182), False
    if choice == "1Y":    return today_ts - pd.Timedelta(days=365), False
    if choice == "3Y":    return today_ts - pd.Timedelta(days=365 * 3), False
    if custom_dates and len(custom_dates) == 2:
        return pd.Timestamp(custom_dates[0]), False
    return prices.index.min(), False


cut, is_today = cutoff_ts(range_choice)
hist_px = prices.loc[prices.index >= cut]
hist_rt = rets.loc[rets.index >= cut]

# ── Build weights ──────────────────────────────────────────────────────────
if weight_mode == "Equal-weighted (excl. benchmark)":
    w = normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)
elif weight_mode == "Custom Weights":
    w = normalize_ex_bench(sliders, bench)
else:
    cols_opt = [c for c in hist_rt.columns if c in chosen and c != bench]
    R_opt    = hist_rt[cols_opt]
    if weight_mode == "Max Sharpe":
        wopt = optimize_max_sharpe(R_opt, rf_annual=rf) if len(cols_opt) >= 2 else pd.Series(np.nan, index=cols_opt)
    else:
        wopt = optimize_min_variance(R_opt) if len(cols_opt) >= 2 else pd.Series(np.nan, index=cols_opt)
    wopt = wopt.dropna()
    w = wopt if wopt.sum() else normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)

# ── Summary pills ─────────────────────────────────────────────────────────
asset_ct = len([s for s in chosen if s != bench])
st.markdown(
    f"""
    <div class='pp-pills'>
      <span class='pp-pill'>Assets: <b>{asset_ct}</b></span>
      <span class='pp-pill'>Benchmark: <b>{bench}</b></span>
      <span class='pp-pill'>Weights: <b>{weight_mode.split(" (")[0]}</b></span>
      <span class='pp-pill'>Window: <b>{'Today' if is_today else str(cut.date())} → {today_ts.date()}</b></span>
      <span class='pp-pill'>RF: <b>{rf*100:.2f}%</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Portfolio & benchmark return series ──────────────────────────────────
port    = portfolio_returns(hist_rt.drop(columns=[bench], errors="ignore"), w)
bench_s = hist_rt.get(bench)

# ── Startup metrics validation (once per session) ─────────────────────────
if not st.session_state.metrics_validated:
    st.session_state.metrics_validated = True
    try:
        val_prices, val_rets = fetch_history(("SPY", "QQQ"))
        if not val_rets.empty and "SPY" in val_rets.columns and "QQQ" in val_rets.columns:
            spy_r = val_rets["SPY"].dropna()
            qqq_r = val_rets["QQQ"].dropna()
            validate_all_metrics(spy_r, qqq_r, rf_annual=0.045, label="SPY_vs_QQQ")
    except Exception as exc:
        logging.warning("Startup validation skipped: %s", exc)

# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
tab_perf, tab_live, tab_vol, tab_risk, tab_div = st.tabs(
    ["📈 Performance", "⚡ Live", "📉 Volatility", "⚠️ Risk & Frontier", "🔀 Diversification"]
)

# ════════════════════════════  PERFORMANCE  ══════════════════════════════
with tab_perf:
    if is_today:
        # Show today's single-day return from the daily price history
        _prior_idx = (
            -2 if prices.index[-1].date() == dt.date.today() and len(prices) > 1
            else -1
        )
        _today_row = prices.iloc[-1]
        _prev_row  = prices.iloc[_prior_idx]
        _today_rets = (_today_row / _prev_row - 1.0).dropna()

        st.markdown('<div class="section-title">Today\'s Return (vs Prior Close)</div>', unsafe_allow_html=True)
        _cols = [s for s in VIEW_SYMS if s in _today_rets.index]
        if _cols:
            _n = min(len(_cols), 4)
            _tcols = st.columns(_n)
            for i, sym in enumerate(_cols[:_n]):
                v = float(_today_rets[sym])
                color = "var(--good)" if v >= 0 else "var(--bad)"
                _tcols[i].markdown(
                    f"""<div class="metric-card">
                      <div class="metric-title">{sym}</div>
                      <div class="metric-value" style="color:{color}">{v:+.2%}</div>
                      <div class="metric-note">vs prior close</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if len(_cols) > 4:
                _extra = st.columns(min(len(_cols) - 4, 4))
                for i, sym in enumerate(_cols[4:8]):
                    v = float(_today_rets[sym])
                    color = "var(--good)" if v >= 0 else "var(--bad)"
                    _extra[i].markdown(
                        f"""<div class="metric-card">
                          <div class="metric-title">{sym}</div>
                          <div class="metric-value" style="color:{color}">{v:+.2%}</div>
                          <div class="metric-note">vs prior close</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Portfolio today
        w_today = w.reindex([s for s in _cols if s != bench]).fillna(0.0)
        if w_today.sum():
            w_today = w_today / w_today.sum()
        _port_ret = float((_today_rets.reindex(w_today.index).fillna(0.0) * w_today).sum())
        _bench_ret = float(_today_rets.get(bench, np.nan)) if bench in _today_rets else np.nan

        st.divider()
        _pc1, _pc2, _pc3 = st.columns(3)
        _pc1.metric("Portfolio Today", f"{_port_ret:+.2%}",
                    help="Weighted daily return of your portfolio vs prior close.")
        if np.isfinite(_bench_ret):
            _pc2.metric(f"{bench} Today", f"{_bench_ret:+.2%}")
            _pc3.metric("Active Return Today", f"{(_port_ret - _bench_ret):+.2%}",
                        help="Portfolio return minus benchmark return today.")
        st.caption(
            f"Prices as of {_today_row.name.date()} · "
            "Switch to 1W / 1M / … for full historical metrics."
        )
    else:
        with st.spinner("Computing metrics…"):
            p_ret  = ann_return(port)
            p_vol  = ann_vol(port)
            p_sr   = sharpe(port,   rf_annual=rf)
            p_so   = sortino(port,  rf_annual=rf)
            p_beta, p_alpha = beta_alpha(port, bench_s, rf_annual=rf)
            p_tr   = treynor(port, bench_s, rf_annual=rf)
            p_te   = tracking_error(port, bench_s)
            p_ir   = information_ratio(port, bench_s)
            p_dd   = max_drawdown(port)
            p_ca   = calmar(port)
            p_var, p_cvar = hist_var_cvar(port, alpha=0.95)

        def _fmt(v, pct=False, digits=2):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:.{digits}%}" if pct else f"{v:.{digits}f}"

        st.markdown('<div class="section-title">Return & Risk-Adjusted</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Ann. Return",   _fmt(p_ret, pct=True),
                    help="Compounded annual return over the selected window.")
        r1c2.metric("Ann. Volatility", _fmt(p_vol, pct=True),
                    help="Annualised standard deviation of daily returns.")
        r1c3.metric("Sharpe Ratio",  _fmt(p_sr),
                    help="(Ann. Return − Rf) / Ann. Vol. Higher = better risk-adjusted return.")
        r1c4.metric("Sortino Ratio", _fmt(p_so),
                    help="(Ann. Return − Rf) / Downside Vol. Penalises only negative volatility.")

        st.markdown('<div class="section-title">Benchmark-Relative</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Beta",           _fmt(p_beta),
                    help="Sensitivity to benchmark. >1 amplifies moves; <1 is defensive.")
        r2c2.metric("Treynor Ratio",  _fmt(p_tr),
                    help="(Ann. Return − Rf) / Beta. Like Sharpe but adjusted for market risk.")
        r2c3.metric("Tracking Error", _fmt(p_te, pct=True),
                    help="Ann. std of (portfolio − benchmark) daily returns.")
        r2c4.metric("Info. Ratio",    _fmt(p_ir),
                    help="Active return / tracking error. >0.5 decent, >1 strong.")

        st.markdown('<div class="section-title">Drawdown & Tail Risk</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        r3c1.metric("Max Drawdown",   _fmt(p_dd.max_drawdown, pct=True),
                    help="Worst peak-to-trough loss in the selected period.")
        r3c2.metric("Calmar Ratio",   _fmt(p_ca),
                    help="Ann. Return / |Max Drawdown|. Higher = better return per unit of drawdown risk.")
        r3c3.metric("VaR (95%, 1D)",  _fmt(p_var, pct=True),
                    help="Historical 1-day loss exceeded only 5% of the time.")
        r3c4.metric("CVaR (95%, 1D)", _fmt(p_cvar, pct=True),
                    help="Average loss on the worst 5% of days (Expected Shortfall).")

        # NAV chart
        st.divider()
        if "nav_assets_toggle" not in st.session_state:
            st.session_state.nav_assets_toggle = False
        show_assets = st.session_state.nav_assets_toggle

        nav_series: Dict[str, pd.Series] = {"Portfolio": nav_from_returns(port)}
        if bench_s is not None and not bench_s.dropna().empty:
            nav_series[bench] = nav_from_returns(bench_s)
        if show_assets:
            for t in [s for s in chosen if s in hist_rt.columns and s != bench]:
                nav_series[t] = nav_from_returns(hist_rt.loc[nav_series["Portfolio"].index, t])

        nav_df = pd.DataFrame(nav_series).dropna(how="any")
        fig_nav = px.line(
            nav_df,
            labels={"value": "NAV (rebased to 1)", "variable": "Series"},
        )
        nice_fig(fig_nav, title="Cumulative Net Asset Value", xlab="Date", ylab="NAV")
        st.plotly_chart(fig_nav, use_container_width=True)
        st.toggle(
            "Show individual assets in NAV chart",
            key="nav_assets_toggle",
            help="Overlay each selected asset as its own line.",
        )

        if bench_s is not None and not bench_s.dropna().empty:
            rel = relative_return_series(port, bench_s)
            fig_rel = px.area(
                rel.rename(f"Relative vs {bench}"),
                labels={"value": "Outperformance", "index": "Date"},
                color_discrete_sequence=["#f5b731"],
            )
            nice_fig(fig_rel, title=f"Cumulative Outperformance vs {bench}", xlab="Date", ylab="Relative Return")
            st.plotly_chart(fig_rel, use_container_width=True)

# ════════════════════════════  LIVE  ══════════════════════════════════════
with tab_live:
    st.subheader("Live Intraday View")
    win_opt = st.select_slider(
        "Show last…",
        options=["30m", "1h", "2h", "3h", "6h", "Full day"],
        value="3h",
        help="Slice the intraday cache to the recent window.",
    )
    _mins = {"30m": 30, "1h": 60, "2h": 120, "3h": 180, "6h": 360}

    _intra_file_usable = False   # set to True only when file exists and is fresh today

    if (INTRA := intraday_file_path()).exists():
        # Check staleness before loading — if the file isn't from today, treat it
        # as missing rather than silently showing old data as "live".
        mtime_ny = pd.Timestamp(INTRA.stat().st_mtime, unit="s", tz="UTC").tz_convert("America/New_York")
        _cache_date = mtime_ny.date()
        _cache_is_stale = _cache_date < dt.date.today()

        if _cache_is_stale:
            st.warning(
                f"⚠️ Intraday cache is from **{_cache_date}** — it is stale and has been ignored. "
                "Start the daemon (`python -m src.live daemon`) during market hours to get live data. "
                "Showing today's return from daily price data instead."
            )
            # Fall through to the daily-data fallback below by pretending file doesn't exist
            _intra_file_usable = False
        else:
            _intra_file_usable = True

        if _intra_file_usable:
            last, intra_close = load_intraday_cached(str(INTRA), tuple(ALL_SYMS))
            if not intra_close.empty:
                if win_opt != "Full day":
                    start_ts = intra_close.index.max() - pd.Timedelta(minutes=_mins[win_opt])
                    intra_close = intra_close.loc[intra_close.index >= start_ts]

                # Prior close reference: if today's daily bar is already in the
                # history we must step back one row to get yesterday's close.
                _prior_idx = (
                    -2 if prices_all.index[-1].date() == dt.date.today() and len(prices_all) > 1
                    else -1
                )
                prev_map = {
                    c: float(prices_all.iloc[_prior_idx][c])
                    for c in prices_all.columns
                }
                cols = [s for s in VIEW_SYMS if s in intra_close.columns]
                if cols:
                    rets_min = intraday_return_matrix(intra_close.loc[:, cols], prev_map)
                    if not rets_min.empty:
                        w_cols = w.reindex([c for c in cols if c != bench]).fillna(0.0)
                        if w_cols.sum():
                            w_cols = w_cols / w_cols.sum()
                        port_today  = rets_min[w_cols.index].dot(w_cols).rename("Portfolio (today)")
                        bench_today = rets_min.get(bench)
                        chart_df    = pd.concat([port_today, bench_today], axis=1).dropna(how="all")
                        npts        = len(chart_df)

                        fig_live = px.line(
                            chart_df,
                            labels={"value": "Intraday Return", "variable": "Series"},
                        )
                        nice_fig(fig_live, title=f"Intraday Return vs Prior Close — {npts:,} pts",
                                 xlab="Time", ylab="Return")
                        st.plotly_chart(fig_live, use_container_width=True)

                        if npts:
                            p_now = float(port_today.dropna().iloc[-1]) if len(port_today.dropna()) else float("nan")
                            b_now = (float(bench_today.dropna().iloc[-1])
                                     if bench_today is not None and len(bench_today.dropna()) else float("nan"))
                            t1, t2, t3 = st.columns(3)
                            for label, val, col in [
                                ("Today (Portfolio)", p_now, t1),
                                (f"Today ({bench})", b_now, t2),
                                ("Today Relative", p_now - b_now if np.isfinite(b_now) else np.nan, t3),
                            ]:
                                col.markdown(
                                    f"""<div class="metric-card">
                                      <div class="metric-title">{label}</div>
                                      <div class="metric-value">{val:.2%}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                        show_ov = st.toggle("Show 7-day overlay (Adj Close)", value=False)
                        if show_ov and not last.empty:
                            ov     = overlay_prices(prices_all, last, prev_map)
                            fig_ov = px.line(ov.loc[ov.index >= ov.index.max() - pd.Timedelta(days=7)],
                                             labels={"value": "Price", "variable": "Ticker"})
                            nice_fig(fig_ov, title="Overlay Adj Close — last 7 days", xlab="Date", ylab="Price")
                            st.plotly_chart(fig_ov, use_container_width=True)

                st.caption(f"Cache updated {mtime_ny:%Y-%m-%d %H:%M ET} · rows: {len(intra_close):,}")
            else:
                st.info("Intraday cache exists but is empty for selected symbols.")
        else:
            _intra_file_usable = False  # stale — warned above; fall through to daily fallback

    # ── Daily-data fallback (no file, or stale file) ──────────────────────
    _intra_missing = not intraday_file_path().exists()
    _show_daily_fallback = _intra_missing or not _intra_file_usable
    if _intra_missing:
        st.info(
            "No intraday cache found. Run `python -m src.live daemon` during market hours "
            "to enable live 1-minute updates."
        )
    if _show_daily_fallback:
        # Fallback: show today's daily return from the price history
        if len(prices_all) >= 2:
            _d_prior_idx = (
                -2 if prices_all.index[-1].date() == dt.date.today() and len(prices_all) > 1
                else -1
            )
            _d_prev  = prices_all.iloc[_d_prior_idx]
            _d_today = prices_all.iloc[-1]
            _d_rets  = (_d_today / _d_prev - 1.0).dropna()
            _d_syms  = [s for s in VIEW_SYMS if s in _d_rets.index]
            if _d_syms:
                st.markdown('<div class="section-title">Today\'s Return (Daily Data)</div>', unsafe_allow_html=True)
                _d_port_w = w.reindex([s for s in _d_syms if s != bench]).fillna(0.0)
                if _d_port_w.sum():
                    _d_port_w = _d_port_w / _d_port_w.sum()
                _d_port_ret  = float((_d_rets.reindex(_d_port_w.index).fillna(0.0) * _d_port_w).sum())
                _d_bench_ret = float(_d_rets.get(bench, np.nan)) if bench in _d_rets else np.nan

                _d1, _d2, _d3 = st.columns(3)
                for _lbl, _val, _col in [
                    ("Today (Portfolio)", _d_port_ret, _d1),
                    (f"Today ({bench})",  _d_bench_ret, _d2),
                    ("Today Relative",
                     _d_port_ret - _d_bench_ret if np.isfinite(_d_bench_ret) else np.nan,
                     _d3),
                ]:
                    _color = (
                        "var(--good)" if np.isfinite(_val) and _val >= 0
                        else "var(--bad)" if np.isfinite(_val)
                        else "var(--muted)"
                    )
                    _col.markdown(
                        f"""<div class="metric-card">
                          <div class="metric-title">{_lbl}</div>
                          <div class="metric-value" style="color:{_color}">
                            {"N/A" if not np.isfinite(_val) else f"{_val:+.2%}"}
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                # Bar chart of per-asset daily returns
                _bar_df = _d_rets.reindex(_d_syms).dropna().sort_values()
                fig_daily = px.bar(
                    _bar_df,
                    labels={"index": "Ticker", "value": "Return"},
                    color=_bar_df.values,
                    color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                    color_continuous_midpoint=0.0,
                )
                fig_daily.update_coloraxes(showscale=False)
                fig_daily.update_yaxes(tickformat=".1%")
                nice_fig(fig_daily, title=f"Daily Returns — {_d_today.name.date()}", xlab="Ticker", ylab="Return")
                st.plotly_chart(fig_daily, use_container_width=True)
                st.caption(f"Based on {prices_all.index[-2].date()} → {prices_all.index[-1].date()} daily close prices.")

# ════════════════════════════  VOLATILITY  ═══════════════════════════════
with tab_vol:
    st.subheader("Rolling Volatility & Beta")
    if is_today:
        st.info("Rolling volatility and beta require multiple days of returns. Switch to 1W / 1M / … to see rolling stats.")
    else:
        df_rb = pd.concat([port.rename("p"), bench_s.rename("b")], axis=1).dropna()
        if df_rb.empty:
            st.info("No overlapping data for portfolio & benchmark.")
        else:
            max_win = int(min(252, len(df_rb)))
            if max_win < 10:
                st.info("Not enough data — choose a longer window.")
            else:
                win = st.slider(
                    "Rolling window (trading days)",
                    min_value=10, max_value=max_win,
                    value=min(60, max_win), step=5,
                    help="Window length for rolling metrics.",
                )
                if len(df_rb) >= win:
                    roll_vol   = df_rb["p"].rolling(win).std(ddof=1) * np.sqrt(TRADING_DAYS)
                    cov_pb     = df_rb["p"].rolling(win).cov(df_rb["b"])
                    var_b      = df_rb["b"].rolling(win).var()
                    beta_roll  = (cov_pb / var_b).replace([np.inf, -np.inf], np.nan)

                    last_idx       = df_rb.index[-win:]
                    roll_vol_view  = roll_vol.loc[last_idx].dropna()
                    beta_roll_view = beta_roll.loc[last_idx].dropna()

                    c1, c2 = st.columns(2)
                    with c1:
                        fv = px.line(
                            roll_vol_view.rename(f"Rolling {win}D Ann. Vol"),
                            labels={"value": "Ann. Volatility", "index": "Date"},
                        )
                        fv.update_yaxes(tickformat=".1%")
                        nice_fig(fv, title=f"Rolling {win}D Annualised Volatility",
                                 xlab="Date", ylab="Ann. Volatility")
                        st.plotly_chart(fv, use_container_width=True)
                    with c2:
                        fb = px.line(
                            beta_roll_view.rename(f"Rolling {win}D Beta"),
                            labels={"value": "Beta", "index": "Date"},
                        )
                        nice_fig(fb, title=f"Rolling {win}D Beta vs {bench}",
                                 xlab="Date", ylab="Beta")
                        st.plotly_chart(fb, use_container_width=True)
                else:
                    st.info(f"Not enough rows for a {win}-day window.")

# ════════════════════════  RISK & FRONTIER  ══════════════════════════════
with tab_risk:
    st.subheader("Risk Contributions & Efficient Frontier")
    if is_today:
        st.info("Switch to a longer window to analyze risk.")
    else:
        cols_sel = [c for c in hist_rt.columns if c in chosen and c != bench]
        R = hist_rt[cols_sel].dropna(how="all")
        if R.empty or len(cols_sel) < 1:
            st.info("Select at least one non-benchmark asset.")
            st.stop()

        shrink = st.slider(
            "Covariance shrinkage λ",
            0.0, 0.9, 0.0, 0.05,
            help="λ=0 → sample covariance; λ>0 → shrink toward diagonal (improves stability).",
        )

        left, right = st.columns([2, 1])
        with left:
            if len(cols_sel) >= 2:
                Sigma = cov_matrix(R, shrink=shrink)
                w_now = w.reindex(cols_sel).fillna(0.0)
                if w_now.sum():
                    w_now = w_now / w_now.sum()
                rc = risk_contributions(w_now, Sigma)
                fig_rc = px.bar(
                    rc.sort_values(ascending=False).rename("Contribution"),
                    labels={"index": "Ticker", "value": "Risk Contribution"},
                    color_discrete_sequence=["#f5b731"],
                )
                fig_rc.update_yaxes(tickformat=".0%")
                nice_fig(fig_rc, title="Risk Contribution by Asset",
                         xlab="Ticker", ylab="Contribution")
                st.plotly_chart(fig_rc, use_container_width=True)
            else:
                st.info("Select ≥ 2 assets for risk contributions.")

        with right:
            pv, bv = ann_vol(port), ann_vol(bench_s)
            vol_s = pd.Series({"Portfolio": pv, bench: bv}, name="Ann. Vol")
            fig_vb = px.bar(
                vol_s,
                labels={"index": "Series", "value": "Ann. Volatility"},
                color_discrete_sequence=["#4cc9f0", "#f5b731"],
            )
            fig_vb.update_yaxes(tickformat=".0%")
            nice_fig(fig_vb, title="Total Annualised Volatility",
                     xlab="", ylab="Ann. Volatility")
            st.plotly_chart(fig_vb, use_container_width=True)

        st.divider()
        st.subheader("Efficient Frontier & VaR")

        ctl1, ctl2, ctl3, ctl4 = st.columns([1.2, 1, 1, 1.2])
        with ctl1:
            n_pts = st.slider("Frontier points", 10, 60, 30, 5)
        with ctl2:
            bmin, bmax = st.slider("Weight bounds", 0.0, 1.0, (0.0, 1.0), 0.05)
        with ctl3:
            var_alpha = st.select_slider(
                "VaR confidence",
                options=[0.90, 0.95, 0.975, 0.99],
                value=0.95,
            )
        with ctl4:
            horizon = st.slider("VaR horizon (days)", 1, 10, 1)

        cA, cB = st.columns([3, 1])
        with cA:
            try:
                if len(cols_sel) >= 2:
                    ef = trace_efficient_frontier(
                        R, n_points=n_pts, bounds=(bmin, bmax), shrink=shrink
                    )
                else:
                    ef = pd.DataFrame()

                if ef.empty:
                    st.caption("Efficient frontier requires ≥ 2 assets and scipy.")
                else:
                    ret_p, vol_p = ann_return(port), ann_vol(port)
                    ew = pd.Series(1.0 / len(cols_sel), index=cols_sel)
                    ret_ew = ann_return(R.dot(ew))
                    vol_ew = ann_vol(R.dot(ew))

                    try:
                        w_ms = optimize_max_sharpe(R, rf_annual=rf, bounds=(bmin, bmax))
                        w_mv = optimize_min_variance(R, bounds=(bmin, bmax))
                    except Exception:
                        w_ms = pd.Series(dtype=float)
                        w_mv = pd.Series(dtype=float)

                    pts = pd.DataFrame({
                        "name":    ["Portfolio", "Equal-Weight"],
                        "vol_ann": [vol_p, vol_ew],
                        "ret_ann": [ret_p, ret_ew],
                    })
                    if not w_ms.dropna().empty:
                        pts = pd.concat([pts, pd.DataFrame({
                            "name":    ["Max Sharpe"],
                            "vol_ann": [ann_vol(R.dot(w_ms))],
                            "ret_ann": [ann_return(R.dot(w_ms))],
                        })], ignore_index=True)
                    if not w_mv.dropna().empty:
                        pts = pd.concat([pts, pd.DataFrame({
                            "name":    ["Min Variance"],
                            "vol_ann": [ann_vol(R.dot(w_mv))],
                            "ret_ann": [ann_return(R.dot(w_mv))],
                        })], ignore_index=True)

                    fig_ef = px.scatter(
                        ef, x="vol_ann", y="ret_ann",
                        labels={"vol_ann": "Volatility (Ann.)", "ret_ann": "Return (Ann.)"},
                    )
                    fig_ef.update_traces(mode="lines+markers")
                    fig_ef.add_scatter(
                        x=pts["vol_ann"], y=pts["ret_ann"],
                        mode="markers+text", text=pts["name"],
                        textposition="top center",
                        marker=dict(size=10, color="#f5b731"),
                        name="Reference Portfolios",
                    )
                    nice_fig(fig_ef, title="Efficient Frontier (Annualised)",
                             xlab="Volatility", ylab="Expected Return")
                    st.plotly_chart(fig_ef, use_container_width=True)
            except Exception as exc:
                st.caption(f"Frontier unavailable: {exc}")

        with cB:
            if not R.empty:
                # Historical VaR on portfolio
                h_var, h_cvar = hist_var_cvar(port, alpha=float(var_alpha))
                # Monte Carlo VaR
                mc_var, mc_es = mc_var_es(
                    R, w.reindex(R.columns).fillna(0.0),
                    alpha=float(var_alpha), horizon_days=int(horizon),
                    n_sims=10_000, seed=42,
                )
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-title">Historical VaR ({int(var_alpha*100)}%)</div>
                  <div class="metric-value">{h_var:.2%}</div>
                  <div class="metric-note">CVaR: {h_cvar:.2%}</div>
                  <div class="metric-title" style="margin-top:.5rem">MC VaR (h={int(horizon)}d)</div>
                  <div class="metric-value">{mc_var:.2%}</div>
                  <div class="metric-note">ES: {mc_es:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Not enough data for VaR.")

# ════════════════════════  DIVERSIFICATION  ══════════════════════════════
with tab_div:
    st.subheader("Diversification Analysis")

    if is_today:
        st.info("Switch to a longer window to see diversification metrics.")
    else:
        asset_cols = [c for c in chosen if c != bench and c in hist_rt.columns]
        w_asset    = w.reindex(asset_cols).fillna(0.0)
        if w_asset.sum():
            w_asset = w_asset / w_asset.sum()

        # ── Diversification Ratio ──────────────────────────────────────
        if len(asset_cols) >= 2:
            vols_ind = hist_rt[asset_cols].std(ddof=1) * np.sqrt(TRADING_DAYS)
            weighted_avg_vol = float((vols_ind * w_asset).sum())
            port_vol_now     = ann_vol(port)
            if port_vol_now > 0 and weighted_avg_vol > 0:
                div_ratio   = weighted_avg_vol / port_vol_now
                div_benefit = (1.0 - port_vol_now / weighted_avg_vol) * 100.0
            else:
                div_ratio   = float("nan")
                div_benefit = float("nan")

            d1, d2, d3 = st.columns(3)
            d1.metric(
                "Diversification Ratio",
                f"{div_ratio:.2f}" if np.isfinite(div_ratio) else "N/A",
                help="Weighted avg. individual vol / portfolio vol. Higher = more diversification benefit.",
            )
            d2.metric(
                "Diversification Benefit",
                f"{div_benefit:.1f}%" if np.isfinite(div_benefit) else "N/A",
                help="% reduction in vol vs simply averaging individual risks.",
            )
            d3.metric(
                "Portfolio Ann. Vol",
                f"{port_vol_now:.2%}" if not np.isnan(port_vol_now) else "N/A",
            )

            if np.isfinite(div_benefit):
                if div_benefit >= 20:
                    st.success(f"✅ Strong diversification — correlation effects reduce portfolio vol by **{div_benefit:.1f}%**.")
                elif div_benefit >= 5:
                    st.info(f"ℹ️ Moderate diversification — **{div_benefit:.1f}%** vol reduction vs simple weighting.")
                else:
                    st.warning(f"⚠️ Low diversification — assets are highly correlated; only **{div_benefit:.1f}%** vol benefit.")

        # ── Correlation heatmap ────────────────────────────────────────
        if len(asset_cols) >= 2:
            st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
            corr_df = hist_rt[asset_cols].corr().round(2)
            fig_corr = px.imshow(
                corr_df,
                color_continuous_scale="RdYlGn_r",
                zmin=-1, zmax=1,
                text_auto=True,
                labels=dict(color="Correlation"),
            )
            fig_corr.update_layout(
                title=dict(text="Pairwise Correlation (–1 low / +1 high)", font=dict(size=14)),
                margin=dict(t=55, r=20, b=20, l=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(title="Corr"),
            )
            fig_corr.update_xaxes(side="bottom")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("🟢 Green = low correlation (good diversification) · 🔴 Red = high correlation (less diversification)")
        else:
            st.info("Add ≥ 2 non-benchmark assets to see the correlation heatmap.")

        # ── Sector / Region pies ──────────────────────────────────────
        ticker_meta = st.session_state.get("ticker_meta", {})
        w_s = pd.Series(w, dtype=float).dropna()
        w_s = w_s[w_s.index != bench]

        if not w_s.empty:
            meta_df = pd.DataFrame(index=w_s.index)
            meta_df["sector"] = [
                ticker_meta.get(t, {}).get("sector", "Unknown") or "Unknown"
                for t in w_s.index
            ]
            meta_df["region"] = [
                ticker_meta.get(t, {}).get("region", "Unknown") or "Unknown"
                for t in w_s.index
            ]

            sec_w = w_s.groupby(meta_df["sector"]).sum().sort_values(ascending=False)
            reg_w = w_s.groupby(meta_df["region"]).sum().sort_values(ascending=False)

            all_unknown = all(v == "Unknown" for v in meta_df["sector"].values)
            if all_unknown:
                st.caption(
                    "ℹ️ Sector/region data is loading from Yahoo Finance — "
                    "try refreshing in a moment."
                )

            c1, c2 = st.columns(2)
            fig_sec = px.pie(
                values=sec_w.values, names=sec_w.index, hole=0.5,
                color_discrete_sequence=px.colors.sequential.Plasma_r,
            )
            fig_sec.update_traces(
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:.2%}<extra></extra>",
            )
            fig_sec.update_layout(
                title="Allocation by Sector",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=55, r=20, b=20, l=20),
            )
            c1.plotly_chart(fig_sec, use_container_width=True)

            fig_reg = px.pie(
                values=reg_w.values, names=reg_w.index, hole=0.5,
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            fig_reg.update_traces(
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:.2%}<extra></extra>",
            )
            fig_reg.update_layout(
                title="Allocation by Region",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=55, r=20, b=20, l=20),
            )
            c2.plotly_chart(fig_reg, use_container_width=True)

    # ── Per-asset metrics table ────────────────────────────────────────
    st.markdown('<div class="section-title">Per-Asset Metrics</div>', unsafe_allow_html=True)

    by_asset = per_ticker_metrics(hist_rt, bench_s, rf, bench)
    if not by_asset.empty:
        order    = [c for c in chosen if c in by_asset.index]
        by_asset = by_asset.reindex(order).dropna(how="all")
        disp = by_asset.copy()
        for c in ["AnnRet", "AnnVol", "TE", "MaxDD", "VaR95"]:
            if c in disp.columns:
                disp[c] = (disp[c] * 100).map("{:.2f}%".format)
        for c in ["Sharpe", "Sortino", "Beta", "IR", "Calmar"]:
            if c in disp.columns:
                disp[c] = disp[c].map("{:.2f}".format)
        st.dataframe(disp, use_container_width=True)
    else:
        st.info("Not enough data to compute per-asset metrics.")

    # ── Downloads ─────────────────────────────────────────────────────
    st.divider()
    nav_df_dl = pd.DataFrame({
        "Portfolio": nav_from_returns(port),
        bench: nav_from_returns(bench_s) if bench_s is not None else pd.Series(dtype=float),
    }).dropna()
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    col_dl1.download_button(
        "📥 Portfolio NAV (CSV)",
        nav_df_dl.to_csv(index=True).encode("utf-8"),
        file_name="portfolio_nav.csv", mime="text/csv",
    )
    if not by_asset.empty:
        col_dl2.download_button(
            "📥 Per-Asset Metrics (CSV)",
            by_asset.to_csv(index=True).encode("utf-8"),
            file_name="by_asset_metrics.csv", mime="text/csv",
        )
    col_dl3.download_button(
        "📥 Weights (CSV)",
        w.rename("weight").to_csv(index=True).encode("utf-8"),
        file_name="weights.csv", mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <footer-bar>
      Data sourced from <b>Yahoo Finance</b> &nbsp;·&nbsp;
      Auto-refreshed every 5 minutes &nbsp;·&nbsp;
      Last loaded: <b>{dt.datetime.now():%H:%M:%S}</b> &nbsp;·&nbsp;
      Portfolio Pulse
    </footer-bar>
    """,
    unsafe_allow_html=True,
)
